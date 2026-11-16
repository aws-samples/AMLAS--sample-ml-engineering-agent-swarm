"""Closed source-only campaign/CAS/wake contracts; SDK-shaped neutral fakes."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import threading

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters
from test_campaign_admission import ACCOUNT, DEPLOYMENT, REGION, fixture_guard

TABLE = "fixture-campaign-jobs"
TRANSPORT = "12345678-1234-4234-9234-00000000bb11"
PROMPT = "Fine-tune one Qwen coding model; preserve this literal prompt."
MODELS = {name: Session().get_service_model(name) for name in ("dynamodb", "sagemaker")}


def coordinator_module():
    try:
        return importlib.import_module("amlas.control_plane.campaign_coordinator")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.campaign_coordinator":
            pytest.fail("missing-symbol: campaign_coordinator.DynamoCampaignCoordinator", pytrace=False)
        raise


class DDB:
    """Uses installed Put/Get/Update models and one server lock for exact-byte CAS."""

    def __init__(self):
        self.items = {}
        self.calls = []
        self.lock = threading.Lock()
        self.lose_put = False
        self.lose_update = False
        self.fail_reads = False

    def _shape(self, operation, kwargs):
        validate_parameters(kwargs, MODELS["dynamodb"].operation_model(operation).input_shape)
        assert kwargs["TableName"] == TABLE
        self.calls.append((operation, copy.deepcopy(kwargs)))

    def get_item(self, **kwargs):
        self._shape("GetItem", kwargs)
        assert kwargs["ConsistentRead"] is True
        if self.fail_reads:
            raise TimeoutError("neutral read unavailable")
        with self.lock:
            row = self.items.get(kwargs["Key"]["job_name"]["S"])
            return {"Item": copy.deepcopy(row)} if row else {}

    def put_item(self, **kwargs):
        self._shape("PutItem", kwargs)
        assert kwargs["ConditionExpression"] == "attribute_not_exists(job_name)"
        key = kwargs["Item"]["job_name"]["S"]
        with self.lock:
            if key in self.items:
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
            self.items[key] = copy.deepcopy(kwargs["Item"])
        if self.lose_put:
            self.lose_put = False
            raise TimeoutError("neutral accepted response unavailable")
        return {}

    def update_item(self, **kwargs):
        self._shape("UpdateItem", kwargs)
        assert kwargs["ConditionExpression"] == (
            "coordinator_revision = :revision AND lease_epoch = :epoch AND private_record = :prior_body"
        )
        assert kwargs["UpdateExpression"] == (
            "SET coordinator_revision = :next, lease_epoch = :next_epoch, private_record = :body"
        )
        values = kwargs["ExpressionAttributeValues"]
        with self.lock:
            row = self.items.get(kwargs["Key"]["job_name"]["S"])
            if not row or any(
                row[field] != values[value]
                for field, value in (
                    ("coordinator_revision", ":revision"),
                    ("lease_epoch", ":epoch"),
                    ("private_record", ":prior_body"),
                )
            ):
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
            row.update(
                {
                    "coordinator_revision": copy.deepcopy(values[":next"]),
                    "lease_epoch": copy.deepcopy(values[":next_epoch"]),
                    "private_record": copy.deepcopy(values[":body"]),
                }
            )
        if self.lose_update:
            self.lose_update = False
            raise TimeoutError("neutral accepted update response unavailable")
        return {}

    def delete_item(self, **kwargs):
        pytest.fail("coordinator must not delete original rows")


class NativeJobs:
    def __init__(self, cc, state="Completed"):
        self.cc = cc
        self.state = state
        self.calls = []
        self.changed_arn = None

    def observe_original(self, original):
        self.calls.append(original)
        return self.cc.NativeCampaignObservation(
            kind=original.kind,
            job_name=original.job_name,
            job_arn=self.changed_arn or original.job_arn,
            request_sha256=original.request_sha256,
            source_binding_sha256=original.source_binding_sha256,
            status=self.state,
        )

    def create_training_job(self, **kwargs):
        pytest.fail("wake never creates a training job")

    def create_processing_job(self, **kwargs):
        pytest.fail("wake never creates a processing job")


def fixture_coordinator():
    cc = coordinator_module()
    ca, guard, reader, clock, approved = fixture_guard()
    ddb = DDB()
    coordinator = cc.DynamoCampaignCoordinator(guard, ddb, table_name=TABLE, clock=clock)
    return cc, ca, coordinator, guard, reader, clock, ddb, approved


def open_and_claim():
    cc, ca, coordinator, guard, reader, clock, ddb, approved = fixture_coordinator()
    witness = guard.verify()
    record = coordinator.open_campaign(witness, TRANSPORT, PROMPT)
    lease = coordinator.claim_lease(witness, record.campaign_ref, expected_revision=record.revision)
    return cc, ca, coordinator, guard, reader, clock, ddb, approved, witness, record, lease


def original(cc, kind="Training"):
    name = "fixture-original-training" if kind == "Training" else "fixture-original-processing"
    resource = "training-job" if kind == "Training" else "processing-job"
    return cc.OriginalCampaignOperation(
        operation_digest=hashlib.sha256((kind + "-original").encode()).hexdigest(),
        phase="sft" if kind == "Training" else "preparation",
        kind=kind,
        account_id=ACCOUNT,
        region=REGION,
        job_name=name,
        job_arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:{resource}/{name}",
        request_sha256=hashlib.sha256(b"neutral fixed original request").hexdigest(),
        source_binding_sha256=hashlib.sha256(b"neutral original private sources").hexdigest(),
    )


def wake(cc, record, operation):
    return cc.WakeHint(
        schema_version=1,
        event_id="fixture-event-1",
        campaign_ref=record.campaign_ref,
        account_id=ACCOUNT,
        region=REGION,
        deployment_id=DEPLOYMENT,
        kind=operation.kind,
        operation_digest=operation.operation_digest,
        job_arn=operation.job_arn,
    )


def test_literal_prompt_mapping_replays_original_ids_without_another_put():
    _, _, coordinator, guard, _, _, ddb, _ = fixture_coordinator()
    witness = guard.verify()
    first = coordinator.open_campaign(witness, TRANSPORT, PROMPT)
    again = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    assert first.campaign_ref == again.campaign_ref
    assert first.campaign_id == again.campaign_id
    assert first.private_store_correlation == again.private_store_correlation
    assert first.literal_prompt == PROMPT
    assert first.prompt_sha256 == hashlib.sha256(PROMPT.encode()).hexdigest()
    assert sum(name == "PutItem" for name, _ in ddb.calls) == 1


def test_alias_rebinding_cannot_overwrite_original_literal_prompt():
    cc, _, coordinator, guard, _, _, ddb, _ = fixture_coordinator()
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    prior = copy.deepcopy(ddb.items)
    with pytest.raises(cc.CampaignCoordinatorError):
        coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT + " changed")
    assert ddb.items == prior
    assert coordinator.load_campaign(guard.verify(), record.campaign_ref).literal_prompt == PROMPT


def test_admission_revocation_stops_before_any_ddb_lookup():
    import dataclasses

    _, ca, coordinator, guard, reader, _, ddb, _ = fixture_coordinator()
    witness = guard.verify()
    reader.observation = dataclasses.replace(reader.observation, endpoint_live_version="2")
    with pytest.raises(ca.AdmissionError):
        coordinator.open_campaign(witness, TRANSPORT, PROMPT)
    assert ddb.calls == []
    assert not hasattr(coordinator, "approval_policy")


def test_lost_open_ack_requires_consistent_exact_original_bytes():
    _, _, coordinator, guard, _, _, ddb, _ = fixture_coordinator()
    ddb.lose_put = True
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    assert record.literal_prompt == PROMPT
    assert sum(name == "PutItem" for name, _ in ddb.calls) == 1
    assert ddb.calls[-1][0] == "GetItem"
    assert ddb.calls[-1][1]["ConsistentRead"] is True


def test_unresolved_write_never_automatically_retries_put():
    cc, _, coordinator, guard, _, _, ddb, _ = fixture_coordinator()
    original_put = ddb.put_item

    def put_and_lose_read(**kwargs):
        ddb.lose_put = True
        try:
            return original_put(**kwargs)
        finally:
            ddb.fail_reads = True

    ddb.put_item = put_and_lose_read
    with pytest.raises(cc.CampaignUnresolved):
        coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    assert sum(name == "PutItem" for name, _ in ddb.calls) == 1
    ddb.fail_reads = False
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    assert record.literal_prompt == PROMPT
    assert sum(name == "PutItem" for name, _ in ddb.calls) == 1


def test_expired_or_old_lease_epoch_cannot_commit_original_operations():
    cc, _, coordinator, guard, _, clock, ddb, _, _, record, lease = open_and_claim()
    clock.value += 61
    fresh = guard.verify()
    current = coordinator.load_campaign(fresh, record.campaign_ref)
    newer = coordinator.claim_lease(fresh, record.campaign_ref, expected_revision=current.revision)
    before = copy.deepcopy(ddb.items)
    with pytest.raises(cc.CampaignCoordinatorError):
        coordinator.register_original(fresh, lease, original(cc))
    assert ddb.items == before
    assert newer.epoch > lease.epoch


@pytest.mark.parametrize("kind", ["Training", "Processing"])
def test_wake_observes_registered_original_without_dispatch(kind):
    cc, _, coordinator, guard, _, _, ddb, _, witness, record, lease = open_and_claim()
    operation = original(cc, kind)
    registered = coordinator.register_original(witness, lease, operation)
    lease = coordinator.claim_lease(
        guard.verify(), record.campaign_ref, expected_revision=registered.revision
    )
    observer = NativeJobs(cc)
    result = coordinator.observe_wake(guard.verify(), lease, wake(cc, record, operation), observer)
    assert observer.calls == [operation]
    assert result.status == "Completed"
    assert result.native_admission is False
    assert result.quality_success is False
    assert sum(name == "PutItem" for name, _ in ddb.calls) == 1
    assert result.operation_digest == operation.operation_digest


def test_unregistered_or_foreign_job_hint_never_reaches_observer():
    cc, _, coordinator, _, _, _, ddb, _, witness, record, lease = open_and_claim()
    observer = NativeJobs(cc)
    before = copy.deepcopy(ddb.items)
    with pytest.raises(cc.CampaignCoordinatorError):
        coordinator.observe_wake(witness, lease, wake(cc, record, original(cc)), observer)
    assert observer.calls == []
    assert ddb.items == before


def test_native_identity_mismatch_is_not_committed_as_completion():
    cc, _, coordinator, guard, _, _, ddb, _, witness, record, lease = open_and_claim()
    operation = original(cc)
    registered = coordinator.register_original(witness, lease, operation)
    lease = coordinator.claim_lease(
        guard.verify(), record.campaign_ref, expected_revision=registered.revision
    )
    observer = NativeJobs(cc)
    observer.changed_arn = operation.job_arn + "-different"
    before = copy.deepcopy(ddb.items)
    with pytest.raises(cc.CampaignCoordinatorError):
        coordinator.observe_wake(guard.verify(), lease, wake(cc, record, operation), observer)
    assert ddb.items == before


def test_closed_wake_payload_cannot_supply_status_policy_or_stage_advance():
    cc = coordinator_module()
    payload = {
        "schema_version": 1,
        "event_id": "fixture-event-1",
        "campaign_ref": "campaign_" + "a" * 64,
        "account_id": ACCOUNT,
        "region": REGION,
        "deployment_id": DEPLOYMENT,
        "kind": "Training",
        "operation_digest": "b" * 64,
        "job_arn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/fixture-original",
        "status": "Completed",
    }
    with pytest.raises(cc.CampaignCoordinatorError):
        cc.parse_wake_hint(json.dumps(payload).encode())
    payload.pop("status")
    payload["approval_policy"] = {"approved": True}
    with pytest.raises(cc.CampaignCoordinatorError):
        cc.parse_wake_hint(json.dumps(payload).encode())


def test_cold_coordinator_reuses_record_and_preserves_exact_prompt():
    cc, _, coordinator, guard, _, clock, ddb, _ = fixture_coordinator()
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    cold = cc.DynamoCampaignCoordinator(guard, ddb, table_name=TABLE, clock=clock)
    restored = cold.open_campaign(
        guard.verify(), TRANSPORT, PROMPT, expected_campaign_ref=record.campaign_ref
    )
    assert restored.campaign_id == record.campaign_id
    assert restored.private_store_correlation == record.private_store_correlation
    assert sum(name == "PutItem" for name, _ in ddb.calls) == 1


def test_revalidation_between_lookup_and_cas_stops_revoked_writer():
    import dataclasses

    _, ca, coordinator, guard, reader, _, ddb, _, _, record, _ = open_and_claim()
    current = coordinator.load_campaign(guard.verify(), record.campaign_ref)
    witness = guard.verify()
    real_get = ddb.get_item

    def read_then_drift(**kwargs):
        reply = real_get(**kwargs)
        reader.observation = dataclasses.replace(reader.observation, endpoint_live_version="2")
        return reply

    ddb.get_item = read_then_drift
    before = copy.deepcopy(ddb.items)
    with pytest.raises(ca.AdmissionError):
        coordinator.claim_lease(witness, record.campaign_ref, expected_revision=current.revision)
    assert ddb.items == before


def test_duplicate_or_reversed_wake_cannot_regress_terminal_observation():
    cc, _, coordinator, guard, _, _, ddb, _, witness, record, lease = open_and_claim()
    operation = original(cc)
    registered = coordinator.register_original(witness, lease, operation)
    lease = coordinator.claim_lease(
        guard.verify(), record.campaign_ref, expected_revision=registered.revision
    )
    observer = NativeJobs(cc)
    first = coordinator.observe_wake(guard.verify(), lease, wake(cc, record, operation), observer)
    before = copy.deepcopy(ddb.items)
    duplicate = coordinator.observe_wake(guard.verify(), lease, wake(cc, record, operation), observer)
    assert duplicate.status == first.status == "Completed"
    assert ddb.items == before
    observer.state = "InProgress"
    hint = wake(cc, record, operation)
    import dataclasses

    hint = dataclasses.replace(hint, event_id="fixture-event-2")
    with pytest.raises(cc.CampaignCoordinatorError):
        coordinator.observe_wake(guard.verify(), lease, hint, observer)
    assert ddb.items == before


def test_unknown_observation_retains_original_identity_without_success():
    cc, _, coordinator, guard, _, _, _, _, witness, record, lease = open_and_claim()
    operation = original(cc)
    registered = coordinator.register_original(witness, lease, operation)
    lease = coordinator.claim_lease(
        guard.verify(), record.campaign_ref, expected_revision=registered.revision
    )
    result = coordinator.observe_wake(
        guard.verify(), lease, wake(cc, record, operation), NativeJobs(cc, state="UNKNOWN")
    )
    restored = coordinator.load_campaign(guard.verify(), record.campaign_ref)
    assert restored.operations[0].original == operation
    assert result.status == "UNKNOWN"
    assert result.native_admission is False
    assert result.quality_success is False


def test_lost_lease_ack_reconciles_exact_updated_bytes_without_second_update():
    _, _, coordinator, guard, _, _, ddb, _ = fixture_coordinator()
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    ddb.lose_update = True
    lease = coordinator.claim_lease(guard.verify(), record.campaign_ref, expected_revision=record.revision)
    assert lease.epoch == 1
    assert sum(name == "UpdateItem" for name, _ in ddb.calls) == 1
    assert ddb.calls[-1][0] == "GetItem"


def test_closed_durable_codec_refuses_extra_claims_after_lookup():
    cc, _, coordinator, guard, _, _, ddb, _ = fixture_coordinator()
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    row = next(iter(ddb.items.values()))
    raw = row["private_record"]["B"]
    value = json.loads(raw)
    value["approved"] = True
    row["private_record"]["B"] = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(cc.CampaignCoordinatorError):
        coordinator.load_campaign(guard.verify(), record.campaign_ref)


def test_original_job_arn_and_kind_are_same_scope_closed_fields():
    cc = coordinator_module()
    operation = original(cc)
    import dataclasses

    with pytest.raises(cc.CampaignCoordinatorError):
        dataclasses.replace(operation, job_arn=operation.job_arn.replace(REGION, "us-east-1"))
    with pytest.raises(cc.CampaignCoordinatorError):
        dataclasses.replace(operation, kind="Processing")


@pytest.mark.parametrize(
    ("phase", "kind"),
    [
        ("dataset_acquisition", "Processing"),
        ("model_acquisition", "Processing"),
        ("case_freeze", "Processing"),
        ("preparation", "Processing"),
        ("paired_evaluation", "Processing"),
        ("sft", "Training"),
        ("reload", "Training"),
        ("generation_base", "Training"),
        ("generation_tuned", "Training"),
    ],
)
def test_phase_kind_matches_actual_product_job_type(phase, kind):
    import dataclasses

    cc = coordinator_module()
    operation = dataclasses.replace(original(cc, kind), phase=phase)
    assert operation.kind == kind
    opposite = "Processing" if kind == "Training" else "Training"
    other = original(cc, opposite)
    with pytest.raises(cc.CampaignCoordinatorError):
        dataclasses.replace(other, phase=phase)


@pytest.mark.parametrize(
    ("transport", "prompt"),
    [("a" * 32, PROMPT), ("a" * 257, PROMPT), (TRANSPORT, True), (TRANSPORT, ""), (TRANSPORT, "a" * 65537)],
)
def test_mapping_bounds_and_wrong_bool_refuse_before_lookup(transport, prompt):
    cc, _, coordinator, guard, _, _, ddb, _ = fixture_coordinator()
    with pytest.raises(cc.CampaignCoordinatorError):
        coordinator.open_campaign(guard.verify(), transport, prompt)
    assert ddb.calls == []


def test_lease_expiry_during_final_admission_read_refuses_before_cas():
    from test_campaign_admission import Clock

    cc, _, _, guard, reader, _, ddb, _ = fixture_coordinator()
    wall = Clock()
    coordinator = cc.DynamoCampaignCoordinator(guard, ddb, table_name=TABLE, clock=wall)
    witness = guard.verify()
    record = coordinator.open_campaign(witness, TRANSPORT, PROMPT)
    lease = coordinator.claim_lease(witness, record.campaign_ref, expected_revision=record.revision)
    before = copy.deepcopy(ddb.items)
    armed = [False]
    real_get, real_native = ddb.get_item, reader.read_current

    def read_and_arm(**kwargs):
        reply = real_get(**kwargs)
        armed[0] = True
        return reply

    def native_with_wall_delay():
        reply = real_native()
        if armed[0]:
            wall.value += 61
        return reply

    ddb.get_item, reader.read_current = read_and_arm, native_with_wall_delay
    with pytest.raises(cc.CampaignCoordinatorError):
        coordinator.register_original(witness, lease, original(cc))
    assert ddb.items == before


@pytest.mark.parametrize("action", ["register", "observe"])
def test_unknown_update_requires_exact_reconciliation_before_duplicate_return(action):
    cc, _, coordinator, guard, _, _, ddb, _, witness, record, lease = open_and_claim()
    operation = original(cc)
    observer = NativeJobs(cc)
    if action == "observe":
        registered = coordinator.register_original(witness, lease, operation)
        lease = coordinator.claim_lease(
            guard.verify(), record.campaign_ref, expected_revision=registered.revision
        )
    real_update = ddb.update_item

    def update_then_lose_read(**kwargs):
        ddb.lose_update = True
        try:
            return real_update(**kwargs)
        finally:
            ddb.fail_reads = True

    ddb.update_item = update_then_lose_read

    def invoke():
        if action == "register":
            return coordinator.register_original(guard.verify(), lease, operation)
        return coordinator.observe_wake(guard.verify(), lease, wake(cc, record, operation), observer)

    with pytest.raises(cc.CampaignUnresolved):
        invoke()
    reads = len(observer.calls)
    writes = sum(name == "UpdateItem" for name, _ in ddb.calls)
    row = next(iter(ddb.items.values()))
    value = json.loads(row["private_record"]["B"])
    value["revision"] += 1
    row["coordinator_revision"]["N"] = str(value["revision"])
    row["private_record"]["B"] = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ddb.fail_reads = False
    with pytest.raises(cc.CampaignUnresolved):
        invoke()
    assert len(observer.calls) == reads
    assert sum(name == "UpdateItem" for name, _ in ddb.calls) == writes
