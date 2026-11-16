"""Explicit neutral entry/protected-origin ports; no native isolation or model Create."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters
from test_campaign_coordinator import TABLE, open_and_claim

from amlas.backends.types import JobHandle
from amlas.data_eng.model_acquisition_schema import expected_digest
from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest


def module():
    try:
        value = importlib.import_module("amlas.control_plane.model_campaign")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.model_campaign":
            pytest.fail("missing-symbol: model_campaign", pytrace=False)
        raise
    for name in (
        "ModelAcquisitionConfig",
        "prepare_model_intent",
        "DynamoModelCampaignJournal",
        "ProtectedModelPublication",
        "ProtectedModelOriginal",
        "ProtectedModelRead",
    ):
        assert hasattr(value, name), "missing-symbol: " + name
    return value


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


class Table:
    def __init__(self):
        self.items = {}
        self.lock = threading.Lock()
        self.calls = []
        self.lose_put = False
        self.lose_update = False
        self.fail_reads = False
        self.retry = 0
        self.models = Session().get_service_model("dynamodb")

    def shape(self, name, kw):
        validate_parameters(kw, self.models.operation_model(name).input_shape)
        assert kw["TableName"] == TABLE
        self.calls.append((name, copy.deepcopy(kw)))

    def get_item(self, **kw):
        self.shape("GetItem", kw)
        assert kw["ConsistentRead"] is True
        if self.fail_reads:
            raise TimeoutError("neutral unavailable read")
        with self.lock:
            row = copy.deepcopy(self.items.get(kw["Key"]["job_name"]["S"]))
        return {"Item": row} if row is not None else {}

    def put_item(self, **kw):
        self.shape("PutItem", kw)
        assert kw["ConditionExpression"] == "attribute_not_exists(job_name)"
        key = kw["Item"]["job_name"]["S"]
        with self.lock:
            if key in self.items:
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
            self.items[key] = copy.deepcopy(kw["Item"])
        if self.lose_put:
            self.lose_put = False
            if getattr(self, "fail_after_put", False):
                self.fail_reads = True
            raise TimeoutError("neutral accepted put acknowledgement lost")
        return {"ResponseMetadata": {"RetryAttempts": self.retry}}

    def update_item(self, **kw):
        self.shape("UpdateItem", kw)
        assert (
            kw["ConditionExpression"]
            == "journal_revision = :revision AND journal_state = :state AND private_record = :prior"
        )
        assert (
            kw["UpdateExpression"]
            == "SET journal_revision = :next, journal_state = :new_state, private_record = :body"
        )
        v = kw["ExpressionAttributeValues"]
        key = kw["Key"]["job_name"]["S"]
        with self.lock:
            row = self.items.get(key)
            if row is None or any(
                row[k] != v[n]
                for k, n in [
                    ("journal_revision", ":revision"),
                    ("journal_state", ":state"),
                    ("private_record", ":prior"),
                ]
            ):
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
            row.update(
                journal_revision=copy.deepcopy(v[":next"]),
                journal_state=copy.deepcopy(v[":new_state"]),
                private_record=copy.deepcopy(v[":body"]),
            )
        if self.lose_update:
            self.lose_update = False
            if getattr(self, "fail_after_update", False):
                self.fail_reads = True
            raise TimeoutError("neutral accepted marker acknowledgement lost")
        return {"ResponseMetadata": {"RetryAttempts": self.retry}}


class Protected:
    """Explicitly neutral qualified port, not evidence of native permissions/isolation."""

    def __init__(self, m):
        self.m = m
        self.originals = {}
        self.writes = 0
        self.reads = 0
        self.qualified = True
        self.identity = "neutral-physical-table"
        self.lose = False
        self.publish_calls = 0

    def publish_first(self, entry_binding, *, planning_ref, spec_sha256, canonical_intent, qualify_io):
        qualify_io()
        self.publish_calls += 1
        key = (entry_binding["campaign_ref"], planning_ref)
        if key not in self.originals:
            digest = hashlib.sha256(canonical_intent).hexdigest()
            pub = self.m.ProtectedModelPublication(
                "protected_fixture_" + digest,
                digest,
                hashlib.sha256(b"neutral_ack" + canonical_intent).hexdigest(),
            )
            self.originals[key] = self.m.ProtectedModelOriginal(pub, canonical_intent)
            self.writes += 1
        elif self.originals[key].canonical_intent != canonical_intent:
            raise ValueError("neutral original conflict")
        qualify_io()
        if self.lose:
            self.lose = False
            raise TimeoutError("neutral protected ack lost")
        return self.originals[key].publication

    def read_original(self, entry_binding, *, planning_ref, qualify_io):
        qualify_io()
        self.reads += 1
        value = self.originals.get((entry_binding["campaign_ref"], planning_ref))
        qualify_io()
        return (
            self.m.ProtectedModelRead("PRESENT", value)
            if value is not None
            else self.m.ProtectedModelRead("ABSENT")
        )

    def require_current(self, publication, original, entry_binding, *, qualify_io):
        qualify_io()
        if not self.qualified:
            raise ValueError("neutral current protection unavailable")
        if publication is None:
            assert original is None
        else:
            assert publication == original.publication
        qualify_io()
        return {"physical_id": self.identity, "native_capability": False}


def fixture():
    m = module()
    _cc, _ca, coord, guard, _reader, clock, _ddb, _approved, _w, record, lease = open_and_claim()
    scope = coord._scope(guard.verify())
    spec = "a" * 64
    entry = SimpleNamespace(
        campaign_ref=record.campaign_ref,
        campaign_id=record.campaign_id,
        private_store_correlation=record.private_store_correlation,
        transport_session_id=record.binding["transport_session_id"],
        literal_prompt=record.literal_prompt,
        prompt_sha256=record.prompt_sha256,
        planning_refs=(SimpleNamespace(kind="spec", reference="spec_fixture", sha256=spec),),
    )
    active = {"yes": True, "checks": 0}

    def current():
        active["checks"] += 1
        if not active["yes"]:
            raise ValueError("neutral current entry revoked")

    def entry_qualifier(value):
        current()
        assert value is entry
        return guard.verify()

    cfg = m.ModelAcquisitionConfig(
        scope["account_id"],
        scope["region"],
        "fixture-model-bucket",
        f"arn:aws:iam::{scope['account_id']}:role/fixture",
        f"{scope['account_id']}.dkr.ecr.{scope['region']}.amazonaws.com/model@sha256:" + "b" * 64,
        "c" * 40,
        "b" * 64,
        scope["deployment_id"],
        scope["deployment_prefix"],
        None,
    )

    def prepare_submission(*, catalog_id, catalog_version, runtime_session_id, job_name=None):
        bindings = {
            "job_name": job_name,
            "runtime_session_id": runtime_session_id,
            "deployment_id": cfg.deployment_id,
            "deployment_prefix": cfg.deployment_prefix,
            "image_uri": cfg.image_uri,
            "source_commit": cfg.source_commit,
        }
        args = ["--catalog-id", catalog_id, "--catalog-version", catalog_version]
        for key, value in bindings.items():
            args += ["--" + key.replace("_", "-"), value]
        tags = [
            {"Key": "amlas:" + k, "Value": v}
            for k, v in [
                ("deployment-id", cfg.deployment_id),
                ("deployment-prefix", cfg.deployment_prefix),
                ("runtime-session-id", runtime_session_id),
            ]
        ]
        prefix = f"sessions/{runtime_session_id}/model-acquisition/{job_name}/"
        req = {
            "ProcessingJobName": job_name,
            "RoleArn": cfg.execution_role_arn,
            "AppSpecification": {
                "ImageUri": cfg.image_uri,
                "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "model-acquisition"],
                "ContainerArguments": args,
            },
            "ProcessingResources": {
                "ClusterConfig": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 80}
            },
            "NetworkConfig": {"EnableNetworkIsolation": False},
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "model_acquisition",
                        "AppManaged": False,
                        "S3Output": {
                            "S3Uri": f"s3://{cfg.bucket}/{prefix}",
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
            "Environment": {"AMLAS_AGENT_REMOTE_FETCH": "1"},
            "Tags": tags,
        }
        arn = f"arn:aws:sagemaker:{cfg.region}:{cfg.account_id}:processing-job/{job_name}"
        return JobHandle(
            job_name,
            "model_processing",
            arn,
            {
                "accepted": False,
                "submission_ambiguous": False,
                "acquired": False,
                "catalog_id": catalog_id,
                "catalog_version": catalog_version,
                "expected_manifest_sha256": expected_digest(
                    expected_model_weight_manifest(catalog_id, catalog_version)
                ),
                "bindings": bindings,
                "bucket": cfg.bucket,
                "account_id": cfg.account_id,
                "region": cfg.region,
                "output_prefix": prefix,
                "request": req,
                "native_status": "Submitted",
            },
        )

    def validate(handle):
        expected = prepare_submission(
            catalog_id="qwen3_5_9b",
            catalog_version="1",
            runtime_session_id=entry.private_store_correlation,
            job_name=handle.id,
        )
        if handle != expected:
            raise ValueError("neutral source rebuilt request differs")
        return expected

    protected = Protected(m)
    table = Table()

    def origin(port, publication, original, binding, evidence):
        assert (
            port is protected
            and protected.qualified
            and evidence["physical_id"] == "neutral-physical-table"
            and evidence["native_capability"] is False
        )

    journal = m.DynamoModelCampaignJournal(
        entry,
        entry_qualifier=entry_qualifier,
        protected_original_port=protected,
        protected_origin_qualifier=origin,
        coordinator=coord,
        table_name=TABLE,
        dynamodb_client=table,
        clock_ms=lambda: int(clock() * 1000),
        validate_prepared=validate,
    )
    intent = m.prepare_model_intent(
        entry,
        entry_qualifier=entry_qualifier,
        planning_ref="spec_fixture",
        spec_sha256=spec,
        config=cfg,
        prepare_submission=prepare_submission,
        validate_prepared=validate,
    )
    return m, journal, intent, protected, table, active, current, lease, coord, entry, cfg, validate


def original_flow(parts):
    _m, j, intent, _port, _table, _a, current, lease, _coord, _entry, _cfg, _validate = parts
    with j.qualified_io(current):
        publication = j.publish_original(intent)
        reserved = j.reserve_original(intent, publication=publication)
        registered = j.register_original(reserved, lease=lease)
    return publication, registered


def test_original_protected_publication_precedes_one_qualified_model_attempt_and_outcome():
    parts = fixture()
    _m, j, _intent, p, t, a, current, lease, *_ = parts
    assert not t.calls and p.writes == 0
    _publication, reservation = original_flow(parts)
    assert p.writes == 1 and reservation.model_ref.startswith("model_acq_")
    with j.qualified_io(current):
        permit = j.consume_create_attempt(reservation, lease=lease)
        j.require_effect(permit, reservation)
        with pytest.raises(ValueError):
            j.require_effect(permit, reservation)
        result = j.record_outcome(
            reservation,
            permit=permit,
            disposition="ACCEPTED",
            native_job_arn=reservation.job_arn,
            error_type=None,
        )
        assert result.state == "ACCEPTED"
    assert p.writes == 1 and a["checks"] > 5


def test_missing_io_or_protected_origin_never_reserves_or_grants_create():
    parts = fixture()
    _m, j, intent, p, t, _a, current, *_ = parts
    with pytest.raises(ValueError):
        j.publish_original(intent)
    with j.qualified_io(current):
        publication = j.publish_original(intent)
        p.qualified = False
        with pytest.raises(ValueError):
            j.reserve_original(intent, publication=publication)
    assert not t.calls


def test_lost_marker_ack_is_descriptive_original_only_never_another_permit():
    parts = fixture()
    m, j, intent, _p, t, _a, current, lease, *_ = parts
    _pub, r = original_flow(parts)
    t.lose_update = True
    with j.qualified_io(current):
        unknown = j.consume_create_attempt(r, lease=lease)
        assert type(unknown) is m.ModelOperationReservation and unknown.state == "CREATE_ATTEMPTED"
        again = j.consume_create_attempt(unknown, lease=lease)
        assert type(again) is m.ModelOperationReservation
        assert j.load_original(intent).job_name == r.job_name


def test_retried_marker_response_cannot_supply_create_permit():
    parts = fixture()
    _m, j, intent, _p, t, _a, current, lease, *_ = parts
    _, r = original_flow(parts)
    t.retry = 1
    with j.qualified_io(current):
        with pytest.raises(ValueError):
            j.consume_create_attempt(r, lease=lease)
        assert j.load_original(intent).state == "CREATE_ATTEMPTED"


def test_cold_duplicate_original_and_notfound_restore_never_creates():
    parts = fixture()
    m, j, intent, p, t, _a, current, _lease, coord, entry, _cfg, validate = parts
    pub, r = original_flow(parts)
    second = m.DynamoModelCampaignJournal(
        entry,
        entry_qualifier=j._entry_qualifier,
        protected_original_port=p,
        protected_origin_qualifier=j._origin_qualifier,
        coordinator=coord,
        table_name=TABLE,
        dynamodb_client=t,
        clock_ms=j._clock,
        validate_prepared=validate,
    )
    before = len([v for v in t.calls if v[0] == "PutItem"])
    with second.qualified_io(current):
        duplicate = second.reserve_original(intent, publication=pub)
        assert (duplicate.job_name, duplicate.model_ref) == (r.job_name, r.model_ref)
        observed = second.restore_original(
            intent,
            describe_original=lambda _: {
                "status": "NotFound",
                "job_name": r.job_name,
                "job_arn": r.job_arn,
                "request_sha256": r.request_sha256,
                "source_binding_sha256": r.source_binding_sha256,
            },
        )
        assert observed.status == "UNKNOWN"
    assert len([v for v in t.calls if v[0] == "PutItem"]) == before


@pytest.mark.parametrize("change", ["request", "spec", "planning", "entry", "image"])
def test_source_rebuilt_intent_replacement_refuses(change):
    parts = fixture()
    _m, j, intent, p, t, _a, current, *_ = parts
    data = json.loads(intent.canonical_bytes)
    if change == "request":
        data["prepared"]["extra"]["request"]["StoppingCondition"]["MaxRuntimeInSeconds"] = 1
    elif change == "spec":
        data["spec_sha256"] = "e" * 64
    elif change == "planning":
        data["planning_ref"] = "other_spec"
    elif change == "entry":
        data["entry"]["private_store_correlation"] = "00000000-0000-4000-8000-00000000ffff"
    else:
        data["config"]["image_uri"] = data["config"]["image_uri"].replace("b" * 64, "e" * 64)
    with j.qualified_io(current):
        with pytest.raises(ValueError):
            j.publish_original(canonical(data))
    assert p.writes == 0 and not t.calls


def test_copied_or_replaced_model_permit_never_grants_a_second_effect():
    parts = fixture()
    _m, j, _intent, _p, _t, _a, current, lease, *_ = parts
    _, r = original_flow(parts)
    with j.qualified_io(current):
        permit = j.consume_create_attempt(r, lease=lease)
        copied = copy.copy(permit)
        j.require_effect(permit, r)
        with pytest.raises(ValueError):
            j.require_effect(copied, r)
        with pytest.raises(ValueError):
            j.require_effect(dataclasses.replace(permit, used=False), r)


def test_reserve_write_retry_violation_is_not_swallowed_as_success():
    parts = fixture()
    _m, j, intent, _p, t, _a, current, *_ = parts
    with j.qualified_io(current):
        pub = j.publish_original(intent)
        t.retry = 1
        with pytest.raises(ValueError):
            j.reserve_original(intent, publication=pub)
    assert len([v for v in t.calls if v[0] == "PutItem"]) == 1


def test_concurrent_reserve_and_attempt_losers_retain_original_without_effect_permit():
    parts = fixture()
    m, j, intent, p, t, _a, current, lease, coord, entry, _cfg, validate = parts
    with j.qualified_io(current):
        pub = j.publish_original(intent)
    other = m.DynamoModelCampaignJournal(
        entry,
        entry_qualifier=j._entry_qualifier,
        protected_original_port=p,
        protected_origin_qualifier=j._origin_qualifier,
        coordinator=coord,
        table_name=TABLE,
        dynamodb_client=t,
        clock_ms=j._clock,
        validate_prepared=validate,
    )

    def reserve(port):
        with port.qualified_io(current):
            return port.reserve_original(intent, publication=pub)

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(reserve, [j, other]))
    assert rows[0].job_name == rows[1].job_name and rows[0].model_ref == rows[1].model_ref
    with j.qualified_io(current):
        registered = j.register_original(rows[0], lease=lease)

    def marker(port):
        with port.qualified_io(current):
            return port.consume_create_attempt(registered, lease=lease)

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = []
        for future in [pool.submit(marker, j), pool.submit(marker, other)]:
            try:
                answers.append(future.result())
            except ValueError:
                answers.append(None)
    assert sum(type(v) is m.ModelCreatePermit for v in answers) == 1


def test_bound_model_row_cannot_project_contradictory_attempt_or_outcome():
    parts = fixture()
    _m, j, intent, _p, t, _a, current, lease, *_ = parts
    _, r = original_flow(parts)
    with j.qualified_io(current):
        _permit = j.consume_create_attempt(r, lease=lease)
        key = next(iter(t.items))
        row = json.loads(t.items[key]["private_record"]["B"])
        row["outcome_arn"] = r.job_arn
        t.items[key]["private_record"]["B"] = canonical(row)
        with pytest.raises(ValueError):
            j.load_original(intent)


def test_intent_matches_roots_actual_pure_adapter_without_sdk(monkeypatch):
    parts = fixture()
    m, j, intent, _p, _t, _a, _current, _lease, _coord, entry, cfg, _ = parts
    from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend

    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", cfg.deployment_id)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", cfg.deployment_prefix)
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    monkeypatch.delenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", raising=False)
    backend = ProcessingModelWeightAcquisitionBackend(
        region=cfg.region,
        role_arn=cfg.execution_role_arn,
        image_uri=cfg.image_uri,
        session_bucket=cfg.bucket,
        source_commit=cfg.source_commit,
        admission_digest=cfg.admission_digest,
    )
    backend._clients = lambda *_: pytest.fail("pure source comparison attempted SDK access")
    actual = m.prepare_model_intent(
        entry,
        entry_qualifier=j._entry_qualifier,
        planning_ref="spec_fixture",
        spec_sha256="a" * 64,
        config=cfg,
        prepare_submission=backend.prepare_submission,
        validate_prepared=backend.validate_prepared,
    )
    assert actual.canonical_bytes == intent.canonical_bytes


@pytest.mark.parametrize("boundary", ["put", "get", "protected", "return"])
def test_current_qualification_revocation_at_actual_boundaries_refuses(boundary):
    parts = fixture()
    _m, j, intent, p, t, a, current, *_ = parts
    with j.qualified_io(current):
        pub = j.publish_original(intent)
        if boundary == "put":
            original = t.put_item

            def changed(**kwargs):
                result = original(**kwargs)
                a["yes"] = False
                return result

            t.put_item = changed
            with pytest.raises(ValueError):
                j.reserve_original(intent, publication=pub)
        else:
            _r = j.reserve_original(intent, publication=pub)
            if boundary == "get":
                original = t.get_item

                def changed(**kwargs):
                    result = original(**kwargs)
                    a["yes"] = False
                    return result

                t.get_item = changed
            elif boundary == "protected":
                p.qualified = False
            else:
                original = j._reservation

                def changed(*args):
                    result = original(*args)
                    a["yes"] = False
                    return result

                # Caller receives no result when the final qualification changes.
                original_clock = j._clock

                def changed_clock():
                    a["yes"] = False
                    return original_clock()

                j._clock = changed_clock
                a["yes"] = False
            with pytest.raises(ValueError):
                j.load_original(intent)


def test_late_outcome_after_new_lease_epoch_refuses_without_mutation():
    parts = fixture()
    _m, j, _intent, _p, t, _a, current, lease, coord, entry, _cfg, _v = parts
    _, r = original_flow(parts)
    with j.qualified_io(current):
        permit = j.consume_create_attempt(r, lease=lease)
        j.require_effect(permit, r)
        record = coord.load_campaign(j._entry_qualifier(entry), entry.campaign_ref)
        # A fresh same-holder lease still advances the fencing epoch.
        coord.claim_lease(j._entry_qualifier(entry), entry.campaign_ref, record.revision, lease_seconds=60)
        before = len([x for x in t.calls if x[0] == "UpdateItem"])
        with pytest.raises(ValueError):
            j.record_outcome(
                r, permit=permit, disposition="ACCEPTED", native_job_arn=r.job_arn, error_type=None
            )
        assert len([x for x in t.calls if x[0] == "UpdateItem"]) == before


def test_lost_protected_publication_is_read_resolved_without_republish():
    parts = fixture()
    _m, j, intent, p, t, _a, current, *_ = parts
    p.lose = True
    with j.qualified_io(current):
        publication = j.publish_original(intent)
        assert publication.intent_sha256 == hashlib.sha256(intent.canonical_bytes).hexdigest()
        assert p.writes == 1 and not t.calls
        same = j.publish_original(intent)
        assert same == publication and p.writes == 1


def test_inherited_journal_missing_qualification_and_publication_replacement_refuse():
    parts = fixture()
    _m, j, intent, _p, t, _a, current, *_ = parts
    with j.qualified_io(current):
        pub = j.publish_original(intent)
        with pytest.raises(ValueError):
            j.reserve_original(intent, publication=dataclasses.replace(pub, intent_sha256="e" * 64))
    j._pid = -1
    with pytest.raises(ValueError):
        with j.qualified_io(current):
            j.load_original(intent)
    assert not t.calls


@pytest.mark.parametrize("field", ["role", "hosts"])
def test_valid_source_plan_from_different_config_cannot_bind_original_config(field):
    parts = fixture()
    m, j, intent, _p, _t, _a, _current, _lease, _coord, entry, cfg, _validate = parts
    original = json.loads(intent.canonical_bytes)["prepared"]

    def alternate(**kwargs):
        wire = copy.deepcopy(original)
        if field == "role":
            wire["extra"]["request"]["RoleArn"] += "-different"
        else:
            wire["extra"]["request"]["Environment"]["AMLAS_AGENT_HTTP_ALLOW_HOSTS"] = "ordinary.example"
        return JobHandle(wire["id"], wire["backend"], wire["arn"], wire["extra"])

    # The other source-configured adapter legitimately validates its own different
    # plan; the journal must ALSO compare its separately frozen config tuple.
    def alternate_validator(handle):
        assert handle == alternate()
        return handle

    with pytest.raises(ValueError):
        m.prepare_model_intent(
            entry,
            entry_qualifier=j._entry_qualifier,
            planning_ref="spec_fixture",
            spec_sha256="a" * 64,
            config=cfg,
            prepare_submission=alternate,
            validate_prepared=alternate_validator,
        )


def test_cold_original_load_uses_protected_planning_selector_not_mutable_row_intent():
    parts = fixture()
    _m, j, intent, p, _t, _a, current, _lease, *_ = parts
    _, reserved = original_flow(parts)
    with j.qualified_io(current):
        restored = j.load_original("spec_fixture")
        assert restored.job_name == reserved.job_name and restored.intent == intent
    assert p.reads > 0


def test_unresolved_reserve_and_marker_never_replay_attempted_mutation():
    parts = fixture()
    _m, j, intent, _p, t, _a, current, lease, *_ = parts
    with j.qualified_io(current):
        pub = j.publish_original(intent)
        t.lose_put = True
        t.fail_after_put = True
        with pytest.raises(ValueError):
            j.reserve_original(intent, publication=pub)
        puts = len([v for v in t.calls if v[0] == "PutItem"])
        t.fail_reads = False
        r = j.reserve_original(intent, publication=pub)
        assert len([v for v in t.calls if v[0] == "PutItem"]) == puts
        r = j.register_original(r, lease=lease)
        t.lose_update = True
        t.fail_after_update = True
        with pytest.raises(ValueError):
            j.consume_create_attempt(r, lease=lease)
        updates = len([v for v in t.calls if v[0] == "UpdateItem"])
        t.fail_reads = False
        again = j.consume_create_attempt(r, lease=lease)
        assert again.state == "CREATE_ATTEMPTED"
        assert len([v for v in t.calls if v[0] == "UpdateItem"]) == updates


def test_callback_cannot_extend_model_effect_deadline_after_last_lease_check():
    parts = fixture()
    _m, j, _intent, _p, _t, _a, current, lease, *_ = parts
    _, r = original_flow(parts)
    with j.qualified_io(current):
        permit = j.consume_create_attempt(r, lease=lease)
        original = j._lease
        crossed = [False]

        def last_lease(*args):
            answer = original(*args)
            crossed[0] = True
            return answer

        j._lease = last_lease

        def expiry_callback():
            current()
            if crossed[0]:
                j._clock = lambda: lease.deadline_ms

        with j.qualified_io(expiry_callback):
            with pytest.raises(ValueError):
                j.require_effect(permit, r)


def test_unqualified_first_publication_refuses_before_any_protected_write():
    parts = fixture()
    _m, j, intent, p, t, _a, current, *_ = parts
    p.qualified = False
    with j.qualified_io(current):
        with pytest.raises(ValueError):
            j.publish_original(intent)
    assert p.publish_calls == 0 and p.writes == 0 and not t.calls


def test_present_original_is_read_only_and_never_calls_first_publisher_again():
    parts = fixture()
    _m, j, intent, p, t, _a, current, *_ = parts
    with j.qualified_io(current):
        first = j.publish_original(intent)
        assert j.publish_original(intent) == first
    assert p.publish_calls == 1 and p.writes == 1 and not t.calls


def test_plain_none_from_protected_reader_is_not_qualified_absence():
    parts = fixture()
    _m, j, intent, p, t, _a, current, *_ = parts
    p.read_original = lambda *args, **kwargs: None
    with j.qualified_io(current):
        with pytest.raises(ValueError):
            j.publish_original(intent)
    assert p.publish_calls == 0 and p.writes == 0 and not t.calls


def test_ambiguous_protected_publication_with_unresolved_reader_never_republishes():
    parts = fixture()
    _m, j, intent, p, t, _a, current, *_ = parts
    original = p.read_original

    def unresolved_after_attempt(*args, **kwargs):
        if p.publish_calls:
            raise TimeoutError("neutral protected read unavailable")
        return original(*args, **kwargs)

    p.read_original = unresolved_after_attempt
    p.lose = True
    with j.qualified_io(current):
        with pytest.raises(ValueError):
            j.publish_original(intent)
        with pytest.raises(ValueError):
            j.publish_original(intent)
    assert p.publish_calls == 1 and p.writes == 1 and not t.calls


def test_unknown_protected_existence_refuses_zero_publications():
    parts = fixture()
    m, j, intent, p, t, _a, current, *_ = parts
    p.read_original = lambda *args, **kwargs: m.ProtectedModelRead("UNKNOWN")
    with j.qualified_io(current):
        with pytest.raises(ValueError):
            j.publish_original(intent)
    assert p.publish_calls == 0 and p.writes == 0 and not t.calls


@pytest.mark.parametrize("leg", ["marker", "outcome"])
def test_callback_expiry_at_model_cas_refuses_before_actual_write(leg):
    parts = fixture()
    _m, j, _intent, _p, t, _a, current, lease, *_ = parts
    _, r = original_flow(parts)
    with j.qualified_io(current):
        if leg == "outcome":
            permit = j.consume_create_attempt(r, lease=lease)
            j.require_effect(permit, r)
        original = j._cas
        crossed = [False]

        def last_cas(*args, **kwargs):
            crossed[0] = True
            return original(*args, **kwargs)

        j._cas = last_cas

        def expiry_callback():
            current()
            if crossed[0]:
                j._clock = lambda: lease.deadline_ms

        before = len([v for v in t.calls if v[0] == "UpdateItem"])
        with j.qualified_io(expiry_callback):
            with pytest.raises(ValueError):
                if leg == "marker":
                    j.consume_create_attempt(r, lease=lease)
                else:
                    j.record_outcome(
                        r, permit=permit, disposition="ACCEPTED", native_job_arn=r.job_arn, error_type=None
                    )
        assert len([v for v in t.calls if v[0] == "UpdateItem"]) == before


def test_postwrite_callback_expiry_cannot_issue_model_effect_permit():
    parts = fixture()
    _m, j, _intent, _p, t, _a, current, lease, *_ = parts
    _, r = original_flow(parts)
    update = t.update_item

    def expire_after_marker(**kwargs):
        value = update(**kwargs)
        j._clock = lambda: lease.deadline_ms
        return value

    t.update_item = expire_after_marker
    with j.qualified_io(current):
        with pytest.raises(ValueError):
            j.consume_create_attempt(r, lease=lease)
        assert j.load_original(r.intent).state == "CREATE_ATTEMPTED"
    assert not j._permits


def test_outcome_exposure_rechecks_deadline_after_last_qualification_callback():
    parts = fixture()
    _m, j, _intent, _p, _t, _a, current, lease, *_ = parts
    _, r = original_flow(parts)
    with j.qualified_io(current):
        permit = j.consume_create_attempt(r, lease=lease)
        j.require_effect(permit, r)
        original = j._reservation
        exposing = [False]

        def last_reservation(*args, **kwargs):
            exposing[0] = True
            return original(*args, **kwargs)

        j._reservation = last_reservation

        def expiry_callback():
            current()
            if exposing[0]:
                j._clock = lambda: lease.deadline_ms

        with j.qualified_io(expiry_callback):
            with pytest.raises(ValueError):
                j.record_outcome(
                    r, permit=permit, disposition="ACCEPTED", native_job_arn=r.job_arn, error_type=None
                )
        assert j.load_original(r.intent).state == "ACCEPTED"


def test_model_intent_uses_private_store_correlation_and_preserves_transport_separately():
    parts = fixture()
    _m, _j, intent, _p, _t, _a, _current, _lease, _coord, entry, _cfg, _validate = parts
    data = json.loads(intent.canonical_bytes)
    assert entry.private_store_correlation != entry.transport_session_id
    assert data["entry"]["private_store_correlation"] == entry.private_store_correlation
    assert data["entry"]["transport_session_id"] == entry.transport_session_id
    extra = data["prepared"]["extra"]
    assert extra["bindings"]["runtime_session_id"] == entry.private_store_correlation
    assert extra["output_prefix"].startswith("sessions/" + entry.private_store_correlation + "/")
    assert {"Key": "amlas:runtime-session-id", "Value": entry.private_store_correlation} in extra["request"][
        "Tags"
    ]


def test_prepared_original_model_is_accepted_by_private_correlation_store_only():
    parts = fixture()
    _m, _j, intent, _p, _t, _a, _current, _lease, _coord, entry, cfg, _validate = parts
    from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
    from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore

    wire = json.loads(intent.canonical_bytes)["prepared"]
    handle = JobHandle(wire["id"], wire["backend"], wire["arn"], wire["extra"])
    adapter = ProcessingModelWeightAcquisitionBackend(
        region=cfg.region,
        role_arn=cfg.execution_role_arn,
        image_uri=cfg.image_uri,
        session_bucket=cfg.bucket,
        source_commit=cfg.source_commit,
        admission_digest=cfg.admission_digest,
    )
    private = SessionModelAcquisitionStore(runtime_session_id=entry.private_store_correlation)
    reference = private.add(adapter, handle)
    assert reference.startswith("model_acq_")
    transport = SessionModelAcquisitionStore(runtime_session_id=entry.transport_session_id)
    with pytest.raises(ValueError):
        transport.add(adapter, handle)


def test_cold_private_store_restores_same_original_opaque_model_ref_without_transport_alias():
    parts = fixture()
    m, j, intent, p, t, _a, current, _lease, coord, entry, cfg, validate = parts
    _, reserved = original_flow(parts)
    cold = m.DynamoModelCampaignJournal(
        entry,
        entry_qualifier=j._entry_qualifier,
        protected_original_port=p,
        protected_origin_qualifier=j._origin_qualifier,
        coordinator=coord,
        table_name=TABLE,
        dynamodb_client=t,
        clock_ms=j._clock,
        validate_prepared=validate,
    )
    with cold.qualified_io(current):
        restored = cold.load_original("spec_fixture")
    assert restored.model_ref == reserved.model_ref and restored.job_name == reserved.job_name
    from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
    from amlas.control_plane.campaign_state import CampaignStateCatalog, _mint_binding
    from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore

    wire = json.loads(restored.intent.canonical_bytes)["prepared"]
    handle = JobHandle(wire["id"], wire["backend"], wire["arn"], wire["extra"])
    adapter = ProcessingModelWeightAcquisitionBackend(
        region=cfg.region,
        role_arn=cfg.execution_role_arn,
        image_uri=cfg.image_uri,
        session_bucket=cfg.bucket,
        source_commit=cfg.source_commit,
        admission_digest=cfg.admission_digest,
    )
    scope = CampaignStateCatalog(
        cfg.account_id,
        cfg.region,
        cfg.deployment_id,
        cfg.deployment_prefix,
        cfg.deployment_prefix + "-training-jobs",
        cfg.execution_role_arn,
        "AROA" + "A" * 16,
    )
    # Neutral server binding fixture is data, no native role/protection grant.
    binding = _mint_binding(scope, entry.campaign_id, entry.private_store_correlation)
    private = SessionModelAcquisitionStore(runtime_session_id=entry.private_store_correlation)
    private._restore_original_producer(restored.model_ref, adapter, handle, binding)
    assert private.existing("qwen3_5_9b", "1") == reserved.model_ref
    assert private._original_producer_handle(reserved.model_ref, binding) == handle
    transport = SessionModelAcquisitionStore(runtime_session_id=entry.transport_session_id)
    with pytest.raises(ValueError):
        transport._restore_original_producer(restored.model_ref, adapter, handle, binding)
    assert transport.existing("qwen3_5_9b", "1") is None
    assert restored.intent == intent


def test_protected_origin_revocation_during_lease_read_refuses_effect_consumption():
    parts = fixture()
    _m, j, _intent, p, _t, _a, current, lease, *_ = parts
    _, r = original_flow(parts)
    with j.qualified_io(current):
        permit = j.consume_create_attempt(r, lease=lease)
        original = j._lease

        def revoked_during_lease(*args, **kwargs):
            answer = original(*args, **kwargs)
            p.qualified = False
            return answer

        j._lease = revoked_during_lease
        with pytest.raises(ValueError):
            j.require_effect(permit, r)
        assert not j._permits[id(permit)]["used"] and not permit.used


@pytest.mark.parametrize(
    "method",
    [
        "publish_original",
        "reserve_original",
        "register_original",
        "consume_create_attempt",
        "require_effect",
        "record_outcome",
        "load_original",
        "restore_original",
    ],
)
def test_inherited_model_journal_refuses_before_entering_any_lifecycle_lock(method):
    parts = fixture()
    _m, j, intent, _p, _t, _a, _current, lease, *_ = parts

    class ForbiddenLock:
        entered = False

        def __enter__(self):
            self.entered = True
            raise RuntimeError("neutral inherited lock must not be touched")

        def __exit__(self, *args):
            return False

    lock = ForbiddenLock()
    j._lock = lock
    j._pid = -1
    args = {
        "publish_original": ((intent,), {}),
        "reserve_original": ((intent,), {"publication": None}),
        "register_original": ((None,), {"lease": lease}),
        "consume_create_attempt": ((None,), {"lease": lease}),
        "require_effect": ((None, None), {}),
        "record_outcome": (
            (None,),
            {"permit": None, "disposition": "ACCEPTED", "native_job_arn": None, "error_type": None},
        ),
        "load_original": ((intent,), {}),
        "restore_original": ((intent,), {"describe_original": lambda _: None}),
    }
    positional, keywords = args[method]
    with pytest.raises(ValueError):
        getattr(j, method)(*positional, **keywords)
    assert not lock.entered


@pytest.mark.parametrize("replacement", [True, 1.0])
def test_embedded_original_intent_requires_exact_canonical_numeric_types(replacement):
    parts = fixture()
    _m, j, intent, _p, t, _a, current, *_ = parts
    original_flow(parts)
    key = next(iter(t.items))
    row = json.loads(t.items[key]["private_record"]["B"])
    row["intent"]["prepared"]["extra"]["request"]["ProcessingResources"]["ClusterConfig"]["InstanceCount"] = (
        replacement
    )
    t.items[key]["private_record"]["B"] = canonical(row)
    with j.qualified_io(current):
        with pytest.raises(ValueError):
            j.load_original(intent)


def test_accepted_model_outcome_with_error_class_refuses_before_update():
    parts = fixture()
    _m, j, _intent, _p, t, _a, current, lease, *_ = parts
    _, r = original_flow(parts)
    with j.qualified_io(current):
        permit = j.consume_create_attempt(r, lease=lease)
        j.require_effect(permit, r)
        before = len([v for v in t.calls if v[0] == "UpdateItem"])
        with pytest.raises(ValueError):
            j.record_outcome(
                r, permit=permit, disposition="ACCEPTED", native_job_arn=r.job_arn, error_type="NeutralError"
            )
        assert len([v for v in t.calls if v[0] == "UpdateItem"]) == before
        assert j.load_original(r.intent).state == "CREATE_ATTEMPTED"
