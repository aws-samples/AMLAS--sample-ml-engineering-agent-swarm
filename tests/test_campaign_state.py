"""Existing-table conditional campaign records; strict SDK fakes, no live AWS.

SDK request oracles: DynamoDB PutItem/GetItem/UpdateItem and STS GetCallerIdentity
service models from the declared botocore closure. Conditional checks run under
one fake server lock. These fixtures establish no native durability/admission.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters
from test_qwen_sft_backend import ACCOUNT, SESSION, backend, catalog

from amlas.backends.adapters.campaign_state import DynamoCampaignStateBackend
from amlas.control_plane.campaign_state import (
    CampaignStateCatalog,
    CampaignStateError,
    TrustedExecutionBinding,
    build_sft_intent,
    operation_digest,
)

ROLE_ID = "AROA" + "X" * 17
CONTROL_ROLE = "arn:aws:iam::" + ACCOUNT + ":role/owned-control-plane"
CAMPAIGN = "12345678-1234-4234-9234-00000000aa01"
MODELS = {name: Session().get_service_model(name) for name in ("dynamodb", "sts", "sagemaker")}


class STS:
    def __init__(self):
        self.role_id = ROLE_ID
        self.account = ACCOUNT
        self.calls = 0

    def get_caller_identity(self, **kwargs):
        validate_parameters(kwargs, MODELS["sts"].operation_model("GetCallerIdentity").input_shape)
        self.calls += 1
        return {
            "Account": self.account,
            "UserId": self.role_id + ":ephemeral",
            "Arn": "arn:aws:sts::" + self.account + ":assumed-role/owned-control-plane/ephemeral",
        }


class DDB:
    def __init__(self):
        self.items = {}
        self.lock = threading.Lock()
        self.calls = []

    def _shape(self, operation, kwargs):
        validate_parameters(kwargs, MODELS["dynamodb"].operation_model(operation).input_shape)
        assert kwargs["TableName"] == "owned-training-jobs"
        self.calls.append((operation, copy.deepcopy(kwargs)))

    def put_item(self, **kwargs):
        self._shape("PutItem", kwargs)
        assert kwargs["ConditionExpression"] == "attribute_not_exists(job_name)"
        name = kwargs["Item"]["job_name"]["S"]
        with self.lock:
            if name in self.items:
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
            self.items[name] = copy.deepcopy(kwargs["Item"])
        return {}

    def get_item(self, **kwargs):
        self._shape("GetItem", kwargs)
        assert kwargs["ConsistentRead"] is True
        with self.lock:
            value = self.items.get(kwargs["Key"]["job_name"]["S"])
            return {"Item": copy.deepcopy(value)} if value else {}

    def update_item(self, **kwargs):
        self._shape("UpdateItem", kwargs)
        assert (
            kwargs["ConditionExpression"]
            == "operation_digest = :digest AND record_revision = :revision AND record_state = :old AND private_record = :prior_body"
        )
        assert (
            kwargs["UpdateExpression"]
            == "SET record_revision = :next, record_state = :new, private_record = :body, #status = :status"
        )
        assert kwargs["ExpressionAttributeNames"] == {"#status": "status"}
        v = kwargs["ExpressionAttributeValues"]
        with self.lock:
            x = self.items.get(kwargs["Key"]["job_name"]["S"])
            if (
                not x
                or any(
                    x[k] != v[q]
                    for k, q in (
                        ("operation_digest", ":digest"),
                        ("record_revision", ":revision"),
                        ("record_state", ":old"),
                    )
                )
                or x["private_record"] != v[":prior_body"]
            ):
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
            for key, field in (
                ("record_revision", ":next"),
                ("record_state", ":new"),
                ("private_record", ":body"),
                ("status", ":status"),
            ):
                x[key] = copy.deepcopy(v[field])
        return {}

    def delete_item(self, **kwargs):
        pytest.fail("private reservation rows must never be deleted")


class SM:
    def __init__(self):
        self.desc = None
        self.tags = None
        self.reads = 0

    def create_training_job(self, **kwargs):
        pytest.fail("durable observer must never call CreateTrainingJob")

    def describe_training_job(self, **kwargs):
        validate_parameters(kwargs, MODELS["sagemaker"].operation_model("DescribeTrainingJob").input_shape)
        self.reads += 1
        if self.desc is None:
            raise ClientError({"Error": {"Code": "ValidationException"}}, "DescribeTrainingJob")
        assert kwargs["TrainingJobName"] == self.desc["TrainingJobName"]
        return copy.deepcopy(self.desc)

    def list_tags(self, **kwargs):
        validate_parameters(kwargs, MODELS["sagemaker"].operation_model("ListTags").input_shape)
        return {"Tags": copy.deepcopy(self.tags)}


def state():
    return CampaignStateCatalog(
        account_id=ACCOUNT,
        region="us-west-2",
        deployment_id=catalog().deployment_id,
        deployment_prefix="owned",
        jobs_table="owned-training-jobs",
        execution_role_arn=CONTROL_ROLE,
        execution_role_id=ROLE_ID,
    )


def fixture():
    ddb, sts, sm = DDB(), STS(), SM()
    store = DynamoCampaignStateBackend(state(), dynamodb_client=ddb, sts_client=sts, sagemaker_client=sm)
    binding = store.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=SESSION)
    _b, _native, _s3, inputs = backend()
    intent = build_sft_intent(binding, catalog(), inputs)
    return store, binding, intent, ddb, sts, sm


def started():
    s, b, i, d, t, sm = fixture()
    r = s.reserve_original_job(b, i)
    return s, b, i, d, t, sm, s.mark_create_attempted(b, r)


def native(sm, r, intent):
    sm.desc = {
        "TrainingJobName": r.native_job_name,
        "TrainingJobArn": r.native_job_arn,
        "TrainingJobStatus": "InProgress",
        "RoleArn": intent.catalog.execution_role_arn,
        "AlgorithmSpecification": {"TrainingImage": intent.catalog.image_uri},
    }
    sm.tags = [
        {"Key": "amlas:" + key, "Value": value}
        for key, value in (
            ("deployment-id", intent.catalog.deployment_id),
            ("deployment-prefix", "owned"),
            ("runtime-session-id", r.binding.runtime_session_id),
        )
    ]


def test_constructor_is_not_a_user_json_binding_factory():
    with pytest.raises((TypeError, CampaignStateError)):
        TrustedExecutionBinding(
            **dataclasses.asdict(state()), campaign_id=CAMPAIGN, runtime_session_id=SESSION
        )


@pytest.mark.parametrize(
    "change",
    [
        {"account_id": "000000000000"},
        {"jobs_table": "amlas-training-jobs"},
        {"execution_role_id": "invalid"},
        {"deployment_prefix": "../bad"},
    ],
)
def test_invalid_catalog_has_no_sdk_calls(change):
    d, t = DDB(), STS()
    with pytest.raises(CampaignStateError):
        DynamoCampaignStateBackend(dataclasses.replace(state(), **change), dynamodb_client=d, sts_client=t)
    assert not d.calls and t.calls == 0


@pytest.mark.parametrize("field,value", [("account", "000000000000"), ("role_id", "AROA" + "Y" * 17)])
def test_native_scope_differs_before_reservation(field, value):
    s, b, i, d, t, _ = fixture()
    setattr(t, field, value)
    with pytest.raises(CampaignStateError):
        s.reserve_original_job(b, i)
    assert not d.calls


def test_concurrent_reserve_one_original_identity_with_real_conditional_requests():
    s, b, i, d, _t, _sm = fixture()
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda _: s.reserve_original_job(b, i), range(2)))
    assert len(d.items) == 1
    assert result[0].native_job_name == result[1].native_job_name
    assert result[0].operation_digest == operation_digest(i)
    assert sum(x.reused for x in result) == 1
    assert result[0].disposition == "RESERVED"


def test_complete_operation_digest_includes_scope_recipe_and_all_private_inputs():
    s, b, i, d, _t, _sm = fixture()
    full = build_sft_intent(b, i.catalog, i.inputs, recipe_id="full_v1")
    assert operation_digest(i) != operation_digest(full)
    changed = dataclasses.replace(i.catalog, source_commit="c" * 40)
    other = build_sft_intent(b, changed, i.inputs)
    assert operation_digest(i) != operation_digest(other)
    r = s.reserve_original_job(b, i)
    assert r.native_job_name.startswith("owned-sft-") and len(r.native_job_name) <= 63
    assert d.items[r.native_job_name]["runtime_session_id"] == {"S": SESSION}


def test_attempt_marker_is_conditional_and_losing_marker_never_dispatches():
    s, b, i, d, _t, _sm = fixture()
    r = s.reserve_original_job(b, i)
    attempted = s.mark_create_attempted(b, r)
    assert attempted.disposition == "CREATE_ATTEMPTED" and attempted.revision == 1
    with pytest.raises(CampaignStateError):
        s.mark_create_attempted(b, r)
    assert len(d.items) == 1


@pytest.mark.parametrize("disposition", ["AMBIGUOUS", "REJECTED"])
def test_submission_dispositions_survive_new_backend_instance_and_are_not_deleted(disposition):
    s, b, i, d, t, sm, r = started()
    x = s.record_submission_outcome(b, r, disposition=disposition, error_code="TransportError")
    restarted = DynamoCampaignStateBackend(state(), dynamodb_client=d, sts_client=t, sagemaker_client=sm)
    bound = restarted.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=SESSION)
    restored = restarted.load_original(bound, r.native_job_name)
    assert restored.disposition == disposition and restored.revision == x.revision
    assert type(restored.intent) is type(i) and type(restored.intent.inputs) is type(i.inputs)
    assert restored.native_job_name == r.native_job_name
    assert restarted.reserve_original_job(bound, i).native_job_name == r.native_job_name


def test_accepted_requires_exact_original_native_identity():
    s, b, _i, d, _t, _sm, r = started()
    with pytest.raises(CampaignStateError):
        s.record_submission_outcome(
            b, r, disposition="ACCEPTED", native_job_arn=r.native_job_arn + "-foreign"
        )
    accepted = s.record_submission_outcome(b, r, disposition="ACCEPTED", native_job_arn=r.native_job_arn)
    assert accepted.disposition == "ACCEPTED" and accepted.revision == 2 and len(d.items) == 1


@pytest.mark.parametrize(
    "change",
    ["digest", "binding", "private_type", "extra_grant", "catalog", "row_revision", "duplicate_json"],
)
def test_authenticated_load_refuses_private_record_tamper(change):
    s, b, i, d, _t, _sm = fixture()
    r = s.reserve_original_job(b, i)
    row = d.items[r.native_job_name]
    payload = json.loads(row["private_record"]["B"])
    if change == "digest":
        payload["operation_digest"] = "f" * 64
    elif change == "binding":
        payload["binding"]["runtime_session_id"] = "foreign-session"
    elif change == "private_type":
        payload["intent"]["inputs"]["type"] = "ApprovalPolicy"
    elif change == "extra_grant":
        payload["policy"] = {"mode": "auto"}
    elif change == "catalog":
        payload["intent"]["catalog"]["fields"]["source_commit"] = "f" * 40
    elif change == "row_revision":
        row["record_revision"] = {"N": "900"}
    else:
        row["private_record"]["B"] = b'{"schema_version":1,"schema_version":1}'
        with pytest.raises(CampaignStateError):
            s.load_original(b, r.native_job_name)
        return
    if change != "row_revision":
        row["private_record"]["B"] = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(CampaignStateError):
        s.load_original(b, r.native_job_name)


@pytest.mark.parametrize(
    "selector,value",
    [("campaign_id", "12345678-1234-4234-9234-00000000aa02"), ("runtime_session_id", "foreign-session")],
)
def test_foreign_campaign_or_session_cannot_restore_original(selector, value):
    s, b, i, _d, _t, _sm = fixture()
    r = s.reserve_original_job(b, i)
    values = {"campaign_id": CAMPAIGN, "runtime_session_id": SESSION, selector: value}
    foreign = s.bind_execution(**values)
    with pytest.raises(CampaignStateError):
        s.load_original(foreign, r.native_job_name)


def test_policy_never_serialized_and_model_authored_intent_dict_refused():
    s, b, i, d, _t, _sm = fixture()
    with pytest.raises(CampaignStateError):
        s.reserve_original_job(b, {"catalog": i.catalog, "policy": {"mode": "auto"}})
    r = s.reserve_original_job(b, i)
    raw = d.items[r.native_job_name]["private_record"]["B"]
    for name in (b"ApprovalPolicy", b"allowlist", b"max_auto_cost", b"responder"):
        assert name not in raw


def test_observer_survives_ambiguous_acceptance_and_performs_zero_creates_or_writes():
    s, b, i, d, _t, sm, r = started()
    r = s.record_submission_outcome(b, r, disposition="AMBIGUOUS", error_code="ReadTimeoutError")
    native(sm, r, i)
    before = len([c for c in d.calls if c[0] in ("PutItem", "UpdateItem")])
    for _ in range(2):
        answer = s.observe_original(b, r.native_job_name)
        assert answer["observed"] is True and answer["native_status"] == "InProgress"
        assert answer["trained"] is False and answer["artifacts_verified"] is False
    assert sm.reads == 2 and len([c for c in d.calls if c[0] in ("PutItem", "UpdateItem")]) == before


def test_attempted_without_observed_job_stays_unresolved_and_never_creates():
    s, b, _i, _d, _t, sm, r = started()
    answer = s.observe_original(b, r.native_job_name)
    assert answer["observed"] is False and answer["status"] == "unresolved"
    assert sm.reads == 1


@pytest.mark.parametrize("change", ["role", "image", "tags"])
def test_native_observer_refuses_foreign_readback(change):
    s, b, i, _d, _t, sm, r = started()
    native(sm, r, i)
    if change == "role":
        sm.desc["RoleArn"] += "-foreign"
    elif change == "image":
        sm.desc["AlgorithmSpecification"]["TrainingImage"] += "-foreign"
    else:
        sm.tags[0]["Value"] = "foreign"
    answer = s.observe_original(b, r.native_job_name)
    assert answer["observed"] is False and answer["status"] == "unresolved"


def test_conditional_update_failure_does_not_replace_or_drop_original_row():
    s, b, _i, d, _t, _sm, r = started()
    first = s.record_submission_outcome(b, r, disposition="AMBIGUOUS", error_code="TimeoutError")
    with pytest.raises(CampaignStateError):
        s.record_submission_outcome(b, r, disposition="ACCEPTED", native_job_arn=r.native_job_arn)
    assert s.load_original(b, r.native_job_name).revision == first.revision and len(d.items) == 1


@pytest.mark.parametrize("change", ["recipe", "operation", "deployment", "source_session"])
def test_replaced_private_intent_is_revalidated_before_reservation(change):
    s, b, i, d, _t, _sm = fixture()
    if change == "recipe":
        forged = dataclasses.replace(i, recipe_id="caller-selected")
    elif change == "operation":
        forged = dataclasses.replace(i, operation="create-arbitrary")
    elif change == "deployment":
        forged = dataclasses.replace(
            i, catalog=dataclasses.replace(i.catalog, deployment_id="12345678-1234-4234-9234-00000000aa02")
        )
    else:
        forged = dataclasses.replace(
            i,
            inputs=dataclasses.replace(
                i.inputs, model=dataclasses.replace(i.inputs.model, runtime_session_id="foreign-session")
            ),
        )
    with pytest.raises(CampaignStateError):
        s.reserve_original_job(b, forged)
    assert not d.calls


def test_foreign_deployment_cannot_restore_original_from_same_selected_table():
    s, b, i, d, t, sm = fixture()
    r = s.reserve_original_job(b, i)
    foreign = DynamoCampaignStateBackend(
        dataclasses.replace(state(), deployment_id="12345678-1234-4234-9234-00000000aa02"),
        dynamodb_client=d,
        sts_client=t,
        sagemaker_client=sm,
    )
    bound = foreign.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=SESSION)
    with pytest.raises(CampaignStateError):
        foreign.load_original(bound, r.native_job_name)
    assert len(d.items) == 1


def test_load_refuses_valid_record_substituted_under_another_original_job_key():
    s, b, i, d, _t, sm = fixture()
    first = s.reserve_original_job(b, i)
    second = s.reserve_original_job(b, build_sft_intent(b, i.catalog, i.inputs, recipe_id="full_v1"))
    transplanted = copy.deepcopy(d.items[second.native_job_name])
    transplanted["job_name"] = {"S": first.native_job_name}
    d.items[first.native_job_name] = transplanted
    before = len([c for c in d.calls if c[0] in ("PutItem", "UpdateItem")])
    with pytest.raises(CampaignStateError):
        s.load_original(b, first.native_job_name)
    with pytest.raises(CampaignStateError):
        s.observe_original(b, first.native_job_name)
    assert sm.reads == 0
    assert len([c for c in d.calls if c[0] in ("PutItem", "UpdateItem")]) == before
