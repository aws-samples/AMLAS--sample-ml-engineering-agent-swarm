"""Normal callback routing DATA over the actual qualified callback path.

Fresh declared map/SDK-shaped ports prove only exact transport selection. The
existing explicit callback scope supplies inert route-observation DATA, not
native IAM, producer/current-context/admission, TASK readiness, cold hydration or
liveness evidence. The receiver must requalify/fence; delivery can be repeated or
reordered and never grants a retry of unknown work. Legacy behavior is separate.
"""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters
from test_campaign_callback import (
    ACCOUNT,
    DEPLOYMENT,
    REGION,
    RUNTIME,
    TABLE,
    AgentCore,
    QualifiedRoute,
    event,
)

from amlas.async_callback import lambda_handler as callback_source

RUN_ID = "1" * 32
TRANSPORT_SESSION = "declared-original-sdk-transport-session-0123456789"
TABLE_ID = "12345678-1234-4234-9234-00000000dd31"
RECORD_KIND = "normal_prompt_wake_job_v1"
NAMESPACE = "PROMPT_WAKE_JOB#"


@pytest.fixture(autouse=True)
def selected_callback_env(monkeypatch):
    for name, value in {
        "AMLAS_CAMPAIGN_CALLBACK_MODE": "1",
        "AMLAS_AWS_ACCOUNT_ID": ACCOUNT,
        "AWS_REGION": REGION,
        "AMLAS_DEPLOYMENT_ID": DEPLOYMENT,
        "JOBS_TABLE": TABLE,
        "AGENTCORE_RUNTIME_ARN": RUNTIME,
        "AGENTCORE_RUNTIME_QUALIFIER": "NORMAL",
        "AGENTCORE_RUNTIME_VERSION": "3",
        "AMLAS_CALLBACK_ROUTE_SHA256": hashlib.sha256(b"declared selected normal callback route").hexdigest(),
    }.items():
        monkeypatch.setenv(name, value)


def _normal_item(value, *, conflicting=False):
    detail = value["detail"]
    body = {
        "schema_version": 1,
        "record_kind": RECORD_KIND,
        "logical_run_id": RUN_ID,
        "transport_session_id": TRANSPORT_SESSION,
        "account_id": ACCOUNT,
        "region": REGION,
        "deployment_id": DEPLOYMENT,
        "table_id": TABLE_ID,
        "kind": "Training",
        "job_name": detail["TrainingJobName"],
        "job_arn": detail["TrainingJobArn"],
        "operation_digest": "d" * 64,
    }
    if conflicting:
        body["job_arn"] += "-other"
    # Canonical SHA is integrity DATA, never original or continuation authority.
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    assert len(raw.encode()) <= 8192
    return {
        "job_name": {"S": NAMESPACE + body["kind"] + "#" + body["job_name"]},
        "record": {"S": raw},
        "record_sha256": {"S": hashlib.sha256(raw.encode()).hexdigest()},
    }


class DeclaredDDB:
    def __init__(self, item, *, unresolved=False):
        self.item, self.unresolved = item, unresolved
        self.calls, self.descriptions = [], []
        self.hints, self.writes, self.hint_reads = {}, [], []
        self.meta = SimpleNamespace(
            config=SimpleNamespace(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10)
        )

    def get_item(self, **kwargs):
        validate_parameters(
            kwargs, Session().get_service_model("dynamodb").operation_model("GetItem").input_shape
        )
        assert set(kwargs) == {"TableName", "Key", "ConsistentRead"}
        assert kwargs["TableName"] == TABLE and kwargs["ConsistentRead"] is True
        self.calls.append(copy.deepcopy(kwargs))
        if self.unresolved:
            raise TimeoutError("Declared routing lookup is unresolved.")
        hint_key = kwargs["Key"]["job_name"]["S"]
        if hint_key in self.hints:
            assert self.item is not None
            self.hint_reads.append(copy.deepcopy(kwargs))
            return {"Item": copy.deepcopy(self.hints[hint_key])}
        if self.item is None or kwargs["Key"] != {"job_name": self.item["job_name"]}:
            return {}
        return {"Item": copy.deepcopy(self.item)}

    def describe_table(self, **kwargs):
        validate_parameters(
            kwargs, Session().get_service_model("dynamodb").operation_model("DescribeTable").input_shape
        )
        assert kwargs == {"TableName": TABLE}
        self.descriptions.append(copy.deepcopy(kwargs))
        return {
            "Table": {
                "TableName": TABLE,
                "TableArn": f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{TABLE}",
                "TableId": TABLE_ID,
                "TableStatus": "ACTIVE",
                "KeySchema": [{"AttributeName": "job_name", "KeyType": "HASH"}],
                "AttributeDefinitions": [{"AttributeName": "job_name", "AttributeType": "S"}],
            }
        }

    def put_item(self, **kwargs):
        validate_parameters(
            kwargs,
            Session()
            .get_service_model("dynamodb")
            .operation_model(
                "PutItem",
            )
            .input_shape,
        )
        assert self.item is not None, "confirmed absent normal route remains read-only"
        assert self.item == _normal_item(event("Training")), (
            "only the exact declared valid present normal route permits hint retention"
        )
        assert kwargs["TableName"] == TABLE
        assert set(kwargs) == {"TableName", "Item", "ConditionExpression"}
        assert kwargs["ConditionExpression"] == "attribute_not_exists(job_name)"
        route = json.loads(self.item["record"]["S"])
        body = {key: value for key, value in route.items() if key != "transport_session_id"}
        body["record_kind"] = "normal_prompt_terminal_hint_v1"
        body["status"] = "Completed"
        body["normal_route_sha256"] = self.item["record_sha256"]["S"]
        raw = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        key = "PROMPT_TERMINAL_HINT#Training#" + route["job_name"]
        expected = {
            "job_name": {"S": key},
            "record": {"S": raw},
            "record_sha256": {"S": hashlib.sha256(raw.encode()).hexdigest()},
        }
        assert len(raw.encode()) <= 8192
        assert kwargs["Item"] == expected
        self.writes.append(copy.deepcopy(kwargs))
        if key in self.hints:
            raise ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException"}},
                "PutItem",
            )
        self.hints[key] = copy.deepcopy(expected)
        return {}

    def __getattr__(self, name):
        if name in ("scan", "query", "put_item", "update_item", "delete_item"):
            return lambda **kwargs: pytest.fail("normal callback routing is exact read-only lookup")
        raise AttributeError(name)


def _declared_route(*, failure=None):
    value = event("Training")
    item = None if failure == "missing" else _normal_item(value, conflicting=failure == "conflicting")
    ddb = DeclaredDDB(item, unresolved=failure == "unresolved")
    runtime = AgentCore(resume_state="continuing")
    route = QualifiedRoute(callback_source)
    route.valid = failure != "denied"
    return value, ddb, runtime, route


def test_terminal_normal_mapping_delivers_exact_owned_resume_selector():
    value, ddb, runtime, route = _declared_route()
    with callback_source.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = callback_source.handler(value)
    assert len(runtime.calls) == 1, "terminal normal mapping must select the existing normal resume route"
    sent = runtime.calls[0]
    assert sent["agentRuntimeArn"] == RUNTIME and sent["qualifier"] == "NORMAL"
    assert sent["runtimeSessionId"] == TRANSPORT_SESSION
    assert sent["runtimeSessionId"] != RUN_ID
    assert json.loads(sent["payload"]) == {"action": "resume_prompt_run", "run_id": RUN_ID}
    normal_key = {"job_name": {"S": NAMESPACE + "Training#" + value["detail"]["TrainingJobName"]}}
    assert any(request["Key"] == normal_key for request in ddb.calls)
    assert result["delivery_status"] == "delivered"
    assert result["lifecycle_verified"] is False and result["quality_success"] is False
    assert runtime.body.closed_called


@pytest.mark.parametrize("failure", ["missing", "conflicting", "denied", "unresolved"])
def test_unresolved_or_refused_normal_mapping_cannot_invoke(failure):
    value, ddb, runtime, route = _declared_route(failure=failure)
    with callback_source.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = callback_source.handler(value)
    assert runtime.calls == []
    assert result["statusCode"] in (400, 403, 404, 409, 503)
    assert result["delivery_status"] != "delivered"
    assert result["lifecycle_verified"] is False and result["quality_success"] is False


class DeclaredFallbackDDB(DeclaredDDB):
    """Fresh campaign DATA is a fallback trap, never receiver/native authority."""

    def __init__(self, value, item):
        from test_campaign_callback import hint_item

        super().__init__(item)
        self.fallback = hint_item(value)

    def get_item(self, **kwargs):
        response = super().get_item(**kwargs)
        if response:
            return response
        if kwargs["Key"] == {"job_name": self.fallback["job_name"]}:
            return {"Item": copy.deepcopy(self.fallback)}
        return {}


def _malformed_normal_item(value, mutation):
    item = _normal_item(value)
    body = json.loads(item["record"]["S"])
    if mutation == "raw_hash":
        item["record_sha256"]["S"] = "e" * 64
        return item
    if mutation == "extra_field":
        body["ordinary_extra"] = "declared extra DATA"
    elif mutation == "missing_field":
        del body["operation_digest"]
    elif mutation == "bool_schema":
        body["schema_version"] = True
    elif mutation == "wrong_table":
        body["table_id"] = "12345678-1234-4234-9234-00000000dd32"
    elif mutation == "foreign_account":
        body["account_id"] = "000000000000"
    elif mutation == "foreign_region":
        body["region"] = "us-east-2"
    elif mutation == "foreign_deployment":
        body["deployment_id"] = "12345678-1234-4234-9234-00000000dd32"
    elif mutation == "foreign_kind":
        body["kind"] = "Processing"
    elif mutation == "foreign_name":
        body["job_name"] = "declared-other-native-job"
    elif mutation == "foreign_arn":
        body["job_arn"] += "-other"
    elif mutation == "logical_run_id":
        body["logical_run_id"] = "declared-invalid-run"
    elif mutation == "transport_session_id":
        body["transport_session_id"] = "short"
    elif mutation == "nonfinite":
        body["schema_version"] = float("nan")
    else:
        assert mutation in ("noncanonical", "duplicate_keys")
    if mutation == "noncanonical":
        raw = json.dumps(body, sort_keys=True, indent=2)
    else:
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if mutation == "duplicate_keys":
        raw = raw[:-1] + ',"schema_version":1}'
    item["record"]["S"] = raw
    # Recomputed integrity DATA makes typed/native-scope checks independently
    # necessary; a coherent self-hash cannot grant routing or continuation.
    item["record_sha256"]["S"] = hashlib.sha256(raw.encode()).hexdigest()
    return item


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_hash",
        "noncanonical",
        "duplicate_keys",
        "nonfinite",
        "extra_field",
        "missing_field",
        "bool_schema",
        "wrong_table",
        "foreign_account",
        "foreign_region",
        "foreign_deployment",
        "foreign_kind",
        "foreign_name",
        "foreign_arn",
        "malformed_selectors",
    ],
)
def test_present_malformed_normal_mapping_never_invokes_or_falls_back(mutation):
    variants = (
        ("logical_run_id", "transport_session_id") if mutation == "malformed_selectors" else (mutation,)
    )
    for variant in variants:
        value = event("Training")
        ddb = DeclaredFallbackDDB(value, _malformed_normal_item(value, variant))
        runtime = AgentCore()
        route = QualifiedRoute(callback_source)
        with callback_source.campaign_callback_scope(
            route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
        ):
            result = callback_source.handler(value)
        assert runtime.calls == [], (
            "present malformed normal DATA must not invoke normal or campaign fallback"
        )
        normal_key = {"job_name": {"S": NAMESPACE + "Training#" + value["detail"]["TrainingJobName"]}}
        assert ddb.calls and all(request["Key"] == normal_key for request in ddb.calls)
        assert result["statusCode"] in (400, 403, 404, 409, 503)
        assert result["delivery_status"] != "delivered"
        assert result["lifecycle_verified"] is False and result["quality_success"] is False


def test_present_normal_retains_hint_but_confirmed_absence_keeps_campaign_read_only():
    value, present, runtime, route = _declared_route()
    with callback_source.campaign_callback_scope(
        route,
        dynamodb_client=present,
        runtime_client=runtime,
        clock=lambda: 1000.0,
    ):
        result = callback_source.handler(value)
    assert result["delivery_status"] == "delivered"
    assert len(runtime.calls) == 1
    assert len(present.writes) == len(present.hints) == 1
    assert present.hint_reads
    assert json.loads(runtime.calls[0]["payload"]) == {
        "action": "resume_prompt_run",
        "run_id": RUN_ID,
    }

    absent = DeclaredFallbackDDB(value, None)
    legacy_runtime = AgentCore()
    legacy_route = QualifiedRoute(callback_source)
    with callback_source.campaign_callback_scope(
        legacy_route,
        dynamodb_client=absent,
        runtime_client=legacy_runtime,
        clock=lambda: 1000.0,
    ):
        legacy = callback_source.handler(value)
    assert legacy["delivery_status"] == "delivered"
    assert len(legacy_runtime.calls) == 1
    assert absent.writes == [] and absent.hints == {}
    assert absent.hint_reads == []
    normal_key = {
        "job_name": {"S": NAMESPACE + "Training#" + value["detail"]["TrainingJobName"]},
    }
    assert absent.calls[0]["Key"] == normal_key
    assert absent.calls[1]["Key"] == {"job_name": absent.fallback["job_name"]}
    assert len(absent.calls) == 2
    fallback_body = json.loads(absent.fallback["private_record"]["B"])
    expected_wake = {name: fallback_body[name] for name in callback_source._WAKE_FIELDS if name != "event_id"}
    expected_wake["event_id"] = value["id"]
    assert json.loads(legacy_runtime.calls[0]["payload"]) == expected_wake


def test_non_string_terminal_status_refuses_before_routing_io():
    for status in (["Completed"], {"state": "Completed"}):
        value, ddb, runtime, route = _declared_route()
        value["detail"]["TrainingJobStatus"] = status
        with callback_source.campaign_callback_scope(
            route,
            dynamodb_client=ddb,
            runtime_client=runtime,
            clock=lambda: 1000.0,
        ):
            result = callback_source.handler(value)
        assert result["statusCode"] == 400
        assert result["delivery_status"] != "delivered"
        assert runtime.calls == []
        assert ddb.calls == [] and ddb.descriptions == []
        assert ddb.writes == [] and ddb.hints == {}
        assert result["lifecycle_verified"] is False
        assert result["quality_success"] is False
