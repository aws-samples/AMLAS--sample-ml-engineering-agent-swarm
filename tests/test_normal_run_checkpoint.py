"""Normal checkpoint enrollment and closed routes; actual SDK, inert AWS ports.

This first RED covers normal wiring, not original SFT recovery or native authority.
Producer-boundary aliases, lost ACK, lease fencing and cold original-job recovery
remain required next slices. No private campaign context or seeded job is used.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import re
import threading
import uuid
from types import SimpleNamespace

import boto3
import model_acquisition_fixtures as model_fixture
import pytest
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters
from strands.models import Model
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401
from test_sdk_pending_handoff import profile_work

from amlas import app as runtime
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy

ACCOUNT = model_fixture.ACCOUNT
REGION = "us-west-2"
PREFIX = "fixture-lab"
TABLE = PREFIX + "-training-jobs"
RUNTIME_ROLE = "arn:aws:iam::" + ACCOUNT + ":role/owned-normal-runtime"
ROLE_ID = "AROA" + "Q" * 17
SESSION = "ordinary-owned-sdk-transport-session-0123456789"
PROMPT = "Research one licensed coding model and record its admitted plan."
MODELS = {name: Session().get_service_model(name) for name in ("sts", "iam", "dynamodb", "s3")}
FORBIDDEN_KEYS = {
    "approval_policy",
    "allowlist",
    "SecretAccessKey",
    "AccessKeyId",
    "SessionToken",
}


def checked(service, operation, kwargs):
    """Use installed SDK wire shapes; this is not native ownership qualification."""
    validate_parameters(kwargs, MODELS[service].operation_model(operation).input_shape)


def decoded(value):
    """Inspect data-only native AttributeValues/JSON; never deserialize objects."""
    if type(value) is dict:
        if set(value) == {"S"}:
            return decoded(value["S"])
        if set(value) == {"B"}:
            return decoded(bytes(value["B"]).decode("utf-8"))
        if set(value) == {"M"}:
            return {key: decoded(item) for key, item in value["M"].items()}
        if set(value) == {"L"}:
            return [decoded(item) for item in value["L"]]
        return {key: decoded(item) for key, item in value.items()}
    if type(value) is str and value[:1] in ("{", "["):
        return decoded(json.loads(value))
    if type(value) is list:
        return [decoded(item) for item in value]
    return value


def values_for(value, key):
    if type(value) is dict:
        found = [value[key]] if key in value else []
        return found + [entry for item in value.values() for entry in values_for(item, key)]
    if type(value) is list:
        return [entry for item in value for entry in values_for(item, key)]
    return []


def assert_safe_data(value):
    if type(value) is dict:
        assert not FORBIDDEN_KEYS.intersection(value), "checkpoint serialized live policy/credentials"
        for item in value.values():
            assert_safe_data(item)
    elif type(value) is list:
        for item in value:
            assert_safe_data(item)


class NativePorts:
    """Small typed native seams for existing config; no producer/factory grants.

    Request oracles: installed STS/IAM/DynamoDB/S3 SDK service models. Conditional
    equality is evaluated under one server lock. Unsupported expressions refuse.
    """

    def __init__(self):
        self.meta = SimpleNamespace(region_name=REGION)
        self.items = {}
        self.objects = {}
        self.calls = []
        self.lock = threading.Lock()
        self.before_model = []

    def get_caller_identity(self, **kwargs):
        checked("sts", "GetCallerIdentity", kwargs)
        self.calls.append(("STS", copy.deepcopy(kwargs)))
        return {
            "Account": ACCOUNT,
            "Arn": "arn:aws:sts::" + ACCOUNT + ":assumed-role/owned-normal-runtime/ordinary",
            "UserId": ROLE_ID + ":ordinary",
        }

    def get_role(self, **kwargs):
        checked("iam", "GetRole", kwargs)
        assert kwargs["RoleName"] == "owned-normal-runtime"
        self.calls.append(("GetRole", copy.deepcopy(kwargs)))
        return {
            "Role": {
                "RoleName": kwargs["RoleName"],
                "Arn": RUNTIME_ROLE,
                "RoleId": ROLE_ID,
                "Path": "/",
                "Tags": self.tags(),
            }
        }

    @staticmethod
    def tags():
        return [
            {"Key": "amlas:deployment-id", "Value": model_fixture.DEPLOYMENT},
            {"Key": "amlas:deployment-prefix", "Value": PREFIX},
            {"Key": "project", "Value": "amlas"},
        ]

    def describe_table(self, **kwargs):
        checked("dynamodb", "DescribeTable", kwargs)
        assert kwargs["TableName"] == TABLE
        self.calls.append(("DescribeTable", copy.deepcopy(kwargs)))
        return {
            "Table": {
                "TableName": TABLE,
                "TableArn": "arn:aws:dynamodb:" + REGION + ":" + ACCOUNT + ":table/" + TABLE,
                "TableStatus": "ACTIVE",
                "KeySchema": [{"AttributeName": "job_name", "KeyType": "HASH"}],
                "AttributeDefinitions": [{"AttributeName": "job_name", "AttributeType": "S"}],
                "TableId": "c4ef4b4a-c954-4a9e-99cc-e036a74353ba",
                "SSEDescription": {"Status": "ENABLED", "SSEType": "KMS"},
            }
        }

    def list_tags_of_resource(self, **kwargs):
        checked("dynamodb", "ListTagsOfResource", kwargs)
        assert kwargs["ResourceArn"] == "arn:aws:dynamodb:" + REGION + ":" + ACCOUNT + ":table/" + TABLE
        self.calls.append(("TableTags", copy.deepcopy(kwargs)))
        return {"Tags": self.tags()}

    def _ddb(self, operation, kwargs):
        checked("dynamodb", operation, kwargs)
        assert kwargs["TableName"] == TABLE
        self.calls.append((operation, copy.deepcopy(kwargs)))

    @staticmethod
    def _condition(item, kwargs):
        aliases = kwargs.get("ExpressionAttributeNames", {})
        values = kwargs.get("ExpressionAttributeValues", {})
        for clause in kwargs["ConditionExpression"].split(" AND "):
            absent = re.fullmatch(r"attribute_not_exists\(([#a-zA-Z0-9_]+)\)", clause)
            if absent:
                field = aliases.get(absent[1], absent[1])
                if item is not None and field in item:
                    return False
                continue
            equal = re.fullmatch(r"([#a-zA-Z0-9_]+) = (:[a-zA-Z0-9_]+)", clause)
            assert equal, "unsupported conditional expression must not be accepted"
            field = aliases.get(equal[1], equal[1])
            if item is None or item.get(field) != values[equal[2]]:
                return False
        return True

    def put_item(self, **kwargs):
        self._ddb("PutItem", kwargs)
        item = kwargs["Item"]
        assert set(item["job_name"]) == {"S"} and kwargs.get("ConditionExpression")
        assert_safe_data(decoded(item))
        with self.lock:
            name = item["job_name"]["S"]
            if not self._condition(self.items.get(name), kwargs):
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
            self.items[name] = copy.deepcopy(item)
        return {}

    def get_item(self, **kwargs):
        self._ddb("GetItem", kwargs)
        assert set(kwargs["Key"]) == {"job_name"} and kwargs["ConsistentRead"] is True
        with self.lock:
            item = self.items.get(kwargs["Key"]["job_name"]["S"])
            return {"Item": copy.deepcopy(item)} if item is not None else {}

    def update_item(self, **kwargs):
        self._ddb("UpdateItem", kwargs)
        assert kwargs.get("ConditionExpression") and kwargs["UpdateExpression"].startswith("SET ")
        with self.lock:
            item = self.items.get(kwargs["Key"]["job_name"]["S"])
            if not self._condition(item, kwargs):
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
            for assignment in kwargs["UpdateExpression"][4:].split(", "):
                match = re.fullmatch(r"([#a-zA-Z0-9_]+) = (:[a-zA-Z0-9_]+)", assignment)
                assert match, "unsupported update expression must not be accepted"
                field = kwargs.get("ExpressionAttributeNames", {}).get(match[1], match[1])
                item[field] = copy.deepcopy(kwargs["ExpressionAttributeValues"][match[2]])
            assert_safe_data(decoded(item))
        return {}

    def get_bucket_versioning(self, **kwargs):
        checked("s3", "GetBucketVersioning", kwargs)
        assert kwargs["Bucket"] == model_fixture.BUCKET
        return {"Status": "Enabled"}

    def put_object(self, **kwargs):
        checked("s3", "PutObject", kwargs)
        assert kwargs["Bucket"] == model_fixture.BUCKET
        body = kwargs["Body"]
        body = body.read() if hasattr(body, "read") else bytes(body)
        assert_safe_data(decoded(body.decode("utf-8")))
        version = "ordinary-version-" + str(len(self.objects) + 1)
        self.objects[(kwargs["Key"], version)] = body
        self.calls.append(("PutObject", {"Key": kwargs["Key"], "VersionId": version}))
        return {"VersionId": version}

    def get_object(self, **kwargs):
        checked("s3", "GetObject", kwargs)
        assert kwargs["Bucket"] == model_fixture.BUCKET and kwargs.get("VersionId")
        body = self.objects[(kwargs["Key"], kwargs["VersionId"])]
        return {"Body": io.BytesIO(body), "VersionId": kwargs["VersionId"], "ContentLength": len(body)}


class AdmittedModel(Model):
    def __init__(self, ports):
        self.ports = ports
        self.calls = 0

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "neutral-normal-checkpoint-fixture"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("normal installed SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.ports.before_model.append(copy.deepcopy(self.ports.items))
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "Admitted plan remains descriptive."},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}


@pytest.fixture
def normal_work(monkeypatch, tmp_path):
    ports = NativePorts()
    model = AdmittedModel(ports)
    work = profile_work(monkeypatch, model=model)
    monkeypatch.chdir(tmp_path)
    # Explicit source-owned binding is a proposed narrow deployment prerequisite.
    # It is not inferred from an STS ARN or supplied by the public request.
    for key, value in {
        "AMLAS_AWS_ACCOUNT_ID": ACCOUNT,
        "AMLAS_AWS_REGION": REGION,
        "AMLAS_JOBS_TABLE": TABLE,
        "AMLAS_SESSION_BUCKET": model_fixture.BUCKET,
        "AMLAS_RUNTIME_EXEC_ROLE_ARN": RUNTIME_ROLE,
        "AMLAS_RUNTIME_EXEC_ROLE_ID": ROLE_ID,
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","qwen_sft_flow"]')

    def client(service, **kwargs):
        assert kwargs.get("region_name", REGION) == REGION
        if service == "bedrock-runtime":
            return work.guard
        assert service in {"sts", "iam", "dynamodb", "s3"}, "unexpected live SDK boundary"
        return ports

    class NativeSession:
        def __init__(self, *args, **kwargs):
            pass

        def client(self, service, **kwargs):
            return client(service, **kwargs)

    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setattr(boto3, "Session", NativeSession)
    monkeypatch.setattr(boto3.session, "Session", NativeSession)
    request_token = BedrockAgentCoreContext._request_id.set(None)
    session_token = BedrockAgentCoreContext._session_id.set(None)
    BedrockAgentCoreContext.set_request_context("ordinary-first-request", SESSION)
    try:
        yield work, ports, model
    finally:
        BedrockAgentCoreContext._request_id.reset(request_token)
        BedrockAgentCoreContext._session_id.reset(session_token)


def collect(payload):
    async def read():
        return [event async for event in runtime.agent_invocation(payload)]

    return asyncio.run(read())


def closed_route(payload):
    try:
        events = collect(payload)
    except ValueError:
        pytest.fail("normal checkpoint action reached legacy fresh-start dispatch")
    if payload.get("action") == "resume_prompt_run" and events[:1] == [
        {"event": "resume_state", "state": "restoring", "run_id": payload["run_id"]}
    ]:
        # The resume action answers its state before any restore work; the closed projection follows.
        events = events[1:]
    assert len(events) == 1, "checkpoint action must return one closed projection"
    return events[0]


def unavailable(run_id):
    return {"event": "prompt_run_status", "status": "unavailable", "run_id": run_id, "found": False}


def test_normal_admitted_prompt_conditionally_enrolls_owned_run_before_model(normal_work):
    work, ports, model = normal_work
    events = collect({"prompt": PROMPT})
    run_id = next(event["run_id"] for event in events if event["event"] == "run_started")
    assert re.fullmatch(r"[0-9a-f]{32}", run_id)
    assert work.native.requests == []
    terminal = events[-1]
    assert terminal["event"] == "run_completed" and terminal["status"] == "response_received"
    assert terminal["response"]["message"] == "Admitted plan remains descriptive."
    policy = work.engines[0].tool_backend.approval_policy
    assert type(policy) is ApprovalPolicy and policy.mode is ApprovalMode.ALLOWLIST
    assert policy.allowlist == ["run_processing_job", "qwen_sft_flow"] and policy.interactive is False
    assert terminal["lifecycle_verified"] is False and terminal["quality_success"] is False
    key = "PROMPT_RUN#" + run_id
    assert key in ports.items, "normal admitted path did not create its conditional durable companion"
    assert model.calls == 1 and key in ports.before_model[0]
    value = decoded(ports.items[key])
    assert run_id in values_for(value, "logical_run_id")
    correlation = work.engines[0].tool_backend._acquisition_store.runtime_session_id
    assert str(uuid.UUID(correlation)) == correlation and correlation != run_id
    assert correlation in values_for(value, "private_store_correlation")
    assert any(name == "STS" for name, _args in ports.calls)
    assert work.seen_prompts[0][0] == PROMPT
    assert work.guard.calls and all(call["source"] == "INPUT" for call in work.guard.calls)
    assert_safe_data(value)
    for body in ports.objects.values():
        assert_safe_data(decoded(body.decode("utf-8")))


@pytest.mark.parametrize("action", ["get_prompt_run_status", "resume_prompt_run"])
def test_missing_normal_run_is_closed_without_engine_or_model(normal_work, action):
    work, ports, model = normal_work
    run_id = "a" * 32
    expected = unavailable(run_id)
    if action == "resume_prompt_run":
        # No such run row: the restore ends as a closed failure, never as an unavailable success.
        expected = {
            "event": "resume_state",
            "state": "failed",
            "run_id": run_id,
            "diagnostic": {"restore_stage": "run_prepare", "exception_class": "ValueError"},
        }
    assert closed_route({"action": action, "run_id": run_id}) == expected
    assert work.engines == [] and model.calls == 0 and work.native.requests == []
    assert not [name for name, _args in ports.calls if name in {"PutItem", "UpdateItem", "PutObject"}]
    assert work.owned_app.get_async_task_info()["active_count"] == 0


def test_run_status_invalid_fields_and_missing_sdk_context_refuse_before_work(normal_work):
    work, ports, model = normal_work
    for payload in (
        {"action": "get_prompt_run_status", "run_id": "ordinary-invalid"},
        {"action": "get_prompt_run_status", "run_id": "b" * 32, "policy": "ordinary"},
        {"action": "get_prompt_run_status", "run_id": "b" * 32, "job_arn": "ordinary"},
    ):
        assert closed_route(payload) == {"event": "error", "error_code": "invalid_run_status_request"}
    BedrockAgentCoreContext.set_request_context("ordinary-no-session", None)
    assert closed_route({"action": "get_prompt_run_status", "run_id": "b" * 32}) == {
        "event": "error",
        "error_code": "run_session_unavailable",
    }
    assert work.engines == [] and model.calls == 0 and ports.calls == []


def test_denied_prompt_never_persists_rejected_content_or_dispatches_model(normal_work):
    work, ports, model = normal_work
    rejected = "ordinary withheld input that must stay outside checkpoints"
    work.guard.deny = lambda text: rejected in text
    events = collect({"prompt": rejected})
    assert model.calls == 0 and work.native.requests == []
    assert work.guard.calls and work.guard.calls[0]["source"] == "INPUT"
    assert work.guard.calls[0]["content"][0]["text"]["text"] == rejected
    terminal = events[-1]
    assert terminal["status"] == "response_received"
    assert terminal["response"]["action"] == "input_admission_held"
    assert terminal["response"]["data"]["input_admission"]["status"] == "held"
    assert len(work.engines) == 1 and work.agents == []
    stored = json.dumps(decoded(list(ports.items.values())), sort_keys=True)
    assert rejected not in stored
    assert all(rejected.encode() not in body for body in ports.objects.values())
    assert events[-1]["lifecycle_verified"] is False and events[-1]["quality_success"] is False
    assert_safe_data(decoded(list(ports.items.values())))


@pytest.mark.parametrize("missing", ["AMLAS_RUNTIME_EXEC_ROLE_ID", "AMLAS_JOBS_TABLE"])
def test_selected_checkpoint_missing_binding_refuses_without_effects(normal_work, monkeypatch, missing):
    work, ports, model = normal_work
    monkeypatch.delenv(missing)
    events = collect({"prompt": PROMPT})
    terminal = events[-1]
    assert terminal["event"] == "run_completed" and terminal["status"] == "failed"
    assert terminal["worker_finished"] is True
    assert model.calls == 0 and work.native.requests == []
    assert ports.items == {} and ports.objects == {}
    assert not [name for name, _args in ports.calls if name in {"PutItem", "UpdateItem", "PutObject"}]
    assert work.owned_app.get_async_task_info()["active_count"] == 0


def test_absent_durable_config_keeps_legacy_response_honest(normal_work, monkeypatch):
    work, ports, model = normal_work
    for name in ("AMLAS_RUNTIME_EXEC_ROLE_ARN", "AMLAS_RUNTIME_EXEC_ROLE_ID"):
        monkeypatch.delenv(name)
    events = collect({"prompt": PROMPT})
    terminal = events[-1]
    assert terminal["event"] == "run_completed" and terminal["status"] == "response_received"
    assert terminal["response"]["message"] == "Admitted plan remains descriptive."
    assert model.calls == 1 and len(work.engines) == len(work.agents) == 1
    assert work.seen_prompts[0][0] == PROMPT and work.guard.calls
    assert terminal["durable_resume"] is False
    assert events[-1]["state_scope"] == "single_invocation"
    assert events[-1]["lifecycle_verified"] is False and events[-1]["quality_success"] is False
    assert ports.items == {} and ports.objects == {} and work.native.requests == []
