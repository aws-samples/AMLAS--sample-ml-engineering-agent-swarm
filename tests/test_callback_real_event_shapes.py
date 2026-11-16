"""Real SageMaker terminal-event shapes through the campaign-mode callback, offline.

The events come from `fixtures/sagemaker_training_state_change_real.json`: seven real training jobs
(Completed, Failed and Stopped; October 2026 and a GPU test run of 2026-11-10) in the documented
"SageMaker Training Job State Change" envelope, with neutral names. The file also holds the delivery
pattern measured on the deployed EventBridge target: two invocations per Completed or Failed job, 6 to
878 ms apart, one per Stopped job. AWS documents that "SageMaker AI may send multiple events to
EventBridge for each state change" and that an asynchronous Lambda can receive the same event more
than once.

The handler is the product's `handler` in campaign mode. DynamoDB and the route qualifier are
in-memory SDK-shaped ports. The runtime is either the closed-state port of `test_campaign_callback`
or the real app (`amlas.app.agent_invocation` under the real BedrockAgentCoreApp task ownership, with
the scripted restore of `test_normal_wake_resume_state`). One process is one microVM: these tests do
not cover two replacement microVMs created for one session by concurrent first calls.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import copy
import datetime
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError
from botocore.session import Session
from botocore.validate import validate_parameters
from test_campaign_callback import (
    ACCOUNT,
    DEPLOYMENT,
    REGION,
    RUNTIME,
    TABLE,
    AgentCore,
    Body,
    QualifiedRoute,
)
from test_normal_wake_resume_state import RUN_ID, SESSION, _settled
from test_normal_wake_resume_state import resume as resume
from test_runtime_busy_ownership import managed_report_context as managed_report_context
from test_runtime_busy_ownership import owned_app as owned_app
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime
from amlas.async_callback import lambda_handler as callback_source

FIXTURE = Path(__file__).parent / "fixtures" / "sagemaker_training_state_change_real.json"
REAL = json.loads(FIXTURE.read_text(encoding="utf-8"))
JOBS = {job["name"]: job for job in REAL["jobs"]}
SFT = "fixture-deploy-sft-ex1-full-01"
TABLE_ID = "12345678-1234-4234-9234-00000000ee01"
STORE = "12345678-1234-4234-9234-00000000ee02"
STREAM_ARN = f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{TABLE}/stream/2026-11-10T00:00:00.000"
OPERATION = hashlib.sha256(b"fixture real-shape sft operation").hexdigest()
ROUTE_SHA = hashlib.sha256(b"fixture real-shape callback route").hexdigest()
MODELS = {name: Session().get_service_model(name) for name in ("dynamodb", "bedrock-agentcore")}
ENVELOPE = {"version", "id", "detail-type", "source", "account", "time", "region", "resources", "detail"}
TRANSPORT = dict(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10)


@pytest.fixture(autouse=True)
def callback_env(monkeypatch):
    for name, value in {
        "AMLAS_CAMPAIGN_CALLBACK_MODE": "1",
        "AMLAS_AWS_ACCOUNT_ID": ACCOUNT,
        "AWS_REGION": REGION,
        "AMLAS_DEPLOYMENT_ID": DEPLOYMENT,
        "JOBS_TABLE": TABLE,
        "AGENTCORE_RUNTIME_ARN": RUNTIME,
        "AGENTCORE_RUNTIME_QUALIFIER": "DEFAULT",
        "AGENTCORE_RUNTIME_VERSION": "3",
        "AMLAS_CALLBACK_ROUTE_SHA256": ROUTE_SHA,
    }.items():
        monkeypatch.setenv(name, value)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def real_event(name=SFT, *, event_id="fixture-real-event-1"):
    """The documented envelope around the real job's detail; resources carry the job ARN."""
    job = JOBS[name]
    detail = copy.deepcopy(job["detail"])
    when = datetime.datetime.fromtimestamp(job["last_modified_ms"] / 1000, datetime.UTC)
    return {
        "version": "0",
        "id": event_id,
        "detail-type": "SageMaker Training Job State Change",
        "source": "aws.sagemaker",
        "account": REAL["account"],
        "time": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "region": REAL["region"],
        "resources": [detail["TrainingJobArn"]],
        "detail": detail,
    }


def _item(key, body, *, ascii_json=False):
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii_json, allow_nan=False)
    return {"job_name": {"S": key}, "record": {"S": raw}, "record_sha256": {"S": _sha(raw)}}


def wake_route(name=SFT):
    """The normal wake row the SFT submit writes before CreateTrainingJob (closed eleven-plus-one fields)."""
    detail = JOBS[name]["detail"]
    return _item(
        "PROMPT_WAKE_JOB#Training#" + name,
        {
            "schema_version": 1,
            "record_kind": "normal_prompt_wake_job_v1",
            "logical_run_id": RUN_ID,
            "transport_session_id": SESSION,
            "account_id": ACCOUNT,
            "region": REGION,
            "deployment_id": DEPLOYMENT,
            "table_id": TABLE_ID,
            "kind": "Training",
            "job_name": name,
            "job_arn": detail["TrainingJobArn"],
            "operation_digest": OPERATION,
        },
    )


def run_row(name=SFT, *, task=True):
    """The PROMPT_RUN row in the SFT stage, before (task=False) and after the TASK checkpoint is saved."""
    body = {
        "schema_version": 1,
        "logical_run_id": RUN_ID,
        "session_sha256": _sha(SESSION),
        "scope": {"account_id": ACCOUNT, "region": REGION, "deployment_id": DEPLOYMENT, "jobs_table": TABLE},
        "table_id": TABLE_ID,
        "private_store_correlation": STORE,
        "admitted_prompt": "Fine-tune the declared fixture model and report the paired result.",
        "history": [],
        "stage": "sft",
        "sft": {
            "native_job_name": name,
            "native_job_arn": JOBS[name]["detail"]["TrainingJobArn"],
            "operation_digest": OPERATION,
        },
    }
    if task:
        body["task_checkpoint_ref"] = {
            "uri": "s3://fixture-bucket/sessions/fixture-session/TASK.json",
            "version_id": "fixture-task-version-1",
            "sha256": "c" * 64,
            "byte_count": 2048,
            "format": "json",
        }
    return _item("PROMPT_RUN#" + RUN_ID, body, ascii_json=True)


def readiness_record(event_id="fixture-task-transition-1"):
    """The stream MODIFY record of the TASK save (the record the mapping retries until it is delivered)."""
    old, new = run_row(task=False), run_row(task=True)
    return {
        "eventID": event_id,
        "eventName": "MODIFY",
        "eventVersion": "1.1",
        "eventSource": "aws:dynamodb",
        "awsRegion": REGION,
        "dynamodb": {
            "ApproximateCreationDateTime": 1762790400.0,
            "Keys": {"job_name": copy.deepcopy(new["job_name"])},
            "OldImage": old,
            "NewImage": new,
            "SequenceNumber": "100000000000000000001",
            "SizeBytes": len(json.dumps(new).encode()),
            "StreamViewType": "NEW_AND_OLD_IMAGES",
        },
        "eventSourceARN": STREAM_ARN,
    }


class Table:
    """Thread-safe DynamoDB port: exact consistent Get, conditional Put, DescribeTable with the stream."""

    def __init__(self, *items, put_barrier=None):
        self.items = {item["job_name"]["S"]: copy.deepcopy(item) for item in items}
        self.lock, self.calls, self.put_barrier = threading.Lock(), [], put_barrier
        self.meta = SimpleNamespace(config=SimpleNamespace(**TRANSPORT))

    def _call(self, operation, kwargs):
        validate_parameters(kwargs, MODELS["dynamodb"].operation_model(operation).input_shape)
        assert kwargs["TableName"] == TABLE
        with self.lock:
            self.calls.append((operation, copy.deepcopy(kwargs)))

    def get_item(self, **kwargs):
        self._call("GetItem", kwargs)
        assert kwargs["ConsistentRead"] is True and set(kwargs["Key"]) == {"job_name"}
        with self.lock:
            item = self.items.get(kwargs["Key"]["job_name"]["S"])
            return {"Item": copy.deepcopy(item)} if item is not None else {}

    def put_item(self, **kwargs):
        self._call("PutItem", kwargs)
        assert kwargs["ConditionExpression"] == "attribute_not_exists(job_name)"
        if self.put_barrier is not None:
            self.put_barrier.wait(timeout=5)
        with self.lock:
            key = kwargs["Item"]["job_name"]["S"]
            if key in self.items:
                raise ClientError(
                    {
                        "Error": {
                            "Code": "ConditionalCheckFailedException",
                            "Message": "The conditional request failed",
                        }
                    },
                    "PutItem",
                )
            self.items[key] = copy.deepcopy(kwargs["Item"])
        return {}

    def describe_table(self, **kwargs):
        self._call("DescribeTable", kwargs)
        return {
            "Table": {
                "TableName": TABLE,
                "TableArn": f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{TABLE}",
                "TableId": TABLE_ID,
                "TableStatus": "ACTIVE",
                "KeySchema": [{"AttributeName": "job_name", "KeyType": "HASH"}],
                "AttributeDefinitions": [{"AttributeName": "job_name", "AttributeType": "S"}],
                "StreamSpecification": {"StreamEnabled": True, "StreamViewType": "NEW_AND_OLD_IMAGES"},
                "LatestStreamArn": STREAM_ARN,
                "LatestStreamLabel": "2026-11-10T00:00:00.000",
            }
        }

    def count(self, operation):
        with self.lock:
            return sum(1 for name, _ in self.calls if name == operation)

    def rows(self):
        with self.lock:
            return copy.deepcopy(self.items)


def hint_key(name=SFT):
    return "PROMPT_TERMINAL_HINT#Training#" + name


def deliver(event, table, port, *, route=None, context=None):
    """One Lambda invocation: its own qualified scope, as an execution environment owns its own clients."""
    route = route or QualifiedRoute(callback_source)
    with callback_source.campaign_callback_scope(
        route, dynamodb_client=table, runtime_client=port, clock=lambda: 1000.0
    ):
        return callback_source.handler(copy.deepcopy(event), context), route


def deliver_in_threads(events, table, port, *, context=None):
    """Concurrent invocations, each in its own thread (as two Lambda environments run them)."""
    results, errors = [None] * len(events), []

    def run(index, value):
        try:
            results[index] = deliver(value, table, port, context=context)
        except BaseException as error:  # surfaced below; a thread must not swallow it
            errors.append(error)

    threads = [threading.Thread(target=run, args=(index, value)) for index, value in enumerate(events)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    assert not errors, errors
    assert all(not thread.is_alive() for thread in threads)
    return results


# ── The fixture is the documented shape with the real job content ──────────────────────────────────


def test_fixture_events_are_the_documented_envelope_with_real_job_content():
    statuses = sorted({job["status"] for job in REAL["jobs"]})
    assert statuses == ["Completed", "Failed", "Stopped"]
    for name, job in JOBS.items():
        event = real_event(name)
        assert set(event) == ENVELOPE
        detail = event["detail"]
        assert detail["TrainingJobName"] == name and detail["TrainingJobStatus"] == job["status"]
        assert event["resources"] == [detail["TrainingJobArn"]]
        assert detail["TrainingJobArn"].rsplit("/", 1)[1] == name, "real ARNs end with the exact job name"
        assert detail["SecondaryStatus"] == job["status"]
        assert ("FailureReason" in detail) is (job["status"] == "Failed")
        assert all(type(detail[key]) is int for key in ("CreationTime", "LastModifiedTime"))
        assert {"SecondaryStatusTransitions", "AlgorithmSpecification", "ResourceConfig", "Tags"} <= set(
            detail
        )
        identity = callback_source._event_identity(event, callback_source._campaign_config())
        assert identity == {
            "event_id": event["id"],
            "kind": "Training",
            "job_name": name,
            "job_arn": detail["TrainingJobArn"],
        }
    delivery = REAL["measured_delivery"]
    assert delivery["completed_or_failed_jobs"] == len(delivery["completed_or_failed_pair_gaps_ms"]) == 20
    assert all(
        job["observed_invocations"] == (1 if job["status"] == "Stopped" else 2) for job in REAL["jobs"]
    )


def test_product_shaped_sft_names_pass_the_native_identity_rule():
    """`qwen_sft` names jobs `<deployment prefix>-sft-<12 hex>`; longer deployment prefixes stay within 63."""
    for prefix in ("fixture-deploy", "amlas-demo", "amlas-pilot-001"):
        name = prefix + "-sft-" + "c3c3c3c3c3c3"
        event = real_event("fixture-deploy-sft-c3c3c3c3c3c3")
        arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/{name}"
        event["detail"].update(TrainingJobName=name, TrainingJobArn=arn)
        event["resources"] = [arn]
        assert callback_source._event_identity(event, callback_source._campaign_config())["job_name"] == name


# ── Live events of real jobs had no wake row: the lookup path, measured offline and live ───────────


@pytest.mark.parametrize("name", sorted(JOBS))
def test_real_terminal_event_without_a_wake_row_is_binding_missing_after_five_qualifications(name):
    table, port = Table(), AgentCore(resume_state="continuing")
    result, route = deliver(real_event(name), table, port)
    assert (result["statusCode"], result["delivery_status"]) == (404, "binding_missing")
    assert len(route.calls) == 5
    keys = [kwargs["Key"]["job_name"]["S"] for operation, kwargs in table.calls if operation == "GetItem"]
    assert keys == ["PROMPT_WAKE_JOB#Training#" + name, "CAMPAIGN_WAKE_JOB#Training#" + name]
    assert table.count("PutItem") == 0 and port.calls == []


class NativeControl:
    """bedrock-agentcore-control and STS ports for the real native qualifier; counts every read."""

    def __init__(self, role_name, role_id):
        self.counts = {"GetCallerIdentity": 0, "GetAgentRuntimeEndpoint": 0, "GetAgentRuntime": 0}
        self.role_name, self.role_id, self.lock = role_name, role_id, threading.Lock()
        self.meta = SimpleNamespace(region_name=REGION, config=SimpleNamespace(**TRANSPORT))

    def _count(self, name):
        with self.lock:
            self.counts[name] += 1
        return {"HTTPStatusCode": 200, "RetryAttempts": 0}

    def get_caller_identity(self):
        meta = self._count("GetCallerIdentity")
        arn = f"arn:aws:sts::{ACCOUNT}:assumed-role/{self.role_name}/fixture-callback-session"
        return {
            "Account": ACCOUNT,
            "Arn": arn,
            "UserId": self.role_id + ":fixture-callback-session",
            "ResponseMetadata": meta,
        }

    def get_agent_runtime_endpoint(self, **kwargs):
        meta = self._count("GetAgentRuntimeEndpoint")
        return {
            "agentRuntimeArn": RUNTIME,
            "name": "DEFAULT",
            "status": "READY",
            "liveVersion": "3",
            "ResponseMetadata": meta,
        }

    def get_agent_runtime(self, **kwargs):
        meta = self._count("GetAgentRuntime")
        return {
            "agentRuntimeArn": RUNTIME,
            "agentRuntimeVersion": "3",
            "status": "READY",
            "ResponseMetadata": meta,
        }


def test_live_pair_control_reads_equal_two_offline_lookups_through_the_native_qualifier(monkeypatch):
    """CloudTrail held 20 endpoint and 10 runtime reads by the callback role for one real SFT pair."""
    role_name, role_id = "fixture-callback-role", "AROA" + "B" * 17
    monkeypatch.setenv("AMLAS_CALLBACK_ROLE_ARN", f"arn:aws:iam::{ACCOUNT}:role/{role_name}")
    monkeypatch.setenv("AMLAS_CALLBACK_ROLE_ID", role_id)
    port, table = NativeControl(role_name, role_id), Table()
    for event_id in ("fixture-pair-1", "fixture-pair-2"):
        qualifier = callback_source._NativeCallbackRouteQualifier(
            callback_source._campaign_config(),
            expected_role_arn=f"arn:aws:iam::{ACCOUNT}:role/{role_name}",
            expected_role_id=role_id,
            sts_client=port,
            control_client=port,
        )
        with callback_source.campaign_callback_scope(
            qualifier, dynamodb_client=table, runtime_client=AgentCore()
        ):
            result = callback_source.handler(real_event(SFT, event_id=event_id))
        assert result["delivery_status"] == "binding_missing"
    live = REAL["measured_delivery"]["cloudtrail_reads_for_one_pair"]
    assert live["job"] == SFT
    assert port.counts["GetAgentRuntimeEndpoint"] == live["GetAgentRuntimeEndpoint"] == 20
    assert port.counts["GetAgentRuntime"] == live["GetAgentRuntime"] == 10
    assert port.counts["GetCallerIdentity"] == 10


def test_every_measured_pair_overlaps_in_time():
    """The second invocation starts while the first still runs, even on the shorter lookup path."""
    delivery = REAL["measured_delivery"]
    assert max(delivery["completed_or_failed_pair_gaps_ms"]) < min(
        delivery["campaign_lookup_path_durations_ms"]
    )


# ── Refused shapes: closed answers, no qualification, no client ────────────────────────────────────


def _refused_variants():
    def wrong_account(event):
        event["account"] = "000000000000"

    def wrong_region(event):
        event["region"] = "us-east-1"

    def missing_resources(event):
        event.pop("resources")

    def empty_resources(event):
        event["resources"] = []

    def resources_without_job(event):
        event["resources"] = [f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/fixture-other-job"]

    def arn_of_other_account(event):
        arn = event["detail"]["TrainingJobArn"].replace(ACCOUNT, "000000000000")
        event["detail"]["TrainingJobArn"] = arn
        event["resources"] = [arn]

    def nonterminal_status(event):
        event["detail"]["TrainingJobStatus"] = "InProgress"

    def missing_status(event):
        event["detail"].pop("TrainingJobStatus")

    def other_source(event):
        event["source"] = "fixture.sagemaker"

    def other_detail_type(event):
        event["detail-type"] = "SageMaker Training Job State Change v2"

    return {
        function.__name__: function
        for function in (
            wrong_account,
            wrong_region,
            missing_resources,
            empty_resources,
            resources_without_job,
            arn_of_other_account,
            nonterminal_status,
            missing_status,
            other_source,
            other_detail_type,
        )
    }


@pytest.mark.parametrize("variant", sorted(_refused_variants()))
def test_refused_shape_is_input_refused_without_qualification_or_effect(variant):
    event = real_event()
    _refused_variants()[variant](event)
    table, port = Table(wake_route()), AgentCore(resume_state="continuing")
    result, route = deliver(event, table, port)
    assert (result["statusCode"], result["delivery_status"]) == (400, "input_refused")
    assert route.calls == [] and table.calls == [] and port.calls == []


@pytest.mark.parametrize("variant", sorted(_refused_variants()))
def test_refused_shape_is_refused_before_the_deployed_bootstrap_builds_any_client(monkeypatch, variant):
    """The deployed entry (no injected scope) checks identity before it constructs STS or control clients."""
    event = real_event()
    _refused_variants()[variant](event)
    monkeypatch.setattr(
        callback_source.boto3, "client", lambda *a, **k: pytest.fail("client built for a refused event")
    )
    result = callback_source.handler(event)
    assert (result["statusCode"], result["delivery_status"]) == (400, "input_refused")


# ── One delivery with a wake row ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    ["fixture-deploy-sft-ex1-full-01", "fixture-deploy-sft-c3c3c3c3c3c3", "fixture-deploy-sft-d4d4d4d4d4d4"],
)
def test_real_terminal_event_with_a_wake_row_retains_its_status_and_resumes_once(name):
    table, port = Table(wake_route(name)), AgentCore(resume_state="continuing")
    result, _ = deliver(real_event(name), table, port)
    assert (result["statusCode"], result["delivery_status"]) == (200, "delivered")
    assert result["run_id"] == RUN_ID and result["operation_digest"] == OPERATION
    hint = json.loads(table.rows()[hint_key(name)]["record"]["S"])
    assert hint["status"] == JOBS[name]["status"] and hint["record_kind"] == "normal_prompt_terminal_hint_v1"
    assert len(port.calls) == 1
    sent = port.calls[0]
    assert sent["runtimeSessionId"] == SESSION and sent["qualifier"] == "DEFAULT"
    assert json.loads(sent["payload"]) == {"action": "resume_prompt_run", "run_id": RUN_ID}
    assert table.count("PutItem") == 1


def test_event_delivery_remote_call_budget():
    """Pins the sequential single-attempt calls of one EventBridge delivery (Lambda timeout 120 s).

    A qualification is 1 STS + 3 control-plane reads. The live lookup path made 22 calls (5 qualifications
    and 2 reads) in 1.4 to 1.9 s, 64 to 86 ms a call; a delivery makes 194 calls before InvokeAgentRuntime,
    about 12 to 17 s. Two deliveries of one state change start 6 to 878 ms apart, so both reach the runtime
    within about one second of each other.
    """
    table, port = Table(wake_route()), AgentCore(resume_state="restoring")
    route = QualifiedRoute(callback_source)
    before_invoke = []
    actual = port.invoke_agent_runtime

    def invoke(**kwargs):
        before_invoke.append(
            len(route.calls) * 4
            + table.count("GetItem")
            + table.count("DescribeTable")
            + table.count("PutItem")
        )
        return actual(**kwargs)

    port.invoke_agent_runtime = invoke
    result, _ = deliver(real_event(), table, port, route=route)
    assert result["delivery_status"] == "outcome_unknown"
    counts = (len(route.calls), table.count("GetItem"), table.count("DescribeTable"), table.count("PutItem"))
    assert counts == (55, 10, 17, 1), counts
    assert len(route.calls) * 4 + sum(counts[1:]) + len(port.calls) == 249
    assert before_invoke == [194]


# ── Duplicate, late and unacknowledged deliveries against the closed-state port ────────────────────


def test_two_concurrent_deliveries_of_one_state_change_both_resume_the_same_run():
    """Two event ids, both past the hint check at once: one hint, one conditional failure, two resumes."""
    table = Table(wake_route(), put_barrier=threading.Barrier(2))
    port = AgentCore(resume_state="continuing")
    events = [real_event(event_id="fixture-state-change-a"), real_event(event_id="fixture-state-change-b")]
    results = deliver_in_threads(events, table, port)
    assert [result["delivery_status"] for result, _ in results] == ["delivered", "delivered"]
    assert table.count("PutItem") == 2
    assert [key for key in table.rows() if key.startswith("PROMPT_TERMINAL_HINT#")] == [hint_key()]
    assert len(port.calls) == 2
    assert {(call["runtimeSessionId"], call["payload"]) for call in port.calls} == {
        (
            SESSION,
            json.dumps({"action": "resume_prompt_run", "run_id": RUN_ID}, separators=(",", ":")).encode(),
        )
    }


def test_redelivered_same_event_keeps_the_hint_and_resumes_again():
    """Lambda's eventually consistent queue may hand the same event (same id) to the function twice."""
    table, port = Table(wake_route()), AgentCore(resume_state="continuing")
    first, _ = deliver(real_event(event_id="fixture-same-id"), table, port)
    hint = table.rows()[hint_key()]
    second, _ = deliver(real_event(event_id="fixture-same-id"), table, port)
    assert first["delivery_status"] == second["delivery_status"] == "delivered"
    assert table.rows()[hint_key()] == hint and table.count("PutItem") == 2 and len(port.calls) == 2


@pytest.mark.parametrize("failure", ["read_timeout", "restoring", "original_active", "failed_transient"])
def test_unacknowledged_resume_returns_without_raising_so_lambda_does_not_retry(failure):
    """Lambda retries an asynchronous event only on a function error; the event path never raises."""
    table = Table(wake_route())
    port = AgentCore(resume_state=None if failure == "read_timeout" else failure)
    if failure == "read_timeout":

        def lost(**kwargs):
            port.calls.append(kwargs)
            raise ReadTimeoutError(endpoint_url="https://bedrock-agentcore.us-west-2.amazonaws.com")

        port.invoke_agent_runtime = lost
    result, _ = deliver(real_event(), table, port)
    assert (result["statusCode"], result["delivery_status"]) == (202, "outcome_unknown")
    assert len(port.calls) == 1 and hint_key() in table.rows()


# ── The stream readiness record across a 3.5 h SFT wait ────────────────────────────────────────────


def _stream(table, port, record, route=None):
    return deliver({"Records": [copy.deepcopy(record)]}, table, port, route=route)


def test_readiness_record_retried_through_a_long_wait_never_calls_the_runtime_before_the_event():
    """3.5 h at the measured 70 to 90 s retry interval is 150 attempts; each fails closed and changes nothing."""
    attempts = (3 * 3600 + 1800) // 84
    assert attempts == 150
    table, port = Table(wake_route(), run_row()), AgentCore(resume_state="continuing")
    before = table.rows()
    record, route = readiness_record(), QualifiedRoute(callback_source)
    for _ in range(attempts):
        with pytest.raises(RuntimeError, match="normal readiness delivery unresolved"):
            _stream(table, port, record, route)
    assert port.calls == [] and table.rows() == before and table.count("PutItem") == 0
    per_attempt = tuple(
        value / attempts for value in (len(route.calls), table.count("GetItem"), table.count("DescribeTable"))
    )
    assert per_attempt == (21.0, 3.0, 7.0), per_attempt
    assert attempts * (21 * 4 + 3 + 7) == 14100, "remote calls of the whole wait, none to the runtime"


def test_readiness_record_resumes_after_the_event_and_is_acknowledged_only_by_a_positive_state():
    table, port = Table(wake_route(), run_row()), AgentCore(resume_state="restoring")
    record = readiness_record()
    with pytest.raises(RuntimeError, match="normal readiness delivery unresolved"):
        _stream(table, port, record)
    assert port.calls == []
    event_result, _ = deliver(real_event(), table, port)
    assert event_result["delivery_status"] == "outcome_unknown"
    outcomes = []
    for state in ("restoring", "restoring", "continuing"):
        port.resume_state, port.body = state, Body()
        try:
            outcomes.append(_stream(table, port, record)[0]["delivery_status"])
        except RuntimeError:
            outcomes.append("retry")
    assert outcomes == ["retry", "retry", "delivered"]
    assert len(port.calls) == 4
    assert {json.loads(call["payload"])["run_id"] for call in port.calls} == {RUN_ID}


# ── The same sequence against the real app's resume admission (one microVM) ────────────────────────


class AppRuntime:
    """bedrock-agentcore port served by the real app on one event loop, as one microVM serves its session."""

    def __init__(self, loop, context):
        self.loop, self.context, self.lock = loop, context, threading.Lock()
        self.calls, self.answers, self.lose = [], [], set()
        self.meta = SimpleNamespace(config=SimpleNamespace(**TRANSPORT))

    def invoke_agent_runtime(self, **kwargs):
        validate_parameters(
            kwargs, MODELS["bedrock-agentcore"].operation_model("InvokeAgentRuntime").input_shape
        )
        assert kwargs["runtimeSessionId"] == SESSION and kwargs["qualifier"] == "DEFAULT"
        with self.lock:
            self.calls.append(copy.deepcopy(kwargs))
            number = len(self.calls)
        event = run_on(self.loop, self.context, self._first(json.loads(kwargs["payload"])))
        with self.lock:
            self.answers.append(event)
        if number in self.lose:
            raise ReadTimeoutError(endpoint_url="https://bedrock-agentcore.us-west-2.amazonaws.com")
        body = Body(("data: " + json.dumps(event) + "\n\n").encode())
        return {"response": body, "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0}}

    @staticmethod
    async def _first(payload):
        stream = runtime.agent_invocation(payload)
        try:
            return await asyncio.wait_for(anext(stream), timeout=3)
        finally:
            await stream.aclose()

    def states(self):
        with self.lock:
            return [answer.get("state") for answer in self.answers]


def run_on(loop, context, coroutine, timeout=10):
    future = concurrent.futures.Future()

    def start():
        task = loop.create_task(coroutine, context=context.copy())

        def done(value):
            if value.cancelled():
                future.cancel()
            elif value.exception() is not None:
                future.set_exception(value.exception())
            else:
                future.set_result(value.result())

        task.add_done_callback(done)

    loop.call_soon_threadsafe(start)
    return future.result(timeout=timeout)


def settle(loop, context, value):
    """Wait for the app's task ownership to end and then for the worker's final record.

    The worker writes its final resume record after the app's async task count reaches zero; a test that
    stops at the count leaves a late `completed` record that the next test's first resume would read.
    """
    run_on(loop, context, _settled(value))
    deadline = time.monotonic() + 5
    while True:
        with runtime._PROMPT_REPORT_LOCK:
            record = runtime._RESUME_WORKERS.get(RUN_ID)
            state = record["state"] if record is not None else None
        if state == "completed" or time.monotonic() > deadline:
            return state
        time.sleep(0.005)


@pytest.fixture
def app_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="fixture-microvm-loop", daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def test_real_app_answers_the_concurrent_pair_with_one_restore_and_a_closed_busy_state(resume, app_loop):
    context = contextvars.copy_context()
    port = AppRuntime(app_loop, context)
    table = Table(wake_route(), run_row(), put_barrier=threading.Barrier(2))
    resume.prepare_gate.clear()
    try:
        events = [
            real_event(event_id="fixture-state-change-a"),
            real_event(event_id="fixture-state-change-b"),
        ]
        results = deliver_in_threads(events, table, port)
        assert resume.prepare_entered.wait(5)
        assert [result["delivery_status"] for result, _ in results] == ["outcome_unknown", "outcome_unknown"]
        assert port.states() == ["restoring", "restoring"]
        assert resume.prepare_calls == 1, "the second resume must not start a second restore"
    finally:
        resume.prepare_gate.set()
    assert settle(app_loop, context, resume) == "completed"
    assert resume.claim_calls == 1 and resume.continue_calls == 1
    late, _ = deliver(real_event(event_id="fixture-late-duplicate"), table, port)
    assert late["delivery_status"] == "delivered" and port.states()[-1] == "completed"
    assert resume.prepare_calls == 1 and resume.continue_calls == 1


def test_real_app_continuing_state_acknowledges_the_readiness_record_once(resume, app_loop):
    context = contextvars.copy_context()
    port = AppRuntime(app_loop, context)
    table = Table(wake_route(), run_row())
    record = readiness_record()
    resume.continue_release.clear()
    try:
        first, _ = deliver(real_event(), table, port)
        assert first["delivery_status"] == "outcome_unknown" and port.states() == ["restoring"]
        assert resume.continue_entered.wait(5)
        acknowledged, _ = _stream(table, port, record)
        assert acknowledged["delivery_status"] == "delivered" and port.states()[-1] == "continuing"
    finally:
        resume.continue_release.set()
    assert settle(app_loop, context, resume) == "completed"
    assert resume.prepare_calls == 1 and resume.claim_calls == 1 and resume.continue_calls == 1


def test_real_app_lost_response_of_the_first_resume_leaves_one_restore(resume, app_loop):
    """The resume reached the app and started the restore; its answer never reached the Lambda."""
    context = contextvars.copy_context()
    port = AppRuntime(app_loop, context)
    port.lose.add(1)
    table = Table(wake_route(), run_row())
    resume.prepare_gate.clear()
    try:
        first, _ = deliver(real_event(), table, port)
        assert first["delivery_status"] == "outcome_unknown"
        assert resume.prepare_entered.wait(5), (port.states(), dict(runtime._RESUME_WORKERS))
        with pytest.raises(RuntimeError, match="normal readiness delivery unresolved"):
            _stream(table, port, readiness_record())
        assert port.states() == ["restoring", "restoring"] and resume.prepare_calls == 1
    finally:
        resume.prepare_gate.set()
    assert settle(app_loop, context, resume) == "completed"
    assert resume.continue_calls == 1


# ── The delivery that did not create the hint waits before it resumes (twin delay) ─────────────────


class LambdaContext:
    """The part of the Lambda context object the handler reads."""

    def __init__(self, remaining_ms):
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self):
        return self.remaining_ms


class Timeline:
    """Per-thread order of put outcomes, pauses and invokes across concurrent deliveries."""

    def __init__(self, monkeypatch, table, port):
        self.lock, self.entries = threading.Lock(), []
        monkeypatch.setattr(
            callback_source, "_twin_pause", lambda seconds: self.note("pause", seconds), raising=False
        )
        actual_put, actual_invoke = table.put_item, port.invoke_agent_runtime

        def put_item(**kwargs):
            try:
                result = actual_put(**kwargs)
            except ClientError as error:
                self.note("put", error.response["Error"]["Code"])
                raise
            self.note("put", "created")
            return result

        def invoke_agent_runtime(**kwargs):
            self.note("invoke", None)
            return actual_invoke(**kwargs)

        table.put_item, port.invoke_agent_runtime = put_item, invoke_agent_runtime

    def note(self, kind, value):
        with self.lock:
            self.entries.append((threading.current_thread().name, kind, value))

    def sequences(self):
        by_thread = {}
        with self.lock:
            for name, kind, value in self.entries:
                by_thread.setdefault(name, []).append((kind, value))
        return sorted(by_thread.values())


CREATOR = [("put", "created"), ("invoke", None)]
TWIN = [("put", "ConditionalCheckFailedException"), ("pause", 40.0), ("invoke", None)]


def test_twin_of_a_concurrent_pair_waits_inside_its_lambda_budget_before_it_resumes(monkeypatch):
    table, port = Table(wake_route(), put_barrier=threading.Barrier(2)), AgentCore(resume_state="restoring")
    timeline = Timeline(monkeypatch, table, port)
    events = [real_event(event_id="fixture-state-change-a"), real_event(event_id="fixture-state-change-b")]
    results = deliver_in_threads(events, table, port, context=LambdaContext(105_000))
    assert timeline.sequences() == sorted([CREATOR, TWIN])
    assert [result["delivery_status"] for result, _ in results] == ["outcome_unknown", "outcome_unknown"]
    assert len(port.calls) == 2 and table.count("PutItem") == 2


def test_twin_delay_defaults_fit_the_lambda_timeout():
    """194 calls at the slowest measured 86 ms, the delay and the post-invoke reserve stay under 120 s."""
    assert callback_source._twin_pause is time.sleep
    assert (callback_source._TWIN_RESUME_DELAY_SECONDS, callback_source._TWIN_RESUME_RESERVE_SECONDS) == (
        40.0,
        50.0,
    )
    assert 194 * 0.086 + 40.0 + 50.0 < 120


def test_single_delivery_resumes_without_waiting(monkeypatch):
    table, port = Table(wake_route("fixture-deploy-sft-d4d4d4d4d4d4")), AgentCore(resume_state="restoring")
    timeline = Timeline(monkeypatch, table, port)
    deliver(real_event("fixture-deploy-sft-d4d4d4d4d4d4"), table, port, context=LambdaContext(105_000))
    assert timeline.sequences() == [CREATOR]


def test_redelivered_event_waits_and_then_receives_the_closed_answer(monkeypatch):
    table, port = Table(wake_route()), AgentCore(resume_state="restoring")
    timeline = Timeline(monkeypatch, table, port)
    deliver(real_event(event_id="fixture-same-id"), table, port, context=LambdaContext(105_000))
    port.body = Body()
    second, _ = deliver(real_event(event_id="fixture-same-id"), table, port, context=LambdaContext(105_000))
    assert second["delivery_status"] == "outcome_unknown"
    assert timeline.sequences() == [CREATOR + TWIN], "one thread delivered both, in order"


def test_put_applied_but_unacknowledged_waits_because_it_proves_publication_not_creation(monkeypatch):
    table, port = Table(wake_route()), AgentCore(resume_state="continuing")
    timeline = Timeline(monkeypatch, table, port)
    applied = table.put_item

    def lost_acknowledgement(**kwargs):
        applied(**kwargs)
        raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "PutItem")

    table.put_item = lost_acknowledgement
    result, _ = deliver(real_event(), table, port, context=LambdaContext(105_000))
    assert result["delivery_status"] == "delivered"
    assert timeline.sequences() == [[("put", "created"), ("pause", 40.0), ("invoke", None)]]


@pytest.mark.parametrize(
    ("remaining_ms", "pause"), [(105_000, 40.0), (70_000, 20.0), (50_000, None), (10_000, None), (None, None)]
)
def test_twin_wait_is_bounded_by_the_remaining_lambda_time(monkeypatch, remaining_ms, pause):
    table = Table(wake_route())
    port = AgentCore(resume_state="restoring")
    deliver(real_event(event_id="fixture-first"), table, port, context=LambdaContext(105_000))
    timeline = Timeline(monkeypatch, table, port)
    port.body = Body()
    context = None if remaining_ms is None else LambdaContext(remaining_ms)
    deliver(real_event(event_id="fixture-second"), table, port, context=context)
    expected = [("put", "ConditionalCheckFailedException")]
    expected += [] if pause is None else [("pause", pause)]
    assert timeline.sequences() == [[*expected, ("invoke", None)]]


def test_readiness_record_never_waits(monkeypatch):
    table, port = Table(wake_route(), run_row()), AgentCore(resume_state="continuing")
    deliver(real_event(), table, port, context=LambdaContext(105_000))
    timeline = Timeline(monkeypatch, table, port)
    port.body = Body()
    result, _ = deliver({"Records": [readiness_record()]}, table, port, context=LambdaContext(105_000))
    assert result["delivery_status"] == "delivered"
    assert timeline.sequences() == [[("invoke", None)]]
