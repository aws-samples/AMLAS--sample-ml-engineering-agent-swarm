"""Closed callback source route; neutral SDK-shaped transports, no native proof."""

import copy
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from botocore.session import Session
from botocore.validate import validate_parameters

ACCOUNT = "111122223333"
REGION = "us-west-2"
DEPLOYMENT = "12345678-1234-4234-9234-00000000dd11"
TABLE = "fixture-training-jobs"
RUNTIME = f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/fixtureRuntime-ABCDEFGHIJ"
REFERENCE = "campaign_" + "a" * 64 + "_" + "b" * 64
TRANSPORT = "12345678-1234-4234-9234-00000000dd12"
MODELS = {name: Session().get_service_model(name) for name in ("dynamodb", "bedrock-agentcore")}


def callback():
    value = importlib.import_module("amlas.async_callback.lambda_handler")
    assert hasattr(value, "campaign_callback_scope"), "missing-symbol: qualified campaign_callback_scope"
    return value


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for name, value in {
        "AMLAS_CAMPAIGN_CALLBACK_MODE": "1",
        "AMLAS_AWS_ACCOUNT_ID": ACCOUNT,
        "AWS_REGION": REGION,
        "AMLAS_DEPLOYMENT_ID": DEPLOYMENT,
        "JOBS_TABLE": TABLE,
        "AGENTCORE_RUNTIME_ARN": RUNTIME,
        "AGENTCORE_RUNTIME_QUALIFIER": "CAMPAIGN",
        "AGENTCORE_RUNTIME_VERSION": "3",
        "AMLAS_CALLBACK_ROUTE_SHA256": hashlib.sha256(
            b"fixture independently reviewed callback route"
        ).hexdigest(),
    }.items():
        monkeypatch.setenv(name, value)


def event(kind="Processing", *, event_id="fixture-event-1", status="Completed"):
    # Fixed neutral job-name fixtures; no credential material.
    name = (
        "fixture-model-acq-0123456789ab"  # pragma: allowlist secret
        if kind == "Processing"
        else "fixture-sft-0123456789ab"  # pragma: allowlist secret
    )
    arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:{'processing-job' if kind == 'Processing' else 'training-job'}/{name}"
    return {
        "version": "0",
        "id": event_id,
        "source": "aws.sagemaker",
        "account": ACCOUNT,
        "region": REGION,
        "detail-type": f"SageMaker {kind} Job State Change",
        "resources": [arn],
        "detail": {
            kind + "JobName": name,
            kind + "JobArn": arn,
            kind + "JobStatus": status,
            "OrdinaryProviderField": {"ignored": True},
        },
    }


def hint_item(value):
    kind = "Processing" if "Processing" in value["detail-type"] else "Training"
    body = {
        "schema_version": 1,
        "record_kind": "campaign_wake_job_v1",
        "account_id": ACCOUNT,
        "region": REGION,
        "deployment_id": DEPLOYMENT,
        "kind": kind,
        "job_name": value["detail"][kind + "JobName"],
        "job_arn": value["detail"][kind + "JobArn"],
        "campaign_ref": REFERENCE,
        "operation_digest": hashlib.sha256((kind + "-original").encode()).hexdigest(),
        "transport_session_id": TRANSPORT,
    }
    return {
        "job_name": {"S": "CAMPAIGN_WAKE_JOB#" + kind + "#" + body["job_name"]},
        "private_record": {"B": json.dumps(body, sort_keys=True, separators=(",", ":")).encode()},
    }


class Body(io.BytesIO):
    def __init__(self, data=b"neutral delivery bytes"):
        super().__init__(data)
        self.closed_called = False

    def read(self, amount=-1):
        assert 0 < amount <= 65536, "callback cannot unbounded-read response"
        return super().read(amount)

    def close(self):
        self.closed_called = True
        super().close()


class DDB:
    def __init__(self, item):
        self.item, self.calls = item, []
        self.meta = SimpleNamespace(
            config=SimpleNamespace(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10)
        )

    def get_item(self, **kwargs):
        validate_parameters(kwargs, MODELS["dynamodb"].operation_model("GetItem").input_shape)
        assert kwargs["ConsistentRead"] is True and kwargs["TableName"] == TABLE
        self.calls.append(kwargs)
        if self.item is None or kwargs["Key"] != {"job_name": self.item["job_name"]}:
            return {}
        return {"Item": copy.deepcopy(self.item)}

    def __getattr__(self, name):
        if name in ("scan", "query", "put_item", "update_item", "delete_item"):
            return lambda **kwargs: pytest.fail("campaign callback may only exact Get")
        raise AttributeError(name)


def resume_state_bytes(state, run_id, *, sse=True, **extra):
    """The first streamed event of the resume action: one closed event, SSE framed by default."""
    event = {"event": "resume_state", "state": state, "run_id": run_id, **extra}
    raw = json.dumps(event, sort_keys=True, separators=(",", ":"))
    return ("data: " + raw + "\n\n").encode() if sse else (raw + "\n").encode()


class AgentCore:
    def __init__(self, resume_state=None):
        # `resume_state` makes the fake answer a resume_prompt_run payload with that closed
        # first event, bound to the run in the payload (campaign wakes keep the neutral body).
        self.calls, self.body, self.fail = [], Body(), False
        self.resume_state = resume_state
        self.meta = SimpleNamespace(
            config=SimpleNamespace(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10)
        )

    def invoke_agent_runtime(self, **kwargs):
        validate_parameters(
            kwargs, MODELS["bedrock-agentcore"].operation_model("InvokeAgentRuntime").input_shape
        )
        self.calls.append(kwargs)
        if self.fail:
            raise TimeoutError("neutral original delivery response unavailable")
        payload = json.loads(kwargs["payload"])
        if self.resume_state is not None and payload.get("action") == "resume_prompt_run":
            self.body = Body(resume_state_bytes(self.resume_state, payload["run_id"]))
        return {"response": self.body, "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0}}


class QualifiedRoute:
    """Typed fixture port is conditional test evidence, never native IAM proof."""

    def __init__(self, module):
        self.module, self.calls, self.valid = module, [], True

    def qualify_current(self, config, identity):
        self.calls.append((config, identity))
        if not self.valid:
            return None
        return self.module.CallbackRouteObservation(
            account_id=config.account_id,
            region=config.region,
            deployment_id=config.deployment_id,
            runtime_arn=config.runtime_arn,
            qualifier=config.qualifier,
            runtime_version=config.runtime_version,
            route_sha256=config.route_sha256,
            observed_at=1000.0,
            expires_at=1030.0,
            process_id=os.getpid(),
            issuer=self,
        )


def fixtures(kind="Processing"):
    module = callback()
    value = event(kind)
    ddb, runtime = DDB(hint_item(value)), AgentCore()
    route = QualifiedRoute(module)
    return module, value, ddb, runtime, route


def test_missing_private_qualification_denies_before_any_get_or_invoke(monkeypatch):
    module = callback()
    monkeypatch.setattr(
        module.boto3, "client", lambda *args, **kwargs: pytest.fail("unqualified path reached SDK")
    )
    result = module.handler(event())
    assert result["statusCode"] == 403 and result["delivery_status"] == "qualification_missing"


@pytest.mark.parametrize("kind", ["Processing", "Training"])
def test_closed_terminal_hint_routes_to_exact_configured_runtime_without_writes(kind):
    module, value, ddb, runtime, route = fixtures(kind)
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "delivered" and result["lifecycle_verified"] is False
    assert len(runtime.calls) == 1
    assert [request["Key"] for request in ddb.calls] == [
        {"job_name": {"S": "PROMPT_WAKE_JOB#" + kind + "#" + value["detail"][kind + "JobName"]}},
        {"job_name": {"S": "CAMPAIGN_WAKE_JOB#" + kind + "#" + value["detail"][kind + "JobName"]}},
    ]
    sent = runtime.calls[0]
    assert sent["agentRuntimeArn"] == RUNTIME and sent["qualifier"] == "CAMPAIGN"
    assert sent["runtimeSessionId"] == TRANSPORT
    wake = json.loads(sent["payload"])
    assert set(wake) == {
        "schema_version",
        "event_id",
        "campaign_ref",
        "account_id",
        "region",
        "deployment_id",
        "kind",
        "operation_digest",
        "job_arn",
    }
    assert wake["event_id"] == value["id"] and wake["campaign_ref"] == REFERENCE
    assert "status" not in wake and runtime.body.closed_called
    assert len(route.calls) >= 3


@pytest.mark.parametrize("field", ["source", "account", "region", "resources"])
def test_foreign_or_inconsistent_event_identity_refuses_before_lookup(field):
    module, value, ddb, runtime, route = fixtures()
    value[field] = [] if field == "resources" else "ordinary-other"
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["statusCode"] == 400 and ddb.calls == [] and runtime.calls == []


@pytest.mark.parametrize("mutation", ["missing", "session", "job", "extra"])
def test_missing_changed_or_unbounded_lookup_never_invokes(mutation):
    module, value, ddb, runtime, route = fixtures()
    if mutation == "missing":
        ddb.item = None
    else:
        body = json.loads(ddb.item["private_record"]["B"])
        if mutation == "session":
            body["transport_session_id"] = 7
        elif mutation == "job":
            body["job_arn"] += "-other"
        else:
            body["owner_policy"] = "caller-claim"
        ddb.item["private_record"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] in ("binding_missing", "binding_refused") and runtime.calls == []


def test_current_route_revocation_after_get_prevents_invoke():
    module, value, ddb, runtime, route = fixtures()
    get = ddb.get_item

    def revoke(**kwargs):
        result = get(**kwargs)
        if kwargs["Key"] == {"job_name": ddb.item["job_name"]}:
            route.valid = False
        return result

    ddb.get_item = revoke
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "qualification_refused" and runtime.calls == []


def test_duplicate_reordered_statuses_are_only_observation_hints():
    module, value, ddb, runtime, route = fixtures()
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        for status in ("Completed", "Failed", "Completed"):
            value["detail"]["ProcessingJobStatus"] = status
            runtime.body = Body()
            assert module.handler(value)["delivery_status"] == "delivered"
    assert len(runtime.calls) == 3
    assert len({request["payload"] for request in runtime.calls}) == 1


def test_unknown_delivery_retains_original_and_never_retries():
    module, value, ddb, runtime, route = fixtures()
    runtime.fail = True
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "outcome_unknown" and result["campaign_ref"] == REFERENCE
    assert result["job_arn"] == value["detail"]["ProcessingJobArn"]
    assert len(runtime.calls) == 1 and result["lifecycle_verified"] is False


def test_response_bound_closes_stream_and_labels_unknown():
    module, value, ddb, runtime, route = fixtures()
    runtime.body = Body(b"a" * (1024 * 1024 + 1))
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "outcome_unknown" and runtime.body.closed_called
    assert len(runtime.calls) == 1


@pytest.mark.parametrize(
    "metadata",
    [
        {"HTTPStatusCode": 500, "RetryAttempts": 0},
        {"HTTPStatusCode": 200, "RetryAttempts": 1},
        {"HTTPStatusCode": True, "RetryAttempts": 0},
    ],
)
def test_nonqualified_invoke_metadata_is_unknown_and_stream_closed(metadata):
    module, value, ddb, runtime, route = fixtures()
    invoke = runtime.invoke_agent_runtime

    def response(**kwargs):
        result = invoke(**kwargs)
        result["ResponseMetadata"] = metadata
        return result

    runtime.invoke_agent_runtime = response
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "outcome_unknown"
    assert runtime.body.closed_called and len(runtime.calls) == 1


@pytest.mark.parametrize(
    "selector", ["AGENTCORE_RUNTIME_QUALIFIER", "AGENTCORE_RUNTIME_ARN", "AMLAS_CALLBACK_ROUTE_SHA256"]
)
def test_missing_server_route_selector_denies_before_get(selector, monkeypatch):
    module, value, ddb, runtime, route = fixtures()
    monkeypatch.delenv(selector)
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["statusCode"] == 400 and ddb.calls == [] and runtime.calls == []


def test_expired_private_route_denies_before_lookup():
    module, value, ddb, runtime, route = fixtures()
    qualify = route.qualify_current

    def expired(*args):
        from dataclasses import replace

        return replace(qualify(*args), expires_at=1000.0)

    route.qualify_current = expired
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "qualification_refused" and ddb.calls == [] and runtime.calls == []


def test_inherited_qualification_scope_denies_before_lookup(monkeypatch):
    module, value, ddb, runtime, route = fixtures()
    pid = os.getpid()
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        monkeypatch.setattr(module.os, "getpid", lambda: pid + 1)
        result = module.handler(value)
    assert result["delivery_status"] == "qualification_refused" and ddb.calls == [] and runtime.calls == []


def test_drain_deadline_closes_original_stream_without_retry():
    module, value, ddb, runtime, route = fixtures()
    now = [1000.0]

    class DelayedBody(Body):
        def read(self, amount=-1):
            result = super().read(amount)
            now[0] += 21
            return result

    runtime.body = DelayedBody()
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: now[0]
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "outcome_unknown" and runtime.body.closed_called
    assert len(runtime.calls) == 1


def test_close_failure_is_unknown_not_delivered():
    module, value, ddb, runtime, route = fixtures()

    class CloseFailure(Body):
        def close(self):
            self.closed_called = True
            super().close()
            raise OSError("neutral original stream close unavailable")

    runtime.body = CloseFailure()
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "outcome_unknown" and runtime.body.closed_called
    assert len(runtime.calls) == 1


def test_legacy_disabled_handler_retains_original_training_route(monkeypatch):
    module = callback()
    monkeypatch.delenv("AMLAS_CAMPAIGN_CALLBACK_MODE")
    calls = []

    class LegacyDDB:
        def get_item(self, **kwargs):
            calls.append(("get", kwargs))
            return {"Item": {"runtime_session_id": {"S": TRANSPORT}}}

        def update_item(self, **kwargs):
            calls.append(("update", kwargs))

    class LegacyRuntime:
        def invoke_agent_runtime(self, **kwargs):
            calls.append(("invoke", kwargs))
            return {"response": io.BytesIO(b"legacy")}

    monkeypatch.setattr(
        module.boto3, "client", lambda name, **kwargs: LegacyDDB() if name == "dynamodb" else LegacyRuntime()
    )
    result = module.handler(event("Training"))
    assert result["statusCode"] == 200
    assert [name for name, _ in calls] == ["get", "update", "invoke"]
    assert calls[-1][1]["qualifier"] == "DEFAULT"
    assert json.loads(calls[-1][1]["payload"])["event"] == "training_complete"


def test_shared_pure_binding_codec_matches_exact_frozen_body():
    module, value, _ddb, _runtime, _route = fixtures()
    row = hint_item(value)
    body = json.loads(row["private_record"]["B"])
    assert module.encode_wake_job_binding(body) == row["private_record"]["B"]
    assert module.wake_job_lookup_key(body["kind"], body["job_name"]) == row["job_name"]["S"]
    body["job_arn"] += "-other"
    with pytest.raises(ValueError):
        module.encode_wake_job_binding(body)
    body["policy"] = "ordinary-untrusted-claim"
    with pytest.raises(ValueError):
        module.encode_wake_job_binding(body)


def tracked_body(module, now, *, close_delay=0, read_callable=True):
    class Tracked:
        def __init__(self):
            self.reads, self.closes = 0, 0

        def read(self, amount):
            self.reads += 1
            return b""

        def close(self):
            self.closes += 1
            now[0] += close_delay

    body = Tracked()
    if not read_callable:
        body.read = None
    return body


def dynamic_route(route, module, now):
    def qualify(config, identity):
        if not route.valid:
            return None
        return module.CallbackRouteObservation(
            account_id=config.account_id,
            region=config.region,
            deployment_id=config.deployment_id,
            runtime_arn=config.runtime_arn,
            qualifier=config.qualifier,
            runtime_version=config.runtime_version,
            route_sha256=config.route_sha256,
            observed_at=now[0],
            expires_at=now[0] + 30,
            process_id=os.getpid(),
            issuer=route,
        )

    route.qualify_current = qualify


@pytest.mark.parametrize("failure", ["clock", "interface"])
def test_owned_original_closes_when_postinvoke_validation_fails(failure):
    module, value, ddb, runtime, route = fixtures()
    now = [1000.0]
    runtime.body = tracked_body(module, now, read_callable=failure != "interface")
    invoke = runtime.invoke_agent_runtime

    def returned(**kwargs):
        result = invoke(**kwargs)
        if failure == "clock":
            now[0] = 999.0
        return result

    runtime.invoke_agent_runtime = returned
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: now[0]
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "outcome_unknown" and result["campaign_ref"] == REFERENCE
    assert runtime.body.closes == 1 and len(runtime.calls) == 1


def test_close_elapsed_overrun_is_unknown_with_fresh_route_timestamps():
    module, value, ddb, runtime, route = fixtures()
    now = [1000.0]
    dynamic_route(route, module, now)
    runtime.body = tracked_body(module, now, close_delay=21)
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: now[0]
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "outcome_unknown"
    assert runtime.body.closes == 1 and len(runtime.calls) == 1


@pytest.mark.parametrize("threaded", [False, True])
def test_retained_scope_copy_after_exit_cannot_get_or_invoke(threaded):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    module, value, ddb, runtime, route = fixtures()
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        saved = copy_context()
    if threaded:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(saved.run, module.handler, value).result(timeout=5)
    else:
        result = saved.run(module.handler, value)
    assert result["delivery_status"] in ("qualification_missing", "qualification_refused")
    assert ddb.calls == [] and runtime.calls == []


def test_active_scope_copy_to_another_thread_is_not_propagatable():
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    module, value, ddb, runtime, route = fixtures()
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        saved = copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(saved.run, module.handler, value).result(timeout=5)
    assert result["delivery_status"] == "qualification_refused"
    assert ddb.calls == [] and runtime.calls == []


def test_postinvoke_revocation_zero_reads_one_original_cleanup():
    module, value, ddb, runtime, route = fixtures()
    now = [1000.0]
    runtime.body = tracked_body(module, now)
    invoke = runtime.invoke_agent_runtime

    def revoke(**kwargs):
        result = invoke(**kwargs)
        route.valid = False
        return result

    runtime.invoke_agent_runtime = revoke
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: now[0]
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "outcome_unknown" and result["job_arn"] == value["resources"][0]
    assert runtime.body.reads == 0 and runtime.body.closes == 1


@pytest.mark.parametrize(
    "client_name,field,limit",
    [("ddb", "connect_timeout", 5), ("ddb", "read_timeout", 10), ("runtime", "connect_timeout", 5)],
)
@pytest.mark.parametrize("invalid", [None, float("inf"), 99])
def test_injected_transport_timeout_must_be_exact_finite_bounded(client_name, field, limit, invalid):
    module, value, ddb, runtime, route = fixtures()
    client = ddb if client_name == "ddb" else runtime
    setattr(client.meta.config, field, invalid)
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "qualification_refused"
    if client_name == "ddb":
        assert ddb.calls == []
    assert runtime.calls == []


@pytest.mark.parametrize("field", ["source", "detail-type"])
def test_consumed_event_discriminator_requires_exact_string(field):
    module, value, ddb, runtime, route = fixtures()

    class Text(str):
        pass

    value[field] = Text(value[field])
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["statusCode"] == 400 and ddb.calls == [] and runtime.calls == []


def test_each_event_resource_is_bounded_before_lookup():
    module, value, ddb, runtime, route = fixtures()
    value["resources"].append("ordinary" * 1000)
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["statusCode"] == 400 and ddb.calls == []


def test_runtime_client_setup_failure_is_not_an_attempt(monkeypatch):
    module, value, ddb, _runtime, route = fixtures()

    def unavailable(*args, **kwargs):
        raise OSError("neutral runtime client setup unavailable")

    monkeypatch.setattr(module.boto3, "client", unavailable)
    with module.campaign_callback_scope(route, dynamodb_client=ddb, clock=lambda: 1000.0):
        result = module.handler(value)
    assert result["delivery_status"] == "setup_unavailable"
    assert result["campaign_ref"] == REFERENCE and result["statusCode"] == 503
    assert [request["Key"] for request in ddb.calls] == [
        {"job_name": {"S": "PROMPT_WAKE_JOB#Processing#" + value["detail"]["ProcessingJobName"]}},
        {"job_name": {"S": "CAMPAIGN_WAKE_JOB#Processing#" + value["detail"]["ProcessingJobName"]}},
    ]


def test_plain_worker_denies_but_separately_fresh_worker_scope_succeeds():
    from concurrent.futures import ThreadPoolExecutor

    module, value, ddb, runtime, route = fixtures()

    def fresh():
        with module.campaign_callback_scope(
            route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
        ):
            return module.handler(value)

    with ThreadPoolExecutor(max_workers=1) as pool:
        denied = pool.submit(module.handler, value).result(timeout=5)
        accepted = pool.submit(fresh).result(timeout=5)
    assert denied["delivery_status"] == "qualification_missing"
    assert accepted["delivery_status"] == "delivered"
    assert len(runtime.calls) == 1
    assert [request["Key"] for request in ddb.calls] == [
        {"job_name": {"S": "PROMPT_WAKE_JOB#Processing#" + value["detail"]["ProcessingJobName"]}},
        {"job_name": {"S": "CAMPAIGN_WAKE_JOB#Processing#" + value["detail"]["ProcessingJobName"]}},
    ]


@pytest.mark.parametrize("failure", ["exception", "pid"])
def test_postinvoke_clock_or_pid_denial_keeps_original_cleanup(failure, monkeypatch):
    module, value, ddb, runtime, route = fixtures()
    now, broken = [1000.0], [False]
    pid = os.getpid()
    runtime.body = tracked_body(module, now)
    invoke = runtime.invoke_agent_runtime

    def returned(**kwargs):
        result = invoke(**kwargs)
        broken[0] = True
        return result

    def clock():
        if failure == "exception" and broken[0]:
            raise OSError("neutral clock unavailable")
        return now[0]

    runtime.invoke_agent_runtime = returned
    with module.campaign_callback_scope(route, dynamodb_client=ddb, runtime_client=runtime, clock=clock):
        if failure == "pid":
            monkeypatch.setattr(module.os, "getpid", lambda: pid + 1 if broken[0] else pid)
        result = module.handler(value)
    assert result["delivery_status"] == "outcome_unknown"
    assert runtime.body.closes == 1 and runtime.body.reads == 0
    assert len(runtime.calls) == 1


@pytest.mark.parametrize(
    "client_name,field", [("ddb", "connect_timeout"), ("ddb", "read_timeout"), ("runtime", "connect_timeout")]
)
def test_missing_transport_timeout_metadata_is_refused(client_name, field):
    module, value, ddb, runtime, route = fixtures()
    delattr((ddb if client_name == "ddb" else runtime).meta.config, field)
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "qualification_refused" and runtime.calls == []


def test_actual_lambda_package_import_with_optional_application_dependencies_unavailable():
    # Fresh ordinary source import only; no handler/provider call or image build.
    code = r"""
import importlib.abc, sys
class OptionalUnavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'yaml','numpy','pandas','torch','strands','bedrock_agentcore'} or fullname.startswith('amlas.config'):
            raise ModuleNotFoundError('optional application dependency unavailable: '+fullname)
sys.meta_path.insert(0,OptionalUnavailable())
import amlas.async_callback.lambda_handler as callback
assert callable(callback.handler)
assert callable(callback.encode_wake_job_binding)
assert not any(name.startswith('amlas.config') for name in sys.modules)
print('LAMBDA_SOURCE_IMPORT_OK')
"""
    run = subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=os.environ.copy(),
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "LAMBDA_SOURCE_IMPORT_OK"
