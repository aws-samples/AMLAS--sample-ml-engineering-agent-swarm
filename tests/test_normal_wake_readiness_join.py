"""Terminal hint retention from an actual normal-route producer; routing DATA only."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters
from test_campaign_callback import AgentCore, Body, QualifiedRoute, resume_state_bytes
from test_normal_run_sft_checkpoint import normal_fixture, start_original
from test_normal_run_sft_checkpoint import original_sft as original_sft
from test_normal_wake_publication import NAMESPACE, _observe_publication
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.async_callback import lambda_handler as callback_source

HINT_NAMESPACE = "PROMPT_TERMINAL_HINT#Training#"


def test_normal_terminal_callback_retains_exact_hint_before_delivery(monkeypatch, original_sft):
    async def scenario():
        value = original_sft
        _observe_publication(monkeypatch, value)
        reader = None
        try:
            reader, _answer, run_id = await start_original(value)
            with value.ports.lock:
                routes = [
                    copy.deepcopy(item)
                    for key, item in value.ports.items.items()
                    if key.startswith(NAMESPACE)
                ]
            assert len(routes) == 1 and len(value.native.calls) == 1
            route_item = routes[0]
            route_body = json.loads(route_item["record"]["S"])
            assert route_body["logical_run_id"] == run_id
            await close_original_worker(value.work, reader)
            reader = None
            ports = value.ports
            ddb = SimpleNamespace(
                meta=SimpleNamespace(
                    config=SimpleNamespace(
                        retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10
                    )
                ),
                get_item=ports.get_item,
                put_item=ports.put_item,
                describe_table=ports.describe_table,
            )
            runtime = AgentCore(resume_state="continuing")
            scope = QualifiedRoute(callback_source)
            runtime_arn = f"arn:aws:bedrock-agentcore:{route_body['region']}:{route_body['account_id']}:runtime/fixtureRuntime-ABCDEFGHIJ"
            for key, selected in {
                "AMLAS_CAMPAIGN_CALLBACK_MODE": "1",
                "AMLAS_AWS_ACCOUNT_ID": route_body["account_id"],
                "AWS_REGION": route_body["region"],
                "AMLAS_DEPLOYMENT_ID": route_body["deployment_id"],
                "JOBS_TABLE": normal_fixture.TABLE,
                "AGENTCORE_RUNTIME_ARN": runtime_arn,
                "AGENTCORE_RUNTIME_QUALIFIER": "NORMAL",
                "AGENTCORE_RUNTIME_VERSION": "3",
                "AMLAS_CALLBACK_ROUTE_SHA256": hashlib.sha256(
                    b"declared selected normal callback route"
                ).hexdigest(),
            }.items():
                monkeypatch.setenv(key, selected)
            event = {
                "version": "0",
                "id": "declared-normal-terminal-hint-1",
                "source": "aws.sagemaker",
                "account": route_body["account_id"],
                "region": route_body["region"],
                "detail-type": "SageMaker Training Job State Change",
                "resources": [route_body["job_arn"]],
                "detail": {
                    "TrainingJobName": route_body["job_name"],
                    "TrainingJobArn": route_body["job_arn"],
                    "TrainingJobStatus": "Completed",
                },
            }
            actual_invoke = runtime.invoke_agent_runtime
            ordered = []

            def invoke(**kwargs):
                with ports.lock:
                    hints = [
                        copy.deepcopy(item)
                        for key, item in ports.items.items()
                        if key.startswith(HINT_NAMESPACE)
                    ]
                ordered.append(hints)
                return actual_invoke(**kwargs)

            monkeypatch.setattr(runtime, "invoke_agent_runtime", invoke)
            before_model = value.model.calls
            with callback_source.campaign_callback_scope(
                scope, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
            ):
                result = callback_source.handler(event)
            assert len(runtime.calls) == 1, "actual normal callback delivery prerequisite was not reached"
            assert result["delivery_status"] == "delivered"
            with ports.lock:
                hints = [
                    copy.deepcopy(item) for key, item in ports.items.items() if key.startswith(HINT_NAMESPACE)
                ]
            assert len(hints) == 1, "actual normal terminal callback must retain its bounded immutable hint"
            hint = hints[0]
            assert set(hint) == {"job_name", "record", "record_sha256"}
            assert hint["job_name"] == {"S": HINT_NAMESPACE + route_body["job_name"]}
            raw = hint["record"]["S"]
            body = json.loads(raw)
            assert raw == json.dumps(
                body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            assert len(raw.encode()) <= 8192 and hint["record_sha256"] == {
                "S": hashlib.sha256(raw.encode()).hexdigest()
            }
            assert (
                body["logical_run_id"] == run_id
                and body["operation_digest"] == route_body["operation_digest"]
            )
            assert body["job_arn"] == route_body["job_arn"] and body["status"] == "Completed"
            assert ordered == [[hint]], "retention must precede delivery"
            assert len(runtime.calls) == 1
            assert runtime.calls[0]["runtimeSessionId"] == route_body["transport_session_id"]
            assert json.loads(runtime.calls[0]["payload"]) == {
                "action": "resume_prompt_run",
                "run_id": run_id,
            }
            assert result["lifecycle_verified"] is False and result["quality_success"] is False
            assert (
                value.model.calls == before_model
                and len(value.native.calls) == 1
                and value.native.stops == []
            )
        finally:
            if reader is not None:
                await close_original_worker(value.work, reader)

    asyncio.run(scenario())


# Every safety case creates its route through the same actual normal producer.
@asynccontextmanager
async def _fresh_normal_callback(monkeypatch, original_sft):
    value = original_sft
    _observe_publication(monkeypatch, value)
    reader = None
    try:
        reader, _answer, run_id = await start_original(value)
        with value.ports.lock:
            routes = [
                copy.deepcopy(item) for key, item in value.ports.items.items() if key.startswith(NAMESPACE)
            ]
        assert len(routes) == 1 and len(value.native.calls) == 1
        route_item = routes[0]
        route_body = json.loads(route_item["record"]["S"])
        assert route_body["logical_run_id"] == run_id
        await close_original_worker(value.work, reader)
        reader = None
        ports = value.ports
        ddb = SimpleNamespace(
            meta=SimpleNamespace(
                config=SimpleNamespace(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10)
            ),
            get_item=ports.get_item,
            put_item=ports.put_item,
            describe_table=ports.describe_table,
        )
        runtime = AgentCore(resume_state="continuing")
        scope = QualifiedRoute(callback_source)
        runtime_arn = f"arn:aws:bedrock-agentcore:{route_body['region']}:{route_body['account_id']}:runtime/fixtureRuntime-ABCDEFGHIJ"
        for key, selected in {
            "AMLAS_CAMPAIGN_CALLBACK_MODE": "1",
            "AMLAS_AWS_ACCOUNT_ID": route_body["account_id"],
            "AWS_REGION": route_body["region"],
            "AMLAS_DEPLOYMENT_ID": route_body["deployment_id"],
            "JOBS_TABLE": normal_fixture.TABLE,
            "AGENTCORE_RUNTIME_ARN": runtime_arn,
            "AGENTCORE_RUNTIME_QUALIFIER": "NORMAL",
            "AGENTCORE_RUNTIME_VERSION": "3",
            "AMLAS_CALLBACK_ROUTE_SHA256": hashlib.sha256(
                b"declared selected normal callback route"
            ).hexdigest(),
        }.items():
            monkeypatch.setenv(key, selected)
        event = {
            "version": "0",
            "id": "declared-normal-terminal-hint-1",
            "source": "aws.sagemaker",
            "account": route_body["account_id"],
            "region": route_body["region"],
            "detail-type": "SageMaker Training Job State Change",
            "resources": [route_body["job_arn"]],
            "detail": {
                "TrainingJobName": route_body["job_name"],
                "TrainingJobArn": route_body["job_arn"],
                "TrainingJobStatus": "Completed",
            },
        }
        yield SimpleNamespace(
            value=value,
            ports=ports,
            ddb=ddb,
            runtime=runtime,
            scope=scope,
            event=event,
            route_item=route_item,
            route_body=route_body,
            run_id=run_id,
            before_model=value.model.calls,
        )
    finally:
        if reader is not None:
            await close_original_worker(value.work, reader)


def _sdk_request(operation, values):
    validate_parameters(
        values,
        Session()
        .get_service_model("dynamodb")
        .operation_model(
            operation,
        )
        .input_shape,
    )


def _canonical_hint_item(item):
    assert set(item) == {"job_name", "record", "record_sha256"}
    raw = item["record"]["S"]
    body = json.loads(raw)
    assert raw == json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert len(raw.encode()) <= 8192
    assert item["record_sha256"] == {
        "S": hashlib.sha256(raw.encode()).hexdigest(),
    }
    return body


@pytest.mark.parametrize(
    "fault",
    (
        "duplicate_identical",
        "conflict",
        "denied_put",
        "lost_ack_exact",
        "unconfirmed_readback",
        "current_route_drift",
    ),
)
def test_fresh_normal_hint_retention_safety(monkeypatch, original_sft, fault):
    async def scenario():
        async with _fresh_normal_callback(monkeypatch, original_sft) as case:
            ports = case.ports
            hint_key = HINT_NAMESPACE + case.route_body["job_name"]
            route_key = case.route_item["job_name"]["S"]
            actual_put, actual_get = ports.put_item, ports.get_item
            puts, gets, route_reads, changed = [], [], [], []
            sequence = []

            def put_item(**values):
                if values["Item"]["job_name"]["S"] != hint_key:
                    return actual_put(**values)
                _sdk_request("PutItem", values)
                assert values["TableName"] == normal_fixture.TABLE
                assert values.get("ConditionExpression")
                body = _canonical_hint_item(values["Item"])
                assert set(body) == (set(case.route_body) - {"transport_session_id"}) | {
                    "status",
                    "normal_route_sha256",
                }
                assert body["record_kind"] == "normal_prompt_terminal_hint_v1"
                assert body["normal_route_sha256"] == (case.route_item["record_sha256"]["S"])
                assert body["logical_run_id"] == case.run_id
                assert body["operation_digest"] == case.route_body["operation_digest"]
                assert body["job_arn"] == case.route_body["job_arn"]
                assert body["status"] == "Completed"
                puts.append(copy.deepcopy(values))
                sequence.append("hint_put")
                if fault == "denied_put":
                    raise ClientError(
                        {"Error": {"Code": "AccessDeniedException"}},
                        "PutItem",
                    )
                if fault == "conflict":
                    conflict = copy.deepcopy(values["Item"])
                    body["status"] = "Failed"
                    raw = json.dumps(
                        body,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    conflict["record"] = {"S": raw}
                    conflict["record_sha256"] = {
                        "S": hashlib.sha256(raw.encode()).hexdigest(),
                    }
                    with ports.lock:
                        ports.items[hint_key] = conflict
                    changed.append(copy.deepcopy(conflict))
                    raise ClientError(
                        {"Error": {"Code": "ConditionalCheckFailedException"}},
                        "PutItem",
                    )
                response = actual_put(**values)
                if fault == "lost_ack_exact":
                    raise TimeoutError("Declared hint Put acknowledgement unavailable.")
                return response

            def get_item(**values):
                key = values["Key"]["job_name"]["S"]
                if key == route_key:
                    route_reads.append(copy.deepcopy(values))
                    sequence.append("route")
                if key != hint_key:
                    return actual_get(**values)
                _sdk_request("GetItem", values)
                assert values["TableName"] == normal_fixture.TABLE
                assert values["ConsistentRead"] is True
                gets.append(copy.deepcopy(values))
                sequence.append("hint_get")
                if fault == "unconfirmed_readback":
                    return {}
                response = actual_get(**values)
                if fault == "current_route_drift" and not changed:
                    assert response.get("Item") is not None
                    drift = copy.deepcopy(case.route_item)
                    body = json.loads(drift["record"]["S"])
                    body["operation_digest"] = hashlib.sha256(
                        b"declared concurrent normal route revision",
                    ).hexdigest()
                    raw = json.dumps(
                        body,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    drift["record"] = {"S": raw}
                    drift["record_sha256"] = {
                        "S": hashlib.sha256(raw.encode()).hexdigest(),
                    }
                    with ports.lock:
                        ports.items[route_key] = drift
                    changed.append(copy.deepcopy(drift))
                return response

            actual_describe = case.ddb.describe_table
            actual_invoke = case.runtime.invoke_agent_runtime

            def describe_table(**values):
                response = actual_describe(**values)
                sequence.append("table")
                return response

            def invoke(**values):
                sequence.append("invoke")
                return actual_invoke(**values)

            monkeypatch.setattr(case.ddb, "describe_table", describe_table)
            monkeypatch.setattr(case.runtime, "invoke_agent_runtime", invoke)
            monkeypatch.setattr(case.ddb, "put_item", put_item)
            monkeypatch.setattr(case.ddb, "get_item", get_item)
            retained = None
            with callback_source.campaign_callback_scope(
                case.scope,
                dynamodb_client=case.ddb,
                runtime_client=case.runtime,
                clock=lambda: 1000.0,
            ):
                if fault == "duplicate_identical":
                    first = callback_source.handler(case.event)
                    assert first["delivery_status"] == "delivered"
                    assert len(case.runtime.calls) == 1
                    with ports.lock:
                        retained = copy.deepcopy(ports.items.get(hint_key))
                    assert retained is not None
                    first_body = case.runtime.body
                    assert first_body.closed_called
                    # Every native Invoke response owns a distinct stream. Keep
                    # the shared request-validating fixture and its call history.
                    case.runtime.body = Body()
                    assert case.runtime.body is not first_body
                result = callback_source.handler(case.event)

            assert puts, "the actual normal immutable-hint Put was never reached"
            assert case.scope.valid is True
            for index, operation in enumerate(sequence):
                if operation not in ("hint_put", "hint_get"):
                    continue
                previous = max(
                    (i for i in range(index) if sequence[i] in ("hint_put", "hint_get", "invoke")), default=-1
                )
                assert {"route", "table"} <= set(sequence[previous + 1 : index])
            for index, operation in enumerate(sequence):
                if operation != "invoke":
                    continue
                previous = max(i for i in range(index) if sequence[i] in ("hint_put", "hint_get"))
                assert {"route", "table"} <= set(sequence[previous + 1 : index])
            if fault in ("duplicate_identical", "lost_ack_exact"):
                assert result["delivery_status"] == "delivered"
                assert gets, "delivery requires exact consistent hint readback"
                assert len(case.runtime.calls) == (2 if fault == "duplicate_identical" else 1)
                with ports.lock:
                    hint = copy.deepcopy(ports.items[hint_key])
                _canonical_hint_item(hint)
                assert hint == puts[0]["Item"]
                if retained is not None:
                    assert hint == retained
                    assert first_body.closed_called
                    assert case.runtime.body.closed_called
                else:
                    assert len(puts) == 1, "lost ACK must not replay Put"
                assert len(route_reads) >= 2
                for invocation in case.runtime.calls:
                    assert invocation["runtimeSessionId"] == (case.route_body["transport_session_id"])
                    assert json.loads(invocation["payload"]) == {
                        "action": "resume_prompt_run",
                        "run_id": case.run_id,
                    }
            else:
                assert case.runtime.calls == []
                assert result["delivery_status"] != "delivered"
                assert len(puts) == 1
                if fault == "conflict":
                    assert gets and len(changed) == 1
                    with ports.lock:
                        assert ports.items[hint_key] == changed[0]
                elif fault == "denied_put":
                    with ports.lock:
                        assert hint_key not in ports.items
                elif fault == "unconfirmed_readback":
                    assert gets
                else:
                    assert gets and len(changed) == 1
                    assert len(route_reads) >= 2
                    with ports.lock:
                        assert ports.items[route_key] == changed[0]
            assert result["lifecycle_verified"] is False
            assert result["quality_success"] is False
            assert case.value.model.calls == case.before_model
            assert len(case.value.native.calls) == 1
            assert case.value.native.stops == []

    asyncio.run(scenario())


def test_fresh_normal_route_drift_after_invoke_cannot_claim_delivery(
    monkeypatch,
    original_sft,
):
    async def scenario():
        async with _fresh_normal_callback(monkeypatch, original_sft) as case:
            ports = case.ports
            route_key = case.route_item["job_name"]["S"]
            # This test supplies and instruments the response body itself: the positive closed event.
            case.runtime.resume_state = None
            run_id = json.loads(case.route_item["record"]["S"])["logical_run_id"]
            case.runtime.body = Body(resume_state_bytes("continuing", run_id))
            actual_read = case.runtime.body.read
            changed = []

            def read(*args, **kwargs):
                response = actual_read(*args, **kwargs)
                if not changed:
                    assert len(case.runtime.calls) == 1
                    drift = copy.deepcopy(case.route_item)
                    body = json.loads(drift["record"]["S"])
                    body["operation_digest"] = hashlib.sha256(
                        b"declared late normal route revision",
                    ).hexdigest()
                    raw = json.dumps(
                        body,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    drift["record"] = {"S": raw}
                    drift["record_sha256"] = {
                        "S": hashlib.sha256(raw.encode()).hexdigest(),
                    }
                    with ports.lock:
                        ports.items[route_key] = drift
                    changed.append(copy.deepcopy(drift))
                return response

            monkeypatch.setattr(case.runtime.body, "read", read)
            with callback_source.campaign_callback_scope(
                case.scope,
                dynamodb_client=case.ddb,
                runtime_client=case.runtime,
                clock=lambda: 1000.0,
            ):
                result = callback_source.handler(case.event)
            assert len(changed) == 1, "the actual post-Invoke drain was not reached"
            assert len(case.runtime.calls) == 1, "late drift must not replay delivery"
            assert result["delivery_status"] != "delivered"
            assert case.runtime.body.closed_called
            assert case.scope.valid is True
            with ports.lock:
                assert ports.items[route_key] == changed[0]
            assert case.value.model.calls == case.before_model
            assert len(case.value.native.calls) == 1
            assert case.value.native.stops == []
            assert result["lifecycle_verified"] is False
            assert result["quality_success"] is False

    asyncio.run(scenario())
