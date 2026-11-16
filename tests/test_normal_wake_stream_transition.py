"""Stream images select routing DATA, never TASK or execution authority."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from botocore.session import Session
from botocore.validate import validate_parameters
from normal_sft_cold_fixture import close, setup
from test_campaign_callback import AgentCore, Body, QualifiedRoute
from test_normal_prepared_promotion import _fresh_unpromoted_original
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.async_callback import lambda_handler as callback_source
from amlas.control_plane.normal_run_checkpoint import activate_normal_run

HINT_NAMESPACE = "PROMPT_TERMINAL_HINT#Training#"
ROUTE_NAMESPACE = "PROMPT_WAKE_JOB#Training#"


def _stream_record(region, stream_arn, old_item, new_item):
    record = {
        "eventID": "declared-normal-task-transition-1",
        "eventName": "MODIFY",
        "eventVersion": "1.1",
        "eventSource": "aws:dynamodb",
        "awsRegion": region,
        "dynamodb": {
            "ApproximateCreationDateTime": 946684801.0,
            "Keys": {"job_name": copy.deepcopy(new_item["job_name"])},
            "OldImage": copy.deepcopy(old_item),
            "NewImage": copy.deepcopy(new_item),
            "SequenceNumber": "100000000000000000001",
            "SizeBytes": len(json.dumps(new_item).encode()),
            "StreamViewType": "NEW_AND_OLD_IMAGES",
        },
    }
    model = Session().get_service_model("dynamodbstreams")
    shape = model.operation_model("GetRecords").output_shape.members["Records"].member
    validate_parameters(record, shape)
    # Lambda adds this native-source binding outside the Streams API record shape.
    record["eventSourceARN"] = stream_arn
    return record


def test_actual_task_publication_stream_re_delivers_retained_original_hint(
    monkeypatch,
    tmp_path,
):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            context, policy, original, run_key, _before, _prepared_raw = await _fresh_unpromoted_original(
                state, monkeypatch
            )
            with state.ports.lock:
                routes = [
                    copy.deepcopy(item)
                    for key, item in state.ports.items.items()
                    if key.startswith(ROUTE_NAMESPACE)
                ]
                before = copy.deepcopy(state.ports.items[run_key])
            assert len(routes) == 1
            route_item = routes[0]
            route = json.loads(route_item["record"]["S"])
            assert route["logical_run_id"] == context.run_id
            assert route["job_name"] == original.id
            assert "task_checkpoint_ref" not in json.loads(before["record"]["S"])
            stream_arn = (
                f"arn:aws:dynamodb:{route['region']}:{route['account_id']}:table/"
                f"{context.catalog.jobs_table}/stream/2000-01-01T00:00:00.000"
            )
            actual_describe = state.ports.describe_table
            descriptions = []

            def describe_table(**values):
                response = copy.deepcopy(actual_describe(**values))
                response["Table"]["StreamSpecification"] = {
                    "StreamEnabled": True,
                    "StreamViewType": "NEW_AND_OLD_IMAGES",
                }
                response["Table"]["LatestStreamArn"] = stream_arn
                response["Table"]["LatestStreamLabel"] = "2000-01-01T00:00:00.000"
                output = (
                    Session()
                    .get_service_model("dynamodb")
                    .operation_model(
                        "DescribeTable",
                    )
                    .output_shape
                )
                assert set(response) <= set(output.members) | {"ResponseMetadata"}
                validate_parameters(
                    {name: value for name, value in response.items() if name in output.members}, output
                )
                descriptions.append(copy.deepcopy(values))
                return response

            ddb = SimpleNamespace(
                meta=SimpleNamespace(
                    config=SimpleNamespace(
                        retries={"total_max_attempts": 1},
                        connect_timeout=5,
                        read_timeout=10,
                    )
                ),
                get_item=state.ports.get_item,
                put_item=state.ports.put_item,
                describe_table=describe_table,
            )
            runtime = AgentCore(resume_state="continuing")
            qualifier = QualifiedRoute(callback_source)
            runtime_arn = (
                f"arn:aws:bedrock-agentcore:{route['region']}:{route['account_id']}:"
                "runtime/fixtureRuntime-ABCDEFGHIJ"
            )
            for name, value in {
                "AMLAS_CAMPAIGN_CALLBACK_MODE": "1",
                "AMLAS_AWS_ACCOUNT_ID": route["account_id"],
                "AWS_REGION": route["region"],
                "AMLAS_DEPLOYMENT_ID": route["deployment_id"],
                "JOBS_TABLE": context.catalog.jobs_table,
                "AGENTCORE_RUNTIME_ARN": runtime_arn,
                "AGENTCORE_RUNTIME_QUALIFIER": "NORMAL",
                "AGENTCORE_RUNTIME_VERSION": "3",
                "AMLAS_CALLBACK_ROUTE_SHA256": hashlib.sha256(
                    b"declared selected normal callback route",
                ).hexdigest(),
            }.items():
                monkeypatch.setenv(name, value)
            terminal = {
                "version": "0",
                "id": "declared-normal-terminal-before-task-1",
                "source": "aws.sagemaker",
                "account": route["account_id"],
                "region": route["region"],
                "detail-type": "SageMaker Training Job State Change",
                "resources": [route["job_arn"]],
                "detail": {
                    "TrainingJobName": route["job_name"],
                    "TrainingJobArn": route["job_arn"],
                    "TrainingJobStatus": "Completed",
                },
            }
            with callback_source.campaign_callback_scope(
                qualifier,
                dynamodb_client=ddb,
                runtime_client=runtime,
                clock=lambda: 1000.0,
            ):
                first = callback_source.handler(terminal)
            assert first["delivery_status"] == "delivered"
            assert len(runtime.calls) == 1 and runtime.body.closed_called
            hint_key = HINT_NAMESPACE + original.id
            with state.ports.lock:
                retained_hint = copy.deepcopy(state.ports.items.get(hint_key))
            assert retained_hint is not None
            model_calls = state.model.calls
            native_calls = copy.deepcopy(state.native.calls)
            with activate_normal_run(context):
                context.promote_prepared_task(policy=policy)
            with state.ports.lock:
                after = copy.deepcopy(state.ports.items[run_key])
            before_body = json.loads(before["record"]["S"])
            after_body = json.loads(after["record"]["S"])
            assert "task_checkpoint_ref" in after_body
            assert after_body == {
                **before_body,
                "task_checkpoint_ref": after_body["task_checkpoint_ref"],
            }
            assert before["job_name"] == after["job_name"] == {"S": run_key}
            record = _stream_record(route["region"], stream_arn, before, after)
            runtime.body = Body()
            with callback_source.campaign_callback_scope(
                qualifier,
                dynamodb_client=ddb,
                runtime_client=runtime,
                clock=lambda: 1000.0,
            ):
                callback_source.handler({"Records": [record]})
            # Freeze all actual producer/hint prerequisites before the intended RED.
            with state.ports.lock:
                assert state.ports.items[hint_key] == retained_hint
                assert state.ports.items[run_key] == after
            assert descriptions and qualifier.valid is True
            assert state.model.calls == model_calls
            assert state.native.calls == native_calls and state.native.stops == []
            assert state.loaded == [] and state.invoked == []
            assert len(runtime.calls) == 2, (
                "actual TASK publication stream must route the retained original hint"
            )
            for invocation in runtime.calls:
                assert invocation["runtimeSessionId"] == route["transport_session_id"]
                assert json.loads(invocation["payload"]) == {
                    "action": "resume_prompt_run",
                    "run_id": context.run_id,
                }
            assert runtime.body.closed_called
        finally:
            await close(state)

    asyncio.run(scenario())


@asynccontextmanager
async def _fresh_actual_task_stream(monkeypatch, tmp_path):
    state = setup(monkeypatch, tmp_path)
    try:
        context, policy, original, run_key, _before, _prepared_raw = await _fresh_unpromoted_original(
            state, monkeypatch
        )
        with state.ports.lock:
            routes = [
                copy.deepcopy(item)
                for key, item in state.ports.items.items()
                if key.startswith(ROUTE_NAMESPACE)
            ]
            before = copy.deepcopy(state.ports.items[run_key])
        assert len(routes) == 1
        route_item = routes[0]
        route = json.loads(route_item["record"]["S"])
        assert route["logical_run_id"] == context.run_id
        assert route["job_name"] == original.id
        assert "task_checkpoint_ref" not in json.loads(before["record"]["S"])
        stream_arn = f"arn:aws:dynamodb:{route['region']}:{route['account_id']}:table/{context.catalog.jobs_table}/stream/2000-01-01T00:00:00.000"
        actual_describe = state.ports.describe_table
        descriptions = []

        def describe_table(**values):
            response = copy.deepcopy(actual_describe(**values))
            response["Table"]["StreamSpecification"] = {
                "StreamEnabled": True,
                "StreamViewType": "NEW_AND_OLD_IMAGES",
            }
            response["Table"]["LatestStreamArn"] = stream_arn
            response["Table"]["LatestStreamLabel"] = "2000-01-01T00:00:00.000"
            output = Session().get_service_model("dynamodb").operation_model("DescribeTable").output_shape
            assert set(response) <= set(output.members) | {"ResponseMetadata"}
            validate_parameters(
                {name: value for name, value in response.items() if name in output.members}, output
            )
            descriptions.append(copy.deepcopy(values))
            return response

        ddb = SimpleNamespace(
            meta=SimpleNamespace(
                config=SimpleNamespace(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10)
            ),
            get_item=state.ports.get_item,
            put_item=state.ports.put_item,
            describe_table=describe_table,
        )
        runtime = AgentCore(resume_state="continuing")
        qualifier = QualifiedRoute(callback_source)
        runtime_arn = f"arn:aws:bedrock-agentcore:{route['region']}:{route['account_id']}:runtime/fixtureRuntime-ABCDEFGHIJ"
        for name, value in {
            "AMLAS_CAMPAIGN_CALLBACK_MODE": "1",
            "AMLAS_AWS_ACCOUNT_ID": route["account_id"],
            "AWS_REGION": route["region"],
            "AMLAS_DEPLOYMENT_ID": route["deployment_id"],
            "JOBS_TABLE": context.catalog.jobs_table,
            "AGENTCORE_RUNTIME_ARN": runtime_arn,
            "AGENTCORE_RUNTIME_QUALIFIER": "NORMAL",
            "AGENTCORE_RUNTIME_VERSION": "3",
            "AMLAS_CALLBACK_ROUTE_SHA256": hashlib.sha256(
                b"declared selected normal callback route"
            ).hexdigest(),
        }.items():
            monkeypatch.setenv(name, value)
        terminal = {
            "version": "0",
            "id": "declared-normal-terminal-before-task-1",
            "source": "aws.sagemaker",
            "account": route["account_id"],
            "region": route["region"],
            "detail-type": "SageMaker Training Job State Change",
            "resources": [route["job_arn"]],
            "detail": {
                "TrainingJobName": route["job_name"],
                "TrainingJobArn": route["job_arn"],
                "TrainingJobStatus": "Completed",
            },
        }
        with callback_source.campaign_callback_scope(
            qualifier, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
        ):
            first = callback_source.handler(terminal)
        assert first["delivery_status"] == "delivered"
        assert len(runtime.calls) == 1 and runtime.body.closed_called
        hint_key = HINT_NAMESPACE + original.id
        with state.ports.lock:
            retained_hint = copy.deepcopy(state.ports.items.get(hint_key))
        assert retained_hint is not None
        model_calls = state.model.calls
        native_calls = copy.deepcopy(state.native.calls)
        with activate_normal_run(context):
            context.promote_prepared_task(policy=policy)
        with state.ports.lock:
            after = copy.deepcopy(state.ports.items[run_key])
        before_body = json.loads(before["record"]["S"])
        after_body = json.loads(after["record"]["S"])
        assert "task_checkpoint_ref" in after_body
        assert after_body == {**before_body, "task_checkpoint_ref": after_body["task_checkpoint_ref"]}
        assert before["job_name"] == after["job_name"] == {"S": run_key}
        record = _stream_record(route["region"], stream_arn, before, after)
        yield SimpleNamespace(
            state=state,
            context=context,
            original=original,
            route=route,
            route_item=route_item,
            stream_arn=stream_arn,
            ddb=ddb,
            runtime=runtime,
            qualifier=qualifier,
            record=record,
            before=before,
            after=after,
            run_key=run_key,
            hint_key=hint_key,
            retained_hint=retained_hint,
            model_calls=model_calls,
            native_calls=native_calls,
        )
    finally:
        await close(state)


def _record_image_with_declared_data(item, field):
    changed = copy.deepcopy(item)
    body = json.loads(changed["record"]["S"])
    if field == "history":
        body["history"].append({"note": "declared history DATA"})
    else:
        body[field] = {"note": "declared bookkeeping DATA"}
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    changed["record"] = {"S": raw}
    changed["record_sha256"] = {"S": hashlib.sha256(raw.encode()).hexdigest()}
    return changed


def _deliver_stream(case, records):
    # Repeated native responses must each own a fresh stream. Request validation,
    # current qualification and the real callback implementation remain intact.
    case.runtime.body = Body()
    with callback_source.campaign_callback_scope(
        case.qualifier,
        dynamodb_client=case.ddb,
        runtime_client=case.runtime,
        clock=lambda: 1000.0,
    ):
        return callback_source.handler({"Records": records})


def _assert_original_unchanged(case):
    assert case.qualifier.valid is True
    assert case.state.model.calls == case.model_calls
    assert case.state.native.calls == case.native_calls
    assert case.state.native.stops == []
    assert case.state.loaded == [] and case.state.invoked == []
    with case.state.ports.lock:
        assert case.state.ports.items[case.run_key] == case.after
        assert case.state.ports.items[case.hint_key] == case.retained_hint
        assert case.state.ports.items == case.rows


@pytest.mark.parametrize(
    "control",
    (
        "non_readiness",
        "unrelated_bookkeeping",
        "native_binding",
        "current_data_hold",
        "duplicate_delivery",
        "current_bookkeeping",
    ),
)
def test_fresh_actual_task_stream_routing_controls(monkeypatch, tmp_path, control):
    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            with case.state.ports.lock:
                case.rows = copy.deepcopy(case.state.ports.items)
            baseline = len(case.runtime.calls)
            assert baseline == 1
            if control == "non_readiness":
                for field in (None, "history", "declared_claim_data", "declared_report_data"):
                    new = (
                        copy.deepcopy(case.after)
                        if field is None
                        else _record_image_with_declared_data(case.after, field)
                    )
                    record = _stream_record(
                        case.route["region"],
                        case.stream_arn,
                        case.after,
                        new,
                    )
                    _deliver_stream(case, [record])
                    assert len(case.runtime.calls) == baseline
                    _assert_original_unchanged(case)
            elif control == "unrelated_bookkeeping":
                with case.state.ports.lock:
                    hint = copy.deepcopy(case.state.ports.items[case.hint_key])
                records = [
                    _stream_record(
                        case.route["region"],
                        case.stream_arn,
                        hint,
                        _record_image_with_declared_data(hint, "declared_bookkeeping_data"),
                    ),
                ]
                unrelated = copy.deepcopy(case.record)
                unrelated["dynamodb"]["Keys"] = {
                    "job_name": {"S": "DECLARED_UNRELATED_STREAM_ROW"},
                }
                for image in ("OldImage", "NewImage"):
                    unrelated["dynamodb"][image]["job_name"] = {
                        "S": "DECLARED_UNRELATED_STREAM_ROW",
                    }
                records.append(unrelated)
                _deliver_stream(case, records)
                assert len(case.runtime.calls) == baseline
                _assert_original_unchanged(case)
            elif control == "native_binding":
                actual_describe = case.ddb.describe_table
                for fault in (
                    "wrong_stream",
                    "missing_stream",
                    "foreign_stream",
                    "disabled",
                    "wrong_view",
                    "replaced_table",
                ):
                    record = copy.deepcopy(case.record)

                    def describe_table(_fault=fault, **values):
                        response = copy.deepcopy(actual_describe(**values))
                        table = response["Table"]
                        if _fault == "missing_stream":
                            table.pop("LatestStreamArn")
                        elif _fault == "disabled":
                            table["StreamSpecification"]["StreamEnabled"] = False
                        elif _fault == "wrong_view":
                            table["StreamSpecification"]["StreamViewType"] = "KEYS_ONLY"
                        elif _fault == "replaced_table":
                            table["TableId"] = "12345678-1234-4234-9234-00000000dd99"
                        return response

                    if fault == "wrong_stream":
                        record["eventSourceARN"] += "-declared-other"
                    elif fault == "foreign_stream":
                        record["eventSourceARN"] = case.stream_arn.replace(
                            ":" + case.route["account_id"] + ":",
                            ":000000000000:",
                            1,
                        )
                    monkeypatch.setattr(case.ddb, "describe_table", describe_table)
                    with pytest.raises(
                        RuntimeError,
                        match=r"^normal readiness delivery unresolved$",
                    ):
                        _deliver_stream(case, [record])
                    assert len(case.runtime.calls) == baseline
                    _assert_original_unchanged(case)
                monkeypatch.setattr(case.ddb, "describe_table", actual_describe)
            elif control == "current_data_hold":
                ports = case.state.ports
                route_key = case.route_item["job_name"]["S"]
                for fault in ("missing_route", "missing_hint", "route_conflict", "hint_conflict"):
                    with ports.lock:
                        if fault == "missing_route":
                            ports.items.pop(route_key)
                        elif fault == "missing_hint":
                            ports.items.pop(case.hint_key)
                        else:
                            key = route_key if fault == "route_conflict" else case.hint_key
                            changed = copy.deepcopy(ports.items[key])
                            body = json.loads(changed["record"]["S"])
                            field = "operation_digest" if fault == "route_conflict" else "normal_route_sha256"
                            body[field] = hashlib.sha256(
                                b"declared current routing conflict",
                            ).hexdigest()
                            raw = json.dumps(
                                body,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            changed["record"] = {"S": raw}
                            changed["record_sha256"] = {
                                "S": hashlib.sha256(raw.encode()).hexdigest(),
                            }
                            ports.items[key] = changed
                    with ports.lock:
                        held_rows = copy.deepcopy(ports.items)
                    with pytest.raises(
                        RuntimeError,
                        match=r"^normal readiness delivery unresolved$",
                    ):
                        _deliver_stream(case, [copy.deepcopy(case.record)])
                    assert len(case.runtime.calls) == baseline
                    with ports.lock:
                        assert ports.items == held_rows
                    # Only restore declared external DATA for the next bounded fault.
                    with ports.lock:
                        ports.items[route_key] = copy.deepcopy(case.route_item)
                        ports.items[case.hint_key] = copy.deepcopy(case.retained_hint)
                    _assert_original_unchanged(case)
            elif control == "current_bookkeeping":
                changed = _record_image_with_declared_data(case.after, "history")
                producer_body = json.loads(case.after["record"]["S"])
                changed_body = json.loads(changed["record"]["S"])
                assert changed_body == {
                    **producer_body,
                    "history": changed_body["history"],
                }
                assert changed_body["history"] == producer_body["history"] + [
                    {"note": "declared history DATA"},
                ]
                assert (
                    changed_body["task_checkpoint_ref"]
                    == (json.loads(case.record["dynamodb"]["NewImage"]["record"]["S"])["task_checkpoint_ref"])
                )
                assert changed["job_name"] == case.after["job_name"]
                expected_rows = copy.deepcopy(case.rows)
                expected_rows[case.run_key] = changed
                with case.state.ports.lock:
                    case.state.ports.items[case.run_key] = copy.deepcopy(changed)
                _deliver_stream(case, [copy.deepcopy(case.record)])
                with case.state.ports.lock:
                    assert case.state.ports.items == expected_rows
                    assert case.state.ports.items[case.hint_key] == case.retained_hint
                assert case.qualifier.valid is True
                assert case.state.model.calls == case.model_calls
                assert case.state.native.calls == case.native_calls
                assert case.state.native.stops == []
                assert case.state.loaded == [] and case.state.invoked == []
                assert len(case.runtime.calls) == baseline + 1, (
                    "current history bookkeeping must not invalidate a real TASK transition"
                )
                assert case.runtime.calls[-1]["runtimeSessionId"] == (case.route["transport_session_id"])
                assert json.loads(case.runtime.calls[-1]["payload"]) == {
                    "action": "resume_prompt_run",
                    "run_id": case.context.run_id,
                }
                assert case.runtime.body.closed_called
            else:
                with case.state.ports.lock:
                    before_rows = copy.deepcopy(case.state.ports.items)
                bodies = []
                for _ in range(2):
                    _deliver_stream(case, [copy.deepcopy(case.record)])
                    bodies.append(case.runtime.body)
                    assert case.runtime.body.closed_called
                    _assert_original_unchanged(case)
                assert len(case.runtime.calls) == baseline + 2
                assert bodies[0] is not bodies[1]
                with case.state.ports.lock:
                    assert case.state.ports.items == before_rows
                for invocation in case.runtime.calls[baseline:]:
                    assert invocation["runtimeSessionId"] == case.route["transport_session_id"]
                    assert json.loads(invocation["payload"]) == {
                        "action": "resume_prompt_run",
                        "run_id": case.context.run_id,
                    }
                # The inert receiver establishes routing only, not effect fencing.

    asyncio.run(scenario())


_ABSENT = object()


def _run_image(item, **changes):
    changed = copy.deepcopy(item)
    body = json.loads(changed["record"]["S"])
    for name, value in changes.items():
        if value is _ABSENT:
            body.pop(name, None)
        else:
            body[name] = value
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    changed["record"] = {"S": raw}
    changed["record_sha256"] = {"S": hashlib.sha256(raw.encode()).hexdigest()}
    return changed


def _enrollment_record(case, event_name):
    """PROMPT_RUN#<run> as the product writes it at enrollment: no SFT association yet."""
    enrolled = _run_image(case.before, stage="enrolled", sft=_ABSENT)
    assert json.loads(enrolled["record"]["S"])["stage"] == "enrolled"
    if event_name == "INSERT":
        record = _stream_record(case.route["region"], case.stream_arn, enrolled, enrolled)
        record["eventName"] = "INSERT"
        record["dynamodb"].pop("OldImage")
    else:
        history = json.loads(enrolled["record"]["S"])["history"] + [{"note": "declared enrollment DATA"}]
        record = _stream_record(
            case.route["region"], case.stream_arn, enrolled, _run_image(enrolled, history=history)
        )
    record["eventID"] = "declared-normal-enrollment-1"
    record["dynamodb"]["SequenceNumber"] = "100000000000000000000"
    return record


@pytest.mark.parametrize("event_name", ["INSERT", "MODIFY"])
def test_enrollment_run_row_is_not_readiness_and_never_holds_the_stream(monkeypatch, tmp_path, event_name):
    """Enrollment precedes any SFT association, so it is ignored rather than refused.

    A refused row under an unlimited-retry stream mapping holds every later record of
    its shard, including the same run's TASK transition (seen live twice on 2026-11-06).
    """

    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            with case.state.ports.lock:
                case.rows = copy.deepcopy(case.state.ports.items)
            baseline = len(case.runtime.calls)
            result = _deliver_stream(case, [_enrollment_record(case, event_name)])
            assert result["statusCode"] == 200 and result["delivery_status"] == "ignored"
            assert len(case.runtime.calls) == baseline
            _assert_original_unchanged(case)

    asyncio.run(scenario())


def test_enrollment_row_ahead_of_task_transition_does_not_block_delivery(monkeypatch, tmp_path):
    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            baseline = len(case.runtime.calls)
            result = _deliver_stream(case, [_enrollment_record(case, "INSERT"), copy.deepcopy(case.record)])
            assert result["statusCode"] == 200 and result["delivery_status"] == "delivered"
            assert len(case.runtime.calls) == baseline + 1
            assert case.runtime.calls[-1]["runtimeSessionId"] == case.route["transport_session_id"]
            assert json.loads(case.runtime.calls[-1]["payload"]) == {
                "action": "resume_prompt_run",
                "run_id": case.context.run_id,
            }
            assert case.runtime.body.closed_called

    asyncio.run(scenario())


def test_unknown_run_stage_still_refuses(monkeypatch, tmp_path):
    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            with case.state.ports.lock:
                case.rows = copy.deepcopy(case.state.ports.items)
            baseline = len(case.runtime.calls)
            other = _run_image(case.after, stage="declared-other")
            record = _stream_record(case.route["region"], case.stream_arn, case.after, other)
            with pytest.raises(RuntimeError, match=r"^normal readiness delivery unresolved$"):
                _deliver_stream(case, [record])
            assert len(case.runtime.calls) == baseline
            _assert_original_unchanged(case)

    asyncio.run(scenario())
