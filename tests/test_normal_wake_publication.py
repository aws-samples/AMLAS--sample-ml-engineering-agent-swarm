"""Fresh normal SFT routing publication order/CAS, not wake or native proof.

Use only the existing fresh normal SDK submission component and its mock upstream
DATA suppliers. Native/current qualifiers, original association, normal API,
SessionSftStore and registered SDK tool remain real Source paths. Strict external
DDB responses model conditional races/ACK loss; no old/cold snapshots, captured
recovery data, permission factories or TASK-ready/liveness claims are used.
Policy revocation in the existing finally helper is cleanup only.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from bedrock_agentcore.runtime import BedrockAgentCoreContext
from botocore.exceptions import ClientError
from test_normal_run_checkpoint import PROMPT
from test_normal_run_sft_checkpoint import MODEL_REF, PREP_REF, TOOL_USE, start_original
from test_normal_run_sft_checkpoint import original_sft as original_sft
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime
from amlas.control_plane.normal_run_checkpoint import current_normal_run

SDK_SESSION = "declared-original-normal-sdk-transport-session-0123456789"
TABLE_ID = "c4ef4b4a-c954-4a9e-99cc-e036a74353ba"
NAMESPACE = "PROMPT_WAKE_JOB#Training#"
FIELDS = {
    "schema_version",
    "record_kind",
    "logical_run_id",
    "transport_session_id",
    "account_id",
    "region",
    "deployment_id",
    "table_id",
    "kind",
    "job_name",
    "job_arn",
    "operation_digest",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _observe_publication(monkeypatch, value, *, mode="new"):
    ports = value.ports
    trace = SimpleNamespace(puts=[], gets=[], readbacks=[], effects=[], existing=None)
    actual_put, actual_get = ports.put_item, ports.get_item

    def is_route(kwargs, field):
        return kwargs[field]["job_name"]["S"].startswith(NAMESPACE)

    def put(**kwargs):
        if not is_route(kwargs, "Item"):
            return actual_put(**kwargs)
        trace.puts.append(copy.deepcopy(kwargs))
        item, key = kwargs["Item"], kwargs["Item"]["job_name"]["S"]
        if mode in ("identical", "conflict", "denied", "local_denied"):
            existing = copy.deepcopy(item)
            if mode == "conflict":
                body = json.loads(existing["record"]["S"])
                body["transport_session_id"] = "declared-conflicting-sdk-transport-session-0123456789"
                existing["record"]["S"] = _canonical(body)
                existing["record_sha256"]["S"] = hashlib.sha256(existing["record"]["S"].encode()).hexdigest()
            trace.existing = copy.deepcopy(existing)
            # Simulate a competing external writer with fresh prospective DATA.
            # This is not a seeded approval/context/SFT journal or native grant.
            with ports.lock:
                ports.items[key] = existing
        if mode == "local_denied":
            raise PermissionError("Declared local route Put denied before attempt.")
        if mode == "denied":
            ports._ddb("PutItem", kwargs)  # retain actual installed wire validation
            raise ClientError({"Error": {"Code": "AccessDeniedException"}}, "PutItem")
        response = actual_put(**kwargs)
        if mode in ("lost_ack_exact", "lost_ack_unresolved"):
            raise TimeoutError("Declared conditional route ACK is unavailable.")
        return response

    def get(**kwargs):
        if is_route(kwargs, "Key"):
            trace.gets.append(copy.deepcopy(kwargs))
            if mode == "lost_ack_unresolved" and trace.puts:
                ports._ddb("GetItem", kwargs)
                raise TimeoutError("Declared consistent route readback is unavailable.")
        response = actual_get(**kwargs)
        if is_route(kwargs, "Key"):
            trace.readbacks.append(copy.deepcopy(response))
        return response

    def effect_record(operation):
        context = current_normal_run()
        with ports.lock:
            routes = {
                key: copy.deepcopy(item) for key, item in ports.items.items() if key.startswith(NAMESPACE)
            }
            normal = ports.items.get("PROMPT_RUN#" + context.run_id) if context is not None else None
            association = json.loads(normal["record"]["S"]).get("sft") if normal is not None else None
        record = {
            "operation": operation,
            "routes": routes,
            "association": association,
            "readbacks": copy.deepcopy(trace.readbacks),
        }
        if context is not None:
            record.update(
                run_id=context.run_id,
                session_hash=context.session_hash,
                sdk_session=BedrockAgentCoreContext.get_session_id(),
                private_correlation=value.work.engines[0].tool_backend._sft_store.runtime_session_id,
                account_id=context.catalog.account_id,
                region=context.catalog.region,
                deployment_id=context.catalog.deployment_id,
            )
        trace.effects.append(record)

    def observe(operation, actual):
        def call(**kwargs):
            effect_record(operation)
            return actual(**kwargs)

        return call

    monkeypatch.setattr(ports, "put_item", put)
    monkeypatch.setattr(ports, "get_item", get)
    for name in ("put_object", "create_multipart_upload", "upload_part_copy", "complete_multipart_upload"):
        monkeypatch.setattr(value.s3, name, observe(name, getattr(value.s3, name)))
    monkeypatch.setattr(
        value.native, "create_training_job", observe("CreateTrainingJob", value.native.create_training_job)
    )
    BedrockAgentCoreContext.set_request_context("declared-publication-request", SDK_SESSION)
    return trace


def _assert_one_conditional_attempt(trace):
    assert len(trace.puts) == 1, "normal SFT requires one prospective conditional routing Put"
    put = trace.puts[0]
    assert put.get("ConditionExpression")
    assert trace.gets and all(get["ConsistentRead"] is True for get in trace.gets)
    assert all(get["Key"] == {"job_name": put["Item"]["job_name"]} for get in trace.gets)


def _assert_exact_route_before_effects(value, trace, original, run_id):
    _assert_one_conditional_attempt(trace)
    assert trace.effects and any(effect["operation"] != "CreateTrainingJob" for effect in trace.effects)
    name = value.native.calls[0]["TrainingJobName"]
    key = NAMESPACE + name
    expected_arn = original["job_arn"]
    for effect in trace.effects:
        assert key in effect["routes"], "normal route must exist before first SFT staging/Create"
        item = effect["routes"][key]
        assert set(item) == {"job_name", "record", "record_sha256"}
        assert item["job_name"] == {"S": key}
        raw = item["record"]["S"]
        body = json.loads(raw)
        assert set(body) == FIELDS and type(body["schema_version"]) is int
        assert body == {
            "schema_version": 1,
            "record_kind": "normal_prompt_wake_job_v1",
            "logical_run_id": run_id,
            "transport_session_id": SDK_SESSION,
            "account_id": effect["account_id"],
            "region": effect["region"],
            "deployment_id": effect["deployment_id"],
            "table_id": TABLE_ID,
            "kind": "Training",
            "job_name": name,
            "job_arn": expected_arn,
            "operation_digest": effect["association"]["operation_digest"],
        }
        assert raw == _canonical(body) and len(raw.encode()) <= 8192
        assert item["record_sha256"] == {"S": hashlib.sha256(raw.encode()).hexdigest()}
        assert item == trace.puts[0]["Item"]
        assert any(readback.get("Item") == item for readback in effect["readbacks"]), (
            "exact consistent full-item confirmation must precede every SFT effect"
        )
        assert effect["association"]["native_job_name"] == name
        assert effect["association"]["native_job_arn"] == expected_arn
        assert effect["run_id"] == run_id and effect["sdk_session"] == SDK_SESSION
        assert SDK_SESSION != effect["private_correlation"]
        assert effect["session_hash"] == hashlib.sha256(SDK_SESSION.encode()).hexdigest()
    assert len(value.native.calls) == 1 and value.native.stops == []


def test_normal_sft_publishes_confirmed_exact_route_before_staging_and_create(monkeypatch, original_sft):
    value = original_sft
    trace = _observe_publication(monkeypatch, value)

    async def scenario():
        reader = None
        try:
            reader, original, run_id = await start_original(value)
            _assert_exact_route_before_effects(value, trace, original, run_id)
        finally:
            if reader is not None:
                await close_original_worker(value.work, reader)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["denied", "conflict", "lost_ack_unresolved", "lost_ack_exact", "identical", "local_denied"])
def test_normal_route_conditional_races_never_overwrite_or_blindly_retry(monkeypatch, original_sft, mode):
    value = original_sft
    trace = _observe_publication(monkeypatch, value, mode=mode)

    async def scenario():
        events = []

        async def read():
            async for event in runtime.agent_invocation({"prompt": PROMPT}):
                events.append(event)

        reader = asyncio.create_task(read())
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not value.work.originals and not reader.done():
                assert asyncio.get_running_loop().time() < deadline, (
                    "registered SFT did not return its declared result"
                )
                await asyncio.sleep(0.005)
            original = next((answer for use, answer in value.work.originals if use == TOOL_USE), None)
            assert original is not None, "actual normal SDK SFT result boundary is required"
            assert len(value.work.engines) == len(value.work.agents) == 1
            assert set(value.sources) == {("model", MODEL_REF), ("preparation", PREP_REF)}
            assert value.work.guard.calls and all(
                call["source"] == "INPUT" for call in value.work.guard.calls
            )
            assert len(trace.puts) == 1, "normal SFT must attempt its prospective routing publication once"
            if mode in ("lost_ack_exact", "identical"):
                assert original.get("accepted") is True
                run_id = next(event["run_id"] for event in events if event["event"] == "run_started")
                _assert_exact_route_before_effects(value, trace, original, run_id)
            else:
                assert original.get("accepted") is not True
                assert trace.effects == [] and value.native.calls == [] and value.native.stops == []
                assert value.s3.mutations == []
            if mode == "local_denied":
                assert trace.gets == [] and trace.readbacks == []
                assert not any(
                    operation == "PutItem" and kwargs["Item"]["job_name"]["S"].startswith(NAMESPACE)
                    for operation, kwargs in value.ports.calls
                )
            if trace.existing is not None:
                key = trace.existing["job_name"]["S"]
                assert value.ports.items[key] == trace.existing
        finally:
            await close_original_worker(value.work, reader)

    asyncio.run(scenario())
