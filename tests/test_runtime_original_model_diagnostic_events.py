"""Original SDK diagnostic DATA must remain visible while its owned worker waits."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest
from botocore.exceptions import ClientError
from test_model_acquisition_native_diagnostics import (
    PRIVATE_DETAIL,
    REQUEST_ID,
    call,
    expected,
    original_record,
    service_error,
)
from test_runtime_ambiguous_model_capture import attempted_original
from test_runtime_connected_warm_model import close_original_worker, connected_work, started_original

from amlas import app as runtime
from amlas.tui.chat import engine as engine_source

pytest_plugins = ("test_runtime_prompt_entry", "test_model_acquisition_native_diagnostics")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


@pytest.mark.parametrize(
    "code,status",
    [
        ("ValidationException", 400),
        ("ResourceInUse", 400),
        ("ResourceLimitExceeded", 400),
        ("ResourceNotFound", 400),
        ("ThrottlingException", 400),
        ("AccessDeniedException", 403),
    ],
)
def test_actual_sdk_known_native_label_retains_one_original_without_authority(monkeypatch, owned, code, status):
    # Create API/Boto3 model and SageMaker CommonErrors define finite labels only.
    # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html
    # https://docs.aws.amazon.com/sagemaker/latest/APIReference/CommonErrors.html
    backend, native, adapter = owned
    native.create_error = service_error("CreateProcessingJob", code=code, status=status)
    result = call(monkeypatch, backend, "original-submit")
    _, record = original_record(backend, native, adapter, result)
    assert result["accepted"] is False and result["status"] == "submission_unknown"
    assert record.handle.extra["accepted"] is False
    assert record.handle.extra["submission_ambiguous"] is True
    assert result.get("submission_diagnostic") == expected(
        "create_boundary", "ClientError", native_code=code, request_id=REQUEST_ID, status=status
    )
    encoded = json.dumps(result)
    assert PRIVATE_DETAIL not in encoded and "HTTPHeaders" not in encoded
    assert len(native.requests) == 1


def test_actual_sdk_unreviewed_native_label_stays_unknown(monkeypatch, owned):
    backend, native, adapter = owned
    native.create_error = service_error("CreateProcessingJob", code="UnreviewedNativeCode")
    result = call(monkeypatch, backend, "original-submit")
    original_record(backend, native, adapter, result)
    assert result["submission_diagnostic"] == expected(
        "create_boundary", "ClientError", request_id=REQUEST_ID, status=400
    )
    assert "UnreviewedNativeCode" not in json.dumps(result)


def source_http400_attempt(monkeypatch, work, *, change_pin=False):
    actual_create, actual_describe = work.native.create_processing_job, work.native.describe_processing_job

    def create(**kwargs):
        # The actual source adapter dispatches one request to the existing strict
        # native-input fake. Never inject positive lifecycle flags or a record.
        actual_create(**kwargs)
        raise service_error("CreateProcessingJob", code="ResourceLimitExceeded")

    def describe(**kwargs):
        actual_describe(**kwargs)
        if change_pin:
            backend = work.engines[0].tool_backend
            pending = backend._warm_pending
            pending.record.handle.extra["neutral_pin_changed"] = True
        raise ClientError(
            {"Error": {"Code": "ValidationException", "Message": PRIVATE_DETAIL}},
            "DescribeProcessingJob",
        )

    monkeypatch.setattr(work.native, "create_processing_job", create)
    monkeypatch.setattr(work.native, "describe_processing_job", describe)


async def read_events(work, events):
    async for event in runtime.agent_invocation({"prompt": work.literal}):
        events.append(event)
    return events


async def observed_while_active(work, events):
    deadline = asyncio.get_running_loop().time() + 5
    while len(work.observations) < 2 or sum(item["event"] == "heartbeat" for item in events) < 2:
        assert asyncio.get_running_loop().time() < deadline, "neutral original observation did not reach two polls"
        await asyncio.sleep(0.005)
    assert not work.api_returned.is_set()
    assert work.owned_app.get_async_task_info()["active_count"] == 1
    assert work.owned_app.get_current_ping_status().value == "HealthyBusy"


def diagnostic_events(events):
    return [item for item in events if item["event"] == "model_acquisition_diagnostic"]


def test_normal_active_app_stream_exposes_closed_original_create_diagnostic_without_replay(monkeypatch):
    work = connected_work(monkeypatch)
    source_http400_attempt(monkeypatch, work)

    async def scenario():
        events = []
        reader = asyncio.create_task(read_events(work, events))
        try:
            owned = await attempted_original(work)
            graph = owned.backend._warm_graph()
            pin = owned.backend._warm_handle_pin(owned.handle)
            messages = copy.deepcopy(work.agents[0].messages)
            await observed_while_active(work, events)
            assert not reader.done()
            emitted = diagnostic_events(events)
            assert len(emitted) == 1, "active normal prompt stream lost the genuine original Create diagnostic"
            event = emitted[0]
            assert event["diagnostic"] == expected(
                "create_boundary", "ClientError", native_code="ResourceLimitExceeded",
                request_id=REQUEST_ID, status=400,
            )
            assert event["worker_finished"] is False
            assert event["lifecycle_verified"] is False and event["quality_success"] is False
            assert {item["run_id"] for item in events} == {event["run_id"]}
            allowed = {
                "event", "diagnostic", "worker_finished", "mode", "run_id", "response_kind",
                "state_scope", "durable_resume", "approval_policy_source", "lifecycle_verified", "quality_success",
            }
            assert set(event) == allowed
            encoded = json.dumps(event)
            assert PRIVATE_DETAIL not in encoded and "HTTPHeaders" not in encoded
            assert owned.reference not in encoded and owned.handle.id not in encoded
            assert len(work.native.requests) == len(work.agents) == len(work.engines) == 1
            assert len(work.seen_prompts) == 1 and work.seen_prompts[0][0] == work.literal
            assert work.model.calls == 2
            assert owned.store._records[owned.reference] is owned.record
            assert owned.record.handle is owned.handle
            assert owned.handle.extra["accepted"] is False and owned.handle.extra["submission_ambiguous"] is True
            assert owned.backend._warm_handle_pin(owned.handle) == pin
            assert all(a is b for a, b in zip(graph, owned.backend._warm_graph(), strict=True))
            assert work.agents[0].messages == messages
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


@pytest.mark.parametrize("refusal", ["policy", "pin"])
def test_current_original_refusal_suppresses_pending_diagnostic_publication(monkeypatch, refusal):
    work = connected_work(monkeypatch, mode="policy" if refusal == "policy" else "normal")
    source_http400_attempt(monkeypatch, work, change_pin=refusal == "pin")

    async def scenario():
        events = []
        reader = asyncio.create_task(read_events(work, events))
        try:
            await asyncio.wait_for(reader, 5)
            assert diagnostic_events(events) == []
            assert len(work.native.requests) == len(work.originals) == len(work.engines) == len(work.agents) == 1
            backend = work.engines[0].tool_backend
            original = work.originals[0][1]
            handle = backend._model_acquisition_store._records[original["request_ref"]].handle
            assert handle.extra["accepted"] is False and handle.extra["submission_ambiguous"] is True
            assert work.model.calls == 2
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_public_diagnostic_without_genuine_original_cannot_emit_native_event(monkeypatch):
    work = connected_work(monkeypatch, mode="no_original")
    actual_engine_type = engine_source._WARM_CHAT_ENGINE_TYPE
    actual_process = actual_engine_type.process_message

    async def public_claim(engine, value, **kwargs):
        response = await actual_process(engine, value, **kwargs)
        response.data["submission_diagnostic"] = expected(
            "create_boundary", "ClientError", native_code="ResourceLimitExceeded", request_id=REQUEST_ID, status=400
        )
        return response

    monkeypatch.setattr(actual_engine_type, "process_message", public_claim)

    async def scenario():
        events = []
        await read_events(work, events)
        assert diagnostic_events(events) == []
        assert work.native.requests == [] and work.observations == [] and work.originals == []
        assert work.owned_app.get_async_task_info()["active_count"] == 0

    asyncio.run(scenario())


def test_normal_accepted_original_without_create_error_emits_no_failure_event(monkeypatch):
    work = connected_work(monkeypatch)

    async def scenario():
        events = []
        reader = asyncio.create_task(read_events(work, events))
        try:
            backend = await started_original(work)
            await observed_while_active(work, events)
            assert diagnostic_events(events) == []
            assert backend._warm_pending is not None
            assert len(work.native.requests) == len(work.agents) == 1
            assert work.model.calls == 2 and not reader.done()
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
