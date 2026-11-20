"""Closed refusal DATA from actual SDK/original stores; neutral AWS-only fixtures."""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest
from test_runtime_connected_warm_model import connected_work
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401
from test_runtime_warm_model_continuation import complete_fixture, original_work, step

from amlas import app as runtime
from amlas.control_plane.approvals import ApprovalPolicy
from amlas.tui.chat import tool_binding_backend as backend_source

PRIVATE_DETAIL = "ordinary private read detail"


def corrupt_one_raw_payload(work):
    key = next(key for key in work.s3.objects if key.endswith(".safetensors"))
    version, body = work.s3.objects[key]
    assert body and not body.startswith(b"!")
    work.s3.objects[key] = (version, b"!" + body[1:])


def diagnostic(step_result, *, phase, reason, error_class):
    value = getattr(step_result, "diagnostic", None)
    assert value is not None, "source refusal phase/reason diagnostic missing"
    assert (value.phase, value.reason, value.error_class) == (phase, reason, error_class)
    assert all(type(field) is str for field in (value.phase, value.reason, value.error_class))
    assert PRIVATE_DETAIL not in (value.phase, value.reason, value.error_class)
    return value


def test_actual_raw_hash_refusal_keeps_source_reason_and_same_closed_diagnostic(monkeypatch, tmp_path):
    work = original_work(monkeypatch)
    complete_fixture(monkeypatch, tmp_path, work)
    corrupt_one_raw_payload(work)
    before = copy.deepcopy(work.agent.messages)
    callback = work.agent.callback_handler
    result = step(work)
    assert result.disposition == "refused" and work.backend._warm_closed == "refused"
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert work.agent.messages == before and work.agent.callback_handler is callback
    record = work.backend._model_acquisition_store._records[work.reference]
    assert record.handle is work.handle and work.backend._warm_pending.reference == work.reference
    reads = len(work.s3.reads)
    repeated = step(work)
    assert repeated.disposition == "refused" and len(work.s3.reads) == reads
    assert work.model.calls == 2 and len(work.native.requests) == 1
    value = diagnostic(result, phase="artifact_result", reason="artifact_verification_failed",
                       error_class="ValueError")
    assert getattr(repeated, "diagnostic", None) is value


@pytest.mark.parametrize("guard", ["policy", "pin"])
def test_actual_original_policy_and_pin_guards_keep_distinct_closed_codes(monkeypatch, guard):
    work = original_work(monkeypatch)
    before = copy.deepcopy(work.agent.messages)
    calls = []
    native_describe = work.native.describe_processing_job

    def describe(**kwargs):
        calls.append(kwargs)
        return native_describe(**kwargs)

    monkeypatch.setattr(work.native, "describe_processing_job", describe)
    original_policy = work.backend.approval_policy
    if guard == "policy":
        work.backend.approval_policy = ApprovalPolicy()
    else:
        work.handle.extra["output_prefix"] += "ordinary-change/"
    result = step(work)
    assert result.disposition == "refused" and calls == [] and work.s3.reads == []
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert work.agent.messages == before and work.backend._warm_pending.reference == work.reference
    reason = "policy_refused" if guard == "policy" else "record_pin_changed"
    value = diagnostic(result, phase="initial_qualification", reason=reason, error_class="ValueError")
    if guard == "policy":
        work.backend.approval_policy = original_policy
    repeated = step(work)
    assert repeated.disposition == "refused" and calls == []
    assert getattr(repeated, "diagnostic", None) is value


def test_real_adapter_unreviewed_read_failure_is_unknown_class_without_private_detail(monkeypatch, tmp_path):
    work = original_work(monkeypatch)
    complete_fixture(monkeypatch, tmp_path, work)
    observed = []

    class UnreviewedReadError(Exception):
        def __str__(self):
            observed.append("stringified")
            return PRIVATE_DETAIL

    get = work.s3.get_object

    def fail_one_payload(**kwargs):
        if kwargs["Key"].endswith(".safetensors"):
            raise UnreviewedReadError()
        return get(**kwargs)

    monkeypatch.setattr(work.s3, "get_object", fail_one_payload)
    before = copy.deepcopy(work.agent.messages)
    result = step(work)
    assert result.disposition == "refused" and work.model.calls == 2
    assert work.agent.messages == before and len(work.native.requests) == 1
    assert observed == []
    diagnostic(result, phase="artifact_result", reason="artifact_verification_failed",
               error_class="UNKNOWN")


@pytest.mark.parametrize("invalid_payload", [True, False])
def test_normal_app_api_projects_only_closed_refusal_data_and_preserves_nominal_shape(
    monkeypatch, tmp_path, invalid_payload
):
    work = connected_work(monkeypatch)

    async def scenario():
        async def read():
            return [event async for event in runtime.agent_invocation({"prompt": work.literal})]

        reader = asyncio.create_task(read())
        try:
            limit = asyncio.get_running_loop().time() + 5
            while not (work.accepted.is_set() and work.model.calls == 2 and work.observing.is_set()):
                assert asyncio.get_running_loop().time() < limit, "original SDK producer not observed"
                await asyncio.sleep(0.005)
            assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            engine, agent = work.engines[0], work.agents[0]
            backend = engine.tool_backend
            original = work.originals[0][1]
            reference = original["request_ref"]
            handle = backend._model_acquisition_store._records[reference].handle
            assert original["accepted"] is True and handle.id == work.native.requests[0]["ProcessingJobName"]
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            graph = backend._warm_graph()
            before = copy.deepcopy(agent.messages)
            # The source reads the original every 10 ms on another thread and holds this lock for
            # each whole read. Under it the completed fixture and the changed payload are one step
            # for the source: no read can meet Completed with the unchanged payload.
            with backend._warm_lock:
                complete_fixture(monkeypatch, tmp_path, SimpleNamespace(handle=handle, adapter=work.adapter,
                                                                       native=work.native))
                if invalid_payload:
                    corrupt_one_raw_payload(work)
            events = await asyncio.wait_for(reader, 5)
            response = events[-1]["response"]
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            assert backend._warm_graph() == graph and backend._model_acquisition_store._records[reference].handle is handle
            assert len(work.seen_prompts) == 1 and work.seen_prompts[0][0] == work.literal
            if invalid_payload:
                assert work.model.calls == 2 and agent.messages == before
                assert response["data"]["warm_model_continuation"] == {
                    "disposition": "refused",
                    "diagnostic": {"phase": "artifact_result", "reason": "artifact_verification_failed",
                                   "error_class": "ValueError"},
                }
                assert "original acquisition is pending" in response["message"]
            else:
                assert work.model.calls == 3
                assert response["data"]["warm_model_continuation"] == {"disposition": "continued_original"}
            assert agent.callback_handler is work.callbacks[0]
        finally:
            if not reader.done():
                complete_fixture(monkeypatch, tmp_path / "cleanup", SimpleNamespace(
                    handle=work.engines[0].tool_backend._model_acquisition_store._records[
                        work.originals[0][1]["request_ref"]].handle,
                    adapter=work.adapter, native=work.native))
                await asyncio.wait_for(reader, 5)

    asyncio.run(scenario())


def test_actual_sdk_cancellation_propagates_without_refusal_diagnostic_conversion(monkeypatch, tmp_path):
    work = original_work(monkeypatch)
    complete_fixture(monkeypatch, tmp_path, work)
    callback = work.agent.callback_handler
    resume_calls = []

    async def cancelled_stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        resume_calls.append("original_resume")
        raise asyncio.CancelledError()
        yield {}

    monkeypatch.setattr(work.model, "stream", cancelled_stream)
    with pytest.raises(asyncio.CancelledError):
        step(work)
    assert resume_calls == ["original_resume"]
    assert work.backend._warm_closed == "refused" and work.agent.callback_handler is callback
    assert len(work.native.requests) == 1
    repeated = step(work)
    assert repeated.disposition == "refused" and resume_calls == ["original_resume"]


def test_actual_same_agent_ordinary_resume_error_keeps_latched_phase_and_closes_capture(
    monkeypatch, tmp_path
):
    work = original_work(monkeypatch)
    complete_fixture(monkeypatch, tmp_path, work)
    callback = work.agent.callback_handler
    original_record = work.backend._model_acquisition_store._records[work.reference]
    resume_calls, closed = [], []
    actual_close = backend_source.GuardrailAssessmentCapture.close

    def close_capture(value):
        result = actual_close(value)
        closed.append(value)
        return result

    async def failed_stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        resume_calls.append("same_original_resume")
        raise RuntimeError(PRIVATE_DETAIL)
        yield {}

    monkeypatch.setattr(backend_source.GuardrailAssessmentCapture, "close", close_capture)
    monkeypatch.setattr(work.model, "stream", failed_stream)
    result = step(work)
    assert result.disposition == "refused" and work.backend._warm_closed == "refused"
    assert resume_calls == ["same_original_resume"] and len(work.native.requests) == 1
    assert work.agent.callback_handler is callback
    assert closed and all(value is closed[0] and value._active is False for value in closed)
    assert work.backend._model_acquisition_store._records[work.reference] is original_record
    assert work.backend._warm_pending.reference == work.reference
    reads = len(work.s3.reads)
    repeated = step(work)
    assert repeated.disposition == "refused" and len(work.s3.reads) == reads
    assert resume_calls == ["same_original_resume"] and len(work.native.requests) == 1
    value = diagnostic(result, phase="resume", reason="continuation_failed", error_class="RuntimeError")
    assert getattr(repeated, "diagnostic", None) is value
