"""Actual normal app/API worker must own its original warm model continuation."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
from types import SimpleNamespace

import model_acquisition_fixtures as f
import pytest
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.hooks.events import AfterToolCallEvent
from test_runtime_busy_ownership import managed_report_context as managed_report_context
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401
from test_runtime_warm_model_continuation import OriginalModelStream, complete_fixture, tool_payload

from amlas import api
from amlas import app as runtime
from amlas.backends.adapters import _util
from amlas.backends.adapters import model_weight_acquisition as adapter_source
from amlas.config import externalized
from amlas.data_eng import model_acquisition_schema
from amlas.flows import model_weight_acquisition_flow
from amlas.tui.chat import engine as engine_source
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

# Handler switch of the connected fixture, read from the environment at every build so that a child process started
# by a test follows it too. Unset (every test but the console test of a replacement process): the fixture assigns
# its own inert handler after the build. "built": the agent keeps the handler the real builder chose.
# "sdk-printer": the agent gets the SDK's console printer, which shows what that printer writes on a scripted run.
AGENT_HANDLER_SWITCH = "AMLAS_TEST_CONNECTED_AGENT_HANDLER"
# Upper bound for the shared worker-cleanup wait. A slow machine needs the room;
# cleanup that never finishes still fails when it ends.
LIVENESS_WAIT_SECONDS = 180


def connected_work(monkeypatch, *, mode="normal", model=None):
    # Inert fixtures must not consume pacing history from earlier tests.
    states = dict(_util._PROCESSING_CREATE_STATES)
    states.pop((f.ACCOUNT, f.REGION), None)
    monkeypatch.setattr(_util, "_PROCESSING_CREATE_STATES", states)
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job"]')
    f.patch_expected(monkeypatch, [adapter_source, model_acquisition_schema])
    native, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(adapter_source, native, s3)
    monkeypatch.setattr(model_weight_acquisition_flow, "_resolve_backend", lambda: adapter)
    owned_app = BedrockAgentCoreApp()
    monkeypatch.setattr(runtime, "app", owned_app)
    model = model if model is not None else OriginalModelStream(intervened=mode == "guardrail")
    engines, agents, originals, seen_prompts, observations, callbacks = [], [], [], [], [], []
    accepted, api_returned, observing = threading.Event(), threading.Event(), threading.Event()
    actual_engine, actual_build, actual_api = (
        engine_source.ChatEngine,
        ToolBindingBackend.build_agent,
        api.run_prompt,
    )
    actual_describe = native.describe_processing_job

    def describe_original(**kwargs):
        observations.append(native.state)
        observing.set()
        if mode == "unknown":
            raise RuntimeError("ordinary unknown original observation")
        return actual_describe(**kwargs)

    monkeypatch.setattr(native, "describe_processing_job", describe_original)

    def capture_engine(*args, **kwargs):
        value = actual_engine(*args, **kwargs)
        engines.append(value)
        return value

    def capture_agent(backend, **kwargs):
        backend._model = model
        value = actual_build(backend, **kwargs)
        choice = os.environ.get(AGENT_HANDLER_SWITCH, "")
        assert choice in {"", "built", "sdk-printer"}, (
            "unknown value of the connected fixture's handler switch"
        )
        if choice == "":
            value.callback_handler = lambda **event: None
        elif choice == "sdk-printer":
            value.callback_handler = PrintingCallbackHandler()
        callbacks.append(value.callback_handler)

        def capture_original(event):
            originals.append((event.tool_use["toolUseId"], tool_payload(event.result)))
            accepted.set()
            if mode == "policy":
                backend.approval_policy = None

        value.hooks.add_callback(AfterToolCallEvent, capture_original)
        agents.append(value)
        return value

    async def actual_prompt(value, **kwargs):
        seen_prompts.append((value, kwargs))
        response = await actual_api(value, **kwargs)
        api_returned.set()
        return response

    monkeypatch.setattr(engine_source, "ChatEngine", capture_engine)
    monkeypatch.setattr(ToolBindingBackend, "build_agent", capture_agent)
    monkeypatch.setattr(api, "run_prompt", actual_prompt)
    literal = "  Research one licensed coding model and continue after its original acquisition.\n"
    if mode == "failed":
        native.state = "Failed"
    if mode == "no_original":

        async def plain_stream(messages, tool_specs=None, system_prompt=None, **kwargs):
            model.calls += 1
            model.messages_seen.append(copy.deepcopy(messages))
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "The model is pending."}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}

        monkeypatch.setattr(model, "stream", plain_stream)
        bound_capture = capture_engine

        def public_claim_engine(*args, **kwargs):
            value = bound_capture(*args, **kwargs)
            bound_message = value.process_message

            async def public_claim(message, **kw):
                response = await bound_message(message, **kw)
                response.data.update(status="PENDING", request_ref="foreign", accepted=True)
                return response

            value.process_message = public_claim
            return value

        monkeypatch.setattr(engine_source, "ChatEngine", public_claim_engine)
    actual_sleep = asyncio.sleep

    async def neutral_sleep(delay, *args, **kwargs):
        if delay == 15.0 and threading.current_thread().name.startswith("prompt-run-"):
            delay = 0.01
        await actual_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", neutral_sleep)
    return SimpleNamespace(
        native=native,
        s3=s3,
        adapter=adapter,
        owned_app=owned_app,
        model=model,
        engines=engines,
        agents=agents,
        originals=originals,
        seen_prompts=seen_prompts,
        observations=observations,
        accepted=accepted,
        api_returned=api_returned,
        observing=observing,
        literal=literal,
        callbacks=callbacks,
    )


def test_normal_app_api_keeps_original_model_worker_busy_until_verified_same_agent_continuation(
    monkeypatch, tmp_path
):
    work = connected_work(monkeypatch)
    native, adapter, owned_app, model = work.native, work.adapter, work.owned_app, work.model
    engines, agents, originals = work.engines, work.agents, work.originals
    accepted, api_returned, observing = work.accepted, work.api_returned, work.observing
    observations, seen_prompts, literal = work.observations, work.seen_prompts, work.literal

    async def scenario():
        async def read():
            return [event async for event in runtime.agent_invocation({"prompt": literal})]

        reader = asyncio.create_task(read())
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not (api_returned.is_set() or observing.is_set()):
                assert asyncio.get_running_loop().time() < deadline, (
                    "normal SDK acquisition did not reach its first boundary"
                )
                await asyncio.sleep(0.005)
            assert accepted.is_set() and len(native.requests) == len(engines) == len(agents) == 1
            assert len(originals) == 1 and originals[0][0] == "original-model-use"
            original = originals[0][1]
            assert original["accepted"] is True and original["status"] == "submitted"
            backend = engines[0].tool_backend
            store = backend._model_acquisition_store
            reference = original["request_ref"]
            handle = store._records[reference].handle
            graph = backend._warm_graph()
            assert len(graph) == 7 and backend._warm_pending.reference == reference
            assert backend._warm_pending.agent is agents[0]
            assert handle.id == native.requests[0]["ProcessingJobName"]
            assert model.calls == 2
            before = copy.deepcopy(agents[0].messages)
            if api_returned.is_set():
                early = await asyncio.wait_for(reader, 5)
                assert (
                    early[-1]["response"]["message"]
                    == "\U0001f527 used: acquire_model_weights\n\nThe original acquisition is pending."
                )
                assert owned_app.get_async_task_info()["active_count"] == 0
            assert not api_returned.is_set(), (
                "normal API returned pending and completed its SDK task before observing the original job"
            )
            assert not reader.done()
            assert owned_app.get_current_ping_status().value == "HealthyBusy"
            assert owned_app.get_async_task_info()["active_count"] == 1
            assert owned_app._handle_ping(None).status_code == 200
            assert observations == ["InProgress"]
            complete_fixture(
                monkeypatch, tmp_path, SimpleNamespace(handle=handle, adapter=adapter, native=native)
            )
            events = await asyncio.wait_for(reader, 5)
            assert (
                events[-1]["response"]["message"]
                == "\U0001f527 used: acquire_model_weights\n\nOriginal exact-version artifacts are verified."
            )
            assert api_returned.is_set() and owned_app.get_async_task_info()["active_count"] == 0
            assert len(native.requests) == len(engines) == len(agents) == 1 and model.calls == 3
            assert backend._warm_graph() == graph and store._records[reference].handle is handle
            assert agents[0].messages[0] == before[0]
            assert len(seen_prompts) == 1 and seen_prompts[0][0] == literal
            results = [
                block["toolResult"]
                for message in agents[0].messages
                for block in message["content"]
                if block.get("toolResult", {}).get("toolUseId") == "original-model-use"
            ]
            assert len(results) == 1
            assert agents[0].messages[: len(before)] == before
            pending_result = tool_payload(results[0])
            assert pending_result == original
            completion = json.loads(model.messages_seen[-1][-1]["content"][0]["text"])
            assert completion["request_ref"] == reference and completion["acquired"] is True
            assert completion["data_only"] is True and completion["file_count"] == 9
            assert completion["manifest_roles"] == ["receipt", "commit", "lineage"]
            verified = store.observe(reference)
            assert verified["acquired"] is True and len(verified["file_refs"]) == 9
            assert set(verified["manifest_refs"]) == {"receipt", "commit", "lineage"}
        finally:
            if not reader.done():
                reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    asyncio.run(scenario())


async def started_original(work):
    deadline = asyncio.get_running_loop().time() + 5
    while not (work.api_returned.is_set() or work.observing.is_set()):
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.005)
    assert work.accepted.is_set()
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    original = work.originals[0][1]
    backend = work.engines[0].tool_backend
    assert original["accepted"] is True
    assert backend._warm_pending.reference == original["request_ref"]
    assert backend._warm_pending.agent is work.agents[0]
    return backend


async def close_original_worker(work, reader):
    deadline = asyncio.get_running_loop().time() + LIVENESS_WAIT_SECONDS
    with runtime._PROMPT_REPORT_LOCK:
        original_reports = frozenset(runtime._ACTIVE_PROMPT_REPORTS)
    if work.engines:
        work.engines[0].tool_backend.approval_policy = None
    while True:
        with runtime._PROMPT_REPORT_LOCK:
            report_worker_active = not original_reports.isdisjoint(runtime._ACTIVE_PROMPT_REPORTS)
        if not work.owned_app.get_async_task_info()["active_count"] and not report_worker_active:
            break
        assert asyncio.get_running_loop().time() < deadline, (
            "original worker did not finish task/report cleanup after source refusal"
        )
        await asyncio.sleep(0.005)
    if not reader.done():
        reader.cancel()
    await asyncio.gather(reader, return_exceptions=True)


@pytest.mark.parametrize(
    "mode,expected",
    [("failed", "terminal_failed"), ("policy", "refused"), ("guardrail", "guardrail_intervened")],
)
def test_connected_original_closure_returns_typed_reason_without_stage_advance(monkeypatch, mode, expected):
    work = connected_work(monkeypatch, mode=mode)

    async def scenario():
        events = [event async for event in runtime.agent_invocation({"prompt": work.literal})]
        assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
        assert work.originals[0][1]["accepted"] is True
        assert work.model.calls == 2
        terminal = events[-1]
        assert terminal["status"] == "response_received" and terminal["quality_success"] is False
        continuation = {"disposition": expected}
        if mode == "policy":
            continuation["diagnostic"] = {
                "phase": "initial_qualification",
                "reason": "policy_refused",
                "error_class": "ValueError",
            }
        assert terminal["response"]["data"]["warm_model_continuation"] == continuation
        if mode == "policy":
            assert work.observations == []
        if mode == "guardrail":
            assert terminal["response"]["data"]["guardrail_outcome"]["action"] == "UNKNOWN"
        assert work.owned_app.get_async_task_info()["active_count"] == 0

    asyncio.run(scenario())


def test_connected_unknown_original_holds_busy_and_current_policy_revocation_closes(monkeypatch):
    work = connected_work(monkeypatch, mode="unknown")

    async def scenario():
        async def read():
            return [event async for event in runtime.agent_invocation({"prompt": work.literal})]

        reader = asyncio.create_task(read())
        try:
            backend = await started_original(work)
            assert not work.api_returned.is_set() and not reader.done()
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            assert work.owned_app.get_current_ping_status().value == "HealthyBusy"
            assert work.model.calls == 2
            backend.approval_policy = None
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["response"]["data"]["warm_model_continuation"] == {
                "disposition": "refused",
                "diagnostic": {
                    "phase": "initial_qualification",
                    "reason": "policy_refused",
                    "error_class": "ValueError",
                },
            }
            assert work.model.calls == 2 and len(work.native.requests) == 1
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


@pytest.mark.parametrize("termination", ["deadline", "disconnect"])
def test_connected_stream_end_leaves_actual_original_worker_busy_until_refusal(monkeypatch, termination):
    work = connected_work(monkeypatch)
    if termination == "deadline":
        monkeypatch.setenv("AMLAS_MAX_RUN_SECONDS", "2")

    async def scenario():
        stream = runtime.agent_invocation({"prompt": work.literal})

        async def read_remaining():
            return [event async for event in stream]

        reader = None
        try:
            assert (await anext(stream))["event"] == "run_started"
            assert (await anext(stream))["event"] == "heartbeat"
            backend = await started_original(work)
            assert not work.api_returned.is_set()
            if termination == "deadline":
                reader = asyncio.create_task(read_remaining())
                events = await asyncio.wait_for(reader, 5)
                assert events[-1]["status"] == "outcome_unknown" and events[-1]["worker_finished"] is False
            else:
                await stream.aclose()
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            assert work.owned_app.get_current_ping_status().value == "HealthyBusy"
            assert work.model.calls == 2 and len(work.native.requests) == 1
            backend.approval_policy = None
        finally:
            if reader is None:
                reader = asyncio.create_task(asyncio.sleep(0))
            await close_original_worker(work, reader)
            await stream.aclose()

    asyncio.run(scenario())


def test_public_pending_claims_without_original_capture_do_not_start_observer(monkeypatch):
    work = connected_work(monkeypatch, mode="no_original")

    async def scenario():
        events = [event async for event in runtime.agent_invocation({"prompt": work.literal})]
        assert events[-1]["response"]["data"]["status"] == "PENDING"
        assert "warm_model_continuation" not in events[-1]["response"]["data"]
        assert work.native.requests == [] and work.observations == [] and work.originals == []
        assert work.model.calls == 1 and len(work.engines) == len(work.agents) == 1
        assert work.owned_app.get_async_task_info()["active_count"] == 0

    asyncio.run(scenario())


def test_cancelled_model_worker_allows_distinct_fresh_prompt_without_original_replay(
    monkeypatch, tmp_path, managed_report_context
):
    """A fresh logical run after cancellation is not failed-run or cold recovery.

    Actual app/API/engine/SDK and source acquisition ACK remain. Only fresh
    scripted Models and strict native provider ports are controlled; no normal
    context, claim qualifier or native execution authority is manufactured.
    """
    from test_runtime_busy_ownership import assert_report_unavailable, entered, idle, managed_report_read

    monkeypatch.chdir(tmp_path)
    cancelled_model, fresh_model = OriginalModelStream(), OriginalModelStream()
    original_stream = cancelled_model.stream

    async def cancel_after_original_ack(messages, tool_specs=None, system_prompt=None, **kwargs):
        if cancelled_model.calls:
            assert cancelled_model.calls == 1, "cancelled acquisition must not replay"
            cancelled_model.messages_seen.append(copy.deepcopy(messages))
            cancelled_model.calls += 1
            raise asyncio.CancelledError("Declared model cancellation after original ACK.")
        async for event in original_stream(messages, tool_specs, system_prompt, **kwargs):
            yield event

    final_text = "A fresh ordinary coding plan remains descriptive."

    async def fresh_final(messages, tool_specs=None, system_prompt=None, **kwargs):
        fresh_model.messages_seen.append(copy.deepcopy(messages))
        fresh_model.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": final_text}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    monkeypatch.setattr(cancelled_model, "stream", cancel_after_original_ack)
    monkeypatch.setattr(fresh_model, "stream", fresh_final)
    actual_build = ToolBindingBackend.build_agent
    work = connected_work(monkeypatch, model=cancelled_model)
    capture_build = ToolBindingBackend.build_agent

    def select_fresh_model(backend, **kwargs):
        if not work.agents:
            return capture_build(backend, **kwargs)
        assert len(work.agents) == 1, "only the separate second prompt may build another Agent"
        backend._model = fresh_model
        value = actual_build(backend, **kwargs)
        value.callback_handler = lambda **event: None
        work.callbacks.append(value.callback_handler)
        work.agents.append(value)
        return value

    monkeypatch.setattr(ToolBindingBackend, "build_agent", select_fresh_model)
    stopped, task_ids, cleaned = [], [], []
    cleanup_entered, allow_cleanup = threading.Event(), threading.Event()
    actual_add, actual_complete = work.owned_app.add_async_task, work.owned_app.complete_async_task

    def no_stop(**kwargs):
        stopped.append(kwargs)
        raise AssertionError("cancellation cannot stop the accepted original native job")

    def add_task(name):
        task_id = actual_add(name)
        task_ids.append(task_id)
        return task_id

    def complete_task(task_id):
        if len(task_ids) == 2 and task_id == task_ids[1]:
            cleanup_entered.set()
            assert allow_cleanup.wait(5), "second SDK cleanup barrier must release"
        actual_complete(task_id)
        cleaned.append(task_id)

    monkeypatch.setattr(work.native, "stop_processing_job", no_stop, raising=False)
    monkeypatch.setattr(work.owned_app, "add_async_task", add_task)
    monkeypatch.setattr(work.owned_app, "complete_async_task", complete_task)
    second_prompt = "Describe a fresh ordinary coding plan without acquiring a model."

    async def scenario():
        stream = reader = None
        managed_report_context.set_request_context("declared-cancel-request", "declared-two-run-session")
        try:
            first = [event async for event in runtime.agent_invocation({"prompt": work.literal})]
            first_run = next(event["run_id"] for event in first if event["event"] == "run_started")
            await idle(work.owned_app)
            assert len(work.originals) == len(work.native.requests) == 1
            original = work.originals[0][1]
            assert work.originals[0][0] == "original-model-use"
            assert original["accepted"] is True and original["status"] == "submitted"
            assert cancelled_model.calls == 2 and fresh_model.calls == 0
            terminal = first[-1]
            assert terminal["event"] == "run_completed" and terminal["status"] == "failed"
            assert terminal["error_type"] == "CancelledError" and terminal["worker_finished"] is True
            assert terminal["lifecycle_verified"] is False and terminal["quality_success"] is False
            assert cleaned == task_ids and len(cleaned) == 1 and stopped == []
            assert_report_unavailable(
                await managed_report_read({"action": "get_prompt_report", "run_id": first_run}), first_run
            )

            managed_report_context.set_request_context("declared-fresh-request", "declared-two-run-session")
            stream = runtime.agent_invocation({"prompt": second_prompt})
            started = await asyncio.wait_for(anext(stream), 5)
            assert started["event"] == "run_started" and started["run_id"] != first_run
            second_run = started["run_id"]

            async def remaining():
                return [event async for event in stream]

            reader = asyncio.create_task(remaining())
            await entered(cleanup_entered)
            assert not reader.done() and work.owned_app.get_async_task_info()["active_count"] == 1
            assert cleaned == task_ids[:1] and len(task_ids) == 2
            lookup = {"action": "get_prompt_report", "run_id": second_run}
            assert_report_unavailable(await managed_report_read(lookup), second_run)
            allow_cleanup.set()
            second = await asyncio.wait_for(reader, 5)
            await idle(work.owned_app)
            terminal = second[-1]
            assert terminal["status"] == "response_received" and terminal["worker_finished"] is True
            assert terminal["run_id"] == second_run
            assert terminal["response"]["action"] == "tool_agent_turn"
            assert terminal["response"]["message"] == final_text
            assert terminal["lifecycle_verified"] is False and terminal["quality_success"] is False
            assert len(work.engines) == len(work.agents) == 2
            assert work.engines[0] is not work.engines[1] and work.agents[0] is not work.agents[1]
            assert work.agents[0].model is cancelled_model and work.agents[1].model is fresh_model
            assert [value for value, _ in work.seen_prompts] == [work.literal, second_prompt]
            assert fresh_model.calls == 1 and cancelled_model.calls == 2
            history = fresh_model.messages_seen[0]
            assert [message["role"] for message in history] == ["user"]
            assert all(set(block) == {"text"} for block in history[0]["content"])
            assert history[0]["content"][0]["text"].startswith(second_prompt)
            assert work.literal not in json.dumps(history) and "original-model-use" not in json.dumps(history)
            assert len(work.native.requests) == len(work.originals) == 1 and stopped == []
            assert cleaned == task_ids and len(set(task_ids)) == 2
            report = await managed_report_read(lookup)
            assert report["status"] == "available" and report["found"] is True
            assert report["report"]["worker_finished"] is True
            assert report["report"]["response"] == terminal["response"]
            assert report["report"]["lifecycle_verified"] is False
            assert report["report"]["quality_success"] is False
            assert cancelled_model.calls == 2 and fresh_model.calls == 1
        finally:
            allow_cleanup.set()
            if reader is not None:
                if not reader.done():
                    reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            if stream is not None:
                await stream.aclose()
            await idle(work.owned_app)

    asyncio.run(scenario())
