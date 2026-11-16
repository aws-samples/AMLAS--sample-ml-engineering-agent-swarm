"""Declared completed-route worker/report mechanics through the actual runtime.

Actual SDK session/task and report helpers remain; declared load/engine/context
seams supply no native/origin/cold authority. No initial goal or model is executed.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from contextlib import contextmanager
from types import SimpleNamespace

from bedrock_agentcore.runtime import BedrockAgentCoreContext
from test_runtime_busy_ownership import assert_report_unavailable, entered, idle, managed_report_read
from test_runtime_busy_ownership import managed_report_context as managed_report_context
from test_runtime_busy_ownership import owned_app as owned_app
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import api
from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import engine as chat_engine
from amlas.tui.chat.models import ChatResponse


def _declared_completed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run_id = "1" * 32
    task = {"component": "declared completed worker DATA"}
    calls = []
    ready, release = threading.Event(), threading.Event()
    response = ChatResponse(message="Declared continued result.", action="component_completed")

    class DeclaredContext:
        def has_task_checkpoint(self, *, policy):
            return True

        def load_qualified_task(self, *, policy):
            return task, (), None, None

        def enroll(self, *args, **kwargs):
            raise AssertionError("resume cannot enroll an initial goal")

    context = DeclaredContext()

    @contextmanager
    def declared_scope(value):
        assert value is context
        yield

    def prepare(value, session_hash, *, transport_session_id):
        assert value == run_id
        assert transport_session_id == BedrockAgentCoreContext.get_session_id()
        assert session_hash == hashlib.sha256(transport_session_id.encode("utf-8")).hexdigest()
        return context

    class DeclaredEngine:
        def __init__(self, *, approval_policy, normal_restore):
            assert normal_restore == (context, task, (), None, None)
            self.tool_backend = SimpleNamespace(
                _normal_original_result={"status": "trained", "trained": True, "full_training_complete": True}
            )

        def claim_normal_completion(self, value, checkpoint_task):
            assert value is context and checkpoint_task is task
            calls.append("claim")

        def load_normal_snapshot(self, value):
            raise AssertionError("completed routing cannot use pending loading")

        def continue_normal_completion(self):
            calls.append(("continue", threading.get_ident()))
            ready.set()
            assert release.wait(5), "declared continuation barrier was not released"
            return response

    async def forbidden_prompt(*args, **kwargs):
        raise AssertionError("completed resume cannot call run_prompt or create a new goal")

    monkeypatch.setattr(checkpoint, "prepare_normal_run", prepare)
    monkeypatch.setattr(checkpoint, "activate_normal_run", declared_scope)
    monkeypatch.setattr(chat_engine, "ChatEngine", DeclaredEngine)
    monkeypatch.setattr(api, "run_prompt", forbidden_prompt)
    return SimpleNamespace(
        run_id=run_id,
        request={"action": "resume_prompt_run", "run_id": run_id},
        ready=ready,
        release=release,
        calls=calls,
        response=response,
    )


async def _next(stream):
    return await asyncio.wait_for(anext(stream), timeout=3)


def test_completed_resume_stream_is_nonblocking_and_duplicate_creates_no_second_worker(
    monkeypatch, tmp_path, managed_report_context, owned_app
):
    app, completed = owned_app
    managed_report_context.set_request_context(
        "declared-completion-request", "declared-completion-sdk-transport-session-0123456789"
    )
    selected = _declared_completed(monkeypatch, tmp_path)

    async def scenario():
        stream = runtime.agent_invocation(selected.request)
        events = []
        try:
            restoring = await _next(stream)
            events.append(restoring)
            assert restoring == {"event": "resume_state", "state": "restoring", "run_id": selected.run_id}
            first = await _next(stream)
            events.append(first)
            assert first["event"] == "run_started" and first["run_id"] == selected.run_id
            await entered(selected.ready)
            assert app.get_async_task_info()["active_count"] == 1
            heartbeat = await _next(stream)
            events.append(heartbeat)
            assert heartbeat["event"] == "heartbeat" and heartbeat["run_id"] == selected.run_id
            continuation_threads = [value[1] for value in selected.calls if type(value) is tuple]
            assert len(continuation_threads) == 1
            assert continuation_threads[0] != threading.get_ident()
            duplicate = [event async for event in runtime.agent_invocation(selected.request)]
            # A duplicate wake during the continuation is answered with the positive state, not a refusal.
            assert duplicate == [{"event": "resume_state", "state": "continuing", "run_id": selected.run_id}]
            assert selected.calls.count("claim") == 1 and len(continuation_threads) == 1
            assert app.get_async_task_info()["active_count"] == 1 and completed == []
            selected.release.set()
            events.extend([event async for event in stream])
            await idle(app)
            terminal = events[-1]
            assert terminal["event"] == "run_completed" and terminal["status"] == "response_received"
            assert terminal["run_id"] == selected.run_id and terminal["worker_finished"] is True
            assert terminal["response"]["message"] == selected.response.message
            assert terminal["lifecycle_verified"] is False and terminal["quality_success"] is False
            assert len(completed) == 1
        finally:
            selected.release.set()
            await stream.aclose()
            await idle(app)

    asyncio.run(scenario())


def test_completed_resume_disconnect_retains_same_run_only_after_actual_sdk_cleanup(
    monkeypatch, tmp_path, managed_report_context, owned_app
):
    app, completed = owned_app
    managed_report_context.set_request_context(
        "declared-disconnect-request", "declared-disconnect-sdk-transport-session-0123456789"
    )
    selected = _declared_completed(monkeypatch, tmp_path)
    cleanup_entered, allow_cleanup = threading.Event(), threading.Event()
    actual_complete = app.complete_async_task

    def complete(task_id):
        cleanup_entered.set()
        assert allow_cleanup.wait(5), "declared SDK cleanup barrier was not released"
        return actual_complete(task_id)

    monkeypatch.setattr(app, "complete_async_task", complete)

    async def scenario():
        stream = runtime.agent_invocation(selected.request)
        try:
            assert (await _next(stream))["event"] == "resume_state"
            first = await _next(stream)
            assert first["event"] == "run_started" and first["run_id"] == selected.run_id
            await entered(selected.ready)
            assert (await _next(stream))["event"] == "heartbeat"
            await stream.aclose()
            lookup = {"action": "get_prompt_report", "run_id": selected.run_id}
            assert_report_unavailable(await managed_report_read(lookup), selected.run_id)
            assert app.get_async_task_info()["active_count"] == 1
            selected.release.set()
            await entered(cleanup_entered)
            assert_report_unavailable(await managed_report_read(lookup), selected.run_id)
            assert completed == [] and app.get_async_task_info()["active_count"] == 1
            allow_cleanup.set()
            await idle(app)
            deadline = asyncio.get_running_loop().time() + 3
            report = await managed_report_read(lookup)
            while report.get("status") == "unavailable":
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
                report = await managed_report_read(lookup)
            assert report["event"] == "prompt_report" and report["status"] == "available"
            assert report["run_id"] == selected.run_id and report["found"] is True
            assert report["report"]["worker_finished"] is True
            assert report["report"]["response"]["message"] == selected.response.message
            assert report["report"]["lifecycle_verified"] is False
            assert report["report"]["quality_success"] is False
            assert selected.calls.count("claim") == 1
            assert len([value for value in selected.calls if type(value) is tuple]) == 1
            assert len(completed) == 1
        finally:
            selected.release.set()
            allow_cleanup.set()
            await stream.aclose()
            await idle(app)

    asyncio.run(scenario())


def test_completed_resume_already_failed_worker_emits_only_closed_status(
    monkeypatch, tmp_path, managed_report_context, owned_app
):
    app, completed = owned_app
    managed_report_context.set_request_context(
        "declared-failed-request", "declared-failed-sdk-transport-session-0123456789"
    )
    selected = _declared_completed(monkeypatch, tmp_path)
    worker_starts = []
    actual_start = threading.Thread.start

    def refuse_continuation(self):
        selected.calls.append("failed_continue")
        raise ValueError("Declared immediate continuation refusal.")

    def finish_worker_before_publication(worker):
        actual_start(worker)
        if worker.name == f"prompt-run-{selected.run_id}":
            worker_starts.append(worker.name)
            worker.join(timeout=3)
            assert not worker.is_alive(), "declared immediate-refusal worker must finish"

    monkeypatch.setattr(chat_engine.ChatEngine, "continue_normal_completion", refuse_continuation)
    monkeypatch.setattr(threading.Thread, "start", finish_worker_before_publication)

    async def scenario():
        events = [event async for event in runtime.agent_invocation(selected.request)]
        await idle(app)
        assert worker_starts == [f"prompt-run-{selected.run_id}"]
        assert selected.calls == ["claim", "failed_continue"]
        assert len(completed) == 1
        assert events == [
            {"event": "resume_state", "state": "restoring", "run_id": selected.run_id},
            {
                "event": "resume_state",
                "state": "failed",
                "run_id": selected.run_id,
                "diagnostic": {"restore_stage": "continuation", "exception_class": "ValueError"},
            },
        ]
        assert_report_unavailable(
            await managed_report_read({"action": "get_prompt_report", "run_id": selected.run_id}),
            selected.run_id,
        )

    asyncio.run(scenario())
