"""Actual SDK health ownership; neutral threads, no provider or native effects."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas import api
from amlas import app as runtime
from amlas.tui.chat.models import ChatResponse


@pytest.fixture
def owned_app(monkeypatch):
    app = BedrockAgentCoreApp()
    completed = []
    actual = app.complete_async_task

    def complete(task_id):
        completed.append((task_id, threading.get_ident()))
        return actual(task_id)

    monkeypatch.setattr(app, "complete_async_task", complete)
    monkeypatch.setattr(runtime, "app", app)
    return app, completed


async def entered(event):
    deadline = asyncio.get_running_loop().time() + 3
    while not event.is_set():
        assert asyncio.get_running_loop().time() < deadline, "normal worker did not start"
        await asyncio.sleep(0.005)


async def idle(app):
    deadline = asyncio.get_running_loop().time() + 3
    while app.get_async_task_info()["active_count"]:
        assert asyncio.get_running_loop().time() < deadline, "worker task ownership leaked"
        await asyncio.sleep(0.005)


async def read(prompt):
    return [event async for event in runtime.agent_invocation({"prompt": prompt})]


def test_actual_sdk_busy_registered_before_normal_worker_and_completed_once(monkeypatch, owned_app):
    app, completed = owned_app
    started, release = threading.Event(), threading.Event()
    seen = []
    literal = "  Keep the ordinary literal prompt.\n"

    async def run_prompt(value, **kwargs):
        seen.append((value, kwargs, app.get_async_task_info()["active_count"]))
        started.set()
        assert release.wait(4)
        return ChatResponse(message="done")

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        task = asyncio.create_task(read(literal))
        try:
            await entered(started)
            assert app.get_current_ping_status().value == "HealthyBusy"
            assert app.get_async_task_info()["active_count"] == 1
            ping = app._handle_ping(None)
            assert ping.status_code == 200 and json.loads(ping.body)["status"] == "HealthyBusy"
        finally:
            release.set()
            events = await asyncio.wait_for(task, 4)
        await idle(app)
        return events

    events = asyncio.run(scenario())
    assert seen == [(literal, {}, 1)]
    assert len(completed) == 1
    assert app.get_current_ping_status().value == "Healthy"
    assert events[-1]["status"] == "response_received"
    assert events[-1]["lifecycle_verified"] is False


def test_two_normal_workers_keep_distinct_tasks_until_each_actual_completion(monkeypatch, owned_app):
    app, completed = owned_app
    starts = {name: threading.Event() for name in ("first", "second")}
    releases = {name: threading.Event() for name in starts}

    async def run_prompt(value):
        starts[value].set()
        assert releases[value].wait(4)
        return ChatResponse(message=value)

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        tasks = {name: asyncio.create_task(read(name)) for name in starts}
        try:
            await entered(starts["first"])
            await entered(starts["second"])
            assert app.get_async_task_info()["active_count"] == 2
            original_ids = set(app._active_tasks)
            assert len(original_ids) == 2
            releases["first"].set()
            await asyncio.wait_for(tasks["first"], 4)
            assert app.get_async_task_info()["active_count"] == 1
            assert app.get_current_ping_status().value == "HealthyBusy"
            assert set(app._active_tasks) < original_ids
        finally:
            for release in releases.values():
                release.set()
            await asyncio.gather(*tasks.values())
        await idle(app)

    asyncio.run(scenario())
    assert len(completed) == 2 and len({item[0] for item in completed}) == 2


def test_worker_exception_completes_original_task_without_raw_detail(monkeypatch, owned_app):
    app, completed = owned_app

    async def run_prompt(_):
        raise LookupError("ordinary private fixture diagnostic")

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    events = asyncio.run(read("ordinary request"))
    assert events[-1]["status"] == "failed" and events[-1]["error_type"] == "LookupError"
    assert "ordinary private fixture diagnostic" not in str(events)
    assert app.get_async_task_info()["active_count"] == 0 and len(completed) == 1


@pytest.mark.parametrize("termination", ["deadline", "disconnect"])
def test_stream_termination_keeps_busy_until_actual_worker_finally(monkeypatch, owned_app, termination):
    app, completed = owned_app
    started, release = threading.Event(), threading.Event()
    if termination == "deadline":
        monkeypatch.setenv("AMLAS_MAX_RUN_SECONDS", "0.02")

    async def run_prompt(_):
        started.set()
        assert release.wait(4)
        return ChatResponse(message="done")

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        stream = runtime.agent_invocation({"prompt": "ordinary request"})
        try:
            first = await anext(stream)
            assert first["event"] == "run_started"
            if termination == "deadline":
                events = [event async for event in stream]
                assert events[-1]["status"] == "outcome_unknown"
                assert events[-1]["worker_finished"] is False
            else:
                heartbeat = await anext(stream)
                assert heartbeat["event"] == "heartbeat"
                await stream.aclose()
            assert app.get_async_task_info()["active_count"] == 1
            assert app.get_current_ping_status().value == "HealthyBusy"
            assert completed == []
        finally:
            release.set()
            await stream.aclose()
        await idle(app)
        assert len(completed) == 1

    asyncio.run(scenario())


def test_rejected_worker_start_leaves_no_task_or_api_call(monkeypatch, owned_app):
    app, _completed = owned_app
    calls = []

    async def run_prompt(_):
        calls.append(True)
        return ChatResponse(message="done")

    def refuse(_thread):
        raise RuntimeError("neutral worker start refusal")

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    monkeypatch.setattr(threading.Thread, "start", refuse)
    events = asyncio.run(read("ordinary request"))
    assert calls == [] and app.get_async_task_info()["active_count"] == 0
    assert events[-1]["status"] == "failed" and events[-1]["worker_finished"] is True
    assert events[-1]["error_type"] == "RuntimeError"


def test_busy_timestamp_changes_only_on_actual_status_transition(monkeypatch, owned_app):
    import bedrock_agentcore.runtime.app as sdk_module

    app, _completed = owned_app
    clock = SimpleNamespace(value=100.0)
    monkeypatch.setattr(sdk_module, "time", SimpleNamespace(time=lambda: clock.value))
    started, release = threading.Event(), threading.Event()

    async def run_prompt(_):
        started.set()
        assert release.wait(4)
        return ChatResponse(message="done")

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        app.get_current_ping_status()
        clock.value = 110.0
        task = asyncio.create_task(read("ordinary request"))
        try:
            await entered(started)
            first = json.loads(app._handle_ping(None).body)
            assert first == {"status": "HealthyBusy", "time_of_last_update": 110}
            clock.value = 120.0
            assert json.loads(app._handle_ping(None).body) == first
        finally:
            release.set()
            await task
        clock.value = 130.0
        assert json.loads(app._handle_ping(None).body) == {"status": "Healthy", "time_of_last_update": 130}
        clock.value = 140.0
        assert json.loads(app._handle_ping(None).body)["time_of_last_update"] == 130

    asyncio.run(scenario())


@pytest.fixture
def report_readback_guard(monkeypatch, owned_app):
    """Local report reads must never enter an execution or native boundary."""
    import boto3

    from amlas.control_plane import runtime_session
    from amlas.tui.chat import engine

    app, _completed = owned_app
    forbidden, task_names = [], []
    actual_add = app.add_async_task

    def refuse(label):
        def call(*_args, **_kwargs):
            forbidden.append(label)
            raise AssertionError("report readback must remain passive")

        return call

    def add_task(name):
        task_names.append(name)
        return actual_add(name)

    monkeypatch.setattr(runtime_session, "invoke_capability", refuse("capability"))
    monkeypatch.setattr(engine, "ChatEngine", refuse("engine"))
    monkeypatch.setattr(boto3, "client", refuse("native"))
    monkeypatch.setattr(boto3.session.Session, "client", refuse("native"))
    monkeypatch.setattr(app, "add_async_task", add_task)
    return forbidden, task_names


@pytest.mark.parametrize("termination", ["disconnect", "deadline"])
@pytest.mark.parametrize("outcome", ["report", "held"])
def test_same_worker_final_report_survives_closed_consumer_for_passive_resume(
    monkeypatch, tmp_path, owned_app, report_readback_guard, termination, outcome
):
    """Proposed local result mapping; this is not managed HTTP recovery proof."""
    app, completed = owned_app
    forbidden, task_names = report_readback_guard
    started, release = threading.Event(), threading.Event()
    calls = []
    literal = "Keep the original normal request."
    monkeypatch.chdir(tmp_path)
    if termination == "deadline":
        monkeypatch.setenv("AMLAS_MAX_RUN_SECONDS", "0.02")

    if outcome == "report":
        response = ChatResponse(
            message="The original coding report is available.",
            action="tool_agent_turn",
            data={"case_count": 164, "quality_success": True, "nested": {"value": 3}},
            suggestions=["Inspect the original receipts."],
        )
    else:
        response = ChatResponse(
            message="The original admission hold remains unresolved.",
            action="input_admission_held",
            data={"decision": "UNKNOWN", "reason": "TRANSPORT_UNKNOWN", "pending": True},
            suggestions=["Use independent approved evidence."],
        )
    expected_response = {
        "message": response.message,
        "action": response.action,
        "data": response.data,
        "suggestions": response.suggestions,
    }

    async def run_prompt(value, **kwargs):
        calls.append((value, kwargs, threading.get_ident()))
        started.set()
        assert release.wait(4), "the consumer must release its same original worker"
        return response

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        stream = runtime.agent_invocation({"prompt": literal})
        try:
            first = await anext(stream)
            assert first["event"] == "run_started"
            original_id = first["run_id"]
            assert isinstance(original_id, str) and len(original_id) == 32
            event = asyncio.create_task(anext(stream))
            await entered(started)
            assert app.get_async_task_info()["active_count"] == 1
            assert completed == []
            assert api.resume(original_id + "-absent", project_dir=str(tmp_path))["found"] is False
            if termination == "disconnect":
                assert (await event)["event"] == "heartbeat"
                await stream.aclose()
            else:
                ended = [await event] + [item async for item in stream]
                assert ended[-1]["status"] == "outcome_unknown"
                assert ended[-1]["worker_finished"] is False
            assert app.get_current_ping_status().value == "HealthyBusy"
            assert app.get_async_task_info()["active_count"] == 1
            assert completed == []
        finally:
            release.set()
            await stream.aclose()
        await idle(app)
        return original_id

    run_id = asyncio.run(scenario())
    assert len(calls) == 1 and calls[0][:2] == (literal, {})
    assert task_names == ["normal_prompt"]
    assert len(completed) == 1 and completed[0][1] == calls[0][2]
    assert app.get_async_task_info()["active_count"] == 0
    assert forbidden == []

    restored = api.resume(run_id, project_dir=str(tmp_path))
    assert restored["found"] is True, "the same worker's final report was not retained"
    assert restored["session_ref"] == run_id
    report = restored["context"]["metadata"]["normal_prompt_report"]
    assert report["run_id"] == run_id and report["mode"] == "prompt"
    assert report["event"] == "run_completed" and report["status"] == "response_received"
    assert report["worker_finished"] is True and report["response"] == expected_response
    assert report["lifecycle_verified"] is False and report["quality_success"] is False
    assert report["durable_resume"] is False and report["state_scope"] == "single_invocation"
    for _ in range(2):
        assert api.resume(run_id, project_dir=str(tmp_path)) == restored
        assert api.resume(run_id + "-absent", project_dir=str(tmp_path))["found"] is False
    assert len(calls) == 1 and len(completed) == 1
    assert task_names == ["normal_prompt"] and forbidden == []


def test_unknown_report_reference_is_passive_and_creates_no_worker(
    tmp_path, owned_app, report_readback_guard
):
    app, completed = owned_app
    forbidden, task_names = report_readback_guard
    for _ in range(2):
        result = api.resume("ordinary-absent-report", project_dir=str(tmp_path))
        assert result["found"] is False and result["context"] == {}
    assert app.get_async_task_info()["active_count"] == 0
    assert completed == [] and task_names == [] and forbidden == []


@pytest.fixture
def managed_report_context():
    """Use actual SDK ContextVars; this does not prove service IAM authentication."""
    from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

    context = BedrockAgentCoreContext
    request_token = context._request_id.set(None)
    session_token = context._session_id.set(None)
    runtime._RESUME_WORKERS.clear()  # resume state is per process; tests reuse run ids
    try:
        yield context
    finally:
        runtime._RESUME_WORKERS.clear()
        context._request_id.reset(request_token)
        context._session_id.reset(session_token)


async def managed_report_read(payload):
    """Proposed passive action on the existing normal entrypoint."""
    try:
        events = [event async for event in runtime.agent_invocation(payload)]
    except ValueError:
        pytest.fail("managed report read action did not reach a passive response")
    assert len(events) == 1, "report lookup must emit one passive response"
    return events[0]


def assert_report_unavailable(event, run_id):
    assert event == {
        "event": "prompt_report",
        "status": "unavailable",
        "run_id": run_id,
        "found": False,
    }, "busy, absent and foreign reports must not disclose a result or owner"


@pytest.mark.parametrize("outcome", ["report", "held"])
def test_managed_report_same_sdk_session_recovers_original_after_finally(
    monkeypatch, tmp_path, owned_app, report_readback_guard, managed_report_context, outcome
):
    app, completed = owned_app
    forbidden, task_names = report_readback_guard
    context = managed_report_context
    context.set_request_context("ordinary-original-request", "ordinary-owner-session")
    monkeypatch.chdir(tmp_path)
    started, release = threading.Event(), threading.Event()
    calls = []
    response = ChatResponse(
        message="Original report." if outcome == "report" else "Original hold remains.",
        action="tool_agent_turn" if outcome == "report" else "input_admission_held",
        data={"case_count": 164, "quality_success": True}
        if outcome == "report"
        else {"decision": "UNKNOWN", "reason": "TRANSPORT_UNKNOWN", "pending": True},
        suggestions=["Inspect original receipts."],
    )
    expected = {
        "message": response.message,
        "action": response.action,
        "data": response.data,
        "suggestions": response.suggestions,
    }

    async def run_prompt(value, **kwargs):
        calls.append((value, kwargs, context.get_session_id(), threading.get_ident()))
        started.set()
        assert release.wait(4), "the test must release its same original worker"
        return response

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        stream = runtime.agent_invocation({"prompt": "Keep one original managed prompt."})
        try:
            first = await anext(stream)
            run_id = first["run_id"]
            assert len(run_id) == 32 and all(c in "0123456789abcdef" for c in run_id)
            heartbeat = await anext(stream)
            await entered(started)
            assert heartbeat["event"] == "heartbeat"
            await stream.aclose()
            assert app.get_async_task_info()["active_count"] == 1 and completed == []
        finally:
            release.set()
            await stream.aclose()
        await idle(app)
        assert len(completed) == 1 and completed[0][1] == calls[0][3]
        context.set_request_context("ordinary-report-read", "ordinary-owner-session")
        payload = {"action": "get_prompt_report", "run_id": run_id}
        deadline = asyncio.get_running_loop().time() + 3
        while True:
            event = await managed_report_read(payload)
            if event.get("status") == "available":
                break
            assert_report_unavailable(event, run_id)
            assert asyncio.get_running_loop().time() < deadline, "final report was not published"
            await asyncio.sleep(0.005)
        assert event["event"] == "prompt_report" and event["status"] == "available"
        assert event["found"] is True and event["run_id"] == run_id
        report = event["report"]
        assert report["event"] == "run_completed" and report["status"] == "response_received"
        assert report["run_id"] == run_id and report["response"] == expected
        assert report["worker_finished"] is True and report["mode"] == "prompt"
        assert report["quality_success"] is False and report["lifecycle_verified"] is False
        assert report["durable_resume"] is False and report["state_scope"] == "single_invocation"
        assert await managed_report_read(payload) == event
        return run_id

    asyncio.run(scenario())
    assert len(calls) == 1 and calls[0][:3] == (
        "Keep one original managed prompt.",
        {},
        "ordinary-owner-session",
    )
    assert task_names == ["normal_prompt"] and forbidden == []
    assert len(completed) == 1 and app.get_async_task_info()["active_count"] == 0


def test_managed_report_unavailable_until_actual_sdk_cleanup(
    monkeypatch, tmp_path, owned_app, report_readback_guard, managed_report_context
):
    app, completed = owned_app
    forbidden, task_names = report_readback_guard
    context = managed_report_context
    context.set_request_context("ordinary-original-request", "ordinary-owner-session")
    monkeypatch.chdir(tmp_path)
    cleanup_entered, cleanup_release = threading.Event(), threading.Event()
    actual_complete = app.complete_async_task
    calls = []

    def complete(task_id):
        cleanup_entered.set()
        assert cleanup_release.wait(4), "test cleanup must always release the original SDK task"
        return actual_complete(task_id)

    async def run_prompt(value, **kwargs):
        calls.append((value, kwargs))
        return ChatResponse(message="One original report.")

    monkeypatch.setattr(app, "complete_async_task", complete)
    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        stream = runtime.agent_invocation({"prompt": "Keep cleanup ownership."})
        event_task = None
        try:
            first = await anext(stream)
            run_id = first["run_id"]
            event_task = asyncio.create_task(anext(stream))
            await entered(cleanup_entered)
            assert app.get_async_task_info()["active_count"] == 1 and completed == []
            context.set_request_context("ordinary-report-read", "ordinary-owner-session")
            event = await managed_report_read({"action": "get_prompt_report", "run_id": run_id})
            assert_report_unavailable(event, run_id)
            assert app.get_async_task_info()["active_count"] == 1 and completed == []
        finally:
            cleanup_release.set()
            if event_task is not None:
                await asyncio.wait_for(event_task, 4)
            await stream.aclose()
        await idle(app)
        assert len(completed) == 1

    asyncio.run(scenario())
    assert calls == [("Keep cleanup ownership.", {})]
    assert task_names == ["normal_prompt"] and forbidden == []


def test_managed_report_cross_sdk_session_is_not_disclosed(
    monkeypatch, tmp_path, owned_app, report_readback_guard, managed_report_context
):
    app, completed = owned_app
    forbidden, task_names = report_readback_guard
    context = managed_report_context
    context.set_request_context("ordinary-original-request", "ordinary-owner-session")
    monkeypatch.chdir(tmp_path)
    calls = []

    async def run_prompt(value, **kwargs):
        calls.append((value, kwargs))
        return ChatResponse(message="Owner-only original report.", data={"ordinary_metric": 7})

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        events = await read("Keep original session binding.")
        run_id = events[0]["run_id"]
        await idle(app)
        assert events[-1]["status"] == "response_received" and len(completed) == 1
        context.set_request_context("ordinary-foreign-read", "ordinary-other-session")
        payload = {"action": "get_prompt_report", "run_id": run_id}
        for _ in range(2):
            assert_report_unavailable(await managed_report_read(payload), run_id)

    asyncio.run(scenario())
    assert calls == [("Keep original session binding.", {})]
    assert task_names == ["normal_prompt"] and forbidden == [] and len(completed) == 1


def test_managed_report_unknown_reference_is_passive(
    tmp_path, owned_app, report_readback_guard, managed_report_context
):
    app, completed = owned_app
    forbidden, task_names = report_readback_guard
    managed_report_context.set_request_context("ordinary-report-read", "ordinary-owner-session")
    run_id = "a" * 32

    async def scenario():
        for _ in range(2):
            event = await managed_report_read({"action": "get_prompt_report", "run_id": run_id})
            assert_report_unavailable(event, run_id)

    asyncio.run(scenario())
    assert app.get_async_task_info()["active_count"] == 0 and completed == []
    assert task_names == [] and forbidden == []


def test_managed_report_invalid_fields_and_missing_sdk_context_refuse_passively(
    tmp_path, owned_app, report_readback_guard, managed_report_context
):
    app, completed = owned_app
    forbidden, task_names = report_readback_guard
    context = managed_report_context
    run_id = "b" * 32
    invalid = [
        {"action": "get_prompt_report", "run_id": "ordinary-invalid"},
        {"action": "get_prompt_report", "run_id": "ordinary/path"},
        {"action": "get_prompt_report", "run_id": run_id, "project_dir": "ordinary-workspace"},
        {"action": "get_prompt_report", "run_id": run_id, "session_id": "ordinary-other-session"},
        {"action": "get_prompt_report", "run_id": run_id, "owner": "ordinary-other-owner"},
    ]

    async def scenario():
        context.set_request_context("ordinary-report-read", "ordinary-owner-session")
        for payload in invalid:
            event = await managed_report_read(payload)
            assert event == {"event": "error", "error_code": "invalid_report_request"}
        context.set_request_context("ordinary-no-session-read", None)
        event = await managed_report_read({"action": "get_prompt_report", "run_id": run_id})
        assert event == {"event": "error", "error_code": "report_session_unavailable"}

    asyncio.run(scenario())
    assert app.get_async_task_info()["active_count"] == 0 and completed == []
    assert task_names == [] and forbidden == []


def test_managed_report_withholds_saved_snapshot_until_original_worker_done(
    monkeypatch, tmp_path, owned_app, report_readback_guard, managed_report_context
):
    from amlas.control_plane import runtime_session

    app, completed = owned_app
    forbidden, task_names = report_readback_guard
    context = managed_report_context
    context.set_request_context("ordinary-original-request", "ordinary-owner-session")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AMLAS_HEARTBEAT_SECONDS", "0.01")
    saved, save_release = threading.Event(), threading.Event()
    actual_save = runtime_session.save_session
    saved_refs, calls = [], []

    def save_session(session, **kwargs):
        result = actual_save(session, **kwargs)
        saved_refs.append(session.session_ref)
        saved.set()
        assert save_release.wait(4), "test cleanup must release the original persistence call"
        return result

    async def run_prompt(value, **kwargs):
        calls.append((value, kwargs))
        return ChatResponse(message="Exact original report after worker completion.")

    monkeypatch.setattr(runtime_session, "save_session", save_session)
    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        stream = runtime.agent_invocation({"prompt": "Keep the original worker boundary."})
        event_task = None
        terminal_events = []

        async def finish_original():
            if event_task is not None:
                terminal_events.append(await event_task)
                terminal_events.extend([event async for event in stream])

        try:
            first = await anext(stream)
            run_id = first["run_id"]
            event_task = asyncio.create_task(anext(stream))
            await entered(saved)
            assert saved_refs == [run_id]
            assert app.get_async_task_info()["active_count"] == 0 and len(completed) == 1
            context.set_request_context("ordinary-report-read", "ordinary-owner-session")
            event = await managed_report_read({"action": "get_prompt_report", "run_id": run_id})
            assert_report_unavailable(event, run_id)
            assert calls == [("Keep the original worker boundary.", {})]
            assert task_names == ["normal_prompt"] and forbidden == []
        finally:
            save_release.set()
            try:
                await asyncio.wait_for(finish_original(), 4)
            finally:
                await stream.aclose()
            await idle(app)
            assert len(completed) == 1 and len(saved_refs) == 1
            assert calls == [("Keep the original worker boundary.", {})]
            assert task_names == ["normal_prompt"] and forbidden == []
        original = terminal_events[-1]
        assert original["event"] == "run_completed"
        assert original["status"] == "response_received" and original["worker_finished"] is True
        assert original["lifecycle_verified"] is False and original["quality_success"] is False
        event = await managed_report_read({"action": "get_prompt_report", "run_id": run_id})
        assert event == {
            "event": "prompt_report",
            "status": "available",
            "run_id": run_id,
            "found": True,
            "report": original,
        }
        assert await managed_report_read({"action": "get_prompt_report", "run_id": run_id}) == event

    asyncio.run(scenario())
    assert app.get_async_task_info()["active_count"] == 0 and len(completed) == 1
    assert calls == [("Keep the original worker boundary.", {})]
    assert task_names == ["normal_prompt"] and forbidden == []
