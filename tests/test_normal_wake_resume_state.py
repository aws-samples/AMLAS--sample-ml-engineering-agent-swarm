"""Resume is an idempotent state query and starter with closed, positive states.

The real app module functions, the real BedrockAgentCoreApp task ownership and the
real SDK session context run here. Restore, claim and continuation are scripted
inert fakes: no model, native or provider effect, and no persistent credential.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from types import SimpleNamespace

import boto3
import pytest
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from test_managed_provider_disclosure import fake_installation
from test_managed_provider_disclosure import install as install_overlay_marker
from test_runtime_busy_ownership import entered, idle
from test_runtime_busy_ownership import managed_report_context as managed_report_context
from test_runtime_busy_ownership import owned_app as owned_app
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import api
from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import engine as chat_engine
from amlas.tui.chat.models import ChatResponse

RUN_ID = "1" * 32
SESSION = "declared-wake-sdk-transport-session-0123456789"


def _state_event(state, **extra):
    return {"event": "resume_state", "state": state, "run_id": RUN_ID, **extra}


def _client_error(code, status=400):
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "Op")


TRANSIENT_ERRORS = [
    lambda: _client_error("ThrottlingException"),
    lambda: _client_error("Throttling"),
    lambda: _client_error("ProvisionedThroughputExceededException"),
    lambda: _client_error("InternalServerError", 500),
    lambda: _client_error("ServiceUnavailable", 503),
    lambda: EndpointConnectionError(endpoint_url="https://declared.invalid"),
    lambda: ConnectionClosedError(endpoint_url="https://declared.invalid"),
    lambda: ConnectTimeoutError(endpoint_url="https://declared.invalid"),
    lambda: ReadTimeoutError(endpoint_url="https://declared.invalid"),
    lambda: TimeoutError("declared socket timeout"),
    lambda: ConnectionResetError("declared reset"),
]
PERMANENT_ERRORS = [
    lambda: ValueError("declared contract violation"),
    lambda: TypeError("declared type violation"),
    lambda: PermissionError("declared denied"),
    lambda: RuntimeError("declared unreviewed runtime failure"),
    lambda: _client_error("AccessDeniedException", 403),
    lambda: _client_error("ExpiredTokenException", 400),
    lambda: _client_error("ValidationException", 400),
    lambda: _client_error("ConditionalCheckFailedException", 400),
    lambda: _client_error("ResourceNotFoundException", 400),
    lambda: KeyError("declared missing field"),
    lambda: AssertionError("declared invariant"),
]


@pytest.fixture
def resume(monkeypatch, tmp_path, managed_report_context, owned_app):
    """Scripted restore/claim/continuation; every gate defaults open."""
    app, completed = owned_app
    managed_report_context.set_request_context("declared-wake-request", SESSION)
    monkeypatch.chdir(tmp_path)
    for name in list(runtime._RESUME_WORKERS) if hasattr(runtime, "_RESUME_WORKERS") else []:
        runtime._RESUME_WORKERS.pop(name, None)
    task = {"component": "declared wake DATA"}
    response = ChatResponse(message="Declared continued result.", action="component_completed")
    state = SimpleNamespace(
        app=app,
        completed=completed,
        run_id=RUN_ID,
        request={"action": "resume_prompt_run", "run_id": RUN_ID},
        prepare_calls=0,
        claim_calls=0,
        continue_calls=0,
        snapshot_calls=0,
        engines=[],
        contexts=[],
        prepare_errors=[],
        claim_errors=[],
        continue_errors=[],
        pending=False,
        pauses=[],
        claim_hooks=[],
        continue_hooks=[],
        prepare_entered=threading.Event(),
        prepare_gate=threading.Event(),
        continue_entered=threading.Event(),
        continue_release=threading.Event(),
        response=response,
    )
    state.prepare_gate.set()
    state.continue_release.set()

    class DeclaredContext:
        def has_task_checkpoint(self, *, policy):
            return True

        def load_qualified_task(self, *, policy):
            return task, (), None, None

    def prepare(value, session_hash, *, transport_session_id=None):
        if value != RUN_ID:
            return None  # an original prompt of another run in the same process has no normal context
        state.prepare_calls += 1
        state.prepare_entered.set()
        assert state.prepare_gate.wait(5), "declared restore gate was not released"
        if state.prepare_errors:
            error = state.prepare_errors.pop(0)
            if error is not None:
                raise error
        context = DeclaredContext()
        state.contexts.append(context)
        return context

    class DeclaredScope:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self.value

        def __exit__(self, *exc):
            return False

    class DeclaredEngine:
        def __init__(self, *, approval_policy, normal_restore):
            state.engines.append(self)
            result = (
                {"status": "RUNNING"}
                if state.pending
                else {
                    "status": "trained",
                    "trained": True,
                    "full_training_complete": True,
                }
            )
            self.tool_backend = SimpleNamespace(_normal_original_result=result)

        def claim_normal_completion(self, value, checkpoint_task):
            state.claim_calls += 1
            for hook in state.claim_hooks:
                hook()
            if state.claim_errors:
                error = state.claim_errors.pop(0)
                if error is not None:
                    raise error

        def load_normal_snapshot(self, value):
            state.snapshot_calls += 1

        def continue_normal_completion(self):
            state.continue_calls += 1
            for hook in state.continue_hooks:
                hook()
            state.continue_entered.set()
            assert state.continue_release.wait(5), "declared continuation gate was not released"
            if state.continue_errors:
                error = state.continue_errors.pop(0)
                if error is not None:
                    raise error
            return response

    async def forbidden_prompt(*args, **kwargs):
        raise AssertionError("resume cannot call run_prompt or create a new goal")

    monkeypatch.setattr(checkpoint, "prepare_normal_run", prepare)
    monkeypatch.setattr(checkpoint, "activate_normal_run", DeclaredScope)
    monkeypatch.setattr(chat_engine, "ChatEngine", DeclaredEngine)
    monkeypatch.setattr(api, "run_prompt", forbidden_prompt)
    # Retry pauses are recorded, never slept: the real bounds are asserted from the record.
    monkeypatch.setattr(runtime, "_restore_pause", state.pauses.append, raising=False)
    return state


async def _next(stream):
    return await asyncio.wait_for(anext(stream), timeout=3)


async def _all(payload):
    return [event async for event in runtime.agent_invocation(payload)]


async def _settled(state):
    await idle(state.app)
    deadline = asyncio.get_running_loop().time() + 3
    while state.app.get_async_task_info()["active_count"]:
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.005)


def test_first_request_answers_restoring_before_the_restore_finishes(resume):
    resume.prepare_gate.clear()

    async def scenario():
        stream = runtime.agent_invocation(resume.request)
        try:
            first = await _next(stream)
            assert first == _state_event("restoring")
            assert resume.app.get_async_task_info()["active_count"] == 1
            await entered(resume.prepare_entered)
            assert resume.continue_calls == 0, "restore must still be running"
            second = await _all(resume.request)
            assert second == [_state_event("restoring")]
            assert resume.prepare_calls == 1, "second request must not start a second restore"
            resume.prepare_gate.set()
            events = [first]
            events.extend([event async for event in stream])
            await _settled(resume)
            return events
        finally:
            resume.prepare_gate.set()
            await stream.aclose()
            await idle(resume.app)

    events = asyncio.run(scenario())
    assert [event["event"] for event in events][:2] == ["resume_state", "run_started"]
    assert events[-1]["event"] == "run_completed" and events[-1]["status"] == "response_received"
    assert resume.claim_calls == 1 and resume.continue_calls == 1


def test_state_is_continuing_after_the_continuation_starts_and_completed_after_it_ends(resume):
    resume.continue_release.clear()

    async def scenario():
        stream = runtime.agent_invocation(resume.request)
        try:
            assert await _next(stream) == _state_event("restoring")
            await entered(resume.continue_entered)
            assert await _all(resume.request) == [_state_event("continuing")]
            resume.continue_release.set()
            rest = [event async for event in stream]
            await _settled(resume)
            late = await _all(resume.request)
            return rest, late
        finally:
            resume.continue_release.set()
            await stream.aclose()
            await idle(resume.app)

    rest, late = asyncio.run(scenario())
    assert rest[-1]["event"] == "run_completed"
    assert late == [_state_event("completed")]
    assert resume.prepare_calls == 1 and resume.claim_calls == 1 and resume.continue_calls == 1


def test_disconnect_after_the_first_event_does_not_stop_the_resume_worker(resume):
    resume.continue_release.clear()

    async def scenario():
        stream = runtime.agent_invocation(resume.request)
        assert await _next(stream) == _state_event("restoring")
        await stream.aclose()
        await entered(resume.continue_entered)
        assert await _all(resume.request) == [_state_event("continuing")]
        resume.continue_release.set()
        await _settled(resume)
        assert await _all(resume.request) == [_state_event("completed")]

    asyncio.run(scenario())
    assert resume.claim_calls == 1 and resume.continue_calls == 1


def test_busy_original_run_is_original_active_and_creates_no_resume_worker(resume, monkeypatch):
    original_started, original_release = threading.Event(), threading.Event()

    async def run_prompt(value, **kwargs):
        original_started.set()
        assert original_release.wait(5)
        return ChatResponse(message="Original finished.")

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        original = runtime.agent_invocation({"prompt": "Declared original goal."})
        try:
            started = await _next(original)
            assert started["event"] == "run_started"
            await entered(original_started)
            run_id = started["run_id"]
            request = {"action": "resume_prompt_run", "run_id": run_id}
            answer = await _all(request)
            assert answer == [{"event": "resume_state", "state": "original_active", "run_id": run_id}]
            assert resume.prepare_calls == 0, "a busy original must never start a restore"
            original_release.set()
            [event async for event in original]
            await _settled(resume)
        finally:
            original_release.set()
            await original.aclose()
            await idle(resume.app)

    asyncio.run(scenario())


def test_permanent_restore_failure_ends_failed_with_closed_diagnostic_and_is_not_repeated(
    resume, monkeypatch
):
    resume.prepare_errors.append(ValueError("declared private detail must not leak"))

    async def scenario():
        first = await _all(resume.request)
        second = await _all(resume.request)
        await _settled(resume)
        return first, second

    first, second = asyncio.run(scenario())
    failed = _state_event(
        "failed", diagnostic={"restore_stage": "run_prepare", "exception_class": "ValueError"}
    )
    assert first == [_state_event("restoring"), failed]
    assert second == [failed], "a permanent failure answers failed without a second restore"
    assert resume.prepare_calls == 1 and resume.pauses == []
    assert "private detail" not in str(first + second)


def test_permanent_failure_cooldown_expires_so_a_misclassified_error_cannot_strand_the_run(
    resume, monkeypatch
):
    resume.prepare_errors.append(ValueError("declared first failure"))
    now = [1000.0]
    monkeypatch.setattr(runtime, "_resume_clock", lambda: now[0], raising=False)

    async def scenario():
        assert (await _all(resume.request))[-1]["state"] == "failed"
        assert await _all(resume.request) == [
            _state_event(
                "failed", diagnostic={"restore_stage": "run_prepare", "exception_class": "ValueError"}
            )
        ]
        now[0] += runtime._RESUME_FAILED_COOLDOWN_SECONDS + 1
        events = await _all(resume.request)
        await _settled(resume)
        return events

    events = asyncio.run(scenario())
    assert events[0] == _state_event("restoring") and events[-1]["event"] == "run_completed"
    assert resume.prepare_calls == 2 and resume.continue_calls == 1


@pytest.mark.parametrize("make_error", TRANSIENT_ERRORS)
def test_transient_restore_error_is_retried_with_bounded_full_jitter_then_succeeds(resume, make_error):
    resume.prepare_errors.extend([make_error(), make_error()])

    events = asyncio.run(_all(resume.request))
    assert events[0] == _state_event("restoring")
    assert events[-1]["event"] == "run_completed" and events[-1]["status"] == "response_received"
    assert resume.prepare_calls == 3 and resume.claim_calls == 1 and resume.continue_calls == 1
    assert len(resume.pauses) == 2
    assert 0.0 <= resume.pauses[0] <= 0.5 and 0.0 <= resume.pauses[1] <= 1.0
    assert len({id(engine) for engine in resume.engines}) == len(resume.engines) == 1


@pytest.mark.parametrize("make_error", TRANSIENT_ERRORS[:3])
def test_transient_errors_exhaust_three_attempts_then_failed_transient_and_next_request_restarts(
    resume, make_error
):
    resume.prepare_errors.extend([make_error(), make_error(), make_error()])

    async def scenario():
        first = await _all(resume.request)
        await _settled(resume)
        second = await _all(resume.request)
        await _settled(resume)
        return first, second

    first, second = asyncio.run(scenario())
    assert first[0] == _state_event("restoring")
    assert first[-1]["state"] == "failed_transient"
    assert first[-1]["diagnostic"]["restore_stage"] == "run_prepare"
    assert resume.pauses[0] <= 0.5 and resume.pauses[1] <= 1.0 and len(resume.pauses) == 2
    assert second[0] == _state_event("restoring") and second[-1]["event"] == "run_completed"
    assert resume.prepare_calls == 4 and resume.continue_calls == 1


@pytest.mark.parametrize("make_error", PERMANENT_ERRORS)
def test_permanent_error_classes_are_never_retried(resume, make_error):
    resume.prepare_errors.append(make_error())

    events = asyncio.run(_all(resume.request))
    assert events[0] == _state_event("restoring") and events[-1]["state"] == "failed"
    assert resume.prepare_calls == 1 and resume.pauses == []
    assert resume.claim_calls == 0 and resume.continue_calls == 0


def test_a_value_error_caused_by_a_transient_error_is_retried(resume):
    try:
        raise ConnectTimeoutError(endpoint_url="https://declared.invalid")
    except ConnectTimeoutError as cause:
        try:
            raise ValueError("normal TASK CAS outcome unresolved") from cause
        except ValueError as wrapped:
            resume.prepare_errors.append(wrapped)

    events = asyncio.run(_all(resume.request))
    assert events[-1]["event"] == "run_completed"
    assert resume.prepare_calls == 2 and len(resume.pauses) == 1


def test_claim_failure_is_retried_with_a_fresh_engine_and_the_continuation_runs_once(resume):
    resume.claim_errors.append(EndpointConnectionError(endpoint_url="https://declared.invalid"))

    events = asyncio.run(_all(resume.request))
    assert events[-1]["event"] == "run_completed"
    assert resume.claim_calls == 2 and resume.continue_calls == 1
    assert len(resume.engines) == 2 and resume.engines[0] is not resume.engines[1]


def test_the_continuation_is_never_inside_the_retry(resume):
    resume.continue_errors.append(EndpointConnectionError(endpoint_url="https://declared.invalid"))

    async def scenario():
        events = await _all(resume.request)
        await _settled(resume)
        return events, await _all(resume.request)

    events, again = asyncio.run(scenario())
    assert resume.continue_calls == 1, "a continuation error must not replay the continuation"
    assert resume.claim_calls == 1 and resume.pauses == []
    assert events[-1] == _state_event(
        "failed", diagnostic={"restore_stage": "continuation", "exception_class": "Exception"}
    )
    assert again == [events[-1]]


def test_pending_original_is_not_a_positive_state_and_is_asked_again(resume):
    resume.pending = True

    async def scenario():
        first = await _all(resume.request)
        await _settled(resume)
        second = await _all(resume.request)
        await _settled(resume)
        return first, second

    first, second = asyncio.run(scenario())
    for events in (first, second):
        assert events[0] == _state_event("restoring")
        assert all(event.get("state") not in ("continuing", "completed") for event in events)
    assert resume.snapshot_calls == 2 and resume.continue_calls == 0 and resume.claim_calls == 0


def test_invalid_request_shape_never_reaches_a_positive_state(resume):
    for payload in (
        {"action": "resume_prompt_run", "run_id": "short"},
        {"action": "resume_prompt_run", "run_id": RUN_ID, "extra": 1},
    ):
        events = asyncio.run(_all(payload))
        assert all(event.get("event") != "resume_state" for event in events)
    assert resume.prepare_calls == 0


def test_transient_error_publishing_the_final_report_is_retried_so_the_run_stays_complete(
    resume, monkeypatch
):
    calls = []
    flags = []

    def retain(run_id, session_hash, report, **flag):
        flags.append(flag)
        calls.append(report["status"])
        if len(calls) == 1:
            raise ConnectTimeoutError(endpoint_url="https://declared.invalid")
        return True

    monkeypatch.setattr(checkpoint, "retain_normal_prompt_report", retain)

    async def scenario():
        events = await _all(resume.request)
        await _settled(resume)
        return events, await _all(resume.request)

    events, late = asyncio.run(scenario())
    assert events[-1]["event"] == "run_completed" and events[-1]["status"] == "response_received"
    assert calls == ["response_received", "response_received"] and len(resume.pauses) == 1
    # The resumed run publishes its final report as the resumed pointer on every attempt.
    assert flags == [{"resumed": True}, {"resumed": True}]
    assert late == [_state_event("completed")]


def test_permanent_error_publishing_the_final_report_is_not_retried_and_is_a_closed_failure(
    resume, monkeypatch
):
    calls = []

    def retain(run_id, session_hash, report, **flag):
        calls.append(1)
        raise ValueError("declared report conflict")

    monkeypatch.setattr(checkpoint, "retain_normal_prompt_report", retain)

    async def scenario():
        events = await _all(resume.request)
        await _settled(resume)
        return events, await _all(resume.request)

    events, again = asyncio.run(scenario())
    assert calls == [1] and resume.pauses == []
    assert events[-1]["event"] == "run_completed" and events[-1]["status"] == "failed"
    assert again == [
        _state_event(
            "failed", diagnostic={"restore_stage": "report_retention", "exception_class": "ValueError"}
        )
    ]


def test_real_app_wire_framing_of_the_first_event_is_accepted_by_the_stream_lambda(resume):
    """The actual BedrockAgentCoreApp serialises the closed event; the Lambda parser reads that wire form."""
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    from amlas.async_callback import lambda_handler as callback_source

    resume.app.entrypoint(runtime.agent_invocation)
    headers = {"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": SESSION}
    with TestClient(resume.app) as client:
        response = client.post("/invocations", json=resume.request, headers=headers)
    assert response.status_code == 200, response.text[:300]
    first = response.content.split(b"\n", 1)[0]
    assert first.startswith(b"data:"), first[:80]
    assert (
        response.content.count(b'"event": "resume_state"') + response.content.count(b'"event":"resume_state"')
        >= 1
    )

    class Services:
        def now(self):
            return 1000.0

    class Wire:
        def __init__(self, raw):
            self.raw, self.at = raw, 0

        def read(self, amount):
            chunk = self.raw[self.at : self.at + amount]
            self.at += len(chunk)
            return chunk

    state = callback_source._first_resume_state(Wire(response.content), Services(), 1020.0, RUN_ID)
    assert state == "restoring"
    assert resume.claim_calls == 1 and resume.continue_calls == 1


def test_resume_path_keeps_the_process_wide_controlled_provider_disclosure_in_place(resume, monkeypatch):
    """The model reads `execution_provider` in its tool results; `unverified` made it stop in runs with a real model.

    The overlay marker is installed once per process by the deployed wrapper before the app serves, so a
    replacement process has it before its first request. The resume path must leave it untouched and must not
    depend on any state of an earlier original run: the disclosure at the claim and at the first tool result of
    the continuation is the controlled one, and the client factory that carries the marker is the same object.
    """
    from amlas.control_plane import provider_disclosure as disclosure

    install_overlay_marker(monkeypatch, fake_installation())
    factory = boto3.session.Session.client
    seen = []

    def read():
        result = disclosure.execution_provider(disclosure.TRAINING)
        seen.append(
            (result["mode"], result["sagemaker_job_backing"], boto3.session.Session.client is factory)
        )

    resume.claim_hooks.append(read)
    resume.continue_hooks.append(read)
    events = asyncio.run(_all(resume.request))
    assert events[-1]["event"] == "run_completed" and events[-1]["status"] == "response_received"
    assert seen == [("declared_controlled", "controlled_provider", True)] * 2
    assert boto3.session.Session.client is factory


def test_resume_without_the_overlay_marker_would_show_the_model_unverified(resume):
    """Control for the test above: this is what a process without the marker prints at the same points."""
    from amlas.control_plane import provider_disclosure as disclosure

    seen = []
    resume.continue_hooks.append(lambda: seen.append(disclosure.execution_provider(disclosure.TRAINING)))
    events = asyncio.run(_all(resume.request))
    assert events[-1]["event"] == "run_completed"
    assert [(item["mode"], item["reason"]) for item in seen] == [("unverified", "marker_absent")]


# A transient restore attempt that the loop repeats names its fixed reason, its attempt and its reviewed category; only
# the attempt that ends the restore writes the final refusal. Since 2026-11-11 that refusal carries `reason=final`,
# its fixed stage and its reviewed category in the text, because logging extras do not reach the runtime log stream;
# the bare text `normal_restore_unavailable` is no longer written. The three test names below keep their ids.
RETRY_LINE = re.compile(
    r"normal_restore_unavailable reason=transient_retry attempt=(?P<attempt>[0-9]) attempts=3"
    r" restore_stage=(?P<stage>[a-z_]+) exception_class=(?P<cls>[A-Za-z]+)"
)
FINAL_LINE = re.compile(
    r"normal_restore_unavailable reason=final restore_stage=(?P<stage>[a-z_]+) exception_class=(?P<cls>[A-Za-z]+)"
)


def _restore_lines(caplog):
    return [
        record
        for record in caplog.records
        if record.name == "amlas.app" and record.getMessage().startswith("normal_restore_unavailable")
    ]


def test_a_healed_transient_restore_attempt_names_its_reason_and_attempt_never_a_bare_refusal(resume, caplog):
    resume.prepare_errors.extend(
        [_client_error("ThrottlingException"), TimeoutError("declared socket timeout")]
    )
    caplog.set_level(logging.WARNING, logger="amlas.app")

    events = asyncio.run(_all(resume.request))
    assert events[-1]["event"] == "run_completed" and resume.prepare_calls == 3
    records = _restore_lines(caplog)
    assert [
        record.getMessage() for record in records if record.getMessage() == "normal_restore_unavailable"
    ] == [], "the bare text is never written"
    assert [record.getMessage() for record in records if FINAL_LINE.fullmatch(record.getMessage())] == [], (
        "a healed transient attempt must not read as a final refusal"
    )
    matches = [RETRY_LINE.fullmatch(record.getMessage()) for record in records]
    assert len(matches) == 2 and all(matches), [record.getMessage() for record in records]
    assert [(m["attempt"], m["stage"], m["cls"]) for m in matches] == [
        ("1", "run_prepare", "Exception"),
        ("2", "run_prepare", "TimeoutError"),
    ]
    assert all(record.levelno == logging.WARNING and record.exc_info is None for record in records)
    assert "declared socket timeout" not in caplog.text and "ThrottlingException" not in caplog.text


def test_exhausted_transient_attempts_end_with_the_unchanged_bare_refusal(resume, caplog):
    resume.prepare_errors.extend([_client_error("ThrottlingException") for _ in range(3)])
    caplog.set_level(logging.WARNING, logger="amlas.app")

    events = asyncio.run(_all(resume.request))
    assert events[-1]["state"] == "failed_transient"
    records = _restore_lines(caplog)
    assert len(records) == 3, [record.getMessage() for record in records]
    assert [RETRY_LINE.fullmatch(record.getMessage())["attempt"] for record in records[:2]] == ["1", "2"]
    final = records[2]
    assert final.getMessage() == (
        "normal_restore_unavailable reason=final restore_stage=run_prepare exception_class=Exception"
    )
    assert final.args == ("run_prepare", "Exception") and final.exc_info is None
    assert [bool(FINAL_LINE.fullmatch(record.getMessage())) for record in records] == [False, False, True]
    assert "ThrottlingException" not in caplog.text
    assert (final.restore_stage, final.exception_class) == ("run_prepare", "Exception")


def test_a_permanent_error_after_a_healed_attempt_ends_with_the_unchanged_bare_refusal(resume, caplog):
    resume.prepare_errors.extend(
        [_client_error("ThrottlingException"), ValueError("declared contract violation")]
    )
    caplog.set_level(logging.WARNING, logger="amlas.app")

    events = asyncio.run(_all(resume.request))
    assert events[-1]["state"] == "failed"
    records = _restore_lines(caplog)
    assert [record.getMessage() for record in records] == [
        "normal_restore_unavailable reason=transient_retry attempt=1 attempts=3 restore_stage=run_prepare"
        " exception_class=Exception",
        "normal_restore_unavailable reason=final restore_stage=run_prepare exception_class=ValueError",
    ]
    assert records[1].args == ("run_prepare", "ValueError")
    assert (records[1].restore_stage, records[1].exception_class) == ("run_prepare", "ValueError")
    assert "declared contract violation" not in caplog.text


def test_a_failed_continuation_writes_exactly_one_final_line_that_names_the_continuation_stage(
    resume, caplog
):
    resume.continue_errors.append(ValueError("declared contract violation"))
    caplog.set_level(logging.WARNING, logger="amlas.app")

    events = asyncio.run(_all(resume.request))
    assert events[-1] == _state_event(
        "failed", diagnostic={"restore_stage": "continuation", "exception_class": "ValueError"}
    )
    records = _restore_lines(caplog)
    assert [record.getMessage() for record in records] == [
        "normal_restore_unavailable reason=final restore_stage=continuation exception_class=ValueError"
    ]
    assert records[0].args == ("continuation", "ValueError")
    assert (records[0].restore_stage, records[0].exception_class) == ("continuation", "ValueError")
    assert resume.continue_calls == 1 and "declared contract violation" not in caplog.text
