"""A guardrail stop on the model call of a worker that took over a run after its fine-tuning job.

The first worker of a normal run saved its checkpoint and ended while the fine-tuning job ran. The job completed
and a resume request restored the run in a replacement worker. A model call of that worker goes through the SDK's
real ``BedrockModel`` and is stopped by a guardrail (a stub client replays one fixed response stream). The SDK then
redacts the admitted input, and the product holds the changed history.

The tests pin what a client gets. The run ends with the held report, stored as the report of the resumed run and
readable afterwards; the first worker's report keeps its own pointer. A stop with the shape that one stopped call
of the model service returned names the blocked entity in that report. A hold on a later model call, after a tool
cycle that submitted a job, ends the same way. So does a hold that the backend's own check before a model call sets
inside the call, with no model call: the held end is taken for every hold that a callback of the backend sets
inside a call that raises while no fence refuses. The held end builds its response, appends the runtime recovery
record and qualifies again, in that order. Every check the held end keeps still fails closed: a replaced backend,
agent, claim or snapshot, a changed system prompt, a removed approval, a changed continuation token or status in
the run record, and a change between the two qualifications. Controls pin what stays as it was: a refusal by a
fence of the continuation, a failure without a hold, a hold seen after the call returned, an interrupt outside
``Exception`` raised with a hold set, a refused report retention, a request in a process without resume memory, and
a second run in the same process.

Scripted models and strict neutral native ports only. The expectation for a stop without a trace follows the
installed SDK, as in ``test_bedrock_guardrail_stop_without_trace.py``. The fixtures establish source behaviour, not
native guardrail classification, native storage or a deployed run.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
from types import SimpleNamespace

import normal_sft_cold_fixture as cold
import pytest
import test_bedrock_guardrail_stop_without_trace as g
import test_recovery_notice as declared
from normal_sft_cold_fixture import close, cloud_data, original_precommit, setup
from strands.hooks import BeforeInvocationEvent, BeforeModelCallEvent, MessageAddedEvent
from test_held_report_guardrail_assessment import EMAIL_RECORD, RECORDED_STOP
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_generation_originals import publish_generation
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import recovery_notice
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.tool_binding_backend import AgentTurn

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")

# Upper bound for the waits of one scenario. A slow machine needs the room; a stage that is never reached still
# fails when the bound ends.
WAIT = g.LIVENESS_WAIT_SECONDS
EARLIER_ACTIONS = (
    "This hold does not undo earlier actions, and any jobs already submitted may still be running."
)
HELD_TEXT = g.HOLD_TEXT + " " + EARLIER_ACTIONS
RECORD_HEADING = "\n\n## Runtime recovery record\n"
RESUMED_POINTER = "resumed_report_checkpoint_ref"
FAILED_CONTINUATION = {"restore_stage": "continuation", "exception_class": "ValueError"}
# The fixed closed error of the continuation call, and the refusal of the first check of the normal path.
CLOSED_ERROR = "normal completion invocation refused or unresolved"
NORMAL_PATH_REFUSAL = "normal continuation lifetime or admission changed"
# The hold line of the backend's own check before a model call.
BEFORE_MODEL_HOLD_LINE = "AMLAS_INPUT_HOLD phase=before_model_qualification reason=callback_refused"


async def requested(payload):
    return [event async for event in runtime.agent_invocation(payload) if event.get("event") != "heartbeat"]


def record_of(state):
    return json.loads(state.ports.items["PROMPT_RUN#" + state.run_id]["record"]["S"])


def rewrite_record(state, transform):
    """Change the stored run record as another writer would: the body and its hash together."""
    item = state.ports.items["PROMPT_RUN#" + state.run_id]
    record = json.loads(item["record"]["S"])
    transform(record)
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    item["record"] = {"S": raw}
    item["record_sha256"] = {"S": hashlib.sha256(raw.encode()).hexdigest()}


def effects(state, seen):
    return (
        seen.calls,
        len(seen.client.requests),
        len(state.native.calls),
        len(state.native.processing),
        len(state.work.engines),
    )


def first_tool_use_only(event):
    for name in ("contentBlockStart", "contentBlockDelta", "contentBlockStop"):
        if name in event:
            return event[name]["contentBlockIndex"] == 0
    return True


def stopped_replacement(
    monkeypatch,
    tmp_path,
    caplog,
    *,
    blocked=True,
    trace=None,
    variant=None,
    scripted_calls=0,
    during_stop=None,
    while_resuming=None,
    after=None,
):
    """One run up to its submitted fine-tuning job; the first worker ends, the job completes, one resume request.

    The model calls of the replacement are decided here. The first ``scripted_calls`` calls answer with the first
    tool use of the scripted model; every later call is the replayed guardrail stop. With ``blocked`` the stop carries
    ``trace``, by default the fixture's blocked-policy trace. ``during_stop(state, engine)``
    runs inside the stopped call before the stop is replayed. ``while_resuming(state, reader)`` runs beside the
    resume request. ``after(state, seen)`` runs in the same event loop once the resume request has ended.
    """
    seen = SimpleNamespace(armed=False, calls=0, state=None, bedrock=None, client=None, after=None)
    scripted = cold.FullGraphModel.stream

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if not seen.armed:
            async for event in scripted(self, messages, tool_specs, system_prompt, **kwargs):
                yield event
            return
        seen.calls += 1
        if seen.calls <= scripted_calls:
            async for event in scripted(self, messages, tool_specs, system_prompt, **kwargs):
                if first_tool_use_only(event):
                    yield event
            return
        self.calls += 1
        if during_stop is not None:
            during_stop(seen.state, seen.state.work.engines[-1])
        async for event in seen.bedrock.stream(messages, tool_specs, system_prompt, **kwargs):
            yield event

    # The fixture keeps the model's stream as it is at setup, so the switch sits on the class before the setup.
    monkeypatch.setattr(cold.FullGraphModel, "stream", stream)

    async def scenario():
        state = seen.state = setup(monkeypatch, tmp_path, variant=variant)
        try:
            await original_precommit(state, monkeypatch)
            await close_original_worker(state.work, state.reader)
            seen.bedrock, seen.client = g.bound_model(
                monkeypatch,
                trace=(trace or g.BLOCKED_TRACE) if blocked else None,
                trace_mode="enabled" if blocked else "disabled",
            )
            cloud_data(state, completed=True)  # installs the completed output of the fine-tuning job
            state.native.states[state.original.id] = "Completed"
            seen.report_before = await requested({"action": "get_prompt_report", "run_id": state.run_id})
            seen.first_report = record_of(state).get("report_checkpoint_ref")
            training, processing = len(state.native.calls), len(state.native.processing)
            caplog.clear()  # the lines read below are those of the resume request
            seen.armed = True
            with caplog.at_level(logging.WARNING):
                reader = asyncio.ensure_future(
                    requested({"action": "resume_prompt_run", "run_id": state.run_id})
                )
                try:
                    if while_resuming is not None:
                        await while_resuming(state, reader)
                    seen.resume = await asyncio.wait_for(reader, WAIT)
                finally:
                    if not reader.done():
                        reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
            seen.record = record_of(state)
            seen.new_training = len(state.native.calls) - training
            seen.new_processing = len(state.native.processing) - processing
            seen.hold_lines = g.hold_lines(caplog)
            seen.restore_lines = [
                record.getMessage().split(" ", 1)[0]
                for record in caplog.records
                if record.getMessage().startswith("normal_restore")
            ]
            seen.report_after = await requested({"action": "get_prompt_report", "run_id": state.run_id})
            if after is not None:
                seen.after = await after(state, seen)
        finally:
            await close(state)

    asyncio.run(scenario())
    return seen


def shape(events):
    return [(event["event"], event.get("state")) for event in events]


def assert_held_end(run, *, hold_line=g.HOLD_LINE):
    """The resume request ends with the held report of the resumed run; returns that final event."""
    assert shape(run.resume) == [
        ("resume_state", "restoring"),
        ("run_started", None),
        ("run_completed", None),
    ]
    terminal = run.resume[-1]
    assert terminal["status"] == "response_received" and terminal["worker_finished"] is True
    response = terminal["response"]
    assert response["action"] == "input_admission_held"
    head, heading, block = response["message"].partition(RECORD_HEADING)
    # The fixed held message with the sentence about earlier actions, then the runtime recovery record.
    assert head == HELD_TEXT
    assert heading and block.splitlines()[-1].startswith("Records: ")
    assert re.search(r"\bcompleted\b", response["message"], re.IGNORECASE) is None
    assert response["data"]["input_admission"]["status"] == "held"
    # The held report lists no tool, carries no intervention outcome and shows neither provider nor redaction text.
    assert response["data"]["tools_used"] == [] and "guardrail_outcome" not in response["data"]
    shown = response["message"] + json.dumps(response["data"])
    assert g.TEXT not in shown and g.REDACTED not in shown
    # The product held once, by default at the changed history; the restore neither failed nor was reported as
    # unavailable.
    assert run.hold_lines == [hold_line]
    assert run.restore_lines == ["normal_restore_completed"]
    # The report is stored as the report of the resumed run and is what a report request returns.
    assert type(run.record.get(RESUMED_POINTER)) is dict and run.record["stage"] == "sft"
    # The first worker's report keeps the pointer it had before the resume request: the held report is stored
    # beside it.
    assert type(run.first_report) is dict and run.record["report_checkpoint_ref"] == run.first_report
    assert run.report_after == [
        {
            "event": "prompt_report",
            "status": "available",
            "run_id": run.state.run_id,
            "found": True,
            "report": terminal,
        }
    ]
    return terminal


def assert_no_report_of_the_stop(run, *, record_readable=True):
    """The resume request failed closed at the continuation: no response, no resumed report, no new job."""
    # The stop was held by the history check inside the model call, and no fence of the continuation refused:
    # this is the case in which the held end runs its own checks.
    assert len(run.hold_lines) == 1, run.hold_lines
    assert run.hold_lines[0].startswith("AMLAS_INPUT_HOLD phase=history_qualification reason=")
    assert shape(run.resume) == [
        ("resume_state", "restoring"),
        ("run_started", None),
        ("resume_state", "failed"),
    ]
    assert run.resume[-1]["diagnostic"] == FAILED_CONTINUATION
    assert all("response" not in event for event in run.resume)
    assert RESUMED_POINTER not in run.record
    if record_readable:
        # A report request still returns what the first worker left.
        assert run.report_after == run.report_before
    else:
        # A run record that no reader accepts gives no report at all.
        assert run.report_after == [
            {"event": "prompt_report", "status": "unavailable", "run_id": run.state.run_id, "found": False}
        ]
    assert EARLIER_ACTIONS not in json.dumps(run.report_after)
    assert run.new_training == run.new_processing == 0


async def later_requests(state, seen):
    """A report, a status and a resume request of the same process after the run ended."""
    before = effects(state, seen)
    status = await requested({"action": "get_prompt_run_status", "run_id": state.run_id})
    again = await requested({"action": "resume_prompt_run", "run_id": state.run_id})
    return SimpleNamespace(status=status, again=again, started_nothing=effects(state, seen) == before)


@pytest.mark.parametrize("case", ["no_trace", "blocked_trace"])
def test_a_guardrail_stop_on_the_replacement_call_ends_the_run_with_the_held_report(
    monkeypatch, tmp_path, caplog, case
):
    blocked = case == "blocked_trace"
    run = stopped_replacement(monkeypatch, tmp_path, caplog, blocked=blocked, after=later_requests)
    run_id = run.state.run_id
    # The replacement asked the model once and submitted no job.
    assert run.calls == 1 and len(run.client.requests) == 1
    assert run.new_training == run.new_processing == 0
    if blocked or g.REDACTS_WITHOUT_TRACE:
        # A blocked trace redacts on every SDK version; without a trace only the newer SDK does.
        terminal = assert_held_end(run)
        assessment = terminal["response"]["data"].get("guardrail_assessment")
        if blocked:
            # The held report carries what the guardrail reported in that turn: the blocked policy, no text.
            assert assessment["assessment_status"] == "complete" and assessment["action"] == "BLOCKED"
            assert [
                (record["direction"], record["policy"], record["category"], record["action"])
                for record in assessment["records"]
            ] == [("INPUT", "CONTENT", "VIOLENCE", "BLOCKED")]
        else:
            assert assessment is None
    else:
        # An older SDK leaves the admitted input alone without a trace: the stop is a plain intervention.
        terminal = run.resume[-1]
        assert terminal["event"] == "run_completed" and terminal["status"] == "response_received"
        assert terminal["response"]["action"] == "guardrail_intervened"
        assert run.hold_lines == [] and type(run.record.get(RESUMED_POINTER)) is dict
    # The status of a run in the fine-tuning stage is unchanged, and a later resume in the process that ended the
    # run answers completed. Neither request called the model, submitted a job or built an engine.
    assert run.after.status == [
        {"event": "prompt_run_status", "status": "unavailable", "run_id": run_id, "found": False}
    ]
    assert run.after.again == [{"event": "resume_state", "state": "completed", "run_id": run_id}]
    assert run.after.started_nothing


def test_a_stop_of_the_recorded_shape_on_the_replacement_call_names_the_blocked_entity(
    monkeypatch, tmp_path, caplog
):
    """The stop has the shape that one stopped call of the model service returned.

    A personal-data entity was blocked on the output side, its policy has no ``regexes`` key, and the trace holds an
    input assessment with only details and metrics. The held end takes the assessment from the capture of its
    invocation, so the held report of the resumed run names the blocked entity, as
    ``test_held_report_guardrail_assessment.py`` requires for a run held in its first worker.
    """
    run = stopped_replacement(monkeypatch, tmp_path, caplog, trace=RECORDED_STOP)
    # The SDK replaced the input for a block on the output side too. The replacement asked the model once and
    # submitted no job, and the run ended with the held report.
    assert run.calls == 1 and len(run.client.requests) == 1
    assert run.new_training == run.new_processing == 0
    terminal = assert_held_end(run)
    assessment = terminal["response"]["data"]["guardrail_assessment"]
    # The record names the blocked entity, and nothing else of the trace is kept.
    assert assessment["records"] == [EMAIL_RECORD]
    # The input assessment with only details and metrics leaves the coverage open: the status is not complete and
    # the summary action is not a known one.
    assert assessment["assessment_status"] == "partial" and assessment["action"] == "UNKNOWN"


async def complete_the_submitted_generation(state, reader):
    """Publish the native output of the one generation job the replacement submitted; its tool call waits for it."""

    def submitted():
        return (
            len(state.work.engines) > 1
            and len(state.work.engines[-1].tool_backend._generation_store._records) == 1
        )

    await cold.wait_for(state, reader, submitted, "one registered generation original", timeout=WAIT)
    handle = next(iter(state.work.engines[-1].tool_backend._generation_store._records.values()))
    ref = handle.extra["run_binding"]["prompts_ref"]
    uri = ref["uri"].split("/", 3)
    body = state.objects.get_object(
        Bucket=uri[2],
        Key=uri[3],
        VersionId=ref["version_id"],
        ExpectedBucketOwner=handle.extra["catalog"].account_id,
    )["Body"]
    try:
        raw = body.read(ref["byte_count"] + 1)
    finally:
        body.close()
    publish_generation(state.objects, handle, [json.loads(line) for line in raw.splitlines()])
    state.native.states[handle.id] = "Completed"


def test_a_hold_on_a_later_model_call_after_a_tool_cycle_ends_the_same_way(monkeypatch, tmp_path, caplog):
    """The first call of the replacement submits one generation job; the call after its result is stopped."""
    run = stopped_replacement(
        monkeypatch,
        tmp_path,
        caplog,
        variant="spec-first",
        scripted_calls=1,
        while_resuming=complete_the_submitted_generation,
        after=later_requests,
    )
    terminal = assert_held_end(run)
    # One job was submitted in this invocation before the hold, and the model was asked twice.
    assert run.new_training == 1 and run.new_processing == 0
    assert run.calls == 2 and len(run.client.requests) == 1
    # The held report names no job; its fixed sentence says that earlier actions stand.
    assert terminal["response"]["data"]["tools_used"] == []
    assert EARLIER_ACTIONS in terminal["response"]["message"]
    assert run.after.status == [
        {"event": "prompt_run_status", "status": "unavailable", "run_id": run.state.run_id, "found": False}
    ]
    assert run.after.again == [{"event": "resume_state", "state": "completed", "run_id": run.state.run_id}]
    assert run.after.started_nothing


def test_a_hold_set_by_the_backend_before_the_model_call_ends_the_same_way(monkeypatch, tmp_path, caplog):
    """The held end is taken for every hold a callback of the backend sets inside the call, whatever its cause.

    The warm state of the replacement closes inside the call, before the first model call. The backend's own check
    before a model call then refuses the research binding qualification, sets the hold and raises. That check was
    registered when the Agent was built, so it runs before the fences of the continuation and no fence refuses.
    """
    continuation = ChatEngine.continue_normal_completion
    premise = {}

    def continue_with_a_warm_state_that_closes(self):
        backend = self.tool_backend
        premise["research_binding"] = backend._research_binding is not None
        premise["warm_owner"] = backend._warm_owner is not None

        def close_the_warm_state(event):
            with backend._warm_capture_lock:
                backend._warm_closed = "refused"
            premise["closed_inside_the_call"] = True

        self._normal_agent.hooks.add_callback(BeforeInvocationEvent, close_the_warm_state)
        return continuation(self)

    monkeypatch.setattr(ChatEngine, "continue_normal_completion", continue_with_a_warm_state_that_closes)
    run = stopped_replacement(monkeypatch, tmp_path, caplog, after=later_requests)
    # The premise of the case: a bound research binding, a warm owner, and the state closed inside the call.
    assert premise == {"research_binding": True, "warm_owner": True, "closed_inside_the_call": True}
    terminal = assert_held_end(run, hold_line=BEFORE_MODEL_HOLD_LINE)
    # No model was asked, the guardrail saw no request and no job was submitted.
    assert run.calls == 0 and len(run.client.requests) == 0
    assert run.new_training == run.new_processing == 0
    assert terminal["response"]["data"].get("guardrail_assessment") is None
    assert run.after.status == [
        {"event": "prompt_run_status", "status": "unavailable", "run_id": run.state.run_id, "found": False}
    ]
    assert run.after.again == [{"event": "resume_state", "state": "completed", "run_id": run.state.run_id}]
    assert run.after.started_nothing


# ------------------------------------------------------------------ each check the held end keeps fails closed


def replace_backend(state, engine):
    engine.tool_backend = copy.copy(engine.tool_backend)


def replace_agent(state, engine):
    engine._normal_agent = SimpleNamespace(system_prompt=engine._normal_agent.system_prompt)


def replace_claim(state, engine):
    engine._normal_completion = (*engine._normal_completion,)


def replace_snapshot(state, engine):
    engine._normal_completion_snapshot = engine._normal_completion_snapshot + " "


@pytest.mark.parametrize(
    "replace",
    [replace_backend, replace_agent, replace_claim, replace_snapshot],
    ids=["backend", "agent", "claim", "snapshot"],
)
def test_a_backend_agent_claim_or_snapshot_replaced_during_the_stopped_call_fails_closed(
    monkeypatch, tmp_path, caplog, replace
):
    run = stopped_replacement(monkeypatch, tmp_path, caplog, during_stop=replace)
    assert_no_report_of_the_stop(run)


def test_a_system_prompt_changed_during_the_stopped_call_fails_closed(monkeypatch, tmp_path, caplog):
    def change_prompt(state, engine):
        engine._normal_agent.system_prompt = engine._normal_agent.system_prompt + " Changed after the notice."

    run = stopped_replacement(monkeypatch, tmp_path, caplog, during_stop=change_prompt)
    assert_no_report_of_the_stop(run)


def test_an_approval_removed_during_the_stopped_call_fails_closed(monkeypatch, tmp_path, caplog):
    def remove_approval(state, engine):
        engine.tool_backend.approval_policy.allowlist.remove("qwen_sft_flow")

    run = stopped_replacement(monkeypatch, tmp_path, caplog, during_stop=remove_approval)
    assert_no_report_of_the_stop(run)


def another_token(record):
    record["continuation"]["token"] = "0" * 32


def another_status(record):
    # The readers of the run record accept one continuation status only, so this record has no reader left.
    record["continuation"]["status"] = "released"


@pytest.mark.parametrize("change", [another_token, another_status], ids=["token", "status"])
def test_a_continuation_token_or_status_changed_in_the_record_fails_closed(
    monkeypatch, tmp_path, caplog, change
):
    run = stopped_replacement(
        monkeypatch,
        tmp_path,
        caplog,
        during_stop=lambda state, engine: rewrite_record(state, change),
    )
    assert_no_report_of_the_stop(run, record_readable=change is another_token)


def test_a_change_between_the_two_qualifications_fails_closed(monkeypatch, tmp_path, caplog):
    """The run record changes after the recovery record was appended to the held response."""
    append = recovery_notice.append_recovery_report
    seen = {"state": None, "appended": 0}

    def append_then_change(response, paragraph):
        result = append(response, paragraph)
        seen["appended"] += 1
        rewrite_record(seen["state"], another_token)
        return result

    def remember(state, engine):
        seen["state"] = state

    monkeypatch.setattr(recovery_notice, "append_recovery_report", append_then_change)
    run = stopped_replacement(monkeypatch, tmp_path, caplog, during_stop=remember)
    assert seen["appended"] == 1, "the held end appended the recovery record before the run record changed"
    assert_no_report_of_the_stop(run)


# ------------------------------------------------------------------ controls through the runtime entry


def test_a_refused_report_retention_is_not_completed(monkeypatch, tmp_path, caplog):
    retain = checkpoint.retain_normal_prompt_report

    def refuse_resumed(run_id, session_hash, report, *, resumed=False):
        if resumed:
            raise ValueError("declared refusal of the resumed report")
        return retain(run_id, session_hash, report)

    monkeypatch.setattr(checkpoint, "retain_normal_prompt_report", refuse_resumed)
    run = stopped_replacement(monkeypatch, tmp_path, caplog, after=later_requests)
    answers = [*run.resume, *run.after.again]
    assert not any(event.get("state") == "completed" for event in answers)
    assert not any(event.get("status") == "response_received" for event in answers)
    assert shape(run.after.again) == [("resume_state", "failed")]
    assert RESUMED_POINTER not in run.record and run.report_after == run.report_before
    assert run.after.started_nothing


def test_a_request_in_a_process_without_resume_memory_answers_restoring_then_failed(
    monkeypatch, tmp_path, caplog
):
    async def without_memory(state, seen):
        runtime._RESUME_WORKERS.clear()
        before = effects(state, seen)
        again = await requested({"action": "resume_prompt_run", "run_id": state.run_id})
        return SimpleNamespace(again=again, started_nothing=effects(state, seen)[:4] == before[:4])

    run = stopped_replacement(monkeypatch, tmp_path, caplog, after=without_memory)
    # The continuation of the run has an owner, so a new process restores nothing and asks no model.
    assert shape(run.after.again) == [("resume_state", "restoring"), ("resume_state", "failed")]
    assert run.after.again[-1]["diagnostic"] == {
        "restore_stage": "snapshot_load",
        "exception_class": "ValueError",
    }
    assert run.after.started_nothing


# ------------------------------------------------------------------ the continuation alone, declared seams


def declared_continuation(monkeypatch, *, admission=None, before_model=None):
    """The declared continuation of ``test_recovery_notice`` with a recorder on every step of its end.

    ``admission`` is registered as a message callback: it runs when the model's answer is added, as the product's
    history check does. ``before_model`` is registered before the fences of the continuation.
    """
    calls = []
    engine, agent, model, context, _prefix, _snapshot = declared._engine(declared._row())
    backend = engine.tool_backend

    def held_turn(*, guardrail_assessment=None):
        calls.append("held_turn")
        return AgentTurn(final_text=g.HOLD_TEXT, executions=[], guardrail_assessment=guardrail_assessment)

    backend._input_hold_turn = held_turn
    qualify, respond = context.qualify_completion, engine._agent_turn_response

    def qualified(*args, **kwargs):
        calls.append("qualify")
        return qualify(*args, **kwargs)

    def responded(turn):
        calls.append("response")
        return respond(turn)

    append = recovery_notice.append_recovery_report

    def appended(response, paragraph):
        calls.append("record")
        return append(response, paragraph)

    context.qualify_completion = qualified
    engine._agent_turn_response = responded
    monkeypatch.setattr(recovery_notice, "append_recovery_report", appended)
    if admission is not None:
        agent.hooks.add_callback(MessageAddedEvent, lambda event: admission(backend, calls))
    if before_model is not None:
        agent.hooks.add_callback(BeforeModelCallEvent, lambda event: before_model(agent, backend, calls))
    return SimpleNamespace(engine=engine, agent=agent, model=model, backend=backend, calls=calls)


def hold_the_answer(backend, calls):
    with backend._input_lock:
        backend._input_hold = True
    calls.append("hold")
    raise RuntimeError("declared admission hold")


def test_the_held_path_runs_response_then_record_then_qualification(monkeypatch):
    run = declared_continuation(monkeypatch, admission=hold_the_answer)
    response = run.engine.continue_normal_completion()
    # After the hold: one qualification, the held turn, its response, the recovery record, one more qualification.
    assert run.calls[run.calls.index("hold") :] == [
        "hold",
        "qualify",
        "held_turn",
        "response",
        "record",
        "qualify",
    ]
    head, heading, block = response.message.partition(RECORD_HEADING)
    assert head == HELD_TEXT and heading and block.splitlines()[-1].startswith("Records: ")
    assert len(run.model.system_prompts) == 1, "nothing was invoked again"
    assert run.backend._input_hold is True


def drift_the_prompt(agent, backend, calls):
    agent.system_prompt = "DRIFT"


def hold_without_refusing(agent, backend, calls):
    with backend._input_lock:
        backend._input_hold = True
    calls.append("hold")


@pytest.mark.parametrize("before_model", [drift_the_prompt, hold_without_refusing], ids=["prompt", "hold"])
def test_a_fence_refusal_still_fails_closed(monkeypatch, before_model):
    """A fence of the continuation refuses the model call: with or without a hold set before it, nothing is returned."""
    run = declared_continuation(monkeypatch, before_model=before_model)
    # The closed error by its text: the checks of the held end refuse with another ValueError.
    with pytest.raises(ValueError, match=CLOSED_ERROR):
        run.engine.continue_normal_completion()
    assert run.model.system_prompts == [], "the fence refused before the model was asked"
    assert run.backend._input_hold is True
    assert not {"held_turn", "response", "record"} & set(run.calls)


def test_a_failure_without_a_hold_fails_as_before(monkeypatch):
    def fail_the_answer(backend, calls):
        calls.append("failure")
        raise RuntimeError("declared failure without a hold")

    run = declared_continuation(monkeypatch, admission=fail_the_answer)
    with pytest.raises(ValueError, match=CLOSED_ERROR):
        run.engine.continue_normal_completion()
    assert run.calls[-1] == "failure" and "held_turn" not in run.calls
    assert run.backend._input_hold is True, "the refused continuation keeps the input hold"


def hold_without_failing(backend, calls):
    with backend._input_lock:
        backend._input_hold = True
    calls.append("hold")


def test_a_hold_seen_after_a_normal_return_keeps_the_refusal_of_the_normal_path(monkeypatch):
    """The held end needs a call that raised. Here a callback sets the hold without failing and the call returns."""
    run = declared_continuation(monkeypatch, admission=hold_without_failing)
    with pytest.raises(ValueError, match=NORMAL_PATH_REFUSAL):
        run.engine.continue_normal_completion()
    assert len(run.model.system_prompts) == 1, "the model answered and the call returned"
    assert run.calls[-1] == "hold", "nothing ran after the hold: no qualification, no held turn, no response"
    assert not {"held_turn", "response", "record"} & set(run.calls)
    assert run.backend._input_hold is True


class WorkerInterrupt(BaseException):
    """Stands for an interrupt of the worker; not a subclass of ``Exception``."""


def hold_then_interrupt(backend, calls):
    with backend._input_lock:
        backend._input_hold = True
    calls.append("hold")
    raise WorkerInterrupt("declared interrupt of the worker")


def test_an_interrupt_outside_the_exception_class_passes_through_with_a_hold_set(monkeypatch):
    """The handler of the call catches ``Exception`` only: an interrupt is neither held nor replaced."""
    run = declared_continuation(monkeypatch, admission=hold_then_interrupt)
    with pytest.raises(WorkerInterrupt):
        run.engine.continue_normal_completion()
    assert run.calls[-1] == "hold", "nothing ran after the interrupt"
    assert not {"held_turn", "response", "record"} & set(run.calls)
    assert run.backend._input_hold is True


def test_the_hold_of_one_run_is_not_seen_by_another_run_in_the_same_process(monkeypatch):
    first = declared_continuation(monkeypatch, admission=hold_the_answer)
    try:
        first.engine.continue_normal_completion()
    except ValueError:
        pass  # how the first run ends is pinned above; here only its hold matters
    assert first.backend._input_hold is True
    second = declared_continuation(monkeypatch)
    response = second.engine.continue_normal_completion()
    assert response.message.split("\n\n", 1)[0] == "done"
    assert second.backend._input_hold is False and "held_turn" not in second.calls
