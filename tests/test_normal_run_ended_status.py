"""What status and resume answer for a normal run that ended before fine-tuning started.

A normal run whose record is still in its first stage and holds a retained report has ended: it has no checkpoint,
no continuation and no wake, so nothing can continue it. Once that report is verified, the status keeps its earlier
keys and values and adds three: that the run has ended, the closed class of the kept report's action, and one
message that says where the report is and how to start again. A resume request ends with the same answer.
A status poll adds the three keys only after the run's worker has released its busy mark, so a report request
that follows such a poll finds the report. The exception is a resume request of the same ended run: while it
is being answered, its worker holds the busy mark, so a poll does not add the keys and a report request is
unavailable.

The controls pin what stays as it was: a run without a report, another session, a row in the fine-tuning stage,
a run that the original worker finished in the fine-tuning stage, and a resume request in a deployment that does
not approve fine-tuning.

Fixtures establish source behaviour over neutral native ports, not native storage or a deployed run.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading

import pytest
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from normal_sft_cold_fixture import close, cloud_data, finish_generation_pair, original_precommit, setup
from test_normal_prompt_report_durability import RUN, enrolled, mutation_calls, report_parts, rewrite_record
from test_normal_run_checkpoint import PROMPT, collect
from test_sdk_pending_handoff import PendingModel

from amlas import app as runtime
from amlas.async_callback import lambda_handler as callback_source
from amlas.control_plane import normal_run_checkpoint as checkpoint

pytest_plugins = ("test_normal_run_checkpoint",)

NEXT = (
    " The final report is kept: read it with the runtime action get_prompt_report and this run_id."
    " The run cannot be resumed. To try again, send a prompt; a new run starts from the beginning."
)
TEXT = {
    "input_admission_held": "This run has ended. Model input was held at the admission boundary.",
    "guardrail_intervened": "This run has ended. A guardrail intervened.",
    "response_received": "This run has ended with a final response.",
}
# The text of a turn that paused for a job. It stays the last response when that job ends without a result.
WAITING = "Waiting for original native operations to complete."
RESTORING = {"event": "resume_state", "state": "restoring", "run_id": RUN}
# Upper bound for one wait on a worker thread. A slow machine needs the room; a wait that never ends still fails.
WAIT_SECONDS = 120


def unavailable(run_id=RUN):
    return {"event": "prompt_run_status", "status": "unavailable", "run_id": run_id, "found": False}


def enrolled_status(run_id=RUN):
    return {
        "event": "prompt_run_status",
        "status": "enrolled",
        "run_id": run_id,
        "found": True,
        "stage": "enrolled",
        "resume_ready": False,
        "data_only": True,
        "durable_resume": False,
        "lifecycle_verified": False,
        "quality_success": False,
    }


def ended_status(action, run_id=RUN):
    return {
        **enrolled_status(run_id),
        "run_state": "ended",
        "ended_action": action,
        "message": TEXT[action] + NEXT,
    }


def status():
    return runtime._read_prompt_run({"action": "get_prompt_run_status", "run_id": RUN})


def resume_events(run_id=RUN):
    runtime._RESUME_WORKERS.clear()
    events = collect({"action": "resume_prompt_run", "run_id": run_id})
    return [event for event in events if event.get("event") != "heartbeat"]


def retain(session_hash, action="tool_agent_turn", message=None, data=None):
    envelope, response = report_parts()
    response["action"] = action
    if message is not None:
        response["message"] = message
    if data is not None:
        response["data"] = {**response["data"], **data}
    runtime._retain_prompt_report(RUN, envelope, response, session_hash)
    return response


def effects(work, ports, model):
    return (
        model.calls,
        len(work.engines),
        copy.deepcopy(ports.items),
        copy.deepcopy(ports.objects),
        mutation_calls(ports),
    )


def stored_report(ports):
    record = json.loads(ports.items["PROMPT_RUN#" + RUN]["record"]["S"])
    descriptor = record["report_checkpoint_ref"]
    return (descriptor["uri"].split("/", 3)[3], descriptor["version_id"])


def test_a_run_without_a_report_reads_enrolled_with_no_ended_key(normal_work):
    """Control: a run that is active, or whose worker left no report, keeps the earlier answers."""
    work, ports, model, _session_hash = enrolled(normal_work)
    before = effects(work, ports, model)
    assert status() == enrolled_status()
    assert resume_events() == [RESTORING, unavailable()]
    assert effects(work, ports, model) == before


@pytest.mark.parametrize(
    "action,expected,message,data",
    [
        ("input_admission_held", "input_admission_held", None, None),
        ("guardrail_intervened", "guardrail_intervened", None, None),
        ("tool_agent_turn", "response_received", None, None),
        ("error", "response_received", None, None),
        # The job this run waited for ended without a result, so the waiting text stayed its last response.
        (
            "tool_agent_turn",
            "response_received",
            WAITING,
            {"warm_model_continuation": {"disposition": "terminal_failed"}},
        ),
    ],
    ids=["input_admission_held", "guardrail_intervened", "tool_agent_turn", "error", "waiting_text"],
)
def test_a_run_that_ended_before_fine_tuning_reads_ended_and_names_why(
    normal_work, action, expected, message, data
):
    work, ports, model, session_hash = enrolled(normal_work)
    kept = retain(session_hash, action, message, data)
    before = effects(work, ports, model)
    for _ in range(2):
        answer = status()
        # The whole answer: the earlier keys with their earlier values and exactly three more.
        assert answer == ended_status(expected)
        # The message says that the run is over, where the kept report is and what to do next, in plain text.
        said = answer["message"]
        assert said.startswith("This run has ended") and "get_prompt_report" in said
        assert "cannot be resumed" in said and "a new run starts from the beginning" in said
        assert said.isascii() and len(said) < 400
    # The report that the message points to is the kept one, whatever its own text says.
    report = runtime._read_prompt_report({"action": "get_prompt_report", "run_id": RUN})
    assert report["status"] == "available" and report["report"]["response"] == kept
    assert effects(work, ports, model) == before


def test_a_run_that_ended_on_its_waiting_text_reads_ended_with_a_final_response(normal_work, monkeypatch):
    """A real prompt whose job failed: the report keeps the waiting text, and the status says the run is over."""
    work, _ports, model = normal_work
    first = PendingModel()

    async def stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        # The scripted model submits one acquisition. That job has failed when the run looks at it.
        model.calls += 1
        async for event in first.stream(messages, tool_specs, system_prompt, **kwargs):
            yield event

    monkeypatch.setattr(model, "stream", stream)
    work.native.state = "Failed"

    async def read(payload):
        return [e async for e in runtime.agent_invocation(payload) if e.get("event") != "heartbeat"]

    async def scenario():
        shown = await asyncio.wait_for(read({"prompt": work.literal}), WAIT_SECONDS)
        assert [event["event"] for event in shown] == ["run_started", "run_completed"]
        run_id, response = shown[0]["run_id"], shown[-1]["response"]
        assert response["action"] == "tool_agent_turn" and response["message"].endswith(WAITING)
        assert response["data"]["warm_model_continuation"] == {"disposition": "terminal_failed"}
        ended = ended_status("response_received", run_id)
        assert await read({"action": "get_prompt_run_status", "run_id": run_id}) == [ended]
        report = await read({"action": "get_prompt_report", "run_id": run_id})
        assert report[0]["status"] == "available" and report[0]["report"] == shown[-1]
        runtime._RESUME_WORKERS.clear()
        assert await read({"action": "resume_prompt_run", "run_id": run_id}) == [
            {"event": "resume_state", "state": "restoring", "run_id": run_id},
            ended,
        ]
        # One model call, one job and one engine: the three requests started nothing.
        assert model.calls == 1 and len(work.native.requests) == 1 and len(work.engines) == 1

    asyncio.run(scenario())


def test_resume_of_a_run_that_ended_before_fine_tuning_ends_with_the_ended_status(normal_work):
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash, "input_admission_held")
    before = effects(work, ports, model)
    for _ in range(2):
        events = resume_events()
        assert events == [RESTORING, ended_status("input_admission_held")]
        # A retained report pointer is still never read as a completed resume, and nothing is remembered.
        assert all(event.get("state") != "completed" for event in events)
        assert RUN not in runtime._RESUME_WORKERS
        # The callback reads only the first event: it parses as before and is not an acknowledgement.
        first = callback_source._resume_event_state(json.dumps(events[0], separators=(",", ":")), RUN)
        assert first == "restoring" and first not in callback_source._RESUME_ACKNOWLEDGED
        with pytest.raises(ValueError):
            callback_source._resume_event_state(json.dumps(events[1], separators=(",", ":")), RUN)
    assert effects(work, ports, model) == before


@pytest.mark.parametrize(
    "damage",
    [
        "descriptor",
        "bytes",
        "missing",
        "shape",
        "changed_during_read",
        "refused_after_read",
        "refused_after_second_record_read",
    ],
)
def test_an_ended_answer_needs_a_verified_report_and_an_unchanged_record(normal_work, monkeypatch, damage):
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    assert status() == ended_status("response_received")
    key = stored_report(ports)
    original = ports.objects[key]
    late = None
    if damage == "descriptor":
        rewrite_record(ports, lambda row: row["report_checkpoint_ref"].update(sha256="f" * 64))
    elif damage == "bytes":
        # The same size and still canonical JSON: only the digest tells that the bytes changed.
        changed = original.replace(b"Declared", b"Declarex", 1)
        assert changed != original and len(changed) == len(original)
        ports.objects[key] = changed
    elif damage == "missing":
        del ports.objects[key]
    elif damage == "shape":
        # Canonical bytes whose digest and size the descriptor names, but whose response action is not text.
        report = json.loads(original)
        report["response"]["action"] = 7
        raw = checkpoint._canonical(report).encode("utf-8")
        ports.objects[key] = raw
        rewrite_record(
            ports,
            lambda row: row["report_checkpoint_ref"].update(
                sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw)
            ),
        )
    elif damage == "changed_during_read":
        # The record changes between the report read and the second record read of the same status read.
        read_object = ports.get_object

        def get_object(**kwargs):
            answer = read_object(**kwargs)
            rewrite_record(ports, lambda row: row["history"].append({"note": "declared later change"}))
            return answer

        monkeypatch.setattr(ports, "get_object", get_object)
    elif damage == "refused_after_read":
        # The runtime role is no longer qualified once the report object has been read.
        read_object, identity = ports.get_object, ports.get_caller_identity
        state = {"read": False}

        def get_object(**kwargs):
            answer = read_object(**kwargs)
            state["read"] = True
            return answer

        def get_caller_identity(**kwargs):
            if state["read"]:
                raise PermissionError("declared later qualification refusal")
            return identity(**kwargs)

        monkeypatch.setattr(ports, "get_object", get_object)
        monkeypatch.setattr(ports, "get_caller_identity", get_caller_identity)
    else:
        # The runtime role stays qualified until the record read that follows the report read has returned.
        # From then on it is refused. Only the qualification that closes the ended read can still notice that.
        load, read_object, identity = checkpoint._load, ports.get_object, ports.get_caller_identity
        late = {"report_read": False, "refuse": False, "refused": 0}

        def get_object(**kwargs):
            answer = read_object(**kwargs)
            late["report_read"] = True
            return answer

        def load_then_refuse(*args, **kwargs):
            answer = load(*args, **kwargs)
            if late["report_read"]:
                late["refuse"] = True
            return answer

        def get_caller_identity(**kwargs):
            if late["refuse"]:
                late["refused"] += 1
                raise PermissionError("declared later qualification refusal")
            return identity(**kwargs)

        monkeypatch.setattr(checkpoint, "_load", load_then_refuse)
        monkeypatch.setattr(ports, "get_object", get_object)
        monkeypatch.setattr(ports, "get_caller_identity", get_caller_identity)
    before = effects(work, ports, model) if damage in ("descriptor", "bytes", "missing", "shape") else None
    # A report that cannot be verified is never called ended: the closed answer says only that nothing is verified.
    assert status() == unavailable()
    if late is not None:
        # The refusal met exactly one call: the qualification that closes the ended read.
        assert late == {"report_read": True, "refuse": True, "refused": 1}
    if before is not None:
        assert resume_events() == [RESTORING, unavailable()]
        assert effects(work, ports, model) == before
    else:
        # The damage of these cases also meets the resume path's own reads: whatever it answers, never ended.
        assert all("run_state" not in event for event in resume_events())
    assert model.calls == 0 and not work.engines


def test_status_says_ended_only_after_the_worker_released_its_busy_mark(normal_work, monkeypatch):
    """The worker stores its report before it releases its busy mark. Until then a poll keeps the plain answer."""
    work, ports, model = normal_work
    published, release = threading.Event(), threading.Event()
    retain_report = runtime._retain_prompt_report

    def retain_then_wait(*args, **kwargs):
        retain_report(*args, **kwargs)
        published.set()
        assert release.wait(WAIT_SECONDS)

    monkeypatch.setattr(runtime, "_retain_prompt_report", retain_then_wait)

    async def read(payload):
        return [e async for e in runtime.agent_invocation(payload) if e.get("event") != "heartbeat"]

    async def scenario():
        shown = []

        async def run():
            async for event in runtime.agent_invocation({"prompt": PROMPT}):
                if event.get("event") != "heartbeat":
                    shown.append(event)

        reader = asyncio.ensure_future(run())
        try:
            deadline = asyncio.get_running_loop().time() + WAIT_SECONDS
            while not published.is_set():
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            run_id = shown[0]["run_id"]
            record = json.loads(ports.items["PROMPT_RUN#" + run_id]["record"]["S"])
            assert record["stage"] == "enrolled" and type(record.get("report_checkpoint_ref")) is dict
            # The report is stored and the worker still holds its busy mark: nothing says ended yet.
            assert await read({"action": "get_prompt_run_status", "run_id": run_id}) == [
                enrolled_status(run_id)
            ]
            assert await read({"action": "get_prompt_report", "run_id": run_id}) == [
                {"event": "prompt_report", "status": "unavailable", "run_id": run_id, "found": False}
            ]
            assert await read({"action": "resume_prompt_run", "run_id": run_id}) == [
                {"event": "resume_state", "state": "original_active", "run_id": run_id}
            ]
            release.set()
            await asyncio.wait_for(reader, WAIT_SECONDS)
            assert [event["event"] for event in shown] == ["run_started", "run_completed"]
            # The worker is done: the status says ended, and the report it points to is readable.
            ended = ended_status("response_received", run_id)
            assert await read({"action": "get_prompt_run_status", "run_id": run_id}) == [ended]
            report = await read({"action": "get_prompt_report", "run_id": run_id})
            assert report[0]["status"] == "available" and report[0]["report"] == shown[-1]
            runtime._RESUME_WORKERS.clear()
            assert await read({"action": "resume_prompt_run", "run_id": run_id}) == [
                {"event": "resume_state", "state": "restoring", "run_id": run_id},
                ended,
            ]
            assert model.calls == 1 and len(work.engines) == 1
        finally:
            release.set()
            if not reader.done():
                reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    asyncio.run(scenario())


def test_a_status_poll_during_a_resume_of_the_same_ended_run_keeps_the_plain_answer(normal_work, monkeypatch):
    """The worker of a resume request holds the busy mark while it reads, so a poll in that moment stays plain."""
    _work, _ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    entered, release = threading.Event(), threading.Event()
    read_status = checkpoint.read_normal_run_status

    def read_then_wait(run_id, session, policy=None, resume=False):
        answer = read_status(run_id, session, policy=policy, resume=resume)
        if resume:
            entered.set()
            assert release.wait(WAIT_SECONDS)
        return answer

    monkeypatch.setattr(checkpoint, "read_normal_run_status", read_then_wait)

    async def read(payload):
        return [e async for e in runtime.agent_invocation(payload) if e.get("event") != "heartbeat"]

    async def scenario():
        runtime._RESUME_WORKERS.clear()
        ended = ended_status("response_received")
        reader = asyncio.ensure_future(read({"action": "resume_prompt_run", "run_id": RUN}))
        try:
            deadline = asyncio.get_running_loop().time() + WAIT_SECONDS
            while not entered.is_set():
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            # The resume worker has read the ended status and still holds the busy mark.
            assert await read({"action": "get_prompt_run_status", "run_id": RUN}) == [enrolled_status()]
            assert await read({"action": "get_prompt_report", "run_id": RUN}) == [
                {"event": "prompt_report", "status": "unavailable", "run_id": RUN, "found": False}
            ]
            release.set()
            assert await asyncio.wait_for(reader, WAIT_SECONDS) == [RESTORING, ended]
            assert await read({"action": "get_prompt_run_status", "run_id": RUN}) == [ended]
            assert model.calls == 0
        finally:
            release.set()
            if not reader.done():
                reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    asyncio.run(scenario())


def test_the_ended_projection_is_only_for_a_first_stage_row_that_holds_a_report(normal_work):
    """The helper decides from the record alone, before any read: no port is passed here."""
    _work, ports, _model, session_hash = enrolled(normal_work)
    retain(session_hash)
    record = json.loads(ports.items["PROMPT_RUN#" + RUN]["record"]["S"])
    assert record["stage"] == "enrolled" and "report_checkpoint_ref" in record
    in_fine_tuning = {**record, "stage": "sft"}
    without_report = {key: record[key] for key in record if key != "report_checkpoint_ref"}
    for row in (in_fine_tuning, without_report):
        assert checkpoint._ended_before_sft(None, None, None, row, None, None) is None


def test_a_row_in_the_fine_tuning_stage_with_a_report_carries_no_ended_key(normal_work):
    """Control: through the status read such a row takes the fine-tuning path, which refuses this association."""
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    rewrite_record(ports, lambda row: row.update(stage="sft", sft={}))
    assert status() == unavailable()
    assert model.calls == 0 and not work.engines


def test_another_session_reads_unavailable_for_a_run_with_a_report(normal_work):
    """Control: no other session learns that the run exists or that it has ended."""
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    own = status()
    assert own["found"] is True and own["status"] == "enrolled"
    BedrockAgentCoreContext.set_request_context("ordinary-other-request", "other-session-" + "9" * 24)
    before = effects(work, ports, model)
    assert status() == unavailable()
    # A resume request of the other session gets what it gets for a run that does not exist.
    absent = "7" * 32
    seen = [{**event, "run_id": absent} for event in resume_events()]
    assert seen == resume_events(absent) and all("run_state" not in event for event in seen)
    assert effects(work, ports, model) == before


def test_resume_without_the_fine_tuning_approval_answers_as_before(normal_work, monkeypatch):
    """Control: a resume request checks the deployment's approval first, for an ended run as for any other."""
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job"]')
    before = effects(work, ports, model)
    events = resume_events()
    assert events[0] == RESTORING and len(events) == 2
    assert events[1]["event"] == "resume_state" and events[1]["state"] == "failed"
    assert events[1]["diagnostic"]["restore_stage"] == "run_prepare"
    assert effects(work, ports, model) == before


@pytest.mark.usefixtures("inert_runtime_setup")
def test_a_run_the_original_worker_finished_in_the_fine_tuning_stage_carries_no_ended_key(
    monkeypatch, tmp_path
):
    """Control: a row in the fine-tuning stage keeps its answers while its job runs and after the run finished."""

    async def read(payload):
        return [e async for e in runtime.agent_invocation(payload) if e.get("event") != "heartbeat"]

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            pending = await read({"action": "get_prompt_run_status", "run_id": state.run_id})
            assert len(pending) == 1 and set(pending[0]) == {"event", "status", "run_id", "found", "progress"}
            assert pending[0]["status"] == "available" and pending[0]["progress"]["pending"] is True
            cloud_data(state, completed=True)
            state.native.states[state.original.id] = "Completed"
            terminal = await finish_generation_pair(state, state.reader)
            assert terminal[-1]["event"] == "run_completed"
            record = json.loads(state.ports.items["PROMPT_RUN#" + state.run_id]["record"]["S"])
            assert record["stage"] == "sft" and type(record.get("report_checkpoint_ref")) is dict
            # The report is kept and readable, and the status still says nothing about an end.
            report = await read({"action": "get_prompt_report", "run_id": state.run_id})
            assert report[0]["status"] == "available" and report[0]["report"] == terminal[-1]
            assert await read({"action": "get_prompt_run_status", "run_id": state.run_id}) == [
                unavailable(state.run_id)
            ]
        finally:
            await close(state)

    asyncio.run(scenario())
