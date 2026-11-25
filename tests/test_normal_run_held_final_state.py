"""What a normal run leaves behind when a guardrail stops the model call that resumes it.

A normal prompt runs through ``agent_invocation`` with the durable run record configured over strict neutral
native ports. The run waits for its original model acquisition. The model call that resumes it goes through the
SDK's real ``BedrockModel`` and is stopped by a guardrail (a stub client replays one fixed response stream).
The tests pin what a client then gets: the final report, a run status that says the run has ended and why, a
resume request that ends with that same status, and that none of these requests starts work again.

The expectations for a stop without a trace follow the installed SDK, as in
``test_bedrock_guardrail_stop_without_trace.py``. Fixtures establish source behaviour, not native guardrail
classification, native storage or a deployed run.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from types import SimpleNamespace

import pytest
import test_bedrock_guardrail_stop_without_trace as g
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from test_normal_run_checkpoint import SESSION, decoded
from test_runtime_warm_model_continuation import complete_fixture
from test_sdk_pending_handoff import PendingModel

from amlas import app as runtime

pytest_plugins = ("test_normal_run_checkpoint",)
WRITES = {"PutItem", "UpdateItem", "PutObject"}
# The status of a run that ended before fine-tuning: the keys of the enrolled answer and three more.
STATUS_KEYS = {
    "event",
    "status",
    "run_id",
    "found",
    "stage",
    "resume_ready",
    "data_only",
    "durable_resume",
    "lifecycle_verified",
    "quality_success",
    "run_state",
    "ended_action",
    "message",
}


async def read(payload):
    return [event async for event in runtime.agent_invocation(payload) if event.get("event") != "heartbeat"]


@pytest.mark.parametrize("case", ["no_trace", "blocked_trace"])
def test_held_run_keeps_its_final_report_and_no_request_starts_it_again(
    normal_work, monkeypatch, tmp_path, caplog, case
):
    work, ports, model = normal_work
    blocked = case == "blocked_trace"
    bedrock, client = g.bound_model(
        monkeypatch,
        trace=g.BLOCKED_TRACE if blocked else None,
        trace_mode="enabled" if blocked else "disabled",
    )
    # A blocked trace redacts on every SDK version; without a trace only the newer SDK does.
    held = blocked or g.REDACTS_WITHOUT_TRACE
    first = PendingModel()

    async def stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        # The scripted model submits the acquisition; the resumed call is the replayed guardrail stop.
        model.calls += 1
        source = first if first.calls == 0 else bedrock
        async for event in source.stream(messages, tool_specs, system_prompt, **kwargs):
            yield event

    monkeypatch.setattr(model, "stream", stream)

    def effects():
        return (
            model.calls,
            len(client.requests),
            len(work.native.requests),
            len(work.engines),
            len(work.agents),
            copy.deepcopy(ports.items),
            dict(ports.objects),
            sum(1 for name, _arguments in ports.calls if name in WRITES),
        )

    async def scenario():
        reader = asyncio.ensure_future(read({"prompt": work.literal}))
        try:
            deadline = asyncio.get_running_loop().time() + g.LIVENESS_WAIT_SECONDS
            while not work.observing.is_set() and not reader.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert not reader.done() and model.calls == 1 and len(work.native.requests) == 1
            waiting = work.engines[0].tool_backend._warm_pending
            assert waiting is not None
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=waiting.record.handle, adapter=work.adapter, native=work.native),
            )
            with caplog.at_level(logging.WARNING):
                shown = await asyncio.wait_for(reader, g.LIVENESS_WAIT_SECONDS)

            # What the user sees: the run ends with one final report.
            assert [event["event"] for event in shown] == ["run_started", "run_completed"]
            run_id, terminal = shown[0]["run_id"], shown[-1]
            assert terminal["status"] == "response_received" and terminal["worker_finished"] is True
            response = terminal["response"]
            if held:
                assert response["action"] == "input_admission_held"
                assert response["message"] == g.HOLD_TEXT
                assert response["data"]["input_admission"]["status"] == "held"
                assert response["data"]["warm_model_continuation"] == g.HELD_CONTINUATION
                assert g.hold_lines(caplog) == [g.HOLD_LINE]
            else:
                # Strands 1.42.0 without a trace: reported as a plain intervention, nothing is held.
                assert response["action"] == "guardrail_intervened"
                assert response["message"].endswith(g.TEXT)
                assert response["data"]["warm_model_continuation"] == {"disposition": "guardrail_intervened"}
                assert g.hold_lines(caplog) == []
            # The resumed call was sent once; the acquisition was submitted once.
            assert model.calls == 2 and len(client.requests) == 1 and len(work.native.requests) == 1

            # The run record stays at its first stage and now names the retained report, the only stored object.
            record = decoded(ports.items["PROMPT_RUN#" + run_id])["record"]
            assert record["stage"] == "enrolled" and type(record.get("report_checkpoint_ref")) is dict
            assert len(ports.objects) == 1
            before = effects()
            for request in ("ordinary-second-request", "ordinary-third-request"):
                # A later request of the same session. The second round also drops this process's resume memory.
                BedrockAgentCoreContext.set_request_context(request, SESSION)
                if request == "ordinary-third-request":
                    runtime._RESUME_WORKERS.clear()
                # The report can be read again and equals what the run returned.
                assert await read({"action": "get_prompt_report", "run_id": run_id}) == [
                    {
                        "event": "prompt_report",
                        "status": "available",
                        "run_id": run_id,
                        "found": True,
                        "report": terminal,
                    }
                ]
                # The status shows a run that cannot be resumed, and says that it has ended and why.
                status = await read({"action": "get_prompt_run_status", "run_id": run_id})
                assert len(status) == 1 and status[0]["event"] == "prompt_run_status"
                assert status[0]["found"] is True and status[0]["status"] == "enrolled"
                assert status[0]["resume_ready"] is False
                ended = "input_admission_held" if held else "guardrail_intervened"
                assert status[0]["run_state"] == "ended" and status[0]["ended_action"] == ended
                assert status[0]["message"].startswith("This run has ended")
                assert "get_prompt_report" in status[0]["message"]
                assert set(status[0]) == STATUS_KEYS
                # A resume request finds nothing to continue and ends with that same answer.
                assert await read({"action": "resume_prompt_run", "run_id": run_id}) == [
                    {"event": "resume_state", "state": "restoring", "run_id": run_id},
                    status[0],
                ]
                # None of the three requests called the model, submitted a job, built an agent or wrote anything.
                assert effects() == before
        finally:
            if not reader.done():
                reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    asyncio.run(scenario())
