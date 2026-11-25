"""What the kept report of a held prompt run says about the guardrail.

When a guardrail stops a model call of a prompt run, the agent SDK replaces the last input with its redaction
text. The product refuses the changed history and holds the run.
``test_bedrock_guardrail_stop_without_trace.py`` pins that for the model call that resumes a run. These tests
pin it for the first model call of a run and for a call after a tool cycle, and they pin a hold from another
cause in a turn whose earlier call carried a trace:

- the held report carries ``guardrail_assessment``, what the guardrail reported in that turn as typed records;
- the action stays ``input_admission_held``, the message stays the fixed text;
- no ``guardrail_outcome`` is added;
- without a trace the held report names no guardrail;
- a stop whose trace names only a masked entity is not held: it is reported as a guardrail intervention.
  Such a run has ended as well: a later request reads its kept report, and no request continues it.

The assessment is turn data. It does not say why the run was held. After a tool cycle the earlier call carried
no trace, so the assessment is ``partial`` and its summary ``action`` reads ``UNKNOWN``: a reader uses
``records``.

Two fixtures have the shape that one stopped call and one unstopped call of the model service returned,
with every model text and matched text replaced and every number set to 1. In that shape the personal-data
policy has no ``regexes`` key, and every trace holds an input assessment with only its details and metrics.
The projection reads such a policy and keeps its entity records. The status of such a trace is ``partial``
or ``unavailable``, never ``complete``.

A stub model-service client replays one fixed response stream for the stopped call, or a scripted model adds
one fixed trace to a call: no provider, guard or network call. Every text is ordinary neutral text. The
expectations for a stop without a trace follow the installed SDK, as in
``test_bedrock_guardrail_stop_without_trace.py``. Fixtures establish source behaviour, not native guardrail
classification, native storage or a deployed run.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import sys
from types import SimpleNamespace

import pytest
import test_bedrock_guardrail_stop_without_trace as g
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from test_normal_research_profile import PROMPT, LifecycleModel, setup
from test_normal_run_checkpoint import SESSION, decoded
from test_runtime_warm_model_continuation import complete_fixture
from test_sdk_pending_handoff import PendingModel

from amlas import api
from amlas import app as runtime
from amlas.config.settings import Settings
from amlas.tools.coding_preparation import list_coding_preparation_profiles
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.guardrail_assessment import project_guardrail_trace

pytest_plugins = ("test_normal_run_checkpoint",)

CONTENT = {"type": "VIOLENCE", "confidence": "HIGH", "detected": True, "action": "BLOCKED"}
BLOCKED = {
    "input": g.BLOCKED_TRACE,
    "output": {
        "guardrail": {"outputAssessments": {"fixtureguardrail": [{"contentPolicy": {"filters": [CONTENT]}}]}}
    },
}
MASKED = {
    "guardrail": {
        "outputAssessments": {
            "fixtureguardrail": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [
                            {"type": "EMAIL", "match": "neutral", "action": "ANONYMIZED", "detected": True}
                        ],
                        "regexes": [],
                    }
                }
            ]
        }
    }
}
DETAILS = {
    "guardrailId": "fixtureguardrail",
    "guardrailVersion": "1",
    "guardrailArn": "neutral",
    "guardrailOrigin": ["neutral"],
    "guardrailOwnership": "neutral",
}
METRICS = {
    "guardrailProcessingLatency": 1,
    "usage": {
        "topicPolicyUnits": 1,
        "contentPolicyUnits": 1,
        "wordPolicyUnits": 1,
        "sensitiveInformationPolicyUnits": 1,
        "sensitiveInformationPolicyFreeUnits": 1,
        "contextualGroundingPolicyUnits": 1,
    },
    "guardrailCoverage": {"textCharacters": {"guarded": 1, "total": 1}},
}
EMAIL_BLOCKED = {"type": "EMAIL", "match": "neutral", "action": "BLOCKED", "detected": True}
REGEX_MATCH = {
    "name": "neutral",
    "match": "neutral",
    "regex": "neutral",
    "action": "BLOCKED",
    "detected": True,
}
ONLY_DETAILS_AND_METRICS = {"appliedGuardrailDetails": DETAILS, "invocationMetrics": METRICS}
# The trace of one stopped call: an e-mail address in the model's output was blocked.
RECORDED_STOP = {
    "guardrail": {
        "actionReason": "No action.\nGuardrail blocked.",
        "inputAssessment": {"fixtureguardrail": ONLY_DETAILS_AND_METRICS},
        "modelOutput": ["neutral"],
        "outputAssessments": {
            "fixtureguardrail": [
                {**ONLY_DETAILS_AND_METRICS, "sensitiveInformationPolicy": {"piiEntities": [EMAIL_BLOCKED]}}
            ]
        },
    }
}
# The trace of one call that was not stopped.
RECORDED_UNSTOPPED = {
    "guardrail": {
        "actionReason": "No action.",
        "inputAssessment": {"fixtureguardrail": ONLY_DETAILS_AND_METRICS},
        "outputAssessments": {"fixtureguardrail": [ONLY_DETAILS_AND_METRICS]},
    }
}
EMAIL_RECORD = {
    "direction": "OUTPUT",
    "policy": "PII",
    "category": "EMAIL",
    "action": "BLOCKED",
    "confidence": None,
    "detected": True,
    "filter_strength": None,
}
FIRST_CALL, AFTER_A_TOOL_CYCLE = 1, 2
HOLD_LINES = {
    "binding_after_the_run": "AMLAS_INPUT_HOLD phase=research_after_run reason=binding_refused",
    "pause_result": "AMLAS_INPUT_HOLD phase=pause_result reason=pause_result_refused",
}


class StopAt(LifecycleModel):
    """The scripted lifecycle model; model call number ``stop_at`` is the replayed guardrail stop."""

    def __init__(self, bedrock, stop_at):
        super().__init__()
        self.bedrock, self.stop_at = bedrock, stop_at

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls + 1 != self.stop_at:
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
            return
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        async for event in self.bedrock.stream(messages, tool_specs, system_prompt, **kwargs):
            yield event


class TraceOnFirstCall(LifecycleModel):
    """The scripted lifecycle model; its first call, a tool use, also carries a ``trace`` in its metadata."""

    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        first = self.calls == 0
        async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
            yield event
        if first and self.trace is not None:
            yield {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    "metrics": {"latencyMs": 1},
                    "trace": copy.deepcopy(self.trace),
                }
            }


def stopped_run(monkeypatch, caplog, trace, stop_at):
    """One prompt through ``api.run_prompt``; model call ``stop_at`` is stopped inside ``BedrockModel``."""
    bedrock, client = g.bound_model(
        monkeypatch, trace=trace, trace_mode="disabled" if trace is None else "enabled"
    )
    model = StopAt(bedrock, stop_at)
    selected = setup(monkeypatch, model_override=model)
    with caplog.at_level(logging.WARNING):
        response = asyncio.run(api.run_prompt(PROMPT, settings=Settings(), policy=selected.policy))
    # The stopped call was the last model call of the run and was sent once.
    assert model.calls == stop_at and len(client.requests) == 1
    return response


def refuse_the_pause_result(monkeypatch):
    def refused(self, agent, result):
        raise RuntimeError("neutral refusal")

    monkeypatch.setattr(binding.ToolBindingBackend, "_pending_pause_result", refused)


def refuse_the_binding_after_the_run(monkeypatch):
    original = binding.ToolBindingBackend._qualify_research_binding
    turn_function = binding.ToolBindingBackend._chat_with_tools_once.__code__

    def refused(self, *args, **kwargs):
        # The turn function itself asks once, after its agent call returned. Every other caller is answered.
        if sys._getframe(1).f_code is turn_function:
            raise RuntimeError("neutral refusal")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(binding.ToolBindingBackend, "_qualify_research_binding", refused)


OTHER_CAUSES = {
    "binding_after_the_run": refuse_the_binding_after_the_run,
    "pause_result": refuse_the_pause_result,
}


def run_held_by_another_cause(monkeypatch, caplog, cause, trace):
    """A run whose agent call returns normally; the check named ``cause`` then refuses and the run is held."""
    model = TraceOnFirstCall(trace)
    selected = setup(monkeypatch, model_override=model, extra_tools=(list_coding_preparation_profiles,))
    OTHER_CAUSES[cause](monkeypatch)
    with caplog.at_level(logging.WARNING):
        response = asyncio.run(api.run_prompt(PROMPT, settings=Settings(), policy=selected.policy))
    # Every scripted model call was made: the hold came after the agent call, from the named check alone.
    assert model.calls == 4
    assert g.hold_lines(caplog) == [HOLD_LINES[cause]]
    return response


def assert_held_with_the_fixed_text_and_no_outcome(response):
    assert response.action == "input_admission_held"
    assert response.message == g.HOLD_TEXT
    assert response.data["input_admission"]["status"] == "held"
    assert "guardrail_outcome" not in response.data


def records(data):
    return [
        (record["direction"], record["policy"], record["category"], record["action"])
        for record in data["guardrail_assessment"]["records"]
    ]


@pytest.mark.parametrize("side", ["input", "output"])
def test_a_blocked_stop_on_the_first_model_call_keeps_the_assessment_in_the_held_report(
    monkeypatch, caplog, side
):
    response = stopped_run(monkeypatch, caplog, BLOCKED[side], FIRST_CALL)
    assert_held_with_the_fixed_text_and_no_outcome(response)
    assert g.hold_lines(caplog) == [g.HOLD_LINE]
    # The trace of the stopped call reaches the report as a typed assessment: it names the blocked policy.
    assert records(response.data) == [(side.upper(), "CONTENT", "VIOLENCE", "BLOCKED")]
    assessment = response.data["guardrail_assessment"]
    assert assessment["assessment_status"] == "complete" and assessment["action"] == "BLOCKED"
    # The report carries no provider text and no redaction text.
    shown = response.message + str(response.data)
    assert g.TEXT not in shown and g.REDACTED not in shown


@pytest.mark.parametrize("side", ["input", "output"])
def test_a_blocked_stop_on_the_call_after_a_tool_cycle_keeps_the_assessment_in_the_held_report(
    monkeypatch, caplog, side
):
    response = stopped_run(monkeypatch, caplog, BLOCKED[side], AFTER_A_TOOL_CYCLE)
    assert_held_with_the_fixed_text_and_no_outcome(response)
    assert g.hold_lines(caplog) == [g.HOLD_LINE]
    # The record names the blocked policy. The earlier call of the turn carried no trace, so the assessment
    # covers only part of the turn and its summary action is not a known one.
    assert records(response.data) == [(side.upper(), "CONTENT", "VIOLENCE", "BLOCKED")]
    assessment = response.data["guardrail_assessment"]
    assert assessment["assessment_status"] == "partial" and assessment["action"] == "UNKNOWN"


@pytest.mark.parametrize("cause", list(OTHER_CAUSES))
def test_a_hold_from_another_cause_keeps_what_the_guardrail_reported_in_that_turn(monkeypatch, caplog, cause):
    response = run_held_by_another_cause(monkeypatch, caplog, cause, MASKED)
    # An earlier call of the turn only masked text. The held report keeps that assessment as turn data: the
    # action stays the held one and no intervention outcome is added.
    assert_held_with_the_fixed_text_and_no_outcome(response)
    assert records(response.data) == [("OUTPUT", "PII", "EMAIL", "ANONYMIZED")]
    assert response.data["guardrail_assessment"]["assessment_status"] == "partial"


@pytest.mark.parametrize("stop_at", [FIRST_CALL, AFTER_A_TOOL_CYCLE])
def test_a_stop_without_a_trace_names_no_guardrail_in_the_report(monkeypatch, caplog, stop_at):
    response = stopped_run(monkeypatch, caplog, None, stop_at)
    # No assessment is invented without a trace.
    assert response.data.get("guardrail_assessment") is None
    if g.REDACTS_WITHOUT_TRACE:
        # The locked SDK redacts on the stop reason alone, so the run is held and nothing names a guardrail.
        assert_held_with_the_fixed_text_and_no_outcome(response)
        assert g.hold_lines(caplog) == [g.HOLD_LINE]
    else:
        # An older SDK leaves the input alone: the stop is reported as a plain intervention.
        assert response.action == "guardrail_intervened" and response.message.endswith(g.TEXT)
        assert response.data["guardrail_outcome"] == g.INTERVENTION
        assert g.hold_lines(caplog) == []


@pytest.mark.parametrize("stop_at", [FIRST_CALL, AFTER_A_TOOL_CYCLE])
def test_a_stop_that_only_masked_text_is_not_held(monkeypatch, caplog, stop_at):
    response = stopped_run(monkeypatch, caplog, MASKED, stop_at)
    # The trace names no blocked policy, so no SDK version replaces the input and nothing is held.
    assert g.hold_lines(caplog) == []
    # What the user sees: a plain intervention with the provider's text, and the masked entity as a record.
    assert response.action == "guardrail_intervened" and response.message.endswith(g.TEXT)
    assert response.data["guardrail_outcome"] == g.INTERVENTION
    assert records(response.data) == [("OUTPUT", "PII", "EMAIL", "ANONYMIZED")]


@pytest.mark.parametrize("cause", list(OTHER_CAUSES))
def test_a_hold_from_another_cause_in_a_turn_without_a_trace_carries_no_assessment(
    monkeypatch, caplog, cause
):
    response = run_held_by_another_cause(monkeypatch, caplog, cause, None)
    assert_held_with_the_fixed_text_and_no_outcome(response)
    assert response.data.get("guardrail_assessment") is None


def output_policy(policy):
    assessment = {"sensitiveInformationPolicy": policy}
    return {"guardrail": {"outputAssessments": {"fixtureguardrail": [assessment]}}}


def test_a_personal_data_policy_without_a_regular_expression_list_keeps_its_entity_records():
    # The policy alone, without the details and metrics around it, so that only the missing list decides.
    partial = project_guardrail_trace(output_policy({"piiEntities": [EMAIL_BLOCKED]})).to_dict()
    # The entity is read. The missing list leaves the coverage open: the summary action is not a known one.
    assert partial["records"] == [EMAIL_RECORD]
    assert partial["assessment_status"] == "partial" and partial["action"] == "UNKNOWN"
    assert "neutral" not in str(partial), "the matched text must not be kept"
    # With an empty list the same policy is complete; without its entity list it stays invalid.
    with_list = output_policy({"piiEntities": [EMAIL_BLOCKED], "regexes": []})
    complete = project_guardrail_trace(with_list).to_dict()
    assert complete["records"] == [EMAIL_RECORD]
    assert complete["assessment_status"] == "complete" and complete["action"] == "BLOCKED"
    invalid = project_guardrail_trace(output_policy({"regexes": []})).to_dict()
    assert invalid["assessment_status"] == "invalid" and invalid["action"] == "UNKNOWN"
    assert invalid["records"] == []
    # A list that is present and not empty, and a value that is not a list, leave the coverage open as well.
    # The entity record is kept, and nothing of the other value is.
    for other in ([REGEX_MATCH], {}):
        policy = {"piiEntities": [EMAIL_BLOCKED], "regexes": other}
        open_coverage = project_guardrail_trace(output_policy(policy)).to_dict()
        assert open_coverage["records"] == [EMAIL_RECORD]
        assert open_coverage["assessment_status"] == "partial" and open_coverage["action"] == "UNKNOWN"
        assert "neutral" not in str(open_coverage), "nothing of the other value must be kept"


@pytest.mark.parametrize("stop_at", [FIRST_CALL, AFTER_A_TOOL_CYCLE])
def test_the_recorded_shape_of_an_output_side_stop_names_the_blocked_entity_in_the_held_report(
    monkeypatch, caplog, stop_at
):
    response = stopped_run(monkeypatch, caplog, RECORDED_STOP, stop_at)
    # The SDK replaced the input for a block on the output side too, so the run is held in the same way.
    assert_held_with_the_fixed_text_and_no_outcome(response)
    assert g.hold_lines(caplog) == [g.HOLD_LINE]
    # The record names the blocked entity and nothing else of the trace is kept.
    assessment = response.data["guardrail_assessment"]
    assert assessment["records"] == [EMAIL_RECORD]
    # The trace also holds an input assessment with only details and metrics: the status is never complete.
    assert assessment["assessment_status"] == "partial" and assessment["action"] == "UNKNOWN"


@pytest.mark.parametrize("cause", list(OTHER_CAUSES))
def test_the_recorded_shape_of_an_unstopped_call_gives_an_assessment_without_records(
    monkeypatch, caplog, cause
):
    response = run_held_by_another_cause(monkeypatch, caplog, cause, RECORDED_UNSTOPPED)
    # With the trace on, a call that was not stopped carries a trace too. The held report then has the key,
    # with no record: nothing was detected.
    assert_held_with_the_fixed_text_and_no_outcome(response)
    assert response.data["guardrail_assessment"] == {
        "schema_version": 1,
        "assessment_status": "unavailable",
        "action": "UNKNOWN",
        "records": [],
    }


async def read(payload):
    return [event async for event in runtime.agent_invocation(payload) if event.get("event") != "heartbeat"]


def test_a_later_report_request_returns_the_assessment_of_a_run_held_on_its_first_model_call(
    normal_work, monkeypatch, caplog
):
    """Through the runtime entry with the durable run record: the stored report keeps the assessment."""
    work, ports, model = normal_work
    bedrock, client = g.bound_model(monkeypatch, trace=g.BLOCKED_TRACE, trace_mode="enabled")

    async def stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        model.calls += 1
        async for event in bedrock.stream(messages, tool_specs, system_prompt, **kwargs):
            yield event

    monkeypatch.setattr(model, "stream", stream)

    async def scenario():
        with caplog.at_level(logging.WARNING):
            shown = await asyncio.wait_for(read({"prompt": work.literal}), g.LIVENESS_WAIT_SECONDS)
        # What the user sees: the run ends with one final report, held on its first and only model call.
        assert [event["event"] for event in shown] == ["run_started", "run_completed"]
        run_id, terminal = shown[0]["run_id"], shown[-1]
        assert terminal["status"] == "response_received"
        response = terminal["response"]
        assert response["action"] == "input_admission_held" and response["message"] == g.HOLD_TEXT
        assert "guardrail_outcome" not in response["data"]
        assert records(response["data"]) == [("INPUT", "CONTENT", "VIOLENCE", "BLOCKED")]
        assert g.hold_lines(caplog) == [g.HOLD_LINE]
        assert model.calls == 1 and len(client.requests) == 1
        # The run record stays at its first stage and names the retained report.
        record = decoded(ports.items["PROMPT_RUN#" + run_id])["record"]
        assert record["stage"] == "enrolled" and type(record.get("report_checkpoint_ref")) is dict
        # A later request of the same session reads the stored report: it equals what the run returned.
        BedrockAgentCoreContext.set_request_context("ordinary-second-request", SESSION)
        assert await read({"action": "get_prompt_report", "run_id": run_id}) == [
            {
                "event": "prompt_report",
                "status": "available",
                "run_id": run_id,
                "found": True,
                "report": terminal,
            }
        ]
        assert model.calls == 1 and len(client.requests) == 1

    asyncio.run(scenario())


WRITES = {"PutItem", "UpdateItem", "PutObject"}


class OneToolCycle:
    """A scripted model call that asks for one read-only tool; the next model call follows a tool cycle."""

    calls = 0

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        use = {"toolUseId": "profile-discovery", "name": "list_coding_preparation_profiles"}
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": use}}}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{}"}}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "tool_use"}}


STOPS = ["first_model_call", "call_after_a_tool_cycle", "call_that_resumes_the_run"]


@pytest.mark.parametrize("stop_on", STOPS)
def test_a_run_whose_stop_only_masked_text_has_ended_and_keeps_its_report(
    normal_work, monkeypatch, tmp_path, caplog, stop_on
):
    """Through the runtime entry with the durable run record: a stop that is not held ends the run too."""
    work, ports, model = normal_work
    bedrock, client = g.bound_model(monkeypatch, trace=g.MASKED_TRACE, trace_mode="enabled")
    # Before the stop the scripted model asks for one read-only tool, or submits the model acquisition that
    # the run then waits for; the next model call is the stop.
    resumed = stop_on == "call_that_resumes_the_run"
    first = PendingModel() if resumed else OneToolCycle() if stop_on == "call_after_a_tool_cycle" else None

    async def stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        model.calls += 1
        source = bedrock if first is None or first.calls else first
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

    async def run_to_its_end():
        reader = asyncio.ensure_future(read({"prompt": work.literal}))
        try:
            if resumed:
                deadline = asyncio.get_running_loop().time() + g.LIVENESS_WAIT_SECONDS
                while not work.observing.is_set() and not reader.done():
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.005)
                assert not reader.done() and model.calls == 1
                waiting = work.engines[0].tool_backend._warm_pending
                assert waiting is not None
                complete_fixture(
                    monkeypatch,
                    tmp_path,
                    SimpleNamespace(handle=waiting.record.handle, adapter=work.adapter, native=work.native),
                )
            return await asyncio.wait_for(reader, g.LIVENESS_WAIT_SECONDS)
        finally:
            if not reader.done():
                reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def scenario():
        shown = await run_to_its_end()
        # What the user sees: the run ends with one final report, a plain intervention with the provider's
        # text. Nothing is held.
        assert [event["event"] for event in shown] == ["run_started", "run_completed"]
        run_id, terminal = shown[0]["run_id"], shown[-1]
        assert terminal["status"] == "response_received"
        response = terminal["response"]
        assert response["action"] == "guardrail_intervened" and response["message"].endswith(g.TEXT)
        assert response["data"]["guardrail_outcome"] == g.INTERVENTION
        assert records(response["data"]) == [("OUTPUT", "PII", "EMAIL", "ANONYMIZED")]
        assert g.hold_lines(caplog) == []
        assert model.calls == (1 if first is None else 2) and len(client.requests) == 1
        # The run record stays at its first stage and names the kept report, the only stored object.
        record = decoded(ports.items["PROMPT_RUN#" + run_id])["record"]
        assert record["stage"] == "enrolled" and type(record.get("report_checkpoint_ref")) is dict
        assert len(ports.objects) == 1
        kept = [
            {
                "event": "prompt_report",
                "status": "available",
                "run_id": run_id,
                "found": True,
                "report": terminal,
            }
        ]
        before = effects()
        # A later request of the same session reads the kept report: it equals what the run returned.
        BedrockAgentCoreContext.set_request_context("ordinary-second-request", SESSION)
        assert await read({"action": "get_prompt_report", "run_id": run_id}) == kept
        # Neither a status request nor a resume request continues the run: no run event, no model call, no
        # job, no new agent and no write. What the two requests answer is pinned by other tests.
        answers = await read({"action": "get_prompt_run_status", "run_id": run_id})
        answers += await read({"action": "resume_prompt_run", "run_id": run_id})
        assert answers and not {"run_started", "run_completed"} & {event["event"] for event in answers}
        assert effects() == before
        assert await read({"action": "get_prompt_report", "run_id": run_id}) == kept
        # The next step for the user: the same prompt again starts a new run from the beginning, and the
        # report of the first run stays as it was.
        BedrockAgentCoreContext.set_request_context("ordinary-third-request", SESSION)
        again = await asyncio.wait_for(read({"prompt": work.literal}), g.LIVENESS_WAIT_SECONDS)
        assert [event["event"] for event in again] == ["run_started", "run_completed"]
        assert again[0]["run_id"] != run_id and again[-1]["status"] == "response_received"
        assert await read({"action": "get_prompt_report", "run_id": run_id}) == kept

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
