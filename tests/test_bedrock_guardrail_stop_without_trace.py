"""A Bedrock guardrail stop that carries no guardrail trace, through the SDK's real ``BedrockModel``.

From Strands 1.54.0 (the lock pins 1.65.0) ``BedrockModel`` treats a ``guardrail_intervened`` stop without a
guardrail trace as blocked content and emits its user-input redaction, by default ``[User input redacted.]``.
Strands 1.42.0 redacted only when a trace named a blocked policy. The product binds its guardrail with id,
version and trace mode only, so the SDK's redaction defaults apply. These tests pin what the product does next:

- an ordinary chat turn is answered as a guardrail intervention; only that turn's agent holds the redacted text;
- the warm model continuation of a normal run is held as changed history, with the fixed hold line.

A stub Bedrock client replays one fixed response stream: no provider, guard or network call. The expectations
follow the installed SDK, so the same file documents the 1.42.0 behaviour. Fixtures establish source behaviour,
not native guardrail classification or a deployed run.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import logging
import threading
from types import SimpleNamespace

import model_acquisition_fixtures as f
import pytest
from strands.handlers.callback_handler import null_callback_handler
from strands.models.bedrock import BedrockModel
from test_normal_research_profile import PROMPT, LifecycleModel, setup
from test_runtime_warm_model_continuation import complete_fixture

from amlas import api
from amlas.backends.adapters import model_weight_acquisition as adapter_source
from amlas.config.guardrails import guardrail_model_kwargs
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode
from amlas.data_eng import model_acquisition_schema
from amlas.flows import model_weight_acquisition_flow
from amlas.tools.coding_preparation import list_coding_preparation_profiles
from amlas.tools.model_weight_tools import acquire_model_weights
from amlas.tui.chat import tool_binding_backend as binding

SDK = tuple(int(part) for part in importlib.metadata.version("strands-agents").split(".")[:2])
# Read from the SDK sources: the stop-reason fallback is in every release from 1.54.0 and in none before it.
REDACTS_WITHOUT_TRACE = SDK >= (1, 54)
REDACTED = "[User input redacted.]"
TEXT = "Neutral provider text for a guardrail stop."
QUESTION = "Ordinary neutral question."
HOLD_LINE = "AMLAS_INPUT_HOLD phase=history_qualification reason=history_prefix_refused"
BLOCKED_TRACE = {
    "guardrail": {
        "inputAssessment": {
            "fixtureguardrail": {
                "contentPolicy": {
                    "filters": [
                        {"type": "VIOLENCE", "confidence": "HIGH", "detected": True, "action": "BLOCKED"}
                    ]
                }
            }
        }
    }
}
INTERVENTION = {
    "schema_version": 1,
    "stop_reason": "guardrail_intervened",
    "outcome": "INTERVENED",
    "assessment_status": "unavailable",
    "action": "UNKNOWN",
}


class ReplayClient:
    """Stands in for the Bedrock runtime client: one fixed guardrail stop per ``converse_stream`` call."""

    meta = SimpleNamespace(region_name="us-west-2")

    def __init__(self, trace=None):
        self.trace = trace
        self.requests = []

    def converse_stream(self, **request):
        self.requests.append(copy.deepcopy(request))
        metadata = {
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        }
        if self.trace is not None:
            metadata["trace"] = copy.deepcopy(self.trace)
        return {
            "stream": iter(
                [
                    {"messageStart": {"role": "assistant"}},
                    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": TEXT}}},
                    {"contentBlockStop": {"contentBlockIndex": 0}},
                    {"messageStop": {"stopReason": "guardrail_intervened"}},
                    {"metadata": metadata},
                ]
            )
        }


class ReplaySession:
    region_name = "us-west-2"

    def __init__(self, client):
        self.replay = client

    def client(self, **kwargs):
        return self.replay


def bound_model(monkeypatch, *, trace=None, trace_mode="disabled"):
    """The SDK's ``BedrockModel`` with the guardrail arguments the product's own configuration produces."""
    with monkeypatch.context() as scoped:
        scoped.setenv("AMLAS_GUARDRAIL_ID", "fixtureguardrail")
        scoped.setenv("AMLAS_GUARDRAIL_VERSION", "1")
        scoped.setenv("AMLAS_GUARDRAIL_TRACE", trace_mode)
        scoped.delenv("AMLAS_GUARDRAIL_ENABLED", raising=False)
        selected = guardrail_model_kwargs()
    assert selected == {
        "guardrail_id": "fixtureguardrail",
        "guardrail_version": "1",
        "guardrail_trace": trace_mode,
    }, "the product names no redaction setting, so the SDK defaults decide"
    client = ReplayClient(trace)
    model = BedrockModel(
        boto_session=ReplaySession(client), model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", **selected
    )
    assert model.client is client
    return model, client


def record_form(messages):
    return [{key: value for key, value in message.items() if key != "tracking_id"} for message in messages]


def hold_lines(caplog):
    return [
        record.getMessage() for record in caplog.records if record.getMessage().startswith("AMLAS_INPUT_HOLD")
    ]


@pytest.mark.parametrize("case", ["no_trace", "blocked_trace"])
def test_sdk_bedrock_model_redacts_user_input_on_a_guardrail_stop(monkeypatch, case):
    model, client = bound_model(
        monkeypatch,
        trace=BLOCKED_TRACE if case == "blocked_trace" else None,
        trace_mode="enabled" if case == "blocked_trace" else "disabled",
    )

    async def events():
        return [event async for event in model.stream([{"role": "user", "content": [{"text": QUESTION}]}])]

    seen = asyncio.run(events())
    assert client.requests[0]["guardrailConfig"] == {
        "guardrailIdentifier": "fixtureguardrail",
        "guardrailVersion": "1",
        "trace": "enabled" if case == "blocked_trace" else "disabled",
    }
    order = [next(iter(event)) for event in seen]
    redactions = [event["redactContent"] for event in seen if "redactContent" in event]
    if case == "blocked_trace" or REDACTS_WITHOUT_TRACE:
        # A blocked trace redacts on every SDK version; without a trace only the newer SDK does.
        assert order == [
            "messageStart",
            "contentBlockDelta",
            "contentBlockStop",
            "messageStop",
            "redactContent",
            "metadata",
        ]
        assert redactions == [{"redactUserContentMessage": REDACTED}]
    else:
        assert order == ["messageStart", "contentBlockDelta", "contentBlockStop", "messageStop", "metadata"]
        assert redactions == []


def test_ordinary_chat_turn_reports_the_intervention_and_holds_nothing(monkeypatch, caplog):
    model, client = bound_model(monkeypatch)
    backend = binding.ToolBindingBackend(Settings(), model=model)
    build, agents = backend.build_agent, []

    def built(**kwargs):
        agents.append(build(**kwargs))
        agents[-1].callback_handler = null_callback_handler  # keep the provider text off the test output
        return agents[-1]

    monkeypatch.setattr(backend, "build_agent", built)
    with caplog.at_level(logging.WARNING):
        turn = backend.chat_with_tools(QUESTION)

    # What the user sees: the provider's text, typed as an intervention, with no invented assessment.
    assert turn.final_text == TEXT
    assert turn.guardrail_outcome.to_dict() == INTERVENTION
    assert turn.guardrail_assessment is None
    # Nothing is held or refused.
    assert backend._input_hold is False and turn.input_admission is None
    assert hold_lines(caplog) == []
    # The model was asked once, with the question as typed.
    assert len(client.requests) == len(agents) == 1
    assert client.requests[0]["messages"] == [{"role": "user", "content": [{"text": QUESTION}]}]
    # What that turn's agent keeps: the newer SDK replaced the question with its redaction text.
    kept = record_form(agents[0].messages)
    assert [message["role"] for message in kept] == ["user", "assistant"]
    assert kept[0]["content"] == [{"text": REDACTED if REDACTS_WITHOUT_TRACE else QUESTION}]
    assert kept[1]["content"] == [{"text": TEXT}]
    # The next turn builds a new agent: neither the question nor the redaction text is sent again.
    backend.chat_with_tools("Second ordinary neutral question.")
    assert len(agents) == 2
    assert client.requests[1]["messages"] == [
        {"role": "user", "content": [{"text": "Second ordinary neutral question."}]}
    ]


class BedrockResume(LifecycleModel):
    """The scripted lifecycle model until the original completes; the resumed call goes through ``BedrockModel``."""

    def __init__(self, bedrock):
        super().__init__(model_acquisition=True)
        self.bedrock = bedrock

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls < len(self.requests):
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
            return
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        async for event in self.bedrock.stream(messages, tool_specs, system_prompt, **kwargs):
            yield event


def test_normal_run_warm_continuation_is_held_when_the_sdk_redacts_the_resumed_input(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    f.patch_expected(monkeypatch, [adapter_source, model_acquisition_schema])
    native, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(adapter_source, native, s3)
    observing = threading.Event()
    describe = native.describe_processing_job

    def observe_original(**kwargs):
        observing.set()
        return describe(**kwargs)

    monkeypatch.setattr(native, "describe_processing_job", observe_original)
    monkeypatch.setattr(model_weight_acquisition_flow, "_resolve_backend", lambda: adapter)
    bedrock, client = bound_model(monkeypatch)
    model = BedrockResume(bedrock)
    selected = setup(
        monkeypatch,
        model_override=model,
        extra_tools=(list_coding_preparation_profiles, acquire_model_weights),
    )
    selected.policy.mode = ApprovalMode.AUTO
    original_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        await original_sleep(0.005 if delay == 15 else delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    async def scenario():
        task = asyncio.create_task(api.run_prompt(PROMPT, settings=Settings(), policy=selected.policy))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not task.done() and not observing.is_set():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert not task.done() and model.calls == 4
            backend, agent = selected.backends[0], selected.agents[0]
            before = copy.deepcopy(agent.messages)
            pending = backend._warm_pending
            assert pending is not None and pending.agent is agent
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=pending.record.handle, adapter=adapter, native=native),
            )
            with caplog.at_level(logging.WARNING):
                response = await asyncio.wait_for(task, 5)
            return response, backend, agent, before
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    response, backend, agent, before = asyncio.run(scenario())

    # The resumed model call was made once and carried the completion DATA as the product built it.
    assert model.calls == 5 and len(client.requests) == 1
    sent = client.requests[0]["messages"][-1]
    assert sent["role"] == "user" and len(sent["content"]) == 1
    assert '"event_type":"original_model_acquisition_completed"' in sent["content"][0]["text"]
    # The conversation before the resume is untouched; the resume added the DATA message and the model's answer.
    assert record_form(agent.messages[: len(before)]) == record_form(before)
    added = record_form(agent.messages[len(before) :])
    assert [message["role"] for message in added] == ["user", "assistant"]
    assert added[1]["content"] == [{"text": TEXT}]
    # Admission records are the same on both SDK versions, and no assessment is invented without a trace.
    records = response.data["input_admission"]["records"]
    assert records[-1]["boundary"] == "COMPLETION_DATA" and records[-1]["outcome"] == "ALLOW"
    assert sum(record["outcome"] == "BLOCKED" for record in records) == 1
    assert response.data.get("guardrail_assessment") is None
    if REDACTS_WITHOUT_TRACE:
        # The SDK replaced the admitted DATA message, so the product holds the run as changed history.
        assert added[0]["content"] == [{"text": REDACTED}]
        assert hold_lines(caplog) == [HOLD_LINE]
        assert backend._input_hold is True and backend._warm_closed == "refused"
        assert response.action == "input_admission_held"
        assert response.data["input_admission"]["status"] == "held"
        assert response.data["warm_model_continuation"] == {
            "disposition": "refused",
            "diagnostic": {"phase": "resume", "reason": "continuation_failed", "error_class": "UNKNOWN"},
        }
        # What the user sees: the fixed hold text, without the provider's text, tool list or redaction text.
        assert response.message == "Model input was held at the admission boundary."
        assert "guardrail_outcome" not in response.data and response.data["tools_used"] == []
    else:
        # Strands 1.42.0 leaves the DATA message alone: the stop is reported as a plain intervention.
        assert added[0]["content"] == sent["content"]
        assert hold_lines(caplog) == []
        assert backend._input_hold is False and backend._warm_closed == "guardrail_intervened"
        assert response.action == "guardrail_intervened"
        assert response.data["input_admission"]["status"] == "quarantined"
        assert response.data["warm_model_continuation"] == {"disposition": "guardrail_intervened"}
        assert response.message.endswith(TEXT)
        assert response.data["guardrail_outcome"] == INTERVENTION
