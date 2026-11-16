"""Real SDK result metadata; neutral fixtures, no model or guardrail invocation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics

from amlas import api
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat import tool_binding_backend as binding

TEXT = "Ordinary provider response remains unchanged."
WIRE = {
    "schema_version": 1,
    "stop_reason": "guardrail_intervened",
    "outcome": "INTERVENED",
    "assessment_status": "unavailable",
    "action": "UNKNOWN",
}


def sdk_result(reason="guardrail_intervened", text=TEXT):
    metrics = EventLoopMetrics()
    metrics.tool_metrics["source"] = SimpleNamespace(
        tool={"name": "research_search", "input": {"query": "ordinary source"}}
    )
    metrics.tool_metrics["observe"] = SimpleNamespace(
        tool={"name": "get_dataset_acquisition", "input": {"request_ref": "opaque-existing-ref"}}
    )
    return AgentResult(
        stop_reason=reason,
        message={"role": "assistant", "content": [{"text": text}]},
        metrics=metrics,
        state={},
    )


@pytest.fixture(autouse=True)
def inert_provider(monkeypatch):
    import boto3

    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setattr(boto3, "Session", lambda *a, **k: pytest.fail("unexpected provider session"))
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("unexpected provider client"))
    monkeypatch.setattr(binding.ToolBindingBackend, "is_tool_capable", property(lambda _: True))


def backend_for(monkeypatch, result):
    backend = binding.ToolBindingBackend(Settings(), approval_policy=ApprovalPolicy())
    calls = []

    def agent(prompt, *, invocation_state):
        calls.append((prompt, invocation_state))
        return result

    monkeypatch.setattr(backend, "build_agent", lambda **_: agent)
    return backend, calls


def test_real_sdk_intervention_is_typed_without_assessment_or_reply_changes(monkeypatch):
    result = sdk_result()
    backend, calls = backend_for(monkeypatch, result)
    turn = backend.chat_with_tools("Explain ordinary source metadata.")
    outcome = getattr(turn, "guardrail_outcome", None)
    assert outcome is not None, "real SDK guardrail stop reason was discarded"
    assert outcome.to_dict() == WIRE
    assert turn.final_text == TEXT
    assert [(item.tool_name, item.status) for item in turn.executions] == [
        ("research_search", "executed"),
        ("get_dataset_acquisition", "executed"),
    ]
    assert turn.executions[1].args is result.metrics.tool_metrics["observe"].tool["input"]
    assert len(calls) == 1, "observability must not retry the model"
    assert calls[0][1]["_amlas_approval_policy"].mode is ApprovalMode.ASK
    assert not any(key in WIRE for key in ("approved", "ready", "quality_success", "blocked"))


def test_normal_api_preserves_reply_and_tools_but_marks_intervened_unknown_action(monkeypatch):
    result = sdk_result()
    calls = []

    def build(_self, **kwargs):
        def agent(prompt, *, invocation_state):
            calls.append((prompt, invocation_state))
            return result

        return agent

    monkeypatch.setattr(binding.ToolBindingBackend, "build_agent", build)
    response = asyncio.run(api.run_prompt("Explain ordinary source metadata.", settings=Settings()))
    assert response.action == "guardrail_intervened", "API projects intervention as ordinary completion"
    assert response.data["guardrail_outcome"] == WIRE
    assert TEXT in response.message
    assert response.data["tools_used"] == ["research_search", "get_dataset_acquisition"]
    assert len(calls) == 1
    assert response.data["guardrail_outcome"]["action"] == "UNKNOWN"
    assert response.data["guardrail_outcome"]["outcome"] != "BLOCKED"


class ReasonSubtype(str):
    pass


@pytest.mark.parametrize(
    "reason",
    [
        "end_turn",
        "content_filtered",
        None,
        True,
        ["guardrail_intervened"],
        ReasonSubtype("guardrail_intervened"),
    ],
)
def test_other_missing_or_nonprimitive_sdk_stop_reason_keeps_legacy_turn(monkeypatch, reason):
    backend, calls = backend_for(monkeypatch, sdk_result(reason))
    turn = backend.chat_with_tools("Explain ordinary source metadata.")
    assert getattr(turn, "guardrail_outcome", None) is None
    assert turn.final_text == TEXT
    assert len(turn.executions) == 2 and len(calls) == 1


def test_blocked_words_and_non_sdk_result_do_not_fabricate_intervention(monkeypatch):
    text = "This request was blocked by content guardrail."
    result = SimpleNamespace(message={"content": [{"text": text}]}, stop_reason="guardrail_intervened")
    backend, calls = backend_for(monkeypatch, result)
    turn = backend.chat_with_tools("Explain ordinary source metadata.")
    assert getattr(turn, "guardrail_outcome", None) is None
    assert turn.final_text == text and len(calls) == 1


def test_end_turn_api_response_retains_existing_action_and_data(monkeypatch):
    result = sdk_result("end_turn")
    monkeypatch.setattr(binding.ToolBindingBackend, "build_agent", lambda _self, **_: lambda *a, **k: result)
    response = asyncio.run(api.run_prompt("Explain ordinary source metadata.", settings=Settings()))
    assert response.action == "tool_agent_turn"
    assert "guardrail_outcome" not in response.data
    assert response.data["tools_used"] == ["research_search", "get_dataset_acquisition"]
    assert TEXT in response.message


def test_blocked_word_reply_with_sdk_intervention_still_has_unknown_action(monkeypatch):
    text = "This request was blocked by content guardrail."
    backend, calls = backend_for(monkeypatch, sdk_result(text=text))
    turn = backend.chat_with_tools("Explain ordinary source metadata.")
    assert turn.final_text == text and len(calls) == 1
    assert turn.guardrail_outcome.to_dict() == WIRE
    assert turn.guardrail_outcome.action == "UNKNOWN"
    assert turn.guardrail_outcome.outcome == "INTERVENED"


def test_intervention_data_is_closed_and_frozen():
    kind = binding.GuardrailOutcome
    value = kind()
    with pytest.raises(AttributeError):
        value.action = "BLOCKED"
    for change in (
        {"schema_version": True},
        {"outcome": "BLOCKED"},
        {"assessment_status": "present"},
        {"action": "BLOCKED"},
    ):
        with pytest.raises(ValueError):
            kind(**change)


def test_sdk_result_subtype_preserves_legacy_reply_and_execution_records(monkeypatch):
    class ResultSubtype(AgentResult):
        pass

    original = sdk_result()
    result = ResultSubtype(original.stop_reason, original.message, original.metrics, original.state)
    backend, calls = backend_for(monkeypatch, result)
    turn = backend.chat_with_tools("Explain ordinary source metadata.")
    assert turn.guardrail_outcome is None
    assert turn.final_text == TEXT and len(calls) == 1
    assert len(turn.executions) == 2
    assert turn.executions[1].args is original.metrics.tool_metrics["observe"].tool["input"]
