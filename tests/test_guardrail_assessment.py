"""Pinned SDK callback and bounded assessment DATA; no provider/network calls."""

from __future__ import annotations

import importlib
import json
import threading
from copy import deepcopy

import pytest
from strands import Agent
from strands.models import Model

from amlas.config.settings import Settings
from amlas.tui.chat import tool_binding_backend as binding

TEXT = "Ordinary provider response stays unchanged."


def trace(action="BLOCKED", policy="content", **extra):
    record = {"type": "VIOLENCE", "confidence": "HIGH", "action": action, **extra}
    assessment = {"contentPolicy": {"filters": [record]}}
    if policy == "pii":
        record = {"type": "EMAIL", "action": action, "match": "ordinary excluded fixture value", **extra}
        assessment = {"sensitiveInformationPolicy": {"piiEntities": [record], "regexes": []}}
    return {"guardrail": {"inputAssessment": {"fixture-guardrail": assessment}}}


class TraceModel(Model):
    def __init__(self, traces):
        self.traces = iter(traces)
        self.calls = 0
        self.event_ids = []

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "offline-assessment-proof"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("ordinary SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": TEXT}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "guardrail_intervened"}}
        value = next(self.traces)
        if value is not None:
            event = {
                "metadata": {
                    "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                    "metrics": {"latencyMs": 0},
                    "trace": value,
                }
            }
            self.event_ids.append(id(event))
            yield event


@pytest.fixture(autouse=True)
def inert_provider(monkeypatch):
    import boto3

    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setattr(boto3, "Session", lambda *a, **k: pytest.fail("unexpected AWS session"))
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("unexpected AWS client"))


def projection_module():
    try:
        module = importlib.import_module("amlas.tui.chat.guardrail_assessment")
    except ModuleNotFoundError:
        pytest.fail("missing bounded guardrail assessment projection")
    return module


def test_real_sdk_normal_callback_receives_original_metadata_once_without_thread_assumption():
    model = TraceModel([trace()])
    seen = []

    def existing(**kwargs):
        event = kwargs.get("event")
        if type(event) is dict and "metadata" in event:
            seen.append((id(event), threading.get_ident()))

    agent = Agent(model=model, tools=[], callback_handler=existing)
    result = agent("Ordinary source metadata.")
    assert result.stop_reason == "guardrail_intervened"
    assert result.message["content"][0]["text"] == TEXT
    assert [value[0] for value in seen] == model.event_ids
    assert len(seen) == model.calls == 1


def backend_with_existing_callback(monkeypatch, model):
    backend = binding.ToolBindingBackend(Settings(), model=model)
    original = backend.build_agent
    callbacks, seen = [], []

    def build(**kwargs):
        agent = original(**kwargs)

        def existing(**event):
            raw = event.get("event")
            if type(raw) is dict and "metadata" in raw:
                seen.append(id(raw))

        agent.callback_handler = existing
        callbacks.append(agent)
        return agent

    monkeypatch.setattr(backend, "build_agent", build)
    return backend, callbacks, seen


def test_normal_backend_captures_safe_action_and_chains_existing_callback_once(monkeypatch):
    model = TraceModel([trace()])
    backend, agents, seen = backend_with_existing_callback(monkeypatch, model)
    turn = backend.chat_with_tools("Ordinary source metadata.")
    value = getattr(turn, "guardrail_assessment", None)
    assert value is not None, "normal SDK trace was lost before assessment projection"
    wire = value.to_dict()
    assert wire["assessment_status"] == "complete" and wire["action"] == "BLOCKED"
    assert wire["records"][0]["direction"] == "INPUT"
    assert wire["records"][0]["category"] == "VIOLENCE"
    assert turn.final_text == TEXT and turn.guardrail_outcome.action == "UNKNOWN"
    assert seen == model.event_ids and model.calls == len(agents) == 1


def test_two_normal_turns_and_saved_callback_do_not_reuse_prior_capture(monkeypatch):
    model = TraceModel([trace("ANONYMIZED", "pii"), None])
    backend, agents, seen = backend_with_existing_callback(monkeypatch, model)
    first = backend.chat_with_tools("Ordinary source metadata.")
    assert first.guardrail_assessment.to_dict()["action"] == "ANONYMIZED"
    retained = agents[0].callback_handler
    second = backend.chat_with_tools("Ordinary source metadata.")
    assert second.guardrail_assessment is None
    assert seen == model.event_ids
    prior = first.guardrail_assessment.to_dict()
    retained(event={"metadata": {"trace": trace()}})
    assert first.guardrail_assessment.to_dict() == prior
    assert second.guardrail_assessment is None and model.calls == 2


@pytest.mark.parametrize(
    ("action", "policy"),
    [("BLOCKED", "content"), ("NONE", "content"), ("ANONYMIZED", "pii"), ("BLOCKED", "pii"), ("NONE", "pii")],
)
def test_complete_native_actions_are_closed_data_with_optional_detected(action, policy):
    value = projection_module().project_guardrail_trace(trace(action, policy)).to_dict()
    assert value["assessment_status"] == "complete" and value["action"] == action
    assert value["records"][0]["detected"] is None
    assert not any(
        key in json.dumps(value) for key in ("match", "modelOutput", "actionReason", "ordinary excluded")
    )


def test_mixed_and_contradictory_actions_stay_unknown():
    module = projection_module()
    data = trace()
    data["guardrail"]["outputAssessments"] = {
        "fixture-guardrail": [trace("ANONYMIZED", "pii")["guardrail"]["inputAssessment"]["fixture-guardrail"]]
    }
    mixed = module.project_guardrail_trace(data).to_dict()
    assert mixed["action"] == "UNKNOWN"
    assert {item["action"] for item in mixed["records"]} == {"BLOCKED", "ANONYMIZED"}
    invalid = module.project_guardrail_trace(trace(detected=False)).to_dict()
    assert invalid["action"] == "UNKNOWN" and invalid["assessment_status"] == "invalid"


@pytest.mark.parametrize("change", ["missing", "nonprimitive", "unknown_policy", "oversize"])
def test_missing_malformed_unknown_and_bounded_partial_never_claim_complete_action(change):
    data = trace()
    if change == "missing":
        data = {}
    elif change == "nonprimitive":
        data["guardrail"]["inputAssessment"]["fixture-guardrail"]["contentPolicy"]["filters"][0]["action"] = (
            True
        )
    elif change == "unknown_policy":
        data["guardrail"]["inputAssessment"]["fixture-guardrail"]["topicPolicy"] = {}
    else:
        items = data["guardrail"]["inputAssessment"]["fixture-guardrail"]["contentPolicy"]["filters"]
        items.extend(deepcopy(items[0]) for _ in range(129))
    value = projection_module().project_guardrail_trace(data).to_dict()
    assert value["action"] == "UNKNOWN" and value["assessment_status"] != "complete"
    assert len(value["records"]) <= 128


class ExcludedLeaf:
    def __str__(self):
        raise AssertionError("excluded protected leaf read")

    def __iter__(self):
        raise AssertionError("excluded protected leaf traversed")


def test_excluded_native_values_are_not_traversed_or_retained():
    data = trace("ANONYMIZED", "pii")
    leaf = ExcludedLeaf()
    data["guardrail"]["modelOutput"] = leaf
    data["guardrail"]["actionReason"] = leaf
    record = data["guardrail"]["inputAssessment"]["fixture-guardrail"]["sensitiveInformationPolicy"][
        "piiEntities"
    ][0]
    record["match"] = leaf
    result = projection_module().project_guardrail_trace(data).to_dict()
    assert result["action"] == "ANONYMIZED"
    assert "ExcludedLeaf" not in repr(result)


def test_required_native_pii_regex_collection_missing_is_not_complete():
    data = trace("ANONYMIZED", "pii")
    data["guardrail"]["inputAssessment"]["fixture-guardrail"]["sensitiveInformationPolicy"].pop("regexes")
    result = projection_module().project_guardrail_trace(data).to_dict()
    assert result["action"] == "UNKNOWN" and result["assessment_status"] != "complete"


def test_unknown_record_fields_degrade_coverage_without_reading_values():
    data = trace()
    data["guardrail"]["inputAssessment"]["fixture-guardrail"]["contentPolicy"]["filters"][0][
        "futurePolicy"
    ] = ExcludedLeaf()
    result = projection_module().project_guardrail_trace(data).to_dict()
    assert result["action"] == "UNKNOWN" and result["assessment_status"] == "partial"


def test_one_turn_missing_model_cycle_assessment_keeps_aggregate_unknown():
    capture = projection_module().GuardrailAssessmentCapture()
    calls = []
    callback = capture.wrap(lambda **kwargs: calls.append(1))
    callback(event={"messageStart": {"role": "assistant"}})
    callback(event={"metadata": {"trace": trace()}})
    callback(event={"messageStart": {"role": "assistant"}})
    callback(event={"messageStop": {"stopReason": "guardrail_intervened"}})
    result = capture.close().to_dict()
    assert calls == [1, 1, 1, 1]
    assert result["action"] == "UNKNOWN" and result["assessment_status"] == "partial"


def test_collector_synchronizes_callback_threads_and_closes_without_reusing_state():
    from concurrent.futures import ThreadPoolExecutor

    capture = projection_module().GuardrailAssessmentCapture()
    seen = []
    lock = threading.Lock()

    def existing(**kwargs):
        with lock:
            seen.append(1)

    callback = capture.wrap(existing)
    callback(event={"messageStart": {"role": "assistant"}})
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(callback, event={"metadata": {"trace": value}})
            for value in (trace(), trace("ANONYMIZED", "pii"))
        ]
        for future in futures:
            future.result(timeout=5)
    result = capture.close()
    assert result.action == "UNKNOWN" and len(result.records) == 2
    callback(event={"metadata": {"trace": trace()}})
    assert len(seen) == 4 and capture.close() is None
    assert len(result.records) == 2


def test_normal_api_exports_assessment_without_changing_intervention_fallback(monkeypatch):
    import asyncio

    from amlas import api

    model = TraceModel([trace("ANONYMIZED", "pii")])
    monkeypatch.setattr(
        binding.ToolBindingBackend,
        "build_agent",
        lambda _self, **kwargs: Agent(model=model, tools=[], callback_handler=lambda **event: None),
    )
    result = asyncio.run(api.run_prompt("Ordinary source metadata.", settings=Settings()))
    assert result.action == "guardrail_intervened" and TEXT in result.message
    assert result.data["guardrail_outcome"] == {
        "schema_version": 1,
        "stop_reason": "guardrail_intervened",
        "outcome": "INTERVENED",
        "assessment_status": "unavailable",
        "action": "UNKNOWN",
    }
    assert result.data["guardrail_assessment"]["action"] == "ANONYMIZED"
    assert result.data["guardrail_assessment"]["assessment_status"] == "complete"
    assert model.calls == 1 and result.data["tools_used"] == []


def test_pinned_sdk_sensitive_information_required_fields_match_closed_input_contract():
    from botocore.loaders import create_loader

    model = create_loader().load_service_model("bedrock-runtime", "service-2")
    shape = model["shapes"]["GuardrailSensitiveInformationPolicyAssessment"]
    assert set(shape["required"]) == {"piiEntities", "regexes"}


def test_unknown_root_policy_field_degrades_without_traversing_value():
    data = trace()
    data["guardrail"]["futurePolicy"] = ExcludedLeaf()
    result = projection_module().project_guardrail_trace(data).to_dict()
    assert result["action"] == "UNKNOWN" and result["assessment_status"] == "partial"


def test_partial_then_unavailable_capture_retains_partial_coverage_and_safe_records():
    capture = projection_module().GuardrailAssessmentCapture()
    callback = capture.wrap(lambda **kwargs: None)
    data = trace()
    data["guardrail"]["inputAssessment"]["fixture-guardrail"]["topicPolicy"] = {}
    callback(event={"messageStart": {"role": "assistant"}})
    callback(event={"metadata": {"trace": data}})
    callback(event={"messageStart": {"role": "assistant"}})
    callback(event={"metadata": {"trace": {"guardrail": {"inputAssessment": {}}}}})
    result = capture.close().to_dict()
    assert result["assessment_status"] == "partial" and result["action"] == "UNKNOWN"
    assert len(result["records"]) == 1


def test_projection_failure_does_not_skip_original_handler_or_change_reply(monkeypatch):
    module = projection_module()
    monkeypatch.setattr(
        module,
        "project_guardrail_trace",
        lambda value: (_ for _ in ()).throw(RuntimeError("ordinary projection unavailable")),
    )
    model = TraceModel([trace()])
    backend, agents, seen = backend_with_existing_callback(monkeypatch, model)
    turn = backend.chat_with_tools("Ordinary source metadata.")
    assert turn.final_text == TEXT and model.calls == 1
    assert seen == model.event_ids and len(agents) == 1
    assert turn.guardrail_outcome.outcome == "INTERVENED"
    assert turn.guardrail_assessment.action == "UNKNOWN"
    assert turn.guardrail_assessment.assessment_status == "invalid"


def test_existing_handler_failure_restores_original_callback_after_agent_failure(monkeypatch):
    model = TraceModel([trace()])
    backend = binding.ToolBindingBackend(Settings(), model=model)
    original = backend.build_agent
    retained = []

    def build(**kwargs):
        agent = original(**kwargs)

        def existing(**event):
            if type(event.get("event")) is dict and "metadata" in event["event"]:
                retained.append(agent.callback_handler)
                raise RuntimeError("ordinary callback unavailable")

        agent.callback_handler = existing
        retained.append((agent, existing))
        return agent

    monkeypatch.setattr(backend, "build_agent", build)
    turn = backend.chat_with_tools("Ordinary source metadata.")
    agent, existing = retained[0]
    assert "ordinary callback unavailable" in turn.final_text
    assert agent.callback_handler is existing and model.calls == 1
    assert len(retained) == 2


def test_empty_individual_output_assessment_cannot_be_masked_by_valid_input():
    data = trace()
    data["guardrail"]["outputAssessments"] = {"fixture-guardrail": [{}]}
    result = projection_module().project_guardrail_trace(data).to_dict()
    assert result["assessment_status"] == "partial" and result["action"] == "UNKNOWN"
    assert len(result["records"]) == 1
    isolated = {"guardrail": {"outputAssessments": {"fixture-guardrail": [{}]}}}
    empty = projection_module().project_guardrail_trace(isolated).to_dict()
    assert empty["assessment_status"] == "unavailable" and empty["action"] == "UNKNOWN"
