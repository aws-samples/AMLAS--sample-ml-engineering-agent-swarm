"""Normal SDK model-bound INPUT admission; neutral fixtures, no provider calls."""

from __future__ import annotations

import asyncio
import copy
import json
import logging

from strands import tool
from strands.models import Model
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.tui.chat import tool_binding_backend as binding

RAW = "ordinary prospective research information"


@tool
def research_fetch_urls() -> str:
    """Return ordinary neutral untrusted external information."""
    return RAW


class ModelFixture(Model):
    def __init__(self):
        self.calls = 0
        self.messages = []

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "ordinary-offline-admission-fixture"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("normal SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        if self.calls == 1:
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "ordinary-source", "name": "research_fetch_urls"}},
                }
            }
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{}"}}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            assert self.calls == 2
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ordinary complete"}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}


class ClientFixture:
    def __init__(self, malformed=False, deny=None):
        self.calls = []
        self.malformed = malformed
        self.deny = deny if deny is not None else lambda text: RAW in text
        self.close_calls = 0

    def close(self):
        # Real SDK clients close owned pools; this fixture has no socket pool.
        self.close_calls += 1

    def apply_guardrail(self, **kwargs):
        self.calls.append(kwargs)
        if self.malformed:
            return {}
        denied = self.deny(kwargs["content"][0]["text"]["text"])
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
            "action": "GUARDRAIL_INTERVENED" if denied else "NONE",
            "outputs": [],
            "assessments": [
                {
                    "contentPolicy": {
                        "filters": [
                            {
                                "type": "PROMPT_ATTACK",
                                "confidence": "LOW" if denied else "NONE",
                                "filterStrength": "HIGH",
                                "action": "BLOCKED" if denied else "NONE",
                                "detected": denied,
                            }
                        ]
                    }
                }
            ],
        }


def configure_guard(monkeypatch, client):
    import boto3

    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", "ordinaryselected")
    monkeypatch.setenv("AMLAS_GUARDRAIL_VERSION", "DRAFT")
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "true")

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def client(self, service, **kwargs):
            assert service == "bedrock-runtime"
            return client

    monkeypatch.setattr(boto3, "Session", Session)
    monkeypatch.setattr(boto3, "client", lambda service, **kwargs: client)


def configured_backend(monkeypatch, client):
    configure_guard(monkeypatch, client)
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [research_fetch_urls])
    model = ModelFixture()
    backend = binding.ToolBindingBackend(Settings(), model=model)
    agents = []
    build = backend.build_agent

    def capture(**kwargs):
        agent = build(**kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(backend, "build_agent", capture)
    return backend, model, agents


def test_normal_sdk_quarantines_denied_new_tool_projection_before_history_or_model(monkeypatch):
    client = ClientFixture()
    backend, model, agents = configured_backend(monkeypatch, client)
    backend.chat_with_tools("ordinary user question")
    assert client.calls, "configured model-bound inputs were never assessed"
    assert model.calls == 1, "denied tool projection dispatched a follow-up model cycle"
    assert all(
        x["source"] == "INPUT"
        and x["guardrailIdentifier"] == "ordinaryselected"
        and x["guardrailVersion"] == "DRAFT"
        for x in client.calls
    )
    assert RAW not in json.dumps(agents[0].messages), "denied new tool text entered SDK history"
    assert RAW not in json.dumps(model.messages), "denied new tool text reached model stream"


def test_missing_input_assessment_action_holds_before_first_model_or_tool(monkeypatch):
    client = ClientFixture(malformed=True)
    backend, model, _agents = configured_backend(monkeypatch, client)
    turn = backend.chat_with_tools("ordinary user question")
    assert model.calls == 0, "missing decision dispatched a model/tool cycle"
    assert client.calls and turn is not None


def test_normal_response_exposes_source_admission_hold_separate_from_native_guardrail():
    from amlas.tui.chat.engine import ChatEngine

    engine = ChatEngine.__new__(ChatEngine)
    engine._fold_agent_turn = lambda *args: None
    evidence = {
        "schema_version": 1,
        "data_only": True,
        "status": "held",
        "records": [],
        "native_admission": False,
        "quality_success": False,
        "blocked_component_attribution": "UNESTABLISHED",
    }
    turn = binding.AgentTurn("ordinary admission hold")
    turn.input_admission = evidence
    response = engine._agent_turn_response(turn)
    assert response.action == "input_admission_held"
    assert response.data["input_admission"] == evidence
    assert "guardrail_outcome" not in response.data


def test_nested_historical_tool_result_is_assessed_before_any_model_dispatch(monkeypatch):
    client = ClientFixture()
    backend, model, agents = configured_backend(monkeypatch, client)
    history = [
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "prior-source", "name": "research_fetch_urls", "input": {}}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "prior-source",
                        "status": "success",
                        "content": [{"json": {"excerpt": RAW}}],
                    }
                }
            ],
        },
    ]
    before = copy.deepcopy(history)
    turn = backend.chat_with_tools("ordinary follow-up question", history=history)
    assert client.calls
    assert model.calls == 0, "unassessed nested history entered the model stream"
    assert not agents
    assert history == before, "historical tool content was rewritten"
    assert turn.input_admission["status"] == "held"


def test_denied_generated_context_holds_before_agent_build(monkeypatch):
    client = ClientFixture()
    backend, model, agents = configured_backend(monkeypatch, client)
    turn = backend.chat_with_tools("ordinary question", context=RAW)
    assert model.calls == 0 and not agents
    assert turn.input_admission["status"] == "held"
    assert RAW not in json.dumps(turn.input_admission)


def test_sdk_hold_preserves_historical_prefix_and_restores_original_callback(monkeypatch):
    client = ClientFixture()
    backend, model, agents = configured_backend(monkeypatch, client)
    history = [
        {"role": "user", "content": [{"text": "ordinary earlier question"}]},
        {"role": "assistant", "content": [{"text": "ordinary earlier reply"}]},
    ]
    before = copy.deepcopy(history)
    callbacks = []
    build = backend.build_agent

    def with_callback(**kwargs):
        agent = build(**kwargs)

        def callback(**event):
            pass

        agent.callback_handler = callback
        callbacks.append(callback)
        return agent

    monkeypatch.setattr(backend, "build_agent", with_callback)
    turn = backend.chat_with_tools("ordinary question", history=history)
    assert model.calls == 1
    assert turn.input_admission["status"] == "held"
    assert binding._record_messages(history) == before
    assert binding._record_messages(agents[0].messages)[: len(before)] == before
    assert agents[0].callback_handler is callbacks[0]
    results = [
        block["toolResult"] for msg in agents[0].messages for block in msg["content"] if "toolResult" in block
    ]
    assert len(results) == 1 and results[0]["toolUseId"] == "ordinary-source"
    assert RAW not in json.dumps(results)


def test_normal_api_denied_producer_retains_one_genuine_accepted_original(monkeypatch):
    import test_runtime_connected_warm_model as connected

    from amlas import app as runtime

    client = ClientFixture(deny=lambda text: "model_acq_" in text)
    configure_guard(monkeypatch, client)
    monkeypatch.setattr(connected, "tool_payload", lambda result: copy.deepcopy(result))
    work = connected.connected_work(monkeypatch)

    async def read_normal_prompt():
        return [event async for event in runtime.agent_invocation({"prompt": work.literal})]

    events = asyncio.run(read_normal_prompt())
    response = events[-1]["response"]
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    assert work.model.calls == 1, "denied producer triggered a follow-up model cycle"
    backend = work.engines[0].tool_backend
    store = backend._model_acquisition_store
    assert len(store._records) == 1
    reference, record = next(iter(store._records.items()))
    pending = backend._warm_pending
    assert pending is not None and pending.reference == reference
    assert pending.record is record and pending.agent is work.agents[0]
    assert pending.tool_use_id == "original-model-use"
    assert record.handle.id == work.native.requests[0]["ProcessingJobName"]
    assert record.handle.extra["accepted"] is True
    assert backend._input_hold and backend._warm_closed == "refused"
    assert backend._warm_consumed == set()
    assert work.agents[0].callback_handler is work.callbacks[0]
    assert reference not in json.dumps(work.agents[0].messages)
    assert response["action"] == "input_admission_held"
    assert response["data"]["input_admission"]["status"] == "held"


def test_denied_original_completion_is_not_consumed_and_exposes_hold(monkeypatch, tmp_path):
    from test_runtime_warm_model_continuation import complete_fixture, original_work, step

    client = ClientFixture(deny=lambda text: '"event_type":"original_model_acquisition_completed"' in text)
    configure_guard(monkeypatch, client)
    work = original_work(monkeypatch)
    pending = work.backend._warm_pending
    before = copy.deepcopy(work.agent.messages)
    callback = work.agent.callback_handler
    consumed = set(work.backend._warm_consumed)
    complete_fixture(monkeypatch, tmp_path, work)
    result = step(work)
    assert result.disposition == "refused"
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert work.agent.messages == before
    assert work.agent.callback_handler is callback
    assert work.backend._warm_pending is pending and pending.record.handle is work.handle
    assert work.backend._warm_consumed == consumed
    assert result.turn is not None, "held completion omitted its source-owned admission response"
    assert result.turn.input_admission["status"] == "held"
    count = len(client.calls)
    repeat = step(work)
    assert repeat.disposition == "refused"
    assert len(client.calls) == count and work.model.calls == 2 and len(work.native.requests) == 1


def test_unguarded_normal_sdk_retains_binary_history_legacy_path(monkeypatch):
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    monkeypatch.delenv("AMLAS_GUARDRAIL_ID", raising=False)
    monkeypatch.delenv("AMLAS_GUARDRAIL_VERSION", raising=False)
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [research_fetch_urls])
    model = ModelFixture()
    backend = binding.ToolBindingBackend(Settings(), model=model)
    history = [
        {
            "role": "user",
            "content": [{"image": {"format": "png", "source": {"bytes": b"ordinary neutral fixture bytes"}}}],
        }
    ]
    before = copy.deepcopy(history)
    turn = backend.chat_with_tools("ordinary question", history=history)
    assert model.calls == 2, "unguarded legacy binary history was held by admission serialization"
    assert turn.input_admission is None
    assert binding._record_messages(history) == before and model.messages[0][: len(before)] == before


def test_returned_admission_records_cannot_mutate_private_receipts(monkeypatch):
    client = ClientFixture(deny=lambda text: False)
    backend, model, _agents = configured_backend(monkeypatch, client)
    turn = backend.chat_with_tools("ordinary question")
    assert model.calls == 2
    before = copy.deepcopy(backend._input_admission_data())
    turn.input_admission["records"][0]["outcome"] = "BLOCKED"
    turn.input_admission["records"][0]["categories"].append("CONTENT:VIOLENCE")
    assert backend._input_admission_data() == before, (
        "public record aliases mutated the private admission ledger"
    )


def test_response_admission_mapping_is_detached_from_source_turn():
    from amlas.tui.chat.engine import ChatEngine

    engine = ChatEngine.__new__(ChatEngine)
    engine._fold_agent_turn = lambda *args: None
    evidence = {
        "schema_version": 1,
        "data_only": True,
        "status": "held",
        "records": [],
        "native_admission": False,
        "quality_success": False,
        "blocked_component_attribution": "UNESTABLISHED",
    }
    turn = binding.AgentTurn("ordinary admission hold", input_admission=evidence)
    response = engine._agent_turn_response(turn)
    response.data["input_admission"]["records"].append({"outcome": "ALLOW"})
    assert turn.input_admission["records"] == [], "normal response retained a mutable source-report alias"


def test_accepted_producer_pin_failure_holds_without_followup_or_replay(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    import test_runtime_connected_warm_model as connected

    from amlas import app as runtime

    client = ClientFixture(deny=lambda text: False)
    configure_guard(monkeypatch, client)
    monkeypatch.setattr(connected, "tool_payload", lambda result: copy.deepcopy(result))

    def unavailable_pin(handle):
        raise ValueError("ordinary original pin unavailable")

    monkeypatch.setattr(binding.ToolBindingBackend, "_warm_handle_pin", staticmethod(unavailable_pin))
    work = connected.connected_work(monkeypatch)

    async def read_normal_prompt():
        return [event async for event in runtime.agent_invocation({"prompt": work.literal})]

    events = asyncio.run(read_normal_prompt())
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    backend = work.engines[0].tool_backend
    assert len(backend._model_acquisition_store._records) == 1
    record = next(iter(backend._model_acquisition_store._records.values()))
    assert (
        record.handle.id == work.native.requests[0]["ProcessingJobName"]
        and record.handle.extra["accepted"] is True
    )
    assert work.model.calls == 1, "capture failure admitted a follow-up model turn without original custody"
    assert backend._input_hold and backend._warm_closed == "refused"
    assert backend._warm_consumed == set()
    assert events[-1]["response"]["action"] == "input_admission_held"

    warnings = [
        record
        for record in caplog.records
        if record.name == binding.__name__ and record.getMessage().startswith("AMLAS_INPUT_HOLD ")
    ]
    assert [record.getMessage() for record in warnings] == [
        "AMLAS_INPUT_HOLD phase=producer_capture reason=callback_refused"
    ], "accepted producer capture refusal omitted its fixed first-hold diagnostic"
    assert all(
        not record.args and record.exc_info is None and record.stack_info is None for record in warnings
    )


def test_accepted_producer_pin_failure_logger_error_preserves_hold_and_history(monkeypatch):
    import test_runtime_connected_warm_model as connected

    from amlas import app as runtime

    client = ClientFixture(deny=lambda text: False)
    configure_guard(monkeypatch, client)
    monkeypatch.setattr(connected, "tool_payload", lambda result: copy.deepcopy(result))
    warning_attempts = []
    expected = "AMLAS_INPUT_HOLD phase=producer_capture reason=callback_refused"

    def unavailable_pin(handle):
        raise ValueError("ordinary original pin unavailable")

    def refuse_warning(message, *args, **kwargs):
        if message == expected:
            warning_attempts.append((message, args, kwargs))
            raise RuntimeError("ordinary diagnostic logger unavailable")

    monkeypatch.setattr(binding.ToolBindingBackend, "_warm_handle_pin", staticmethod(unavailable_pin))
    monkeypatch.setattr(binding.logger, "warning", refuse_warning)
    work = connected.connected_work(monkeypatch)

    async def read_normal_prompt():
        return [event async for event in runtime.agent_invocation({"prompt": work.literal})]

    events = asyncio.run(read_normal_prompt())
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    backend = work.engines[0].tool_backend
    assert len(backend._model_acquisition_store._records) == 1
    record = next(iter(backend._model_acquisition_store._records.values()))
    assert (
        record.handle.id == work.native.requests[0]["ProcessingJobName"]
        and record.handle.extra["accepted"] is True
    )
    assert work.model.calls == 1, "logging failure admitted a follow-up model turn"
    assert backend._input_hold and backend._warm_closed == "refused"
    assert backend._warm_consumed == set()
    assert events[-1]["response"]["action"] == "input_admission_held"
    messages = work.agents[0].messages
    initial_history = work.model.messages_seen[0]
    assert binding._record_messages(messages)[: len(initial_history)] == initial_history
    results = [
        block["toolResult"] for message in messages for block in message["content"] if "toolResult" in block
    ]
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert results[0]["content"] == [{"text": "Tool content is unavailable at the input admission boundary."}]
    assert warning_attempts == [(expected, (), {})], (
        "producer capture warning was not attempted at the real refusal seam"
    )
