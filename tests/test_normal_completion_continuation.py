"""Fresh real-SDK handoff mechanics with declared qualification seams only.

No native/context/research authority, original claim, cold recovery or ML proof is
supplied. Hook events are actual SDK events; model/tool effects are inert counters.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import threading
from types import SimpleNamespace

import pytest
from strands import Agent, Snapshot, tool
from strands.hooks import BeforeModelCallEvent, BeforeToolCallEvent
from strands.models import Model
from test_normal_checkpoint_cas_contract import _policy

from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend


def _declared_handoff(monkeypatch, mode):
    state = SimpleNamespace(
        allowed=mode != "initial_refusal",
        current_event=None,
        event_index=0,
        fences={},
        events=[],
        calls=[],
        invocations=[],
        states=[],
        tool_effects=0,
    )

    def mark_event(event):
        assert event.agent is agent
        name = "model" if isinstance(event, BeforeModelCallEvent) else "tool"
        state.event_index += 1
        state.current_event = (name, state.event_index)
        state.events.append(name)
        state.fences[state.current_event] = set()

    def require_effect_fences(name):
        assert state.current_event[0] == name
        assert state.fences[state.current_event] == {"completion", "research"}

    class ScriptedModel(Model):
        def __init__(self):
            self.calls = 0

        def update_config(self, **kwargs):
            pass

        def get_config(self):
            return {"model_id": "declared-continuation-script"}

        def structured_output(self, *args, **kwargs):
            raise AssertionError("declared continuation uses only the SDK stream")

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            require_effect_fences("model")
            self.calls += 1
            yield {"messageStart": {"role": "assistant"}}
            if mode in {"positive", "postinvoke_revocation"} and self.calls == 1:
                yield {
                    "contentBlockStart": {
                        "contentBlockIndex": 0,
                        "start": {"toolUse": {"toolUseId": "declared-tool-use", "name": "declared_tick"}},
                    }
                }
                yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{}"}}}}
                yield {"contentBlockStop": {"contentBlockIndex": 0}}
                if mode == "postinvoke_revocation":
                    state.allowed = False
                yield {"messageStop": {"stopReason": "tool_use"}}
            else:
                yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
                yield {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": "Declared SDK continuation result."},
                    }
                }
                yield {"contentBlockStop": {"contentBlockIndex": 0}}
                if mode == "postinvoke_revocation":
                    state.allowed = False
                yield {"messageStop": {"stopReason": "end_turn"}}

    @tool
    def declared_tick() -> str:
        """Increment one inert declared unit counter."""
        require_effect_fences("tool")
        state.tool_effects += 1
        return "Declared inert tool result."

    model = ScriptedModel()
    history = [{"role": "user", "content": [{"text": "Declared already-admitted owner goal."}]}]
    origin = Agent(model=model, tools=[declared_tick], messages=copy.deepcopy(history), callback_handler=None)
    ready = origin.take_snapshot(preset="session").to_dict()
    agent = Agent(model=model, tools=[declared_tick], callback_handler=None)
    agent.load_snapshot(Snapshot.from_dict(ready))
    agent.hooks.add_callback(BeforeModelCallEvent, mark_event)
    agent.hooks.add_callback(BeforeToolCallEvent, mark_event)
    canonical_ready = checkpoint._canonical(ready)

    class DeclaredContext:
        def __init__(self):
            self.attempted = False
            self.lock = threading.Lock()
            # The restoration notice reads the claimed continuation epoch from the receipt the real context holds.
            self._completion_receipt = ("1" * 32, checkpoint._canonical({"epoch": 1}), "declared-task-ref")

        def qualify_completion(self, token, *, policy, qualify_original, sdk_snapshot):
            assert token == "1" * 32 and policy is backend.approval_policy
            assert checkpoint._canonical(sdk_snapshot) == canonical_ready
            state.calls.append("qualify_completion")
            if not state.allowed:
                raise ValueError("declared completion qualification refused")
            if state.current_event is not None:
                state.fences[state.current_event].add("completion")
            qualify_original()

        def begin_completion_invocation(self, token, *, policy, qualify_original, sdk_snapshot):
            self.qualify_completion(
                token, policy=policy, qualify_original=qualify_original, sdk_snapshot=sdk_snapshot
            )
            with self.lock:
                if self.attempted:
                    raise ValueError("declared invocation already attempted")
                self.attempted = True
                state.calls.append("begin")
                self._completion_receipt = (
                    "1" * 32,
                    checkpoint._canonical({"epoch": 2}),
                    "declared-task-ref",
                )

    context = DeclaredContext()
    backend = object.__new__(ToolBindingBackend)
    backend.approval_policy = _policy()
    backend._input_lock = threading.RLock()
    backend._input_hold = False
    backend._input_receipts = []
    backend._input_quarantined = False
    backend._input_cycle = "declared-handoff-cycle"

    def research_current(*args, **kwargs):
        state.calls.append("qualify_research")
        if not state.allowed:
            raise ValueError("declared research qualification refused")
        if state.current_event is not None:
            state.fences[state.current_event].add("research")
        return None  # This unit seam returns no research grant or restored phase.

    def invocation_state():
        fresh = {"declared_unit_state": len(state.states)}
        state.states.append(fresh)
        return fresh

    def forbidden_build(*args, **kwargs):
        raise AssertionError("continuation must reuse its already-loaded Agent")

    monkeypatch.setattr(backend, "_qualify_research_binding", research_current)
    monkeypatch.setattr(backend, "_invocation_state", invocation_state)
    monkeypatch.setattr(backend, "build_agent", forbidden_build)
    monkeypatch.setattr(backend, "_selected_input_checker", lambda: None)
    monkeypatch.setattr(backend, "_pending_pause_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "_extract_final_text", lambda result: "Declared SDK continuation result.")
    monkeypatch.setattr(backend, "_extract_executions", lambda result: [])
    monkeypatch.setattr(backend, "_extract_guardrail_outcome", lambda result: None)
    monkeypatch.setattr(backend, "_input_admission_data", lambda: {"status": "declared-unit"})
    original_call = Agent.__call__

    def exact_call(value, prompt=None, **kwargs):
        assert value is agent and prompt is None
        assert context.attempted and state.calls.count("begin") == 1
        assert state.states and kwargs["invocation_state"] is state.states[-1]
        state.invocations.append("same-agent-none")
        return original_call(value, prompt, **kwargs)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("continuation cannot reload its ready snapshot")

    monkeypatch.setattr(Agent, "__call__", exact_call)
    monkeypatch.setattr(agent, "load_snapshot", forbidden_load)
    engine = object.__new__(ChatEngine)
    engine.tool_backend = backend
    engine._normal_agent = agent
    engine._normal_completion = (context, "1" * 32, lambda: None)
    engine._normal_completion_snapshot = canonical_ready
    monkeypatch.setattr(engine, "_agent_turn_response", lambda turn: turn)
    return engine, state, model, context, history


def _continue(engine):
    result = engine.continue_normal_completion()
    return asyncio.run(result) if inspect.isawaitable(result) else result


def test_initial_completion_refusal_has_no_model_or_begin(monkeypatch):
    engine, state, model, context, history = _declared_handoff(monkeypatch, "initial_refusal")
    with pytest.raises((ValueError, RuntimeError)):
        _continue(engine)
    assert state.invocations == [] and model.calls == 0 and not context.attempted
    assert engine._normal_agent.messages == history


def test_actual_sdk_model_and_tool_fences_one_none_invocation_and_repeat_refusal(monkeypatch):
    engine, state, model, context, _history = _declared_handoff(monkeypatch, "positive")
    result = _continue(engine)
    assert result is not None
    assert state.invocations == ["same-agent-none"] and context.attempted
    assert model.calls == 2 and state.tool_effects == 1
    assert state.events.count("model") == 2 and state.events.count("tool") == 1
    with pytest.raises((ValueError, RuntimeError)):
        _continue(engine)
    assert state.invocations == ["same-agent-none"] and model.calls == 2


def test_postinvoke_revocation_cannot_return_ordinary_output_or_retry(monkeypatch):
    engine, state, model, context, _history = _declared_handoff(monkeypatch, "postinvoke_revocation")
    with pytest.raises((ValueError, RuntimeError)):
        _continue(engine)
    assert model.calls == 1 and state.invocations == ["same-agent-none"] and context.attempted
    assert state.events.count("tool") == 1 and state.tool_effects == 0
    with pytest.raises((ValueError, RuntimeError)):
        _continue(engine)
    assert model.calls == 1 and state.invocations == ["same-agent-none"]
