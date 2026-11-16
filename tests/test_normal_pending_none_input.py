"""Pinned SDK nullable input contract; declared nonnative pending-owner seams."""

from __future__ import annotations

import threading

import pytest
from normal_sft_cold_fixture import FullGraphModel
from strands import Agent
from strands.hooks import BeforeInvocationEvent

from amlas.tui.chat.tool_binding_backend import ToolBindingBackend, _InputAdmissionHeldError


def test_sdk_none_input_pending_hook_preserves_existing_history(monkeypatch):
    model = FullGraphModel()
    history = [{"role": "user", "content": [{"text": "Declared restored history."}]}]
    agent = Agent(model=model, messages=history, callback_handler=None)
    backend = object.__new__(ToolBindingBackend)
    backend._research_binding = {"original_completion_ready": True}
    backend._pending_batch_pause = None
    backend._input_hold = False
    backend._input_lock = threading.RLock()
    backend._tool_registry_cache = {}
    checker = object()
    monkeypatch.setattr(backend, "_pause_profile_current", lambda: backend._research_binding)
    monkeypatch.setattr(backend, "_selected_input_checker", lambda: checker)
    state = {}
    event = BeforeInvocationEvent(agent=agent, invocation_state=state, messages=None)
    assert event.messages is None
    try:
        backend._start_pending_pause(event)
    except TypeError:
        raise AssertionError("source_pending_none_invocation_refused") from None
    assert backend._pending_batch_pause.incoming == ()
    assert backend._pending_batch_pause.invocation_state is state
    assert backend._pending_batch_pause.request_state is state["request_state"]
    assert agent.messages == history and model.calls == 0
    incoming = [{"role": "user", "content": [{"text": "Declared new input."}]}]
    another = {}
    backend._start_pending_pause(
        BeforeInvocationEvent(agent=agent, invocation_state=another, messages=incoming)
    )
    assert backend._pending_batch_pause.incoming == tuple(
        backend._pause_history(message) for message in incoming
    )
    for invalid in ("declared-not-messages", {}, (), 1, False):
        backend._input_hold = False
        wrong = {}
        event = BeforeInvocationEvent(agent=agent, invocation_state=wrong, messages=invalid)
        with pytest.raises(_InputAdmissionHeldError):
            backend._start_pending_pause(event)
        assert backend._input_hold is True and "request_state" not in wrong
    assert agent.messages == history and model.calls == 0
