"""The normal-run chat agent keeps up to 100 messages before the SDK trims its history.

``_build.build_agent`` passed no conversation manager, so Strands applied its default
``SlidingWindowConversationManager(window_size=40)``. That manager trims the oldest messages once the
history holds more than 40, and a trim after the fine-tune pause edits a history the product's witness
then refuses (``history_prefix_refused``). One real run (2026-11-12) held 33 messages at the call after the
pair result. These tests pin the 100-message window on the normal-run build only, the unchanged saved
state of the manager, the restore of a state written by the default manager, and the SDK default on
every other build path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sdk_scripted_model import ScriptedModel
from strands import Snapshot
from strands.agent.conversation_manager import SlidingWindowConversationManager

from amlas import _build
from amlas.config.settings import Settings
from amlas.tui.chat import tool_binding_backend as binding

NORMAL_RUN_WINDOW = 100
SDK_DEFAULT_WINDOW = 40
SRC = Path(__file__).resolve().parents[1] / "src" / "amlas"


def _history(count):
    """``count`` alternating user/assistant text messages, starting with the user."""
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": [{"text": f"message {index}"}]}
        for index in range(count)
    ]


def _normal_agent(monkeypatch, messages=None):
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [])
    backend = binding.ToolBindingBackend(Settings(), model=ScriptedModel([]))
    assert backend.is_tool_capable, "the window is a property of the installed SDK agent, never a shim"
    return backend.build_agent(messages=messages)


def _default_agent(messages=None):
    """Another ``build_agent`` caller: no conversation manager is named."""
    return _build.build_agent(
        name="other_caller", system_prompt="neutral", model=ScriptedModel([]), messages=messages
    )


def test_normal_run_agent_carries_a_100_message_sliding_window(monkeypatch):
    manager = _normal_agent(monkeypatch).conversation_manager
    assert type(manager) is SlidingWindowConversationManager
    assert manager.window_size == NORMAL_RUN_WINDOW
    # Only the window changes; the other SDK defaults stay as they were.
    assert manager.should_truncate_results is True
    assert manager.per_turn is False
    assert manager.removed_message_count == 0


def test_each_normal_run_build_gets_its_own_manager(monkeypatch):
    first = _normal_agent(monkeypatch).conversation_manager
    second = _normal_agent(monkeypatch).conversation_manager
    assert first is not second


@pytest.mark.parametrize("count", [41, 60, 99, 100])
def test_a_41_to_100_message_history_is_not_trimmed(monkeypatch, count):
    history = _history(count)
    agent = _normal_agent(monkeypatch, messages=history)
    agent.conversation_manager.apply_management(agent)
    assert agent.messages == history
    assert agent.conversation_manager.removed_message_count == 0


def test_the_window_still_bounds_a_101_message_history(monkeypatch):
    agent = _normal_agent(monkeypatch, messages=_history(101))
    agent.conversation_manager.apply_management(agent)
    assert len(agent.messages) <= NORMAL_RUN_WINDOW
    assert agent.conversation_manager.removed_message_count == 101 - len(agent.messages)


def test_the_sdk_default_window_trims_a_41_message_history():
    """The hazard the normal-run window removes, shown on the default build path."""
    agent = _default_agent(messages=_history(41))
    agent.conversation_manager.apply_management(agent)
    assert len(agent.messages) < 41
    assert agent.conversation_manager.removed_message_count > 0


def test_saved_manager_state_keeps_its_keys_and_class_name(monkeypatch):
    normal = _normal_agent(monkeypatch).conversation_manager.get_state()
    default = _default_agent().conversation_manager.get_state()
    assert set(normal) == set(default) == {"__name__", "removed_message_count", "model_call_count"}
    assert (
        normal
        == default
        == {
            "__name__": "SlidingWindowConversationManager",
            "removed_message_count": 0,
            "model_call_count": 0,
        }
    )


def test_snapshot_written_by_the_default_manager_restores(monkeypatch):
    history = _history(33)
    # The stored form: the product keeps the session snapshot in its four-field record inventory.
    snapshot = Snapshot.from_dict(binding._record_session_snapshot(_default_agent(messages=history)))
    assert set(snapshot.data) == {"messages", "state", "conversation_manager_state", "interrupt_state"}
    assert snapshot.data["conversation_manager_state"]["removed_message_count"] == 0

    agent = _normal_agent(monkeypatch)
    agent.load_snapshot(snapshot)
    assert agent.messages == history
    assert agent.conversation_manager.window_size == NORMAL_RUN_WINDOW
    assert agent.conversation_manager.get_state() == snapshot.data["conversation_manager_state"]
    assert binding._record_session_snapshot(agent)["data"] == snapshot.data


def test_restore_accepts_a_state_written_by_the_default_manager(monkeypatch):
    written = SlidingWindowConversationManager().get_state() | {
        "removed_message_count": 3,
        "model_call_count": 5,
    }
    manager = _normal_agent(monkeypatch).conversation_manager
    assert manager.restore_from_session(written) is None
    assert manager.get_state() == written
    assert manager.window_size == NORMAL_RUN_WINDOW


@pytest.mark.parametrize("named", [False, True])
def test_other_build_paths_keep_the_sdk_default(named):
    kwargs = {"conversation_manager": None} if named else {}
    agent = _build.build_agent(
        name="other_caller", system_prompt="neutral", model=ScriptedModel([]), **kwargs
    )
    assert type(agent.conversation_manager) is SlidingWindowConversationManager
    assert agent.conversation_manager.window_size == SDK_DEFAULT_WINDOW


def test_build_agent_forwards_a_named_conversation_manager():
    manager = SlidingWindowConversationManager(window_size=7)
    agent = _build.build_agent(
        name="other_caller", system_prompt="neutral", model=ScriptedModel([]), conversation_manager=manager
    )
    assert agent.conversation_manager is manager


def test_only_the_normal_run_chat_backend_names_a_conversation_manager():
    naming = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "conversation_manager=" in path.read_text(encoding="utf-8")
    )
    assert naming == ["tui/chat/tool_binding_backend.py"]
