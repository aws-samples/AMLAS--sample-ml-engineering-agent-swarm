"""Declared public SDK mechanics only; no native or cold qualification proof."""

from __future__ import annotations

import copy
import json
from importlib.metadata import version
from types import SimpleNamespace

import pytest
from strands import Agent, Snapshot, tool
from strands.event_loop import streaming
from strands.hooks.events import BeforeInvocationEvent
from strands.models import Model

from amlas.control_plane.normal_run_checkpoint import _captured_snapshot_messages
from amlas.tui.chat.tool_binding_backend import _record_messages, _record_session_snapshot

FINAL_TEXT = "Declared continuation is complete."


@pytest.fixture(autouse=True)
def scripted_model_has_no_measured_first_byte_time(monkeypatch):
    """Keep the metadata of every assistant message as the scripted model wrote it.

    The SDK adds a measured `timeToFirstByteMs` to a message once its first stream event comes 1 ms or more
    after the model call, which happens only when the test process is held up. The SDK's clock reading in
    its stream module is zero here, and the SDK then adds no such field.
    """
    monkeypatch.setattr(streaming, "time", SimpleNamespace(time=lambda: 0.0))


class FinalTextModel(Model):
    def __init__(self):
        self.received_messages = []
        self.config = {"model_id": "declared-sdk-none-component"}

    def update_config(self, **kwargs):
        self.config.update(kwargs)

    def get_config(self):
        return copy.deepcopy(self.config)

    def structured_output(self, *args, **kwargs):
        raise AssertionError("this component cannot request structured output")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.received_messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": FINAL_TEXT}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }


def _ready_prefix():
    return [
        {"role": "user", "content": [{"text": "Declared mechanical continuation."}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "declared-ready-use",
                        "name": "declared_ready_step",
                        "input": {},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "declared-ready-use",
                        "status": "success",
                        "content": [{"text": "Declared step already complete."}],
                    }
                }
            ],
        },
    ]


def _record_inventory(record_property, snapshot):
    record_property("strands_agents_version", version("strands-agents"))
    record_property("session_snapshot_fields", json.dumps(sorted(snapshot)))
    record_property("session_snapshot_data_fields", json.dumps(sorted(snapshot["data"])))


def test_sdk_none_continues_declared_ready_snapshot_without_new_turn_or_tool_replay(record_property):
    prefix = _ready_prefix()
    producer_model = FinalTextModel()
    producer = Agent(model=producer_model, tools=[], messages=copy.deepcopy(prefix), callback_handler=None)
    ready = producer.take_snapshot(preset="session").to_dict()
    ready_before = copy.deepcopy(ready)
    _record_inventory(record_property, ready)
    snapshot = Snapshot.from_dict(ready)
    snapshot_before = copy.deepcopy(snapshot.to_dict())
    tool_calls = []

    @tool
    def declared_ready_step() -> str:
        """Return a harmless declared component response."""
        tool_calls.append("called")
        return "Declared step already complete."

    model = FinalTextModel()
    agent = Agent(model=model, tools=[declared_ready_step], callback_handler=None)
    agent_identity = id(agent)
    agent.load_snapshot(snapshot)
    hook_agents = []
    hook_states = []

    def before_invocation(event):
        hook_agents.append(event.agent)
        hook_states.append(copy.deepcopy(event.invocation_state))

    agent.hooks.add_callback(BeforeInvocationEvent, before_invocation)
    fresh_marker = {"declared_invocation": "fresh-ready-component"}
    marker_before = copy.deepcopy(fresh_marker)
    supplied_state = copy.deepcopy(marker_before)
    agent(None, invocation_state=supplied_state)

    assert model.received_messages == [prefix], "None must send exactly the declared ready prefix"
    assert producer_model.received_messages == []
    assert tool_calls == [], "completed history must not replay its declared tool"
    assert id(agent) == agent_identity
    assert len(hook_agents) == 1 and hook_agents[0] is agent
    assert hook_states == [marker_before]
    assert fresh_marker == marker_before
    assert supplied_state["declared_invocation"] == marker_before["declared_invocation"]
    assert ready == ready_before
    assert snapshot.to_dict() == snapshot_before
    assert _record_messages(agent.messages) == [
        *prefix,
        {
            "role": "assistant",
            "content": [{"text": FINAL_TEXT}],
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            },
        },
    ]


def test_default_session_snapshot_is_accepted_by_current_checkpoint_validator(record_property):
    prefix = _ready_prefix()
    model = FinalTextModel()
    agent = Agent(model=model, tools=[], messages=copy.deepcopy(prefix), callback_handler=None)
    snapshot = _record_session_snapshot(agent)
    before = copy.deepcopy(snapshot)
    _record_inventory(record_property, snapshot)

    assert _captured_snapshot_messages(snapshot) == prefix
    assert snapshot == before
    assert model.received_messages == []
