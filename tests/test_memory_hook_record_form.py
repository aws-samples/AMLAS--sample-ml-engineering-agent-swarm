"""The Memory hook hands the Memory client each new message without the SDK's top-level ``tracking_id``.

Strands 1.65.0 puts a random ``tracking_id`` on every live message before ``MessageAddedEvent`` fires; Strands
1.42.0 wrote no such key. On both, the client must receive the message shape 1.42.0 produced, taken from a copy,
while the live SDK messages stay as the SDK left them. A recording client stands in for the Memory service: no
network, no AWS client. These tests say nothing about what a real Memory resource accepts.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

from sdk_scripted_model import ScriptedModel
from strands import Agent
from strands.hooks import MessageAddedEvent

from amlas.memory.hooks import MemoryHookProvider

ID = "tracking_id"


class RecordingClient:
    """Keeps what each ``create_event`` call received: a deep copy of the arguments and the message objects."""

    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []
        self.received = []

    def get_last_k_turns(self, **kwargs):
        return []

    def create_event(self, **kwargs):
        self.received.append(list(kwargs["messages"]))
        self.calls.append(copy.deepcopy(kwargs))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("neutral Memory write failure")
        return {}


def _provider(client):
    return MemoryHookProvider(client=client, memory_id="memory", actor_id="actor", session_id="session")


def _sent(client):
    return [message for call in client.calls for message in call["messages"]]


def _stubbed_live_messages():
    """Three live messages in the shape Strands 1.65.0 leaves on ``agent.messages`` (ids are fixed text here)."""
    return [
        {
            "role": "user",
            "content": [{"text": "neutral question"}],
            ID: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1",
        },
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "use-1", "name": "lookup", "input": {ID: "caller value"}}}],
            "metadata": {
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 1},
            },
            ID: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2",
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "use-1", "status": "success", "content": [{"json": {ID: "x"}}]}}
            ],
            ID: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa3",
        },
    ]


def test_stubbed_sdk_messages_reach_the_client_without_the_top_level_id():
    live = _stubbed_live_messages()
    before = copy.deepcopy(live)
    client = RecordingClient()
    provider = _provider(client)

    provider.on_message_added(SimpleNamespace(agent=SimpleNamespace(messages=live)))

    assert len(client.calls) == 1
    call = client.calls[0]
    assert {key: call[key] for key in ("memory_id", "actor_id", "session_id")} == {
        "memory_id": "memory",
        "actor_id": "actor",
        "session_id": "session",
    }
    assert sorted(call) == ["actor_id", "memory_id", "messages", "session_id"]
    expected = [{key: value for key, value in message.items() if key != ID} for message in before]
    assert call["messages"] == expected
    assert [sorted(message) for message in call["messages"]] == [
        ["content", "role"],
        ["content", "metadata", "role"],
        ["content", "role"],
    ]
    # Only the top-level key goes: a caller's own nested key of the same name is content and stays.
    assert call["messages"][1]["content"][0]["toolUse"]["input"] == {ID: "caller value"}
    assert call["messages"][2]["content"][0]["toolResult"]["content"] == [{"json": {ID: "x"}}]
    # The live messages are the SDK's: same objects, same ids, nothing removed.
    assert live == before
    assert all(sent is not kept for sent, kept in zip(client.received[0], live, strict=True))


def test_real_sdk_run_sends_the_same_shape_and_keeps_the_live_messages():
    client = RecordingClient()
    seen = []
    agent = Agent(
        model=ScriptedModel([("text", "neutral answer")]),
        hooks=[_provider(client)],
        callback_handler=None,
    )
    agent.hooks.add_callback(
        MessageAddedEvent, lambda event: seen.append((event.message, copy.deepcopy(event.message)))
    )

    agent("neutral question")

    assert [message["role"] for message in agent.messages] == ["user", "assistant"]
    assert len(client.calls) == 2, "one Memory write per added message"
    sent = _sent(client)
    assert [sorted(message) for message in sent] == [["content", "role"], ["content", "metadata", "role"]]
    assert sent == [{key: value for key, value in message.items() if key != ID} for message in agent.messages]
    # What the SDK put on each live message when it was added is still there afterwards.
    assert [live for live, _ in seen] == agent.messages
    assert all(live is kept for (live, _), kept in zip(seen, agent.messages, strict=True))
    assert [copied for _, copied in seen] == agent.messages
    assert all(message is not live for call in client.received for message in call for live in agent.messages)


def test_only_new_messages_are_sent_and_a_failed_write_is_sent_again():
    live = _stubbed_live_messages()
    client = RecordingClient(failures=1)
    provider = _provider(client)
    agent = SimpleNamespace(messages=live[:1])

    provider.on_message_added(SimpleNamespace(agent=agent))  # the write fails and is swallowed
    agent.messages.append(live[1])
    provider.on_message_added(SimpleNamespace(agent=agent))  # both messages are sent again, then counted
    agent.messages.append(live[2])
    provider.on_message_added(SimpleNamespace(agent=agent))
    provider.on_message_added(SimpleNamespace(agent=agent))  # nothing new: no call

    record = [{key: value for key, value in message.items() if key != ID} for message in live]
    assert [call["messages"] for call in client.calls] == [record[:1], record[:2], record[2:]]
