"""A caller's chat history stays as the caller built it, and a normal run passes no history.

``chat_with_tools(history=...)`` replays earlier turns into a new agent. The agent SDK may add its own fields
to the message dictionaries it is given (strands-agents 1.47.0 and later do; 1.42.0 does not). The agent
builder therefore gives the SDK a new dictionary for every message. A caller that passes one list on two turns
finds it equal to what it built, and the input guard is shown the same history text on both turns.

The copy is one level deep: nested content blocks are still shared with the caller, and these tests do not
cover them. A normal run (one new chat engine per prompt) passes no history to any agent build; the last
test pins that on the full scripted run, on a run restored after its job finished and on a run that was
also resumed while its job still ran. Neutral scripted model, guard and native fixtures; no provider or
network call; nothing here is a statement about a deployed run.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import json

import pytest
import test_input_admission_wiring as w
from normal_sft_cold_fixture import (
    close,
    cloud_data,
    finish_generation_pair,
    original_precommit,
    setup,
    wait_for,
)
from strands import Agent
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas import _build
from amlas import app as runtime
from amlas.config.settings import Settings
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")

# Upper bound for the waits on the stages of the scripted run. A slow machine needs the room; a stage that is
# never reached still fails when the bound ends.
LIVENESS_WAIT_SECONDS = 180


def earlier_turns():
    return [
        {"role": "user", "content": [{"text": "ordinary earlier question"}]},
        {"role": "assistant", "content": [{"text": "ordinary earlier reply"}]},
    ]


def as_guard_texts(message):
    """The texts of one history message as the product presents them to the guard: its role, then its text."""
    return [message["role"], *[block["text"] for block in message["content"]]]


def history_texts(calls, question):
    """The guard calls of one turn between its question and its tool result: the history texts."""
    texts = [call["content"][0]["text"]["text"] for call in calls]
    if not texts:
        return []
    assert texts[0] == question and json.loads(texts[-1]) == [{"text": w.RAW}]
    return texts[1:-1]


@pytest.mark.parametrize("guard", [True, False], ids=["guard-bound", "no-guard"])
def test_one_history_list_passed_on_two_turns_stays_equal_and_reads_the_same_to_the_guard(monkeypatch, guard):
    client = w.ClientFixture(deny=lambda text: False)  # the guard allows everything it is shown
    backend, model, _agents = w.configured_backend(monkeypatch, client)
    if not guard:
        monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
        monkeypatch.delenv("AMLAS_GUARDRAIL_ID", raising=False)
        monkeypatch.delenv("AMLAS_GUARDRAIL_VERSION", raising=False)
        backend = binding.ToolBindingBackend(Settings(), model=model)
    history = earlier_turns()
    built = copy.deepcopy(history)
    expected = [text for message in built for text in as_guard_texts(message)] if guard else []
    assert expected == (
        ["user", "ordinary earlier question", "assistant", "ordinary earlier reply"] if guard else []
    )

    shown = []
    for question in ("ordinary question", "ordinary second question"):
        mark = len(client.calls)
        model.calls = 0  # the scripted model starts its two-step script again
        model.messages.clear()
        turn = backend.chat_with_tools(question, history=history)
        assert turn.final_text == "ordinary complete" and model.calls == 2
        # The earlier turns were replayed to the model in front of the question.
        assert binding._record_messages(model.messages[0][: len(built)]) == built
        # The caller's dictionaries are equal to what the caller built.
        assert history == built
        shown.append(history_texts(client.calls[mark:], question))
    # The guard was shown the same history text on both turns: the messages exactly as the caller built them.
    assert shown == [expected, expected]


def recorded_agent_inputs(monkeypatch):
    """Record the ``messages`` argument of every SDK agent construction and the history of every chat turn."""
    constructions, turns = [], []
    init = Agent.__init__
    signature = inspect.signature(init)

    @functools.wraps(init)
    def recording_init(self, *args, **kwargs):
        constructions.append(copy.deepcopy(signature.bind(self, *args, **kwargs).arguments.get("messages")))
        return init(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", recording_init)
    chat = binding.ToolBindingBackend.chat_with_tools

    @functools.wraps(chat)
    def recording_chat(self, message, context="", *, thought_trace=None, history=None):
        turns.append(copy.deepcopy(history))
        return chat(self, message, context, thought_trace=thought_trace, history=history)

    monkeypatch.setattr(binding.ToolBindingBackend, "chat_with_tools", recording_chat)
    return constructions, turns


def test_builder_gives_the_sdk_equal_messages_and_nothing_for_an_empty_history(monkeypatch):
    constructions, _turns = recorded_agent_inputs(monkeypatch)
    history = earlier_turns()
    built = copy.deepcopy(history)
    for messages in (None, [], history):
        _build.build_agent(
            name="ordinary", system_prompt="ordinary", model=w.ModelFixture(), messages=messages
        )
    # No history and an empty history give the SDK no messages argument value; a history arrives equal.
    assert constructions == [None, None, built]
    assert history == built


async def events(payload):
    return [event async for event in runtime.agent_invocation(payload)]


@pytest.mark.parametrize("path", ["uninterrupted", "restored", "resumed_while_the_job_runs"])
def test_normal_run_passes_no_history_to_any_agent_build(monkeypatch, tmp_path, path):
    constructions, turns = recorded_agent_inputs(monkeypatch)

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            reader = state.reader
            if path != "uninterrupted":
                # The original worker ends while its fine-tune job runs.
                await close_original_worker(state.work, state.reader)
                if path == "resumed_while_the_job_runs":
                    # A resume while the job still runs restores the run in a new engine and builds its agent.
                    # The request ends with the status of the waiting run; nothing is continued.
                    engines, built = len(state.work.engines), len(constructions)
                    waiting = await events({"action": "resume_prompt_run", "run_id": state.run_id})
                    assert waiting[0] == {
                        "event": "resume_state",
                        "state": "restoring",
                        "run_id": state.run_id,
                    }
                    assert [event["event"] for event in waiting[1:]] == ["prompt_run_status"]
                    progress = waiting[1]["progress"]
                    assert progress["status"] == "InProgress" and progress["pending"] is True
                    assert len(state.work.engines) == engines + 1 and len(constructions) > built
                # A resume after the job has finished restores the run in a new engine and continues it.
                engines = len(state.work.engines)
                cloud_data(state, completed=True)
                state.native.states[state.original.id] = "Completed"
                reader = asyncio.ensure_future(
                    events({"action": "resume_prompt_run", "run_id": state.run_id})
                )
                await wait_for(state, reader, lambda: len(state.work.engines) > engines, "replacement engine")
            else:
                cloud_data(state, completed=True)
                state.native.states[state.original.id] = "Completed"
            deadline = asyncio.get_running_loop().time() + LIVENESS_WAIT_SECONDS
            shown = await finish_generation_pair(state, reader, deadline=deadline)
            assert shown[-1]["event"] == "run_completed" and shown[-1]["status"] == "response_received"
            return len(state.work.engines), len(state.work.seen_prompts)
        finally:
            await close(state)

    engines, prompts = asyncio.run(scenario())
    # The prompt went through the API once; every resume built one more engine.
    assert (
        prompts == 1 and engines == {"uninterrupted": 1, "restored": 2, "resumed_while_the_job_runs": 3}[path]
    )
    # Every agent of the run was built without messages, and the one chat turn carried no history.
    assert len(constructions) >= engines and all(not messages for messages in constructions)
    assert len(turns) == 1 and all(not history for history in turns)
