"""Every model call logs the canonical size of the conversation it sends (numbers only).

The line `normal_conversation_size run_id= model_call= bytes= messages= limit=` follows the
validated content-free pattern of `normal_sft_capture_size`. A real SDK Agent runs against a scripted
model; no provider, native job or AWS call happens.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from types import SimpleNamespace

import pytest
from sdk_scripted_model import ScriptedModel
from strands.hooks.events import BeforeModelCallEvent
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.control_plane import normal_run_checkpoint
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
LOGGER = "amlas.tui.chat.tool_binding_backend"
RUN_ID = "cd" * 16
LIMIT = 160 * 1024
LINE = re.compile(
    r"^normal_conversation_size run_id=([0-9a-f]{32}) model_call=([0-9]+) bytes=([0-9]+) "
    r"messages=([0-9]+) limit=([0-9]+)$"
)
MARKER = "EXCLUDED_CONVERSATION_CONTENT_SENTINEL"


def canonical_bytes(messages):
    return len(json.dumps(messages, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def lines(caplog):
    return [
        r
        for r in caplog.records
        if r.name == LOGGER and r.getMessage().startswith("normal_conversation_size")
    ]


def agent_for(model, history=None):
    """Build the real agent; `agent.seen` holds `agent.messages` in record form at each BeforeModelCall."""
    backend = ToolBindingBackend(Settings())
    backend._model = model
    agent = backend.build_agent(messages=history)
    agent.callback_handler = lambda *args, **kwargs: None
    agent.seen = []
    agent.hooks.add_callback(
        BeforeModelCallEvent,
        lambda event: agent.seen.append(binding._record_messages(copy.deepcopy(event.agent.messages))),
    )
    return agent


@pytest.fixture
def active_run(monkeypatch):
    monkeypatch.setattr(normal_run_checkpoint, "current_normal_run", lambda: SimpleNamespace(run_id=RUN_ID))


def test_each_model_call_logs_exact_canonical_bytes_message_count_and_limit(active_run, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    model = ScriptedModel([("text", "first answer " + "é" * 40), ("text", "second answer")])
    agent = agent_for(model)
    agent("first prompt with accents é and quotes \"'")
    agent("second prompt")
    observed = lines(caplog)
    parsed = [LINE.fullmatch(record.getMessage()) for record in observed]
    assert all(parsed), [record.getMessage() for record in observed]
    assert [int(m.group(2)) for m in parsed] == [1, 2]
    assert [int(m.group(4)) for m in parsed] == [len(seen) for seen in agent.seen] == [1, 3]
    assert [int(m.group(3)) for m in parsed] == [canonical_bytes(seen) for seen in agent.seen]
    assert {m.group(1) for m in parsed} == {RUN_ID} and {int(m.group(5)) for m in parsed} == {LIMIT}
    assert all(r.exc_info is None and r.stack_info is None for r in observed)


def test_bytes_equal_what_the_sft_capture_measures_for_the_same_messages(active_run, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    model = ScriptedModel([("text", "answer one"), ("text", "answer two")])
    agent = agent_for(model)
    agent("prompt one")
    agent("prompt two")
    reported = int(LINE.fullmatch(lines(caplog)[1].getMessage()).group(3))
    messages = agent.seen[1]
    assert reported == len(normal_run_checkpoint._canonical(messages).encode("utf-8"))
    measured = []
    identifier = "tooluse_abcdefghijklmnop"
    with pytest.raises(ValueError, match="assistant SFT batch"):
        normal_run_checkpoint._causal_call(
            identifier, messages, _diagnostic_size=lambda count, outcome: measured.append((count, outcome))
        )
    # The capture envelope adds {"causal_prefix":...,"pending_tool_use_id":"<id>"} around the same bytes.
    assert measured == [(reported + 43 + len(identifier), "ok")]


def test_an_over_cap_conversation_is_reported_with_its_real_size(active_run, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    history = [
        {"role": "user", "content": [{"text": "x" * (LIMIT + 5000)}]},
        {"role": "assistant", "content": [{"text": "noted"}]},
    ]
    model = ScriptedModel([("text", "done")])
    agent = agent_for(model, history)
    agent("next prompt")
    only = LINE.fullmatch(lines(caplog)[0].getMessage())
    assert int(only.group(3)) == canonical_bytes(agent.seen[0]) > LIMIT
    assert int(only.group(4)) == 3


def test_no_line_without_an_active_normal_run(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    agent_for(ScriptedModel([("text", "done")]))("prompt")
    assert lines(caplog) == []


def test_content_never_reaches_the_log(active_run, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    agent_for(ScriptedModel([("text", "done")]))(MARKER)
    assert lines(caplog) and all(MARKER not in r.getMessage() for r in caplog.records if r.name == LOGGER)


def test_unmeasurable_history_and_logger_failure_cannot_change_the_model_call(
    active_run, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger=LOGGER)
    model = ScriptedModel([("text", "ok")])
    agent = agent_for(model)
    agent.messages.append({"role": "user", "content": [{"text": "bytes follow", "blob": b"\x00\x01"}]})
    agent.messages.append({"role": "assistant", "content": [{"text": "noted"}]})
    agent("prompt")
    assert model.calls == 1 and lines(caplog) == []

    def refuse(*args, **kwargs):
        raise RuntimeError("ordinary diagnostic logger unavailable")

    monkeypatch.setattr(binding.logger, "warning", refuse)
    model = ScriptedModel([("text", "ok")])
    agent_for(model)("prompt")
    assert model.calls == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ("x" * 32, 1, 10, 1),
        ("A" * 32, 1, 10, 1),
        (RUN_ID, 0, 10, 1),
        (RUN_ID, True, 10, 1),
        (RUN_ID, 1, -1, 1),
        (RUN_ID, 1, 10, -1),
        (RUN_ID, 1, 64 * 1024 * 1024 + 1, 1),
        (RUN_ID, 1, 10.0, 1),
    ],
)
def test_size_helper_rejects_values_outside_the_closed_set(caplog, arguments):
    caplog.set_level(logging.INFO, logger=LOGGER)
    binding._warn_normal_conversation_size(*arguments)
    assert lines(caplog) == []
