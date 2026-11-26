"""Text that cannot be encoded as UTF-8 (a lone surrogate) at each model-input boundary, as a full chat turn.

The input checker encodes every text strictly before it is assessed; text that cannot be encoded gets the
decision ``UNKNOWN`` / ``INVALID_INPUT``. With an input guard bound the turn is held: the guard is never shown
that text and no further model call is made. Without a guard nothing is assessed and the turn completes with
the text as given. The same holds for a tool result in text form or JSON form, for caller history and for the
prompt. The checker rule itself is pinned in ``test_input_admission.py``; these tests pin the whole turn.

Neutral scripted model and guard fixtures; no provider or network call. These tests say nothing about what a
real model endpoint does with such text or about a deployed run.
"""

from __future__ import annotations

import logging

import pytest
import test_input_admission_wiring as w
from strands import tool
from strands.handlers.callback_handler import null_callback_handler
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.tui.chat import tool_binding_backend as binding

SURROGATE = "\ud800"
LONE = "ordinary neutral text " + SURROGATE + " end"  # str.encode("utf-8") refuses a lone surrogate
PLAIN = "ordinary neutral text"
QUESTION = "ordinary user question"
HOLD_TEXT = "Model input was held at the admission boundary."
ADMISSION_HOLD = "AMLAS_INPUT_HOLD phase=input_admission reason=admission_refused"
RESULT_HOLD = "AMLAS_INPUT_HOLD phase=research_after_result reason=callback_refused"
ALLOWED = ("ALLOW", "NATIVE_NONE")
INVALID = ("UNKNOWN", "INVALID_INPUT")


def strings(value):
    """Every string inside a nested message structure."""
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for item in value.values():
            yield from strings(item)
    elif type(value) in (list, tuple):
        for item in value:
            yield from strings(item)


def carries_surrogate(value):
    return any(SURROGATE in text for text in strings(value))


def turn_with(monkeypatch, caplog, *, guard, where, text):
    """One chat turn through the real backend and SDK with ``text`` placed at one input boundary."""
    if where == "tool_json":

        @tool
        def research_fetch_urls() -> dict:
            """Return ordinary neutral untrusted external information."""
            return {"status": "success", "content": [{"json": {"excerpt": text}}]}

    else:

        @tool
        def research_fetch_urls() -> str:
            """Return ordinary neutral untrusted external information."""
            return text if where == "tool_text" else PLAIN

    client = None
    if guard:
        client = w.ClientFixture(deny=lambda value: False)  # the guard allows everything it is shown
        w.configure_guard(monkeypatch, client)
    else:
        monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
        monkeypatch.delenv("AMLAS_GUARDRAIL_ID", raising=False)
        monkeypatch.delenv("AMLAS_GUARDRAIL_VERSION", raising=False)
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [research_fetch_urls])
    model = w.ModelFixture()
    backend = binding.ToolBindingBackend(Settings(), model=model)
    agents, build = [], backend.build_agent

    def built(**kwargs):
        agents.append(build(**kwargs))
        agents[-1].callback_handler = null_callback_handler
        return agents[-1]

    monkeypatch.setattr(backend, "build_agent", built)
    history = None
    if where == "history":
        history = [
            {"role": "user", "content": [{"text": "ordinary earlier question"}]},
            {"role": "assistant", "content": [{"text": text}]},
        ]
    with caplog.at_level(logging.WARNING):
        turn = backend.chat_with_tools(text if where == "prompt" else QUESTION, history=history)
    admission = turn.input_admission or {}
    return {
        "turn": turn,
        "backend": backend,
        "model": model,
        "agents": agents,
        "status": admission.get("status"),
        "records": [
            (record["boundary"], record["outcome"], record["reason"])
            for record in admission.get("records", [])
        ],
        "guard_texts": [call["content"][0]["text"]["text"] for call in (client.calls if client else [])],
        "hold_lines": [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("AMLAS_INPUT_HOLD")
        ],
    }


# where -> (admission records, model calls, texts shown to the guard, hold lines, agents built)
HELD = {
    "tool_text": (
        [("OWNER_PROMPT", *ALLOWED), ("TOOL_RESULT", *INVALID)],
        1,
        [QUESTION],
        [ADMISSION_HOLD, RESULT_HOLD],
        1,
    ),
    "tool_json": (
        [("OWNER_PROMPT", *ALLOWED), ("TOOL_RESULT", *INVALID)],
        1,
        [QUESTION],
        [ADMISSION_HOLD, RESULT_HOLD],
        1,
    ),
    "history": (
        # The earlier question with its role word is allowed. The reply cannot be encoded, so it is one
        # text, the whole message, and that text is refused.
        [("OWNER_PROMPT", *ALLOWED), *[("HISTORY_TEXT", *ALLOWED)] * 2, ("HISTORY_TEXT", *INVALID)],
        0,
        [QUESTION, "user", "ordinary earlier question"],
        [ADMISSION_HOLD],
        0,
    ),
    "prompt": ([("OWNER_PROMPT", *INVALID)], 0, [], [ADMISSION_HOLD], 0),
}


@pytest.mark.parametrize("where", sorted(HELD))
def test_guard_bound_turn_is_held_and_the_guard_is_never_shown_the_text(monkeypatch, caplog, where):
    seen = turn_with(monkeypatch, caplog, guard=True, where=where, text=LONE)
    records, model_calls, guard_texts, hold_lines, agents_built = HELD[where]
    # What the user sees: the fixed hold text and a held admission.
    assert seen["turn"].final_text == HOLD_TEXT
    assert seen["status"] == "held" and seen["backend"]._input_hold is True
    # The admission records end with the refused text; everything before it was allowed.
    assert seen["records"] == records
    # The guard was shown only the texts before it, never the text that cannot be encoded.
    assert seen["guard_texts"] == guard_texts
    assert not carries_surrogate(seen["guard_texts"])
    # No model call follows the refusal, and the model was never sent the text.
    assert seen["model"].calls == model_calls and len(seen["agents"]) == agents_built
    assert not carries_surrogate(seen["model"].messages)
    assert seen["hold_lines"] == hold_lines
    if where in ("tool_text", "tool_json"):
        # The agent's history keeps a fixed unavailable result in place of the tool's text.
        assert not carries_surrogate(seen["agents"][0].messages)


@pytest.mark.parametrize("where", ["tool_text", "tool_json", "history"])
def test_guard_bound_turn_with_ordinary_text_is_admitted_and_shown_to_the_guard(monkeypatch, caplog, where):
    """The control: the same turn with text that can be encoded reaches the guard and completes."""
    seen = turn_with(monkeypatch, caplog, guard=True, where=where, text=PLAIN)
    assert seen["turn"].final_text == "ordinary complete" and seen["model"].calls == 2
    assert seen["status"] == "admitted" and seen["backend"]._input_hold is False
    assert all(record[1:] == ALLOWED for record in seen["records"])
    assert [record[0] for record in seen["records"]] == (
        ["OWNER_PROMPT", "HISTORY_TEXT", "HISTORY_TEXT", "HISTORY_TEXT", "HISTORY_TEXT", "TOOL_RESULT"]
        if where == "history"
        else ["OWNER_PROMPT", "TOOL_RESULT"]
    )
    assert len(seen["guard_texts"]) == len(seen["records"])
    assert any(PLAIN in text for text in seen["guard_texts"][1:])
    assert seen["hold_lines"] == []


@pytest.mark.parametrize("where", sorted(HELD))
def test_turn_without_a_guard_completes_with_the_text_as_given(monkeypatch, caplog, where):
    seen = turn_with(monkeypatch, caplog, guard=False, where=where, text=LONE)
    # Nothing is assessed, so nothing is held: the scripted model is called twice and its answer is returned.
    assert seen["turn"].final_text == "ordinary complete" and seen["model"].calls == 2
    assert seen["turn"].input_admission is None and seen["backend"]._input_hold is False
    assert seen["records"] == [] and seen["guard_texts"] == [] and seen["hold_lines"] == []
    # The product does not rewrite the text: the model receives it as given.
    assert carries_surrogate(seen["model"].messages)
