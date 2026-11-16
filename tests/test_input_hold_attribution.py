"""Every place that latches the invocation INPUT hold names its closed phase literal.

The static test reads the source: each `_input_hold = True` is followed, in the same function, by a
`_warn_input_hold("AMLAS_INPUT_HOLD ...")` call whose literal is in the closed set (a literal outside
the set is dropped without a trace). The behaviour tests drive the sites that were silent before.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.tui.chat import engine as engine_module
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
LOGGER = "amlas.tui.chat.tool_binding_backend"
PREFIX = "AMLAS_INPUT_HOLD "
SHAPE = re.compile(r"^AMLAS_INPUT_HOLD phase=[a-z_]+ reason=[a-z_]+$")
SOURCES = (Path(binding.__file__), Path(engine_module.__file__))
BLOCK_FIELDS = ("body", "orelse", "finalbody")


def parse(path):
    tree = ast.parse(path.read_text())
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent
    return tree


def is_latch(node):
    return (
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        and any(isinstance(t, ast.Attribute) and t.attr == "_input_hold" for t in node.targets)
    )


def warn_literals(node):
    found = []
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "_warn_input_hold":
            found += [
                c.value
                for c in ast.walk(call)
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.startswith(PREFIX)
            ]
    return found


def siblings_after(node):
    """Statements from `node` onward in each enclosing block, innermost first, up to its function."""
    child = node
    while not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
        parent = child.parent
        for field in BLOCK_FIELDS:
            block = getattr(parent, field, None)
            if isinstance(block, list) and child in block:
                yield block[block.index(child) :]
        for handler in getattr(parent, "handlers", []):
            if child in handler.body:
                yield handler.body[handler.body.index(child) :]
        child = parent


def latch_sites():
    for path in SOURCES:
        for node in ast.walk(parse(path)):
            if is_latch(node):
                yield path.name, node.lineno, node


def test_every_latch_site_is_followed_by_a_closed_phase_literal():
    sites = list(latch_sites())
    assert len([s for s in sites if s[0] == "tool_binding_backend.py"]) >= 25
    silent = []
    for name, line, node in sites:
        literals = [lit for block in siblings_after(node) for stmt in block for lit in warn_literals(stmt)]
        if not literals:
            silent.append(f"{name}:{line}")
    assert silent == [], f"hold latch without a phase literal: {silent}"


def test_every_literal_in_a_warn_call_is_in_the_closed_set_and_every_member_is_used():
    used = set()
    for path in SOURCES:
        used.update(lit for node in ast.walk(parse(path)) for lit in warn_literals(node))
    assert used <= binding._INPUT_HOLD_WARNINGS, sorted(used - binding._INPUT_HOLD_WARNINGS)
    assert binding._INPUT_HOLD_WARNINGS <= used, sorted(binding._INPUT_HOLD_WARNINGS - used)
    assert all(SHAPE.fullmatch(lit) for lit in binding._INPUT_HOLD_WARNINGS)


def hold_records(caplog):
    return [r for r in caplog.records if r.name == LOGGER and r.getMessage().startswith(PREFIX)]


def backend():
    return ToolBindingBackend(Settings())


def boom(*args, **kwargs):
    raise RuntimeError("EXCLUDED_ERROR_TEXT_SENTINEL")


def expect_one(caplog, literal):
    records = hold_records(caplog)
    assert [r.getMessage() for r in records] == [literal]
    assert not records[0].args and records[0].exc_info is None


def test_research_binding_not_unused_names_its_phase(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    target = backend()
    target._research_binding = {"phase": "ACTIVE"}
    with pytest.raises(binding._InputAdmissionHeldError):
        target._activate_research_binding()
    assert target._input_hold is True
    expect_one(caplog, "AMLAS_INPUT_HOLD phase=research_activation reason=binding_not_unused")


def test_failed_input_admission_names_its_phase_once(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    target = backend()
    monkeypatch.setattr(target, "_selected_input_checker", boom)
    with pytest.raises(binding._InputAdmissionHeldError):
        target._admit_input("neutral text", "OWNER_PROMPT")
    with pytest.raises(binding._InputAdmissionHeldError):
        target._admit_input("neutral text", "OWNER_PROMPT")
    expect_one(caplog, "AMLAS_INPUT_HOLD phase=input_admission reason=admission_refused")


@pytest.mark.parametrize(
    ("state", "incoming", "reason"),
    [
        ({"request_state": {}}, None, "invocation_state_refused"),
        ("not a mapping", None, "invocation_state_refused"),
        ({}, "not a list", "incoming_messages_refused"),
    ],
)
def test_pause_start_refusals_name_their_reason(monkeypatch, caplog, state, incoming, reason):
    caplog.set_level(logging.INFO, logger=LOGGER)
    target = backend()
    monkeypatch.setattr(target, "_pause_profile_current", lambda: object())
    event = SimpleNamespace(invocation_state=state, messages=incoming, agent=SimpleNamespace(messages=[]))
    with pytest.raises(binding._InputAdmissionHeldError):
        target._start_pending_pause(event)
    assert target._input_hold is True
    expect_one(caplog, f"AMLAS_INPUT_HOLD phase=pause_start reason={reason}")


def test_history_checker_selection_failure_names_its_phase(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    target = backend()
    monkeypatch.setattr(target, "_admit_input", lambda text, boundary, **kwargs: text)
    monkeypatch.setattr(target, "_selected_input_checker", boom)
    turn = target._chat_with_tools_once("neutral prompt", history=[])
    assert target._input_hold is True and turn.input_admission is not None
    expect_one(caplog, "AMLAS_INPUT_HOLD phase=history_qualification reason=checker_selection_failed")


def test_unserializable_history_names_its_phase(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    target = backend()
    monkeypatch.setattr(target, "_admit_input", lambda text, boundary, **kwargs: text)
    monkeypatch.setattr(target, "_selected_input_checker", lambda: object())
    history = [{"role": "user", "content": [{"text": float("nan")}]}]
    turn = target._chat_with_tools_once("neutral prompt", history=history)
    assert target._input_hold is True and turn.input_admission is not None
    expect_one(caplog, "AMLAS_INPUT_HOLD phase=history_qualification reason=history_not_serializable")


def scripted_turn(monkeypatch, target):
    from sdk_scripted_model import ScriptedModel

    target._model = ScriptedModel([("text", "neutral answer")])
    original = target.build_agent

    def build(*args, **kwargs):
        agent = original(*args, **kwargs)
        agent.callback_handler = lambda *a, **k: None
        return agent

    monkeypatch.setattr(target, "build_agent", build)
    return target._chat_with_tools_once("neutral prompt")


def test_research_binding_refusal_after_the_run_names_its_phase(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    target = backend()

    def refuse_after_the_model_ran(*args, **kwargs):
        # Building the agent also qualifies the (absent) binding; only the check after the run refuses.
        if getattr(target._model, "calls", 0):
            raise RuntimeError("EXCLUDED_ERROR_TEXT_SENTINEL")

    monkeypatch.setattr(target, "_qualify_research_binding", refuse_after_the_model_ran)
    turn = scripted_turn(monkeypatch, target)
    assert target._input_hold is True, turn.final_text
    assert turn.input_admission is not None
    expect_one(caplog, "AMLAS_INPUT_HOLD phase=research_after_run reason=binding_refused")


def test_pause_result_failure_names_its_phase(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    target = backend()
    monkeypatch.setattr(target, "_pending_pause_result", boom)
    turn = scripted_turn(monkeypatch, target)
    assert target._input_hold is True and turn.input_admission is not None
    expect_one(caplog, "AMLAS_INPUT_HOLD phase=pause_result reason=pause_result_refused")


@pytest.mark.parametrize(("marked", "reason"), [(False, "unmarked_stop"), (True, "stop_state_mismatch")])
def test_pause_result_latches_name_their_reason(monkeypatch, caplog, marked, reason):
    caplog.set_level(logging.INFO, logger=LOGGER)
    target = backend()
    agent = SimpleNamespace(messages=[])
    witness = binding._PendingBatchPause(agent, {}, {"stop_event_loop": True}, None, None, os.getpid(), "[]")
    witness.marked = marked
    target._pending_batch_pause = witness
    monkeypatch.setattr(target, "_extract_guardrail_outcome", lambda result: None)
    monkeypatch.setattr(target, "_qualify_pending_pause", lambda current: None)
    with pytest.raises(binding._InputAdmissionHeldError):
        target._pending_pause_result(agent, object())
    assert target._input_hold is True
    expect_one(caplog, f"AMLAS_INPUT_HOLD phase=pause_result reason={reason}")


def test_refused_completion_admission_names_its_phase(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    target = backend()
    monkeypatch.setattr(target, "_qualify_precommit_original", boom)
    monkeypatch.setattr(binding, "_warn_precommit_handoff", lambda *args, **kwargs: None)
    with pytest.raises(binding._InputAdmissionHeldError):
        target._admit_precommit_original(
            "neutral", SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), None
        )
    assert target._input_hold is True
    expect_one(caplog, "AMLAS_INPUT_HOLD phase=pending_bookkeeping reason=completion_admission_refused")


def test_the_literal_helper_still_rejects_text_outside_the_closed_set(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    binding._warn_input_hold("AMLAS_INPUT_HOLD phase=made_up reason=not_in_the_set")
    assert hold_records(caplog) == []
