"""The two size-diagnostic swallows use contextlib.suppress, not `except Exception: pass`.

bandit B110 (try/except/pass) is an Info finding of the security scanner. The behaviour stays: a failing logger or
callback never changes the capture decision.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from amlas.control_plane import normal_run_checkpoint
from amlas.tui.chat import tool_binding_backend as binding

RUN_ID = "ef" * 16


def functions(path, name):
    return [
        node
        for node in ast.walk(ast.parse(Path(path).read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]


def bare_pass_handlers(function):
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.ExceptHandler) and all(isinstance(stmt, ast.Pass) for stmt in node.body)
    ]


@pytest.mark.parametrize(
    ("path", "name"),
    [
        (binding.__file__, "_warn_normal_sft_capture_size"),
        (normal_run_checkpoint.__file__, "report_size"),
        (binding.__file__, "_warn_normal_conversation_size"),
        (binding.__file__, "_warn_normal_unknown_poll"),
        (binding.__file__, "_warn_model_call_retry"),
    ],
)
def test_new_and_earlier_size_diagnostics_have_no_try_except_pass(path, name):
    found = functions(path, name)
    assert len(found) == 1, f"{name} not found once in {path}"
    assert bare_pass_handlers(found[0]) == []


def test_logger_failure_still_cannot_change_the_size_report(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("ordinary diagnostic logger unavailable")

    monkeypatch.setattr(binding.logger, "warning", refuse)
    assert binding._warn_normal_sft_capture_size(RUN_ID, 10, "ok") is None


def test_size_report_is_still_logged(caplog):
    caplog.set_level(logging.INFO, logger=binding.__name__)
    binding._warn_normal_sft_capture_size(RUN_ID, 10, "ok")
    assert [r.getMessage() for r in caplog.records if r.name == binding.__name__] == [
        f"normal_sft_capture_size run_id={RUN_ID} bytes=10 limit={160 * 1024} outcome=ok"
    ]


def test_logger_failure_does_not_swallow_process_control(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(binding.logger, "warning", interrupted)
    with pytest.raises(KeyboardInterrupt):
        binding._warn_normal_sft_capture_size(RUN_ID, 10, "ok")


def test_failing_size_callback_cannot_change_the_causal_call_refusal():
    def refuse(count, outcome):
        raise RuntimeError("ordinary size callback unavailable")

    prefix = [{"role": "user", "content": [{"text": "neutral"}]}]
    with pytest.raises(ValueError, match="assistant SFT batch"):
        normal_run_checkpoint._causal_call("tooluse_abcdefghijklmnop", prefix, _diagnostic_size=refuse)


def test_failing_size_callback_does_not_swallow_process_control():
    def interrupted(count, outcome):
        raise KeyboardInterrupt()

    prefix = [{"role": "user", "content": [{"text": "neutral"}]}]
    with pytest.raises(KeyboardInterrupt):
        normal_run_checkpoint._causal_call("tooluse_abcdefghijklmnop", prefix, _diagnostic_size=interrupted)
