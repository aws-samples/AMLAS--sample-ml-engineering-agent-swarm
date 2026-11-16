"""Numbers-only SFT capture size diagnostic: closed labels, no content, no change to any refusal."""

from __future__ import annotations

import inspect
import logging

import pytest

from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import tool_binding_backend as binding

MESSAGE = "normal_sft_capture_size run_id=%s bytes=%d limit=%d outcome=%s"


def _prefix(text):
    return [
        {"role": "user", "content": [{"text": text}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "ordinary-use",
                        "name": "submit_qwen_sft",
                        "input": {
                            "model_request_ref": "model_acq_" + "1" * 32,
                            "preparation_request_ref": "prep_" + "2" * 32,
                        },
                    }
                }
            ],
        },
    ]


def test_causal_call_reports_the_exact_canonical_size_inside_the_cap():
    seen = []
    prefix = _prefix("ordinary history")
    call = checkpoint._causal_call(
        "ordinary-use", prefix, _diagnostic_size=lambda count, outcome: seen.append((count, outcome))
    )
    expected = len(
        checkpoint._canonical({"pending_tool_use_id": "ordinary-use", "causal_prefix": prefix}).encode(
            "utf-8"
        )
    )
    assert seen == [(expected, "ok")]
    assert call["tool_use"]["name"] == "submit_qwen_sft"


def test_causal_call_reports_the_size_over_the_cap_and_still_refuses():
    seen = []
    with pytest.raises(ValueError, match="size exceeded"):
        checkpoint._causal_call(
            "ordinary-use",
            _prefix("ordinary history " + "x" * (170 * 1024)),
            _diagnostic_size=lambda count, outcome: seen.append((count, outcome)),
        )
    assert len(seen) == 1
    count, outcome = seen[0]
    assert outcome == "over" and checkpoint._MAX_BYTES < count < 200 * 1024


def test_size_callback_failure_changes_neither_refusal_nor_success():
    def unavailable(count, outcome):
        raise RuntimeError("withheld diagnostic failure")

    with pytest.raises(ValueError, match="size exceeded"):
        checkpoint._causal_call("ordinary-use", _prefix("x" * (170 * 1024)), _diagnostic_size=unavailable)
    call = checkpoint._causal_call("ordinary-use", _prefix("ordinary"), _diagnostic_size=unavailable)
    assert call["pending_tool_use_id"] == "ordinary-use"


def test_non_finite_number_refuses_without_a_size_report():
    prefix = _prefix("ordinary")
    prefix[0]["content"][0]["score"] = float("nan")
    seen = []
    with pytest.raises(ValueError):
        checkpoint._causal_call(
            "ordinary-use", prefix, _diagnostic_size=lambda count, outcome: seen.append((count, outcome))
        )
    assert seen == []


def test_default_call_without_a_size_callback_is_unchanged():
    assert (
        checkpoint._causal_call("ordinary-use", _prefix("ordinary"))["tool_use"]["name"] == "submit_qwen_sft"
    )


def test_closed_size_warning_validates_every_label_and_number(caplog):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    binding._warn_normal_sft_capture_size("a" * 32, 123456, "over")
    binding._warn_normal_sft_capture_size("a" * 31, 1, "ok")
    binding._warn_normal_sft_capture_size("A" * 32, 1, "ok")
    binding._warn_normal_sft_capture_size("a" * 32, True, "ok")
    binding._warn_normal_sft_capture_size("a" * 32, -1, "ok")
    binding._warn_normal_sft_capture_size("a" * 32, 10**9, "ok")
    binding._warn_normal_sft_capture_size("a" * 32, 5.0, "ok")
    binding._warn_normal_sft_capture_size("a" * 32, 5, "other")
    binding._warn_normal_sft_capture_size(None, 5, "ok")
    records = [record for record in caplog.records if record.msg == MESSAGE]
    assert len(records) == 1
    assert records[0].args == ("a" * 32, 123456, checkpoint._MAX_BYTES, "over")
    assert records[0].exc_info is None and records[0].stack_info is None


def test_size_warning_failure_does_not_change_process_control(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("withheld diagnostic failure")

    monkeypatch.setattr(binding.logger, "warning", unavailable)
    assert binding._warn_normal_sft_capture_size("a" * 32, 7, "ok") is None
    control = KeyboardInterrupt()

    def stop(*args, **kwargs):
        raise control

    monkeypatch.setattr(binding.logger, "warning", stop)
    with pytest.raises(KeyboardInterrupt) as observed:
        binding._warn_normal_sft_capture_size("a" * 32, 7, "ok")
    assert observed.value is control


def test_size_callback_is_plumbed_through_capture_and_the_tool_hook():
    assert "_diagnostic_size" in inspect.signature(checkpoint.NormalRunContext.capture_sft_call).parameters
    assert (
        "_diagnostic_size"
        in inspect.signature(binding.ToolBindingBackend._capture_normal_sft_call).parameters
    )
