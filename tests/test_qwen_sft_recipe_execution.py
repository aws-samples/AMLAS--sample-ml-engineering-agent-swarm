"""Actual tiny native fit recipe execution; full target floors cannot be lowered."""

from __future__ import annotations

import importlib

import pytest
import torch
from test_qwen_sft_native_checkpoint import tiny_model


def module():
    m = importlib.import_module("amlas.training.qwen_sft")
    assert callable(getattr(m, "train_fixed_recipe", None)), "missing-symbol: train_fixed_recipe"
    return m


def test_actual_tiny_fit_runs_fixed_optimizer_steps_and_records_validation():
    m = module()
    base = tiny_model()
    base.config.text_config.vocab_size = 64
    model = m.attach_language_lora(base)
    # Actual model tensors with a bounded already-tokenized synthetic row.
    rows = [{"input_ids": [3, 4, 5, 6, 2], "attention_mask": [1] * 5, "labels": [-100, -100, 5, 6, 2]}]
    result = m.train_fixed_recipe(model, rows, rows, "fit_v1", pad_token_id=0)
    assert result["optimizer_steps"] == 32
    assert result["supervised_tokens"] == 32 * 8 * 3
    assert result["changed_tensor_count"] > 0 and result["validation_history"]
    assert result["status"] == "checkpoint_ready" and result["reload_pending"] is True
    assert result["quality_success"] is False
    assert result["elapsed_seconds"] > 0 and result["tokens_per_second"] > 0
    assert all(torch.isfinite(torch.tensor(item["loss"])) for item in result["validation_history"])


@pytest.mark.parametrize("case", ["empty_train", "empty_validation", "no_supervision", "unknown_recipe"])
def test_incomplete_recipe_inputs_refuse(case):
    m = module()
    model = m.attach_language_lora(tiny_model())
    row = {"input_ids": [3, 4, 2], "attention_mask": [1] * 3, "labels": [-100, 4, 2]}
    training = [row]
    validation = [row]
    identity = "fit_v1"
    if case == "empty_train":
        training = []
    elif case == "empty_validation":
        validation = []
    elif case == "no_supervision":
        row["labels"] = [-100] * 3
    else:
        identity = "caller_authored"
    with pytest.raises(ValueError):
        m.train_fixed_recipe(model, training, validation, identity, pad_token_id=0)


def test_full_recipe_cannot_count_replayed_control_row_as_unique_corpus(monkeypatch):
    m = module()
    model = m.attach_language_lora(tiny_model())
    row = {"input_ids": [3, 4, 2], "attention_mask": [1] * 3, "labels": [-100, 4, 2]}
    monkeypatch.setattr(
        m, "optimizer_update", lambda *args, **kwargs: pytest.fail("full update before unique corpus gate")
    )
    with pytest.raises(ValueError, match=r"unique.*corpus|corpus.*unique"):
        m.train_fixed_recipe(model, [row], [row], "full_v1", pad_token_id=0)
