"""Reproduced Astra trainer semantics and fixed-recipe checkpoint defects."""

from __future__ import annotations

import hashlib
import json
import struct

import pytest
import torch
from test_qwen_sft_native_checkpoint import module, tiny_model


@pytest.mark.parametrize("content", [None, "answer"])
def test_historical_reasoning_is_explicitly_refused(content):
    m = module()
    value = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": content, "reasoning_content": "preserve this reasoning"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "final answer"},
        ],
        "tools": [],
    }
    with pytest.raises(ValueError, match="historical_reasoning_unsupported"):
        m.normalize_row(value)


def test_lora_initial_state_is_seeded_before_creation():
    m = module()
    first = tiny_model()
    torch.manual_seed(9876)
    a = m.attach_language_lora(first)
    second = tiny_model()
    torch.manual_seed(4567)
    b = m.attach_language_lora(second)
    left = {name: p.detach() for name, p in a.named_parameters() if p.requires_grad}
    right = {name: p.detach() for name, p in b.named_parameters() if p.requires_grad}
    assert left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def test_full_unique_floor_counts_consumed_canonical_examples_not_original_variants(monkeypatch):
    m = module()
    model = m.attach_language_lora(tiny_model())
    monkeypatch.setattr(
        m,
        "optimizer_update",
        lambda *_: pytest.fail("optimizer reached with only250 unique consumed examples"),
    )
    rows = []
    for index in range(1000):
        canonical = index % 250
        ids = [3 + canonical // 60, 3 + canonical % 60] + [3] * 4094
        rows.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * 4096,
                "labels": [-100, *ids[1:]],
                "original_row_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            }
        )
    with pytest.raises(ValueError, match=r"unique.*corpus"):
        m.train_fixed_recipe(model, rows, [rows[0]], "full_v1", pad_token_id=0)


def save(tmp_path, model=None):
    m = module()
    output = tmp_path / "adapter"
    output.mkdir()
    if model is None:
        model = m.attach_language_lora(tiny_model())
    m.save_adapter_checkpoint(
        model,
        output,
        {
            "recipe_id": "fit_v1",
            "optimizer_steps": 0,
            "supervised_tokens": 0,
            "greedy_input_ids": [[3]],
            "greedy_output_ids": [[3, 2]],
        },
    )
    return m, output


def test_reload_requires_complete_source_target_inventory(tmp_path):
    from peft import LoraConfig, get_peft_model

    m = module()
    base = tiny_model()
    target = next(
        name
        for name, child in base.named_modules()
        if name.startswith("model.language_model.layers.") and name.endswith(".q_proj")
    )
    subset = get_peft_model(
        base,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=[target], task_type="CAUSAL_LM"
        ),
    )
    m, output = save(tmp_path, subset)
    with pytest.raises(ValueError):
        m.reload_adapter_checkpoint(tiny_model(), output)


def test_reload_probe_tokens_use_actual_model_vocabulary(tmp_path):
    m, output = save(tmp_path)
    path = output / "CHECKPOINT.json"
    value = json.loads(path.read_bytes())
    value["greedy_input_ids"] = [[64]]
    value["greedy_output_ids"] = [[64, 2]]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        m.reload_adapter_checkpoint(tiny_model(), output)


@pytest.mark.parametrize("case", ["wrong_shape", "oversized_header"])
def test_safe_header_refuses_before_tensor_materialization(tmp_path, monkeypatch, case):
    m, output = save(tmp_path)
    path = output / "adapter_model.safetensors"
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + length])
    receipt_path = output / "CHECKPOINT.json"
    receipt = json.loads(receipt_path.read_bytes())
    if case == "wrong_shape":
        names = [name for name in header if name.endswith(".lora_A.weight")]
        first = names[0]
        second = next(name for name in names if header[name]["shape"] != header[first]["shape"])
        header[first], header[second] = header[second], header[first]
        receipt["tensor_keys"][first]["shape"] = header[first]["shape"]
        receipt["tensor_keys"][second]["shape"] = header[second]["shape"]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        assert len(encoded) <= length
        changed = raw[:8] + encoded + b" " * (length - len(encoded)) + raw[8 + length :]
    else:
        changed = struct.pack("<Q", 2097153) + raw[8:]
    path.write_bytes(changed)
    receipt["files"][path.name] = {"sha256": hashlib.sha256(changed).hexdigest(), "byte_count": len(changed)}
    receipt_path.write_text(json.dumps(receipt))
    import safetensors.torch

    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda *_a, **_kw: pytest.fail("unqualified tensor materialization")
    )
    with pytest.raises(
        ValueError, match="native adapter header keyshape" if case == "wrong_shape" else "header bound"
    ):
        m.reload_adapter_checkpoint(tiny_model(), output)


def test_managed_checkpoint_ref_matches_metadata_consumed_by_reload(tmp_path):
    m, output = save(tmp_path)
    raw = (output / "CHECKPOINT.json").read_bytes()
    reference = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    value = json.loads(raw)
    value["optimizer_steps"] = 1
    (output / "CHECKPOINT.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        m.reload_adapter_checkpoint(tiny_model(), output, expected_ref=reference)
