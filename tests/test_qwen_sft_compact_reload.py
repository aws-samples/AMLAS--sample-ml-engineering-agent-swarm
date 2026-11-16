"""Actual PEFT target compaction must retain the complete fixed reload inventory."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch
from peft import get_peft_model_state_dict
from peft.utils.constants import MIN_TARGET_MODULES_FOR_OPTIMIZATION
from test_qwen_sft_native_checkpoint import module, tiny_model
from transformers import Qwen3_5ForConditionalGeneration


def compact_model():
    """Synthetic native model crosses the pinned PEFT optimization threshold."""
    config = copy.deepcopy(tiny_model().config)
    config.text_config.num_hidden_layers = 4
    config.text_config.layer_types = ["linear_attention", "full_attention"] * 2
    torch.manual_seed(12345)
    return Qwen3_5ForConditionalGeneration(config).float()


def test_real_peft_compacted_target_checkpoint_reloads_complete_language_state(tmp_path, monkeypatch):
    m = module()
    base = compact_model()
    targets = {
        name
        for name, child in base.named_modules()
        if name.startswith("model.language_model.layers.")
        and name.rsplit(".", 1)[-1] in m.LANGUAGE_LINEAR_SUFFIXES
        and type(child) is torch.nn.Linear
    }
    assert len(targets) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION
    model = m.attach_language_lora(base)
    original = {name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(model).items()}
    output = tmp_path / "adapter"
    output.mkdir()
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
    config = json.loads((output / "adapter_config.json").read_bytes())
    assert set(config["target_modules"]) == m.LANGUAGE_LINEAR_SUFFIXES
    import peft

    native_attach = peft.get_peft_model
    received = []

    def record_attach(base, configuration):
        received.append(set(configuration.target_modules))
        return native_attach(base, configuration)

    monkeypatch.setattr(peft, "get_peft_model", record_attach)
    restored = m.reload_adapter_checkpoint(compact_model(), output)
    assert received == [targets]
    actual = get_peft_model_state_dict(restored)
    assert set(actual) == set(original)
    assert all(torch.equal(actual[name], original[name]) for name in original)
    assert all("language_model.layers." in name for name in actual)
    assert all(not parameter.requires_grad for parameter in restored.parameters())


def save_compacted(tmp_path):
    m = module()
    output = tmp_path / "adapter"
    output.mkdir()
    m.save_adapter_checkpoint(
        m.attach_language_lora(compact_model()),
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


@pytest.mark.parametrize("case", ["missing", "extra", "mixed", "duplicate", "vision", "partial_path"])
def test_compacted_targets_refuse_incomplete_or_foreign_selectors(tmp_path, monkeypatch, case):
    m, output = save_compacted(tmp_path)
    path = output / "adapter_config.json"
    config = json.loads(path.read_bytes())
    targets = config["target_modules"]
    if case == "missing":
        targets.pop()
    elif case == "extra":
        targets.append("unrecognized_linear")
    elif case == "mixed":
        targets[0] = "model.language_model.layers.0.mlp.down_proj"
    elif case == "duplicate":
        targets.append(targets[0])
    elif case == "vision":
        targets[0] = "model.visual.q_proj"
    else:
        config["target_modules"] = ["model.language_model.layers.0.mlp.down_proj"]
    raw = json.dumps(config).encode()
    path.write_bytes(raw)
    receipt_path = output / "CHECKPOINT.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["files"][path.name] = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    receipt_path.write_text(json.dumps(receipt))
    import peft
    import safetensors.torch

    monkeypatch.setattr(peft, "get_peft_model", lambda *_: pytest.fail("unqualified target attachment"))
    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda *_a, **_kw: pytest.fail("unqualified tensor materialization")
    )
    with pytest.raises(ValueError):
        m.reload_adapter_checkpoint(compact_model(), output)


def test_compacted_targets_cannot_expand_into_matching_vision_suffix(tmp_path):
    m, output = save_compacted(tmp_path)
    base = compact_model()
    base.model.visual.q_proj = torch.nn.Linear(32, 32, bias=False)
    original = base.model.visual.q_proj.weight.detach().clone()
    restored = m.reload_adapter_checkpoint(base, output)
    assert torch.equal(base.model.visual.q_proj.weight, original)
    assert not base.model.visual.q_proj.weight.requires_grad
    assert all("language_model.layers." in name for name in get_peft_model_state_dict(restored))


def test_compacted_targets_still_require_complete_native_tensor_inventory(tmp_path, monkeypatch):
    m, output = save_compacted(tmp_path)
    import safetensors.torch

    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda *_a, **_kw: pytest.fail("unqualified tensor materialization")
    )
    with pytest.raises(ValueError, match="native adapter header keyshape"):
        m.reload_adapter_checkpoint(tiny_model(), output)


def test_trained_threshold_compaction_preserves_actual_values_and_greedy_reload(tmp_path):
    m = module()
    model = m.attach_language_lora(compact_model())
    before = {name: value.detach().clone() for name, value in get_peft_model_state_dict(model).items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    batch = {
        "input_ids": torch.tensor([[3, 4, 5, 6, 7, 2]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 5, 6, 7, 2]]),
    }
    result = m.optimizer_update(model, optimizer, [batch])
    actual = get_peft_model_state_dict(model)
    assert result["optimizer_steps"] == 1 and result["supervised_tokens"] == 4
    assert any(not torch.equal(actual[name], before[name]) for name in actual)
    model.eval()
    query = torch.tensor([[3, 4, 5]])
    expected = model.generate(input_ids=query, max_new_tokens=3, do_sample=False).tolist()
    output = tmp_path / "trained-adapter"
    output.mkdir()
    m.save_adapter_checkpoint(
        model,
        output,
        {
            "recipe_id": "fit_v1",
            "optimizer_steps": 1,
            "supervised_tokens": 4,
            "greedy_input_ids": query.tolist(),
            "greedy_output_ids": expected,
        },
    )
    config = json.loads((output / "adapter_config.json").read_bytes())
    assert set(config["target_modules"]) == m.LANGUAGE_LINEAR_SUFFIXES
    restored = m.reload_adapter_checkpoint(compact_model(), output)
    restored_state = get_peft_model_state_dict(restored)
    assert set(restored_state) == set(actual)
    assert all(torch.equal(restored_state[name], actual[name]) for name in actual)
    assert restored.generate(input_ids=query, max_new_tokens=3, do_sample=False).tolist() == expected
    import os
    import subprocess
    import sys
    from pathlib import Path

    test_directory = Path(__file__).resolve().parent
    child_environment = dict(os.environ)
    child_paths = [str(test_directory.parent / "src"), str(test_directory)]
    inherited_paths = child_environment.get("PYTHONPATH")
    if inherited_paths:
        child_paths.append(inherited_paths)
    child_environment["PYTHONPATH"] = os.pathsep.join(child_paths)
    script = tmp_path / "fresh_trained_reload.py"
    script.write_text(
        "import hashlib, json, sys, torch\n"
        "from test_qwen_sft_compact_reload import compact_model\n"
        "from amlas.training.qwen_sft import reload_adapter_checkpoint\n"
        "from peft import get_peft_model_state_dict\n"
        "model = reload_adapter_checkpoint(compact_model(), sys.argv[1])\n"
        "state = get_peft_model_state_dict(model)\n"
        "print(json.dumps({'greedy': model.generate(input_ids=torch.tensor([[3,4,5]]), "
        "max_new_tokens=3, do_sample=False).tolist(), 'state': "
        "{name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() "
        "for name, value in state.items()}}))\n"
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(output)],
        env=child_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr[-4096:]
    fresh = json.loads(completed.stdout.splitlines()[-1])
    assert fresh["greedy"] == expected
    assert fresh["state"] == {
        name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        for name, value in actual.items()
    }
