"""Real tiny native Qwen/PEFT backward, safe checkpoint and fresh reload contracts."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration


def module():
    value = importlib.import_module("amlas.training.qwen_sft")
    for name in (
        "attach_language_lora",
        "optimizer_update",
        "save_adapter_checkpoint",
        "reload_adapter_checkpoint",
    ):
        assert callable(getattr(value, name, None)), "missing-symbol: " + name
    return value


def tiny_model():
    torch.manual_seed(12345)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 128,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 2, 4],
                "partial_rotary_factor": 1.0,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    return Qwen3_5ForConditionalGeneration(config).float()


def test_native_tiny_backward_changes_only_language_lora_and_safe_reload(tmp_path):
    m = module()
    base = tiny_model()
    vision = {
        name: parameter.detach().clone() for name, parameter in base.named_parameters() if ".visual." in name
    }
    model = m.attach_language_lora(base)
    selected = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert selected and all("language_model.layers." in name and "lora_" in name for name in selected)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    batch = {
        "input_ids": torch.tensor([[3, 4, 5, 6, 7, 2]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 5, 6, 7, 2]]),
    }
    result = m.optimizer_update(model, optimizer, [batch])
    assert result["optimizer_steps"] == 1 and result["supervised_tokens"] == 4 and result["loss"] > 0
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in before
    )
    for name, parameter in base.named_parameters():
        if name in vision:
            assert torch.equal(vision[name], parameter)
    model.eval()
    query = torch.tensor([[3, 4, 5]])
    expected = model.generate(input_ids=query, max_new_tokens=3, do_sample=False).tolist()
    out = tmp_path / "adapter"
    out.mkdir()
    receipt = m.save_adapter_checkpoint(
        model,
        out,
        {
            "recipe_id": "fit_v1",
            "optimizer_steps": 1,
            "supervised_tokens": 4,
            "greedy_input_ids": query.tolist(),
            "greedy_output_ids": expected,
        },
    )
    assert receipt["status"] == "checkpoint_saved" and receipt["reload_pending"] is True
    assert receipt["quality_success"] is False
    assert (out / "adapter_model.safetensors").is_file()
    assert not any(p.suffix in {".pt", ".pth", ".pkl", ".bin"} for p in out.iterdir())
    restored = m.reload_adapter_checkpoint(tiny_model(), out)
    restored.eval()
    assert restored.generate(input_ids=query, max_new_tokens=3, do_sample=False).tolist() == expected
    # A separate process constructs the same actual native base and uses fixed reload.
    script = 'from pathlib import Path; import json; from test_qwen_sft_native_checkpoint import tiny_model; from amlas.training.qwen_sft import reload_adapter_checkpoint; import torch; m=reload_adapter_checkpoint(tiny_model(),Path(__import__("sys").argv[1]));m.eval();print(json.dumps(m.generate(input_ids=torch.tensor([[3,4,5]]),max_new_tokens=3,do_sample=False).tolist()))'
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(out)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == expected


@pytest.mark.parametrize("case", ["extra_pickle", "adapter_hash", "config_hash", "unknown_recipe"])
def test_fixed_reload_refuses_unbound_checkpoint(tmp_path, case):
    m = module()
    model = m.attach_language_lora(tiny_model())
    out = tmp_path / "adapter"
    out.mkdir()
    m.save_adapter_checkpoint(
        model,
        out,
        {
            "recipe_id": "fit_v1",
            "optimizer_steps": 0,
            "supervised_tokens": 0,
            "greedy_input_ids": [[3]],
            "greedy_output_ids": [[3, 2]],
        },
    )
    if case == "extra_pickle":
        (out / "optimizer.pt").write_bytes(b"untrusted")
    elif case == "adapter_hash":
        p = out / "adapter_model.safetensors"
        raw = p.read_bytes()
        p.write_bytes(b"!" + raw[1:])
    elif case == "config_hash":
        p = out / "adapter_config.json"
        p.write_bytes(p.read_bytes() + b" ")
    else:
        p = out / "CHECKPOINT.json"
        value = json.loads(p.read_bytes())
        value["recipe_id"] = "model_authored"
        p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        m.reload_adapter_checkpoint(tiny_model(), out)
