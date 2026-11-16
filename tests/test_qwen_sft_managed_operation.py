"""Managed fixed operation admission and loader flags; native GPU proof is separate."""

from __future__ import annotations

import importlib

import pytest


def module():
    m = importlib.import_module("amlas.training.qwen_sft")
    for name in ("managed_operation", "load_fixed_gpu_model"):
        assert callable(getattr(m, name, None)), "missing-symbol: " + name
    return m


@pytest.mark.parametrize("case", ["operation", "recipe", "source", "binding", "proof"])
def test_managed_operation_refuses_named_gate_before_model_load(monkeypatch, case):
    m = module()
    monkeypatch.setattr(
        m, "load_fixed_gpu_model", lambda *_: pytest.fail("model loaded before named refusal")
    )
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "a" * 40)
    operation, recipe_id, source = "train", "fit_v1", "a" * 40
    binding = {"schema_version": 1, "catalog_id": "qwen3_5_9b", "catalog_version": "1", "recipe_id": "fit_v1"}
    proof = {
        "res_uid": [10001] * 3,
        "res_gid": [10001] * 3,
        "groups": [],
        "cap_eff": 0,
        "cap_prm": 0,
        "cap_inh": 0,
        "cap_amb": 0,
        "cap_bnd": 0,
        "no_new_privs": 1,
        "fs_uid": 10001,
        "fs_gid": 10001,
        "keep_caps": 0,
        "seccomp": 2,
        "securebits": 0,
    }
    expected = "managed source operation"
    if case == "operation":
        operation = "script"
    elif case == "recipe":
        recipe_id = "auto"
        expected = "recipe"
    elif case == "source":
        source = "b" * 40
    elif case == "binding":
        binding = {"approved": True}
        expected = "managed fixed binding"
    else:
        proof = {"root": True}
        expected = "privilege"
    with pytest.raises(ValueError, match=expected):
        m.managed_operation(
            operation,
            recipe_id,
            source_commit=source,
            stage_binding=binding,
            privilege_proof=proof,
            stage_manifest_sha256="b" * 64,
        )


def test_native_gpu_gate_is_reached_with_valid_fixed_loader(monkeypatch, tmp_path):
    m = module()
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        Qwen3_5ForConditionalGeneration,
        "from_pretrained",
        lambda *_a, **_kw: pytest.fail("model loaded without CUDA"),
    )
    from test_qwen_sft_consumption_boundary import fixture

    _, _, held, _ = fixture(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="BF16 GPU capability"):
            m.load_fixed_gpu_model(held)
    finally:
        held.close()


def test_native_fixed_loader_cannot_use_remote_code_or_pickle(monkeypatch, tmp_path):
    m = module()
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    calls = []

    class Loaded:
        def to(self, device):
            assert device == "cuda:0"
            return self

    def loader(path, **kwargs):
        calls.append((str(path), kwargs))
        return Loaded()

    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "from_pretrained", loader)
    from test_qwen_sft_consumption_boundary import fixture

    _, _, held, _ = fixture(tmp_path, monkeypatch)
    try:
        expected_alias = held.alias
        m.load_fixed_gpu_model(held)
    finally:
        held.close()
    path, kwargs = calls[0]
    assert path == expected_alias
    assert kwargs.get("revision") == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # pragma: allowlist secret
    assert (
        kwargs["local_files_only"] is True
        and kwargs["trust_remote_code"] is False
        and kwargs["use_safetensors"] is True
    )
    assert kwargs["dtype"] is torch.bfloat16 and kwargs["attn_implementation"] == "eager"


def test_production_gpu_loader_refuses_missing_input_authority(monkeypatch):
    m = module()
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    class Loaded:
        def to(self, _):
            return self

    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "from_pretrained", lambda *_a, **_kw: Loaded())
    with pytest.raises(ValueError, match="input authority"):
        m.load_fixed_gpu_model()
