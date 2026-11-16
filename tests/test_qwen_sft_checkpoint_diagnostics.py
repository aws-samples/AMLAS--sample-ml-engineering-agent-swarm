"""Source-managed scratch collision and closed checkpoint diagnostics; no native claim."""

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from amlas.sagemaker_entrypoints import qwen_sft_train as worker
from amlas.training import qwen_sft as core

EVIDENCE = {
    "recipe_id": "fit_v1",
    "optimizer_steps": 0,
    "supervised_tokens": 0,
    "greedy_input_ids": [[3]],
    "greedy_output_ids": [[3, 2]],
}


def format_failure(function):
    with pytest.raises(core.FixedStageError) as found:
        core.run_fixed_stage("checkpoint_save", function)
    result = json.loads(worker._format_refusal(found.value))
    assert result["status"] == "fixed_qwen_operation_refused"
    assert result["stage"] == "checkpoint_save"
    assert "private-content" not in json.dumps(result)
    return result


def test_actual_managed_tmpdir_collision_refuses_before_framework_save(tmp_path, monkeypatch):
    source = (Path(__file__).parents[1] / "processing_bootstrap/src/main.rs").read_text()
    selected = re.search(r'"TMPDIR",\s*if profile == WorkerProfile::QwenSft \{\s*"([^"]+)"', source)
    assert selected is not None
    source_tmpdir = selected.group(1)
    assert source_tmpdir == "/home/appuser/.cache/amlas-qwen-sft"
    fixed_output = "/opt/ml/model"
    output = tmp_path / "model"
    output.mkdir()
    # Recreate the exact source-selected namespace relationship without native mounts.
    scratch = output if source_tmpdir == fixed_output else tmp_path / "separate-fixed-scratch"
    scratch.mkdir(exist_ok=True)
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", None)
    directory = tempfile.mkdtemp()
    assert Path(directory).parent == scratch
    called = []

    def forbidden_save(*_args, **_kwargs):
        called.append(True)
        raise ValueError("private-content")

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *_a, **_k: None))
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if source_tmpdir == fixed_output:
            result = format_failure(
                lambda: core.save_adapter_checkpoint(
                    SimpleNamespace(save_pretrained=forbidden_save), output, EVIDENCE, output_fd=fd
                )
            )
            assert called == []
            assert result.get("checkpoint_substage") == "output_empty"
        else:
            result = format_failure(
                lambda: core.save_adapter_checkpoint(
                    SimpleNamespace(save_pretrained=forbidden_save), output, EVIDENCE, output_fd=fd
                )
            )
            assert result.get("checkpoint_substage") == "framework_save"
            assert called == [True]
        assert os.fstat(fd).st_ino == output.stat().st_ino
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("evidence", "evidence"),
        ("nonempty", "output_empty"),
        ("framework", "framework_save"),
        ("inventory", "file_inventory"),
    ],
)
def test_closed_checkpoint_substage_never_echoes_exception(tmp_path, monkeypatch, case, expected):
    output = tmp_path / "model"
    output.mkdir()
    evidence = dict(EVIDENCE)
    if case == "evidence":
        evidence["optimizer_steps"] = True
    if case == "nonempty":
        (output / "synthetic-scratch").mkdir()

    def save(*_args, **_kwargs):
        if case == "framework":
            raise ValueError("private-content")

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *_a, **_k: None))
    result = format_failure(
        lambda: core.save_adapter_checkpoint(SimpleNamespace(save_pretrained=save), output, evidence)
    )
    assert result.get("checkpoint_substage") == expected
    assert result.get("checkpoint_reason") == expected + "_refused"
    assert result["cause_type"] == "ValueError"


def test_checkpoint_diagnostic_unknown_fields_are_not_public_output():
    error = core.FixedStageError("checkpoint_save", "ValueError")
    error.checkpoint_substage = "private-content"
    error.checkpoint_reason = "private-content"
    result = json.loads(worker._format_refusal(error))
    assert "checkpoint_substage" not in result and "private-content" not in json.dumps(result)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("authority", "output_authority"),
        ("card", "model_card"),
        ("readback", "file_readback"),
        ("header", "tensor_header"),
        ("schema", "checkpoint_schema"),
        ("commit", "receipt_commit"),
    ],
)
def test_remaining_checkpoint_failures_keep_closed_stage(tmp_path, monkeypatch, case, expected):
    output = tmp_path / "model"
    output.mkdir()
    if case == "authority":
        output.rmdir()

    class Header:
        def __enter__(self):
            if case == "header":
                raise ValueError("private-content")
            return self

        def __exit__(self, *_args):
            return False

        def keys(self):
            return ["base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"]

        def get_slice(self, _name):
            return SimpleNamespace(
                get_shape=lambda: [8 if case == "schema" else 16, 32], get_dtype=lambda: "F32"
            )

    def save(alias, **kwargs):
        assert kwargs == {"safe_serialization": True, "save_embedding_layers": False}
        destination = Path(alias)
        (destination / "adapter_model.safetensors").write_bytes(
            b"" if case == "readback" else b"synthetic-inert-header"
        )
        (destination / "adapter_config.json").write_bytes(b"{}")
        if case == "card":
            (destination / "README.md").symlink_to(destination / "adapter_config.json")

    if case == "commit":
        monkeypatch.setattr(os, "fsync", lambda *_a: (_ for _ in ()).throw(ValueError("private-content")))
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *_a, **_k: Header()))
    result = format_failure(
        lambda: core.save_adapter_checkpoint(SimpleNamespace(save_pretrained=save), output, EVIDENCE)
    )
    assert result.get("checkpoint_substage") == expected
    assert result.get("checkpoint_reason") == expected + "_refused"


def test_historical_output_tmpdir_collision_is_characterization_only(tmp_path, monkeypatch):
    output = tmp_path / "model"
    output.mkdir()
    monkeypatch.setenv("TMPDIR", str(output))
    monkeypatch.setattr(tempfile, "tempdir", None)
    tempfile.mkdtemp()
    called = []
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *_a, **_k: None))
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        result = format_failure(
            lambda: core.save_adapter_checkpoint(
                SimpleNamespace(save_pretrained=lambda *_a, **_k: called.append(1)),
                output,
                EVIDENCE,
                output_fd=fd,
            )
        )
        assert called == []
        assert result["checkpoint_substage"] == "output_empty"
    finally:
        os.close(fd)
