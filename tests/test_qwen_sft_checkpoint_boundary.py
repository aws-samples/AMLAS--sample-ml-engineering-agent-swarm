"""Checkpoint schema and retained-file integrity regressions on real native adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_qwen_sft_native_checkpoint import module, tiny_model


def checkpoint(tmp_path):
    m = module()
    output = tmp_path / "adapter"
    output.mkdir()
    m.save_adapter_checkpoint(
        m.attach_language_lora(tiny_model()),
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


@pytest.mark.parametrize(
    "case",
    ["extra_grant", "forged_catalog", "boolean_counter", "negative_id", "overlong_generation", "nonfinite"],
)
def test_checkpoint_closed_metadata_refuses_before_tensor_loading(tmp_path, case):
    m, output = checkpoint(tmp_path)
    path = output / "CHECKPOINT.json"
    value = json.loads(path.read_bytes())
    if case == "extra_grant":
        value["approved"] = True
    elif case == "forged_catalog":
        value["catalog_id"] = "arbitrary_model"
    elif case == "boolean_counter":
        value["optimizer_steps"] = True
    elif case == "negative_id":
        value["greedy_input_ids"] = [[-1]]
    elif case == "overlong_generation":
        value["greedy_output_ids"] = [[3] + [2] * 17]
    else:
        value["supervised_tokens"] = float("nan")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        m.reload_adapter_checkpoint(tiny_model(), output)


def test_checkpoint_evidence_cannot_override_source_owned_identity(tmp_path):
    m = module()
    output = tmp_path / "adapter"
    output.mkdir()
    with pytest.raises(ValueError):
        m.save_adapter_checkpoint(
            m.attach_language_lora(tiny_model()),
            output,
            {
                "recipe_id": "fit_v1",
                "optimizer_steps": 0,
                "supervised_tokens": 0,
                "greedy_input_ids": [[3]],
                "greedy_output_ids": [[3, 2]],
                "catalog_id": "arbitrary_model",
            },
        )
    assert list(output.iterdir()) == []


def test_checkpoint_original_ancestor_symlink_refused(tmp_path):
    m, output = checkpoint(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        m.reload_adapter_checkpoint(tiny_model(), alias / output.name)


def test_checkpoint_verified_metadata_is_not_reopened_by_path(tmp_path, monkeypatch):
    m, output = checkpoint(tmp_path)
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("verified metadata reopened by path"))
    m.reload_adapter_checkpoint(tiny_model(), output)


def test_checkpoint_same_inode_publication_mutation_refuses(tmp_path, monkeypatch):
    m = module()
    output = tmp_path / "adapter"
    output.mkdir()
    original = os.fsync
    mutated = []

    def fsync(fd):
        original(fd)
        name = os.readlink("/proc/self/fd/" + str(fd))
        if name.endswith("CHECKPOINT.json"):
            with open(name, "r+b", buffering=0) as stream:
                stream.seek(0)
                stream.write(b"!")
            mutated.append(True)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(ValueError):
        m.save_adapter_checkpoint(
            m.attach_language_lora(tiny_model()),
            output,
            {
                "recipe_id": "fit_v1",
                "optimizer_steps": 0,
                "supervised_tokens": 0,
                "greedy_input_ids": [[3]],
                "greedy_output_ids": [[3, 2]],
            },
        )
    assert mutated
