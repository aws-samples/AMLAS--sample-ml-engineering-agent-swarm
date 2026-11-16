"""Hash-first fixed staging/manifest admission, synthetic bytes only."""

from __future__ import annotations

import hashlib
import importlib
import json

import pytest


def module():
    value = importlib.import_module("amlas.training.qwen_sft")
    assert callable(getattr(value, "validate_staged_inputs", None)), "missing-symbol: validate_staged_inputs"
    return value


def fixture(monkeypatch, tmp_path):
    from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest

    m = module()
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    base = tmp_path / "base"
    base.mkdir()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for item in expected["files"]:
        raw = ("synthetic " + item["path"]).encode()
        item["sha256"] = hashlib.sha256(raw).hexdigest()
        item["byte_count"] = len(raw)
        (base / item["path"]).write_bytes(raw)
    monkeypatch.setattr(m, "expected_model_weight_manifest", lambda *args: expected)
    files = {}
    row = {
        "messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}],
        "tools": [],
    }
    for name in [
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "membership.jsonl",
        "lineage.jsonl",
        "REPORT.json",
    ]:
        raw = (json.dumps(row) + "\n").encode() if name.endswith(".jsonl") else b"{}\n"
        (prepared / name).write_bytes(raw)
        files[name] = {
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "format": "jsonl" if name.endswith(".jsonl") else "json",
            "record_count": 1,
        }
    header = {
        "job_name": "fixture-prep",
        "runtime_session_id": "fixture-session",
        "deployment_id": "fd0d82ab-4e03-44c0-9bf1-908af5b2b031",
        "deployment_prefix": "fixture",
        "image_uri": "fixture-image@sha256:" + "a" * 64,
        "source_commit": "b" * 40,
        "profile_id": "openai.explicit_conversations.v1",
        "policy_version": "coding-preparation.v1",
    }
    committed = {
        **header,
        "privilege_profile": "sagemaker_native_ceiling_v1",
        "selection_sha256": "c" * 64,
        "exclusions_sha256": "d" * 64,
        "coverage": {},
        "guest_evidence": {},
        "schema_version": 2,
        "profile_id": "openai.explicit_conversations.v1",
        "policy_version": "coding-preparation.v1",
        "files": files,
        "counts": {"split_counts": {"train": 1, "validation": 1, "test": 1}},
    }
    raw = (json.dumps(committed) + "\n").encode()
    (prepared / "COMMITTED.json").write_bytes(raw)
    binding = {
        "schema_version": 1,
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "recipe_id": "fit_v1",
        "model_files": expected["files"],
        "preparation_header": header,
        "preparation_commit": {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)},
        "split_refs": {name: files[name + ".jsonl"] for name in ["train", "validation", "test"]},
    }
    return m, base, prepared, binding


def test_fixed_staging_hashes_before_parsing(monkeypatch, tmp_path):
    m, base, prepared, binding = fixture(monkeypatch, tmp_path)
    result = m.validate_staged_inputs(binding, base, prepared)
    assert result["train_record_count"] == 1 and result["validation_record_count"] == 1
    assert result["catalog_id"] == "qwen3_5_9b" and result["trusted"] is False
    assert result["test_used_for_training"] is False


@pytest.mark.parametrize(
    "case",
    [
        "base_hash",
        "base_extra",
        "base_link",
        "parent_link",
        "commit_hash",
        "split_hash",
        "empty_split",
        "count_mismatch",
        "grant",
        "unknown_catalog",
        "unknown_recipe",
        "duplicate_json",
    ],
)
def test_unbound_or_mutated_inputs_refuse_before_framework(monkeypatch, tmp_path, case):
    m, base, prepared, binding = fixture(monkeypatch, tmp_path)
    if case == "base_hash":
        p = base / binding["model_files"][0]["path"]
        p.write_bytes(b"!" + p.read_bytes()[1:])
    elif case == "base_extra":
        (base / "pytorch_model.bin").write_bytes(b"pickle")
    elif case == "base_link":
        p = base / binding["model_files"][0]["path"]
        target = tmp_path / "target"
        p.rename(target)
        p.symlink_to(target)
    elif case == "parent_link":
        original = base
        base = tmp_path / "alias"
        base.symlink_to(original, target_is_directory=True)
    elif case == "commit_hash":
        (prepared / "COMMITTED.json").write_bytes(b"{}")
    elif case == "split_hash":
        p = prepared / "train.jsonl"
        p.write_bytes(b"!" + p.read_bytes()[1:])
    elif case == "empty_split":
        (prepared / "validation.jsonl").write_bytes(b"")
    elif case == "count_mismatch":
        binding["split_refs"]["train"]["record_count"] = 2
    elif case == "grant":
        binding["approved"] = True
    elif case == "unknown_catalog":
        binding["catalog_id"] = "model_authored"
    elif case == "unknown_recipe":
        binding["recipe_id"] = "auto"
    else:
        raw = b'{"schema_version":2,"schema_version":2}'
        (prepared / "COMMITTED.json").write_bytes(raw)
        binding["preparation_commit"] = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    with pytest.raises(ValueError):
        m.validate_staged_inputs(binding, base, prepared)
