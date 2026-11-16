"""Producer provenance is distinct from repaired consumer source; no GPU effects."""

from __future__ import annotations

import hashlib

import pytest
from test_qwen_sft_compact_reload import compact_model
from test_qwen_sft_native_checkpoint import module


class BaseInputsReachedError(RuntimeError):
    """Neutral stop after original metadata checks and before any ML/GPU load."""


def staged_checkpoint(tmp_path, monkeypatch, producer):
    m = module()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    metadata = m.save_adapter_checkpoint(
        m.attach_language_lora(compact_model()),
        adapter,
        {
            "recipe_id": "fit_v1",
            "optimizer_steps": 0,
            "supervised_tokens": 0,
            "greedy_input_ids": [[3]],
            "greedy_output_ids": [[3, 2]],
            "source_commit": producer,
            "stage_manifest_sha256": "d" * 64,
        },
    )
    assert metadata["source_commit"] == producer
    base = tmp_path / "base"
    base.mkdir()
    inventory = m.expected_model_weight_manifest("qwen3_5_9b", "1")["files"]
    for item in inventory:
        (base / item["path"]).write_bytes(b"synthetic-placeholder")
    original_open = m._open_directory_original
    monkeypatch.setattr(
        m,
        "_open_directory_original",
        lambda path: original_open(base if str(path) == m.FIXED_BASE else adapter),
    )
    original_read = m._read_exact_relative
    base_reads = []

    def read(directory, name, reference, limit, collect):
        if name == "CHECKPOINT.json":
            return original_read(directory, name, reference, limit, collect)
        base_reads.append(name)
        assert name in {item["path"] for item in inventory} and collect is False
        return None

    monkeypatch.setattr(m, "_read_exact_relative", read)
    raw = (adapter / "CHECKPOINT.json").read_bytes()
    binding = {
        "schema_version": 1,
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "recipe_id": "fit_v1",
        "model_files": inventory,
        "checkpoint_ref": {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)},
    }
    return m, binding, base_reads, inventory


@pytest.mark.parametrize("mode", ["legacy_same", "legacy_cross", "qualified_v2"])
def test_managed_reload_retains_original_producer_until_explicit_compatibility_gate(
    tmp_path, monkeypatch, mode
):
    consumer = "a" * 40
    producer = consumer if mode == "legacy_same" else "b" * 40
    m, binding, base_reads, inventory = staged_checkpoint(tmp_path, monkeypatch, producer)
    if mode == "qualified_v2":
        # Closed data-only native RUN. Fresh server-side compatibility selection
        # and native authority remain separate backend integration requirements.
        binding.update(
            schema_version=2,
            operation="reload",
            source_commit=consumer,
            checkpoint_source_commit=producer,
            recipe_version="1",
            reload_contract_version="qwen3_5_9b.safe_adapter_reload.v1",
            compatibility_digest="e" * 64,
        )
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", consumer)
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

    def stop(*_):
        raise BaseInputsReachedError()

    monkeypatch.setattr(m, "_HeldNativeInputs", stop)
    monkeypatch.setattr(m, "load_fixed_gpu_model", lambda *_: pytest.fail("GPU model loading reached"))

    def run():
        return m.managed_operation(
            "reload",
            "fit_v1",
            source_commit=consumer,
            stage_binding=binding,
            privilege_proof=proof,
            stage_manifest_sha256="c" * 64,
        )

    if mode == "legacy_cross":
        with pytest.raises(ValueError, match="reload recipe binding"):
            run()
        return
    with pytest.raises(m.FixedStageError) as found:
        run()
    assert found.value.stage == "base_input_verify"
    assert found.value.cause_type == "BaseInputsReachedError"
    assert base_reads == [item["path"] for item in inventory]
