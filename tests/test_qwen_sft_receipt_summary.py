"""Bounded receipt summary follows complete private replay; no native proof."""

import copy
import hashlib
import json
import struct

import pytest

from amlas.data_eng.commitpack_preparation import COVERAGE, SUPERVISION_KIND
from amlas.sagemaker_entrypoints import qwen_sft_train as worker
from amlas.training import qwen_sft as core

DOMAIN = b"amlas.commitpack.canonical_sha256_split.v1"


def replay_result(large=False):
    splits = (
        {"train": 26319, "validation": 3283, "test": 3458}
        if large
        else {"train": 3, "validation": 1, "test": 1}
    )
    retained = sum(splits.values())
    index = 1
    digests = {}
    for name, count in splits.items():
        digests[name] = [format(value, "064x") for value in range(index, index + count)]
        index += count
    return {
        "counts": {
            "source": 56025 if large else retained,
            "normalized": 33595 if large else retained,
            "normalization_rejected": 22430 if large else 0,
            "excluded_origins": 0,
            "included_origins": 33595 if large else retained,
            "retained_representatives": retained,
            "duplicate_origins": 535 if large else 0,
            "split_counts": splits,
        },
        "canonical_sha256_by_split": digests,
        "supervision_kind": SUPERVISION_KIND,
        "coverage": dict(COVERAGE),
        "exclusion_policy_sha256": "a" * 64,
    }


def expected_digest(name, values):
    return hashlib.sha256(
        DOMAIN
        + b"\0"
        + name.encode("ascii")
        + b"\0"
        + struct.pack(">Q", len(values))
        + b"".join(bytes.fromhex(value) for value in values)
    ).hexdigest()


def test_actual_33060_shape_fits_unchanged_receipt_bound_after_compaction():
    full = replay_result(large=True)
    raw = (
        json.dumps(
            {"input_summary": {"commitpack": full}}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()
    assert len(raw) > 2215119
    with pytest.raises(ValueError):
        core._checkpoint_json({"input_summary": {"commitpack": full}})
    original = copy.deepcopy(full)
    summary = core._compact_commitpack_replay(full)
    assert full == original, "private validated replay must not be truncated or mutated"
    assert "canonical_sha256_by_split" not in summary
    assert len(core._checkpoint_json({"input_summary": {"commitpack": summary}})) < 8192
    assert summary["counts"] == full["counts"] and summary["coverage"] == full["coverage"]
    for name, values in full["canonical_sha256_by_split"].items():
        assert summary["splits"][name] == {
            "record_count": len(values),
            "ordered_canonical_sha256_sha256": expected_digest(name, values),
        }


def test_digest_binds_domain_split_count_and_original_order():
    full = replay_result()
    first = core._compact_commitpack_replay(full)
    full["canonical_sha256_by_split"]["train"].reverse()
    second = core._compact_commitpack_replay(full)
    assert first["splits"]["train"] != second["splits"]["train"]
    assert first["splits"]["validation"] == second["splits"]["validation"]
    assert expected_digest("train", ["b" * 64]) != expected_digest("test", ["b" * 64])


@pytest.mark.parametrize(
    "case",
    ["boolean_count", "count_drift", "list_short", "malformed_digest", "extra", "coverage", "exclusion"],
)
def test_compaction_refuses_malformed_raw_replay(case):
    full = replay_result()
    if case == "boolean_count":
        full["counts"]["source"] = True
    elif case == "count_drift":
        full["counts"]["included_origins"] += 1
    elif case == "list_short":
        full["canonical_sha256_by_split"]["train"].pop()
    elif case == "malformed_digest":
        full["canonical_sha256_by_split"]["train"][0] = "z" * 64
    elif case == "extra":
        full["ready"] = True
    elif case == "coverage":
        full["coverage"]["unknown"] = True
    else:
        full["exclusion_policy_sha256"] = "z" * 64
    with pytest.raises(ValueError):
        core._compact_commitpack_replay(full)


@pytest.mark.parametrize("case", ["version", "domain", "count", "hash", "extra", "source_counts", "coverage"])
def test_closed_summary_validator_refuses_drift(case):
    summary = core._compact_commitpack_replay(replay_result())
    if case == "version":
        summary["schema_version"] = True
    elif case == "domain":
        summary["digest_domain"] = "caller-domain"
    elif case == "count":
        summary["splits"]["train"]["record_count"] += 1
    elif case == "hash":
        summary["splits"]["train"]["ordered_canonical_sha256_sha256"] = "z" * 64
    elif case == "extra":
        summary["approved"] = True
    elif case == "source_counts":
        summary["counts"]["source"] += 1
    else:
        summary["coverage"]["unknown"] = True
    with pytest.raises(ValueError):
        core._validate_commitpack_summary(summary)


@pytest.mark.parametrize("stage", ["training_receipt", "final_readback", "final_publication"])
def test_post_checkpoint_failures_have_closed_stage_without_content(stage):
    def fail():
        raise ValueError("synthetic-private-value")

    with pytest.raises(core.FixedStageError) as found:
        core.run_fixed_stage(stage, fail)
    value = json.loads(worker._format_refusal(found.value))
    assert value["stage"] == stage and value["cause_type"] == "ValueError"
    assert "synthetic-private" not in json.dumps(value)


@pytest.mark.parametrize("failure", ["training_receipt", "final_readback", "final_publication"])
def test_actual_managed_postsave_boundary_is_wrapped(tmp_path, monkeypatch, failure):
    import os
    import sys
    from types import SimpleNamespace

    from amlas.data_eng import inspection_schema

    class Held:
        alias = "/proc/self/fd/123"

        def __init__(self, *_args):
            pass

        def verify(self):
            if failure == "final_readback":
                raise ValueError("synthetic-private-value")

        def close(self):
            pass

    class Query:
        def tolist(self):
            return [[1]]

    model = SimpleNamespace(
        eval=lambda: None, generate=lambda **_kw: SimpleNamespace(tolist=lambda: [[1, 2]])
    )
    fake_torch = SimpleNamespace(
        __version__="2.14.0",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
        long=object(),
        tensor=lambda *_a, **_k: Query(),
    )
    fake_tokenizer = SimpleNamespace(from_pretrained=lambda *_a, **_k: SimpleNamespace(pad_token_id=0))
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(inspection_schema, "validate_native_privilege_proof", lambda *_a: None)
    monkeypatch.setattr(core, "validate_staged_inputs", lambda *_a: {"test_used_for_training": False})
    monkeypatch.setattr(core, "_HeldNativeInputs", Held)
    monkeypatch.setattr(core, "_fixed_output_fd", lambda: 3)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=fake_tokenizer))
    monkeypatch.setattr(
        core, "_open_directory_original", lambda *_a: os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    )
    monkeypatch.setattr(core, "_load_split_encoded", lambda *_a: [{"input_ids": [1, 2]}])
    monkeypatch.setattr(core, "load_fixed_gpu_model", lambda *_a: model)
    monkeypatch.setattr(core, "attach_language_lora", lambda value: value)
    monkeypatch.setattr(
        core,
        "train_fixed_recipe",
        lambda *_a, **_k: {"target_met": True, "optimizer_steps": 32, "supervised_tokens": 61847},
    )
    monkeypatch.setattr(core, "save_adapter_checkpoint", lambda *_a, **_k: {"files": {}})
    calls = []

    def publish(_fd, name, _value):
        calls.append(name)
        if (failure == "training_receipt" and name == "TRAINING.json") or (
            failure == "final_publication" and name == "COMMITTED.json"
        ):
            raise ValueError("synthetic-private-value")
        return {"sha256": "b" * 64, "byte_count": 1}

    monkeypatch.setattr(core, "_publish_fixed_json", publish)
    with pytest.raises(core.FixedStageError) as found:
        core.managed_operation(
            "train",
            "fit_v1",
            source_commit="a" * 40,
            privilege_proof={},
            stage_binding={
                "catalog_id": "qwen3_5_9b",
                "catalog_version": "1",
                "recipe_id": "fit_v1",
                "model_files": [],
                "split_refs": {"train": {}, "validation": {}},
            },
            stage_manifest_sha256="b" * 64,
        )
    value = json.loads(worker._format_refusal(found.value))
    assert value["stage"] == failure and "synthetic-private" not in json.dumps(value)
    assert calls == (
        ["TRAINING.json", "COMMITTED.json"] if failure == "final_publication" else ["TRAINING.json"]
    )
