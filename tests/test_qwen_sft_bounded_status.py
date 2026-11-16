"""Bounded public SFT status over a declared verified neutral observation.

The actual registered tool, typed SessionSftStore and closed full metrics validator
are used; the observation/evidence bytes are declared DATA, not native S3 proof.
No AWS, model, GPU, guard transport or training submission executes here.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import pytest
from test_qwen_sft_backend import metrics
from test_qwen_sft_binding import allowed, context, real_store

from amlas.backends.adapters._util import raw_tool
from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.backends.types import DatasetArtifactRef
from amlas.tools.qwen_sft import get_qwen_sft
from amlas.tui.chat.input_admission import MAX_INPUT_BYTES


def _encoded(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()


def _declared_completed_observation(monkeypatch, recipe="full_v1"):
    store, native, objects, *_ = real_store(monkeypatch)
    actual = metrics(recipe)
    actual["train_history"][-1]["loss"] = 1.25
    actual["validation_history"][-1]["loss"] = 1.25
    SageMakerQwenSftBackend._metrics(actual, recipe, maximum_training_rows=1000)
    evidence = _encoded(
        {
            "metrics": actual,
            "input_summary": {
                "test_used_for_training": False,
                "preparation_commit_sha256": "7" * 64,
            },
            "quality_success": False,
        }
    )
    training_ref = DatasetArtifactRef(
        "s3://neutral-sft-evidence/owned/TRAINING.json",
        "neutral-training-version-1",
        hashlib.sha256(evidence).hexdigest(),
        len(evidence),
        "json",
    )
    checkpoint_ref = DatasetArtifactRef(
        "s3://neutral-sft-evidence/owned/CHECKPOINT.json",
        "neutral-checkpoint-version-1",
        "8" * 64,
        123,
        "json",
    )
    ref = "sft_" + "3" * 32
    result = {
        "success": True,
        "accepted": True,
        "trained": True,
        "status": "checkpoint_saved",
        "request_ref": ref,
        "job_name": "neutral-original-sft",
        "job_arn": "arn:aws:sagemaker:us-west-2:111122223333:training-job/neutral-original-sft",
        "operation": "train",
        "recipe_id": recipe,
        "training_stage": "full" if recipe == "full_v1" else "fit",
        "full_training_complete": recipe == "full_v1",
        "quality_success": False,
        "production_admission": False,
        "reload_pending": True,
        "durability_status": "verified",
        "metrics": actual,
        "training_ref": training_ref,
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_refs": {"CHECKPOINT.json": checkpoint_ref},
        "commit_ref": dataclasses.replace(
            checkpoint_ref, uri="s3://neutral-sft-evidence/owned/COMMITTED.json"
        ),
    }
    reads = []

    def observe(selected):
        assert selected == ref
        reads.append(selected)
        return result

    monkeypatch.setattr(store, "observe", observe)
    return store, native, objects, result, evidence, reads


@pytest.mark.parametrize("recipe", ["full_v1", "fit_v1"])
def test_registered_status_is_bounded_and_keeps_exact_training_evidence(monkeypatch, recipe):
    store, native, objects, declared, evidence, reads = _declared_completed_observation(monkeypatch, recipe)
    before = copy.deepcopy(declared)
    complete_metrics = copy.deepcopy(declared["metrics"])
    if recipe == "full_v1":
        assert len(_encoded(complete_metrics)) > MAX_INPUT_BYTES, (
            "legitimate full history must reproduce INPUT_LIMIT"
        )
    answer = raw_tool(get_qwen_sft)(
        request_ref=declared["request_ref"], tool_context=context(store, allowed())
    )
    assert len(_encoded(answer)) < 16384, (
        "public status must leave headroom below unchanged INPUT admission bound"
    )
    assert answer["metrics"] == {
        key: value
        for key, value in complete_metrics.items()
        if key not in {"train_history", "validation_history"}
    }
    summary = answer["metrics_history_summary"]
    assert summary["view"] == "bounded_v1" and summary["evidence_field"] == "training_ref"
    for name in ("train", "validation"):
        history = complete_metrics[name + "_history"]
        assert summary[name] == {
            "record_count": len(history),
            "final": history[-1],
            "best_loss": min(history, key=lambda row: row["loss"]),
        }
    for key in (
        "success",
        "accepted",
        "trained",
        "status",
        "request_ref",
        "job_name",
        "job_arn",
        "operation",
        "recipe_id",
        "training_stage",
        "full_training_complete",
        "quality_success",
        "production_admission",
        "reload_pending",
        "durability_status",
    ):
        assert answer[key] == declared[key]
    for key in ("training_ref", "checkpoint_ref", "commit_ref"):
        assert answer[key] == dataclasses.asdict(declared[key])
    assert answer["checkpoint_refs"] == {
        name: dataclasses.asdict(value) for name, value in declared["checkpoint_refs"].items()
    }
    assert answer["plan"]["catalog_id"] == "qwen3_5_9b"
    assert answer["plan"]["catalog_version"] == "1"
    assert answer["executed"] is False
    assert json.loads(evidence)["metrics"] == complete_metrics
    assert json.loads(evidence)["input_summary"]["test_used_for_training"] is False
    assert hashlib.sha256(evidence).hexdigest() == answer["training_ref"]["sha256"]
    assert len(evidence) == answer["training_ref"]["byte_count"]
    assert declared == before and reads == [declared["request_ref"]]
    assert native.calls == objects.mutations == []


def test_status_summary_does_not_replace_missing_full_evidence(monkeypatch):
    store, native, objects, declared, _evidence, reads = _declared_completed_observation(monkeypatch)
    declared.pop("training_ref")
    answer = raw_tool(get_qwen_sft)(
        request_ref=declared["request_ref"], tool_context=context(store, allowed())
    )
    assert answer["success"] is False and answer["trained"] is False
    assert answer["full_training_complete"] is False and answer["quality_success"] is False
    assert answer["status"] == "refused" and "metrics" not in answer
    assert reads == [declared["request_ref"]] and native.calls == objects.mutations == []
