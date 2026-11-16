"""Fresh typed caller approval precedes every managed SFT effect/observation."""

from __future__ import annotations

import dataclasses
from typing import Any

from ..backends.types import DatasetArtifactRef
from ..control_plane.approvals import ApprovalPolicy, request_approval
from ..control_plane.provider_disclosure import TRAINING, execution_provider
from ..control_plane.sft_state import SessionSftStore, SftConfigurationMissingError


def _public(value: Any) -> Any:
    if type(value) is DatasetArtifactRef:
        return dataclasses.asdict(value)
    if type(value) is dict:
        return {k: _public(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [_public(v) for v in value]
    return value


def _bounded_training_status(result: dict) -> dict:
    """Project verified histories while keeping their complete exact-version evidence."""
    if result.get("trained") is not True or "metrics" not in result:
        return result
    if type(result.get("training_ref")) is not DatasetArtifactRef:
        raise ValueError("complete typed training evidence required for metric summary")
    metrics = result["metrics"]
    if type(metrics) is not dict:
        raise ValueError("verified training metrics required")
    summary = {"view": "bounded_v1", "evidence_field": "training_ref"}
    for name in ("train", "validation"):
        history = metrics[name + "_history"]
        if type(history) is not list or not history:
            raise ValueError("complete verified metric history required")
        summary[name] = {
            "record_count": len(history),
            "final": dict(history[-1]),
            "best_loss": dict(min(history, key=lambda row: row["loss"])),
        }
    return {
        **result,
        "metrics": {
            key: value for key, value in metrics.items() if key not in {"train_history", "validation_history"}
        },
        "metrics_history_summary": summary,
    }


def run_qwen_sft_flow(
    *,
    operation: str = "train",
    model_request_ref: str = "",
    preparation_request_ref: str = "",
    request_ref: str = "",
    recipe_id: str = "fit_v1",
    plan_only: bool = False,
    policy: ApprovalPolicy | None = None,
    store: SessionSftStore | None = None,
) -> dict:
    stage = "fit" if recipe_id == "fit_v1" else "full"
    plan = {
        "flow_id": "qwen_sft_flow",
        "operation": operation,
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "recipe_id": recipe_id,
        "recipe_version": "1",
        "training_stage": stage,
        "optimizer_steps": 32 if recipe_id == "fit_v1" else 1024,
        "minimum_unique_corpus_examples": 0 if recipe_id == "fit_v1" else 1000,
        "minimum_unique_supervised_tokens": 0 if recipe_id == "fit_v1" else 1000000,
        "minimum_supervised_tokens_processed": 0 if recipe_id == "fit_v1" else 1000000,
        "quality_success": False,
        "steps": [
            "verify private model and preparation versions",
            "conditionally stage exact raw files and RUN",
            "submit fixed isolated native training",
            "verify original native job and exact-version outputs",
        ],
        "cost_estimate": None,
        "cost_uncertainty": "native GPU staging/training has no current quote",
        "lifecycle_recipe": "full_v1",
        "submission_rule": (
            "This conversation keeps one training submission: the first submit_qwen_sft call that is not a "
            "plan_only preview. Poll it with get_qwen_sft; a second submission is not available. Call "
            "submit_qwen_sft alone in its message: a message that mixes it with other tool calls is refused "
            "whole."
        ),
    }
    if recipe_id == "fit_v1":
        plan["recipe_note"] = (
            "fit_v1 is a 32-step pilot that proves fit only. Coding generation and the coding spec accept only "
            "a full_v1 checkpoint, and no second submission is kept. Choose full_v1 now unless the pilot is "
            "the whole goal."
        )
    base = {
        "success": False,
        "accepted": False,
        "trained": False,
        "quality_success": False,
        "production_admission": False,
        "full_training_complete": False,
        "training_stage": stage,
        "executed": False,
        "plan": plan,
        "execution_provider": execution_provider(TRAINING),
    }
    if operation not in ("train", "reload", "observe") or recipe_id not in ("fit_v1", "full_v1"):
        return {**base, "status": "unsupported_recipe"}
    if plan_only:
        return {**base, "status": "planned"}
    if not isinstance(store, SessionSftStore):
        return {**base, "status": "denied"}
    caller = policy if isinstance(policy, ApprovalPolicy) else ApprovalPolicy()
    decision = request_approval(
        action="qwen_sft_flow",
        cost_estimate=None,
        impact="managed GPU SFT or exact-version observation uses selected resources",
        policy=caller,
    )
    if decision["decision"] != "approved":
        return {**base, "status": "denied", "decision": decision}
    try:
        result = (
            store.submit(model_request_ref, preparation_request_ref, recipe_id)
            if operation == "train"
            else store.reload(request_ref)
            if operation == "reload"
            else store.observe(request_ref)
        )
        result = _bounded_training_status(result)
        return _public(
            {
                **base,
                **result,
                "executed": operation != "observe" and result.get("accepted") is True,
                "decision": decision,
            }
        )
    except SftConfigurationMissingError as error:
        return {**base, "status": "configuration_missing", "missing_selectors": list(error.missing_selectors)}
    except Exception as error:
        return {**base, "status": "refused", "error_type": type(error).__name__}
