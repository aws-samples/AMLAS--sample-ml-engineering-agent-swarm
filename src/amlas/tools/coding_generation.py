"""Opaque generation tools; SDK context is the only authority channel."""

from __future__ import annotations

from typing import Any

from strands import tool

from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.generation_state import CodingSpecValidationError, SessionCodingGenerationStore
from ..flows.coding_generation_flow import run_coding_generation_flow


def _context(context):
    state = getattr(context, "invocation_state", None)
    state = state if type(state) is dict else {}
    policy, store = state.get("_amlas_approval_policy"), state.get("_amlas_generation_store")
    return {
        "policy": policy if isinstance(policy, ApprovalPolicy) else None,
        "store": store if type(store) is SessionCodingGenerationStore else None,
    }


_NEXT_STEP = {
    "bounded_text_required": "Pass the specification text, at most 64 KiB.",
    "parse_or_schema_invalid": (
        "Call generate_spec_from_description with planning_data and pass its spec_text unchanged; "
        "validate_spec lists the failing paths of a text you edited."
    ),
    "task_type_mismatch": "task_type must be text_generation; take spec_text from generate_spec_from_description.",
    "primary_metric_mismatch": "primary_metric must be pass_at_1.",
    "generation_required": "Add the fixed generation block; generate_spec_from_description with planning_data returns it.",
    "generation_quality_metric_mismatch": "generation.quality.metric must be pass_at_1.",
    "model_pin_required": (
        "generation.model needs repo_id and revision; take them from the spec_text of "
        "generate_spec_from_description (planning_data.model_catalog selects the entry and is not a YAML key)."
    ),
    "tokenizer_pin_mismatch": "generation.tokenizer must equal generation.model.",
    "model_profile_mismatch": "generation.model must be the pinned Qwen/Qwen3.5-9B catalog revision.",
    "recipe_required": "Add generation.recipe with id full_v1 and version 1.",
    "recipe_profile_mismatch": "generation.recipe must be full_v1 version 1.",
}


_BOUND_NEXT_STEP = (
    "Next call resolve_coding_evaluation_spec with case_request_ref and this spec_ref; it returns the resolved "
    "spec_ref (case_bound true) that submit_coding_generation needs. The specification stage ends after that call "
    "succeeds; this result is not an evaluation."
)


@tool(context=True)
def bind_coding_generation_spec(yaml_text: str, tool_context: Any = None) -> dict:
    """Freeze a researched coding specification as planning data in this conversation.

    Pass the spec_text returned by generate_spec_from_description for the fixed Qwen coding task, called
    with planning_data (model_catalog, target, golden_dataset, acceptance, compute_budget or
    unbounded_budget). The spec needs task_type text_generation, primary_metric pass_at_1 and a generation
    block that pins the model and tokenizer and the recipe full_v1 version 1. A refusal returns
    reason_codes, field_paths and a next_step. Pass spec_text unchanged: do not fill generation.preparation or
    generation.evaluation by hand. Success returns spec_ref; resolve it with resolve_coding_evaluation_spec next.

    Args:
        yaml_text: Validated specification frontmatter and notes, at most 64 KiB.
    """
    store = _context(tool_context)["store"]
    if store is None:
        return {"success": False, "status": "denied", "quality_success": False, "native_admission": False}
    try:
        return {"success": True, **store.bind_spec(yaml_text), "next_step": _BOUND_NEXT_STEP}
    except CodingSpecValidationError as error:
        if type(error) is CodingSpecValidationError:
            return {
                "success": False,
                "status": "refused",
                "error_type": "ValueError",
                "validation": error.validation.as_data(),
                "next_step": _NEXT_STEP.get(error.validation.reason_code, _NEXT_STEP["parse_or_schema_invalid"]),
                "quality_success": False,
                "native_admission": False,
            }
        return {
            "success": False, "status": "refused", "error_type": "ValueError",
            "quality_success": False, "native_admission": False,
        }
    except Exception as error:
        return {
            "success": False,
            "status": "refused",
            "error_type": type(error).__name__,
            "quality_success": False,
            "native_admission": False,
        }


@tool(context=True)
def submit_coding_generation(
    model_request_ref: str,
    case_request_ref: str,
    spec_ref: str,
    sft_request_ref: str = "",
    plan_only: bool = False,
    tool_context: Any = None,
) -> dict:
    """Generate fixed164 coding candidates from private original sources.

    Args:
        model_request_ref: Same-conversation opaque acquired model reference.
        case_request_ref: Same-conversation opaque frozen case reference.
        spec_ref: Opaque reference returned by bind_coding_generation_spec.
        sft_request_ref: Original full training reference for tuned generation; empty selects base.
        plan_only: Preview without observing sources or submitting a job.
    """
    return run_coding_generation_flow(
        model_request_ref=model_request_ref,
        case_request_ref=case_request_ref,
        spec_ref=spec_ref,
        sft_request_ref=sft_request_ref,
        plan_only=plan_only,
        **_context(tool_context),
    )


@tool(context=True)
def get_coding_generation(request_ref: str, tool_context: Any = None) -> dict:
    """Poll the original generation job and verify committed output, without resubmission.

    Args:
        request_ref: Same-conversation opaque generation reference.
    """
    return run_coding_generation_flow(operation="observe", request_ref=request_ref, **_context(tool_context))
