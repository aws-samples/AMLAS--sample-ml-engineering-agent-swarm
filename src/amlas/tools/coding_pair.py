"""Opaque paired-evaluation tools; model input never supplies policy or native selectors."""

from __future__ import annotations

from typing import Any

from strands import tool

from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.generation_state import SessionCodingGenerationStore
from ..control_plane.paired_state import SessionPairedCodingStore
from ..flows.coding_pair_flow import run_coding_pair_flow


def _context(context):
    state = getattr(context, "invocation_state", None)
    state = state if type(state) is dict else {}
    policy, generation, store = (
        state.get(name) for name in ("_amlas_approval_policy", "_amlas_generation_store", "_amlas_pair_store")
    )
    return {
        "policy": policy if isinstance(policy, ApprovalPolicy) else None,
        "generation": generation if type(generation) is SessionCodingGenerationStore else None,
        "store": store if type(store) is SessionPairedCodingStore else None,
    }


@tool(context=True)
def resolve_coding_evaluation_spec(
    case_request_ref: str, spec_ref: str, plan_only: bool = False, tool_context: Any = None
) -> dict:
    """Resolve a new immutable case-bound spec before both generation arms.

    Args:
        case_request_ref: Original same-conversation frozen case reference.
        spec_ref: Original researched specification reference.
        plan_only: Preview without observing private cases.
    """
    return run_coding_pair_flow(
        operation="resolve",
        case_ref=case_request_ref,
        spec_ref=spec_ref,
        plan_only=plan_only,
        **_context(tool_context),
    )


@tool(context=True)
def submit_coding_pair(
    case_request_ref: str,
    base_generation_ref: str,
    tuned_generation_ref: str,
    resolved_spec_ref: str,
    plan_only: bool = False,
    tool_context: Any = None,
) -> dict:
    """Submit fixed AWS evaluation from original base/tuned outputs and private cases.

    Args:
        case_request_ref: Original same-conversation frozen case reference.
        base_generation_ref: Original complete base generation reference.
        tuned_generation_ref: Original complete tuned generation reference.
        resolved_spec_ref: Immutable reference returned by resolve_coding_evaluation_spec.
        plan_only: Preview without reading sources or dispatching a job.
    """
    return run_coding_pair_flow(
        case_ref=case_request_ref,
        base_ref=base_generation_ref,
        tuned_ref=tuned_generation_ref,
        spec_ref=resolved_spec_ref,
        plan_only=plan_only,
        **_context(tool_context),
    )


@tool(context=True)
def get_coding_pair(request_ref: str, tool_context: Any = None) -> dict:
    """Observe the original paired evaluation; never create replacement sessions/jobs.

    Args:
        request_ref: Original same-conversation opaque paired evaluation reference.
    """
    return run_coding_pair_flow(operation="observe", request_ref=request_ref, **_context(tool_context))
