"""Opaque-reference inspection tools; model input carries no native authority."""

from __future__ import annotations

from typing import Any

from strands import tool

from ..control_plane.acquisition_state import SessionAcquisitionStore
from ..control_plane.approvals import ApprovalPolicy
from ..flows.dataset_inspection_flow import run_dataset_inspection_flow


@tool(context=True)
def inspect_acquired_dataset(
    acquisition_ref: str,
    plan_only: bool = False,
    target_column: str | None = None,
    tool_context: Any = None,
) -> dict[str, Any]:
    """Submit fixed isolated inspection of one verified acquired artifact.

    No raw URI, role, image, source override or native handle is accepted.
    Submission is pending inspection; poll get_dataset_inspection.

    Args:
        acquisition_ref: Opaque acquisition reference from this conversation.
        plan_only: Preview only, without staging bytes or submitting a job.
        target_column: Optional bounded literal column for sampled counts.
    """
    state = getattr(tool_context, "invocation_state", None)
    state = state if isinstance(state, dict) else {}
    policy = state.get("_amlas_approval_policy")
    store = state.get("_amlas_acquisition_store")
    return run_dataset_inspection_flow(
        acquisition_ref=acquisition_ref,
        plan_only=plan_only,
        target_column=target_column,
        policy=policy if isinstance(policy, ApprovalPolicy) else None,
        store=store if isinstance(store, SessionAcquisitionStore) else None,
    )


@tool(context=True)
def get_dataset_inspection(inspection_ref: str, tool_context: Any = None) -> dict[str, Any]:
    """Read a bounded verified inspection by an opaque conversation reference.

    Args:
        inspection_ref: Opaque reference returned by inspect_acquired_dataset.
    """
    state = getattr(tool_context, "invocation_state", None)
    store = state.get("_amlas_acquisition_store") if isinstance(state, dict) else None
    if not isinstance(store, SessionAcquisitionStore):
        return {
            "success": False,
            "accepted": False,
            "inspected": False,
            "status": "unknown_reference",
            "prepared": False,
            "trained": False,
        }
    return store.observe_inspection(inspection_ref)
