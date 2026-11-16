"""Caller-authorized fixed managed inspection in the collection approval domain."""

from __future__ import annotations

from typing import Any

from ..backends.ports import DataInspectionBackend
from ..backends.registry import REGISTRY
from ..control_plane.acquisition_state import SessionAcquisitionStore
from ..control_plane.approvals import ApprovalPolicy, request_approval


def _resolve_backend() -> Any:
    return REGISTRY.resolve(DataInspectionBackend)


def run_dataset_inspection_flow(
    *,
    acquisition_ref: str,
    plan_only: bool = False,
    target_column: str | None = None,
    policy: ApprovalPolicy | None = None,
    store: SessionAcquisitionStore | None = None,
) -> dict[str, Any]:
    plan = {
        "operation": "fixed_managed_dataset_inspection",
        "cost_estimate": None,
        "acquisition_ref": acquisition_ref,
        "target_column": target_column,
        "prepared": False,
        "trained": False,
    }
    base = {
        "success": False,
        "approved": False,
        "executed": False,
        "accepted": False,
        "inspected": False,
        "prepared": False,
        "trained": False,
        "plan": plan,
    }
    if plan_only:
        return {**base, "status": "planned"}
    if not isinstance(store, SessionAcquisitionStore):
        return {**base, "status": "denied", "detail": "Trusted conversation state is required."}
    decision = request_approval(
        action="data_collection_flow",
        cost_estimate=None,
        impact="fixed isolated inspection; native resource quote unavailable",
        policy=policy if isinstance(policy, ApprovalPolicy) else ApprovalPolicy(),
    )
    if decision["decision"] != "approved":
        return {**base, "status": "denied", "decision": decision}
    try:
        source = store._verified_inspection_source(acquisition_ref)
        selected = _resolve_backend()
        if selected.adapter_name != "processing":
            raise ValueError("inspection requires the fixed managed adapter")
        handle = selected.adapter.submit(source=source, target_column=target_column)
        reference = store.add_inspection(selected.adapter, handle)
        return {
            **base,
            "approved": True,
            "executed": True,
            "accepted": True,
            "status": "submitted",
            "inspection_ref": reference,
            "provider": handle.backend,
            "poll_after_seconds": 15,
        }
    except Exception as error:
        return {
            **base,
            "approved": True,
            "status": "failed",
            "error_type": type(error).__name__,
            "detail": "Inspection submission failed; no local parsing fallback was attempted.",
        }
