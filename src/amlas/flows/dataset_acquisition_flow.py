"""Approval-gated data collection acquisition through the backend port."""

from __future__ import annotations

from typing import Any

from ..backends.ports import DataAcquisitionBackend
from ..backends.registry import REGISTRY
from ..control_plane.acquisition_state import SessionAcquisitionStore
from ..control_plane.approvals import ApprovalPolicy, request_approval
from ..control_plane.plan_mode import preview_dataset_acquisition_plan
from ..control_plane.provider_disclosure import PROCESSING, execution_provider


def _resolve_backend() -> Any:
    return REGISTRY.resolve(DataAcquisitionBackend)


def run_dataset_acquisition_flow(
    *, repo_id: str, revision: str, filename: str,
    output_dir: str | None = None, plan_only: bool = False,
    policy: ApprovalPolicy | None = None, store: SessionAcquisitionStore | None = None,
) -> dict[str, Any]:
    """Preview or submit one authorized acquisition; retained refs support polling."""
    plan = preview_dataset_acquisition_plan({
        "repo_id": repo_id, "revision": revision, "filename": filename,
        **({"output_dir": output_dir} if output_dir is not None else {}),
    })
    base = {
        "success": False, "approved": False, "executed": False,
        "accepted": False, "acquired": False, "prepared": False, "trained": False,
        "plan": plan,
        "execution_provider": execution_provider(PROCESSING),
    }
    if plan_only:
        return {**base, "status": "planned"}
    if not isinstance(store, SessionAcquisitionStore):
        return {**base, "status": "denied", "detail": "Trusted conversation acquisition state is required."}
    caller_policy = policy if isinstance(policy, ApprovalPolicy) else ApprovalPolicy()
    decision = request_approval(
        action="data_collection_flow", cost_estimate=None,
        impact="dataset acquisition; selected Processing resource cost quote is unavailable",
        policy=caller_policy,
    )
    if decision["decision"] != "approved":
        return {**base, "status": "denied", "decision": decision}
    try:
        selected = _resolve_backend()
        if output_dir is not None and selected.adapter_name != "local":
            return {
                **base, "status": "failed", "approved": True, "decision": decision,
                "detail": "output_dir is available only for an explicitly selected local adapter.",
            }
        resources: dict[str, Any] = {"runtime_session_id": store.runtime_session_id}
        if output_dir is not None:
            resources["output_dir"] = output_dir
        handle = selected.adapter.submit(
            repo_id=repo_id, revision=revision, filename=filename, resources=resources,
        )
        reference = store.add(selected.adapter, handle)
        result = {
            **base, "status": "submitted", "approved": True, "executed": True,
            "accepted": True, "request_ref": reference,
            "job_name": handle.id, "job_arn": handle.arn, "provider": handle.backend,
            "native_status": handle.extra.get("native_status", "Submitted"),
            "observation_error": handle.extra.get("observation_error", ""),
            "poll_after_seconds": 15, "decision": decision,
        }
        if handle.backend == "local":
            return {**result, **store.observe(reference)}
        return result
    except Exception as error:
        return {
            **base, "status": "failed", "approved": True, "decision": decision,
            "error_type": type(error).__name__,
            "detail": "Acquisition submission failed; no local fallback was attempted.",
        }
