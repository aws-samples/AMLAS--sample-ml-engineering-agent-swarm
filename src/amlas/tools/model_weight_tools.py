"""Model-visible catalog/ref selectors only; authority arrives in typed SDK context."""

from __future__ import annotations

from typing import Any, Literal

from strands import tool

from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.model_acquisition_state import SessionModelAcquisitionStore
from ..flows.model_weight_acquisition_flow import run_model_weight_acquisition_flow


@tool(context=True)
def acquire_model_weights(
    catalog_id: str,
    catalog_version: str,
    plan_only: bool = False,
    tool_context: Any = None,
    *,
    mode: Literal["acquire", "reuse_verified"] = "acquire",
    pack_ref: str | None = None,
) -> dict[str, Any]:
    """Preview or submit the fixed source catalog model; submission is not acquisition.

    Args:
        catalog_id: Source-owned model catalog identifier.
        catalog_version: Exact source-owned catalog version.
        plan_only: Preview with no native writes.
        mode: Acquire source catalog bytes or reuse a configured verified pack.
        pack_ref: Opaque verified pack selector, required for reuse_verified.
    """
    state = getattr(tool_context, "invocation_state", None)
    state = state if type(state) is dict else {}
    policy, store = state.get("_amlas_approval_policy"), state.get("_amlas_model_acquisition_store")
    return run_model_weight_acquisition_flow(
        catalog_id=catalog_id,
        catalog_version=catalog_version,
        plan_only=plan_only,
        mode=mode,
        pack_ref=pack_ref,
        policy=policy if isinstance(policy, ApprovalPolicy) else None,
        store=store if isinstance(store, SessionModelAcquisitionStore) else None,
    )


@tool(context=True)
def get_model_weight_acquisition(request_ref: str, tool_context: Any = None) -> dict[str, Any]:
    """Observe only the current conversation's retained native model-fetch record.

    Args:
        request_ref: Opaque model_acq reference returned by acquisition.
    """
    state = getattr(tool_context, "invocation_state", None)
    store = state.get("_amlas_model_acquisition_store") if type(state) is dict else None
    if not isinstance(store, SessionModelAcquisitionStore):
        return {
            "success": False,
            "accepted": False,
            "acquired": False,
            "trusted": False,
            "ready": False,
            "trained": False,
            "status": "unknown_reference",
        }
    return store.observe(request_ref)


@tool(context=True)
def list_model_weight_catalog(tool_context: Any = None) -> list[dict[str, Any]]:
    """List fresh source-approved expected model bytes; no acquisition or authority."""
    from ..data_eng.model_weight_manifest import expected_model_weight_manifest

    catalogs = [expected_model_weight_manifest("qwen3_5_9b", "1")]
    state = getattr(tool_context, "invocation_state", None)
    store = state.get("_amlas_model_acquisition_store") if type(state) is dict else None
    if isinstance(store, SessionModelAcquisitionStore):
        candidates = store.verified_pack_candidates()
        for catalog in catalogs:
            matching = [
                {key: candidate[key] for key in ("catalog_id", "catalog_version", "pack_ref")}
                for candidate in candidates
                if candidate["catalog_id"] == catalog["catalog_id"]
                and candidate["catalog_version"] == catalog["catalog_version"]
            ]
            if matching:
                catalog["verified_packs"] = matching
    return catalogs
