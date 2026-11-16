"""Case operations use current typed caller policy, never case JSON authority."""

from __future__ import annotations

from ..control_plane.approvals import ApprovalPolicy, request_approval
from ..control_plane.coding_case_state import (
    CaseFreezeConfigurationMissingError,
    SessionCodingCaseFreezeStore,
)
from ..control_plane.provider_disclosure import PROCESSING, execution_provider


def run_coding_case_freeze_flow(
    *,
    acquisition_request_ref="",
    case_ref="",
    catalog_id="humaneval_164",
    catalog_version="1",
    operation="freeze",
    plan_only=False,
    policy=None,
    store=None,
):
    base = {
        "success": False,
        "accepted": False,
        "frozen": False,
        "executed": False,
        "execution_status": "not_executed",
        "quality_success": False,
        "native_admission": False,
        "execution_provider": execution_provider(PROCESSING),
    }
    plan = {
        "flow_id": "coding_case_freeze_flow",
        "catalog_id": catalog_id,
        "catalog_version": catalog_version,
        "case_count": 164,
        "cost_estimate": None,
        "cost_uncertainty": "fixed Processing and exact-version verification have no current quote",
        "steps": [
            "verify original private acquisition",
            "stage exact immutable raw source",
            "freeze data-only projections",
            "verify five outputs; expose public prompts only",
        ],
    }
    if operation not in ("freeze", "observe") or (catalog_id, catalog_version) != ("humaneval_164", "1"):
        return {**base, "status": "unsupported_catalog"}
    if plan_only:
        return {**base, "status": "planned", "plan": plan}
    if not isinstance(store, SessionCodingCaseFreezeStore) or not isinstance(policy, ApprovalPolicy):
        return {**base, "status": "denied"}
    decision = request_approval(
        action="coding_case_freeze_flow",
        cost_estimate=None,
        impact="managed case freeze or exact-version observation",
        policy=policy,
    )
    if decision["decision"] != "approved":
        return {**base, "status": "denied", "decision": decision}
    try:
        result = (
            store.submit(acquisition_request_ref, catalog_id, catalog_version)
            if operation == "freeze"
            else store.observe(case_ref)
        )
        return {**base, **result, "executed": operation == "freeze" and result.get("accepted") is True}
    except CaseFreezeConfigurationMissingError:
        return {**base, "status": "configuration_missing"}
    except Exception as error:
        return {**base, "status": "refused", "error_type": type(error).__name__}
