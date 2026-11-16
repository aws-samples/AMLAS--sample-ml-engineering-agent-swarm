"""Fresh caller approval gates fixed managed generation and polling."""

from __future__ import annotations

from ..control_plane.approvals import ApprovalPolicy, request_approval
from ..control_plane.generation_state import SessionCodingGenerationStore
from ..control_plane.provider_disclosure import TRAINING, execution_provider
from ..control_plane.sft_state import SftConfigurationMissingError
from .qwen_sft_flow import _public


def run_coding_generation_flow(
    *,
    operation="submit",
    model_request_ref="",
    case_request_ref="",
    spec_ref="",
    sft_request_ref="",
    request_ref="",
    plan_only=False,
    policy=None,
    store=None,
):
    base = {
        "success": False,
        "accepted": False,
        "generated": False,
        "quality_success": False,
        "native_admission": False,
        "executed": False,
        "cost_estimate": None,
        "execution_provider": execution_provider(TRAINING),
    }
    if plan_only:
        return {**base, "status": "planned", "recipe_id": "humaneval_greedy_v1", "case_count": 164}
    if type(store) is not SessionCodingGenerationStore or operation not in ("submit", "observe"):
        return {**base, "status": "denied"}
    decision = request_approval(
        action="coding_generation_flow",
        cost_estimate=None,
        impact="fixed native GPU generation or exact-version observation",
        policy=policy if isinstance(policy, ApprovalPolicy) else ApprovalPolicy(),
    )
    if decision["decision"] != "approved":
        return {**base, "status": "denied", "decision": decision}
    try:
        result = (
            store.submit(model_request_ref, case_request_ref, spec_ref, sft_request_ref)
            if operation == "submit"
            else store.observe(request_ref)
        )
        return _public(
            {**base, **result, "executed": operation == "submit" and result.get("accepted") is True}
        )
    except SftConfigurationMissingError as error:
        return {**base, "status": "configuration_missing", "missing_selectors": list(error.missing_selectors)}
    except Exception as error:
        return {**base, "status": "refused", "error_type": type(error).__name__}
