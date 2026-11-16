"""Fresh caller approval gates resolved case planning and managed paired execution."""

from __future__ import annotations

from ..control_plane.approvals import ApprovalPolicy, request_approval
from ..control_plane.generation_state import SessionCodingGenerationStore
from ..control_plane.paired_state import PairConfigurationMissingError, SessionPairedCodingStore
from ..control_plane.provider_disclosure import PROCESSING, execution_provider
from .qwen_sft_flow import _public

_PAIR_CORE_FILES = ("COMMITTED.json", "RECEIPTS.jsonl", "EVALUATION.json", "INVOCATIONS_MANIFEST.jsonl")


def _bounded_pair_observation(result):
    """Hand the agent the four core refs and a record count; the manifest ref binds every verified record."""
    files = result.get("files")
    if type(files) is not dict:
        return result
    kept = {name: files[name] for name in _PAIR_CORE_FILES if name in files}
    return {
        **result,
        "files": kept,
        "invocation_records_view": {
            "view": "bounded_v1",
            "evidence_field": "files.INVOCATIONS_MANIFEST.jsonl",
            "record_count": len(files) - len(kept),
        },
    }


def run_coding_pair_flow(
    *,
    operation="submit",
    case_ref="",
    base_ref="",
    tuned_ref="",
    spec_ref="",
    request_ref="",
    plan_only=False,
    policy=None,
    generation=None,
    store=None,
):
    base = {
        "success": False,
        "accepted": False,
        "evaluated": False,
        "quality_success": False,
        "native_admission": False,
        "executed": False,
        "cost_estimate": None,
        "execution_provider": execution_provider(PROCESSING),
    }
    if plan_only:
        return {
            **base,
            "status": "planned",
            "case_count": 164,
            "required_receipts": 328,
            "case_wall_seconds": 10,
        }
    if (
        operation not in ("resolve", "submit", "observe")
        or (operation == "resolve" and type(generation) is not SessionCodingGenerationStore)
        or (operation != "resolve" and type(store) is not SessionPairedCodingStore)
    ):
        return {**base, "status": "denied"}
    decision = request_approval(
        action="coding_pair_flow",
        cost_estimate=None,
        impact="original case observation or fixed AWS paired evaluation",
        policy=policy if isinstance(policy, ApprovalPolicy) else ApprovalPolicy(),
    )
    if decision["decision"] != "approved":
        return {**base, "status": "denied", "decision": decision}
    try:
        result = (
            generation.resolve_case_spec(case_ref, spec_ref)
            if operation == "resolve"
            else store.submit(case_ref, base_ref, tuned_ref, spec_ref)
            if operation == "submit"
            else _bounded_pair_observation(store.observe(request_ref))
        )
        return _public(
            {**base, **result, "executed": operation == "submit" and result.get("accepted") is True}
        )
    except PairConfigurationMissingError as error:
        return {**base, "status": "configuration_missing", "missing_selectors": list(error.missing_selectors)}
    except Exception as error:
        return {**base, "status": "refused", "error_type": type(error).__name__}
