"""Fresh caller approval before managed preparation source reads or effects."""

from __future__ import annotations

from typing import Any

from ..control_plane.approvals import ApprovalPolicy, request_approval
from ..control_plane.preparation_state import PreparationConfigurationMissingError, SessionPreparationStore
from ..control_plane.provider_disclosure import PROCESSING, execution_provider
from ..data_eng import commitpack_preparation
from ..sagemaker_entrypoints import coding_preparation as canonical_preparation


def _profile_family(profile_id: Any, profile_version: Any) -> tuple[bool, str | None]:
    """Whether the pair names a listed profile, and the source family it requires.

    A profile argument is never authority: the private acquisition sources decide which preparation
    runs. Each supported profile is accepted with its advertised policy version or the tool default "1".
    """
    if profile_id == canonical_preparation.PROFILE_ID and profile_version in (
        "1",
        canonical_preparation.POLICY_VERSION,
    ):
        return True, None
    if profile_id == commitpack_preparation.PROFILE_ID and profile_version in (
        "1",
        commitpack_preparation.POLICY_VERSION,
    ):
        return True, "commitpack"
    return False, None


def run_coding_preparation_flow(
    *,
    acquisition_request_refs: list[str],
    case_request_ref: str | None = None,
    profile_id: str = "openai.explicit_conversations.v1",
    profile_version: str = "1",
    plan_only: bool = False,
    policy: ApprovalPolicy | None = None,
    store: SessionPreparationStore | None = None,
) -> dict[str, Any]:
    plan = {
        "flow_id": "data_preparation_flow",
        "profile_id": profile_id,
        "profile_version": profile_version,
        "steps": [
            "verify exact acquired versions",
            "stage immutable selection",
            "submit fixed Processing preparation",
            "verify committed outputs",
        ],
        "cost_estimate": None,
        "cost_uncertainty": "selected native resources have no current quote",
    }
    base = {
        "success": False,
        "accepted": False,
        "prepared": False,
        "trained": False,
        "executed": False,
        "plan": plan,
        "execution_provider": execution_provider(PROCESSING),
    }
    if plan_only:
        if not _profile_family(profile_id, profile_version)[0]:
            return {**base, "status": "unsupported_profile"}
        return {**base, "status": "planned"}
    if not isinstance(store, SessionPreparationStore):
        return {**base, "status": "denied", "detail": "Trusted conversation preparation state is required."}
    caller = policy if isinstance(policy, ApprovalPolicy) else ApprovalPolicy()
    decision = request_approval(
        action="data_preparation_flow",
        cost_estimate=None,
        impact="managed canonical preparation stages versioned artifacts and runs Processing",
        policy=caller,
    )
    if decision["decision"] != "approved":
        return {**base, "status": "denied", "decision": decision}
    return {
        **base,
        **execute_coding_preparation_flow(
            acquisition_request_refs=acquisition_request_refs,
            case_request_ref=case_request_ref,
            profile_id=profile_id,
            profile_version=profile_version,
            store=store,
        ),
        "decision": decision,
    }


def execute_coding_preparation_flow(
    *,
    acquisition_request_refs: list[str],
    case_request_ref: str | None = None,
    profile_id: str = "openai.explicit_conversations.v1",
    profile_version: str = "1",
    store: SessionPreparationStore | None = None,
) -> dict[str, Any]:
    """Post-approval private capability executor; never model-supplied authority."""
    base = {"success": False, "accepted": False, "prepared": False, "trained": False, "executed": False}
    if not isinstance(store, SessionPreparationStore):
        return {**base, "status": "denied", "detail": "Trusted conversation preparation state is required."}
    supported, family = _profile_family(profile_id, profile_version)
    if not supported:
        return {**base, "status": "unsupported_profile"}
    options: dict[str, Any] = {} if case_request_ref is None else {"case_request_ref": case_request_ref}
    if family is not None:
        options["profile_family"] = family
    try:
        result = store.submit(acquisition_request_refs, **options)
        return {**base, **result, "executed": result.get("accepted") is True}
    except PreparationConfigurationMissingError as error:
        return {**base, "status": "configuration_missing", "missing_selectors": list(error.missing_selectors)}
    except Exception as error:
        return {
            **base,
            "status": "refused",
            "error_type": type(error).__name__,
            "detail": "Preparation did not dispatch; no local or unsupported-profile fallback.",
        }
