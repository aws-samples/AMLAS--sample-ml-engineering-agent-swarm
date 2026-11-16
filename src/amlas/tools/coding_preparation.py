"""Opaque canonical-preparation tools; authority comes only from SDK state."""

from __future__ import annotations

from typing import Any

from strands import tool

from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.preparation_state import SessionPreparationStore
from ..flows.coding_preparation_flow import run_coding_preparation_flow


@tool(context=True)
def prepare_coding_dataset(
    acquisition_request_refs: list[str],
    case_request_ref: str | None = None,
    profile_id: str = "openai.explicit_conversations.v1",
    profile_version: str = "1",
    plan_only: bool = False,
    tool_context: Any = None,
) -> dict[str, Any]:
    """Dispatch canonical JSONL preparation and return an opaque pending reference.

    The result preparation_request_ref equals the compatibility request_ref.
    Poll that original selector until prepared before passing it to submit_qwen_sft;
    preparation_ref is a committed artifact descriptor, never the SFT selector.

    Args:
        acquisition_request_refs: Opaque acquisitions in this conversation.
        case_request_ref: Optional verified original frozen case in this conversation;
            supplies governed exclusions without a global exclusions configuration.
        profile_id: The canonical or CommitPackFT profile id from the listing; sources must match.
        profile_version: That profile's listed policy_version, or 1 (the default) as an alias.
        plan_only: Preview without source reads, staging or job creation.
    """
    state = getattr(tool_context, "invocation_state", None)
    state = state if isinstance(state, dict) else {}
    policy = state.get("_amlas_approval_policy")
    store = state.get("_amlas_preparation_store")
    return run_coding_preparation_flow(
        acquisition_request_refs=acquisition_request_refs,
        case_request_ref=case_request_ref,
        profile_id=profile_id,
        profile_version=profile_version,
        plan_only=plan_only,
        policy=policy if isinstance(policy, ApprovalPolicy) else None,
        store=store if isinstance(store, SessionPreparationStore) else None,
    )


@tool(context=True)
def get_coding_preparation(request_ref: str, tool_context: Any = None) -> dict[str, Any]:
    """Read an existing retained preparation; polling never submits another job.

    An owned result retains equal preparation_request_ref/request_ref selectors.
    Pass that opaque selector to submit_qwen_sft, never the preparation_ref artifact.

    Args:
        request_ref: Opaque reference returned by prepare_coding_dataset.
    """
    state = getattr(tool_context, "invocation_state", None)
    store = state.get("_amlas_preparation_store") if isinstance(state, dict) else None
    if not isinstance(store, SessionPreparationStore):
        return {
            "success": False,
            "accepted": False,
            "prepared": False,
            "trained": False,
            "status": "unknown_reference",
        }
    return store.observe(request_ref)


@tool(context=True)
def list_coding_preparation_profiles(
    repo_id: str | None = None,
    revision: str | None = None,
    filename: str | None = None,
    tool_context: Any = None,
) -> dict[str, Any]:
    """List existing preparation profiles before acquiring a public dataset.

    Optional source pins check shape compatibility only. Configured selections
    and matching source shapes do not verify bytes, permission or native readiness.
    Resource configuration is separate from the complete global exclusions catalog.
    Missing global exclusions can be supplied by a verified original case_request_ref
    from this conversation; discovery never observes or claims that case is ready.
    No payload, provider or AWS call occurs; unsupported formats have no fallback.

    Args:
        repo_id: Optional public Hub identifier, at most 96 characters per segment.
        revision: Optional exact lowercase 40-character public commit.
        filename: Optional confined relative filename, at most 1024 characters.
    """
    from ..control_plane.preparation_capabilities import describe_preparation_capabilities

    state = getattr(tool_context, "invocation_state", None)
    store = state.get("_amlas_preparation_store") if type(state) is dict else None
    catalog = vars(store).get("_catalog") if type(store) is SessionPreparationStore else None
    return describe_preparation_capabilities(
        repo_id=repo_id, revision=revision, filename=filename, catalog=catalog
    )
