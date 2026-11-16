"""Fixed SFT tools; hidden SDK state is the only source of caller authority."""

from __future__ import annotations

from typing import Any

from strands import tool

from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.sft_state import SessionSftStore
from ..flows.qwen_sft_flow import run_qwen_sft_flow


def _context(context: Any) -> dict:
    state = getattr(context, "invocation_state", None)
    state = state if type(state) is dict else {}
    policy = state.get("_amlas_approval_policy")
    store = state.get("_amlas_sft_store")
    return {
        "policy": policy if isinstance(policy, ApprovalPolicy) else None,
        "store": store if isinstance(store, SessionSftStore) else None,
    }


@tool(context=True)
def submit_qwen_sft(
    model_request_ref: str,
    preparation_request_ref: str,
    recipe_id: str = "fit_v1",
    plan_only: bool = False,
    tool_context: Any = None,
) -> dict[str, Any]:
    """Submit fixed Qwen SFT from verified sources in this conversation.

    Every result carries execution_provider. In mode declared_controlled a controlled provider answers the
    SageMaker calls and no instance starts: label results as its result_label says. In mode unverified a real
    job may start. Use recipe_id full_v1 for the lifecycle: coding generation and the coding spec accept only
    a full_v1 checkpoint.

    A conversation keeps one training submission: the first call that is not a plan_only preview. Poll it
    with get_qwen_sft. Call submit_qwen_sft alone in its own message, never together with other tools: a
    message that mixes it with other tool calls is refused whole and none of its tools run. Repeating the
    same call returns the recorded submission; a call with other inputs is refused and names the submission.

    Args:
        model_request_ref: Opaque acquired model reference.
        preparation_request_ref: Opaque preparation_request_ref (or compatibility
            request_ref) from prepare_coding_dataset/get_coding_preparation; never
            the preparation_ref artifact, an artifact URI or recipe_id.
        recipe_id: fit_v1 is a 32-step pilot that proves fit only; full_v1 is the lifecycle recipe.
        plan_only: Preview without source observation, staging or submission.
    """
    return run_qwen_sft_flow(
        model_request_ref=model_request_ref,
        preparation_request_ref=preparation_request_ref,
        recipe_id=recipe_id,
        plan_only=plan_only,
        **_context(tool_context),
    )


@tool(context=True)
def get_qwen_sft(request_ref: str, tool_context: Any = None) -> dict[str, Any]:
    """Observe the same native SFT job; never submit a replacement.

    Args:
        request_ref: Opaque SFT reference in this conversation.
    """
    return run_qwen_sft_flow(operation="observe", request_ref=request_ref, **_context(tool_context))


@tool(context=True)
def reload_qwen_sft(request_ref: str, plan_only: bool = False, tool_context: Any = None) -> dict[str, Any]:
    """Reload the verified original checkpoint in a new fixed native process.

    Args:
        request_ref: Original training reference in this conversation.
        plan_only: Preview without reading or staging checkpoint artifacts.
    """
    return run_qwen_sft_flow(
        operation="reload", request_ref=request_ref, plan_only=plan_only, **_context(tool_context)
    )
