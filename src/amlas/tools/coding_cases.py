"""Public case tools never return private oracle or exclusion artifact locations."""

from __future__ import annotations

from typing import Any

from strands import tool

from ..flows.coding_case_freeze_flow import run_coding_case_freeze_flow


@tool
def list_coding_case_catalog() -> dict:
    """List the fixed official164 case source and metadata, never golden code."""
    from ..data_eng.coding_case_profile import (
        CARD_SHA256,
        SOURCE_BYTES,
        SOURCE_FILENAME,
        SOURCE_REPO,
        SOURCE_REVISION,
        SOURCE_SHA256,
    )

    return {
        "catalog_id": "humaneval_164",
        "catalog_version": "1",
        "case_count": 164,
        "frozen": False,
        "acquired": False,
        "execution_status": "not_executed",
        "source": {
            "repo_id": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "filename": SOURCE_FILENAME,
            "sha256": SOURCE_SHA256,
            "byte_count": SOURCE_BYTES,
            "format": "parquet",
            "license": "MIT",
            "card_sha256": CARD_SHA256,
        },
    }


@tool(context=True)
def freeze_coding_cases(
    acquisition_request_ref: str,
    catalog_id: str = "humaneval_164",
    catalog_version: str = "1",
    plan_only: bool = False,
    tool_context: Any = None,
) -> dict:
    """Freeze an original same-conversation opaque acquisition; no role/resource inputs."""
    state = getattr(tool_context, "invocation_state", None)
    state = state if isinstance(state, dict) else {}
    return run_coding_case_freeze_flow(
        acquisition_request_ref=acquisition_request_ref,
        catalog_id=catalog_id,
        catalog_version=catalog_version,
        plan_only=plan_only,
        policy=state.get("_amlas_approval_policy"),
        store=state.get("_amlas_case_freeze_store"),
    )


@tool(context=True)
def get_coding_case_freeze(case_ref: str, tool_context: Any = None) -> dict:
    """Observe one opaque case reference; private artifacts never enter this result."""
    state = getattr(tool_context, "invocation_state", None)
    state = state if isinstance(state, dict) else {}
    return run_coding_case_freeze_flow(
        case_ref=case_ref,
        operation="observe",
        policy=state.get("_amlas_approval_policy"),
        store=state.get("_amlas_case_freeze_store"),
    )
