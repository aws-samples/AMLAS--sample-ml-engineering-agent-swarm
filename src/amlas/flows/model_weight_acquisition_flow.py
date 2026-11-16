"""Caller policy gates one catalog-only Processing model fetch, no local fallback."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Literal

from ..backends.adapters.model_weight_acquisition import (
    MODEL_ACQUISITION_CONFIGURATION_ENV_KEYS,
    ModelAcquisitionConfigurationError,
    _project_model_diagnostic,
)
from ..backends.ports import ModelWeightAcquisitionBackend
from ..backends.registry import REGISTRY
from ..config.externalized import model_weight_acquisition_selectors
from ..control_plane.approvals import ApprovalPolicy, request_approval
from ..control_plane.model_acquisition_state import SessionModelAcquisitionStore
from ..control_plane.provider_disclosure import PROCESSING, execution_provider
from ..data_eng.model_weight_manifest import expected_model_weight_manifest


def _resolve_backend() -> ModelWeightAcquisitionBackend:
    resolved = REGISTRY.resolve(ModelWeightAcquisitionBackend)
    try:
        selectors = model_weight_acquisition_selectors()
    except ValueError:
        missing = tuple(
            name for name in MODEL_ACQUISITION_CONFIGURATION_ENV_KEYS if not os.environ.get(name, "").strip()
        )
        raise ModelAcquisitionConfigurationError(
            "model_acquisition_configuration_missing"
            if missing
            else "model_acquisition_configuration_invalid",
            missing or MODEL_ACQUISITION_CONFIGURATION_ENV_KEYS,
        ) from None
    if selectors["state"] == "unconfigured":
        raise ModelAcquisitionConfigurationError(
            "model_acquisition_configuration_missing", MODEL_ACQUISITION_CONFIGURATION_ENV_KEYS
        )
    return resolved.adapter


def run_model_weight_acquisition_flow(
    *,
    catalog_id: str,
    catalog_version: str,
    plan_only: bool = False,
    mode: Literal["acquire", "reuse_verified"] = "acquire",
    pack_ref: str | None = None,
    policy: ApprovalPolicy | None = None,
    store: SessionModelAcquisitionStore | None = None,
) -> dict[str, Any]:
    expected = expected_model_weight_manifest(catalog_id, catalog_version)
    base = {
        "success": False,
        "accepted": False,
        "acquired": False,
        "trusted": False,
        "ready": False,
        "prepared": False,
        "trained": False,
        "executed": False,
        "plan": {
            "catalog_id": catalog_id,
            "catalog_version": catalog_version,
            "repo_id": expected["repo_id"],
            "revision": expected["revision"],
            "expected_bytes": sum(item["byte_count"] for item in expected["files"]),
            "cost_estimate": None,
            "native_runtime_seconds": 7200,
            "volume_size_gb": 80,
        },
        "execution_provider": execution_provider(PROCESSING),
    }
    if type(mode) is not str or mode not in ("acquire", "reuse_verified"):
        raise ValueError("model acquisition mode differs")
    if type(plan_only) is not bool:
        raise ValueError("model plan selector must be an actual boolean")
    if plan_only:
        return {**base, "status": "planned"}
    if not isinstance(policy, ApprovalPolicy) or not isinstance(store, SessionModelAcquisitionStore):
        return {
            **base,
            "status": "denied",
            "detail": "Typed caller policy and private model state are required.",
        }
    snapshot = replace(policy, allowlist=list(policy.allowlist))
    decision = request_approval(
        action="run_processing_job",
        cost_estimate=None,
        impact="fixed public model-weight Processing acquisition; exact cost quote unavailable",
        policy=snapshot,
    )
    if decision["decision"] != "approved":
        return {**base, "status": "denied", "decision": decision}
    if not store._submission_lock.acquire(blocking=False):
        return {**base, "status": "busy", "decision": decision}
    legacy_invoked = False
    campaign_invoked = False
    reuse_invoked = False
    try:
        store._require_invocation_context()
        if mode == "reuse_verified":
            reuse_invoked = True
            result = store.reuse_verified(
                catalog_id=catalog_id,
                catalog_version=catalog_version,
                pack_ref=pack_ref,
                policy=snapshot,
            )
            return {**base, **result, "decision": decision}
        if store._campaign_binding is not None or store._campaign_entry is not None:
            if store._campaign_binding is None:
                store.require_campaign_services()
                store.bind_campaign(_resolve_backend())
            campaign_invoked = True
            original = store.submit_original(
                catalog_id=catalog_id, catalog_version=catalog_version, policy=snapshot
            )
            handle = original.handle
            diagnostic = _project_model_diagnostic(
                handle.extra.get("submission_diagnostic"), "create_boundary"
            )
            return {
                **base,
                "executed": original.executed,
                "accepted": handle.extra.get("accepted") is True,
                "status": "submission_unknown"
                if original.disposition == "UNKNOWN"
                else "submitted"
                if original.executed
                else "observed_original",
                "request_ref": original.reference,
                "job_name": handle.id,
                "job_arn": handle.arn,
                "reused": not original.executed,
                "decision": decision,
                "poll_after_seconds": 30,
                **({"submission_diagnostic": diagnostic} if diagnostic is not None else {}),
            }
        existing = store.existing(catalog_id, catalog_version)
        if existing:
            return {**base, **store.observe(existing), "request_ref": existing, "decision": decision}
        adapter = _resolve_backend()
        legacy_invoked = True
        handle = adapter.submit(
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            runtime_session_id=store.runtime_session_id,
        )
        reference = store.add(adapter, handle)
        diagnostic = _project_model_diagnostic(handle.extra.get("submission_diagnostic"), "create_boundary")
        return {
            **base,
            "executed": True,
            "accepted": handle.extra.get("accepted") is True,
            "status": "submission_unknown" if handle.extra.get("submission_ambiguous") else "submitted",
            "request_ref": reference,
            "job_name": handle.id,
            "job_arn": handle.arn,
            "decision": decision,
            "poll_after_seconds": 30,
            **({"submission_diagnostic": diagnostic} if diagnostic is not None else {}),
        }
    except Exception as error:
        from ..control_plane.model_campaign_binding import ModelSubmissionAttemptError

        if type(error) is ModelSubmissionAttemptError:
            if error.dispatch_attempted is False:
                # Only the declared exact before_create refusal supplies this
                # independent no-dispatch fact; it grants no identity/effect.
                return {
                    **base,
                    "status": "submission_failed",
                    "error_type": type(error).__name__,
                    "decision": decision,
                }
            outcome = {
                **base,
                "executed": None,
                "status": "outcome_unknown",
                "error_type": type(error).__name__,
                "decision": decision,
            }
            if error.currently_qualified is True:
                outcome["executed"] = error.dispatch_attempted
                if error.result is not None:
                    outcome.update(
                        request_ref=error.result.reference,
                        job_name=error.result.handle.id,
                        job_arn=error.result.handle.arn,
                    )
            return outcome
        result = {
            **base,
            "executed": None if legacy_invoked or campaign_invoked or reuse_invoked else False,
            "status": "outcome_unknown"
            if legacy_invoked or campaign_invoked or reuse_invoked
            else "submission_failed",
            "error_type": type(error).__name__,
            "decision": decision,
        }
        if type(error) is ModelAcquisitionConfigurationError:
            result.update(
                error_code=error.error_code,
                configuration_keys=list(error.configuration_keys),
                detail=error.detail,
            )
        return result
    finally:
        store._submission_lock.release()
