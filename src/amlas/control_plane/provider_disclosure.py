"""Informational report of which provider answers a flow's SageMaker calls; it grants nothing.

The report is computed from the live process on every call and is never cached or accepted from a caller.
Controlled execution is reported only when the process-wide SageMaker client factory carries a live, well-formed
installation of the declared controlled provider whose operation scope covers every operation the flow uses.
Every other state, including any error while checking, is reported as unverified with a static reason code.
The report is a trusted-runtime assertion about the live process, not an attestation: code that controls the
process can forge it. No model-supplied value reaches it.
"""

from __future__ import annotations

import os
import types
from typing import Any

PROCESSING = ("CreateProcessingJob", "DescribeProcessingJob")
TRAINING = ("CreateTrainingJob", "DescribeTrainingJob")

_MARKER = "_managed_test_overlay_owner"
_CONTROLLED_LABEL = (
    "SIMULATED (controlled provider): no SageMaker instance was started for these operations. Job responses, "
    "outputs and metrics are provider-produced and are not model-quality evidence. Inputs fetched from external "
    "sources are real."
)
_PLAN_NOTE = (
    "Plan steps describe the native flow. In this deployment a controlled provider answers these SageMaker calls."
)
_UNVERIFIED_LABEL = (
    "Controlled execution is not verified for these operations: a real SageMaker job may start. "
    "Do not label results simulated and do not assume they are real."
)


def _allowed_operations() -> tuple[frozenset[str] | None, str]:
    """Return the live installation's operation scope, or None with a static reason code."""
    try:
        import boto3

        factory = boto3.session.Session.client
        if type(factory) is not types.FunctionType:
            return None, "marker_absent"
        owner = getattr(factory, _MARKER, None)
        if owner is None:
            return None, "marker_absent"
        kind = type(owner)
        if kind.__qualname__ != "Installation" or "managed_sagemaker_client_overlay" not in str(kind.__module__):
            return None, "check_failed"
        allowed = owner._allowed
        if (
            type(allowed) is not frozenset
            or not allowed
            or any(type(name) is not str for name in allowed)
            or owner._pid != os.getpid()
            or owner._closed is not False
        ):
            return None, "check_failed"
        owner._check()
        return allowed, ""
    except Exception:
        return None, "check_failed"


def _unverified(reason: str) -> dict[str, Any]:
    return {
        "mode": "unverified",
        "sagemaker_job_backing": "unverified",
        "real_sagemaker_execution_possible": True,
        "reason": reason,
        "result_label": _UNVERIFIED_LABEL,
    }


def execution_provider(operations: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Report the provider for the given SageMaker operations; static text and operation names only."""
    try:
        names = list(operations)
        if not names or any(type(name) is not str for name in names):
            raise ValueError("operation names required")
        required = sorted(set(names))
    except Exception:
        return _unverified("check_failed")
    allowed, reason = _allowed_operations()
    if allowed is not None and not set(required) <= allowed:
        allowed, reason = None, "operations_not_covered"
    if allowed is None:
        return _unverified(reason)
    return {
        "mode": "declared_controlled",
        "sagemaker_job_backing": "controlled_provider",
        "controlled_operations": required,
        "sagemaker_instances_started": False,
        "result_label": _CONTROLLED_LABEL,
        "plan_note": _PLAN_NOTE,
    }
