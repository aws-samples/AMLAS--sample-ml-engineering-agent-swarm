"""Python-API surface — the programmatic entry over the M7 capability funnel (ADR-010).

The fourth surface (alongside Chat / TUI / CLI): a clean, importable Python API so power
users + the Agent-SDK call the SAME capabilities every interactive surface reaches. It is
a THIN adapter — every function delegates to ``control_plane.capability.invoke_capability``
(the single funnel) or a control-plane facet; NO flow logic lives here (ADR-010: surfaces
are renderers, never reimplementations).

    from amlas import api
    api.run_flow("data_quality_flow", params={"dataset_uri": "data/x.csv"})   # generic
    api.run_dq_flow(dataset_uri="data/x.csv")                                 # named helper
    api.list_capabilities()                                                    # discover
    api.preview("deploy_flow", params={...})                                   # Plan Mode
    api.run_headless("data_quality_flow", params={...})                        # CI / -p
    api.resume("my-session")                                                   # --resume
    api.mention("@dataset:silver/sensors")                                     # @-mention

``run_flow`` returns the surface-neutral :class:`CapabilityResult` (the ADR-010 parity
contract); the named helpers (``run_dq_flow``/``deploy_endpoint``/…) are ergonomic
wrappers that build the params dict for a specific capability. Pure delegation; no AWS.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config.settings import Settings
    from .tui.chat.models import ChatResponse

from .control_plane.approvals import ApprovalPolicy
from .control_plane.capability import (
    CapabilityResult,
    describe_capability,
    invoke_capability,
)
from .control_plane.capability import (
    list_capabilities as _list_capabilities,
)
from .control_plane.mention import mention_context as _mention_context
from .control_plane.runtime_session import (
    resume_session as _resume_session,
)
from .control_plane.runtime_session import (
    run_headless as _run_headless,
)

__all__ = [
    "deploy_endpoint",
    "describe_capability",
    "list_capabilities",
    "mention",
    "open_pr_for_artifacts",
    "preview",
    "resume",
    # named ergonomic helpers (thin wrappers over run_flow)
    "run_dq_flow",
    "run_flow",
    "run_headless",
    "run_imputation_flow",
    "run_monitor_flow",
    "run_prompt",
]


class _PromptModelDiagnosticSlot:
    """One invocation-local optional DATA publication; never an observer selector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._worker_thread = None
        self._worker_active = False
        self._consumer_active = True
        self._candidate = None
        self._latched_key = None

    def _begin_worker(self):
        with self._lock:
            if self._pid != os.getpid() or self._worker_thread is not None:
                raise ValueError("diagnostic worker scope differs")
            self._worker_thread = threading.get_ident()
            self._worker_active = True

    def _end_worker(self):
        with self._lock:
            self._worker_active = False
            self._candidate = None

    def _is_active(self):
        with self._lock:
            return self._pid == os.getpid() and self._worker_active and self._consumer_active

    def _producer_current(self):
        return (
            _PROMPT_MODEL_DIAGNOSTIC_SLOT.get() is self
            and self._worker_thread == threading.get_ident()
            and self._is_active()
        )

    def _stage(self, publication):
        from .tui.chat.tool_binding_backend import _OriginalModelDiagnosticPublication

        if type(publication) is not _OriginalModelDiagnosticPublication or not self._producer_current():
            return
        with self._lock:
            if (
                not self._worker_active
                or not self._consumer_active
                or publication.pid != self._pid
                or publication.worker_thread != self._worker_thread
                or self._candidate is not None
                or self._latched_key == publication.key
            ):
                return
            self._latched_key = publication.key
            self._candidate = publication

    def take_current(self):
        with self._lock:
            if self._pid != os.getpid() or not self._worker_active or not self._consumer_active:
                return None
            publication, self._candidate = self._candidate, None
        if publication is None:
            return None
        try:
            return publication.current_data(self)
        except Exception as error:
            from .tui.chat.tool_binding_backend import _warm_error_class

            logging.getLogger(__name__).debug(
                "MODEL_DIAGNOSTIC_PUBLICATION_DROP phase=consumer error_class=%s", _warm_error_class(error)
            )
            # Optional metadata cannot close/cancel the original ML worker.
            return None

    def close_consumer(self):
        with self._lock:
            self._consumer_active = False
            self._candidate = None


_PROMPT_MODEL_DIAGNOSTIC_SLOT = ContextVar("amlas_original_model_diagnostic_slot", default=None)


@contextmanager
def _prompt_model_diagnostic_scope(slot):
    if type(slot) is not _PromptModelDiagnosticSlot:
        raise ValueError("source-owned diagnostic slot required")
    slot._begin_worker()
    token = _PROMPT_MODEL_DIAGNOSTIC_SLOT.set(slot)
    try:
        yield
    finally:
        try:
            _PROMPT_MODEL_DIAGNOSTIC_SLOT.reset(token)
        finally:
            slot._end_worker()


def _current_prompt_model_diagnostic_slot():
    slot = _PROMPT_MODEL_DIAGNOSTIC_SLOT.get()
    return slot if type(slot) is _PromptModelDiagnosticSlot and slot._is_active() else None


async def run_prompt(
    prompt: str,
    *,
    settings: Settings | None = None,
    policy: ApprovalPolicy | None = None,
) -> ChatResponse:
    """Send a natural prompt through the normal chat engine.

    Caller-owned policy accompanies the capability path outside model input.
    Missing policy keeps the existing safe noninteractive approval behavior.
    This adapter supplies no experiment specification, data or forced tools.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must contain a nonempty natural-language request")
    if policy is not None and not isinstance(policy, ApprovalPolicy):
        raise TypeError("policy must be an ApprovalPolicy supplied by the caller")

    from .control_plane.campaign_runtime import campaign_prompt_scope
    from .tui.chat.engine import _WARM_CHAT_ENGINE_TYPE, ChatEngine

    with campaign_prompt_scope(prompt, policy=policy) as context:
        engine = ChatEngine(settings=settings, approval_policy=policy)
        if context is not None:
            context.require_current()
        response = await engine.process_message(prompt)
        if context is not None:
            context.require_current()
        if type(engine) is _WARM_CHAT_ENGINE_TYPE:
            try:
                while engine._has_pending_original_model():
                    if context is not None:
                        context.require_current()
                    step = await engine._observe_pending_model_once()
                    if context is not None:
                        context.require_current()
                    if step.disposition == "awaiting_original":
                        diagnostic_slot = _current_prompt_model_diagnostic_slot()
                        if diagnostic_slot is not None:
                            engine.tool_backend._stage_original_model_diagnostic(diagnostic_slot)
                        if context is not None:
                            context.require_current()
                        await asyncio.sleep(15.0)
                        if context is not None:
                            context.require_current()
                        continue
                    if step.turn is not None:
                        response = engine._agent_turn_response(step.turn)
                    continuation = {"disposition": step.disposition}
                    if step.diagnostic is not None:
                        continuation["diagnostic"] = step.diagnostic.as_data()
                    response.data = {
                        **response.data,
                        "warm_model_continuation": continuation,
                    }
                    if context is not None:
                        context.require_current()
                    if (
                        step.disposition in {"continued_original", "terminal_failed"}
                        and engine._has_pending_original_model()
                    ):
                        continue
                    return response
            except asyncio.CancelledError:
                # Sleep cancellation also closes the original; observation
                # cancellation has already joined its exact worker in the engine.
                engine._close_original_model()
                raise
        return response


# ── the generic Python-API entry (the ADR-010 funnel) ────────────────────────
def run_flow(
    flow_id: str,
    *,
    params: dict[str, Any] | None = None,
    policy: ApprovalPolicy | None = None,
    steering: Any | None = None,
) -> CapabilityResult:
    """Invoke any registered capability from Python (the ADR-010 single funnel).

    Identical to the Chat/TUI/CLI path — it calls ``invoke_capability`` with
    ``surface="api"``, so the SAME flow id + params + policy yields an equivalent result
    on every surface (the parity guarantee). Spend/irreversible flows route through the
    approval gate; cheap reads stream through.

    Args:
        flow_id: The capability/flow id (e.g. ``"data_quality_flow"``).
        params: Keyword params forwarded to the flow executor.
        policy: The approval policy (default safe ``ask``; non-interactive denies gated).
        steering: Optional steering profile.

    Returns:
        A :class:`CapabilityResult` (the surface-neutral outcome contract).
    """
    return invoke_capability(
        capability_id=flow_id, params=params, policy=policy, steering=steering, surface="api"
    )


def preview(
    flow_id: str, *, params: dict[str, Any] | None = None, steering: Any | None = None
) -> CapabilityResult:
    """Plan-Mode preview a capability from Python — build the read-only plan, run nothing."""
    return invoke_capability(
        capability_id=flow_id, params=params, steering=steering, surface="api", plan_only=True
    )


def list_capabilities() -> list[dict[str, Any]]:
    """List every registered capability (the discovery surface for the Python API)."""
    return _list_capabilities()


def run_headless(
    flow_id: str,
    *,
    params: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    max_budget: float | None = None,
    bypass: bool = False,
) -> dict[str, Any]:
    """Run a capability non-interactively (T-RUN-05) — the Python entry to headless mode."""
    return _run_headless(
        capability_id=flow_id,
        params=params,
        allowed_tools=allowed_tools,
        max_budget=max_budget,
        bypass=bypass,
    )


def resume(session_ref: str, *, project_dir: str = ".") -> dict[str, Any]:
    """Resume a prior session (T-RUN-06) — the Python entry to session resume."""
    return _resume_session(session_ref=session_ref, project_dir=project_dir)


def mention(ref: str, *, project_dir: str = ".", **kwargs: Any) -> dict[str, Any]:
    """Resolve an ``@`` mention into context (T-LAKE-07) — the Python entry to @-mention."""
    return _mention_context(ref, project_dir=project_dir, **kwargs)


# ── named ergonomic helpers (thin wrappers; doc 07 M7 work-item 4) ───────────
def run_dq_flow(*, dataset_uri: str, policy: ApprovalPolicy | None = None, **params: Any) -> CapabilityResult:
    """Run the data-quality assessment flow (``run_dq_flow(...)`` — doc 07 M7 example)."""
    return run_flow("data_quality_flow", params={"dataset_uri": dataset_uri, **params}, policy=policy)


def run_imputation_flow(*, policy: ApprovalPolicy | None = None, **params: Any) -> CapabilityResult:
    """Run the imputation flow (diagnose → recommend → apply → validate)."""
    return run_flow("imputation_flow", params=dict(params), policy=policy)


def deploy_endpoint(*, policy: ApprovalPolicy | None = None, **params: Any) -> CapabilityResult:
    """Run the deploy flow (``deploy_endpoint(...)`` — doc 07 M7 example; irreversible → gated)."""
    return run_flow("deploy_flow", params=dict(params), policy=policy)


def run_monitor_flow(*, policy: ApprovalPolicy | None = None, **params: Any) -> CapabilityResult:
    """Run the monitoring flow (metrics → drift → alerts → retrain-trigger decision)."""
    return run_flow("monitor_flow", params=dict(params), policy=policy)


def open_pr_for_artifacts(
    *, paths: list[str] | str, branch: str, policy: ApprovalPolicy | None = None, **params: Any
) -> CapabilityResult:
    """Run the devops git flow (commit artifacts → open PR/MR; push/MR gated; US-CC.08)."""
    return run_flow("devops_git_flow", params={"paths": paths, "branch": branch, **params}, policy=policy)
