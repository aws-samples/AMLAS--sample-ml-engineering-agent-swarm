"""Opt-in private entry over current ingress and protected original campaign data.

Selected ports and explicit server approval are conditional configuration, never
native authentication proof. No public payload/type/header/session/STS value can
supply ingress approval. Unknown wake state cannot invoke an agent. This module
creates no ML job, persists no policy/witness/client, and implements no artifact
codec or native ingress verifier; those independently qualified ports are required.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Protocol

from .approvals import ApprovalPolicy
from .campaign_admission import AdmissionGuard
from .campaign_coordinator import (
    CampaignRecord,
    DynamoCampaignCoordinator,
    OriginalCampaignOperation,
    WakeHint,
    _sha,
)
from .campaign_initialization import DynamoCampaignInitialization, InitializationObservation

_SEAL = object()
_RUNTIME: ContextVar[object] = ContextVar("amlas_private_campaign_runtime", default=None)
_ENTRY: ContextVar[object] = ContextVar("amlas_private_campaign_entry", default=None)


class CampaignRuntimeError(ValueError):
    """Private original entry/qualification is missing, unknown or inconsistent."""


def _text(value, limit=256):
    if type(value) is not str or not 0 < len(value.encode()) <= limit or any(ord(c) < 32 for c in value):
        raise CampaignRuntimeError("closed bounded private campaign text required")
    return value


def _digest(value):
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    except (TypeError, ValueError, UnicodeError):
        raise CampaignRuntimeError("closed private request encoding refused") from None
    if len(raw) > 65536 + 4096:
        raise CampaignRuntimeError("private entry request exceeds bound")
    return hashlib.sha256(raw).hexdigest()


def _policy(value):
    if value is not None and not isinstance(value, ApprovalPolicy):
        raise CampaignRuntimeError("fresh caller-owned typed policy required")
    return None if value is None else replace(value, allowlist=list(value.allowlist))


@dataclass(frozen=True)
class CampaignIngressRequirement:
    admission_manifest_sha256: str
    ingress_boundary_sha256: str
    approval_reference: str

    def __post_init__(self):
        _sha(self.admission_manifest_sha256)
        _sha(self.ingress_boundary_sha256)
        _text(self.approval_reference)


@dataclass(frozen=True, init=False)
class ApprovedCampaignIngress:
    requirement: CampaignIngressRequirement = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, requirement, *, _token=None):
        if _token is not _SEAL or type(requirement) is not CampaignIngressRequirement:
            raise CampaignRuntimeError(
                "explicit independently approved server ingress configuration required"
            )
        object.__setattr__(self, "requirement", requirement)
        object.__setattr__(self, "_seal", _SEAL)


def approve_campaign_ingress(requirement):
    """Trusted server conditional-config seam; performs no native approval/read."""
    return ApprovedCampaignIngress(requirement, _token=_SEAL)


@dataclass(frozen=True)
class CampaignIngressObservation:
    admission_manifest_sha256: str
    ingress_boundary_sha256: str
    mode: str
    transport_session_id: str
    request_sha256: str
    observed_at: float
    expires_at: float
    process_id: int
    reader_binding: object = field(repr=False, compare=False)

    def validate_fields(self):
        try:
            for name in ("admission_manifest_sha256", "ingress_boundary_sha256", "request_sha256"):
                _sha(getattr(self, name))
            _text(self.mode)
            _text(self.transport_session_id)
        except ValueError:
            raise CampaignRuntimeError("closed private ingress fields required") from None


@dataclass(frozen=True)
class CampaignPlanningRef:
    kind: str
    reference: str
    sha256: str

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in ("research", "spec"):
            raise CampaignRuntimeError("closed planning data kind required")
        _text(self.reference, 128)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,127}", self.reference):
            raise CampaignRuntimeError("opaque planning data reference required")
        _sha(self.sha256)


@dataclass(frozen=True)
class CampaignTerminalVerification:
    operation_digest: str
    job_arn: str
    request_sha256: str
    source_binding_sha256: str
    status: str
    artifact_sha256: str
    verified: bool

    def validate_fields(self):
        try:
            for name in ("operation_digest", "request_sha256", "source_binding_sha256", "artifact_sha256"):
                _sha(getattr(self, name))
            _text(self.job_arn, 2048)
            if type(self.status) is not str or self.status not in ("Completed", "Failed", "Stopped"):
                raise ValueError
            if type(self.verified) is not bool:
                raise ValueError
        except ValueError:
            raise CampaignRuntimeError("closed original terminal verification fields required") from None


class CampaignIngressReader(Protocol):
    def read_current(self, *, request_sha256: str, purpose: str) -> CampaignIngressObservation: ...


class CampaignPlanningReader(Protocol):
    def load_original_refs(
        self, *, campaign_ref: str, private_store_correlation: str
    ) -> tuple[CampaignPlanningRef, ...]: ...


class CampaignTerminalVerifier(Protocol):
    def verify_terminal_original(
        self, original: OriginalCampaignOperation, *, policy: ApprovalPolicy | None
    ) -> CampaignTerminalVerification: ...


@dataclass(frozen=True, init=False)
class CampaignEntryContext:
    campaign_ref: str
    campaign_id: str
    private_store_correlation: str
    transport_session_id: str
    literal_prompt: str = field(repr=False)
    prompt_sha256: str
    mode: str
    planning_refs: tuple[CampaignPlanningRef, ...]
    native_admission: bool
    action_approved: bool
    _factory: object = field(repr=False, compare=False)
    _request_sha256: str = field(repr=False)
    _purpose: str = field(repr=False)
    _pid: int = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, factory, initialized, record, observation, planning_refs, purpose, *, _token=None):
        if _token is not _SEAL or type(factory) is not CampaignRuntimeFactory:
            raise CampaignRuntimeError("sealed server-created original entry required")
        if type(initialized) is not InitializationObservation or type(record) is not CampaignRecord:
            raise CampaignRuntimeError("protected original initialization/record required")
        for name in (
            "campaign_ref",
            "campaign_id",
            "private_store_correlation",
            "literal_prompt",
            "prompt_sha256",
        ):
            if getattr(initialized, name) != getattr(record, name):
                raise CampaignRuntimeError("protected original identity/record differs")
            object.__setattr__(self, name, getattr(initialized, name))
        for name, value in (
            ("transport_session_id", observation.transport_session_id),
            ("mode", observation.mode),
            ("planning_refs", planning_refs),
            ("native_admission", False),
            ("action_approved", False),
            ("_factory", factory),
            ("_request_sha256", observation.request_sha256),
            ("_purpose", purpose),
            ("_pid", os.getpid()),
            ("_seal", _SEAL),
        ):
            object.__setattr__(self, name, value)

    def _check(self):
        if self._seal is not _SEAL or self._pid != os.getpid():
            raise CampaignRuntimeError("foreign/inherited private entry refused")

    def require_current(self):
        self._check()
        return self._factory._require_context(self)

    def qualify_current_io(self):
        """Fresh native/ingress/protection gate, no DDB recursion or grant return."""
        self._check()
        self._factory._qualify(
            self._request_sha256, self._purpose, transport=self.transport_session_id, mode=self.mode
        )

    def load_original_campaign(self):
        with self._factory._context_io(self):
            return self._factory.coordinator.load_campaign(self.require_current(), self.campaign_ref)

    def claim_original_lease(self, *, expected_revision, lease_seconds=60):
        with self._factory._context_io(self):
            return self._factory.coordinator.claim_lease(
                self.require_current(), self.campaign_ref, expected_revision, lease_seconds=lease_seconds
            )


@dataclass(frozen=True)
class CampaignWakeOutcome:
    disposition: str
    campaign_ref: str
    operation_digest: str
    job_arn: str
    native_status: str
    context: CampaignEntryContext | None = field(default=None, repr=False)
    native_admission: bool = field(default=False, init=False)
    action_approved: bool = field(default=False, init=False)


class CampaignRuntimeFactory:
    def __init__(
        self,
        *,
        guard,
        initializer,
        coordinator,
        approved_ingress,
        ingress_reader,
        private_observer,
        planning_reader=None,
        terminal_verifier=None,
        clock=time.monotonic,
    ):
        if (
            type(guard) is not AdmissionGuard
            or type(initializer) is not DynamoCampaignInitialization
            or type(coordinator) is not DynamoCampaignCoordinator
            or initializer._guard is not guard
            or coordinator.guard is not guard
            or type(approved_ingress) is not ApprovedCampaignIngress
            or approved_ingress._seal is not _SEAL
            or not callable(clock)
        ):
            raise CampaignRuntimeError(
                "current exact guard/protected initializer/coordinator/approved ingress required"
            )
        for port, method in ((ingress_reader, "read_current"), (private_observer, "observe_original")):
            if not callable(getattr(port, method, None)):
                raise CampaignRuntimeError("selected current private ingress/original observer required")
        if planning_reader is not None and not callable(getattr(planning_reader, "load_original_refs", None)):
            raise CampaignRuntimeError("selected private planning data reader required")
        if terminal_verifier is not None and not callable(
            getattr(terminal_verifier, "verify_terminal_original", None)
        ):
            raise CampaignRuntimeError("selected private terminal artifact verifier required")
        self._pid = os.getpid()
        self.guard, self.initializer, self.coordinator = guard, initializer, coordinator
        self._approved, self._ingress, self._issuer = approved_ingress, ingress_reader, ingress_reader
        self._required = self._requirements(approved_ingress.requirement)
        if self._required[0] != guard.approved.manifest_sha256:
            raise CampaignRuntimeError("approved ingress/native baseline differs")
        self._observer, self._planning, self._terminal = private_observer, planning_reader, terminal_verifier
        self._clock, self._clock_lock, self._highwater = clock, threading.Lock(), None
        cfg = initializer._configuration()  # retained independently qualified table scope
        self._table_pin = (cfg.account_id, cfg.region, cfg.table_name, cfg.table_arn, cfg.table_id)
        self._table_binding()

    @staticmethod
    def _requirements(requirement):
        if type(requirement) is not CampaignIngressRequirement:
            raise CampaignRuntimeError("closed independently approved ingress requirement required")
        requirement.__post_init__()
        return (
            requirement.admission_manifest_sha256,
            requirement.ingress_boundary_sha256,
            requirement.approval_reference,
        )

    def _table_binding(self):
        cfg = self.initializer._configuration()
        current = (cfg.account_id, cfg.region, cfg.table_name, cfg.table_arn, cfg.table_id)
        manifest = self.guard.approved.manifest
        if (
            current != self._table_pin
            or self.coordinator._table != cfg.table_name
            or (cfg.account_id, cfg.region) != (manifest.account_id, manifest.region)
        ):
            raise CampaignRuntimeError("coordinator/protected native table selection differs")

    @contextmanager
    def _io_scope(self, request_sha, purpose, *, transport=None, mode=None):
        def current():
            self._qualify(request_sha, purpose, transport=transport, mode=mode)

        with self.initializer.qualified_io(current), self.coordinator.qualified_io(current):
            yield
            current()

    def _context_io(self, context):
        context._check()
        if context._factory is not self:
            raise CampaignRuntimeError("same private runtime context required")
        return self._io_scope(
            context._request_sha256,
            context._purpose,
            transport=context.transport_session_id,
            mode=context.mode,
        )

    def _process(self):
        if self._pid != os.getpid():
            raise CampaignRuntimeError("inherited runtime factory requires independent construction")

    def _now(self):
        self._process()
        with self._clock_lock:
            value = self._clock()
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (self._highwater is not None and value < self._highwater)
            ):
                raise CampaignRuntimeError("finite nondecreasing private ingress clock required")
            self._highwater = value
            return value

    def _qualify(self, request_sha, purpose, *, transport=None, mode=None):
        self._process()
        self._table_binding()
        if (
            self._requirements(self._approved.requirement) != self._required
            or self._ingress is not self._issuer
        ):
            raise CampaignRuntimeError("original approved ingress configuration changed")
        before = self._now()
        try:
            observation = self._ingress.read_current(request_sha256=request_sha, purpose=purpose)
        except Exception:
            raise CampaignRuntimeError("current independently authenticated ingress unavailable") from None
        after = self._now()
        if type(observation) is not CampaignIngressObservation:
            raise CampaignRuntimeError("current ingress evidence unknown")
        observation.validate_fields()
        if (observation.admission_manifest_sha256, observation.ingress_boundary_sha256) != self._required[:2]:
            raise CampaignRuntimeError("current approved ingress/native boundary differs")
        if (
            observation.request_sha256 != request_sha
            or type(observation.process_id) is not int
            or observation.process_id != self._pid
            or observation.reader_binding is not self._issuer
        ):
            raise CampaignRuntimeError("foreign ingress issuer/request/process refused")
        allowed = ("new_prompt", "redelivery") if purpose == "prompt" else ("private_wake",)
        if type(observation.mode) is not str or observation.mode not in allowed:
            raise CampaignRuntimeError("independently verified intake/wake disposition required")
        if (transport is not None and observation.transport_session_id != transport) or (
            mode is not None and observation.mode != mode
        ):
            raise CampaignRuntimeError("original ingress correlation/disposition changed")
        DynamoCampaignCoordinator._inputs(observation.transport_session_id, "ingress")
        if (
            type(observation.observed_at) not in (int, float)
            or type(observation.expires_at) not in (int, float)
            or not math.isfinite(observation.observed_at)
            or not math.isfinite(observation.expires_at)
            or not before <= observation.observed_at <= after
            or not observation.observed_at < observation.expires_at <= observation.observed_at + 60
            or after >= observation.expires_at
        ):
            raise CampaignRuntimeError("current ingress deadline unsupported/expired")
        deadline = observation.expires_at
        witness = self.guard.verify()
        self.initializer._check(witness)
        self._table_binding()
        if self._requirements(self._approved.requirement) != self._required or self._now() >= deadline:
            raise CampaignRuntimeError("current ingress expired/drifted during native qualification")
        return observation, witness

    def _refs(self, initialized):
        if self._planning is None:
            return ()
        refs = self._planning.load_original_refs(
            campaign_ref=initialized.campaign_ref,
            private_store_correlation=initialized.private_store_correlation,
        )
        if (
            type(refs) is not tuple
            or len(refs) > 8
            or any(type(ref) is not CampaignPlanningRef for ref in refs)
        ):
            raise CampaignRuntimeError("closed bounded original planning DATA refs required")
        for ref in refs:
            ref.__post_init__()
        if len({(ref.kind, ref.reference) for ref in refs}) != len(refs):
            raise CampaignRuntimeError("original planning refs duplicate")
        return refs

    def _context(self, initialized, record, observed, purpose):
        self._qualify(
            observed.request_sha256, purpose, transport=observed.transport_session_id, mode=observed.mode
        )
        refs = self._refs(initialized)
        self._qualify(
            observed.request_sha256, purpose, transport=observed.transport_session_id, mode=observed.mode
        )
        return CampaignEntryContext(self, initialized, record, observed, refs, purpose, _token=_SEAL)

    def begin_prompt(self, prompt, *, policy=None):
        _policy(policy)
        if type(prompt) is not str or not prompt.strip():
            raise CampaignRuntimeError("literal nonempty original prompt required")
        request_sha = _digest({"prompt": prompt})
        observed, witness = self._qualify(request_sha, "prompt")
        with self._io_scope(
            request_sha, "prompt", transport=observed.transport_session_id, mode=observed.mode
        ):
            if observed.mode == "new_prompt":
                initialized = self.initializer.initialize(witness, observed.transport_session_id, prompt)
            else:
                initialized = self.initializer.resolve_original(
                    witness, observed.transport_session_id, literal_prompt=prompt
                )
            _, witness = self._qualify(
                request_sha, "prompt", transport=observed.transport_session_id, mode=observed.mode
            )
            record = self.coordinator.load_campaign(witness, initialized.campaign_ref)
            return self._context(initialized, record, observed, "prompt")

    def _require_context(self, context):
        self._process()
        if type(context) is not CampaignEntryContext or context._factory is not self:
            raise CampaignRuntimeError("same original private runtime context required")
        observed, witness = self._qualify(
            context._request_sha256,
            context._purpose,
            transport=context.transport_session_id,
            mode=context.mode,
        )
        with self._context_io(context):
            initialized = self.initializer.resolve_original(
                witness, observed.transport_session_id, literal_prompt=context.literal_prompt
            )
            if (
                initialized.campaign_ref,
                initialized.campaign_id,
                initialized.private_store_correlation,
                initialized.prompt_sha256,
            ) != (
                context.campaign_ref,
                context.campaign_id,
                context.private_store_correlation,
                context.prompt_sha256,
            ):
                raise CampaignRuntimeError("protected original context binding differs")
            return self._qualify(
                context._request_sha256,
                context._purpose,
                transport=context.transport_session_id,
                mode=context.mode,
            )[1]

    def refresh_original_planning(self, entry):
        """Return a new qualified original DATA view; never mutate the sealed entry."""
        self._process()
        if type(entry) is not CampaignEntryContext or entry._factory is not self:
            raise CampaignRuntimeError("same original factory planning entry required")
        entry._check()
        observed, witness = self._qualify(
            entry._request_sha256,
            entry._purpose,
            transport=entry.transport_session_id,
            mode=entry.mode,
        )
        with self._context_io(entry):
            initialized = self.initializer.resolve_original(
                witness, entry.transport_session_id, literal_prompt=entry.literal_prompt
            )
            if (
                initialized.campaign_ref,
                initialized.campaign_id,
                initialized.private_store_correlation,
                initialized.literal_prompt,
                initialized.prompt_sha256,
            ) != (
                entry.campaign_ref,
                entry.campaign_id,
                entry.private_store_correlation,
                entry.literal_prompt,
                entry.prompt_sha256,
            ):
                raise CampaignRuntimeError("protected original planning binding differs")
            _, witness = self._qualify(
                entry._request_sha256,
                entry._purpose,
                transport=entry.transport_session_id,
                mode=entry.mode,
            )
            record = self.coordinator.load_campaign(witness, initialized.campaign_ref)
            fresh = self._context(initialized, record, observed, entry._purpose)
            if (
                fresh.campaign_ref,
                fresh.campaign_id,
                fresh.private_store_correlation,
                fresh.literal_prompt,
                fresh.prompt_sha256,
                fresh.transport_session_id,
                fresh.mode,
                fresh._request_sha256,
                fresh._purpose,
            ) != (
                entry.campaign_ref,
                entry.campaign_id,
                entry.private_store_correlation,
                entry.literal_prompt,
                entry.prompt_sha256,
                entry.transport_session_id,
                entry.mode,
                entry._request_sha256,
                entry._purpose,
            ):
                raise CampaignRuntimeError("original planning context changed")
            fresh.require_current()
            return fresh

    def begin_wake(self, hint, *, policy=None):
        policy = _policy(policy)
        if type(hint) is not WakeHint:
            raise CampaignRuntimeError("exact closed original wake hint required")
        hint.__post_init__()
        from .campaign_coordinator import _WAKE

        request_sha = _digest({name: getattr(hint, name) for name in _WAKE})
        observed, witness = self._qualify(request_sha, "private_wake")
        with self._io_scope(
            request_sha, "private_wake", transport=observed.transport_session_id, mode=observed.mode
        ):
            initialized = self.initializer.resolve_original(witness, observed.transport_session_id)
            if hint.campaign_ref != initialized.campaign_ref:
                raise CampaignRuntimeError("private wake does not match protected original handoff")
            _, witness = self._qualify(
                request_sha, "private_wake", transport=observed.transport_session_id, mode=observed.mode
            )
            record = self.coordinator.load_campaign(witness, initialized.campaign_ref)
            operation = next(
                (
                    row.original
                    for row in record.operations
                    if row.original.operation_digest == hint.operation_digest
                ),
                None,
            )
            if operation is None:
                raise CampaignRuntimeError("wake cannot dispatch an unregistered operation")
            scope = record.binding
            if (hint.account_id, hint.region, hint.deployment_id, hint.kind, hint.job_arn) != (
                scope["account_id"],
                scope["region"],
                scope["deployment_id"],
                operation.kind,
                operation.job_arn,
            ):
                raise CampaignRuntimeError("full original wake scope/native identity differs")
            lease = self.coordinator.claim_lease(witness, record.campaign_ref, record.revision)
            _, witness = self._qualify(
                request_sha, "private_wake", transport=observed.transport_session_id, mode=observed.mode
            )
            result = self.coordinator.observe_wake(
                witness, lease, hint, self._observer, fresh_observation=True
            )
            disposition = "awaiting_original"
            if result.status == "Completed" and self._terminal is not None:
                self._qualify(
                    request_sha, "private_wake", transport=observed.transport_session_id, mode=observed.mode
                )
                try:
                    verification = self._terminal.verify_terminal_original(operation, policy=policy)
                except Exception:
                    # An unresolved artifact check cannot authorize a model turn.
                    verification = None
                if type(verification) is CampaignTerminalVerification:
                    verification.validate_fields()
                    expected = (
                        operation.operation_digest,
                        operation.job_arn,
                        operation.request_sha256,
                        operation.source_binding_sha256,
                        "Completed",
                    )
                    actual = (
                        verification.operation_digest,
                        verification.job_arn,
                        verification.request_sha256,
                        verification.source_binding_sha256,
                        verification.status,
                    )
                    if verification.verified is True and actual == expected:
                        _sha(verification.artifact_sha256)
                        disposition = "continue_original"
            _, witness = self._qualify(
                request_sha, "private_wake", transport=observed.transport_session_id, mode=observed.mode
            )
            context = None
            if disposition == "continue_original":
                record = self.coordinator.load_campaign(witness, record.campaign_ref)
                context = self._context(initialized, record, observed, "private_wake")
            return CampaignWakeOutcome(
                disposition, record.campaign_ref, hint.operation_digest, hint.job_arn, result.status, context
            )


def current_campaign_runtime():
    factory = _RUNTIME.get()
    if factory is not None:
        if type(factory) is not CampaignRuntimeFactory:
            raise CampaignRuntimeError("server-only typed runtime factory required")
        factory._process()
    return factory


def current_campaign_entry():
    context = _ENTRY.get()
    if context is not None:
        if type(context) is not CampaignEntryContext:
            raise CampaignRuntimeError("sealed private entry required")
        context._check()
    return context


@contextmanager
def activate_campaign_runtime(factory):
    if type(factory) is not CampaignRuntimeFactory:
        raise CampaignRuntimeError("server-only typed runtime factory required")
    factory._process()
    token = _RUNTIME.set(factory)
    try:
        yield factory
    finally:
        _RUNTIME.reset(token)


@contextmanager
def activate_campaign_entry(context):
    if type(context) is not CampaignEntryContext:
        raise CampaignRuntimeError("sealed private entry required")
    context._check()
    context.require_current()
    token = _ENTRY.set(context)
    try:
        yield context
    finally:
        _ENTRY.reset(token)


@contextmanager
def campaign_prompt_scope(prompt, *, policy=None):
    _policy(policy)
    context = current_campaign_entry()
    if context is not None:
        factory = current_campaign_runtime()
        if factory is not None and context._factory is not factory:
            raise CampaignRuntimeError("conflicting private runtime/entry factory refused")
        if prompt != context.literal_prompt:
            raise CampaignRuntimeError("original private prompt cannot be replaced")
        context.require_current()
        yield context
        return
    factory = current_campaign_runtime()
    if factory is None:
        yield None
        return
    context = factory.begin_prompt(prompt, policy=policy)
    with activate_campaign_entry(context):
        yield context
