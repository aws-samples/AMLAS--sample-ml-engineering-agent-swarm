"""Closed fixed co-owner admission comparison; no identity or policy mutation.

Approved configuration comes from trusted server code. A native observation is
untrusted metadata, not operator approval. Native reader/baseline qualification
and effective IAM admission remain deployment gates. These sequential reads do
not form an atomic IAM snapshot or fence administrative changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

_SEAL = object()
_ACCOUNT = re.compile(r"[0-9]{12}")
_REGION = re.compile(r"[a-z]{2}(?:-[a-z]+)+-[0-9]+")
_PREFIX = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
_VERSION = re.compile(r"[1-9][0-9]{0,4}")
_ROLE = re.compile(r"arn:(aws|aws-cn|aws-us-gov):iam::([0-9]{12}):role/([A-Za-z0-9+=,.@_/-]{1,512})")
_RUNTIME = re.compile(
    r"arn:(aws|aws-cn|aws-us-gov):bedrock-agentcore:([a-z0-9-]+):([0-9]{12}):runtime/"
    r"[a-zA-Z][a-zA-Z0-9_]{0,47}-[a-zA-Z0-9]{10}"
)
_ENDPOINT_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}")
_SHA = re.compile(r"[a-f0-9]{64}")


class AdmissionError(ValueError):
    """Missing, unapproved, expired or changed admission evidence."""


def _text(value: Any, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise AdmissionError("closed bounded admission text required")
    try:
        raw = value.encode("utf-8")
    except UnicodeError as error:
        raise AdmissionError("valid admission UTF8 required") from error
    if len(raw) > maximum * 4 or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise AdmissionError("closed bounded admission text required")
    return value


def _uuid(value: Any) -> str:
    _text(value, maximum=36)
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError) as error:
        raise AdmissionError("canonical admission UUID required") from error
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AdmissionError("duplicate policy field")
        result[key] = value
    return result


def _bad_constant(_):
    raise AdmissionError("nonfinite policy data refused")


def _depth(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise AdmissionError("bounded policy depth required")
    if type(value) is dict:
        if len(value) > 2048:
            raise AdmissionError("bounded policy object required")
        for child in value.values():
            _depth(child, depth + 1)
    elif type(value) is list:
        if len(value) > 4096:
            raise AdmissionError("bounded policy array required")
        for child in value:
            _depth(child, depth + 1)
    elif type(value) is float and not math.isfinite(value):
        raise AdmissionError("nonfinite parsed policy number refused")


def _policy(raw: Any, digest: Any) -> str:
    if type(raw) is not bytes or not 1 <= len(raw) <= 20480:
        raise AdmissionError("missing or unsupported exact native policy bytes")
    if type(digest) is not str or not _SHA.fullmatch(digest):
        raise AdmissionError("exact policy SHA required")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise AdmissionError("policy bytes differ from declared SHA")
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_bad_constant)
        if type(value) is not dict:
            raise AdmissionError("native policy must be JSON object bytes")
        _depth(value)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise AdmissionError("unsupported native policy JSON bytes") from error
    return text


@dataclass(frozen=True)
class RoleIncarnation:
    role_arn: str
    role_id: str
    trust_policy: bytes = field(repr=False)
    trust_sha256: str

    def __post_init__(self):
        _role_fields(self)


def _role_fields(value: Any, account: str | None = None, partition: str | None = None) -> dict:
    if type(value) is not RoleIncarnation:
        raise AdmissionError("exact role-incarnation metadata required")
    match = _ROLE.fullmatch(_text(value.role_arn, maximum=2048))
    if not match or (account is not None and match[2] != account):
        raise AdmissionError("role incarnation has foreign account or unsupported ARN")
    if partition is not None and match[1] != partition:
        raise AdmissionError("role incarnation has foreign partition")
    if type(value.role_id) is not str or not re.fullmatch(r"AROA[A-Z0-9]{17}", value.role_id):
        raise AdmissionError("exact native IAM role unique ID required")
    return {
        "role_arn": value.role_arn,
        "role_id": value.role_id,
        "trust_policy": _policy(value.trust_policy, value.trust_sha256),
        "trust_sha256": value.trust_sha256,
    }


@dataclass(frozen=True)
class AdmissionObservation:
    account_id: str
    region: str
    deployment_id: str
    deployment_prefix: str
    runtime_arn: str
    runtime_execution_role: RoleIncarnation
    endpoint_arn: str
    endpoint_name: str
    runtime_version: str
    endpoint_live_version: str
    endpoint_target_version: str | None
    runtime_policy: bytes = field(repr=False)
    runtime_policy_sha256: str
    endpoint_policy: bytes = field(repr=False)
    endpoint_policy_sha256: str
    co_owner_roles: tuple[RoleIncarnation, ...]


@dataclass(frozen=True)
class AdmissionManifest:
    account_id: str
    region: str
    deployment_id: str
    deployment_prefix: str
    runtime_arn: str
    runtime_execution_role: RoleIncarnation
    endpoint_arn: str
    endpoint_name: str
    runtime_version: str
    endpoint_live_version: str
    endpoint_target_version: str | None
    runtime_policy: bytes = field(repr=False)
    runtime_policy_sha256: str
    endpoint_policy: bytes = field(repr=False)
    endpoint_policy_sha256: str
    co_owner_roles: tuple[RoleIncarnation, ...]
    owner_namespace: str
    change_control_reference: str

    def __post_init__(self):
        _manifest_fields(self)


def _native_fields(value: Any) -> dict:
    if type(value) not in (AdmissionManifest, AdmissionObservation):
        raise AdmissionError("exact closed native admission metadata required")
    if type(value.account_id) is not str or not _ACCOUNT.fullmatch(value.account_id):
        raise AdmissionError("selected account ID required")
    if type(value.region) is not str or not _REGION.fullmatch(value.region):
        raise AdmissionError("selected Region required")
    _uuid(value.deployment_id)
    if type(value.deployment_prefix) is not str or not _PREFIX.fullmatch(value.deployment_prefix):
        raise AdmissionError("closed deployment prefix required")
    matched = _RUNTIME.fullmatch(_text(value.runtime_arn, maximum=2048))
    if not matched or (matched[2], matched[3]) != (value.region, value.account_id):
        raise AdmissionError("selected native Runtime ARN required")
    if type(value.endpoint_name) is not str or not _ENDPOINT_NAME.fullmatch(value.endpoint_name):
        raise AdmissionError("selected endpoint name required")
    if (
        type(value.endpoint_arn) is not str
        or value.endpoint_arn != value.runtime_arn + "/runtime-endpoint/" + value.endpoint_name
    ):
        raise AdmissionError("selected exact qualified endpoint ARN required")
    if type(value.runtime_version) is not str or not _VERSION.fullmatch(value.runtime_version):
        raise AdmissionError("selected native runtime version required")
    if type(value.endpoint_live_version) is not str or value.endpoint_live_version != value.runtime_version:
        raise AdmissionError("native endpoint liveVersion differs from selected version")
    if value.endpoint_target_version is not None and (
        type(value.endpoint_target_version) is not str
        or value.endpoint_target_version != value.runtime_version
    ):
        raise AdmissionError("native endpoint has a differing target version")
    if type(value.co_owner_roles) is not tuple or not 1 <= len(value.co_owner_roles) <= 16:
        raise AdmissionError("explicit bounded co-owner role cohort required")
    roles = [_role_fields(role, value.account_id, matched[1]) for role in value.co_owner_roles]
    if len({role["role_arn"] for role in roles}) != len(roles) or len(
        {role["role_id"] for role in roles}
    ) != len(roles):
        raise AdmissionError("duplicate role incarnation refused")
    return {
        "account_id": value.account_id,
        "region": value.region,
        "deployment_id": value.deployment_id,
        "deployment_prefix": value.deployment_prefix,
        "runtime_arn": value.runtime_arn,
        "runtime_execution_role": _role_fields(value.runtime_execution_role, value.account_id, matched[1]),
        "endpoint_arn": value.endpoint_arn,
        "endpoint_name": value.endpoint_name,
        "runtime_version": value.runtime_version,
        "endpoint_live_version": value.endpoint_live_version,
        "endpoint_target_version": value.endpoint_target_version,
        "runtime_policy": _policy(value.runtime_policy, value.runtime_policy_sha256),
        "runtime_policy_sha256": value.runtime_policy_sha256,
        "endpoint_policy": _policy(value.endpoint_policy, value.endpoint_policy_sha256),
        "endpoint_policy_sha256": value.endpoint_policy_sha256,
        "co_owner_roles": roles,
    }


def _manifest_fields(value: Any) -> dict:
    if type(value) is not AdmissionManifest:
        raise AdmissionError("exact unapproved admission manifest required")
    return {
        "schema_version": 1,
        **_native_fields(value),
        "owner_namespace": _uuid(value.owner_namespace),
        "change_control_reference": _text(value.change_control_reference, maximum=256),
    }


def _manifest_bytes(value: AdmissionManifest) -> bytes:
    try:
        raw = json.dumps(
            _manifest_fields(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeError, ValueError, RecursionError) as error:
        raise AdmissionError("closed manifest encoding failed") from error
    if len(raw) > 256 * 1024:
        raise AdmissionError("bounded approved manifest required")
    return raw


@dataclass(frozen=True, init=False)
class ApprovedAdmissionBaseline:
    manifest: AdmissionManifest = field(repr=False)
    manifest_sha256: str
    approval_reference: str
    _seal: object = field(repr=False, compare=False)

    def __init__(self, manifest, approval_reference, *, _token=None):
        if _token is not _SEAL:
            raise AdmissionError("explicit trusted-server baseline approval required")
        raw = _manifest_bytes(manifest)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "manifest_sha256", hashlib.sha256(raw).hexdigest())
        object.__setattr__(self, "approval_reference", _text(approval_reference, maximum=256))
        object.__setattr__(self, "_seal", _SEAL)


def approve_manifest(manifest: AdmissionManifest, *, approval_reference: str) -> ApprovedAdmissionBaseline:
    """Explicit server-config seam; never infer approval from observation or JSON.

    Deployment code must supply separately operator-approved native configuration.
    This function makes no native read and does not attest approval provenance.
    It is intentionally not an agent tool or public-payload parser.
    """
    return ApprovedAdmissionBaseline(manifest, approval_reference, _token=_SEAL)


@dataclass(frozen=True, init=False)
class AdmissionWitness:
    owner_namespace: str
    baseline_sha256: str
    process_id: int
    issued_at: float
    expires_at: float
    _issuer: object = field(repr=False, compare=False)
    _bound_times: tuple = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _token=None, **fields):
        if _token is not _SEAL:
            raise AdmissionError("current process-private admission witness required")
        for name in (
            "owner_namespace",
            "baseline_sha256",
            "process_id",
            "issued_at",
            "expires_at",
            "_issuer",
        ):
            object.__setattr__(self, name, fields[name])
        object.__setattr__(self, "_bound_times", (fields["issued_at"], fields["expires_at"]))
        object.__setattr__(self, "_seal", _SEAL)


class AdmissionGuard:
    def __init__(self, approved, reader, *, clock=time.monotonic, max_age_seconds=30):
        if (
            type(max_age_seconds) not in (int, float)
            or not math.isfinite(max_age_seconds)
            or not 0 < max_age_seconds <= 60
        ):
            raise AdmissionError("admission witness age must be finite within sixty seconds")
        if not callable(clock):
            raise AdmissionError("private admission clock required")
        self._approved, self._reader, self._clock = approved, reader, clock
        self._max_age = max_age_seconds
        self._issuer = object()
        self._process_id = os.getpid()
        self._clock_lock = threading.Lock()
        self._clock_highwater = None

    def _require_process(self):
        if os.getpid() != self._process_id:
            raise AdmissionError("inherited admission guard requires independent construction")

    @property
    def approved(self) -> ApprovedAdmissionBaseline:
        self._require_process()
        approved = self._approved
        if type(approved) is not ApprovedAdmissionBaseline or approved._seal is not _SEAL:
            raise AdmissionError("missing or unapproved server admission baseline")
        if hashlib.sha256(_manifest_bytes(approved.manifest)).hexdigest() != approved.manifest_sha256:
            raise AdmissionError("approved manifest bytes changed")
        return approved

    def _now(self) -> float:
        self._require_process()
        # Numeric process-local observation only; never a witness/grant cache.
        with self._clock_lock:
            try:
                value = self._clock()
            except Exception as error:
                raise AdmissionError("current admission clock unavailable") from error
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise AdmissionError("finite current admission clock required")
            if self._clock_highwater is not None and value < self._clock_highwater:
                raise AdmissionError("admission clock moved below observed high-water mark")
            self._clock_highwater = value
            return value

    def _compare_current(self):
        self._require_process()
        approved = self.approved
        try:
            observed = self._reader.read_current()
        except Exception as error:
            raise AdmissionError("current native admission evidence unavailable") from error
        if type(observed) is not AdmissionObservation or _native_fields(observed) != _native_fields(
            approved.manifest
        ):
            raise AdmissionError("current native admission differs from approved baseline")

    def verify(self) -> AdmissionWitness:
        self._require_process()
        approved = self.approved
        pid, issued = os.getpid(), self._now()
        self._compare_current()
        after = self._now()
        if os.getpid() != pid or after < issued or after >= issued + self._max_age:
            raise AdmissionError("current admission verification expired or changed process")
        return AdmissionWitness(
            _token=_SEAL,
            owner_namespace=approved.manifest.owner_namespace,
            baseline_sha256=approved.manifest_sha256,
            process_id=pid,
            issued_at=issued,
            expires_at=issued + self._max_age,
            _issuer=self._issuer,
        )

    def _check_witness(self, witness):
        approved = self.approved
        if (
            type(witness) is not AdmissionWitness
            or witness._seal is not _SEAL
            or witness._issuer is not self._issuer
            or type(witness.process_id) is not int
            or witness.process_id != os.getpid()
            or witness.baseline_sha256 != approved.manifest_sha256
            or witness.owner_namespace != approved.manifest.owner_namespace
        ):
            raise AdmissionError("foreign or inherited admission witness refused")
        if (
            type(witness.issued_at) not in (int, float)
            or type(witness.expires_at) not in (int, float)
            or not math.isfinite(witness.issued_at)
            or not math.isfinite(witness.expires_at)
            or witness.issued_at < 0
            or witness.expires_at != witness.issued_at + self._max_age
            or (witness.issued_at, witness.expires_at) != witness._bound_times
        ):
            raise AdmissionError("sealed witness interval fields changed")
        now = self._now()
        if now < witness.issued_at or now >= witness.expires_at:
            raise AdmissionError("admission witness expired or clock moved backwards")

    def require(self, witness: AdmissionWitness) -> None:
        self._require_process()
        self._check_witness(witness)
        self._compare_current()
        self._check_witness(witness)
