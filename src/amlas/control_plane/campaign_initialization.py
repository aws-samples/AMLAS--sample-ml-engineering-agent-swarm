"""Atomic ORIGINAL initialization under separately qualified publication protection.

The protection wrapper is explicit trusted-server configuration, not a native IAM
attestation. A distinct Python client never proves credential isolation. Native
key protection, fixed publisher confinement, table identity/retention, permissions
and intake redelivery remain separate qualification gates. No ML effects or
application activation occur here; results contain data, not action grants.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

from .campaign_admission import AdmissionError, AdmissionGuard
from .campaign_coordinator import (
    _BINDING,
    DynamoCampaignCoordinator,
    _digest,
    _initial_campaign_item,
    _json,
    _keys,
    _parse,
    _reference,
    _sha,
    _uuid,
)

_SEAL = object()
_H_FIELDS = (
    "schema_version",
    "record_kind",
    "selector",
    "campaign_ref",
    "binding",
    "binding_sha256",
    "initial_campaign_body_sha256",
)
_CONFIG = (
    "account_id",
    "region",
    "deployment_id",
    "table_name",
    "table_arn",
    "table_id",
    "publisher_role_arn",
    "publisher_role_id",
    "protection_evidence_sha256",
    "retention_evidence_sha256",
    "initializer_code_sha256",
)


def _qualified_return(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._qualify_io()
        return result

    return call


class InitializationError(ValueError):
    """Unknown or inconsistent initialization/protection; no effect eligibility."""


class InitializationUnresolved(InitializationError):  # noqa: N818 - reviewed public contract name
    """Original transaction outcome remains unresolved; no mutation replay."""


@dataclass(frozen=True)
class InitializationProtectionConfig:
    account_id: str
    region: str
    deployment_id: str
    table_name: str
    table_arn: str
    table_id: str
    publisher_role_arn: str
    publisher_role_id: str
    protection_evidence_sha256: str
    retention_evidence_sha256: str
    initializer_code_sha256: str

    def __post_init__(self):
        _config_fields(self)


def _config_fields(value):
    if type(value) is not InitializationProtectionConfig:
        raise InitializationError("exact independently reviewed protection configuration required")
    try:
        if type(value.account_id) is not str or not re.fullmatch(r"[0-9]{12}", value.account_id):
            raise ValueError
        if type(value.region) is not str or not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-[0-9]+", value.region):
            raise ValueError
        _uuid(value.deployment_id)
        _uuid(value.table_id)
        if type(value.table_name) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{3,255}", value.table_name):
            raise ValueError
        if (
            type(value.table_arn) is not str
            or value.table_arn
            != f"arn:aws:dynamodb:{value.region}:{value.account_id}:table/{value.table_name}"
        ):
            raise ValueError
        if type(value.publisher_role_arn) is not str or not re.fullmatch(
            r"arn:aws:iam::" + value.account_id + r":role/[A-Za-z0-9+=,.@_/-]{1,512}",
            value.publisher_role_arn,
        ):
            raise ValueError
        if type(value.publisher_role_id) is not str or not re.fullmatch(
            r"AROA[A-Z0-9]{17}", value.publisher_role_id
        ):
            raise ValueError
        for name in ("protection_evidence_sha256", "retention_evidence_sha256", "initializer_code_sha256"):
            _sha(getattr(value, name))
    except (ValueError, TypeError, AttributeError):
        raise InitializationError("closed protection configuration differs") from None
    return {name: getattr(value, name) for name in _CONFIG}


@dataclass(frozen=True, init=False)
class ApprovedInitializationBoundary:
    configuration: InitializationProtectionConfig = field(repr=False)
    approval_reference: str
    configuration_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __init__(self, configuration, approval_reference, *, _token=None):
        if _token is not _SEAL:
            raise InitializationError("explicit separately approved server initialization boundary required")
        fields = _config_fields(configuration)
        if (
            type(approval_reference) is not str
            or not 1 <= len(approval_reference.encode()) <= 256
            or any(ord(c) < 32 for c in approval_reference)
        ):
            raise InitializationError("explicit independent protection approval reference required")
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "approval_reference", approval_reference)
        object.__setattr__(self, "configuration_sha256", _digest(fields))
        object.__setattr__(self, "_seal", _SEAL)


def approve_initialization_boundary(configuration, *, approval_reference):
    """Trusted server configuration seam, never native proof or public metadata.

    Root must independently qualify/approve the protection/publication boundary.
    This function performs no native verification and must not be model-exposed.
    """
    return ApprovedInitializationBoundary(configuration, approval_reference, _token=_SEAL)


@dataclass(frozen=True)
class InitializationProtectionEvidence:
    configuration: InitializationProtectionConfig
    observed_at: float
    expires_at: float
    process_id: int
    reader_binding: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class InitializationObservation:
    campaign_ref: str
    campaign_id: str
    private_store_correlation: str
    literal_prompt: str = field(repr=False)
    prompt_sha256: str
    current_revision: int
    native_admission: bool = field(default=False, init=False)
    action_approved: bool = field(default=False, init=False)


@dataclass(frozen=True)
class InitializationPlanningObservation:
    """Verified protected original publication DATA; no authority is persisted."""

    original: InitializationObservation
    initial_campaign_body_sha256: str
    initialization_publication_sha256: str
    native_admission: bool = field(default=False, init=False)
    action_approved: bool = field(default=False, init=False)

    def __post_init__(self):
        if type(self.original) is not InitializationObservation:
            raise InitializationError("exact original initialization DATA required")
        _sha(self.initial_campaign_body_sha256)
        _sha(self.initialization_publication_sha256)


class InitializationTransportError(InitializationError):
    """Retry refusal with historical data only; never current qualification or grant."""

    def __init__(self, confirmed_original):
        if confirmed_original is not None and type(confirmed_original) is not InitializationObservation:
            raise InitializationError("exact confirmed original observation required")
        self.confirmed_original = confirmed_original
        super().__init__(
            "write transport reported retries; historical original observed, currently unqualified, no write replay"
            if confirmed_original is not None
            else "write transport reported retries; original outcome unresolved, currently unqualified, no write replay"
        )

    @property
    def currently_qualified(self):
        return False

    @property
    def confirmed_original_is_historical(self):
        return self.confirmed_original is not None


def _item_size(item):
    """Closed S/B byte accounting and conservative N storage; not JSON wire size."""
    size = 0
    if type(item) is not dict:
        raise InitializationError("closed DynamoDB item required")
    for name, value in item.items():
        if type(name) is not str or type(value) is not dict or len(value) != 1:
            raise InitializationError("closed DynamoDB item attribute required")
        size += len(name.encode("utf-8"))
        kind, payload = next(iter(value.items()))
        if kind == "B" and type(payload) is bytes:
            size += len(payload)
        elif kind == "S" and type(payload) is str:
            size += len(payload.encode("utf-8"))
        elif kind == "N" and type(payload) is str and re.fullmatch(r"0|[1-9][0-9]{0,18}", payload):
            # DynamoDB number storage: approximately one byte per two significant
            # digits plus one byte; use the larger UTF8 count as a conservative cap.
            size += max(len(payload.encode()), (len(payload) + 1) // 2 + 1)
        else:
            raise InitializationError("unsupported DynamoDB item attribute")
    if size > 400 * 1024:
        raise InitializationError("DynamoDB item size exceeds native bound")
    return size


class DynamoCampaignInitialization:
    def __init__(
        self,
        guard,
        approved_boundary,
        *,
        private_transaction_client,
        boundary_reader=None,
        clock=time.monotonic,
    ):
        if (
            type(guard) is not AdmissionGuard
            or type(approved_boundary) is not ApprovedInitializationBoundary
            or approved_boundary._seal is not _SEAL
        ):
            raise InitializationError("separate approved admission and initialization boundary required")
        if not callable(clock) or not callable(getattr(boundary_reader, "read_current", None)):
            raise InitializationError("current independent protection reader required")
        self._process_id = os.getpid()
        self._io_qualification = ContextVar("amlas_private_io_qualification", default=None)
        self._guard, self._boundary = guard, approved_boundary
        self._ddb, self._boundary_reader, self._clock = private_transaction_client, boundary_reader, clock
        self._lock, self._clock_lock = threading.RLock(), threading.Lock()
        self._clock_highwater, self._pending, self._attempted = None, {}, set()
        self._reader_binding = boundary_reader
        fields = _config_fields(approved_boundary.configuration)
        self._approved_configuration_bytes = _json(fields)
        self._approved_configuration_sha256 = _digest(fields)
        self._configuration()
        self._transport()

    @contextmanager
    def qualified_io(self, current_qualification=None):
        self._process()
        if current_qualification is None:
            yield self
            return
        if not callable(current_qualification):
            raise InitializationError("private current qualification callable required")
        previous = self._io_qualification.get()

        def combined():
            if previous is not None:
                previous()
            current_qualification()

        token = self._io_qualification.set(combined)
        try:
            yield self
        finally:
            self._io_qualification.reset(token)

    def _qualify_io(self):
        self._process()
        current = self._io_qualification.get()
        if current is not None:
            current()

    def _process(self):
        if self._process_id != os.getpid():
            raise InitializationError("inherited initialization context requires independent construction")

    def _configuration(self):
        self._process()
        fields = _config_fields(self._boundary.configuration)
        if (
            _json(fields) != self._approved_configuration_bytes
            or _digest(fields) != self._approved_configuration_sha256
            or self._boundary.configuration_sha256 != self._approved_configuration_sha256
        ):
            raise InitializationError("approved protection configuration changed")
        return self._boundary.configuration

    def _transport(self):
        self._process()
        meta = getattr(self._ddb, "meta", None)
        retries = getattr(getattr(meta, "config", None), "retries", None)
        if (
            type(retries) is not dict
            or type(retries.get("total_max_attempts")) is not int
            or retries["total_max_attempts"] != 1
        ):
            raise InitializationError("initialization transport requires exactly one wire attempt")
        for name in ("transact_get_items", "transact_write_items"):
            if not callable(getattr(self._ddb, name, None)):
                raise InitializationError("private fixed transaction transport required")

    def _now(self):
        self._process()
        with self._clock_lock:
            value = self._clock()
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (self._clock_highwater is not None and value < self._clock_highwater)
            ):
                raise InitializationError("finite nondecreasing initialization clock required")
            self._clock_highwater = value
            return value

    def _check(self, witness):
        self._process()
        self._guard.require(witness)
        cfg = self._configuration()
        manifest = self._guard.approved.manifest
        if (cfg.account_id, cfg.region, cfg.deployment_id) != (
            manifest.account_id,
            manifest.region,
            manifest.deployment_id,
        ):
            raise InitializationError("initialization protection scope differs from admitted scope")
        if self._boundary_reader is not self._reader_binding:
            raise InitializationError("configured private protection reader changed")
        before = self._now()
        try:
            current = self._boundary_reader.read_current()
        except Exception:
            raise InitializationError(
                "current independently qualified protection evidence unavailable"
            ) from None
        after = self._now()
        if (
            type(current) is not InitializationProtectionEvidence
            or type(current.configuration) is not InitializationProtectionConfig
            or type(current.process_id) is not int
            or current.process_id != self._process_id
            or current.reader_binding is not self._reader_binding
        ):
            raise InitializationError("current protection evidence is unknown")
        if _config_fields(current.configuration) != _config_fields(cfg):
            raise InitializationError("initializer/table/protection/retention evidence drifted")
        if (
            type(current.observed_at) not in (int, float)
            or type(current.expires_at) not in (int, float)
            or not math.isfinite(current.observed_at)
            or not math.isfinite(current.expires_at)
            or not before <= current.observed_at <= after
            or not current.observed_at < current.expires_at <= current.observed_at + 60
            or after >= current.expires_at
        ):
            raise InitializationError("current protection evidence expired or unsupported")
        deadline = current.expires_at
        self._process()
        self._guard.require(witness)
        cfg = self._configuration()
        if _config_fields(current.configuration) != _config_fields(cfg):
            raise InitializationError("current protection changed during final admission observation")
        self._transport()
        if self._now() >= deadline:
            raise InitializationError("current protection expired during final admission observation")
        return cfg

    def _scope(self, witness):
        self._check(witness)
        approved = self._guard.approved
        m = approved.manifest
        return {
            "account_id": m.account_id,
            "region": m.region,
            "deployment_id": m.deployment_id,
            "deployment_prefix": m.deployment_prefix,
            "owner_namespace": m.owner_namespace,
            "baseline_sha256": approved.manifest_sha256,
            "runtime_arn": m.runtime_arn,
            "endpoint_arn": m.endpoint_arn,
            "endpoint_name": m.endpoint_name,
            "runtime_version": m.runtime_version,
        }

    def _read_pair(self, witness, selector):
        cfg = self._configuration()
        kwargs = {
            "TransactItems": [
                {"Get": {"TableName": cfg.table_name, "Key": {"job_name": {"S": prefix + selector}}}}
                for prefix in ("CAMPAIGN#", "CAMPAIGN_HANDOFF#")
            ]
        }
        try:
            from botocore.session import Session
            from botocore.validate import validate_parameters

            validate_parameters(
                kwargs,
                Session().get_service_model("dynamodb").operation_model("TransactGetItems").input_shape,
            )
        except Exception:
            raise InitializationUnresolved("transactional original pair request malformed") from None
        self._check(witness)
        self._qualify_io()
        try:
            reply = self._ddb.transact_get_items(**kwargs)
        except Exception:
            raise InitializationUnresolved("transactional original pair read unavailable") from None
        self._qualify_io()
        if type(reply) is not dict or set(reply) - {"Responses", "ConsumedCapacity", "ResponseMetadata"}:
            raise InitializationUnresolved("transactional original pair response malformed")
        metadata = reply.get("ResponseMetadata")
        if (
            type(metadata) is not dict
            or type(metadata.get("HTTPStatusCode")) is not int
            or metadata["HTTPStatusCode"] != 200
            or type(metadata.get("RetryAttempts")) is not int
            or metadata["RetryAttempts"] != 0
        ):
            raise InitializationUnresolved("transactional read response metadata unavailable")
        slots = reply.get("Responses")
        if type(slots) is not list or len(slots) != 2:
            raise InitializationUnresolved("transactional pair population differs")
        result = []
        for slot in slots:
            if type(slot) is not dict or set(slot) - {"Item"}:
                raise InitializationUnresolved("transactional pair slot malformed")
            if "Item" in slot and type(slot["Item"]) is not dict:
                raise InitializationUnresolved("reported transaction item is malformed, not absent")
            result.append(slot.get("Item"))
        return tuple(result)

    def _resolve(self, scope, selector, pair, literal_prompt):
        c, h = pair
        if c is None and h is None:
            return None
        if c is None or h is None:
            raise InitializationError(
                "initialized deletion or unqualified orphan cannot be recreated/adopted"
            )
        try:
            _keys(h, ("job_name", "private_record"))
            _keys(h["job_name"], ("S",))
            _keys(h["private_record"], ("B",))
            if h["job_name"]["S"] != "CAMPAIGN_HANDOFF#" + selector:
                raise ValueError
            _item_size(h)
            body = _keys(_parse(h["private_record"]["B"]), _H_FIELDS)
            if (
                type(body["schema_version"]) is not int
                or body["schema_version"] != 1
                or body["record_kind"] != "campaign-initialization-v1"
                or body["selector"] != selector
            ):
                raise ValueError
            binding = _keys(body["binding"], _BINDING)
            selected, reference, initial = _initial_campaign_item(
                scope,
                binding["transport_session_id"],
                binding["literal_prompt"],
                binding["campaign_id"],
                binding["private_store_correlation"],
            )
            if (
                selected != selector
                or reference != body["campaign_ref"]
                or binding != _parse(initial["private_record"]["B"])["binding"]
            ):
                raise ValueError
            expected = _reference(reference)[1]
            if (
                body["binding_sha256"] != expected
                or body["initial_campaign_body_sha256"]
                != hashlib.sha256(initial["private_record"]["B"]).hexdigest()
            ):
                raise ValueError
            _item_size(c)
            record = DynamoCampaignCoordinator._body(c, "CAMPAIGN#" + selector, scope, expected)[1]
            if record.binding != binding or (
                literal_prompt is not None and record.literal_prompt != literal_prompt
            ):
                raise ValueError
            return InitializationObservation(
                record.campaign_ref,
                record.campaign_id,
                record.private_store_correlation,
                record.literal_prompt,
                record.prompt_sha256,
                record.revision,
            )
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError):
            raise InitializationError("protected original handoff/campaign integrity differs") from None

    @_qualified_return
    def resolve_original(self, witness, transport_session_id, *, literal_prompt=None):
        self._process()
        with self._lock:
            scope = self._scope(witness)
            DynamoCampaignCoordinator._inputs(
                transport_session_id, literal_prompt if literal_prompt is not None else "resolution"
            )
            selector = _digest({**scope, "transport_session_id": transport_session_id})
            result = self._resolve(scope, selector, self._read_pair(witness, selector), literal_prompt)
            if result is None:
                raise InitializationUnresolved("no committed original initialization visible")
            self._check(witness)
            return result

    def resolve_original_planning(self, witness, transport_session_id, *, literal_prompt=None, qualify_io):
        """One freshly qualified original C/H read; commitments are DATA only."""
        self._process()
        if not callable(qualify_io):
            raise InitializationError("current private planning qualification required")

        def current():
            if qualify_io() is not None:
                raise InitializationError("planning qualifier returns no permission value")

        with self.qualified_io(current), self._lock:
            current()
            scope = self._scope(witness)
            DynamoCampaignCoordinator._inputs(
                transport_session_id, literal_prompt if literal_prompt is not None else "resolution"
            )
            selector = _digest({**scope, "transport_session_id": transport_session_id})
            pair = self._read_pair(witness, selector)
            original = self._resolve(scope, selector, pair, literal_prompt)
            if original is None:
                raise InitializationUnresolved("no committed original planning initialization visible")
            # _resolve already verified the protected H schema and original initial-C
            # reconstruction. The mutable current C revision is not this commitment.
            raw_h = pair[1]["private_record"]["B"]
            h = _parse(raw_h)
            answer = InitializationPlanningObservation(
                original,
                h["initial_campaign_body_sha256"],
                hashlib.sha256(raw_h).hexdigest(),
            )
            self._check(witness)
            current()
            return answer

    @_qualified_return
    def initialize(self, witness, transport_session_id, literal_prompt):
        self._process()
        with self._lock:
            scope = self._scope(witness)
            DynamoCampaignCoordinator._inputs(transport_session_id, literal_prompt)
            selector = _digest({**scope, "transport_session_id": transport_session_id})
            existing = self._resolve(scope, selector, self._read_pair(witness, selector), literal_prompt)
            if existing is not None:
                self._pending.pop(selector, None)
                self._check(witness)
                return existing
            if selector in self._attempted:
                raise InitializationUnresolved(
                    "original transaction outcome remains unresolved; no write replay"
                )
            if selector not in self._pending:
                selected, reference, campaign = _initial_campaign_item(
                    scope, transport_session_id, literal_prompt, str(uuid.uuid4()), str(uuid.uuid4())
                )
                binding = _parse(campaign["private_record"]["B"])["binding"]
                handoff_body = {
                    "schema_version": 1,
                    "record_kind": "campaign-initialization-v1",
                    "selector": selected,
                    "campaign_ref": reference,
                    "binding": binding,
                    "binding_sha256": _reference(reference)[1],
                    "initial_campaign_body_sha256": hashlib.sha256(
                        campaign["private_record"]["B"]
                    ).hexdigest(),
                }
                handoff = {
                    "job_name": {"S": "CAMPAIGN_HANDOFF#" + selected},
                    "private_record": {"B": _json(handoff_body)},
                }
                if sum(_item_size(item) for item in (campaign, handoff)) > 4 * 1024 * 1024:
                    raise InitializationError("transaction aggregate size exceeds native bound")
                self._pending[selector] = (binding["campaign_id"], campaign, handoff)
            token, campaign, handoff = self._pending[selector]
            candidate = self._resolve(scope, selector, (campaign, handoff), literal_prompt)
            if candidate is None or token != candidate.campaign_id:
                raise InitializationError("retained candidate differs from exact current initialization")
            cfg = self._configuration()
            kwargs = {
                "TransactItems": [
                    {
                        "Put": {
                            "TableName": cfg.table_name,
                            "Item": item,
                            "ConditionExpression": "attribute_not_exists(job_name)",
                        }
                    }
                    for item in (campaign, handoff)
                ],
                "ClientRequestToken": token,
            }
            unresolved_message = "original transaction not visible; retained candidate unresolved"
            try:
                from botocore.session import Session
                from botocore.validate import validate_parameters

                validate_parameters(
                    kwargs,
                    Session().get_service_model("dynamodb").operation_model("TransactWriteItems").input_shape,
                )
            except Exception:
                raise InitializationError("original transaction request malformed") from None
            self._check(witness)
            self._qualify_io()
            self._attempted.add(selector)
            retry_violation = False
            try:
                acknowledgement = self._ddb.transact_write_items(**kwargs)
                if type(acknowledgement) is dict:
                    metadata = acknowledgement.get("ResponseMetadata")
                    if type(metadata) is dict and "RetryAttempts" in metadata:
                        retries = metadata["RetryAttempts"]
                        retry_violation = type(retries) is not int or retries != 0
            except Exception:
                # One bounded outcome read handles lost acknowledgement/conflicting
                # candidates. Never loop or infer uncommitted originals from absence.
                unresolved_message = (
                    "original transaction acknowledgement unavailable; retained candidate unresolved"
                )
            self._qualify_io()
            try:
                result = self._resolve(scope, selector, self._read_pair(witness, selector), literal_prompt)
            except InitializationError as error:
                if retry_violation:
                    raise InitializationTransportError(None) from error
                raise
            if result is None:
                if retry_violation:
                    raise InitializationTransportError(None)
                raise InitializationUnresolved(unresolved_message)
            try:
                self._check(witness)
            except (InitializationError, AdmissionError) as error:
                if retry_violation:
                    raise InitializationTransportError(result) from error
                raise
            if retry_violation:
                raise InitializationTransportError(result)
            self._pending.pop(selector, None)
            return result
