"""Closed campaign mapping and fenced observations; no native dispatch or admission grant."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

_MAX = 262144
_PHASE_KIND = {
    "dataset_acquisition": "Processing",
    "model_acquisition": "Processing",
    "case_freeze": "Processing",
    "preparation": "Processing",
    "sft": "Training",
    "reload": "Training",
    "generation_base": "Training",
    "generation_tuned": "Training",
    "paired_evaluation": "Processing",
}
_NATIVE = {"InProgress", "Stopping", "Completed", "Failed", "Stopped"}
_TERMINAL = {"Completed", "Failed", "Stopped"}
_OPERATION = (
    "operation_digest",
    "phase",
    "kind",
    "account_id",
    "region",
    "job_name",
    "job_arn",
    "request_sha256",
    "source_binding_sha256",
)
_WAKE = (
    "schema_version",
    "event_id",
    "campaign_ref",
    "account_id",
    "region",
    "deployment_id",
    "kind",
    "operation_digest",
    "job_arn",
)
_BINDING = (
    "account_id",
    "region",
    "deployment_id",
    "deployment_prefix",
    "owner_namespace",
    "baseline_sha256",
    "runtime_arn",
    "endpoint_arn",
    "endpoint_name",
    "runtime_version",
    "transport_session_id",
    "campaign_id",
    "private_store_correlation",
    "literal_prompt",
    "prompt_sha256",
)
_SCOPE = _BINDING[:10]
_BODY = (
    "schema_version",
    "binding",
    "binding_sha256",
    "revision",
    "lease_epoch",
    "lease_holder",
    "lease_until_ms",
    "operations",
)
_ROW = (
    "original",
    "status",
    "last_confirmed_native_status",
    "last_event_id",
    "last_event_digest",
    "observation_digest",
    "observed_at_ms",
)
_LEASE_SEAL = object()


def _qualified_return(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._qualify_io()
        return result

    return call


class CampaignCoordinatorError(ValueError):
    """Closed source/campaign/lease/provenance refusal; original state remains intact."""


class CampaignUnresolved(CampaignCoordinatorError):  # noqa: N818 - frozen public contract name
    """An original write or read outcome is unresolved; no automatic mutation replay."""


def _text(value, limit=4096):
    if (
        type(value) is not str
        or not 0 < len(value.encode("utf-8")) <= limit
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise CampaignCoordinatorError("bounded exact campaign text required")
    return value


def _sha(value):
    if type(value) is not str or not re.fullmatch("[a-f0-9]{64}", value):
        raise CampaignCoordinatorError("closed campaign digest required")
    return value


def _integer(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise CampaignCoordinatorError("closed campaign integer required")
    return value


def _uuid(value):
    if type(value) is not str:
        raise CampaignCoordinatorError("canonical server campaign UUID required")
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise CampaignCoordinatorError("canonical server campaign UUID required") from None
    return value


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise CampaignCoordinatorError("closed campaign field inventory required")
    return value


def _json(value, limit=_MAX):
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise CampaignCoordinatorError("campaign JSON encoding refused") from error
    if not 0 < len(raw) <= limit:
        raise CampaignCoordinatorError("campaign byte bound exceeded")
    return raw


def _parse(raw, limit=_MAX, canonical=True):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise CampaignCoordinatorError("bounded campaign bytes required")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise CampaignCoordinatorError("duplicate campaign field")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                CampaignCoordinatorError("nonfinite campaign value")
            ),
        )
        if canonical and _json(value, limit) != raw:
            raise CampaignCoordinatorError("canonical campaign bytes differ")
        return value
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise CampaignCoordinatorError("campaign JSON refused") from error


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _native_identity(kind, account, region, name, arn):
    if (
        kind not in ("Training", "Processing")
        or type(kind) is not str
        or type(account) is not str
        or not re.fullmatch("[0-9]{12}", account)
        or type(region) is not str
        or not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-[0-9]", region)
        or type(name) is not str
        or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", name)
    ):
        raise CampaignCoordinatorError("closed native campaign identity required")
    resource = "training-job" if kind == "Training" else "processing-job"
    if type(arn) is not str or arn != f"arn:aws:sagemaker:{region}:{account}:{resource}/{name}":
        raise CampaignCoordinatorError("original native ARN scope/kind differs")


@dataclass(frozen=True)
class OriginalCampaignOperation:
    operation_digest: str
    phase: str
    kind: str
    account_id: str
    region: str
    job_name: str
    job_arn: str
    request_sha256: str
    source_binding_sha256: str

    def __post_init__(self):
        for value in (self.operation_digest, self.request_sha256, self.source_binding_sha256):
            _sha(value)
        if type(self.phase) is not str or _PHASE_KIND.get(self.phase) != self.kind:
            raise CampaignCoordinatorError("original phase/native kind differs")
        _native_identity(self.kind, self.account_id, self.region, self.job_name, self.job_arn)


def _operation_wire(original):
    if type(original) is not OriginalCampaignOperation:
        raise CampaignCoordinatorError("exact original campaign operation required")
    return {name: getattr(original, name) for name in _OPERATION}


@dataclass(frozen=True)
class NativeCampaignObservation:
    kind: str
    job_name: str
    job_arn: str
    request_sha256: str
    source_binding_sha256: str
    status: str

    def __post_init__(self):
        if (
            self.kind not in ("Training", "Processing")
            or type(self.kind) is not str
            or type(self.status) is not str
            or self.status not in _NATIVE | {"UNKNOWN"}
        ):
            raise CampaignCoordinatorError("closed native campaign observation required")
        _text(self.job_name, 63)
        _text(self.job_arn)
        _sha(self.request_sha256)
        _sha(self.source_binding_sha256)


def _observation_wire(value):
    if type(value) is not NativeCampaignObservation:
        raise CampaignCoordinatorError("exact native observation DTO required")
    return {
        name: getattr(value, name)
        for name in ("kind", "job_name", "job_arn", "request_sha256", "source_binding_sha256", "status")
    }


@dataclass(frozen=True)
class WakeHint:
    schema_version: int
    event_id: str
    campaign_ref: str
    account_id: str
    region: str
    deployment_id: str
    kind: str
    operation_digest: str
    job_arn: str

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise CampaignCoordinatorError("closed wake schema required")
        _text(self.event_id, 128)
        _reference(self.campaign_ref)
        _sha(self.operation_digest)
        _uuid(self.deployment_id)
        for field in (self.account_id, self.region, self.kind, self.job_arn):
            _text(field)
        if self.kind not in ("Training", "Processing") or not re.fullmatch("[0-9]{12}", self.account_id):
            raise CampaignCoordinatorError("closed wake scope/kind required")
        _json({name: getattr(self, name) for name in _WAKE}, 4096)


def parse_wake_hint(raw_bytes):
    value = _keys(_parse(raw_bytes, 4096, canonical=False), _WAKE)
    return WakeHint(**value)


def _reference(value):
    match = re.fullmatch(r"campaign_([a-f0-9]{64})_([a-f0-9]{64})", value) if type(value) is str else None
    if match is None:
        raise CampaignCoordinatorError("qualified original campaign reference required")
    return match[1], match[2]


@dataclass(frozen=True)
class CampaignOperationRecord:
    original: OriginalCampaignOperation
    status: str
    last_confirmed_native_status: str | None
    last_event_id: str | None
    last_event_digest: str | None
    observation_digest: str | None
    observed_at_ms: int | None


@dataclass(frozen=True)
class CampaignRecord:
    campaign_ref: str
    _binding_bytes: bytes
    revision: int
    lease_epoch: int
    lease_holder: str | None
    lease_until_ms: int | None
    operations: tuple[CampaignOperationRecord, ...]
    _body_bytes: bytes

    @property
    def binding(self):
        return _parse(self._binding_bytes)

    @property
    def campaign_id(self):
        return self.binding["campaign_id"]

    @property
    def private_store_correlation(self):
        return self.binding["private_store_correlation"]

    @property
    def literal_prompt(self):
        return self.binding["literal_prompt"]

    @property
    def prompt_sha256(self):
        return self.binding["prompt_sha256"]


@dataclass(frozen=True, init=False)
class CampaignLease:
    campaign_ref: str
    holder: str
    epoch: int
    issued_at_ms: int
    deadline_ms: int
    process_id: int
    _seal: object

    def __init__(
        self, campaign_ref=None, holder=None, epoch=None, issued_at_ms=None, deadline_ms=None, *, _seal=None
    ):
        if _seal is not _LEASE_SEAL:
            raise CampaignCoordinatorError("server-issued original campaign lease required")
        for name, value in (
            ("campaign_ref", campaign_ref),
            ("holder", holder),
            ("epoch", epoch),
            ("issued_at_ms", issued_at_ms),
            ("deadline_ms", deadline_ms),
            ("process_id", os.getpid()),
            ("_seal", _seal),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class WakeObservation:
    campaign_ref: str
    operation_digest: str
    status: str
    last_confirmed_native_status: str | None
    native_admission: bool = False
    quality_success: bool = False


def _row_wire(row):
    return {
        "original": _operation_wire(row.original),
        "status": row.status,
        "last_confirmed_native_status": row.last_confirmed_native_status,
        "last_event_id": row.last_event_id,
        "last_event_digest": row.last_event_digest,
        "observation_digest": row.observation_digest,
        "observed_at_ms": row.observed_at_ms,
    }


def _row(value):
    _keys(value, _ROW)
    original = OriginalCampaignOperation(**_keys(value["original"], _OPERATION))
    if (
        type(value["status"]) is not str
        or value["status"] not in _NATIVE | {"REGISTERED", "UNKNOWN"}
        or (
            value["last_confirmed_native_status"] is not None
            and value["last_confirmed_native_status"] not in _NATIVE
        )
    ):
        raise CampaignCoordinatorError("closed durable campaign status required")
    for name in ("last_event_digest", "observation_digest"):
        if value[name] is not None:
            _sha(value[name])
    if value["last_event_id"] is not None:
        _text(value["last_event_id"], 128)
    if value["observed_at_ms"] is not None:
        _integer(value["observed_at_ms"])
    if any(
        value[name] is None
        for name in ("last_event_id", "last_event_digest", "observation_digest", "observed_at_ms")
    ) and any(
        value[name] is not None
        for name in ("last_event_id", "last_event_digest", "observation_digest", "observed_at_ms")
    ):
        raise CampaignCoordinatorError("durable observation metadata incomplete")
    if value["status"] == "REGISTERED" and (
        value["last_confirmed_native_status"] is not None or value["last_event_id"] is not None
    ):
        raise CampaignCoordinatorError("registered original cannot invent observation")
    if value["status"] in _NATIVE and value["last_confirmed_native_status"] != value["status"]:
        raise CampaignCoordinatorError("durable confirmed campaign status differs")
    return CampaignOperationRecord(original, *(value[name] for name in _ROW[1:]))


class DynamoCampaignCoordinator:
    """One scoped jobs-table row; CAS fences commits, never external AWS execution."""

    def __init__(self, guard, private_dynamodb_client, *, table_name, clock=time.time):
        from .campaign_admission import AdmissionGuard

        if (
            type(guard) is not AdmissionGuard
            or not callable(clock)
            or type(table_name) is not str
            or not re.fullmatch(r"[A-Za-z0-9_.-]{3,255}", table_name)
        ):
            raise CampaignCoordinatorError("explicit trusted coordinator guard/table/clock required")
        meta = getattr(private_dynamodb_client, "meta", None)
        if meta is not None:
            retries = getattr(getattr(meta, "config", None), "retries", None)
            if (
                type(retries) is not dict
                or type(retries.get("total_max_attempts")) is not int
                or retries["total_max_attempts"] != 1
            ):
                raise CampaignCoordinatorError("native coordinator mutation retries must equal one")
        for name in ("get_item", "put_item", "update_item"):
            if not callable(getattr(private_dynamodb_client, name, None)):
                raise CampaignCoordinatorError("private closed DynamoDB transport required")
        self.guard, self._ddb, self._table, self._clock = guard, private_dynamodb_client, table_name, clock
        self._process_id = os.getpid()
        self._io_qualification = ContextVar("amlas_private_io_qualification", default=None)
        self._holder = str(uuid.uuid4())
        self._pending_puts, self._pending_updates = {}, {}
        self._binding_pins = {}
        self._last_time = None
        self._lock = threading.RLock()

    @contextmanager
    def qualified_io(self, current_qualification=None):
        self._require_process()
        if current_qualification is None:
            yield self
            return
        if not callable(current_qualification):
            raise CampaignCoordinatorError("private current qualification callable required")
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
        self._require_process()
        current = self._io_qualification.get()
        if current is not None:
            current()

    def _require_process(self):
        if self._process_id != os.getpid():
            raise CampaignCoordinatorError("inherited campaign coordinator cannot perform I/O")

    def _scope(self, witness):
        self._require_process()
        self.guard.require(witness)
        approved = self.guard.approved
        manifest = approved.manifest
        return {
            "account_id": manifest.account_id,
            "region": manifest.region,
            "deployment_id": manifest.deployment_id,
            "deployment_prefix": manifest.deployment_prefix,
            "owner_namespace": manifest.owner_namespace,
            "baseline_sha256": approved.manifest_sha256,
            "runtime_arn": manifest.runtime_arn,
            "endpoint_arn": manifest.endpoint_arn,
            "endpoint_name": manifest.endpoint_name,
            "runtime_version": manifest.runtime_version,
        }

    def _now(self):
        value = self._clock()
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (self._last_time is not None and value < self._last_time)
        ):
            raise CampaignCoordinatorError("finite nondecreasing campaign clock required")
        self._last_time = value
        return _integer(int(value * 1000))

    def _get(self, witness, key):
        self.guard.require(witness)
        self._qualify_io()
        response = self._ddb.get_item(
            TableName=self._table, Key={"job_name": {"S": key}}, ConsistentRead=True
        )
        self._qualify_io()
        if type(response) is not dict:
            raise CampaignUnresolved("campaign consistent read shape unavailable")
        return response.get("Item")

    @staticmethod
    def _body(item, key, scope, expected_binding_sha256):
        _keys(item, ("job_name", "coordinator_revision", "lease_epoch", "private_record"))
        for field, kind in (
            ("job_name", "S"),
            ("coordinator_revision", "N"),
            ("lease_epoch", "N"),
            ("private_record", "B"),
        ):
            _keys(item[field], (kind,))
        if item["job_name"]["S"] != key:
            raise CampaignCoordinatorError("original campaign row key differs")
        raw = item["private_record"]["B"]
        body = _keys(_parse(raw), _BODY)
        if type(body["schema_version"]) is not int or body["schema_version"] != 1:
            raise CampaignCoordinatorError("durable campaign schema differs")
        binding = _keys(body["binding"], _BINDING)
        if any(type(binding[name]) is not str or binding[name] != scope[name] for name in _SCOPE):
            raise CampaignCoordinatorError("durable campaign admission/owner scope differs")
        _uuid(binding["campaign_id"])
        _uuid(binding["private_store_correlation"])
        DynamoCampaignCoordinator._inputs(binding["transport_session_id"], binding["literal_prompt"])
        expected_binding_sha256 = _sha(expected_binding_sha256)
        actual_binding_sha256 = _digest(binding)
        if (
            binding["prompt_sha256"] != hashlib.sha256(binding["literal_prompt"].encode()).hexdigest()
            or body["binding_sha256"] != actual_binding_sha256
            or actual_binding_sha256 != expected_binding_sha256
        ):
            raise CampaignCoordinatorError("durable original immutable campaign commitment differs")
        expected_key = "CAMPAIGN#" + _digest(
            {**scope, "transport_session_id": binding["transport_session_id"]}
        )
        if key != expected_key:
            raise CampaignCoordinatorError("durable campaign mapping differs")
        for name in ("revision", "lease_epoch"):
            _integer(body[name])
        if (
            body["revision"] < body["lease_epoch"]
            or item["coordinator_revision"]["N"] != str(body["revision"])
            or item["lease_epoch"]["N"] != str(body["lease_epoch"])
        ):
            raise CampaignCoordinatorError("durable campaign projection differs")
        if (body["lease_holder"] is None) != (body["lease_until_ms"] is None):
            raise CampaignCoordinatorError("durable campaign lease incomplete")
        if body["lease_holder"] is not None:
            _uuid(body["lease_holder"])
            _integer(body["lease_until_ms"])
            if body["lease_epoch"] == 0:
                raise CampaignCoordinatorError("durable campaign lease epoch absent")
        if type(body["operations"]) is not list or len(body["operations"]) > 32:
            raise CampaignCoordinatorError("bounded campaign operation population required")
        originals, identities = set(), set()
        rows = []
        for value in body["operations"]:
            row = _row(value)
            DynamoCampaignCoordinator._original_scope(row.original, scope)
            if row.original.operation_digest in originals or row.original.job_arn in identities:
                raise CampaignCoordinatorError("duplicate original campaign operation")
            originals.add(row.original.operation_digest)
            identities.add(row.original.job_arn)
            rows.append(row)
        return body, CampaignRecord(
            "campaign_" + key[len("CAMPAIGN#") :] + "_" + expected_binding_sha256,
            _json(binding),
            body["revision"],
            body["lease_epoch"],
            body["lease_holder"],
            body["lease_until_ms"],
            tuple(rows),
            raw,
        )

    @staticmethod
    def _inputs(transport, prompt):
        _text(transport, 1024)
        if not 33 <= len(transport) <= 256:
            raise CampaignCoordinatorError("transport session bounds differ")
        if type(prompt) is not str or not 0 < len(prompt.encode("utf-8")) <= 65536:
            raise CampaignCoordinatorError("literal campaign prompt byte bounds differ")

    @staticmethod
    def _item(key, body):
        return {
            "job_name": {"S": key},
            "coordinator_revision": {"N": str(body["revision"])},
            "lease_epoch": {"N": str(body["lease_epoch"])},
            "private_record": {"B": _json(body)},
        }

    def _reconcile(self, witness, key, intended):
        try:
            observed = self._get(witness, key)
        except Exception as error:
            from .campaign_admission import AdmissionError

            if isinstance(error, AdmissionError):
                raise
            raise CampaignUnresolved("original campaign acknowledgement unreadable") from error
        if observed != intended:
            raise CampaignUnresolved("original campaign write acknowledgement unresolved")
        return observed

    def _cas(self, witness, key, prior, body, purpose):
        intended = self._item(key, body)
        if key in self._pending_updates:
            old_purpose, pending = self._pending_updates[key]
            if old_purpose != purpose:
                raise CampaignUnresolved("another original campaign update remains unresolved")
            reconciled = self._reconcile(witness, key, pending)
            self._pending_updates.pop(key)
            return reconciled
        self.guard.require(witness)
        if body["lease_until_ms"] is None or self._now() >= body["lease_until_ms"]:
            raise CampaignCoordinatorError("original campaign lease expired during admission recheck")
        self._qualify_io()
        self.guard.require(witness)
        if body["lease_until_ms"] is None or self._now() >= body["lease_until_ms"]:
            raise CampaignCoordinatorError("original campaign lease expired during current IO qualification")
        self._pending_updates[key] = (purpose, intended)
        try:
            self._ddb.update_item(
                TableName=self._table,
                Key={"job_name": {"S": key}},
                ConditionExpression="coordinator_revision = :revision AND lease_epoch = :epoch AND private_record = :prior_body",
                UpdateExpression="SET coordinator_revision = :next, lease_epoch = :next_epoch, private_record = :body",
                ExpressionAttributeValues={
                    ":revision": prior["coordinator_revision"],
                    ":epoch": prior["lease_epoch"],
                    ":prior_body": prior["private_record"],
                    ":next": intended["coordinator_revision"],
                    ":next_epoch": intended["lease_epoch"],
                    ":body": intended["private_record"],
                },
            )
        except Exception as error:
            if self._conditional(error):
                self._pending_updates.pop(key)
                raise CampaignCoordinatorError("campaign revision/epoch/body conflict") from error
            result = self._reconcile(witness, key, intended)
            self._pending_updates.pop(key)
            return result
        self._qualify_io()
        self._pending_updates.pop(key)
        return intended

    @staticmethod
    def _conditional(error):
        response = getattr(error, "response", None)
        details = response.get("Error") if type(response) is dict else None
        return type(details) is dict and details.get("Code") == "ConditionalCheckFailedException"

    @_qualified_return
    def open_campaign(self, witness, transport_session_id, literal_prompt, *, expected_campaign_ref=None):
        self._require_process()
        with self._lock:
            scope = self._scope(witness)
            self._inputs(transport_session_id, literal_prompt)
            selector = _digest({**scope, "transport_session_id": transport_session_id})
            key = "CAMPAIGN#" + selector
            expected = self._binding_pins.get(key)
            if expected_campaign_ref is not None:
                selected, supplied = _reference(expected_campaign_ref)
                if selected != selector or (expected is not None and supplied != expected):
                    raise CampaignCoordinatorError("original qualified campaign mapping differs")
                expected = supplied
            pending = self._pending_puts.get(key)
            if pending is not None:
                if expected is None:
                    raise CampaignUnresolved("original campaign commitment unavailable")
                item = self._reconcile(witness, key, pending)
            else:
                item = self._get(witness, key)
                if item is None:
                    if expected is not None:
                        raise CampaignUnresolved("qualified original campaign mapping unavailable")
                    binding = {
                        **scope,
                        "transport_session_id": transport_session_id,
                        "campaign_id": str(uuid.uuid4()),
                        "private_store_correlation": str(uuid.uuid4()),
                        "literal_prompt": literal_prompt,
                        "prompt_sha256": hashlib.sha256(literal_prompt.encode()).hexdigest(),
                    }
                    expected = _digest(binding)
                    body = {
                        "schema_version": 1,
                        "binding": binding,
                        "binding_sha256": expected,
                        "revision": 0,
                        "lease_epoch": 0,
                        "lease_holder": None,
                        "lease_until_ms": None,
                        "operations": [],
                    }
                    intended = self._item(key, body)
                    self.guard.require(witness)
                    # This process retains its original attempted commitment, never a row's self-hash.
                    self._qualify_io()
                    self.guard.require(witness)
                    self._binding_pins[key] = expected
                    self._pending_puts[key] = intended
                    try:
                        self._ddb.put_item(
                            TableName=self._table,
                            Item=intended,
                            ConditionExpression="attribute_not_exists(job_name)",
                        )
                    except Exception as error:
                        if self._conditional(error):
                            raise CampaignUnresolved(
                                "conflicting campaign creation lacks original acknowledgement"
                            ) from error
                        item = self._reconcile(witness, key, intended)
                    else:
                        self._qualify_io()
                        item = intended
                elif expected is None:
                    raise CampaignCoordinatorError(
                        "cold transport lookup requires independently retained campaign ref"
                    )
            _body, record = self._body(item, key, scope, expected)
            if record.literal_prompt != literal_prompt:
                raise CampaignCoordinatorError("original literal campaign prompt cannot be rebound")
            self._binding_pins[key] = expected
            self._pending_puts.pop(key, None)
            return record

    @_qualified_return
    def load_campaign(self, witness, campaign_ref):
        self._require_process()
        with self._lock:
            scope = self._scope(witness)
            selector, expected = _reference(campaign_ref)
            key = "CAMPAIGN#" + selector
            if key in self._binding_pins and self._binding_pins[key] != expected:
                raise CampaignCoordinatorError("original local campaign commitment differs")
            pending_put = self._pending_puts.get(key)
            pending_update = self._pending_updates.get(key)
            if pending_put is not None and pending_update is not None:
                raise CampaignUnresolved("conflicting original campaign acknowledgements remain unresolved")
            intended = (
                pending_put
                if pending_put is not None
                else pending_update[1]
                if pending_update is not None
                else None
            )
            item = (
                self._reconcile(witness, key, intended) if intended is not None else self._get(witness, key)
            )
            if item is None:
                raise CampaignCoordinatorError("original campaign mapping absent")
            record = self._body(item, key, scope, expected)[1]
            self._binding_pins[key] = expected
            if pending_put is not None:
                self._pending_puts.pop(key)
            if pending_update is not None:
                self._pending_updates.pop(key)
            return record

    @_qualified_return
    def claim_lease(self, witness, campaign_ref, expected_revision, *, lease_seconds=60):
        self._require_process()
        with self._lock:
            scope = self._scope(witness)
            if (
                type(lease_seconds) not in (int, float)
                or not math.isfinite(lease_seconds)
                or not 0 < lease_seconds <= 60
            ):
                raise CampaignCoordinatorError("bounded campaign lease duration required")
            _integer(expected_revision)
            selector, expected = _reference(campaign_ref)
            key = "CAMPAIGN#" + selector
            if key in self._binding_pins and self._binding_pins[key] != expected:
                raise CampaignCoordinatorError("original local campaign commitment differs")
            purpose = ("claim", expected_revision)
            if key in self._pending_updates:
                old_purpose, pending = self._pending_updates[key]
                if old_purpose != purpose:
                    raise CampaignUnresolved("original lease acknowledgement remains unresolved")
                item = self._reconcile(witness, key, pending)
                body, record = self._body(item, key, scope, expected)
                now = self._now()
                if body["lease_holder"] != self._holder or now >= body["lease_until_ms"]:
                    raise CampaignCoordinatorError("reconciled original lease expired or foreign")
                self._binding_pins[key] = expected
                self._pending_updates.pop(key)
                return CampaignLease(
                    campaign_ref,
                    self._holder,
                    record.lease_epoch,
                    now,
                    record.lease_until_ms,
                    _seal=_LEASE_SEAL,
                )
            prior = self._get(witness, key)
            if prior is None:
                raise CampaignCoordinatorError("original campaign mapping absent")
            body, record = self._body(prior, key, scope, expected)
            self._binding_pins[key] = expected
            now = self._now()
            if record.revision != expected_revision or (
                record.lease_holder is not None
                and record.lease_holder != self._holder
                and now < record.lease_until_ms
            ):
                raise CampaignCoordinatorError("campaign lease holder or revision conflict")
            body.update(
                revision=record.revision + 1,
                lease_epoch=record.lease_epoch + 1,
                lease_holder=self._holder,
                lease_until_ms=now + int(lease_seconds * 1000),
            )
            result = self._cas(witness, key, prior, body, purpose)
            _body, changed = self._body(result, key, scope, expected)
            return CampaignLease(
                campaign_ref,
                self._holder,
                changed.lease_epoch,
                now,
                changed.lease_until_ms,
                _seal=_LEASE_SEAL,
            )

    def _leased(self, witness, lease):
        if (
            type(lease) is not CampaignLease
            or lease._seal is not _LEASE_SEAL
            or lease.process_id != os.getpid()
            or lease.holder != self._holder
        ):
            raise CampaignCoordinatorError("original process-bound campaign lease required")
        scope = self._scope(witness)
        now = self._now()
        if now < lease.issued_at_ms or now >= lease.deadline_ms:
            raise CampaignCoordinatorError("original campaign lease expired or clock rolled back")
        selector, expected = _reference(lease.campaign_ref)
        key = "CAMPAIGN#" + selector
        if key in self._binding_pins and self._binding_pins[key] != expected:
            raise CampaignCoordinatorError("original lease campaign commitment differs")
        pending = self._pending_updates.get(key)
        if pending is not None:
            purpose, intended = pending
            if purpose[0] not in ("register", "observe"):
                raise CampaignUnresolved("another original lease update remains unresolved")
            prior = self._reconcile(witness, key, intended)
        else:
            prior = self._get(witness, key)
        if prior is None:
            raise CampaignCoordinatorError("original campaign mapping absent")
        body, record = self._body(prior, key, scope, expected)
        if (record.lease_holder, record.lease_epoch, record.lease_until_ms) != (
            lease.holder,
            lease.epoch,
            lease.deadline_ms,
        ):
            raise CampaignCoordinatorError("original campaign lease fenced by newer epoch")
        if self._now() >= lease.deadline_ms:
            raise CampaignCoordinatorError("original campaign lease expired during reconciliation")
        if pending is not None:
            self._pending_updates.pop(key)
        return scope, key, prior, body, record

    @staticmethod
    def _original_scope(operation, scope):
        if type(operation) is not OriginalCampaignOperation or (operation.account_id, operation.region) != (
            scope["account_id"],
            scope["region"],
        ):
            raise CampaignCoordinatorError("original campaign operation scope differs")

    @_qualified_return
    def register_original(self, witness, lease, original_operation):
        self._require_process()
        with self._lock:
            scope, key, prior, body, record = self._leased(witness, lease)
            self._original_scope(original_operation, scope)
            for row in record.operations:
                if row.original.operation_digest == original_operation.operation_digest:
                    if row.original != original_operation:
                        raise CampaignCoordinatorError(
                            "original operation identity/request/source replacement refused"
                        )
                    return record
                if row.original.job_arn == original_operation.job_arn:
                    raise CampaignCoordinatorError("native identity cannot select another operation digest")
            if len(record.operations) >= 32:
                raise CampaignCoordinatorError("bounded original operation population exhausted")
            row = CampaignOperationRecord(original_operation, "REGISTERED", None, None, None, None, None)
            body["operations"].append(_row_wire(row))
            body["revision"] += 1
            if self._now() >= lease.deadline_ms:
                raise CampaignCoordinatorError("original campaign lease expired before CAS")
            result = self._cas(
                witness,
                key,
                prior,
                body,
                (
                    "register",
                    original_operation.operation_digest,
                    _digest(_operation_wire(original_operation)),
                ),
            )
            return self._body(result, key, scope, _reference(lease.campaign_ref)[1])[1]

    @staticmethod
    def _wake_result(record, row):
        return WakeObservation(
            record.campaign_ref, row.original.operation_digest, row.status, row.last_confirmed_native_status
        )

    @_qualified_return
    def observe_wake(self, witness, lease, hint, private_observer, *, fresh_observation=False):
        self._require_process()
        with self._lock:
            scope, key, prior, body, record = self._leased(witness, lease)
            if (
                type(hint) is not WakeHint
                or hint.campaign_ref != record.campaign_ref
                or (hint.account_id, hint.region, hint.deployment_id)
                != (scope["account_id"], scope["region"], scope["deployment_id"])
            ):
                raise CampaignCoordinatorError("original registered wake scope differs")
            index = next(
                (
                    i
                    for i, row in enumerate(record.operations)
                    if row.original.operation_digest == hint.operation_digest
                ),
                None,
            )
            if index is None:
                raise CampaignCoordinatorError("wake cannot create an unregistered operation")
            row = record.operations[index]
            original = row.original
            if (hint.kind, hint.job_arn) != (original.kind, original.job_arn):
                raise CampaignCoordinatorError("wake original native identity differs")
            event_digest = _digest({name: getattr(hint, name) for name in _WAKE})
            if type(fresh_observation) is not bool:
                raise CampaignCoordinatorError("exact fresh observation selector required")
            if row.last_event_id == hint.event_id:
                if row.last_event_digest != event_digest:
                    raise CampaignCoordinatorError("original event identifier was rebound")
                if not fresh_observation:
                    return self._wake_result(record, row)
            self.guard.require(witness)
            if self._now() >= lease.deadline_ms:
                raise CampaignCoordinatorError("original campaign lease expired before observation")
            self._qualify_io()
            self.guard.require(witness)
            if self._now() >= lease.deadline_ms:
                raise CampaignCoordinatorError(
                    "original campaign lease expired during pre-observation qualification"
                )
            try:
                observation = private_observer.observe_original(original)
            except Exception:
                observation = NativeCampaignObservation(
                    original.kind,
                    original.job_name,
                    original.job_arn,
                    original.request_sha256,
                    original.source_binding_sha256,
                    "UNKNOWN",
                )
            self._qualify_io()
            self.guard.require(witness)
            now = self._now()
            if now >= lease.deadline_ms:
                raise CampaignCoordinatorError("original campaign lease expired after observation")
            wire = _observation_wire(observation)
            if any(
                wire[name] != getattr(original, name)
                for name in ("kind", "job_name", "job_arn", "request_sha256", "source_binding_sha256")
            ):
                raise CampaignCoordinatorError("native observed original identity/request/source differs")
            status = observation.status
            confirmed = row.last_confirmed_native_status
            if status != "UNKNOWN":
                if (confirmed in _TERMINAL and status != confirmed) or (
                    confirmed == "Stopping" and status == "InProgress"
                ):
                    raise CampaignCoordinatorError(
                        "native campaign observation cannot regress or replace terminal state"
                    )
                confirmed = status
            changed = CampaignOperationRecord(
                original, status, confirmed, hint.event_id, event_digest, _digest(wire), now
            )
            body["operations"][index] = _row_wire(changed)
            body["revision"] += 1
            if self._now() >= lease.deadline_ms:
                raise CampaignCoordinatorError("original campaign lease expired before observation CAS")
            result = self._cas(
                witness,
                key,
                prior,
                body,
                ("observe", original.operation_digest, event_digest, changed.observation_digest),
            )
            final = self._body(result, key, scope, _reference(lease.campaign_ref)[1])[1]
            return self._wake_result(final, final.operations[index])


def _initial_campaign_item(
    scope, transport_session_id, literal_prompt, campaign_id, private_store_correlation
):
    """Pure initial item builder; no admission, SDK call or authority is minted."""
    _keys(scope, _SCOPE)
    DynamoCampaignCoordinator._inputs(transport_session_id, literal_prompt)
    _uuid(campaign_id)
    _uuid(private_store_correlation)
    selector = _digest({**scope, "transport_session_id": transport_session_id})
    binding = {
        **scope,
        "transport_session_id": transport_session_id,
        "campaign_id": campaign_id,
        "private_store_correlation": private_store_correlation,
        "literal_prompt": literal_prompt,
        "prompt_sha256": hashlib.sha256(literal_prompt.encode()).hexdigest(),
    }
    binding_sha = _digest(binding)
    body = {
        "schema_version": 1,
        "binding": binding,
        "binding_sha256": binding_sha,
        "revision": 0,
        "lease_epoch": 0,
        "lease_holder": None,
        "lease_until_ms": None,
        "operations": [],
    }
    key = "CAMPAIGN#" + selector
    item = DynamoCampaignCoordinator._item(key, body)
    record = DynamoCampaignCoordinator._body(item, key, scope, binding_sha)[1]
    return selector, record.campaign_ref, item
