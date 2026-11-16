"""Conditional first-model journal: protected origin, one attempt, no native Create.

Explicit server qualification ports remain separate from action/native authority.
A row hash, typed DTO, lease, witness or callback shape cannot approve an effect.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ..backends.types import JobHandle
from ..data_eng.model_acquisition_schema import expected_digest
from ..data_eng.model_weight_manifest import expected_model_weight_manifest
from .campaign_coordinator import CampaignLease, DynamoCampaignCoordinator, OriginalCampaignOperation

_MAX = 256 * 1024
_PERMIT_SEAL = object()
_STATES = {"RESERVED", "CREATE_ATTEMPTED", "ACCEPTED", "UNKNOWN", "REJECTED"}
_HANDLE = {
    "accepted",
    "submission_ambiguous",
    "acquired",
    "catalog_id",
    "catalog_version",
    "expected_manifest_sha256",
    "bindings",
    "bucket",
    "account_id",
    "region",
    "output_prefix",
    "request",
    "native_status",
}
_INTENT = {
    "schema_version",
    "entry",
    "planning_ref",
    "spec_sha256",
    "config",
    "catalog_id",
    "catalog_version",
    "selection_digest",
    "operation_digest",
    "model_ref",
    "expected_manifest_sha256",
    "prepared",
}
_ENTRY = {"campaign_ref", "campaign_id", "private_store_correlation", "transport_session_id", "prompt_sha256"}
_ROW = {
    "schema_version",
    "intent",
    "intent_sha256",
    "revision",
    "state",
    "registered",
    "attempt_token",
    "attempt_epoch",
    "attempt_holder",
    "attempt_deadline_ms",
    "outcome_arn",
    "error_type",
}


class ModelCampaignError(ValueError):
    """Original model journal is unknown, unqualified or inconsistent."""


def _sha(value):
    if type(value) is not str or not re.fullmatch("[a-f0-9]{64}", value):
        raise ModelCampaignError("exact model digest required")
    return value


def _json(value):
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ModelCampaignError("closed model JSON encoding refused") from None
    if not 0 < len(raw) <= _MAX:
        raise ModelCampaignError("model journal byte bound exceeded")
    return raw


def _parse(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX:
        raise ModelCampaignError("bounded canonical model bytes required")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ModelCampaignError("duplicate model field")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ModelCampaignError("finite model JSON required")),
        )
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ModelCampaignError("model JSON decoding refused") from error
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 20:
            raise ModelCampaignError("model journal nesting bound exceeded")
        if type(item) is dict:
            pending.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            pending.extend((v, depth + 1) for v in item)
        elif type(item) is float and not math.isfinite(item):
            raise ModelCampaignError("finite model value required")
    if _json(value) != raw:
        raise ModelCampaignError("canonical model bytes differ")
    return value


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ModelCampaignError("closed model field inventory required")
    return value


def _integer(value):
    if type(value) is not int or not 0 <= value <= 10**16:
        raise ModelCampaignError("bounded model integer required")
    return value


@dataclass(frozen=True)
class ModelAcquisitionConfig:
    account_id: str
    region: str
    bucket: str
    execution_role_arn: str
    image_uri: str
    source_commit: str
    admission_digest: str
    deployment_id: str
    deployment_prefix: str
    http_allow_hosts: str | None = None

    def validate(self):
        if any(
            type(getattr(self, f.name)) is not str
            for f in dataclasses.fields(self)
            if f.name != "http_allow_hosts"
        ):
            raise ModelCampaignError("model configuration primitive type differs")
        if (
            not re.fullmatch("[0-9]{12}", self.account_id)
            or self.account_id == "0" * 12
            or not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-\d", self.region)
        ):
            raise ModelCampaignError("model configuration account/region differs")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", self.bucket) or ".." in self.bucket:
            raise ModelCampaignError("model bucket differs")
        if not re.fullmatch(
            "arn:aws:iam::" + self.account_id + r":role/[A-Za-z0-9+=,.@_/-]+", self.execution_role_arn
        ):
            raise ModelCampaignError("model execution role differs")
        image = re.fullmatch(
            self.account_id
            + r"\.dkr\.ecr\."
            + re.escape(self.region)
            + r"\.amazonaws\.com/[a-z0-9][a-z0-9._/-]*@sha256:([a-f0-9]{64})",
            self.image_uri,
        )
        if (
            not image
            or image[1] != _sha(self.admission_digest)
            or not re.fullmatch("[a-f0-9]{40}", self.source_commit)
        ):
            raise ModelCampaignError("fixed model image/source pair differs")
        if (
            not re.fullmatch("[a-z][a-z0-9-]{0,39}", self.deployment_prefix)
            or str(uuid.UUID(self.deployment_id)) != self.deployment_id
        ):
            raise ModelCampaignError("model deployment identity differs")
        if self.http_allow_hosts is not None and (
            type(self.http_allow_hosts) is not str
            or not 0 < len(self.http_allow_hosts.encode()) <= 4096
            or any(ord(c) < 32 for c in self.http_allow_hosts)
        ):
            raise ModelCampaignError("bounded explicit model host configuration required")


@dataclass(frozen=True)
class ModelProcessingIntent:
    canonical_bytes: bytes


@dataclass(frozen=True)
class ProtectedModelPublication:
    reference: str
    intent_sha256: str
    publication_sha256: str

    def validate(self):
        if (
            type(self.reference) is not str
            or not 0 < len(self.reference) <= 512
            or any(ord(c) < 33 for c in self.reference)
        ):
            raise ModelCampaignError("original protected model reference required")
        _sha(self.intent_sha256)
        _sha(self.publication_sha256)


@dataclass(frozen=True)
class ProtectedModelOriginal:
    publication: ProtectedModelPublication
    canonical_intent: bytes


@dataclass(frozen=True)
class ProtectedModelRead:
    """Closed private read result; ABSENT still needs independent origin qualification."""

    status: str
    original: ProtectedModelOriginal | None = None

    def validate(self):
        if type(self.status) is not str or self.status not in {"PRESENT", "ABSENT", "UNKNOWN"}:
            raise ModelCampaignError("closed protected model read status required")
        if (self.status == "PRESENT") != (type(self.original) is ProtectedModelOriginal):
            raise ModelCampaignError("protected model read shape differs")
        if self.status != "PRESENT" and self.original is not None:
            raise ModelCampaignError("absent/unknown protected model has original data")


@dataclass(frozen=True)
class ModelOperationReservation:
    intent: ModelProcessingIntent
    revision: int
    state: str
    registered: bool
    _body: bytes = field(repr=False)

    @property
    def job_name(self):
        return _parse(self.intent.canonical_bytes)["prepared"]["id"]

    @property
    def job_arn(self):
        return _parse(self.intent.canonical_bytes)["prepared"]["arn"]

    @property
    def model_ref(self):
        return _parse(self.intent.canonical_bytes)["model_ref"]

    @property
    def request_sha256(self):
        return hashlib.sha256(
            _json(_parse(self.intent.canonical_bytes)["prepared"]["extra"]["request"])
        ).hexdigest()

    @property
    def source_binding_sha256(self):
        value = _parse(self.intent.canonical_bytes)
        return hashlib.sha256(
            _json(
                {
                    "config": value["config"],
                    "manifest": value["expected_manifest_sha256"],
                    "entry": value["entry"],
                    "planning_ref": value["planning_ref"],
                    "spec_sha256": value["spec_sha256"],
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class ModelOriginalInspection:
    disposition: str
    reservation: ModelOperationReservation | None


@dataclass(frozen=True)
class ModelReservationOutcome:
    reservation: ModelOperationReservation
    disposition: str


@dataclass(frozen=True)
class ModelObservation:
    reservation: ModelOperationReservation
    status: str
    native_admission: bool = False
    acquired: bool = False


@dataclass
class ModelCreatePermit:
    reservation_sha256: str
    attempt_token: str
    process_id: int
    lease: CampaignLease
    owner: object = field(repr=False)
    seal: object = field(repr=False)
    used: bool = False
    outcome_written: bool = False


def _binding(entry, qualifier):
    if not callable(qualifier):
        raise ModelCampaignError("explicit private entry qualifier required")
    witness = qualifier(entry)
    value = {name: getattr(entry, name) for name in _ENTRY}
    if not re.fullmatch(r"campaign_[a-f0-9]{64}_[a-f0-9]{64}", value["campaign_ref"]):
        raise ModelCampaignError("qualified original campaign reference required")
    for name in ("campaign_id", "private_store_correlation"):
        if type(value[name]) is not str or str(uuid.UUID(value[name])) != value[name]:
            raise ModelCampaignError("original campaign UUID required")
    if type(value["transport_session_id"]) is not str or not re.fullmatch(
        "[A-Za-z0-9_-]{1,128}", value["transport_session_id"]
    ):
        raise ModelCampaignError("original model transport session required")
    _sha(value["prompt_sha256"])
    if hashlib.sha256(entry.literal_prompt.encode()).hexdigest() != value["prompt_sha256"]:
        raise ModelCampaignError("original model prompt binding differs")
    return value, witness


def _planning(entry, reference, spec):
    _sha(spec)
    if type(reference) is not str or not re.fullmatch("[A-Za-z][A-Za-z0-9_]{1,127}", reference):
        raise ModelCampaignError("original opaque spec pointer required")
    refs = entry.planning_refs
    if (
        type(refs) is not tuple
        or sum(
            getattr(v, "kind", None) == "spec"
            and getattr(v, "reference", None) == reference
            and getattr(v, "sha256", None) == spec
            for v in refs
        )
        != 1
    ):
        raise ModelCampaignError("original typed spec pointer differs")


def _handle_wire(handle, validator):
    if type(handle) is not JobHandle or handle.backend != "model_processing" or not callable(validator):
        raise ModelCampaignError("source model plan/validator required")
    validator(handle)
    extra = _keys(handle.extra, _HANDLE)
    if (
        any(extra[v] is not False for v in ("accepted", "submission_ambiguous", "acquired"))
        or extra["native_status"] != "Submitted"
    ):
        raise ModelCampaignError("unattempted model plan required")
    return _parse(_json({"id": handle.id, "backend": handle.backend, "arn": handle.arn, "extra": extra}))


def prepare_model_intent(
    entry,
    *,
    entry_qualifier,
    planning_ref,
    spec_sha256,
    config,
    prepare_submission,
    validate_prepared,
    catalog_id="qwen3_5_9b",
    catalog_version="1",
):
    binding, _ = _binding(entry, entry_qualifier)
    _planning(entry, planning_ref, spec_sha256)
    if type(config) is not ModelAcquisitionConfig:
        raise ModelCampaignError("source-configured model DTO required")
    config.validate()
    expected = expected_digest(expected_model_weight_manifest(catalog_id, catalog_version))
    selected = {
        "entry": binding,
        "planning_ref": planning_ref,
        "spec_sha256": spec_sha256,
        "config": dataclasses.asdict(config),
        "catalog_id": catalog_id,
        "catalog_version": catalog_version,
        "expected_manifest_sha256": expected,
    }
    digest = hashlib.sha256(_json(selected)).hexdigest()
    name = config.deployment_prefix + "-model-acq-" + digest[:12]
    handle = prepare_submission(
        catalog_id=catalog_id,
        catalog_version=catalog_version,
        runtime_session_id=binding["private_store_correlation"],
        job_name=name,
    )
    value = {
        "schema_version": 1,
        **selected,
        "selection_digest": digest,
        "model_ref": "model_acq_" + digest[:32],
        "prepared": _handle_wire(handle, validate_prepared),
    }
    value["operation_digest"] = hashlib.sha256(_json(value)).hexdigest()
    intent = ModelProcessingIntent(_json(value))
    _binding(entry, entry_qualifier)
    return validate_model_intent(
        intent, entry, entry_qualifier=entry_qualifier, validate_prepared=validate_prepared
    )


def validate_model_intent(intent, entry, *, entry_qualifier, validate_prepared):
    raw = intent.canonical_bytes if type(intent) is ModelProcessingIntent else intent
    value = _keys(_parse(raw), _INTENT)
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ModelCampaignError("model intent schema differs")
    binding, _ = _binding(entry, entry_qualifier)
    if value["entry"] != binding:
        raise ModelCampaignError("original model entry replacement refused")
    _planning(entry, value["planning_ref"], value["spec_sha256"])
    config = ModelAcquisitionConfig(
        **_keys(value["config"], {f.name for f in dataclasses.fields(ModelAcquisitionConfig)})
    )
    config.validate()
    expected = expected_digest(expected_model_weight_manifest(value["catalog_id"], value["catalog_version"]))
    selection = {
        k: value[k]
        for k in (
            "entry",
            "planning_ref",
            "spec_sha256",
            "config",
            "catalog_id",
            "catalog_version",
            "expected_manifest_sha256",
        )
    }
    digest = hashlib.sha256(_json(selection)).hexdigest()
    prepared = _keys(value["prepared"], {"id", "backend", "arn", "extra"})
    handle = JobHandle(prepared["id"], prepared["backend"], prepared["arn"], copy_json(prepared["extra"]))
    if (
        expected != value["expected_manifest_sha256"]
        or value["selection_digest"] != digest
        or value["model_ref"] != "model_acq_" + digest[:32]
        or handle.id != config.deployment_prefix + "-model-acq-" + digest[:12]
        or handle.arn != f"arn:aws:sagemaker:{config.region}:{config.account_id}:processing-job/{handle.id}"
    ):
        raise ModelCampaignError("original model identity/catalog replacement refused")
    extra = _handle_wire(handle, validate_prepared)["extra"]
    if (extra["account_id"], extra["region"], extra["bucket"], extra["expected_manifest_sha256"]) != (
        config.account_id,
        config.region,
        config.bucket,
        expected,
    ) or extra["bindings"] != {
        "job_name": handle.id,
        "runtime_session_id": binding["private_store_correlation"],
        "deployment_id": config.deployment_id,
        "deployment_prefix": config.deployment_prefix,
        "image_uri": config.image_uri,
        "source_commit": config.source_commit,
    }:
        raise ModelCampaignError("source model request/configuration differs")
    environment = {"AMLAS_AGENT_REMOTE_FETCH": "1"}
    if config.http_allow_hosts is not None:
        environment["AMLAS_AGENT_HTTP_ALLOW_HOSTS"] = config.http_allow_hosts
    if (
        extra["request"]["RoleArn"] != config.execution_role_arn
        or extra["request"]["AppSpecification"]["ImageUri"] != config.image_uri
        or extra["request"]["Environment"] != environment
    ):
        raise ModelCampaignError("original configured model role/egress differs")
    operation = dict(value)
    operation.pop("operation_digest")
    if value["operation_digest"] != hashlib.sha256(_json(operation)).hexdigest():
        raise ModelCampaignError("complete original model operation digest differs")
    return ModelProcessingIntent(raw)


def copy_json(value):
    return _parse(_json(value))


class DynamoModelCampaignJournal:
    def __init__(
        self,
        entry,
        *,
        entry_qualifier,
        protected_original_port,
        protected_origin_qualifier,
        coordinator,
        table_name,
        dynamodb_client,
        clock_ms,
        validate_prepared,
    ):
        if (
            type(coordinator) is not DynamoCampaignCoordinator
            or table_name != coordinator._table
            or not all(
                callable(v)
                for v in (entry_qualifier, protected_origin_qualifier, clock_ms, validate_prepared)
            )
        ):
            raise ModelCampaignError("explicit current model journal boundary required")
        for method in ("publish_first", "read_original", "require_current"):
            if not callable(getattr(protected_original_port, method, None)):
                raise ModelCampaignError("explicit protected original model port required")
        from botocore.client import BaseClient

        if isinstance(dynamodb_client, BaseClient):
            if dynamodb_client.meta.config.retries.get("total_max_attempts") != 1:
                raise ModelCampaignError("model journal requires one effective SDK attempt")
            if dynamodb_client.meta.region_name != coordinator.guard.approved.manifest.region:
                raise ModelCampaignError("model journal client region differs")
        self.entry, self._entry_qualifier, self._protected, self._origin_qualifier = (
            entry,
            entry_qualifier,
            protected_original_port,
            protected_origin_qualifier,
        )
        self.coordinator, self._table, self._ddb, self._clock, self._validator = (
            coordinator,
            table_name,
            dynamodb_client,
            clock_ms,
            validate_prepared,
        )
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._io = ContextVar("private_model_journal_io_" + uuid.uuid4().hex, default=None)
        self._highwater = None
        self._owner = object()
        self._permits = {}
        self._publication_attempts = {}
        self._binding, _ = _binding(entry, entry_qualifier)

    def _process(self):
        if self._pid != os.getpid():
            raise ModelCampaignError("inherited model journal refused")

    @contextmanager
    def qualified_io(self, current_qualification=None):
        self._process()
        if not callable(current_qualification):
            raise ModelCampaignError("fresh model IO qualifier required")
        token = self._io.set(current_qualification)
        try:
            self._gate()
            yield self
        finally:
            self._io.reset(token)

    def _gate(self):
        self._process()
        callback = self._io.get()
        if not callable(callback):
            raise ModelCampaignError("model IO boundary absent")
        if callback() is not None:
            raise ModelCampaignError("model IO callback grants nothing")

    def _now(self):
        self._process()
        value = self._clock()
        _integer(value)
        if self._highwater is not None and value < self._highwater:
            raise ModelCampaignError("model journal clock rollback")
        self._highwater = value
        return value

    def _intent(self, intent):
        self._gate()
        value = validate_model_intent(
            intent, self.entry, entry_qualifier=self._entry_qualifier, validate_prepared=self._validator
        )
        binding, witness = _binding(self.entry, self._entry_qualifier)
        scope = self.coordinator._scope(witness)
        config = _parse(value.canonical_bytes)["config"]
        if binding != self._binding or tuple(
            config[k] for k in ("account_id", "region", "deployment_id", "deployment_prefix")
        ) != tuple(scope[k] for k in ("account_id", "region", "deployment_id", "deployment_prefix")):
            raise ModelCampaignError("model scope drift")
        self._gate()
        return value

    def _protected_read(self, planning_ref):
        self._gate()
        try:
            result = self._protected.read_original(
                self._binding, planning_ref=planning_ref, qualify_io=self._gate
            )
        except Exception:
            self._gate()
            raise ModelCampaignError("protected original model read unresolved") from None
        self._gate()
        if type(result) is not ProtectedModelRead:
            raise ModelCampaignError("qualified protected model read status absent")
        result.validate()
        if result.status == "UNKNOWN":
            raise ModelCampaignError("protected original model existence unresolved")
        return result

    def _origin(self, publication, original):
        # None/None is ONLY first-publisher configuration qualification, never
        # an original commitment, qualified absence, or an action grant.
        self._gate()
        evidence = self._protected.require_current(
            publication, original, self._binding, qualify_io=self._gate
        )
        self._gate()
        if (
            self._origin_qualifier(self._protected, publication, original, self._binding, evidence)
            is not None
        ):
            raise ModelCampaignError("protected origin qualifier grants nothing")
        self._gate()

    def _protected_original(self, intent, publication=None):
        intent = self._intent(intent)
        data = _parse(intent.canonical_bytes)
        result = self._protected_read(data["planning_ref"])
        if result.status != "PRESENT":
            raise ModelCampaignError("qualified original model handoff absent")
        original = result.original
        if type(original.publication) is not ProtectedModelPublication:
            raise ModelCampaignError("qualified original model publication absent")
        original.publication.validate()
        if (
            original.canonical_intent != intent.canonical_bytes
            or original.publication.intent_sha256 != hashlib.sha256(intent.canonical_bytes).hexdigest()
            or (publication is not None and publication != original.publication)
        ):
            raise ModelCampaignError("independent original model commitment differs")
        self._origin(original.publication, original)
        return intent, original.publication

    def publish_original(self, intent):
        self._process()
        with self._lock:
            intent = self._intent(intent)
            data = _parse(intent.canonical_bytes)
            result = self._protected_read(data["planning_ref"])
            if result.status == "PRESENT":
                _, publication = self._protected_original(intent)
                self._gate()
                return publication
            self._origin(None, None)
            # Re-read after current publisher/table/retention qualification.
            result = self._protected_read(data["planning_ref"])
            if result.status == "PRESENT":
                _, publication = self._protected_original(intent)
                self._gate()
                return publication
            key = (self._binding["campaign_ref"], data["planning_ref"])
            if key in self._publication_attempts:
                raise ModelCampaignError("original model publication attempt unresolved")
            self._origin(None, None)
            self._gate()
            self._publication_attempts[key] = (uuid.uuid4().hex, intent.canonical_bytes)
            try:
                publication = self._protected.publish_first(
                    self._binding,
                    planning_ref=data["planning_ref"],
                    spec_sha256=data["spec_sha256"],
                    canonical_intent=intent.canonical_bytes,
                    qualify_io=self._gate,
                )
            except Exception:
                publication = None
            self._gate()
            _, publication = self._protected_original(intent, publication)
            self._gate()
            return publication

    def _key(self, intent):
        value = _parse(intent.canonical_bytes)
        return "MODEL_CAMPAIGN#" + value["entry"]["campaign_ref"][9:] + "#" + value["selection_digest"]

    def _row(self, intent):
        return {
            "schema_version": 1,
            "intent": _parse(intent.canonical_bytes),
            "intent_sha256": hashlib.sha256(intent.canonical_bytes).hexdigest(),
            "revision": 0,
            "state": "RESERVED",
            "registered": False,
            "attempt_token": None,
            "attempt_epoch": None,
            "attempt_holder": None,
            "attempt_deadline_ms": None,
            "outcome_arn": None,
            "error_type": None,
        }

    def _item(self, intent, row):
        return {
            "job_name": {"S": self._key(intent)},
            "journal_revision": {"N": str(row["revision"])},
            "journal_state": {"S": row["state"]},
            "private_record": {"B": _json(row)},
        }

    def _read(self, intent):
        self._protected_original(intent)
        self._gate()
        try:
            response = self._ddb.get_item(
                TableName=self._table, Key={"job_name": {"S": self._key(intent)}}, ConsistentRead=True
            )
        except Exception:
            self._gate()
            raise ModelCampaignError("original model journal read unresolved") from None
        self._gate()
        self._protected_original(intent)
        item = response.get("Item")
        if item is None:
            self._gate()
            return None
        _keys(item, {"job_name", "journal_revision", "journal_state", "private_record"})
        row = _keys(_parse(item["private_record"]["B"]), _ROW)
        if (
            _json(row["intent"]) != intent.canonical_bytes
            or row["intent_sha256"] != hashlib.sha256(intent.canonical_bytes).hexdigest()
            or type(row["schema_version"]) is not int
            or row["schema_version"] != 1
            or row["state"] not in _STATES
            or type(row["registered"]) is not bool
            or self._item(intent, row) != item
        ):
            raise ModelCampaignError("original model row/commitment replacement refused")
        _integer(row["revision"])
        if row["state"] == "RESERVED" and any(
            row[k] is not None
            for k in (
                "attempt_token",
                "attempt_epoch",
                "attempt_holder",
                "attempt_deadline_ms",
                "outcome_arn",
                "error_type",
            )
        ):
            raise ModelCampaignError("reserved model has attempt metadata")
        if row["state"] != "RESERVED":
            if (
                type(row["attempt_token"]) is not str
                or str(uuid.UUID(row["attempt_token"])) != row["attempt_token"]
            ):
                raise ModelCampaignError("original model attempt identity differs")
            _integer(row["attempt_epoch"])
            _integer(row["attempt_deadline_ms"])
            if (
                not row["registered"]
                or row["attempt_epoch"] == 0
                or type(row["attempt_holder"]) is not str
                or str(uuid.UUID(row["attempt_holder"])) != row["attempt_holder"]
                or row["revision"] < 2
            ):
                raise ModelCampaignError("model attempt state is contradictory")
        if row["registered"] and row["revision"] < 1:
            raise ModelCampaignError("model registration revision differs")
        if row["state"] == "CREATE_ATTEMPTED" and (
            row["outcome_arn"] is not None or row["error_type"] is not None
        ):
            raise ModelCampaignError("unresolved model attempt cannot claim an outcome")
        if row["state"] == "ACCEPTED" and (
            row["outcome_arn"] != _parse(intent.canonical_bytes)["prepared"]["arn"]
            or row["error_type"] is not None
        ):
            raise ModelCampaignError("accepted original model outcome differs")
        if row["state"] in {"UNKNOWN", "REJECTED"} and row["outcome_arn"] is not None:
            raise ModelCampaignError("unconfirmed model outcome cannot claim acceptance")
        if row["error_type"] is not None and (
            type(row["error_type"]) is not str or not re.fullmatch("[A-Za-z0-9_]{1,64}", row["error_type"])
        ):
            raise ModelCampaignError("model outcome error class differs")
        self._gate()
        return row

    def _reservation(self, intent, row, *, lease=None):
        self._protected_original(intent)
        self._gate()
        if lease is not None:
            self._lease_deadline(lease)
        return ModelOperationReservation(intent, row["revision"], row["state"], row["registered"], _json(row))

    def inspect_original(self, intent):
        """Qualified original-first DATA; no row or absence result grants Create."""
        self._process()
        with self._lock:
            intent = self._intent(intent)
            planning_ref = _parse(intent.canonical_bytes)["planning_ref"]
            result = self._protected_read(planning_ref)
            if result.status == "ABSENT":
                self._origin(None, None)
                result = self._protected_read(planning_ref)
            if result.status == "ABSENT":
                self._gate()
                return ModelOriginalInspection("ABSENT", None)
            self._protected_original(intent)
            row = self._read(intent)
            reservation = None if row is None else self._reservation(intent, row)
            self._gate()
            return ModelOriginalInspection("PRESENT", reservation)

    def reserve_for_submission(self, intent, *, publication):
        """Transport acknowledgement DATA; sealed attempt CAS remains mandatory."""
        self._process()
        with self._lock:
            intent, _ = self._protected_original(intent, publication)
            row = self._read(intent)
            disposition = "EXISTING"
            if row is None:
                row = self._row(intent)
                self._protected_original(intent)
                self._gate()
                try:
                    response = self._ddb.put_item(
                        TableName=self._table,
                        Item=self._item(intent, row),
                        ConditionExpression="attribute_not_exists(job_name)",
                    )
                except Exception as error:
                    observed = self._read(intent)
                    if observed is None:
                        raise ModelCampaignError("original model reservation unresolved") from None
                    from botocore.exceptions import ClientError

                    conflict = (
                        isinstance(error, ClientError)
                        and type(error.response) is dict
                        and type(error.response.get("Error")) is dict
                        and error.response["Error"].get("Code") == "ConditionalCheckFailedException"
                    )
                    # Readback can show a later valid original row; never remint.
                    row = observed
                    disposition = "EXISTING" if conflict else "UNKNOWN"
                else:
                    self._gate()
                    self._retry(response)
                    disposition = "NEW_CONFIRMED"
            result = ModelReservationOutcome(self._reservation(intent, row), disposition)
            self._gate()
            return result

    def reserve_original(self, intent, *, publication):
        """Legacy reservation DATA API, without fresh-owner inference."""
        return self.reserve_for_submission(intent, publication=publication).reservation

    @staticmethod
    def _retry(response):
        if (
            type(response) is not dict
            or type(response.get("ResponseMetadata")) is not dict
            or type(response["ResponseMetadata"].get("RetryAttempts")) is not int
            or response["ResponseMetadata"]["RetryAttempts"] != 0
        ):
            raise ModelCampaignError("model journal write transport retried or unreported")

    def _original(self, reservation):
        value = _parse(reservation.intent.canonical_bytes)
        config = value["config"]
        return OriginalCampaignOperation(
            value["operation_digest"],
            "model_acquisition",
            "Processing",
            config["account_id"],
            config["region"],
            reservation.job_name,
            reservation.job_arn,
            reservation.request_sha256,
            reservation.source_binding_sha256,
        )

    def _lease_deadline(self, lease):
        if (
            type(lease) is not CampaignLease
            or lease.process_id != os.getpid()
            or lease.campaign_ref != self._binding["campaign_ref"]
            or self._now() < lease.issued_at_ms
            or self._now() >= lease.deadline_ms
        ):
            raise ModelCampaignError("fresh original model lease required")

    def _lease(self, reservation, lease):
        self._lease_deadline(lease)
        _, witness = _binding(self.entry, self._entry_qualifier)
        with self.coordinator.qualified_io(self._gate):
            _, _, _, _, record = self.coordinator._leased(witness, lease)
        if not any(v.original == self._original(reservation) for v in record.operations):
            raise ModelCampaignError("original model coordinator registration absent")
        self._gate()
        self._lease_deadline(lease)
        return record

    def _current(self, reservation):
        if type(reservation) is not ModelOperationReservation:
            raise ModelCampaignError("original model reservation required")
        intent = self._intent(reservation.intent)
        row = self._read(intent)
        if row is None:
            raise ModelCampaignError("original model row absent")
        return intent, row

    def _cas(self, intent, prior, new, *, lease):
        self._protected_original(intent)
        self._gate()
        self._lease_deadline(lease)
        try:
            response = self._ddb.update_item(
                TableName=self._table,
                Key={"job_name": {"S": self._key(intent)}},
                ConditionExpression="journal_revision = :revision AND journal_state = :state AND private_record = :prior",
                UpdateExpression="SET journal_revision = :next, journal_state = :new_state, private_record = :body",
                ExpressionAttributeValues={
                    ":revision": {"N": str(prior["revision"])},
                    ":state": {"S": prior["state"]},
                    ":prior": {"B": _json(prior)},
                    ":next": {"N": str(new["revision"])},
                    ":new_state": {"S": new["state"]},
                    ":body": {"B": _json(new)},
                },
            )
        except Exception:
            observed = self._read(intent)
            if observed == new:
                self._gate()
                self._lease_deadline(lease)
                return new, False
            raise ModelCampaignError("model marker/outcome unresolved") from None
        self._gate()
        self._retry(response)
        self._protected_original(intent)
        self._gate()
        self._lease_deadline(lease)
        return new, True

    def register_original(self, reservation, *, lease):
        self._process()
        with self._lock:
            intent, row = self._current(reservation)
            _, witness = _binding(self.entry, self._entry_qualifier)
            self._gate()
            with self.coordinator.qualified_io(self._gate):
                self.coordinator.register_original(witness, lease, self._original(reservation))
            self._gate()
            self._lease(reservation, lease)
            if not row["registered"]:
                new = {**row, "registered": True, "revision": row["revision"] + 1}
                row, _ = self._cas(intent, row, new, lease=lease)
            return self._reservation(intent, row, lease=lease)

    def consume_create_attempt(self, reservation, *, lease):
        self._process()
        with self._lock:
            intent, row = self._current(reservation)
            self._lease(reservation, lease)
            if row["state"] != "RESERVED":
                return self._reservation(intent, row, lease=lease)
            if not row["registered"]:
                raise ModelCampaignError("original model registration required")
            token = str(uuid.uuid4())
            new = {
                **row,
                "state": "CREATE_ATTEMPTED",
                "revision": row["revision"] + 1,
                "attempt_token": token,
                "attempt_epoch": lease.epoch,
                "attempt_holder": lease.holder,
                "attempt_deadline_ms": lease.deadline_ms,
            }
            row, confirmed = self._cas(intent, row, new, lease=lease)
            if not confirmed:
                return self._reservation(intent, row, lease=lease)
            self._lease(reservation, lease)
            self._protected_original(intent)
            self._gate()
            self._lease_deadline(lease)
            permit = ModelCreatePermit(
                hashlib.sha256(intent.canonical_bytes).hexdigest(),
                token,
                os.getpid(),
                lease,
                self._owner,
                _PERMIT_SEAL,
            )
            self._permits[id(permit)] = {
                "permit": permit,
                "binding": (
                    permit.reservation_sha256,
                    permit.attempt_token,
                    permit.process_id,
                    permit.lease,
                    permit.owner,
                    permit.seal,
                ),
                "used": False,
                "outcome_written": False,
            }
            return permit

    def _permit(self, permit, reservation):
        retained = self._permits.get(id(permit))
        if (
            type(permit) is not ModelCreatePermit
            or retained is None
            or retained["permit"] is not permit
            or retained["binding"]
            != (
                permit.reservation_sha256,
                permit.attempt_token,
                permit.process_id,
                permit.lease,
                permit.owner,
                permit.seal,
            )
            or permit.owner is not self._owner
            or permit.seal is not _PERMIT_SEAL
            or permit.process_id != os.getpid()
            or permit.reservation_sha256 != hashlib.sha256(reservation.intent.canonical_bytes).hexdigest()
        ):
            raise ModelCampaignError("original same-process model permit required")
        intent, row = self._current(reservation)
        self._lease(reservation, permit.lease)
        if (
            row["state"] != "CREATE_ATTEMPTED"
            or row["attempt_token"] != permit.attempt_token
            or row["attempt_epoch"] != permit.lease.epoch
            or row["attempt_holder"] != permit.lease.holder
        ):
            raise ModelCampaignError("stale model attempt/outcome refused")
        return intent, row

    def require_effect(self, permit, reservation):
        self._process()
        with self._lock:
            intent, _ = self._permit(permit, reservation)
            self._protected_original(intent)
            retained = self._permits[id(permit)]
            if retained["used"]:
                raise ModelCampaignError("model effect permit consumed")
            self._gate()
            self._lease_deadline(permit.lease)
            retained["used"] = True
            permit.used = True

    def record_outcome(self, reservation, *, permit, disposition, native_job_arn, error_type):
        self._process()
        with self._lock:
            intent, row = self._permit(permit, reservation)
            retained = self._permits[id(permit)]
            if (
                not retained["used"]
                or retained["outcome_written"]
                or disposition not in ("ACCEPTED", "UNKNOWN", "REJECTED")
                or (
                    disposition == "ACCEPTED"
                    and (native_job_arn != reservation.job_arn or error_type is not None)
                )
                or (disposition != "ACCEPTED" and native_job_arn is not None)
            ):
                raise ModelCampaignError("original model outcome shape/owner differs")
            if error_type is not None and (
                type(error_type) is not str or not re.fullmatch("[A-Za-z0-9_]{1,64}", error_type)
            ):
                raise ModelCampaignError("bounded model outcome error class required")
            new = {
                **row,
                "revision": row["revision"] + 1,
                "state": disposition,
                "outcome_arn": native_job_arn,
                "error_type": error_type,
            }
            updated, _ = self._cas(intent, row, new, lease=permit.lease)
            retained["outcome_written"] = True
            permit.outcome_written = True
            return self._reservation(intent, updated, lease=permit.lease)

    def load_original(self, intent):
        self._process()
        with self._lock:
            if type(intent) is str:
                planning_ref = intent
                if not re.fullmatch("[A-Za-z][A-Za-z0-9_]{1,127}", planning_ref):
                    raise ModelCampaignError("original protected planning selector required")
                result = self._protected_read(planning_ref)
                if result.status != "PRESENT":
                    raise ModelCampaignError("protected original model intent absent")
                original = result.original
                intent = ModelProcessingIntent(original.canonical_intent)
                validated = self._intent(intent)
                if _parse(validated.canonical_bytes)["planning_ref"] != planning_ref:
                    raise ModelCampaignError("protected original planning identity differs")
                self._protected_original(validated, original.publication)
            intent = self._intent(intent)
            row = self._read(intent)
            if row is None:
                raise ModelCampaignError("original model journal absent")
            return self._reservation(intent, row)

    def restore_original(self, intent, *, describe_original):
        self._process()
        with self._lock:
            original = self.load_original(intent)
            if not callable(describe_original):
                raise ModelCampaignError("private read-only model observer required")
            self._protected_original(original.intent)
            self._gate()
            value = describe_original(original)
            self._gate()
            self._protected_original(original.intent)
            _keys(value, {"status", "job_name", "job_arn", "request_sha256", "source_binding_sha256"})
            if (
                value["job_name"],
                value["job_arn"],
                value["request_sha256"],
                value["source_binding_sha256"],
            ) != (
                original.job_name,
                original.job_arn,
                original.request_sha256,
                original.source_binding_sha256,
            ) or value["status"] not in {
                "NotFound",
                "Unknown",
                "InProgress",
                "Completed",
                "Failed",
                "Stopped",
            }:
                raise ModelCampaignError("original model observation replacement refused")
            self._gate()
            return ModelObservation(
                original, "UNKNOWN" if value["status"] in {"NotFound", "Unknown"} else value["status"]
            )
