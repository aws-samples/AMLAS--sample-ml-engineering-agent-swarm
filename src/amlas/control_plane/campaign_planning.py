"""Private original planning DATA publication; no native protected port is supplied."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Protocol

from .campaign_runtime import CampaignEntryContext, CampaignPlanningRef, CampaignRuntimeFactory
from .generation_state import SessionCodingGenerationStore

_MAX = 256 * 1024
_TEXT_MAX = 65536


class CampaignPlanningError(ValueError):
    """Missing or inconsistent protected planning; no effect authority."""


def _sha(value):
    if type(value) is not str or not re.fullmatch("[a-f0-9]{64}", value):
        raise CampaignPlanningError("exact planning SHA required")
    return value


def _text(value, maximum=2048):
    if type(value) is not str or not 0 < len(value) <= maximum or any(ord(c) < 32 for c in value):
        raise CampaignPlanningError("bounded planning text required")
    return value


def _ref(value):
    if type(value) is not str or not re.fullmatch("spec_[a-f0-9]{32}", value):
        raise CampaignPlanningError("original opaque generation spec ref required")
    return value


def _json(value, maximum=_MAX):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise CampaignPlanningError("bounded planning DATA encoding refused") from error
    if not 0 < len(raw) <= maximum:
        raise CampaignPlanningError("planning DATA byte bound exceeded")
    return raw


def _parse(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX:
        raise CampaignPlanningError("bounded original planning bytes required")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CampaignPlanningError("duplicate planning field")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(CampaignPlanningError("nonfinite planning")),
        )
        if _json(value) != raw:
            raise CampaignPlanningError("canonical original planning bytes differ")
        return value
    except (UnicodeError, RecursionError) as error:
        raise CampaignPlanningError("planning DATA decoding refused") from error


@dataclass(frozen=True)
class PlanningBinding:
    account_id: str
    region: str
    deployment_id: str
    deployment_prefix: str
    campaign_ref: str
    campaign_id: str
    private_store_correlation: str
    prompt_sha256: str
    initial_campaign_body_sha256: str
    initialization_publication_sha256: str

    def validate(self):
        if type(self.account_id) is not str or not re.fullmatch("[0-9]{12}", self.account_id):
            raise CampaignPlanningError("planning account differs")
        for name in (
            "region",
            "deployment_id",
            "deployment_prefix",
            "campaign_ref",
            "campaign_id",
            "private_store_correlation",
        ):
            _text(getattr(self, name), 256)
        for name in ("prompt_sha256", "initial_campaign_body_sha256", "initialization_publication_sha256"):
            _sha(getattr(self, name))


@dataclass(frozen=True)
class ProtectedPlanningAck:
    reference: str
    version_id: str
    raw_sha256: str
    spec_sha256: str
    body_sha256: str
    publication_sha256: str

    def validate(self):
        _ref(self.reference)
        _text(self.version_id, 1024)
        if self.version_id.lower() in ("null", "none"):
            raise CampaignPlanningError("exact protected planning version required")
        for name in ("raw_sha256", "spec_sha256", "body_sha256", "publication_sha256"):
            _sha(getattr(self, name))


@dataclass(frozen=True)
class ProtectedPlanningOriginal:
    ack: ProtectedPlanningAck
    canonical_body: bytes


@dataclass(frozen=True)
class ActivePlanningSelection:
    ack: ProtectedPlanningAck
    revision: int
    previous_publication_sha256: str | None

    def validate(self):
        if type(self.ack) is not ProtectedPlanningAck:
            raise CampaignPlanningError("protected planning acknowledgement required")
        self.ack.validate()
        if type(self.revision) is not int or self.revision < 1:
            raise CampaignPlanningError("original active planning revision required")
        if self.previous_publication_sha256 is not None:
            _sha(self.previous_publication_sha256)
        if (self.revision == 1) != (self.previous_publication_sha256 is None):
            raise CampaignPlanningError("original active predecessor differs")


@dataclass(frozen=True)
class ProtectedPlanningRead:
    status: str
    selection: ActivePlanningSelection | None = None
    original: ProtectedPlanningOriginal | None = None

    def validate(self):
        if type(self.status) is not str or self.status not in ("PRESENT", "ABSENT", "UNKNOWN"):
            raise CampaignPlanningError("explicit protected planning existence required")
        if self.status == "PRESENT":
            if (
                type(self.selection) is not ActivePlanningSelection
                or type(self.original) is not ProtectedPlanningOriginal
            ):
                raise CampaignPlanningError("original protected planning body and selection required")
            self.selection.validate()
            if self.original.ack != self.selection.ack:
                raise CampaignPlanningError("protected original planning selection differs")
        elif self.selection is not None or self.original is not None:
            raise CampaignPlanningError("absent/unknown planning carries no original")


@dataclass(frozen=True)
class BoundPlanningRead:
    status: str
    reference: str | None = None
    spec_sha256: str | None = None

    def validate(self):
        if type(self.status) is not str or self.status not in ("UNBOUND", "BOUND", "UNKNOWN"):
            raise CampaignPlanningError("explicit current model-spec binding required")
        if self.status == "BOUND":
            _ref(self.reference)
            _sha(self.spec_sha256)
        elif self.reference is not None or self.spec_sha256 is not None:
            raise CampaignPlanningError("unbound/unknown selection carries no model ref")


class ProtectedPlanningPort(Protocol):
    """Native implementation must independently qualify atomic protected publication."""

    def require_current(self, binding, *, selection, original, qualify_io): ...
    def read_active(self, binding, *, qualify_io) -> ProtectedPlanningRead: ...
    def read_original(self, binding, *, reference, qualify_io) -> ProtectedPlanningRead: ...
    def publish_first(
        self, binding, *, original_ref, canonical_body, expected_active, qualify_io
    ) -> ActivePlanningSelection: ...


def validate_original_planning(binding, original):
    """Pure closed DATA validation; independently qualified origin is still required."""
    from ..arbiter.coding import coding_spec_bytes, coding_spec_sha256
    from ..spec import load_spec_from_text

    if type(binding) is not PlanningBinding:
        raise CampaignPlanningError("exact planning binding DATA required")
    binding.validate()
    if type(original) is not ProtectedPlanningOriginal or type(original.ack) is not ProtectedPlanningAck:
        raise CampaignPlanningError("protected original planning DATA required")
    original.ack.validate()
    raw = original.canonical_body
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX:
        raise CampaignPlanningError("bounded original planning body required")
    if hashlib.sha256(raw).hexdigest() != original.ack.body_sha256:
        raise CampaignPlanningError("original planning body digest differs")
    value = _parse(raw)
    expected = {
        "schema_version",
        "binding",
        "original_ref",
        "raw_text",
        "raw_sha256",
        "spec_projection",
        "spec_sha256",
        "research",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["binding"] != dataclasses.asdict(binding)
        or value["original_ref"] != original.ack.reference
        or type(value["raw_text"]) is not str
        or not 0 < len(value["raw_text"].encode()) <= _TEXT_MAX
    ):
        raise CampaignPlanningError("closed original planning binding differs")
    _ref(value["original_ref"])
    raw_sha = hashlib.sha256(value["raw_text"].encode()).hexdigest()
    spec = load_spec_from_text(value["raw_text"])
    canonical = coding_spec_bytes(spec)
    digest = coding_spec_sha256(spec)
    if (
        value["raw_sha256"] != raw_sha
        or original.ack.raw_sha256 != raw_sha
        or value["spec_sha256"] != digest
        or original.ack.spec_sha256 != digest
        or _json(value["spec_projection"], _TEXT_MAX) != canonical
    ):
        raise CampaignPlanningError("original raw/canonical specification differs")
    research = value["research"]
    if type(research) is not list or len(research) > 7:
        raise CampaignPlanningError("bounded original research pointers required")
    for row in research:
        CampaignPlanningStore._research_row(row)
    return value


class CampaignPlanningStore:
    """PID-bound private DATA hook; callbacks/ports never enter serialized planning."""

    def __init__(
        self,
        entry,
        binding,
        generation_store,
        *,
        protected_port=None,
        origin_qualifier=None,
        bound_spec_reader=None,
    ):
        if (
            type(entry) is not CampaignEntryContext
            or type(entry._factory) is not CampaignRuntimeFactory
            or type(binding) is not PlanningBinding
            or type(generation_store) is not SessionCodingGenerationStore
        ):
            raise CampaignPlanningError("sealed original entry and typed planning stores required")
        entry._check()
        entry._factory._process()
        binding.validate()
        if (
            any(
                getattr(entry, name) != getattr(binding, name)
                for name in ("campaign_ref", "campaign_id", "private_store_correlation", "prompt_sha256")
            )
            or generation_store.runtime_session_id != binding.private_store_correlation
        ):
            raise CampaignPlanningError("same original planning correlation required")
        self.entry, self.binding, self._generation = entry, binding, generation_store
        self._port, self._origin_qualifier, self._bound_reader = (
            protected_port,
            origin_qualifier,
            bound_spec_reader,
        )
        self._pid, self._lock = os.getpid(), threading.RLock()
        self._research, self._attempts = [], {}
        entry.require_current()
        self._gate()

    def _gate(self):
        if self._pid != os.getpid():
            raise CampaignPlanningError("inherited planning store refused")
        self.entry._check()
        self.entry.qualify_current_io()
        witness = self.entry._factory.guard.verify()
        scope = self.entry._factory.coordinator._scope(witness)
        if any(
            scope[name] != getattr(self.binding, name)
            for name in ("account_id", "region", "deployment_id", "deployment_prefix")
        ):
            raise CampaignPlanningError("current qualified planning scope differs")

    def _call(self, name, **kwargs):
        self._gate()
        method = getattr(self._port, name, None)
        if not callable(method):
            raise CampaignPlanningError("protected planning capability missing")
        try:
            result = method(self.binding, qualify_io=self._gate, **kwargs)
        finally:
            self._gate()
        return result

    def _origin(self, selection, original):
        if not callable(self._origin_qualifier):
            raise CampaignPlanningError("independent current planning origin missing")
        evidence = self._call("require_current", selection=selection, original=original)
        self._gate()
        if self._origin_qualifier(self._port, self.binding, selection, original, evidence) is not None:
            raise CampaignPlanningError("planning origin qualifier grants nothing")
        self._gate()

    def _material(self, original):
        return validate_original_planning(self.binding, original)

    @staticmethod
    def _research_row(row):
        if type(row) is not dict or set(row) != {
            "tool",
            "url",
            "sources",
            "observed_at",
            "available_at",
            "coverage",
            "result_sha256",
            "error_sources",
            "errors_sha256",
            "license_statement",
            "license_coverage",
            "untrusted_data",
        }:
            raise CampaignPlanningError("closed research DATA pointer required")
        if row["tool"] not in ("research_search", "research_fetch_urls"):
            raise CampaignPlanningError("known research tool required")
        if row["url"] is not None:
            _text(row["url"])
            if not row["url"].startswith(("https://", "http://")):
                raise CampaignPlanningError("descriptive source URL required")
        if (
            type(row["sources"]) is not list
            or len(row["sources"]) > 4
            or any(type(item) is not str or len(item) > 64 for item in row["sources"])
            or type(row["observed_at"]) not in (int, float)
            or not math.isfinite(row["observed_at"])
            or row["observed_at"] < 0
            or row["untrusted_data"] is not True
            or row["license_statement"] is not None
            or row["license_coverage"] != "unknown"
            or type(row["error_sources"]) is not list
            or len(row["error_sources"]) > 4
            or any(type(item) is not str or len(item) > 64 for item in row["error_sources"])
        ):
            raise CampaignPlanningError("research DATA supplies no verification/license grant")
        _sha(row["result_sha256"])
        _sha(row["errors_sha256"])
        _text(row["coverage"], 256)
        if row["available_at"] is not None:
            _text(row["available_at"], 128)

    def observe_tool_result(self, name, result, *, observed_at):
        """Root's trusted SDK observer only; no public tool accepts this method."""
        self._gate()
        if name not in ("research_search", "research_fetch_urls") or type(result) is not dict:
            raise CampaignPlanningError("known returned research DATA required")
        digest = hashlib.sha256(_json(result, _TEXT_MAX)).hexdigest()
        records = result.get("results", []) if name == "research_search" else result.get("pages", [])
        if type(records) is not list:
            raise CampaignPlanningError("bounded returned research rows required")
        selected = []
        errors = result.get("errors", {})
        errors_raw = _json(errors, 4096)
        error_sources = list(errors) if type(errors) is dict else (["provided_urls"] if errors else [])
        for item in records[:7] if records else [{"url": None}]:
            if type(item) is not dict:
                raise CampaignPlanningError("research pointer object required")
            row = {
                "tool": name,
                "url": item.get("url"),
                "sources": item.get("sources", result.get("queried_sources", []))
                if name == "research_search"
                else ["supplied_public_url"],
                "observed_at": observed_at,
                "available_at": item.get("published"),
                "coverage": str(item.get("content_coverage", result.get("coverage", "unknown"))),
                "result_sha256": digest,
                "error_sources": error_sources,
                "errors_sha256": hashlib.sha256(errors_raw).hexdigest(),
                "license_statement": None,
                "license_coverage": "unknown",
                "untrusted_data": True,
            }
            self._research_row(row)
            selected.append(row)
        with self._lock:
            for row in selected:
                if row not in self._research:
                    if len(self._research) >= 7:
                        raise CampaignPlanningError("planning research pointer population exceeded")
                    self._research.append(row)
            self._gate()

    def _read(self):
        self._origin(None, None)  # Configuration qualification never establishes absence.
        read = self._call("read_active")
        if type(read) is not ProtectedPlanningRead:
            raise CampaignPlanningError("qualified protected planning status absent")
        read.validate()
        if read.status == "UNKNOWN":
            raise CampaignPlanningError("protected original planning existence unresolved")
        self._origin(read.selection, read.original)
        if read.status == "PRESENT":
            self._material(read.original)
        return read

    def _returned_ref(self, read):
        value = self._material(read.original)
        self._gate()
        return CampaignPlanningRef("spec", value["original_ref"], value["spec_sha256"])

    def publish_validated_spec(self, yaml_text):
        self._gate()
        with self._lock:
            read = self._read()
            selected = self._generation.bind_spec(yaml_text)
            ref = selected["spec_ref"]
            retained_text, canonical, digest = self._generation._specs[ref]
            if retained_text != yaml_text.encode():
                raise CampaignPlanningError("original semantic spec text cannot be silently replaced")
            value = {
                "schema_version": 1,
                "binding": dataclasses.asdict(self.binding),
                "original_ref": ref,
                "raw_text": yaml_text,
                "raw_sha256": hashlib.sha256(yaml_text.encode()).hexdigest(),
                "spec_projection": json.loads(canonical),
                "spec_sha256": digest,
                "research": list(self._research),
            }
            raw = _json(value)
            body_sha = hashlib.sha256(raw).hexdigest()
            if read.status == "PRESENT":
                if read.original.canonical_body == raw:
                    return self._returned_ref(read)
                if not callable(self._bound_reader):
                    raise CampaignPlanningError("current original model binding unknown")
                self._gate()
                bound = self._bound_reader(self.binding, qualify_io=self._gate)
                self._gate()
                if type(bound) is not BoundPlanningRead:
                    raise CampaignPlanningError("qualified model absence required for active update")
                bound.validate()
                if bound.status != "UNBOUND":
                    raise CampaignPlanningError("active update cannot redirect original model spec")
            if body_sha in self._attempts:
                raise CampaignPlanningError("uncertain original planning publication never replays")
            self._origin(read.selection, read.original)
            self._attempts[body_sha] = ref
            returned = None
            try:
                returned = self._call(
                    "publish_first", original_ref=ref, canonical_body=raw, expected_active=read.selection
                )
            except Exception:
                # One current read may resolve a lost acknowledgement; never retry the publication.
                resolved = self._read()
                if resolved.status != "PRESENT" or resolved.original.canonical_body != raw:
                    raise CampaignPlanningError("original planning publication unresolved") from None
                return self._returned_ref(resolved)
            resolved = self._read()
            if (
                type(returned) is not ActivePlanningSelection
                or resolved.status != "PRESENT"
                or returned != resolved.selection
                or resolved.original.canonical_body != raw
            ):
                raise CampaignPlanningError("independent original planning acknowledgement differs")
            returned.validate()
            if returned.revision != (
                1 if read.selection is None else read.selection.revision + 1
            ) or returned.previous_publication_sha256 != (
                None if read.selection is None else read.selection.ack.publication_sha256
            ):
                raise CampaignPlanningError("original active selection predecessor differs")
            return self._returned_ref(resolved)

    def load_original_refs(self, *, campaign_ref, private_store_correlation):
        self._gate()
        if (campaign_ref, private_store_correlation) != (
            self.binding.campaign_ref,
            self.binding.private_store_correlation,
        ):
            raise CampaignPlanningError("original planning reader scope differs")
        with self._lock:
            read = self._read()
            if read.status == "ABSENT":
                return ()
            value = self._material(read.original)
            refs = [self._returned_ref(read)]
            for row in value["research"]:
                raw = _json(row, _TEXT_MAX)
                digest = hashlib.sha256(raw).hexdigest()
                refs.append(CampaignPlanningRef("research", "research_" + digest[:32], digest))
            self._gate()
            return tuple(refs)

    def restore_original_spec(self):
        self._gate()
        with self._lock:
            read = self._read()
            if read.status != "PRESENT":
                raise CampaignPlanningError("original planning absent")
            value = self._material(read.original)
            self._generation.restore_original_spec(
                value["original_ref"], value["raw_text"], expected_spec_sha256=value["spec_sha256"]
            )
            self._gate()
            return self._returned_ref(read)
