"""Bounded native enum projection; never retain trace, matches or model content."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

MAX_RECORDS = 128
MAX_GUARDRAILS = 8
MAX_OUTPUTS = 16
_CONTENT = frozenset({"INSULTS", "HATE", "SEXUAL", "VIOLENCE", "MISCONDUCT", "PROMPT_ATTACK"})
_PII = frozenset(
    {
        "ADDRESS",
        "AGE",
        "AWS_ACCESS_KEY",
        "AWS_SECRET_KEY",
        "CA_HEALTH_NUMBER",
        "CA_SOCIAL_INSURANCE_NUMBER",
        "CREDIT_DEBIT_CARD_CVV",
        "CREDIT_DEBIT_CARD_EXPIRY",
        "CREDIT_DEBIT_CARD_NUMBER",
        "DRIVER_ID",
        "EMAIL",
        "INTERNATIONAL_BANK_ACCOUNT_NUMBER",
        "IP_ADDRESS",
        "LICENSE_PLATE",
        "MAC_ADDRESS",
        "NAME",
        "PASSWORD",
        "PHONE",
        "PIN",
        "SWIFT_CODE",
        "UK_NATIONAL_HEALTH_SERVICE_NUMBER",
        "UK_NATIONAL_INSURANCE_NUMBER",
        "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER",
        "URL",
        "USERNAME",
        "US_BANK_ACCOUNT_NUMBER",
        "US_BANK_ROUTING_NUMBER",
        "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER",
        "US_PASSPORT_NUMBER",
        "US_SOCIAL_SECURITY_NUMBER",
        "VEHICLE_IDENTIFICATION_NUMBER",
    }
)
_LEVELS = frozenset({"NONE", "LOW", "MEDIUM", "HIGH"})
_ACTIONS = {"CONTENT": frozenset({"BLOCKED", "NONE"}), "PII": frozenset({"BLOCKED", "ANONYMIZED", "NONE"})}
_STATUSES = frozenset({"complete", "partial", "invalid", "unavailable"})


def _mapping(value, limit=16):
    return type(value) is dict and len(value) <= limit and all(type(key) is str for key in value)


@dataclass(frozen=True)
class SafeAssessment:
    direction: str
    policy: str
    category: str
    action: str
    confidence: str | None = None
    detected: bool | None = None
    filter_strength: str | None = None

    def __post_init__(self):
        values = (self.direction, self.policy, self.category, self.action)
        if (
            any(type(value) is not str for value in values)
            or self.direction not in {"INPUT", "OUTPUT"}
            or self.policy not in _ACTIONS
        ):
            raise ValueError("closed assessment enums required")
        if (
            self.category not in (_CONTENT if self.policy == "CONTENT" else _PII)
            or self.action not in _ACTIONS[self.policy]
        ):
            raise ValueError("closed assessment category/action required")
        if self.detected is not None and type(self.detected) is not bool:
            raise ValueError("exact optional detected required")
        for value in (self.confidence, self.filter_strength):
            if value is not None and (type(value) is not str or value not in _LEVELS):
                raise ValueError("closed assessment level required")

    def to_dict(self):
        return {
            "direction": self.direction,
            "policy": self.policy,
            "category": self.category,
            "action": self.action,
            "confidence": self.confidence,
            "detected": self.detected,
            "filter_strength": self.filter_strength,
        }


@dataclass(frozen=True)
class GuardrailAssessment:
    assessment_status: str
    records: tuple[SafeAssessment, ...] = ()

    def __post_init__(self):
        if (
            type(self.assessment_status) is not str
            or self.assessment_status not in _STATUSES
            or type(self.records) is not tuple
            or len(self.records) > MAX_RECORDS
            or any(type(item) is not SafeAssessment for item in self.records)
        ):
            raise ValueError("closed bounded assessment summary required")

    @property
    def action(self):
        values = {item.action for item in self.records}
        return next(iter(values)) if self.assessment_status == "complete" and len(values) == 1 else "UNKNOWN"

    def to_dict(self):
        return {
            "schema_version": 1,
            "assessment_status": self.assessment_status,
            "action": self.action,
            "records": [item.to_dict() for item in self.records],
        }


def project_guardrail_trace(value) -> GuardrailAssessment:
    """Read selected native primitives only; ignored leaf values are never visited."""
    if not _mapping(value) or not _mapping(value.get("guardrail")):
        return GuardrailAssessment("unavailable")
    root = value["guardrail"]
    records = []
    status = (
        "partial"
        if set(root) - {"inputAssessment", "outputAssessments", "modelOutput", "actionReason"}
        else "complete"
    )

    def degrade(new):
        nonlocal status
        if new == "invalid" or status != "invalid":
            status = new

    def entries(items, direction, policy):
        if type(items) is not list:
            degrade("invalid")
            return
        if len(items) > MAX_RECORDS:
            degrade("partial")
        for item in items[:MAX_RECORDS]:
            if len(records) >= MAX_RECORDS:
                degrade("partial")
                break
            if not _mapping(item):
                degrade("invalid")
                continue
            allowed = (
                {"type", "action", "detected", "confidence", "filterStrength"}
                if policy == "CONTENT"
                else {"type", "action", "detected", "match"}
            )
            if set(item) - allowed:
                degrade("partial")
            detected = item.get("detected")
            if "detected" in item and type(detected) is not bool:
                degrade("invalid")
                continue
            try:
                record = SafeAssessment(
                    direction,
                    policy,
                    item.get("type"),
                    item.get("action"),
                    item.get("confidence") if policy == "CONTENT" else None,
                    detected,
                    item.get("filterStrength") if policy == "CONTENT" else None,
                )
                if policy == "CONTENT" and record.confidence is None:
                    raise ValueError("content confidence missing")
            except ValueError:
                degrade("invalid")
                continue
            records.append(record)
            if detected is False and record.action != "NONE":
                degrade("invalid")

    def assessment(item, direction):
        prior_count = len(records)
        if not _mapping(item):
            degrade("invalid")
            return
        if set(item) - {
            "contentPolicy",
            "sensitiveInformationPolicy",
            "invocationMetrics",
            "appliedGuardrailDetails",
        }:
            degrade("partial")
        if "contentPolicy" in item:
            content = item["contentPolicy"]
            if not _mapping(content) or set(content) != {"filters"}:
                degrade("invalid")
            else:
                entries(content["filters"], direction, "CONTENT")
        if "sensitiveInformationPolicy" in item:
            pii = item["sensitiveInformationPolicy"]
            if not _mapping(pii) or not {"piiEntities", "regexes"}.issubset(pii):
                degrade("invalid")
            else:
                if set(pii) - {"piiEntities", "regexes"}:
                    degrade("partial")
                if "regexes" in pii and (type(pii["regexes"]) is not list or pii["regexes"]):
                    degrade("partial")
                entries(pii["piiEntities"], direction, "PII")
        if len(records) == prior_count:
            degrade("partial")

    for field, direction in (("inputAssessment", "INPUT"), ("outputAssessments", "OUTPUT")):
        if field not in root:
            continue
        collection = root[field]
        if not _mapping(collection, MAX_GUARDRAILS):
            degrade("invalid")
            continue
        for item in collection.values():
            if direction == "INPUT":
                assessment(item, direction)
            elif type(item) is not list:
                degrade("invalid")
            else:
                if len(item) > MAX_OUTPUTS:
                    degrade("partial")
                for output in item[:MAX_OUTPUTS]:
                    assessment(output, direction)
    if not records and status != "invalid":
        status = "unavailable"
    return GuardrailAssessment(status, tuple(records))


class GuardrailAssessmentCapture:
    """Per-turn synchronized primitive-only capture; never an authority context."""

    def __init__(self):
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._active = True
        self._seen = False
        self._records = []
        self._status = "complete"
        self._cycles = 0
        self._covered_cycles = set()

    def observe(self, kwargs):
        if os.getpid() != self._pid:
            return
        with self._lock:
            if not self._active:
                return
            event = kwargs.get("event")
            if not _mapping(event):
                return
            if "messageStart" in event:
                self._cycles += 1
                if self._cycles > 64:
                    self._status = "partial" if self._status != "invalid" else "invalid"
            if not _mapping(event.get("metadata")):
                return
            metadata = event["metadata"]
            if "trace" not in metadata:
                return
            summary = project_guardrail_trace(metadata["trace"])
            self._seen = True
            if 0 < self._cycles <= 64:
                self._covered_cycles.add(self._cycles)
            else:
                self._status = "partial" if self._status != "invalid" else "invalid"
            if summary.assessment_status == "invalid":
                self._status = "invalid"
            elif summary.assessment_status != "complete" and self._status != "invalid":
                self._status = "partial"
            available = MAX_RECORDS - len(self._records)
            if len(summary.records) > available:
                self._status = "partial" if self._status != "invalid" else "invalid"
            self._records.extend(summary.records[:available])

    def wrap(self, callback):
        def composed(**kwargs):
            try:
                self.observe(kwargs)
            except Exception:
                # Observability failure never changes the original SDK handler,
                # captures an exception/value, retries or alters model behavior.
                if os.getpid() == self._pid:
                    with self._lock:
                        if self._active:
                            self._seen = True
                            self._status = "invalid"
            return callback(**kwargs)

        return composed

    def close(self):
        if os.getpid() != self._pid:
            return None
        with self._lock:
            if not self._active:
                return None
            self._active = False
            if self._seen and len(self._covered_cycles) != self._cycles and self._status != "invalid":
                self._status = "partial"
            status = "unavailable" if not self._records and self._status != "invalid" else self._status
            result = GuardrailAssessment(status, tuple(self._records)) if self._seen else None
            self._records.clear()
            self._covered_cycles.clear()
            return result
