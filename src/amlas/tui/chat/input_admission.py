"""Private pre-exposure INPUT decisions; no text ledger, client factory or authority."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from botocore import exceptions as sdk_errors

from .guardrail_assessment import _CONTENT, _LEVELS, _PII, project_guardrail_trace

MAX_INPUT_BYTES = 65536
MAX_RECORDS = 128
_BOUNDARIES = frozenset(
    {"OWNER_PROMPT", "TOOL_RESULT", "GENERATED_CONTEXT", "HISTORY_TEXT", "COMPLETION_DATA"}
)
_OUTCOMES = frozenset({"ALLOW", "ANONYMIZED", "BLOCKED", "UNKNOWN"})
_REASONS = frozenset(
    {
        "NATIVE_NONE",
        "NATIVE_MASKED",
        "NATIVE_BLOCKED",
        "INVALID_INPUT",
        "INPUT_LIMIT",
        "NATIVE_INVALID",
        "NATIVE_UNKNOWN",
        "MASK_INVALID",
        "TRANSPORT_UNKNOWN",
    }
)
_DOMAIN = b"amlas.input_admission.v1\x00"


def _mapping(value, limit=32):
    return type(value) is dict and len(value) <= limit and all(type(key) is str for key in value)


@dataclass(frozen=True)
class InputAdmissionConfig:
    guardrail_id: str
    guardrail_version: str
    region: str

    def __post_init__(self):
        if (
            type(self.guardrail_id) is not str
            or re.fullmatch(r"[a-z0-9]{1,64}", self.guardrail_id) is None
            or type(self.guardrail_version) is not str
            or re.fullmatch(r"DRAFT|[1-9][0-9]{0,7}", self.guardrail_version) is None
            or type(self.region) is not str
            or re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+", self.region) is None
        ):
            raise ValueError("selected immutable guard identity and region required")


_TRANSPORT_CLASSES = {
    sdk_errors.ReadTimeoutError: "ReadTimeoutError",
    sdk_errors.ConnectTimeoutError: "ConnectTimeoutError",
    sdk_errors.EndpointConnectionError: "EndpointConnectionError",
    sdk_errors.ConnectionClosedError: "ConnectionClosedError",
    sdk_errors.SSLError: "SSLError",
    sdk_errors.NoCredentialsError: "NoCredentialsError",
    sdk_errors.PartialCredentialsError: "PartialCredentialsError",
    sdk_errors.CredentialRetrievalError: "CredentialRetrievalError",
    sdk_errors.TokenRetrievalError: "TokenRetrievalError",
    sdk_errors.ParamValidationError: "ParamValidationError",
    TimeoutError: "TimeoutError",
}
_TRANSPORT_CODES = frozenset(
    {
        "AccessDeniedException",
        "ThrottlingException",
        "InternalServerException",
        "ServiceUnavailableException",
        "ValidationException",
        "ServiceQuotaExceededException",
        "ExpiredToken",
        "ExpiredTokenException",
        "InvalidClientTokenId",
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "SignatureDoesNotMatch",
        "RequestExpired",
        "RequestTimeout",
    }
)
_REQUEST_ID = r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"


@dataclass(frozen=True)
class InputTransportDiagnostic:
    error_class: str = "UNKNOWN"
    native_code: str = "UNKNOWN"
    http_status: int | None = None
    request_id: str | None = None
    retry_attempts: int | None = None

    def __post_init__(self):
        if (
            type(self.error_class) is not str
            or self.error_class not in {"UNKNOWN", "ClientError", *_TRANSPORT_CLASSES.values()}
            or type(self.native_code) is not str
            or self.native_code not in {"UNKNOWN", *_TRANSPORT_CODES}
            or (
                self.http_status is not None
                and (type(self.http_status) is not int or not 100 <= self.http_status <= 599)
            )
            or (
                self.request_id is not None
                and (type(self.request_id) is not str or re.fullmatch(_REQUEST_ID, self.request_id) is None)
            )
            or (
                self.retry_attempts is not None
                and (type(self.retry_attempts) is not int or not 0 <= self.retry_attempts <= 100)
            )
        ):
            raise ValueError("closed guard transport diagnostic required")

    def to_dict(self):
        return {
            "phase": "APPLY_GUARDRAIL",
            "error_class": self.error_class,
            "native_code": self.native_code,
            "http_status": self.http_status,
            "request_id": self.request_id,
            "retry_attempts": self.retry_attempts,
        }


def _transport_diagnostic(error):
    """Finite labels only; never stringify errors or retain provider bodies."""
    if not isinstance(error, sdk_errors.ClientError):
        return InputTransportDiagnostic(error_class=_TRANSPORT_CLASSES.get(type(error), "UNKNOWN"))
    response = getattr(error, "response", None)
    if not _mapping(response):
        return InputTransportDiagnostic(error_class="ClientError")
    native = response.get("Error")
    metadata = response.get("ResponseMetadata")
    code = native.get("Code") if _mapping(native) else None
    status = metadata.get("HTTPStatusCode") if _mapping(metadata) else None
    request = metadata.get("RequestId") if _mapping(metadata) else None
    retries = metadata.get("RetryAttempts") if _mapping(metadata) else None
    return InputTransportDiagnostic(
        error_class="ClientError",
        native_code=code if type(code) is str and code in _TRANSPORT_CODES else "UNKNOWN",
        http_status=status if type(status) is int and 100 <= status <= 599 else None,
        request_id=request if type(request) is str and re.fullmatch(_REQUEST_ID, request) else None,
        retry_attempts=retries if type(retries) is int and 0 <= retries <= 100 else None,
    )


@dataclass(frozen=True)
class InputAdmissionReceipt:
    check_id: str
    opaque_cycle: str
    boundary: str
    guardrail_id: str
    guardrail_version: str
    region: str
    byte_count: int | None
    keyed_digest: str | None
    outcome: str
    reason: str
    categories: tuple[str, ...] = ()
    tool_name: str = "UNKNOWN"
    character_count: int | None = None
    byte_measurement: str = "UNAVAILABLE"
    byte_count_lower_bound: int | None = None
    transport_diagnostic: InputTransportDiagnostic | None = None
    blocking_categories: tuple[str, ...] = ()
    limit_bytes: int = field(default=MAX_INPUT_BYTES, init=False)
    schema_version: int = field(default=1, init=False)
    source: str = field(default="INPUT", init=False)
    data_only: bool = field(default=True, init=False)
    native_admission: bool = field(default=False, init=False)
    quality_success: bool = field(default=False, init=False)
    # Outside the closed field inventory, the constructor and to_dict: what the guardrail reported for a text
    # that was not admitted unchanged, as a GuardrailAssessment of closed enums. The checker sets it beside the
    # receipt for readers in this process. It never holds a match, an output or a text.
    assessment = None

    def __post_init__(self):
        if (
            (
                self.transport_diagnostic is not None
                and (
                    type(self.transport_diagnostic) is not InputTransportDiagnostic
                    or self.reason != "TRANSPORT_UNKNOWN"
                    or self.outcome != "UNKNOWN"
                )
            )
            or type(self.check_id) is not str
            or re.fullmatch(r"[a-f0-9]{32}", self.check_id) is None
            or type(self.opaque_cycle) is not str
            or re.fullmatch(r"[a-f0-9]{32}", self.opaque_cycle) is None
            or type(self.boundary) is not str
            or self.boundary not in _BOUNDARIES
            or type(self.outcome) is not str
            or self.outcome not in _OUTCOMES
            or type(self.reason) is not str
            or self.reason not in _REASONS
            or (self.byte_count is not None and (type(self.byte_count) is not int or self.byte_count < 0))
            or (
                self.keyed_digest is not None
                and (
                    type(self.keyed_digest) is not str
                    or re.fullmatch(r"[a-f0-9]{64}", self.keyed_digest) is None
                )
            )
            or type(self.tool_name) is not str
            or re.fullmatch(r"UNKNOWN|[a-z][a-z0-9_]{0,63}", self.tool_name) is None
            or (self.boundary != "TOOL_RESULT" and self.tool_name != "UNKNOWN")
            or (
                self.character_count is not None
                and (type(self.character_count) is not int or self.character_count < 0)
            )
            or type(self.byte_measurement) is not str
            or self.byte_measurement not in {"EXACT", "LOWER_BOUND", "UNAVAILABLE"}
            or (
                self.byte_measurement == "EXACT"
                and (self.byte_count is None or self.byte_count_lower_bound is not None)
            )
            or (
                self.byte_measurement == "LOWER_BOUND"
                and (
                    self.byte_count is not None
                    or type(self.byte_count_lower_bound) is not int
                    or not MAX_INPUT_BYTES < self.byte_count_lower_bound <= 4 * (MAX_INPUT_BYTES + 1)
                )
            )
            or (
                self.byte_measurement == "UNAVAILABLE"
                and (self.byte_count is not None or self.byte_count_lower_bound is not None)
            )
            or type(self.categories) is not tuple
            or len(self.categories) > MAX_RECORDS
            or any(
                type(item) is not str
                or item not in {*("CONTENT:" + v for v in _CONTENT), *("PII:" + v for v in _PII)}
                for item in self.categories
            )
        ):
            raise ValueError("closed content-free admission receipt required")
        if (
            type(self.blocking_categories) is not tuple
            or len(self.blocking_categories) > MAX_RECORDS
            or any(
                type(item) is not str
                or item not in {*("CONTENT:" + v for v in _CONTENT), *("PII:" + v for v in _PII)}
                for item in self.blocking_categories
            )
            or tuple(sorted(set(self.blocking_categories))) != self.blocking_categories
            or not set(self.blocking_categories) <= set(self.categories)
            or (self.blocking_categories and (self.outcome != "BLOCKED" or self.reason != "NATIVE_BLOCKED"))
        ):
            raise ValueError("closed blocking-category diagnostic required")
        InputAdmissionConfig(self.guardrail_id, self.guardrail_version, self.region)

    def to_dict(self):
        result = {
            "schema_version": self.schema_version,
            "check_id": self.check_id,
            "opaque_cycle": self.opaque_cycle,
            "boundary": self.boundary,
            "source": self.source,
            "guardrail_id": self.guardrail_id,
            "guardrail_version": self.guardrail_version,
            "region": self.region,
            "byte_count": self.byte_count,
            "tool_name": self.tool_name,
            "limit_bytes": self.limit_bytes,
            "character_count": self.character_count,
            "byte_measurement": self.byte_measurement,
            "byte_count_lower_bound": self.byte_count_lower_bound,
            "keyed_digest": self.keyed_digest,
            "outcome": self.outcome,
            "reason": self.reason,
            "categories": list(self.categories),
            "data_only": self.data_only,
            "native_admission": self.native_admission,
            "quality_success": self.quality_success,
        }
        if self.transport_diagnostic is not None:
            result["transport_diagnostic"] = self.transport_diagnostic.to_dict()
        # Absent legacy diagnostics remain unavailable; never infer from the union.
        if self.blocking_categories:
            result["blocking_categories"] = list(self.blocking_categories)
        return result


@dataclass(frozen=True)
class InputAdmissionDecision:
    outcome: str
    forward_text: str | None = field(repr=False)
    receipt: InputAdmissionReceipt

    def __post_init__(self):
        if (
            type(self.receipt) is not InputAdmissionReceipt
            or type(self.outcome) is not str
            or self.outcome not in _OUTCOMES
            or self.outcome != self.receipt.outcome
            or (self.outcome in {"ALLOW", "ANONYMIZED"} and type(self.forward_text) is not str)
            or (self.outcome in {"BLOCKED", "UNKNOWN"} and self.forward_text is not None)
        ):
            raise ValueError("closed admission decision required")


@dataclass(frozen=True, init=False, repr=False)
class InputAdmissionChecker:
    config: InputAdmissionConfig
    _apply: Callable = field(repr=False)
    _key: bytes = field(repr=False)
    _close: Callable | None = field(repr=False)

    def __init__(self, config, *, client, hmac_key, close_before_assessment=False):
        if (
            type(config) is not InputAdmissionConfig
            or type(hmac_key) is not bytes
            or not 32 <= len(hmac_key) <= 64
            or type(close_before_assessment) is not bool
        ):
            raise ValueError("selected config and ephemeral binary HMAC key required")
        method = getattr(client, "apply_guardrail", None)
        if not callable(method):
            raise ValueError("selected guard client required")
        close = getattr(client, "close", None) if close_before_assessment else None
        if close_before_assessment and not callable(close):
            raise ValueError("owned guard client pool closure required")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "_apply", method)
        object.__setattr__(self, "_key", hmac_key)
        object.__setattr__(self, "_close", close)

    def _with_client(self, client):
        """Trusted owner transport replacement; immutable policy/key stay identical."""
        return InputAdmissionChecker(
            self.config,
            client=client,
            hmac_key=self._key,
            close_before_assessment=self._close is not None,
        )

    def check(self, text, *, boundary, opaque_cycle, tool_name="UNKNOWN"):
        if (
            type(boundary) is not str
            or boundary not in _BOUNDARIES
            or type(opaque_cycle) is not str
            or re.fullmatch(r"[a-f0-9]{32}", opaque_cycle) is None
        ):
            raise ValueError("closed boundary and source-owned opaque cycle required")
        size = digest = lower_bound = None
        characters = len(text) if type(text) is str else None
        measurement = "UNAVAILABLE"
        categories = blocking_categories = ()
        native = None

        def decision(outcome, reason, body=None, transport=None):
            receipt = InputAdmissionReceipt(
                secrets.token_hex(16),
                opaque_cycle,
                boundary,
                self.config.guardrail_id,
                self.config.guardrail_version,
                self.config.region,
                size,
                digest,
                outcome,
                reason,
                categories,
                tool_name=tool_name,
                character_count=characters,
                byte_measurement=measurement,
                byte_count_lower_bound=lower_bound,
                transport_diagnostic=transport,
                blocking_categories=blocking_categories,
            )
            if native is not None and outcome != "ALLOW":
                object.__setattr__(receipt, "assessment", _reported_assessment(native))
            return InputAdmissionDecision(outcome, body, receipt)

        if type(text) is not str or not text:
            return decision("UNKNOWN", "INVALID_INPUT")
        if len(text) > MAX_INPUT_BYTES:
            try:
                lower_bound = len(text[: MAX_INPUT_BYTES + 1].encode("utf-8", errors="strict"))
                measurement = "LOWER_BOUND"
            except UnicodeEncodeError:
                pass  # The unchanged size rejection retains unavailable byte measurement.
            return decision("UNKNOWN", "INPUT_LIMIT")
        try:
            raw = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return decision("UNKNOWN", "INVALID_INPUT")
        size = len(raw)
        measurement = "EXACT"
        if size > MAX_INPUT_BYTES:
            return decision("UNKNOWN", "INPUT_LIMIT")
        identity = json.dumps(
            [
                self.config.guardrail_id,
                self.config.guardrail_version,
                self.config.region,
                boundary,
                opaque_cycle,
                size,
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest = hmac.new(self._key, _DOMAIN + identity + b"\x00" + raw, hashlib.sha256).hexdigest()
        try:
            if self._close is not None:
                # Retire idle owned connections before this assessment's sole attempt.
                self._close()
            response = self._apply(
                guardrailIdentifier=self.config.guardrail_id,
                guardrailVersion=self.config.guardrail_version,
                source="INPUT",
                outputScope="FULL",
                content=[{"text": {"text": text}}],
            )
        except Exception as error:
            try:
                transport = _transport_diagnostic(error)
            except Exception:
                transport = InputTransportDiagnostic()
            return decision("UNKNOWN", "TRANSPORT_UNKNOWN", transport=transport)
        # The service counts guardrailCoverage.textCharacters in UTF-16 code units. The strict
        # UTF-8 encode above refused any surrogate code point, so this encode cannot fail.
        parsed = _parse_response(response, self.config, len(text.encode("utf-16-le")) // 2)
        if parsed is None:
            return decision("UNKNOWN", "NATIVE_INVALID")
        action, records, masks = parsed
        native = response
        categories = tuple(sorted({policy + ":" + category for policy, category, _, _ in records}))
        if any(record_action == "NONE" and detected is True for _, _, record_action, detected in records):
            return decision("UNKNOWN", "NATIVE_INVALID")
        if any(record_action == "BLOCKED" for _, _, record_action, _ in records):
            blocking_categories = tuple(
                sorted(
                    {
                        policy + ":" + category
                        for policy, category, record_action, _ in records
                        if record_action == "BLOCKED"
                    }
                )
            )
            return decision("BLOCKED", "NATIVE_BLOCKED")
        if action == "NONE":
            outputs = response.get("outputs", [])
            if type(outputs) is not list or outputs:
                return decision("UNKNOWN", "NATIVE_INVALID")
            if any(record_action != "NONE" for _, _, record_action, _ in records):
                return decision("UNKNOWN", "NATIVE_INVALID")
            return decision("ALLOW", "NATIVE_NONE", text)
        if not masks:
            return decision("UNKNOWN", "NATIVE_UNKNOWN")
        replacement = _service_mask(response, text, masks)
        if replacement is None:
            return decision("UNKNOWN", "MASK_INVALID")
        return decision("ANONYMIZED", "NATIVE_MASKED", replacement)


def _reported_assessment(response):
    """Closed enums of a validated native answer; its matches, outputs and texts are never read."""
    try:
        collection = {str(index): item for index, item in enumerate(response["assessments"])}
        return project_guardrail_trace({"guardrail": {"inputAssessment": collection}})
    except Exception:
        # A projection for a later reader never changes the decision of this check.
        return None


def _parse_response(response, config, utf16_units):
    if not _mapping(response):
        return None
    action = response.get("action")
    if type(action) is not str or action not in {"NONE", "GUARDRAIL_INTERVENED"}:
        return None
    metadata = response.get("ResponseMetadata")
    if not _mapping(metadata):
        return None
    if type(metadata.get("HTTPStatusCode")) is not int or metadata["HTTPStatusCode"] != 200:
        return None
    if type(metadata.get("RetryAttempts")) is not int or metadata["RetryAttempts"] != 0:
        return None
    coverage = response.get("guardrailCoverage")
    if coverage is not None:
        counts = coverage.get("textCharacters") if _mapping(coverage) else None
        if not _mapping(counts) or any(type(counts.get(k)) is not int for k in ("guarded", "total")):
            return None
        if counts["guarded"] != counts["total"] or counts["total"] != utf16_units:
            return None
    assessments = response.get("assessments")
    if type(assessments) is not list or not 1 <= len(assessments) <= MAX_RECORDS:
        return None
    records, masks = [], []
    for assessment in assessments:
        if not _mapping(assessment) or set(assessment) - {
            "contentPolicy",
            "sensitiveInformationPolicy",
            "invocationMetrics",
            "appliedGuardrailDetails",
        }:
            return None
        details = assessment.get("appliedGuardrailDetails")
        if details is not None:
            if (
                not _mapping(details)
                or details.get("guardrailId") != config.guardrail_id
                or details.get("guardrailVersion") != config.guardrail_version
            ):
                return None
        for policy_key, label in (("contentPolicy", "CONTENT"), ("sensitiveInformationPolicy", "PII")):
            if policy_key not in assessment:
                continue
            policy = assessment[policy_key]
            if not _mapping(policy):
                return None
            key = "filters" if label == "CONTENT" else "piiEntities"
            allowed_keys = {"filters"} if label == "CONTENT" else {"piiEntities", "regexes"}
            if set(policy) - allowed_keys or type(policy.get(key)) is not list:
                return None
            if (
                label == "PII"
                and "regexes" in policy
                and (type(policy["regexes"]) is not list or policy["regexes"])
            ):
                return None
            entries = policy[key]
            if len(records) + len(entries) > MAX_RECORDS:
                return None
            for entry in entries:
                allowed = (
                    {"type", "action", "confidence", "detected", "filterStrength"}
                    if label == "CONTENT"
                    else {"type", "action", "detected", "match"}
                )
                if not _mapping(entry) or set(entry) - allowed:
                    return None
                category, record_action = entry.get("type"), entry.get("action")
                if type(category) is not str or category not in (_CONTENT if label == "CONTENT" else _PII):
                    return None
                if type(record_action) is not str or record_action not in (
                    {"NONE", "BLOCKED"} if label == "CONTENT" else {"NONE", "BLOCKED", "ANONYMIZED"}
                ):
                    return None
                detected = entry.get("detected")
                if "detected" in entry and type(detected) is not bool:
                    return None
                if label == "CONTENT":
                    for level in ("confidence", "filterStrength"):
                        if type(entry.get(level)) is not str or entry[level] not in _LEVELS:
                            return None
                if detected is False and record_action != "NONE":
                    return None
                records.append((label, category, record_action, detected))
                if record_action == "ANONYMIZED":
                    match = entry.get("match")
                    if type(match) is not str or not match or len(match) > MAX_INPUT_BYTES:
                        return None
                    masks.append((category, match))
    return (action, records, masks) if records else None


def _service_mask(response, original, masks):
    outputs = response.get("outputs")
    if (
        type(outputs) is not list
        or len(outputs) != 1
        or not _mapping(outputs[0])
        or set(outputs[0]) != {"text"}
    ):
        return None
    replacement = outputs[0]["text"]
    if type(replacement) is not str or not replacement or len(replacement) > MAX_INPUT_BYTES:
        return None
    try:
        if len(replacement.encode("utf-8", errors="strict")) > MAX_INPUT_BYTES:
            return None
    except UnicodeEncodeError:
        return None
    if replacement == original:
        return None
    # Two comparisons read a masked answer, and either one supplies the service text. The first takes one
    # record for each reported value with every copy replaced. Only it reads a match that begins with "{".
    if _every_copy_replaced(original, replacement, masks):
        return replacement
    # The service reports one record for each occurrence it detected and masks only those occurrences.
    # Every record needs a part of the original of its own, so records longer in total cannot be met.
    if sum(len(match) for _, match in masks) > len(original):
        return None
    owed = Counter(masks)
    values = [match for _, match in owed]
    for index, value in enumerate(values):
        if any(value in other or other in value for other in values[index + 1 :]):
            return None
    # Walk both texts from the start. Equal characters are stepped over. Anywhere else the service text
    # must hold the placeholder of a reported match's own category where the original holds that match.
    # A match that begins with "{" is stepped over as equal text, so it is never read as replaced.
    i = j = 0
    while j < len(replacement):
        if i < len(original) and original[i] == replacement[j]:
            i += 1
            j += 1
            continue
        for category, match in owed:
            token = "{" + category + "}"
            if replacement.startswith(token, j) and original.startswith(match, i):
                owed[category, match] -= 1
                i += len(match)
                j += len(token)
                break
        else:
            return None
    # Nothing of the original may be left over, and every record needs a replacement of its own.
    if i != len(original) or any(left > 0 for left in owed.values()):
        return None
    return replacement


def _every_copy_replaced(original, replacement, masks):
    # One record for each reported value, each value in the original, and no value inside another. The
    # service text must be the original with every copy of every value replaced, the longest value first.
    if len(set(masks)) != len(masks):
        return False
    for i, (_, match) in enumerate(masks):
        if match not in original or any(match in other or other in match for _, other in masks[i + 1 :]):
            return False
    expected = original
    for category, match in sorted(masks, key=lambda item: len(item[1]), reverse=True):
        expected = expected.replace(match, "{" + category + "}")
    return replacement == expected
