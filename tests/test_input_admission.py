"""Private INPUT admission; strict neutral native ports, no provider calls.

Before the new primitive exists, four frozen contracts expose the actual legacy
OUTPUT monitor's direction and fail-open behavior. The exact missing-module
bridge never qualifies a green: all four assertions fail on that implementation.
Each future checker key is generated for this test invocation and kept in memory.
"""

from __future__ import annotations

import importlib
import json
import re
import secrets
from dataclasses import FrozenInstanceError, dataclass, field
from typing import Any

import pytest
from botocore.session import get_session
from botocore.validate import validate_parameters

from amlas.backends.adapters.governance import BedrockGuardrailBackend

MODULE = "amlas.tui.chat.input_admission"
GUARD = "fixtureguard"
CYCLE = "0" * 32
TEXT = "An ordinary public source summary."
CONTENT_TYPES = ("SEXUAL", "VIOLENCE", "HATE", "INSULTS", "MISCONDUCT", "PROMPT_ATTACK")


def none_response() -> dict[str, Any]:
    """Shape follows Root's native FULL/NONE control; empty outputs are valid."""
    return {
        "action": "NONE",
        "actionReason": "No action.",
        "assessments": [
            {
                "contentPolicy": {
                    "filters": [
                        {
                            "type": category,
                            "action": "NONE",
                            "confidence": "NONE",
                            "filterStrength": "HIGH",
                            "detected": False,
                        }
                        for category in CONTENT_TYPES
                    ]
                },
                "appliedGuardrailDetails": {"guardrailId": GUARD, "guardrailVersion": "DRAFT"},
            }
        ],
        "outputs": [],
        "guardrailCoverage": {"textCharacters": {"guarded": len(TEXT), "total": len(TEXT)}},
        "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
    }


class RecordingGuardClient:
    """Validate the installed SDK ApplyGuardrail input without creating a client.

    Primary contract: docs.aws.amazon.com/bedrock/latest/APIReference/
    API_runtime_ApplyGuardrail.html. No native API, model or network is called.
    """

    def __init__(self, response):
        self.response = response
        self.calls = []
        self.input_shape = (
            get_session().get_service_model("bedrock-runtime").operation_model("ApplyGuardrail").input_shape
        )

    def apply_guardrail(self, **kwargs):
        validate_parameters(kwargs, self.input_shape)
        self.calls.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


@dataclass(frozen=True)
class LegacyDecision:
    outcome: str
    forward_text: str | None = field(repr=False)


class LegacyContractBridge:
    """Baseline-only bridge; no replacement implementation or guard selection."""

    def __init__(self, client):
        self.monitor = BedrockGuardrailBackend(region="us-west-2", guardrail_id=GUARD, bedrock_client=client)

    def check(self, text, *, boundary, opaque_cycle):
        del boundary, opaque_cycle
        result = self.monitor.check(text=text)
        allowed = result["passed"] is True
        return LegacyDecision("ALLOW" if allowed else "BLOCKED", text if allowed else None)


def checker(client):
    try:
        source = importlib.import_module(MODULE)
    except ModuleNotFoundError as error:
        if error.name != MODULE:
            raise
        return LegacyContractBridge(client)
    config = source.InputAdmissionConfig(GUARD, "DRAFT", "us-west-2")
    return source.InputAdmissionChecker(config, client=client, hmac_key=secrets.token_bytes(32))


def test_input_direction_and_full_scope_are_mandatory():
    client = RecordingGuardClient(none_response())
    decision = checker(client).check(TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert len(client.calls) == 1
    assert client.calls[0]["source"] == "INPUT"
    assert client.calls[0]["outputScope"] == "FULL"
    assert client.calls[0]["guardrailIdentifier"] == GUARD
    assert client.calls[0]["guardrailVersion"] == "DRAFT"
    assert client.calls[0]["content"] == [{"text": {"text": TEXT}}]
    assert decision.outcome == "ALLOW"
    assert decision.forward_text is TEXT


def test_missing_native_action_cannot_admit_original_text():
    response = none_response()
    del response["action"]
    client = RecordingGuardClient(response)
    decision = checker(client).check(TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN"
    assert decision.forward_text is None
    assert len(client.calls) == 1


def test_malformed_assessment_cannot_admit_original_text():
    response = none_response()
    response["assessments"] = {"ordinary": "invalid assessment collection"}
    client = RecordingGuardClient(response)
    decision = checker(client).check(TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN"
    assert decision.forward_text is None
    assert len(client.calls) == 1


def test_none_action_with_blocked_assessment_cannot_admit_original_text():
    response = none_response()
    record = response["assessments"][0]["contentPolicy"]["filters"][-1]
    record.update(action="BLOCKED", confidence="HIGH", detected=True)
    client = RecordingGuardClient(response)
    decision = checker(client).check(TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome in {"BLOCKED", "UNKNOWN"}
    assert decision.forward_text is None
    assert len(client.calls) == 1


def private_checker(client, key=None):
    source = importlib.import_module(MODULE)
    config = source.InputAdmissionConfig(GUARD, "DRAFT", "us-west-2")
    return source.InputAdmissionChecker(
        config, client=client, hmac_key=secrets.token_bytes(32) if key is None else key
    )


def mask_response(text="Hello Jordan."):
    response = none_response()
    response["action"] = "GUARDRAIL_INTERVENED"
    response["guardrailCoverage"]["textCharacters"] = {"guarded": len(text), "total": len(text)}
    response["assessments"][0]["sensitiveInformationPolicy"] = {
        "piiEntities": [{"type": "NAME", "action": "ANONYMIZED", "match": "Jordan"}],
        "regexes": [],
    }
    response["outputs"] = [{"text": "Hello {NAME}."}]
    return response


def test_full_native_none_keeps_original_and_only_content_free_provenance():
    key = secrets.token_bytes(32)
    client = RecordingGuardClient(none_response())
    admission = private_checker(client, key)
    decision = admission.check(TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "ALLOW" and decision.forward_text is TEXT
    data = decision.receipt.to_dict()
    assert data["source"] == "INPUT" and data["boundary"] == "TOOL_RESULT"
    assert data["guardrail_id"] == GUARD and data["guardrail_version"] == "DRAFT"
    assert data["region"] == "us-west-2" and data["opaque_cycle"] == CYCLE
    assert data["byte_count"] == len(TEXT.encode())
    assert re.fullmatch(r"[a-f0-9]{32}", data["check_id"])
    assert re.fullmatch(r"[a-f0-9]{64}", data["keyed_digest"])
    assert data["data_only"] is True
    assert data["native_admission"] is data["quality_success"] is False
    assert TEXT not in json.dumps(data) and TEXT not in repr(decision)
    assert key.hex() not in json.dumps(data) and repr(key) not in repr(admission)
    with pytest.raises(FrozenInstanceError):
        decision.outcome = "UNKNOWN"
    with pytest.raises(FrozenInstanceError):
        admission.config = None


def test_blocked_service_message_cannot_be_forwarded_or_enter_receipt():
    response = none_response()
    response["action"] = "GUARDRAIL_INTERVENED"
    response["assessments"][0]["contentPolicy"]["filters"][-1].update(
        action="BLOCKED", detected=True, confidence="LOW"
    )
    response["outputs"] = [{"text": "Ordinary blocked service notice."}]
    decision = private_checker(RecordingGuardClient(response)).check(
        TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE
    )
    assert decision.outcome == "BLOCKED" and decision.forward_text is None
    assert "CONTENT:PROMPT_ATTACK" in decision.receipt.categories
    assert "Ordinary blocked service notice." not in json.dumps(decision.receipt.to_dict())
    assert TEXT not in repr(decision)


def test_valid_service_only_pii_mask_is_the_only_replacement_forwarded():
    original = "Hello Jordan."
    response = mask_response(original)
    client = RecordingGuardClient(response)
    decision = private_checker(client).check(original, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "ANONYMIZED"
    assert decision.forward_text is response["outputs"][0]["text"]
    assert decision.forward_text == "Hello {NAME}."
    data = decision.receipt.to_dict()
    assert data["byte_count"] == len(original.encode()) and "PII:NAME" in data["categories"]
    assert "Jordan" not in json.dumps(data) and "Jordan" not in repr(decision)
    assert client.calls[0]["content"] == [{"text": {"text": original}}]


@pytest.mark.parametrize(
    "change",
    [
        "missing_output",
        "rewritten_output",
        "multiple_outputs",
        "missing_match",
        "unknown_pii",
        "contradictory_action",
    ],
)
def test_invalid_masking_never_supplies_any_body(change):
    original = "Hello Jordan."
    response = mask_response(original)
    if change == "missing_output":
        response["outputs"] = []
    elif change == "rewritten_output":
        response["outputs"] = [{"text": "An unrelated service sentence."}]
    elif change == "multiple_outputs":
        response["outputs"] *= 2
    elif change == "missing_match":
        del response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0]["match"]
    elif change == "unknown_pii":
        response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0]["type"] = "UNKNOWN"
    else:
        response["action"] = "NONE"
    decision = private_checker(RecordingGuardClient(response)).check(
        original, boundary="TOOL_RESULT", opaque_cycle=CYCLE
    )
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert original not in json.dumps(decision.receipt.to_dict())


@pytest.mark.parametrize(
    "change",
    [
        "unknown_action",
        "nonprimitive_action",
        "bool_http",
        "retried",
        "empty_assessments",
        "too_many_records",
        "foreign_policy",
        "unknown_category",
        "nonbool_detected",
        "null_detected",
        "malformed_outputs",
        "none_with_output",
        "uncovered",
    ],
)
def test_malformed_or_unqualified_native_response_is_unknown_without_body(change):
    response = none_response()
    if change == "unknown_action":
        response["action"] = "UNKNOWN"
    elif change == "nonprimitive_action":
        response["action"] = True
    elif change == "bool_http":
        response["ResponseMetadata"]["HTTPStatusCode"] = True
    elif change == "retried":
        response["ResponseMetadata"]["RetryAttempts"] = 1
    elif change == "empty_assessments":
        response["assessments"] = []
    elif change == "too_many_records":
        response["assessments"][0]["contentPolicy"]["filters"] *= 22
    elif change == "foreign_policy":
        response["assessments"][0]["appliedGuardrailDetails"]["guardrailId"] = "otherguard"
    elif change == "unknown_category":
        response["assessments"][0]["contentPolicy"]["filters"][0]["type"] = "UNKNOWN"
    elif change in {"nonbool_detected", "null_detected"}:
        response["assessments"][0]["contentPolicy"]["filters"][0]["detected"] = (
            0 if change == "nonbool_detected" else None
        )
    elif change == "malformed_outputs":
        response["outputs"] = {"text": "Ordinary invalid container."}
    elif change == "none_with_output":
        response["outputs"] = [{"text": "An unrelated service sentence."}]
    else:
        response["guardrailCoverage"]["textCharacters"]["guarded"] -= 1
    decision = private_checker(RecordingGuardClient(response)).check(
        TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE
    )
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None


def test_transport_error_exposes_only_closed_unknown_reason():
    client = RecordingGuardClient(TimeoutError("Ordinary fixture transport detail."))
    decision = private_checker(client).check(TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert decision.receipt.reason == "TRANSPORT_UNKNOWN" and len(client.calls) == 1
    assert "Ordinary fixture transport detail." not in json.dumps(decision.receipt.to_dict())


def test_process_control_propagates_without_retry():
    client = RecordingGuardClient(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        private_checker(client).check(TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "text",
    ["", None, "é" * 32769, "x" * 65537, "\ud800"],
    ids=["empty", "nontext", "utf8_overcap", "character_overcap", "invalid_unicode"],
)
def test_invalid_or_overcap_input_is_not_truncated_split_or_sent(text):
    client = RecordingGuardClient(none_response())
    decision = private_checker(client).check(text, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert client.calls == []


def test_digest_is_keyed_and_binds_cycle_boundary_and_policy():
    key = secrets.token_bytes(32)

    def digest(*, cycle=CYCLE, boundary="TOOL_RESULT", selected_key=key):
        return (
            private_checker(RecordingGuardClient(none_response()), selected_key)
            .check(TEXT, boundary=boundary, opaque_cycle=cycle)
            .receipt.keyed_digest
        )

    expected = digest()
    assert digest() == expected
    assert digest(cycle="1" * 32) != expected
    assert digest(boundary="GENERATED_CONTEXT") != expected
    assert digest(selected_key=secrets.token_bytes(32)) != expected


def test_unknown_boundary_or_nonopaque_cycle_denies_before_native_call():
    client = RecordingGuardClient(none_response())
    admission = private_checker(client)
    with pytest.raises(ValueError):
        admission.check(TEXT, boundary="UNKNOWN", opaque_cycle=CYCLE)
    with pytest.raises(ValueError):
        admission.check(TEXT, boundary="TOOL_RESULT", opaque_cycle="ordinary-cycle")
    assert client.calls == []


def test_valid_pii_mask_with_contradictory_none_detection_is_not_forwardable():
    original = "Hello Jordan."
    response = mask_response(original)
    record = response["assessments"][0]["contentPolicy"]["filters"][0]
    record.update(action="NONE", detected=True, confidence="HIGH")
    client = RecordingGuardClient(response)
    decision = private_checker(client).check(original, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN"
    assert decision.forward_text is None
    assert len(client.calls) == 1
    assert client.calls[0]["source"] == "INPUT" and client.calls[0]["outputScope"] == "FULL"
    assert original not in json.dumps(decision.receipt.to_dict())
    assert "Hello {NAME}." not in json.dumps(decision.receipt.to_dict())


def test_character_limit_receipt_measures_only_bounded_utf8_prefix_without_body():
    source = importlib.import_module(MODULE)
    text = "ordinary metadata " * 5000
    client = RecordingGuardClient(none_response())
    decision = private_checker(client).check(text, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert client.calls == [] and source.MAX_INPUT_BYTES == 65536
    record = decision.receipt.to_dict()
    assert record["tool_name"] == "UNKNOWN"
    assert record["limit_bytes"] == 65536
    assert record["character_count"] == len(text)
    assert record["byte_measurement"] == "LOWER_BOUND"
    assert record["byte_count"] is None
    assert record["byte_count_lower_bound"] == len(text[:65537].encode("utf-8"))
    assert record["keyed_digest"] is None and text not in json.dumps(record)


def test_utf8_byte_limit_receipt_retains_exact_measurement_without_body():
    text = "é" * 35000
    client = RecordingGuardClient(none_response())
    decision = private_checker(client).check(text, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert client.calls == []
    record = decision.receipt.to_dict()
    assert record["tool_name"] == "UNKNOWN"
    assert record["limit_bytes"] == 65536
    assert record["character_count"] == len(text)
    assert record["byte_measurement"] == "EXACT"
    assert record["byte_count"] == 70000 and record["byte_count_lower_bound"] is None
    assert record["keyed_digest"] is None and text not in json.dumps(record)
