"""INPUT admission of text that holds a character outside the Basic Multilingual Plane.

The native service reports ``guardrailCoverage.textCharacters`` in UTF-16 code units, so a character
outside the Basic Multilingual Plane (an emoji, for example) counts as two. Recorded with ``outputScope``
FULL on the INPUT side: an 84-character sentence with one such character gave guarded 85 and total 85.
``recorded_none_response`` keeps the keys and counts of that recorded answer; the guard identity and its
ARN are fixture values and every sentence is the test's own. No native API, model or network is called.

The checker rule for ordinary text is pinned in ``test_input_admission.py``; these tests pin the unit of
the coverage count.
"""

from __future__ import annotations

import copy
import importlib
import json
import secrets
from typing import Any

import pytest
from botocore.session import get_session
from botocore.validate import validate_parameters

MODULE = "amlas.tui.chat.input_admission"
GUARD = "fixtureguard"
CYCLE = "0" * 32
CONTENT_TYPES = ("SEXUAL", "VIOLENCE", "HATE", "INSULTS", "PROMPT_ATTACK", "MISCONDUCT")
PII_TYPES = (
    "EMAIL",
    "PHONE",
    "NAME",
    "US_SOCIAL_SECURITY_NUMBER",
    "CREDIT_DEBIT_CARD_NUMBER",
    "US_BANK_ACCOUNT_NUMBER",
    "IP_ADDRESS",
)
SENTENCE = "The weather report for today says light rain in the afternoon and a clear evening."
SUPPLEMENTARY = "\U0001f419"  # one character, two UTF-16 code units, four UTF-8 bytes
# 84 characters, 85 UTF-16 code units, 87 UTF-8 bytes.
TEXT = SENTENCE + " " + SUPPLEMENTARY


def utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def recorded_none_response(text: str = TEXT) -> dict[str, Any]:
    """Keys and values of the recorded native FULL/NONE answer; the counts are UTF-16 code units."""
    coverage = {"textCharacters": {"guarded": utf16_units(text), "total": utf16_units(text)}}
    usage = {
        "automatedReasoningPolicies": 0,
        "automatedReasoningPolicyUnits": 0,
        "contentPolicyImageUnits": 0,
        "contentPolicyUnits": 1,
        "contextualGroundingPolicyUnits": 0,
        "sensitiveInformationPolicyFreeUnits": 0,
        "sensitiveInformationPolicyUnits": 1,
        "topicPolicyUnits": 0,
        "wordPolicyUnits": 0,
    }
    return {
        "action": "NONE",
        "actionReason": "No action.",
        "assessments": [
            {
                "appliedGuardrailDetails": {
                    "guardrailArn": "arn:aws:bedrock:us-west-2:fixture-account:guardrail/" + GUARD,
                    "guardrailId": GUARD,
                    "guardrailOrigin": ["REQUEST"],
                    "guardrailOwnership": "SELF",
                    "guardrailVersion": "DRAFT",
                },
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
                "invocationMetrics": {
                    "guardrailCoverage": copy.deepcopy(coverage),
                    "guardrailProcessingLatency": 98,
                    "usage": dict(usage),
                },
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {"type": entity, "action": "NONE", "detected": False, "match": ""}
                        for entity in PII_TYPES
                    ],
                    "regexes": [],
                },
            }
        ],
        "guardrailCoverage": coverage,
        "outputs": [],
        "usage": usage,
        "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
    }


def recorded_mask_response(text: str, name: str, replacement: str) -> dict[str, Any]:
    """The recorded masked answer's shape: one detected NAME entry and one service output."""
    response = recorded_none_response(text)
    response["action"] = "GUARDRAIL_INTERVENED"
    response["actionReason"] = "Guardrail masked."
    for entry in response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]:
        if entry["type"] == "NAME":
            entry.update(action="ANONYMIZED", detected=True, match=name)
    response["outputs"] = [{"text": replacement}]
    return response


class RecordingGuardClient:
    """Validate the installed SDK ApplyGuardrail input without creating a client."""

    def __init__(self, response):
        self.response = response
        self.calls = []
        self.input_shape = (
            get_session().get_service_model("bedrock-runtime").operation_model("ApplyGuardrail").input_shape
        )

    def apply_guardrail(self, **kwargs):
        validate_parameters(kwargs, self.input_shape)
        self.calls.append(kwargs)
        return self.response


def private_checker(client):
    source = importlib.import_module(MODULE)
    config = source.InputAdmissionConfig(GUARD, "DRAFT", "us-west-2")
    return source.InputAdmissionChecker(config, client=client, hmac_key=secrets.token_bytes(32))


def check(text, response):
    client = RecordingGuardClient(response)
    return client, private_checker(client).check(text, boundary="TOOL_RESULT", opaque_cycle=CYCLE)


def test_recorded_fixture_counts_utf16_code_units():
    assert (len(TEXT), utf16_units(TEXT), len(TEXT.encode())) == (84, 85, 87)
    assert recorded_none_response()["guardrailCoverage"]["textCharacters"] == {"guarded": 85, "total": 85}


def test_native_none_for_text_with_a_supplementary_character_keeps_the_original():
    client, decision = check(TEXT, recorded_none_response())
    assert len(client.calls) == 1
    assert client.calls[0]["content"] == [{"text": {"text": TEXT}}]
    data = decision.receipt.to_dict()
    assert (decision.outcome, data["reason"]) == ("ALLOW", "NATIVE_NONE")
    assert decision.forward_text is TEXT
    # The receipt keeps its code point and UTF-8 measurements.
    assert (data["character_count"], data["byte_count"]) == (84, 87)
    assert len(data["categories"]) == 13
    assert TEXT not in json.dumps(data) and TEXT not in repr(decision)


@pytest.mark.parametrize("count", [2, 5])
def test_native_none_for_text_with_several_supplementary_characters_keeps_the_original(count):
    text = SENTENCE + (" " + SUPPLEMENTARY) * count
    assert utf16_units(text) == len(text) + count
    _, decision = check(text, recorded_none_response(text))
    assert (decision.outcome, decision.receipt.reason) == ("ALLOW", "NATIVE_NONE")
    assert decision.forward_text is text
    assert decision.receipt.character_count == len(text)


def test_service_mask_for_text_with_a_supplementary_character_is_forwarded():
    original = "Hello Jordan " + SUPPLEMENTARY + "."
    response = recorded_mask_response(original, "Jordan", "Hello {NAME} " + SUPPLEMENTARY + ".")
    _, decision = check(original, response)
    data = decision.receipt.to_dict()
    assert (decision.outcome, data["reason"]) == ("ANONYMIZED", "NATIVE_MASKED")
    assert decision.forward_text is response["outputs"][0]["text"]
    assert "PII:NAME" in data["categories"]
    assert "Jordan" not in json.dumps(data) and "Jordan" not in repr(decision)


@pytest.mark.parametrize("change", ["uncovered", "total_above_text", "total_below_text"])
def test_coverage_that_is_not_the_whole_text_stays_unknown(change):
    response = recorded_none_response()
    counts = response["guardrailCoverage"]["textCharacters"]
    if change == "uncovered":
        counts["guarded"] -= 1
    elif change == "total_above_text":
        counts["guarded"] = counts["total"] = utf16_units(TEXT) + 1
    else:
        counts["guarded"] = counts["total"] = len(TEXT) - 1
    client, decision = check(TEXT, response)
    assert len(client.calls) == 1
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert decision.receipt.reason == "NATIVE_INVALID"


def test_coverage_one_code_unit_short_of_the_text_stays_unknown():
    """A total equal to the number of characters leaves one UTF-16 code unit of this text uncovered."""
    response = recorded_none_response()
    response["guardrailCoverage"]["textCharacters"] = {"guarded": len(TEXT), "total": len(TEXT)}
    assert len(TEXT) == utf16_units(TEXT) - 1
    _, decision = check(TEXT, response)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert decision.receipt.reason == "NATIVE_INVALID"


def test_masked_answer_with_short_coverage_supplies_no_body():
    original = "Hello Jordan " + SUPPLEMENTARY + "."
    response = recorded_mask_response(original, "Jordan", "Hello {NAME} " + SUPPLEMENTARY + ".")
    response["guardrailCoverage"]["textCharacters"]["guarded"] -= 1
    _, decision = check(original, response)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert decision.receipt.reason == "NATIVE_INVALID"


@pytest.mark.parametrize(
    "text",
    ["Café notes → naïve summary.", "東京の天気は晴れです", "Żółw, Ærø and 信 in one line."],
    ids=["accented", "cjk", "mixed"],
)
def test_text_inside_the_basic_plane_is_admitted_as_before(text):
    assert len(text) == utf16_units(text) and len(text.encode()) > len(text)
    response = recorded_none_response(text)
    assert response["guardrailCoverage"]["textCharacters"]["total"] == len(text)
    _, decision = check(text, response)
    assert (decision.outcome, decision.receipt.reason) == ("ALLOW", "NATIVE_NONE")
    assert decision.forward_text is text
    assert decision.receipt.character_count == len(text)
    assert decision.receipt.byte_count == len(text.encode())


@pytest.mark.parametrize(
    "text",
    [
        "ordinary neutral text \ud800 " + SUPPLEMENTARY,
        SUPPLEMENTARY + " ordinary neutral text \udc19",
        "ordinary neutral text \ud83d\udc19",
    ],
    ids=["lone_high_surrogate", "lone_low_surrogate", "two_surrogate_code_points"],
)
def test_text_that_cannot_be_encoded_stays_refused_without_a_native_call(text):
    """Surrogate code points are not a supplementary character; no coverage count is ever compared."""
    with pytest.raises(UnicodeEncodeError):
        text.encode("utf-8")
    client, decision = check(text, recorded_none_response(SENTENCE))
    assert client.calls == []
    data = decision.receipt.to_dict()
    assert (decision.outcome, data["reason"]) == ("UNKNOWN", "INVALID_INPUT")
    assert decision.forward_text is None
    assert data["character_count"] == len(text)
    assert (data["byte_measurement"], data["byte_count"], data["keyed_digest"]) == ("UNAVAILABLE", None, None)
