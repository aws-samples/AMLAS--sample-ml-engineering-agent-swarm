"""Closed blocking categories from native FULL records; no policy or retry change.

Neutral native-shaped fixtures use existing SDK request validation and the actual
INPUT checker. Fixture text is ordinary; no provider/model/network is called.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
from botocore.exceptions import NoCredentialsError
from test_input_admission import CYCLE, TEXT, RecordingGuardClient, none_response, private_checker


def native_full(*, content=(), pii=()):
    response = none_response()
    response["action"] = "GUARDRAIL_INTERVENED"
    filters = response["assessments"][0]["contentPolicy"]["filters"]
    for entry in filters:
        if entry["type"] in content:
            entry.update(action="BLOCKED", confidence="HIGH", detected=True)
    if pii:
        response["assessments"][0]["sensitiveInformationPolicy"] = {
            "piiEntities": [
                {"type": "NAME", "action": "NONE", "detected": False},
                *[
                    {
                        "type": category,
                        "action": "BLOCKED",
                        "detected": True,
                        "match": "ordinary fixture match",
                    }
                    for category in pii
                ],
            ],
            "regexes": [],
        }
    return response


@pytest.mark.parametrize(
    "content,pii,expected",
    [
        (("PROMPT_ATTACK",), (), ("CONTENT:PROMPT_ATTACK",)),
        ((), ("EMAIL",), ("PII:EMAIL",)),
        (
            ("PROMPT_ATTACK",),
            ("EMAIL", "EMAIL"),
            ("CONTENT:PROMPT_ATTACK", "PII:EMAIL"),
        ),
    ],
    ids=["content", "pii", "mixed_deduplicated"],
)
def test_full_receipt_separates_actual_blockers_without_changing_hold(content, pii, expected):
    response = native_full(content=content, pii=pii)
    original = copy.deepcopy(response)
    client = RecordingGuardClient(response)
    decision = private_checker(client).check(TEXT, boundary="OWNER_PROMPT", opaque_cycle=CYCLE)
    assert decision.outcome == "BLOCKED" and decision.forward_text is None
    assert decision.receipt.reason == "NATIVE_BLOCKED" and len(client.calls) == 1
    all_categories = {
        "CONTENT:" + value["type"] for value in response["assessments"][0]["contentPolicy"]["filters"]
    }
    if pii:
        all_categories |= {"PII:NAME", "PII:EMAIL"}
    assert decision.receipt.categories == tuple(sorted(all_categories))
    assert hasattr(decision.receipt, "blocking_categories")
    assert decision.receipt.blocking_categories == expected
    data = decision.receipt.to_dict()
    assert data["categories"] == sorted(all_categories)
    assert data["blocking_categories"] == list(expected)
    assert set(data["blocking_categories"]) <= set(data["categories"])
    assert data["data_only"] is True
    assert data["native_admission"] is data["quality_success"] is False
    assert "ordinary fixture match" not in json.dumps(data)
    assert TEXT not in json.dumps(data)
    assert response == original


@pytest.mark.parametrize("kind", ["allowed", "unknown_transport"])
def test_nonblocking_receipt_keeps_legacy_serialization_and_no_block_attribution(kind):
    value = none_response() if kind == "allowed" else NoCredentialsError()
    client = RecordingGuardClient(value)
    decision = private_checker(client).check(TEXT, boundary="OWNER_PROMPT", opaque_cycle=CYCLE)
    assert decision.outcome == ("ALLOW" if kind == "allowed" else "UNKNOWN")
    assert len(client.calls) == 1
    assert getattr(decision.receipt, "blocking_categories", ()) == ()
    assert "blocking_categories" not in decision.receipt.to_dict()


@pytest.mark.parametrize(
    "bad",
    [
        ["CONTENT:PROMPT_ATTACK"],
        ("CONTENT:ordinary",),
        ("PII:ADDRESS",),
        ("CONTENT:PROMPT_ATTACK", "CONTENT:PROMPT_ATTACK"),
        ("PII:EMAIL", "CONTENT:PROMPT_ATTACK"),
    ],
    ids=["non_tuple", "unknown_enum", "unobserved_category", "duplicate", "unsorted"],
)
def test_blocking_category_diagnostic_refuses_unqualified_metadata(bad):
    client = RecordingGuardClient(native_full(content=("PROMPT_ATTACK",), pii=("EMAIL",)))
    decision = private_checker(client).check(TEXT, boundary="OWNER_PROMPT", opaque_cycle=CYCLE)
    assert decision.outcome == "BLOCKED" and decision.forward_text is None
    with pytest.raises(ValueError):
        replace(decision.receipt, blocking_categories=bad)
    assert len(client.calls) == 1


def test_legacy_blocked_receipt_remains_unattributed_without_optional_field():
    client = RecordingGuardClient(native_full(content=("PROMPT_ATTACK",)))
    decision = private_checker(client).check(TEXT, boundary="OWNER_PROMPT", opaque_cycle=CYCLE)
    receipt = decision.receipt
    legacy = type(receipt)(
        check_id=receipt.check_id,
        opaque_cycle=receipt.opaque_cycle,
        boundary=receipt.boundary,
        guardrail_id=receipt.guardrail_id,
        guardrail_version=receipt.guardrail_version,
        region=receipt.region,
        byte_count=receipt.byte_count,
        keyed_digest=receipt.keyed_digest,
        outcome=receipt.outcome,
        reason=receipt.reason,
        categories=receipt.categories,
        character_count=receipt.character_count,
        byte_measurement=receipt.byte_measurement,
        byte_count_lower_bound=receipt.byte_count_lower_bound,
    )
    assert legacy.outcome == "BLOCKED" and legacy.reason == "NATIVE_BLOCKED"
    assert legacy.categories == receipt.categories
    assert getattr(legacy, "blocking_categories", ()) == ()
    assert "blocking_categories" not in legacy.to_dict()
    assert len(client.calls) == 1


@pytest.mark.parametrize("kind", ["allowed", "unknown_transport"])
def test_nonblocking_decision_rejects_nonempty_blocking_diagnostic(kind):
    value = none_response() if kind == "allowed" else NoCredentialsError()
    client = RecordingGuardClient(value)
    decision = private_checker(client).check(TEXT, boundary="OWNER_PROMPT", opaque_cycle=CYCLE)
    assert decision.outcome == ("ALLOW" if kind == "allowed" else "UNKNOWN")
    with pytest.raises(ValueError):
        replace(decision.receipt, blocking_categories=("CONTENT:PROMPT_ATTACK",))
    assert len(client.calls) == 1


def test_malformed_native_blocker_cannot_publish_trusted_attribution():
    response = native_full(content=("PROMPT_ATTACK",))
    response["assessments"][0]["contentPolicy"]["filters"][-1]["type"] = "ordinary invalid category"
    original = copy.deepcopy(response)
    client = RecordingGuardClient(response)
    decision = private_checker(client).check(TEXT, boundary="OWNER_PROMPT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert decision.receipt.reason == "NATIVE_INVALID"
    assert getattr(decision.receipt, "blocking_categories", ()) == ()
    assert "blocking_categories" not in decision.receipt.to_dict()
    assert response == original and len(client.calls) == 1
