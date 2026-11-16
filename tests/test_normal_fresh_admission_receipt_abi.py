"""Actual closed receipt DATA exercises the cold helper; no native/model calls.

The response wrapper supplies only the helper's existing admitted DATA envelope.
Fresh receipts and ephemeral HMAC keys come from the actual checker and strict
neutral ApplyGuardrail port. This is ABI coverage, not lifecycle/admission proof.
"""

from __future__ import annotations

import copy

import pytest
from botocore.exceptions import NoCredentialsError
from test_input_admission import CYCLE, TEXT, RecordingGuardClient, none_response, private_checker
from test_normal_sft_cold_continuation import fresh_admission_evidence


def response(records):
    return {
        "data": {
            "input_admission": {
                "schema_version": 1,
                "data_only": True,
                "status": "admitted",
                "records": records,
                "native_admission": False,
                "quality_success": False,
                "blocked_component_attribution": "UNESTABLISHED",
            }
        }
    }


def allowed(boundaries=("OWNER_PROMPT",), cycle=CYCLE):
    client = RecordingGuardClient(none_response())
    checker = private_checker(client)
    records = []
    for boundary in boundaries:
        decision = checker.check(TEXT, boundary=boundary, opaque_cycle=cycle)
        assert decision.outcome == "ALLOW" and decision.forward_text is TEXT
        assert decision.receipt.blocking_categories == ()
        record = decision.receipt.to_dict()
        assert "blocking_categories" not in record
        records.append(record)
    assert len(client.calls) == len(boundaries)
    assert all(call["source"] == "INPUT" and call["outputScope"] == "FULL" for call in client.calls)
    return response(records)


def test_actual_receipt_optional_omission_retains_fresh_checks_guards_and_hmacs():
    warm = allowed(("OWNER_PROMPT", "TOOL_RESULT"))
    cold = allowed(("HISTORY_TEXT", "TOOL_RESULT"), cycle="1" * 32)
    warm_original, cold_original = copy.deepcopy(warm), copy.deepcopy(cold)
    warm_evidence = fresh_admission_evidence(warm, required_boundaries={"OWNER_PROMPT", "TOOL_RESULT"})
    cold_evidence = fresh_admission_evidence(cold, required_boundaries={"HISTORY_TEXT", "TOOL_RESULT"})
    assert warm_evidence[3] == cold_evidence[3]
    for first, replacement in zip(warm_evidence[:3], cold_evidence[:3], strict=True):
        assert first.isdisjoint(replacement)
    assert warm == warm_original and cold == cold_original


def test_required_receipt_field_is_not_defaulted():
    value = allowed()
    del value["data"]["input_admission"]["records"][0]["check_id"]
    with pytest.raises(KeyError, match="check_id"):
        fresh_admission_evidence(value, required_boundaries={"OWNER_PROMPT"})


def test_extra_receipt_field_cannot_disappear_during_closed_roundtrip():
    value = allowed()
    value["data"]["input_admission"]["records"][0]["future_optional_source_field"] = None
    with pytest.raises(AssertionError, match="closed Source ABI"):
        fresh_admission_evidence(value, required_boundaries={"OWNER_PROMPT"})


def test_actual_allow_receipt_requires_nonnull_hmac():
    value = allowed()
    value["data"]["input_admission"]["records"][0]["keyed_digest"] = None
    with pytest.raises((AssertionError, ValueError)):
        fresh_admission_evidence(value, required_boundaries={"OWNER_PROMPT"})


@pytest.mark.parametrize(
    "blocking", [[], ["CONTENT:PROMPT_ATTACK"]], ids=["noncanonical_empty", "nonblocking_attribution"]
)
def test_present_optional_blocker_data_is_not_ignored(blocking):
    value = allowed()
    value["data"]["input_admission"]["records"][0]["blocking_categories"] = blocking
    with pytest.raises((AssertionError, ValueError)):
        fresh_admission_evidence(value, required_boundaries={"OWNER_PROMPT"})


def test_actual_unknown_receipt_cannot_satisfy_admitted_evidence():
    client = RecordingGuardClient(NoCredentialsError())
    decision = private_checker(client).check(TEXT, boundary="OWNER_PROMPT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None and len(client.calls) == 1
    record = decision.receipt.to_dict()
    assert "blocking_categories" not in record
    with pytest.raises((AssertionError, ValueError)):
        fresh_admission_evidence(response([record]), required_boundaries={"OWNER_PROMPT"})
