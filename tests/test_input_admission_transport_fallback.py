"""Optional diagnostics never replace the original closed admission hold."""

from __future__ import annotations

import json

from botocore.exceptions import ClientError
from test_input_admission import CYCLE, TEXT, RecordingGuardClient, private_checker


def test_unreadable_sdk_response_keeps_closed_hold_with_one_call():
    reads = []

    class UnreadableResponseError(ClientError):
        def __getattribute__(self, name):
            if name == "response":
                reads.append(name)
                raise RuntimeError("ordinary private diagnostic extraction detail")
            return super().__getattribute__(name)

    error = UnreadableResponseError(
        {"Error": {"Code": "ValidationException", "Message": "ordinary private service detail"}},
        "ApplyGuardrail",
    )
    client = RecordingGuardClient(error)
    decision = private_checker(client).check(TEXT, boundary="COMPLETION_DATA", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert decision.receipt.reason == "TRANSPORT_UNKNOWN" and len(client.calls) == 1
    assert reads == ["response"]
    data = decision.receipt.to_dict()
    assert data["transport_diagnostic"] == {
        "phase": "APPLY_GUARDRAIL",
        "error_class": "UNKNOWN",
        "native_code": "UNKNOWN",
        "http_status": None,
        "request_id": None,
        "retry_attempts": None,
    }
    assert "ordinary private" not in json.dumps(data)
