"""Closed SDK transport diagnostics; neutral fixtures, no native calls."""

from __future__ import annotations

import copy
import json

import pytest
from botocore.exceptions import ClientError, NoCredentialsError, ReadTimeoutError
from test_input_admission import CYCLE, TEXT, RecordingGuardClient, private_checker
from test_input_admission_wiring import ClientFixture, configure_guard

REQUEST_ID = "abcdefab-cdef-4abc-8def-abcdefabcdef"
DETAIL = "ordinary private provider diagnostic detail"


def client_error(code="ThrottlingException", status=429, request_id=REQUEST_ID, retries=0):
    # Official ApplyGuardrail errors and standard botocore ClientError response shape.
    return ClientError(
        {
            "Error": {"Code": code, "Message": DETAIL},
            "ResponseMetadata": {
                "HTTPStatusCode": status,
                "RequestId": request_id,
                "RetryAttempts": retries,
                "HTTPHeaders": {"fixture": DETAIL},
            },
        },
        "ApplyGuardrail",
    )


@pytest.mark.parametrize(
    "error,expected",
    [
        (
            client_error(),
            {
                "error_class": "ClientError",
                "native_code": "ThrottlingException",
                "http_status": 429,
                "request_id": REQUEST_ID,
                "retry_attempts": 0,
            },
        ),
        (
            client_error("ExpiredTokenException", 403),
            {
                "error_class": "ClientError",
                "native_code": "ExpiredTokenException",
                "http_status": 403,
                "request_id": REQUEST_ID,
                "retry_attempts": 0,
            },
        ),
        (
            ReadTimeoutError(endpoint_url="https://ordinary.example", error=DETAIL),
            {
                "error_class": "ReadTimeoutError",
                "native_code": "UNKNOWN",
                "http_status": None,
                "request_id": None,
                "retry_attempts": None,
            },
        ),
        (
            NoCredentialsError(),
            {
                "error_class": "NoCredentialsError",
                "native_code": "UNKNOWN",
                "http_status": None,
                "request_id": None,
                "retry_attempts": None,
            },
        ),
    ],
)
def test_transport_metadata_is_closed_and_does_not_retry_or_admit(error, expected):
    client = RecordingGuardClient(error)
    decision = private_checker(client).check(TEXT, boundary="COMPLETION_DATA", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert decision.receipt.reason == "TRANSPORT_UNKNOWN" and len(client.calls) == 1
    data = decision.receipt.to_dict()
    assert data["transport_diagnostic"] == {"phase": "APPLY_GUARDRAIL", **expected}
    assert data["native_admission"] is data["quality_success"] is False
    assert DETAIL not in json.dumps(data) and TEXT not in json.dumps(data)
    data["transport_diagnostic"]["native_code"] = "ordinary mutation"
    assert decision.receipt.to_dict()["transport_diagnostic"]["native_code"] == expected["native_code"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("code", DETAIL),
        ("status", True),
        ("status", 999),
        ("request_id", DETAIL),
        ("retries", True),
        ("retries", 1000),
    ],
)
def test_unqualified_service_fields_are_not_reflected(field, value):
    values = dict(code="ThrottlingException", status=429, request_id=REQUEST_ID, retries=0)
    values[field] = value
    client = RecordingGuardClient(client_error(**values))
    decision = private_checker(client).check(TEXT, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and len(client.calls) == 1
    diagnostic = decision.receipt.to_dict()["transport_diagnostic"]
    output_key = {
        "code": "native_code",
        "status": "http_status",
        "request_id": "request_id",
        "retries": "retry_attempts",
    }[field]
    assert diagnostic[output_key] == ("UNKNOWN" if field == "code" else None)
    assert DETAIL not in json.dumps(decision.receipt.to_dict())


def test_unknown_exception_is_not_named_or_stringified():
    accesses = []

    class OrdinaryUnknownError(Exception):
        def __str__(self):
            accesses.append("str")
            return DETAIL

    client = RecordingGuardClient(OrdinaryUnknownError())
    decision = private_checker(client).check(TEXT, boundary="COMPLETION_DATA", opaque_cycle=CYCLE)
    assert accesses == [] and decision.outcome == "UNKNOWN" and len(client.calls) == 1
    assert decision.receipt.to_dict()["transport_diagnostic"] == {
        "phase": "APPLY_GUARDRAIL",
        "error_class": "UNKNOWN",
        "native_code": "UNKNOWN",
        "http_status": None,
        "request_id": None,
        "retry_attempts": None,
    }


def test_original_warm_completion_preserves_custody_on_transport_failure(monkeypatch, tmp_path):
    from test_runtime_warm_model_continuation import complete_fixture, original_work, step

    class CompletionTransport(ClientFixture):
        def __init__(self):
            super().__init__(deny=lambda text: False)
            self.failures = 0

        def apply_guardrail(self, **kwargs):
            if '"event_type":"original_model_acquisition_completed"' in kwargs["content"][0]["text"]["text"]:
                self.calls.append(kwargs)
                self.failures += 1
                raise client_error("ServiceUnavailableException", 503)
            return super().apply_guardrail(**kwargs)

    client = CompletionTransport()
    configure_guard(monkeypatch, client)
    work = original_work(monkeypatch)
    pending = work.backend._warm_pending
    before = copy.deepcopy(work.agent.messages)
    callback = work.agent.callback_handler
    consumed = set(work.backend._warm_consumed)
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert client.calls and client.failures == 0
    complete_fixture(monkeypatch, tmp_path, work)
    result = step(work)
    assert result.disposition == "refused" and client.failures == 1
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert work.agent.messages == before and work.agent.callback_handler is callback
    assert work.backend._warm_pending is pending and pending.record.handle is work.handle
    assert work.backend._warm_consumed == consumed
    assert result.turn.input_admission["status"] == "held"
    calls = len(client.calls)
    assert step(work).disposition == "refused" and len(client.calls) == calls
    assert result.turn.input_admission["records"][-1]["transport_diagnostic"] == {
        "phase": "APPLY_GUARDRAIL",
        "error_class": "ClientError",
        "native_code": "ServiceUnavailableException",
        "http_status": 503,
        "request_id": REQUEST_ID,
        "retry_attempts": 0,
    }
