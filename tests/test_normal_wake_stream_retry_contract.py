"""Declare Lambda stream retry semantics without native state or SDK calls.

A DynamoDB event-source mapping retries failed function invocations; an HTTP-style
status dictionary alone is not a function failure. Whole-batch failure is the
chosen contract here; no partial-batch response or mapping setting is introduced.

https://docs.aws.amazon.com/lambda/latest/dg/with-ddb.html
https://docs.aws.amazon.com/lambda/latest/dg/services-ddb-batchfailurereporting.html
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from test_campaign_callback import QualifiedRoute
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.async_callback import lambda_handler as callback_source

FAILURE = "normal readiness delivery unresolved"


@contextmanager
def _declared_stream_boundary(monkeypatch, result):
    event = {"Records": [{"declared_input": "fresh-retry-boundary"}]}
    calls = []
    sdk_calls = []

    def unexpected_sdk(*_args, **_kwargs):
        sdk_calls.append("unexpected")
        raise AssertionError("SDK creation is outside this declared boundary")

    def declared_stream(received):
        assert received is event
        calls.append(received)
        assert len(calls) == 1 and sdk_calls == []
        return result

    monkeypatch.setenv("AMLAS_CAMPAIGN_CALLBACK_MODE", "1")
    monkeypatch.setattr(callback_source, "_normal_stream_handler", declared_stream)
    monkeypatch.setattr(callback_source.boto3, "client", unexpected_sdk)
    monkeypatch.setattr(callback_source.boto3, "Session", unexpected_sdk)
    monkeypatch.setattr(callback_source.boto3.session, "Session", unexpected_sdk)
    # This unit oracle covers result translation only, not native authentication.
    # Unscoped production bootstrap and input bounds have separate regressions.
    with callback_source.campaign_callback_scope(QualifiedRoute(callback_source), clock=lambda: 1000.0):
        yield event, calls, sdk_calls


@pytest.mark.parametrize(
    ("code", "disposition"),
    [
        (403, "qualification_unavailable"),
        (409, "readiness_refused"),
        (503, "readiness_unavailable"),
    ],
)
def test_stream_nondelivery_is_fixed_function_failure(monkeypatch, code, disposition):
    result = {"statusCode": code, "delivery_status": disposition}
    with _declared_stream_boundary(monkeypatch, result) as (event, calls, sdk_calls):
        with pytest.raises(RuntimeError, match=r"^normal readiness delivery unresolved$") as raised:
            callback_source.handler(event)

        assert str(raised.value) == FAILURE
        assert calls == [event] and sdk_calls == []


@pytest.mark.parametrize("disposition", ["delivered", "ignored"])
def test_stream_success_returns_original_result(monkeypatch, disposition):
    result = {"statusCode": 200, "delivery_status": disposition}
    with _declared_stream_boundary(monkeypatch, result) as (event, calls, sdk_calls):
        assert callback_source.handler(event) is result
        assert calls == [event] and sdk_calls == []


@pytest.mark.parametrize(
    "result",
    [None, {"statusCode": 200.0, "delivery_status": "delivered"}],
    ids=["absent", "float_status"],
)
def test_stream_malformed_result_is_fixed_function_failure(monkeypatch, result):
    with _declared_stream_boundary(monkeypatch, result) as (event, calls, sdk_calls):
        with pytest.raises(RuntimeError, match=r"^normal readiness delivery unresolved$") as raised:
            callback_source.handler(event)

        assert str(raised.value) == FAILURE
        assert calls == [event] and sdk_calls == []
