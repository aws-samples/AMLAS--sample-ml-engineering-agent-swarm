"""The stream record is acknowledged only by the runtime's positive closed event.

The runtime body used to be discarded and every HTTP 200 counted as delivered, so a
busy original, a failed restore and a cold start were all acknowledged. Only the
first complete closed event `resume_state` with state continuing or completed, bound
to this delivery's run, acknowledges; everything else is outcome_unknown (202).
The runtime is our own code behind IAM, so a body can only cause retries unless it
carries that event; a body that merely claims success outside the schema is refused.
"""

from __future__ import annotations

import json

import pytest
from test_campaign_callback import Body, resume_state_bytes
from test_normal_wake_routing import RUN_ID, _declared_route
from test_normal_wake_routing import selected_callback_env as selected_callback_env

from amlas.async_callback import lambda_handler as callback_source

OTHER_RUN = "2" * 32


def _deliver(body, *, clock=lambda: 1000.0):
    value, ddb, runtime, route = _declared_route()
    runtime.resume_state = None  # the default route answers continuing; here the body is under test
    runtime.body = body
    with callback_source.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=clock
    ):
        result = callback_source.handler(value)
    return result, runtime


def _event_bytes(**fields):
    event = {"event": "resume_state", "state": "continuing", "run_id": RUN_ID}
    event.update(fields)
    event = {name: value for name, value in event.items() if value is not None}
    return json.dumps(event, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("state", ["continuing", "completed"])
@pytest.mark.parametrize("sse", [True, False])
def test_positive_closed_state_for_this_run_is_delivered(state, sse):
    result, runtime = _deliver(Body(resume_state_bytes(state, RUN_ID, sse=sse)))
    assert result["statusCode"] == 200 and result["delivery_status"] == "delivered"
    assert len(runtime.calls) == 1 and runtime.body.closed_called
    assert result["lifecycle_verified"] is False and result["quality_success"] is False


@pytest.mark.parametrize("state", ["restoring", "original_active", "failed_transient", "failed"])
def test_every_other_state_is_outcome_unknown_so_the_stream_retries(state):
    extra = {"diagnostic": {"restore_stage": "run_prepare", "exception_class": "ValueError"}}
    for fields in ({}, extra if state.startswith("failed") else {}):
        result, runtime = _deliver(Body(resume_state_bytes(state, RUN_ID, **fields)))
        assert result["statusCode"] == 202 and result["delivery_status"] == "outcome_unknown"
        assert len(runtime.calls) == 1 and runtime.body.closed_called


MALFORMED = {
    "empty": b"",
    "old neutral bytes": b"neutral delivery bytes",
    "json array": b'["resume_state","continuing"]\n',
    "claim outside the schema": b'{"status":"continuing"}\n',
    "prompt_run_status claiming continuing": _event_bytes(event="prompt_run_status") + b"\n",
    "extra key": _event_bytes(extra=1) + b"\n",
    "missing run": _event_bytes(run_id=None) + b"\n",
    "missing state": _event_bytes(state=None) + b"\n",
    "other run": _event_bytes(run_id=OTHER_RUN) + b"\n",
    "unknown state": _event_bytes(state="started") + b"\n",
    "state not text": _event_bytes(state=["continuing"]) + b"\n",
    "state number": _event_bytes(state=1) + b"\n",
    "run not text": _event_bytes(run_id=1) + b"\n",
    "duplicate key": (
        b'{"event":"resume_state","state":"restoring","state":"continuing","run_id":"'
        + RUN_ID.encode()
        + b'"}\n'
    ),
    "nonfinite": b'{"event":"resume_state","state":"continuing","run_id":"'
    + RUN_ID.encode()
    + b'","n":NaN}\n',
    "not utf8": b"\xff\xfe" + _event_bytes() + b"\n",
    "data without json": b"data:\n\n",
    "sse event field first": b"event: resume_state\n" + b"data: " + _event_bytes() + b"\n\n",
    "unterminated line": b"data: " + _event_bytes(),
    "heartbeat first": (
        b'data: {"event":"heartbeat","n":1,"run_id":"' + RUN_ID.encode() + b'"}\n\n'
        b"data: " + _event_bytes() + b"\n\n"
    ),
    "restoring first then continuing": (
        b"data: " + _event_bytes(state="restoring") + b"\n\ndata: " + _event_bytes() + b"\n\n"
    ),
    "failed first then completed": (
        b"data: " + _event_bytes(state="failed") + b"\n\ndata: " + _event_bytes(state="completed") + b"\n\n"
    ),
    "oversized without newline": b"a" * (64 * 1024 + 1),
    "valid event after the bound": b"\n" * (64 * 1024) + b"data: " + _event_bytes() + b"\n\n",
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_a_body_that_does_not_match_the_closed_schema_is_never_acknowledged(name):
    result, runtime = _deliver(Body(MALFORMED[name]))
    assert result["statusCode"] == 202 and result["delivery_status"] == "outcome_unknown", name
    assert len(runtime.calls) == 1 and runtime.body.closed_called


def test_sse_comments_before_the_event_do_not_break_or_forge_the_gate():
    result, _ = _deliver(Body(b": keepalive\n\ndata: " + _event_bytes() + b"\n\n"))
    assert result["delivery_status"] == "delivered"
    result, _ = _deliver(Body(b": keepalive\n\ndata: " + _event_bytes(state="restoring") + b"\n\n"))
    assert result["delivery_status"] == "outcome_unknown"


class OpenStreamBody(Body):
    """A live stream that has delivered only its first event; a read past it would block."""

    def __init__(self, first):
        super().__init__(first)
        self.available = len(first)
        self.reads = []

    def read(self, amount=-1):
        assert 0 < amount <= 65536, "callback cannot unbounded-read response"
        self.reads.append(amount)
        remaining = self.available - self.tell()
        if remaining <= 0:
            raise AssertionError("read past the first event: the open stream would block until timeout")
        if amount > remaining:
            raise TimeoutError("a read larger than the delivered bytes blocks on a live stream")
        return super().read(amount)


def test_the_gate_stops_at_the_first_event_and_never_blocks_on_the_open_stream():
    body = OpenStreamBody(b"data: " + _event_bytes() + b"\n\n")
    result, _runtime = _deliver(body)
    assert result["delivery_status"] == "delivered", "must not wait for end of stream"
    assert body.closed_called and body.reads


def test_drain_deadline_before_the_event_is_never_acknowledged():
    now = [1000.0]

    class Late(Body):
        def read(self, amount=-1):
            result = super().read(amount)
            now[0] += 21
            return result

    result, runtime = _deliver(Late(b"data: " + _event_bytes() + b"\n\n"), clock=lambda: now[0])
    assert result["statusCode"] == 202 and result["delivery_status"] == "outcome_unknown"
    assert runtime.body.closed_called and len(runtime.calls) == 1


def test_runtime_error_and_close_failure_are_never_acknowledged():
    value, ddb, runtime, route = _declared_route()
    runtime.fail = True
    with callback_source.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        assert callback_source.handler(value)["delivery_status"] == "outcome_unknown"

    class CloseFailure(Body):
        def close(self):
            super().close()
            raise OSError("neutral close failure")

    result, _ = _deliver(CloseFailure(b"data: " + _event_bytes() + b"\n\n"))
    assert result["delivery_status"] == "outcome_unknown"


def test_deployed_default_qualifier_delivers_and_acknowledges(monkeypatch):
    monkeypatch.setenv("AGENTCORE_RUNTIME_QUALIFIER", "DEFAULT")
    result, runtime = _deliver(Body(resume_state_bytes("continuing", RUN_ID)))
    assert result["delivery_status"] == "delivered"
    assert runtime.calls[0]["qualifier"] == "DEFAULT"
    monkeypatch.setenv("AGENTCORE_RUNTIME_QUALIFIER", "DEFAULT")
    result, runtime = _deliver(Body(resume_state_bytes("restoring", RUN_ID)))
    assert result["delivery_status"] == "outcome_unknown" and runtime.calls[0]["qualifier"] == "DEFAULT"


def test_campaign_wake_body_is_still_discarded_untrusted_data():
    from test_campaign_callback import fixtures

    module, value, ddb, runtime, route = fixtures("Training")
    runtime.body = Body(b"neutral delivery bytes")
    with module.campaign_callback_scope(
        route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0
    ):
        result = module.handler(value)
    assert result["delivery_status"] == "delivered", "campaign wake keeps its existing discard contract"
