"""Owned pool clearing before a new INPUT assessment; no network or retry.

The pinned SDK close/reuse canary establishes reusable client pool semantics.
The existing strict GuardTransport validates the installed ApplyGuardrail shape;
legacy factory retirement remains separate from pool clearing.
"""

from __future__ import annotations

import inspect

import pytest
from botocore.exceptions import ConnectionClosedError
from test_input_admission_connection_lifecycle import ENDPOINT, GUARD, GuardTransport
from test_input_admission_wiring import configured_backend
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.tui.chat.input_admission import InputAdmissionChecker, InputAdmissionConfig
from amlas.tui.chat.tool_binding_backend import _InputAdmissionHeldError

CYCLE = "c" * 32
KEY = b"k" * 32


class ReusablePoolTransport(GuardTransport):
    """SDK close clears owned pools without invalidating the client object."""

    def __init__(self, *, fail=False, close_error=None):
        super().__init__("owned", [], fail=fail)
        self.close_calls = 0
        self.close_error = close_error

    def close(self):
        self.close_calls += 1
        self.events.append(("pool_clear", self.label))
        if self.close_error is not None:
            raise self.close_error
        self.idle = False


def owned_checker(client):
    if "close_before_assessment" not in inspect.signature(InputAdmissionChecker).parameters:
        raise AssertionError("missing-symbol: InputAdmissionChecker.close_before_assessment")
    return InputAdmissionChecker(
        InputAdmissionConfig(GUARD, "DRAFT", "us-west-2"),
        client=client,
        hmac_key=KEY,
        close_before_assessment=True,
    )


@pytest.mark.parametrize("fail_fresh", [False, True], ids=["idle-pool-cleared", "fresh-failure-no-replay"])
@pytest.mark.usefixtures("inert_runtime_setup")
def test_ordinary_tool_result_clears_owned_pool_without_replacing_checker(monkeypatch, fail_fresh):
    client = ReusablePoolTransport()
    backend, model, _agents = configured_backend(monkeypatch, client)
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", GUARD)
    assert backend._admit_input("ordinary owner request", "OWNER_PROMPT") == "ordinary owner request"
    checker = backend._input_checker
    config, key, cycle, policy = checker.config, checker._key, backend._input_cycle, backend.approval_policy
    graph = backend._warm_graph()
    calls, closes = len(client.calls), client.close_calls
    client.idle = True
    client.fail = fail_fresh
    returned = None
    held = False
    try:
        returned = backend._admit_input("ordinary tool facts", "TOOL_RESULT", tool_name="ordinary_tool")
    except _InputAdmissionHeldError:
        held = True
    assert client.close_calls == closes + 1, "owned TOOL_RESULT did not clear its pool before assessment"
    assert len(client.calls) == calls + 1
    assert client.events[-2:] == [("pool_clear", "owned"), ("apply", "owned")]
    assert backend._input_checker is checker and backend._input_guard_client is client
    assert checker.config is config and checker._key is key and backend._input_cycle == cycle
    assert backend.approval_policy is policy and backend._warm_graph() == graph and model.calls == 0
    receipt = backend._input_admission_data()["records"][-1]
    if fail_fresh:
        assert held and returned is None and backend._input_hold
        assert receipt["outcome"] == "UNKNOWN" and receipt["reason"] == "TRANSPORT_UNKNOWN"
        assert receipt["transport_diagnostic"]["error_class"] == "ConnectionClosedError"
        before = list(client.events)
        with pytest.raises(_InputAdmissionHeldError):
            backend._admit_input("ordinary followup", "TOOL_RESULT", tool_name="ordinary_tool")
        assert client.events == before and backend._input_checker is checker
    else:
        assert not held and returned == "ordinary tool facts" and not backend._input_hold
        assert receipt["outcome"] == "ALLOW" and receipt["reason"] == "NATIVE_NONE"


def test_owned_close_failure_returns_unknown_without_apply_or_replay():
    client = ReusablePoolTransport(close_error=ConnectionClosedError(endpoint_url=ENDPOINT))
    checker = owned_checker(client)
    decision = checker.check("ordinary tool facts", boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert client.close_calls == 1 and client.calls == []
    assert decision.outcome == "UNKNOWN" and decision.forward_text is None
    assert decision.receipt.reason == "TRANSPORT_UNKNOWN"
    assert decision.receipt.transport_diagnostic.error_class == "ConnectionClosedError"
    assert decision.receipt.native_admission is decision.receipt.quality_success is False


def test_invalid_input_has_no_owned_pool_or_assessment_effect():
    client = ReusablePoolTransport()
    decision = owned_checker(client).check("", boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "UNKNOWN" and decision.receipt.reason == "INVALID_INPUT"
    assert client.close_calls == 0 and client.calls == []


def test_default_injected_client_retains_caller_transport_ownership():
    client = ReusablePoolTransport()
    checker = InputAdmissionChecker(
        InputAdmissionConfig(GUARD, "DRAFT", "us-west-2"), client=client, hmac_key=KEY
    )
    decision = checker.check("ordinary tool facts", boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert decision.outcome == "ALLOW" and decision.forward_text == "ordinary tool facts"
    assert client.close_calls == 0 and len(client.calls) == 1
    assert checker._apply.__self__ is client and checker._key is KEY


def test_owned_close_process_control_propagates_without_apply():
    client = ReusablePoolTransport(close_error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        owned_checker(client).check("ordinary tool facts", boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert client.close_calls == 1 and client.calls == []


def test_owned_checker_replacement_preserves_pool_clear_and_binary_key():
    original = ReusablePoolTransport()
    replacement = ReusablePoolTransport()
    checker = owned_checker(original)
    changed = checker._with_client(replacement)
    replacement.idle = True
    decision = changed.check("ordinary tool facts", boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert changed is not checker and changed.config is checker.config and changed._key is checker._key
    assert decision.outcome == "ALLOW" and decision.forward_text == "ordinary tool facts"
    assert replacement.close_calls == 1 and len(replacement.calls) == 1
    assert replacement.events == [("pool_clear", "owned"), ("apply", "owned")]
    assert original.close_calls == 0 and original.calls == []
