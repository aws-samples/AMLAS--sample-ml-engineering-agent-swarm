"""Per-request opaque case binding reaches preparation only after fresh approval."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_coding_preparation_binding import REF, auto, store_pair

from amlas.backends.adapters._util import raw_tool
from amlas.flows.coding_preparation_flow import run_coding_preparation_flow
from amlas.tools.coding_preparation import prepare_coding_dataset

CASE = "case_" + "a" * 32


def test_approved_normal_flow_forwards_opaque_case_without_private_values(monkeypatch):
    acquired, backend, store = store_pair()
    calls = []

    def submit(refs, *, case_request_ref=None):
        calls.append((refs, case_request_ref))
        return {"success": True, "accepted": True, "request_ref": "prep_" + "b" * 32}

    monkeypatch.setattr(store, "submit", submit)
    result = run_coding_preparation_flow(
        acquisition_request_refs=[REF], case_request_ref=CASE, policy=auto(), store=store
    )
    assert result["accepted"] is True and result["executed"] is True
    assert calls == [([REF], CASE)]
    assert acquired.calls == backend.calls == 0
    assert "exclusions_ref" not in str(result) and "oracle" not in str(result)


@pytest.mark.parametrize("preview,policy", [(True, auto()), (False, None), (False, auto(1))])
def test_preview_or_denied_case_binding_makes_no_observation(preview, policy, monkeypatch):
    acquired, backend, store = store_pair()
    monkeypatch.setattr(store, "submit", lambda *_a, **_k: pytest.fail("private observation"))
    result = run_coding_preparation_flow(
        acquisition_request_refs=[REF],
        case_request_ref=CASE,
        plan_only=preview,
        policy=policy,
        store=store,
    )
    assert result["status"] == ("planned" if preview else "denied")
    assert result["executed"] is False
    assert acquired.calls == backend.calls == 0


def test_tool_forwards_case_ref_and_hidden_policy_store(monkeypatch):
    _, _, store = store_pair()
    calls = []

    def submit(refs, *, case_request_ref=None):
        calls.append((refs, case_request_ref))
        return {"success": True, "accepted": True}

    monkeypatch.setattr(store, "submit", submit)
    result = raw_tool(prepare_coding_dataset)(
        acquisition_request_refs=[REF],
        case_request_ref=CASE,
        tool_context=SimpleNamespace(
            invocation_state={
                "_amlas_preparation_store": store,
                "_amlas_approval_policy": auto(),
            }
        ),
    )
    assert result["accepted"] is True
    assert calls == [([REF], CASE)]
