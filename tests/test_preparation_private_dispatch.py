"""Private capability state never enters public plans, schemas or JSON results."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from strands import Agent
from test_coding_preparation_binding import REF, auto, store_pair
from test_sdk_prompt_policy import ScriptedModel

from amlas.backends.adapters._util import raw_tool
from amlas.control_plane.acquisition_state import acquisition_agent_tool
from amlas.control_plane.capability import required_params_for
from amlas.tools.coding_preparation import prepare_coding_dataset
from amlas.tools.orchestration_tools import run_ml_capability


def context(store):
    return SimpleNamespace(
        invocation_state={"_amlas_preparation_store": store, "_amlas_approval_policy": auto()}
    )


@pytest.mark.parametrize("plan_only", [True, False])
def test_normal_capability_tool_never_exposes_private_state(plan_only):
    acquired, backend, store = store_pair()
    result = raw_tool(run_ml_capability)(
        "data_preparation_flow",
        json.dumps({"acquisition_request_refs": [REF]}),
        plan_only=plan_only,
        tool_context=context(store),
    )
    assert "store" not in result["plan"].get("params", {})
    encoded = json.dumps(result)
    assert all(
        key not in encoded for key in ("_amlas_preparation_store", "_catalog", "_adapter", "exclusions_ref")
    )
    assert "Prepared source" not in encoded
    assert acquired.calls == backend.calls == (0 if plan_only else 1)


def test_capability_introspection_omits_private_store_catalog_and_policy():
    required, optional = required_params_for("data_preparation_flow")
    assert required == ["acquisition_request_refs"]
    assert not {"store", "catalog", "policy"}.intersection(optional)


def test_normal_capability_exception_reports_no_private_values(monkeypatch):
    _, backend, store = store_pair()
    monkeypatch.setattr(
        store, "submit", lambda _: (_ for _ in ()).throw(RuntimeError("private-catalog-sentinel"))
    )
    result = raw_tool(run_ml_capability)(
        "data_preparation_flow",
        json.dumps({"acquisition_request_refs": [REF]}),
        plan_only=False,
        tool_context=context(store),
    )
    assert result["executed"] is False and result["result"]["status"] == "refused"
    assert "private-catalog-sentinel" not in json.dumps(result)
    assert backend.calls == 0


def test_model_authored_store_or_policy_is_never_an_execution_grant():
    _, backend, store = store_pair()
    result = raw_tool(run_ml_capability)(
        "data_preparation_flow",
        json.dumps(
            {"acquisition_request_refs": [REF], "store": {"accepted": True}, "policy": {"mode": "auto"}}
        ),
        plan_only=False,
        tool_context=context(store),
    )
    assert result["executed"] is False
    assert backend.calls == 0
    json.dumps(result)


@pytest.mark.parametrize("foreign", [False, True])
def test_actual_sdk_specialist_forwards_only_same_conversation_store(foreign):
    acquired, backend, store = store_pair()
    parent_acquired = store_pair()[0] if foreign else acquired
    child = Agent(
        name="data_collection_agent",
        model=ScriptedModel(
            [
                [
                    {
                        "id": "prepare",
                        "name": "prepare_coding_dataset",
                        "input": {"acquisition_request_refs": [REF]},
                    }
                ]
            ]
        ),
        tools=[prepare_coding_dataset],
        callback_handler=lambda *_a, **_k: None,
    )
    delegate = acquisition_agent_tool(child, name="data_collection_agent")
    name = next(iter(delegate.tool_spec["inputSchema"]["json"]["properties"]))
    parent = Agent(
        model=ScriptedModel(
            [
                [
                    {
                        "id": "delegate",
                        "name": "data_collection_agent",
                        "input": {name: "Prepare canonical coding data."},
                    }
                ]
            ]
        ),
        tools=[delegate],
        callback_handler=lambda *_a, **_k: None,
    )
    parent(
        "delegate",
        invocation_state={
            "_amlas_acquisition_store": parent_acquired,
            "_amlas_preparation_store": store,
            "_amlas_approval_policy": auto(),
        },
    )
    assert backend.calls == (0 if foreign else 1)
