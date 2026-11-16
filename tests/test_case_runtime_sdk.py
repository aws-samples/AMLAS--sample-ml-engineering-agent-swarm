"""Actual SDK forwarding and TUI direct context never accept model-authored grants."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from strands import Agent, tool
from test_coding_case_tools import Backend
from test_managed_coding_case_envelope import ACCOUNT, BUCKET, HEADER, IMAGE
from test_sdk_acquisition_binding import request
from test_sdk_prompt_policy import ScriptedModel

from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.coding_case_state import CaseFreezeCatalog, SessionCodingCaseFreezeStore
from amlas.leads.declarations import ML_EXECUTION_LEAD, SubAgentDeclaration, build_lead


def cases(acquired):
    catalog = CaseFreezeCatalog(
        ACCOUNT,
        "us-west-2",
        BUCKET,
        "arn:aws:iam::" + ACCOUNT + ":role/owned",
        IMAGE,
        "b" * 40,
        "a" * 64,
        HEADER["deployment_id"],
        "owned",
    )
    return SessionCodingCaseFreezeStore(acquired, Backend(), catalog)


def test_actual_collection_delegation_forwards_same_case_store_and_revokes_policy():
    seen = []

    @tool(context=True)
    def case_context_probe(tool_context: Any = None) -> dict:
        """Observe typed runtime state; never execute a job."""
        seen.append(tool_context.invocation_state)
        return {"observed": True}

    child = Agent(
        name="data_collection_agent",
        model=ScriptedModel(
            [[request("probe", "case_context_probe", {})], [request("again", "case_context_probe", {})]]
        ),
        tools=[case_context_probe],
        callback_handler=lambda *_a, **_k: None,
    )
    declaration = replace(
        ML_EXECUTION_LEAD,
        sub_agents=(
            SubAgentDeclaration("data_collection_agent", lambda **_: child, "data_collection_model"),
        ),
    )
    parameter = next(
        iter(
            child.as_tool(name="data_collection_agent", preserve_context=True).tool_spec["inputSchema"][
                "json"
            ]["properties"]
        )
    )
    parent = build_lead(
        declaration,
        name="case-parent",
        model=ScriptedModel(
            [
                [request("delegate", "data_collection_agent", {parameter: "inspect opaque case state"})],
                [request("next", "data_collection_agent", {parameter: "read after revocation"})],
            ]
        ),
    )
    parent.callback_handler = lambda *_a, **_k: None
    acquired = SessionAcquisitionStore()
    store = cases(acquired)
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=None)
    parent(
        "delegate",
        invocation_state={
            "_amlas_acquisition_store": acquired,
            "_amlas_case_freeze_store": store,
            "_amlas_approval_policy": policy,
        },
    )
    assert seen[0].get("_amlas_case_freeze_store") is store
    assert (
        seen[0]["_amlas_approval_policy"] is not policy
        and seen[0]["_amlas_approval_policy"].mode is ApprovalMode.AUTO
    )
    parent(
        "again", invocation_state={"_amlas_acquisition_store": acquired, "_amlas_case_freeze_store": store}
    )
    assert (
        seen[1].get("_amlas_case_freeze_store") is store
        and seen[1]["_amlas_approval_policy"].mode is ApprovalMode.ASK
    )
    schema = parent.tool_registry.registry["data_collection_agent"].tool_spec["inputSchema"]["json"][
        "properties"
    ]
    assert set(schema) == {parameter}


def test_tui_has_case_before_preparation_and_fresh_direct_tool_context():
    from amlas.config.settings import Settings
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    backend = ToolBindingBackend(
        Settings(), approval_policy=ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=None)
    )
    state = backend._invocation_state()
    store = state.get("_amlas_case_freeze_store")
    assert type(store) is SessionCodingCaseFreezeStore
    assert store._acquired is backend._acquisition_store and backend._preparation_store._case_store is store
    seen = []

    def probe(case_ref, tool_context=None):
        seen.append(tool_context.invocation_state if tool_context else {})
        return {"observed": True}

    backend._invoke_direct_tool("get_coding_case_freeze", probe, {"case_ref": "case_" + "a" * 32})
    assert seen[0].get("_amlas_case_freeze_store") is store
    backend.approval_policy = None
    backend._invoke_direct_tool("get_coding_case_freeze", probe, {"case_ref": "case_" + "a" * 32})
    assert seen[1]["_amlas_approval_policy"].mode is ApprovalMode.ASK
    with pytest.raises(ValueError):
        backend._invoke_direct_tool(
            "freeze_coding_cases", probe, {"tool_context": {"approved": True}, "case_ref": "fake"}
        )


def test_research_catalog_and_collection_case_routes_use_existing_specialists():
    from amlas.subagents.declarations import (
        DATA_COLLECTION_AGENT,
        RESEARCH_DISCOVERY,
        RESEARCH_SCOUT,
        tool_name_of,
    )

    assert {"list_coding_case_catalog", "freeze_coding_cases", "get_coding_case_freeze"} <= {
        tool_name_of(value) for value in DATA_COLLECTION_AGENT.tools
    }
    for research in (RESEARCH_DISCOVERY, RESEARCH_SCOUT):
        assert "list_coding_case_catalog" in {tool_name_of(value) for value in research.tools}


def test_two_concurrent_parent_contexts_never_share_case_capability():
    from concurrent.futures import ThreadPoolExecutor

    from amlas.control_plane.acquisition_state import acquisition_agent_tool

    seen = []

    def run_one(index):
        @tool(context=True)
        def probe_isolation(tool_context: Any = None) -> dict:
            """Record only typed context; no case data/job effect."""
            state = tool_context.invocation_state
            seen.append((index, state.get("_amlas_case_freeze_store"), state.get("_amlas_acquisition_store")))
            return {"observed": True}

        child = Agent(
            name="data_collection_agent",
            model=ScriptedModel([[request("probe", "probe_isolation", {})]]),
            tools=[probe_isolation],
            callback_handler=lambda *_a, **_k: None,
        )
        delegated = acquisition_agent_tool(child, name="data_collection_agent")
        parameter = next(iter(delegated.tool_spec["inputSchema"]["json"]["properties"]))
        parent = Agent(
            model=ScriptedModel(
                [[request("delegate", "data_collection_agent", {parameter: "read own opaque context"})]]
            ),
            tools=[delegated],
            callback_handler=lambda *_a, **_k: None,
        )
        acquired = SessionAcquisitionStore()
        store = cases(acquired)
        parent(
            "delegate",
            invocation_state={
                "_amlas_acquisition_store": acquired,
                "_amlas_case_freeze_store": store,
                "_amlas_approval_policy": ApprovalPolicy(mode=ApprovalMode.AUTO),
            },
        )
        return acquired, store

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_one, [0, 1]))
    assert len(seen) == 2
    for index, store, acquired in seen:
        assert (acquired, store) == results[index]
    assert results[0][1] is not results[1][1]


def test_same_session_but_foreign_original_store_is_not_forwarded():
    from amlas.control_plane.acquisition_state import acquisition_agent_tool

    seen = []

    @tool(context=True)
    def foreign_probe(tool_context: Any = None) -> dict:
        """Read only the capability presence, never a private artifact."""
        seen.append(tool_context.invocation_state.get("_amlas_case_freeze_store"))
        return {"observed": True}

    child = Agent(
        name="data_collection_agent",
        model=ScriptedModel([[request("probe", "foreign_probe", {})]]),
        tools=[foreign_probe],
        callback_handler=lambda *_a, **_k: None,
    )
    delegated = acquisition_agent_tool(child, name="data_collection_agent")
    parameter = next(iter(delegated.tool_spec["inputSchema"]["json"]["properties"]))
    parent = Agent(
        model=ScriptedModel(
            [[request("delegate", "data_collection_agent", {parameter: "attempt foreign capability"})]]
        ),
        tools=[delegated],
        callback_handler=lambda *_a, **_k: None,
    )
    original = SessionAcquisitionStore()
    foreign = SessionAcquisitionStore()
    foreign.runtime_session_id = original.runtime_session_id
    parent(
        "delegate",
        invocation_state={
            "_amlas_acquisition_store": original,
            "_amlas_case_freeze_store": cases(foreign),
            "_amlas_approval_policy": ApprovalPolicy(mode=ApprovalMode.AUTO),
        },
    )
    assert seen == [None]


def test_ml_evaluation_direct_route_discovers_and_observes_cases_without_gold():
    from amlas.leads.declarations import ML_EXECUTION_LEAD
    from amlas.subagents.declarations import tool_name_of

    names = {tool_name_of(value) for value in ML_EXECUTION_LEAD.direct_tools}
    assert {"list_coding_case_catalog", "get_coding_case_freeze"} <= names
    assert "freeze_coding_cases" not in names
