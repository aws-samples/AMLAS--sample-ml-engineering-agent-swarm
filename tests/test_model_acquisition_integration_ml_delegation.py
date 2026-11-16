"""Actual normal ML Lead delegation must preserve typed model acquisition state."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from strands import Agent, tool
from test_sdk_acquisition_binding import request
from test_sdk_prompt_policy import ScriptedModel

from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.leads.declarations import ML_EXECUTION_LEAD, SubAgentDeclaration, build_lead


def test_actual_ml_execution_lead_trainer_keeps_private_model_state():
    seen = []

    @tool(context=True)
    def probe_trainer_context(tool_context: Any = None) -> dict[str, bool]:
        """Observe typed state only; never run model acquisition or AWS."""
        state = tool_context.invocation_state
        seen.append(state)
        return {
            "typed": isinstance(state.get("_amlas_model_acquisition_store"), SessionModelAcquisitionStore)
        }

    child = Agent(
        name="trainer_tuner",
        model=ScriptedModel(
            [
                [request("probe", "probe_trainer_context", {})],
                [request("probe_next", "probe_trainer_context", {})],
            ]
        ),
        tools=[probe_trainer_context],
        callback_handler=lambda *a, **k: None,
    )
    declaration = replace(
        ML_EXECUTION_LEAD,
        sub_agents=(SubAgentDeclaration("trainer_tuner", lambda **_: child, "trainer_tuner_model"),),
    )
    parameter = next(
        iter(
            child.as_tool(name="trainer_tuner", preserve_context=True).tool_spec["inputSchema"]["json"][
                "properties"
            ]
        )
    )
    parent = build_lead(
        declaration,
        model=ScriptedModel(
            [
                [request("delegate", "trainer_tuner", {parameter: "Inspect model state."})],
                [request("delegate_next", "trainer_tuner", {parameter: "Poll model state."})],
            ]
        ),
        name="ml_execution_lead",
    )
    parent.callback_handler = lambda *a, **k: None
    acquired = SessionAcquisitionStore()
    models = SessionModelAcquisitionStore(runtime_session_id=acquired.runtime_session_id)
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO)
    parent(
        "delegate",
        invocation_state={
            "_amlas_acquisition_store": acquired,
            "_amlas_model_acquisition_store": models,
            "_amlas_approval_policy": policy,
        },
    )
    assert seen[0].get("_amlas_model_acquisition_store") is models
    assert seen[0]["_amlas_approval_policy"].mode is ApprovalMode.AUTO

    assert seen[0]["_amlas_approval_policy"] is not policy
    public = parent.tool_registry.registry["trainer_tuner"].tool_spec["inputSchema"]["json"]["properties"]
    assert set(public) == {parameter}
    parent(
        "delegate again",
        invocation_state={"_amlas_acquisition_store": acquired, "_amlas_model_acquisition_store": models},
    )
    assert seen[1].get("_amlas_model_acquisition_store") is models
    assert seen[1]["_amlas_approval_policy"].mode is ApprovalMode.ASK


def test_other_lead_delegation_retains_ordinary_sdk_state():
    seen = []

    @tool(context=True)
    def probe_other_context(tool_context: Any = None) -> dict[str, bool]:
        """Observe ordinary delegation; never run an external effect."""
        seen.append(tool_context.invocation_state)
        return {"observed": True}

    child = Agent(
        name="algorithm_specialist",
        model=ScriptedModel([[request("probe", "probe_other_context", {})]]),
        tools=[probe_other_context],
        callback_handler=lambda *a, **k: None,
    )
    declaration = replace(
        ML_EXECUTION_LEAD,
        sub_agents=(SubAgentDeclaration("algorithm_specialist", lambda **_: child, "algorithm_model"),),
    )
    parameter = next(
        iter(
            child.as_tool(name="algorithm_specialist", preserve_context=True).tool_spec["inputSchema"][
                "json"
            ]["properties"]
        )
    )
    parent = build_lead(
        declaration,
        model=ScriptedModel(
            [[request("delegate", "algorithm_specialist", {parameter: "Inspect ordinary state."})]]
        ),
        name="ml_execution_lead",
    )
    parent.callback_handler = lambda *a, **k: None
    acquired = SessionAcquisitionStore()
    models = SessionModelAcquisitionStore(runtime_session_id=acquired.runtime_session_id)
    parent(
        "delegate",
        invocation_state={
            "_amlas_acquisition_store": acquired,
            "_amlas_model_acquisition_store": models,
            "_amlas_approval_policy": ApprovalPolicy(mode=ApprovalMode.AUTO),
        },
    )
    assert seen[0].get("_amlas_model_acquisition_store") is None
    assert seen[0].get("_amlas_approval_policy") is None
