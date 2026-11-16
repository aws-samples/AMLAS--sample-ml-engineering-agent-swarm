"""Source discovery distinguishes resource shape from original-case provenance."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from test_coding_preparation_binding import catalog, store_pair

from amlas.backends.adapters._util import raw_tool
from amlas.control_plane.preparation_capabilities import describe_preparation_capabilities
from amlas.tools.coding_preparation import list_coding_preparation_profiles


def without_global_exclusions():
    selected = catalog()
    return dataclasses.replace(
        selected,
        exclusions_ref=dataclasses.replace(
            selected.exclusions_ref, uri="", version_id="", sha256="", byte_count=0
        ),
    )


def test_configured_resources_disclose_case_route_without_case_readiness():
    result = describe_preparation_capabilities(catalog=without_global_exclusions())
    assert all(p["configuration_status"] == "unconfigured" for p in result["profiles"])
    assert result["resource_configuration_status"] == "configured"
    assert result["missing_resource_selectors"] == []
    assert all(
        p["configuration_scope"] == "worker_resources_and_global_exclusions" for p in result["profiles"]
    )
    assert (
        result["exclusions_requirement"]
        == "configured_global_exclusions_or_verified_original_case_request_ref"
    )
    assert result["case_request_ref_route"] == {
        "supported": True,
        "tool": "prepare_coding_dataset",
        "argument": "case_request_ref",
        "case_readiness": "unknown",
        "required_provenance": "original_same_conversation_frozen_case_and_verified_EXCLUSIONS_COMMITTED",  # pragma: allowlist secret - public provenance label
        "normal_tool_order": ["freeze_coding_cases", "get_coding_case_freeze", "prepare_coding_dataset"],
    }
    assert result["readiness"] == "unknown"
    assert all(result[k] is False for k in ("native_admission", "executed", "prepared", "trained"))


@pytest.mark.parametrize("selected", [None, dataclasses.replace(catalog(), image_uri="invalid")])
def test_unknown_catalog_resources_do_not_claim_case_or_readiness(selected):
    result = describe_preparation_capabilities(catalog=selected)
    assert result["resource_configuration_status"] == "unknown"
    assert result["case_request_ref_route"]["case_readiness"] == "unknown"
    assert result["native_admission"] is False


def test_missing_worker_resource_remains_unconfigured_with_only_selector_names():
    result = describe_preparation_capabilities(
        catalog=dataclasses.replace(without_global_exclusions(), image_uri="")
    )
    assert result["resource_configuration_status"] == "unconfigured"
    assert result["missing_resource_selectors"] == ["AMLAS_DATA_PREPARATION_IMAGE_URI"]
    assert result["native_admission"] is False


def test_existing_tool_uses_only_held_catalog_no_case_observation_or_effect():
    acquired, effects, store = store_pair(without_global_exclusions())
    result = raw_tool(list_coding_preparation_profiles)(
        tool_context=SimpleNamespace(invocation_state={"_amlas_preparation_store": store})
    )
    assert result["resource_configuration_status"] == "configured"
    assert result["case_request_ref_route"]["case_readiness"] == "unknown"
    assert acquired.calls == effects.calls == effects.observations == 0
    assert set(list_coding_preparation_profiles.tool_spec["inputSchema"]["json"]["properties"]) == {
        "repo_id",
        "revision",
        "filename",
    }


def test_actual_direct_discovery_describes_route_without_case_or_effect():
    from amlas.config.settings import Settings
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    acquired, effects, store = store_pair(without_global_exclusions())
    backend = ToolBindingBackend(Settings())
    backend._preparation_store = store
    result = backend.execute_tool_use("list_coding_preparation_profiles", {})
    assert result.status == "executed"
    assert result.result["resource_configuration_status"] == "configured"
    assert result.result["case_request_ref_route"]["case_readiness"] == "unknown"
    assert result.result["native_admission"] is False
    assert acquired.calls == effects.calls == effects.observations == 0


def test_actual_sdk_discovery_reports_same_provenance_requirement_without_effect():
    import json

    from strands import Agent
    from strands.hooks.events import AfterToolCallEvent
    from test_sdk_prompt_policy import ScriptedModel

    acquired, effects, store = store_pair(without_global_exclusions())
    values = []
    agent = Agent(
        model=ScriptedModel([[{"id": "discovery", "name": "list_coding_preparation_profiles", "input": {}}]]),
        tools=[list_coding_preparation_profiles],
        callback_handler=lambda *_a, **_kw: None,
    )

    def capture(event):
        for block in event.result["content"]:
            if "json" in block:
                values.append(block["json"])
            elif "text" in block:
                try:
                    values.append(json.loads(block["text"]))
                except json.JSONDecodeError:
                    pass

    agent.hooks.add_callback(AfterToolCallEvent, capture)
    agent("Inspect preparation capability", invocation_state={"_amlas_preparation_store": store})
    assert values[0]["resource_configuration_status"] == "configured"
    assert all(p["configuration_status"] == "unconfigured" for p in values[0]["profiles"])
    assert values[0]["case_request_ref_route"]["case_readiness"] == "unknown"
    assert values[0]["native_admission"] is False
    assert acquired.calls == effects.calls == effects.observations == 0
