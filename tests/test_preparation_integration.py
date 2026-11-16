"""Normal managed-preparation bindings; actual SDK policy, no AWS effects."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from test_coding_preparation_binding import REF, auto, catalog, store_pair

from amlas.backends import ports
from amlas.backends.registry import AdapterUnavailableError, BackendRegistry
from amlas.config import externalized
from amlas.config.settings import Settings
from amlas.control_plane import capability
from amlas.control_plane.approvals import ActionClass, ApprovalPolicy, classify_action
from amlas.control_plane.preparation_binding import build_preparation_store
from amlas.control_plane.preparation_state import SessionPreparationStore
from amlas.flows.coding_preparation_flow import run_coding_preparation_flow
from amlas.tui.chat import tool_binding_backend


def configuration_env():
    selected = catalog()
    return {
        "AMLAS_AWS_ACCOUNT_ID": selected.account_id,
        "AMLAS_AWS_REGION": selected.region,
        "AMLAS_SESSION_BUCKET": selected.bucket,
        "SAGEMAKER_EXECUTION_ROLE_ARN": selected.execution_role_arn,
        "AMLAS_DEPLOYMENT_ID": selected.deployment_id,
        "AMLAS_DEPLOYMENT_PREFIX": selected.deployment_prefix,
        "AMLAS_DATA_PREPARATION_IMAGE_URI": selected.image_uri,
        "AMLAS_DATA_PREPARATION_SOURCE_COMMIT": selected.source_commit,
        "AMLAS_DATA_PREPARATION_ADMISSION_DIGEST": selected.admission_digest,
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_URI": selected.exclusions_ref.uri,
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID": selected.exclusions_ref.version_id,
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_SHA256": selected.exclusions_ref.sha256,
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_BYTE_COUNT": str(selected.exclusions_ref.byte_count),
    }


def test_exact_preparation_action_is_expensive():
    assert classify_action("data_preparation_flow") is ActionClass.EXPENSIVE_JOB


def test_normal_registry_uses_managed_constructor_without_sdk(monkeypatch):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *_a, **_k: pytest.fail("startup SDK"))
    port = getattr(ports, "CodingPreparationBackend", None)
    assert port in ports.ALL_PORTS
    assert ports.port_key(port) == "DATA_PREPARATION_BACKEND"
    resolved = BackendRegistry().resolve(port)
    assert resolved.adapter_name == "processing"
    assert isinstance(resolved.adapter, port)
    monkeypatch.setenv("DATA_PREPARATION_BACKEND", "local")
    with pytest.raises(AdapterUnavailableError):
        BackendRegistry().resolve(port)


def test_startup_snapshot_never_calls_ssm_or_sdk(monkeypatch):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *_a, **_k: pytest.fail("startup SDK"))
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *_a, **_k: pytest.fail("startup SSM"))
    acquired, _, _ = store_pair()
    store = build_preparation_store(acquired, env=configuration_env())
    assert isinstance(store, SessionPreparationStore)
    assert store._acquired is acquired
    assert store._catalog == catalog()
    assert not hasattr(store, "policy")


def test_missing_startup_selectors_are_bounded_names_before_source_sdk():
    acquired, _, _ = store_pair()
    store = build_preparation_store(acquired, env={})
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    assert result["status"] == "configuration_missing"
    assert result["executed"] is False and result["accepted"] is False
    assert acquired.calls == 0
    assert 1 <= len(result["missing_selectors"]) <= 13
    assert all(name in configuration_env() for name in result["missing_selectors"])
    assert not any(value in str(result) for value in (catalog().image_uri, catalog().exclusions_ref.uri))


@pytest.mark.parametrize(
    "change",
    [
        {"admission_digest": "c" * 64},
        {"source_commit": "main"},
        {"image_uri": "caller:latest"},
        {"exclusions_ref": dataclasses.replace(catalog().exclusions_ref, version_id="null")},
    ],
)
def test_malformed_or_unadmitted_catalog_remains_refused(change):
    acquired, backend, store = store_pair(catalog(**change))
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    assert result["status"] == "refused" and result["executed"] is False
    assert acquired.calls == backend.calls == 0


def test_backend_nonaccepted_result_does_not_claim_execution(monkeypatch):
    _, _, store = store_pair()
    monkeypatch.setattr(store, "submit", lambda _: {"success": False, "accepted": False, "status": "refused"})
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    assert result["executed"] is False


def test_accepted_handle_preserved_when_observation_fails(monkeypatch):
    _, _, store = store_pair()
    pending = {
        "success": False,
        "accepted": True,
        "status": "observation_failed",
        "request_ref": "prep_" + "e" * 32,
    }
    monkeypatch.setattr(store, "submit", lambda _: pending)
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    assert result["executed"] is True and result["request_ref"] == pending["request_ref"]


@pytest.mark.parametrize("policy", [None, ApprovalPolicy(), auto(100)])
def test_normal_capability_unknown_cost_denies_before_effects(policy):
    acquired, backend, store = store_pair()
    result = capability.invoke_capability(
        capability_id="data_preparation_flow",
        params={"acquisition_request_refs": [REF]},
        _trusted_preparation_store=store,
        policy=policy,
        plan_only=False,
    )
    assert result.status == "denied" and result.executed is False
    assert acquired.calls == backend.calls == 0
    assert result.plan["estimated_cost"] is None


def test_normal_capability_single_fresh_approval_and_no_double_gate():
    acquired, backend, store = store_pair()
    result = capability.invoke_capability(
        capability_id="data_preparation_flow",
        params={"acquisition_request_refs": [REF]},
        _trusted_preparation_store=store,
        policy=auto(),
        plan_only=False,
    )
    assert result.approved and result.executed
    assert result.result["accepted"] is True and backend.calls == acquired.calls == 1
    denied = capability.invoke_capability(
        capability_id="data_preparation_flow",
        params={"acquisition_request_refs": [REF]},
        _trusted_preparation_store=store,
        plan_only=False,
    )
    assert denied.status == "denied" and backend.calls == 1


def test_preparation_capability_plan_only_remains_pure():
    acquired, backend, store = store_pair()
    result = capability.invoke_capability(
        capability_id="data_preparation_flow",
        params={"acquisition_request_refs": [REF]},
        _trusted_preparation_store=store,
        policy=auto(),
        plan_only=True,
    )
    assert result.status == "planned" and result.executed is False
    assert result.plan["estimated_cost"] is None and acquired.calls == backend.calls == 0


def test_tui_normal_constructor_and_per_turn_state_are_private_and_fresh(monkeypatch):
    monkeypatch.setattr(tool_binding_backend.ToolBindingBackend, "is_tool_capable", property(lambda _: True))
    states = []

    def agent(_prompt, *, invocation_state):
        states.append(invocation_state)
        return SimpleNamespace(message={"content": [{"text": "done"}]})

    backend = tool_binding_backend.ToolBindingBackend(Settings(), approval_policy=auto())
    monkeypatch.setattr(backend, "build_agent", lambda **_: agent)
    backend.chat_with_tools("prepare coding data")
    backend.approval_policy = None
    backend.chat_with_tools("observe")
    assert isinstance(states[0].get("_amlas_preparation_store"), SessionPreparationStore)
    assert states[0]["_amlas_preparation_store"] is states[1]["_amlas_preparation_store"]
    assert states[0]["_amlas_preparation_store"]._acquired is states[0]["_amlas_acquisition_store"]
    assert states[0]["_amlas_approval_policy"] is not states[1]["_amlas_approval_policy"]
    assert states[1]["_amlas_approval_policy"] == ApprovalPolicy()
    other = tool_binding_backend.ToolBindingBackend(Settings())
    assert other._preparation_store.runtime_session_id != backend._preparation_store.runtime_session_id


def test_tools_are_bound_to_tui_and_collection_specialist_only():
    from amlas.subagents.declarations import DATA_COLLECTION_AGENT
    from amlas.tui.chat.agent_tools import READ_ONLY_TOOLS, get_safe_llm_tools, tool_display_name

    names = {tool_display_name(value) for value in get_safe_llm_tools()}
    assert {"prepare_coding_dataset", "get_coding_preparation"} <= names
    assert {"prepare_coding_dataset", "get_coding_preparation"} <= {
        tool_display_name(value) for value in DATA_COLLECTION_AGENT.tools
    }
    assert "get_coding_preparation" in READ_ONLY_TOOLS
    assert "prepare_coding_dataset" not in READ_ONLY_TOOLS
