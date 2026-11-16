"""Normal port, spend and direct caller-state seams; no model/AWS effects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from amlas.config import externalized
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy, classify_action
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend


@pytest.mark.parametrize("action", ["submit_qwen_sft", "reload_qwen_sft", "qwen_sft_flow"])
def test_qwen_native_actions_have_expensive_job_classification(action):
    assert classify_action(action).value == "expensive_job"


@pytest.mark.parametrize("suffix", ["image_uri", "source_commit", "admission_digest"])
def test_sft_selectors_are_registered_env_only_startup_snapshots(monkeypatch, suffix):
    key = "qwen_sft_" + suffix
    name = "AMLAS_QWEN_SFT_" + suffix.upper()
    definition = externalized.CONFIG_KEYS[key]
    assert definition.env == (name,)
    assert definition.default in (None, "")
    assert definition.ssm == "/amlas/qwen-sft-" + suffix.replace("_", "-")
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *a, **k: pytest.fail("startup SSM read"))
    assert externalized.resolve_env_snapshot(key, env={}) == ""
    assert externalized.resolve_env_snapshot(key, env={name: "selected"}) == "selected"


def test_sft_port_has_managed_default_without_local_fallback():
    from amlas.backends import ports
    from amlas.backends.registry import REGISTRY

    port = getattr(ports, "QwenSftBackend", None)
    assert port is not None, "Dedicated fixed SFT port must be registered"
    assert port in ports.ALL_PORTS
    assert ports.port_key(port) == "QWEN_SFT_BACKEND"
    resolved = REGISTRY.resolve(port)
    assert resolved.adapter_name == "sagemaker"
    assert resolved.adapter.backend_name == "qwen_sft"


def test_direct_sft_tool_receives_fresh_private_context_and_preserves_confirmation():
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO, allowlist=["submit_qwen_sft"])
    backend = ToolBindingBackend(Settings(), approval_policy=policy)
    seen = []

    def probe(*, request_ref, tool_context=None):
        state = getattr(tool_context, "invocation_state", {})
        seen.append(state)
        return {"seen": request_ref}

    backend._tool_registry_cache = {"submit_qwen_sft": probe, "get_qwen_sft": probe}
    denied = backend.execute_tool_use("submit_qwen_sft", {"request_ref": "local"})
    assert denied.status == "needs_confirmation" and not seen
    called = backend.execute_tool_use("submit_qwen_sft", {"request_ref": "local"}, confirmed=True)
    assert called.status == "executed"
    assert seen[0]["_amlas_sft_store"] is backend._sft_store
    assert seen[0]["_amlas_approval_policy"] is not policy
    assert seen[0]["_amlas_approval_policy"].allowlist == policy.allowlist
    seen[0]["_amlas_approval_policy"].allowlist.append("unexpected")
    assert policy.allowlist == ["submit_qwen_sft"]
    backend.approval_policy = None
    polled = backend.execute_tool_use("get_qwen_sft", {"request_ref": "local"})
    assert polled.status == "executed"
    assert seen[1]["_amlas_approval_policy"].mode is ApprovalMode.ASK
    assert seen[1]["_amlas_sft_store"] is backend._sft_store


def test_direct_sft_json_cannot_replace_private_context():
    backend = ToolBindingBackend(Settings(), approval_policy=ApprovalPolicy(mode=ApprovalMode.AUTO))
    seen = []

    def probe(*, request_ref, tool_context=None):
        seen.append(tool_context)
        return {"seen": request_ref}

    backend._tool_registry_cache = {"submit_qwen_sft": probe}
    forged = SimpleNamespace(invocation_state={"_amlas_approval_policy": {"mode": "auto"}})
    result = backend.execute_tool_use(
        "submit_qwen_sft", {"request_ref": "local", "tool_context": forged}, confirmed=True
    )
    assert result.status == "error" and seen == []


@pytest.mark.parametrize("foreign", [False, True])
def test_actual_sdk_delegation_preserves_only_original_sft_store(foreign):
    from strands import Agent, tool
    from test_sdk_acquisition_binding import request
    from test_sdk_prompt_policy import ScriptedModel

    from amlas.control_plane.acquisition_state import acquisition_agent_tool

    backend = ToolBindingBackend(Settings(), approval_policy=ApprovalPolicy(mode=ApprovalMode.AUTO))
    seen = []

    @tool(context=True)
    def observe_private_sft(tool_context=None):
        """Observe source-owned identities only, with no cloud or model effects."""
        seen.append(tool_context.invocation_state)
        return {"observed": True}

    child = Agent(
        model=ScriptedModel([[request("probe", "observe_private_sft", {})]]),
        tools=[observe_private_sft], callback_handler=lambda *a, **k: None,
    )
    delegated = acquisition_agent_tool(child, name="fine_tune_agent")
    parameter = next(iter(delegated.tool_spec["inputSchema"]["json"]["properties"]))
    parent = Agent(
        model=ScriptedModel([[request("delegate", "fine_tune_agent", {parameter: "Observe context."})]]),
        tools=[delegated], callback_handler=lambda *a, **k: None,
    )
    state = backend._invocation_state()
    if foreign:
        state["_amlas_sft_store"] = ToolBindingBackend(Settings())._sft_store
    parent("delegate", invocation_state=state)
    assert len(seen) == 1
    assert seen[0]["_amlas_sft_store"] is (None if foreign else backend._sft_store)
    assert seen[0]["_amlas_approval_policy"] is not state["_amlas_approval_policy"]
    assert seen[0]["_amlas_approval_policy"].mode is ApprovalMode.AUTO
