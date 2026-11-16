"""Additive model-pack API controls over the installed SDK; no model/provider bytes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import model_acquisition_fixtures as f
import pytest
from strands import Agent
from strands.hooks import AfterToolCallEvent
from test_sdk_acquisition_binding import invoke, request
from test_sdk_prompt_policy import ScriptedModel

from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest
from amlas.flows import model_weight_acquisition_flow as flow
from amlas.flows._util import raw
from amlas.tools import model_weight_tools as tools
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

PACK = "model_pack_" + "a" * 64
PARAMS = {"catalog_id": f.CATALOG, "catalog_version": f.VERSION, "mode": "reuse_verified", "pack_ref": PACK}


@pytest.fixture(autouse=True)
def namespace(monkeypatch):
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *a, **k: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")

    def refuse_fetch():
        raise AssertionError("reuse resolved ordinary fetch")

    monkeypatch.setattr(flow, "_resolve_backend", refuse_fetch)


def test_additive_public_mode_and_context_only_catalog_schema():
    properties = tools.acquire_model_weights.tool_spec["inputSchema"]["json"]["properties"]
    assert "mode" in properties, "missing additive mode schema"
    assert properties["mode"]["enum"] == ["acquire", "reuse_verified"]
    assert set(properties) == {"catalog_id", "catalog_version", "plan_only", "mode", "pack_ref"}
    assert tools.list_model_weight_catalog.tool_spec["inputSchema"]["json"]["properties"] == {}
    assert raw(tools.list_model_weight_catalog)() == [expected_model_weight_manifest(f.CATALOG, f.VERSION)]


def test_actual_sdk_reuse_keeps_original_hidden_store_snapshot_and_lock(monkeypatch):
    seen = []

    def reuse(self, *, catalog_id, catalog_version, pack_ref, policy):
        assert self._submission_lock.locked()
        assert (catalog_id, catalog_version, pack_ref) == (f.CATALOG, f.VERSION, PACK)
        assert type(policy) is ApprovalPolicy and policy.mode is ApprovalMode.AUTO
        seen.append((self, policy))
        return {
            "success": True,
            "accepted": True,
            "acquired": True,
            "trusted": False,
            "ready": False,
            "trained": False,
            "executed": True,
            "status": "reused_verified",
            "request_ref": "model_acq_" + "b" * 32,
        }

    monkeypatch.setattr(SessionModelAcquisitionStore, "reuse_verified", reuse, raising=False)
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO)
    backend = ToolBindingBackend(Settings(), approval_policy=policy)
    result = invoke(
        monkeypatch,
        backend,
        [request("reuse", "acquire_model_weights", PARAMS)],
        tools=[tools.acquire_model_weights],
    )
    assert result["reuse"].get("status") == "reused_verified"
    assert result["reuse"]["request_ref"] == "model_acq_" + "b" * 32
    assert result["reuse"]["trained"] is False and result["reuse"]["ready"] is False
    assert len(seen) == 1 and seen[0][0] is backend._model_acquisition_store
    assert seen[0][1] is not policy and seen[0][1].allowlist == policy.allowlist
    assert not backend._model_acquisition_store._submission_lock.locked()


@pytest.mark.parametrize(
    "policy", [None, ApprovalPolicy(), ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=10)]
)
def test_actual_sdk_reuse_denies_before_provider_read_or_copy(monkeypatch, policy):
    seen = []
    monkeypatch.setattr(
        SessionModelAcquisitionStore, "reuse_verified", lambda *a, **k: seen.append("effect"), raising=False
    )
    backend = ToolBindingBackend(Settings(), approval_policy=policy)
    result = invoke(
        monkeypatch,
        backend,
        [request("reuse", "acquire_model_weights", PARAMS)],
        tools=[tools.acquire_model_weights],
    )
    assert result["reuse"].get("status") == "denied"
    assert seen == []
    assert not backend._model_acquisition_store._records


def test_actual_sdk_catalog_uses_only_typed_hidden_store_and_matching_opaque_candidates(monkeypatch):
    seen = []
    store = SessionModelAcquisitionStore(runtime_session_id="model-catalog-consumer")
    entries = [
        {"catalog_id": f.CATALOG, "catalog_version": f.VERSION, "pack_ref": PACK},
        {"catalog_id": "foreign-catalog", "catalog_version": f.VERSION, "pack_ref": "model_pack_" + "c" * 64},
    ]

    def candidates(self):
        seen.append(self)
        return entries

    monkeypatch.setattr(SessionModelAcquisitionStore, "verified_pack_candidates", candidates, raising=False)
    model = ScriptedModel([[request("catalog", "list_model_weight_catalog", {})]])
    agent = Agent(model=model, tools=[tools.list_model_weight_catalog], callback_handler=lambda *a, **k: None)
    observed = []

    def collect(event: AfterToolCallEvent):
        for block in event.result["content"]:
            if "json" in block:
                observed.append(block["json"])
            elif "text" in block:
                observed.append(json.loads(block["text"]))

    agent.hooks.add_callback(AfterToolCallEvent, collect)
    agent(
        "List expected model and available opaque verified packs.",
        invocation_state={"_amlas_model_acquisition_store": store},
    )
    assert seen == [store]
    assert observed[0][0]["verified_packs"] == entries[:1]
    assert set(observed[0][0]["verified_packs"][0]) == {"catalog_id", "catalog_version", "pack_ref"}
    assert raw(tools.list_model_weight_catalog)(tool_context=SimpleNamespace(invocation_state={})) == [
        expected_model_weight_manifest(f.CATALOG, f.VERSION)
    ]


def test_typed_store_without_pack_candidates_preserves_legacy_catalog(monkeypatch):
    store = SessionModelAcquisitionStore(runtime_session_id="unconfigured-model-consumer")
    monkeypatch.setattr(
        SessionModelAcquisitionStore, "verified_pack_candidates", lambda self: [], raising=False
    )
    context = SimpleNamespace(invocation_state={"_amlas_model_acquisition_store": store})
    assert raw(tools.list_model_weight_catalog)(tool_context=context) == [
        expected_model_weight_manifest(f.CATALOG, f.VERSION)
    ]
