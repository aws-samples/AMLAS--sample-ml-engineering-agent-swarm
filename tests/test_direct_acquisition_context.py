"""Actual normal direct tools with neutral records; no native admission."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_acquisition_binding import PIN, RecordingBackend, allowed

from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.flows import dataset_acquisition_flow as flow
from amlas.tools import huggingface_data
from amlas.tui.chat import tool_binding_backend
from amlas.tui.chat.agent_tools import LONG_RUNNING_TOOLS


class StrictAcquisitionBackend(RecordingBackend):
    def __init__(self):
        super().__init__()
        self.expected_session = None
        self.handles = []
        self.observed = []

    def submit(self, **kwargs):
        assert kwargs == {**PIN, "resources": {"runtime_session_id": self.expected_session}}
        handle = super().submit(**kwargs)
        self.handles.append(handle)
        return handle

    def artifacts(self, handle):
        assert any(handle is original for original in self.handles)
        self.observed.append(handle)
        return super().artifacts(handle)


@pytest.fixture
def direct(monkeypatch):
    effects = StrictAcquisitionBackend()
    monkeypatch.setattr(
        flow,
        "_resolve_backend",
        lambda: SimpleNamespace(adapter=effects, adapter_name=effects.backend_name),
    )
    monkeypatch.setattr(
        tool_binding_backend,
        "get_safe_llm_tools",
        lambda: [huggingface_data.acquire_huggingface_dataset, huggingface_data.get_dataset_acquisition],
    )
    backend = tool_binding_backend.ToolBindingBackend(Settings(), approval_policy=allowed())
    effects.expected_session = backend._acquisition_store.runtime_session_id
    states = []
    actual_state = backend._invocation_state

    def capture():
        state = actual_state()
        states.append(state)
        return state

    monkeypatch.setattr(backend, "_invocation_state", capture)
    return backend, effects, states


def seed_original(backend):
    result = flow.run_dataset_acquisition_flow(**PIN, store=backend._acquisition_store, policy=allowed())
    assert result["accepted"] is True
    return result["request_ref"]


def test_direct_actual_acquire_uses_original_store_and_current_policy_snapshot(direct):
    backend, effects, states = direct
    caller = backend.approval_policy
    submitted = backend.execute_tool_use("acquire_huggingface_dataset", dict(PIN), confirmed=True)
    assert submitted.status == "executed"
    assert submitted.result["accepted"] is True, submitted.result
    reference = submitted.result["request_ref"]
    assert reference.startswith("acq_")
    original = backend._acquisition_store._records[reference]
    assert original.adapter is effects and original.handle is effects.handles[0]
    assert states, "fresh backend private snapshot was not constructed"
    assert states[-1]["_amlas_acquisition_store"] is backend._acquisition_store
    policy = states[-1]["_amlas_approval_policy"]
    assert policy is not caller and policy.allowlist is not caller.allowlist
    assert policy.mode == ApprovalMode.ALLOWLIST
    caller.allowlist.clear()
    assert policy.allowlist == ["data_collection_flow"]
    denied = backend.execute_tool_use("acquire_huggingface_dataset", dict(PIN), confirmed=True)
    assert denied.result["status"] == "denied" and len(effects.calls) == 1
    assert states, "fresh backend private snapshot was not constructed"
    assert states[-1]["_amlas_approval_policy"].allowlist == []


def test_direct_actual_read_only_poll_uses_original_record_and_fresh_ask_policy(direct):
    backend, effects, states = direct
    reference = seed_original(backend)
    backend.approval_policy = None
    read = backend.execute_tool_use("get_dataset_acquisition", {"request_ref": reference})
    assert read.status == "executed"
    assert read.result["accepted"] is True, read.result
    assert read.result["request_ref"] == reference
    assert read.result["acquired"] is False
    assert effects.observed == [effects.handles[0]] and len(effects.calls) == 1
    assert states, "fresh backend private snapshot was not constructed"
    assert states[-1]["_amlas_acquisition_store"] is backend._acquisition_store
    assert states, "fresh backend private snapshot was not constructed"
    assert states[-1]["_amlas_approval_policy"].mode == ApprovalMode.ASK
    foreign = tool_binding_backend.ToolBindingBackend(Settings(), approval_policy=allowed())
    polled = foreign.execute_tool_use("get_dataset_acquisition", {"request_ref": reference})
    assert polled.result["status"] == "unknown_reference"
    assert effects.observed == [effects.handles[0]]


def test_confirmation_precedes_hidden_state_and_backend_resolution(direct):
    backend, effects, states = direct
    result = backend.execute_tool_use("acquire_huggingface_dataset", dict(PIN))
    assert result.status == "needs_confirmation"
    assert result.result is None and not states and not effects.calls


@pytest.mark.parametrize("policy", [None, ApprovalPolicy()])
def test_confirmation_is_not_acquisition_approval(direct, policy):
    backend, effects, states = direct
    backend.approval_policy = policy
    result = backend.execute_tool_use("acquire_huggingface_dataset", dict(PIN), confirmed=True)
    assert result.result["status"] == "denied" and not effects.calls
    assert states, "fresh backend private snapshot was not constructed"
    assert states[-1]["_amlas_approval_policy"].mode == ApprovalMode.ASK


def test_direct_plan_only_is_preview_without_effect_or_policy_grant(direct):
    backend, effects, states = direct
    backend.approval_policy = None
    result = backend.execute_tool_use(
        "acquire_huggingface_dataset",
        {**PIN, "plan_only": True},
        confirmed=True,
    )
    assert result.status == "executed" and result.result["status"] == "planned"
    assert result.result["executed"] is False and result.result["accepted"] is False
    assert not effects.calls and not backend._acquisition_store._records
    assert states, "fresh backend private snapshot was not constructed"
    assert states[-1]["_amlas_approval_policy"].mode == ApprovalMode.ASK


@pytest.mark.parametrize("name", ["acquire_huggingface_dataset", "get_dataset_acquisition"])
@pytest.mark.parametrize("override", ["tool_context", "_amlas_acquisition_store", "_amlas_approval_policy"])
def test_direct_acquisition_rejects_caller_private_context(direct, name, override):
    backend, effects, states = direct
    args = dict(PIN) if name == "acquire_huggingface_dataset" else {"request_ref": "acq_" + "a" * 32}
    private = SimpleNamespace(
        invocation_state={
            "_amlas_acquisition_store": backend._acquisition_store,
            "_amlas_approval_policy": allowed(),
        }
    )
    args[override] = private if override == "tool_context" else backend._acquisition_store
    result = backend.execute_tool_use(name, args, confirmed=True)
    assert result.status == "error", result
    assert not states and not effects.calls and not effects.observed


def test_acquisition_names_keep_inline_classification_and_closed_public_schemas(direct):
    backend, effects, _states = direct
    assert not {"acquire_huggingface_dataset", "get_dataset_acquisition"} & LONG_RUNNING_TOOLS

    class NoBackground:
        def submit(self, *args, **kwargs):
            raise AssertionError("acquisition must retain inline opaque submission")

    backend.job_manager = NoBackground()
    result = backend.execute_tool_use("acquire_huggingface_dataset", dict(PIN), confirmed=True)
    assert result.status == "executed" and result.result["accepted"] is True
    assert len(effects.calls) == 1
    assert set(
        huggingface_data.acquire_huggingface_dataset.tool_spec["inputSchema"]["json"]["properties"]
    ) == {
        "repo_id",
        "revision",
        "filename",
        "output_dir",
        "plan_only",
    }
    assert set(huggingface_data.get_dataset_acquisition.tool_spec["inputSchema"]["json"]["properties"]) == {
        "request_ref",
    }
