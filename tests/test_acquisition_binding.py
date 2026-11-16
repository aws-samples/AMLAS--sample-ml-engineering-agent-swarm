"""Typed caller authority and opaque conversation references guard acquisition."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from amlas.backends.types import JobHandle, JobState, JobStatus
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy, request_approval
from amlas.flows import dataset_acquisition_flow as flow
from amlas.flows._util import raw
from amlas.tools import huggingface_data

PIN = {"repo_id": "fixture/coding", "revision": "a" * 40, "filename": "train/data.parquet"}
ACTION = "data_collection_flow"


class RecordingBackend:
    backend_name = "processing"

    def __init__(self):
        self.calls = []
        self.state = JobState.RUNNING
        self.lock = threading.Lock()

    def submit(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            number = len(self.calls)
        return JobHandle(
            id=f"fixture-job-{number}", backend=self.backend_name, arn=f"fixture-native-arn-{number}",
            extra={"accepted": True, "acquired": False, "native_status": "InProgress"},
        )

    def status(self, handle):
        return JobStatus(self.state, raw={"ProcessingJobStatus": "InProgress"})

    def artifacts(self, handle):
        acquired = self.state == JobState.SUCCEEDED
        return {
            "success": acquired, "acquired": acquired, "status": self.state.value,
            "job_name": handle.id, "job_arn": handle.arn, "provider": self.backend_name,
            "local_path": None,
            **({"artifact_uri": "s3://fixture-bucket/data.parquet", "format": "parquet"} if acquired else {}),
        }


@pytest.fixture
def backend(monkeypatch):
    value = RecordingBackend()
    monkeypatch.setattr(
        flow, "_resolve_backend",
        lambda: SimpleNamespace(adapter=value, adapter_name=value.backend_name), raising=False,
    )
    # Keep the old broken direct wrapper from performing network I/O in red.
    monkeypatch.setattr(
        huggingface_data, "acquire_huggingface_dataset_local",
        lambda *args, **kwargs: {"legacy_local_download": True}, raising=False,
    )
    return value


def context(store=None, policy=None):
    return SimpleNamespace(invocation_state={
        "_amlas_acquisition_store": store,
        "_amlas_approval_policy": policy,
    })


def allowed():
    return ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=[ACTION])


@pytest.mark.parametrize("policy", [None, {"mode": "auto"}, ApprovalPolicy()])
def test_missing_malformed_default_ask_denies_before_backend(backend, policy):
    result = raw(huggingface_data.acquire_huggingface_dataset)(
        **PIN, tool_context=context(SessionAcquisitionStore(), policy),
    )
    assert result["status"] == "denied"
    assert result["approved"] is False and result["accepted"] is False
    assert backend.calls == []


def test_typed_policy_with_missing_or_dict_store_denies_before_backend(backend):
    for store in (None, {"runtime_session_id": "model-session"}):
        result = raw(huggingface_data.acquire_huggingface_dataset)(
            **PIN, tool_context=context(store, allowed()),
        )
        assert result["status"] == "denied"
    assert backend.calls == []


def test_preview_has_actual_acquisition_steps_and_no_effect_or_quote(backend):
    result = raw(huggingface_data.acquire_huggingface_dataset)(**PIN, plan_only=True)
    assert result["status"] == "planned" and result["executed"] is False
    assert result["plan"]["estimated_cost"] is None
    assert [step["step"] for step in result["plan"]["steps"]] == [
        "submit_pinned_acquisition", "observe_native_job", "verify_acquisition_artifacts",
    ]
    assert "labeling" not in str(result["plan"]).lower()
    assert backend.calls == []


def test_approved_submit_uses_hidden_session_not_model_resources(backend):
    store = SessionAcquisitionStore()
    result = raw(huggingface_data.acquire_huggingface_dataset)(
        **PIN, tool_context=context(store, allowed()),
    )
    assert result["status"] == "submitted"
    assert result["accepted"] is True and result["acquired"] is False
    assert "request_ref" in result and "handle" not in result
    assert backend.calls == [{**PIN, "resources": {"runtime_session_id": store.runtime_session_id}}]
    assert store.runtime_session_id


def test_opaque_status_refs_are_isolated_and_report_verified_acquisition(backend):
    own, foreign = SessionAcquisitionStore(), SessionAcquisitionStore()
    result = flow.run_dataset_acquisition_flow(**PIN, store=own, policy=allowed())
    assert own.runtime_session_id != foreign.runtime_session_id
    running = own.observe(result["request_ref"])
    assert running["accepted"] is True and running["acquired"] is False
    assert foreign.observe(result["request_ref"])["status"] == "unknown_reference"
    backend.state = JobState.SUCCEEDED
    completed = own.observe(result["request_ref"])
    assert completed["acquired"] is True
    assert completed["artifact_uri"] == "s3://fixture-bucket/data.parquet"


def test_status_tool_uses_only_opaque_ref_and_hidden_typed_store(backend):
    store = SessionAcquisitionStore()
    result = flow.run_dataset_acquisition_flow(**PIN, store=store, policy=allowed())
    status_tool = huggingface_data.get_dataset_acquisition
    schema = status_tool.tool_spec["inputSchema"]["json"]
    assert set(schema["properties"]) == {"request_ref"}
    read = raw(status_tool)(result["request_ref"], tool_context=context(store))
    assert read["accepted"] is True
    assert raw(status_tool)(result["request_ref"])["status"] == "unknown_reference"


def test_model_authority_and_resource_parameters_are_not_exposed(backend):
    schema = huggingface_data.acquire_huggingface_dataset.tool_spec["inputSchema"]["json"]
    assert set(schema["properties"]) == {"repo_id", "revision", "filename", "output_dir", "plan_only"}
    for injected in ({"policy": {"mode": "auto"}}, {"resources": {"runtime_session_id": "foreign"}}):
        with pytest.raises(TypeError):
            raw(huggingface_data.acquire_huggingface_dataset)(**PIN, **injected)
    assert backend.calls == []


def test_output_directory_is_only_available_for_selected_local_adapter(backend):
    result = flow.run_dataset_acquisition_flow(
        **PIN, output_dir="/workspace/data", store=SessionAcquisitionStore(), policy=allowed(),
    )
    assert result["status"] == "failed" and result["accepted"] is False
    assert backend.calls == []
    backend.backend_name = "local"
    local = flow.run_dataset_acquisition_flow(
        **PIN, output_dir="/workspace/data", store=SessionAcquisitionStore(), policy=allowed(),
    )
    assert local["accepted"] is True
    assert backend.calls[-1]["resources"]["output_dir"] == "/workspace/data"


def test_two_conversations_and_concurrent_refs_share_no_job_authority(backend):
    left, right = SessionAcquisitionStore(), SessionAcquisitionStore()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(flow.run_dataset_acquisition_flow, **PIN, store=left, policy=allowed())
        second = executor.submit(flow.run_dataset_acquisition_flow, **PIN, store=right, policy=allowed())
        a, b = first.result(timeout=5), second.result(timeout=5)
    assert a["request_ref"] != b["request_ref"]
    assert left.observe(b["request_ref"])["status"] == right.observe(a["request_ref"])["status"] == "unknown_reference"
    assert {call["resources"]["runtime_session_id"] for call in backend.calls} == {
        left.runtime_session_id, right.runtime_session_id,
    }


@pytest.mark.parametrize("mode", [ApprovalMode.AUTO, ApprovalMode.ALLOWLIST])
def test_unknown_cost_denies_capped_automatic_spend_and_uncapped_can_proceed(mode):
    policy = ApprovalPolicy(mode=mode, allowlist=[ACTION], max_auto_cost=10.0)
    result = request_approval(action=ACTION, cost_estimate=None, policy=policy)
    assert result["decision"] == "denied" and result["cost_estimate"] is None
    policy.max_auto_cost = None
    assert request_approval(action=ACTION, cost_estimate=None, policy=policy)["decision"] == "approved"


@pytest.mark.parametrize("quote", [math.nan, math.inf, -1.0, True])
def test_invalid_cost_quotes_cannot_authorize_spend(quote):
    result = request_approval(action=ACTION, cost_estimate=quote, policy=ApprovalPolicy(mode=ApprovalMode.AUTO))
    assert result["decision"] == "denied"


def test_read_only_unknown_cost_and_valid_float_quotes_keep_existing_behavior():
    assert request_approval(action="get_dataset_acquisition", cost_estimate=None)["decision"] == "approved"
    policy = ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=[ACTION], max_auto_cost=2.0)
    assert request_approval(action=ACTION, cost_estimate=1.0, policy=policy)["decision"] == "approved"
    assert request_approval(action=ACTION, cost_estimate=3.0, policy=policy)["decision"] == "denied"


def test_approval_hook_preserves_explicit_unknown_cost():
    from amlas.backends.hooks import HookContext
    from amlas.control_plane.approvals import approval_gate_hook

    policy = ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=10.0)
    verdict = approval_gate_hook(policy).action(HookContext(tool=ACTION, args={"cost_estimate": None}))
    assert verdict.allowed is False
    assert verdict.detail["approval"]["cost_estimate"] is None


def test_backend_retains_store_but_transports_fresh_policy_each_turn(monkeypatch):
    from amlas.config.settings import Settings
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    calls = []

    class RecordingAgent:
        def __call__(self, prompt, **kwargs):
            calls.append(kwargs["invocation_state"])
            return SimpleNamespace(message={"content": [{"text": "done"}]})

    backend = ToolBindingBackend(Settings(), approval_policy=allowed())
    monkeypatch.setattr(ToolBindingBackend, "is_tool_capable", property(lambda _: True))
    monkeypatch.setattr(backend, "build_agent", lambda **kwargs: RecordingAgent())
    backend.chat_with_tools("start")
    backend.approval_policy = None
    backend.chat_with_tools("poll")
    assert isinstance(calls[0]["_amlas_acquisition_store"], SessionAcquisitionStore)
    assert calls[0]["_amlas_acquisition_store"] is calls[1]["_amlas_acquisition_store"]
    assert calls[0]["_amlas_approval_policy"].mode == ApprovalMode.ALLOWLIST
    assert calls[1]["_amlas_approval_policy"].mode == ApprovalMode.ASK
    assert calls[1]["_amlas_approval_policy"].allowlist == []
