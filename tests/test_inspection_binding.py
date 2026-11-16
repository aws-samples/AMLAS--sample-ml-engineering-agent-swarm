"""Opaque inspection authority boundary; no grant/resource JSON in model schema."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from amlas.backends.types import JobHandle, JobState, JobStatus
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.approvals import ApprovalPolicy


def tools() -> Any:
    try:
        return importlib.import_module("amlas.tools.dataset_inspection")
    except ModuleNotFoundError:
        pytest.fail("missing-symbol: opaque dataset inspection tools")


def test_tool_schema_excludes_authority_and_native_controls() -> None:
    module = tools()
    properties = module.inspect_acquired_dataset.tool_spec["inputSchema"]["json"]["properties"]
    assert set(properties) == {"acquisition_ref", "plan_only", "target_column"}
    poll = module.get_dataset_inspection.tool_spec["inputSchema"]["json"]["properties"]
    assert set(poll) == {"inspection_ref"}


def test_missing_context_and_foreign_references_do_not_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    module = tools()
    flow = importlib.import_module("amlas.flows.dataset_inspection_flow")
    monkeypatch.setattr(flow, "_resolve_backend", lambda: pytest.fail("unauthorized dispatch"))
    assert module.inspect_acquired_dataset("acq_" + "a" * 32)["status"] == "denied"
    store = SessionAcquisitionStore()
    context = SimpleNamespace(invocation_state={"_amlas_acquisition_store": store})
    assert module.inspect_acquired_dataset("acq_" + "a" * 32, tool_context=context)["success"] is False
    assert (
        module.get_dataset_inspection("inspect_" + "b" * 32, tool_context=context)["status"]
        == "unknown_reference"
    )


def test_private_snapshot_uses_fresh_verified_version_and_original_native_provenance() -> None:
    from test_managed_dataset_inspection import source

    store = SessionAcquisitionStore()
    item = source()

    class Acquisition:
        backend_name = "processing"

        def submit(self, **kwargs: Any) -> Any:
            raise AssertionError("not called")

        def status(self, handle: Any) -> JobStatus:
            return JobStatus(JobState.SUCCEEDED)

        def artifacts(self, handle: Any) -> dict[str, Any]:
            return {
                "success": True,
                "acquired": True,
                "status": "SUCCEEDED",
                "artifact_ref": {
                    "uri": item.artifact_ref.uri,
                    "version_id": item.artifact_ref.version_id,
                    "sha256": item.artifact_ref.sha256,
                    "byte_count": item.artifact_ref.byte_count,
                    "format": item.artifact_ref.format,
                },
            }

    store.runtime_session_id = item.runtime_session_id
    handle = JobHandle(
        id="acquisition",
        backend="processing",
        arn=item.acquisition_job_arn,
        extra={
            "account_id": item.account_id,
            "region": item.region,
            "bucket": item.bucket,
            "runtime_session_id": item.runtime_session_id,
            "deployment_id": item.deployment_id,
            "deployment_prefix": item.deployment_prefix,
        },
    )
    reference = store.add(Acquisition(), handle)
    method = getattr(store, "_verified_inspection_source", None)
    assert callable(method), "missing-symbol: private typed source accessor"
    snapshot = method(reference)
    assert snapshot == item
    assert not any(isinstance(value, ApprovalPolicy) for value in vars(store).values())


def test_preview_is_side_effect_free_and_uses_unknown_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    module = tools()
    flow = importlib.import_module("amlas.flows.dataset_inspection_flow")
    monkeypatch.setattr(flow, "_resolve_backend", lambda: pytest.fail("preview submitted"))
    result = module.inspect_acquired_dataset("acq_" + "a" * 32, plan_only=True)
    assert result["status"] == "planned" and result["executed"] is False
    assert result["plan"]["cost_estimate"] is None


def test_defaults_and_registration_preserve_managed_only_inspection() -> None:
    module = tools()
    from amlas.backends.ports import DataInspectionBackend, port_key
    from amlas.backends.registry import REGISTRY
    from amlas.tui.chat.agent_tools import READ_ONLY_TOOLS, STATE_CHANGING_TOOLS, get_safe_llm_tools

    assert port_key(DataInspectionBackend) == "DATA_INSPECTION_BACKEND"
    assert REGISTRY.resolve(DataInspectionBackend).adapter_name == "processing"
    names = {getattr(tool, "tool_name", "") for tool in get_safe_llm_tools()}
    assert {"inspect_acquired_dataset", "get_dataset_inspection"} <= names
    assert "inspect_acquired_dataset" in STATE_CHANGING_TOOLS
    assert "get_dataset_inspection" in READ_ONLY_TOOLS
    assert (
        module.inspect_acquired_dataset.tool_spec["inputSchema"]["json"]["properties"]["plan_only"]["default"]
        is False
    )


def test_capped_unknown_cost_denies_and_uncapped_authority_retains_opaque_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_managed_dataset_inspection import source

    from amlas.control_plane.approvals import ApprovalMode

    module = tools()
    flow = importlib.import_module("amlas.flows.dataset_inspection_flow")
    store = SessionAcquisitionStore()
    item = source()
    calls: list[Any] = []

    class Inspection:
        backend_name = "processing"

        def submit(self, *, source: Any, target_column: Any = None) -> JobHandle:
            assert source == item
            calls.append(source)
            return JobHandle(
                id="inspection", backend="processing", arn="fixture-native-arn", extra={"accepted": True}
            )

        def status(self, handle: Any) -> JobStatus:
            return JobStatus(JobState.RUNNING)

        def report(self, handle: Any) -> dict[str, Any]:
            return {
                "success": False,
                "accepted": True,
                "inspected": False,
                "status": "running",
                "prepared": False,
                "trained": False,
            }

    monkeypatch.setattr(store, "_verified_inspection_source", lambda ref: item)
    monkeypatch.setattr(
        flow, "_resolve_backend", lambda: SimpleNamespace(adapter_name="processing", adapter=Inspection())
    )
    context = SimpleNamespace(
        invocation_state={
            "_amlas_acquisition_store": store,
            "_amlas_approval_policy": ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=10),
        }
    )
    assert module.inspect_acquired_dataset("acq_" + "a" * 32, tool_context=context)["status"] == "denied"
    assert calls == []
    context.invocation_state["_amlas_approval_policy"] = ApprovalPolicy(
        mode=ApprovalMode.AUTO, max_auto_cost=None
    )
    accepted = module.inspect_acquired_dataset("acq_" + "a" * 32, tool_context=context)
    assert accepted["status"] == "submitted" and accepted["accepted"] is True
    assert accepted["inspection_ref"].startswith("inspect_") and len(calls) == 1
    assert (
        module.get_dataset_inspection(accepted["inspection_ref"], tool_context=context)["status"] == "running"
    )
    foreign = SimpleNamespace(invocation_state={"_amlas_acquisition_store": SessionAcquisitionStore()})
    assert (
        module.get_dataset_inspection(accepted["inspection_ref"], tool_context=foreign)["status"]
        == "unknown_reference"
    )
    assert len(calls) == 1
