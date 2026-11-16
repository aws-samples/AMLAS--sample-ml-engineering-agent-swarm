"""Proposed additive reuse API REDs; strict neutral fixtures, no native/model bytes."""

from __future__ import annotations

import inspect
from copy import deepcopy

import model_acquisition_fixtures as f
import pytest

from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.flows import model_weight_acquisition_flow as flow

PACK_REF = "model_pack_" + "a" * 64


def _require_additive_flow_signature():
    parameters = inspect.signature(flow.run_model_weight_acquisition_flow).parameters
    assert {"mode", "pack_ref"} <= set(parameters), "missing additive reuse_verified mode/pack_ref contract"


@pytest.fixture(autouse=True)
def namespace(monkeypatch):
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")


def test_existing_model_tool_exposes_only_additive_mode_and_opaque_pack_selector():
    from amlas.tools import model_weight_tools

    schema = model_weight_tools.acquire_model_weights.tool_spec["inputSchema"]["json"]["properties"]
    assert {"mode", "pack_ref"} <= set(schema)
    assert not set(schema) & {
        "producer_session_id",
        "native_handle",
        "job_arn",
        "bucket",
        "version_id",
        "policy",
        "store",
        "role_arn",
    }
    assert model_weight_tools.list_model_weight_catalog.tool_spec["inputSchema"]["json"]["properties"] == {}


def test_reuse_preview_has_no_pack_reads_copy_or_fetch_resolution(monkeypatch):
    _require_additive_flow_signature()
    monkeypatch.setattr(flow, "_resolve_backend", lambda: pytest.fail("reuse preview resolved fetch"))
    result = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG,
        catalog_version=f.VERSION,
        mode="reuse_verified",
        pack_ref=PACK_REF,
        plan_only=True,
    )
    assert result["status"] == "planned"
    assert result["executed"] is False and result["acquired"] is False and result["trained"] is False


def test_unknown_configured_pack_does_not_fallback_to_hub_fetch(monkeypatch):
    _require_additive_flow_signature()
    monkeypatch.setattr(flow, "_resolve_backend", lambda: pytest.fail("reuse fell through to fetch"))
    store = SessionModelAcquisitionStore(runtime_session_id="new-model-consumer")
    result = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG,
        catalog_version=f.VERSION,
        mode="reuse_verified",
        pack_ref=PACK_REF,
        policy=ApprovalPolicy(mode=ApprovalMode.AUTO),
        store=store,
    )
    assert result["success"] is False and result["acquired"] is False
    assert result["executed"] is False and result["status"] in {
        "reuse_unconfigured",
        "unknown_pack_reference",
    }
    assert not store._records and not store._catalogs


def test_missing_policy_denies_reuse_before_fetch_or_copy(monkeypatch):
    _require_additive_flow_signature()
    monkeypatch.setattr(flow, "_resolve_backend", lambda: pytest.fail("denied reuse resolved fetch"))
    result = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG,
        catalog_version=f.VERSION,
        mode="reuse_verified",
        pack_ref=PACK_REF,
        store=SessionModelAcquisitionStore(runtime_session_id="new-model-consumer"),
    )
    assert result["status"] == "denied" and result["executed"] is False


def test_foreign_original_handle_still_refuses_before_any_artifact_read(monkeypatch):
    from test_model_weight_acquisition_binding import modules

    _, _, backend = modules()
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(backend, sm, s3)
    producer = SessionModelAcquisitionStore(runtime_session_id=f.SESSION)
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    original_ref = producer.add(adapter, handle)
    original_extra = deepcopy(handle.extra)
    consumer = SessionModelAcquisitionStore(runtime_session_id="new-model-consumer")
    consumer._records[original_ref] = producer._records[original_ref]  # negative corruption fixture only
    monkeypatch.setattr(adapter, "artifacts", lambda *args: pytest.fail("foreign original reached read"))
    with pytest.raises(ValueError, match="foreign original"):
        consumer._verified_sft_model_snapshot(original_ref)
    assert handle.extra == original_extra
    assert producer._records[original_ref].handle is handle
    assert len(sm.requests) == 1
