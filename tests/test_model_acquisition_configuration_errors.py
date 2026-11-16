"""Fixed startup diagnostics; neutral source fixtures, no native model dispatch."""

from __future__ import annotations

import json
from types import SimpleNamespace

import model_acquisition_fixtures as f
import pytest

from amlas.backends.adapters import model_weight_acquisition as adapter_module
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.flows import model_weight_acquisition_flow as flow
from amlas.tools import model_weight_tools

SELECTORS = (
    "AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI",
    "AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT",
    "AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST",
)


@pytest.fixture
def isolated(monkeypatch):
    import boto3

    from amlas.config import externalized

    for name in SELECTORS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("MODEL_WEIGHT_ACQUISITION_BACKEND", raising=False)
    monkeypatch.setenv("AMLAS_AWS_ACCOUNT_ID", f.ACCOUNT)
    monkeypatch.setenv("AMLAS_AWS_REGION", f.REGION)
    monkeypatch.setattr(boto3, "Session", lambda *a, **k: pytest.fail("unexpected SDK session"))
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *a, **k: pytest.fail("unexpected SSM"))
    monkeypatch.setattr(
        adapter_module.ProcessingModelWeightAcquisitionBackend,
        "submit",
        lambda *a, **k: pytest.fail("unexpected model submit"),
    )
    policy = ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["run_processing_job"])
    store = SessionModelAcquisitionStore(runtime_session_id=f.SESSION)
    return SimpleNamespace(
        invocation_state={"_amlas_approval_policy": policy, "_amlas_model_acquisition_store": store}
    )


def acquire(context):
    return model_weight_tools.acquire_model_weights(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, tool_context=context
    )


def test_real_tool_reports_exact_absent_runtime_selectors_before_submit(isolated):
    result = acquire(isolated)
    assert result["error_code"] == "model_acquisition_configuration_missing"
    assert result["configuration_keys"] == list(SELECTORS)
    assert result["error_type"] == "ModelAcquisitionConfigurationError"
    assert result["executed"] is False and result["accepted"] is False
    assert result["status"] == "submission_failed"
    assert "unconfigured" in result["detail"]
    assert not any(key in result for key in ("request_ref", "job_name", "job_arn"))
    assert isolated.invocation_state["_amlas_model_acquisition_store"]._records == {}


@pytest.mark.parametrize("present", range(3))
def test_partial_tuple_reports_only_missing_names_without_values(monkeypatch, isolated, present):
    values = (f.IMAGE, f.COMMIT, "a" * 64)
    monkeypatch.setenv(SELECTORS[present], values[present])
    result = acquire(isolated)
    assert result["error_code"] == "model_acquisition_configuration_missing"
    assert result["configuration_keys"] == [name for index, name in enumerate(SELECTORS) if index != present]
    diagnostic = json.dumps({key: result[key] for key in ("error_code", "configuration_keys", "detail")})
    assert values[present] not in diagnostic
    assert result["executed"] is False and result["accepted"] is False


def test_invalid_complete_tuple_reports_closed_error_without_raw_configuration(monkeypatch, isolated):
    marker = "ordinary-unqualified-value"
    for name, value in zip(SELECTORS, (f.IMAGE, f.COMMIT, marker), strict=True):
        monkeypatch.setenv(name, value)
    result = acquire(isolated)
    assert result["error_code"] == "model_acquisition_configuration_invalid"
    assert result["configuration_keys"] == list(SELECTORS)
    diagnostic = json.dumps({key: result[key] for key in ("error_code", "configuration_keys", "detail")})
    assert marker not in diagnostic and f.IMAGE not in diagnostic and f.COMMIT not in diagnostic
    assert result["executed"] is False and result["accepted"] is False


@pytest.mark.parametrize("case", ["ask", "missing_policy", "plan"])
def test_fresh_policy_and_plan_precede_configuration_diagnostics(isolated, case):
    if case == "ask":
        isolated.invocation_state["_amlas_approval_policy"] = ApprovalPolicy(mode=ApprovalMode.ASK)
    elif case == "missing_policy":
        isolated.invocation_state.pop("_amlas_approval_policy")
    result = model_weight_tools.acquire_model_weights(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, plan_only=case == "plan", tool_context=isolated
    )
    assert result["status"] == ("planned" if case == "plan" else "denied")
    assert result["executed"] is False and "error_code" not in result


def test_unrelated_adapter_exception_keeps_unknown_outcome_and_no_configuration_claim(monkeypatch, isolated):
    calls = []

    class UncertainAdapter:
        def submit(self, **kwargs):
            calls.append(kwargs)
            raise ValueError("ordinary implementation error")

    monkeypatch.setattr(flow, "_resolve_backend", lambda: UncertainAdapter())
    result = acquire(isolated)
    assert len(calls) == 1
    assert result["status"] == "outcome_unknown" and result["executed"] is None
    assert result["error_type"] == "ValueError" and "error_code" not in result
    assert "ordinary implementation error" not in json.dumps(result)


def test_declared_configuration_error_projection_is_closed_and_readonly():
    kind = getattr(adapter_module, "ModelAcquisitionConfigurationError", None)
    assert kind is not None, "missing bounded model configuration error"
    error = kind("model_acquisition_configuration_missing", SELECTORS)
    assert error.error_code == "model_acquisition_configuration_missing"
    assert error.configuration_keys == SELECTORS
    with pytest.raises(AttributeError):
        error.error_code = "changed"
    for code, keys in [
        ("arbitrary", SELECTORS),
        ("model_acquisition_configuration_missing", ("CALLER_FIELD",)),
    ]:
        with pytest.raises(ValueError):
            kind(code, keys)


def test_configuration_error_subclass_from_adapter_keeps_unknown_outcome(monkeypatch, isolated):
    class AdapterDiagnosticSubtype(adapter_module.ModelAcquisitionConfigurationError):
        pass

    class UncertainAdapter:
        def submit(self, **kwargs):
            raise AdapterDiagnosticSubtype("model_acquisition_configuration_missing", SELECTORS)

    monkeypatch.setattr(flow, "_resolve_backend", lambda: UncertainAdapter())
    result = acquire(isolated)
    assert result["status"] == "outcome_unknown" and result["executed"] is None
    assert result["error_type"] == "AdapterDiagnosticSubtype"
    assert not any(key in result for key in ("error_code", "configuration_keys", "detail", "request_ref"))
