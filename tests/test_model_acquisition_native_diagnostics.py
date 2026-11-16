"""Real SDK model diagnostics over strict neutral providers; no native effects."""

from __future__ import annotations

import json

import model_acquisition_fixtures as f
import pytest
from botocore.exceptions import ClientError, ParamValidationError
from test_sdk_acquisition_binding import invoke, request

from amlas.backends.adapters import model_weight_acquisition as adapter_module
from amlas.config import externalized
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.flows import model_weight_acquisition_flow as flow
from amlas.tools import model_weight_tools
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

REQUEST_ID = "11111111-2222-3333-4444-abcdefabcdef"
PRIVATE_DETAIL = "ordinary private provider detail"
PARAMS = {"catalog_id": f.CATALOG, "catalog_version": f.VERSION}


def service_error(operation, *, code="ValidationException", request_id=REQUEST_ID, status=400):
    return ClientError(
        {
            "Error": {"Code": code, "Message": PRIVATE_DETAIL},
            "ResponseMetadata": {
                "RequestId": request_id,
                "HTTPStatusCode": status,
                "HTTPHeaders": {"fixture-detail": PRIVATE_DETAIL},
            },
        },
        operation,
    )


class DiagnosticProcessing(f.Processing):
    def __init__(self):
        super().__init__()
        self.create_error = None
        self.describe_error = None
        self.describe_calls = []

    def create_processing_job(self, **kwargs):
        # The inherited strict provider validates the installed native input shape.
        response = super().create_processing_job(**kwargs)
        if self.create_error is not None:
            raise self.create_error
        return response

    def describe_processing_job(self, **kwargs):
        self.describe_calls.append(dict(kwargs))
        if self.describe_error is not None:
            assert kwargs["ProcessingJobName"] == self.requests[-1]["ProcessingJobName"]
            raise self.describe_error
        return super().describe_processing_job(**kwargs)


@pytest.fixture
def owned(monkeypatch):
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    native = DiagnosticProcessing()
    adapter = f.configured_backend(adapter_module, native, f.S3())
    monkeypatch.setattr(flow, "_resolve_backend", lambda: adapter)
    backend = ToolBindingBackend(Settings(), approval_policy=ApprovalPolicy(mode=ApprovalMode.AUTO))
    return backend, native, adapter


def call(monkeypatch, backend, identifier, *, tool_name="acquire_model_weights", params=None):
    # Restore the ordinary constructor after the helper's preceding turn override.
    monkeypatch.setattr(backend, "build_agent", ToolBindingBackend.build_agent.__get__(backend))
    result = invoke(
        monkeypatch,
        backend,
        [request(identifier, tool_name, PARAMS if params is None else params)],
        tools=[model_weight_tools.acquire_model_weights, model_weight_tools.get_model_weight_acquisition],
    )
    assert set(result) == {identifier}  # Exact actual SDK toolUseId correlation.
    return result[identifier]


def original_record(backend, native, adapter, result):
    reference = result["request_ref"]
    store = backend._model_acquisition_store
    record = store._records[reference]
    assert record.adapter is adapter
    assert record.handle.id == result["job_name"] == native.requests[0]["ProcessingJobName"]
    assert record.handle.arn == result["job_arn"]
    assert record.handle.extra["bindings"]["runtime_session_id"] == store.runtime_session_id
    assert len(native.requests) == 1 and len(store._records) == 1
    return reference, record


def expected(phase, error_class, *, native_code="UNKNOWN", request_id=None, status=None):
    return {
        "phase": phase,
        "error_class": error_class,
        "error_code": "model_acquisition_submission_unknown"
        if phase == "create_boundary"
        else "model_acquisition_observation_unknown",
        "native_error_code": native_code,
        "request_id": request_id,
        "http_status": status,
    }


@pytest.mark.parametrize("kind", ["parameter", "service", "lost_response"])
def test_actual_sdk_submission_diagnostic_retains_exact_original(monkeypatch, owned, kind):
    backend, native, adapter = owned
    native.create_error = {
        "parameter": ParamValidationError(report=PRIVATE_DETAIL),
        "service": service_error("CreateProcessingJob"),
        "lost_response": TimeoutError(PRIVATE_DETAIL),
    }[kind]
    result = call(monkeypatch, backend, "original-submit")
    assert result["status"] == "submission_unknown" and result["accepted"] is False
    original_record(backend, native, adapter, result)
    assert PRIVATE_DETAIL not in json.dumps(result)
    diagnostic = expected("create_boundary", type(native.create_error).__name__)
    if kind == "service":
        diagnostic = expected("create_boundary", "ClientError", native_code="ValidationException",
                              request_id=REQUEST_ID, status=400)
    assert result.get("submission_diagnostic") == diagnostic


def test_actual_sdk_original_describe_diagnostic_is_exposed_without_replacement(monkeypatch, owned):
    backend, native, adapter = owned
    first = call(monkeypatch, backend, "accepted-submit")
    assert first["accepted"] is True and first["status"] == "submitted"
    reference, record = original_record(backend, native, adapter, first)
    native.describe_error = service_error("DescribeProcessingJob")
    result = call(monkeypatch, backend, "original-observe", tool_name="get_model_weight_acquisition",
                  params={"request_ref": reference})
    assert result["request_ref"] == reference and result["job_name"] == record.handle.id
    assert result["status"] == "UNKNOWN" and result["accepted"] is True
    assert len(native.requests) == 1 and len(native.describe_calls) == 1
    assert backend._model_acquisition_store._records[reference] is record
    assert PRIVATE_DETAIL not in json.dumps(result)
    assert result.get("observation_diagnostic") == expected(
        "describe", "ClientError", native_code="ValidationException", request_id=REQUEST_ID, status=400)


def test_actual_sdk_unknown_attempt_duplicate_never_replays_create(monkeypatch, owned):
    backend, native, adapter = owned
    native.create_error = TimeoutError(PRIVATE_DETAIL)
    first = call(monkeypatch, backend, "uncertain-submit")
    assert first["status"] == "submission_unknown" and first["accepted"] is False
    reference, record = original_record(backend, native, adapter, first)
    native.describe_error = service_error("DescribeProcessingJob")
    second = call(monkeypatch, backend, "duplicate-submit")
    assert second["request_ref"] == reference and second["job_name"] == record.handle.id
    assert second["accepted"] is False and len(native.requests) == 1
    assert second["status"] == "UNKNOWN" and len(native.describe_calls) == 1
    assert backend._model_acquisition_store._records[reference] is record
    assert len(backend._model_acquisition_store._records) == 1


@pytest.mark.parametrize("request_id,status", [(PRIVATE_DETAIL, True), ("a" * 129, "400")])
def test_actual_sdk_unreviewed_or_malformed_provider_metadata_is_closed(monkeypatch, owned,
                                                                       request_id, status):
    backend, native, adapter = owned
    native.create_error = service_error("CreateProcessingJob", code=PRIVATE_DETAIL,
                                        request_id=request_id, status=status)
    result = call(monkeypatch, backend, "malformed-submit")
    assert result["status"] == "submission_unknown" and result["accepted"] is False
    original_record(backend, native, adapter, result)
    assert PRIVATE_DETAIL not in json.dumps(result)
    assert "HTTPHeaders" not in json.dumps(result) and "Error" not in result
    assert result.get("submission_diagnostic") == expected("create_boundary", "ClientError")
