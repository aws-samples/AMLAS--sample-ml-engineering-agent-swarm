"""Normal flow must consume the dedicated fixed managed port before any SDK."""

from __future__ import annotations

import pytest
from test_model_acquisition_integration import configured

from amlas.backends.registry import AdapterUnavailableError
from amlas.flows import model_weight_acquisition_flow as flow


def test_normal_model_flow_unknown_selector_refuses_before_sdk(monkeypatch):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("unconfigured SDK"))
    monkeypatch.setenv("MODEL_WEIGHT_ACQUISITION_BACKEND", "local")
    with pytest.raises(AdapterUnavailableError):
        flow._resolve_backend()


def test_normal_model_flow_absent_tuple_refuses_before_sdk(monkeypatch):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("unconfigured SDK"))
    for suffix in ("IMAGE_URI", "SOURCE_COMMIT", "ADMISSION_DIGEST"):
        monkeypatch.delenv("AMLAS_MODEL_WEIGHT_ACQUISITION_" + suffix, raising=False)
    monkeypatch.delenv("MODEL_WEIGHT_ACQUISITION_BACKEND", raising=False)
    with pytest.raises(ValueError, match="unconfigured"):
        flow._resolve_backend()


def test_normal_model_flow_resolves_only_configured_managed_port(monkeypatch):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("constructor SDK"))
    for name, value in configured().items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MODEL_WEIGHT_ACQUISITION_BACKEND", raising=False)
    from amlas.backends.ports import ModelWeightAcquisitionBackend

    backend = flow._resolve_backend()
    assert isinstance(backend, ModelWeightAcquisitionBackend)
    assert backend.backend_name == "model_processing"
