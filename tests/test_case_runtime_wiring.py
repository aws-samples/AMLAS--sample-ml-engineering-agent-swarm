"""Normal case startup/port/tool routes are pure and retain original private stores."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_coding_case_tools import Backend
from test_managed_coding_case_envelope import ACCOUNT, BUCKET, HEADER, IMAGE

from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.case_binding import build_case_freeze_store
from amlas.control_plane.coding_case_state import CaseFreezeConfigurationMissingError, validate_catalog
from amlas.tools.coding_cases import freeze_coding_cases, get_coding_case_freeze, list_coding_case_catalog


def env():
    return {
        "AMLAS_AWS_ACCOUNT_ID": ACCOUNT,
        "AMLAS_AWS_REGION": "us-west-2",
        "AMLAS_SESSION_BUCKET": BUCKET,
        "SAGEMAKER_EXECUTION_ROLE_ARN": "arn:aws:iam::" + ACCOUNT + ":role/owned",
        "AMLAS_CODING_CASE_FREEZE_IMAGE_URI": IMAGE,
        "AMLAS_CODING_CASE_FREEZE_SOURCE_COMMIT": "b" * 40,
        "AMLAS_CODING_CASE_FREEZE_ADMISSION_DIGEST": "a" * 64,
        "AMLAS_DEPLOYMENT_ID": HEADER["deployment_id"],
        "AMLAS_DEPLOYMENT_PREFIX": "owned",
    }


@pytest.fixture
def original(monkeypatch):
    from amlas.backends.registry import REGISTRY

    acquired = SessionAcquisitionStore()
    backend = Backend()

    def resolve(port):
        assert port.__name__ == "CodingCaseFreezeBackend"
        return SimpleNamespace(adapter=backend)

    monkeypatch.setattr(REGISTRY, "resolve", resolve)
    return acquired, backend


def test_factory_pure_constructor_reuses_original_store_and_no_sdk_ssm(original, monkeypatch):
    import boto3

    from amlas.config import externalized

    monkeypatch.setattr(boto3, "client", lambda *_a, **_kw: pytest.fail("startupSDK"))
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *_a, **_kw: pytest.fail("startupSSM"))
    acquired, backend = original
    store = build_case_freeze_store(acquired, env=env())
    assert store._acquired is acquired and store.runtime_session_id == acquired.runtime_session_id
    assert store._adapter is backend and backend.creates == backend.reads == 0
    validate_catalog(store._catalog)


def test_missing_tuple_is_unadmitted_without_source_or_native_effects(original):
    acquired, backend = original
    store = build_case_freeze_store(acquired, env={})
    with pytest.raises(CaseFreezeConfigurationMissingError):
        validate_catalog(store._catalog)
    assert backend.creates == backend.reads == 0


@pytest.mark.parametrize(
    "change",
    [
        {"AMLAS_CODING_CASE_FREEZE_SOURCE_COMMIT": ""},
        {"AMLAS_CODING_CASE_FREEZE_SOURCE_COMMIT": True},
        {"AMLAS_CODING_CASE_FREEZE_ADMISSION_DIGEST": "bad"},
        {"AMLAS_CODING_CASE_FREEZE_IMAGE_URI": "foreign@sha256:" + "a" * 64},
    ],
)
def test_selected_partial_malformed_or_foreign_tuple_refuses_at_startup(original, change):
    acquired, backend = original
    with pytest.raises(ValueError):
        build_case_freeze_store(acquired, env={**env(), **change})
    assert backend.creates == backend.reads == 0


def test_exact_shared_port_processing_only_and_lazy_registration(monkeypatch):
    from amlas.backends.adapters.coding_case_freeze import ProcessingCodingCaseFreezeBackend
    from amlas.backends.ports import ALL_PORTS, CodingCaseFreezeBackend, port_key
    from amlas.backends.registry import REGISTRY, AdapterUnavailableError

    assert (
        CodingCaseFreezeBackend in ALL_PORTS
        and port_key(CodingCaseFreezeBackend) == "CODING_CASE_FREEZE_BACKEND"
    )
    monkeypatch.delenv("CODING_CASE_FREEZE_BACKEND", raising=False)
    assert type(REGISTRY.resolve(CodingCaseFreezeBackend).adapter) is ProcessingCodingCaseFreezeBackend
    monkeypatch.setenv("CODING_CASE_FREEZE_BACKEND", "local")
    with pytest.raises(AdapterUnavailableError):
        REGISTRY.resolve(CodingCaseFreezeBackend)


def test_all_normal_safe_tools_classifications_and_hidden_schema():
    from amlas.tui.chat.agent_tools import (
        READ_ONLY_TOOLS,
        STATE_CHANGING_TOOLS,
        get_safe_llm_tools,
        tool_display_name,
    )

    names = [tool_display_name(tool) for tool in get_safe_llm_tools()]
    assert all(
        names.count(name) == 1
        for name in ("list_coding_case_catalog", "freeze_coding_cases", "get_coding_case_freeze")
    )
    assert {"list_coding_case_catalog", "get_coding_case_freeze"} <= READ_ONLY_TOOLS
    assert "freeze_coding_cases" in STATE_CHANGING_TOOLS
    for tool in (freeze_coding_cases, get_coding_case_freeze, list_coding_case_catalog):
        assert not set(tool.tool_spec["inputSchema"]["json"]["properties"]) & {
            "tool_context",
            "policy",
            "store",
            "role",
            "image_uri",
            "source_commit",
            "native_handle",
            "oracle_access",
        }


def test_preparation_factory_passes_same_original_case_store_without_startup_exclusions(
    original, monkeypatch
):
    from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
    from amlas.backends.registry import REGISTRY
    from amlas.control_plane import preparation_binding

    acquired, _backend = original
    cases = build_case_freeze_store(acquired, env=env())
    monkeypatch.setattr(
        REGISTRY, "resolve", lambda _p: SimpleNamespace(adapter=ProcessingCodingPreparationBackend())
    )
    prepared = preparation_binding.build_preparation_store(acquired, case_store=cases, env={})
    assert prepared._case_store is cases and prepared._acquired is acquired


def test_case_flow_discovery_reports_fixed_data_only_unknown_cost_spend():
    from amlas.flows import get_flow

    flow = get_flow("coding_case_freeze_flow")
    assert flow is not None and flow.backend_port == "CodingCaseFreezeBackend"
    assert flow.spend_incurring is True and flow.irreversible is False
