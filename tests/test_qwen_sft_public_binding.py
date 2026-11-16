"""Public fixed SFT wiring and pure same-session startup, no AWS or model calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_qwen_sft_backend import catalog
from test_qwen_sft_binding import allowed, real_store

from amlas.backends.adapters._util import raw_tool
from amlas.control_plane import sft_binding
from amlas.control_plane.sft_state import SessionSftStore
from amlas.tools.qwen_sft import get_qwen_sft, reload_qwen_sft, submit_qwen_sft

NAMES = {"submit_qwen_sft", "get_qwen_sft", "reload_qwen_sft"}


def env():
    value = catalog()
    return {
        "AMLAS_AWS_ACCOUNT_ID": value.account_id,
        "AMLAS_AWS_REGION": value.region,
        "AMLAS_SESSION_BUCKET": value.bucket,
        "SAGEMAKER_EXECUTION_ROLE_ARN": value.execution_role_arn,
        "AMLAS_QWEN_SFT_IMAGE_URI": value.image_uri,
        "AMLAS_QWEN_SFT_SOURCE_COMMIT": value.source_commit,
        "AMLAS_QWEN_SFT_ADMISSION_DIGEST": value.admission_digest,
        "AMLAS_DEPLOYMENT_ID": value.deployment_id,
        "AMLAS_DEPLOYMENT_PREFIX": value.deployment_prefix,
    }


def factory(monkeypatch, configured=True):
    source, native, s3, model_ref, prep_ref, reads, *_ = real_store(monkeypatch)
    from amlas.backends.registry import REGISTRY

    def resolve(port):
        assert port.__name__ == "QwenSftBackend"
        return SimpleNamespace(adapter=source._adapter)

    monkeypatch.setattr(REGISTRY, "resolve", resolve)
    result = sft_binding.build_sft_store(source._models, source._prepared, env=env() if configured else {})
    return result, source, native, s3, model_ref, prep_ref, reads


def test_pure_factory_keeps_exact_original_stores_session_catalog_and_no_grants(monkeypatch):
    import boto3

    from amlas.config import externalized

    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: pytest.fail("startup SDK"))
    monkeypatch.setattr(
        externalized, "_default_ssm_getter", lambda *args, **kwargs: pytest.fail("startup SSM")
    )
    result, source, native, _s3, _m, _p, reads = factory(monkeypatch)
    assert isinstance(result, SessionSftStore)
    assert result._models is source._models and result._prepared is source._prepared
    assert result.runtime_session_id == source.runtime_session_id
    assert result._catalog == catalog() and not hasattr(result, "policy")
    assert reads == {"model": 0, "prep": 0} and native.calls == []


def test_missing_factory_catalog_reports_selector_names_before_sources_sdk(monkeypatch):
    result, _source, native, _s3, model_ref, prep_ref, reads = factory(monkeypatch, configured=False)
    answer = raw_tool(submit_qwen_sft)(
        model_request_ref=model_ref,
        preparation_request_ref=prep_ref,
        tool_context=SimpleNamespace(
            invocation_state={"_amlas_sft_store": result, "_amlas_approval_policy": allowed()}
        ),
    )
    assert answer["status"] == "configuration_missing" and answer["accepted"] is False
    assert set(answer["missing_selectors"]) == set(env())
    assert reads == {"model": 0, "prep": 0} and native.calls == []


@pytest.mark.parametrize(
    "key", ["AMLAS_QWEN_SFT_SOURCE_COMMIT", "AMLAS_DEPLOYMENT_ID", "AMLAS_AWS_ACCOUNT_ID"]
)
def test_startup_nonstring_selectors_never_coerce_or_call_custom_values(monkeypatch, key):
    result, source, *_ = factory(monkeypatch)
    selected = env()
    selected[key] = True
    with pytest.raises(ValueError):
        sft_binding.build_sft_store(source._models, source._prepared, env=selected)
    assert isinstance(result, SessionSftStore)


def test_factory_foreign_session_or_public_dict_refuses(monkeypatch):
    _result, source, *_ = factory(monkeypatch)
    from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore

    with pytest.raises(ValueError):
        sft_binding.build_sft_store(
            SessionModelAcquisitionStore(runtime_session_id="foreign-session"), source._prepared, env=env()
        )
    with pytest.raises(ValueError):
        sft_binding.build_sft_store(
            {"runtime_session_id": source.runtime_session_id}, source._prepared, env=env()
        )


def test_all_three_real_safe_tools_and_classifications_are_source_bound():
    from amlas.tui.chat.agent_tools import (
        LONG_RUNNING_TOOLS,
        READ_ONLY_TOOLS,
        STATE_CHANGING_TOOLS,
        get_safe_llm_tools,
        tool_display_name,
    )

    tools = get_safe_llm_tools()
    names = [tool_display_name(value) for value in tools]
    assert NAMES <= set(names) and all(names.count(name) == 1 for name in NAMES)
    assert "get_qwen_sft" in READ_ONLY_TOOLS
    assert {"submit_qwen_sft", "reload_qwen_sft"} <= STATE_CHANGING_TOOLS
    assert not NAMES & LONG_RUNNING_TOOLS
    for value in (submit_qwen_sft, get_qwen_sft, reload_qwen_sft):
        fields = value.tool_spec["inputSchema"]["json"]["properties"]
        assert not set(fields) & {
            "policy",
            "store",
            "tool_context",
            "role",
            "image_uri",
            "source_commit",
            "native_handle",
            "approval",
        }


def test_fixed_flow_discovery_and_recipe_plan_preserve_fit_full_quality():
    from amlas.flows import get_flow

    value = get_flow("qwen_sft_flow")
    assert value is not None and value.backend_port == "QwenSftBackend"
    assert value.spend_incurring is True and value.irreversible is False
    for recipe, steps in [("fit_v1", 32), ("full_v1", 1024)]:
        answer = raw_tool(submit_qwen_sft)(
            model_request_ref="unused", preparation_request_ref="unused", recipe_id=recipe, plan_only=True
        )
        assert answer["status"] == "planned" and answer["quality_success"] is False
        assert answer["full_training_complete"] is False and answer["plan"]["optimizer_steps"] == steps


def test_specialists_bind_fixed_tools_without_changing_existing_role_defaults():
    from amlas.subagents.declarations import FINE_TUNE, TRAINER_TUNER, tool_name_of

    for declaration in (TRAINER_TUNER, FINE_TUNE):
        assert NAMES <= {tool_name_of(value) for value in declaration.tools}
    assert TRAINER_TUNER.role == "bounded" and FINE_TUNE.role == "finetune"


def test_ml_lead_fine_tune_uses_existing_typed_sdk_delegation(monkeypatch):
    from amlas.control_plane import acquisition_state
    from amlas.leads import declarations as leads

    calls = []
    original = acquisition_state.acquisition_agent_tool

    def wrapped(agent, *, name):
        calls.append(name)
        return original(agent, name=name)

    monkeypatch.setattr(acquisition_state, "acquisition_agent_tool", wrapped)
    from test_sdk_prompt_policy import ScriptedModel

    leads.build_lead(
        leads.ML_EXECUTION_LEAD,
        name="ml_execution_lead",
        model=ScriptedModel([]),
        algo_specialist_model=ScriptedModel([]),
        trainer_tuner_model=ScriptedModel([]),
        fine_tune_model=ScriptedModel([]),
    )
    assert "trainer_tuner" in calls and "fine_tune" in calls


def test_fresh_ask_policy_and_foreign_refs_keep_zero_effects(monkeypatch):
    store, _source, native, _s3, model_ref, prep_ref, reads = factory(monkeypatch)
    from amlas.control_plane.approvals import ApprovalPolicy

    result = raw_tool(submit_qwen_sft)(
        model_request_ref=model_ref,
        preparation_request_ref=prep_ref,
        tool_context=SimpleNamespace(
            invocation_state={"_amlas_sft_store": store, "_amlas_approval_policy": ApprovalPolicy()}
        ),
    )
    assert result["status"] == "denied" and reads == {"model": 0, "prep": 0} and native.calls == []
    forged = raw_tool(submit_qwen_sft)(
        model_request_ref=model_ref,
        preparation_request_ref=prep_ref,
        tool_context=SimpleNamespace(
            invocation_state={"_amlas_sft_store": {"ready": True}, "_amlas_approval_policy": {"mode": "auto"}}
        ),
    )
    assert forged["status"] == "denied" and native.calls == []
