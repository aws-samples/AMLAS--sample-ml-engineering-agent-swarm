"""The project-wide switch disables checks without inventing safety evidence."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from amlas.backends.adapters import governance
from amlas.config import guardrails
from amlas.config.settings import Settings
from amlas.tools import governance_neutral
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.input_admission import InputAdmissionChecker, InputAdmissionConfig


@pytest.fixture(autouse=True)
def isolated_guardrail_environment(monkeypatch):
    for name in (
        "AMLAS_GUARDRAIL_ENABLED",
        "AMLAS_GUARDRAIL_ID",
        "AMLAS_GUARDRAIL_VERSION",
        "AGENTICML_GUARDRAIL_ENABLED",
        "AGENTICML_GUARDRAIL_ID",
        "AGENTICML_GUARDRAIL_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


def assert_disabled(result):
    assert result["action"] == "DISABLED"
    assert result["enabled"] is False
    assert result["passed"] is False
    assert result["violations"] == []


def test_global_predicate_is_independent_of_provisioned_guard_id(monkeypatch):
    assert guardrails.guardrail_checks_enabled() is True
    assert guardrails.guardrail_enabled() is False
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    assert guardrails.guardrail_checks_enabled() is False
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", "ordinaryselected")
    assert guardrails.guardrail_checks_enabled() is False
    assert guardrails.guardrail_model_kwargs() == {}
    assert guardrails.converse_guardrail_config() is None


def test_disabled_inprocess_skips_all_classification_even_explicit_policy(monkeypatch):
    backend = governance.InProcessGuardrailBackend()
    detect_pii = Mock(side_effect=AssertionError("PII evaluation must not run"))
    detect_injection = Mock(side_effect=AssertionError("classification must not run"))
    backend._redactor = SimpleNamespace(detect=detect_pii)
    monkeypatch.setattr(governance, "detect_prompt_injection", detect_injection)
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    assert_disabled(backend.check(
        text="ordinary neutral fixture",
        policy={"blocked_terms": ["ordinary"], "detect_pii": True, "detect_injection": True},
        untrusted=True,
    ))
    detect_pii.assert_not_called()
    detect_injection.assert_not_called()


def test_disabled_bedrock_skips_client_even_with_explicit_guard_policy(monkeypatch):
    client = SimpleNamespace(apply_guardrail=Mock(return_value={"action": "NONE"}))
    backend = governance.BedrockGuardrailBackend(
        region="us-west-2", guardrail_id="ordinaryselected", bedrock_client=client,
    )
    make_client = Mock(return_value=client)
    monkeypatch.setattr(backend, "_client", make_client)
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    assert_disabled(backend.check(
        text="ordinary neutral fixture",
        policy={"guardrail_id": "anotherselected", "guardrail_version": "DRAFT"},
    ))
    make_client.assert_not_called()
    client.apply_guardrail.assert_not_called()


def test_disabled_tool_skips_registry_and_reports_no_safety_pass(monkeypatch):
    resolve = Mock(side_effect=AssertionError("registry resolution must not run"))
    monkeypatch.setattr(governance_neutral.REGISTRY, "resolve", resolve)
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    assert_disabled(governance_neutral.run_guardrail_check(
        text="ordinary neutral fixture", policy={"guardrail_id": "ordinaryselected"},
    ))
    resolve.assert_not_called()


def test_existing_idless_inprocess_enabled_behavior_is_unchanged():
    backend = governance.InProcessGuardrailBackend()
    assert backend.check(text="ordinary neutral fixture", policy={"detect_pii": False}) == {
        "action": "NONE", "passed": True, "violations": [], "backend": "inprocess",
    }
    assert guardrails.guardrail_model_kwargs() == {}
    assert guardrails.converse_guardrail_config() is None


def test_existing_explicit_bedrock_enabled_behavior_is_unchanged():
    client = SimpleNamespace(apply_guardrail=Mock(return_value={"action": "NONE", "assessments": []}))
    backend = governance.BedrockGuardrailBackend(
        region="us-west-2", guardrail_id="ordinaryselected", bedrock_client=client,
    )
    assert backend.check(text="ordinary neutral fixture") == {
        "action": "NONE", "passed": True, "assessments": [], "backend": "bedrock",
    }
    client.apply_guardrail.assert_called_once_with(
        guardrailIdentifier="ordinaryselected", guardrailVersion="DRAFT", source="OUTPUT",
        content=[{"text": {"text": "ordinary neutral fixture"}}],
    )


def test_switch_is_checked_at_call_time_on_existing_backend(monkeypatch):
    backend = governance.InProcessGuardrailBackend()
    assert backend.check(text="ordinary neutral fixture", policy={"detect_pii": False})["passed"] is True
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    assert_disabled(backend.check(text="ordinary neutral fixture"))
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "true")
    assert backend.check(text="ordinary neutral fixture", policy={"detect_pii": False})["passed"] is True


def test_disabled_tui_with_configured_id_has_zero_guard_calls(monkeypatch):
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", "ordinaryselected")
    monkeypatch.setenv("AMLAS_GUARDRAIL_VERSION", "DRAFT")
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    backend = binding.ToolBindingBackend(Settings(), model=object())
    check = Mock(side_effect=AssertionError("INPUT guard check must not run"))
    monkeypatch.setattr(InputAdmissionChecker, "check", check)
    assert backend._admit_input("ordinary neutral fixture", "OWNER_PROMPT") == "ordinary neutral fixture"
    assert backend._input_checker is None
    assert backend._input_admission_data() is None
    assert guardrails.guardrail_model_kwargs() == {}
    assert guardrails.converse_guardrail_config() is None
    check.assert_not_called()


def test_disabling_does_not_reopen_original_native_hold(monkeypatch):
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", "ordinaryselected")
    monkeypatch.setenv("AMLAS_GUARDRAIL_VERSION", "DRAFT")
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "true")
    backend = binding.ToolBindingBackend(Settings(), model=object())
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
        "assessments": [{"contentPolicy": {"filters": [{
            "type": "MISCONDUCT", "confidence": "HIGH", "filterStrength": "HIGH",
            "action": "BLOCKED", "detected": True,
        }]}}],
    }
    apply = Mock(return_value=response)
    checker = InputAdmissionChecker(
        InputAdmissionConfig("ordinaryselected", "DRAFT", "us-west-2"),
        client=SimpleNamespace(apply_guardrail=apply), hmac_key=b"n" * 32,
    )
    selected = backend._selected_input_checker
    monkeypatch.setattr(backend, "_selected_input_checker", lambda: checker)
    with pytest.raises(binding._InputAdmissionHeldError):
        backend._admit_input("ordinary neutral fixture", "OWNER_PROMPT")
    assert backend._input_hold is True
    monkeypatch.setattr(backend, "_selected_input_checker", selected)
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    with pytest.raises(binding._InputAdmissionHeldError):
        backend._admit_input("ordinary neutral fixture", "OWNER_PROMPT")
    assert backend._input_hold is True
    assert backend._input_admission_data()["status"] == "held"
    apply.assert_called_once()
