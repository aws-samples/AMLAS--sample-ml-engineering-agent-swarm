"""Neutral source exposure/attempt facts, not native qualification."""

import dataclasses

import pytest
from test_model_campaign_bridge import bridge_fixture


def test_binding_exposure_requires_current_original_ref_and_final_boundary(monkeypatch):
    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    assert callable(getattr(binding, "require_current_exposure", None)), (
        "missing-symbol: require_current_exposure"
    )
    result = binding.submit_original("qwen3_5_9b", "1", x.policy)
    binding.require_current_exposure(result.reference)
    x.config.current = dataclasses.replace(
        x.config.current, execution_role_arn=x.config.current.execution_role_arn + "-different"
    )
    with pytest.raises(ValueError):
        binding.require_current_exposure(result.reference)
    assert len(x.native.requests) == 1


def test_post_native_revocation_is_declared_unknown_and_withholds_original_ids(monkeypatch):
    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    create = x.native.create_processing_job

    def revoked_after_create(**kwargs):
        response = create(**kwargs)
        x.active["current"] = False
        return response

    x.native.create_processing_job = revoked_after_create
    try:
        binding.submit_original("qwen3_5_9b", "1", x.policy)
    except Exception as error:
        assert type(error).__name__ == "ModelSubmissionAttemptError"
        assert error.dispatch_attempted is True
        assert error.currently_qualified is False and error.result is None
        assert "model-acq" not in str(error)
    else:
        pytest.fail("post-native revocation must refuse qualified exposure")
    assert len(x.native.requests) == 1


def test_outcome_write_failure_retains_declared_known_attempt_and_original_data(monkeypatch):
    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)

    def fail_outcome(*args, **kwargs):
        raise TimeoutError("neutral outcome acknowledgement unresolved")

    binding.journal.record_outcome = fail_outcome
    try:
        binding.submit_original("qwen3_5_9b", "1", x.policy)
    except Exception as error:
        assert type(error).__name__ == "ModelSubmissionAttemptError"
        assert error.dispatch_attempted is True and error.currently_qualified is True
        assert type(error.result) is x.m.ModelSubmissionResult
        assert error.result.executed is True and error.result.disposition == "UNKNOWN"
        assert error.result.handle.id == x.native.requests[0]["ProcessingJobName"]
    else:
        pytest.fail("unrecorded outcome must not masquerade as a completed submission")
    assert len(x.native.requests) == 1


def test_final_runtime_callback_cannot_hide_revoked_model_boundary(monkeypatch):
    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    original = binding._basic
    seen = [0]

    def final_runtime_leg():
        original()
        seen[0] += 1
        if seen[0] == 2:
            x.active["current"] = False

    binding._basic = final_runtime_leg
    with pytest.raises(ValueError):
        binding._callback()
    assert not x.native.requests


def test_known_private_pre_create_refusal_is_not_a_dispatch_attempt(monkeypatch):
    from amlas.control_plane.approvals import ApprovalMode

    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    x.hint.after_publish = lambda: setattr(x.policy, "mode", ApprovalMode.ASK)
    with pytest.raises(x.m.ModelSubmissionAttemptError) as caught:
        binding.submit_original("qwen3_5_9b", "1", x.policy)
    assert caught.value.dispatch_attempted is False and caught.value.result is None
    assert caught.value.currently_qualified is True and not x.native.requests


def test_opaque_adapter_exception_cannot_mint_known_dispatch_fact(monkeypatch):
    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)

    def opaque(*args, **kwargs):
        raise RuntimeError("neutral opaque source call uncertainty")

    monkeypatch.setattr(x.adapter, "submit_prepared", opaque)
    with pytest.raises(x.m.ModelSubmissionAttemptError) as caught:
        binding.submit_original("qwen3_5_9b", "1", x.policy)
    assert caught.value.dispatch_attempted is None and caught.value.result is None
    assert not x.native.requests


def test_closed_attempt_error_forbids_unqualified_result_or_nonprimitive_flags(monkeypatch):
    x = bridge_fixture(monkeypatch)
    for attempted, qualified in [("ordinary", True), (True, "ordinary")]:
        with pytest.raises(ValueError):
            x.m.ModelSubmissionAttemptError(dispatch_attempted=attempted, currently_qualified=qualified)


@pytest.mark.parametrize(
    "name,value", [("dispatch_attempted", False), ("currently_qualified", True), ("result", object())]
)
def test_closed_attempt_error_facts_are_readonly(name, value, monkeypatch):
    x = bridge_fixture(monkeypatch)
    error = x.m.ModelSubmissionAttemptError(dispatch_attempted=True, currently_qualified=False, result=None)
    with pytest.raises(AttributeError):
        setattr(error, name, value)
