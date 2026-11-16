"""Offline native-state and deterministic grading proofs for baseline flow safety."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from amlas.arbiter.arbiter import grade
from amlas.backends.ports import (
    EvaluationBackend,
    ExperimentTrackingBackend,
    ModelRegistryBackend,
    TrainingBackend,
)
from amlas.backends.types import JobHandle, JobState, JobStatus, ModelRef, PackageRef
from amlas.control_plane import capability
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.flows import baseline_train_flow as flow
from amlas.spec.schema import SpecModel


class TrainingCapture:
    """Only native typed job effects are captured; no AWS/model calls occur."""

    def __init__(self):
        self.state = JobState.SUCCEEDED
        self.handle = JobHandle("existing-native-job", backend="capture", extra={"tracking_error": "accepted job"})
        self.calls = []
        self.status_error = None
        self.metrics_error = None
        self.submit_error = None

    def submit(self, **kwargs):
        self.calls.append(("submit", kwargs))
        if self.submit_error:
            raise self.submit_error
        return self.handle

    def status(self, handle):
        assert handle is self.handle
        self.calls.append(("status", handle))
        if self.status_error:
            raise self.status_error
        return JobStatus(self.state, detail="native observation")

    def metrics(self, handle):
        assert handle is self.handle
        self.calls.append(("metrics", handle))
        if self.metrics_error:
            raise self.metrics_error
        return {"loss": 0.2}

    def artifact(self, handle):
        assert handle is self.handle
        self.calls.append(("artifact", handle))
        return ModelRef("s3://fixture/model", backend="capture")


class EvaluationCapture:
    def __init__(self):
        self.calls = []
        self.report = None
        self.error = None

    def grade(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.report if self.report is not None else grade(**kwargs)


class TrackingCapture:
    def __init__(self):
        self.calls = []

    def log_run(self, **kwargs):
        self.calls.append(kwargs)
        return "tracked-run"


class RegistryCapture:
    def __init__(self):
        self.calls = []

    def register(self, **kwargs):
        self.calls.append(kwargs)
        return PackageRef("fixture-package", group=kwargs["group"])


@pytest.fixture
def effects(monkeypatch):
    captures = SimpleNamespace(
        training=TrainingCapture(), evaluation=EvaluationCapture(),
        tracking=TrackingCapture(), registry=RegistryCapture(),
    )
    ports = {
        TrainingBackend: captures.training, EvaluationBackend: captures.evaluation,
        ExperimentTrackingBackend: captures.tracking, ModelRegistryBackend: captures.registry,
    }

    def resolve(port, **kwargs):
        assert port in ports, "unexpected backend boundary"
        return SimpleNamespace(adapter=ports[port])

    monkeypatch.setattr(flow.REGISTRY, "resolve", resolve)
    return captures


@pytest.fixture
def params():
    spec = SpecModel(
        project_id="fixture", task_type="classification", target="label", primary_metric="f1_macro",
        golden_dataset="s3://fixture/eval", baseline={"type": "fixed", "f1_macro": 0.5},
        acceptance={"primary_metric_delta_vs_baseline": ">= 0.1"},
    )
    return dict(algo="sklearn", train_uri="s3://fixture/train", target="label", spec=spec,
                y_true=[0, 1, 0, 1], y_pred=[0, 1, 0, 1], register_group="fixture-group")


def assert_no_promotion(effects):
    assert effects.registry.calls == []


@pytest.mark.parametrize("state", [JobState.PENDING, JobState.RUNNING, JobState.UNKNOWN])
def test_existing_nonterminal_job_waits_without_downstream_effects(effects, params, state):
    effects.training.state = state
    result = flow.execute_baseline_train_flow(**params)
    assert result.status == "waiting"
    assert (result.job_id, result.train_backend) == (effects.training.handle.id, "capture")
    assert result.detail["state"] == state.value
    assert [name for name, _ in effects.training.calls] == ["submit", "status"]
    assert effects.evaluation.calls == effects.tracking.calls == []
    assert_no_promotion(effects)
    assert "resubmit" not in json.dumps(result.detail).lower()


@pytest.mark.parametrize("state", [JobState.FAILED, JobState.STOPPED])
def test_only_observed_terminal_failure_is_failed(effects, params, state):
    effects.training.state = state
    result = flow.execute_baseline_train_flow(**params)
    assert result.status == "failed"
    assert result.job_id == effects.training.handle.id
    assert effects.evaluation.calls == effects.tracking.calls == []
    assert_no_promotion(effects)


@pytest.mark.parametrize("missing", [("spec",), ("y_true",), ("y_pred",), ("spec", "y_true", "y_pred")])
def test_success_without_complete_grading_is_unassessed(effects, params, missing):
    for key in missing:
        params[key] = None
    result = flow.execute_baseline_train_flow(**params)
    assert result.status == "unassessed"
    assert result.job_id == effects.training.handle.id
    assert result.metrics == {"loss": 0.2}
    assert result.arbiter_report == result.registered == {}
    assert effects.evaluation.calls == effects.tracking.calls == []
    assert "artifact" not in [name for name, _ in effects.training.calls]
    assert_no_promotion(effects)


BAD_REPORTS = [
    {}, {"score": 1.0}, {"accepted": "false"}, {"accepted": "true"},
    {"passed": "false"}, {"is_accept": 1}, {"passes_acceptance": None},
    {"accepted": 0}, {"verdict": ""}, {"decision": "unknown"},
    {"accepted": True, "passed": False}, {"accepted": True, "passes_acceptance": False},
    {"accepted": True, "decision": "reject"}, {"verdict": "accept", "decision": "reject"},
    {"passes_acceptance": True, "accepted": "false"},
]


@pytest.mark.parametrize("report", BAD_REPORTS)
def test_missing_malformed_or_contradictory_decisions_cannot_accept(effects, params, report):
    effects.evaluation.report = report
    assert flow._is_accept(report) is False
    result = flow.execute_baseline_train_flow(**params)
    assert result.status == "unassessed"
    assert result.arbiter_report == report
    assert result.job_id == effects.training.handle.id
    assert_no_promotion(effects)


@pytest.mark.parametrize("report", [
    {"accepted": True}, {"passed": True}, {"is_accept": True}, {"passes_acceptance": True},
    {"verdict": "accept"}, {"decision": "approved"},
    {"passes_acceptance": True, "accepted": True, "decision": "pass"},
])
def test_explicit_consistent_positive_decision_may_register(effects, params, report):
    effects.evaluation.report = report
    assert flow._is_accept(report) is True
    result = flow.execute_baseline_train_flow(**params)
    assert result.status == "accepted"
    assert len(effects.registry.calls) == 1
    assert result.registered["arn"] == "fixture-package"


@pytest.mark.parametrize("report", [
    {"accepted": False}, {"passed": False}, {"is_accept": False}, {"passes_acceptance": False},
    {"verdict": "rejected"}, {"decision": "fail"},
    {"passes_acceptance": False, "accepted": False, "decision": "reject"},
])
def test_explicit_consistent_negative_decision_is_rejected(effects, params, report):
    effects.evaluation.report = report
    result = flow.execute_baseline_train_flow(**params)
    assert result.status == "rejected"
    assert result.arbiter_report == report
    assert_no_promotion(effects)


@pytest.mark.parametrize("predictions,accepted", [([0, 1, 0, 1], True), ([1, 0, 1, 0], False)])
def test_real_tabular_arbiter_controls_registration(effects, params, predictions, accepted):
    params["y_pred"] = predictions
    result = flow.execute_baseline_train_flow(**params)
    assert result.arbiter_report["passes_acceptance"] is accepted
    assert result.status == ("accepted" if accepted else "rejected")
    assert len(effects.registry.calls) == int(accepted)


def test_status_read_failure_keeps_accepted_job_identity_waiting(effects, params):
    effects.training.status_error = RuntimeError("observation unavailable")
    result = flow.execute_baseline_train_flow(**params)
    assert result.status == "waiting"
    assert (result.job_id, result.train_backend) == (effects.training.handle.id, "capture")
    assert [name for name, _ in effects.training.calls] == ["submit", "status"]
    assert_no_promotion(effects)


@pytest.mark.parametrize("stage", ["metrics", "grade"])
def test_successful_job_with_unavailable_assessment_is_unassessed(effects, params, stage):
    if stage == "metrics":
        effects.training.metrics_error = RuntimeError("metrics unavailable")
    else:
        effects.evaluation.error = RuntimeError("grading unavailable")
    result = flow.execute_baseline_train_flow(**params)
    assert result.status == "unassessed"
    assert result.job_id == effects.training.handle.id
    assert_no_promotion(effects)


def test_actual_default_approval_gate_still_denies_before_submission(effects, params):
    result = flow.run_baseline_train_flow(**params)
    assert result.status == "denied" and result.approved is False
    assert effects.training.calls == []
    assert_no_promotion(effects)


@pytest.mark.parametrize("state,graded,result_status", [
    (JobState.RUNNING, True, "waiting"), (JobState.SUCCEEDED, False, "unassessed"),
])
def test_actual_capability_caller_preserves_nested_flow_status(effects, params, monkeypatch, state, graded, result_status):
    effects.training.state = state
    if not graded:
        params["y_pred"] = None
    monkeypatch.setitem(capability._EXECUTOR_CACHE, "baseline_train_flow", flow.execute_baseline_train_flow)
    result = capability.invoke_capability(
        capability_id="baseline_train_flow", params=params,
        policy=ApprovalPolicy(mode=ApprovalMode.AUTO),
    )
    wire = result.to_dict()
    wire["result"] = json.loads(json.dumps(wire["result"]))
    assert wire["approved"] is True and wire["executed"] is True
    assert wire["result"]["status"] == result_status
    assert wire["result"]["job_id"] == effects.training.handle.id
    assert wire["result"]["train_backend"] == "capture"
    assert len([call for call in effects.training.calls if call[0] == "submit"]) == 1
    assert_no_promotion(effects)


def test_submit_failure_without_handle_is_an_operation_failure(effects, params):
    effects.training.submit_error = RuntimeError("submit refused")
    result = flow.execute_baseline_train_flow(**params)
    assert result.status == "failed" and result.job_id == ""
    assert [name for name, _ in effects.training.calls] == ["submit"]
    assert effects.evaluation.calls == effects.tracking.calls == []
    assert_no_promotion(effects)
