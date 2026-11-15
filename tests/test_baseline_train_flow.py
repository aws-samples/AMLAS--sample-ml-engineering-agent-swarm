"""Baseline training must honor the canonical Arbiter acceptance result."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from amlas.arbiter.schema import ArbiterReport
from amlas.backends.ports import (
    EvaluationBackend,
    ExperimentTrackingBackend,
    ModelRegistryBackend,
    TrainingBackend,
)
from amlas.backends.types import JobHandle, JobState, JobStatus, ModelRef, PackageRef
from amlas.flows import baseline_train_flow


@pytest.mark.parametrize(
    ("passes_acceptance", "expected_status", "expected_registrations"),
    [(False, "rejected", 0), (True, "accepted", 1)],
)
def test_canonical_arbiter_result_controls_registration(
    monkeypatch: pytest.MonkeyPatch,
    passes_acceptance: bool,
    expected_status: str,
    expected_registrations: int,
) -> None:
    handle = JobHandle(id="training-job", backend="stub")
    artifact = ModelRef(uri="model://candidate", backend="stub")
    report = ArbiterReport(
        task_type="classification",
        primary_metric="f1_macro",
        overall_score=0.79 if passes_acceptance else 0.69,
        direction="maximize",
        baseline_score=0.70,
        delta_vs_baseline=0.09 if passes_acceptance else -0.01,
        passes_acceptance=passes_acceptance,
        acceptance_reason="acceptance threshold result",
    )

    class TrainingStub:
        def submit(self, **_: Any) -> JobHandle:
            return handle

        def status(self, _: JobHandle) -> JobStatus:
            return JobStatus(state=JobState.SUCCEEDED)

        def metrics(self, _: JobHandle) -> dict[str, float]:
            return {"f1_macro": report.overall_score}

        def artifact(self, _: JobHandle) -> ModelRef:
            return artifact

    class EvaluationStub:
        def grade(self, **_: Any) -> ArbiterReport:
            return report

    class TrackingStub:
        def log_run(self, **_: Any) -> str:
            return "run-1"

    class RegistryStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def register(self, **kwargs: Any) -> PackageRef:
            self.calls.append(kwargs)
            return PackageRef(arn="package://candidate/1", group="candidates")

    registry = RegistryStub()
    adapters = {
        TrainingBackend: TrainingStub(),
        EvaluationBackend: EvaluationStub(),
        ExperimentTrackingBackend: TrackingStub(),
        ModelRegistryBackend: registry,
    }

    def resolve(port: type, **_: Any) -> SimpleNamespace:
        if port is ModelRegistryBackend and not passes_acceptance:
            raise AssertionError("A rejected candidate must not access Model Registry")
        return SimpleNamespace(adapter=adapters[port])

    monkeypatch.setattr(baseline_train_flow.REGISTRY, "resolve", resolve)

    result = baseline_train_flow.execute_baseline_train_flow(
        algo="stub-algorithm",
        train_uri="dataset://train",
        spec=object(),
        y_true=[0, 1],
        y_pred=[0, 1],
        baseline_score=report.baseline_score,
        register_group="candidates",
    )

    assert result.status == expected_status
    assert result.arbiter_report == report.model_dump()
    assert len(registry.calls) == expected_registrations
    if passes_acceptance:
        assert result.registered == {
            "arn": "package://candidate/1",
            "group": "candidates",
            "version": 1,
            "approval_status": "PendingManualApproval",
        }
    else:
        assert result.registered == {}
