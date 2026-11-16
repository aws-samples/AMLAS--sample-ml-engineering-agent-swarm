"""Preparation and dispatch acceptance must not masquerade as actual training."""

from __future__ import annotations

import pytest

from amlas.cloud.dispatch import DispatchResult


def result(events: list[dict]) -> DispatchResult:
    return DispatchResult(events, "session", "", "", {})


@pytest.mark.parametrize(
    "events",
    [
        [{"event": "data_staged", "n_train": 100}],
        [{"event": "run_started", "project_id": "coding"}],
        [{"event": "data_staged"}, {"event": "run_started"},
         {"event": "error", "error": "trainer unavailable"}],
        [{"event": "training_job_started", "job_status": "InProgress"}],
        [{"event": "training_job_started", "job_name": "coding-trial",
          "job_status": "Submitted"}],
    ],
)
def test_unproven_training_progress_is_false(events: list[dict]) -> None:
    assert not result(events).reached_training


@pytest.mark.parametrize("status", ["InProgress", "Completed"])
def test_native_training_progress_requires_job_identity_and_observed_state(status: str) -> None:
    assert result([{
        "event": "training_job_started",
        "job_name": "coding-trial",
        "job_status": status,
    }]).reached_training
