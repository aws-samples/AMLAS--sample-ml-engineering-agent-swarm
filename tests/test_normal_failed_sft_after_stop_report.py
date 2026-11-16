"""After a worker stop, an SFT job that ends Failed or Stopped still gets a truthful final report.

The original worker saved the TASK checkpoint and stopped while the SFT job ran. The job then ended Failed or Stopped,
so no checkpoint exists and nothing can continue. A replacement in a fresh process restores the run, retains one
source-written report that names the job, its terminal state and the class of its failure reason, and answers
completed. It creates no job, stops nothing, calls no model, and a later resume reads the same report.
"""

from __future__ import annotations

import asyncio

import pytest
from normal_sft_cold_fixture import close, cloud_data, original_precommit, setup
from test_normal_sft_cold_continuation import child
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def _replacement_report(monkeypatch, tmp_path, terminal, *, unselected=False, failure_class="unclassified"):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            # The original is a preference-list job that SageMaker never gave an instance: every description lacks
            # the Selected fields, and the terminal one shows only Starting and Pending before it ended.
            state.native.unselected_training = unselected
            await original_precommit(state, monkeypatch)
            assert (state.original.id in state.native.unselected) is unselected
            assert "InstancePreferences" in state.original.extra["request"]["ResourceConfig"]
            data = cloud_data(
                state
            )  # TASK saved, job InProgress: the durable state of a stop in the SFT wait
            data["states"][data["original_job"]] = terminal
            answer = child(data, tmp_path / ("replacement-" + terminal.lower()))
            states = [
                (e.get("event"), e.get("state") or e.get("status"), e.get("diagnostic"))
                for e in answer["events"]
                if e.get("event") != "heartbeat"
            ]
            reports = [e for e in answer["report"] if e.get("event") == "prompt_report"]
            assert answer["new_training"] == [] and answer["stops"] == []
            assert reports and reports[0].get("status") == "available", (
                f"a replacement for a {terminal} SFT job retained no report; resume events: {states}; "
                f"report read: {[(e.get('event'), e.get('status')) for e in answer['report']]}"
            )
            assert states == [
                ("resume_state", "restoring", None),
                ("resume_state", "completed", None),
            ], states
            assert answer["model_calls"] == 0 and answer["invoked"] == [] and answer["new_processing"] == []
            report = reports[0]["report"]
            assert report["status"] == "response_received" and report["worker_finished"] is True
            response = report["response"]
            assert response["action"] == "original_sft_ended" and response["suggestions"] == []
            facts = response["data"]["original_sft"]
            assert facts["job_name"] == state.original.id and facts["native_status"] == terminal
            assert facts["failure_class"] == failure_class and facts["checkpoint"] is False
            # The report names the instance SageMaker selected from the job's list; none when it never selected.
            selected = {
                key: facts[key] for key in ("SelectedInstanceType", "SelectedInstanceCount") if key in facts
            }
            assert selected == (
                {} if unselected else {"SelectedInstanceType": "ml.g6e.16xlarge", "SelectedInstanceCount": 1}
            )
            assert set(facts) == {
                "job_name",
                "native_status",
                "failure_class",
                "recipe_id",
                "checkpoint",
                *selected,
            }
            assert state.original.id in response["message"] and terminal in response["message"]
            assert "no job was submitted again" in response["message"]
            # A later resume of the same run reads the retained report and starts nothing.
            again = child(answer["transport"], tmp_path / ("again-" + terminal.lower()))
            assert [e.get("state") for e in again["events"] if e.get("event") == "resume_state"] == [
                "restoring",
                "completed",
            ]
            assert again["report"] == answer["report"]
            assert again["new_training"] == again["stops"] == [] and again["model_calls"] == 0
        finally:
            await close(state)

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", ["Failed", "Stopped"])
def test_replacement_reports_a_failed_or_stopped_sft_job(monkeypatch, tmp_path, terminal):
    _replacement_report(monkeypatch, tmp_path, terminal)


@pytest.mark.parametrize("terminal,failure_class", [("Failed", "unclassified"), ("Stopped", "Stopped")])
def test_replacement_reports_a_preference_job_that_ended_before_selection(
    monkeypatch, tmp_path, terminal, failure_class
):
    """The same report when the job ended while it waited for capacity, so SageMaker never selected an instance."""
    _replacement_report(monkeypatch, tmp_path, terminal, unselected=True, failure_class=failure_class)


@pytest.mark.parametrize(
    "status,description,expected",
    [
        ("Failed", {"FailureReason": "AlgorithmError: neutral script exit"}, "AlgorithmError"),
        ("Failed", {"FailureReason": "CapacityError: neutral capacity loss"}, "CapacityError"),
        ("Failed", {"FailureReason": "ClientError: neutral input"}, "ClientError"),
        ("Failed", {"FailureReason": "InternalServerError: neutral"}, "InternalServerError"),
        ("Failed", {"FailureReason": "Unlisted wording without a class"}, "unclassified"),
        ("Failed", {"FailureReason": 7}, "unclassified"),
        ("Failed", {}, "unclassified"),
        ("Stopped", {"SecondaryStatus": "MaxRuntimeExceeded"}, "MaxRuntimeExceeded"),
        ("Stopped", {"SecondaryStatus": "Stopped"}, "Stopped"),
        ("Stopped", {"SecondaryStatus": "Neutral other"}, "unclassified"),
    ],
)
def test_failure_class_is_a_closed_label_never_native_text(status, description, expected):
    from amlas import app as runtime

    label = runtime._original_sft_failure_class(status, {"TrainingJobStatus": status, **description})
    assert label == expected
    assert "neutral" not in label


@pytest.mark.parametrize(
    "selected,expected",
    [
        (
            {"SelectedInstanceType": "ml.g6e.12xlarge", "SelectedInstanceCount": 1},
            {"SelectedInstanceType": "ml.g6e.12xlarge", "SelectedInstanceCount": 1},
        ),
        ({}, {}),
        (None, {}),
        ({"SelectedInstanceType": "ml.g6e.12xlarge"}, {}),
        ({"SelectedInstanceCount": 1}, {}),
        ({"SelectedInstanceType": 7, "SelectedInstanceCount": 1}, {}),
        ({"SelectedInstanceType": "ml.g6e.12xlarge", "SelectedInstanceCount": True}, {}),
        (
            {
                "SelectedInstanceType": "ml.g6e.12xlarge",
                "SelectedInstanceCount": 1,
                "FailureReason": "neutral",
            },
            {"SelectedInstanceType": "ml.g6e.12xlarge", "SelectedInstanceCount": 1},
        ),
    ],
)
def test_report_facts_keep_both_selected_fields_or_neither(selected, expected):
    from amlas import app as runtime

    end = runtime._OriginalSftEnd("owned-job", "Failed", "unclassified", "full_v1", selected)
    assert end.selected == expected
    assert runtime._OriginalSftEnd("owned-job", "Failed", "unclassified", "full_v1").selected == {}
