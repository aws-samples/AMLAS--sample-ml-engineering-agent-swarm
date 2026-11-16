"""Readback of training jobs whose stored request carries ResourceConfig.InstancePreferences.

The recorded answers in `fixtures/sagemaker_instance_preference_readback_real.json` come from three real
preference-list jobs (SFT, reload, tuned generation) and one fixed-type job. SageMaker adds SelectedInstanceType and
SelectedInstanceCount to the described ResourceConfig once it has chosen capacity. The readback accepts them only when
they match the immutable stored request, and keeps every other ResourceConfig comparison exact.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_qwen_sft_backend import SESSION, backend, catalog

from amlas.backends.types import JobState

FIXTURE = Path(__file__).parent / "fixtures" / "sagemaker_instance_preference_readback_real.json"
RECORDED = {job["name"]: job for job in json.loads(FIXTURE.read_text())["jobs"]}
PREFERENCE_JOBS = [
    name for name, job in RECORDED.items() if "InstancePreferences" in job["stored_request_resource_config"]
]
FIXED_JOB = "fixture-deploy-gen-base"
SFT_JOB = "fixture-deploy-sft-full"


def readback(stored, described, *, state="Completed", secondary="Completed", transitions=None, observe=None):
    adapter, native, _s3, inputs = backend()
    handle = adapter.submit(inputs=inputs, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    handle.extra["request"]["ResourceConfig"] = copy.deepcopy(stored)
    native.state = state
    response = copy.deepcopy(native.describe_training_job(TrainingJobName=handle.id))
    response["ResourceConfig"] = copy.deepcopy(described)
    if secondary is not None:
        response["SecondaryStatus"] = secondary
    if transitions is not None:
        response["SecondaryStatusTransitions"] = copy.deepcopy(transitions)
    native.describe_training_job = lambda **_kw: response
    request_before = copy.deepcopy(handle.extra["request"])
    response_before = copy.deepcopy(response)
    status = adapter.status(handle)
    if observe is not None:
        observe(adapter.artifacts(handle))
    assert handle.extra["request"] == request_before
    assert response == response_before
    return status


def recorded(name):
    job = RECORDED[name]
    return copy.deepcopy(job["stored_request_resource_config"]), copy.deepcopy(
        job["described_resource_config"]
    )


def without(config, *keys):
    return {k: v for k, v in config.items() if k not in keys}


def test_fixture_holds_the_three_recorded_preference_jobs_and_one_fixed_job():
    assert sorted(PREFERENCE_JOBS) == [
        "fixture-deploy-gen-tuned",
        "fixture-deploy-reload",
        "fixture-deploy-sft-full",
    ]
    for name in PREFERENCE_JOBS:
        stored, described = recorded(name)
        assert {"SelectedInstanceType", "SelectedInstanceCount"} <= set(described)
        assert without(described, "SelectedInstanceType", "SelectedInstanceCount") == stored
    assert "InstanceType" in RECORDED[FIXED_JOB]["stored_request_resource_config"]


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_recorded_jobs_read_back_their_native_state_and_keep_the_selected_instance(name):
    job = RECORDED[name]
    stored, described = recorded(name)
    status = readback(stored, described, state=job["TrainingJobStatus"], secondary=job["SecondaryStatus"])
    assert status.state == JobState.SUCCEEDED
    assert status.raw["ResourceConfig"] == described
    if name in PREFERENCE_JOBS:
        assert status.raw["ResourceConfig"]["SelectedInstanceType"] == described["SelectedInstanceType"]
        assert status.raw["ResourceConfig"]["SelectedInstanceCount"] == described["SelectedInstanceCount"]


@pytest.mark.parametrize("name", PREFERENCE_JOBS)
@pytest.mark.parametrize(
    "state,secondary,expected",
    [
        ("InProgress", "Downloading", JobState.RUNNING),
        ("InProgress", "Training", JobState.RUNNING),
        ("InProgress", "Uploading", JobState.RUNNING),
        ("Stopping", "Stopping", JobState.RUNNING),
        ("Failed", "Failed", JobState.FAILED),
        ("Stopped", "Stopped", JobState.STOPPED),
    ],
)
def test_selected_instance_reads_back_in_every_later_state(name, state, secondary, expected):
    stored, described = recorded(name)
    assert readback(stored, described, state=state, secondary=secondary).state == expected


@pytest.mark.parametrize("secondary", ["Starting", "Pending"])
def test_both_selected_fields_absent_read_running_only_before_selection(secondary):
    stored, described = recorded(SFT_JOB)
    absent = without(described, "SelectedInstanceType", "SelectedInstanceCount")
    status = readback(stored, absent, state="InProgress", secondary=secondary)
    assert status.state == JobState.RUNNING
    assert "SelectedInstanceType" not in status.raw["ResourceConfig"]


@pytest.mark.parametrize(
    "state,secondary",
    [
        ("InProgress", "Downloading"),
        ("InProgress", "Training"),
        ("InProgress", "Uploading"),
        ("InProgress", None),
        ("InProgress", "LaunchingMLInstances"),
        ("Completed", "Completed"),
        ("Completed", "Pending"),
        ("Failed", "Failed"),
        ("Stopped", "Stopped"),
        ("Stopping", "Stopping"),
    ],
)
def test_both_selected_fields_absent_after_selection_refuse(state, secondary):
    stored, described = recorded(SFT_JOB)
    absent = without(described, "SelectedInstanceType", "SelectedInstanceCount")
    assert readback(stored, absent, state=state, secondary=secondary).state == JobState.UNKNOWN


@pytest.mark.parametrize("missing", ["SelectedInstanceType", "SelectedInstanceCount"])
@pytest.mark.parametrize(
    "state,secondary", [("InProgress", "Pending"), ("InProgress", "Training"), ("Completed", "Completed")]
)
def test_one_selected_field_alone_refuses(missing, state, secondary):
    stored, described = recorded(SFT_JOB)
    assert (
        readback(stored, without(described, missing), state=state, secondary=secondary).state
        == JobState.UNKNOWN
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("SelectedInstanceType", "ml.p5.48xlarge"),
        ("SelectedInstanceType", "ml.g6e.48xlarge"),
        ("SelectedInstanceType", "ml.g6e.16xlarge "),
        ("SelectedInstanceCount", 2),
        ("SelectedInstanceCount", 0),
    ],
)
def test_unexpected_selected_values_refuse(field, value):
    stored, described = recorded(SFT_JOB)
    described[field] = value
    assert readback(stored, described).state == JobState.UNKNOWN


@pytest.mark.parametrize(
    "field,value",
    [
        ("SelectedInstanceType", None),
        ("SelectedInstanceType", ""),
        ("SelectedInstanceType", 7),
        ("SelectedInstanceType", ["ml.g6e.16xlarge"]),
        ("SelectedInstanceType", {"InstanceType": "ml.g6e.16xlarge"}),
        ("SelectedInstanceCount", None),
        ("SelectedInstanceCount", "1"),
        ("SelectedInstanceCount", True),
        ("SelectedInstanceCount", 1.0),
    ],
)
def test_malformed_selected_values_refuse(field, value):
    stored, described = recorded(SFT_JOB)
    described[field] = value
    assert readback(stored, described).state == JobState.UNKNOWN


def _reordered(config):
    config["InstancePreferences"] = list(reversed(config["InstancePreferences"]))


def _dropped(config):
    config["InstancePreferences"] = config["InstancePreferences"][:-1]


def _extra(config):
    config["InstancePreferences"] = [*config["InstancePreferences"][:-1], {"InstanceType": "ml.g6e.48xlarge"}]


def _per_preference_count(config):
    config["InstancePreferences"][1]["InstanceCount"] = 1


def _volume(config):
    config["VolumeSizeInGB"] = 200


def _count(config):
    config["InstanceCount"] = 2
    config["SelectedInstanceCount"] = 2


def _fixed_type_beside_list(config):
    config["InstanceType"] = "ml.g6e.16xlarge"


def _future_field(config):
    config["UnknownFutureField"] = "x"


def _malformed_count(config):
    config.clear()
    config["InstanceCount"] = [1]


@pytest.mark.parametrize(
    "change",
    [
        _reordered,
        _dropped,
        _extra,
        _per_preference_count,
        _volume,
        _count,
        _fixed_type_beside_list,
        _future_field,
        _malformed_count,
    ],
)
def test_other_resource_config_drift_still_refuses(change):
    stored, described = recorded(SFT_JOB)
    change(described)
    assert readback(stored, described).state == JobState.UNKNOWN


def test_non_mapping_described_resource_config_refuses():
    stored, _described = recorded(SFT_JOB)
    assert readback(stored, None).state == JobState.UNKNOWN
    assert readback(stored, [stored]).state == JobState.UNKNOWN


@pytest.mark.parametrize(
    "extra",
    [
        {"SelectedInstanceType": "ml.g6e.4xlarge", "SelectedInstanceCount": 1},
        {"SelectedInstanceType": "ml.g6e.4xlarge"},
        {"SelectedInstanceCount": 1},
        {"InstancePreferences": [{"InstanceType": "ml.g6e.4xlarge"}]},
    ],
)
def test_fixed_type_request_is_compared_unchanged(extra):
    stored, described = recorded(FIXED_JOB)
    assert readback(stored, described).state == JobState.SUCCEEDED
    assert readback(stored, {**described, **extra}).state == JobState.UNKNOWN
    pending = {**described, **extra}
    assert readback(stored, pending, state="InProgress", secondary="Pending").state == JobState.UNKNOWN


# --- a preference-list job that ended or is stopping before SageMaker selected an instance ---------------------------

# Status messages of a real fixed-type job that SageMaker stopped while it waited for capacity (DescribeTrainingJob in
# us-west-2, read 2026-11-11): Starting, Pending, then Stopped, no Downloading or Training.
_REAL_MESSAGES = {
    "Starting": "Starting the training job",
    "Pending": "Training job stopped",
    "Stopped": "Training job stopped",
}


def transitions(*statuses):
    """SecondaryStatusTransitions in the native shape: Status, StartTime, EndTime and StatusMessage per entry."""
    start = datetime(2026, 11, 10, 5, 1, 22, tzinfo=UTC)
    return [
        {
            "Status": status,
            "StartTime": start + timedelta(seconds=index),
            "EndTime": start + timedelta(seconds=index + 1),
            "StatusMessage": _REAL_MESSAGES.get(status, "neutral status message"),
        }
        for index, status in enumerate(statuses)
    ]


def _absent_selection():
    stored, described = recorded(SFT_JOB)
    return stored, without(described, "SelectedInstanceType", "SelectedInstanceCount")


@pytest.mark.parametrize(
    "state,secondary,statuses,expected",
    [
        ("Stopped", "Stopped", ("Starting", "Pending", "Stopped"), JobState.STOPPED),
        ("Stopped", "Stopped", ("Starting", "Pending", "Stopping", "Stopped"), JobState.STOPPED),
        ("Stopped", "Stopped", ("Starting", "Stopped"), JobState.STOPPED),
        ("Failed", "Failed", ("Starting", "Pending", "Failed"), JobState.FAILED),
        ("Failed", "Failed", ("Starting", "Failed"), JobState.FAILED),
        ("Stopping", "Stopping", ("Starting", "Pending", "Stopping"), JobState.RUNNING),
        ("Stopping", "Pending", ("Starting", "Pending"), JobState.RUNNING),
    ],
)
def test_absent_selection_of_a_job_that_never_left_the_waiting_phase_reads_its_state(
    state, secondary, statuses, expected
):
    """For preference lists too, a job stopped or failed before selection reads its terminal state."""
    stored, absent = _absent_selection()
    seen = []
    status = readback(
        stored,
        absent,
        state=state,
        secondary=secondary,
        transitions=transitions(*statuses),
        observe=seen.append,
    )
    assert status.state == expected
    assert status.detail == ""
    assert not {"SelectedInstanceType", "SelectedInstanceCount"} & set(status.raw["ResourceConfig"])
    assert seen[0]["status"] == expected.value
    assert not {"SelectedInstanceType", "SelectedInstanceCount"} & set(seen[0])


def test_the_real_waiting_then_stopped_shape_reads_stopped():
    stored, absent = _absent_selection()
    status = readback(
        stored,
        absent,
        state="Stopped",
        secondary="Stopped",
        transitions=transitions("Starting", "Pending", "Stopped"),
    )
    assert status.state == JobState.STOPPED


@pytest.mark.parametrize(
    "state,secondary,statuses",
    [
        # Absent after the job reached Downloading, Training or Uploading: SageMaker had selected an instance.
        ("Stopped", "Stopped", ("Starting", "Pending", "Downloading", "Stopped")),
        (
            "Stopped",
            "Stopped",
            ("Starting", "Pending", "Downloading", "Training", "Stopping", "Uploading", "Stopped"),
        ),
        ("Failed", "Failed", ("Starting", "Pending", "Downloading", "Training", "Failed")),
        ("Stopping", "Stopping", ("Starting", "Pending", "Training", "Stopping")),
        ("Stopped", "Stopped", ("Starting", "LaunchingMLInstances", "Stopped")),
        # No completed job was ever without an instance.
        ("Completed", "Completed", ("Starting", "Pending", "Completed")),
        # The last transition must be the matching terminal entry, once, and equal SecondaryStatus.
        ("Stopped", "Stopped", ("Starting", "Pending", "Failed")),
        ("Failed", "Failed", ("Starting", "Pending", "Stopped")),
        ("Stopped", "MaxRuntimeExceeded", ("Starting", "Pending", "Stopped")),
        ("Stopped", None, ("Starting", "Pending", "Stopped")),
        ("Stopped", "Stopped", ("Starting", "Pending", "Stopped", "Stopped")),
        ("Stopped", "Stopped", ("Stopped",)),
        ("Stopped", "Stopped", ("Starting", "Stopped", "Pending", "Stopped")),
        ("Stopping", "Stopping", ("Starting", "Pending")),
        ("Stopping", "Pending", ("Starting", "Pending", "Stopping")),
        ("Failed", "Failed", ("Starting", "Pending", "Stopping", "Stopping", "Failed")),
        # The waiting phase is not enough for a job still in progress past it.
        ("InProgress", "Downloading", ("Starting", "Pending", "Downloading")),
    ],
)
def test_absent_selection_after_any_other_transition_still_refuses(state, secondary, statuses):
    stored, absent = _absent_selection()
    status = readback(stored, absent, state=state, secondary=secondary, transitions=transitions(*statuses))
    assert status.state == JobState.UNKNOWN


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"Status": "Stopped"},
        "Starting,Pending,Stopped",
        [{"Status": "Starting"}, "Pending", {"Status": "Stopped"}],
        [{"Status": "Starting"}, {"Status": 7}, {"Status": "Stopped"}],
        [{"Status": "Starting"}, {}, {"Status": "Stopped"}],
        [("Status", "Starting"), {"Status": "Stopped"}],
    ],
)
def test_malformed_transitions_refuse_an_absent_selection(value):
    stored, absent = _absent_selection()
    adapter_status = readback(stored, absent, state="Stopped", secondary="Stopped", transitions=value)
    assert adapter_status.state == JobState.UNKNOWN


@pytest.mark.parametrize("missing", ["SelectedInstanceType", "SelectedInstanceCount"])
def test_one_selected_field_alone_refuses_after_the_waiting_phase_too(missing):
    stored, described = recorded(SFT_JOB)
    status = readback(
        stored,
        without(described, missing),
        state="Stopped",
        secondary="Stopped",
        transitions=transitions("Starting", "Pending", "Stopped"),
    )
    assert status.state == JobState.UNKNOWN


def test_a_fixed_type_request_that_ended_while_waiting_is_compared_unchanged():
    stored, described = recorded(FIXED_JOB)
    waiting = transitions("Starting", "Pending", "Stopped")
    assert readback(stored, described, state="Stopped", secondary="Stopped", transitions=waiting).state == (
        JobState.STOPPED
    )
    extra = {"SelectedInstanceType": stored["InstanceType"], "SelectedInstanceCount": 1}
    assert (
        readback(
            stored, {**described, **extra}, state="Stopped", secondary="Stopped", transitions=waiting
        ).state
        == JobState.UNKNOWN
    )
