"""An in-progress preference-list fine-tune reads as a known, running original; an unreadable one never does.

Documents: `fixtures/sagemaker_training_tag_read_real.json`. `list_job` holds every distinct DescribeTrainingJob answer
of one real preference-list job from Pending to Stopped, read every 10 s. `sft_job` holds the stored native request
of the first real normal-path fine-tune, its DescribeTrainingJob answer after completion and its tags.

The in-progress documents of the fine-tune itself were not kept, so `in_progress()` joins the two real sources: the
request-bound fields of the fine-tune's own answer, and the status-bound fields (status, secondary status,
transitions, the selected instance and which time and artifact fields exist yet) of the list job's answer in that
state. Starting and InProgress/Uploading were not caught by a 10 s read and are not claimed here.

The run was held because IAM refused the ownership-tag read of every observation. The last tests keep that
refusal: an observation whose tags cannot be read, or name another session, is UNKNOWN in every state.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from test_precommit_unknown_wall_clock import CAP_POLLS, LOGGER, Rig, active_run, lines  # noqa: F401
from test_qwen_sft_backend import ACCOUNT, catalog
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.backends.adapters.qwen_sft import (
    SageMakerQwenSftBackend,
    _selected_instance_view,
    selected_instance,
)
from amlas.backends.types import JobHandle, JobState
from amlas.tui.chat import tool_binding_backend as binding

pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
FIXTURE = Path(__file__).parent / "fixtures" / "sagemaker_training_tag_read_real.json"
RECORDED = json.loads(FIXTURE.read_text().replace("ACCOUNT", ACCOUNT))
LIST_JOB, SFT_JOB = RECORDED["list_job"], RECORDED["sft_job"]
PHASES = {
    (item["describe"]["TrainingJobStatus"], item["describe"]["SecondaryStatus"]): item["describe"]
    for item in LIST_JOB["describes"]
}
IN_PROGRESS = [
    ("InProgress", "Pending"),
    ("InProgress", "Downloading"),
    ("InProgress", "Training"),
    ("Stopping", "Stopping"),
    ("Stopping", "Uploading"),
]
SELECTED = {"SelectedInstanceType": "ml.g6e.12xlarge", "SelectedInstanceCount": 1}
STATUS_BOUND = (
    "TrainingJobStatus",
    "SecondaryStatus",
    "SecondaryStatusTransitions",
    "LastModifiedTime",
    "TrainingStartTime",
    "TrainingTimeInSeconds",
    "TrainingEndTime",
    "BillableTimeInSeconds",
    "ModelArtifacts",
)


def in_progress(phase):
    """The fine-tune's own answer with the status-bound fields of the real list job in `phase`."""
    real, completed = PHASES[phase], SFT_JOB["completed_describe"]
    described = {key: copy.deepcopy(value) for key, value in completed.items() if key not in STATUS_BOUND}
    described.update({key: copy.deepcopy(real[key]) for key in STATUS_BOUND if key in real})
    resource = copy.deepcopy(SFT_JOB["stored_request"]["ResourceConfig"])
    resource.update({key: real["ResourceConfig"][key] for key in SELECTED if key in real["ResourceConfig"]})
    described["ResourceConfig"] = resource
    # The joined answer has exactly the fields of the real one in this state (the list job has no input channels).
    assert set(described) == set(real) | {"InputDataConfig"}
    return described


class Native:
    """Answers one DescribeTrainingJob document and one ListTags answer, or refuses the tag read as IAM did."""

    def __init__(self, described, tags, *, refused=False):
        self.described, self.tags, self.refused, self.calls = described, tags, refused, []

    def describe_training_job(self, **values):
        self.calls.append(("DescribeTrainingJob", values))
        return copy.deepcopy(self.described)

    def list_tags(self, **values):
        self.calls.append(("ListTags", values))
        if self.refused:
            # CloudTrail recorded the code AccessDenied; the SDK-side code name is the service's exception name.
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": RECORDED["refused_tag_read"]["message"],
                    }
                },
                "ListTags",
            )
        return {"Tags": copy.deepcopy(self.tags)}


def observe(described, *, tags=None, refused=False):
    request = copy.deepcopy(SFT_JOB["stored_request"])
    native = Native(described, SFT_JOB["tags"] if tags is None else tags, refused=refused)
    adapter = SageMakerQwenSftBackend(sagemaker_client=native, s3_client=object())
    handle = JobHandle(
        request["TrainingJobName"],
        "qwen_sft",
        described["TrainingJobArn"],
        {
            "catalog": catalog(),
            "request": request,
            "recipe_id": "fit_v1",
            "operation": "train",
            "submission_state": "accepted",
        },
    )
    return adapter, handle, native


def test_fixture_holds_the_real_states_of_one_list_job_in_order():
    assert list(PHASES) == [*IN_PROGRESS, ("Stopped", "Stopped")]
    assert [
        item["InstanceType"] for item in LIST_JOB["request"]["ResourceConfig"]["InstancePreferences"]
    ] == [
        "ml.g6e.16xlarge",
        "ml.g6e.12xlarge",
        "ml.g6e.8xlarge",
        "ml.g6e.4xlarge",
    ]
    for phase, described in PHASES.items():
        chosen = {
            key: described["ResourceConfig"][key] for key in SELECTED if key in described["ResourceConfig"]
        }
        # Before selection the described ResourceConfig is the stored one; afterwards it adds exactly both fields.
        assert chosen == ({} if phase == ("InProgress", "Pending") else SELECTED)
        assert {k: v for k, v in described["ResourceConfig"].items() if k not in SELECTED} == LIST_JOB[
            "request"
        ]["ResourceConfig"]
        assert [item["Status"] for item in described["SecondaryStatusTransitions"]][-1] == phase[1]
        assert ("ModelArtifacts" in described) == (phase == ("Stopped", "Stopped"))
        assert ("TrainingStartTime" in described) == (phase != ("InProgress", "Pending"))
    assert sorted(tag["Key"] for tag in SFT_JOB["tags"]) == [
        "amlas:deployment-id",
        "amlas:deployment-prefix",
        "amlas:runtime-session-id",
    ]
    assert SFT_JOB["completed_describe"]["ResourceConfig"] == {
        **SFT_JOB["stored_request"]["ResourceConfig"],
        **SELECTED,
    }


@pytest.mark.parametrize("phase", list(PHASES), ids=["-".join(phase) for phase in PHASES])
def test_real_list_job_answers_keep_the_stored_resource_config_in_every_state(phase):
    described, request = PHASES[phase], LIST_JOB["request"]
    before = copy.deepcopy(described)
    assert _selected_instance_view(described, request["ResourceConfig"]) == request["ResourceConfig"]
    assert selected_instance(described, request) == ({} if phase == ("InProgress", "Pending") else SELECTED)
    assert described == before


@pytest.mark.parametrize("phase", IN_PROGRESS, ids=["-".join(phase) for phase in IN_PROGRESS])
def test_in_progress_fine_tune_reads_running_with_its_identity_and_ownership_checked(phase):
    adapter, handle, native = observe(in_progress(phase))
    status = adapter.status(handle)
    assert status.state is JobState.RUNNING
    chosen = {} if phase == ("InProgress", "Pending") else SELECTED
    assert status.detail == " ".join(f"{key}={value}" for key, value in chosen.items())
    assert [name for name, _ in native.calls] == ["DescribeTrainingJob", "ListTags"]
    assert native.calls[1][1] == {"ResourceArn": handle.arn}
    result = adapter.artifacts(handle)
    assert result["status"] == "RUNNING" and result["success"] is False and result["trained"] is False
    assert {key: result[key] for key in SELECTED if key in result} == chosen
    assert (result["job_name"], result["job_arn"]) == (handle.id, handle.arn)


def test_real_completed_and_stopped_answers_read_their_terminal_state():
    adapter, handle, _native = observe(SFT_JOB["completed_describe"])
    completed = adapter.status(handle)
    assert completed.state is JobState.SUCCEEDED
    assert completed.detail == "SelectedInstanceType=ml.g6e.12xlarge SelectedInstanceCount=1"
    adapter, handle, _native = observe(in_progress(("Stopped", "Stopped")))
    assert adapter.status(handle).state is JobState.STOPPED


@pytest.mark.parametrize(
    "phase", [*IN_PROGRESS, None], ids=[*("-".join(phase) for phase in IN_PROGRESS), "Completed"]
)
def test_refused_tag_read_is_unknown_in_every_state(phase):
    described = SFT_JOB["completed_describe"] if phase is None else in_progress(phase)
    adapter, handle, native = observe(described, refused=True)
    status = adapter.status(handle)
    assert status.state is JobState.UNKNOWN
    assert status.raw == {"error_type": "ClientError"}
    # The request readback passed and the ownership read was made: nothing was skipped or assumed.
    assert [name for name, _ in native.calls] == ["DescribeTrainingJob", "ListTags"]
    result = adapter.artifacts(handle)
    assert result["status"] == "UNKNOWN" and result["success"] is False
    assert not set(SELECTED) & set(result)


@pytest.mark.parametrize(
    "change",
    [
        lambda tags: [
            {**tag, "Value": "another-session"} if tag["Key"].endswith("session-id") else tag for tag in tags
        ],
        lambda tags: tags[:-1],
        lambda tags: [*tags, {"Key": "amlas:extra", "Value": "x"}],
        lambda tags: [],
    ],
    ids=["another-session", "one-missing", "one-added", "none"],
)
@pytest.mark.parametrize(
    "phase", [("InProgress", "Pending"), ("InProgress", "Training")], ids=["Pending", "Training"]
)
def test_foreign_ownership_tags_are_unknown_while_in_progress(phase, change):
    adapter, handle, _native = observe(in_progress(phase), tags=change(copy.deepcopy(SFT_JOB["tags"])))
    assert adapter.status(handle).state is JobState.UNKNOWN


def test_changed_stored_request_is_unknown_while_in_progress():
    described = in_progress(("InProgress", "Training"))
    described["StoppingCondition"] = {"MaxRuntimeInSeconds": 7200}
    adapter, handle, native = observe(described)
    assert adapter.status(handle).state is JobState.UNKNOWN
    assert [name for name, _ in native.calls] == ["DescribeTrainingJob"]


def wait_rig(monkeypatch, phases, *, refused=False):
    """The precommit wait over a store that answers, as the SFT store does, with the adapter's own observation."""
    observations = []

    def script(call):
        phase = phases[min(call, len(phases)) - 1]
        adapter, handle, native = observe(in_progress(phase), refused=refused)
        observations.append((phase, native))
        return adapter.artifacts(handle)

    return Rig(monkeypatch, "sft", script, limit=400), observations


def test_wait_keeps_a_real_job_that_waits_then_trains_longer_than_the_cap(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    # 40 observations while it waits for capacity (600 s), then Downloading and 80 while it trains: 1,920 s in all.
    phases = (
        [("InProgress", "Pending")] * 40
        + [("InProgress", "Downloading")] * 8
        + [("InProgress", "Training")] * 80
    )
    rig, observations = wait_rig(monkeypatch, phases)
    for _ in phases:
        assert rig.read() is None
        assert rig.backend._precommit_unresolved == {}
        rig.clock.now += 15.0
    assert rig.clock.now - 5000.0 == 1920.0 > binding._PRECOMMIT_UNKNOWN_WALL_CLOCK_SECONDS
    assert rig.backend._warm_closed is None
    assert lines(caplog) == []
    assert [phase for phase, _ in observations] == phases
    assert all(
        [name for name, _ in native.calls] == ["DescribeTrainingJob", "ListTags"]
        for _, native in observations
    )


def test_wait_closes_at_the_cap_when_every_tag_read_is_refused(monkeypatch):
    phases = (
        [("InProgress", "Pending")] * 34 + [("InProgress", "Downloading")] * 7 + [("InProgress", "Training")]
    )
    rig, observations = wait_rig(monkeypatch, phases, refused=True)
    with pytest.raises(binding._InputAdmissionHeldError):
        rig.loop()
    assert rig.store.calls == CAP_POLLS
    assert rig.backend._warm_closed == RECORDED["held_report"]["disposition"]
    assert rig.backend._warm_closed_diagnostic.as_data() == RECORDED["held_report"]["diagnostic"]
    assert {phase for phase, _ in observations} == set(phases)
