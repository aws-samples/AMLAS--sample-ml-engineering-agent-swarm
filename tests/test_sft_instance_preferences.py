"""Ordered instance-type preferences in the fixed SFT, reload and generation requests.

The described ResourceConfig is the real shape of one fine-tune job of 2026-11-10, which SageMaker placed on
ml.g6e.16xlarge from a five-type list. No AWS call.
"""

from __future__ import annotations

import copy

import pytest
from botocore.session import Session
from botocore.validate import validate_parameters
from test_qwen_sft_backend import SESSION, catalog

from amlas.backends.adapters.qwen_sft import _native_request_view
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.generation_state import fixed_generation_request
from amlas.control_plane.sft_state import (
    GENERATION_INSTANCE_PREFERENCES,
    RELOAD_INSTANCE_PREFERENCES,
    RELOAD_PREFERENCE_CONTRACT,
    RELOAD_RESOURCE_CONTRACT,
    TRAIN_INSTANCE_PREFERENCES,
    fixed_resource_config,
    fixed_sft_request,
)


def _run_ref(family, job):
    c = catalog()
    return DatasetArtifactRef(
        "s3://" + c.bucket + "/sessions/" + SESSION + "/" + family + "/" + job + "/input/bindings/RUN.json",
        "version-1",
        "a" * 64,
        10,
        "json",
    )


def _requests():
    c = catalog()
    job = c.deployment_prefix + "-sft-preferences"
    gen = c.deployment_prefix + "-gen-preferences"
    return {
        "train": fixed_sft_request(c, SESSION, job, "train", "full_v1", _run_ref("sft", job)),
        "reload": fixed_sft_request(
            c,
            SESSION,
            job,
            "reload",
            "full_v1",
            _run_ref("sft", job),
            resource_contract=RELOAD_PREFERENCE_CONTRACT,
        ),
        "generation": fixed_generation_request(c, SESSION, gen, _run_ref("generation", gen), "base"),
    }


def _described(request, selected="ml.g6e.16xlarge", count=1, status=("Completed", "Completed")):
    value = copy.deepcopy(request["ResourceConfig"])
    if selected is not None:
        value["SelectedInstanceType"] = selected
    if count is not None:
        value["SelectedInstanceCount"] = count
    return {
        **copy.deepcopy(request),
        "ResourceConfig": value,
        "TrainingJobStatus": status[0],
        "SecondaryStatus": status[1],
    }


@pytest.mark.parametrize("family", ["train", "reload", "generation"])
def test_fixed_requests_list_preferences_the_installed_sdk_accepts(family):
    request = _requests()[family]
    validate_parameters(
        {key: value for key, value in request.items()},
        Session().get_service_model("sagemaker").operation_model("CreateTrainingJob").input_shape,
    )
    config = request["ResourceConfig"]
    assert "InstanceType" not in config and config["InstanceCount"] == 1 and config["VolumeSizeInGB"] == 100
    expected = {
        "train": TRAIN_INSTANCE_PREFERENCES,
        "reload": RELOAD_INSTANCE_PREFERENCES,
        "generation": GENERATION_INSTANCE_PREFERENCES,
    }[family]
    assert [item["InstanceType"] for item in config["InstancePreferences"]] == list(expected)


def test_installed_sdk_describes_the_selected_instance():
    shape = Session().get_service_model("sagemaker").shape_for("ResourceConfig")
    assert {"InstancePreferences", "SelectedInstanceType", "SelectedInstanceCount"} <= set(shape.members)


@pytest.mark.parametrize("selected", ["ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.4xlarge"])
def test_real_described_selection_reads_back_as_the_request(selected):
    request = _requests()["train"]
    view = _native_request_view(_described(request, selected=selected), request)
    assert view["ResourceConfig"] == request["ResourceConfig"]


def test_pending_description_without_selection_reads_back_as_the_request():
    request = _requests()["reload"]
    pending = _described(request, selected=None, count=None, status=("InProgress", "Pending"))
    view = _native_request_view(pending, request)
    assert view["ResourceConfig"] == request["ResourceConfig"]


@pytest.mark.parametrize(
    "selected,count",
    [
        ("ml.g6e.12xlarge", 1),  # not in the generation list
        ("ml.g6e.24xlarge", 1),  # in no list (ml.g6e.16xlarge is a generation type since 2026-11-11)
        ("ml.g6e.2xlarge", 1),  # in no list
        ("ml.p4d.24xlarge", 1),
        ("ml.g6e.4xlarge", 2),
        ("ml.g6e.4xlarge", True),
        ("ml.g6e.4xlarge", None),
        (None, 1),
    ],
)
def test_selection_outside_the_request_is_refused(selected, count):
    request = _requests()["generation"]
    with pytest.raises(ValueError, match="selected instance"):
        _native_request_view(_described(request, selected=selected, count=count), request)


@pytest.mark.parametrize("selected", ["ml.g6e.16xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge"])
def test_every_generation_type_reads_back_as_the_request(selected):
    request = _requests()["generation"]
    assert [item["InstanceType"] for item in request["ResourceConfig"]["InstancePreferences"]] == [
        "ml.g6e.16xlarge",
        "ml.g6e.4xlarge",
        "ml.g6e.8xlarge",
    ]
    view = _native_request_view(_described(request, selected=selected), request)
    assert view["ResourceConfig"] == request["ResourceConfig"]


def test_one_entry_reload_list_reads_back_only_its_one_type():
    """The reload lists ml.g6e.12xlarge alone: that selection reads back, any other selected type refuses."""
    request = _requests()["reload"]
    assert request["ResourceConfig"]["InstancePreferences"] == [{"InstanceType": "ml.g6e.12xlarge"}]
    view = _native_request_view(_described(request, selected="ml.g6e.12xlarge"), request)
    assert view["ResourceConfig"] == request["ResourceConfig"]
    for other in ("ml.g6e.16xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge", "ml.g6e.2xlarge"):
        with pytest.raises(ValueError, match="selected instance"):
            _native_request_view(_described(request, selected=other), request)


def test_fixed_type_request_does_not_drop_selection_fields():
    request = _requests()["train"]
    legacy = {
        **request,
        "ResourceConfig": {"InstanceType": "ml.g6e.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100},
    }
    view = _native_request_view(_described(legacy), legacy)
    assert view["ResourceConfig"] != legacy["ResourceConfig"]


@pytest.mark.parametrize(
    "preferences",
    [
        (),
        ("ml.g6e.4xlarge",) * 2,
        (
            "ml.g6e.4xlarge",
            "ml.g6e.8xlarge",
            "ml.g6e.16xlarge",
            "ml.g6e.2xlarge",
            "ml.g6e.12xlarge",
            "ml.g6e.24xlarge",
        ),
        ("ml.p4d.24xlarge",),
        ["ml.g6e.4xlarge"],
        # Only the three pinned lists are accepted: other members or another order refuse.
        ("ml.g6e.12xlarge", "ml.g6e.16xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge", "ml.g6e.2xlarge"),
        ("ml.g6e.4xlarge", "ml.g6e.8xlarge", "ml.g6e.16xlarge", "ml.g6e.2xlarge"),
        ("ml.g6e.12xlarge", "ml.g6e.16xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge"),
        ("ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.8xlarge"),
        ("ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge", "ml.g6e.2xlarge"),
        ("ml.g6e.4xlarge", "ml.g6e.8xlarge", "ml.g6e.16xlarge", "ml.g6e.12xlarge"),
        ("ml.g6e.8xlarge", "ml.g6e.4xlarge", "ml.g6e.16xlarge"),
        ("ml.g6e.4xlarge",),
        # The lists of the first port that shared ml.g6e.16xlarge between reload and generation.
        ("ml.g6e.4xlarge", "ml.g6e.8xlarge", "ml.g6e.16xlarge"),
        ("ml.g6e.12xlarge", "ml.g6e.16xlarge"),
        ("ml.g6e.8xlarge", "ml.g6e.4xlarge"),
        # The reload and generation lists of 2026-11-10, replaced on 2026-11-11, and one-entry lists
        # other than the reload's.
        ("ml.g6e.16xlarge", "ml.g6e.12xlarge"),
        ("ml.g6e.4xlarge", "ml.g6e.8xlarge"),
        ("ml.g6e.16xlarge",),
        ("ml.g6e.8xlarge",),
    ],
)
def test_preference_list_bounds_refuse(preferences):
    with pytest.raises(ValueError):
        fixed_resource_config(preferences)


@pytest.mark.parametrize(
    "operation,contract,historical,instance_type",
    [
        ("train", None, True, "ml.g6e.12xlarge"),  # a training record written before the capacity option
        ("reload", None, False, "ml.g6e.12xlarge"),  # reload record schemas 1 to 3
        ("reload", RELOAD_RESOURCE_CONTRACT, False, "ml.g6e.4xlarge"),  # reload_single_gpu_v1 records
    ],
)
def test_historical_requests_keep_their_exact_fixed_type(operation, contract, historical, instance_type):
    c = catalog()
    job = c.deployment_prefix + "-sft-preferences"
    request = fixed_sft_request(
        c,
        SESSION,
        job,
        operation,
        "full_v1",
        _run_ref("sft", job),
        resource_contract=contract,
        historical=historical,
    )
    assert request["ResourceConfig"] == {
        "InstanceType": instance_type,
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
    }


def test_historical_flag_is_training_only():
    c = catalog()
    job = c.deployment_prefix + "-sft-preferences"
    with pytest.raises(ValueError):
        fixed_sft_request(
            c,
            SESSION,
            job,
            "reload",
            "full_v1",
            _run_ref("sft", job),
            resource_contract=RELOAD_PREFERENCE_CONTRACT,
            historical=True,
        )
