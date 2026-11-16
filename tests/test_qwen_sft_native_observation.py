"""Measured SageMaker defaults preserve the original request and strict drift refusal."""

from __future__ import annotations

import copy

import pytest
from test_qwen_sft_backend import SESSION, backend, catalog

from amlas.backends.types import JobState


def observed(defaults, state="InProgress", mutation=None):
    adapter, native, _s3, inputs = backend()
    handle = adapter.submit(inputs=inputs, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    native.state = state
    response = copy.deepcopy(native.describe_training_job(TrainingJobName=handle.id))
    if "algorithm" in defaults:
        response["AlgorithmSpecification"]["EnableSageMakerMetricsTimeSeries"] = False
    if "output" in defaults:
        response["OutputDataConfig"]["KmsKeyId"] = ""
    if "channel" in defaults:
        for channel in response["InputDataConfig"]:
            channel["RecordWrapperType"] = "None"
    if mutation is not None:
        field, key, value = mutation
        if field == "channel":
            response["InputDataConfig"][0][key] = value
        else:
            response[field][key] = value
    native.describe_training_job = lambda **_kw: response
    return adapter, handle, response


@pytest.mark.parametrize(
    "defaults", [("algorithm",), ("output",), ("channel",), ("algorithm", "output", "channel")]
)
@pytest.mark.parametrize(
    "state,expected",
    [("InProgress", JobState.RUNNING), ("Completed", JobState.SUCCEEDED), ("Failed", JobState.FAILED)],
)
def test_exact_native_defaults_report_original_state_without_mutation(defaults, state, expected):
    adapter, handle, response = observed(defaults, state)
    request_before = copy.deepcopy(handle.extra["request"])
    response_before = copy.deepcopy(response)
    assert adapter.status(handle).state == expected
    assert handle.extra["request"] == request_before
    assert response == response_before


@pytest.mark.parametrize(
    "mutation",
    [
        ("AlgorithmSpecification", "EnableSageMakerMetricsTimeSeries", True),
        ("AlgorithmSpecification", "EnableSageMakerMetricsTimeSeries", 0),
        ("AlgorithmSpecification", "EnableSageMakerMetricsTimeSeries", None),
        ("OutputDataConfig", "KmsKeyId", "foreign-key"),
        ("OutputDataConfig", "KmsKeyId", None),
        ("OutputDataConfig", "KmsKeyId", False),
        ("channel", "RecordWrapperType", "RecordIO"),
        ("channel", "RecordWrapperType", None),
        ("channel", "RecordWrapperType", False),
        ("channel", "UnknownFutureField", False),
        ("AlgorithmSpecification", "UnknownFutureField", False),
        ("OutputDataConfig", "UnknownFutureField", ""),
    ],
)
def test_unknown_malformed_and_changed_defaults_still_refuse(mutation):
    adapter, handle, _response = observed(("algorithm", "output", "channel"), mutation=mutation)
    assert adapter.status(handle).state == JobState.UNKNOWN
