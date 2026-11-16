"""Declared training controls survive the port and actual native request builder."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters

from amlas.backends.adapters.training import SageMakerTrainingBackend
from amlas.tools.sagemaker import submit_training_job

ACCOUNT = "1111" * 3
REGION = "us-west-2"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/fixture-training"
IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/fixture-training@sha256:" + "a" * 64
TRAIN = "s3://fixture-bucket/train/"
VALIDATION = "s3://fixture-bucket/validation/"


class NativeSageMaker:
    def __init__(self):
        self.meta = SimpleNamespace(region_name=REGION)
        self.calls = []

    def create_training_job(self, **kwargs):
        model = Session().get_service_model("sagemaker").operation_model("CreateTrainingJob")
        validate_parameters(kwargs, model.input_shape)
        assert kwargs["ResourceConfig"]["InstanceCount"] >= 1
        self.calls.append(kwargs)
        return {"TrainingJobArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/" + kwargs["TrainingJobName"]}


class NativeDynamoDB:
    def __init__(self, fail_update=False):
        self.calls = []
        self.fail_update = fail_update

    def _call(self, operation, kwargs):
        shape = Session().get_service_model("dynamodb").operation_model(operation).input_shape
        validate_parameters(kwargs, shape)
        self.calls.append((operation, kwargs))
        return {}

    def put_item(self, **kwargs):
        return self._call("PutItem", kwargs)

    def update_item(self, **kwargs):
        result = self._call("UpdateItem", kwargs)
        if self.fail_update:
            raise ClientError({"Error": {"Code": "InternalFailure", "Message": "fixture"}}, "UpdateItem")
        return result


def backend(sm, ddb):
    return SageMakerTrainingBackend(
        region=REGION, execution_role_arn=ROLE, session_bucket="fixture-bucket",
        jobs_table="fixture-jobs", runtime_session_id="fixture-session",
        dynamodb_client=ddb, sagemaker_client=sm,
    )


def controls():
    return {
        "image_uri": IMAGE, "instance_type": "ml.g6e.12xlarge", "instance_count": 2,
        "volume_size_gb": 512, "max_runtime_seconds": 7200,
    }


def test_flat_hyperparameters_keep_declared_controls_and_validation_outside_model_params():
    sm, ddb = NativeSageMaker(), NativeDynamoDB()
    handle = backend(sm, ddb).submit(
        algo="fine_tune", train_uri=TRAIN, val_uri=VALIDATION, target="label",
        hyperparams={"learning_rate": 0.001}, resources=controls(),
        budget={"max_runtime_seconds": 1800},
    )
    native = sm.calls[0]
    assert native["AlgorithmSpecification"]["TrainingImage"] == IMAGE
    assert native["ResourceConfig"] == {
        "InstanceType": "ml.g6e.12xlarge", "InstanceCount": 2, "VolumeSizeInGB": 512,
    }
    assert native["StoppingCondition"] == {"MaxRuntimeInSeconds": 1800}
    assert native["Environment"]["AMLAS_TARGET"] == "label"
    assert {item["ChannelName"]: item["DataSource"]["S3DataSource"]["S3Uri"] for item in native["InputDataConfig"]} == {
        "training": TRAIN, "validation": VALIDATION,
    }
    assert native["HyperParameters"] == {"learning_rate": "0.001"}
    assert handle.arn.endswith("/" + handle.id)
    assert handle.extra["jobs_table"] == "fixture-jobs"
    assert [op for op, _ in ddb.calls] == ["PutItem", "UpdateItem"]


def test_existing_nested_config_is_preserved_with_explicit_port_arguments():
    sm, ddb = NativeSageMaker(), NativeDynamoDB()
    original = {
        **controls(), "hyperparameters": {"epochs": 2}, "test_s3_uri": "s3://fixture-bucket/test/",
        "content_type": "application/jsonlines", "s3_input": "s3://stale-bucket/input/",
    }
    backend(sm, ddb).submit(algo="fine_tune", train_uri=TRAIN, val_uri=VALIDATION, hyperparams=original)
    request = sm.calls[0]
    channels = {item["ChannelName"]: item for item in request["InputDataConfig"]}
    assert set(channels) == {"training", "validation", "test"}
    assert channels["training"]["DataSource"]["S3DataSource"]["S3Uri"] == TRAIN
    assert channels["validation"]["DataSource"]["S3DataSource"]["S3Uri"] == VALIDATION
    assert channels["test"]["DataSource"]["S3DataSource"]["S3Uri"] == "s3://fixture-bucket/test/"
    assert all(item["ContentType"] == "application/jsonlines" for item in channels.values())
    assert request["HyperParameters"] == {"epochs": "2"}
    assert original["s3_input"] == "s3://stale-bucket/input/"


@pytest.mark.parametrize("resource,budget,expected", [
    (None, 1200, 1200), (900, None, 900), (900, 1200, 900), (None, None, 3600),
])
def test_explicit_deadline_budget_is_preserved_without_inference(resource, budget, expected):
    sm, ddb = NativeSageMaker(), NativeDynamoDB()
    res = {"image_uri": IMAGE}
    if resource is not None:
        res["max_runtime_seconds"] = resource
    limits = {"max_cost_usd": 10000}
    if budget is not None:
        limits["max_runtime_seconds"] = budget
    backend(sm, ddb).submit(algo="fine_tune", train_uri=TRAIN, resources=res, budget=limits)
    assert sm.calls[0]["StoppingCondition"]["MaxRuntimeInSeconds"] == expected


@pytest.mark.parametrize("key,value", [
    ("instance_count", 0), ("instance_count", True), ("instance_count", 2.5),
    ("volume_size_gb", -1), ("volume_size_gb", math.inf),
    ("max_runtime_seconds", 0), ("max_runtime_seconds", math.nan),
])
def test_invalid_declared_resources_deny_before_pending_or_native_writes(key, value):
    sm, ddb = NativeSageMaker(), NativeDynamoDB()
    resource = {**controls(), key: value}
    with pytest.raises(ValueError):
        backend(sm, ddb).submit(algo="fine_tune", train_uri=TRAIN, resources=resource)
    assert ddb.calls == [] and sm.calls == []


def test_invalid_budget_deadline_denies_before_writes():
    sm, ddb = NativeSageMaker(), NativeDynamoDB()
    with pytest.raises(ValueError):
        backend(sm, ddb).submit(
            algo="fine_tune", train_uri=TRAIN, resources={"image_uri": IMAGE},
            budget={"max_runtime_seconds": True},
        )
    assert ddb.calls == [] and sm.calls == []


def test_direct_submit_config_count_and_validation_reach_native_shape():
    sm, ddb = NativeSageMaker(), NativeDynamoDB()
    submit_training_job(
        training_type="fine_tune",
        config={**controls(), "s3_input": TRAIN, "validation_s3_uri": VALIDATION, "target_column": "label"},
        runtime_session_id="fixture-session", execution_role_arn=ROLE, session_bucket="fixture-bucket",
        jobs_table="fixture-jobs", region=REGION, dynamodb_client=ddb, sagemaker_client=sm,
    )
    assert sm.calls[0]["ResourceConfig"]["InstanceCount"] == 2
    assert {c["ChannelName"] for c in sm.calls[0]["InputDataConfig"]} == {"training", "validation"}
    assert sm.calls[0]["Environment"]["AMLAS_TARGET"] == "label"


def test_native_accepted_handle_recovery_preserves_selectors_after_tracking_failure():
    sm, ddb = NativeSageMaker(), NativeDynamoDB(fail_update=True)
    handle = backend(sm, ddb).submit(
        algo="fine_tune", train_uri=TRAIN, val_uri=VALIDATION, resources=controls(),
    )
    assert handle.arn == f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/" + handle.id
    assert handle.extra["tracking_error"]
    assert handle.extra["jobs_table"] == "fixture-jobs"
    assert sm.calls[0]["RoleArn"] == ROLE
    assert sm.calls[0]["OutputDataConfig"]["S3OutputPath"].startswith("s3://fixture-bucket/")
