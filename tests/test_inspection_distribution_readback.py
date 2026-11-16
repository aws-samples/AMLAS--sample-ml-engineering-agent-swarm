"""Replay accepted inspection readback defaults with public-only saved handles."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import pytest
import test_inspection_native_ceiling as native
import test_managed_dataset_inspection as managed

from amlas.backends.adapters.dataset_inspection import _native_field
from amlas.backends.types import JobHandle, JobState

JOB = "fixture-lab-data-inspect-" + "a" * 12


@pytest.fixture(autouse=True)
def context(monkeypatch: pytest.MonkeyPatch) -> Any:
    yield from native.fixture_context.__wrapped__(monkeypatch)


def accepted_request() -> dict[str, Any]:
    prefix = f"s3://{managed.BUCKET}/sessions/session-a/inspection/{JOB}/"
    pairs = [
        ("--sha256", managed.DIGEST),
        ("--byte-count", str(len(managed.PAYLOAD))),
        ("--format", "csv"),
        ("--job-name", JOB),
        ("--runtime-session-id", "session-a"),
        ("--deployment-id", managed.DEPLOYMENT),
        ("--deployment-prefix", "fixture-lab"),
        ("--image-uri", managed.IMAGE),
        ("--source-commit", managed.SOURCE_COMMIT),
        ("--source-version-id", "source-version"),
        ("--staged-version-id", "staged-version"),
    ]
    return {
        "ProcessingJobName": JOB,
        "RoleArn": managed.ROLE,
        "AppSpecification": {
            "ImageUri": managed.IMAGE,
            "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "inspection"],
            "ContainerArguments": [part for pair in pairs for part in pair],
        },
        "NetworkConfig": {"EnableNetworkIsolation": True},
        "ProcessingResources": {
            "ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.xlarge", "VolumeSizeInGB": 30}
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 900},
        "ProcessingInputs": [
            {
                "InputName": "inspection_input",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": prefix + "input/",
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "inspection",
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": prefix + "output/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
    }


def saved_handle(request: dict[str, Any]) -> JobHandle:
    prefix = f"sessions/session-a/inspection/{JOB}/"
    return JobHandle(
        id=JOB,
        backend="processing",
        arn=f"arn:aws:sagemaker:{managed.REGION}:{managed.ACCOUNT}:processing-job/{JOB}",
        extra={
            "accepted": True,
            "inspected": False,
            "source": asdict(managed.source()),
            "role_arn": managed.ROLE,
            "image_uri": managed.IMAGE,
            "source_commit": managed.SOURCE_COMMIT,
            "privilege_profile": native.PROFILE,
            "target_column": "label",
            "bucket": managed.BUCKET,
            "region": managed.REGION,
            "account_id": managed.ACCOUNT,
            "runtime_session_id": "session-a",
            "deployment_id": managed.DEPLOYMENT,
            "deployment_prefix": "fixture-lab",
            "input_prefix": prefix + "input/",
            "input_key": prefix + "input/data.csv",
            "staged_version_id": "staged-version",
            "output_prefix": prefix + "output/",
            "request": request,
        },
    )


class SavedProcessing(managed.Processing):
    def __init__(self, request: dict[str, Any], state: str, distribution: Any = "FullyReplicated") -> None:
        super().__init__()
        self.saved_request = deepcopy(request)
        self.state = state
        inputs = deepcopy(request["ProcessingInputs"])
        if distribution == "omitted":
            inputs[0]["S3Input"].pop("S3DataDistributionType", None)
        else:
            inputs[0]["S3Input"]["S3DataDistributionType"] = distribution
        self.description = {
            **deepcopy(request),
            "ProcessingJobArn": self.arn(JOB),
            "ProcessingJobStatus": state,
            "ProcessingInputs": inputs,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "fixture-readback-request",
                "HTTPHeaders": {"content-type": "application/x-amz-json-1.1"},
            },
        }
        self.descriptions: list[str] = []

    def describe_processing_job(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["ProcessingJobName"] == JOB
        self.descriptions.append(kwargs["ProcessingJobName"])
        return deepcopy(self.description)

    def create_processing_job(self, **kwargs: Any) -> Any:
        pytest.fail("replay tried to resubmit an accepted job")


@pytest.mark.parametrize(
    "state,expected", [("InProgress", JobState.RUNNING), ("Completed", JobState.SUCCEEDED)]
)
def test_same_accepted_native_handle_admits_documented_default(state: str, expected: JobState) -> None:
    request = accepted_request()
    original = deepcopy(request)
    handle = saved_handle(request)
    sm, s3 = SavedProcessing(request, state), managed.S3()
    adapter = native.native_adapter(sm, s3)
    observed = adapter.status(handle)
    assert observed.state == expected
    assert observed.raw["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert sm.descriptions == [JOB] and sm.calls == []
    assert request == original and handle.id == JOB and handle.extra["request"] == original
    assert s3.reads == [] and s3.copies == []


@pytest.mark.parametrize("reported", ["ShardedByS3Key", "invalid", None, 1, False])
def test_changed_invalid_or_null_distribution_refuses_before_output_reads(reported: Any) -> None:
    request = accepted_request()
    handle = saved_handle(request)
    sm, s3 = SavedProcessing(request, "Completed", reported), managed.S3()
    adapter = native.native_adapter(sm, s3)
    assert adapter.status(handle).state == JobState.UNKNOWN
    assert adapter.report(handle)["status"] == "observation_failed"
    assert s3.reads == [] and s3.copies == [] and sm.calls == []
    assert handle.id == JOB


@pytest.mark.parametrize(
    "requested,reported,expected",
    [
        ("FullyReplicated", "omitted", JobState.SUCCEEDED),
        ("ShardedByS3Key", "ShardedByS3Key", JobState.SUCCEEDED),
        ("ShardedByS3Key", "FullyReplicated", JobState.UNKNOWN),
        ("ShardedByS3Key", "omitted", JobState.UNKNOWN),
    ],
)
def test_explicit_distribution_remains_bound(requested: str, reported: str, expected: JobState) -> None:
    request = accepted_request()
    request["ProcessingInputs"][0]["S3Input"]["S3DataDistributionType"] = requested
    sm, s3 = SavedProcessing(request, "Completed", reported), managed.S3()
    assert native.native_adapter(sm, s3).status(saved_handle(request)).state == expected
    assert sm.calls == [] and s3.reads == []


def test_default_comparison_copies_nested_inputs_and_preserves_existing_defaults() -> None:
    request = accepted_request()
    reported = deepcopy(request["ProcessingInputs"])
    reported[0]["S3Input"].update(S3DataDistributionType="FullyReplicated", S3CompressionType="None")
    before = deepcopy(reported)
    assert _native_field("ProcessingInputs", reported) == _native_field(
        "ProcessingInputs", request["ProcessingInputs"]
    )
    assert reported == before
    assert json.loads(
        _native_field(
            "NetworkConfig", {"EnableNetworkIsolation": True, "EnableInterContainerTrafficEncryption": False}
        )
    ) == {"EnableNetworkIsolation": True}


def test_completed_saved_handle_verifies_normal_native_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    request = accepted_request()
    handle = saved_handle(request)
    sm, s3 = SavedProcessing(request, "Completed"), managed.S3()
    native.publish_fixture(monkeypatch, tmp_path, handle, s3)
    key = handle.extra["output_prefix"] + "INSPECTED.json"
    version, raw = s3.versions[key]
    receipt = json.loads(raw)
    receipt.update(schema_version=2, privilege_profile=native.PROFILE)
    receipt["guest_evidence"]["privilege_proof"] = native.native_proof()
    s3.versions[key] = (version, json.dumps(receipt).encode())
    report = native.native_adapter(sm, s3).report(handle)
    assert report["success"] is True and report["inspected"] is True
    assert report["prepared"] is False and report["trained"] is False
    assert all(item["VersionId"] for item in s3.reads)
    assert sm.calls == [] and s3.copies == [] and handle.id == JOB
