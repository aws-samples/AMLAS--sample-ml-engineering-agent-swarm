"""Fixed inspection lifecycle: native SDK shapes, closed synthetic providers."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

ACCOUNT = "1234" * 3
REGION = "us-west-2"
BUCKET = "fixture-private-bucket"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/fixture-processing"
IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/inspection@sha256:" + "a" * 64
SOURCE_COMMIT = "b" * 40
DEPLOYMENT = "fd0d82ab-4e03-44c0-9bf1-908af5b2b031"
PAYLOAD = b"label,value\na,1\nb,2\na,3\n"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def modules() -> tuple[Any, Any]:
    try:
        return (
            importlib.import_module("amlas.backends.adapters.dataset_inspection"),
            importlib.import_module("amlas.backends.types"),
        )
    except ModuleNotFoundError:
        pytest.fail("missing-symbol: fixed managed dataset inspection backend")


def source() -> Any:
    _, types = modules()
    cls = getattr(types, "DatasetInspectionSource", None)
    ref = getattr(types, "DatasetArtifactRef", None)
    assert cls is not None and ref is not None, "missing-symbol: typed inspection source"
    return cls(
        artifact_ref=ref(
            uri=f"s3://{BUCKET}/sessions/session-a/acquisition/job/data.csv",
            version_id="source-version",
            sha256=DIGEST,
            byte_count=len(PAYLOAD),
            format="csv",
        ),
        account_id=ACCOUNT,
        region=REGION,
        bucket=BUCKET,
        runtime_session_id="session-a",
        deployment_id=DEPLOYMENT,
        deployment_prefix="fixture-lab",
        acquisition_job_arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/acquisition",
    )


class Processing:
    def __init__(self) -> None:
        self.meta = SimpleNamespace(region_name=REGION)
        self.calls: list[dict[str, Any]] = []
        self.state = "InProgress"
        self.fail_read = False
        self.extra_description: dict[str, Any] = {}

    def create_processing_job(self, **kwargs: Any) -> dict[str, str]:
        from botocore.session import Session
        from botocore.validate import validate_parameters

        validate_parameters(
            kwargs,
            Session().get_service_model("sagemaker").operation_model("CreateProcessingJob").input_shape,
        )
        self.calls.append(kwargs)
        return {"ProcessingJobArn": self.arn(kwargs["ProcessingJobName"])}

    def arn(self, name: str) -> str:
        return f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/{name}"

    def describe_processing_job(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_read:
            raise RuntimeError("closed provider observation failure")
        request = self.calls[-1]
        assert kwargs["ProcessingJobName"] == request["ProcessingJobName"]
        return {
            "ProcessingJobName": request["ProcessingJobName"],
            "ProcessingJobArn": self.arn(request["ProcessingJobName"]),
            "ProcessingJobStatus": self.state,
            "RoleArn": request["RoleArn"],
            "AppSpecification": request["AppSpecification"],
            "ProcessingInputs": request["ProcessingInputs"],
            "ProcessingOutputConfig": request["ProcessingOutputConfig"],
            "NetworkConfig": request["NetworkConfig"],
            "Environment": request.get("Environment", {}),
            "ProcessingResources": request["ProcessingResources"],
            "StoppingCondition": request["StoppingCondition"],
            **self.extra_description,
        }


class S3:
    def __init__(self) -> None:
        self.meta = SimpleNamespace(region_name=REGION)
        self.versions = {"sessions/session-a/acquisition/job/data.csv": ("source-version", PAYLOAD)}
        self.copies: list[dict[str, Any]] = []
        self.reads: list[dict[str, Any]] = []
        self.extra_input = False
        self.extra_output = False

    def get_bucket_location(self, **kwargs: Any) -> dict[str, str]:
        assert kwargs["ExpectedBucketOwner"] == ACCOUNT
        return {"LocationConstraint": REGION}

    def get_bucket_versioning(self, **kwargs: Any) -> dict[str, str]:
        return {"Status": "Enabled"}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        version, data = self.versions[kwargs["Key"]]
        assert kwargs.get("VersionId", version) == version
        assert kwargs["ExpectedBucketOwner"] == ACCOUNT
        return {"VersionId": version, "ContentLength": len(data)}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.reads.append(kwargs)
        version, data = self.versions[kwargs["Key"]]
        assert kwargs["VersionId"] == version
        return {"Body": io.BytesIO(data), "ContentLength": len(data), "VersionId": version}

    def copy_object(self, **kwargs: Any) -> dict[str, Any]:
        from botocore.handlers import handle_copy_source_param
        from botocore.session import Session
        from botocore.validate import validate_parameters

        wire = dict(kwargs)
        handle_copy_source_param(wire)
        validate_parameters(
            wire,
            Session().get_service_model("s3").operation_model("CopyObject").input_shape,
        )
        assert kwargs["CopySource"]["VersionId"] == "source-version"
        assert kwargs["ExpectedSourceBucketOwner"] == ACCOUNT
        assert kwargs["ExpectedBucketOwner"] == ACCOUNT
        self.copies.append(kwargs)
        self.versions[kwargs["Key"]] = ("staged-version", PAYLOAD)
        return {"VersionId": "staged-version", "CopyObjectResult": {"ETag": '"fixture"'}}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        keys = [key for key in self.versions if key.startswith(kwargs["Prefix"])]
        if "/input/" in kwargs["Prefix"] and self.extra_input:
            keys.append(kwargs["Prefix"] + "extra.csv")
        if "/output/" in kwargs["Prefix"] and self.extra_output:
            keys.append(kwargs["Prefix"] + "extra.bin")
        return {
            "Contents": [
                {"Key": key, "Size": len(self.versions[key][1]) if key in self.versions else 1}
                for key in keys
            ],
            "IsTruncated": False,
        }


@pytest.fixture(autouse=True)
def deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    for name in ("AMLAS_OFFLINE", "AMLAS_BACKENDS_OFFLINE", "DATA_INSPECTION_BACKEND"):
        monkeypatch.delenv(name, raising=False)


def backend(sm: Any = None, s3: Any = None, **kwargs: Any) -> Any:
    module, _ = modules()
    return module.ProcessingDatasetInspectionBackend(
        region=REGION,
        role_arn=ROLE,
        image_uri=IMAGE,
        session_bucket=BUCKET,
        source_commit=SOURCE_COMMIT,
        admission_digest=kwargs.pop("admission_digest", "a" * 64),
        sagemaker_client=sm or Processing(),
        s3_client=s3 or S3(),
        **kwargs,
    )


def test_request_stages_exact_version_and_uses_fixed_isolated_processing() -> None:
    sm, s3 = Processing(), S3()
    handle = backend(sm, s3).submit(source=source(), target_column="label")
    request = sm.calls[0]
    assert request["NetworkConfig"] == {"EnableNetworkIsolation": True}
    assert request["AppSpecification"]["ImageUri"] == IMAGE
    assert request["AppSpecification"]["ContainerEntrypoint"] == [
        "/usr/local/bin/amlas-processing-init",
        "inspection",
    ]
    assert request["ProcessingInputs"][0]["AppManaged"] is False
    assert request["ProcessingInputs"][0]["S3Input"]["S3InputMode"] == "File"
    assert request["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3UploadMode"] == "EndOfJob"
    assert not request.get("Environment")
    assert len(s3.copies) == 1
    assert s3.copies[0]["CopySource"]["VersionId"] == "source-version"
    assert all(read["VersionId"] for read in s3.reads)
    assert handle.extra["accepted"] is True
    assert handle.extra["inspected"] is False
    assert "policy" not in handle.extra


@pytest.mark.parametrize("marker", [None, "", "c" * 64])
def test_missing_or_stale_admission_denies_before_writes(marker: Any) -> None:
    sm, s3 = Processing(), S3()
    with pytest.raises(ValueError, match="admission"):
        backend(sm, s3, admission_digest=marker).submit(source=source())
    assert sm.calls == [] and s3.copies == []


def test_source_hash_and_input_extras_block_submission() -> None:
    sm, s3 = Processing(), S3()
    original = source()
    changed = replace(original, artifact_ref=replace(original.artifact_ref, sha256="d" * 64))
    with pytest.raises(ValueError, match="hash"):
        backend(sm, s3).submit(source=changed)
    assert sm.calls == [] and s3.copies == []
    s3.extra_input = True
    with pytest.raises(ValueError, match="input"):
        backend(sm, s3).submit(source=source())
    assert sm.calls == []


def test_arbitrary_dict_source_and_unbounded_target_never_submit() -> None:
    sm, s3 = Processing(), S3()
    for value, target in [({}, None), (source(), "x" * 129), (source(), "line\nbreak")]:
        with pytest.raises((ValueError, TypeError)):
            backend(sm, s3).submit(source=value, target_column=target)
    assert sm.calls == [] and s3.copies == []


def test_accepted_handle_survives_provider_read_failure() -> None:
    sm = Processing()
    sm.fail_read = True
    handle = backend(sm).submit(source=source())
    assert handle.arn and handle.extra["accepted"] is True
    assert len(sm.calls) == 1


def test_report_requires_native_isolation_and_fixed_outputs() -> None:
    sm, s3 = Processing(), S3()
    adapter = backend(sm, s3)
    handle = adapter.submit(source=source())
    sm.state = "Completed"
    sm.extra_description = {"NetworkConfig": {"EnableNetworkIsolation": False}}
    refused = adapter.report(handle)
    assert refused["success"] is False and refused["inspected"] is False
    sm.extra_description = {}
    s3.extra_output = True
    assert adapter.report(handle)["success"] is False


def test_version_pin_is_mandatory() -> None:
    sm, s3 = Processing(), S3()
    item = source()
    for version in ("", "null"):
        changed = replace(item, artifact_ref=replace(item.artifact_ref, version_id=version))
        with pytest.raises(ValueError, match="version"):
            backend(sm, s3).submit(source=changed)
    assert sm.calls == [] and s3.copies == []


def test_completed_report_verifies_worker_outputs_and_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    request: pytest.FixtureRequest,
) -> None:
    from amlas.sagemaker_entrypoints import dataset_inspection as worker

    sm, s3 = Processing(), S3()
    adapter = backend(sm, s3)
    handle = adapter.submit(source=source(), target_column="label")
    inputs = tmp_path / "input"
    outputs = tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    (inputs / "data.csv").write_bytes(PAYLOAD)
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", SOURCE_COMMIT)
    monkeypatch.setattr(
        worker,
        "_guest_preflight",
        lambda: {
            "output_directory_proof": {
                "fd": 3,
                "uid": 10001,
                "gid": 10001,
                "mode": 0o700,
                "device": 1,
                "inode": 1,
                "lease_active": True,
            },
            "credential_present": False,
            "provider_method": None,
            "credential_env_names": [],
            "metadata_reachable": False,
            "network_reachable": False,
            "privilege_proof": {
                "res_uid": [10001, 10001, 10001],
                "res_gid": [10001, 10001, 10001],
                "groups": [],
                "cap_eff": 0,
                "cap_prm": 0,
                "cap_inh": 0,
                "cap_amb": 0,
                "cap_bnd": 0,
                "no_new_privs": 1,
            },
        },
    )
    descriptor = os.open(outputs, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    monkeypatch.setattr(worker, "_OUTPUT_FD", descriptor)
    request.addfinalizer(lambda: os.close(descriptor))
    worker.run_inspection(
        input_dir=inputs,
        output_dir=outputs,
        expected_sha256=DIGEST,
        byte_count=len(PAYLOAD),
        format="csv",
        job_name=handle.id,
        runtime_session_id="session-a",
        deployment_id=DEPLOYMENT,
        deployment_prefix="fixture-lab",
        image_uri=IMAGE,
        source_commit=SOURCE_COMMIT,
        source_version_id="source-version",
        staged_version_id="staged-version",
        target_column="label",
    )
    for name in ("SUMMARY.json", "INSPECTED.json"):
        s3.versions[handle.extra["output_prefix"] + name] = (name + "-version", (outputs / name).read_bytes())
    sm.state = "Completed"
    report = adapter.report(handle)
    assert report["success"] is True and report["inspected"] is True
    assert report["summary"]["target_distribution"] == {"a": 2, "b": 1}
    assert report["summary_ref"]["version_id"] == "SUMMARY.json-version"
    assert report["prepared"] is False and report["trained"] is False
    key = handle.extra["output_prefix"] + "SUMMARY.json"
    version, raw = s3.versions[key]
    changed = json.loads(raw)
    changed["unexpected"] = "filesystem export"
    s3.versions[key] = (version, json.dumps(changed).encode())
    assert adapter.report(handle)["success"] is False


@pytest.mark.parametrize("defaults", ["network", "input", "both"])
def test_native_documented_defaults_preserve_fixed_binding(defaults: str) -> None:
    """Processing API's optional false/None defaults preserve fixed semantics."""
    from amlas.backends.types import JobState

    sm = Processing()
    adapter = backend(sm)
    handle = adapter.submit(source=source())
    original = deepcopy(sm.calls[0])
    if defaults in {"network", "both"}:
        sm.extra_description["NetworkConfig"] = {
            **original["NetworkConfig"],
            "EnableInterContainerTrafficEncryption": False,
        }
    if defaults in {"input", "both"}:
        inputs = deepcopy(original["ProcessingInputs"])
        inputs[0]["S3Input"]["S3CompressionType"] = "None"
        sm.extra_description["ProcessingInputs"] = inputs
    assert adapter.status(handle).state == JobState.RUNNING
    assert sm.calls[0] == original, "native normalization mutated the accepted request"


@pytest.mark.parametrize(
    "case",
    [
        "environment",
        "resources",
        "runtime",
        "missing_resources",
        "missing_runtime",
        "role",
        "image",
        "arguments",
        "entrypoint",
        "isolation",
        "vpc",
        "encryption",
        "compression",
        "input_uri",
        "input_path",
        "input_mode",
        "output_uri",
        "output_path",
        "upload_mode",
    ],
)
def test_native_semantic_deviations_refuse_observation(case: str) -> None:
    from amlas.backends.types import JobState

    sm = Processing()
    adapter = backend(sm)
    handle = adapter.submit(source=source())
    request = deepcopy(sm.calls[0])
    changes = {}
    if case == "environment":
        changes["Environment"] = {"UNEXPECTED_EXPORT": "enabled"}
    elif case in {"resources", "missing_resources"}:
        resources = deepcopy(request["ProcessingResources"])
        resources["ClusterConfig"]["InstanceCount"] = 2
        changes["ProcessingResources"] = resources if case == "resources" else None
    elif case in {"runtime", "missing_runtime"}:
        changes["StoppingCondition"] = {"MaxRuntimeInSeconds": 901} if case == "runtime" else None
    elif case == "role":
        changes["RoleArn"] = ROLE + "-other"
    elif case in {"image", "arguments", "entrypoint"}:
        app = deepcopy(request["AppSpecification"])
        if case == "image":
            app["ImageUri"] = IMAGE[:-64] + "c" * 64
        elif case == "arguments":
            app["ContainerArguments"] += ["--unexpected-script", "other.py"]
        else:
            app["ContainerEntrypoint"] = ["python", "/app/other.py"]
        changes["AppSpecification"] = app
    elif case in {"isolation", "vpc", "encryption"}:
        network = deepcopy(request["NetworkConfig"])
        if case == "isolation":
            network["EnableNetworkIsolation"] = False
        elif case == "vpc":
            network["VpcConfig"] = {"Subnets": ["subnet-fixture"], "SecurityGroupIds": ["sg-fixture"]}
        else:
            network["EnableInterContainerTrafficEncryption"] = True
        changes["NetworkConfig"] = network
    elif case in {"compression", "input_uri", "input_path", "input_mode"}:
        inputs = deepcopy(request["ProcessingInputs"])
        field, value = {
            "compression": ("S3CompressionType", "Gzip"),
            "input_uri": ("S3Uri", "s3://other-bucket/input/"),
            "input_path": ("LocalPath", "/opt/ml/processing/other"),
            "input_mode": ("S3InputMode", "Pipe"),
        }[case]
        inputs[0]["S3Input"][field] = value
        changes["ProcessingInputs"] = inputs
    else:
        output = deepcopy(request["ProcessingOutputConfig"])
        field, value = {
            "output_uri": ("S3Uri", "s3://other-bucket/output/"),
            "output_path": ("LocalPath", "/opt/ml/processing/other"),
            "upload_mode": ("S3UploadMode", "Continuous"),
        }[case]
        output["Outputs"][0]["S3Output"][field] = value
        changes["ProcessingOutputConfig"] = output
    sm.extra_description = changes
    assert adapter.status(handle).state == JobState.UNKNOWN
    assert sm.calls[0] == request
    assert len(sm.calls) == 1


@pytest.mark.parametrize("requested,reported", [("label", None), ("label", "value"), (None, "label")])
def test_rehashed_valid_summary_cannot_change_requested_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    requested: Any,
    reported: Any,
    request: pytest.FixtureRequest,
) -> None:
    from amlas.data_eng.inspection_schema import decode_summary
    from amlas.sagemaker_entrypoints import dataset_inspection as worker

    sm, s3 = Processing(), S3()
    adapter = backend(sm, s3)
    handle = adapter.submit(source=source(), target_column=requested)
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    (inputs / "data.csv").write_bytes(PAYLOAD)
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", SOURCE_COMMIT)
    monkeypatch.setattr(
        worker,
        "_guest_preflight",
        lambda: {
            "output_directory_proof": {
                "fd": 3,
                "uid": 10001,
                "gid": 10001,
                "mode": 0o700,
                "device": 1,
                "inode": 1,
                "lease_active": True,
            },
            "credential_present": False,
            "provider_method": None,
            "credential_env_names": [],
            "metadata_reachable": False,
            "network_reachable": False,
            "privilege_proof": {
                "res_uid": [10001, 10001, 10001],
                "res_gid": [10001, 10001, 10001],
                "groups": [],
                "cap_eff": 0,
                "cap_prm": 0,
                "cap_inh": 0,
                "cap_amb": 0,
                "cap_bnd": 0,
                "no_new_privs": 1,
            },
        },
    )
    descriptor = os.open(outputs, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    monkeypatch.setattr(worker, "_OUTPUT_FD", descriptor)
    request.addfinalizer(lambda: os.close(descriptor))
    worker.run_inspection(
        input_dir=inputs,
        output_dir=outputs,
        expected_sha256=DIGEST,
        byte_count=len(PAYLOAD),
        format="csv",
        job_name=handle.id,
        runtime_session_id="session-a",
        deployment_id=DEPLOYMENT,
        deployment_prefix="fixture-lab",
        image_uri=IMAGE,
        source_commit=SOURCE_COMMIT,
        source_version_id="source-version",
        staged_version_id="staged-version",
        target_column=requested,
    )
    summary = json.loads((outputs / "SUMMARY.json").read_bytes())
    summary["target_column"] = reported
    summary["target_distribution"] = None
    if reported == "label":
        summary["target_distribution"] = {"a": 2, "b": 1}
    elif reported == "value":
        summary["target_distribution"] = {"1": 1, "2": 1, "3": 1}
    raw = json.dumps(summary).encode()
    assert decode_summary(raw)["target_column"] == reported, "tampered summary must otherwise be valid"
    receipt = json.loads((outputs / "INSPECTED.json").read_bytes())
    receipt["summary_sha256"] = hashlib.sha256(raw).hexdigest()
    s3.versions[handle.extra["output_prefix"] + "SUMMARY.json"] = ("summary-version-2", raw)
    s3.versions[handle.extra["output_prefix"] + "INSPECTED.json"] = (
        "receipt-version-2",
        json.dumps(receipt).encode(),
    )
    sm.state = "Completed"
    report = adapter.report(handle)
    assert report["success"] is False and report["inspected"] is False
    assert report["status"] == "verification_failed"
