"""Acquisition lifecycle tests use native SDK request validation, never live jobs."""

from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from amlas.backends.adapters import dataset_acquisition as acquisition_backend
from amlas.backends.types import JobState

ACCOUNT = "1234" * 3
REGION = "us-west-2"
IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/processing@sha256:" + "a" * 64
ROLE = f"arn:aws:iam::{ACCOUNT}:role/fixture-processing"
REVISION = "b" * 40
DEPLOYMENT = "fd0d82ab-4e03-44c0-9bf1-908af5b2b031"


class NativeProcessing:
    """Validate SDK request structure before accepting this fake native job."""

    def __init__(self, state="InProgress", describe_error=False):
        self.meta = SimpleNamespace(region_name=REGION)
        self.calls = []
        self.state = state
        self.describe_error = describe_error
        self.request = {}

    def create_processing_job(self, **kwargs):
        from botocore.session import Session
        from botocore.validate import validate_parameters

        operation = Session().get_service_model("sagemaker").operation_model("CreateProcessingJob")
        validate_parameters(kwargs, operation.input_shape)
        assert 1 <= kwargs["ProcessingResources"]["ClusterConfig"]["InstanceCount"] <= 100
        assert len(kwargs["ProcessingJobName"]) <= 63
        self.request = kwargs
        self.calls.append(kwargs)
        return {"ProcessingJobArn": self.arn(kwargs["ProcessingJobName"])}

    def arn(self, name):
        return f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/{name}"

    def describe_processing_job(self, **kwargs):
        if self.describe_error:
            raise RuntimeError("fixture observation failure")
        assert kwargs["ProcessingJobName"] == self.request["ProcessingJobName"]
        return {
            "ProcessingJobArn": self.arn(kwargs["ProcessingJobName"]),
            "ProcessingJobName": kwargs["ProcessingJobName"],
            "ProcessingJobStatus": self.state,
            "FailureReason": "fixture native failure" if self.state == "Failed" else "",
            "AppSpecification": self.request["AppSpecification"],
            "ProcessingOutputConfig": self.request["ProcessingOutputConfig"],
            "RoleArn": self.request["RoleArn"],
            "Environment": self.request.get("Environment", {}),
        }


class NativeS3:
    def __init__(self):
        self.objects = {}
        self.heads = {}
        self.meta = SimpleNamespace(region_name=REGION)

    def get_bucket_location(self, **kwargs):
        return {"LocationConstraint": REGION}

    def get_bucket_versioning(self, **kwargs):
        return {"Status": "Enabled"}

    def get_object(self, **kwargs):
        if kwargs.get("VersionId", "fixture-version") != "fixture-version":
            raise ValueError("NoSuchVersion")
        body = self.objects[kwargs["Key"]]
        return {"Body": io.BytesIO(body), "ContentLength": len(body), "VersionId": "fixture-version"}

    def head_object(self, **kwargs):
        return self.heads[kwargs["Key"]]


@pytest.fixture
def ownership(monkeypatch):
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    for key in ("AMLAS_OFFLINE", "AMLAS_BACKENDS_OFFLINE", "DATA_ACQUISITION_BACKEND"):
        monkeypatch.delenv(key, raising=False)


def adapter(sm=None, s3=None, **kwargs):
    return acquisition_backend.ProcessingDatasetAcquisitionBackend(
        region=REGION,
        role_arn=ROLE,
        image_uri=IMAGE,
        session_bucket="fixture-private-bucket",
        source_commit="c" * 40,
        sagemaker_client=sm or NativeProcessing(),
        s3_client=s3 or NativeS3(),
        **kwargs,
    )


def submit(backend, **kwargs):
    return backend.submit(
        repo_id="fixture/open-code",
        revision=REVISION,
        filename="data/train.jsonl",
        resources={"runtime_session_id": "session-a", **kwargs},
    )


def populate(handle, s3, *, data_path="acquisition-test/data.jsonl", checksum=None):
    data = b'{"messages":[]}\n'
    digest = hashlib.sha256(data).hexdigest()
    prefix = handle.extra["output_prefix"]
    lineage = {
        "repo_id": "fixture/open-code",
        "revision": REVISION,
        "filename": "data/train.jsonl",
        "source_uri": f"https://huggingface.co/datasets/fixture/open-code/resolve/{REVISION}/data/train.jsonl",
        "sha256": digest,
        "byte_count": len(data),
        "format": "jsonl",
        "local_path": "/opt/ml/processing/output/" + data_path,
        "artifact_uri": "/opt/ml/processing/output/" + data_path,
        "prepared": False,
        "trained": False,
    }
    encoded_lineage = json.dumps(lineage).encode()
    committed = {
        "data": "data.jsonl",
        "lineage": "lineage.json",
        "sha256": digest,
        "lineage_sha256": hashlib.sha256(encoded_lineage).hexdigest(),
    }
    encoded_commit = json.dumps(committed).encode()
    receipt = {
        "schema_version": 1,
        "job_name": handle.id,
        "runtime_session_id": "session-a",
        "deployment_id": DEPLOYMENT,
        "deployment_prefix": "fixture-lab",
        "image_uri": IMAGE,
        "source_commit": "c" * 40,
        "repo_id": "fixture/open-code",
        "revision": REVISION,
        "filename": "data/train.jsonl",
        "data": data_path,
        "lineage": "acquisition-test/lineage.json",
        "commit_manifest": "acquisition-test/COMMITTED.json",
        "commit_manifest_sha256": hashlib.sha256(encoded_commit).hexdigest(),
    }
    s3.objects[prefix + "ACQUIRED.json"] = json.dumps(receipt).encode()
    s3.objects[prefix + "acquisition-test/lineage.json"] = encoded_lineage
    s3.objects[prefix + "acquisition-test/COMMITTED.json"] = encoded_commit
    s3.objects[prefix + data_path] = data
    s3.heads[prefix + data_path] = {
        "ContentLength": len(data),
        "VersionId": "fixture-version",
        **({"ChecksumSHA256": checksum, "ChecksumType": "FULL_OBJECT"} if checksum else {}),
    }
    return digest


def test_submit_native_tags_and_pinned_image_without_acquired_claim(ownership):
    sm = NativeProcessing()
    backend = adapter(sm=sm)
    handle = submit(backend)
    assert handle.backend == "processing" and handle.arn == sm.arn(handle.id)
    assert handle.extra["accepted"] is True and handle.extra["acquired"] is False
    assert sm.request["AppSpecification"]["ImageUri"] == IMAGE
    assert sm.request["AppSpecification"]["ContainerEntrypoint"] == [
        "/usr/local/bin/amlas-processing-init",
        "acquisition",
    ]
    assert "--revision" in sm.request["AppSpecification"]["ContainerArguments"]
    assert {tag["Key"]: tag["Value"] for tag in sm.request["Tags"]} == {
        "amlas:deployment-id": DEPLOYMENT,
        "amlas:deployment-prefix": "fixture-lab",
        "amlas:runtime-session-id": "session-a",
    }
    assert backend.status(handle).state == JobState.RUNNING
    assert backend.artifacts(handle)["success"] is False


def test_native_acceptance_survives_observation_failure_without_fallback(ownership):
    sm = NativeProcessing(describe_error=True)
    handle = submit(adapter(sm=sm))
    assert handle.arn and handle.extra["accepted"] is True
    assert handle.extra["observation_error"] == "RuntimeError"
    assert handle.extra["acquired"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("role_arn", ""),
        ("image_uri", ""),
        ("session_bucket", ""),
        ("source_commit", ""),
        ("role_arn", "arn:aws:iam::000000000000:role/processing"),
        ("image_uri", IMAGE.replace(REGION, "us-east-1")),
        ("image_uri", IMAGE.split("@")[0] + ":latest"),
    ],
)
def test_missing_or_unpinned_configuration_fails_before_submit(ownership, field, value):
    sm = NativeProcessing()
    config = dict(
        region=REGION,
        role_arn=ROLE,
        image_uri=IMAGE,
        session_bucket="fixture-private-bucket",
        source_commit="c" * 40,
        sagemaker_client=sm,
        s3_client=NativeS3(),
    )
    config[field] = value
    with pytest.raises(ValueError):
        submit(acquisition_backend.ProcessingDatasetAcquisitionBackend(**config))
    assert not sm.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "main"},
        {"filename": "../secret"},
        {"repo_id": "../bad"},
    ],
)
def test_invalid_pin_fails_before_submit(ownership, changes):
    sm = NativeProcessing()
    kwargs = dict(
        repo_id="fixture/open-code",
        revision=REVISION,
        filename="data/train.jsonl",
        resources={"runtime_session_id": "session-a"},
    )
    kwargs.update(changes)
    with pytest.raises(ValueError):
        adapter(sm=sm).submit(**kwargs)
    assert not sm.calls


@pytest.mark.parametrize("missing", ["AMLAS_DEPLOYMENT_ID", "runtime_session_id"])
def test_missing_ownership_fails_before_submit(ownership, monkeypatch, missing):
    sm = NativeProcessing()
    if missing == "AMLAS_DEPLOYMENT_ID":
        monkeypatch.delenv(missing)
    with pytest.raises(ValueError):
        submit(adapter(sm=sm), **({"runtime_session_id": ""} if missing == "runtime_session_id" else {}))
    assert not sm.calls


def test_wrong_bucket_region_fails_before_submit(ownership):
    sm, s3 = NativeProcessing(), NativeS3()
    s3.get_bucket_location = Mock(return_value={"LocationConstraint": "us-east-1"})
    with pytest.raises(ValueError):
        submit(adapter(sm=sm, s3=s3))
    assert not sm.calls


@pytest.mark.parametrize(
    "native_state,expected",
    [
        ("Failed", JobState.FAILED),
        ("Stopped", JobState.STOPPED),
        ("Stopping", JobState.RUNNING),
    ],
)
def test_failed_stopped_jobs_do_not_return_acquired_artifacts(ownership, native_state, expected):
    sm = NativeProcessing(state=native_state)
    backend = adapter(sm=sm)
    handle = submit(backend)
    assert backend.status(handle).state == expected
    result = backend.artifacts(handle)
    assert result["success"] is False and result["acquired"] is False
    assert result["job_name"] == handle.id


def test_completed_job_verifies_bundle_and_returns_s3_without_local_path(ownership):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    digest = populate(handle, s3)
    result = backend.artifacts(handle)
    assert result["success"] is True and result["acquired"] is True
    assert result["artifact_uri"].startswith("s3://fixture-private-bucket/")
    assert result["format"] == "jsonl" and result["sha256"] == digest
    assert result["local_path"] is None and result["provider"] == "processing"
    assert result["integrity"]["payload_sha256_source"] == "processing_container"
    assert result["integrity"]["s3_full_object_sha256_verified"] is False
    assert result["prepared"] is False and result["trained"] is False


@pytest.mark.parametrize(
    "data_path",
    [
        "/etc/passwd",
        "../secret",
        "acquisition-test/../secret",
        "other/data.jsonl",
    ],
)
def test_unconfined_or_mismatched_manifest_path_is_rejected(ownership, data_path):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    populate(handle, s3, data_path=data_path)
    assert backend.artifacts(handle)["success"] is False


def test_tampered_lineage_and_wrong_output_object_size_fail(ownership):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    populate(handle, s3)
    prefix = handle.extra["output_prefix"]
    s3.objects[prefix + "acquisition-test/lineage.json"] = b"{}"
    assert backend.artifacts(handle)["success"] is False
    populate(handle, s3)
    s3.heads[prefix + "acquisition-test/data.jsonl"]["ContentLength"] = 1
    assert backend.artifacts(handle)["success"] is False


def test_completed_missing_commit_remains_failure(ownership):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    populate(handle, s3)
    del s3.objects[handle.extra["output_prefix"] + "acquisition-test/COMMITTED.json"]
    assert backend.artifacts(handle)["success"] is False


def test_registry_managed_default_explicit_offline_and_unknown_no_fallback(ownership, monkeypatch):
    from amlas.backends.ports import DataAcquisitionBackend, port_key
    from amlas.backends.registry import REGISTRY, AdapterUnavailableError

    assert port_key(DataAcquisitionBackend) == "DATA_ACQUISITION_BACKEND"
    assert REGISTRY.resolve(DataAcquisitionBackend).adapter_name == "processing"
    monkeypatch.setenv("AMLAS_OFFLINE", "true")
    assert REGISTRY.resolve(DataAcquisitionBackend).adapter_name == "local"
    monkeypatch.setenv("DATA_ACQUISITION_BACKEND", "missing")
    with pytest.raises(AdapterUnavailableError):
        REGISTRY.resolve(DataAcquisitionBackend)


def test_acquisition_auto_ignores_small_scale_and_low_budget_local_hints(ownership):
    from amlas.backends.ports import DataAcquisitionBackend
    from amlas.backends.registry import REGISTRY

    assert (
        REGISTRY.resolve(DataAcquisitionBackend, scale="small", budget={"prefer": "local"}).adapter_name
        == "processing"
    )


def test_s3_full_object_checksum_matches_and_mismatch_is_rejected(ownership):
    import base64

    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    digest = populate(handle, s3)
    head = s3.heads[handle.extra["output_prefix"] + "acquisition-test/data.jsonl"]
    head.update(ChecksumType="FULL_OBJECT", ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode())
    assert backend.artifacts(handle)["integrity"]["s3_full_object_sha256_verified"] is True
    head["ChecksumSHA256"] = base64.b64encode(b"x" * 32).decode()
    assert backend.artifacts(handle)["success"] is False


def test_native_create_failure_is_not_retried_via_local_acquisition(ownership):
    sm = NativeProcessing()
    sm.create_processing_job = Mock(side_effect=RuntimeError("fixture create failure"))
    with pytest.raises(RuntimeError, match="fixture create failure"):
        submit(adapter(sm=sm))
    assert sm.create_processing_job.call_count == 1


def test_managed_remote_fetch_off_denies_before_any_native_submit(ownership, monkeypatch):
    sm = NativeProcessing()
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "off")
    with pytest.raises(ValueError):
        submit(adapter(sm=sm), AMLAS_AGENT_REMOTE_FETCH="true")
    assert sm.calls == []


def test_managed_trusted_egress_snapshot_is_carried_and_binding_checked(ownership, monkeypatch):
    sm = NativeProcessing(state="Completed")
    s3 = NativeS3()
    monkeypatch.setenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", "huggingface.co,cdn.example")
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend, AMLAS_AGENT_HTTP_ALLOW_HOSTS="model-host.example")
    assert sm.request["Environment"] == {
        "AMLAS_AGENT_REMOTE_FETCH": "1",
        "AMLAS_AGENT_HTTP_ALLOW_HOSTS": "huggingface.co,cdn.example",
    }
    monkeypatch.setenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", "changed-after-submit.example")
    populate(handle, s3)
    assert backend.artifacts(handle)["success"] is True
    sm.request["Environment"] = {"AMLAS_AGENT_REMOTE_FETCH": "1"}
    assert backend.artifacts(handle)["success"] is False


def test_unversioned_bucket_denies_before_submit(ownership):
    sm, s3 = NativeProcessing(), NativeS3()
    s3.get_bucket_versioning = Mock(return_value={})
    with pytest.raises(ValueError):
        submit(adapter(sm=sm, s3=s3))
    assert sm.calls == []


def test_completed_refs_pin_all_versions_and_payload_is_independently_hashed(ownership):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    digest = populate(handle, s3)
    result = backend.artifacts(handle)
    assert result["artifact_ref"]["version_id"] == "fixture-version"
    assert result["artifact_ref"]["sha256"] == digest
    assert result["integrity"]["version_bytes_sha256_verified"] is True
    assert set(result["manifest_refs"]) == {"receipt", "commit", "lineage"}
    assert all(value["version_id"] == "fixture-version" for value in result["manifest_refs"].values())


def test_equal_length_payload_replacement_is_not_acquired(ownership):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    populate(handle, s3)
    key = handle.extra["output_prefix"] + "acquisition-test/data.jsonl"
    s3.objects[key] = b"x" * len(s3.objects[key])
    assert backend.artifacts(handle)["success"] is False


@pytest.mark.parametrize("version", [None, "", "null", "foreign-version"])
def test_missing_null_or_foreign_payload_version_is_not_acquired(ownership, version):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    populate(handle, s3)
    s3.heads[handle.extra["output_prefix"] + "acquisition-test/data.jsonl"]["VersionId"] = version
    assert backend.artifacts(handle)["success"] is False


def test_payload_hash_io_has_fixed_chunks_closes_body_and_honors_deadline(ownership, monkeypatch):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    populate(handle, s3)
    original = s3.get_object
    bodies = []

    class BoundedBody(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            return super().read(size)

    def get_object(**kwargs):
        value = original(**kwargs)
        if kwargs.get("VersionId"):
            body = BoundedBody(value["Body"].read())
            value["Body"].close()
            value["Body"] = body
            bodies.append(body)
        return value

    s3.get_object = get_object
    assert backend.artifacts(handle)["success"] is True
    assert bodies and all(body.closed for body in bodies)
    monkeypatch.setattr(acquisition_backend, "PAYLOAD_HASH_SECONDS", 0, raising=False)
    assert backend.artifacts(handle)["success"] is False


def test_payload_deadline_closes_a_blocked_reader(ownership, monkeypatch):
    import threading

    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    populate(handle, s3)
    original = s3.get_object
    closed = threading.Event()
    released = []

    class BlockedBody:
        def read(self, size):
            released.append(closed.wait(timeout=0.5))
            return b""

        def close(self):
            closed.set()

    def get_object(**kwargs):
        result = original(**kwargs)
        if kwargs.get("VersionId"):
            result["Body"].close()
            result["Body"] = BlockedBody()
        return result

    s3.get_object = get_object
    monkeypatch.setattr(acquisition_backend, "PAYLOAD_HASH_SECONDS", 0.05)
    assert backend.artifacts(handle)["success"] is False
    assert closed.is_set()
    assert released == [True], "Deadline must interrupt the read before final cleanup closes it"


@pytest.mark.parametrize(
    "resources",
    [
        {"volume_size_gb": 0},
        {"volume_size_gb": True},
        {"max_runtime_seconds": 0},
        {"instance_type": "container-shell"},
        {"runtime_session_id": "../another-session"},
    ],
)
def test_invalid_processing_resource_boundaries_do_not_submit(ownership, resources):
    sm = NativeProcessing()
    with pytest.raises(ValueError):
        submit(adapter(sm=sm), **resources)
    assert not sm.calls


def test_completed_native_identity_or_output_substitution_is_not_acquired(ownership):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    populate(handle, s3)
    sm.request["AppSpecification"]["ImageUri"] = IMAGE.replace("a" * 64, "e" * 64)
    assert backend.artifacts(handle)["success"] is False


def test_entrypoint_rewrites_absolute_core_paths_to_confined_relative_receipt(tmp_path, monkeypatch):
    from amlas.sagemaker_entrypoints import dataset_acquisition as entrypoint

    bundle = tmp_path / "acquisition-test"
    bundle.mkdir()
    data = bundle / "data.jsonl"
    data.write_bytes(b'{"messages":[]}\n')
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    lineage = bundle / "lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "repo_id": "fixture/open-code",
                "revision": REVISION,
                "filename": "data/train.jsonl",
                "format": "jsonl",
                "source_uri": f"https://huggingface.co/datasets/fixture/open-code/resolve/{REVISION}/data/train.jsonl",
                "sha256": digest,
                "byte_count": data.stat().st_size,
            }
        )
    )
    committed = bundle / "COMMITTED.json"
    committed.write_text(
        json.dumps(
            {
                "data": "data.jsonl",
                "lineage": "lineage.json",
                "sha256": digest,
                "lineage_sha256": hashlib.sha256(lineage.read_bytes()).hexdigest(),
            }
        )
    )
    core = Mock(
        return_value={
            "success": True,
            "local_path": str(data),
            "lineage_path": str(lineage),
            "commit_manifest_path": str(committed),
        }
    )
    monkeypatch.setattr(entrypoint, "_acquire_local", core, raising=False)
    result = entrypoint.run_acquisition(
        repo_id="fixture/open-code",
        revision=REVISION,
        filename="data/train.jsonl",
        output_dir=str(tmp_path),
        job_name="fixture-job",
        runtime_session_id="session-a",
        deployment_id=DEPLOYMENT,
        deployment_prefix="fixture-lab",
        image_uri=IMAGE,
        source_commit="c" * 40,
    )
    assert result["data"] == "acquisition-test/data.jsonl"
    assert json.loads((tmp_path / "ACQUIRED.json").read_text()) == result
    assert core.call_args.kwargs["output_dir"] == str(tmp_path)


def test_entrypoint_rejects_core_absolute_path_outside_output(tmp_path, monkeypatch):
    from amlas.sagemaker_entrypoints import dataset_acquisition as entrypoint

    monkeypatch.setattr(
        entrypoint,
        "_acquire_local",
        Mock(
            return_value={
                "success": True,
                "local_path": "/etc/passwd",
                "lineage_path": "/etc/passwd",
                "commit_manifest_path": "/etc/passwd",
            }
        ),
        raising=False,
    )
    with pytest.raises(ValueError):
        entrypoint.run_acquisition(
            repo_id="fixture/open-code",
            revision=REVISION,
            filename="data/train.jsonl",
            output_dir=str(tmp_path),
            job_name="fixture-job",
            runtime_session_id="session-a",
            deployment_id=DEPLOYMENT,
            deployment_prefix="fixture-lab",
            image_uri=IMAGE,
            source_commit="c" * 40,
        )
    assert not (tmp_path / "ACQUIRED.json").exists()
