"""Preparation gets one private verified snapshot, never caller-authored authority."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import pytest

from amlas.backends.types import JobHandle, JobState, JobStatus
from amlas.control_plane.acquisition_state import SessionAcquisitionStore


def record():
    store = SessionAcquisitionStore()
    account, region, bucket = "1234" * 3, "us-west-2", "fixture-private-bucket"
    job = "fixture-data-acq"
    output = f"s3://{bucket}/sessions/{store.runtime_session_id}/acquisition/{job}/"
    handle = JobHandle(
        id=job,
        backend="processing",
        arn=f"arn:aws:sagemaker:{region}:{account}:processing-job/{job}",
        extra={
            "account_id": account,
            "region": region,
            "bucket": bucket,
            "runtime_session_id": store.runtime_session_id,
            "deployment_id": "fd0d82ab-4e03-44c0-9bf1-908af5b2b031",
            "deployment_prefix": "fixture",
            "output_uri": output,
            "repo_id": "fixture/open-code",
            "revision": "a" * 40,
            "filename": "data/train.jsonl",
            "image_uri": f"{account}.dkr.ecr.{region}.amazonaws.com/processing@sha256:" + "b" * 64,
            "source_commit": "c" * 40,
        },
    )
    result = {
        "success": True,
        "acquired": True,
        "artifact_ref": {
            "uri": output + "bundle/data.jsonl",
            "version_id": "payload-version",
            "sha256": "d" * 64,
            "byte_count": 24,
            "format": "jsonl",
        },
        "manifest_refs": {
            name: {
                "uri": output + path,
                "version_id": name + "-version",
                "sha256": char * 64,
            }
            for name, path, char in [
                ("receipt", "ACQUIRED.json", "e"),
                ("commit", "bundle/COMMITTED.json", "f"),
                ("lineage", "bundle/lineage.json", "a"),
            ]
        },
    }

    class Acquisition:
        backend_name = "processing"
        calls = 0

        def submit(self, **kwargs):
            raise AssertionError("snapshot cannot submit")

        def status(self, handle):
            return JobStatus(JobState.SUCCEEDED)

        def artifacts(self, handle):
            self.calls += 1
            return copy.deepcopy(result)

    adapter = Acquisition()
    return store, store.add(adapter, handle), adapter, handle, result


def accessor(store):
    method = getattr(store, "_verified_preparation_source", None)
    assert callable(method), "missing-symbol: private verified preparation source accessor"
    return method


def test_private_source_has_one_fresh_observation_original_pins_and_no_grants():
    store, reference, adapter, handle, result = record()
    result.update(repo_id="spoof/foreign", revision="0" * 40, filename="spoof.jsonl")
    snapshot = accessor(store)(reference)
    assert adapter.calls == 1
    assert snapshot.repo_id == handle.extra["repo_id"]
    assert snapshot.revision == handle.extra["revision"]
    assert snapshot.filename == handle.extra["filename"]
    assert snapshot.acquisition_image_uri == handle.extra["image_uri"]
    assert snapshot.acquisition_source_commit == handle.extra["source_commit"]
    assert snapshot.inspection_source.artifact_ref.version_id == "payload-version"
    assert snapshot.manifest_refs == tuple(
        (name, fields["uri"], fields["version_id"], fields["sha256"])
        for name, fields in sorted(result["manifest_refs"].items())
    )
    assert "policy" not in vars(snapshot) and "approval" not in vars(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.repo_id = "changed/source"


@pytest.mark.parametrize("reference", [None, {}, "acq_" + "0" * 32, "native-job-arn"])
def test_caller_reference_cannot_construct_or_route_a_native_record(reference):
    store, _, adapter, _, _ = record()
    with pytest.raises(ValueError):
        accessor(store)(reference)
    assert adapter.calls == 0


def test_reference_is_local_to_its_conversation():
    _, reference, adapter, _, _ = record()
    with pytest.raises(ValueError):
        accessor(SessionAcquisitionStore())(reference)
    assert adapter.calls == 0


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("receipt", "version_id", "null"),
        ("commit", "version_id", ""),
        ("lineage", "sha256", "incorrect"),
        ("receipt", "uri", "s3://foreign-private-bucket/ACQUIRED.json"),
        ("commit", "uri", "s3://fixture-private-bucket/other-job/COMMITTED.json"),
        ("lineage", "uri", "https://example.com/lineage.json"),
    ],
)
def test_manifests_need_exact_versions_hashes_and_owned_job_scope(name, field, value):
    store, reference, adapter, _, result = record()
    result["manifest_refs"][name][field] = value
    with pytest.raises(ValueError):
        accessor(store)(reference)
    assert adapter.calls == 1


@pytest.mark.parametrize("field", ["manifest_refs", "artifact_ref"])
def test_missing_verified_bytes_do_not_become_preparation_input(field):
    store, reference, adapter, _, result = record()
    result.pop(field)
    with pytest.raises(ValueError):
        accessor(store)(reference)
    assert adapter.calls == 1


@pytest.mark.parametrize("field", ["success", "acquired"])
def test_pending_or_failed_acquisition_cannot_supply_a_source(field):
    store, reference, adapter, _, result = record()
    result[field] = False
    with pytest.raises(ValueError):
        accessor(store)(reference)
    assert adapter.calls == 1


def test_wrong_private_session_refuses_before_native_observation():
    store, reference, adapter, handle, _ = record()
    handle.extra["runtime_session_id"] = "foreign-session"
    with pytest.raises(ValueError):
        accessor(store)(reference)
    assert adapter.calls == 0
