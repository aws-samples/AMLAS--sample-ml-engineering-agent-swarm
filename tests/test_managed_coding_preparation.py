"""Native-shape Processing/S3 fakes; no real service or dataset calls."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

import pytest
from test_coding_preparation_binding import ACCOUNT, BUCKET, IMAGE, catalog, source

from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
from amlas.backends.types import DatasetArtifactRef, JobState

SESSION = "test_session"


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class S3:
    def __init__(self):
        self.values = {}
        self.calls = []
        self.sequence = 0
        self.versioning = "Enabled"
        self.location = "us-west-2"

    def add(self, key, raw, version=None):
        self.sequence += 1
        version = version or "v" + str(self.sequence)
        self.values[key] = (raw, version)
        return version

    def get_bucket_location(self, **kw):
        self.calls.append(("location", kw))
        assert kw["ExpectedBucketOwner"] == ACCOUNT
        return {"LocationConstraint": self.location}

    def get_bucket_versioning(self, **kw):
        self.calls.append(("versioning", kw))
        assert kw["ExpectedBucketOwner"] == ACCOUNT
        return {"Status": self.versioning}

    def head_object(self, **kw):
        self.calls.append(("head", kw))
        assert kw["ExpectedBucketOwner"] == ACCOUNT
        raw, version = self.values[kw["Key"]]
        if "VersionId" in kw:
            assert kw["VersionId"] == version
        return {"ContentLength": len(raw), "VersionId": version}

    def get_object(self, **kw):
        self.calls.append(("get", kw))
        assert kw["ExpectedBucketOwner"] == ACCOUNT
        raw, version = self.values[kw["Key"]]
        assert kw["VersionId"] == version
        return {"ContentLength": len(raw), "VersionId": version, "Body": io.BytesIO(raw)}

    def copy_object(self, **kw):
        self.calls.append(("copy", kw))
        assert kw["ExpectedBucketOwner"] == kw["ExpectedSourceBucketOwner"] == ACCOUNT
        original = kw["CopySource"]
        raw, version = self.values[original["Key"]]
        assert original["VersionId"] == version
        return {"VersionId": self.add(kw["Key"], raw)}

    def put_object(self, **kw):
        self.calls.append(("put", kw))
        assert kw["ExpectedBucketOwner"] == ACCOUNT
        return {"VersionId": self.add(kw["Key"], kw["Body"])}

    def list_objects_v2(self, **kw):
        self.calls.append(("list", kw))
        assert kw["ExpectedBucketOwner"] == ACCOUNT
        values = [
            {"Key": key, "Size": len(raw)}
            for key, (raw, _version) in self.values.items()
            if key.startswith(kw["Prefix"])
        ]
        return {"Contents": values[: kw["MaxKeys"]], "IsTruncated": len(values) > kw["MaxKeys"]}


class SM:
    def __init__(self):
        self.creates = []
        self.state = "InProgress"
        self.read_error = False
        self.drift = False

    def create_processing_job(self, **kw):
        # Native request shape: no Lambda-only/SageMaker SDK convenience fields.
        assert set(kw) == {
            "ProcessingJobName",
            "RoleArn",
            "AppSpecification",
            "ProcessingResources",
            "ProcessingInputs",
            "ProcessingOutputConfig",
            "StoppingCondition",
            "NetworkConfig",
            "Tags",
        }
        assert kw["NetworkConfig"]["EnableNetworkIsolation"] is True
        names = [item["InputName"] for item in kw["ProcessingInputs"]]
        names += [item["OutputName"] for item in kw["ProcessingOutputConfig"]["Outputs"]]
        assert len(names) == len(set(names)), "native Processing input/output names must be globally unique"
        assert kw["AppSpecification"]["ContainerEntrypoint"] == [
            "/usr/local/bin/amlas-processing-init",
            "preparation",
        ]
        self.creates.append(kw)
        return {
            "ProcessingJobArn": f"arn:aws:sagemaker:us-west-2:{ACCOUNT}:processing-job/"
            + kw["ProcessingJobName"]
        }

    def describe_processing_job(self, **kw):
        if self.read_error:
            raise RuntimeError("native read unavailable")
        request = self.creates[0]
        assert kw["ProcessingJobName"] == request["ProcessingJobName"]
        value = {
            **request,
            "ProcessingJobArn": f"arn:aws:sagemaker:us-west-2:{ACCOUNT}:processing-job/"
            + kw["ProcessingJobName"],
            "ProcessingJobStatus": self.state,
        }
        value.pop("Tags")
        if self.drift:
            value["NetworkConfig"] = {"EnableNetworkIsolation": False}
        return value


@pytest.fixture
def fixture(monkeypatch):
    # Test-only immutable owner dependency; production has no path/provider selector.
    owner_path = os.environ.get("AMLAS_TEST_PREPARATION_NATIVE_SCHEMA")
    if owner_path:
        spec = importlib.util.spec_from_file_location("owned_native_schema", Path(owner_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(
            "amlas.backends.adapters.coding_preparation._validate_native_proof",
            module.validate_native_privilege_proof,
        )
    s3, sm = S3(), SM()
    original = source(SESSION)
    data = b"".join(
        encoded(
            {
                "repository": f"owner/repo-{index}",
                "instance_id": f"case-{index}",
                "trajectory_id": f"trajectory-{index}",
                "license": "MIT",
                "resolved": 1,
                "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}],
                "tools": [],
            }
        )
        for index in range(4)
    )
    ref = dataclasses.replace(original.inspection_source.artifact_ref, sha256=sha(data), byte_count=len(data))
    original = dataclasses.replace(
        original, inspection_source=dataclasses.replace(original.inspection_source, artifact_ref=ref)
    )
    s3.add(ref.uri.split("/", 3)[3], data, ref.version_id)
    lineage = encoded(
        {
            "sha256": ref.sha256,
            "byte_count": ref.byte_count,
            "format": "jsonl",
            "repo_id": original.repo_id,
            "revision": original.revision,
            "filename": original.filename,
        }
    )
    commit = encoded({"sha256": ref.sha256, "lineage_sha256": sha(lineage)})
    receipt = encoded(
        {
            "repo_id": original.repo_id,
            "revision": original.revision,
            "filename": original.filename,
            "image_uri": original.acquisition_image_uri,
            "source_commit": original.acquisition_source_commit,
            "commit_manifest_sha256": sha(commit),
        }
    )
    refs = []
    for name, raw in (("receipt", receipt), ("commit", commit), ("lineage", lineage)):
        uri = f"s3://{BUCKET}/sessions/{SESSION}/acquisition/job/" + name + ".json"
        version = name + "-v1"
        s3.add(uri.split("/", 3)[3], raw, version)
        refs.append((name, uri, version, sha(raw)))
    original = dataclasses.replace(original, manifest_refs=tuple(refs))
    excluded = encoded({"instance_ids": [], "patch_sha256": []})
    selected = catalog(
        exclusions_ref=DatasetArtifactRef(
            uri=f"s3://{BUCKET}/exclusions/cases.json",
            version_id="excluded-v1",
            sha256=sha(excluded),
            byte_count=len(excluded),
            format="json",
        )
    )
    s3.add("exclusions/cases.json", excluded, "excluded-v1")
    return ProcessingCodingPreparationBackend(sagemaker_client=sm, s3_client=s3), sm, s3, selected, original


def submit(fixture):
    backend, _sm, _s3, selected, original = fixture
    return backend.submit(sources=(original,), catalog=selected, runtime_session_id=SESSION)


def test_submit_uses_fixed_native_shape_exact_version_staging_and_atomic_tags(fixture):
    handle = submit(fixture)
    _, sm, s3, _, _ = fixture
    assert handle.arn and len(sm.creates) == 1
    request = sm.creates[0]
    assert request["AppSpecification"]["ImageUri"] == IMAGE
    assert request["NetworkConfig"]["EnableNetworkIsolation"] is True
    assert request["ProcessingResources"]["ClusterConfig"]["InstanceCount"] == 1
    assert {tag["Key"] for tag in request["Tags"]} == {
        "amlas:deployment-id",
        "amlas:deployment-prefix",
        "amlas:runtime-session-id",
    }
    assert all(call[1].get("VersionId") for call in s3.calls if call[0] == "get")
    assert all(call[1]["CopySource"].get("VersionId") for call in s3.calls if call[0] == "copy")
    assert fixture[0].artifacts(handle)["prepared"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"admission_digest": ""},
        {"source_commit": "not-pinned"},
        {"image_uri": IMAGE.replace("us-west-2", "us-east-1")},
        {"max_runtime_seconds": True},
    ],
)
def test_invalid_catalog_refuses_before_any_client_call(fixture, changes):
    backend, sm, s3, selected, original = fixture
    with pytest.raises(ValueError):
        backend.submit(
            sources=(original,), catalog=dataclasses.replace(selected, **changes), runtime_session_id=SESSION
        )
    assert sm.creates == [] and s3.calls == []


def test_versioning_and_region_refusal_precede_artifact_mutations(fixture):
    fixture[2].versioning = "Suspended"
    with pytest.raises(ValueError):
        submit(fixture)
    assert fixture[1].creates == [] and not any(name in ("copy", "put") for name, _ in fixture[2].calls)


def test_source_equal_length_tamper_never_stages_or_creates(fixture):
    key = fixture[4].inspection_source.artifact_ref.uri.split("/", 3)[3]
    raw, version = fixture[2].values[key]
    fixture[2].values[key] = (raw.replace(b"answer", b"ANSWER"), version)
    with pytest.raises(ValueError):
        submit(fixture)
    assert fixture[1].creates == []


def test_accepted_job_read_failure_is_retained_without_resubmit(fixture):
    fixture[1].read_error = True
    handle = submit(fixture)
    assert handle.arn and len(fixture[1].creates) == 1
    result = fixture[0].artifacts(handle)
    assert result["prepared"] is False and result["accepted"] is True
    assert len(fixture[1].creates) == 1


@pytest.mark.parametrize("native", ["Failed", "Stopped", "InProgress"])
def test_noncompleted_job_never_becomes_prepared(fixture, native):
    handle = submit(fixture)
    fixture[1].state = native
    assert fixture[0].artifacts(handle)["prepared"] is False
    assert len(fixture[1].creates) == 1


def test_native_network_drift_is_observation_failure_not_ready(fixture):
    handle = submit(fixture)
    fixture[1].state = "Completed"
    fixture[1].drift = True
    assert fixture[0].status(handle).state == JobState.UNKNOWN
    assert fixture[0].artifacts(handle)["prepared"] is False


def complete_outputs(fixture, handle):
    from test_coding_preparation_worker import PROOF

    _backend, sm, s3, _selected, _source = fixture
    args = sm.creates[0]["AppSpecification"]["ContainerArguments"]
    cli = {name[2:].replace("-", "_"): value for name, value in zip(args[::2], args[1::2], strict=True)}
    prefix = sm.creates[0]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"].split("/", 3)[3]
    counts = {
        "source_count": 4,
        "retained_representatives": 4,
        "primary_drop_count": 0,
        "unresolved_record_count": 0,
        "split_counts": {"train": 2, "validation": 1, "test": 1},
    }
    coverage = {
        "near_dedup": "not_implemented",
        "exclusion_scope": "supplied_instance_ids_and_patch_digests_only",
    }
    payloads = {
        "train.jsonl": b"{}\n{}\n",
        "validation.jsonl": b"{}\n",
        "test.jsonl": b"{}\n",
        "membership.jsonl": b"{}\n{}\n{}\n{}\n",
        "lineage.jsonl": b"{}\n{}\n{}\n{}\n",
        "REPORT.json": encoded(
            {
                "schema_version": 1,
                "status": "complete",
                "counts": counts,
                "primary_drop_counts": {},
                "secret_pattern_counts": {},
                "coverage": coverage,
            }
        ),
    }
    files = {}
    for name, raw in payloads.items():
        files[name] = {
            "byte_count": len(raw),
            "sha256": sha(raw),
            "format": "json" if name.endswith(".json") else "jsonl",
            "record_count": len(raw.splitlines()) if name.endswith(".jsonl") else 1,
        }
        s3.add(prefix + name, raw)
    header = {
        name: cli[name]
        for name in (
            "job_name",
            "runtime_session_id",
            "deployment_id",
            "deployment_prefix",
            "image_uri",
            "source_commit",
            "profile_id",
            "policy_version",
        )
    }
    committed = {
        "schema_version": 2,
        **header,
        "privilege_profile": "sagemaker_native_ceiling_v1",
        "selection_sha256": cli["selection_sha256"],
        "exclusions_sha256": cli["exclusions_sha256"],
        "files": files,
        "counts": counts,
        "coverage": coverage,
        "guest_evidence": {
            "privilege_proof": PROOF,
            "credential_observation": "env_and_home_file_presence",
            "network_observation": "fixed_public_and_metadata_tcp_denial",
        },
    }
    s3.add(prefix + "COMMITTED.json", encoded(committed))
    sm.state = "Completed"
    return prefix, committed


def test_completed_outputs_return_only_verified_version_bound_refs(fixture):
    handle = submit(fixture)
    _prefix, _commit = complete_outputs(fixture, handle)
    result = fixture[0].artifacts(handle)
    assert result["success"] is True and result["prepared"] is True and result["trained"] is False
    assert set(result["split_refs"]) == {"train", "validation", "test"}
    for ref in [result["preparation_ref"], *result["split_refs"].values(), *result["sidecar_refs"].values()]:
        assert set(ref) == {"uri", "version_id", "sha256", "byte_count", "format"}
        assert ref["version_id"] not in (None, "", "null")
    before = len([name for name, _ in fixture[2].calls if name == "put"])
    fixture[0].artifacts(handle)
    assert len([name for name, _ in fixture[2].calls if name == "put"]) == before
    assert len(fixture[1].creates) == 1


@pytest.mark.parametrize(
    "mutation", ["source", "profile", "proof", "count", "extra", "missing", "equal_length", "null_version"]
)
def test_tampered_or_incomplete_completed_outputs_never_prepare(fixture, mutation):
    handle = submit(fixture)
    prefix, committed = complete_outputs(fixture, handle)
    s3 = fixture[2]
    if mutation == "source":
        committed["source_commit"] = "c" * 40
    elif mutation == "profile":
        committed["privilege_profile"] = "model_widened_profile"
    elif mutation == "proof":
        committed["guest_evidence"]["privilege_proof"] = {"cap_eff": 0}
    elif mutation == "count":
        committed["counts"]["source_count"] = 3
    elif mutation == "extra":
        s3.add(prefix + "unexpected.json", b"{}")
    elif mutation == "missing":
        s3.values.pop(prefix + "test.jsonl")
    elif mutation == "equal_length":
        raw, version = s3.values[prefix + "train.jsonl"]
        s3.values[prefix + "train.jsonl"] = (raw.replace(b"{}", b"[]", 1), version)
    elif mutation == "null_version":
        raw, _version = s3.values[prefix + "lineage.jsonl"]
        s3.values[prefix + "lineage.jsonl"] = (raw, "null")
    if mutation in ("source", "profile", "proof", "count"):
        s3.add(prefix + "COMMITTED.json", encoded(committed))
    result = fixture[0].artifacts(handle)
    assert result["prepared"] is False and result["success"] is False
    assert len(fixture[1].creates) == 1
