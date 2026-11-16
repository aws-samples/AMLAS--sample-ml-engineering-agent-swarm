"""Synthetic byte streams and closed SDK providers; never real model weights."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CATALOG = "qwen3_5_9b"
VERSION = "1"
ACCOUNT = "1234" * 3
REGION = "us-west-2"
BUCKET = "fixture-private-model-bucket"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/fixture-model-fetch"
IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/model-fetch@sha256:" + "a" * 64
COMMIT = "b" * 40
DEPLOYMENT = "fd0d82ab-4e03-44c0-9bf1-908af5b2b031"
SESSION = "model-session-a"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fixture_manifest() -> tuple[dict[str, Any], dict[str, bytes]]:
    from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest

    value = expected_model_weight_manifest(CATALOG, VERSION)
    bodies = {item["path"]: ("closed synthetic bytes " + item["path"]).encode() for item in value["files"]}
    for item in value["files"]:
        item["byte_count"] = len(bodies[item["path"]])
        item["sha256"] = digest(bodies[item["path"]])
    value["weights_total_bytes"] = sum(
        item["byte_count"] for item in value["files"] if item["role"] == "weight_shard"
    )
    return value, bodies


def require_module(path: str, *symbols: str) -> Any:
    module = importlib.import_module(path)
    for symbol in symbols:
        assert callable(getattr(module, symbol, None)), "missing-symbol: " + symbol
    return module


def patch_expected(monkeypatch: Any, modules: list[Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    value, bodies = fixture_manifest()
    for module in modules:
        monkeypatch.setattr(module, "expected_model_weight_manifest", lambda *args: deepcopy(value))
    return value, bodies


def bindings() -> dict[str, str]:
    return {
        "job_name": "fixture-model-fetch",
        "runtime_session_id": SESSION,
        "deployment_id": DEPLOYMENT,
        "deployment_prefix": "fixture-lab",
        "image_uri": IMAGE,
        "source_commit": COMMIT,
    }


def evidence() -> dict[str, Any]:
    return {
        "privilege_profile": "sagemaker_native_ceiling_v1",
        "privilege_proof": {
            "res_uid": [10001] * 3,
            "res_gid": [10001] * 3,
            "groups": [],
            "cap_eff": 0,
            "cap_prm": 0,
            "cap_inh": 0,
            "cap_amb": 0,
            "cap_bnd": 0x200404FB,
            "no_new_privs": 1,
            "keep_caps": 0,
            "seccomp": 2,
            "fs_uid": 10001,
            "fs_gid": 10001,
            "securebits": 0,
        },
        "output_directory_proof": {
            "fd": 3,
            "uid": 10001,
            "gid": 10001,
            "mode": 0o700,
            "device": 1,
            "inode": 1,
            "lease_active": True,
        },
        "credential_observation": {
            "environment_names": [],
            "provider_files_present": False,
            "metadata_endpoint_reachable": False,
        },
        "guest_claim_attested": False,
    }


class Processing:
    def __init__(self) -> None:
        self.meta = SimpleNamespace(region_name=REGION)
        self.requests: list[dict[str, Any]] = []
        self.state = "InProgress"
        self.overrides: dict[str, Any] = {}
        self.read_failure = False
        self.ambiguous_submit = False

    def create_processing_job(self, **kwargs: Any) -> dict[str, str]:
        from botocore.session import Session
        from botocore.validate import validate_parameters

        validate_parameters(
            kwargs,
            Session().get_service_model("sagemaker").operation_model("CreateProcessingJob").input_shape,
        )
        self.requests.append(deepcopy(kwargs))
        if self.ambiguous_submit:
            raise TimeoutError("closed ambiguous provider result")
        return {"ProcessingJobArn": self.arn(kwargs["ProcessingJobName"])}

    def arn(self, name: str) -> str:
        return f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/{name}"

    def describe_processing_job(self, **kwargs: Any) -> dict[str, Any]:
        if self.read_failure:
            raise RuntimeError("closed provider read failure")
        request = self.requests[-1]
        assert kwargs["ProcessingJobName"] == request["ProcessingJobName"]
        return {
            **deepcopy(request),
            "ProcessingJobArn": self.arn(request["ProcessingJobName"]),
            "ProcessingJobStatus": self.state,
            **self.overrides,
        }


class S3:
    def __init__(self) -> None:
        self.meta = SimpleNamespace(region_name=REGION)
        self.objects: dict[str, tuple[str, bytes]] = {}
        self.reads: list[dict[str, Any]] = []
        self.versioning = "Enabled"
        self.extra_key = False
        self.full_checksums = False
        self.version_mismatch = False

    def get_bucket_location(self, **kwargs: Any) -> dict[str, str]:
        assert kwargs["ExpectedBucketOwner"] == ACCOUNT
        return {"LocationConstraint": REGION}

    def get_bucket_versioning(self, **kwargs: Any) -> dict[str, str]:
        return {"Status": self.versioning}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        result = [
            {"Key": key, "Size": len(value[1])}
            for key, value in self.objects.items()
            if key.startswith(kwargs["Prefix"])
        ]
        if self.extra_key:
            result.append({"Key": kwargs["Prefix"] + "extra.bin", "Size": 1})
        return {"Contents": result, "IsTruncated": False, "KeyCount": len(result)}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        version, raw = self.objects[kwargs["Key"]]
        if "VersionId" in kwargs:
            assert kwargs["VersionId"] == version
        result: dict[str, Any] = {"VersionId": version, "ContentLength": len(raw)}
        if self.full_checksums:
            import base64

            result.update(
                ChecksumType="FULL_OBJECT",
                ChecksumSHA256=base64.b64encode(hashlib.sha256(raw).digest()).decode(),
            )
        return result

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.reads.append(kwargs)
        version, raw = self.objects[kwargs["Key"]]
        assert kwargs.get("VersionId") == version, "model parent used mutable latest GET"
        return {
            "VersionId": "foreign-version" if self.version_mismatch else version,
            "ContentLength": len(raw),
            "Body": io.BytesIO(raw),
        }


def configured_backend(module: Any, sm: Any, s3: Any, **overrides: Any) -> Any:
    settings = {
        "region": REGION,
        "role_arn": ROLE,
        "image_uri": IMAGE,
        "session_bucket": BUCKET,
        "source_commit": COMMIT,
        "admission_digest": "a" * 64,
        "sagemaker_client": sm,
        "s3_client": s3,
    }
    settings.update(overrides)
    return module.ProcessingModelWeightAcquisitionBackend(**settings)


def publish_fixture(monkeypatch: Any, tmp_path: Path, adapter: Any, handle: Any) -> None:
    core = require_module("amlas.data_eng.model_weight_acquisition", "acquire_model_into_fd")
    schema = importlib.import_module("amlas.data_eng.model_acquisition_schema")
    value, bodies = patch_expected(monkeypatch, [core, schema, importlib.import_module(adapter.__module__)])
    monkeypatch.setattr(
        core,
        "stream_model_catalog_file",
        lambda name, sink, **kwargs: (
            sink.write(bodies[name]),
            {"byte_count": len(bodies[name]), "sha256": digest(bodies[name])},
        )[1],
    )
    output = tmp_path / "managed-output"
    output.mkdir(mode=0o700)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        core.acquire_model_into_fd(
            catalog_id=CATALOG,
            catalog_version=VERSION,
            output_fd=fd,
            output_directory=output,
            bindings={**bindings(), "job_name": handle.id},
            evidence=evidence(),
        )
    finally:
        os.close(fd)
    for path in output.iterdir():
        adapter._s3.objects[handle.extra["output_prefix"] + path.name] = (
            path.name + "-version",
            path.read_bytes(),
        )
    assert value["repo_id"] == "Qwen/Qwen3.5-9B"
