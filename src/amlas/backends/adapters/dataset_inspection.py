"""Fixed managed inspection orchestration; the agent process never parses inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit

from ...config.aws_region import resolve_aws_client_region, resolve_aws_region
from ...config.deployment import deployment_job_tags, resolve_deployment_prefix
from ...config.externalized import resolve
from ...data_eng.inspection_schema import (
    FORMATS,
    MAX_INPUT_BYTES,
    MAX_RECEIPT_BYTES,
    MAX_SUMMARY_BYTES,
    NATIVE_CEILING_PROFILE,
    SHA256,
    ZERO_BOUNDING_PROFILE,
    decode_receipt,
    decode_summary,
    target_text,
)
from ..ports import DataInspectionBackend
from ..registry import register_adapter
from ..types import DatasetArtifactRef, DatasetInspectionSource, JobHandle, JobState, JobStatus


def _version(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value == "null"
        or len(value) > 1024
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("a non-null immutable object version is required")
    return value


def _source(source: Any) -> tuple[str, str]:
    if not isinstance(source, DatasetInspectionSource) or not isinstance(
        source.artifact_ref, DatasetArtifactRef
    ):
        raise TypeError("inspection requires a private typed acquired source")
    ref = source.artifact_ref
    parsed = urlsplit(ref.uri)
    if (
        parsed.scheme != "s3"
        or parsed.netloc != source.bucket
        or not parsed.path[1:]
        or parsed.query
        or parsed.fragment
        or "\\" in ref.uri
        or any(ord(char) < 32 for char in ref.uri)
        or not re.fullmatch(r"[0-9]{12}", source.account_id)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", source.runtime_session_id)
        or ref.format not in FORMATS
        or not SHA256.fullmatch(ref.sha256)
        or type(ref.byte_count) is not int
        or not 1 <= ref.byte_count <= MAX_INPUT_BYTES
    ):
        raise ValueError("invalid acquired source snapshot")
    _version(ref.version_id)
    if source.acquisition_job_arn != (
        f"arn:aws:sagemaker:{source.region}:{source.account_id}:processing-job/"
        + source.acquisition_job_arn.rsplit("/", 1)[-1]
    ):
        raise ValueError("acquisition native identity differs from its trusted snapshot")
    return parsed.netloc, parsed.path[1:]


def _native_field(name: str, value: Any) -> str:
    """Normalize only documented optional fixed-job defaults, without mutation."""
    if name == "NetworkConfig" and isinstance(value, dict):
        value = dict(value)
        if value.get("EnableInterContainerTrafficEncryption") is False:
            value.pop("EnableInterContainerTrafficEncryption")
    elif name == "ProcessingInputs" and isinstance(value, list):
        value = [dict(item) if isinstance(item, dict) else item for item in value]
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("S3Input"), dict):
                item["S3Input"] = dict(item["S3Input"])
                if item["S3Input"].get("S3CompressionType") == "None":
                    item["S3Input"].pop("S3CompressionType")
                if item["S3Input"].get("S3DataDistributionType") == "FullyReplicated":
                    item["S3Input"].pop("S3DataDistributionType")
    # Canonical JSON also distinguishes native booleans/integers/floats.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ProcessingDatasetInspectionBackend:
    """Stage pinned bytes and consume only two verified bounded JSON outputs."""

    backend_name = "processing"

    privilege_profile = ZERO_BOUNDING_PROFILE

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str | None = None,
        image_uri: str | None = None,
        session_bucket: str | None = None,
        source_commit: str | None = None,
        admission_digest: str | None = None,
        sagemaker_client: Any = None,
        s3_client: Any = None,
    ) -> None:
        self._region, self._role, self._image = region, role_arn, image_uri
        self._bucket, self._source, self._admission = session_bucket, source_commit, admission_digest
        self._sm, self._s3 = sagemaker_client, s3_client

    def _clients(self, region: str) -> tuple[Any, Any]:
        if self._sm is None or self._s3 is None:
            import boto3

            session = boto3.Session(region_name=region)
            self._sm = self._sm or session.client("sagemaker")
            self._s3 = self._s3 or session.client("s3")
        resolve_aws_client_region(self._sm, region)
        resolve_aws_client_region(self._s3, region)
        return self._sm, self._s3

    @staticmethod
    def _hash_object(s3: Any, bucket: str, key: str, version: str, account: str, size: int) -> str:
        response = s3.get_object(
            Bucket=bucket,
            Key=key,
            VersionId=_version(version),
            ExpectedBucketOwner=account,
        )
        body = response["Body"]
        if response.get("VersionId") != version or response.get("ContentLength") != size:
            body.close()
            raise ValueError("version-bound source byte count mismatch")
        expired = threading.Event()
        timer = threading.Timer(120.0, lambda: (expired.set(), body.close()))
        timer.daemon = True
        digest, count = hashlib.sha256(), 0
        deadline = time.monotonic() + 120.0
        timer.start()
        try:
            while True:
                chunk = body.read(min(1024 * 1024, size - count + 1))
                if expired.is_set() or time.monotonic() >= deadline:
                    raise ValueError("version-bound source hash deadline exceeded")
                if not chunk:
                    break
                count += len(chunk)
                if count > size:
                    raise ValueError("source hash byte budget exceeded")
                digest.update(chunk)
            if count != size:
                raise ValueError("version-bound source was truncated")
            return digest.hexdigest()
        finally:
            timer.cancel()
            body.close()
            timer.join(timeout=0.1)

    @staticmethod
    def _keys(s3: Any, bucket: str, prefix: str, account: str) -> list[dict[str, Any]]:
        result = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=3,
            ExpectedBucketOwner=account,
        )
        if result.get("IsTruncated"):
            raise ValueError("inspection prefix contains extra objects")
        return result.get("Contents", [])

    def submit(self, *, source: DatasetInspectionSource, target_column: str | None = None) -> JobHandle:
        bucket, key = _source(source)
        target = target_text(target_column)
        region = resolve_aws_region(self._region)
        role = (
            self._role if self._role is not None else resolve("sagemaker_execution_role_arn", region=region)
        )
        image = (
            self._image if self._image is not None else resolve("data_inspection_image_uri", region=region)
        )
        configured_bucket = (
            self._bucket if self._bucket is not None else resolve("session_bucket", region=region)
        )
        commit = (
            self._source
            if self._source is not None
            else resolve("data_inspection_source_commit", region=region)
        )
        marker = (
            self._admission
            if self._admission is not None
            else resolve("data_inspection_admission_digest", region=region)
        )
        role_match = re.fullmatch(r"arn:aws:iam::([0-9]{12}):role/[A-Za-z0-9+=,.@_/-]+", role)
        image_match = re.fullmatch(
            r"([0-9]{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com/"
            r"[a-z0-9][a-z0-9._/-]*@sha256:([a-f0-9]{64})",
            image,
        )
        if (
            not role_match
            or role_match[1] != source.account_id
            or source.account_id == "0" * 12
            or not image_match
            or image_match[1] != source.account_id
            or image_match[2] != region
            or source.region != region
            or configured_bucket != bucket
            or not re.fullmatch(r"[a-f0-9]{40}", commit)
        ):
            raise ValueError("fixed inspection deployment binding mismatch")
        if not isinstance(marker, str) or marker != image_match[3]:
            raise ValueError("inspection image lacks matching native admission")
        if source.deployment_prefix != resolve_deployment_prefix() or source.deployment_id != os.environ.get(
            "AMLAS_DEPLOYMENT_ID"
        ):
            raise ValueError("inspection source belongs to another deployment")
        tags = deployment_job_tags(source.runtime_session_id)
        if not tags:
            raise ValueError("inspection requires deployment ownership")
        sm, s3 = self._clients(region)
        location = (
            s3.get_bucket_location(Bucket=bucket, ExpectedBucketOwner=source.account_id).get(
                "LocationConstraint"
            )
            or "us-east-1"
        )
        if (
            location != region
            or s3.get_bucket_versioning(Bucket=bucket, ExpectedBucketOwner=source.account_id).get("Status")
            != "Enabled"
        ):
            raise ValueError("inspection requires a same-region versioned bucket")
        ref = source.artifact_ref
        head = s3.head_object(
            Bucket=bucket, Key=key, VersionId=ref.version_id, ExpectedBucketOwner=source.account_id
        )
        if head.get("VersionId") != ref.version_id or head.get("ContentLength") != ref.byte_count:
            raise ValueError("acquired source version or size changed")
        if (
            self._hash_object(s3, bucket, key, ref.version_id, source.account_id, ref.byte_count)
            != ref.sha256
        ):
            raise ValueError("acquired source hash mismatch")
        job = f"{source.deployment_prefix}-data-inspect-{uuid.uuid4().hex[:12]}"
        prefix = f"sessions/{source.runtime_session_id}/inspection/{job}/"
        input_prefix, output_prefix = prefix + "input/", prefix + "output/"
        input_key = input_prefix + "data." + ref.format
        copied = s3.copy_object(
            Bucket=bucket,
            Key=input_key,
            CopySource={"Bucket": bucket, "Key": key, "VersionId": ref.version_id},
            ExpectedSourceBucketOwner=source.account_id,
            ExpectedBucketOwner=source.account_id,
        )
        staged_version = _version(copied.get("VersionId"))
        if (
            self._hash_object(s3, bucket, input_key, staged_version, source.account_id, ref.byte_count)
            != ref.sha256
        ):
            raise ValueError("staged source hash mismatch")
        listed = self._keys(s3, bucket, input_prefix, source.account_id)
        if len(listed) != 1 or listed[0].get("Key") != input_key or listed[0].get("Size") != ref.byte_count:
            raise ValueError("inspection input prefix contains unexpected objects")
        args = [
            "--sha256",
            ref.sha256,
            "--byte-count",
            str(ref.byte_count),
            "--format",
            ref.format,
            "--job-name",
            job,
            "--runtime-session-id",
            source.runtime_session_id,
            "--deployment-id",
            source.deployment_id,
            "--deployment-prefix",
            source.deployment_prefix,
            "--image-uri",
            image,
            "--source-commit",
            commit,
            "--source-version-id",
            ref.version_id,
            "--staged-version-id",
            staged_version,
        ]
        if target is not None:
            args += ["--target-column", target]
        request = {
            "ProcessingJobName": job,
            "RoleArn": role,
            "AppSpecification": {
                "ImageUri": image,
                "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "inspection"],
                "ContainerArguments": args,
            },
            "NetworkConfig": {"EnableNetworkIsolation": True},
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 30,
                }
            },
            "ProcessingInputs": [
                {
                    "InputName": "inspection_input",
                    "AppManaged": False,
                    "S3Input": {
                        "S3Uri": f"s3://{bucket}/{input_prefix}",
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
                            "S3Uri": f"s3://{bucket}/{output_prefix}",
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 900},
            "Tags": tags,
        }
        accepted = sm.create_processing_job(**request)
        arn = accepted.get("ProcessingJobArn", "")
        if arn != f"arn:aws:sagemaker:{region}:{source.account_id}:processing-job/{job}":
            raise RuntimeError("inspection accepted an unexpected native identity")
        return JobHandle(
            id=job,
            backend=self.backend_name,
            arn=arn,
            extra={
                "accepted": True,
                "inspected": False,
                "source": asdict(source),
                "role_arn": role,
                "image_uri": image,
                "source_commit": commit,
                "privilege_profile": self.privilege_profile,
                "target_column": target,
                "bucket": bucket,
                "region": region,
                "account_id": source.account_id,
                "runtime_session_id": source.runtime_session_id,
                "deployment_id": source.deployment_id,
                "deployment_prefix": source.deployment_prefix,
                "input_prefix": input_prefix,
                "input_key": input_key,
                "staged_version_id": staged_version,
                "output_prefix": output_prefix,
                "request": request,
            },
        )

    def _description(self, handle: JobHandle) -> dict[str, Any]:
        if (
            handle.backend != self.backend_name
            or not handle.arn
            or handle.extra.get("privilege_profile") != self.privilege_profile
        ):
            raise ValueError("inspection handle belongs to another backend or fixed profile")
        sm, _ = self._clients(handle.extra["region"])
        desc = sm.describe_processing_job(ProcessingJobName=handle.id)
        request = handle.extra["request"]
        if (
            desc.get("ProcessingJobName") != handle.id
            or desc.get("ProcessingJobArn") != handle.arn
            or any(
                _native_field(name, desc.get(name)) != _native_field(name, request[name])
                for name in (
                    "RoleArn",
                    "AppSpecification",
                    "ProcessingInputs",
                    "ProcessingOutputConfig",
                    "NetworkConfig",
                    "ProcessingResources",
                    "StoppingCondition",
                )
            )
            or _native_field("Environment", desc.get("Environment", {}))
            != _native_field("Environment", request.get("Environment", {}))
        ):
            raise ValueError("native inspection binding or isolation changed")
        return desc

    def status(self, handle: JobHandle) -> JobStatus:
        try:
            desc = self._description(handle)
        except Exception as error:
            return JobStatus(
                JobState.UNKNOWN, "inspection observation failed", {"error_type": type(error).__name__}
            )
        state = {
            "InProgress": JobState.RUNNING,
            "Stopping": JobState.RUNNING,
            "Completed": JobState.SUCCEEDED,
            "Failed": JobState.FAILED,
            "Stopped": JobState.STOPPED,
        }.get(desc.get("ProcessingJobStatus"), JobState.UNKNOWN)
        return JobStatus(state, raw=desc)

    def _read_output(self, handle: JobHandle, name: str, limit: int) -> tuple[bytes, str]:
        _, s3 = self._clients(handle.extra["region"])
        key = handle.extra["output_prefix"] + name
        head = s3.head_object(
            Bucket=handle.extra["bucket"], Key=key, ExpectedBucketOwner=handle.extra["account_id"]
        )
        version = _version(head.get("VersionId"))
        if type(head.get("ContentLength")) is not int or not 0 < head["ContentLength"] <= limit:
            raise ValueError("inspection output exceeds its size bound")
        response = s3.get_object(
            Bucket=handle.extra["bucket"],
            Key=key,
            VersionId=version,
            ExpectedBucketOwner=handle.extra["account_id"],
        )
        body = response["Body"]
        try:
            data = body.read(limit + 1)
        finally:
            body.close()
        if response.get("VersionId") != version or len(data) != head["ContentLength"]:
            raise ValueError("inspection output version or size changed")
        return data, version

    def report(self, handle: JobHandle) -> dict[str, Any]:
        state = self.status(handle)
        base = {
            "success": False,
            "accepted": True,
            "inspected": False,
            "prepared": False,
            "trained": False,
            "provider": self.backend_name,
            "status": {
                JobState.RUNNING: "running",
                JobState.SUCCEEDED: "verification_failed",
                JobState.FAILED: "failed",
                JobState.STOPPED: "stopped",
            }.get(state.state, "observation_failed"),
        }
        if state.state != JobState.SUCCEEDED:
            return base
        try:
            _, s3 = self._clients(handle.extra["region"])
            expected = handle.extra
            keys = self._keys(s3, expected["bucket"], expected["output_prefix"], expected["account_id"])
            names = {item.get("Key") for item in keys}
            if len(keys) != 2 or names != {
                expected["output_prefix"] + name for name in ("SUMMARY.json", "INSPECTED.json")
            }:
                raise ValueError("inspection output contains missing or extra objects")
            raw, summary_version = self._read_output(handle, "SUMMARY.json", MAX_SUMMARY_BYTES)
            receipt_raw, receipt_version = self._read_output(handle, "INSPECTED.json", MAX_RECEIPT_BYTES)
            receipt = decode_receipt(receipt_raw, expected_profile=self.privilege_profile)
            for name in (
                "runtime_session_id",
                "deployment_id",
                "deployment_prefix",
                "image_uri",
                "source_commit",
                "staged_version_id",
            ):
                if receipt[name] != expected[name]:
                    raise ValueError("inspection receipt differs from accepted native binding")
            ref = expected["source"]["artifact_ref"]
            if (
                receipt["job_name"] != handle.id
                or receipt["source_version_id"] != ref["version_id"]
                or any(receipt[name] != ref[name] for name in ("sha256", "byte_count", "format"))
                or receipt["summary_sha256"] != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError("inspection receipt input/output digest mismatch")
            summary = decode_summary(raw)
            if (
                any(summary[name] != ref[name] for name in ("sha256", "byte_count", "format"))
                or summary["target_column"] != expected["target_column"]
            ):
                raise ValueError("summary describes another input or target")
            return {
                **base,
                "success": True,
                "inspected": True,
                "status": "completed",
                "summary": summary,
                "summary_ref": {
                    "uri": f"s3://{expected['bucket']}/{expected['output_prefix']}SUMMARY.json",
                    "version_id": summary_version,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "receipt_ref": {
                    "uri": f"s3://{expected['bucket']}/{expected['output_prefix']}INSPECTED.json",
                    "version_id": receipt_version,
                    "sha256": hashlib.sha256(receipt_raw).hexdigest(),
                },
            }
        except Exception as error:
            return {**base, "error_type": type(error).__name__}


class SageMakerNativeDatasetInspectionBackend(ProcessingDatasetInspectionBackend):
    """Source-owned native profile; admission still requires exact image evidence."""

    privilege_profile = NATIVE_CEILING_PROFILE


register_adapter(
    DataInspectionBackend, "processing", SageMakerNativeDatasetInspectionBackend, is_default=True
)
