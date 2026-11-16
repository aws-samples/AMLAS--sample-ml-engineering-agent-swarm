"""Pinned source acquisition via Processing; local execution is explicit only."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

from ...config.aws_region import resolve_aws_client_region, resolve_aws_region
from ...config.deployment import deployment_job_tags, resolve_deployment_prefix
from ...config.externalized import resolve
from ..ports import DataAcquisitionBackend
from ..registry import REGISTRY, register_adapter
from ..types import JobHandle, JobState, JobStatus
from ._util import _initial_processing_create
from .artifact_rejection import ImmutableArtifactRejectionError, rejection_data
from .coding_preparation import record_step

_MANIFEST_LIMIT = 1024 * 1024
_SHA256 = re.compile(r"[a-f0-9]{64}")
PAYLOAD_HASH_MAX_BYTES = 512 * 1024 * 1024
PAYLOAD_HASH_CHUNK_BYTES = 1024 * 1024
PAYLOAD_HASH_SECONDS = 120
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "job_name",
        "repo_id",
        "revision",
        "filename",
        "runtime_session_id",
        "deployment_id",
        "deployment_prefix",
        "image_uri",
        "source_commit",
        "data",
        "lineage",
        "commit_manifest",
        "commit_manifest_sha256",
    }
)


def _version_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.lower() == "null"
        or len(value) > 1024
        or any(c.isspace() or ord(c) < 32 for c in value)
    ):
        raise ValueError("acquisition requires a non-null object version")
    return value


def validate_source(repo_id: str, revision: str, filename: str) -> None:
    """Reject mutable revisions and unsafe identifiers before any SDK boundary."""
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("dataset repo_id must be an owner/name identifier")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("dataset revision must be a full lowercase commit hash")
    path = PurePosixPath(filename)
    if (
        not filename
        or path.is_absolute()
        or "\\" in filename
        or any(part in ("", ".", "..") for part in filename.split("/"))
        or any(ord(c) < 32 for c in filename)
        or len(filename) > 1024
        or path.suffix.lower() not in (".csv", ".parquet", ".json", ".jsonl")
    ):
        raise ValueError("dataset filename must be a supported confined relative path")


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid acquisition manifest path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("acquisition manifest path escapes output")
    return value


def _source_uri(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quote(filename, safe='/')}"


class ProcessingDatasetAcquisitionBackend:
    """Thin managed submit/poll/verify adapter; accepted jobs are never acquired."""

    backend_name = "processing"
    receipt_schema_version = 1
    privilege_profile = None

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str | None = None,
        image_uri: str | None = None,
        session_bucket: str | None = None,
        source_commit: str | None = None,
        sagemaker_client: Any = None,
        s3_client: Any = None,
    ) -> None:
        self._region = region
        self._role = role_arn
        self._image = image_uri
        self._bucket = session_bucket
        self._source = source_commit
        self._sm = sagemaker_client
        self._s3 = s3_client
        self._qualified_io_pid = os.getpid()
        self._qualified_io = ContextVar("dataset_observation_" + uuid.uuid4().hex, default=None)

    @contextmanager
    def qualified_io(self, current_qualification=None):
        """Private observation lifetime; callback data never grants submission."""
        if os.getpid() != self._qualified_io_pid:
            raise ValueError("inherited processing observation scope refused")
        if current_qualification is not None and not callable(current_qualification):
            raise ValueError("private processing observation qualifier required")
        scope = {
            "callback": current_qualification,
            "active": True,
            "thread_id": threading.get_ident(),
            "parent": self._qualified_io.get(),
        }
        token = self._qualified_io.set(scope)
        try:
            yield self
        finally:
            scope["active"] = False
            self._qualified_io.reset(token)

    def _qualify_scope(self, scope, *, between_chunks=False):
        if os.getpid() != self._qualified_io_pid:
            raise ValueError("inherited processing observation refused")
        scopes = []
        while scope is not None:
            if not scope["active"] or scope["thread_id"] != threading.get_ident():
                raise ValueError("expired or foreign-thread processing observation refused")
            scopes.append(scope)
            scope = scope["parent"]
        for current in reversed(scopes):
            callback = current["callback"]
            if callback is not None:
                if between_chunks:
                    record_step(callback)
                else:
                    callback()

    def _qualify_io(self):
        self._qualify_scope(self._qualified_io.get())

    def _qualify_chunk(self):
        """Between two chunks of one object body: the scope checks run, a `RecordRecheck` offers its cheap variant."""
        self._qualify_scope(self._qualified_io.get(), between_chunks=True)

    def _observation_io(self, method, **kwargs):
        self._qualify_io()
        result = method(**kwargs)
        try:
            self._qualify_io()
        except BaseException:
            body = result.get("Body") if type(result) is dict else None
            close = getattr(body, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    pass
            raise
        return result

    @contextmanager
    def _owned_observation_body(self, body, *, seconds=None, deadline=None, join=False):
        """Own returned streams before timer setup; retain any in-flight failure."""
        closer, started = None, False
        pending = False
        try:
            if seconds is not None or deadline is not None:
                if deadline is None:
                    deadline = time.monotonic() + seconds
                delay = max(0, deadline - time.monotonic())
                closer = threading.Timer(delay, body.close)
                closer.daemon = True
                closer.start()
                started = True
            yield deadline
        except BaseException:
            pending = True
            raise
        finally:
            cleanup_error = None
            if started:
                try:
                    closer.cancel()
                except BaseException as error:
                    cleanup_error = error
            try:
                body.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            if started and join:
                try:
                    closer.join(timeout=0.1)
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None and not pending:
                raise cleanup_error

    def _observation_read(self, body, count):
        self._qualify_chunk()
        result = body.read(count)
        self._qualify_chunk()
        return result

    def _clients(self, region: str) -> tuple[Any, Any]:
        if self._sm is None or self._s3 is None:
            import boto3
            from botocore.config import Config

            # Managed callers use their role credential chain. Developer
            # selectors are never forwarded to the Processing container.
            session = boto3.Session(region_name=region)
            configuration = Config(connect_timeout=5, read_timeout=30, retries={"total_max_attempts": 1})
            self._sm = self._sm or session.client("sagemaker", config=configuration)
            self._s3 = self._s3 or session.client(
                "s3",
                config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 1}),
            )
        resolve_aws_client_region(self._sm, region)
        resolve_aws_client_region(self._s3, region)
        return self._sm, self._s3

    def submit(
        self,
        *,
        repo_id: str,
        revision: str,
        filename: str,
        resources: dict[str, Any] | None = None,
    ) -> JobHandle:
        validate_source(repo_id, revision, filename)
        if os.environ.get("AMLAS_AGENT_REMOTE_FETCH", "").strip().lower() in ("0", "false", "no", "off"):
            raise ValueError("managed remote acquisition is disabled by deployment policy")
        egress_environment = {"AMLAS_AGENT_REMOTE_FETCH": "1"}
        hosts = os.environ.get("AMLAS_AGENT_HTTP_ALLOW_HOSTS", "").strip()
        if hosts:
            egress_environment["AMLAS_AGENT_HTTP_ALLOW_HOSTS"] = hosts
        resources = dict(resources or {})
        region = resolve_aws_region(self._region)
        role = (
            self._role if self._role is not None else resolve("sagemaker_execution_role_arn", region=region)
        )
        image = (
            self._image if self._image is not None else resolve("data_acquisition_image_uri", region=region)
        )
        bucket = self._bucket if self._bucket is not None else resolve("session_bucket", region=region)
        source = (
            self._source
            if self._source is not None
            else resolve("data_acquisition_source_commit", region=region)
        )
        role_match = re.fullmatch(
            r"arn:(aws(?:-us-gov|-cn)?):iam::([0-9]{12}):role/[A-Za-z0-9+=,.@_/-]+", role
        )
        image_match = re.fullmatch(
            r"([0-9]{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?/"
            r"[a-z0-9][a-z0-9._/-]*@sha256:([a-f0-9]{64})",
            image,
        )
        if not role_match or role_match[2] == "0" * 12:
            raise ValueError("managed acquisition requires a configured execution role")
        if not image_match or image_match[2] != region or image_match[1] != role_match[2]:
            raise ValueError("managed acquisition requires a same-account, same-region digest-pinned image")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) or ".." in bucket:
            raise ValueError("managed acquisition requires a configured session bucket")
        if not re.fullmatch(r"[a-f0-9]{40}", source):
            raise ValueError("managed acquisition requires the published image source commit")
        session_id = str(resources.get("runtime_session_id") or os.environ.get("RUNTIME_SESSION_ID") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("managed acquisition requires a confined runtime session id")
        tags = deployment_job_tags(session_id)
        if not tags:
            raise ValueError("managed acquisition requires deployment ownership")
        tag_map = {tag["Key"]: tag["Value"] for tag in tags}
        prefix = resolve_deployment_prefix()
        instance = str(resources.get("instance_type", "ml.m5.xlarge"))
        volume = resources.get("volume_size_gb", 30)
        max_runtime = resources.get("max_runtime_seconds", 3600)
        if not re.fullmatch(r"ml\.[a-z0-9]+\.[a-z0-9]+", instance):
            raise ValueError("invalid Processing instance type")
        if type(volume) is not int or not 1 <= volume <= 16384:
            raise ValueError("Processing volume_size_gb must be a positive supported integer")
        if type(max_runtime) is not int or not 1 <= max_runtime <= 86400:
            raise ValueError("Processing max_runtime_seconds must be between 1 and 86400")
        sm, s3 = self._clients(region)
        bucket_region = (
            s3.get_bucket_location(
                Bucket=bucket,
                ExpectedBucketOwner=role_match[2],
            ).get("LocationConstraint")
            or "us-east-1"
        )
        if bucket_region == "EU":
            bucket_region = "eu-west-1"
        if bucket_region != region:
            raise ValueError("selected session bucket is in another region")
        if (
            s3.get_bucket_versioning(
                Bucket=bucket,
                ExpectedBucketOwner=role_match[2],
            ).get("Status")
            != "Enabled"
        ):
            raise ValueError("managed acquisition requires enabled session bucket versioning")
        job_name = f"{prefix}-data-acq-{uuid.uuid4().hex[:12]}"
        output_prefix = f"sessions/{session_id}/acquisition/{job_name}/"
        output_uri = f"s3://{bucket}/{output_prefix}"
        args = [
            "--repo-id",
            repo_id,
            "--revision",
            revision,
            "--filename",
            filename,
            "--job-name",
            job_name,
            "--runtime-session-id",
            session_id,
            "--deployment-id",
            tag_map["amlas:deployment-id"],
            "--deployment-prefix",
            prefix,
            "--image-uri",
            image,
            "--source-commit",
            source,
        ]
        with _initial_processing_create(role_match[2], region):
            self._qualify_io()
            if os.environ.get("AMLAS_AGENT_REMOTE_FETCH", "").strip().lower() in ("0", "false", "no", "off"):
                raise ValueError("managed remote acquisition is disabled by deployment policy")
            accepted = sm.create_processing_job(
                ProcessingJobName=job_name,
                RoleArn=role,
                AppSpecification={
                    "ImageUri": image,
                    "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "acquisition"],
                    "ContainerArguments": args,
                },
                ProcessingResources={
                    "ClusterConfig": {
                        "InstanceCount": 1,
                        "InstanceType": instance,
                        "VolumeSizeInGB": volume,
                    }
                },
                ProcessingOutputConfig={
                    "Outputs": [
                        {
                            "OutputName": "acquisition",
                            "S3Output": {
                                "S3Uri": output_uri,
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob",
                            },
                        }
                    ]
                },
                StoppingCondition={"MaxRuntimeInSeconds": max_runtime},
                Tags=tags,
                Environment=egress_environment,
            )
        arn = str(accepted.get("ProcessingJobArn") or "")
        expected_arn = f"arn:{role_match[1]}:sagemaker:{region}:{role_match[2]}:processing-job/{job_name}"
        if arn != expected_arn:
            raise RuntimeError("Processing accepted an unexpected native job identity")
        extra = {
            "accepted": True,
            "acquired": False,
            "repo_id": repo_id,
            "revision": revision,
            "filename": filename,
            "bucket": bucket,
            "output_prefix": output_prefix,
            "output_uri": output_uri,
            "region": region,
            "role_arn": role,
            "account_id": role_match[2],
            "image_uri": image,
            "image_digest": image_match[3],
            "source_commit": source,
            "runtime_session_id": session_id,
            "deployment_id": tag_map["amlas:deployment-id"],
            "deployment_prefix": prefix,
            "native_status": "Submitted",
            "observation_error": "",
            "egress_environment": egress_environment,
        }
        if self.privilege_profile is not None:
            extra["privilege_profile"] = self.privilege_profile
            extra["receipt_schema_version"] = self.receipt_schema_version
        handle = JobHandle(id=job_name, backend=self.backend_name, arn=arn, extra=extra)
        observed = self.status(handle)
        extra["native_status"] = observed.raw.get("ProcessingJobStatus", "Submitted")
        extra["observation_error"] = observed.raw.get("observation_error", "")
        return handle

    def _description(self, handle: JobHandle) -> dict[str, Any]:
        if handle.backend != self.backend_name:
            raise ValueError("acquisition handle belongs to another backend")
        if self.privilege_profile is not None and (
            handle.extra.get("privilege_profile") != self.privilege_profile
            or type(handle.extra.get("receipt_schema_version")) is not int
            or handle.extra["receipt_schema_version"] != self.receipt_schema_version
        ):
            raise ValueError("acquisition handle belongs to another fixed privilege profile")
        sm, _ = self._clients(handle.extra["region"])
        desc = self._observation_io(sm.describe_processing_job, ProcessingJobName=handle.id)
        expected = handle.extra
        outputs = desc.get("ProcessingOutputConfig", {}).get("Outputs", [])
        if (
            desc.get("ProcessingJobArn") != handle.arn
            or desc.get("ProcessingJobName") != handle.id
            or desc.get("RoleArn") != expected["role_arn"]
            or desc.get("AppSpecification", {}).get("ImageUri") != expected["image_uri"]
            or desc.get("AppSpecification", {}).get("ContainerEntrypoint")
            != ["/usr/local/bin/amlas-processing-init", "acquisition"]
            or desc.get("Environment", {}) != expected["egress_environment"]
            or len(outputs) != 1
            or outputs[0].get("S3Output", {}).get("S3Uri") != expected["output_uri"]
        ):
            raise ValueError("native acquisition description does not match the accepted job")
        self._qualify_io()
        return desc

    def status(self, handle: JobHandle) -> JobStatus:
        try:
            desc = self._description(handle)
        except Exception as error:
            return JobStatus(
                JobState.UNKNOWN,
                "native job observation failed",
                {"observation_error": type(error).__name__, "accepted": bool(handle.arn)},
            )
        state = {
            "InProgress": JobState.RUNNING,
            "Stopping": JobState.RUNNING,
            "Completed": JobState.SUCCEEDED,
            "Failed": JobState.FAILED,
            "Stopped": JobState.STOPPED,
        }.get(desc.get("ProcessingJobStatus"), JobState.UNKNOWN)
        self._qualify_io()
        return JobStatus(state, str(desc.get("FailureReason") or ""), desc)

    def _read_manifest(self, handle: JobHandle, relative: str) -> tuple[bytes, dict[str, Any], str]:
        _, s3 = self._clients(handle.extra["region"])
        response = self._observation_io(
            s3.get_object,
            Bucket=handle.extra["bucket"],
            Key=handle.extra["output_prefix"] + _relative_path(relative),
            ExpectedBucketOwner=handle.extra["account_id"],
        )
        body = response["Body"]
        with self._owned_observation_body(body):
            version = _version_id(response.get("VersionId"))
            if response.get("ContentLength", 0) > _MANIFEST_LIMIT:
                raise ValueError("acquisition manifest exceeds its byte limit")
            data = self._observation_read(body, _MANIFEST_LIMIT + 1)
        if len(data) > _MANIFEST_LIMIT:
            raise ValueError("acquisition manifest exceeds its byte limit")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise ValueError("acquisition manifest must be an object")
        self._qualify_io()
        return data, parsed, version

    def _verify_version_payload(
        self,
        handle: JobHandle,
        relative: str,
        version: str,
        digest: str,
        count: int,
    ) -> None:
        """Hash bounded exact-version bytes; never parse or deserialize a dataset."""
        if count > PAYLOAD_HASH_MAX_BYTES or PAYLOAD_HASH_SECONDS <= 0:
            raise ValueError("payload verification exceeds its bounded I/O budget")
        _, s3 = self._clients(handle.extra["region"])
        deadline = time.monotonic() + PAYLOAD_HASH_SECONDS
        response = self._observation_io(
            s3.get_object,
            Bucket=handle.extra["bucket"],
            Key=handle.extra["output_prefix"] + relative,
            VersionId=version,
            ExpectedBucketOwner=handle.extra["account_id"],
        )
        body = response["Body"]
        with self._owned_observation_body(body, deadline=deadline):
            if response.get("VersionId") != version or response.get("ContentLength") != count:
                raise ValueError("exact-version payload response differs from the committed artifact")
            measured, result = 0, hashlib.sha256()
            while True:
                if time.monotonic() >= deadline:
                    raise ValueError("payload verification deadline exceeded")
                chunk = self._observation_read(body, PAYLOAD_HASH_CHUNK_BYTES)
                if time.monotonic() >= deadline:
                    raise ValueError("payload verification deadline exceeded")
                if not chunk:
                    break
                measured += len(chunk)
                if measured > count or measured > PAYLOAD_HASH_MAX_BYTES:
                    raise ValueError("payload verification byte budget exceeded")
                result.update(chunk)
            if measured != count or result.hexdigest() != digest:
                raise ImmutableArtifactRejectionError("artifact_hash_mismatch")
        self._qualify_io()

    def _validate_receipt_profile(self, receipt: dict[str, Any]) -> dict[str, Any]:
        if receipt.get("schema_version") != 1:
            raise ValueError("acquisition receipt has another schema")
        return {}

    def artifacts(self, handle: JobHandle) -> dict[str, Any]:
        status = self.status(handle)
        base = {
            "success": False,
            "acquired": False,
            "provider": self.backend_name,
            "job_name": handle.id,
            "job_arn": handle.arn,
            "status": status.state.value,
            "native_status": status.raw.get("ProcessingJobStatus", "Unknown"),
            "accepted": bool(handle.arn),
            "local_path": None,
            "prepared": False,
            "trained": False,
        }
        if status.state != JobState.SUCCEEDED:
            return {
                **base,
                "detail": status.detail,
                "observation_error": status.raw.get("observation_error", ""),
            }
        try:
            receipt_bytes, receipt, receipt_version = self._read_manifest(handle, "ACQUIRED.json")
            privilege_observation = self._validate_receipt_profile(receipt)
            if receipt.get("job_name") != handle.id:
                raise ValueError("acquisition receipt has another job identity")
            for key in (
                "repo_id",
                "revision",
                "filename",
                "runtime_session_id",
                "deployment_id",
                "deployment_prefix",
                "image_uri",
                "source_commit",
            ):
                if receipt.get(key) != handle.extra[key]:
                    raise ValueError("acquisition receipt provenance differs from the accepted request")
            data_path = _relative_path(receipt["data"])
            lineage_path = _relative_path(receipt["lineage"])
            commit_path = _relative_path(receipt["commit_manifest"])
            commit_bytes, commit, commit_version = self._read_manifest(handle, commit_path)
            if hashlib.sha256(commit_bytes).hexdigest() != receipt["commit_manifest_sha256"]:
                raise ValueError("acquisition commit manifest hash mismatch")
            bundle = PurePosixPath(commit_path).parent
            if (
                PurePosixPath(commit_path).name != "COMMITTED.json"
                or data_path != str(bundle / _relative_path(commit["data"]))
                or lineage_path != str(bundle / _relative_path(commit["lineage"]))
            ):
                raise ValueError("acquisition manifest bundle paths do not match")
            lineage_bytes, lineage, lineage_version = self._read_manifest(handle, lineage_path)
            if hashlib.sha256(lineage_bytes).hexdigest() != commit["lineage_sha256"]:
                raise ValueError("acquisition lineage hash mismatch")
            digest = lineage["sha256"]
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest) or digest != commit["sha256"]:
                raise ValueError("acquisition payload hash declaration mismatch")
            for key in ("repo_id", "revision", "filename"):
                if lineage.get(key) != handle.extra[key]:
                    raise ValueError("acquisition lineage differs from the requested source")
            if lineage.get("source_uri") != _source_uri(
                handle.extra["repo_id"],
                handle.extra["revision"],
                handle.extra["filename"],
            ):
                raise ValueError("acquisition lineage lacks the canonical pinned source")
            count = lineage["byte_count"]
            fmt = lineage["format"]
            if (
                type(count) is not int
                or not 1 <= count <= PAYLOAD_HASH_MAX_BYTES
                or fmt not in ("csv", "parquet", "json", "jsonl")
            ):
                raise ValueError("acquisition lineage has invalid artifact metadata")
            _, s3 = self._clients(handle.extra["region"])
            head = self._observation_io(
                s3.head_object,
                Bucket=handle.extra["bucket"],
                Key=handle.extra["output_prefix"] + data_path,
                ChecksumMode="ENABLED",
                ExpectedBucketOwner=handle.extra["account_id"],
            )
            if head["ContentLength"] != count:
                raise ValueError("acquisition output object size differs from the committed artifact")
            payload_version = _version_id(head.get("VersionId"))
            checksum_verified = False
            if head.get("ChecksumType") == "FULL_OBJECT" and head.get("ChecksumSHA256"):
                if base64.b64decode(head["ChecksumSHA256"], validate=True).hex() != digest:
                    raise ValueError("S3 full-object payload checksum mismatch")
                checksum_verified = True
            if not checksum_verified:
                self._verify_version_payload(handle, data_path, payload_version, digest, count)
            manifest_refs = {
                "receipt": {
                    "uri": handle.extra["output_uri"] + "ACQUIRED.json",
                    "version_id": receipt_version,
                    "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                },
                "commit": {
                    "uri": handle.extra["output_uri"] + commit_path,
                    "version_id": commit_version,
                    "sha256": hashlib.sha256(commit_bytes).hexdigest(),
                },
                "lineage": {
                    "uri": handle.extra["output_uri"] + lineage_path,
                    "version_id": lineage_version,
                    "sha256": hashlib.sha256(lineage_bytes).hexdigest(),
                },
            }
            self._qualify_io()
            return {
                **base,
                **privilege_observation,
                "success": True,
                "acquired": True,
                "artifact_uri": handle.extra["output_uri"] + data_path,
                "artifact_ref": {
                    "uri": handle.extra["output_uri"] + data_path,
                    "version_id": payload_version,
                    "sha256": digest,
                    "byte_count": count,
                    "format": fmt,
                },
                "manifest_refs": manifest_refs,
                "lineage_uri": handle.extra["output_uri"] + lineage_path,
                "commit_manifest_uri": handle.extra["output_uri"] + commit_path,
                "format": fmt,
                "sha256": digest,
                "byte_count": count,
                "repo_id": lineage["repo_id"],
                "revision": lineage["revision"],
                "filename": lineage["filename"],
                "source_uri": lineage["source_uri"],
                "image_uri": handle.extra["image_uri"],
                "image_digest": handle.extra["image_digest"],
                "source_commit": handle.extra["source_commit"],
                "integrity": {
                    "commit_manifest_verified": True,
                    "lineage_hash_verified": True,
                    "object_size_verified": True,
                    "payload_sha256_source": "processing_container",
                    "s3_full_object_sha256_verified": checksum_verified,
                    "version_bytes_sha256_verified": not checksum_verified,
                    **({"guest_privilege_claim_attested": False} if privilege_observation else {}),
                },
            }
        except Exception as error:
            return {
                **base,
                "detail": "completed job artifact verification failed",
                "error_type": type(error).__name__,
                **rejection_data(error),
            }


class SageMakerNativeDatasetAcquisitionBackend(ProcessingDatasetAcquisitionBackend):
    """Fixed versioned native ceiling; observed guest fields are untrusted telemetry."""

    receipt_schema_version = 2
    privilege_profile = "sagemaker_native_ceiling_v1"

    def submit(
        self,
        *,
        repo_id: str,
        revision: str,
        filename: str,
        resources: dict[str, Any] | None = None,
    ) -> JobHandle:
        if any(
            key in (resources or {})
            for key in ("privilege_profile", "receipt_schema_version", "initializer_feature")
        ):
            raise ValueError("caller cannot select a Processing privilege profile")
        return super().submit(repo_id=repo_id, revision=revision, filename=filename, resources=resources)

    def _validate_receipt_profile(self, receipt: dict[str, Any]) -> dict[str, Any]:
        from ...data_eng.inspection_schema import validate_native_privilege_proof

        if (
            set(receipt) != _RECEIPT_KEYS | {"privilege_profile", "privilege_proof"}
            or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != self.receipt_schema_version
            or receipt["privilege_profile"] != self.privilege_profile
        ):
            raise ValueError("acquisition receipt differs from the fixed native profile")
        return {
            "privilege_profile": self.privilege_profile,
            "privilege_proof": validate_native_privilege_proof(receipt["privilege_proof"]),
        }


class LocalDatasetAcquisitionBackend:
    """Explicit offline adapter; managed failures never dispatch here."""

    backend_name = "local"

    def submit(
        self, *, repo_id: str, revision: str, filename: str, resources: dict[str, Any] | None = None
    ) -> JobHandle:
        from ...data_eng.hub_dataset import acquire_huggingface_dataset_local

        validate_source(repo_id, revision, filename)
        result = acquire_huggingface_dataset_local(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            output_dir=(resources or {}).get("output_dir"),
        )
        return JobHandle(
            id="local-acquisition-" + uuid.uuid4().hex, backend="local", extra={"result": result}
        )

    def status(self, handle: JobHandle) -> JobStatus:
        result = self.artifacts(handle)
        return JobStatus(JobState.SUCCEEDED if result.get("success") else JobState.FAILED, raw=result)

    def artifacts(self, handle: JobHandle) -> dict[str, Any]:
        if handle.backend != self.backend_name:
            raise ValueError("acquisition handle belongs to another backend")
        return dict(handle.extra["result"])


def submit_dataset_acquisition(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    resources: dict[str, Any] | None = None,
    steering: Any = None,
) -> tuple[Any, JobHandle]:
    """Post-approval flow binding: resolve once and retain adapter with its handle."""
    resolved = REGISTRY.resolve(DataAcquisitionBackend, steering=steering)
    handle = resolved.adapter.submit(
        repo_id=repo_id, revision=revision, filename=filename, resources=resources
    )
    return resolved.adapter, handle


register_adapter(
    DataAcquisitionBackend, "processing", SageMakerNativeDatasetAcquisitionBackend, is_default=True
)
register_adapter(DataAcquisitionBackend, "local", LocalDatasetAcquisitionBackend, is_local=True)
