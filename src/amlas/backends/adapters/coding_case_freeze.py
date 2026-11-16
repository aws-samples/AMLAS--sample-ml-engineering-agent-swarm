"""Fixed native case freeze; raw I/O only, no caller overrides or local fallback."""

from __future__ import annotations

import dataclasses
import hashlib
import re
import uuid

from ...control_plane.coding_case_state import producer_header, validate_catalog
from ...data_eng.coding_case_codec import COVERAGE, FILES, json_bytes
from ...data_eng.coding_case_managed import LIMIT, _expected, validate_managed_case_bundle
from ..types import DatasetArtifactRef, JobHandle, JobState, JobStatus
from .artifact_rejection import ImmutableArtifactRejectionError, rejection_data
from .coding_preparation import ProcessingCodingPreparationBackend


def _version(value):
    if (
        type(value) is not str
        or value in ("", "null")
        or len(value) > 1024
        or any(ord(c) < 33 or ord(c) > 126 for c in value)
    ):
        raise ValueError("immutable case version missing")
    return value


def _native(value):
    if type(value) is dict:
        return {k: _native(v) for k, v in value.items() if not (k == "AppManaged" and v is False)}
    if type(value) is list:
        return [_native(v) for v in value]
    return value


class ProcessingCodingCaseFreezeBackend(ProcessingCodingPreparationBackend):
    backend_name = "case_freeze"

    def _put(self, catalog, key, raw):
        _sm, s3 = self._clients(catalog.region)
        result = s3.put_object(
            Bucket=catalog.bucket, Key=key, ExpectedBucketOwner=catalog.account_id, IfNoneMatch="*", Body=raw
        )
        version = _version(result.get("VersionId"))
        self._raw(
            s3,
            catalog.bucket,
            key,
            version,
            catalog.account_id,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            count_lines=False,
        )
        return DatasetArtifactRef(
            "s3://" + catalog.bucket + "/" + key, version, hashlib.sha256(raw).hexdigest(), len(raw), "json"
        )

    def submit(self, *, source, catalog, runtime_session_id):
        validate_catalog(catalog)
        job = catalog.deployment_prefix + "-case-freeze-" + uuid.uuid4().hex[:10]
        if len(job) > 63 or not re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}", job):
            raise ValueError("case native name differs before staging")
        header = producer_header(catalog, runtime_session_id, job)
        _expected(source, header)
        origin = source.inspection_source
        if (origin.account_id, origin.region, origin.bucket) != (
            catalog.account_id,
            catalog.region,
            catalog.bucket,
        ):
            raise ValueError("case original source ownership differs")
        sm, s3 = self._clients(catalog.region)
        if (
            s3.get_bucket_versioning(Bucket=catalog.bucket, ExpectedBucketOwner=catalog.account_id).get(
                "Status"
            )
            != "Enabled"
        ):
            raise ValueError("case versioned storage required")
        prefix = "sessions/" + runtime_session_id + "/case-freeze/" + job + "/"
        ref = origin.artifact_ref
        raw, _sha, _lines = self._raw(
            s3,
            catalog.bucket,
            ref.uri.split("/", 3)[3],
            ref.version_id,
            catalog.account_id,
            ref.byte_count,
            ref.sha256,
            metadata=True,
            metadata_limit=262144,
            count_lines=False,
        )
        self._put(catalog, prefix + "input/source.parquet", raw)
        manifests = {}
        for name, uri, version, sha in source.manifest_refs:
            key = uri.split("/", 3)[3]
            head = s3.head_object(
                Bucket=catalog.bucket, Key=key, VersionId=version, ExpectedBucketOwner=catalog.account_id
            )
            if head.get("VersionId") != version or type(head.get("ContentLength")) is not int:
                raise ValueError("original case manifest identity differs")
            body, _hash, _count = self._raw(
                s3,
                catalog.bucket,
                key,
                version,
                catalog.account_id,
                head["ContentLength"],
                sha,
                metadata=True,
                metadata_limit=8 * 1024**2,
                count_lines=False,
            )
            self._put(catalog, prefix + "input/source_" + name + ".json", body)
            manifests[name] = {"sha256": sha, "byte_count": len(body)}
        binding = json_bytes(
            {
                "schema_version": 1,
                "source": dataclasses.asdict(source),
                "producer": header,
                "manifest_files": manifests,
            }
        )
        if len(binding) > 65536:
            raise ValueError("case binding bound exceeded")
        bound = self._put(catalog, prefix + "input/BINDING.json", binding)
        request = {
            "ProcessingJobName": job,
            "RoleArn": catalog.execution_role_arn,
            "AppSpecification": {
                "ImageUri": catalog.image_uri,
                "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "case-freeze"],
                "ContainerArguments": [
                    "--binding-sha256",
                    bound.sha256,
                    "--binding-byte-count",
                    str(bound.byte_count),
                    "--source-commit",
                    catalog.source_commit,
                ],
            },
            "ProcessingResources": {
                "ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.xlarge", "VolumeSizeInGB": 30}
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 900},
            "NetworkConfig": {"EnableNetworkIsolation": True, "EnableInterContainerTrafficEncryption": False},
            "ProcessingInputs": [
                {
                    "InputName": "cases-input",
                    "S3Input": {
                        "S3Uri": "s3://" + catalog.bucket + "/" + prefix + "input/",
                        "LocalPath": "/opt/ml/processing/input/cases",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                        "S3CompressionType": "None",
                    },
                }
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "cases-output",
                        "S3Output": {
                            "S3Uri": "s3://" + catalog.bucket + "/" + prefix + "output/",
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "Tags": [
                {"Key": "amlas:" + key, "Value": value}
                for key, value in (
                    ("deployment-id", catalog.deployment_id),
                    ("deployment-prefix", catalog.deployment_prefix),
                    ("runtime-session-id", runtime_session_id),
                )
            ],
        }
        arn = "arn:aws:sagemaker:" + catalog.region + ":" + catalog.account_id + ":processing-job/" + job
        handle = JobHandle(
            job,
            self.backend_name,
            arn,
            {
                "runtime_session_id": runtime_session_id,
                "source": source,
                "catalog": catalog,
                "header": header,
                "request": request,
                "output_prefix": prefix + "output/",
                "binding_ref": bound,
                "submission_state": "ambiguous",
            },
        )
        try:
            result = sm.create_processing_job(**request)
            if type(result) is not dict or result.get("ProcessingJobArn") != arn:
                raise ValueError("case native acceptance identity differs")
            handle.extra["submission_state"] = "accepted"
        except Exception as error:
            response = getattr(error, "response", {})
            details = response.get("Error") if type(response) is dict else None
            code = details.get("Code") if type(details) is dict else None
            if type(code) is str and code in (
                "ValidationException",
                "AccessDeniedException",
                "ResourceLimitExceeded",
            ):
                handle.extra["submission_state"] = "rejected"
            handle.extra["error_type"] = type(error).__name__
        return handle

    def status(self, handle):
        try:
            if type(handle) is not JobHandle or handle.backend != self.backend_name:
                raise ValueError("original case handle required")
            catalog = handle.extra["catalog"]
            validate_catalog(catalog)
            _expected(handle.extra["source"], handle.extra["header"])
            sm, _s3 = self._clients(catalog.region)
            description = self._observation_io(sm.describe_processing_job, ProcessingJobName=handle.id)
            if (
                description.get("ProcessingJobArn") != handle.arn
                or any(
                    json_bytes(_native(description.get(key))) != json_bytes(_native(value))
                    for key, value in handle.extra["request"].items()
                    if key != "Tags"
                )
                or description.get("Environment") not in (None, {})
            ):
                raise ValueError("original case native request differs")
            tags = self._observation_io(sm.list_tags, ResourceArn=handle.arn).get("Tags")
            if type(tags) is not list or sorted((item["Key"], item["Value"]) for item in tags) != sorted(
                (item["Key"], item["Value"]) for item in handle.extra["request"]["Tags"]
            ):
                raise ValueError("original case ownership tags differ")
            state = {
                "InProgress": JobState.RUNNING,
                "Completed": JobState.SUCCEEDED,
                "Failed": JobState.FAILED,
                "Stopped": JobState.STOPPED,
                "Stopping": JobState.RUNNING,
            }.get(description.get("ProcessingJobStatus"), JobState.UNKNOWN)
            self._qualify_io()
            return JobStatus(state, raw=description)
        except Exception as error:
            return JobStatus(JobState.UNKNOWN, detail=type(error).__name__)

    def artifacts(self, handle):
        state = self.status(handle)
        base = {
            "success": False,
            "frozen": False,
            "status": state.state.value,
            "execution_status": "not_executed",
        }
        if state.state != JobState.SUCCEEDED:
            return base
        try:
            catalog = handle.extra["catalog"]
            _sm, s3 = self._clients(catalog.region)
            prefix = handle.extra["output_prefix"]
            objects = self._observation_io(
                s3.list_objects_v2,
                Bucket=catalog.bucket,
                Prefix=prefix,
                ExpectedBucketOwner=catalog.account_id,
                MaxKeys=6,
            )
            if objects.get("IsTruncated") or {x["Key"] for x in objects.get("Contents", [])} != {
                prefix + name for name in FILES
            }:
                raise ValueError("closed five case outputs required")
            bundle, refs = {}, {}
            for name in FILES:
                raw, ref, _lines = self._output(
                    s3, catalog, prefix, name, metadata=True, metadata_limit=LIMIT
                )
                bundle[name], refs[name] = raw, DatasetArtifactRef(**ref)
            try:
                checked = validate_managed_case_bundle(
                    bundle, source=handle.extra["source"], header=handle.extra["header"]
                )
            except ValueError as error:
                raise ImmutableArtifactRejectionError("closed_manifest_invalid") from error
            self._qualify_io()
            return {
                **base,
                **checked,
                "success": True,
                "frozen": True,
                "status": "frozen",
                "job_name": handle.id,
                "job_arn": handle.arn,
                "files": refs,
                "coverage": dict(COVERAGE),
            }
        except Exception as error:
            return {**base, "status": "verification_failed", "error_type": type(error).__name__,
                    "job_name": handle.id, "job_arn": handle.arn, **rejection_data(error)}
