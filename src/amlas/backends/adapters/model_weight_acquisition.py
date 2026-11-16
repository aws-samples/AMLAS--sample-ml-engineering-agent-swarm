"""Fixed managed model fetch; accepted handles and exact-version untrusted bytes."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from ...config.aws_region import resolve_aws_client_region, resolve_aws_region
from ...config.deployment import deployment_job_tags, resolve_deployment_prefix
from ...config.externalized import resolve
from ...data_eng import model_acquisition_schema as schema
from ...data_eng.model_weight_manifest import MAX_FILE_BYTES, expected_model_weight_manifest
from ..types import JobHandle, JobState, JobStatus
from ._util import _initial_processing_create
from .coding_preparation import record_step
from .failure_cause import cause_fields, code_site, record_change
from .process_pins import ProcessRecords, fingerprint, reusable_fingerprint

MODEL_ACQUISITION_CONFIGURATION_ENV_KEYS = (
    "AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI",
    "AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT",
    "AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST",
)
_CONFIGURATION_DETAILS = {
    "model_acquisition_configuration_missing": (
        "Model acquisition is unconfigured: required deployment selectors are missing. "
        "Supply independently qualified server configuration; catalog metadata does not establish native admission."
    ),
    "model_acquisition_configuration_invalid": (
        "Model acquisition deployment selectors are invalid or inconsistent. "
        "Verify independently qualified server configuration; catalog metadata does not establish native admission."
    ),
}


_MODEL_DIAGNOSTIC_CODES = {
    "create_boundary": "model_acquisition_submission_unknown",
    "describe": "model_acquisition_observation_unknown",
}
_MODEL_ERROR_CLASSES = ("ParamValidationError", "ClientError", "TimeoutError", "UNKNOWN")
_MODEL_NATIVE_ERROR_CODES = frozenset(
    {
        "ValidationException",
        "ResourceInUse",
        "ResourceLimitExceeded",
        "ResourceNotFound",
        "ThrottlingException",
        "AccessDeniedException",
    }
)


def _model_progress(phase: str, state: str, verified_payloads: int, cause: tuple = ()) -> None:
    """Keep optional diagnostic imports and logger failures outside job outcomes."""
    try:
        from ...control_plane.normal_run_checkpoint import _warn_normal_model_progress

        _warn_normal_model_progress(phase, state, verified_payloads, cause)
    except Exception:
        return


def _interrupted(error: Exception, state: str, verified_payloads: int) -> bool:
    """Log and report True when the caller's own run-record recheck ended this observation.

    Such an error carries no verdict on the bytes: the pass logs one closed `model_verification_interrupted` line,
    keeps every byte proof and re-raises, so the caller's recheck decides. Anything else stays a refusal.
    """
    reason = record_change(error)
    if reason is None:
        return False
    _model_progress("model_verification_interrupted", state, verified_payloads, (reason, code_site(error)))
    return True


def _project_model_diagnostic(value: Any, phase: str) -> dict[str, Any] | None:
    """Copy only the closed diagnostic scalars; never retain provider envelopes."""
    if type(value) is not dict or phase not in _MODEL_DIAGNOSTIC_CODES:
        return None
    error_class = value.get("error_class")
    native_code = value.get("native_error_code")
    request_id = value.get("request_id")
    status = value.get("http_status")
    return {
        "phase": phase,
        "error_class": error_class
        if type(error_class) is str and error_class in _MODEL_ERROR_CLASSES
        else "UNKNOWN",
        "error_code": _MODEL_DIAGNOSTIC_CODES[phase],
        "native_error_code": native_code
        if type(native_code) is str and native_code in _MODEL_NATIVE_ERROR_CODES
        else "UNKNOWN",
        "request_id": request_id
        if type(request_id) is str
        and len(request_id) == 36
        and re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", request_id)
        else None,
        "http_status": status if type(status) is int and 100 <= status <= 599 else None,
    }


def _model_failure_diagnostic(error: Exception, phase: str) -> dict[str, Any] | None:
    from botocore.exceptions import ClientError, ParamValidationError

    value: dict[str, Any] = {"error_class": "UNKNOWN"}
    for kind in (ParamValidationError, ClientError, TimeoutError):
        if isinstance(error, kind):
            value["error_class"] = kind.__name__
            break
    if isinstance(error, ClientError) and type(error.response) is dict:
        code = error.response.get("Error")
        metadata = error.response.get("ResponseMetadata")
        if type(code) is dict:
            value["native_error_code"] = code.get("Code")
        if type(metadata) is dict:
            value["request_id"] = metadata.get("RequestId")
            value["http_status"] = metadata.get("HTTPStatusCode")
    return _project_model_diagnostic(value, phase)


class ModelAcquisitionConfigurationError(ValueError):
    """Bounded startup configuration DATA; no values, action grants or native claims."""

    def __init__(self, error_code: str, configuration_keys: tuple[str, ...]) -> None:
        if (
            type(error_code) is not str
            or error_code not in _CONFIGURATION_DETAILS
            or type(configuration_keys) is not tuple
            or not configuration_keys
            or any(type(key) is not str for key in configuration_keys)
            or tuple(key for key in MODEL_ACQUISITION_CONFIGURATION_ENV_KEYS if key in configuration_keys)
            != configuration_keys
        ):
            raise ValueError("closed model configuration diagnostic required")
        self._error_code = error_code
        self._configuration_keys = configuration_keys
        super().__init__(_CONFIGURATION_DETAILS[error_code])

    @property
    def error_code(self) -> str:
        return self._error_code

    @property
    def configuration_keys(self) -> tuple[str, ...]:
        return self._configuration_keys

    @property
    def detail(self) -> str:
        return _CONFIGURATION_DETAILS[self._error_code]


def _version(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or value.lower() == "null"
        or len(value) > 1024
        or any(char.isspace() or ord(char) < 32 for char in value)
    ):
        raise ValueError("model artifacts require a non-null exact object version")
    return value


def _native(field: str, value: Any) -> str:
    if field == "NetworkConfig" and type(value) is dict:
        value = {"EnableNetworkIsolation": False, "EnableInterContainerTrafficEncryption": False, **value}
    if field == "ProcessingInputs" and value is None:
        value = []
    if field == "ProcessingOutputConfig" and type(value) is dict:
        value = json.loads(json.dumps(value))
        for output in value.get("Outputs", []):
            output.setdefault("AppManaged", False)
    return json.dumps(value, sort_keys=True)


@dataclass(frozen=True)
class _ModelByteProof:
    """Private, bounded byte witness; never a serialized acquisition authority."""

    handle: JobHandle
    binding: str
    manifests: str
    # URI, version, SHA, count, format, path and role; no mutable caller dictionaries.
    payloads: tuple[tuple[Any, ...], ...]
    # The HEAD fingerprint of each payload's exact version at its raw hash; None when a payload has none to offer.
    fingerprints: tuple[tuple[Any, ...], ...] | None = None


@dataclass(frozen=True)
class _SharedByteProof:
    """A byte proof of one acquisition for every adapter in this process: immutable evidence, no handle and no client."""

    manifests: str
    payloads: tuple[tuple[Any, ...], ...]
    fingerprints: tuple[tuple[Any, ...], ...]


@dataclass
class _ModelObservation:
    """Same-handle serialization, retained only while callers use or wait on it."""

    handle: JobHandle
    lock: Any
    users: int = 0


_PROOF_FIELDS = ("uri", "version_id", "sha256", "byte_count", "format", "path", "role")
_MAX_BYTE_PROOFS = 64
# Byte proofs shared by every adapter in the process, keyed by the acquisition's content binding.
_PROCESS_PROOFS = ProcessRecords(_MAX_BYTE_PROOFS)


class ProcessingModelWeightAcquisitionBackend:
    """Source-owned model Processing profile; no local/model-selected fallback."""

    backend_name = "model_processing"

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
        self._proof_lock = threading.RLock()
        self._byte_proofs: OrderedDict[int, _ModelByteProof] = OrderedDict()
        self._observations: dict[int, _ModelObservation] = {}
        self._qualified_io_pid = os.getpid()
        self._qualified_io = ContextVar("model_adapter_io_" + uuid.uuid4().hex, default=None)

    @contextmanager
    def qualified_io(self, current_qualification=None):
        if os.getpid() != self._qualified_io_pid:
            raise ValueError("inherited model adapter IO scope refused")
        if current_qualification is not None and not callable(current_qualification):
            raise ValueError("private model IO qualifier required")
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

    def _qualify_io(self, *, between_chunks=False):
        """Run the scope checks and every callback; between two chunks a `RecordRecheck` offers its cheap variant."""
        if os.getpid() != self._qualified_io_pid:
            raise ValueError("inherited model adapter IO refused")
        scope = self._qualified_io.get()
        if scope is None:
            return
        scopes = []
        while scope is not None:
            if not scope["active"] or scope["thread_id"] != threading.get_ident():
                raise ValueError("expired or foreign-thread model IO scope refused")
            scopes.append(scope)
            scope = scope["parent"]
        for current in reversed(scopes):
            callback = current["callback"]
            if callback is not None:
                if between_chunks:
                    record_step(callback)
                else:
                    callback()

    def _sdk_io(self, method, **kwargs):
        self._qualify_io()
        result = method(**kwargs)
        try:
            self._qualify_io()
        except BaseException:
            # A returned SDK stream stays owned even when current qualification fails.
            body = result.get("Body") if type(result) is dict else None
            close = getattr(body, "close", None)
            if callable(close):
                close()
            raise
        return result

    def _clients(self, region: str) -> tuple[Any, Any]:
        if self._sm is None or self._s3 is None:
            import boto3
            from botocore.config import Config

            session = boto3.Session(region_name=region)
            configuration = Config(connect_timeout=5, read_timeout=30, retries={"total_max_attempts": 1})
            self._sm = self._sm or session.client("sagemaker", config=configuration)
            self._s3 = self._s3 or session.client("s3", config=configuration)
        resolve_aws_client_region(self._sm, region)
        resolve_aws_client_region(self._s3, region)
        return self._sm, self._s3

    def submit(self, *, catalog_id: str, catalog_version: str, runtime_session_id: str) -> JobHandle:
        # The unchanged legacy route may resolve deployment configuration from SSM.
        handle = self._prepare_submission(
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            runtime_session_id=runtime_session_id,
            resolve_configuration=True,
        )
        return self._dispatch_prepared(handle)

    def prepare_submission(
        self,
        *,
        catalog_id: str,
        catalog_version: str,
        runtime_session_id: str,
        job_name: str | None = None,
    ) -> JobHandle:
        """Pure fixed-request data; no SDK access, action grant or native admission."""
        return self._prepare_submission(
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            runtime_session_id=runtime_session_id,
            job_name=job_name,
            resolve_configuration=False,
        )

    def _prepare_submission(
        self,
        *,
        catalog_id: str,
        catalog_version: str,
        runtime_session_id: str,
        job_name: str | None = None,
        resolve_configuration: bool,
    ) -> JobHandle:
        expected = expected_model_weight_manifest(catalog_id, catalog_version)
        region = resolve_aws_region(self._region)

        def selected(explicit: str | None, key: str) -> str:
            if explicit is not None:
                return explicit
            try:
                return resolve(
                    key, region=region, **({} if resolve_configuration else {"ssm_getter": lambda *args: ""})
                )
            except (KeyError, ValueError):
                raise ValueError("model acquisition configuration is unqualified") from None

        role = selected(self._role, "sagemaker_execution_role_arn")
        image = selected(self._image, "model_weight_acquisition_image_uri")
        bucket = selected(self._bucket, "session_bucket")
        source = selected(self._source, "model_weight_acquisition_source_commit")
        admission = selected(self._admission, "model_weight_acquisition_admission_digest")
        role_match = re.fullmatch(r"arn:aws:iam::([0-9]{12}):role/[A-Za-z0-9+=,.@_/-]+", role)
        image_match = re.fullmatch(
            r"([0-9]{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com/[a-z0-9][a-z0-9._/-]*@sha256:([a-f0-9]{64})",
            image,
        )
        if (
            not role_match
            or role_match[1] == "0" * 12
            or not image_match
            or image_match[1] != role_match[1]
            or image_match[2] != region
            or type(admission) is not str
            or admission != image_match[3]
            or not re.fullmatch(r"[a-f0-9]{40}", source)
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket)
            or ".." in bucket
            or type(runtime_session_id) is not str
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", runtime_session_id)
            or os.environ.get("AMLAS_AGENT_REMOTE_FETCH", "").strip().lower() in ("0", "false", "off", "no")
        ):
            raise ValueError("model fetch image/admission/source/deployment binding refused")
        tags = deployment_job_tags(runtime_session_id)
        tag_map = {item["Key"]: item["Value"] for item in tags}
        if not tags or not tag_map.get("amlas:deployment-id"):
            raise ValueError("model fetch requires trusted deployment ownership")
        prefix = resolve_deployment_prefix()
        job = f"{prefix}-model-acq-{uuid.uuid4().hex[:12]}" if job_name is None else job_name
        if type(job) is not str or not re.fullmatch(re.escape(prefix) + r"-model-acq-[a-f0-9]{12}", job):
            raise ValueError("fixed original model job name differs")
        bindings = {
            "job_name": job,
            "runtime_session_id": runtime_session_id,
            "deployment_id": tag_map["amlas:deployment-id"],
            "deployment_prefix": prefix,
            "image_uri": image,
            "source_commit": source,
        }
        schema.validate_bindings(bindings)
        arguments = ["--catalog-id", catalog_id, "--catalog-version", catalog_version]
        for key, value in bindings.items():
            arguments.extend(["--" + key.replace("_", "-"), value])
        environment = {"AMLAS_AGENT_REMOTE_FETCH": "1"}
        if "AMLAS_AGENT_HTTP_ALLOW_HOSTS" in os.environ:
            environment["AMLAS_AGENT_HTTP_ALLOW_HOSTS"] = os.environ["AMLAS_AGENT_HTTP_ALLOW_HOSTS"]
        output_prefix = f"sessions/{runtime_session_id}/model-acquisition/{job}/"
        request = {
            "ProcessingJobName": job,
            "RoleArn": role,
            "AppSpecification": {
                "ImageUri": image,
                "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "model-acquisition"],
                "ContainerArguments": arguments,
            },
            "ProcessingResources": {
                "ClusterConfig": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 80}
            },
            "NetworkConfig": {"EnableNetworkIsolation": False},
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "model_acquisition",
                        "AppManaged": False,
                        "S3Output": {
                            "S3Uri": f"s3://{bucket}/{output_prefix}",
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
            "Environment": environment,
            "Tags": tags,
        }
        arn = f"arn:aws:sagemaker:{region}:{role_match[1]}:processing-job/{job}"
        handle = JobHandle(
            id=job,
            backend=self.backend_name,
            arn=arn,
            extra={
                "accepted": False,
                "submission_ambiguous": False,
                "acquired": False,
                "catalog_id": catalog_id,
                "catalog_version": catalog_version,
                "expected_manifest_sha256": schema.expected_digest(expected),
                "bindings": bindings,
                "bucket": bucket,
                "account_id": role_match[1],
                "region": region,
                "output_prefix": output_prefix,
                "request": request,
                "native_status": "Submitted",
            },
        )
        return handle

    @staticmethod
    def _plan_json(extra: dict[str, Any]) -> str:
        if type(extra) is not dict:
            raise ValueError("exact model plan data required")
        try:
            raw = json.dumps(extra, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError, RecursionError) as error:
            raise ValueError("closed model plan data differs") from error
        if len(raw.encode("utf-8")) > schema.MAX_METADATA_BYTES:
            raise ValueError("model plan byte bound differs")
        return raw

    def validate_prepared(self, handle: JobHandle) -> JobHandle:
        """Rebuild current fixed configuration before any prepared native effect."""
        if (
            type(handle) is not JobHandle
            or handle.backend != self.backend_name
            or type(handle.id) is not str
            or type(handle.arn) is not str
            or type(handle.extra) is not dict
        ):
            raise ValueError("exact original model plan required")
        try:
            expected = self.prepare_submission(
                catalog_id=handle.extra["catalog_id"],
                catalog_version=handle.extra["catalog_version"],
                runtime_session_id=handle.extra["bindings"]["runtime_session_id"],
                job_name=handle.id,
            )
        except (KeyError, TypeError) as error:
            raise ValueError("original model plan fields missing") from error
        if expected.arn != handle.arn or self._plan_json(expected.extra) != self._plan_json(handle.extra):
            raise ValueError("original prepared model configuration differs")
        return expected

    def submit_prepared(self, handle: JobHandle, *, before_create: Any = None) -> JobHandle:
        """Dispatch exact source-validated data; the journal owns attempt permission."""
        if before_create is not None and not callable(before_create):
            raise ValueError("private model effect qualifier required")
        expected = self.validate_prepared(handle)

        def qualify_effect() -> None:
            self.validate_prepared(expected)
            if before_create is not None:
                before_create()
            self.validate_prepared(expected)

        return self._dispatch_prepared(expected, before_create=qualify_effect)

    def _dispatch_prepared(self, handle: JobHandle, *, before_create: Any = None) -> JobHandle:
        region, bucket, account = handle.extra["region"], handle.extra["bucket"], handle.extra["account_id"]
        sm, s3 = self._clients(region)
        location = (
            self._sdk_io(s3.get_bucket_location, Bucket=bucket, ExpectedBucketOwner=account).get(
                "LocationConstraint"
            )
            or "us-east-1"
        )
        if (
            location != region
            or self._sdk_io(s3.get_bucket_versioning, Bucket=bucket, ExpectedBucketOwner=account).get(
                "Status"
            )
            != "Enabled"
        ):
            raise ValueError("model fetch requires a same-region versioned bucket")
        request, arn = handle.extra["request"], handle.arn
        with _initial_processing_create(account, region):
            # Refusal here precedes Create; it must not be interpreted as lost native acceptance.
            if before_create is not None:
                before_create()
            self._qualify_io()
            try:
                response = sm.create_processing_job(**request)
                self._qualify_io()
                returned_arn = response.get("ProcessingJobArn")
            except Exception as error:
                # A lost native response cannot be called a rejected job or safely resubmitted.
                handle.extra.update(
                    submission_ambiguous=True,
                    native_status="SubmissionUnknown",
                    observation_error=type(error).__name__,
                    submission_diagnostic=_model_failure_diagnostic(error, "create_boundary"),
                )
                return handle
        if returned_arn != arn:
            handle.extra.update(
                submission_ambiguous=True,
                native_status="SubmissionUnknown",
                observation_error="NativeIdentityMismatch",
            )
            return handle
        handle.extra["accepted"] = True
        return handle

    def _description(self, handle: JobHandle) -> dict[str, Any]:
        if type(handle) is not JobHandle or handle.backend != self.backend_name or not handle.arn:
            raise ValueError("model acquisition private handle differs")
        expected = expected_model_weight_manifest(handle.extra["catalog_id"], handle.extra["catalog_version"])
        if handle.extra["expected_manifest_sha256"] != schema.expected_digest(expected):
            raise ValueError("model handle source catalog changed")
        sm, _ = self._clients(handle.extra["region"])
        desc = self._sdk_io(sm.describe_processing_job, ProcessingJobName=handle.id)
        request = handle.extra["request"]
        if (
            desc.get("ProcessingJobName") != handle.id
            or desc.get("ProcessingJobArn") != handle.arn
            or any(
                _native(key, desc.get(key)) != _native(key, request.get(key))
                for key in (
                    "RoleArn",
                    "AppSpecification",
                    "ProcessingResources",
                    "NetworkConfig",
                    "ProcessingInputs",
                    "ProcessingOutputConfig",
                    "StoppingCondition",
                    "Environment",
                )
            )
        ):
            raise ValueError("model native execution readback changed")
        return desc

    @contextmanager
    def _observation(self, handle: JobHandle) -> Iterator[None]:
        # Global lock protects only memory; unrelated native observations never
        # wait behind another handle's potentially hour-long first raw transfer.
        with self._proof_lock:
            observation = self._observations.get(id(handle))
            if observation is None:
                observation = _ModelObservation(handle, threading.RLock())
                self._observations[id(handle)] = observation
            observation.users += 1
        try:
            with observation.lock:
                yield
        finally:
            with self._proof_lock:
                observation.users -= 1
                if observation.users == 0:
                    self._observations.pop(id(handle), None)

    def _discard_byte_proof(self, handle: JobHandle) -> None:
        with self._proof_lock:
            self._byte_proofs.pop(id(handle), None)
        try:
            _PROCESS_PROOFS.forget(self._content_binding(handle))
        except Exception:  # nosec B110 - a handle without the binding fields has no shared proof
            pass

    def status(self, handle: JobHandle) -> JobStatus:
        with self._observation(handle):
            _model_progress("model_native_observation", "UNKNOWN", 0)
            observed = self._status(handle)
            if observed.state in (JobState.PENDING, JobState.RUNNING):
                _model_progress("model_native_wait", observed.state.value, 0)
            if observed.state != JobState.SUCCEEDED:
                self._discard_byte_proof(handle)
            return observed

    def _status(self, handle: JobHandle) -> JobStatus:
        try:
            desc = self._description(handle)
        except Exception as error:
            if _interrupted(error, JobState.UNKNOWN.value, 0):
                raise
            return JobStatus(
                JobState.UNKNOWN,
                "model observation failed; retain handle",
                {
                    "error_type": type(error).__name__,
                    "observation_diagnostic": _model_failure_diagnostic(error, "describe"),
                },
            )
        state = {
            "InProgress": JobState.RUNNING,
            "Stopping": JobState.RUNNING,
            "Completed": JobState.SUCCEEDED,
            "Failed": JobState.FAILED,
            "Stopped": JobState.STOPPED,
        }.get(desc.get("ProcessingJobStatus"), JobState.UNKNOWN)
        return JobStatus(state, "", desc)

    def _object(self, handle: JobHandle, name: str, *, limit: int, deadline: float) -> tuple[bytes, str]:
        _, s3 = self._clients(handle.extra["region"])
        kwargs = {
            "Bucket": handle.extra["bucket"],
            "Key": handle.extra["output_prefix"] + name,
            "ExpectedBucketOwner": handle.extra["account_id"],
        }
        head = self._sdk_io(s3.head_object, **kwargs)
        version = _version(head.get("VersionId"))
        count = head.get("ContentLength")
        if type(count) is not int or not 0 < count <= limit:
            raise ValueError("model metadata size bound differs")
        result = self._sdk_io(s3.get_object, **kwargs, VersionId=version)
        body = result["Body"]
        timer = threading.Timer(max(0, deadline - time.monotonic()), body.close)
        timer.daemon = True
        timer.start()
        try:
            self._qualify_io()
            raw = body.read(limit + 1)
            self._qualify_io()
            if (
                time.monotonic() >= deadline
                or result.get("VersionId") != version
                or result.get("ContentLength") != count
                or len(raw) != count
            ):
                raise ValueError("model metadata exact-version bytes differ")
            return raw, version
        finally:
            timer.cancel()
            body.close()

    def _payload(self, handle: JobHandle, ref: dict[str, Any], deadline: float) -> dict[str, Any]:
        # Retain the original private getter and its initial checksum/stream semantics.
        return self._payload_observed(handle, ref, deadline)[0]

    def _payload_observed(
        self,
        handle: JobHandle,
        ref: dict[str, Any],
        deadline: float,
        *,
        pinned: tuple[Any, ...] | None = None,
        expected_fingerprint: tuple[Any, ...] | None = None,
        fingerprints_out: list | None = None,
    ) -> tuple[dict[str, Any], bool]:
        _, s3 = self._clients(handle.extra["region"])
        kwargs = {
            "Bucket": handle.extra["bucket"],
            "Key": handle.extra["output_prefix"] + ref["path"],
            "ExpectedBucketOwner": handle.extra["account_id"],
        }
        if time.monotonic() >= deadline:
            raise ValueError("model parent raw hash deadline exceeded")
        head = self._sdk_io(s3.head_object, **kwargs, ChecksumMode="ENABLED")
        if time.monotonic() >= deadline:
            raise ValueError("model parent raw hash deadline exceeded")
        version = _version(head.get("VersionId"))
        if head.get("ContentLength") != ref["byte_count"] or ref["byte_count"] > MAX_FILE_BYTES:
            raise ValueError("model exact object size differs")
        if pinned is not None:
            # Latest HEAD detects replacement/deletion; exact HEAD separately proves
            # the originally hashed version remains available to this bucket owner.
            if version != pinned[1]:
                raise ValueError("model original verified version changed")
            exact = self._sdk_io(s3.head_object, **kwargs, VersionId=pinned[1], ChecksumMode="ENABLED")
            if time.monotonic() >= deadline:
                raise ValueError("model parent raw hash deadline exceeded")
            if (
                _version(exact.get("VersionId")) != pinned[1]
                or type(exact.get("ContentLength")) is not int
                or exact.get("ContentLength") != ref["byte_count"]
            ):
                raise ValueError("model original verified version HEAD differs")
            if expected_fingerprint is not None and fingerprint(exact) != expected_fingerprint:
                raise ValueError("model original verified version fingerprint differs")
            head = exact
        observed_fingerprint = expected_fingerprint if pinned is not None else None
        full = head.get("ChecksumType") == "FULL_OBJECT" and bool(head.get("ChecksumSHA256"))
        if full:
            if base64.b64decode(head["ChecksumSHA256"], validate=True).hex() != ref["sha256"]:
                raise ValueError("model service full-object SHA differs")
        elif pinned is None:
            result = self._sdk_io(s3.get_object, **kwargs, VersionId=version)
            body = result["Body"]
            file_deadline = min(deadline, time.monotonic() + 900)
            timer = threading.Timer(max(0, file_deadline - time.monotonic()), body.close)
            timer.daemon = True
            timer.start()
            try:
                if result.get("VersionId") != version or result.get("ContentLength") != ref["byte_count"]:
                    raise ValueError("model exact-version response differs")
                digest, count = hashlib.sha256(), 0
                while True:
                    if time.monotonic() >= file_deadline:
                        raise ValueError("model parent raw hash deadline exceeded")
                    self._qualify_io(between_chunks=True)
                    chunk = body.read(min(1024 * 1024, ref["byte_count"] - count + 1))
                    self._qualify_io(between_chunks=True)
                    if time.monotonic() >= file_deadline:
                        raise ValueError("model parent raw hash deadline exceeded")
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > ref["byte_count"]:
                        raise ValueError("model parent raw hash count exceeded")
                    digest.update(chunk)
                if count != ref["byte_count"] or digest.hexdigest() != ref["sha256"]:
                    raise ValueError("model parent exact-version raw hash differs")
                observed_fingerprint = reusable_fingerprint(head, result)
            finally:
                timer.cancel()
                body.close()
        if fingerprints_out is not None:
            fingerprints_out.append(observed_fingerprint)
        result_ref = {
            "uri": f"s3://{handle.extra['bucket']}/{handle.extra['output_prefix']}{ref['path']}",
            "version_id": version,
            "sha256": ref["sha256"],
            "byte_count": ref["byte_count"],
            "format": "safetensors"
            if ref["role"] == "weight_shard"
            else "jinja"
            if ref["role"] == "chat_template"
            else "json",
            "path": ref["path"],
            "role": ref["role"],
        }
        if pinned is not None and tuple(result_ref[key] for key in _PROOF_FIELDS) != pinned:
            raise ValueError("model original byte witness tuple differs")
        return result_ref, pinned is None and not full

    def _proof_binding(self, handle: JobHandle) -> str:
        # Snapshot only original execution/catalog selectors. Observation flags are
        # mutable state; neither they nor caller-returned artifact JSON grant reuse.
        return json.dumps(
            {
                "id": handle.id,
                "backend": handle.backend,
                "arn": handle.arn,
                "clients": [id(self._sm), id(self._s3)],
                "extra": {
                    key: handle.extra[key]
                    for key in (
                        "catalog_id",
                        "catalog_version",
                        "expected_manifest_sha256",
                        "bindings",
                        "bucket",
                        "account_id",
                        "region",
                        "output_prefix",
                        "request",
                    )
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _content_binding(self, handle: JobHandle) -> str:
        """The proof binding without the identity of this adapter's clients: what the acquisition is, not who reads it."""
        value = json.loads(self._proof_binding(handle))
        del value["clients"]
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def _shared_proof(self, content: str) -> tuple[_SharedByteProof | None, Any]:
        """The process proof of this acquisition, or the lead of its first full pass; a second asker waits for the leader."""
        until = time.monotonic() + 3600

        def check() -> None:
            if time.monotonic() >= until:
                raise ValueError("model parent raw hash deadline exceeded")
            self._qualify_io()

        while True:
            shared = _PROCESS_PROOFS.get(content)
            if shared is not None:
                return shared, None
            flight, leader = _PROCESS_PROOFS.claim(content)
            if leader:
                shared = _PROCESS_PROOFS.get(content)
                if shared is not None:
                    _PROCESS_PROOFS.release(content, flight)
                    return shared, None
                return None, flight
            _PROCESS_PROOFS.wait(flight, check)

    def artifacts(self, handle: JobHandle) -> dict[str, Any]:
        # Serialize each original handle's whole observation, not unrelated jobs.
        with self._observation(handle):
            return self._artifacts(handle)

    def _artifacts(self, handle: JobHandle) -> dict[str, Any]:
        status = self.status(handle)
        base = {
            "success": False,
            "accepted": handle.extra.get("accepted") is True,
            "acquired": False,
            "trusted": False,
            "ready": False,
            "prepared": False,
            "trained": False,
            "guest_claim_attested": False,
            "job_name": handle.id,
            "job_arn": handle.arn,
            "status": status.state.value,
            "native_status": status.raw.get("ProcessingJobStatus", "Unknown"),
        }
        for key, value, phase in (
            ("submission_diagnostic", handle.extra.get("submission_diagnostic"), "create_boundary"),
            ("observation_diagnostic", status.raw.get("observation_diagnostic"), "describe"),
        ):
            diagnostic = _project_model_diagnostic(value, phase)
            if diagnostic is not None:
                base[key] = diagnostic
        if status.state != JobState.SUCCEEDED:
            return base
        verified_payloads = 0
        flight, content = None, None
        _model_progress("model_artifact_verification", status.state.value, verified_payloads)
        try:
            binding = self._proof_binding(handle)
            with self._proof_lock:
                proof = self._byte_proofs.get(id(handle))
            if proof is not None and (proof.handle is not handle or proof.binding != binding):
                raise ValueError("model original byte witness binding changed")
            shared, epoch = None, 0
            if proof is None:
                # Another adapter in this process may already hold the byte proof of this acquisition, or be hashing it.
                content = self._content_binding(handle)
                epoch = _PROCESS_PROOFS.epoch(content)
                shared, flight = self._shared_proof(content)
            expected = expected_model_weight_manifest(
                handle.extra["catalog_id"], handle.extra["catalog_version"]
            )
            _, s3 = self._clients(handle.extra["region"])
            listed = self._sdk_io(
                s3.list_objects_v2,
                Bucket=handle.extra["bucket"],
                Prefix=handle.extra["output_prefix"],
                ExpectedBucketOwner=handle.extra["account_id"],
                MaxKeys=len(expected["files"]) + 4,
            )
            names = {item["Key"] for item in listed.get("Contents", [])}
            wanted = {handle.extra["output_prefix"] + item["path"] for item in expected["files"]} | {
                handle.extra["output_prefix"] + name
                for name in ("LINEAGE.json", "MODEL_ACQUIRED.json", "COMMITTED.json")
            }
            if listed.get("IsTruncated") or names != wanted or len(listed.get("Contents", [])) != len(wanted):
                raise ValueError("model output contains missing/extra/duplicate keys")
            deadline = time.monotonic() + 3600
            records: dict[str, tuple[bytes, str, dict[str, Any]]] = {}
            for name, kind in (
                ("COMMITTED.json", "commit"),
                ("LINEAGE.json", "lineage"),
                ("MODEL_ACQUIRED.json", "receipt"),
            ):
                raw, version = self._object(handle, name, limit=schema.MAX_METADATA_BYTES, deadline=deadline)
                value = schema.decode_metadata(
                    raw,
                    kind=kind,
                    catalog_id=handle.extra["catalog_id"],
                    catalog_version=handle.extra["catalog_version"],
                    bindings=handle.extra["bindings"],
                )
                records[kind] = raw, version, value
            committed = records["commit"][2]
            for kind in ("lineage", "receipt"):
                raw = records[kind][0]
                if committed[kind]["sha256"] != hashlib.sha256(raw).hexdigest() or committed[kind][
                    "byte_count"
                ] != len(raw):
                    raise ValueError("model final commit metadata chain differs")
            receipt = records["receipt"][2]
            if receipt["lineage"] != committed["lineage"]:
                raise ValueError("model receipt lineage chain differs")
            manifests = json.dumps(
                {
                    kind: {
                        "uri": f"s3://{handle.extra['bucket']}/{handle.extra['output_prefix']}{name}",
                        "version_id": records[kind][1],
                        "sha256": hashlib.sha256(records[kind][0]).hexdigest(),
                        "byte_count": len(records[kind][0]),
                        "format": "json",
                    }
                    for kind, name in (
                        ("commit", "COMMITTED.json"),
                        ("lineage", "LINEAGE.json"),
                        ("receipt", "MODEL_ACQUIRED.json"),
                    )
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            witness = proof if proof is not None else shared
            if witness is not None and witness.manifests != manifests:
                raise ValueError("model original byte witness manifest changed")
            observations = []
            fingerprints: list = []
            for index, ref in enumerate(expected["files"]):
                observations.append(
                    self._payload_observed(
                        handle,
                        ref,
                        deadline,
                        pinned=witness.payloads[index] if witness is not None else None,
                        expected_fingerprint=(
                            witness.fingerprints[index]
                            if witness is not None and witness.fingerprints is not None
                            else None
                        ),
                        fingerprints_out=fingerprints,
                    )
                )
                verified_payloads += 1
                _model_progress("model_artifact_verification", status.state.value, verified_payloads)
            refs = [value[0] for value in observations]
            manifest_refs = {
                kind: {
                    "uri": f"s3://{handle.extra['bucket']}/{handle.extra['output_prefix']}{name}",
                    "version_id": records[kind][1],
                    "sha256": hashlib.sha256(records[kind][0]).hexdigest(),
                    "byte_count": len(records[kind][0]),
                    "format": "json",
                }
                for kind, name in (
                    ("commit", "COMMITTED.json"),
                    ("lineage", "LINEAGE.json"),
                    ("receipt", "MODEL_ACQUIRED.json"),
                )
            }
            if self._proof_binding(handle) != binding:
                raise ValueError("model binding changed during byte verification")
            # A successful service FULL_OBJECT checksum remains valid initial
            # integrity evidence, but it cannot manufacture an actual raw-byte proof.
            if witness is not None or all(raw_verified for _, raw_verified in observations):
                payloads = tuple(tuple(ref[key] for key in _PROOF_FIELDS) for ref in refs)
                complete = None not in fingerprints and len(fingerprints) == len(payloads)
                with self._proof_lock:
                    self._byte_proofs[id(handle)] = _ModelByteProof(
                        handle=handle,
                        binding=binding,
                        manifests=manifests,
                        payloads=payloads,
                        fingerprints=tuple(fingerprints) if complete else None,
                    )
                    self._byte_proofs.move_to_end(id(handle))
                    while len(self._byte_proofs) > _MAX_BYTE_PROOFS:
                        self._byte_proofs.popitem(last=False)
                if complete and content is not None:
                    _PROCESS_PROOFS.put(
                        content, _SharedByteProof(manifests, payloads, tuple(fingerprints)), epoch
                    )
            _model_progress("model_artifact_ready", status.state.value, verified_payloads)
            return {
                **base,
                "success": True,
                "acquired": True,
                "status": "SUCCEEDED",
                "catalog_id": handle.extra["catalog_id"],
                "catalog_version": handle.extra["catalog_version"],
                "file_refs": refs,
                "manifest_refs": manifest_refs,
                "integrity": {
                    "source_catalog_verified": True,
                    "exact_versions_verified": True,
                    "guest_claim_attested": False,
                },
            }
        except Exception as error:
            if _interrupted(error, status.state.value, verified_payloads):
                raise
            self._discard_byte_proof(handle)
            _model_progress(
                "model_artifact_refused", status.state.value, verified_payloads, cause_fields(error)
            )
            return {**base, "status": "verification_failed", "error_type": type(error).__name__}
        finally:
            if flight is not None:
                _PROCESS_PROOFS.release(content, flight)
