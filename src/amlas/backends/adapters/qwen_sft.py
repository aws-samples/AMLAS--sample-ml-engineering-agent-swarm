"""Fixed native Qwen SFT; exact-version raw I/O, never a model loader."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import quote

from ...control_plane.sft_state import (
    RELOAD_PREFERENCE_CONTRACT,
    SftCatalog,
    SftSubmissionSnapshot,
    VerifiedSftCheckpointSource,
    VerifiedSftInputs,
    VerifiedSftModelSource,
    fixed_reload_binding,
    fixed_sft_request,
    reload_compatibility_digest,
    select_reload_compatibility,
    validate_catalog,
    validate_inputs,
    validate_model,
    validate_ref,
)
from ...data_eng.inspection_schema import decode_object, validate_native_privilege_proof
from ...data_eng.model_weight_manifest import expected_model_weight_manifest
from ..types import DatasetArtifactRef, JobHandle, JobState, JobStatus
from .artifact_rejection import ImmutableArtifactRejectionError, rejection_data
from .coding_preparation import record_step
from .process_pins import FINGERPRINT_FIELDS, ProcessRecords, reusable_fingerprint

_METADATA_LIMIT = 262144
_PART_BYTES = 128 * 1024**2
_STATES = {
    "InProgress": JobState.RUNNING,
    "Stopping": JobState.RUNNING,
    "Completed": JobState.SUCCEEDED,
    "Failed": JobState.FAILED,
    "Stopped": JobState.STOPPED,
}


def _omit_native_default(value: Any, expected: Any, key: str, default: Any) -> dict:
    if type(value) is not dict or type(expected) is not dict:
        raise ValueError("SFT native request mapping differs")
    copied = dict(value)
    if key not in expected and key in copied:
        if type(copied[key]) is not type(default) or copied[key] != default:
            raise ValueError("SFT native request default differs")
        del copied[key]
    return copied


_SELECTED_INSTANCE_FIELDS = ("SelectedInstanceType", "SelectedInstanceCount")
# Secondary states in which SageMaker may not yet have chosen an instance from InstancePreferences.
_PRESELECTION_SECONDARY = ("Starting", "Pending")


def _ended_before_selection(desc: dict) -> bool:
    """True only for a Stopped, Failed or Stopping job whose transitions never left the waiting phase.

    SecondaryStatusTransitions must be a list of mappings with string Status values: one or more Starting/Pending
    entries, then for Stopped or Failed an optional Stopping entry and the matching terminal entry, for Stopping an
    optional Stopping entry. The last entry must equal SecondaryStatus. Any other entry (Downloading, Training,
    Uploading, an unlisted status) or a malformed value means SageMaker may have selected an instance.
    """
    state = desc.get("TrainingJobStatus")
    transitions = desc.get("SecondaryStatusTransitions")
    if state not in ("Stopped", "Failed", "Stopping") or type(transitions) is not list or not transitions:
        return False
    statuses = []
    for item in transitions:
        if type(item) is not dict or type(item.get("Status")) is not str:
            return False
        statuses.append(item["Status"])
    waiting = 0
    while waiting < len(statuses) and statuses[waiting] in _PRESELECTION_SECONDARY:
        waiting += 1
    rest = statuses[waiting:]
    allowed = ([], ["Stopping"]) if state == "Stopping" else ([state], ["Stopping", state])
    return waiting > 0 and rest in allowed and desc.get("SecondaryStatus") == statuses[-1]


def _selected_instance_view(desc: dict, expected: Any) -> Any:
    """Check and drop the instance SageMaker selected from a stored InstancePreferences list.

    SelectedInstanceType must be one of the stored preference types and SelectedInstanceCount the stored
    InstanceCount. Both may be absent only before selection: while the job is InProgress in the Starting/Pending
    secondary states, or when it is Stopped, Failed or Stopping and its SecondaryStatusTransitions never left them
    (`_ended_before_selection`, the shape of a job stopped or failed while it waited for capacity). Otherwise both are
    required. One alone, an unlisted type, another count or a malformed value refuses. The remaining ResourceConfig is
    still compared exactly (preference order, count, volume). A stored request without a preference list is returned
    unchanged, so any described Selected field on it refuses.
    """
    value = desc.get("ResourceConfig")
    if type(expected) is not dict or "InstancePreferences" not in expected or type(value) is not dict:
        return value
    copied = dict(value)
    selected = {key: copied.pop(key) for key in _SELECTED_INSTANCE_FIELDS if key in copied}
    if not selected:
        secondary = desc.get("SecondaryStatus")
        waiting = (
            desc.get("TrainingJobStatus") == "InProgress"
            and type(secondary) is str
            and secondary in _PRESELECTION_SECONDARY
        )
        if not waiting and not _ended_before_selection(desc):
            raise ValueError("SFT native selected instance absent")
        return copied
    preferences = expected["InstancePreferences"]
    kind = selected.get("SelectedInstanceType")
    count = selected.get("SelectedInstanceCount")
    if (
        len(selected) != len(_SELECTED_INSTANCE_FIELDS)
        or type(kind) is not str
        or type(count) is not int
        or type(preferences) is not list
        or kind not in [item.get("InstanceType") for item in preferences if type(item) is dict]
        or count != expected.get("InstanceCount")
    ):
        raise ValueError("SFT native selected instance differs")
    return copied


def selected_instance(description: Any, request: Any) -> dict:
    """The instance SageMaker selected from a stored InstancePreferences list, by its native field names.

    Read from a description that `status()` already validated, so both fields are present together or absent
    together (before selection). Empty for a fixed-type request, before selection and for any other value.
    """
    resource = request.get("ResourceConfig") if type(request) is dict else None
    value = description.get("ResourceConfig") if type(description) is dict else None
    if type(resource) is not dict or "InstancePreferences" not in resource or type(value) is not dict:
        return {}
    kind, count = value.get("SelectedInstanceType"), value.get("SelectedInstanceCount")
    if type(kind) is not str or type(count) is not int:
        return {}
    return {"SelectedInstanceType": kind, "SelectedInstanceCount": count}


def observed_selected_instance(status: Any, request: Any) -> dict:
    """`selected_instance` of one observation; empty when the observation is UNKNOWN.

    UNKNOWN is a failed read or a native state outside `_STATES` (for example Deleting, or a description without
    TrainingJobStatus). Neither is a resolved state of the job, so the observation names no selected instance, even
    when the description carried SelectedInstanceType and SelectedInstanceCount.
    """
    if type(status) is not JobStatus or status.state is JobState.UNKNOWN:
        return {}
    return selected_instance(status.raw, request)


def selected_instance_detail(selected: dict) -> str:
    """The status detail that names the selected instance; empty when none is selected."""
    return " ".join(key + "=" + str(selected[key]) for key in _SELECTED_INSTANCE_FIELDS if key in selected)


def _native_request_view(desc: dict, request: dict) -> dict:
    view = dict(desc)
    view["ResourceConfig"] = _selected_instance_view(desc, request.get("ResourceConfig"))
    view["AlgorithmSpecification"] = _omit_native_default(
        desc.get("AlgorithmSpecification"),
        request["AlgorithmSpecification"],
        "EnableSageMakerMetricsTimeSeries",
        False,
    )
    view["OutputDataConfig"] = _omit_native_default(
        desc.get("OutputDataConfig"),
        request["OutputDataConfig"],
        "KmsKeyId",
        "",
    )
    channels = desc.get("InputDataConfig")
    expected_channels = request["InputDataConfig"]
    if type(channels) is not list or len(channels) != len(expected_channels):
        raise ValueError("SFT native channel population differs")
    view["InputDataConfig"] = [
        _omit_native_default(channel, expected, "RecordWrapperType", "None")
        for channel, expected in zip(channels, expected_channels, strict=True)
    ]
    return view


def _json(value: Any) -> bytes:
    import json

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


# Directories below a job's own S3OutputPath that may hold its uncompressed model. A real job (CompressionType NONE)
# reports <S3OutputPath>/<TrainingJobName>/output/model with no trailing separator in every recorded
# DescribeTrainingJob answer; the simulated provider reports <S3OutputPath>/model, which no real job produces.
_NATIVE_MODEL_DIRECTORY = "output/model"
_SIMULATED_MODEL_DIRECTORY = "model"
# The service grammars of a training job name and an S3 bucket name.
_JOB_NAME = re.compile(r"[A-Za-z0-9](-*[A-Za-z0-9]){0,62}")
_BUCKET_NAME = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")


def _native_uncompressed_output_prefix(uri: Any, *, s3_output_path: str, job_name: str, bucket: str) -> str:
    """Key prefix, with exactly one trailing "/", of the uncompressed model directory a job reports.

    The single canonicalisation is dropping one trailing "/" from the reported value. What remains must equal,
    character for character, a job-bound directory below the request's own S3OutputPath in the job's own bucket.
    Another bucket, root, job or depth, an archive name, dot segments, a query, a fragment and a backslash are
    refused, as is any value that is not a string.
    """
    if (
        type(uri) is not str
        or type(s3_output_path) is not str
        or type(job_name) is not str
        or type(bucket) is not str
        or _JOB_NAME.fullmatch(job_name) is None
        or _BUCKET_NAME.fullmatch(bucket) is None
        or not s3_output_path.startswith("s3://" + bucket + "/")
        or s3_output_path.endswith("/")
        or any(char in uri for char in ("?", "#", "\\"))
    ):
        raise ValueError("native uncompressed output layout unqualified")
    canonical = uri[:-1] if uri.endswith("/") else uri
    if canonical not in (
        s3_output_path + "/" + job_name + "/" + _NATIVE_MODEL_DIRECTORY,
        s3_output_path + "/" + _SIMULATED_MODEL_DIRECTORY,
    ):
        raise ValueError("native uncompressed output layout unqualified")
    key = canonical.split("/", 3)[3]
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("native uncompressed output layout unqualified")
    return key + "/"


def _version(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or value == "null"
        or len(value) > 1024
        or any(ord(c) < 33 or ord(c) > 126 for c in value)
    ):
        raise ValueError("SFT output version absent")
    return value


def _deadline(value: float) -> None:
    if time.monotonic() >= value:
        raise ValueError("SFT raw I/O deadline exceeded")


# Slowest measured exact-version stream in the runtime (original pack objects of 5 MiB parts: 39.9 to 45.6 MiB/s;
# the 128 MiB-part staged copies stream at 72 to 95 MiB/s) divided by three. The time that correct hashing may need
# is a function of the bytes to hash, so a deadline that covers it is computed from them, not fixed.
_HASH_FLOOR_BYTES_PER_SECOND = 13 * 1024**2
_PIN_FINGERPRINT_FIELDS = FINGERPRINT_FIELDS


def _hash_seconds(byte_count: int) -> int:
    """Seconds a full read-and-hash of this many bytes may take at a third of the slowest measured stream."""
    return math.ceil(byte_count / _HASH_FLOOR_BYTES_PER_SECOND)


def _pin_key(ref: DatasetArtifactRef) -> tuple:
    return (ref.uri, ref.version_id, ref.sha256, ref.byte_count)


def _pin_fingerprint(head: dict) -> tuple:
    """The HEAD fields that cannot change for one immutable object version; an absent field differs from a present one."""
    return tuple((field, head[field]) for field in _PIN_FINGERPRINT_FIELDS if field in head)


# Exact versions this process has read and hashed in full, with the HEAD fingerprint of that read.
_PROCESS_PINS = ProcessRecords()


def _process_key(ref: DatasetArtifactRef, catalog: Any) -> tuple | None:
    """The key of a process pin, or None for a ref whose URI names a bucket other than the one that is read."""
    if not ref.uri.startswith("s3://" + catalog.bucket + "/"):
        return None
    return (catalog.account_id, catalog.region, catalog.bucket, *_pin_key(ref))


def _descriptor(ref: DatasetArtifactRef) -> dict:
    return {"sha256": ref.sha256, "byte_count": ref.byte_count}


def _fixed_output_descriptor_matches(value, ref, filename):
    if (
        type(value) is not dict
        or type(value.get("sha256")) is not str
        or type(value.get("byte_count")) is not int
    ):
        return False
    expected = _descriptor(ref)
    if set(value) == {"sha256", "byte_count"}:
        return value == expected
    # Historical fixed TRAINING/RELOAD writers included their known JSON format.
    if (
        filename in {"TRAINING.json", "RELOAD.json"}
        and set(value) == {"sha256", "byte_count", "format"}
        and type(value["format"]) is str
        and value["format"] == "json"
    ):
        return {key: value[key] for key in expected} == expected
    return False


class SageMakerQwenSftBackend:
    backend_name = "qwen_sft"

    def __init__(self, *, sagemaker_client: Any = None, s3_client: Any = None):
        self._sm, self._s3 = sagemaker_client, s3_client
        self._qualified_io_pid = os.getpid()
        self._qualified_io = ContextVar("sft_observation_" + uuid.uuid4().hex, default=None)

    @contextmanager
    def qualified_io(self, current_qualification=None):
        """Private observation lifetime; callback data never grants submission."""
        if os.getpid() != self._qualified_io_pid:
            raise ValueError("inherited SFT observation scope refused")
        if current_qualification is not None and not callable(current_qualification):
            raise ValueError("private SFT observation qualifier required")
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
            raise ValueError("inherited SFT observation refused")
        scopes = []
        while scope is not None:
            if not scope["active"] or scope["thread_id"] != threading.get_ident():
                raise ValueError("expired or foreign-thread SFT observation refused")
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

    @staticmethod
    def _require_reload_selection(selection, catalog, reader):
        if (
            selection is not None
            and select_reload_compatibility(reader, catalog, selection.checkpoint) != selection
        ):
            raise ValueError("fresh original reload compatibility differs")

    def _clients(self, region: str) -> tuple[Any, Any]:
        if self._sm is None or self._s3 is None:
            import boto3
            from botocore.config import Config

            options = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 0})
            if self._sm is None:
                self._sm = boto3.client("sagemaker", region_name=region, config=options)
            if self._s3 is None:
                self._s3 = boto3.client("s3", region_name=region, config=options)
        return self._sm, self._s3

    def _raw(
        self,
        ref: DatasetArtifactRef,
        catalog: SftCatalog,
        deadline: float,
        *,
        collect: bool = False,
        seen: dict | None = None,
    ) -> bytes | None:
        """Verify one exact version by size and SHA-256; a given `seen` dict records the pin and its HEAD fingerprint
        only after the whole verification (size, hash and qualification) has succeeded."""
        _deadline(deadline)
        s3 = self._clients(catalog.region)[1]
        key = ref.uri.split("/", 3)[3]
        args = {
            "Bucket": catalog.bucket,
            "Key": key,
            "ExpectedBucketOwner": catalog.account_id,
            "VersionId": ref.version_id,
        }
        head = self._observation_io(s3.head_object, **args, ChecksumMode="ENABLED")
        _deadline(deadline)
        if (
            head.get("VersionId") != ref.version_id
            or type(head.get("ContentLength")) is not int
            or head["ContentLength"] != ref.byte_count
        ):
            raise ValueError("SFT exact-version size differs")
        if not collect and head.get("ChecksumType") == "FULL_OBJECT" and head.get("ChecksumSHA256"):
            if base64.b64decode(head["ChecksumSHA256"], validate=True).hex() != ref.sha256:
                raise ValueError("SFT native full-object SHA differs")
            _deadline(deadline)
            self._qualify_io()
            if seen is not None:
                seen[_pin_key(ref)] = _pin_fingerprint(head)
            return None
        if collect and ref.byte_count > _METADATA_LIMIT:
            raise ValueError("SFT metadata size bound exceeded")
        result = self._observation_io(s3.get_object, **args)
        body = result["Body"]
        with self._owned_observation_body(body, deadline=deadline):
            if (
                result.get("VersionId") != ref.version_id
                or type(result.get("ContentLength")) is not int
                or result["ContentLength"] != ref.byte_count
            ):
                raise ValueError("SFT read version differs")
            if "ETag" in head and "ETag" in result and head["ETag"] != result["ETag"]:
                # The fingerprint recorded for a pin must be that of the object whose bytes were hashed.
                raise ValueError("SFT read ETag differs from the HEAD")
            count = 0
            hashed = hashlib.sha256()
            parts = []
            while True:
                _deadline(deadline)
                chunk = self._observation_read(body, min(1024 * 1024, ref.byte_count - count + 1))
                _deadline(deadline)
                if not chunk:
                    break
                count += len(chunk)
                if count > ref.byte_count:
                    raise ValueError("SFT raw byte bound exceeded")
                hashed.update(chunk)
                if collect:
                    parts.append(chunk)
            if count != ref.byte_count or hashed.hexdigest() != ref.sha256:
                raise ValueError("SFT independently read raw SHA differs")
            value = b"".join(parts) if collect else None
            fingerprint = None if collect else reusable_fingerprint(head, result)
        self._qualify_io()
        if seen is not None:
            seen[_pin_key(ref)] = _pin_fingerprint(head)
        key = None if fingerprint is None else _process_key(ref, catalog)
        if key is not None:
            _PROCESS_PINS.put(key, fingerprint)
        return value

    def _reverify(
        self, ref: DatasetArtifactRef, catalog: SftCatalog, deadline: float, verified: dict | None
    ) -> None:
        """Confirm a pin that was hash-verified earlier is still the same object version; otherwise verify it in full.

        S3 object versions are immutable, so a HEAD on the exact version that returns the fingerprint recorded at the
        verification shows the same stored object without reading its bytes again. A missing record, a different
        fingerprint, a removed version or any failure takes the full `_raw` path (or raises), never a pass.
        """
        recorded = verified.get(_pin_key(ref)) if type(verified) is dict else None
        full = max(deadline, time.monotonic() + _hash_seconds(ref.byte_count))
        if recorded is None:
            self._raw(ref, catalog, full)
            return
        _deadline(deadline)
        s3 = self._clients(catalog.region)[1]
        head = self._observation_io(
            s3.head_object,
            Bucket=catalog.bucket,
            Key=ref.uri.split("/", 3)[3],
            ExpectedBucketOwner=catalog.account_id,
            VersionId=ref.version_id,
            ChecksumMode="ENABLED",
        )
        _deadline(deadline)
        if (
            head.get("VersionId") == ref.version_id
            and type(head.get("ContentLength")) is int
            and head["ContentLength"] == ref.byte_count
            and _pin_fingerprint(head) == recorded
        ):
            self._qualify_io()
            return
        self._raw(ref, catalog, full)

    def _verified(self, ref: DatasetArtifactRef, catalog: Any, deadline: float) -> None:
        """Verify an exact version against its size and SHA-256 once per process; afterwards by an exact-version HEAD.

        The first verification of the version in this process is _raw, a full read, which leaves a pin (the HEAD
        fingerprint of the bytes it hashed). Later calls, from any adapter in the process, HEAD the exact version and
        compare it with the pin. A missing pin, a different fingerprint or a changed size or version is a full _raw, which
        hashes the bytes or refuses (the mismatched pin is dropped first); a failing HEAD raises and leaves the pin, which
        never admits anything without a later HEAD that matches it. One thread reads a version at a time: another asking for it
        waits for that read under its own deadline and qualifier, then makes its own HEAD. Metadata that is parsed
        (collect=True) is not read through here.
        """
        key = _process_key(ref, catalog)
        if key is None:
            self._raw(ref, catalog, deadline)
            return
        while True:
            recorded = _PROCESS_PINS.get(key)
            if recorded is not None:
                if self._head_matches(ref, catalog, deadline, recorded):
                    return
                _PROCESS_PINS.forget(key)
            flight, leader = _PROCESS_PINS.claim(key)
            if leader:
                try:
                    if _PROCESS_PINS.get(key) is None:
                        self._raw(ref, catalog, deadline)
                        return
                finally:
                    _PROCESS_PINS.release(key, flight)
                continue
            _PROCESS_PINS.wait(flight, lambda: (_deadline(deadline), self._qualify_io()))

    def _head_matches(self, ref: DatasetArtifactRef, catalog: Any, deadline: float, recorded: tuple) -> bool:
        _deadline(deadline)
        s3 = self._clients(catalog.region)[1]
        head = self._observation_io(
            s3.head_object,
            Bucket=catalog.bucket,
            Key=ref.uri.split("/", 3)[3],
            ExpectedBucketOwner=catalog.account_id,
            VersionId=ref.version_id,
            ChecksumMode="ENABLED",
        )
        _deadline(deadline)
        if (
            head.get("VersionId") == ref.version_id
            and type(head.get("ContentLength")) is int
            and head["ContentLength"] == ref.byte_count
            and _pin_fingerprint(head) == recorded
        ):
            self._qualify_io()
            return True
        return False

    def _copy(
        self,
        ref: DatasetArtifactRef,
        key: str,
        catalog: SftCatalog,
        deadline: float,
        *,
        seen: dict | None = None,
    ) -> DatasetArtifactRef:
        s3 = self._clients(catalog.region)[1]
        _deadline(deadline)
        args = {"Bucket": catalog.bucket, "Key": key, "ExpectedBucketOwner": catalog.account_id}
        upload = s3.create_multipart_upload(**args)
        uid = upload["UploadId"]
        complete = False
        try:
            source = (
                catalog.bucket
                + "/"
                + quote(ref.uri.split("/", 3)[3], safe="/")
                + "?versionId="
                + quote(ref.version_id, safe="")
            )
            parts = []
            for number, start in enumerate(range(0, ref.byte_count, _PART_BYTES), 1):
                _deadline(deadline)
                result = s3.upload_part_copy(
                    **args,
                    UploadId=uid,
                    PartNumber=number,
                    CopySource=source,
                    **(
                        {"CopySourceRange": f"bytes={start}-{min(start + _PART_BYTES, ref.byte_count) - 1}"}
                        if ref.byte_count > 5 * 1024**2
                        else {}
                    ),
                    ExpectedSourceBucketOwner=catalog.account_id,
                )
                _deadline(deadline)
                if result.get("CopySourceVersionId") != ref.version_id:
                    raise ValueError("SFT copied source version differs")
                parts.append({"ETag": result["CopyPartResult"]["ETag"], "PartNumber": number})
            _deadline(deadline)
            result = s3.complete_multipart_upload(
                **args, UploadId=uid, MultipartUpload={"Parts": parts}, IfNoneMatch="*"
            )
            complete = True
            staged = dataclasses.replace(
                ref, uri="s3://" + catalog.bucket + "/" + key, version_id=_version(result.get("VersionId"))
            )
            self._raw(staged, catalog, deadline, **({} if seen is None else {"seen": seen}))
            return staged
        finally:
            if not complete:
                try:
                    s3.abort_multipart_upload(**args, UploadId=uid)
                except Exception:
                    pass  # Own incomplete upload only; original error remains primary.

    def _existing_version(self, key: str, catalog: SftCatalog) -> str | None:
        """Version of the object already staged at a deterministic key, else None.

        A failed lookup reads as absent: the conditional create that follows then fails closed if the object
        exists.
        """
        s3 = self._clients(catalog.region)[1]
        try:
            head = s3.head_object(Bucket=catalog.bucket, Key=key, ExpectedBucketOwner=catalog.account_id)
        except Exception:
            return None
        version = head.get("VersionId") if type(head) is dict else None
        return version if type(version) is str and version else None

    def _adopt_or_copy(
        self,
        ref: DatasetArtifactRef,
        key: str,
        catalog: SftCatalog,
        deadline: float,
        *,
        seen: dict | None = None,
    ) -> DatasetArtifactRef:
        """Stage a copy at a deterministic key; an object already there is adopted only when its bytes verify."""
        extra = {} if seen is None else {"seen": seen}
        existing = self._existing_version(key, catalog)
        if existing is None:
            return self._copy(ref, key, catalog, deadline, **extra)
        staged = dataclasses.replace(
            ref, uri="s3://" + catalog.bucket + "/" + key, version_id=_version(existing)
        )
        self._raw(staged, catalog, deadline, **extra)
        return staged

    @staticmethod
    def _error_code(error: Exception) -> str | None:
        response = getattr(error, "response", {})
        nested = response.get("Error") if type(response) is dict else None
        code = nested.get("Code") if type(nested) is dict else None
        return code if type(code) is str else None

    @staticmethod
    def _is_this_native_job(describe: Any, key: str, name: str, arn_field: str, arn: str) -> bool:
        """After ResourceInUse for a deterministic name: true only for the job that carries this operation ARN."""
        try:
            value = describe(**{key: name})
        except Exception:
            return False
        return type(value) is dict and value.get(arn_field) == arn

    def submit(
        self,
        *,
        inputs: VerifiedSftInputs,
        catalog: SftCatalog,
        recipe_id: str,
        runtime_session_id: str,
        _campaign_state: Any = None,
        _execution_binding: Any = None,
        _source_refs: tuple[str, str] | None = None,
        _causal_parent_ref: DatasetArtifactRef | None = None,
    ) -> JobHandle:
        validate_inputs(inputs, catalog, runtime_session_id)
        if recipe_id not in ("fit_v1", "full_v1"):
            raise ValueError("SFT fixed recipe required")
        self._campaign_context(_campaign_state, _execution_binding, catalog, runtime_session_id)
        intent = None
        if _campaign_state is not None:
            from ...control_plane.campaign_state import build_sft_intent

            intent = build_sft_intent(
                _execution_binding,
                catalog,
                inputs,
                recipe_id=recipe_id,
                source_refs=_source_refs,
                **({"causal_parent_ref": _causal_parent_ref} if _causal_parent_ref is not None else {}),
            )
        elif _source_refs is not None or _causal_parent_ref is not None:
            raise ValueError("private source refs/causal parent require the original SFT journal")
        prepared = inputs.prepared
        files = {
            "COMMITTED.json": prepared.preparation_ref,
            **{k + ".jsonl": v for k, v in prepared.split_refs},
            **{
                {"report": "REPORT.json", "lineage": "lineage.jsonl", "membership": "membership.jsonl"}[k]: v
                for k, v in prepared.sidecar_refs
            },
        }
        binding = {
            "schema_version": 1,
            "catalog_id": "qwen3_5_9b",
            "catalog_version": "1",
            "recipe_id": recipe_id,
            "model_files": expected_model_weight_manifest("qwen3_5_9b", "1")["files"],
            "preparation_header": dict(prepared.header),
            "preparation_commit": _descriptor(prepared.preparation_ref),
            "split_refs": {k: _descriptor(v) for k, v in prepared.split_refs},
        }
        return self._submit(
            model=inputs.model,
            catalog=catalog,
            session=runtime_session_id,
            recipe_id=recipe_id,
            operation="train",
            other_files=files,
            other_channel="prepared",
            binding=binding,
            prepared=prepared,
            campaign_state=_campaign_state,
            execution_binding=_execution_binding,
            intent=intent,
        )

    def reload(
        self,
        *,
        model: VerifiedSftModelSource,
        checkpoint: VerifiedSftCheckpointSource,
        catalog: SftCatalog,
        runtime_session_id: str,
        _campaign_state: Any = None,
        _execution_binding: Any = None,
        _durable_inputs: Any = None,
        _reload_compatibility_reader: Any = None,
    ) -> JobHandle:
        validate_catalog(catalog)
        self._campaign_context(_campaign_state, _execution_binding, catalog, runtime_session_id)
        selection = None
        if (
            type(checkpoint) is VerifiedSftCheckpointSource
            and checkpoint.source_commit != catalog.source_commit
        ):
            selection = select_reload_compatibility(_reload_compatibility_reader, catalog, checkpoint)
        intent = None
        if _campaign_state is not None:
            from ...control_plane.campaign_state import build_sft_intent

            if type(_durable_inputs) is not VerifiedSftInputs or _durable_inputs.model != model:
                raise ValueError("original durable reload inputs required")
            intent = build_sft_intent(
                _execution_binding,
                catalog,
                _durable_inputs,
                recipe_id=checkpoint.recipe_id,
                operation="reload",
                checkpoint=checkpoint,
                reload_compatibility=selection,
                reload_resource_contract=RELOAD_PREFERENCE_CONTRACT,
            )
        elif _durable_inputs is not None:
            raise ValueError("durable reload inputs require server context")
        validate_model(model, catalog, runtime_session_id)
        if (
            type(checkpoint) is not VerifiedSftCheckpointSource
            or checkpoint.runtime_session_id != runtime_session_id
            or (selection is None and checkpoint.source_commit != catalog.source_commit)
            or checkpoint.recipe_id not in ("fit_v1", "full_v1")
        ):
            raise ValueError("SFT checkpoint original source/session differs")
        if not checkpoint.producer_job_arn.startswith(
            "arn:aws:sagemaker:"
            + catalog.region
            + ":"
            + catalog.account_id
            + ":training-job/"
            + catalog.deployment_prefix
            + "-"
        ):
            raise ValueError("SFT checkpoint foreign native producer")
        files = dict(checkpoint.file_refs)
        if (
            len(files) != len(checkpoint.file_refs)
            or set(files) != {"CHECKPOINT.json", "adapter_config.json", "adapter_model.safetensors"}
            or files["CHECKPOINT.json"] != checkpoint.checkpoint_ref
        ):
            raise ValueError("closed SFT checkpoint inventory required")
        for name, ref in files.items():
            validate_ref(
                ref,
                catalog,
                runtime_session_id,
                limit=2 * 1024**3 if name.endswith(".safetensors") else _METADATA_LIMIT,
            )
            if ref.format != ("safetensors" if name.endswith(".safetensors") else "json"):
                raise ValueError("SFT checkpoint format differs")
        binding = fixed_reload_binding(catalog, checkpoint, selection)
        return self._submit(
            model=model,
            catalog=catalog,
            session=runtime_session_id,
            recipe_id=checkpoint.recipe_id,
            operation="reload",
            other_files=files,
            other_channel="adapter",
            binding=binding,
            prepared=None,
            campaign_state=_campaign_state,
            execution_binding=_execution_binding,
            intent=intent,
            reload_compatibility=selection,
            reload_compatibility_reader=_reload_compatibility_reader,
            resource_contract=RELOAD_PREFERENCE_CONTRACT,
        )

    @staticmethod
    def _campaign_context(campaign_state, execution_binding, catalog, session):
        from ...backends.campaign_state import CampaignStateBackend
        from ...control_plane.campaign_state import TrustedExecutionBinding, _binding_fields

        if (campaign_state is None) != (execution_binding is None):
            raise ValueError("complete private campaign context required")
        if campaign_state is None:
            return
        if (
            not isinstance(campaign_state, CampaignStateBackend)
            or type(execution_binding) is not TrustedExecutionBinding
        ):
            raise ValueError("typed private campaign context required")
        fields = _binding_fields(execution_binding)
        if (
            fields["account_id"],
            fields["region"],
            fields["deployment_id"],
            fields["deployment_prefix"],
            fields["runtime_session_id"],
        ) != (catalog.account_id, catalog.region, catalog.deployment_id, catalog.deployment_prefix, session):
            raise ValueError("private campaign SFT scope differs")

    @staticmethod
    def _provisional(reservation, *, uncertainty=None):
        intent = reservation.intent
        handle = JobHandle(
            reservation.native_job_name,
            "qwen_sft",
            reservation.native_job_arn,
            {
                "runtime_session_id": reservation.binding.runtime_session_id,
                "catalog": intent.catalog,
                "model_source": intent.inputs.model,
                "prepared_source": intent.inputs.prepared,
                "operation": intent.operation,
                "recipe_id": intent.recipe_id,
                "submission_state": "rejected" if reservation.disposition == "REJECTED" else "ambiguous",
                "campaign_reservation": reservation,
                "operation_digest": reservation.operation_digest,
                "durability_status": uncertainty or "recorded",
                "restoration_status": "unresolved",
                **(
                    {"reload_compatibility": intent.reload_compatibility}
                    if intent.reload_compatibility is not None
                    else {}
                ),
            },
        )
        return handle

    def _submit(
        self,
        *,
        model,
        catalog,
        session,
        recipe_id,
        operation,
        other_files,
        other_channel,
        binding,
        prepared,
        campaign_state=None,
        execution_binding=None,
        intent=None,
        reload_compatibility=None,
        reload_compatibility_reader=None,
        resource_contract=None,
    ):
        if campaign_state is None:
            return self._stage_create(
                model=model,
                catalog=catalog,
                session=session,
                recipe_id=recipe_id,
                operation=operation,
                other_files=other_files,
                other_channel=other_channel,
                binding=binding,
                prepared=prepared,
                job=catalog.deployment_prefix + "-sft-" + uuid.uuid4().hex[:12],
                reload_compatibility=reload_compatibility,
                reload_compatibility_reader=reload_compatibility_reader,
                resource_contract=resource_contract,
            )
        from ...control_plane.campaign_state import OriginalJobReservation, _native_identity, operation_digest

        name, arn = _native_identity(intent)
        reservation = OriginalJobReservation(execution_binding, intent, operation_digest(intent), name, arn)
        try:
            reservation = campaign_state.reserve_original_job(execution_binding, intent)
        except Exception as error:
            handle = self._provisional(reservation, uncertainty="reservation_uncertain")
            handle.extra["error_type"] = type(error).__name__
            return handle
        if reservation.reused:
            try:
                return self.restore_original(
                    reservation,
                    fresh_inputs=intent.inputs,
                    _reload_compatibility_reader=reload_compatibility_reader,
                )
            except Exception as error:
                handle = self._provisional(reservation)
                handle.extra["error_type"] = type(error).__name__
                return handle
        try:
            reservation = campaign_state.mark_create_attempted(execution_binding, reservation)
        except Exception as error:
            handle = self._provisional(reservation, uncertainty="marker_uncertain")
            handle.extra["error_type"] = type(error).__name__
            return handle
        try:
            return self._stage_create(
                model=model,
                catalog=catalog,
                session=session,
                recipe_id=recipe_id,
                operation=operation,
                other_files=other_files,
                other_channel=other_channel,
                binding=binding,
                prepared=prepared,
                job=name,
                reservation=reservation,
                campaign_state=campaign_state,
                execution_binding=execution_binding,
                reload_compatibility=reload_compatibility,
                reload_compatibility_reader=reload_compatibility_reader,
            )
        except Exception as error:
            handle = self._provisional(reservation)
            handle.extra.update(submission_state="rejected", error_type=type(error).__name__)
            try:
                handle.extra["campaign_reservation"] = campaign_state.record_submission_outcome(
                    execution_binding, reservation, disposition="REJECTED", error_code=type(error).__name__
                )
            except Exception as persistence_error:
                handle.extra.update(
                    durability_status="outcome_uncertain",
                    persistence_error_type=type(persistence_error).__name__,
                )
            return handle

    @staticmethod
    def _restore_selection(intent):
        prepared = intent.inputs.prepared
        binding = {
            "schema_version": 1,
            "catalog_id": "qwen3_5_9b",
            "catalog_version": "1",
            "recipe_id": intent.recipe_id,
            "model_files": expected_model_weight_manifest("qwen3_5_9b", "1")["files"],
        }
        if intent.operation == "train":
            files = {
                "COMMITTED.json": prepared.preparation_ref,
                **{k + ".jsonl": v for k, v in prepared.split_refs},
                **{
                    {"report": "REPORT.json", "lineage": "lineage.jsonl", "membership": "membership.jsonl"}[
                        k
                    ]: v
                    for k, v in prepared.sidecar_refs
                },
            }
            binding.update(
                preparation_header=dict(prepared.header),
                preparation_commit=_descriptor(prepared.preparation_ref),
                split_refs={k: _descriptor(v) for k, v in prepared.split_refs},
            )
            return files, binding, prepared
        binding = fixed_reload_binding(intent.catalog, intent.checkpoint, intent.reload_compatibility)
        return dict(intent.checkpoint.file_refs), binding, None

    def restore_original(self, reservation, *, fresh_inputs, _reload_compatibility_reader=None):
        from ...control_plane.campaign_state import _encode_private_record

        _encode_private_record(reservation)
        intent = reservation.intent
        self._require_reload_selection(
            intent.reload_compatibility, intent.catalog, _reload_compatibility_reader
        )
        if type(fresh_inputs) is not VerifiedSftInputs or fresh_inputs != intent.inputs:
            raise ValueError("fresh original private SFT inputs differ")
        validate_inputs(fresh_inputs, intent.catalog, reservation.binding.runtime_session_id)
        handle = self._provisional(reservation)
        snapshot = reservation.submission_snapshot
        if snapshot is None or reservation.disposition in ("RESERVED", "REJECTED"):
            return handle
        deadline = time.monotonic() + 3600
        files, binding, prepared = self._restore_selection(intent)
        binding, count = self._verified_run_binding(
            model=fresh_inputs.model,
            catalog=intent.catalog,
            session=reservation.binding.runtime_session_id,
            other_files=files,
            binding=binding,
            prepared=prepared,
            deadline=deadline,
        )
        raw = _json(binding)
        if (
            snapshot.run_ref.sha256 != hashlib.sha256(raw).hexdigest()
            or snapshot.run_ref.byte_count != len(raw)
            or snapshot.training_record_count != count
        ):
            raise ValueError("original RUN/count replay differs")
        for _name, ref in snapshot.staged_refs:
            self._verified(ref, intent.catalog, deadline)
        if self._raw(snapshot.run_ref, intent.catalog, deadline, collect=True) != raw:
            raise ValueError("original RUN bytes differ")
        handle.extra.update(
            request=json.loads(snapshot.request_json),
            run_ref=snapshot.run_ref,
            staged_refs=dict(snapshot.staged_refs),
            training_record_count=count,
        )
        status = self.status(handle, _reload_compatibility_reader=_reload_compatibility_reader)
        if status.state is not JobState.UNKNOWN:
            handle.extra.update(submission_state="accepted", restoration_status="verified")
            if reservation.disposition != "ACCEPTED":
                handle.extra["durability_status"] = "outcome_uncertain"
        self._require_reload_selection(
            intent.reload_compatibility, intent.catalog, _reload_compatibility_reader
        )
        return handle

    def _verified_run_binding(self, *, model, catalog, session, other_files, binding, prepared, deadline):
        validate_catalog(catalog)
        _sm, s3 = self._clients(catalog.region)
        binding = dict(binding)
        if (
            s3.get_bucket_versioning(Bucket=catalog.bucket, ExpectedBucketOwner=catalog.account_id).get(
                "Status"
            )
            != "Enabled"
        ):
            raise ValueError("SFT stage bucket versioning required")
        # All original immutable data is independently verified before any staging write.
        original = [
            *dict(model.file_refs).values(),
            *dict(model.manifest_refs).values(),
            *other_files.values(),
        ]
        if prepared is not None:
            original.append(prepared.exclusions_ref)
        for ref in original:
            self._verified(ref, catalog, deadline)
        if prepared is not None:
            commit = decode_object(
                self._raw(prepared.preparation_ref, catalog, deadline, collect=True), limit=_METADATA_LIMIT
            )
            if (
                any(commit.get(k) != v for k, v in prepared.header)
                or type(commit.get("schema_version")) is not int
                or commit["schema_version"] != 2
                or type(commit.get("files")) is not dict
                or set(commit["files"]) != set(other_files) - {"COMMITTED.json"}
            ):
                raise ValueError("SFT preparation original commit differs")
            if commit.get("exclusions_sha256") != prepared.exclusions_ref.sha256:
                raise ValueError("SFT original exclusion policy pin differs")
            for name, ref in other_files.items():
                if name == "COMMITTED.json":
                    continue
                value = commit["files"][name]
                if (
                    type(value) is not dict
                    or set(value) != {"sha256", "byte_count", "format", "record_count"}
                    or {k: value[k] for k in ("sha256", "byte_count")} != _descriptor(ref)
                    or value["format"] != ref.format
                    or type(value["record_count"]) is not int
                    or not 0 < value["record_count"] <= 100000
                ):
                    raise ValueError("SFT preparation exact descriptor differs")
            binding["split_refs"] = {
                split: dict(commit["files"][split + ".jsonl"]) for split in ("train", "validation", "test")
            }
            split_counts = commit.get("counts", {}).get("split_counts")
            if (
                type(split_counts) is not dict
                or set(split_counts) != {"train", "validation", "test"}
                or any(
                    type(count) is not int
                    or not 0 < count <= 100000
                    or count != binding["split_refs"][split]["record_count"]
                    for split, count in split_counts.items()
                )
            ):
                raise ValueError("SFT preparation split counter binding differs")

            counts = commit.get("counts", {})
            source_count = counts.get("source", counts.get("source_count"))
            if prepared.source_row_counts and (
                type(source_count) is not int
                or source_count != sum(dict(prepared.source_row_counts).values())
            ):
                raise ValueError("SFT preparation private original row counts differ")
        if prepared is None:
            from ...training.qwen_sft import _checkpoint_metadata, validate_reload_binding

            binding = validate_reload_binding(
                binding, source_commit=catalog.source_commit, recipe_id=binding["recipe_id"]
            )
            original_checkpoint = _checkpoint_metadata(
                decode_object(
                    self._raw(other_files["CHECKPOINT.json"], catalog, deadline, collect=True),
                    limit=_METADATA_LIMIT,
                )
            )
            producer = (
                catalog.source_commit
                if binding["schema_version"] == 1
                else binding["checkpoint_source_commit"]
            )
            if (
                original_checkpoint["source_commit"] != producer
                or original_checkpoint["recipe_id"] != binding["recipe_id"]
                or original_checkpoint["files"]
                != {
                    name: _descriptor(other_files[name])
                    for name in ("adapter_config.json", "adapter_model.safetensors")
                }
            ):
                raise ValueError("original reload metadata source/files differ")
        count = binding["split_refs"]["train"]["record_count"] if prepared is not None else None
        return binding, count

    def _stage_create(
        self,
        *,
        model: VerifiedSftModelSource,
        catalog: SftCatalog,
        session: str,
        recipe_id: str,
        operation: str,
        other_files: dict,
        other_channel: str,
        binding: dict,
        prepared: Any,
        job: str,
        reservation: Any = None,
        campaign_state: Any = None,
        execution_binding: Any = None,
        reload_compatibility: Any = None,
        reload_compatibility_reader: Any = None,
        resource_contract: str | None = None,
    ) -> JobHandle:
        if reservation is not None:
            # The original reserved intent, never a request/body field, selects resources.
            resource_contract = reservation.intent.reload_resource_contract
        self._require_reload_selection(reload_compatibility, catalog, reload_compatibility_reader)
        deadline = time.monotonic() + 3600
        binding, training_count = self._verified_run_binding(
            model=model,
            catalog=catalog,
            session=session,
            other_files=other_files,
            binding=binding,
            prepared=prepared,
            deadline=deadline,
        )
        sm, s3 = self._clients(catalog.region)
        prefix = "sessions/" + session + "/sft/" + job + "/"
        staged = {}
        for channel, files in [("base_model", dict(model.file_refs)), (other_channel, other_files)]:
            for name, ref in files.items():
                self._require_reload_selection(reload_compatibility, catalog, reload_compatibility_reader)
                staged[channel + "/" + name] = self._copy(
                    ref, prefix + "input/" + channel + "/" + name, catalog, deadline
                )
        raw = _json(binding)
        if not 0 < len(raw) <= _METADATA_LIMIT:
            raise ValueError("SFT RUN bound exceeded")
        key = prefix + "input/bindings/RUN.json"
        self._require_reload_selection(reload_compatibility, catalog, reload_compatibility_reader)
        result = s3.put_object(
            Bucket=catalog.bucket,
            Key=key,
            Body=raw,
            IfNoneMatch="*",
            ExpectedBucketOwner=catalog.account_id,
            ContentType="application/json",
        )
        run_ref = DatasetArtifactRef(
            "s3://" + catalog.bucket + "/" + key,
            _version(result.get("VersionId")),
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            "json",
        )
        self._raw(run_ref, catalog, deadline)
        staged["bindings/RUN.json"] = run_ref
        request = fixed_sft_request(
            catalog, session, job, operation, recipe_id, run_ref, resource_contract=resource_contract
        )
        arn = "arn:aws:sagemaker:" + catalog.region + ":" + catalog.account_id + ":training-job/" + job
        handle = JobHandle(
            job,
            "qwen_sft",
            arn,
            {
                "runtime_session_id": session,
                "catalog": catalog,
                "request": request,
                "model_source": model,
                "prepared_source": prepared,
                "run_ref": run_ref,
                "training_record_count": training_count,
                "staged_refs": staged,
                "operation": operation,
                "recipe_id": recipe_id,
                "submission_state": "ambiguous",
                **(
                    {"reload_compatibility": reload_compatibility} if reload_compatibility is not None else {}
                ),
            },
        )
        if campaign_state is not None:
            handle.extra.update(
                campaign_reservation=reservation,
                operation_digest=reservation.operation_digest,
                durability_status="recorded",
            )
            snapshot = SftSubmissionSnapshot(
                reservation.operation_digest,
                job,
                arn,
                json.dumps(
                    request, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
                ),
                run_ref,
                tuple(sorted(staged.items())),
                training_count,
            )
            try:
                reservation = campaign_state.attach_submission_snapshot(
                    execution_binding, reservation, snapshot
                )
                handle.extra["campaign_reservation"] = reservation
            except Exception as error:
                handle.extra.update(durability_status="snapshot_uncertain", error_type=type(error).__name__)
                return handle
        try:
            self._require_reload_selection(reload_compatibility, catalog, reload_compatibility_reader)
            result = sm.create_training_job(**request)
            if type(result) is not dict or result.get("TrainingJobArn") != arn:
                raise ValueError("SFT native acceptance identity differs")
            handle.extra["submission_state"] = "accepted"
        except Exception as error:
            response = getattr(error, "response", {})
            code = response.get("Error", {}).get("Code") if type(response) is dict else None
            if code in ("ValidationException", "AccessDeniedException", "ResourceLimitExceeded"):
                handle.extra["submission_state"] = "rejected"
            handle.extra["error_type"] = type(error).__name__
        if campaign_state is not None:
            try:
                reservation = campaign_state.record_submission_outcome(
                    execution_binding,
                    reservation,
                    disposition=handle.extra["submission_state"].upper(),
                    native_job_arn=arn if handle.extra["submission_state"] == "accepted" else None,
                    error_code=None
                    if handle.extra["submission_state"] == "accepted"
                    else handle.extra["error_type"],
                )
                handle.extra["campaign_reservation"] = reservation
            except Exception as error:
                # Native acceptance is retained even if durable outcome acknowledgement fails.
                handle.extra.update(
                    durability_status="outcome_uncertain", persistence_error_type=type(error).__name__
                )
        return handle

    def status(self, handle: JobHandle, *, _reload_compatibility_reader=None) -> JobStatus:
        try:
            if type(handle) is not JobHandle or handle.backend != self.backend_name:
                raise ValueError("original SFT handle required")
            catalog = handle.extra["catalog"]
            validate_catalog(catalog)
            self._require_reload_selection(
                handle.extra.get("reload_compatibility"), catalog, _reload_compatibility_reader
            )
            sm = self._clients(catalog.region)[0]
            desc = self._observation_io(sm.describe_training_job, TrainingJobName=handle.id)
            request = handle.extra["request"]
            fields = (
                "TrainingJobName",
                "RoleArn",
                "AlgorithmSpecification",
                "InputDataConfig",
                "ResourceConfig",
                "StoppingCondition",
                "EnableNetworkIsolation",
                "EnableInterContainerTrafficEncryption",
                "OutputDataConfig",
            )
            view = _native_request_view(desc, request)
            if desc.get("TrainingJobArn") != handle.arn or any(view.get(k) != request[k] for k in fields):
                raise ValueError("SFT native request readback differs")
            for field in ("Environment", "HyperParameters"):
                if field in desc and (type(desc[field]) is not dict or desc[field]):
                    raise ValueError("SFT forbidden native field differs")
            tags = self._observation_io(sm.list_tags, ResourceArn=handle.arn)
            values = tags.get("Tags")
            if type(values) is not list or any(
                type(tag) is not dict
                or set(tag) != {"Key", "Value"}
                or type(tag["Key"]) is not str
                or type(tag["Value"]) is not str
                for tag in values
            ):
                raise ValueError("SFT native tag shape differs")
            observed = {tag["Key"]: tag["Value"] for tag in values}
            if (
                tags.get("NextToken")
                or len(observed) != len(values)
                or observed != {tag["Key"]: tag["Value"] for tag in request["Tags"]}
            ):
                raise ValueError("SFT native ownership tags differ")
            self._require_reload_selection(
                handle.extra.get("reload_compatibility"), catalog, _reload_compatibility_reader
            )
            self._qualify_io()
            observed = JobStatus(_STATES.get(desc.get("TrainingJobStatus"), JobState.UNKNOWN), "", desc)
            # The detail names the selected instance only for a mapped native state.
            return dataclasses.replace(
                observed, detail=selected_instance_detail(observed_selected_instance(observed, request))
            )
        except Exception as error:
            return JobStatus(
                JobState.UNKNOWN, "native SFT observation failed", {"error_type": type(error).__name__}
            )

    def _output(
        self, handle: JobHandle, key: str, deadline: float, *, limit: int, collect: bool = False
    ) -> tuple[DatasetArtifactRef, bytes | None]:
        catalog = handle.extra["catalog"]
        s3 = self._clients(catalog.region)[1]
        _deadline(deadline)
        head = self._observation_io(
            s3.head_object,
            Bucket=catalog.bucket,
            Key=key,
            ExpectedBucketOwner=catalog.account_id,
            ChecksumMode="ENABLED",
        )
        _deadline(deadline)
        count = head.get("ContentLength")
        version = _version(head.get("VersionId"))
        if type(count) is not int or not 0 < count <= limit:
            raise ValueError("SFT output size bound differs")
        # Unknown digest is computed from this exact version, even if service checksum exists.
        result = self._observation_io(
            s3.get_object,
            Bucket=catalog.bucket,
            Key=key,
            ExpectedBucketOwner=catalog.account_id,
            VersionId=version,
        )
        body = result["Body"]
        with self._owned_observation_body(body, deadline=deadline):
            if (
                result.get("VersionId") != version
                or type(result.get("ContentLength")) is not int
                or result["ContentLength"] != count
            ):
                raise ValueError("SFT output version changed")
            total = 0
            hashed = hashlib.sha256()
            parts = []
            while True:
                _deadline(deadline)
                chunk = self._observation_read(body, min(1024 * 1024, count - total + 1))
                _deadline(deadline)
                if not chunk:
                    break
                total += len(chunk)
                if total > count:
                    raise ValueError("SFT output raw count exceeded")
                hashed.update(chunk)
                if collect:
                    parts.append(chunk)
            if total != count:
                raise ValueError("SFT output count differs")
            value = (
                DatasetArtifactRef(
                    "s3://" + catalog.bucket + "/" + key,
                    version,
                    hashed.hexdigest(),
                    count,
                    "safetensors" if key.endswith(".safetensors") else "json",
                ),
                b"".join(parts) if collect else None,
            )
        self._qualify_io()
        return value

    def artifacts(self, handle: JobHandle, *, _reload_compatibility_reader=None) -> dict:
        status = self.status(handle, _reload_compatibility_reader=_reload_compatibility_reader)
        recipe_id = handle.extra["recipe_id"]
        operation = handle.extra["operation"]
        base = {
            "success": False,
            "accepted": handle.extra["submission_state"] == "accepted",
            "trained": False,
            "status": status.state.value,
            "job_name": handle.id,
            "job_arn": handle.arn,
            "recipe_id": recipe_id,
            "operation": operation,
            "training_stage": "fit" if recipe_id == "fit_v1" else "full",
            "full_training_complete": False,
            "quality_success": False,
            "production_admission": False,
            **observed_selected_instance(status, handle.extra.get("request")),
        }
        if status.state != JobState.SUCCEEDED:
            return base
        try:
            catalog = handle.extra["catalog"]
            selection = handle.extra.get("reload_compatibility")
            self._require_reload_selection(selection, catalog, _reload_compatibility_reader)
            selected_version = 2 if selection is not None else 1
            compatibility_fields = (
                {}
                if selection is None
                else {
                    "checkpoint_source_commit": selection.checkpoint.source_commit,
                    "reload_contract_version": selection.contract_version,
                    "compatibility_digest": reload_compatibility_digest(selection),
                }
            )
            prefix = _native_uncompressed_output_prefix(
                (status.raw.get("ModelArtifacts") or {}).get("S3ModelArtifacts"),
                s3_output_path=handle.extra["request"]["OutputDataConfig"]["S3OutputPath"],
                job_name=handle.id,
                bucket=catalog.bucket,
            )
            s3 = self._clients(catalog.region)[1]
            deadline = time.monotonic() + 600
            commit_ref, raw = self._output(
                handle, prefix + "COMMITTED.json", deadline, limit=_METADATA_LIMIT, collect=True
            )
            commit = decode_object(raw, limit=_METADATA_LIMIT)
            expected = (
                {"schema_version", "operation", "source_commit", "files", "quality_success"}
                | set(compatibility_fields)
                | ({"status", "reload_pending"} if operation == "train" else set())
            )
            if (
                set(commit) != expected
                or type(commit["schema_version"]) is not int
                or commit["schema_version"] != selected_version
                or any(commit[name] != value for name, value in compatibility_fields.items())
                or commit["operation"] != operation
                or commit["source_commit"] != catalog.source_commit
                or commit["quality_success"] is not False
            ):
                raise ImmutableArtifactRejectionError("closed_manifest_invalid")
            names = (
                {"adapter_model.safetensors", "adapter_config.json", "CHECKPOINT.json", "TRAINING.json"}
                if operation == "train"
                else {"RELOAD.json"}
            )
            if (
                type(commit["files"]) is not dict
                or set(commit["files"]) != names
                or (
                    operation == "train"
                    and (commit["status"] != "checkpoint_saved" or commit["reload_pending"] is not True)
                )
            ):
                raise ValueError("SFT final closed inventory differs")
            listing = self._observation_io(
                s3.list_objects_v2,
                Bucket=catalog.bucket,
                Prefix=prefix,
                ExpectedBucketOwner=catalog.account_id,
                MaxKeys=8,
            )
            if (
                listing.get("IsTruncated")
                or {x["Key"] for x in listing.get("Contents", [])}
                != {prefix + x for x in names | {"COMMITTED.json"}}
                or len(listing.get("Contents", [])) != len(names) + 1
            ):
                raise ValueError("SFT output contains extra/missing files")
            refs = {}
            values = {}
            for name in sorted(names):
                ref, raw = self._output(
                    handle,
                    prefix + name,
                    deadline,
                    limit=2 * 1024**3 if name.endswith(".safetensors") else _METADATA_LIMIT,
                    collect=not name.endswith(".safetensors"),
                )
                if not _fixed_output_descriptor_matches(commit["files"][name], ref, name):
                    raise ValueError("SFT final output exact digest/count differs")
                refs[name] = ref
                if name.endswith(".json"):
                    values[name] = decode_object(raw, limit=_METADATA_LIMIT)
            receipt = values["TRAINING.json" if operation == "train" else "RELOAD.json"]
            if (
                receipt.get("source_commit") != catalog.source_commit
                or receipt.get("recipe_id") != recipe_id
                or receipt.get("operation") != operation
                or receipt.get("catalog_id") != "qwen3_5_9b"
                or receipt.get("catalog_version") != "1"
                or receipt.get("quality_success") is not False
            ):
                raise ValueError("SFT operation receipt source differs")
            if any(receipt.get(name) != value for name, value in compatibility_fields.items()):
                raise ValueError("SFT selected reload compatibility receipt differs")
            receipt_keys = (
                set(compatibility_fields)
                | {
                    "schema_version",
                    "operation",
                    "source_commit",
                    "catalog_id",
                    "catalog_version",
                    "recipe_id",
                    "privilege_proof",
                    "quality_success",
                }
                | (
                    {"input_summary", "metrics", "status", "reload_pending"}
                    if operation == "train"
                    else {"checkpoint_sha256", "fresh_process_greedy_equal"}
                )
            )
            if (
                set(receipt) != receipt_keys
                or type(receipt["schema_version"]) is not int
                or receipt["schema_version"] != selected_version
            ):
                raise ValueError("SFT closed operation receipt differs")
            if operation == "train":
                summary = receipt["input_summary"]
                if (
                    type(summary) is not dict
                    or summary.get("test_used_for_training") is not False
                    or summary.get("preparation_commit_sha256")
                    != handle.extra["prepared_source"].preparation_ref.sha256
                    or receipt["status"] != "checkpoint_saved"
                    or receipt["reload_pending"] is not True
                ):
                    raise ValueError("SFT original preparation consumption differs")
                prepared = handle.extra["prepared_source"]
                original_header = dict(prepared.header)
                if (original_header["profile_id"], original_header["policy_version"]) == (
                    "bigcode.commitpackft.python.change_sft.v1",
                    "coding-preparation.commitpack.v3",
                ):
                    from ...training.qwen_sft import _validate_commitpack_summary

                    compact = summary.get("commitpack")
                    _validate_commitpack_summary(compact)
                    if prepared.preparation_ref.byte_count > _METADATA_LIMIT:
                        raise ValueError("SFT original preparation metadata bound differs")
                    original = decode_object(
                        self._raw(prepared.preparation_ref, catalog, deadline, collect=True),
                        limit=_METADATA_LIMIT,
                    )
                    if (
                        summary.get("preparation_profile_id") != original_header["profile_id"]
                        or summary.get("preparation_policy_version") != original_header["policy_version"]
                        or any(original.get(name) != part for name, part in original_header.items())
                        or compact["counts"] != original.get("counts")
                        or _json(compact["coverage"]) != _json(original.get("coverage"))
                        or compact["exclusion_policy_sha256"] != prepared.exclusions_ref.sha256
                        or compact["counts"]["source"] != sum(dict(prepared.source_row_counts).values())
                    ):
                        raise ValueError("SFT bounded summary original preparation differs")
                    # Parent checks original pins/counts/closed formats. It does not independently
                    # authenticate the guest's ordered-row digest without replaying those rows.
            else:
                if (
                    receipt["checkpoint_sha256"]
                    != handle.extra["staged_refs"]["adapter/CHECKPOINT.json"].sha256
                ):
                    raise ValueError("SFT original reload checkpoint differs")
            validate_native_privilege_proof(receipt["privilege_proof"])
            if operation == "reload":
                if receipt.get("fresh_process_greedy_equal") is not True:
                    raise ValueError("SFT fresh reload equality absent")
                self._require_reload_selection(selection, catalog, _reload_compatibility_reader)
                self._qualify_io()
                return {
                    **base,
                    "success": True,
                    "status": "reloaded",
                    "fresh_process_greedy_equal": True,
                    "reload_ref": refs["RELOAD.json"],
                    "commit_ref": commit_ref,
                }
            from ...training.qwen_sft import _checkpoint_metadata

            checkpoint = _checkpoint_metadata(values["CHECKPOINT.json"])
            if (
                checkpoint["source_commit"] != catalog.source_commit
                or checkpoint["stage_manifest_sha256"] != handle.extra["run_ref"].sha256
                or checkpoint["recipe_id"] != recipe_id
                or checkpoint["files"]
                != {k: _descriptor(refs[k]) for k in ("adapter_config.json", "adapter_model.safetensors")}
            ):
                raise ValueError("SFT original RUN/checkpoint differs")
            metrics = receipt.get("metrics")
            self._metrics(metrics, recipe_id, maximum_training_rows=handle.extra["training_record_count"])
            if (checkpoint["optimizer_steps"], checkpoint["supervised_tokens"]) != (
                metrics["optimizer_steps"],
                metrics["supervised_tokens"],
            ):
                raise ValueError("SFT checkpoint exposure counters differ")
            self._qualify_io()
            return {
                **base,
                "success": True,
                "trained": True,
                "status": "checkpoint_saved",
                "reload_pending": True,
                "metrics": metrics,
                "checkpoint_ref": refs["CHECKPOINT.json"],
                "checkpoint_refs": {
                    k: refs[k]
                    for k in ("CHECKPOINT.json", "adapter_model.safetensors", "adapter_config.json")
                },
                "commit_ref": commit_ref,
                "training_ref": refs["TRAINING.json"],
                "full_training_complete": recipe_id == "full_v1",
            }
        except Exception as error:
            return {
                **base,
                "status": "verification_failed",
                "error_type": type(error).__name__,
                **rejection_data(error),
            }

    @staticmethod
    def _metrics(value: Any, recipe_id: str, *, maximum_training_rows: int | None = None) -> None:
        if (
            type(value) is not dict
            or value.get("recipe_id") != recipe_id
            or value.get("target_met") is not True
            or value.get("quality_success") is not False
            or value.get("reload_pending") is not True
            or value.get("vision_frozen") is not True
            or value.get("optimizer_state_saved") is not False
        ):
            raise ValueError("SFT actual metrics required")
        expected_keys = {
            "schema_version",
            "recipe_id",
            "optimizer_steps",
            "supervised_tokens",
            "unique_training_rows",
            "original_provenance_rows",
            "unique_supervised_tokens",
            "optimizer_exposure_tokens",
            "epochs_completed",
            "changed_tensor_count",
            "train_history",
            "validation_history",
            "early_stop_would_trigger",
            "target_met",
            "status",
            "reload_pending",
            "quality_success",
            "vision_frozen",
            "peak_device_memory_bytes",
            "elapsed_seconds",
            "tokens_per_second",
            "optimizer_state_saved",
        }
        if (
            set(value) != expected_keys
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["status"] != "checkpoint_ready"
            or type(value["early_stop_would_trigger"]) is not bool
        ):
            raise ValueError("SFT closed actual metrics differ")
        counts = (
            "optimizer_steps",
            "supervised_tokens",
            "unique_training_rows",
            "original_provenance_rows",
            "unique_supervised_tokens",
            "optimizer_exposure_tokens",
            "epochs_completed",
            "changed_tensor_count",
            "peak_device_memory_bytes",
        )
        if (
            any(
                type(value.get(k)) is not int
                or not (0 if k == "epochs_completed" else 1) <= value[k] <= 10**15
                for k in counts
            )
            or value["optimizer_steps"] != (32 if recipe_id == "fit_v1" else 1024)
            or value["optimizer_exposure_tokens"] != value["supervised_tokens"]
        ):
            raise ValueError("SFT actual recipe counters differ")
        if recipe_id == "full_v1" and (
            value["unique_training_rows"] < 1000
            or value["unique_supervised_tokens"] < 1000000
            or value["supervised_tokens"] < 1000000
        ):
            raise ValueError("SFT full unique/exposure floors absent")
        for k in ("elapsed_seconds", "tokens_per_second"):
            if type(value.get(k)) not in (int, float) or not math.isfinite(value[k]) or value[k] <= 0:
                raise ValueError("SFT finite timing required")
        if value["unique_training_rows"] > value["original_provenance_rows"]:
            raise ValueError("SFT unique/provenance row counters differ")
        if maximum_training_rows is not None and (
            type(maximum_training_rows) is not int
            or not 0 < maximum_training_rows <= 100000
            or value["original_provenance_rows"] > maximum_training_rows
        ):
            raise ValueError("SFT original prepared TRAIN count exceeded")
        steps = value["optimizer_steps"]
        history = value["train_history"]
        if type(history) is not list or len(history) != steps:
            raise ValueError("SFT optimizer history coverage differs")
        for index, row in enumerate(history, 1):
            if (
                type(row) is not dict
                or set(row) != {"step", "loss", "supervised_tokens", "gradient_norm"}
                or type(row["step"]) is not int
                or row["step"] != index
                or type(row["supervised_tokens"]) is not int
                or not 0 < row["supervised_tokens"] <= 8 * 4096
                or any(
                    type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
                    for k in ("loss", "gradient_norm")
                )
            ):
                raise ValueError("SFT actual optimizer step witness differs")
        if sum(row["supervised_tokens"] for row in history) != value["supervised_tokens"]:
            raise ValueError("SFT optimizer exposure accounting differs")
        validation = value["validation_history"]
        expected_steps = [step for step in range(1, steps + 1) if step % 64 == 0 or step == steps]
        if type(validation) is not list or len(validation) != len(expected_steps):
            raise ValueError("SFT validation witness coverage differs")
        for expected, row in zip(expected_steps, validation, strict=True):
            if (
                type(row) is not dict
                or set(row)
                != {"step", "loss", "best_loss", "nonimproving_evaluations", "early_stop_would_trigger"}
                or type(row["step"]) is not int
                or row["step"] != expected
                or type(row["nonimproving_evaluations"]) is not int
                or not 0 <= row["nonimproving_evaluations"] <= len(expected_steps)
                or type(row["early_stop_would_trigger"]) is not bool
                or any(
                    type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
                    for k in ("loss", "best_loss")
                )
            ):
                raise ValueError("SFT actual validation step witness differs")
        if value["early_stop_would_trigger"] != validation[-1]["early_stop_would_trigger"]:
            raise ValueError("SFT last validation witness differs")
