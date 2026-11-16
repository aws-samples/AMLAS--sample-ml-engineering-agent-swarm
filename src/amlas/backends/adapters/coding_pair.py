"""Fixed paired Processing dispatch; bounded raw artifacts, never model or code execution."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import time

from ...control_plane.generation_state import VerifiedCodingGenerationSource, validate_cases
from ...control_plane.paired_state import PairedCodingInputs, pair_operation_digest, validate_pair_catalog
from ...control_plane.sft_state import validate_catalog, validate_ref
from ...data_eng.inspection_schema import decode_object
from ...training.coding_generation import canonical_bytes
from ..types import DatasetArtifactRef, JobHandle, JobState, JobStatus
from .artifact_rejection import ImmutableArtifactRejectionError, rejection_data
from .qwen_sft import SageMakerQwenSftBackend, _deadline, _json, _version

BUDGETS = {
    "case_wall_seconds": 10,
    "session_timeout_seconds": 60,
    "max_value_calls": 512,
    "allocation_excluded": True,
    "cleanup_seconds": 30,
}
_CORE_NAMES = {"RECEIPTS.jsonl", "EVALUATION.json", "COMMITTED.json", "INVOCATIONS_MANIFEST.jsonl"}
_MAX_RECORDS = 1_000_000
_INVOCATION_NAME = re.compile(
    r"INVOCATIONS/(base|tuned)/([0-9]{3})/(session|calls)/([0-9]{8})-([a-z-]+)\.json"
)
_SESSION_STAGES = {
    "start-attempt",
    "start-accepted",
    "start-unknown",
    "stop-attempt",
    "stop-accepted",
    "get-attempt",
    "get-observed",
    "cleanup-unknown",
}
_CALL_STAGES = {"attempt", "response", "event", "completed", "unknown"}
# Tag namespaces that only the product (amlas:) or AWS (aws:) writes. A stored request may carry other tags that
# the deployment adds at create, such as its own project and retention tags; they do not bear on ownership.
_RESERVED_TAG_PREFIXES = ("amlas:", "aws:")
# The real 164-case pair of 2026-11-10 wrote 12,786 invocation records. Through this adapter, with plain
# clients from the job's own Region, artifacts() took 2,159.7 s, almost all of it the record pass of five S3 calls a
# record: 0.169 s a record (an earlier shimmed pass took 0.111 s). After the fixed budget of the core outputs, the
# record pass gets about three times the slower rate per declared record.
_CORE_SECONDS = 600
_RECORD_MILLISECONDS = 500
_STATES = {
    "InProgress": JobState.RUNNING,
    "Stopping": JobState.RUNNING,
    "Completed": JobState.SUCCEEDED,
    "Failed": JobState.FAILED,
    "Stopped": JobState.STOPPED,
}


def validate_pair_artifacts(*args, **kwargs):
    from ...evaluation.coding_pair import validate_pair_artifacts as validate

    return validate(*args, **kwargs)


def verify_pair_result(*args, **kwargs):
    from ...evaluation.coding_pair import verify_pair_result as verify

    return verify(*args, **kwargs)


def paired_runner_sha256(*, oracle_launcher_sha256):
    from ...evaluation.coding_pair_runner import paired_runner_sha256 as fingerprint

    return fingerprint(oracle_launcher_sha256=oracle_launcher_sha256)


def _input_groups(inputs, catalog, session):
    if type(inputs) is not PairedCodingInputs:
        raise ValueError("private typed paired inputs required")
    cases = validate_cases(inputs.cases, catalog, session)
    groups = {"cases": cases}
    for arm in ("base", "tuned"):
        source = getattr(inputs, arm)
        if (
            type(source) is not VerifiedCodingGenerationSource
            or source.runtime_session_id != session
            or source.arm != arm
        ):
            raise ValueError("original typed paired arm/session differs")
        prefix = (
            "arn:aws:sagemaker:"
            + catalog.region
            + ":"
            + catalog.account_id
            + ":training-job/"
            + catalog.deployment_prefix
            + "-"
        )
        if (
            type(source.producer_job_arn) is not str
            or not source.producer_job_arn.startswith(prefix)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", source.evaluation_id)
        ):
            raise ValueError("original generation producer/evaluation differs")
        refs = dict(source.file_refs)
        if len(refs) != len(source.file_refs) or set(refs) != {
            "CANDIDATES.jsonl",
            "GENERATION.json",
            "COMMITTED.json",
        }:
            raise ValueError("original closed generation inventory required")
        for name, artifact in refs.items():
            validate_ref(
                artifact, catalog, session, limit=12 * 1024**2 if name == "CANDIDATES.jsonl" else 262144
            )
            if artifact.format != ("jsonl" if name.endswith(".jsonl") else "json"):
                raise ValueError("original generation artifact format differs")
        groups[arm] = refs
    if (
        inputs.base.evaluation_id != inputs.tuned.evaluation_id
        or inputs.base.producer_job_arn == inputs.tuned.producer_job_arn
    ):
        raise ValueError("original paired identities differ")
    if (
        type(inputs.spec_bytes) is not bytes
        or not 0 < len(inputs.spec_bytes) <= 65536
        or type(inputs.spec_sha256) is not str
        or hashlib.sha256(inputs.spec_bytes).hexdigest() != inputs.spec_sha256
    ):
        raise ValueError("original canonical spec bytes differ")
    return groups


def _request(catalog, session, job, run_ref, *, oracle_launcher_sha256):
    validate_catalog(catalog)
    if type(oracle_launcher_sha256) is not str or not re.fullmatch("[a-f0-9]{64}", oracle_launcher_sha256):
        raise ValueError("fixed trusted launcher environment digest required")
    validate_ref(run_ref, catalog, session, limit=262144)
    prefix = "sessions/" + session + "/paired/" + job + "/"
    if (
        not re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}", job)
        or run_ref.uri != "s3://" + catalog.bucket + "/" + prefix + "input/bindings/RUN.json"
    ):
        raise ValueError("original paired job/RUN namespace differs")
    return {
        "ProcessingJobName": job,
        "RoleArn": catalog.execution_role_arn,
        "Environment": {"AMLAS_CODING_PAIR_LAUNCHER_SHA256": oracle_launcher_sha256},
        "AppSpecification": {
            "ImageUri": catalog.image_uri,
            "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "coding-pair"],
            "ContainerArguments": [
                "--binding-sha256",
                run_ref.sha256,
                "--binding-byte-count",
                str(run_ref.byte_count),
                "--source-commit",
                catalog.source_commit,
            ],
        },
        "ProcessingResources": {
            "ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.xlarge", "VolumeSizeInGB": 30}
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 14400},
        "NetworkConfig": {"EnableNetworkIsolation": False, "EnableInterContainerTrafficEncryption": False},
        "ProcessingInputs": [
            {
                "InputName": "paired-" + group,
                "S3Input": {
                    "S3Uri": "s3://" + catalog.bucket + "/" + prefix + "input/" + group + "/",
                    "LocalPath": "/opt/ml/processing/input/" + group,
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            }
            for group in ("cases", "base", "tuned", "bindings")
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "paired-output",
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
                ("runtime-session-id", session),
            )
        ],
    }


def _run_binding(inputs, catalog, session, evaluation_binding, staged, spec_ref, *, oracle_launcher_sha256):
    groups = _input_groups(inputs, catalog, session)
    return {
        "schema_version": 1,
        "operation": "paired_eval",
        "source_commit": catalog.source_commit,
        "oracle_launcher_sha256": oracle_launcher_sha256,
        "session": session,
        "account_id": catalog.account_id,
        "region": catalog.region,
        "deployment_id": catalog.deployment_id,
        "deployment_prefix": catalog.deployment_prefix,
        "evaluation_binding": evaluation_binding.model_dump(mode="json"),
        "spec_sha256": inputs.spec_sha256,
        "spec_descriptor": {"sha256": spec_ref.sha256, "byte_count": spec_ref.byte_count, "format": "json"},
        "cases_source": dataclasses.asdict(inputs.cases),
        "generation_sources": {
            "base": dataclasses.asdict(inputs.base),
            "tuned": dataclasses.asdict(inputs.tuned),
        },
        "original_refs": {
            group: {name: dataclasses.asdict(ref) for name, ref in refs.items()}
            for group, refs in groups.items()
        },
        "staged_refs": {
            group: {name: dataclasses.asdict(ref) for name, ref in refs.items()}
            for group, refs in staged.items()
        },
        "staged_spec_ref": dataclasses.asdict(spec_ref),
    }


def _native_fields(description, request):
    """Normalize the documented false AppManaged default on copied I/O entries only."""
    if type(description) is not dict:
        raise ValueError("native Processing description required")
    view = dict(description)
    for field in ("ProcessingInputs", "ProcessingOutputConfig"):
        actual = description.get(field)
        expected = request[field]
        output = field == "ProcessingOutputConfig"
        if output:
            if type(actual) is not dict or set(actual) != set(expected):
                raise ValueError("native Processing output configuration differs")
            entries, wanted = actual["Outputs"], expected["Outputs"]
        else:
            entries, wanted = actual, expected
        if type(entries) is not list or len(entries) != len(wanted):
            raise ValueError("native Processing I/O inventory differs")
        copied = []
        for entry, original in zip(entries, wanted, strict=True):
            if type(entry) is not dict:
                raise ValueError("native Processing I/O entry differs")
            normalized = dict(entry)
            if "AppManaged" not in original and "AppManaged" in normalized:
                if normalized["AppManaged"] is not False:
                    raise ValueError("native AppManaged default differs")
                del normalized["AppManaged"]
            copied.append(normalized)
        view[field] = {**actual, "Outputs": copied} if output else copied
    return view


def _record_pass_seconds(record_count):
    """Seconds the exact-version read of `record_count` invocation records may take, at a third of the measured rate."""
    return math.ceil(record_count * _RECORD_MILLISECONDS / 1000)


def _submitted_tags(stored, fixed):
    """The tag list of a stored request: exactly the fixed ownership tags, plus unique tags outside the reserved
    namespaces. Any other set is foreign and refused."""
    if type(stored) is not list or any(
        type(tag) is not dict
        or set(tag) != {"Key", "Value"}
        or type(tag["Key"]) is not str
        or type(tag["Value"]) is not str
        for tag in stored
    ):
        raise ValueError("original paired request tag shape differs")
    keys = [tag["Key"] for tag in stored]
    if len(set(keys)) != len(keys):
        raise ValueError("original paired request tag keys repeat")
    reserved = [tag for tag in stored if tag["Key"].startswith(_RESERVED_TAG_PREFIXES)]
    if _json(sorted(reserved, key=lambda tag: tag["Key"])) != _json(
        sorted(fixed, key=lambda tag: tag["Key"])
    ):
        raise ValueError("original paired ownership tags differ")
    return stored


def _record_name(name):
    match = _INVOCATION_NAME.fullmatch(name)
    if (
        not match
        or int(match[2]) > 163
        or match[5] not in (_SESSION_STAGES if match[3] == "session" else _CALL_STAGES)
    ):
        raise ValueError("closed invocation record name required")
    return match


def _validate_record(raw, name, evaluation_id):
    match = _record_name(name)
    record = decode_object(raw, limit=65536)
    case_id = "HumanEval/" + str(int(match[2]))
    if match[3] == "calls":
        from ...evaluation.coding_executor import _record_bytes

        _canonical_record, kind = _record_bytes(record)
        if (
            kind != match[5]
            or record["evaluation_id"] != evaluation_id
            or record["case_id"] != case_id
            or record["arm"] != match[1]
        ):
            raise ValueError("original hash-only invocation record scope differs")
    else:
        from ...evaluation.coding_pair_runner import validate_session_record

        validate_session_record(
            record, kind=match[5], case_id=case_id, arm=match[1], evaluation_id=evaluation_id
        )


class SageMakerPairedCodingBackend(SageMakerQwenSftBackend):
    """One fixed native broker; original handles survive Create and observer ambiguity."""

    backend_name = "paired_coding"

    def _read(self, ref, catalog, session, deadline, *, limit, current=True, collect=True, on_chunk=None):
        validate_ref(ref, catalog, session, limit=limit)
        _deadline(deadline)
        s3 = self._clients(catalog.region)[1]
        args = {
            "Bucket": catalog.bucket,
            "Key": ref.uri.split("/", 3)[3],
            "ExpectedBucketOwner": catalog.account_id,
        }
        head = self._observation_io(s3.head_object, **args, VersionId=ref.version_id)
        _deadline(deadline)
        if (
            head.get("VersionId") != ref.version_id
            or type(head.get("ContentLength")) is not int
            or head["ContentLength"] != ref.byte_count
        ):
            raise ValueError("paired exact-version size differs")
        if current and self._observation_io(s3.head_object, **args).get("VersionId") != ref.version_id:
            raise ValueError("paired current object version differs")
        _deadline(deadline)
        response = self._observation_io(s3.get_object, **args, VersionId=ref.version_id)
        body = response["Body"]
        with self._owned_observation_body(body, deadline=deadline):
            if (
                response.get("VersionId") != ref.version_id
                or type(response.get("ContentLength")) is not int
                or response["ContentLength"] != ref.byte_count
            ):
                raise ValueError("paired raw response version/count differs")
            parts, count, digest = [], 0, hashlib.sha256()
            while True:
                _deadline(deadline)
                chunk = self._observation_read(body, min(65536, ref.byte_count - count + 1))
                _deadline(deadline)
                if type(chunk) is not bytes:
                    raise ValueError("paired raw response must contain bytes")
                if not chunk:
                    break
                count += len(chunk)
                if count > ref.byte_count:
                    raise ValueError("paired raw response exceeds byte bound")
                digest.update(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
                if collect:
                    parts.append(chunk)
            if count != ref.byte_count or digest.hexdigest() != ref.sha256:
                raise ValueError("paired independently read SHA/count differs")
            if current and self._observation_io(s3.head_object, **args).get("VersionId") != ref.version_id:
                raise ValueError("paired current object changed during read")
            _deadline(deadline)
            value = b"".join(parts) if collect else None
        self._qualify_io()
        return value

    def _put(self, raw, key, catalog, session, deadline):
        _deadline(deadline)
        existing = self._existing_version(key, catalog)
        response = (
            {"VersionId": existing}
            if existing is not None
            else self._clients(catalog.region)[1].put_object(
                Bucket=catalog.bucket,
                Key=key,
                Body=raw,
                ExpectedBucketOwner=catalog.account_id,
                IfNoneMatch="*",
                ContentType="application/json",
            )
        )
        ref = DatasetArtifactRef(
            "s3://" + catalog.bucket + "/" + key,
            _version(response.get("VersionId")),
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            "json",
        )
        self._read(ref, catalog, session, deadline, limit=262144)
        return ref

    def submit(self, *, inputs, catalog, runtime_session_id):
        pair_catalog = catalog
        catalog = validate_pair_catalog(pair_catalog)
        groups = _input_groups(inputs, catalog, runtime_session_id)
        deadline = time.monotonic() + 600
        sm, s3 = self._clients(catalog.region)
        if (
            s3.get_bucket_versioning(Bucket=catalog.bucket, ExpectedBucketOwner=catalog.account_id).get(
                "Status"
            )
            != "Enabled"
        ):
            raise ValueError("paired evaluation requires versioned selected storage")
        payloads = {}
        for group, refs in groups.items():
            for name, ref in refs.items():
                payloads[group + "/" + name] = self._read(
                    ref,
                    catalog,
                    runtime_session_id,
                    deadline,
                    limit=12 * 1024**2
                    if name == "CANDIDATES.jsonl"
                    else 2 * 1024**2
                    if group == "cases"
                    else 262144,
                )
        runner_hash = paired_runner_sha256(oracle_launcher_sha256=pair_catalog.launcher_sha256)
        if type(runner_hash) is not str or not re.fullmatch("[a-f0-9]{64}", runner_hash):
            raise ValueError("fixed source runner fingerprint missing")
        evaluation_binding, _cases, _candidates = validate_pair_artifacts(
            inputs,
            payloads,
            runner_sha256=runner_hash,
            runtime_sha256=catalog.admission_digest,
            budgets_sha256=hashlib.sha256(canonical_bytes(BUDGETS)).hexdigest(),
        )
        # The native name is a function of the request: a repeat after a worker loss meets the same job.
        job = (
            catalog.deployment_prefix
            + "-pair-"
            + pair_operation_digest(runtime_session_id, pair_catalog, inputs)[:16]
        )
        if len(job) > 63:
            raise ValueError("paired native job name bound exceeded")
        prefix = "sessions/" + runtime_session_id + "/paired/" + job + "/"
        staged = {}
        for group, refs in groups.items():
            staged[group] = {}
            for name, ref in refs.items():
                copied = self._adopt_or_copy(ref, prefix + "input/" + group + "/" + name, catalog, deadline)
                self._read(
                    copied,
                    catalog,
                    runtime_session_id,
                    deadline,
                    limit=12 * 1024**2
                    if name == "CANDIDATES.jsonl"
                    else 2 * 1024**2
                    if group == "cases"
                    else 262144,
                )
                staged[group][name] = copied
        spec_ref = self._put(
            inputs.spec_bytes, prefix + "input/bindings/SPEC.json", catalog, runtime_session_id, deadline
        )
        binding = _run_binding(
            inputs,
            catalog,
            runtime_session_id,
            evaluation_binding,
            staged,
            spec_ref,
            oracle_launcher_sha256=pair_catalog.launcher_sha256,
        )
        raw = _json(binding)
        if len(raw) > 262144:
            raise ValueError("paired RUN byte bound exceeded")
        run_ref = self._put(raw, prefix + "input/bindings/RUN.json", catalog, runtime_session_id, deadline)
        request = _request(
            catalog, runtime_session_id, job, run_ref, oracle_launcher_sha256=pair_catalog.launcher_sha256
        )
        arn = "arn:aws:sagemaker:" + catalog.region + ":" + catalog.account_id + ":processing-job/" + job
        handle = JobHandle(
            job,
            self.backend_name,
            arn,
            {
                "runtime_session_id": runtime_session_id,
                "catalog": catalog,
                "pair_catalog": pair_catalog,
                "request": request,
                "run_ref": run_ref,
                "run_binding": binding,
                "evaluation_binding": evaluation_binding,
                "inputs": inputs,
                "staged_refs": staged,
                "spec_ref": spec_ref,
                "output_prefix": prefix + "output/",
                "submission_state": "ambiguous",
            },
        )
        try:
            result = sm.create_processing_job(**request)
            if type(result) is not dict or result.get("ProcessingJobArn") != arn:
                raise ValueError("paired native accepted identity differs")
            handle.extra["submission_state"] = "accepted"
        except Exception as error:
            response = getattr(error, "response", {})
            nested = response.get("Error") if type(response) is dict else None
            code = nested.get("Code") if type(nested) is dict else None
            if code == "ResourceInUse" and self._is_this_native_job(
                sm.describe_processing_job, "ProcessingJobName", job, "ProcessingJobArn", arn
            ):
                # The deterministic name already belongs to this operation: a repeat after a worker loss.
                handle.extra["submission_state"] = "accepted"
                handle.extra["adopted_existing"] = True
                return handle
            if type(code) is str and code in (
                "ValidationException",
                "AccessDeniedException",
                "ResourceLimitExceeded",
            ):
                handle.extra["submission_state"] = "rejected"
                handle.extra["rejection_code"] = code
            handle.extra["error_type"] = type(error).__name__
        return handle

    def _handle(self, handle):
        if type(handle) is not JobHandle or handle.backend != self.backend_name:
            raise ValueError("original private paired handle required")
        pair_catalog = handle.extra["pair_catalog"]
        catalog = validate_pair_catalog(pair_catalog)
        if handle.extra["catalog"] != catalog:
            raise ValueError("original pair-only execution snapshot differs")
        session = handle.extra["runtime_session_id"]
        inputs = handle.extra["inputs"]
        groups = _input_groups(inputs, catalog, session)
        staged = handle.extra["staged_refs"]
        if type(staged) is not dict or set(staged) != set(groups):
            raise ValueError("original paired staged groups differ")
        prefix = "s3://" + catalog.bucket + "/sessions/" + session + "/paired/" + handle.id + "/input/"
        for group, originals in groups.items():
            if type(staged[group]) is not dict or set(staged[group]) != set(originals):
                raise ValueError("original paired staged inventory differs")
            for name, original in originals.items():
                artifact = staged[group][name]
                validate_ref(
                    artifact,
                    catalog,
                    session,
                    limit=12 * 1024**2
                    if name == "CANDIDATES.jsonl"
                    else 2 * 1024**2
                    if group == "cases"
                    else 262144,
                )
                if artifact.uri != prefix + group + "/" + name or (
                    artifact.sha256,
                    artifact.byte_count,
                    artifact.format,
                ) != (original.sha256, original.byte_count, original.format):
                    raise ValueError("original paired staged artifact differs")
        spec_ref = handle.extra["spec_ref"]
        validate_ref(spec_ref, catalog, session, limit=65536)
        if (
            spec_ref.uri != prefix + "bindings/SPEC.json"
            or spec_ref.sha256 != inputs.spec_sha256
            or spec_ref.byte_count != len(inputs.spec_bytes)
            or spec_ref.format != "json"
        ):
            raise ValueError("original paired staged specification differs")
        expected_binding = _run_binding(
            inputs,
            catalog,
            session,
            handle.extra["evaluation_binding"],
            staged,
            spec_ref,
            oracle_launcher_sha256=pair_catalog.launcher_sha256,
        )
        raw = _json(expected_binding)
        run_ref = handle.extra["run_ref"]
        if (
            _json(handle.extra["run_binding"]) != raw
            or run_ref.sha256 != hashlib.sha256(raw).hexdigest()
            or run_ref.byte_count != len(raw)
            or run_ref.format != "json"
        ):
            raise ValueError("original paired RUN packet bytes differ")
        request = _request(
            catalog, session, handle.id, run_ref, oracle_launcher_sha256=pair_catalog.launcher_sha256
        )
        stored = handle.extra["request"]
        if (
            type(stored) is not dict
            or set(stored) != set(request)
            or _json({key: value for key, value in request.items() if key != "Tags"})
            != _json({key: value for key, value in stored.items() if key != "Tags"})
            or handle.arn
            != "arn:aws:sagemaker:"
            + catalog.region
            + ":"
            + catalog.account_id
            + ":processing-job/"
            + handle.id
        ):
            raise ValueError("original private paired request/identity differs")
        # The live tags are compared with the stored request's own tags, which `_submitted_tags` bounds.
        return catalog, session, {**request, "Tags": _submitted_tags(stored["Tags"], request["Tags"])}

    def status(self, handle):
        try:
            catalog, _session, request = self._handle(handle)
            sm = self._clients(catalog.region)[0]
            description = self._observation_io(sm.describe_processing_job, ProcessingJobName=handle.id)
            view = _native_fields(description, request)
            if description.get("ProcessingJobArn") != handle.arn or any(
                _json(view.get(key)) != _json(value) for key, value in request.items() if key != "Tags"
            ):
                raise ValueError("paired original native Processing request differs")
            if (
                type(description.get("Environment")) is not dict
                or description["Environment"] != request["Environment"]
            ):
                raise ValueError("paired trusted native Environment differs")
            tags = self._observation_io(sm.list_tags, ResourceArn=handle.arn)
            values = tags.get("Tags")
            if type(values) is not list or any(
                type(tag) is not dict
                or set(tag) != {"Key", "Value"}
                or type(tag["Key"]) is not str
                or type(tag["Value"]) is not str
                for tag in values
            ):
                raise ValueError("paired native tag shape differs")
            observed = {tag["Key"]: tag["Value"] for tag in values}
            if (
                tags.get("NextToken")
                or len(observed) != len(values)
                or observed != {tag["Key"]: tag["Value"] for tag in request["Tags"]}
            ):
                raise ValueError("paired original ownership tags differ")
            self._qualify_io()
            return JobStatus(
                _STATES.get(description.get("ProcessingJobStatus"), JobState.UNKNOWN), raw=description
            )
        except Exception as error:
            return JobStatus(
                JobState.UNKNOWN,
                detail="paired native observation failed",
                raw={"error_type": type(error).__name__},
            )

    def _inventory(self, s3, catalog, prefix, deadline):
        names, token, tokens = set(), None, set()
        while True:
            _deadline(deadline)
            response = self._observation_io(
                s3.list_objects_v2,
                Bucket=catalog.bucket,
                Prefix=prefix,
                ExpectedBucketOwner=catalog.account_id,
                MaxKeys=1000,
                **({"ContinuationToken": token} if token else {}),
            )
            _deadline(deadline)
            entries = response.get("Contents", [])
            if type(entries) is not list or type(response.get("IsTruncated")) is not bool:
                raise ValueError("paired output inventory shape differs")
            for entry in entries:
                if (
                    type(entry) is not dict
                    or type(entry.get("Key")) is not str
                    or not entry["Key"].startswith(prefix)
                ):
                    raise ValueError("paired output key differs")
                name = entry["Key"][len(prefix) :]
                if name not in _CORE_NAMES:
                    _record_name(name)
                if name in names or len(names) >= _MAX_RECORDS + 4:
                    raise ValueError("paired output duplicate/count bound exceeded")
                names.add(name)
            if not response["IsTruncated"]:
                if response.get("NextContinuationToken"):
                    raise ValueError("paired terminal inventory continuation differs")
                break
            token = response.get("NextContinuationToken")
            if type(token) is not str or not token or token in tokens:
                raise ValueError("paired output continuation differs")
            tokens.add(token)
        if not _CORE_NAMES <= names:
            raise ValueError("paired three core outputs required")
        self._qualify_io()
        return names

    def _declared_output(self, catalog, prefix, name, descriptor, fmt, limit, deadline):
        if (
            type(descriptor) is not dict
            or set(descriptor) != {"sha256", "byte_count", "format"}
            or type(descriptor["sha256"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", descriptor["sha256"])
            or type(descriptor["byte_count"]) is not int
            or not 0 < descriptor["byte_count"] <= limit
            or descriptor["format"] != fmt
        ):
            raise ValueError("paired committed artifact descriptor differs")
        _deadline(deadline)
        head = self._observation_io(
            self._clients(catalog.region)[1].head_object,
            Bucket=catalog.bucket,
            Key=prefix + name,
            ExpectedBucketOwner=catalog.account_id,
        )
        _deadline(deadline)
        if type(head.get("ContentLength")) is not int or head["ContentLength"] != descriptor["byte_count"]:
            raise ValueError("paired committed output size differs")
        artifact = DatasetArtifactRef(
            "s3://" + catalog.bucket + "/" + prefix + name,
            _version(head.get("VersionId")),
            descriptor["sha256"],
            descriptor["byte_count"],
            fmt,
        )

        self._qualify_io()
        return artifact

    def artifacts(self, handle):
        status = self.status(handle)
        base = {
            "success": False,
            "accepted": handle.extra["submission_state"] == "accepted",
            "evaluated": False,
            "quality_success": False,
            "native_admission": False,
            "status": status.state.value,
            "job_name": handle.id,
            "job_arn": handle.arn,
        }
        if status.state != JobState.SUCCEEDED:
            return base
        try:
            catalog, session, _request_value = self._handle(handle)
            deadline = time.monotonic() + _CORE_SECONDS
            prefix = handle.extra["output_prefix"]
            expected_prefix = "sessions/" + session + "/paired/" + handle.id + "/output/"
            if prefix != expected_prefix:
                raise ValueError("paired original output namespace differs")
            s3 = self._clients(catalog.region)[1]
            names = self._inventory(s3, catalog, prefix, deadline)
            commit_ref, raw = self._output(
                handle, prefix + "COMMITTED.json", deadline, limit=262144, collect=True
            )
            commit = decode_object(raw, limit=262144)
            if (
                set(commit)
                != {
                    "schema_version",
                    "operation",
                    "source_commit",
                    "case_count",
                    "quality_success",
                    "native_admission",
                    "files",
                    "invocations_manifest",
                }
                or type(commit["schema_version"]) is not int
                or commit["schema_version"] != 1
                or commit["operation"] != "paired_eval"
                or commit["source_commit"] != catalog.source_commit
                or type(commit["case_count"]) is not int
                or commit["case_count"] != 164
                or commit["quality_success"] is not False
                or commit["native_admission"] is not False
                or type(commit["files"]) is not dict
                or set(commit["files"]) != {"RECEIPTS.jsonl", "EVALUATION.json"}
            ):
                raise ImmutableArtifactRejectionError("closed_manifest_invalid")
            refs, bundle = {"COMMITTED.json": commit_ref}, {"COMMITTED.json": raw}
            for name in ("RECEIPTS.jsonl", "EVALUATION.json"):
                descriptor = commit["files"][name]
                fmt = "jsonl" if name == "RECEIPTS.jsonl" else "json"
                limit = 2 * 1024**2 if name == "RECEIPTS.jsonl" else 262144
                ref = self._declared_output(catalog, prefix, name, descriptor, fmt, limit, deadline)
                bundle[name] = self._read(ref, catalog, session, deadline, limit=limit)
                refs[name] = ref
            manifest = commit["invocations_manifest"]
            if (
                type(manifest) is not dict
                or set(manifest) != {"sha256", "byte_count", "format", "record_count"}
                or type(manifest["record_count"]) is not int
                or not 0 <= manifest["record_count"] <= _MAX_RECORDS
            ):
                raise ValueError("paired invocation manifest descriptor differs")
            manifest_ref = self._declared_output(
                catalog,
                prefix,
                "INVOCATIONS_MANIFEST.jsonl",
                {key: value for key, value in manifest.items() if key != "record_count"},
                "jsonl",
                64 * 1024**2,
                deadline,
            )
            refs["INVOCATIONS_MANIFEST.jsonl"] = manifest_ref
            from ...evaluation.coding_pair import PairedTraceAccumulator

            trace = PairedTraceAccumulator(handle.extra["evaluation_binding"])
            seen, buffered = set(), bytearray()

            def consume(chunk, deadline):
                buffered.extend(chunk)
                while b"\n" in buffered:
                    line, _, remaining = buffered.partition(b"\n")
                    buffered[:] = remaining
                    if not line and manifest["record_count"] == 0 and manifest_ref.byte_count == 1:
                        continue
                    if not line or len(line) > 4096:
                        raise ValueError("paired invocation manifest row bound differs")
                    entry = decode_object(bytes(line), limit=4096)
                    if set(entry) != {"path", "sha256", "byte_count"} or type(entry["path"]) is not str:
                        raise ValueError("paired invocation manifest row differs")
                    name = entry["path"]
                    _record_name(name)
                    if name in seen or name not in names or len(seen) >= manifest["record_count"]:
                        raise ValueError("paired invocation manifest inventory differs")
                    ref = self._declared_output(
                        catalog,
                        prefix,
                        name,
                        {"sha256": entry["sha256"], "byte_count": entry["byte_count"], "format": "json"},
                        "json",
                        65536,
                        deadline,
                    )
                    record_raw = self._read(ref, catalog, session, deadline, limit=65536)
                    _validate_record(record_raw, name, handle.extra["inputs"].base.evaluation_id)
                    trace.add(name, decode_object(record_raw, limit=65536))
                    refs[name] = ref
                    seen.add(name)
                if len(buffered) > 4096:
                    raise ValueError("paired invocation manifest partial row bound differs")

            # The manifest is read whole, verified and its stream closed before the first record is read, so a
            # record pass of any length cannot time out an open stream.
            manifest_raw = self._read(manifest_ref, catalog, session, deadline, limit=64 * 1024**2)
            deadline = time.monotonic() + _CORE_SECONDS + _record_pass_seconds(manifest["record_count"])
            consume(manifest_raw, deadline)
            if buffered or len(seen) != manifest["record_count"] or seen != names - _CORE_NAMES:
                raise ValueError("paired complete invocation trace inventory required")
            _deadline(deadline)
            current = self._observation_io(
                s3.head_object,
                Bucket=catalog.bucket,
                Key=manifest_ref.uri.split("/", 3)[3],
                ExpectedBucketOwner=catalog.account_id,
            )
            if current.get("VersionId") != manifest_ref.version_id:
                raise ValueError("paired invocation manifest changed during the record pass")
            lines = bundle["RECEIPTS.jsonl"].splitlines(keepends=True)
            if len(lines) != 328 or any(not line.endswith(b"\n") or len(line) > 65536 for line in lines):
                raise ValueError("all328 paired result rows required")
            summary = verify_pair_result(
                bundle,
                binding=handle.extra["evaluation_binding"],
                source_commit=catalog.source_commit,
                invocations=(manifest_ref.sha256, manifest_ref.byte_count, len(seen)),
                trace=trace.finish(),
            )
            if (
                type(summary) is not dict
                or type(summary.get("case_count")) is not int
                or summary["case_count"] != 164
            ):
                raise ValueError("paired complete descriptive summary required")
            self._read(commit_ref, catalog, session, deadline, limit=262144)
            self._qualify_io()
            return {
                **base,
                **summary,
                "files": refs,
                "case_count": 164,
                "receipt_count": 328,
                "quality_success": False,
                "native_admission": False,
            }
        except Exception as error:
            return {
                **base,
                "status": "output_verification_failed",
                "error_type": type(error).__name__,
                **rejection_data(error),
            }
