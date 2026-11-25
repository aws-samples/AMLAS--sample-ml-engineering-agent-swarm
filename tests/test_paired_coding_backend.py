"""Closed native Processing/S3 fixtures; no candidate, oracle or AWS execution."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from botocore.session import Session
from botocore.validate import validate_parameters
from test_qwen_sft_backend import ACCOUNT, SESSION, catalog

from amlas.backends.adapters import coding_pair as module
from amlas.backends.types import DatasetArtifactRef, JobState
from amlas.control_plane.coding_case_state import VerifiedCaseEvaluationSource
from amlas.control_plane.generation_state import VerifiedCodingGenerationSource
from amlas.control_plane.paired_state import (
    PairConfigurationMissingError,
    PairedCodingCatalog,
    PairedCodingInputs,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Objects:
    """S3 exact-version and conditional writes, checked against the native service model."""

    def __init__(self):
        self.objects = {}
        self.versions = {}
        self.calls = []
        self.pending = {}
        self.model = Session().get_service_model("s3")
        self.counter = 0

    def check(self, operation, kwargs):
        validate_parameters(kwargs, self.model.operation_model(operation).input_shape)
        self.calls.append((operation, copy.deepcopy(kwargs)))
        assert kwargs.get("ExpectedBucketOwner", ACCOUNT) == ACCOUNT

    def install(self, key, raw, version=None):
        self.counter += 1
        version = version or "version-" + str(self.counter)
        self.objects[(key, version)] = raw
        self.versions[key] = version
        return DatasetArtifactRef(
            "s3://" + catalog().bucket + "/" + key,
            version,
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            "jsonl" if key.endswith(".jsonl") else "json",
        )

    def get_bucket_versioning(self, **kwargs):
        self.check("GetBucketVersioning", kwargs)
        return {"Status": "Enabled"}

    def head_object(self, **kwargs):
        self.check("HeadObject", kwargs)
        key = kwargs["Key"]
        version = kwargs.get("VersionId", self.versions[key])
        raw = self.objects[(key, version)]
        return {"VersionId": version, "ContentLength": len(raw)}

    def get_object(self, **kwargs):
        self.check("GetObject", kwargs)
        assert kwargs.get("VersionId")
        raw = self.objects[(kwargs["Key"], kwargs["VersionId"])]
        return {"VersionId": kwargs["VersionId"], "ContentLength": len(raw), "Body": io.BytesIO(raw)}

    def put_object(self, **kwargs):
        self.check("PutObject", kwargs)
        assert kwargs["IfNoneMatch"] == "*" and kwargs["Key"] not in self.versions
        raw = kwargs["Body"]
        assert type(raw) is bytes
        return {"VersionId": self.install(kwargs["Key"], raw).version_id}

    def create_multipart_upload(self, **kwargs):
        self.check("CreateMultipartUpload", kwargs)
        uid = "upload-" + str(len(self.pending))
        self.pending[uid] = (kwargs["Key"], {})
        return {"UploadId": uid}

    def upload_part_copy(self, **kwargs):
        self.check("UploadPartCopy", kwargs)
        from urllib.parse import unquote

        source, version = kwargs["CopySource"].split("?versionId=")
        key = unquote(source.split("/", 1)[1])
        version = unquote(version)
        raw = self.objects[(key, version)]
        if "CopySourceRange" in kwargs:
            assert len(raw) > 5 * 1024**2
            first, last = map(int, kwargs["CopySourceRange"][6:].split("-"))
            raw = raw[first : last + 1]
        self.pending[kwargs["UploadId"]][1][kwargs["PartNumber"]] = raw
        return {"CopySourceVersionId": version, "CopyPartResult": {"ETag": "part"}}

    def complete_multipart_upload(self, **kwargs):
        self.check("CompleteMultipartUpload", kwargs)
        assert kwargs["IfNoneMatch"] == "*" and kwargs["Key"] not in self.versions
        key, parts = self.pending.pop(kwargs["UploadId"])
        return {"VersionId": self.install(key, b"".join(parts[index] for index in sorted(parts))).version_id}

    def abort_multipart_upload(self, **kwargs):
        self.check("AbortMultipartUpload", kwargs)
        self.pending.pop(kwargs["UploadId"], None)
        return {}

    def list_objects_v2(self, **kwargs):
        self.check("ListObjectsV2", kwargs)
        names = sorted(key for key in self.versions if key.startswith(kwargs["Prefix"]))
        return {
            "IsTruncated": len(names) > kwargs["MaxKeys"],
            "Contents": [
                {"Key": key, "Size": len(self.objects[(key, self.versions[key])])}
                for key in names[: kwargs["MaxKeys"]]
            ],
        }


class Processing:
    """Fixed SDK Processing operations only; no training or evaluation runs."""

    def __init__(self):
        self.model = Session().get_service_model("sagemaker")
        self.calls = []
        self.request = None
        self.error = None
        self.drift = None
        self.state = "InProgress"
        self.tags_extra = {}

    def create_processing_job(self, **kwargs):
        validate_parameters(kwargs, self.model.operation_model("CreateProcessingJob").input_shape)
        assert "ExperimentConfig" not in kwargs
        assert kwargs["Environment"] == {"AMLAS_CODING_PAIR_LAUNCHER_SHA256": "d" * 64}
        self.calls.append("create")
        self.request = copy.deepcopy(kwargs)
        if self.error is not None:
            raise self.error
        return {"ProcessingJobArn": self.arn}

    @property
    def arn(self):
        return (
            "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/" + self.request["ProcessingJobName"]
        )

    def describe_processing_job(self, **kwargs):
        validate_parameters(kwargs, self.model.operation_model("DescribeProcessingJob").input_shape)
        self.calls.append("describe")
        assert kwargs == {"ProcessingJobName": self.request["ProcessingJobName"]}
        value = {
            **copy.deepcopy(self.request),
            "ProcessingJobArn": self.arn,
            "ProcessingJobStatus": self.state,
        }
        value.pop("Tags")
        if self.drift:
            self.drift(value)
        return value

    def list_tags(self, **kwargs):
        validate_parameters(kwargs, self.model.operation_model("ListTags").input_shape)
        self.calls.append("tags")
        return {"Tags": copy.deepcopy(self.request["Tags"]), **self.tags_extra}


@pytest.fixture
def packet(monkeypatch):
    s3, sm = Objects(), Processing()
    groups = {}
    for group, names in (
        (
            "cases",
            ("PROMPTS.jsonl", "ORACLES.jsonl", "EXCLUSIONS.json", "CASE_MANIFEST.json", "COMMITTED.json"),
        ),
        ("base", ("CANDIDATES.jsonl", "GENERATION.json", "COMMITTED.json")),
        ("tuned", ("CANDIDATES.jsonl", "GENERATION.json", "COMMITTED.json")),
    ):
        groups[group] = tuple(
            (name, s3.install("sessions/" + SESSION + "/original/" + group + "/" + name, b"{}\n"))
            for name in names
        )
    c = catalog()
    cases = VerifiedCaseEvaluationSource(
        SESSION,
        "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/owned-cases",
        tuple(
            sorted(
                {
                    "runtime_session_id": SESSION,
                    "deployment_id": c.deployment_id,
                    "deployment_prefix": c.deployment_prefix,
                }.items()
            )
        ),
        None,
        groups["cases"],
    )
    base = VerifiedCodingGenerationSource(
        SESSION,
        "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":training-job/owned-base",
        "paired-id",
        "base",
        groups["base"],
    )
    tuned = dataclasses.replace(
        base,
        producer_job_arn=base.producer_job_arn.replace("base", "tuned"),
        arm="tuned",
        file_refs=groups["tuned"],
    )
    spec = b"{}"
    inputs = PairedCodingInputs(cases, base, tuned, spec, hashlib.sha256(spec).hexdigest())
    validated = []

    def validate(inputs, payloads, **pins):
        assert set(payloads) == {group + "/" + name for group, refs in groups.items() for name, _ref in refs}
        assert all(raw == b"{}\n" for raw in payloads.values())
        assert pins == {
            "runner_sha256": "d" * 64,
            "runtime_sha256": c.admission_digest,
            "budgets_sha256": hashlib.sha256(
                canonical(
                    {
                        "case_wall_seconds": 10,
                        "session_timeout_seconds": 60,
                        "max_value_calls": 512,
                        "allocation_excluded": True,
                        "cleanup_seconds": 30,
                    }
                )
            ).hexdigest(),
        }
        validated.append(inputs)
        from test_coding_arbiter import binding

        bound = binding().model_copy(update={"evaluation_id": "paired-id", "spec_sha256": inputs.spec_sha256})
        return bound, (), {}

    # Only the pure metadata ownership seam is scoped here; native transport uses real API shapes.
    monkeypatch.setattr(module, "validate_pair_artifacts", validate, raising=False)
    monkeypatch.setattr(
        module, "paired_runner_sha256", lambda *, oracle_launcher_sha256: "d" * 64, raising=False
    )
    backend = module.SageMakerPairedCodingBackend()
    backend._sm, backend._s3 = sm, s3
    return backend, inputs, c, sm, s3, validated


def submit(packet):
    backend, inputs, c, *_ = packet
    return backend.submit(inputs=inputs, catalog=PairedCodingCatalog(c, "d" * 64), runtime_session_id=SESSION)


def test_fixed_sdk_processing_request_and_exact_source_stage(packet):
    handle = submit(packet)
    backend, inputs, c, sm, s3, validated = packet
    assert validated == [inputs] and handle.extra["submission_state"] == "accepted"
    request = handle.extra["request"]
    assert request["RoleArn"] == c.execution_role_arn
    assert request["AppSpecification"]["ImageUri"] == c.image_uri
    assert request["AppSpecification"]["ContainerEntrypoint"] == [
        "/usr/local/bin/amlas-processing-init",
        "coding-pair",
    ]
    assert request["AppSpecification"]["ContainerArguments"] == [
        "--binding-sha256",
        handle.extra["run_ref"].sha256,
        "--binding-byte-count",
        str(handle.extra["run_ref"].byte_count),
        "--source-commit",
        c.source_commit,
    ]
    assert request["NetworkConfig"] == {
        "EnableNetworkIsolation": False,
        "EnableInterContainerTrafficEncryption": False,
    }
    assert request["ProcessingResources"]["ClusterConfig"] == {
        "InstanceCount": 1,
        "InstanceType": "ml.m5.xlarge",
        "VolumeSizeInGB": 30,
    }
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 14400}
    assert len(request["ProcessingInputs"]) == 4
    run = handle.extra["run_binding"]
    assert run["operation"] == "paired_eval" and run["spec_sha256"] == inputs.spec_sha256
    assert set(run["original_refs"]) == set(run["staged_refs"]) == {"cases", "base", "tuned"}
    assert "approval" not in canonical(run).decode()
    assert len([call for call in sm.calls if call == "create"]) == 1
    gets = [args for name, args in s3.calls if name == "GetObject"]
    assert all(args.get("VersionId") for args in gets)
    assert not any("safetensors" in args["Key"] for args in gets)
    before = (len(sm.calls), len(s3.calls))
    assert backend.status(handle).state == JobState.RUNNING
    assert sm.calls[before[0] :] == ["describe", "tags"] and len(s3.calls) == before[1]


@pytest.mark.parametrize("nested", [None, "plain", [], {"Code": []}])
def test_malformed_create_error_retains_original_ambiguous_identity(packet, nested):
    error = RuntimeError("neutral unavailable acknowledgement")
    error.response = {"Error": nested}
    packet[3].error = error
    handle = submit(packet)
    assert handle.arn and handle.extra["submission_state"] == "ambiguous"
    assert packet[0].artifacts(handle)["evaluated"] is False
    assert packet[3].calls.count("create") == 1


@pytest.mark.parametrize("drift", ["session", "arm", "spec", "version", "inventory", "oversize"])
def test_private_input_drift_refuses_before_staging_or_create(packet, drift):
    backend, inputs, c, sm, s3, _validated = packet
    if drift == "session":
        inputs = dataclasses.replace(
            inputs, base=dataclasses.replace(inputs.base, runtime_session_id="foreign")
        )
    elif drift == "arm":
        inputs = dataclasses.replace(inputs, tuned=dataclasses.replace(inputs.tuned, arm="base"))
    elif drift == "spec":
        inputs = dataclasses.replace(inputs, spec_sha256="f" * 64)
    elif drift == "version":
        first = dict(inputs.base.file_refs)["GENERATION.json"]
        s3.install(first.uri.split("/", 3)[3], b"changed\n")
    elif drift == "inventory":
        inputs = dataclasses.replace(
            inputs, base=dataclasses.replace(inputs.base, file_refs=inputs.base.file_refs[:-1])
        )
    else:
        refs = tuple(
            (
                name,
                dataclasses.replace(ref, byte_count=12 * 1024**2 + 1) if name == "CANDIDATES.jsonl" else ref,
            )
            for name, ref in inputs.base.file_refs
        )
        inputs = dataclasses.replace(inputs, base=dataclasses.replace(inputs.base, file_refs=refs))
    with pytest.raises(ValueError):
        backend.submit(inputs=inputs, catalog=PairedCodingCatalog(c, "d" * 64), runtime_session_id=SESSION)
    assert sm.calls == [] and not any(
        name in {"PutObject", "CreateMultipartUpload"} for name, _args in s3.calls
    )


def test_missing_pair_catalog_refuses_without_clients(packet):
    backend, inputs, c, sm, s3, _validated = packet
    with pytest.raises(PairConfigurationMissingError):
        backend.submit(
            inputs=inputs,
            catalog=PairedCodingCatalog(dataclasses.replace(c, image_uri=""), "d" * 64),
            runtime_session_id=SESSION,
        )
    assert not sm.calls and not s3.calls


@pytest.mark.parametrize("drift", ["role", "isolation", "input", "appmanaged", "environment", "tags_page"])
def test_native_request_drift_is_unknown_no_writes(packet, drift):
    handle = submit(packet)
    backend, _inputs, _c, sm, s3, _validated = packet

    def change(value):
        if drift == "role":
            value["RoleArn"] += "-other"
        elif drift == "isolation":
            value["NetworkConfig"]["EnableNetworkIsolation"] = True
        elif drift == "input":
            value["ProcessingInputs"][0]["S3Input"]["S3Uri"] += "other/"
        elif drift == "appmanaged":
            value["ProcessingInputs"][0]["AppManaged"] = None
        elif drift == "environment":
            value["Environment"] = {"MODEL_SELECTOR": "other"}

    sm.drift = change
    if drift == "tags_page":
        sm.tags_extra = {"NextToken": "unfinished"}
    before = len(s3.calls)
    assert backend.status(handle).state == JobState.UNKNOWN
    assert len(s3.calls) == before and sm.calls.count("create") == 1


def test_documented_false_appmanaged_is_normalized_without_mutation(packet):
    handle = submit(packet)
    backend, *_ = packet

    def add_defaults(value):
        for entry in value["ProcessingInputs"] + value["ProcessingOutputConfig"]["Outputs"]:
            entry["AppManaged"] = False
        value["Environment"] = {"AMLAS_CODING_PAIR_LAUNCHER_SHA256": "d" * 64}

    packet[3].drift = add_defaults
    original = copy.deepcopy(handle.extra["request"])
    assert backend.status(handle).state == JobState.RUNNING
    assert handle.extra["request"] == original


def outputs(packet, handle):
    _backend, _inputs, c, sm, s3, _validated = packet
    sm.state = "Completed"
    prefix = handle.extra["output_prefix"]
    payloads = {
        "RECEIPTS.jsonl": b"{}\n" * 328,
        "EVALUATION.json": b"{}",
        "INVOCATIONS/base/000/calls/00000001-attempt.json": canonical(
            {
                "evaluation_id": "paired-id",
                "case_id": "HumanEval/0",
                "arm": "base",
                "ordinal": 0,
                "binding": {
                    "evaluation_id": "paired-id",
                    "case_id": "HumanEval/0",
                    "arm": "base",
                    "ordinal": 0,
                },
                "session_id": "original-session",
                "interpreter_id": "aws.codeinterpreter.v1",
                "kind": "attempt",
                "time": 1.0,
                "deadline": 11.0,
                "request_sha256": "a" * 64,
                "code_sha256": "b" * 64,
            }
        ),
    }
    manifest = {
        name: {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "format": "jsonl" if name.endswith(".jsonl") else "json",
        }
        for name, raw in payloads.items()
        if not name.startswith("INVOCATIONS/")
    }
    traces = b"".join(
        canonical({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}) + b"\n"
        for name, raw in payloads.items()
        if name.startswith("INVOCATIONS/")
    )
    payloads["INVOCATIONS_MANIFEST.jsonl"] = traces
    trace_descriptor = {
        "sha256": hashlib.sha256(traces).hexdigest(),
        "byte_count": len(traces),
        "format": "jsonl",
        "record_count": 1,
    }
    payloads["COMMITTED.json"] = canonical(
        {
            "schema_version": 1,
            "operation": "paired_eval",
            "source_commit": c.source_commit,
            "case_count": 164,
            "quality_success": False,
            "native_admission": False,
            "files": manifest,
            "invocations_manifest": trace_descriptor,
        }
    )
    for name, raw in payloads.items():
        s3.install(prefix + name, raw)
    return payloads


def test_verified_complete_outputs_are_descriptive_and_polling_never_writes(packet, monkeypatch):
    handle = submit(packet)
    expected = outputs(packet, handle)
    calls = []

    def verify(bundle, **kwargs):
        assert bundle == {
            name: raw
            for name, raw in expected.items()
            if name in {"RECEIPTS.jsonl", "EVALUATION.json", "COMMITTED.json"}
        }
        assert kwargs.pop("trace").record_count == 1
        assert kwargs == {
            "binding": handle.extra["evaluation_binding"],
            "source_commit": packet[2].source_commit,
            "invocations": (
                hashlib.sha256(expected["INVOCATIONS_MANIFEST.jsonl"]).hexdigest(),
                len(expected["INVOCATIONS_MANIFEST.jsonl"]),
                1,
            ),
        }
        calls.append(True)
        return {
            "case_count": 164,
            "success": True,
            "evaluated": True,
            "status": "evaluated",
            "quality_success": False,
            "native_admission": False,
        }

    monkeypatch.setattr(module, "verify_pair_result", verify, raising=False)
    before = len(packet[4].calls)
    result = packet[0].artifacts(handle)
    assert result["success"] is result["evaluated"] is True
    assert result["case_count"] == 164 and result["receipt_count"] == 328 and calls == [True]
    assert result["quality_success"] is result["native_admission"] is False
    assert set(result["files"]) == set(expected)
    assert all(type(ref) is DatasetArtifactRef and ref.version_id for ref in result["files"].values())
    assert not any(
        name in {"PutObject", "CreateMultipartUpload", "CompleteMultipartUpload"}
        for name, _args in packet[4].calls[before:]
    )
    assert packet[3].calls.count("create") == 1


@pytest.mark.parametrize("drift", ["extra", "hash", "commit_type", "rows"])
def test_completed_outputs_still_refuse_incomplete_or_uncommitted_bytes(packet, monkeypatch, drift):
    handle = submit(packet)
    payloads = outputs(packet, handle)
    s3 = packet[4]
    prefix = handle.extra["output_prefix"]
    if drift == "extra":
        s3.install(prefix + "other.py", b"neutral")
    elif drift == "hash":
        s3.install(prefix + "EVALUATION.json", b"changed")
    elif drift == "commit_type":
        commit = json.loads(payloads["COMMITTED.json"])
        commit["case_count"] = 164.0
        s3.install(prefix + "COMMITTED.json", canonical(commit))
    else:
        payloads["RECEIPTS.jsonl"] = b"{}\n" * 327
        commit = json.loads(payloads["COMMITTED.json"])
        commit["files"]["RECEIPTS.jsonl"] = {
            "sha256": hashlib.sha256(payloads["RECEIPTS.jsonl"]).hexdigest(),
            "byte_count": len(payloads["RECEIPTS.jsonl"]),
            "format": "jsonl",
        }
        s3.install(prefix + "RECEIPTS.jsonl", payloads["RECEIPTS.jsonl"])
        s3.install(prefix + "COMMITTED.json", canonical(commit))
    monkeypatch.setattr(
        module, "verify_pair_result", lambda *args, **kwargs: {"case_count": 328}, raising=False
    )
    result = packet[0].artifacts(handle)
    assert result["evaluated"] is False and result["status"] == "output_verification_failed"


@pytest.mark.parametrize(
    "drift",
    ["trace_hash", "manifest_duplicate", "manifest_missing", "path", "count_type", "partial", "trace_size"],
)
def test_streamed_invocation_manifest_refuses_unbound_traces(packet, monkeypatch, drift):
    handle = submit(packet)
    payloads = outputs(packet, handle)
    s3 = packet[4]
    prefix = handle.extra["output_prefix"]
    commit = json.loads(payloads["COMMITTED.json"])
    manifest_raw = payloads["INVOCATIONS_MANIFEST.jsonl"]
    entry = json.loads(manifest_raw)
    if drift == "trace_hash":
        s3.install(prefix + entry["path"], b"changed")
    elif drift == "manifest_duplicate":
        manifest_raw += manifest_raw
        commit["invocations_manifest"]["record_count"] = 2
    elif drift == "manifest_missing":
        s3.install(prefix + "INVOCATIONS/tuned/001/calls/00000002-unknown.json", b"{}")
    elif drift == "path":
        entry["path"] = "INVOCATIONS/base/164/calls/00000001-attempt.json"
        manifest_raw = canonical(entry) + b"\n"
    elif drift == "count_type":
        commit["invocations_manifest"]["record_count"] = True
    elif drift == "partial":
        manifest_raw = manifest_raw[:-1]
    else:
        entry["byte_count"] = 65537
        manifest_raw = canonical(entry) + b"\n"
    commit["invocations_manifest"].update(
        sha256=hashlib.sha256(manifest_raw).hexdigest(), byte_count=len(manifest_raw)
    )
    s3.install(prefix + "INVOCATIONS_MANIFEST.jsonl", manifest_raw)
    s3.install(prefix + "COMMITTED.json", canonical(commit))
    calls = []
    monkeypatch.setattr(
        module,
        "verify_pair_result",
        lambda *args, **kwargs: (
            calls.append(True)
            or {"case_count": 164, "success": True, "evaluated": True, "status": "evaluated"}
        ),
        raising=False,
    )
    result = packet[0].artifacts(handle)
    assert result["evaluated"] is False and result["status"] == "output_verification_failed"
    assert calls == [] and packet[3].calls.count("create") == 1


def test_zero_invocations_preserve_complete_failed_generation_population(packet, monkeypatch):
    handle = submit(packet)
    payloads = outputs(packet, handle)
    s3 = packet[4]
    prefix = handle.extra["output_prefix"]
    for key in list(s3.versions):
        if key.startswith(prefix + "INVOCATIONS/"):
            del s3.versions[key]
    commit = json.loads(payloads["COMMITTED.json"])
    raw = b"\n"
    commit["invocations_manifest"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": 1,
        "format": "jsonl",
        "record_count": 0,
    }
    s3.install(prefix + "INVOCATIONS_MANIFEST.jsonl", raw)
    s3.install(prefix + "COMMITTED.json", canonical(commit))

    def verify(bundle, **kwargs):
        assert kwargs["invocations"] == (hashlib.sha256(raw).hexdigest(), 1, 0)
        return {"case_count": 164, "success": True, "evaluated": True, "status": "evaluated"}

    monkeypatch.setattr(module, "verify_pair_result", verify, raising=False)
    result = packet[0].artifacts(handle)
    assert result["evaluated"] is True and result["receipt_count"] == 328
    assert result["quality_success"] is result["native_admission"] is False


def test_paginated_inventory_checks_distinct_continuation_and_all_exact_names(packet, monkeypatch):
    handle = submit(packet)
    outputs(packet, handle)
    s3 = packet[4]
    original = s3.list_objects_v2
    calls = []

    def paged(**kwargs):
        current = original(**kwargs)
        calls.append(kwargs.get("ContinuationToken"))
        current["Contents"] = (
            current["Contents"][2:] if kwargs.get("ContinuationToken") else current["Contents"][:2]
        )
        current["IsTruncated"] = not kwargs.get("ContinuationToken")
        if current["IsTruncated"]:
            current["NextContinuationToken"] = "page-two"
        return current

    monkeypatch.setattr(s3, "list_objects_v2", paged)
    monkeypatch.setattr(
        module,
        "verify_pair_result",
        lambda *args, **kwargs: {
            "case_count": 164,
            "success": True,
            "evaluated": True,
            "status": "evaluated",
        },
        raising=False,
    )
    assert packet[0].artifacts(handle)["evaluated"] is True
    assert calls == [None, "page-two"]


def test_accepted_native_identity_mismatch_is_ambiguous_without_retry(packet, monkeypatch):
    sm = packet[3]
    original = sm.create_processing_job

    def lost(**kwargs):
        original(**kwargs)
        return {"ProcessingJobArn": "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/other"}

    monkeypatch.setattr(sm, "create_processing_job", lost)
    handle = submit(packet)
    assert handle.extra["submission_state"] == "ambiguous"
    assert handle.arn == sm.arn and packet[0].status(handle).state == JobState.RUNNING
    assert sm.calls.count("create") == 1


@pytest.mark.parametrize("code", ["ValidationException", "AccessDeniedException", "ResourceLimitExceeded"])
def test_only_closed_sdk_rejection_codes_change_original_disposition(packet, code):
    error = RuntimeError("neutral rejection")
    error.response = {"Error": {"Code": code}}
    packet[3].error = error
    handle = submit(packet)
    assert handle.extra["submission_state"] == "rejected" and handle.arn
    assert packet[3].calls.count("create") == 1


@pytest.mark.parametrize("drift", ["run", "evaluation", "spec_ref", "staged_ref"])
def test_original_handle_packet_drift_refuses_before_native_observation(packet, drift):
    handle = submit(packet)
    if drift == "run":
        handle.extra["run_binding"]["spec_sha256"] = "f" * 64
    elif drift == "evaluation":
        handle.extra["evaluation_binding"] = SimpleNamespace(
            model_dump=lambda **_: {"evaluation_id": "different", "spec_sha256": "f" * 64}
        )
    elif drift == "spec_ref":
        handle.extra["spec_ref"] = dataclasses.replace(handle.extra["spec_ref"], version_id="different")
    else:
        handle.extra["staged_refs"]["base"]["CANDIDATES.jsonl"] = dataclasses.replace(
            handle.extra["staged_refs"]["base"]["CANDIDATES.jsonl"], version_id="different"
        )
    before = len(packet[3].calls)
    assert packet[0].status(handle).state == JobState.UNKNOWN
    assert packet[3].calls[before:] == []


@pytest.mark.parametrize("drift", ["json_type", "fields", "scope"])
def test_rehashed_trace_record_still_requires_source_owned_metadata(packet, monkeypatch, drift):
    handle = submit(packet)
    payloads = outputs(packet, handle)
    path = "INVOCATIONS/base/000/calls/00000001-attempt.json"
    if drift == "json_type":
        raw = b"[]"
    elif drift == "fields":
        raw = canonical({"kind": "attempt", "source": "neutral"})
    else:
        record = json.loads(payloads[path])
        record["case_id"] = record["binding"]["case_id"] = "HumanEval/1"
        raw = canonical(record)
    entry = (
        canonical({"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}) + b"\n"
    )
    commit = json.loads(payloads["COMMITTED.json"])
    commit["invocations_manifest"].update(sha256=hashlib.sha256(entry).hexdigest(), byte_count=len(entry))
    s3, prefix = packet[4], handle.extra["output_prefix"]
    for name, body in (
        (path, raw),
        ("INVOCATIONS_MANIFEST.jsonl", entry),
        ("COMMITTED.json", canonical(commit)),
    ):
        s3.install(prefix + name, body)
    monkeypatch.setattr(
        module,
        "verify_pair_result",
        lambda *args, **kwargs: {
            "case_count": 164,
            "success": True,
            "evaluated": True,
            "status": "evaluated",
        },
        raising=False,
    )
    assert packet[0].artifacts(handle)["evaluated"] is False


def session_outputs(packet, handle, *, evaluation_id="paired-id"):
    payloads = outputs(packet, handle)
    prefix, s3 = handle.extra["output_prefix"], packet[4]
    for key in list(s3.versions):
        if key.startswith(prefix + "INVOCATIONS/"):
            del s3.versions[key]
    name = "INVOCATIONS/base/000/session/00000000-start-accepted.json"
    raw = (
        canonical(
            {
                "evaluation_id": evaluation_id,
                "case_id": "HumanEval/0",
                "arm": "base",
                "session_id": "original-session",
            }
        )
        + b"\n"
    )
    manifest = (
        canonical({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}) + b"\n"
    )
    commit = json.loads(payloads["COMMITTED.json"])
    commit["invocations_manifest"].update(
        sha256=hashlib.sha256(manifest).hexdigest(), byte_count=len(manifest)
    )
    for output_name, body in (
        (name, raw),
        ("INVOCATIONS_MANIFEST.jsonl", manifest),
        ("COMMITTED.json", canonical(commit)),
    ):
        s3.install(prefix + output_name, body)


def test_session_trace_requires_original_evaluation_identity(packet, monkeypatch):
    handle = submit(packet)
    session_outputs(packet, handle, evaluation_id="different")
    monkeypatch.setattr(
        module,
        "verify_pair_result",
        lambda *args, **kwargs: {
            "case_count": 164,
            "success": True,
            "evaluated": True,
            "status": "evaluated",
        },
        raising=False,
    )
    assert packet[0].artifacts(handle)["evaluated"] is False


def test_original_session_record_uses_shared_validator(packet, monkeypatch):
    handle = submit(packet)
    session_outputs(packet, handle)
    monkeypatch.setattr(
        module,
        "verify_pair_result",
        lambda *args, **kwargs: {
            "case_count": 164,
            "success": True,
            "evaluated": True,
            "status": "evaluated",
        },
        raising=False,
    )
    assert packet[0].artifacts(handle)["evaluated"] is True


def actual_output_binding(packet, monkeypatch):
    from test_coding_arbiter import binding

    original = module.validate_pair_artifacts
    bound = binding().model_copy(update={"evaluation_id": packet[1].base.evaluation_id})

    def validate(*args, **kwargs):
        _declared, cases, candidates = original(*args, **kwargs)
        return bound, cases, candidates

    monkeypatch.setattr(module, "validate_pair_artifacts", validate)
    return bound


def install_actual_verifier_outputs(packet, handle, bound, *, cleanup_complete, passed=False):
    from amlas.arbiter.coding import CodingCasePin, CodingCaseReceipt, CodingEvaluatorFactory
    from amlas.evaluation.coding_oracle_catalog import EXPECTED_CASES

    pins = tuple(
        CodingCasePin(case_id=row["case_id"], oracle_sha256=row["oracle_record_sha256"])
        for row in EXPECTED_CASES
    )
    receipts = tuple(
        CodingCaseReceipt(
            case_id=pin.case_id,
            arm=arm,
            binding=bound,
            model_sha256=bound.base_model_sha256 if arm == "base" else bound.tuned_model_sha256,
            candidate_sha256="a" * 64,
            oracle_sha256=pin.oracle_sha256,
            outcome="passed" if passed else "incomplete",
            oracle_completed=passed,
            exit_code=0 if passed else None,
            execution_id="fixture-" + arm + "-" + str(index),
            stdout_sha256=hashlib.sha256(b"").hexdigest(),
        )
        for arm in ("base", "tuned")
        for index, pin in enumerate(pins)
    )
    summary = (
        CodingEvaluatorFactory().build_report(binding=bound, case_pins=pins, receipts=receipts).describe()
    )
    trace_raw = b"\n"
    trace = {
        "sha256": hashlib.sha256(trace_raw).hexdigest(),
        "byte_count": 1,
        "format": "jsonl",
        "record_count": 0,
    }
    source_commit = packet[2].source_commit
    evaluation = {
        "schema_version": 1,
        "operation": "paired_eval",
        "source_commit": source_commit,
        "binding": bound.model_dump(mode="json"),
        "case_count": 164,
        "receipt_count": 328,
        "quality_success": False,
        "native_admission": False,
        "cleanup_complete": cleanup_complete,
        "summary": summary,
        "invocations_manifest": trace,
    }
    payloads = {
        "RECEIPTS.jsonl": b"".join(
            canonical(receipt.model_dump(mode="json")) + b"\n" for receipt in receipts
        ),
        "EVALUATION.json": canonical(evaluation),
    }
    files = {
        name: {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "format": "jsonl" if name.endswith(".jsonl") else "json",
        }
        for name, raw in payloads.items()
    }
    payloads["COMMITTED.json"] = canonical(
        {
            "schema_version": 1,
            "operation": "paired_eval",
            "source_commit": source_commit,
            "case_count": 164,
            "quality_success": False,
            "native_admission": False,
            "files": files,
            "invocations_manifest": trace,
        }
    )
    payloads["INVOCATIONS_MANIFEST.jsonl"] = trace_raw
    s3, prefix = packet[4], handle.extra["output_prefix"]
    for name, raw in payloads.items():
        s3.install(prefix + name, raw)
    packet[3].state = "Completed"
    return payloads


def test_actual_verifier_cleanup_unresolved_is_not_overwritten_by_adapter(packet, monkeypatch):
    bound = actual_output_binding(packet, monkeypatch)
    handle = submit(packet)
    payloads = install_actual_verifier_outputs(packet, handle, bound, cleanup_complete=False)
    from amlas.evaluation.coding_pair import PairedTraceAccumulator, verify_pair_result

    original = verify_pair_result(
        {name: raw for name, raw in payloads.items() if name != "INVOCATIONS_MANIFEST.jsonl"},
        binding=bound,
        source_commit=packet[2].source_commit,
        invocations=(hashlib.sha256(b"\n").hexdigest(), 1, 0),
        trace=PairedTraceAccumulator(bound).finish(),
    )
    assert (
        original["status"] == "cleanup_unresolved" and original["success"] is original["evaluated"] is False
    )
    result = packet[0].artifacts(handle)
    assert result["success"] is False
    assert result["evaluated"] is False and result["status"] == "cleanup_unresolved"
    assert result["cleanup_complete"] is False and result["receipt_count"] == 328
    assert result["quality_success"] is result["native_admission"] is False
    assert set(result["files"]) == set(payloads)


@pytest.mark.parametrize("status", ["cleanup_unresolved", "incomplete"])
def test_descriptive_incomplete_verifier_state_is_preserved(packet, monkeypatch, status):
    handle = submit(packet)
    outputs(packet, handle)
    result_from_verifier = {
        "case_count": 164,
        "receipt_count": 328,
        "success": False,
        "evaluated": False,
        "status": status,
        "cleanup_complete": False,
    }
    monkeypatch.setattr(module, "verify_pair_result", lambda *args, **kwargs: result_from_verifier)
    result = packet[0].artifacts(handle)
    assert result["success"] is result["evaluated"] is False and result["status"] == status
    assert result["cleanup_complete"] is False and result["receipt_count"] == 328
    assert result["quality_success"] is result["native_admission"] is False


def test_actual_passed_receipts_without_any_native_trace_are_refused(packet, monkeypatch):
    bound = actual_output_binding(packet, monkeypatch)
    handle = submit(packet)
    install_actual_verifier_outputs(packet, handle, bound, cleanup_complete=True, passed=True)
    result = packet[0].artifacts(handle)
    assert result["evaluated"] is False and result["success"] is False
    assert result["status"] == "output_verification_failed"


def test_actual_verifier_statement_reaches_the_adapter_result_for_an_evaluated_pair(packet, monkeypatch):
    bound = actual_output_binding(packet, monkeypatch)
    handle = submit(packet)
    install_actual_verifier_outputs(packet, handle, bound, cleanup_complete=True)
    result = packet[0].artifacts(handle)
    assert result["status"] == "evaluated" and result["evaluated"] is result["success"] is True
    # Fixture population: no case of either arm has an answer, so nothing passed and nothing is without a verdict.
    assert result["result_statement"] == (
        "Base passed 0 of 164 cases. Tuned passed 0 of 164. "
        "No case ended without a verdict (timeout or infrastructure error). "
        "The difference in passed cases, tuned minus base, is 0. "
        "Each case that ran had 10 seconds for its test and all of its remote calls, "
        "counted from the moment its remote session was accepted. "
        "Neither arm ran a case a second time."
    )
    assert result["next_step"] == "Put result_statement into your final report unchanged."
    assert result["summary"]["base_pass_count_range"] == result["summary"]["tuned_pass_count_range"] == [0, 0]
    assert result["summary"]["delta_pass_count_range"] == [0, 0]
    assert result["quality_success"] is result["native_admission"] is False


def test_actual_verifier_unresolved_cleanup_reaches_the_adapter_result_without_a_statement(
    packet, monkeypatch
):
    bound = actual_output_binding(packet, monkeypatch)
    handle = submit(packet)
    install_actual_verifier_outputs(packet, handle, bound, cleanup_complete=False)
    result = packet[0].artifacts(handle)
    assert result["status"] == "cleanup_unresolved" and result["evaluated"] is result["success"] is False
    assert result["summary"]["delta_pass_count_range"] == [0, 0]
    assert "result_statement" not in result and "next_step" not in result
