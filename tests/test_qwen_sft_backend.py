"""Fixed native SFT request, versioned staging and honest lifecycle controls."""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from botocore.session import Session
from botocore.validate import validate_parameters

from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.backends.types import DatasetArtifactRef, JobState
from amlas.control_plane.sft_state import (
    SftCatalog,
    VerifiedSftInputs,
    VerifiedSftModelSource,
    VerifiedSftPreparedSource,
)
from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest

MODELS = {name: Session().get_service_model(name) for name in ("s3", "sagemaker")}

ACCOUNT = "111122223333"
SESSION = "isolated-sft"
BUCKET = "owned-test-bucket"
SOURCE = "a" * 40
IMAGE = ACCOUNT + ".dkr.ecr.us-west-2.amazonaws.com/source@sha256:" + "b" * 64
HEADER = {
    "job_name": "owned-data-prep-a",
    "runtime_session_id": SESSION,
    "deployment_id": "12345678-1234-4234-9234-111122223333",
    "deployment_prefix": "owned",
    "image_uri": IMAGE,
    "source_commit": SOURCE,
    "profile_id": "bigcode.commitpackft.python.change_sft.v1",
    "policy_version": "coding-preparation.commitpack.v3",
}


def catalog(**change):
    return SftCatalog(
        **{
            **dict(
                account_id=ACCOUNT,
                region="us-west-2",
                bucket=BUCKET,
                execution_role_arn="arn:aws:iam::" + ACCOUNT + ":role/owned",
                image_uri=IMAGE,
                source_commit=SOURCE,
                admission_digest="b" * 64,
                deployment_id=HEADER["deployment_id"],
                deployment_prefix="owned",
            ),
            **change,
        }
    )


def encoded(v):
    return (json.dumps(v, sort_keys=True, separators=(",", ":")) + "\n").encode()


def ref(name, raw=None, sha=None, count=None, fmt="json"):
    return DatasetArtifactRef(
        "s3://" + BUCKET + "/sessions/" + SESSION + "/original/" + name,
        "v-" + name.replace("/", "-"),
        sha or hashlib.sha256(raw).hexdigest(),
        count if count is not None else len(raw),
        fmt,
    )


class S3:
    def __init__(self):
        self.objects = {}
        self.uploads = {}
        self.mutations = []
        self.bodies = []

    def install(self, r, raw=None):
        self.objects[r.uri.split("/", 3)[3]] = (r, raw)

    def head_object(self, **kw):
        r, _raw = self.objects[kw["Key"]]
        if "VersionId" in kw:
            assert kw["VersionId"] == r.version_id
        return {
            "VersionId": r.version_id,
            "ContentLength": r.byte_count,
            "ChecksumType": "FULL_OBJECT",
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(r.sha256)).decode(),
        }

    def get_object(self, **kw):
        r, raw = self.objects[kw["Key"]]
        assert kw["VersionId"] == r.version_id
        body = io.BytesIO(raw)
        self.bodies.append(body)
        return {"VersionId": r.version_id, "ContentLength": r.byte_count, "Body": body}

    def get_bucket_versioning(self, **kw):
        return {"Status": "Enabled"}

    def create_multipart_upload(self, **kw):
        validate_parameters(kw, MODELS["s3"].operation_model("CreateMultipartUpload").input_shape)
        uid = str(len(self.uploads))
        self.uploads[uid] = {"kw": kw, "parts": []}
        self.mutations.append("multipart")
        return {"UploadId": uid}

    def upload_part_copy(self, **kw):
        from urllib.parse import unquote

        validate_parameters(kw, MODELS["s3"].operation_model("UploadPartCopy").input_shape)
        source, version = kw["CopySource"].split("?versionId=")
        key = unquote(source.split("/", 1)[1])
        r, raw = self.objects[key]
        assert unquote(version) == r.version_id
        # Primary S3 API: a copy range is allowed only for source objects >5MB.
        assert r.byte_count > 5 * 1024**2 or "CopySourceRange" not in kw
        self.uploads[kw["UploadId"]]["source"] = (r, raw)
        self.uploads[kw["UploadId"]]["parts"].append(kw)
        return {"CopySourceVersionId": r.version_id, "CopyPartResult": {"ETag": '"part"'}}

    def complete_multipart_upload(self, **kw):
        validate_parameters(kw, MODELS["s3"].operation_model("CompleteMultipartUpload").input_shape)
        assert kw["IfNoneMatch"] == "*" and kw["Key"] not in self.objects
        r, raw = self.uploads[kw["UploadId"]]["source"]
        new = dataclasses.replace(
            r, uri="s3://" + BUCKET + "/" + kw["Key"], version_id="copied-" + kw["UploadId"]
        )
        self.install(new, raw)
        return {"VersionId": new.version_id, "ETag": '"complete"'}

    def abort_multipart_upload(self, **kw):
        self.mutations.append("abort")

    def put_object(self, **kw):
        assert kw["IfNoneMatch"] == "*" and kw["Key"] not in self.objects
        raw = kw["Body"]
        r = DatasetArtifactRef(
            "s3://" + BUCKET + "/" + kw["Key"],
            "put-version",
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            "json",
        )
        self.install(r, raw)
        self.mutations.append("put")
        return {"VersionId": r.version_id}

    def list_objects_v2(self, **kw):
        keys = [k for k in self.objects if k.startswith(kw["Prefix"])]
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


def described_resource_config(resource):
    """ResourceConfig as DescribeTrainingJob returns it once SageMaker has chosen an instance.

    For a request with InstancePreferences the description adds SelectedInstanceType (here the first listed type)
    and SelectedInstanceCount, as the recorded real descriptions in
    fixtures/sagemaker_instance_preference_readback_real.json do. A fixed-type request is described unchanged.
    """
    value = copy.deepcopy(resource)
    if type(value) is dict and "InstancePreferences" in value:
        value["SelectedInstanceType"] = value["InstancePreferences"][0]["InstanceType"]
        value["SelectedInstanceCount"] = value["InstanceCount"]
    return value


class Native:
    meta = SimpleNamespace(region_name="us-west-2")

    def __init__(self):
        self.calls = []
        self.state = "InProgress"
        self.error = None
        self.reads = 0

    def create_training_job(self, **kw):
        validate_parameters(kw, MODELS["sagemaker"].operation_model("CreateTrainingJob").input_shape)
        self.calls.append(kw)
        if self.error:
            raise self.error
        return {
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:"
            + ACCOUNT
            + ":training-job/"
            + kw["TrainingJobName"]
        }

    def describe_training_job(self, **kw):
        self.reads += 1
        r = self.calls[0]
        return {
            **{k: v for k, v in r.items() if k != "Tags"},
            "ResourceConfig": described_resource_config(r["ResourceConfig"]),
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:"
            + ACCOUNT
            + ":training-job/"
            + r["TrainingJobName"],
            "TrainingJobStatus": self.state,
            "ModelArtifacts": {"S3ModelArtifacts": r["OutputDataConfig"]["S3OutputPath"] + "/model/"},
        }

    def list_tags(self, **kw):
        return {"Tags": self.calls[0]["Tags"]}


def sources():
    s3 = S3()
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    model = []
    for item in expected["files"]:
        fmt = (
            "safetensors"
            if item["role"] == "weight_shard"
            else "jinja"
            if item["role"] == "chat_template"
            else "json"
        )
        r = ref("model/" + item["path"], sha=item["sha256"], count=item["byte_count"], fmt=fmt)
        s3.install(r)
        model.append((item["path"], r))
    manifests = []
    for name in ("commit", "lineage", "receipt"):
        raw = b"{}"
        r = ref("model/" + name, raw)
        s3.install(r, raw)
        manifests.append((name, r))
    files = {}
    refs = {}
    for name in (
        "REPORT.json",
        "lineage.jsonl",
        "membership.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    ):
        raw = b"{}\n"
        r = ref("prepared/" + name, raw, fmt="jsonl" if name.endswith(".jsonl") else "json")
        s3.install(r, raw)
        refs[name] = r
        files[name] = {
            "sha256": r.sha256,
            "byte_count": r.byte_count,
            "format": r.format,
            "record_count": {
                "REPORT.json": 1,
                "lineage.jsonl": 100,
                "membership.jsonl": 100,
                "train.jsonl": 80,
                "validation.jsonl": 10,
                "test.jsonl": 10,
            }[name],
        }
    committed = {
        "schema_version": 2,
        **HEADER,
        "files": files,
        "counts": {
            "source": 100,
            "normalized": 100,
            "normalization_rejected": 0,
            "included_origins": 100,
            "excluded_origins": 0,
            "duplicate_origins": 0,
            "retained_representatives": 100,
            "split_counts": {"train": 80, "validation": 10, "test": 10},
        },
    }
    from amlas.data_eng.commitpack_preparation import COVERAGE

    committed["coverage"] = dict(COVERAGE)
    committed["exclusions_sha256"] = hashlib.sha256(b'{"repositories":[]}\n').hexdigest()
    raw = encoded(committed)
    commit = ref("prepared/COMMITTED.json", raw)
    s3.install(commit, raw)
    raw = b'{"repositories":[]}\n'
    ex = ref("exclusions.json", raw)
    s3.install(ex, raw)
    mod = VerifiedSftModelSource(
        SESSION,
        "qwen3_5_9b",
        "1",
        "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/owned-model-a",
        tuple(sorted(HEADER.items())),
        tuple(model),
        tuple(manifests),
    )
    prep = VerifiedSftPreparedSource(
        SESSION,
        "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/owned-data-prep-a",
        tuple(sorted(HEADER.items())),
        commit,
        tuple((s, refs[s + ".jsonl"]) for s in ("train", "validation", "test")),
        tuple(
            (n, refs[f])
            for n, f in [
                ("report", "REPORT.json"),
                ("lineage", "lineage.jsonl"),
                ("membership", "membership.jsonl"),
            ]
        ),
        ex,
        (("c" * 64, 100),),
    )
    return s3, VerifiedSftInputs(mod, prep)


def backend():
    s3, inputs = sources()
    native = Native()
    return SageMakerQwenSftBackend(sagemaker_client=native, s3_client=s3), native, s3, inputs


def test_fixed_native_request_stages_all_exact_versions_and_large_shard():
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    assert h.backend == "qwen_sft" and len(n.calls) == 1
    request = n.calls[0]
    assert (
        request["EnableNetworkIsolation"] is True
        and "Environment" not in request
        and "HyperParameters" not in request
    )
    assert {x["ChannelName"] for x in request["InputDataConfig"]} == {"base_model", "prepared", "bindings"}
    assert request["AlgorithmSpecification"]["ContainerEntrypoint"] == [
        "/usr/local/bin/amlas-processing-init",
        "qwen-sft",
    ]
    assert request["ResourceConfig"] == {
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
        "InstancePreferences": [
            {"InstanceType": name}
            for name in (
                "ml.g6e.16xlarge",
                "ml.g6e.12xlarge",
                "ml.g6e.8xlarge",
                "ml.g6e.4xlarge",
            )
        ],
    }
    assert (
        request["StoppingCondition"]["MaxRuntimeInSeconds"] == 3600
        and request["OutputDataConfig"]["CompressionType"] == "NONE"
    )
    assert len(h.extra["staged_refs"]) == 17  # 9 base+7 prepared+RUN
    large = [
        v
        for v in s.uploads.values()
        if v.get("source", [None])[0] and v["source"][0].byte_count > 5 * 1024**3
    ]
    assert len(large) == 1 and len(large[0]["parts"]) > 1
    assert all(body.closed for body in s.bodies)


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_uri", ""),
        ("admission_digest", ""),
        ("source_commit", ""),
        ("image_uri", "foreign"),
        ("admission_digest", "a" * 64),
        ("account_id", "9" * 12),
        ("region", "us-east-1"),
    ],
)
def test_missing_or_foreign_catalog_denies_before_staging(field, value):
    b, n, s, i = backend()
    with pytest.raises(ValueError):
        b.submit(inputs=i, catalog=catalog(**{field: value}), recipe_id="fit_v1", runtime_session_id=SESSION)
    assert n.calls == s.mutations == []


@pytest.mark.parametrize("change", ["session", "digest", "version", "extra", "boolcount"])
def test_source_drift_refuses_before_cloud_writes(change):
    b, n, s, i = backend()
    first = i.model.file_refs[0]
    if change == "session":
        i = dataclasses.replace(i, model=dataclasses.replace(i.model, runtime_session_id="foreign"))
    else:
        r = first[1]
        r = dataclasses.replace(
            r,
            **(
                {"sha256": "f" * 64}
                if change == "digest"
                else {"version_id": None}
                if change == "version"
                else {"byte_count": True}
                if change == "boolcount"
                else {}
            ),
        )
        records = ((first[0], r), *i.model.file_refs[1:])
        if change == "extra":
            records = (*records, ("evil.py", r))
        i = dataclasses.replace(i, model=dataclasses.replace(i.model, file_refs=records))
    with pytest.raises(ValueError):
        b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    assert n.calls == s.mutations == []


def test_unversioned_bucket_refuses_before_mutation():
    b, n, s, i = backend()
    s.get_bucket_versioning = lambda **kw: {}
    with pytest.raises(ValueError):
        b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    assert n.calls == s.mutations == []


def test_ambiguous_create_preserves_original_handle_and_native_status():
    b, n, _s, i = backend()
    n.error = RuntimeError("interpretation fault")
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    assert h.extra["submission_state"] == "ambiguous" and h.id == n.calls[0]["TrainingJobName"]
    assert b.status(h).state == JobState.RUNNING


def test_running_failed_and_unknown_never_yield_trained_artifacts():
    b, n, _s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    for state in ("InProgress", "Failed", "Mystery"):
        n.state = state
        result = b.artifacts(h)
        assert result["trained"] is False and result["quality_success"] is False


@pytest.mark.parametrize("change", ["role", "image", "channel", "isolation"])
def test_native_description_drift_refuses_before_output_reads(change):
    b, n, _s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    original = n.describe_training_job

    def altered(**kw):
        value = original(**kw)
        if change == "role":
            value["RoleArn"] = "foreign"
        elif change == "image":
            value["AlgorithmSpecification"] = {**value["AlgorithmSpecification"], "TrainingImage": "foreign"}
        elif change == "channel":
            value["InputDataConfig"] = []
        else:
            value["EnableNetworkIsolation"] = False
        return value

    n.describe_training_job = altered
    assert b.status(h).state == JobState.UNKNOWN


def metrics(recipe="fit_v1"):
    steps = 32 if recipe == "fit_v1" else 1024
    return {
        "schema_version": 1,
        "recipe_id": recipe,
        "optimizer_steps": steps,
        "supervised_tokens": 32000 if recipe == "fit_v1" else 1024000,
        "unique_training_rows": 80 if recipe == "fit_v1" else 1000,
        "original_provenance_rows": 80 if recipe == "fit_v1" else 1000,
        "unique_supervised_tokens": 8000 if recipe == "fit_v1" else 1000000,
        "optimizer_exposure_tokens": 32000 if recipe == "fit_v1" else 1024000,
        "epochs_completed": 0,
        "changed_tensor_count": 1,
        "train_history": [
            {"step": i + 1, "loss": 1.0, "supervised_tokens": 1000, "gradient_norm": 0.1}
            for i in range(steps)
        ],
        "validation_history": [
            {
                "step": step,
                "loss": 1.0,
                "best_loss": 1.0,
                "nonimproving_evaluations": 0,
                "early_stop_would_trigger": False,
            }
            for step in range(1, steps + 1)
            if step % 64 == 0 or step == steps
        ],
        "early_stop_would_trigger": False,
        "target_met": True,
        "status": "checkpoint_ready",
        "reload_pending": True,
        "quality_success": False,
        "vision_frozen": True,
        "peak_device_memory_bytes": 1000,
        "elapsed_seconds": 1.0,
        "tokens_per_second": 32000.0 if recipe == "fit_v1" else 1024000.0,
        "optimizer_state_saved": False,
    }


def test_fit_pilot_before_complete_epoch_is_valid_actual_metrics():
    SageMakerQwenSftBackend._metrics(metrics(), "fit_v1")


@pytest.mark.parametrize(
    "change",
    [
        {"optimizer_steps": True},
        {"elapsed_seconds": float("nan")},
        {"unique_training_rows": 999},
        {"unique_supervised_tokens": 999999},
        {"supervised_tokens": 999999},
        {"quality_success": True},
    ],
)
def test_full_counter_and_quality_substitutions_refuse(change):
    with pytest.raises(ValueError):
        SageMakerQwenSftBackend._metrics({**metrics("full_v1"), **change}, "full_v1")


def test_actual_preparation_descriptor_schema_preserved_in_run():
    b, _n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    commit = json.loads(s.objects[i.prepared.preparation_ref.uri.split("/", 3)[3]][1])
    run = json.loads(s.objects[h.extra["run_ref"].uri.split("/", 3)[3]][1])
    for split in ("train", "validation", "test"):
        assert run["split_refs"][split] == commit["files"][split + ".jsonl"]
        assert set(run["split_refs"][split]) == {"sha256", "byte_count", "format", "record_count"}


def output_prefix(request, layout="controlled"):
    """Key prefix (one trailing slash) of the model directory a completed job leaves.

    "controlled" is the layout the simulated provider and the older fixtures use: <S3OutputPath>/model/.
    "native" is what DescribeTrainingJob returns for a real job with CompressionType NONE:
    <S3OutputPath>/<TrainingJobName>/output/model (the trailing slash is not part of the real value).
    """
    root = request["OutputDataConfig"]["S3OutputPath"].split("/", 3)[3]
    if layout == "controlled":
        return root + "/model/"
    assert layout == "native"
    return root + "/" + request["TrainingJobName"] + "/output/model/"


def real_description(n, *, slash=False):
    """Make the fake SageMaker answer with the real ModelArtifacts shape (see output_prefix, layout "native")."""
    original = n.describe_training_job

    def described(**kw):
        value = original(**kw)
        request = n.calls[0]
        uri = "s3://" + BUCKET + "/" + output_prefix(request, "native")
        value["ModelArtifacts"] = {"S3ModelArtifacts": uri if slash else uri[:-1]}
        return value

    n.describe_training_job = described


def install_output(b, n, s, h, *, recipe="fit_v1", layout="controlled"):
    from amlas.training.qwen_sft import NORMALIZATION_VERSION, TEMPLATE_SHA256

    prefix = output_prefix(n.calls[0], layout)

    def put(name, raw):
        r = DatasetArtifactRef(
            "s3://" + BUCKET + "/" + prefix + name,
            "out-" + name,
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            "safetensors" if name.endswith(".safetensors") else "json",
        )
        s.install(r, raw)
        return r

    tensors = put("adapter_model.safetensors", b"bounded raw tensor fixture")
    config_ref = put("adapter_config.json", b'{"fixed":true}\n')
    checkpoint = {
        "schema_version": 1,
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "base_revision": expected_model_weight_manifest("qwen3_5_9b", "1")["revision"],
        "template_sha256": TEMPLATE_SHA256,
        "normalization_version": NORMALIZATION_VERSION,
        "tensor_keys": {
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": {
                "shape": [16, 32],
                "dtype": "BF16",
            }
        },
        "files": {
            "adapter_model.safetensors": {"sha256": tensors.sha256, "byte_count": tensors.byte_count},
            "adapter_config.json": {"sha256": config_ref.sha256, "byte_count": config_ref.byte_count},
        },
        "status": "checkpoint_saved",
        "reload_pending": True,
        "quality_success": False,
        "recipe_id": recipe,
        "optimizer_steps": 32 if recipe == "fit_v1" else 1024,
        "supervised_tokens": 32000 if recipe == "fit_v1" else 1024000,
        "greedy_input_ids": [[1]],
        "greedy_output_ids": [[1, 2]],
        "source_commit": SOURCE,
        "stage_manifest_sha256": h.extra["run_ref"].sha256,
    }
    checkpoint_ref = put("CHECKPOINT.json", encoded(checkpoint))
    proof = {
        "res_uid": [10001] * 3,
        "res_gid": [10001] * 3,
        "groups": [],
        "cap_eff": 0,
        "cap_prm": 0,
        "cap_inh": 0,
        "cap_amb": 0,
        "cap_bnd": 0,
        "no_new_privs": 1,
        "keep_caps": 0,
        "seccomp": 2,
        "fs_uid": 10001,
        "fs_gid": 10001,
        "securebits": 0,
    }
    from amlas.data_eng.commitpack_preparation import COVERAGE, SUPERVISION_KIND
    from amlas.training.qwen_sft import _compact_commitpack_replay

    original = json.loads(s.objects[h.extra["prepared_source"].preparation_ref.uri.split("/", 3)[3]][1])
    compact = _compact_commitpack_replay(
        {
            "counts": original["counts"],
            "coverage": dict(COVERAGE),
            "supervision_kind": SUPERVISION_KIND,
            "exclusion_policy_sha256": h.extra["prepared_source"].exclusions_ref.sha256,
            "canonical_sha256_by_split": {
                name: ["a" * 64] * count for name, count in original["counts"]["split_counts"].items()
            },
        }
    )
    receipt = {
        "schema_version": 1,
        "operation": "train",
        "source_commit": SOURCE,
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "recipe_id": recipe,
        "input_summary": {
            "preparation_commit_sha256": h.extra["prepared_source"].preparation_ref.sha256,
            "test_used_for_training": False,
            "preparation_profile_id": HEADER["profile_id"],
            "preparation_policy_version": HEADER["policy_version"],
            "commitpack": compact,
        },
        "privilege_proof": proof,
        "metrics": metrics(recipe),
        "status": "checkpoint_saved",
        "reload_pending": True,
        "quality_success": False,
    }
    training_ref = put("TRAINING.json", encoded(receipt))
    files = {
        r.uri.rsplit("/", 1)[1]: {"sha256": r.sha256, "byte_count": r.byte_count}
        for r in (tensors, config_ref, checkpoint_ref, training_ref)
    }
    commit = {
        "schema_version": 1,
        "operation": "train",
        "source_commit": SOURCE,
        "files": files,
        "status": "checkpoint_saved",
        "reload_pending": True,
        "quality_success": False,
    }
    put("COMMITTED.json", encoded(commit))
    n.state = "Completed"
    return prefix, receipt, commit, put


def test_completed_fit_outputs_are_versioned_not_full_or_quality():
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    install_output(b, n, s, h)
    result = b.artifacts(h)
    assert result["success"] is True and result["trained"] is True
    assert (
        result["training_stage"] == "fit"
        and result["full_training_complete"] is False
        and result["quality_success"] is False
    )
    assert result["checkpoint_ref"].version_id == "out-CHECKPOINT.json"
    assert all(body.closed for body in s.bodies)


@pytest.mark.parametrize(
    "change",
    [
        "same_length_bytes",
        "missing_version",
        "foreign_layout",
        "extra_object",
        "extra_receipt",
        "extra_metrics",
        "source",
        "bool_steps",
        "test_used",
    ],
)
def test_completed_native_outputs_refuse_tampering_even_with_rehashed_envelopes(change):
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    prefix, receipt, commit, put = install_output(b, n, s, h)
    if change == "same_length_bytes":
        ref, raw = s.objects[prefix + "adapter_model.safetensors"]
        s.objects[prefix + "adapter_model.safetensors"] = (ref, b"x" * len(raw))
    elif change == "missing_version":
        ref, raw = s.objects[prefix + "CHECKPOINT.json"]
        s.objects[prefix + "CHECKPOINT.json"] = (dataclasses.replace(ref, version_id=None), raw)
    elif change == "foreign_layout":
        original = n.describe_training_job

        def altered(**kw):
            value = original(**kw)
            value["ModelArtifacts"] = {"S3ModelArtifacts": "s3://foreign/output/"}
            return value

        n.describe_training_job = altered
    elif change == "extra_object":
        put("unexpected.pkl", b"data")
    else:
        if change == "extra_receipt":
            receipt["approved"] = True
        elif change == "extra_metrics":
            receipt["metrics"]["ready"] = True
        elif change == "source":
            receipt["source_commit"] = "f" * 40
        elif change == "bool_steps":
            receipt["metrics"]["optimizer_steps"] = True
        else:
            receipt["input_summary"]["test_used_for_training"] = True
        ref = put("TRAINING.json", encoded(receipt))
        commit["files"]["TRAINING.json"] = {"sha256": ref.sha256, "byte_count": ref.byte_count}
        put("COMMITTED.json", encoded(commit))
    assert b.artifacts(h)["trained"] is False


def _fresh_reload_uses_original_pinned_adapter_then_separate_completed_job(layout):
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    if layout == "native":
        real_description(n)
    install_output(b, n, s, h, layout=layout)
    result = b.artifacts(h)
    assert result["success"] is True, result
    from amlas.control_plane.sft_state import VerifiedSftCheckpointSource

    source = VerifiedSftCheckpointSource(
        SESSION, h.arn, "fit_v1", SOURCE, result["checkpoint_ref"], tuple(result["checkpoint_refs"].items())
    )
    n.state = "InProgress"
    reloaded = b.reload(model=i.model, checkpoint=source, catalog=catalog(), runtime_session_id=SESSION)
    assert len(n.calls) == 2 and h.id != reloaded.id
    req = n.calls[-1]
    assert {c["ChannelName"] for c in req["InputDataConfig"]} == {"base_model", "adapter", "bindings"}
    # The fake observer now returns the second original request; no new Create for polling.
    n.calls[:] = [req]
    n.state = "Completed"
    prefix = output_prefix(req, layout)
    raw = encoded(
        {
            "schema_version": 1,
            "operation": "reload",
            "source_commit": SOURCE,
            "catalog_id": "qwen3_5_9b",
            "catalog_version": "1",
            "recipe_id": "fit_v1",
            "checkpoint_sha256": result["checkpoint_ref"].sha256,
            "fresh_process_greedy_equal": True,
            "quality_success": False,
            "privilege_proof": {
                "res_uid": [10001] * 3,
                "res_gid": [10001] * 3,
                "groups": [],
                "cap_eff": 0,
                "cap_prm": 0,
                "cap_inh": 0,
                "cap_amb": 0,
                "cap_bnd": 0,
                "no_new_privs": 1,
                "keep_caps": 0,
                "seccomp": 2,
                "fs_uid": 10001,
                "fs_gid": 10001,
                "securebits": 0,
            },
        }
    )
    r = DatasetArtifactRef(
        "s3://" + BUCKET + "/" + prefix + "RELOAD.json",
        "reload-version",
        hashlib.sha256(raw).hexdigest(),
        len(raw),
        "json",
    )
    s.install(r, raw)
    raw = encoded(
        {
            "schema_version": 1,
            "operation": "reload",
            "source_commit": SOURCE,
            "files": {"RELOAD.json": {"sha256": r.sha256, "byte_count": r.byte_count}},
            "quality_success": False,
        }
    )
    s.install(
        DatasetArtifactRef(
            "s3://" + BUCKET + "/" + prefix + "COMMITTED.json",
            "reload-commit",
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            "json",
        ),
        raw,
    )
    got = b.artifacts(reloaded)
    assert (
        got["success"] is True
        and got["fresh_process_greedy_equal"] is True
        and got["quality_success"] is False
    )


def test_fresh_reload_uses_original_pinned_adapter_then_separate_completed_job():
    _fresh_reload_uses_original_pinned_adapter_then_separate_completed_job("controlled")


def test_fresh_reload_uses_original_pinned_adapter_then_separate_completed_job_native_layout():
    _fresh_reload_uses_original_pinned_adapter_then_separate_completed_job("native")


@pytest.mark.parametrize("field", ["Environment", "HyperParameters"])
@pytest.mark.parametrize("value", [{"unexpected": "1"}, None, False, []])
def test_native_forbidden_fields_refuse_before_output_read(field, value):
    b, n, _s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    original = n.describe_training_job

    def changed(**kw):
        result = original(**kw)
        result[field] = value
        return result

    n.describe_training_job = changed
    assert b.status(h).state == JobState.UNKNOWN


@pytest.mark.parametrize("field", ["Environment", "HyperParameters"])
def test_native_absent_or_empty_map_forbidden_field_is_harmless(field):
    b, n, _s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    original = n.describe_training_job

    def changed(**kw):
        result = original(**kw)
        result[field] = {}
        return result

    n.describe_training_job = changed
    assert b.status(h).state == JobState.RUNNING


def test_native_tag_order_changes_are_semantically_equal():
    b, n, _s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    n.list_tags = lambda **kw: {"Tags": list(reversed(n.calls[0]["Tags"]))}
    assert b.status(h).state == JobState.RUNNING


@pytest.mark.parametrize("kind", ["duplicate", "changed", "missing", "extra", "pagination"])
def test_native_tag_inventory_substitution_refuses(kind):
    b, n, _s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    tags = [dict(x) for x in n.calls[0]["Tags"]]
    value = {"Tags": tags}
    if kind == "duplicate":
        tags.append(dict(tags[0]))
    elif kind == "changed":
        tags[0]["Value"] = "foreign"
    elif kind == "missing":
        tags.pop()
    elif kind == "extra":
        tags.append({"Key": "unexpected", "Value": "1"})
    else:
        value["NextToken"] = "more"
    n.list_tags = lambda **kw: value
    assert b.status(h).state == JobState.UNKNOWN


@pytest.mark.parametrize("recipe", ["fit_v1", "full_v1"])
def test_rehashed_malformed_histories_never_promote(recipe):
    value = metrics(recipe)
    value["train_history"] = [{"loss": 1.0}]
    value["validation_history"] = [{"loss": 1.0}]
    with pytest.raises(ValueError):
        SageMakerQwenSftBackend._metrics(value, recipe)


def test_unique_corpus_tokens_and_optimizer_exposure_are_independent():
    value = metrics("fit_v1")
    value["unique_supervised_tokens"] = value["supervised_tokens"] + 1
    SageMakerQwenSftBackend._metrics(value, "fit_v1")


def test_original_prepared_train_count_bounds_completion_even_when_rehashed():
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="full_v1", runtime_session_id=SESSION)
    install_output(b, n, s, h, recipe="full_v1")  # Only80 prepared rows cannot witness1000unique rows.
    assert b.artifacts(h)["full_training_complete"] is False
    assert b.artifacts(h)["trained"] is False


def test_original_catalog_owned_bucket_global_exclusions_are_supported():
    b, n, s, i = backend()
    old = i.prepared.exclusions_ref
    new = dataclasses.replace(old, uri="s3://" + BUCKET + "/config/exclusions.json")
    s.install(new, s.objects[old.uri.split("/", 3)[3]][1])
    i = dataclasses.replace(i, prepared=dataclasses.replace(i.prepared, exclusions_ref=new))
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    assert h.extra["submission_state"] == "accepted" and len(n.calls) == 1


def test_deployment_prefix_preserves_upstream_40_character_grammar():
    from amlas.control_plane.sft_state import validate_catalog

    validate_catalog(catalog(deployment_prefix="a" * 40))
    with pytest.raises(ValueError):
        validate_catalog(catalog(deployment_prefix="a" * 41))


def test_original_exclusion_digest_cannot_change_with_rehashed_ref():
    b, n, s, i = backend()
    raw = b'{"repositories":["held-out"]}\n'
    changed = ref("different-exclusions.json", raw)
    s.install(changed, raw)
    i = dataclasses.replace(i, prepared=dataclasses.replace(i.prepared, exclusions_ref=changed))
    with pytest.raises(ValueError):
        b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    assert n.calls == s.mutations == []
