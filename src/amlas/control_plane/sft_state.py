"""Private immutable SFT sources; descriptors carry no approval or readiness."""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..backends.types import DatasetArtifactRef, JobHandle
from ..data_eng.model_weight_manifest import expected_model_weight_manifest

if TYPE_CHECKING:
    from ..backends.campaign_state import CampaignStateBackend
    from .campaign_state import TrustedExecutionBinding


@dataclass(frozen=True)
class SftCatalog:
    account_id: str
    region: str
    bucket: str
    execution_role_arn: str
    image_uri: str
    source_commit: str
    admission_digest: str
    deployment_id: str
    deployment_prefix: str


@dataclass(frozen=True)
class VerifiedSftModelSource:
    runtime_session_id: str
    catalog_id: str
    catalog_version: str
    producer_job_arn: str
    producer_bindings: tuple[tuple[str, str], ...]
    file_refs: tuple[tuple[str, DatasetArtifactRef], ...]
    manifest_refs: tuple[tuple[str, DatasetArtifactRef], ...]


@dataclass(frozen=True)
class VerifiedSftAdoptedModelSource:
    runtime_session_id: str
    catalog_id: str
    catalog_version: str
    original: VerifiedSftModelSource
    file_refs: tuple[tuple[str, DatasetArtifactRef], ...]
    manifest_refs: tuple[tuple[str, DatasetArtifactRef], ...]
    adoption_ref: DatasetArtifactRef


@dataclass(frozen=True)
class VerifiedSftPreparedSource:
    runtime_session_id: str
    producer_job_arn: str
    header: tuple[tuple[str, str], ...]
    preparation_ref: DatasetArtifactRef
    split_refs: tuple[tuple[str, DatasetArtifactRef], ...]
    sidecar_refs: tuple[tuple[str, DatasetArtifactRef], ...]
    exclusions_ref: DatasetArtifactRef
    source_row_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class VerifiedSftInputs:
    model: VerifiedSftModelSource | VerifiedSftAdoptedModelSource
    prepared: VerifiedSftPreparedSource


@dataclass(frozen=True)
class VerifiedSftCheckpointSource:
    runtime_session_id: str
    producer_job_arn: str
    recipe_id: str
    source_commit: str
    checkpoint_ref: DatasetArtifactRef
    file_refs: tuple[tuple[str, DatasetArtifactRef], ...]


@dataclass(frozen=True)
class ReloadCompatibilitySelection:
    """Exact descriptive producer/consumer tuple; persisted values grant nothing."""

    producer_catalog: SftCatalog
    consumer_catalog: SftCatalog
    checkpoint: VerifiedSftCheckpointSource
    contract_version: str
    base_manifest_sha256: str


_RELOAD_CATALOG_SEAL = object()


@dataclass(frozen=True)
class _TrustedReloadCompatibilityCatalog:
    selections: tuple[ReloadCompatibilitySelection, ...]
    issuer_pid: int
    seal: object


def _reload_base_manifest_sha256() -> str:
    import hashlib
    import json

    files = expected_model_weight_manifest("qwen3_5_9b", "1")["files"]
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_reload_compatibility(
    selection: ReloadCompatibilitySelection,
    catalog: SftCatalog,
    checkpoint: VerifiedSftCheckpointSource,
) -> None:
    from ..training.qwen_sft import RELOAD_CONTRACT_VERSION

    if type(selection) is not ReloadCompatibilitySelection:
        raise ValueError("typed reload compatibility required")
    validate_catalog(selection.producer_catalog)
    validate_catalog(selection.consumer_catalog)
    if type(checkpoint) is not VerifiedSftCheckpointSource or selection.checkpoint != checkpoint:
        raise ValueError("original reload checkpoint differs")
    if (
        selection.consumer_catalog != catalog
        or selection.contract_version != RELOAD_CONTRACT_VERSION
        or selection.base_manifest_sha256 != _reload_base_manifest_sha256()
        or checkpoint.source_commit != selection.producer_catalog.source_commit
        or checkpoint.source_commit == catalog.source_commit
        or checkpoint.recipe_id not in ("fit_v1", "full_v1")
        or type(checkpoint.runtime_session_id) is not str
    ):
        raise ValueError("reload compatibility source/contract differs")
    fields = ("account_id", "region", "bucket", "deployment_id", "deployment_prefix")
    if any(getattr(selection.producer_catalog, k) != getattr(catalog, k) for k in fields):
        raise ValueError("reload compatibility deployment differs")
    if type(checkpoint.producer_job_arn) is not str or not checkpoint.producer_job_arn.startswith(
        "arn:aws:sagemaker:"
        + catalog.region
        + ":"
        + catalog.account_id
        + ":training-job/"
        + catalog.deployment_prefix
        + "-"
    ):
        raise ValueError("original reload producer differs")
    refs = _pairs(
        checkpoint.file_refs, {"CHECKPOINT.json", "adapter_config.json", "adapter_model.safetensors"}
    )
    if refs["CHECKPOINT.json"] != checkpoint.checkpoint_ref:
        raise ValueError("reload checkpoint reference differs")
    for name, ref in refs.items():
        validate_ref(
            ref,
            catalog,
            checkpoint.runtime_session_id,
            limit=2 * 1024**3 if name.endswith(".safetensors") else 262144,
        )
        if ref.format != ("safetensors" if name.endswith(".safetensors") else "json"):
            raise ValueError("reload compatibility format differs")


def reload_compatibility_digest(selection: ReloadCompatibilitySelection) -> str:
    import dataclasses
    import hashlib
    import json

    validate_reload_compatibility(selection, selection.consumer_catalog, selection.checkpoint)
    raw = json.dumps(
        dataclasses.asdict(selection),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    if len(raw) > 65536:
        raise ValueError("reload compatibility tuple bound exceeded")
    return hashlib.sha256(b"amlas.reload.compatibility.v1\0" + raw).hexdigest()


def _seal_reload_compatibility_catalog(selections: tuple) -> _TrustedReloadCompatibilityCatalog:
    import os

    if type(selections) is not tuple or len(selections) > 16:
        raise ValueError("bounded reload compatibility catalog required")
    digests = [reload_compatibility_digest(value) for value in selections]
    if len(set(digests)) != len(digests):
        raise ValueError("duplicate reload compatibility selection")
    return _TrustedReloadCompatibilityCatalog(selections, os.getpid(), _RELOAD_CATALOG_SEAL)


def _fresh_reload_catalog(reader) -> _TrustedReloadCompatibilityCatalog:
    import os

    if not callable(reader):
        raise ValueError("fresh server reload compatibility reader required")
    value = reader()
    if (
        type(value) is not _TrustedReloadCompatibilityCatalog
        or value.issuer_pid != os.getpid()
        or value.seal is not _RELOAD_CATALOG_SEAL
    ):
        raise ValueError("fresh trusted reload compatibility catalog required")
    for selection in value.selections:
        reload_compatibility_digest(selection)
    return value


def select_reload_compatibility(reader, catalog, checkpoint) -> ReloadCompatibilitySelection:
    matches = [
        v
        for v in _fresh_reload_catalog(reader).selections
        if v.consumer_catalog == catalog and v.checkpoint == checkpoint
    ]
    if len(matches) != 1:
        raise ValueError("exact fresh reload compatibility absent")
    validate_reload_compatibility(matches[0], catalog, checkpoint)
    return matches[0]


def select_historical_reload_compatibility(reader, catalog, producer_catalog, producer_arn, session):
    matches = [
        v
        for v in _fresh_reload_catalog(reader).selections
        if v.consumer_catalog == catalog
        and v.producer_catalog == producer_catalog
        and v.checkpoint.producer_job_arn == producer_arn
        and v.checkpoint.runtime_session_id == session
    ]
    if len(matches) != 1:
        raise ValueError("exact fresh historical producer selection absent")
    validate_reload_compatibility(matches[0], catalog, matches[0].checkpoint)
    return matches[0]


def fixed_reload_binding(catalog, checkpoint, selection=None) -> dict:
    if type(checkpoint) is not VerifiedSftCheckpointSource:
        raise ValueError("original typed reload checkpoint required")
    value = {
        "schema_version": 1,
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "recipe_id": checkpoint.recipe_id,
        "model_files": expected_model_weight_manifest("qwen3_5_9b", "1")["files"],
        "checkpoint_ref": {
            "sha256": checkpoint.checkpoint_ref.sha256,
            "byte_count": checkpoint.checkpoint_ref.byte_count,
        },
    }
    if selection is None:
        if checkpoint.source_commit != catalog.source_commit:
            raise ValueError("reload compatibility absent")
        return value
    validate_reload_compatibility(selection, catalog, checkpoint)
    value.update(
        schema_version=2,
        operation="reload",
        recipe_version="1",
        source_commit=catalog.source_commit,
        checkpoint_source_commit=checkpoint.source_commit,
        reload_contract_version=selection.contract_version,
        compatibility_digest=reload_compatibility_digest(selection),
    )
    return value


@dataclass(frozen=True)
class SftSubmissionSnapshot:
    """Closed descriptive original submission evidence, never action authority."""

    operation_digest: str
    native_job_name: str
    native_job_arn: str
    request_json: str
    run_ref: DatasetArtifactRef
    staged_refs: tuple[tuple[str, DatasetArtifactRef], ...]
    training_record_count: int | None


# Historical reload contract: exactly one ml.g6e.4xlarge. Records that carry it keep reconstructing that request.
RELOAD_RESOURCE_CONTRACT = "reload_single_gpu_v1"
# New ordinary reloads pin this contract: one instance chosen by SageMaker from RELOAD_INSTANCE_PREFERENCES (one
# type since 2026-11-11, still sent as a list).
RELOAD_PREFERENCE_CONTRACT = "reload_instance_preferences_v1"
RELOAD_RESOURCE_CONTRACTS = (RELOAD_RESOURCE_CONTRACT, RELOAD_PREFERENCE_CONTRACT)

# Ordered instance types SageMaker may choose from (ResourceConfig.InstancePreferences, mutually exclusive with
# InstanceType; botocore >= 1.43.90). The account allows one training job per ml.g6e type. Requests that named one
# fixed type got no capacity 4 of 4 times on 2026-11-10; requests with a list got capacity in 61 s to 37 min in the
# recorded real jobs, where SageMaker selected ml.g6e.16xlarge for full and fit training and for the reload, and
# ml.g6e.4xlarge for a generation. Every listed type has NVIDIA L40S GPUs with 48 GiB each; the trainer and the
# reload use cuda:0 only. ml.g6e.2xlarge is not listed (64 GiB host memory). Generation does not list
# ml.g6e.12xlarge: its four GPUs change the generation server's device layout.
# SageMaker refuses a create when ANY listed type is at the account limit and names that type (CloudTrail,
# 2026-11-10), and a rejected fine-tune or reload is never submitted again. The fine-tune runs alone, so it lists all
# four types. The reload and a generation may be submitted in either order in one message, so their lists share no
# type: a job that holds a type of one list cannot refuse the other. Base and tuned generation share a list; the
# second is refused while the first holds its type and is submitted again after it ends.
# Split of 2026-11-11 (before: reload 16xlarge, 12xlarge; generation 4xlarge, 8xlarge). In the full run the stages
# after the fine-tune have no wake path and must finish inside one 8-hour session, and the two generation jobs run
# one after the other, so generation is the capacity bottleneck. Measured in this account since 2026-10-01: list
# jobs that received a machine selected 12xlarge 6 times and 16xlarge 4 times; 4xlarge was delivered 5 times after
# waits of 69 to 14,985 s and missed in three 20-minute checks; 8xlarge has never been delivered. Generation
# therefore lists 16xlarge first, then 4xlarge, then 8xlarge, and the reload keeps 12xlarge alone. In every
# interleaving of the reload and the base generation (either order) followed by the tuned generation, the only
# create that can be refused is the tuned generation while the base generation holds its type, and that create is
# submitted again.
# The reload's one type is sent as a one-entry InstancePreferences list, not as a fixed InstanceType: the service
# model of the locked SDK (botocore 1.43.127) and the API reference allow 1 to 5 entries, and the list shape keeps
# SelectedInstanceType in the reload's status and artifacts. Not yet on record: a real create with a one-entry list
# (the account's 14 list jobs carried 2 to 5 types). A capacity check that sends this exact ResourceConfig before a
# real run settles it; a refused reload create would not be submitted again.
# These three lists are the only ones a fixed request may carry; a list with other members or another order is refused.
TRAIN_INSTANCE_PREFERENCES = ("ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge")
RELOAD_INSTANCE_PREFERENCES = ("ml.g6e.12xlarge",)
GENERATION_INSTANCE_PREFERENCES = ("ml.g6e.16xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge")
PINNED_INSTANCE_PREFERENCES = (
    TRAIN_INSTANCE_PREFERENCES,
    RELOAD_INSTANCE_PREFERENCES,
    GENERATION_INSTANCE_PREFERENCES,
)


def _fixed_type_resource_config(instance_type: str) -> dict:
    """The historical request shape: exactly one instance of one named type."""
    return {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 100}


def fixed_resource_config(preferences: tuple[str, ...]) -> dict:
    """The fixed native ResourceConfig for one pinned preference list: one instance of a listed type."""
    if (
        type(preferences) is not tuple
        or any(type(name) is not str for name in preferences)
        or preferences not in PINNED_INSTANCE_PREFERENCES
    ):
        raise ValueError("fixed instance preference list differs")
    return {
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
        "InstancePreferences": [{"InstanceType": name} for name in preferences],
    }


def fixed_sft_request(
    catalog: SftCatalog,
    session: str,
    job: str,
    operation: str,
    recipe_id: str,
    run_ref: DatasetArtifactRef,
    *,
    resource_contract: str | None = None,
    historical: bool = False,
) -> dict:
    """Reproduce the fixed native request without constructing clients or effects.

    Training lists TRAIN_INSTANCE_PREFERENCES; `historical=True` reproduces the earlier fixed ml.g6e.12xlarge training
    request, so a record written before the preference lists still restores. A reload without a contract (record
    schemas 1 to 3) keeps the exact ml.g6e.12xlarge request, `reload_single_gpu_v1` the exact ml.g6e.4xlarge one,
    and `reload_instance_preferences_v1` lists RELOAD_INSTANCE_PREFERENCES.
    """
    validate_catalog(catalog)
    if operation not in ("train", "reload") or recipe_id not in ("fit_v1", "full_v1"):
        raise ValueError("fixed SFT operation required")
    if resource_contract is not None and (
        operation != "reload"
        or type(resource_contract) is not str
        or resource_contract not in RELOAD_RESOURCE_CONTRACTS
    ):
        raise ValueError("fixed reload resource contract differs")
    if type(historical) is not bool or (historical and operation != "train"):
        raise ValueError("historical fixed request is a training request")
    validate_ref(run_ref, catalog, session, limit=262144)
    prefix = "sessions/" + session + "/sft/" + job + "/"
    if run_ref.uri != "s3://" + catalog.bucket + "/" + prefix + "input/bindings/RUN.json":
        raise ValueError("original SFT RUN namespace differs")
    other_channel = "prepared" if operation == "train" else "adapter"
    tags = [
        {"Key": "amlas:" + k, "Value": v}
        for k, v in [
            ("deployment-id", catalog.deployment_id),
            ("deployment-prefix", catalog.deployment_prefix),
            ("runtime-session-id", session),
        ]
    ]
    cli = [
        "--operation",
        operation,
        "--recipe-id",
        recipe_id,
        "--recipe-version",
        "1",
        "--source-commit",
        catalog.source_commit,
        "--manifest-sha256",
        run_ref.sha256,
        "--manifest-byte-count",
        str(run_ref.byte_count),
    ]
    request = {
        "TrainingJobName": job,
        "RoleArn": catalog.execution_role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": catalog.image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "qwen-sft"],
            "ContainerArguments": cli,
        },
        "InputDataConfig": [
            {
                "ChannelName": channel,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://" + catalog.bucket + "/" + prefix + "input/" + channel + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
                "CompressionType": "None",
            }
            for channel in ("base_model", other_channel, "bindings")
        ],
        "ResourceConfig": (
            fixed_resource_config(RELOAD_INSTANCE_PREFERENCES)
            if resource_contract == RELOAD_PREFERENCE_CONTRACT
            else _fixed_type_resource_config("ml.g6e.4xlarge")
            if resource_contract == RELOAD_RESOURCE_CONTRACT
            else fixed_resource_config(TRAIN_INSTANCE_PREFERENCES)
            if operation == "train" and not historical
            else _fixed_type_resource_config("ml.g6e.12xlarge")
        ),
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600 if recipe_id == "fit_v1" else 86400},
        "EnableNetworkIsolation": True,
        "EnableInterContainerTrafficEncryption": False,
        "OutputDataConfig": {
            "S3OutputPath": "s3://" + catalog.bucket + "/" + prefix + "output",
            "CompressionType": "NONE",
        },
        "Tags": tags,
    }
    return request


class SftConfigurationMissingError(ValueError):
    def __init__(self, names: tuple[str, ...]):
        self.missing_selectors = names
        super().__init__("SFT startup configuration is missing")


def validate_catalog(catalog: SftCatalog) -> None:
    if type(catalog) is not SftCatalog:
        raise ValueError("typed SFT catalog required")
    fields = {
        "AMLAS_AWS_ACCOUNT_ID": catalog.account_id,
        "AMLAS_AWS_REGION": catalog.region,
        "AMLAS_SESSION_BUCKET": catalog.bucket,
        "SAGEMAKER_EXECUTION_ROLE_ARN": catalog.execution_role_arn,
        "AMLAS_QWEN_SFT_IMAGE_URI": catalog.image_uri,
        "AMLAS_QWEN_SFT_SOURCE_COMMIT": catalog.source_commit,
        "AMLAS_QWEN_SFT_ADMISSION_DIGEST": catalog.admission_digest,
        "AMLAS_DEPLOYMENT_ID": catalog.deployment_id,
        "AMLAS_DEPLOYMENT_PREFIX": catalog.deployment_prefix,
    }
    missing = tuple(k for k, v in fields.items() if type(v) is str and v == "")
    if missing:
        raise SftConfigurationMissingError(missing)
    if any(type(v) is not str for v in fields.values()):
        raise ValueError("SFT catalog scalar differs")
    if not re.fullmatch(r"\d{12}", catalog.account_id) or not re.fullmatch(
        r"[a-z]{2}(?:-[a-z]+)+-\d", catalog.region
    ):
        raise ValueError("SFT account/region differs")
    image = re.fullmatch(
        catalog.account_id
        + r"\.dkr\.ecr\."
        + re.escape(catalog.region)
        + r"\.amazonaws\.com/[a-z0-9][a-z0-9._/-]*@sha256:([a-f0-9]{64})",
        catalog.image_uri,
    )
    if (
        not image
        or image[1] != catalog.admission_digest
        or not re.fullmatch("[a-f0-9]{40}", catalog.source_commit)
    ):
        raise ValueError("immutable paired SFT image/source admission required")
    if not re.fullmatch(
        r"arn:aws:iam::" + catalog.account_id + r":role/[A-Za-z0-9_+=,.@/-]{1,512}",
        catalog.execution_role_arn,
    ):
        raise ValueError("SFT role belongs to another account")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", catalog.bucket) or not re.fullmatch(
        r"[a-z][a-z0-9-]{0,39}", catalog.deployment_prefix
    ):
        raise ValueError("SFT namespace differs")
    try:
        uuid.UUID(catalog.deployment_id)
    except (ValueError, TypeError):
        raise ValueError("SFT deployment identity differs") from None


def validate_ref(
    ref: DatasetArtifactRef,
    catalog: SftCatalog,
    session: str,
    *,
    limit: int = 8 * 1024**3,
    catalog_owned: bool = False,
) -> None:
    if (
        type(ref) is not DatasetArtifactRef
        or type(session) is not str
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session)
    ):
        raise ValueError("typed confined SFT artifact required")
    prefix = "s3://" + catalog.bucket + ("/" if catalog_owned else "/sessions/" + session + "/")
    if (
        type(ref.uri) is not str
        or not ref.uri.startswith(prefix)
        or any(x in ref.uri[len(prefix) :].split("/") for x in ("", ".", ".."))
        or any(c in ref.uri for c in ("?", "#", "\\"))
    ):
        raise ValueError("SFT artifact belongs to another namespace")
    if (
        type(ref.version_id) is not str
        or not 0 < len(ref.version_id) <= 1024
        or ref.version_id == "null"
        or any(ord(c) < 33 or ord(c) > 126 for c in ref.version_id)
    ):
        raise ValueError("immutable SFT version required")
    if (
        type(ref.sha256) is not str
        or not re.fullmatch("[a-f0-9]{64}", ref.sha256)
        or type(ref.byte_count) is not int
        or not 0 < ref.byte_count <= limit
    ):
        raise ValueError("SFT artifact byte/hash bound differs")
    if ref.format not in ("json", "jsonl", "jinja", "safetensors"):
        raise ValueError("SFT artifact format differs")


def _pairs(value: Any, names: set[str]) -> dict:
    if (
        type(value) is not tuple
        or len(value) != len(names)
        or any(type(x) is not tuple or len(x) != 2 for x in value)
    ):
        raise ValueError("closed SFT immutable mapping required")
    result = dict(value)
    if len(result) != len(value) or set(result) != names:
        raise ValueError("SFT mapping inventory differs")
    return result


def _validate_adopted_model(model: VerifiedSftAdoptedModelSource, catalog: SftCatalog, session: str) -> None:
    original = model.original
    if (
        model.runtime_session_id != session
        or (model.catalog_id, model.catalog_version) != ("qwen3_5_9b", "1")
        or type(original) is not VerifiedSftModelSource
        or (original.catalog_id, original.catalog_version) != (model.catalog_id, model.catalog_version)
    ):
        raise ValueError("adopted SFT model source session/catalog differs")
    # Preserve the actual producer and its original namespace; adoption grants no
    # authority to relabel the original or weaken the existing original validator.
    validate_model(original, catalog, original.runtime_session_id)
    expected = {item["path"] for item in expected_model_weight_manifest("qwen3_5_9b", "1")["files"]}
    original_files = _pairs(original.file_refs, expected)
    for name, ref in _pairs(model.file_refs, expected).items():
        validate_ref(ref, catalog, session)
        prior = original_files[name]
        if (ref.sha256, ref.byte_count, ref.format) != (prior.sha256, prior.byte_count, prior.format):
            raise ValueError("adopted SFT model source bytes differ")
    original_manifests = _pairs(original.manifest_refs, {"commit", "lineage", "receipt"})
    for name, ref in _pairs(model.manifest_refs, set(original_manifests)).items():
        validate_ref(ref, catalog, session, limit=8 * 1024**2)
        prior = original_manifests[name]
        if ref.format != "json" or (ref.sha256, ref.byte_count, ref.format) != (
            prior.sha256,
            prior.byte_count,
            prior.format,
        ):
            raise ValueError("adopted SFT model manifest bytes differ")
    validate_ref(model.adoption_ref, catalog, session, limit=1024**2)
    if model.adoption_ref.format != "json":
        raise ValueError("adopted SFT model receipt format differs")


def validate_model(
    model: VerifiedSftModelSource | VerifiedSftAdoptedModelSource,
    catalog: SftCatalog,
    session: str,
) -> None:
    if type(model) is VerifiedSftAdoptedModelSource:
        _validate_adopted_model(model, catalog, session)
        return
    if (
        type(model) is not VerifiedSftModelSource
        or model.runtime_session_id != session
        or (model.catalog_id, model.catalog_version) != ("qwen3_5_9b", "1")
    ):
        raise ValueError("SFT model source session/catalog differs")
    _producer(model.producer_job_arn, catalog)
    bindings = dict(model.producer_bindings)
    if (
        bindings.get("runtime_session_id") != session
        or bindings.get("deployment_id") != catalog.deployment_id
        or bindings.get("deployment_prefix") != catalog.deployment_prefix
    ):
        raise ValueError("SFT model original bindings differ")
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")["files"]
    refs = _pairs(model.file_refs, {x["path"] for x in expected})
    for item in expected:
        ref = refs[item["path"]]
        validate_ref(ref, catalog, session)
        fmt = (
            "safetensors"
            if item["role"] == "weight_shard"
            else "jinja"
            if item["role"] == "chat_template"
            else "json"
        )
        if (ref.sha256, ref.byte_count, ref.format) != (item["sha256"], item["byte_count"], fmt):
            raise ValueError("SFT model source bytes differ")
    for ref in _pairs(model.manifest_refs, {"commit", "lineage", "receipt"}).values():
        validate_ref(ref, catalog, session, limit=8 * 1024**2)


def _producer(arn: str, catalog: SftCatalog) -> None:
    if type(arn) is not str or not arn.startswith(
        "arn:aws:sagemaker:"
        + catalog.region
        + ":"
        + catalog.account_id
        + ":processing-job/"
        + catalog.deployment_prefix
        + "-"
    ):
        raise ValueError("foreign SFT producer identity")


def validate_inputs(inputs: VerifiedSftInputs, catalog: SftCatalog, session: str) -> None:
    validate_catalog(catalog)
    if type(inputs) is not VerifiedSftInputs:
        raise ValueError("typed SFT inputs required")
    validate_model(inputs.model, catalog, session)
    prep = inputs.prepared
    if type(prep) is not VerifiedSftPreparedSource or prep.runtime_session_id != session:
        raise ValueError("SFT preparation session differs")
    _producer(prep.producer_job_arn, catalog)
    header = _pairs(
        prep.header,
        {
            "job_name",
            "runtime_session_id",
            "deployment_id",
            "deployment_prefix",
            "image_uri",
            "source_commit",
            "profile_id",
            "policy_version",
        },
    )
    if (
        any(type(v) is not str or not v for v in header.values())
        or header["runtime_session_id"] != session
        or header["deployment_id"] != catalog.deployment_id
        or header["deployment_prefix"] != catalog.deployment_prefix
        or not prep.producer_job_arn.endswith("/" + header["job_name"])
    ):
        raise ValueError("SFT original preparation header differs")
    if (header["profile_id"], header["policy_version"]) not in {
        ("bigcode.commitpackft.python.change_sft.v1", "coding-preparation.commitpack.v3"),
        ("openai.explicit_conversations.v1", "coding-preparation.v1"),
        ("nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2", "coding-preparation.open-swe.v2"),
    }:
        raise ValueError("SFT preparation profile/policy pair differs")
    refs = [
        prep.preparation_ref,
        *_pairs(prep.split_refs, {"train", "validation", "test"}).values(),
        *_pairs(prep.sidecar_refs, {"report", "lineage", "membership"}).values(),
    ]
    for ref in refs:
        validate_ref(ref, catalog, session, limit=256 * 1024**2)
    if (
        prep.preparation_ref.format != "json"
        or any(r.format != "jsonl" for r in dict(prep.split_refs).values())
        or dict(prep.sidecar_refs)["report"].format != "json"
        or any(dict(prep.sidecar_refs)[n].format != "jsonl" for n in ("lineage", "membership"))
        or sum(r.byte_count for r in refs) > 256 * 1024**2
    ):
        raise ValueError("SFT preparation format/aggregate differs")
    validate_ref(prep.exclusions_ref, catalog, session, limit=8 * 1024**2, catalog_owned=True)
    if prep.exclusions_ref.format != "json":
        raise ValueError("SFT catalog exclusions format differs")
    counts = prep.source_row_counts
    if (
        type(counts) is not tuple
        or any(
            type(x) is not tuple
            or len(x) != 2
            or type(x[0]) is not str
            or not re.fullmatch("[a-f0-9]{64}", x[0])
            or type(x[1]) is not int
            or not 0 <= x[1] <= 100000
            for x in counts
        )
        or len(dict(counts)) != len(counts)
    ):
        raise ValueError("SFT private original counts differ")
    if header["policy_version"] == "coding-preparation.commitpack.v3" and (
        not counts or not 0 < sum(dict(counts).values()) <= 100000
    ):
        raise ValueError("SFT original CommitPack counts required")


def _validate_campaign_binding(catalog, runtime_session_id, campaign_state, execution_binding) -> None:
    """Reuse the pure same-scope binding checks without granting authority."""
    from ..backends.campaign_state import CampaignStateBackend
    from .campaign_state import TrustedExecutionBinding, _binding_fields

    if (campaign_state is None) != (execution_binding is None):
        raise ValueError("complete private campaign context required")
    if campaign_state is not None:
        if (
            not isinstance(campaign_state, CampaignStateBackend)
            or type(execution_binding) is not TrustedExecutionBinding
        ):
            raise ValueError("typed server campaign context required")
        fields = _binding_fields(execution_binding)
        if (
            fields["account_id"],
            fields["region"],
            fields["deployment_id"],
            fields["deployment_prefix"],
            fields["runtime_session_id"],
        ) != (
            catalog.account_id,
            catalog.region,
            catalog.deployment_id,
            catalog.deployment_prefix,
            runtime_session_id,
        ):
            raise ValueError("server campaign SFT scope differs")


class SessionSftStore:
    """Handles persist during one conversation; no approval or restart grants."""

    def __init__(
        self,
        model_store: Any,
        preparation_store: Any,
        adapter: Any,
        catalog: SftCatalog,
        *,
        campaign_state: Any = None,
        execution_binding: Any = None,
        reload_compatibility_reader: Any = None,
    ):
        from ..backends.ports import QwenSftBackend
        from .model_acquisition_state import SessionModelAcquisitionStore
        from .preparation_state import SessionPreparationStore

        if (
            not isinstance(model_store, SessionModelAcquisitionStore)
            or not isinstance(preparation_store, SessionPreparationStore)
            or not isinstance(adapter, QwenSftBackend)
            or type(catalog) is not SftCatalog
            or model_store.runtime_session_id != preparation_store.runtime_session_id
        ):
            raise ValueError("typed same-session SFT state required")
        _validate_campaign_binding(catalog, model_store.runtime_session_id, campaign_state, execution_binding)
        if reload_compatibility_reader is not None and not callable(reload_compatibility_reader):
            raise ValueError("private fresh compatibility reader required")
        self._reload_compatibility_reader = reload_compatibility_reader
        self._campaign_state, self._execution_binding = campaign_state, execution_binding
        self._before_submit: (
            Callable[[str, str, str, VerifiedSftInputs], DatasetArtifactRef | None] | None
        ) = None
        self._source_refs: dict[str, tuple[str, str]] = {}
        self.runtime_session_id = model_store.runtime_session_id
        self._models, self._prepared, self._adapter, self._catalog = (
            model_store,
            preparation_store,
            adapter,
            catalog,
        )
        self._records: dict[str, JobHandle] = {}
        self._keys: dict[tuple, str] = {}
        self._historical_train_keys: set[tuple] = set()
        self._lock = threading.RLock()

    def bind_durable(
        self,
        campaign_state: CampaignStateBackend,
        execution_binding: TrustedExecutionBinding,
        before_submit: Callable[[str, str, str, VerifiedSftInputs], DatasetArtifactRef | None],
    ) -> None:
        """Bind an empty original store once; the callback receives only input DATA."""
        with self._lock:
            if (
                self._campaign_state is not None
                or self._execution_binding is not None
                or self._before_submit is not None
                or self._records
                or self._keys
                or self._source_refs
                or self._historical_train_keys
            ):
                raise ValueError("original SFT store is already bound or occupied")
            if campaign_state is None or execution_binding is None or not callable(before_submit):
                raise ValueError("complete durable SFT binding and DATA callback required")
            validate_catalog(self._catalog)
            if (
                self._models.runtime_session_id != self.runtime_session_id
                or self._prepared.runtime_session_id != self.runtime_session_id
            ):
                raise ValueError("original SFT graph correlation differs")
            try:
                if str(uuid.UUID(self.runtime_session_id)) != self.runtime_session_id:
                    raise ValueError("noncanonical correlation")
            except (AttributeError, TypeError, ValueError):
                raise ValueError("normal durable SFT correlation must be a canonical UUID") from None
            _validate_campaign_binding(
                self._catalog, self.runtime_session_id, campaign_state, execution_binding
            )
            self._campaign_state = campaign_state
            self._execution_binding = execution_binding
            self._before_submit = before_submit

    def _check_original_catalog(self, catalog, job_arn, operation, compatibility=None):
        if compatibility is not None:
            fresh = select_reload_compatibility(
                self._reload_compatibility_reader, self._catalog, compatibility.checkpoint
            )
            if fresh != compatibility or catalog != self._catalog:
                raise ValueError("fresh original recovery selection differs")
            return
        if catalog == self._catalog:
            return
        if operation != "train":
            raise ValueError("historical consumer operation lacks exact selection")
        return select_historical_reload_compatibility(
            self._reload_compatibility_reader, self._catalog, catalog, job_arn, self.runtime_session_id
        )

    def _check_original_result(self, handle, result):
        compatibility = handle.extra.get("reload_compatibility")
        selected = self._check_original_catalog(
            handle.extra["catalog"], handle.arn, handle.extra["operation"], compatibility
        )
        if (
            handle.extra["catalog"] != self._catalog
            and result.get("success") is True
            and result.get("trained") is True
        ):
            observed = VerifiedSftCheckpointSource(
                self.runtime_session_id,
                handle.arn,
                handle.extra["recipe_id"],
                handle.extra["catalog"].source_commit,
                result["checkpoint_ref"],
                tuple(sorted(result["checkpoint_refs"].items())),
            )
            if observed != selected.checkpoint:
                raise ValueError("historical output differs from independently selected checkpoint")

    def _reload_observation_kwargs(self, compatibility) -> dict:
        return (
            {"_reload_compatibility_reader": self._reload_compatibility_reader}
            if compatibility is not None
            else {}
        )

    def _campaign_kwargs(self) -> dict:
        if self._campaign_state is None:
            return {}
        return {"_campaign_state": self._campaign_state, "_execution_binding": self._execution_binding}

    def _fresh_inputs(self, model_ref: str, preparation_ref: str) -> VerifiedSftInputs:
        if (
            type(model_ref) is not str
            or not re.fullmatch("model_acq_[a-f0-9]{32}", model_ref)
            or type(preparation_ref) is not str
            or not re.fullmatch("prep_[a-f0-9]{32}", preparation_ref)
        ):
            raise ValueError("original opaque SFT source selectors required")
        inputs = VerifiedSftInputs(
            self._models._verified_sft_model_snapshot(model_ref),
            self._prepared._verified_sft_snapshot(preparation_ref),
        )
        validate_inputs(inputs, self._catalog, self.runtime_session_id)
        return inputs

    def submit(self, model_ref: str, preparation_ref: str, recipe_id: str) -> dict:
        validate_catalog(self._catalog)
        if recipe_id not in ("fit_v1", "full_v1"):
            raise ValueError("source-owned SFT recipe required")
        if (
            type(model_ref) is not str
            or not re.fullmatch("model_acq_[a-f0-9]{32}", model_ref)
            or type(preparation_ref) is not str
            or not re.fullmatch("prep_[a-f0-9]{32}", preparation_ref)
        ):
            raise ValueError("opaque SFT source refs required")
        key = (model_ref, preparation_ref, recipe_id, "train")
        admitted = self._admitted_train()
        if admitted is not None:
            # A durable (normal-run) store keeps one original training submission. The same logical inputs
            # return its recorded state; other inputs are refused by name. Neither reaches the adapter,
            # the DATA callback or a second parent capture.
            return (
                self._recorded_submission(admitted[0])
                if key == admitted[1]
                else self._already_admitted(admitted[0])
            )
        with self._lock:
            if key in self._keys:
                handle = self._records[self._keys[key]]
                if handle.extra["catalog"] != self._catalog or handle.extra["operation"] != "train":
                    raise ValueError("current training dedup catalog differs")
                return self._submitted(self._keys[key], reused=True)
            if key in self._historical_train_keys:
                raise ValueError("historical observation cannot satisfy current training selection")
            inputs = self._fresh_inputs(model_ref, preparation_ref)
            campaign_kwargs = self._campaign_kwargs()
            if self._before_submit is not None:
                causal_parent_ref = self._before_submit(model_ref, preparation_ref, recipe_id, inputs)
                if causal_parent_ref is not None:
                    if type(causal_parent_ref) is not DatasetArtifactRef:
                        raise ValueError("typed durable SFT parent DATA required")
                    campaign_kwargs["_causal_parent_ref"] = causal_parent_ref
                campaign_kwargs["_source_refs"] = (model_ref, preparation_ref)
            handle = self._adapter.submit(
                inputs=inputs,
                catalog=self._catalog,
                recipe_id=recipe_id,
                runtime_session_id=self.runtime_session_id,
                **campaign_kwargs,
            )
            return self._retain(key, handle)

    def _admitted_train(self) -> tuple[str, tuple] | None:
        """(reference, source key) of the one current original training record of a durable store."""
        if self._campaign_state is None:
            return None
        with self._lock:
            for key, ref in self._keys.items():
                handle = self._records.get(ref)
                if (
                    len(key) == 4
                    and key[3] == "train"
                    and handle is not None
                    and handle.extra.get("operation") == "train"
                    and handle.extra.get("catalog") == self._catalog
                ):
                    return ref, key
        return None

    def repeat_kind(self, model_ref: Any, preparation_ref: Any, recipe_id: Any) -> str | None:
        """Classify a training call against the admitted original: same logical inputs, other inputs, or none.

        The logical inputs are the opaque source references and the recipe. The conversation prefix and the
        tool-use identifier are not part of them.
        """
        admitted = self._admitted_train()
        if admitted is None:
            return None
        return "same" if admitted[1] == (model_ref, preparation_ref, recipe_id, "train") else "different"

    def _recorded_submission(self, ref: str) -> dict:
        """The admitted original's current recorded state as the result of a repeat call."""
        observed = self.observe(ref)
        return {**self._submitted(ref, reused=True), **observed, "reused": True, "request_ref": ref}

    def _already_admitted(self, ref: str) -> dict:
        h = self._records[ref]
        return {
            "success": False,
            "accepted": False,
            "trained": False,
            "quality_success": False,
            "production_admission": False,
            "full_training_complete": False,
            "status": "sft_already_admitted",
            "operation": "train",
            "request_ref": ref,
            "recipe_id": h.extra["recipe_id"],
            "job_name": h.id,
            "job_arn": h.arn,
            "reused": False,
            "training_stage": "fit" if h.extra["recipe_id"] == "fit_v1" else "full",
            "message": (
                "This run already admitted its one training submission ("
                + ref
                + "). A submission with other inputs is not available; poll the admitted one with get_qwen_sft."
            ),
        }

    def _retain(self, key: tuple, handle: JobHandle) -> dict:
        if (
            type(handle) is not JobHandle
            or handle.backend != "qwen_sft"
            or handle.extra.get("runtime_session_id") != self.runtime_session_id
        ):
            raise ValueError("SFT native handle session differs")
        if self._campaign_state is None:
            ref = "sft_" + uuid.uuid4().hex
        else:
            digest = handle.extra.get("operation_digest")
            if type(digest) is not str or not re.fullmatch("[a-f0-9]{64}", digest):
                raise ValueError("original private SFT operation identity required")
            ref = "sft_" + digest[:32]
            existing = self._records.get(ref)
            if existing is not None and (existing.id, existing.arn) != (handle.id, handle.arn):
                raise ValueError("private SFT reference identity collision")
            if (
                existing is not None
                and existing.extra.get("submission_state") == "accepted"
                and handle.extra.get("submission_state") != "accepted"
            ):
                handle = existing
        self._records[ref] = handle
        self._keys[key] = ref
        if len(key) == 4:
            self._source_refs[ref] = (key[0], key[1])
        elif key[0] in self._source_refs:
            self._source_refs[ref] = self._source_refs[key[0]]
        return self._submitted(ref)

    def _submitted(self, ref: str, *, reused: bool = False) -> dict:
        h = self._records[ref]
        state = h.extra["submission_state"]
        result = {
            "success": state == "accepted",
            "accepted": state == "accepted",
            "request_ref": ref,
            "status": "submitted" if state == "accepted" else state,
            "operation": h.extra["operation"],
            "recipe_id": h.extra["recipe_id"],
            "job_name": h.id,
            "job_arn": h.arn,
            "reused": reused,
            "trained": False,
            "quality_success": False,
            "production_admission": False,
            "full_training_complete": False,
            "training_stage": "fit" if h.extra["recipe_id"] == "fit_v1" else "full",
        }
        if self._campaign_state is not None:
            result["durability_status"] = h.extra.get("durability_status", "unresolved")
        return result

    def observe(self, ref: str) -> dict:
        with self._lock:
            h = self._records.get(ref) if type(ref) is str and re.fullmatch("sft_[a-f0-9]{32}", ref) else None
        if h is None:
            return {
                "success": False,
                "accepted": False,
                "trained": False,
                "quality_success": False,
                "status": "unknown_reference",
            }
        try:
            compatibility = h.extra.get("reload_compatibility")
            self._check_original_catalog(h.extra["catalog"], h.arn, h.extra["operation"], compatibility)
            private = self._reload_observation_kwargs(compatibility)
            if self._campaign_state is not None:
                model_ref, preparation_ref = self._source_refs[ref]
                reservation = self._campaign_state.load_original(self._execution_binding, h.id)
                self._check_original_catalog(
                    reservation.intent.catalog,
                    reservation.native_job_arn,
                    reservation.intent.operation,
                    reservation.intent.reload_compatibility,
                )
                inputs = self._fresh_inputs(model_ref, preparation_ref)
                checked = self._adapter.restore_original(reservation, fresh_inputs=inputs, **private)
                result = self._adapter.artifacts(checked, **private)
                self._check_original_result(checked, result)
                with self._lock:
                    if self._records.get(ref) is not h:
                        raise ValueError("original SFT record changed during observation")
                    if (
                        checked.extra.get("submission_state") == "accepted"
                        or h.extra.get("submission_state") != "accepted"
                    ):
                        for name in ("submission_state", "restoration_status", "durability_status"):
                            if name in checked.extra:
                                h.extra[name] = checked.extra[name]
                return {
                    **result,
                    "request_ref": ref,
                    "durability_status": checked.extra["durability_status"],
                }
            result = self._adapter.artifacts(h, **private)
            self._check_original_result(h, result)
            return {**result, "request_ref": ref}
        except Exception as error:
            return {
                **self._submitted(ref),
                "success": False,
                "status": "observation_failed",
                "error_type": type(error).__name__,
            }

    def resume_original(self, job_name: str, model_ref: str, preparation_ref: str) -> dict:
        """Server-only read-only restoration; original upstream records must exist."""
        try:
            if self._campaign_state is None:
                raise ValueError("legacy SFT state has no durable resume")
            reservation = self._campaign_state.load_original(self._execution_binding, job_name)
            compatibility = reservation.intent.reload_compatibility
            historical = (
                reservation.intent.operation == "train" and reservation.intent.catalog != self._catalog
            )
            ordinary_key = (
                model_ref,
                preparation_ref,
                reservation.intent.recipe_id,
                reservation.intent.operation,
            )
            if historical:
                with self._lock:
                    self._historical_train_keys.add(ordinary_key)
            self._check_original_catalog(
                reservation.intent.catalog,
                reservation.native_job_arn,
                reservation.intent.operation,
                compatibility,
            )
            private = self._reload_observation_kwargs(compatibility)
            inputs = self._fresh_inputs(model_ref, preparation_ref)
            handle = self._adapter.restore_original(reservation, fresh_inputs=inputs, **private)
            key = ordinary_key
            if historical:
                key = ("historical_train", reservation.native_job_arn, *ordinary_key)
            elif compatibility is not None:
                key = (*key, reload_compatibility_digest(compatibility))
            artifacts = self._adapter.artifacts(handle, **private)
            self._check_original_result(handle, artifacts)
            with self._lock:
                result = self._retain(key, handle)
                self._source_refs[result["request_ref"]] = (model_ref, preparation_ref)
            return {**result, **artifacts, "reused": True}
        except Exception as error:
            return {
                "success": False,
                "accepted": False,
                "trained": False,
                "quality_success": False,
                "production_admission": False,
                "full_training_complete": False,
                "status": "resume_unresolved",
                "error_type": type(error).__name__,
            }

    def reload(self, ref: str) -> dict:
        validate_catalog(self._catalog)
        with self._lock:
            original = (
                self._records.get(ref) if type(ref) is str and re.fullmatch("sft_[a-f0-9]{32}", ref) else None
            )
            if original is None or original.extra["operation"] != "train":
                raise ValueError("original SFT train ref required")
            self._check_original_catalog(original.extra["catalog"], original.arn, "train")
            result = self._adapter.artifacts(original)
            if result.get("success") is not True or result.get("trained") is not True:
                raise ValueError("verified original checkpoint required")
            checkpoint = VerifiedSftCheckpointSource(
                self.runtime_session_id,
                original.arn,
                original.extra["recipe_id"],
                original.extra["catalog"].source_commit,
                result["checkpoint_ref"],
                tuple(sorted(result["checkpoint_refs"].items())),
            )
            selection = None
            if checkpoint.source_commit != self._catalog.source_commit:
                selection = select_reload_compatibility(
                    self._reload_compatibility_reader, self._catalog, checkpoint
                )
            key = (
                (ref, "reload")
                if selection is None
                else (ref, "reload", reload_compatibility_digest(selection))
            )
            if key in self._keys:
                return self._submitted(self._keys[key], reused=True)
            model = original.extra["model_source"]
            private = self._campaign_kwargs()
            private.update(self._reload_observation_kwargs(selection))
            if self._campaign_state is not None:
                refs = self._source_refs[ref]
                fresh = self._fresh_inputs(*refs)
                if fresh.model != model or fresh.prepared != original.extra["prepared_source"]:
                    raise ValueError("fresh original reload source differs")
                private["_durable_inputs"] = fresh
            handle = self._adapter.reload(
                model=model,
                checkpoint=checkpoint,
                catalog=self._catalog,
                runtime_session_id=self.runtime_session_id,
                **private,
            )
            return self._retain(key, handle)

    def _verified_generation_checkpoint(self, ref, model):
        """Fresh original full checkpoint transfer; public dictionaries grant nothing."""
        with self._lock:
            original = (
                self._records.get(ref) if type(ref) is str and re.fullmatch("sft_[a-f0-9]{32}", ref) else None
            )
        if (
            original is None
            or original.extra.get("operation") != "train"
            or original.extra.get("recipe_id") != "full_v1"
            or original.extra.get("runtime_session_id") != self.runtime_session_id
            or original.extra.get("model_source") != model
        ):
            raise ValueError("original same-model full train reference required")
        result = self._adapter.artifacts(original)
        if (
            result.get("success") is not True
            or result.get("trained") is not True
            or result.get("full_training_complete") is not True
        ):
            raise ValueError("verified original full checkpoint required")
        return VerifiedSftCheckpointSource(
            self.runtime_session_id,
            original.arn,
            original.extra["recipe_id"],
            original.extra["catalog"].source_commit,
            result["checkpoint_ref"],
            tuple(sorted(result["checkpoint_refs"].items())),
        )
