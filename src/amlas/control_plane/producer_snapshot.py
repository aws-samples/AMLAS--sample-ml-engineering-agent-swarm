"""Closed original model/preparation persistence for read-only SFT restoration.

This server library seam authenticates execution-role scope through the existing
campaign port. It establishes no AgentCore transport/end-user identity or policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from ..backends.types import DatasetArtifactRef, DatasetInspectionSource, JobHandle, VerifiedPreparationSource
from .campaign_state import TrustedExecutionBinding, _binding_fields
from .preparation_state import PreparationCatalog, validate_catalog, validate_source

_MAX = 256 * 1024
_KIND = "amlas-private-sft-producers-v1"
_ALIAS_KIND = "amlas-private-sft-producers-v2"
_SEAL = object()
_ACTIVE: ContextVar[Any] = ContextVar("amlas_server_sft_campaign", default=None)
_BINDING_KEYS = {
    "job_name",
    "runtime_session_id",
    "deployment_id",
    "deployment_prefix",
    "image_uri",
    "source_commit",
}
_HEADER_KEYS = _BINDING_KEYS | {"profile_id", "policy_version"}


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("closed producer field inventory differs")
    return value


def _text(value, maximum=4096):
    if type(value) is not str or not 0 < len(value) <= maximum or any(ord(c) < 32 for c in value):
        raise ValueError("bounded producer text required")
    return value


def _sha(value):
    if type(value) is not str or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("exact producer SHA required")
    return value


def _number(value, maximum=64 * 1024**3):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("bounded producer count required")
    return value


def _json(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("producer encoding refused") from error
    if not 0 < len(raw) <= _MAX:
        raise ValueError("producer snapshot byte bound exceeded")
    return raw


def _parse(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX:
        raise ValueError("bounded producer snapshot bytes required")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate producer field")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite producer field")),
        )
        if _json(value) != raw:
            raise ValueError("canonical producer bytes differ")
        return value
    except (UnicodeError, RecursionError) as error:
        raise ValueError("producer JSON refused") from error


def _ref(value):
    _keys(value, ("uri", "version_id", "sha256", "byte_count", "format"))
    if value["version_id"] in ("", "null"):
        raise ValueError("exact original artifact version required")
    return DatasetArtifactRef(
        _text(value["uri"]),
        _text(value["version_id"]),
        _sha(value["sha256"]),
        _number(value["byte_count"]),
        _text(value["format"], 32),
    )


def _ref_wire(value):
    if type(value) is not DatasetArtifactRef:
        raise ValueError("typed original artifact required")
    result = {
        "uri": value.uri,
        "version_id": value.version_id,
        "sha256": value.sha256,
        "byte_count": value.byte_count,
        "format": value.format,
    }
    _ref(result)
    return result


def _catalog(value):
    _keys(
        value,
        (
            "account_id",
            "region",
            "bucket",
            "execution_role_arn",
            "image_uri",
            "source_commit",
            "admission_digest",
            "deployment_id",
            "deployment_prefix",
            "exclusions_ref",
            "instance_type",
            "volume_size_gb",
            "max_runtime_seconds",
        ),
    )
    result = PreparationCatalog(
        *(
            _text(value[k])
            for k in (
                "account_id",
                "region",
                "bucket",
                "execution_role_arn",
                "image_uri",
                "source_commit",
                "admission_digest",
                "deployment_id",
                "deployment_prefix",
            )
        ),
        _ref(value["exclusions_ref"]),
        _text(value["instance_type"]),
        _number(value["volume_size_gb"], 16384),
        _number(value["max_runtime_seconds"], 86400),
    )
    validate_catalog(result, exclusions_limit=8 * 1024**2)
    return result


def _catalog_wire(value):
    if type(value) is not PreparationCatalog:
        raise ValueError("typed original preparation catalog required")
    result = {
        key: getattr(value, key)
        for key in (
            "account_id",
            "region",
            "bucket",
            "execution_role_arn",
            "image_uri",
            "source_commit",
            "admission_digest",
            "deployment_id",
            "deployment_prefix",
            "instance_type",
            "volume_size_gb",
            "max_runtime_seconds",
        )
    }
    result["exclusions_ref"] = _ref_wire(value.exclusions_ref)
    _catalog(result)
    return result


def _acquisition_source_invariants(source):
    """Pure original acquisition namespace checks; fresh bundle bytes still replay later."""
    from urllib.parse import urlsplit

    from ..backends.adapters.dataset_acquisition import validate_source as validate_dataset_source

    bound = source.inspection_source
    validate_dataset_source(source.repo_id, source.revision, source.filename)
    matched = re.fullmatch(
        r"arn:aws:sagemaker:"
        + re.escape(bound.region)
        + ":"
        + re.escape(bound.account_id)
        + r":processing-job/("
        + re.escape(bound.deployment_prefix)
        + r"-[A-Za-z0-9-]{1,62})",
        bound.acquisition_job_arn,
    )
    if not matched or len(matched[1]) > 63:
        raise ValueError("original acquisition native identity differs")
    image = re.fullmatch(
        re.escape(bound.account_id)
        + r"\.dkr\.ecr\."
        + re.escape(bound.region)
        + r"\.amazonaws\.com(?:\.cn)?/[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}",
        source.acquisition_image_uri,
    )
    if not image or not re.fullmatch("[a-f0-9]{40}", source.acquisition_source_commit):
        raise ValueError("original acquisition image/source differs")
    prefix = (
        "s3://" + bound.bucket + "/sessions/" + bound.runtime_session_id + "/acquisition/" + matched[1] + "/"
    )

    def confined(uri, version):
        parsed = urlsplit(uri)
        relative = uri[len(prefix) :] if uri.startswith(prefix) else ""
        if (
            len(uri) > 4096
            or parsed.scheme != "s3"
            or parsed.netloc != bound.bucket
            or parsed.query
            or parsed.fragment
            or "\\" in relative
            or any(ord(c) < 32 for c in uri)
            or any(part in ("", ".", "..") for part in relative.split("/"))
        ):
            raise ValueError("original acquisition object is outside its native namespace")
        if (
            type(version) is not str
            or not version
            or version.lower() == "null"
            or len(version) > 1024
            or any(c.isspace() or ord(c) < 32 for c in version)
        ):
            raise ValueError("original acquisition exact version missing")

    confined(bound.artifact_ref.uri, bound.artifact_ref.version_id)
    refs = {row[0]: row for row in source.manifest_refs}
    if set(refs) != {"receipt", "commit", "lineage"} or len(refs) != len(source.manifest_refs):
        raise ValueError("original acquisition manifest roles differ")
    for row in refs.values():
        confined(row[1], row[2])
    if (
        refs["receipt"][1] != prefix + "ACQUIRED.json"
        or refs["commit"][1].rsplit("/", 1)[-1] != "COMMITTED.json"
    ):
        raise ValueError("original acquisition fixed manifest identity differs")
    bundle = refs["commit"][1].rsplit("/", 1)[0] + "/"
    if not bound.artifact_ref.uri.startswith(bundle) or not refs["lineage"][1].startswith(bundle):
        raise ValueError("original acquisition payload/lineage bundle differs")


def _source(value):
    _keys(
        value,
        (
            "inspection_source",
            "manifest_refs",
            "repo_id",
            "revision",
            "filename",
            "acquisition_image_uri",
            "acquisition_source_commit",
        ),
    )
    original = _keys(
        value["inspection_source"],
        (
            "artifact_ref",
            "account_id",
            "region",
            "bucket",
            "runtime_session_id",
            "deployment_id",
            "deployment_prefix",
            "acquisition_job_arn",
        ),
    )
    inspection = DatasetInspectionSource(
        _ref(original["artifact_ref"]),
        *(
            _text(original[k])
            for k in (
                "account_id",
                "region",
                "bucket",
                "runtime_session_id",
                "deployment_id",
                "deployment_prefix",
                "acquisition_job_arn",
            )
        ),
    )
    refs = value["manifest_refs"]
    if (
        type(refs) is not list
        or len(refs) != 3
        or any(type(row) is not list or len(row) != 4 for row in refs)
    ):
        raise ValueError("closed original source manifests required")
    manifests = tuple(tuple(_text(item) for item in row) for row in refs)
    if {row[0] for row in manifests} != {"receipt", "commit", "lineage"}:
        raise ValueError("original source manifest roles differ")
    for row in manifests:
        if row[2] == "null":
            raise ValueError("original manifest version missing")
        _sha(row[3])
    result = VerifiedPreparationSource(
        inspection,
        manifests,
        *(
            _text(value[k])
            for k in ("repo_id", "revision", "filename", "acquisition_image_uri", "acquisition_source_commit")
        ),
    )
    _acquisition_source_invariants(result)
    return result


def _source_wire(value):
    if (
        type(value) is not VerifiedPreparationSource
        or type(value.inspection_source) is not DatasetInspectionSource
    ):
        raise ValueError("typed original preparation source required")
    original = value.inspection_source
    result = {
        key: getattr(value, key)
        for key in ("repo_id", "revision", "filename", "acquisition_image_uri", "acquisition_source_commit")
    }
    result["manifest_refs"] = [list(row) for row in value.manifest_refs]
    result["inspection_source"] = {
        key: getattr(original, key)
        for key in (
            "account_id",
            "region",
            "bucket",
            "runtime_session_id",
            "deployment_id",
            "deployment_prefix",
            "acquisition_job_arn",
        )
    }
    result["inspection_source"]["artifact_ref"] = _ref_wire(original.artifact_ref)
    _source(result)
    return result


def _request(
    raw,
    *,
    name,
    scope,
    image,
    role,
    output_prefix,
    bucket,
    header,
    model,
    catalog_id=None,
    catalog_version=None,
    catalog=None,
    selection_sha=None,
):
    if type(raw) is not str:
        raise ValueError("closed original request JSON required")
    value = _parse(raw.encode())
    expected_keys = {
        "ProcessingJobName",
        "RoleArn",
        "AppSpecification",
        "ProcessingResources",
        "NetworkConfig",
        "ProcessingOutputConfig",
        "StoppingCondition",
        "Tags",
    } | ({"Environment"} if model else {"ProcessingInputs"})
    _keys(value, expected_keys)
    if value["ProcessingJobName"] != name or value["RoleArn"] != role:
        raise ValueError("original request identity differs")
    app = _keys(value["AppSpecification"], ("ImageUri", "ContainerEntrypoint", "ContainerArguments"))
    if app["ImageUri"] != image or app["ContainerEntrypoint"] != [
        "/usr/local/bin/amlas-processing-init",
        "model-acquisition" if model else "preparation",
    ]:
        raise ValueError("fixed producer entrypoint/image differs")
    args = app["ContainerArguments"]
    if (
        type(args) is not list
        or not 0 < len(args) <= 64
        or len(args) % 2
        or any(type(x) is not str or not 0 < len(x) <= 4096 for x in args)
    ):
        raise ValueError("bounded fixed producer arguments required")
    cli = dict(zip(args[::2], args[1::2], strict=True))
    if len(cli) * 2 != len(args):
        raise ValueError("duplicate producer argument")
    if model:
        expected = {
            "--catalog-id": catalog_id,
            "--catalog-version": catalog_version,
            **{"--" + key.replace("_", "-"): val for key, val in header.items()},
        }
        if cli != expected:
            raise ValueError("original model arguments differ")
        network = {"EnableNetworkIsolation": False}
        resources = {
            "ClusterConfig": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 80}
        }
        stopping = {"MaxRuntimeInSeconds": 7200}
        environment = value["Environment"]
        if (
            type(environment) is not dict
            or set(environment) - {"AMLAS_AGENT_REMOTE_FETCH", "AMLAS_AGENT_HTTP_ALLOW_HOSTS"}
            or environment.get("AMLAS_AGENT_REMOTE_FETCH") != "1"
        ):
            raise ValueError("closed original model environment required")
        if "AMLAS_AGENT_HTTP_ALLOW_HOSTS" in environment:
            _text(environment["AMLAS_AGENT_HTTP_ALLOW_HOSTS"])
        outputs = [
            {
                "OutputName": "model_acquisition",
                "AppManaged": False,
                "S3Output": {
                    "S3Uri": "s3://" + bucket + "/" + output_prefix,
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ]
    else:
        expected = {"--" + key.replace("_", "-"): val for key, val in header.items()}
        _keys(
            cli,
            set(expected)
            | {"--selection-sha256", "--selection-bytes", "--exclusions-sha256", "--exclusions-bytes"},
        )
        if (
            any(cli[k] != val for k, val in expected.items())
            or cli["--selection-sha256"] != selection_sha
            or cli["--exclusions-sha256"] != catalog.exclusions_ref.sha256
            or cli["--exclusions-bytes"] != str(catalog.exclusions_ref.byte_count)
            or not re.fullmatch("[1-9][0-9]{0,7}", cli["--selection-bytes"])
        ):
            raise ValueError("original preparation arguments differ")
        inputs = [
            {
                "InputName": "preparation-input",
                "S3Input": {
                    "S3Uri": "s3://"
                    + bucket
                    + "/sessions/"
                    + scope["runtime_session_id"]
                    + "/preparation/"
                    + name
                    + "/input/",
                    "LocalPath": "/opt/ml/processing/input/preparation",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            }
        ]
        if _json(value["ProcessingInputs"]) != _json(inputs):
            raise ValueError("fixed original preparation input differs")
        network = {"EnableNetworkIsolation": True, "EnableInterContainerTrafficEncryption": False}
        resources = {
            "ClusterConfig": {
                "InstanceType": catalog.instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": catalog.volume_size_gb,
            }
        }
        stopping = {"MaxRuntimeInSeconds": catalog.max_runtime_seconds}
        outputs = [
            {
                "OutputName": "preparation-output",
                "S3Output": {
                    "S3Uri": "s3://" + bucket + "/" + output_prefix,
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ]
    tags = [
        {"Key": "amlas:" + key.replace("_", "-"), "Value": scope[key]}
        for key in ("deployment_id", "deployment_prefix", "runtime_session_id")
    ]
    if any(
        _json(value[key]) != _json(expected)
        for key, expected in (
            ("NetworkConfig", network),
            ("ProcessingResources", resources),
            ("StoppingCondition", stopping),
            ("ProcessingOutputConfig", {"Outputs": outputs}),
            ("Tags", tags),
        )
    ):
        raise ValueError("fixed original producer request differs")
    return value


def _producer(value, scope, *, model):
    common = {"request_ref", "id", "backend", "arn", "request_json", "output_prefix"}
    fields = (
        {
            "catalog_id",
            "catalog_version",
            "expected_manifest_sha256",
            "bindings",
            "bucket",
            "account_id",
            "region",
        }
        if model
        else {
            "runtime_session_id",
            "catalog",
            "header",
            "selection_sha256",
            "exclusions_sha256",
            "source_records",
            "sources",
            "source_row_counts",
        }
    )
    _keys(value, common | fields)
    reference = _text(value["request_ref"])
    if not re.fullmatch(("model_acq_" if model else "prep_") + "[a-f0-9]{32}", reference):
        raise ValueError("original opaque producer alias required")
    name = _text(value["id"], 63)
    if not name.startswith(scope["deployment_prefix"] + "-") or not re.fullmatch(
        "[A-Za-z0-9][A-Za-z0-9-]{0,62}", name
    ):
        raise ValueError("original producer name differs")
    arn = "arn:aws:sagemaker:" + scope["region"] + ":" + scope["account_id"] + ":processing-job/" + name
    backend = "model_processing" if model else "processing"
    if value["arn"] != arn or value["backend"] != backend:
        raise ValueError("original producer service identity differs")
    if model:
        header = _keys(value["bindings"], _BINDING_KEYS)
        bucket = _text(value["bucket"])
        if (value["account_id"], value["region"]) != (scope["account_id"], scope["region"]):
            raise ValueError("original model scope differs")
        _sha(value["expected_manifest_sha256"])
        from ..data_eng.model_acquisition_schema import expected_digest
        from ..data_eng.model_weight_manifest import expected_model_weight_manifest

        if value["expected_manifest_sha256"] != expected_digest(
            expected_model_weight_manifest(value["catalog_id"], value["catalog_version"])
        ):
            raise ValueError("original model source catalog digest differs")
        if (value["catalog_id"], value["catalog_version"]) != ("qwen3_5_9b", "1"):
            raise ValueError("fixed source model catalog required")
        image = _text(header["image_uri"])
        role = _parse(value["request_json"].encode())["RoleArn"]
        extra = {
            key: value[key]
            for key in (
                "catalog_id",
                "catalog_version",
                "expected_manifest_sha256",
                "bindings",
                "bucket",
                "account_id",
                "region",
                "output_prefix",
            )
        }
        extra.update(
            accepted=False, submission_ambiguous=True, acquired=False, native_status="RestoreUnverified"
        )
        catalog = None
    else:
        catalog = _catalog(value["catalog"])
        bucket = catalog.bucket
        image = catalog.image_uri
        role = catalog.execution_role_arn
        if (catalog.account_id, catalog.region, catalog.deployment_id, catalog.deployment_prefix) != (
            scope["account_id"],
            scope["region"],
            scope["deployment_id"],
            scope["deployment_prefix"],
        ):
            raise ValueError("original preparation scope differs")
        header = _keys(value["header"], _HEADER_KEYS)
        if header["image_uri"] != catalog.image_uri or header["source_commit"] != catalog.source_commit:
            raise ValueError("original preparation header/catalog image or source differs")
        if type(value["sources"]) is not list or not 1 <= len(value["sources"]) <= 16:
            raise ValueError("bounded original source population required")
        sources = tuple(_source(row) for row in value["sources"])
        for source in sources:
            validate_source(source, catalog, scope["runtime_session_id"])
        from ..backends.adapters.coding_preparation import _preparation_pair

        profile, version = _preparation_pair(sources)
        if (header["profile_id"], header["policy_version"]) != (profile, version):
            raise ValueError("original preparation source profile differs")
        if value["source_records"] is not None:
            _number(value["source_records"], 10000000)
        counts = value["source_row_counts"]
        if type(counts) is not dict or len(counts) > 16:
            raise ValueError("closed original source counts required")
        for key, count in counts.items():
            _text(key)
            _number(count, 10000000)
        if profile == "bigcode.commitpackft.python.change_sft.v1":
            from ..data_eng.commitpack_preparation import commitpack_source_key

            expected_keys = {
                commitpack_source_key(
                    source.inspection_source.artifact_ref.uri,
                    source.inspection_source.artifact_ref.version_id,
                    source.inspection_source.artifact_ref.sha256,
                )
                for source in sources
            }
            if (
                set(counts) != expected_keys
                or type(value["source_records"]) is not int
                or sum(counts.values()) != value["source_records"]
            ):
                raise ValueError("original CommitPack source counts differ")
        _sha(value["selection_sha256"])
        _sha(value["exclusions_sha256"])
        if (
            value["exclusions_sha256"] != catalog.exclusions_ref.sha256
            or value["runtime_session_id"] != scope["runtime_session_id"]
        ):
            raise ValueError("original preparation binding differs")
        extra = {
            key: value[key]
            for key in (
                "runtime_session_id",
                "header",
                "output_prefix",
                "selection_sha256",
                "exclusions_sha256",
                "source_records",
            )
        }
        extra.update(catalog=catalog, sources=sources)
        if counts:
            extra["source_row_counts"] = counts
    if (
        any(type(item) is not str or not 0 < len(item) <= 4096 for item in header.values())
        or any(
            header[key] != scope[key] for key in ("runtime_session_id", "deployment_id", "deployment_prefix")
        )
        or header["job_name"] != name
    ):
        raise ValueError("original producer header differs")
    if not re.fullmatch("[a-f0-9]{40}", header["source_commit"]) or not re.fullmatch(
        re.escape(scope["account_id"])
        + r"\.dkr\.ecr\."
        + re.escape(scope["region"])
        + r"\.amazonaws\.com/[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}",
        image,
    ):
        raise ValueError("original producer image/source differs")
    if not re.fullmatch("arn:aws:iam::" + scope["account_id"] + r":role/[A-Za-z0-9_+=,.@/-]{1,512}", role):
        raise ValueError("original producer role differs")
    expected_prefix = (
        "sessions/"
        + scope["runtime_session_id"]
        + ("/model-acquisition/" if model else "/preparation/")
        + name
        + ("/" if model else "/output/")
    )
    if value["output_prefix"] != expected_prefix:
        raise ValueError("original producer output prefix differs")
    extra["request"] = _request(
        value["request_json"],
        name=name,
        scope=scope,
        image=image,
        role=role,
        output_prefix=expected_prefix,
        bucket=bucket,
        header=header,
        model=model,
        catalog_id=value.get("catalog_id"),
        catalog_version=value.get("catalog_version"),
        catalog=catalog,
        selection_sha=value.get("selection_sha256"),
    )
    return JobHandle(name, backend, arn, extra)


def _capture(handle, reference, *, model):
    if type(handle) is not JobHandle:
        raise ValueError("typed original producer handle required")
    extra = handle.extra
    result = {
        "request_ref": reference,
        "id": handle.id,
        "backend": handle.backend,
        "arn": handle.arn,
        "request_json": _json(extra["request"]).decode(),
        "output_prefix": extra["output_prefix"],
    }
    if model:
        result.update(
            {
                key: extra[key]
                for key in (
                    "catalog_id",
                    "catalog_version",
                    "expected_manifest_sha256",
                    "bindings",
                    "bucket",
                    "account_id",
                    "region",
                )
            }
        )
    else:
        result.update(
            {
                key: extra[key]
                for key in (
                    "runtime_session_id",
                    "header",
                    "selection_sha256",
                    "exclusions_sha256",
                    "source_records",
                )
            }
        )
        result.update(
            catalog=_catalog_wire(extra["catalog"]),
            sources=[_source_wire(source) for source in extra["sources"]],
            source_row_counts=extra.get("source_row_counts", {}),
        )
    return result


@dataclass(frozen=True, init=False)
class SftCampaignContext:
    campaign_state: Any = field(repr=False)
    execution_binding: TrustedExecutionBinding = field(repr=False)
    snapshot: bytes = field(repr=False)
    original_job_name: str
    original_sft_ref: str
    original_model_ref: str
    original_preparation_ref: str
    _seal: object = field(repr=False, compare=False)
    _pid: int = field(repr=False, compare=False)

    def __init__(self, campaign_state, binding, raw, record, *, _token=None):
        if _token is not _SEAL:
            raise ValueError("server-only producer context required")
        value = _decode(raw, binding, record)
        for key, item in {
            "campaign_state": campaign_state,
            "execution_binding": binding,
            "snapshot": raw,
            "original_job_name": record.native_job_name,
            "original_sft_ref": value["sft_ref"],
            "original_model_ref": value["model"]["request_ref"],
            "original_preparation_ref": value["preparation"]["request_ref"],
            "_seal": _SEAL,
            "_pid": os.getpid(),
        }.items():
            object.__setattr__(self, key, item)


def _decode(raw, binding, record):
    scope = _binding_fields(binding)
    value = _parse(raw)
    version = value.get("schema_version") if type(value) is dict else None
    fields = {
        "schema_version",
        "binding",
        "operation_digest",
        "job_name",
        "job_arn",
        "sft_ref",
        "model",
        "preparation",
    }
    if type(version) is not int or version not in (1, 2):
        raise ValueError("original producer schema differs")
    if version == 2:
        fields.update(("preparation_selection", "training_parent"))
    _keys(value, fields)
    if (
        value["binding"] != scope
        or value["operation_digest"] != record.operation_digest
        or value["job_name"] != record.native_job_name
        or value["job_arn"] != record.native_job_arn
        or value["sft_ref"] != "sft_" + record.operation_digest[:32]
    ):
        raise ValueError("original producer manifest scope/operation differs")
    model = _producer(value["model"], scope, model=True)
    prepared = _producer(value["preparation"], scope, model=False)
    original_model, original_prepared = record.intent.inputs.model, record.intent.inputs.prepared
    if (
        model.arn != original_model.producer_job_arn
        or (model.extra["catalog_id"], model.extra["catalog_version"])
        != (original_model.catalog_id, original_model.catalog_version)
        or tuple(sorted(model.extra["bindings"].items())) != original_model.producer_bindings
        or prepared.arn != original_prepared.producer_job_arn
        or tuple(sorted(prepared.extra["header"].items())) != original_prepared.header
        or prepared.extra["catalog"].exclusions_ref != original_prepared.exclusions_ref
        or tuple(sorted(prepared.extra.get("source_row_counts", {}).items()))
        != original_prepared.source_row_counts
        or model.extra["bucket"] != record.intent.catalog.bucket
        or prepared.extra["catalog"].bucket != record.intent.catalog.bucket
    ):
        raise ValueError("producer identities/bindings differ from reserved original SFT inputs")
    if version == 2:
        _selection(value["preparation_selection"], len(prepared.extra["sources"]))
        _parent(value["training_parent"], binding, record)
    return value


def _selection(value, source_count):
    _keys(value, {"acquisitions", "case_request_ref"})
    acquisitions = value["acquisitions"]
    if (
        value["case_request_ref"] is not None
        or type(acquisitions) is not list
        or not 1 <= len(acquisitions) <= 256
        or len(acquisitions) != source_count
    ):
        raise ValueError("original non-case preparation selection required")
    aliases = []
    for index, row in enumerate(acquisitions):
        _keys(row, {"request_ref", "source_index"})
        if (
            type(row["request_ref"]) is not str
            or not re.fullmatch("acq_[a-f0-9]{32}", row["request_ref"])
            or type(row["source_index"]) is not int
            or row["source_index"] != index
        ):
            raise ValueError("original acquisition alias/source index differs")
        aliases.append(row["request_ref"])
    if aliases != sorted(set(aliases)):
        raise ValueError("original acquisition aliases must be sorted and unique")
    return tuple(aliases)


def _current_recovery_selection(record, reader, *, issuer_pid=None):
    """Fresh configuration evidence only; stored selection and row hashes grant nothing."""
    from .sft_state import select_reload_compatibility

    selected = record.intent.reload_compatibility
    if selected is None:
        return None
    if issuer_pid is not None and issuer_pid != os.getpid():
        raise ValueError("inherited producer recovery repository refused")
    try:
        current = select_reload_compatibility(reader, record.intent.catalog, record.intent.checkpoint)
    except Exception:
        raise ValueError("fresh current producer recovery selection absent") from None
    if current != selected:
        raise ValueError("fresh current producer recovery selection differs")
    return current


def _parent(value, binding, record):
    from .campaign_state import _native_identity, build_sft_intent, operation_digest

    if record.intent.operation == "train":
        if value is not None:
            raise ValueError("train projection cannot select a parent")
        return
    if record.intent.operation != "reload":
        raise ValueError("original SFT operation differs")
    _keys(value, {"request_ref", "native_job_name", "operation_digest"})
    catalog = record.intent.catalog
    if record.intent.reload_compatibility is not None:
        from .sft_state import validate_reload_compatibility

        validate_reload_compatibility(record.intent.reload_compatibility, catalog, record.intent.checkpoint)
        # This deterministic linkage uses recorded data, never action authority.
        catalog = record.intent.reload_compatibility.producer_catalog
    intent = build_sft_intent(binding, catalog, record.intent.inputs, recipe_id=record.intent.recipe_id)
    digest = operation_digest(intent)
    name, arn = _native_identity(intent)
    if (
        value["request_ref"] != "sft_" + digest[:32]
        or value["native_job_name"] != name
        or value["operation_digest"] != digest
        or record.intent.checkpoint.producer_job_arn != arn
    ):
        raise ValueError("original reload training-parent linkage differs")


def _single_attempt_client(client):
    """Real injected SDK clients must enforce the same one-wire-attempt contract."""
    from botocore.client import BaseClient

    if isinstance(client, BaseClient):
        retry = client.meta.config.retries
        if (
            type(retry) is not dict
            or type(retry.get("total_max_attempts")) is not int
            or retry["total_max_attempts"] != 1
        ):
            raise ValueError("producer publication requires effective single-attempt SDK configuration")


class ProducerSnapshotRepository:
    """One immutable original-operation row in the already selected jobs table."""

    def __init__(
        self, campaign_state, execution_binding, *, dynamodb_client=None, reload_compatibility_reader=None
    ):
        from ..backends.campaign_state import CampaignStateBackend

        if (
            not isinstance(campaign_state, CampaignStateBackend)
            or type(execution_binding) is not TrustedExecutionBinding
        ):
            raise ValueError("current typed server campaign port/binding required")
        if reload_compatibility_reader is not None and not callable(reload_compatibility_reader):
            raise ValueError("private producer recovery reader required")
        self._reload_compatibility_reader = reload_compatibility_reader
        self._pid = os.getpid()
        self.campaign_state, self.binding, self._ddb = campaign_state, execution_binding, dynamodb_client
        _binding_fields(execution_binding)

    def _client(self):
        if self._ddb is None:
            import boto3
            from botocore.config import Config

            self._ddb = boto3.client(
                "dynamodb",
                region_name=self.binding.scope.region,
                config=Config(retries={"total_max_attempts": 1}),
            )
        return self._ddb

    def _key(self, record, *, schema_version=1):
        segment = "-sft-producers-" if schema_version == 1 else "-sft-producers-v2-"
        return self.binding.scope.deployment_prefix + segment + record.operation_digest

    def _read(self, record, *, schema_version=1):
        result = self._client().get_item(
            TableName=self.binding.scope.jobs_table,
            Key={"job_name": {"S": self._key(record, schema_version=schema_version)}},
            ConsistentRead=True,
        )
        item = result.get("Item")
        if item is None:
            return None
        _keys(
            item,
            (
                "job_name",
                "runtime_session_id",
                "record_kind",
                "operation_digest",
                "snapshot_sha256",
                "snapshot",
            ),
        )
        if (
            item["job_name"] != {"S": self._key(record, schema_version=schema_version)}
            or item["runtime_session_id"] != {"S": self.binding.runtime_session_id}
            or item["record_kind"] != {"S": _KIND if schema_version == 1 else _ALIAS_KIND}
            or item["operation_digest"] != {"S": record.operation_digest}
            or type(item["snapshot"]) is not dict
            or set(item["snapshot"]) != {"B"}
            or type(item["snapshot_sha256"]) is not dict
            or set(item["snapshot_sha256"]) != {"S"}
        ):
            raise ValueError("original producer row envelope differs")
        raw = item["snapshot"]["B"]
        if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != _sha(item["snapshot_sha256"]["S"]):
            raise ValueError("original producer snapshot bytes differ")
        if _decode(raw, self.binding, record)["schema_version"] != schema_version:
            raise ValueError("versioned original producer row differs")
        return raw

    def publish_from_store(self, store, sft_ref):
        """Publish diagnostic schema1 data; selected recovery cannot load from it."""
        return self._publish(store, sft_ref, schema_version=1)

    def publish_restore_aliases_from_store(self, store, sft_ref):
        """Publish exact original non-case selection and reload-parent provenance."""
        return self._publish(store, sft_ref, schema_version=2)

    def _publish(self, store, sft_ref, *, schema_version):
        from .sft_state import SessionSftStore

        if (
            type(store) is not SessionSftStore
            or store._campaign_state is not self.campaign_state
            or _binding_fields(store._execution_binding) != _binding_fields(self.binding)
        ):
            raise ValueError("original scoped SFT store required")
        with store._lock:
            handle = store._records.get(sft_ref)
            refs = store._source_refs.get(sft_ref)
        if type(handle) is not JobHandle or refs is None:
            raise ValueError("original SFT ref/dependencies missing")
        record = self.campaign_state.load_original(self.binding, handle.id)
        recovery_selection = _current_recovery_selection(
            record, store._reload_compatibility_reader, issuer_pid=self._pid
        )
        if recovery_selection is not None and store._catalog != record.intent.catalog:
            raise ValueError("current producer recovery store catalog differs")
        model = store._models._original_producer_handle(refs[0], self.binding)
        prep = store._prepared._original_producer_handle(refs[1], self.binding)
        value = {
            "schema_version": schema_version,
            "binding": _binding_fields(self.binding),
            "operation_digest": record.operation_digest,
            "job_name": record.native_job_name,
            "job_arn": record.native_job_arn,
            "sft_ref": sft_ref,
            "model": _capture(model, refs[0], model=True),
            "preparation": _capture(prep, refs[1], model=False),
        }
        if schema_version == 2:
            with store._prepared._lock:
                keys = [key for key, alias in store._prepared._submissions.items() if alias == refs[1]]
            if len(keys) != 1:
                raise ValueError("one original preparation selection key required")
            key = keys[0]
            if type(key) is not tuple:
                raise ValueError("original non-case preparation key required")
            value["preparation_selection"] = {
                "acquisitions": [
                    {"request_ref": alias, "source_index": index} for index, alias in enumerate(key)
                ],
                "case_request_ref": None,
            }
            value["training_parent"] = None
            if record.intent.operation == "reload":
                from .sft_state import reload_compatibility_digest

                key_length = 2 if recovery_selection is None else 3
                digest = (
                    None if recovery_selection is None else reload_compatibility_digest(recovery_selection)
                )
                with store._lock:
                    parents = [
                        key[0]
                        for key, alias in store._keys.items()
                        if type(key) is tuple
                        and len(key) == key_length
                        and key[1] == "reload"
                        and alias == sft_ref
                        and (recovery_selection is None or key[2] == digest)
                    ]
                    if len(parents) != 1:
                        raise ValueError("one original train-to-reload alias key required")
                    parent_ref = parents[0]
                    parent_handle = store._records.get(parent_ref)
                    parent_sources = store._source_refs.get(parent_ref)
                if type(parent_handle) is not JobHandle or parent_sources != refs:
                    raise ValueError("original same-store train handle required")
                parent_record = self.campaign_state.load_original(self.binding, parent_handle.id)
                value["training_parent"] = {
                    "request_ref": parent_ref,
                    "native_job_name": parent_record.native_job_name,
                    "operation_digest": parent_record.operation_digest,
                }
                if (
                    parent_handle.backend != "qwen_sft"
                    or parent_handle.arn != parent_record.native_job_arn
                    or parent_handle.extra.get("operation_digest") != parent_record.operation_digest
                    or parent_record.intent.operation != "train"
                ):
                    raise ValueError("original native training-parent handle differs")
                _parent(value["training_parent"], self.binding, record)
        raw = _json(value)
        _decode(raw, self.binding, record)
        existing = self._read(record, schema_version=schema_version)
        if existing is None:
            client = self._client()
            _single_attempt_client(client)
            try:
                _current_recovery_selection(record, store._reload_compatibility_reader, issuer_pid=self._pid)
                client.put_item(
                    TableName=self.binding.scope.jobs_table,
                    Item={
                        "job_name": {"S": self._key(record, schema_version=schema_version)},
                        "runtime_session_id": {"S": self.binding.runtime_session_id},
                        "record_kind": {"S": _KIND if schema_version == 1 else _ALIAS_KIND},
                        "operation_digest": {"S": record.operation_digest},
                        "snapshot_sha256": {"S": hashlib.sha256(raw).hexdigest()},
                        "snapshot": {"B": raw},
                    },
                    ConditionExpression="attribute_not_exists(job_name)",
                )
            except Exception as error:
                existing = self._read(record, schema_version=schema_version)
                if existing != raw:
                    raise ValueError("producer publication unresolved; original row retained") from error
        elif existing != raw:
            raise ValueError("original producer aliases or bytes changed")
        _current_recovery_selection(record, store._reload_compatibility_reader, issuer_pid=self._pid)
        return {
            "published": True,
            "record_key": self._key(record, schema_version=schema_version),
            "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        }

    def load_original(self, job_name, *, expected_snapshot_sha256=None):
        """Schema2 needs the original publisher SHA from a qualified server handoff.

        The caller qualifies that independent origin; a row hash is not a grant
        or a substitute. Schema1 remains descriptive when no commitment is given.
        """
        expected = None if expected_snapshot_sha256 is None else _sha(expected_snapshot_sha256)
        record = self.campaign_state.load_original(self.binding, job_name)
        _current_recovery_selection(record, self._reload_compatibility_reader, issuer_pid=self._pid)
        raw = self._read(record, schema_version=2)
        if raw is None:
            if record.intent.reload_compatibility is not None:
                raise ValueError("selected recovery original alias projection missing")
            if expected is not None:
                raise ValueError("committed original alias projection missing")
            raw = self._read(record)
        elif expected is None or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("independent original producer commitment missing or differs")
        if raw is None:
            raise ValueError("original producer snapshot missing")
        _current_recovery_selection(record, self._reload_compatibility_reader, issuer_pid=self._pid)
        context = SftCampaignContext(self.campaign_state, self.binding, raw, record, _token=_SEAL)
        _current_recovery_selection(record, self._reload_compatibility_reader, issuer_pid=self._pid)
        return context


def current_sft_campaign():
    context = _ACTIVE.get()
    if context is not None and context._pid != os.getpid():
        raise ValueError("inherited fork producer activation is unsupported")
    return context


@contextmanager
def activate_sft_campaign(context):
    if type(context) is not SftCampaignContext or context._seal is not _SEAL or context._pid != os.getpid():
        raise ValueError("typed server producer context required")
    token = _ACTIVE.set(context)
    try:
        yield context
    finally:
        _ACTIVE.reset(token)


def _approved_read(policy):
    from dataclasses import replace

    from .approvals import ApprovalPolicy, request_approval

    if not isinstance(policy, ApprovalPolicy):
        raise ValueError("fresh caller policy required for producer restoration")
    current = replace(policy, allowlist=list(policy.allowlist))
    decision = request_approval(
        action="qwen_sft_flow",
        cost_estimate=None,
        impact="read-only original SFT producer restoration",
        policy=current,
    )
    if decision["decision"] != "approved":
        raise ValueError("fresh caller policy denied producer restoration")


class _ApprovedReads:
    """Ephemeral per-read policy guard, removed before the verified graph is exposed."""

    def __init__(self, client, policy, denial=None):
        self.client, self.policy = client, policy
        self.denial = [False] if denial is None else denial

    @property
    def meta(self):
        return self.client.meta

    def __getattr__(self, name):
        if name not in {
            "describe_processing_job",
            "describe_training_job",
            "list_tags",
            "head_object",
            "get_object",
            "list_objects_v2",
            "get_bucket_location",
            "get_bucket_versioning",
        }:
            raise ValueError("temporary producer restoration permits read-only SDK calls")

        def call(*args, **kwargs):
            try:
                _approved_read(self.policy)
            except (TypeError, ValueError):
                self.denial[0] = True
                raise
            return getattr(self.client, name)(*args, **kwargs)

        return call


def hydrate_sft_campaign(context, *, policy=None, env=None):
    """Build/verify a temporary graph completely; publish no partially valid store."""
    if type(context) is not SftCampaignContext or context._seal is not _SEAL or context._pid != os.getpid():
        raise ValueError("typed server producer context required")
    _approved_read(policy)
    from ..backends.ports import ModelWeightAcquisitionBackend
    from ..backends.registry import REGISTRY
    from ..config.externalized import model_weight_acquisition_selectors, resolve_env_snapshot
    from .acquisition_state import SessionAcquisitionStore
    from .model_acquisition_state import SessionModelAcquisitionStore
    from .preparation_binding import build_preparation_store
    from .sft_binding import build_sft_store

    record = context.campaign_state.load_original(context.execution_binding, context.original_job_name)
    value = _decode(context.snapshot, context.execution_binding, record)
    scope = _binding_fields(context.execution_binding)
    acquired = SessionAcquisitionStore._for_original_correlation(context.execution_binding)
    models = SessionModelAcquisitionStore(runtime_session_id=scope["runtime_session_id"])
    prepared = build_preparation_store(acquired, env=env)
    model_handle = _producer(value["model"], scope, model=True)
    prep_handle = _producer(value["preparation"], scope, model=False)
    selectors = model_weight_acquisition_selectors(env=env)
    bindings = model_handle.extra["bindings"]
    if (
        selectors.get("image_uri") != bindings["image_uri"]
        or selectors.get("source_commit") != bindings["source_commit"]
        or selectors.get("admission_digest") != bindings["image_uri"].rsplit(":", 1)[1]
        or resolve_env_snapshot("session_bucket", env=env) != model_handle.extra["bucket"]
        or resolve_env_snapshot("sagemaker_execution_role_arn", env=env)
        != model_handle.extra["request"]["RoleArn"]
    ):
        raise ValueError("current model startup catalog differs")
    original = prep_handle.extra["catalog"]
    current = prepared._catalog
    if original != current:
        raise ValueError("current preparation startup catalog differs")
    model_adapter = REGISTRY.resolve(ModelWeightAcquisitionBackend).adapter
    models._restore_original_producer(
        value["model"]["request_ref"], model_adapter, model_handle, context.execution_binding
    )
    prepared._restore_original_producer(
        value["preparation"]["request_ref"], prep_handle, context.execution_binding
    )
    sft = build_sft_store(
        models,
        prepared,
        env=env,
        campaign_state=context.campaign_state,
        execution_binding=context.execution_binding,
    )
    recovery_selection = _current_recovery_selection(record, sft._reload_compatibility_reader)
    if recovery_selection is not None and value["schema_version"] != 2:
        raise ValueError("selected recovery requires committed original alias projection")
    adapters = (model_adapter, prepared._adapter, sft._adapter)
    originals = []
    denial = [False]
    try:
        # These are fresh temporary native adapters, never a shared or persisted client.
        for adapter in adapters:
            returned = adapter._clients(scope["region"])
            if (
                type(returned) is not tuple
                or len(returned) != 2
                or any(client is None for client in returned)
            ):
                raise ValueError("fresh producer SDK client pair missing")
            for name, client in zip(("_sm", "_s3"), returned, strict=True):
                # Preserve each factory's caching lifecycle after removing guards.
                originals.append((adapter, name, getattr(adapter, name)))
                setattr(adapter, name, _ApprovedReads(client, policy, denial))

        def resume(name, reference):
            resumed = sft.resume_original(name, context.original_model_ref, context.original_preparation_ref)
            _approved_read(policy)
            handle = sft._records.get(reference)
            verified = type(handle) is JobHandle and handle.extra.get("restoration_status") == "verified"
            honest_pending_or_terminal = resumed.get("status") in {"RUNNING", "FAILED", "STOPPED"}
            verified_completed = resumed.get("success") is True and resumed.get("status") in {
                "checkpoint_saved",
                "reloaded",
            }
            if (
                denial[0]
                or not verified
                or not (honest_pending_or_terminal or verified_completed)
                or resumed.get("request_ref") != reference
                or resumed.get("job_name") != name
            ):
                raise ValueError("original cold SFT verification is unresolved")
            return resumed

        parent = value.get("training_parent")
        if value["schema_version"] == 2 and parent is not None:
            _approved_read(policy)
            parent_record = context.campaign_state.load_original(
                context.execution_binding, parent["native_job_name"]
            )
            _approved_read(policy)
            if (
                parent_record.operation_digest != parent["operation_digest"]
                or parent_record.intent.operation != "train"
                or parent_record.intent.catalog
                != (
                    record.intent.catalog
                    if recovery_selection is None
                    else recovery_selection.producer_catalog
                )
                or parent_record.intent.inputs != record.intent.inputs
                or parent_record.native_job_arn != record.intent.checkpoint.producer_job_arn
            ):
                raise ValueError("fresh original training-parent reservation differs")
            observed = resume(parent["native_job_name"], parent["request_ref"])
            from .sft_state import VerifiedSftCheckpointSource

            verified_checkpoint = VerifiedSftCheckpointSource(
                scope["runtime_session_id"],
                parent_record.native_job_arn,
                parent_record.intent.recipe_id,
                parent_record.intent.catalog.source_commit,
                observed.get("checkpoint_ref"),
                tuple(sorted(observed.get("checkpoint_refs", {}).items())),
            )
            if observed.get("trained") is not True or verified_checkpoint != record.intent.checkpoint:
                raise ValueError("fresh original training-parent checkpoint differs")
        resume(context.original_job_name, context.original_sft_ref)
        if value["schema_version"] == 2:
            selection = _selection(value["preparation_selection"], len(prep_handle.extra["sources"]))
            _approved_read(policy)
            with prepared._lock:
                if prepared._submissions:
                    raise ValueError("alias restore requires empty temporary selection maps")
                prepared._submissions[selection] = context.original_preparation_ref
            if parent is not None:
                with sft._lock:
                    key = (parent["request_ref"], "reload")
                    if recovery_selection is not None:
                        from .sft_state import reload_compatibility_digest

                        key = (*key, reload_compatibility_digest(recovery_selection))
                    if key in sft._keys:
                        raise ValueError("original reload alias key was already populated")
                    sft._keys[key] = context.original_sft_ref
            _approved_read(policy)
    finally:
        for adapter, name, client in originals:
            setattr(adapter, name, client)
    _approved_read(policy)
    _current_recovery_selection(record, sft._reload_compatibility_reader)
    return acquired, models, prepared, sft
