"""Five managed envelopes preserve pure bytes; caller owns source authentication."""

from __future__ import annotations

import dataclasses
import json
import re

from .coding_case_codec import FILES, descriptor, freeze_humaneval_cases, json_bytes
from .coding_case_manifest import validate_coding_case_bundle
from .coding_case_profile import CARD_SHA256, PROFILE_ID, CodingCaseSource, source_metadata
from .inspection_schema import decode_object

LIMIT = 2 * 1024**2 + 65536
HEADER = frozenset(
    {
        "job_name",
        "runtime_session_id",
        "deployment_id",
        "deployment_prefix",
        "image_uri",
        "source_commit",
        "profile_id",
        "profile_version",
    }
)
ROLES = {
    "PROMPTS.jsonl": "public_prompt",
    "ORACLES.jsonl": "private_oracle",
    "EXCLUSIONS.json": "private_preparation_exclusions",
    "CASE_MANIFEST.json": "lineage_membership_manifest",
}


def _expected(source, header):
    from ..backends.types import DatasetArtifactRef, DatasetInspectionSource, VerifiedPreparationSource

    if (
        type(source) is not VerifiedPreparationSource
        or type(source.inspection_source) is not DatasetInspectionSource
    ):
        raise ValueError("original private case source required")
    origin = source.inspection_source
    ref = origin.artifact_ref
    if type(ref) is not DatasetArtifactRef:
        raise ValueError("exact original case artifact required")
    pin = CodingCaseSource(
        source.repo_id,
        source.revision,
        source.filename,
        ref.sha256,
        ref.byte_count,
        ref.format,
        "MIT",
        CARD_SHA256,
    )
    source_metadata(pin)
    if (
        type(header) is not dict
        or set(header) != HEADER
        or any(type(v) is not str or not v for v in header.values())
        or header["profile_id"] != PROFILE_ID
        or header["profile_version"] != "1"
        or any(
            header[k] != getattr(origin, k)
            for k in ("runtime_session_id", "deployment_id", "deployment_prefix")
        )
        or not re.fullmatch("[a-f0-9]{40}", header["source_commit"])
    ):
        raise ValueError("closed original case producer differs")
    arn_prefix = (
        "arn:aws:sagemaker:"
        + origin.region
        + ":"
        + origin.account_id
        + ":processing-job/"
        + origin.deployment_prefix
        + "-"
    )
    image_pattern = (
        re.escape(origin.account_id + ".dkr.ecr." + origin.region + ".amazonaws.com/")
        + r"[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}"
    )
    if (
        type(origin.acquisition_job_arn) is not str
        or not origin.acquisition_job_arn.startswith(arn_prefix)
        or not re.fullmatch(image_pattern, source.acquisition_image_uri)
        or not re.fullmatch(image_pattern, header["image_uri"])
        or not re.fullmatch("[a-f0-9]{40}", source.acquisition_source_commit)
    ):
        raise ValueError("original case producer namespace/image differs")
    prefix = (
        "s3://"
        + origin.bucket
        + "/sessions/"
        + origin.runtime_session_id
        + "/acquisition/"
        + origin.acquisition_job_arn.rsplit("/", 1)[-1]
        + "/"
    )
    if (
        type(ref.uri) is not str
        or not ref.uri.startswith(prefix)
        or any(part in ("", ".", "..") for part in ref.uri[len(prefix) :].split("/"))
        or any(c in ref.uri for c in ("?", "#", "\\"))
        or type(ref.version_id) is not str
        or ref.version_id in ("", "null")
        or len(ref.version_id) > 1024
        or any(ord(c) < 33 or ord(c) > 126 for c in ref.version_id)
        or type(source.manifest_refs) is not tuple
        or len(source.manifest_refs) != 3
        or any(type(x) is not tuple or len(x) != 4 for x in source.manifest_refs)
        or {x[0] for x in source.manifest_refs} != {"receipt", "commit", "lineage"}
    ):
        raise ValueError("immutable original case source differs")
    for _name, uri, version, sha in source.manifest_refs:
        if (
            type(uri) is not str
            or not uri.startswith(prefix)
            or any(part in ("", ".", "..") for part in uri[len(prefix) :].split("/"))
            or any(c in uri for c in ("?", "#", "\\"))
            or type(version) is not str
            or len(version) > 1024
            or any(ord(c) < 33 or ord(c) > 126 for c in version)
            or version in ("", "null")
            or type(sha) is not str
            or not re.fullmatch("[a-f0-9]{64}", sha)
        ):
            raise ValueError("immutable original case manifest differs")
    # Normalize immutable tuples to their canonical JSON representation; no grants.
    return pin, json.loads(json_bytes(dataclasses.asdict(source)))


def _wrap(pure, source, header):
    bundle = {name: pure[name] for name in FILES[:3]}
    bundle["CASE_MANIFEST.json"] = json_bytes(
        {
            "schema_version": 2,
            "producer": header,
            "source": source,
            "pure_manifest": decode_object(pure["CASE_MANIFEST.json"], limit=LIMIT),
        }
    )
    files = {
        name: descriptor(
            raw,
            role=ROLES[name],
            record_count=164 if name.endswith(".jsonl") else 1,
            format="jsonl" if name.endswith(".jsonl") else "json",
        )
        for name, raw in bundle.items()
    }
    bundle["COMMITTED.json"] = json_bytes(
        {
            "schema_version": 2,
            "producer": header,
            "source": source,
            "pure_committed": decode_object(pure["COMMITTED.json"], limit=LIMIT),
            "files": files,
        }
    )
    if sum(map(len, bundle.values())) > LIMIT:
        raise ValueError("complete managed case byte bound exceeded")
    return bundle


def freeze_managed_case_bundle(rows, *, source, header):
    """Data-only projection; typed metadata alone does not authenticate original bytes."""
    pin, origin = _expected(source, header)
    return _wrap(freeze_humaneval_cases(rows, source=pin), origin, dict(header))


def validate_managed_case_bundle(bundle, *, source, header):
    """Strict producer/source envelope and independent untouched pure-byte replay."""
    pin, origin = _expected(source, header)
    if (
        type(bundle) is not dict
        or set(bundle) != set(FILES)
        or any(type(raw) is not bytes or not raw for raw in bundle.values())
        or sum(map(len, bundle.values())) > LIMIT
    ):
        raise ValueError("closed bounded managed case files required")
    manifest = decode_object(bundle["CASE_MANIFEST.json"], limit=LIMIT)
    commit = decode_object(bundle["COMMITTED.json"], limit=LIMIT)
    for value, keys in (
        (manifest, {"schema_version", "producer", "source", "pure_manifest"}),
        (commit, {"schema_version", "producer", "source", "pure_committed", "files"}),
    ):
        if (
            set(value) != keys
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 2
            or json_bytes(value["producer"]) != json_bytes(header)
            or json_bytes(value["source"]) != json_bytes(origin)
        ):
            raise ValueError("managed case producer/source envelope differs")
    pure = {name: bundle[name] for name in FILES[:3]}
    pure["CASE_MANIFEST.json"] = json_bytes(manifest["pure_manifest"])
    pure["COMMITTED.json"] = json_bytes(commit["pure_committed"])
    result = validate_coding_case_bundle(pure, source=pin)
    expected = _wrap(pure, origin, dict(header))
    if any(bundle[name] != expected[name] for name in FILES):
        raise ValueError("managed case original projection differs")
    return result
