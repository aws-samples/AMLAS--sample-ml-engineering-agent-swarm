"""Exact private-source/producer envelope and untouched pure164 projections."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
from test_coding_case_codec import rows

from amlas.backends.types import DatasetArtifactRef, DatasetInspectionSource
from amlas.control_plane.preparation_state import VerifiedPreparationSource
from amlas.data_eng.coding_case_managed import freeze_managed_case_bundle, validate_managed_case_bundle
from amlas.data_eng.coding_case_profile import (
    PROFILE_ID,
    SOURCE_BYTES,
    SOURCE_FILENAME,
    SOURCE_REPO,
    SOURCE_REVISION,
    SOURCE_SHA256,
)

SESSION = "case-session"
BUCKET = "owned-case-bucket"
ACCOUNT = "111122223333"
IMAGE = ACCOUNT + ".dkr.ecr.us-west-2.amazonaws.com/cases@sha256:" + "a" * 64
HEADER = {
    "job_name": "owned-case-freeze-a",
    "runtime_session_id": SESSION,
    "deployment_id": "12345678-1234-4234-9234-111122223333",
    "deployment_prefix": "owned",
    "image_uri": IMAGE,
    "source_commit": "b" * 40,
    "profile_id": PROFILE_ID,
    "profile_version": "1",
}


def source():
    prefix = "s3://" + BUCKET + "/sessions/" + SESSION + "/acquisition/owned-data-acq-a/"
    inspection = DatasetInspectionSource(
        artifact_ref=DatasetArtifactRef(
            prefix + "data.parquet", "raw-version", SOURCE_SHA256, SOURCE_BYTES, "parquet"
        ),
        account_id=ACCOUNT,
        region="us-west-2",
        bucket=BUCKET,
        runtime_session_id=SESSION,
        deployment_id=HEADER["deployment_id"],
        deployment_prefix="owned",
        acquisition_job_arn="arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/owned-data-acq-a",
    )
    return VerifiedPreparationSource(
        inspection,
        tuple(
            (name, prefix + name + ".json", "manifest-" + name, "c" * 64)
            for name in ("receipt", "commit", "lineage")
        ),
        SOURCE_REPO,
        SOURCE_REVISION,
        SOURCE_FILENAME,
        IMAGE,
        "d" * 40,
    )


def encode(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def test_five_managed_outputs_preserve_pure_bytes_roles_and_original_versions():
    bundle = freeze_managed_case_bundle(rows(), source=source(), header=HEADER)
    assert list(bundle) == [
        "PROMPTS.jsonl",
        "ORACLES.jsonl",
        "EXCLUSIONS.json",
        "CASE_MANIFEST.json",
        "COMMITTED.json",
    ]
    manifest = json.loads(bundle["CASE_MANIFEST.json"])
    commit = json.loads(bundle["COMMITTED.json"])
    assert manifest["schema_version"] == commit["schema_version"] == 2
    assert manifest["producer"] == commit["producer"] == HEADER
    assert manifest["source"]["inspection_source"]["artifact_ref"]["version_id"] == "raw-version"
    assert manifest["pure_manifest"]["case_count"] == 164
    assert commit["pure_committed"]["schema_version"] == 1
    assert b"PRIVATE_REFERENCE" not in bundle["PROMPTS.jsonl"]
    verified = validate_managed_case_bundle(bundle, source=source(), header=HEADER)
    assert verified["case_count"] == 164 and verified["execution_status"] == "not_executed"


@pytest.mark.parametrize("target", ["CASE_MANIFEST.json", "COMMITTED.json"])
@pytest.mark.parametrize("drift", ["extra", "producer", "source_version", "schema_bool", "pure_extra"])
def test_mixed_or_rehashed_envelope_fields_refuse(target, drift):
    bundle = freeze_managed_case_bundle(rows(), source=source(), header=HEADER)
    obj = json.loads(bundle[target])
    if drift == "extra":
        obj["oracle_access"] = True
    elif drift == "producer":
        obj["producer"]["runtime_session_id"] = "foreign"
    elif drift == "source_version":
        obj["source"]["inspection_source"]["artifact_ref"]["version_id"] = "replaced"
    elif drift == "schema_bool":
        obj["schema_version"] = True
    else:
        obj["pure_manifest" if target == "CASE_MANIFEST.json" else "pure_committed"]["ready"] = True
    bundle[target] = encode(obj)
    with pytest.raises(ValueError):
        validate_managed_case_bundle(bundle, source=source(), header=HEADER)


@pytest.mark.parametrize("field", ["repo_id", "revision", "filename"])
def test_unrecognized_source_cannot_choose_another_profile(field):
    wrong = dataclasses.replace(source(), **{field: "other"})
    with pytest.raises(ValueError):
        freeze_managed_case_bundle(rows(), source=wrong, header=HEADER)


def test_equal_length_oracle_change_with_rehashed_outer_ref_is_inconsistent():
    bundle = freeze_managed_case_bundle(rows(), source=source(), header=HEADER)
    bundle["ORACLES.jsonl"] = bundle["ORACLES.jsonl"].replace(b"return 0", b"return 9", 1)
    committed = json.loads(bundle["COMMITTED.json"])
    committed["files"]["ORACLES.jsonl"]["sha256"] = hashlib.sha256(bundle["ORACLES.jsonl"]).hexdigest()
    bundle["COMMITTED.json"] = encode(committed)
    with pytest.raises(ValueError):
        validate_managed_case_bundle(bundle, source=source(), header=HEADER)


def test_public_json_or_source_subclass_is_never_a_private_snapshot():
    for value in (dataclasses.asdict(source()), None, {"ready": True}):
        with pytest.raises(ValueError):
            freeze_managed_case_bundle(rows(), source=value, header=HEADER)


@pytest.mark.parametrize(
    "drift",
    [
        "other_producer_uri",
        "foreign_job",
        "null_manifest_version",
        "traversal_manifest",
        "foreign_header_image",
        "wrong_acquisition_image",
        "wrong_acquisition_source",
    ],
)
def test_original_namespace_and_image_source_metadata_remain_closed(drift):
    value = source()
    header = dict(HEADER)
    if drift == "other_producer_uri":
        origin = value.inspection_source
        value = dataclasses.replace(
            value,
            inspection_source=dataclasses.replace(
                origin,
                artifact_ref=dataclasses.replace(
                    origin.artifact_ref, uri=origin.artifact_ref.uri.replace("owned-data-acq-a", "other-job")
                ),
            ),
        )
    elif drift == "foreign_job":
        value = dataclasses.replace(
            value,
            inspection_source=dataclasses.replace(
                value.inspection_source,
                acquisition_job_arn="arn:aws:sagemaker:us-west-2:111122223333:processing-job/foreign",
            ),
        )
    elif drift in ("null_manifest_version", "traversal_manifest"):
        entries = list(value.manifest_refs)
        name, uri, version, sha = entries[0]
        entries[0] = (
            name,
            uri.replace("receipt.json", "../receipt.json") if drift == "traversal_manifest" else uri,
            "null" if drift == "null_manifest_version" else version,
            sha,
        )
        value = dataclasses.replace(value, manifest_refs=tuple(entries))
    elif drift == "foreign_header_image":
        header["image_uri"] = "foreign@sha256:" + "a" * 64
    elif drift == "wrong_acquisition_image":
        value = dataclasses.replace(value, acquisition_image_uri="foreign")
    else:
        value = dataclasses.replace(value, acquisition_source_commit="not-a-commit")
    with pytest.raises(ValueError):
        freeze_managed_case_bundle(rows(), source=value, header=header)
