"""Parent v2 profile/byte/sidecar checks; opaque synthetic Parquet, no parser/cloud."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from test_managed_coding_preparation import (
    SESSION,
    complete_outputs,
    encoded,
    sha,
    submit,
)
from test_managed_coding_preparation import (
    fixture as fixture,
)

from amlas.control_plane.preparation_state import validate_source
from amlas.data_eng.open_swe_canonical import CARD_SHA256, DATASET_REVISION

PROFILE = "nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2"
POLICY = "coding-preparation.open-swe.v2"
LINKAGE = "derived_not_original_source_verified"
FILENAME = "data/openhands/qwen38_27b/scale-swe/train-00005-of-00006.parquet"


def parquet_fixture(fixture):
    backend, sm, s3, catalog, original = fixture
    # This is deliberately not a corpus or parseable file. Parent responsibility
    # is exact-byte staging, never row discovery from binary or a Parquet parser.
    raw = b"PAR1" + b"opaque\nbinary\n" * 7 + b"PAR1"
    ref = dataclasses.replace(
        original.inspection_source.artifact_ref,
        uri=original.inspection_source.artifact_ref.uri.replace(".jsonl", ".parquet"),
        format="parquet",
        sha256=sha(raw),
        byte_count=len(raw),
    )
    source = dataclasses.replace(
        original,
        inspection_source=dataclasses.replace(original.inspection_source, artifact_ref=ref),
        repo_id="nvidia/Open-SWE-Traces",
        revision=DATASET_REVISION,
        filename=FILENAME,
    )
    s3.add(ref.uri.split("/", 3)[3], raw, ref.version_id)
    lineage = encoded(
        {
            "sha256": ref.sha256,
            "byte_count": ref.byte_count,
            "format": "parquet",
            "repo_id": source.repo_id,
            "revision": source.revision,
            "filename": source.filename,
        }
    )
    commit = encoded({"sha256": ref.sha256, "lineage_sha256": sha(lineage)})
    receipt = encoded(
        {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "filename": source.filename,
            "image_uri": source.acquisition_image_uri,
            "source_commit": source.acquisition_source_commit,
            "commit_manifest_sha256": sha(commit),
        }
    )
    refs = []
    for name, body in (("receipt", receipt), ("commit", commit), ("lineage", lineage)):
        item = next(x for x in original.manifest_refs if x[0] == name)
        version = s3.add(item[1].split("/", 3)[3], body)
        refs.append((name, item[1], version, sha(body)))
    return backend, sm, s3, catalog, dataclasses.replace(source, manifest_refs=tuple(refs))


def cli(fixture):
    args = fixture[1].creates[0]["AppSpecification"]["ContainerArguments"]
    return {name[2:].replace("-", "_"): value for name, value in zip(args[::2], args[1::2], strict=True)}


def test_private_pinned_parquet_source_is_valid_without_widening_other_sources(fixture):
    value = parquet_fixture(fixture)
    validate_source(value[4], value[3], SESSION)
    assert value[1].creates == [] and value[2].calls == []


def test_staging_preserves_raw_parquet_and_infers_closed_profile_not_newline_count(fixture):
    value = parquet_fixture(fixture)
    handle = submit(value)
    args = cli(value)
    assert args["profile_id"] == PROFILE and args["policy_version"] == POLICY
    assert len(args) == 12 and handle.extra["source_records"] is None
    selection_key = next(key for key in value[2].values if key.endswith("/input/selection.json"))
    selection = json.loads(value[2].values[selection_key][0])
    assert selection["schema_version"] == 1
    data = selection["sources"][0]["data"]
    assert data["relative_name"] == "data/000000.parquet" and data["format"] == "parquet"
    input_prefix = selection_key.removesuffix("selection.json")
    staged = value[2].values[input_prefix + data["relative_name"]][0]
    original = value[2].values[value[4].inspection_source.artifact_ref.uri.split("/", 3)[3]][0]
    assert staged == original and sha(staged) == data["sha256"]
    assert data["original_version_id"] == value[4].inspection_source.artifact_ref.version_id
    assert len(value[1].creates) == 1


def test_parent_raw_binary_hash_never_counts_newlines(fixture):
    value = parquet_fixture(fixture)
    ref = value[4].inspection_source.artifact_ref

    class NoLineCounting(bytes):
        def count(self, *args):
            pytest.fail("parent counted binary newlines")

    class Body:
        def __init__(self):
            self.done = False
            self.closed = False

        def read(self, _n):
            if self.done:
                return b""
            self.done = True
            return NoLineCounting(value[2].values[ref.uri.split("/", 3)[3]][0])

        def close(self):
            self.closed = True

    body = Body()
    original = value[2].get_object
    value[2].get_object = lambda **kw: {**original(**kw), "Body": body}
    _raw, digest, records = value[0]._raw(
        value[2],
        value[3].bucket,
        ref.uri.split("/", 3)[3],
        ref.version_id,
        value[3].account_id,
        ref.byte_count,
        ref.sha256,
        count_lines=False,
    )
    assert digest == ref.sha256 and records is None and body.closed


@pytest.mark.parametrize(
    "change",
    [
        {"repo_id": "caller/fork"},
        {"revision": "a" * 40},
        {"filename": "data/unknown/qwen38_27b/train.parquet"},
        {"filename": "data/openhands/qwen38_27b/scale-swe/train.csv"},
    ],
)
def test_foreign_unpinned_or_unsupported_parquet_refuses_before_clients(fixture, change):
    value = parquet_fixture(fixture)
    source = dataclasses.replace(value[4], **change)
    with pytest.raises(ValueError):
        value[0].submit(sources=(source,), catalog=value[3], runtime_session_id=SESSION)
    assert value[1].creates == [] and value[2].calls == []


def test_mixed_v1_v2_selection_refuses_before_any_sdk(fixture):
    value = parquet_fixture(fixture)
    with pytest.raises(ValueError):
        value[0].submit(sources=(fixture[4], value[4]), catalog=value[3], runtime_session_id=SESSION)
    assert value[1].creates == [] and value[2].calls == []


def test_same_session_version_hash_guard_is_preserved_for_parquet(fixture):
    value = parquet_fixture(fixture)
    for changes in [{"runtime_session_id": "foreign"}, {"account_id": "9876" * 3}]:
        source = dataclasses.replace(
            value[4], inspection_source=dataclasses.replace(value[4].inspection_source, **changes)
        )
        with pytest.raises(ValueError):
            validate_source(source, value[3], SESSION)
    for changes in [{"version_id": "null"}, {"sha256": "wrong"}, {"format": "csv"}]:
        ref = dataclasses.replace(value[4].inspection_source.artifact_ref, **changes)
        source = dataclasses.replace(
            value[4], inspection_source=dataclasses.replace(value[4].inspection_source, artifact_ref=ref)
        )
        with pytest.raises(ValueError):
            validate_source(source, value[3], SESSION)


def source_provenance(value, ordinal):
    source = value[4]
    ref = source.inspection_source.artifact_ref
    return {
        "profile_id": PROFILE,
        "dataset_repo": source.repo_id,
        "revision": source.revision,
        "config_name": "v1.2",
        "filename": source.filename,
        "artifact_uri": ref.uri,
        "artifact_version_id": ref.version_id,
        "artifact_sha256": ref.sha256,
        "row_index": ordinal,
        "data_license": "CC-BY-4.0",
        "card_sha256": CARD_SHA256,
        "coverage": "qualified singleton derived linkage; no original-link verification, pinned corpus acceptance or near-dedup claim",
    }


def v2_outputs(value, handle):
    prefix, committed = complete_outputs(value, handle)
    counts = {
        "source_count": 6,
        "retained_representatives": 4,
        "primary_drop_count": 2,
        "unresolved_record_count": 0,
        "split_counts": {"train": 2, "validation": 1, "test": 1},
    }
    lineages = []
    for ordinal in range(6):
        prov = source_provenance(value, ordinal)
        ref = value[4].inspection_source.artifact_ref
        norm = {
            "profile_id": PROFILE,
            "status": "accepted" if ordinal < 5 else "rejected",
            "normalized_record_sha256": "b" * 64 if ordinal < 5 else None,
            "normalization_lineage": [],
            "source_provenance": prov,
        }
        if ordinal == 0:
            norm["normalization_lineage"] = [
                {
                    "source_row_ordinal": 0,
                    "result_message_index": 2,
                    "assistant_message_index": 1,
                    "call_index": 0,
                    "explicit_originating_call_id": "source-call",
                    "original_result_id_state": "absent",
                    "normalization_policy_version": PROFILE,
                    "linkage_status": LINKAGE,
                }
            ]
        if ordinal == 5:
            norm["reason"] = "unresolved_tool_calls"
        lineages.append(
            {
                "source_locator": ref.uri + "#version=" + ref.version_id + "#row=" + str(ordinal),
                "source_version_id": ref.version_id,
                "source_sha256": ref.sha256,
                "record_id": "c" * 63 + str(ordinal) if ordinal < 5 else None,
                "decision": "retained"
                if ordinal < 4
                else "resolution_failed"
                if ordinal == 4
                else "normalization_unresolved_tool_calls",
                "cluster_id": None,
                "representative_id": None,
                "normalization": norm,
            }
        )
    for ordinal, entry in enumerate(lineages[:5]):
        entry.update(
            repository=f"owner/repo-{ordinal}",
            instance_id=f"case-{ordinal}",
            trajectory_id=f"trajectory-{ordinal}",
            license="MIT",
        )
        if ordinal < 4:
            entry["cluster_id"] = "d" * 64
            entry["representative_id"] = entry["record_id"]
    report = {
        "schema_version": 2,
        "status": "complete",
        "counts": counts,
        "primary_drop_counts": {"resolution_failed": 1},
        "secret_pattern_counts": {},
        "coverage": committed["coverage"],
        "normalization_source_count": 6,
        "normalization_accepted_count": 5,
        "normalization_rejected_count": 1,
        "normalization_rejection_reasons": {"unresolved_tool_calls": 1},
        "derived_edge_count": 1,
        "linkage_status": LINKAGE,
    }

    def write(name, body):
        value[2].add(prefix + name, body)
        committed["files"][name] = {
            "byte_count": len(body),
            "sha256": sha(body),
            "format": "json" if name.endswith(".json") else "jsonl",
            "record_count": 1 if name.endswith(".json") else len(body.splitlines()),
        }

    write("lineage.jsonl", b"".join(encoded(x) for x in lineages))
    write("REPORT.json", encoded(report))
    committed["counts"] = counts
    value[2].add(prefix + "COMMITTED.json", encoded(committed))
    return prefix, committed, report, lineages, write


def test_v2_commit_report_original_row_conservation_and_qualified_lineage_verify(fixture):
    value = parquet_fixture(fixture)
    handle = submit(value)
    v2_outputs(value, handle)
    result = value[0].artifacts(handle)
    assert result["prepared"] is True and result["success"] is True and result["trained"] is False
    assert result["profile_id"] == PROFILE and result["policy_version"] == POLICY
    assert result["counts"]["source_count"] == 6 and result["counts"]["primary_drop_count"] == 2
    assert result["normalization"]["normalization_accepted_count"] == 5
    assert result["normalization"]["normalization_rejected_count"] == 1
    assert result["normalization"]["linkage_status"] == LINKAGE
    before = len(value[1].creates)
    value[0].artifacts(handle)
    assert len(value[1].creates) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "report_schema",
        "wrong_policy",
        "bool_count",
        "double_rejected",
        "accepted",
        "rejected_reason",
        "derived_count",
        "linkage_status",
        "duplicate_row",
        "foreign_source",
        "wrong_hash",
        "wrong_ordinal",
        "missing_normalization",
        "forged_edge",
        "rejected_digest",
        "rejected_edge",
        "unknown_reason",
        "lineage_decision",
    ],
)
def test_rehashed_v2_report_or_original_row_lineage_drift_refuses(fixture, mutation):
    value = parquet_fixture(fixture)
    handle = submit(value)
    prefix, committed, report, lineages, write = v2_outputs(value, handle)
    if mutation == "report_schema":
        report["schema_version"] = 1
    elif mutation == "wrong_policy":
        committed["policy_version"] = "coding-preparation.v1"
    elif mutation == "bool_count":
        report["normalization_accepted_count"] = True
    elif mutation == "double_rejected":
        committed["counts"]["primary_drop_count"] = 3
    elif mutation == "accepted":
        report["normalization_accepted_count"] = 4
    elif mutation == "rejected_reason":
        report["normalization_rejection_reasons"] = {"unresolved_tool_calls": 2}
    elif mutation == "derived_count":
        report["derived_edge_count"] = 2
    elif mutation == "linkage_status":
        report["linkage_status"] = "original_source_verified"
    elif mutation == "duplicate_row":
        lineages[-1] = copy.deepcopy(lineages[0])
    elif mutation == "foreign_source":
        lineages[0]["normalization"]["source_provenance"]["filename"] = "data/foreign.parquet"
    elif mutation == "wrong_hash":
        lineages[0]["normalization"]["source_provenance"]["artifact_sha256"] = "d" * 64
    elif mutation == "wrong_ordinal":
        lineages[0]["normalization"]["source_provenance"]["row_index"] = 9
    elif mutation == "missing_normalization":
        lineages[0].pop("normalization")
    elif mutation == "forged_edge":
        lineages[0]["normalization"]["normalization_lineage"][0]["linkage_status"] = (
            "original_source_verified"
        )
    elif mutation == "rejected_digest":
        lineages[-1]["normalization"]["normalized_record_sha256"] = "e" * 64
    elif mutation == "rejected_edge":
        lineages[-1]["normalization"]["normalization_lineage"] = copy.deepcopy(
            lineages[0]["normalization"]["normalization_lineage"]
        )
    elif mutation == "unknown_reason":
        lineages[-1]["normalization"]["reason"] = "caller_granted"
    elif mutation == "lineage_decision":
        lineages[0]["decision"] = "unknown"
    write("REPORT.json", encoded(report))
    write("lineage.jsonl", b"".join(encoded(x) for x in lineages))
    value[2].add(prefix + "COMMITTED.json", encoded(committed))
    result = value[0].artifacts(handle)
    assert result["success"] is False and result["prepared"] is False
    assert len(value[1].creates) == 1


def test_v1_closed_profile_and_report_remain_unchanged(fixture):
    handle = submit(fixture)
    complete_outputs(fixture, handle)
    result = fixture[0].artifacts(handle)
    assert result["prepared"] is True
    assert result["profile_id"] == "openai.explicit_conversations.v1"
    assert result["policy_version"] == "coding-preparation.v1" and "normalization" not in result
