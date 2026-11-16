"""Exact prepared derived-v2 JSON corpus consumption; no parser/native/grants."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_qwen_sft_inputs import fixture as canonical_fixture

PROFILE = "nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2"
POLICY = "coding-preparation.open-swe.v2"
LINKAGE = "derived_not_original_source_verified"
REVISION = "1db674173119d71df45c22007af99e214600711a"  # pragma: allowlist secret
CARD = "8fd748230ebacb5a07b01c0299b3d99c1249567f47677c1c935163f843652341"  # pragma: allowlist secret
COVERAGE = {
    "near_dedup": "not_implemented",
    "exclusion_scope": "supplied_instance_ids_and_patch_digests_only",
}


def encoded(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)[:-1]).hexdigest()


def fixture(monkeypatch, tmp_path):
    m, base, prepared, binding = canonical_fixture(monkeypatch, tmp_path)
    committed = json.loads((prepared / "COMMITTED.json").read_bytes())
    committed["profile_id"] = PROFILE
    committed["policy_version"] = POLICY
    binding["preparation_header"]["profile_id"] = PROFILE
    binding["preparation_header"]["policy_version"] = POLICY
    counts = {
        "source_count": 5,
        "retained_representatives": 3,
        "primary_drop_count": 2,
        "unresolved_record_count": 0,
        "split_counts": {"train": 1, "validation": 1, "test": 1},
    }
    committed["counts"] = counts
    committed["coverage"] = COVERAGE
    lineages = []
    members = []
    splits = {}
    uri = "s3://fixture-bucket/raw/part.parquet"
    version = "fixture-original-version"
    raw_sha = "a" * 64
    for ordinal in range(5):
        row = {
            "messages": [
                {"role": "user", "content": f"question-{ordinal}"},
                {"role": "assistant", "content": "answer"},
            ],
            "tools": [],
        }
        if ordinal == 0:
            row = {
                "messages": [
                    {"role": "user", "content": "run tool"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "source-call",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "name": "f", "tool_call_id": "source-call", "content": "result"},
                    {"role": "assistant", "content": "answer"},
                ],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            }
        prov = {
            "profile_id": PROFILE,
            "dataset_repo": "nvidia/Open-SWE-Traces",
            "revision": REVISION,
            "config_name": "v1.2",
            "filename": "data/openhands/qwen38_27b/scale-swe/train-00005-of-00006.parquet",
            "artifact_uri": uri,
            "artifact_version_id": version,
            "artifact_sha256": raw_sha,
            "row_index": ordinal,
            "data_license": "CC-BY-4.0",
            "card_sha256": CARD,
            "coverage": "qualified singleton derived linkage; no original-link verification, pinned corpus acceptance or near-dedup claim",
        }
        norm = {
            "profile_id": PROFILE,
            "status": "accepted" if ordinal < 4 else "rejected",
            "normalized_record_sha256": digest(row) if ordinal < 4 else None,
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
        common = {
            "source_locator": uri + "#version=" + version + "#row=" + str(ordinal),
            "source_version_id": version,
            "source_sha256": raw_sha,
            "record_id": None,
            "decision": "normalization_unresolved_tool_calls"
            if ordinal == 4
            else "retained"
            if ordinal < 3
            else "resolution_failed",
            "cluster_id": None,
            "representative_id": None,
            "normalization": norm,
        }
        if ordinal == 4:
            norm["reason"] = "unresolved_tool_calls"
        else:
            common.update(
                repository=f"owner/repo-{ordinal}",
                instance_id=f"case-{ordinal}",
                trajectory_id=f"trajectory-{ordinal}",
                license="MIT",
            )
            identity = {
                key: common[key]
                for key in (
                    "source_locator",
                    "source_version_id",
                    "source_sha256",
                    "repository",
                    "instance_id",
                    "trajectory_id",
                )
            }
            common["record_id"] = digest(identity)
            if ordinal < 3:
                common["cluster_id"] = hashlib.sha256(("cluster-" + str(ordinal)).encode()).hexdigest()
                common["representative_id"] = common["record_id"]
                split = ("train", "validation", "test")[ordinal]
                splits[split] = row
                members.append(
                    {
                        "record_id": common["record_id"],
                        "group_id": hashlib.sha256(("group-" + str(ordinal)).encode()).hexdigest(),
                        "split": split,
                    }
                )
        lineages.append(common)
    report = {
        "schema_version": 2,
        "status": "complete",
        "counts": counts,
        "primary_drop_counts": {"resolution_failed": 1},
        "secret_pattern_counts": {},
        "coverage": COVERAGE,
        "normalization_source_count": 5,
        "normalization_accepted_count": 4,
        "normalization_rejected_count": 1,
        "normalization_rejection_reasons": {"unresolved_tool_calls": 1},
        "derived_edge_count": 1,
        "linkage_status": LINKAGE,
    }
    bodies = {split + ".jsonl": encoded(row) for split, row in splits.items()}
    bodies.update(
        {
            "membership.jsonl": b"".join(encoded(x) for x in members),
            "lineage.jsonl": b"".join(encoded(x) for x in lineages),
            "REPORT.json": encoded(report),
        }
    )

    def commit():
        for name, raw in bodies.items():
            (prepared / name).write_bytes(raw)
            committed["files"][name] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
                "format": "json" if name.endswith(".json") else "jsonl",
                "record_count": 1 if name.endswith(".json") else len(raw.splitlines()),
            }
        binding["split_refs"] = {
            name: copy.deepcopy(committed["files"][name + ".jsonl"])
            for name in ("train", "validation", "test")
        }
        raw = encoded(committed)
        (prepared / "COMMITTED.json").write_bytes(raw)
        binding["preparation_commit"] = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}

    commit()
    return m, base, prepared, binding, committed, report, lineages, members, bodies, commit


def test_complete_derived_v2_preserves_qualified_counts_and_provenance(monkeypatch, tmp_path):
    m, base, prepared, binding, *_ = fixture(monkeypatch, tmp_path)
    value = m.validate_staged_inputs(binding, base, prepared)
    assert value["preparation_profile_id"] == PROFILE and value["preparation_policy_version"] == POLICY
    assert value["normalization"] == {
        "source_count": 5,
        "accepted_count": 4,
        "rejected_count": 1,
        "rejection_reasons": {"unresolved_tool_calls": 1},
        "derived_edge_count": 1,
        "linkage_status": LINKAGE,
    }
    assert value["trusted"] is False and value["test_used_for_training"] is False


@pytest.mark.parametrize(
    "case",
    [
        "profile",
        "policy",
        "report_extra",
        "report_schema",
        "core_normalization_conflation",
        "reason_count",
        "original_count",
        "lineage_missing",
        "row_repeat",
        "raw_source_hash",
        "source_revision",
        "source_filename",
        "normalized_digest",
        "edge_original_claim",
        "edge_extra",
        "edge_ordinal",
        "edge_call",
        "rejected_digest",
        "rejected_edges",
        "membership_missing",
        "membership_split",
        "split_row_drift",
        "coverage_claim",
    ],
)
def test_v2_semantic_drift_refuses_after_recomputed_outer_hashes(monkeypatch, tmp_path, case):
    m, base, prepared, binding, committed, report, lineages, members, bodies, commit = fixture(
        monkeypatch, tmp_path
    )
    if case == "profile":
        committed["profile_id"] = "caller.profile"
        binding["preparation_header"]["profile_id"] = "caller.profile"
    elif case == "policy":
        committed["policy_version"] = "coding-preparation.v1"
        binding["preparation_header"]["policy_version"] = "coding-preparation.v1"
    elif case == "report_extra":
        report["approved"] = True
    elif case == "report_schema":
        report["schema_version"] = 1
    elif case == "core_normalization_conflation":
        report["primary_drop_counts"]["normalization_unresolved_tool_calls"] = 1
    elif case == "reason_count":
        report["normalization_rejection_reasons"]["unresolved_tool_calls"] = 2
    elif case == "original_count":
        committed["counts"]["source_count"] = 4
    elif case == "lineage_missing":
        lineages.pop()
    elif case == "row_repeat":
        lineages[-1]["normalization"]["source_provenance"]["row_index"] = 0
    elif case == "raw_source_hash":
        lineages[0]["normalization"]["source_provenance"]["artifact_sha256"] = "b" * 64
    elif case == "source_revision":
        lineages[0]["normalization"]["source_provenance"]["revision"] = "a" * 40
    elif case == "source_filename":
        lineages[0]["normalization"]["source_provenance"]["filename"] = "unknown.parquet"
    elif case == "normalized_digest":
        lineages[0]["normalization"]["normalized_record_sha256"] = "b" * 64
    elif case == "edge_original_claim":
        lineages[0]["normalization"]["normalization_lineage"][0]["linkage_status"] = (
            "original_source_verified"
        )
    elif case == "edge_extra":
        lineages[0]["normalization"]["normalization_lineage"][0]["approved"] = True
    elif case == "edge_ordinal":
        lineages[0]["normalization"]["normalization_lineage"][0]["source_row_ordinal"] = 1
    elif case == "edge_call":
        lineages[0]["normalization"]["normalization_lineage"][0]["explicit_originating_call_id"] = (
            "other-call"
        )
    elif case == "rejected_digest":
        lineages[-1]["normalization"]["normalized_record_sha256"] = "b" * 64
    elif case == "rejected_edges":
        lineages[-1]["normalization"]["normalization_lineage"] = copy.deepcopy(
            lineages[0]["normalization"]["normalization_lineage"]
        )
    elif case == "membership_missing":
        members.pop()
    elif case == "membership_split":
        members[0]["split"] = "validation"
    elif case == "split_row_drift":
        bodies["train.jsonl"] = encoded(
            {
                "messages": [
                    {"role": "user", "content": "changed"},
                    {"role": "assistant", "content": "answer"},
                ],
                "tools": [],
            }
        )
    else:
        report["coverage"]["near_dedup"] = "verified"
    bodies["REPORT.json"] = encoded(report)
    bodies["lineage.jsonl"] = b"".join(encoded(x) for x in lineages)
    bodies["membership.jsonl"] = b"".join(encoded(x) for x in members)
    commit()
    with pytest.raises(ValueError):
        m.validate_staged_inputs(binding, base, prepared)


@pytest.mark.parametrize(
    "case", ["artifact_conflicting_hash", "group_cross_split", "dropped_reference_object", "report_boolean"]
)
def test_reviewed_v2_internal_consistency_refuses(monkeypatch, tmp_path, case):
    m, base, prepared, binding, _committed, report, lineages, members, bodies, commit = fixture(
        monkeypatch, tmp_path
    )
    if case == "artifact_conflicting_hash":
        entry = lineages[4]
        prov = entry["normalization"]["source_provenance"]
        prov["row_index"] = 0
        prov["artifact_sha256"] = "b" * 64
        entry["source_locator"] = lineages[0]["source_locator"]
        entry["source_sha256"] = "b" * 64
    elif case == "group_cross_split":
        members[1]["group_id"] = members[0]["group_id"]
    elif case == "dropped_reference_object":
        lineages[3]["cluster_id"] = {"unknown": True}
        lineages[3]["representative_id"] = {"approved": True}
    else:
        report["counts"] = copy.deepcopy(report["counts"])
        report["counts"]["split_counts"]["train"] = True
    bodies["REPORT.json"] = encoded(report)
    bodies["lineage.jsonl"] = b"".join(encoded(x) for x in lineages)
    bodies["membership.jsonl"] = b"".join(encoded(x) for x in members)
    commit()
    with pytest.raises(ValueError):
        m.validate_staged_inputs(binding, base, prepared)
