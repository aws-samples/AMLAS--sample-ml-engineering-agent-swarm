"""Pure complete-origin CommitPack v3 replay; synthetic rows, no corpus/model calls."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import pytest
from test_commitpack_profile import row, source

from amlas.data_eng import commitpack_preparation as prep


def exclusions(**changes):
    result = {
        "schema_version": 1,
        "repositories": [],
        "commits": [],
        "canonical_sha256": [],
        "input_sha256": [],
        "code_pair_sha256": [],
    }
    result.update(changes)
    return result


def records():
    values = [
        prep.CommitPackParsedRow(row(i + 1), source(i), hashlib.sha256(f"raw-{i}\n".encode()).hexdigest())
        for i in range(100)
    ]
    values.append(
        prep.CommitPackParsedRow(
            row(1, commit="f" * 40, repos="team/repo-1,other/shared", license="apache-2.0"),
            source(100),
            "a" * 64,
        )
    )
    values.append(prep.CommitPackParsedRow(row(999, license="unknown"), source(101), "b" * 64))
    return values


def build(policy=None):
    return prep.build_commitpack_preparation(records(), exclusion_policy=policy or exclusions())


def replay(result):
    return prep.replay_commitpack_preparation(
        lineage_records=result["lineage"],
        membership_records=result["membership"],
        split_records=result["splits"],
        report=result["report"],
    )


def test_complete_origin_conservation_and_no_execution_claims():
    result = build()
    assert result["status"] == "complete"
    counts = result["report"]["counts"]
    assert counts["source"] == 102 and counts["normalized"] == 101
    assert counts["normalization_rejected"] == counts["duplicate_origins"] == 1
    assert counts["retained_representatives"] == 100
    assert len(result["lineage"]) == 102 and len(result["membership"]) == 100
    assert all(result["splits"].values())
    verified = replay(result)
    assert verified["counts"] == counts and verified["coverage"] == prep.COVERAGE
    normalized = [entry for entry in result["lineage"] if entry["origin_record"] is not None]
    assert all(
        set(entry["origin_record"])
        == {
            "source_provenance",
            "source_fields",
            "commit",
            "license",
            "repositories",
            "canonical_sha256",
            "input_sha256",
            "code_pair_sha256",
            "transformations",
        }
        for entry in normalized
    )
    for values in result["splits"].values():
        assert all(set(value) == {"messages", "tools"} for value in values)
    rejected = result["lineage"][-1]
    assert rejected["decision"] == "normalization_rejected" and rejected["origin_record"] is None
    assert rejected["raw_line_sha256"] == "b" * 64
    assert result["report"]["supervision_kind"] == "human_commit_change_supervision"


def test_exclude_component_for_nonrepresentative_origin_with_exclusive_priority():
    baseline = build()
    value = next(x for x in baseline["lineage"] if x["source_provenance"]["row_index"] == 100)
    policy = exclusions(
        repositories=["other/shared"],
        commits=["f" * 40],
        input_sha256=[value["origin_record"]["input_sha256"]],
    )
    result = build(policy)
    excluded = [x for x in result["lineage"] if x["decision"] == "component_excluded"]
    assert len(excluded) == 2
    assert result["report"]["exclusion_reason_counts"]["repositories"] == 2
    assert result["report"]["exclusion_reason_counts"]["commits"] == 0
    original = {
        (x["source_provenance"]["row_index"]): (x["group_id"], x["split"]) for x in baseline["lineage"]
    }
    assert all((x["group_id"], x["split"]) == original[x["source_provenance"]["row_index"]] for x in excluded)
    assert replay(result)["counts"]["excluded_origins"] == 2


def test_input_shuffle_is_identical_and_duplicate_origin_fields_survive():
    original = build()
    shuffled = prep.build_commitpack_preparation(list(reversed(records())), exclusion_policy=exclusions())
    assert original == shuffled
    duplicate = next(x for x in original["lineage"] if x["source_provenance"]["row_index"] == 100)
    assert duplicate["origin_record"]["commit"] == "f" * 40
    assert duplicate["origin_record"]["license"] == "Apache-2.0"
    assert duplicate["origin_record"]["source_fields"]["repos"] == "team/repo-1,other/shared"


@pytest.mark.parametrize(
    "change",
    [
        {"repositories": ["Owner/Repo"]},
        {"commits": ["A" * 40]},
        {"repositories": ["a/b", "a/b"]},
        {"canonical_sha256": ["bad"]},
        {"schema_version": True},
        {"image_uri": "grant"},
    ],
)
def test_malformed_or_noncanonical_exclusions_refuse(change):
    with pytest.raises(ValueError):
        prep.validate_commitpack_exclusions(exclusions(**change))


def test_exact_canonical_exclusion_encoding_is_utf8_with_one_newline():
    policy = exclusions()
    raw = prep.encode_commitpack_metadata(policy)
    assert (
        raw
        == (
            json.dumps(policy, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        ).encode()
    )
    assert replay(build())["exclusion_policy_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        "repo",
        "commit",
        "license",
        "raw_fields",
        "digest",
        "transform",
        "origin_missing",
        "ordinal",
        "membership",
        "split_payload",
        "report_bool",
        "coverage",
        "exclusion_count",
    ],
)
def test_replay_rejects_rehashed_metadata_substitution(mutation):
    result = copy.deepcopy(build())
    first = next(x for x in result["lineage"] if x["origin_record"] is not None)
    record = first["origin_record"]
    if mutation == "repo":
        record["repositories"] = ["other/foreign"]
    elif mutation == "commit":
        record["commit"] = "0" * 40
    elif mutation == "license":
        record["license"] = "BSD-3-Clause"
    elif mutation == "raw_fields":
        record["source_fields"]["new_contents"] += "changed"
    elif mutation == "digest":
        record["canonical_sha256"] = "0" * 64
    elif mutation == "transform":
        record["transformations"] = []
    elif mutation == "origin_missing":
        result["lineage"].pop()
    elif mutation == "ordinal":
        first["source_provenance"]["row_index"] = 999
    elif mutation == "membership":
        result["membership"][0]["split"] = "test"
    elif mutation == "split_payload":
        result["splits"]["train"][0]["messages"][1]["content"] = "foreign"
    elif mutation == "report_bool":
        result["report"]["counts"]["source"] = True
    elif mutation == "coverage":
        result["report"]["coverage"]["tokenization"] = "ready"
    else:
        result["report"]["exclusion_reason_counts"]["repositories"] = 1
    with pytest.raises(ValueError):
        replay(result)


def test_empty_or_all_excluded_is_explicitly_nonusable():
    for values, policy in [
        ([], exclusions()),
        (
            records(),
            exclusions(
                repositories=sorted({"team/repo-" + str(i) for i in range(1, 101)} | {"other/shared"})
            ),
        ),
    ]:
        result = prep.build_commitpack_preparation(values, exclusion_policy=policy)
        assert result["status"] == "incomplete" and all(not x for x in result["splits"].values())
        assert result["nonusable_outcome"]["scan_complete"] is False
        assert result["nonusable_outcome"]["usable_split_refs"] is False


def test_missing_conflicting_or_repeated_ordinal_refuses():
    values = records()
    values[1] = dataclasses.replace(values[1], source=source(999))
    with pytest.raises(ValueError):
        prep.build_commitpack_preparation(values, exclusion_policy=exclusions())


def test_metadata_only_replay_does_not_authenticate_unused_original_message():
    original = build()
    changed = copy.deepcopy(original)
    entry = next(value for value in changed["lineage"] if value["origin_record"] is not None)
    entry["origin_record"]["source_fields"]["message"] += "unused metadata changed"
    # This layer has no trusted raw source bytes: it characterizes the frozen
    # raw_digest_limit, not authentication after replacement of every outer pin.
    result = replay(changed)
    assert result["counts"] == replay(original)["counts"]
    assert "original_source_verified" not in result and "training_ready" not in result
