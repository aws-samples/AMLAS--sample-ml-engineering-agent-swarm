"""Pure CommitPack v3 preparation/replay; no parser, model, SDK or authority."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from . import commitpack_profile as profile
from .commitpack_profile import CommitPackSource

PROFILE_ID = "bigcode.commitpackft.python.change_sft.v1"
POLICY_VERSION = "coding-preparation.commitpack.v3"
SUPERVISION_KIND = "human_commit_change_supervision"
MAX_EXCLUSIONS_BYTES = 8 * 1024 * 1024
COMMITPACK_REPORT_LIMIT = MAX_EXCLUSIONS_BYTES + 262144
SOURCE_BYTES = 135858935
MAX_ROWS = 100000
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
SPLITS = ("train", "validation", "test")
EXCLUSION_KEYS = ("repositories", "commits", "canonical_sha256", "input_sha256", "code_pair_sha256")
COVERAGE = {
    "repository_affiliations": "all_declared",
    "commit_edges": "exact_source_commit",
    "conversation_duplicates": "exact",
    "full_input_code_duplicates": "exact",
    "near_duplicates": "not_implemented",
    "complete_upstream_fork_graph": "not_verified",
    "heldout_exclusions": "exact_list_component_exclusions",
    "tokenization": "not_applied",
    "exclusion_scope": "exact_listed_identity_and_digest_matches_only",
}
LINEAGE_KEYS = frozenset(
    {
        "source_provenance",
        "raw_line_sha256",
        "decision",
        "origin_record",
        "canonical_sha256",
        "group_id",
        "split",
        "representative_origin_sha256",
        "rejection_reason",
    }
)
ORIGIN_KEYS = frozenset(
    {
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
)
REPORT_KEYS = frozenset(
    {
        "schema_version",
        "profile_id",
        "policy_version",
        "status",
        "counts",
        "normalization_rejection_reasons",
        "exclusion_policy",
        "exclusion_reason_counts",
        "coverage",
        "supervision_kind",
    }
)
COUNT_KEYS = frozenset(
    {
        "source",
        "normalization_rejected",
        "normalized",
        "excluded_origins",
        "included_origins",
        "retained_representatives",
        "duplicate_origins",
        "split_counts",
    }
)
_SHA = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class CommitPackParsedRow:
    row: dict[str, Any]
    source: CommitPackSource
    raw_line_sha256: str


def _require(value: bool, message: str = "CommitPack preparation metadata differs") -> None:
    if not value:
        raise ValueError(message)


def _plain(value: Any, *, limit: int = MAX_EXPANDED_BYTES) -> None:
    stack = [(value, 0)]
    nodes = size = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _require(depth <= 32 and nodes <= MAX_ROWS * 256, "CommitPack metadata envelope exceeded")
        kind = type(item)
        if kind is dict:
            _require(len(item) <= 100000 and all(type(k) is str for k in item))
            for key, child in item.items():
                size += len(key.encode("utf-8"))
                stack.append((child, depth + 1))
        elif kind is list:
            _require(len(item) <= MAX_ROWS)
            stack.extend((child, depth + 1) for child in item)
        elif kind is str:
            try:
                size += len(item.encode("utf-8"))
            except UnicodeError as error:
                raise ValueError("CommitPack text encoding differs") from error
        elif kind is int:
            _require(-(2**63) <= item < 2**63)
        else:
            _require(item is None or kind is bool)
        _require(size <= limit, "CommitPack expanded metadata bound exceeded")


def encode_commitpack_metadata(value: Any) -> bytes:
    _plain(value)
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _digest(value: Any) -> str:
    _plain(value)
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _same(a: Any, b: Any) -> bool:
    return encode_commitpack_metadata(a) == encode_commitpack_metadata(b)


def validate_commitpack_exclusions(value: Any) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == {"schema_version", *EXCLUSION_KEYS})
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    count = 0
    result = {"schema_version": 1}
    for name in EXCLUSION_KEYS:
        values = value[name]
        _require(type(values) is list and len(values) <= MAX_ROWS)
        _require(all(type(item) is str for item in values))
        _require(values == sorted(set(values)), "noncanonical CommitPack exclusions")
        count += len(values)
        _require(count <= MAX_ROWS)
        if name == "repositories":
            _require(
                all(
                    profile._REPO.fullmatch(item) and ".." not in item and item == item.casefold()
                    for item in values
                )
            )
        elif name == "commits":
            _require(all(re.fullmatch(r"[a-f0-9]{40}", item) for item in values))
        else:
            _require(all(_SHA.fullmatch(item) for item in values))
        result[name] = list(values)
    _require(len(encode_commitpack_metadata(result)) <= MAX_EXCLUSIONS_BYTES)
    return result


def _provenance(value: Any) -> CommitPackSource:
    source = profile._source_from_provenance(value)
    _require(source is not None, "CommitPack source provenance differs")
    return source


def _source_key(provenance: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(provenance[name] for name in ("artifact_uri", "artifact_version_id", "artifact_sha256"))


def _check_origins(entries: list[tuple[dict[str, Any], str]]) -> None:
    ordinals: dict[tuple[str, str, str], set[int]] = {}
    for provenance, raw_digest in entries:
        _provenance(provenance)
        _require(type(raw_digest) is str and _SHA.fullmatch(raw_digest) is not None)
        values = ordinals.setdefault(_source_key(provenance), set())
        ordinal = provenance["row_index"]
        _require(ordinal not in values, "CommitPack original ordinal repeats")
        values.add(ordinal)
    _require(
        all(values == set(range(len(values))) for values in ordinals.values()),
        "CommitPack original ordinal coverage differs",
    )


def _nonusable(reason: str, observed: int, normalized: int, rejected: int) -> dict[str, Any]:
    return {
        "status": "incomplete",
        "splits": {name: [] for name in SPLITS},
        "membership": [],
        "lineage": [],
        "report": None,
        "nonusable_outcome": {
            "schema_version": 1,
            "profile_id": PROFILE_ID,
            "policy_version": POLICY_VERSION,
            "status": "incomplete",
            "scan_complete": False,
            "reason": reason,
            "observed_prefix_counts": {
                "source": observed,
                "normalized": normalized,
                "normalization_rejected": rejected,
            },
            "usable_split_refs": False,
            "committed_success_marker": False,
            "trained": False,
            "training_ready": False,
        },
    }


def _assemble(
    envelopes: list[dict[str, Any]], raw_digests: list[str], exclusion_policy: dict[str, Any]
) -> dict[str, Any]:
    exclusions = validate_commitpack_exclusions(exclusion_policy)
    _check_origins(
        [
            (envelope["source_provenance"], digest)
            for envelope, digest in zip(envelopes, raw_digests, strict=True)
        ]
    )
    normalized = sum(envelope["status"] == "normalized" for envelope in envelopes)
    rejected = len(envelopes) - normalized
    grouped = profile.group_commitpack_records(envelopes)
    if grouped["status"] != "grouped":
        reason = grouped["issues"][0]["reason"] if grouped["issues"] else "grouping_incomplete"
        return _nonusable(reason, len(envelopes), normalized, rejected)
    groups: dict[str, list[dict[str, Any]]] = {}
    for values in grouped["splits"].values():
        for record in values:
            groups.setdefault(record["group_id"], []).append(record)
    excluded_groups = {}
    excluded_reasons = {name: 0 for name in EXCLUSION_KEYS}
    for group_id, representatives in groups.items():
        all_origins = [origin for record in representatives for origin in record["origin_records"]]
        for name in EXCLUSION_KEYS:
            selected = set(exclusions[name])
            if any(
                bool(selected.intersection(origin[name]))
                if name == "repositories"
                else origin["commit" if name == "commits" else name] in selected
                for origin in all_origins
            ):
                excluded_groups[group_id] = name
                excluded_reasons[name] += len(all_origins)
                break
    origin_info = {}
    split_records = {name: [] for name in SPLITS}
    membership = []
    for split in SPLITS:
        values = sorted(grouped["splits"][split], key=lambda record: record["canonical_sha256"])
        for record in values:
            group_id = record["group_id"]
            representative = record["source_origins"][0]
            rep_digest = _digest(representative)
            excluded = group_id in excluded_groups
            if not excluded:
                split_records[split].append({"messages": record["messages"], "tools": record["tools"]})
                membership.append(
                    {"canonical_sha256": record["canonical_sha256"], "group_id": group_id, "split": split}
                )
            for origin in record["origin_records"]:
                prov = origin["source_provenance"]
                decision = (
                    "component_excluded"
                    if excluded
                    else "retained_representative"
                    if prov == representative
                    else "duplicate_origin"
                )
                origin_info[_digest(prov)] = {
                    "origin_record": origin,
                    "canonical_sha256": record["canonical_sha256"],
                    "group_id": group_id,
                    "split": split,
                    "representative_origin_sha256": rep_digest,
                    "decision": decision,
                    "rejection_reason": None,
                }
    if not all(split_records.values()):
        return _nonusable(
            "split_population_incomplete_after_exclusions", len(envelopes), normalized, rejected
        )
    lineage = []
    reasons = Counter()
    order = sorted(
        range(len(envelopes)),
        key=lambda index: (
            *_source_key(envelopes[index]["source_provenance"]),
            envelopes[index]["source_provenance"]["row_index"],
        ),
    )
    for index in order:
        envelope = envelopes[index]
        provenance = envelope["source_provenance"]
        entry = {"source_provenance": provenance, "raw_line_sha256": raw_digests[index]}
        if envelope["status"] == "normalized":
            entry.update(origin_info[_digest(provenance)])
        else:
            reason = envelope["issues"][0]["reason"]
            _require(reason in profile._REASONS and reason != "source_provenance_invalid")
            reasons[reason] += 1
            entry.update(
                {
                    "decision": "normalization_rejected",
                    "origin_record": None,
                    "canonical_sha256": None,
                    "group_id": None,
                    "split": None,
                    "representative_origin_sha256": None,
                    "rejection_reason": reason,
                }
            )
        lineage.append(entry)
    included = normalized - sum(excluded_reasons.values())
    retained = len(membership)
    counts = {
        "source": len(envelopes),
        "normalization_rejected": rejected,
        "normalized": normalized,
        "excluded_origins": sum(excluded_reasons.values()),
        "included_origins": included,
        "retained_representatives": retained,
        "duplicate_origins": included - retained,
        "split_counts": {name: len(split_records[name]) for name in SPLITS},
    }
    report = {
        "schema_version": 3,
        "profile_id": PROFILE_ID,
        "policy_version": POLICY_VERSION,
        "status": "complete",
        "counts": counts,
        "normalization_rejection_reasons": dict(sorted(reasons.items())),
        "exclusion_policy": exclusions,
        "exclusion_reason_counts": excluded_reasons,
        "coverage": dict(COVERAGE),
        "supervision_kind": SUPERVISION_KIND,
    }
    try:
        # A complete producer result must fit the exact consumer representations.
        for value in (lineage, membership, split_records):
            _plain(value, limit=MAX_EXPANDED_BYTES)
    except ValueError:
        return _nonusable("output_resource_bound", len(envelopes), normalized, rejected)
    return {
        "status": "complete",
        "splits": split_records,
        "membership": membership,
        "lineage": lineage,
        "report": report,
    }


def build_commitpack_preparation(
    parsed_rows: list[CommitPackParsedRow], *, exclusion_policy: dict[str, Any]
) -> dict[str, Any]:
    validate_commitpack_exclusions(exclusion_policy)
    _require(type(parsed_rows) is list)
    if len(parsed_rows) > MAX_ROWS:
        return _nonusable("group_resource_bound", 0, 0, 0)
    envelopes = []
    digests = []
    measured = 0
    for item in parsed_rows:
        _require(type(item) is CommitPackParsedRow and type(item.source) is CommitPackSource)
        envelope = profile.normalize_commitpack_row(item.row, source=item.source)
        _require(envelope["source_provenance"] is not None, "CommitPack source provenance differs")
        measured += max(len(profile._encoded(envelope)), len(profile._encoded(item.row)))
        if measured > MAX_EXPANDED_BYTES:
            return _nonusable(
                "group_resource_bound",
                len(envelopes),
                sum(x["status"] == "normalized" for x in envelopes),
                sum(x["status"] == "rejected" for x in envelopes),
            )
        envelopes.append(envelope)
        digests.append(item.raw_line_sha256)
    return _assemble(envelopes, digests, exclusion_policy)


def _validate_report(report: Any) -> None:
    _plain(report, limit=COMMITPACK_REPORT_LIMIT)
    _require(type(report) is dict and set(report) == REPORT_KEYS)
    _require(type(report["schema_version"]) is int and report["schema_version"] == 3)
    _require(
        (report["profile_id"], report["policy_version"], report["status"], report["supervision_kind"])
        == (PROFILE_ID, POLICY_VERSION, "complete", SUPERVISION_KIND)
    )
    _require(_same(report["coverage"], COVERAGE))
    counts = report["counts"]
    _require(type(counts) is dict and set(counts) == COUNT_KEYS)
    for name, value in counts.items():
        if name != "split_counts":
            _require(type(value) is int and 0 <= value <= MAX_ROWS)
    _require(type(counts["split_counts"]) is dict and set(counts["split_counts"]) == set(SPLITS))
    _require(all(type(value) is int and 0 < value <= MAX_ROWS for value in counts["split_counts"].values()))
    _require(
        counts["source"] == counts["normalization_rejected"] + counts["normalized"]
        and counts["normalized"] == counts["excluded_origins"] + counts["included_origins"]
        and counts["included_origins"] == counts["retained_representatives"] + counts["duplicate_origins"]
        and counts["retained_representatives"] == sum(counts["split_counts"].values())
    )
    reasons = report["normalization_rejection_reasons"]
    _require(
        type(reasons) is dict
        and all(
            name in profile._REASONS
            and name != "source_provenance_invalid"
            and type(count) is int
            and 0 < count <= MAX_ROWS
            for name, count in reasons.items()
        )
    )
    _require(sum(reasons.values()) == counts["normalization_rejected"])
    reasons = report["exclusion_reason_counts"]
    _require(
        type(reasons) is dict
        and set(reasons) == set(EXCLUSION_KEYS)
        and all(type(count) is int and 0 <= count <= MAX_ROWS for count in reasons.values())
    )
    _require(sum(reasons.values()) == counts["excluded_origins"])
    validate_commitpack_exclusions(report["exclusion_policy"])


def replay_commitpack_preparation(
    *,
    lineage_records: list[dict[str, Any]],
    membership_records: list[dict[str, Any]],
    split_records: dict[str, list[dict[str, Any]]],
    report: dict[str, Any],
    expected_sources: tuple[dict[str, Any], ...] | None = None,
    expected_source_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    _validate_report(report)
    _require(type(lineage_records) is list and len(lineage_records) == report["counts"]["source"] <= MAX_ROWS)
    _require(
        type(membership_records) is list and type(split_records) is dict and set(split_records) == set(SPLITS)
    )
    _plain(lineage_records)
    _plain(membership_records)
    _plain(split_records)
    source_keys = None
    if expected_sources is not None:
        _require(type(expected_sources) is tuple and 0 < len(expected_sources) <= 256)
        source_keys = set()
        for item in expected_sources:
            _require(type(item) is dict and set(item) == {"repo_id", "revision", "filename", "artifact_ref"})
            ref = item["artifact_ref"]
            _require(
                type(ref) is dict and set(ref) == {"uri", "version_id", "sha256", "byte_count", "format"}
            )
            _require(
                (
                    item["repo_id"],
                    item["revision"],
                    item["filename"],
                    ref["sha256"],
                    ref["byte_count"],
                    ref["format"],
                )
                == (
                    profile.DATASET_REPO,
                    profile.REVISION,
                    profile.FILENAME,
                    profile.ARTIFACT_SHA256,
                    SOURCE_BYTES,
                    "jsonl",
                )
                and type(ref["byte_count"]) is int
            )
            source_keys.add((ref["uri"], ref["version_id"], ref["sha256"]))
    envelopes, digests = [], []
    for entry in lineage_records:
        _require(type(entry) is dict and set(entry) == LINEAGE_KEYS)
        provenance = entry["source_provenance"]
        source = _provenance(provenance)
        if source_keys is not None:
            _require(_source_key(provenance) in source_keys, "CommitPack foreign original source")
        record = entry["origin_record"]
        if entry["decision"] == "normalization_rejected":
            reason = entry["rejection_reason"]
            _require(reason in profile._REASONS and reason != "source_provenance_invalid")
            _require(
                all(
                    entry[name] is None
                    for name in (
                        "origin_record",
                        "canonical_sha256",
                        "group_id",
                        "split",
                        "representative_origin_sha256",
                    )
                )
            )
            envelope = {
                "status": "rejected",
                "record": None,
                "issues": [{"reason": reason, "count": 1}],
                "source_provenance": provenance,
                "data_only": True,
            }
        else:
            _require(
                type(record) is dict and set(record) == ORIGIN_KEYS and entry["rejection_reason"] is None
            )
            envelope = profile.normalize_commitpack_row(record["source_fields"], source=source)
            _require(envelope["status"] == "normalized")
            expected = {name: envelope["record"][name] for name in ORIGIN_KEYS if name != "source_provenance"}
            expected["source_provenance"] = provenance
            _require(_same(record, expected), "CommitPack normalized original row changed")
        envelopes.append(envelope)
        digests.append(entry["raw_line_sha256"])
    if source_keys is not None:
        _require(
            {_source_key(item["source_provenance"]) for item in lineage_records} == source_keys,
            "CommitPack missing original source",
        )
    if expected_source_counts is not None:
        _require(type(expected_source_counts) is dict and bool(expected_source_counts))
        _require(
            all(
                type(key) is str and _SHA.fullmatch(key) and type(count) is int and 0 <= count <= MAX_ROWS
                for key, count in expected_source_counts.items()
            )
        )
        measured = Counter(
            commitpack_source_key(*_source_key(entry["source_provenance"])) for entry in lineage_records
        )
        _require(
            set(measured) == set(expected_source_counts) and dict(measured) == expected_source_counts,
            "CommitPack private per-source count differs",
        )
    rebuilt = _assemble(envelopes, digests, report["exclusion_policy"])
    _require(rebuilt["status"] == "complete")
    for name, provided in (
        ("lineage", lineage_records),
        ("membership", membership_records),
        ("splits", split_records),
        ("report", report),
    ):
        _require(_same(rebuilt[name], provided), "CommitPack replayed " + name + " differs")
    return {
        "counts": rebuilt["report"]["counts"],
        "canonical_sha256_by_split": {
            name: [item["canonical_sha256"] for item in rebuilt["membership"] if item["split"] == name]
            for name in SPLITS
        },
        "supervision_kind": SUPERVISION_KIND,
        "coverage": dict(COVERAGE),
        "exclusion_policy_sha256": hashlib.sha256(
            encode_commitpack_metadata(report["exclusion_policy"])
        ).hexdigest(),
    }


def matches_commitpack_source(repo_id: str, revision: str, filename: str, artifact_ref: Any) -> bool:
    return (
        (repo_id, revision, filename, artifact_ref.format, artifact_ref.sha256)
        == (profile.DATASET_REPO, profile.REVISION, profile.FILENAME, "jsonl", profile.ARTIFACT_SHA256)
        and type(artifact_ref.byte_count) is int
        and artifact_ref.byte_count == SOURCE_BYTES
    )


def validate_nonusable_outcome(value: Any) -> dict[str, Any]:
    keys = {
        "schema_version",
        "profile_id",
        "policy_version",
        "status",
        "scan_complete",
        "reason",
        "observed_prefix_counts",
        "usable_split_refs",
        "committed_success_marker",
        "trained",
        "training_ready",
    }
    _plain(value, limit=262144)
    _require(type(value) is dict and set(value) == keys)
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    _require(
        (value["profile_id"], value["policy_version"], value["status"])
        == (PROFILE_ID, POLICY_VERSION, "incomplete")
    )
    _require(
        all(
            value[name] is False
            for name in (
                "scan_complete",
                "usable_split_refs",
                "committed_success_marker",
                "trained",
                "training_ready",
            )
        )
    )
    _require(type(value["reason"]) is str and re.fullmatch(r"[a-z0-9_]{1,96}", value["reason"]) is not None)
    counts = value["observed_prefix_counts"]
    _require(type(counts) is dict and set(counts) == {"source", "normalized", "normalization_rejected"})
    _require(all(type(count) is int and 0 <= count <= MAX_ROWS for count in counts.values()))
    _require(counts["normalized"] + counts["normalization_rejected"] <= counts["source"])
    return value


def commitpack_source_key(uri: str, version_id: str, sha256: str) -> str:
    _require(type(uri) is str and type(version_id) is str and type(sha256) is str)
    return _digest({"artifact_uri": uri, "artifact_version_id": version_id, "artifact_sha256": sha256})
