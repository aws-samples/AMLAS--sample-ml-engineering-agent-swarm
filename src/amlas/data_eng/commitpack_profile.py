"""Pure parsed CommitPack change supervision; no parser, execution or authority."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from . import coding_preparation as core

PROFILE_ID = "bigcode.commitpackft.python.change_sft.v1"
DATASET_REPO = "bigcode/commitpackft"
REVISION = "fc56fe33c030c6daa414c2b112c932b8eed085e6"  # pragma: allowlist secret - public revision
FILENAME = "data/python/data.jsonl"
ARTIFACT_SHA256 = "d167da37e1058371c48e057cd8815d03700c867dd8bcf58e61420d4dcd288d73"  # pragma: allowlist secret - public artifact digest
CARD_SHA256 = "69799047051e9d487d20e8930d73de1b3023ee1c75cf36dc3d2dfca2643f0bb0"  # pragma: allowlist secret - public card digest
_FIELDS = frozenset(
    {
        "commit",
        "old_file",
        "new_file",
        "old_contents",
        "new_contents",
        "subject",
        "message",
        "lang",
        "license",
        "repos",
    }
)
_ROW_BYTES = 262144
_CODE_BYTES = 98304
_MAX_RECORDS = 100000
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
_LICENSE_MAP = {label.casefold(): label for label in core._LICENSES}
_REASONS = frozenset(
    {
        "source_provenance_invalid",
        "row_shape_invalid",
        "row_text_invalid",
        "row_oversize",
        "language_unsupported",
        "commit_invalid",
        "file_identity_invalid",
        "supervision_empty",
        "license_unsupported",
        "repositories_invalid",
        "secret_detected",
        "chat_control_detected",
    }
)
_ENVELOPE = frozenset({"status", "record", "issues", "source_provenance", "data_only"})
_TRANSFORMS = [
    "change_wrapper_v1",
    "license_case_map_existing_whitelist",
    "all_repo_affiliations_casefold_dedup",
    "commit_hex_casefold",
    "full_source_code_preserved",
]


@dataclass(frozen=True)
class CommitPackSource:
    """Immutable version-bound data provenance; not a grant or attestation."""

    dataset_repo: str
    revision: str
    filename: str
    artifact_uri: str
    artifact_version_id: str
    artifact_sha256: str
    row_index: int


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _text(value: Any, limit: int, *, controls: bool = True) -> bool:
    if type(value) is not str or len(value) > limit:
        return False
    try:
        if len(value.encode("utf-8")) > limit:
            return False
    except UnicodeError:
        return False
    return not any(ord(c) < 32 and (not controls or c not in "\n\r\t") for c in value)


def _provenance(source: Any) -> dict[str, Any] | None:
    if type(source) is not CommitPackSource:
        return None
    try:
        data = {name: object.__getattribute__(source, name) for name in CommitPackSource.__dataclass_fields__}
    except AttributeError:
        return None
    if (
        any(type(data[k]) is not str for k in data if k != "row_index")
        or type(source.row_index) is not int
        or not 0 <= source.row_index < 2**63
    ):
        return None
    if (source.dataset_repo, source.revision, source.filename, source.artifact_sha256) != (
        DATASET_REPO,
        REVISION,
        FILENAME,
        ARTIFACT_SHA256,
    ):
        return None
    if (
        not _text(source.artifact_uri, 4096, controls=False)
        or not _text(source.artifact_version_id, 1024, controls=False)
        or not source.artifact_version_id
        or source.artifact_version_id.casefold() == "null"
    ):
        return None
    try:
        uri = urlsplit(source.artifact_uri)
    except ValueError:
        return None
    if uri.scheme == "s3":
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", uri.netloc)
            or uri.query
            or uri.fragment
            or not uri.path.startswith("/")
            or any(x in {"", ".", ".."} for x in uri.path[1:].split("/"))
            or "\\" in source.artifact_uri
            or "%" in source.artifact_uri
        ):
            return None
    elif (
        source.artifact_uri != f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{REVISION}/{FILENAME}"
    ):
        return None
    return {
        **data,
        "profile_id": PROFILE_ID,
        "card_sha256": CARD_SHA256,
        "dataset_license": "MIT",
        "coverage": "declared_source_identity_only; no artifact authentication or readiness",
    }


def _rejected(reason: str, provenance: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "status": "rejected",
        "record": None,
        "issues": [{"reason": reason, "count": 1}],
        "source_provenance": provenance,
        "data_only": True,
    }


def _file_name(value: str) -> bool:
    return (
        bool(value)
        and _text(value, 4096, controls=False)
        and not value.startswith("/")
        and "\\" not in value
        and not any(part in {"", ".", ".."} for part in value.split("/"))
    )


def normalize_commitpack_row(row: Any, *, source: CommitPackSource) -> dict[str, Any]:
    """Normalize complete already-parsed strings without executing source content."""
    provenance = _provenance(source)
    if provenance is None:
        return _rejected("source_provenance_invalid", None)
    if (
        type(row) is not dict
        or len(row) != len(_FIELDS)
        or any(type(key) is not str for key in row)
        or set(row) != _FIELDS
    ):
        return _rejected("row_shape_invalid", provenance)
    if any(type(value) is not str for value in row.values()):
        return _rejected("row_text_invalid", provenance)
    for key, value in row.items():
        limit = _CODE_BYTES if key in {"old_contents", "new_contents"} else 4096
        if len(value) > limit:
            return _rejected("row_oversize", provenance)
        if not _text(value, limit):
            return _rejected("row_text_invalid", provenance)
    if sum(len(v.encode("utf-8")) for v in row.values()) > _ROW_BYTES:
        return _rejected("row_oversize", provenance)
    if row["lang"].strip().casefold() != "python":
        return _rejected("language_unsupported", provenance)
    if not re.fullmatch(r"[a-fA-F0-9]{40}", row["commit"]):
        return _rejected("commit_invalid", provenance)
    if not _file_name(row["old_file"]) or not _file_name(row["new_file"]):
        return _rejected("file_identity_invalid", provenance)
    if any(not row[k].strip() for k in ["subject", "old_contents", "new_contents"]):
        return _rejected("supervision_empty", provenance)
    license_label = _LICENSE_MAP.get(row["license"].strip().casefold())
    if license_label is None:
        return _rejected("license_unsupported", provenance)
    affiliations = row["repos"].split(",")
    if not 1 <= len(affiliations) <= 16:
        return _rejected("repositories_invalid", provenance)
    tokens = [token.strip() for token in affiliations]
    if any(not _REPO.fullmatch(token) or ".." in token for token in tokens):
        return _rejected("repositories_invalid", provenance)
    for value in row.values():
        if any(pattern.search(value) for pattern in core._SECRET_PATTERNS.values()):
            return _rejected("secret_detected", provenance)
        if any(
            marker in value
            for marker in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|assistant|>", "<|system|>"]
        ):
            return _rejected("chat_control_detected", provenance)
    user = f"Apply the following recorded change.\nSubject: {row['subject']}\nOriginal file: {row['old_file']}\nUpdated file: {row['new_file']}\nOriginal contents:\n{row['old_contents']}\nReturn the complete updated file contents."
    canonical = {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["new_contents"]},
        ],
        "tools": [],
    }
    record = {
        **canonical,
        "source_fields": dict(row),
        "commit": row["commit"].lower(),
        "repositories": sorted({token.casefold() for token in tokens}),
        "license": license_label,
        "canonical_sha256": _digest(canonical),
        "input_sha256": _digest({"subject": row["subject"], "old_contents": row["old_contents"]}),
        "code_pair_sha256": _digest(
            {"old_contents": row["old_contents"], "new_contents": row["new_contents"]}
        ),
        "purpose": "human_commit_change_supervision",
        "transformations": list(_TRANSFORMS),
    }
    return {
        "status": "normalized",
        "record": record,
        "issues": [],
        "source_provenance": provenance,
        "data_only": True,
    }


_COVERAGE = {
    "repository_affiliations": "all_declared",
    "commit_edges": "exact_source_commit",
    "conversation_duplicates": "exact",
    "full_input_code_duplicates": "exact",
    "near_duplicates": "not_implemented",
    "complete_upstream_fork_graph": "not_verified",
    "heldout_exclusions": "not_applied",
    "tokenization": "not_applied",
}


def _incomplete(reason: str, stats: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "status": "incomplete",
        "splits": {"train": [], "validation": [], "test": []},
        "issues": [{"reason": reason, "count": 1}],
        "stats": stats or {},
        "coverage": dict(_COVERAGE),
        "data_only": True,
    }


def _source_from_provenance(value: Any) -> CommitPackSource | None:
    if type(value) is not dict:
        return None
    fields = CommitPackSource.__dataclass_fields__
    if set(value) != set(fields) | {"profile_id", "card_sha256", "dataset_license", "coverage"}:
        return None
    source = CommitPackSource(**{key: value[key] for key in fields})
    return source if _provenance(source) == value else None


def _plain_envelope(value: Any) -> bool:
    stack = [(value, 0)]
    nodes = 0
    byte_count = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 8 or nodes > 256:
            return False
        kind = type(item)
        if kind is dict:
            if len(item) > 32 or any(type(key) is not str or not _text(key, 4096) for key in item):
                return False
            byte_count += sum(len(key.encode("utf-8")) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif kind is list:
            if len(item) > 32:
                return False
            stack.extend((child, depth + 1) for child in item)
        elif kind is str:
            if not _text(item, 131072):
                return False
            byte_count += len(item.encode("utf-8"))
        elif kind is int:
            if not -(2**63) <= item < 2**63:
                return False
        elif item is not None and kind is not bool:
            return False
        if byte_count > 1024 * 1024:
            return False
    return True


def group_commitpack_records(normalized_records: Any) -> dict[str, Any]:
    """Union all exact source edges before canonical deduplication and splitting."""
    if type(normalized_records) is not list or len(normalized_records) > _MAX_RECORDS:
        return _incomplete("group_input_invalid")
    records = []
    rejected = 0
    total = 0
    immutable_origins = {}
    for envelope in normalized_records:
        if (
            not _plain_envelope(envelope)
            or type(envelope) is not dict
            or set(envelope) != _ENVELOPE
            or envelope["data_only"] is not True
        ):
            return _incomplete("group_envelope_invalid")
        try:
            total += len(_encoded(envelope))
        except (TypeError, ValueError, UnicodeError, RecursionError):
            return _incomplete("group_envelope_invalid")
        if total > _MAX_TOTAL_BYTES:
            return _incomplete("group_resource_bound")
        if envelope["status"] == "rejected":
            issues = envelope["issues"]
            if (
                envelope["record"] is not None
                or type(issues) is not list
                or len(issues) != 1
                or type(issues[0]) is not dict
                or set(issues[0]) != {"reason", "count"}
                or type(issues[0]["reason"]) is not str
                or issues[0]["reason"] not in _REASONS
                or type(issues[0]["count"]) is not int
                or issues[0]["count"] != 1
            ):
                return _incomplete("group_envelope_invalid")
            if (
                envelope["source_provenance"] is not None
                and _source_from_provenance(envelope["source_provenance"]) is None
            ):
                return _incomplete("group_provenance_invalid")
            rejected += 1
            continue
        source = _source_from_provenance(envelope["source_provenance"])
        record = envelope["record"]
        if (
            envelope["status"] != "normalized"
            or source is None
            or type(record) is not dict
            or type(record.get("source_fields")) is not dict
        ):
            return _incomplete("group_envelope_invalid")
        replay = normalize_commitpack_row(record["source_fields"], source=source)
        if replay != envelope:
            return _incomplete("group_record_changed")
        origin_key = _digest(envelope["source_provenance"])
        row_digest = _digest(record["source_fields"])
        if origin_key in immutable_origins and immutable_origins[origin_key] != row_digest:
            return _incomplete("source_origin_conflict")
        immutable_origins[origin_key] = row_digest
        records.append(envelope)
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen = {}
    key_sets = []
    for index, envelope in enumerate(records):
        r = envelope["record"]
        keys = [("repo", repo) for repo in r["repositories"]] + [
            ("commit", r["commit"]),
            ("canonical", r["canonical_sha256"]),
            ("input", r["input_sha256"]),
            ("code", r["code_pair_sha256"]),
        ]
        key_sets.append(keys)
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    groups = {}
    for index in range(len(records)):
        groups.setdefault(find(index), []).append(index)
    splits = {"train": [], "validation": [], "test": []}
    unique = 0
    for indices in groups.values():
        group_keys = sorted({key for index in indices for key in key_sets[index]})
        group_id = _digest({"profile_id": PROFILE_ID, "keys": group_keys})
        bucket = int(group_id[:16], 16) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        by_canonical = {}
        for index in indices:
            by_canonical.setdefault(records[index]["record"]["canonical_sha256"], []).append(records[index])
        for _canonical_digest, origins in sorted(by_canonical.items()):
            origins = sorted(origins, key=lambda e: _encoded(e["source_provenance"]))
            representative = origins[0]["record"]
            affiliations = sorted({repo for origin in origins for repo in origin["record"]["repositories"]})
            splits[split].append(
                {
                    **representative,
                    "repositories": affiliations,
                    "group_id": group_id,
                    "source_origins": [origin["source_provenance"] for origin in origins],
                    "origin_records": [
                        {
                            "source_provenance": origin["source_provenance"],
                            "source_fields": dict(origin["record"]["source_fields"]),
                            "commit": origin["record"]["commit"],
                            "license": origin["record"]["license"],
                            "repositories": list(origin["record"]["repositories"]),
                            "canonical_sha256": origin["record"]["canonical_sha256"],
                            "input_sha256": origin["record"]["input_sha256"],
                            "code_pair_sha256": origin["record"]["code_pair_sha256"],
                            "transformations": list(origin["record"]["transformations"]),
                        }
                        for origin in origins
                    ],
                }
            )
            unique += 1
    for values in splits.values():
        values.sort(key=lambda r: (r["group_id"], r["canonical_sha256"]))
    stats = {
        "input_rows": len(normalized_records),
        "normalized": len(records),
        "rejected": rejected,
        "duplicates": len(records) - unique,
        "unique_records": unique,
        "components": len(groups),
    }
    if not all(splits.values()):
        return _incomplete("split_population_incomplete", stats)
    return {
        "status": "grouped",
        "splits": splits,
        "issues": [],
        "stats": stats,
        "coverage": dict(_COVERAGE),
        "data_only": True,
    }
