"""Pure bounded HumanEval projection; code strings are never executed or imported."""

from __future__ import annotations

import hashlib
import json
import keyword
from typing import Any

from .coding_case_profile import (
    CASE_IDS,
    MAX_BUNDLE_BYTES,
    MAX_DECODED_BYTES,
    MAX_FIELD_BYTES,
    ORIGINAL_REPO,
    ORIGINAL_REVISION,
    PROFILE_ID,
    PROFILE_VERSION,
    SOURCE_FIELDS,
    CodingCaseSource,
    source_metadata,
)

FILES = ("PROMPTS.jsonl", "ORACLES.jsonl", "EXCLUSIONS.json", "CASE_MANIFEST.json", "COMMITTED.json")
ROLES = {
    "PROMPTS.jsonl": "public_prompt",
    "ORACLES.jsonl": "private_oracle",
    "EXCLUSIONS.json": "private_preparation_exclusions",
    "CASE_MANIFEST.json": "lineage_membership_manifest",
}
COVERAGE = {
    "case_population": "complete_164_source_ids",
    "literal_code_pair_matches": "exact",
    "input_wrapper_overlap": "not_verified",
    "near_or_semantic_overlap": "not_verified",
    "pretraining_contamination": "unknown",
    "raw_bytes_origin": "caller_integrity_gate_required",
}


def json_bytes(value: Any, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return raw + b"\n" if newline else raw


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_rows(rows: Any) -> list[dict[str, str]]:
    if type(rows) not in (list, tuple) or len(rows) != len(CASE_IDS):
        raise ValueError("complete ordered164 decoded cases required")
    total = 0
    result = []
    for index, row in enumerate(rows):
        if (
            type(row) is not dict
            or set(row) != SOURCE_FIELDS
            or any(type(value) is not str or not value for value in row.values())
        ):
            raise ValueError("closed five-string case fields required")
        if row["task_id"] != CASE_IDS[index]:
            raise ValueError("case ID population/order differs")
        entry = row["entry_point"]
        if not entry.isascii() or not entry.isidentifier() or keyword.iskeyword(entry):
            raise ValueError("case entry point must be a plain Python identifier")
        try:
            sizes = [len(value.encode("utf-8")) for value in row.values()]
        except UnicodeError:
            raise ValueError("invalid coding-case UTF8") from None
        if any(size > MAX_FIELD_BYTES for size in sizes):
            raise ValueError("case field byte bound exceeded")
        total += sum(sizes)
        if total > MAX_DECODED_BYTES:
            raise ValueError("complete decoded case bound exceeded")
        result.append(dict(row))
    return result


def exclusion_projection(rows: list[dict[str, str]]) -> dict:
    # Same compact UTF8 JSON digest domain as CommitPack's literal old/new pair.
    pairs = sorted(
        {
            digest(
                json_bytes(
                    {
                        "old_contents": row["prompt"],
                        "new_contents": row["prompt"] + row["canonical_solution"],
                    },
                    newline=False,
                )
            )
            for row in rows
        }
    )
    return {
        "schema_version": 1,
        "repositories": [ORIGINAL_REPO],
        "commits": [ORIGINAL_REVISION],
        "canonical_sha256": [],
        "input_sha256": [],
        "code_pair_sha256": pairs,
    }


def descriptor(raw: bytes, *, role: str, record_count: int, format: str) -> dict:
    return {
        "sha256": digest(raw),
        "byte_count": len(raw),
        "role": role,
        "record_count": record_count,
        "format": format,
    }


def freeze_humaneval_cases(rows: Any, *, source: CodingCaseSource) -> dict[str, bytes]:
    """Return canonical bytes only; source identity is descriptive, not authentication."""
    pin = source_metadata(source)
    original = validate_rows(rows)
    prompts = [
        {"case_id": row["task_id"], "prompt": row["prompt"], "entry_point": row["entry_point"]}
        for row in original
    ]
    oracles = [
        {"case_id": row["task_id"], "test": row["test"], "canonical_solution": row["canonical_solution"]}
        for row in original
    ]
    bundle = {
        "PROMPTS.jsonl": b"".join(json_bytes(row) for row in prompts),
        "ORACLES.jsonl": b"".join(json_bytes(row) for row in oracles),
        "EXCLUSIONS.json": json_bytes(exclusion_projection(original)),
    }
    files = {
        name: descriptor(
            raw,
            role=ROLES[name],
            record_count=164 if name.endswith(".jsonl") else 1,
            format="jsonl" if name.endswith(".jsonl") else "json",
        )
        for name, raw in bundle.items()
    }
    cases = []
    for index, row in enumerate(original):
        cases.append(
            {
                "case_id": row["task_id"],
                "source_record_ordinal": index,
                "field_sha256": {key: digest(value.encode("utf-8")) for key, value in row.items()},
                "source_record_sha256": digest(json_bytes(row, newline=False)),
                "prompt_record_sha256": digest(json_bytes(prompts[index], newline=False)),
                "oracle_record_sha256": digest(json_bytes(oracles[index], newline=False)),
            }
        )
    manifest = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "source": pin,
        "case_count": 164,
        "cases": cases,
        "files": files,
        "coverage": COVERAGE,
        "metric_profile": "pass_at_1_descriptive",
        "execution_status": "not_executed",
        "conservation": {
            "source_records": 164,
            "prompt_records": 164,
            "oracle_records": 164,
            "excluded_records": 0,
        },
        "digest_domains": {
            "field": "exact UTF8 value",
            "source_record": "sorted compact UTF8 original five-field JSON without newline",
            "code_pair": "CommitPack sorted compact UTF8 old_contents/new_contents JSON without newline",
        },
    }
    bundle["CASE_MANIFEST.json"] = json_bytes(manifest)
    committed_files = {
        **files,
        "CASE_MANIFEST.json": descriptor(
            bundle["CASE_MANIFEST.json"], role=ROLES["CASE_MANIFEST.json"], record_count=1, format="json"
        ),
    }
    bundle["COMMITTED.json"] = json_bytes(
        {
            "schema_version": 1,
            "profile_id": PROFILE_ID,
            "profile_version": PROFILE_VERSION,
            "source": pin,
            "case_count": 164,
            "files": committed_files,
        }
    )
    if sum(len(raw) for raw in bundle.values()) > MAX_BUNDLE_BYTES:
        raise ValueError("complete coding-case bundle bound exceeded")
    return bundle
