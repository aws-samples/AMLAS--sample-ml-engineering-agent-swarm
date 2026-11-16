"""Pure bounded case-bundle replay from retained original fields, no native trust."""

from __future__ import annotations

from typing import Any

from .coding_case_codec import FILES, freeze_humaneval_cases
from .coding_case_profile import CASE_IDS, MAX_BUNDLE_BYTES, CodingCaseSource, source_metadata


def _object(raw: bytes) -> dict:
    from .inspection_schema import decode_object

    return decode_object(raw, limit=MAX_BUNDLE_BYTES)


def _records(raw: bytes, keys: set[str]) -> list[dict]:
    # Bound total bytes before allocation; exactly164 records are required, no firstN.
    lines = raw.splitlines()
    if len(lines) != 164 or any(not line for line in lines):
        raise ValueError("complete coding-case JSONL record count differs")
    values = []
    for index, line in enumerate(lines):
        value = _object(line)
        if (
            set(value) != keys
            or any(type(item) is not str for item in value.values())
            or value["case_id"] != CASE_IDS[index]
        ):
            raise ValueError("coding-case role fields/population/order differ")
        values.append(value)
    return values


def validate_coding_case_bundle(bundle: Any, *, source: CodingCaseSource) -> dict:
    """Replay all semantic projections; caller independently authenticates raw source."""
    source_metadata(source)
    if (
        type(bundle) is not dict
        or set(bundle) != set(FILES)
        or any(type(raw) is not bytes or not raw for raw in bundle.values())
        or sum(len(raw) for raw in bundle.values()) > MAX_BUNDLE_BYTES
    ):
        raise ValueError("closed bounded five-file coding-case bundle required")
    # Parse every envelope with duplicate/nonfinite rejection before byte replay.
    for name in ("EXCLUSIONS.json", "CASE_MANIFEST.json", "COMMITTED.json"):
        _object(bundle[name])
    prompts = _records(bundle["PROMPTS.jsonl"], {"case_id", "prompt", "entry_point"})
    oracles = _records(bundle["ORACLES.jsonl"], {"case_id", "test", "canonical_solution"})
    original = [
        {
            "task_id": prompt["case_id"],
            "prompt": prompt["prompt"],
            "entry_point": prompt["entry_point"],
            "test": oracle["test"],
            "canonical_solution": oracle["canonical_solution"],
        }
        for prompt, oracle in zip(prompts, oracles, strict=True)
    ]
    # Recompute ordered hashes/roles/counts/exclusions from retained original values.
    # A coherently replaced whole bundle cannot authenticate its rawParquet origin.
    expected = freeze_humaneval_cases(original, source=source)
    if any(bundle[name] != expected[name] for name in FILES):
        raise ValueError("coding-case original-field replay/projection differs")
    return {
        "profile_id": "openai.humaneval.164.hub.parquet.v1",
        "profile_version": "1",
        "case_count": 164,
        "metric_profile": "pass_at_1_descriptive",
        "execution_status": "not_executed",
        "raw_bytes_origin": "caller_integrity_gate_required",
    }
