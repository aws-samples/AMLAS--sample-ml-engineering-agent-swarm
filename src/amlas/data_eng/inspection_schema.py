"""Small strict JSON contract shared by the fixed guest and its parent verifier."""

from __future__ import annotations

import json
import math
import re
from typing import Any

MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_SUMMARY_BYTES = 65536
MAX_RECEIPT_BYTES = 16384
MAX_JSON_DEPTH = 16
MAX_ROWS = 4096
MAX_COLUMNS = 128
MAX_DECODED_BYTES = 256 * 1024 * 1024
FORMATS = frozenset({"csv", "json", "jsonl", "parquet"})
SHA256 = re.compile(r"[a-f0-9]{64}")
SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "format",
        "byte_count",
        "sha256",
        "rows_scanned",
        "rows_total",
        "columns",
        "dtypes",
        "missing_counts",
        "numeric_stats",
        "target_column",
        "target_distribution",
        "sample_rows",
        "sample_truncated",
        "coverage",
        "untrusted_data",
        "prepared",
        "trained",
    }
)
RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "job_name",
        "runtime_session_id",
        "deployment_id",
        "deployment_prefix",
        "image_uri",
        "source_commit",
        "source_version_id",
        "staged_version_id",
        "sha256",
        "byte_count",
        "format",
        "summary_sha256",
        "guest_evidence",
        "prepared",
        "trained",
    }
)
GUEST_KEYS = frozenset(
    {
        "credential_present",
        "provider_method",
        "credential_env_names",
        "metadata_reachable",
        "network_reachable",
        "privilege_proof",
        "output_directory_proof",
    }
)


PRIVILEGE_KEYS = frozenset(
    {"res_uid", "res_gid", "groups", "cap_eff", "cap_prm", "cap_inh", "cap_amb", "cap_bnd", "no_new_privs"}
)


def validate_privilege_proof(value: Any) -> dict[str, Any]:
    """Admit only a complete actual permanent-drop proof from the fixed worker."""
    if not isinstance(value, dict) or set(value) != PRIVILEGE_KEYS:
        raise ValueError("incomplete process privilege proof")
    for name in ("res_uid", "res_gid"):
        ids = value[name]
        if (
            not isinstance(ids, list)
            or len(ids) != 3
            or any(type(item) is not int or item != 10001 for item in ids)
        ):
            raise ValueError("process privilege identity differs from fixed worker")
    if not isinstance(value["groups"], list) or value["groups"]:
        raise ValueError("process privilege supplementary groups remain")
    for name in ("cap_eff", "cap_prm", "cap_inh", "cap_amb", "cap_bnd"):
        if type(value[name]) is not int or value[name] != 0:
            raise ValueError("process privilege capabilities remain")
    if type(value["no_new_privs"]) is not int or value["no_new_privs"] != 1:
        raise ValueError("process privilege no-new-privileges is missing")
    return value


ZERO_BOUNDING_PROFILE = "zero_bounding_v1"
NATIVE_CEILING_PROFILE = "sagemaker_native_ceiling_v1"
NATIVE_CAPABILITY_CEILING = 0x200404FB
NATIVE_PRIVILEGE_KEYS = PRIVILEGE_KEYS | {"keep_caps", "seccomp", "fs_uid", "fs_gid", "securebits"}


def validate_native_privilege_proof(value: Any) -> dict[str, Any]:
    """Validate descriptive native proof, never grant authority from a guest claim."""
    if not isinstance(value, dict) or set(value) != NATIVE_PRIVILEGE_KEYS:
        raise ValueError("incomplete native process privilege proof")
    # Reuse the unchanged strict proof for active capabilities/IDs/NNP only.
    strict = {name: value[name] for name in PRIVILEGE_KEYS}
    strict["cap_bnd"] = 0
    validate_privilege_proof(strict)
    bounding = value["cap_bnd"]
    if type(bounding) is not int or bounding < 0 or bounding & ~NATIVE_CAPABILITY_CEILING:
        raise ValueError("native process privilege bounding ceiling differs")
    for name, expected in (("keep_caps", 0), ("seccomp", 2), ("fs_uid", 10001), ("fs_gid", 10001)):
        if type(value[name]) is not int or value[name] != expected:
            raise ValueError("native process privilege filesystem or kernel control differs")
    securebits = value["securebits"]
    if type(securebits) is not int or not 0 <= securebits <= 0xFF or securebits & 0x14:
        raise ValueError("native process privilege securebits differ")
    return value


OUTPUT_DIRECTORY_KEYS = frozenset({"fd", "uid", "gid", "mode", "device", "inode", "lease_active"})


def validate_output_directory_proof(value: Any) -> dict[str, Any]:
    """Current descriptive evidence; never a caller-selected authority grant."""
    if not isinstance(value, dict) or set(value) != OUTPUT_DIRECTORY_KEYS:
        raise ValueError("incomplete output directory proof")
    for name, expected in (("fd", 3), ("uid", 10001), ("gid", 10001), ("mode", 0o700)):
        if type(value[name]) is not int or value[name] != expected:
            raise ValueError("fixed output directory authority differs")
    for name, minimum in (("device", 0), ("inode", 1)):
        if type(value[name]) is not int or not minimum <= value[name] <= 2**64 - 1:
            raise ValueError("output directory identity exceeds its bounds")
    if value["lease_active"] is not True:
        raise ValueError("output directory current lease is missing")
    return value


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _admit_json_depth(text: str) -> None:
    """Bound containers before decoding, excluding quoted/escaped delimiters."""
    stack: list[str] = []
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            if len(stack) >= MAX_JSON_DEPTH:
                raise ValueError("JSON structural depth exceeds inspection bounds")
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if not stack or stack.pop() != char:
                raise ValueError("mismatched JSON structure")
    if stack or quoted:
        raise ValueError("incomplete JSON structure")


def decode_object(data: bytes, *, limit: int) -> dict[str, Any]:
    if not isinstance(data, bytes) or not 0 < len(data) <= limit:
        raise ValueError("bounded JSON object required")

    def reject_constant(value: str) -> Any:
        raise ValueError("nonfinite JSON number")

    # Preserve the standard byte decoder's UTF-8/16/32 and BOM handling.
    text = data.decode(json.detect_encoding(data), "surrogatepass")
    _admit_json_depth(text)
    value = json.loads(text, object_pairs_hook=_pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def target_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or any(ord(c) < 32 for c in value):
        raise ValueError("target column must be bounded literal text")
    return value


def _integer(value: Any, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside inspection bounds")


def _cell(value: Any, depth: int = 0) -> None:
    if depth > 5:
        raise ValueError("sample depth exceeds bounds")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite sample")
        return
    if isinstance(value, str) and len(value) <= 512:
        return
    if isinstance(value, list) and len(value) <= 10:
        for item in value:
            _cell(item, depth + 1)
        return
    if isinstance(value, dict) and len(value) <= 10:
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError("sample key exceeds bounds")
            _cell(item, depth + 1)
        return
    raise ValueError("sample value exceeds fixed bounds")


def validate_summary(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != SUMMARY_KEYS or value.get("schema_version") != 1:
        raise ValueError("unknown inspection summary schema")
    if (
        value["format"] not in FORMATS
        or not isinstance(value["sha256"], str)
        or not SHA256.fullmatch(value["sha256"])
    ):
        raise ValueError("invalid inspected input identity")
    _integer(value["byte_count"], 1, MAX_INPUT_BYTES)
    _integer(value["rows_scanned"], 0, MAX_ROWS)
    if value["rows_total"] is not None:
        _integer(value["rows_total"], value["rows_scanned"], 2**63 - 1)
    columns = value["columns"]
    if (
        not isinstance(columns, list)
        or len(columns) > MAX_COLUMNS
        or any(not isinstance(item, str) or not 1 <= len(item) <= 128 for item in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("invalid bounded inspection columns")
    for field in ("dtypes", "missing_counts"):
        if not isinstance(value[field], dict) or set(value[field]) != set(columns):
            raise ValueError("column summary mismatch")
    for dtype in value["dtypes"].values():
        if not isinstance(dtype, str) or len(dtype) > 128:
            raise ValueError("invalid dtype description")
    for count in value["missing_counts"].values():
        _integer(count, 0, value["rows_scanned"])
    stats = value["numeric_stats"]
    if not isinstance(stats, dict) or not set(stats) <= set(columns):
        raise ValueError("invalid numeric summary")
    for item in stats.values():
        if not isinstance(item, dict) or set(item) != {"min", "max", "mean"}:
            raise ValueError("invalid numeric summary fields")
        if any(
            number is not None and (type(number) not in (int, float) or not math.isfinite(number))
            for number in item.values()
        ):
            raise ValueError("invalid numeric summary value")
    target = target_text(value["target_column"])
    distribution = value["target_distribution"]
    if distribution is not None:
        if target not in columns or not isinstance(distribution, dict) or len(distribution) > 10:
            raise ValueError("invalid sampled target distribution")
        for key, count in distribution.items():
            if not isinstance(key, str) or len(key) > 512:
                raise ValueError("invalid target category")
            _integer(count, 0, value["rows_scanned"])
        if sum(distribution.values()) > value["rows_scanned"]:
            raise ValueError("target distribution exceeds sampled rows")
    sample = value["sample_rows"]
    if not isinstance(sample, list) or len(sample) > min(3, value["rows_scanned"]):
        raise ValueError("sample row bound exceeded")
    for row in sample:
        if not isinstance(row, dict) or not set(row) <= set(columns):
            raise ValueError("sample column mismatch")
        for item in row.values():
            _cell(item)
    if type(value["sample_truncated"]) is not bool or value["coverage"] not in {"complete", "sampled"}:
        raise ValueError("invalid coverage declaration")
    if value["coverage"] == "complete" and value["rows_total"] != value["rows_scanned"]:
        raise ValueError("complete coverage lacks a full row count")
    if value["untrusted_data"] is not True or value["prepared"] is not False or value["trained"] is not False:
        raise ValueError("inspection cannot assert preparation or trust")
    return value


def decode_summary(data: bytes) -> dict[str, Any]:
    return validate_summary(decode_object(data, limit=MAX_SUMMARY_BYTES))


def decode_receipt(data: bytes, *, expected_profile: str = ZERO_BOUNDING_PROFILE) -> dict[str, Any]:
    value = decode_object(data, limit=MAX_RECEIPT_BYTES)
    if expected_profile == ZERO_BOUNDING_PROFILE:
        version, keys = 1, RECEIPT_KEYS
    elif expected_profile == NATIVE_CEILING_PROFILE:
        version, keys = 2, RECEIPT_KEYS | {"privilege_profile"}
        if value.get("privilege_profile") != NATIVE_CEILING_PROFILE:
            raise ValueError("native inspection receipt profile differs")
    else:
        raise ValueError("unknown source-owned inspection profile")
    if (
        set(value) != keys
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != version
    ):
        raise ValueError("unknown inspection receipt schema")
    if value["prepared"] is not False or value["trained"] is not False:
        raise ValueError("inspection receipt claims preparation")
    if not isinstance(value["guest_evidence"], dict) or set(value["guest_evidence"]) != GUEST_KEYS:
        raise ValueError("incomplete guest privilege evidence")
    evidence = value["guest_evidence"]
    if expected_profile == NATIVE_CEILING_PROFILE:
        validate_native_privilege_proof(evidence["privilege_proof"])
    else:
        validate_privilege_proof(evidence["privilege_proof"])
    validate_output_directory_proof(evidence["output_directory_proof"])
    if (
        evidence["credential_present"] is not False
        or evidence["provider_method"] is not None
        or evidence["credential_env_names"] != []
        or evidence["metadata_reachable"] is not False
        or evidence["network_reachable"] is not False
    ):
        raise ValueError("guest credential or network evidence failed")
    return value
