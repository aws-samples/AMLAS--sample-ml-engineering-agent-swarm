"""Pure pinned expected-byte descriptions; never acquisition or authority grants.

Public metadata: Qwen/Qwen3.5-9B, source catalog version1, Apache-2.0 card.
No network, filesystem, framework, model loading, callbacks or tool registration.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

MAX_FILE_BYTES = 8 * 1024**3
MAX_BASE_BYTES = 300 * 1024**3
MAX_INDEX_BYTES = 8 * 1024**2
_CATALOG_ID = "qwen3_5_9b"
_CATALOG_VERSION = "1"
_FILE_KEYS = frozenset({"path", "role", "byte_count", "sha256", "file_type"})
_PATH = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
_DIGEST = re.compile(r"[a-f0-9]{64}")
_TENSOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.]{0,255}")

# Immutable source text, not caller-supplied metadata. Checksums are public pins.
_PINNED_MODEL_JSON = (
    "{\n"
    '  "acquired": false,\n'
    '  "active_language_parameters": 9000000000,\n'
    '  "architectures": [\n'
    '    "Qwen3_5ForConditionalGeneration"\n'
    "  ],\n"
    '  "catalog_id": "qwen3_5_9b",\n'
    '  "catalog_version": "1",\n'
    '  "files": [\n'
    "    {\n"
    '      "byte_count": 7756,\n'
    '      "file_type": "regular",\n'
    '      "path": "chat_template.jinja",\n'
    '      "role": "chat_template",\n'
    '      "sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"\n'  # pragma: allowlist secret
    "    },\n"
    "    {\n"
    '      "byte_count": 3126,\n'
    '      "file_type": "regular",\n'
    '      "path": "config.json",\n'
    '      "role": "model_config",\n'
    '      "sha256": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"\n'  # pragma: allowlist secret
    "    },\n"
    "    {\n"
    '      "byte_count": 5276436216,\n'
    '      "file_type": "regular",\n'
    '      "path": "model.safetensors-00001-of-00004.safetensors",\n'
    '      "role": "weight_shard",\n'
    '      "sha256": "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e"\n'  # pragma: allowlist secret
    "    },\n"
    "    {\n"
    '      "byte_count": 5335161512,\n'
    '      "file_type": "regular",\n'
    '      "path": "model.safetensors-00002-of-00004.safetensors",\n'
    '      "role": "weight_shard",\n'
    '      "sha256": "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330"\n'  # pragma: allowlist secret
    "    },\n"
    "    {\n"
    '      "byte_count": 5368717440,\n'
    '      "file_type": "regular",\n'
    '      "path": "model.safetensors-00003-of-00004.safetensors",\n'
    '      "role": "weight_shard",\n'
    '      "sha256": "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec"\n'  # pragma: allowlist secret
    "    },\n"
    "    {\n"
    '      "byte_count": 3325995712,\n'
    '      "file_type": "regular",\n'
    '      "path": "model.safetensors-00004-of-00004.safetensors",\n'
    '      "role": "weight_shard",\n'
    '      "sha256": "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6"\n'  # pragma: allowlist secret
    "    },\n"
    "    {\n"
    '      "byte_count": 79657,\n'
    '      "file_type": "regular",\n'
    '      "path": "model.safetensors.index.json",\n'
    '      "role": "weight_index",\n'
    '      "sha256": "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5"\n'  # pragma: allowlist secret
    "    },\n"
    "    {\n"
    '      "byte_count": 12807982,\n'
    '      "file_type": "regular",\n'
    '      "path": "tokenizer.json",\n'
    '      "role": "tokenizer",\n'
    '      "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"\n'  # pragma: allowlist secret
    "    },\n"
    "    {\n"
    '      "byte_count": 16710,\n'
    '      "file_type": "regular",\n'
    '      "path": "tokenizer_config.json",\n'
    '      "role": "tokenizer_config",\n'
    '      "sha256": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"\n'  # pragma: allowlist secret
    "    }\n"
    "  ],\n"
    '  "fit": "unmeasured",\n'
    '  "index_metadata": {\n'
    '    "total_size": 19306216416\n'
    "  },\n"
    '  "index_shard_paths": [\n'
    '    "model.safetensors-00001-of-00004.safetensors",\n'
    '    "model.safetensors-00002-of-00004.safetensors",\n'
    '    "model.safetensors-00003-of-00004.safetensors",\n'
    '    "model.safetensors-00004-of-00004.safetensors"\n'
    "  ],\n"
    '  "index_tensor_count": 775,\n'
    '  "license": "apache-2.0",\n'
    '  "model_type": "qwen3_5",\n'
    '  "native_verified": false,\n'
    '  "ready": false,\n'
    '  "repo_id": "Qwen/Qwen3.5-9B",\n'
    '  "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",\n'  # pragma: allowlist secret
    '  "schema_version": 1,\n'
    '  "state": "expected_bytes_only",\n'
    '  "stored_parameters": 9653104368,\n'
    '  "trusted": false,\n'
    '  "weights_total_bytes": 19306310880\n'
    "}\n"
)


def _select(catalog_id: Any, catalog_version: Any) -> dict[str, Any]:
    if (
        type(catalog_id) is not str
        or type(catalog_version) is not str
        or catalog_id != _CATALOG_ID
        or catalog_version != _CATALOG_VERSION
    ):
        raise ValueError("unknown source-owned expected model catalog")
    return json.loads(_PINNED_MODEL_JSON)


def expected_model_weight_manifest(catalog_id: str, catalog_version: str) -> dict[str, Any]:
    """Return a fresh description of expected bytes, without trust/readiness."""
    return _select(catalog_id, catalog_version)


def _admit_json(value: Any, depth: int, remaining: list[int]) -> None:
    remaining[0] -= 1
    if remaining[0] < 0 or depth > 12:
        raise ValueError("expected model description exceeds structural bounds")
    kind = type(value)
    if value is None or kind is bool:
        return
    if kind is int and 0 <= value <= 2**64 - 1:
        return
    if kind is str and len(value) <= 2048 and not any(ord(char) < 32 for char in value):
        return
    if kind is list:
        if len(value) > 256:
            raise ValueError("expected model description list exceeds bounds")
        for item in value:
            _admit_json(item, depth + 1, remaining)
        return
    if kind is dict:
        if len(value) > 256:
            raise ValueError("expected model description object exceeds bounds")
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("expected model description keys must be literal strings")
            _admit_json(key, depth + 1, remaining)
            _admit_json(item, depth + 1, remaining)
        return
    raise ValueError("expected model description requires builtin JSON values")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def validate_expected_model_weight_manifest(
    value: Any, *, catalog_id: str, catalog_version: str
) -> dict[str, Any]:
    """Require exact source-catalog descriptors; never validate actual weight bytes."""
    expected = _select(catalog_id, catalog_version)
    _admit_json(value, 0, [4096])
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("expected model description fields differ from source catalog")
    files = value["files"]
    if type(files) is not list or len(files) != len(expected["files"]):
        raise ValueError("expected model file count differs")
    names, total = set(), 0
    for item in files:
        if type(item) is not dict or set(item) != _FILE_KEYS:
            raise ValueError("expected model file descriptor fields differ")
        name, size, digest = item["path"], item["byte_count"], item["sha256"]
        if (
            type(name) is not str
            or not _PATH.fullmatch(name)
            or name in names
            or type(size) is not int
            or not 1 <= size <= MAX_FILE_BYTES
            or type(digest) is not str
            or not _DIGEST.fullmatch(digest)
            or item["file_type"] != "regular"
        ):
            raise ValueError("expected model file identity or bounds differ")
        names.add(name)
        total += size
    if total > MAX_BASE_BYTES:
        raise ValueError("expected model total bytes exceed source bound")
    # Canonical JSON preserves boolean/integer distinction and contains no callbacks.
    normalized = json.loads(_canonical(value))
    normalized["files"] = sorted(normalized["files"], key=lambda item: item["path"])
    if _canonical(normalized) != _canonical(expected):
        raise ValueError("expected model description differs from source catalog")
    return expected


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError("duplicate expected model index field")
        result[name] = value
    return result


def _constant(value: str) -> Any:
    raise ValueError("nonfinite expected model index number")


def validate_expected_weight_index(raw: bytes, *, catalog_id: str, catalog_version: str) -> dict[str, Any]:
    """Check supplied index bytes only; no shard acquisition/loading/attestation."""
    expected = _select(catalog_id, catalog_version)
    ref = next(item for item in expected["files"] if item["role"] == "weight_index")
    if (
        type(raw) is not bytes
        or not 0 < len(raw) <= MAX_INDEX_BYTES
        or len(raw) != ref["byte_count"]
        or hashlib.sha256(raw).hexdigest() != ref["sha256"]
    ):
        raise ValueError("expected model index original byte binding differs")
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    if (
        type(value) is not dict
        or set(value) != {"metadata", "weight_map"}
        or _canonical(value["metadata"]) != _canonical(expected["index_metadata"])
        or type(value["weight_map"]) is not dict
        or len(value["weight_map"]) != expected["index_tensor_count"]
    ):
        raise ValueError("expected model index structure differs")
    mapping = value["weight_map"]
    if any(
        type(name) is not str
        or not _TENSOR.fullmatch(name)
        or type(path) is not str
        or not path.endswith(".safetensors")
        or not _PATH.fullmatch(path)
        for name, path in mapping.items()
    ) or set(mapping.values()) != set(expected["index_shard_paths"]):
        raise ValueError("expected model index shard set differs")
    return {
        "state": "expected_bytes_only",
        "tensor_count": len(mapping),
        "shard_paths": expected["index_shard_paths"],
        "native_verified": False,
    }
