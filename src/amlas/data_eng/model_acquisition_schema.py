"""Strict producer metadata, never model parsing, authority grants or admission."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .inspection_schema import decode_object, validate_native_privilege_proof, validate_output_directory_proof
from .model_weight_manifest import expected_model_weight_manifest

MAX_METADATA_BYTES = 262144
MAX_METADATA_DEPTH = 12
BINDING_KEYS = frozenset(
    {"job_name", "runtime_session_id", "deployment_id", "deployment_prefix", "image_uri", "source_commit"}
)
STATE = {"acquired": True, "prepared": False, "trained": False, "trusted": False, "ready": False}


def validate_provider_inventory(value: Any) -> dict[str, Any]:
    """Closed earlier OS observation; never current absence or caller authority."""
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "inventory_kind", "observation", "providers"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["inventory_kind"] != "raw_model_provider_absence_before_drop_v1"
        or value["observation"] != "earlier_original_component_absence"
        or type(value["providers"]) is not list
        or len(value["providers"]) != 4
    ):
        raise ValueError("model earlier provider inventory differs")
    for index, record in enumerate(value["providers"]):
        if (
            type(record) is not dict
            or set(record) != {"provider_index", "missing_component_index", "ancestors"}
            or type(record["provider_index"]) is not int
            or record["provider_index"] != index
            or type(record["missing_component_index"]) is not int
            or not 1 <= record["missing_component_index"] <= (4 if index < 2 else 3)
            or type(record["ancestors"]) is not list
            or len(record["ancestors"]) != record["missing_component_index"]
        ):
            raise ValueError("model earlier provider ordering differs")
        for component, node in enumerate(record["ancestors"]):
            if (
                type(node) is not dict
                or set(node) != {"component_index", "device", "inode", "uid", "gid", "mode"}
                or any(type(item) is not int or not 0 <= item < 2**64 for item in node.values())
                or node["component_index"] != component
                or node["inode"] == 0
                or node["mode"] > 0o7777
            ):
                raise ValueError("model earlier provider ancestor differs")
    return value


def _validate_credential_observation(value: Any) -> None:
    legacy = {
        "environment_names": [],
        "provider_files_present": False,
        "metadata_endpoint_reachable": False,
    }
    if type(value) is dict and value == legacy:
        if any(
            type(value[key]) is not bool for key in ("provider_files_present", "metadata_endpoint_reachable")
        ):
            raise ValueError("model credential boolean observation differs")
        return
    keys = {
        "observation_profile",
        "environment_names",
        "earlier_inventory_sha256",
        "earlier_inventory",
        "provider_path_observations",
        "metadata_endpoint_reachable",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or value["observation_profile"] != "model_provider_inventory_v2"
        or value["environment_names"] != []
        or type(value["environment_names"]) is not list
        or value["metadata_endpoint_reachable"] is not False
        or type(value["earlier_inventory_sha256"]) is not str
        or not re.fullmatch(r"[a-f0-9]{64}", value["earlier_inventory_sha256"])
        or type(value["provider_path_observations"]) is not list
        or len(value["provider_path_observations"]) != 4
    ):
        raise ValueError("model sealed credential observation differs")
    validate_provider_inventory(value["earlier_inventory"])
    for index, record in enumerate(value["provider_path_observations"]):
        if (
            type(record) is not dict
            or set(record) != {"provider_index", "current_state"}
            or type(record["provider_index"]) is not int
            or record["provider_index"] != index
            or record["current_state"] not in ("current_absent", "denied_after_drop")
        ):
            raise ValueError("model current provider observation differs")


def encoded(value: Any) -> bytes:
    raw = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
    ).encode()
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("model metadata byte bound exceeded")
    return raw


def expected_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def validate_bindings(value: Any) -> dict[str, str]:
    if (
        type(value) is not dict
        or set(value) != BINDING_KEYS
        or any(type(v) is not str for v in value.values())
    ):
        raise ValueError("model job binding fields differ")
    for name in ("job_name", "runtime_session_id", "deployment_prefix"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value[name]):
            raise ValueError("model job namespace differs")
    if (
        len(value["job_name"]) > 63
        or not re.fullmatch(r"[a-f0-9]{40}", value["source_commit"])
        or not re.fullmatch(r"[a-f0-9-]{36}", value["deployment_id"])
        or not re.fullmatch(
            r"[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?/[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}",
            value["image_uri"],
        )
    ):
        raise ValueError("model source/image/deployment binding differs")
    return value


def validate_evidence(value: Any) -> dict[str, Any]:
    keys = {
        "privilege_profile",
        "privilege_proof",
        "output_directory_proof",
        "credential_observation",
        "guest_claim_attested",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or value["privilege_profile"] != "sagemaker_native_ceiling_v1"
        or value["guest_claim_attested"] is not False
    ):
        raise ValueError("model guest evidence fields differ")
    validate_native_privilege_proof(value["privilege_proof"])
    validate_output_directory_proof(value["output_directory_proof"])
    _validate_credential_observation(value["credential_observation"])
    return value


def context(
    catalog_id: str, catalog_version: str, bindings: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = expected_model_weight_manifest(catalog_id, catalog_version)
    validate_bindings(bindings)
    return expected, {
        "schema_version": 1,
        "catalog_id": catalog_id,
        "catalog_version": catalog_version,
        **bindings,
        "expected_manifest_sha256": expected_digest(expected),
    }


def lineage(catalog_id: str, catalog_version: str, bindings: dict[str, str]) -> dict[str, Any]:
    expected, base = context(catalog_id, catalog_version, bindings)
    return {**base, **STATE, "expected_manifest": expected, "files": expected["files"]}


def receipt(
    catalog_id: str,
    catalog_version: str,
    bindings: dict[str, str],
    evidence: dict[str, Any],
    lineage_raw: bytes,
) -> dict[str, Any]:
    _, base = context(catalog_id, catalog_version, bindings)
    return {
        **base,
        **STATE,
        "evidence": validate_evidence(evidence),
        "lineage": {
            "path": "LINEAGE.json",
            "sha256": hashlib.sha256(lineage_raw).hexdigest(),
            "byte_count": len(lineage_raw),
        },
    }


def commit(
    catalog_id: str, catalog_version: str, bindings: dict[str, str], lineage_raw: bytes, receipt_raw: bytes
) -> dict[str, Any]:
    expected, base = context(catalog_id, catalog_version, bindings)
    return {
        **base,
        "lineage": {
            "path": "LINEAGE.json",
            "sha256": hashlib.sha256(lineage_raw).hexdigest(),
            "byte_count": len(lineage_raw),
        },
        "receipt": {
            "path": "MODEL_ACQUIRED.json",
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "byte_count": len(receipt_raw),
        },
        "payload_paths": sorted(item["path"] for item in expected["files"]),
    }


def _admit_metadata_depth(raw: bytes) -> None:
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_METADATA_BYTES:
        raise ValueError("model metadata byte bound exceeded")
    text = raw.decode(json.detect_encoding(raw), "surrogatepass")
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
            if len(stack) >= MAX_METADATA_DEPTH:
                raise ValueError("model metadata structural depth exceeds bounds")
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if not stack or stack.pop() != char:
                raise ValueError("model metadata structure differs")
    if stack or quoted:
        raise ValueError("model metadata structure incomplete")


def decode_metadata(
    raw: bytes, *, kind: str, catalog_id: str, catalog_version: str, bindings: dict[str, str]
) -> dict[str, Any]:
    _admit_metadata_depth(raw)
    value = decode_object(raw, limit=MAX_METADATA_BYTES)
    expected, base = context(catalog_id, catalog_version, bindings)
    extras = {
        "lineage": set(STATE) | {"expected_manifest", "files"},
        "receipt": set(STATE) | {"evidence", "lineage"},
        "commit": {"lineage", "receipt", "payload_paths"},
    }
    if (
        kind not in extras
        or set(value) != set(base) | extras[kind]
        or any(type(value.get(key)) is not type(v) or value.get(key) != v for key, v in base.items())
    ):
        raise ValueError("model metadata source/provenance fields differ")
    if kind != "commit" and any(value[name] is not flag for name, flag in STATE.items()):
        raise ValueError("model acquisition metadata claims trust/preparation/training")
    if kind == "lineage":
        if encoded(value["expected_manifest"]) != encoded(expected) or encoded(value["files"]) != encoded(
            expected["files"]
        ):
            raise ValueError("model acquired inventory differs from source catalog")
    if kind == "receipt":
        validate_evidence(value["evidence"])
    for name, path in (("lineage", "LINEAGE.json"), ("receipt", "MODEL_ACQUIRED.json")):
        if kind == "lineage" or name not in value:
            continue
        ref = value[name]
        if (
            type(ref) is not dict
            or set(ref) != {"path", "sha256", "byte_count"}
            or ref["path"] != path
            or type(ref["sha256"]) is not str
            or not re.fullmatch(r"[a-f0-9]{64}", ref["sha256"])
            or type(ref["byte_count"]) is not int
            or not 0 < ref["byte_count"] <= MAX_METADATA_BYTES
        ):
            raise ValueError("model metadata reference differs")
    if kind == "commit" and value["payload_paths"] != sorted(item["path"] for item in expected["files"]):
        raise ValueError("model commit payload set differs")
    return value
