"""Pure source-catalog model metadata; actual public index, no weight loading."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

IDS = ("qwen3_5_9b",)


def module() -> Any:
    result = importlib.import_module("amlas.data_eng.model_weight_manifest")
    for name in (
        "expected_model_weight_manifest",
        "validate_expected_model_weight_manifest",
        "validate_expected_weight_index",
    ):
        assert callable(getattr(result, name, None)), "missing-symbol: " + name
    return result


def expected(identifier: str = IDS[0]) -> dict[str, Any]:
    return module().expected_model_weight_manifest(identifier, "1")


@pytest.mark.parametrize("identifier", IDS)
def test_source_catalog_is_expected_only_and_has_real_complete_shard_refs(identifier: str) -> None:
    mod = module()
    value = expected(identifier)
    assert value["schema_version"] == 1
    assert value["catalog_id"] == identifier and value["catalog_version"] == "1"
    assert value["state"] == "expected_bytes_only"
    assert value["acquired"] is False and value["native_verified"] is False
    assert value["trusted"] is False and value["ready"] is False
    assert value["fit"] == "unmeasured"
    assert value["license"] == "apache-2.0"
    assert value["stored_parameters"] >= value["active_language_parameters"]
    files = value["files"]
    weights = [item for item in files if item["role"] == "weight_shard"]
    assert len(weights) == 4
    assert {item["path"] for item in weights} == set(value["index_shard_paths"])
    assert all(item["file_type"] == "regular" and len(item["sha256"]) == 64 for item in files)
    assert max(item["byte_count"] for item in files) <= 8 * 1024**3
    assert sum(item["byte_count"] for item in files) <= 300 * 1024**3
    assert (
        mod.validate_expected_model_weight_manifest(value, catalog_id=identifier, catalog_version="1")
        == value
    )
    value["files"].reverse()
    assert mod.validate_expected_model_weight_manifest(
        value, catalog_id=identifier, catalog_version="1"
    ) == expected(identifier)


@pytest.mark.parametrize(
    "identifier,version",
    [
        ("unknown", "1"),
        ("qwen3_8_flash_next", "1"),
        (IDS[0], "main"),
        (IDS[0], "latest"),
        (IDS[0], "2"),
        (True, "1"),
        (IDS[0], 1),
    ],
)
def test_unknown_or_nonliteral_catalog_selection_never_establishes_approval(
    identifier: Any, version: Any
) -> None:
    with pytest.raises(ValueError):
        module().expected_model_weight_manifest(identifier, version)


@pytest.mark.parametrize(
    "case",
    [
        "revision",
        "repo",
        "license",
        "architecture",
        "model_type",
        "version",
        "stored",
        "active",
        "ready",
        "native",
        "fit",
        "grant",
        "refcount",
    ],
)
def test_manifest_source_or_claim_drift_refuses(case: str) -> None:
    value = expected()
    changes = {
        "revision": ("revision", "a" * 40),
        "repo": ("repo_id", "attacker/model"),
        "license": ("license", "other"),
        "architecture": ("architectures", ["RemoteModel"]),
        "model_type": ("model_type", "unknown"),
        "version": ("catalog_version", "2"),
        "stored": ("stored_parameters", 9000000000),
        "active": ("active_language_parameters", value["stored_parameters"]),
        "ready": ("ready", True),
        "native": ("native_verified", True),
        "fit": ("fit", "fits_gpu"),
        "grant": ("approval", True),
        "refcount": ("index_tensor_count", value["index_tensor_count"] + 1),
    }
    key, change = changes[case]
    value[key] = change
    with pytest.raises(ValueError):
        module().validate_expected_model_weight_manifest(value, catalog_id=IDS[0], catalog_version="1")


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "duplicate",
        "traversal",
        "absolute",
        "backslash",
        "pickle",
        "bin",
        "pt",
        "python",
        "link",
        "digest",
        "upperdigest",
        "count",
        "boolcount",
        "cap",
        "extra_field",
        "role",
        "remote",
    ],
)
def test_exact_file_inventory_and_bounds_refuse_drift(case: str) -> None:
    value = expected()
    item = value["files"][0]
    if case == "missing":
        value["files"].pop()
    elif case == "extra":
        value["files"].append({**deepcopy(item), "path": "extra.safetensors"})
    elif case == "duplicate":
        value["files"][1] = deepcopy(item)
    elif case in {"traversal", "absolute", "backslash", "pickle", "bin", "pt", "python"}:
        item["path"] = {
            "traversal": "../model.safetensors",
            "absolute": "/tmp/model.safetensors",
            "backslash": "folder\\model.safetensors",
            "pickle": "model.pkl",
            "bin": "pytorch_model.bin",
            "pt": "optimizer.pt",
            "python": "model.py",
        }[case]
    elif case == "link":
        item["file_type"] = "symlink"
    elif case == "digest":
        item["sha256"] = "a" * 64
    elif case == "upperdigest":
        item["sha256"] = item["sha256"].upper()
    elif case == "count":
        item["byte_count"] += 1
    elif case == "boolcount":
        item["byte_count"] = True
    elif case == "cap":
        item["byte_count"] = 8 * 1024**3 + 1
    elif case == "extra_field":
        item["link_target"] = "/private"
    elif case == "role":
        item["role"] = "remote_code"
    else:
        value["trust_remote_code"] = True
    with pytest.raises(ValueError):
        module().validate_expected_model_weight_manifest(value, catalog_id=IDS[0], catalog_version="1")


def test_manifest_rejects_dataclass_custom_objects_and_cyclic_non_json_values() -> None:
    @dataclass
    class ForgedApproval:
        trusted: bool = True

    for value in (ForgedApproval(), [expected()], None, {**expected(), "callback": ForgedApproval()}):
        with pytest.raises(ValueError):
            module().validate_expected_model_weight_manifest(value, catalog_id=IDS[0], catalog_version="1")
    value = expected()
    value["files"].append(value)
    with pytest.raises(ValueError):
        module().validate_expected_model_weight_manifest(value, catalog_id=IDS[0], catalog_version="1")


def test_factory_return_mutation_cannot_rewrite_source_catalog() -> None:
    value = expected()
    value["files"].clear()
    assert expected()["files"]


def index_bytes() -> bytes:
    return (Path(__file__).parent / "fixtures/model_weight_manifest/qwen3_5_9b_index.json").read_bytes()


def test_captured_index_original_bytes_bind_hash_size_and_exact_shard_set() -> None:
    raw = index_bytes()
    value = expected()
    ref = next(item for item in value["files"] if item["role"] == "weight_index")
    assert len(raw) == ref["byte_count"] == 79657
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    result = module().validate_expected_weight_index(raw, catalog_id=IDS[0], catalog_version="1")
    assert result == {
        "state": "expected_bytes_only",
        "tensor_count": 775,
        "shard_paths": value["index_shard_paths"],
        "native_verified": False,
    }


@pytest.mark.parametrize(
    "case",
    [
        "other_model",
        "extra_shard",
        "missing_shard",
        "traversal",
        "bin",
        "duplicate",
        "count",
        "append",
        "truncated",
        "oversize",
        "nonbytes",
    ],
)
def test_index_drift_refuses_before_any_weight_loading(case: str) -> None:
    raw: Any = index_bytes()
    identifier = IDS[0]
    if case == "other_model":
        identifier = "qwen3_8_27b"
    elif case == "append":
        raw += b" "
    elif case == "truncated":
        raw = raw[:-1]
    elif case == "oversize":
        raw = b" " * (8 * 1024**2 + 1)
    elif case == "nonbytes":
        raw = bytearray(raw)
    elif case == "duplicate":
        raw = b'{"metadata":{},"metadata":{},"weight_map":{}}'
    else:
        value = json.loads(raw)
        key = next(iter(value["weight_map"]))
        if case == "extra_shard":
            value["weight_map"][key] = "extra.safetensors"
        elif case == "missing_shard":
            shard = value["weight_map"][key]
            value["weight_map"] = {name: path for name, path in value["weight_map"].items() if path != shard}
        elif case == "traversal":
            value["weight_map"][key] = "../model.safetensors"
        elif case == "bin":
            value["weight_map"][key] = "pytorch_model.bin"
        else:
            value["metadata"]["total_size"] += 1
        raw = json.dumps(value).encode()
    with pytest.raises(ValueError):
        module().validate_expected_weight_index(raw, catalog_id=identifier, catalog_version="1")


def test_component_imports_only_pure_stdlib_and_dataset_limit_stays_unchanged() -> None:
    module()
    source = Path(__file__).parents[1] / "src/amlas/data_eng/model_weight_manifest.py"
    tree = ast.parse(source.read_text())
    imports = {node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)}
    assert imports <= {"hashlib", "json", "re"}
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not calls & {"open", "exec", "eval", "__import__", "getattr", "setattr"}
    from amlas.data_eng.hub_dataset import MAX_FILE_BYTES

    assert MAX_FILE_BYTES == 512 * 1024**2
