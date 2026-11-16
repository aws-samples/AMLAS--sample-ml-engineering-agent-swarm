"""Reload RUN-v2 startup: metadata only, fixed guard fakes and actual pure validator."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_NATIVE_RUN_SHA = "eaaab64b7090ea38173af06bc90eecc3da00e5551ea226efbc951c0fafa42065"  # pragma: allowlist secret -- bounded RUN metadata integrity SHA256, not a credential.


def recovery_run():
    from amlas.training import qwen_sft as core

    external = os.environ.get("AMLAS_QWEN_RECOVERY_RUN_FIXTURE")
    if external:
        raw = Path(external).read_bytes()
        assert len(raw) == 2194 and hashlib.sha256(raw).hexdigest() == _NATIVE_RUN_SHA
        print("PINNED_NATIVE_RUN bytes=2194 sha256=" + _NATIVE_RUN_SHA)
        return raw, json.loads(raw)
    value = {
        "schema_version": 2,
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "recipe_id": "full_v1",
        "model_files": core.expected_model_weight_manifest("qwen3_5_9b", "1")["files"],
        "checkpoint_ref": {"sha256": "b" * 64, "byte_count": 512},
        "operation": "reload",
        "recipe_version": "1",
        "source_commit": "a" * 40,
        "checkpoint_source_commit": "c" * 40,
        "reload_contract_version": core.RELOAD_CONTRACT_VERSION,
        "compatibility_digest": "d" * 64,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), value


def startup(monkeypatch, tmp_path, raw, value, *, operation="reload", recipe=None):
    from amlas.training import qwen_sft as actual_core

    worker = importlib.import_module("amlas.sagemaker_entrypoints.qwen_sft_train")
    for name in worker.FORBIDDEN:
        monkeypatch.delenv(name, raising=False)
    recipe = recipe or value["recipe_id"]
    path = tmp_path / "RUN.json"
    path.write_bytes(raw)
    monkeypatch.setattr(worker, "BINDING", path)
    events, calls, validations = [], [], []
    proof = {"neutral_guard_fixture": True}
    monkeypatch.setattr(worker, "_guard", lambda: events.append("guard") or proof)
    monkeypatch.setattr(worker, "_guard_output", lambda: events.append("output"))
    monkeypatch.setattr(
        worker, "_provider_observation", lambda stage: events.append(stage) or {"fixture": "same"}
    )

    def validate(binding, *, source_commit, recipe_id):
        validations.append((copy.deepcopy(binding), source_commit, recipe_id))
        return actual_core.validate_reload_binding(binding, source_commit=source_commit, recipe_id=recipe_id)

    def load():
        events.append("core_import")
        return SimpleNamespace(
            validate_reload_binding=validate,
            managed_operation=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or {"neutral_operation": True}
            ),
        )

    monkeypatch.setattr(worker, "_load_core", load)
    source = value.get("source_commit", "a" * 40)
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen_sft_train.py",
            "--operation",
            operation,
            "--recipe-id",
            recipe,
            "--recipe-version",
            "1",
            "--source-commit",
            source,
            "--manifest-sha256",
            hashlib.sha256(raw).hexdigest(),
            "--manifest-byte-count",
            str(len(raw)),
        ],
    )
    return worker, events, calls, validations, proof


def test_pinned_recovery_run_v2_reaches_exact_reload_dispatch_after_guards(monkeypatch, tmp_path):
    raw, value = recovery_run()
    worker, events, calls, validations, proof = startup(monkeypatch, tmp_path, raw, value)
    assert worker.main() == {"neutral_operation": True}
    assert events == ["guard", "before_core_import", "output", "core_import", "after_core_import"]
    assert len(validations) == 1 and validations[0] == (value, value["source_commit"], "full_v1")
    assert calls[0][0] == ("reload", "full_v1")
    assert calls[0][1]["stage_binding"] == value and calls[0][1]["privilege_proof"] is proof
    assert calls[0][1]["stage_manifest_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "operation,recipe", [("train", "fit_v1"), ("reload", "full_v1"), ("generate", "humaneval_greedy_v1")]
)
def test_legacy_v1_startup_dispatch_remains_unchanged(monkeypatch, tmp_path, operation, recipe):
    value = {"schema_version": 1, "catalog_id": "qwen3_5_9b", "catalog_version": "1", "recipe_id": recipe}
    raw = json.dumps(value, indent=2).encode()
    worker, _events, calls, validations, _proof = startup(
        monkeypatch, tmp_path, raw, value, operation=operation
    )
    worker.main()
    assert calls[0][0] == (operation, recipe) and not validations


@pytest.mark.parametrize("operation,recipe", [("train", "full_v1"), ("generate", "humaneval_greedy_v1")])
def test_recovery_v2_never_routes_train_or_generate(monkeypatch, tmp_path, operation, recipe):
    _raw, value = recovery_run()
    value["recipe_id"] = recipe
    raw = json.dumps(value).encode()
    worker, _events, calls, _validations, _proof = startup(
        monkeypatch, tmp_path, raw, value, operation=operation
    )
    with pytest.raises(ValueError):
        worker.main()
    assert not calls


@pytest.mark.parametrize(
    "change",
    ["source", "extra", "recipe", "recipe_version", "contract", "operation", "model_files", "schema_bool"],
)
def test_recovery_v2_closed_core_validation_refuses_before_dispatch(monkeypatch, tmp_path, change):
    _raw, value = recovery_run()
    if change == "source":
        # CLI/compiled source stays original while RUN substitutes another consumer.
        value["source_commit"] = "e" * 40
    elif change == "extra":
        value["ordinary_extra"] = "ordinary"
    elif change == "recipe":
        value["recipe_id"] = "fit_v1"
    elif change == "recipe_version":
        value["recipe_version"] = "2"
    elif change == "contract":
        value["reload_contract_version"] = "ordinary.unsupported.v2"
    elif change == "operation":
        value["operation"] = "train"
    elif change == "model_files":
        value["model_files"][0]["byte_count"] += 1
    else:
        value["schema_version"] = True
    raw = json.dumps(value).encode()
    worker, _events, calls, _validations, _proof = startup(
        monkeypatch, tmp_path, raw, value, recipe="full_v1"
    )
    if change == "source":
        source = "a" * 40
        monkeypatch.setenv("AMLAS_SOURCE_COMMIT", source)
        sys.argv[sys.argv.index("--source-commit") + 1] = source
    with pytest.raises(ValueError):
        worker.main()
    assert not calls
