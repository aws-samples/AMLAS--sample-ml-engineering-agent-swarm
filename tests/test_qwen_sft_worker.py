"""Fixed managed operation refuses before ML imports; no AWS/GPU effects."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


def module():
    m = importlib.import_module("amlas.sagemaker_entrypoints.qwen_sft_train")
    for name in ("main", "_guard", "_load_core"):
        assert callable(getattr(m, name, None)), "missing-symbol: " + name
    return m


def test_fixed_worker_identity_gate_precedes_core(monkeypatch):
    m = module()
    monkeypatch.setattr(m.os, "getresuid", lambda: (0, 0, 0))
    monkeypatch.setattr(m, "_load_core", lambda: pytest.fail("ML imported before identity"))
    with pytest.raises(ValueError):
        m._guard()


@pytest.mark.parametrize(
    "name",
    [
        "AWS_ACCESS_KEY_ID",
        "AWS_PROFILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "HTTP_PROXY",
        "https_proxy",
        "AWS_EC2_METADATA_DISABLED",
    ],
)
def test_provider_hints_never_become_fake_absence(monkeypatch, name):
    m = module()
    monkeypatch.setattr(m.os, "getresuid", lambda: (10001,) * 3)
    monkeypatch.setattr(m.os, "getresgid", lambda: (10001,) * 3)
    monkeypatch.setattr(m.os, "getgroups", lambda: [])
    monkeypatch.setenv(name, "private-fixture-do-not-log")
    with pytest.raises(ValueError) as error:
        m._guard()
    assert "private-fixture-do-not-log" not in str(error.value)


@pytest.mark.parametrize("case", ["unknown_operation", "unknown_recipe", "source", "extra_grant"])
def test_fixed_operation_binding_refuses_before_core(monkeypatch, case):
    m = module()
    monkeypatch.setattr(m, "_guard", lambda: None)
    monkeypatch.setattr(m, "_load_core", lambda: pytest.fail("core loaded before binding"))
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "a" * 40)
    args = [
        "qwen_sft_train.py",
        "--operation",
        "train",
        "--recipe-id",
        "fit_v1",
        "--source-commit",
        "a" * 40,
        "--recipe-version",
        "1",
        "--manifest-sha256",
        "b" * 64,
        "--manifest-byte-count",
        "1",
    ]
    if case == "unknown_operation":
        args[2] = "caller_code"
    elif case == "unknown_recipe":
        args[4] = "auto"
    elif case == "source":
        args[6] = "b" * 40
    else:
        args += ["--approved", "true"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises((ValueError, SystemExit)):
        m.main()


def test_fixed_startup_manifest_binding_is_hash_checked_before_core(monkeypatch, tmp_path):
    import hashlib
    import json

    m = module()
    # This positive fixture models the fixed child after provider environment clearing.
    for name in m.FORBIDDEN:
        monkeypatch.delenv(name, raising=False)
    assert not any(name in m.os.environ for name in m.FORBIDDEN)
    binding = {"schema_version": 1, "catalog_id": "qwen3_5_9b", "catalog_version": "1", "recipe_id": "fit_v1"}
    raw = json.dumps(binding).encode()
    path = tmp_path / "RUN.json"
    path.write_bytes(raw)
    monkeypatch.setattr(m, "BINDING", path)
    monkeypatch.setattr(m, "_guard", lambda: None)
    monkeypatch.setattr(m, "_guard_output", lambda: None)
    # This manifest transport fixture explicitly bypasses the new provider guard.
    # Actual sealed-FD/provider controls are in test_qwen_sft_provider_inventory.
    monkeypatch.setattr(m, "_provider_observation", lambda _stage: {"fixture_provider_guard": "unchanged"})
    called = []
    monkeypatch.setattr(
        m,
        "_load_core",
        lambda: SimpleNamespace(managed_operation=lambda *args, **kwargs: called.append((args, kwargs))),
    )
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "a" * 40)
    args = [
        "qwen_sft_train.py",
        "--operation",
        "train",
        "--recipe-id",
        "fit_v1",
        "--recipe-version",
        "1",
        "--source-commit",
        "a" * 40,
        "--manifest-sha256",
        hashlib.sha256(raw).hexdigest(),
        "--manifest-byte-count",
        str(len(raw)),
    ]
    monkeypatch.setattr(sys, "argv", args)
    m.main()
    assert len(called) == 1 and called[0][1]["stage_binding"] == binding
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        m.main()
    assert len(called) == 1


def test_exact_raw_startup_manifest_digest_reaches_core(monkeypatch, tmp_path):
    import hashlib
    import json

    m = module()
    # This positive fixture models the fixed child after provider environment clearing.
    for name in m.FORBIDDEN:
        monkeypatch.delenv(name, raising=False)
    assert not any(name in m.os.environ for name in m.FORBIDDEN)
    binding = {"schema_version": 1, "catalog_id": "qwen3_5_9b", "catalog_version": "1", "recipe_id": "fit_v1"}
    raw = json.dumps(binding, indent=2).encode()
    path = tmp_path / "RUN.json"
    path.write_bytes(raw)
    monkeypatch.setattr(m, "BINDING", path)
    monkeypatch.setattr(m, "_guard", lambda: None)
    monkeypatch.setattr(m, "_guard_output", lambda: None)
    # This manifest transport fixture explicitly bypasses the new provider guard.
    # Actual sealed-FD/provider controls are in test_qwen_sft_provider_inventory.
    monkeypatch.setattr(m, "_provider_observation", lambda _stage: {"fixture_provider_guard": "unchanged"})
    calls = []
    monkeypatch.setattr(
        m, "_load_core", lambda: SimpleNamespace(managed_operation=lambda *a, **kw: calls.append(kw))
    )
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen_sft_train.py",
            "--operation",
            "train",
            "--recipe-id",
            "fit_v1",
            "--recipe-version",
            "1",
            "--source-commit",
            "a" * 40,
            "--manifest-sha256",
            hashlib.sha256(raw).hexdigest(),
            "--manifest-byte-count",
            str(len(raw)),
        ],
    )
    m.main()
    assert calls[0].get("stage_manifest_sha256") == hashlib.sha256(raw).hexdigest()
