"""Output authority and original stage components precede data/core imports."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest
from test_qwen_sft_worker import module


def argv(raw):
    return [
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


def test_fixed_output_guard_precedes_binding_decode_and_core(monkeypatch, tmp_path):
    m = module()
    raw = json.dumps(
        {"schema_version": 1, "catalog_id": "qwen3_5_9b", "catalog_version": "1", "recipe_id": "fit_v1"}
    ).encode()
    path = tmp_path / "RUN.json"
    path.write_bytes(raw)
    monkeypatch.setattr(m, "BINDING", path)
    # This ordering/ancestor fixture bypasses provider measurement explicitly;
    # real sealed-FD/current-state behavior has its separate kernel/worker suite.
    for name in m.FORBIDDEN:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(m, "_provider_observation", lambda _stage: {"fixture_provider_guard": "unchanged"})
    monkeypatch.setattr(m, "_guard", lambda: None)
    monkeypatch.setattr(
        m, "_guard_output", lambda: (_ for _ in ()).throw(ValueError("output FD refused")), raising=False
    )
    monkeypatch.setattr(m, "_load_core", lambda: pytest.fail("core loaded before output authority"))
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(sys, "argv", argv(raw))
    with pytest.raises(ValueError, match="output FD"):
        m.main()


def test_original_stage_ancestor_symlink_refuses_before_core(monkeypatch, tmp_path):
    m = module()
    raw = json.dumps(
        {"schema_version": 1, "catalog_id": "qwen3_5_9b", "catalog_version": "1", "recipe_id": "fit_v1"}
    ).encode()
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "RUN.json").write_bytes(raw)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(m, "BINDING", alias / "RUN.json")
    # This ordering/ancestor fixture bypasses provider measurement explicitly;
    # real sealed-FD/current-state behavior has its separate kernel/worker suite.
    for name in m.FORBIDDEN:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(m, "_provider_observation", lambda _stage: {"fixture_provider_guard": "unchanged"})
    monkeypatch.setattr(m, "_guard", lambda: None)
    monkeypatch.setattr(m, "_guard_output", lambda: None, raising=False)
    monkeypatch.setattr(m, "_load_core", lambda: pytest.fail("core loaded through ancestor alias"))
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(sys, "argv", argv(raw))
    with pytest.raises(ValueError):
        m.main()
