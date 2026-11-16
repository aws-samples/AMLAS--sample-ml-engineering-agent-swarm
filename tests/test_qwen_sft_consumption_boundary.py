"""Source-only held input authority; native ownership remains a separate real gate."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest


def module():
    return importlib.import_module("amlas.training.qwen_sft")


def test_native_inputs_refuse_actual_worker_owned_directory(tmp_path, monkeypatch):
    m = module()
    assert callable(getattr(m, "_require_native_readonly_directory", None)), (
        "missing-symbol: native readonly input guard"
    )
    # Full account CI runs as root; construct a genuinely non-root-owned input
    # rather than rely on the test process's incidental UID or ancestor ownership.
    import os

    owned = tmp_path / "worker-owned"
    owned.mkdir()
    if os.geteuid() == 0:
        os.chown(owned, 10001, -1)
    assert owned.stat().st_uid != 0
    with pytest.raises(ValueError):
        m._require_native_readonly_directory(owned)


def fixture(tmp_path, monkeypatch):
    m = module()
    assert callable(getattr(m, "_HeldNativeInputs", None)), "missing-symbol: held native input authority"
    original = tmp_path / "base"
    original.mkdir()
    raw = b"fixed metadata"
    (original / "config.json").write_bytes(raw)
    monkeypatch.setattr(m, "FIXED_BASE", str(original))
    monkeypatch.setattr(m, "_require_native_readonly_directory", lambda _: None)
    monkeypatch.setattr(m, "_require_native_readonly_file", lambda _: None)
    refs = {"config.json": {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}}
    held = m._HeldNativeInputs(str(original), refs)
    return m, original, held, raw


def test_framework_alias_retains_original_directory_and_detects_path_swap(tmp_path, monkeypatch):
    _m, original, held, raw = fixture(tmp_path, monkeypatch)
    try:
        original.rename(tmp_path / "old")
        original.mkdir()
        (original / "config.json").write_bytes(b"other bytes")
        assert (Path(held.alias) / "config.json").read_bytes() == raw
        with pytest.raises(ValueError):
            held.verify()
    finally:
        held.close()


def test_framework_held_file_rehash_detects_same_inode_mutation(tmp_path, monkeypatch):
    _m, original, held, raw = fixture(tmp_path, monkeypatch)
    try:
        (original / "config.json").write_bytes(b"X" + raw[1:])
        with pytest.raises(ValueError):
            held.verify()
    finally:
        held.close()
