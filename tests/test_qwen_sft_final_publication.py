"""Final manifest publication checks every earlier expected artifact on held FDs."""

from __future__ import annotations

import hashlib
import importlib
import os

import pytest


@pytest.mark.parametrize("artifact", ["TRAINING.json", "adapter_model.safetensors"])
def test_final_publication_refuses_same_inode_earlier_artifact_change(tmp_path, monkeypatch, artifact):
    m = importlib.import_module("amlas.training.qwen_sft")
    raw = b'{"fixed":"original"}\n'
    target = tmp_path / artifact
    target.write_bytes(raw)
    ref = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    monkeypatch.setattr(m, "FIXED_OUTPUT", str(tmp_path))
    original = os.fsync

    def fsync(descriptor):
        original(descriptor)
        if os.readlink("/proc/self/fd/" + str(descriptor)).endswith("/COMMITTED.json"):
            with target.open("r+b", buffering=0) as stream:
                stream.write(b"!")

    monkeypatch.setattr(os, "fsync", fsync)
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            m._publish_fixed_json(
                directory, "COMMITTED.json", {"schema_version": 1, "files": {artifact: ref}}
            )
        assert (tmp_path / "COMMITTED.json").read_bytes() == b""
    finally:
        os.close(directory)


def test_final_publication_keeps_pinned_inventory(tmp_path, monkeypatch):
    m = importlib.import_module("amlas.training.qwen_sft")
    monkeypatch.setattr(m, "FIXED_OUTPUT", str(tmp_path))
    raw = b'{"actual":1}\n'
    (tmp_path / "RELOAD.json").write_bytes(raw)
    ref = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        result = m._publish_fixed_json(
            directory, "COMMITTED.json", {"schema_version": 1, "files": {"RELOAD.json": ref}}
        )
    finally:
        os.close(directory)
    assert result["sha256"] == hashlib.sha256((tmp_path / "COMMITTED.json").read_bytes()).hexdigest()


def test_failed_marker_invalidation_never_unlinks_path_replacement(tmp_path, monkeypatch):
    m = importlib.import_module("amlas.training.qwen_sft")
    monkeypatch.setattr(m, "FIXED_OUTPUT", str(tmp_path))
    raw = b'{"actual":1}\n'
    target = tmp_path / "TRAINING.json"
    target.write_bytes(raw)
    ref = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    original_fsync = os.fsync
    original_named = m._checkpoint_named
    foreign = b'{"foreign":true}\n'
    swapped = []

    def fsync(fd):
        original_fsync(fd)
        if os.readlink("/proc/self/fd/" + str(fd)).endswith("/COMMITTED.json"):
            target.write_bytes(b"X" + raw[1:])

    def named(directory, descriptor, name):
        original_named(directory, descriptor, name)
        if name == "COMMITTED.json" and not swapped:
            # Actual interleaving after the successful identity check.
            (tmp_path / name).rename(tmp_path / "OWNED_FAILED.json")
            (tmp_path / name).write_bytes(foreign)
            swapped.append(True)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(m, "_checkpoint_named", named)
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            m._publish_fixed_json(
                directory, "COMMITTED.json", {"schema_version": 1, "files": {"TRAINING.json": ref}}
            )
    finally:
        os.close(directory)
    assert swapped and (tmp_path / "COMMITTED.json").read_bytes() == foreign
    assert (tmp_path / "OWNED_FAILED.json").read_bytes() == b""


def test_exact_rollback_syscall_boundary_preserves_foreign_marker(tmp_path, monkeypatch):
    m = importlib.import_module("amlas.training.qwen_sft")
    monkeypatch.setattr(m, "FIXED_OUTPUT", str(tmp_path))
    raw = b'{"actual":1}\n'
    target = tmp_path / "TRAINING.json"
    target.write_bytes(raw)
    ref = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    original_fsync, original_unlink, original_truncate = os.fsync, os.unlink, os.ftruncate
    foreign = b'{"foreign":true}\n'
    swapped = []

    def swap():
        if not swapped:
            (tmp_path / "COMMITTED.json").rename(tmp_path / "OWNED_FAILED.json")
            (tmp_path / "COMMITTED.json").write_bytes(foreign)
            swapped.append(True)

    def fsync(fd):
        original_fsync(fd)
        if os.readlink("/proc/self/fd/" + str(fd)).endswith("/COMMITTED.json"):
            target.write_bytes(b"X" + raw[1:])

    def unlink(name, **kwargs):
        if name == "COMMITTED.json":
            swap()
        return original_unlink(name, **kwargs)

    def truncate(fd, count):
        if os.readlink("/proc/self/fd/" + str(fd)).endswith("/COMMITTED.json"):
            swap()
        return original_truncate(fd, count)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "ftruncate", truncate)
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            m._publish_fixed_json(
                directory, "COMMITTED.json", {"schema_version": 1, "files": {"TRAINING.json": ref}}
            )
    finally:
        os.close(directory)
    assert swapped and (tmp_path / "COMMITTED.json").read_bytes() == foreign
    assert (tmp_path / "OWNED_FAILED.json").read_bytes() == b""
