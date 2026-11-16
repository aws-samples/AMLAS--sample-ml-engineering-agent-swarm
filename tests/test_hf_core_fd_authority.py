"""Retained-FD root authority with actual filesystem swaps; no live data calls."""

from __future__ import annotations

import fcntl
import inspect
import os
from pathlib import Path

import pytest
from test_huggingface_commit_boundary import core as core
from test_huggingface_ingestion import PAYLOAD, REVISION, SHA256


def acquire(core, path, descriptor):
    function = getattr(core, "_acquire_huggingface_dataset_into_fd", None)
    assert callable(function), "missing-symbol: private _acquire_huggingface_dataset_into_fd"
    return function("example/coding", REVISION, "v1/train.jsonl", str(path), descriptor)


def open_root(path):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def test_private_core_duplicates_caller_authority_without_closing_or_unlocking_it(core, tmp_path):
    root = tmp_path / "output"
    root.mkdir(mode=0o700)
    held = open_root(root)
    observer = open_root(root)
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = os.fstat(held)
        result = acquire(core, root, held)
        assert result["success"] is True
        assert result["sha256"] == SHA256
        assert Path(result["local_path"]).read_bytes() == PAYLOAD
        assert Path(result["commit_manifest_path"]).is_file()
        assert os.fstat(held).st_ino == identity.st_ino
        with pytest.raises(BlockingIOError):
            fcntl.flock(observer, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(observer)
        os.close(held)


@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_retained_root_swap_before_core_cannot_write_new_authority(core, tmp_path, replacement):
    root = tmp_path / "output"
    root.mkdir(mode=0o700)
    original = tmp_path / "original"
    destination = tmp_path / "destination"
    destination.mkdir()
    held = open_root(root)
    try:
        root.rename(original)
        if replacement == "directory":
            root.mkdir(mode=0o700)
        else:
            root.symlink_to(destination, target_is_directory=True)
        result = acquire(core, root, held)
        assert result["success"] is False
        assert os.listdir(held) == []
        assert list(root.iterdir()) == []
        assert os.fstat(held).st_ino == original.stat().st_ino
    finally:
        os.close(held)


def test_private_core_never_reopens_output_string_for_write_authority(core, monkeypatch, tmp_path):
    root = tmp_path / "output"
    root.mkdir(mode=0o700)
    held = open_root(root)
    try:
        monkeypatch.setattr(core, "_output_directory", lambda *args: pytest.fail("FD core reopened string authority"))
        monkeypatch.setattr(Path, "resolve", lambda *args, **kwargs: pytest.fail("original lexical path was resolved"))
        result = acquire(core, root, held)
        assert result["success"] is True
        assert len(os.listdir(held)) == 1
    finally:
        os.close(held)


def test_swap_at_mkdir_keeps_actual_write_on_original_fd_and_refuses_commit(core, monkeypatch, tmp_path):
    root = tmp_path / "output"
    root.mkdir(mode=0o700)
    original = tmp_path / "original"
    held = open_root(root)
    original_identity = os.fstat(held)
    mkdir = os.mkdir
    observed = []

    def mkdir_then_check(path, mode=0o777, *, dir_fd=None):
        if isinstance(path, str) and path.startswith("acquisition-"):
            identity = os.fstat(dir_fd)
            assert (identity.st_dev, identity.st_ino) == (original_identity.st_dev, original_identity.st_ino)
            root.rename(original)
            root.mkdir(mode=0o700)
            mkdir(path, mode, dir_fd=dir_fd)
            observed.append((os.listdir(held), list(root.iterdir())))
            return
        return mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", mkdir_then_check)
    try:
        result = acquire(core, root, held)
        assert observed and len(observed[0][0]) == 1 and observed[0][1] == []
        assert result["success"] is False
        assert os.listdir(held) == [] and list(root.iterdir()) == []
    finally:
        os.close(held)


def test_pending_manifest_root_swap_cleans_original_bundle_not_replacement(core, monkeypatch, tmp_path):
    root = tmp_path / "output"
    root.mkdir(mode=0o700)
    original = tmp_path / "original"
    held = open_root(root)
    write_json = core._write_json

    def swap_after_pending(descriptor, name, value):
        result = write_json(descriptor, name, value)
        if name == ".commit-pending.json":
            root.rename(original)
            root.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(core, "_write_json", swap_after_pending)
    try:
        result = acquire(core, root, held)
        assert result["success"] is False
        assert os.listdir(held) == [] and list(root.iterdir()) == []
    finally:
        os.close(held)


def test_original_ancestor_symlink_alias_cannot_authorize_retained_root(core, tmp_path):
    parent = tmp_path / "parent"
    root = parent / "output"
    root.mkdir(parents=True, mode=0o700)
    held = open_root(root)
    try:
        parent.rename(tmp_path / "parent-original")
        parent.symlink_to(tmp_path / "parent-original", target_is_directory=True)
        result = acquire(core, root, held)
        assert result["success"] is False and os.listdir(held) == []
    finally:
        os.close(held)


def test_fd_does_not_override_allowed_root_or_lexical_identity(core, monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "allowed-other"
    outside.mkdir()
    held = open_root(outside)
    monkeypatch.setattr(core.pathref, "_allowed_roots", lambda: [allowed])
    try:
        result = acquire(core, outside, held)
        assert result["success"] is False and os.listdir(held) == []
    finally:
        os.close(held)


def test_mismatched_named_path_cannot_relabel_valid_fd(core, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    held = open_root(first)
    try:
        result = acquire(core, second, held)
        assert result["success"] is False
        assert list(first.iterdir()) == list(second.iterdir()) == []
    finally:
        os.close(held)


@pytest.mark.parametrize("descriptor", [None, True, -1, 999999])
def test_missing_invalid_fd_never_falls_back_to_local_path(core, tmp_path, descriptor):
    root = tmp_path / "output"
    root.mkdir()
    result = acquire(core, root, descriptor)
    assert result["success"] is False and list(root.iterdir()) == []


def test_regular_file_descriptor_is_not_directory_authority(core, tmp_path):
    file = tmp_path / "file"
    file.write_bytes(b"original")
    held = os.open(file, os.O_RDONLY)
    try:
        result = acquire(core, tmp_path, held)
        assert result["success"] is False and file.read_bytes() == b"original"
    finally:
        os.close(held)


def test_public_local_signature_and_model_tool_schema_do_not_expose_fd_authority(core):
    assert list(inspect.signature(core.acquire_huggingface_dataset_local).parameters) == [
        "repo_id", "revision", "filename", "output_dir",
    ]
    from amlas.tools.huggingface_data import acquire_huggingface_dataset

    assert "output_fd" not in acquire_huggingface_dataset.tool_spec["inputSchema"]["json"]["properties"]


@pytest.mark.parametrize("mutation", ["replacement", "same_inode"])
def test_streamed_receipt_cannot_commit_replaced_or_mutated_artifact(core, monkeypatch, tmp_path, mutation):
    root = tmp_path / "output"
    root.mkdir()
    held = open_root(root)
    stream = core.public_fetch.stream_public_binary

    def change_after_stream(url, sink, **kwargs):
        receipt = stream(url, sink, **kwargs)
        sink.flush()
        name = Path(os.readlink(f"/proc/self/fd/{sink.fileno()}"))
        if mutation == "replacement":
            name.rename(tmp_path / "original-payload")
        name.write_bytes(b"x" * len(PAYLOAD))
        return receipt

    monkeypatch.setattr(core.public_fetch, "stream_public_binary", change_after_stream)
    try:
        result = acquire(core, root, held)
        assert result["success"] is False
        assert not list(root.glob("*/COMMITTED.json"))
        assert os.fstat(held).st_ino == root.stat().st_ino
    finally:
        os.close(held)


def test_same_inode_mutation_after_verification_cannot_commit(core, monkeypatch, tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    held = open_root(root)
    write_json = core._write_json

    def mutate_during_pending(descriptor, name, value):
        if name == ".commit-pending.json":
            data = os.open("data.jsonl", os.O_WRONLY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                os.write(data, b"x" * len(PAYLOAD))
                os.fsync(data)
            finally:
                os.close(data)
        return write_json(descriptor, name, value)

    monkeypatch.setattr(core, "_write_json", mutate_during_pending)
    try:
        result = acquire(core, root, held)
        assert result["success"] is False
        assert not list(root.glob("*/COMMITTED.json"))
    finally:
        os.close(held)


@pytest.mark.parametrize("phase", ["lineage", "post_commit"])
def test_same_length_rewrite_with_unchanged_stat_still_fails_integrity(core, monkeypatch, tmp_path, phase):
    root = tmp_path / "output"
    root.mkdir()
    held = open_root(root)
    write_json = core._write_json
    publish = os.link
    identity = core._file_identity
    rewrites = []

    def rewrite(directory):
        descriptor = os.open("data.jsonl", os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
        try:
            original = identity(descriptor)
            os.write(descriptor, b"x" * len(PAYLOAD))
            os.fsync(descriptor)
            monkeypatch.setattr(core, "_file_identity", lambda data_fd: original)
            rewrites.append(original)
        finally:
            os.close(descriptor)

    def lineage_rewrite(directory, name, value):
        result = write_json(directory, name, value)
        if phase == "lineage" and name == "lineage.json":
            rewrite(directory)
        return result

    def commit_rewrite(src, dst, **kwargs):
        result = publish(src, dst, **kwargs)
        if phase == "post_commit" and dst == "COMMITTED.json":
            rewrite(kwargs["dst_dir_fd"])
        return result

    monkeypatch.setattr(core, "_write_json", lineage_rewrite)
    monkeypatch.setattr(os, "link", commit_rewrite)
    try:
        result = acquire(core, root, held)
        assert rewrites, "actual same-length data rewrite must occur at the intended publication boundary"
        assert result["success"] is False, "unchanged stat signature is insufficient integrity proof"
        assert not list(root.glob("*/COMMITTED.json"))
        assert os.fstat(held).st_ino == root.stat().st_ino
    finally:
        os.close(held)
