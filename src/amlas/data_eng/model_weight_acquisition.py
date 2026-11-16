"""Private retained-FD raw model writer; no model/config/tokenizer parser."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import time
from pathlib import Path
from typing import Any

from . import model_acquisition_schema as schema
from .model_public_fetch import stream_model_catalog_file
from .model_weight_manifest import MAX_BASE_BYTES, MAX_FILE_BYTES, expected_model_weight_manifest

DISK_CAP_BYTES = 80 * 1024**3
DISK_RESERVE_BYTES = 10 * 1024**3
_DIRECTORY_FLAGS = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _walk_directory(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("model output requires original absolute directory components")
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for name in path.parts[1:]:
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _same(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _root_current(fd: int, path: Path) -> None:
    named = _walk_directory(path)
    try:
        if not _same(os.fstat(fd), os.fstat(named)):
            raise ValueError("model output original directory identity changed")
    finally:
        os.close(named)


def _verify_file(root_fd: int, name: str, fd: int, count: int, digest: str, deadline: float) -> None:
    before = os.fstat(fd)
    named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != count
        or not _same(before, named)
        or stat.S_IMODE(before.st_mode) != 0o600
    ):
        raise ValueError("model retained file identity/size/mode differs")
    actual, measured = hashlib.sha256(), 0
    while True:
        if time.monotonic() >= deadline:
            raise ValueError("model retained file hash deadline exceeded")
        chunk = os.pread(fd, min(1024 * 1024, count - measured + 1), measured)
        if not chunk:
            break
        measured += len(chunk)
        if measured > count:
            raise ValueError("model retained file exceeded declared count")
        actual.update(chunk)
    after = os.fstat(fd)
    named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)

    def signature(item: os.stat_result) -> tuple[int, ...]:
        return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns

    if (
        signature(before) != signature(after)
        or not _same(after, named)
        or measured != count
        or actual.hexdigest() != digest
    ):
        raise ValueError("model retained bytes or original name changed")


def _disk_budget(fd: int, remaining: int) -> None:
    free = os.fstatvfs(fd)
    if remaining > DISK_CAP_BYTES or free.f_bavail * free.f_frsize < remaining + DISK_RESERVE_BYTES:
        raise ValueError("model declared disk budget or reserve unavailable")


def acquire_model_into_fd(
    *,
    catalog_id: str,
    catalog_version: str,
    output_fd: int,
    output_directory: Path,
    bindings: dict[str, str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Only a guarded fixed worker calls this private writer; paths are metadata."""
    expected = expected_model_weight_manifest(catalog_id, catalog_version)
    schema.validate_bindings(bindings)
    schema.validate_evidence(evidence)
    total = sum(item["byte_count"] for item in expected["files"])
    if (
        total > MAX_BASE_BYTES
        or total > DISK_CAP_BYTES
        or any(
            type(item["byte_count"]) is not int or not 0 < item["byte_count"] <= MAX_FILE_BYTES
            for item in expected["files"]
        )
    ):
        raise ValueError("model source byte budget differs")
    root = os.fstat(output_fd)
    if not stat.S_ISDIR(root.st_mode) or stat.S_IMODE(root.st_mode) != 0o700:
        raise ValueError("model output directory authority differs")
    fcntl.flock(output_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _root_current(output_fd, output_directory)
    if os.listdir(output_fd):
        raise ValueError("model acquisition output must start empty")
    _disk_budget(output_fd, total)
    held: list[tuple[int, str, int, str]] = []
    commit_fd = None
    deadline = time.monotonic() + 7200.0
    acquired = False
    try:
        remaining = total
        for ref in expected["files"]:
            _root_current(output_fd, output_directory)
            _disk_budget(output_fd, remaining)
            fd = os.open(
                ref["path"],
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=output_fd,
            )
            held.append((fd, ref["path"], ref["byte_count"], ref["sha256"]))
            with os.fdopen(os.dup(fd), "wb") as sink:
                receipt = stream_model_catalog_file(ref["path"], sink, deadline=deadline)
                sink.flush()
                os.fsync(sink.fileno())
            if receipt != {"byte_count": ref["byte_count"], "sha256": ref["sha256"]}:
                raise ValueError("model downloader receipt differs from catalog")
            _verify_file(output_fd, ref["path"], fd, ref["byte_count"], ref["sha256"], deadline)
            _root_current(output_fd, output_directory)
            remaining -= ref["byte_count"]

        def publish(name: str, raw: bytes) -> int:
            nonlocal commit_fd
            _root_current(output_fd, output_directory)
            fd = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=output_fd,
            )
            if name == "COMMITTED.json":
                # Own the exact opened marker before any write/fsync can fail.
                commit_fd = fd
            digest = hashlib.sha256(raw).hexdigest()
            held.append((fd, name, len(raw), digest))
            with os.fdopen(os.dup(fd), "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            _verify_file(output_fd, name, fd, len(raw), digest, deadline)
            return fd

        lineage = schema.encoded(schema.lineage(catalog_id, catalog_version, bindings))
        publish("LINEAGE.json", lineage)
        receipt = schema.encoded(schema.receipt(catalog_id, catalog_version, bindings, evidence, lineage))
        publish("MODEL_ACQUIRED.json", receipt)
        for fd, name, count, digest in held:
            _verify_file(output_fd, name, fd, count, digest, deadline)
        committed = schema.encoded(schema.commit(catalog_id, catalog_version, bindings, lineage, receipt))
        commit_fd = publish("COMMITTED.json", committed)
        os.fsync(output_fd)
        for fd, name, count, digest in held:
            _verify_file(output_fd, name, fd, count, digest, deadline)
        if set(os.listdir(output_fd)) != {ref["path"] for ref in expected["files"]} | {
            "LINEAGE.json",
            "MODEL_ACQUIRED.json",
            "COMMITTED.json",
        }:
            raise ValueError("model output inventory changed")
        _root_current(output_fd, output_directory)
        for fd, name, count, digest in held:
            _verify_file(output_fd, name, fd, count, digest, deadline)
        _root_current(output_fd, output_directory)
        acquired = True
        return {
            "success": True,
            "acquired": True,
            "prepared": False,
            "trained": False,
            "trusted": False,
            "ready": False,
            "catalog_id": catalog_id,
            "catalog_version": catalog_version,
            "commit_sha256": hashlib.sha256(committed).hexdigest(),
        }
    finally:
        if not acquired and commit_fd is not None:
            try:
                named = os.stat("COMMITTED.json", dir_fd=output_fd, follow_symlinks=False)
                if _same(named, os.fstat(commit_fd)):
                    os.unlink("COMMITTED.json", dir_fd=output_fd)
            except OSError:
                pass
        for fd, _, _, _ in held:
            os.close(fd)
