"""Fixed acquisition worker; descriptors retain output authority and integrity."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

_LIMIT = 1024 * 1024
_DATA_LIMIT = 512 * 1024 * 1024
_OUTPUT_PATH = "/opt/ml/processing/output"
_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _acquire_local(**kwargs: Any) -> dict[str, Any]:
    from ..data_eng.hub_dataset import _acquire_huggingface_dataset_into_fd

    return _acquire_huggingface_dataset_into_fd(**kwargs)


def _open_directory(raw: str) -> int:
    if not isinstance(raw, str) or not raw.startswith("/") or "\\" in raw or "%" in raw or "://" in raw:
        raise ValueError("unsafe acquisition root path")
    parts = raw.split("/")[1:]
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("unsafe acquisition root path")
    fd = os.open("/", _FLAGS)
    try:
        for part in parts:
            child = os.open(part, _FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except OSError as error:
        os.close(fd)
        raise ValueError("unsafe symbolic acquisition root path") from error


@dataclass
class _RootAuthority:
    path: Path
    fd: int

    def close(self) -> None:
        os.close(self.fd)

    def verify_name(self) -> None:
        named = _open_directory(str(self.path))
        try:
            a, b = os.fstat(self.fd), os.fstat(named)
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise ValueError("acquisition output root identity changed")
        finally:
            os.close(named)


def _authority(output_dir: str, output_fd: int | None = None) -> _RootAuthority:
    if output_fd is None:
        fd = _open_directory(output_dir)
    else:
        if type(output_fd) is not int or output_fd < 0:
            raise ValueError("acquisition output descriptor is invalid")
        fd = os.dup(output_fd)
    root = _RootAuthority(Path(output_dir), fd)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError("acquisition output descriptor is not a directory")
        root.verify_name()
        return root
    except Exception:
        root.close()
        raise


def _relative_parts(raw: str, root: _RootAuthority) -> tuple[str, ...]:
    prefix = str(root.path) + "/"
    if not isinstance(raw, str) or not raw.startswith(prefix) or "\\" in raw or "%" in raw:
        raise ValueError("acquisition core returned an unsafe file path")
    parts = tuple(raw[len(prefix) :].split("/"))
    if any(p in ("", ".", "..") or any(ord(c) < 32 for c in p) for p in parts):
        raise ValueError("acquisition core returned an unsafe file path")
    return parts


@dataclass
class _OpenedArtifact:
    stream: BinaryIO
    parts: tuple[str, ...]
    root: _RootAuthority
    owns_root: bool = False
    verified_signature: tuple[int, ...] | None = None
    verified_digest: str | None = None

    def close(self) -> None:
        self.stream.close()
        if self.owns_root:
            self.root.close()

    def verify_name(self) -> None:
        if self.verified_signature is None or _stat_signature(self) != self.verified_signature:
            raise ValueError("acquisition artifact changed after verification")
        expected_signature, expected_digest = self.verified_signature, self.verified_digest
        if expected_digest is None or _file_digest(self) != expected_digest:
            raise ValueError("acquisition artifact content changed after verification")
        if self.verified_signature != expected_signature:
            raise ValueError("acquisition artifact changed after verification")
        named = _confined_file(str(self.root.path / Path(*self.parts)), self.root)
        try:
            a, b = os.fstat(self.stream.fileno()), os.fstat(named.stream.fileno())
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise ValueError("acquisition artifact identity changed")
            if _stat_signature(named) != self.verified_signature:
                raise ValueError("acquisition artifact changed after verification")
        finally:
            named.close()


def _confined_file(raw: str, root: Path | _RootAuthority) -> _OpenedArtifact:
    owns_root = isinstance(root, Path)
    authority = _authority(str(root)) if owns_root else root
    directory = None
    handle = None
    try:
        parts = _relative_parts(raw, authority)
        directory = os.dup(authority.fd)
        for part in parts[:-1]:
            child = os.open(part, _FLAGS, dir_fd=directory)
            os.close(directory)
            directory = child
        handle = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > _DATA_LIMIT:
            raise ValueError("acquisition artifact is not a bounded regular file")
        stream = os.fdopen(handle, "rb")
        handle = None
        return _OpenedArtifact(stream, parts, authority, owns_root)
    except OSError as error:
        if owns_root:
            authority.close()
        raise ValueError("unsafe symbolic acquisition artifact component") from error
    except Exception:
        if owns_root:
            authority.close()
        raise
    finally:
        if directory is not None:
            os.close(directory)
        if handle is not None:
            os.close(handle)


def _stat_signature(file: _OpenedArtifact) -> tuple[int, ...]:
    info = os.fstat(file.stream.fileno())
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_json(file: _OpenedArtifact) -> tuple[bytes, dict[str, Any]]:
    before = _stat_signature(file)
    file.stream.seek(0)
    data = file.stream.read(_LIMIT + 1)
    if len(data) > _LIMIT or before != _stat_signature(file):
        raise ValueError("acquisition manifest exceeds its bound or changed")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("acquisition manifest is not an object")
    file.verified_signature = before
    file.verified_digest = hashlib.sha256(data).hexdigest()
    return data, value


def _file_digest(file: _OpenedArtifact) -> str:
    before = _stat_signature(file)
    file.stream.seek(0)
    digest = hashlib.sha256()
    count = 0
    for chunk in iter(lambda: file.stream.read(1024 * 1024), b""):
        count += len(chunk)
        if count > _DATA_LIMIT:
            raise ValueError("acquisition payload exceeds byte bound")
        digest.update(chunk)
    if before != _stat_signature(file) or count != before[2]:
        raise ValueError("acquisition payload changed during verification")
    file.verified_signature = before
    file.verified_digest = digest.hexdigest()
    return file.verified_digest


def _verify_generated_receipt(fd: int, root_fd: int, expected: bytes, digest: str) -> None:
    """Read the still-open publication FD; the name is only an identity check."""
    before = os.fstat(fd)
    named = os.stat("ACQUIRED.json", dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not stat.S_ISREG(named.st_mode)
        or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        or before.st_size != len(expected)
    ):
        raise ValueError("generated acquisition receipt identity or count changed")
    actual = os.pread(fd, len(expected) + 1, 0)
    after = os.fstat(fd)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        len(actual) != len(expected)
        or hashlib.sha256(actual).hexdigest() != digest
        or actual != expected
        or any(getattr(before, field) != getattr(after, field) for field in fields)
    ):
        raise ValueError("generated acquisition receipt bytes changed")


def _run_acquisition(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    output_dir: str,
    job_name: str,
    runtime_session_id: str,
    deployment_id: str,
    deployment_prefix: str,
    image_uri: str,
    source_commit: str,
    _output_fd: int | None = None,
    _privilege_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicit local oracle or descriptor-bound managed worker; no managed fallback."""
    root = _authority(output_dir, _output_fd)
    files = []
    receipt_created = False
    try:
        result = _acquire_local(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            output_dir=str(root.path),
            output_fd=root.fd,
        )
        root.verify_name()
        if not result.get("success"):
            from ..data_eng.hub_dataset import acquisition_failure_diagnostic

            raise RuntimeError(acquisition_failure_diagnostic(result)) from None
        data = _confined_file(result["local_path"], root)
        files.append(data)
        lineage_file = _confined_file(result["lineage_path"], root)
        files.append(lineage_file)
        commit_file = _confined_file(result["commit_manifest_path"], root)
        files.append(commit_file)
        if commit_file.parts[-1] != "COMMITTED.json" or not (
            data.parts[:-1] == lineage_file.parts[:-1] == commit_file.parts[:-1]
        ):
            raise ValueError("acquisition core returned inconsistent bundle paths")
        commit_bytes, commit = _read_json(commit_file)
        lineage_bytes, lineage = _read_json(lineage_file)
        if commit.get("data") != data.parts[-1] or commit.get("lineage") != lineage_file.parts[-1]:
            raise ValueError("acquisition core committed paths do not match the bundle")
        digest = _file_digest(data)
        if (
            digest != commit.get("sha256")
            or digest != lineage.get("sha256")
            or hashlib.sha256(lineage_bytes).hexdigest() != commit.get("lineage_sha256")
            or os.fstat(data.stream.fileno()).st_size != lineage.get("byte_count")
            or ("sha256" in result and digest != result["sha256"])
            or ("byte_count" in result and os.fstat(data.stream.fileno()).st_size != result["byte_count"])
        ):
            raise ValueError("acquisition core bundle integrity verification failed")
        expected_uri = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quote(filename, safe='/')}"
        )
        expected = {
            "repo_id": repo_id,
            "revision": revision,
            "filename": filename,
            "source_uri": expected_uri,
        }
        if any(lineage.get(key) != value for key, value in expected.items()):
            raise ValueError("acquisition lineage does not match requested pinned source")
        if not re.fullmatch(r"[a-f0-9]{40}", source_commit):
            raise ValueError("acquisition image source commit is required")
        receipt = {
            "schema_version": 1,
            "job_name": job_name,
            "runtime_session_id": runtime_session_id,
            "deployment_id": deployment_id,
            "deployment_prefix": deployment_prefix,
            "image_uri": image_uri,
            "source_commit": source_commit,
            "repo_id": repo_id,
            "revision": revision,
            "filename": filename,
            "data": "/".join(data.parts),
            "lineage": "/".join(lineage_file.parts),
            "commit_manifest": "/".join(commit_file.parts),
            "commit_manifest_sha256": hashlib.sha256(commit_bytes).hexdigest(),
        }
        if _privilege_proof is not None:
            receipt.update(
                schema_version=2,
                privilege_profile="sagemaker_native_ceiling_v1",
                privilege_proof=_privilege_proof,
            )
        expected_receipt = (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8")
        if len(expected_receipt) > _LIMIT:
            raise ValueError("generated acquisition receipt exceeds byte bound")
        expected_receipt_digest = hashlib.sha256(expected_receipt).hexdigest()
        root.verify_name()
        for file in files:
            file.verify_name()
        fd = os.open(
            "ACQUIRED.json", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root.fd
        )
        receipt_created = True
        with os.fdopen(fd, "r+", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            _verify_generated_receipt(stream.fileno(), root.fd, expected_receipt, expected_receipt_digest)
            root.verify_name()
            for file in files:
                file.verify_name()
            _verify_generated_receipt(stream.fileno(), root.fd, expected_receipt, expected_receipt_digest)
        return receipt
    except Exception:
        if receipt_created:
            os.unlink("ACQUIRED.json", dir_fd=root.fd)
        raise
    finally:
        for file in files:
            file.close()
        root.close()


def run_acquisition(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    output_dir: str,
    job_name: str,
    runtime_session_id: str,
    deployment_id: str,
    deployment_prefix: str,
    image_uri: str,
    source_commit: str,
    _output_fd: int | None = None,
) -> dict[str, Any]:
    """Explicit unchanged strict/offline schema1 acquisition path."""
    return _run_acquisition(
        repo_id=repo_id,
        revision=revision,
        filename=filename,
        output_dir=output_dir,
        job_name=job_name,
        runtime_session_id=runtime_session_id,
        deployment_id=deployment_id,
        deployment_prefix=deployment_prefix,
        image_uri=image_uri,
        source_commit=source_commit,
        _output_fd=_output_fd,
    )


def run_acquisition_native(**arguments: Any) -> dict[str, Any]:
    """Source-owned native schema2; observed kernel proof is never supplied by caller."""
    proof = _verify_worker_native_identity()
    return _run_acquisition(**arguments, _privilege_proof=proof)


def _verify_output_lease() -> None:
    try:
        info = os.fstat(3)
        if not stat.S_ISDIR(info.st_mode) or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (
            10001,
            10001,
            0o700,
        ):
            raise RuntimeError("acquisition output authority refused")
        fcntl.flock(3, fcntl.LOCK_EX | fcntl.LOCK_NB)
        authority = _authority(_OUTPUT_PATH, 3)
        authority.close()
    except (OSError, ValueError) as error:
        raise RuntimeError("acquisition output authority refused") from error


def _verify_worker_identity() -> None:
    if (
        os.getresuid() != (10001, 10001, 10001)
        or os.getresgid() != (10001, 10001, 10001)
        or os.getgroups() != []
    ):
        raise RuntimeError("acquisition worker identity refused")
    with open("/proc/self/status", "rb") as stream:
        data = stream.read(32769)
    if len(data) > 32768:
        raise RuntimeError("acquisition worker identity refused")
    values = {}
    for line in data.decode("ascii").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key in values:
                raise RuntimeError("acquisition worker identity refused")
            values[key] = value.strip()
    if values.get("NoNewPrivs") != "1" or any(
        int(values.get(key, "-1"), 16) != 0 for key in ("CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd")
    ):
        raise RuntimeError("acquisition worker identity refused")


def _verify_worker_native_identity() -> dict[str, Any]:
    from processing_identity import verify_native_identity

    try:
        return verify_native_identity()
    except ValueError:
        raise RuntimeError("acquisition worker identity refused") from None


def _main_managed(runner) -> None:
    parser = argparse.ArgumentParser(description="Acquire one pinned public data artifact")
    for key in (
        "repo_id",
        "revision",
        "filename",
        "job_name",
        "runtime_session_id",
        "deployment_id",
        "deployment_prefix",
        "image_uri",
        "source_commit",
    ):
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    arguments = vars(parser.parse_args())
    source = os.environ.get("AMLAS_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[a-f0-9]{40}", source) or arguments["source_commit"] != source:
        raise RuntimeError("acquisition source binding refused")
    _verify_output_lease()
    runner(**arguments, output_dir=_OUTPUT_PATH, _output_fd=3)


def main() -> None:
    _verify_worker_identity()
    _main_managed(run_acquisition)


def main_native() -> None:
    _verify_worker_native_identity()
    _main_managed(run_acquisition_native)


if __name__ == "__main__":
    main()
