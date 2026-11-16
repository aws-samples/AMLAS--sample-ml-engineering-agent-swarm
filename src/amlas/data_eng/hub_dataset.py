"""Portable anonymous Hub metadata/acquisition code, without an agent or Hub SDK.

The REST wire shape comes from the Hub's published OpenAPI and dataset_info API.
This module acquires one integrity-checked artifact. It does not prepare data,
execute repository scripts, accept gated licenses, or train a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from ..tools import pathref
from ..tools.research import public_fetch

MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_FILES = 512
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?")
_REVISION = re.compile(r"[a-f0-9]{40}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_FORMATS = {".parquet": "parquet", ".jsonl": "jsonl", ".ndjson": "jsonl", ".json": "json", ".csv": "csv"}


_FAILURE_STAGES = frozenset(
    {
        "validation",
        "output_directory",
        "metadata",
        "file_selection",
        "bundle_create",
        "download",
        "integrity",
        "lineage",
        "commit",
    }
)
_FAILURE_TYPES = frozenset(
    {
        "HubDatasetError",
        "PublicFetchError",
        "UnexpectedError",
        "PermissionError",
        "FileNotFoundError",
        "FileExistsError",
        "NotADirectoryError",
        "IsADirectoryError",
        "OSError",
        "BlockingIOError",
        "TimeoutError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "BrokenPipeError",
        "gaierror",
        "SSLError",
        "SSLCertVerificationError",
        "RemoteDisconnected",
        "IncompleteRead",
        "BadStatusLine",
        "HTTPException",
        "ValueError",
        "TypeError",
        "OverflowError",
        "MemoryError",
        "UnicodeDecodeError",
        "JSONDecodeError",
        "RuntimeError",
        "AssertionError",
    }
)


def _safe_error_type(error: BaseException) -> str:
    name = type(error).__name__
    return name if name in _FAILURE_TYPES else "UnexpectedError"


def _failure_context(stage: str, error: BaseException) -> dict[str, Any]:
    cause = error.__cause__ or error.__context__
    return {
        "error_stage": stage,
        "error_type": _safe_error_type(error),
        "error_cause_type": _safe_error_type(cause) if isinstance(cause, BaseException) else None,
    }


def acquisition_failure_diagnostic(result: dict[str, Any]) -> str:
    """Format only closed diagnostic labels; never copy error/URL/data text."""
    stage = result.get("error_stage")
    if not isinstance(stage, str) or stage not in _FAILURE_STAGES:
        stage = "unknown"
    kind = result.get("error_type")
    if not isinstance(kind, str) or kind not in _FAILURE_TYPES:
        kind = "UnexpectedError"
    message = f"dataset acquisition failed stage={stage} error_type={kind}"
    cause = result.get("error_cause_type")
    if isinstance(cause, str) and cause in _FAILURE_TYPES:
        message += " cause_type=" + cause
    return message


class HubDatasetError(ValueError):
    """A public repository, selected artifact or local commit was rejected."""


def _repository(repo_id: str) -> None:
    if not isinstance(repo_id, str) or not _REPO.fullmatch(repo_id) or ".." in repo_id:
        raise HubDatasetError("invalid public Hub repository identifier")


def _filename(filename: str) -> str:
    if (
        not isinstance(filename, str)
        or not filename
        or len(filename) > 1024
        or filename.startswith("/")
        or any(part in ("", ".", "..") for part in filename.split("/"))
        or any(char in filename for char in ("\\", "%", "\0", "?", "#"))
        or any(ord(char) < 32 for char in filename)
    ):
        raise HubDatasetError("invalid repository artifact path")
    return _FORMATS.get(PurePosixPath(filename).suffix.lower(), "")


def _only_unsupported_characters(filename: Any) -> bool:
    """True for a structurally sound repository path whose only defect is a character the artifact
    rules refuse ('#', '?', '%', backslash), such as bigcode/commitpackft data/c#/data.jsonl."""
    return (
        isinstance(filename, str)
        and 0 < len(filename) <= 1024
        and not filename.startswith("/")
        and all(part not in ("", ".", "..") for part in filename.split("/"))
        and not any(ord(char) < 32 for char in filename)
        and any(char in filename for char in ("\\", "%", "?", "#"))
    )


def _metadata(repo_id: str, revision: str | None) -> dict[str, Any]:
    _repository(repo_id)
    if not pathref.remote_fetch_enabled():
        raise HubDatasetError("remote public data retrieval is disabled")
    selected = revision or "main"
    if (
        not isinstance(selected, str)
        or len(selected) > 200
        or any(ord(char) < 32 for char in selected)
        or any(char in selected for char in ("%", "\\", "?", "#"))
    ):
        raise HubDatasetError("invalid Hub revision selector")
    url = (
        "https://huggingface.co/api/datasets/"
        + quote(repo_id, safe="/")
        + "/revision/"
        + quote(selected, safe="")
        + "?blobs=true"
    )
    info = public_fetch.fetch_public_json(url, max_bytes=1024 * 1024, timeout=20.0)
    if info.get("id") != repo_id or not _REVISION.fullmatch(str(info.get("sha", ""))):
        raise HubDatasetError("public Hub metadata lacks a matching repository and resolved revision")
    if not isinstance(info.get("siblings"), list):
        raise HubDatasetError("public Hub metadata lacks a file listing")
    return info


def _files(
    info: dict[str, Any], *, selected_filename: str | None = None
) -> tuple[list[dict[str, Any]], bool]:
    result: list[dict[str, Any]] = []
    siblings = info["siblings"]
    candidates = siblings if selected_filename is not None else siblings[:MAX_FILES]
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("rfilename")
        if selected_filename is not None and filename != selected_filename:
            continue
        try:
            file_format = _filename(filename)
        except HubDatasetError:
            continue
        if not file_format:
            continue
        lfs = entry.get("lfs")
        sha = lfs.get("sha256") if isinstance(lfs, dict) else None
        size = entry.get("size")
        if size is None and isinstance(lfs, dict):
            size = lfs.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            size = None
        if isinstance(lfs, dict) and lfs.get("size") != size:
            raise HubDatasetError("public Hub file size metadata is inconsistent")
        result.append(
            {
                "filename": filename,
                "format": file_format,
                "size": size,
                "sha256": sha if isinstance(sha, str) and _SHA256.fullmatch(sha) else None,
                "blob_id": (
                    entry.get("blobId")
                    if isinstance(entry.get("blobId"), str) and _REVISION.fullmatch(entry["blobId"])
                    else None
                ),
            }
        )
    return result, len(siblings) > MAX_FILES


# Lowered 2026-11-07 from 60000 / 27000 / 12500: real-model test runs measured 23.7-24.7 KB per page for large
# inventories and every continuation page the same. Worst case: files 8000 + declarations 2 x 2800
# (configs appear twice) + splits 1500 + license 2 x 2000 + about 1300 fixed fields = 20400.
MAX_INSPECTION_PROJECTION_BYTES = 21000
_FILES_PAGE_BYTES = 8000
_DECLARATIONS_PAGE_BYTES = 2800


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _sdk_projection_bytes(value: Any) -> int:
    """Installed Strands dict results use default JSON inside a text ContentBlock."""
    content = [{"text": json.dumps(value, allow_nan=False)}]
    return _json_bytes(content)


def _whole_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise HubDatasetError("metadata field exceeds the whole-field inspection contract")
    return value


def _declaration_inventory(
    card: dict[str, Any], *, offset: int
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Validate/count all whole declarations; allocate output only for this page."""
    source = card.get("configs", [])
    if not isinstance(source, list):
        raise HubDatasetError("dataset card configs are not a declared list")
    declarations: list[dict[str, Any]] = []
    splits: set[str] = set()
    total = 0
    used = 2
    page_full = False

    def retain(name: str | None, split: str | None, path: Any, *, empty: bool = False) -> None:
        nonlocal total, used, page_full
        ordinal = total
        total += 1
        if ordinal < offset or page_full:
            return
        if len(declarations) == MAX_FILES:
            page_full = True
            return
        record = {"config_name": name, "data_files": [] if empty else [{"split": split, "path": path}]}
        cost = _sdk_projection_bytes(record) + 1
        if used + cost > _DECLARATIONS_PAGE_BYTES:
            if not declarations:
                raise HubDatasetError("whole metadata record exceeds its page budget")
            page_full = True
            return
        declarations.append(record)
        used += cost

    for config in source:
        if not isinstance(config, dict):
            raise HubDatasetError("dataset card config is not declared metadata")
        name = _whole_text(config.get("config_name"), 100)
        files = config.get("data_files", [])
        if not isinstance(files, list):
            raise HubDatasetError("dataset card data_files are not a declared list")
        if not files:
            retain(name, None, None, empty=True)
        for file in files:
            if not isinstance(file, dict):
                raise HubDatasetError("dataset card file declaration is malformed")
            split = _whole_text(file.get("split"), 100)
            if split is not None:
                splits.add(split)
            paths = file.get("path")
            values = paths if isinstance(paths, list) and paths else [paths]
            for value in values:
                # An explicitly empty list remains empty; strings are never sliced.
                path = [] if value == [] else _whole_text(value, 1024)
                retain(name, split, path)
    info = card.get("dataset_info", [])
    if isinstance(info, dict):
        info = [info]
    if isinstance(info, list):
        for item in info:
            if isinstance(item, dict) and isinstance(item.get("splits"), list):
                for split in item["splits"]:
                    if isinstance(split, dict) and split.get("name") is not None:
                        name = _whole_text(split["name"], 100)
                        if name is not None:
                            splits.add(name)
    if offset > total:
        raise HubDatasetError("metadata page offset exceeds the resolved inventory")
    following = offset + len(declarations)
    page = {
        "offset": offset,
        "returned": len(declarations),
        "total": total,
        "next_offset": following if following < total else None,
        "coverage": "complete" if offset == 0 and following == total else "partial",
    }
    return declarations, page, sorted(splits)


def _metadata_page(
    records: list[dict[str, Any]], offset: int, byte_budget: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offset > len(records):
        raise HubDatasetError("metadata page offset exceeds the resolved inventory")
    selected: list[dict[str, Any]] = []
    used = 2
    for record in records[offset:]:
        cost = _sdk_projection_bytes(record) + 1
        if len(selected) == MAX_FILES or used + cost > byte_budget:
            break
        selected.append(record)
        used += cost
    if offset < len(records) and not selected:
        raise HubDatasetError("whole metadata record exceeds its page budget")
    following = offset + len(selected)
    return selected, {
        "offset": offset,
        "returned": len(selected),
        "total": len(records),
        "next_offset": following if following < len(records) else None,
        "coverage": "complete" if offset == 0 and following == len(records) else "partial",
    }


def inspect_huggingface_dataset_metadata(
    repo_id: str,
    revision: str | None = None,
    *,
    file_offset: int = 0,
    declaration_offset: int = 0,
) -> dict[str, Any]:
    """Resolve byte-bounded whole metadata pages; continuations require a full pin."""
    try:
        for offset in (file_offset, declaration_offset):
            if type(offset) is not int or not 0 <= offset <= 1024 * 1024:
                raise HubDatasetError("invalid metadata page offset")
        if (file_offset or declaration_offset) and (
            not isinstance(revision, str) or not _REVISION.fullmatch(revision)
        ):
            raise HubDatasetError("metadata page continuation requires the resolved full revision")
        info = _metadata(repo_id, revision)
        if isinstance(revision, str) and _REVISION.fullmatch(revision) and info["sha"] != revision:
            raise HubDatasetError("public Hub metadata changed the pinned revision")
        inventory: list[dict[str, Any]] = []
        unsupported_paths = 0
        for entry in info["siblings"]:
            if not isinstance(entry, dict):
                raise HubDatasetError("public Hub file listing contains malformed metadata")
            try:
                _filename(entry.get("rfilename"))
            except HubDatasetError:
                # A sound name with a refused character cannot be acquired; count it so coverage
                # stays explicit. Any structural defect still fails the whole inspection.
                if not _only_unsupported_characters(entry.get("rfilename")):
                    raise
                unsupported_paths += 1
                continue
            selected, _ = _files({"siblings": [entry]})
            inventory.extend(selected)
        files, files_page = _metadata_page(inventory, file_offset, _FILES_PAGE_BYTES)
        card = info.get("cardData")
        card = card if isinstance(card, dict) else {}
        configs, declarations_page, declared_splits = _declaration_inventory(card, offset=declaration_offset)
        splits: list[str] = []
        for split in declared_splits:
            if len(splits) == 64 or _sdk_projection_bytes([*splits, split]) > 1500:
                break
            splits.append(split)
        splits_truncated = len(splits) < len(declared_splits)
        license_value = card.get("license")
        if isinstance(license_value, str):
            license_value = _whole_text(license_value, 200)
        elif isinstance(license_value, list) and len(license_value) <= 20:
            license_value = [_whole_text(item, 200) for item in license_value]
        elif license_value is not None:
            raise HubDatasetError("license metadata exceeds the whole-field inspection contract")
        if _sdk_projection_bytes(license_value) > 2000:
            raise HubDatasetError("whole license metadata exceeds the inspection byte budget")
        gated = info.get("gated")
        gated = (
            _whole_text(gated, 100) if isinstance(gated, str) else gated if isinstance(gated, bool) else None
        )
        partial = (
            files_page["coverage"] != "complete"
            or declarations_page["coverage"] != "complete"
            or splits_truncated
        )
        result = {
            "success": True,
            "repo_id": repo_id,
            "revision": info["sha"],
            "license": license_value,
            "card": {"license": license_value, "configs": configs},
            "private": info.get("private") is not False,
            "public": info.get("private") is False and gated is False and info.get("disabled") is not True,
            "gated": gated,
            "disabled": info.get("disabled") is True,
            "files": files,
            "configs": configs,
            "splits": splits,
            "files_truncated": files_page["coverage"] != "complete",
            "files_unsupported_path_count": unsupported_paths,
            "files_page": files_page,
            "declarations_page": declarations_page,
            "next_page": (
                {
                    "repo_id": repo_id,
                    "revision": info["sha"],
                    "file_offset": files_page["next_offset"]
                    if files_page["next_offset"] is not None
                    else files_page["total"],
                    "declaration_offset": declarations_page["next_offset"]
                    if declarations_page["next_offset"] is not None
                    else declarations_page["total"],
                }
                if files_page["next_offset"] is not None or declarations_page["next_offset"] is not None
                else None
            ),
            "splits_truncated": splits_truncated,
            "splits_total": len(declared_splits),
            "coverage": "partial" if partial else "complete",
            "projection_budget_bytes": MAX_INSPECTION_PROJECTION_BYTES,
            "page_instruction": "Continue by passing the exact next_page mapping to inspect_huggingface_dataset. next_page is null when both inventories are exhausted; exhausted offsets use total, never null. Each page contains whole source fields and is metadata only.",
            "untrusted_data": True,
        }
        if _sdk_projection_bytes(result) > MAX_INSPECTION_PROJECTION_BYTES:
            raise HubDatasetError("metadata projection exceeds the inspection byte budget")
        return result
    except (HubDatasetError, public_fetch.PublicFetchError) as error:
        return {"success": False, "error": str(error)}
    except Exception:
        return {"success": False, "error": "public dataset metadata failed"}


def _output_path(output_dir: str | None) -> tuple[Path, Path]:
    """Validate lexical output scope; this path never substitutes FD authority."""
    if output_dir is None:
        from ..config.settings import Settings

        candidate = Settings.load().local.resolved_data / "huggingface"
    else:
        if (
            "://" in output_dir
            or any(part in (".", "..") for part in output_dir.replace("\\", "/").split("/"))
            or "\\" in output_dir
            or "%" in output_dir
        ):
            raise HubDatasetError("output directory must be a confined local path")
        candidate = Path(os.path.expanduser(output_dir))
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    roots = pathref._allowed_roots()
    root = next((root for root in roots if candidate == root or candidate.is_relative_to(root)), None)
    if root is None:
        raise HubDatasetError("output directory is outside configured local roots")
    return candidate, root


def _output_directory(output_dir: str | None) -> tuple[Path, int]:
    """Open/create original local components under the usual allowed-root guard."""
    candidate, root = _output_path(output_dir)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    current = Path("/")
    try:
        for part in candidate.parts[1:]:
            current = current / part
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not current.is_relative_to(root):
                    raise
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return candidate, descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_json(descriptor: int, name: str, value: dict[str, Any]) -> bytes:
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    handle = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
    with os.fdopen(handle, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    return data


def _named_directory_is_current(path: Path, descriptor: int) -> bool:
    """Compare original no-follow components with a held directory authority."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    chain: list[tuple[int, str, int]] = []
    handles: list[int] = []
    try:
        parent = os.open("/", flags)
        handles.append(parent)
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=parent)
            handles.append(child)
            chain.append((parent, part, child))
            parent = child
        expected = os.fstat(descriptor)
        named = os.fstat(parent)
        if (expected.st_dev, expected.st_ino) != (named.st_dev, named.st_ino):
            return False
        for parent, name, child in chain:
            actual = os.fstat(child)
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(current.st_mode) or (actual.st_dev, actual.st_ino) != (
                current.st_dev,
                current.st_ino,
            ):
                return False
        return True
    except OSError:
        return False
    finally:
        for handle in reversed(handles):
            os.close(handle)


def _duplicate_output_directory(output_dir: str, output_fd: int) -> tuple[Path, int]:
    """Duplicate only caller-owned verified authority, retaining the same OFD lease."""
    if not isinstance(output_dir, str) or type(output_fd) is not int or output_fd < 0:
        raise HubDatasetError("private acquisition requires an output directory descriptor")
    out, _ = _output_path(output_dir)
    descriptor = os.dup(output_fd)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode) or not _named_directory_is_current(out, descriptor):
            raise HubDatasetError("output descriptor does not match the original named directory")
        return out, descriptor
    except Exception:
        os.close(descriptor)
        raise


def _file_identity(descriptor: int) -> tuple[int, int, int, int, int, int]:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        raise HubDatasetError("artifact descriptor is not a regular data file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def _named_file_is_current(name: str, directory: int, descriptor: int, expected: tuple[int, ...]) -> bool:
    try:
        actual = _file_identity(descriptor)
        named = os.stat(name, dir_fd=directory, follow_symlinks=False)
        return (
            actual == expected and stat.S_ISREG(named.st_mode) and (named.st_dev, named.st_ino) == actual[:2]
        )
    except (OSError, HubDatasetError):
        return False


def _verify_artifact_fd(descriptor: int, expected_size: int) -> tuple[tuple[int, ...], str, str, bytes]:
    """Hash/count the retained write FD, bounded independently of stream claims."""
    before = _file_identity(descriptor)
    if before[2] != expected_size:
        raise HubDatasetError("artifact descriptor has a different size")
    os.lseek(descriptor, 0, os.SEEK_SET)
    sha = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(expected_size).encode() + b"\0", usedforsecurity=False)
    count = 0
    prefix = b""
    while True:
        chunk = os.read(descriptor, min(65536, expected_size - count + 1))
        if not chunk:
            break
        count += len(chunk)
        if count > expected_size or count > MAX_FILE_BYTES:
            raise HubDatasetError("artifact descriptor exceeds its bounded size")
        if len(prefix) < 64:
            prefix += chunk[: 64 - len(prefix)]
        sha.update(chunk)
        blob.update(chunk)
    after = _file_identity(descriptor)
    if count != expected_size or before != after:
        raise HubDatasetError("artifact descriptor changed while being verified")
    return after, sha.hexdigest(), blob.hexdigest(), prefix


def _rehash_artifact_fd(
    descriptor: int,
    expected_size: int,
    expected_sha: str,
    expected_blob: str | None,
) -> None:
    """Recheck actual held bytes; unchanged filesystem timestamps prove nothing."""
    _, sha, blob, _ = _verify_artifact_fd(descriptor, expected_size)
    if sha != expected_sha or (expected_blob is not None and blob != expected_blob):
        raise HubDatasetError("artifact bytes changed around manifest publication")


def _acquire_huggingface_dataset_into_fd(
    repo_id: str,
    revision: str,
    filename: str,
    output_dir: str,
    output_fd: int,
) -> dict[str, Any]:
    """Private trusted-worker ABI; FD/privilege/image/namespace trust stays caller-owned.

    The original caller FD is never closed or unlocked. No path-only fallback is
    available; returned absolute paths are metadata for the caller's FD reader.
    """
    return _acquire_huggingface_dataset(
        repo_id,
        revision,
        filename,
        lambda: _duplicate_output_directory(output_dir, output_fd),
    )


def acquire_huggingface_dataset_local(
    repo_id: str,
    revision: str,
    filename: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Acquire via the original local-root guard; success is data, not training."""
    return _acquire_huggingface_dataset(
        repo_id,
        revision,
        filename,
        lambda: _output_directory(output_dir),
    )


def _acquire_huggingface_dataset(
    repo_id: str,
    revision: str,
    filename: str,
    open_authority: Callable[[], tuple[Path, int]],
) -> dict[str, Any]:
    """Common bounded writer, exclusively relative to its held root descriptor."""
    out_fd: int | None = None
    bundle_fd: int | None = None
    data_fd: int | None = None
    bundle_name: str | None = None
    bundle_created = False
    artifact_name: str | None = None
    committed = False
    stage = "validation"
    try:
        _repository(repo_id)
        file_format = _filename(filename)
        if not file_format or not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise HubDatasetError("acquisition requires a full resolved revision and supported data file")
        if not pathref.remote_fetch_enabled():
            raise HubDatasetError("remote public data retrieval is disabled")
        stage = "output_directory"
        out, out_fd = open_authority()
        stage = "metadata"
        info = _metadata(repo_id, revision)
        if info["sha"] != revision:
            raise HubDatasetError("public metadata resolved to a different revision")
        if (
            info.get("private") is not False
            or info.get("gated") is not False
            or info.get("disabled") is not False
        ):
            raise HubDatasetError("only anonymous public ungated enabled datasets can be acquired")
        stage = "file_selection"
        files, _ = _files(info, selected_filename=filename)
        selected = next((entry for entry in files if entry["filename"] == filename), None)
        if selected is None:
            raise HubDatasetError("selected file is not in the bounded pinned repository listing")
        if selected["size"] is None or not 0 < selected["size"] <= MAX_FILE_BYTES:
            raise HubDatasetError("selected file has no usable size or exceeds the 512 MiB cap")
        if selected["sha256"] is None and selected["blob_id"] is None:
            raise HubDatasetError("selected file lacks a verifiable LFS or Git blob hash")
        source_uri = (
            "https://huggingface.co/datasets/"
            + quote(repo_id, safe="/")
            + "/resolve/"
            + revision
            + "/"
            + quote(filename, safe="/")
        )
        stage = "bundle_create"
        if not _named_directory_is_current(out, out_fd):
            raise HubDatasetError("artifact destination changed before bundle creation")
        bundle_name = "acquisition-" + uuid.uuid4().hex
        os.mkdir(bundle_name, mode=0o700, dir_fd=out_fd)
        bundle_created = True
        bundle_fd = os.open(bundle_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=out_fd)
        artifact_name = "data" + PurePosixPath(filename).suffix.lower()
        data_fd = os.open(
            artifact_name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=bundle_fd
        )
        stage = "download"
        with os.fdopen(os.dup(data_fd), "wb") as sink:
            receipt = public_fetch.stream_public_binary(
                source_uri,
                sink,
                max_bytes=MAX_FILE_BYTES,
                timeout=900.0,
                expected_size=selected["size"],
            )
            sink.flush()
            os.fsync(sink.fileno())
        stage = "integrity"
        if receipt["byte_count"] != selected["size"]:
            raise HubDatasetError("acquired payload has a different size")
        if selected["sha256"] is not None:
            if receipt["sha256"] != selected["sha256"]:
                raise HubDatasetError("acquired payload differs from its LFS SHA-256")
        elif receipt["git_blob_sha1"] != selected["blob_id"]:
            raise HubDatasetError("acquired payload differs from its Git-framed blob SHA-1")
        data_identity, actual_sha, actual_blob, prefix = _verify_artifact_fd(data_fd, selected["size"])
        if actual_sha != receipt["sha256"] or (
            selected["sha256"] is None and actual_blob != selected["blob_id"]
        ):
            raise HubDatasetError("artifact descriptor differs from the verified source integrity")
        if not _named_file_is_current(artifact_name, bundle_fd, data_fd, data_identity):
            raise HubDatasetError("artifact name no longer refers to the verified file descriptor")
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise HubDatasetError("a Git LFS pointer is not a dataset payload")
        local_path = str(out / bundle_name / artifact_name)
        lineage_path = str(out / bundle_name / "lineage.json")
        result = {
            "success": True,
            "local_path": local_path,
            "artifact_uri": local_path,
            "provider": "local",
            "format": file_format,
            "sha256": receipt["sha256"],
            "byte_count": receipt["byte_count"],
            "repo_id": repo_id,
            "revision": revision,
            "filename": filename,
            "source_uri": source_uri,
            "lineage_path": lineage_path,
        }
        lineage = {
            **result,
            "prepared": False,
            "trained": False,
            "integrity": "lfs_sha256" if selected["sha256"] is not None else "git_blob_sha1",
            "expected_size": selected["size"],
            "expected_sha256": selected["sha256"],
            "expected_blob_id": selected["blob_id"],
        }
        stage = "lineage"
        lineage_bytes = _write_json(bundle_fd, "lineage.json", lineage)
        if (
            not _named_directory_is_current(out, out_fd)
            or not _named_directory_is_current(out / bundle_name, bundle_fd)
            or not _named_file_is_current(artifact_name, bundle_fd, data_fd, data_identity)
        ):
            raise HubDatasetError("artifact destination changed before its commit")
        stage = "commit"
        _write_json(
            bundle_fd,
            ".commit-pending.json",
            {
                "data": artifact_name,
                "sha256": receipt["sha256"],
                "lineage": "lineage.json",
                "lineage_sha256": hashlib.sha256(lineage_bytes).hexdigest(),
            },
        )
        _rehash_artifact_fd(
            data_fd, selected["size"], actual_sha, selected["blob_id"] if selected["sha256"] is None else None
        )
        if not _named_file_is_current(artifact_name, bundle_fd, data_fd, data_identity):
            raise HubDatasetError("artifact descriptor changed before manifest publication")
        # A link publishes the complete manifest atomically and fails if it exists.
        os.link(
            ".commit-pending.json",
            "COMMITTED.json",
            src_dir_fd=bundle_fd,
            dst_dir_fd=bundle_fd,
            follow_symlinks=False,
        )
        os.unlink(".commit-pending.json", dir_fd=bundle_fd)
        os.fsync(bundle_fd)
        _rehash_artifact_fd(
            data_fd, selected["size"], actual_sha, selected["blob_id"] if selected["sha256"] is None else None
        )
        if (
            not _named_directory_is_current(out, out_fd)
            or not _named_directory_is_current(out / bundle_name, bundle_fd)
            or not _named_file_is_current(artifact_name, bundle_fd, data_fd, data_identity)
        ):
            raise HubDatasetError("artifact destination changed during its commit")
        committed = True
        result["commit_manifest_path"] = str(out / bundle_name / "COMMITTED.json")
        return result
    except (HubDatasetError, public_fetch.PublicFetchError) as error:
        return {"success": False, "provider": "local", "error": str(error), **_failure_context(stage, error)}
    except Exception as error:
        return {
            "success": False,
            "provider": "local",
            "error": "public dataset acquisition failed (" + _safe_error_type(error) + ")",
            **_failure_context(stage, error),
        }
    finally:
        if data_fd is not None:
            os.close(data_fd)
        if bundle_fd is not None:
            if not committed:
                for name in (artifact_name, "lineage.json", ".commit-pending.json", "COMMITTED.json"):
                    if name is not None:
                        try:
                            os.unlink(name, dir_fd=bundle_fd)
                        except FileNotFoundError:
                            pass
            os.close(bundle_fd)
        if out_fd is not None:
            if bundle_created and bundle_name is not None and not committed:
                try:
                    os.rmdir(bundle_name, dir_fd=out_fd)
                except OSError:
                    pass
            os.close(out_fd)
