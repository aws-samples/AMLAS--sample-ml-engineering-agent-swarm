"""Fixed standalone data-only preparation worker; no SDK/model or privilege selectors."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import socket
import stat
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_ID = "openai.explicit_conversations.v1"
POLICY_VERSION = "coding-preparation.v1"
DERIVED_PROFILE_ID = "nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2"
DERIVED_POLICY_VERSION = "coding-preparation.open-swe.v2"
COMMITPACK_PROFILE_ID = "bigcode.commitpackft.python.change_sft.v1"
COMMITPACK_POLICY_VERSION = "coding-preparation.commitpack.v3"
COMMITPACK_EXCLUSIONS_LIMIT = 8 * 1024 * 1024
COMMITPACK_REPORT_LIMIT = COMMITPACK_EXCLUSIONS_LIMIT + 262144
PRIVILEGE_PROFILE = "sagemaker_native_ceiling_v1"
OUTPUT_FILES = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "membership.jsonl",
    "lineage.jsonl",
    "REPORT.json",
)
INPUT_ROOT = Path("/opt/ml/processing/input/preparation")
OUTPUT_ROOT = Path("/opt/ml/processing/output")
MAX_BYTES = 512 * 1024 * 1024
MAX_METADATA = 2 * 1024 * 1024
MAX_RECORD = 32 * 1024 * 1024
HEADER = (
    "job_name",
    "runtime_session_id",
    "deployment_id",
    "deployment_prefix",
    "image_uri",
    "source_commit",
    "profile_id",
    "policy_version",
)


@dataclass(frozen=True)
class WorkerBinding:
    selection_sha256: str
    selection_bytes: int
    exclusions_sha256: str
    exclusions_bytes: int
    job_name: str
    runtime_session_id: str
    deployment_id: str
    deployment_prefix: str
    image_uri: str
    source_commit: str
    profile_id: str = PROFILE_ID
    policy_version: str = POLICY_VERSION


def _root_module(name: str, path: str) -> Any:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_uid != 0
            or value.st_nlink != 1
            or value.st_mode & 0o022
        ):
            raise ValueError("fixed source module refused")
    finally:
        os.close(descriptor)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ValueError("fixed source module unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _native_identity() -> dict[str, Any]:
    # Refuse a direct privileged invocation before loading even the fixed helper.
    if os.getresuid() != (10001, 10001, 10001) or os.getresgid() != (10001, 10001, 10001) or os.getgroups():
        raise ValueError("preparation identity refused")
    helper = _root_module("preparation_identity", "/app/processing_identity.py")
    return helper.verify_native_identity()


def _inherited_descriptor_gate() -> None:
    for name in os.listdir("/proc/self/fd"):
        if not name.isdigit():
            raise ValueError("preparation descriptor inventory refused")
        descriptor = int(name)
        if descriptor <= 3:
            continue
        try:
            os.fstat(descriptor)
        except OSError as error:
            if error.errno == errno.EBADF:
                continue
            raise ValueError("preparation descriptor observation refused") from None
        raise ValueError("unexpected preparation inherited descriptor")
    for descriptor in range(4):
        os.fstat(descriptor)


def _load_core() -> Any:
    return _root_module("preparation_core", "/app/coding_preparation_core.py")


def _output_owner_ok(value: Any) -> bool:
    return value.st_uid == value.st_gid == 10001


def _encoded(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def decode_object(raw: bytes, *, limit: int = MAX_METADATA, depth: int = 16) -> dict[str, Any]:
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError("bounded preparation JSON required")
    text = raw.decode("utf-8")
    stack = []
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            if len(stack) >= depth:
                raise ValueError("preparation JSON depth exceeded")
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if not stack or stack.pop() != char:
                raise ValueError("preparation JSON structure refused")
    if stack or quoted:
        raise ValueError("preparation JSON structure refused")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate preparation JSON field")
            value[key] = item
        return value

    def constant(_value):
        raise ValueError("nonfinite preparation JSON")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if type(value) is not dict:
        raise ValueError("preparation JSON object required")
    return value


def _relative(name: Any) -> tuple[str, ...]:
    if type(name) is not str or not name or name.startswith("/") or "\\" in name:
        raise ValueError("confined preparation filename required")
    parts = tuple(name.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("preparation filename escaped")
    return parts


def _directory(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("fixed original directory required")
    current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _open_relative(directory: int, name: str) -> int:
    parts = _relative(name)
    current = os.dup(directory)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
            os.close(current)
            current = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
    finally:
        os.close(current)


def _signature(value: Any) -> tuple[Any, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_nlink)


def _hash_fd(descriptor: int, expected_count: int, expected_sha: str) -> None:
    before = os.fstat(descriptor)
    if (
        type(expected_count) is not int
        or not 0 < expected_count <= MAX_BYTES
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != expected_count
    ):
        raise ValueError("preparation input count/type refused")
    os.lseek(descriptor, 0, os.SEEK_SET)
    count = 0
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, expected_count - count + 1))
        if not chunk:
            break
        count += len(chunk)
        if count > expected_count:
            raise ValueError("preparation input byte bound exceeded")
        digest.update(chunk)
    if (
        count != expected_count
        or digest.hexdigest() != expected_sha
        or _signature(before) != _signature(os.fstat(descriptor))
    ):
        raise ValueError("preparation exact input changed")
    os.lseek(descriptor, 0, os.SEEK_SET)


def _named_input(directory: int, name: str, descriptor: int) -> None:
    probe = _open_relative(directory, name)
    try:
        if _signature(os.fstat(probe)) != _signature(os.fstat(descriptor)):
            raise ValueError("preparation input name changed")
    finally:
        os.close(probe)


def _output_identity(path: Path, descriptor: int) -> None:
    held = os.fstat(descriptor)
    if not stat.S_ISDIR(held.st_mode) or not _output_owner_ok(held) or stat.S_IMODE(held.st_mode) != 0o700:
        raise ValueError("private output authority refused")
    probe = _directory(path)
    try:
        named = os.fstat(probe)
        if (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
            raise ValueError("original output root changed")
    finally:
        os.close(probe)


def _binding(binding: WorkerBinding) -> None:
    if type(binding) is not WorkerBinding or (binding.profile_id, binding.policy_version) not in (
        (PROFILE_ID, POLICY_VERSION),
        (DERIVED_PROFILE_ID, DERIVED_POLICY_VERSION),
        (COMMITPACK_PROFILE_ID, COMMITPACK_POLICY_VERSION),
    ):
        raise ValueError("unsupported preparation profile")
    if (
        not re.fullmatch(r"[a-f0-9]{40}", binding.source_commit)
        or os.environ.get("AMLAS_SOURCE_COMMIT") != binding.source_commit
    ):
        raise ValueError("preparation source binding differs")
    for count, limit in (
        (binding.selection_bytes, MAX_METADATA),
        (
            binding.exclusions_bytes,
            COMMITPACK_EXCLUSIONS_LIMIT if binding.profile_id == COMMITPACK_PROFILE_ID else MAX_METADATA,
        ),
    ):
        if type(count) is not int or not 0 < count <= limit:
            raise ValueError("invalid preparation metadata count")
    for digest in (binding.selection_sha256, binding.exclusions_sha256):
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("invalid preparation metadata digest")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", binding.runtime_session_id) or not re.fullmatch(
        r"[A-Za-z0-9-]{1,63}", binding.job_name
    ):
        raise ValueError("invalid preparation correlation")


def _read_metadata(
    input_fd: int, name: str, count: int, digest: str, held: list, *, limit: int = MAX_METADATA
) -> dict[str, Any]:
    descriptor = _open_relative(input_fd, name)
    try:
        _hash_fd(descriptor, count, digest)
        raw = os.read(descriptor, count + 1)
        if len(raw) != count or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("preparation consumed metadata bytes differ")
        value = decode_object(raw, limit=limit)
        _hash_fd(descriptor, count, digest)
        _named_input(input_fd, name, descriptor)
        held.append((name, descriptor, count, digest))
        return value
    except BaseException:
        os.close(descriptor)
        raise


def _entry(entry: dict[str, Any], *, data: bool) -> None:
    required = {
        "relative_name",
        "byte_count",
        "sha256",
        "original_uri",
        "original_version_id",
        "staged_version_id",
    } | ({"format"} if data else {"name"})
    if type(entry) is not dict or set(entry) != required:
        raise ValueError("fixed preparation file binding required")
    _relative(entry["relative_name"])
    if type(entry["byte_count"]) is not int or not 0 < entry["byte_count"] <= (
        MAX_BYTES if data else MAX_METADATA
    ):
        raise ValueError("invalid preparation file count")
    if type(entry["sha256"]) is not str or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
        raise ValueError("invalid preparation file digest")
    for name in ("original_uri", "original_version_id", "staged_version_id"):
        value = entry[name]
        if type(value) is not str or value in ("", "null") or any(ord(char) < 32 for char in value):
            raise ValueError("immutable preparation file identity required")
    if data and entry["format"] not in ("jsonl", "parquet"):
        raise ValueError("unsupported preparation parser mapping")


def _load_derived_profile() -> Any:
    """Fixed root-owned pure package, loaded only after common worker guards."""
    import importlib

    _root_module("preparation_profiles", "/app/preparation_profiles/__init__.py")
    for name in ("coding_preparation", "open_swe_canonical", "open_swe_derived"):
        _root_module("preparation_profiles." + name, "/app/preparation_profiles/" + name + ".py")
    return importlib.import_module("preparation_profiles.open_swe_derived")


def _load_commitpack_profile() -> Any:
    import importlib

    _root_module("preparation_profiles", "/app/preparation_profiles/__init__.py")
    for name in ("coding_preparation", "commitpack_profile", "commitpack_preparation"):
        _root_module("preparation_profiles." + name, "/app/preparation_profiles/" + name + ".py")
    return importlib.import_module("preparation_profiles.commitpack_preparation")


def _derived_rows(
    descriptor: int,
    source: dict[str, Any],
    policy: Any,
    codec: Any,
    selection_budget: dict[str, int] | None = None,
):
    """Parse selected complete columns in the isolated guest, never in a parent."""
    import pyarrow.parquet as parquet

    data = source["data"]
    if (
        source["repo_id"] != "nvidia/Open-SWE-Traces"
        or source["revision"]
        != "1db674173119d71df45c22007af99e214600711a"  # pragma: allowlist secret - public dataset revision
        or data["format"] != "parquet"
    ):
        raise ValueError("unsupported pinned OpenSWE source")
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        parsed = parquet.ParquetFile(
            stream,
            thrift_string_size_limit=16 * 1024**2,
            thrift_container_size_limit=1000000,
            arrow_extensions_enabled=False,
            pre_buffer=False,
            page_checksum_verification=True,
        )
        columns = ["instance_id", "repo", "license", "trajectory_id", "messages", "tools", "resolved"]
        if (
            not set(columns) <= set(parsed.schema_arrow.names)
            or len(set(parsed.schema_arrow.names)) != len(parsed.schema_arrow.names)
            or len(parsed.schema_arrow.names) > 1024
            or parsed.metadata.num_columns > 1024
            or parsed.metadata.num_row_groups > policy.max_records
            or parsed.metadata.num_rows > policy.max_records
        ):
            raise ValueError("OpenSWE Parquet schema or row bound refused")
        budget = selection_budget if selection_budget is not None else {"expanded": 0, "json": 0}
        for group_index in range(parsed.metadata.num_row_groups):
            group = parsed.metadata.row_group(group_index)
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                if column.path_in_schema.split(".", 1)[0] not in columns:
                    continue
                expanded = column.total_uncompressed_size
                if type(expanded) is not int or expanded < 0:
                    raise ValueError("OpenSWE Parquet expansion metadata refused")
                budget["expanded"] += expanded
                if budget["expanded"] > policy.max_total_bytes:
                    raise ValueError("OpenSWE Parquet expansion exceeds selection bound")
        decoded = 0
        ordinal = 0
        for batch in parsed.iter_batches(batch_size=1, columns=columns):
            if batch.nbytes > MAX_RECORD or batch.nbytes > policy.max_total_bytes - decoded:
                raise ValueError("OpenSWE decoded batch exceeds bounds")
            decoded += batch.nbytes
            for row in batch.to_pylist():
                encoded_count = len(_encoded(row))
                budget["json"] += encoded_count
                if encoded_count > MAX_RECORD or budget["json"] > policy.max_total_bytes:
                    raise ValueError("OpenSWE decoded JSON exceeds selection bound")
                bound = codec.OpenSWESource(
                    dataset_repo=source["repo_id"],
                    revision=source["revision"],
                    config_name="v1.2",
                    filename=source["filename"],
                    artifact_uri=data["original_uri"],
                    artifact_version_id=data["original_version_id"],
                    artifact_sha256=data["sha256"],
                    row_index=ordinal,
                )
                yield ordinal, row, codec.normalize_open_swe_row_derived(row, source=bound)
                ordinal += 1
        if ordinal != parsed.metadata.num_rows:
            raise ValueError("OpenSWE complete row count differs")


def _verify_held(input_fd: int, held: list) -> None:
    for name, descriptor, count, digest in held:
        _hash_fd(descriptor, count, digest)
        _named_input(input_fd, name, descriptor)


def _write_output(output_fd: int, name: str, raw: bytes, held: list) -> dict[str, Any]:
    descriptor = os.open(
        name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=output_fd
    )
    held.append((name, descriptor, raw))
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise ValueError("preparation output write refused")
        offset += written
    os.fsync(descriptor)
    _verify_outputs(output_fd, [(name, descriptor, raw)])
    return {
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "format": "json" if name.endswith(".json") else "jsonl",
        "record_count": len(raw.splitlines()) if name.endswith(".jsonl") else 1,
    }


def _verify_outputs(output_fd: int, held: list) -> None:
    for name, descriptor, raw in held:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != len(raw):
            raise ValueError("preparation output identity differs")
        os.lseek(descriptor, 0, os.SEEK_SET)
        count = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, len(raw) - count + 1))
            if not chunk:
                break
            count += len(chunk)
            digest.update(chunk)
            if count > len(raw):
                raise ValueError("preparation output count differs")
        named = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
        if (
            count != len(raw)
            or digest.hexdigest() != hashlib.sha256(raw).hexdigest()
            or _signature(before) != _signature(os.fstat(descriptor))
            or _signature(named) != _signature(before)
        ):
            raise ValueError("preparation published bytes changed")


def _commitpack_nonusable(
    outcome: dict[str, Any],
    input_fd: int,
    held_inputs: list,
    output_root: Path,
    output_fd: int,
    held_outputs: list,
) -> dict[str, Any]:
    _verify_held(input_fd, held_inputs)
    _output_identity(output_root, output_fd)
    _write_output(output_fd, "REPORT.json", _encoded(outcome), held_outputs)
    _verify_outputs(output_fd, held_outputs)
    _output_identity(output_root, output_fd)
    if set(os.listdir(output_fd)) != {"REPORT.json"}:
        raise ValueError("CommitPack nonusable output inventory differs")
    return outcome


def run_bound_preparation(
    binding: WorkerBinding, input_root: Path, output_root: Path, output_fd: int
) -> dict[str, Any]:
    """Private trusted-FD oracle; managed main exposes no path or FD selector."""
    proof = _native_identity()
    _binding(binding)
    _output_identity(output_root, output_fd)
    fcntl.flock(output_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.listdir(output_fd):
        raise ValueError("preparation output must be empty")
    input_fd = _directory(input_root)
    held_inputs = []
    held_outputs = []
    try:
        selection = _read_metadata(
            input_fd, "selection.json", binding.selection_bytes, binding.selection_sha256, held_inputs
        )
        if (
            set(selection) != {"schema_version", *HEADER, "sources", "exclusions"}
            or type(selection["schema_version"]) is not int
            or selection["schema_version"] != 1
            or any(selection[name] != getattr(binding, name) for name in HEADER)
        ):
            raise ValueError("preparation selection header differs")
        commitpack = binding.profile_id == COMMITPACK_PROFILE_ID
        exclusions = _read_metadata(
            input_fd,
            "exclusions.json",
            binding.exclusions_bytes,
            binding.exclusions_sha256,
            held_inputs,
            limit=COMMITPACK_EXCLUSIONS_LIMIT if commitpack else MAX_METADATA,
        )
        cp = _load_commitpack_profile() if commitpack else None
        if commitpack:
            cp.validate_commitpack_exclusions(exclusions)
            if (
                hashlib.sha256(cp.encode_commitpack_metadata(exclusions)).hexdigest()
                != binding.exclusions_sha256
            ):
                raise ValueError("CommitPack staged exclusion bytes differ")
        elif set(exclusions) != {"instance_ids", "patch_sha256"}:
            raise ValueError("fixed preparation exclusion schema required")
        bound_exclusions = selection["exclusions"]
        if (
            bound_exclusions.get("byte_count") != binding.exclusions_bytes
            or bound_exclusions.get("sha256") != binding.exclusions_sha256
            or not bound_exclusions.get("original_version_id")
            or not bound_exclusions.get("staged_version_id")
        ):
            raise ValueError("preparation exclusion identity differs")
        sources = selection["sources"]
        if type(sources) is not list or not 0 < len(sources) <= 256:
            raise ValueError("bounded preparation selection required")
        core = _load_core()
        policy = core.PreparationPolicy()
        records = []
        commitpack_rows = []
        commitpack_budget = {"expanded": 0, "normalized": 0, "rejected": 0}
        derived = binding.profile_id == DERIVED_PROFILE_ID
        codec = _load_derived_profile() if derived else None
        normalization = {}
        normalization_refused = []
        normalization_reasons = Counter()
        normalization_source_count = 0
        selection_budget = {"expanded": 0, "json": 0}
        input_names = {"selection.json", "exclusions.json"}
        for source in sources:
            if type(source) is not dict or set(source) != {
                "data",
                "manifest_files",
                "repo_id",
                "revision",
                "filename",
                "acquisition_image_uri",
                "acquisition_source_commit",
            }:
                raise ValueError("fixed preparation source schema required")
            data = source["data"]
            _entry(data, data=True)
            if data["format"] != ("parquet" if derived else "jsonl"):
                raise ValueError("source format differs from fixed preparation profile")
            manifests = source["manifest_files"]
            if type(manifests) is not list or len(manifests) != 3:
                raise ValueError("complete acquisition bundle required")
            parsed = {}
            raw_digests = {}
            for entry in manifests:
                _entry(entry, data=False)
                if (
                    entry["name"] in parsed
                    or entry["name"] not in ("receipt", "commit", "lineage")
                    or entry["relative_name"] in input_names
                ):
                    raise ValueError("duplicate preparation manifest")
                input_names.add(entry["relative_name"])
                parsed[entry["name"]] = _read_metadata(
                    input_fd, entry["relative_name"], entry["byte_count"], entry["sha256"], held_inputs
                )
                raw_digests[entry["name"]] = entry["sha256"]
            if (
                parsed["receipt"].get("commit_manifest_sha256") != raw_digests["commit"]
                or parsed["commit"].get("lineage_sha256") != raw_digests["lineage"]
                or parsed["commit"].get("sha256") != data["sha256"]
            ):
                raise ValueError("acquisition bundle integrity differs")
            lineage = parsed["lineage"]
            if (
                any(lineage.get(key) != data[key] for key in ("sha256", "byte_count", "format"))
                or any(
                    lineage.get(key) != source[key] or parsed["receipt"].get(key) != source[key]
                    for key in ("repo_id", "revision", "filename")
                )
                or parsed["receipt"].get("image_uri") != source["acquisition_image_uri"]
                or parsed["receipt"].get("source_commit") != source["acquisition_source_commit"]
            ):
                raise ValueError("acquisition source provenance differs")
            if data["relative_name"] in input_names:
                raise ValueError("duplicate preparation input")
            input_names.add(data["relative_name"])
            descriptor = _open_relative(input_fd, data["relative_name"])
            held_inputs.append((data["relative_name"], descriptor, data["byte_count"], data["sha256"]))
            _hash_fd(descriptor, data["byte_count"], data["sha256"])
            if commitpack:
                if (
                    source["repo_id"],
                    source["revision"],
                    source["filename"],
                    data["sha256"],
                    data["byte_count"],
                ) != (
                    cp.profile.DATASET_REPO,
                    cp.profile.REVISION,
                    cp.profile.FILENAME,
                    cp.profile.ARTIFACT_SHA256,
                    cp.SOURCE_BYTES,
                ):
                    raise ValueError("fixed CommitPack source pins differ")
                parsed_digest = hashlib.sha256()
                parsed_bytes = 0
                with os.fdopen(os.dup(descriptor), "rb") as stream:
                    ordinal = 0
                    while True:
                        raw = stream.readline(MAX_RECORD + 1)
                        if not raw:
                            break
                        parsed_digest.update(raw)
                        parsed_bytes += len(raw)
                        if len(commitpack_rows) >= cp.MAX_ROWS:
                            outcome = cp._nonusable(
                                "source_row_bound",
                                len(commitpack_rows),
                                commitpack_budget["normalized"],
                                commitpack_budget["rejected"],
                            )["nonusable_outcome"]
                            return _commitpack_nonusable(
                                outcome, input_fd, held_inputs, output_root, output_fd, held_outputs
                            )
                        try:
                            if len(raw) > MAX_RECORD or not raw.strip():
                                raise ValueError("CommitPack complete row decoding refused")
                            row = decode_object(raw, limit=MAX_RECORD, depth=16)
                        except (ValueError, TypeError, UnicodeError, RecursionError):
                            outcome = cp._nonusable(
                                "source_decode_failure",
                                len(commitpack_rows),
                                commitpack_budget["normalized"],
                                commitpack_budget["rejected"],
                            )["nonusable_outcome"]
                            return _commitpack_nonusable(
                                outcome, input_fd, held_inputs, output_root, output_fd, held_outputs
                            )
                        bound = cp.CommitPackSource(
                            dataset_repo=source["repo_id"],
                            revision=source["revision"],
                            filename=source["filename"],
                            artifact_uri=data["original_uri"],
                            artifact_version_id=data["original_version_id"],
                            artifact_sha256=data["sha256"],
                            row_index=ordinal,
                        )
                        envelope = cp.profile.normalize_commitpack_row(row, source=bound)
                        if envelope["source_provenance"] is None:
                            raise ValueError("CommitPack source provenance refused")
                        # The retained parsed row must not hide behind a small rejected envelope.
                        commitpack_budget["expanded"] += max(
                            len(cp.profile._encoded(envelope)), len(_encoded(row))
                        )
                        commitpack_budget[
                            "normalized" if envelope["status"] == "normalized" else "rejected"
                        ] += 1
                        if commitpack_budget["expanded"] > cp.MAX_EXPANDED_BYTES:
                            outcome = cp._nonusable(
                                "group_resource_bound",
                                len(commitpack_rows) + 1,
                                commitpack_budget["normalized"],
                                commitpack_budget["rejected"],
                            )["nonusable_outcome"]
                            return _commitpack_nonusable(
                                outcome, input_fd, held_inputs, output_root, output_fd, held_outputs
                            )
                        commitpack_rows.append(
                            cp.CommitPackParsedRow(row, bound, hashlib.sha256(raw).hexdigest())
                        )
                        ordinal += 1
                if parsed_bytes != data["byte_count"] or parsed_digest.hexdigest() != data["sha256"]:
                    raise ValueError("CommitPack consumed source bytes differ")
                _hash_fd(descriptor, data["byte_count"], data["sha256"])
                _named_input(input_fd, data["relative_name"], descriptor)
                continue
            if derived:
                for ordinal, _original, normalized in _derived_rows(
                    descriptor, source, policy, codec, selection_budget
                ):
                    locator = (
                        data["original_uri"]
                        + "#version="
                        + data["original_version_id"]
                        + "#row="
                        + str(ordinal)
                    )
                    provenance = {
                        "source_locator": locator,
                        "source_version_id": data["original_version_id"],
                        "source_sha256": data["sha256"],
                    }
                    status = normalized["status"]
                    sidecar = {
                        "profile_id": DERIVED_PROFILE_ID,
                        "status": "accepted" if status == "normalized" else "rejected",
                        "normalized_record_sha256": normalized["normalized_record_sha256"],
                        "normalization_lineage": normalized["normalization_lineage"],
                        "source_provenance": normalized["source_provenance"],
                    }
                    normalization_source_count += 1
                    if normalization_source_count > policy.max_records:
                        raise ValueError("OpenSWE selection row bound exceeded")
                    if status == "normalized":
                        record = normalized["record"]
                        record.update(provenance)
                        records.append(record)
                        normalization[locator] = sidecar
                    elif status == "rejected" and normalized["issues"]:
                        reason = normalized["issues"][0]["reason"]
                        if type(reason) is not str or not re.fullmatch(r"[a-z0-9_]{1,128}", reason):
                            raise ValueError("unknown OpenSWE normalization reason")
                        normalization_reasons[reason] += 1
                        sidecar["reason"] = reason
                        normalization_refused.append(
                            {
                                **provenance,
                                "record_id": None,
                                "decision": "normalization_" + reason,
                                "cluster_id": None,
                                "representative_id": None,
                                "normalization": sidecar,
                            }
                        )
                    else:
                        raise ValueError("unknown OpenSWE normalization result")
                _hash_fd(descriptor, data["byte_count"], data["sha256"])
                _named_input(input_fd, data["relative_name"], descriptor)
                continue
            stream = os.fdopen(os.dup(descriptor), "rb")
            try:
                ordinal = 0
                while True:
                    raw = stream.readline(MAX_RECORD + 1)
                    if not raw:
                        break
                    if len(raw) > MAX_RECORD or not raw.strip():
                        raise ValueError("preparation record envelope refused")
                    record = decode_object(raw, limit=MAX_RECORD, depth=policy.max_json_depth)
                    if any(
                        type(record.get(key)) is not str
                        or not record[key]
                        or any(ord(char) < 32 for char in record[key])
                        for key in ("repository", "instance_id", "trajectory_id")
                    ):
                        raise ValueError("unadmitted canonical source identity")
                    record["source_locator"] = (
                        data["original_uri"]
                        + "#version="
                        + data["original_version_id"]
                        + "#row="
                        + str(ordinal)
                    )
                    record["source_version_id"] = data["original_version_id"]
                    record["source_sha256"] = data["sha256"]
                    records.append(record)
                    ordinal += 1
                    if len(records) > policy.max_records:
                        raise ValueError("preparation selection exceeded profile envelope")
            finally:
                stream.close()
            _hash_fd(descriptor, data["byte_count"], data["sha256"])
            _named_input(input_fd, data["relative_name"], descriptor)
        if commitpack:
            result = cp.build_commitpack_preparation(commitpack_rows, exclusion_policy=exclusions)
            if result["status"] != "complete":
                return _commitpack_nonusable(
                    result["nonusable_outcome"], input_fd, held_inputs, output_root, output_fd, held_outputs
                )
            _verify_held(input_fd, held_inputs)
            _output_identity(output_root, output_fd)
            files = {}
            for split in cp.SPLITS:
                files[split + ".jsonl"] = _write_output(
                    output_fd,
                    split + ".jsonl",
                    b"".join(cp.encode_commitpack_metadata(value) for value in result["splits"][split]),
                    held_outputs,
                )
            for name in ("membership", "lineage"):
                files[name + ".jsonl"] = _write_output(
                    output_fd,
                    name + ".jsonl",
                    b"".join(cp.encode_commitpack_metadata(value) for value in result[name]),
                    held_outputs,
                )
            report_raw = cp.encode_commitpack_metadata(result["report"])
            if len(report_raw) > COMMITPACK_REPORT_LIMIT:
                raise ValueError("CommitPack report bound exceeded")
            files["REPORT.json"] = _write_output(output_fd, "REPORT.json", report_raw, held_outputs)
            committed = {
                "schema_version": 2,
                **{name: getattr(binding, name) for name in HEADER},
                "privilege_profile": PRIVILEGE_PROFILE,
                "selection_sha256": binding.selection_sha256,
                "exclusions_sha256": binding.exclusions_sha256,
                "files": files,
                "counts": result["report"]["counts"],
                "coverage": result["report"]["coverage"],
                "guest_evidence": {
                    "privilege_proof": proof,
                    "credential_observation": "env_and_home_file_presence",
                    "network_observation": "fixed_public_and_metadata_tcp_denial",
                },
            }
            _verify_held(input_fd, held_inputs)
            _verify_outputs(output_fd, held_outputs)
            _output_identity(output_root, output_fd)
            _write_output(output_fd, "COMMITTED.json", _encoded(committed), held_outputs)
            _verify_held(input_fd, held_inputs)
            _verify_outputs(output_fd, held_outputs)
            _output_identity(output_root, output_fd)
            if set(os.listdir(output_fd)) != {*OUTPUT_FILES, "COMMITTED.json"}:
                raise ValueError("CommitPack output inventory differs")
            return committed
        result = core.prepare_canonical_conversations(records, policy=policy, exclusions=exclusions)
        if result["status"] != "complete" or result["report"]["conservation"] is not True:
            raise ValueError("preparation selection incomplete")
        if derived:
            for entry in result["lineage"]:
                entry["normalization"] = normalization[entry["source_locator"]]
            result["lineage"].extend(normalization_refused)
            if (
                normalization_source_count != len(records) + len(normalization_refused)
                or len(result["lineage"]) != normalization_source_count
            ):
                raise ValueError("OpenSWE normalization lineage conservation differs")
        _verify_held(input_fd, held_inputs)
        _output_identity(output_root, output_fd)
        files = {}
        splits = {name: [] for name in ("train", "validation", "test")}
        if len(result["canonical_records"]) != len(result["membership"]):
            raise ValueError("preparation membership differs")
        for record, member in zip(result["canonical_records"], result["membership"], strict=True):
            splits[member["split"]].append(record)
        for split, values in splits.items():
            files[split + ".jsonl"] = _write_output(
                output_fd, split + ".jsonl", b"".join(_encoded(value) for value in values), held_outputs
            )
        for name in ("membership", "lineage"):
            files[name + ".jsonl"] = _write_output(
                output_fd, name + ".jsonl", b"".join(_encoded(value) for value in result[name]), held_outputs
            )
        counts = {
            "source_count": normalization_source_count if derived else result["report"]["source_count"],
            "retained_representatives": result["report"]["retained_representatives"],
            "primary_drop_count": sum(result["primary_drop_counts"].values()) + len(normalization_refused),
            "unresolved_record_count": result["report"]["unresolved_record_count"],
            "split_counts": result["report"]["split_counts"],
        }
        report = {
            "schema_version": 1,
            "status": "complete",
            "counts": counts,
            "primary_drop_counts": result["primary_drop_counts"],
            "secret_pattern_counts": result["report"]["secret_pattern_counts"],
            "coverage": {
                "near_dedup": "not_implemented",
                "exclusion_scope": result["report"]["exclusion_scope"],
            },
        }
        if derived:
            report.update(
                schema_version=2,
                normalization_source_count=normalization_source_count,
                normalization_accepted_count=len(records),
                normalization_rejected_count=len(normalization_refused),
                normalization_rejection_reasons=dict(sorted(normalization_reasons.items())),
                derived_edge_count=sum(len(item["normalization_lineage"]) for item in normalization.values()),
                linkage_status="derived_not_original_source_verified",
            )
        files["REPORT.json"] = _write_output(output_fd, "REPORT.json", _encoded(report), held_outputs)
        committed = {
            "schema_version": 2,
            **{name: getattr(binding, name) for name in HEADER},
            "privilege_profile": PRIVILEGE_PROFILE,
            "selection_sha256": binding.selection_sha256,
            "exclusions_sha256": binding.exclusions_sha256,
            "files": files,
            "counts": counts,
            "coverage": report["coverage"],
            "guest_evidence": {
                "privilege_proof": proof,
                "credential_observation": "env_and_home_file_presence",
                "network_observation": "fixed_public_and_metadata_tcp_denial",
            },
        }
        _verify_held(input_fd, held_inputs)
        _verify_outputs(output_fd, held_outputs)
        _output_identity(output_root, output_fd)
        _write_output(output_fd, "COMMITTED.json", _encoded(committed), held_outputs)
        _verify_held(input_fd, held_inputs)
        _verify_outputs(output_fd, held_outputs)
        _output_identity(output_root, output_fd)
        if set(os.listdir(output_fd)) != {*OUTPUT_FILES, "COMMITTED.json"}:
            raise ValueError("preparation output inventory changed")
        return committed
    except BaseException:
        for name, _descriptor, _raw in reversed(held_outputs):
            try:
                os.unlink(name, dir_fd=output_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        for _name, descriptor, _count, _digest in held_inputs:
            os.close(descriptor)
        for _name, descriptor, _raw in held_outputs:
            os.close(descriptor)
        os.close(input_fd)


def _absence_observations() -> None:
    forbidden = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    }
    if any(key in os.environ for key in forbidden) or any(
        key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"} for key in os.environ
    ):
        raise ValueError("preparation credential/proxy presence refused")
    for name in ("credentials", "config"):
        if (Path.home() / ".aws" / name).exists():
            raise ValueError("preparation credential-file presence refused")
    for address, port in (("1.1.1.1", 443), ("169.254.169.254", 80)):
        try:
            connection = socket.create_connection((address, port), timeout=0.5)
        except OSError:
            continue
        else:
            connection.close()
            raise ValueError("preparation fixed network observation refused")


def main() -> None:
    _native_identity()
    _inherited_descriptor_gate()
    parser = argparse.ArgumentParser()
    for name in (
        "selection-sha256",
        "selection-bytes",
        "exclusions-sha256",
        "exclusions-bytes",
        *[name.replace("_", "-") for name in HEADER],
    ):
        parser.add_argument("--" + name, required=True, type=int if name.endswith("-bytes") else str)
    binding = WorkerBinding(**vars(parser.parse_args()))
    _binding(binding)
    _absence_observations()
    run_bound_preparation(binding, INPUT_ROOT, OUTPUT_ROOT, 3)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError):
        print("PREPARATION-REFUSED", file=sys.stderr)
        raise SystemExit(78) from None
