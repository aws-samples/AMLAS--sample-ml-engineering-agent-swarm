"""Fixed data-only inspection guest; standalone image copies this audited file."""

from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import signal
import socket
import stat
import sys
from pathlib import Path
from typing import Any

schema: Any = None


def _load_schema() -> Any:
    """Load only fixed image-owned code after the permanent privilege guard."""
    global schema
    if schema is None:
        if __package__:
            from amlas.data_eng import inspection_schema as loaded
        else:
            if Path(__file__).resolve() != Path("/app/inspect_dataset.py"):
                raise ValueError("inspection requires the fixed image-owned worker path")
            sys.path.insert(0, "/app")
            import inspection_schema as loaded
        schema = loaded
    return schema


_CREDENTIAL_ENV = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT",
    }
)


def _network_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _credential_probe() -> tuple[bool, str | None]:
    import boto3

    session = boto3.Session()
    session._session.set_config_variable("metadata_service_timeout", 1)
    session._session.set_config_variable("metadata_service_num_attempts", 1)

    def timed_out(signum: int, frame: Any) -> None:
        raise ValueError("credential presence probe exceeded its deadline")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, 5.0)
    try:
        credentials = session.get_credentials()
        return credentials is not None, getattr(credentials, "method", None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


_PROXY_ENV = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)

_PROCESS_FIELDS = frozenset({"CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd", "NoNewPrivs"})


def _read_process_status() -> dict[str, str]:
    """Read only bounded kernel privilege metadata, never credentials or input data."""
    descriptor = os.open("/proc/self/status", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read(16385)
    if not 0 < len(raw) <= 16384:
        raise ValueError("process privilege metadata exceeds its bound")
    values: dict[str, str] = {}
    try:
        for line in raw.decode("ascii").splitlines():
            name, separator, value = line.partition(":")
            if separator and name in _PROCESS_FIELDS:
                if name in values:
                    raise ValueError("duplicate process privilege metadata")
                values[name] = value.strip()
    except UnicodeError:
        raise ValueError("invalid process privilege metadata") from None
    if set(values) != _PROCESS_FIELDS:
        raise ValueError("incomplete process privilege metadata")
    return values


def _process_privilege_proof() -> dict[str, Any]:
    """Verify actual IDs/capabilities/NNP before importing the SDK or data libraries."""
    uid, gid, groups = list(os.getresuid()), list(os.getresgid()), os.getgroups()
    if uid != [10001] * 3 or gid != [10001] * 3 or groups:
        raise ValueError("process privilege identity has not permanently dropped")
    values = _read_process_status()
    try:
        proof = {
            "res_uid": uid,
            "res_gid": gid,
            "groups": groups,
            "cap_eff": int(values["CapEff"], 16),
            "cap_prm": int(values["CapPrm"], 16),
            "cap_inh": int(values["CapInh"], 16),
            "cap_amb": int(values["CapAmb"], 16),
            "cap_bnd": int(values["CapBnd"], 16),
            "no_new_privs": int(values["NoNewPrivs"]),
        }
    except (KeyError, ValueError):
        raise ValueError("invalid process privilege metadata") from None
    if (
        any(proof[name] != 0 for name in ("cap_eff", "cap_prm", "cap_inh", "cap_amb", "cap_bnd"))
        or proof["no_new_privs"] != 1
    ):
        raise ValueError("process privilege capabilities or no-new-privileges differ")
    return proof


_OUTPUT_FD = 3
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_directory_path(path: Path) -> int:
    """Open the original lexical components; never resolve away a symlink."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("fixed directory requires an absolute original path")
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in path.parts[1:]:
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _input_stat_binding(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _verify_output_identity(path: Path) -> None:
    try:
        named = _open_directory_path(path)
        try:
            if not _same_identity(os.fstat(_OUTPUT_FD), os.fstat(named)):
                raise ValueError("output directory named identity changed")
        finally:
            os.close(named)
    except OSError:
        raise ValueError("output directory original path refused") from None


def _output_directory_proof() -> dict[str, Any]:
    """Observe current fixed-directory authority and lease without an ENV grant."""
    directory = os.fstat(_OUTPUT_FD)
    if (
        not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != 10001
        or directory.st_gid != 10001
        or stat.S_IMODE(directory.st_mode) != 0o700
    ):
        raise ValueError("output directory ownership or mode differs")
    _verify_output_identity(Path("/opt/ml/processing/output"))
    # Preserve the inherited open-file-description lease for the worker lifetime.
    # Reacquisition proves only the current observation, not historical continuity.
    fcntl.flock(_OUTPUT_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contender = _open_directory_path(Path("/opt/ml/processing/output"))
    try:
        try:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in {errno.EAGAIN, errno.EACCES}:
                raise ValueError("output directory lease probe refused") from None
        else:
            fcntl.flock(contender, fcntl.LOCK_UN)
            raise ValueError("output directory lease is not exclusive")
    finally:
        os.close(contender)
    for name in os.listdir("/proc/self/fd"):
        descriptor = int(name)
        if descriptor <= 3:
            continue
        try:
            os.fstat(descriptor)
        except OSError as error:
            if error.errno == errno.EBADF:
                continue  # The temporary enumeration FD has already closed.
            raise
        raise ValueError("unexpected inherited file descriptor")
    return {
        "fd": 3,
        "uid": directory.st_uid,
        "gid": directory.st_gid,
        "mode": stat.S_IMODE(directory.st_mode),
        "device": directory.st_dev,
        "inode": directory.st_ino,
        "lease_active": True,
    }


def _native_process_privilege_proof() -> dict[str, Any]:
    """Consume only the fixed image-owned, stdlib-only shared native guard."""
    if not __package__:
        if Path(__file__).resolve() != Path("/app/inspect_dataset.py"):
            raise ValueError("process privilege requires the fixed image-owned native worker")
        sys.path.insert(0, "/app")
    import processing_identity

    return processing_identity.verify_native_identity()


def _guest_preflight() -> dict[str, Any]:
    return _preflight_with_privilege(_process_privilege_proof())


def _native_guest_preflight() -> dict[str, Any]:
    return _preflight_with_privilege(_native_process_privilege_proof())


def _preflight_with_privilege(privilege: dict[str, Any]) -> dict[str, Any]:
    output = _output_directory_proof()
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise ValueError("inspection requires admitted non-root AMD64 execution")
    if "AWS_EC2_METADATA_DISABLED" in os.environ:
        raise ValueError("disabled metadata discovery cannot establish credential absence")
    if any(name in os.environ for name in _PROXY_ENV):
        raise ValueError("proxy provider environment is present")
    names = sorted(name for name in _CREDENTIAL_ENV if name in os.environ)
    if names:
        raise ValueError("credential provider environment is present")
    present, method = _credential_probe()
    metadata = _network_reachable("169.254.169.254", 80) or _network_reachable("169.254.170.2", 80)
    network = _network_reachable("1.1.1.1", 443)
    result = {
        "credential_present": present,
        "provider_method": method,
        "credential_env_names": names,
        "metadata_reachable": metadata,
        "network_reachable": network,
        "privilege_proof": privilege,
        "output_directory_proof": output,
    }
    if present or method is not None or metadata or network:
        raise ValueError("guest credential or network-denial probe failed")
    return result


def _object_bytes(value: Any, limit: int) -> int:
    import itertools

    count = 0
    pending = [iter((value,))]
    while pending:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        count += sys.getsizeof(item)
        if count > limit:
            raise ValueError("decoded inspection allocation bound exceeded")
        if isinstance(item, dict):
            pending.append(itertools.chain(item.keys(), item.values()))
        elif isinstance(item, (list, tuple)):
            pending.append(iter(item))
    return count


def _inspect_file(file: Any, fmt: str, target: str | None) -> dict[str, Any]:
    import pandas as pd

    records: list[dict[str, Any]] = []
    columns: set[str] = set()
    decoded = 0
    total: int | None = None
    extra = False

    def admit(row: Any) -> None:
        nonlocal decoded
        if not isinstance(row, dict):
            raise ValueError("inspection rows must be data objects")
        columns.update(row)
        if len(columns) > schema.MAX_COLUMNS or any(
            not isinstance(col, str) or not 1 <= len(col) <= 128 for col in columns
        ):
            raise ValueError("inspection column bound exceeded")
        decoded += _object_bytes(row, schema.MAX_DECODED_BYTES - decoded)
        if decoded + (len(records) + 1) * (len(columns) + 1) * 16 > schema.MAX_DECODED_BYTES:
            raise ValueError("dense inspection allocation bound exceeded")
        records.append(row)

    if fmt == "csv":
        csv.field_size_limit(8 * 1024 * 1024)
        text = io.TextIOWrapper(file, encoding="utf-8", newline="")
        try:
            reader = csv.reader(text)
            header = next(reader, [])
            if len(header) > schema.MAX_COLUMNS or len(set(header)) != len(header):
                raise ValueError("invalid bounded CSV header")
            for row in reader:
                if len(records) == schema.MAX_ROWS:
                    extra = True
                    break
                if len(row) != len(header):
                    raise ValueError("CSV row width differs from its header")
                admit(dict(zip(header, row, strict=True)))
            if not extra:
                total = len(records)
        finally:
            # The authoritative binary stream stays open for post-parse fstat.
            text.detach()
    elif fmt == "jsonl":
        while True:
            line = file.readline(8 * 1024 * 1024 + 1)
            if not line:
                total = len(records)
                break
            if len(line) > 8 * 1024 * 1024:
                raise ValueError("JSONL record exceeds inspection bounds")
            if not line.strip():
                continue
            if len(records) == schema.MAX_ROWS:
                extra = True
                break
            admit(json.loads(line))
    elif fmt == "json":
        raw = file.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("JSON document exceeds inspection bounds")
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError("JSON data must contain a record array")
        total, extra = len(rows), len(rows) > schema.MAX_ROWS
        for row in rows[: schema.MAX_ROWS]:
            admit(row)
    elif fmt == "parquet":
        import pyarrow.parquet as parquet

        source = parquet.ParquetFile(
            file,
            thrift_string_size_limit=16 * 1024 * 1024,
            thrift_container_size_limit=1000000,
        )
        names = source.schema_arrow.names
        if len(names) > schema.MAX_COLUMNS:
            raise ValueError("Parquet column bound exceeded")
        total = source.metadata.num_rows
        for batch in source.iter_batches(batch_size=8):
            if batch.nbytes > schema.MAX_DECODED_BYTES - decoded:
                raise ValueError("Parquet decoded batch exceeds inspection bounds")
            for row in batch.to_pylist():
                if len(records) == schema.MAX_ROWS:
                    extra = True
                    break
                admit(row)
            if extra:
                break
    else:
        raise ValueError("unsupported data-only inspection format")
    frame = pd.DataFrame.from_records(records)
    names = [str(col) for col in frame.columns]
    numeric: dict[str, Any] = {}
    for col in frame.select_dtypes(include=["number"]).columns:

        def finite(number: Any) -> float | None:
            number = float(number)
            return number if math.isfinite(number) else None

        numeric[str(col)] = {
            "min": finite(frame[col].min()),
            "max": finite(frame[col].max()),
            "mean": finite(frame[col].mean()),
        }
    distribution = None
    if target in frame.columns:
        values = frame[target]
        if not any(isinstance(value, (dict, list, tuple)) for value in values):
            distribution = {
                str(key)[:512]: int(count) for key, count in values.value_counts().head(10).items()
            }
    preview_state = {"characters": 0, "truncated": False}

    def preview(value: Any, depth: int = 0) -> Any:
        if depth > 4:
            preview_state["truncated"] = True
            return "<truncated>"
        if value is None or type(value) in (bool, int):
            return value
        if type(value) is float:
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            limit = min(512, max(0, 8192 - preview_state["characters"]))
            preview_state["truncated"] |= len(value) > limit
            preview_state["characters"] += min(len(value), limit)
            return value[:limit]
        if isinstance(value, dict):
            preview_state["truncated"] |= len(value) > 10
            return {str(key)[:128]: preview(item, depth + 1) for key, item in list(value.items())[:10]}
        if isinstance(value, (list, tuple)):
            preview_state["truncated"] |= len(value) > 10
            return [preview(item, depth + 1) for item in value[:10]]
        if hasattr(value, "item"):
            return preview(value.item(), depth)
        preview_state["truncated"] = True
        return str(value)[:512]

    samples = [
        {col: preview(value) for col, value in row.items()} for row in frame.head(3).to_dict(orient="records")
    ]
    return {
        "rows_scanned": len(frame),
        "rows_total": total,
        "columns": names,
        "dtypes": {str(col): str(dtype) for col, dtype in frame.dtypes.items()},
        "missing_counts": {str(col): int(count) for col, count in frame.isna().sum().items()},
        "numeric_stats": numeric,
        "target_column": target,
        "target_distribution": distribution,
        "sample_rows": samples,
        "sample_truncated": preview_state["truncated"],
        "coverage": "sampled" if extra or total != len(frame) else "complete",
    }


def _verify_output_bytes(descriptor: int, name: str, expected: bytes) -> None:
    """Bounded count/SHA readback from the original retained write FD only."""
    metadata = os.fstat(descriptor)
    named = os.stat(name, dir_fd=_OUTPUT_FD, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != len(expected)
        or not _same_identity(metadata, named)
    ):
        raise ValueError("output integrity identity or size changed")
    digest = hashlib.sha256()
    count = 0
    while chunk := os.pread(descriptor, min(65536, len(expected) - count + 1), count):
        count += len(chunk)
        if count > len(expected):
            raise ValueError("output bytes exceeded their expected count")
        digest.update(chunk)
    if count != len(expected) or digest.hexdigest() != hashlib.sha256(expected).hexdigest():
        raise ValueError("output hash or byte count changed")
    after = os.fstat(descriptor)
    named = os.stat(name, dir_fd=_OUTPUT_FD, follow_symlinks=False)
    if (
        not _same_identity(metadata, after)
        or after.st_size != len(expected)
        or not _same_identity(after, named)
    ):
        raise ValueError("output integrity changed during readback")


def run_inspection(
    *,
    input_dir: Path,
    output_dir: Path,
    expected_sha256: str,
    byte_count: int,
    format: str,
    job_name: str,
    runtime_session_id: str,
    deployment_id: str,
    deployment_prefix: str,
    image_uri: str,
    source_commit: str,
    source_version_id: str,
    staged_version_id: str,
    target_column: str | None = None,
) -> None:
    """Retain the original zero-bounding source-owned inspection path."""
    if source_commit != os.environ.get("AMLAS_SOURCE_COMMIT"):
        raise ValueError("fixed worker source binding mismatch")
    _run_inspection_with_evidence(
        evidence=_guest_preflight(),
        expected_profile="zero_bounding_v1",
        input_dir=input_dir,
        output_dir=output_dir,
        expected_sha256=expected_sha256,
        byte_count=byte_count,
        format=format,
        job_name=job_name,
        runtime_session_id=runtime_session_id,
        deployment_id=deployment_id,
        deployment_prefix=deployment_prefix,
        image_uri=image_uri,
        source_commit=source_commit,
        source_version_id=source_version_id,
        staged_version_id=staged_version_id,
        target_column=target_column,
    )


def _run_native_inspection(
    *,
    input_dir: Path,
    output_dir: Path,
    expected_sha256: str,
    byte_count: int,
    format: str,
    job_name: str,
    runtime_session_id: str,
    deployment_id: str,
    deployment_prefix: str,
    image_uri: str,
    source_commit: str,
    source_version_id: str,
    staged_version_id: str,
    target_column: str | None = None,
) -> None:
    """Fixed image path; no caller, environment or CLI profile selector."""
    if source_commit != os.environ.get("AMLAS_SOURCE_COMMIT"):
        raise ValueError("fixed worker source binding mismatch")
    _run_inspection_with_evidence(
        evidence=_native_guest_preflight(),
        expected_profile="sagemaker_native_ceiling_v1",
        input_dir=input_dir,
        output_dir=output_dir,
        expected_sha256=expected_sha256,
        byte_count=byte_count,
        format=format,
        job_name=job_name,
        runtime_session_id=runtime_session_id,
        deployment_id=deployment_id,
        deployment_prefix=deployment_prefix,
        image_uri=image_uri,
        source_commit=source_commit,
        source_version_id=source_version_id,
        staged_version_id=staged_version_id,
        target_column=target_column,
    )


def _run_inspection_with_evidence(
    *,
    evidence: dict[str, Any],
    expected_profile: str,
    input_dir: Path,
    output_dir: Path,
    expected_sha256: str,
    byte_count: int,
    format: str,
    job_name: str,
    runtime_session_id: str,
    deployment_id: str,
    deployment_prefix: str,
    image_uri: str,
    source_commit: str,
    source_version_id: str,
    staged_version_id: str,
    target_column: str | None = None,
) -> None:
    if source_commit != os.environ.get("AMLAS_SOURCE_COMMIT"):
        raise ValueError("fixed worker source binding mismatch")
    _load_schema()
    target = schema.target_text(target_column)
    if (
        format not in schema.FORMATS
        or not schema.SHA256.fullmatch(expected_sha256)
        or type(byte_count) is not int
        or not 1 <= byte_count <= schema.MAX_INPUT_BYTES
        or source_commit != os.environ.get("AMLAS_SOURCE_COMMIT")
        or not source_version_id
        or source_version_id == "null"
        or not staged_version_id
        or staged_version_id == "null"
    ):
        raise ValueError("fixed worker input binding mismatch")
    _verify_output_identity(output_dir)
    if os.listdir(_OUTPUT_FD):
        raise ValueError("inspection output must start empty")
    input_directory = _open_directory_path(input_dir)
    filename = "data." + format
    try:
        if os.listdir(input_directory) != [filename]:
            raise ValueError("inspection requires exactly one fixed regular input")
        try:
            descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=input_directory)
        except OSError:
            raise ValueError("inspection fixed input refused") from None
        with os.fdopen(descriptor, "rb") as file:
            metadata = os.fstat(file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("inspection requires one single-link regular input")
            if metadata.st_size != byte_count:
                raise ValueError("worker input byte count mismatch")
            digest = hashlib.sha256()
            count = 0
            while chunk := file.read(1024 * 1024):
                count += len(chunk)
                if count > byte_count:
                    raise ValueError("worker input exceeded byte count")
                digest.update(chunk)
            if count != byte_count or digest.hexdigest() != expected_sha256:
                raise ValueError("worker input hash mismatch")
            if not _same_identity(metadata, os.stat(filename, dir_fd=input_directory, follow_symlinks=False)):
                raise ValueError("worker input named identity changed")
            named_input = _open_directory_path(input_dir)
            try:
                if not _same_identity(os.fstat(input_directory), os.fstat(named_input)):
                    raise ValueError("worker input directory identity changed")
            finally:
                os.close(named_input)
            if _input_stat_binding(os.fstat(file.fileno())) != _input_stat_binding(metadata):
                raise ValueError("input stability changed while hashing")
            file.seek(0)
            body = _inspect_file(file, format, target)
            if _input_stat_binding(os.fstat(file.fileno())) != _input_stat_binding(metadata):
                raise ValueError("input stability changed during parsing")
            # Coarse filesystem timestamps cannot establish byte stability.
            # Re-read only this same open FD, bounded by the admitted count+1.
            file.seek(0)
            post_digest = hashlib.sha256()
            post_count = 0
            while chunk := file.read(min(1024 * 1024, byte_count - post_count + 1)):
                post_count += len(chunk)
                if post_count > byte_count:
                    raise ValueError("input post-parse hash exceeded byte count")
                post_digest.update(chunk)
            if post_count != byte_count or post_digest.hexdigest() != expected_sha256:
                raise ValueError("input changed: post-parse hash mismatch")
            if _input_stat_binding(os.fstat(file.fileno())) != _input_stat_binding(metadata):
                raise ValueError("input stability changed during post-parse hashing")
            if not _same_identity(metadata, os.stat(filename, dir_fd=input_directory, follow_symlinks=False)):
                raise ValueError("input named identity changed during parsing")
    finally:
        os.close(input_directory)
    _verify_output_identity(output_dir)
    summary = {
        "schema_version": 1,
        "format": format,
        "byte_count": byte_count,
        "sha256": expected_sha256,
        **body,
        "untrusted_data": True,
        "prepared": False,
        "trained": False,
    }
    schema.validate_summary(summary)
    raw = (json.dumps(summary, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(raw) > schema.MAX_SUMMARY_BYTES:
        raise ValueError("inspection summary exceeds fixed output bound")
    receipt = {
        "schema_version": 1,
        "job_name": job_name,
        "runtime_session_id": runtime_session_id,
        "deployment_id": deployment_id,
        "deployment_prefix": deployment_prefix,
        "image_uri": image_uri,
        "source_commit": source_commit,
        "source_version_id": source_version_id,
        "staged_version_id": staged_version_id,
        "sha256": expected_sha256,
        "byte_count": byte_count,
        "format": format,
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "guest_evidence": evidence,
        "prepared": False,
        "trained": False,
    }
    if expected_profile == schema.NATIVE_CEILING_PROFILE:
        receipt["schema_version"] = 2
        receipt["privilege_profile"] = schema.NATIVE_CEILING_PROFILE
    receipt_raw = (json.dumps(receipt, allow_nan=False) + "\n").encode()
    schema.decode_receipt(receipt_raw, expected_profile=expected_profile)
    held = []
    try:
        for name, data in (("SUMMARY.json", raw), ("INSPECTED.json", receipt_raw)):
            _verify_output_identity(output_dir)
            descriptor = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=_OUTPUT_FD,
            )
            stream = os.fdopen(descriptor, "w+b")
            held.append((stream, name, data))
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
            _verify_output_bytes(descriptor, name, data)
            _verify_output_identity(output_dir)
        # Keep both original descriptions alive across the final publication checks.
        for stream, name, data in held:
            _verify_output_bytes(stream.fileno(), name, data)
        if sorted(os.listdir(_OUTPUT_FD)) != ["INSPECTED.json", "SUMMARY.json"]:
            raise ValueError("unexpected inspection output files")
        _verify_output_identity(output_dir)
        for stream, name, data in held:
            _verify_output_bytes(stream.fileno(), name, data)
        _verify_output_identity(output_dir)
    finally:
        for stream, _, _ in held:
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "sha256",
        "byte-count",
        "format",
        "job-name",
        "runtime-session-id",
        "deployment-id",
        "deployment-prefix",
        "image-uri",
        "source-commit",
        "source-version-id",
        "staged-version-id",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--target-column")
    args = vars(parser.parse_args())
    args["expected_sha256"] = args.pop("sha256")
    args["byte_count"] = int(args["byte_count"])
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    try:
        _run_native_inspection(
            input_dir=Path("/opt/ml/processing/input"),
            output_dir=Path("/opt/ml/processing/output"),
            **args,
        )
    except Exception as error:
        # Emit only a closed privilege-stage label, never untrusted error details.
        stage = (
            ", process privilege"
            if isinstance(error, ValueError) and str(error).startswith("process privilege")
            else ""
        )
        print("fixed inspection refused (" + type(error).__name__ + stage + ")", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
