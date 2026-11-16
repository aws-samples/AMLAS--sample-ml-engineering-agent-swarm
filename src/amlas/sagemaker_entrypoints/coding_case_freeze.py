"""Fixed data-only Parquet case freeze; retained descriptors, never code execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import time

from ..data_eng.coding_case_codec import FILES
from ..data_eng.coding_case_managed import _expected, freeze_managed_case_bundle, validate_managed_case_bundle
from ..data_eng.coding_case_profile import SOURCE_FIELDS
from ..data_eng.inspection_schema import decode_object


def _hash_fd(fd, count, expected):
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != count:
        raise ValueError("case input descriptor differs")
    os.lseek(fd, 0, os.SEEK_SET)
    digest, measured = hashlib.sha256(), 0
    deadline = time.monotonic() + 60
    while True:
        if time.monotonic() >= deadline:
            raise ValueError("case input deadline exceeded")
        raw = os.read(fd, min(1024 * 1024, count - measured + 1))
        if not raw:
            break
        measured += len(raw)
        if measured > count:
            raise ValueError("case input exceeded exact bound")
        digest.update(raw)
    after = os.fstat(fd)
    if (
        (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size)
        or measured != count
        or digest.hexdigest() != expected
    ):
        raise ValueError("case input exact bytes changed")
    os.lseek(fd, 0, os.SEEK_SET)


def _output_root(fd):
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("private retained case output required")
    return info.st_dev, info.st_ino


def _published(output_fd, held):
    for name, fd, raw in held:
        _hash_fd(fd, len(raw), hashlib.sha256(raw).hexdigest())
        named = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
        current = os.fstat(fd)
        if not stat.S_ISREG(named.st_mode) or (named.st_dev, named.st_ino) != (
            current.st_dev,
            current.st_ino,
        ):
            raise ValueError("case published name changed")


def run_case_freeze(input_fd, output_fd, *, source, header):
    """Inner fixed worker; caller's native initializer supplies the privilege gate."""
    _expected(source, header)
    ref = source.inspection_source.artifact_ref
    _hash_fd(input_fd, ref.byte_count, ref.sha256)
    root = _output_root(output_fd)
    if os.listdir(output_fd):
        raise ValueError("case output must be exclusive and empty")
    # Fixed data-only library import follows the raw integrity/FD gates.
    import pyarrow as pa
    import pyarrow.parquet as pq

    with os.fdopen(os.dup(input_fd), "rb") as stream:
        parquet = pq.ParquetFile(stream)
        if (
            parquet.metadata.num_rows != 164
            or len(parquet.schema_arrow.names) != 5
            or set(parquet.schema_arrow.names) != SOURCE_FIELDS
            or any(not pa.types.is_string(field.type) for field in parquet.schema_arrow)
        ):
            raise ValueError("complete fixed case Parquet schema differs")
        records = []
        for batch in parquet.iter_batches(batch_size=164, use_threads=False):
            records.extend(batch.to_pylist())
            if len(records) > 164:
                raise ValueError("complete case count exceeded")
    _hash_fd(input_fd, ref.byte_count, ref.sha256)
    bundle = freeze_managed_case_bundle(records, source=source, header=header)
    result = validate_managed_case_bundle(bundle, source=source, header=header)
    held = []
    try:
        for name in FILES:
            _hash_fd(input_fd, ref.byte_count, ref.sha256)
            if _output_root(output_fd) != root:
                raise ValueError("case output authority changed")
            _published(output_fd, held)
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=output_fd)
            raw = bundle[name]
            held.append((name, fd, raw))
            offset = 0
            while offset < len(raw):
                written = os.write(fd, raw[offset:])
                if written <= 0:
                    raise ValueError("case output write failed")
                offset += written
            os.fsync(fd)
            _published(output_fd, held)
        _hash_fd(input_fd, ref.byte_count, ref.sha256)
        _published(output_fd, held)
        if set(os.listdir(output_fd)) != set(FILES) or _output_root(output_fd) != root:
            raise ValueError("case committed inventory changed")
        os.fsync(output_fd)
        return result
    finally:
        for _name, fd, _raw in held:
            os.close(fd)


def _input_directory():
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in ("opt", "ml", "processing", "input", "cases"):
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            info = os.fstat(fd)
            if info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o022:
                raise ValueError("case input ancestor ownership differs")
        return fd
    except Exception:
        os.close(fd)
        raise


def _input_file(root_fd, name):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or stat.S_IMODE(info.st_mode) & 0o022
        or info.st_nlink != 1
    ):
        os.close(fd)
        raise ValueError("immutable service-owned case input required")
    return fd


def _source(value):
    from ..backends.types import DatasetArtifactRef, DatasetInspectionSource, VerifiedPreparationSource

    if type(value) is not dict or set(value) != {
        "inspection_source",
        "manifest_refs",
        "repo_id",
        "revision",
        "filename",
        "acquisition_image_uri",
        "acquisition_source_commit",
    }:
        raise ValueError("closed original case source binding required")
    inspection = value["inspection_source"]
    if type(inspection) is not dict or set(inspection) != {
        "artifact_ref",
        "account_id",
        "region",
        "bucket",
        "runtime_session_id",
        "deployment_id",
        "deployment_prefix",
        "acquisition_job_arn",
    }:
        raise ValueError("closed original case inspection binding required")
    artifact = inspection["artifact_ref"]
    if (
        type(artifact) is not dict
        or set(artifact) != {"uri", "version_id", "sha256", "byte_count", "format"}
        or type(value["manifest_refs"]) is not list
    ):
        raise ValueError("closed original case artifact binding required")
    return VerifiedPreparationSource(
        DatasetInspectionSource(**{**inspection, "artifact_ref": DatasetArtifactRef(**artifact)}),
        tuple(tuple(item) for item in value["manifest_refs"]),
        value["repo_id"],
        value["revision"],
        value["filename"],
        value["acquisition_image_uri"],
        value["acquisition_source_commit"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--binding-byte-count", required=True, type=int)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    input_root = None
    held = []
    try:
        if (
            not re.fullmatch("[a-f0-9]{40}", args.source_commit)
            or os.environ.get("AMLAS_SOURCE_COMMIT") != args.source_commit
        ):
            raise ValueError("case worker image source differs")
        # Root-owned fixed helper; no profile/privilege fields from binding are used.
        from processing_identity import verify_native_identity

        verify_native_identity()
        if not 0 < args.binding_byte_count <= 65536:
            raise ValueError("case binding size differs")
        input_root = _input_directory()
        binding_fd = _input_file(input_root, "BINDING.json")
        held.append(binding_fd)
        _hash_fd(binding_fd, args.binding_byte_count, args.binding_sha256)
        binding = decode_object(os.read(binding_fd, args.binding_byte_count + 1), limit=65536)
        if (
            set(binding) != {"schema_version", "source", "producer", "manifest_files"}
            or type(binding["schema_version"]) is not int
            or binding["schema_version"] != 1
        ):
            raise ValueError("closed case worker binding required")
        source = _source(binding["source"])
        _expected(source, binding["producer"])
        if binding["producer"]["source_commit"] != args.source_commit:
            raise ValueError("case producer image source differs")
        manifests = binding["manifest_files"]
        if type(manifests) is not dict or set(manifests) != {"receipt", "commit", "lineage"}:
            raise ValueError("original case manifest inventory differs")
        expected = {name: sha for name, _uri, _version, sha in source.manifest_refs}
        for name, item in manifests.items():
            if (
                type(item) is not dict
                or set(item) != {"sha256", "byte_count"}
                or item["sha256"] != expected[name]
                or type(item["byte_count"]) is not int
                or not 0 < item["byte_count"] <= 8 * 1024**2
            ):
                raise ValueError("case original manifest descriptor differs")
            fd = _input_file(input_root, "source_" + name + ".json")
            held.append(fd)
            _hash_fd(fd, item["byte_count"], item["sha256"])
        raw_fd = _input_file(input_root, "source.parquet")
        held.append(raw_fd)
        result = run_case_freeze(raw_fd, 3, source=source, header=binding["producer"])
        for fd, name in ((binding_fd, "BINDING.json"), (raw_fd, "source.parquet")):
            named = os.stat(name, dir_fd=input_root, follow_symlinks=False)
            current = os.fstat(fd)
            if (named.st_dev, named.st_ino) != (current.st_dev, current.st_ino):
                raise ValueError("case original input name changed")
        _hash_fd(binding_fd, args.binding_byte_count, args.binding_sha256)
        print(
            json.dumps(
                {"case_count": result["case_count"], "execution_status": "not_executed"}, sort_keys=True
            )
        )
        return 0
    except Exception:
        print("CASE-FREEZE-REFUSED")
        return 78
    finally:
        for fd in held:
            os.close(fd)
        if input_root is not None:
            os.close(input_root)


if __name__ == "__main__":
    raise SystemExit(main())
