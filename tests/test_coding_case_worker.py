"""Actual bounded Parquet parsing, inert code strings and retained output descriptors."""

from __future__ import annotations

import dataclasses
import hashlib
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_managed_coding_case_envelope import HEADER, rows, source

from amlas.data_eng.coding_case_managed import validate_managed_case_bundle
from amlas.sagemaker_entrypoints.coding_case_freeze import run_case_freeze


@pytest.fixture
def input_case(tmp_path, monkeypatch):
    import amlas.data_eng.coding_case_profile as profile

    records = rows()
    records[0]["canonical_solution"] = "raise RuntimeError('INERT_CASE_CODE')\n"
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(records), sink)
    raw = sink.getvalue().to_pybytes()
    monkeypatch.setattr(profile, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(profile, "SOURCE_BYTES", len(raw))
    original = source()
    original = dataclasses.replace(
        original,
        inspection_source=dataclasses.replace(
            original.inspection_source,
            artifact_ref=dataclasses.replace(
                original.inspection_source.artifact_ref,
                sha256=hashlib.sha256(raw).hexdigest(),
                byte_count=len(raw),
            ),
        ),
    )
    target = tmp_path / "source.parquet"
    target.write_bytes(raw)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    return target, output, original, records


def test_actual_data_only_worker_freezes_complete_rows_without_executing_strings(input_case):
    target, output, original, _records = input_case
    fd = os.open(target, os.O_RDONLY)
    out = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        result = run_case_freeze(fd, out, source=original, header=HEADER)
    finally:
        os.close(fd)
        os.close(out)
    bundle = {path.name: path.read_bytes() for path in output.iterdir()}
    assert result["case_count"] == 164 and result["execution_status"] == "not_executed"
    assert validate_managed_case_bundle(bundle, source=original, header=HEADER)["case_count"] == 164
    assert b"INERT_CASE_CODE" not in bundle["PROMPTS.jsonl"] and b"INERT_CASE_CODE" in bundle["ORACLES.jsonl"]


def test_hash_failure_precedes_any_parquet_decoder_and_publishes_nothing(input_case, monkeypatch):
    target, output, original, _records = input_case
    raw = target.read_bytes()
    target.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    monkeypatch.setattr(pq, "ParquetFile", lambda *_a, **_kw: pytest.fail("decoder ran before raw integrity"))
    fd = os.open(target, os.O_RDONLY)
    out = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            run_case_freeze(fd, out, source=original, header=HEADER)
    finally:
        os.close(fd)
        os.close(out)
    assert list(output.iterdir()) == []


def test_existing_output_cannot_be_overwritten(input_case):
    target, output, original, _records = input_case
    marker = output / "COMMITTED.json"
    marker.write_bytes(b"original")
    fd = os.open(target, os.O_RDONLY)
    out = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            run_case_freeze(fd, out, source=original, header=HEADER)
    finally:
        os.close(fd)
        os.close(out)
    assert marker.read_bytes() == b"original"


def test_same_length_held_output_mutation_before_commit_refuses(input_case, monkeypatch):
    target, output, original, _records = input_case
    actual = os.fsync
    mutated = []

    def fsync(fd):
        actual(fd)
        if not mutated and os.fstat(fd).st_size > 0:
            first = os.pread(fd, 1, 0)
            os.pwrite(fd, bytes([first[0] ^ 1]), 0)
            mutated.append(True)

    monkeypatch.setattr(os, "fsync", fsync)
    fd = os.open(target, os.O_RDONLY)
    out = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            run_case_freeze(fd, out, source=original, header=HEADER)
    finally:
        os.close(fd)
        os.close(out)
    assert mutated == [True] and not (output / "COMMITTED.json").exists()


def test_real_main_rejects_mutable_service_input_before_parser(input_case, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    import amlas.sagemaker_entrypoints.coding_case_freeze as worker

    target, _output, original, _records = input_case
    body = b"{}\n"
    sha = hashlib.sha256(body).hexdigest()
    original = dataclasses.replace(
        original,
        manifest_refs=tuple((name, uri, version, sha) for name, uri, version, _old in original.manifest_refs),
    )
    binding = {
        "schema_version": 1,
        "source": dataclasses.asdict(original),
        "producer": HEADER,
        "manifest_files": {
            name: {"sha256": sha, "byte_count": len(body)} for name in ("receipt", "commit", "lineage")
        },
    }
    raw = (json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n").encode()
    bound = target.parent / "BINDING.json"
    bound.write_bytes(raw)
    bound.chmod(0o666)
    for name in ("receipt", "commit", "lineage"):
        (target.parent / ("source_" + name + ".json")).write_bytes(body)
    monkeypatch.setitem(
        sys.modules, "processing_identity", SimpleNamespace(verify_native_identity=lambda: None)
    )
    monkeypatch.setattr(
        worker, "_input_directory", lambda: os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    )
    calls = []
    monkeypatch.setattr(
        worker, "run_case_freeze", lambda *_a, **_kw: calls.append("parsed") or {"case_count": 164}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--binding-sha256",
            hashlib.sha256(raw).hexdigest(),
            "--binding-byte-count",
            str(len(raw)),
            "--source-commit",
            HEADER["source_commit"],
        ],
    )
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", HEADER["source_commit"])
    assert worker.main() == 78 and calls == []
