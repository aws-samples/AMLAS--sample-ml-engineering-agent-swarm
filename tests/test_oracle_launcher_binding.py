"""Neutral trusted-digest/ordering fixtures; no installed launcher or SDK executes."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from amlas.evaluation import coding_pair_runner as runner
from amlas.sagemaker_entrypoints import coding_pair as worker


def test_runner_fingerprint_binds_trusted_binary_digest_without_arm_filesystem_read(monkeypatch):
    original = Path.read_bytes
    reads = []

    def read(path):
        reads.append(str(path))
        assert str(path) != "/usr/local/bin/amlas-oracle-init"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    first = runner.paired_runner_sha256(oracle_launcher_sha256="a" * 64)
    second = runner.paired_runner_sha256(oracle_launcher_sha256="b" * 64)
    assert first != second and len(first) == len(second) == 64
    closure = {Path(name).name: hashlib.sha256(original(Path(name))).hexdigest() for name in reads[:5]}
    expected = {
        "domain": "amlas-paired-runner-v2",
        "python_closure": closure,
        "oracle_launcher_sha256": "a" * 64,
    }
    assert first == hashlib.sha256(runner._canonical(expected)).hexdigest()


@pytest.mark.parametrize("value", [None, True, "neutral", "A" * 64, "a" * 63])
def test_unconfigured_or_malformed_digest_refuses_before_closure_reads(monkeypatch, value):
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("untrusted selector reached closure"))
    with pytest.raises(ValueError):
        runner.paired_runner_sha256(oracle_launcher_sha256=value)


def test_worker_verifies_trusted_selector_before_any_input_read_or_parse(monkeypatch):
    # Model the fixed image identity independently of the test runner account.
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(worker.os, name, lambda: 10001)
    monkeypatch.setattr(worker.os, "getgroups", lambda: [10001])
    monkeypatch.setenv("AMLAS_CODING_PAIR_SOURCE_COMMIT", "b" * 40)
    monkeypatch.setenv("AMLAS_CODING_PAIR_LAUNCHER_SHA256", "c" * 64)
    monkeypatch.setattr(
        sys,
        "argv",
        ["worker", "--binding-sha256", "a" * 64, "--binding-byte-count", "1", "--source-commit", "b" * 40],
    )
    error = ValueError("neutral installed launcher mismatch")
    seen = []

    def verify(expected):
        seen.append(expected)
        raise error

    monkeypatch.setattr(worker, "verify_oracle_launcher", verify, raising=False)
    monkeypatch.setattr(
        worker, "_read", lambda *_args, **_kwargs: pytest.fail("input read preceded binary verification")
    )
    with pytest.raises(ValueError) as caught:
        worker.main()
    assert caught.value is error and seen == ["c" * 64]


def test_fixed_installed_binary_is_hashed_with_closed_owner_metadata_and_expected_digest(
    tmp_path, monkeypatch
):
    verify = getattr(runner, "verify_oracle_launcher", None)
    assert callable(verify)
    target = tmp_path / "amlas-oracle-init"
    body = b"neutral compiled launcher fixture"
    target.write_bytes(body)
    target.chmod(0o555)
    monkeypatch.setattr(runner, "ORACLE_LAUNCHER", str(target))
    original = os.fstat

    # Override only source-owned metadata observation; no permission mutation or execution.
    def owned(fd):
        actual = original(fd)
        values = {name: getattr(actual, name) for name in dir(actual) if name.startswith("st_")}
        values.update(st_uid=0, st_gid=0)
        if stat.S_ISDIR(actual.st_mode):
            # This declared closed image fixture is independent of host TMPDIR modes.
            values["st_mode"] &= ~0o022
        return SimpleNamespace(**values)

    monkeypatch.setattr(os, "fstat", owned)
    expected = hashlib.sha256(body).hexdigest()
    assert verify(expected) == expected
    with pytest.raises(ValueError):
        verify("f" * 64)


def test_closed_run_cannot_replace_trusted_startup_digest_before_binding_or_artifact_reads(monkeypatch):
    seen = []

    def verify(expected):
        seen.append(expected)
        return expected

    monkeypatch.setattr(worker, "verify_oracle_launcher", verify)
    monkeypatch.setattr(worker, "_read", lambda *_a, **_k: pytest.fail("foreign RUN reached artifact reads"))
    values = {
        name: None
        for name in (
            "session",
            "account_id",
            "region",
            "deployment_id",
            "deployment_prefix",
            "evaluation_binding",
            "spec_sha256",
            "spec_descriptor",
            "cases_source",
            "generation_sources",
            "original_refs",
            "staged_refs",
            "staged_spec_ref",
        )
    }
    values.update(
        schema_version=1, operation="paired_eval", source_commit="b" * 40, oracle_launcher_sha256="f" * 64
    )
    with pytest.raises(ValueError, match="closed paired RUN"):
        worker.load_packet(
            Path("/neutral"),
            raw_binding=json.dumps(values).encode(),
            source_commit="b" * 40,
            runtime_sha256="d" * 64,
            oracle_launcher_sha256="c" * 64,
        )
    assert seen == ["c" * 64]
