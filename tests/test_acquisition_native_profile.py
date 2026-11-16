"""Fixed native acquisition consumer/versioning; no native cloud or fetch."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from amlas.sagemaker_entrypoints import dataset_acquisition as worker

PROOF = {
    "res_uid": [10001] * 3,
    "res_gid": [10001] * 3,
    "groups": [],
    "cap_eff": 0,
    "cap_prm": 0,
    "cap_inh": 0,
    "cap_amb": 0,
    "cap_bnd": 0x200404FB,
    "no_new_privs": 1,
    "keep_caps": 0,
    "seccomp": 2,
    "fs_uid": 10001,
    "fs_gid": 10001,
    "securebits": 0,
}


def arguments(tmp_path):
    return dict(
        repo_id="fixture/open",
        revision="a" * 40,
        filename="data.csv",
        output_dir=str(tmp_path),
        job_name="fixture",
        runtime_session_id="session",
        deployment_id="deployment",
        deployment_prefix="prefix",
        image_uri="fixed-image",
        source_commit="b" * 40,
    )


def core(root):
    bundle = root / "bundle"
    bundle.mkdir()
    data = bundle / "data.csv"
    data.write_bytes(b"code\n")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    lineage = bundle / "lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "repo_id": "fixture/open",
                "revision": "a" * 40,
                "filename": "data.csv",
                "source_uri": "https://huggingface.co/datasets/fixture/open/resolve/"
                + "a" * 40
                + "/data.csv",
                "sha256": digest,
                "byte_count": 5,
            }
        )
    )
    commit = bundle / "COMMITTED.json"
    commit.write_text(
        json.dumps(
            {
                "data": "data.csv",
                "lineage": "lineage.json",
                "sha256": digest,
                "lineage_sha256": hashlib.sha256(lineage.read_bytes()).hexdigest(),
            }
        )
    )
    return dict(
        success=True,
        local_path=str(data),
        lineage_path=str(lineage),
        commit_manifest_path=str(commit),
        sha256=digest,
        byte_count=5,
    )


def test_native_schema2_records_actual_source_guard_before_core(tmp_path, monkeypatch):
    assert callable(getattr(worker, "run_acquisition_native", None)), (
        "missing fixed native acquisition entrypoint"
    )
    events = []
    monkeypatch.setattr(
        worker, "_verify_worker_native_identity", lambda: events.append("kernel-guard") or dict(PROOF)
    )
    monkeypatch.setattr(worker, "_acquire_local", lambda **kwargs: events.append("core") or core(tmp_path))
    receipt = worker.run_acquisition_native(**arguments(tmp_path))
    assert events[0] == "kernel-guard"
    assert receipt["schema_version"] == 2 and receipt["privilege_profile"] == "sagemaker_native_ceiling_v1"
    assert receipt["privilege_proof"] == PROOF
    assert json.loads((tmp_path / "ACQUIRED.json").read_text()) == receipt


def test_old_offline_strict_schema1_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "_acquire_local", lambda **kwargs: core(tmp_path))
    receipt = worker.run_acquisition(**arguments(tmp_path))
    assert (
        receipt["schema_version"] == 1
        and "privilege_profile" not in receipt
        and "privilege_proof" not in receipt
    )


def test_native_root_refuses_before_parser_or_core(monkeypatch):
    guard_path = Path(__file__).parents[1] / "processing_bootstrap/processing_identity.py"
    spec = importlib.util.spec_from_file_location("processing_identity", guard_path)
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setitem(sys.modules, "processing_identity", guard)
    assert callable(getattr(worker, "main_native", None)), "missing fixed native main"
    monkeypatch.setattr(worker.os, "getresuid", lambda: (0, 0, 0))
    monkeypatch.setattr(worker.os, "getresgid", lambda: (0, 0, 0))
    parser = Mock(side_effect=AssertionError("native parser ran as root"))
    core_call = Mock(side_effect=AssertionError("native core ran as root"))
    monkeypatch.setattr(worker.argparse, "ArgumentParser", parser)
    monkeypatch.setattr(worker, "_acquire_local", core_call)
    with pytest.raises((ValueError, RuntimeError), match="identity"):
        worker.main_native()
    parser.assert_not_called()
    core_call.assert_not_called()


def test_image_fixes_native_feature_and_source_owned_worker():
    root = Path(__file__).parents[1]
    recipe = (root / "Dockerfile.processing").read_text()
    assert "--features sagemaker-native-v1" in recipe
    assert "processing_bootstrap/acquisition_worker_native.py /app/acquisition_worker.py" in recipe
    assert "processing_bootstrap/processing_identity.py /app/processing_identity.py" in recipe
    launcher = root / "processing_bootstrap/acquisition_worker_native.py"
    assert launcher.is_file()
    text = launcher.read_text()
    assert "verify_native_identity" in text and "main_native" in text
    assert "AMLAS_PRIVILEGE_PROFILE" not in text
