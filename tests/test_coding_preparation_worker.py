"""Actual confined worker I/O with synthetic JSONL; no SDK, native job or corpus."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import pytest

from amlas.data_eng import coding_preparation as core
from amlas.sagemaker_entrypoints import coding_preparation as worker

ACCOUNT = "1" * 12
IMAGE = f"{ACCOUNT}.dkr.ecr.us-west-2.amazonaws.com/preparation@sha256:" + "a" * 64
SOURCE = "b" * 40
SESSION = "test_session"
DEPLOYMENT = "00000000-0000-4000-8000-" + "000000" + "000001"
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


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir(mode=0o700)
    (inputs / "data").mkdir()
    (inputs / "manifests").mkdir()
    records = [
        {
            "repository": f"owner/repo-{index}",
            "instance_id": f"case-{index}",
            "trajectory_id": f"trajectory-{index}",
            "license": "MIT",
            "resolved": 1,
            "source_locator": "row-authored-provenance-ignored",
            "source_version_id": "row-fake",
            "source_sha256": "0" * 64,
            "messages": [
                {"role": "user", "content": f"task {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ],
            "tools": [],
        }
        for index in range(100)
    ]
    data = b"".join(encoded(row) for row in records)
    (inputs / "data/000000.jsonl").write_bytes(data)
    lineage = {
        "sha256": digest(data),
        "byte_count": len(data),
        "format": "jsonl",
        "repo_id": "public/canonical",
        "revision": "c" * 40,
        "filename": "records.jsonl",
    }
    raw_lineage = encoded(lineage)
    commit = {"sha256": digest(data), "lineage_sha256": digest(raw_lineage)}
    raw_commit = encoded(commit)
    receipt = {
        "repo_id": "public/canonical",
        "revision": "c" * 40,
        "filename": "records.jsonl",
        "image_uri": IMAGE,
        "source_commit": SOURCE,
        "commit_manifest_sha256": digest(raw_commit),
    }
    manifests = []
    for name, raw in (("receipt", encoded(receipt)), ("commit", raw_commit), ("lineage", raw_lineage)):
        relative = "manifests/000000-" + name + ".json"
        (inputs / relative).write_bytes(raw)
        manifests.append(
            {
                "name": name,
                "relative_name": relative,
                "byte_count": len(raw),
                "sha256": digest(raw),
                "original_uri": "s3://bucket/source/" + name + ".json",
                "original_version_id": name + "-v1",
                "staged_version_id": name + "-staged",
            }
        )
    exclusions = encoded({"instance_ids": [], "patch_sha256": []})
    (inputs / "exclusions.json").write_bytes(exclusions)
    selection = {
        "schema_version": 1,
        "job_name": "amlas-test-data-prep-job",
        "runtime_session_id": SESSION,
        "deployment_id": DEPLOYMENT,
        "deployment_prefix": "amlas-test",
        "image_uri": IMAGE,
        "source_commit": SOURCE,
        "profile_id": worker.PROFILE_ID,
        "policy_version": worker.POLICY_VERSION,
        "sources": [
            {
                "data": {
                    "relative_name": "data/000000.jsonl",
                    "byte_count": len(data),
                    "sha256": digest(data),
                    "format": "jsonl",
                    "original_uri": "s3://bucket/source/records.jsonl",
                    "original_version_id": "data-v1",
                    "staged_version_id": "data-staged",
                },
                "manifest_files": manifests,
                "repo_id": "public/canonical",
                "revision": "c" * 40,
                "filename": "records.jsonl",
                "acquisition_image_uri": IMAGE,
                "acquisition_source_commit": SOURCE,
            }
        ],
        "exclusions": {
            "byte_count": len(exclusions),
            "sha256": digest(exclusions),
            "original_uri": "s3://bucket/exclusions.json",
            "original_version_id": "exclusions-v1",
            "staged_version_id": "exclusions-staged",
        },
    }
    selection_raw = encoded(selection)
    (inputs / "selection.json").write_bytes(selection_raw)
    binding = worker.WorkerBinding(
        selection_sha256=digest(selection_raw),
        selection_bytes=len(selection_raw),
        exclusions_sha256=digest(exclusions),
        exclusions_bytes=len(exclusions),
        job_name=selection["job_name"],
        runtime_session_id=SESSION,
        deployment_id=DEPLOYMENT,
        deployment_prefix="amlas-test",
        image_uri=IMAGE,
        source_commit=SOURCE,
    )
    monkeypatch.setattr(worker, "_native_identity", lambda: dict(PROOF))
    monkeypatch.setattr(worker, "_load_core", lambda: core)
    monkeypatch.setattr(worker, "_output_owner_ok", lambda _stat: True)
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", SOURCE)
    fd = os.open(outputs, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield inputs, outputs, fd, binding, selection
    finally:
        os.close(fd)


def run(fixture, binding=None):
    inputs, outputs, fd, selected, _ = fixture
    return worker.run_bound_preparation(binding or selected, inputs, outputs, fd)


def reseal(fixture):
    inputs, _, _, binding, selection = fixture
    raw = encoded(selection)
    (inputs / "selection.json").write_bytes(raw)
    return dataclasses.replace(binding, selection_sha256=digest(raw), selection_bytes=len(raw))


def test_valid_worker_commits_exact_outputs_last_and_keeps_original_fd(fixture):
    result = run(fixture)
    _, outputs, fd, _, _ = fixture
    assert set(path.name for path in outputs.iterdir()) == {*worker.OUTPUT_FILES, "COMMITTED.json"}
    assert os.fstat(fd).st_ino == outputs.stat().st_ino
    assert result["schema_version"] == 2 and result["privilege_profile"] == worker.PRIVILEGE_PROFILE
    assert result["counts"]["source_count"] == 100 and result["counts"]["retained_representatives"] == 100
    for name, fields in result["files"].items():
        raw = (outputs / name).read_bytes()
        assert len(raw) == fields["byte_count"] and digest(raw) == fields["sha256"]
    lineage = (outputs / "lineage.jsonl").read_text()
    assert "row-authored-provenance-ignored" not in lineage and "data-v1" in lineage


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_commit", "e" * 40),
        ("selection_sha256", "0" * 64),
        ("selection_bytes", 1),
        ("exclusions_sha256", "0" * 64),
        ("profile_id", "guessed-miniswe"),
    ],
)
def test_fixed_source_byte_or_profile_binding_refuses_before_outputs(fixture, field, value):
    with pytest.raises(ValueError):
        run(fixture, dataclasses.replace(fixture[3], **{field: value}))
    assert not any(fixture[1].iterdir())


def test_privilege_refusal_precedes_core_or_input_parsing(fixture, monkeypatch):
    monkeypatch.setattr(
        worker, "_native_identity", lambda: (_ for _ in ()).throw(ValueError("closed identity refusal"))
    )
    monkeypatch.setattr(worker, "_load_core", lambda: pytest.fail("core imported before identity"))
    with pytest.raises(ValueError):
        run(fixture)
    assert not any(fixture[1].iterdir())


def test_unadmitted_parquet_mapping_never_falls_back(fixture):
    fixture[4]["sources"][0]["data"]["format"] = "parquet"
    with pytest.raises(ValueError):
        run(fixture, reseal(fixture))
    assert not any(fixture[1].iterdir())


def test_input_symlink_and_prefix_escape_refuse(fixture):
    path = fixture[0] / "data/000000.jsonl"
    original = fixture[0] / "data/original.jsonl"
    path.rename(original)
    path.symlink_to(original.name)
    with pytest.raises((ValueError, OSError)):
        run(fixture)
    assert not any(fixture[1].iterdir())


def test_missing_causality_emits_no_committed_or_partial_corpus(fixture):
    inputs, _, _, _, selection = fixture
    records = [json.loads(line) for line in (inputs / "data/000000.jsonl").read_text().splitlines()]
    records[0]["messages"] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ],
        }
    ]
    records[0]["tools"] = [
        {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
    ]
    raw = b"".join(encoded(value) for value in records)
    (inputs / "data/000000.jsonl").write_bytes(raw)
    selection["sources"][0]["data"]["sha256"] = digest(raw)
    selection["sources"][0]["data"]["byte_count"] = len(raw)
    # Reseal every original lineage/commit/receipt binding so this reaches the reducer.
    source = selection["sources"][0]
    fields = {entry["name"]: entry for entry in source["manifest_files"]}
    lineage = json.loads((inputs / fields["lineage"]["relative_name"]).read_bytes())
    lineage.update(sha256=digest(raw), byte_count=len(raw))
    raw_lineage = encoded(lineage)
    commit = json.loads((inputs / fields["commit"]["relative_name"]).read_bytes())
    commit.update(sha256=digest(raw), lineage_sha256=digest(raw_lineage))
    raw_commit = encoded(commit)
    receipt = json.loads((inputs / fields["receipt"]["relative_name"]).read_bytes())
    receipt["commit_manifest_sha256"] = digest(raw_commit)
    for name, payload in (("lineage", raw_lineage), ("commit", raw_commit), ("receipt", encoded(receipt))):
        entry = fields[name]
        (inputs / entry["relative_name"]).write_bytes(payload)
        entry.update(byte_count=len(payload), sha256=digest(payload))
    with pytest.raises(ValueError, match="selection incomplete"):
        run(fixture, reseal(fixture))
    assert not any(fixture[1].iterdir())


def test_equal_length_source_tamper_fails_exact_held_bytes(fixture):
    path = fixture[0] / "data/000000.jsonl"
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"answer 0", b"ANSWER 0", 1))
    with pytest.raises(ValueError):
        run(fixture)
    assert not any(fixture[1].iterdir())


def test_original_output_root_replacement_cannot_redirect_writes(fixture):
    inputs, output, fd, binding, _ = fixture
    saved = output.with_name("original-output")
    output.rename(saved)
    output.mkdir(mode=0o700)
    with pytest.raises(ValueError):
        worker.run_bound_preparation(binding, inputs, output, fd)
    assert not any(output.iterdir()) and not any(saved.iterdir())


def test_generated_commit_mutation_at_fsync_cannot_return_success(fixture, monkeypatch):
    real = os.fsync
    output = fixture[1]

    def mutate(fd):
        real(fd)
        path = output / "COMMITTED.json"
        if path.exists():
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b'"source_count":100', b'"source_count":101', 1))

    monkeypatch.setattr(worker.os, "fsync", mutate)
    with pytest.raises(ValueError):
        run(fixture)
    assert not (output / "COMMITTED.json").exists()


def test_unexpected_inherited_descriptor_refuses_precisely(fixture, monkeypatch):
    descriptor = os.open(fixture[0] / "selection.json", os.O_RDONLY)
    try:
        monkeypatch.setattr(worker.os, "listdir", lambda _path: ["0", "1", "2", "3", str(descriptor)])
        with pytest.raises(ValueError):
            worker._inherited_descriptor_gate()
    finally:
        os.close(descriptor)


def test_worker_keeps_output_ofd_lease_after_return(fixture):
    import fcntl

    run(fixture)
    contender = os.open(fixture[1], os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(contender)


def test_source_mutation_after_reducer_is_rehashed_on_same_open_input(fixture, monkeypatch):
    class MutatingCore:
        PreparationPolicy = core.PreparationPolicy

        @staticmethod
        def prepare_canonical_conversations(records, **kwargs):
            result = core.prepare_canonical_conversations(records, **kwargs)
            path = fixture[0] / "data/000000.jsonl"
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b"answer 0", b"ANSWER 0", 1))
            return result

    monkeypatch.setattr(worker, "_load_core", lambda: MutatingCore)
    with pytest.raises(ValueError):
        run(fixture)
    assert not any(fixture[1].iterdir())


def test_direct_nonprivileged_wrong_uid_refuses_before_source_loader(monkeypatch):
    monkeypatch.setattr(worker.os, "getresuid", lambda: (0, 0, 0))
    monkeypatch.setattr(
        worker, "_root_module", lambda *_args: pytest.fail("loader reached before UID refusal")
    )
    with pytest.raises(ValueError):
        worker._native_identity()
