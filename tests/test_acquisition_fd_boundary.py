"""Actual post-check path aliases/swaps must not redirect acquisition reads."""

import hashlib

import pytest

from amlas.sagemaker_entrypoints import dataset_acquisition as worker


def test_intermediate_symlink_is_rejected_even_when_it_stays_inside_root(tmp_path):
    root = tmp_path / "output"
    real = root / "real"
    real.mkdir(parents=True)
    (real / "data.csv").write_bytes(b"inside\n")
    (root / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match=r"unsafe|symbolic|symlink"):
        worker._confined_file(str(root / "alias" / "data.csv"), root)


def test_checked_descendant_swap_cannot_redirect_digest_outside(tmp_path):
    root = tmp_path / "output"
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    data = bundle / "data.csv"
    data.write_bytes(b"inside\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.csv").write_bytes(b"outside\n")
    checked = worker._confined_file(str(data), root)
    bundle.rename(root / "retained")
    bundle.symlink_to(outside, target_is_directory=True)
    try:
        assert worker._file_digest(checked) == hashlib.sha256(b"inside\n").hexdigest()
    finally:
        if hasattr(checked, "close"):
            checked.close()


def test_checked_root_swap_cannot_redirect_digest_outside(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    data = root / "data.csv"
    data.write_bytes(b"inside\n")
    checked = worker._confined_file(str(data), root)
    root.rename(tmp_path / "retained-output")
    root.mkdir()
    (root / "data.csv").write_bytes(b"outside\n")
    try:
        assert worker._file_digest(checked) == hashlib.sha256(b"inside\n").hexdigest()
    finally:
        if hasattr(checked, "close"):
            checked.close()


@pytest.mark.parametrize("victim", ["data.jsonl", "lineage.json", "COMMITTED.json"])
def test_same_inode_mutation_during_receipt_publication_refuses(tmp_path, monkeypatch, victim):
    import json

    revision = "b" * 40
    bundle = tmp_path / "acquisition-fixture"
    bundle.mkdir()
    data = bundle / "data.jsonl"
    data.write_bytes(b"inside\n")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    lineage = bundle / "lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "repo_id": "fixture/open-code",
                "revision": revision,
                "filename": "data/train.jsonl",
                "source_uri": f"https://huggingface.co/datasets/fixture/open-code/resolve/{revision}/data/train.jsonl",
                "sha256": digest,
                "byte_count": data.stat().st_size,
            }
        )
        + "\n"
    )
    commit = bundle / "COMMITTED.json"
    commit.write_text(
        json.dumps(
            {
                "data": data.name,
                "lineage": lineage.name,
                "sha256": digest,
                "lineage_sha256": hashlib.sha256(lineage.read_bytes()).hexdigest(),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        worker,
        "_acquire_local",
        lambda **kwargs: {
            "success": True,
            "local_path": str(data),
            "lineage_path": str(lineage),
            "commit_manifest_path": str(commit),
        },
    )
    original_dump = worker.json.dump
    target = bundle / victim
    before = target.stat()

    def mutate_after_dump(value, stream, *args, **kwargs):
        result = original_dump(value, stream, *args, **kwargs)
        body = target.read_bytes()
        replacement = b"INSIDE\n" if victim == "data.jsonl" else body[:-1] + b" "
        assert len(replacement) == len(body)
        target.write_bytes(replacement)
        assert target.stat().st_ino == before.st_ino
        return result

    monkeypatch.setattr(worker.json, "dump", mutate_after_dump)
    with pytest.raises(ValueError, match=r"changed|identity|integrity"):
        worker.run_acquisition(
            repo_id="fixture/open-code",
            revision=revision,
            filename="data/train.jsonl",
            output_dir=str(tmp_path),
            job_name="fixture-job",
            runtime_session_id="session-a",
            deployment_id="fixture-id",
            deployment_prefix="fixture-lab",
            image_uri="fixture-image",
            source_commit="c" * 40,
        )
    assert not (tmp_path / "ACQUIRED.json").exists()


def test_generated_receipt_same_inode_rewrite_at_fsync_refuses(tmp_path, monkeypatch):
    import json
    import os

    revision = "b" * 40
    bundle = tmp_path / "acquisition-fixture"
    bundle.mkdir()
    data = bundle / "data.jsonl"
    data.write_bytes(b"inside\n")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    lineage = bundle / "lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "repo_id": "fixture/open-code",
                "revision": revision,
                "filename": "data/train.jsonl",
                "source_uri": f"https://huggingface.co/datasets/fixture/open-code/resolve/{revision}/data/train.jsonl",
                "sha256": digest,
                "byte_count": data.stat().st_size,
            }
        )
        + "\n"
    )
    commit = bundle / "COMMITTED.json"
    commit.write_text(
        json.dumps(
            {
                "data": data.name,
                "lineage": lineage.name,
                "sha256": digest,
                "lineage_sha256": hashlib.sha256(lineage.read_bytes()).hexdigest(),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        worker,
        "_acquire_local",
        lambda **kwargs: {
            "success": True,
            "local_path": str(data),
            "lineage_path": str(lineage),
            "commit_manifest_path": str(commit),
        },
    )
    original_fsync = worker.os.fsync
    receipt = tmp_path / "ACQUIRED.json"
    rewritten = []

    def rewrite_at_fsync(fd):
        original_fsync(fd)
        if receipt.exists() and os.fstat(fd).st_ino == receipt.stat().st_ino:
            before = receipt.stat()
            body = receipt.read_bytes()
            value = json.loads(body)
            value["source_commit"] = "d" * 40
            replacement = (json.dumps(value, sort_keys=True) + "\n").encode()
            assert len(replacement) == len(body)
            receipt.write_bytes(replacement)
            assert receipt.stat().st_ino == before.st_ino
            rewritten.append(True)

    monkeypatch.setattr(worker.os, "fsync", rewrite_at_fsync)
    with pytest.raises(ValueError, match=r"receipt|changed|integrity"):
        worker.run_acquisition(
            repo_id="fixture/open-code",
            revision=revision,
            filename="data/train.jsonl",
            output_dir=str(tmp_path),
            job_name="fixture-job",
            runtime_session_id="session-a",
            deployment_id="fixture-id",
            deployment_prefix="fixture-lab",
            image_uri="fixture-image",
            source_commit="c" * 40,
        )
    assert rewritten == [True]
    assert not receipt.exists()
