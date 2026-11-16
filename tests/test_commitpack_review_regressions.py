"""Concrete concurrent-write and retained-data budget probes; no native effects."""

from __future__ import annotations

import hashlib
import json

import pytest
from test_coding_preparation_worker import encoded
from test_commitpack_preparation_worker import commitpack_fixture
from test_commitpack_preparation_worker import fixture as fixture
from test_commitpack_profile import row, source

from amlas.data_eng import commitpack_preparation as prep
from amlas.sagemaker_entrypoints import coding_preparation as worker


def test_same_inode_hash_parse_restore_refuses_before_commit(fixture, monkeypatch):
    binding = commitpack_fixture(fixture, monkeypatch)
    inputs, outputs, fd, _, selection = fixture
    path = inputs / "data/000000.jsonl"
    original = path.read_bytes()
    records = [json.loads(line) for line in original.splitlines()]
    records[0]["new_contents"] = records[0]["new_contents"].replace("return 2", "return 3")
    replacement = b"".join(encoded(record) for record in records)
    assert len(replacement) == len(original) and replacement != original
    expected = selection["sources"][0]["data"]["sha256"]
    real_hash = worker._hash_fd
    real_decode = worker.decode_object
    state = {"swapped": False, "rows": 0}

    def hashed(descriptor, count, digest):
        real_hash(descriptor, count, digest)
        if digest == expected and not state["swapped"]:
            inode = path.stat().st_ino
            path.write_bytes(replacement)
            assert path.stat().st_ino == inode
            state["swapped"] = True

    def decoded(raw, **kwargs):
        result = real_decode(raw, **kwargs)
        if "old_contents" in result:
            state["rows"] += 1
            if state["rows"] == 100:
                path.write_bytes(original)
        return result

    monkeypatch.setattr(worker, "_hash_fd", hashed)
    monkeypatch.setattr(worker, "decode_object", decoded)
    with pytest.raises(ValueError):
        worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert not (outputs / "COMMITTED.json").exists()


def test_rejected_decoded_rows_do_not_remain_outside_expansion_budget(fixture, monkeypatch):
    values = [row(i + 1) for i in range(100)] + [{"unexpected": "x" * (2 * 1024 * 1024)}]
    binding = commitpack_fixture(fixture, monkeypatch, values)
    monkeypatch.setattr(prep, "MAX_EXPANDED_BYTES", 1024 * 1024)
    inputs, outputs, fd, _, _ = fixture
    result = worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert result.get("status") == "incomplete" and result["reason"] == "group_resource_bound"
    assert not (outputs / "COMMITTED.json").exists()


def test_builder_complete_result_is_always_within_its_replay_envelope(monkeypatch):
    # Isolate actual whole-output accounting with a reduced test-owned envelope;
    # the source profile's normalization/grouping itself still runs unchanged.
    values = [
        prep.CommitPackParsedRow(row(i + 1), source(i), hashlib.sha256(str(i).encode()).hexdigest())
        for i in range(100)
    ]
    policy = {
        "schema_version": 1,
        "repositories": [],
        "commits": [],
        "canonical_sha256": [],
        "input_sha256": [],
        "code_pair_sha256": [],
    }
    original = prep.build_commitpack_preparation(values, exclusion_policy=policy)
    raw_size = sum(len(json.dumps(entry).encode()) for entry in original["lineage"])
    monkeypatch.setattr(prep, "MAX_EXPANDED_BYTES", raw_size // 2)
    result = prep.build_commitpack_preparation(values, exclusion_policy=policy)
    if result["status"] == "complete":
        prep.replay_commitpack_preparation(
            lineage_records=result["lineage"],
            membership_records=result["membership"],
            split_records=result["splits"],
            report=result["report"],
        )
    else:
        assert result["nonusable_outcome"]["usable_split_refs"] is False


def test_complete_builder_output_never_exceeds_actual_replay_envelope():
    import dataclasses

    from amlas.data_eng import commitpack_profile as profile

    uri = "s3://abc/" + "a" * 4087
    values = [
        prep.CommitPackParsedRow(
            row(i % 100 + 1),
            dataclasses.replace(source(i), artifact_uri=uri, artifact_version_id="v" * 1024),
            "a" * 64,
        )
        for i in range(27000)
    ]
    policy = {
        "schema_version": 1,
        "repositories": [],
        "commits": [],
        "canonical_sha256": [],
        "input_sha256": [],
        "code_pair_sha256": [],
    }
    assert (
        sum(
            len(profile._encoded(profile.normalize_commitpack_row(item.row, source=item.source)))
            for item in values
        )
        < prep.MAX_EXPANDED_BYTES
    )
    result = prep.build_commitpack_preparation(values, exclusion_policy=policy)
    if result["status"] == "complete":
        prep.replay_commitpack_preparation(
            lineage_records=result["lineage"],
            membership_records=result["membership"],
            split_records=result["splits"],
            report=result["report"],
        )
    else:
        assert result["nonusable_outcome"]["usable_split_refs"] is False
