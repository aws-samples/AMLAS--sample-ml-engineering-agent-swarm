"""Actual retained-FD fixed worker over synthetic complete CommitPack JSONL."""

from __future__ import annotations

import dataclasses
import json

import pytest
from test_coding_preparation_worker import digest, encoded
from test_coding_preparation_worker import fixture as worker_fixture
from test_commitpack_profile import row

from amlas.data_eng import commitpack_preparation as prep
from amlas.data_eng import commitpack_profile as profile
from amlas.sagemaker_entrypoints import coding_preparation as worker


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    yield from worker_fixture.__wrapped__(tmp_path, monkeypatch)


def commitpack_fixture(fixture, monkeypatch, selected=None):
    inputs, _outputs, _fd, binding, selection = fixture
    values = [row(i + 1) for i in range(100)] if selected is None else selected
    raw = b"".join(encoded(value) for value in values)
    monkeypatch.setattr(profile, "ARTIFACT_SHA256", digest(raw))
    monkeypatch.setattr(prep, "SOURCE_BYTES", len(raw), raising=False)
    monkeypatch.setattr(worker, "_load_commitpack_profile", lambda: prep, raising=False)
    (inputs / "data/000000.jsonl").write_bytes(raw)
    source = selection["sources"][0]
    source.update(repo_id=profile.DATASET_REPO, revision=profile.REVISION, filename=profile.FILENAME)
    source["data"].update(byte_count=len(raw), sha256=digest(raw))
    metadata = {
        entry["name"]: json.loads((inputs / entry["relative_name"]).read_bytes())
        for entry in source["manifest_files"]
    }
    metadata["lineage"].update(
        sha256=digest(raw),
        byte_count=len(raw),
        repo_id=profile.DATASET_REPO,
        revision=profile.REVISION,
        filename=profile.FILENAME,
    )
    metadata["commit"]["sha256"] = digest(raw)
    metadata["receipt"].update(
        repo_id=profile.DATASET_REPO, revision=profile.REVISION, filename=profile.FILENAME
    )
    metadata["commit"]["lineage_sha256"] = digest(encoded(metadata["lineage"]))
    metadata["receipt"]["commit_manifest_sha256"] = digest(encoded(metadata["commit"]))
    for entry in source["manifest_files"]:
        body = encoded(metadata[entry["name"]])
        (inputs / entry["relative_name"]).write_bytes(body)
        entry.update(byte_count=len(body), sha256=digest(body))
    exclusions = encoded(
        {
            "schema_version": 1,
            "repositories": [],
            "commits": [],
            "canonical_sha256": [],
            "input_sha256": [],
            "code_pair_sha256": [],
        }
    )
    (inputs / "exclusions.json").write_bytes(exclusions)
    selection["exclusions"].update(byte_count=len(exclusions), sha256=digest(exclusions))
    selection.update(profile_id=prep.PROFILE_ID, policy_version=prep.POLICY_VERSION)
    body = encoded(selection)
    (inputs / "selection.json").write_bytes(body)
    return dataclasses.replace(
        binding,
        selection_sha256=digest(body),
        selection_bytes=len(body),
        exclusions_sha256=digest(exclusions),
        exclusions_bytes=len(exclusions),
        profile_id=prep.PROFILE_ID,
        policy_version=prep.POLICY_VERSION,
    )


def test_fixed_worker_dispatches_independent_profile_and_replayable_six_files(fixture, monkeypatch):
    binding = commitpack_fixture(fixture, monkeypatch)
    inputs, outputs, fd, _, _ = fixture
    committed = worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert committed["schema_version"] == 2 and committed["counts"]["source"] == 100
    assert committed["counts"]["normalization_rejected"] == 0 and committed["coverage"] == prep.COVERAGE
    assert set(p.name for p in outputs.iterdir()) == {*worker.OUTPUT_FILES, "COMMITTED.json"}
    report = json.loads((outputs / "REPORT.json").read_bytes())
    assert report["schema_version"] == 3 and report["counts"] == committed["counts"]
    lineage = [json.loads(line) for line in (outputs / "lineage.jsonl").read_bytes().splitlines()]
    membership = [json.loads(line) for line in (outputs / "membership.jsonl").read_bytes().splitlines()]
    splits = {
        name: [json.loads(line) for line in (outputs / (name + ".jsonl")).read_bytes().splitlines()]
        for name in prep.SPLITS
    }
    assert (
        prep.replay_commitpack_preparation(
            lineage_records=lineage, membership_records=membership, split_records=splits, report=report
        )["counts"]
        == report["counts"]
    )
    assert lineage[0]["raw_line_sha256"] == digest(encoded(row(1)))


def test_empty_selection_returns_bounded_nonusable_without_success_commit(fixture, monkeypatch):
    binding = commitpack_fixture(fixture, monkeypatch, [row(1, license="unknown")])
    inputs, outputs, fd, _, _ = fixture
    result = worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert result["status"] == "incomplete" and result["usable_split_refs"] is False
    assert (
        result["scan_complete"] is False and result["observed_prefix_counts"]["normalization_rejected"] == 1
    )
    assert not (outputs / "COMMITTED.json").exists()
    assert set(p.name for p in outputs.iterdir()) == {"REPORT.json"}


def test_malformed_full_row_never_becomes_partial_usable_output(fixture, monkeypatch):
    binding = commitpack_fixture(fixture, monkeypatch, [row(1), {"unexpected": "row"}])
    inputs, outputs, fd, _, _ = fixture
    result = worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert result["status"] == "incomplete" and not (outputs / "COMMITTED.json").exists()


def test_expanded_bound_stops_before_parsing_remaining_rows_without_usable_files(fixture, monkeypatch):
    binding = commitpack_fixture(fixture, monkeypatch)
    monkeypatch.setattr(prep, "MAX_EXPANDED_BYTES", 1)
    original = worker.decode_object
    rows = []

    def decode(raw, **kwargs):
        value = original(raw, **kwargs)
        if "commit" in value:
            rows.append(value)
        return value

    monkeypatch.setattr(worker, "decode_object", decode)
    inputs, outputs, fd, _, _ = fixture
    outcome = worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert len(rows) == 1 and outcome["reason"] == "group_resource_bound"
    assert outcome["scan_complete"] is False and not (outputs / "COMMITTED.json").exists()
    assert set(p.name for p in outputs.iterdir()) == {"REPORT.json"}


def test_decoding_failure_writes_prefix_only_nonusable_report(fixture, monkeypatch):
    binding = commitpack_fixture(fixture, monkeypatch)
    original = worker.decode_object
    seen = []

    def decode(raw, **kwargs):
        if b'"old_contents"' in raw:
            seen.append(raw)
            if len(seen) == 2:
                raise ValueError("untrusted JSON error text")
        return original(raw, **kwargs)

    monkeypatch.setattr(worker, "decode_object", decode)
    inputs, outputs, fd, _, _ = fixture
    outcome = worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert outcome["reason"] == "source_decode_failure" and outcome["scan_complete"] is False
    assert outcome["observed_prefix_counts"]["source"] == 1
    assert "untrusted JSON error" not in json.dumps(outcome)
    assert set(p.name for p in outputs.iterdir()) == {"REPORT.json"}


def test_source_row_bound_preserves_actual_observed_prefix_counts(fixture, monkeypatch):
    binding = commitpack_fixture(fixture, monkeypatch)
    monkeypatch.setattr(prep, "MAX_ROWS", 3)
    original = worker.decode_object
    rows = []

    def decode(raw, **kwargs):
        value = original(raw, **kwargs)
        if "commit" in value:
            rows.append(value)
        return value

    monkeypatch.setattr(worker, "decode_object", decode)
    inputs, outputs, fd, _, _ = fixture
    outcome = worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert len(rows) == 3
    assert outcome["observed_prefix_counts"] == {"source": 3, "normalized": 3, "normalization_rejected": 0}
    assert outcome["reason"] == "source_row_bound" and not (outputs / "COMMITTED.json").exists()
