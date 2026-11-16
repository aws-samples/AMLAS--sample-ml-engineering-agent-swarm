"""Actual bounded Parquet worker fixtures; no cloud, model weights or readiness."""

from __future__ import annotations

import copy
import dataclasses
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_coding_preparation_worker import digest, encoded
from test_coding_preparation_worker import fixture as fixture

from amlas.data_eng import open_swe_derived
from amlas.sagemaker_entrypoints import coding_preparation as worker

PROFILE = "nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2"
POLICY = "coding-preparation.open-swe.v2"
REVISION = "1db674173119d71df45c22007af99e214600711a"  # pragma: allowlist secret - public dataset revision
FILENAME = "data/openhands/qwen38_27b/scale-swe/train-00005-of-00006.parquet"


def rows():
    result = []
    tool = {"type": "function", "function": {"name": "run", "parameters": {"type": "object"}}}
    for index in range(100):
        result.append(
            {
                "repo": f"owner/repo-{index}",
                "license": "MIT",
                "instance_id": f"case-{index}",
                "trajectory_id": f"trajectory-{index}",
                "resolved": 1 if index < 99 else 0,
                "tools": [json.dumps(tool)],
                "messages": [
                    {"role": "user", "content": f"task {index}", "tool_calls": None},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "function": {"name": "run", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "content": f"result {index}", "tool_calls": None},
                    {"role": "assistant", "content": f"answer {index}", "tool_calls": None},
                ],
                "metadata": {"reference_patch": {"patch": "GOLD-MUST-NOT-ENTER-CORPUS"}},
            }
        )
    return result


def parquet_fixture(fixture, selected=None):
    inputs, _outputs, _fd, binding, selection = fixture
    selected = rows() if selected is None else selected
    path = inputs / "data/000000.parquet"
    pq.write_table(pa.Table.from_pylist(selected), path)
    raw = path.read_bytes()
    source = selection["sources"][0]
    source.update(repo_id="nvidia/Open-SWE-Traces", revision=REVISION, filename=FILENAME)
    source["data"].update(
        relative_name="data/000000.parquet",
        format="parquet",
        original_uri="s3://bucket/source/data.parquet",
        byte_count=len(raw),
        sha256=digest(raw),
    )
    metadata = {}
    for entry in source["manifest_files"]:
        metadata[entry["name"]] = json.loads((inputs / entry["relative_name"]).read_text())
    metadata["lineage"].update(
        format="parquet",
        byte_count=len(raw),
        sha256=digest(raw),
        repo_id=source["repo_id"],
        revision=source["revision"],
        filename=source["filename"],
    )
    metadata["commit"]["sha256"] = digest(raw)
    for name in ("repo_id", "revision", "filename"):
        metadata["receipt"][name] = source[name]
    lineage_raw = encoded(metadata["lineage"])
    metadata["commit"]["lineage_sha256"] = digest(lineage_raw)
    commit_raw = encoded(metadata["commit"])
    metadata["receipt"]["commit_manifest_sha256"] = digest(commit_raw)
    for entry in source["manifest_files"]:
        body = encoded(metadata[entry["name"]])
        (inputs / entry["relative_name"]).write_bytes(body)
        entry.update(byte_count=len(body), sha256=digest(body))
    selection.update(profile_id=PROFILE, policy_version=POLICY)
    body = encoded(selection)
    (inputs / "selection.json").write_bytes(body)
    return dataclasses.replace(
        binding,
        profile_id=PROFILE,
        policy_version=POLICY,
        selection_sha256=digest(body),
        selection_bytes=len(body),
    )


def test_parquet_full_rows_reconcile_and_preserve_derived_lineage(fixture, monkeypatch):
    binding = parquet_fixture(fixture)
    monkeypatch.setattr(worker, "_load_derived_profile", lambda: open_swe_derived, raising=False)
    inputs, outputs, fd, _, _ = fixture
    result = worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert result["profile_id"] == PROFILE
    assert result["counts"]["source_count"] == 100
    assert result["counts"]["retained_representatives"] == 99
    assert result["counts"]["primary_drop_count"] == 1
    lineage = [json.loads(line) for line in (outputs / "lineage.jsonl").read_text().splitlines()]
    assert len(lineage) == 100
    accepted = [row for row in lineage if row["normalization"]["status"] == "accepted"]
    assert len(accepted) == 100
    failed = [row for row in accepted if row["decision"] == "resolution_failed"]
    assert len(failed) == 1 and failed[0]["instance_id"] == "case-99"
    assert sum(row["decision"] == "retained" for row in accepted) == 99
    assert all(
        row["normalization"]["normalization_lineage"][0]["linkage_status"]
        == "derived_not_original_source_verified"
        for row in accepted
    )
    assert all(row["source_version_id"] == "data-v1" for row in accepted)
    assert all(row["normalization"]["normalized_record_sha256"] for row in accepted)
    assert all(
        b"GOLD-MUST-NOT-ENTER-CORPUS" not in (outputs / name).read_bytes()
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
    )
    report = json.loads((outputs / "REPORT.json").read_text())
    assert report["schema_version"] == 2
    assert report["normalization_source_count"] == 100
    assert report["normalization_accepted_count"] == 100
    assert report["normalization_rejected_count"] == 0
    assert report["normalization_rejection_reasons"] == {}
    assert report["derived_edge_count"] == 100


def test_ambiguous_row_refusal_is_distinct_from_resolution_drop(fixture, monkeypatch):
    selected = rows()
    ambiguous = copy.deepcopy(selected[0])
    ambiguous.update(instance_id="ambiguous-case", trajectory_id="ambiguous-trajectory")
    other = copy.deepcopy(ambiguous["messages"][1]["tool_calls"][0])
    other["id"] = "second-call"
    ambiguous["messages"][1]["tool_calls"].append(other)
    selected.append(ambiguous)
    binding = parquet_fixture(fixture, selected)
    monkeypatch.setattr(worker, "_load_derived_profile", lambda: open_swe_derived, raising=False)
    inputs, outputs, fd, _, _ = fixture
    result = worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert result["counts"]["source_count"] == 101
    assert result["counts"]["retained_representatives"] == 99
    assert result["counts"]["primary_drop_count"] == 2
    lineage = [json.loads(line) for line in (outputs / "lineage.jsonl").read_text().splitlines()]
    assert len(lineage) == 101
    refused = [row for row in lineage if row["normalization"]["status"] == "rejected"]
    assert len(refused) == 1
    assert refused[0]["normalization"]["normalized_record_sha256"] is None
    assert refused[0]["normalization"]["normalization_lineage"] == []
    assert sum(row["decision"] == "resolution_failed" for row in lineage) == 1
    report = json.loads((outputs / "REPORT.json").read_text())
    assert report["normalization_source_count"] == 101
    assert report["normalization_accepted_count"] == 100
    assert report["normalization_rejected_count"] == 1
    assert sum(report["normalization_rejection_reasons"].values()) == 1
    assert report["derived_edge_count"] == 100


def test_parquet_size_preflight_refuses_before_batch_decode(fixture, monkeypatch):
    parquet_fixture(fixture, rows()[:1])
    inputs, _, _, _, selection = fixture
    original = pq.ParquetFile
    decoded = []

    class ObservedFile:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def iter_batches(self, *args, **kwargs):
            decoded.append(True)
            return self.inner.iter_batches(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", ObservedFile)
    policy = dataclasses.replace(open_swe_derived.core.PreparationPolicy(), max_total_bytes=1)
    with (inputs / "data/000000.parquet").open("rb") as stream:
        with pytest.raises(ValueError):
            list(worker._derived_rows(stream.fileno(), selection["sources"][0], policy, open_swe_derived))
    assert not decoded


def test_parquet_reader_disables_extensions_and_prefetch(fixture, monkeypatch):
    parquet_fixture(fixture, rows()[:1])
    inputs, _, _, _, selection = fixture
    original = pq.ParquetFile
    observed = []

    def reader(*args, **kwargs):
        observed.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", reader)
    with (inputs / "data/000000.parquet").open("rb") as stream:
        assert (
            len(
                list(
                    worker._derived_rows(
                        stream.fileno(),
                        selection["sources"][0],
                        open_swe_derived.core.PreparationPolicy(),
                        open_swe_derived,
                    )
                )
            )
            == 1
        )
    assert observed[0]["arrow_extensions_enabled"] is False
    assert observed[0]["pre_buffer"] is False
    assert observed[0]["page_checksum_verification"] is True


@pytest.mark.parametrize("mutation", ["revision", "repo", "format", "profile"])
def test_pinned_profile_refuses_mismatched_source_before_output(fixture, monkeypatch, mutation):
    binding = parquet_fixture(fixture)
    inputs, outputs, fd, _, selection = fixture
    if mutation == "revision":
        selection["sources"][0]["revision"] = "d" * 40
    elif mutation == "repo":
        selection["sources"][0]["repo_id"] = "foreign/dataset"
    elif mutation == "format":
        selection["sources"][0]["data"]["format"] = "jsonl"
    else:
        binding = dataclasses.replace(binding, profile_id="caller.profile")
    raw = encoded(selection)
    (inputs / "selection.json").write_bytes(raw)
    binding = dataclasses.replace(binding, selection_sha256=digest(raw), selection_bytes=len(raw))
    monkeypatch.setattr(worker, "_load_derived_profile", lambda: open_swe_derived, raising=False)
    with pytest.raises(ValueError):
        worker.run_bound_preparation(binding, inputs, outputs, fd)
    assert not list(outputs.iterdir())
