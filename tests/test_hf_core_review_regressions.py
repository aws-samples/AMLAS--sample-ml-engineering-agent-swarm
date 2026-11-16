"""Concrete post-review regressions; local synthetic data, no cloud/network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from test_huggingface_commit_boundary import core as core
from test_huggingface_ingestion import REVISION


def test_destination_swap_after_pending_manifest_does_not_report_success(
    core: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    moved = tmp_path / "moved"
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    write_json = core._write_json

    def write_then_swap(descriptor: int, name: str, value: dict[str, Any]) -> bytes:
        result = write_json(descriptor, name, value)
        if name == ".commit-pending.json":
            target.rename(moved)
            target.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(core, "_write_json", write_then_swap)
    result = core.acquire_huggingface_dataset_local(
        "example/coding",
        REVISION,
        "v1/train.jsonl",
        str(target),
    )
    assert result["success"] is False
    assert list(outside.iterdir()) == []
    assert list(moved.iterdir()) == [], "failed publication removes its own uncommitted bundle"


def test_scalar_target_distribution_remains_bounded_and_successful(tmp_path: Path) -> None:
    from amlas.tools.data_tools import describe_dataset_local as describe_dataset

    path = tmp_path / "labels.csv"
    pd.DataFrame({"label": ["a", "b", "a"]}).to_csv(path, index=False)
    try:
        result = describe_dataset(str(path), target_column="label")
    except AttributeError as error:
        pytest.fail("scalar target distribution is converted to dict before Series.head: " + str(error))
    assert result["success"] is True
    assert result["target_distribution"] == {"a": 2, "b": 1}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("suffix", [".jsonl", ".json"])
def test_sparse_json_records_reject_dense_expansion_before_frame_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
) -> None:
    from amlas.tools import pathref

    path = tmp_path / ("sparse" + suffix)
    records = [{f"field{i}": i} for i in range(1000)]
    path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n" if suffix == ".jsonl" else json.dumps(records)
    )
    assert path.stat().st_size < 65536
    monkeypatch.setattr(pathref, "REMOTE_MAX_BYTES", 65536)
    monkeypatch.setattr(
        pd.DataFrame,
        "from_records",
        lambda *args, **kwargs: pytest.fail("dense DataFrame allocated before materialization guard"),
    )
    with pytest.raises(pathref.PathRefError, match="materialization"):
        pathref.read_dataframe(pathref.resolve_ref(str(path), sandbox_local=False))


@pytest.mark.parametrize("suffix", [".jsonl", ".json"])
def test_decoded_json_object_budget_precedes_frame_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
) -> None:
    from amlas.tools import pathref

    path = tmp_path / ("decoded" + suffix)
    records = [{"tokens": list(range(1000))}, {"tokens": list(range(1000, 2000))}]
    path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n" if suffix == ".jsonl" else json.dumps(records)
    )
    assert path.stat().st_size < 65536
    monkeypatch.setattr(pathref, "REMOTE_MAX_BYTES", 65536)
    monkeypatch.setattr(
        pd.DataFrame,
        "from_records",
        lambda *args, **kwargs: pytest.fail("decoded object budget checked after DataFrame allocation"),
    )
    with pytest.raises(pathref.PathRefError, match="materialization"):
        pathref.read_dataframe(pathref.resolve_ref(str(path), sandbox_local=False))


@pytest.mark.parametrize("suffix", [".jsonl", ".json"])
def test_json_record_width_has_a_finite_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
) -> None:
    from amlas.tools import pathref

    path = tmp_path / ("wide" + suffix)
    row = {f"field{i}": i for i in range(1025)}
    path.write_text(json.dumps(row) + "\n" if suffix == ".jsonl" else json.dumps([row]))
    monkeypatch.setattr(
        pd.DataFrame,
        "from_records",
        lambda *args, **kwargs: pytest.fail("wide DataFrame allocated before width guard"),
    )
    with pytest.raises(pathref.PathRefError, match="columns"):
        pathref.read_dataframe(pathref.resolve_ref(str(path), sandbox_local=False))


def test_small_nested_jsonl_preserves_the_existing_frame_contract(tmp_path: Path) -> None:
    from amlas.tools import pathref

    records = [{"messages": [{"role": "assistant", "content": "fixed"}], "resolved": True}]
    path = tmp_path / "small.jsonl"
    path.write_text(json.dumps(records[0]) + "\n")
    result = pathref.read_dataframe(pathref.resolve_ref(str(path), sandbox_local=False))
    assert result["resolved"].tolist() == [True]
    assert result.iloc[0]["messages"] == records[0]["messages"]
