"""Independent commit-boundary controls for preexisting/raced destinations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_huggingface_ingestion import PAYLOAD, REVISION, SHA256, wire


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    from amlas.data_eng import hub_dataset
    from amlas.tools import pathref
    from amlas.tools.research import public_fetch

    monkeypatch.setattr(pathref, "_allowed_roots", lambda: [tmp_path])
    monkeypatch.delenv("AMLAS_AGENT_REMOTE_FETCH", raising=False)
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: wire())

    def streamed(url: str, sink: Any, **kwargs: Any) -> dict[str, Any]:
        sink.write(PAYLOAD)
        return {"byte_count": len(PAYLOAD), "sha256": SHA256, "git_blob_sha1": None}

    monkeypatch.setattr(public_fetch, "stream_public_binary", streamed)
    return hub_dataset


def test_preexisting_empty_bundle_is_preserved_on_name_collision(
    core: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(core.uuid, "uuid4", lambda: type("Identity", (), {"hex": "fixed"})())
    existing = tmp_path / "acquisition-fixed"
    existing.mkdir()
    result = core.acquire_huggingface_dataset_local(
        "example/coding",
        REVISION,
        "v1/train.jsonl",
        str(tmp_path),
    )
    assert result["success"] is False
    assert existing.is_dir(), "a preexisting destination is never owned for cleanup"
    assert list(existing.iterdir()) == []


def test_destination_swap_cannot_publish_a_successful_committed_path(
    core: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from amlas.tools.research import public_fetch

    target = tmp_path / "target"
    target.mkdir()
    moved = tmp_path / "moved"
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()

    def metadata(*args: Any, **kwargs: Any) -> dict[str, Any]:
        target.rename(moved)
        target.symlink_to(outside, target_is_directory=True)
        return wire()

    monkeypatch.setattr(public_fetch, "fetch_public_json", metadata)
    result = core.acquire_huggingface_dataset_local(
        "example/coding",
        REVISION,
        "v1/train.jsonl",
        str(target),
    )
    assert result["success"] is False
    assert list(outside.iterdir()) == []
    assert list(moved.iterdir()) == []


def test_wide_nested_preview_has_an_aggregate_text_budget(tmp_path: Path) -> None:
    import json

    from amlas.tools.data_tools import load_dataset_local as load_dataset

    path = tmp_path / "wide.jsonl"
    path.write_text(json.dumps({f"field{i}": "x" * 5000 for i in range(50)}) + "\n")
    result = load_dataset(str(path))
    assert result["success"] is True
    assert result["sample_truncated"] is True
    assert len(json.dumps(result, allow_nan=False)) < 20000


def test_research_scout_can_inspect_dataset_metadata_without_acquiring() -> None:
    from amlas.subagents.declarations import RESEARCH_SCOUT

    names = {getattr(tool, "tool_name", "") for tool in RESEARCH_SCOUT.tools}
    assert "inspect_huggingface_dataset" in names
    assert "acquire_huggingface_dataset" not in names


def test_describe_tool_path_contract_advertises_supported_formats() -> None:
    from amlas.tools.data_tools import describe_dataset

    description = describe_dataset.tool_spec["inputSchema"]["json"]["properties"]["path"]["description"]
    assert all(name in description for name in ("Parquet", "JSON", "JSONL"))


def test_structured_read_failure_does_not_claim_a_csv_reader(tmp_path: Path) -> None:
    from amlas.tools.data_tools import describe_dataset_local as describe_dataset

    path = tmp_path / "malformed.jsonl"
    path.write_text("not-json\n")
    result = describe_dataset(str(path))
    assert result["success"] is False
    assert result["error"].startswith("Failed to read dataset:")
