"""Whole pinned Hub metadata pages through the registered SDK; no native calls."""

from __future__ import annotations

import json

import pytest
from test_huggingface_ingestion import REVISION, wire
from test_input_admission_wiring import ClientFixture, ModelFixture, configure_guard
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.data_eng import hub_dataset
from amlas.tools.huggingface_data import inspect_huggingface_dataset
from amlas.tools.research import public_fetch
from amlas.tui.chat import tool_binding_backend as binding


def wide_wire():
    info = wire()
    sample = info["siblings"][0]
    info["siblings"] = [
        {**sample, "rfilename": f"data/{i:04d}-" + "资料" * 60 + ".jsonl"} for i in range(180)
    ]
    info["cardData"]["configs"] = [
        {
            "config_name": f"group-{i:02d}",
            "data_files": [
                {
                    "split": "train",
                    "path": [f"group-{i:02d}/{j:02d}-" + "示例" * 60 + "/*.jsonl" for j in range(8)],
                }
            ],
        }
        for i in range(20)
    ]
    assert len(json.dumps(info, ensure_ascii=False).encode()) < 1024 * 1024
    return info


def sdk_projection_bytes(result):
    # Installed Strands decorator.py uses default json.dumps inside text ContentBlock.
    content = [{"text": json.dumps(result, allow_nan=False)}]
    return len(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode())


class HubModel(ModelFixture):
    async def stream(self, *args, **kwargs):
        async for event in super().stream(*args, **kwargs):
            if "contentBlockStart" in event:
                use = event["contentBlockStart"]["start"].get("toolUse")
                if use:
                    use["name"] = "inspect_huggingface_dataset"
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "toolUse" in delta:
                    delta["toolUse"]["input"] = json.dumps({"repo_id": "example/coding"})
            yield event


def test_registered_hub_sdk_result_fits_admission_and_reaches_next_model(monkeypatch):
    info = wide_wire()
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *a, **k: info)
    client = ClientFixture(deny=lambda text: False)
    configure_guard(monkeypatch, client)
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [inspect_huggingface_dataset])
    model = HubModel()
    backend = binding.ToolBindingBackend(Settings(), model=model)
    turn = backend.chat_with_tools("inspect public coding metadata")
    assert model.calls == 2, "registered Hub metadata exceeded the model input boundary"
    assert turn.input_admission["status"] != "held"
    projections = [call["content"][0]["text"]["text"] for call in client.calls]
    assert all(len(text.encode()) <= 65536 for text in projections)


def test_revision_bound_pages_cover_whole_files_and_config_paths(monkeypatch):
    info = wide_wire()
    requested = []

    def fetch(url, **kwargs):
        requested.append(url)
        return info

    monkeypatch.setattr(public_fetch, "fetch_public_json", fetch)
    first = hub_dataset.inspect_huggingface_dataset_metadata("example/coding")
    assert first["success"] is True
    assert sdk_projection_bytes(first) <= 60000, "Hub ToolResult has no aggregate byte budget"
    assert first["coverage"] == "partial"
    filenames = []
    offset = 0
    for _ in range(100):
        page = hub_dataset.inspect_huggingface_dataset_metadata(
            "example/coding", REVISION, file_offset=offset
        )
        assert page["success"] is True and sdk_projection_bytes(page) <= 60000
        assert page["revision"] == REVISION
        filenames.extend(item["filename"] for item in page["files"])
        assert all(item["sha256"] == info["siblings"][0]["lfs"]["sha256"] for item in page["files"])
        following = page["files_page"]["next_offset"]
        if following is None:
            break
        assert following > offset
        offset = following
    else:
        pytest.fail("metadata file pages did not terminate")
    assert filenames == [item["rfilename"] for item in info["siblings"]]
    paths = []
    offset = 0
    for _ in range(100):
        page = hub_dataset.inspect_huggingface_dataset_metadata(
            "example/coding", REVISION, declaration_offset=offset
        )
        assert page["success"] is True and sdk_projection_bytes(page) <= 60000
        for config in page["configs"]:
            for declaration in config["data_files"]:
                values = declaration["path"]
                values = values if isinstance(values, list) else [values]
                paths.extend((config["config_name"], declaration["split"], value) for value in values)
        following = page["declarations_page"]["next_offset"]
        if following is None:
            break
        assert following > offset
        offset = following
    else:
        pytest.fail("metadata declaration pages did not terminate")
    expected = [
        (c["config_name"], d["split"], p)
        for c in info["cardData"]["configs"]
        for d in c["data_files"]
        for p in d["path"]
    ]
    assert paths == expected
    assert requested and all("blobs=true" in url for url in requested)


@pytest.mark.parametrize("revision,offset", [(None, 1), ("main", 1), (REVISION, -1), (REVISION, True)])
def test_invalid_or_unpinned_page_never_fetches(monkeypatch, revision, offset):
    calls = []
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *a, **k: calls.append(a))
    result = hub_dataset.inspect_huggingface_dataset_metadata("example/coding", revision, file_offset=offset)
    assert result["success"] is False and calls == []


def test_changed_resolved_revision_refuses_pinned_continuation(monkeypatch):
    info = wide_wire()
    info["sha"] = "b" * 40
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *a, **k: info)
    result = hub_dataset.inspect_huggingface_dataset_metadata("example/coding", REVISION, file_offset=1)
    assert result["success"] is False


@pytest.mark.parametrize("entry", [None, {"rfilename": "data/../ordinary.jsonl"}])
def test_malformed_file_listing_cannot_claim_complete_metadata_coverage(monkeypatch, entry):
    info = wire()
    info["siblings"].append(entry)
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    result = hub_dataset.inspect_huggingface_dataset_metadata("example/coding")
    assert result["success"] is False, (
        "malformed source file metadata was silently omitted from complete coverage"
    )


def test_unbalanced_inventory_returns_executable_pinned_next_page(monkeypatch):
    info = wide_wire()
    info["cardData"]["configs"] = wire()["cardData"]["configs"]
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    first = hub_dataset.inspect_huggingface_dataset_metadata("example/coding")
    assert first["files_page"]["next_offset"] is not None
    assert first["declarations_page"]["next_offset"] is None
    continuation = first["next_page"]
    assert continuation["revision"] == REVISION
    assert continuation["file_offset"] == first["files_page"]["next_offset"]
    assert continuation["declaration_offset"] == first["declarations_page"]["total"]
    second = hub_dataset.inspect_huggingface_dataset_metadata(**continuation)
    assert second["success"] is True and second["files"]
    assert second["configs"] == [], "exhausted declarations were repeated"
    assert sdk_projection_bytes(second) <= 60000


def test_large_valid_declarations_count_without_eager_output_allocation(monkeypatch):
    import tracemalloc

    info = wire()
    info["cardData"]["configs"][0]["data_files"][0]["path"] = ["x"] * 200000
    assert len(json.dumps(info).encode()) < 1024 * 1024
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    tracemalloc.start()
    try:
        result = hub_dataset.inspect_huggingface_dataset_metadata("example/coding")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["success"] is True
    assert result["declarations_page"]["total"] == 200000
    assert 0 < result["declarations_page"]["returned"] < 200000
    assert all(c["data_files"][0]["path"] == "x" for c in result["configs"])
    assert sdk_projection_bytes(result) <= 60000
    assert peak < 32 * 1024 * 1024, f"declaration expansion allocated {peak} bytes for one bounded page"


def test_declaration_validation_reaches_malformed_tail_after_page_full(monkeypatch):
    info = wire()
    info["cardData"]["configs"][0]["data_files"][0]["path"] = ["x"] * 10000 + ["y" * 1025]
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    assert hub_dataset.inspect_huggingface_dataset_metadata("example/coding")["success"] is False
