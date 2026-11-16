"""Tool descriptions and per-result caps found in runs with a real model (no network, no model).

(a) generate_spec_from_description states the two rules the model broke in 14 of 15 runs;
(b) inspect_huggingface_dataset pages stay small and read_file results are capped with explicit markers;
(c) a Hub data file whose name carries '#' (bigcode/commitpackft: data/c#, data/f#) is counted, not fatal.
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError
from test_huggingface_ingestion import REVISION, wire

from amlas.data_eng import hub_dataset
from amlas.tools import fs_tools, pathref
from amlas.tools.huggingface_data import inspect_huggingface_dataset
from amlas.tools.research import public_fetch
from amlas.tools.spec_generation_tools import _CodingPlanningInput, generate_spec_from_description

# ── (a) planning rules in the tool description and schema ──────────────────────────────────────────────


def deref(schema, node):
    while True:
        if "$ref" in node:
            node = schema["$defs"][node["$ref"].split("/")[-1]]
        elif "anyOf" in node:
            node = next(item for item in node["anyOf"] if item.get("type") != "null")
        elif "allOf" in node:
            node = node["allOf"][0]
        else:
            return node


def test_description_states_the_acceptance_string_and_catalog_id_rules():
    spec = generate_spec_from_description.tool_spec
    text = spec["description"]
    assert "primary_metric_delta_vs_baseline" in text and 'a STRING such as "> 0.0"' in text
    assert "never a bare number" in text
    assert "model_catalog" in text and "both id and version" in text
    argument = spec["inputSchema"]["json"]["properties"]["planning_data"]["description"]
    assert 'STRING such as "> 0.0"' in argument and "both id and version" in argument


def test_schema_types_the_acceptance_delta_as_string_and_requires_catalog_id_and_version():
    schema = generate_spec_from_description.tool_spec["inputSchema"]["json"]
    planning = deref(schema, schema["properties"]["planning_data"])
    catalog = deref(schema, planning["properties"]["model_catalog"])
    assert set(catalog["required"]) == {"id", "version"}
    acceptance = deref(schema, planning["properties"]["acceptance"])
    delta = acceptance["properties"]["primary_metric_delta_vs_baseline"]
    assert delta["type"] == "string" and "primary_metric_delta_vs_baseline" in acceptance["required"]


@pytest.mark.parametrize(
    "bad",
    [
        {"acceptance": {"primary_metric_delta_vs_baseline": 0.0}},
        {"acceptance": {"primary_metric_delta_vs_baseline": 1}},
        {"acceptance": {"no_subgroup_regression": True}},
        {"acceptance": {"primary_metric_delta_vs_baseline": "> 0.0", "unsupported": 1}},
        {"model_catalog": {"version": "1"}},
        {"model_catalog": {"id": "qwen3_5_9b"}},
        {"model_catalog": {"id": "qwen3_5_9b", "version": "1", "extra": "x"}},
    ],
)
def test_planning_input_rejects_the_shapes_the_model_sent(bad):
    with pytest.raises(ValidationError):
        TypeAdapter(_CodingPlanningInput).validate_python(bad)


def test_planning_input_accepts_the_documented_shape():
    good = {
        "model_catalog": {"id": "qwen3_5_9b", "version": "1"},
        "acceptance": {"primary_metric_delta_vs_baseline": "> 0.0"},
        "target": "pass_at_1",
    }
    assert TypeAdapter(_CodingPlanningInput).validate_python(good) == good


# ── (b) inspect_huggingface_dataset page size ────────────────────────────────────────────────────────


def sdk_projection_bytes(result):
    content = [{"text": json.dumps(result, allow_nan=False)}]
    return len(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode())


def big_wire(count=600):
    info = wire()
    sample = info["siblings"][0]
    info["siblings"] = [
        {**sample, "rfilename": f"data/shard-{i:04d}/train-00000.jsonl"} for i in range(count)
    ]
    return info


PAGE_CAP = 21000


def test_a_large_inventory_page_stays_under_the_page_cap_with_explicit_markers(monkeypatch):
    info = big_wire()
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    first = hub_dataset.inspect_huggingface_dataset_metadata("example/coding")
    assert first["success"] is True
    assert sdk_projection_bytes(first) <= PAGE_CAP
    assert first["projection_budget_bytes"] == PAGE_CAP == hub_dataset.MAX_INSPECTION_PROJECTION_BYTES
    assert first["coverage"] == "partial" and first["files_truncated"] is True
    assert first["files_page"]["next_offset"] == first["files_page"]["returned"] > 0
    assert first["next_page"]["revision"] == REVISION
    assert first["next_page"]["file_offset"] == first["files_page"]["next_offset"]


def test_every_page_of_a_large_inventory_is_under_the_cap_and_paging_still_covers_all_files(monkeypatch):
    info = big_wire()
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    names, sizes, page = [], [], hub_dataset.inspect_huggingface_dataset_metadata("example/coding")
    for _ in range(200):
        assert page["success"] is True
        sizes.append(sdk_projection_bytes(page))
        names.extend(item["filename"] for item in page["files"])
        if page["next_page"] is None:
            break
        page = hub_dataset.inspect_huggingface_dataset_metadata(**page["next_page"])
    else:
        pytest.fail("pages did not terminate")
    assert names == [item["rfilename"] for item in info["siblings"]]
    assert max(sizes) <= PAGE_CAP and len(sizes) > 3


def test_the_tool_description_tells_the_model_not_to_page_for_a_known_file():
    text = inspect_huggingface_dataset.tool_spec["description"]
    assert "exact file name" in text and "do not page" in text.lower()


# ── (b) read_file result cap ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(pathref, "_allowed_roots", lambda: [tmp_path])
    return tmp_path


def test_a_small_file_is_returned_whole_with_no_next_marker(sandbox):
    target = sandbox / "small.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    result = fs_tools.read_file(str(target))
    assert result["success"] is True and result["content"] == "alpha\nbeta\n"
    assert result["truncated"] is False and result["next_offset"] is None
    assert result["content_offset"] == 0 and result["content_total_bytes"] == result["content_returned_bytes"]


def test_a_large_file_is_capped_per_result_and_pages_rebuild_it_exactly(sandbox):
    text = "".join(f"line {i:05d} " + "x" * 40 + "\n" for i in range(2000))
    target = sandbox / "big.txt"
    target.write_text(text, encoding="utf-8")
    first = fs_tools.read_file(str(target))
    assert first["success"] is True and first["truncated"] is True
    assert 0 < len(first["content"].encode()) <= fs_tools.READ_RESULT_BYTES == 10_000
    assert first["next_offset"] == first["content_returned_bytes"] == len(first["content"].encode())
    assert first["content_total_bytes"] == len(text.encode()) and first["content"].endswith("\n")
    rebuilt, following, calls = first["content"], first["next_offset"], 1
    while following is not None:
        page = fs_tools.read_file(str(target), offset=following)
        assert page["content_offset"] == following
        assert len(page["content"].encode()) <= fs_tools.READ_RESULT_BYTES
        rebuilt, following, calls = rebuilt + page["content"], page["next_offset"], calls + 1
        assert calls < 100
    assert rebuilt == text and calls > 5


def test_pages_of_one_long_line_never_split_a_character(sandbox):
    text = "é" * 15000
    target = sandbox / "line.txt"
    target.write_text(text, encoding="utf-8")
    rebuilt, following, calls = "", 0, 0
    while following is not None:
        page = fs_tools.read_file(str(target), offset=following)
        assert page["success"] is True and len(page["content"].encode()) <= fs_tools.READ_RESULT_BYTES
        rebuilt, following, calls = rebuilt + page["content"], page["next_offset"], calls + 1
        assert calls < 20
    assert rebuilt == text and calls == 3


def test_head_and_tail_selections_keep_working_and_are_capped_too(sandbox):
    target = sandbox / "lines.txt"
    target.write_text("".join(f"row {i}\n" for i in range(5000)), encoding="utf-8")
    assert fs_tools.read_file(str(target), head=2)["content"] == "row 0\nrow 1\n"
    assert fs_tools.read_file(str(target), tail=1)["content"] == "row 4999\n"
    wide = fs_tools.read_file(str(target), head=4000)
    assert len(wide["content"].encode()) <= fs_tools.READ_RESULT_BYTES and wide["next_offset"] is not None


@pytest.mark.parametrize("offset", [-1, True, 1.5, "0"])
def test_invalid_offset_is_refused(sandbox, offset):
    target = sandbox / "small.txt"
    target.write_text("x\n", encoding="utf-8")
    assert fs_tools.read_file(str(target), offset=offset)["success"] is False


def test_offset_past_the_end_returns_empty_content_and_no_next(sandbox):
    target = sandbox / "small.txt"
    target.write_text("x\n", encoding="utf-8")
    result = fs_tools.read_file(str(target), offset=100)
    assert result["success"] is True and result["content"] == "" and result["next_offset"] is None


# ── (c) '#' in a Hub data file name ──────────────────────────────────────────────────────────────────


def with_siblings(*names):
    info = wire()
    info["siblings"] += [{"rfilename": name, "size": 5, "blobId": "c" * 40} for name in names]
    return info


def test_hash_in_a_data_file_name_is_counted_and_inspection_succeeds(monkeypatch):
    info = with_siblings("data/c#/data.jsonl", "data/f#/data.jsonl", "data/abap/data.jsonl")
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    result = hub_dataset.inspect_huggingface_dataset_metadata("example/coding")
    assert result["success"] is True, result
    assert [item["filename"] for item in result["files"]] == [
        "v1/train.jsonl",
        "data/abap/data.jsonl",
    ]
    assert result["files_unsupported_path_count"] == 2


def test_no_unsupported_names_reports_zero(monkeypatch):
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: wire())
    assert (
        hub_dataset.inspect_huggingface_dataset_metadata("example/coding")["files_unsupported_path_count"]
        == 0
    )


@pytest.mark.parametrize(
    "name",
    [
        "data/c#/../x.jsonl",
        "/abs#/x.jsonl",
        "data//x#.jsonl",
        "data/./x#.jsonl",
        "data/a\tb#.jsonl",
        "x" * 1100 + "#",
    ],
)
def test_structurally_malformed_names_still_fail_closed(monkeypatch, name):
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: with_siblings(name))
    assert hub_dataset.inspect_huggingface_dataset_metadata("example/coding")["success"] is False


def test_acquisition_still_refuses_a_hash_file_name():
    with pytest.raises(hub_dataset.HubDatasetError, match="invalid repository artifact path"):
        hub_dataset._filename("data/c#/data.jsonl")


def test_a_multibyte_character_cut_by_the_binary_sniff_window_is_still_text(sandbox):
    target = sandbox / "accents.txt"
    target.write_text("x" * 8191 + "é" * 100 + "\n", encoding="utf-8")
    result = fs_tools.read_file(str(target))
    assert result["success"] is True and result["content"].startswith("x" * 8191 + "é")


def test_real_binary_content_is_still_refused(sandbox):
    target = sandbox / "blob.bin"
    target.write_bytes(b"\x80\x81\x82" * 100)
    assert fs_tools.read_file(str(target))["success"] is False
    other = sandbox / "nul.bin"
    other.write_bytes(b"text\x00more")
    assert fs_tools.read_file(str(other))["success"] is False
