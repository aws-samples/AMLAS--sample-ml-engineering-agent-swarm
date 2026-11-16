"""Synthetic parsed records reflect primary schema, never task corpus samples."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from amlas.data_eng.open_swe_canonical import OpenSWESource, normalize_open_swe_row


def source(**changes):
    value = OpenSWESource(
        dataset_repo="nvidia/Open-SWE-Traces",
        revision="1db674173119d71df45c22007af99e214600711a",  # pragma: allowlist secret - public dataset revision
        config_name="v1.2",
        filename="data/openhands/qwen38_27b/scale-swe/train-00000-of-00006.parquet",
        artifact_uri="s3://amlas-test-acquired/data.parquet",
        artifact_version_id="version-one",
        artifact_sha256="a" * 64,
        row_index=0,
    )
    return dataclasses.replace(value, **changes)


def row(**changes):
    value = {
        "repo": "public/example",
        "instance_id": "synthetic-instance",
        "trajectory_id": "synthetic-trajectory",
        "license": "MIT",
        "resolved": 1,
        "messages": [
            {"role": "user", "content": "Inspect the synthetic function."},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Inspect before changing.",
                "tool_calls": [
                    {
                        "id": "native-call-one",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command":"read synthetic.py"}'},
                    }
                ],
            },
            {"role": "tool", "content": "synthetic output", "tool_call_id": "native-call-one"},
            {"role": "assistant", "content": "Synthetic change verified."},
        ],
        "tools": [
            json.dumps(
                {
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                }
            )
        ],
        "metadata": {
            "reference_patch": {"patch": "GOLD-SENTINEL"},
            "model_patch": {"patch": "MODEL-SENTINEL"},
        },
        "language": "python",
        "hf_dataset_name": "source-display",
    }
    value.update(changes)
    return value


def test_explicit_identity_and_causal_tools_are_preserved_without_gold():
    original = row()
    before = copy.deepcopy(original)
    result = normalize_open_swe_row(original, source=source())
    assert result["status"] == "normalized" and result["data_only"] is True
    record = result["record"]
    assert record["repository"] == original["repo"] and record["license"] == "MIT" and record["resolved"] == 1
    assert record["messages"] == original["messages"] and record["tools"][0] == json.loads(
        original["tools"][0]
    )
    assert record["source_locator"] == "s3://amlas-test-acquired/data.parquet#version=version-one#row=0"
    assert record["source_sha256"] == "a" * 64 and record["source_version_id"] == "version-one"
    assert original == before
    assert "GOLD-SENTINEL" not in json.dumps(result) and "MODEL-SENTINEL" not in json.dumps(result)
    assert result["source_provenance"]["data_license"] == "CC-BY-4.0"
    assert not {"prepared", "trained", "admitted", "grants", "policy"}.intersection(result)


@pytest.mark.parametrize("resolved", [-1, 0, 1])
def test_resolution_is_preserved_not_invented(resolved):
    result = normalize_open_swe_row(row(resolved=resolved), source=source())
    assert result["status"] == "normalized" and result["record"]["resolved"] == resolved


@pytest.mark.parametrize("resolved", [True, False, 1.0, "1", None, 2])
def test_invalid_resolution_has_bounded_reason_no_record(resolved):
    result = normalize_open_swe_row(row(resolved=resolved), source=source())
    assert result["status"] == "rejected" and result["record"] is None
    assert result["issues"] == [{"reason": "resolution_invalid", "count": 1}]


@pytest.mark.parametrize(
    "field", ["repo", "instance_id", "trajectory_id", "license", "resolved", "messages", "tools"]
)
def test_missing_required_source_field_is_not_synthesized(field):
    value = row()
    value.pop(field)
    result = normalize_open_swe_row(value, source=source())
    assert result["status"] == "rejected" and result["record"] is None


def test_missing_result_id_from_declared_schema_is_explicit_gap():
    value = row()
    value["messages"][2].pop("tool_call_id")
    result = normalize_open_swe_row(value, source=source())
    assert result["record"] is None and result["issues"] == [{"reason": "missing_tool_result_id", "count": 1}]


@pytest.mark.parametrize("mutation", ["orphan", "duplicate", "dangling"])
def test_causality_is_not_inferred_from_order_or_text(mutation):
    value = row()
    if mutation == "orphan":
        value["messages"][2]["tool_call_id"] = "other"
    if mutation == "duplicate":
        value["messages"][1]["tool_calls"].append(copy.deepcopy(value["messages"][1]["tool_calls"][0]))
    if mutation == "dangling":
        value["messages"].pop(2)
    result = normalize_open_swe_row(value, source=source())
    assert result["status"] == "rejected" and result["record"] is None


@pytest.mark.parametrize(
    "tool",
    ['{"type":"function","type":"function"}', '{"x":NaN}', "[]", '{"x":Infinity}', "[" * 65 + "0" + "]" * 65],
)
def test_tool_json_is_bounded_closed_and_duplicate_nonfinite_safe(tool):
    result = normalize_open_swe_row(row(tools=[tool]), source=source())
    assert result["status"] == "rejected" and result["record"] is None


def test_null_arrow_optional_fields_do_not_create_ids_or_change_reasoning():
    value = row()
    value["messages"][0].update(reasoning_content=None, tool_calls=None, name=None, tool_call_id=None)
    result = normalize_open_swe_row(value, source=source())
    assert result["status"] == "normalized"
    message = result["record"]["messages"][0]
    assert message["reasoning_content"] is None and message["tool_calls"] is None
    assert "name" not in message and "tool_call_id" not in message


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "b" * 40},
        {"config_name": "v1.0"},
        {"dataset_repo": "caller/fork"},
        {"artifact_version_id": "null"},
        {"artifact_sha256": "wrong"},
        {"row_index": True},
        {"filename": "../data.parquet"},
    ],
)
def test_foreign_unpinned_or_malformed_provenance_never_normalizes(change):
    result = normalize_open_swe_row(row(), source=source(**change))
    assert result["status"] == "rejected" and result["issues"] == [
        {"reason": "source_provenance_invalid", "count": 1}
    ]


def test_model_dict_is_not_source_and_row_authored_identity_cannot_override():
    assert normalize_open_swe_row(row(), source=dataclasses.asdict(source()))["record"] is None
    result = normalize_open_swe_row(row(source_locator="CALLER", source_sha256="b" * 64), source=source())
    assert result["status"] == "normalized" and result["record"]["source_sha256"] == "a" * 64


def test_repository_license_is_explicit_and_unsupported_is_a_gap():
    result = normalize_open_swe_row(row(license="Unknown-License"), source=source())
    assert result["record"] is None and result["issues"] == [
        {"reason": "repository_license_unsupported", "count": 1}
    ]
    assert "Unknown-License" not in json.dumps(result)


def test_oversized_tool_is_refused_before_embedded_json_decode(monkeypatch):
    from amlas.data_eng import open_swe_canonical as module

    calls = []
    monkeypatch.setattr(module.core, "_argument_object", lambda *_args: calls.append(True))
    result = normalize_open_swe_row(row(tools=["x" * (module.MAX_TOOL_JSON_BYTES + 1)]), source=source())
    assert result["record"] is None and result["issues"] == [
        {"reason": "tool_json_envelope_invalid", "count": 1}
    ]
    assert calls == []


def test_unknown_message_shape_and_plain_json_depth_refuse():
    value = row()
    value["messages"][0]["image_url"] = "ignored-multimodal-is-not-supported"
    assert normalize_open_swe_row(value, source=source())["record"] is None
    nested = 0
    for _ in range(70):
        nested = [nested]
    result = normalize_open_swe_row(row(metadata={"nested": nested}), source=source())
    assert result["record"] is None and result["issues"] == [{"reason": "max_json_depth", "count": 1}]
