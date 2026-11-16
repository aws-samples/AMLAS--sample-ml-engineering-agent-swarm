"""Complete synthetic source-shaped rows; never native samples or training corpus."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import pytest
from test_open_swe_canonical import row, source

from amlas.data_eng import coding_preparation as core
from amlas.data_eng import open_swe_derived as module
from amlas.data_eng.open_swe_canonical import normalize_open_swe_row

PROFILE = "nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2"
STATUS = "derived_not_original_source_verified"


def derived_row():
    value = row()
    value["messages"][2].pop("tool_call_id")
    return value


def normalize(value, **source_changes):
    return module.normalize_open_swe_row_derived(value, source=source(**source_changes))


def refuse(value, reason, **source_changes):
    result = normalize(value, **source_changes)
    assert result["status"] == "rejected"
    assert result["record"] is None and result["normalization_lineage"] == []
    assert result["normalized_record_sha256"] is None
    assert result["issues"] == [{"reason": reason, "count": 1}]
    return result


def edge(assistant=1, result=2, call=0, call_id="native-call-one", ordinal=0):
    return {
        "source_row_ordinal": ordinal,
        "result_message_index": result,
        "assistant_message_index": assistant,
        "call_index": call,
        "explicit_originating_call_id": call_id,
        "original_result_id_state": "absent",
        "normalization_policy_version": PROFILE,
        "linkage_status": STATUS,
    }


def test_singleton_complete_row_exact_qualified_lineage_digest_and_no_mutation():
    value = derived_row()
    original = copy.deepcopy(value)
    result = normalize(value, row_index=7)
    assert result["status"] == "normalized" and result["data_only"] is True
    record = result["record"]
    assert record["messages"][2]["tool_call_id"] == "native-call-one"
    assert result["normalization_lineage"] == [edge(ordinal=7)]
    assert record["normalization_lineage"] == result["normalization_lineage"]
    assert record["source_sha256"] == result["source_provenance"]["artifact_sha256"] == "a" * 64
    assert result["source_provenance"]["profile_id"] == PROFILE
    assert result["source_provenance"]["data_license"] == "CC-BY-4.0"
    canonical = {key: record[key] for key in ("messages", "tools")}
    expected = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert result["normalized_record_sha256"] == expected != record["source_sha256"]
    assert original == value
    rendered = json.dumps(result)
    assert "GOLD-SENTINEL" not in rendered and "MODEL-SENTINEL" not in rendered
    assert not {"ready", "admitted", "grants", "source_verified", "policy"}.intersection(record)
    assert core._canonical(record, core.PreparationPolicy()) == canonical
    # The reducer currently discards extra metadata; retaining this sidecar is
    # a Root managed-binding dependency, not proof of final preparation survival.
    assert "normalization_lineage" not in core._canonical(record, core.PreparationPolicy())


def test_deterministic_replay_and_source_dictionary_order_do_not_change_digest():
    value = derived_row()
    replay = {key: value[key] for key in reversed(value)}
    assert normalize(value) == normalize(copy.deepcopy(value)) == normalize(replay)


def test_successive_singleton_exchanges_preserve_original_indexes():
    value = derived_row()
    second_call = copy.deepcopy(value["messages"][1])
    second_call["tool_calls"][0]["id"] = "native-call-two"
    value["messages"][3:3] = [second_call, {"role": "tool", "content": "second synthetic output"}]
    result = normalize(value)
    assert result["status"] == "normalized"
    assert result["normalization_lineage"] == [edge(), edge(assistant=3, result=4, call_id="native-call-two")]


def multi_row():
    value = row()
    other = copy.deepcopy(value["messages"][1]["tool_calls"][0])
    other["id"] = "native-call-two"
    value["messages"][1]["tool_calls"].append(other)
    value["messages"].insert(
        2, {"role": "tool", "content": "second result", "tool_call_id": "native-call-two"}
    )
    return value


def test_multiple_pending_accept_only_exact_explicit_results_in_any_order():
    value = multi_row()
    value["messages"][2]["name"] = "terminal"
    result = normalize(value)
    assert result["status"] == "normalized" and result["normalization_lineage"] == []
    assert result["record"]["messages"] == value["messages"]


@pytest.mark.parametrize("with_name", [False, True])
def test_multiple_pending_do_not_use_name_or_observation_content(with_name):
    value = multi_row()
    value["messages"][2].pop("tool_call_id")
    value["messages"][2]["content"] = "native-call-two completed terminal"
    if with_name:
        value["messages"][2]["name"] = "terminal"
    refuse(value, "ambiguous_tool_result_id")


def test_multiple_originating_exchange_requires_explicit_ids_even_after_one_consumption():
    value = multi_row()
    value["messages"][3].pop("tool_call_id")
    refuse(value, "ambiguous_tool_result_id")


def test_zero_pending_never_uses_matching_name_or_result_text():
    value = derived_row()
    value["messages"][1].pop("tool_calls")
    value["messages"][1]["content"] = "Use native-call-one."
    value["messages"][2].update(name="terminal", content="native-call-one completed")
    refuse(value, "causality_invalid")


@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant"])
def test_intervening_nonresult_event_refuses_entire_row(role):
    value = derived_row()
    value["messages"].insert(2, {"role": role, "content": "Intervening event."})
    refuse(value, "unresolved_tool_calls")


def test_unknown_event_is_refused_before_any_linkage(monkeypatch):
    value = derived_row()
    value["messages"].append({"role": "unknown", "content": "Unknown event."})
    calls = []
    monkeypatch.setattr(module, "_derive_linkage", lambda *_: calls.append(True))
    refuse(value, "unsupported_message_role")
    assert calls == []


def test_later_malformed_arguments_are_refused_before_first_derivation(monkeypatch):
    value = derived_row()
    later = copy.deepcopy(value["messages"][1])
    later["tool_calls"][0]["id"] = "later"
    later["tool_calls"][0]["function"]["arguments"] = '{"x":1,"x":2}'
    value["messages"].append(later)
    calls = []
    monkeypatch.setattr(module, "_derive_linkage", lambda *_: calls.append(True))
    refuse(value, "duplicate_tool_argument_key")
    assert calls == []


def test_explicit_mismatch_never_falls_back_to_singleton_inference():
    value = row()
    value["messages"][2]["tool_call_id"] = "not-pending"
    refuse(value, "causality_invalid")


def test_explicit_null_is_not_absent():
    value = row()
    value["messages"][2]["tool_call_id"] = None
    refuse(value, "null_tool_result_id")
    assert normalize(derived_row())["status"] == "normalized"


@pytest.mark.parametrize("value", ["", 1, True, [], {}, "bad\nID"])
def test_malformed_explicit_result_id_is_not_absent(value):
    item = row()
    item["messages"][2]["tool_call_id"] = value
    refuse(item, "malformed_tool_result_id")


@pytest.mark.parametrize("name", ["different", None, 1, "bad\nname"])
def test_explicit_optional_name_must_be_a_valid_matching_function(name):
    value = derived_row()
    value["messages"][2]["name"] = name
    reason = "tool_result_name_mismatch" if name == "different" else "unsupported_message_name"
    refuse(value, reason)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_call",
        "duplicate_consumed_result",
        "orphan_explicit",
        "reused_call",
        "dangling",
        "terminal_marker",
    ],
)
def test_duplicate_orphan_dangling_and_terminal_marker_never_produce_partial_rows(mutation):
    value = derived_row()
    if mutation == "duplicate_call":
        value["messages"][1]["tool_calls"].append(copy.deepcopy(value["messages"][1]["tool_calls"][0]))
    elif mutation == "duplicate_consumed_result":
        value["messages"].insert(
            3, {"role": "tool", "content": "duplicate", "tool_call_id": "native-call-one"}
        )
    elif mutation == "orphan_explicit":
        value["messages"].append({"role": "tool", "content": "orphan", "tool_call_id": "other"})
    elif mutation == "reused_call":
        value["messages"][3:3] = [copy.deepcopy(value["messages"][1]), {"role": "tool", "content": "later"}]
    else:
        value["messages"] = value["messages"][:2]
        if mutation == "terminal_marker":
            value["messages"][1]["tool_calls"][0]["function"]["arguments"] = (
                '{"command":"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}'
            )
    reason = "unresolved_tool_calls" if mutation in {"dangling", "terminal_marker"} else "causality_invalid"
    refuse(value, reason)


@pytest.mark.parametrize(
    "arguments,reason",
    [
        ("[]", "unsupported_tool_arguments"),
        ('{"x":NaN}', "nonfinite_json"),
        ('{"x":1,"x":2}', "duplicate_tool_argument_key"),
        ("not JSON", "unsupported_tool_arguments"),
    ],
)
def test_function_arguments_are_validated_before_derivation(arguments, reason):
    value = derived_row()
    value["messages"][1]["tool_calls"][0]["function"]["arguments"] = arguments
    refuse(value, reason)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("bad_role", "unsupported_message_role"),
        ("missing_content", "unsupported_message_content"),
        ("content_object", "unsupported_message_content"),
        ("reasoning_object", "unsupported_reasoning_content"),
        ("unknown_field", "unsupported_message_shape"),
        ("bad_calls", "unsupported_tool_calls"),
        ("undeclared_name", "unsupported_tool_arguments"),
        ("missing_call_id", "missing_tool_call_id"),
    ],
)
def test_message_and_call_shapes_are_validated_in_full(mutation, reason):
    value = derived_row()
    msg = value["messages"][1]
    if mutation == "bad_role":
        msg["role"] = 1
    if mutation == "missing_content":
        msg.pop("content")
    if mutation == "content_object":
        msg["content"] = {}
    if mutation == "reasoning_object":
        msg["reasoning_content"] = {}
    if mutation == "unknown_field":
        msg["image_url"] = "unsupported"
    if mutation == "bad_calls":
        msg["tool_calls"] = "invalid"
    if mutation == "undeclared_name":
        msg["tool_calls"][0]["function"]["name"] = "unknown"
    if mutation == "missing_call_id":
        msg["tool_calls"][0].pop("id")
    refuse(value, reason)


def test_resource_envelopes_refuse_without_truncating_and_keep_no_partial_lineage(monkeypatch):
    value = derived_row()
    monkeypatch.setattr(
        module, "_POLICY", dataclasses.replace(core.PreparationPolicy(), max_messages=2), raising=False
    )
    refuse(value, "max_messages")


def test_large_embedded_arguments_refuse_before_argument_json_decoder(monkeypatch):
    value = derived_row()
    value["messages"][1]["tool_calls"][0]["function"]["arguments"] = "x" * (1024 * 1024 + 1)
    calls = []
    original = core._argument_object

    def decode(text, policy):
        if text.startswith("x"):
            calls.append(True)
        return original(text, policy)

    monkeypatch.setattr(core, "_argument_object", decode)
    refuse(value, "tool_argument_envelope_invalid")
    assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "b" * 40},
        {"dataset_repo": "caller/fork"},
        {"config_name": "other"},
        {"artifact_version_id": "null"},
        {"artifact_sha256": "wrong"},
        {"row_index": True},
        {"filename": "../data.parquet"},
    ],
)
def test_exact_typed_source_pins_versions_and_ordinal_are_required(change):
    refuse(derived_row(), "source_provenance_invalid", **change)


def test_untyped_source_and_custom_metadata_are_not_interpreted():
    result = module.normalize_open_swe_row_derived(derived_row(), source=dataclasses.asdict(source()))
    assert result["issues"] == [{"reason": "source_provenance_invalid", "count": 1}]

    class Callback:
        def __str__(self):
            pytest.fail("custom metadata callback")

    value = derived_row()
    value["metadata"] = {"custom": Callback()}
    refuse(value, "unsupported_json_shape")


@pytest.mark.parametrize("resolved", [True, False, 1.0, "1", None, 2])
def test_resolution_scalar_is_not_coerced(resolved):
    refuse(derived_row() | {"resolved": resolved}, "resolution_invalid")


@pytest.mark.parametrize("resolved", [-1, 0, 1])
def test_resolution_semantics_are_preserved_for_unchanged_preparation_filters(resolved):
    value = derived_row()
    value["resolved"] = resolved
    result = normalize(value)
    assert result["status"] == "normalized" and result["record"]["resolved"] == resolved


def test_license_provenance_and_gold_metadata_do_not_become_model_content():
    value = derived_row()
    value["license"] = "Unknown-License"
    refuse(value, "repository_license_unsupported")
    value = derived_row()
    value.update(
        source_sha256="b" * 64, source_locator="caller", ready=True, normalization_lineage=[{"forged": True}]
    )
    result = normalize(value)
    assert result["status"] == "normalized"
    assert result["record"]["source_sha256"] == "a" * 64
    assert result["normalization_lineage"] == [edge()]
    assert "ready" not in result["record"]


def test_strict_v1_is_unchanged_and_nullable_noncausal_arrow_fields_remain_data():
    value = derived_row()
    assert normalize_open_swe_row(value, source=source())["issues"] == [
        {"reason": "missing_tool_result_id", "count": 1}
    ]
    value["messages"][0].update(reasoning_content=None, tool_calls=None, name=None, tool_call_id=None)
    result = normalize(value)
    assert result["status"] == "normalized"
    message = result["record"]["messages"][0]
    assert message["reasoning_content"] is None and message["tool_calls"] is None
    assert "tool_call_id" not in message and "name" not in message


@pytest.mark.parametrize(
    "field", ["repo", "instance_id", "trajectory_id", "license", "resolved", "messages", "tools"]
)
def test_complete_source_required_fields_are_not_synthesized(field):
    value = derived_row()
    value.pop(field)
    refuse(value, "required_source_field_missing")


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("[]", "unsupported_tool_arguments"),
        ('{"x":NaN}', "nonfinite_json"),
        ('{"type":"function","type":"function"}', "duplicate_tool_argument_key"),
        (
            '{"type":"function","function":{"name":"terminal","parameters":{"$ref":"https://example.invalid"}}}',
            "external_tool_schema_ref",
        ),
    ],
)
def test_tool_schema_string_decoding_is_closed_before_linkage(raw, reason):
    value = derived_row()
    value["tools"] = [raw]
    refuse(value, reason)
