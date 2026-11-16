"""Pure parsed-record controls; synthetic data, no corpus or managed calls."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from amlas.data_eng.coding_preparation import PreparationPolicy, prepare_canonical_conversations

POLICY = PreparationPolicy()
EXCLUSIONS = {"instance_ids": [], "patch_sha256": []}


def row(index: int, **changes):
    value = {
        "source_locator": f"public/source@fixed/shard#row-{index}",
        "source_version_id": "version-fixed",
        "source_sha256": hashlib.sha256(b"synthetic-artifact").hexdigest(),
        "repository": f"owner/repository-{index}",
        "instance_id": f"instance-{index}",
        "trajectory_id": f"trajectory-{index}",
        "license": "MIT",
        "resolved": 1,
        "messages": [
            {"role": "user", "content": f"task {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ],
        "tools": [],
    }
    value.update(changes)
    return value


def rows():
    return [row(index) for index in range(100)]


def prepare(values, policy=POLICY, exclusions=EXCLUSIONS):
    return prepare_canonical_conversations(values, policy=policy, exclusions=exclusions)


def tool_row(index=1000):
    return row(
        index,
        messages=[
            {"role": "user", "content": "repair this code"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": None,
                "tool_calls": [
                    {
                        "id": "native-call-id",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd":"echo ok"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "native-call-id", "content": "ok"},
            {"role": "assistant", "content": "fixed"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
            }
        ],
    )


def test_complete_conserves_records_and_reports_near_scope():
    result = prepare(rows())
    assert result["status"] == "complete"
    assert len(result["canonical_records"]) == 100
    assert len(result["canonical_jsonl"].splitlines()) == 100
    assert len(result["membership"]) == 100
    assert sum(result["report"]["split_counts"].values()) == 100
    assert all(result["report"]["split_counts"].values())
    assert result["report"]["conservation"] is True
    assert result["report"]["near_dedup"] == "not_implemented"
    assert "committed_manifest" not in result and "prepared" not in result


@pytest.mark.parametrize(
    "value,reason",
    [
        (True, "resolution_invalid"),
        (1.0, "resolution_invalid"),
        ("1", "resolution_invalid"),
        (None, "resolution_unknown"),
        (-1, "resolution_unknown"),
        (0, "resolution_failed"),
        (2, "resolution_invalid"),
    ],
)
def test_resolution_scalar_never_coerces(value, reason):
    result = prepare([*rows(), row(1000, resolved=value)])
    assert result["status"] == "complete"
    assert result["primary_drop_counts"][reason] == 1
    assert len(result["canonical_records"]) == 100
    assert result["report"]["conservation"] is True


@pytest.mark.parametrize("license_id", [None, "GPL-3.0", "mit", "MIT OR Apache-2.0"])
def test_fixed_license_policy_has_one_primary_drop(license_id):
    result = prepare([*rows(), row(1000, license=license_id, reference_patch="GOLD-DO-NOT-COPY")])
    assert result["primary_drop_counts"]["license_excluded"] == 1
    assert "GOLD-DO-NOT-COPY" not in json.dumps(result)


def test_assistant_data_native_ids_nullable_fields_and_gold_separation():
    source = tool_row()
    source["reference_patch"] = "GOLD-DO-NOT-COPY"
    source["model_patch"] = "MODEL-PATCH-METADATA-DO-NOT-COPY"
    before = copy.deepcopy(source)
    result = prepare([*rows(), source])
    assert result["status"] == "complete"
    assert {"messages": before["messages"], "tools": before["tools"]} in result["canonical_records"]
    assert source == before
    assert "GOLD-DO-NOT-COPY" not in json.dumps(result)
    assert "MODEL-PATCH-METADATA-DO-NOT-COPY" not in json.dumps(result)


def test_missing_result_id_blocks_selection_without_guess():
    item = tool_row()
    item["messages"][2].pop("tool_call_id")
    result = prepare([*rows(), item])
    assert result["status"] == "incomplete"
    assert result["canonical_records"] == [] and result["canonical_jsonl"] == ""
    assert any(issue["reason"] == "missing_tool_result_id" for issue in result["unresolved_profiles"])


def test_terminal_marker_does_not_admit_missing_result():
    item = tool_row()
    item["messages"] = item["messages"][:2]
    item["messages"][1]["tool_calls"][0]["function"]["arguments"] = (
        '{"cmd":"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}'
    )
    result = prepare([*rows(), item])
    assert result["status"] == "incomplete"
    assert any(issue["reason"] == "unresolved_tool_calls" for issue in result["unresolved_profiles"])
    assert result["canonical_records"] == []


def test_explicit_orphan_id_is_dropped_once():
    item = tool_row()
    item["messages"][2]["tool_call_id"] = "foreign-call"
    result = prepare([*rows(), item])
    assert result["status"] == "complete"
    assert result["primary_drop_counts"]["causality_invalid"] == 1


def test_global_exact_dedup_precedes_grouping_and_conserves_counts():
    values = rows()
    duplicate = copy.deepcopy(values[0])
    duplicate.update(
        source_locator="public/source@fixed/other#row-0",
        repository="other/repository",
        instance_id="other-instance",
        trajectory_id="other-trajectory",
    )
    result = prepare([*values, duplicate])
    assert result["primary_drop_counts"]["duplicate_exact"] == 1
    assert len(result["canonical_records"]) == 100
    assert result["report"]["source_count"] == 101
    assert result["report"]["conservation"] is True
    cluster = [
        entry
        for entry in result["lineage"]
        if entry.get("cluster_id")
        and entry["source_locator"] in (values[0]["source_locator"], duplicate["source_locator"])
    ]
    assert len(cluster) == 2 and cluster[0]["cluster_id"] == cluster[1]["cluster_id"]


def test_dedup_normalizes_ids_only_in_fingerprint_view():
    first = tool_row()
    second = copy.deepcopy(first)
    second.update(
        source_locator="public/source@fixed/other#row-1",
        repository="other/tools",
        instance_id="other-tool-instance",
        trajectory_id="other-tool-trajectory",
    )
    second["messages"][1]["tool_calls"][0]["id"] = "different-native-id"
    second["messages"][2]["tool_call_id"] = "different-native-id"
    result = prepare([*rows(), first, second])
    assert result["primary_drop_counts"]["duplicate_exact"] == 1
    retained = [value for value in result["canonical_records"] if value["tools"]]
    assert len(retained) == 1
    assert retained[0] in (
        {"messages": first["messages"], "tools": first["tools"]},
        {"messages": second["messages"], "tools": second["tools"]},
    )


def test_duplicate_bridge_and_instance_bridge_cannot_cross_splits():
    values = rows()
    duplicate = copy.deepcopy(values[0])
    duplicate.update(
        source_locator="public/source@fixed/bridge#row-0",
        repository=values[1]["repository"],
        instance_id="bridge-instance",
        trajectory_id="bridge-trajectory",
    )
    values[2]["instance_id"] = values[1]["instance_id"]
    result = prepare([*values, duplicate])
    membership = {entry["record_id"]: entry for entry in result["membership"]}
    relevant = [
        entry
        for entry in result["lineage"]
        if entry["repository"] in {values[0]["repository"], values[1]["repository"], values[2]["repository"]}
    ]
    representative_ids = {entry["representative_id"] for entry in relevant}
    assert len({membership[identity]["split"] for identity in representative_ids}) == 1
    assert len({membership[identity]["group_id"] for identity in representative_ids}) == 1


def test_shuffle_is_byte_identical_and_does_not_mutate_input():
    values = rows()
    before = copy.deepcopy(values)
    assert prepare(values) == prepare(list(reversed(values)))
    assert values == before


def test_high_confidence_secret_excludes_without_snippet_or_hash():
    token = "AK" + "IA" + "A" * 16
    item = row(
        1000, messages=[{"role": "user", "content": "instruction"}, {"role": "assistant", "content": token}]
    )
    result = prepare([*rows(), item])
    assert result["primary_drop_counts"]["secret_pattern"] == 1
    assert result["report"]["secret_pattern_counts"]["aws_access_key_id"] == 1
    assert token not in json.dumps(result)
    assert hashlib.sha256(token.encode()).hexdigest() not in json.dumps(result)


def test_harmless_password_word_and_high_entropy_code_are_preserved():
    item = row(
        1000,
        messages=[
            {"role": "user", "content": "explain password hashing"},
            {"role": "assistant", "content": "abcdef0123456789" * 20},
        ],
    )
    result = prepare([*rows(), item])
    assert result["status"] == "complete"
    assert {"messages": item["messages"], "tools": []} in result["canonical_records"]


def test_eval_instance_and_patch_exclusions_are_exclusive():
    patch = hashlib.sha256(b"synthetic patch identity").hexdigest()
    values = [*rows(), row(1000, patch_sha256=patch), row(1001)]
    result = prepare(
        values, exclusions={"instance_ids": ["instance-1000", "instance-1001"], "patch_sha256": [patch]}
    )
    assert result["primary_drop_counts"]["evaluation_overlap"] == 2
    assert result["report"]["conservation"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_records", 10),
        ("max_record_bytes", 100),
        ("max_total_bytes", 100),
        ("max_messages", 1),
        ("max_json_depth", 1),
        ("max_json_nodes", 10),
    ],
)
def test_resource_envelope_never_silently_truncates(field, value):
    result = prepare(rows(), policy=replace(POLICY, **{field: value}))
    assert result["status"] == "incomplete"
    assert result["canonical_records"] == [] and result["membership"] == []
    assert result["report"]["source_count"] == 100
    assert result["unresolved_profiles"]


def test_unsupported_shape_or_profile_is_not_a_quality_drop():
    item = row(
        1000,
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "untrusted"}}]}],
    )
    result = prepare([*rows(), item])
    assert result["status"] == "incomplete"
    assert "content_schema_invalid" not in result["primary_drop_counts"]
    result = prepare(rows(), policy=replace(POLICY, profile="unadmitted_miniswe"))
    assert result["status"] == "incomplete"
    assert any(issue["reason"] == "unsupported_profile" for issue in result["unresolved_profiles"])


def test_duplicate_locator_and_insufficient_groups_block_completeness():
    result = prepare([*rows(), copy.deepcopy(rows()[0])])
    assert result["status"] == "incomplete"
    assert any(issue["reason"] == "duplicate_source_locator" for issue in result["unresolved_profiles"])
    result = prepare([row(0)])
    assert result["status"] == "incomplete" and result["canonical_records"] == []


def test_pure_function_never_opens_files():
    with (
        patch("builtins.open", side_effect=AssertionError("unexpected filesystem effect")),
        patch("os.open", side_effect=AssertionError("unexpected filesystem effect")),
    ):
        assert prepare(rows())["status"] == "complete"


@pytest.mark.parametrize("field,value", [("seed", True), ("max_records", 0), ("max_messages", float("inf"))])
def test_invalid_policy_is_rejected_before_processing(field, value):
    with pytest.raises(ValueError):
        prepare(rows(), policy=replace(POLICY, **{field: value}))


def test_malformed_license_shape_is_one_drop_not_an_exception():
    result = prepare([*rows(), row(1000, license=["MIT"])])
    assert result["primary_drop_counts"]["license_excluded"] == 1
    assert result["status"] == "complete"


def test_malformed_role_shape_is_selection_incomplete_not_an_exception():
    item = row(1000, messages=[{"role": ["assistant"], "content": "data"}])
    assert prepare([*rows(), item])["status"] == "incomplete"


def test_malformed_function_name_shape_is_incomplete_not_an_exception():
    item = tool_row()
    item["messages"][1]["tool_calls"][0]["function"]["name"] = ["bash"]
    assert prepare([*rows(), item])["status"] == "incomplete"


def test_duplicate_embedded_argument_keys_cannot_be_silently_overwritten():
    item = tool_row()
    item["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"cmd":"first","cmd":"second"}'
    result = prepare([*rows(), item])
    assert result["status"] == "incomplete"
    assert result["canonical_records"] == []


def test_embedded_argument_depth_is_admitted_before_decoder():
    item = tool_row()
    item["messages"][1]["tool_calls"][0]["function"]["arguments"] = (
        '{"cmd":' + "[" * 65 + "0" + "]" * 65 + "}"
    )
    with patch(
        "amlas.data_eng.coding_preparation.json.loads",
        side_effect=AssertionError("decoder reached before depth gate"),
    ) as decoder:
        result = prepare([item])
    assert result["status"] == "incomplete"
    assert decoder.call_count == 0


def test_malformed_license_metadata_is_never_copied_to_lineage():
    result = prepare([*rows(), row(1000, license={"note": "UNTRUSTED-LICENSE-SNIPPET"})])
    assert result["primary_drop_counts"]["license_excluded"] == 1
    assert "UNTRUSTED-LICENSE-SNIPPET" not in json.dumps(result)


def test_secret_in_identity_metadata_has_no_snippet_or_credential_hash():
    token = "AK" + "IA" + "A" * 16
    result = prepare([*rows(), row(1000, source_locator=token)])
    assert result["primary_drop_counts"]["secret_pattern"] == 1
    assert result["report"]["secret_pattern_counts"]["aws_access_key_id"] == 1
    assert token not in json.dumps(result)
    assert hashlib.sha256(token.encode()).hexdigest() not in json.dumps(result)
