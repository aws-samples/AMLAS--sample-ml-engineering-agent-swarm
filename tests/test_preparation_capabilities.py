"""Ordinary read-only preparation discovery; no payloads or native effects."""

from __future__ import annotations

import dataclasses
import importlib
import json
from types import SimpleNamespace

import pytest
from test_coding_preparation_binding import catalog, source, store_pair

from amlas.backends.adapters._util import raw_tool
from amlas.control_plane.preparation_state import validate_source
from amlas.data_eng import commitpack_preparation as cp
from amlas.data_eng import commitpack_profile as pins

NAME = "list_coding_preparation_profiles"
OPEN_REPO = "nvidia/Open-SWE-Traces"
OPEN_REV = "1db674173119d71df45c22007af99e214600711a"  # pragma: allowlist secret - public revision
OPEN_FILE = "data/sweagent/qwen38_27b/sample/records.parquet"
CANONICAL = "openai.explicit_conversations.v1"
DERIVED = "nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2"


def discover(**kwargs):
    return importlib.import_module(
        "amlas.control_plane.preparation_capabilities"
    ).describe_preparation_capabilities(**kwargs)


def bound_tool():
    return getattr(importlib.import_module("amlas.tools.coding_preparation"), NAME)


def test_pure_export_lists_exact_existing_pairs_without_native_readiness():
    result = discover()
    assert result["query_status"] == "not_queried"
    assert {(p["profile_id"], p["policy_version"]) for p in result["profiles"]} == {
        (CANONICAL, "coding-preparation.v1"),
        (DERIVED, "coding-preparation.open-swe.v2"),
        (pins.PROFILE_ID, cp.POLICY_VERSION),
    }
    assert result["public_preparation_selector"] == {"profile_id": CANONICAL, "profile_version": "1"}
    assert result["readiness"] == "unknown" and result["data_only"] is True
    assert all(result[k] is False for k in ("native_admission", "executed", "prepared", "trained"))
    assert all(p["configuration_status"] == "unknown" for p in result["profiles"])


def test_actual_normal_and_collection_registries_expose_read_only_tool():
    from amlas.subagents.declarations import DATA_COLLECTION_AGENT, tool_name_of
    from amlas.tui.chat.agent_tools import READ_ONLY_TOOLS, STATE_CHANGING_TOOLS, get_safe_llm_tools

    assert NAME in {tool_name_of(t) for t in get_safe_llm_tools()}
    assert NAME in {tool_name_of(t) for t in DATA_COLLECTION_AGENT.tools}
    assert NAME in READ_ONLY_TOOLS and NAME not in STATE_CHANGING_TOOLS


def test_public_schema_is_only_optional_bounded_source_query():
    schema = bound_tool().tool_spec["inputSchema"]["json"]
    assert set(schema["properties"]) == {"repo_id", "revision", "filename"}
    assert schema.get("required", []) == []


@pytest.mark.parametrize(
    "repo,revision,filename,profile",
    [
        ("public/canonical", "c" * 40, "records.jsonl", CANONICAL),
        (pins.DATASET_REPO, pins.REVISION, pins.FILENAME, pins.PROFILE_ID),
        (OPEN_REPO, OPEN_REV, OPEN_FILE, DERIVED),
    ],
)
def test_metadata_shape_matches_current_source_validator_neutral_fixture(repo, revision, filename, profile):
    current = source("neutral-session", repo_id=repo, revision=revision, filename=filename)
    artifact = dataclasses.replace(
        current.inspection_source.artifact_ref,
        format="parquet" if filename.endswith(".parquet") else "jsonl",
        sha256=pins.ARTIFACT_SHA256 if repo == pins.DATASET_REPO else "e" * 64,
        byte_count=cp.SOURCE_BYTES if repo == pins.DATASET_REPO else 20,
    )
    current = dataclasses.replace(
        current, inspection_source=dataclasses.replace(current.inspection_source, artifact_ref=artifact)
    )
    validate_source(current, catalog(), "neutral-session")
    result = discover(repo_id=repo, revision=revision, filename=filename)
    assert result["query_status"] == "source_shape_supported"
    assert result["matched_profile_id"] == profile
    assert result["native_admission"] is False


@pytest.mark.parametrize(
    "repo,revision,filename",
    [
        (OPEN_REPO, "c" * 40, OPEN_FILE),
        (OPEN_REPO, OPEN_REV, "data/other/qwen38_27b/sample/records.parquet"),
        (OPEN_REPO, OPEN_REV, "data/sweagent/unknown/sample/records.parquet"),
        (pins.DATASET_REPO, "c" * 40, pins.FILENAME),
        (pins.DATASET_REPO, pins.REVISION, "data/python/other.jsonl"),
        ("public/other", "c" * 40, "records.parquet"),
    ],
)
def test_known_pin_mismatch_and_generic_parquet_are_unsupported(repo, revision, filename):
    current = source("neutral-session", repo_id=repo, revision=revision, filename=filename)
    artifact = dataclasses.replace(
        current.inspection_source.artifact_ref,
        format="parquet" if filename.endswith(".parquet") else "jsonl",
    )
    current = dataclasses.replace(
        current, inspection_source=dataclasses.replace(current.inspection_source, artifact_ref=artifact)
    )
    with pytest.raises(ValueError):
        validate_source(current, catalog(), "neutral-session")
    result = discover(repo_id=repo, revision=revision, filename=filename)
    assert result["query_status"] == "unsupported" and result["matched_profile_id"] is None


def test_swe_smith_reports_missing_profile_and_requirements_without_substitution():
    result = discover(
        repo_id="SWE-bench/SWE-smith-trajectories",
        revision="08e109b4a59eaeebf80e4675cd125d42e7ac99a4",  # pragma: allowlist secret - public revision
        filename="tool/train-00004-of-00008.parquet",
    )
    assert result["query_status"] == "unsupported"
    assert result["reason"] == "source_profile_missing"
    assert result["missing_requirements"] == [
        "trajectory_format_profile",
        "original_task_repository_revision_license_provenance",
        "bounded_complete_cohort_accounting",
    ]
    assert "recommended_dataset" not in result


@pytest.mark.parametrize(
    "query", [{"repo_id": "public/data"}, {"revision": "a" * 40}, {"filename": "data.jsonl"}]
)
def test_partial_queries_do_not_claim_compatibility(query):
    assert discover(**query)["query_status"] == "incomplete"


@pytest.mark.parametrize(
    "query",
    [
        {"repo_id": ""},
        {"repo_id": True},
        {"repo_id": "invalid/name/extra"},
        {"repo_id": "public/" + "a" * 201},
        {"revision": "MAIN"},
        {"revision": "A" * 40},
        {"filename": "../records.jsonl"},
        {"filename": "/records.jsonl"},
        {"filename": "data//records.jsonl"},
        {"filename": "data/./records.jsonl"},
        {"filename": "a" * 1025},
        {"filename": "data\nrecords.jsonl"},
    ],
)
def test_invalid_queries_are_bounded_data_refusal_without_reflecting_values(query):
    result = discover(**query)
    assert result["query_status"] == "invalid_input"
    assert result["matched_profile_id"] is None
    assert "query" not in result


def test_configuration_uses_only_exact_held_catalog_and_pure_validator():
    selected = catalog()
    result = discover(catalog=selected)
    assert all(p["configuration_status"] == "configured" for p in result["profiles"])
    assert result["readiness"] == "unknown" and result["native_admission"] is False
    encoded = json.dumps(result)
    for private in (selected.image_uri, selected.execution_role_arn, selected.account_id, selected.bucket):
        assert private not in encoded
    missing = discover(catalog=dataclasses.replace(selected, image_uri=""))
    assert all(p["configuration_status"] == "unconfigured" for p in missing["profiles"])
    assert all(p["missing_selectors"] == ["AMLAS_DATA_PREPARATION_IMAGE_URI"] for p in missing["profiles"])
    malformed = discover(catalog=dataclasses.replace(selected, image_uri="changed"))
    assert all(p["configuration_status"] == "unknown" for p in malformed["profiles"])


def test_per_profile_exclusion_limit_is_existing_pure_configuration_bound():
    selected = catalog()
    selected = dataclasses.replace(
        selected, exclusions_ref=dataclasses.replace(selected.exclusions_ref, byte_count=3 * 1024 * 1024)
    )
    profiles = {p["profile_id"]: p for p in discover(catalog=selected)["profiles"]}
    assert profiles[pins.PROFILE_ID]["configuration_status"] == "configured"
    assert profiles[CANONICAL]["configuration_status"] == "unknown"
    assert profiles[DERIVED]["configuration_status"] == "unknown"


def test_untyped_or_supplier_catalog_is_never_called():
    class NotCatalog:
        def __getattribute__(self, name):
            raise AssertionError("untyped configuration must not be inspected")

    assert all(p["configuration_status"] == "unknown" for p in discover(catalog=NotCatalog())["profiles"])


def test_actual_direct_ordinary_readonly_call_uses_owned_state_without_effect(monkeypatch):
    from amlas.config.settings import Settings
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    backend = ToolBindingBackend(Settings())
    acquired, effects, held = store_pair()
    backend._preparation_store = held
    result = backend.execute_tool_use(NAME, {})
    assert result.status == "executed", result
    assert all(p["configuration_status"] == "configured" for p in result.result["profiles"])
    assert acquired.calls == effects.calls == effects.observations == 0
    denied = backend.execute_tool_use(NAME, {"tool_context": SimpleNamespace(invocation_state={})})
    assert denied.status == "error"
    assert acquired.calls == effects.calls == effects.observations == 0


def test_actual_sdk_ordinary_tool_loop_transports_only_hidden_original_store():
    from strands import Agent
    from strands.hooks.events import AfterToolCallEvent
    from test_sdk_prompt_policy import ScriptedModel

    from amlas.tui.chat.agent_tools import get_safe_llm_tools

    acquired, effects, held = store_pair()
    results = []
    agent = Agent(
        model=ScriptedModel([[{"id": "profiles", "name": NAME, "input": {}}]]),
        tools=get_safe_llm_tools(),
        callback_handler=lambda *_a, **_kw: None,
    )

    def collect(event):
        if event.tool_use["name"] != NAME:
            return
        for block in event.result["content"]:
            if "json" in block:
                results.append(block["json"])
            elif "text" in block:
                try:
                    results.append(json.loads(block["text"]))
                except json.JSONDecodeError:
                    pass

    agent.hooks.add_callback(AfterToolCallEvent, collect)
    agent("Inspect preparation capabilities", invocation_state={"_amlas_preparation_store": held})
    assert results and all(p["configuration_status"] == "configured" for p in results[0]["profiles"])
    assert acquired.calls == effects.calls == effects.observations == 0


def test_descriptor_calls_are_fresh_json_data_without_mutating_source():
    result = discover()
    result["profiles"].clear()
    assert len(discover()["profiles"]) == 3
    json.dumps(discover(), allow_nan=False)
    assert raw_tool(bound_tool())()["readiness"] == "unknown"


@pytest.mark.parametrize(
    "repo,filename",
    [
        ("public/data", "data/x?y.jsonl"),
        ("public/data", "data/x%y.jsonl"),
        ("public/data", "data/x#y.jsonl"),
        ("a" * 97 + "/data", "data/records.jsonl"),
    ],
)
def test_before_acquisition_query_reuses_exact_pure_hub_refusals(repo, filename):
    from amlas.data_eng import hub_dataset

    with pytest.raises(hub_dataset.HubDatasetError):
        hub_dataset._repository(repo)
        hub_dataset._filename(filename)
    result = discover(repo_id=repo, revision="c" * 40, filename=filename)
    assert result["query_status"] == "invalid_input"
    assert result["matched_profile_id"] is None


@pytest.mark.parametrize("repo", ["public", "a" * 96 + "/" + "b" * 96])
def test_before_acquisition_query_preserves_hub_supported_repository_shapes(repo):
    from amlas.data_eng import hub_dataset

    hub_dataset._repository(repo)
    assert hub_dataset._filename("data/records.jsonl") == "jsonl"
    assert (
        discover(repo_id=repo, revision="c" * 40, filename="data/records.jsonl")["query_status"]
        == "source_shape_supported"
    )
