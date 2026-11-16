"""Finite planning refusal through original private store and actual SDK tools."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_managed_generation_product import spec_text
from test_sdk_acquisition_binding import invoke, request

from amlas.backends.adapters._util import raw_tool
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalPolicy
from amlas.control_plane.generation_state import SessionCodingGenerationStore
from amlas.tools.coding_generation import bind_coding_generation_spec
from amlas.tools.spec_tools import validate_spec
from amlas.tui.chat.agent_tools import get_safe_llm_tools, tool_display_name
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytest_plugins = ("test_managed_generation_product",)
SCHEMA = "amlas.fixed_qwen_coding_spec.v1"
SOURCE_PIN = json.loads(spec_text().split("---")[1])["generation"]["model"]
PRIVATE_TEXT = "ordinary private specification detail"


def diagnostic(code, path):
    return {
        "schema_id": SCHEMA,
        "status": "invalid",
        "reason_codes": [code],
        "field_paths": [path],
        "data_only": True,
        "native_admission": False,
        "quality_success": False,
    }


def bind_with_owner(store, text):
    return raw_tool(bind_coding_generation_spec)(
        text, tool_context=SimpleNamespace(invocation_state={"_amlas_generation_store": store})
    )


def assert_no_effects(state):
    store, adapter, reads, *_ = state
    assert store._specs == {} and store._spec_identity == {}
    assert adapter.calls == [] and adapter.reads == 0 and reads == []


def test_actual_registered_sdk_generic_valid_legacy_spec_gets_actionable_private_refusal(state, monkeypatch):
    store = state[0]
    # A legacy generic regression/pass_at_1 document is valid under the actual
    # existing generic schema, but carries no generation/model/recipe authority.
    value = {
        "project_id": "legacy_coding_draft", "task_type": "regression",
        "target": "python_module", "primary_metric": "pass_at_1",
        "golden_dataset": "./unresolved_cases.json",
        "baseline": {"type": "declared_base", "pass_at_1": None},
        "acceptance": {"primary_metric_delta_vs_baseline": ">= 0.02"},
    }
    text = "---\n" + json.dumps(value) + "\n---\n" + PRIVATE_TEXT
    backend = ToolBindingBackend(Settings(), approval_policy=ApprovalPolicy())
    backend._generation_store = store
    graph = backend._warm_graph()
    names = {tool_display_name(tool) for tool in get_safe_llm_tools()}
    assert {"validate_spec", "bind_coding_generation_spec"} <= names
    results = invoke(
        monkeypatch, backend,
        [request("generic", "validate_spec", {"yaml_text": text}),
         request("coding", "bind_coding_generation_spec", {"yaml_text": text})],
        tools=[validate_spec, bind_coding_generation_spec],
    )
    assert results["generic"]["valid"] is True
    result = results["coding"]
    assert result["success"] is False and result["status"] == "refused"
    assert result["error_type"] == "ValueError" and "spec_ref" not in result
    assert result.get("validation") == diagnostic("task_type_mismatch", "task_type")
    assert PRIVATE_TEXT not in json.dumps(result)
    assert all(a is b for a, b in zip(graph, backend._warm_graph(), strict=True))
    assert backend._generation_store is store
    assert_no_effects(state)


@pytest.mark.parametrize(
    "text,code,path",
    [
        ("", "bounded_text_required", "$"),
        ("ordinary text without frontmatter", "parse_or_schema_invalid", "$"),
        (spec_text(task_type="text_generation", generation=None), "generation_required", "generation"),
        (spec_text(primary_metric="resolved_rate", generation=None), "primary_metric_mismatch", "primary_metric"),
        (spec_text(generation={"quality": {"metric": "pass_at_1"}}), "model_pin_required", "generation.model"),
        (spec_text(generation={"model": {"repo_id": "Qwen/other", "revision": "a" * 40},
                               "tokenizer": {"repo_id": "Qwen/other", "revision": "a" * 40},
                               "quality": {"metric": "pass_at_1"}, "recipe": {"id": "full_v1", "version": "1"}}),
         "model_profile_mismatch", "generation.model"),
        (spec_text(generation={"model": SOURCE_PIN,
                               "tokenizer": {"repo_id": "Qwen/other", "revision": "a" * 40},
                               "quality": {"metric": "pass_at_1"}, "recipe": {"id": "full_v1", "version": "1"}}),
         "tokenizer_pin_mismatch", "generation.tokenizer"),
        (spec_text(generation={"model": SOURCE_PIN,
                               "tokenizer": SOURCE_PIN,
                               "quality": {"metric": "pass_at_1"}}), "recipe_required", "generation.recipe"),
        (spec_text(generation={"model": SOURCE_PIN,
                               "tokenizer": SOURCE_PIN,
                               "quality": {"metric": "pass_at_1"}, "recipe": {"id": "full_v1", "version": "2"}}),
         "recipe_profile_mismatch", "generation.recipe"),
    ],
)
def test_known_fixed_profile_refusals_are_closed_data_without_binding(state, text, code, path):
    result = bind_with_owner(state[0], text)
    assert result["success"] is False and result["status"] == "refused"
    assert result["error_type"] == "ValueError" and "spec_ref" not in result
    assert result.get("validation") == diagnostic(code, path)
    assert_no_effects(state)


@pytest.mark.parametrize("kind", ["missing", "foreign", "subclass"])
def test_public_or_foreign_context_does_not_get_private_validation_authority(state, kind):
    if kind == "missing":
        store = None
    elif kind == "foreign":
        store = {"catalog": "public planning claims", "ready": True}
    else:
        class UnqualifiedStore(SessionCodingGenerationStore):
            pass
        store = object.__new__(UnqualifiedStore)
    result = bind_with_owner(store, spec_text())
    assert result == {"success": False, "status": "denied", "quality_success": False, "native_admission": False}
    assert "validation" not in result and "spec_ref" not in result
    assert_no_effects(state)


def test_unexpected_binder_error_stays_opaque(state, monkeypatch):
    def fail(_):
        raise RuntimeError(PRIVATE_TEXT)
    monkeypatch.setattr(state[0], "bind_spec", fail)
    result = bind_with_owner(state[0], spec_text())
    assert result["success"] is False and result["status"] == "refused"
    assert "validation" not in result and "spec_ref" not in result
    assert PRIVATE_TEXT not in json.dumps(result)
    assert_no_effects(state)


def test_valid_original_binding_still_deduplicates_canonical_bytes_without_reads(state):
    first = bind_with_owner(state[0], spec_text())
    again = bind_with_owner(state[0], spec_text())
    assert first["success"] is True and first == again
    assert first["native_admission"] is first["quality_success"] is False
    assert len(state[0]._specs) == 1 and state[1].calls == [] and state[2] == []
    assert "validation" not in first
