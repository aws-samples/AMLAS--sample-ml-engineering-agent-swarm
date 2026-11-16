"""Specification stage: the generator and the validator tell the model the next call.

Nine of 51 runs with a real model ended before PAIR in the specification stage. The raw tool records show that
18 of 19 generate_spec_from_description results were the classification stub (no spec_text) although every call
carried planning_data, and that 24 of 26 validate_spec results were invalid on paths the model could not act on.
These tests use the call shapes, paths and error types of those runs (numbers and product literals only).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml
from test_managed_generation_product import spec_text as fixed_spec_text

from amlas.backends.adapters._util import raw_tool
from amlas.control_plane.generation_state import SessionCodingGenerationStore
from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest
from amlas.flows._util import raw
from amlas.spec.loader import load_spec_from_text
from amlas.tools import coding_generation, spec_generation_tools
from amlas.tools.coding_generation import bind_coding_generation_spec
from amlas.tools.spec_generation_tools import _CodingPlanningData, generate_spec_from_description
from amlas.tools.spec_heuristics import _infer_task_type
from amlas.tools.spec_tools import list_example_specs, validate_spec

pytest_plugins = ("test_managed_generation_product",)

METRIC = "pass_at_1"
HUMANEVAL_PIN = "openai/openai_humaneval@7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
GOLDEN_FORMS = [
    HUMANEVAL_PIN,
    "case_aa5e529f802d495197be7e1d0ccd33a3",
    "humaneval_164",
    "s3://example-private-bucket/case-freeze/output/PROMPTS.jsonl",
]
# Wordings with the structure of the nine runs: versioned model name, no phrase the old heuristic knew.
DESCRIPTIONS = [
    "Fixed Qwen coding task: SFT Qwen3.5-9B (full_v1) on prepared Python data, paired base vs tuned pass_at_1.",
    "Coding fine-tune of Qwen3.5-9B on Python change data with paired HumanEval-164 pass@1 evaluation.",
    "Supervised fine-tune Qwen3.5-9B with recipe full_v1; evaluate pass_at_1 on frozen cases, base versus tuned.",
]
PLANNING = {
    "model_catalog": {"id": "qwen3_5_9b", "version": "1"},
    "target": "pass_at_1",
    "acceptance": {"primary_metric_delta_vs_baseline": "> 0.0"},
    "compute_budget": {"gpu_hours": 1, "max_retreats": 1, "trials": 1, "wall_clock_minutes": 120},
}


def generate(description, **kwargs):
    return raw(generate_spec_from_description)(description=description, **kwargs)


def pin():
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    return {"repo_id": expected["repo_id"], "revision": expected["revision"]}


def valid_document():
    return {
        "project_id": "qwen_coding",
        "task_type": "text_generation",
        "target": "python_module",
        "primary_metric": "pass_at_1",
        "golden_dataset": "./cases.json",
        "baseline": {"type": "untouched_base_model", METRIC: None},
        "acceptance": {"primary_metric_delta_vs_baseline": "> 0.0"},
        "compute_budget": {"gpu_hours": 1, "trials": 1, "wall_clock_minutes": 120, "max_retreats": 1},
        "generation": {
            "model": pin(),
            "tokenizer": pin(),
            "recipe": {"id": "full_v1", "version": "1"},
            "quality": {"metric": "pass_at_1"},
        },
    }


def as_text(document):
    return "---\n" + yaml.safe_dump(document, sort_keys=False) + "---\n\nordinary notes\n"


def validated(document):
    return raw(validate_spec)(as_text(document))


# ── 1. the generator returns bindable text for the wordings the nine runs used ───────────────────────────


@pytest.mark.parametrize("golden", GOLDEN_FORMS)
@pytest.mark.parametrize("description", DESCRIPTIONS)
def test_planning_data_selects_the_coding_branch_and_the_text_binds_unchanged(description, golden):
    result = generate(description, planning_data={**PLANNING, "golden_dataset": golden})
    assert result["task_type"] == "text_generation"
    assert result["planning_status"] == "schema_valid", result
    assert result["primary_metric"] == "pass_at_1" and result["success_threshold"] is None
    assert result["target"] == "pass_at_1"
    assert result["task_type_source"] == "planning_data" or result["task_type_source"] == "description"
    text = result["spec_text"]
    SessionCodingGenerationStore._original_spec_material(text)
    parsed = load_spec_from_text(text)
    assert (
        parsed.generation.preparation.manifest_ref is parsed.generation.evaluation.case_manifest_ref is None
    )
    assert generate(description, planning_data={**PLANNING, "golden_dataset": golden})["spec_text"] == text


def test_wording_without_any_coding_phrase_is_still_coding_when_planning_data_is_present():
    result = generate("Paired evaluation of a tuned model against its base.", planning_data=dict(PLANNING))
    assert result["task_type"] == "text_generation"
    assert result["task_type_source"] == "planning_data"
    assert "primary_metric" in result["needs_research"]
    assert result["planning_status"] == "research_needed" and not result.get("spec_text")


def test_an_empty_planning_mapping_still_selects_the_coding_branch():
    result = generate("Paired evaluation of a tuned model against its base.", planning_data={})
    assert result["task_type"] == "text_generation" and result["planning_status"] == "research_needed"
    assert {"model_selection", "target", "golden_dataset", "acceptance"} <= set(result["needs_research"])


def test_planning_data_never_reaches_the_tabular_column_inspection(monkeypatch):
    calls = []
    monkeypatch.setattr(spec_generation_tools, "_validate_column_in_data", lambda *args: calls.append(args))
    result = generate(
        "Paired evaluation of a tuned model against its base.", data_path="./ordinary.csv", planning_data={}
    )
    assert result["task_type"] == "text_generation" and calls == []


# ── 2. the keyword heuristic knows the wordings of the nine runs without planning_data ───────────────────


@pytest.mark.parametrize(
    "description",
    [*DESCRIPTIONS, "Evaluate on HumanEval with a tuned model.", "Tune Qwen3.5 and report pass@1."],
)
def test_heuristic_recognises_versioned_qwen_pass_at_1_and_humaneval(description):
    assert _infer_task_type(description) == "text_generation"


@pytest.mark.parametrize(
    ("description", "task"),
    [
        ("Use Qwen3.5 embeddings to classify support tickets with F1.", "classification"),
        ("Forecast demand with RMSE using Qwen3.5.", "regression"),
        ("Predict churn; report accuracy.", "classification"),
    ],
)
def test_classification_and_regression_wording_still_win_over_the_new_signals(description, task):
    assert _infer_task_type(description) == task
    assert "spec_text" not in generate(description)


# ── 3. the generator result names the next call ──────────────────────────────────────────────────────────


def test_a_valid_coding_result_names_bind_then_resolve_and_forbids_hand_filled_refs():
    result = generate(DESCRIPTIONS[0], planning_data={**PLANNING, "golden_dataset": "./cases.json"})
    step = result["next_step"]
    assert "bind_coding_generation_spec" in step and "spec_text" in step and "unchanged" in step
    assert "generation.preparation" in step and "generation.evaluation" in step
    assert "resolve_coding_evaluation_spec" in step
    assert "validate_spec" in step
    assert "---" not in step and len(step) < 700


def test_research_needed_lists_path_expected_form_and_example_and_the_examples_are_accepted():
    description = "Fine-tune Qwen/Qwen3.5-9B as a coding model; evaluate held-out pass@1."
    result = generate(description)
    assert result["planning_status"] == "research_needed"
    detail = {item["name"]: item for item in result["needs_research_detail"]}
    assert set(detail) == set(result["needs_research"]) >= {"target", "golden_dataset", "acceptance"}
    assert "unbounded_budget" in detail["compute_budget"]["planning_data_path"]
    for item in detail.values():
        assert set(item) == {"name", "planning_data_path", "expected", "example"}
        assert item["expected"] and item["example"] not in (None, "")
    assert "generate_spec_from_description" in result["next_step"]
    # Feed the product's own examples back: the second call must return bindable text.
    filled = {}
    for item in detail.values():
        key = item["planning_data_path"].split("planning_data.")[1].split(" ")[0]
        filled[key] = item["example"]
    _CodingPlanningData.model_validate(filled, strict=True)
    again = generate(description, planning_data=filled)
    assert again["planning_status"] == "schema_valid", again
    SessionCodingGenerationStore._original_spec_material(again["spec_text"])


def test_an_omitted_budget_with_planning_data_keeps_its_schema_defaults_and_is_listed():
    result = generate(DESCRIPTIONS[0], planning_data={**PLANNING, "golden_dataset": "./cases.json"})
    assert result["planning_status"] == "schema_valid"
    # PLANNING gives all four ceilings; drop one: the generator reports the default instead of asking again.
    partial = {**PLANNING, "golden_dataset": "./cases.json", "compute_budget": {"trials": 1}}
    again = generate(DESCRIPTIONS[0], planning_data=partial)
    assert again["planning_status"] == "schema_valid"
    assert "compute_budget.gpu_hours" in again["source_schema_defaults"]


def test_a_missing_model_selection_is_reported_with_the_catalog_form():
    result = generate(
        "Paired evaluation of a tuned model with pass_at_1.", planning_data={"target": "python_module"}
    )
    detail = {item["name"]: item for item in result["needs_research_detail"]}
    assert detail["model_selection"]["planning_data_path"] == "planning_data.model_catalog"
    assert detail["model_selection"]["example"] == {"id": "qwen3_5_9b", "version": "1"}


def test_the_classification_stub_keeps_its_shape():
    result = generate("Classification of customer churn. The target is 'outcome'. Require F1 score.")
    assert result["task_type"] == "classification"
    assert not {"next_step", "planning_status", "task_type_source", "needs_research_detail"} & set(result)


# ── 4. validate_spec: error paths the nine runs hit, with path, expected form and example ───────────────


def paths_of(result):
    return {item["path"] for item in result["error_details"]}


def mutated(**change):
    document = valid_document()
    for key, value in change.items():
        if value is KeyError:
            document.pop(key)
        else:
            document[key] = value
    return document


def test_first_hand_written_spec_of_run_s2_is_explained_path_by_path():
    document = mutated(baseline=KeyError, golden_dataset=HUMANEVAL_PIN)
    document["generation"]["model"] = {"id": "qwen3_5_9b", "version": "1", **pin()}
    result = validated(document)
    assert result["valid"] is False and result["errors"][0].startswith("Spec validation failed:")
    assert {"golden_dataset", "baseline", "generation.model.id", "generation.model.version"} <= paths_of(
        result
    )
    by_path = {item["path"]: item for item in result["error_details"]}
    assert "s3://" in by_path["golden_dataset"]["expected"] and by_path["golden_dataset"]["example"]
    assert "untouched_base_model" in by_path["baseline"]["example"]
    assert "repo_id" in by_path["generation.model.id"]["expected"]
    assert "planning_data" in by_path["generation.model.id"]["expected"]
    assert HUMANEVAL_PIN not in json.dumps(result["error_details"])


def test_string_valued_preparation_and_evaluation_entries_are_told_to_stay_as_generated():
    document = valid_document()
    document["generation"]["preparation"] = {
        "profile": "example.profile.v1",
        "manifest_ref": "s3://b/COMMITTED.json",
    }
    document["generation"]["evaluation"] = {
        "harness": "humaneval_164",
        "case_manifest_ref": "s3://b/PROMPTS.jsonl",
    }
    result = validated(document)
    # Four failing entries (profile, manifest_ref, harness, case_manifest_ref) get one record per block.
    assert paths_of(result) == {"generation.preparation", "generation.evaluation"}
    for item in result["error_details"]:
        assert "spec_text" in item["expected"] and "null" in item["expected"]
        assert "resolve_coding_evaluation_spec" in item["expected"]


def test_dict_valued_refs_and_harness_pins_the_model_cannot_know_are_explained():
    document = valid_document()
    document["generation"]["preparation"] = {
        "manifest_ref": {
            "uri": "s3://example-bucket/prep/COMMITTED.json",
            "version_id": "v1",
            "sha256": "a" * 64,
        }
    }
    document["generation"]["evaluation"] = {
        "harness": {"id": "humaneval_164", "version": "1"},
        "case_manifest_ref": {
            "uri": "s3://example-bucket/cases/PROMPTS.jsonl",
            "version_id": "v1",
            "sha256": "b" * 64,
            "byte_count": 10,
            "format": "jsonl",
        },
    }
    result = validated(document)
    # Missing byte_count and format, harness id/version and agent/evaluator pins, format jsonl: two records.
    assert paths_of(result) == {"generation.preparation", "generation.evaluation"}
    by_path = {item["path"]: item for item in result["error_details"]}
    text = by_path["generation.evaluation"]["expected"]
    assert "json" in text and "repo_id" in text and "null" in text
    assert (
        by_path["generation.evaluation"]["example"] == "evaluation: {harness: null, case_manifest_ref: null}"
    )


def test_the_pathless_generation_declaration_error_names_both_metric_paths_and_the_fix():
    document = valid_document()
    document["generation"].pop("quality")
    result = validated(document)
    assert result["valid"] is False and len(result["error_details"]) == 1
    item = result["error_details"][0]
    assert "primary_metric" in item["path"] and "generation.quality.metric" in item["path"]
    assert "pass_at_1" in item["expected"] and "quality" in item["example"]
    assert "coding generation planning requires" in result["errors"][0]


def test_keys_from_planning_data_and_misspelled_pins_are_explained():
    document = mutated(model_catalog={"id": "qwen3_5_9b", "version": "1"}, splits={"strategy": "holdout"})
    document["generation"]["recipe_id"] = "full_v1"
    document["generation"]["recipe_version"] = "1"
    document["generation"]["quality_metric"] = "pass_at_1"
    document["generation"]["recipe"] = {"id": "full_v1", "version": 1}
    result = validated(document)
    by_path = {item["path"]: item for item in result["error_details"]}
    assert {
        "model_catalog",
        "generation.recipe_id",
        "generation.recipe_version",
        "generation.recipe.version",
    } <= set(by_path)
    assert "planning_data" in by_path["model_catalog"]["expected"]
    assert "quote" in by_path["generation.recipe.version"]["expected"].lower()
    assert "splits.strategy" in by_path and "stratified" in by_path["splits.strategy"]["expected"]


def test_unknown_keys_are_never_reflected():
    document = valid_document()
    document["x" * 300] = 1
    document["odd key; with & chars"] = 2
    document["generation"]["odd nested key"] = 3
    result = validated(document)
    text = json.dumps(result["error_details"]) + result["next_step"]
    assert "x" * 40 not in text and "odd key" not in text and "odd nested key" not in text
    assert paths_of(result) == {"(unknown key)", "generation.(unknown key)"}


def test_the_detail_list_is_capped_and_counts_what_it_leaves_out():
    from amlas.tools.spec_guidance import MAX_DETAILS, spec_error_details

    names = [
        "golden_dataset",
        "baseline",
        "model_catalog",
        "splits.strategy",
        "generation.model.id",
        "generation.model.version",
        "generation.tokenizer.id",
        "generation.tokenizer.version",
        "generation.recipe_id",
        "generation.recipe_version",
        "generation.quality_metric",
        "generation.preparation.profile",
        "generation.preparation.manifest_ref",
    ]
    issues = [(tuple(name.split(".")), "extra_forbidden") for name in names]
    details, omitted = spec_error_details(issues)
    # profile and manifest_ref share one record (generation.preparation): 12 distinct paths, 8 shown.
    assert MAX_DETAILS == 8 and len(details) == 8 and omitted == 4
    assert [item["path"] for item in details] == names[:8]


def test_error_details_are_deterministic_and_the_legacy_error_text_is_kept_first():
    document = mutated(baseline=KeyError, golden_dataset="bare/name")
    first, second = validated(document), validated(document)
    assert first == second
    assert first["errors"][0].startswith("Spec validation failed:\n2 validation errors for SpecModel")


def test_an_invalid_result_starts_the_next_step_with_the_generator():
    result = validated(mutated(baseline=KeyError))
    step = result["next_step"]
    assert step.startswith("Call generate_spec_from_description with planning_data")
    assert "spec_text" in step and "unchanged" in step and "error_details" in step


def test_a_valid_coding_spec_points_to_bind_and_a_valid_classification_spec_gets_no_hint():
    coding = validated(valid_document())
    assert coding["valid"] is True and "error_details" not in coding
    assert "bind_coding_generation_spec" in coding["next_step"]
    classification = raw(validate_spec)(
        "---\n"
        + yaml.safe_dump(
            {
                "project_id": "churn",
                "task_type": "classification",
                "target": "outcome",
                "primary_metric": "f1_macro",
                "golden_dataset": "./churn.csv",
                "baseline": {"type": "heuristic_default", "f1_macro": 0.0},
                "acceptance": {"primary_metric_delta_vs_baseline": ">= 0.0"},
            },
            sort_keys=False,
        )
        + "---\n\nnotes\n"
    )
    assert classification["valid"] is True and "next_step" not in classification


def test_unparseable_front_matter_gets_a_fixed_record_without_an_excerpt():
    result = raw(validate_spec)("ordinary text without front matter")
    assert result["valid"] is False
    assert "---" in result["next_step"] and "ordinary text" not in json.dumps(result["error_details"])


# ── 5. the binder and the example list ───────────────────────────────────────────────────────────────────


def bind_with_owner(store, text):
    return raw_tool(bind_coding_generation_spec)(
        text, tool_context=SimpleNamespace(invocation_state={"_amlas_generation_store": store})
    )


def test_a_schema_refusal_sends_the_model_to_the_generator_first_and_keeps_the_closed_record(state):
    result = bind_with_owner(state[0], "---\nproject_id: hand_written\n---\n\nnotes\n")
    assert result["validation"]["reason_codes"] == ["parse_or_schema_invalid"]
    assert result["validation"]["field_paths"] == ["$"]
    step = result["next_step"]
    assert step.startswith("Call generate_spec_from_description with planning_data")
    assert "validate_spec" in step and "unchanged" in step
    assert result["next_step"] == coding_generation._NEXT_STEP["parse_or_schema_invalid"]


def test_the_model_pin_refusal_does_not_invite_catalog_ids_into_the_pin():
    step = coding_generation._NEXT_STEP["model_pin_required"]
    assert "repo_id" in step and "revision" in step and "generate_spec_from_description" in step


def test_a_successful_bind_names_the_resolve_call_and_does_not_claim_completion(state):
    result = bind_with_owner(state[0], fixed_spec_text())
    assert result["success"] is True and result["spec_ref"].startswith("spec_")
    step = result["next_step"]
    assert "resolve_coding_evaluation_spec" in step and "spec_ref" in step
    assert "submit_coding_generation" in step and "resolved" in step
    assert result["native_admission"] is result["quality_success"] is False


def test_the_example_list_drops_the_repository_guide_and_says_the_examples_are_not_the_coding_task():
    result = raw(list_example_specs)()
    names = {item["name"] for item in result["specs"]}
    assert "AGENTS" not in names and "ai4i_binary_failure" in names
    assert "generate_spec_from_description" in result["next_step"] and "classification" in result["next_step"]


def test_tool_descriptions_state_the_selection_and_the_unchanged_binding_rule():
    generator = generate_spec_from_description.tool_spec["description"]
    assert "selects the coding task" in generator and "bind_coding_generation_spec" in generator
    assert "unchanged" in generator
    assert "error_details" in validate_spec.tool_spec["description"]
    assert "unchanged" in bind_coding_generation_spec.tool_spec["description"]
