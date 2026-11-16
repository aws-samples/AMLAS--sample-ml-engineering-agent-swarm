"""Planning-only schemas preserve legacy bytes and never manufacture native trust."""

from __future__ import annotations

import copy
import dataclasses
import json
import socket

import pytest
from pydantic import ValidationError

from amlas.backends.types import DatasetArtifactRef
from amlas.spec import generation
from amlas.spec.loader import load_spec_from_text
from amlas.spec.schema import SpecModel

LEGACY_JSON = '{"project_id":"legacy_project","task_type":"regression","target":"score","primary_metric":"rmse","golden_dataset":"./golden.csv","constraints":{"no_pii_features":false,"inference_latency_ms":null,"max_model_size_mb":null},"splits":{"strategy":"stratified","cutoff":null,"group_col":null},"baseline":{"type":"mean","rmse":1.2},"compute_budget":{"gpu_hours":0.0,"trials":50,"wall_clock_minutes":30,"max_retreats":3},"acceptance":{"primary_metric_delta_vs_baseline":"<= -0.1","no_subgroup_regression":true},"medallion":null,"domain_notes":""}'
PIN = {"repo_id": "Qwen/Qwen3.8-27B", "revision": "a" * 40}
SELECTOR = {"id": "fixture_profile", "version": "1"}
REF = {"uri": "s3://fixture-data/manifest.json", "version_id": "fixture-version-1",
       "sha256": "b" * 64, "byte_count": 256, "format": "json"}


def model(name):
    return getattr(generation, name)


def payload(block=None):
    return {
        "project_id": "coding_plan", "task_type": "text_generation", "target": "messages",
        "primary_metric": "resolved_rate", "golden_dataset": "./cases.json",
        "baseline": {"type": "pinned_base", "resolved_rate": None},
        "acceptance": {"primary_metric_delta_vs_baseline": ">= 0.01"},
        "generation": {} if block is None else block,
    }


def valid_spec(value):
    result, error = None, None
    try:
        result = SpecModel.model_validate(value)
    except ValidationError as failure:
        error = failure
    assert result is not None, f"declared planning data should validate: {error}"
    return result


@pytest.mark.parametrize("explicit_null", [False, True])
def test_legacy_serialization_is_byte_equivalent_and_omits_generation(explicit_null):
    source = json.loads(LEGACY_JSON)
    if explicit_null:
        source["generation"] = None
    result = valid_spec(source)
    assert result.model_dump_json() == LEGACY_JSON
    assert result.model_dump() == json.loads(LEGACY_JSON)
    assert "generation" not in result.model_dump(include={"generation"})


@pytest.mark.parametrize("state", ["unmeasured", "pending", "failed"])
def test_draft_nullable_baseline_and_unresolved_bindings_are_planning_data(state):
    result = valid_spec(payload({"baseline_measurement": {"state": state}}))
    assert result.baseline.model_dump()["resolved_rate"] is None
    assert result.generation.baseline_measurement.state == state
    assert result.generation.model is result.generation.tokenizer is result.generation.recipe is None
    assert result.generation.preparation.manifest_ref is None
    assert result.generation.evaluation.harness is None
    assert result.generation.evaluation.case_manifest_ref is None
    wire = result.model_dump_json()
    assert SpecModel.model_validate_json(wire).model_dump_json() == wire


def test_normal_frontmatter_loader_retains_typed_generation_without_native_effects(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("planning schema must not resolve resources or catalogs")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    source = payload({
        "model": PIN, "tokenizer": PIN, "recipe": SELECTOR,
        "preparation": {"profile": SELECTOR, "manifest_ref": REF},
        "evaluation": {"harness": {"agent": PIN, "evaluator": PIN}, "case_manifest_ref": REF},
        "baseline_measurement": {"state": "pending"},
    })
    result = None
    try:
        result = load_spec_from_text("---\n" + json.dumps(source) + "\n---\nPlanning only.")
    except Exception as error:
        raise AssertionError(f"normal loader should accept descriptive planning types: {error}") from error
    assert result.generation.model.revision == PIN["revision"]
    assert result.generation.preparation.manifest_ref.model_dump() == REF
    assert result.domain_notes == "Planning only."
    assert result.baseline.model_dump()["resolved_rate"] is None


@pytest.mark.parametrize("revision", ["main", "latest", "a" * 39, "a" * 41, "A" * 40, "g" * 40, 1, None])
def test_hub_pin_requires_exact_immutable_commit(revision):
    cls = model("HubPin")
    assert cls.model_validate(PIN).revision == PIN["revision"]
    with pytest.raises(ValidationError):
        cls.model_validate(dict(PIN, revision=revision))


@pytest.mark.parametrize("repo", ["", "model", "owner/repo/extra", "../repo", "https://host/repo", "owner/ bad", "o/" + "x" * 200])
def test_hub_pin_repository_is_bounded_and_not_a_url_or_path(repo):
    with pytest.raises(ValidationError):
        model("HubPin").model_validate(dict(PIN, repo_id=repo))


@pytest.mark.parametrize("version", ["", "latest", "main", "*", ">=1", " version ", 1, True, "v" * 65])
def test_catalog_selection_is_bounded_explicit_data_not_a_floating_selector(version):
    cls = model("CatalogSelection")
    assert cls.model_validate(SELECTOR).id == SELECTOR["id"]
    with pytest.raises(ValidationError):
        cls.model_validate(dict(SELECTOR, version=version))


def test_manifest_reference_keeps_exact_existing_five_field_shape():
    result = model("ManifestRef").model_validate(REF)
    assert result.model_dump() == REF
    assert set(result.model_dump()) == {field.name for field in dataclasses.fields(DatasetArtifactRef)}


@pytest.mark.parametrize("change", [
    {"format": "parquet"}, {"format": "jsonl"}, {"format": "pickle"},
    {"uri": "https://example.com/data"}, {"uri": "/local/data"}, {"uri": "s3://bucket"},
    {"uri": "s3://bucket/key?versionId=x"}, {"uri": "s3://user@bucket/key"},
    {"version_id": ""}, {"version_id": "null"}, {"version_id": None},
    {"sha256": "b" * 63}, {"sha256": "G" * 64}, {"sha256": True},
    {"byte_count": 0}, {"byte_count": -1}, {"byte_count": True},
    {"byte_count": 256.0}, {"byte_count": float("inf")}, {"provider": "trusted"},
])
def test_malformed_or_wrong_format_manifest_reference_is_rejected(change):
    cls = model("ManifestRef")
    assert cls.model_validate(REF).format == "json"
    with pytest.raises(ValidationError):
        cls.model_validate(dict(REF, **change))


@pytest.mark.parametrize("grant", [
    "image_uri", "role_arn", "account_id", "region", "approval_policy", "ready",
    "bypass", "native_handle", "native_job_arn", "native_verified", "_amlas_approval_policy",
])
def test_generation_plan_rejects_all_extra_grants(grant):
    cls = model("GenerationPlan")
    cls.model_validate({})
    with pytest.raises(ValidationError) as error:
        cls.model_validate({grant: True})
    assert any(item["type"] == "extra_forbidden" for item in error.value.errors())


@pytest.mark.parametrize("name,valid", [
    ("HubPin", PIN), ("CatalogSelection", SELECTOR), ("ManifestRef", REF),
    ("PreparationPlan", {}), ("EvaluationHarnessPin", {"agent": PIN, "evaluator": PIN}),
    ("EvaluationPlan", {}), ("CodingQuality", {}), ("BaselineMeasurement", {}),
])
def test_every_nested_planning_model_rejects_extra_authority(name, valid):
    cls = model(name)
    cls.model_validate(valid)
    with pytest.raises(ValidationError) as error:
        cls.model_validate(dict(valid, ready=True))
    assert any(item["type"] == "extra_forbidden" for item in error.value.errors())


@pytest.mark.parametrize("change", [
    {"metric": "accuracy"}, {"comparison": "judge"}, {"coverage": "optional_cases"},
    {"failure_policy": "ignore"}, {"harness_controls": "skip"},
])
def test_coding_quality_keeps_fixed_execution_evidence_rules(change):
    with pytest.raises(ValidationError):
        model("CodingQuality").model_validate(change)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), True, False, "0.5", -0.1, 1.1])
def test_generation_baseline_score_rejects_nonfinite_bool_coercion_and_out_of_range(score):
    source = payload({"baseline_measurement": {"state": "measured", "report_ref": REF}})
    valid = copy.deepcopy(source)
    valid["baseline"]["resolved_rate"] = 0.5
    valid_spec(valid)
    source["baseline"]["resolved_rate"] = score
    with pytest.raises(ValidationError):
        SpecModel.model_validate(source)


@pytest.mark.parametrize("score", [0, 0.5, 1])
def test_descriptive_measured_score_requires_report_but_does_not_create_native_trust(score):
    source = payload({"baseline_measurement": {"state": "measured", "report_ref": REF}})
    source["baseline"]["resolved_rate"] = score
    result = valid_spec(source)
    assert result.generation.baseline_measurement.report_ref.model_dump() == REF
    assert result.baseline.model_dump()["resolved_rate"] == score
    source["generation"]["baseline_measurement"]["report_ref"] = None
    with pytest.raises(ValidationError):
        SpecModel.model_validate(source)


@pytest.mark.parametrize("change", [
    {"task_type": "classification"}, {"task_type": "regression"}, {"primary_metric": "rmse"},
    {"baseline": {"type": "pinned_base", "resolved_rate": 0.5}},
    {"baseline": {"type": "pinned_base", "resolved_rate": None, "ready": True}},
    {"acceptance": {"primary_metric_delta_vs_baseline": ">= 0.01", "ready": True}},
])
def test_contradictory_generation_task_or_baseline_declaration_is_rejected(change):
    valid_spec(payload())
    source = payload()
    source.update(change)
    with pytest.raises(ValidationError):
        SpecModel.model_validate(source)


def test_extreme_out_of_range_integer_score_raises_validation_not_overflow():
    source = payload({"baseline_measurement": {"state": "measured", "report_ref": REF}})
    source["baseline"]["resolved_rate"] = 10 ** 500
    error = None
    try:
        SpecModel.model_validate(source)
    except Exception as failure:
        error = failure
    assert isinstance(error, ValidationError), "numeric bounds must report schema validation, not overflow"


@pytest.mark.parametrize("change", [{"uri": "s3://fixture-data/key\x00"}, {"version_id": "v1\x00"}])
def test_manifest_reference_rejects_control_characters(change):
    with pytest.raises(ValidationError):
        model("ManifestRef").model_validate(dict(REF, **change))
