"""Explicit functional metric planning; no grading or readiness authority."""

import copy

import pytest
from pydantic import ValidationError
from test_generation_planning_types import LEGACY_JSON, REF, payload

from amlas.spec.schema import SpecModel


def functional(state="unmeasured", score=None):
    value = payload({"quality": {"metric": "pass_at_1"}, "baseline_measurement": {"state": state}})
    value["primary_metric"] = "pass_at_1"
    value["baseline"] = {"type": "pinned_base", "pass_at_1": score}
    if state == "measured":
        value["generation"]["baseline_measurement"]["report_ref"] = REF
    return value


def test_functional_pass_at_one_is_explicit_and_roundtrips():
    source = functional()
    before = copy.deepcopy(source)
    spec = SpecModel.model_validate(source)
    assert source == before
    assert spec.generation.quality.metric == spec.primary_metric == "pass_at_1"
    assert spec.baseline.model_dump()["pass_at_1"] is None
    assert spec.generation.quality.coverage == "all_required_cases"
    assert spec.generation.quality.comparison == "paired_same_cases"
    assert SpecModel.model_validate_json(spec.model_dump_json()).model_dump_json() == spec.model_dump_json()


def test_measured_functional_baseline_is_descriptive_only():
    spec = SpecModel.model_validate(functional("measured", 0.5))
    assert spec.baseline.model_dump()["pass_at_1"] == 0.5
    assert spec.generation.baseline_measurement.report_ref.uri == REF["uri"]


@pytest.mark.parametrize("metric", ["resolved_rate", "accuracy", "pass@1"])
def test_functional_quality_does_not_relabel_primary_metric(metric):
    value = functional()
    value["primary_metric"] = metric
    with pytest.raises(ValidationError):
        SpecModel.model_validate(value)


def test_existing_resolved_rate_default_and_legacy_bytes_remain_identical():
    spec = SpecModel.model_validate(payload())
    assert spec.primary_metric == spec.generation.quality.metric == "resolved_rate"
    assert SpecModel.model_validate_json(LEGACY_JSON).model_dump_json() == LEGACY_JSON
