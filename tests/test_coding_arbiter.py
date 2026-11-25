"""Closed HumanEval164 paired arithmetic, never native execution authority."""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module

import pytest
from test_generation_planning_types import payload

from amlas.arbiter.schema import InvalidSubmissionError
from amlas.data_eng.coding_case_profile import SOURCE_SHA256
from amlas.spec.schema import SpecModel


def module():
    return import_module("amlas.arbiter.coding")


def binding():
    return module().CodingEvaluationBinding(
        evaluation_id="fixture-paired-evaluation",
        spec_sha256=module().coding_spec_sha256(spec()),
        case_manifest={
            "uri": "s3://fixture-coding-bucket/cases/CASE_MANIFEST.json",
            "version_id": "fixture-case-version",
            "sha256": "1" * 64,
            "byte_count": 12000,
            "format": "json",
        },
        source_sha256=SOURCE_SHA256,
        runner_sha256="2" * 64,
        runtime_sha256="3" * 64,
        budgets_sha256="4" * 64,
        generation_sha256="5" * 64,
        base_model_sha256="6" * 64,
        tuned_model_sha256="7" * 64,
    )


def pins():
    return tuple(
        module().CodingCasePin(case_id=f"HumanEval/{i}", oracle_sha256=f"{i + 1:064x}") for i in range(164)
    )


def receipts(bound=None):
    bound = bound or binding()
    return tuple(
        module().CodingCaseReceipt(
            case_id=f"HumanEval/{i}",
            arm=arm,
            binding=bound,
            model_sha256=bound.base_model_sha256 if arm == "base" else bound.tuned_model_sha256,
            candidate_sha256=f"{1000 + i:064x}",
            oracle_sha256=f"{i + 1:064x}",
            outcome="passed" if i < (80 if arm == "base" else 90) else "failed",
            oracle_completed=True,
            exit_code=0 if i < (80 if arm == "base" else 90) else 1,
            execution_id=f"fixture-{arm}-{i}",
            stdout_sha256="8" * 64,
        )
        for arm in ("base", "tuned")
        for i in range(164)
    )


def report(values=None, bound=None):
    b = bound or binding()
    return (
        module()
        .CodingEvaluatorFactory()
        .build_report(binding=b, case_pins=pins(), receipts=receipts(b) if values is None else values)
    )


def spec(metric="pass_at_1"):
    manifest = {
        "uri": "s3://fixture-coding-bucket/cases/CASE_MANIFEST.json",
        "version_id": "fixture-case-version",
        "sha256": "1" * 64,
        "byte_count": 12000,
        "format": "json",
    }
    value = payload({"quality": {"metric": metric}, "evaluation": {"case_manifest_ref": manifest}})
    value["primary_metric"] = metric
    value["baseline"] = {"type": "pinned_base", metric: None}
    return SpecModel.model_validate(value)


def test_exact_paired_counts_and_two_times164_denominator():
    summary = report().describe()
    assert summary["base_pass_count"] == 80 and summary["tuned_pass_count"] == 90
    assert summary["case_count"] == 164 and summary["execution_count"] == 328
    assert (
        summary["both_pass"] == 80
        and summary["base_only"] == 0
        and summary["tuned_only"] == 10
        and summary["both_fail"] == 74
    )
    assert summary["base_pass_at_1"] == 80 / 164 and summary["tuned_pass_at_1"] == 90 / 164
    assert summary["native_admission"] is False and summary["grading_integrity_verified"] is False


@pytest.mark.parametrize("case", ["missing", "duplicate", "foreign", "single_arm"])
def test_missing_duplicate_foreign_or_single_arm_population_refused(case):
    values = list(receipts())
    if case == "missing":
        values.pop()
    elif case == "duplicate":
        values[-1] = values[0]
    elif case == "foreign":
        values[-1] = values[-1].model_copy(update={"case_id": "HumanEval/999"})
    else:
        values = values[:164]
    with pytest.raises(InvalidSubmissionError):
        report(tuple(values))


@pytest.mark.parametrize(
    "field",
    [
        "source_sha256",
        "runner_sha256",
        "runtime_sha256",
        "budgets_sha256",
        "generation_sha256",
        "base_model_sha256",
        "tuned_model_sha256",
        "evaluation_id",
    ],
)
def test_receipt_binding_drift_refuses(field):
    values = list(receipts())
    bad = values[0].binding.model_copy(update={field: "f" * 64 if field != "evaluation_id" else "foreign"})
    values[0] = values[0].model_copy(update={"binding": bad})
    with pytest.raises(InvalidSubmissionError):
        report(tuple(values))


@pytest.mark.parametrize("field", ["model_sha256", "oracle_sha256"])
def test_model_arm_or_private_case_hash_substitution_refuses(field):
    values = list(receipts())
    values[0] = values[0].model_copy(update={field: "f" * 64})
    with pytest.raises(InvalidSubmissionError):
        report(tuple(values))


@pytest.mark.parametrize(
    "outcome", ["timeout", "truncated", "infrastructure_error", "incomplete", "syntax_error", "runtime_error"]
)
def test_all_failure_and_infrastructure_outcomes_keep_denominator(outcome):
    values = list(receipts())
    values[0] = values[0].model_copy(
        update={"outcome": outcome, "oracle_completed": False, "exit_code": None}
    )
    summary = report(tuple(values)).describe()
    assert summary["case_count"] == 164 and summary["execution_count"] == 328
    assert summary["base_pass_count"] == 79 and summary["both_pass"] == 79
    assert summary["base_failure_counts"][outcome] == 1
    assert summary["quality_success"] is False


@pytest.mark.parametrize("update", [{"oracle_completed": False}, {"exit_code": None}, {"exit_code": 1}])
def test_exit0_or_stdout_cannot_make_uncompleted_or_failed_oracle_pass(update):
    values = list(receipts())
    values[0] = values[0].model_copy(update=update)
    with pytest.raises(InvalidSubmissionError):
        report(tuple(values))


def test_existing_backend_grades_private_pair_but_cannot_accept_unqualified_quality():
    from amlas.backends.adapters.evaluation import DeterministicArbiter

    result = DeterministicArbiter().grade(spec=spec(), y_true=None, y_pred=report())
    assert result.task_type == "text_generation" and result.primary_metric == "pass_at_1"
    assert result.overall_score == 90 / 164 and result.baseline_score == 80 / 164
    assert result.delta_vs_baseline == pytest.approx(10 / 164)
    assert result.n_graded == 164 and result.n_excluded_unlabelled == 0
    assert result.passes_acceptance is False
    assert "unqualified" in result.acceptance_reason


@pytest.mark.parametrize(
    "value",
    [
        {"overall_score": 1.0, "native_admission": True},
        [True] * 164,
        {"receipts": [], "quality_success": True},
    ],
)
def test_caller_metrics_or_outcomes_json_cannot_enter_coding_arbiter(value):
    from amlas.arbiter.arbiter import grade

    with pytest.raises(InvalidSubmissionError):
        grade(spec=spec(), y_true=None, y_pred=value)


def test_spec_profile_and_baseline_override_mismatch_refuse():
    from amlas.arbiter.arbiter import grade

    with pytest.raises(InvalidSubmissionError):
        grade(spec=spec("resolved_rate"), y_true=None, y_pred=report())
    value = spec().model_dump()
    value["generation"]["evaluation"]["case_manifest_ref"]["version_id"] = "foreign"
    with pytest.raises(InvalidSubmissionError):
        grade(spec=SpecModel.model_validate(value), y_true=None, y_pred=report())
    with pytest.raises(InvalidSubmissionError):
        grade(spec=spec(), y_true=None, y_pred=report(), baseline_score=1.0)


def test_report_does_not_accept_data_type_metrics_or_mutate_inputs():
    b = binding()
    values = receipts(b)
    before = deepcopy(values)
    result = report(values, b)
    assert values == before
    public = result.describe()
    public["native_admission"] = True
    public["tuned_pass_at_1"] = 1.0
    assert result.describe()["native_admission"] is False and result.describe()["tuned_pass_at_1"] == 90 / 164


def test_closed_schema_rejects_grants_extra_metrics_bool_counts_and_bypassed_constructor():
    from pydantic import ValidationError

    b = binding().model_dump()
    b["native_admission"] = True
    with pytest.raises(ValidationError):
        module().CodingEvaluationBinding.model_validate(b)
    value = receipts()[0].model_dump()
    value["passed"] = True
    with pytest.raises(ValidationError):
        module().CodingCaseReceipt.model_validate(value)
    value = receipts()[0].model_dump()
    value["exit_code"] = False
    with pytest.raises(ValidationError):
        module().CodingCaseReceipt.model_validate(value)
    values = list(receipts())
    values[0] = module().CodingCaseReceipt.model_construct(**value)
    with pytest.raises(InvalidSubmissionError):
        report(tuple(values))
    with pytest.raises(InvalidSubmissionError):
        module()._VerifiedCodingReport(binding(), {"tuned_pass_at_1": 1}, object())


def test_shuffled_complete_receipts_pair_by_case_ids_not_row_position():
    values = tuple(reversed(receipts()))
    assert report(values).describe() == report().describe()


def test_duplicate_executor_and_case_pins_refuse():
    values = list(receipts())
    values[1] = values[1].model_copy(update={"execution_id": values[0].execution_id})
    with pytest.raises(InvalidSubmissionError):
        report(tuple(values))
    original = list(pins())
    original[-1] = original[0]
    with pytest.raises(InvalidSubmissionError):
        module().CodingEvaluatorFactory().build_report(
            binding=binding(), case_pins=tuple(original), receipts=receipts()
        )


def test_predictive_route_and_registered_grader_population_stay_unchanged():
    from amlas.arbiter.arbiter import grade
    from amlas.arbiter.registry import REGISTRY
    from amlas.spec.schema import SpecModel

    value = {
        "project_id": "predictive_fixture",
        "golden_dataset": "./fixture.csv",
        "task_type": "classification",
        "target": "label",
        "primary_metric": "f1_macro",
        "baseline": {"type": "fixture", "f1_macro": 0.0},
        "acceptance": {"primary_metric_delta_vs_baseline": ">= 0.0", "no_subgroup_regression": False},
    }
    predictive = SpecModel.model_validate(value)
    result = grade(spec=predictive, y_true=[0, 1, 0, 1], y_pred=[0, 1, 0, 1])
    assert result.overall_score == 1.0 and result.passes_acceptance is True
    assert result.n_graded == 4 and result.n_excluded_unlabelled == 0
    assert set(REGISTRY) == {"classification", "regression", "detection", "segmentation"}


def test_review_spec_generation_budget_changes_cannot_reuse_prior_pair():
    from amlas.arbiter.arbiter import grade

    prior = report()
    value = spec().model_dump()
    value["compute_budget"]["trials"] += 1
    changed = SpecModel.model_validate(value)
    with pytest.raises(InvalidSubmissionError):
        grade(spec=changed, y_true=None, y_pred=prior)


def test_review_private_report_ordinary_mutation_cannot_change_math_or_manifest():
    from amlas.arbiter.arbiter import grade

    prior = report()
    with pytest.raises((AttributeError, TypeError)):
        prior._summary = {"tuned_pass_at_1": 1.0}
    with pytest.raises((AttributeError, TypeError)):
        prior._binding = binding().model_copy(update={"runtime_sha256": "f" * 64})
    assert grade(spec=spec(), y_true=None, y_pred=prior).overall_score == 90 / 164


def test_review_backend_result_retains_declared_provenance_without_oracles():
    from amlas.arbiter.arbiter import grade

    result = grade(spec=spec(), y_true=None, y_pred=report()).model_dump()
    record = result["phase_log"][-1]["coding_evaluation"]
    assert record["binding"] == binding().model_dump()
    assert record["provenance_status"] == "declared_consistency_checked_not_authenticated"
    assert "test" not in str(record) and "canonical_solution" not in str(record)
    assert record["native_admission"] is False


@pytest.mark.parametrize(
    "outcome", ["timeout", "truncated", "infrastructure_error", "incomplete", "syntax_error", "runtime_error"]
)
def test_review_uncompleted_failure_states_cannot_claim_oracle_completed(outcome):
    values = list(receipts())
    values[0] = values[0].model_copy(update={"outcome": outcome, "oracle_completed": True, "exit_code": 0})
    with pytest.raises(InvalidSubmissionError):
        report(tuple(values))


@pytest.mark.parametrize(
    "log",
    [
        [{"coding_evaluation": {"native_admission": True}}],
        [{"nested": [[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]}],
        [{"oversize": "x" * 100000}],
    ],
)
def test_review_caller_phase_log_cannot_spoof_or_exceed_bound(log):
    from amlas.arbiter.arbiter import grade

    with pytest.raises(InvalidSubmissionError):
        grade(spec=spec(), y_true=None, y_pred=report(), phase_log=log)


def test_review_empty_receipt_tuple_is_not_replaced_by_fixture():
    with pytest.raises(InvalidSubmissionError):
        report(())


@pytest.mark.parametrize("mode", ["all_pass", "all_fail", "negative", "base_only"])
def test_review_extreme_and_real_base_only_pairs(mode):
    values = []
    for receipt in receipts():
        passed = (
            mode == "all_pass"
            or (mode == "negative" and receipt.arm == "base")
            or (mode == "base_only" and receipt.arm == "base" and receipt.case_id == "HumanEval/0")
        )
        values.append(
            receipt.model_copy(
                update={
                    "outcome": "passed" if passed else "failed",
                    "oracle_completed": True,
                    "exit_code": 0 if passed else 1,
                }
            )
        )
    summary = report(tuple(values)).describe()
    assert summary["case_count"] == 164 and summary["execution_count"] == 328
    expected = {
        "all_pass": (164, 164, 0, 0),
        "all_fail": (0, 0, 0, 0),
        "negative": (164, 0, 164, 0),
        "base_only": (1, 0, 1, 0),
    }[mode]
    assert (
        summary["base_pass_count"],
        summary["tuned_pass_count"],
        summary["base_only"],
        summary["tuned_only"],
    ) == expected


def test_review_caller_logs_are_defensive_and_predictive_filtering_is_unchanged():
    from amlas.arbiter.arbiter import grade

    log = [{"caller_note": {"values": [1, 2]}}]
    result = grade(spec=spec(), y_true=None, y_pred=report(), phase_log=log)
    log[0]["caller_note"]["values"].append(3)
    assert result.phase_log[0]["caller_note"]["values"] == [1, 2]
    value = {
        "project_id": "predictive_fixture",
        "task_type": "regression",
        "target": "label",
        "golden_dataset": "./fixture.csv",
        "primary_metric": "rmse",
        "baseline": {"type": "fixture", "rmse": 1.0},
        "acceptance": {"primary_metric_delta_vs_baseline": ">= 0.0", "no_subgroup_regression": False},
    }
    predictive = SpecModel.model_validate(value)
    result = grade(spec=predictive, y_true=[1.0, float("nan"), 2.0], y_pred=[1.0, 9.0, 2.0])
    assert result.overall_score == 0 and result.n_graded == 2 and result.n_excluded_unlabelled == 1
    with pytest.raises(InvalidSubmissionError):
        grade(spec=predictive, y_true=[float("nan")], y_pred=[0])


def test_review_source_model_and_manifest_binding_drift_and_coding_ytrue_refuse():
    from amlas.arbiter.arbiter import grade

    for field in ("source_sha256", "tuned_model_sha256"):
        b = binding().model_copy(
            update={field: "f" * 64 if field == "source_sha256" else binding().base_model_sha256}
        )
        with pytest.raises(InvalidSubmissionError):
            report(receipts(b), b)
    values = list(receipts())
    bad = values[0].binding.model_copy(
        update={"case_manifest": values[0].binding.case_manifest.model_copy(update={"version_id": "foreign"})}
    )
    values[0] = values[0].model_copy(update={"binding": bad})
    with pytest.raises(InvalidSubmissionError):
        report(tuple(values))
    with pytest.raises(InvalidSubmissionError):
        grade(spec=spec(), y_true=["HumanEval/0"], y_pred=report())


def _summary_with_outcomes(changes):
    """Summary of the fixture population after the named arm and case index got another outcome."""
    values = []
    for receipt in receipts():
        index = int(receipt.case_id.split("/")[1])
        outcome = changes.get((receipt.arm, index))
        if outcome is None:
            values.append(receipt)
        elif outcome == "passed":
            values.append(
                receipt.model_copy(update={"outcome": "passed", "oracle_completed": True, "exit_code": 0})
            )
        elif outcome == "failed":
            values.append(
                receipt.model_copy(update={"outcome": "failed", "oracle_completed": True, "exit_code": 1})
            )
        else:
            values.append(
                receipt.model_copy(update={"outcome": outcome, "oracle_completed": False, "exit_code": None})
            )
    return report(tuple(values)).describe()


@pytest.mark.parametrize("outcome", ["timeout", "infrastructure_error"])
def test_no_verdict_outcomes_give_count_ranges_for_each_arm(outcome):
    assert set(module().NO_VERDICT_OUTCOMES) == {"timeout", "infrastructure_error"}
    summary = _summary_with_outcomes({("base", 0): outcome})
    assert summary["base_pass_count"] == 79 and summary["tuned_pass_count"] == 90
    assert summary["base_pass_count_range"] == [79, 80] and summary["tuned_pass_count_range"] == [90, 90]
    assert summary["delta_pass_count_range"] == [10, 11]
    # A case without a verdict still counts as not passed in the counts and the rates.
    assert summary["base_pass_at_1"] == 79 / 164 and summary["delta_pass_at_1"] == 11 / 164
    summary = _summary_with_outcomes({("tuned", 0): outcome})
    assert summary["base_pass_count_range"] == [80, 80] and summary["tuned_pass_count_range"] == [89, 90]
    assert summary["delta_pass_count_range"] == [9, 10]
    assert summary["tuned_pass_at_1"] == 89 / 164 and summary["delta_pass_at_1"] == 9 / 164


@pytest.mark.parametrize("outcome", ["runtime_error", "truncated", "incomplete", "syntax_error", "failed"])
def test_outcomes_caused_by_the_answer_do_not_widen_the_range(outcome):
    summary = _summary_with_outcomes({("base", 0): outcome, ("tuned", 1): outcome})
    assert summary["base_pass_count"] == 79 and summary["tuned_pass_count"] == 89
    assert summary["base_pass_count_range"] == [79, 79] and summary["tuned_pass_count_range"] == [89, 89]
    assert summary["delta_pass_count_range"] == [10, 10]


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, [10, 10]),
        (
            {
                **{("base", i): "passed" for i in range(80, 89)},
                ("base", 100): "timeout",
                ("base", 101): "timeout",
            },
            [-1, 1],
        ),
        ({**{("base", i): "passed" for i in range(80, 90)}, ("tuned", 100): "timeout"}, [0, 1]),
        ({**{("base", i): "passed" for i in range(80, 91)}, ("tuned", 100): "timeout"}, [-1, 0]),
        ({("base", i): "passed" for i in range(80, 90)}, [0, 0]),
        ({**{("base", i): "passed" for i in range(80, 95)}, ("tuned", 100): "timeout"}, [-5, -4]),
        # The same case is without a verdict in both arms.
        ({("base", 100): "infrastructure_error", ("tuned", 100): "timeout"}, [9, 11]),
    ],
)
def test_delta_range_for_ties_and_for_ranges_that_touch_zero(changes, expected):
    summary = _summary_with_outcomes(changes)
    assert summary["delta_pass_count_range"] == expected
    low = summary["tuned_pass_count_range"][0] - summary["base_pass_count_range"][1]
    high = summary["tuned_pass_count_range"][1] - summary["base_pass_count_range"][0]
    assert [low, high] == expected
    assert low <= summary["tuned_pass_count"] - summary["base_pass_count"] <= high


def test_ranges_equal_the_extremes_over_every_way_to_resolve_the_missing_verdicts():
    import itertools

    open_cases = [("base", 3), ("base", 100), ("tuned", 3), ("tuned", 120)]
    summary = _summary_with_outcomes(dict.fromkeys(open_cases, "timeout"))
    seen = {"base": set(), "tuned": set(), "delta": set()}
    for choice in itertools.product(("passed", "failed"), repeat=len(open_cases)):
        resolved = _summary_with_outcomes(dict(zip(open_cases, choice, strict=True)))
        seen["base"].add(resolved["base_pass_count"])
        seen["tuned"].add(resolved["tuned_pass_count"])
        seen["delta"].add(resolved["tuned_pass_count"] - resolved["base_pass_count"])
    assert summary["base_pass_count_range"] == [min(seen["base"]), max(seen["base"])]
    assert summary["tuned_pass_count_range"] == [min(seen["tuned"]), max(seen["tuned"])]
    assert summary["delta_pass_count_range"] == [min(seen["delta"]), max(seen["delta"])]
    assert len(seen["delta"]) == 5


def test_range_fields_keep_every_existing_summary_key_and_value():
    summary = report().describe()
    added = {
        "base_pass_count_range": [80, 80],
        "tuned_pass_count_range": [90, 90],
        "delta_pass_count_range": [10, 10],
    }
    before = {
        "profile_id": "openai.humaneval.164.pass_at_1.v1",
        "evaluation_id": "fixture-paired-evaluation",
        "case_count": 164,
        "execution_count": 328,
        "base_pass_count": 80,
        "tuned_pass_count": 90,
        "base_pass_at_1": 80 / 164,
        "tuned_pass_at_1": 90 / 164,
        "delta_pass_at_1": 10 / 164,
        "both_pass": 80,
        "base_only": 0,
        "tuned_only": 10,
        "both_fail": 74,
        "base_failure_counts": {"failed": 84},
        "tuned_failure_counts": {"failed": 74},
        "binding": binding().model_dump(),
        "provenance_status": "declared_consistency_checked_not_authenticated",
        "native_admission": False,
        "containment_verified": False,
        "grading_integrity_verified": False,
        "quality_success": False,
    }
    assert set(summary) == set(before) | set(added) | {"case_outcomes"}
    for key, value in {**before, **added}.items():
        assert summary[key] == value, key
    assert summary["case_outcomes"] == [
        {
            "case_id": f"HumanEval/{i}",
            "base": "passed" if i < 80 else "failed",
            "tuned": "passed" if i < 90 else "failed",
        }
        for i in range(164)
    ]
