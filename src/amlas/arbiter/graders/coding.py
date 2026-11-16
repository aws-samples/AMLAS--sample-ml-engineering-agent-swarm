"""Fixed paired HumanEval report grading; source-only reports cannot promote."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from ..coding import coding_spec_sha256, require_private_report
from ..schema import ArbiterReport, InvalidSubmissionError


def _coding_phase_log(value: object) -> list[dict]:
    if value is None:
        return []
    if type(value) is not list or len(value) > 64 or any(type(row) is not dict for row in value):
        raise InvalidSubmissionError("bounded coding phase log required")
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 8 or count > 4096:
            raise InvalidSubmissionError("coding phase log structural bound exceeded")
        if type(item) not in (dict, list, str, int, float, bool, type(None)):
            raise InvalidSubmissionError("coding phase log primitive values required")
        if type(item) is str and len(item) > 4096:
            raise InvalidSubmissionError("coding phase log string bound exceeded")
        if type(item) is float and not math.isfinite(item):
            raise InvalidSubmissionError("coding phase log finite values required")
        if type(item) is dict:
            if any(type(key) is not str or len(key) > 128 or key == "coding_evaluation" for key in item):
                raise InvalidSubmissionError("coding phase log reserved key or invalid key")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > 16384:
        raise InvalidSubmissionError("coding phase log byte bound exceeded")
    return json.loads(raw)


def grade_coding(
    *,
    spec: Any,
    y_true: Any,
    y_pred: Any,
    baseline_score: float | None = None,
    phase_log: list[dict[str, Any]] | None = None,
    llm_total_tokens: int = 0,
) -> ArbiterReport:
    start = time.monotonic()
    report = require_private_report(y_pred)
    generation = spec.generation
    if (
        spec.task_type != "text_generation"
        or spec.primary_metric != "pass_at_1"
        or generation is None
        or generation.quality.metric != "pass_at_1"
        or generation.quality.comparison != "paired_same_cases"
        or generation.quality.coverage != "all_required_cases"
        or generation.evaluation.case_manifest_ref != report.binding.case_manifest
        or report.binding.spec_sha256 != coding_spec_sha256(spec)
        or y_true is not None
    ):
        raise InvalidSubmissionError("fixed HumanEval pass_at_1 spec/profile/manifest required")
    values = report.describe()
    caller_log = _coding_phase_log(phase_log)
    if baseline_score is not None and baseline_score != values["base_pass_at_1"]:
        raise InvalidSubmissionError("caller baseline differs from private paired execution report")
    declared = spec.baseline.model_dump().get("pass_at_1")
    if declared is not None and declared != values["base_pass_at_1"]:
        raise InvalidSubmissionError("declared baseline differs from private paired report")
    return ArbiterReport(
        task_type="text_generation",
        primary_metric="pass_at_1",
        overall_score=values["tuned_pass_at_1"],
        baseline_score=values["base_pass_at_1"],
        delta_vs_baseline=values["delta_pass_at_1"],
        direction="maximize",
        passes_acceptance=False,
        acceptance_reason="unqualified native containment and grading integrity; descriptive paired arithmetic only",
        slices={},
        wall_clock_seconds=time.monotonic() - start,
        llm_total_tokens=llm_total_tokens,
        phase_log=[*caller_log, {"coding_evaluation": values}],
        n_graded=164,
        n_excluded_unlabelled=0,
    )
