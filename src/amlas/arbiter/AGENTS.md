# src/amlas/arbiter

## Purpose

Grade predictive results and fixed private coding reports deterministically without delegating acceptance to an LLM.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep grading deterministic; evaluate supplied labels/predictions against the selected specification, metric and thresholds.
- Register new grading tasks through `registry.py`; return the declared report/schema instead of inventing an accepted result.
- `arbiter.py:grade` — Score y_pred vs y_true per the spec's task_type and emit an ArbiterReport.
- `registry.py:register` — Register a grader. Duplicate task_type registrations raise ValueError.
- `registry.py:get_grader` — Return the registered grader for a task_type; raises KeyError if missing.
- `schema.py:InvalidSubmissionError` — Raised by a grader when the agent's submission is malformed.
- `schema.py:Grader` — Binding of task_type → scoring function.
- `coding.py` owns closed descriptive HumanEval164 execution bindings/receipts and exact paired2x164 aggregation. Its owning factory emits a private report only; all native/containment/grading-integrity/quality flags remain source-owned false in this pure slice. It performs no execution or origin authentication. Reports retain immutable canonical binding/summary bytes and declared consistency provenance; exact canonical spec SHA prevents cross-spec reuse. Coding-only phase logs are bounded/deep-copied and reserve the coding_evaluation key.
- `arbiter.py` branches fixed text_generation/pass_at_1 before predictive label filtering and requires the original private report, matching case manifest and paired baseline. Resolved_rate/public metrics JSON cannot enter this profile; incomplete/infrastructure outcomes retain the164 denominator and cannot qualify acceptance.
- `coding.py:NO_VERDICT_OUTCOMES` names `timeout` and `infrastructure_error`, the outcomes that end a case without a verdict. The summary carries `base_pass_count_range`, `tuned_pass_count_range` and `delta_pass_count_range`. An arm's range runs from its pass count to that count plus its cases without a verdict. The delta range runs from the tuned low end minus the base high end to the tuned high end minus the base low end. Pass counts, rates and the164 denominator keep their meaning: a case without a verdict counts as not passed, and no other outcome widens a range.
- Owned files: `__init__.py`, `arbiter.py`, `registry.py`, `schema.py`, `coding.py`.

- `coding_spec_bytes` normalizes the validated SpecModel JSON projection before canonical encoding; `coding_spec_sha256` hashes those exact bytes. Generation spec retention uses the same projection, so default numeric values keep one fingerprint after serialized validation/restoration. This source fix does not rewrite historical native artifacts or their original source-bound fingerprints.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `graders/AGENTS.md` | Register the classification, regression, detection and segmentation grading implementations. |
