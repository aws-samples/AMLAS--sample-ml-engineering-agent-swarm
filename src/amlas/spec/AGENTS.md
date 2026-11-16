# src/amlas/spec

## Purpose

Parse task specifications and acceptance configuration into validated models.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Validate inputs through `schema.py` and `loader.py`; report schema/acceptance errors rather than silently repairing intent.
- `text_generation` is a valid task type. Generation spec validation does not
  establish a supported trainer or grader. Unmeasured baseline scores remain null.
- `generation.py:GenerationPlan` is strict descriptive coding-SFT planning data:
  bounded repository/full-commit pins, explicit catalog selectors, five-field JSON
  manifest references, composite agent/evaluator pins and fixed coding quality.
  Schema-valid references or measured-state declarations never establish native
  readiness, trust, ownership or execution. No lookup, dispatch or grant fields exist.
- Optional generation blocks are omitted when absent/None, preserving legacy
  serialization through the supported wrap serializer. Generation task/metric
  contradictions and extra authority fields are rejected; nonmeasured scores stay
  null. Measured declarations need a report reference and finite non-bool score in
  [0,1], without asserting that the referenced evidence exists or is trusted.
- `loader.py:SpecLoadError` — Raised when a task_spec.md cannot be parsed or validated.
- `loader.py:load_spec_from_text` — Parse a spec string. Raises SpecLoadError on any parse or validation failure.
- Owned files: `__init__.py`, `generation.py`, `loader.py`, `schema.py`.

- Coding quality explicitly supports resolved_rate or pass_at_1; resolved_rate stays the default. The declared primary/baseline metric must match exactly. Functional planning remains descriptive, never evidence of execution or quality.

(Added 2026-11-07) `SpecLoadError` carries `issues` (pydantic `(loc, type)` pairs, no input values or messages) and `kind`
(`schema` or `front_matter`) so `tools/spec_guidance.py` can explain a failure without parsing text. The generation
declaration rule (task_type text_generation and primary_metric equal to `generation.quality.metric`) raises the closed
error type `generation_declaration_mismatch` with a message that names both fields and the fixed-task fix; it was a
pathless `value_error` in 11 of 24 invalid `validate_spec` results of the real-model test harness. Acceptance is unchanged.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
