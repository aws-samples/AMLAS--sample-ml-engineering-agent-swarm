# src/amlas/arbiter/graders

## Purpose

Own predictive grader registrations and the fixed private coding-report branch.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Maintain the interfaces and file formats owned by the modules listed below; parent contracts apply.
- `coding.py` grades only the closed private HumanEval164 pass_at_1 profile through Arbiter's early branch. Native admission, containment and semantic grading integrity remain unqualified, so acceptance is always false. Predictive registration and behavior remain unchanged.
- Owned files: `__init__.py`, `classification.py`, `coding.py`, `detection.py`, `regression.py`, `segmentation.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
