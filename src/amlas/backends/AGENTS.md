# src/amlas/backends

## Purpose

Define typed backend ports, domain handles and adapter resolution at the workflow-to-compute boundary.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Flows depend on `ports.py` protocols and `types.py` domain handles; managed SDK details belong in adapters.
- `registry.py` resolves steering pins, explicit selectors and automatic choice; explicit unavailable choices must fail rather than silently switch.
- `ports.py:DataAcquisitionBackend` separates pinned source submission, native
  job status and verified artifact retrieval. A completed native job alone is
  insufficient to report acquired data.
- `DataInspectionBackend` accepts a private frozen source snapshot and returns
  accepted handles plus strictly verified bounded reports. It has no local adapter.
  `DatasetArtifactRef` retains five versioned-byte fields; provenance is separate.
- `hooks.py:HookContext` — The context a hook matcher/action inspects.
- `types.py:JobState` — Backend-neutral lifecycle state of an async job (training/processing/etc.).
- Owned files: `campaign_state.py`, `__init__.py`, `hooks.py`, `ports.py`, `registry.py`, `types.py`.

- `CodingPreparationBackend` is the managed-only private preparation port; selector
  `DATA_PREPARATION_BACKEND=processing`. Its lazy constructor makes no SDK call.
  Explicit unsupported/local selection refuses, and no local parser fallback is bound.

- `ModelWeightAcquisitionBackend` is a dedicated fixed catalog/version/session
  port, registered managed-only by model_registration. Its selector is
  MODEL_WEIGHT_ACQUISITION_BACKEND; unavailable choices refuse without fallback.

- `QwenSftBackend` is registered managed-only with QWEN_SFT_BACKEND=sagemaker; its fixed SFT/reload requests never have a local fallback or construct SDK clients at startup.

- `campaign_state.py:CampaignStateBackend` is the separate unregistered
  server-only port for original-job reservation, conditional Create-attempt and
  submission dispositions, typed reload, and read-only original-job observation.
  It supplies no training Create method, policy persistence or public JSON input.

- VerifiedPreparationSource is a frozen data-only DTO in types.py; preparation_state re-exports the same class for compatibility. It carries provenance, not grants or native verification authority.

- The private campaign port additionally exposes one conditional
  `attach_submission_snapshot`; it supplies no Create method. Original intent
  remains immutable, while exact original staged evidence is attached once
  before native Create. Existing selected-table/authentication boundaries apply.

- `CodingGenerationBackend` is managed-only with `CODING_GENERATION_BACKEND=sagemaker`. It accepts private original model/case/checkpoint sources, stages only public prompts and an optional adapter, and never loads a model or executes candidates in the caller. Registration grants no native admission.

- `PairedCodingBackend` is managed-only under `CODING_PAIR_BACKEND=sagemaker`; it dispatches fixed CPU parent evaluation from private original case/base/tuned/spec references with no local execution fallback.

- `types.py:is_capacity_rejection` and `CAPACITY_REJECTION_CODE` (added 2026-11-10): a handle is a capacity
  rejection only when `extra["submission_state"] == "rejected"` and `extra["rejection_code"] == "ResourceLimitExceeded"`.
  Adapters record `rejection_code` for every definite rejection; stores use the predicate to let a later identical submit
  create again.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `adapters/AGENTS.md` | Implement and register concrete local and managed-service backend adapters. |

- `coding_case_port.py` is the dedicated data-only case-freeze protocol. Shared registry/config/startup wiring is separate; this port adds no native image, resource or oracle-access authority.
