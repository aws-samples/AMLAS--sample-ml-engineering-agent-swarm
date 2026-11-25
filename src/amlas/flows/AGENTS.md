# src/amlas/flows

## Purpose

Declare and register reusable ML lifecycle flows with execution and approval metadata.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Registered flow identifiers and spend/irreversibility metadata are dispatch contracts.
- Use typed backend ports; report execution evidence and grading separately from a workflow status.
- `dataset_acquisition_flow.py` gates acquisition using the data collection
  domain action and an acquisition-specific plan, then dispatches the selected
  DataAcquisitionBackend. Only caller-owned typed state supplies session context.
  Submitted native handles are retained before later observations.
- Solution survey derives omitted text modality for generation tasks and rejects
  an explicit contradictory modality before catalog calls.
- Fixed inspection uses the collection approval domain with unknown cost.
  Capped policies deny; trusted uncapped approval and matching native image
  admission are required. Plan-only inspection never stages bytes or creates jobs.
- `__init__.py:Flow` — A registered flow descriptor (``flow_id`` → metadata).
- `__init__.py:register_flow` — Register ``flow`` under its ``flow_id`` (idempotent — last write wins).
- `_util.py:raw` — Return the undecorated callable behind a Strands ``@tool`` (or the fn itself).
- `baseline_train_flow.py:BaselineTrainFlowResult` — The invariant output contract of one ``baseline_train_flow`` run.
- Baseline results distinguish `waiting` for nonterminal or unreadable native status
  from `failed` for observed terminal failure or submit failure. Preserve submitted
  job identity and never advise resubmission. Native success without complete valid
  grading is `unassessed`, with no artifact retrieval, tracking or registration.
  Acceptance requires an explicit actual bool (including `passes_acceptance`) or
  known verdict; every supplied decision field must be valid and agree. Canonical
  false rejects. Waiting/unassessed are result states, not durable resume or a
  generation grader. Caller envelopes may complete while the nested flow is waiting.
- `coding_preparation_flow.py` requests fresh typed caller approval with unknown
  cost for `data_preparation_flow` before source reads, staging or Processing.
  Missing/model-authored/capped-unknown policy denies; preview has no effects.
  Only the canonical JSONL and the listed CommitPackFT profile ids are supported,
  each with its advertised policy version or the tool default version 1 (added
  2026-11-06, changed the same day). A profile argument names a
  listed profile and is never authority: the private sources decide which
  preparation runs, and the CommitPackFT selector is refused unless every source
  is the recognized CommitPack repo. The plan_only preview reports an unsupported
  profile instead of planned. Native accepted is pending, never
  prepared/trained; no local or inferred collector fallback.

- The SFT plan states `submission_rule` and `lifecycle_recipe` full_v1, and `recipe_note` for fit_v1
  (added 2026-11-06): a normal run keeps one original training call, and a second effectful
  submit_qwen_sft ends the run, so the preview is where the agent chooses the recipe.
- Every managed flow result and `plan_only` preview carries `execution_provider` (added 2026-11-06)
  from `control_plane/provider_disclosure.py`: the flow's own SageMaker operations decide whether
  it reads controlled or unverified. It is informational; no approval, allowlist or ownership
  check reads it.

- `data_preparation_flow` is spend-incurring and runs on CodingPreparationBackend.
  The registry's private executor override avoids double gating; unknown quote None
  remains unknown. Configuration/refusal means executed=False; accepted native handles
  remain executed even if later observation fails. Plan mode observes no sources.
- `model_weight_acquisition_flow.py` requires caller-owned typed policy/model
  state for execution, preserves unknown cost and existing processing approval,
  and denies capped unknown quotes. Preview has no native side effects. Retain
  opaque accepted/ambiguous handles; repeated catalog requests observe the same
  record. No managed failure uses a local download fallback or model-owned grants.
  Registration/configuration/native qualification remain separate integration.
  Reuse_verified delegates only to the private store after the same fresh
  approval and under the existing submission lock; preview resolves no backend.
  Missing provider refuses without acquisition fallback. Returned current-owned
  model refs and unknown copy outcomes retain their closed store projections.
- Owned files: `__init__.py`, `_util.py`, `baseline_train_flow.py`, `business_framing_flow.py`, `data_collection_flow.py`, `data_quality_flow.py`, `deploy_flow.py`, `devops_git_flow.py`, `evaluation_flow.py`, `feature_engineering_flow.py`, `hpo_flow.py`, `improve_retrain_flow.py`, `imputation_flow.py`, `monitor_flow.py`, `readiness_flow.py`, `research_intel_flow.py`, `research_scan_flow.py`, `solution_survey_flow.py`, `coding_preparation_flow.py`, `model_weight_acquisition_flow.py`.

- Model acquisition resolves the dedicated managed-only port at dispatch,
  then requires a pure configured selector tuple before SDK access. Unknown
  selectors and unconfigured tuples refuse; preview/policy/store/idempotency
  and ambiguous accepted-handle retention remain unchanged.


- `qwen_sft_flow.py` gates fixed train/observe/reload with fresh typed ApprovalPolicy and hidden SessionSftStore. Unknown GPU/staging costs stay None; capped policies/ASK deny before source observations or SDK. Preview has no effects. Fit/full stages and full_training_complete are explicit; checkpoint/reload never grant quality approval or registration. Generic/tabular baseline flow remains independent.

- Verified SFT status exposes every scalar metric plus a bounded history summary
  with exact train/validation counts, final rows and earliest minimum-loss rows.
  Complete histories remain in the unchanged typed, version/hash/size-pinned
  training_ref; missing full evidence refuses before public projection. Private
  receipt/native/input/checkpoint validation and all state/quality/ref fields remain
  intact. Projection copies DATA, adds no reads/effects or larger INPUT limit, and
  never gives a summary authority to train, reload, generate or approve quality.

- No-effect plans show source-owned optimizer32/1024, full1000unique examples/1Munique supervised/1Mprocessed token floors, and quality_successfalse.
- `qwen_sft_flow` is registered as spend-incurring over QwenSftBackend, distinct
  from generic training/baseline flows. Discovery describes fit32/full1024 and
  separate reload without quality promotion. Its public implementation retains
  unknown cost None and requires fresh typed caller policy before effects.

- Preparation accepts optional opaque case_request_ref after fresh caller approval. Preview/denied policies observe no cases; no public oracle/exclusion descriptors become authority.

- `coding_generation_flow` is spend-incurring over CodingGenerationBackend. It requires fresh typed caller approval with unknown cost before managed source reads/submission/polling; plan-only is effect-free. Native accepted, generated bytes and execution quality remain separate.

- `coding_pair_flow` gates private case resolution and managed paired submit/poll on a fresh typed caller policy. Unknown cost stays None; capped/ASK approval cannot reach source reads or SDK effects. The fixed164-case/328-receipt plan and unchanged10-second case budget grant no execution quality.
  The observe result handed to the agent keeps the four core refs (COMMITTED.json, RECEIPTS.jsonl,
  EVALUATION.json, INVOCATIONS_MANIFEST.jsonl) and `invocation_records_view` (bounded_v1, the verified record
  count, evidence field `files.INVOCATIONS_MANIFEST.jsonl`) in place of one ref per invocation record (added
  2026-11-12: 13,146 refs were about 6.1 MB in the conversation). The adapter and `store.observe`
  still read, verify and return every record; results without a `files` dict pass through unchanged.
  `result_statement` and `next_step` of an evaluated pair reach the agent unchanged; with the three range keys
  and the longest statement the tool result stays within the 24,576-byte bound of the projection test.

- Normal model flow fences current campaign ownership before legacy resolution
  or alias reuse. Only the declared private model attempt error class supplies
  post-dispatch outcome DATA; unqualified outcomes report `outcome_unknown`,
  `executed=None` and no request/job/ARN. Known pre-dispatch refusal remains false;
  an arbitrary error after adapter invocation is unresolved, never proof of no
  Create. No error grants retry, approval, model readiness or native admission.

- Model startup selector failures expose only fixed missing/invalid error_code, configuration_keys and detail. Empty/partial tuples name the missing selected ENV keys; malformed or inconsistent tuples echo no values. Registry precedence, fresh caller approval and preview remain first; arbitrary adapter errors preserve unresolved outcomes and cannot become configuration or no-dispatch claims. No error supplies selectors or native admission.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.

- `coding_case_freeze_flow.py` gates freeze/observe on fresh typed SDK ApprovalPolicy for `coding_case_freeze_flow`; unknown-cost capped policies deny. Plan-only has no source/SDK effect. Data frozen does not mean cases executed, quality success or native admission. The normal flow registry declares its dedicated CodingCaseFreezeBackend and spend-incurring, reversible M6 metadata; discovery does not grant execution or native admission.

- An enabled campaign model flow uses only its typed private bridge after normal
  fresh caller approval; missing private services refuses before legacy dispatch.
  Journal original ref/job survives cold reuse; executed/reused/unknown describe
  actual source dispatch separately. Disabled/no-entry legacy behavior stays.

- Model acquisition returns the original bounded submission diagnostic separately
  from status/accepted; normal observation returns its own closed read diagnostic.
  Existing approval, original-store/ref retention and no-replay semantics remain.
  Unknown diagnostics supply no native cause, admission or warm ownership grant.

- Generic deploy flow retains one original private package through the approved
  package-to-deploy scope. Creating/unreadable original endpoints remain waiting;
  native failure remains failed. Smoke, release, monitoring and edge work require
  confirmed InService. Unknown observations preserve accepted identity with no
  creation replay. A no-ARN monitor is degraded/wiredFalse; monitor identity is
  configuration DATA, never model quality or native admission.

- Monitoring wiring requires a closed native schedule ARN bound to its returned
  schedule name and the original endpoint partition/service/Region/account. Empty,
  malformed, foreign-resource or mismatched references remain degraded; identity
  validation is DATA and supplies no IAM, configuration or quality admission.
