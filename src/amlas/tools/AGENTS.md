# src/amlas/tools

## Purpose

Expose agent-callable data, research, ML, deployment and operational tools.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- [Added 2026-11-05] `run_guardrail_check` honors global disable before registry resolution and returns truthful DISABLED evidence without evaluating content or claiming a safety pass. Existing held operations remain held.

- `submit_qwen_sft` tool text (added 2026-11-07) tells the model to call it alone in its message (a message
  that mixes it with other tools is refused whole), that repeating the same call returns the recorded submission and
  that a call with other inputs is refused by name. The preview rule in `flows/qwen_sft_flow.py` carries the same
  sentence. Enforcement is in `tui/chat/tool_binding_backend.py`; the text only lowers how often it fires.

- Coding preparation tool guidance identifies `preparation_request_ref` as the
  opaque SFT selector, equal to compatibility `request_ref`. The separate
  `preparation_ref` artifact descriptor and `recipe_id` are never SFT selectors.
  Public arguments, hidden state, caller policy and all native guards remain unchanged.

- inspect_huggingface_dataset exposes file_offset/declaration_offset metadata pagination only. Nonzero offsets require the resolved full revision; preserve whole metadata records and explicit partial coverage. These offsets cannot supply a native handle, policy or artifact authority.

- AWS readiness distinguishes permitted actions from selected configuration;
  configured references are not existence, ownership or authorization proof.
- `sagemaker.py` validates finite integer instance count, volume and runtime
  against native API bounds before pending/native writes. Preserve existing
  training/test channels, add validation only when supplied, and route explicit
  target_column through the existing AMLAS_TARGET environment contract.
- Generation inference/recommendations preserve text intent and report research
  gaps; they must not fall back to tabular training or invent baseline scores.
- Inference preserves explicit classification/numeric prediction/forecasting objectives when a
  language model is the implementation. Negated task exclusions are not positive
  objective signals; generation aliases share the canonical recommendation path.
- Spec rendering preserves supplied dataset, splits, constraints and budget.
- Spec rendering also carries an explicitly supplied generation block without
  manufacturing pins. validate_spec remains pure. save_spec's SDK-hidden typed
  campaign planning hook publishes original DATA without path resolution/disk
  fallback; missing/malformed capability refuses. Legacy calls preserve exact
  user text/order and existing disk behavior. Public schemas carry no hook,
  binding, policy, protected acknowledgement or ownership selector.
- `huggingface_data.py` exposes bounded metadata inspection and a thin pinned
  acquisition wrapper around portable job code; provider placement stays explicit.
  Discovery is read-only; acquisition persists an integrity/lineage artifact.
- Acquisition delegates to the gated backend flow, never directly to local core.
  Policy and typed acquisition store come only from SDK-injected ToolContext.
  The model schema exposes source pins/local output option/preview, without
  policy, native handles or resource JSON. `get_dataset_acquisition` takes only
  an opaque request ref and reads the current conversation's trusted record.
- `pathref.read_dataframe` selects CSV/Parquet/JSON/JSONL while remote bytes retain
  gated `open_bytes` and generic HTTP redirect denial. New formats have finite
  decoded-size/record caps. Nested coding summaries are bounded and JSON-safe.
  Parquet footer and decoded batches are checked before combination; nested
  preview text has an aggregate budget.
- JSON/JSONL records have a pre-DataFrame decoded-object/dense-cell estimate,
  1024-column width cap and existing row cap. These admission bounds do not
  establish parser isolation or deserialization trust.
- Scalar target distributions bound the Series before conversion to a JSON dict.
- Public `load_dataset`, `describe_dataset` and `split_dataset` deny before raw
  caller-process parsing and direct agents to managed inspection/preparation.
  Explicit `*_local` helpers remain unregistered offline APIs; never bind them
  to an agent or use them as a fallback after a managed failure.
- `dataset_inspection.py` accepts an opaque acquisition ref, preview and bounded
  target text only; hidden typed state/policy supply source and authority.
  Submission returns an opaque inspection ref; polling reads the retained job.
- `run_ml_capability` accepts only a typed caller policy from SDK-injected
  ToolContext; hidden context is excluded from the model tool schema. Prompt
  text and params JSON do not authorize execution. Other tools keep their guards.
- `agent_tools.py:data_verify_tool` — Produce a compact data profile (shape, dtypes, NaN%, sample) for agent consumption.
- `agent_tools.py:leakage_check_tool` — Run leakage checks against train/test frames.
- `aws_control.py:ensure_s3_prefix` — Ensure an S3 lake prefix exists (idempotent), so a job can write there.
- `aws_control.py:ensure_glue_database` — Ensure a Glue Data Catalog database exists (idempotent).
- `backend_neutral.py:submit_training_run` — Submit a training run for ``algo`` on ``train_uri`` (T-TRAIN-01, backend-neutral).
- `code_interpreter.py:register_staged_file` — Record a ``{sandbox_target: local_source}`` pair so FIX-22 can re-stage it on reprovision.
- `code_interpreter.py:sandbox_egress_enabled` — True only if sandbox network egress is EXPLICITLY enabled (SEC-A15; default OFF).
- `coding_preparation.py` exposes canonical preparation by opaque acquisition
  refs and a read-only opaque preparation poll. SDK-injected typed policy/store
  are hidden from model schema; no role/image/resource/native-handle JSON is
  accepted. Source/operator/native control data is never injected into prompts.
  Root binds the tools/store into normal interfaces after separate integration.

- Normal `run_ml_capability` routes preparation through a separate trusted hidden
  store channel; public JSON params cannot provide a store, catalog or policy.
  Plans/introspection/exception results remain JSON-safe and exclude private values.
- `model_weight_tools.py` adds acquire/reuse_verified mode and an opaque pack_ref
  selector to catalog/version/preview acquisition and opaque
  polling only. Caller policy and typed model state come from hidden SDK context;
  resource/environment/native-handle JSON cannot authorize model downloads.
  It delegates to the managed flow and has no raw downloader/local fallback.
  Tool registration and caller-state forwarding require separate integration.
- Owned files: `__init__.py`, `agent_tools.py`, `aws_control.py`, `backend_neutral.py`, `code_interpreter.py`, `dashboard_tools.py`, `data_etl_neutral.py`, `data_quality_tools.py`, `data_tools.py`, `data_verify.py`, `decision_tools.py`, `deploy_neutral.py`, `dispatch_preflight.py`, `evaluation_tools.py`, `experiment_tools.py`, `fe_advanced_tools.py`, `fe_tools.py`, `fs_tools.py`, `git_neutral.py`, `git_tools.py`, `governance_m8_neutral.py`, `governance_neutral.py`, `hpo_tools.py`, `leakage_check.py`, `leakage_columns.py`, `local_tools.py`, `ml_neutral.py`, `monitor_neutral.py`, `orchestration_tools.py`, `pathref.py`, `pii_tools.py`, `research_neutral.py`, `research_tools.py`, `retrieval_neutral.py`, `sagemaker.py`, `sagemaker_source_bundle.py`, `scheduler_tools.py`, `spec_generation_tools.py`, `spec_guidance.py`, `spec_heuristics.py`, `spec_tools.py`, `training_jobs.py`, `training_tools.py`, `uri_resolution.py`, `coding_preparation.py`, `model_weight_tools.py`.

- list_model_weight_catalog returns fresh source-approved expected-byte model
  descriptors, including catalog/version/revision/files/license/architecture.
  Acquired/trusted/ready/native_verified remain false; discovery grants no
  admission and makes no network/framework/model-loading calls. Hidden typed
  model state may append matching verified_packs catalog/version/pack_ref DATA;
  absent context preserves the exact legacy catalog and the public schema stays
  empty. Candidate selectors expose no producer handles, resources or policy.


- `qwen_sft.py` exposes submit_qwen_sft(model_request_ref,preparation_request_ref,recipe_id,plan_only), get_qwen_sft(opaque request_ref), reload_qwen_sft(original train request_ref,plan_only). Typed policy and per-conversation store arrive only through SDK ToolContext invocation_state; hidden state/catalog/resource selectors are absent from model schema and results. Removing policy denies observations/effects even while original handles remain. No arbitrary role/image/channel/source/runtime or approval JSON parameters exist.

- prepare_coding_dataset accepts optional opaque case_request_ref in addition to original acquisition refs. Hidden same-conversation private stores own immutable exclusion selection; callers cannot provide case handles, catalogs or oracle paths.

- `coding_generation.py` exposes bounded planning-data `bind_coding_generation_spec(yaml_text)` and opaque `submit_coding_generation`/`get_coding_generation`. The fixed recipe, resources, policy, store and native identity never enter model schemas. Fresh typed policy/store arrive only in SDK invocation_state. A spec reference conveys planning data, never execution approval; generated outputs retain quality/native admission false.

- `coding_pair.py` provides `resolve_coding_evaluation_spec`, `submit_coding_pair` and `get_coding_pair`. Only opaque refs and previews are public; role/image/resources, oracle code, budgets, policies and state cannot enter model schemas. Spec resolution precedes both generation arms. The result of `get_coding_pair` for an evaluated pair carries `result_statement` and `next_step` as result data; the input schemas are unchanged.

- `list_coding_preparation_profiles` is a read-only pre-acquisition source-shape query. Only optional bounded repo/revision/filename are public; hidden exact conversation state supplies held catalog DATA. It reports configured/unconfigured/unknown and bounded missing selector names without values or AWS calls. Matching metadata does not verify bytes or native readiness; fixed unsupported sources have no fallback or replacement recommendation.

- Preparation discovery describes the existing verified original-case exclusion route separately from worker resources and the global fallback catalog. Public optional source query schema remains unchanged; it cannot provide refs/policy or claim a case exists/ready. prepare_coding_dataset still verifies genuine same-conversation case_request_ref outputs before privately replacing exclusions.

- The registered spec generator preserves legacy objective inference and emits
  coding spec_text only from explicit target/data/acceptance/budget facts and the
  existing fixed source model catalog plus SpecModel/GenerationPlan. Missing
  context remains finite research-needed DATA; no score, target, cap, measurement
  or artifact reference is guessed. Planning keeps native/quality/execution false.
  Private coding bind still owns exact profile/canonical bytes and later original
  producer/approval qualification; generated text is never an effect grant.

### Closed coding planning DATA

(Added 2026-11-07) The planning_data schema types `acceptance.primary_metric_delta_vs_baseline` as a required
string and requires both `id` and `version` in `model_catalog`, and the tool description states both rules: the real-model
test harness saw 14 of 15 runs fail the first validation on a numeric delta or a missing catalog id. Unsupported keys stay refused.

`read_file` returns at most `READ_RESULT_BYTES` (10 KB) of content per call, ending a page on a line boundary and never
inside a UTF-8 character. `next_offset` is null when the content is complete; otherwise the model calls again with
`offset=next_offset`. `content_offset`, `content_returned_bytes` and `content_total_bytes` describe the page and
`truncated` is also true while `next_offset` is set. The 256 KB read cap is unchanged. The binary sniff decodes with
`final=False`, so a multi-byte character cut by the 8192-byte sniff window no longer marks a text file as binary. Why: results of 13.9 KB (test harness) and
22.7 KB (a deployed run) were the second largest conversation contributor after research.

`generate_spec_from_description` accepts only source-catalog model selection, target,
golden dataset, explicit acceptance and existing ComputeBudget planning DATA. Supplied
ceilings are preserved; omitted budget fields are listed as source_schema_defaults.
An explicit overall unbounded declaration sets all ceilings to zero and rejects
contradictory finite values. Text-derived unlimited budgets require an affirmative
same-clause declaration; negation remains research-needed without explicit typed
budget facts. No pins, artifact refs, approval or execution fields
are accepted, and generated spec text never grants admission or quality success.

For inferred text_generation, data_path and golden_dataset stay planning DATA.
The generator skips caller-process column/CSV parsing for those paths; legacy
explicit tabular column validation remains unchanged. Managed producer tools own
any actual coding-dataset inspection or preparation.

- `bind_coding_generation_spec` retains its private same-session owner and fixed
  model/tokenizer/full-recipe predicates. Known refusals return closed schema,
  reason and field-path DATA without reflecting inputs, errors or values; missing
  ownership stays denied and unexpected exceptions remain opaque. No public
  signature, plan-only bypass, source read or execution grant is introduced.

(Added 2026-11-07) Specification stage of the real-model test harness. In nine of 51 harness runs the
model ended before PAIR in the specification stage. The raw tool records show why. 18 of 19
`generate_spec_from_description` results were the classification stub with no `spec_text`, although all 19 calls carried
`planning_data`: the keyword test knew neither a versioned name such as `Qwen3.5-9B` (`\bqwen\b` does not match it) nor the
pass@1 metric. The model then wrote YAML by hand and 24 of 26 `validate_spec` results were invalid, 11 of them on a rule
that named no path. Changes, all additive and none relaxes a validator:

- Presence of `planning_data` (even an empty object) selects the coding task; the result carries `task_type_source`
  (`planning_data` or `description`). `_infer_task_type` also recognises a versioned `qwen` name, the pass@1 metric and
  `humaneval`; explicit classification or regression wording still wins. `_extract_metric` and `_extract_threshold` take an
  optional `task_type`. Without `planning_data` and without those words the legacy stub is unchanged.
- A schema-valid coding result carries `next_step`: bind `spec_text` unchanged, leave `generation.preparation` and
  `generation.evaluation` null (the product adds the case manifest in `resolve_coding_evaluation_spec`, and a different
  hand-written reference is refused). A research-needed result adds `needs_research_detail` with, per missing fact, the
  `planning_data_path`, the expected form and an example the schema accepts. `needs_research` keeps its names.
- `spec_guidance.py` (strands-free) turns the structured `SpecLoadError.issues` into `error_details`
  (`path`, `problem`, `expected`, `example`, at most `MAX_DETAILS` = 8, the rest counted in `error_details_omitted`).
  It never reads the rendered pydantic text, input values or error context; a path segment is shown only when it is a
  schema field name or one of four misplaced keys the harness runs wrote, every other key is `(unknown key)`. `validate_spec`
  adds `error_details` and `next_step` (generator first) to an invalid result and `next_step` (bind) to a valid coding
  spec; `errors` keeps its text. `list_example_specs` skips the directory guide and says the examples are classification.
- `bind_coding_generation_spec`: the closed `validation` record is unchanged. `parse_or_schema_invalid` and
  `model_pin_required` name the generator first; a successful bind adds `next_step` (call `resolve_coding_evaluation_spec`;
  this result is not an evaluation). Tool descriptions gain one additive sentence each.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `data_quality/AGENTS.md` | Profile data quality through the catalog, patterns, missingness analysis and scorecard. |
| `imputation/AGENTS.md` | Select and evaluate numeric, categorical, temporal and LLM-assisted missing-value methods. |
| `lake/AGENTS.md` | Resolve partitioned medallion datasets and retain the lake read/write boundary. |
| `mlops/AGENTS.md` | Expose managed ML registry, pipeline, feature-store, explainability and monitoring operations. |
| `research/AGENTS.md` | Aggregate research-provider results and bounded URL retrieval with explicit source/coverage information. |
| `synthetic/AGENTS.md` | Generate or augment rows and evaluate fidelity, utility and privacy evidence. |

- `coding_cases.py` provides pure fixed catalog discovery, opaque acquisition-to-case freeze and same-session polling. SDK context/store/policy remain hidden; public schemas include no resources, handles, oracle access or approval fields. Public results omit ORACLES/EXCLUSIONS locations/content. Normal registry/delegation/startup binding is a separate Root integration seam.

- Deployment package handoff is an exact private frozen source-owned object in a
  live same-process lexical scope. Expired/copied/foreign objects, changed backend
  configuration and public package DTOs refuse before endpoint creation. Public
  tool schemas stay unchanged; package results remain JSON descriptions. Existing
  flow approval precedes packaging; handoff ownership grants no new permission.

- Post-Create package closure preserves accepted endpoint DATA in its private
  scope and exposes only Unknown/outcome_unknown, never a new creation grant or
  private handle. Final public projection rechecks original scope/configuration.

- Original managed package configuration is captured before native model creation
  and freshly compared with the held/current selected adapter afterward. Drift
  during Create cannot be blessed by a later snapshot or followed by deployment.
