# src/amlas/backends/adapters

## Purpose

Implement and register concrete local and managed-service backend adapters.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- [Added 2026-11-05] Global guardrail disable returns before direct in-process classification or Bedrock client access, including explicit backend/policy ids. Enabled behavior remains unchanged; a disabled check does not imply a safety pass.

<!-- model-loading-progress-contract:begin -->
- MODEL observation and exact artifact verification emit optional closed normal
  loading warnings through a lazy helper. Native wait is reported only from the
  original validated PENDING/RUNNING result; verified count increments only after
  each unchanged payload check, and ready follows the existing final binding and
  proof checks. Verification refusal retains the last successful count. Ordinary
  import/logger failures preserve original outcomes, while process control
  propagates. No extra SDK call, ownership, permission, retry or history change
  follows from these diagnostics.
<!-- model-loading-progress-contract:end -->

- Coding generation accepts the exact original nine-file acquisition inventory
  plus its verified provenance refs when co-located in the same prefix. Retain
  strict listing/cardinality, pinned version/hash and latest-version checks before
  staging. Reuse the existing version-pinned S3 copy helper to isolate only the
  nine model files under the native base_model input; acquisition receipts never
  enter that channel. Unknown extra files refuse before staging or Create. The
  training worker, input hashes, original base refs, image and native guards stay
  bound to their existing contracts.

- Qwen SFT `submit` accepts private optional `_causal_parent_ref` DATA only with
  the existing original journal. A present value reaches `build_sft_intent`
  validation/digest before reservation/staging/Create; None adds no constructor
  argument or intent field. Public tools, training/reload guards and staging
  behavior remain unchanged; the generic campaign backend reuses the same closed
  codec for schema6 without new resources, permissions or descriptor adoption.

- Normal durable training forwards the exact freshly resolved source alias pair
  through its existing pre-submit callback and adapter into the original intent
  before reservation/staging/Create. Other callers omit it and retain legacy
  identity. A pair without the existing journal refuses; aliases cannot confer
  native permission or replace source, request, artifact and policy validation.

- Import registration belongs in `__init__.py`; preserve adapter names and declared port conformance.
- Propagate service failures and retain accepted native handles; configuration/Region comes through the shared selectors.
- `training.py` preserves declared native image/hardware/storage/runtime controls
  separately from flat or nested model hyperparameters. Explicit training,
  validation and target arguments take precedence without mutating caller data.
  An explicit budget runtime is a cap on an explicit resource runtime; no USD
  conversion is inferred. Role/session/table selection and accepted-handle
  recovery remain unchanged.
- Native acquisition uses the fixed source-owned ceiling subclass and schema2
  discriminator/14-field proof. Its handle binds expected profile/source/image;
  caller resources cannot select a profile. Guest privilege fields are telemetry,
  never admission or attestation. The strict schema1 class remains compatible.
- `dataset_acquisition.py` registers Processing as the acquisition default and
  local as an explicit offline adapter. Submission follows the collection flow's
  approval gate; no service failure falls back to a local download.
- Acquisition requires a configured execution role, selected session bucket,
  image digest/source commit, deployment UUID and session correlation. Validate
  Region and bucket owner before creation; apply ownership tags atomically.
- Native Completed plus bound receipt/commit/lineage hashes and expected S3 object
  length prove acquisition. Report container-computed SHA separately from an S3
  FULL_OBJECT checksum; acquisition never establishes deserialization trust,
  preparation or training.
- Returned artifact paths are S3 URIs; never return a Processing container's local
  path to a remote caller or use an unconfined manifest path.
- Fixed inspection requires the published image/source pair and an image-matching
  native admission digest before writes. Stage exact-version hashed bytes into one
  unique input prefix; always request/read back network isolation. Accept only
  versioned bounded SUMMARY.json/INSPECTED.json with matching native provenance.
- Inspection native entrypoint is fixed `/usr/local/bin/amlas-processing-init`
  profile `inspection`; Python/direct-worker overrides cannot bypass initialization.
  Existing version/hash/job/session/image/source/target binding pairs remain exact.
  Parent acceptance also requires the actual permanent-drop receipt proof. The
  retained `ProcessingDatasetInspectionBackend` expects strict schema1/Bnd0;
  fixed `SageMakerNativeDatasetInspectionBackend` is the production Processing
  registration and expects schema2/`sagemaker_native_ceiling_v1` only. Expected
  profile belongs to source, not constructor/env/model input; accepted handle and
  native request/source/image/role/version hashes remain bound. Registration cannot
  create admission, and a guest proof is not independent native attestation.
- Native inspection readback normalizes only false inter-container encryption
  and `None` S3 compression defaults plus `FullyReplicated` S3 distribution.
  Normalize copied views only; explicit sharding, null/invalid values and changed
  bindings stay distinct. Role/image/I/O, environment, resources and
  runtime must match the accepted request; summaries retain its exact target column.
- Inspection is pending after submit; failed reads retain the handle. No
  model-supplied image/script/environment/resources or local parsing fallback.
- Acquisition submits and verifies the fixed OS initializer/profile entrypoint;
  model/job arguments cannot select child code or privilege operations. Runtime
  output permission initialization is in place before verified privilege drop.
- Managed acquisition honors deployment-owned remote-fetch denial and snapshots
  only remote-fetch/host-allowlist controls into verified native Environment.
- Submission requires enabled S3 versioning. Acquired results include exact-version
  payload/manifest refs and hashes; a mutable URI or equal object length is insufficient.
  Prefer FULL_OBJECT SHA256, otherwise hash exact-version raw bytes with fixed
  small chunks, a finite deadline and 512 MiB cap. This control-plane I/O performs
  no dataset parsing and establishes no deserialization trust. Future consumers
  must retain pinned refs and recheck hashes before parsing.
- `_util.py:raw_tool` — Return the undecorated callable behind a Strands ``@tool`` (or the fn itself).
- `automl.py:LocalSweepBackend` — Local/offline AutoMLBackend — a tiny seeded quick-sweep over the ``train_model`` IP.
- `automl.py:AutopilotV2Backend` — Managed AutoMLBackend — orchestrates SageMaker Autopilot / AutoML v2 (T-AML-01).
- `bedrock_alt.py:BedrockFineTuneBackend` — Bedrock-native ``TrainingBackend`` — orchestrates a model-customization (fine-tune) job.
- `bedrock_alt.py:BedrockHostedBackend` — Bedrock-native ``DeploymentBackend`` — hosts via provisioned-throughput (not a SM endpoint).
- `coding_preparation.py:ProcessingCodingPreparationBackend` requires private
  catalog admission before any client calls, validates all original exact object
  versions/hashes before staging, rehashes staged copies and creates one fixed
  `amlas-processing-init preparation` job with atomic session/deployment tags.
  Network isolation, role/image/source/resources/I/O must match readback.
  Normalize only omitted or exact boolean false AppManaged on every input/output
  entry, using copies. True/null/non-bool values and unrelated fields remain
  refused; no accepted request or native response is mutated.
  Native completion is insufficient: require exact committed inventory/schema,
  owner native privilege validator, source/frame/count conservation and every
  versioned output's actual raw byte hash/line count. Polling performs no Put or
  Create. Guest observations are descriptive, never independent attestation.
  Missing owner proof helper or native image admission fails closed. This module
  is registered by `preparation_registration.py` under the shared managed-only port.
- `model_weight_acquisition.py` provides a dedicated fixed model Processing
  adapter with same-account/region role, immutable image/source and matching
  `model_weight_acquisition_admission_digest`. Missing qualification refuses.
  Its fixed initializer profile, no inputs, bounded public-fetch environment,
  one m5.xlarge/80GiB/7200seconds and EndOfJob output are verified natively.
  Preserve accepted/ambiguous handles, including faults interpreting a native
  Create response; observer failures never cause resubmission.
  Accept only the exact catalog payload/three-metadata inventory, bounded strict
  provenance/hash chains, non-null versions and FULL_OBJECT SHA256 or bounded
  exact-version raw rehash. Both checksum and stream paths check the shared
  verification deadline before/after payload HEAD. Caller payloads/config/tokenizers are never parsed.
  Guest claims grant no admission; raw acquisition establishes no model trust.
  Only an entirely successful raw-byte inventory creates a private in-memory
  witness for the original handle, execution/catalog tuple, manifest versions and
  exact payload URI/version/SHA/count/format. Repeated observations still verify
  native identity, inventory, strict fresh manifest bytes and owner-confined exact
  version HEADs. Any failed observation or binding/version drift invalidates the
  witness; checksum-only successes cannot create raw witnesses. Per-handle
  observations serialize first reads; a short lock protects the bounded map and
  unrelated job status stays independent. No witness persists or returns authority.
- Owned files: `artifact_rejection.py`, `campaign_state.py`, `model_registration.py`, `__init__.py`, `_util.py`, `automl.py`, `bedrock_alt.py`, `dataquality.py`, `deployment.py`, `etl.py`, `evaluation.py`, `explainability.py`, `featurestore.py`, `governance.py`, `labeling.py`, `lake.py`, `modelhub.py`, `monitoring.py`, `orchestration.py`, `orchestration_pipeline.py`, `pipeline.py`, `query.py`, `registry_be.py`, `retrieval.py`, `stubs.py`, `tracking.py`, `training.py`, `tuning.py`, `coding_preparation.py`, `preparation_registration.py`, `model_weight_acquisition.py`, `failure_cause.py`.

- `model_registration.py` registers only the fixed managed model acquisition
  adapter through the dedicated port; construction and imports perform no SDK
  operations and cannot establish image/profile admission.

- Managed preparation infers only all-canonical-JSONL v1 or all exact pinned
  NVIDIA OpenSWE supported Parquet v2 from private verified sources. Mixed/foreign
  profiles refuse before clients; no caller profile override exists. Preserve raw
  extension/format/bytes/version/hash during staging. Parent Parquet row count stays
  unknown; never parse Parquet or count its binary newlines in the control plane.
  Keep the fixed12 CLI pairs and selection schema1. Closed profile/policy pairs
  distinguish v1 REPORT schema1 from v2 schema2; COMMITTED schema2 remains shared.
  V2 aggregate primary drops include normalization refusals, while the report's
  primary reason map is core-only. Enforce original=retained+aggregate drops,
  normalized=retained+core drops and original=normalized+normalization refusals.
  Stream exact-version bounded lineage JSONL metadata with raw hash/count checks;
  require one original source/version/hash/ordinal entry, canonical digest and
  qualified eight-field derived edges. Missing/duplicate/foreign lineage or
  rehashed count/provenance/status drift cannot produce prepared refs. This is
  data preparation, not historical linkage verification or model readiness.

- Managed CommitPack selects only the exact source-owned profile/policy pair,
  preserves rawJSONL hashes/versions, and stages canonical newline-bound exclusions.
  Completion verifies six files plusCOMMITTED2, REPORT3 eight counts/nine coverage
  keys, all origin rows/group/membership/projection/exclusions through shared pure
  replay, and caller-private original sources. CP-only exclusion/report caps are
  8MiB and8MiB+262144; legacy caps remain unchanged. A standalone bounded
  nonusable report retains native identity and returns no split refs. Prepared
  metadata always reports training_ready false; token floors belong to trainer.

- CommitPack review regressions bind the bytes actually consumed by JSON decoding,
  account retained rejected rows and assembled output envelopes, and retain trusted
  per-source stream counts privately. Parent replay requires exact counts for every
  source; trainer may omit that independent constraint and must not claim raw-source
  authentication. Hash/parse/restore and coherent101/99 substitutions refuse.

- CommitPack replay memory (added 2026-11-07): `_commitpack_artifacts` still lists, heads, reads and
  hashes every body at its exact version and runs every qualification callback on every pass. Only the pure
  `replay_commitpack_preparation` call is taken from `_REPLAY_MEMO` when the same call ran before in this process.
  The key (`_replay_key`) is the SHA-256 of the canonical encoding of: profile and policy ids; the module constants the
  replay reads (dataset repo, revision, filename, source SHA-256 and byte count, row bound, supervision kind, splits,
  coverage); the exact-version identity (URI, version id, SHA-256, byte count, format) that this pass measured for all six
  bodies; the record count of each parsed list; the expected sources in order; and the private per-source counts. An
  input that is not plain data gives no key, so the replay runs and refuses as before. Memory is process-local, never
  persisted, holds at most `REPLAY_MEMO_ENTRIES` (2) results (the result only, about 6.5 MB of strings at 56,700 rows,
  never the parsed bodies), keeps each as one immutable JSON bytes snapshot (a result that does not round-trip unchanged
  is deep-copied), returns a private decoded copy on a miss and on a hit, never stores an exception, and lets one thread compute
  a key while others wait for it at most `REPLAY_MEMO_WAIT_SECONDS` (300) before computing alone. The comparisons after
  the replay (committed counts, coverage, exclusions digest, source record count) run on every pass. A replacement
  microVM starts with empty memory, so its first pass pays the full replay and the later ones hit. Why: the replay
  normalizes every row twice (113,400 `normalize_commitpack_row` calls per pass at 56,700 rows; 141 of 193 profiled
  seconds of a pass that takes about 63 s), and a restore runs 28 passes before its first model call (about 1,900 s). `data_eng/commitpack_*` are
  untouched because the Processing and SFT images copy them.


- `qwen_sft.py:SageMakerQwenSftBackend` owns fixed native SFT/reload, separate from generic training. Validate private source/catalog before SDK writes; stage exact S3 versions with raw SHA/count verification, source-version multipart copy and conditional complete (small whole objects omit ranges). Fixed base_model/prepared/bindings or base_model/adapter/bindings channels, six primitive pairs, isolation true and no provider Environment/HyperParameters. Original accepted/ambiguous/rejected handle is retained; observations never resubmit. Uncompressed native output layout is qualified separately; success requires native completion plus independently pinned closed output versions/hashes. Fit32 yields checkpoint only, full_training_completefalse/quality_successfalse; full1024 additionally checks unique/exposure floors. No parent model/tensor/pickle loader exists.

- [Added 2026-11-10, preference lists] Ordinary reloads pin `reload_instance_preferences_v1` (`sft_state.RELOAD_INSTANCE_PREFERENCES`); training and generation requests list their pinned preferences (see `../../control_plane/AGENTS.md`). The readback is `_selected_instance_view` (paragraph below). `selected_instance(description, request)` returns `SelectedInstanceType` and `SelectedInstanceCount` for a validated preference-list description (empty for a fixed-type request, before selection and for an UNKNOWN observation); `status()` names them in `JobStatus.detail` (`SelectedInstanceType=<type> SelectedInstanceCount=<n>`, empty otherwise) and SFT, reload and generation `artifacts()` (and therefore every store observation) carry the two keys. A fixed-type request still reads back with an empty detail and no such keys.
- [Added 2026-11-10] An UNKNOWN observation names no selected instance. `status()` maps only InProgress, Stopping, Completed, Failed and Stopped (`_STATES`); any other TrainingJobStatus (Deleting, a later value, a description without the field) reads UNKNOWN with the validated description still in `JobStatus.raw`. Before, such an observation kept `SelectedInstanceType=... SelectedInstanceCount=...` in its detail and both keys in `artifacts()`, because `selected_instance` read the description and not the state. `observed_selected_instance(status, request)` now returns the selection only for a state other than UNKNOWN; `status()` builds its detail from it, and SFT, reload and generation `artifacts()` take their two keys from it. `selected_instance(description, request)` is unchanged and stays the reader for callers that checked the native state themselves (`app._original_sft_end`).
- Until 2026-11-10 ordinary Qwen reload alone chose fixed `reload_single_gpu_v1` (records that carry it keep it): one
  ml.g6e.4xlarge/100GiB. Durable schema4 reserves that exact contract before staging
  and Create; staging uses the original reserved intent. Historical schemas1/2/3
  keep original12xlarge request bytes, identity and readback; restore never submits.
  Training, RUN/checkpoint bytes, source/image/role/approval and all other fixed
  request fields remain unchanged. This source selection proves no GPU capacity
  or native4xlarge reload acceptance.

- Native readback accepts only absent/empty-map Environment and HyperParameters. Ownership tags compare exact key/value inventories independently of order, refusing duplicates/pagination. Step witnesses cover32/1024 updates and reconcile exposure; distinct corpus/provenance counts are bounded by original prepared TRAIN descriptors. Unique corpus tokens remain independent from optimizer exposure.
- `qwen_sft_registration.py` lazily binds the fixed managed SFT adapter only; source/image/native admission is verified separately from registry construction.

- `campaign_state.py:DynamoCampaignStateBackend` uses only the configured
  selected deployment jobs table. Fresh STS account/assumed-role incarnation
  checks bind each call to the configured server execution role; there is no
  verified end-user principal. Conditional Put reserves one digest-derived native
  name, and revision/digest/state conditional updates retain attempted, accepted,
  ambiguous and rejected records. Typed reload binds the decoded native name to
  the exact selected table key; a valid record under another key must refuse.
  No table/IAM creation or failure-row deletion
  is allowed. Native observation performs Describe/ListTags only and verifies
  original identity/role/image/tags; it never creates or writes, and reports no
  artifact verification, trained model or quality success. Full native request
  and artifact revalidation remain the consuming lifecycle's separate gates.

- Optional private durable SFT context is passed per call, never cached on the
  shared adapter. Reserve the complete intent/name before staging; only a
  successfully acknowledged conditional attempt marker can stage and Create.
  A closed exact request/RUN/staged-version/count snapshot must be conditionally
  acknowledged before the one original Create. Reused/uncertain reservations
  restore/observe only. Stage and accepted/ambiguous/rejected outcome failures
  retain original identity; failed persistence never downgrades native acceptance.
- `restore_original` accepts only closed original records and freshly verified
  same-store input snapshots. Replay original COMMITTED/RUN counts and bytes,
  recheck original/staged exact refs, then use the complete original request/tag
  and output-artifact verifiers. Restore/observation performs no staging, Create,
  deletion or record update. Interrupted records without complete evidence stay
  unresolved; legacy uninjected submission retains its original behavior.
- Campaign snapshot attachment and later transitions compare the exact original
  persisted body alongside digest/revision/state. Stored bytes support conditional
  integrity only and carry no action grant; schema1 can upgrade by the same CAS.

- `coding_generation.py:SageMakerCodingGenerationBackend` reuses exact-version SFT byte transport and native request readback. The original base prefix must contain exactly nine pinned files at current versions; GPU inputs contain public PROMPTS only, never oracle/exclusion files. Fixed `generate/humaneval_greedy_v1`, g6e.4xlarge, network isolation and immutable source/image selectors are server-owned. One native Create retains an accepted or ambiguous original handle; polling never resubmits. Generation completion requires exact three-file COMMITTED inventory, hashes, original RUN/spec/model/tokenizer/contract identity, all164 retained outcomes and descriptive privilege checks; native admission and quality remain false.

- Adopted base model in the generation inventory (added 2026-11-08): when the model source is a
  `VerifiedSftAdoptedModelSource`, its `adoption_ref` (`adoption.json`, written by `ModelPackAdoptionProvider.adopt` into
  the same prefix as the copied model files) joins the exact base-prefix inventory of `SageMakerCodingGenerationBackend.submit`
  and is hash-verified with the provenance files; it is never staged into the native model channel, and a record outside
  the prefix is not counted. Why: the first generation submit on an adopted base model refused its own prefix listing as
  "extra or missing files"; `tests/test_generation_adopted_model_layout.py` covers the adopted layout.
- Completed native generation artifact roots may omit exactly one trailing separator. Validate the unchanged original URI against its request-owned bucket/output scope and every nonempty plain segment before constructing the slash-terminated S3 inventory prefix. Root-only, empty/dot/dotdot/double segments, query/fragment/backslash and foreign scopes refuse before output inventory reads. Preserve raw native URI/status, exact three-file inventory and all current version/hash/size/population checks; this normalization grants no native or quality admission.
- Completed native SFT and reload artifact roots follow the same one-optional-separator rule and are bound to the job. `qwen_sft.py:_native_uncompressed_output_prefix` drops at most one trailing separator, then requires the value to equal `<request S3OutputPath>/<TrainingJobName>/output/model` (what real SageMaker returns for an uncompressed output, with no trailing separator) or `<request S3OutputPath>/model` (the simulated provider's layout) in the catalog bucket, and returns the slash-terminated key prefix. Another bucket, job, depth or root, an archive name, dot or empty segments, query, fragment, backslash and non-string values refuse before any object read. The exact closed inventory, hash and receipt checks that follow are unchanged.
- [Added 2026-11-10] Paired request tags: `coding_pair.py:_handle` compares the stored request with the request it rebuilds in every field but `Tags`; `_submitted_tags` then accepts the stored tag list only when its `amlas:`/`aws:` entries equal the three fixed ownership tags exactly and every other key is unique, string-valued and outside those namespaces (a deployment's own `project`, `auto-delete` or `purpose` tags). `status()` compares the live tags with that stored list, as the SFT and generation readbacks do. A missing, changed or extra ownership tag, an `aws:` key, a repeated key, a malformed entry and any live tag set that differs from the stored list refuse. Why: the real pair job of 2026-11-10 carried its deployment's tags; the rebuilt-request equality refused it before the tag check could run, and the tag check refused it after, so no tag set beyond the three could pass both.
- [Added 2026-11-10] Paired invocation records: `artifacts()` reads `INVOCATIONS_MANIFEST.jsonl` whole (version, size and SHA-256 verified, stream closed) before it reads the first record, then verifies each record exactly as before. The record pass gets its own deadline, `_CORE_SECONDS` (600) plus `_record_pass_seconds` of the declared record count at 0.5 s a record, and the manifest must still be the current version after the pass. Why: the real 164-case pair wrote 12,786 records, five S3 calls each; through this adapter with plain clients from the job's Region, `artifacts()` took 2,159.7 s (0.169 s a record; a shimmed pass took 1,415 s); with the reads inside the open manifest stream and one fixed 600 s deadline, the run hit the deadline, a mid-stream `IncompleteReadError` and the body-close timer.
- [Added 2026-11-10] Absent selection after the waiting phase: `_selected_instance_view` also accepts both Selected fields absent when the job is Stopped, Failed or Stopping and `_ended_before_selection` holds: `SecondaryStatusTransitions` is a list of mappings whose Status values are one or more Starting/Pending entries, then an optional Stopping entry and, for Stopped or Failed, the matching terminal entry, and the last entry equals `SecondaryStatus`. Any later transition, an unlisted status, a Completed job, malformed transitions, one field alone, an unlisted type or another count still refuse. Why: a preference-list job stopped or failed while it waited for capacity has no selection (real fixed-type jobs stopped while Pending show Starting, Pending, Stopped), and reading it UNKNOWN brought back the earlier resume failure at checkpoint_load. Not observed: a real description of a preference-list job that ended before selection; the absence of both fields is taken from the waiting-state description and the API text ("returned after an instance type is selected").
- [Added 2026-11-10] Preference-list readback: `qwen_sft.py:_selected_instance_view` runs inside `_native_request_view`, so SFT, reload and generation `status()` all use it. For a stored request whose ResourceConfig carries `InstancePreferences`, the described `SelectedInstanceType` must be one of the stored types and `SelectedInstanceCount` the stored `InstanceCount`; both are required once the job has left the Starting/Pending secondary states and may be absent only while it is InProgress in one of them. One alone, an unlisted type, another count or a malformed value refuses; the rest of ResourceConfig (preference order, count, volume) is still compared exactly, and a fixed-type request is compared unchanged. The selected values stay in `JobStatus.raw`. When this paragraph was written no submit path created a preference list; since the preference-list paragraph above the training, reload and generation submits do. Why: the three real preference-list jobs of 2026-11-10 read back UNKNOWN because their descriptions add the two Selected fields.

- Generation Create error inspection accepts only exact nested dictionaries and string service codes; malformed payloads retain the original ambiguous handle. The output consumer reads exact-version public PROMPTS bytes, verifies their canonical population/hash, and checks each candidate prompt hash and the closed worker outcome/stage/partial-source retention contract. Manifest/candidate equality uses canonical bytes so boolean/float values cannot alias integer counters. All164 outcomes remain retained; this verification grants no native containment or quality authority.

- `coding_pair.py` reads exact eleven original case/base/tuned artifacts, validates resolved spec and immutable producer metadata, conditionally stages one fixed Processing packet whose closed oracle_launcher_sha256 comes from the trusted pair-only catalog, and retains accepted/ambiguous identity before/after one Create. The broker needs AWS API network access; candidate execution remains in separately observed CodeInterpreter SANDBOX sessions. Native readback verifies fixed request/tags and only documented defaults. Output acceptance requires complete328 receipts, exact committed core files, streamed versioned trace hashes, bounded closed session/call replay, original scope and receipt-to-session/value-call chain coverage. Stream the owning internal trace proof alongside exact artifact hashes; hashes alone cannot prove execution coverage. ARM callers pass the trusted expected launcher SHA into the source-owned runner fingerprint and never read an AMD64 launcher. Retain the frozen pair catalog alongside the unchanged raw execution catalog; expected digest/RUN/execution drift refuses before Describe. Processing Environment is exactly the trusted AMLAS_CODING_PAIR_LAUNCHER_SHA256 field; missing, changed or extra fields refuse native readback, and no generic/provider environment is forwarded. The worker must independently verify its fixed installed launcher against that trusted selector and the RUN binding; digest consistency alone grants no authentication. Native admission and quality remain false. Preserve the pure verifier's unresolved cleanup or incomplete success/evaluated/status fields; verified artifact hashes do not upgrade lifecycle completion.

- Coding generation and paired registrations install local-import factories over their original managed-only ports/defaults. Bootstrap must not import adapters that re-enter partially initialized conversation state. Fresh normal orchestration→ToolBindingBackend regression verifies both defaults without prior registration imports or SDK client creation; registration grants no admission.

- Qualified cross-source Qwen reload builds exactly the closed12-field RUNv2
  from a fresh server-only exact compatibility selection. Reconstructed current
  consumer/source/image and original producer/checkpoint identities remain distinct.
  Recheck selection before staging each file/RUN and native Create, and on original
  restoration/status/output readback. Stored handles contain data only, no reader
  or grants. Schema2 RELOAD/COMMITTED must match the originally selected producer,
  contract and compatibility digest; rehashed substitutions and schema1 fallback
  refuse. Legacy same-source RUN/receipts retain schema1 and their exact fields.

- Fixed receipt descriptors are canonical two-field SHA/count maps. The consumer
  additionally recognizes exactly the historical three-field format=json descriptor
  for TRAINING.json/RELOAD.json only; unknown fields, formats, boolean counts and
  substituted refs refuse. This handles the existing fixed-writer/consumer mismatch
  without rewriting old output. Qualified native status/restoration/artifact success
  rereads exact current server selection immediately before return; sequential checks
  do not promise an atomic configuration or IAM snapshot.

- ModelAcquisitionConfigurationError is bounded startup DATA: two fixed missing/invalid codes, readonly closed model selector-name tuples and fixed text. It stores no selector values, credentials, handles, witness or action grant. Existing request/configuration validation and native effect/observation guards remain unchanged.

- Normal paired Processing request invokes only compiled `/usr/local/bin/amlas-processing-init coding-pair` with its same three source/binding pairs. Image USERroot runs only static OS startup; fixed empty output FD repair and permanent broker UID/GID10001/group[10001]/activecaps0/NNP precede SDK imports. The role metadata exception remains only a validated native relative URI; candidate/oracle isolation and all request/native readback pinning remain unchanged.

- Dataset/model acquisition initial Creates share only the private `_util` process-local account/Region gate: mutex and monotonic initial sends conservatively separated from the prior block completion by fixed1.25s, one initial SDK attempt, reservation retained on refused/lost reply. Completion timestamp readback failure poisons future sends without masking the original accepted/ambiguous result or exception. Wait precedes the same original model before_create/private I/O checks; dataset repeats its private I/O and deployment-fetch denial immediately after waiting. Both mutating SDK clients use total_max_attempts1. No Describe pacing, polling delay, Start retry, reminted original, extra service or cross-process/account-global coordination exists. Inherited process state, backward/nonfinite clocks and incomplete waits refuse.

- Generation and pair native identity (added 2026-11-07): the native job name of a generation or pair
  submission is `<prefix>-gen-<digest16>` or `<prefix>-pair-<digest16>` from `generation_operation_digest` /
  `pair_operation_digest` (control_plane), not a random suffix, so one request has one native identity. Staged
  inputs and RUN/SPEC objects live at deterministic keys under that name: `_adopt_or_copy` and the RUN/SPEC put take
  an object already at the key only when its exact version bytes verify (`_raw`); a failed lookup reads as absent and
  the conditional create then fails closed. A repeat Create that the service answers with `ResourceInUse` is adopted
  only when `describe` returns this operation ARN (`_is_this_native_job`); the handle then reports
  `submission_state=accepted` and `adopted_existing=true`. Any other error keeps the old classification
  (rejected or ambiguous). `status` and `artifacts` still verify the native job against the recomputed request
  and tags, so an adopted job is checked exactly like one this process created.

- Process pins (added 2026-11-07): `process_pins.py` owns a bounded, thread-safe, fork-reset record store
  with single-flight claims (`ProcessRecords`) and the HEAD fingerprint helpers. `SageMakerQwenSftBackend._raw` records
  a pin in the module store `_PROCESS_PINS` after a full, non-collect read whose size, SHA-256 and qualification all
  succeeded, only when the HEAD and the GET name the same ETag and the HEAD carries a checksum field; the key is account,
  Region, bucket, URI, version id, SHA-256 and byte count, and a ref whose URI names another bucket is never pinned.
  `_verified(ref, catalog, deadline)` is the first full read in a process and an exact-version HEAD afterwards, from
  any adapter: it compares version id, size and the whole fingerprint (an absent field differs from a present one) with
  the pin; a missing pin, a different fingerprint, size or version takes `_raw` (which hashes the bytes or refuses), and a
  failing HEAD raises and leaves the pin (a pin admits nothing without a later HEAD that matches it). A pin that did not
  match is dropped before the full read. One thread reads a version at
  a time; another asking for it waits on that read (polling its own deadline and `qualified_io` qualifier every
  second) and then makes its own HEAD, so a status poll, the after-tool wait loop and a cold restore never hash the same
  object together. `restore_original` (staged copies), `_verified_run_binding` (the originals) and
  `ModelPackAdoptionProvider._verify_staged` call `_verified`; parsed metadata (`collect=True`), objects with a
  service SHA-256, and `_copy`'s read of its own new copy keep the full `_raw`. Authority, identity, scope, namespace,
  policy and qualifier checks run on every pass, pinned or not; a pin certifies bytes only. After the first full hash in a
  process, a later silent S3 corruption is not detected until the next process (as with the generation pins). The session
  bucket is SSE-KMS; whether the checksum HEAD exercises the kms:Decrypt permission that a GET does is not verified for
  general purpose buckets (the HeadObject reference states that requirement for directory buckets). Why: at production size one `restore_original` re-hashed
  2 x 17.99 GiB and one adoption restore 17.99 GiB, a 15 s loop iteration cost 9 to 12.5 minutes, and a status poll
  blocked the app's event loop for about 35 minutes.

- Model byte proof sharing (added 2026-11-07): the model adapter's `_ModelByteProof` still belongs to one
  handle object and one adapter. After a full raw pass over all nine payloads it also publishes a `_SharedByteProof`
  (manifest versions, payload tuples and each payload's HEAD fingerprint, no handle, no client) in `_PROCESS_PROOFS`,
  keyed by the proof binding without the client identities. Another adapter in the process, or a re-decoded handle, finds
  it in `_artifacts` and runs the existing pinned branch with its own clients and `ExpectedBucketOwner`: the latest HEAD
  names the pinned version, an exact-version HEAD returns the same size and, now, the recorded fingerprint, and the
  result tuple and manifests are compared; any difference fails closed as before and drops the proof. A first pass is
  single-flight per binding. A status that is not SUCCEEDED, or any refusal, drops both proofs and bumps the key's epoch,
  so a pass that began earlier cannot store a proof after the drop. Why: every cold restore and every second provider
  rebuilt an adapter and hashed the pack again (505 to 576 s).

- Refusal causes and interrupted model passes (added 2026-11-10): `failure_cause.py` names a caught
  exception for a log line with closed fields only, never its message: the nearest reviewed class of its MRO (else
  `Exception`), the closed native code of a botocore `ClientError` (`other` outside the set, `none` for other classes),
  the innermost traceback frame inside the `amlas` package as `file.py:line` (`unknown:0` without one) and the closed
  class of the root of an explicit `raise ... from` (`none`). The model adapter's `model_artifact_refused` line carries
  them. A designed run-record recheck raises `record_changed(text, reason)`, the exact `ValueError` with its unchanged
  single text plus a private marker that `record_change` reads; a subclass, a forged reason or an error raised `from`
  it reads as no record change. When such an error ends a model observation (`_status`) or a byte pass (`_artifacts`),
  the adapter keeps its own and the process-wide byte proof, logs one closed `model_verification_interrupted` line
  (reason and site) and re-raises the error, so the caller's own recheck decides. Every other error, a transient SDK
  fault included, stays a refusal: both proofs are dropped and `verification_failed` is returned as before. Why: in one
  run a run-row write landed inside a status read's pass; the recheck's ValueError was logged as a refused artifact
  without its class (five transient faults give the same line), and the dropped process proof cost a 10 min 43 s
  re-hash.

- Generation pin re-verification (added 2026-11-07): `SageMakerCodingGenerationBackend.submit` hash-verifies
  every pin it hands to the native job (the nine original model files, the staged copies, the prompts, the adapter files
  and RUN.json) and records each one in `handle.extra["verified_pins"]` with the HEAD fingerprint of that verification
  (version id, size, ETag, checksum fields, storage class). `_raw(..., seen=)` records a pin only after size, SHA-256 and
  qualification have all succeeded, and a streamed body whose ETag differs from the HEAD it follows is refused;
  `_copy` and `_adopt_or_copy` pass `seen` through and every other caller omits it.
  `artifacts` replaces the re-hash of those 18 files by `_reverify`: a HEAD on the exact version that returns the
  recorded fingerprint shows the same immutable object and reads no bytes; a missing record, a different fingerprint
  or any failure takes the full `_raw` read, and a removed version or a mismatch fails closed as
  `output_verification_failed`. Each staged key must also still resolve to its pinned version (HEAD without a version
  id), because the native channel reads the current version; the original keys are not checked that way. The record
  travels with the handle, so a replacement process gets its own from `submit` (a `ResourceInUse` adoption included),
  and a handle without one is verified in full. The observation deadline is 600 s for the fixed work plus
  `_hash_seconds` of the bytes that have no record (13 MiB/s, a third of the slowest measured stream), and a fallback
  read extends it by the same rule. Status, output reads, COMMITTED/GENERATION/CANDIDATES checks, the 164-row checks
  and the privilege proof run on every call as before, and the worker still hashes the input files it consumes
  (`_HeldNativeInputs.verify`). Why: a call re-streamed 2 x 17.99 GiB at 42 to 95 MiB/s (388 to 620 s) against a fixed
  600 s deadline, `get_coding_generation` repeated it on every poll and the pair submit twice more, so the same
  correct work could fail the same way every time.

- Recorded rejection code (added 2026-11-10): `coding_generation.py` and `coding_pair.py` set
  `extra["rejection_code"]` to the native code (ValidationException, AccessDeniedException or ResourceLimitExceeded)
  when a definite rejection sets `submission_state=rejected`. The classification, the single
  create per call and the ambiguous rule are unchanged; accepted and ambiguous handles carry no such key.
  [Added 2026-11-10] For ResourceLimitExceeded `coding_generation.py` also sets `extra["rejection_type"]`
  (`_limit_type`): the one instance type the native message names in the form "'<type> for training job usage'",
  kept only when the message is a string of at most 2,048 characters that names exactly one type and the request lists
  that type; otherwise None. No other text of the message is kept. Why: SageMaker refuses a create when any listed
  type is at the account limit and names that one type; the generation store's capacity note names only it. The pair
  adapter records no type (a pair submit result carries no capacity note). [Added 2026-11-10] The stores
  answer a read of a capacity-rejected handle from these stored fields and never call `status()` or `artifacts()`
  for it: no native job exists under its name.

## Work Guidance

Preparation uses globally distinct Processing input/output names
`preparation-input` and `preparation-output`; fixed local paths and source-bound
CLI remain unchanged. Native readback validates the accepted exact request.

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.

- Every successful canonical/Open-SWE/CommitPack preparation artifact result includes `job_name` and `job_arn` from the original native handle after successful status/readback. Private SFT handoff consumes this verified adapter protocol directly; the public observe merge supplies no authority. Native dictionary refs remain closed five-field descriptors.

- Fixed Qwen SFT observation compares copied native request views, normalizing only exact omitted-request metrics=false, output KmsKeyId=empty string and channel RecordWrapperType=None string defaults. Explicit request values, malformed types, unknown fields and all ownership/image/isolation bindings remain distinct. Native response and accepted request remain unchanged.

- CP SFT receipt consumption requires the closed bounded summary and original
  private exact-version COMMITTED/header/profile/counts/coverage/exclusion pins.
  Parent checks count/schema/digest format, not independent ordered-row digest
  authentication without row replay. Legacy preparation profiles remain unchanged.

- `coding_case_freeze.py` owns fixed Processing case freeze: exact original source versions/SHA/counts, conditional staged objects, immutable binding, isolated m5.xlarge/1/30GB/900s, no Environment or caller overrides. Retain accepted/ambiguous/rejected original handles; never resubmit on observation. Unknown or malformed nested SDK rejection metadata retains the original ambiguous handle; only exact string rejection codes change its disposition. Completed data requires five exact-version/hash outputs, strict managed producer/source envelopes and unchanged pure164 replay. Private ORACLES/EXCLUSIONS are backend-only refs. Native/image/worker qualification remains external.

- Model acquisition `prepare_submission` is pure fixed-request data from explicit
  or environment configuration: no SDK/configuration fetch and no action grant.
  An optional original job name must match the fixed source-owned job grammar.
  `validate_prepared` reconstructs every current fixed field and compares canonical
  bounded data before clients. `submit_prepared` retains original identity, leaves
  caller plan data unchanged and revalidates after destination reads and on both
  sides of its private `before_create` callback immediately before Create. Callback
  refusal precedes the native try and cannot be called ambiguous native acceptance.
  Legacy `submit` retains its existing SSM resolution and guarded native dispatch;
  durable reservation/attempt permissions and fresh caller policy are separate.

- Model acquisition `qualified_io` is an optional private owning-thread/PID scope.
  Each SDK leg and metadata/payload stream read checks the active qualifier before
  and after; post-Get refusal closes the original stream. Scope exit revokes saved
  context copies before reset. Legacy unscoped behavior is unchanged. Pre-Create
  qualification refuses outside the native try; post-Create refusal retains exact
  original identity as ambiguous. This hook supplies no native authentication,
  permission, hard transport deadline or immutable-source proof by itself.

- Model acquisition nested `qualified_io` retains each lexical parent scope.
  Every actual SDK/stream boundary validates PID/thread/active lifetime and calls
  outer then inner qualifiers before/after I/O; innerNone cannot erase the parent.
  Revoked outer policy refuses the next read. Exiting restores the original parent
  and revokes the child scope. Artifact integrity and native request guards are
  unchanged; qualifier composition supplies no permission or cached grant.

- DATA/CASE/PREP qualified_io is a private per-instance PID/owning-thread observation
  scope with lexical parent composition and finally revocation/reset. Native
  Describe/ListTags and S3 observation List/Head/Get, each bounded stream read and
  qualified final status/artifact exposure recheck current callbacks. Post-Get
  denial closes the returned body; CASE inherited readers use its own scope. Lazy
  lineage captures its original lexical scope before iteration and rechecks it
  before reads, buffered yields and final verified exhaustion. Legacy unscoped
  behavior and all version/hash/count/request/closed-output guards remain intact.
  Submit/Create/staging/retry/permission behavior is unchanged: this scope grants
  no native authority and solves no attempt ownership, complete DATA request or
  PREP native tag gap. Callback code must not recursively invoke these readers.

- Observation body ownership begins before timer construction/start. Always close
  each returned stream, including timer startup failure; cancel/join only a timer
  whose start completed. Preserve in-flight BaseException/process-control identity
  if body or timer cleanup fails, so cancellation never becomes ordinary artifact
  failure DATA. Normal cleanup failure still refuses qualified output. Existing
 120-second bounds remain; this is not native lifecycle cancellation authority.

- PREP returned-body allowance (added 2026-11-06): the 120-second wall-clock allowance of a
  returned PREP stream (raw and lineage bodies) bounds stream, decode and consumer work. Time spent inside
  the active `qualified_io` callbacks while that body is open (the lineage `qualify` calls and the
  before/after checks of each `_observation_read`) is credited back, including the call still running, up to
  1800 seconds per body, and the close timer re-arms for the credited time. Every qualification call site, check and refusal is unchanged;
  slow reads, decode or consumer work still exhaust the allowance. Why: one lineage pass makes about 62,000
  qualifier calls, each re-serializing the paused conversation and walking the pending originals, so passes
  of a run that had adopted a model ran the allowance out (an identical 245 s UNKNOWN cycle for seven hours).
  Each body logs one `preparation_body_budget` summary (calls, credited and wall seconds, outcome class), and
  an UNKNOWN status carries a `reason` only when its ValueError wording is on the explicit allowlist of this
  module's fixed refusal messages (`_REASONS`); no raw message, id or path leaves. Sibling adapters keep their fixed allowances until their own evidence.

- PREP lineage per-record check (added 2026-11-06): `_stream_lineage_objects` still runs the scope check
  (process, scope lifetime, thread) and every scope callback before each yielded record. A callback that is a
  `RecordRecheck` supplies its own per-record variant (`record_check`): its local checks every record and its full
  check once `interval` seconds have passed since the start of the last full one that returned normally (a failed
  check never counts; a read bracket's full call restarts the interval). Plain callbacks run in full for every
  record, as before, and every read, body open and close still run the full callback. Why: the SFT parent capture
  callback makes about four native round trips and a row read per call, one lineage pass yields about 62,000 records,
  and a run held its first SFT call for more than four hours (two earlier runs 52 minutes and 2 h 38 min). The lineage
  reader asks for `LINEAGE_READ_BYTES` (1 MiB) per read, so the before/after read brackets run about once per MiB
  (a 184 MB object is about 175 brackets instead of 2,800).

- Object body chunk cadence (added 2026-11-07): the byte readers that hash an object in 1 MiB chunks
  (`qwen_sft._raw` and `_output`, `model_weight_acquisition._payload_observed`,
  `dataset_acquisition._verify_version_payload` and the preparation adapter's `_raw`) bracket every chunk read through
  `_qualify_chunk` (`_qualify_io(between_chunks=True)` in the model adapter; the preparation adapter reuses
  `_qualify_scope_record`). The lineage stream keeps its own read brackets (full around every read). The scope checks
  (process, scope lifetime, thread) still run for every chunk. A scope callback that is a `RecordRecheck` is asked for
  `record_check()` through `coding_preparation.record_step`, which runs its local checks and its full check once per
  interval; any other callback runs in full for every chunk, as before. Every request bracket (HEAD, GET, list) and
  the bracket after the last chunk still run the full callback, so a pass ends on a full check. Verification (size,
  version, SHA-256, deadlines) is unchanged. Why: the adopted model pack is 19.3 GB, one pass is about 19,300 chunks, and a full check
  per chunk bracket (two per MiB, each about four native round trips plus the adoption provider's own) made one pass
  88 minutes (one run: two silent restores of 88.7 and 87.3 minutes). A 64 MiB stand-in pass: 135 full checks before,
  7 after. The shared `SageMakerQwenSftBackend` reader also serves the original SFT input binding
  (`_verified_run_binding`), the staged-copy re-hash (`_copy`), `restore_original` and the post-SFT output read.

- Body cleanup tracks pending failure explicitly inside its own setup/yield.
  Ambient caller except state cannot suppress a normal close error. Retain the
  original internal process-control exception, not unrelated handled exceptions.

- Model submission/observation failures expose only closed diagnostic DATA: fixed
  create_boundary/describe phase and code, finite error class, ValidationException
  or UNKNOWN provider code, optional exact UUID request ID and integer HTTP status.
  Never expose provider message/envelope/request/headers or selector values. Retained
  diagnostics are reprojected as bounded scalars; original status/accepted/handle
  and qualification/readback guards are unchanged. No diagnostic grants retry.

- Model Describe is native observation evidence only; it never rewrites the
  original Create-produced accepted/submission_ambiguous lifecycle pair. Preserve
  both the normalTrue/False and lost-replyFalse/True provenance and private handle
  pin through fresh matched readback; no acceptance promotion or replay occurs.
  Independent native completion/artifact verification remains required.

- Generic managed deployment consumes its original private package without a
  second CreateModel or caller model-name replacement. Single endpoint creation
  retains its accepted identity across unreadable/mismatched native observations;
  exact observed status is DATA, never fabricated InService or a retry grant.
  Local predictor binding reports confirmed local readiness. Shared EndpointRef
  schema, tabular/serverless contracts and existing role/Region checks remain.

- Managed deployment role selection preserves explicit constructor precedence,
  then the shared pure ENV ordering SAGEMAKER_EXECUTION_ROLE_ARN before
  AMLAS_SAGEMAKER_ROLE_ARN. Trim and placeholder handling match other role
  consumers; missing selection retains the existing invalid placeholder.
  Startup remains SDK/SSM/profile-free and clients stay lazy. Selected role DATA
  is not IAM/hosting admission; original package/configuration guards remain.

- Private original package qualification is fresh before/after each config,
  endpoint, description and autoscaling boundary and at exposure; no guard spans
  SDK execution. Accepted endpoint identity is retained privately before postwrite
  checks. Closure reports UNKNOWN without replay or publishing an unqualified
  handle. Describe fallback logs only its finite failure class; no raw exceptions,
  requests, responses or coordinates. BaseException control flow propagates.

- Endpoint ARN identity must bind the requested name and selected Region with a
  closed native partition/service/account shape. Matching contradictory Create/
  Describe ARNs remain UNKNOWN; ARN equality alone cannot establish InService.
  Legacy transport ARN omission can use separately valid original readback DATA.

- Model failure diagnostics retain only finite ValidationException, ResourceInUse,
  ResourceLimitExceeded, ResourceNotFound, ThrottlingException and
  AccessDeniedException labels plus UNKNOWN. A native label/HTTP status is DATA,
  never rejection, permission, admission or resubmission authority. Existing
  bounded request-ID/class/status projection and raw-message exclusion remain.

- SFT/reload, generation and paired observation inherit one per-instance
  `qualified_io` scope with original PID/thread/lexical lifetime and composed
  parent callbacks. Explicit native status/tags, S3 inventory/HEAD/GET and every
  streaming read qualify before/after; returned bodies are owned before timer
  setup and closed before a fresh exposure check. Cleanup preserves an original
  process-control exception. The private scope grants no Submit/Create, staging,
  retry, admission or observer mutation; legacy unscoped requests remain intact.
  Paired committed output descriptors qualify before/after HeadObject and freshly
  before returning the unchanged exact-version/hash/count/format reference.


- `artifact_rejection.py` owns only exact-type finite immutable output refusals.
  The two source reasons are raw hash/count mismatch and known pure closed-manifest
  rejection after original native Completed qualification. Projection is DATA only;
  generic ValueError/subclasses, missing publication, deadline, SDK/transport and
  cleanup failures remain unmarked uncertainty. No text parsing or grant is added.

- The same exact immutable rejection schema is projected by DATA after a conclusive
  exact-version payload hash/count failure and by SFT/train+reload, generation and
  paired evaluation after their parsed original COMMITTED source/schema guards.
  Existing native Completed/request/tag verification precedes those catches;
  unchanged legacy error_type ValueError and false result flags remain. Missing
  publication, read/cleanup faults and deadlines never gain a closed marker.
