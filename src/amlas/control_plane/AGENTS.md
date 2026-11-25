# src/amlas/control_plane

## Purpose

Own capability dispatch, approval/plan policy, cost boundaries, project memory, sessions and integration hooks.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Owned preparation submit/reuse/observation results expose equal opaque
  `preparation_request_ref` and compatibility `request_ref` selectors. The
  versioned `preparation_ref` artifact descriptor stays separate and unchanged.
  Aliases identify the existing same-conversation record only; they confer no
  approval, artifact readiness or native authority. Unknown/denied refs gain no alias.

- Normal SFT capture accepts an optional per-call phase observer. It carries only
  fixed phase labels, grants no authority and is never serialized. Ordinary
  observer failures leave the existing capture checks and their exceptions intact.
  Capture forwards only fixed labels immediately before causal bounds, checkpoint
  DATA, prefix canonicalization, assistant batch, unique tool use, effective SFT
  inputs and full-call canonicalization. Check order, byte budgets, immutable
  assignment and all refusal behavior remain unchanged; process control propagates.
  An optional per-call predicate observer receives only a fixed code at the
  existing effective-input refusal; no second validation or DATA is exposed.

<!-- normal-model-progress-diagnostic-contract:begin -->
- Optional normal MODEL warnings validate the current exact local context, then
  log only its opaque run ID, one of five fixed loading phases, one of six fixed
  job states and an exact integer verified-payload count0..9. Absent or stale
  context and ordinary logger failures emit nothing and preserve outcomes;
  process-control exceptions propagate. This helper performs no native I/O,
  graph traversal, checkpoint/history mutation or extra permission check.
  CloudWatch diagnostics do not establish liveness, readiness or recovery authority.
<!-- normal-model-progress-diagnostic-contract:end -->

<!-- normal-handoff-progress-diagnostic-contract:begin -->
- Optional normal handoff warnings publish only the current exact local context's
  opaque run correlation and fixed phase/kind/state/bool scalars. READY alone
  requires readyTrue. Missing/stale context and ordinary logging errors emit
  nothing; process-control exceptions propagate. No native observation, extra
  qualification, graph/history mutation or authority follows from diagnostics.
<!-- normal-handoff-progress-diagnostic-contract:end -->

<!-- normal-native-report-pointer-contract:begin -->
- `normal_run_checkpoint.py` accepts only the optional closed
  `report_checkpoint_ref` on existing enrolled/SFT rows. The fixed `reports`
  object kind uses the original run/private-correlation namespace and 160 KiB
  cap. Existing immutable object publication/readback precedes single-assignment
  pointer CAS; no ownership, stage, lease, epoch or producer field changes.
  Fresh native role/table/incarnation and SDK hash qualify retention/readback;
  report metadata CAS requires native owner authority and does not invent an ML
  approval. Existing producer `_replace_record` policy gates remain unchanged.
  Passive report reading is native read-only and unknown/missing/mutated pointers
  refuse. Absent-pointer old rows stay valid; no new lifecycle state is introduced.
  Report helpers test canonical catalog absence before entering normal resume
  preparation; unconfigured local paths construct no selected-native context.
  The private transient enrollment-confirmed flag is set only after existing
  enrollment acknowledgement or exact readback. It selects initial-worker report
  applicability, never native permission or persisted owner/lease/epoch state.
  Resumed report pointer (added 2026-11-07): the original worker's report (a held, refused or
  pause-turn response of an early-ended original) and the final report of a replacement used to share the one
  single-assignment `report_checkpoint_ref`, so the replacement's different final report was refused after the run
  had finished all its effects and `get_prompt_report` kept returning the original's. The replacement now passes
  `retain_normal_prompt_report(..., resumed=True)` and publishes the separate single-assignment
  `resumed_report_checkpoint_ref` (same closed descriptor, same reports namespace and cap, same object-then-CAS
  order). The write needs a started replacement invocation (`_replacement_started`: stage `sft`, continuation
  schema 2 or 3, `consumed-ready`, `invocation_state` `invoking`); a row that carries the resumed pointer without
  it is invalid. The original pointer is never changed, a different second report on either pointer is still
  refused, an identical replay keeps the winning version, and `read_normal_prompt_report` returns the resumed
  report once its pointer exists and refuses (never falls back to the original's report) when that pointer is
  damaged. Before the replacement finishes, the original's report is what a reader gets.
<!-- normal-native-report-pointer-contract:end -->

- Ended read (added 2026-11-23): `read_normal_run_status` is a second reader of the report object. For a row
  in stage `enrolled` that holds `report_checkpoint_ref`, `_ended_before_sft` verifies the report with
  `_verified_report` (exact version, digest, size, canonical bytes, closed shape), reads the row again,
  requires it unchanged and qualifies once more, as the passive report read does. It projects the response
  action into `input_admission_held`, `guardrail_intervened` or `response_received`. The status keeps the
  `enrolled` answer and adds `run_state` `ended`, `ended_action` and `message`; the resume read returns that
  answer where it returned `unavailable`. This is a read-time projection. It writes nothing, changes no
  stage, pointer, ownership, lease, epoch or producer field, introduces no lifecycle state, and is no
  completion proof for a resume. A row in stage `sft` is never read as ended: its original pointer may hold
  the report of an early-ended original that a replacement continues. An unverifiable report or a changed
  row gives the closed `unavailable` answer. That such a row is final rests on the worker order, as read
  from the code: `_associate`, the only step out of stage `enrolled`, has one call site, inside the run's own
  worker before that worker returns and retains its report. No test forces a late association.

- The two normal original/journal current callbacks select `_load`'s private
  immutable exact-raw projection. LRU4 retains only validated primitive DATA:
  raw/association/context strings and an optional canonical prepared descriptor.
  Every fresh policy/role/table, consistent GetItem, envelope/size/SHA, current
  run/session/catalog/incarnation and prepared-bucket/ref check remains active.
  Mutating journal methods retain exact canonical continuity. Read-only
  `bind_execution` and `load_original` may observe one absent-to-present,
  versioned JSON TASK descriptor in the same original run/private namespace,
  provided every other canonical field remains identical. Once observed, the
  descriptor stays byte-identical for that call; removal, replacement, malformed
  scope and unrelated changes refuse. This pointer is continuity DATA and grants
  no object read, permission or continuation; the unchanged full TASK loader still
  verifies immutable contents and native original linkage. Default callers retain
  independent mutable parses. No permission, client, artifact observation or
  mutable hidden map/list is cached, and changed bytes revalidate.

- Normal row exact-byte LRU4 validation caches only deterministic canonical and
  continuation DATA validation with a None result. Every fresh role/table/policy,
  consistent GetItem, raw SHA and contextual ownership check remains active;
  the default path returns a detached JSON parse. Changed bytes and eviction revalidate;
  no policy, native observation, original qualification or mutable DTO is cached.
  Duplicate-key rejection remains in the exact-raw canonical validator. Its
  successful validation precedes each detached decode; the second parse omits
  only the repeated hook and still constructs fresh DATA.

- Normal context optionally retains exact SDK transport routing DATA in a final
  keyword-only field, hash-bound to its existing session namespace on construction
  and every fresh check. Absent transport preserves legacy behavior; private graph
  UUID/native intent bindings remain unchanged. DATA grants no caller permission.
- After confirmed original SFT association and before adapter staging/Create,
  normal publication freshly checks native role/table incarnation and the complete
  owned SFT association. It conditionally writes one canonical <=8192byte UTF8
  normal_prompt_wake_job_v1 record at PROMPT_WAKE_JOB#Training#native-name in the
  selected jobs table. One authoritative consistent full-item Get confirms success,
  an identical conditional race or ambiguous acknowledgement. Explicit service or local permission
  denial, conflicting/unknown readback and current refusal stop effects; no overwrite
  or second Put occurs. Publication neither activates a callback qualifier nor proves
  TASK readiness, replacement fencing, native permission, cold recovery or full O4.

- `claim_completion` requires fresh caller/native/original qualification and exact
  assembled completion/TASK DATA. New closed schema3 stores only the canonical
  completed `sdk_snapshot_sha256`, consumed-ready at epoch1, `invocation_state=ready`
  and no invocation token. The existing authenticated TASK retains the SDK state;
  the sole allowed transition appends one fixed original SFT completion event.
  Every non-message SDK field stays exact and both TASK/snapshot retain160KiB caps.
  A fresh context may reconstruct only exact pre-invocation DATA after current
  checks; it has not won invocation ownership. Legacy schema1 and started schema2/3
  claims refuse cold reuse. Existing READY schema2 rows retain their inline bytes.
- `begin_completion_invocation` uses the existing whole-row raw-body/SHA CAS to
  move schema2/3 READY to epoch2 INVOKING with a unique32hex invocation token. Every
  contender's intended after item differs, so lost acknowledgement resolves only
  its exact complete record. The winner updates its local canonical receipt;
  stale receipts and losing contenders refuse. An attempted CAS spends its local
  attempt, and INVOKING remains closed after loss or unknown external effects.
  Legacy locally owned schema1 retains its process-local one-shot behavior.
- Original completion rule (added 2026-11-11): `normal_run_checkpoint._original_training_complete(observed)`
  is the one rule for "the original SFT is complete and the run may continue from it" on the restore path:
  `trained` is True and either the observed `recipe_id` is `fit_v1` or `full_training_complete` is True.
  `qualify_original_completion`, `app._resume_normal_prompt_body`, `ChatEngine.assemble_normal_completion` and
  `_restore_runtime_research_profile` call it, and the original worker's own wait (`ToolBindingBackend`, kind
  `sft`) accepts the same results. The adapter sets `full_training_complete` for a `full_v1` original alone, so
  a restore check on that flag by itself can never continue a fit pilot. The alias, job name and job ARN
  comparisons, the fresh native observation, every refusal text and `_verified_generation_checkpoint`
  (`full_v1` with its full-training proof) are unchanged. The function sits at the end of the module so that no
  earlier line moves. Why: a replacement session answered `unavailable` for a completed `fit_v1` original
  (0 model calls, no claim, no report) while the original worker finished the same job; found offline on the
  planned pilot shape with one session end during Training.
- Prior-report fact (added 2026-11-09): a confirmed `claim_completion` also records
  `NormalRunContext._completion_prior_report`, True when the claimed row already held the original
  worker's `report_checkpoint_ref` and False when it did not (None until a claim is confirmed). It is a
  private field beside the receipt (the receipt keeps its three strings) and is read only by
  `tui/chat/recovery_notice.py`, which chooses the wording of the runtime recovery record from it. No
  check, qualification or write uses it, and a refused claim leaves it None. Both attributes are installed in one loop
  with a single `object.__setattr__` call site (updated 2026-11-09): `tests/test_normal_sft_cold_continuation.py`
  finds the process-loss cut point of the claim by that one source line and requires `_completion_receipt` to be
  None when it is reached, so a second call site would move the cut.
- Completion qualification repeats current policy/native/original checks and
  compares the whole continuation and immutable TASK descriptor. Schema3 rereads
  the original exact-version TASK under current owned-row checks, derives the fixed
  transition, and matches canonical snapshot digests against both stored DATA and
  the actual load snapshot. Schema2/3 INVOKING requires the winner's local attempted
  token and exact updated receipt; persisted DATA grants no replacement invocation.
  No new object namespace/publication, table, lease, scheduler, automatic wake,
  final-report contract or arbitrary effect replay is introduced.
- Optional normal-run `continuation` retains strict closed schema1/2 decoding.
  Schema3 replaces only inline `sdk_snapshot` with lowercase64hex `sdk_snapshot_sha256`.
  Schema2/3 READY requires epoch1/null attempt; INVOKING requires epoch2/a lowercase32hex
  attempt distinct from its stable claim token. Existing160KiB DATA caps remain;
  malformed/extra fields, mixed versions and phase/epoch/attempt combinations refuse.

- Normal parent read scopes preserve the exact MODEL adapter's underlying
  client identities, which bind its existing strict raw-byte witness. Its own
  `qualified_io(current)` still brackets SDK/body/chunk reads before and after.
  DATA/CASE/PREP retain the external client proxy plus existing adapter scope;
  current role/table/row/catalog/owner/policy and all artifact guards stay unchanged.

- Normal parent per-I/O qualification uses the unchanged live owner/witness,
  PID, research/checker/request-state/precommit/registry and fresh policy checks
  without traversing original producers. Existing full original qualification
  remains before/after capture and brackets the whole parent graph outside its
  adapter I/O scopes. Native role/table/owned-row/catalog checks still bracket
  every actual read/publication; no approval/witness is cached or serialized.

- Size diagnostic swallow (added 2026-11-07): `_causal_call`'s size callback guard uses
  `contextlib.suppress(Exception)` instead of `try/except/pass` (bandit B110, an Info finding of the security scanner). Behaviour is identical:
  an ordinary callback failure never changes the capture decision, and process-control exceptions propagate.

- SFT parent capture per-record cadence (added 2026-11-06): the parent-capture callback is a `RecordRecheck`. Every
  adapter read bracket, publication and `_NativeIO` call still runs the full check (role, table, owned row, catalog,
  owner/policy/witness). Between the records of a lineage stream its local checks (context, catalog, owner/witness)
  run for every record and the native role/table/row checks run once per `_PARENT_RECORD_RECHECK_SECONDS` (1.0) after
  the start of the last full one; a failed check never counts. No approval or witness is cached.

- Repeat training submissions (added 2026-11-07): a store bound to the durable journal answers a second
  `SessionSftStore.submit` before `_fresh_inputs`, the DATA callback and the adapter. `repeat_kind` compares the
  call with the one current-catalog original training record (`_admitted_train`) by its source key (model reference,
  preparation reference, recipe). Same key: `_recorded_submission` returns the submitted record (reused) merged with
  `observe`. Other key: `_already_admitted` returns status `sft_already_admitted`. Legacy stores (no journal) and
  historical or reload keys are unchanged.

- Operation identity of specs, generation and pair (added 2026-11-07): `spec_`, `gen_` and `pair_`
  references are functions of the request, like `sft_` (digest[:32]). `_operation_digest` is a domain-separated
  SHA-256 of canonical JSON. The original spec reference hashes the canonical spec bytes; the resolved spec reference
  hashes the case reference, the resolved bytes and the pinned case manifest and committed references;
  `generation_operation_digest` hashes the session, the deployment catalog, the verified model, cases and optional
  checkpoint, the spec digest, the evaluation id and the arm; `pair_operation_digest` hashes the session, the pair
  catalog, the verified cases, base and tuned generation sources and the spec digest. The conversation, the tool-use
  id, the process and the clock are not inputs, so a replacement worker that repeats a request derives the same
  reference. Distinct inputs give distinct references; there is no replicate parameter, so a second sample of the same
  request is not a supported operation. The in-memory registries are unchanged. A reference that a fresh store has
  never derived still answers `unknown_reference` or the unknown-spec refusal; it never triggers a submission.
  Why: only the SFT reference survived a worker loss, so a replacement that repeated a generation request created a
  second GEN or PAIR job with a new random name.

- Restore-path record cadence (added 2026-11-07): the callbacks that a replacement worker hands to lineage-reading
  scopes (prepared original selection, original TASK selection, completion association, restoration row, original
  association and the hydrate-parent resource check) are `RecordRecheck` objects built by `_record_recheck(check)`.
  `check(True)` is the unchanged full check and runs at every read bracket, publication and `_NativeIO` call;
  between the records of a lineage stream `check(False)` runs instead and skips only the remote steps (the native
  role/table qualification and the run-row read), which run again once `_PARENT_RECORD_RECHECK_SECONDS` (1.0) has
  passed since the start of the last full one. Policy, catalog, owner and context checks run for every record. An
  interval of 0 restores the strict every-record profile (the exact-count decode-budget test uses it).
  Why: one lineage pass yields about 62,000 records, a restore call costs about 0.1 s, and a pass longer than the
  1,920 s body allowance cannot finish.

- Marked run-record rechecks (added 2026-11-10): each of the 13 run-record recheck raises of
  `normal_run_checkpoint` (prepared original selection, original TASK selection, completion association, continuation
  selection, restoration row, parent capture row and catalog, journal enrollment, original association and catalog)
  raises `record_changed(text, RUN_RECORD_CHANGED or SFT_CATALOG_CHANGED)`, still the exact `ValueError` with the same
  text, so every exact-type and text match keeps working; `tests/test_failure_cause.py` pins the set from the source.
  `SessionModelAcquisitionStore.observe` re-raises such an error instead of answering `observation_failed`, so a load's
  own recheck ends the load with its own text. `_warn_normal_model_progress(phase, state, count, cause=())`: a
  `model_artifact_refused` line always renders `exception_class native_code site cause_class` (the closed defaults when
  the cause is not closed), a `model_verification_interrupted` line renders `reason site` and is dropped without a
  closed reason, and every other phase refuses a cause; the other phases keep their exact text.

- Model-pack adoption cadence (added 2026-11-07): `ModelPackAdoptionProvider.adopt`/`restore` build one
  qualifier per call with `_checker(policy, qualify)`, a `RecordRecheck` built by `_record_recheck`. Called, it is the
  unchanged full `_current` (policy, configured scope, the caller's qualifier, STS identity, jobs-table description and
  incarnation, the caller's qualifier, policy, scope) and it runs at every request bracket and object-level bracket.
  Between two chunks of a pack body its local variant runs for every chunk (policy, configured scope, and the caller's own
  qualifier through `record_step`, so a `RecordRecheck` caller keeps its own cadence and a plain caller still runs in
  full), and the full variant once `_PARENT_RECORD_RECHECK_SECONDS` (1.0) after the start of the last full one.
  `SessionModelAcquisitionStore._verified_adoption` builds its qualifier the same way, so each poll
  (`observe` of an adopted ref), the SFT capture (`adoption_hook` and `_adopted_sft_snapshot`: two restores), the cold
  restore (`_restore_adopted_parent`, `_adopted_sft_snapshot`) and `restore_original` check by time, not by chunk.
  The re-hash itself is unchanged by this cadence change (the process pins of 2026-11-07, below, later made a
  repeat restore in one process a HEAD check). Why: see the object body chunk cadence in
  `backends/adapters/AGENTS.md`; the adoption provider added an STS and a table description to every bracket, which
  made the per-chunk cost the largest part of a pass.

- Model-pack restore and process pins (added 2026-11-07): `ModelPackAdoptionProvider._verify_staged` verifies each
  staged adoption object through the copy adapter's `_verified`, so `restore` reads the staged pack (17.99 GiB) once per
  process and checks it by exact-version HEAD afterwards, whichever provider or adapter asks. `restore` still decodes
  and compares the adoption record, the producer attribution, the model inventory and the original model pack
  (`_source`) on every call, and the namespace, `..` and URL-character refusals run before any S3 call. `adopt` verifies
  a copy it has just written through the same call. The read of one status poll, one loop iteration of the after-tool wait
  and one cold restore therefore share the first hash. `_pending` and `SftStore.observe` are unchanged in what they
  require; their `restore_original` and `restore` calls are cheap after the first full pass. The lineage re-qualification
  of `_prepared._verified_sft_snapshot` is not part of this change. Why: see the process pins paragraph in
  `backends/adapters/AGENTS.md`.

- Adopted model parent on the cold restore and claim reads (added 2026-11-08): `NormalRunContext.restore_sft_store`
  and `qualify_original_completion` fence the model adapter of the original SFT only when its model alias is in
  `SessionModelAcquisitionStore._records` (a produced model). A verified pack restored through `_restore_adopted_parent`
  sits in `_adoptions`, has no adapter to fence and is left out of the fenced reads exactly as `hydrate_parent_graph` leaves
  it out; both sites share `_model_read_adapters`. An alias in neither map is still refused with KeyError, never skipped.
  Why: the first replacement of one deployed run raised KeyError at `restore_sft_store` on every attempt, because every offline
  restore test used a produced model parent; `tests/test_adopted_parent_restore_sites.py` builds the adopted parent through
  `hydrate_parent_graph` and covers adopted, produced and mixed stores.

- Captured normal SFT parent publication enters the retained native role/table
  scope before any parent observation. Its same owned row, current catalog and
  live owner policy/witness qualifier bracket every existing adapter read and
  immutable parent publication; denial closes bodies before exposure. The live
  qualifier stays only in the source-owned callback, never checkpoint DATA.
- Effectful normal SFT capture follows the public tool's two required source refs,
  optional recipe (default `fit_v1`) and optional exact `plan_only=False`. Preserve
  the exact raw SDK call/prefix; resolve effective inputs only for validation and
  original native-intent/recipe comparison. Preview, nonboolean flags, unknown
  fields or invalid refs confer no capture authority. TASK schema, byte bounds,
  current owner/policy and all original guards remain unchanged.
- Optional `capture_sft_call(..., agent_snapshot_dict=...)` retains only the actual
  admitted public SDK session snapshot DATA. Fresh TASK schema2 stores the history
  once in that snapshot and derives its causal prefix through strict validation;
  legacy schema1 retains its explicit prefix with exact canonical JSON equality.
  Preflight, prepared and admitted writers share the compact inventory and retain
  the160KiB cap, snapshot state, native/ref bounds and before-effect checks.
  Existing authenticated prepared objects keep their original wire version and
  bytes on admission. Readers never expand the wire before canonical sizing;
  policy, parent/ref authorization, ownership/CAS and no-replay guards remain.

- Optional train-only `causal_parent_ref` is exact DatasetArtifactRef DATA with
  immutable version, lowercase SHA256, positive int byte count<=1MiB and JSON.
  It requires original model/PREP `source_refs` and confines the selected bucket
  to `sessions/<private correlation>/prompt-runs/<campaign UUID hex>/parents/<SHA>.json`.
  Schema6 commits that descriptor into canonical intent/native identity and the
  private record codec. Absent descriptor preserves schema1-5 fields/bytes/name;
  reload cannot acquire one. A descriptor/self-hash grants no reads or effects;
  independent current native/artifact/policy checks remain mandatory.

<!-- verified-model-pack-adopted-sft-parent-contract:begin -->
- Exact `VerifiedSftAdoptedModelSource` separates the consumer-owned model files,
  copied JSON manifests and versioned adoption receipt from its unchanged typed
  original producer. Validation repeats the original validator in the original
  session, then confines current references to the consumer and matches every
  original byte/hash/format pin. Ordinary foreign-session originals still refuse;
  descriptors and aliases grant no reads, approval or native effects.
- Normal parent schema1 optionally carries the closed `verified_pack_adoption`
  model origin and the provider's unchanged canonical snapshot. Capture and cold
  hydration use the owning store's fresh adoption hooks, with live policy and
  qualification repeated on every verification read, including the second
  warm/cold model snapshot verification. The parent live qualifier travels through
  the owning model store; revocation fences the next provider I/O. Adopted restore selects no legacy MODEL
  adapter, handle, worker or download. Missing, changed or refused selected
  adoption propagates; DATA/CASE/PREP and ordinary MODEL guards remain unchanged.
<!-- verified-model-pack-adopted-sft-parent-contract:end -->

- `normal_run_parents.py` owns one closed normal-run parent DATA schema (<=1MiB),
  preserving original model, acquired DATA, frozen CASE and case-inclusive PREP
  aliases/native requests/catalogs/exact artifact versions. Existing producer
  schema2 stays unchanged. Temporary hydration uses current policy/resource/native
  qualification and read-only body guards; no partial store is exposed, and no
  policy/client/witness/pickle or new producer is reconstructed from JSON.
- Qualified reads are private to their context (added 2026-11-07):
  `normal_run_parents._qualified_reads` no longer replaces `_sm` and `_s3` on the shared adapter objects. It enters
  `scoped_native_clients.scoped_clients`, which keeps the read-only `_NativeIO` view in a `ContextVar` and serves it
  through data descriptors on the adapter class (installed once, idempotent; the raw client stays in the instance
  `__dict__` under the same name). Only the context that entered the scope sees the view. The tools of one
  assistant message run on `asyncio.to_thread` workers while a later tool's BeforeToolCall fence qualifies inline
  on the event-loop thread, so those tools keep their write-capable clients; overlapping scopes cannot leave a
  proxy behind; a nested scope wraps the outer view and both qualifiers run. A worker that inherits the context
  (`to_thread`, `copy_context().run`) inside a scope sees the read-only view; a plain `threading.Thread` started
  inside a scope would see the raw clients, so no thread is started on the qualification path. The
  `ProcessingModelWeightAcquisitionBackend` exception, `qualified_io`, the `_NativeIO` allow-list and the
  `RecordRecheck` cadence are unchanged, and the fence still qualifies with read-only clients only.
  Why: after a cold replacement `reload_qwen_sft` issued beside another tool was rejected in 22 of 26 calls
  (`AttributeError: create_multipart_upload`) because the fence of the other tool had swapped the clients for
  that moment. Tests: `tests/test_scoped_native_clients.py`.
- Selected precommit uses `NormalRunContext.capture_sft_call` only after actual
  argument admission/current source custody. Its bounded copied prefix and unique
  SFT use/aliases precede parent publication and native intent construction. A
  native-bound exact parent ref lives under the original private UUID/run namespace;
  uncaptured ordinary SFT preserves legacy descriptor absence and observation-only
  behavior. Caller witness qualification never follows from role selectors or DATA.
- `save_admitted_task(..., original_job_name=..., policy=...)` stores the admitted
  public SDK snapshot DATA (not an SDK TASK preset) with the frozen TASK envelope,
  exact native original/parent/call linkage and <=160KiB bound. Tasks use versioned
  content keys and original-row CAS; uncertain writes reconcile only their exact
  result, never repeat. `load_qualified_task(policy=...)` qualifies the accepted
  native original, full parent graph and causal snapshot before returning its
  internal tuple. It neither invokes an agent nor consumes completion; completion
  CAS, fresh input admission and SDK continuation remain caller-owned gates.

- Optional private training `source_refs` is exactly the verified model/preparation
  alias tuple. Schema5 commits it into original intent/digest/native job identity;
  pending lookup matches the companion pair against that accepted original.
  Absent refs retain legacy schema2-4 bytes/identities. Partial/malformed or mixed
  reload refs refuse. This DATA is no policy/grant or independent upstream
  publication; existing source/native/artifact/current-policy checks remain.

- `normal_run_checkpoint.py` owns normal-run enrollment, prospective original SFT
  association and verified pending status DATA. `prepare_normal_run` validates server-selected runtime role ARN/ID,
  existing CampaignStateCatalog account/Region/deployment/table and SDK-session
  hash without SDK calls. Both role fields absent preserve legacy; partial or
  invalid selection refuses before model work. Scoped ContextVar DATA resets in
  finally, carries no client/policy/grant, and requires the original process/config.
- Actual admitted prompt/history and original acquisition UUID enroll one
  `PROMPT_RUN#<32hex>` row before model dispatch. Plain bounded canonical JSON
  rejects live objects, duplicate keys and credential/policy fields. Fresh native
  STS assumed-role/RoleId and exact active table ARN/name/PK/incarnation checks
  bracket DynamoDB I/O; sequential checks do not prove atomic IAM admission.
  SDK calls use one attempt/connect5/read20. Conditional Put never blindly repeats;
  lost acknowledgement resolves only the exact original via a consistent read.
- One original SFT alias/intent/digest/job association is conditionally committed
  before adapter staging/Create, retaining the original TableId and graph UUID.
  The scoped existing journal obtains fresh native scope around each I/O; an
  unsupported generic observer refuses rather than creating an unqualified client.
- Passive and explicit resume lookups validate the exact owned association and only
  an ACCEPTED original with a complete staged snapshot. Existing SFT restore rehashes
  exact versions, RUN/count and native request/tags. Current catalog, role, TableId,
  association and fresh resume policy qualify every read; UNKNOWN/ambiguous/held or
  changed/missing records stay unavailable. Pending projection starts no new goal,
  model, job, staging, Stop or SDK task and keeps lifecycle/quality flags false.
  IDs/hashes/DTOs remain DATA, not authority or full parent provenance.
- `SessionSftStore.bind_durable` binds one empty original store under its lock using
  constructor-equivalent typed scope checks and a canonical UUID correlation. Prior
  bindings/callback or any records/keys/source refs/history refuse; all graph object
  references remain unchanged. The optional callback defaults to None for legacy.
  `before_submit(model_ref, preparation_ref, recipe_id, inputs)` follows fresh input
  verification and precedes adapter submission under the same lock. None retains
  legacy omission; a typed versioned `DatasetArtifactRef` is forwarded only as
  `_causal_parent_ref` DATA for the original intent codec to validate before staging
  or Create. Callback errors/untyped values refuse; descriptors carry no authority.
- Durable SFT observation uses a freshly restored handle only for original native
  and artifact verification. The captured original record object and producer pin
  remain unchanged. Under the store lock, the same original entry must still exist
  before merging only `submission_state`, `restoration_status` and `durability_status`;
  an accepted original is never downgraded by unresolved restoration. Substitution
  refuses before completion exposure and cannot overwrite the changed entry.
- Known-pending observation is not execution continuation or native R1-R4 proof.
  Full independent parent/source/case publication, admitted causal history, leases,
  lost-ACK recovery, idempotent advanced enrollment, completed-result consumption,
  automatic wake and final durable report remain required next slices. No prototype
  factory or self-hash is manufactured as a publication/authentication witness.

- Current runtime qualification rechecks the retained protected/coordinator table
  selection before and after its native reads, including primitive IO callbacks.
  A qualifier does not renew an existing lease or witness: coordinator rechecks
  both after it and immediately before Update/native observation. Pre-Put denial
  creates no attempted pending marker; reset can make one ordinary qualified Put,
  while genuinely attempted unknown writes keep their original reconciliation.

- Optional initializer/coordinator `qualified_io(callback)` uses a PID-bound,
  per-instance scoped ContextVar, resets finally and preserves legacy None. Current
  callbacks run before actual SDK reads/writes/native observation, after responses
  and before returned data; denial is outside SDK-attempt catches. They return no
  permission value/cache and do not establish an atomic IAM snapshot.
- Runtime binds the retained protected table account/Region/name/ARN/incarnation
  and validates full original wake scope/kind/job before lease mutation. It requests
  `observe_wake(..., fresh_observation=True)` so identical-event historical status
  cannot become fresh readiness; defaultFalse retains legacy duplicate semantics.
  `context.qualify_current_io()` is a fresh ingress/native/protected callback with
  no DDB recursion/grant result; independent model-intent publication checks remain
  the journal owner's responsibility. `require_current()` still resolves C+H.

- `campaign_runtime.py` owns opt-in private normal-entry/context routing over the
  existing exact AdmissionGuard/protected initializer/coordinator. Independently
  approved ingress configuration is conditional server input, never native proof;
  fresh private observation must actually verify ingress and new/redelivery/wake
  disposition. No ContextVar/type/session/header/STS/self-hash grants identity.
  Missing factory keeps legacy local behavior; configured unknown/drifted/expired
  native/ingress/protected qualification refuses before original entry exposure.
- Runtime context holds sealed PID-bound original protected ref/UUID/private
  correlation/literal and bounded planning-ref DATA. Current methods obtain fresh
  ephemeral witnesses/protected comparisons; no policy/witness/client is persisted.
  Scope activation always resets; fork/clock/type/issuer/request mismatch refuses.
  Selected terminal verifier receives a transient fresh policy snapshot outside
  context/model input. Actual model pre-effect intent/attempt journals remain with
  their separate backend/store owner; REGISTERED is not an exactly-once marker.
- A private cold wake returns awaiting_original for UNKNOWN/running/stopping,
  failed/stopped or unverified/unresolved artifact checks, with no agent call.
  Only fresh Completed plus matching private closed original-artifact verification
  returns a continuation context. No JSON/assistant claim can enter that verifier;
  no native, containment, grading, quality or effect permission follows from type.
  Public wake parsing/authenticated transport and ingress deployment qualification
  remain independently required; no callback/CDK/source activation is supplied here.

- A known write retry violation survives declared final protection/admission
  qualification failure as `InitializationTransportError`, chaining that failure.
  Its confirmed original DTO is historical no-grant data only; the error's
  `currently_qualified` property is permanently False. It cannot turn a failed
  final check into success, noncommit, replay or active exposure. Unexpected
  exceptions retain existing propagation; no broad catch grants permission.

- Initialization request construction/SDK validation precedes final protection
  qualification. After the final native AdmissionGuard read, recheck the original
  evidence deadline and independently retained configuration before dispatch or
  return. An unattempted pending candidate must match the current literal prompt,
  scope/reference and token before any write; refused old candidates never commit
  under a changed prompt. Sequential checks do not claim atomic native admission.
- Contradictory write retry metadata is a transport qualification refusal even if
  one outcome TxGet confirms the original pair. `InitializationTransportError`
  retains only a frozen no-grant original observation, or None with an unresolved
  read cause; it never declares absence/noncommit or replays the mutation. Fresh
  read-only resolution rechecks current boundaries. S/B size counts are exact;
  N storage uses a conservative bound and each item stays <=400KiB.

- `campaign_initialization.py` is a source-only atomic initialization protocol in
  the existing campaign table. One conditional two-item transaction creates the
  original `CAMPAIGN#<selector>` and separately protected immutable
  `CAMPAIGN_HANDOFF#<selector>` binding. Original prompt, campaign/store UUIDs,
  admitted scope and 138-byte reference survive the first confirmed commit.
  One transactional outcome read resolves lost acknowledgement/concurrent winner;
  an unresolved attempted selector never replays its mutation. H-only deleted-C
  and C-only legacy orphan refuse reconstruction/adoption. Advanced C retains the
  same original H commitment without requiring initial empty-C bytes.
- Initialization approval is explicit independently reviewed server configuration,
  never client/header/native metadata approval. Its private current protection
  reader returns untrusted configuration, finite observed/expiry times, exact
  construction PID and reader-object binding. Each TxGet/TxWrite and returned
  result obtains a fresh reader call, compares retained original configuration
  bytes, and rechecks current AdmissionGuard before and after that read. Clock
  rollback, expiry, inherited context, changed config, missing/unknown evidence,
  retried transports and malformed response/absence refuse. This is sequential
  read qualification, never an atomic IAM/protection snapshot or action grant.
- Native table/role capability confinement, independent H namespace protection,
  retention and publication permission remain separately unproven. Different SDK
  clients do not prove credential isolation. This library supplies no app/API
  activation, producer graph/effects, automatic redelivery or exactly-once AWS
  execution. `campaign_coordinator._initial_campaign_item` and static `_body` are
  pure additive helpers; legacy coordinator runtime behavior stays unchanged.

- Separate planned, approved and executed results; Plan Mode must not invoke the executor.
- Explicit unknown cost is `None`, never zero. Capped automatic policies deny
  unknown quotes; finite nonnegative numeric quotes retain the existing rules.
- `acquisition_state.py:SessionAcquisitionStore` holds typed native records and
  opaque refs for one conversation, independently of approval policies. Its
  in-memory lifetime is not process-restart durability or a recovery credential.
- Its narrow collection-agent delegation adapter supplies fresh hidden policy
  and typed store state through each child SDK invocation, without model input
  grants or cached authority. Other delegated-policy paths remain separate.
- `_verified_preparation_source` uses one fresh verified acquisition observation
  plus original private source/image pins and exact-version receipt/commit/lineage
  refs confined to the owned job prefix. It returns a frozen no-grant DTO; foreign
  sessions, caller-authored records, missing versions/hashes and pending jobs refuse.
- The same store privately snapshots verified acquisition bytes/provenance for
  inspection and retains opaque inspection handles. References select records
  only within that conversation; neither store record contains approval grants.
- Spend/irreversible boundaries retain the existing approval and budget checks; each caller must use the documented capability path.
- Default AWS mutation scope follows the selected deployment prefix at call time.
  Explicit owner resource-prefix overrides retain precedence.
- `aws_resource_policy.py:resolve_control_region` — Resolve the same selected region used by every AWS control tool.
- `aws_resource_policy.py:ResourceControlPolicy` — The configured guard (prefix + region pin + allowlist override).
- `capability.py:register_capability_executor` — Register an explicit executor callable for ``flow_id`` (last write wins).
- `capability.py:resolve_executor` — Resolve the executor callable for ``flow_id`` (registry → override → convention).
- `preparation_state.py` defines the frozen no-grant `VerifiedPreparationSource`
  from Root's one-observation private acquisition accessor, plus the private
  deployment `PreparationCatalog` and conversation-local `SessionPreparationStore`.
  Never reconstruct source/native authority from a public dict or model handle.
  Validate image/source/admission and exact session/profile before source reads;
  idempotent same-selection submission retains native handles and opaque refs.
  Records contain no policies; polling is read-only and never resubmits. This
  in-memory store is not process-restart durability or a recovery credential.

- `preparation_binding.py:build_preparation_store` snapshots explicit ENV/runtime
  overlay only; construction makes no SDK/SSM read and invents no required resource.
  Missing startup fields retain a typed UNADMITTED catalog. Fresh submission returns
  `configuration_missing` with bounded selector names only before source reads/SDK;
  malformed fields, mismatched admission and mutable/foreign versions remain refused.
- The exact `data_preparation_flow` action is expensive. Its public flow self-gates;
  the capability override reaches only the inner executor after the authoritative
  outer caller gate. Unknown quote and presentation remain None; capped automatic
  policies deny it. Only an actually accepted handle yields executed=True.
- `_trusted_preparation_store` is a private capability channel, separate from public
  params/plans. Public prep params are acquisition refs and fixed semantic selectors;
  model-authored store/policy/catalog fields refuse. Introspection excludes these
  internal parameters, and JSON results never serialize private state or reprs.
- Collection delegation carries a preparation store only when it references the
  same typed acquisition store/session. Every invocation gets a fresh policy copy;
  stores retain records, never approval grants. No process-restart durability claim.
- `model_acquisition_state.py` owns typed private model-fetch handles and opaque
  refs per conversation, independently of caller approval. Cross-session, dict
  handles and model-authored records refuse. A private consumer obtains a fresh
  parent-verified versioned snapshot; this transports bytes without trust/grants.
  The in-memory store is not process-restart durability or native admission.
- Owned files: `normal_run_checkpoint.py`, `normal_run_parents.py`, `protected_planning_read.py`, `campaign_planning.py`, `campaign_runtime.py`, `campaign_initialization.py`, `campaign_native_admission.py`, `campaign_admission.py`, `campaign_coordinator.py`, `campaign_state.py`, `__init__.py`, `approvals.py`, `aws_resource_policy.py`, `capability.py`, `cost_confidence.py`, `cost_governance.py`, `custom_commands.py`, `guardrails.py`, `mcp_bridge.py`, `mention.py`, `plan_mode.py`, `project_memory.py`, `runtime_session.py`, `token_cost.py`, `traceability.py`, `preparation_state.py`, `preparation_binding.py`, `model_acquisition_state.py`.

- Collection delegation forwards a model store only when it is typed and its
  runtime session matches the typed acquisition store. Public child input cannot
  supply grants; missing/malformed/foreign state is omitted, policy stays ASK.

- `preparation_state.validate_source` admits canonical JSONL as before and only
  exact pinned nvidia/Open-SWE-Traces v1.2 supported Parquet names for the closed
  derived profile. Preserve private source type/session/account/region/deployment,
  owned URI, native versions/hash/size and manifest checks. Format metadata is not
  caller authority; no CSV/generic Parquet/profile/policy/role bypass is accepted.

- CommitPack selection is inferred only from private exact repo/revision/file/
  JSONL byte-count/hash refs. Recognized altered pins cannot fall back to legacy.
- `provider_disclosure.py` (added 2026-11-06) reports from the live process which provider
  answers a flow's SageMaker operations. Controlled is reported only for a live, well-formed
  installation whose scope covers every operation the flow uses; every other state, including
  a failed check, is `unverified` with a static reason. It is recomputed on each call, never
  cached, never read from a caller, and grants nothing: approvals do not branch on it.
  `SessionPreparationStore.submit(profile_family=...)` (added 2026-11-06) is
  the flow's check that a named CommitPack profile matches every private source;
  it runs after catalog validation, before the reuse return, and grants nothing.
  Initial private catalog validation permits the fixed maximum8MiB exclusion
  envelope; trusted sources narrow legacy back to2MiB before adapter submission.
  Public policy/session/grant and five-field artifact contracts remain unchanged.


- `sft_state.py` owns frozen no-grant source DTOs, SftCatalog and conversation-local SessionSftStore. The private original model/preparation accessors make one fresh verified artifact observation after same-session checks; public result dictionaries never enter authority. Model files preserve source expected bytes, preparation preserves five-field refs and original counts; exact COMMITTED metadata supplies unchanged four-field RUN split descriptors. Store holds native handles and operation keys only, no policies; concurrent duplicates reuse one original native identity, including ambiguous acceptance. No process-restart durability is claimed.

- Only the original private preparation catalog supplies exclusions; its exact version/hash/count may be bucket-global in the same owned bucket. Model/preparation payload confinement remains same-session; original COMMITTED exclusion SHA must agree. Prefix grammar follows the selected lowercase1..40 deployment contract.
- `sft_binding.py:build_sft_store` is pure startup ENV binding over the exact
  original typed model/preparation stores and their shared session. It creates no
  new correlation UUID, performs no SDK/SSM/source observation and stores no policy.
  Nine required catalog selectors have no resource/admission defaults; missing
  fields stay typed UNADMITTED and submission reports bounded configuration names
  before source/SDK effects. Malformed selector types refuse without coercion.

- Fixed SFT submit/reload/flow actions are EXPENSIVE_JOB. Delegation carries `_amlas_sft_store` only when its private model/preparation objects are the same typed original conversation objects; each invocation gets a fresh caller-policy snapshot.

- `campaign_state.py` owns the closed server-only operation intent/record codec.
  A complete canonical digest binds execution role incarnation, deployment,
  campaign/session, fixed trainer catalog/source/image, original private
  model/preparation descriptors and counts, and fixed recipe/operation/checkpoint.
  Server bindings are minted only after the adapter's native role verification;
  caller JSON, persisted descriptors and opaque handles never grant authority.
  Restored closed DTOs require fresh original native/artifact validation before
  consumption and a fresh caller policy. This transport authenticates deployment
  execution-role scope, not an end-user identity or completed campaign recovery.
- Preparation may bind an opaque frozen case from the exact same acquisition/case store. Resource validation precedes source reads; fresh private case verification supplies only original immutable exclusions. Per-job catalogs and dedup keys retain case COMMITTED/exclusion/producer identity; startup exclusion bytes are unnecessary on this path. Legacy preparation retains complete startup-catalog validation.

- VerifiedPreparationSource is imported/re-exported from data-only backend types so fixed case workers do not load conversation/SDK state. Class identity and closed fields remain unchanged.

- Durable SFT injection is optional and server-only: `SessionSftStore` receives
  the complete campaign backend/binding pair and forwards it per adapter call,
  with no policy or shared-adapter context cache. Uninjected state remains
  conversation-local. `resume_original` needs original same-store source refs;
  fresh private model/preparation observations and exact current catalog/intent
  equality precede native/artifact verification. Missing upstream state is
  unresolved. Durable opaque refs derive from the immutable operation digest.
- `SftSubmissionSnapshot` is a closed no-grant request/RUN/staged-version/count
  DTO. `fixed_sft_request` reproduces only the fixed native request without SDK
  effects. Campaign schemas2/3/4 attach that snapshot once while CREATE_ATTEMPTED;
  original operation digest remains unchanged. Schema1 reads stay descriptive.
  Conditional transitions bind exact prior private bytes, so an outcome update
  cannot replace staged evidence. No runtime/upstream restoration, end-user
  authentication or native durability is established by this source interface.

- [Added 2026-11-10, preference lists] `sft_state.py` pins three ordered `ResourceConfig.InstancePreferences` lists, each one instance with 100 GiB: `TRAIN_INSTANCE_PREFERENCES` (full and fit training) is ml.g6e.16xlarge, 12xlarge, 8xlarge, 4xlarge; `RELOAD_INSTANCE_PREFERENCES` (new reload contract `reload_instance_preferences_v1`) is ml.g6e.12xlarge alone, sent as a one-entry list; `GENERATION_INSTANCE_PREFERENCES` (base and tuned generation, `generation_state.fixed_generation_request`) is ml.g6e.16xlarge, 4xlarge, 8xlarge (reload and generation lists changed later on 2026-11-10 and again on 2026-11-11, see the next two paragraphs). ml.g6e.2xlarge is in no list (64 GiB host memory) and generation omits 12xlarge (four GPUs change the generation server's device layout). `fixed_resource_config` accepts only these three tuples, so a list with other members or another order refuses; `validate_submission_snapshot` accepts a training snapshot that equals the preference request or the earlier fixed 12xlarge request (`fixed_sft_request(historical=True)`), nothing else, and a reload snapshot must equal the request its recorded contract reconstructs. Historical requests stay exact: reload schemas 1 to 3 keep the 12xlarge request and `reload_single_gpu_v1` the 4xlarge one. Needs botocore 1.43.90 or later (lock 1.43.127). Why: the account allows one training job per ml.g6e type; fixed-type requests got no capacity 4 of 4 times on 2026-11-10, while preference lists got capacity in 61 s to 37 min in the recorded trial jobs (SageMaker selected 16xlarge for training and reload, 4xlarge for a generation). The trainer and reload use cuda:0 only, so the recipe does not depend on the selected type. The second paragraph below describes the earlier fixed reload contract, which existing records keep.
- [Added 2026-11-10, quota rule and list split] SageMaker refuses a create when ANY requested type is at the account limit of one training job per ml.g6e type, and its message names that type (CloudTrail CreateTrainingJob in a test account, us-west-2, 2026-11-10: a list of 12xlarge, 16xlarge and 2xlarge was refused naming 2xlarge while 12xlarge and 16xlarge were free, and the same request was accepted 2 s after the fixed 2xlarge job stopped). A fixed-type job holds its type from the create on, also while it is Pending. A list job that has not selected a type holds nothing: a fixed-type create and a list create were accepted while such a job waited. Not measured: a create that lists a type which a list job has already selected; the lists and the test stand-in (`tests/test_capacity_preference_lists.py`, `Account`) take the selected type as held. What follows for the lists: the fine-tune runs alone and keeps four types. A rejected fine-tune or reload is never submitted again (unchanged), and the agent may submit the reload and a generation in either order in one message, so the reload list and the generation list share no type (16xlarge, 12xlarge and 4xlarge, 8xlarge when this paragraph was written; 12xlarge and 16xlarge, 4xlarge, 8xlarge since 2026-11-11, next paragraph). Base and tuned generation share a list, so the second is refused while the first holds its type and is submitted again after it ends. Operating rule: until a run's fine-tune is created, no other training job of the account may hold ml.g6e.16xlarge, 12xlarge, 8xlarge or 4xlarge, and until its reload is created none may hold 12xlarge (16xlarge or 12xlarge before 2026-11-11); a fixed-type job on one of them rejects that create for good. The four-type reload list and the three-type generation list of the first port are refused now.
- [Added 2026-11-11, reload and generation lists changed] `RELOAD_INSTANCE_PREFERENCES` is ml.g6e.12xlarge alone and `GENERATION_INSTANCE_PREFERENCES` is ml.g6e.16xlarge, 4xlarge, 8xlarge in this order; `TRAIN_INSTANCE_PREFERENCES` is unchanged. Why: in the full run the stages after the fine-tune have no wake path and must finish inside one 8-hour session, and the two generation jobs run one after the other, so generation was the capacity bottleneck with the list 4xlarge, 8xlarge. Measured in this account since 2026-10-01: list jobs that received a machine selected 12xlarge 6 times and 16xlarge 4 times; 4xlarge was delivered 5 times after waits of 69 to 14,985 s and missed in three 20-minute checks; 8xlarge has never been delivered. The rules of the paragraph above are unchanged: one training job per ml.g6e type, a create is refused when ANY listed type is at that limit, a refused fine-tune or reload is never submitted again, a refused generation is submitted again after the job that holds the type ends, the reload and the base generation may arrive in one message in either order, and generation lists no 12xlarge (four GPUs change the generation server's device layout). The two lists still share no type and together hold the four training types. In every interleaving of reload and base generation followed by the tuned generation the only refused create is the tuned generation while the base generation holds its type, and it is submitted again (`tests/test_capacity_preference_lists.py::test_every_interleaving_of_reload_and_base_then_tuned_is_created_or_submitted_again`, 24 cases). Shape of the one-type reload: still `ResourceConfig.InstancePreferences`, with one entry, built by the unchanged `fixed_resource_config`; the contract name `reload_instance_preferences_v1` is unchanged. The service model of the locked SDK (botocore 1.43.127, `InstancePreferenceList` min 1, max 5) and the API reference (read 2026-11-11) allow one entry, and the list shape keeps `SelectedInstanceType` and `SelectedInstanceCount` in the reload's status and artifacts through the one readback path. The historical fixed-type shape would equal the contract-free 12xlarge reload request of record schemas 1 to 3 and name no selected instance. Not yet on record: a real create with a one-entry list; the account's 14 list jobs since 2026-10-01 carried 2 to 5 types. Send this exact ResourceConfig in a capacity check before a real run, because a refused reload create is not submitted again. Operating rule now: until a run's reload is created no other training job of the account may hold ml.g6e.12xlarge. Records: a `reload_instance_preferences_v1` record or a generation original written with the lists of 2026-11-10 (16xlarge, 12xlarge; 4xlarge, 8xlarge) no longer equals the request its contract reconstructs and does not restore; the same held for the first port's lists after the list split of 2026-11-10. No schema, contract name or readback rule changed.
- [Added 2026-11-10, pending time] The fixed training, reload and generation requests carry no `StoppingCondition.MaxPendingTimeInSeconds`, and that is deliberate. The API does not need the field with `InstancePreferences`: the installed service model (botocore 1.43.127) and the API reference list it as not required, 1,800 to 2,419,200 seconds, with no stated default, and a list request without it was accepted on 2026-11-10 (it waited 1,052 s Pending and then selected ml.g6e.12xlarge). With a list the field bounds the total time SageMaker spends on the whole list. The default is not documented. Measured once in this account: a fixed-type ml.g6e.12xlarge training job without the field stayed Pending from 2026-10-20 to 2026-10-22 (48 hours), and SageMaker then stopped it with the transition message "Training job pending time exceeds max pending time for capacity. Specify max pending time parameter in CreateTrainingJob request to set or increase this value". That is six times the longest wait that ended in a start (28,840 s) and twice the fine-tune's `MaxRuntimeInSeconds` of 86400. Not measured: the default for a list request (the two list requests without the field ended after 1,052 s and 1,211 s). The recorded trial list jobs sent 432000 and waited at most 2,219 s. Adding the field changes the pinned request bytes (`StoppingCondition` is compared exactly in the native readback and is part of the stored submission snapshot), so it is added only when a wait longer than the default is wanted or when a list request is seen to stop sooner; a job stopped for pending time reads STOPPED and is reported like any other ended original.
- New ordinary reloads pin private `reload_single_gpu_v1` in schema4 before
  reservation/digest/Create: one ml.g6e.4xlarge with100GiB. The selector is reload-only
  and part of the original operation identity; nullable compatibility remains
  descriptive and freshly selected. LegacyNone is omitted from intent bytes and
  schemas1/2/3 retain exact12xlarge request reconstruction and historical identity.
  Training remains unchanged. Request/body fields never select a resource contract;
  missing/unknown schema4 selectors or original request mismatch refuse.

- `generation_binding.py:build_generation_store` constructs the grant-free generation store over the exact original conversation model/case/SFT stores, with explicit image/source/admission selectors and no SDK startup call. Missing configuration stays unadmitted. `generation_state.py` freezes bounded researched SpecModel bytes plus the actual canonical specification hash and returns a same-session opaque spec reference. Shared base/tuned evaluation identity derives from private original case/model/spec pins. Opaque submit/poll refs retain original native handles, including unknown acceptance, without policy caching or restart recovery. `_verified_evaluation_source` freshly verifies original generated outputs into frozen `VerifiedCodingGenerationSource`; neither this DTO nor public result flags grant grading/containment authority. `SessionSftStore._verified_generation_checkpoint` freshly observes only the original same-model full training record.
- `coding_generation_flow` and `submit_coding_generation` are EXPENSIVE_JOB; unknown GPU costs remain None and capped automatic/ASK policies deny before source reads or Create.

- Generation spec binding reuses the opaque reference for identical canonical validated bytes. Submission deduplication keys those bytes rather than a caller-visible spec alias, retaining one original accepted/unknown job for equivalent same-session selections.

- `resolve_case_spec` mints a new immutable private-case-bound specification and preserves old bytes/hashes. Both generation arms require that resolved same-session reference; public manifest claims cannot substitute for fresh original case records. `paired_state.py` retains opaque accepted/unknown original Processing handles over the exact generation/case objects; equivalent selections never resubmit. `paired_binding.py` snapshots explicit image/source/admission and launcher-SHA selectors without SDK effects. The frozen pair-only `PairedCodingCatalog(execution: SftCatalog, launcher_sha256: str)` composes the unchanged shared execution catalog; it grants no action authority. Missing launcher configuration stays deferred/unadmitted until submission before upstream/SDK reads, and malformed selectors refuse before registry resolution. Fresh policy remains only in hidden invocation state, not any store or JSON artifact.

- `producer_snapshot.py` owns one closed schema1 model/preparation projection row
  keyed by the complete original SFT operation digest in the selected jobs table.
  Canonical bytes bind original opaque aliases, role incarnation, campaign/store
  correlation, catalogs, requests, headers, sources and counts. Only explicit
  projections persist; adapters, clients, byte witnesses, cached results and policy
  never do. One conditional Put uses SDK mutation retries1; conflict or lost
  acknowledgement requires consistent exact-byte readback, otherwise unresolved.
- `ProducerSnapshotRepository.load_original` verifies the existing role-scoped SFT
  reservation and complete producer row before minting a sealed server-library
  `SftCampaignContext`. `activate_sft_campaign` transports only this typed context;
  it is not AgentCore transport/end-user authentication or callback authorization.
- Original correlation/model/preparation restore hooks populate empty temporary
  typed stores without reminting refs. Hydration reconstructs fresh fixed adapters
  from current startup settings, rechecks fresh caller approval before every native
  artifact read, verifies both producers and existing `resume_original`, then
  exposes the whole graph only with explicit verified original restoration.
  RUNNING/FAILED/STOPPED remain honest observations; completed jobs require
  verified artifacts. Swallowed UNKNOWN/verification failures and any temporary
  denial latch refuse, with a fresh final policy check. Factory-returned client
  pairs are guarded, then each adapter's original caching lifecycle is restored.
  Per-call checks cover model/preparation/SFT SDK reads; campaign STS/DDB metadata
  reads use entry/final policy checks, not a universal per-RPC claim. Stores retain
  no policy. Missing/ASK/revoked authority, schema/scope/source drift
  and incomplete projections refuse; restoration never calls Create or stages.
- Activation and hydration are PID-bound; an inherited fork context refuses.
  Injected real botocore DDB clients must have effective total_max_attempts1 before
  publication. Neutral fake clients retain their closed test transport contract.
  Context decode also binds producer identities/headers to reserved SFT inputs,
  preparation header image/source to catalog, and nested acquisition provenance
  to owned native ARN/image/job/bundle paths with exact manifest roles/versions.

- This foundation does not recover before producer publication, restore case/spec/
  generation/pair records, acquire campaign leases or own durable wakeups. Existing
  SFT schema1/2 and missing-upstream negatives remain unchanged.

- `publish_restore_aliases_from_store` adds a separate immutable producer schema2
  row/key alongside unchanged schema1 projections and native reservations.
  Alias-complete publication requires one original non-case preparation selection:
  sorted unique1..256 acquisition aliases bound to exact captured source indices.
  Selected reload additionally binds the actual original train alias/name/full
  operation digest to its original checkpoint producer and same private sources.
  Missing or case-bound provenance refuses; no alias/parent reminting occurs.
  Malformed schema2 never downgrades to schema1; absent schema2 retains valid
  historical descriptive/read-only loading without alias-completeness claims.
- `load_original(..., expected_snapshot_sha256=...)` requires the independently
  retained original publication SHA from a qualified private server handoff for
  every present schema2 row, before context minting or hydration. Compare actual
  canonical bytes; missing/malformed/unequal commitment refuses without downgrade,
  and a supplied commitment cannot select an absent schema2 row. No commitment
  plus absent schema2 preserves schema1 descriptive loading. The mutable row's
  own SHA, public/model input and client metadata cannot supply this commitment.
  The caller must qualify its origin separately; a digest supplies no grant and
  this library does not implement durable app/callback handoff qualification.
- Alias hydration verifies the original train before selected reload, with fresh
  original reservation/client/artifact/checkpoint and caller-policy checks.
  Only verified temporary stores receive the original preparation selection and
  normal `(train_ref, reload)` lookup key, then the complete graph returns.
  Serialized metadata, successful replay and restored keys supply no native
  admission, quality or successor effect grant. Case/full-graph/app/native gates
  remain separate and open.

- `campaign_admission.py` compares an explicit trusted-server approved co-owner
  manifest with fresh closed native metadata. Native projection, account ownership,
  runtime STS, session/user headers and public dictionaries never approve a cohort.
  Blank/missing policy evidence remains UNKNOWN and refuses before campaign reads.
  Its sealed issuer/PID/monotonic-time witness is bounded to sixty seconds, checks
  exact raw policy/hash/role-incarnation/trust/qualified-endpoint/version pins and
  revalidates before each coordinator lookup/CAS/observer and after observation.
  No witness or grant is serialized/cached. Sequential native reads are not an
  atomic IAM snapshot; operator baseline approval/native readers remain live gates.
  A concurrency-safe process-local numeric high-water mark rejects any observed
  clock decrease; it stores no witness/grant. AdmissionGuard binds its construction
  PID and refuses inherited use before any clock lock, approved-data access or
  reader I/O; a child constructs an independent guard without resetting/reusing
  inherited locks or grants. Policy JSON traversal rejects numeric
  overflow to nonfinite floats as well as literal nonfinite constants.
- `campaign_coordinator.py` owns one bounded canonical existing-table row per
  approved scope/owner/transport mapping. The literal prompt/hash and independent
  server campaign/store UUIDs are immutable; opaque refs select records, not grants.
  Refs are exactly138ASCII bytes: campaign_<selector64>_<originalbinding64>. Every
  reference/lease/wake/reconciliation load checks the independently retained
  original commitment, not only the row self-hash. Same-process pins retain data
  only. Cold existing transport-only opens and unqualified conflicts refuse; cold
  expected_campaign_ref must come from a separately qualified original handoff,
  never that mutable row or model payload. Legacy refs have no automatic upgrade.
  Conditional Put and revision/epoch/exact-body CAS make one mutation attempt;
  lost replies require consistent exact original bytes, never blind retry or
  conflict overwrite. Fresh-admission load resolves exact pending acknowledgements
  read-only after lease expiry, clearing local pending bytes only on whole-item
  equality; later work requires a new lease epoch. The coordinator construction
  PID is checked before locks/I/O; child processes construct independent holders.
  Sealed PID leases fence only DDB commits, not external AWS
  effects. Registered original Training/Processing descriptors alone route typed
  observation hints; payload status/policy/owner/stage overrides refuse. UNKNOWN
  retains original identity/last-confirmed state and no observation grants native
  admission, quality or successor dispatch. Reload/SFT/base/tuned generation use
  Training; acquisition/model/case/preparation/paired evaluation use Processing.
  This library foundation has no app/callback/factory/graph wiring, prepublication
  crash recovery, real table/native admission or full campaign acceptance.

- `campaign_native_admission.py` supplies a constructor-PID-bound native READ
  adapter over exact unapproved server selectors and injected one-attempt SDK
  transports. GetRuntime/version, GetEndpoint/liveVersion, native deployment tags,
  runtime/endpoint policy and selected IAM role incarnation/trust are read afresh
  sequentially. HTTP must be explicitly reported; omitted protocol evidence never
  acquires a compatibility default. SDK validation provides structural checks,
  not every modeled pattern. Consumed ARN/name/status/protocol/version/tag/role/
  trust constraints are checked explicitly. Endpoint id remains unused authority;
  real DEFAULT id metadata is not refused by a blanket generic model regex. Missing/blank policy, unsupported response/type/transport, drift
  and inherited use refuse. Selectors/observations never approve an effective
  cohort or identify a caller. Resource-policy string UTF8 bytes remain exact;
  SDK-decoded IAM trust objects use defined sorted compact UTF8 JSON bytes, not
  raw HTTP bytes, and must match explicit canonical pins. Unreported targetVersion
  (absent or null) maps to None without fabricating liveVersion; actual Guard
  comparison rejects None against a separately approved reported target string.
  Responses/unused environment values are never stored/logged and observations
  are not cached. No defaults, policy writes, approval, graph or app activation
  exist. IAM GetRole and AgentCore GetRuntime/GetEndpoint/GetResourcePolicy/
  ListTagsForResource permissions, real transport/account qualification and
  operator baseline approval are separate integration gates; source grants none.

- Cross-source reload requires a fresh private `ReloadCompatibilitySelection` over
  exact producer/consumer catalogs, original checkpoint refs and fixed base/recipe
  contract. `sft_binding.resolve_reload_compatibility_catalog` rereads explicit
  bounded server configuration; absent configuration yields no selection. Its
  process-bound seal identifies configuration provenance, never native/action
  approval. Public tool fields and stored descriptions cannot select compatibility.
  `SessionSftStore` retains historical producer catalogs and uses a distinct
  compatibility-digest reload key; old failed handles remain unchanged. Native
  restoration/readback compares freshly supplied selection before consumption.
  Selected cross-source reservations alone use closed schema3 and include the
  descriptive tuple in their immutable intent digest. Legacy schema1/2 bytes and
  same-source operation digests omit that field and remain unchanged.

- Historical TRAIN restoration retains a separate observation index and never
  fills current TRAIN submission deduplication. Reuse an existing exact current
  catalog/operation TRAIN entry before applying the history-only blocker; a wrong
  catalog entry still refuses rather than falling through to another Create.
  A discovered historical selection
  cannot fall through to another Create when current selection is absent or wrong.
  Before exposing successful history, fresh server selection must equal the actual
  observed CHECKPOINT and all three immutable refs. Observation/restoration updates
  follow the final fresh selection check; revoked configuration refuses success.

- Producer schema2 alias projections also consume a managed schema3 recovery
  reservation without changing projection bytes/keys or legacy behavior. Deterministic
  parent linkage uses the validated recorded producer catalog as data only. Publication
  requires the actual current store reader and exact digest-qualified successor key.
  `ProducerSnapshotRepository(..., reload_compatibility_reader=None)` adds a private,
  nonserialized current reader required only for selected managed recovery; missing,
  exceptional, malformed, changed or inherited-PID evidence refuses before context
  minting and final return. The independently retained original producer SHA remains
  mandatory and never comes from a mutable row/public payload. Hydration freshly
  validates current selection, then original TRAIN/checkpoint/source and the selected
  successor; it restores only the three-part digest key and returns after fresh policy
  and selection checks. Historical catalogs never become current TRAIN authority.

- A selected managed recovery cannot mint context from diagnostic producer schema1,
  including after deletion/absence of its alias-complete schema2 row. The correct
  fresh reader is insufficient: original schema2 bytes and independently retained
  publication SHA are required before minting. `publish_from_store` may retain
  diagnostic schema1 data, but it is unusable as a selected recovery context.
  Genuine legacy reservations without a compatibility selection retain their
  descriptive schema1 fallback and unchanged rows/bytes. Hydration guards remain.

- `model_campaign.py` owns only the conditional first model-acquisition Processing
  journal. Root's pure prepare/validate adapter freezes exact original request,
  config/manifest/planning/spec and protected campaign IDs into bounded canonical
  data. Typed DTOs, leases and hashes confer no approval. Explicit independently
  qualified protected-original publisher/reader+origin qualifier must retain the
  first original intent BEFORE ordinary reservation; missing/unknown origin or
  retention/table qualification denies. No default/native protected port is supplied.
- Journal `qualified_io` uses a PID-bound per-instance scoped callback, resetfinally,
  at each actual read/write/coordinator leg/effect/exposure. The callback carries no
  policy/witness and does not recursively read DDB. SDK mutation retries must be one
  effective attempt; contradictory response retries refuse. Exact conditional rows
  preserve original job/model_ref through duplicate/lost/conflict responses. Only
  an unambiguously acknowledged consumed marker issues a retained-object, PID-bound,
  single-use effect permit; ambiguous/cold/serialized/copied permits never authorize
  Create. Registration/lease is not exactly-once AWS execution. Outcome CAS requires
  the current original lease/attempt owner; late/stale outcomes and state regression
  refuse. Read-only original restore/NotFound cannot reset an attempt or create jobs.
  Model acquisition artifacts/native isolation, action policy, protected origin
  capability and full pipeline/runtime bridge remain separate required integration.

- Normal model stores retain their constructor-time sealed entry/factory anchor.
  An active campaign cannot adopt a legacy store or catalog alias; matching UUIDs
  never permit a different factory or private services object to rebind it.
  `CampaignRuntimeFactory.refresh_original_planning(entry)` reads the qualified
  original protected handoff/campaign and current bounded planning refs, returning
  a new same-factory sealed DATA view without mutating the old entry or approvals.
  Model results/aliases publish only after fresh runtime plus declared full model
  exposure checks, with staged ordinary maps and no native/journal rollback.
  Post-return refusal uses the closed bridge attempt-error DATA contract; current
  qualification loss withholds identities and never asserts no native effect.
- `campaign_planning.py` owns only PID-bound private planning DATA publication over
  an exact sealed runtime entry and independently qualified protected port/origin.
  Default missing capability and UNKNOWN deny; typed binding/ack/body hashes are
  data, never original ownership or absence proof. Conditional atomic publication
  retains original raw text/canonical spec/ref and protected active predecessor.
  Lost acknowledgement resolves once by current original read; attempted uncertainty
  never replays. Bound or unqualified model-spec selection prevents redirect.
  Cold loading restores the original generation ref without reminting; Root owns
  new factory-sealed planning views, middleware and native capability qualification.
  Research observation is trusted SDK capture only and serializes bounded untrusted
  pointers/coverage/error hashes, never policy, witnesses, clients or license grants.
- `protected_planning_read.py` composes the same original protected publication
  with explicit initialization/current-scope ports and independent origin callbacks.
  Pre-entry reads construct no ENTRY/generation store and make no writes or factory
  mutation. Exact four-field current scope and exact InitializationPlanningObservation
  remain DATA, not grants; missing/UNKNOWN/current-origin drift refuses. The pure
  shared planning validator preserves original raw text and canonical bytes.
  ModelPlanningPortAdapter reads the sealed entry-selected original and emits only
  canonical_spec_json_v1/full SpecModel JSON with its unchanged canonical SHA.
  Raw YAML remains separate. Native current ingress/protection/bootstrap supply is
  external and absent by default; ordinary rows/hashes/types cannot establish origin.

- `campaign_planning_services.py` selects exact same-factory/PID private planning
  configuration with a shared active lifetime; copied selection after exit/fork
  refuses. Intended API-worker configuration propagation is not authentication.
  It builds planning stores only from freshly qualified original initialization
  DATA, current coordinator scope and complete selected protected/origin ports.
  `InitializationPlanningObservation` exposes verified original initial-C/H-byte
  commitments without grants. Its accessor reads one protected C+H transaction
  under mandatory current I/O qualification, never current-C selfhash authority
  or cached policy/witness. Native protection/bootstrap remains separate.

- `preparation_capabilities.py` describes the existing canonical JSONL, fixed OpenSWE and fixed CommitPack source shapes. Only held exact PreparationCatalog DATA enters the existing pure validator; no supplier/client/environment resolution occurs. Configuration shape and query matches grant no native admission. Public descriptors preserve unknown readiness and existing byte/provenance/cohort requirements. Discovery reuses the existing pure Hub repository/filename validators before matching source shapes; no metadata fetch is called. No source mapper or preparation behavior changes.

- Preparation discovery separates held worker-resource shape from legacy complete global-exclusion configuration. The existing pure resource validator supplies only status/missing selector names; source-supported original same-conversation case_request_ref routing is descriptive DATA with unknown case readiness. No case/store lookup, reference creation, execution or native grant occurs. Legacy configuration_status remains labelled worker_resources_and_global_exclusions.

- `model_pack_adoption.py` resolves only an optional exact-version server catalog
  descriptor, verifies the historical native producer and pinned artifact chain,
  and uses existing S3 staging/jobs-table conditional reservation and completion.
  Consumer adoption and original producer attribution remain distinct. Reserved
  or uncertain records never replay copies; read-only restore rechecks current
  role/table incarnation, configuration, policy and exact immutable object bytes.
  Consumer catalog identity and the complete file/manifest path, hash, size and
  format sets must match the freshly verified original; self-consistent replacement
  or omission cannot authorize restored model use.
  JSON metadata is bounded/canonical; no SDK/client/policy or model deserialization
  is persisted. `build_model_acquisition_store` preserves ordinary construction
  when the optional configured catalog ref is absent and otherwise binds only a
  live policy getter; catalog availability grants no execution permission.

- Capacity rejections are not a permanent answer (added 2026-11-10): `SessionCodingGenerationStore.submit`
  and `SessionPairedCodingStore.submit` answer a repeated request from the cached record unless that original handle is a
  capacity rejection (`backends.types.is_capacity_rejection`: `submission_state=rejected` with
  `rejection_code=ResourceLimitExceeded`). Such a create made no job, so the same request reaches the adapter again with
  the same reference and deterministic native name, and the new handle replaces the record. ValidationException and
  AccessDeniedException rejections, ambiguous and accepted outcomes keep today's cached answer. A generation capacity
  rejection also carries `instance_types` (the generation list), `blocking_instance_type` (added 2026-11-10: the one
  listed type SageMaker named as at its limit, from the handle's closed `rejection_type`, or None), `type_holders` (each
  listed type mapped to the job of this run that holds it, or None), `waiting_job_names` (PENDING or RUNNING list jobs
  of this run with no selected type yet; they hold no type), `holding_job_name` (the job of this run that holds the
  blocking type, or None) and a plain `next_step` that names only the blocking type: "Job <a> of this run holds the
  ml.g6e.4xlarge capacity. Nothing was created. Submit this arm again with the same arguments after that job ends.",
  or "Another job in the account holds the ml.g6e.8xlarge capacity. ..." when no job of this run holds it. When the
  message named no listed type, the note starts "SageMaker did not name a listed instance type as the one at its
  limit.", adds one sentence for each holder of this run and ends "after a job on a listed type ends.". Why: the
  earlier note called every listed type without a holder in this run held by another account job, although SageMaker
  refuses for one type and names it. `_active_holders` reads, only after such a rejection,
  the other generation originals and the SFT store's originals whose request names any listed type (fixed type or
  list), and for each one that observes PENDING or RUNNING takes the type it holds: its fixed type, or the
  SelectedInstanceType of its validated description (changed 2026-11-10 from the single
  `GENERATION_INSTANCE_TYPE`). Base and tuned generation share one list: the second create is refused while the first
  holds its type and reaches CreateTrainingJob again on a later submit. No
  busy check runs before a create: without the
  applied quota a pre-check would refuse a create that an account with a larger quota accepts. Unchanged on purpose: the
  SFT store (the durable path keeps a REJECTED reservation as the one original submission;
  `tests/test_qwen_sft_durable_submission.py::test_native_submission_outcomes_are_retained_and_retries_never_create` pins
  that a retry never creates), RELOAD (the same durable reservation) and the case freeze store.
- [Added 2026-11-10, wording and lookup changed the same day] A read of a capacity-refused reference
  answers the refusal. Before, `observe` asked the adapter for a job that the refused create never made and answered
  UNKNOWN (pair: `observation_failed`) with no next step. `SessionCodingGenerationStore.observe` now answers a
  capacity rejection through `_capacity_refusal`: the `_submitted` fields (`status=submission_rejected`),
  `rejection_reason=capacity_limit` (`CAPACITY_REJECTION_REASON`) and the capacity note built with `read=True` from
  the stored `rejection_type` and a fresh `_observed_holders` read (`_active_holders` plus `unread`, the jobs of this
  run whose observation failed or read UNKNOWN). While a job of this run holds the named type the `next_step` is the
  submit's own sentence. Otherwise the note starts "No running job of this run was found on the <type> capacity." (it
  does not claim that another account job holds the type: the refusal is no longer fresh). It then names each job of
  this run on another listed type and says to submit after that job ends (those jobs for more than one). When no job
  of this run was found on a listed type it names each job of this run that waits without a type ("waits for a
  machine and has no instance type yet") and each one it could not read ("could not be read"; "The jobs of this run
  could not be read." when the lookup itself failed), and ends "Submit this arm again with the same arguments now. If
  another job holds a listed type, SageMaker refuses again and creates nothing." `holding_job_name` stays the holder
  of the named type (None when a job of this run holds only another listed type; `type_holders` names it).
  `SessionPairedCodingStore.observe` answers `submission_rejected`, `rejection_reason=capacity_limit` and
  `PAIR_CAPACITY_NEXT_STEP` (no type and no holder: the pair adapter records none), built under the lock that read
  the record. A read describes nothing under the refused name, creates and retries nothing and writes nothing to the
  record. The generation holders are read outside the store lock and the answer is built from the record the store
  holds after that read: a submit that created the job meanwhile is observed as a job, one refused again is answered
  from its own refusal. Other rejections, ambiguous and accepted originals read as before (only a capacity-refused
  record is ever replaced), and `_verified_evaluation_source` still refuses a refused generation.
- `NormalRunContext._original_result_delivered` (added 2026-11-10): a private process-local bool set by
  `note_original_result_delivered()` when the original worker's model received its SFT call result in this process. No
  check, qualification or write uses it; only `app._record_finished_original` reads it when the original worker ends.

- Ended original SFT (added 2026-11-10): `NormalRunContext.restore_sft_store` returns a verified original whose
  native job ended FAILED or STOPPED (identity, restoration status and request reference still checked); only
  `resume_unresolved` and UNKNOWN stay refused. `has_retained_report(policy=...)` returns qualified presence DATA of
  `report_checkpoint_ref` or `resumed_report_checkpoint_ref`. The app uses both to report an SFT job that ended without
  a checkpoint (see `src/amlas/AGENTS.md`).

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.


Prepared normal SDK snapshots are versioned canonical DATA in the original private `prepared-tasks` namespace, bounded to 160 KiB. The optional descriptor is confirmed by an owned-row CAS and readback after SFT association and before wake publication or adapter effects. Publication uses the live server-only approval policy getter and current/original qualifications for each I/O; it preserves the actual prospective SDK snapshot and grants no accepted TASK, readiness, or restore authority. Missing capture preserves the legacy path; a present capture requires the policy getter.

## Child DOX Index

None.

- `_verified_sft_snapshot` requires one fresh successful adapter observation of the original preparation job. It preserves the original typed catalog/header/ARN, accepts only exact `DatasetArtifactRef` or closed five-field internal dictionaries, and validates version/SHA/count/format plus fixed original producer output filename roles and same-session ownership. Subclasses, duck types, foreign identities, header drift and mutable refs refuse. This is a private source transfer, not public JSON authority, restart durability or native GPU admission.

- `coding_case_state.py` holds conversation-local case handles over the exact original acquisition store and no policies. `case_[a-f0-9]{32}` is opaque. Fresh verified output projections expose PROMPTS only; private roles/coverage cannot leak through arbitrary result fields. `_verified_preparation_exclusions` returns frozen `VerifiedCasePreparationExclusions` (session, original producer ARN/header/source, exact EXCLUSIONS and COMMITTED refs) after complete fresh five-file verification. There is no alternate unverified accessor, public grant or restart-durability claim.

- `_verified_evaluation_source` returns frozen `VerifiedCaseEvaluationSource` only after one fresh original five-file case verification. All immutable role refs, original producer header/source and same-session identity stay private; public observation still exposes PROMPTS only. This transports provenance, never policy, native execution admission or grading authority.

- Native admission endpoint ARN and initialization protection table ARN require
  exact strings before comparison; untrusted equality behavior cannot satisfy
  a consumed selector. Matching primitive fields still establish no native grant.


The conditional first-model journal reads an explicit `ProtectedModelRead`
(PRESENT/ABSENT/UNKNOWN) before a first conditional publication. Only known
ABSENT with independently current publisher/table/retention origin qualification
may publish; None/UNKNOWN deny, PRESENT reads the original without republishing.
The None/None origin mode qualifies the first-publisher boundary only and grants
no original commitment, qualified absence, or action. An attempted publication
is retained before its call; unresolved acknowledgement/read never replays it.
After publication all reservation/attempt/effect legs require exact protected
original bytes and acknowledgement. Qualification callbacks cannot renew local
lease deadlines; marker/effect/outcome gates recheck deadlines after callbacks.
The protected publisher implementation and native capability remain unqualified.

The first-model journal uses the original `private_store_correlation` as the
internal adapter/store/artifact `runtime_session_id`. The distinct original
`transport_session_id` remains protected intent data for AgentCore invocation
routing. Both are committed; they are never interchangeable restore aliases.

Model-journal lifecycle methods reject inherited PID before acquiring any lock.
Embedded ordinary-row intent is compared as canonical bytes, preserving numeric
types. Effect consumption rechecks the independently protected original after
coordinator lease I/O. ACCEPTED outcomes require the exact ARN and no error class
before the conditional write; writer and reader retain the same closed shape.


The private model campaign bridge activates exact PID-bound ModelCampaignServices
and builds only for the same sealed runtime factory/entry and actual qualified
model adapter. Independently original spec body/typed pointer/current origins,
fixed current configuration and required immutable native-name hint precede
consumed attempt CAS. Internal runtime_session_id uses private_store_correlation;
transport is callback-only original data. Missing private ports/factory refuse.
Fresh caller approval uses the normal action/unknown cost before effect and stays
outside persistence. Cold/PRESENT/EXISTING/UNKNOWN originals remain read-only.
Journal inspect/reserve dispositions are DATA; only existing sealed consumed CAS
and current independent boundaries can gate an effect. Lost acknowledgements
never mint another job/ref. observe_original keeps the same original handle for
adapter byte proofs, checks immutable source/request/ref and scopes actual SDK
artifact reads before current qualified exposure. Native protection/role/table/
route implementations and normal caller integration remain separate admission.

- Campaign-bound model stores preserve the original journal model_ref and private
  correlation. Typed bindings/services are private; submission and observation
  require the same active factory/original campaign before and after exposure.
  A fresh store may restore an original handle through the qualified bridge but
  cannot mint another alias or use a raw artifact path after scope revocation.
  `_for_campaign_entry` creates only an empty acquisition shell from exact sealed
  current entry data; it grants no acquired/native effect authority.

Private model binding exposure repeats current model planning/configuration and
independently protected original intent/row/ref after runtime checks; no cached
witness/grant is returned. Pure model boundary qualification repeats after the
last runtime I/O. ModelSubmissionAttemptError is closed historical DATA: actual
returned adapter accepted/ambiguous proves attempted, owned preCreate callback
refusal is false, opaque exceptions stay unknown. Lost current qualification
withholds all result IDs; callers report executed=None/outcome_unknown. Permits
and reservations alone never prove a native Create.

ModelPlanningBody carries an exact private body_format discriminator: legacy
frontmatter_v1 retains the existing parser; canonical_spec_json_v1 requires strict
bounded canonical JSON, validated full SpecModel and identical coding_spec_bytes.
Both bind the exact original spec pointer to the SHA of their actual body bytes;
no wrapper/relabelled raw hash, arbitrary decoder, authority or public selector.

- Fixed coding-spec binder validation preserves its exact ordered Qwen model,
  identical tokenizer, pass_at_1 and full_v1/version1 predicates and canonical
  original bytes. Known refusal carries a frozen closed schema/reason/static-path
  diagnostic with data_onlyTrue/native_admissionFalse/quality_successFalse; no
  input/error/value/ID reflection or spec_ref is emitted on failure. Exact private
  same-owner SDK store remains required; unexpected errors stay opaque. This adds
  no planning bypass, native reads/effects, public selector or authority helper.

Prepared-to-TASK promotion reads the exact canonical prepared version and causal
parent, verifies the accepted journal association and current native RUN request,
then requires verified restoration with RUNNING or SUCCEEDED status before TASK
publication. It uses the existing conditional publisher and confirms exact TASK
bytes and row linkage; it never seeds capture state, creates or replays a job, or
loads or invokes the SDK. Full parent hydration remains a separate loader gate.

Promotion pins the verified full row through the publisher's private expected-row
argument; a changed inner read refuses before publication or conditional CAS.

Promotion compares pinned and confirmed rows by exact canonical JSON, preserving
scalar types across every current-row check and the deliberate TASK-row advance.

Prepared checkpoint presence uses the same fresh policy, native/table ownership and
exact owned-row reader as TASK presence. Its boolean is routing DATA only; promotion
and the full qualified loader retain their independent current-state gates.

### Durable normal RELOAD consumer journal

The normal journal keeps `record["sft"]` and TASK associated with the original
TRAIN; its TRAIN equality guard and primary-only readers remain unchanged.
A RELOAD reservation requalifies that TRAIN and requires exact binding, catalog,
typed inputs and recipe, with checkpoint producer ARN, runtime, recipe and source
commit matching the producer. Registered reload retains its existing exact
checkpoint/file-reference verification. The consumer uses the existing operation
rows, distinct operation digest/native identity and reservation/snapshot/outcome
CAS; fresh role/table/policy/owner checks and whole-row fences remain required.
This adds no schema, service, checkpoint authority, legacy backfill or UNKNOWN
retry/replay permission.
