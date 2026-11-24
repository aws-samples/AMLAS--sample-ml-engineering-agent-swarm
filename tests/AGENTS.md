# tests

## Purpose

Own the shipped automated regressions and publication/governance oracles.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- 2026-11-07. `test_ci_shard_plan.py`, `test_ci_shard_merge.py`, `test_ci_shard_plugin.py` and
  `test_ci_shard_orchestrator.py` cover `scripts/ci_shard` with synthetic suites, a fake `docker` on a temporary
  path and real `pytest` subprocesses on a three-file fixture suite. They require exact cover of the collection by
  the shards, byte-equal inventories, refusal of every missing, duplicated, extra, wrong-shard, failing, empty and
  mismatched shard (each leaves the output directory unchanged), the observer's assertions on merged output with a
  mutation that must fail them, the single pipeline's docker flags with only the shard differences, and refusal of
  an original asset that changed. They contain no skip, so the suite's skip count is unchanged. They prove the tools
  against the file formats; they say nothing about a sharded run of the whole suite.

- `test_qwen_sft_bounded_status.py` exercises the registered read-only SFT tool
  with an actual typed store and a declared already-verified observation. The real
  closed metrics validator accepts neutral full1024/16 and fit histories; full DATA
  exceeds unchanged65536 INPUT bytes while public JSON must stay below16384.
  Require exact scalar/state/quality/model-plan/ref preservation, explicit counts
  and final/earliest minimum-loss rows, unchanged complete pinned evidence, one
  observation and zero native submission/staging. Missing full evidence refuses.
  These source fixtures establish no native S3/guard/model/lifecycle proof.

- `test_preparation_request_ref_alias.py` obtains every positive preparation selector
  from the registered SDK submission tool and real SessionPreparationStore over
  strict neutral native ports. Require one original submission, equal opaque
  preparation_request_ref/request_ref aliases across both submit paths and polls,
  unchanged committed artifact DTOs, and the actual normal SFT input validator.
  Artifact/recipe and foreign-store/fresh-policy denials stay intact. This proves
  source interface compatibility only, never native admission or lifecycle success.

- `test_normal_callback_activity_diagnostics.py` reuses the actual installed SDK
  scripted model, strict normal ports and original precommit fixture. Finite inert
  barriers distinguish qualification, model body, tool body and after-tool async
  completion using the owning callback's paired labels. Require closed private
  logging, current/copied/concurrent correlation, default/stale silence, optional
  ordinary logger faults and exact process-control propagation. Preserve INPUT
  guards, snapshot/history bytes and original effect/worker cleanup counts.
  Assert typed assistant message role/content rather than SDK rendered-string
  formatting. These source diagnostics establish no native cause or phase
  admission, deployed lifecycle completion or model quality.

- Causal SFT diagnostics exercise real validators with neutral public SDK session
  snapshots. Fixed subphases distinguish bounded call shape, checkpoint DATA,
  first canonical DATA, final assistant batch, unique actual SFT use, effective
  public SFT inputs and final canonical call budget. Preserve all existing errors,
  the160KiB/DEP24 bounds, immutable prefix/snapshot and zero model/native effects.
  Optional callback forwarding is silent by default; ordinary callback faults do
  not replace refusal and process control propagates before capture. Existing
  injected broad-phase cases prove wiring only; they do not identify a live cause.
  No raw history, arguments, references, error text or traceback is logged/stored.
  Input predicate tests retain the public default and preview guards, verify each
  fixed refusal code and the real SDK hold/cancel and warning order, and preserve
  the original history and native Create count through a refusal.

- Normal SFT capture diagnostic tests cover actual SDK budget refusal, closed
  labels and run correlation, isolated concurrent phases, unchanged snapshots,
  safe logger/callback failures and successful capture without a diagnostic.
  Fixture evidence grants no native admission or lifecycle completion.

- `test_model_verification_interruption.py` (added 2026-11-10) drives the real prompt to its accepted
  original SFT (`normal_sft_cold_fixture`, produced nine-payload pack) and the real `load_qualified_task`, with payload
  HEAD/GET carrying an ETag and a CRC64NVME field so loads after the first run pinned passes from the process-wide proof
  (the shape of a real run). A report-pointer row write before the first run-row read after the observation line and after each
  verification line 0..8 must give no refused line, one `model_verification_interrupted` line (reason, site of the TASK
  recheck), the recheck's own text and an unchanged process proof and epoch. S3 503, S3 connection error, STS throttle,
  DynamoDB throttle, a changed version and a changed size at verification line 2 stay refusals that drop the proof and
  return `verification_failed`, with class, native code and innermost product site on the refused line. Only the
  loader's thread is observed; the original worker keeps polling in its own thread. About 3 to 4 minutes.
- `test_failure_cause.py` (added 2026-11-10) pins the closed cause vocabulary, the exact-type record-change marker and,
  from the source, that all 13 run-record rechecks of `normal_run_checkpoint` raise the marked error.
- `test_normal_model_progress_diagnostics.py` freezes closed WARNING metadata for
  genuine current normal MODEL loading: fixed phase/state labels and typed0..9
  counts, no native/checkpoint/history effects, stale/default-context silence,
  optional ordinary logger/import failure and exact process-control propagation.
  Integration composes the SDK work once and binds its own strict normal ports
  to that same guard; it never layers a second model/capture fixture. Optional
  import faults target only the diagnostic wrapper, accept Python None-fromlist
  ABI, and restore the import hook before pytest assertions/reporting. The
  existing actual-SDK precommit fixture waits on its original MODEL and verifies
  nine tiny neutral payloads with unchanged custody/no-replay assertions. Source
  tests and diagnostic counts grant no native stage, admission or lifecycle proof.
  Since 2026-11-10 a refused line always carries the closed cause fields (closed defaults for a cause
  that is not closed), an interrupted line its closed reason and site, and every other phase refuses a cause.

- `test_input_admission_initial_producer_ack.py` separates stable preassessment
  pool clearing from factory retirement. Retirement still verifies the exact
  accepted/captured original before close/create ordering; native identity,
  custody, checker and no-replay assertions remain authoritative.

- `test_input_admission_owned_transport.py` uses the existing strict neutral
  ApplyGuardrail fixture with reusable SDK pool semantics. Require owned-client
  preassessment close with one apply, stable checker/config/HMAC/cycle/custody,
  fresh-failure hold without replay, close-failure zero-apply, invalid-input
  no-effects, process-control propagation and default injected-client ownership. Legacy completion retirement
  fixtures remain separate. Backend owned checks are serialized by its RLock;
  standalone opted-in checkers require trusted caller serialization. Source
  fixtures prove no native guard outcome.

<!-- normal-callback-stack-configuration-contract:begin -->
- `test_normal_callback_stack_configuration.py` synthesizes the actual two existing
  stacks and real CDK app offline. Require default/explicit-false legacy template
  identity, validated opt-in route/current-role Arn+RoleId, exact runtime/endpoint
  read and invocation resources, and the same jobs table NEW_AND_OLD_IMAGES stream
  with batch1/LATEST mapping. Unknown switches or incomplete/foreign routes refuse;
  two namespaces and actual SageMaker rules remain distinct. A fixed unrelated
  build timestamp makes whole-template comparison deterministic. No deployment,
  native authorization, readiness event or autonomous lifecycle is established.
<!-- normal-callback-stack-configuration-contract:end -->
- [Added 2026-11-10; changed 2026-11-10]
  `test_normal_callback_stack_configuration.py` also pins the AsyncTrainingStack
  runtime binding per path. Opt-in: the three runtime actions and the Lambda
  `AGENTCORE_RUNTIME_ARN` value are the literal runtime ARN and endpoint; the
  `AgentRuntimeArn` parameter is declared with its earlier shape (String, the literal
  as default, the same description) and nothing outside the Parameters section reads
  it; the `RuntimeArnSupplied` condition is absent; resolving the template with any
  stored or passed parameter value (empty, an earlier runtime, the literal) changes no
  resource; the real-app opt-in test asserts the literal binding and the declared
  default for the app path. Opt-in then opt-out: the legacy template, resolved with
  the value the opt-in deploy left stored, grants InvokeAgentRuntime on that one
  runtime; the account/Region fallback appears only for a stack that stored an empty
  value before the opt-in and was given none. Legacy: the parameter (String, default
  empty, the same description), the condition, the `Ref` in the Lambda environment
  and the conditional InvokeAgentRuntime resource stay as they were. These are
  template assertions with a small model of CloudFormation's stored-parameter rules
  (a passed value, then the stored value, then the template default); no
  CloudFormation update is run.

- `test_input_admission_blocking_categories.py` uses the existing neutral FULL
  ApplyGuardrail fixtures and actual checker to separate native BLOCKED CONTENT/PII
  categories from the unchanged all-category union. Preserve one call, BLOCKED
  hold and absent forward text, legacy/nonblocking serialization, closed metadata
  refusal and malformed-native UNKNOWN behavior. The diagnostic exposes no raw
  assessment/match/text or new identifiers and grants no retry/admission. Tests
  establish source behavior only; held native runs remain immutable.

<!-- normal-prompt-report-durability-contract:begin -->
- `test_normal_prompt_report_durability.py` uses one tiny declared terminal
  conversation, existing strict native role/table/S3 transport and the normal
  retain/passive-read helpers. Fresh CWD reads must recover the exact report from
  its owned immutable object and jobs-row descriptor without model/tool/job replay
  or native writes. Identical retention keeps the original winning version with
  no additional S3/DDB mutation; conflicting content refuses while preserving
  the first report. A selected passive reader explicitly forbids API resume or
  local fallback and performs no owner/producer claim. Missing or changed
  descriptors and same-size object hash
  changes hold even when an old local copy exists. Unconfigured normal runtime
  retains its legacy local report path and never enters a selected normal resume
  factory or provider when the canonical normal catalog is absent. No ML/GEN/PAIR lifecycle is replayed;
  declared transport success is not native IAM or managed durability acceptance.
<!-- normal-prompt-report-durability-contract:end -->

<!-- normal-completion-stable-admission-contract:begin -->
- Cold/warm final conversation equality retains every response field and every
  stable admission field; only `input_admission.records` is compared by independent
  validity instead of byte equality. Source `InputAdmissionReceipt.to_dict` includes
  new check IDs/cycles and an owner-ephemeral keyed digest. Cold history must receive
  fresh INPUT checks; restoring keys/receipts to force equality is forbidden.
  Both instances require the complete receipt ABI, admitted ALLOW/NATIVE_NONE,
  EXACT byte measurement, non-null HMAC and Source-valid sorted unique category
  metadata. Categories describe observed policies, including undetected NONE
  entries; they need not be empty. Warm requires OWNER_PROMPT/TOOL_RESULT,
  while cold requires HISTORY_TEXT/TOOL_RESULT. Preserve the same selected guard
  and disjoint fresh check/cycle/HMAC identities. Exact passive report durability equality and all
  workflow/history/native-ID/count assertions remain. A normal warm chat-return
  regression requires the same admitted causal trace as cold return, with one
  invocation and unchanged history/current metrics; ordinary extraction is unchanged.
- `test_normal_fresh_admission_receipt_abi.py` feeds genuine neutral checker
  receipts to the existing cold evidence helper. Only the known optional
  `blocking_categories` omission may use its empty tuple default; required fields,
  present diagnostic DATA and extra fields retain the exact Source roundtrip.
  ALLOW/NATIVE_NONE, guard identity, fresh checks/cycles, disjoint HMACs and UNKNOWN
  refusal remain. These tests cover DATA ABI only; no provider/model or lifecycle
  acceptance follows.
<!-- normal-completion-stable-admission-contract:end -->

<!-- normal-completion-tool-history-contract:begin -->
- `test_normal_completion_tool_history.py` uses fresh declared tool history,
  actual public SDK session snapshots and fresh Agents with an inert Model to
  require the cold final report trace to retain matched admitted pre-SFT results
  before current post-SFT metrics. No SDK metrics are restored or mutated.
  Unmatched, pending, duplicate, malformed, failed or reverse-ordered historical
  calls cannot be counted as executed. Ordinary metrics extraction retains its
  order and argument identity; rejected INPUT admission prevents snapshot load,
  model construction and history extraction. This is report DATA coverage only,
  not native execution, quality, durable storage or autonomous lifecycle proof.
<!-- normal-completion-tool-history-contract:end -->

- `test_normal_journal_task_publication.py` interleaves the genuine prepared
  TASK promotion with an original journal read. The same original must remain
  readable through that one validated pointer addition, while all write methods,
  private-correlation drift, unrelated history changes and existing TASK pointer
  replacement/removal remain refused. Source creates the original, prepared
  snapshot and TASK; provider faults cannot seed authority or another operation.
  Existing 60-second fixture handoffs are test liveness bounds, not production
  job deadlines. This proves Source concurrency behavior, not native lifecycle.

- `test_normal_record_projection.py` requires exact-raw immutable DATA views
  under fresh consistent GetItem/envelope/hash/current checks. Canonical association
  strings and correlation remain immutable; default callers retain detached rows.
  Changed/duplicate/malformed bytes, context/selection drift, current refusal,
  native-call parity and LRU4 eviction controls prove only DATA reuse boundaries.
  No native admission, cached authority, SDK recovery or lifecycle proof follows.

- `normal_sft_cold_fixture.py` consumes only the readonly public Parquet selected
  by task-fixture `AMLAS_TEST_HUMANEVAL_PARQUET`. Source size/hash checks precede
  parsing; all164 official prompt/oracle pins, field schema and order must match.
  CASE publication retains the original source/revision/license/producer envelope;
  generic synthetic codec fixtures remain separate.
- `test_normal_pair_content_contract.py` runs the actual CASE/GEN publishers,
  canonical spec/ref binding and managed/candidate/whole-pair validators with328
  candidate checks and no SDK/native or candidate/oracle execution. Compatibility
  grants no execution permission, cold recovery or lifecycle/quality proof.
- Owning CASE/GEN/pair runs provision the repository's hash-locked CPU test layer
  and five pinned metadata files, then source its environment with existing paths
  preserved. The layer and DATA mounts remain readonly during tests; shared seed
  environments and model weights stay untouched.
- Normal warm-fixture waits retain60 seconds independently for each handoff
  and final report. The optional cold helper deadline does not pass timeout
  arguments on this default path, preserving existing fixture call signatures.
  These finite test liveness bounds are not production job/runtime deadlines
  or release performance criteria; functional assertions and verification/
  cleanup oracles stay unchanged.
- Distinct-process cold continuation tests attach their own event list and wait
  for the existing post-claim `run_started` signal or a closed reader, rather than
  engine construction. One finite900-second child budget supplies the remaining
  time to readiness, both registered generation records, paired evaluation and
  final report; the subprocess has the same900-second outer bound. This bounded
  recovery liveness allowance changes no production deadline or performance
  floor. All completion-CAS, no-duplicate-effect, history, cleanup and durable-
  report assertions remain unchanged.

- `test_normal_continuation_validation_cache.py` uses strict fresh declared GetItem
  DATA and actual typed read policy to require one continuation validation for
  identical bytes while retaining both reads/current gates. Detachment, changed
  invalid bytes, hash/policy/context drift, replacement and LRU4 eviction controls
  establish pure cache boundaries, not native or cold authority.
  Warm decode work-count, cold/after-valid duplicate-key rejection, canonical
  numeric/Unicode equality and nested mutation controls retain detached DATA
  without repeating the validated duplicate-key hook.

- `test_generation_published_model_layout.py` uses the actual generation
  adapter and the acquired nine-model/three-provenance layout. It requires nine
  exact-version copied model inputs, unchanged original refs and a clean native
  channel. Existing extra-file, provenance, ambiguity and tuned-source checks
  remain. These strict inert ports prove adapter behavior, not real ML execution.
- Cold fixture manifest/inventory tests retain the real binding validator and
  S3 SDK request shapes, and use one shared publisher/staging namespace. They
  establish fixture compatibility only; they cannot confer native admission,
  autonomous completion or model quality.

- `test_normal_wake_readiness_join.py` obtains routes through actual normal SFT
  publication, then requires immutable terminal-hint retention before callback
  delivery. Strict external ports cover duplicate/lost-ACK exact reconciliation,
  conflict/denial/unconfirmed/route-drift holds, malformed status and post-response
  drift. Fresh response bodies and both closures model repeated SDK invocations;
  no native or consumption authority follows from these tests.
- `test_normal_wake_routing.py` contrasts valid present-normal hint retention with
  confirmed-absence campaign read-only behavior. Its adapter permits only exact
  conditional canonical hint writes for that valid route; all other writes/query/
  scan remain forbidden. Campaign payload retains its current event ID. These
  controls prove retention/routing fences, not stream convergence, native bootstrap,
  cold loading, replacement effect fencing or autonomous lifecycle completion.

- 2026-11-10. `test_callback_real_event_shapes.py` runs the campaign-mode `handler` on
  `fixtures/sagemaker_training_state_change_real.json`: seven real training jobs (Completed,
  Failed, Stopped) in the documented EventBridge envelope, with the delivery pattern measured
  on the deployed target (two invocations 6 to 878 ms apart per Completed or Failed job, one per
  Stopped job). It pins identity and refusal before any client, the five-qualification lookup
  (equal to the CloudTrail reads of one live pair), the 249-call delivery budget (194 before the
  invoke), one hint and two resumes for a concurrent pair, no raise on an unacknowledged resume,
  150 readiness retries with no runtime call, and, against the real app admission in one process,
  one restore with a closed busy answer for the second resume and `completed` for a late one. One
  process is one microVM: two replacement microVMs for one session are outside these tests.
  The twin-delay tests require that, with a Lambda context, the delivery that did not create
  the hint pauses 40 s (bounded by the remaining time minus 50 s) before it resumes, and that a
  single delivery, a direct call and the readiness stream path do not pause.

- `test_normal_prepared_resume_route.py` drives the same-session ordinary resume
  selector over fresh actual accepted/prepared state. External TASK publication
  must precede the public loader outcome; no loader/graph hooks or successful
  hydration are manufactured. Pending recovery may load at most one actual SDK
  snapshot, with no new goal, model, job or SDK invocation. Closed unavailable
  output retains only approved stage/category evidence. This is route wiring,
  not cold/native/autonomous lifecycle acceptance.
- `test_normal_promotion_canonical_row.py` changes only declared owned-row history
  int1 to boolTrue after actual successful TASK CAS. Python-equal but canonically
  distinct DATA must refuse, retain the changed row/exact artifacts and cause no
  additional execution. `test_normal_restore_diagnostics.py` covers the fixed
  checkpoint_promote stage with existing no-message/stack/payload assertions.

- `test_normal_prepared_promotion.py` uses actual fresh accepted/prepared SDK
  prerequisites and a refused original TASK CAS to require exact idempotent
  publication with no dispatch, stopped/unknown-native refusal and unchanged
  concurrent rows. The private expected-row argument must reject replacement
  before writes. Its scoped S3 provider models conditional412 and latest-version
  resolution only for existing Source-created TASK objects; exact-version/hash
  assertions remain. These are publication controls, not physical worker-loss,
  cold graph/SDK-load, automatic wake or autonomous lifecycle proof.
  Shared original cleanup snapshots the existing locked active-report entries and
  waits for those entries and its original SDK tasks within the same five-second
  budget before cancelling/draining the reader. Ended/disconnected readers stay
  supported; production SDK busy semantics do not change. Direct and shared
  prepared fixtures settle the original SDK/report worker before the first manual
  journal read; concurrent whole-row changes still refuse. Prepared promotion
  captures its exact baseline after this settlement, verifies cleanup changed only
  an optional single report pointer, and retains that pointer in every subsequent
  exact row/CAS comparison. Closed SFT recovery exports after the same barrier;
  original journal, prepared bytes and all native/tamper assertions remain exact.

- `test_normal_prepared_checkpoint.py` uses fresh actual PRECOMMIT SDK capture
  over strict inert providers to require exact prepared descriptor/snapshot bytes
  before wake/staging/Create, with no accepted/TASK-ready authority. Prepared Put
  denial or unavailable exact-version confirmation must be reached after SDK
  validation, terminate publicly and cause zero SFT effects. The no-fault wrapper
  control and finite sixty-second per-handoff fixture budget qualify test wiring;
  original upstream/success assertions remain. No cold/promotion/liveness proof.

- `test_normal_wake_publication.py` uses the actual fresh normal SFT app/API/SDK
  submission and strict inert external ports to require one conditional route Put
  and exact consistent readback before staging/Create. Denial/conflict/unresolved
  acknowledgement yields zero effects; exact lost acknowledgement/identical rows
  do not overwrite or blindly retry. `test_normal_transport_binding.py` checks
  private SDK transport/hash binding, absent-role legacy and positional ABI.
  These controls establish wiring/ordering, not native permission, TASK readiness,
  cold replacement, reliable wake or lifecycle/quality completion. Existing prepare
  spies/session fixtures follow the SDK shape without weakening their assertions.

- `test_normal_wake_routing.py` uses the actual qualified callback handlers with
  explicit inert route-observation scope and strict SDK-shaped external ports.
  Twenty cases require exact original-session normal resume delivery and closed
  refusal for malformed/hash/scope/incarnation/transport DATA before campaign
  fallback. Confirmed missing normal records preserve the campaign route.
  Routing DATA confers no TASK readiness, native admission, cold restoration,
  reliable wake, duplicate-effect fencing or lifecycle/quality proof.

- Three `test_normal_completion_runtime_worker.py` cases drive the actual managed
  entrypoint/task/report helpers with declared restoration/engine seams: responsive
  same-run worker, duplicate exclusion, disconnect, cleanup-before-report and
  already-finished refusal before start publication.
  These establish host mechanics, not native/cold qualification or an autonomous
  lifecycle. Existing closed restoration and overlay-order assertions stay exact.

- `test_normal_completion_admission.py` has two freshly declared checker cases
  requiring TOOL_RESULT admission before claim/build/load. Two stateless
  `test_normal_completion_route.py` cases preserve pending routing and require the
  completed claim before any pending load, with the response still closed.
- Two `test_normal_completion_sdk_snapshot.py` cases use fresh actual SDK Agents
  and public session snapshots to check append-only completion, unchanged history
  and preserved state/conversation/interrupt DATA with zero model streams. These
  declared mechanical tests establish no native/context qualification, real
  completion CAS, takeover/wake/report or cold recovery proof.
- 2026-11-15. Tests pass product-facing session snapshots through `_record_session_snapshot`.
  This helper is in `src/amlas/tui/chat/tool_binding_backend.py`.
  This rule covers captures and the `sdk_snapshot` of a TASK.
  Snapshot field-set assertions use record form.
  Snapshots read back from restored agents are compared in the same form.
  `test_normal_parent_publication_boundaries.py` searches the TASK size boundary on record-form bytes.
  Those are the bytes the product budgets.
  The tests run on strands-agents 1.42.0 and 1.65.0.
  These tests do not prove anything about the raw SDK inventory.
  These tests do not prove anything about a deployed runtime.
  These tests do not prove anything about a stored record written by an earlier version.
  Live SDK messages compared with stored, expected or another run's messages use `_record_messages(live)`.
  These comparisons still fail when role, content, metadata, order or count differs.
  `normal_sft_cold_fixture.py` stays byte-identical because its checksum is pinned.
  Tests that read its `state.prefix` convert it at their own comparison line.
  The agent builder gives the SDK a new dictionary per message.
  Caller history dictionaries stay as built on strands-agents 1.65.0 and 1.42.0.
  `test_caller_history_stays_unchanged.py` requires this.
  The copy is one level deep. Nested content blocks are outside the check.
  `test_input_admission_wiring.py` still compares in record form.


- Three `test_normal_completion_continuation.py` cases use the actual SDK and
  harmless scripted model/tool events to require current completion/research fences,
  one retained-agent no-prompt invocation, refusal before effects and no output or
  retry after revocation. Explicit inert qualification/renderer seams prove no
  native authority, cold restoration, wake or final lifecycle report.

- `test_normal_completion_durable_attempt.py` exercises real schema/claim/start CAS
  over strict inert service-shaped ports. Require fresh READY receipt reconstruction,
  independent contender/unique lost-ACK ownership, stale-receipt and INVOKING/legacy
  closure, unspent denied start and malformed schema2 refusal. Existing invocation,
  receipt/content, codec, SDK and cold-continuation controls remain unchanged.
  Source fixtures establish no native admission, automatic wake, real ML effects or
  universal post-invocation recovery.

- Three `test_normal_completion_invocation_claim.py` cases retain actual completion
  parsing/qualification with explicit inert native/original seams to test one-shot
  begin, concurrent exclusion and policy refusal. Two
  `test_normal_completion_sdk_none_probe.py` cases use only fresh actual SDK state
  and a scripted Model to verify no-prompt continuation and snapshot wire fields;
  these mechanics prove no native/origin/guard or cold restoration authority.

- Five `test_normal_completion_content_fence.py` cases use declared DATA receipts
  and actual row parsing/qualification to preserve complete claim/ref and local
  snapshot continuity; missing local receipt refuses. Three
  `test_normal_completion_claim_receipt.py` cases retain actual claim/CAS/readback
  code to test detached receipt capture, stale CAS and changed readback. Explicit
  inert transport/original/object seams establish no native/origin/guard authority
  or cold-restoration proof. Keep the confirmed-claim-loss trace callback local to the exact `NormalRunContext.claim_completion` code object: return `None` for unrelated code so other worker frames do not receive line tracing. Preserve the existing cut-line selector, ten guards, confirmed readback, receipt write/flush/fsync, exit 73, and 300-second child/inner, 60-second wait, and 5-second cleanup budgets. Callback controls qualify instrumentation only; the unchanged cold regression must still qualify recovery.

- `test_normal_completion_record_codec.py` exercises the actual row decoder with
  fresh declared GetItem DATA and installed SDK wire-shape validation. Positive
  epochs and legacy rows retain acceptance; malformed continuation fields, SDK
  snapshots and TASK descriptors refuse. This bounded codec fixture proves no
  original qualification, completion CAS, native permission or cold restoration.


- `test_runtime_cold_config_initialization.py` drives the real runtime entrypoint
  with stateless handlers and an overlay spy. Resume/status/report routes require
  exactly one `agentcore` overlay before handler/catalog use. Source-only ordering
  proof grants no checkpoint recovery, runtime identity or lifecycle completion.

- `test_normal_sft_causal_task_checkpoint.py` freezes three internal DATA ABI
  requirements after genuine full warm SDK prerequisites: prospective native-bound
  causal parent descriptor, admitted public SDK TASK snapshot before PRECOMMIT wait,
  and coherent self-hash/reference tamper refusal. Guard-denied/UNKNOWN projections
  publish no TASK; legacy absence retains prior identity. No store/journal/grant is
  seeded, and existing75/cold3 assertions stay unchanged. This component slice is
  not native IAM, actual worker-loss/wake, or full lifecycle/quality acceptance.

- `test_normal_run_checkpoint.py` requires the real normal app/API/ChatEngine/SDK
  path to conditionally enroll one server run in the existing jobs table before
  model dispatch. Strict native transport shapes and actual SDK session context
  supply data only; no private campaign factory/granted context or seeded SFT
  operation is used. Rejected input and live policy/credentials stay out of
  checkpoints; new status/recovery grammar fails closed. A server-only runtime role
  ARN/RoleId pair explicitly selects checkpoints; a partial pair or missing table
  refuses before model/effect, while absence of both preserves honest legacy behavior. This first enrollment slice does not establish original
  SFT alias/publication custody, lost-ACK reconciliation, lease fencing, cold
  original-job recovery, automatic wake, native IAM or full R1-R4 acceptance.
  Existing report-route/39 assertions remain unchanged.

- Managed report tests propose the exact passive action
  `{action: "get_prompt_report", run_id: <server-generated32hex>}` on the existing
  invocation entrypoint. `prompt_report/available` contains the exact original
  terminal envelope; busy, absent and foreign-session reads share the same closed
  `unavailable/foundFalse` result. Malformed/extra authority or path fields refuse.
  Ownership uses the actual SDK session ContextVar, not payload/header-supplied
  identity. Six cases retain the same worker and SDK cleanup barrier; repeat reads
  create no prompt, engine, task, model, capability or native dispatch. Held/UNKNOWN
  data and false lifecycle/quality envelope flags remain intact. These fixtures do
  not prove service IAM/header authentication, writable native storage, real client
  retrieval or cold durability. Existing local report tests remain unchanged.
  One additional persistence-return barrier saves the actual snapshot and then
  withholds the passive result until the original worker completion boundary;
  SDK task clearance alone cannot publish a `worker_finished: True` report.
  Cleanup always releases that barrier and drains the same original worker.

- `test_runtime_busy_ownership.py` proposes a local server-owned final-report
  mapping: the normal prompt's opaque `run_started.run_id` identifies a saved
  generic session whose `metadata.normal_prompt_report` retains the complete
  typed terminal envelope and response. The same worker must finish after stream
  disconnect/deadline, clear its SDK task once, and remain readable through the
  existing passive `api.resume` without new execution, native calls or quality
  claims. Held/UNKNOWN typed responses stay held; unknown references remain absent.
  This is a new storage mapping under test, not an existing managed HTTP result
  route, durable execution resume, native admission or full lifecycle proof.

- `test_input_admission_initial_producer_ack.py` retains a genuine SFT original
  through the ordinary consumed-profile SDK boundary before requiring first-ACK
  guard-pool retirement. Preserve custody-before-close, one logical assessment,
  same Session/checker policy/cycle/current witness and original final completion.
  Successful inline PRECOMMIT returns the ordinary tool-agent action; assert its
  admitted/consumed final SFT result, same original/history and finished worker
  directly. The optional outer warm-loop metadata is not its completion contract.
  Its inert idle transport models freshness only, not the native incident cause.

- Runtime binding and infrastructure namespace regressions require ListTags-only
  CASE ownership-tag reads on default and isolated deployment processing-job scopes
  in both supported deployment paths, separate from existing AddTags/control-plane
  grants. Hermetic policy/synthesis tests establish no live IAM or lifecycle admission.

- `test_sdk_resume_redaction.py` uses actual SDK hooks and selected INPUT admission
  with inert guard/model/native ports. Neutral provider redaction and independent
  prefix mutation remain held, while plain guardrail stop is distinct. Callback
  BLOCKED/NONE/UNKNOWN summaries survive only as bounded metadata on the same
  refusal; no-trace controls publish no invented assessment. Fixtures establish
  source behavior, not actual native guard classification or live ML acceptance.

- `test_bedrock_guardrail_stop_without_trace.py` (added 2026-11-15): the SDK's real `BedrockModel`, bound with
  the product's own guardrail arguments (id, version and trace mode; no redaction setting), over a stub client
  that replays one guardrail stop without a guardrail trace. From Strands 1.54.0 (locked: 1.65.0) the SDK then
  emits its user-input redaction `[User input redacted.]`; Strands 1.42.0 does so only for a trace with a blocked
  policy. Pinned on the installed SDK: the event order; an ordinary chat turn stays an intervention with the
  provider text, no hold and no assessment, and only that turn's agent keeps the redacted text; a normal-run warm
  continuation is held (`history_prefix_refused`, `input_admission_held`, `continuation_failed`, the fixed hold
  text) on the newer SDK and is reported as `guardrail_intervened` on 1.42.0. The expectations follow the
  installed SDK version; keep both branches while both versions are compared. Fixtures establish source
  behavior, not native guardrail classification or a deployed run.
  Since 2026-11-21, the resumed-run scenario lives in the `resumed_run` helper.
  A second test replays a stop whose trace names a blocked policy.
  It requires a hold on every SDK version and a typed assessment in the report.
  Both tests exclude provider text and redaction text from the report. Waits use `LIVENESS_WAIT_SECONDS` as the
  upper bound.

- `test_memory_hook_record_form.py` (added 2026-11-15): `MemoryHookProvider` with a recording client, on stubbed
  live messages in the Strands 1.65.0 shape and on a real SDK run with the scripted model. Every `create_event`
  call must receive copies without the top-level `tracking_id`, with every other key and a nested key of that
  name kept, while the live messages keep their objects and ids. Only new messages are sent, and a failed write
  is sent again. The tests pass on Strands 1.65.0 and 1.42.0. The recording client proves nothing about what a
  real Memory resource accepts.

- `test_sdk_pending_handoff.py` exercises the ordinary app/API/ChatEngine and
  pinned SDK with inert native/model ports. Trusted-profile pending batches yield
  before an extra model cycle, retain all66 mixed sibling uses/results, and resume
  the same Agent after original completion. Prior admitted discovery and producer
  completion chains retain exact history; counterfeit stop state, mutation, current
  policy revocation and later BLOCKED/UNKNOWN siblings refuse. Default behavior,
  native Create counts, worker busy lifetime and completion once remain required.
  The existing explicit bulk clock is confined to inert65-producer volume cases;
  real deadlines and production pacing remain unchanged. These tests establish
  source behavior, not live guard admission or full ML lifecycle acceptance.

- Model/Qwen startup ordering oracles locate the actual profile-aware native drop
  call inside `run()`, then preserve capture, same sealed FD handoff, profile policy
  and fixed exec ordering. A separate legacy-helper oracle retains its fixed
  CaseFreeze delegation and permanent native group/UID/GID/capability/final checks.
  The whole initializer keeps its explicit-error/no-assertion contract; cfg(test)
  predicates use equivalent explicit panic branches rather than widening that gate.

- Warm closure fixtures snapshot immutable SDK history only after the actual initial process_message coroutine returns. Source AfterToolCall capture counts and model stream-start counts do not prove the installed SDK turn is settled. Preserve all original identities, Create counts, history equality, current-policy closure, exact worker cleanup and existing 30s bounds; use a passive threading.Event at the real return boundary, never arbitrary sleeps.

- [Added 2026-11-16] `case_prep_continuation_fixtures.start_original_target` returns only after the source's first read of the target original (`target.actual_observed`). The stream stand-ins count a call when their stream starts and the SDK appends that call's answer afterwards, so a call count does not prove the answer is in the history. `test_original_target_settled_boundary.py` holds the fourth answer behind an event and requires the fixture to wait for it. `test_normal_research_profile.py` waits for the first read of the original MODEL (a passive `threading.Event` on the native describe) before it reads the pending original or copies history: the Create record and the stream start are both earlier than the capture hook and the tool-result message. `test_runtime_pair_original_capture.py` and the DATA-then-MODEL case of `test_sdk_pending_handoff.py` wait until the source has retained the original before they read it: the connected fixture's result recorder and a tool's own read of its submission both run before the source's after-tool hook. Keep every existing 5s bound and assertion; a new continuation test that copies history for an equality check waits for a source read boundary first.

- [Added 2026-11-17] In continuation tests, the source reads the pending original every 10 ms on another thread. A read can fall between separate changes to stand-ins or controls. The persistent CASE failure test in `tests/test_case_unknown_observation_budget.py` takes counts, sets the native state to Completed and replaces `adapter.artifacts` under the backend's `_warm_lock`. The warm-completion test in `tests/test_normal_research_profile.py` changes the policy or guard version before completing the fixture. Both keep their assertions unchanged. A new continuation test that changes more than one stand-in or control after the source starts reading makes those changes one step for the source. It puts the dependent change first and the state change last, or makes all changes under `_warm_lock`. It takes counts compared by later assertions under that lock.

- [Added 2026-11-17; changed 2026-11-20] The SDK reads the wall clock at the first stream event of a model call and adds `timeToFirstByteMs` to that assistant message's metadata once the time since the call reaches 1 ms. A scripted model answers at once, so the field appears only when the test process is held up. Two tests of `tests/test_normal_run_console_silent.py` run the same scripted prompt twice and require equal `normal_conversation_size` lines and equal record-form messages; with the field in one run the logged size differs by 22 or 23 bytes. The fixture `scripted_model_has_no_measured_first_byte_time`, used by every test of that file, sets the clock reading in `strands.event_loop.streaming` to zero, and the SDK then adds no such field. Assertions and the product are unchanged: `normal_conversation_size` keeps counting the message metadata that records keep. `test_guard_slow_first_stream_event_leaves_the_scripted_metadata_unchanged` makes the scripted model wait 5 ms before its first stream event and requires the metadata of all three assistant messages to equal the scripted `USAGE`; it fails every time without the fixture. A test that compares messages, records or logged sizes from two runs of a scripted model keeps the SDK's measured time to first byte out of both runs by using `scripted_model_has_no_measured_first_byte_time` or living in that file, and keeps the slow-first-event test in the same file. The one exception is a test that needs the field present: it gives every model call of every compared run the same fixed reading with `every_model_call_measures_the_same_first_byte_time` from `tests/test_sdk_record_form.py`.

- [Added 2026-11-17] `tests/test_case_unknown_observation_budget.py`, `tests/test_model_unknown_observation_budget.py` and `tests/test_runtime_ambiguous_model_capture.py` waited for the eighth failed or unknown read of the pending original, paused with `asyncio.sleep(0.03)`, and asserted `work.api_returned.is_set()`. The API returns on the worker's thread after that read. A worker delay longer than about 30 ms could therefore fail the assertion during a correct run. The tests now wait for `work.api_returned` under the existing 5 s bound before making the same assertion. The assertions and product are unchanged. The exact counts remain eight failures or eight reads, one completion and one create, and still refuse a ninth read. A new test waits for the state it asserts under the 5 s bound and avoids a fixed pause before a positive assertion about another thread.

- [Added 2026-11-17] In the invalid-payload case of `tests/test_warm_model_refusal_diagnostics.py`, the test completed the fixture, which sets the native state to Completed, and then changed the stored payload in a separate statement. The source reads the pending original every 10 ms on another thread. A read between those statements saw Completed with the unchanged payload, continued with a third model call, and failed the refusal assertions. Both statements now run under the backend's `_warm_lock`, which the source holds for each whole read. The assertions and product are unchanged. A new continuation test that changes more than one stand-in or control after the source starts reading makes those changes one step for the source.

- [Added 2026-11-17; changed 2026-11-20] In `tests/test_warm_model_closure.py`, one scripted assistant message carries two same-catalog `acquire_model_weights` tool uses, which the SDK dispatches at once. The store takes `_submission_lock` without blocking and returns `busy` when it is held. A first-call delay of 50 ms gave the second call `busy`, failing the assertion requiring both results to carry the pending original's reference. The test now wraps the store's lock in `TurnTakingLock`, defined in the test file, so submissions take the real lock in turn. The assertions and product are unchanged. The test in `tests/test_warm_model_closure.py` covers two accepted captures of one original. The `busy` answer to overlapping calls is covered by `test_overlapping_submission_is_answered_busy_without_a_second_create` in `tests/test_normal_model_campaign_flow.py`. A new test needing both concurrent calls to pass a non-blocking lock orders them at that lock.

- [Added 2026-11-20] `test_overlapping_submission_is_answered_busy_without_a_second_create` in `tests/test_normal_model_campaign_flow.py` records the overlap contract for `run_model_weight_acquisition_flow`. After approval, a held `_submission_lock` gives `status` `busy`, `accepted` false, and the approval decision. This path makes no native call. The test uses one campaign-bound store and `bridge_fixture`. A wrapped `create_processing_job` holds the first call at the stand-in boundary with Events. Every wait uses an Event or a thread join. The second call has a 10 s join bound. It must return while the first holds the lock. Its decision must be `approved`. Its remaining fields must match the plan-only preview. Exactly one create must occur. Releasing the first call must yield a recorded job and a free lock. A later call must reuse that job. The test does not cover SDK tool dispatch or a native service. Future overlap tests must hold the first call inside the product lock with an Event and bound the second with a join so waiting fails within the bound.

- [Added 2026-11-17] `tests/test_sdk_record_form.py` and `tests/test_normal_completion_sdk_none_probe.py` also compare scripted messages: two captures with each other and with a stored golden record, a record sized to the byte at the 160 KiB cap, and the record form of a final message. A measured `timeToFirstByteMs` on one stream changed those bytes and failed three tests whenever the first stream event came 1 ms or more after the model call. Both files now carry the fixture `scripted_model_has_no_measured_first_byte_time` described above. The assertions and product are unchanged. A test file that compares messages, records or sizes produced through a scripted model carries this fixture.

- [Added 2026-11-20] `test_a_measured_first_byte_time_stays_in_the_record_counts_in_the_size_and_is_equal_in_two_runs` in `tests/test_sdk_record_form.py` proves that `timeToFirstByteMs` stays in the captured record with value 125. It checks the field in the live history at capture. It checks that `normal_conversation_size` matches the canonical record size. Each assistant message with the field adds 24 bytes. Two runs give equal records, size lines, histories and witness text. A real SDK agent that loads the stored record holds the stored messages in record form. The helper `every_model_call_measures_the_same_first_byte_time` replaces both SDK clocks with one clock. Each reading advances by exactly 0.125 seconds. Every model call measures 125 ms. The helper runs inside `monkeypatch.context()`. The zero-clock fixture resumes afterwards. The size conversations run before `record_catalog` is requested. That fixture replaces the model in every agent the product builds with its own stand-in. Future tests that need the field present must use the helper for every model call of every compared run.

- [Added 2026-11-20] The registered tool `save_spec` checks the real path before writing. It writes below the configured workspace, data, output or specs folders, the working directory, or the system temp folder. Other paths get a saved false reply with error text. The model chooses the path. This rule keeps writes in the places set up for the project. Pytest puts `tmp_path` below `--basetemp`. Six tests of `tests/test_generated_coding_spec_tool_connection.py` failed when that folder was outside the system temp folder. The refused save stopped the scripted model before the bind step. The `spec_folder` fixture sets `AMLAS_WORKSPACE_DIR` to the test's own `tmp_path` and returns it. The three saving tests use that folder. Future tests that let a tool write a file must set their workspace folder or write below the working directory.

- [Added 2026-11-21] `test_caller_history_stays_unchanged.py` reuses caller history across scripted chat turns.
  Caller dictionaries must stay equal with and without a guard. Guard text must match across guarded turns.
  The builder must give the SDK no messages for absent or empty history and an equal list for supplied history.
  An uninterrupted run, a run restored after its fine-tune job finished, and a run resumed while its fine-tune
  job still ran and restored after the job finished each require every SDK agent construction to receive no
  messages and the one chat turn to carry no history. The resume request that arrives while the job still runs
  must restore the run in a new engine, build its agent, and end with the waiting run's status of `InProgress`.
  All three full-run cases need the full scripted fixture, which does not run on the strands-agents 1.42.0
  comparison environment.
  The dictionary checks cover one level only. Nested content blocks are outside the checks.

- [Added 2026-11-21] `test_normal_run_held_final_state.py` uses the runtime entry and a durable run record.
  It covers a resumed stop without a trace and a resumed stop with a blocked policy trace.
  It requires the held response, the retained report, and the `enrolled` record.
  It checks the report, status, and resume requests with and without resume memory.
  These requests must start no work and write nothing.
  The no-trace expectation follows the installed SDK. The fixture covers scripted runs only.
  It proves nothing about a deployed run.

- [Added 2026-11-21] `test_input_admission_unencodable_text.py` uses scripted chat turns.
  It requires a hold for unencodable tool text, tool JSON, history text, or prompt text with a guard bound.
  The guardrail and model must receive none of the unencodable text in those cases.
  The turns must complete without a guard. Ordinary text must be admitted with a guard.
  The guard-bound cases must fail if the checker's strict encode is made lenient.
  The fixture covers chat turns only. It proves nothing about a real model endpoint.

- [Added 2026-11-21] `test_chat_llm_absent_warning.py` sets
  `amlas.tui.chat.engine._WARNED_CHAT_LLM_ABSENT` to `False`.
  It calls `_warn_chat_llm_absent` and requires exactly one WARNING record from the module logger.
  The warning must name `strands-agents` and `pyproject.toml [project.dependencies]`.
  The text must contain no version comparison operators or dotted numbers.
  No number, operator, or extras bracket may directly follow the package name.
  `pyproject.toml` must declare `strands-agents` exactly once in `[project.dependencies]`.
  A second call logs nothing, and the next call after `reset_degradation_warning` logs the same text again.
  The tests cover only the text and the once-per-process switch, without building a chat engine
  or forcing the `strands` import to fail.

- [Added 2026-11-17] The `native` fixture of `tests/test_coding_case_native.py` changes `SOURCE_SHA256` and `SOURCE_BYTES` in `amlas.data_eng.coding_case_profile` for one test. `amlas.arbiter.coding` and `amlas.evaluation.coding_pair` copy `SOURCE_SHA256` when they are first imported. When that first import happened during a test that used the fixture, the module kept the changed hash for the rest of the process and refused a later coding report: `tests/test_case_unknown_observation_budget.py` or `tests/test_runtime_case_prep_continuation.py` followed by `tests/test_normal_late_resume_after_finished_run.py` failed in one process. The whole suite passed because other test modules import both modules at collection. The fixture now imports both modules before it changes the profile. The assertions and product are unchanged. A fixture that changes a module constant first imports the product modules that copy that constant at import.

- Warm multi-producer regressions require genuine registered SDK target submission,
  exact private original/native identity and current policy, then actual owning
  artifacts before same-Agent append-only continuation. Preserve independent live
  originals and captures made during resume; retain source PREP1..256. Full SFT
  and distinct reload, separate base/tuned generation164, and paired328/164 source
  verification stay distinct. Neutral upstream supplier ports and provider fixtures
  grant no native admission, cold durability, actual ML quality or goal acceptance.
  Read-only per-leg scopes compose PID/thread/lexical current callbacks, close raw
  streams and drain cancellation before completing the exact SDK busy task.
  Historical foreign payloads, public refs/flags and copied/expired scope refuse;
  observers never stage/Create/reload/cancel/replay/adopt. Preserve actual reds,
  counter/timing/composition faults and failed runs without inventing pass coverage.

- The managed model image regression pins the build-only installer checksum and
  patched-version/order, binary-only locked transport install and final installer
  absence contract. Dedicated actual-image red/green checks require distributions
  and importable pip/setuptools/wheel paths absent, with the fixed launcher/source
  and raw transport imports preserved. Source checks alone grant no native admission.

- Callback review regressions retain actual cleanup-before-clock/interface,
  close-overrun, copied-scope lifetime/thread, postInvoke-beforeRead, timeout,
  exact discriminator/resource and setup-failure reds. Positive fresh-thread and
  isolated Lambda package import controls use no provider/handler/model call;
  close/read/invoke counts and retained uncertainty are explicit. Source metadata
  is conditional evidence, never native auth/timing or lifecycle qualification.

- `test_campaign_callback.py` uses harmless SDK-input-model transports for exact
  eleven-field lookup/nine-field wake, selected qualifier, no statuswrite/Scan/Create,
  current qualification/type/config/PID refusal and duplicate/reordered hints.
  Actual missing-feature and response-metadata reds are retained. Bounded drain,
  timeout/close/retry uncertainty always preserves original identifiers without
  re-invoke; legacydisabled compatibility is separate. No fixture pass establishes
  native callback IAM/cohort, protected publication, receiver completion or ML proof.

- Closure regressions retain actual current-table drift, post-callback 61s expiry
  at CAS/observer and denied pre-Put poisoning. They count native fake Update/Put
  and observer calls, preserve witness/lease deadlines, and require no mutation
  under expiry plus one eligible Put only after known-unattempted denial resets.

- Runtime-entry review reds count all TxWrite/Put/Update mutations and retain
  ingress revocation at actual IO/observer/return, historical duplicate readiness,
  table mismatch, full wake prelease validation, consumer ContextVar leakage,
  conflicting factories, model dispatch/result exposure and legacy/cold setup.
  Scoped callback reset/default and stream-handoff denial controls supplement the
  owning legacy suites. Schema-only mocks remain no native ingress/protection proof.

- `test_campaign_runtime_entry.py` uses exact current initializer/coordinator
  stores with neutral private ingress/artifact ports and ordinary API/ChatEngine
  entry fixtures. Original reds require normal context/thread/reset and no-agent
  unknown wakes; further reds close untyped equality claims and transient policy
  snapshot aliasing. Cold-object recovery preserves protected originals after an
  honest lease expiry; current native/ingress/clock/issuer/ref/operation mismatch,
  verifier uncertainty, public fields and inherited/unsealed contexts refuse.
  Default legacy entry and cancellation/nesting remain covered. No mock/source
  pass proves native ingress, C+H protection, real restart/redelivery, authority,
  artifact/ML admission or full one-prompt lifecycle. Root owns processing journal
  bridge and final native/ordinary agent integration qualification.

- Neutral initialization child processes derive source/test import paths from the
  owning test file and extend a copied environment, preserving existing CPU
  dependency/metadata paths and offline variables. Parent PYTHONPATH need not
  contain tests. Explicit child stderr accompanies exit-code failure; the 15s
  timeout, distinct PID/UUID/prompt/call equality and zero consumer writes stay
  unchanged. This test-route fix grants no production/native authority.

- The initialization error-envelope regression combines committed RetryAttempts1,
  one confirmed outcome read and final target5 protection expiry. Its retained
  actual red requires transport classification, the qualification cause, historical
  original diagnostics and permanent currently-unqualified refusal. Original C/H
  remain intact and fresh read-only recovery makes no new write; no test approves
  native protection, application activation, redelivery or an ML effect.

- Initialization review regressions retain actual reds for protection expiry during
  final native admission at read/write/return, wrong pending-prompt creation and
  contradictory write retries (including an unavailable outcome read). SDK/model
  planning delays precede final qualification. A deterministic barrier forces two
  initial empty reads and two competing tokens before winner recovery. A separate
  producer process fsyncs neutral fake C/H storage; a distinct consumer process
  recovers original IDs/prompt/reference with zero writes and fresh fixture approval.
  This is source-library process recovery, not AgentCore/native credentials,
  protected retention, redelivery or ML-effect acceptance. The existing same-process
  new-object control is named accordingly; no production grant is persisted.

- `test_campaign_initialization.py` uses harmless SDK-model-validated transaction
  fakes and explicit trusted-code qualification fixtures. It retains actual reds
  for missing atomic protocol/protection port, malformed absence causing creation,
  configuration/digest substitution, unknown-outcome write replay and admission
  revocation during current protection read. It verifies original winner/cold
  object recovery, original token/one outcome read, deleted-C/orphan refusal,
  current evidence PID/reader/type/finite interval/config/clock checks and strict
  native item-size bounds. Fixture greens prove no IAM isolation, protected
  publication role, retention, table identity, permissions, true runtime restart,
  app continuation, native admission or external AWS effect execution.

- Admission account-drift fixtures use the reviewed deliberately invalid all-zero
  placeholder; unreviewed numeric/account values remain release failures.

- `test_campaign_native_admission.py` uses installed SDK-shaped neutral private
  transports for fixed fresh reads, raw policy preservation, canonical SDK trust,
  missing evidence/role/endpoint/tag/type/config/PID refusal and no mutation path.
  Actual Guard consumers retain honest None for missing/null target evidence and
  require refusal against an approved reported target string. Fake/shape greens
  establish no native read permission, cohort approval, caller identity, atomic
  IAM snapshot, SDK authentication, table durability or campaign activation.
  Semantic review reds require explicitly reported HTTP and a model-conforming
  positive fixture. A separate retained-native DEFAULT-id control preserves its
  non-authoritative status and does not imply all modeled patterns are enforced.

- `test_campaign_admission.py` uses neutral explicit server approval fixtures and
  closed native metadata to retain absent/unapproved/blank-policy denial, exact
  policy/role/endpoint drift, type/boolean/PID/clock/deadline and changed sealed
  interval refusal. Fixture comparison grants no native cohort/IAM authority.
- `test_campaign_coordinator.py` uses installed SDK-shaped neutral DDB and native
  observation ports for immutable literal mapping/cold original UUID replay,
  bounded inputs, exact conditional/unknown response handling, lease epoch/expiry
  fencing and registered Training/Processing observations. It covers all nine
  actual phase-kind pairs, fresh admission before reads/CAS/observer, duplicate
  event/terminal regression and unknown last-confirmed preservation. It creates
  no native jobs or table and proves no native authorization/durability/full graph
  recovery, exactly-once AWS effects, asynchronous health or trained quality.


- `test_campaign_admission_review.py` retains actual neutral full-code review reds
  for coherent immutable-binding replacement, inherited coordinator ownership,
  inside-window clock rollback, exact expired-lease acknowledgement resolution and
  finite-number overflow. It checks qualified138-byte original references, cold
  transport-only/legacy refusal and independently retained original-ID positives;
  no test authenticates a cold row using its own hash or grants native admission.
  Neutral pre-lock probes and an actual bounded fork while a helper thread holds
  the parent guard mutex require inherited verification to refuse without I/O or
  deadlock. The owned child is reaped and helper released; an independently
  constructed child guard remains supported. No candidate/oracle code executes.

- `test_runtime_campaign_policy.py` covers explicit deployment-only typed approval,
  default denial, malformed/unrelated action refusal, real approval decisions,
  public payload override rejection and unchanged conversation-only evidence.
  These neutral fixtures grant no native lifecycle or durable continuation proof.

- `test_runtime_busy_ownership.py` uses the installed BedrockAgentCoreApp and
  neutral blocked API workers to require task registration before worker entry,
  independent worker/task ownership, completion on actual worker finally,
  no accepted task on startup refusal, and busy ownership after stream deadline
  or disconnect. Ping timestamps change only with actual SDK status transitions.
  These controls make no provider calls and establish no durable continuation.
- `test_runtime_prompt_entry.py` covers unchanged prompt delegation, closed public
  fields, typed conversation-only completion, safe approval default, trusted ContextVar
  propagation, heartbeat and unknown deadline outcome while preserving the legacy spec
  route. Ordinary fake API responses grant no deployed runtime, durable session,
  async task health, training or quality evidence.

- `test_qwen_cpu_ci_provisioning.py` requires both real CI paths to provision the dedicated hashed CPU closure and exact five metadata pins before complete pytest, plus a strict CPU dependency audit. It retains corrupt-byte/destination refusal and CPU-only/no-CUDA checks; actual tokenizer/TinyQwen/PEFT cases remain required rather than skipped.

- `test_audit_pin_sets.py` runs `scripts/audit_pin_sets.py` as a subprocess on a generated tree with an offline stand-in. It writes the stand-in as a `pip_audit` package in its temporary folder and puts that folder on `PYTHONPATH` for the script. It requires the script to refuse `--pip-audit` with a usage error and exit code 2 before running anything. It requires exit code 2 for `--root --help`, with `usage:` and `argument --root: expected one argument` on standard error, empty standard output, and no start of the check tool. It requires every tool start to use the test's interpreter, `-P`, `-m`, `pip_audit`, the fixed options, and the requirements and report files for that round in the script's own temporary directory. It puts a `pip_audit` package in the working folder and requires that it does not run, that all four rounds use the stand-in named in `PYTHONPATH`, that each start includes `-P`, and that the reported row still makes the script exit with code 1. It checks all three sets and their fixed options. It covers clean reports, unaccepted rows, ID and alias matches, inclusive expiry, and rejection of expired, unmatched, incomplete, or wrong-scope entries. It covers tool version mismatches and three tool errors. It sets `TMPDIR` for the script's process to a temporary directory below the test's own temporary folder, so the test file also passes where the system temp folder is absent or read-only. It requires exit code 1 when the tool writes a complete report with no rows and exits with code 1 or with a code other than 0 or 1, a report leaves out a pin, the tool reports a pin as skipped, a tool file names no version or is absent, or a requirement has no lowest version. A single-set run leaves other sets' entries alone. It checks the shipped file's empty state and format and the repository's shared uv version. It runs without skips.

- `test_github_workflow_contract.py` requires `uv run --frozen --no-sync python scripts/audit_pin_sets.py` exactly once in the GitLab `pip-audit` job script. It requires the same line exactly once as a step in the GitHub `verify` job. It runs the fetch step of `processing-bootstrap` with `bash -e` and a stand-in `docker` and checks that the pulls use `linux/amd64` and run in order, with a second pull only if the first fails. It checks that success passes on the reference that was pulled and that two refusals pass on nothing, fail the step, and end with a line saying no source delivered the image. It also checks that each container step runs the supplied reference once when the variable is set and fails without starting a container when it is unset. Nine changed copies of the workflow text show that each part is checked.

- Governance regressions distinguish exact reviewed public model commits from
  numeric fragments, altered tokens and separately retained private identifiers.

- Maintain the interfaces and file formats owned by the modules listed below; parent contracts apply.
- Inspection suites cover native SDK shapes, exact-version staging, image admission,
  strict fixed outputs, guest refusal boundaries and opaque authority transport.
  Regressions admit only fixed native defaults, reject environment/resource/runtime
  deviations and rehashed target substitution, and bound JSON depth before decoding
  across supported encodings while preserving escaped-string behavior.
  Permanent-drop regressions cover root/saved IDs/groups/capability/NNP refusal
  before SDK, strict receipt proof, isolated import order and proxy presence
  without values. Fixed initializer/image/readback binding remains a native gate.
  Version2 regressions require source binding before SDK/schema, original-component
  no-follow input admission, authoritative FD-relative exclusive JSON publication,
  output substitution refusal and strict descriptive directory/current-lease proof.
  Same-inode equal-length input mutation with restored mtime must refuse before
  either output file; staged input readonly behavior is not assumed by this oracle.
  A separate coarse-time oracle returns constant same-FD stat signatures during an
  equal-length mutation and still requires refusal through post-parser SHA/count.
  Persistent same-inode SUMMARY/INSPECTED mutation at fsync must refuse through
  bounded retained-write-FD actual byte count/SHA readback before worker success.
  Closed fixture probes do not establish native guest credential/egress absence.
- `test_preparation_appmanaged_readback.py` retains saved native input/output
  false defaults across v1/v2 running/completed/failed observations, original
  handle identity and non-mutating comparison. True/null/0/foreign fields refuse
  before output reads; versioned outputs and rehashed provenance/proof/count
  substitution checks still apply. These offline fixtures grant no native admission.
- `test_inspection_distribution_readback.py` replays accepted native handles against
  documented `FullyReplicated` readback without resubmission, including running/
  completed200 replies, versioned schema2 reports, non-mutating defaults and
  sharding/null/type refusal before output reads. Fixtures use public placeholders.
- `test_inspection_native_ceiling.py` preserves old strict schema/function tests and
  covers additive14-field native proof, fixed ceiling/FSID/securebits drift refusal,
  source-owned default/entrypoint/schema2 discriminator and parent expected-profile
  plus native provenance binding. Its closed shared-guard/SDK fixtures establish
  consumer behavior only; native isolation, usernamespace denial and training remain
  separate gates. No selector or synthetic admission is introduced.
- `test_huggingface_ingestion.py` covers real Hub metadata shapes, pin/hash/size,
  confined artifact commits, guarded binary transport and nested data summaries.
  Integrity/metadata cases call the portable core; public-tool caller authority
  belongs to the separate acquisition binding and native SDK suites.
- `test_hf_core_fd_authority.py` retains original directory FDs across actual
  root/ancestor swaps, checks actual mkdir authority, same-OFD lease retention,
  no resolve/string reopening/fallback, local-root scope and unchanged public
  signature/tool schema plus real replacement/same-inode mutation at stream and
  commit boundaries, including actual rewrites with deliberately unchanged stat
  signatures. These offline controls do not grant native admission or universal
  immutability.
- `test_huggingface_commit_boundary.py` covers destination collisions/races and
  aggregate preview limits; `test_parquet_materialization_bound.py` covers decoded
  batch rejection before full Parquet combination.
- `test_hf_core_review_regressions.py` covers post-check publication races,
  scalar target summaries and JSON decoded/width/dense-allocation rejection.
- `test_agent_deployment_context.py` covers configured references and selected mutation scope.
- `test_generation_task_contract.py` covers generation routing/spec round-trips and tabular controls.
- `test_qwen_sft*.py` covers actual pinned tokenizer render/token/span equivalence,
  tools/reasoning/multiturn/padding and explicit developer refusal; a real tiny native
  Qwen/PEFT model proves backward, changed language adapters, frozen vision, safe save
  and a fixed fresh-process reload. Recipe regressions separate unique corpus floors
  from repeated exposure. Closed staging/source/provider/FD-before-import guards,
  ancestor aliases, forged checkpoint fields, exact raw startup digest transport and
  retained-byte mutation at checkpoint/final publication are red-first oracles.
  The dedicated CPU environment and five hash-pinned public metadata files are
  prerequisites via AMLAS_QWEN_TEST_METADATA. No tensor weights are test fixtures;
  tiny CPU and Docker source checks do not claim9B/GPU/native admission or quality.
  `test_qwen_sft_compact_reload.py` crosses the real pinned PEFT target-compaction
  threshold with synthetic TinyQwen, requires reconstructed complete full paths at
  attachment and exact restored state, and preserves missing/extra/mixed/duplicate/
  vision/partial selector plus complete native tensor-inventory refusal.
- `test_model_weight_manifest.py` covers one exact source-owned9B expected-byte
  catalog, public original-index hash/count/shard oracle and JSON/object/descriptor
  drift refusal. It invokes no network, model loader, acquisition or authority grant.
  The approved fixture contains metadata only; candidate lists do not prove fit.
- `test_generation_planning_types.py` preserves captured legacy serialization
  bytes and checks normal-loader drafts, bounded immutable pins, five-field JSON
  references, nullable baseline declarations, numeric/control-character refusal,
  extra-grant denial and task/metric consistency. These pure schemas do not prove
  native readiness, object trust, recipe admission or trained quality.
- `test_sonnet_defaults.py` and `test_sonnet_imputation.py` cover shared model
  defaults, direct consumers, explicit overrides and native parameter compatibility.
- `test_prompt_policy.py` covers normal prompt delegation and caller authority
  transport/snapshot with malformed model-authored policy controls.
- GPT reviewer and Harness catalog tests cover actual model capabilities, strict
  cache/sampling fields and review-model invocation grants.
- `test_mcp_security_floor.py` rejects vulnerable MCP resolutions/major drift,
  checks exact1.30.0 PyPI artifact hashes in the real lock, and requires the running
  SDK test environment to import fixed MCP1.x. Manifest/lock checks supplement
  meaningful SDK policy tests and the strict exported-third-party dependency audit.
- `test_sdk_prompt_policy.py` exercises the installed SDK concurrent tool path
  and real approval gate, including spoof denial, snapshot/reuse and overlapping
  session isolation. It also records the capability-only direct-tool boundary.
- `test_processing_initializer.py` covers native fixed entrypoint/readback and
  refusal before acquisition argument/core parsing. The explicit remote Docker
  acceptance oracle proves real root0755 bind writes after verified drop, private
  permissions, root-regain denial and refusal of unsafe filesystem/environment/
  profile inputs. Rust descriptor tests reject name swaps and symlink aliases.
  `test_acquisition_fd_boundary.py` rejects intermediate aliases and proves
  checked root/descendant swaps cannot redirect hashing outside the held authority.
  Same-inode/same-length publication hooks against data, lineage and COMMITTED
  refuse success and remove the owned receipt after verified signatures change.
  A valid same-length/same-inode generated receipt rewrite at fsync must also
  refuse through retained-publication-FD expected-byte/count/digest checks.
  Controlled source mismatch refuses before mutation; an independent helper must
  refuse while the first worker retains its output lease. No new ready grants exist.
  `processing_capability_fixture.rs` is a fixed test-only static Rust launcher,
  excluded from production images. It proves actual post-exec effective/permitted/
  inheritable capabilities0xcb with empty bounding set and no CAP_SETPCAP, READ0
  and absent DROP EPERM, then requires permanent-drop worker success. A present
  bounding set without CAP_SETPCAP must still refuse before workload execution.
  Static ELF/fixture proof does not grant native credential/network admission.
- `test_processing_native_identity.py` validates the shared fixed14-field native
  guard and independent old9-field strict guard with no caller/profile selector.
  `test_acquisition_native_profile.py` covers source-owned pre-parser native refusal,
  schema2 observed proof publication and unchanged schema1. The explicit native
  controlled oracle uses the measured0x200404fb/noSETPCAP kernel, validated actual
  setuid/setgid/filecap positive controls, restore/NNP/ambient/capset/fs-ID denial,
  fork/exec inheritance, initial drift-before-mutation and real setter failures.
  Self-userns setns is recorded as limited evidence; alternate-target/native gates
  remain separate. Its fixture binaries never enter a product image.
- `test_processing_ci_contract.py` requires both source CI paths to check the
  initializer crate with the image's pinned compiler, exact-toolchain formatting,
  locked/offline tests and static release build; the public check container is
  offline and mounts source read-only. Pipeline-shape tests supplement real
  remote toolchain execution. It requires that `processing_bootstrap/Cargo.toml`
  have no dependency table of any kind at any depth and that
  `processing_bootstrap/Cargo.lock` list only the initializer crate itself. Nine
  changed copies show that the check can fail with a message telling the
  maintainer that no dependency check reads the lock file, so the first registry
  crate must come with a check of its own.
  It requires the GitLab pipeline to name exactly two images, the default
  `public.ecr.aws/docker/library/python:3.11-slim` image and the compiler image,
  both by digest and with a 64-character SHA-256 digest for the default. It
  refuses top-level `image` and `services` keys and `services` keys under
  `default` or any job. Four changed copies of the pipeline text show that the
  first check can fail, and four more show the additional refusals. It requires
  exactly two image references by digest in the public workflow, both in the
  first step of `processing-bootstrap`: the recipe's compiler image on Amazon
  ECR Public first and the same tag and digest on Docker Hub second. Four
  changed copies of the workflow text show that this check can fail. Three
  copies change the second source. One copy makes a container step name the
  image by digest itself. It also searches the whole workflow text for a further
  image reference by digest. One changed copy, with a reference by digest in the
  `verify` job, shows that this search can fail. It refuses `container` or
  `services` keys in any job, `docker://` in any step's `uses` entry, and the
  word `docker` in `run` text outside `processing-bootstrap`. It requires the
  run steps of `processing-bootstrap` to start `docker` exactly three times: one
  `docker pull` in the fetch step and one `docker run` in each of the two
  container steps. It requires the word `docker` exactly five times in the whole
  workflow text: in those three starts and in the two image references. So a
  quoted command name or a step's `shell` key that spells `docker` is noticed.
  It requires both container steps to run the reference passed in
  `PINNED_COMPILER_IMAGE`. Eleven changed copies of the workflow text show that
  these checks can fail. It requires the name `PINNED_COMPILER_IMAGE` exactly
  three times in the workflow text: in the fetch step's write and in each
  container step's read. So an `env` entry of the workflow, a job or a step that
  spells the name is noticed. It requires the run text of `processing-bootstrap`
  to name `GITHUB_ENV` once, so a second write to the job environment by that
  name in that run text is noticed. Five changed copies of the workflow text
  show that these checks can fail. It requires `--platform linux/amd64` among
  the options of each of the two container steps in `processing-bootstrap`. It
  does not count the same option on the fetch step's pull command. Four more
  changed copies of the workflow text show that this check can fail. These image
  checks read only the two CI files and do not follow scripts that steps run,
  actions that steps use, or files that the pipeline includes. They do not see
  another container tool in any job of the public workflow, a `docker` command
  in a GitLab job script, or a command name that a step puts together while it
  runs. The counts of `docker`, `PINNED_COMPILER_IMAGE` and `GITHUB_ENV` look
  for each word as it is spelled, so they do not see a word written with a YAML
  escape in an `env` entry or a `shell` key, a word put together at run time, or
  a write to the job environment that does not spell `PINNED_COMPILER_IMAGE` and
  either goes through the `github.env` context or stands in a `shell` key.
- `test_acquisition_failure_diagnostics.py` covers closed failure-stage/type/cause
  propagation and raw URL/data-detail refusal without assumed image permissions.
- `test_native_acquisition_profile.py` retains strict schema1 and rejects native
  profile/proof/handle drift before payload or native I/O; observed guest claims
  never attest isolation. `test_preparation_source_binding.py` proves one verified
  private snapshot, original pins, immutable manifest refs and session boundaries.
- `test_managed_dataset_acquisition.py` verifies native request shapes, ownership,
  configuration boundaries, honest async status and data-only manifest integrity.
- `test_acquisition_binding.py` covers typed authority, unknown-cost safety,
  previews, opaque reference isolation and per-turn snapshots.
- `test_sdk_acquisition_binding.py` exercises the actual SDK tool loop and
  delegated context boundary with harmless scripted model/provider effects.
- `test_acquisition_release_config.py` captures runtime selector propagation,
  immutable pair/native role validation and post-READY publication choices.
- Managed acquisition tests also cover egress snapshots, versioning, pinned
  artifact refs, independent bounded raw hash I/O and equal-length tampering.
- `test_baseline_flow_safety.py` covers typed native waiting/failure observations,
  accepted identity preservation, missing/malformed/contradictory grading refusal,
  real deterministic tabular acceptance/rejection, approval denial and the actual
  capability caller's nested result serialization. No native jobs or generation
  grader/durable resume are established by these captures.
- `test_training_config_parity.py` captures real request-building controls,
  validation/target propagation, deadline precedence and pre-write rejection,
  retaining native accepted-job recovery and selector checks.
- `test_inspection_release_config.py` captures image/source/admission transport,
  fail-closed mismatches and post-READY publication without synthetic admission.
  Real setup validator/publication fragments plus preseeded owned SSM state prove
  omitted admission cannot leave an old marker beside a new pair, foreign ownership
  stays protected, and read-only setup/direct runtime bootstrap remain available.
- `test_coding_preparation.py` exercises synthetic parsed conversations only:
  exact resolution/license filtering, native-ID/null preservation, gold and
  secret projection, known invalid versus unadmitted causality, bounded embedded
  JSON before decoding, global exact clusters and bridge-safe grouped splits,
  shuffle equality, conservation, no silent truncation or filesystem effects.
  Incomplete selections cannot emit usable records; near coverage is explicitly
  not implemented. These tests prove no managed preparation/training admission.
- Managed canonical preparation tests use synthetic records and native-shape
  offline fakes. Binding tests cover hidden real SDK context, fresh per-turn
  policy, idempotency, opaque/cross-session denial and retained observation errors.
  Worker tests cover exact original descriptors, source/version/hash/manifest gates,
  reducer incompleteness, held input rehash and publication byte refusal. Adapter
  tests require immutable admitted native requests/staging, strict complete output
  hashes/versions/counts/profile proof and no polling writes/resubmission. Owner
  native-schema dependency is mounted read-only for isolated tests; no fixture or
  source green establishes native admission, prepared corpus or trained quality.
- Model acquisition suites use synthetic bytes and closed native SDK providers.
  They cover dedicated source-only large-file transport without altering dataset
  caps, retained-FD commit/mutation/fsync refusal, metadata-depth/strict versions,
  native execution drift and missing qualification, hidden typed caller authority,
  session isolation and accepted/ambiguous handle retention. Actual SDK policy
  compatibility controls stay separate. Provider hints/inaccessible paths and
  unknown descriptors must refuse without logging values or inventing absence.
  Fixtures/tests establish no native model download, image or training admission.

- `test_model_acquisition_proof_reuse.py` measures model-only GET byte transfers
  and requires one whole raw inventory under concurrent repeated observations.
  Fresh native/manifest/owner/exact-version HEAD checks remain mandatory; deletion,
  version/count/binding drift, incomplete reads, checksum-only initial success and
  caller result JSON cannot create or retain a private witness. Service-shaped
  fixtures grant no native model or training admission.

- Model acquisition fixtures isolate same-length wrong SHA, duplicate keys inside
  otherwise valid metadata and invalid privilege proof with consistent outer refs.
  Checksum matrices distinguish FULL_OBJECT from composite/missing exact-version
  raw reads and enforce common deadlines before/after HEAD. Post-Create decode
  faults retain original opaque handles across repeated catalog prompts; resource
  probe errors refuse while all four provider paths and fixed endpoint probes remain.
- Owned files: `test_image_version_file.py`, `test_version_single_source.py`, `test_qwen_sft_durable_submission.py`, `test_campaign_state.py`, `runtime_image_acceptance.py`, `test_astra_safeguards.py`, `test_baseline_train_flow.py`, `test_cli_persisted_discovery.py`, `test_client_deployment_namespace.py`, `test_cloudtrail_capacity_preflight.py`, `test_deployment_manifest_cleanup.py`, `test_github_workflow_contract.py`, `test_infra_deployment_namespace.py`, `test_managed_job_ownership.py`, `test_manifest_cli_credential_chain.py`, `test_pipeline_dag_compilation.py`, `test_prepare_host.py`, `test_region_client_metadata.py`, `test_repository_governance.py`, `test_repository_policy.py`, `test_research_public_fallback.py`, `test_runtime_binding_completeness.py`, `test_runtime_default_endpoint.py`, `test_runtime_deployment_advisory.py`, `test_runtime_ownership_controller.py`, `test_runtime_role_lifecycle_advisory.py`, `test_runtime_role_portability.py`, `test_setup_deployment_namespace.py`, `test_setup_owned_publication.py`, `test_setup_reuse_witness.py`, `test_shell_entrypoint_advisory.py`, `test_tavily_setup_choice.py`, `model_acquisition_fixtures.py`, `test_model_weight_acquisition_core.py`, `test_managed_model_weight_acquisition.py`, `test_model_weight_acquisition_binding.py`.

- Model normal-integration tests use the installed SDK with closed providers
  to prove startup0SDK/SSM, managed-only port selection, pure catalog discovery,
  hidden per-conversation state, fresh policy, capped/default/malformed denial,
  opaque foreignref/injection refusal and concurrent same-session delegation.
  They grant no native image/model acquisition or trained-quality admission.

- `test_open_swe_canonical.py` uses synthetic parsed rows reflecting primary schema:
  explicit license/resolved/identity/source pins, bounded serialized tools, exact causal
  IDs/nullability, ignored gold/model material and spoof/unknown-profile refusal.
  Missing official-schema result IDs are a reported gap, never invented associations.
  These unit cases prove no dataset parsing, native admission or prepared task corpus.

- `test_open_swe_derived.py` uses complete synthetic source-shaped rows to prove
  qualified singleton linkage, full prevalidation, exact source/digest/edge lineage,
  ambiguity/null/refusal boundaries and unchanged strict v1. These are pure unit
  controls, not truncated native samples, corpus acceptance or training proof.

- `test_public_dns_pipe.py` uses actual fork/anonymous pipes, fake DNS and a
  read-only /dev/shm/network-none focused fixture. It proves semaphore/pickle
  independence, measured atomic framing, strict wire/IP validation, malformed/
  partial/64-bit/oversize/EOF/helper-failure refusal and bounded reaping/deadlines.
  One dependence on machine speed remains: on a machine starved to a few percent of a
  CPU, a helper can outlast both 0.1 s cleanup waits after SIGTERM and SIGKILL, so a
  case can get the cleanup refusal instead of the expected refusal.
  Cases about a frame, guard, schema or cleanup rule give the helper `LIVENESS` (60 s).
  The budget only ends a hung case; a call returns as soon as the helper answers.
  Match every expected refusal from `_checked_addresses` by its reason so the time
  limit cannot stand in for the rule under test. Each URL guard case matches the reason
  of its own guard. The two decoder test functions call `_decode_dns_wire` directly
  and expect only `PublicFetchError` because the decoder gives the same reason for
  every rule.
  Stalled helpers sleep for `STALL` (five times `LIVENESS`). Their cases assert that
  the refusal arrives within `LIVENESS`, a bound that only ends a hang.
  The partial-frame cases and the time-limit case record every wait the parent asks
  for: the timeout of each `poll` of its reader and each `join` of its helper, and the
  duration of each `sleep` of the product module. They check those values without
  reading a clock. In the time-limit case, no poll asks for more than the case's budget
  and all waits together stay below 1 s. In the partial-frame cases, the first wait is
  the poll that meets the partial frame and all later waits together stay below 1 s.
  The time-limit case keeps `Process.start` from returning until the helper reports
  that it ignores SIGTERM. The case requires exactly one SIGKILL.
  In new cases, assert promptness or enforcement of a time limit using the waits the
  parent asks for, never elapsed time.
  The deadline recheck uses a stand-in clock. A fixture ends any helper that a
  failing case leaves behind. Never give a real fork a budget of a few seconds
  in a new case.
  These tests add no native model, filesystem allowlist or egress admission.

- `test_managed_open_swe_preparation.py` uses opaque synthetic Parquet bytes and
  native-shape providers to prove trusted profile selection, format/byte/version
  preservation, no binary line counting, original-row/normalization/core-drop
  conservation and qualified lineage refusal even after outer hashes are recomputed.
  The v1 positive control remains unchanged. Parent parses metadata sidecars only;
  tests establish no real Parquet schema/corpus/native/training acceptance.

- `test_commitpack_profile.py` exercises complete synthetic ten-string rows,
  pinned typed provenance, all repository/commit/duplicate union, deterministic
  split conservation, fixed SPDX mapping and exact code preservation. Adversarial
  null/oversize/secret/control/extra-field/custom-object and malformed envelopes
  refuse before callbacks or encoding. Advisory regressions retain everyduplicate
  origin, refuse immutable-origin conflicts and close absent typed fields/early
  dictionary-width copying. Fixture green does not prove raw corpus,
  managed parser, source licenses, held-out coverage or4096/token-floor acceptance.

- CommitPack preparation/worker/managed tests exercise complete synthetic rows,
  all-origin conservation, nonrepresentative component exclusions, private exact
  source selection, typed/canonical exclusions, caps and prefix-only failures.
  Parent replays normalized content/group/licensing/membership and verifies
  exact-version outputs; tampered metadata remains refused even with recomputed
  outer file descriptors. An explicit positive characterization retains the
  unused-message metadata authentication limit; caller-private trusted hash/ref
  enforcement is separate. Legacy v1/v2 and hidden authority tests remain intact.

- CommitPack review regressions bind the bytes actually consumed by JSON decoding,
  account retained rejected rows and assembled output envelopes, and retain trusted
  per-source stream counts privately. Parent replay requires exact counts for every
  source; trainer may omit that independent constraint and must not claim raw-source
  authentication. Hash/parse/restore and coherent101/99 substitutions refuse.
- `test_qwen_sft_provider_inventory.py` uses actual readonly four-sealed memfd
  bytes plus closed current/provider/network fixtures. Malformed/extra/deep/unknown
  inventory, hints, writable/unsealed FD4, present providers, resource errors and
  same-bytes descriptor swaps refuse. Main rechecks actual states/hash/identity after
  core import before operation. Existing manifest-only positives explicitly scope
  their provider-guard fake; independent kernel controls verify the real boundary.
  `qwen_trainer_provider_kernel.py` is a fixed test-only OS startup fixture: root
  captures original absence without contents, seals FD4, permanently drops identity
  with empty groups/active caps0/NNP1 and uses the actual shared guard/FD3 lease.
  Actual root0700 permission denials remain denied_after_drop; extra/unsealed/
  writable descriptors refuse. Controlled Docker network-none evidence does not
  qualify SageMaker/GPU/native isolation or source authenticity.


- Fixed native Qwen SFT tests use actual SageMaker/S3 service-model request validation plus a fake that rejects small ranged UploadPartCopy. They cover exact versions/SHA/counts, large multipart staging, original COMMITTED descriptor propagation, honest nonterminal/failed/fit/full outcomes, closed checkpoint/reload envelopes, rehashed tampering and heldout refusals. Original typed store/actual Strands tests prove concurrent duplicate reuse, fresh policy/defaultASK/cap denial, cross-store/session refusal and model hidden-parameter injection denial. No mocked green admits real GPU/provider/input ownership or uncompressed native output layout.

- `test_qwen_sft_public_binding.py` checks exact safe-tool/schema/classification
  and specialist lists, registered fit/full plan metadata, actual typed SDK lead
  wrapping, pure same-original-store factory, missing/nonstring/cross-session
  refusal and ASK/model-dict zero effects. Scoped factory tests substitute only
  lazy registry resolution while Root separately qualifies registered/context
  paths; these controls do not claim native GPU or hidden-store delegation proof.

- `test_qwen_sft_root_routing.py` verifies fixed managed port/config/spend classification, normal direct-tool confirmation and private context, caller spoof refusal, and actual SDK same-object versus foreign delegation. These controls grant no native GPU admission.


- `test_coding_case_codec.py` uses only synthetic164 public case strings and source metadata. It freezes complete order/schema/private roles, mutation/no-truncation bounds, unchanged code-pair hash domain and independent original-field replay. Rehashed inconsistent projections refuse; a coherent whole-bundle replacement is explicitly characterized as unauthenticated without the later managed raw hash gate. Real control bounds/codec compatibility are measured outside product source and do not execute benchmark code.

- test_qwen_sft_stage_diagnostics.py checks closed-stage refusal, no raw exception leakage, bounded framework versions and fixed managed-stage wiring in the inert image namespace; no native/GPU/model execution proof.

- test_qwen_sft_precore_diagnostics.py verifies real injected inventory/credential refusal attribution and fixed metadata/unknown codes without provider/error content. It establishes no native absence/admission.

- `test_coding_arbiter.py` proves exact paired2x164 arithmetic, complete ID/oracle/execution inventories, bound source/runner/runtime/budget/model arms, closed typed data and public JSON refusal. Failure/infrastructure outcomes retain denominators; exit0/stdout are not oracle completion. The owning pure factory and private report grant no native/containment/grading-integrity/quality acceptance. Ordinary predictive graders remain unchanged.
- `test_campaign_state.py` validates DynamoDB/STS/SageMaker request shapes
  against installed botocore service models with closed offline fakes. It covers
  concurrent conditional reservation, original identity/disposition retention,
  stale writes, closed typed codec/digest tamper, native role and foreign
  deployment/campaign/session refusal, absent serialized policies, and read-only
  observation with zero Create/writes. Fakes establish no live table durability,
  caller end-user identity, native admission, artifact validity or trained quality.
- test_preparation_case_flow.py and test_preparation_case_state.py cover fresh approval/no-read previews, same-store original case binding, immutable dedup identity, per-job catalogs, namespace/role mutation refusal and absent operator startup exclusions. No AWS job executes.

- `test_coding_generation.py` requires all164 public prompts, oracle/extra-field refusal before model calls, fixed identical pair budgets, exact template/render/token projection, terminal-token/output bounds and explicit no-repair/no-retry failures. Actual tiny native Qwen/PEFT CPU generation is separately identified from deterministic fake164 controls and proves no9B/GPU/held-loader admission.
- Case image contract/source startup/package closure tests verify fixed source equality before inputs and actual isolated imports with SDK/model/application modules blocked. Native image execution is separate.

- `test_qwen_sft_durable_submission.py` uses strict botocore-shape/conditional
  and native/S3 fakes. It covers one concurrent staging/Create owner, lost
  reservation/marker/snapshot/outcome acknowledgements, accepted-handle retention,
  immutable attached evidence, complete request/RUN/source/version/count refusal,
  native rejected/ambiguous outcomes, schema1 descriptive reads/CAS upgrade,
  stable opaque refs and fresh same-store recovery, missing upstream/current-
  catalog refusal, read-only restored fit/reload, and unchanged uninjected flow.
  These controls establish no real table durability, GPU/artifact admission,
  end-user identity, runtime/upstream restore or trained model quality.

- `test_account_image_build.py` uses neutral offline metadata and command fixtures
  for Darwin/foreign-account/Region/STS/remote-daemon refusals and real callsite
  ordering. It performs no native metadata request, Docker mutation or AWS call;
  fixture success does not qualify a real EC2 builder.

- `test_coding_executor.py`, `test_coding_execution_records.py` and `test_paired_coding_runner.py` retain durable original attempt/unknown identities and the2x164 denominator. Neutral delayed transport fixtures prove actual Linux main-thread interruption at the unchanged ten-second case deadline and remaining thirty-second shared stream-close/Stop/Get cap, single-owner cleanup, close-all/deferred-final uncertainty ordering, no postdeadline delayed-writer extension, no reinvocation, setup/teardown signal-window restoration, preservation of unrelated cancellation and refusal of unsupported or conflicting signal contexts. The real ten-second neutral unknown-fsync probe measures cleanup onset before/after a bounded one-second injected delay; its10.7-second observation discriminator never changes the fixed10-second budget or grants native SLA/quality acceptance. Expiry preserves unresolved original durable prefixes, original stream close and single Stop/Get cleanup. Fixture success grants no native admission or coding-quality proof.

- `test_paired_coding_runner.py` also models a flushed verdict before interpreter exit and requires zero normal process status plus EOF without killing a successful child. Neutral fixtures reject nonzero/signal exit, trailing bytes and post-verdict timeout, preserve unrelated cancellation and measure actual main-thread interruption of a mock blocking wait at the unchanged10s deadline; they spawn no subprocess or official/candidate code. Original child cleanup and the328 denominator remain required.

- `test_oracle_launcher.py` covers the actual fixed argv/environment seam, pinned image build/copy contract and exactly four reviewed official-import file leaves while freezing the original runtime catalog. It never executes official source, the launcher or any confinement syscall. The matching Rust regression requires READ_FILE-only leaves, unchanged import-directory rights and no `_typing` grant. Rust mock and real native checks run only in the AWS account.

- Oracle launcher positive metadata fixtures model root UID/GID and nonwritable directories consistently, independent of temporary-directory modes. `test_oracle_launcher_metadata_portability.py` separately retains actual writable selected ancestor modes, owner/group/file mode, symlink and digest refusals with scoped monkeypatch restoration. Production verifier bytes/guards remain unchanged; fixtures prove no actual root-owned filesystem, launcher execution or native containment.

- `test_oracle_launcher_binding.py` uses neutral source/metadata mocks for ARM-safe pure expected-digest binding, malformed-selector refusal before closure reads, independent fixed installed-byte match and worker verification before input parsing. The ordering fixture explicitly models the fixed image UID/GID/groups10001 independently of the test runner account; production identity refusal remains unchanged. No launcher execution, role call or native admission occurs.

- `test_sft_producer_snapshot.py` exercises closed producer persistence, original
  alias/operation scope, coherent type/source drift refusal, exact lost-response
  readback, concurrent publication and fresh per-read policy. A separate process
  rebuilds empty upstream stores and fresh real adapters, then reaches normal
  api.run_prompt/get_qwen_sft over strict neutral SDK/S3 fixtures; all Create and
  staging mutations are blocked. Synthetic byte/catalog pins grant no live model,
  native durability, transport/end-user identity or full campaign recovery proof.

- `test_sft_restore_aliases.py` obtains original refs through normal preparation/
  SFT/reload tools, then repeats normal prepare/reload through fresh-process
  api.run_prompt with fresh real adapters and strict neutral SDK/S3 fixtures.
  Original aliases and one Create per original operation must remain unchanged.
  Closed schema2 controls refuse missing/duplicate/index/parent/rehashed drift,
  revoked current policy and absent original provenance. A separately retained
  publisher acknowledgement supplies the private expected SHA in cold fixtures;
  valid-alias coherent rehash, missing/malformed/unequal/foreign commitment,
  implicit row-selfhash and absent selected schema2 refuse. Actual two-source
  preparation preserves sorted aliases and exact source indices through fresh
  normal reuse. Schema1 descriptive recovery remains valid. Fixture transport
  does not qualify a protected server handoff. No native workload,
  case/full-graph/app recovery or cached approval is established.

- `test_sft_producer_review_fixes.py` uses real botocore retry logic with an
  in-memory neutral HTTP transport, uninjected preparation client factories,
  failed/unknown/completed/refused exposure, coherent provenance changes and
  neutral fork inheritance. It proves source boundaries, no native credentials,
  authentication, table qualification or model/hostile workload execution.

- `test_qwen_sft_reload_source_compatibility.py` stops actual managed source
  checks before ML/GPU loading: legacy same-source control, unqualified cross-source
  denial and explicit closedv2 producer/consumer metadata.
  `test_qwen_sft_reload_compatibility_path.py` covers fresh server selection, normal backend/store keys,
  schema3 cold exact-handle reuse, removed selection and rehashed selected-version
  receipt drift with strict fake SDK clients. These tests confer no native admission
  or end-user authority. Trained threshold-compacted TinyQwen uses a real optimizer
  update and fixed safe reload to compare actual adapter values and greedy output;
  synthetic CPU proof remains separate from9B/GPU reload and coding quality.

- `test_qwen_sft_reload_compatibility_review.py` uses actual fixed writer bytes
  for training/reload consumer roundtrips, coherent independently different checkpoint
  versions, historical submit-cache boundaries and removal during native/artifact reads.
  A strict per-job fake observer verifies an old reload is terminal FAILED before and
  after the distinct qualified recovery. A current accepted TRAIN entry remains
  reusable after historical restoration for the same aliases/recipe, with identical
  ref/job and zero additional Create; historical-only/wrong-catalog refusals remain.
  These neutral controls create no native jobs
  and do not establish real9B reload or atomic policy/configuration observations.

- `test_sft_producer_recovery_compatibility.py` exercises ordinary publication,
  independently committed load and temporary hydration of selected managed schema3
  recovery using strict fake SDKs. It preserves original producer/TRAIN/checkpoint
  identities, terminal failed legacy reload, digest-qualified successor reuse and
  current TRAIN dedup precedence with zero additional Create. Missing/changed/late
  revoked selection, wrong reader PID/current catalog, callback errors and missing
  independent SHA refuse. Fixture acknowledgements do not qualify a production
  protected handoff, native admission or actual9B recovery.

- Producer recovery commitment controls exercise ordinary diagnostic-only schema1
  publication and deletion of an acknowledged schema2 row. Both selected no-SHA
  fallback routes refuse before context minting; correct committed schema2 and
  genuine legacy schema1 positives remain. No graph/Create bypass is inferred from
  the retained pre-mint defect; no native service or permission is exercised.

- `test_normal_chat_live_research.py` requires actual SDK tool names/schemas and
  normal ToolBindingBackend direct/SDK dispatch of the existing search/fetch tools.
  Benign provider/DNS/HTTP fixtures retain the real aggregate/fetch envelope and
  prove explicit failures/coverage, supplied URLs only, finite page/byte/time
  budgets, untrusted data and fresh ASK policies across turns, with no AWS clients.
  This source test proves no live provider coverage, DNS/TLS admission, verified
  research, model invocation or native effect authority.

- `test_model_campaign_journal.py` uses explicit neutral current entry/protected
  original ports and SDK-model-validated conditional fake rows. It verifies exact
  comparison with Root's pure model adapter, protected publish-before-row, planning/
  source/request replacement refusal, concurrent original reserve/marker behavior,
  lost/retried acknowledgements, retained single-use permit identity, lease fencing,
  current I/O revocation and original read-only NotFound with no Create. Fixtures
  prove no native protected publisher/retention/table isolation, credentials or
  model readiness/quality; a spec pointer remains data rather than validated body.

- Normal-model review fixtures cover late active-campaign legacy refusal,
  matching-identity factory replacement, same-entry empty-to-published planning,
  post-Create revocation/outcome-record uncertainty, model boundary revocation in
  store runtime checks and failure-atomic ordinary alias publication. Separate
  planning-refresh fixtures preserve exact original identity/transport/prompt and
  old sealed view, refusing foreign/PID/reader/current-ingress drift. Neutral SDK
  fakes prove source behavior only; no native authorization or ML effect occurs.
- `test_campaign_planning_publication.py` retains missing-module/renderer/restore/
  hidden-save reds, exact legacy text and public-schema positives. Real SDK save
  uses strict neutral conditional protected ports; duplicate/lost/unknown attempt,
  current origin, coherent replacement, active model binding and cold original-ref
  controls preserve no-disk fallback and no reminting. Mock protected callbacks
  grant no native protection/retention, verified research, model admission or effect
  authority; Root separately owns runtime view refresh and native port qualification.

- Planning SDK/accessor regressions use actual installed Strands hooks and normal
  API/direct tools with strict neutral C+H transactions and protected DATA ports.
  They retain original commitments through advanced C, refuse drift/PID/current
  revocation with no writes, preserve exact cold refs and default disk denial,
  distinguish genuine legacy key absence and capture actual bounded UNTRUSTED
  research through sequential SDK cycles. Source fixtures grant no native origin,
  caller authentication, bootstrap, model readiness or fullgraph acceptance.

- `test_protected_planning_read.py` retains missing-reader reds and uses the actual
  initializer accessor, shared protected publication and model bridge over neutral
  qualified ports. It covers no pre-entry constructor/factory mutation/writes,
  full canonical JSON/hash/domain notes/raw separation, one fake Create/cold same-ref,
  explicit scope/supplier/UNKNOWN/origin/foreign-version denial and cold fresh read.
  These tests grant no native ingress, storage protection, publisher capability,
  model/provider acceptance or full lifecycle authority.

- `test_reload_fixture_checksum_gates.py` keeps exact six-line verified SHA annotations and rejects changed/extra/account annotations, unreviewed/altered/partial hash tokens, bare/nested ARN/ECR/bucket/domain coordinates and private literals. Neutral baseline fixture derivations independently preserve all six frozen expected values; secret baseline/plugins and original publication gates remain unchanged.

- `test_qwen_sft_reload_resources.py` binds independent pre-change legacy
  schema1/2/3 record and train-request hashes. Strict installed SDK fakes exercise
  ordinary same/cross-source reload schema4/4xlarge selector/digest/snapshot before
  Create, cold original12xlarge restoration and new4xlarge repeated-call reuse.
  Unknown/train selectors, missing schema4 selector and changed snapshot resources
  refuse; unchanged training and original checkpoint/source/role links remain.
  Fixtures are source behavior evidence only, never native GPU capacity/admission.

- test_model_acquisition_configuration_errors.py calls the real acquire_model_weights tool with typed neutral caller/store fixtures, refusing every SDK/SSM/model submit for absent, partial or malformed startup selectors. It preserves policy/preview precedence, original empty store, closed readonly diagnostics and unknown arbitrary adapter outcomes without raw error/configuration values. Source fixture green establishes no native acceptance, image qualification or retry permission.

- test_guardrail_outcome.py uses actual installed SDK AgentResult and the real normal api.run_prompt/ChatEngine path with an inert provider. Preserve the actual result-loss/API-action reds, unchanged reply/tool records, ASK policy, zero provider calls/retries, exact primitive reasons, legacy missing/other/non-SDK/type controls, blocked-word nonclassification and frozen UNKNOWN-assessment DATA. SDK intervention alone may reflect anonymization and must never imply BLOCKED, filter cause, native admission or permission.

- test_guardrail_assessment.py proves the installed SDK normal raw-metadata callback using inert streams, original callback/event identity/count, per-turn closure, synchronized callbacks, actual API projection and unchanged five-field fallback. Preserve actual missing-capture/projector reds and completeness reds, unsupported/unknown/mixed/overflow/detected controls plus excluded-leaf sentinels. Only safe enum records survive; no real model/provider/network or protected content is collected. Source green is not native assessment cause, policy admission or ML lifecycle acceptance.

- `test_preparation_capabilities.py` retains missing-export/ordinary-registry/schema/read-only reds, metadata parity against the existing strict source validator, closed source/configuration queries and actual installed SDK/direct no-confirmation controls. Neutral source fixtures count zero native/payload effects and establish no dataset bytes, native admission or lifecycle proof.

- Model release integration tests invoke the actual governed helper with closed
  native-shape CLI fixtures. Retain SSM-only three-name omission red, valid ENV
  no-SSM control, all-absent/unadmitted state and partial/foreign/denied/mutable/
  mismatched prewrite refusal controls. Fixtures never grant native admission or
  execute AWS, model, image publication or deployment effects.

- test_preparation_case_route_discovery.py retains actual missing resource/route DATA reds and the earlier scratch-only harness fault separately. Pure held-catalog plus actual SDK/direct controls preserve complete-catalog unconfigured status while exposing configured resources and unknown-case routing, with zero source/backend effects and no public authority fields. Greens do not prove deployed agent understanding or native readiness.

- `test_generated_coding_spec_tool_connection.py` requires the actual registered
  SDK generator to supply every spec byte consumed by validate/save/private bind.
  Only tool choices and explicit natural planning DATA are scripted; no fixture
  canonical YAML, injected public authority/ref or mocked generator is accepted.
  The original typed generation graph and zero source reads/Create remain; source
  catalog pins, null baseline/unresolved refs and declared acceptance/budget stay
  planning DATA, never native readiness or quality proof. Run neutral tests AWS-only.

### Explicit coding planning DATA connection

The real registered SDK generator's finite and per-field zero budgets preserve
owner/researched values through its own spec text, validation, save and private
binder. An explicit overall no-ceiling declaration requires no internal retreat
vocabulary. Omitted budget provenance remains source_schema_defaults. Tests use
the original trusted planning fixture with zero native Create/read effects.

The registered coding generator's public-placeholder S3 path is descriptive DATA:
column-parser and CSV-parser sentinels must remain untouched, while the normal
validate/save/private planning bind and legacy tabular contracts remain intact.

- `test_coding_spec_validation_diagnostics.py` drives actual registered SDK
  validation and private binding over the original seven-store graph. Preserve
  finite known refusals, immutable canonical bytes/deduplication and zero source
  read/Create controls; missing/foreign context and unexpected errors still deny.
- `test_warm_resume_append_only_provider.py` retains the complete original raw
  history and one private completion DATA event through the same Agent. Actual
  SDK provider-shape, Unicode4096-byte bound, closure and current-policy controls
  grant no native, coding-quality, cold-restart or full-lifecycle acceptance.

- `test_normal_research_profile.py` uses normal `api.run_prompt`, the original
  ChatEngine and installed SDK; only deployment selector and inert provider ports
  are injected, never the private research binder. Retain default-off/public
  injection holds, value and responder identity revocation, blocked-body/reference
  removal, terminal consumption, ordinary registered discovery and original
  same-Agent warm completion history/checker controls. These tests prove source
  integration, not native guard service, research correctness or ML completion.

- `test_empty_chat_context.py` uses fresh neutral API/ChatEngine/installed-SDK
  input to require no context for no state and unchanged literal owner input.
  Preserve the meaningful-state control: independently allowed components whose
  composite is blocked must hold before model/tool dispatch. No rejected native
  prompt, replay, guard relaxation or provider traffic belongs in this fixture.

- Paired deployability regressions reject a persistent final-image Python loader
  hint and require an isolated fixed-path build import with explicit `-B`. They do not prove actual
  image startup. That check retains Config.Env, normal compiled entrypoint,
  measured native capability mask and original empty output mount; bounded
  native/drop proof and the expected missing-binding refusal establish startup
  reachability only, without SDK or candidate execution.

- `test_research_exhaustion.py` uses normal API/ChatEngine/SDK and neutral inert
  source/guard ports to require two independent strict ALLOW units followed by a
  blocked last unit to receive one separately allowed fixed notice and an honest
  gap conclusion with no additional fetch/native call. Preserve all-blocked,
  masked-only prior evidence, UNKNOWN, notice masking/block/unknown, same-origin,
  current guard/policy drift and retry holds. Mixed-history controls require a
  related or masked earlier unit plus an unrelated strict-ALLOW witness to remain
  held at exhaustion, retaining history and making no additional fetch/native call.
  No rejected native payload or direct binder appears in these fixtures; a notice
  establishes no native/C1/ML grant.

- `test_model_unknown_observation_budget.py` and the updated ambiguous-missing
  original test require exactly eight qualified UNKNOWN polls to return an honest
  unresolved outcome with one SDK task completion and unchanged original handle,
  graph/history/flags. Retain no Create replay/remint, long RUNNING waits, counter
  reset before same-Agent verified completion, and policy/guard/cancellation/drain
  controls. These normal API/SDK fixtures use inert native transports only; they
  never operate the live throttled invocation, request quotas or classify it as a
  definite native rejection.

- Initial acquisition pacing tests use concurrent actual dataset/model adapter calls with a fake monotonic clock and explicit wait events. They require fixed1.25s initial-send spacing, independent validated keys, fresh post-wait qualifier refusal and one attempt on lost reply. Compatibility runs may substitute only the clock/wait under an isolated per-test state; source/native request and no-replay assertions remain intact. This process-local prevention grants no cross-runtime rate guarantee or native acceptance.

- Only the two existing 65-original fake SDK volume tests explicitly install
  `multi_producer_fixtures.install_bulk_initial_create_clock` before MODEL/DATA
  construction. The thread-safe virtual clock replaces only adapter `_util`
  initial-Create now/sleep/state ports and restores them through monkeypatch.
  Prove 66 gate effects and 65 unchanged 1.25-second virtual waits. Keep all 65
  original/count/history/once/policy/actual-worker assertions and real 30-second
  initial/observation and 5-second cleanup bounds. No autouse/external plugin,
  global time/asyncio/Event patch, production interval or real SDK pacing test
  change is permitted. This fixture proves volume ownership, not wall throughput.

- test_case_unknown_observation_budget.py uses actual normal app/API/SDK originals
  and existing inert CASE/full_v1 supplier ports. Persistent local ValueError and
  RuntimeError observations must return finite UNKNOWN after eight failures with
  original refs/IDs/history, no new producer I/O and one actual task completion.
  A healthy RUNNING observation resets the streak before same-Agent completion;
  a healthy full training job reporting three hours stays busy beyond that count.
  At the eighth observation, policy/record refusal and actual same-length immutable
  artifact rejection retain precedence over generic exhaustion. Keep all MODEL
  UNKNOWN, CASE/PREP and full-train/reload controls unchanged. These
  fixtures establish an independent source liveness defect, not the cause of the observed live failure,
  native admission, trained quality or an overall healthy-job age bound.

- `test_runtime_precommit_original_completion.py` uses the normal app/API/ChatEngine
  and installed concurrent Strands SDK, exact deployment profile and three fixed
  research batches over inert guard/model/native/artifact ports. It checks pending
  acknowledgements remain uncommitted, original final result IDs/native Creates
  occur once, prior history and the same Agent remain, no synthetic user completion
  or reentry occurs, and worker completion is truthful. Default legacy remains a
  separate control. Only the new source-owned precommit read-wait method is locally
  shortened; real deadlines/pacing and global clocks remain unchanged. Completion
  fixtures select records by captured original tool-use IDs and exercise both DATA
  registration orders; dictionary insertion order does not define tool-use order.
  Preserve the original completion waits and native/history assertions. Fixture
  acceptance does not establish deployed native guard or full ML acceptance.

- `test_input_admission_connection_lifecycle.py` models SDK pool clear/reuse on
  the same client and separately marks factory-replaced clients retired, while
  retaining strict legacy completion
  tests and drives normal app/API/installed concurrent SDK precommit MODEL+2DATA
  originals. Assert old-pool retirement before fresh-client creation/first final
  assessment, same owning Session/config/HMAC/cycle and current graph/policy,
  exact original IDs/history/Create once and one final assessment without retry.
  A fresh transport failure stays UNKNOWN/held with all originals retained and
  zero consumption or next model cycle. Inert idle flags demonstrate the source
  lifecycle omission; they do not identify the native transport failure cause.


- `test_research_hold_logging.py` uses the installed SDK normal API research profile
  with neutral source/guard ports. It retains honest last-unit quarantine, strict
  post-notice refusal and exact model/fetch/no-native counts, while requiring fixed
  content-free warning labels and no sensitive exception/policy echoes. These
  fixtures prove source logging behavior, not native cause or full ML acceptance.
  A neutral raising warning handler must preserve actual SDK BeforeTool refusal,
  sticky hold/cancellation, one model cycle and zero source/native dispatch.

- `test_callback_normal_bootstrap.py` exercises the actual mode1 handler with a
  test-owned private bootstrap factory and existing native-shaped normal routing
  fixtures. Require one factory/normal delivery, unknown dependency denial before
  I/O, preserved legacy behavior and no scope-signature or fixture-version drift.
  `test_callback_native_qualifier.py` uses supplied STS/AgentCore-control service
  request schemas and closed metadata to require current role incarnation and
  exact READY endpoint/runtime fields on each qualification. Fixture ports and
  observations establish no live IAM, native notification or automatic wake.

- `test_guardrail_pii_directions.py` synthesizes the actual GuardrailStack offline and requires all seven unchanged PII type/legacy-action entries with explicit INPUT/OUTPUT enable and direction actions tied to PiiDefaultAction. Preserve HIGH/default choices, all content/PROMPT_ATTACK controls, blocked messages and the current Fn::GetAtt Guardrail.Version selector. Missing direction fields are unspecified, not proof of service-side disablement; Source synth grants no native policy selection or lifecycle admission. <!-- GUARDRAIL_PII_DIRECTION_TEST_CONTRACT -->

- `test_nonresearch_first_hold_diagnostic.py` uses actual projection callbacks and
  neutral guard fixtures to require one fixed first-refusal warning after the
  unchanged sticky latch/error replacement. Producer and read-only variants keep
  prior ALLOW receipts, perform no extra assessment, and cannot reopen on later
  callbacks. Logs contain no content, references, exception strings or traceback;
  this tests Source diagnostics only, not a native failure cause or retry grant.

- The real SDK CASE-to-preparation handoff covers both configured startup
  exclusions and a verified original case as the sole exclusion source. Require
  canonical `case_ref` equality with the original freeze receipt, retain generic
  `request_ref` compatibility, exactly one preparation submission and immutable
  case-exclusion provenance. The warm CASE envelope requires both equal aliases;
  preparation and MODEL completions retain their exact existing shape. These
  source fixtures establish no native ML or autonomous lifecycle completion.

- 2026-11-10. `test_normal_late_resume_after_finished_run.py` runs the cold normal fixture in one process: the
  original finishes the run after its SFT job completes, and a later `resume_prompt_run` must call no model, create no
  job, log no restore line and leave the retained report unchanged. While the original is busy the answer stays
  `original_active`; after it finished, two resumes answer `completed`, which the stream Lambda's own parser treats as
  acknowledged; an original that ended before its SFT result (held) is still restored and continued.
- 2026-11-10. `test_generation_capacity_rejection_retry.py` and `test_capacity_rejection_retry_pair.py` require a
  ResourceLimitExceeded rejection to be recorded with its code and a later identical submit to create again (rejected
  twice, created once, then reused), while ValidationException, AccessDeniedException, a rejected handle without the code
  and ambiguous outcomes keep the cached answer, and a lost create response still never repeats. Through the registered
  `submit_coding_generation` tool, a base generation rejected while a reload of this run holds ml.g6e.4xlarge names
  that reload job for ml.g6e.4xlarge and says to submit again after that job ends (changed 2026-11-10: the note
  names only the type SageMaker named, `blocking_instance_type`, and no longer calls the other listed types held); the same call after the reload ends is a real create. The
  holder lookup reads a running generation of this run and skips ended jobs, other types and the request itself.
  [Added 2026-11-10] `test_rejection_type_is_one_listed_type_or_none` drives 18 native messages through the real
  adapter: a listed type named once (or the same one twice) is kept; an unlisted type, two types, another wording or
  case, an over-long, empty or non-string message read None, the note then says SageMaker named no listed type, and no
  message text reaches the handle. Other rejection codes record no `rejection_type`.
- [Added 2026-11-10] `test_sft_store_observation_selected_instance.py` pins `SelectedInstanceType` and
  `SelectedInstanceCount` in `SessionSftStore.observe` for both of its branches (durable: campaign state injected,
  the original is restored from its reservation; session: no campaign state): each of the four listed training types
  described as selected reads back as that type with count 1 beside RUNNING, a completed job (`install_output`)
  carries both keys beside `checkpoint_saved`, and a job that waits for a machine (Pending, no Selected field)
  carries neither until a later observation names the selection. Why: a store that dropped every key starting with
  "Selected" in both branches passed all earlier tests (found by a mutation check); that
  mutant now fails all 12 cases, and the same change in one branch fails that branch's 6.
- [Added 2026-11-10] `test_unmapped_native_state_names_no_selection.py` describes a preference-list job with a
  TrainingJobStatus outside the adapter's map (Deleting, Paused, a lower-case value, an empty value, no field) and its
  Selected fields present: the SFT `status()` reads UNKNOWN with an empty detail and the description in `raw`,
  `artifacts()` carries no Selected key, and the generation store, the durable SFT store and the session SFT store
  observations read UNKNOWN without the keys. The five mapped states keep the detail, and InProgress, Stopping, Failed
  and Stopped keep the keys in `artifacts()`.
- [Added 2026-11-10] `test_capacity_rejection_read.py` reads a capacity-refused reference through the store and
  the registered `get_coding_generation` and `get_coding_pair` tools. Generation: while the base job of this run holds
  ml.g6e.16xlarge (first listed since 2026-11-11) the read answers `submission_rejected`, `rejection_reason=capacity_limit`, the blocking type, the
  holder and the submit's own `next_step`; after that job ended the next step says to submit now, the same submit then
  creates exactly once and is reused, and the accepted original reads RUNNING with its selected instance and no
  refusal field. Four reads around the holder's end create nothing, describe only the running job of this run and
  leave the record object, its keys and its stored refusal fields unchanged. Further rows [added 2026-11-10]:
  a holder outside the run; a job of this run on the other listed type (named, with
  `holding_job_name` None); an unnamed limit with and without a holder; a job of this run that waits without a type
  and one whose observation reads UNKNOWN (each named, never reported as free); a holder lookup that raises (the
  refusal is still the answer); the note for a list of three types ("after those jobs end"); a submit that creates
  the job while the holders are read (the read observes the job) and one that is refused again meanwhile (answered
  from its own refusal); ValidationException and AccessDeniedException reads and an accepted read without capacity
  fields. Pair: the read answers the refusal with its fixed next step and no native read, the submit then creates
  once, a submit that runs right after the read took the record never mixes two records, and four other outcomes
  read as before.

- 2026-11-10. `test_normal_failed_sft_after_stop_report.py` stops the original after the TASK checkpoint, marks
  the SFT job Failed or Stopped and runs a replacement in a fresh process: it must create no job, stop nothing and call no
  model, answer restoring then completed, and retain one report with action `original_sft_ended` that names the job,
  the native status and the reason class; a second fresh resume answers the same states and reads the same report. The
  reason class comes from closed sets only (FailureReason prefix, SecondaryStatus), never native text.
  [Added 2026-11-10] The report facts also carry `SelectedInstanceType` (ml.g6e.16xlarge in the fixture) and
  `SelectedInstanceCount`, and neither for a job that ended before selection; `_OriginalSftEnd` keeps both or neither.
- `test_sdk_record_form.py` (added 2026-11-15):
  A real SDK Agent with a scripted model exercises capture, witness, restore and size code.
  It makes no provider, native job or AWS call. It passes on strands-agents 1.42.0 and 1.65.0.
  Captures pin exactly four session-snapshot DATA fields: `messages`, `state`,
  `conversation_manager_state`, `interrupt_state`. Captures leave live history unchanged.
  Captured messages omit `tracking_id`. Empty `model_state` is dropped.
  Other `model_state` values are refused with "public SDK session snapshot DATA inventory differs".
  Witness checks tolerate real SDK appends and top-level id changes, including after restore without ids.
  Other message changes hold with `history_prefix_refused`. Tool-result changes hold with
  `history_batch_refused`.
  Earlier-release TASK reload preserves stored bytes and `sdk_snapshot_sha256` and completes with one model
  call. Two captures equal the golden apart from `created_at`. Size checks pin the 160 KiB record-form
  boundary.
  The raw-SDK tripwire pins snapshot fields and message keys. A person decides record treatment before
  re-pinning.
  The checksum pins come from `test_normal_durable_registered_reload.py`; they are never copied.
  Worker observation qualifies the witness four times while raw history stays byte-equal and the SDK adds
  no id.
  `sdk_record_form_golden.json` holds one TASK (schema version 2) in a neutral envelope.
  Its fixed scripted `sdk_snapshot` is captured by the product under strands-agents 1.42.0.
  The golden is not an object of a real run. `GOLDEN_SNAPSHOT_SHA256` pins its canonical stored snapshot's
  sha256.
  Regenerate the golden only with strands-agents 1.42.0. Never edit it by hand. Re-pin the constant with it.
  Keep the module's `without_ids` separate from the product helper so the helper never checks itself.
  This is component evidence. It establishes nothing about a deployed runtime, a native job or a worker loss.

- `tests/test_changelog_shape.py` reads root `CHANGELOG.md`, root `AGENTS.md` and `CONTRIBUTING.md`.
  It requires the linked introduction that names `VERSION` and `## [Unreleased]` first.
  It requires release headings of the form `## [N.NN] - YYYY-MM-DD` with strictly decreasing versions
  and dates that never increase. Versions compare as the pair of `N` and `NN` read as integers. Each release
  follows the one below it by one counter step or starts a new release line at `00`.
  It requires allowed groups, bullet format and content rules.
  Both rule files must require one Unreleased line for every user- or maintainer-visible change in the same
  change and the `VERSION` step at release. Both must state when `N` rises and that a publication without
  a visible change keeps its `VERSION` and gets no new heading. The contributor guide must name `uv.lock`
  in the release change. Drifted heading lists fed to its helper must be reported.
  It checks shape and rule text. It does not check whether bullets are true or every change has a bullet.
  [Added 2026-11-17] It holds the note on release numbers assigned afterwards as `ASSIGNED_NUMBERS_NOTE`.
  It requires the introduction to keep that note exactly once, word for word, as one paragraph.
  It feeds seven changed inputs to `assigned_numbers_note_problems` and requires each to be reported.

- `tests/test_version_single_source.py` reads root `VERSION`, `pyproject.toml`, `uv.lock`, `CHANGELOG.md`
  and the Python files under `src/amlas`. It requires `VERSION` to hold `N.NN` and one newline.
  It requires the `pyproject.toml` project version to equal the mapped `N.<NN as an integer>.0`,
  with no second version source. It requires the `uv.lock` record of the project's own package to equal it.
  It requires the newest release heading of `CHANGELOG.md` to name the version in `VERSION`.
  It requires no version literal in the application source. It requires `amlas --version`,
  `amlas info version --json` and `amlas.__version__` to report the mapped version from a checkout.
  Its helpers are also fed drifted inputs and must report each one. It does not build or install the package.
  Wheel and source-distribution metadata remain a release build check.

- `tests/test_image_version_file.py` reads the root `Dockerfile*` recipes, root `VERSION` and
  `src/amlas/_version.py`. It requires the recipes whose final stage copies the whole `amlas` package to be
  the listed ones. It requires each of them to copy root `VERSION` from the build context to the parent of
  the copied `src` directory, and no ignore file to leave `VERSION` out of the build context.
  It lays the package and `VERSION` out as each recipe copies them and requires the real reader to return
  the mapped version. Its helpers are also fed changed recipes, including each real recipe without its
  `VERSION` line, and must report each one. It builds no image. The version a built image prints remains
  an image build check.
- `test_qwen_cpu_case_file_provisioning.py` (added 2026-11-15): Contains 30 offline tests with no skip. Helper tests use stand-in files and parameterized pins through a local file address or a loopback HTTP server. They check read-only writes, hash and size failures, missing sources, HTTP 404, disallowed redirects, temporary 503 retries, the one-byte read bound, and default HTTPS requests. They check verified-file reuse without fetching, differing named-file refusal without requests or writes, and preservation of existing destinations. Parsed constants must equal SOURCE_REPO, SOURCE_REVISION, SOURCE_FILENAME, SOURCE_SHA256, SOURCE_BYTES. The address must be the public resolve address of that revision. Incomplete, duplicate, computed or misshapen constants refuse. Script tests copy the real script and helper into a temporary project with stand-in constants, metadata and `uv`. They check four exact `environment.sh` lines, export placement, sourced values, and failure without `environment.sh`. One test requires `AMLAS_TEST_HUMANEVAL_PARQUET` to name the provisioned file and checks product pins; absence fails. These tests make no public-source request. The real fetch is proven only by a script run with network.

## Work Guidance

`test_qwen_provider_boundary_startup.py` fixes the before-drop inventory and
post-drop sealed handoff order for Qwen training without adopting the downloader
filesystem policy. These source checks do not prove native GPU admission.

Positive fixed-trainer manifest fixtures explicitly model the post-clear child
environment. Provider-presence rejection fixtures retain their independent hints;
CI environment variables must never require weakening the production guard.

`test_model_provider_inventory.py` exercises real readonly sealed memfd objects,
closed bounded JSON, malformed/current-denial distinctions and model-only O_PATH
walks. Existing legacy provider/resource refusal controls remain authoritative;
these fixtures do not prove fixed native startup or filesystem enforcement.
`test_model_boundary_startup.py` checks the fixed capture/drop/handoff/policy/exec
order and model-only FD4/output-temp contracts. Actual native enforcement and
postexec descriptor/kernel behavior remain separate controls.

`test_open_swe_preparation_worker.py` exercises actual synthetic Parquet through
the fixed worker, reconciling original rows, structural normalization and reducer
drops while retaining qualified derived lineage and excluding gold columns.
Allocation refusals precede batch decoding; explicit parser options and whole-row
ambiguity refusal supplement the unchanged strict JSONL controls.
`test_preparation_image_contract.py` checks the closed locked parser/codec source
closure and fixed local installer checksum/size/version/order/removal contract.
Actual AWS image red/green assertions require both installer distribution and
module-path absence, unchanged locked PyArrow and fixed worker/launcher bytes.
They establish no base-layer byte erasure or native managed-worker admission.
Native image execution and exact-version output verification remain
separate gates.

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

- `test_direct_acquisition_context.py` calls actual normal execute_tool_use and acquisition/poll functions with strict neutral adapter records. Preserve original store/ref identity, fresh copied policy, safe ASK/confirmation/plan-only behavior, caller-context refusal, foreign-ref isolation, unchanged public schemas and inline classification. Source fixtures perform no native/AWS/model effect.

- `test_processing_terminal_callback_rule.py` inspects actual synthesized default/two isolated/24-character boundary deployment templates for both stacks, exact scoped Processing terminal pattern, unchanged Training pattern/shared callback target, rule-ARN Lambda permission whose FunctionName equals the target ARN, and unchanged callback configuration. Existing campaign callback tests preserve default/missing-qualification denial and closed untrusted wake semantics. Synthesis/fixture green is source evidence only; no deployment, native routing or autonomy acceptance follows.

Normal TASK capture budget coverage uses actual public SDK snapshots and the existing qualified normal-context fixture: separately valid causal-call and SDK-history DATA must remain capturable without storing a second copy of identical messages. Preserve exact snapshot state/message restoration and the existing 160-KiB cap; individually oversized DATA still refuses. This test-only size reproducer is not a measurement of the private live run and does not authorize any effect or relax snapshot, ownership, native-reference, CAS, or duplicate-effect guards. Compact/legacy compatibility controls keep exact versioned key inventories, reject boolean/string/unknown schemas, preserve authenticated legacy wire identity and JSON-typed message order, return independent history copies, require the strict public SDK inventory, and refuse an oversized compact whole wire before projection. The legacy prepare/admission integration uses the existing inert original lifecycle and real versioned publication path, pins schema1 only at initial preparation, then verifies admitted bytes/schema and the prepared descriptor remain identical through the real authenticated reader. Legal legacy task-only records also republish through the qualified normal writer with unchanged original bytes, authenticated versioned refs, current-context/CAS checks and one native effect; the test changes only the historical record shape through the existing CAS writer.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `fixtures/AGENTS.md` | Bounded public regression captures |

- `test_qwen_sft_native_preparation_handoff.py` runs the real Processing/CommitPack artifact verifier with strict native/S3 fakes before private SFT transfer. It reproduces missing producer identity in raw success (public observe had hidden it), separately covers exact typed-ref compatibility while measuring current dict refs, and retains malformed fields/roles/producer paths/header/source/session/subclass negatives. The SFT binding fake follows the real preparation output layout. No AWS/model bytes or native training job executes.

- test_qwen_sft_native_observation.py retains exact measured service defaults across running/completed/failed states, verifies no response/request mutation and refuses malformed defaults or unknown fields. Fixture green does not prove a new native submission or trained model.

- `test_qwen_sft_socket_diagnostics.py` injects native-shaped failures at each fixed
  socket stage, validates bounded closed output and exact original-errno22 trigger,
  preserves refusal for reachable/resource/unknown states, and proves one close
  attempt per created socket even on unexpected failures. A malformed one-argument
  OSError is explicitly unknown; these hermetic controls grant no native isolation.

- `test_qwen_sft_checkpoint_diagnostics.py` reproduces source-managed TMPDIR/output
  coupling through real retained directory FD and zero framework calls, then
  verifies separate scratch plus all closed save substages without error content.
  Synthetic header oracles stay separate from actual tiny-PEFT positives.

- `test_qwen_sft_receipt_summary.py` measures actual33060-row digest shape against
  unchanged262144 cap, complete ordered/domain/count binding and actual managed
  post-save failure hooks. `test_qwen_sft_receipt_consumer.py` retains rehashed
  schema/private-pin/count refusals and explicit metadata-only digest coverage.
  Native-shape CP fixtures use current closed summary/COMMITTED grammar; all old
  acceptance/refusal/reload assertions remain. No native/training admission claim.

- `test_managed_coding_case_envelope.py`, `test_coding_case_tools.py`, `test_coding_case_native.py` and `test_coding_case_worker.py` exercise closed original producer/source envelopes, untouched pure replay, actual SDK hidden-context/approval denial, private role isolation, opaque original-handle reuse, native SDK request shapes, exact S3 versions/SHA and actual inert Parquet/retained-FD mutation controls. Fixtures are synthetic; no gold corpus, AWS, trained/graded model or native admission follows from green tests. The native case suite also requires null/string/list SDK Error metadata to retain one original ambiguous opaque/native identity across repeated store submissions, with exactly one Create.

- `test_managed_generation_product.py` exercises the actual artifact consumer with strict neutral native/S3 fixtures for both native no-slash and legacy slash roots, preserving raw observation, three-file/version/hash/size checks and one original Create. Malformed/foreign/empty-relative roots refuse before S3 inventory; no model/candidate/native execution occurs.
- `test_qwen_sft_real_output_layout.py` feeds the DescribeTrainingJob shape real SageMaker returns for an uncompressed job (`<S3OutputPath>/<TrainingJobName>/output/model`, no trailing separator, seven recorded jobs with bucket, session segment and job names made up) through the actual `SageMakerQwenSftBackend.artifacts` and reload paths, and pins the simulated-provider layout, the one-separator canonicalisation and 29 refusals (other bucket, root, job or depth, archive name, dot/empty segments, query, fragment, backslash, non-strings) before any object read. The generation product test pins the same real layout for generation. No native job runs.
- `test_sft_instance_preferences.py` checks that the fixed training, reload and generation requests validate against the installed SDK model and list their pinned preferences, that a described selection of a listed type reads back as the request (and both fields absent only while InProgress Pending), that a selection outside the request refuses, that any list other than the three pinned ones refuses, and that historical requests keep their exact fixed type. With botocore 1.43.57 its SDK tests fail by design.
- [Added 2026-11-10, lists and quota rule changed 2026-11-10] `test_capacity_preference_lists.py` pins the chosen lists and their order (training ml.g6e.16xlarge, 12xlarge, 8xlarge, 4xlarge; the new reload contract ml.g6e.12xlarge alone as a one-entry list; generation ml.g6e.16xlarge, 4xlarge, 8xlarge; reload and generation share no type; reload and generation values changed 2026-11-11, see the next note), the new contract name, unchanged historical acceptance (contract-free reloads 12xlarge, `reload_single_gpu_v1` 4xlarge, the fixed 12xlarge training snapshot), refusal of 19 reordered, extended, shortened or earlier lists in `fixed_resource_config`, of those and the other two pinned lists in a training snapshot and in a reload snapshot, and `SelectedInstanceType`/`SelectedInstanceCount` in the SFT status detail and observation (none for fixed-type or UNKNOWN). `Account` applies the quota rule read from CloudTrail on 2026-11-10: a create is refused naming the first requested type that a job holds, a fixed-type job holds its type from the create on, a list job without a selection holds nothing; a selected type of a list job counts as held, which is not measured. Over it: reload then generation and generation then reload are both created whichever listed type each holds; the tuned generation is refused while base holds ml.g6e.16xlarge (the first listed type), the note names only that type and its holder, and the same call is a real create after base ends; two list jobs accepted before either selects are both created; a job outside the run on ml.g6e.8xlarge refuses a generation although 16xlarge and 4xlarge are free; a waiting list job of this run is reported but never named as the holder; a job on any of the four training types rejects the fine-tune once and it is never created again, also after that job ends; a job on 12xlarge does the same to the reload, and a job on 16xlarge, 4xlarge, 8xlarge or 2xlarge does not refuse the reload. No native job runs.
- [Added 2026-11-11, reload and generation lists changed] The list tests pin the lists of 2026-11-11: reload ml.g6e.12xlarge alone, generation ml.g6e.16xlarge, 4xlarge, 8xlarge in this order. `test_capacity_preference_lists.py`: `OTHER_LISTS` keeps every list it refused before by its value and adds the two lists of 2026-11-10 and four variants of the new ones (19 lists); `test_one_type_reload_is_sent_as_a_one_entry_preference_list` pins the reload request shape (one-entry `InstancePreferences`, no `InstanceType`, valid for CreateTrainingJob in the installed SDK) and the SDK model's `InstancePreferenceList` bounds 1 to 5, so an SDK that no longer allows one entry turns the test red; `test_one_entry_reload_is_created_recorded_restored_and_read_back` drives the ordinary reload through the durable store, restores its record without a create, and reads `SelectedInstanceType=ml.g6e.12xlarge` and `SelectedInstanceCount=1` in `status()` (detail and raw description) and in `artifacts()` for the created and the restored handle while the job is InProgress; `test_every_interleaving_of_reload_and_base_then_tuned_is_created_or_submitted_again` (24 cases: either order, each generation type as the base job's type, base ended or running, reload ended or running) requires over `Account` and `QuotaNative` that the reload and the base generation are each created once, and that the tuned generation is created at once after base ended or refused naming base's type and created by the same call after base ends. The base job of the strict fakes now holds ml.g6e.16xlarge (the first listed type), so the expected type changed from 4xlarge to 16xlarge in `test_capacity_rejection_read.py`, `test_generation_capacity_rejection_retry.py` (where a limit message naming ml.g6e.16xlarge is now kept as a listed type; still 18 messages), `test_unmapped_native_state_names_no_selection.py` and `test_managed_generation_product.py`. `test_qwen_sft_reload_resources.py::test_reordered_preference_snapshot_cannot_restore` keeps its id: a one-entry list has no second order, so it now refuses seven changed lists (the earlier two-type list in both orders, the pinned type twice, a second type before and after it, another single type, an empty list). `test_sft_instance_preferences.py` adds the readback of every generation type and of the one-entry reload list (12xlarge reads back, four other selected types refuse), replaces the refused selection 16xlarge for generation by 24xlarge and adds four refused lists to the bounds test. Test ids replaced or added here are new; no earlier test id is removed. No native job runs.
- [Added 2026-11-10] `test_qwen_sft_backend.described_resource_config` is the one place the strict fakes describe a preference-list ResourceConfig: SelectedInstanceType is the first listed type and SelectedInstanceCount the InstanceCount, as in `fixtures/sagemaker_instance_preference_readback_real.json`. `Native`, `normal_sft_cold_fixture.py` and the describe stand-ins of `test_runtime_sft_reload_originals.py`, `test_runtime_generation_originals.py`, `test_input_admission_initial_producer_ack.py` and `test_other_producer_immutable_rejection.py` use it.
- `test_paired_coding_real_path.py` rebuilds the shapes of the real pair job of 2026-11-10 over the strict native fakes of `test_paired_coding_backend.py`: the deployment's own tags beside the three ownership tags (accepted in any live order), ten foreign or drifted tag sets that must refuse with no S3 read, a manifest stream that must be closed before any record read, the real volume of 12,786 records at the measured 0.0338 s per S3 call (2,160 s, more than the old fixed 600 s), a far slower pass that must still stop at its deadline, and a manifest replaced during the record pass. No native job runs.
- `test_paired_resolved_spec_binding.py` runs the pair's own `validate_pair_artifacts` on actual managed case bytes and two published generation arms: both sides bound to the resolved spec pass; generation bound to the unresolved spec refuses at the generation-binding guard, an unresolved pair spec at the case-manifest guard, and non-canonical resolved bytes at the canonical-bytes guard. Over the normal stores, generation refuses the unresolved reference and binds the resolved digest, and `submit_coding_pair` hands the pair the same resolved bytes and digest. It also passes on the source before it was added: the normal path has no unresolved-spec defect; the one seen on 2026-11-10 came from a direct driver outside the normal path.
- `test_sft_instance_preference_readback.py` (added 2026-11-10) drives the SFT `status()` with the recorded ResourceConfig answers of `fixtures/sagemaker_instance_preference_readback_real.json`: the three real preference-list jobs and the fixed-type job read back their native state with the selected values kept in `JobStatus.raw`; both Selected fields absent read RUNNING only in InProgress Starting/Pending; one field alone, an unlisted type, another count, malformed values, reordered, dropped or added preferences, another volume or count, an InstanceType beside the list and a future field refuse; a fixed-type request with any Selected or preference field refuses. No native job runs.
- [Added 2026-11-10] `test_sft_instance_preference_readback.py` also covers a preference-list job that ended or is stopping before SageMaker selected an instance: with both Selected fields absent, Stopped, Failed and Stopping read STOPPED, FAILED and RUNNING only when `SecondaryStatusTransitions` holds Starting/Pending entries, an optional Stopping entry and the matching terminal entry that equals `SecondaryStatus` (the real shape of a job stopped while Pending). Absent fields after Downloading, Training, Uploading or an unlisted status, a Completed job, a mismatched or repeated terminal entry, malformed transitions and one field alone refuse, and the absent-field rows without transitions still refuse. `normal_sft_cold_fixture.NativeJobs.unselected_training` describes every training job it creates that way (`waiting_description`; carried in the transport as `unselected`), and `test_normal_failed_sft_after_stop_report.py::test_replacement_reports_a_preference_job_that_ended_before_selection` proves the final report for it. `test_normal_durable_registered_reload.py` re-pins the fixture bytes for these `unselected` jobs.
- [Added 2026-11-11] `test_runtime_training_tag_read.py` replays the SageMaker calls the runtime execution
  role made in the first real normal-path run (`fixtures/sagemaker_training_tag_read_real.json`) against the policy
  the deploy script generates, with a small Allow matcher (action and resource patterns; these statements carry no Deny
  and no Condition). Every recorded call must be allowed, the tag reads are exactly the two ListTags-only statements
  (owned processing jobs, owned training jobs) and no other action on a training job is added.
  An inventory of the SageMaker job operations of the nine normal-path adapters is compared with the adapter sources
  (every client operation named in a module, by its botocore name) and each inventory row, with AddTags for a Create,
  must be allowed for a job of that family: a new native call needs an inventory row and a grant in the same change.
  `test_infra_deployment_namespace.py::test_runtime_reads_ownership_tags_of_owned_training_jobs` asserts the same
  statement in the RuntimeStack template. The two earlier tag-read tests keep their processing statement and now list
  the training statement after it.
- [Added 2026-11-11] `test_normal_fit_completion_restore.py` exercises the four restore sites that decide
  "the original SFT is complete" (`app._resume_normal_prompt_body`, `ChatEngine.assemble_normal_completion`,
  `_restore_runtime_research_profile`, `NormalRunContext.qualify_original_completion`) and their shared rule with
  declared observations: a trained `fit_v1` original is accepted at each site; a trained `full_v1` original
  without `full_training_complete`, an unknown recipe, a missing recipe and an untrained fit are refused at each
  site with the unchanged refusal and before any admission; a fit observation of another alias, job name or job
  ARN is refused, and `_verified_generation_checkpoint` still refuses a fit original before any observation.
  It reuses `_routing_case` of
  `test_normal_completion_route.py` (which takes an optional declared observation and has a `refused` kind) and
  the seams of `test_adopted_parent_restore_sites.py`. `normal_sft_cold_fixture.py` has the variant `fit-only`
  (`FitOnlyModel`: one `fit_v1` submission, one `get_qwen_sft`, then `FIT_FINAL_TEXT`; `post_sft_stage` may return
  `REPORT_NOW`); `original_precommit` requires the accepted recipe to equal the script's `original_recipe_id`
  (`full_v1` for every other script) and `cloud_data` installs the outputs of that recipe.
  `test_normal_sft_cold_continuation.py::test_normal_cold_completed_fit_sft_continues_to_the_final_report` runs
  the pilot through the real restore in a new process: restoring, run_started, run_completed, no native create,
  the report available to a second replacement. These are source fixtures with a scripted model; they prove no
  native job, IAM, AgentCore session life or model behaviour. `test_normal_durable_registered_reload.py` is
  re-pinned for the changed bytes of `normal_sft_cold_fixture.py` and `normal_run_checkpoint.py`.
- [Added 2026-11-11] `test_sft_real_in_progress_wait.py` reads the real DescribeTrainingJob answers of one
  preference-list job (Pending, Downloading, Training, Stopping, Stopping/Uploading, Stopped) and the stored request
  of the real fine-tune through `SageMakerQwenSftBackend` and the precommit wait (the inert rig of
  `test_precommit_unknown_wall_clock.py`, with a store that answers the adapter's own observation). An in-progress
  job whose tags can be read is RUNNING in every state and never spends the unresolved budget, also when it waits and
  trains for longer than the cap. A refused tag read, foreign or missing ownership tags and a changed stored request
  are UNKNOWN in every state; with every tag read refused the wait closes at the cap with the diagnostic of the real
  held report. Starting and InProgress/Uploading answers were not caught by the 10 s read and are not in the fixture.
- [Added 2026-11-10] `test_reload_single_gpu_contract_record.py` writes a schema-4 reload record the way the source before the preference lists did (the ordinary reload path with the contract name `reload_single_gpu_v1`, exactly one ml.g6e.4xlarge) and proves on the current source that it decodes, validates, restores as verified in a fresh adapter with no write or create, observes SUCCEEDED with no selected instance, accepts none of seven other requests (preference lists, other types, counts or volumes) and cannot be relabelled as `reload_instance_preferences_v1`. [Added 2026-11-10] The written bytes and operation digests, without and with a reload compatibility selection, are pinned to those the source before the preference lists wrote over the same fixtures (measured by running that source), and the restore runs for both.
- `test_runtime_profile_passthrough.py` runs the real `deploy_agentcore_runtime.sh` over the fake AWS CLI of `test_runtime_deployment_advisory.py`: `AMLAS_RUNTIME_RESEARCH_PROFILE` and `AMLAS_GUARDRAIL_ENABLED` reach the runtime environment when set, stay absent when unset, and an unknown profile or switch value (including empty and a different case) fails before any runtime write.
- [Added 2026-11-10] `test_runtime_env_budget.py` measures the runtime environment as AgentCore does: the fresh-deployment selection (52 keys, value lengths of a measured deployment, documentation account) is over 4,000 bytes plain and is sent within 3,600 bytes by the control client and by the real deploy script; oversized selections and a missing image switch refuse before any AgentCore call, naming the bytes, the limit and the three largest keys. It proves the pack lossless (602 deterministic random sub-selections of the fresh-deployment and default-deploy selections), the import-time expansion in a fresh interpreter, 16 malformed-pack refusals that leave the environment unchanged, the frozen approved-action profile and identical approval and generation pin decisions after expansion, and re-runs eight existing control suites through `runtime_env_pack_controls_plugin.py`, which feeds every startup reader the expanded environment and asserts it equals the plain one.
- [Added 2026-11-10] `test_preparation_release_config.py` also covers the triple-only selection (explicit ENV, SSM and the deploy overlay), the stale-exclusions guard (present or unreadable parameters refuse), partial exclusions and exclusions without the triple, and `publish_preparation_selection` publishing exactly the selected values.

- test_qwen_coding_generation_managed covers closed original-artifact binding, separate checkpoint/image sources, fixed startup/shared-gate ordering and generation-only bounded FD-relative publication with COMMITTED last. Neutral mocked boundary fixtures establish source behavior, not native GPU or execution admission.

- `test_model_processing_plan.py` uses only neutral strict SDK fixtures to verify
  pure preparation, exact original job/request reconstruction, caller-plan
  immutability, current configuration drift and qualification refusal before
  Create. It establishes no durable journal, native admission or acquired model.

- `test_campaign_consumed_field_types.py` uses neutral equality objects to require
  exact native/protection selector primitives before witness/config construction.
  These are source validation controls and perform no native/security probe.


Neutral model-journal proofs cover explicit protected PRESENT/ABSENT/UNKNOWN,
first-origin denial before writes, read-only duplicates, unresolved publication
without replay, and deadline expiry after callbacks at marker/effect/outcome.
The fake protected port is explicitly source-test evidence, never a native
protection or action grant. Preserve exact adapter comparison and consumed-marker
proofs; no native/model/tensor effects belong in these tests.

Model journal correlation fixtures must keep private and transport IDs distinct.
Prove actual adapter plan equality, private-store acceptance and same original
opaque-ref cold restoration; the transport-correlated store must refuse.

Keep neutral review regressions for origin revocation during coordinator lease
read, inherited PID denial before a forbidden lock, bool/float numeric intent
substitution, and accepted/error refusal before any additional UpdateItem.

- `test_model_adapter_qualified_io.py` uses neutral strict SDK providers to verify
  per-leg qualification, mid-read revocation before later calls, scope reset/PID/
  copied-context lifetime, post-Get stream cleanup and post-Create uncertainty.
  No native model, credential, socket or containment workload executes.


Model campaign bridge tests use real sealed RuntimeA and actual fixed adapter
with strict neutral SDK ports. They require original validated coding spec body,
protected publication/hint/marker before one Create, unknown response retention,
fresh effect approval and SDK revocation, same private-store cold opaque ref,
concurrent one-owner behavior and scoped read-only artifacts. DATA inspection and
reservation acknowledgements never grant Create; fixtures are not native origin
qualification or model/readiness/full-campaign acceptance.

- `test_normal_model_campaign_flow.py` uses actual sealed private bridge/runtime,
  strict neutral SDK providers and the normal API/installed SDK tool path. It
  verifies one original Create, same ref/job cold reuse, qualified observation,
  missing services/default denial and foreign active factory refusal. Synthetic
  scripted models/planning ports establish no live research/native authentication
  or end-to-end trained/quality acceptance.

Binding exposure regressions retain exact oneCreate while refusing postnative
revocation, report unresolved outcome writes as known historical attempt data,
recheck the final runtime/model boundary and deny changed original configuration.
Known preCreate refusal and opaque adapter exception remain distinct; unqualified
errors expose no IDs and no native/action permission is inferred.
Recovery RUN startup tests use fixed guard/output/provider fakes plus the actual
no-follow/raw hash reader and pure reload binding validator. An independently
pinned bounded native RUN metadata fixture can exercise the exact historical
consumer source without storing native artifacts in product source. Portable
synthetic metadata uses the same public catalog. Preserve v1 operation positives
and v2 train/generate/source/recipe/contract/extra-field denials. No socket probe,
model/tensor load or native GPU effect occurs in this fixture.

Canonical planning-body fixtures use actual SpecModel projection and the source
bridge with strict neutral SDK ports. Preserve canonical-body one-Create/cold-ref
reuse, legacy frontmatter default and unknown/noncanonical/duplicate/nonfinite/
wrong-schema/cross-format refusals before any protected/native write. Source-only.

Sdist root-link regressions require paired root DOX exclusions and inspect actual
archive link closure when an independently built artifact is supplied. Preserve
legitimate nested Rust test sources; missing artifact is an explicit skip, never
an artifact-inspection PASS.

- `test_runtime_warm_model_continuation.py`, `test_warm_model_closure.py` and
  `test_model_warm_qualified_io.py` use actual installed SDK/normal engine/owned stores
  over strict neutral Processing/S3 ports. Require one original Create/ref/history,
  exact-version/hash verified nine-file completion (raw/FULL_OBJECT), same-Agent append-only completion-event
  causality and fresh policy/qualification, no public selector or replay; preserve
  unknown/failed/intervention/revocation/foreign-record/two-owner negatives. Nested
  guards compose outer+inner, innerNone retains parent, mid-I/O revocation blocks
  next reads. Pending/closed owners do not revive on new turns; finite consumed
  bound, callback finally restoration and actual two-tool capture remain required.
  These are isolated manual one-step controls, not connected API/app background
  continuation, cold durability, native admission or full coding-goal proof.

- The actual resumed SDK CancelledError control must require propagated cancellation
  with pytest.raises, plus callback restoration and permanent original closure;
  returning refused cannot substitute for process-control propagation.

- test_runtime_connected_warm_model.py drives actual app worker/installed SDK
  busy tasks and real API/engine/registered model tool over neutral strict ports.
  It requires one original Create/ref/Agent/seven stores, API wait through exact
  nine-version/hash verification, same-Agent append-only completion-event history causality and task
  completion only on actual worker finish. Public pending DATA cannot select work.
  Failed/refused/guardrail/unknown/current-policy and deadline/disconnect controls
  retain honest closed DATA and busy ownership without replay. Controlled15s sleep
  is a neutral test seam; no public/env cadence or authority is added. These controls
  prove no native admission, cold durability or full ML lifecycle/quality.

- The connected two-invocation cancellation case uses a freshly scripted process-control
  cancellation after the genuine registered acquisition ACK, actual SDK task cleanup,
  and a distinct fresh ordinary prompt/Model/Agent. Require one original Create, no
  Stop/replay, clean second history and passive report retention only after worker
  cleanup. The second prompt is a new logical run, not retry or cold recovery of the
  cancelled run; fixture ports establish no native admission or lifecycle success.

- The manual one-step original_work fixture drives actual ChatEngine.process_message
  with its original typed caller policy and actual SDK/store capture. Its private-step
  assertions and earlier actual API red receipts stay intact. Connected API acceptance
  belongs only to the actual app/API suite; no production cadence/guard is bypassed.

- The normal campaign API/SDK submission-correlation regression keeps actual
  api.run_prompt and activated original campaign/services; its native fake reports
  terminal Failed after acceptance and must return closed terminal_failed DATA.
  It retains oneCreate/original-store correlation, never simulates verified-ready
  artifacts. Exact completed bytes remain owned by the connected API suite.

- DATA/CASE/PREP observation-only qualification regressions live in
  test_data_case_prep_observation_io.py. They use existing strict neutral native
  fixtures, original fake-only accepted handles, harmless callback/barrier data,
  copied-context and thread lifetime controls. Preserve zero additional Create or
  staging, pre/post read fences, returned stream closure, final exposure refusal
  and lazy lineage scope/yield expiry. Observation guards alone establish no
  native admission or multi-producer continuation.
  Original AWS50-case missing-method red is retained; run all owning verification
  inside AWS only. These fixtures establish no native qualification.

- Returned-body cleanup regressions exercise actual DATA payload, inherited CASE/
  PREP raw and lazy lineage readers before timer construction/start, plus post-Get
  cancellation whose close fails. Preserve original process-control exception and
  close every owned stream before first read; cleanup must never turn cancellation
  into UNKNOWN or failed artifact DATA. Synthetic timer/body fixtures run AWS-only.

- Observation cleanup pending-failure state belongs to that lexical operation,
  never an ambient caller except suite. A normal operation whose close fails must
  refuse output even inside a caller's handled unrelated exception context.

- `test_model_acquisition_native_diagnostics.py` drives the actual installed SDK,
  registered model tools and same private original store over strict neutral SDK
  providers. It freezes finite submission/observation error class, phase and code
  projections with bounded provider request IDs/status, no raw message/request/
  headers and no unknown-attempt Create replay. Test greens grant no native cause,
  acquisition/admission, UNKNOWN warm ownership or lifecycle recovery authority.

- test_runtime_ambiguous_model_capture.py uses actual installed
  SDK/app/API tools and the strict native fake recording ONE original Create before
  its reply is lost. No positive acceptance/ambiguity flags are injected. Preserve
  original store/record/Agent/graph/PID/pin and same toolResult causality through
  verified completion or UNKNOWN hold, policy/guardrail closure and zero replay.
  Pure source-prepared/no-attempt and public-only claims are negative DATA controls.
  Original six-case AWS red (four expected failures/two negative passes) remains
  retained. Fixture passes grant no native admission, cold recovery or capacity.

- Ambiguous MODEL review controls require source Create lifecycle/pin unchanged by
  Describe, normalTrue/False preserved, invalid/missing/None/nonbool/mixed private
  pairs and publicTrue/privateFalseFalse ignored. Flag edits occur only in negative
  corruption fixtures. Actual app/API worker cancellation of a genuine lost-reply
  original closes and drains its blocked observer before SDK task completion and
  propagates CancelledError without Create replay or model resume. No runtime or
  provider code is replaced; neutral owned-task/barrier fixtures run AWS-only.

- `test_warm_model_refusal_diagnostics.py` uses actual SDK/original-owner and normal
  app/API fixtures over strict synthetic native providers. Planned diagnostic reds
  require source raw-hash failure, policy/pin closure, bounded UNKNOWN class, exact
  repeated latch/no extra I/O and final client projection, with nominal response
  shape and strict cancellation propagation retained. No fixture grants native
  cause/readiness, original-store adoption, model execution or lifecycle completion.

- `test_deploy_flow_truth.py` drives actual deployment flow/tools and SageMaker
  adapters over installed-SDK input/output-validated neutral recorders. Require
  one original package consumed without a public model-name override; native
  Creating/Failed/unknown cannot claim InService/deployed, invoke smoke or cut a
  release; uncertain Describe never replays endpoint creation. A failed actual
  monitor setup has no ARN and cannot claim wired. Keep successful endpoint and
  independently explicit-baseline/role/output monitor identity controls separate
  from the flow's incomplete monitor configuration. No fixture grants native
  serving/monitor admission, Qwen/GPU/quality, AWS authorization or lifecycle proof.
- Deployment truth controls also construct the actual managed adapter from
  canonical ENV/alias selection and an explicit override, preserving shared trim
  and placeholder semantics. Require one model with the selected execution role,
  the same original package/endpoint hosting identity and zero startup/native/SSM
  calls. Neutral SDK-shaped flow success grants no actual role/hosting admission.
- Private package controls also refuse replaced model/package/PID, expired copied
  scope, changed backend selection and public package DTO reuse before any endpoint
  creation. Keep unknown accepted endpoint identity, ASK denial, serverless and
  actual harmless local-predictor compatibility separate from native/Qwen admission.
- Neutral endpoint-observer logging retains only a closed failure class, with no
  exception message or endpoint coordinates. A fresh thread/event barrier expires
  the original package during config creation and must prevent endpoint creation;
  source fixture fences prove no native permission revocation or Qwen admission.
- Review controls freeze configuration before actual model creation, bind native
  schedule/endpoint ARN resource names, preserve correct no-ARN Create/readback,
  verify frozen resolver fields, and exercise actual description/each autoscaling
  expiry boundary. These neutral controls establish no AWS/native admission.

- Missing Create ARN coverage is an explicitly legacy transport projection after
  validated native creation, never a native-shaped success: the installed SDK
  requires EndpointArn. The corrected native Describe remains separately strict.

- `test_runtime_original_model_diagnostic_events.py` uses installed SDK registered
  model tools, the actual normal app/API worker and strict neutral providers.
  Planned reds freeze finite known provider-code labels and one closed original
  Create diagnostic event while UNKNOWN retains the same record/graph/pin and busy
  task. Public-only fields and currently refused originals cannot publish; nominal
  accepted flow has no failure event. No error message/request/headers or public
  reference is emitted, no retry/Create replay follows, and all existing lifecycle
  and quality envelope flags remain false. These test-only controls are not native
  error-cause, capacity, rejection, admission or full-lifecycle evidence.

- `test_runtime_original_model_diagnostic_slot.py` retains a genuine installed-SDK
  original with its second neutral observation held. Private diagnostic-only
  scope/context expiry, current postqualification policy/pin loss, competing
  consumers, worker-finish invalidation and optional/process-control failures
  cannot grant native I/O or alter the original task. Publication consumer calls
  no producer Describe/Get/Create; callbacks/scopes remain private source objects.
  These harmless source controls run AWS-only and establish no native cause.

- The original diagnostic publication regression forces genuine worker finally
  between a legitimate consumer read and outward yield. Recheck existing stream
  completion/exposure gates; diagnostic DATA cannot assert an active worker after
  actual task completion. Natural policy refusal closes the owned fixture; no
  cancellation, Create replay, public original adoption or native cause follows.

- `test_runtime_worker_failure_diagnostics.py` closes the actual normal SSE
  consumer before synchronized work-body or cleanup/report failures. Preserve
  exactly one rendered/structured closed warning, exact type fallback, cancellation
  cleanup, no private error text, ordinary logger-refusal cleanup and exact-object
  logger-origin process-control propagation. Source-location fixtures require
  the deepest fixed mapped label through unknown private frames, closed unknown/0
  for absent or unmapped tracebacks and no error text, raw paths or traceback.
  These neutral AWS-only fixtures retain original task/report/status behavior and prove no native cause,
  worker loss, recovery, SFT handoff or lifecycle completion.

- The inert `connected_work` fixture privately copies Processing pacing state and
  removes only its synthetic account/region entry before creating its own workers.
  Other scopes retain object identity; monkeypatch restores the previous mapping.
  Keep the actual1.25s completion-based interval, PID/lock, real clocks and all
  existing5s readiness assertions unchanged. `test_connected_fixture_processing_scope.py`
  proves prior-map isolation/restoration and real intra-scope pacing without AWS
  effects; this fixture isolation establishes no native throughput or cause.

### Digest-backed cold research completion receipts

The cold research completion regression must consume the original schema 3
receipt without requiring an inline SDK snapshot. Retain the exact completion
snapshot and all current actor, original-job, token, approved-history, policy and
zero-new-effects assertions. A changed snapshot digest must refuse before research
restoration or admission; legacy receipt refusal and strict schema 1/2 TASK reader
controls remain required.

### Durable registered RELOAD operation association

`test_normal_durable_registered_reload.py` ports the qualified normal ChatEngine
and registered SDK reload regression. Its test-only
`normal_reload_original_completion.py` is the hash-qualified controlled
original-completion fixture; it does not establish native ML or guardrail proof.
Require the actual normal journal, accepted original TRAIN, byte-identical primary
TRAIN row, and a separately accepted RELOAD reservation with its own request and
staged references. Preserve primary TRAIN/TASK readers, exact producer ancestry,
catalog/owner/policy/artifact fences, duplicate suppression and UNKNOWN refusal.
The fixture supplies controlled original publications and seeds no SDK history,
current witness or consumer operation row.

The same genuine reload case keeps the consumer's own operation digest and native
identity through duplicate reservation and adapter restoration. Foreign producer,
runtime, recipe or catalog input and stale runtime ownership must refuse before the
generic reserve delegate; tampered consumer restoration must refuse without new
Create. Every control preserves the complete operation rows and primary TRAIN
bytes. These controlled fixture checks establish no live/native ML claim.

- `test_research_profile_guardrail_disable.py` verifies explicit disabled-mode
  normal API research, exact finite scope, revocation and repeat refusal, ordinary
  registered-tool consumption and actual cold completion without refetch. Require
  zero ApplyGuardrail calls/receipts and empty strict native admission; enabled
  missing-checker behavior still refuses. Consumed completion keeps the absent
  guard transport (no close, factory, receipt or graph change), and the same
  refresh with checks enabled refuses without creating a client. Use the existing
  exact pinned readonly HumanEval fixture for cold setup; these are source
  integration tests only.

- `test_model_pack_byte_qualification_rate.py` (added 2026-11-07): a 64 MiB synthetic pack (neutral bytes,
  virtual clock, counting qualifiers) checks that the 1 MiB chunk readers of the model pack (`qwen_sft._raw` and
  `_output`, `model_weight_acquisition._payload_observed`, `dataset_acquisition._verify_version_payload`) and the
  adoption provider (`adopt`, `restore`, the store poll) run their full checks by time (at most
  `ceil(elapsed / 1.0 s)` plus the fixed object-level brackets, measured on a 1 MiB pack) and not by chunk, that a plain
  callback still runs for every bracket, and that a local refusal, a full refusal, a revoked policy, a changed configured
  scope, a closed or foreign-thread or inherited scope, a failing identity read and a replaced table each stop the pass
  within one chunk or one interval. Keep the size-independence test (1 MiB against 64 MiB) when changing a chunk reader.

- `test_normal_resumed_final_report.py` (added 2026-11-07): the cold fixture ends the original worker on its own
  (source refusal at PRECOMMIT, so it retains a held report under `report_checkpoint_ref`), exports the rows and objects
  to a fresh process, lets the replacement finish generation and pairing, and asserts that the replacement's terminal
  event is `response_received`, that `get_prompt_report` returns it (the original's report before, never after), that the
  original pointer is unchanged, that there are no extra jobs and that a second fresh replacement reads the same report.
  Light-fixture controls pin the resumed pointer rules (needs a started replacement, single assignment, identical replay,
  different second report refused on both pointers, damaged pointer fails closed and never revives the original's report,
  a row with the pointer but no started replacement is invalid) and the `_replacement_started` gate on synthetic rows.
  Stream controls pin that both pointers are bookkeeping for the immutable stream linkage (unit and the real TASK
  transition fixture) while the TASK descriptor and association stay bound. Keep the cold test on the real claim path.
- `test_normal_wake_resume_state.py` (changed 2026-11-07): the two report-publication fakes accept the keyword
  of the resumed call (`resumed=True`); the retry test asserts that every attempt of a resumed run carries it.
  Changed 2026-11-10: a healed transient attempt logs only the `reason=transient_retry attempt=N` form,
  exhausted attempts and a permanent error after a healed attempt end with the final refusal.
  Changed 2026-11-11: the final refusal is asserted as the exact text `normal_restore_unavailable reason=final
  restore_stage=<stage> exception_class=<category>` with arguments equal to the two extra fields; a healed attempt
  writes neither that form nor the bare text; a failed continuation writes exactly one final line that names
  `continuation`. The three earlier test names still say `bare_refusal`: they are kept so the test ids do not change.
- `test_normal_restore_diagnostics.py` (changed 2026-11-11): the route cases assert the exact final text,
  its format string and its two arguments beside the unchanged two extra fields. Direct calls of the two warning
  functions pin the final text for each of the five logged stages, the `run_prepare` fallback for any other stage
  (`report_retention` included), four whole `key=value` tokens for every stage and every reviewed category, and the
  retry form (text, format string, arguments, no extra field) as an unchanged control. Do not relax the exact-text
  assertions to a prefix match: readers outside this repository decide by the `reason` token.
  `test_normal_prepared_resume_route.py` and `test_normal_sft_cold_continuation.py` select the final refusal by that
  exact text built from the record's own two extra fields; the prepared route also requires that no other
  `normal_restore_unavailable` line was written.
- `test_normal_durable_registered_reload.py` (re-pinned 2026-11-10 for the marked run-record rechecks, before that
  2026-11-07): the byte pin of
  `normal_run_checkpoint.py` follows the resumed report pointer change, as the earlier edits of that module did.
- `test_normal_run_console_silent.py` (added 2026-11-14): the real builder under an active normal-run context, a
  scripted model that writes one sentence before each of two tool calls and then a final answer, and the real runtime
  entry point over strict neutral native ports. Tests named `red` require an empty stdout for the first-turn build,
  for both restore builders (after `load_snapshot` and through the restored invocation), for a prompt through
  `agent_invocation` (only the entry point's own startup line stays) and for a run lookup that fails (a context that
  fails its own check, or any other ordinary fault).
  Tests named `guard` pin what stays: a build with a thought trace or without a run keeps its handler and its console
  output, the guardrail capture wraps a callable handler and sees every model cycle, and the SDK contract the rule
  relies on (`null_callback_handler`, the handler read at each invocation). The red tests also require that the
  `normal_callback_activity` and `normal_conversation_size` lines equal those of the same run with the SDK printer
  and that the caller's response and the stored report are equal apart from per-run identifiers. Keep the baseline
  run with `PrintingCallbackHandler` in those comparisons: it proves that the text would have been printed. These
  source tests say nothing about the log of a deployed runtime.
- `test_normal_run_console_silent_replacement_process.py` (added 2026-11-14): the cold fixture's original worker
  in the test process, then two replacement workers in new processes on the same transport DATA (`child` of
  `test_normal_sft_cold_continuation.py`, through `resume_prompt_run`). `connected_work` in
  `test_runtime_connected_warm_model.py` reads its handler switch `AMLAS_TEST_CONNECTED_AGENT_HANDLER` from the
  environment at every build, so a child process follows it. Unset, the fixture assigns its own inert handler after
  the build, as before; every other test runs that way. `built` keeps the handler the real builder chose,
  `sdk-printer` assigns the SDK's `PrintingCallbackHandler`, and any other value fails the build. The replacement
  with `sdk-printer` proves the trigger: its stdout holds `Tool #1` to `Tool #N` lines and the final text in front
  of the fixture's answer line. The replacement with `built` must print the answer line and nothing in front of
  it, and the original worker only the entry point's own startup line. Both replacements must reach the same final
  message, model call count and marked tool results, and the stderr of both must hold no tool line and no final
  text. Keep the `sdk-printer` run: it proves that the text would have been printed in that process. Set the
  switch only in a test of console output. This test says nothing about the log of a deployed runtime.
