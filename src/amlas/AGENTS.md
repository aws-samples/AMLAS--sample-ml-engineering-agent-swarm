# src/amlas

## Purpose

Provide the shared application, agent factory and runtime invocation surfaces for Agentic Machine Learning Engineering.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

<!-- normal-native-report-contract:begin -->
- Selected normal prompt reports publish the exact bounded terminal conversation
  to the existing immutable versioned session S3 namespace, then append its closed
  descriptor by exact existing jobs-row CAS. Reconstructed post-worker context
  verifies the original role/table/run/SDK owner; it enrolls or claims no owner,
  lease, epoch or job. Same-content retention keeps the original winning version;
  conflicting content and unresolved publication fail explicitly after SDK cleanup.
  Passive selected-native reads verify the current row and exact S3 version/hash/
  size/bucket owner without API resume, local fallback, model, write or producer.
  Unconfigured runtime retains its local Session behavior. Report flags remain
  conversation-only with false lifecycle/quality claims; no SFT completion is
  required for an ordinary terminal conversation.
  Original prompt workers select native report applicability only from their
  source-owned context after verified enrollment acknowledgement/readback.
  An INPUT-denied unenrolled prompt returns its held response and creates no
  native report/row; its public/model response fields cannot select this choice.
  Direct retention and cold completed contexts remain native-required.
<!-- normal-native-report-contract:end -->

- Normal prompt entry captures its transport session only from the existing
  BedrockAgentCoreContext accessor, once with its derived report/session hash.
  The optional private prepare keyword is supplied only when raw SDK DATA exists;
  no public payload/header selects it and caller signature errors are never retried.
  The existing passive hash-only helper contract remains. Resume workers independently
  capture their current SDK transport/hash under copied context. Private graph UUIDs,
  policies, guards and original job identity remain unchanged; routing DATA grants
  no authentication, native permission, automatic wake or lifecycle acceptance.

- Managed normal resume uses the existing copied-context worker on the original
  run/session, with atomic busy ownership, SDK cleanup and report retention before
  worker completion. Qualified completed work invokes the retained Agent once;
  pending/refused preparation keeps its exact closed status and no final report.
  Start/heartbeat publication waits for the trusted ready-to-continue signal on
  resume; ordinary prompt behavior and current campaign qualification remain.
  Disconnect/deadline do not clear a live worker. No new goal is enrolled/replayed.

- Normal completed resume selects freshly qualified schema2 READY continuation
  DATA before snapshot loading, then wins the existing unique whole-row INVOKING
  CAS before the retained Agent is called. Stable claim tokens preserve SDK hook
  identity; stale receipts and losing contexts refuse. Legacy/started cold claims
  remain closed. Pending/refused preparation retains its status and private scope;
  arbitrary post-effect recovery, automatic wake and lifecycle proof remain separate.


- Every AgentCore invocation applies the existing idempotent `agentcore` layered
  environment before recovery/read dispatch or policy/catalog snapshots. Cold
  report/status/resume routes retain their no-model/no-producer boundaries and
  existing unavailable envelope; initialization keeps existing best-effort errors.

- Normal cold pending resume keeps the original server run/SDK namespace and
  accepts only the fresh native-bound parent/TASK handoff. Normal constructors
  preserve exact acquired/model/CASE/PREP aliases and build same-correlation
  SFT/GEN/PAIR shells. Original SFT restore is read-only and freshly qualified;
  pending mode creates no new goal, SDK task, model/tool invoke or producer.
- Every entire immutable captured prefix message obtains current INPUT admission
  before public `Snapshot.from_dict`/`Agent.load_snapshot`. Constructor cycle and
  receipts are fresh; no saved checker/policy/witness is restored. Only absent
  TASK retains legacy observation; selected checkpoint/guard/UNKNOWN refusal is
  uniform unavailable. Private activation always unwinds before outward stream
  yield. Only the limited first consume/load is provided; invocation/report and
  broader recovery remain separate required slices.


- Wake entry activation exists only in isolated worker/API execution, never across
  outward generator yields. Consumer close/cancellation leaves no private ENTRY;
  incompatible runtime/entry factories refuse. Current qualification follows engine
  construction, precedes model dispatch, follows awaited result and precedes stream
  publication. Revocation produces refusal without exposing a conversation result.
- Explicit spec_path/spec_text precedence remains legacy when campaign extras are
  unrelated. Qualified prompt/wake share cold env/error-log/secret setup before
  policy use; setup does not authenticate callers or approve native boundaries.

- Normal API prompt processing enters the optional private campaign scope around
  the existing ChatEngine. Literal prompt and fresh typed caller policy stay on
  the normal path; no factory means unchanged legacy behavior. App daemon threads
  copy the private context, and API/error/cancellation scopes reset afterward.
  Public prompt remains exactly {prompt}; it cannot attach identity/policy/context.
- App recognizes only the existing exact nine-field private WakeHint when a
  separately qualified server runtime factory is configured. Fresh private ingress,
  protected original C+H and native observation must resolve the original prompt.
  awaiting_original never calls API/agent; verified continuation calls the same
  normal API with the original literal. Conversation response never claims native
  lifecycle/quality acceptance. Missing authentication/configuration refuses.

- `_build.py:build_agent` — Build a Strands Agent with Amlas canonical defaults.
- `_build.py` filters provider sampling/cache fields through shared model
  capabilities; explicit model overrides retain precedence over role defaults.
- `_build.py:build_agent` takes an optional `conversation_manager` (added 2026-11-12). `None` leaves the key out of
  the Agent kwargs, so the installed SDK default applies (a 40-message sliding window on the locked Strands 1.65.0, as
  on 1.42.0; `build_agent` never passes the newer `context_manager` argument, which would replace it); a named manager
  reaches `Agent` only if the installed `Agent.__init__` declares it. Only the normal-run chat backend names one
  (window 100, see `tui/chat/AGENTS.md`).
- `_build.py:context_preamble` — Compose a null-safe steering/problem PREAMBLE for a subagent system prompt (H2-6).
- [Added 2026-11-10] `_runtime_env_pack.py` is the stdlib-only, import-free codec for the
  managed runtime's 4,000-byte environment limit. `pack` folds the closed stage-pin family
  (`AMLAS_<STAGE>_IMAGE_URI`, `_SOURCE_COMMIT`, `_ADMISSION_DIGEST`, the pair launcher digest and
  the four preparation exclusions selectors) and the byte-identical canonical approved-action
  list (frozen profile `campaign_v1`, never the live closed set) into `AMLAS_RUNTIME_ENV_PACK`,
  with its sha256 in `AMLAS_RUNTIME_ENV_PACK_SHA256`. Admission digests stay literal. `unpack`
  refuses a digest mismatch, any key present both plain and packed, and any unknown version,
  stage, field or profile. `__init__.py:_initialize_environment` expands before the root `.env`
  layer, all or nothing, removes both keys, writes one stderr line
  `RUNTIME_ENV_PACK_EXPANDED keys=<n> sha256=<digest>` (no values) and refuses a pack that a `.env`
  file adds. Without a pack it changes nothing. Stage images never receive a pack: adapters pass
  their values as explicit job Environment.
- `_version.py:package_version` — Returns the package version without stating it in the source. A source checkout reads root `VERSION` and maps `N.NN` to `N.<NN as an integer>.0`. An installed package reads its own distribution metadata. A copy with neither reports `unknown`. `__init__.py` resolves `__version__` through it on first access and does no version work at import. `amlas --version` and `amlas info version` print it. It uses only the standard library.
- `_runtime_guard.py:apply_openmp_safety_guard` — Neutralise the macOS duplicate-libomp OpenMP abort (behaviour-preserving).
- `_runtime_guard.py:install_error_excepthook` — Funnel ANY uncaught exception into the centralized ``ERROR_LOG`` (process safety net).
- `api.py:preview` — Plan-Mode preview a capability from Python — build the read-only plan, run nothing.
- `api.py:run_prompt` delegates exact natural input to the normal ChatEngine.
  Caller-supplied ApprovalPolicy is trusted code input, not model-authored JSON.
- `app.py:agent_invocation` — Entry point invoked by AgentCore Runtime.
- AgentCore prompt-only invocations accept exactly a nonempty `prompt` and delegate
  its unchanged string to `api.run_prompt` with default noninteractive approval.
  Optional server-only `AMLAS_RUNTIME_APPROVED_ACTIONS` is a JSON array of closed
  coding-campaign action IDs, transported as a typed ALLOWLIST policy outside
  model input. Explicit configured actions have no automatic cost ceiling; absent
  or empty configuration grants nothing, and malformed/unrelated IDs refuse before
  the API starts. Public payload cannot configure or override this policy.
  Public payload cannot set policy, identity, backend catalogs, native grants,
  environment or session selectors. Legacy spec payloads retain their existing route.
  Prompt heartbeat/completion envelopes label typed ChatResponse data as conversation;
  assistant claims never establish training, lifecycle verification or coding quality.
  Trusted request ContextVars follow the daemon worker. Each API call constructs
  one chat engine and retains it for its privately captured original producer waiting;
  cross-invocation resume and durable stores remain separate required integration. An explicit ops deadline reports
  `outcome_unknown`; returning or disconnecting cannot prove worker/job cancellation.
- The normal prompt worker retains its existing typed response snapshot under
  the server-generated opaque `run_id` in the existing server-CWD Session store.
  Its same-worker write follows successful completion of its own SDK task and
  current entry qualification; persistence failures never mask task cleanup.
  A lock-protected set contains only active original run IDs, installed before
  worker startup. Passive reads remain unavailable even after the snapshot is
  saved until the same worker signals completion in finally and removes its ID.
  Startup refusal also removes the active ID; no completed-entry registry grows.
  `metadata.normal_prompt_report_owner` privately binds schema1 to a hash of the
  SDK session captured at prompt entry, outside the unchanged public envelope.
  The early passive `{action: "get_prompt_report", run_id: <32hex>}` branch starts
  no prompt, engine, SDK task, model or producer. Malformed or extra fields refuse;
  absent SDK context refuses. Absent, unfinished, unbound and foreign-session
  records share one unavailable result without path, owner or existence details.
  Same-session reads return the exact original typed terminal report, preserving
  held/UNKNOWN data and false lifecycle/quality flags. Other workers' busy status
  does not block a completed original. Existing local `api.resume` stays intact.
  SDK session context is not a typed caller-principal claim; actual service IAM,
  native storage writability and client retrieval need live qualification. Cold
  durability and execution resume remain separate work. Save errors log only
  closed labels, never response, session or filesystem details.
- Normal prompt workers register one SDK async task before `Thread.start` and
  complete that exact task in the actual worker finally. Startup refusal releases
  the unowned task and returns only the error type. Deadline or stream disconnect
  leaves a running worker busy until it finishes. SDK ping timestamps follow actual
  status transitions; this ownership proves neither warm continuation nor durable
  restart, managed-job cancellation, native admission or lifecycle completion.
- Caught normal worker failures publish one best-effort warning from the worker,
  independent of SSE consumer lifetime. Closed phases are `work_body` and
  `cleanup_report`; exact reviewed exception types map to fixed categories and
  other types to `Exception`. The deepest reviewed Source frame adds a fixed
  `source_label` and integer `source_line` in 1..200000; an unmapped or absent
  traceback closes to `unknown`/0. Only these labels, line and the validated opaque
  server run ID appear in rendered/structured logs, with no error message, raw
  traceback, path, response, session or private reference. Ordinary logger errors cannot replace the original
  outcome or skip SDK/task/active-set cleanup; logger-origin process control retains
  its existing propagation through worker finally. Existing status/report semantics and worker
  lifetime remain unchanged; a warning does not prove native cause or recovery.

- Owned files: `__init__.py`, `_build.py`, `_runtime_env_pack.py`, `_runtime_guard.py`, `_version.py`, `api.py`, `app.py`, `orchestrator.py`, `prompts.py`, `thesis_experiment.py`.

- api.run_prompt transparently returns the chat engine response, including closed guardrail intervention DATA when verified by the real SDK result. Guardrail action remains UNKNOWN and assessment unavailable; final text and existing execution records are preserved. This outcome is not native admission, blocked-content proof, permission, retry or ML completion. No API input/normalization or guardrail policy change is introduced.

- Normal API returns the original conversation response with closed
  `warm_model_continuation.disposition=submission_unresolved` when the exact
  original MODEL exhausts its finite UNKNOWN observation budget. It grants no
  acceptance/acquisition/native completion or retry. The actual worker exits and
  completes its existing SDK task once in finally, so worker_finished is truthful;
  known PENDING/RUNNING work retains its uncapped wait. No live invocation is
  migrated or reopened by a source update.

- An original CASE's persistent source-qualified verification/observation failure
  may close its conversation wait through the existing submission_unresolved
  disposition and explicit native outcome UNKNOWN. Exact originals/history remain
  retained with no consumption/replay/cancel. Healthy same-original observations
  reset the failure streak; this adds no overall training or runtime-age deadline.

- Normal checkpoint configuration is a private server runtime-role ARN/RoleId
  pair. Both absent preserves legacy behavior; partial/malformed selection refuses
  before model/effect dispatch. Inert run/session data is activated only within the
  original daemon worker and reset finally, never across outward stream yields.
  New exact status/recovery actions select only the server run ID under the actual
  SDK session; payload cannot add identity, policy or native selectors. A verified
  ACCEPTED pending original may be observed under fresh native and resume policy
  gates; UNKNOWN/ambiguous records remain held. These reads start no fresh goal,
  model, staging, Create, Stop or async task. Completed-result continuation and
  automatic replacement wake remain separate required integration. The existing passive report schema and completion barrier remain
  unchanged. Enrollment alone establishes neither cold SFT recovery, automatic
  continuation nor lifecycle/quality completion; those require separate proof.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `arbiter/AGENTS.md` | Grade predictive results deterministically and produce acceptance reports without delegating the decision to an LLM. |
| `async_callback/AGENTS.md` | Translate managed training events into callback/job-state handling and optional runtime resumption. |
| `backends/AGENTS.md` | Define typed backend ports, domain handles and adapter resolution at the workflow-to-compute boundary. |
| `cli/AGENTS.md` | Expose the command-line application and its shared configuration/display entry points. |
| `cloud/AGENTS.md` | Stage datasets/specifications, dispatch managed-runtime calls and resolve configured research secrets. |
| `config/AGENTS.md` | Resolve runtime configuration, selected AWS Region/profile and deployment namespaces at call time. |
| `control_plane/AGENTS.md` | Own capability dispatch, approval/plan policy, cost boundaries, project memory, sessions and integration hooks. |
| `data_eng/AGENTS.md` | Own domain constraints and portable public dataset metadata/acquisition job code. |
| `decisions/AGENTS.md` | Produce rules-based experiment recommendations, next-step decisions and model-card content. |
| `demos/AGENTS.md` | Provide the optional code-interpreter demonstration runner and its explicit staging/execution boundary. |
| `errors/AGENTS.md` | Classify failures, produce user-facing error messages and apply bounded recovery/retry behavior. |
| `evaluation/AGENTS.md` | Store and compare experiment results, leaderboards and statistical evidence. |
| `experiments/AGENTS.md` | Configure and execute experiment plans and hyperparameter selections. |
| `flows/AGENTS.md` | Declare and register reusable ML lifecycle flows with execution and approval metadata. |
| `golden/AGENTS.md` | Load evaluation fixtures and retain frozen train/test split manifests for repeatable grading. |
| `harness/AGENTS.md` | Provision and invoke opted-in managed Lead hosting and resolve its supported contracts. |
| `knowledge/AGENTS.md` | Provide algorithm knowledge and persisted learning used by recommendations. |
| `leads/AGENTS.md` | Declare and build the seven lifecycle Lead agents and their shared tool/prompt contracts. |
| `memory/AGENTS.md` | Connect agent hooks to managed memory and session state. |
| `pipeline/AGENTS.md` | Coordinate local and agentic experiment pipeline execution and result summaries. |
| `pricing/AGENTS.md` | Cache public AWS pricing inputs used for cost estimates. |
| `sagemaker_entrypoints/AGENTS.md` | Run managed tabular training/processing entrypoints and write their expected result artifacts. |
| `scheduler/AGENTS.md` | Manage project queues, research cadence and scheduler/dashboard state. |
| `security/AGENTS.md` | Enforce shared model-artifact loading boundaries across application layers. |
| `spec/AGENTS.md` | Parse task specifications and acceptance configuration into validated models. |
| `subagents/AGENTS.md` | Declare bounded specialist agents and their reusable tools/prompts. |
| `telemetry/AGENTS.md` | Write run traces, error records and execution state for inspection. |
| `tools/AGENTS.md` | Expose agent-callable data, research, ML, deployment and operational tools. |
| `tui/AGENTS.md` | Provide the Textual application, feature flags and shared capability presentation. |
| `training/AGENTS.md` | Fixed Qwen SFT and independent safe adapter reload |

- Normal API retains its original engine/campaign scope while any captured private
  original producer remains pending. Model and DATA/CASE/PREP/SFT/reload/GEN/PAIR
  observations use their exact owning records and seven-store graph. Each tick
  observes one retained original; fixed source-owned15s async sleep rechecks current
  context before/after observation/wait/exposure. Current caller policy and original
  native/artifact boundaries qualify every read and exposure. Independently verified
  completion appends bounded DATA to the same Agent with fresh invocation_state;
  historical ToolResults and signed thinking remain unchanged. Recognized native
  failures retain other live originals. Public response/status/ref/text cannot select
  work; observers never stage/Create/reload/cancel/replay/adopt. Normal AgentTurn
  rendering/fold is reused. App SDK busy ownership ends only in the actual worker
  finally. UNKNOWN holds and current refusal/intervention closes honestly. This is
  warm continuation only; cold durability, native admission, lifecycle completion
  and coding quality remain separate required evidence.

- Direct API cancellation permanently refuses its exact engine-owned originals.
  In-flight producer observation is shielded and drained/joined before cancellation
  propagates; repeated cancellation cannot abandon the copied-context thread.
  Pending-sleep cancellation closes the same original. No native job cancellation,
  resubmission, hard preemption or cold durability is asserted. App client stream
  deadline/disconnect still leaves its independent actual worker/task alive.

- API warm_model_continuation optionally includes the exact source-owned closed
  refusal diagnostic scalars. Pure projection retains message/tools/disposition,
  and no-diagnostic nominal results keep their prior data shape. This is source
  failure observability, never native cause proof, readiness or permission.

- Normal prompt workers privately scope one bounded original-model diagnostic slot.
  UNKNOWN observations may stage immutable six-field Create diagnostics only from
  the exact source-owned original. Stream publication freshly requalifies that
  owner in its copied source context, suppresses current refusal/expiry, and emits
  no refs/jobs/prompt/provider bodies. Worker/stream close expires publication,
  never cancels the original task; public prompt/API signatures remain unchanged.

- Optional diagnostic publication rechecks actual worker completion and any
  authoritative active entry after consumer qualification and immediately before
  yield. A finished worker follows the existing run_completed path; current entry
  refusal suppresses diagnostic DATA without inventing a legacy entry or cancelling
  the worker. This is a bounded publication fence, not atomic-delivery evidence.

## Normal restoration diagnostics

The normal resume failure path remains closed. Its private warning records only a fixed restoration stage and a reviewed builtin exception category; unknown types become Exception. Do not log messages, arguments, exception objects, traceback/frame/source details, snapshots or provider/resource values. A logging failure must preserve the closed failure response and zero effects. The stage set also contains
`continuation` (added 2026-11-07) for an error raised by the continuation itself.
A transient attempt that the restore loop repeats logs `normal_restore_unavailable reason=transient_retry attempt=N
attempts=3 restore_stage=<stage> exception_class=<category>` instead (added 2026-11-10): the same closed
stage and reviewed category, plus the fixed reason and the attempt number. Only the attempt that ends the restore, by
success of a later attempt, a permanent error or the last attempt, decides.
The final refusal logs `normal_restore_unavailable reason=final restore_stage=<stage> exception_class=<category>`
(changed 2026-11-11). Logging extras do not reach the runtime log stream, so the bare text
`normal_restore_unavailable` could not be read for its cause; that bare text is no longer written. The stage and the
category in the text are the same closed values as the two extra fields, which are kept: one of `run_prepare`,
`checkpoint_promote`, `checkpoint_load`, `snapshot_load`, `continuation` (any other stage, `report_retention`
included, still reads `run_prepare` in this line) and a reviewed builtin category or `Exception`. Both forms are whole
`key=value` tokens separated by single spaces. A reader tells them apart by the `reason` token: `final` is the
refusal, `transient_retry` is an attempt the loop repeats, and any other reason is unrecognized. Keep the retry form
byte-equal and add no free text, identifier or path to either line.

Normal resume checks qualified prepared presence only when accepted TASK is absent.
It promotes under the current policy, rechecks TASK presence, then uses the unchanged
qualified loader and SDK path. When neither descriptor is present, resume preserves
the legacy status path. Promotion failure stays closed with the fixed
checkpoint_promote warning stage.

## Status and report polls run off the event loop (added 2026-11-07)

`agent_invocation` runs `get_prompt_run_status` and `get_prompt_report` through `_read_off_loop`. The read (`_read_prompt_run`,
`_read_prompt_report`) runs on a daemon thread started with the caller's contextvars, so the app's event loop keeps serving
heartbeats, other polls and the prompt streams while a read verifies stored model bytes. Polls whose payload is exactly
`action` and `run_id` share one run per (action, run, session hash); each waiter receives its own deep copy of the answer, a
cancelled waiter does not cancel the read, and the entry ends with the read. A poll waits at most
`AMLAS_STATUS_RESPONSE_SECONDS` (default 60, below the client's 120 s read timeout) and then answers the closed
`unavailable` projection (`found` false), which says only that no verified status is available now; the read continues and
the next poll joins it. Requests of any other shape get their closed refusal inline, with no I/O. The read itself is unchanged,
as are `resume_prompt_run` and the worker threads. Why: the first status poll after the SFT task was accepted re-hashed the
17.99 GiB model pack on the loop thread for about 35 minutes; no heartbeat or poll was served and the response stream broke.

## Normal resume is a state query and starter (added 2026-11-07)

A `resume_prompt_run` request answers with one closed first event, emitted before any restore work:
`{"event":"resume_state","state":S,"run_id":...}`. `S` is `restoring` (a resume-owned worker exists
and has not started the continuation; this request started it or found it running), `continuing` (that
worker claimed the completion and is starting the continuation), `completed` (this process finished the
run, or a resume worker of this process found the original SFT job FAILED or STOPPED with its report
retained; a retained report alone is no completion proof, see below), `original_active` (the run is active
in this process but was
not started by a resume), `failed_transient` or `failed` (the previous resume worker's restore ended with
a classified error; the event then also carries `diagnostic` with exactly `restore_stage` and
`exception_class`, both from the closed sets of the restoration warning). The request never blocks its
response on the restore, so the caller's read timeout and drain deadline do not decide the outcome; the
worker thread owns the restore and the continuation and survives the stream ending or being abandoned.
The state is decided under `_PROMPT_REPORT_LOCK`, atomically with the busy fence, and lives in
`_RESUME_WORKERS`: a started restore is `restoring` until `on_ready` (completion claimed, continuation
about to start) flips it to `continuing`; the worker's final step records `completed` or the failure in
the same critical section that releases the fence. A permanent `failed` answers `failed` for 30 s
(`_RESUME_FAILED_COOLDOWN_SECONDS`) without a second restore and then restores again, so a misclassified
error cannot strand a live process; `failed_transient`, a pending original and an unqualified projection
let the next request restore again.

Which original is continued (added 2026-11-11): the restore claims the completion when
`control_plane.normal_run_checkpoint._original_training_complete` accepts the freshly observed original SFT
result: trained, and for any recipe other than `fit_v1` also `full_training_complete`. This is the rule of the
original worker's own wait. A pending original, a FAILED or STOPPED original and every other observation are
handled as before. Why: the restore required the full-training flag, which a `fit_v1` pilot never carries, and
answered `unavailable` for a completed pilot.

The replay-safe restore section (fresh context and engine, qualified reads, conditional writes that
reconcile an unknown outcome by reading the row back) runs at most three times with full-jitter pauses
of `[0, 0.5]` then `[0, 1.0]` seconds, only for throttling, HTTP 5xx, endpoint-connection,
connection-closed, connect-timeout and read-timeout classes (a `ValueError` raised `from` such a cause
takes its class). The continuation is never inside the retry. ValueError/TypeError, byte-limit, digest,
schema, binding and immutable-prefix refusals, access denied, qualification refusals and unreviewed
exception types are never retried. boto3 clients stay single-attempt. `completed` is recorded only by
the worker that finished the continuation in this process, or by an original worker whose model received its
SFT call result (late resume, below). A retained report pointer on the run row is
NOT a completion proof: the original worker also retains its pause-turn response there, so a replacement
that read the pointer as completion would acknowledge a wake it never continued. A late duplicate wake
that reaches a fresh process after the run finished therefore attempts the restore, is refused by the
completion claim (the continuation already has an owner) and answers `failed`.

Status and resume of a run that ended before fine-tuning (added 2026-11-23): a row in stage `enrolled` that
holds `report_checkpoint_ref` is a run that has ended. It has no checkpoint, no continuation and no wake, so
nothing can continue it. `read_normal_run_status` verifies that retained report with the read that
`get_prompt_report` uses, keeps every key and value of the `enrolled` answer (`status` `enrolled`,
`resume_ready` false) and adds exactly three keys: `run_state` `ended`, `ended_action` (`input_admission_held`,
`guardrail_intervened`, or `response_received` for every other response action) and `message` (the run has
ended, the final report is kept, how to read it, how to start again). `resume_prompt_run` for such a row
answers `restoring` and then that same `prompt_run_status` answer, where it used to end with `unavailable`.
No `resume_state` value changes, the answer is never `completed`, and no entry stays in `_RESUME_WORKERS`, so
every later admitted resume reads the row again. The resume path still checks the deployment's approval first: without
the fine-tuning approval it ends with `failed` at `run_prepare`, as for every run. A report that cannot be
verified, or a row that changes during the read, gives the closed `unavailable` answer; before, such a row
read `enrolled`, because the status read never touched the object. A row in stage `sft` is never read this
way, so a run that ended in or after fine-tuning reads as before. A row without a report (an active run, or a
worker whose report could not be stored) and a prompt held before its run was enrolled (no row) also read as
before. Nothing is written. A status poll adds the three keys only after the run's worker released its busy
mark: `_read_prompt_run` drops them for `get_prompt_run_status` while the run id is in
`_ACTIVE_PROMPT_REPORTS`. The worker stores its report before it releases the mark, and a report read answers
`unavailable` until then, so `ended` never precedes a readable report. A poll in that moment gives the plain
`enrolled` answer, or `unavailable` when the report cannot be verified just then. The mark is process-local,
as is the busy check of the report read: a process that holds no mark for the run answers both requests
from the stored report, so the two requests agree in every process. The resume read is not gated, because
the resume worker holds the mark itself; while a resume request of the same ended run is being answered, a
status poll therefore shows the plain `enrolled` answer and a report read answers `unavailable`, as under
every busy worker. Why: such a run read `enrolled` like a live run and its resume ended with `unavailable`,
so a caller could not tell that the run was over or where its report was. Regression:
`tests/test_normal_run_ended_status.py`.

Final report of a resumed run (added 2026-11-07): the worker of a resumed run publishes its typed report
through `_retain_prompt_report(..., _resumed=True)` (`_prompt_worker_events(..., report_resumed=True)`), which passes
`resumed=True` to `retain_normal_prompt_report`: the report goes to the separate resumed pointer and supersedes the
report that an early-ended original worker retained under `report_checkpoint_ref`. The original worker's own path is
unchanged (three-argument call, conflict refusal kept). The resume envelope names the approval policy in force
(`deployment_allowlist` when `AMLAS_RUNTIME_APPROVED_ACTIONS` selects one, else `default_noninteractive`), as the
original's envelope does; it used to say `default_noninteractive` whatever policy the replacement ran under. The
other typed constants of the report (`durable_resume` false, `lifecycle_verified` false, `quality_success` false,
`state_scope` `single_invocation`) are fixed by the closed report validators and are not lifecycle evidence for a
resumed run; read the durable rows for that.

Late resume of a run the original finished (added 2026-11-10): with no worker stop, the original worker
waits for its SFT job in process, its model receives the `submit_qwen_sft` result, it finishes the run and retains its
report, and it never writes a continuation. The stream record for that job keeps retrying while the original is busy
(`original_active` is not acknowledged), so a delivery can arrive after the original finished. The tool backend marks
the delivery on the original's `NormalRunContext` (`note_original_result_delivered`, process-local), and the original
worker's final step (`_record_finished_original`, the `on_finish` that `_prompt_events` passes to `_prompt_worker_events`,
under `_PROMPT_REPORT_LOCK` with the release of the busy fence) records `completed` in `_RESUME_WORKERS` when the worker
returned a response and retained its report. A later resume in this process answers `completed` as its first event (the
stream Lambda acknowledges it) and starts no restore, claim, model call or job; the retained report is unchanged. While
the original is busy the answer stays `original_active`. An original that ended before its SFT result reached the model
(a pause or held turn, a stopped worker) records nothing, so a later resume restores and continues as before. Limit: the
mark is process-local. Neither the run row nor the retained report holds a durable fact that the original's model received
the SFT result (an early-ended original writes the same report pointer), so a late resume that reaches a fresh process
still restores and continues the run. Regression: `tests/test_normal_late_resume_after_finished_run.py`.

Failed or stopped SFT after a worker stop (added 2026-11-10): when the restored original SFT job is FAILED
or STOPPED (`restore_sft_store` now returns such a verified original instead of refusing it), there is no checkpoint
and nothing to continue. If the run row holds no report yet (`NormalRunContext.has_retained_report`), the resume worker
reads the job once more (`_original_sft_end`: native status and a closed reason class from `FailureReason` prefix or
`SecondaryStatus`, never the native text) and, outside the replay-safe restore attempts, retains one source-written
report under the original pointer (`_report_original_sft_end`, action `original_sft_ended`, `data.original_sft` with
`job_name`, `native_status`, `failure_class`, `recipe_id`, `checkpoint` false, and since 2026-11-10
`SelectedInstanceType` and `SelectedInstanceCount` of a preference-list job, both or neither: absent for a fixed-type
job and for a job that ended before SageMaker selected a type). It then answers `completed`; a later
resume answers `completed` from the retained report. No model call, no job, no stop and no retry. Limit: an original
that had already retained an early-end report keeps that report, which does not name the SFT failure. Regression:
`tests/test_normal_failed_sft_after_stop_report.py`.
