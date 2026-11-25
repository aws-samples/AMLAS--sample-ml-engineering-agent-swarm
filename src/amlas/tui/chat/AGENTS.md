# src/amlas/tui/chat

## Purpose

Own conversational turns, deterministic command handling, tool binding, dispatch and visible thought/tool traces.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- The tool-agent prompt directs coding preparation to retain and poll its original
  opaque selector, then pass `preparation_request_ref` (or equal `request_ref`) to
  SFT. Artifact `preparation_ref`/URIs and recipe IDs are distinct. This guidance
  changes no capture, INPUT admission, sticky hold, cancellation or effect boundary.

- Current normal callbacks emit optional closed WARNING activity from inside the
  existing qualification/capture/admission callbacks. Paired before-model,
  before-tool and after-tool enter/exit labels surround their original work;
  after-model emits completion entry only, including SDK error completion.
  Async precommit after-tool exit follows the entire awaited original callback,
  without duplicate inner markers. Exit means the callback returned, never
  dispatch, admission or completion of a native phase. Only the fresh validated
  normal run ID, fixed phase and closed tool category appear; no tool names,
  arguments, results, errors or snapshots log. Default/stale context is silent.
  Ordinary logger/import failures preserve the original behavior; process control
  propagates, and no finally logging masks an original exception. There is no
  diagnostic store, native call, status ABI, retry or authority change.

- SFT capture refusals additionally log only a validated normal run ID, one of
  fixed capture phases, including seven causal validator subphases, and an exact
  built-in exception category. Custom
  types map to Exception; messages, snapshots and other private DATA never log.
  Logging preserves the existing generic warning, hold and cancellation path.
  A later optional warning identifies only the fixed failed SFT input predicate
  and validated run ID; it contains no arguments, references or error text.

<!-- precommit-handoff-progress-diagnostic-contract:begin -->
- Optional precommit warnings observe only already-qualified original sibling
  reads/readiness and completion admission/refusal. A batch marker follows all
  original consumption and history settlement. Fixed kind/status projection
  reads no graph, handle, policy, history or reference. Only DATA's source-owned
  running/submitted/completed labels map to canonical states; unknown values and
  other producer kinds receive no inferred alias. Lazy import/logger errors
  preserve the existing outcome; process-control exceptions propagate. Existing
  guards, holds, waits, ownership and native calls are unchanged. Markers locate
  observations only and grant no readiness, retry or lifecycle authority.
<!-- precommit-handoff-progress-diagnostic-contract:end -->

<!-- normal-completion-tool-trace-contract:begin -->
- Qualified normal warm and cold reports join the current owning Agent's
  INPUT-admitted causal history with current SDK executions. Only a unique ordered assistant
  toolUse and successful user toolResult pair contributes historical display DATA;
  pending, unmatched, ambiguous, malformed or failed pairs grant no executed trace.
  Preserve first-seen tool-name deduplication, the ordinary renderer and current
  metrics argument identity. No SDK metric restoration, history mutation, producer
  effect, admission/retry authority or ML/quality claim follows from this trace.
  Current claim/history qualification precedes extraction and report exposure;
  ordinary `_extract_executions` callers and guardrail callbacks are unchanged.
  Initial normal returns and retained MODEL/producer continuation returns use
  this same projection only while the existing normal context and owning backend
  remain current. Unselected ordinary paths retain their original extractor;
  native/guard qualification stays in the owning callers before and after return.
<!-- normal-completion-tool-trace-contract:end -->

- `continue_normal_completion` invokes only the exact loaded/claimed Agent with
  no new owner message, fresh invocation state and one context-owned attempt.
  Canonical ready DATA is materialized independently for SDK load. Current
  completion/research qualification fences every model/tool boundary and output;
  refusal keeps input hold and the attempt spent. SDK-wrapped failures become a
  fixed closed error, except the held end described under "Held end of the continuation"
  below. No native phase grant is inferred from history. Runtime wake/report,
  cold restoration and replacement ownership remain separate required work.

- A successfully claimed normal completion may rebuild only the terminal research
  restriction from its exact native-bound TASK/prefix, source-owned current units
  and fresh immutable result admission. Missing, duplicate, foreign, changed or
  withheld/unknown result linkage refuses; saved phase/policy/checker flags grant
  nothing. The consumed profile forbids re-fetching and does not replace current
  native/effect qualification. Existing SDK snapshot and conditional claim remain.

- Pending-pause capture accepts the SDK's `messages=None` for history-only
  invocation as an empty incoming batch. Existing Agent history stays intact;
  malformed non-list values refuse. Current owner/research/checker/request-state
  qualification remains unchanged and no new goal or tool batch is synthesized.

- Normal completion assembly remains limited to the existing single-SFT batch.
  Its new source-generated result obtains fresh immutable `TOOL_RESULT` admission
  with `submit_qwen_sft` before consume CAS, Agent construction or snapshot load;
  every complete history message also receives current immutable INPUT admission.
  The original TASK/prefix and non-message SDK state remain unchanged. First
  claim/load grants no inference, takeover, wake or final report; current guards
  and the broader required recovery remain in force.

- SFT batch admission (added 2026-11-07): `submit_qwen_sft` is called alone in its assistant
  message. `ToolBindingBackend._refuse_mixed_sft_batch` is a BeforeToolCall callback registered before the
  input-control callback. When the newest assistant message holds an effectful `submit_qwen_sft` (not the
  `plan_only` preview) together with any other tool use, or a second SFT use, it cancels every tool use of that
  message before any dispatch and before `capture_sft_call`. Each tool use gets its own error result: a closed
  JSON text with status `batch_refused`; the SFT use says no tool in the message ran and to call the SFT alone,
  every sibling says it did not run and to call it again later. The assistant message and the results stay in the
  history and in the causal prefix (nothing is trimmed or rewritten), so the next message can call the SFT alone
  and checkpoint normally. The refused results are admitted by `_mark_pending_pause` only when they equal the
  text returned (`_refused_batch_result`); research bookkeeping, SFT capture and research-result admission skip a
  refused tool use. The exactly-one-toolUse check in `ChatEngine.assemble_normal_completion` stays as the restore
  assertion for TASK checkpoints written before this rule. Why: the live model batched SFT with other tools and
  GEN; the TASK then held a multi-tool final message that no replacement can continue.

- Completed original on restore (added 2026-11-11): `ChatEngine.assemble_normal_completion` and
  `_restore_runtime_research_profile` decide "the original SFT is complete" with
  `control_plane.normal_run_checkpoint._original_training_complete`, the rule of the precommit wait in
  `ToolBindingBackend` (kind `sft`): trained, and for any recipe other than `fit_v1` also
  `full_training_complete`. Their identity comparisons and refusal texts are unchanged. Why: both required the
  full-training flag, which the adapter never sets for `fit_v1`, so a replacement could not continue a
  completed fit pilot.

- Repeat SFT submissions (added 2026-11-07): a durable (normal-run) `SessionSftStore` keeps one original
  training submission. A later effectful `submit_qwen_sft` is classified by `SessionSftStore.repeat_kind` from its
  logical inputs (the opaque model and preparation references and the recipe; never the conversation prefix or the
  tool-use id). `_capture_normal_sft_call` then neither captures nor commits a pause for it, `_sft_repeat_ids`
  keeps it out of the native-producer paths, and the store answers: the same inputs return the admitted
  submission current recorded state (`SessionSftStore._recorded_submission`, `reused` true), other inputs return
  status `sft_already_admitted` naming the admitted reference. Both are normal tool results and the run continues.
  A mismatch during the original admission still holds.

- Specification references after a replacement (added 2026-11-07): a replacement worker's
  `SessionCodingGenerationStore` gets `ToolBindingBackend._restore_planning_refs` as its `_planning_restorer`. On a
  lookup miss (`resolve_case_spec`, `submit`, `_verified_resolved_spec`) the restorer reads the restored
  conversation (`ChatEngine._normal_agent.messages`) and re-registers, originals first, each `bind_coding_generation_spec`
  result whose canonical digest `restore_original_spec` reproduces from the call's `yaml_text`, then each
  `resolve_coding_evaluation_spec` result whose reference a fresh original-case `resolve_case_spec` reproduces. It
  runs once per miss, submits nothing and skips any entry that does not verify, so an unverifiable reference stays
  unknown and the tool still refuses it. Why: the TASK prefix holds specification references from before the stop and
  the replacement store starts empty; a live test run showed the old reference refused five times after a replacement.


- Normal parent per-I/O qualification uses the unchanged live owner/witness,
  PID, research/checker/request-state/precommit/registry and fresh policy checks
  without traversing original producers. Existing full original qualification
  remains before/after capture and brackets the whole parent graph outside its
  adapter I/O scopes. Native role/table/owned-row/catalog checks still bracket
  every actual read/publication; no approval/witness is cached or serialized.

- The selected normal PRECOMMIT BeforeTool boundary supplies its actual public
  session snapshot to the DATA-only capture before any SFT effect. Fresh compact
  TASK schema2 keeps one exact SDK history; strict schema1 reads retain its explicit
  prefix. The full wire and snapshot state remain bounded to160KiB, with no history
  truncation or threshold increase. Completion and runtime-profile consumers derive
  independent prefix DATA through the shared validated accessor. ACK publication
  retains the prepared wire version, exact prefix/call checks and current authority.
  Legacy omitted capture retains the existing session-snapshot fallback.
- Normal binding passes a private live owner qualifier to the existing SFT
  before-submit callback. It reads the current same-backend witness and typed
  policy afresh; no policy/client/witness is saved or reconstructed from JSON.
  Existing graph objects, approval and INPUT admission remain unchanged.

- A selected qualified normal SFT PRECOMMIT call captures only its exact current
  SDK assistant batch/prefix and unique toolUseId before the producer effect.
  Original witness/graph/checker/policy qualification remains fresh; DATA capture
  grants no permission. After unchanged ACK INPUT admission and current custody,
  the same Agent's public session snapshot is saved as bounded plain TASK DATA
  before its first observation/wait. SDK clients, policies, witnesses and runtime
  objects are not checkpointed. Missing selected proof refuses; absent normal
  context or unselected precommit retains legacy behavior. Cold SDK hydration,
  completion ownership and automatic wake are separate required checks.

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
  yield. Completed consume/invoke/report gates remain separate required slices.


- MODEL and generic original-producer resume failures retain only that invocation's
  already captured immutable GuardrailAssessment on the existing held turn.
  Absent trace remains None; bounded UNKNOWN remains separate from known records.
  Guard settings, held/refused disposition, diagnostics, native authority, history
  and callback restoration stay unchanged. Closed/late callbacks and cancellation
  cannot manufacture a response, populate another invocation or permit continuation.

- An uninterrupted turn that is held after its agent call retains that turn's
  already captured immutable GuardrailAssessment on the existing held turn in the
  same way, at three returns of `_chat_with_tools_once`: the SDK-wrapped failure
  with the hold latch set, the return taken when the hold latch is set after the
  agent call returned and the pause-result refusal. A research-binding refusal
  after the run is one cause of the second return; the latch may also have been
  set earlier in the turn by a callback that did not raise. The assessment is turn
  DATA and names no hold cause. Absent trace remains None; no GuardrailOutcome is
  added, and action, message, hold line and diagnostics stay unchanged. The return
  for an unwrapped `_InputAdmissionHeldError` is unchanged: no test pins that
  return with a captured trace.

- A trusted `qwen9b_core_v1` binding with an incomplete research phase may hand
  a complete admitted SDK tool batch to the existing original-handle warm waiter. The SDK request-state flag
  `stop_event_loop` stops only the next model cycle; every sibling result remains committed.
  strands-agents 1.42.0 reads the flag, and 1.65.0 still honours it on a path it marks deprecated.
  A private per-invocation witness validates literal input, prior admitted batches,
  exact toolUse/result history, current profile/policy/owner/registry/checker and
  original records at initial, MODEL and producer-resume results. Fresh invocation
  state and finally cleanup prevent witness reuse. Default behavior is unchanged;
  public flags, mutated history, BLOCKED/UNKNOWN, producer denials and native
  guardrail_intervened never qualify as pauses. No authority, replay, cancellation,
  history reconstruction or denied-session adoption is introduced.

- Only the actual runtime profile binder may select precommit original completion.
  Its exact source-owned three research units must all be attempted and settled
  before the first ordinary tool; that boundary latches readiness permanently.
  Incomplete/direct-acquisition, anonymous/manual and default bindings keep legacy
  behavior. Current owner, typed policy, guard/checker, registry, plan and original
  custody qualify every read/exposure. An async installed-SDK AfterToolCall hook
  waits at the existing 15s cadence for independent exact-original artifact proof,
  then admits one fixed bounded DATA ToolResult under the current INPUT guard,
  retaining the original toolUseId and retryFalse. No synthetic user completion,
  Agent reentry, postcommit history change, guard-policy/cycle change or native effect occurs.
  All siblings finish; exact whole-batch history/admitted bytes and fresh custody
  precede atomic consumption of only matching originals. This mode never sets
  the legacy pause stop flag. Denial/revocation and native uncertainty retain their
  existing holds/budgets; healthy work has no new age cap. A qualified immutable
  closed artifact rejection is checked before CASE uncertainty counting, so it
  stops after the first original observation. UNKNOWN retains its legacy wait;
  nonterminal CASE uncertainty still uses its existing eight-failure budget. Cancellation closes
  read qualification and shields/drains the exact in-flight read before propagating,
  without native cancellation or late result commitment. Native guard and full ML
  acceptance still require fresh deployed evidence.

- Free-form chat uses the tool-bound agent; deterministic slash commands keep their own handler paths.
- Hub metadata inspection is read-only; pinned acquisition is state-changing and
  retains its provider/lineage boundary in tool results.
- `inspect_acquired_dataset` is managed state-changing inspection; its poll tool
  is read-only. Both use the existing trusted acquisition store context key.
- Model/output/history configuration is owned by the shared configuration accessors; preserve the visible degraded-mode behavior.
- Honor existing scoped execution/spend authority in the initial request while
  retaining actual approval/resource gates. Reuse selected deployment references;
  report concrete execution handles or blockers rather than a staged-work claim.
- Optional caller policy is snapshotted per turn and transported through SDK
  invocation_state. It is never inserted into model input or stored globally.
  Missing/invalid authority retains the capability funnel's safe default.
- Direct `acquire_huggingface_dataset` and `get_dataset_acquisition` share the SDK's exact original typed acquisition store and fresh caller-policy snapshot through existing hidden context. Acquisition confirmation remains a separate prerequisite, never approval; missing/ASK policy still denies mutation. Poll is read-only and resolves only the original conversation ref. Caller-authored tool_context/_amlas values refuse before dispatch. Both names remain inline async-handle tools, outside LONG_RUNNING_TOOLS; no generic context or background injection is added.

- Each ToolBindingBackend owns a typed acquisition store with a unique session
  correlation. Every SDK turn receives that store and a fresh policy snapshot
  (safe ASK when absent). The store retains handles, never approval grants.
  References from another conversation cannot resolve. Process restart recovery
  needs a future authenticated checkpoint/native validation path.
- `agent_tools.py:tool_display_name` — Best-effort tool name (Strands ``@tool`` sets ``tool_name``/``__name__``).
- `agent_tools.py:get_safe_llm_tools` — Return the curated list of real ``@tool`` callables bound to the LLM agent.
- Normal chat binds the existing `research_search` and `research_fetch_urls`
  SDK tools lazily as read-only. Their provider failures, source/coverage fields,
  supplied-URL-only retrieval and existing public URL/DNS/TLS/byte/time budgets
  remain owned by the unchanged research layer. Research output is untrusted
  source data, never approval, execution authority or verified research by itself.
- `command_queue.py:QueuedCommand` — One queued chat submission: an immutable (id, text) pair.
- `command_queue.py:CommandQueue` — FIFO queue of pending chat submissions with front-promotion.
- Owned files: `__init__.py`, `agent_tools.py`, `command_queue.py`, `commands.py`, `engine.py`, `flow_dispatch.py`, `guardrail_assessment.py`, `input_admission.py`, `intent_agent.py`, `models.py`, `prompt_cache.py`, `self_context.py`, `thought_trace.py`, `tool_binding_backend.py`.

- Normal construction creates a grant-free preparation store over the same
  conversation acquisition store via the pure ENV snapshot factory. The live SDK
  turn supplies `_amlas_preparation_store` in hidden state alongside fresh caller
  policy. `prepare_coding_dataset` and read-only `get_coding_preparation` are bound;
  model schemas never contain policies, catalogs, native handles or runtime grants.

- Each conversation also owns one grant-free SessionModelAcquisitionStore over
  the same runtime session. Fresh policy and typed model store travel only in
  SDK invocation_state. Pure catalog/opaque poll are read-only; acquisition is
  state-changing and retains normal unknown-cost/native admission gates.

- Safe tools include submit_qwen_sft/get_qwen_sft/reload_qwen_sft. Get is
  read-only; submit/reload are state-changing. They return accepted async native
  handles and use separate polling, not LONG_RUNNING background resubmission.
  Store construction/SDK state and direct-dispatch context remain separately
  owned; public tool binding itself grants no native/image/resource admission.

- Normal chat and the three direct SFT tools share fresh private invocation state. Direct SFT retains the existing confirmation gate, rejects caller-authored context, and never converts confirmation into AUTO policy.

- Normal construction creates one generation store over the original model/case/SFT objects. Hidden `_amlas_generation_store` accompanies fresh caller policy on SDK and direct generation dispatch. Curated tools include spec binding, submit and read-only polling; public tool arguments cannot supply private context. Collection delegation forwards generation only when all original store objects and conversation identity agree.

- Normal chat creates and transports the paired store over its exact original generation/case stores; direct tools and same-session delegation use hidden `_amlas_pair_store` plus fresh caller policy. Curated submit/resolve actions are state-changing, opaque poll is read-only.

- Normal construction may read a sealed server-only producer ContextVar. When
  present, current caller policy gates complete temporary original-producer/SFT
  verification before the backend exposes any restored store. All ordinary SDK
  tools still receive fresh hidden policy and exact original objects. Without
  context, construction remains invocation-local; no public prompt/session/native
  handle JSON can attach a campaign. App/API/callback transport routing is separate.

- A backend constructed before campaign activation cannot later dispatch model
  tools through its legacy state. Direct model acquire/observe fences the retained
  store's original factory before calling the tool; the normal flow enforces the
  same boundary for installed SDK invocation. Lazy model binding refreshes planning
  only through the original factory's fresh sealed DATA view; no frozen ENTRY or
  caller-supplied planning ref is updated to grant authority.

- Selected private planning services build the store over the backend's exact
  original generation graph. SDK/direct save_spec share only hidden typed state;
  genuine legacy state omits the planning key, while active missing capability
  denies before disk publication. Instance-local actual Strands AfterToolCallEvent
  captures bounded research result DATA without changing event/result/retry or
  installing a global wrapper. Public schemas and current policy stay unchanged.
  Backend never mutates factory._planning; cold pre-entry/body readers are separately
  selected ports and native bootstrap qualification remains unestablished.

- A real installed SDK AgentResult with exact primitive stop_reason=guardrail_intervened yields frozen GuardrailOutcome DATA. Normal chat/API preserve reply and existing tool records while exposing action=guardrail_intervened and only schema_version/stop_reason/outcome=INTERVENED/assessment_status=unavailable/action=UNKNOWN. This cannot prove BLOCKED, filter cause, success, permission or retry: PII anonymization can also intervene. Non-SDK/missing/nonprimitive/other reasons retain legacy output. No text classification, raw trace/reasoning, filtering or DRAFT change occurs.

- guardrail_assessment.py projects only bounded pinned SDK content/PII enums and optional exact detected/levels into immutable DATA. Native matches/modelOutput/regex text/actionReason/reasoning/messages are never traversed, retained, serialized or logged. Unsupported policies/fields, incomplete cycles/collections, overflow, malformed/mixed/contradictory actions keep aggregate UNKNOWN. Per-turn instance callback composition chains the existing handler exactly once with unchanged events, restores it and closes capture in finally; capture is synchronized/PID-bound with no unsupported thread restriction. Reply/tool records and guardrail configuration/redaction/retry/policy remain unchanged. Optional guardrail_assessment response DATA supplements rather than changes the five-field INTERVENED fallback; no action or permission is granted.

- A personal-data policy of a trace that has `piiEntities` and no `regexes` collection keeps its entity records with partial coverage, so the aggregate stays UNKNOWN. The model service returned that form for a guardrail that has no regular-expression rule; what it returns for such a rule that did not match was not observed. A policy without `piiEntities` stays invalid. A present `regexes` value that is not an empty list stays partial.

- `list_coding_preparation_profiles` is registered read-only for ordinary SDK and no-confirmation direct dispatch. Direct dispatch injects original hidden state only for its declared tool name and refuses public context/private fields. Discovery reads held catalog DATA without source reads, native calls or lifecycle effects; prepare/get dispatch behavior remains unchanged.

- INPUT receipts attach only a curated currently registered tool name or UNKNOWN,
  the unchanged65536-byte source limit and character count. UTF8 measurement is
  EXACT, a LOWER_BOUND from at most65537 characters, or UNAVAILABLE; no rejected
  body/arguments/URLs/native refs are retained. The ledger remains bounded128 and
  all DATA admission/quality flags stay false. Diagnostic labels never select or
  grant producer authority, alter immutable refs, or replay native effects.

- INPUT receipts optionally expose `blocking_categories` only from validated
  CONTENT/PII records whose native action is BLOCKED. The sorted unique closed
  subset preserves the existing all-category union and requires the unchanged
  BLOCKED/NATIVE_BLOCKED outcome. Missing legacy metadata remains unavailable;
  never infer blockers from the union. Ordinary nonblocking serialization stays
  unchanged. No raw response, text, match, trace, new identifier or permission is
  added; guard decisions, holds, policy, model configuration and retries stay intact.

- Normal ChatEngine construction, shared by API/TUI, may select only server-owned
  `AMLAS_RUNTIME_RESEARCH_PROFILE=qwen9b_core_v1` after original backend warm-owner
  binding. Missing/`off` retains legacy sticky hold; any other configured value
  refuses before model work. The closed source catalog has exactly three separate
  model/licence, training-corpus/licence and benchmark/licence requirements with
  pinned primary URL batches. Public prompts/tool JSON cannot select or enlarge it.
  Current typed policy values, original engine/backend/PID, settings and frozen
  global guardrail mode qualify every bound model/tool/result boundary. When
  checks are enabled, the selected INPUT checker/configuration remains required.
  Explicit `AMLAS_GUARDRAIL_ENABLED=false` permits absent content checks while
  retaining scope, ownership, bookkeeping, consumption and recovery controls;
  forwarded findings create no native ALLOW evidence or assessment receipts.
  Mode/policy/config revocation holds without retry. The existing system prompt
  lists exact finite batches; backend limits remain authoritative. Existing holds
  stay sticky. This source profile supplies no native ingress, permission, full ML
  or quality proof.
  The first ordinary registered nonresearch tool permanently consumes this finite
  phase; later research refuses. Ordinary ML tools retain their existing gates.
  Consumed qualification permits only the exact owner-created checker replacement
  at existing warm COMPLETION_DATA rotation. With checks explicitly disabled there
  is no checker, client or factory to replace: that rotation, precommit original
  completion and the initial SFT acknowledgement requalify bound research and
  return no checker, with no close, factory call, assessment or receipt. An absent
  checker while checks are enabled, or beside any owned transport, refuses.
  Active research never changes its
  checker/cycle; current deployment/policy/config checks remain fresh. Unknown,
  producer and state-changing result denials retain their permanent hold.
  The system prompt requests every fixed research unit before ordinary tools;
  backend bounds, not instructions, enforce authority. One plan is bound per
  single owner-prompt lifecycle; a later owner prompt on the same backend cannot
  reactivate consumed research. Generic subsequent-chat reuse is separate work.

- `_build_context` returns an empty string when no actual state sections exist.
  Empty state supplies no synthetic command hint or context delimiter; normal
  API/TUI preserves the literal owner prompt and its admission. Any meaningful
  state still receives the existing independent GENERATED_CONTEXT admission and
  composite owner/context admission before agent construction. Individually
  allowed components never override a blocked composite. This does not reopen a
  held invocation, alter guard settings, or grant research/native authority.

- A curated research result with actual typed NATIVE_BLOCKED still discards its
  whole body/returned references and preserves the original ToolUse/history.
  The unchanged source-withheld notice is independently assessed by the same
  current guard even at the final unit. Notice must be strict unchanged ALLOW.
  Continuation then requires an eligible unattempted independent unit or a
  nonempty prior forwarded-unit basis in which every retained unit ID came from
  an actual typed strict NATIVE_NONE/ALLOW decision with unchanged source data.
  Track all forwarded unit IDs, including masked source data, separately from
  strict admitted IDs. Every retained unit must have a separate origin, no blocked
  URL overlap, dependencies or variant relationship. An unrelated witness cannot
  qualify other retained history; attempts, results, masks and the notice cannot
  establish this basis. Do not rebuild or purge history. The eligible unattempted
  unit path keeps its existing scoped semantics.
  With neither independent path, the invocation remains held. All UNKNOWN,
  producer/state-changing and notice/policy/checker denials remain sticky; no
  retry, rephrase, new unit or native/licensing/capability grant follows. Ordinary
  tools retain their existing gates and consume the finite phase. Conclusions
  must retain withheld research/licence uncertainty, never claim full C1/ML proof.

- An exact original MODEL has a source-owned budget of eight consecutive fresh
  qualified UNKNOWN/observation_failed polls at the existing normal API 15-second
  spacing. At exhaustion, latch `submission_unresolved` with closed
  `native_observation/original_unknown_budget_exhausted/UNKNOWN` diagnostic DATA.
  Retain the same pending object/store/ref/handle/ARN, accepted/ambiguous flags and
  every historical SDK byte; do not consume, cancel, replay Create or imply native
  rejection/termination. This closes only the conversational wait and prevents
  reminting. PENDING/RUNNING resets the counter and keeps legitimate long work
  uncapped; verification before the budget continues on the same Agent. Current
  qualification, guard/policy revocation and cancellation/drain take precedence.
  Other producer waits, native job limits and general runtime deadlines are unchanged.

- Only a source-qualified original CASE has an eight-consecutive-failure budget
  for explicit verification_failed/observation_failed observations. Current owner,
  record/graph/pin/policy qualification and closed immutable rejection precedence
  remain unchanged. An authoritative same-original PENDING/RUNNING observation
  resets only that CASE's streak; healthy jobs have no age deadline. Exhaustion
  reuses submission_unresolved and the closed UNKNOWN diagnostic, with explicit
  user-visible native outcome UNKNOWN. Keep the original pending/store/ref/job IDs
  and every historical byte; do not consume, replay, cancel or grant another effect.
  Other producer kinds and native UNKNOWN waits retain their existing behavior.


- SFT BeforeTool capture refusal latches the existing INPUT hold and cancellation
  before its fixed `sft_capture/callback_refused` warning. Full TASK size, current
  custody and snapshot/history qualification remain unchanged; logger failure
  cannot reopen the tool or permit a native effect. No exception or task data is logged.

- Accepted producer-capture refusal latches the existing INPUT hold and then
  emits its fixed `producer_capture/callback_refused` warning once. Original
  custody, history, no-replay and refusal remain unchanged; ordinary logger
  failure cannot alter those outcomes. No exception or provider data is logged.

- Existing refusal sites emit fixed WARNING records beginning `AMLAS_INPUT_HOLD`
  with compile-time phase/reason labels only. Research before/result, nonresearch
  tool projection, before-model,
  pending bookkeeping and existing history latches retain their original decisions.
  Do not log exceptions, traces, caller/source text, URLs, references, IDs, hashes or
  keys. Already-held sequential cascades are suppressed; records locate an observed
  source refusal phase, not a native blocked component or a globally ordered cause.
  Logger configuration, guard settings, eligibility and response schemas are unchanged.
  Existing latch/cancel/refusal outputs precede best-effort warning delivery; the
  private helper accepts only the fixed literal whitelist and discards ordinary
  logging exceptions without inspecting them. Process-control BaseException
  propagation remains unchanged.

- Hold attribution (added 2026-11-07): every place that latches the invocation INPUT
  hold (`_input_hold = True`) in `tool_binding_backend.py` and in `engine.py`'s normal-completion fence names a
  closed `AMLAS_INPUT_HOLD phase=... reason=...` literal from `_INPUT_HOLD_WARNINGS`, emitted only for the first latch of
  an invocation. An earlier version left thirteen latches silent (research activation, `_admit_input`, duplicate original, history
  checker selection, history serialization, completed-original change, research binding after the run, pause
  result, pause start, completion admission) plus two in the completion fence; a hold that began there was
  reported by no literal, and a later site saw the hold already set and stayed silent too.
  `tests/test_input_hold_attribution.py` reads the source and fails when a latch has no literal after it in its
  function or a literal is outside the set. Decisions, latching order and responses are unchanged. Why: a held run
  could not be attributed to a phase from the log.

- Unresolved original polling cap (added 2026-11-07): while a pending original (DATA, CASE, PREP, SFT,
  RELOAD, GEN, PAIR) completes before its tool result commits, consecutive UNKNOWN, observation_failed,
  verification_failed or output_verification_failed observations are counted with a monotonic clock
  (`_precommit_unknown_clock`). Once the run spans `_PRECOMMIT_UNKNOWN_WALL_CLOCK_SECONDS` (900 s: 7.5 times the
  120 s MODEL/CASE budget, 5 times the longest single observation round of a healthy run, 1/16 of the 4 h driver
  budget) the wait closes: `_warm_closed = "submission_unresolved"` with the closed
  `native_observation/original_unknown_budget_exhausted/UNKNOWN` diagnostic, and `_InputAdmissionHeldError` ends the
  wait exactly as the MODEL and CASE budgets do. Any other answer (PENDING, RUNNING, a complete result) clears the run,
  so healthy long jobs have no age limit, and success, failure and rejection rules are unchanged. The API warm
  loop's producer observation (`_observe_pending_producer_once`) counts the same way and returns
  `submission_unresolved` with the same diagnostic at the cap. The log line
  `normal_unknown_poll run_id= operation= polls= elapsed_s= limit_s= outcome=waiting|closed` is numbers and fixed
  labels only; it is written for the first unresolved poll, every eighth, and the closing one. Why: a publisher
  exception reports UNKNOWN on every poll, and the loop had no cap, so one defect held a run for the whole 4 h.

- Per-model-call conversation size (added 2026-11-07): a BeforeModelCall callback registered before the
  admission callbacks logs `normal_conversation_size run_id= model_call= bytes= messages= limit=` while a normal run
  is active. `bytes` is the canonical JSON length of the record form of `agent.messages` (`_record_messages`),
  exactly as `_canonical` measures a causal prefix
  (the SFT capture envelope adds `43 + len(toolUseId)` bytes around it); `limit` is the fixed 160 KiB cap; the count
  is not capped, so an over-cap conversation is reported with its real size. The callback logs numbers only, ignores
  every failure, and never raises into the model call.

- Model-call retry (added 2026-11-07): an AfterModelCall callback repeats a failed model call up to three
  times (`_MODEL_CALL_RETRIES`) for a closed set of transient errors: `ModelThrottledException`, a botocore
  `ClientError` whose code is a throttling or server code or whose HTTP status is 500, 502, 503 or 504, and the
  botocore endpoint-connection, connection-closed, read-timeout and connect-timeout errors. Backoff is full jitter
  with caps of 1, 2 and 4 seconds. The callback sets `event.retry`, so only the failed call is repeated: the prompt is
  not appended again and no tool runs twice. The budget restarts after a successful call. A held run is never
  retried. Other errors, including access, validation, expired-credential and context-window errors, end the run as
  before. The line `model_call_retry run_id= attempt= limit= reason= delay_ms=` is numbers and fixed labels only.
  Why: the SDK retries throttling only and no 5xx or connection error, so one Bedrock 5xx after the boto retries ended
  a run.

- A selected normal checkpoint may enroll the original empty graph only after
  OWNER_PROMPT/generated context/history admission and before build_agent/model
  dispatch. It receives admitted forward text and the original acquisition-store
  correlation; no policy/grant or rejected content is serialized. A selected valid
  SFT catalog binds the SAME empty SFT object to the scoped original journal and
  prospective DATA callback. Missing unselected SFT configuration preserves ordinary
  prompts; SFT submission keeps its fail-closed catalog validation. Drift or binding
  failure never permits unjournaled selected SFT. Construction and warm graph
  identity pins remain unchanged. A checkpoint refusal escapes before
  any model/native producer; the normal worker retains its typed failure cleanup.

- Normal model-store construction may select the configured verified-pack provider.
  Current policy is read through the owning backend getter, never from persisted
  snapshot DATA. Adopted SFT sources use an explicit seven-field completion codec
  retaining the actual original producer and consumer staging refs; no reflection,
  retry, private handle adoption or native readiness follows from codec DATA.

- Verified original CASE completion DATA retains its canonical `case_ref` alongside
  the unchanged generic `request_ref` in both precommit and warm continuation.
  Both identify the same freshly qualified original; non-CASE envelopes remain
  unchanged. Consumers may pass that verified reference as `case_request_ref`
  for governed preparation exclusions. The alias grants no new authority and
  does not alter reference/job identity, INPUT admission or lifecycle readiness.

- Original SFT result delivery (added 2026-11-10): `_note_normal_sft_result_delivered()` marks the active
  `NormalRunContext` when the original's model receives its SFT call result: in the precommit wait right after
  `event.result` is set for an `sft` producer, and in the warm producer path right after an `sft` producer is consumed.
  It never raises and changes no tool result, admission or history; the app uses the mark to answer a late resume
  `completed`.

- Normal-run conversation window (added 2026-11-12): `ToolBindingBackend.build_agent` passes a new
  `SlidingWindowConversationManager(window_size=100)` (`_NORMAL_RUN_CONVERSATION_WINDOW`) to `_build.build_agent` on
  every build. The SDK default (40) trims the oldest messages once the history holds more than 40, and a trim after
  the fine-tune pause edits a history the input witness refuses (`history_prefix_refused`); one real run held 33 messages
  at the call after the pair result. The saved manager state keeps the same keys and class name, so
  `conversation_manager_state` keeps its shape and a state written by the default manager restores. No other
  `build_agent` caller names a manager.

- Prompt-run console (added 2026-11-14): while a normal run is active (`current_normal_run()` returns a
  context, or the lookup raises an ordinary exception such as a context that fails its own check) and no thought
  trace is given, `ToolBindingBackend.build_agent` assigns the SDK's `null_callback_handler`. The SDK default (`PrintingCallbackHandler`) printed every assistant text delta and one
  `Tool #N: name` line per tool call to the runtime's stdout, and the product stores none of that text except the
  final report. The rule covers the three builders of the run's agent: the first turn, the pending restore
  (`load_normal_snapshot`) and the completed restore (`claim_normal_completion`). The handler stays a callable, never
  `None`, because the guardrail capture wraps it at every invocation. A build with a thought trace keeps the trace
  handler and a build without a run context keeps the SDK printer. No log line, hook, report field, refusal,
  identity, record schema or checksum control changes. Agents that a tool builds through `_build.build_agent` keep
  the SDK default.

- SDK record form (added 2026-11-15):
  Live SDK data becomes record form at the SDK boundary. Stored-record reads are never normalised.
  Stored records, schemas, versions, refusal texts, hashes and runtime log lines stay unchanged.
  `_record_messages(messages)` returns a new list of copied message dicts without top-level `tracking_id`.
  It removes nothing else. It keeps `metadata` and nested `tracking_id` keys. It leaves live messages unchanged.
  `_record_session_snapshot(agent)` calls `agent.take_snapshot(preset="session").to_dict()`.
  It deletes `model_state` from snapshot DATA only when the value is an exactly empty dict.
  Any other value is refused with "public SDK session snapshot DATA inventory differs".
  That refusal leaves the capture in phase `snapshot_create`. Snapshot messages then become record form.
  `_pause_history(messages)` produces record-form witness text for one message or a message list.
  `_pause_result(result)` produces witness text for one tool result exactly as given.
  Any added top-level tool-result key, including `tracking_id`, holds with `history_batch_refused`.
  Fine-tune call capture and admitted TASK publication use `_record_session_snapshot` and record-form messages.
  `normal_conversation_size` measures record-form bytes of `agent.messages` for the 160 KiB cap.
  `ChatEngine.load_normal_snapshot` compares loaded history in record form with the stored prefix.
  Any other difference still raises "loaded SDK snapshot changed admitted history".
  A message's top-level `tracking_id` may be present, absent or any value. It is never recorded.
  An empty `model_state` in a live SDK snapshot is tolerated. It is never recorded.
  Changes to message role, content, metadata, order or count remain refused.
  Any other added message key remains refused. Any tool-result change remains refused.
  `normal_run_checkpoint.py` keeps its exact four-field validator and refuses stored `model_state`.
  New live-history comparisons, captures and size measurements use these helpers. No other key is stripped.
  strands-agents writes `tracking_id` from 1.47.0 and `model_state` in the session preset from 1.43.0;
  `tests/test_sdk_record_form.py` pins these rules on 1.42.0 and 1.65.0.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.


Normal SFT binding supplies a private getter for the backend's current typed approval policy. Prepared snapshot publication retains the existing live original/current qualification callbacks; the getter and policy are never serialized into SDK DATA or checkpoint descriptors.

## Child DOX Index

None.

- A current campaign creates initial store shells with its private correlation,
  and defers model binding until the model tool, after original spec publication.
  Direct model acquire/observe dispatch carries fresh typed hidden caller state
  through the same SDK context path; public args cannot supply private context.
  Existing SFT and campaign contexts must share original campaign/private UUID.

- Legacy private one-step model observation is bound to the exact normal ChatEngine,
  backend, original seven-store graph and construction PID. Binding selects only
  warm lifetime DATA; it grants no caller/native permission. Standalone backend
  turns remain legacy. A registered SDK model result may retain only its actual
  owning record/ref/handle/toolUseId and same Agent/history. Public response,
  status/ref/context fields cannot select an observer. Original native/S3 reads,
  model dispatch and exposure recheck fresh caller policy and original bindings.
  PENDING/RUNNING/UNKNOWN hold; independently verified nine-file exact-version/hash
  completion leaves its matching pending ToolResult/history/thinking prefix unchanged
  and calls the same Agent with one source-owned user completion DATA ContentBlock
  and fresh invocation_state. Consumed originals remain latched for the warm lifetime and never remint work.
  Pending/closed owners refuse new turns without discarding/ref-reviving history;
  failures/refusal/intervention close without retry grants. Capture and calls are
  separately synchronized; ambiguous captures refuse instead of last-wins. Per-turn
  guardrail callbacks restore in finally. No API/app automatic loop, persistence,
  cold recovery, native admission, quality or full lifecycle acceptance is implied.

- A failed first SDK turn after original acceptance closes its engine-owned pending
  continuation. Repeated qualification rejects a latched closure before another
  read/history/resume; consumption is latched under the capture lock without historical mutation. Cancelled resume restores callbacks and permanently closes in finally
  while SDK process-control propagation remains unchanged.

- API warm selection uses only the exact engine/backend/pending/PID/seven-store
  anchors under the existing capture lock; selection is no permission. Current
  one-step qualification still fences all reads/model/exposure. The normal
  AgentTurn renderer/fold is shared with same-Agent continuation without calling
  process_message again or reconstructing SDK history. No app/TBB grant/guard or
  guardrail strength changes are introduced by the API connection.

- Exact per-engine cancellation closure sets the existing capture-latched refusal
  under its lock. Existing post-I/O qualification refuses further reads/model/
  exposure. Async observation shields and joins the same worker before propagating
  cancellation; callback finally restoration stays inside that worker. No capture
  lock is held across SDK/hooks, no native cancellation or retry grant is added.

- Original MODEL capture derives eligibility from the exact stored JobHandle's
  strict bool lifecycle pair: acceptedTrue/ambiguousFalse or acceptedFalse/
  ambiguousTrue. Actual SDK event/store/ref/native identity remains required;
  public accepted/status DATA grants nothing. PreparedFalse/False, mixed/untyped
  lifecycle and untyped handles are ignored. Lost native reply retains the same
  original for read-only UNKNOWN hold or independent nine-file/three-manifest
  completion; never promote accepted flags, mint a ref or replay Create. Fresh
  graph/PID/pin/correlation/policy/IO and SDK append-only history/same-Agent checks remain.
  Competing MODEL captures still refuse; the closed source-owned producer
  registry separately retains genuine non-MODEL originals without granting effects.

- Warm refusal diagnostics are frozen optional phase/reason/error_class DATA from
  closed source enums. Existing source qualification/artifact/history/resume gates
  retain their behavior; no exception text, reflection codec, trace or IDs are
  exposed. The first diagnostic is retained with the permanent refusal latch under
  the existing lock and repeated closure returns that same object without I/O.
  Cancellation/BaseException propagation, callback finally, capture/provenance and
  caller/native permission remain unchanged; diagnostic DATA grants no retry.

- Original MODEL diagnostics are source-only DATA from the genuine SDK capture,
  retained private record, current graph/PID/pin/context and typed caller policy.
  One frozen publication has a 15-second telemetry TTL and invocation/thread scope;
  copied contexts cannot outlive either scope. Consumer current qualification and
  postqualification local fences precede six-scalar exposure without producer SDK
  I/O or locks across qualification/yield. No public selector, callback serialization,
  rejection/retry grant or warm-worker lifecycle decision is added.

- The private frozen original-model completion event is minted only after current
  original owner/graph/PID/policy/pins, independent nine-file/three-manifest proof
  and unique historical toolUse causality. Its fixed canonical JSON DATA envelope
  retains only exact opaque ref/toolUseId and fixed descriptive completion fields,
  with a4096-byte cap; existing identifier bounds are unchanged. In the legacy path,
  one consumed event appends through supported SAME-Agent user ContentBlocks, preserving every prior
  toolResult/thinking/message byte. The external literal prompt remains once; no
  public constructor/schema parameter, source selector, instruction, native URI,
  error, approval/retry/native grant or new graph/controller/journal is added.

- Warm normal continuation captures only actual registered SDK producers from
  their same hidden original store/native record, Agent, graph/PID and closed typed
  pin. Retain unrelated and next-resume originals without a lower workload cap.
  Fresh current policy/IO and owning artifact verification precede one source
  completion ContentBlock on the same Agent; historical provider prefix and pending
  toolResults remain untouched. Observers never stage/Create/reload/cancel/adopt.
  Before observation, current DATA/PREP record adapters and other producer-store
  adapters must remain the captured original instance receiving the private IO scope.

- Each producer completion additionally binds its unique historical toolUse result
  payload to the exact captured opaque ref and native job name/ARN. No historical
  result is rewritten. Failed/stopped MODEL consumption closes only that alias
  while genuine sibling originals remain live. Source-selected Pair pins explicitly
  project exact CodingEvaluationBinding/ManifestRef fields; no public dictionaries
  or arbitrary model serialization mint private original identity. Read-only UNKNOWN
  and verification uncertainty hold the original, never replay an effect.

MODEL completion requires one retained historical ToolResult with the original
toolUseId AND bounded original request_ref/native job name/ARN. Substituted DATA
refuses before consumption or another model call; original reads, record, raw
history/thinking prefix and current policy remain unchanged. No public authority
or history rewrite follows from the causal comparison.

- A freshly qualified owning observation may carry the closed source-produced
  verification_rejection DATA discriminator. Exact ref/native identity and marker
  types precede consuming/tombstoning only that rejected pending original. Genuine
  sibling entries remain current; no successful Agent completion, replay, native
  cancellation or public JSON adoption follows. Unmarked verification/transport/
  deadline/publication uncertainty still holds the original.

- Configured normal chat privately checks new owner/context/history/completion text
  and new SDK tool projections using the same selected guardrail with INPUT
  semantics before model exposure. Rejection/unknown holds are sticky across tool
  callbacks. Accepted producer handles remain in their original private stores;
  historical messages/signed thinking/pending results are never rewritten.
  Model guard settings and caller authority remain unchanged. Finite admission
  receipts are DATA only and do not identify a historical native blocked component.
  Standalone INPUT checks do not replace the native model guardrail.
- input_admission.py is a private pre-exposure INPUT primitive with source-selected
  immutable guard identity/region/client and an owner-ephemeral HMAC key. One
  INPUT/FULL check retains the exact original only on strict native NONE; only
  verified service PII substitution may supply an ANONYMIZED replacement.
  BLOCKED/UNKNOWN has no forwardable body.65536UTF8bytes/128records/oneoutput
  caps fail closed without truncation or splitting. Frozen content-free receipts
  carry opaque check/cycle/boundary, bytes/keyed digest/policy/outcome/categories
  and false authority flags; no body/match/key/header/native handle persists.
  Transport/malformed/contradictory/invalid masking refuses; process control
  propagates. The primitive exposes no public tool and never changes client configuration
  or retries. Trusted backend-owned clients opt into a frozen pool-close callback
  immediately before each new valid assessment's sole attempt. The same client,
  checker, guard/HMAC and active research witnesses remain selected; invalid input
  touches no transport. Pool-close or assessment failure retains UNKNOWN without
  retry. Caller-injected clients keep ownership by default. Backend-owned pool
  clearing is serialized by its input RLock; standalone trusted opt-in callers
  must serialize access themselves. Pinned SDK close/reuse compatibility must be
  verified when its dependency changes. OUTPUT monitoring
  semantics and original history remain unchanged.
  Transport failures optionally retain an immutable six-field diagnostic: fixed
  phase, finite SDK class/service code, qualified HTTP status/UUID request ID/retry
  count. Unknown fields stay UNKNOWN/null; exception messages, headers and bodies
  are never retained or stringified. Extraction failure supplies the default
  closed diagnostic without replacing the UNKNOWN hold. Diagnostics do not grant
  retry or admission; process control still propagates.
  Before each legacy COMPLETION_DATA or source-qualified final original TOOL_RESULT
  assessment, the owning backend retires
  its selected client pool and constructs a fresh client from the same SDK
  Session under the input lock/current configuration check. The frozen checker
  keeps its exact guard policy and HMAC. This is proactive transport lifecycle,
  never replay of UNKNOWN/failed assessments or native producers. Fresh-client
  creation/assessment uncertainty still holds before continuation consumption.
  Final precommit retirement freshly qualifies the exact original under the input
  lock and moves only its consumed research/current private witness to the owner-
  created checker. Current checker qualification uses that same lock; active
  research and arbitrary TOOL_RESULT checks cannot rotate. Keep config/HMAC/cycle,
  exact custody/history and post-ALLOW qualification unchanged. Close/factory or
  assessment failure remains sticky before result replacement or consumption.
  An initial `submit_qwen_sft` acknowledgement with a uniquely captured original
  and current qualified precommit witness reuses that same retirement/admission
  before its first assessment. Register only the exact original in the witness;
  consumed research, literal batch/history, policy/custody and checker fences stay
  current under the input lock. Admit its unchanged acknowledgement as DATA only;
  no completion or consumption is inferred. Uncaptured/default acknowledgements
  and all other initial producer paths retain their existing behavior. The final
  original completion still performs its separate first-assessment retirement.
  Keep one ApplyGuardrail attempt; failed/UNKNOWN assessments are never reassessed.

- The coverage check of input_admission.py reads `guardrailCoverage.textCharacters`.
  The native service reported that count in UTF-16 code units.
  A character outside the Basic Multilingual Plane counts as two UTF-16 code units.
  The check requires `guarded` to equal `total`.
  It requires `total` to equal the length of the sent text in UTF-16 code units.
  A `total` that counts code points for text with such a character is too short.
  Any coverage that fails the check stays UNKNOWN with NATIVE_INVALID and no forwardable body.
  A native answer with `guardrailCoverage` absent or set to `None` is not checked or refused
  for coverage, but every other check still applies.
  The checker encodes the text strictly as UTF-8 before the native call, so text with a
  surrogate code point is refused before any count is taken.
  The receipt's `character_count` stays the number of code points.
  The unit was recorded for one guardrail in one Region.

- One instance-local AfterToolCall coordinator explicitly captures producer custody
  before checking its new projection; readonly content admission precedes planning
  capture. SDK reverse callback registration cannot reorder these internal steps.
  Whole original history messages include nested results and are checked immutably.
  Source-owned holds survive SDK hook wrappers and warm refusal, retain original
  records/callbacks and refuse completion before consumption. Admission receipts
  retain at most128 records; these source checks grant no native ML acceptance.

- Private admission receipts remain frozen objects; every report and normal
  response gets detached record/category mappings. Capture uncertainty preserves
  an accepted store record and latches refusal before further model dispatch.
  Genuine guard-disabled legacy history passes through without admission-only
  serialization; prior holds or selected-config drift still refuse.

### Completion receipt research restoration

Research restoration reads the current completion receipt representation. Exact
integer schema 3 compares the completed SDK snapshot through the checkpoint
snapshot-digest validator; legacy inline receipts retain their canonical snapshot
comparison. Keep the existing SDK inventory and 160 KiB cap, strict schema 1/2 TASK
reads, original-job and approved-history checks, token/current-context ownership,
research-phase fencing, authorization and CAS unchanged. The actual cold research
restore regression and changed-digest refusal qualify this reader before the full
workflow gates.

### Pending restore with guardrail checks disabled (added 2026-11-07)

`ChatEngine.load_normal_snapshot` (the PENDING/RUNNING restore branch) no longer refuses solely because
the selected INPUT checker is None. With `AMLAS_GUARDRAIL_ENABLED=false` (the deployed configuration) the
checker is None by design and `_admit_input` is a pass-through, as on the completed-claim path. Enabled
checks without a selected checker, and a snapshot that differs from the TASK causal prefix, still refuse
with "current normal snapshot admission unavailable".

### Restoration notice in the continuation (added 2026-11-08)

`recovery_notice.py` owns the one runtime-originated notice that a conversation restored from its checkpoint receives.
`ChatEngine.continue_normal_completion` runs `check_recovery_notice_ready` before the single-use
`begin_completion_invocation` (the Agent system prompt must be text or absent and without the marker, its content blocks must all be text so a written-back
string drops nothing, the claimed receipt must carry a readable epoch, both epochs must render and the SDK setter must accept the value, so a refusal burns nothing) and `apply_recovery_notice` after the start CAS has succeeded and before the one
`agent(None, ...)` call. The notice is appended to `Agent.system_prompt` only: the SDK messages, the completed-task snapshot,
`sdk_snapshot_sha256`, the 160 KiB accounting, every qualification pass and the conditional claim stay exactly as claimed, and no message
block or ToolResult text is added. The notice states the restore from the saved checkpoint, the invoking epoch of the claimed continuation
row (read from the receipt: epoch 2 for a schema 3 row, epoch 1 for a legacy row), that the invocation which made the `submit_qwen_sft`
call has ended, that the earlier history is unchanged and that the runtime supplied the call result after it verified the job completed. It
names no cause and no identifier, and it asks the agent to state the restoration as a fact in the persistence and recovery part of its
final report (the runtime does not know why the earlier invocation ended, so it says nothing about that). An Agent built without a system prompt receives the notice alone. The helper is idempotent (a second call is a no-op), refuses an
inconsistent earlier notice, and logs only the epoch and a hash prefix. A lost start CAS applies no notice and makes no model call. `current()` also refuses when the
system prompt differs from the prompt that carried the notice, so a changed prompt never reaches the model. The notice is not durable
history and the closed 12-key report object is unchanged: the continuation row epoch and the resumed report checkpoint remain the
durable record of a recovery. Regression: `tests/test_recovery_notice.py`.

### Runtime recovery record and restore line (added 2026-11-09)

`recovery_notice.py` also writes two records that do not come from the model.

- Runtime recovery record. After the start CAS, `continue_normal_completion` builds a `## Runtime recovery record` block
  (`recovery_report_paragraph`) from the invoking receipt and appends it with `append_recovery_report`, after one blank line, to
  `ChatResponse.message`: once, after the turn response exists and before the last `current()` qualification. The block is the last block of the message. It says the worker invocation that began
  the run ended, that a later invocation restored the run from its saved checkpoint and continued it as continuation epoch 2, and
  prints the SFT operation digest, the SDK snapshot sha256 and the task checkpoint sha256 of the row, each as eight groups of eight
  hex characters so that no decimal run longer than eight digits occurs (three plain digests hold a 12-digit run in roughly 40 percent of
  cases, computed and not measured, and such a run is read as an account number). The wording follows one recorded fact, `context._completion_prior_report`
  (set by `claim_completion` from the presence of `report_checkpoint_ref` in the claimed row): absent gives "stopped before it saved
  any report" and "replacement"; present gives "ended after saving its pause report" and names neither a stop nor a replacement; not
  recorded gives the neutral wording, which says the receipt does not establish why the earlier invocation ended. Only a schema 3
  receipt that is `invoking` at epoch exactly 2 gets a block. Schema 1 and 2 receipts and `ready` receipts get none, and their
  response is byte-identical to the one without this change. Any other schema 3 epoch, an id that is not 64 lowercase hex, a task
  descriptor without `sha256` or a prior-report fact that is not a boolean is a `ValueError`, found on the `ready` receipt by
  `check_recovery_notice_ready` before the start CAS so that a refusal burns nothing. The block is not in the system prompt, the SDK
  messages or the snapshot, and the closed 12-key report object, its size bound and its content hash rules are unchanged.
- Restore line. `log_restore_completed(context)` is called right after `begin_completion_invocation` returned (the start CAS is won) and
  before the first model call, not only when the system-prompt notice was applied. It writes one WARNING line of logger
  `amlas.tui.chat.recovery_notice`: `normal_restore_completed run_id=<hex32> phase=continuation_started epoch=2 stage=sft
  operation_digest=<hex64> sdk_snapshot_sha256=<hex64> task_checkpoint_sha256=<hex64> prior_report=<absent|present|unknown>`. It means
  that the checkpoint is restored and the continuation has started, not that the run completed. A record that cannot be built gives
  `normal_restore_record_irregular run_id=<hex32|none> reason=<bad_run_id|bad_epoch|bad_digest|bad_descriptor>`. A lost start CAS, a
  failed restore and a receipt of another schema write neither line, and every logging failure is swallowed. WARNING is the level
  that reaches the runtime log: the runtime image configures no logging, so Python's last-resort handler writes a WARNING message bare
  to stderr and drops INFO (the INFO line `recovery_notice_applied` reached CloudWatch 0 times in one run; `normal_callback_activity` from the
  same package at WARNING reached it 178 times). Neither line holds a term that the readback collects as a warning.
  Regression: `tests/test_recovery_report_record.py`.

### Held end of the continuation (added 2026-11-23)

`ChatEngine.continue_normal_completion` makes one `agent(None, ...)` call. It ends the run with the fixed held turn only
if that call raises, an admission hook sets the input hold inside the call, and no fence refuses. `current()` runs just
before the call and refuses a hold, so a hold seen after the call was set inside it. `fence` marks a refusal in
`notice_state["fence_refused"]`. A fence refusal and a failure without a hold keep the closed error
`normal completion invocation refused or unresolved`. A `BaseException` outside the `Exception` class passes through as
before. The function makes no second call. Why: before this end existed, the invocation raised when a guardrail stopped
a model call of the replacement worker and the product held the changed history. The resume answered `resume_state`
`failed`, no report described the stop, and every later resume failed.

The held end applies to every hold that a backend callback sets inside a call that raises if no fence refuses.
The cause of the hold plays no part. One cause is a guardrail stop that changes the admitted history. Another
cause is the backend check before a model call. That check is the `require_input_open` callback in `build_agent`
in `tool_binding_backend.py`. The callback is registered when the `Agent` is built. It runs before `fence`.
The callback can refuse the research binding qualification when the warm state closes inside the call. On that
refusal, it sets the hold and raises. No fence runs. The model call that the check guards does not run. The run
ends with the held report. The hold line is
`AMLAS_INPUT_HOLD phase=before_model_qualification reason=callback_refused`. Without the held end, that resume
answers `failed`. The first worker returns the same held turn for that refusal in `_chat_with_tools_once`. A hold
seen after a normal return does not lead to the held end. `current()` refuses it with
`normal continuation lifetime or admission changed`. `tests/test_normal_resumed_run_held.py` has one control per
condition. It covers the backend refusal in a full run.

The held end runs `held_current()` twice. The first pass runs before `backend._input_hold_turn` builds the held turn.
The second runs after `append_recovery_report` adds the runtime recovery record. `held_current()` uses the checks in
`current()` with two terms left out. It skips the hold term because the hold is what leads to this path. It also skips
`_qualify_research_binding`. With no research binding, the helper returns at once and checks nothing. With a bound
research binding, it refuses whenever a hold is set. Skipping the helper also skips its checks for the process, warm
owner, graph identity, and research phase. The first worker returns its held turn at that same refusal. It uses
`_input_hold_turn` too.

Both passes still check the four lifetime identities: the backend, `Agent`, claim, and snapshot. They also check the
system prompt that carries the restoration notice. They keep `context.qualify_completion`, which checks approval, the
original job, the claimed token, the run record, and the snapshot. If either pass refuses, the function raises inside
its `try`. The hold stays, and the function returns nothing. The resume answers `failed` at stage `continuation`. It
stores no report.

The held turn takes `guardrail_assessment` from this invocation's capture. The assessment records what the guardrail
reports. It does not say what causes the hold. With a trace that names a blocked policy the held report carries it.
Without a trace, the report has no `guardrail_assessment`. A hold rules out an execution list, so the held turn has no
executions. Its report lists no tool and names no job. The text starts with the fixed held message. This sentence
follows, whether or not tools run:
`This hold does not undo earlier actions, and any jobs already submitted may still be running.`

`_agent_turn_response` builds the response. The runtime recovery record follows the message, as on the normal path.
The replacement worker stores the report under `resumed_report_checkpoint_ref` in the run record. The held end adds
no log line, no new response action, and no run-record field.

The normal path stays first in the function. Its statements stay in the same order.
`tests/test_recovery_report_record.py::test_order_inside_continue_normal_completion` checks that order. Keep changes
to the held end after the normal path in the source. See `tests/test_normal_resumed_run_held.py` for the tests.
