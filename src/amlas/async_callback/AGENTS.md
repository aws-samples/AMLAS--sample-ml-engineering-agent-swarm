# src/amlas/async_callback

## Purpose

Translate managed training events into callback/job-state handling and optional runtime resumption.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Callback scope is synchronous owning-thread/PID only and non-propagatable.
  Shared scope object is revoked before ContextVar reset; retained/copied contexts
  after exit or across threads deny before client/Get/invoke. A fresh worker scope
  is independently required; no shared concurrent clock/grant authority.
- Cleanup owns a returned closeable original before any clock/qualifier/interface
  failure. Post-Invoke qualification precedes body reads; all denial/error paths
  attempt one original close. Elapsed deadline is checked after successful close;
  an overrun remains outcome_unknown, never hard preemption/cancellation proof.
  Both selected transports require finite exact connect<=5/read<=10 metadata;
  metadata/source fixtures do not prove native socket behavior/authentication.
- Consumed source/detail-type use exact strings; each resource is bounded text.
  Runtime client setup failure after resolved lookup returns setup_unavailable with
  original IDs and zero invoke, not an attempted delivery status. Actual isolated
  package import excludes optional application dependencies; module stays stdlib
  plus provided boto3/botocore. Native permissions/ingress remain separate gates.

- Opt-in server-only AMLAS_CAMPAIGN_CALLBACK_MODE=1 selects the closed campaign
  branch; missing mode preserves the legacy Training status/update/default route.
  Campaign mode requires explicit account/Region/deployment/table/runtime ARN,
  qualifier/version/route SHA and fresh private current-route/role qualifier.
  Missing/expired/inherited/drifted qualification denies before Get/invoke. Types,
  injected clients/env/digests/Lambda context are conditional data, not native auth.
- Mode1 events consume bounded Processing/Training source/id/account/Region,
  native name/ARN/terminal status/resources only; provider extras are data. It first
  performs one consistent PROMPT_WAKE_JOB#kind#name lookup in the selected table.
  Only confirmed absence permits the existing CAMPAIGN_WAKE_JOB#kind#name lookup;
  each namespace gets at most one Get, with no Scan/query/status mutation/Create.
  A present malformed, conflicting or unresolved normal record never falls through.
- Normal routing uses a closed twelve-field schema1 normal_prompt_wake_job_v1
  body in the exact job_name/record/record_sha256 envelope. Canonical UTF8 JSON is
  bounded to8192bytes; duplicate/nonfinite/extra/untyped fields, wrong raw SHA and
  changed native identity refuse. A freshly qualified DescribeTable must match the
  selected active table ARN/name, job_name string HASH key and recorded incarnation.
  The record is DATA only; existing route/role/current qualifications remain required.
- Normal delivery selects only the configured runtime ARN/qualifier and the original
  SDK transport session, with exactly resume_prompt_run and the original run_id.
  It reuses single-attempt bounded invoke/drain/close and current exposure checks;
  delivery labels grant no SDK restoration, native permission or lifecycle/quality.
  The standalone handler remains stdlib plus supplied boto3/botocore. The default mode1 native qualifier
  installer remains conditional on scoped metadata-read permissions and verified
  deployment. Source construction establishes no effective native authorization,
  TASK readiness, replacement lease, later-stage routing or durable final-report
  acceptance; these remain separately qualified O4 integrations.
- Campaign hints retain their eleven-field canonical DATA and existing nine-field
  payload; the receiver independently verifies protected original C+H/intent/op.
- Campaign payload contains only the existing nine WakeHint fields; status never
  authority. Selected configured runtime qualifier and original transport are
  immutable routing inputs, not caller proof. Duplicate/reordered events can be
  delivered as hints without stage advance or claimed model completion.
- Single-attempt invoke uses bounded configured connect/read timeout, 64KiB chunks,
  1MiB total and 20s measured drain deadline; original stream always closes. Campaign
  wakes discard the body; normal resume deliveries read only the first closed event
  (see the acknowledgement gate below, added 2026-11-07). Unknown
  response/retry/timeout/close/qualification after attempted delivery retains original
  identifiers and honest outcome_unknown; no retry/cancellation/noncommit inference.
  Delivery labels never establish native lifecycle/quality. Native blocking/close
  timing and role/cohort/endpoint admission remain separate Root qualification.

- An event callback is not evidence of end-to-end runtime resumption; target binding and runtime input shape remain separate contracts.
- Retain job-state correlation and propagate meaningful service failures without credential/payload logging.
- `lambda_handler.py:build_resume_payload` — The utf-8 JSON payload sent to ``invoke_agent_runtime`` (pure).
- `lambda_handler.py:build_invoke_kwargs` — The exact kwargs for ``bedrock-agentcore.invoke_agent_runtime`` (pure).
- Owned files: `__init__.py`, `lambda_handler.py`.

- Mode1 installs `_normal_callback_bootstrap` only when no existing private
  `_CallbackServices` scope is present. The installer enters the existing
  qualifier-required `campaign_callback_scope`; it never changes that scope's
  public signature or treats event/env/client DATA as native authentication.
  An unavailable native provider fails closed before lookup/invoke. Explicit
  private scopes, legacy mode and readiness-stream failure/retry behavior remain.
  The approved route digest is selected opaque DATA; no substitute self-hash is
  authorization. Native metadata-read permissions and actual current-role/route
  qualification remain deployment gates, not source-test claims.

- The default native installer reads private server-only
  `AMLAS_CALLBACK_ROLE_ARN` and `AMLAS_CALLBACK_ROLE_ID` bindings. It uses fresh
  STS assumed-role account/ARN/RoleId identity and exact READY endpoint/liveVersion
  plus selected-version runtime metadata on every qualification, bracketed by
  unchanged config/role selectors, owning PID/thread and finite monotonic clock.
  Metadata clients retain one attempt and connect<=5/read<=10. Unknown metadata,
  role/route drift, copied/closed providers or absent bindings deny before the
  callback's DDB/runtime effects. Missing native metadata-read permissions are
  deployment failures, never permission inferred from source fixture green.
- Unscoped mode1 performs the existing pure event identity/config checks or
  stream4MiB/depth/records1..16/candidate checks before native provider/client
  construction. Malformed readiness streams retain the fixed RuntimeError retry
  contract; existing private scoped paths retain their current fences.

- The bootstrap owns only the STS/control SDK clients it creates, registers each
  returned unique client immediately, and attempts one close on success, dispatch
  failure and partial setup/qualifier-construction failure. Scope reset and native
  qualifier invalidation precede cleanup. Ordinary client-close exceptions are
  suppressed without logging so they cannot replace the original setup/dispatch
  result; this records an attempt, not successful socket cleanup. The qualifier's
  own `close` only invalidates it and never closes externally injected clients.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.

The qualified normal callback retains one immutable canonical terminal hint before
runtime delivery. The hint is bounded to 8192 UTF-8 bytes and binds the original
run, native identity/status, operation digest, scope and exact normal-route digest.
One conditional Put and exact consistent full-item readback are bracketed by fresh
route/table qualifications and exact normal-route rereads; denial, conflict,
unconfirmed outcome or drift prevents invocation. Terminal hints grant no TASK
acceptance, consumption, lease, readiness or permission. Campaign fallback remains
read-only when the normal route is confirmed absent; no stream or activation path
is established by this contract.

Normal delivery uses the exact current-route/table qualifier for the post-invoke
drain and final exposure gate. Post-response drift closes the outcome without
claiming delivery success or retrying an already attempted invocation.

Mode-1 readiness stream handling selects only TASK descriptor or exact original
journal revision/digest transitions from bounded NEW_AND_OLD_IMAGES DATA. Journal
private_record stays opaque. Every current-row read and delivery gate freshly
verifies the selected active table/PK/incarnation, enabled NEW_AND_OLD_IMAGES and
exact native LatestStreamArn against eventSourceARN. Current producer, normal route
and immutable terminal hint must retain exact linkage and bytes; readiness delivery
uses the original session/resume selector and existing guarded runtime tail without
another hint write. Images grant no acceptance, readiness, lease or execution rights;
runtime owns those checks. This Source contract establishes no stream deployment,
native qualification bootstrap, durable consumption or lifecycle success.

A PROMPT_RUN row at stage enrolled (written before any SFT association) is not
readiness: identity is still verified, then the row is ignored without native I/O.
Any other non-sft stage still refuses. A refused enrollment row would hold an
unlimited-retry stream mapping behind every new run (added 2026-11-05).

The complete stream event, including both images and JSON escaping, is bounded to
4 MiB before client setup or native I/O. Only finite plain JSON DATA within the
fixed depth bound is accepted; private binary content is never decoded or logged.

A TASK transition selects the exact descriptor and canonical immutable original
linkage, allowing only unrelated history/continuation/claim/report bookkeeping to
advance before delivery. The freshly qualified complete current producer row is
then pinned around every later I/O and delivery check. Journal revision matching
remains exact; a newer journal revision is held for its corresponding event.

Report pointers are bookkeeping of that linkage (added 2026-11-07): `_normal_stream_linkage` leaves out
`report_checkpoint_ref` (the original worker's report) and `resumed_report_checkpoint_ref` (a replacement's final
report) with history, continuation, claim and report. Before this, the exclusion list named a key `report` that no
run row carries, so an original worker that ended on its own and retained its report between the TASK event and a
retried delivery made every later delivery of that record refuse (`current normal TASK original linkage differs`),
and with an unlimited-retry mapping the record held its shard for good. TASK descriptor, association, scope,
session and correlation stay in the linkage. A change to this file needs a new stream function version.

- Native DynamoDB stream entry fails the invocation with a fixed, data-free RuntimeError when delivery is unresolved. Only closed 200 delivered/ignored results acknowledge the batch. Whole-batch retries require the receiver to retain original-operation and duplicate-effect fences; do not treat HTTP-style refusal objects as native retry evidence. Partial-batch response settings remain deferred.

## Normal wake acknowledgement gate (added 2026-11-07)

A normal wake is acknowledged (`delivered`, HTTP 200 equivalent) only by the runtime's positive closed
first event: `{"event":"resume_state","state":"continuing"|"completed","run_id":<this delivery's run>}`.
`restoring`, `original_active`, `failed_transient`, `failed`, a timeout, an absent, oversized or malformed
body and any other event are `outcome_unknown` (202), so the stream handler raises and the event source
mapping retries the same record (MaximumRetryAttempts -1). This replaces the earlier rule that the
response body is discarded untrusted data for normal deliveries: the runtime is our own code behind IAM,
so the body can only cause retries unless it carries exactly this event, and a body that merely claims
success outside the closed schema is refused. A permanent runtime failure therefore keeps retrying; the
failure is visible as the runtime's `failed` answer and its restoration warning, and no wake is ever
acknowledged on a busy original or a failed restore. `_drain(..., resume_run_id=...)` reads the stream one
byte at a time (a read of n bytes on the open stream blocks until n bytes arrive; the runtime keeps it open
after its events), stops at the first complete line, never reads past it, accepts `data: <json>` SSE
framing or a bare JSON line, skips blank and `:` comment lines, rejects duplicate keys and non-finite
numbers, and reads at most 64 KiB inside the 20 s deadline, which now starts after the post-Invoke
qualification so a slow metadata re-check cannot consume the body-read budget. The campaign wake keeps the
discard-the-body contract. The stream function is a separate artifact: its zip carries the single changed
member `amlas/async_callback/lambda_handler.py` and a new immutable version plus mapping update are needed
for any change to this file.

## Second delivery of one state change waits (added 2026-11-10)

SageMaker sends two terminal events for each Completed or Failed training job (one for Stopped). On the
deployed EventBridge target their deliveries started 6 to 878 ms apart (20 real jobs), and one delivery
makes 194 sequential remote calls before `InvokeAgentRuntime`, so both reach the runtime within about a
second of each other. After a planned worker stop the session has no microVM, and two near-simultaneous
first calls on such a session have created two replacement microVMs that each restored the run (observed in one run).
`_retain_normal_terminal_hint` therefore reports whether this delivery's conditional Put created the hint.
A delivery that confirmed a hint it did not create (conditional failure, or an applied write whose answer
was lost) calls `_twin_pause` for `_TWIN_RESUME_DELAY_SECONDS` (40 s) before it resumes, capped at the
invocation's remaining Lambda time minus `_TWIN_RESUME_RESERVE_SECONDS` (50 s); without a Lambda context
(direct calls, tests) it does not wait. The delayed delivery still resumes, so it keeps its second
chance; it reaches the session after the first call, and the runtime answers it with its closed busy
state. The readiness stream path does not wait. This narrows the race for the event pair only: a
readiness retry that lands in the binding window, or a first call delayed past 40 s, can still meet a
second first call. Tests: `tests/test_callback_real_event_shapes.py`.
