# src/amlas/evaluation

## Purpose

Store and compare experiment results, leaderboards and statistical evidence.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `judge.py:JudgeVerdict` — The structured result of an LLM-as-judge evaluation.
- `judge.py:Judge` — An agent-output judge — evaluate a (prompt, response) pair against criteria.
- `BedrockLLMJudge` resolves the shared reviewer model unless explicitly overridden.
  The separate reviewer default is GPT-6.1 Sol; execution agents use Sonnet 5.5.
  Filter sampling fields through the shared model compatibility helper and parse only
  visible text blocks; adaptive-thinking blocks are never evaluation output.
- `leaderboard.py:Leaderboard` — Provides ranking, comparison, and historical trend analysis over experiment results.
- `statistics.py:ModelComparator` — Statistical significance testing for model comparisons.
- `store.py:ExperimentResult` — Canonical schema for a single experiment run result.
- `store.py:EvaluationStore` — Thread-safe JSON-based experiment results store.
- `coding_executor.py:FileInvocationRecorder` holds one caller-owned empty run directory, writes bounded closed hash-only records to exclusive regular0600 files, and fsyncs each file and directory before returning. Partial files remain evidence on failure; names are never reused. This source sink grants no native containment, cleanup, oracle or quality authority.
- Owned files: `__init__.py`, `coding_executor.py`, `judge.py`, `leaderboard.py`, `statistics.py`, `store.py`.

- `coding_oracle_catalog.py` packs only fixed public hash metadata. Its inline
  secret-scan annotations cover the original public source digest and independently
  recomputed656 hash fields; exact-byte and annotation regressions reject altered or extra content.
  This annotation grants no oracle, containment or native admission.

- Fixed paired evaluation verifies source-owned official164 prompt/oracle hashes before any oracle exec. Candidate/bootstrap/value RPC code runs only in original CodeInterpreter sessions; trusted assertions use the fixed no-argument Rust oracle launcher, then fixed Python `-I -S` with cleared provider environment and isolated source copies. Source policy checks do not establish actual native confinement, credential absence or admission; kernel/loader/import/FD/parent-SDK qualification is required. A verdict is retained only after the original child exits normally with actual status0 and stdout EOF within the same remaining case deadline; signal/nonzero exit, trailing protocol bytes and post-verdict hangs refuse. The watchdog stays armed until terminal acceptance; failure/expiry kills and boundedly reaps only the original child. Tagged values preserve tuples and non-string dictionary keys; the trusted prompt namespace preserves fixed helper bindings. Source fake tests never execute either code path. The ten-second whole-case absolute deadline excludes allocation and interrupts bootstrap/transport/assertion waits on the fixed Linux main thread. An observation timer never calls the SDK. Original streams, Stop and terminal Get remain owned by one synchronous cleanup with a shared thirty-second cap. Ordinary close failures still attempt every unique original stream; at most one same-session uncertainty record follows Stop/Get while the original cap remains armed. Expiry or interrupted persistence retains the existing prefix as unresolved with no postdeadline write or extra allowance; no credential thread is abandoned and no cleanup lock race is used. Separate cleanup client uses the same IAM principal; no revocation-independent cleanup claim exists. All328 arm/case receipts keep the164 denominator. Report arithmetic, native containment and quality admission remain separate.
- The recorder accepts only the exact built-in `aws.codeinterpreter.v1` in addition to its original bounded custom identifiers; session/evaluation identifiers remain strict. Session/call records are fsynced before effects and retain hash-only data.

- `coding_pair.py:PairedTraceAccumulator` retains bounded closed hash-only primitives in a frozen internal binding/count proof. Completed oracle receipts require original allocation/session identity, contiguous channel indexes and bootstrap/value ordinals, response/event/completion order, and no cross-case token/session/request replay. Cleanup completion requires original Stop/Get terminal evidence; one exact final same-session cleanup-unknown after terminal evidence remains unresolved, and other follow-on records refuse. Valid durable prefixes need no forced postdeadline error write; producer cleanupFalse never upgrades from readable terminal bytes alone; unknown allocation/cleanup and timeout rows retain their denominator slots. This is trace consistency, never SDK acceptance, isolation or oracle attestation.

- `paired_runner_sha256` requires a trusted keyword-only oracle launcher digest and binds it with the fixed v2 domain and Python closure without reading an AMD64 file on the ARM caller. `verify_oracle_launcher` independently walks the fixed binary ancestors with no-follow held FDs, requires root-owned nonwritable executable single-link bytes and exact trusted startup digest; mismatch or unconfigured intent refuses before input authority. Computed hashes and returned digests alone grant no admission.

- A single fully framed original native result with exact bounded nonzero integer exit status and bounded stdout/stderr strings raises only private `CandidateExecutionError`; the parent records `runtime_error` with oracle_completedFalse. It grants no oracle failure/pass or isolation claim. Missing/zero/malformed status, ambiguous isError and transport/protocol uncertainty remain infrastructure errors; raw output is never retained in diagnostics.
- Cleanup may reconcile failed read-only Get observations on the same accepted session after one accepted Stop, under the unchanged30s/12poll budget. Every actual read has get-attempt then get-failed or get-observed; no Start/Stop replay or unknown-ID adoption. Closed optional typed SDK metadata includes only validated service code/HTTPstatus/requestID. Unknown allocation and unresolved cleanup remain unqualified; old records stay compatible.
- `coding_pair_runner.py`: `timeout` covers cases with no verdict by the case limit, including a remote call that fails to return in time or an answer that runs too long. Nothing is tried again.
- `coding_pair_runner.py` reads the limits it enforces from `BUDGETS`: the whole-case seconds for the deadline, its watchdog and the returned `case_wall_seconds`, the session timeout of the start request and the value call limit. The adapter holds an equal table and both sides hash it into `budgets_sha256`. The values are unchanged. The bound of 512 value calls also stands as a literal in `coding_oracle_child.py` and in the ordinal checks of `coding_executor.py`, and the job's client read timeout of 10 seconds is a literal in `sagemaker_entrypoints/coding_pair.py`; a changed table value does not move those.
- `coding_pair_runner.py:bootstrap_code` and `value_code` build the two remote texts. The load text binds `os` and `sys` only under the private names `_amlas_os` and `_amlas_sys`. Beyond `json` and `math` an answer finds no module under its own name that it did not import, as before: an answer that uses `os` or `sys` without importing it finds no such name and fails. While the answer loads and while its function and the encoding of its value run, `sys.stdout` points at the null device. Each text saves the stream it finds and puts it back before it prints the reply; the value text does so in a `finally`. The answer's source is embedded byte for byte. `_stdout` is unchanged: the reply must be the whole standard output, so output that bypasses the stream still gives `infrastructure_error`. A load that fails leaves the stream redirected; the case ends as `runtime_error` and no further call is sent in that session. Source tests parse the built texts and never run them; what the remote service does with them is measured by a live probe outside the source tests.
- `coding_pair.py:verify_pair_result` returns `result_statement` and `next_step` only for an evaluated pair. `pair_result_statement` builds fixed sentences from the verified summary: the pass counts, the cases without a verdict, the difference and its range, and the case limit read from `coding_pair_runner.BUDGETS`, the table whose hash the binding carries. `next_step` tells the agent to put those sentences into its final report unchanged. A result whose cleanup was not confirmed keeps status `cleanup_unresolved` and carries neither key. The sentences restate verified arithmetic; they grant no quality or native admission.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.

- `coding_executor.py:invoke_recorded` records one original CodeInterpreter request through a trusted synchronously durable host sink: hash-only attempt precedes invoke, response identity precedes bounded incremental events, and completion is transport data only. Native/session/stream/persistence/cancellation failures preserve the original durable prefix and re-raise without retry. An uncertainty record is attempted only before the original deadline and never after the owned signal sentinel; a missing terminal record remains unresolved, never completed. Remaining-budget uncertainty persistence propagates owned deadline and unrelated BaseException cancellation; ordinary sink errors retain the original failure. The deadline and remaining64-event allowance are checked before each stream read; reaching64 retains those records and refuses an additional read rather than assuming end-of-stream. A Linux main-thread SIGALRM guard interrupts blocking invoke/iterator waits using a private cancellation sentinel outside SDK retry handlers. Expiry starts no new uncertainty write or reporting allowance. The existing durable attempt/response/event prefix retains original identity, and available original streams still reach the unchanged parent cleanup cap; interrupted persistence remains unresolved. Unsupported threads/platforms, blocked SIGALRM and unrelated active alarms refuse before SDK; owned nested guards never extend the original deadline, and prior handler/timer state is restored on success, timeout and cancellation. Owned one-shot interruption during arming/disarming/handler restoration is translated only after restoration; unrelated KeyboardInterrupt/SystemExit retains its original exception. A trusted parent retains original streams for its shared cleanup cap; standalone invocation closes available original streams once under a separate thirty-second cap without claiming cleanup success. Bounded plain data is checked before retention. This seam grants no caller policy, containment, oracle pass or quality authority; the parent owns original-session cleanup.
