# Fixed oracle native controls

## Purpose

Prepare and run bounded account-only native regression fixtures against unchanged
fixed oracle policy bytes. These controls provide preexec fixture evidence.

## Ownership

Inherits ../AGENTS.md. This subtree owns only test helpers, source preparation,
and receipt auditing. Product policy, Python sources, runners and infrastructure
retain their existing owners and bytes.

## Local Contracts

- `prepare.py` refuses source-pin drift and existing output before writing. Its
  scratch composite is the exact production Rust prefix plus `fixture_suffix.rs`;
  unchanged three Python inputs retain their exact relative include paths.
  `manifest.json` freezes qualified source/image/launcher, source bytes, helper,
  preparer, runner and composite hashes. None grants native admission.
- Exact reviewed source/launcher/helper integrity pins have line-scoped secret
  scanner annotations. Their complete values and source checks remain unchanged;
  no file-wide exemption or scanner/threshold change is permitted.
- Input schema v2 calls the unpublished image pin `qualified_image_config_id`.
  `run.py --production-image` accepts only that full local sha256 config ID;
  registry-digest syntax and mutable tags refuse. Docker image inspection must
  independently match ID/linux/amd64, image User10001:10001, source revision label
  and exactly one matching `AMLAS_CODING_PAIR_SOURCE_COMMIT`.
- `build_with_local_base` is a build-only helper. FROM uses a task-owned local tag
  derived from the fixed helper SHA and complete inspected config ID; a config ID
  is never treated as a registry digest or repository name. Existing conflicting
  or inaccessible tags refuse before mutation; a newly bound tag receives an
  exact ID/source/platform/user/name readback immediately before and after build.
  Default context must resolve to a local Unix Docker socket; the explicit default
  Buildx builder must have the docker driver and one running local node. Daemon ID
  is compared before/after. Remote transport overrides refuse. Build uses
  `--builder default --load --pull=false --network none`; no registry push, login,
  token request, builder creation or automatic legacy fallback is implemented.
- Root must control local tag writers during this bounded build. Tag binding and
  before/after inspection are not atomic compare-and-swap; temporary substitution
  cannot be excluded by those observations. The build receipt explicitly records
  `atomic_tag_binding=false`, `trusted_local_writers_assumed=true` and that race
  limit. Actual builder compatibility requires an account-only build receipt;
  neutral helper tests do not establish native transport or isolation.
- `fixture_suffix.rs` is only a `cfg(test)` same-module suffix. Compile with
  `rustc --cfg test`, never `--test`: its standalone subject has one thread.
  Three fixed compile modes are baseline, complete and landlock. No product
  selector, source replacement, privilege fallback or trace-grown rule exists.
- Every compilation and native invocation runs only on owned EC2 in the AWS
  account. Developer hosts prepare source only. Root verifies current STS/account/EC2
  identity and writes an external proof with `execution_environment` equal to
  `owned-aws-ec2` and `same_account_targets` true; the runner records its hash,
  without independently proving that external assertion.
- `run.py` requires baseline completion before complete/landlock controls. One
  attempt per step, 180-second compile/build bounds, 25-second native bound,
  15-second subject alarm and 512-KiB output bound. Unsupported ABI<6, failed
  baseline, missing/duplicate/failed controls or incomplete original-PID reap
  refuse. The unconfined supervisor owns its subject's original PID; no scan,
  PID reuse cleanup, retry or unrelated target exists.
- Native containers use create/inspect/start without `--rm`. The CID file and
  Docker create output must agree on the original immutable full container ID.
  Before start, inspection requires the expected image and user, no added groups,
  networknone/NNP/capdropALL and private PID/IPC with no binds/devices/mounts.
  Actual subject state still requires exact groups[10001]; no group repair occurs.
  Cleanup inspects this original CID before killing a running container, verifies
  stopped/PID0 before removal and verifies exact-ID absence afterwards. Missing,
  substituted or inaccessible IDs refuse without a cleanup-success claim.
- Only fixed canary content, owned same-UID peer memory, loopback listeners and a
  fixed abstract UNIX name are used. Native containers are network-none,
  capability-free, NNP1 and UID/GID10001/groups[10001]. The derived disposable
  fixture is writable for the identical baseline; product files remain unchanged.
  Outer Docker seccomp is unconfined only for this regression fixture so the exact
  installed production filter can be attributed. No provider request occurs.
- Complete mode uses actual private state/validation/stdio/fixed-plan/recheck/
  loader/Landlock/seccomp functions and mandatory descriptor cleanup in production
  order. Original low/high FD read controls distinguish retained handles from
  closure. Landlock-only observations remain separately labelled: UDP and opened
  FDs remain usable, while complete mode additionally denies their syscall/FD path.
- `CONTROL` rows, one `FIXTURE ... PASS` row and `receipt.json` require complete
  mode-specific inventory, zero exit status and original-PID reap. Failed partial
  logs are retained; no admission marker is generated.
- No Python exec, live metadata/provider availability, official-case grading,
  Processing, model quality, ptrace/pidfd/socketpair or alternate-ABI qualification
  is claimed. Success does not qualify production end-to-end isolation.

## Work Guidance

Inside the owned AWS workspace, run preparation unit tests first. Then invoke
`prepare.py --root <repository> --output <fresh-scratch>`. Invoke
`run.py --prepared <fresh-scratch> --production-image <qualified-local-config-id>
--output <fresh-evidence> --account-proof <verified-proof.json>`.

Do not run these commands on developer hosts. Require the pinned compiler and qualified
image to exist locally; there is no pull fallback. Preserve evidence, then use
the owning safe-delete workflow for scratch. Containers have scoped original-ID
cleanup; use Docker's scoped image cleanup for the owned derived image.
Execution/review ownership remains with the root agent.

For the reviewed build-only phase, Root can call `build_with_local_base` directly
with its verified account proof, fixed image/source/helper pins, generated fixture
Dockerfile text and owned prepared/evidence paths. It performs no fixture
invocation. Do not call `run.py` until the separate native phase is authorized.

## Verification

On AWS only: `python3 -m unittest discover -s
processing_bootstrap/tests/oracle_native -p 'test_*.py' -v`, native `run.py`, and
the repository's recursive DOX lint. Passing source tests establishes only
preparation/audit behavior; compile and kernel receipts are independent gates.

## Child DOX Index

None.
