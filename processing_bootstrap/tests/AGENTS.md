# Initializer policy and descriptor tests

## Purpose

Test fail-closed bounding capability decisions and real Linux descriptor/name identity changes without invoking privilege initialization.

## Ownership

Inherits ../AGENTS.md.

## Local Contracts

- Bounding decision tests require no DROP for absent bits, successful DROP plus zero
  readback for present bits, and refusal of syscall/nonboolean/range failures.
- Native metadata validators reject ceiling/root/groups/thread/PRCTL/securebit/
  fs-ID/NNP/keepcaps/seccomp drift, require final exact initial bounding mask,
  and preserve locked-off known flag admission without ignoring setter failures.
- Test files remain in the configured temporary workspace; no AWS or external network calls.
  Fixed TCP boundary probes use only loopback in an isolated network-none container.

- `preparation_cli.rs` executes the actual helper in the compiler container: native
  exact primitive pairs reach the following trusted-path gate; code/path/grant flags
  refuse. The compiler fixture has no worker image and executes no data workload.
  Default strict executable continues to reject preparation entirely.
- `model_acquisition_cli.rs` exercises the actual executable's fixed eight-pair
  admission, unknown/path/grant/duplicate/control-character refusal, and provider/
  proxy refusal before environment clearing. Strict builds reject the profile.
  These compiler checks do not execute a downloader or establish native admission.
- `qwen_sft_cli.rs` exercises six fixed training pairs, closed recipes/operations/
  manifest bounds and provider refusal before clearing; strict builds reject it.
  Compiler admission is not a native mount, GPU, training or reload proof.

- `model_file_boundary.rs` tests actual kernel memfd seals/read-only OFD and
  write/resize/shared-write-mmap denial; fixed absence versus unknown/presence;
  original ancestor/alias substitution and mode drift; exact ABI6 mask/syscall
  order/refusal and output rights. Test-only syscall fakes do not establish native
  enforcement. Retained fixture names are unique across container PID namespaces.

- Advisory follow-up covers all28 production-assembled nodes/rights, original
  OFD authority, immutable runtime/devnull/output metadata, allfive alias targets
  and real denied provider traversal. A matched static post-drop/preexec oracle
  supplies the proc-owner refusal case; dumpability0/1/unknown states stay distinct.
  Exact policy mutation oracles do not claim native enforcement.

- `model_fd_handoff.rs` runs actual descriptor-table cases only in isolated
  executable children: swapped/fixed/colliding/high source FDs, typed producer,
  readonly fourseals/count/identity/currentlease, invalid owners/modes/stdio,
  unrelated low/high-FD closure and live standard streams. Parent harness file
  ownership is never closed by the tested helper. Fixture green is not native admission.

- case_freeze_cli.rs checks actual executable fixed three-pair admission, code/path/grant/duplicate refusal and strict-build denial; compiler checks execute no data worker.

- `oracle_boundary` unit modules exercise only typed state, argv, environment, catalog and an in-memory BPF evaluator. They execute no real filter or denial syscall. Rust compilation/run remains on the account builder; actual startup/FD/import/network/proc/peer-process denial cases are AWS-only and grant no admission until measured.

- Oracle follow-up tests use only a typed in-memory seccomp_data/BPF evaluator: getdents64 positive, legacy/unknown refusal, full64-bit prlimit pid/new_limit halves, loader read-execute mask and exact known link bytes. No peer resource-limit or loader mutation syscall executes on developer hosts; actual checks stay in AWS.

- `case_output_parent/cases.rs` uses isolated actual directory/FD child fixtures.
  CaseFreeze root parent succeeds while USER parent refuses; legacy Processing
  profiles retain the opposite exact ownership predicate. Original parent/output
  inodes, no parent repair and private0700 final output are required. These account
  compiler tests do not qualify SageMaker mounts or native case publication.

## Work Guidance

Use real directories and retained open descriptors for race guards.

## Verification

Run cargo test --workspace --locked.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `model_tcp_boundary/AGENTS.md` | Fixed TCP boundary syscall/kernel oracles |
| `qwen_tcp_boundary/AGENTS.md` | SFT-only TCP denial/state/descriptor controls |
| `qwen_scratch/AGENTS.md` | Fixed postdrop scratch component controls |
| `case_output_parent/AGENTS.md` | Compiled CASE-only Processing parent and original output FD controls |
| `oracle_native/AGENTS.md` | Account-only unchanged-policy native controls |

- `qwen_tcp_startup.rs` checks fixed SFT policy/denial metadata ordering after
  original FD3/4 handoff and before Python; default/native CLI and all existing
  descriptor guards remain required. Host ABI failure is unsupported/unqualified,
  never a positive enforcement/GPU result.

- `qwen_scratch_startup.rs` fixes TMPDIR separation and postdrop ordering.
  Private `qwen_scratch/cases.rs` exercises actual dropped UID10001 original FDs,
  absent/valid/private versus symlink/owner/mode/type/nonempty refusal.
  These controls do not qualify SageMaker mounts or explain a prior native failure.
