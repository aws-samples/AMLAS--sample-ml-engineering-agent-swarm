# Initializer source

## Purpose

Provide the fixed Linux initializer executable.

## Ownership

Inherits ../AGENTS.md.

## Local Contracts

- Production guards use explicit errors, never assertions.
- Processing output initialization takes the existing compiled WorkerProfile;
  CaseFreeze and CodingPair selects root0:0/0755 parent, never a caller owner selector.
  All other Processing profiles keep USER:USER/0755; Qwen's separate model path
  remains unchanged. Strict output tuples, empty/inode/name/lock/FD-only mutation
  and permanent-drop checks are unchanged. Case image output is worker0700.
- The binary performs only fixed OS operations before launching a non-root child.
- Native build metadata validation precedes output mutation. Fixed ceiling and
  exact retained mask, fs IDs, known securebits and every syscall failure guard
  supplement permanent IDs/EPIA0/NNP1. Strict default behavior stays separate.
- Native startup stdout is bounded fixed observed metadata; it is not attestation.

- `WorkerProfile::Preparation` chooses fixed source paths only in native builds.
  Exactly twelve declared pairs are accepted; inspection-only target-column and
  arbitrary entrypoint/output/policy arguments refuse before output mutation.
  The fixed root-owned `preparation_profiles` package and its four declared
  modules are verified before privilege transition and Python imports.
- `WorkerProfile::ModelAcquisition` likewise selects fixed source paths and eight
  primitive pairs only in native builds. Public-fetch controls are distinct from
  provider acceptance: the model profile refuses credentials and proxy hints
  before environment clearing while retaining the existing dataset behavior.
- `WorkerProfile::QwenSft` admits only fixed train/reload and fit/full-v1 recipes,
  version1 and bounded canonical manifest count/SHA. It uses a separate fixed
  `/opt/ml/model` empty-target routine, retaining original FD/inode/lease authority;
  Processing parent/output contracts remain unchanged.

- `model_file_boundary.rs` owns only fixed provider absence capture, four-seal
  readonly memfd and raw-model ABI6 filesystem restriction. Public APIs have no
  path/profile/environment/CLI selectors; injected observations/syscalls compile
  only in test facades. Returned finite inode metadata is not a grant.
- Original O_PATH/no-follow ancestors, named inode/mode/owner and known alias target
  checks remain held/rechecked around policy application. Any unsupported ABI,
  unknown link, missing required runtime path or failed syscall refuses.

- Proc qualification distinguishes the actual post-drop/preexec getter0 owner
  state from postexec state. `process_dumpability` is observed0/1 only, rechecked;
  no setter or caller selector changes it. PID directory and final leaf owners
  are qualified separately while original same-PID inodes stay held.
- Current flock acquisition/renewal on the private original supplied output OFD
  is not an assertion of historical uninterrupted lease continuity.

- `model_fd_handoff.rs` preserves both same-OFD authorities across fixed3/4
  collisions. Consuming owners drop before target overwrite; safe CLOEXEC copies
  precede any dup2, then close before unrelated raw-FD cleanup. Current lease is
  retained, not historical continuity attested. Unsupported cleanup refuses.

- `model_tcp_boundary.rs` separately stacks only fixed raw-model TCP port rules.
  `apply_model_tcp_boundary(&NativeState)` requires the initializer's same freshly
  measured native-final state/verified original bounding mask. Reuse the existing
  final validator, require one original thread and independently READ NNP1 before
  mutation. LinuxAMD64 ABI>=4 is mandatory; no downgrade or best-effort fallback.
  Exactly16-byte ruleset attrs handle network mask3/filesystem0; NET_PORT2 rules
  allow only CONNECT_TCP2 for host-endian u64 ports53 and443. No BIND rule exists.
  Rule FD ownership closes on every error and success; no FD/port/path/env selector.
  Five-field `raw_model_tcp_ports_v1` proof is descriptive ABI/mask/connect/bind
  metadata, never native admission/attestation. IPs/private443/UDP and existing
  connections are unfiltered; even newer ABIs do not gain an implicit UDP policy.
  Original filesystem policy and credential interpretation remain separate.
  Injection facade is private cfg(test) only; failed state/read/create/add/restrict
  steps return bounded errors before caller child execution.

- `qwen_tcp_boundary.rs` is fixed SFT-only TCP bind/connect denial, with no
  NET_PORT/add-rule syscall or caller/env/path/port selectors. Fresh native-final
  metadata, one thread and independent NNP1 read precede ABI/create/restrict.
  Main installs it after FD3/4 handoff, requires fixed numeric endpoint permission
  denials and emits closed metadata before Python exec. Unsupported kernel or
  any unexpected endpoint outcome refuses; no downloader behavior changes.

- `oracle_boundary.rs` and `oracle_main.rs` are a separately compiled fixed oracle-child slice. The fixed no-argument unprivileged launcher sets/verifies NNP, requires UID/GID slots10001/groups[10001]/EPIA0/single-thread, records bounding without widening it, verifies fixed root-owned source bytes/runtime leaf identities, stacks Landlock ABI>=6/mask0xffff plus signal/abstract-UNIX scope3 and immutable default-deny AMD64/x32-checked seccomp. Three isolated source copies and reviewed runtime leaves receive only fixed read/actual-exec rights; reviewed import directory READ_DIR exposes names/metadata (including descendants), not unknown file content. All preparation descriptors drop before mandatory close_range; only correctly directed pipes and devnull stdio reach fixed Python-I-S. Native image/provider/startup/import/denial/parent-SDK qualification remains separate; no privilege/group fallback or native admission exists from source tests.

- Original catalog-pinned164 prompt/test imports justify exactly four additional CPython3.11.15 read-only leaves: `typing.py`, `copy.py`, `weakref.py` and `string.py`. Exact image bytes match the reviewed source inventory; existing ancestor/import-directory rules remain fixed. `_typing` receives no grant because the pinned typing implementation handles its ImportError internally. No new directory, execute, syscall or network right is introduced. Official-case compatibility and actual host confinement remain account-only native gates.

- Reviewed follow-up permits AMD64 getdents64(217), not legacy getdents(78); prlimit64(302) permits only full64-bit pid0/new_limitNULL self queries. Every PRCTL call supplies all four machine-width argument slots. Actual ELF-interpreter leaf alone gains READ_FILE+EXECUTE; fixed loader aliases retain original parent/link/directory IDs and exact link bytes, resolve to the same canonical granted inode and recheck before/after Landlock. Symlink0777 is normal; nonwritable root-owned parents protect it. Source flags explicitly include-I-S-B-u; no bytecache, extra process or privilege fallback.

- Compiled CodingPair shares only CaseFreeze's root-owned fixed Processing parent. Three closed binding primitives, source/launcher consistency, pair-only native relative role metadata URI and fixed venv child with exactly root-owned `/opt/venv/bin/python → /usr/local/bin/python → python3 → python3.11` aliases and immutable no-follow parents/real interpreter preserve permanent native identity with pair-only existing oracle group[10001], active-capability zero, NNP and original FD3. Other provider/proxy policies and model paths are unchanged.

## Work Guidance

Raw-model main installs the fixed TCP policy and verifies two HTTP metadata
endpoint denials before fixed Python exec. A deferred relative hint is never
forwarded and never grants authority. Unsupported kernel/enforcement refuses.

Fixed Qwen startup captures the same original provider inventory before dropping
privileges, then uses the sealed output/inventory handoff. It preserves its fixed
train/reload CLI and does not apply the downloader Landlock policy to CUDA.

Keep FFI limited to required Linux syscalls and preserve descriptor identity checks.

## Verification

Run the parent locked Rust and controlled image tests.

## Child DOX Index

None.

- Fixed SFT scratch creation is postdrop and outside checkpoint output; original
  component FDs retain identity through validation, and cache/leaf creation is
  limited to source names/0700. Unknown metadata or nonempty leaf refuses.
