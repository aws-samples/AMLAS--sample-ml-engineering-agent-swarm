# Processing initialization

## Purpose

Own the fixed OS-only privilege boundary before managed data workers.

## Ownership

Inherits the repository root contract. Existing workload code remains Python; security-sensitive initialization is static Rust.

## Local Contracts

- The crate has no external dependencies and targets locked AMD64 musl.
- Only compiled fixed workload profiles select fixed child code; no caller selects paths or privilege operations.
- No data, ML, agent or SDK code runs before verified privilege drop.
- `/usr/local/bin/amlas-processing-init acquisition|inspection` forwards only the
  profile's existing primitive worker pairs after dropping privilege. No caller
  chooses an initializer path, child code, environment or output path.
- Startup is one immutable initial process with no untrusted concurrent writer.
  Walk fixed parents with no-follow FDs. Compiled CaseFreeze and CodingPair requires
  Processing parent0:0/0755; other Processing profiles retain10001:10001/0755.
  and accept only an empty real output0:0/0755 or10001:10001/0700. Lock/recheck inode
  identity; change only that FD to10001:10001/0700. Never repair parents, create,
  rename, recurse or follow links. Failed operations stop before workload code.
- The default `zero_bounding_v1` build retains the original strict algorithm.
  Query each valid Linux bounding capability with PR_CAPBSET_READ. Skip DROP only
  for a verified absent bit; present bits require successful DROP and zero READ
  readback. Failed/nonboolean reads, failed drops and unverifiable readback refuse.
  This supports an already-empty bounding set without CAP_SETPCAP while preserving
  refusal of present undroppable bits and the complete final zero-capability guard.
- Explicit Cargo feature `sagemaker-native-v1` implements only the fixed measured
  `sagemaker_native_ceiling_v1` profile. Before output mutation it validates one
  thread, root ID triples, groups[] or[0], E/P/Bnd subsets0x200404fb, I/A0,
  no SETPCAP, NNP0or1, Seccomp2 and known securebits0..255 with KEEP_CAPS and
  NO_SETUID_FIXUP behavior bits off. Every valid PR_CAPBSET_READ must be boolean
  and agree with the observed mask. This profile makes no bounding DROP call.
- Native transition explicitly sets/verifies NNP1, disables/verifies keepcaps,
  clears ambient and all supplementary groups except the compiled pair broker’s exact[10001], sets all GID then UID slots10001 and capsets E/P/I0.
  Every setter failure refuses, including locked-off flags. Final bounding equals
  the exact verified initial mask; res/fs IDs10001, groups[] for data workers and exact[10001] for pair, E/P/I/A0,
  NNP1/keepcaps0/Seccomp2 and known safe securebits precede fixed Python exec.
  A bounded native startup proof records actual masks and metadata, never a grant.
- `processing_identity.py` is a standalone root-owned OS-only child verifier.
  Explicit no-argument verify_native_identity returns14 observed fields (existing9
  plus keepcaps/seccomp/fs_uid/fs_gid/securebits); verify_zero_identity preserves
  strict9. No CLI/environment/profile parameter chooses or widens the profile.
  `acquisition_worker_native.py` is the image's fixed native source wrapper;
  the original strict wrapper remains available for its original profile.
- Clear supplementary groups, real/effective/saved IDs, every capability set and
  bounding/ambient capabilities in the strict profile; native retains its verified
  ceiling. Verify UID/GID10001, groups[], active capabilities0 and
  NoNewPrivs1, retain only the now-nonprivileged original output FD3 and its same-OFD flock
  lease until worker exit, close/verify every other privileged/inherited FD,
  then exec fixed root-owned Python-I
  code from `/app`. Worker guards repeat the checks before fetch/parse imports.
- Required CLI source commit must equal the validated image environment commit
  before any filesystem/core work; this consistency check is not independent image
  attestation. No descriptor or ready flag is accepted from environment or CLI.
- Child environment has fixed PATH/HOME/TMPDIR/workspace/locale/Python flags plus
  full image source commit and acquisition's declared remote-fetch/host policy.
  Inspection rejects pinned-SDK credential/profile/config/provider/metadata hints
  and all HTTP/HTTPS/ALL/NO proxy variable spellings before clearing; clearing
  never establishes credential absence. Normal SDK and
  native credential/network admission remain mandatory.
- Reject loader/startup Python hooks. The pinned Python image's exact VERSION and
  SHA256 build metadata are inert admitted exceptions, discarded before exec;
  changed metadata is rejected. No generic PYTHON* exception is permitted.
- Build with official Rust1.90.0 Alpine3.22 AMD64 digest
  `ee9b9f2ebd8741e4ecf1ade08918e492d642c4e625be481bb7b7954b89e9d504`,
  locked/offline x86_64-unknown-linux-musl. The ELF has no interpreter or DT_NEEDED.

- Fixed `preparation` workload profile exists only in the compiled native feature.
  It forwards exactly selection/exclusions SHA and bytes, job/session/deployment/prefix,
  image/source, profile-id/policy-version primitives after the common permanent drop.
  Fixed `/app/coding_preparation_worker.py`, `/app/coding_preparation_core.py` and the
  existing identity module are root-owned and checked; no code/path/grant selector.
  Credential/proxy/loader refusal, source consistency and job-lifetime FD3 lease reuse
  the same initializer implementation. Default strict build refuses this new profile.
- Fixed `model-acquisition` exists only in the native build and accepts exactly
  catalog/version, job/session/deployment/prefix and image/source pairs. It selects
  `/app/acquire_model.py`, retains FD3 and all permanent-drop checks, forwards only
  the existing remote-fetch controls, and refuses provider/proxy hints before
  clearing the environment. The worker's four credential-file probes stay intact;
  inaccessible observations refuse. Image/native admission remains separate.
- Fixed native-only `qwen-sft` accepts six declared operation/recipe/source/manifest
  primitives, selects fixed worker/core files, and privately hands off only the
  empty original `/opt/ml/model` FD3. Root-owned `/opt/ml` must be0755; target is
  only0:0/0755 or10001:10001/0700. Lock, named-inode and parent identity checks
  precede FD-only fchown/fchmod; no parent, recursive or0777 repair. Permanent
  IDs/EPIA0/NNP/source/provider refusal stays common. Actual native training mounts,
  input reads and GPU access remain qualification gates.

- `model_file_boundary` is an additive raw-model-only OS module. Its no-argument
  pre-drop capture walks only the four fixed home/root `.aws` credential/config
  paths with retained O_PATH/no-follow descriptors. Presence, symlinks and unknown
  errors refuse; exact ENOENT records the earliest missing component and finite
  original ancestor identities/modes. Rechecks precede the sealed handoff.
- The provider record is bounded16384bytes and sealed with WRITE/GROW/SHRINK/SEAL.
  The returned original OFD is read-only; fixed model inventoryFD4 integration is
  initializer-owned. It records earlier producer observation, never post-policy
  absence, permission, attestation or native admission.
- After permanent drop/NNP1, `apply_model_file_boundary` accepts only original
  outputFD3 from the private initializer handoff and acquires/renews the current
  exclusive lease on that supplied OFD; this does not verify historical continuity. ABI6 is required; all16 filesystem
  actions including IOCTL_DEV are handled. No ABI downgrade or syscall fallback.
  Exactly28 source-owned runtime/current-PID/output inode rules and five exact
  root-owned stable aliases are qualified with original components. Only original
  output is writable; `/dev/null` is readonly character1:3. Root/home, devshm,
  general proc/usr/etc and Processing-parent writes receive no blanket rule.
- The eight-field `FileBoundaryProof` adds observed `process_dumpability`.
  GET_DUMPABLE only accepts0/1 and is stable-checked around policy application.
  Same-PID directory remains10001:10001; final status/fd nodes require0:0 under
  measured preexec dumpability0 and10001:10001 under1. Unknown/2 refuses; there is
  no dumpability setter, new path/right or inference about postexec inode/read
  qualification. Returned metadata remains data, not authority.
- Runtime/library/CA compatibility measurements and syscall fakes are source proof.
  Native TLS, enforcement, aliases, providers, inherited descriptors and descendant
  qualification remain separate gates. Existing strict/dataset profiles stay separate.

- `model_fd_handoff::handoff_model_fds` consumes the private original output File
  and typed sealed producer inventory. It validates10001/private0700/currentlease
  and readonly/seals15/anonymous/count/identity; stdio sources refuse. Both OFDs
  are duplicated to safe CLOEXEC FDs>=5 before originals drop and fixed3/4 dup2.
  Targets stay inheritable, temporaries close, and required Linux close_range plus
  finite supported RLIMIT/F_GETFD checks remove unrelated inherited descriptors.
  The caller first drops other live Rust-owned file objects; stdio0/1/2 stays live.
  No FD/path/environment/CLI selector or executable handle exists. Main/drop/
  Landlock/worker/native integration remains separately owned.

- Native Qwen SFT stacks a separate `qwen_sft_tcp_deny_v1` restriction after its
  original output/inventory FD3/4 handoff and permanent native-final drop. ABI>=4,
  singlethread/NNP1, exact16-byte filesystem0/network3 ruleset and no port grants
  are fixed. Owned ruleset FDs close on success/failure; unsupported/install
  failure refuses before Python. Numeric metadata80 and fixed public443 checks
  require errno1/13 only; EINVAL/timeout/reachable/unknown states still refuse.
  Bounded profile/ABI/empty-port observations are descriptive, not LSM attribution
  or attestation. UDP/IP/other protocols and existing sockets are not revoked.
  Downloader TCP53/443/filesystem policy and native GPU qualification stay separate.

- SFT-only TMPDIR is fixed `/home/appuser/.cache/amlas-qwen-sft`, separate from
  input/checkpoint output. After permanent drop and original FD3/4 handoff, walk
  original no-follow home/appuser/cache/leaf FDs; require home0:0/0755 and
  apphome/cache/leaf10001:10001/0700. Only absent compiled cache/leaf components
  may be created0700. Existing types/owners/modes/nonempty leaf or named identity
  drift refuse. All new descriptors close before TCP gate/Python; no provider
  `.aws` mutation, caller selector, parent repair or native mount claim exists.

- Native-only case-freeze accepts binding SHA/count and exact image source; selects root-owned wrapper/minimal data package after the common permanent drop. Strict builds reject it. SDK/model/application state is absent from this worker closure.

- Optional `oracle-child-v1` builds only fixed `/usr/local/bin/amlas-oracle-init` from the same dependency-free crate; other initializer profiles keep their existing feature gates. This unprivileged child never receives caller path/policy/source/argv selectors, changes groups/IDs/caps, or grants runtime authority from an observed trace. Account-only native compilation/confinement/image tests are required; developer hosts run only neutral source/command tests.

- Native-only `coding-pair` accepts exactly binding SHA/count/source, checks both source selectors and trusted launcher SHA, and repairs only original empty fixed output under root-owned Processing0755. Pair-only supplementary group[10001] preserves the existing oracle contract; all common permanent IDs/EPIA0/NNP and original FD3 lease precede fixed root-owned `/opt/venv/bin/python -I /app/coding_pair_worker.py`; no root Python import, path or executable selector exists. The pair SDK broker alone retains a bounded `/v2/credentials/` single-segment native role metadata URI; static credentials, profiles, custom/full endpoints, authorization hints and proxies still refuse. Other workload provider refusal is unchanged. Credentials remain in the broker; candidate/oracle execution stays in its existing isolated path. This intentional SDK role access differs from data-worker credential absence and requires separate native account/role/interpreter proof.

## Work Guidance

Fixed native raw-model startup may defer only a validated relative container
credential path. It never forwards that pointer; child execution requires the
fixed kernel TCP53/443-only/no-bind policy and actual EACCES/EPERM observations
for both HTTP metadata endpoints. Other credential hints/profiles still refuse.
This is port-scoped denial; IP/private443/UDP/future endpoints are not covered.

Model acquisition and fixed Qwen training capture four original provider-path absence
observations before the permanent drop, privately hands off locked outputFD3 and
readonly sealed inventoryFD4 with collision-safe same-OFD duplication, then
applies the fixed ABI6 filesystem policy before Python exec. Fixed temporary paths
stay under the original output tree. Bounded preexec rule/dumpability observations
are descriptive and require separate native postexec/enforcement qualification.
Qwen training uses the same collision-safe FD3/FD4 handoff without applying the
raw downloader filesystem policy to CUDA. Other profiles keep their original
descriptor and startup contracts.

Use explicit fail-closed errors and red-first tests. Fixture green is not native admission. Original-user-namespace root regain and
local self-setns denial do not prove alternate namespace-join closure; native
unshare/setns admission and provider-filter content remain separate gates.

## Verification

Run cargo test --workspace --locked and the controlled Processing initializer image tests on the approved remote host. The binary's unit tests share the fixed Processing directory and its lock, so run tests with one test thread (`RUST_TEST_THREADS=1`). The native startup test needs a container capability set within the processing ceiling (Docker's default set without SETPCAP, NET_RAW, MKNOD and SETFCAP). The descriptor handoff test needs an open-file limit above 4096; the boundary tests need Linux 5.13 or newer.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `src/AGENTS.md` | Static Linux initializer source |
| `tests/AGENTS.md` | Real descriptor identity regression |

- The fixed Qwen SFT profile also routes generate only with humaneval_greedy_v1/version1. Train/reload retain fit_v1/full_v1. Cross-operation recipes refuse before fixed worker opening; all existing six-pair source/digest/count, drop, FD3/4, scratch and TCP restrictions remain unchanged.
