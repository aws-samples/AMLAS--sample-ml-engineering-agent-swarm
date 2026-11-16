# src/amlas/sagemaker_entrypoints

## Purpose

Run managed tabular training/processing entrypoints and write their expected result artifacts.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Honor managed training input/output channels and write the expected summary/prediction artifacts.
- Runtime credentials come from the managed role chain, not a developer profile name.
- Label/prediction vectors use the shared data-only writer; never serialize
  object arrays with pickle to preserve string labels.
- `dataset_acquisition.py` runs guarded pinned acquisition, verifies data and
  lineage against the atomic core COMMITTED manifest, and writes ACQUIRED.json
  with confined relative paths for native Processing EndOfJob S3 upload.
  Its native fixed guarded launcher verifies res/fsUID/GID10001, groups[],
  effective/permitted/inheritable/ambient0, Bnd subset0x200404fb, NoNewPrivs1,
  keepcaps0/Seccomp2 and known securebits with behavior0x14 off before application
  import. Source-owned main_native/run_acquisition_native produce schema2 with
  privilege_profile and actual14-field privilege_proof; caller proof is not accepted.
  The original strict main/schema1 still requires Bnd0 and remains separate.
  Native proof fields are descriptive, not grants or independent attestation. The fixed original output FD3 retains its same-OFD lease
  for worker lifetime; source CLI/environment equality and output owner/mode/name
  checks precede workload operations. Managed main has no local fallback or FD/path
  selector. Private core FD writes retain this authority; returned paths are metadata.
  Readers walk original components with O_NOFOLLOW, retain each opened file and
  hash/read/stat that same FD. Original root/artifact identity and integrity checks
  precede exclusive FD-relative receipt publication. Each verified opened artifact
  retains its dev/inode/size/mtime_ns/ctime_ns signature across JSON/hash checks;
  compare that same-FD signature and its retained content digest with bounded
  same-held-FD rehashes before and after receipt publication, because filesystem
  timestamp granularity can hide immediate same-length writes, as well as
  current named identity. These observations do not guarantee future immutability;
  native exact-version raw rehash remains necessary. Generated ACQUIRED keeps its
  readable write FD open through final checks. Bounded actual bytes/count/digest
  must match precomputed JSON before and after postchecks plus named identity;
  the receipt path is never reopened as reading authority. Explicit offline oracles open
  original local components without resolving away aliases.
  Managed `main` repeats identity verification before parsing.
  It never executes dataset scripts or deserializes pickle/joblib/Torch objects.
  Failed core results report only closed stage/type/cause context. Generic commit
  failure does not erase the diagnostic, and raw error details are never logged.
- `dataset_inspection.py` is fixed root-owned standalone AMD64 code entered only
  through `amlas-processing-init inspection` and its frozen primitive binding pairs.
  Before schema/SDK/data imports it requires actual permanent UID/GID10001,
  empty supplementary groups, effective/permitted/inheritable/ambient/bounding
  capabilities zero and no-new-privileges1 in its retained strict function path.
  The fixed image main uses a separate source-owned native path with the shared
  OS-only `/app/processing_identity.py:verify_native_identity()` guard. It requires
  active capabilities0, filesystem IDs10001, keep_caps0/seccomp2/NNP1 and only fixed
  bounding-ceiling bits0x200404fb, with KEEP_CAPS/NO_SETUID_FIXUP securebits off.
  No CLI/env/model profile selector exists. Native outputs use schema2 plus the exact
  profile discriminator and actual14-field proof; strict outputs remain schema1.
  Only the image-owned stdlib guard precedes the worker identity recheck; data/schema/SDK
  imports follow actual identity and directory gates. Direct root execution refuses.
  Version2 consumes only the fixed inherited outputFD3, owner10001:10001/private0700,
  and retains its exclusive open-file-description lease until process exit. Actual
  directory/name/inode/current-lease and unexpected-FD guards precede SDK/data/schema.
  Source CLI must match compiled ENV before these operations; no FD/ready selector.
  Input components open in their original lexical form with O_NOFOLLOW; size/hash
  and parsing share one single-link regular file FD. Bounded pre/post same-FD
  stat compares device/inode/size/mtime_ns/ctime_ns after hashing and parsing;
  same-inode content mutation refuses before publication without assuming readonly
  service inputs. Coarse/constant timestamps are insufficient: a bounded same-FD
  post-parser rehash must equal the pinned expected SHA/count before publication;
  no path reopen occurs. This detects retained byte mutation, not future atomic
  immutability. CSV wrappers detach so the authoritative stream survives this check.
  Both bounded JSON files use
  FD3-relative exclusive publication with named-root/file identity rechecks. Both
  original output write FDs remain open through publication/final checks. Bounded
  same-FD actual byte count/SHA readbacks match expected encoded bytes after each
  fsync and across both-file final publication checks; no path reopen occurs.
  Persistent byte mutation refuses in the worker. This is not future immutability;
  native exact-version parent verification remains a separate required defense. Receipt
  output proof is descriptive current evidence, not proof of historical continuity
  or an authority grant; all other credential/network/admission gates still apply.
  It probes credential-provider/env presence and metadata/network denial before
  parsing, rejects forced metadata-disable evidence, hashes the single fixed input,
  bounds formats/allocations/runtime and emits only SUMMARY.json/INSPECTED.json.
  It performs no SDK service calls, repository execution or pickle conversion.
  Its external TCP probe is fixed at `1.1.1.1:443` with a 0.5-second timeout;
  native admission requires a fresh successful operator control to that same
  endpoint. A failed connection establishes only this bounded target observation.
  Proxy hints in uppercase/lowercase HTTP(S)/ALL/NO_PROXY names refuse by presence
  before SDK discovery; values are never logged. The strict receipt retains the
  actual kernel privilege proof; SDK/provider/metadata/network checks remain
  independent. Request flags and fixture green are not native admission.
- Qwen pre-core diagnostics run only after the original exact metadata-unknown
  refusal with captured errno22. Two fixed numeric AF_INET/TCP targets receive
  independent socket-create/timeout-setup/connect/close observations; each connect
  has a0.5-second timeout and every created socket is closed once. Only indices,
  closed stages/outcomes/errno categories are emitted (at most8 records). No DNS,
  send/receive, provider contents, retries or acceptance occurs. Unexpected probe
  failures cannot mask the original refusal; even reachable diagnostics still
  refuse. Existing provider guards, errno allowlist and Rust policy stay independent.
- `qwen_sft_train.py` is the fixed image-owned child at `/app/qwen_sft_train.py`.
  Its root-owned standalone core is `/app/qwen_sft_core.py`. Six required primitive
  pairs are operation(train/reload), recipe-id(fit_v1/full_v1), recipe-version1,
  source-commit40hex, manifest-sha25664hex and manifest-byte-count1..262144.
  No caller paths, images, roles, code, profile, environment or FD selectors exist.
  Permanent identity/native capability checks, credential/provider hint refusal,
  unknown provider-file refusal and bounded fixed network observations precede core
  loading. Reuse the unchanged stdlib-only root-owned `/app/model_worker.py` sealed
  inventory/parser and four original provider probes; no main/model-acquisition
  profile or filesystem policy is invoked. FD4 must be readonly, single original
  four-sealed memfd with nlink0 and bounded16KiB closed JSON. Earlier absence is
  descriptive; current ENOENT stays absent and current EACCES/EPERM stays denied
  after drop only with valid earlier inventory. Present/unknown states and provider
  hints refuse without values. The fixed public endpoint also uses closed errno/
  timeout observations; unexpected errors refuse. Bounded log lines contain only
  inventory hash/four states/stage and guest_claim_attested=false, never a grant.
  Revalidate readonly/seals/closed inventory at probe completion, then compare
  the original FD identity/hash; same-object writable rebinding refuses. These
  endpoint checks do not prove uninterrupted continuity against arbitrary in-process
  concurrent mutation. Recheck the same FD identity/hash/current states after core import before the
  operation; changed observations refuse. The guard still returns the actual14-field
  native proof. FD3 must be private10001:10001/0700 with original name/current lease;
  only validated FD4 may join inherited0..3. Other descriptors refuse and an
  independent output contender must fail.
  Original no-follow RUN components, raw exact count/hash, bounded finite/duplicate-free
  JSON and retained-byte/name checks precede application imports. Pass the actual raw
  startup digest into the core; canonical reserialization cannot substitute for it.
  CLI source equals compiled ENV; provider presence is rechecked after core import.
  No SDK, pickle or repository code is loaded. Refusal emits bounded JSON with closed exception
  type and exits2. Known template incompatibilities add only closed reason/counts,
  never conversation payload. Native GPU/mount/input-access/provider/egress qualification remains
  separate; a failed bounded endpoint connection does not prove universal denial.
- `tabular_train.py:run_training` — Fit ``algo`` on the full train split, predict the test split, write y_pred_test.npy.
- `tabular_train.py:resolve_sagemaker_paths` — Resolve the train/test dirs + output dir from SageMaker's SM_* env vars.
- `coding_preparation.py` is the fixed standalone canonical JSONL and pinned
  Open-SWE Parquet worker. Its source-owned v2 pair selects only the approved
  revision and a fixed pure codec package after common identity/source/FD guards.
  The locked PyArrow reader disables extension conversion and prefetch, verifies
  available page checksums, bounds Thrift metadata and selected uncompressed
  column totals before batch decoding, and bounds decoded batches/JSON across
  the entire selection. Gold metadata columns are never materialized.
  Known whole-row codec refusals retain original lineage separately from reducer
  drops; structurally normalized rows can still fail resolved/license filtering.
  REPORT v2 counts both stages, with one lineage per original row and derived
  linkage explicitly unverified against original result IDs. JSONL v1 remains
  unchanged. Decoder bounds do not prove absence of native parser exploits.
  It has
  no guest SDK/model dependency. It uses only source-owned native identity helper
  `/app/processing_identity.py` and unchanged pure `/app/coding_preparation_core.py`.
  Root-owned code, source equality, exact inherited descriptors0..3 and actual
  privilege/profile gates precede CLI/data parsing; no caller-selected profile,
  path, FD or loader is exposed by managed main. Env/home-file/network observations
  are bounded/descriptive; separate native SDK/credential/IPC isolation is required.
  Read exact selection/exclusion/source/acquisition manifests through original
  no-follow held descriptors, hash before/after parsing and inject source metadata
  from verified binding instead of row-authored fields. Unknown mapping/incomplete
  selection cannot commit. Preserve original FD3/same-OFD lease; publish data-only
  train/validation/test/membership/lineage/report files exclusively, COMMITTED last,
  and verify retained-write-FD actual bytes across final checks. No pickle/SDK loads,
  first-N fallback, readiness marker or prepared/trained claim is made by the guest.
- `model_weight_acquisition.py` is the fixed no-SDK/no-agent/no-model-parser
  downloader child for the dedicated model initializer profile. Its eight binding
  pairs have no URL/path/role/profile selector. Compiled source, actual shared
  native identity, original outputFD3/current lease and closed-FD checks precede
  package imports; only EBADF establishes a closed descriptor. The guarded image
  package path is literal `/app/src`. Provider/proxy/env hints, four fixed provider
  paths and metadata endpoint observations remain independent controls. Inaccessible
  provider paths are unknown/refused, never absence. Metadata probe resource or
  unknown errors refuse; qualified fixed-target denial/timeouts are bounded
  observations, not universal network/credential absence. Clearing hints cannot create
  evidence. Recheck after package startup. Output proofs are guest observations,
  not native attestation. Shared model-profile/image qualification remains required.
  The additive private inventory consumer validates a readonly four-sealed memfd
  with bounded closed JSON before package imports. Earlier original-component
  absence stays distinct from current absence or denied-after-drop probes;
  malformed/unsealed/writable inventory and other OS errors refuse. All four
  provider probes remain. Model-only named walks use O_PATH ancestors without
  broad directory-read grants. Startup handoff/native policy integration is a
  separate gate; legacy evidence and inaccessible-path refusal stay unchanged.
  Fixed main requires inventoryFD4 before package imports and repeats all original
  provider probes after package startup. Only this source-owned v2 path can report
  denied_after_drop alongside sealed earlier absence; it does not report current
  absence for denied probes or guest-domain attestation.
- Owned files: `__init__.py`, `tabular_train.py`, `coding_preparation.py`, `model_weight_acquisition.py`.

- The fixed CommitPack v3 path loads root-owned pure profile/replay code only
  after identity/input/output gates. It reads complete JSONL rows and rehashes
  retained FDs, preserves every original raw-line digest/ordinal and normalized
  origin, applies global grouping before component exclusions, and writes the
  fixed six files thenCOMMITTED2 last. Row/expanded/decode failures emit only a
  bounded scan_complete false prefix report; no truncation or success marker.
  Native privilege/provider observations remain descriptive, not admission.

- CommitPack review regressions bind the bytes actually consumed by JSON decoding,
  account retained rejected rows and assembled output envelopes, and retain trusted
  per-source stream counts privately. Parent replay requires exact counts for every
  source; trainer may omit that independent constraint and must not claim raw-source
  authentication. Hash/parse/restore and coherent101/99 substitutions refuse.

- Fixed Qwen refusal JSON may carry admitted source-owned stage/cause class while ordinary errors remain generic. No native exception text, paths, model/data contents or credential values are logged.

- Pre-core provider inventory/credential/core-import failures retain only source-owned stage, exception class and exact closed refusal code. Unknown messages are never echoed; every existing inventory/provider denial remains a denial.

- Fixed case freezing also requires source-commit CLI equality with image environment and private producer binding before input parsing. The image uses only a minimal source-owned data package; no SDK/model/application-state import.

- `coding_pair.py` is the fixed AWS Processing CPU broker at `/app/coding_pair_worker.py`, invoked with three source/hash/count pairs. Original file bytes and source-owned oracle pins precede any child; credentials remain in broker SDK custody. The image must provide unprivileged execution and a matching baked source selector; missing native image/role qualification refuses. The worker publishes328 receipts and paired summary, a streamed hash-only invocation manifest, and COMMITTED last. Feed the same owning trace accumulator from fsynced record bytes and verify receipt/cleanup chains before writing COMMITTED. It performs no pickle/model loading and never executes candidate code in the broker. Native process/isolation/role/deadline qualification is not established by source unit tests.

- Paired broker startup requires the trusted `AMLAS_CODING_PAIR_LAUNCHER_SHA256` selector and actual fixed installed binary verification before reading/decoding RUN or using SDK/oracle. Closed RUN includes the same `oracle_launcher_sha256`; independent held-FD hash/owner checks and runner domain must match. Caller/model data cannot choose the selector; missing/malformed/foreign values remain unconfigured or refuse. Immutable image, native provider and containment qualification remain separate.

- The paired broker requires exact returned built-in ID aws.codeinterpreter.v1
  and AWS-publisher CodeInterpreter ARN in the original run's selected region
  and SDK-resolved partition before session dispatch. Missing/foreign identity
  refuses without response-derived scope or grants. Existing READY and optional
  None/SANDBOX network, role/filesystem checks stay separate; omitted metadata
  is not configuration-isolation or zero-egress proof.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.

- Checkpoint-save diagnostic formatting permits only the closed core substage and
  matching fixed reason pair; arbitrary strings/fields are omitted. It grants
  no checkpoint, reload, training or quality acceptance.

- Training-receipt/final-readback/final-publication failures expose only closed
  source stages/classes; no raw exception or model/data/tensor content.

- `coding_case_freeze.py` is a fixed data-only case worker. Verify inherited native identity and retained input/output FD authority, hash exact raw bytes before Parquet decode, enforce all164/five string fields, publish the five canonical outputs with COMMITTED last and held-FD readback. Case code strings never execute/import; no SDK/model loading or dynamic loader callback. Rust mode, image/dependency packing and native qualification are separate gates.

- Qwen worker accepts only train/reload with fit_v1/full_v1 or generate with humaneval_greedy_v1, all version1. Existing source, permanent-drop/provider/output/raw manifest gates precede core dispatch; no new CLI path/model/policy selector exists.

Fixed Qwen startup retains RUN-v1 for train/reload/generate. RUN-v2 is reload-only
and must pass the existing pure core validate_reload_binding with exact consumer
source/recipe, closed field inventory, base file metadata and compatibility
contract after unchanged provider rechecks and before managed dispatch. Allowing
this metadata schema never grants checkpoint/native/GPU compatibility; OS/output/
source/no-follow/raw hash guards and core artifact checks remain required.
