# src/amlas/data_eng

## Purpose

Own data-engineering constraints, portable acquisition code and pure coding-conversation preparation.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Hub inspection returns aggregate21000-byte SDK metadata projections (lowered from 60000, with the files page from 27000 to
  8000 and the declarations page from 12500 to 2800 bytes, on 2026-11-07: the real-model test harness measured 23.7-24.7 KB
  for every page of a large inventory) with whole file/hash/license/path fields. Files and flattened config declarations have explicit offset/returned/total/next_offset coverage; continuation requires the full resolved revision and rejects revision drift. next_page supplies directly executable nonnull offsets, using total for an exhausted inventory; both exhausted returns null. Declaration scanning validates/counts the complete bounded source but allocates output records only for the selected page. Ineligible whole fields or an oversized whole record fail instead of slicing. Split summaries retain explicit total/partial coverage. Pinned acquisition independently selects its exact filename across the bounded source listing, including files beyond the first inspection page; size/hash checks remain unchanged. Paging grants no acquisition, parser or training authority.

- Hub file names with a refused character (added 2026-11-07): inspection used to fail as a whole with
  `invalid repository artifact path` when any listed file name held `#`, `?`, `%` or a backslash
  (bigcode/commitpackft lists `data/c#/data.jsonl` and `data/f#/data.jsonl`). A structurally sound name whose only defect is
  one of those characters cannot be acquired, so inspection omits it from the file inventory and reports
  `files_unsupported_path_count`; every other malformed name (empty, absolute, `.`/`..`/empty segment, control
  character, over 1024 characters, non-string) still fails the whole inspection, and `_filename` for acquisition is unchanged.

- `steering.py:SteeringProfile` — Domain-steering profile for the imputation / synthetic engines.
- `inspection_schema.py` validates fixed bounded JSON summaries/receipts without
  agent or SDK imports. Reject extras, duplicate keys, nonfinite values and claims
  of preparation/training/trust. Samples remain untrusted quoted data.
  Fixed inspection receipts require exact actual UID/GID10001 triples, empty
  groups, zero effective/permitted/inheritable/ambient/bounding capabilities and
  integer no-new-privileges1. Reject missing/extra/noninteger privilege fields.
  Additive receipt schema2 requires `sagemaker_native_ceiling_v1` plus an independently
  source-selected expected profile. Native proof has exactly14 fields: the old nine
  plus keep_caps0/seccomp2/fs_uid+fs_gid10001 and bounded known securebits with
  KEEP_CAPS/NO_SETUID_FIXUP off. Active capabilities remain zero; actual bounding
  bits must be a subset of fixed0x200404fb. Schema1/default decode stays zero-bounding.
  Pure validation describes guest claims; native source/image/job/version binding and
  admission remain independent. No guest claim or profile string grants authority.
  Version2 also requires fixed descriptor3/owner10001:10001/mode448, bounded integer
  device/inode and true current lease_active in exact output-directory proof fields.
  This proof describes the guest observation; it grants no caller-selected authority.
  Admit at most 16 structural JSON containers before decoding, respecting quoted
  text, escapes and the standard byte encodings; retain byte and schema limits.
- `hub_dataset.py` inspects anonymous public metadata and acquires one pinned
  listed artifact with size/LFS SHA-256 or Git-framed blob validation. It imports
  no agent, model, Hub SDK or executable repository code.
  Acquisition failures add closed stage/exception/cause labels; native diagnostics
  format those labels without copying raw error, URL, credential or dataset text.
- The local adapter returns absolute `artifact_uri`/`local_path` and `provider=local`;
  success proves an artifact, never preparation or training.
- Writes use configured local roots and no-follow directory descriptors. A private
  unique bundle's `COMMITTED.json` atomically commits both data and lineage.
  Preserve preexisting name collisions; verify named directory identity before
  and after publishing the commit manifest. A changed path fails and removes the
  owned bundle instead of returning a successful replaced/symlinked result path.
- Private `_acquire_huggingface_dataset_into_fd(repo_id, revision, filename,
  output_dir, output_fd)` duplicates a trusted worker's verified directory FD and
  retains its same open-file-description lease. The caller's original FD is never
  closed/unlocked. All mkdir/data/hash/lineage/commit/cleanup operations use that
  same held root. Retain the original artifact write/read FD through commit;
  independently verify its exact size/count/source hash with pre/post inode,
  nanosecond mtime/ctime and link identity, then reject changed names/snapshots
  before and after publication. Independently rehash/count the same FD on both
  sides of COMMITTED publication against original SHA/Git integrity; equal inode,
  size and timestamps are not immutable-byte proof. Named directory checks walk original components without resolve
  or symlink following. The path is metadata, never replacement write authority.
  Invalid/missing/mismatched FD cannot fall back to the public local opener.
  Privilege/image/source/namespace admission stays in the trusted caller; this FD
  argument is absent from public model/tool/profile schemas.
- Public local acquisition retains its original allowed-root guard and opener,
  then uses the same private bounded writer. Prefix lookalikes and symlink aliases
  cannot enlarge local authority. Existing return schema and integrity limits hold.
- `coding_preparation.py:prepare_canonical_conversations` takes already parsed
  plain JSON records, fixed versioned `PreparationPolicy` and explicit exclusion
  lists. It performs no filesystem, SDK, model, loader or permission actions.
  Preserve explicit native call/result IDs and nullable text/reasoning fields;
  unknown conventions or resource envelopes make the selection incomplete with
  no usable records. Never infer miniswe result IDs or terminal omissions.
  Require exact integer resolved1 and fixed SPDX eligibility; exclude named
  high-confidence secret forms without copying snippets or credential hashes.
  Ignore gold/reference/model patch content. Exact duplicate clusters precede
  repository/instance/duplicate connected-group deterministic80/10/10 splitting;
  require all splits, count conservation and source-order independence.
  Near deduplication is explicitly not implemented. Pure complete status is
  descriptive, never a committed manifest, native readiness or trust grant.
- `model_weight_manifest.py` is pure expected-byte metadata for one source-owned
  Qwen3.5-9B catalog/version/revision. It performs no network/FS/framework loading,
  callbacks, grants or registration. Fixed8GiB/file300GiB/base limits do not alter
  the dataset512MiB cap. Exact config/tokenizer/template/index/shard hashes/counts,
  regular-file/path inventory, Apache/native architecture and real stored versus
  active-language parameter fields are source-pinned. Reject custom objects,
  missing/extra/duplicate refs, traversal/links/pickle/remote-code and ready claims.
  Original bounded index bytes must match the source hash/count before decoding;
  exact tensor/shard counts and shard sets remain descriptive, never native proof.
  Independent acquisition/integrity/loading/fit/native admission stays elsewhere.
- Model acquisition is a separate raw-byte path: `model_public_fetch.py` selects
  only source-catalog HTTPS refs and retains public DNS/IP pinning, TLS, per-hop
  checks, deadlines and fixed8GiB/file caps. Generic dataset512MiB limits remain.
  `model_weight_acquisition.py` retains private outputFD authority/current lease,
  no-follow exclusive0600 files, disk reserve and all writeFDs through last
  COMMITTED publication plus bounded same-FD byte/hash/name checks. Failed marker
  writes remove only the owned inode. This detects mutation, not future immutability.
  `model_acquisition_schema.py` bounds metadata262144bytes/depth12 before decoding,
  fixes source/provenance/hash chains and accepts raw acquired state only;
  preparation/training/trust/readiness remain false. No model parser or SDK runs.
- Owned files: `__init__.py`, `steering.py`, `hub_dataset.py`, `coding_preparation.py`, `model_public_fetch.py`, `model_weight_acquisition.py`, `model_acquisition_schema.py`, `model_weight_manifest.py`, `open_swe_canonical.py`, `open_swe_derived.py`, `commitpack_profile.py`.

- `open_swe_canonical.py` is a pure parsed-row profile bound to the captured
  NVIDIA Open-SWE revision/config/card. It performs no Parquet/filesystem/network/
  SDK/model operations and grants no authority or readiness. Root's isolated parser
  supplies parsed JSON plus immutable artifact/version/hash/row provenance.
  Preserve explicit repo/license/resolution semantics (-1 unknown,0 failed,1 solved);
  dataset CC-BY attribution is separate from repository SPDX. Bounded tool JSON uses
  the existing closed reducer decoder; duplicate/nonfinite/deep/oversized values refuse.
  Preserve null text/reasoning and native causal IDs. Absent nullable optional fields
  create no IDs. Missing/orphan/duplicate/dangling links refuse with bounded counts;
  never infer adjacency or extract linkage from observation text. Gold/model patches
  and row-authored provenance never enter normalized records. The live official info
  schema lacks tool result IDs; this does not attest compatibility of pinned Parquet
  rows. No near-duplicate/leakage completeness or corpus-substitution claim is made.

- `open_swe_derived.py:normalize_open_swe_row_derived` is a separate fixed v2
  parsed-row profile; strict v1 and the preparation reducer remain unchanged.
  Validate complete bounded JSON/tool/message/call shapes before linkage. Only
  an uninterrupted assistant exchange originating exactly one explicit call may
  derive an absent result ID. Null/malformed IDs, mismatches, duplicates, orphan
  results, intervening events and dangling calls refuse the whole row. Exchanges
  originating multiple calls require explicit IDs for every result. Never infer
  from content, name, adjacency or a terminal marker.
  Keep each derived edge's original row/message/call indexes, native call ID,
  absence state, policy version and `derived_not_original_source_verified` status.
  Original artifact hash/version/row provenance stays separate from the canonical
  messages/tools digest. Retain the complete sidecar in normalized output; the
  existing reducer drops extra metadata, so managed preparation must preserve a
  separate source-row/digest-bound sidecar before claiming final lineage survival.
  No corpus/parser/tokenizer/native acceptance, readiness or authority is implied.

- `commitpack_profile.py` is a fixed parsed-row change-SFT profile for the public
  pinned CommitPackFT Python file. Exactly ten plain strings and immutable typed
  artifact/version/hash/ordinal provenance are bounded before any copying/encoding.
  No parser, filesystem, network, SDK, model, execution or callbacks exist.
- Preserve subject/old and new filenames/full old code in the recorded user wrapper;
  exact full new code is assistant supervision, tools empty. Original fields remain
  data-only provenance. No truncation, invented repository/resolution/execution or
  historical reasoning. Map row licenses only to the existing four SPDX labels;
  unknown/mixed labels, malformed all-affiliation CSV, secrets and chat controls
  refuse the whole row without source snippets.
- Group all declared repository affiliations, repeated source commits, exact
  canonical/input/full-code-pair duplicate edges before canonical deduplication.
  Keep every duplicate-origin raw row, commit, row license and digest bound to its
  source provenance; conflicting raw rows at one immutable artifact/version/hash/
  ordinal make grouping incomplete. Exact-width/type/absent-source guards precede
  key copying; deterministic connected-component hash buckets80/10/10
  require all splits. Closed-envelope replay/digest checks and count conservation
  precede usable data; resource/unknown input failures expose no usable splits.
  Near duplicates, complete upstream fork graph, held-out exclusion, tokenizer/
  context/token-floor, managed preparation and native/training acceptance remain
  explicitly unverified. Descriptive grouped output is never readiness or authority.

- `commitpack_preparation.py` implements the closed CommitPack v3 serializer,
  whole-component exclusions, one original-ordinal lineage and all-origin replay.
  It preserves the pure100000/256MiB envelope and exact8MiB exclusions bounds;
  REPORT3/COMMITTED2 counts/coverage are fixed. Nonusable prefix reports expose
  no split refs or successful commit. Replay checks declared transformations and
  metadata consistency; it cannot independently authenticate unused raw source
  fields after replacement of every caller's trusted outer pin. No SDK/model,
  resolution/execution claim, tokenizer floor or readiness grant is added.

- CommitPack review regressions bind the bytes actually consumed by JSON decoding,
  account retained rejected rows and assembled output envelopes, and retain trusted
  per-source stream counts privately. Parent replay requires exact counts for every
  source; trainer may omit that independent constraint and must not claim raw-source
  authentication. Hash/parse/restore and coherent101/99 substitutions refuse.


- `coding_case_profile.py`, `coding_case_codec.py`, `coding_case_manifest.py` own only the pure fixed HumanEval164 decoded-data projection and replay. The eight-field source descriptor is descriptive; caller independently verifies raw pinned Parquet before decoding. Exact ordered IDs/five string fields, MIT/card/file pins and bounded complete selection are required. PROMPTS retains only id/prompt/entry_point; private ORACLES retains id/test/canonical_solution. Membership/lineage/hash roles embed in CASE_MANIFEST; EXCLUSIONS uses only known original repo/commit and unchanged literal CommitPack code-pair domain. Unknown wrapper/near/pretraining coverage is explicit. COMMITTED is final returned bytes, never an FS/native publication or readiness grant. Strict bounded duplicate/nonfinite/depth JSON replay recomputes all projections from retained original fields; consistent whole replacement cannot authenticate raw origin. No source codec performs I/O, Parquet decoding, imports/eval of case strings, model/job/sandbox execution or authority assembly. Spec/grader/managed integration remain separate.

## Work Guidance

Model acquisition evidence admits only the unchanged legacy observation or the
closed `model_provider_inventory_v2` record. Validate exactly four ordered
earlier absence records, bounded original ancestors and four truthful current
states; deny unknown/extra grants and universal-absence claims. These records are
descriptive guest observations, not native admission or attestation.

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract, `tests/test_huggingface_ingestion.py` and
`tests/test_huggingface_commit_boundary.py` and `tests/test_coding_preparation.py`.

## Child DOX Index

None.

- `coding_case_managed.py` wraps untouched pure schema1 CASE_MANIFEST/COMMITTED bytes in closed schema2 source/producer envelopes while retaining five outputs. It replays the original complete164 ordered projection and original immutable acquisition lineage. Metadata alone authenticates no raw source or caller authority; no execution, grant, contamination-clean or readiness claims.
