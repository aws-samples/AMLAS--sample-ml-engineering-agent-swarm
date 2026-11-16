# src/amlas/training

## Purpose

Own fixed one-model Qwen SFT, exact local staging checks, and independent adapter reload.

## Ownership

Inherits ../AGENTS.md; owns direct files and the dedicated training dependency lock.

## Local Contracts

- Only source catalog `qwen3_5_9b` version1 and its nine exact expected files are accepted.
  Expected pins describe bytes; they grant no acquisition, image, native or training admission.
  Load `Qwen3_5ForConditionalGeneration` locally with safe weights and no repository code.
- Native paths are fixed `/opt/ml/input/data/base_model`, `prepared`, `adapter` and
  `/opt/ml/model`. The trusted parent stages exact versions and passes only the fixed
  primitive manifest digest/count through the guarded worker. JSON contains no role,
  image, executable, profile, path or authority selector.
- Staging walks original components without following symlinks, hashes same-FD bytes
  and rejects extra files, links, provenance/count/schema drift. Preparation accepts
  schema2 canonical `openai.explicit_conversations.v1`/`coding-preparation.v1` or
  exactly `nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2`/
  `coding-preparation.open-swe.v2`, with the source-owned native discriminator.
  No caller profile override or mixed pair is accepted. Held-out test rows never train.
- Derived v2 accepts only the pinned1db Open-SWE v1.2 provenance and unchanged six
  JSON artifacts. REPORTschema2 separates core drops from normalization refusals.
  Original rows/ordinals/artifact versions and hashes remain distinct from normalized
  canonical messages/tools digests; eight-field derived edges always retain
  `derived_not_original_source_verified`. Counts, exact lineage schema, contiguous
  original ordinals, immutable URI/version attribution, one declared group per split,
  strict REPORT integer types, dropped identity-reference nullability, membership,
  split hashes/digests and retained causal edge indices
  reconcile before consumption. Unknown fields, grants, coverage claims, version/hash
  drift or conservation failures refuse even if outer hashes are recomputed.
  Parsing stays in the fixed trainer guest; no Parquet or SDK/parser fallback is added.
  All derived JSON bytes share256MiB aggregate/32MiB row/depth64 bounds. Compatibility
  admission describes verified prepared input, never native authority or trained quality.
- CommitPack accepts only `bigcode.commitpackft.python.change_sft.v1`/
  `coding-preparation.commitpack.v3`, alongside the unchanged existing pairs.
  COMMITTEDschema2 binds the same six files and exact typed eight REPORTschema3
  counters plus closed nine-key coverage. The source-owned pure replay revalidates
  every normalized original, including duplicate and excluded origins, before
  component exclusions and deduplication; complete canonical splits remain messages/tools.
  All declared repositories, commits, SPDX mappings, three digests, transformations,
  original ordinals, raw-line observations, memberships and component split buckets
  reconcile. No resolution, execution or original-source-authentication claim is added.
  Exclusion lists use fixed priority and complete compact UTF8 bytes with one newline;
  the replayed digest equals COMMITTED's independently staged exclusion pin.
  This pair alone permits REPORT up to8MiB+262144 at the same-FD read boundary;
  COMMITTED and legacy metadata retain262144. Before prepared payload reads, typed
  counts/conservation and descriptor cardinalities must agree. Bounded line iteration
  refuses the excess record before decoding, enforces byte/depth bounds through the
  strict object decoder and avoids splitlines allocation. The privately pinned pair
  selects nonblocking same-FD regular-file opens; FIFO/device inputs cannot wait for
  a writer before refusal. Legacy reader defaults and parsers remain unchanged.
  All prepared JSON shares256MiB aggregate limits; incomplete wrappers
  cannot supply split refs or successful input admission. Raw-line hashes/rejected
  bodies cannot be reconstructed from metadata; actual version/hash source binding
  belongs to the trusted parent. Preparation does not qualify tokenizer context,
  unique TRAINtoken floors, native training or quality.
- Rows contain exactly messages/tools. Developer instructions are incompatible and
  refuse with a structured reason; never fold them into system or silently discard.
  `qwen_tool_arguments_v1` validates JSON-string function arguments and normalizes
  them only for official rendering while preserving the original canonical row hash.
- The hash-pinned official template has no generation markers. Source instrumentation
  must render byte-identically for the same normalized row, preserve real token IDs,
  and match actual assistant token spans including reasoning/tool XML. Empty or
  truncated supervision refuses. Padding labels remain -100; max sequence4096.
- Train only language-layer LoRA rank16/alpha32/dropout0; freeze vision and base weights.
  `fit_v1` performs32 actual optimizer updates and proves fit only. `full_v1` requires
 1024 updates, at least1000 unique accepted canonical examples and1M unique supervised
  tokens before any update, plus1M actual optimizer-exposure tokens. Repeated examples
  cannot satisfy unique floors. Validation/early-stop observations never lower targets
  or imply quality; changed adapters, finite losses and timing/memory are measured.
- Historical nonempty reasoning omitted by the official full-row template refuses
  as `historical_reasoning_unsupported`; aggregate counts accompany developer-role
  compatibility refusal without row text. Unique examples and unique supervised
  occurrences share the actual canonical input_ids/attention_mask/labels identity;
  original hashes are separate provenance. Seed12345 precedes LoRA initialization.
- Managed framework inputs retain every source-pinned base file and original directory
  FD through tokenizer/model consumption and final checks. Only fixed source-held
  `/proc/self/fd` directory aliases reach frameworks. Native original ancestors,
  directories and files must be root-owned and unwritable by the dropped worker;
  symlinks, executable files, extra files and changed held bytes/name refuse. Actual
  platform ownership/access qualification is still required; no input repair exists.
- Reload's consumed CHECKPOINT bytes equal the externally pinned digest/count. Require
  the full native-model-derived target/key/shape inventory and model-relative probe
  token IDs. Before tensor materialization, a2MiB bounded safetensor header checks
  dtype/rank/shape, complete contiguous byte spans and fixed metadata; actual loaded
  adapter values must equal the verified state. Serialized targets must equal the
  complete native language Linear full-path inventory or its exact complete suffix
  set emitted by locked PEFT. Reconstruct sorted native full paths for attachment;
  serialized suffixes never select modules. Mixed, partial and foreign selectors
  refuse; tensor keys/shapes and all retained-FD guards remain required.
  CPU controls do not admit a parser.
- Checkpoints contain only `adapter_model.safetensors`, `adapter_config.json` and
  bounded `CHECKPOINT.json`; no optimizer/RNG/pickle objects. Schema is closed, token
  shapes/budget/counters/provenance are bounded, and caller evidence cannot override
  source identity. The managed checkpoint includes actual source/raw RUN manifest pins.
- Framework save uses an already-open output directory; safe reload retains the
  original single-link file FDs, decodes bounded metadata from those FDs and applies
  safetensors through a fixed PEFT config/state API. Rehash, original-name and original
  directory checks surround loading/publication. CHECKPOINT and final manifest writes
  retain their FDs and compare actual bytes after fsync. Final COMMITTED also retains
  and verifies all earlier expected artifacts; it cannot adopt changed current bytes.
  These bounded observations do not promise future immutability or exploit immunity.
- Train emits a closed checkpoint/TRAINING/COMMITTED inventory and reports
  `checkpoint_saved`, `reload_pending=true`, `quality_success=false`. Reload is a
  fixed operation in a fresh same-image process with independently staged pinned base
  and adapter; it requires exact bounded greedy equality and emits RELOAD/COMMITTED.
  A reload receipt proves that operation only; parent native and paired held-out coding
  evaluation are separate. No guest receipt can mint an admission marker.
- `qwen_sft.requirements.lock` is a dedicated hashed56-package candidate; Transformers
 5.20.0/PEFT0.21.2/Tokenizers0.23.1/SafeTensors0.8.0 and Torch2.14.0 are pinned.
  CPU native controls use an isolated overlay plus existing CPU Torch; CUDA/image/GPU
  compatibility requires actual native qualification. No SDK/agent package enters the image.

Production GPU loading requires a verified held input object and refuses an absent
handle before ML imports; no weaker pathname API exists. Critical input verification
precedes final COMMITTED. Failed final publication invalidates only the operation's
original write FD through truncate/fsync; it never unlinks a pathname. A substituted
foreign inode remains untouched, an empty failed marker is unusable, and failure
stays primary. Parent success requires successful native process state and
independent exact-version/hash verification, never marker existence.

- Fixed native operation diagnostics use a closed source-owned stage enum and exception class only. ML-import/GPU/tokenizer/model/optimizer phases retain all existing guards and no retry. Runtime observations expose only bounded numeric Torch/CUDA versions; no raw exception, data/model text, credential or provider value. Diagnostics grant no admission.

- `coding_generation.py` owns only the fixed HumanEval164 public-PROMPTS held-object generation routine. Exact three fields/order/sourceSHA, prompt-authored native chat/termination marker refusal, official template hash, fresh fixed GenerationConfig, greedy seed, total4096 context reserving1024 generation tokens and64KiB retained-source bounds and identical wrapper are required. It imports no loader/SDK/network or oracle fields; caller-held objects and binding hashes are descriptive until the owning native loader verifies them. All164 admitted cases persist with explicit setup/invalid/truncated/incomplete/termination/overflow/error outcomes and bounded stage codes; bounded partial decoded source/hash stays retained and overflow records its observed hash/count. EOS is single tokenizer EOS, terminal only, with cap conservatively truncated; no repair/retry, pass or quality flag exists. Canonical manifest bytes bind fixed contract, public prompts, model/tokenizer/spec and arm identities. Tiny CPU/fake/tokenizer tests do not qualify9B/GPU/native generation. Seeding changes RNG state and eval changes held model mode; this seam is pure with respect to loading/external I/O, not referential state. Framework max tokens and post-return byte checks are not a sandbox around arbitrary held objects.

- Native reload RUNschema1 retains same-source producer/consumer equality.
  Closed12-field schema2 independently binds current baked consumer source and the
  consumed original checkpoint producer, fixed safe-adapter contract and a data-only
  compatibility commitment. Parent fresh server selection and native qualification
  remain independently required; guest metadata/digests mint no admission. The
  original CHECKPOINT schema/provenance is never rewritten. Schema2 receipts retain
  the selected producer/consumer/contract/digest; legacy schema1 bytes stay unchanged.

- `_publish_fixed_json` returns canonical SHA/count descriptors for TRAINING,
  RELOAD and COMMITTED. Actual writer-to-parent roundtrip is verified; final held-FD
  inventory/hash/name checks stay unchanged. Known legacy JSON descriptors are a
  parent compatibility rule, not a writer schema fallback or admission claim.

## Work Guidance

Use red-first regressions and the fixed native class. Isolate input ownership,
production API authority and failed-publication rollback oracles from unrelated refusals. Keep optimal-control evidence
separate from AMLAS's literal single-prompt behavior. Preserve all provider/isolation
refusals and root-owned startup/source/FD boundaries.

## Verification

Run `test_qwen_sft*.py` with the separately locked CPU overlay and five pinned public
metadata files supplied through `AMLAS_QWEN_TEST_METADATA`; never skip the actual
native/tokenizer cases. Run Ruff, strict locked dependency audit, security/public-tree
checks and recursive DOX. Tiny CPU tests and image-source checks do not admit9B/GPU jobs.

## Child DOX Index

None.

- Checkpoint refusal diagnostics carry only closed source-owned substages/reasons
  and the bounded original exception class. Framework exception text is never
  inspected. Output-empty, source/rank/dtype/key and retained-FD integrity guards
  remain unchanged; separate fixed scratch prevents known namespace coupling.
  This source fix does not confirm a historical native failure's exact cause.

- CommitPack full raw replay remains private and completes before TRAINING summary
  compaction. Closed `commitpack_replay_summary_v1` retains exact counts/coverage/
  supervision/exclusion pin plus each split count and domain-separated SHA of the
  original ordered canonical hashes. DomainASCII+NUL+splitASCII+NUL+uint64BEcount
  and every32-byte hash participate; no sorting/truncation or JSON cap increase.
- Post-checkpoint training_receipt/final_readback/final_publication diagnostics
  preserve only fixed stage and bounded class. Partial checkpoint files without
  TRAINING/COMMITTED do not establish completed fit, reload or quality.

- Fixed `generate`/`humaneval_greedy_v1` validates a closed data-only RUN against the current image source, exact base inventory, public PROMPTS and generation contract. Tuned separately pins original checkpoint producer source; it never rewrites or equates that source with the new image. Hold original prompt/base/adapter files through local-only loading, all164 greedy outcomes and final rechecks. CANDIDATES.jsonl has a12MiB aggregate cap; GENERATION/COMMITTED retain262KiB. COMMITTED is last and its held FD is invalidated if final retained input/output verification fails. Receipts keep quality/native admission false; actual GPU and paired execution qualification remain separate.
