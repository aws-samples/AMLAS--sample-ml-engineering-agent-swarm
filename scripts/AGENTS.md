# Deployment Script Instructions

## Purpose

Validate, deploy, and clean up the portable application through explicit cloud selectors.

## Ownership

This directory owns shell entrypoints and the dependency-free deployment ownership ledger.
The root instructions own repository authority and the full verification contract.

## Local Contracts

- `provision_qwen_cpu_tests.sh NEW_OUTPUT_DIRECTORY` provisions the separate hash-locked Python3.11/LinuxAMD64 CPU test closure into a new overlay, fetches only five fixed source-catalog metadata files, and provides the one pinned public evaluation file through `qwen_cpu_test_case_file.py`. Source the emitted `environment.sh` before complete pytest; GitHub transports these fixed variables through `GITHUB_ENV` to its next test step. The downloader reuses guarded public TLS/DNS transport, verifies exact SHA/count and accepts no model/revision/file selector; no tensor weights or repository code are downloaded. CPU dependencies receive a separate strict OSV audit (the official CPU local version is absent from PyPI), without excluding Torch; the GPU/runtime lock is unchanged.

- `qwen_cpu_test_case_file.py --destination ABSOLUTE_FILE` provides the pinned public evaluation file for `AMLAS_TEST_HUMANEVAL_PARQUET`. It parses repository, revision, file name, byte count and SHA256 literals from `src/amlas/data_eng/coding_case_profile.py` at run time without importing the package or duplicating pins. The command line has no source, revision, file or scheme selector. With the variable unset, it uses the Python standard library and the public resolve address. It requests HTTPS only, follows redirects only to HTTPS, reads at most one byte past the pinned size, and makes three attempts for transport faults and temporary server statuses. Each attempt sends a new request object for the source address, because the standard library's proxy handler rewrites the object it sends. A request that no longer names the source host over HTTPS, or an answer whose final address is not HTTPS, refuses. A named verified file is copied without requests. An absent or differing named file refuses. It writes the read-only file only after size and SHA256 match. A failed fetch, a wrong size or hash, or an existing destination with other bytes exits 1 with one `public case file not provided:` line on stderr. A failed fetch or a wrong size or hash writes nothing under the final name. An existing destination with other bytes is left as it is and is never overwritten or removed. `provision_qwen_cpu_tests.sh` runs it after metadata and before writing `environment.sh`. That file adds `export AMLAS_TEST_HUMANEVAL_PARQUET=<NEW_OUTPUT_DIRECTORY>/humaneval/test-00000-of-00001.parquet` on its own line between metadata and offline switches. Other lines remain unchanged. A refusal stops the script before `environment.sh` exists.

- Independently verified synthetic reload fixture checksums receive only exact whole-token integrity handling. The reviewed TRAIN_REQUEST SHA is a complete request-byte checksum; its interior numeric run is never an account exemption. Preserve all six frozen legacy/digest/request values, exact inline nonsecret annotations, altered/unreviewed/fragment/ARN/ECR/bucket/domain negatives and the original secret/private scans. No path or general hash allowlist applies.

- `verify_repository_governance.py` excludes only reviewed complete public model
  revision tokens and21 exact HumanEval metadata SHA256 tokens from numeric
  account detection. Oracle pins are recomputed from the original214438-byte
  public JSONL with source SHA256 `1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2`.
  Source inline secret-scan annotations apply only to independently verified public
  hashes. Exact-line regressions freeze the complete oracle metadata literal and
  reviewed annotations; changed or extra annotated values refuse.
  No path or generic hash exemption applies. Private literals and secrets
  scan original bytes first; fragments, altered pins and separate accounts refuse.
  Complete whitespace-delimited pip `--hash=sha256:`64-hex tokens receive only
  integrity-context numeric handling. Malformed/extended tokens, adjacent account
  values, private hosts and all original secret/private-literal scans stay active;
  this is not a package trust or file exemption.

- `dox_lint.py` is byte-vendored from the canonical DOX lint. `--recursive` enforces maintained nested-directory coverage; it emits `DOX PASS`/`DOX FAIL` and exits0/1. CI invokes it before dependency preparation.

- `lib/deployment_manifest.py` runs with standard-library Python before project dependency
  installation. Its commands are `init`, `refresh`, `plan`, `delete`, and `record-session`.
  Successful commands emit JSON on stdout; failures emit sanitized diagnostics on stderr.
- `init --file PATH --account ACCOUNT --region REGION --prefix PREFIX --stacks NAME...`
  also receives `--secret-id`, `--runtime-name`, and optional `--role-stack`, `--role-name`
  and `--profile`, plus immutable `--ssm-prefix` and `--model-group` selectors.
  `--tool-bin` records the setup-selected CLI directory for noninstalling cleanup lookup. It records native identities before writes and returns the stable UUID
  `deployment_id`. Reruns with the same selectors reuse that manifest and UUID.
- Setup initializes `output/deploy/deployment-manifest.json` after preflight and before
  cloud writes. New resources carry `amlas:deployment-id` and `amlas:deployment-prefix`.
  An occupied mutable target is rejected for every prefix unless the existing manifest proves ownership. Resources
  present before setup are never adopted. Explicit immutable execution-role reuse requires
  the preflight-verified native RoleARN/RoleId witness. Successful compatibility
  preflight emits `RESOLVED_REUSED_RUNTIME_ROLE_ARN` and
  `RESOLVED_REUSED_RUNTIME_ROLE_ID`; setup forwards this exact pair to ledger init.
  Missing, duplicate or wrong-account witness fields fail before initialization. An old schema may upgrade with the same
  UUID only when resources, sessions, stack inventory are empty and all baseline values are null.
- Setup refreshes the ledger after write phases and from its exit trap. Refresh retains
  partial native handles when a later metadata read fails. Missing or foreign ownership
  tags on expected new resources, changed identities, or incomplete metadata verification
  block cleanup until a successful refresh resolves the uncertainty.
- `record-session --file PATH --runtime-id ID --session-id ID [--qualifier DEFAULT]`
  records only sessions of an owned runtime. AgentCore session IDs have 33–256 characters.
  Record the session before invocation and refresh after invocation to capture generated logs.
- Teardown requires `--manifest PATH` or the default setup manifest. `--dry-run` observes
  native metadata and emits the guarded plan without AWS mutations. Before the first
  delete, every owned target must match the account, region, native identity, original
  preexisting flag, and ownership proof. Each mutation rechecks its own target.
- Stop recorded sessions and verify runtime absence. Inventory standalone training and
  processing jobs by atomic UUID/prefix/session creation tags, stop owned active jobs, and
  verify terminal states before supporting object, role, stack, or key cleanup. Native terminal
  job history remains explicitly reported. Prefix alone never owns a managed job. Remove its auxiliary role
  stack and the selected application stacks in reverse order using the recorded native
  StackId and the same verified CloudFormation service RoleARN and RoleId.
  Cleanup never directly creates or deletes IAM roles.
- Inventory retained ECR, S3, DynamoDB, KMS, model groups, SSM parameters and log groups
  through exact CloudFormation/native handles. Empty all S3 versions and delete markers;
  enforce service batch limits. Generated runtime and Lambda logs bind to a recorded
  runtime ID or exact native FunctionPhysicalId in an owned StackId. An explicit foreign
  ownership UUID overrides that derived proof and protects the log group. Snapshot future
  Lambda log namespaces before execution and require creation-incarnation evidence; older
  untagged groups remain protected even when a new function reuses their fixed names.
- Keep encrypted data usable until it has been removed. Schedule owned customer KMS keys
  with the seven-day minimum window and verify `PendingDeletion`. Pending keys remain
  explicitly reported with their actual deletion date; they are not reported as absent.
- Shared bootstrap resources and reused roles remain outside the product cleanup ledger.
  `--nuke-bootstrap` is rejected because those resources are not owned by this manifest.
- AWS execution uses argv and the existing credential chain or selected local profile.
  Do not retrieve secret values or SSM values for ownership discovery, log credentials,
  install dependencies during teardown, or perform account-wide prefix deletion sweeps.

- `lib/write_research_secret.py create|put` consumes JSON stdin and calls the
  native Secrets Manager SDK through the declared project interpreter. It uses
  the exact selected local profile or default chain and an explicit Region.
  Create preserves ownership tags; put preserves the intended secret value.
  Success emits no payload or response; errors return one with a fixed sanitized
  diagnostic. Keys never enter argv, temporary files, dotenv or helper/uv
  environment. Real SDK/profile/stdin checks supplement fake transport tests.

- `preflight_cloudtrail_capacity()` uses the selected account/Region and normal
  AWS CLI wrapper. ListTrails includes shadow trails and must be readable and
  structurally valid. The fixed five-trail limit is checked before new secret or
  infrastructure writes when DetectionsStack is selected. Exact selected-trail
  presence exempts only capacity; ownership remains the ledger's independent
  guard. No automatic trail deletion, quota request or new authentication path.
  Operator metadata permissions include `cloudtrail:ListTrails`.

- `lib/agentcore_runtime_control.py` updates the runtime and verifies its exact
  image, role, environment, network/protocol, platform, version, ownership and MMDSv2.
  DEFAULT is managed by runtime updates; never call UpdateAgentRuntimeEndpoint for
  DEFAULT. Poll its native READY status and exact liveVersion. targetVersion may be
  absent or null for DEFAULT; a present target must still match. Missing/null/wrong
  liveVersion never passes. This exception does not change named-endpoint contracts.
- Setup publishes the verified selected SageMaker role and optional acquisition
  image/source pair through the existing owned SSM publisher only after native
  runtime READY verification. Partial/malformed pairs fail before writes.
- Inspection image/source and trusted admission digest use the selected-namespace
  owned publisher after READY. Ordinary setup requires a supplied matching native
  admission marker when a pair is selected, both before cloud writes and before
  READY discovery publication. Omission never retains an old marker beside a new
  pair. Read-only setup and direct runtime bootstrap may configure an unadmitted
  pair; neither authorizes ordinary inspection publication. Never synthesize admission.

- Preparation configuration is resolved by the operator-only complete-tuple helper
  before runtime writes and transported in the native runtime overlay. Setup publishes
  only the complete verified returned tuple through the existing owned parameter writer.
  Explicit partial overrides never borrow old SSM admission/exclusion fields.

- coding_case_worker_native.py is the root-owned fixed case image wrapper; imports only the minimal case_worker_pkg after static initialization, with no caller paths or script selectors.

- Runtime setup verifies the selected-account EC2/STS location before host
  prerequisites even with `--skip-deps`; actual image construction verifies the
  local Docker binding again. Docker-free preflight, synth and skip-runtime modes
  retain their existing contracts.

- `coding_pair_worker_native.py` is the fixed image-owned wrapper; it verifies native permanent identity and original private outputFD3 before adding the fixed source package path and importing the paired worker main. `paired_worker_requirements.py` selects the reviewed CPU import closure from immutable uv.lock versions/wheel hashes, refuses missing/ambiguous selected packages and has no unpinned or source-build fallback. Its requirements generation/install are image-build operations restricted to the verified AWS builder; unit tests inspect the function in memory.

- Governed setup/runtime deploy resolves the closed model acquisition selector
  tuple through the operator-only helper before writes. Explicit complete ENV
  wins; absent selectors may come only from the selected exact SSM namespace.
  Partial/unverified selections refuse. Transport grants no native admission and
  adds no model parameter publication or TUI startup reads.

## Work Guidance

Read the root instructions before editing. Reproduce lifecycle failures with closed native
AWS state before changing behavior. Keep resource identity and ownership uncertainty visible.
Maintain runtime, CFN actor, retained-resource and partial-failure regression coverage.

## Verification

```bash
python3 -I tests/test_deployment_manifest_cleanup.py
bash -n scripts/teardown.sh
uv run --frozen --extra dev ruff check scripts/lib/deployment_manifest.py tests/test_deployment_manifest_cleanup.py
```

Run the root's full locked suite and source review before live execution. Fixture success
proves the guards against the fake native state; it does not prove live AWS cleanup.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `lib/AGENTS.md` | Maintained lib boundary |
| `ci_shard/AGENTS.md` | Sharded full-CI tools: plan, shard plugin, merge, orchestrator, observer and equality checks |
