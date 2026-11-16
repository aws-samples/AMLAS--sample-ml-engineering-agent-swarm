# scripts/lib

## Purpose

Own the maintained scripts/lib boundary and its checked-in inputs and outputs.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Runtime checkpoint bindings `AMLAS_RUNTIME_EXEC_ROLE_ARN` and
  `AMLAS_RUNTIME_EXEC_ROLE_ID` come only from the verified native execution-role
  readback on both new and reused paths. Missing native RoleId refuses managed
  runtime writes. Ambient selectors never override them; the selected SageMaker
  job role remains the distinct `SAGEMAKER_EXECUTION_ROLE_ARN` binding.
- Runtime checkpoint table metadata reads use separate `DdbJobsMetadataRead`:
  only `dynamodb:DescribeTable` on the exact selected jobs-table ARN, without
  index or wildcard scope. Existing role compatibility remains fail-closed;
  deployment never changes reused trust or grants to satisfy the new policy.

- Runtime CASE ownership-tag reads use a separate `SageMakerProcessingTagRead`
  statement: only `sagemaker:ListTags` on the selected deployment prefix processing-job
  ARN. Preserve existing AddTags actions/resources, read/stop scopes and conditions.

- [Added 2026-11-11] Runtime fine-tune, reload and generation ownership-tag reads use a
  separate `SageMakerTrainingTagRead` statement: only `sagemaker:ListTags` on the selected
  deployment prefix training-job ARN. `SageMakerQwenSftBackend.status()` and the restore path
  make that read after the request readback; without the grant every observation of a healthy
  training job is UNKNOWN and the wait closes at its cap (first real normal-path run, 2026-11-11).
  A role created before this statement lacks it: reused-role compatibility stays fail-closed, so
  such a role is replaced or granted the statement by its owner, never by the deploy script.

- Selected model profiles and Bedrock invocation resources come from the shared
  configuration catalog. Unadmitted profiles fail preflight; selected reused-role
  compatibility remains fail-closed after catalog changes.
- Acquisition image/source selectors are optional together and immutable when
  configured; validate same-account/Region digest plus source commit before
  unrelated writes. Runtime JSON retains explicit backend and deployment-owned
  remote-fetch/host policy without developer profiles or credentials.
- Inspection reuses immutable pair validation. A pair without an admission
  marker is permitted for bootstrap; ordinary inspection still denies. A supplied
  marker must equal the configured image digest, comes only from the trusted
  deployer and is never inferred or synthesized. Runtime/READY output preserves
  exactly the supplied selectors. `validate_inspection_publication_config` requires
  a supplied matching marker for a selected pair before ordinary setup writes and
  post-READY owned publication; an entirely unconfigured pair remains valid.
  Direct runtime bootstrap retains pair-without-marker support and publishes no SSM.
- [Added 2026-11-10] `deploy_agentcore_runtime.sh` passes `AMLAS_RUNTIME_RESEARCH_PROFILE` and `AMLAS_GUARDRAIL_ENABLED` into the runtime environment when they are set, and refuses before any write a profile other than `qwen9b_core_v1` or `off` and a guardrail switch other than `true` or `false`. Unset keys stay absent. Why: the managed completion path and the pause witness switch on only with the research profile, and the deployed configuration runs with guardrail checks off; a runtime built by this script had neither.
- Selected SageMaker roles are never replaced silently. Normal runtime deploy
  reads native ARN and service trust before writes; initial setup may provision
  its default role in selected stacks before that read.
- Runtime READY output supplies exact selected role/acquisition bindings for
  ownership-guarded setup publication. S3 version reads remain scoped to the
  selected session bucket and objects.
- `agentcore_runtime_control.py:RuntimeDeployError` — A required control operation or readback failed.
- [Added 2026-11-10] `agentcore_runtime_control.py:wire_environment` measures the runtime
  environment as AgentCore does (sum of key and value lengths; refused above 4,000 bytes) before
  any AgentCore call. A selection that fits is sent as plain keys, so older images and rollbacks
  run it unchanged. A larger one is packed losslessly by `src/amlas/_runtime_env_pack.py` (loaded by
  path) and sent only with `AMLAS_RUNTIME_IMAGE_EXPANDS_ENV_PACK=1`, which `setup.sh` sets for the
  image it built from this tree. Refusals name the byte count, the limit and the three largest keys,
  never values. Readback compares the packed map; output adds `RUNTIME_ENVIRONMENT_BYTES` and
  `RUNTIME_ENVIRONMENT_PACKED`. Why: UpdateAgentRuntime refused the full fresh-deployment selection
  (4,097 bytes measured; 4,508 with all seven preparation values).
- [Added 2026-11-10] `deploy_common.sh:publish_preparation_selection` publishes the verified
  preparation selection after READY: nothing, the triple, or all seven; the exclusions parameters
  only when selected, since SSM stores no empty value. A partial selection publishes nothing.
- `deployment_manifest.py:LedgerError` — An identity or ownership gate failed; cleanup must stop.
- `write_research_secret.py:write_secret` — Use the selected profile or default credential chain and explicit region.
- Owned files: `agentcore_runtime_control.py`, `build_runtime_image.sh`, `deploy_agentcore_runtime.sh`, `deploy_common.sh`, `deployment_manifest.py`, `prepare_host.sh`, `account_image_build.py`, `require_account_build_host.sh`, `write_research_secret.py`, `preparation_release_config.py`, `model_acquisition_release_config.py`.

- `preparation_release_config.py` is operator-only stdlib configuration binding.
  [Changed 2026-11-10] Resolve none, the resource triple (image, source commit,
  admission digest) or all seven selectors from explicit ENV or the selected qualified
  SSM namespace; never mix new ENV image with old SSM admission. With the triple the
  runtime prepares only on a same-conversation frozen case (`case_request_ref`), which
  supplies the governed exclusions; a triple-only ENV selection refuses while any of the
  four exclusions parameters remains in SSM. Entirely absent stays
  unconfigured; partial/malformed/foreign/read-error responses refuse before writes.
  Native GetParameter Name must exactly match; missing parameters alone are absent.
  Runtime overlay carries the verified tuple; TUI construction makes no SSM call and
  the runtime IAM policy is unchanged. Post-READY owned setup publication validates
  the complete returned tuple without rediscovery or synthetic admission markers.

- `model_acquisition_release_config.py` is the operator-only three-selector model
  transport. Exact selected namespace/Region parameter names fill ENV only when
  all selectors are absent; explicit complete ENV takes precedence without SSM.
  Missing parameters require an anchored AWS CLI GetParameter ParameterNotFound
  operation error after leading ASCII whitespace framing; unrelated or embedded
  errors mentioning that text remain refusals. JSON
  duplicate keys and nonfinite values, including float overflow, refuse.
  Successful SSM responses with empty/whitespace values refuse; only a genuine
  missing-parameter operation error represents absence. ENV-empty conventions
  remain unchanged.
  Partial ENV never mixes with stored values. Missing partial, foreign, denied,
  mutable or digest-mismatched selections refuse before deployment writes.
  The existing explicit marker is preserved, never inferred or created. Entirely
  absent remains unconfigured; startup stays ENV-only and UNADMITTED. This bridge
  adds no READY echo, SSM publication, TUI persistence, IAM or native admission.

- `require_account_build_host.sh` invokes stdlib `account_image_build.py` with the
  explicit target account and Region before image construction or packaging.
  Linux EC2 IMDSv2 identity and fresh STS must agree with those targets; the full
  check additionally requires the default local Unix Docker socket, same-host
  Linux daemon and running default Docker buildx driver. Environment overrides,
  remote contexts/drivers and unavailable or malformed native evidence refuse.
  `--location-only` permits prerequisite preparation after EC2/STS verification
  and returns no Docker authorization. The runtime helper rechecks the full
  binding before ECR login/build and scopes short-lived Docker auth to workspace
  tmp. Require successful mktemp, a nonempty usable real directory and mode700
  before exporting the isolated Docker config or attempting ECR authentication.
  When host preparation placed buildx only in the previous workspace
  config, copy just its already-guarded executable into the isolated auth config;
  never copy shared authentication or configuration JSON. System-installed plugins
  retain normal discovery. Cloud Desktop and Mac image builds/exports are unsupported. These checks
  do not provision builders, change IAM, create credentials or grant publication.

- `build_coding_pair_image.sh` verifies the selected EC2 account/Region/local daemon through the existing account guard, native x86 architecture and exact clean source commit before any build. It loads one fixed AMD64 review artifact and never pushes. Runtime ENV forwarding uses an explicit lifecycle allowlist for model/case/SFT/generation/pair selectors, including the trusted AMLAS_CODING_PAIR_LAUNCHER_SHA256, and explicit `AMLAS_RUNTIME_APPROVED_ACTIONS`; absent authority remains absent and public payloads cannot supply it.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
