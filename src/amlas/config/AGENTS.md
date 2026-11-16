# Configuration Instructions

## Purpose

Resolve portable runtime, cloud discovery, and deployment naming configuration.

## Ownership

This directory owns the shared selector and resource-default contracts consumed
by shell setup, CDK stacks, runtime discovery, and managed-job callers.

## Local Contracts

- [Added 2026-11-05] `AMLAS_GUARDRAIL_ENABLED` controls guardrail checks at call time independently of a provisioned id. In-band model binding still requires an id. Disabled checks report `action=DISABLED`, `enabled=false`, `passed=false`; they never establish safety-pass evidence.

- `deployment.py` uses only the Python standard library. Running it directly emits
  JSON `deployment_prefix` and `ssm_prefix`; invalid selectors exit with status 2.
  Shell setup must be able to validate names before installing project dependencies.
- `AMLAS_DEPLOYMENT_PREFIX` defaults to `amlas`. A prefix starts with a lowercase
  letter, contains lowercase letters, digits and single separating hyphens, and
  has at most 24 characters. `deployment_stack_name(logical_id, prefix)` retains
  each legacy physical stack name for `amlas` and uses `<prefix>-<logical_id>` otherwise.
- `AMLAS_SSM_PREFIX` defaults to `/<deployment-prefix>`. It is an absolute path
  with non-empty segments containing letters, digits, dots, underscores or hyphens.
  A trailing slash is normalized. Discovery remaps registered `/amlas/` suffixes
  to the selected namespace when called, including harness discovery.
- Selectors and generated resource defaults resolve at call time. Imports must
  never capture the active deployment or override a subsequently loaded runtime overlay.
- Shared dotenv loading removes only empty AWS_PROFILE/AWS_DEFAULT_PROFILE selectors
  before SDK construction so the saved default chain remains usable. Preserve every
  nonempty profile literally, shell precedence, and all other credential-chain fields.
- Explicit config and environment resource overrides retain their precedence.
  Selected-prefix defaults are `<prefix>-models`, `<prefix>/tavily-api-key`, and
  `<prefix-with-underscores>_features`. Secret configuration carries an id only.
- `AMLAS_JOBS_TABLE` selects the managed training state table; legacy `JOBS_TABLE`
  remains a fallback. An explicit caller table wins, otherwise the default is
  `<prefix>-training-jobs`. A submitted backend handle retains its table for status reads.
- Generated training and processing job names start with `<prefix>-`, satisfy the
  SageMaker name grammar, and have at most 63 characters. Keep their names aligned
  with the runtime and backend IAM resource patterns. When `AMLAS_DEPLOYMENT_ID` is configured,
  managed creates atomically include UUID, prefix and runtime-session correlation tags. Validate
  that UUID before any SDK write; preserve the accepted native handle after later tracking or
  describe failure. Legacy SDK callers without a deployment UUID remain unowned.
- The default managed training adapter resolves its execution role and session
  bucket from the runtime environment when no explicit adapter/resource value is supplied.
- `externalized.py` also resolves `sagemaker_execution_role_arn` from the existing
  execution-role environment aliases or selected SSM binding; unresolved remains empty.
- `DATA_ACQUISITION_BACKEND` selects Processing by default or explicit local/offline
  execution. `AMLAS_DATA_ACQUISITION_IMAGE_URI` requires an ECR digest, and
  `AMLAS_DATA_ACQUISITION_SOURCE_COMMIT` names its published source commit.
  Missing managed acquisition bindings never become placeholder resources.
- Acquisition image/source discover through selected SSM suffixes as well as
  explicit environment selectors; configured env values retain precedence.
- Fixed inspection resolves `AMLAS_DATA_INSPECTION_IMAGE_URI`,
  `AMLAS_DATA_INSPECTION_SOURCE_COMMIT` and `AMLAS_DATA_INSPECTION_ADMISSION_DIGEST`
  through env/selected SSM. Missing/stale admission fails before ordinary writes;
  root/operator native probes establish the exact-image admission marker.
- Model defaults and admission are shared by chat/role selectors, CDK and shell
  preflight/IAM generation. Canonical, legacy and explicit overrides retain precedence.
  Direct Bedrock consumers use the shared sampling/cache compatibility policy.
- Execution roles and chat default to Sonnet 5.5; the independent reviewer defaults
  to GPT-6.1 Sol. GPT requests omit explicit Converse cache points and sampling.

- `resolve_env_snapshot` reads only explicit ENV aliases, without SSM, local profile
  discovery, safe resource defaults or required-value coercion. This is the startup
  snapshot seam for a typed UNADMITTED preparation catalog.
- Preparation registers `AMLAS_DATA_PREPARATION_IMAGE_URI`, `SOURCE_COMMIT`,
  `ADMISSION_DIGEST`, `EXCLUSIONS_URI`, `EXCLUSIONS_VERSION_ID`, `EXCLUSIONS_SHA256`,
  and `EXCLUSIONS_BYTE_COUNT` under the common DATA_PREPARATION prefix and matching
  selected SSM suffixes. None has a synthetic image, marker or exclusions default.

- Model acquisition registers IMAGE_URI/SOURCE_COMMIT/ADMISSION_DIGEST under
  AMLAS_MODEL_WEIGHT_ACQUISITION and matching selected SSM suffixes, with no
  resource/marker defaults. model_weight_acquisition_selectors is an ENV-only
  startup snapshot: all absent is unconfigured, partial/malformed/foreign or
  digest-mismatched tuples refuse. A matching tuple remains UNADMITTED/descriptive;
  no SDK/SSM call or readiness grant occurs at startup.

- Qwen SFT registers IMAGE_URI/SOURCE_COMMIT/ADMISSION_DIGEST under AMLAS_QWEN_SFT and selected SSM suffixes. ENV-only startup snapshots carry no synthetic admission or required resource defaults.

## Work Guidance

Read the repository instructions first. Preserve the default `amlas` regressions
and reproduce selected-deployment failures before behavior changes. Keep naming
validation dependency-free and keep AWS reads behind the existing injectable seams.

## Verification

Run the focused client namespace and existing regional boundary tests:

```bash
uv run --frozen --extra dev pytest -q tests/test_client_deployment_namespace.py tests/test_region_client_metadata.py
```

Run the repository-wide verification required by the root instructions before review.
Fixture success does not prove deployment or live managed invocation acceptance.

## Child DOX Index

None.
