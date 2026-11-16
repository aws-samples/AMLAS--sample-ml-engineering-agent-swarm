"""Unified config-externalization model (OSS-4b P2, plan §6).

This is the SINGLE documented seam through which an open-source adopter supplies
every account/org-specific value. Each value resolves with a fixed precedence:

    1. Environment variable(s)   — highest priority; the first non-empty wins.
    2. SSM Parameter Store        — stack-published outputs under ``/amlas/*``.
    3. Safe default               — a NON-secret, NON-account default, or ``None``
                                     for REQUIRED values that must be supplied.

NO account/org value is hardcoded in this package. A REQUIRED value that resolves
to empty raises :class:`MissingRequiredConfigError` — it never silently falls back
to a baked-in literal (the SEC-A01 anti-pattern).

The env names below keep the generic legacy ``AGENTICML_`` prefix as a
back-compat fallback plus the canonical ``AMLAS_`` name
(listed first so it wins). The package/env rename to a single ``AMLAS_`` prefix
(P3) and the SSM namespace move to ``/amlas`` (P4) are complete; this module is
the seam those phases collapsed onto. Secret *values* (e.g. the Tavily API key) live in
Secrets Manager (see :mod:`..cloud.secrets`); this layer carries only the secret
*id*, never the value.

The model unifies the seams already added in WAFR/SEC remediation rather than
duplicating them — guardrail id/version (:mod:`.guardrails`), budget caps and
runtime/network/egress knobs (control_plane / tools), region (:mod:`.runtime`),
and the SSM-published lake/registry/session outputs.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .aws_region import AWS_REGION_ENV_KEYS, resolve_aws_region
from .deployment import namespace_parameter_name, resolve_deployment_prefix

# SSM namespace the CDK stacks publish under (de-branded in OSS-4b P4).
SSM_NAMESPACE = "/amlas"


class MissingRequiredConfigError(RuntimeError):
    """Raised when a REQUIRED externalized value resolves to empty.

    The whole point of fail-loud: a required account/org value that an adopter
    forgot to supply must surface immediately, NOT silently resolve to a
    hardcoded fallback (which is exactly the leakage OSS-4b removes).
    """


@dataclass(frozen=True)
class ConfigKey:
    """A single externalized configuration value and how to resolve it.

    Attributes:
        env: ordered env-var names checked first→last; the first non-empty wins.
            The forward-looking ``AMLAS_*`` name is listed first, the legacy
            ``AGENTICML_`` or AWS-standard names after for back-compat.
        ssm: SSM Parameter Store key (under :data:`SSM_NAMESPACE`) checked when no
            env var is set; ``None`` if the value is not SSM-published.
        default: safe NON-secret, NON-account fallback when neither env nor SSM
            resolves; ``None`` means "no default" (REQUIRED keys must set
            ``required=True`` and leave this ``None``).
        required: when True, an empty resolution raises
            :class:`MissingRequiredConfigError` instead of returning ``""``.
        secret: True if the value is secret material — by convention only the
            secret *id* is stored here; the value is fetched from Secrets Manager.
        description: human description for the ``.env*.example`` templates / docs.
    """

    env: tuple[str, ...]
    ssm: str | None = None
    default: str | None = None
    required: bool = False
    secret: bool = False
    description: str = ""


# Type of an injectable SSM getter: (ssm_key, region, client=None) -> value|"".
SsmGetter = Callable[..., str]


def _is_placeholder(value: str) -> bool:
    """True if ``value`` is an UNFILLED ``<...>`` template placeholder (FIX-12).

    A ``.env.example`` copied verbatim leaves values like ``<SESSION_BUCKET>`` or
    ``<AWS_ACCOUNT_ID>`` in the environment. These are NOT real values and must
    never shadow a stack-published SSM parameter, so callers treat them as unset.
    """
    v = (value or "").strip()
    return "<" in v and ">" in v


def _default_ssm_getter(key: str, region: str, client: object | None = None) -> str:
    """Best-effort SSM read; returns '' on any failure (offline-safe, no raise).

    boto3 is imported lazily so this module stays import-safe without AWS creds
    (the offline unit suite never touches AWS; callers may inject a fake getter).
    """
    try:  # pragma: no cover - exercised only in live/integration paths
        if client is None:
            import boto3
            from botocore.config import Config

            # FIX-24: BOUND the SSM read. This resolver runs synchronously on the chat
            # worker thread (via dispatch's ``resolve_runtime_arn``/``resolve_session_bucket``),
            # so an un-timed ``get_parameter`` against a slow/unreachable SSM (or a hanging
            # SSO/STS credential resolve) parks the whole turn for the default ~60s x retries —
            # the operator sees the thought-ticker "loop" while the worker is blocked. A tight
            # timeout + single attempt means an unavailable SSM degrades to '' fast, so the
            # SEC-01 fail-fast fires cleanly instead of hanging. Env still wins over SSM (this
            # path only runs when no env value is set), so a configured deploy never pays it.
            client = boto3.client(
                "ssm",
                region_name=resolve_aws_region(region),
                config=Config(connect_timeout=2, read_timeout=3, retries={"max_attempts": 1}),
            )
        resp = client.get_parameter(Name=key)  # type: ignore[union-attr]
        return str(resp["Parameter"]["Value"])
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# The externalized-value registry (plan §6.2 value→mechanism mapping).
#
# Env names: AMLAS_* first (canonical), then generic AGENTICML_/AWS
# names for back-compat. SSM keys under the /amlas namespace.
# No default embeds an account id or a live bucket.
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_KEYS: dict[str, ConfigKey] = {
    # ── Identity / region ────────────────────────────────────────────────────
    "aws_account_id": ConfigKey(
        env=("AMLAS_AWS_ACCOUNT_ID", "AWS_ACCOUNT_ID"),
        ssm=None,
        default=None,
        required=True,  # fail loud — never a hardcoded account id
        description="12-digit AWS account id of the deployment account (REQUIRED).",
    ),
    "aws_region": ConfigKey(
        env=AWS_REGION_ENV_KEYS,
        ssm=None,
        default=None,
        description="Selected AWS target region; no package-wide region default.",
    ),
    "kms_key_arn": ConfigKey(
        env=("AMLAS_KMS_KEY_ARN",),
        ssm=f"{SSM_NAMESPACE}/kms-key-arn",
        default="",  # empty → per-stack CMK is created by CDK (WAFR seam)
        description="Optional BYO KMS key ARN; empty creates a per-stack CMK.",
    ),
    # ── Lake / data ──────────────────────────────────────────────────────────
    "lake_bucket": ConfigKey(
        env=("AMLAS_LAKE_BUCKET", "AGENTICML_LAKE_BUCKET"),
        ssm=f"{SSM_NAMESPACE}/lake-bucket",
        default="",  # required at runtime for lake ops; supplied by adopter
        description="Medallion lake S3 bucket name (SSM-published by LakeStack).",
    ),
    "lake_silver_prefix": ConfigKey(
        env=("AMLAS_LAKE_SILVER_PREFIX", "AGENTICML_LAKE_SILVER_PREFIX"),
        ssm=f"{SSM_NAMESPACE}/lake-silver-prefix",
        default="",
        description="Silver-tier S3 prefix, e.g. s3://<bucket>/silver/.",
    ),
    "lake_gold_prefix": ConfigKey(
        env=("AMLAS_LAKE_GOLD_PREFIX", "AGENTICML_LAKE_GOLD_PREFIX"),
        ssm=f"{SSM_NAMESPACE}/lake-gold-prefix",
        default="",
        description="Gold-tier S3 prefix, e.g. s3://<bucket>/gold/.",
    ),
    "glue_database": ConfigKey(
        env=("AMLAS_GLUE_DATABASE", "AGENTICML_GLUE_DATABASE"),
        ssm=f"{SSM_NAMESPACE}/glue-database",
        default="",
        description="Glue Data Catalog database for the lake.",
    ),
    # ── MLOps (SageMaker) ────────────────────────────────────────────────────
    "model_package_group": ConfigKey(
        env=("AMLAS_MODEL_PACKAGE_GROUP", "AGENTICML_MODEL_PACKAGE_GROUP"),
        ssm=f"{SSM_NAMESPACE}/model-package-group",
        default="amlas-models",  # resource NAME, not account-specific
        description="SageMaker Model Registry package group name.",
    ),
    "feature_store_db": ConfigKey(
        env=("AMLAS_FEATURE_STORE_DB", "AGENTICML_FEATURE_STORE_DB"),
        ssm=f"{SSM_NAMESPACE}/feature-store-db",
        default="amlas_features",  # resource NAME, not account-specific
        description="SageMaker Feature Store offline database name.",
    ),
    "athena_workgroup": ConfigKey(
        env=("AMLAS_ATHENA_WORKGROUP", "AGENTICML_ATHENA_WORKGROUP"),
        ssm=f"{SSM_NAMESPACE}/athena-workgroup",
        default="primary",  # AWS default workgroup — non-account-specific
        description="Athena workgroup for feature-store point-in-time queries.",
    ),
    # ── Runtime / dispatch ───────────────────────────────────────────────────
    "coding_case_freeze_image_uri": ConfigKey(
        env=("AMLAS_CODING_CASE_FREEZE_IMAGE_URI",),
        ssm=f"{SSM_NAMESPACE}/coding-case-freeze-image-uri",
        default="",
        description="Immutable selected managed case-freeze image; no default.",
    ),
    "coding_case_freeze_source_commit": ConfigKey(
        env=("AMLAS_CODING_CASE_FREEZE_SOURCE_COMMIT",),
        ssm=f"{SSM_NAMESPACE}/coding-case-freeze-source-commit",
        default="",
        description="Published fixed case-freeze source commit; no default.",
    ),
    "coding_case_freeze_admission_digest": ConfigKey(
        env=("AMLAS_CODING_CASE_FREEZE_ADMISSION_DIGEST",),
        ssm=f"{SSM_NAMESPACE}/coding-case-freeze-admission-digest",
        default="",
        description="Trusted exact-image case-freeze qualification digest; no default.",
    ),
    "qwen_sft_image_uri": ConfigKey(
        env=("AMLAS_QWEN_SFT_IMAGE_URI",),
        ssm=f"{SSM_NAMESPACE}/qwen-sft-image-uri",
        default="",
        description="Immutable selected fixed managed Qwen SFT image; no admission default.",
    ),
    "qwen_sft_source_commit": ConfigKey(
        env=("AMLAS_QWEN_SFT_SOURCE_COMMIT",),
        ssm=f"{SSM_NAMESPACE}/qwen-sft-source-commit",
        default="",
        description="Published fixed Qwen SFT source commit; no default.",
    ),
    "qwen_sft_admission_digest": ConfigKey(
        env=("AMLAS_QWEN_SFT_ADMISSION_DIGEST",),
        ssm=f"{SSM_NAMESPACE}/qwen-sft-admission-digest",
        default="",
        description="Trusted exact-image native Qwen SFT admission digest; no default.",
    ),
    "data_preparation_image_uri": ConfigKey(
        env=("AMLAS_DATA_PREPARATION_IMAGE_URI",),
        ssm=f"{SSM_NAMESPACE}/data-preparation-image-uri",
        default="",
        description="Immutable selected managed canonical-preparation image.",
    ),
    "data_preparation_source_commit": ConfigKey(
        env=("AMLAS_DATA_PREPARATION_SOURCE_COMMIT",),
        ssm=f"{SSM_NAMESPACE}/data-preparation-source-commit",
        default="",
        description="Published preparation source commit.",
    ),
    "data_preparation_admission_digest": ConfigKey(
        env=("AMLAS_DATA_PREPARATION_ADMISSION_DIGEST",),
        ssm=f"{SSM_NAMESPACE}/data-preparation-admission-digest",
        default="",
        description="Trusted exact-image native admission digest; no default.",
    ),
    "data_preparation_exclusions_uri": ConfigKey(
        env=("AMLAS_DATA_PREPARATION_EXCLUSIONS_URI",),
        ssm=f"{SSM_NAMESPACE}/data-preparation-exclusions-uri",
        default="",
        description="Selected exclusions S3 object; version/hash bound.",
    ),
    "data_preparation_exclusions_version_id": ConfigKey(
        env=("AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID",),
        ssm=f"{SSM_NAMESPACE}/data-preparation-exclusions-version-id",
        default="",
        description="Exact non-null exclusions S3 VersionId.",
    ),
    "data_preparation_exclusions_sha256": ConfigKey(
        env=("AMLAS_DATA_PREPARATION_EXCLUSIONS_SHA256",),
        ssm=f"{SSM_NAMESPACE}/data-preparation-exclusions-sha256",
        default="",
        description="Independently verified exclusions bytes SHA256.",
    ),
    "data_preparation_exclusions_byte_count": ConfigKey(
        env=("AMLAS_DATA_PREPARATION_EXCLUSIONS_BYTE_COUNT",),
        ssm=f"{SSM_NAMESPACE}/data-preparation-exclusions-byte-count",
        default="",
        description="Exact bounded exclusions byte count.",
    ),
    "model_weight_acquisition_image_uri": ConfigKey(
        env=("AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI",),
        ssm=f"{SSM_NAMESPACE}/model-weight-acquisition-image-uri",
        default="",
        description="Selected fixed model acquisition image uri; no default or admission inference.",
    ),
    "model_weight_acquisition_source_commit": ConfigKey(
        env=("AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT",),
        ssm=f"{SSM_NAMESPACE}/model-weight-acquisition-source-commit",
        default="",
        description="Selected fixed model acquisition source commit; no default or admission inference.",
    ),
    "model_weight_acquisition_admission_digest": ConfigKey(
        env=("AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST",),
        ssm=f"{SSM_NAMESPACE}/model-weight-acquisition-admission-digest",
        default="",
        description="Selected fixed model acquisition admission digest; no default or admission inference.",
    ),
    "data_acquisition_image_uri": ConfigKey(
        env=("AMLAS_DATA_ACQUISITION_IMAGE_URI",),
        ssm=f"{SSM_NAMESPACE}/data-acquisition-image-uri",
        default="",
        description="Digest-pinned ECR image for managed dataset acquisition.",
    ),
    "data_acquisition_source_commit": ConfigKey(
        env=("AMLAS_DATA_ACQUISITION_SOURCE_COMMIT",),
        ssm=f"{SSM_NAMESPACE}/data-acquisition-source-commit",
        default="",
        description="Source commit verified when publishing the acquisition image.",
    ),
    "data_inspection_image_uri": ConfigKey(
        env=("AMLAS_DATA_INSPECTION_IMAGE_URI",),
        ssm=f"{SSM_NAMESPACE}/data-inspection-image-uri",
        default="",
        description="Digest-pinned AMD64 image for fixed isolated inspection.",
    ),
    "data_inspection_source_commit": ConfigKey(
        env=("AMLAS_DATA_INSPECTION_SOURCE_COMMIT",),
        ssm=f"{SSM_NAMESPACE}/data-inspection-source-commit",
        default="",
        description="Published source commit compiled into the inspection image.",
    ),
    "data_inspection_admission_digest": ConfigKey(
        env=("AMLAS_DATA_INSPECTION_ADMISSION_DIGEST",),
        ssm=f"{SSM_NAMESPACE}/data-inspection-admission-digest",
        default="",
        description="Image digest admitted after exact-image native credential/egress probes.",
    ),
    "sagemaker_execution_role_arn": ConfigKey(
        env=("SAGEMAKER_EXECUTION_ROLE_ARN", "AMLAS_SAGEMAKER_ROLE_ARN"),
        ssm=f"{SSM_NAMESPACE}/sagemaker-execution-role-arn",
        default="",
        description="Selected managed-training execution role; never invent a role.",
    ),
    "session_bucket": ConfigKey(
        env=("AMLAS_SESSION_BUCKET", "SESSION_BUCKET"),
        ssm=f"{SSM_NAMESPACE}/session-bucket",
        default="",  # supplied by adopter for cloud dispatch
        description="S3 bucket for AgentCore cloud-dispatch session payloads.",
    ),
    "agentcore_runtime_arn": ConfigKey(
        env=("AMLAS_AGENTCORE_RUNTIME_ARN", "AGENTCORE_RUNTIME_ARN"),
        ssm=f"{SSM_NAMESPACE}/agentcore-runtime-arn",
        default="",
        description="AgentCore runtime ARN for cloud dispatch.",
    ),
    "runtime_ecr_uri": ConfigKey(
        env=("AMLAS_RUNTIME_ECR_URI",),
        ssm=f"{SSM_NAMESPACE}/runtime-ecr-uri",
        default="",
        description="ECR image URI for the AgentCore runtime container.",
    ),
    # ── Guardrails (WAFR WA-02 seam — unified, not duplicated) ───────────────
    "guardrail_id": ConfigKey(
        env=("AMLAS_GUARDRAIL_ID",),
        ssm=f"{SSM_NAMESPACE}/guardrail-id",
        default="",  # empty → guardrails NO-OP (offline-safe default)
        description="Bedrock Guardrail id; empty disables in-band guardrails.",
    ),
    "guardrail_version": ConfigKey(
        env=("AMLAS_GUARDRAIL_VERSION",),
        ssm=f"{SSM_NAMESPACE}/guardrail-version",
        default="DRAFT",
        description="Bedrock Guardrail version (default DRAFT).",
    ),
    # ── Networking (VPC) ─────────────────────────────────────────────────────
    "vpc_id": ConfigKey(
        env=("AMLAS_VPC_ID",),
        ssm=f"{SSM_NAMESPACE}/vpc-id",
        default="",
        description="VPC id for private-mode runtime/training networking.",
    ),
    "private_subnet_ids": ConfigKey(
        env=("AMLAS_PRIVATE_SUBNET_IDS",),
        ssm=f"{SSM_NAMESPACE}/private-subnet-ids",
        default="",
        description="Comma-separated private subnet ids for VPC networking.",
    ),
    # ── Secrets (id only — value lives in Secrets Manager) ───────────────────
    "tavily_secret_id": ConfigKey(
        env=("AMLAS_TAVILY_SECRET_ID", "TAVILY_SECRET_ID"),
        ssm=f"{SSM_NAMESPACE}/tavily-secret-id",
        default="amlas/tavily-api-key",  # secret NAME, not the value
        secret=True,
        description="Secrets Manager secret id holding the Tavily API key (id only).",
    ),
}


def resolve(
    name: str,
    *,
    region: str | None = None,
    ssm_getter: SsmGetter | None = None,
) -> str:
    """Resolve a single externalized config value: env → SSM → default.

    Args:
        name: a key in :data:`CONFIG_KEYS`.
        region: AWS region for the SSM read (defaults to the resolved aws_region).
        ssm_getter: injectable ``(key, region, client=None) -> str`` for isolated validation;
            defaults to a best-effort boto3 SSM read that returns '' offline.

    Returns:
        The resolved value (possibly '' for non-required keys with an empty default).

    Raises:
        KeyError: if ``name`` is not a known config key.
        MissingRequiredConfigError: if a REQUIRED key resolves to empty.
    """
    key = CONFIG_KEYS[name]
    if name == "aws_region":
        return resolve_aws_region(region, required=False) or ""

    # 1. Environment (first non-empty wins).
    #    FIX-12 (generalized): an UNFILLED ``<PLACEHOLDER>``-shaped value (e.g. a
    #    ``.env.example`` line copied verbatim like ``AMLAS_SESSION_BUCKET=<SESSION_BUCKET>``
    #    or an ARN-shaped ``<AWS_ACCOUNT_ID>``) must NEVER shadow a real
    #    stack-published SSM value. Angle-bracket placeholders are treated as
    #    unset so resolution falls through to SSM — closing the FIX-08 trap class
    #    for ALL keys, not just the runtime ARN.
    for env_name in key.env:
        val = os.environ.get(env_name, "").strip()
        if val and not _is_placeholder(val):
            return val

    # 2. SSM Parameter Store (when published).
    if key.ssm:
        getter = ssm_getter or _default_ssm_getter
        eff_region = region or _resolve_region(ssm_getter=ssm_getter)
        ssm_val = (getter(namespace_parameter_name(key.ssm), eff_region) or "").strip()
        if ssm_val:
            return ssm_val

    # 3. Safe default / fail-loud for required.
    if key.default is not None and key.default != "":
        prefix = resolve_deployment_prefix()
        namespace_defaults = {
            "model_package_group": f"{prefix}-models",
            "feature_store_db": f"{prefix.replace('-', '_')}_features",
            "tavily_secret_id": f"{prefix}/tavily-api-key",
        }
        return namespace_defaults.get(name, key.default)
    if key.required:
        raise MissingRequiredConfigError(
            f"required config '{name}' is unset: supply one of env {list(key.env)}"
            + (f" or SSM '{namespace_parameter_name(key.ssm)}'" if key.ssm else "")
            + " (no hardcoded fallback exists by design)."
        )
    return key.default or ""


def _resolve_region(*, ssm_getter: SsmGetter | None = None) -> str:
    """Resolve the region without recursing into SSM (region itself is env/default)."""
    return resolve_aws_region(required=False) or ""


def resolve_all(*, ssm_getter: SsmGetter | None = None) -> dict[str, str]:
    """Resolve every NON-required key (best-effort) into a flat dict for inspection.

    Required keys are skipped when unset (so this never raises) — call
    :func:`resolve` directly for a required value to get fail-loud behavior.
    """
    out: dict[str, str] = {}
    region = _resolve_region(ssm_getter=ssm_getter)
    for name, key in CONFIG_KEYS.items():
        if key.required:
            # Only include if actually resolvable from env (avoid raising here).
            for env_name in key.env:
                v = os.environ.get(env_name, "").strip()
                if v:
                    out[name] = v
                    break
            continue
        out[name] = resolve(name, region=region, ssm_getter=ssm_getter)
    return out


def resolve_env_snapshot(key: str, *, env: Mapping[str, str] | None = None) -> str:
    """Read explicit startup ENV/overlay only, never SSM/profile/default resources."""
    source = os.environ if env is None else env
    for name in CONFIG_KEYS[key].env:
        value = source.get(name, "").strip()
        if value and not _is_placeholder(value):
            return value
    return ""


def model_weight_acquisition_selectors(*, env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Pure startup snapshot; configured selectors never establish native admission."""
    selected = os.environ if env is None else env
    names = (
        "AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI",
        "AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT",
        "AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST",
    )
    values = tuple(selected.get(name, "") for name in names)
    if any(type(value) is not str for value in values):
        raise ValueError("model acquisition startup selector type differs")
    values = tuple(value.strip() for value in values)
    if not any(values):
        return {"state": "unconfigured", "native_admitted": False}
    if not all(values):
        raise ValueError("model acquisition startup selector tuple is incomplete")
    image, source, marker = values
    matched = re.fullmatch(
        r"([0-9]{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com/[a-z0-9][a-z0-9._/-]*@sha256:([a-f0-9]{64})",
        image,
    )
    account = selected.get("AMLAS_AWS_ACCOUNT_ID", selected.get("AWS_ACCOUNT_ID", ""))
    region = next((selected.get(name, "") for name in AWS_REGION_ENV_KEYS if selected.get(name, "")), "")
    if (
        not matched
        or not re.fullmatch(r"[a-f0-9]{40}", source)
        or not re.fullmatch(r"[a-f0-9]{64}", marker)
        or marker != matched[3]
        or not re.fullmatch(r"[0-9]{12}", account)
        or account == "0" * 12
        or matched[1] != account
        or matched[2] != region
    ):
        raise ValueError("model acquisition startup selector binding differs")
    return {
        "state": "unadmitted",
        "native_admitted": False,
        "image_uri": image,
        "source_commit": source,
        "admission_digest": marker,
    }
