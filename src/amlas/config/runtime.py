"""Runtime configuration — model IDs, region, profile, retry + cache settings.

Pure data module; no boto3 / strands imports so it stays directly verifiable without AWS creds.
The agent factory (`_build.py`) consumes these values to construct the real clients.
"""
# NOTE: all Anthropic Claude model IDs below are accessed exclusively through
# Amazon Bedrock (cross-region inference profiles) — never Anthropic's own API.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# OSS-4b P3: the single env back-compat seam. Canonical env names are ``AMLAS_*``;
# the helper transparently falls back to the legacy ``AMLAS_*``/``AGENTICML_*``
# names (one-time deprecation) so existing deployments keep working for one release.
from ._env_compat import env_with_compat
from .aws_region import resolve_aws_profile, resolve_aws_region

# --- Model IDs (Bedrock global cross-region inference profiles) ---
# Verified live in us-west-2 via `aws bedrock list-inference-profiles`
# (2026-06-04: global.anthropic.claude-opus-4-8 + ...-sonnet-4-6 confirmed present).
# Opus/Sonnet drop the date suffix; Haiku keeps the 20251001-v1:0 suffix.
OPUS_4_8 = "global.anthropic.claude-opus-4-8"  # latest Opus — ML-work swarm default
OPUS_4_7 = "global.anthropic.claude-opus-4-7"
SONNET_4_6 = "global.anthropic.claude-sonnet-4-6"  # TUI NL chat default
HAIKU_4_5 = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# 1M-context beta header — attached via BedrockModel.additional_request_fields
# to ALL Sonnet 4.x + Opus 4.x calls (user directive 2026-06-04: "1 million
# context endpoints on Bedrock for all: sonnet + opus"). The legacy name is kept
# as an alias so existing imports continue to resolve.
CONTEXT_1M_BETA = "context-1m-2025-08-07"
OPUS_4_7_CONTEXT_1M_BETA = CONTEXT_1M_BETA  # backward-compat alias

# Canonical set of Bedrock inference profiles the agents may invoke. This is the
# SINGLE source of truth for the IAM allowlist (CDK ``_bedrock_resources``) AND
# the per-role model tiers — keeping the deployed runtime's
# ``bedrock:InvokeModel*`` Resource list in lockstep with whatever models the
# swarm actually uses, so adding a tier (e.g. Opus 4.8) can never silently
# AccessDenied in-cloud. Opus 4.8 is the ML-work default; 4.7 retained for
# back-compat/rollback; Sonnet 4.6 powers the TUI chat; Haiku 4.5 the reviewer.
BEDROCK_INVOKE_PROFILES: tuple[str, ...] = (OPUS_4_8, OPUS_4_7, SONNET_4_6, HAIKU_4_5)

# Default region & profile
DEFAULT_REGION: None = None
DEFAULT_PROFILE: str | None = None

# Default budgets (may be overridden per-spec)
DEFAULT_MAX_RETREATS = 3
DEFAULT_WALL_CLOCK_MINUTES = 30
DEFAULT_TOOL_CALL_LIMIT = 200
DEFAULT_TRIALS = 50

# --- Backend-port selectors (the `*_BACKEND` env keys; doc 05 §4.3/§8.5) ---
# The `BackendRegistry` (backends/registry.py) reads these `*_BACKEND` env vars to
# pin a port → adapter (precedence: hard-steering pin → explicit env → auto). The
# PROD DEFAULTS below preserve today's behavior (SageMaker/AWS where the behavior
# exists, in-process for the rest) so a normal launch is unchanged.
# AMLAS_BACKENDS_OFFLINE makes the registry's auto path lean to the LOCAL adapter
# and avoid AWS. These constants are the single source the
# `.env.*.example` templates mirror; they are advisory documentation of the keys —
# the registry reads ``os.environ`` directly so a layered ``.env`` overlay wins.
BACKEND_ENV_KEYS: tuple[str, ...] = (
    "TRAINING_BACKEND",
    "TUNING_BACKEND",
    "AUTOML_BACKEND",
    "MODELHUB_BACKEND",
    "ETL_BACKEND",
    "DQ_BACKEND",
    "FEATURE_STORE_BACKEND",
    "LAKE_BACKEND",
    "QUERY_BACKEND",
    "LABELING_BACKEND",
    "DEPLOYMENT_BACKEND",
    "EXPLAINABILITY_BACKEND",
    "EVALUATION_BACKEND",
    "MODEL_REGISTRY_BACKEND",
    "MONITORING_BACKEND",
    "EXPERIMENT_TRACKING_BACKEND",
    "PIPELINE_BACKEND",
    "ORCHESTRATION_BACKEND",
    "GOVERNANCE_BACKEND",
    "GUARDRAIL_BACKEND",
)

# Prod-default adapter selector per port (doc 05 §8.5 table). SageMaker/AWS where
# the behavior exists today; in-process/local for the M0-local ports. `auto` lets
# the registry pick by scale/budget/offline (these are the no-pin defaults).
DEFAULT_BACKEND_SELECTORS: dict[str, str] = {
    "TRAINING_BACKEND": "sagemaker",
    "EVALUATION_BACKEND": "arbiter",  # the deterministic Arbiter (ADR-003)
    "MODEL_REGISTRY_BACKEND": "sagemaker",
    "LAKE_BACKEND": "s3medallion",
    "PIPELINE_BACKEND": "sagemaker",
    "MONITORING_BACKEND": "sagemaker",
    "EXPLAINABILITY_BACKEND": "clarify",
    "FEATURE_STORE_BACKEND": "sagemaker",
    "EXPERIMENT_TRACKING_BACKEND": "local",  # local leaderboard
}

# Retry defaults (boto3 adaptive mode + Strands ModelRetryStrategy alignment)
DEFAULT_BOTO_READ_TIMEOUT = 900  # 15 min; some long sklearn fits need this
DEFAULT_BOTO_MAX_ATTEMPTS = 6
# PERF-02: connection-pool sizing for boto3 clients. urllib3's default pool is 10,
# which caps how many concurrent Bedrock/tool calls a single client can have
# in-flight (the agent loop + any ThreadPool fan-out). 50 unblocks parallel calls
# without being so large it exhausts file descriptors. Override per-env via
# AMLAS_BOTO_MAX_POOL. Pure pool sizing — request/response bytes are identical.
DEFAULT_BOTO_MAX_POOL_CONNECTIONS = 50

ModelRole = Literal["planner", "reviewer", "specialist", "bounded", "finetune"]


@dataclass(frozen=True)
class ModelConfig:
    """Per-role model selection. Mixed-model routing from AutoKaggle + ADSA patterns.

    User directive (2026-06-04): "Opus 4.8 for all agents in the swarm related to
    ML work." The ML-doing roles — planner (Research Lead, Leads, Algorithm-
    Specialist, FineTune), specialist (feature/algo hypothesis), bounded
    (TrainerTuner, VisionFE), and finetune — now default to Opus 4.8. The
    ``reviewer`` stays on the cheap Haiku tier (a meta critic, not ML work). The
    TUI NL chat is separately pinned to Sonnet 4.6 via ``AMLAS_CHAT_MODEL``.
    """

    planner: str = OPUS_4_8  # Research Lead, Leads, AlgorithmSpecialist, FineTune
    reviewer: str = HAIKU_4_5  # Phase reviewers, DataIntegrity critic (cheap meta tier)
    specialist: str = OPUS_4_8  # feature/algo hypothesis
    bounded: str = OPUS_4_8  # TrainerTuner, VisionFE (ML work → Opus 4.8)
    finetune: str = OPUS_4_8

    def for_role(self, role: ModelRole) -> str:
        return str(getattr(self, role))


@dataclass(frozen=True)
class BedrockConfig:
    region: str | None = None
    profile: str | None = None
    read_timeout: int = DEFAULT_BOTO_READ_TIMEOUT
    max_attempts: int = DEFAULT_BOTO_MAX_ATTEMPTS
    # PERF-02: urllib3 connection-pool size for the BedrockModel boto client.
    # Raised above urllib3's default (10) so a single client can keep many
    # concurrent Bedrock/tool calls in-flight. Pure pool sizing — output unchanged.
    max_pool_connections: int = DEFAULT_BOTO_MAX_POOL_CONNECTIONS
    retry_mode: Literal["legacy", "standard", "adaptive"] = "adaptive"
    cache_prompt: Literal["default", "off"] = "default"
    cache_tools: Literal["default", "off"] = "default"
    max_tokens: int = 64000
    # NOTE: Opus 4.7 has deprecated `temperature` — do NOT pass it in additional_request_fields.
    # Sonnet 4.6 and Haiku 4.5 still accept it. Agent factory sets temperature per-role.
    temperature: float | None = None
    # Anthropic beta headers to attach — 1M ctx on Opus 4.7 by default
    anthropic_betas: tuple[str, ...] = (OPUS_4_7_CONTEXT_1M_BETA,)


@dataclass(frozen=True)
class Budgets:
    max_retreats: int = DEFAULT_MAX_RETREATS
    wall_clock_minutes: int = DEFAULT_WALL_CLOCK_MINUTES
    tool_calls: int = DEFAULT_TOOL_CALL_LIMIT
    trials: int = DEFAULT_TRIALS


@dataclass(frozen=True)
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    budgets: Budgets = field(default_factory=Budgets)


def load_config() -> Config:
    """Resolve config from env vars with sensible defaults.

    Env overrides (OSS-4b P3 canonical ``AMLAS_*`` names; legacy ``AMLAS_*``
    still honored for one release via :func:`env_with_compat`):
        AMLAS_AWS_PROFILE, AMLAS_AWS_REGION
        AMLAS_MODEL_PLANNER, AMLAS_MODEL_REVIEWER, AMLAS_MODEL_BOUNDED
        AMLAS_MAX_RETREATS, AMLAS_WALL_CLOCK_MIN, AMLAS_TOOL_CALL_LIMIT
    """
    return Config(
        models=ModelConfig(
            planner=env_with_compat("AMLAS_MODEL_PLANNER", OPUS_4_7),
            reviewer=env_with_compat("AMLAS_MODEL_REVIEWER", HAIKU_4_5),
            specialist=env_with_compat("AMLAS_MODEL_SPECIALIST", OPUS_4_7),
            bounded=env_with_compat("AMLAS_MODEL_BOUNDED", SONNET_4_6),
            finetune=env_with_compat("AMLAS_MODEL_FINETUNE", OPUS_4_7),
        ),
        bedrock=BedrockConfig(
            region=resolve_aws_region(profile_name=resolve_aws_profile(), required=False),
            profile=resolve_aws_profile(),
            # PERF-02: env-overridable connection-pool size (default 50).
            max_pool_connections=int(
                env_with_compat("AMLAS_BOTO_MAX_POOL", str(DEFAULT_BOTO_MAX_POOL_CONNECTIONS))
            ),
        ),
        budgets=Budgets(
            max_retreats=int(env_with_compat("AMLAS_MAX_RETREATS", str(DEFAULT_MAX_RETREATS))),
            wall_clock_minutes=int(env_with_compat("AMLAS_WALL_CLOCK_MIN", str(DEFAULT_WALL_CLOCK_MINUTES))),
            tool_calls=int(env_with_compat("AMLAS_TOOL_CALL_LIMIT", str(DEFAULT_TOOL_CALL_LIMIT))),
            trials=int(env_with_compat("AMLAS_TRIALS", str(DEFAULT_TRIALS))),
        ),
    )
