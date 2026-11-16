"""Adapter registry bootstrap — importing this package self-registers every adapter.

The :class:`~amlas.backends.registry.BackendRegistry` lazily does
``from . import adapters`` on first ``resolve``; importing each module below runs
its ``register_adapter(...)`` calls (the strangler self-registration, doc 05 §9.3).

Import order is irrelevant (registration is idempotent). The M0 set:
  * **Real wrapping adapters** for the existing ports (default AWS + local):
    training, evaluation (Arbiter), lake, registry_be, monitoring, explainability,
    featurestore, pipeline, tracking.
  * **Milestone stubs** for the remaining ports (``stubs``) — Protocol defined,
    behavior raises ``NotImplementedError`` with the target-milestone TODO.

Each import is wrapped so a single adapter module's optional-dependency import error
never breaks the whole seam (the registry logs + continues; a missing adapter then
fails fast only when that specific port is resolved).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Real wrapping adapters (the 8 existing ports + the existing local leaderboard +
# the M1 unified OrchestrationBackend LocalDriver).
_REAL_MODULES = (
    "training",
    "evaluation",
    "lake",
    "registry_be",
    "monitoring",
    "explainability",
    "featurestore",
    "pipeline",
    "tracking",
    "orchestration",  # M1 — LocalDriverBackend (replaces the OrchestrationBackend stub)
    "dataquality",  # M2 — InProcessDQ / DataBrew / GlueDataQuality (replaces the DQ stub)
    "etl",  # M2 — InProcessPandas / ProcessingJob / GlueETL (replaces the ETL stub)
    "dataset_acquisition",  # Pinned source acquisition: Processing / explicit local
    "coding_case_registration",  # Fixed data-only case freeze; no local fallback
    "preparation_registration",  # Managed only; no local parser fallback
    "model_registration",  # Fixed catalog-only managed model acquisition
    "coding_generation_registration",
    "coding_pair_registration",
    "qwen_sft_registration",  # Fixed managed SFT/reload; no local fallback
    "dataset_inspection",  # Fixed isolated inspection: managed only
    "tuning",  # M3 — OptunaBackend / SageMakerAMTBackend (replaces the Tuning stub)
    "automl",  # M3 — LocalSweepBackend / AutopilotV2Backend (replaces the AutoML stub)
    "deployment",  # M4 — LocalServeBackend / SageMakerDeploymentBackend (the priority GAP, ADR-007)
    "governance",  # M4 — Local/SageMaker Governance + InProcess/Bedrock Guardrail (readiness gates)
    "retrieval",  # M6 — LocalRetrievalBackend / BedrockKnowledgeBasesBackend (RetrievalBackend, ADR-017, 20th port)
    "modelhub",  # M6 — LocalModelHubBackend / JumpStartBedrockModelHubBackend (replaces the ModelHub stub)
    "query",  # M6 — DuckDBBackend / AthenaBackend (replaces the Query stub)
    "labeling",  # M6 — StubLabelingBackend / GroundTruthBackend (replaces the Labeling stub)
    "bedrock_alt",  # M8 — BedrockFineTuneBackend (Training) / BedrockHostedBackend (Deployment): portability-swap alts (ADR-005)
    "orchestration_pipeline",  # M8 — SageMakerPipelineOrchestrationBackend: exploration→production graduation (ADR-011)
)


def _safe_import(name: str) -> None:
    try:
        __import__(f"{__name__}.{name}", fromlist=["*"])
    except Exception as e:
        logger.warning("backends.adapters: failed to import %s: %s", name, e)


for _m in _REAL_MODULES:
    _safe_import(_m)

# Milestone stubs LAST so a real adapter always wins a duplicate selector (the stub
# uses generic "default"/"local" names that never collide with the real adapters'
# named selectors, but importing last keeps the intent explicit).
_safe_import("stubs")
