"""Pure selected-environment binding; missing deployment selectors remain unadmitted."""

from __future__ import annotations

import os

from ..backends.ports import CodingGenerationBackend
from ..backends.registry import REGISTRY
from ..config.externalized import resolve_env_snapshot
from .generation_state import SessionCodingGenerationStore
from .sft_state import SftCatalog


def build_generation_store(models, cases, sft, *, env=None):
    source = os.environ if env is None else env

    def value(key):
        return resolve_env_snapshot(key, env=source)

    names = (
        "AMLAS_CODING_GENERATION_IMAGE_URI",
        "AMLAS_CODING_GENERATION_SOURCE_COMMIT",
        "AMLAS_CODING_GENERATION_ADMISSION_DIGEST",
        "AMLAS_DEPLOYMENT_ID",
        "AMLAS_DEPLOYMENT_PREFIX",
    )
    if any(type(source.get(name, "")) is not str for name in names):
        raise ValueError("generation startup selector type differs")
    catalog = SftCatalog(
        value("aws_account_id"),
        value("aws_region"),
        value("session_bucket"),
        value("sagemaker_execution_role_arn"),
        *(source.get(name, "").strip() for name in names),
    )
    return SessionCodingGenerationStore(
        models, cases, sft, REGISTRY.resolve(CodingGenerationBackend).adapter, catalog
    )
