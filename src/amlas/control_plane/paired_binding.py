"""Pure paired startup selector snapshot over original private conversation stores."""

from __future__ import annotations

import os
import re

from ..backends.ports import PairedCodingBackend
from ..backends.registry import REGISTRY
from ..config.externalized import resolve_env_snapshot
from .paired_state import PairedCodingCatalog, SessionPairedCodingStore
from .sft_state import SftCatalog


def build_pair_store(generation, cases, *, env=None):
    source = os.environ if env is None else env
    names = (
        "AMLAS_CODING_PAIR_IMAGE_URI",
        "AMLAS_CODING_PAIR_SOURCE_COMMIT",
        "AMLAS_CODING_PAIR_ADMISSION_DIGEST",
        "AMLAS_DEPLOYMENT_ID",
        "AMLAS_DEPLOYMENT_PREFIX",
    )
    launcher_key = "AMLAS_CODING_PAIR_LAUNCHER_SHA256"
    if any(type(source.get(name, "")) is not str for name in (*names, launcher_key)):
        raise ValueError("paired startup selector type differs")

    launcher_sha = source.get(launcher_key, "").strip()
    if launcher_sha and not re.fullmatch("[a-f0-9]{64}", launcher_sha):
        raise ValueError("paired startup launcher digest differs")

    def value(key):
        return resolve_env_snapshot(key, env=source)

    catalog = SftCatalog(
        value("aws_account_id"),
        value("aws_region"),
        value("session_bucket"),
        value("sagemaker_execution_role_arn"),
        *(source.get(name, "").strip() for name in names),
    )
    return SessionPairedCodingStore(
        generation,
        cases,
        REGISTRY.resolve(PairedCodingBackend).adapter,
        PairedCodingCatalog(catalog, launcher_sha),
    )
