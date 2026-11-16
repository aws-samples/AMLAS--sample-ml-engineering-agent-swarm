"""Pure startup ENV snapshot; missing preparation resources remain UNADMITTED."""

from __future__ import annotations

import os
from collections.abc import Mapping

from ..backends.ports import CodingPreparationBackend
from ..backends.registry import REGISTRY
from ..backends.types import DatasetArtifactRef
from ..config.externalized import resolve_env_snapshot
from .acquisition_state import SessionAcquisitionStore
from .preparation_state import PreparationCatalog, SessionPreparationStore


def build_preparation_store(
    acquisition_store: SessionAcquisitionStore, *, case_store=None, env: Mapping[str, str] | None = None
) -> SessionPreparationStore:
    """Bind a conversation without SDK/SSM reads, grants or synthetic resources."""
    source = os.environ if env is None else env

    def value(key: str) -> str:
        return resolve_env_snapshot(key, env=source)

    text = value("data_preparation_exclusions_byte_count")
    try:
        count = int(text) if text else 0
    except ValueError:
        count = -1
    catalog = PreparationCatalog(
        account_id=value("aws_account_id"),
        region=value("aws_region"),
        bucket=value("session_bucket"),
        execution_role_arn=value("sagemaker_execution_role_arn"),
        image_uri=value("data_preparation_image_uri"),
        source_commit=value("data_preparation_source_commit"),
        admission_digest=value("data_preparation_admission_digest"),
        deployment_id=source.get("AMLAS_DEPLOYMENT_ID", "").strip(),
        deployment_prefix=source.get("AMLAS_DEPLOYMENT_PREFIX", "").strip(),
        exclusions_ref=DatasetArtifactRef(
            uri=value("data_preparation_exclusions_uri"),
            version_id=value("data_preparation_exclusions_version_id"),
            sha256=value("data_preparation_exclusions_sha256"),
            byte_count=count,
            format="json",
        ),
    )
    return SessionPreparationStore(
        acquisition_store, REGISTRY.resolve(CodingPreparationBackend).adapter, catalog, case_store=case_store
    )
