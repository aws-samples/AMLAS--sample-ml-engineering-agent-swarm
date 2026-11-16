"""Pure SFT startup binding; no SDK, source observation or native admission."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping

from ..backends.ports import QwenSftBackend
from ..backends.registry import REGISTRY
from ..config.externalized import CONFIG_KEYS, resolve_env_snapshot
from .model_acquisition_state import SessionModelAcquisitionStore
from .preparation_state import SessionPreparationStore
from .sft_state import (
    ReloadCompatibilitySelection,
    SessionSftStore,
    SftCatalog,
    VerifiedSftCheckpointSource,
    _seal_reload_compatibility_catalog,
)


def build_sft_store(
    model_store: SessionModelAcquisitionStore,
    preparation_store: SessionPreparationStore,
    *,
    env: Mapping[str, str] | None = None,
    campaign_state=None,
    execution_binding=None,
) -> SessionSftStore:
    """Reuse the exact conversation stores; missing selectors remain UNADMITTED."""
    source = os.environ if env is None else env
    common = ("aws_account_id", "aws_region", "session_bucket", "sagemaker_execution_role_arn")
    fixed = (
        "AMLAS_QWEN_SFT_IMAGE_URI",
        "AMLAS_QWEN_SFT_SOURCE_COMMIT",
        "AMLAS_QWEN_SFT_ADMISSION_DIGEST",
        "AMLAS_DEPLOYMENT_ID",
        "AMLAS_DEPLOYMENT_PREFIX",
    )
    names = (*fixed, *(name for key in common for name in CONFIG_KEYS[key].env))
    if any(type(source.get(name, "")) is not str for name in names):
        raise ValueError("SFT startup selector type differs")

    def value(key: str) -> str:
        return resolve_env_snapshot(key, env=source)

    catalog = SftCatalog(
        account_id=value("aws_account_id"),
        region=value("aws_region"),
        bucket=value("session_bucket"),
        execution_role_arn=value("sagemaker_execution_role_arn"),
        image_uri=source.get("AMLAS_QWEN_SFT_IMAGE_URI", "").strip(),
        source_commit=source.get("AMLAS_QWEN_SFT_SOURCE_COMMIT", "").strip(),
        admission_digest=source.get("AMLAS_QWEN_SFT_ADMISSION_DIGEST", "").strip(),
        deployment_id=source.get("AMLAS_DEPLOYMENT_ID", "").strip(),
        deployment_prefix=source.get("AMLAS_DEPLOYMENT_PREFIX", "").strip(),
    )
    return SessionSftStore(
        model_store,
        preparation_store,
        REGISTRY.resolve(QwenSftBackend).adapter,
        catalog,
        campaign_state=campaign_state,
        execution_binding=execution_binding,
        reload_compatibility_reader=lambda: resolve_reload_compatibility_catalog(env=source, catalog=catalog),
    )


def resolve_reload_compatibility_catalog(*, env: Mapping[str, str], catalog: SftCatalog):
    """Re-read explicit trusted startup configuration; no SDK or native approval."""
    from ..backends.types import DatasetArtifactRef
    from ..data_eng.inspection_schema import decode_object

    raw = env.get("AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG", "")
    if type(raw) is not str or len(raw.encode()) > 65536:
        raise ValueError("reload compatibility startup bound differs")
    if not raw:
        return _seal_reload_compatibility_catalog(())
    value = decode_object(raw.encode(), limit=65536)
    if (
        set(value) != {"schema_version", "selections"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["selections"]) is not list
        or len(value["selections"]) > 16
    ):
        raise ValueError("closed reload compatibility startup fields required")

    def dto(cls, data):
        if type(data) is not dict or set(data) != {f.name for f in dataclasses.fields(cls)}:
            raise ValueError("closed reload compatibility DTO required")
        return cls(**data)

    selections = []
    for data in value["selections"]:
        if type(data) is not dict or set(data) != {
            f.name for f in dataclasses.fields(ReloadCompatibilitySelection)
        }:
            raise ValueError("closed reload compatibility selection required")
        data = dict(data)
        data["producer_catalog"] = dto(SftCatalog, data["producer_catalog"])
        data["consumer_catalog"] = dto(SftCatalog, data["consumer_catalog"])
        checkpoint = dict(data["checkpoint"]) if type(data["checkpoint"]) is dict else None
        if checkpoint is None or set(checkpoint) != {
            f.name for f in dataclasses.fields(VerifiedSftCheckpointSource)
        }:
            raise ValueError("closed original compatibility checkpoint required")
        checkpoint["checkpoint_ref"] = dto(DatasetArtifactRef, checkpoint["checkpoint_ref"])
        pairs = checkpoint["file_refs"]
        if (
            type(pairs) is not list
            or len(pairs) != 3
            or any(type(p) is not list or len(p) != 2 or type(p[0]) is not str for p in pairs)
        ):
            raise ValueError("closed compatibility file inventory required")
        checkpoint["file_refs"] = tuple((name, dto(DatasetArtifactRef, ref)) for name, ref in pairs)
        data["checkpoint"] = dto(VerifiedSftCheckpointSource, checkpoint)
        selection = ReloadCompatibilitySelection(**data)
        if selection.consumer_catalog != catalog:
            raise ValueError("current consumer catalog differs from compatibility startup")
        selections.append(selection)
    return _seal_reload_compatibility_catalog(tuple(selections))
