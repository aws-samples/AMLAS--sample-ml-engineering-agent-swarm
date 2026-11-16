"""Pure original-conversation startup; selector absence confers no admission."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from ..backends.ports import CodingCaseFreezeBackend
from ..backends.registry import REGISTRY
from ..config.externalized import CONFIG_KEYS, resolve_env_snapshot
from .acquisition_state import SessionAcquisitionStore
from .coding_case_state import (
    CaseFreezeCatalog,
    CaseFreezeConfigurationMissingError,
    SessionCodingCaseFreezeStore,
    validate_catalog,
)


def build_case_freeze_store(
    acquisition_store: SessionAcquisitionStore, *, env: Mapping[str, str] | None = None
) -> SessionCodingCaseFreezeStore:
    """Read ENV only; never call SDK/SSM or reconstruct source authority."""
    source = os.environ if env is None else env
    keys = (
        "aws_account_id",
        "aws_region",
        "session_bucket",
        "sagemaker_execution_role_arn",
        "coding_case_freeze_image_uri",
        "coding_case_freeze_source_commit",
        "coding_case_freeze_admission_digest",
    )
    names = [name for key in keys for name in CONFIG_KEYS[key].env] + [
        "AMLAS_DEPLOYMENT_ID",
        "AMLAS_DEPLOYMENT_PREFIX",
    ]
    if any(type(source.get(name, "")) is not str for name in names):
        raise ValueError("case startup selector type differs")
    values = {key: resolve_env_snapshot(key, env=source) for key in keys}
    image, commit, digest = (
        values["coding_case_freeze_" + name] for name in ("image_uri", "source_commit", "admission_digest")
    )
    if any((image, commit, digest)) and (
        not all((image, commit, digest))
        or not re.fullmatch(
            r"[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9][a-z0-9._/-]*@sha256:"
            + re.escape(digest),
            image,
        )
        or not re.fullmatch("[a-f0-9]{64}", digest)
        or not re.fullmatch("[a-f0-9]{40}", commit)
    ):
        raise ValueError("case immutable selector tuple differs")
    catalog = CaseFreezeCatalog(
        values["aws_account_id"],
        values["aws_region"],
        values["session_bucket"],
        values["sagemaker_execution_role_arn"],
        image,
        commit,
        digest,
        source.get("AMLAS_DEPLOYMENT_ID", "").strip(),
        source.get("AMLAS_DEPLOYMENT_PREFIX", "").strip(),
    )
    if image:
        try:
            validate_catalog(catalog)
        except CaseFreezeConfigurationMissingError:
            # Missing deployment resources stay unconfigured; never infer them.
            pass
    return SessionCodingCaseFreezeStore(
        acquisition_store, REGISTRY.resolve(CodingCaseFreezeBackend).adapter, catalog
    )
