"""Pure preparation source descriptors; configured shape conveys no admission."""

from __future__ import annotations

import re
from typing import Any, Literal

from ..data_eng import commitpack_preparation, commitpack_profile, hub_dataset
from ..sagemaker_entrypoints.coding_preparation import (
    DERIVED_POLICY_VERSION,
    DERIVED_PROFILE_ID,
    POLICY_VERSION,
    PROFILE_ID,
)
from .preparation_state import (
    PreparationCatalog,
    PreparationConfigurationMissingError,
    _validate_catalog_resources,
    validate_catalog,
)

ConfigurationStatus = Literal["configured", "unconfigured", "unknown"]
QueryStatus = Literal["not_queried", "incomplete", "invalid_input", "source_shape_supported", "unsupported"]
_OPEN_REPO = "nvidia/Open-SWE-Traces"
_OPEN_REVISION = "1db674173119d71df45c22007af99e214600711a"  # pragma: allowlist secret - public revision
_OPEN_FILE = (
    r"data/(openhands|sweagent|minisweagent)/(qwen38_27b|qwen38_flash_next)/"
    r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\.parquet"
)
_REQUIRED_FACTS = (
    "same_conversation_verified_acquisition",
    "exact_object_version_sha256_byte_count",
    "original_receipt_commit_lineage_bundle",
    "original_governed_exclusions",
    "strict_profile_row_message_license_and_split_validation",
    "separate_native_image_privilege_and_permission_admission",
)


def _configuration(catalog: Any, *, exclusions_limit: int) -> tuple[ConfigurationStatus, list[str]]:
    if type(catalog) is not PreparationCatalog:
        return "unknown", []
    try:
        validate_catalog(catalog, exclusions_limit=exclusions_limit)
    except PreparationConfigurationMissingError as error:
        return "unconfigured", list(error.missing_selectors)
    except (ValueError, TypeError, AttributeError):
        return "unknown", []
    return "configured", []


def _resource_configuration(catalog: Any) -> tuple[ConfigurationStatus, list[str]]:
    if type(catalog) is not PreparationCatalog:
        return "unknown", []
    try:
        _validate_catalog_resources(catalog)
    except PreparationConfigurationMissingError as error:
        return "unconfigured", list(error.missing_selectors)
    except (ValueError, TypeError, AttributeError):
        return "unknown", []
    return "configured", []


def _profiles(catalog: Any) -> list[dict[str, Any]]:
    profiles = [
        {
            "profile_id": PROFILE_ID,
            "policy_version": POLICY_VERSION,
            "format": "jsonl",
            "public_source_requirements": {
                "repo_id": "pinned_public_acquisition_except_recognized_commitpack",
                "revision": "exact_lowercase_40_character_commit",
                "filename": "confined_relative_path_ending_jsonl",
                "row_schema": "explicit_canonical_messages_tools_and_source_metadata",
            },
            "coverage_limits": ["canonical_metadata_and_exact_grouping_only", "native_readiness_unverified"],
        },
        {
            "profile_id": DERIVED_PROFILE_ID,
            "policy_version": DERIVED_POLICY_VERSION,
            "format": "parquet",
            "public_source_requirements": {
                "repo_id": _OPEN_REPO,
                "revision": _OPEN_REVISION,
                "filename_pattern": _OPEN_FILE,
                "row_schema": "fixed_open_swe_trajectory_profile",
            },
            "coverage_limits": ["derived_singleton_identity", "original_repository_linkage_unverified"],
        },
        {
            "profile_id": commitpack_preparation.PROFILE_ID,
            "policy_version": commitpack_preparation.POLICY_VERSION,
            "format": "jsonl",
            "public_source_requirements": {
                "repo_id": commitpack_profile.DATASET_REPO,
                "revision": commitpack_profile.REVISION,
                "filename": commitpack_profile.FILENAME,
                "artifact_sha256": commitpack_profile.ARTIFACT_SHA256,
                "artifact_byte_count": commitpack_preparation.SOURCE_BYTES,
                "card_sha256": commitpack_profile.CARD_SHA256,
                "row_schema": "fixed_python_human_commit_change_supervision",
            },
            "coverage_limits": ["all_declared_repository_affiliations", "near_duplicates_not_implemented"],
        },
    ]
    for profile in profiles:
        limit = (
            commitpack_preparation.MAX_EXCLUSIONS_BYTES
            if profile["profile_id"] == commitpack_preparation.PROFILE_ID
            else 2 * 1024 * 1024
        )
        status, missing = _configuration(catalog, exclusions_limit=limit)
        profile.update(
            configuration_status=status,
            configuration_scope="worker_resources_and_global_exclusions",
            missing_selectors=missing,
            required_verification_facts=list(_REQUIRED_FACTS),
        )
    return profiles


def _valid_query_part(name: str, value: Any) -> bool:
    if type(value) is not str or not value or any(ord(char) < 32 for char in value):
        return False
    if name == "revision":
        return re.fullmatch(r"[a-f0-9]{40}", value) is not None
    if len(value) > (193 if name == "repo_id" else 1024):
        return False
    try:
        if name == "repo_id":
            hub_dataset._repository(value)
        else:
            hub_dataset._filename(value)
    except hub_dataset.HubDatasetError:
        return False
    return True


def _query(repo_id: Any, revision: Any, filename: Any) -> tuple[QueryStatus, str | None, str | None]:
    values = {"repo_id": repo_id, "revision": revision, "filename": filename}
    if all(value is None for value in values.values()):
        return "not_queried", None, None
    if any(value is not None and not _valid_query_part(name, value) for name, value in values.items()):
        return "invalid_input", None, "invalid_source_query"
    if any(value is None for value in values.values()):
        return "incomplete", None, "complete_source_query_required"
    if repo_id == commitpack_profile.DATASET_REPO:
        if (revision, filename) == (commitpack_profile.REVISION, commitpack_profile.FILENAME):
            return "source_shape_supported", commitpack_preparation.PROFILE_ID, None
        return "unsupported", None, "recognized_source_pin_mismatch"
    if filename.endswith(".jsonl"):
        return "source_shape_supported", PROFILE_ID, None
    if (
        repo_id == _OPEN_REPO
        and revision == _OPEN_REVISION
        and re.fullmatch(_OPEN_FILE, filename)
        and ".." not in filename
    ):
        return "source_shape_supported", DERIVED_PROFILE_ID, None
    if repo_id == "SWE-bench/SWE-smith-trajectories" and filename.endswith(".parquet"):
        return "unsupported", None, "source_profile_missing"
    return "unsupported", None, "source_shape_not_supported"


def describe_preparation_capabilities(
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    filename: str | None = None,
    catalog: PreparationCatalog | None = None,
) -> dict[str, Any]:
    """Describe current source shapes using held catalog DATA, with no native reads.

    A matched shape still requires actual source bytes and all existing source
    validators. Catalog validation reports selected configuration shape only.
    """
    status, matched, reason = _query(repo_id, revision, filename)
    resource_status, missing_resources = _resource_configuration(catalog)
    return {
        "schema_version": 1,
        "data_only": True,
        "query_status": status,
        "matched_profile_id": matched,
        "reason": reason,
        "missing_requirements": (
            [
                "trajectory_format_profile",
                "original_task_repository_revision_license_provenance",
                "bounded_complete_cohort_accounting",
            ]
            if reason == "source_profile_missing"
            else []
        ),
        "profiles": _profiles(catalog),
        "resource_configuration_status": resource_status,
        "missing_resource_selectors": missing_resources,
        "exclusions_requirement": "configured_global_exclusions_or_verified_original_case_request_ref",
        "case_request_ref_route": {
            "supported": True,
            "tool": "prepare_coding_dataset",
            "argument": "case_request_ref",
            "case_readiness": "unknown",
            "required_provenance": "original_same_conversation_frozen_case_and_verified_EXCLUSIONS_COMMITTED",  # pragma: allowlist secret - public provenance label
            "normal_tool_order": ["freeze_coding_cases", "get_coding_case_freeze", "prepare_coding_dataset"],
        },
        "public_preparation_selector": {"profile_id": PROFILE_ID, "profile_version": "1"},
        "source_artifact_bytes_limit": 512 * 1024 * 1024,
        "decoded_selection_bytes_limit": 256 * 1024 * 1024,
        "max_acquisition_refs": 256,
        "mixed_profile_selection_supported": False,
        "readiness": "unknown",
        "native_admission": False,
        "executed": False,
        "prepared": False,
        "trained": False,
    }
