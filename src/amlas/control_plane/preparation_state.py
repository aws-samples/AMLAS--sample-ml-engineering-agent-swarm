"""Private preparation source/handle types; no caller grants or durability claims."""

from __future__ import annotations

import copy
import re
import threading
import uuid
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from ..backends.types import (
    DatasetArtifactRef,
    DatasetInspectionSource,
    JobHandle,
    JobStatus,
    VerifiedPreparationSource,
)
from .acquisition_state import SessionAcquisitionStore


@dataclass(frozen=True)
class PreparationCatalog:
    account_id: str
    region: str
    bucket: str
    execution_role_arn: str
    image_uri: str
    source_commit: str
    admission_digest: str
    deployment_id: str
    deployment_prefix: str
    exclusions_ref: DatasetArtifactRef
    instance_type: str = "ml.m5.xlarge"
    volume_size_gb: int = 30
    max_runtime_seconds: int = 3600


@runtime_checkable
class CodingPreparationBackend(Protocol):
    backend_name: str

    def submit(
        self,
        *,
        sources: tuple[VerifiedPreparationSource, ...],
        catalog: PreparationCatalog,
        runtime_session_id: str,
    ) -> JobHandle: ...
    def status(self, handle: JobHandle) -> JobStatus: ...
    def artifacts(self, handle: JobHandle) -> dict[str, Any]: ...


class PreparationConfigurationMissingError(ValueError):
    """Only absent trusted startup selectors; contains bounded names, no values."""

    def __init__(self, missing_selectors: tuple[str, ...]) -> None:
        self.missing_selectors = missing_selectors
        super().__init__("preparation startup configuration is missing")


def _validate_catalog_resources(catalog: PreparationCatalog) -> None:
    """Pure configured/admission checks before any source observation or SDK call."""
    if type(catalog) is not PreparationCatalog:
        raise ValueError("trusted preparation catalog required")
    required = {
        "AMLAS_AWS_ACCOUNT_ID": catalog.account_id,
        "AMLAS_AWS_REGION": catalog.region,
        "AMLAS_SESSION_BUCKET": catalog.bucket,
        "SAGEMAKER_EXECUTION_ROLE_ARN": catalog.execution_role_arn,
        "AMLAS_DEPLOYMENT_ID": catalog.deployment_id,
        "AMLAS_DEPLOYMENT_PREFIX": catalog.deployment_prefix,
        "AMLAS_DATA_PREPARATION_IMAGE_URI": catalog.image_uri,
        "AMLAS_DATA_PREPARATION_SOURCE_COMMIT": catalog.source_commit,
        "AMLAS_DATA_PREPARATION_ADMISSION_DIGEST": catalog.admission_digest,
    }
    missing = tuple(name for name, value in required.items() if type(value) is str and value == "")
    if missing:
        raise PreparationConfigurationMissingError(missing)
    image = re.fullmatch(
        r"([0-9]{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com/[a-z0-9][a-z0-9._/-]*@sha256:([a-f0-9]{64})",
        catalog.image_uri,
    )
    role = re.fullmatch(r"arn:aws:iam::([0-9]{12}):role/[A-Za-z0-9+=,.@_/-]+", catalog.execution_role_arn)
    if (
        not image
        or not role
        or catalog.account_id == "0" * 12
        or image[1] != catalog.account_id
        or role[1] != catalog.account_id
        or image[2] != catalog.region
    ):
        raise ValueError("preparation requires same-account/region immutable catalog")
    if not re.fullmatch(r"[a-f0-9]{40}", catalog.source_commit) or catalog.admission_digest != image[3]:
        raise ValueError("preparation image/source native admission is missing")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", catalog.bucket) or ".." in catalog.bucket:
        raise ValueError("invalid selected preparation bucket")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", catalog.deployment_prefix):
        raise ValueError("invalid preparation deployment namespace")
    try:
        if str(uuid.UUID(catalog.deployment_id)) != catalog.deployment_id:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ValueError("invalid preparation deployment identity") from None
    if not re.fullmatch(r"ml\.[a-z0-9]+\.[a-z0-9]+", catalog.instance_type):
        raise ValueError("invalid preparation instance")
    for value, high in ((catalog.volume_size_gb, 16384), (catalog.max_runtime_seconds, 86400)):
        if type(value) is not int or not 1 <= value <= high:
            raise ValueError("invalid preparation resource envelope")


def validate_catalog(catalog: PreparationCatalog, *, exclusions_limit: int = 2 * 1024 * 1024) -> None:
    """Pure complete resources and immutable exclusion checks."""
    try:
        _validate_catalog_resources(catalog)
    except PreparationConfigurationMissingError as error:
        missing = error.missing_selectors
        if type(catalog.exclusions_ref) is DatasetArtifactRef:
            missing += tuple(
                name
                for name, value in (
                    ("AMLAS_DATA_PREPARATION_EXCLUSIONS_URI", catalog.exclusions_ref.uri),
                    ("AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID", catalog.exclusions_ref.version_id),
                    ("AMLAS_DATA_PREPARATION_EXCLUSIONS_SHA256", catalog.exclusions_ref.sha256),
                )
                if type(value) is str and value == ""
            )
            if type(catalog.exclusions_ref.byte_count) is int and catalog.exclusions_ref.byte_count == 0:
                missing += ("AMLAS_DATA_PREPARATION_EXCLUSIONS_BYTE_COUNT",)
        raise PreparationConfigurationMissingError(missing) from None
    if type(exclusions_limit) is not int or exclusions_limit not in (2 * 1024 * 1024, 8 * 1024 * 1024):
        raise ValueError("fixed preparation exclusion limit required")
    if type(catalog.exclusions_ref) is DatasetArtifactRef:
        required = {
            "AMLAS_DATA_PREPARATION_EXCLUSIONS_URI": catalog.exclusions_ref.uri,
            "AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID": catalog.exclusions_ref.version_id,
            "AMLAS_DATA_PREPARATION_EXCLUSIONS_SHA256": catalog.exclusions_ref.sha256,
        }
        missing = tuple(name for name, value in required.items() if type(value) is str and value == "")
        if type(catalog.exclusions_ref.byte_count) is int and catalog.exclusions_ref.byte_count == 0:
            missing += ("AMLAS_DATA_PREPARATION_EXCLUSIONS_BYTE_COUNT",)
        if missing:
            raise PreparationConfigurationMissingError(missing)
    ref = catalog.exclusions_ref
    if (
        type(ref) is not DatasetArtifactRef
        or ref.format != "json"
        or type(ref.byte_count) is not int
        or not 0 < ref.byte_count <= exclusions_limit
    ):
        raise ValueError("pinned preparation exclusions required")
    if (
        not _version(ref.version_id)
        or not re.fullmatch(r"[a-f0-9]{64}", ref.sha256)
        or not ref.uri.startswith("s3://" + catalog.bucket + "/")
    ):
        raise ValueError("preparation exclusions are foreign or mutable")


def _version(value: Any) -> bool:
    return (
        type(value) is str
        and value not in ("", "null")
        and len(value) <= 1024
        and not any(ord(char) < 32 for char in value)
    )


def validate_source(source: VerifiedPreparationSource, catalog: PreparationCatalog, session: str) -> None:
    if (
        type(source) is not VerifiedPreparationSource
        or type(source.inspection_source) is not DatasetInspectionSource
    ):
        raise ValueError("private verified acquisition source required")
    bound = source.inspection_source
    if (
        bound.runtime_session_id,
        bound.account_id,
        bound.region,
        bound.bucket,
        bound.deployment_id,
        bound.deployment_prefix,
    ) != (
        session,
        catalog.account_id,
        catalog.region,
        catalog.bucket,
        catalog.deployment_id,
        catalog.deployment_prefix,
    ):
        raise ValueError("preparation source belongs to another conversation/deployment")
    ref = bound.artifact_ref
    prefix = "s3://" + catalog.bucket + "/sessions/" + session + "/acquisition/"
    if (
        type(ref) is not DatasetArtifactRef
        or ref.format not in ("jsonl", "parquet")
        or type(ref.byte_count) is not int
        or not 0 < ref.byte_count <= 512 * 1024 * 1024
    ):
        raise ValueError("only admitted canonical JSONL or pinned OpenSWE Parquet is supported")
    if (
        not ref.uri.startswith(prefix)
        or not _version(ref.version_id)
        or not re.fullmatch(r"[a-f0-9]{64}", ref.sha256)
    ):
        raise ValueError("acquisition artifact is foreign or mutable")
    if type(source.manifest_refs) is not tuple or len(source.manifest_refs) != 3:
        raise ValueError("verified acquisition manifest bundle required")
    names = set()
    for item in source.manifest_refs:
        if type(item) is not tuple or len(item) != 4:
            raise ValueError("invalid acquisition manifest binding")
        name, uri, version, digest = item
        if (
            name in names
            or name not in ("receipt", "commit", "lineage")
            or type(uri) is not str
            or not uri.startswith(prefix)
            or not _version(version)
            or type(digest) is not str
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            raise ValueError("foreign/mutable acquisition manifest")
        names.add(name)
    if (
        type(source.revision) is not str
        or not re.fullmatch(r"[a-f0-9]{40}", source.revision)
        or type(source.acquisition_source_commit) is not str
        or not re.fullmatch(r"[a-f0-9]{40}", source.acquisition_source_commit)
        or type(source.filename) is not str
    ):
        raise ValueError("unsupported or unpinned acquisition profile")
    from ..data_eng import commitpack_preparation, commitpack_profile

    if source.repo_id == commitpack_profile.DATASET_REPO:
        if not commitpack_preparation.matches_commitpack_source(
            source.repo_id, source.revision, source.filename, ref
        ):
            raise ValueError("recognized CommitPack source has unsupported immutable pins")
    elif ref.format == "jsonl":
        if not source.filename.endswith(".jsonl"):
            raise ValueError("unsupported or unpinned acquisition profile")
    elif (
        source.repo_id != "nvidia/Open-SWE-Traces"
        or source.revision != "1db674173119d71df45c22007af99e214600711a"  # pragma: allowlist secret
        or not re.fullmatch(
            r"data/(openhands|sweagent|minisweagent)/(qwen38_27b|qwen38_flash_next)/"
            r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\.parquet",
            source.filename,
        )
        or ".." in source.filename
    ):
        raise ValueError("only fixed pinned OpenSWE Parquet profile is supported")


@dataclass
class _PreparationRecord:
    adapter: CodingPreparationBackend
    handle: JobHandle
    result: dict[str, Any] | None = None


class SessionPreparationStore:
    """Conversation-local handles, never policy grants or restart durability."""

    def __init__(
        self,
        acquisition_store: SessionAcquisitionStore,
        adapter: CodingPreparationBackend,
        catalog: PreparationCatalog,
        *,
        case_store=None,
    ) -> None:
        if (
            not isinstance(acquisition_store, SessionAcquisitionStore)
            or not isinstance(adapter, CodingPreparationBackend)
            or type(catalog) is not PreparationCatalog
        ):
            raise TypeError("typed preparation state required")
        self.runtime_session_id = acquisition_store.runtime_session_id
        self._acquired = acquisition_store
        self._adapter = adapter
        self._catalog = catalog
        if case_store is not None:
            from .coding_case_state import SessionCodingCaseFreezeStore

            if (
                type(case_store) is not SessionCodingCaseFreezeStore
                or case_store._acquired is not acquisition_store
                or case_store.runtime_session_id != self.runtime_session_id
            ):
                raise ValueError("original same-conversation case store required")
        self._case_store = case_store
        self._records: dict[str, _PreparationRecord] = {}
        self._submissions: dict[tuple, str] = {}
        self._lock = threading.RLock()

    def _require_profile_family(self, source_key: tuple[str, ...], family: str) -> None:
        """A named profile family must match every private source; the caller never grants it."""
        from ..data_eng import commitpack_profile

        sources = tuple(self._acquired._verified_preparation_source(ref) for ref in source_key)
        if family != "commitpack" or any(s.repo_id != commitpack_profile.DATASET_REPO for s in sources):
            raise ValueError("selected profile does not match the acquisition sources")

    def submit(
        self,
        references: list[str],
        *,
        case_request_ref: str | None = None,
        profile_family: str | None = None,
    ) -> dict[str, Any]:
        if (
            type(references) is not list
            or not 0 < len(references) <= 256
            or any(type(ref) is not str or not re.fullmatch(r"acq_[a-f0-9]{32}", ref) for ref in references)
            or len(set(references)) != len(references)
        ):
            raise ValueError("confined unique acquisition references required")
        # Validate selected immutable configuration before any source observation.
        # The exact trusted source pair narrows the optional8MiB cap below.
        selected = self._catalog
        if case_request_ref is None:
            validate_catalog(selected, exclusions_limit=8 * 1024 * 1024)
        else:
            _validate_catalog_resources(selected)
        case_identity = ()
        if case_request_ref is not None:
            from .coding_case_state import VerifiedCasePreparationExclusions
            from .sft_state import validate_ref

            if (
                type(case_request_ref) is not str
                or not re.fullmatch("case_[a-f0-9]{32}", case_request_ref)
                or self._case_store is None
                or self._case_store._acquired is not self._acquired
                or self._case_store.runtime_session_id != self.runtime_session_id
            ):
                raise ValueError("original opaque same-conversation frozen case required")
            frozen = self._case_store._verified_preparation_exclusions(case_request_ref)
            if (
                type(frozen) is not VerifiedCasePreparationExclusions
                or frozen.runtime_session_id != self.runtime_session_id
            ):
                raise ValueError("original verified case exclusions required")
            header = dict(frozen.producer_header)
            names = {
                "job_name",
                "runtime_session_id",
                "deployment_id",
                "deployment_prefix",
                "image_uri",
                "source_commit",
                "profile_id",
                "profile_version",
            }
            if (
                type(frozen.producer_header) is not tuple
                or len(frozen.producer_header) != len(names)
                or set(header) != names
                or any(type(v) is not str or not v for v in header.values())
                or header["runtime_session_id"] != self.runtime_session_id
                or header["deployment_id"] != selected.deployment_id
                or header["deployment_prefix"] != selected.deployment_prefix
                or frozen.producer_job_arn
                != "arn:aws:sagemaker:"
                + selected.region
                + ":"
                + selected.account_id
                + ":processing-job/"
                + header["job_name"]
            ):
                raise ValueError("original case producer namespace differs")
            prefix = (
                "s3://"
                + selected.bucket
                + "/sessions/"
                + self.runtime_session_id
                + "/case-freeze/"
                + header["job_name"]
                + "/output/"
            )
            for ref, name in (
                (frozen.exclusions_ref, "EXCLUSIONS.json"),
                (frozen.committed_ref, "COMMITTED.json"),
            ):
                validate_ref(ref, selected, self.runtime_session_id, limit=2 * 1024**2 + 65536)
                if ref.uri != prefix + name or ref.format != "json":
                    raise ValueError("original immutable case role/path differs")
            selected = replace(selected, exclusions_ref=frozen.exclusions_ref)
            case_identity = (
                frozen.producer_job_arn,
                *frozen.producer_header,
                frozen.committed_ref,
                frozen.exclusions_ref,
            )
        validate_catalog(selected, exclusions_limit=8 * 1024 * 1024)
        source_key = tuple(sorted(references))
        key = source_key if not case_identity else (source_key, case_identity)
        with self._lock:
            if profile_family is not None:
                self._require_profile_family(source_key, profile_family)
            if key in self._submissions:
                reference = self._submissions[key]
                handle = self._records[reference].handle
                return {
                    "success": True,
                    "accepted": True,
                    "prepared": False,
                    "trained": False,
                    "status": "submitted",
                    "request_ref": reference,
                    "preparation_request_ref": reference,
                    "job_name": handle.id,
                    "job_arn": handle.arn,
                    "reused": True,
                }
            sources = tuple(self._acquired._verified_preparation_source(ref) for ref in source_key)
            from ..data_eng import commitpack_preparation, commitpack_profile

            commitpack = all(source.repo_id == commitpack_profile.DATASET_REPO for source in sources)
            validate_catalog(
                selected,
                exclusions_limit=commitpack_preparation.MAX_EXCLUSIONS_BYTES
                if commitpack
                else 2 * 1024 * 1024,
            )
            for source in sources:
                validate_source(source, selected, self.runtime_session_id)
            handle = self._adapter.submit(
                sources=sources, catalog=selected, runtime_session_id=self.runtime_session_id
            )
            if (
                type(handle) is not JobHandle
                or handle.backend != self._adapter.backend_name
                or handle.extra.get("runtime_session_id") != self.runtime_session_id
            ):
                raise ValueError("preparation returned another native binding")
            reference = "prep_" + uuid.uuid4().hex
            self._records[reference] = _PreparationRecord(self._adapter, handle)
            self._submissions[key] = reference
            return {
                "success": True,
                "accepted": True,
                "prepared": False,
                "trained": False,
                "status": "submitted",
                "request_ref": reference,
                "preparation_request_ref": reference,
                "job_name": handle.id,
                "job_arn": handle.arn,
                "poll_after_seconds": 15,
            }

    def _original_producer_handle(self, reference, binding):
        from .campaign_state import _binding_fields

        fields = _binding_fields(binding)
        with self._lock:
            record = self._records.get(reference) if type(reference) is str else None
        if record is None or self.runtime_session_id != fields["runtime_session_id"]:
            raise ValueError("original scoped preparation producer missing")
        if record.handle.extra.get("runtime_session_id") != self.runtime_session_id:
            raise ValueError("original preparation correlation differs")
        return record.handle

    def _restore_original_producer(self, reference, handle, binding):
        from .campaign_state import _binding_fields

        fields = _binding_fields(binding)
        if (
            type(reference) is not str
            or not re.fullmatch("prep_[a-f0-9]{32}", reference)
            or type(handle) is not JobHandle
            or handle.backend != "processing"
            or fields["runtime_session_id"] != self.runtime_session_id
            or handle.extra.get("runtime_session_id") != self.runtime_session_id
            or type(handle.extra.get("catalog")) is not PreparationCatalog
        ):
            raise ValueError("typed original preparation producer required")
        with self._lock:
            if self._records or self._submissions:
                raise ValueError("original preparation restore requires empty temporary state")
            self._records[reference] = _PreparationRecord(self._adapter, handle)

    def _restore_original_selection(self, reference, acquisition_refs, case_ref, binding):
        """Rebuild the original case-inclusive key from freshly verified sources."""
        from .campaign_state import _binding_fields

        fields = _binding_fields(binding)
        if (
            self.runtime_session_id != fields["runtime_session_id"]
            or type(acquisition_refs) is not tuple
            or not 1 <= len(acquisition_refs) <= 16
            or tuple(sorted(set(acquisition_refs))) != acquisition_refs
        ):
            raise ValueError("original preparation selection differs")
        case_identity = ()
        selected = self._catalog
        if case_ref is not None:
            if self._case_store is None:
                raise ValueError("original preparation case store missing")
            frozen = self._case_store._verified_preparation_exclusions(case_ref)
            selected = replace(selected, exclusions_ref=frozen.exclusions_ref)
            case_identity = (
                frozen.producer_job_arn,
                *frozen.producer_header,
                frozen.committed_ref,
                frozen.exclusions_ref,
            )
        sources = tuple(self._acquired._verified_preparation_source(ref) for ref in acquisition_refs)
        with self._lock:
            record = self._records.get(reference)
            if (
                len(self._records) != 1
                or record is None
                or self._submissions
                or record.handle.extra.get("catalog") != selected
                or record.handle.extra.get("sources") != sources
            ):
                raise ValueError("original preparation selection provenance differs")
            key = acquisition_refs if not case_identity else (acquisition_refs, case_identity)
            self._submissions[key] = reference

    def observe(self, reference: str) -> dict[str, Any]:
        with self._lock:
            record = (
                self._records.get(reference)
                if type(reference) is str and re.fullmatch(r"prep_[a-f0-9]{32}", reference)
                else None
            )
        if record is None:
            return {
                "success": False,
                "accepted": False,
                "prepared": False,
                "trained": False,
                "status": "unknown_reference",
            }
        base = {
            "accepted": True,
            "prepared": False,
            "trained": False,
            "request_ref": reference,
            "preparation_request_ref": reference,
            "job_name": record.handle.id,
            "job_arn": record.handle.arn,
        }
        try:
            result = record.adapter.artifacts(record.handle)
            if type(result) is not dict:
                raise ValueError("invalid preparation observation")
            with self._lock:
                record.result = copy.deepcopy(result)
            return {
                **result,
                **base,
                "prepared": result.get("success") is True and result.get("prepared") is True,
            }
        except Exception as error:
            return {
                **base,
                "success": False,
                "status": "observation_failed",
                "error_type": type(error).__name__,
            }

    def _verified_sft_snapshot(self, reference: str):
        """One fresh original artifact observation; public result dictionaries are never input."""
        from .sft_state import VerifiedSftPreparedSource, validate_ref

        with self._lock:
            record = (
                self._records.get(reference)
                if type(reference) is str and re.fullmatch("prep_[a-f0-9]{32}", reference)
                else None
            )
        if record is None or record.handle.extra.get("runtime_session_id") != self.runtime_session_id:
            raise ValueError("unknown or foreign original SFT preparation ref")
        original_catalog = record.handle.extra.get("catalog")
        if type(original_catalog) is not PreparationCatalog:
            raise ValueError("original typed SFT preparation catalog required")
        validate_catalog(original_catalog, exclusions_limit=8 * 1024 * 1024)
        result = record.adapter.artifacts(record.handle)
        if (
            type(result) is not dict
            or result.get("success") is not True
            or result.get("prepared") is not True
        ):
            raise ValueError("preparation has no verified SFT snapshot")
        if result.get("job_name") != record.handle.id or result.get("job_arn") != record.handle.arn:
            raise ValueError("SFT preparation native identity differs")

        header = record.handle.extra.get("header")
        names = {
            "job_name",
            "runtime_session_id",
            "deployment_id",
            "deployment_prefix",
            "image_uri",
            "source_commit",
            "profile_id",
            "policy_version",
        }
        expected = {
            "job_name": record.handle.id,
            "runtime_session_id": self.runtime_session_id,
            "deployment_id": original_catalog.deployment_id,
            "deployment_prefix": original_catalog.deployment_prefix,
            "image_uri": original_catalog.image_uri,
            "source_commit": original_catalog.source_commit,
        }
        original_arn = (
            "arn:aws:sagemaker:"
            + original_catalog.region
            + ":"
            + original_catalog.account_id
            + ":processing-job/"
            + record.handle.id
        )
        if (
            type(header) is not dict
            or set(header) != names
            or any(type(value) is not str or not value for value in header.values())
            or any(header[name] != value for name, value in expected.items())
            or record.handle.arn != original_arn
        ):
            raise ValueError("SFT original preparation header differs")
        prefix = (
            "s3://"
            + original_catalog.bucket
            + "/sessions/"
            + self.runtime_session_id
            + "/preparation/"
            + record.handle.id
            + "/output/"
        )

        def artifact(value, filename, format):
            if type(value) is DatasetArtifactRef:
                ref = value
            elif type(value) is dict and set(value) == {
                "uri",
                "version_id",
                "sha256",
                "byte_count",
                "format",
            }:
                ref = DatasetArtifactRef(**value)
            else:
                raise ValueError("SFT preparation ref shape differs")
            validate_ref(ref, original_catalog, self.runtime_session_id, limit=256 * 1024**2)
            if type(ref.format) is not str or ref.format != format or ref.uri != prefix + filename:
                raise ValueError("SFT preparation original role/path differs")
            return ref

        if (
            type(result.get("split_refs")) is not dict
            or set(result["split_refs"]) != {"train", "validation", "test"}
            or type(result.get("sidecar_refs")) is not dict
            or set(result["sidecar_refs"]) != {"report", "lineage", "membership"}
        ):
            raise ValueError("SFT preparation original inventory differs")
        counts = record.handle.extra.get("source_row_counts", {})
        if type(counts) is not dict:
            raise ValueError("SFT preparation private source counts differ")
        return VerifiedSftPreparedSource(
            self.runtime_session_id,
            record.handle.arn,
            tuple(sorted(header.items())),
            artifact(result["preparation_ref"], "COMMITTED.json", "json"),
            tuple(
                sorted(
                    (name, artifact(ref, name + ".jsonl", "jsonl"))
                    for name, ref in result["split_refs"].items()
                )
            ),
            tuple(
                sorted(
                    (
                        name,
                        artifact(
                            ref,
                            "REPORT.json" if name == "report" else name + ".jsonl",
                            "json" if name == "report" else "jsonl",
                        ),
                    )
                    for name, ref in result["sidecar_refs"].items()
                )
            ),
            original_catalog.exclusions_ref,
            tuple(sorted(counts.items())),
        )
