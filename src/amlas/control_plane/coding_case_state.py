"""Conversation-owned case handles; no cached policy or public oracle grant."""

from __future__ import annotations

import dataclasses
import re
import threading
import uuid
from dataclasses import dataclass

from ..backends.coding_case_port import CodingCaseFreezeBackend
from ..backends.types import DatasetArtifactRef, JobHandle
from ..data_eng.coding_case_codec import COVERAGE, FILES
from ..data_eng.coding_case_managed import _expected
from ..data_eng.coding_case_profile import PROFILE_ID
from .acquisition_state import SessionAcquisitionStore
from .preparation_state import VerifiedPreparationSource


@dataclass(frozen=True)
class CaseFreezeCatalog:
    account_id: str
    region: str
    bucket: str
    execution_role_arn: str
    image_uri: str
    source_commit: str
    admission_digest: str
    deployment_id: str
    deployment_prefix: str


@dataclass(frozen=True)
class VerifiedCasePreparationExclusions:
    runtime_session_id: str
    producer_job_arn: str
    producer_header: tuple[tuple[str, str], ...]
    original_source: VerifiedPreparationSource
    exclusions_ref: DatasetArtifactRef
    committed_ref: DatasetArtifactRef


@dataclass(frozen=True)
class VerifiedCaseEvaluationSource:
    """Fresh original case references; provenance only, never an execution grant."""

    runtime_session_id: str
    producer_job_arn: str
    producer_header: tuple[tuple[str, str], ...]
    original_source: VerifiedPreparationSource
    file_refs: tuple[tuple[str, DatasetArtifactRef], ...]


class CaseFreezeConfigurationMissingError(ValueError):
    pass


def validate_catalog(catalog):
    if type(catalog) is not CaseFreezeCatalog:
        raise ValueError("typed case catalog required")
    fields = dataclasses.asdict(catalog)
    if any(type(v) is not str for v in fields.values()):
        raise ValueError("case catalog scalar differs")
    if any(not v for v in fields.values()):
        raise CaseFreezeConfigurationMissingError("case startup selectors missing")
    if not re.fullmatch(r"[0-9]{12}", catalog.account_id) or catalog.account_id == "0" * 12:
        raise ValueError("case account differs")
    image = re.fullmatch(
        re.escape(catalog.account_id + ".dkr.ecr." + catalog.region + ".amazonaws.com/")
        + r"[a-z0-9][a-z0-9._/-]*@sha256:([a-f0-9]{64})",
        catalog.image_uri,
    )
    if (
        not image
        or image[1] != catalog.admission_digest
        or not re.fullmatch("[a-f0-9]{40}", catalog.source_commit)
    ):
        raise ValueError("case immutable image/source admission missing")
    if (
        not re.fullmatch(
            r"arn:aws:iam::" + catalog.account_id + r":role/[A-Za-z0-9_+=,.@/-]+", catalog.execution_role_arn
        )
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", catalog.bucket)
        or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", catalog.deployment_prefix)
    ):
        raise ValueError("case role/namespace differs")
    try:
        if str(uuid.UUID(catalog.deployment_id)) != catalog.deployment_id:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("case deployment differs") from None


def producer_header(catalog, session, job):
    return {
        "job_name": job,
        "runtime_session_id": session,
        "deployment_id": catalog.deployment_id,
        "deployment_prefix": catalog.deployment_prefix,
        "image_uri": catalog.image_uri,
        "source_commit": catalog.source_commit,
        "profile_id": PROFILE_ID,
        "profile_version": "1",
    }


class SessionCodingCaseFreezeStore:
    """Same conversation only; handles persist but grants never do."""

    def __init__(self, acquired, adapter, catalog):
        if (
            not isinstance(acquired, SessionAcquisitionStore)
            or not isinstance(adapter, CodingCaseFreezeBackend)
            or type(catalog) is not CaseFreezeCatalog
        ):
            raise ValueError("typed case state required")
        self._acquired, self._adapter, self._catalog = acquired, adapter, catalog
        self.runtime_session_id = acquired.runtime_session_id
        self._records, self._keys = {}, {}
        self._lock = threading.RLock()

    def _restore_original_producer(self, reference, handle, acquisition_ref, binding):
        """Restore one original alias only after fresh acquired-source comparison."""
        from .campaign_state import _binding_fields

        fields = _binding_fields(binding)
        validate_catalog(self._catalog)
        if (
            type(reference) is not str
            or not re.fullmatch("case_[a-f0-9]{32}", reference)
            or type(acquisition_ref) is not str
            or not re.fullmatch("acq_[a-f0-9]{32}", acquisition_ref)
            or type(handle) is not JobHandle
            or handle.backend != "case_freeze"
            or self.runtime_session_id != fields["runtime_session_id"]
            or handle.extra.get("runtime_session_id") != self.runtime_session_id
            or handle.extra.get("catalog") != self._catalog
            or handle.extra.get("header")
            != producer_header(self._catalog, self.runtime_session_id, handle.id)
            or handle.extra.get("source") != self._acquired._verified_preparation_source(acquisition_ref)
        ):
            raise ValueError("typed original case binding differs")
        with self._lock:
            if self._records or self._keys:
                raise ValueError("case restoration requires empty temporary state")
            self._records[reference] = handle
            self._keys[(acquisition_ref, "humaneval_164", "1")] = reference

    def submit(self, acquisition_ref, catalog_id="humaneval_164", catalog_version="1"):
        validate_catalog(self._catalog)
        if (
            (catalog_id, catalog_version) != ("humaneval_164", "1")
            or type(acquisition_ref) is not str
            or not re.fullmatch("acq_[a-f0-9]{32}", acquisition_ref)
            or self.runtime_session_id != self._acquired.runtime_session_id
        ):
            raise ValueError("original opaque case acquisition required")
        key = (acquisition_ref, catalog_id, catalog_version)
        with self._lock:
            if key in self._keys:
                return self._submitted(self._keys[key], reused=True)
            source = self._acquired._verified_preparation_source(acquisition_ref)
            header = producer_header(
                self._catalog, self.runtime_session_id, self._catalog.deployment_prefix + "-case-freeze-check"
            )
            _expected(source, header)
            origin = source.inspection_source
            if (origin.account_id, origin.region, origin.bucket) != (
                self._catalog.account_id,
                self._catalog.region,
                self._catalog.bucket,
            ):
                raise ValueError("case source belongs to another deployment")
            handle = self._adapter.submit(
                source=source, catalog=self._catalog, runtime_session_id=self.runtime_session_id
            )
            if (
                type(handle) is not JobHandle
                or handle.backend != "case_freeze"
                or handle.extra.get("runtime_session_id") != self.runtime_session_id
                or handle.extra.get("source") != source
                or handle.extra.get("catalog") != self._catalog
            ):
                raise ValueError("case original native handle differs")
            ref = "case_" + uuid.uuid4().hex
            self._records[ref] = handle
            self._keys[key] = ref
            return self._submitted(ref)

    def _submitted(self, ref, *, reused=False):
        handle = self._records[ref]
        state = handle.extra["submission_state"]
        return {
            "success": state == "accepted",
            "accepted": state == "accepted",
            "case_ref": ref,
            "status": "submitted" if state == "accepted" else state,
            "frozen": False,
            "reused": reused,
            "job_name": handle.id,
            "job_arn": handle.arn,
            "execution_status": "not_executed",
        }

    def _verified(self, case_ref):
        with self._lock:
            handle = (
                self._records.get(case_ref)
                if type(case_ref) is str and re.fullmatch("case_[a-f0-9]{32}", case_ref)
                else None
            )
        if (
            handle is None
            or handle.extra["runtime_session_id"] != self.runtime_session_id
            or self._acquired.runtime_session_id != self.runtime_session_id
        ):
            raise ValueError("unknown original case reference")
        original_catalog = handle.extra.get("catalog")
        validate_catalog(original_catalog)
        expected_header = producer_header(original_catalog, self.runtime_session_id, handle.id)
        if handle.extra.get("header") != expected_header:
            raise ValueError("original case producer header differs")
        _expected(handle.extra["source"], expected_header)
        result = self._adapter.artifacts(handle)
        if type(result) is not dict or result.get("success") is not True or result.get("frozen") is not True:
            return handle, result, None
        if (
            result.get("job_name") != handle.id
            or result.get("job_arn") != handle.arn
            or type(result.get("case_count")) is not int
            or result["case_count"] != 164
            or type(result.get("files")) is not dict
            or set(result["files"]) != set(FILES)
            or result.get("coverage") != COVERAGE
        ):
            raise ValueError("original case output identity/inventory differs")
        prefix = (
            "s3://"
            + self._catalog.bucket
            + "/sessions/"
            + self.runtime_session_id
            + "/case-freeze/"
            + handle.id
            + "/output/"
        )
        for name, ref in result["files"].items():
            if (
                type(ref) is not DatasetArtifactRef
                or ref.uri != prefix + name
                or type(ref.version_id) is not str
                or ref.version_id in ("", "null")
                or type(ref.sha256) is not str
                or not re.fullmatch("[a-f0-9]{64}", ref.sha256)
                or type(ref.byte_count) is not int
                or not 0 < ref.byte_count <= 2 * 1024**2 + 65536
                or ref.format != ("jsonl" if name.endswith(".jsonl") else "json")
            ):
                raise ValueError("immutable original case output role differs")
        return handle, result, result["files"]

    def observe(self, case_ref):
        try:
            handle, result, files = self._verified(case_ref)
            if files is None:
                return {
                    "success": False,
                    "accepted": handle.extra["submission_state"] == "accepted",
                    "frozen": False,
                    "case_ref": case_ref,
                    "status": result.get("status", "pending"),
                    "execution_status": "not_executed",
                    "job_name": handle.id,
                    "job_arn": handle.arn,
                    **(
                        {"verification_rejection": result["verification_rejection"]}
                        if "verification_rejection" in result
                        else {}
                    ),
                }
            return {
                "success": True,
                "accepted": True,
                "frozen": True,
                "case_ref": case_ref,
                "status": "frozen",
                "case_count": 164,
                "prompt_ref": dataclasses.asdict(files["PROMPTS.jsonl"]),
                "coverage": dict(COVERAGE),
                "execution_status": "not_executed",
                "metric_profile": "pass_at_1_descriptive",
                "job_name": handle.id,
                "job_arn": handle.arn,
            }
        except ValueError:
            with self._lock:
                original = self._records.get(case_ref) if type(case_ref) is str else None
            return {
                "success": False,
                "accepted": original is not None and original.extra["submission_state"] == "accepted",
                "frozen": False,
                "status": "verification_failed" if original is not None else "unknown_reference",
                "execution_status": "not_executed",
            }
        except Exception as error:
            return {
                "success": False,
                "frozen": False,
                "status": "observation_failed",
                "error_type": type(error).__name__,
            }

    def _verified_evaluation_source(self, case_ref):
        handle, _result, files = self._verified(case_ref)
        if files is None:
            raise ValueError("case outputs not verified")
        return VerifiedCaseEvaluationSource(
            self.runtime_session_id,
            handle.arn,
            tuple(sorted(handle.extra["header"].items())),
            handle.extra["source"],
            tuple(sorted(files.items())),
        )

    def _verified_preparation_exclusions(self, case_ref):
        handle, _result, files = self._verified(case_ref)
        if files is None:
            raise ValueError("case outputs not verified")
        return VerifiedCasePreparationExclusions(
            self.runtime_session_id,
            handle.arn,
            tuple(sorted(handle.extra["header"].items())),
            handle.extra["source"],
            files["EXCLUSIONS.json"],
            files["COMMITTED.json"],
        )
