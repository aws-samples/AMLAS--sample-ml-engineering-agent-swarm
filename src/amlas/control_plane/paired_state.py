"""Opaque original paired evaluation handles, without policy or quality grants."""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass

from ..backends.ports import PairedCodingBackend
from ..backends.types import JobHandle, is_capacity_rejection
from .coding_case_state import SessionCodingCaseFreezeStore, VerifiedCaseEvaluationSource
from .generation_state import (
    CAPACITY_REJECTION_REASON,
    SessionCodingGenerationStore,
    VerifiedCodingGenerationSource,
)
from .sft_state import SftCatalog, SftConfigurationMissingError, validate_catalog


def pair_operation_digest(runtime_session_id, pair_catalog, inputs):
    """Identity of one paired evaluation request: the session, the pair catalog and the verified inputs."""
    from .generation_state import _operation_digest

    return _operation_digest(
        "amlas.pair.operation.v1",
        {
            "session": runtime_session_id,
            "catalog": asdict(pair_catalog),
            "cases": asdict(inputs.cases),
            "base": asdict(inputs.base),
            "tuned": asdict(inputs.tuned),
            "spec": inputs.spec_sha256,
        },
    )


# What a read of a capacity-refused paired evaluation tells the model to do. The pair adapter records no instance
# type and the store has no read of the processing jobs of this run, so the read names no type and no holder.
PAIR_CAPACITY_NEXT_STEP = (
    "SageMaker refused this paired evaluation for a capacity limit. Nothing was created. "
    "Submit the paired evaluation again with the same arguments. "
    "If a job in the account still holds that capacity, SageMaker refuses again and creates nothing."
)


@dataclass(frozen=True)
class PairedCodingInputs:
    cases: VerifiedCaseEvaluationSource
    base: VerifiedCodingGenerationSource
    tuned: VerifiedCodingGenerationSource
    spec_bytes: bytes
    spec_sha256: str


@dataclass(frozen=True)
class PairedCodingCatalog:
    """Pair-only expected image/binary data; never caller action authority."""

    execution: SftCatalog
    launcher_sha256: str


class PairConfigurationMissingError(ValueError):
    def __init__(self, names):
        self.missing_selectors = names
        super().__init__("paired evaluation configuration missing")


def validate_pair_catalog(catalog):
    if type(catalog) is not PairedCodingCatalog or type(catalog.execution) is not SftCatalog:
        raise ValueError("typed pair-only execution catalog required")
    if type(catalog.launcher_sha256) is not str or (
        catalog.launcher_sha256 and not re.fullmatch("[a-f0-9]{64}", catalog.launcher_sha256)
    ):
        raise ValueError("paired launcher digest format differs")
    missing = []
    try:
        validate_catalog(catalog.execution)
    except SftConfigurationMissingError as error:
        rename = {
            "AMLAS_QWEN_SFT_IMAGE_URI": "AMLAS_CODING_PAIR_IMAGE_URI",
            "AMLAS_QWEN_SFT_SOURCE_COMMIT": "AMLAS_CODING_PAIR_SOURCE_COMMIT",
            "AMLAS_QWEN_SFT_ADMISSION_DIGEST": "AMLAS_CODING_PAIR_ADMISSION_DIGEST",
        }
        missing.extend(rename.get(name, name) for name in error.missing_selectors)
    if not catalog.launcher_sha256:
        missing.append("AMLAS_CODING_PAIR_LAUNCHER_SHA256")
    if missing:
        raise PairConfigurationMissingError(tuple(missing))
    return catalog.execution


class SessionPairedCodingStore:
    def __init__(self, generation, cases, adapter, catalog):
        if (
            type(generation) is not SessionCodingGenerationStore
            or type(cases) is not SessionCodingCaseFreezeStore
            or generation._cases is not cases
            or generation.runtime_session_id != cases.runtime_session_id
            or not isinstance(adapter, PairedCodingBackend)
            or type(catalog) is not PairedCodingCatalog
            or type(catalog.execution) is not SftCatalog
        ):
            raise ValueError("typed original paired conversation stores required")
        self._generation, self._cases, self._adapter, self._catalog = generation, cases, adapter, catalog
        self.runtime_session_id = generation.runtime_session_id
        self._records, self._keys = {}, {}
        self._lock = threading.RLock()

    def submit(self, case_ref, base_ref, tuned_ref, spec_ref):
        validate_pair_catalog(self._catalog)
        if (
            self._generation.runtime_session_id != self.runtime_session_id
            or self._cases.runtime_session_id != self.runtime_session_id
        ):
            raise ValueError("paired conversation changed")
        for ref, prefix in (
            (case_ref, "case_"),
            (base_ref, "gen_"),
            (tuned_ref, "gen_"),
            (spec_ref, "spec_"),
        ):
            if type(ref) is not str or not re.fullmatch(prefix + "[a-f0-9]{32}", ref):
                raise ValueError("opaque paired original references required")
        with self._lock:
            key = (case_ref, base_ref, tuned_ref, spec_ref)
            # A capacity rejection made no job: the same request reaches the adapter again (same reference and
            # deterministic native name). Every other recorded outcome stays the answer.
            if key in self._keys and not is_capacity_rejection(self._records[self._keys[key]]):
                return self._submitted(self._keys[key], reused=True)
            spec_bytes, spec_sha, cases = self._generation._verified_resolved_spec(case_ref, spec_ref)
            base = self._generation._verified_evaluation_source(base_ref)
            tuned = self._generation._verified_evaluation_source(tuned_ref)
            if (
                type(base) is not VerifiedCodingGenerationSource
                or type(tuned) is not VerifiedCodingGenerationSource
                or base.arm != "base"
                or tuned.arm != "tuned"
                or base.evaluation_id != tuned.evaluation_id
                or base.producer_job_arn == tuned.producer_job_arn
                or base.runtime_session_id != self.runtime_session_id
                or tuned.runtime_session_id != self.runtime_session_id
            ):
                raise ValueError("original base/tuned paired identities differ")
            inputs = PairedCodingInputs(cases, base, tuned, spec_bytes, spec_sha)
            handle = self._adapter.submit(
                inputs=inputs, catalog=self._catalog, runtime_session_id=self.runtime_session_id
            )
            if (
                type(handle) is not JobHandle
                or handle.backend != "paired_coding"
                or handle.extra.get("runtime_session_id") != self.runtime_session_id
            ):
                raise ValueError("original paired job handle differs")
            ref = "pair_" + pair_operation_digest(self.runtime_session_id, self._catalog, inputs)[:32]
            self._records[ref], self._keys[key] = handle, ref
            return self._submitted(ref)

    def _submitted(self, ref, *, reused=False):
        handle = self._records[ref]
        state = handle.extra["submission_state"]
        return {
            "success": False,
            "accepted": state == "accepted",
            "evaluated": False,
            "quality_success": False,
            "native_admission": False,
            "request_ref": ref,
            "job_name": handle.id,
            "job_arn": handle.arn,
            "status": "submitted"
            if state == "accepted"
            else "submission_rejected"
            if state == "rejected"
            else "acceptance_unknown",
            "reused": reused,
        }

    def observe(self, ref):
        with self._lock:
            handle = (
                self._records.get(ref)
                if type(ref) is str and re.fullmatch("pair_[a-f0-9]{32}", ref)
                else None
            )
            if is_capacity_rejection(handle):
                # The refused create made no native job: the read answers the stored refusal, describes nothing,
                # creates nothing and leaves the record as it is. A later submit reaches the create again. The
                # answer is built under the lock that read the record, so it never mixes two records.
                return {
                    **self._submitted(ref),
                    "rejection_reason": CAPACITY_REJECTION_REASON,
                    "next_step": PAIR_CAPACITY_NEXT_STEP,
                }
        if handle is None:
            return {
                "success": False,
                "evaluated": False,
                "quality_success": False,
                "native_admission": False,
                "status": "unknown_reference",
            }
        try:
            return {**self._submitted(ref), **self._adapter.artifacts(handle), "request_ref": ref}
        except Exception as error:
            return {
                **self._submitted(ref),
                "status": "observation_failed",
                "error_type": type(error).__name__,
            }
