"""Closed paired HumanEval metadata and arithmetic, without native authority."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..data_eng.coding_case_profile import CASE_IDS, SOURCE_SHA256
from ..spec.generation import ManifestRef
from .schema import InvalidSubmissionError

Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identity = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
Outcome = Literal[
    "passed",
    "failed",
    "syntax_error",
    "runtime_error",
    "timeout",
    "truncated",
    "infrastructure_error",
    "incomplete",
]


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CodingEvaluationBinding(_Closed):
    profile_id: Literal["openai.humaneval.164.pass_at_1.v1"] = "openai.humaneval.164.pass_at_1.v1"
    evaluation_id: Identity
    spec_sha256: Sha256
    case_manifest: ManifestRef
    source_sha256: Sha256
    runner_sha256: Sha256
    runtime_sha256: Sha256
    budgets_sha256: Sha256
    generation_sha256: Sha256
    base_model_sha256: Sha256
    tuned_model_sha256: Sha256


class CodingCasePin(_Closed):
    case_id: Annotated[str, Field(min_length=11, max_length=13, pattern=r"^HumanEval/[0-9]{1,3}$")]
    oracle_sha256: Sha256


class CodingCaseReceipt(_Closed):
    """Descriptive result data; neither construction nor flags prove execution."""

    case_id: Annotated[str, Field(min_length=11, max_length=13, pattern=r"^HumanEval/[0-9]{1,3}$")]
    arm: Literal["base", "tuned"]
    binding: CodingEvaluationBinding
    model_sha256: Sha256
    candidate_sha256: Sha256
    oracle_sha256: Sha256
    outcome: Outcome
    oracle_completed: bool
    exit_code: Annotated[int, Field(ge=-255, le=255)] | None
    execution_id: Identity
    stdout_sha256: Sha256


_REPORT_KEY = object()


@dataclass(frozen=True, slots=True, init=False)
class _VerifiedCodingReport:
    """Immutable canonical private bytes; source-only, never native authority."""

    _binding_bytes: bytes
    _summary_bytes: bytes
    _key: object

    def __init__(self, binding: CodingEvaluationBinding, summary: dict, key: object) -> None:
        if key is not _REPORT_KEY:
            raise InvalidSubmissionError("owning coding evaluator factory required")
        object.__setattr__(self, "_binding_bytes", binding.model_dump_json().encode())
        object.__setattr__(
            self,
            "_summary_bytes",
            json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
        )
        object.__setattr__(self, "_key", key)

    @property
    def binding(self) -> CodingEvaluationBinding:
        return CodingEvaluationBinding.model_validate_json(self._binding_bytes)

    def describe(self) -> dict:
        return {
            **json.loads(self._summary_bytes),
            "binding": self.binding.model_dump(),
            "provenance_status": "declared_consistency_checked_not_authenticated",
            "native_admission": False,
            "containment_verified": False,
            "grading_integrity_verified": False,
            "quality_success": False,
        }


def coding_spec_bytes(spec: object) -> bytes:
    """Canonical validated JSON projection, stable across defaults and restoration."""
    from ..spec.schema import SpecModel

    if type(spec) is not SpecModel or spec.task_type != "text_generation":
        raise InvalidSubmissionError("exact coding SpecModel required")
    # Pydantic may retain an integer default for a float field until revalidation.
    # Revalidate the JSON projection before fingerprinting so serialized recovery
    # and the original researched specification describe identical canonical bytes.
    projection = SpecModel.model_validate(spec.model_dump(mode="json")).model_dump(mode="json")
    raw = json.dumps(projection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > 65536:
        raise InvalidSubmissionError("coding specification byte bound exceeded")
    return raw


def coding_spec_sha256(spec: object) -> str:
    return hashlib.sha256(coding_spec_bytes(spec)).hexdigest()


def require_private_report(value: object) -> _VerifiedCodingReport:
    if type(value) is not _VerifiedCodingReport or value._key is not _REPORT_KEY:
        raise InvalidSubmissionError("private coding evaluator report required; caller JSON is not evidence")
    return value


class CodingEvaluatorFactory:
    """Owning source-code seam, not an agent tool or native verifier.

    This pure first slice validates descriptive data and produces unqualified
    reports only. Actual executor qualification is a separate future integration.
    """

    def build_report(
        self,
        *,
        binding: CodingEvaluationBinding,
        case_pins: tuple[CodingCasePin, ...],
        receipts: tuple[CodingCaseReceipt, ...],
    ) -> _VerifiedCodingReport:
        try:
            if type(binding) is not CodingEvaluationBinding:
                raise ValueError("closed coding binding required")
            bound = CodingEvaluationBinding.model_validate(binding.model_dump())
            if bound.source_sha256 != SOURCE_SHA256 or bound.base_model_sha256 == bound.tuned_model_sha256:
                raise ValueError("fixed case source and distinct model arms required")
            if type(case_pins) is not tuple or len(case_pins) != 164:
                raise ValueError("complete164 private case pins required")
            if type(receipts) is not tuple or len(receipts) != 328:
                raise ValueError("exact paired2x164 execution population required")
            expected = {}
            for index, pin in enumerate(case_pins):
                if type(pin) is not CodingCasePin:
                    raise ValueError("closed private case pin required")
                checked = CodingCasePin.model_validate(pin.model_dump())
                if checked.case_id != CASE_IDS[index]:
                    raise ValueError("private case ID order/population differs")
                expected[checked.case_id] = checked.oracle_sha256
            arms: dict[str, dict[str, str]] = {"base": {}, "tuned": {}}
            executions = set()
            for receipt in receipts:
                if type(receipt) is not CodingCaseReceipt:
                    raise ValueError("closed execution receipt required")
                item = CodingCaseReceipt.model_validate(receipt.model_dump(warnings=False))
                if item.binding != bound or item.case_id not in expected:
                    raise ValueError("case execution source/runner/runtime/budget binding differs")
                if item.case_id in arms[item.arm] or item.execution_id in executions:
                    raise ValueError("duplicate case/arm or execution identity")
                model = bound.base_model_sha256 if item.arm == "base" else bound.tuned_model_sha256
                if item.model_sha256 != model or item.oracle_sha256 != expected[item.case_id]:
                    raise ValueError("case oracle or model-arm identity differs")
                if item.outcome == "passed" and (item.oracle_completed is not True or item.exit_code != 0):
                    raise ValueError("exit/stdout cannot substitute for completed oracle")
                if item.outcome not in ("passed", "failed") and item.oracle_completed is not False:
                    raise ValueError("uncompleted failure outcome cannot claim completed oracle")
                if item.outcome == "failed" and item.oracle_completed is not True:
                    raise ValueError("failed assertions require completed oracle; use incomplete status")
                arms[item.arm][item.case_id] = item.outcome
                executions.add(item.execution_id)
            if any(set(values) != set(CASE_IDS) for values in arms.values()):
                raise ValueError("both arms require all164 case IDs")
            base_pass = {case for case, outcome in arms["base"].items() if outcome == "passed"}
            tuned_pass = {case for case, outcome in arms["tuned"].items() if outcome == "passed"}
            summary = {
                "profile_id": bound.profile_id,
                "evaluation_id": bound.evaluation_id,
                "case_count": 164,
                "execution_count": 328,
                "base_pass_count": len(base_pass),
                "tuned_pass_count": len(tuned_pass),
                "base_pass_at_1": len(base_pass) / 164,
                "tuned_pass_at_1": len(tuned_pass) / 164,
                "delta_pass_at_1": (len(tuned_pass) - len(base_pass)) / 164,
                "both_pass": len(base_pass & tuned_pass),
                "base_only": len(base_pass - tuned_pass),
                "tuned_only": len(tuned_pass - base_pass),
                "both_fail": 164 - len(base_pass | tuned_pass),
                "base_failure_counts": dict(
                    sorted(Counter(value for value in arms["base"].values() if value != "passed").items())
                ),
                "tuned_failure_counts": dict(
                    sorted(Counter(value for value in arms["tuned"].values() if value != "passed").items())
                ),
                "case_outcomes": [
                    {"case_id": case, "base": arms["base"][case], "tuned": arms["tuned"][case]}
                    for case in CASE_IDS
                ],
            }
            return _VerifiedCodingReport(bound, summary, _REPORT_KEY)
        except (ValueError, TypeError, ValidationError):
            raise InvalidSubmissionError(
                "invalid fixed coding execution report; closed binding/population required"
            ) from None
