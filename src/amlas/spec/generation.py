"""Strict descriptive generation planning data, without execution or native trust.

A valid pin, catalog selection, reference or measured-state declaration is only
schema data. Trusted execution bindings and native verification live elsewhere.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BaselineMeasurement", "CatalogSelection", "CodingQuality", "EvaluationHarnessPin",
    "EvaluationPlan", "GenerationPlan", "HubPin", "ManifestRef", "PreparationPlan",
]


class _PlanningData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class HubPin(_PlanningData):
    """Bounded repository name plus full immutable commit, not metadata admission."""

    repo_id: Annotated[str, Field(
        min_length=3, max_length=193,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$",
    )]
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class CatalogSelection(_PlanningData):
    """Explicit catalog identity/version; no lookup or inferred admission."""

    id: Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    version: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")]

    @field_validator("version")
    @classmethod
    def _explicit_version(cls, value: str) -> str:
        if value.lower() in {"latest", "main", "master", "current", "default", "auto"}:
            raise ValueError("catalog selection requires an explicit version")
        return value


class ManifestRef(_PlanningData):
    """The existing five-field artifact shape, restricted to a JSON manifest.

    Well-formed fields do not prove ownership, object existence, integrity,
    producer admission or completed preparation/evaluation.
    """

    uri: Annotated[str, Field(min_length=8, max_length=2048)]
    version_id: Annotated[str, Field(min_length=1, max_length=1024)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    byte_count: Annotated[int, Field(gt=0)]
    format: Literal["json"]

    @field_validator("uri")
    @classmethod
    def _s3_object_uri(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            not value.startswith("s3://") or parsed.scheme != "s3"
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)
            or not parsed.path.startswith("/") or len(parsed.path) == 1
            or parsed.query or parsed.fragment
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("manifest reference requires a bounded S3 object URI without query or userinfo")
        return value

    @field_validator("version_id")
    @classmethod
    def _versioned_object(cls, value: str) -> str:
        if value.lower() in {"null", "none"} or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value
        ):
            raise ValueError("manifest reference requires an explicit object version")
        return value


class PreparationPlan(_PlanningData):
    profile: CatalogSelection | None = None
    manifest_ref: ManifestRef | None = None


class EvaluationHarnessPin(_PlanningData):
    agent: HubPin
    evaluator: HubPin


class EvaluationPlan(_PlanningData):
    harness: EvaluationHarnessPin | None = None
    case_manifest_ref: ManifestRef | None = None


class CodingQuality(_PlanningData):
    metric: Literal["resolved_rate", "pass_at_1"] = "resolved_rate"
    comparison: Literal["paired_same_cases"] = "paired_same_cases"
    coverage: Literal["all_required_cases"] = "all_required_cases"
    failure_policy: Literal["unresolved"] = "unresolved"
    harness_controls: Literal["gold_and_empty"] = "gold_and_empty"


class BaselineMeasurement(_PlanningData):
    """Declared measurement state; actual native/evaluation evidence is separate."""

    state: Literal["unmeasured", "pending", "failed", "measured"] = "unmeasured"
    report_ref: ManifestRef | None = None

    @model_validator(mode="after")
    def _measured_report(self) -> BaselineMeasurement:
        if self.state == "measured" and self.report_ref is None:
            raise ValueError("a measured baseline declaration requires a descriptive report reference")
        return self


class GenerationPlan(_PlanningData):
    """Optional unresolved planning bindings; no readiness or dispatch methods."""

    contract_version: Literal["1"] = "1"
    kind: Literal["coding_sft"] = "coding_sft"
    model: HubPin | None = None
    tokenizer: HubPin | None = None
    preparation: PreparationPlan = Field(default_factory=PreparationPlan)
    recipe: CatalogSelection | None = None
    evaluation: EvaluationPlan = Field(default_factory=EvaluationPlan)
    quality: CodingQuality = Field(default_factory=CodingQuality)
    baseline_measurement: BaselineMeasurement = Field(default_factory=BaselineMeasurement)
