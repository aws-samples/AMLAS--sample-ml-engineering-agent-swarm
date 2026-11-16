"""Spec generation from natural language description using heuristic parsing.

Provides:
  - generate_spec_from_description: @tool that parses free-text into SpecModel fields

This uses regex + keyword matching (no LLM API call). The LLM agent (spec_writer)
can call this tool and validate/refine the result.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Required

from pydantic import BaseModel, ConfigDict, Field, model_validator
from strands import tool
from typing_extensions import TypedDict

from ..errors import tool_error_boundary
from ..spec.generation import CatalogSelection
from ..spec.schema import AcceptanceCriteria, ComputeBudget

# DRY-01 fix: the pure heuristic NL-parsing helpers now live in the
# STRANDS-FREE `spec_heuristics` module so this @tool wrapper AND the boto3-only
# cloud-dispatch client (`cloud.spec_builder`) share ONE source of truth instead
# of re-inlining identical logic. Re-imported here (and re-exported via the
# module namespace) so existing callers of these helpers keep working.
from .spec_heuristics import (
    _extract_metric,
    _extract_target_column,
    _extract_threshold,
    _extract_title,
    _infer_task_type,
    _validate_column_in_data,
)


class _CatalogInput(TypedDict):
    """Both fields are required; the schema shows the model that an omitted id is refused."""

    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)

    id: str
    version: str


class _AcceptanceInput(TypedDict, total=False):
    """The delta is a comparison STRING; a bare number is refused."""

    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)

    primary_metric_delta_vs_baseline: Required[str]
    no_subgroup_regression: bool


class _CodingPlanningInput(TypedDict, total=False):
    """Preserve caller key presence before applying the existing source schemas."""

    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)

    model_catalog: _CatalogInput
    target: str
    golden_dataset: str
    acceptance: _AcceptanceInput
    compute_budget: dict[str, int | float]
    unbounded_budget: bool


class _CodingPlanningData(BaseModel):
    """Closed owner/researched planning facts; no artifact or execution authority."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    model_catalog: CatalogSelection | None = None
    target: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    golden_dataset: Annotated[str, Field(min_length=1, max_length=2048)] | None = None
    acceptance: AcceptanceCriteria | None = None
    compute_budget: ComputeBudget | None = None
    unbounded_budget: bool = False

    @model_validator(mode="after")
    def _closed_values(self):
        if self.acceptance is not None and self.acceptance.model_extra:
            raise ValueError("planning acceptance has unsupported fields")
        if self.compute_budget is not None:
            supplied = self.compute_budget.model_dump(exclude_unset=True)
            if any(not math.isfinite(value) for value in supplied.values()):
                raise ValueError("planning budget values must be finite")
            if self.unbounded_budget and any(value != 0 for value in supplied.values()):
                raise ValueError("finite planning budget contradicts overall unbounded declaration")
        return self


def _project_id_from_title(title: str) -> str:
    """Derive a SpecModel-valid project_id (``[a-zA-Z0-9_]+``, <=48 chars)."""
    pid = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower()).strip("_")
    pid = pid or "generated_spec"
    return pid[:48].rstrip("_") or "generated_spec"


def _normalize_golden_dataset(data_path: str | None) -> str:
    """Coerce a data path into a SpecModel-valid golden_dataset string.

    SpecModel requires ``s3://``, an absolute (``/``), or relative (``./``)
    prefix. Bare paths are normalized to ``./<path>``; missing paths get a
    sensible placeholder the user is expected to edit.
    """
    if not data_path:
        return "./data/train.csv"
    if data_path.startswith(("s3://", "/", "./")):
        return data_path
    return f"./{data_path}"


def spec_dict_to_frontmatter(spec: dict[str, Any]) -> str:
    """Render a generated spec dict into SpecModel-valid front-matter text (SPEC-02).

    Maps the heuristic writer output (task_type/target/primary_metric/
    success_threshold/...) onto the full ``SpecModel`` contract — adding the
    required ``project_id``, ``golden_dataset``, ``baseline``, and
    ``acceptance`` blocks with sensible placeholders — so the spec-writer
    workflow's two ends are contract-aligned and the result passes
    ``validate_spec``.
    """
    import yaml as _yaml

    generation = spec.get("task_type") == "text_generation"
    primary_metric = spec.get("primary_metric", "" if generation else "f1_macro")
    if generation and not primary_metric:
        raise ValueError("text_generation requires a researched primary_metric before rendering")
    title = spec.get("title") or spec.get("description", "Generated Spec")
    project_id = spec.get("project_id") or _project_id_from_title(title)
    baseline = spec.get("baseline")
    if not isinstance(baseline, dict):
        baseline = (
            {"type": "untouched_base_model", primary_metric: None}
            if generation
            else {"type": "heuristic_default", primary_metric: 0.0}
        )
    acceptance = spec.get("acceptance")
    if not isinstance(acceptance, dict):
        if generation:
            raise ValueError("text_generation requires explicit acceptance criteria before rendering")
        acceptance = {"primary_metric_delta_vs_baseline": ">= 0.0"}

    metadata = {
        "project_id": project_id,
        "task_type": spec.get("task_type", "classification"),
        "target": spec.get("target", "target"),
        "primary_metric": primary_metric,
        "golden_dataset": _normalize_golden_dataset(spec.get("golden_dataset") or spec.get("data_path")),
        "baseline": baseline,
        "acceptance": acceptance,
    }
    for name in ("constraints", "splits", "compute_budget", "medallion", "generation"):
        if name in spec:
            metadata[name] = spec[name]
    body = spec.get("description", "").strip() or title
    front = _yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    return f"---\n{front}---\n\n{body}\n"


# What each missing planning fact is, where it goes and a value the schema accepts. Product constants: the nine
# real-model test runs that ended before PAIR needed a second call to learn these one at a time.
_RESEARCH_DETAIL: dict[str, tuple[str, str, Any]] = {
    "model_selection": (
        "planning_data.model_catalog",
        "an object with both id and version: the entry chosen from list_model_weight_catalog",
        {"id": "qwen3_5_9b", "version": "1"},
    ),
    "primary_metric": (
        "description",
        "the description names the metric pass@1 (pass_at_1); the fixed coding task is scored by pass_at_1",
        "Paired base versus tuned evaluation scored by pass_at_1.",
    ),
    "target": (
        "planning_data.target",
        "a non-empty string of at most 128 characters that names what the model generates",
        "python_module",
    ),
    "golden_dataset": (
        "planning_data.golden_dataset",
        "a non-empty string that names the held-out cases: an s3:// URI, an absolute path or a ./relative path "
        "(other text is stored as ./text)",
        "./heldout_cases.json",
    ),
    "acceptance": (
        "planning_data.acceptance",
        "an object whose primary_metric_delta_vs_baseline is a quoted comparison string (>, >=, <, <= or == "
        "followed by a number), never a bare number",
        {"primary_metric_delta_vs_baseline": "> 0.0"},
    ),
    "compute_budget": (
        "planning_data.compute_budget or planning_data.unbounded_budget",
        "an object with any of gpu_hours, trials, wall_clock_minutes, max_retreats (omitted ones keep the schema "
        "defaults), or unbounded_budget true to declare all four ceilings zero",
        {"gpu_hours": 1, "trials": 1, "wall_clock_minutes": 120, "max_retreats": 1},
    ),
}
_RESEARCH_NEXT_STEP = (
    "Call generate_spec_from_description again with the missing facts listed in needs_research_detail "
    "(planning_data_path, expected, example) and keep the planning_data you already sent. spec_text appears when "
    "planning_status is schema_valid."
)
_BIND_NEXT_STEP = (
    "Call bind_coding_generation_spec with yaml_text = spec_text exactly as returned (unchanged); validate_spec is "
    "not needed first. Do not fill generation.preparation or generation.evaluation: they stay null and the "
    "product adds the case manifest in resolve_coding_evaluation_spec (call it with the bound spec_ref, then "
    "submit_coding_generation with the resolved spec_ref)."
)


def _coding_planning_text(
    description: str,
    data_path: str | None,
    fields: dict[str, Any],
    planning_data: _CodingPlanningInput | None = None,
) -> dict[str, Any]:
    """Produce source-catalog/schema planning DATA only from explicit input facts."""
    from ..data_eng.model_weight_manifest import expected_model_weight_manifest
    from ..spec import load_spec_from_text
    from ..spec.generation import GenerationPlan

    lower = description.lower()
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    selected = expected["repo_id"].lower() in lower
    explicit_target = re.search(r"\btarget\s+(?:column\s+)?(?:is|variable|:)\s+", lower) is not None
    acceptance = None
    acceptance_data = None
    # A strict paired improvement defines a zero DELTA, never an absolute score.
    if (
        re.search(r"\bsame\s+cases\b", lower)
        and re.search(r"\btuned\s+(?:model\s+)?pass(?:@|_at_)1\b", lower)
        and re.search(r"\b(?:strictly\s+greater\s+than|strictly\s+exceed(?:s)?|must\s+exceed)\b", lower)
        and re.search(r"\b(?:untouched[- ]base|baseline)\b", lower)
    ):
        acceptance = "> 0.0"
    explicit_delta = re.search(
        r"\bprimary_metric_delta_vs_baseline\s*[:=]\s*['\"]?((?:>=|<=|>|<|==)\s*-?\d+(?:\.\d+)?)",
        description,
    )
    if explicit_delta:
        acceptance = explicit_delta.group(1).strip()
    budget = None
    source_defaults = []
    target = fields["target"]
    golden_dataset = data_path
    # Every zero must follow an explicit no-ceiling declaration, not a default.
    unbounded_clause = re.search(r"\bunbounded\b([^.;]+)", lower)
    if unbounded_clause:
        prefix = re.split(r"[.;]", lower[: unbounded_clause.start()])[-1]
        if re.search(
            r"\b(?:not|never|reject(?:s|ed)?|refus(?:e|es|ed)|"
            r"avoid(?:s|ed)?|forbid(?:s|den)?)\b|n't\b",
            prefix,
        ) or re.search(r"\bno(?:\s+use\s+of)?\s+$", prefix):
            unbounded_clause = None
    if unbounded_clause:
        clause = unbounded_clause.group(1)
        if (
            re.search(r"\btrials\b", clause)
            and re.search(r"\bwall[- ]clock\s+minutes\b", clause)
            and re.search(r"\bretreats\b", clause)
            and re.search(r"\bno\s+(?:declared\s+)?gpu[- ]hour\s+ceiling\b", clause)
            and not re.search(r"\b(?:but|except)\b", clause)
        ):
            budget = {"gpu_hours": 0, "trials": 0, "wall_clock_minutes": 0, "max_retreats": 0}
    if planning_data is not None:
        planning_data = _CodingPlanningData.model_validate(planning_data, strict=True)
        if planning_data.model_catalog is not None:
            selection = planning_data.model_catalog
            expected = expected_model_weight_manifest(selection.id, selection.version)
            selected = True
        if planning_data.target is not None:
            target = planning_data.target
            explicit_target = True
        if planning_data.golden_dataset is not None:
            golden_dataset = planning_data.golden_dataset
        if planning_data.acceptance is not None:
            acceptance = planning_data.acceptance.primary_metric_delta_vs_baseline
            acceptance_data = planning_data.acceptance.model_dump(exclude_unset=True)
        if planning_data.unbounded_budget:
            budget = dict.fromkeys(ComputeBudget.model_fields, 0)
        elif planning_data.compute_budget is not None:
            budget = planning_data.compute_budget.model_dump(exclude_unset=True)
        elif budget is None:
            budget = {}
        source_defaults = [
            f"compute_budget.{name}" for name in ComputeBudget.model_fields if name not in budget
        ]
    required = [
        name
        for name, present in (
            ("model_selection", selected),
            ("primary_metric", fields["primary_metric"] == "pass_at_1"),
            ("target", explicit_target),
            ("golden_dataset", bool(golden_dataset)),
            ("acceptance", acceptance is not None),
            ("compute_budget", budget is not None),
        )
        if not present
    ]
    result: dict[str, Any] = {
        "planning_status": "research_needed" if required else "schema_valid",
        "needs_research": required,
        "data_only": True,
        "native_admission": False,
        "quality_success": False,
        "executed": False,
        "source_schema_defaults": source_defaults,
    }
    if required:
        result["needs_research_detail"] = [
            {"name": name, "planning_data_path": path, "expected": form, "example": example}
            for name in required
            for path, form, example in (_RESEARCH_DETAIL[name],)
        ]
        result["next_step"] = _RESEARCH_NEXT_STEP
        return result
    pin = {"repo_id": expected["repo_id"], "revision": expected["revision"]}
    plan = GenerationPlan.model_validate(
        {
            "model": pin,
            "tokenizer": pin,
            "recipe": {"id": "full_v1", "version": "1"},
            "quality": {"metric": "pass_at_1"},
        }
    )
    planning = {
        **fields,
        "target": target,
        "golden_dataset": golden_dataset,
        "baseline": {"type": "untouched_base_model", "pass_at_1": None},
        "acceptance": acceptance_data or {"primary_metric_delta_vs_baseline": acceptance},
        "compute_budget": budget,
        "generation": plan.model_dump(mode="json"),
    }
    text = spec_dict_to_frontmatter(planning)
    load_spec_from_text(text)
    result["target"] = target
    result["spec_text"] = text
    result["next_step"] = _BIND_NEXT_STEP
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool
@tool_error_boundary(source="tools.spec_generation_tools")
def generate_spec_from_description(
    description: str,
    data_path: str | None = None,
    planning_data: _CodingPlanningInput | None = None,
) -> dict[str, Any]:
    """Generate heuristic task fields and explicit coding planning DATA.

    The tool:
    1. Takes free-text description of business problem
    2. Infers task_type, target_column, metric, success_threshold
    3. If data_path provided, examines data to validate column names
    4. For coding, report missing owner/research context or source-derived spec_text.
       Only returned spec_text has passed the shared schema; planning grants no ML effect.
       Supplying planning_data (even an empty object) selects the coding task whatever the wording; the
       result then carries next_step. Bind a returned spec_text with bind_coding_generation_spec
       unchanged, and do not add generation.preparation or generation.evaluation entries by hand.

    Two planning_data rules that a failed validation otherwise teaches one call at a time:
    acceptance.primary_metric_delta_vs_baseline is a STRING such as "> 0.0" or ">= 0.005",
    never a bare number; model_catalog needs both id and version (the entry chosen from
    list_model_weight_catalog).

    Example input: "Predict which spaceship passengers were transported to
    another dimension. The target is the 'Transported' column.
    We need at least 75% F1 score."

    Args:
        description: Natural language description of the ML problem.
        data_path: Optional path to CSV data file to validate column names.
        planning_data: Optional closed owner/researched DATA; supplying it selects the coding task: model_catalog id/version,
            target, golden_dataset, acceptance.primary_metric_delta_vs_baseline and
            compute_budget (gpu_hours, trials, wall_clock_minutes, max_retreats).
            acceptance.primary_metric_delta_vs_baseline is a STRING such as "> 0.0", never a number;
            model_catalog needs both id and version.
            unbounded_budget=True declares all four ceilings zero; contradictory
            finite values are rejected. Omitted budget fields retain source schema
            defaults identified in source_schema_defaults. No refs or pin overrides.

    Returns:
        Dict with title, description, task_type, target, primary_metric,
        success_threshold and optional data_path. Coding planning adds finite
        needs_research or schema-valid spec_text; no score, budget or ref is guessed.
    """
    if not description or not description.strip():
        return {
            "error": "Empty description provided.",
            "task_type": "classification",
            "target": "target",
            "primary_metric": "f1_macro",
            "success_threshold": 0.80,
            "title": "Untitled ML Task",
            "description": "",
        }

    # Parse components from description. planning_data is the closed coding-planning argument, so its presence
    # (not its truthiness) selects the coding task even when the wording has no phrase the heuristic knows: in the
    # real-model test runs 18 of 19 results were the classification stub for calls that all carried planning_data.
    task_type = _infer_task_type(description)
    task_type_source = "description"
    if planning_data is not None and task_type != "text_generation":
        task_type, task_type_source = "text_generation", "planning_data"
    target = _extract_target_column(description)
    primary_metric = _extract_metric(description, task_type)
    success_threshold = _extract_threshold(description, task_type)
    title = _extract_title(description)

    # Coding paths are descriptive DATA; managed tools own dataset inspection.
    if data_path and task_type != "text_generation":
        if _validate_column_in_data(data_path, target):
            pass  # Column exists — good
        else:
            # Try to find a close match
            try:
                import pandas as pd

                df = pd.read_csv(data_path, nrows=0)
                # Look for case-insensitive match
                for col in df.columns:
                    if col.lower() == target.lower():
                        target = col
                        break
            except Exception:  # nosec B110
                pass

    result: dict[str, Any] = {
        "title": title,
        "description": description.strip(),
        "task_type": task_type,
        "target": target,
        "primary_metric": primary_metric,
        "success_threshold": success_threshold,
    }

    if data_path:
        result["data_path"] = data_path

    if task_type == "text_generation":
        result["task_type_source"] = task_type_source
        result.update(_coding_planning_text(description, data_path, result, planning_data))

    return result
