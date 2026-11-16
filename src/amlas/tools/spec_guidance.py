"""Closed, product-written guidance for specification validation failures (STRANDS-FREE).

``validate_spec`` used to return the raw rendered pydantic text. In real-model test runs 24 of 26 results were
invalid and the model could not act on them: 11 had no path at all, others named a path without the expected form,
and several asked for values only the product holds (harness pins, a JSON case manifest). This module turns the
structured issues (``loc`` and ``type`` only) into ``{path, problem, expected, example}`` records.

Rules that keep the records safe and stable:
  - The rendered pydantic text, the input value and the error context are never read. A path segment is kept only
    when it is a field name of the spec schema (or one of a few misplaced keys nine test runs wrote); any other
    key becomes ``(unknown key)``, so a model-supplied key is never reflected.
  - Output is deterministic (order of the issues, first record per path) and bounded (``MAX_DETAILS`` records).
  - The advice is a product constant. Examples that need a fixed pin read it from the model-weight manifest.
"""

from __future__ import annotations

import re
from typing import Any

MAX_DETAILS = 8
UNKNOWN_KEY = "(unknown key)"
WHOLE_DOCUMENT = "(whole document)"
_DECLARATION_PATH = "primary_metric and generation.quality.metric"

# Keys nine real-model test runs wrote in the wrong place; they have advice, so they are shown by name.
_ADVISED_KEYS = frozenset({"model_catalog", "recipe_id", "recipe_version", "quality_metric"})

_PROBLEM = {
    "missing": "required field is missing",
    "extra_forbidden": "field is not allowed here",
    "model_type": "wrong shape, a mapping is required",
    "dict_type": "wrong shape, a mapping is required",
    "string_type": "must be a string",
    "literal_error": "value is not one of the allowed values",
    "value_error": "value rejected by a schema rule",
    "generation_declaration_mismatch": "task_type, primary_metric and generation.quality.metric disagree",
}

INVALID_CODING_NEXT_STEP = (
    "Call generate_spec_from_description with planning_data and use its spec_text unchanged: it validates and "
    "binds. If you edited a spec, fix the error_details paths and call validate_spec again, then "
    "bind_coding_generation_spec."
)
INVALID_NEXT_STEP = "Fix the paths in error_details (path, expected, example) and call validate_spec again."
FRONT_MATTER_NEXT_STEP = (
    "A spec starts with a line ---, then YAML fields, then a line ---, then free notes. For the fixed Qwen coding "
    "task call generate_spec_from_description with planning_data and use its spec_text unchanged."
)
VALID_CODING_NEXT_STEP = (
    "The text is valid. For the fixed Qwen coding task call bind_coding_generation_spec with this same yaml_text, "
    "unchanged, then resolve_coding_evaluation_spec with the bound spec_ref."
)

_LEAVE_NULL = (
    "leave it as in spec_text (null entries): the product binds the case manifest in "
    "resolve_coding_evaluation_spec and refuses a hand-written one. A manifest takes format json only and harness "
    "pins need repo_id and revision, so no valid hand-written entry exists"
)


def _known_segments() -> frozenset[str]:
    from ..spec import generation, schema

    names: set[str] = set()
    for module in (schema, generation):
        for value in vars(module).values():
            fields = getattr(value, "model_fields", None)
            if isinstance(fields, dict) and getattr(value, "__module__", "") == module.__name__:
                names.update(fields)
    return frozenset(names)


_KNOWN: frozenset[str] | None = None


def _pin_example() -> str:
    from ..data_eng.model_weight_manifest import expected_model_weight_manifest

    manifest = expected_model_weight_manifest("qwen3_5_9b", "1")
    return f"{{repo_id: {manifest['repo_id']}, revision: {manifest['revision']}}}"


def canonical_path(loc: Any) -> str:
    """Dotted schema path of a pydantic ``loc``; list indices are dropped, unknown keys are not reflected."""
    global _KNOWN
    if _KNOWN is None:
        _KNOWN = _known_segments()
    parts: list[str] = []
    for segment in loc if isinstance(loc, (tuple, list)) else ():
        if isinstance(segment, int):
            continue
        text = segment if isinstance(segment, str) else ""
        known = text in _KNOWN or text in _ADVISED_KEYS
        parts.append(text if known and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", text) else UNKNOWN_KEY)
    # One record per block the product owns: the advice for every entry under it is the same.
    if len(parts) > 2 and parts[0] == "generation" and parts[1] in {"preparation", "evaluation"}:
        parts = parts[:2]
    return ".".join(parts) if parts else WHOLE_DOCUMENT


def _advice(path: str, kind: str) -> tuple[str, str | None]:
    """Product-written (expected form, example) for a canonical path and error type."""
    if kind == "generation_declaration_mismatch":
        return (
            "task_type text_generation and primary_metric equal to generation.quality.metric (default "
            "resolved_rate); for the fixed coding task both are pass_at_1",
            "primary_metric: pass_at_1\ngeneration:\n  quality:\n    metric: pass_at_1",
        )
    if path in {"generation.preparation", "generation.evaluation"}:
        sample = "preparation: {profile: null, manifest_ref: null}"
        if path == "generation.evaluation":
            sample = "evaluation: {harness: null, case_manifest_ref: null}"
        return _LEAVE_NULL, sample
    if path == "golden_dataset":
        return (
            "a string starting with s3://, / or ./ (the coding task keeps the golden_dataset line of spec_text, "
            "where a bare name became ./name)",
            "./heldout_cases.json",
        )
    if path == "baseline":
        return (
            "a mapping with a type key; for the fixed coding task type untouched_base_model and pass_at_1 null "
            "(the score stays null until it is measured)",
            "baseline: {type: untouched_base_model, pass_at_1: null}",
        )
    if path == "model_catalog":
        return (
            "model_catalog belongs in planning_data of generate_spec_from_description, not in the YAML; remove it. "
            "The model is pinned by generation.model",
            f"generation: {{model: {_pin_example()}}}",
        )
    if re.fullmatch(r"generation\.(model|tokenizer)(\.(id|version))?", path) and kind == "extra_forbidden":
        return (
            "a pin has only repo_id and revision (40-hex commit); the catalog id and version belong in "
            "planning_data.model_catalog, never in the YAML pin",
            "model: " + _pin_example(),
        )
    if re.fullmatch(r"generation\.(model|tokenizer)(\.(repo_id|revision))?", path):
        return (
            "the pin needs repo_id and revision (a 40-hex commit); generation.tokenizer equals generation.model",
            ("model: " + _pin_example()),
        )
    if path in {"generation.recipe_id", "generation.recipe_version"}:
        return (
            "the recipe is one mapping under generation with id and a quoted version",
            "recipe: {id: full_v1, version: '1'}",
        )
    if path.startswith("generation.recipe"):
        return (
            "id full_v1 and version as a quoted string; quote the value so YAML keeps it a string",
            "recipe: {id: full_v1, version: '1'}",
        )
    if path in {"quality_metric", "generation.quality_metric"} or path.startswith("generation.quality"):
        return (
            "the metric lives under generation.quality; for the fixed coding task it is pass_at_1",
            "quality: {metric: pass_at_1}",
        )
    if path == "splits.strategy":
        return "one of random, stratified, time_based or group", "splits: {strategy: stratified}"
    if path == "task_type":
        return (
            "one of classification, regression, segmentation, detection or text_generation; the coding task is "
            "text_generation",
            "task_type: text_generation",
        )
    if path.startswith("acceptance"):
        return (
            "a quoted comparison string: >, >=, <, <= or == followed by a number",
            "acceptance: {primary_metric_delta_vs_baseline: '> 0.0'}",
        )
    if path.startswith("compute_budget"):
        return (
            "numbers of at least 0: gpu_hours, trials, wall_clock_minutes, max_retreats",
            "compute_budget: {gpu_hours: 1, trials: 1, wall_clock_minutes: 120, max_retreats: 1}",
        )
    if path == "project_id":
        return "letters, digits and underscores only, at most 48 characters", "project_id: qwen_coding"
    if path.endswith(UNKNOWN_KEY) or kind == "extra_forbidden":
        return (
            "not a spec field; remove it (fields: project_id, task_type, target, primary_metric, golden_dataset, "
            "constraints, splits, baseline, compute_budget, acceptance, medallion, generation)",
            None,
        )
    if kind == "missing":
        return "a required field of the spec schema; add it", None
    return "see errors[0] for the rule that failed", None


def spec_error_details(issues: Any) -> tuple[list[dict[str, str | None]], int]:
    """Bounded, deterministic ``error_details`` for structured ``(loc, type)`` issues, plus the omitted count."""
    details: list[dict[str, str | None]] = []
    seen: set[str] = set()
    omitted = 0
    for issue in issues or ():
        try:
            loc, kind = issue
        except (TypeError, ValueError):
            continue
        kind = kind if isinstance(kind, str) else "value_error"
        path = _DECLARATION_PATH if kind == "generation_declaration_mismatch" else canonical_path(loc)
        if path in seen:
            continue
        seen.add(path)
        if len(details) >= MAX_DETAILS:
            omitted += 1
            continue
        expected, example = _advice(path, kind)
        details.append(
            {
                "path": path,
                "problem": _PROBLEM.get(kind, "invalid value"),
                "expected": expected,
                "example": example,
            }
        )
    return details, omitted


def _is_coding(text: str, details: list[dict[str, str | None]]) -> bool:
    if any(
        item["path"].startswith(("generation", "model_catalog")) or "pass_at_1" in item["path"]
        for item in details
    ):
        return True
    try:
        import frontmatter

        metadata = dict(frontmatter.loads(text).metadata)
    except Exception:
        return False
    return metadata.get("task_type") == "text_generation" or "generation" in metadata


def failure_guidance(yaml_text: str, error: Any) -> dict[str, Any]:
    """Fields ``validate_spec`` adds to an invalid result (``error`` is the ``SpecLoadError``)."""
    if getattr(error, "kind", "schema") == "front_matter":
        record = {
            "path": "(front matter)",
            "problem": "the text has no readable YAML front matter",
            "expected": "a line ---, YAML fields, a line ---, then free notes",
            "example": "---\nproject_id: example_project\n---\n\nnotes",
        }
        return {"error_details": [record], "next_step": FRONT_MATTER_NEXT_STEP}
    details, omitted = spec_error_details(getattr(error, "issues", ()))
    result: dict[str, Any] = {
        "error_details": details,
        "next_step": INVALID_CODING_NEXT_STEP if _is_coding(yaml_text, details) else INVALID_NEXT_STEP,
    }
    if omitted:
        result["error_details_omitted"] = omitted
    return result
