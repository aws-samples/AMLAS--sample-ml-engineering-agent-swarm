"""Strands @tool wrappers for spec generation workflow.

Tools:
  - validate_spec: Validates YAML front-matter + body against SpecModel without saving
  - save_spec: Validates + writes spec to disk
  - list_example_specs: Returns available exemplar specs from specs/ directory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Phase E2 FIX-1 — decouple the pure-filesystem tools from the agent runtime.
    from strands import tool
except ModuleNotFoundError:  # pragma: no cover - exercised via the strands-absent test
    # ``list_example_specs`` / ``validate_spec`` / ``save_spec`` are plain
    # functions (pure disk/validation work); they do NOT need the Strands agent
    # runtime. On a partial/minimal install where ``strands`` is un-importable
    # they must still be callable directly (the reported "`/spec list` doesn't
    # work offline" bug — audit §3.2 / FIX-1). This no-op passthrough decorator
    # lets the module import + the functions run without ``strands`` present,
    # while a full install still wraps them as real ``@tool``s (behavior
    # unchanged when ``strands`` IS importable).
    def tool(fn: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        """Minimal passthrough ``@tool`` fallback (used only when ``strands`` is absent)."""
        return fn if fn is not None else (lambda f: f)


from ..errors import tool_error_boundary
from ..spec import SpecLoadError, load_spec_from_text
from .spec_guidance import VALID_CODING_NEXT_STEP, failure_guidance

# Specs directory — sibling to src/ at repo root
SPECS_DIR = Path(__file__).resolve().parents[3] / "specs"

_EXAMPLES_NEXT_STEP = (
    "These examples are classification specs. For the fixed Qwen coding task do not copy them: call "
    "generate_spec_from_description with planning_data and bind its spec_text unchanged."
)


@tool
@tool_error_boundary(source="tools.spec_tools")
def validate_spec(yaml_text: str) -> dict[str, Any]:
    """Validate a spec YAML front-matter + markdown body against the SpecModel schema.

    Returns a dict with:
      - valid: bool — whether the spec is valid
      - spec: dict — the validated spec fields (if valid)
      - errors: list[str] — validation error messages (if invalid)
      - error_details: list — when invalid, one {path, problem, expected, example} per failing path
      - next_step: str — the next call. For the fixed Qwen coding task a hand-edited spec is rarely needed: the
        spec_text of generate_spec_from_description (called with planning_data) is valid and binds unchanged.

    Args:
        yaml_text: Full spec text with YAML front-matter delimited by --- and markdown body.
    """
    if not yaml_text or not yaml_text.strip():
        return {"valid": False, "errors": ["Empty input — spec text must contain YAML front-matter."]}

    try:
        spec = load_spec_from_text(yaml_text)
        result: dict[str, Any] = {
            "valid": True,
            "spec": spec.model_dump(),
            "errors": [],
        }
        if spec.task_type == "text_generation" and spec.generation is not None:
            result["next_step"] = VALID_CODING_NEXT_STEP
        return result
    except SpecLoadError as e:
        return {
            "valid": False,
            "errors": [str(e)],
            **failure_guidance(yaml_text, e),
        }


@tool(context=True)
@tool_error_boundary(source="tools.spec_tools")
def save_spec(yaml_text: str, path: str, tool_context: Any = None) -> dict[str, Any]:
    """Validate a spec and save it to disk if valid.

    Will NOT write the file if validation fails. Returns success/failure status.

    Args:
        yaml_text: Full spec text with YAML front-matter delimited by --- and markdown body.
        path: Filesystem path where the spec file should be written.
    """
    if not yaml_text or not yaml_text.strip():
        return {"saved": False, "errors": ["Empty input — nothing to save."]}

    try:
        # Validate first
        load_spec_from_text(yaml_text)
    except SpecLoadError as e:
        return {"saved": False, "errors": [str(e)]}

    from ..control_plane.campaign_runtime import current_campaign_entry

    state = getattr(tool_context, "invocation_state", None)
    state = state if type(state) is dict else {}
    hook = state.get("_amlas_campaign_planning_store")
    if current_campaign_entry() is not None or "_amlas_campaign_planning_store" in state:
        from ..control_plane.campaign_planning import CampaignPlanningStore

        if type(hook) is not CampaignPlanningStore:
            return {"saved": False, "published": False, "status": "denied", "errors": []}
        try:
            reference = hook.publish_validated_spec(yaml_text)
        except Exception as error:
            return {
                "saved": False,
                "published": False,
                "status": "refused",
                "error_type": type(error).__name__,
                "errors": [],
            }
        return {
            "saved": False,
            "published": True,
            "status": "planning_data",
            "planning_ref": reference.reference,
            "spec_sha256": reference.sha256,
            "errors": [],
        }

    # Write to disk — traversal containment (Security review, CWE-22): the
    # agent-supplied path must stay inside the sandbox/CWD/tempdir roots.
    from .pathref import resolve_dataset_path

    try:
        p = resolve_dataset_path(path)
    except PermissionError as e:
        return {"saved": False, "errors": [str(e)]}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml_text, encoding="utf-8")
    return {"saved": True, "path": str(p), "errors": []}


@tool
@tool_error_boundary(source="tools.spec_tools")
def list_example_specs() -> dict[str, Any]:
    """Return all available example specs from the specs/ directory.

    Each entry includes the spec filename and its full content, suitable for
    few-shot prompting the spec writer agent. Only files with YAML front matter are specs. The examples
    are classification specs, not the fixed Qwen coding task.
    """
    specs: list[dict[str, str]] = []
    if SPECS_DIR.exists():
        for spec_file in sorted(SPECS_DIR.glob("*.md")):
            content = spec_file.read_text(encoding="utf-8")
            if not content.lstrip().startswith("---"):
                continue  # the directory guide (AGENTS.md) is not a spec
            specs.append({"name": spec_file.stem, "content": content})
    return {"specs": specs, "next_step": _EXAMPLES_NEXT_STEP}
