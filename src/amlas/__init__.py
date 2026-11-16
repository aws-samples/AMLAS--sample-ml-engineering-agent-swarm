"""amlas — multi-agent ML engineering on AgentCore + Strands."""

from __future__ import annotations

import os
import sys

from ._runtime_guard import apply_openmp_safety_guard


def __getattr__(name: str) -> str:
    """Resolve ``__version__`` on first access; nothing about the version runs at import."""
    if name == "__version__":
        from ._version import package_version

        return package_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# macOS duplicate-libomp OpenMP crash guard (found in  e2e).
# ─────────────────────────────────────────────────────────────────
# Every entrypoint — the CLI (``python -m amlas.cli`` / ``bin/agentcli``), the
# TUI (``python -m amlas.tui`` / ``bin/agenttui``) and the demo scripts — imports
# this package FIRST, before any numpy/scipy/sklearn/xgboost import is triggered
# through ``amlas.*``. On macOS a duplicate ``libomp.dylib`` runtime otherwise
# SIGSEGVs the interpreter the moment an OpenMP-parallel learner (e.g. xgboost)
# enters a parallel region — which crashed ``amlas run train --algorithm
# xgboost`` in the  A4AI e2e. The guard applies the production-safe
# neutralisation at startup: it is macOS-only (Linux/SageMaker throughput
# is untouched) and uses ``setdefault`` (an explicit operator ``OMP_NUM_THREADS``
# always wins), so it is fully behaviour-preserving. See ``_runtime_guard`` for
# the proven root-cause analysis.
apply_openmp_safety_guard()


def _load_dotenv() -> None:
    """Load the SHARED-BASE root ``.env`` into ``os.environ`` at package import.

    CENTRALIZED-ENV (project standard): every entrypoint — the TUI
    (``python -m amlas.tui`` / ``bin/agenttui``) and the CLI
    (``python -m amlas.cli`` / ``bin/agentcli``) — imports this
    package first, so honoring the root ``.env`` here means ONE gitignored file
    configures the whole app: the chat tier's AWS profile/region (which decide
    whether its Strands agent can reach Bedrock at all), plus the lake / model-group /
    AgentCore-runtime / Tavily-secret references the tools resolve at runtime.

    This is the ROOT layer of the layered-config system (see
    ``config/env_loader.py``). The per-entrypoint overlays — ``.env.tui`` (TUI),
    ``.env.agentcore`` (AgentCore/Strands runtime), ``.env.swarm`` (agent swarm)
    — are applied ON TOP of this base by each entrypoint via
    ``load_layered_env(<layer>)``.

    Discipline:
      * Shell wins — a value already exported in the launching shell (e.g.
        ``AWS_PROFILE=<your-profile> uv run ...``) ALWAYS beats every ``.env``.
      * Best-effort — if ``python-dotenv`` is missing or no ``.env`` exists,
        this is a silent no-op; the app still runs on shell env + the layered
        ``.agenticml.yaml`` config.
    """
    try:
        from .config.env_loader import load_root_env
    except Exception:
        return
    try:
        load_root_env()
    except Exception:
        return


def _expand_runtime_environment() -> None:
    """Expand the deployment's environment pack before any reader runs.

    The managed runtime refuses environments over 4,000 bytes, so a large
    selection arrives as one checked pack (``_runtime_env_pack``). Expansion only
    sets the restored keys and removes the two pack keys; a refusal raises before
    anything changes and names no value. Without a pack this is a no-op.
    """
    from ._runtime_env_pack import PACK_KEY, expand_environ

    try:
        expanded = expand_environ(os.environ)
    except ValueError as error:
        raise RuntimeError(f"{PACK_KEY} refused: {error}") from None
    if expanded is not None:
        count, digest = expanded
        sys.stderr.write(f"RUNTIME_ENV_PACK_EXPANDED keys={count} sha256={digest}\n")


def _initialize_environment() -> None:
    """Deployment values first (as shell values, they win), then the root ``.env`` layer."""
    from ._runtime_env_pack import PACK_KEY, PACK_SHA256_KEY

    _expand_runtime_environment()
    _load_dotenv()
    if PACK_KEY in os.environ or PACK_SHA256_KEY in os.environ:
        raise RuntimeError(f"{PACK_KEY} refused: only the process environment may carry a pack")


_initialize_environment()
