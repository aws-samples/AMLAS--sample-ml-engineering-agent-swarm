"""Stage-4 collection specialist: inventory, metadata, gated acquisition and polling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import DATA_COLLECTION_AGENT, build_subagent


def build_data_collection_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "data_collection_agent",
) -> Agent:
    """Build the bounded collection specialist; acquisition enforces caller authority."""
    return build_subagent(
        DATA_COLLECTION_AGENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
