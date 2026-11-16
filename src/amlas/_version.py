"""Report the application's own version without stating it in the source.

Root ``VERSION`` is the one source and holds ``N.NN``; ``pyproject.toml`` carries the mapped package
version ``N.<NN as an integer>.0``. A source checkout reads ``VERSION``, an installed package reads its
own distribution metadata, and a copy with neither reports ``unknown``. Standard library only; nothing
here runs when ``amlas`` is imported.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

DISTRIBUTION = "agentic-ml-lifecycle-automation-surface"
UNKNOWN = "unknown"

_VERSION_TEXT = re.compile(r"([0-9]+)\.([0-9]{2,})\n?")


def _checkout_version() -> str | None:
    """Map root ``VERSION`` when this package runs from ``<checkout>/src/amlas``; otherwise ``None``."""
    package = Path(__file__).resolve().parent
    if package.parent.name != "src":
        return None
    try:
        text = (package.parent.parent / "VERSION").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    match = _VERSION_TEXT.fullmatch(text)
    if match is None:
        return None
    return f"{int(match.group(1))}.{int(match.group(2))}.0"


def package_version() -> str:
    """Return the package version: mapped from ``VERSION`` in a checkout, else the installed metadata."""
    found = _checkout_version()
    if found is not None:
        return found
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return UNKNOWN
