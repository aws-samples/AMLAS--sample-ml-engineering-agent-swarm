"""Private finite research plan DATA validation; no approval or native authority.

Trusted source must bind and qualify a plan before execution. These validators
check declared structure/URLs only, not semantic independence, DNS/redirects or
owner approval. Original history remains intact and unknown/producer holds have
no recovery path through data alone.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class _ResearchUnit:
    unit_id: str
    origin_id: str
    urls: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    variant_of: str | None = None


@dataclass(frozen=True)
class _ResearchPlan:
    units: tuple[_ResearchUnit, ...]


_WITHHELD = (
    "Research source content was withheld by input admission. Do not retry this request or source. "
    "Continue only with independent approved research."
)
_CANCELLED = "Research request is unavailable at the input admission boundary."


def _identity(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ValueError("bounded research identity required")


def _primary_url(value):
    if (
        type(value) is not str
        or not 0 < len(value) <= 2048
        or any(ord(char) < 33 or ord(char) > 126 for char in value)
    ):
        raise ValueError("bounded primary research URL required")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.port not in (None, 443)
            or parsed.netloc != parsed.netloc.lower()
            or host.endswith(".")
            or host in {"localhost", "metadata.amazonaws.com"}
            or host.endswith((".localhost", ".local", ".internal"))
        ):
            raise ValueError("fixed primary HTTPS source required")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if "." not in host or any(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part) is None
                for part in host.split(".")
            ):
                raise ValueError("public primary research host required") from None
        else:
            if not address.is_global:
                raise ValueError("public primary research address required")
    except (ValueError, UnicodeError):
        raise ValueError("fixed primary research URL refused") from None


def _validate_plan(plan):
    if type(plan) is not _ResearchPlan or type(plan.units) is not tuple or not 1 <= len(plan.units) <= 8:
        raise ValueError("exact bounded private research plan required")
    units = {}
    for unit in plan.units:
        if type(unit) is not _ResearchUnit:
            raise ValueError("exact private research unit required")
        _identity(unit.unit_id)
        _identity(unit.origin_id)
        if unit.unit_id in units:
            raise ValueError("duplicate research unit refused")
        if type(unit.urls) is not tuple or not 1 <= len(unit.urls) <= 3:
            raise ValueError("closed primary research sources required")
        for url in unit.urls:
            _primary_url(url)
        if len(set(unit.urls)) != len(unit.urls):
            raise ValueError("duplicate primary research source refused")
        if type(unit.dependencies) is not tuple or len(unit.dependencies) > 8:
            raise ValueError("closed research dependency graph required")
        for dependency in unit.dependencies:
            _identity(dependency)
        if len(set(unit.dependencies)) != len(unit.dependencies):
            raise ValueError("duplicate research dependency refused")
        if unit.variant_of is not None:
            _identity(unit.variant_of)
        units[unit.unit_id] = unit
    if set(units) & {unit.origin_id for unit in units.values()}:
        raise ValueError("research origin/unit namespaces differ")
    edges = {
        name: unit.dependencies + (() if unit.variant_of is None else (unit.variant_of,))
        for name, unit in units.items()
    }
    if any(edge not in units for values in edges.values() for edge in values):
        raise ValueError("complete research dependency graph required")
    active, visited = set(), set()

    def visit(name):
        if name in active:
            raise ValueError("acyclic research dependency graph required")
        if name in visited:
            return
        active.add(name)
        for dependency in edges[name]:
            visit(dependency)
        active.remove(name)
        visited.add(name)

    for name in units:
        visit(name)
    return units
