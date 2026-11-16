"""Shared helpers for the M0 adapters.

The adapters WRAP existing tool functions (the strangler pattern, doc 05 §4.4):
today's behavior becomes the default/local adapter, byte-for-byte. The existing
``@tool``-decorated functions expose their raw callable at ``._tool_func`` (the
convention the existing callers may use, e.g. ``read_lake_slice._tool_func(...)``);
:func:`raw_tool` returns it so an adapter delegates to the SAME code path the tool
ran before — guaranteeing identical outputs (the behavior-preserving invariant).
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


def raw_tool(tool: Any) -> Callable[..., Any]:
    """Return the undecorated callable behind a Strands ``@tool`` (or the fn itself).

    Strands wraps a ``@tool`` function and exposes the original at ``._tool_func``;
    plain functions are returned unchanged. This lets an adapter call the exact
    existing implementation (no behavior change) without going through tool-spec
    validation.
    """
    fn: Callable[..., Any] = getattr(tool, "_tool_func", tool)
    return fn


def _csv_safe_cell(value: Any) -> Any:
    """Prevent spreadsheet formula execution while preserving non-string values."""
    if not isinstance(value, str):
        return value
    candidate = value.lstrip(" ")
    if candidate.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


_PROCESSING_CREATE_PID = os.getpid()
_PROCESSING_CREATE_LOCK = threading.Lock()
_PROCESSING_CREATE_STATES: dict[tuple[str, str], list[Any]] = {}
_PROCESSING_CREATE_INTERVAL = 1.25
_processing_create_now = time.monotonic
_processing_create_sleep = time.sleep


@contextmanager
def _initial_processing_create(account: str, region: str):
    """Pace only initial dataset/model Creates in this process, never retry one."""
    if os.getpid() != _PROCESSING_CREATE_PID:
        raise ValueError("inherited initial Processing Create state refused")
    if (
        type(account) is not str
        or not re.fullmatch("[0-9]{12}", account)
        or type(region) is not str
        or not re.fullmatch("[a-z]{2}(?:-[a-z]+)+-[0-9]", region)
    ):
        raise ValueError("verified Processing account and Region required")
    key = (account, region)
    with _PROCESSING_CREATE_LOCK:
        if os.getpid() != _PROCESSING_CREATE_PID:
            raise ValueError("inherited initial Processing Create state refused")
        state = _PROCESSING_CREATE_STATES.setdefault(key, [threading.Lock(), None])
    with state[0]:
        if os.getpid() != _PROCESSING_CREATE_PID:
            raise ValueError("inherited initial Processing Create state refused")
        now = _processing_create_now()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError("finite initial Processing Create clock required")
        prior = state[1]
        if prior is not None:
            if type(prior) not in (int, float) or not math.isfinite(prior):
                raise ValueError("original Processing Create completion clock unresolved")
            if now < prior:
                raise ValueError("initial Processing Create clock moved backwards")
            remaining = prior + _PROCESSING_CREATE_INTERVAL - now
            if remaining > 0:
                _processing_create_sleep(remaining)
                now = _processing_create_now()
                if (
                    type(now) not in (int, float)
                    or not math.isfinite(now)
                    or now < prior + _PROCESSING_CREATE_INTERVAL
                ):
                    raise ValueError("initial Processing Create wait incomplete")
        if os.getpid() != _PROCESSING_CREATE_PID:
            raise ValueError("inherited initial Processing Create state refused")
        # Reserve before qualification and retain it on refusal or a lost native reply.
        state[1] = now
        try:
            yield
        finally:
            # Conservative completion-based spacing covers arbitrarily slow final
            # qualification and SDK calls. Never mask their original result/error.
            try:
                completed = _processing_create_now()
                if type(completed) not in (int, float) or not math.isfinite(completed) or completed < now:
                    state[1] = float("inf")
                else:
                    state[1] = completed
            except Exception:
                state[1] = float("inf")
