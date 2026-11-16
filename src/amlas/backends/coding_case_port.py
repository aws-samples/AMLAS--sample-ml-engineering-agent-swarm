"""Dedicated fixed data-only case freeze; resources and sources stay private."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import JobHandle, JobStatus


@runtime_checkable
class CodingCaseFreezeBackend(Protocol):
    def submit(self, *, source: Any, catalog: Any, runtime_session_id: str) -> JobHandle: ...
    def status(self, handle: JobHandle) -> JobStatus: ...
    def artifacts(self, handle: JobHandle) -> dict: ...
