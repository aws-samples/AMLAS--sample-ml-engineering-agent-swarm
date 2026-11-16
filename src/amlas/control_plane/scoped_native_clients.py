"""Context-private read-only client views for the adapters a qualification scope wraps.

`normal_run_parents._qualified_reads` gives the code inside its `with` block read-only, per-call qualified
clients. It used to do that by replacing `_sm` and `_s3` on the adapter object itself. The adapters are shared
by the whole session: the tools of one assistant message run on worker threads (`asyncio.to_thread`) while the
BeforeToolCall fence of a later tool runs its qualification inline on the event-loop thread. A tool that
reached for its write-capable client during that window got the read-only proxy
(`AttributeError: create_multipart_upload`), and two overlapping scopes restored the clients in the wrong
order and could leave a proxy behind.

Here the view lives in a `ContextVar`. `_sm` and `_s3` become data descriptors on the adapter class that return
the view only in the context that entered the scope and the raw client everywhere else, so every read of
`adapter._clients()`, `adapter._sm` and `adapter._s3` in the scope's own context is qualified and no other
thread or task sees it. A context copied into a worker inside the scope (`asyncio.to_thread`,
`contextvars.copy_context().run`) keeps the read-only view; a plain `threading.Thread` starts with an empty
context and gets the raw clients, which is why nothing on the qualification path may start one (none does).
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

_NAMES = ("_sm", "_s3")
_MISSING = object()
# adapter identity -> (adapter, {"_sm": client, "_s3": client}); replaced, never mutated, so a context
# copied from this one cannot change the original's mapping.
_VIEWS: ContextVar[Mapping[int, tuple[Any, Mapping[str, Any]]]] = ContextVar(
    "amlas_scoped_native_clients", default=MappingProxyType({})
)
_INSTALL = threading.Lock()


class _ScopedClient:
    """Data descriptor: the context's read-only view if one is entered for this adapter, else the raw client."""

    def __init__(self, name: str, default: Any = _MISSING) -> None:
        self.name, self.default = name, default

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None:
            return self
        view = _VIEWS.get().get(id(obj))
        if view is not None and view[0] is obj:
            return view[1][self.name]
        try:
            return obj.__dict__[self.name]
        except KeyError:
            if self.default is _MISSING:
                raise AttributeError(self.name) from None
            return self.default

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__[self.name] = value  # lazy construction and injection always set the raw client

    def __delete__(self, obj: Any) -> None:
        try:
            del obj.__dict__[self.name]
        except KeyError:
            raise AttributeError(self.name) from None


def _install(cls: type) -> None:
    """Idempotent. The raw client stays in the instance __dict__ under the same name, so a thread that reads it
    while the descriptor is being installed sees the same object before and after."""
    for name in _NAMES:
        if isinstance(inspect.getattr_static(cls, name, None), _ScopedClient):
            continue
        with _INSTALL:
            current = inspect.getattr_static(cls, name, _MISSING)
            if not isinstance(current, _ScopedClient):
                setattr(cls, name, _ScopedClient(name, current))


@contextmanager
def scoped_clients(adapter: Any, sagemaker_client: Any, s3_client: Any) -> Iterator[None]:
    """Show adapter._sm and adapter._s3 as these clients to the current context only, until exit."""
    _install(type(adapter))
    views = _VIEWS.get()
    token = _VIEWS.set(
        MappingProxyType({**views, id(adapter): (adapter, {"_sm": sagemaker_client, "_s3": s3_client})})
    )
    try:
        yield
    finally:
        _VIEWS.reset(token)
