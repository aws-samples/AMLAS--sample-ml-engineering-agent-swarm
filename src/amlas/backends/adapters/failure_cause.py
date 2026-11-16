"""Closed cause fields for refusal log lines, and the marker of a designed durable-record recheck.

`record_changed(text, reason)` builds the error that a durable-record recheck raises when the record it re-read differs
from the one its operation started with, or when the selected SFT catalog moved. It stays an exact `ValueError` with the
same single text, so every handler that matches `type(error) is ValueError` and the text keeps working. The private
marker lets a byte verifier tell its caller's own recheck from a refusal of the bytes: `record_change(error)` returns the
reason for exactly such an error and None for anything else (a subclass, a forged reason, an error raised `from` it).

`cause_fields(error)` names a caught exception for a log line and never carries its message, which can hold identifiers:
the nearest class of its MRO in a closed vocabulary (`Exception` when none), the error code of a botocore `ClientError`
from a closed vocabulary (`other` outside it, `none` for any other class), the innermost traceback frame inside the
`amlas` package as `file.py:line` (`unknown:0` without one), and the closed class of the root of an explicit
`raise ... from` chain (`none` without one). `closed_cause` and `closed_interruption` re-check such fields where a log
line is rendered.
"""

from __future__ import annotations

import builtins
import os
import re
from typing import Any

RUN_RECORD_CHANGED = "run_record_changed"
SFT_CATALOG_CHANGED = "sft_catalog_changed"
RECORD_CHANGE_REASONS = frozenset({RUN_RECORD_CHANGED, SFT_CATALOG_CHANGED})
_MARKER = "_amlas_record_change"
_BUILTIN_CLASSES = (
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "LookupError",
    "AttributeError",
    "RuntimeError",
    "RecursionError",
    "NotImplementedError",
    "AssertionError",
    "OSError",
    "PermissionError",
    "FileNotFoundError",
    "TimeoutError",
    "ConnectionError",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "ConnectionRefusedError",
    "BrokenPipeError",
    "UnicodeError",
    "ArithmeticError",
    "OverflowError",
    "ZeroDivisionError",
    "MemoryError",
)
_BOTOCORE_CLASSES = (
    "ClientError",
    "EndpointConnectionError",
    "ConnectionClosedError",
    "ConnectTimeoutError",
    "ReadTimeoutError",
    "ParamValidationError",
    "NoCredentialsError",
    "CredentialRetrievalError",
    "IncompleteReadError",
    "ResponseStreamingError",
    "HTTPClientError",
    "ProxyConnectionError",
    "SSLError",
    "BotoCoreError",
)
NATIVE_CODES = frozenset(
    {
        "Throttling",
        "ThrottlingException",
        "ThrottledException",
        "RequestThrottled",
        "RequestThrottledException",
        "TooManyRequestsException",
        "ProvisionedThroughputExceededException",
        "RequestLimitExceeded",
        "SlowDown",
        "InternalError",
        "InternalServerError",
        "InternalFailure",
        "InternalServerException",
        "ServiceUnavailable",
        "ServiceUnavailableException",
        "RequestTimeout",
        "RequestTimeTooSkewed",
        "AccessDenied",
        "AccessDeniedException",
        "ExpiredToken",
        "ExpiredTokenException",
        "InvalidClientTokenId",
        "NoSuchKey",
        "NoSuchVersion",
        "NoSuchBucket",
        "PreconditionFailed",
        "ConditionalCheckFailedException",
        "ResourceNotFoundException",
        "ValidationException",
        "400",
        "403",
        "404",
        "412",
        "500",
        "503",
    }
)
CLASS_NAMES = frozenset((*_BUILTIN_CLASSES, *_BOTOCORE_CLASSES, "Exception"))
_DEFAULT_CAUSE = ("Exception", "none", "unknown:0", "none")
_SITE = re.compile(r"[a-z0-9_]+\.py:[0-9]+")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_REVIEWED: dict[type, str] | None = None


def record_changed(text: str, reason: str) -> ValueError:
    """The designed recheck error: an exact ValueError with `text`, marked with a closed reason."""
    if type(text) is not str or not text or reason not in RECORD_CHANGE_REASONS:
        raise ValueError("closed record-change text and reason required")
    error = ValueError(text)
    setattr(error, _MARKER, reason)
    return error


def record_change(error: Any) -> str | None:
    """The reason of an error built by `record_changed`, or None."""
    if type(error) is not ValueError:
        return None
    reason = getattr(error, _MARKER, None)
    return reason if type(reason) is str and reason in RECORD_CHANGE_REASONS else None


def _reviewed() -> dict[type, str]:
    global _REVIEWED
    if _REVIEWED is None:
        reviewed = {getattr(builtins, name): name for name in _BUILTIN_CLASSES}
        try:
            from botocore import exceptions
        except ImportError:  # pragma: no cover - botocore is a runtime dependency
            exceptions = None
        for name in _BOTOCORE_CLASSES if exceptions is not None else ():
            kind = getattr(exceptions, name, None)
            if isinstance(kind, type):
                reviewed[kind] = name
        _REVIEWED = reviewed
    return _REVIEWED


def closed_class(error: Any) -> str:
    """The nearest reviewed class in the MRO of `error`, else `Exception`."""
    reviewed = _reviewed()
    for kind in type(error).__mro__:
        name = reviewed.get(kind)
        if name is not None:
            return name
    return "Exception"


def native_code(error: Any) -> str:
    try:
        from botocore.exceptions import ClientError
    except ImportError:  # pragma: no cover - botocore is a runtime dependency
        return "none"
    if not isinstance(error, ClientError):
        return "none"
    response = error.response if type(error.response) is dict else {}
    fields = response.get("Error") if type(response.get("Error")) is dict else {}
    code = fields.get("Code")
    return code if type(code) is str and code in NATIVE_CODES else "other"


def code_site(error: Any) -> str:
    """`file.py:line` of the innermost traceback frame of `error` inside the amlas package, else `unknown:0`."""
    found = None
    trace = getattr(error, "__traceback__", None)
    for _ in range(512):
        if trace is None:
            break
        path = os.path.realpath(trace.tb_frame.f_code.co_filename)
        if path.startswith(_ROOT + os.sep):
            found = os.path.basename(path) + ":" + str(trace.tb_lineno)
        trace = trace.tb_next
    return found if found is not None and _SITE.fullmatch(found) is not None else "unknown:0"


def cause_fields(error: Any) -> tuple[str, str, str, str]:
    """(exception_class, native_code, site, cause_class) of a caught exception; closed values only."""
    root = error
    for _ in range(8):
        cause = getattr(root, "__cause__", None)
        if cause is None:
            break
        root = cause
    return (
        closed_class(error),
        native_code(error),
        code_site(error),
        "none" if root is error else closed_class(root),
    )


def _site_or_unknown(value: Any) -> str:
    return value if type(value) is str and (value == "unknown:0" or _SITE.fullmatch(value) is not None) else "unknown:0"


def closed_cause(value: Any) -> tuple[str, str, str, str]:
    """`value` when it is a closed (exception_class, native_code, site, cause_class), else the closed defaults."""
    if type(value) is not tuple or len(value) != 4 or any(type(item) is not str for item in value):
        return _DEFAULT_CAUSE
    name, code, site, root = value
    if (
        name not in CLASS_NAMES
        or (code not in NATIVE_CODES and code not in ("none", "other"))
        or _site_or_unknown(site) != site
        or (root not in CLASS_NAMES and root != "none")
    ):
        return _DEFAULT_CAUSE
    return value


def closed_interruption(value: Any) -> tuple[str, str] | None:
    """(reason, site) of a closed interruption cause, with an unclosed site read `unknown:0`; None for anything else."""
    if type(value) is not tuple or len(value) != 2 or value[0] not in RECORD_CHANGE_REASONS:
        return None
    return value[0], _site_or_unknown(value[1])
