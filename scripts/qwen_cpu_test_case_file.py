#!/usr/bin/env python3
"""Provide the one pinned public evaluation file that the complete CPU tests read."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import http.client
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

ENVIRONMENT_NAME = "AMLAS_TEST_HUMANEVAL_PARQUET"
PROFILE_RELATIVE = "src/amlas/data_eng/coding_case_profile.py"
_PIN_NAMES = ("SOURCE_REPO", "SOURCE_REVISION", "SOURCE_FILENAME", "SOURCE_SHA256", "SOURCE_BYTES")
_MAX_BYTES = 16 * 1024 * 1024
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_DELAYS = (1.0, 3.0)
_REQUEST_HEADERS = {"User-Agent": "amlas-cpu-test-provisioning", "Accept-Encoding": "identity"}
_NAME_PART = r"[A-Za-z0-9][A-Za-z0-9._-]*"


class CaseFileError(RuntimeError):
    """The pinned public file could not be provided; nothing was written under the final name."""


class CasePins(NamedTuple):
    repo: str
    revision: str
    filename: str
    sha256: str
    byte_count: int


def _checked(pins: CasePins) -> CasePins:
    if (
        any(type(value) is not str for value in pins[:4])
        or type(pins.byte_count) is not int
        or not re.fullmatch(f"{_NAME_PART}/{_NAME_PART}", pins.repo)
        or not re.fullmatch(r"[0-9a-f]{40}", pins.revision)
        or not re.fullmatch(f"{_NAME_PART}(/{_NAME_PART})*", pins.filename)
        or not re.fullmatch(r"[0-9a-f]{64}", pins.sha256)
        or not 0 < pins.byte_count <= _MAX_BYTES
    ):
        raise CaseFileError("product source constants have an unexpected shape")
    return pins


def product_pins(project: Path) -> CasePins:
    """Read the product's source constants as plain literals; the package is not imported."""
    path = Path(project) / PROFILE_RELATIVE
    try:
        body = ast.parse(path.read_text(encoding="utf-8")).body
    except (OSError, SyntaxError, ValueError) as error:
        raise CaseFileError(f"product source constants unreadable: {PROFILE_RELATIVE}") from error
    found: dict[str, object] = {}
    for node in body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in _PIN_NAMES:
            continue
        if target.id in found or not isinstance(node.value, ast.Constant):
            raise CaseFileError(f"product source constants: {target.id} must be one plain literal")
        found[target.id] = node.value.value
    if set(found) != set(_PIN_NAMES):
        raise CaseFileError("product source constants incomplete")
    return _checked(CasePins(*(found[name] for name in _PIN_NAMES)))


def source_url(pins: CasePins) -> str:
    """The public resolve address of the pinned revision."""
    return f"https://huggingface.co/datasets/{pins.repo}/resolve/{pins.revision}/{pins.filename}"


def _matches(raw: bytes, pins: CasePins) -> bool:
    return len(raw) == pins.byte_count and hashlib.sha256(raw).hexdigest() == pins.sha256


def _read_bounded(stream, limit: int) -> bytes:
    """At most one byte past the pinned size, so a longer source is seen and never stored."""
    parts = []
    total = 0
    while total <= limit:
        part = stream.read(min(65536, limit + 1 - total))
        if not part:
            break
        parts.append(part)
        total += len(part)
    return b"".join(parts)


def _read_file(path: Path, pins: CasePins) -> bytes | None:
    try:
        with open(path, "rb") as stream:
            return _read_bounded(stream, pins.byte_count)
    except OSError:
        return None


class _SchemeGuard(urllib.request.HTTPRedirectHandler):
    def __init__(self, schemes: tuple[str, ...]):
        self._schemes = schemes

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme not in self._schemes:
            fp.close()
            raise CaseFileError("redirect to a scheme outside the allowed set refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _require_source_target(request: urllib.request.Request, url: str, schemes: tuple[str, ...]) -> None:
    """The object about to be sent must still name the source address, its host and an allowed scheme."""
    source = urlsplit(url)
    if (
        request.full_url != url
        or request.type != source.scheme
        or request.type not in schemes
        or request.host != source.netloc
    ):
        raise CaseFileError("request with another scheme or away from the source host refused")


def _fetch(
    url: str, pins: CasePins, *, schemes: tuple[str, ...], timeout: float, attempts: int, pause
) -> bytes:
    if urlsplit(url).scheme not in schemes:
        raise CaseFileError("source scheme outside the allowed set refused")
    opener = urllib.request.build_opener(_SchemeGuard(schemes))
    reason = "no attempt made"
    made = 0
    for attempt in range(attempts):
        if attempt:
            pause(_DELAYS[min(attempt, len(_DELAYS)) - 1])
        made += 1
        # A new object for every attempt: the standard library's proxy handler rewrites the object it
        # sends, and a second send of the same object can leave for port 80 without TLS.
        request = urllib.request.Request(url, headers=dict(_REQUEST_HEADERS))
        _require_source_target(request, url, schemes)
        try:
            with opener.open(request, timeout=timeout) as response:
                if urlsplit(response.geturl()).scheme not in schemes:
                    raise CaseFileError("answer from a final address outside the allowed schemes refused")
                return _read_bounded(response, pins.byte_count)
        except urllib.error.HTTPError as error:
            reason = f"HTTP {error.code}"
            error.close()
            if error.code not in _RETRY_STATUS:
                break
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            reason = f"{type(error).__name__}: {getattr(error, 'reason', error)}"
    raise CaseFileError(f"fetch failed after {made} attempt{'s' if made != 1 else ''}: {reason}")


def _write_verified(destination: Path, raw: bytes) -> None:
    parent = destination.parent
    created = not parent.exists()
    if created:
        parent.mkdir(mode=0o700)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(dir=parent, prefix=".case-file-")
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            if temporary is not None:
                os.unlink(temporary)
            if created:
                parent.rmdir()
        raise


def provide(
    destination: Path,
    pins: CasePins,
    *,
    existing: str | None = None,
    url: str | None = None,
    schemes: tuple[str, ...] = ("https",),
    timeout: float = 60.0,
    attempts: int = 3,
    pause=time.sleep,
) -> dict:
    """Leave the exact pinned bytes at ``destination`` or raise; a refusal writes nothing.

    ``existing`` names a file the caller already holds: when it has the pinned size
    and hash its bytes are used and no request is made; when it differs the call refuses.
    """
    destination = Path(destination)
    pins = _checked(pins)
    if not destination.is_absolute():
        raise CaseFileError("absolute destination required")
    if os.path.lexists(destination):
        raw = None if destination.is_symlink() or not destination.is_file() else _read_file(destination, pins)
        if raw is None or not _matches(raw, pins):
            raise CaseFileError("existing destination differs from the pinned public file; left as it is")
        source = "present"
    else:
        if existing:
            raw = _read_file(Path(existing), pins)
            if raw is None or not _matches(raw, pins):
                raise CaseFileError(
                    f"{ENVIRONMENT_NAME} names a file that is absent or differs from the pinned public file "
                    f"({pins.byte_count} bytes, sha256 {pins.sha256}); unset it or name the exact file"
                )
            source = "reused"
        else:
            raw = _fetch(
                source_url(pins) if url is None else url,
                pins,
                schemes=tuple(schemes),
                timeout=timeout,
                attempts=attempts,
                pause=pause,
            )
            if len(raw) != pins.byte_count:
                raise CaseFileError(f"fetched size differs from the pinned {pins.byte_count} bytes")
            if hashlib.sha256(raw).hexdigest() != pins.sha256:
                raise CaseFileError(f"fetched sha256 differs from the pinned {pins.sha256}")
            source = "fetched"
        _write_verified(destination, raw)
    return {"path": str(destination), "byte_count": pins.byte_count, "sha256": pins.sha256, "source": source}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        pins = product_pins(Path(__file__).resolve().parents[1])
        result = provide(args.destination, pins, existing=os.environ.get(ENVIRONMENT_NAME) or None)
    except (CaseFileError, OSError) as error:
        print(f"public case file not provided: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
