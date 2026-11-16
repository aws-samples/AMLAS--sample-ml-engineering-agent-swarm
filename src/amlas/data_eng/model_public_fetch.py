"""Catalog-only model byte transport; preserve the separate dataset512MiB limit."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

from ..tools.pathref import remote_fetch_enabled
from ..tools.research.public_fetch import (
    PublicFetchError,
    _checked_addresses,
    _stream_request_public,
)
from .model_weight_manifest import MAX_FILE_BYTES, expected_model_weight_manifest

MODEL_FILE_SECONDS = 900.0
MODEL_TOTAL_SECONDS = 7200.0


def stream_model_catalog_file(filename: str, sink: Any, *, deadline: float) -> dict[str, Any]:
    """Fetch one exact source-catalog file; no generic URL/size/policy selector."""
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    ref = next((item for item in expected["files"] if item["path"] == filename), None)
    now = time.monotonic()
    if (
        type(filename) is not str
        or ref is None
        or type(ref["byte_count"]) is not int
        or not 0 < ref["byte_count"] <= MAX_FILE_BYTES
        or type(deadline) not in (float, int)
        or not math.isfinite(deadline)
        or not now < deadline <= now + MODEL_TOTAL_SECONDS + 1
        or not remote_fetch_enabled()
    ):
        raise PublicFetchError("fixed model retrieval binding or budget refused")
    limit = ref["byte_count"]
    deadline = min(float(deadline), now + MODEL_FILE_SECONDS)
    current = (
        "https://huggingface.co/"
        + quote(expected["repo_id"], safe="/")
        + "/resolve/"
        + expected["revision"]
        + "/"
        + quote(filename, safe="")
    )
    visited: set[str] = set()
    digest = hashlib.sha256()
    try:
        for hop in range(6):
            if urlsplit(current).scheme != "https" or current in visited:
                raise PublicFetchError("model redirect binding refused")
            visited.add(current)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PublicFetchError("model retrieval total deadline exceeded")
            ips = _checked_addresses(current, min(remaining, 5.0))
            status, headers, count = _stream_request_public(current, ips, deadline, limit, sink, digest, None)
            if status in (301, 302, 303, 307, 308):
                if hop == 5 or not headers.get("location"):
                    raise PublicFetchError("model redirect budget exhausted")
                current = urljoin(current, headers["location"])
                continue
            if status != 200 or count != limit or digest.hexdigest() != ref["sha256"]:
                raise PublicFetchError("model retrieved bytes differ from source catalog")
            return {"byte_count": count, "sha256": digest.hexdigest()}
    except PublicFetchError:
        raise
    except Exception:
        # Signed CDN capabilities and remote response/error text stay out of receipts.
        raise PublicFetchError("guarded anonymous model retrieval failed") from None
    raise PublicFetchError("model redirect budget exhausted")
