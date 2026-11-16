#!/usr/bin/env python3
"""Fetch only the five immutable public metadata files for real CPU regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest
from amlas.tools.research.public_fetch import stream_public_binary

_METADATA_NAMES = frozenset(
    {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    }
)


def metadata_manifest() -> tuple[dict, ...]:
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    refs = tuple(ref for ref in expected["files"] if ref["path"] in _METADATA_NAMES)
    if (
        len(refs) != 5
        or {ref["path"] for ref in refs} != _METADATA_NAMES
        or any(ref["role"] == "weight_shard" for ref in refs)
    ):
        raise ValueError("fixed CPU metadata manifest refused")
    return refs


def provision_metadata(destination: Path) -> dict:
    """A new directory only; never download tensors or accept caller model selectors."""
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    refs = metadata_manifest()
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    files = []
    for ref in refs:
        url = f"https://huggingface.co/{expected['repo_id']}/resolve/{expected['revision']}/{ref['path']}"
        with (destination / ref["path"]).open("xb") as stream:
            stream_public_binary(url, stream, max_bytes=ref["byte_count"], timeout=90)
        raw = (destination / ref["path"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != ref["byte_count"] or digest != ref["sha256"]:
            raise ValueError("pinned CPU metadata integrity mismatch")
        files.append({"path": ref["path"], "byte_count": len(raw), "sha256": digest, "role": ref["role"]})
    return {
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "revision": expected["revision"],
        "files": files,
        "weight_shards_downloaded": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(provision_metadata(args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
