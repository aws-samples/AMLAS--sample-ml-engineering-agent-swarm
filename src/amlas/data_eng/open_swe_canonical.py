"""Conservative parsed Open-SWE profile. No parser, SDK, model or readiness grants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

from . import coding_preparation as core

PROFILE_ID = "nvidia.open_swe_traces.v1.2.explicit_ids.v1"
DATASET_REPO = "nvidia/Open-SWE-Traces"
DATASET_REVISION = (
    "1db674173119d71df45c22007af99e214600711a"  # pragma: allowlist secret - public dataset revision
)
CARD_SHA256 = "8fd748230ebacb5a07b01c0299b3d99c1249567f47677c1c935163f843652341"  # pragma: allowlist secret - public card digest
DATA_LICENSE = "CC-BY-4.0"
MAX_TOOL_JSON_BYTES = 1024 * 1024
_POLICY = core.PreparationPolicy()
_FILE = re.compile(
    r"data/(openhands|sweagent|minisweagent)/(qwen38_27b|qwen38_flash_next)/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\.parquet"
)


@dataclass(frozen=True)
class OpenSWESource:
    """Version-bound data provenance. The type grants no resource/caller authority."""

    dataset_repo: str
    revision: str
    config_name: str
    filename: str
    artifact_uri: str
    artifact_version_id: str
    artifact_sha256: str
    row_index: int


def _provenance(source: Any) -> dict[str, Any] | None:
    if type(source) is not OpenSWESource:
        return None
    if (
        source.dataset_repo != DATASET_REPO
        or source.revision != DATASET_REVISION
        or source.config_name != "v1.2"
        or type(source.filename) is not str
        or not _FILE.fullmatch(source.filename)
        or ".." in source.filename
        or not core._text(source.artifact_uri, 4096)
        or not core._text(source.artifact_version_id, 1024)
        or source.artifact_version_id == "null"
        or type(source.artifact_sha256) is not str
        or not re.fullmatch(r"[a-f0-9]{64}", source.artifact_sha256)
        or type(source.row_index) is not int
        or not 0 <= source.row_index < 2**63
    ):
        return None
    try:
        uri = urlsplit(source.artifact_uri)
    except ValueError:
        return None
    if uri.scheme == "s3":
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", uri.netloc)
            or not uri.path.startswith("/")
            or uri.path == "/"
            or uri.query
            or uri.fragment
            or "\\" in source.artifact_uri
            or any(x in {"", ".", ".."} for x in uri.path[1:].split("/"))
        ):
            return None
    elif source.artifact_uri != (
        "https://huggingface.co/datasets/"
        + DATASET_REPO
        + "/resolve/"
        + DATASET_REVISION
        + "/"
        + quote(source.filename, safe="/")
    ):
        return None
    return {
        "profile_id": PROFILE_ID,
        "dataset_repo": DATASET_REPO,
        "revision": DATASET_REVISION,
        "config_name": "v1.2",
        "filename": source.filename,
        "artifact_uri": source.artifact_uri,
        "artifact_version_id": source.artifact_version_id,
        "artifact_sha256": source.artifact_sha256,
        "row_index": source.row_index,
        "data_license": DATA_LICENSE,
        "card_sha256": CARD_SHA256,
        "coverage": "explicit_ids_only; no pinned Parquet schema acceptance or near-dedup claim",
    }


def _rejected(reason: str, provenance: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "status": "rejected",
        "record": None,
        "issues": [{"reason": reason, "count": 1}],
        "source_provenance": provenance,
        "data_only": True,
    }


def normalize_open_swe_row(row: dict[str, Any], *, source: OpenSWESource) -> dict[str, Any]:
    """Normalize already-parsed plain JSON. Missing causal evidence always refuses."""
    provenance = _provenance(source)
    if provenance is None:
        return _rejected("source_provenance_invalid", None)
    if type(row) is not dict:
        return _rejected("row_shape_invalid", provenance)
    required = {"repo", "instance_id", "trajectory_id", "license", "resolved", "messages", "tools"}
    if not required <= row.keys():
        return _rejected("required_source_field_missing", provenance)
    if type(row["resolved"]) is not int or row["resolved"] not in {-1, 0, 1}:
        return _rejected("resolution_invalid", provenance)
    if type(row["license"]) is not str or row["license"] not in core._LICENSES:
        return _rejected("repository_license_unsupported", provenance)
    if (
        not core._text(row["repo"], 200)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["repo"])
        or ".." in row["repo"]
        or not core._text(row["instance_id"], 2048)
        or not core._text(row["trajectory_id"], 2048)
    ):
        return _rejected("record_identity_invalid", provenance)
    try:
        core._admit_json(row, _POLICY)
        if type(row["tools"]) is not list or len(row["tools"]) > _POLICY.max_tools:
            return _rejected("unsupported_tool_schema", provenance)
        tools = []
        for raw in row["tools"]:
            if (
                type(raw) is not str
                or len(raw) > MAX_TOOL_JSON_BYTES
                or len(raw.encode("utf-8")) > MAX_TOOL_JSON_BYTES
            ):
                return _rejected("tool_json_envelope_invalid", provenance)
            tools.append(core._argument_object(raw, _POLICY))
        if type(row["messages"]) is not list:
            return _rejected("unsupported_conversation_shape", provenance)
        messages = []
        for original in row["messages"]:
            if type(original) is not dict:
                return _rejected("unsupported_message_shape", provenance)
            # Arrow nullable optional columns are absent evidence, never fabricated IDs.
            messages.append(
                {
                    key: value
                    for key, value in original.items()
                    if not (key in {"name", "tool_call_id"} and value is None)
                }
            )
        canonical = core._canonical({"messages": messages, "tools": tools}, _POLICY)
        core._causality(canonical)
    except (core._IssueError, core._DropError) as error:
        # Existing closed reducer labels, never source contents or exception reprs.
        return _rejected(str(error), provenance)
    except (UnicodeError, ValueError, OverflowError):
        return _rejected("unsupported_json_encoding", provenance)
    record = {
        **canonical,
        "repository": row["repo"],
        "instance_id": row["instance_id"],
        "trajectory_id": row["trajectory_id"],
        "license": row["license"],
        "resolved": row["resolved"],
        "source_locator": source.artifact_uri
        + "#version="
        + source.artifact_version_id
        + "#row="
        + str(source.row_index),
        "source_version_id": source.artifact_version_id,
        "source_sha256": source.artifact_sha256,
    }
    return {
        "status": "normalized",
        "record": record,
        "issues": [],
        "source_provenance": provenance,
        "data_only": True,
    }
