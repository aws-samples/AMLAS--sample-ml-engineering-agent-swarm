"""Qualified singleton linkage on complete parsed data; no parser/SDK/grants."""

from __future__ import annotations

import re
from typing import Any

from . import coding_preparation as core
from . import open_swe_canonical as strict
from .open_swe_canonical import OpenSWESource

PROFILE_ID = "nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2"
LINKAGE_STATUS = "derived_not_original_source_verified"
_POLICY = core.PreparationPolicy()
_MESSAGE_KEYS = {"role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name"}
_ROLES = {"system", "developer", "user", "assistant", "tool"}


def _source_provenance(source: Any) -> dict[str, Any] | None:
    # Check every scalar before the existing typed profile reads it. Custom
    # objects are neither source metadata nor callback-enabled coercions.
    if type(source) is not OpenSWESource or any(
        type(getattr(source, name)) is not str
        for name in (
            "dataset_repo",
            "revision",
            "config_name",
            "filename",
            "artifact_uri",
            "artifact_version_id",
            "artifact_sha256",
        )
    ):
        return None
    if type(source.row_index) is not int or source.artifact_version_id.lower() == "null":
        return None
    provenance = strict._provenance(source)
    if provenance is not None:
        provenance["profile_id"] = PROFILE_ID
        provenance["coverage"] = (
            "qualified singleton derived linkage; no original-link verification, "
            "pinned corpus acceptance or near-dedup claim"
        )
    return provenance


def _rejected(reason: str, provenance: dict[str, Any] | None) -> dict[str, Any]:
    return {
        **strict._rejected(reason, provenance),
        "normalization_lineage": [],
        "normalized_record_sha256": None,
    }


def _embedded_json(text: Any, reason: str) -> None:
    if (
        type(text) is not str
        or len(text) > strict.MAX_TOOL_JSON_BYTES
        or len(text.encode("utf-8")) > strict.MAX_TOOL_JSON_BYTES
    ):
        raise core._IssueError(reason)


def _validated_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the complete original shape before considering any derived edge."""
    raw_tools = row["tools"]
    if type(raw_tools) is not list or len(raw_tools) > _POLICY.max_tools:
        raise core._IssueError("unsupported_tool_schema")
    tools = []
    for raw in raw_tools:
        _embedded_json(raw, "tool_json_envelope_invalid")
        tools.append(core._argument_object(raw, _POLICY))
    # The existing closed tool-schema validator runs once, without any source
    # result IDs or fabricated associations. The dummy text never enters output.
    core._canonical({"messages": [{"role": "user", "content": ""}], "tools": tools}, _POLICY)
    names = {item["function"]["name"] for item in tools}
    raw_messages = row["messages"]
    if type(raw_messages) is not list or not raw_messages:
        raise core._IssueError("unsupported_conversation_shape")
    if len(raw_messages) > _POLICY.max_messages:
        raise core._IssueError("max_messages")
    messages, seen = [], set()
    for original in raw_messages:
        if type(original) is not dict or not set(original) <= _MESSAGE_KEYS:
            raise core._IssueError("unsupported_message_shape")
        role = original.get("role")
        if type(role) is not str or role not in _ROLES:
            raise core._IssueError("unsupported_message_role")
        if "content" not in original or (
            original["content"] is not None and type(original["content"]) is not str
        ):
            raise core._IssueError("unsupported_message_content")
        if (
            "reasoning_content" in original
            and original["reasoning_content"] is not None
            and type(original["reasoning_content"]) is not str
        ):
            raise core._IssueError("unsupported_reasoning_content")
        message = dict(original)
        # Retain known nullable Arrow noncausal columns without inventing IDs.
        if role != "tool":
            for key in ("name", "tool_call_id"):
                if key in message and message[key] is None:
                    message.pop(key)
        if "name" in message and not core._text(message["name"], 128):
            raise core._IssueError("unsupported_message_name")
        if role != "tool" and "tool_call_id" in message:
            raise core._IssueError("unsupported_result_association")
        if role == "tool" and "tool_call_id" in message:
            if message["tool_call_id"] is None:
                raise core._IssueError("null_tool_result_id")
            if not core._text(message["tool_call_id"], 512):
                raise core._IssueError("malformed_tool_result_id")
        calls = message.get("tool_calls")
        if calls is not None and (role != "assistant" or type(calls) is not list):
            raise core._IssueError("unsupported_tool_calls")
        for call in calls or []:
            if type(call) is not dict or not core._text(call.get("id"), 512):
                raise core._IssueError("missing_tool_call_id")
            if set(call) != {"id", "type", "function"} or call["type"] != "function":
                raise core._IssueError("unsupported_tool_call")
            function = call["function"]
            if (
                type(function) is not dict
                or set(function) != {"name", "arguments"}
                or not core._text(function["name"], 128)
                or function["name"] not in names
                or type(function["arguments"]) is not str
            ):
                raise core._IssueError("unsupported_tool_arguments")
            _embedded_json(function["arguments"], "tool_argument_envelope_invalid")
            core._argument_object(function["arguments"], _POLICY)
            if call["id"] in seen:
                raise core._IssueError("causality_invalid")
            seen.add(call["id"])
        messages.append(message)
    return messages


def _derive_linkage(messages: list[dict[str, Any]], source: OpenSWESource) -> list[dict[str, Any]]:
    """Plan all edges first; only a fully valid uninterrupted exchange is applied."""
    pending: dict[str, tuple[int, int, str, bool]] = {}
    edges = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role != "tool" and pending:
            raise core._IssueError("unresolved_tool_calls")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            for call_index, call in enumerate(calls):
                pending[call["id"]] = (index, call_index, call["function"]["name"], len(calls) == 1)
        elif role == "tool":
            if not pending:
                raise core._IssueError("causality_invalid")
            absent = "tool_call_id" not in message
            if absent:
                if len(pending) != 1 or not next(iter(pending.values()))[3]:
                    raise core._IssueError("ambiguous_tool_result_id")
                identity = next(iter(pending))  # Cardinality one; never ordering inference.
            else:
                identity = message["tool_call_id"]
            if identity not in pending:
                raise core._IssueError("causality_invalid")
            assistant_index, call_index, name, _ = pending[identity]
            if "name" in message and message["name"] != name:
                raise core._IssueError("tool_result_name_mismatch")
            if absent:
                edges.append(
                    {
                        "source_row_ordinal": source.row_index,
                        "result_message_index": index,
                        "assistant_message_index": assistant_index,
                        "call_index": call_index,
                        "explicit_originating_call_id": identity,
                        "original_result_id_state": "absent",
                        "normalization_policy_version": PROFILE_ID,
                        "linkage_status": LINKAGE_STATUS,
                    }
                )
            pending.pop(identity)
    if pending:
        raise core._IssueError("unresolved_tool_calls")
    for edge in edges:
        messages[edge["result_message_index"]]["tool_call_id"] = edge["explicit_originating_call_id"]
    return edges


def normalize_open_swe_row_derived(row: dict[str, Any], *, source: OpenSWESource) -> dict[str, Any]:
    """Pure qualified normalization; complete-row refusal emits no partial edges."""
    provenance = _source_provenance(source)
    if provenance is None:
        return _rejected("source_provenance_invalid", None)
    if type(row) is not dict:
        return _rejected("row_shape_invalid", provenance)
    required = {"repo", "instance_id", "trajectory_id", "license", "resolved", "messages", "tools"}
    if not required <= row.keys():
        return _rejected("required_source_field_missing", provenance)
    try:
        core._admit_json(row, _POLICY)
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
        messages = _validated_messages(row)
        edges = _derive_linkage(messages, source)
        result = strict.normalize_open_swe_row({**row, "messages": messages}, source=source)
        if result["status"] != "normalized":
            return _rejected(result["issues"][0]["reason"], provenance)
    except (core._IssueError, core._DropError) as error:
        return _rejected(str(error), provenance)
    except (UnicodeError, ValueError, OverflowError):
        return _rejected("unsupported_json_encoding", provenance)
    record = result["record"]
    digest = core._digest({key: record[key] for key in ("messages", "tools")})
    record["normalization_lineage"] = edges
    return {
        **result,
        "source_provenance": provenance,
        "normalization_lineage": edges,
        "normalized_record_sha256": digest,
    }
