"""Pure parsed-data preparation. Results describe candidates, never native readiness."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from typing import Any

_VERSION = "coding-preparation.v1"
_PROFILE = "openai.explicit_conversations.v1"
_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})
_SHA256 = re.compile(r"[a-f0-9]{64}")
_IDENTITY = (
    "source_locator",
    "source_version_id",
    "source_sha256",
    "repository",
    "instance_id",
    "trajectory_id",
)
_SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key_id": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(
        r"\b(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b"
    ),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
}


@dataclass(frozen=True)
class PreparationPolicy:
    """Fixed semantic profile plus explicit resource envelopes, not permission fields."""

    version: str = _VERSION
    profile: str = _PROFILE
    seed: int = 42
    max_records: int = 100000
    max_total_bytes: int = 256 * 1024 * 1024
    max_record_bytes: int = 32 * 1024 * 1024
    max_messages: int = 16384
    max_tools: int = 256
    max_json_depth: int = 64
    max_json_nodes: int = 500000


class _IssueError(ValueError):
    """Closed selection-level reason; never contains data or exception text."""


class _DropError(ValueError):
    """One intentional primary reason for a known supported record."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, maximum: int = 2048) -> bool:
    return type(value) is str and 0 < len(value) <= maximum and not any(ord(char) < 32 for char in value)


def _admit_json(value: Any, policy: PreparationPolicy) -> int:
    """Bound plain JSON before serialization; no custom object/iterator callbacks."""
    nodes = 0
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > policy.max_json_nodes:
            raise _IssueError("max_json_nodes")
        kind = type(item)
        if kind in (dict, list):
            if depth >= policy.max_json_depth:
                raise _IssueError("max_json_depth")
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise _IssueError("unsupported_json_shape")
                stack.extend((part, depth + 1) for pair in item.items() for part in pair)
            else:
                stack.extend((part, depth + 1) for part in item)
        elif kind is float:
            if not math.isfinite(item):
                raise _IssueError("nonfinite_json")
        elif kind not in (str, int, bool, type(None)):
            raise _IssueError("unsupported_json_shape")
    try:
        count = len(_json(value).encode("utf-8"))
    except (UnicodeError, ValueError, OverflowError):
        raise _IssueError("unsupported_json_encoding") from None
    if count > policy.max_record_bytes:
        raise _IssueError("max_record_bytes")
    return count


def _identity(record: dict[str, Any]) -> dict[str, Any]:
    if any(not _text(record.get(name)) for name in _IDENTITY):
        raise _DropError("identity_invalid")
    if record["source_version_id"] == "null" or not _SHA256.fullmatch(record["source_sha256"]):
        raise _DropError("identity_invalid")
    patch = record.get("patch_sha256")
    if patch is not None and (type(patch) is not str or not _SHA256.fullmatch(patch)):
        raise _DropError("identity_invalid")
    identity = {name: record[name] for name in _IDENTITY}
    identity["record_id"] = _digest(identity)
    identity["license"] = record.get("license") if _text(record.get("license"), 128) else None
    return identity


def _secret_hits(value: Any) -> Counter[str]:
    hits: Counter[str] = Counter()
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is str:
            for name, pattern in _SECRET_PATTERNS.items():
                hits[name] += len(pattern.findall(item))
        elif type(item) is list:
            stack.extend(item)
        elif type(item) is dict:
            stack.extend(part for pair in item.items() for part in pair)
    return +hits


def _argument_object(text: str, policy: PreparationPolicy) -> dict[str, Any]:
    # Bound serialized containers before the decoder can allocate their graph.
    stack = []
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            if len(stack) >= policy.max_json_depth:
                raise _IssueError("max_json_depth")
            stack.append("]" if char == "[" else "}")
        elif char in "]}":
            if not stack or stack.pop() != char:
                raise _IssueError("unsupported_tool_arguments")
    if stack or quoted:
        raise _IssueError("unsupported_tool_arguments")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise _IssueError("duplicate_tool_argument_key")
            result[key] = value
        return result

    def constant(_value: str) -> Any:
        raise _IssueError("nonfinite_json")

    try:
        decoded = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except _IssueError:
        raise
    except (ValueError, TypeError, RecursionError):
        raise _IssueError("unsupported_tool_arguments") from None
    if type(decoded) is not dict:
        raise _IssueError("unsupported_tool_arguments")
    _admit_json(decoded, policy)
    return decoded


def _canonical(record: dict[str, Any], policy: PreparationPolicy) -> dict[str, Any]:
    messages, tools = record.get("messages"), record.get("tools")
    if type(messages) is not list or not messages or type(tools) is not list:
        raise _IssueError("unsupported_conversation_shape")
    if len(messages) > policy.max_messages:
        raise _IssueError("max_messages")
    if len(tools) > policy.max_tools:
        raise _IssueError("max_tools")
    tool_names = set()
    for tool in tools:
        if type(tool) is not dict or set(tool) != {"type", "function"} or tool["type"] != "function":
            raise _IssueError("unsupported_tool_schema")
        function = tool["function"]
        if (
            type(function) is not dict
            or not _text(function.get("name"), 128)
            or type(function.get("parameters")) is not dict
        ):
            raise _IssueError("unsupported_tool_schema")
        if not set(function) <= {"name", "description", "parameters", "strict"}:
            raise _IssueError("unsupported_tool_schema")
        if function["name"] in tool_names:
            raise _DropError("content_schema_invalid")
        tool_names.add(function["name"])
        stack = [function["parameters"]]
        while stack:
            item = stack.pop()
            if type(item) is dict:
                if "$ref" in item and (type(item["$ref"]) is not str or not item["$ref"].startswith("#")):
                    raise _IssueError("external_tool_schema_ref")
                stack.extend(item.values())
            elif type(item) is list:
                stack.extend(item)
    for message in messages:
        if type(message) is not dict or not set(message) <= {
            "role",
            "content",
            "reasoning_content",
            "tool_calls",
            "tool_call_id",
            "name",
        }:
            raise _IssueError("unsupported_message_shape")
        if type(message.get("role")) is not str or message["role"] not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise _IssueError("unsupported_message_role")
        if "content" not in message or (
            message["content"] is not None and type(message["content"]) is not str
        ):
            raise _IssueError("unsupported_message_content")
        if (
            "reasoning_content" in message
            and message["reasoning_content"] is not None
            and type(message["reasoning_content"]) is not str
        ):
            raise _IssueError("unsupported_reasoning_content")
        if "name" in message and not _text(message["name"], 128):
            raise _IssueError("unsupported_message_name")
        calls = message.get("tool_calls")
        if calls is not None and (message["role"] != "assistant" or type(calls) is not list):
            raise _IssueError("unsupported_tool_calls")
        for call in calls or []:
            if type(call) is not dict or not _text(call.get("id"), 512):
                raise _IssueError("missing_tool_call_id")
            if set(call) != {"id", "type", "function"} or call["type"] != "function":
                raise _IssueError("unsupported_tool_call")
            function = call["function"]
            if (
                type(function) is not dict
                or set(function) != {"name", "arguments"}
                or not _text(function["name"], 128)
                or function["name"] not in tool_names
                or type(function["arguments"]) is not str
            ):
                raise _IssueError("unsupported_tool_arguments")
            _argument_object(function["arguments"], policy)
        if message["role"] == "tool" and not _text(message.get("tool_call_id"), 512):
            raise _IssueError("missing_tool_result_id")
        if message["role"] != "tool" and "tool_call_id" in message:
            raise _IssueError("unsupported_result_association")
    # Copy only canonical data; ignored reference/model/gold patches never enter it.
    return json.loads(_json({"messages": messages, "tools": tools}))


def _causality(canonical: dict[str, Any]) -> None:
    seen, pending = set(), set()
    supervised = False
    for message in canonical["messages"]:
        role = message["role"]
        if role != "tool" and pending:
            raise _IssueError("unresolved_tool_calls")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            supervised |= bool(message["content"] or calls)
            for call in calls:
                if call["id"] in seen:
                    raise _DropError("causality_invalid")
                seen.add(call["id"])
                pending.add(call["id"])
        elif role == "tool":
            identity = message["tool_call_id"]
            if identity not in pending:
                raise _DropError("causality_invalid")
            pending.remove(identity)
    if pending:
        raise _IssueError("unresolved_tool_calls")
    if not supervised:
        raise _DropError("content_schema_invalid")


def _fingerprint(canonical: dict[str, Any]) -> str:
    view = json.loads(_json(canonical))
    ids = {}
    for message in view["messages"]:
        for call in message.get("tool_calls") or []:
            native = call["id"]
            ids[native] = f"call-{len(ids)}"
            call["id"] = ids[native]
        if message["role"] == "tool":
            message["tool_call_id"] = ids[message["tool_call_id"]]
    return _digest(view)


def _result(
    source_count: int,
    candidates: list[dict[str, Any]],
    drops: Counter[str],
    issues: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    secrets: Counter[str],
    policy: PreparationPolicy,
) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        clusters[candidate["fingerprint"]].append(candidate)
    representatives = []
    for cluster_id, members in sorted(clusters.items()):
        representative = min(
            members,
            key=lambda value: (
                value["canonical_digest"],
                value["identity"]["source_locator"],
                value["identity"]["record_id"],
            ),
        )
        representatives.append(representative)
        for member in members:
            member["representative_id"] = representative["identity"]["record_id"]
            if member is not representative:
                drops["duplicate_exact"] += 1
            lineage.append(
                {
                    **member["identity"],
                    "cluster_id": cluster_id,
                    "representative_id": representative["identity"]["record_id"],
                    "decision": "retained" if member is representative else "duplicate_exact",
                }
            )
    parents = {value["identity"]["record_id"]: value["identity"]["record_id"] for value in candidates}

    def find(identity: str) -> str:
        while parents[identity] != identity:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity

    def join(first: str, second: str) -> None:
        first, second = find(first), find(second)
        parents[max(first, second)] = min(first, second)

    owners = {}
    for candidate in sorted(candidates, key=lambda value: value["identity"]["record_id"]):
        identity = candidate["identity"]["record_id"]
        for key in (
            ("repository", candidate["identity"]["repository"]),
            ("instance", candidate["identity"]["instance_id"]),
            ("duplicate", candidate["fingerprint"]),
        ):
            if key in owners:
                join(identity, owners[key])
            owners[key] = identity
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        components[find(candidate["identity"]["record_id"])].append(candidate)
    group_info = {}
    for members in components.values():
        group_id = _digest(
            {
                "repositories": sorted({value["identity"]["repository"] for value in members}),
                "instances": sorted({value["identity"]["instance_id"] for value in members}),
            }
        )
        number = int.from_bytes(hashlib.sha256(_json([policy.seed, group_id]).encode()).digest()[:8], "big")
        split = (
            "train"
            if number < (2**64 * 80) // 100
            else "validation"
            if number < (2**64 * 90) // 100
            else "test"
        )
        for member in members:
            group_info[member["identity"]["record_id"]] = (group_id, split)
    ordered = sorted(representatives, key=lambda value: value["identity"]["record_id"])
    membership = [
        {
            "record_id": value["identity"]["record_id"],
            "group_id": group_info[value["identity"]["record_id"]][0],
            "split": group_info[value["identity"]["record_id"]][1],
        }
        for value in ordered
    ]
    split_counts = {
        name: sum(value["split"] == name for value in membership) for name in ("train", "validation", "test")
    }
    if not all(split_counts.values()):
        issues.append({"reason": "insufficient_split_groups", "source_locator": None, "count": 0})
    unresolved_count = sum(value["count"] for value in issues)
    complete = not issues
    records = [value["canonical"] for value in ordered] if complete else []
    return {
        "status": "complete" if complete else "incomplete",
        "canonical_records": records,
        "canonical_jsonl": "".join(_json(value) + "\n" for value in records),
        "membership": membership if complete else [],
        "lineage": sorted(
            lineage,
            key=lambda value: (
                value.get("source_locator") or "",
                value.get("record_id") or "",
                value.get("decision") or "",
            ),
        ),
        "primary_drop_counts": dict(sorted(drops.items())),
        "unresolved_profiles": sorted(
            issues, key=lambda value: (value["reason"], value.get("source_locator") or "")
        ),
        "report": {
            "source_count": source_count,
            "candidate_representatives": len(ordered),
            "retained_representatives": len(records),
            "unresolved_record_count": unresolved_count,
            "split_counts": split_counts,
            "conservation": source_count == len(ordered) + sum(drops.values()) + unresolved_count,
            "near_dedup": "not_implemented",
            "exclusion_scope": "supplied_instance_ids_and_patch_digests_only",
            "secret_pattern_counts": dict(sorted(secrets.items())),
        },
    }


def prepare_canonical_conversations(
    records: list[dict[str, Any]], *, policy: PreparationPolicy, exclusions: dict[str, Any]
) -> dict[str, Any]:
    """Prepare bounded already-parsed data; unknown conventions emit no usable corpus."""
    if type(policy) is not PreparationPolicy:
        raise ValueError("fixed preparation policy required")
    for field in fields(policy):
        if field.name not in ("version", "profile"):
            value = getattr(policy, field.name)
            if type(value) is not int or not (0 if field.name == "seed" else 1) <= value <= 2**63 - 1:
                raise ValueError("invalid preparation resource policy")
    if (
        type(records) is not list
        or type(exclusions) is not dict
        or set(exclusions) != {"instance_ids", "patch_sha256"}
    ):
        raise ValueError("plain parsed records and fixed exclusions required")
    for name, values in exclusions.items():
        if type(values) is not list or any(
            not _text(value) or (name == "patch_sha256" and not _SHA256.fullmatch(value)) for value in values
        ):
            raise ValueError("invalid preparation exclusions")
    drops: Counter[str] = Counter()
    secrets: Counter[str] = Counter()
    candidates, lineage, issues = [], [], []
    total = len(records)
    if policy.version != _VERSION or policy.profile != _PROFILE or total > policy.max_records:
        reason = (
            "unsupported_profile"
            if policy.version != _VERSION or policy.profile != _PROFILE
            else "max_records"
        )
        return _result(
            total,
            [],
            drops,
            [{"reason": reason, "source_locator": None, "count": total}],
            [],
            secrets,
            policy,
        )
    locations = Counter(
        record.get("source_locator")
        for record in records
        if type(record) is dict and type(record.get("source_locator")) is str
    )
    total_bytes = 0
    excluded_ids, excluded_patches = set(exclusions["instance_ids"]), set(exclusions["patch_sha256"])
    for record in records:
        source = (
            record.get("source_locator")
            if type(record) is dict
            and _text(record.get("source_locator"))
            and not _secret_hits(record["source_locator"])
            else None
        )
        identity = None
        try:
            if type(record) is not dict:
                raise _IssueError("unsupported_record_shape")
            if source is not None and locations[source] > 1:
                raise _IssueError("duplicate_source_locator")
            total_bytes += _admit_json(record, policy)
            if total_bytes > policy.max_total_bytes:
                return _result(
                    total,
                    [],
                    Counter(),
                    [{"reason": "max_total_bytes", "source_locator": None, "count": total}],
                    [],
                    Counter(),
                    policy,
                )
            metadata_hits = _secret_hits({name: record.get(name) for name in (*_IDENTITY, "license")})
            if metadata_hits:
                secrets.update(metadata_hits)
                raise _DropError("secret_pattern")
            identity = _identity(record)
            resolved = record.get("resolved")
            if resolved is None or (type(resolved) is int and resolved == -1):
                raise _DropError("resolution_unknown")
            if type(resolved) is not int or resolved not in (0, 1):
                raise _DropError("resolution_invalid")
            if resolved == 0:
                raise _DropError("resolution_failed")
            if type(record.get("license")) is not str or record["license"] not in _LICENSES:
                raise _DropError("license_excluded")
            canonical = _canonical(record, policy)
            hits = _secret_hits(canonical)
            if hits:
                secrets.update(hits)
                raise _DropError("secret_pattern")
            if record["instance_id"] in excluded_ids or record.get("patch_sha256") in excluded_patches:
                raise _DropError("evaluation_overlap")
            _causality(canonical)
            candidates.append(
                {
                    "identity": identity,
                    "canonical": canonical,
                    "canonical_digest": _digest(canonical),
                    "fingerprint": _fingerprint(canonical),
                }
            )
        except _DropError as error:
            reason = str(error)
            drops[reason] += 1
            if identity is not None:
                lineage.append(
                    {**identity, "decision": reason, "cluster_id": None, "representative_id": None}
                )
        except _IssueError as error:
            issues.append({"reason": str(error), "source_locator": source, "count": 1})
    return _result(total, candidates, drops, issues, lineage, secrets, policy)
