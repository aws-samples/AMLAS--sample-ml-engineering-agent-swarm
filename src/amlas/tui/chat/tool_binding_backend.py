"""DP-1 — ToolBindingBackend: the TUI's embedded, tool-bound chat agent.

Upgrades the TUI's natural-language brain from suggestion-only chat to **real
function-calling** by wrapping a Strands ``Agent`` built via the proven
``_build.build_agent`` factory with ``SAFE_LLM_TOOLS`` — the identical
pattern that powers ``pipeline.agentic_runner.build_pipeline_agent``
(see ``docs/audit/phase7-road-to-100-design.md`` §2).

Graceful degradation (design §2.3): compatibility validation may replace
``strands`` with an inert identity shim. There, ``is_tool_capable`` is ``False``, so
the engine skips this tier. It used to fall back to the raw-converse
``BedrockChatBackend``; issue #35 deleted that second brain, so the fallback is now
the engine's DETERMINISTIC tier-5 keyword responder — still no LLM under the shim.

Confirmation gate (safety default #3): read-only tools auto-execute; state-changing
tools surface a ``needs_confirmation`` prompt before running.

Phase 11 (DP-2) handoff: ``agent_tools.LONG_RUNNING_TOOLS`` enumerates the bound
tools that should become background jobs; for Phase 10 they execute synchronously.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from contextvars import Context, copy_context
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from ...backends.types import JobHandle
from ...config.externalized import model_weight_acquisition_selectors
from ...config.settings import Settings
from ...control_plane.acquisition_state import SessionAcquisitionStore
from ...control_plane.approvals import ApprovalPolicy
from ...control_plane.case_binding import build_case_freeze_store
from ...control_plane.generation_binding import build_generation_store
from ...control_plane.model_acquisition_state import (
    SessionModelAcquisitionStore,
    build_model_acquisition_store,
)
from ...control_plane.paired_binding import build_pair_store
from ...control_plane.preparation_binding import build_preparation_store
from ...control_plane.sft_binding import build_sft_store
from .agent_tools import (
    LONG_RUNNING_TOOLS,
    READ_ONLY_TOOLS,
    STATE_CHANGING_TOOLS,
    TUI_SYSTEM_PROMPT,
    get_safe_llm_tools,
    tool_display_name,
)
from .guardrail_assessment import GuardrailAssessment, GuardrailAssessmentCapture
from .thought_trace import ThoughtEvent, ThoughtTrace, _make_trace_callback

logger = logging.getLogger(__name__)

_INPUT_HOLD_WARNINGS = frozenset(
    (
        "AMLAS_INPUT_HOLD phase=before_model_qualification reason=callback_refused",
        "AMLAS_INPUT_HOLD phase=history_qualification reason=callback_refused",
        "AMLAS_INPUT_HOLD phase=history_qualification reason=history_batch_refused",
        "AMLAS_INPUT_HOLD phase=history_qualification reason=history_prefix_refused",
        "AMLAS_INPUT_HOLD phase=pending_bookkeeping reason=callback_refused",
        "AMLAS_INPUT_HOLD phase=producer_capture reason=callback_refused",
        "AMLAS_INPUT_HOLD phase=sft_capture reason=callback_refused",
        "AMLAS_INPUT_HOLD phase=research_after_result reason=callback_refused",
        "AMLAS_INPUT_HOLD phase=tool_projection reason=callback_refused",
        "AMLAS_INPUT_HOLD phase=research_before_tool reason=callback_refused",
        "AMLAS_INPUT_HOLD phase=research_activation reason=binding_not_unused",
        "AMLAS_INPUT_HOLD phase=input_admission reason=admission_refused",
        "AMLAS_INPUT_HOLD phase=pending_bookkeeping reason=duplicate_original",
        "AMLAS_INPUT_HOLD phase=pending_bookkeeping reason=completion_admission_refused",
        "AMLAS_INPUT_HOLD phase=history_qualification reason=checker_selection_failed",
        "AMLAS_INPUT_HOLD phase=history_qualification reason=history_not_serializable",
        "AMLAS_INPUT_HOLD phase=history_qualification reason=completed_original_changed",
        "AMLAS_INPUT_HOLD phase=research_after_run reason=binding_refused",
        "AMLAS_INPUT_HOLD phase=pause_result reason=pause_result_refused",
        "AMLAS_INPUT_HOLD phase=pause_result reason=unmarked_stop",
        "AMLAS_INPUT_HOLD phase=pause_result reason=stop_state_mismatch",
        "AMLAS_INPUT_HOLD phase=pause_start reason=invocation_state_refused",
        "AMLAS_INPUT_HOLD phase=pause_start reason=incoming_messages_refused",
        "AMLAS_INPUT_HOLD phase=completion_fence reason=fence_refused",
        "AMLAS_INPUT_HOLD phase=completion_run reason=completion_refused",
    )
)


def _warn_normal_sft_input_predicate(run_id, phase, predicate):
    """Report only a fixed predicate for this run's causal SFT input check."""
    predicates = {
        "INPUT_TYPE",
        "REQUIRED_KEYS",
        "EXTRA_KEYS",
        "MODEL_REF",
        "PREPARATION_REF",
        "RECIPE",
        "PLAN_TYPE",
        "PLAN_PREVIEW",
    }
    if (
        type(run_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
        or type(phase) is not str
        or phase != "causal_sft_inputs"
        or type(predicate) is not str
        or predicate not in predicates
    ):
        return
    try:
        logger.warning("normal_sft_input_predicate run_id=%s check=%s", run_id, predicate)
    except Exception:
        pass


def _warn_normal_sft_capture_size(run_id, byte_count, outcome):
    """Log only run label, canonical byte count, the fixed cap and a fixed outcome; never content."""
    from ...control_plane.normal_run_checkpoint import _MAX_BYTES

    if (
        type(run_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
        or type(byte_count) is not int
        or not 0 <= byte_count <= 64 * 1024 * 1024
        or type(outcome) is not str
        or outcome not in {"ok", "over"}
    ):
        return
    with contextlib.suppress(Exception):
        logger.warning(
            "normal_sft_capture_size run_id=%s bytes=%d limit=%d outcome=%s",
            run_id,
            byte_count,
            _MAX_BYTES,
            outcome,
        )


def _active_run_label():
    """Return the active normal run's validated 32-hex label, else "none"; never raises."""
    label = "none"
    with contextlib.suppress(Exception):
        from ...control_plane.normal_run_checkpoint import current_normal_run

        context = current_normal_run()
        candidate = context.run_id if context is not None else None
        if type(candidate) is str and re.fullmatch(r"[0-9a-f]{32}", candidate) is not None:
            label = candidate
    return label


def _note_normal_sft_result_delivered():
    """Record on the active normal run that its original model received the SFT call result; never raises."""
    with contextlib.suppress(Exception):
        from ...control_plane.normal_run_checkpoint import current_normal_run

        context = current_normal_run()
        if context is not None:
            context.note_original_result_delivered()


def _record_messages(messages):
    """Return message DATA as records keep it: without the SDK's per-message ``tracking_id``.

    strands-agents 1.47.0 and later write a random ``tracking_id`` on every message of the live history.
    The SDK does not send it to the model and no record schema names it, so it never enters a causal
    prefix, a TASK snapshot, the history witness or a size measurement. Nothing else is removed.
    """
    if type(messages) is not list:
        return messages
    return [
        {key: value for key, value in message.items() if key != "tracking_id"}
        if type(message) is dict and "tracking_id" in message
        else message
        for message in messages
    ]


def _tool_result_admission_text(content):
    """Return the one text in which a tool result's content list reaches the input check.

    A fresh result at capture, a stored result inside a history message and a stored research result at a
    restore all go through this function, so content in the same key order is the same text at every
    check. The key order is kept as given; a checkpoint stores every object with sorted keys.
    """
    return json.dumps(content, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


# One text of the input check holds at most this many UTF-8 bytes. The number is ``MAX_INPUT_BYTES`` of
# ``input_admission.py``, a module that is loaded only when a guardrail is selected.
_HISTORY_MESSAGE_MAX_BYTES = 65536

# The fields of a stored message that the history form reads: the role, the content list and the agent SDK's
# metadata. Field names outside the content are sent in no single text, so a message with any other field is
# sent whole and the input check sees that name.
_HISTORY_MESSAGE_FIELDS = frozenset({"role", "content", "metadata"})


def _history_message_fits_one_check(serialized):
    """True when the compact JSON of a whole stored message is a text that the input check assesses.

    The input check refuses a text of more than 65,536 UTF-8 bytes, and a text that strict UTF-8 cannot
    encode, before it asks the guardrail. A message with such a compact JSON is sent as that one text, so
    that refusal stays, also when every part of the message would pass on its own.
    """
    try:
        return len(serialized.encode("utf-8", errors="strict")) <= _HISTORY_MESSAGE_MAX_BYTES
    except UnicodeEncodeError:
        return False


def _history_value_texts(value, texts, skipped, keys):
    """Append every scalar leaf of plain JSON DATA as its own text; False for any other value.

    A string is sent as it is and a number or boolean as its JSON text. An empty string and null are
    counted in ``skipped`` and not sent. A string of only white space is no such value: the service gives
    no verdict on it alone, so the caller sends its block or message whole. With ``keys`` every object key
    is sent like a string value.
    """
    kind = type(value)
    if kind is str:
        if not value:
            skipped[0] += 1
        elif value.strip():
            texts.append(value)
        else:
            return False
    elif kind is bool:
        texts.append("true" if value else "false")
    elif kind is int or kind is float:
        texts.append(json.dumps(value, allow_nan=False))
    elif value is None:
        skipped[1] += 1
    elif kind is list:
        for item in value:
            if not _history_value_texts(item, texts, skipped, keys):
                return False
    elif kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                return False
            if keys and not _history_value_texts(key, texts, skipped, False):
                return False
            if not _history_value_texts(item, texts, skipped, keys):
                return False
    else:
        return False
    return True


def _history_block_texts(block, texts, skipped):
    """Append the texts of one content block of a known closed shape; False for any other block.

    A key is compared only when it is exactly a string, and a tool result decides to go whole before
    its content list is serialized.
    """
    if type(block) is not dict or len(block) != 1:
        return False
    ((kind, body),) = block.items()
    # A key is compared only when it is exactly a string; any other key sends the block whole.
    if type(kind) is not str:
        return False
    if kind == "text":
        return type(body) is str and _history_value_texts(body, texts, skipped, False)
    if type(body) is not dict or not all(type(key) is str for key in body):
        return False
    if kind == "toolUse":
        # The call id, the tool name and every key and scalar leaf of the input, each on its own.
        return (
            set(body) == {"toolUseId", "name", "input"}
            and type(body["toolUseId"]) is str
            and type(body["name"]) is str
            and all(_history_value_texts(item, texts, skipped, key == "input") for key, item in body.items())
        )
    if kind == "toolResult":
        if (
            set(body) != {"toolUseId", "status", "content"}
            or type(body["toolUseId"]) is not str
            or type(body["status"]) is not str
            or type(body["content"]) is not list
        ):
            return False
        # A blank id or status sends the block whole; that is decided before the content is serialized.
        if any(item and not item.strip() for item in (body["toolUseId"], body["status"])):
            return False
        for key, item in body.items():
            if key == "content":
                texts.append(_tool_result_admission_text(item))
            elif not _history_value_texts(item, texts, skipped, False):
                return False
        return True
    if kind == "reasoningContent":
        # The reasoning text, each signature and a redacted value, each on its own. The names inside a
        # redacted value that is an object or a list would stand in no single text, so that block goes whole.
        reasoning = body.get("reasoningText", {})
        redacted = type(body.get("redactedContent"))
        return (
            bool(body)
            and set(body) <= {"reasoningText", "redactedContent"}
            and redacted is not dict
            and redacted is not list
            and type(reasoning) is dict
            and all(type(key) is str for key in reasoning)
            and set(reasoning) <= {"text", "signature"}
            and all(type(item) is str for item in reasoning.values())
            and _history_value_texts(body, texts, skipped, False)
        )
    return False


def _history_admission_plan(message):
    """Derive the texts in which one stored message reaches the input check: ``(texts, empty, null)``.

    Content goes in the form of its first check: a text block as its text and a tool result as the text
    of ``_tool_result_admission_text`` for its content list. Machine-written values go singly, in stored
    order: the result's call id and status; a tool call's id, name and every key and scalar leaf of its
    input; the reasoning text, each signature and a redacted value; every scalar leaf outside the content
    list. A block of another shape, or with a single value of only white space, is sent whole as compact
    JSON, and so is a message that cannot be walked, that holds a field outside ``_HISTORY_MESSAGE_FIELDS``,
    that holds such a value outside its content list, that yields no text, or whose compact JSON the input
    check refuses for its size or its encoding (``_history_message_fits_one_check``). A reasoning block
    whose redacted value is an object or a list is a block of another shape: the names inside such a value
    would stand in no single text. ``empty`` and ``null`` count the empty strings and nulls that are not
    sent. The message is never changed, and a value that JSON cannot carry raises as the whole message did.
    """

    def whole(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    texts, skipped = [], [0, 0]
    walked = (
        type(message) is dict
        and all(type(key) is str for key in message)
        and _HISTORY_MESSAGE_FIELDS.issuperset(message)
        and type(message.get("role")) is str
        and type(message.get("content")) is list
    )
    try:
        for key, value in message.items() if walked else ():
            if key == "content":
                for block in value:
                    found, counted = [], [0, 0]
                    try:
                        known = _history_block_texts(block, found, counted)
                    except (TypeError, ValueError):
                        known = False
                    if not known:
                        found, counted = [whole(block)], [0, 0]
                    texts += found
                    skipped = [skipped[0] + counted[0], skipped[1] + counted[1]]
            elif not _history_value_texts(value, texts, skipped, False):
                walked = False
                break
    except RecursionError:
        walked = False
    serialized = whole(message)
    if not walked or not texts or not _history_message_fits_one_check(serialized):
        return [serialized], 0, 0
    return texts, skipped[0], skipped[1]


def _history_admission_texts(message):
    """Return the ordered texts of ``_history_admission_plan`` for one stored message."""
    return _history_admission_plan(message)[0]


def _record_session_snapshot(agent):
    """Take the public SDK session snapshot in the four-field record inventory.

    strands-agents 1.43.0 and later add ``model_state`` to the session preset. Only an empty one is
    dropped; any provider state is refused, so the stored inventory stays the one written under 1.42.0.
    """
    snapshot = agent.take_snapshot(preset="session").to_dict()
    data = snapshot.get("data") if type(snapshot) is dict else None
    if type(data) is dict:
        if "model_state" in data:
            if type(data["model_state"]) is not dict or data["model_state"]:
                raise ValueError("public SDK session snapshot DATA inventory differs")
            del data["model_state"]
        if "messages" in data:
            data["messages"] = _record_messages(data["messages"])
    return snapshot


def _conversation_bytes(messages):
    """Canonical size of a message list as `_canonical` measures it.

    The dump is ASCII (ensure_ascii default), so the character count equals the UTF-8 byte count.
    """
    return len(json.dumps(messages, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _warn_normal_conversation_size(run_id, model_call, byte_count, message_count):
    """Log only run label, model-call ordinal, canonical bytes, message count and the fixed cap; never content."""
    from ...control_plane.normal_run_checkpoint import _MAX_BYTES

    if (
        type(run_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
        or type(model_call) is not int
        or not 1 <= model_call <= 1_000_000
        or type(byte_count) is not int
        or not 0 <= byte_count <= 64 * 1024 * 1024
        or type(message_count) is not int
        or not 0 <= message_count <= 1_000_000
    ):
        return
    with contextlib.suppress(Exception):
        logger.warning(
            "normal_conversation_size run_id=%s model_call=%d bytes=%d messages=%d limit=%d",
            run_id,
            model_call,
            byte_count,
            message_count,
            _MAX_BYTES,
        )


# Unresolved-observation cap for an original operation awaited before its tool result commits.
# MODEL and CASE already close after 8 consecutive unresolved polls (120 s at the 15 s cadence).
# DATA, PREP, SFT, RELOAD, GEN and PAIR had no cap, so one publisher exception (reported as UNKNOWN
# or observation_failed on every poll) held a run for the whole 4 h driver budget. The cap is wall
# clock over one run of consecutive unresolved answers: 900 s is 7.5 times the MODEL/CASE budget,
# 5 times the longest single observation round seen in a healthy run (164 s), and 1/16 of 4 h.
# PENDING/RUNNING (or any other answer) ends the run, so healthy long jobs are never capped.
_PRECOMMIT_UNKNOWN_WALL_CLOCK_SECONDS = 900.0
_PRECOMMIT_UNKNOWN_LOG_EVERY = 8
_PRECOMMIT_UNRESOLVED_STATUSES = frozenset(
    {"UNKNOWN", "observation_failed", "verification_failed", "output_verification_failed"}
)
_UNKNOWN_POLL_OPERATIONS = {
    "data": "DATA",
    "case": "CASE",
    "preparation": "PREP",
    "sft": "SFT",
    "reload": "RELOAD",
    "generation": "GEN",
    "pair": "PAIR",
}


def _precommit_unknown_clock():
    return time.monotonic()


def _warn_normal_unknown_poll(operation, polls, elapsed_seconds, outcome):
    """Log only a fixed operation label, poll and second counts and a fixed outcome; never content."""
    if (
        type(operation) is not str
        or operation not in _UNKNOWN_POLL_OPERATIONS.values()
        or type(polls) is not int
        or not 1 <= polls <= 10_000_000
        or type(elapsed_seconds) is not int
        or not 0 <= elapsed_seconds <= 10_000_000
        or type(outcome) is not str
        or outcome not in {"waiting", "closed"}
    ):
        return
    with contextlib.suppress(Exception):
        logger.warning(
            "normal_unknown_poll run_id=%s operation=%s polls=%d elapsed_s=%d limit_s=%d outcome=%s",
            _active_run_label(),
            operation,
            polls,
            elapsed_seconds,
            int(_PRECOMMIT_UNKNOWN_WALL_CLOCK_SECONDS),
            outcome,
        )


# Messages the normal-run chat agent keeps before the SDK trims its history (SDK default: 40).
_NORMAL_RUN_CONVERSATION_WINDOW = 100

# Run-level retry of one failed model call. The SDK retries ModelThrottledException only (6 attempts,
# 4-64 s) and no 5xx or connection error, so a Bedrock 5xx after the boto retries ended the run.
# The retry repeats the failed call alone through AfterModelCallEvent.retry: the prompt is not
# appended again and no tool runs twice.
_MODEL_CALL_RETRIES = 3
_MODEL_RETRY_CAP_SECONDS = (1.0, 2.0, 4.0)
_THROTTLE_CODES = frozenset({"ThrottlingException", "throttlingException", "TooManyRequestsException"})
_SERVER_CODES = frozenset(
    {
        "InternalServerException",
        "internalServerException",
        "ServiceUnavailableException",
        "serviceUnavailableException",
        "ModelStreamErrorException",
        "modelStreamErrorException",
        "ModelTimeoutException",
        "modelTimeoutException",
    }
)
_SERVER_STATUSES = frozenset({500, 502, 503, 504})


def _transient_model_error_reason(error):
    """Closed classification of a model-call failure as retryable; None for every other error."""
    from botocore import exceptions as sdk_errors
    from strands.types.exceptions import ModelThrottledException

    if isinstance(error, ModelThrottledException):
        return "throttled"
    if isinstance(error, sdk_errors.ClientError):
        response = getattr(error, "response", None)
        native = response.get("Error") if type(response) is dict else None
        metadata = response.get("ResponseMetadata") if type(response) is dict else None
        code = native.get("Code") if type(native) is dict else None
        status = metadata.get("HTTPStatusCode") if type(metadata) is dict else None
        if type(code) is str and code in _THROTTLE_CODES:
            return "throttled"
        if (type(code) is str and code in _SERVER_CODES) or (
            type(status) is int and status in _SERVER_STATUSES
        ):
            return "server_error"
        return None
    if isinstance(
        error,
        sdk_errors.EndpointConnectionError
        | sdk_errors.ConnectionClosedError
        | sdk_errors.ReadTimeoutError
        | sdk_errors.ConnectTimeoutError,
    ):
        return "connection"
    return None


def _model_retry_delay(attempt, uniform=None):
    """Full-jitter backoff: a uniform draw from 0 to the cap of this attempt (1, 2 and 4 seconds)."""
    caps = _MODEL_RETRY_CAP_SECONDS
    cap = caps[min(max(attempt, 1), len(caps)) - 1]
    draw = uniform if uniform is not None else secrets.SystemRandom().uniform
    return draw(0.0, cap)


async def _model_retry_sleep(delay):
    await asyncio.sleep(delay)


def _warn_model_call_retry(attempt, reason, delay_seconds):
    """Log only the attempt count, a fixed reason and the delay; never the error text."""
    if (
        type(attempt) is not int
        or not 1 <= attempt <= _MODEL_CALL_RETRIES
        or type(reason) is not str
        or reason not in {"throttled", "server_error", "connection"}
        or type(delay_seconds) is not float
        or not 0.0 <= delay_seconds <= _MODEL_RETRY_CAP_SECONDS[-1]
    ):
        return
    with contextlib.suppress(Exception):
        logger.warning(
            "model_call_retry run_id=%s attempt=%d limit=%d reason=%s delay_ms=%d",
            _active_run_label(),
            attempt,
            _MODEL_CALL_RETRIES,
            reason,
            round(delay_seconds * 1000),
        )


def _is_sft_plan_preview(use):
    """Classify only the exact supported no-effect SFT preview."""
    if type(use) is not dict or use.get("name") != "submit_qwen_sft":
        return False
    arguments = use.get("input")
    return type(arguments) is dict and arguments.get("plan_only") is True


def _is_effectful_sft_use(use):
    """submit_qwen_sft calls other than the no-effect plan_only preview."""
    return type(use) is dict and use.get("name") == "submit_qwen_sft" and not _is_sft_plan_preview(use)


def _mixed_sft_batch_refusal(messages, tool_use):
    """Closed refusal text for a tool use whose assistant message mixes an effectful SFT with another use.

    The original SFT TASK keeps exactly one tool use in its final assistant message, so the whole message is
    refused before any of its tools runs. None when the message is not such a mixture.
    """
    if type(messages) is not list or type(tool_use) is not dict:
        return None
    assistants = [m for m in messages if type(m) is dict and m.get("role") == "assistant"]
    content = assistants[-1].get("content") if assistants else None
    if type(content) is not list:
        return None
    uses = [
        block["toolUse"] for block in content if type(block) is dict and type(block.get("toolUse")) is dict
    ]
    identifier = tool_use.get("toolUseId")
    if len(uses) < 2 or not any(_is_effectful_sft_use(use) for use in uses):
        return None
    if not any(use.get("toolUseId") == identifier for use in uses):
        return None
    sft = _is_effectful_sft_use(tool_use)
    return json.dumps(
        {
            "success": False,
            "executed": False,
            "status": "batch_refused",
            "tool": tool_use.get("name"),
            "message": (
                "No tool in this message was executed: submit_qwen_sft must be the only tool call in its "
                "message. Call submit_qwen_sft alone, wait for its result, then call other tools in later "
                "messages."
                if sft
                else "Not executed: this message also called submit_qwen_sft, which must be the only tool call "
                "in its message. Call submit_qwen_sft alone first; call this tool again in a later message."
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _warn_normal_sft_capture(run_id, phase, error):
    """Log only fixed local capture labels; never expose exception content."""
    phases = {
        "current_before",
        "message_binding",
        "snapshot_create",
        "causal_call",
        "causal_bounds",
        "causal_checkpoint_data",
        "causal_canonical_data",
        "causal_assistant_batch",
        "causal_unique_sft_use",
        "causal_sft_inputs",
        "causal_call_budget",
        "snapshot_inventory",
        "snapshot_budget",
        "capture_repeat",
        "current_after",
    }
    categories = {
        Exception: "Exception",
        AssertionError: "AssertionError",
        AttributeError: "AttributeError",
        KeyError: "KeyError",
        RuntimeError: "RuntimeError",
        TypeError: "TypeError",
        ValueError: "ValueError",
    }
    if (
        type(run_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
        or type(phase) is not str
        or phase not in phases
    ):
        return
    try:
        logger.warning(
            "normal_sft_capture_unavailable run_id=%s phase=%s error_class=%s",
            run_id,
            phase,
            categories.get(type(error), "Exception"),
        )
    except Exception:
        pass


_NORMAL_CALLBACK_TOOL_CATEGORIES = {
    "research_search": "RESEARCH",
    "research_fetch_urls": "RESEARCH",
    "acquire_model_weights": "MODEL",
    "get_model_acquisition": "MODEL",
    "acquire_huggingface_dataset": "DATA",
    "get_dataset_acquisition": "DATA",
    "freeze_coding_cases": "CASE",
    "get_coding_cases": "CASE",
    "prepare_coding_dataset": "PREP",
    "get_coding_preparation": "PREP",
    "submit_qwen_sft": "SFT",
    "get_qwen_sft": "SFT",
    "reload_qwen_sft": "RELOAD",
    "submit_coding_generation": "GEN",
    "get_coding_generation": "GEN",
    "submit_coding_pair": "PAIR",
    "get_coding_pair": "PAIR",
}


def _warn_normal_callback_activity(phase, *, tool_category="NONE"):
    """Optional current-run metadata; callback return is not dispatch or admission."""
    try:
        if (
            type(phase) is not str
            or phase
            not in {
                "before_model_enter",
                "before_model_exit",
                "after_model_enter",
                "before_tool_enter",
                "before_tool_exit",
                "after_tool_enter",
                "after_tool_exit",
            }
            or type(tool_category) is not str
            or tool_category
            not in {
                "NONE",
                "RESEARCH",
                "MODEL",
                "DATA",
                "CASE",
                "PREP",
                "SFT",
                "RELOAD",
                "GEN",
                "PAIR",
                "OTHER",
            }
        ):
            return
        from ...control_plane.normal_run_checkpoint import current_normal_run

        context = current_normal_run()
        if context is not None:
            logger.warning(
                "normal_callback_activity run_id=%s phase=%s tool_category=%s",
                context.run_id,
                phase,
                tool_category,
            )
    except Exception:
        return


def _normal_run_console_silent():
    """True while a prompt run is active in the runtime, also when its context fails its own check."""
    from ...control_plane.normal_run_checkpoint import current_normal_run

    try:
        return current_normal_run() is not None
    except Exception:
        return True


def _warn_input_hold(message):
    """Best-effort closed metadata; ordinary logging failures cannot alter refusal."""
    if type(message) is str and message in _INPUT_HOLD_WARNINGS:
        try:
            logger.warning(message)
        except Exception:
            pass


def _warn_precommit_handoff(phase, pending, status=None, ready=False):
    """Optional fixed scalars; never inspect original graph, handles or history."""
    try:
        if phase == "BATCH_COMMITTED" and pending is None:
            kind = "BATCH"
        elif type(pending) is _PendingOriginalModel:
            kind = "MODEL"
        elif type(pending) is _PendingOriginalProducer:
            kind = {"data": "DATA", "case": "CASE", "preparation": "PREP"}.get(pending.kind)
            if kind is None:
                return
        else:
            return
        states = {
            "PENDING": "PENDING",
            "RUNNING": "RUNNING",
            "SUCCEEDED": "SUCCEEDED",
            "FAILED": "FAILED",
            "STOPPED": "STOPPED",
            "UNKNOWN": "UNKNOWN",
            "observation_failed": "OBSERVATION_FAILED",
            "verification_failed": "VERIFICATION_FAILED",
            "output_verification_failed": "VERIFICATION_FAILED",
            "failed": "FAILED",
            "stopped": "STOPPED",
            "submission_rejected": "SUBMISSION_REJECTED",
        }
        state = states.get(status, "UNKNOWN") if type(status) is str else "UNKNOWN"
        if kind == "DATA" and type(status) is str:
            state = {"running": "RUNNING", "submitted": "PENDING", "completed": "SUCCEEDED"}.get(
                status, state
            )
        from ...control_plane.normal_run_checkpoint import _warn_normal_handoff

        _warn_normal_handoff(phase, kind, state, ready)
    except Exception:
        return


# Execution status values for a single tool invocation.
ExecStatus = str  # "executed" | "needs_confirmation" | "error" | "job_submitted"


@dataclass
class ToolExecution:
    """The outcome of dispatching one tool_use through the confirmation gate."""

    tool_name: str
    args: dict[str, Any]
    status: ExecStatus
    result: Any = None
    result_summary: str = ""


@dataclass(frozen=True)
class GuardrailOutcome:
    """Intervention DATA only; assessment/action are unavailable, never a grant."""

    schema_version: int = 1
    stop_reason: str = "guardrail_intervened"
    outcome: str = "INTERVENED"
    assessment_status: str = "unavailable"
    action: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or any(
                type(value) is not str or value != expected
                for value, expected in (
                    (self.stop_reason, "guardrail_intervened"),
                    (self.outcome, "INTERVENED"),
                    (self.assessment_status, "unavailable"),
                    (self.action, "UNKNOWN"),
                )
            )
        ):
            raise ValueError("closed guardrail intervention DATA required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stop_reason": self.stop_reason,
            "outcome": self.outcome,
            "assessment_status": self.assessment_status,
            "action": self.action,
        }


@dataclass
class AgentTurn:
    """A complete tool-bound agent turn: final assistant text + tool trace.

    ``ChatEngine`` renders this back into the chat pane (a compact "🔧 used: …"
    affordance line above the final text) so users see which REAL tools fired.
    """

    final_text: str
    executions: list[ToolExecution] = field(default_factory=list)
    guardrail_outcome: GuardrailOutcome | None = None
    guardrail_assessment: GuardrailAssessment | None = None
    input_admission: dict[str, Any] | None = None


_MODEL_UNKNOWN_OBSERVATION_LIMIT = 8


@dataclass(frozen=True)
class _PendingOriginalModel:
    tool_use_id: str
    reference: str
    store: SessionModelAcquisitionStore
    record: Any
    agent: Any
    graph: tuple[Any, ...]
    handle_pin: str
    pid: int


@dataclass
class _PendingBatchPause:
    agent: Any
    invocation_state: dict
    request_state: dict
    research: Any
    checker: Any
    pid: int
    prefix: str
    incoming: tuple[str, ...] = ()
    batch: str | None = None
    admitted: dict = field(default_factory=dict)
    originals: dict = field(default_factory=dict)
    history: str | None = None
    marked: bool = False
    settled: bool = False
    accepted: bool = False
    precommit: bool = False
    completed: dict = field(default_factory=dict)
    handled: set = field(default_factory=set)
    registry: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _PendingOriginalProducer:
    kind: str
    tool_use_id: str
    reference: str
    store: Any
    record: Any
    adapter: Any
    agent: Any
    graph: tuple[Any, ...]
    handle_pin: Any
    pid: int


_ORIGINAL_PRODUCERS = {
    "acquire_huggingface_dataset": (
        "data",
        "_acquisition_store",
        "_amlas_acquisition_store",
        "request_ref",
        "acq_",
        "data_collection_flow",
    ),
    "freeze_coding_cases": (
        "case",
        "_case_freeze_store",
        "_amlas_case_freeze_store",
        "case_ref",
        "case_",
        "coding_case_freeze_flow",
    ),
    "prepare_coding_dataset": (
        "preparation",
        "_preparation_store",
        "_amlas_preparation_store",
        "request_ref",
        "prep_",
        "data_preparation_flow",
    ),
    "submit_qwen_sft": ("sft", "_sft_store", "_amlas_sft_store", "request_ref", "sft_", "qwen_sft_flow"),
    "reload_qwen_sft": ("reload", "_sft_store", "_amlas_sft_store", "request_ref", "sft_", "qwen_sft_flow"),
    "submit_coding_generation": (
        "generation",
        "_generation_store",
        "_amlas_generation_store",
        "request_ref",
        "gen_",
        "coding_generation_flow",
    ),
    "submit_coding_pair": (
        "pair",
        "_pair_store",
        "_amlas_pair_store",
        "request_ref",
        "pair_",
        "coding_pair_flow",
    ),
}


_COMPLETION_EVENT_MAX_BYTES = 4096


@dataclass(frozen=True)
class _OriginalModelCompletionEvent:
    """Source-owned completion DATA; never a prompt selector or authority grant."""

    request_ref: str
    tool_use_id: str

    def as_content(self) -> list[dict[str, str]]:
        if (
            type(self.request_ref) is not str
            or not re.fullmatch(r"model_acq_[a-f0-9]{32}", self.request_ref)
            or type(self.tool_use_id) is not str
            or not 1 <= len(self.tool_use_id) <= 256
        ):
            raise ValueError("original completion DATA identifiers differ")
        value = {
            "schema_version": 1,
            "event_type": "original_model_acquisition_completed",
            "data_only": True,
            "request_ref": self.request_ref,
            "tool_use_id": self.tool_use_id,
            "status": "SUCCEEDED",
            "acquired": True,
            "file_count": 9,
            "manifest_roles": ["receipt", "commit", "lineage"],
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        if len(encoded.encode("utf-8")) > _COMPLETION_EVENT_MAX_BYTES:
            raise ValueError("original completion DATA exceeds its fixed transport bound")
        return [{"text": encoded}]


_MODEL_DIAGNOSTIC_FIELDS = (
    "phase",
    "error_class",
    "error_code",
    "native_error_code",
    "request_id",
    "http_status",
)


def _model_diagnostic_now():
    return time.monotonic()


@dataclass(frozen=True)
class _OriginalModelDiagnosticPublication:
    """Private fresh source binding; only six immutable scalars may leave it."""

    owner: Any
    pending: _PendingOriginalModel
    context: Any
    values: tuple
    deadline: float
    pid: int
    worker_thread: int

    def __post_init__(self):
        from ...backends.adapters.model_weight_acquisition import _project_model_diagnostic

        if (
            type(self.owner) is not ToolBindingBackend
            or type(self.pending) is not _PendingOriginalModel
            or type(self.context) is not Context
            or type(self.values) is not tuple
            or len(self.values) != 6
            or type(self.pid) is not int
            or self.pid != os.getpid()
            or type(self.worker_thread) is not int
            or self.worker_thread != threading.get_ident()
            or type(self.deadline) is not float
            or not math.isfinite(self.deadline)
            or not _model_diagnostic_now() < self.deadline <= _model_diagnostic_now() + 15.0
        ):
            raise ValueError("closed original diagnostic publication required")
        value = _project_model_diagnostic(
            dict(zip(_MODEL_DIAGNOSTIC_FIELDS, self.values, strict=True)), "create_boundary"
        )
        if value is None or tuple(value[key] for key in _MODEL_DIAGNOSTIC_FIELDS) != self.values:
            raise ValueError("closed original diagnostic scalars required")

    @property
    def key(self):
        return id(self.owner), id(self.pending.record), self.pending.handle_pin

    def current_data(self, slot):
        from ...backends.adapters.model_weight_acquisition import _project_model_diagnostic

        if self.pid != os.getpid() or not slot._is_active() or _model_diagnostic_now() >= self.deadline:
            return None

        def qualified_data():
            self.owner._qualify_warm_original(self.pending)
            self.owner._require_diagnostic_origin_current(self.pending)
            current = _project_model_diagnostic(
                self.pending.record.handle.extra.get("submission_diagnostic"), "create_boundary"
            )
            if current is None or tuple(current[key] for key in _MODEL_DIAGNOSTIC_FIELDS) != self.values:
                return None
            if not slot._is_active() or self.pid != os.getpid() or _model_diagnostic_now() >= self.deadline:
                return None
            return dict(zip(_MODEL_DIAGNOSTIC_FIELDS, self.values, strict=True))

        return self.context.run(qualified_data)


_WARM_REFUSAL_PHASES = frozenset(
    [
        "initial_qualification",
        "native_observation",
        "post_observation_qualification",
        "observation_identity",
        "artifact_result",
        "artifact_inventory",
        "history_causality",
        "consume_transition",
        "resume_qualification",
        "resume",
        "post_resume_qualification",
        "exposure",
    ]
)
_WARM_REFUSAL_REASONS = frozenset(
    [
        "owner_unavailable",
        "pending_closed_or_replaced",
        "graph_changed",
        "consumption_limit",
        "original_unknown_budget_exhausted",
        "invocation_context_unqualified",
        "record_pin_changed",
        "correlation_changed",
        "campaign_unqualified",
        "policy_refused",
        "observation_identity_changed",
        "artifact_verification_failed",
        "artifact_result_unverified",
        "artifact_inventory_invalid",
        "artifact_pin_invalid",
        "history_causality_changed",
        "transition_closed_or_replaced",
        "continuation_failed",
        "exposure_closed_or_replaced",
        "UNKNOWN",
    ]
)
_WARM_REFUSAL_CLASSES = frozenset(
    [
        "ValueError",
        "TypeError",
        "RuntimeError",
        "TimeoutError",
        "ClientError",
        "ParamValidationError",
        "UNKNOWN",
    ]
)


@dataclass(frozen=True)
class _WarmRefusalDiagnostic:
    """Closed source-phase DATA, never caller/native permission or error text."""

    phase: str
    reason: str
    error_class: str

    def __post_init__(self):
        if (
            type(self.phase) is not str
            or self.phase not in _WARM_REFUSAL_PHASES
            or type(self.reason) is not str
            or self.reason not in _WARM_REFUSAL_REASONS
            or type(self.error_class) is not str
            or self.error_class not in _WARM_REFUSAL_CLASSES
        ):
            raise ValueError("closed warm refusal diagnostic required")

    def as_data(self):
        self.__post_init__()
        return {"phase": self.phase, "reason": self.reason, "error_class": self.error_class}


class _WarmSourceRefusalError(ValueError):
    def __init__(self, reason, error_class="ValueError"):
        if type(reason) is not str or reason not in _WARM_REFUSAL_REASONS:
            raise ValueError("closed warm refusal reason required")
        if type(error_class) is not str or error_class not in _WARM_REFUSAL_CLASSES:
            raise ValueError("closed warm refusal class required")
        self.reason, self.error_class = reason, error_class
        super().__init__("original warm source boundary refused")


def _warm_error_class(error):
    from botocore.exceptions import ClientError, ParamValidationError

    if isinstance(error, ParamValidationError):
        return "ParamValidationError"
    if isinstance(error, ClientError):
        return "ClientError"
    if isinstance(error, ValueError):
        return "ValueError"
    if isinstance(error, TypeError):
        return "TypeError"
    if isinstance(error, RuntimeError):
        return "RuntimeError"
    if isinstance(error, TimeoutError):
        return "TimeoutError"
    return "UNKNOWN"


@dataclass(frozen=True)
class _WarmModelStep:
    disposition: str
    turn: AgentTurn | None = None
    diagnostic: _WarmRefusalDiagnostic | None = None

    def __post_init__(self):
        if (
            type(self.disposition) is not str
            or self.disposition
            not in {
                "awaiting_original",
                "submission_unresolved",
                "continued_original",
                "terminal_failed",
                "refused",
                "guardrail_intervened",
            }
            or (self.turn is not None and type(self.turn) is not AgentTurn)
            or (self.diagnostic is not None and type(self.diagnostic) is not _WarmRefusalDiagnostic)
        ):
            raise ValueError("closed warm model DATA required")
        if self.diagnostic is not None:
            self.diagnostic.__post_init__()


def _resolve_callable(tool: Any) -> Callable[..., Any]:
    """Return the underlying plain function for a (possibly decorated) tool.

    Under the confcompatibility shim ``@tool`` is the identity, so ``tool`` is already the
    raw function. With real Strands, the ``DecoratedFunctionTool`` exposes the
    original via one of these attributes — we prefer it so direct invocation in
    ``execute_tool_use`` calls the real pure-Python implementation.
    """
    for attr in ("_tool_func", "func", "__wrapped__", "_func", "original"):
        candidate = getattr(tool, attr, None)
        if callable(candidate):
            resolved: Callable[..., Any] = candidate
            return resolved
    fallback: Callable[..., Any] = tool
    return fallback


def _summarize(result: Any) -> str:
    """One-line, chat-friendly summary of a tool result for the trace affordance."""
    if isinstance(result, dict):
        parts: list[str] = []
        for k, v in result.items():
            if isinstance(v, (list, dict)):
                continue
            parts.append(f"{k}={v}")
            if len(parts) >= 4:
                break
        return ", ".join(parts)[:200] or "ok"
    return str(result)[:200]


def render_agent_turn(turn: AgentTurn) -> str:
    """Render an ``AgentTurn`` into chat markdown: tool trace + final text.

    - No tool calls → the plain final text (indistinguishable from normal chat).
    - With tool calls → a "🔧 used: …" line for executed tools, any pending
      confirmation prompts, any errors, then the final assistant text.
    """
    if not turn.executions:
        return turn.final_text

    executed = [e for e in turn.executions if e.status == "executed"]
    pending = [e for e in turn.executions if e.status == "needs_confirmation"]
    errored = [e for e in turn.executions if e.status == "error"]

    lines: list[str] = []
    if executed:
        lines.append("🔧 used: " + ", ".join(e.tool_name for e in executed))
    for e in pending:
        lines.append(f"⏸️ {e.result_summary}")
    for e in errored:
        lines.append(f"⚠️ {e.tool_name}: {e.result_summary}")
    if turn.final_text:
        lines.append("")
        lines.append(turn.final_text)
    return "\n".join(lines)


class _InputAdmissionHeldError(Exception):
    """Private pre-exposure hold; carries no rejected text or native handle."""


class ToolBindingBackend:
    """Builds/owns the Strands tool-bound agent and the confirmation-gated dispatch.

    The deterministic, shim-safe seam is :meth:`execute_tool_use` — it invokes the
    SAME real ``@tool`` callable a live Strands agent would, so offline validation prove the
    NL→tool→real-effect wiring without a live LLM. :meth:`chat_with_tools` is the
    full LLM path, exercised live (``agents`` extra + creds) or via an injected
    model/agent in tests.
    """

    def __init__(
        self,
        settings: Settings,
        model: Any | None = None,
        job_manager: Any | None = None,
        *,
        approval_policy: ApprovalPolicy | None = None,
        normal_restore=None,
    ) -> None:
        self.settings = settings
        self.approval_policy = approval_policy
        from ...control_plane.producer_snapshot import current_sft_campaign, hydrate_sft_campaign

        context = current_sft_campaign()
        from ...control_plane.campaign_runtime import current_campaign_entry
        from ...control_plane.model_campaign_binding import current_model_campaign_services

        entry = current_campaign_entry()
        services = current_model_campaign_services() if entry is not None else None
        if entry is not None:
            entry.require_current()
        self._model_selectors = model_weight_acquisition_selectors()
        self._normal_original_result = None
        if normal_restore is not None:
            from ...control_plane.normal_run_checkpoint import NormalRunContext, current_normal_run

            normal, _task, parents, original, binding = normal_restore
            if (
                type(normal) is not NormalRunContext
                or current_normal_run() is not normal
                or entry is not None
                or context is not None
            ):
                raise ValueError("current owned normal restoration required")
            acquired, models, cases, prepared = parents
            sft, self._normal_original_result = normal.restore_sft_store(
                parents, original, binding, policy=self.approval_policy
            )
        elif context is None:
            acquired = (
                SessionAcquisitionStore._for_campaign_entry(entry)
                if entry is not None
                else SessionAcquisitionStore()
            )
            models = build_model_acquisition_store(
                runtime_session_id=acquired.runtime_session_id,
                campaign_entry=entry,
                campaign_services=services,
                policy_getter=lambda: self.approval_policy,
            )
            cases = build_case_freeze_store(acquired)
            prepared = build_preparation_store(acquired, case_store=cases)
            sft = build_sft_store(models, prepared)
        else:
            if entry is not None and (
                context.execution_binding.campaign_id != entry.campaign_id
                or context.execution_binding.runtime_session_id != entry.private_store_correlation
            ):
                raise ValueError("original campaign/SFT correlation differs")
            acquired, models, prepared, sft = hydrate_sft_campaign(context, policy=self.approval_policy)
            cases = build_case_freeze_store(acquired)
            prepared._case_store = cases
        models._policy_getter = lambda: self.approval_policy
        generation = build_generation_store(models, cases, sft)
        if normal_restore is not None:
            generation._planning_restorer = self._restore_planning_refs
        paired = build_pair_store(generation, cases)
        # Publish the entire temporary graph only after original producer/SFT verification.
        (
            self._acquisition_store,
            self._model_acquisition_store,
            self._case_freeze_store,
            self._preparation_store,
            self._sft_store,
            self._generation_store,
            self._pair_store,
        ) = acquired, models, cases, prepared, sft, generation, paired
        from ...control_plane.campaign_planning_services import current_campaign_planning_services

        self._campaign_planning_services = current_campaign_planning_services()
        self._campaign_planning_store = (
            self._campaign_planning_services.build_for_entry(entry, generation)
            if entry is not None and self._campaign_planning_services is not None
            else None
        )
        self._model = model  # optional pre-built model; callers may inject a real BedrockModel
        # Name → real callable registry (the tools the LLM may invoke). Built lazily
        # because the tool modules import ``strands`` (optional dep) — constructing
        # the backend (e.g. at TUI launch without ``strands``) must NOT require it.
        self._tool_registry_cache: dict[str, Callable[..., Any]] | None = None
        # Per-instance read-only set (copy so callers can extend it deterministically).
        self._readonly_names: set[str] = set(READ_ONLY_TOOLS)
        # DP-2 (Phase 11): when set, a CONFIRMED long-running tool enqueues a
        # background job instead of blocking the agent turn (design §2.2 / §3). When
        # None (the default / tests), confirmed long ops execute inline as before.
        self.job_manager = job_manager
        # Feature 2 (ephemeral thought ticker): the per-turn ``ThoughtTrace`` for
        # the coarse ``execute_tool_use`` fallback. ``chat_with_tools`` sets this at
        # turn start (and clears it at turn end) so the deterministic tool seam can
        # emit tool_start/tool_end ticker content even when the granular Strands
        # streaming callback shape is unavailable (design §3.2b). ``None`` (the
        # default) → NO emission, so ``execute_tool_use`` is byte-identical when no
        # ticker is wired (every existing caller / headless path).
        self._thought_trace: ThoughtTrace | None = None
        self._warm_lock = threading.RLock()
        self._warm_pending: _PendingOriginalModel | None = None
        self._warm_pending_producers = {}
        self._warm_producer_cursor = 0
        self._warm_case_failure_counts = {}
        self._warm_closed: str | None = None
        self._warm_closed_diagnostic: _WarmRefusalDiagnostic | None = None
        self._warm_capture_lock = threading.Lock()
        self._warm_owner = None
        self._warm_pid = os.getpid()
        self._warm_consumed = set()
        self._warm_unknown_pending = None
        self._warm_unknown_polls = 0
        self._precommit_unresolved = {}
        self._input_checker = None
        self._input_guard_client = None
        self._input_guard_client_factory = None
        self._input_cycle = secrets.token_hex(16)
        self._input_receipts: list[Any] = []
        self._input_hold = False
        self._input_quarantined = False
        self._input_lock = threading.RLock()
        self._research_binding = None
        self._research_started = False
        self._pending_batch_pause = None
        self._runtime_original_completion = None
        # Tool uses answered by the SFT batch admission (id -> exact refusal text) and repeat
        # submit_qwen_sft uses answered by the SFT store; neither is a native producer.
        self._refused_tool_results = {}
        self._sft_repeat_ids = set()

    def _restore_planning_refs(self, store):
        """Re-register the specification references an earlier worker handed out, from the admitted history.

        The restored conversation still holds the earlier bind and resolve calls and their results. Each original
        specification is re-registered only when its canonical digest equals the one the earlier result carried,
        and each resolved specification only when a fresh original-case observation reproduces the same
        reference. Nothing is submitted; an entry that does not verify is skipped and its reference stays unknown.
        """
        agent = getattr(getattr(self, "_warm_owner", None), "_normal_agent", None)
        messages = getattr(agent, "messages", None)
        if type(messages) is not list:
            return
        uses, results = {}, {}
        for message in messages:
            content = message.get("content") if type(message) is dict else None
            for block in content if type(content) is list else ():
                if type(block) is dict and type(block.get("toolUse")) is dict:
                    uses[block["toolUse"].get("toolUseId")] = block["toolUse"]
                elif type(block) is dict and type(block.get("toolResult")) is dict:
                    results[block["toolResult"].get("toolUseId")] = block["toolResult"]

        def payload(identifier, flag):
            result = results.get(identifier)
            if type(result) is not dict or result.get("status") != "success":
                return None
            try:
                value = self._planning_result_payload(result)
            except Exception:
                return None
            return value if value.get(flag) is True else None

        def restore_original(use, value):
            store.restore_original_spec(
                value["spec_ref"], use["input"]["yaml_text"], expected_spec_sha256=value["spec_sha256"]
            )

        def restore_resolved(use, value):
            restored = store.resolve_case_spec(use["input"]["case_request_ref"], use["input"]["spec_ref"])
            if restored["spec_ref"] != value["spec_ref"]:
                raise ValueError("resolved specification reference differs")

        # Originals first: a resolved specification is minted from its original.
        for name, flag, restore in (
            ("bind_coding_generation_spec", "success", restore_original),
            ("resolve_coding_evaluation_spec", "case_bound", restore_resolved),
        ):
            for identifier, use in uses.items():
                if type(use) is not dict or use.get("name") != name:
                    continue
                value = payload(identifier, flag)
                if value is None:
                    continue
                try:
                    restore(use, value)
                except Exception as error:
                    logger.debug("planning reference not restored (%s)", type(error).__name__)

    def _refuse_mixed_sft_batch(self, event):
        """BeforeToolCall: refuse every tool use of a message that mixes an effectful SFT with other tools."""
        text = _mixed_sft_batch_refusal(event.agent.messages, event.tool_use)
        if text is None:
            return
        self.__dict__.setdefault("_refused_tool_results", {})[event.tool_use["toolUseId"]] = text
        event.cancel_tool = text

    def _refused_batch_result(self, event):
        """True for the exact error result the SFT batch admission returned for this tool use."""
        identifier = event.tool_use.get("toolUseId") if type(event.tool_use) is dict else None
        text = getattr(self, "_refused_tool_results", {}).get(identifier)
        return text is not None and event.result == {
            "toolUseId": identifier,
            "status": "error",
            "content": [{"text": text}],
        }

    def _is_original_producer_use(self, use):
        """Producer tools that create native originals; previews and repeat SFT answers create none."""
        name = use.get("name") if type(use) is dict else None
        return (
            (name in _ORIGINAL_PRODUCERS or name == "acquire_model_weights")
            and not _is_sft_plan_preview(use)
            and use.get("toolUseId") not in getattr(self, "_sft_repeat_ids", ())
        )

    def _sft_repeat_kind(self, use):
        """'same' or 'different' once this normal run holds an original SFT, else None."""
        from ...control_plane.normal_run_checkpoint import _effective_sft_inputs

        try:
            selection = _effective_sft_inputs(use.get("input"))
        except ValueError:
            return None
        repeat_kind = getattr(getattr(self, "_sft_store", None), "repeat_kind", None)
        if not callable(repeat_kind):
            return None
        return repeat_kind(
            selection["model_request_ref"], selection["preparation_request_ref"], selection["recipe_id"]
        )

    def _bind_research_units(self, plan, *, owner, current_qualification):
        """Private trusted-code preexecution binding; unit DATA grants nothing."""
        from ...config.guardrails import guardrail_checks_enabled
        from .engine import _WARM_CHAT_ENGINE_TYPE
        from .research_input_recovery import _validate_plan

        with self._input_lock:
            if (
                self._research_binding is not None
                or self._research_started
                or self._input_hold
                or self._input_receipts
                or type(owner) is not _WARM_CHAT_ENGINE_TYPE
                or owner.tool_backend is not self
                or self._warm_owner is not owner
                or self._warm_pid != os.getpid()
                or not callable(current_qualification)
            ):
                raise ValueError("original preexecution research owner required")
            units = _validate_plan(plan)
            current_qualification()
            checks_enabled = guardrail_checks_enabled()
            checker = self._selected_input_checker()
            if checker is None and checks_enabled:
                raise ValueError("selected research INPUT checker required")
            from ...tools.research import strands_tools

            fetch = self._tool_registry.get("research_fetch_urls")
            if fetch is not _resolve_callable(strands_tools.research_fetch_urls):
                raise ValueError("original curated research registry required")
            self._research_binding = {
                "owner": owner,
                "pid": os.getpid(),
                "checker": checker,
                "config": checker.config if checker is not None else None,
                "checks_enabled": checks_enabled,
                "qualify": current_qualification,
                "plan": plan,
                "units": units,
                "fetch": fetch,
                "phase": "UNUSED",
                "cycle": None,
                "pending": False,
                "attempted": set(),
                "tool_uses": {},
                "blocked": set(),
                "withheld": set(),
                "results": set(),
                "admitted": set(),
                "forwarded": set(),
                "original_completion_ready": False,
            }

    def _qualify_research_binding(self, *, active=True):
        state = self._research_binding
        if state is None:
            return None
        from .engine import _WARM_CHAT_ENGINE_TYPE

        if (
            self._input_hold
            or state["pid"] != os.getpid()
            or type(state["owner"]) is not _WARM_CHAT_ENGINE_TYPE
            or self._warm_owner is not state["owner"]
            or state["owner"].tool_backend is not self
            or self._warm_closed is not None
            or any(
                now is not original
                for now, original in zip(self._warm_graph(), self._warm_owner_graph, strict=True)
            )
            or (
                active
                and state["phase"] != "CONSUMED"
                and (state["phase"] != "ACTIVE" or state["cycle"] != self._input_cycle)
            )
        ):
            raise _InputAdmissionHeldError()
        state["qualify"]()
        from ...config.guardrails import guardrail_checks_enabled

        checker = self._selected_input_checker()
        if (
            guardrail_checks_enabled() is not state["checks_enabled"]
            or checker is not state["checker"]
            or (checker.config if checker is not None else None) != state["config"]
            or self._tool_registry.get("research_fetch_urls") is not state["fetch"]
        ):
            raise _InputAdmissionHeldError()
        return state

    def _activate_research_binding(self):
        with self._input_lock:
            state = self._research_binding
            self._research_started = True
            if state is None:
                return
            if state["phase"] != "UNUSED":
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold("AMLAS_INPUT_HOLD phase=research_activation reason=binding_not_unused")
                raise _InputAdmissionHeldError()
            self._qualify_research_binding(active=False)
            state["cycle"] = self._input_cycle
            state["phase"] = "ACTIVE"

    @staticmethod
    def _research_unit_eligible(state, unit):
        if unit.unit_id in state["attempted"]:
            return False
        if unit.dependencies or unit.variant_of is not None:
            return False
        blocked_origins = {state["units"][name].origin_id for name in state["blocked"]}
        blocked_urls = {url for name in state["blocked"] for url in state["units"][name].urls}
        return unit.origin_id not in blocked_origins and not set(unit.urls) & blocked_urls

    @staticmethod
    def _research_independent_admitted(state):
        """Every retained prior source unit must be strict ALLOW and independent."""
        blocked_origins = {state["units"][name].origin_id for name in state["blocked"]}
        blocked_urls = {url for name in state["blocked"] for url in state["units"][name].urls}
        return bool(state["forwarded"]) and all(
            name in state["admitted"]
            and name not in state["blocked"]
            and not unit.dependencies
            and unit.variant_of is None
            and unit.origin_id not in blocked_origins
            and not set(unit.urls) & blocked_urls
            for name in state["forwarded"]
            for unit in (state["units"][name],)
        )

    def _before_research_tool(self, event):
        from .research_input_recovery import _CANCELLED

        with self._input_lock:
            already_held = self._input_hold
            try:
                state = self._qualify_research_binding()
                if self._input_hold or (state is not None and state["pending"]):
                    raise _InputAdmissionHeldError()
                if state is None:
                    return
                use = event.tool_use
                if type(use) is dict and use.get("name") not in {"research_fetch_urls", "research_search"}:
                    if use.get("name") not in self._tool_registry:
                        raise _InputAdmissionHeldError()
                    # A single fixed research phase precedes ordinary ML tools.
                    # Consumption is terminal even if the ordinary tool refuses;
                    # it grants no native/capability permission or new research.
                    if state["phase"] != "CONSUMED":
                        state["original_completion_ready"] = self._runtime_research_settled(state)
                        state["phase"] = "CONSUMED"
                    ready = state["original_completion_ready"] is True
                    witness = self._pending_batch_pause
                    if ready:
                        if (
                            witness is None
                            or witness.agent is not event.agent
                            or witness.invocation_state is not event.invocation_state
                        ):
                            raise _InputAdmissionHeldError()
                        witness.precommit = True
                    return
                if state["phase"] == "CONSUMED":
                    raise _InputAdmissionHeldError()
                if (
                    type(use) is not dict
                    or use.get("name") != "research_fetch_urls"
                    or use["name"] not in self._tool_registry
                    or type(use.get("toolUseId")) is not str
                    or not 1 <= len(use["toolUseId"]) <= 256
                ):
                    raise _InputAdmissionHeldError()
                arguments = use.get("input")
                if (
                    type(arguments) is not dict
                    or set(arguments) - {"urls", "max_pages"}
                    or type(arguments.get("urls")) is not list
                    or any(type(url) is not str for url in arguments["urls"])
                    or type(arguments.get("max_pages", 3)) is not int
                    or arguments.get("max_pages", 3) != 3
                ):
                    raise _InputAdmissionHeldError()
                candidates = [
                    unit
                    for unit in state["units"].values()
                    if unit.urls == tuple(arguments["urls"]) and self._research_unit_eligible(state, unit)
                ]
                if len(candidates) != 1 or use["toolUseId"] in state["tool_uses"]:
                    raise _InputAdmissionHeldError()
                unit = candidates[0]
                state["attempted"].add(unit.unit_id)
                state["tool_uses"][use["toolUseId"]] = unit.unit_id
                self._qualify_research_binding()
            except Exception:
                self._input_hold = True
                event.cancel_tool = _CANCELLED
                if not already_held:
                    _warn_input_hold("AMLAS_INPUT_HOLD phase=research_before_tool reason=callback_refused")

    def _capture_normal_sft_call(
        self, event, *, _diagnostic_phase=None, _diagnostic_predicate=None, _diagnostic_size=None
    ):
        """Retain qualified original call DATA before its producer effect."""
        from ...control_plane.normal_run_checkpoint import current_normal_run

        use = event.tool_use
        if type(use) is not dict or use.get("name") != "submit_qwen_sft":
            return
        context = current_normal_run()
        witness = self._pending_batch_pause
        if context is None or witness is None:
            return

        def mark_phase(value):
            if _diagnostic_phase is not None:
                try:
                    _diagnostic_phase(value)
                except Exception:
                    pass

        mark_phase("current_before")
        if type(witness) is not _PendingBatchPause or type(witness.precommit) is not bool:
            raise _InputAdmissionHeldError()
        if not witness.precommit:
            return
        self._qualify_pending_pause(witness)
        mark_phase("message_binding")
        messages = event.agent.messages
        if (
            event.agent is not witness.agent
            or event.invocation_state is not witness.invocation_state
            or not messages
            or witness.batch is None
            or self._pause_history(messages[-1]) != witness.batch
            or self._pause_history(messages[:-1]) != witness.prefix
        ):
            raise _InputAdmissionHeldError()
        uses = [block["toolUse"] for block in messages[-1]["content"] if "toolUse" in block]
        if sum(value == use for value in uses) != 1:
            raise _InputAdmissionHeldError()
        if _is_sft_plan_preview(use):
            # The public preview returns before source observation or producer effects.
            self._qualify_pending_pause(witness)
            return
        if self._sft_repeat_kind(use) is not None:
            # An original SFT is already admitted: a repeat is neither captured nor a producer. The store
            # answers it with the recorded submission (same inputs) or a named refusal (other inputs).
            self.__dict__.setdefault("_sft_repeat_ids", set()).add(use["toolUseId"])
            self._qualify_pending_pause(witness)
            return
        capture_sft_call = context.capture_sft_call
        identifier = use["toolUseId"]
        mark_phase("snapshot_create")
        snapshot = _record_session_snapshot(event.agent)
        capture_sft_call(
            identifier,
            _record_messages(messages),
            agent_snapshot_dict=snapshot,
            _diagnostic_phase=mark_phase,
            _diagnostic_predicate=_diagnostic_predicate,
            _diagnostic_size=_diagnostic_size,
        )
        mark_phase("current_after")
        self._qualify_pending_pause(witness)

    def _admit_research_result(self, event, text, name):
        """Atomic readonly result quarantine; never reset a prior hold."""
        if self._research_binding is None or name != "research_fetch_urls":
            return text, False
        if self._refused_batch_result(event):
            return text, False
        from .input_admission import InputAdmissionDecision
        from .research_input_recovery import _WITHHELD

        with self._input_lock:
            state = self._qualify_research_binding()
            use, result = event.tool_use, event.result
            identifier = use.get("toolUseId")
            if (
                state["phase"] != "ACTIVE"
                or state["pending"]
                or identifier not in state["tool_uses"]
                or identifier in state["results"]
                or set(result) != {"toolUseId", "status", "content"}
                or result.get("toolUseId") != identifier
                or result.get("status") != "success"
                or name not in self._tool_registry
            ):
                raise _InputAdmissionHeldError()
            unit_id = state["tool_uses"][identifier]
            arguments = use.get("input")
            if (
                type(arguments) is not dict
                or set(arguments) - {"urls", "max_pages"}
                or type(arguments.get("urls")) is not list
                or tuple(arguments["urls"]) != state["units"][unit_id].urls
                or type(arguments.get("max_pages", 3)) is not int
                or arguments.get("max_pages", 3) != 3
            ):
                raise _InputAdmissionHeldError()
            payload = self._planning_result_payload(result)
            if type(payload.get("pages")) is not list:
                raise _InputAdmissionHeldError()
            state["results"].add(identifier)
            state["pending"] = True
            checker = state["checker"]
            if checker is None:
                if state["checks_enabled"]:
                    raise _InputAdmissionHeldError()
                self._qualify_research_binding()
                state["forwarded"].add(unit_id)
                return text, True
            decision = checker.check(
                text, boundary="TOOL_RESULT", opaque_cycle=self._input_cycle, tool_name=name
            )
            if type(decision) is not InputAdmissionDecision:
                raise _InputAdmissionHeldError()
            self._input_receipts.append(decision.receipt)
            self._input_receipts = self._input_receipts[-128:]
            self._qualify_research_binding()
            if decision.outcome in {"ALLOW", "ANONYMIZED"} and decision.forward_text is not None:
                state["forwarded"].add(unit_id)
                if (
                    decision.outcome == "ALLOW"
                    and decision.receipt.reason == "NATIVE_NONE"
                    and decision.forward_text == text
                ):
                    state["admitted"].add(unit_id)
                return decision.forward_text, True
            if decision.outcome != "BLOCKED" or decision.receipt.reason != "NATIVE_BLOCKED":
                raise _InputAdmissionHeldError()
            state["blocked"].add(unit_id)
            notice = checker.check(
                _WITHHELD, boundary="TOOL_RESULT", opaque_cycle=self._input_cycle, tool_name=name
            )
            if type(notice) is not InputAdmissionDecision:
                raise _InputAdmissionHeldError()
            self._input_receipts.append(notice.receipt)
            self._input_receipts = self._input_receipts[-128:]
            self._qualify_research_binding()
            if notice.outcome != "ALLOW" or notice.forward_text != _WITHHELD:
                raise _InputAdmissionHeldError()
            if not any(
                self._research_unit_eligible(state, unit) for unit in state["units"].values()
            ) and not self._research_independent_admitted(state):
                raise _InputAdmissionHeldError()
            event.result = {"toolUseId": identifier, "status": "error", "content": [{"text": _WITHHELD}]}
            state["withheld"].add(identifier)
            self._input_quarantined = True
            state["pending"] = False
            return None, True

    def _research_withheld_result(self, event):
        state = self._research_binding
        return state is not None and event.tool_use.get("toolUseId") in state["withheld"]

    def _selected_input_checker(self):
        """Reuse the selected native guard; never change guard scope or settings."""
        from ...config.aws_region import resolve_aws_region
        from ...config.guardrails import guardrail_model_kwargs

        selected = guardrail_model_kwargs()
        if not selected:
            if self._input_checker is not None:
                raise _InputAdmissionHeldError()
            return None
        from .input_admission import InputAdmissionChecker, InputAdmissionConfig

        region = resolve_aws_region(self.settings.aws.effective_bedrock_region)
        config = InputAdmissionConfig(selected["guardrail_id"], selected["guardrail_version"], region)
        if self._input_checker is None:
            import boto3
            from botocore.config import Config

            session = boto3.Session(profile_name=self.settings.aws.effective_profile, region_name=region)
            options = Config(connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 1})
            self._input_guard_client_factory = lambda: session.client(
                "bedrock-runtime", region_name=region, config=options
            )
            client = self._input_guard_client_factory()
            self._input_guard_client = client
            self._input_checker = InputAdmissionChecker(
                config, client=client, hmac_key=secrets.token_bytes(32), close_before_assessment=True
            )
        if self._input_checker.config != config:
            raise _InputAdmissionHeldError()
        return self._input_checker

    def _refresh_input_checker(self, checker, research):
        """Owned proactive pool retirement, never retry of a failed assessment.

        Explicitly disabled checks own no checker, client or factory: there is no
        pool to retire, nothing is assessed and no receipt is written, so this
        only requalifies bound research and returns None. An absent checker in
        any other state (enabled, partially owned transport) refuses.
        """
        if checker is None:
            from ...config.guardrails import guardrail_checks_enabled

            if (
                guardrail_checks_enabled()
                or self._input_checker is not None
                or self._input_guard_client is not None
                or self._input_guard_client_factory is not None
                or (research is not None and (research["checks_enabled"] or research["checker"] is not None))
            ):
                raise _InputAdmissionHeldError()
            if research is not None:
                self._qualify_research_binding(active=False)
            return None
        self._input_guard_client.close()
        client = self._input_guard_client_factory()
        checker = checker._with_client(client)
        self._input_guard_client = client
        self._input_checker = checker
        if research is not None:
            research["checker"] = checker
            self._qualify_research_binding(active=False)
        return checker

    def _admit_input(
        self, text: str, boundary: str, *, immutable: bool = False, tool_name: str = "UNKNOWN"
    ) -> str:
        with self._input_lock:
            if self._input_hold:
                raise _InputAdmissionHeldError()
            try:
                checker = self._selected_input_checker()
                if checker is None:
                    return text
                if boundary == "COMPLETION_DATA":
                    # Retire an owned pool before a new logical assessment, never
                    # after a failed/unknown assessment and never as its retry.
                    research = self._qualify_research_binding(active=False)
                    if research is not None and research["phase"] != "CONSUMED":
                        raise _InputAdmissionHeldError()
                    checker = self._refresh_input_checker(checker, research)
                decision = checker.check(
                    text, boundary=boundary, opaque_cycle=self._input_cycle, tool_name=tool_name
                )
                self._input_receipts.append(decision.receipt)
                self._input_receipts = self._input_receipts[-128:]
                if decision.outcome not in {"ALLOW", "ANONYMIZED"} or decision.forward_text is None:
                    raise _InputAdmissionHeldError()
                if immutable and decision.forward_text != text:
                    raise _InputAdmissionHeldError()
                return decision.forward_text
            except Exception:
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold("AMLAS_INPUT_HOLD phase=input_admission reason=admission_refused")
                raise _InputAdmissionHeldError() from None

    def _input_admission_data(self):
        with self._input_lock:
            if not self._input_receipts and not self._input_hold:
                return None
            return {
                "schema_version": 1,
                "data_only": True,
                "status": "held"
                if self._input_hold
                else ("quarantined" if self._input_quarantined else "admitted"),
                "records": [receipt.to_dict() for receipt in self._input_receipts[-128:]],
                "native_admission": False,
                "quality_success": False,
                "blocked_component_attribution": "UNESTABLISHED",
            }

    def _input_hold_turn(self, *, guardrail_assessment: GuardrailAssessment | None = None):
        if guardrail_assessment is not None and type(guardrail_assessment) is not GuardrailAssessment:
            raise TypeError("exact closed guardrail assessment required")
        with self._warm_capture_lock:
            if self._warm_owner is not None:
                self._warm_closed = self._warm_closed or "refused"
        return AgentTurn(
            final_text="Model input was held at the admission boundary.",
            executions=[],
            guardrail_assessment=guardrail_assessment,
            input_admission=self._input_admission_data(),
        )

    def _guard_new_tool_projection(self, event, *, producer: bool):
        name = event.tool_use.get("name") if type(event.tool_use) is dict else None
        is_producer = self._is_original_producer_use(event.tool_use)
        if is_producer is not producer:
            return
        # This callback runs before SDK history/pending-result commitment. Native
        # producer ownership is captured by the preceding private callbacks.
        already_held = self._input_hold
        try:
            result = event.result
            if type(result) is not dict or type(result.get("content")) is not list:
                raise _InputAdmissionHeldError()
            text = _tool_result_admission_text(result["content"])
            curated = READ_ONLY_TOOLS | STATE_CHANGING_TOOLS | {"acquire_model_weights"}
            source_name = (
                name if type(name) is str and name in curated and name in self._tool_registry else "UNKNOWN"
            )
            admitted, research_checked = (
                (text, False) if producer else self._admit_research_result(event, text, source_name)
            )
            if research_checked and admitted is None:
                return
            if not research_checked:
                if producer and name == "submit_qwen_sft" and self._admit_initial_sft_ack(event, text):
                    admitted = text
                else:
                    admitted = self._admit_input(
                        text, "TOOL_RESULT", immutable=producer, tool_name=source_name
                    )
            if admitted != text:
                replacement = json.loads(admitted)
                if type(replacement) is not list or len(replacement) != len(result["content"]):
                    raise _InputAdmissionHeldError()
                for before, after in zip(result["content"], replacement, strict=True):
                    if (
                        type(before) is not dict
                        or type(after) is not dict
                        or set(before) != set(after)
                        or len(after) != 1
                    ):
                        raise _InputAdmissionHeldError()
                    if set(after) == {"text"}:
                        if type(after["text"]) is not str:
                            raise _InputAdmissionHeldError()
                    elif set(after) == {"json"}:
                        if type(after["json"]) is not type(before["json"]):
                            raise _InputAdmissionHeldError()
                    else:
                        raise _InputAdmissionHeldError()
                result = dict(result)
                result["content"] = replacement
                event.result = result
            if research_checked:
                with self._input_lock:
                    state = self._qualify_research_binding()
                    state["pending"] = False
        except Exception:
            self._input_quarantined = True
            # A producer's private accepted handle survives; its denied projection
            # cannot authorize a follow-up model cycle or warm observation.
            # The invocation hold is sticky across every tool callback. A later
            # readonly completion cannot reopen a denied producer or unknown input.
            self._input_hold = True
            unavailable = "Tool content is unavailable at the input admission boundary."
            event.result = {
                "toolUseId": event.tool_use["toolUseId"],
                "status": "error",
                "content": [{"text": unavailable}],
            }
            if not already_held:
                _warn_input_hold(
                    "AMLAS_INPUT_HOLD phase=research_after_result reason=callback_refused"
                    if name in {"research_fetch_urls", "research_search"}
                    else "AMLAS_INPUT_HOLD phase=tool_projection reason=callback_refused"
                )

    def _admit_initial_sft_ack(self, event, text):
        """Retire only a captured original's pool before its first ACK assessment."""
        witness = self._pending_batch_pause
        if witness is None or not witness.precommit:
            return False
        with self._input_lock:
            self._qualify_pending_pause(witness)
            with self._warm_capture_lock:
                selected = [
                    pending
                    for pending in self._warm_pending_producers.values()
                    if pending.kind == "sft"
                    and pending.agent is event.agent
                    and pending.tool_use_id == event.tool_use.get("toolUseId")
                ]
            if not selected:
                return False
            if len(selected) != 1 or event.retry is not False:
                raise _InputAdmissionHeldError()
            pending = selected[0]
            previous = witness.originals.get(pending.tool_use_id)
            if previous is not None and previous is not pending:
                raise _InputAdmissionHeldError()
            witness.originals[pending.tool_use_id] = pending
            self._admit_precommit_original(text, event, witness, pending, threading.Event())
            return True

    @property
    def _tool_registry(self) -> dict[str, Callable[..., Any]]:
        """Lazily build (and cache) the name→callable registry of bound tools.

        Returns an empty registry if the tool modules can't be imported (no
        ``strands``), so the backend stays constructible/usable for the plain-chat
        fallback path without crashing the TUI at launch.
        """
        if self._tool_registry_cache is None:
            try:
                self._tool_registry_cache = {
                    tool_display_name(t): _resolve_callable(t) for t in get_safe_llm_tools()
                }
            except Exception as exc:
                logger.warning("Tool registry unavailable (no strands?): %s", exc)
                self._tool_registry_cache = {}
        return self._tool_registry_cache

    # ─── Capability detection (graceful degradation) ──────────────────────────

    @property
    def is_tool_capable(self) -> bool:
        """True only when a REAL Strands package is importable (not the shim).

        The hermetic compatibility shim registers ``strands`` as a bare ``types.ModuleType``
        with no ``__file__``; an installed package always has one. We use that as the
        cheap, reliable marker that distinguishes the inert shim from real Strands.
        """
        try:
            import strands
        except Exception:
            return False
        return getattr(strands, "__file__", None) is not None

    def build_agent(
        self,
        thought_trace: ThoughtTrace | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Construct the tool-bound Strands ``Agent`` via the proven factory.

        Raises ``RuntimeError`` when not tool-capable (the shim path) so callers
        never accidentally invoke the inert shim Agent.

        Feature 2 (additive): when ``thought_trace`` is provided, a
        ``callback_handler`` mapping Strands streaming events → ``ThoughtEvent``s is
        attached to the built Agent so the ephemeral ticker sees live steps. The
        canonical ``_build.build_agent`` factory has a fixed signature (it forwards
        only the kwargs the installed ``Agent.__init__`` accepts,
        [`_build.py:27-60`](../../_build.py:27)), so we attach the handler
        DEFENSIVELY on the returned agent — never breaking the build if the installed
        Strands ``Agent`` doesn't expose ``callback_handler``. Default ``None`` keeps
        the build byte-identical for every existing caller/test.

        Multi-turn history (additive): when ``messages`` is provided it is the PRIOR
        conversation history (Strands ``Message`` dicts) replayed into the Agent's
        ``.messages`` at construction, so an affirmative like "yes to both" resolves
        against the assistant's previous question. ``None`` (the default) builds a
        fresh single-turn Agent exactly as before.
        """
        if not self.is_tool_capable:  # nosemgrep
            raise RuntimeError(
                "ToolBindingBackend.build_agent called without real Strands "
                "(install the 'agents' extra); use the plain-chat fallback instead."
            )

        from strands.agent.conversation_manager import SlidingWindowConversationManager

        from ..._build import build_agent
        from ...config.chat_model import chat_max_tokens, chat_model_id

        # TUI NL chat runs on the chat tier (Sonnet 4.6 by default, overridable
        # via AMLAS_CHAT_MODEL in .env.tui) — NOT the swarm's Opus 4.8 planner
        # tier. The explicit ``model_id`` wins over ``role`` in build_agent,
        # so the conversational TUI agent stays on Sonnet while the leads/sub-agents
        # keep Opus 4.8 for ML work (user directive 2026-06-04).
        #
        # Issue #35 criterion 4.1: the model id is resolved by
        # [`config.chat_model.chat_model_id`](../../config/chat_model.py:53) — the ONE
        # definition — instead of this call site re-reading ``AMLAS_CHAT_MODEL``
        # itself. It used to be resolved here, in the raw ``bedrock_backend``, and a
        # third time in ``engine._chat_model_label``, so the activity panel could
        # report a different tier than the agent was actually built on.
        #
        # ``max_tokens`` comes from the same source
        # ([`chat_max_tokens`](../../config/chat_model.py:65)), which defaults to
        # ``cfg.bedrock.max_tokens`` (64000) — the value this path already got from
        # the factory — so the consolidation does NOT re-cap chat to the deleted raw
        # path's hardcoded 2048 (criterion 4.2, design Property 7). Passing it
        # explicitly is what makes ``AMLAS_CHAT_MAX_TOKENS`` able to cap chat alone,
        # leaving the swarm tier on ``cfg.bedrock.max_tokens``.
        from .research_runtime_profile import _research_profile_prompt

        # The normal run keeps up to _NORMAL_RUN_CONVERSATION_WINDOW messages before the
        # SDK trims the oldest. With the SDK default (40), a trim after the fine-tune
        # pause edits a history that the input witness then refuses
        # (history_prefix_refused); one real run held 33 messages at the call after the pair
        # result. A new manager per build keeps the per-agent state separate; its saved
        # state keeps the same keys and class name as the default.
        agent = build_agent(
            name="tui_chat_agent",
            system_prompt=TUI_SYSTEM_PROMPT + _research_profile_prompt(self),
            tools=get_safe_llm_tools(),
            model=self._model,
            role="planner",
            model_id=chat_model_id(),
            max_tokens=chat_max_tokens(),
            description="Embedded TUI chat agent with real ML tool-calling.",
            messages=messages,
            conversation_manager=SlidingWindowConversationManager(
                window_size=_NORMAL_RUN_CONVERSATION_WINDOW
            ),
        )
        if thought_trace is not None:
            # Best-effort: set the streaming callback if the installed Agent supports
            # it. Non-fatal on version drift (the coarse execute_tool_use fallback
            # still guarantees tool-level ticker content).
            try:
                agent.callback_handler = _make_trace_callback(thought_trace)
            except Exception:  # pragma: no cover — version-drift defensive
                logger.debug("could not attach thought-trace callback_handler")
        elif _normal_run_console_silent():
            # A prompt run in the runtime stores no assistant text except its final report, so the
            # SDK's console printer (text deltas and "Tool #N: name") stays off its stdout. The
            # handler remains a callable: the guardrail capture wraps it at every invocation.
            from strands.handlers.callback_handler import null_callback_handler

            agent.callback_handler = null_callback_handler
        from strands.hooks.events import (
            AfterModelCallEvent,
            AfterToolCallEvent,
            BeforeInvocationEvent,
            BeforeModelCallEvent,
            BeforeToolCallEvent,
            MessageAddedEvent,
        )

        agent.hooks.add_callback(BeforeInvocationEvent, self._start_pending_pause)
        agent.hooks.add_callback(MessageAddedEvent, self._commit_pending_pause)

        input_admission_enabled = self._selected_input_checker() is not None
        input_controls_enabled = input_admission_enabled or self._research_binding is not None
        capture_planning = None

        def require_input_open(event):
            _warn_normal_callback_activity("before_model_enter")
            already_held = self._input_hold
            try:
                with self._input_lock:
                    state = self._qualify_research_binding()
                    if (
                        self._input_hold
                        or (input_admission_enabled and self._selected_input_checker() is None)
                        or (state is not None and state["pending"])
                    ):
                        raise _InputAdmissionHeldError()
            except Exception:
                self._input_hold = True
                if not already_held:
                    _warn_input_hold(
                        "AMLAS_INPUT_HOLD phase=before_model_qualification reason=callback_refused"
                    )
                raise _InputAdmissionHeldError() from None
            _warn_normal_callback_activity("before_model_exit")

        def after_model(event):
            _warn_normal_callback_activity("after_model_enter")

        agent.hooks.add_callback(AfterModelCallEvent, after_model)
        model_calls = 0

        def log_conversation_size(event):
            nonlocal model_calls
            model_calls += 1
            # Registered before the admission callbacks so a held call is still measured.
            with contextlib.suppress(Exception):
                from ...control_plane.normal_run_checkpoint import current_normal_run

                context = current_normal_run()
                if context is not None:
                    messages = _record_messages(event.agent.messages)
                    _warn_normal_conversation_size(
                        context.run_id, model_calls, _conversation_bytes(messages), len(messages)
                    )

        agent.hooks.add_callback(BeforeModelCallEvent, log_conversation_size)
        model_failures = 0

        async def retry_transient_model_failure(event):
            nonlocal model_failures
            error = getattr(event, "exception", None)
            if error is None:
                model_failures = 0
                return
            if getattr(event, "retry", True) or self._input_hold or model_failures >= _MODEL_CALL_RETRIES:
                return
            reason = _transient_model_error_reason(error)
            if reason is None:
                return
            model_failures += 1
            delay = _model_retry_delay(model_failures)
            _warn_model_call_retry(model_failures, reason, delay)
            await _model_retry_sleep(delay)
            if not self._input_hold:
                event.retry = True

        agent.hooks.add_callback(AfterModelCallEvent, retry_transient_model_failure)
        # Registered before the input-control callback so a refused message is cancelled first.
        agent.hooks.add_callback(BeforeToolCallEvent, self._refuse_mixed_sft_batch)
        if input_controls_enabled:
            agent.hooks.add_callback(BeforeModelCallEvent, require_input_open)

            def before_tool(event):
                from .research_input_recovery import _CANCELLED

                name = event.tool_use.get("name") if type(event.tool_use) is dict else None
                category = (
                    _NORMAL_CALLBACK_TOOL_CATEGORIES.get(name, "OTHER") if type(name) is str else "OTHER"
                )
                _warn_normal_callback_activity("before_tool_enter", tool_category=category)
                if event.cancel_tool and event.tool_use.get("toolUseId") in getattr(
                    self, "_refused_tool_results", {}
                ):
                    # The whole message was refused: no research bookkeeping, no SFT capture.
                    _warn_normal_callback_activity("before_tool_exit", tool_category=category)
                    return
                self._before_research_tool(event)
                if not self._input_hold:
                    phase = "current_before"
                    predicate = None
                    run_id = None
                    try:
                        from ...control_plane.normal_run_checkpoint import current_normal_run

                        context = current_normal_run()
                        if context is not None:
                            run_id = context.run_id
                    except Exception:
                        pass

                    def mark_phase(value):
                        nonlocal phase
                        phase = value

                    def mark_predicate(value):
                        nonlocal predicate
                        predicate = value

                    def mark_size(count, outcome):
                        _warn_normal_sft_capture_size(run_id, count, outcome)

                    try:
                        self._capture_normal_sft_call(
                            event,
                            _diagnostic_phase=mark_phase,
                            _diagnostic_predicate=mark_predicate,
                            _diagnostic_size=mark_size,
                        )
                    except Exception as error:
                        self._input_hold = True
                        event.cancel_tool = _CANCELLED
                        _warn_input_hold("AMLAS_INPUT_HOLD phase=sft_capture reason=callback_refused")
                        _warn_normal_sft_capture(run_id, phase, error)
                        _warn_normal_sft_input_predicate(run_id, phase, predicate)
                _warn_normal_callback_activity("before_tool_exit", tool_category=category)

            agent.hooks.add_callback(BeforeToolCallEvent, before_tool)
        if self._campaign_planning_store is not None:
            from strands.hooks.events import AfterToolCallEvent

            def capture(event):
                name = event.tool_use.get("name") if type(event.tool_use) is dict else None
                if name not in {"research_search", "research_fetch_urls"}:
                    return
                if (
                    type(event.invocation_state) is not dict
                    or event.invocation_state.get("_amlas_campaign_planning_store")
                    is not self._campaign_planning_store
                ):
                    raise ValueError("original SDK planning store differs")
                payload = self._planning_result_payload(event.result)
                self._capture_planning_result(name, payload)

            capture_planning = capture
        from strands.hooks.events import AfterToolCallEvent

        def capture_original_model(event):
            if self._warm_owner is None:
                return
            use = event.tool_use
            if type(use) is not dict or use.get("name") != "acquire_model_weights":
                return
            state = event.invocation_state
            if (
                type(state) is not dict
                or state.get("_amlas_model_acquisition_store") is not self._model_acquisition_store
            ):
                return
            payload = self._planning_result_payload(event.result)
            reference = payload.get("request_ref")
            if type(reference) is not str:
                return
            if not re.fullmatch(r"model_acq_[a-f0-9]{32}", reference):
                return
            store = self._model_acquisition_store
            with store._lock:
                record = store._records.get(reference)
            if (
                record is None
                or type(record.handle) is not JobHandle
                or type(record.handle.extra) is not dict
                or (payload.get("job_name"), payload.get("job_arn")) != (record.handle.id, record.handle.arn)
            ):
                return
            # Only the retained source-produced lifecycle selects read-only work.
            lifecycle = (
                record.handle.extra.get("accepted"),
                record.handle.extra.get("submission_ambiguous"),
            )
            if any(type(value) is not bool for value in lifecycle) or lifecycle not in (
                (True, False),
                (False, True),
            ):
                return
            identifier = use.get("toolUseId")
            if type(identifier) is not str or not 1 <= len(identifier) <= 256:
                return
            if record.handle.extra["bindings"]["runtime_session_id"] != store.runtime_session_id:
                return
            candidate = _PendingOriginalModel(
                identifier,
                reference,
                store,
                record,
                agent,
                self._warm_graph(),
                self._warm_handle_pin(record.handle),
                os.getpid(),
            )
            with self._warm_capture_lock:
                key = (id(store), reference, candidate.handle_pin)
                if key in self._warm_consumed or payload.get("acquired") is True:
                    return
                if self._warm_pending is not None:
                    self._warm_closed = "refused"
                    return
                self._warm_pending = candidate

        def capture_original_producer(event):
            use = event.tool_use
            if (
                self._warm_owner is None
                or type(use) is not dict
                or use.get("name") not in _ORIGINAL_PRODUCERS
            ):
                return
            kind, attribute, state_key, ref_key, prefix, _action = _ORIGINAL_PRODUCERS[use["name"]]
            store = getattr(self, attribute)
            if type(event.invocation_state) is not dict or event.invocation_state.get(state_key) is not store:
                return
            payload = self._planning_result_payload(event.result)
            reference, identifier = payload.get(ref_key), use.get("toolUseId")
            if (
                type(reference) is not str
                or not re.fullmatch(prefix + "[a-f0-9]{32}", reference)
                or type(identifier) is not str
                or not 1 <= len(identifier) <= 256
            ):
                return
            with store._lock:
                record = store._records.get(reference)
            if record is None:
                return
            handle = record.handle if kind in {"data", "preparation"} else record
            adapter = record.adapter if kind in {"data", "preparation"} else store._adapter
            if type(handle) is not JobHandle or type(handle.extra) is not dict:
                return
            if (payload.get("job_name"), payload.get("job_arn")) != (handle.id, handle.arn):
                return
            if kind == "data":
                if handle.extra.get("accepted") is not True:
                    return
            elif kind != "preparation" and handle.extra.get("submission_state") not in {
                "accepted",
                "ambiguous",
            }:
                return
            if handle.extra.get("runtime_session_id") != store.runtime_session_id:
                return
            if kind in {"sft", "reload"} and handle.extra.get("operation") != (
                "train" if kind == "sft" else "reload"
            ):
                return
            candidate = _PendingOriginalProducer(
                kind,
                identifier,
                reference,
                store,
                record,
                adapter,
                agent,
                self._warm_graph(),
                self._producer_handle_pin(kind, handle),
                os.getpid(),
            )
            key = self._producer_key(candidate)
            with self._warm_capture_lock:
                rejected_key = ("rejected_original", kind, id(store), reference)
                if (
                    key not in self._warm_consumed
                    and rejected_key not in self._warm_consumed
                    and self._warm_closed is None
                ):
                    self._warm_pending_producers.setdefault(key, candidate)

        def capture_then_admit(event, *, mark=True):
            # SDK AfterToolCall callbacks reverse registration order. Keep the
            # private custody and public admission steps in one explicit sequence.
            name = event.tool_use.get("name") if type(event.tool_use) is dict else None
            category = _NORMAL_CALLBACK_TOOL_CATEGORIES.get(name, "OTHER") if type(name) is str else "OTHER"
            if mark:
                _warn_normal_callback_activity("after_tool_enter", tool_category=category)
            producer = self._is_original_producer_use(event.tool_use)
            if producer:
                try:
                    capture_original_model(event)
                    capture_original_producer(event)
                except Exception:
                    # A native effect may already exist. Retain its store record
                    # and stop before any new model cycle; never replay capture.
                    already_held = self._input_hold
                    self._input_hold = True
                    if not already_held:
                        _warn_input_hold("AMLAS_INPUT_HOLD phase=producer_capture reason=callback_refused")
                    self._guard_new_tool_projection(event, producer=True)
                    if mark:
                        _warn_normal_callback_activity("after_tool_exit", tool_category=category)
                    return
                if input_controls_enabled:
                    self._guard_new_tool_projection(event, producer=True)
            else:
                if input_controls_enabled:
                    self._guard_new_tool_projection(event, producer=False)
                if (
                    capture_planning is not None
                    and not self._input_hold
                    and not self._research_withheld_result(event)
                ):
                    capture_planning(event)
            if mark and not self._input_hold:
                try:
                    self._mark_pending_pause(event)
                except Exception:
                    already_held = self._input_hold
                    self._input_hold = True
                    if not already_held:
                        _warn_input_hold("AMLAS_INPUT_HOLD phase=pending_bookkeeping reason=callback_refused")
            if mark:
                _warn_normal_callback_activity("after_tool_exit", tool_category=category)

        async def complete_then_admit(event):
            witness = self._pending_batch_pause
            name = event.tool_use.get("name") if type(event.tool_use) is dict else None
            category = _NORMAL_CALLBACK_TOOL_CATEGORIES.get(name, "OTHER") if type(name) is str else "OTHER"
            _warn_normal_callback_activity("after_tool_enter", tool_category=category)
            producer = self._is_original_producer_use(event.tool_use)
            if witness is not None and witness.precommit and producer:
                identifier = event.tool_use.get("toolUseId")
                if identifier in witness.handled:
                    already_held = self._input_hold
                    self._input_hold = True
                    if not already_held:
                        _warn_input_hold(
                            "AMLAS_INPUT_HOLD phase=pending_bookkeeping reason=duplicate_original"
                        )
                    self._guard_new_tool_projection(event, producer=True)
                    _warn_normal_callback_activity("after_tool_exit", tool_category=category)
                    return
                witness.handled.add(identifier)
            capture_then_admit(event, mark=False)
            if not self._input_hold:
                try:
                    if witness is not None and witness.precommit and producer:
                        await self._complete_original_before_commit(event, witness)
                    self._mark_pending_pause(event)
                except Exception:
                    already_held = self._input_hold
                    self._input_hold = True
                    if not already_held:
                        _warn_input_hold("AMLAS_INPUT_HOLD phase=pending_bookkeeping reason=callback_refused")
                    self._guard_new_tool_projection(event, producer=producer)
            _warn_normal_callback_activity("after_tool_exit", tool_category=category)

        agent.hooks.add_callback(
            AfterToolCallEvent,
            complete_then_admit if self._runtime_original_completion is not None else capture_then_admit,
        )
        return agent

    # ─── Confirmation gate + real tool dispatch (shim-safe seam) ──────────────

    def confirmation_prompt(self, tool_name: str, args: dict[str, Any]) -> str:
        """Human-readable y/n confirmation line for a state-changing tool."""
        arg_hint = ""
        if args:
            shown = ", ".join(
                f"{k}={v}" for k, v in list(args.items())[:3] if not isinstance(v, (list, dict))
            )
            if shown:
                arg_hint = f" ({shown})"
        return f"Run `{tool_name}`{arg_hint}? [y/n]"

    def _emit_trace(self, kind: str, text: str, tool: str | None = None) -> None:
        """Coarse ticker emission for the ephemeral thought trace (Feature 2).

        A no-op unless a per-turn ``ThoughtTrace`` is active (set by
        ``chat_with_tools`` for the current turn). This is the offline-safe fallback
        that guarantees tool-level ticker content even when the granular Strands
        streaming callback shape is unavailable (design §3.2b). Never raises — the
        ``ThoughtTrace.emit`` sink discipline already swallows sink errors.
        """
        trace = self._thought_trace
        if trace is None:
            return
        try:
            trace.emit(ThoughtEvent(kind=kind, text=text, tool=tool))
        except Exception:  # pragma: no cover — defensive; emit never raises anyway  # nosec B110
            pass

    @staticmethod
    def _planning_result_payload(result):
        import json
        import math

        limit = 65536
        nodes = [0]

        def bounded(value, depth=0):
            nodes[0] += 1
            if nodes[0] > 2048 or depth > 12:
                raise ValueError("bounded returned research DATA required")
            if value is None or type(value) in (bool, int):
                if type(value) is int and value.bit_length() > 256:
                    raise ValueError("bounded returned research number required")
                return
            if type(value) is float:
                if not math.isfinite(value):
                    raise ValueError("finite returned research DATA required")
                return
            if type(value) is str:
                if len(value) > limit:
                    raise ValueError("returned research text exceeds bound")
                return
            if type(value) is list:
                if len(value) > 128:
                    raise ValueError("returned research population exceeds bound")
                for item in value:
                    bounded(item, depth + 1)
                return
            if type(value) is dict:
                if len(value) > 128 or any(type(key) is not str for key in value):
                    raise ValueError("closed returned research keys required")
                for key, item in value.items():
                    bounded(key, depth + 1)
                    bounded(item, depth + 1)
                return
            raise ValueError("returned research DATA type unsupported")

        bounded(result)
        total = 0
        for chunk in json.JSONEncoder(allow_nan=False, ensure_ascii=False, separators=(",", ":")).iterencode(
            result
        ):
            total += len(chunk.encode("utf-8"))
            if total > limit:
                raise ValueError("returned research ToolResult exceeds byte bound")
        if type(result) is not dict or type(result.get("content")) is not list:
            raise ValueError("actual SDK returned research content required")
        for block in result["content"]:
            if type(block) is not dict:
                raise ValueError("actual SDK content block required")
            if "json" in block and type(block["json"]) is dict:
                return block["json"]
            if "text" in block and type(block["text"]) is str:
                try:
                    value = json.loads(
                        block["text"],
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            ValueError("nonfinite returned research DATA")
                        ),
                    )
                except (ValueError, RecursionError):
                    continue
                if type(value) is dict:
                    bounded(value)
                    return value
        raise ValueError("bounded actual research result object required")

    def _capture_planning_result(self, name, result):
        store = self._campaign_planning_store
        if store is None:
            return
        from ...control_plane.campaign_planning import CampaignPlanningStore
        from ...control_plane.campaign_runtime import current_campaign_runtime

        if type(store) is not CampaignPlanningStore:
            raise ValueError("exact original planning result store required")
        runtime = current_campaign_runtime()
        if runtime is not None and runtime is not store.entry._factory:
            raise ValueError("foreign planning result factory refused")
        if self._campaign_planning_services is not None:
            observed = self._campaign_planning_services.observed_at(store.entry)
        else:
            import time

            observed = time.time()
        store.observe_tool_result(name, result, observed_at=observed)

    def _invocation_state(self) -> dict[str, Any]:
        """Fresh caller-owned state; conversation stores retain records, never grants."""
        selected_policy = self.approval_policy
        if selected_policy is not None and not isinstance(selected_policy, ApprovalPolicy):
            raise TypeError("approval_policy must be a caller-owned ApprovalPolicy")
        policy = (
            replace(selected_policy, allowlist=list(selected_policy.allowlist))
            if selected_policy is not None
            else ApprovalPolicy()
        )
        state = {
            "_amlas_approval_policy": policy,
            "_amlas_acquisition_store": self._acquisition_store,
            "_amlas_case_freeze_store": self._case_freeze_store,
            "_amlas_preparation_store": self._preparation_store,
            "_amlas_model_acquisition_store": self._model_acquisition_store,
            "_amlas_sft_store": self._sft_store,
            "_amlas_generation_store": self._generation_store,
            "_amlas_pair_store": self._pair_store,
        }
        from ...control_plane.campaign_runtime import current_campaign_entry, current_campaign_runtime

        if (
            self._campaign_planning_store is not None
            or current_campaign_entry() is not None
            or current_campaign_runtime() is not None
        ):
            if self._campaign_planning_services is not None:
                self._campaign_planning_services.require_current_selection(
                    self._campaign_planning_store.entry
                )
            state["_amlas_campaign_planning_store"] = self._campaign_planning_store
        return state

    def _invoke_direct_tool(self, tool_name: str, fn: Callable[..., Any], args: dict[str, Any]) -> Any:
        if tool_name in {"acquire_model_weights", "get_model_weight_acquisition"}:
            self._model_acquisition_store._require_invocation_context()
        if tool_name in {
            "acquire_huggingface_dataset",
            "get_dataset_acquisition",
            "list_coding_preparation_profiles",
            "save_spec",
            "acquire_model_weights",
            "get_model_weight_acquisition",
            "resolve_coding_evaluation_spec",
            "submit_coding_pair",
            "get_coding_pair",
            "bind_coding_generation_spec",
            "submit_coding_generation",
            "get_coding_generation",
            "submit_qwen_sft",
            "get_qwen_sft",
            "reload_qwen_sft",
            "freeze_coding_cases",
            "get_coding_case_freeze",
        }:
            if "tool_context" in args or any(key.startswith("_amlas_") for key in args):
                raise ValueError("Caller-authored private SFT context is refused")
            context = SimpleNamespace(invocation_state=self._invocation_state())
            return fn(**args, tool_context=context)
        result = fn(**args)
        if tool_name in {"research_search", "research_fetch_urls"}:
            self._capture_planning_result(tool_name, result)
        return result

    def execute_tool_use(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        confirmed: bool = False,
    ) -> ToolExecution:
        """Dispatch one tool_use: gate by class FIRST, then invoke the REAL tool.

        Order matters (and is the correct design): the confirmation gate is a
        pure NAME-classification decision, so it fires BEFORE callable resolution.
        A state-changing tool must surface ``needs_confirmation`` even before we
        bother resolving its callable — and this keeps the gate robust when a test
        patches the underlying tool symbol (which would change the registry key).

        - State-changing tool without ``confirmed`` → ``needs_confirmation`` (the
          real tool is NOT run, callable not even resolved).
        - Read-only tool, or confirmed state-changing tool → resolve + execute.
        - Unknown/unresolvable tool (once past the gate) → ``error`` (never raises).

        Feature 2 (additive, no behavior change): when a per-turn ``ThoughtTrace`` is
        active, coarse ``tool_start``/``tool_end`` ticker events are emitted around a
        successful execution (design §3.2b). With no trace (the default / every
        existing caller), ``_emit_trace`` is a no-op and this method is byte-identical.
        """
        is_read_only = tool_name in self._readonly_names
        if not is_read_only and not confirmed:
            return ToolExecution(
                tool_name=tool_name,
                args=args,
                status="needs_confirmation",
                result=None,
                result_summary=self.confirmation_prompt(tool_name, args),
            )

        fn = self._tool_registry.get(tool_name)
        if fn is None:
            return ToolExecution(
                tool_name=tool_name,
                args=args,
                status="error",
                result=None,
                result_summary=f"Unknown tool: {tool_name}",
            )

        # Coarse ticker: announce the tool is starting (the gate has passed).
        self._emit_trace("tool_start", f"running {tool_name}", tool=tool_name)

        # DP-2 (Phase 11): a CONFIRMED long-running tool is handed to the
        # JobManager so the agent turn completes fast and the work runs in the
        # background (design §2.2/§3). The confirmation gate has ALREADY passed
        # above (LONG_RUNNING_TOOLS ⊆ STATE_CHANGING_TOOLS), so backgrounding
        # layers cleanly on top. Only when a JobManager is wired (the app); when
        # None (validation coverage) the tool executes inline below, unchanged.
        if self.job_manager is not None and tool_name in LONG_RUNNING_TOOLS:
            job = self.job_manager.submit(
                tool_name,
                lambda progress_cb, cancel_token, _fn=fn, _args=args: _fn(**_args),
                args=dict(args),
            )
            self._emit_trace("tool_end", f"{tool_name} submitted", tool=tool_name)
            return ToolExecution(
                tool_name=tool_name,
                args=args,
                status="job_submitted",
                result={"job_id": job.id},
                result_summary=f"{tool_name} submitted as background job {job.id}",
            )

        try:
            result = self._invoke_direct_tool(tool_name, fn, args)
        except Exception as exc:
            logger.warning("Tool %s raised: %s", tool_name, exc)
            self._emit_trace("tool_end", f"{tool_name} failed", tool=tool_name)
            return ToolExecution(
                tool_name=tool_name,
                args=args,
                status="error",
                result=None,
                result_summary=f"{tool_name} failed: {exc}",
            )

        self._emit_trace("tool_end", f"{tool_name} done", tool=tool_name)
        return ToolExecution(
            tool_name=tool_name,
            args=args,
            status="executed",
            result=result,
            result_summary=_summarize(result),
        )

    # ─── Full LLM tool-calling turn (live path / injected model in tests) ─────

    def chat_with_tools(self, message, context="", *, thought_trace=None, history=None):
        if self._warm_owner is not None and self._warm_pid != os.getpid():
            raise ValueError("inherited warm owner refused")
        with self._warm_lock:
            with self._warm_capture_lock:
                if self._warm_closed is not None:
                    return AgentTurn(final_text="The original model continuation is closed.")
                if self._warm_pending is not None:
                    return AgentTurn(final_text="The original model acquisition is still pending.")
            return self._chat_with_tools_once(message, context, thought_trace=thought_trace, history=history)

    def _chat_with_tools_once(
        self,
        message: str,
        context: str = "",
        *,
        thought_trace: ThoughtTrace | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentTurn | None:
        """Run a full tool-bound agent turn; return an ``AgentTurn`` or ``None``.

        Returns ``None`` when not tool-capable, which sends ``ChatEngine`` to its
        DETERMINISTIC tier-5 keyword fallback — the raw-converse brain that used to
        sit between them was deleted by issue #35 (6.1/6.2), so ``None`` now means
        "no LLM answers this turn", not "a different LLM answers it". On the live
        path, builds the agent, invokes it, and parses the final assistant text
        (+ any tool trace) into an ``AgentTurn``.

        Feature 2 (additive): when ``thought_trace`` is provided, it is (a) attached
        as the Strands ``callback_handler`` for granular streaming events AND (b) set
        as the per-turn coarse trace so ``execute_tool_use`` emits tool-level ticker
        content offline. The coarse trace is ALWAYS cleared in a ``finally`` so it
        never leaks into a later turn. Default ``None`` keeps the turn byte-identical
        for every existing caller/test.

        Multi-turn history (additive, the stateful-chat fix): ``history`` is the
        PRIOR conversation (Strands ``Message`` dicts, user+assistant, already
        bounded by the engine) replayed into the freshly-built Agent's ``.messages``
        so the model sees the earlier turns — e.g. an affirmative "yes to both"
        resolves against the two questions the assistant asked last turn. The CURRENT
        ``message`` is still the prompt (never duplicated into ``history``). ``None``
        (the default) preserves the exact single-turn build for every existing caller.
        """
        if not self.is_tool_capable:  # nosemgrep
            return None

        self._input_cycle = secrets.token_hex(16)
        self._input_receipts = []
        self._input_quarantined = False
        try:
            admitted_message = self._admit_input(message, "OWNER_PROMPT")
            self._activate_research_binding()
            admitted_context = self._admit_input(context, "GENERATED_CONTEXT") if context else ""
            try:
                check_history = self._selected_input_checker() is not None
            except Exception:
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold(
                        "AMLAS_INPUT_HOLD phase=history_qualification reason=checker_selection_failed"
                    )
                raise _InputAdmissionHeldError() from None
            if check_history:
                for prior in history or []:
                    # Assess every value of the original message: content in the form of its first
                    # check, machine-written values singly, any other shape as whole JSON.
                    # Admission never rewrites a historical or signed content block.
                    try:
                        prior_texts = _history_admission_texts(prior)
                    except (TypeError, ValueError):
                        already_held = self._input_hold
                        self._input_hold = True
                        if not already_held:
                            _warn_input_hold(
                                "AMLAS_INPUT_HOLD phase=history_qualification reason=history_not_serializable"
                            )
                        raise _InputAdmissionHeldError() from None
                    for prior_text in prior_texts:
                        self._admit_input(prior_text, "HISTORY_TEXT", immutable=True)
            prompt = (
                admitted_message
                if not admitted_context
                else (f"{admitted_message}\n\n## Context\n{admitted_context}")
            )
            if admitted_context:
                prompt = self._admit_input(prompt, "GENERATED_CONTEXT", immutable=True)
        except _InputAdmissionHeldError:
            return self._input_hold_turn()
        # Publish only the admitted prompt/history before the first model/effect.
        from ...control_plane.normal_run_checkpoint import current_normal_run

        normal_run = current_normal_run()
        if normal_run is not None:
            normal_run.enroll(self._acquisition_store.runtime_session_id, prompt, history)

            def qualify_normal_parent():
                from ...control_plane.producer_snapshot import _approved_read

                self._qualify_pending_owner(self._pending_batch_pause)
                _approved_read(self.approval_policy)

            def qualify_normal_parent_originals():
                from ...control_plane.producer_snapshot import _approved_read

                self._qualify_pending_pause(self._pending_batch_pause)
                _approved_read(self.approval_policy)

            normal_run.bind_sft_store(
                self._sft_store,
                current_qualification=qualify_normal_parent,
                original_qualification=qualify_normal_parent_originals,
                approval_policy_getter=lambda: self.approval_policy,
            )
        # Activate the coarse fallback trace for THIS turn only (cleared in finally).
        self._thought_trace = thought_trace
        capture = GuardrailAssessmentCapture()
        agent = previous_callback = composed_callback = None
        try:
            try:
                agent = self.build_agent(thought_trace=thought_trace, messages=history)
                previous_callback = getattr(agent, "callback_handler", None)
                if callable(previous_callback):
                    composed_callback = capture.wrap(previous_callback)
                    agent.callback_handler = composed_callback
                result = agent(prompt, invocation_state=self._invocation_state())
            except _InputAdmissionHeldError:
                return self._input_hold_turn()
            except Exception as exc:
                # The SDK may wrap a hook failure. The source-owned hold latch
                # remains authoritative; never infer a guard decision from text.
                if self._input_hold:
                    return self._input_hold_turn(guardrail_assessment=capture.close())
                with self._warm_capture_lock:
                    if self._warm_owner is not None and self._warm_pending is not None:
                        self._warm_closed = self._warm_closed or "refused"
                logger.error("Tool-bound agent error: %s", exc)
                # Surface an expired-SSO / missing-creds failure as a clear, actionable
                # re-auth message (the shared CREDS-01 diagnostic) instead of a cryptic
                # botocore trace; otherwise keep the raw tool-agent error.
                from ...config.bedrock_creds import diagnose_aws_error

                creds_msg = diagnose_aws_error(exc, self.settings.aws.effective_profile)
                if creds_msg is not None:
                    return AgentTurn(final_text=creds_msg, executions=[])
                return AgentTurn(final_text=f"⚠️ Tool-agent error: {exc}", executions=[])

            try:
                with self._input_lock:
                    self._qualify_research_binding()
            except Exception:
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold("AMLAS_INPUT_HOLD phase=research_after_run reason=binding_refused")
            if self._input_hold:
                return self._input_hold_turn(guardrail_assessment=capture.close())
            try:
                paused = self._pending_pause_result(agent, result)
            except Exception:
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold("AMLAS_INPUT_HOLD phase=pause_result reason=pause_result_refused")
                return self._input_hold_turn(guardrail_assessment=capture.close())
            if paused:
                return AgentTurn(
                    final_text="Waiting for original native operations to complete.",
                    executions=self._extract_qualified_normal_executions(result, agent),
                    input_admission=self._input_admission_data(),
                )
            if (
                self._warm_owner is not None
                and self._warm_pending is not None
                and self._extract_guardrail_outcome(result) is not None
            ):
                self._warm_closed = "guardrail_intervened"
            return AgentTurn(
                final_text=self._extract_final_text(result),
                executions=self._extract_qualified_normal_executions(result, agent),
                guardrail_outcome=self._extract_guardrail_outcome(result),
                guardrail_assessment=capture.close(),
                input_admission=self._input_admission_data(),
            )
        finally:
            capture.close()
            if (
                agent is not None
                and composed_callback is not None
                and getattr(agent, "callback_handler", None) is composed_callback
            ):
                agent.callback_handler = previous_callback
            self._pending_batch_pause = None
            # The per-turn trace must never bleed into a subsequent turn.
            self._thought_trace = None

    @staticmethod
    def _pause_history(messages):
        """Witness text of one message or a message list, in record form (no SDK tracking_id)."""
        messages = _record_messages([messages])[0] if type(messages) is dict else _record_messages(messages)
        return json.dumps(messages, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    @staticmethod
    def _pause_result(result):
        """Witness text of one tool result, exactly as given."""
        return json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    def _pause_profile_current(self):
        from .research_runtime_profile import _ENV, _PROFILE

        if os.environ.get(_ENV) != _PROFILE or self._research_binding is None:
            return None
        return self._qualify_research_binding()

    def _runtime_research_settled(self, state):
        """Only the exact deployment-bound, exhausted research stage can select it."""
        binding = self._runtime_original_completion
        if binding is None:
            return False
        if (
            type(binding) is not tuple
            or len(binding) != 3
            or binding[0] is not state
            or binding[1] is not state["qualify"]
            or binding[2] is not state["plan"]
            or set(state["units"]) != {unit.unit_id for unit in binding[2].units}
            or any(state["units"].get(unit.unit_id) is not unit for unit in binding[2].units)
        ):
            raise _InputAdmissionHeldError()
        binding[1]()
        units = set(state["units"])
        return (
            state["pending"] is False
            and state["attempted"] == units
            and set(state["tool_uses"].values()) == units
            and len(state["tool_uses"]) == len(units)
            and state["results"] == set(state["tool_uses"])
        )

    def _qualify_pending_owner(self, witness):
        """Current owner/witness checks only; performs no original graph walk."""
        with self._input_lock:
            if (
                type(witness) is not _PendingBatchPause
                or witness is not self._pending_batch_pause
                or witness.pid != os.getpid()
                or witness.research is not self._pause_profile_current()
                or self._input_hold
                or self._warm_closed is not None
                or witness.checker is not self._selected_input_checker()
                or witness.invocation_state.get("request_state") is not witness.request_state
            ):
                raise _InputAdmissionHeldError()
            if witness.precommit:
                binding = self._runtime_original_completion
                if (
                    type(binding) is not tuple
                    or len(binding) != 3
                    or binding[0] is not witness.research
                    or binding[1] is not witness.research["qualify"]
                    or not self._runtime_research_settled(witness.research)
                    or witness.research["original_completion_ready"] is not True
                    or any(
                        self._tool_registry.get(name) is not original
                        for name, original in witness.registry.items()
                    )
                ):
                    raise _InputAdmissionHeldError()
                binding[1]()

    def _qualify_pending_pause(self, witness):
        self._qualify_pending_owner(witness)
        for pending in tuple(witness.originals.values()):
            if type(pending) is _PendingOriginalModel:
                self._qualify_warm_original(pending)
            elif type(pending) is _PendingOriginalProducer:
                self._qualify_warm_producer(pending)
            else:
                raise _InputAdmissionHeldError()

    def _start_pending_pause(self, event):
        if self._pause_profile_current() is None:
            self._pending_batch_pause = None
            return
        state = event.invocation_state
        if type(state) is not dict or "request_state" in state:
            already_held = self._input_hold
            self._input_hold = True
            if not already_held:
                _warn_input_hold("AMLAS_INPUT_HOLD phase=pause_start reason=invocation_state_refused")
            raise _InputAdmissionHeldError()
        incoming = event.messages
        if incoming is not None and type(incoming) is not list:
            already_held = self._input_hold
            self._input_hold = True
            if not already_held:
                _warn_input_hold("AMLAS_INPUT_HOLD phase=pause_start reason=incoming_messages_refused")
            raise _InputAdmissionHeldError()
        request = {}
        state["request_state"] = request
        self._pending_batch_pause = _PendingBatchPause(
            event.agent,
            state,
            request,
            self._research_binding,
            self._selected_input_checker(),
            os.getpid(),
            self._pause_history(event.agent.messages),
            tuple(self._pause_history(message) for message in (incoming if incoming is not None else [])),
            precommit=self._research_binding["original_completion_ready"] is True,
            registry={
                name: self._tool_registry.get(name)
                for name in (*_ORIGINAL_PRODUCERS, "acquire_model_weights")
            },
        )

    def _mark_pending_pause(self, event):
        witness = self._pending_batch_pause
        if witness is None:
            return
        self._qualify_pending_pause(witness)
        if event.agent is not witness.agent or event.invocation_state is not witness.invocation_state:
            raise _InputAdmissionHeldError()
        use, result = event.tool_use, event.result
        identifier = use.get("toolUseId")
        if (
            type(use) is not dict
            or use.get("name") not in self._tool_registry
            or type(identifier) is not str
            or type(result) is not dict
            or result.get("toolUseId") != identifier
            or (
                result.get("status") != "success"
                and not self._research_withheld_result(event)
                and not self._refused_batch_result(event)
            )
        ):
            return
        witness.admitted[identifier] = self._pause_result(result)
        with self._warm_capture_lock:
            candidates = ([self._warm_pending] if self._warm_pending is not None else []) + list(
                self._warm_pending_producers.values()
            )
        for pending in candidates:
            if pending.agent is event.agent and pending.tool_use_id == identifier:
                witness.originals[identifier] = pending
        if not witness.precommit and witness.originals:
            self._qualify_pending_pause(witness)
            witness.request_state["stop_event_loop"] = True
            witness.marked = True

    def _commit_pending_pause(self, event):
        witness = self._pending_batch_pause
        if witness is None:
            return
        try:
            self._qualify_pending_pause(witness)
        except Exception:
            already_held = self._input_hold
            self._input_hold = True
            if not already_held:
                _warn_input_hold("AMLAS_INPUT_HOLD phase=history_qualification reason=callback_refused")
            raise _InputAdmissionHeldError() from None
        messages = event.agent.messages
        if event.agent is not witness.agent or not messages or messages[-1] is not event.message:
            raise _InputAdmissionHeldError()
        content = event.message.get("content", [])
        results = [block["toolResult"] for block in content if "toolResult" in block]
        uses = [block["toolUse"] for block in content if "toolUse" in block]
        if witness.incoming:
            if (
                self._pause_history(messages[:-1]) != witness.prefix
                or self._pause_history(event.message) != witness.incoming[0]
            ):
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold(
                        "AMLAS_INPUT_HOLD phase=history_qualification reason=history_prefix_refused"
                    )
                raise _InputAdmissionHeldError()
            witness.incoming = witness.incoming[1:]
            witness.prefix = self._pause_history(messages)
            return
        if uses:
            if witness.batch is not None or self._pause_history(messages[:-1]) != witness.prefix:
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold(
                        "AMLAS_INPUT_HOLD phase=history_qualification reason=history_prefix_refused"
                    )
                raise _InputAdmissionHeldError()
            witness.batch = self._pause_history(event.message)
            return
        if not results:
            if witness.batch is not None or self._pause_history(messages[:-1]) != witness.prefix:
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold(
                        "AMLAS_INPUT_HOLD phase=history_qualification reason=history_prefix_refused"
                    )
                raise _InputAdmissionHeldError()
            witness.prefix = self._pause_history(messages)
            return
        batch_uses = [block["toolUse"] for block in messages[-2]["content"] if "toolUse" in block]
        if (
            self._pause_history(messages[-2]) != witness.batch
            or len(results) != len(batch_uses)
            or len({value["toolUseId"] for value in results}) != len(results)
            or {value["toolUseId"] for value in batch_uses} != {value["toolUseId"] for value in results}
            or any(witness.admitted.get(value["toolUseId"]) != self._pause_result(value) for value in results)
            or self._pause_history(messages[:-2]) != witness.prefix
        ):
            already_held = self._input_hold
            self._input_hold = True
            if not already_held:
                _warn_input_hold("AMLAS_INPUT_HOLD phase=history_qualification reason=history_batch_refused")
            raise _InputAdmissionHeldError()
        had_completed = False
        if witness.completed:
            had_completed = True
            self._qualify_pending_pause(witness)
            with self._warm_capture_lock:
                for identifier, pending in witness.completed.items():
                    if witness.originals.get(identifier) is not pending:
                        already_held = self._input_hold
                        self._input_hold = True
                        if not already_held:
                            _warn_input_hold(
                                "AMLAS_INPUT_HOLD phase=history_qualification reason=completed_original_changed"
                            )
                        raise _InputAdmissionHeldError()
                    if type(pending) is _PendingOriginalModel:
                        if self._warm_pending is not pending:
                            raise _InputAdmissionHeldError()
                    elif self._warm_pending_producers.get(self._producer_key(pending)) is not pending:
                        raise _InputAdmissionHeldError()
                for identifier, pending in witness.completed.items():
                    if type(pending) is _PendingOriginalModel:
                        self._warm_consumed.add((id(pending.store), pending.reference, pending.handle_pin))
                        self._warm_pending = None
                    else:
                        key = self._producer_key(pending)
                        self._warm_consumed.add(key)
                        self._warm_pending_producers.pop(key, None)
                        self._warm_case_failure_counts.pop(key, None)
                    witness.originals.pop(identifier)
                witness.completed.clear()
        if witness.marked:
            witness.history = self._pause_history(messages)
            witness.settled = True
        else:
            witness.prefix = self._pause_history(messages)
            witness.batch = None
            witness.admitted.clear()
        if had_completed:
            _warn_precommit_handoff("BATCH_COMMITTED", None)

    async def _wait_precommit_original(self):
        """Existing source cadence; tests may replace only this lexical wait."""
        await asyncio.sleep(15.0)

    def _qualify_precommit_original(self, event, witness, pending, closed):
        if closed.is_set():
            raise _InputAdmissionHeldError()
        self._qualify_pending_pause(witness)
        if (
            event.agent is not witness.agent
            or event.invocation_state is not witness.invocation_state
            or witness.originals.get(pending.tool_use_id) is not pending
            or witness.batch is None
            or self._pause_history(event.agent.messages[-1]) != witness.batch
            or self._pause_history(event.agent.messages[:-1]) != witness.prefix
        ):
            raise _InputAdmissionHeldError()
        uses = [block["toolUse"] for block in event.agent.messages[-1]["content"] if "toolUse" in block]
        if sum(use == event.tool_use for use in uses) != 1:
            raise _InputAdmissionHeldError()
        if type(pending) is _PendingOriginalModel:
            self._qualify_warm_original(pending)
        else:
            self._qualify_warm_producer(pending)
        if closed.is_set() or self._input_hold:
            raise _InputAdmissionHeldError()

    def _note_precommit_unresolved(self, pending):
        """Count one unresolved observation; close the wait once the run spans the wall-clock cap."""
        now = _precommit_unknown_clock()
        with self._warm_capture_lock:
            started, polls = self._precommit_unresolved.get(pending.tool_use_id, (now, 0))
            polls += 1
            self._precommit_unresolved[pending.tool_use_id] = (started, polls)
            elapsed = now - started
            exhausted = elapsed >= _PRECOMMIT_UNKNOWN_WALL_CLOCK_SECONDS
            if exhausted and self._warm_closed is None:
                self._warm_closed = "submission_unresolved"
                self._warm_closed_diagnostic = _WarmRefusalDiagnostic(
                    "native_observation", "original_unknown_budget_exhausted", "UNKNOWN"
                )
        if exhausted or polls == 1 or polls % _PRECOMMIT_UNKNOWN_LOG_EVERY == 0:
            _warn_normal_unknown_poll(
                _UNKNOWN_POLL_OPERATIONS.get(pending.kind, "OTHER"),
                polls,
                int(elapsed),
                "closed" if exhausted else "waiting",
            )
        if exhausted:
            raise _InputAdmissionHeldError()

    def _clear_precommit_unresolved(self, pending):
        with self._warm_capture_lock:
            self._precommit_unresolved.pop(pending.tool_use_id, None)

    def _read_precommit_original(self, event, witness, pending, closed):
        # The adapter's ContextVar scope must be created inside this copied-context
        # thread, never around the await. This read never reenters Agent or Create.
        def qualify():
            self._qualify_precommit_original(event, witness, pending, closed)

        qualify()
        _warn_precommit_handoff("OBSERVING", pending)
        adapter = pending.record.adapter if type(pending) is _PendingOriginalModel else pending.adapter
        with adapter.qualified_io(qualify):
            observed = pending.store.observe(pending.reference)
        qualify()
        handle = (
            pending.record.handle
            if type(pending) is _PendingOriginalModel
            else self._producer_handle(pending)
        )
        ref_key = (
            "case_ref"
            if type(pending) is _PendingOriginalProducer and pending.kind == "case"
            else "request_ref"
        )
        status = observed.get("status")
        _warn_precommit_handoff("OBSERVED", pending, status=status)
        if type(pending) is _PendingOriginalModel:
            if observed.get(ref_key) != pending.reference or (
                observed.get("job_name"),
                observed.get("job_arn"),
            ) != (handle.id, handle.arn):
                if status in {"observation_failed", "UNKNOWN"}:
                    if self._await_original_model_unknown(pending).disposition == "awaiting_original":
                        return None
                raise _InputAdmissionHeldError()
            if status in {"UNKNOWN", "observation_failed"}:
                if self._await_original_model_unknown(pending).disposition == "awaiting_original":
                    return None
                raise _InputAdmissionHeldError()
            if status in {"PENDING", "RUNNING"}:
                with self._warm_capture_lock:
                    self._warm_unknown_pending, self._warm_unknown_polls = pending, 0
                return None
            if (
                status != "SUCCEEDED"
                or observed.get("success") is not True
                or observed.get("acquired") is not True
            ):
                raise _InputAdmissionHeldError()
            files, manifests = observed.get("file_refs"), observed.get("manifest_refs")
            if (
                type(files) is not list
                or len(files) != 9
                or type(manifests) is not dict
                or set(manifests) != {"receipt", "commit", "lineage"}
            ):
                raise _InputAdmissionHeldError()
            for artifact in [*files, *manifests.values()]:
                if (
                    type(artifact) is not dict
                    or type(artifact.get("version_id")) is not str
                    or not artifact["version_id"]
                    or type(artifact.get("sha256")) is not str
                    or not re.fullmatch("[a-f0-9]{64}", artifact["sha256"])
                ):
                    raise _InputAdmissionHeldError()
            content = _OriginalModelCompletionEvent(pending.reference, pending.tool_use_id).as_content()
            _warn_precommit_handoff("READY", pending, status=status, ready=True)
            return content
        from ...backends.adapters.artifact_rejection import is_closed_rejection

        if status in {"verification_failed", "output_verification_failed"} and is_closed_rejection(
            observed.get("verification_rejection")
        ):
            raise _InputAdmissionHeldError()
        if pending.kind == "case":
            if status in {"verification_failed", "observation_failed"} and observed.get("success") is False:
                if self._await_original_case_failure(pending).disposition == "awaiting_original":
                    return None
                raise _InputAdmissionHeldError()
            if (
                status in {"PENDING", "RUNNING"}
                and observed.get("accepted") is True
                and observed.get(ref_key) == pending.reference
                and (observed.get("job_name"), observed.get("job_arn")) == (handle.id, handle.arn)
            ):
                self._warm_case_failure_counts.pop(self._producer_key(pending), None)
        if status in {"FAILED", "STOPPED", "failed", "stopped", "submission_rejected"}:
            raise _InputAdmissionHeldError()
        complete = {
            "data": observed.get("success") is True and observed.get("acquired") is True,
            "case": observed.get("success") is True
            and observed.get("frozen") is True
            and observed.get("case_count") == 164,
            "preparation": observed.get("success") is True and observed.get("prepared") is True,
            "sft": observed.get("success") is True
            and observed.get("trained") is True
            and (
                handle.extra.get("recipe_id") != "full_v1" or observed.get("full_training_complete") is True
            ),
            "reload": observed.get("success") is True
            and status == "reloaded"
            and observed.get("fresh_process_greedy_equal") is True,
            "generation": observed.get("success") is True and observed.get("generated") is True,
            "pair": observed.get("success") is True and observed.get("evaluated") is True,
        }[pending.kind]
        if not complete:
            if status in _PRECOMMIT_UNRESOLVED_STATUSES:
                self._note_precommit_unresolved(pending)
            else:
                self._clear_precommit_unresolved(pending)
            return None
        self._clear_precommit_unresolved(pending)
        if observed.get(ref_key) != pending.reference or (
            observed.get("job_name"),
            observed.get("job_arn"),
        ) != (handle.id, handle.arn):
            raise _InputAdmissionHeldError()
        value = {
            "schema_version": 1,
            "event_type": "original_" + pending.kind + "_completed",
            "data_only": True,
            "request_ref": pending.reference,
            "tool_use_id": pending.tool_use_id,
            "status": "verified_completed",
        }
        if pending.kind == "case":
            value["case_ref"] = pending.reference
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        if len(encoded.encode()) > _COMPLETION_EVENT_MAX_BYTES:
            raise _InputAdmissionHeldError()
        _warn_precommit_handoff("READY", pending, status=status, ready=True)
        return [{"text": encoded}]

    def _admit_precommit_original(self, text, event, witness, pending, closed):
        with self._input_lock:
            try:
                self._qualify_precommit_original(event, witness, pending, closed)
                _warn_precommit_handoff("ADMITTING", pending)
                research = self._qualify_research_binding()
                if research is not witness.research or research["phase"] != "CONSUMED":
                    raise _InputAdmissionHeldError()
                checker = self._selected_input_checker()
                checker = self._refresh_input_checker(checker, research)
                # Only this current owner-created replacement can move the active
                # completion witness; arbitrary TOOL_RESULT/research cannot rotate.
                witness.checker = checker
                self._qualify_precommit_original(event, witness, pending, closed)
                self._admit_input(text, "TOOL_RESULT", immutable=True, tool_name=event.tool_use["name"])
                self._qualify_precommit_original(event, witness, pending, closed)
                _warn_precommit_handoff("ADMITTED", pending)
            except Exception:
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold(
                        "AMLAS_INPUT_HOLD phase=pending_bookkeeping reason=completion_admission_refused"
                    )
                _warn_precommit_handoff("ADMISSION_REFUSED", pending)
                raise _InputAdmissionHeldError() from None

    async def _complete_original_before_commit(self, event, witness):
        self._qualify_pending_pause(witness)
        with self._warm_capture_lock:
            candidates = ([self._warm_pending] if self._warm_pending is not None else []) + list(
                self._warm_pending_producers.values()
            )
        selected = [
            pending
            for pending in candidates
            if pending.agent is event.agent and pending.tool_use_id == event.tool_use.get("toolUseId")
        ]
        if not selected:
            return
        if len(selected) != 1 or event.retry is not False:
            raise _InputAdmissionHeldError()
        pending = selected[0]
        witness.originals[pending.tool_use_id] = pending
        closed = threading.Event()
        task = None
        try:
            if type(pending) is _PendingOriginalProducer and pending.kind == "sft":
                from ...control_plane.normal_run_checkpoint import current_normal_run

                context = current_normal_run()
                if context is not None:
                    self._qualify_precommit_original(event, witness, pending, closed)
                    admitted = _record_messages(event.agent.messages)
                    snapshot = context._captured_sft_snapshot(pending.tool_use_id, admitted)
                    if snapshot is None:
                        snapshot = _record_session_snapshot(event.agent)
                    context.save_admitted_task(
                        snapshot,
                        pending.tool_use_id,
                        admitted,
                        original_job_name=pending.record.id,
                        policy=self.approval_policy,
                    )
                    self._qualify_precommit_original(event, witness, pending, closed)
            while True:
                self._qualify_precommit_original(event, witness, pending, closed)
                task = asyncio.create_task(
                    asyncio.to_thread(self._read_precommit_original, event, witness, pending, closed)
                )
                content = await asyncio.shield(task)
                self._qualify_precommit_original(event, witness, pending, closed)
                if content is not None:
                    result = {"toolUseId": pending.tool_use_id, "status": "success", "content": content}
                    text = json.dumps(content, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    self._admit_precommit_original(text, event, witness, pending, closed)
                    self._qualify_precommit_original(event, witness, pending, closed)
                    event.result = result
                    witness.completed[pending.tool_use_id] = pending
                    if type(pending) is _PendingOriginalProducer and pending.kind == "sft":
                        _note_normal_sft_result_delivered()
                    return
                await self._wait_precommit_original()
        except asyncio.CancelledError:
            closed.set()
            with self._warm_capture_lock:
                self._warm_closed = self._warm_closed or "refused"
            if task is not None:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if task.done() and not task.cancelled():
                    task.exception()
            raise
        finally:
            closed.set()

    def _pending_pause_result(self, agent, result):
        from strands.agent.agent_result import AgentResult

        witness = self._pending_batch_pause
        if witness is None or witness.agent is not agent:
            return False
        if self._extract_guardrail_outcome(result) is not None:
            return False
        if not witness.marked:
            if witness.request_state.get("stop_event_loop"):
                already_held = self._input_hold
                self._input_hold = True
                if not already_held:
                    _warn_input_hold("AMLAS_INPUT_HOLD phase=pause_result reason=unmarked_stop")
                raise _InputAdmissionHeldError()
            return False
        self._qualify_pending_pause(witness)
        if (
            type(result) is not AgentResult
            or type(result.stop_reason) is not str
            or result.stop_reason != "tool_use"
            or not witness.settled
            or not witness.originals
            or witness.request_state.get("stop_event_loop") is not True
            or self._pause_history(agent.messages) != witness.history
        ):
            already_held = self._input_hold
            self._input_hold = True
            if not already_held:
                _warn_input_hold("AMLAS_INPUT_HOLD phase=pause_result reason=stop_state_mismatch")
            raise _InputAdmissionHeldError()
        witness.accepted = True
        return True

    def _bind_warm_owner(self, owner):
        from .engine import _WARM_CHAT_ENGINE_TYPE

        if type(owner) is not _WARM_CHAT_ENGINE_TYPE or owner.tool_backend is not self:
            raise ValueError("exact original engine/backend lifetime required")
        if self._warm_owner is not None or self._warm_pid != os.getpid():
            raise ValueError("warm owner already bound or inherited")
        self._warm_owner = owner
        self._warm_owner_graph = self._warm_graph()

    def _close_warm_original(self, owner):
        """Permanently refuse only this engine's work; never cancel native jobs."""
        from .engine import _WARM_CHAT_ENGINE_TYPE

        if (
            type(owner) is not _WARM_CHAT_ENGINE_TYPE
            or owner.tool_backend is not self
            or self._warm_owner is not owner
            or self._warm_pid != os.getpid()
        ):
            raise ValueError("exact original warm closure owner required")
        with self._warm_capture_lock:
            self._warm_closed = "refused"

    def _warm_graph(self):
        return tuple(
            getattr(self, name)
            for name in (
                "_acquisition_store",
                "_model_acquisition_store",
                "_case_freeze_store",
                "_preparation_store",
                "_sft_store",
                "_generation_store",
                "_pair_store",
            )
        )

    @staticmethod
    def _warm_handle_pin(handle):
        return json.dumps(
            [handle.backend, handle.id, handle.arn, handle.extra],
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )

    def _qualify_warm_original(self, pending):
        from ...control_plane.approvals import request_approval
        from .engine import _WARM_CHAT_ENGINE_TYPE

        if (
            pending.pid != os.getpid()
            or self._warm_pid != os.getpid()
            or type(self._warm_owner) is not _WARM_CHAT_ENGINE_TYPE
            or self._warm_owner.tool_backend is not self
        ):
            raise _WarmSourceRefusalError("owner_unavailable")
        with self._warm_capture_lock:
            if self._warm_closed is not None or self._warm_pending is not pending:
                raise _WarmSourceRefusalError("pending_closed_or_replaced")
        graph = self._warm_graph()
        if any(a is not b for a, b in zip(graph, pending.graph, strict=True)) or any(
            a is not b for a, b in zip(graph, self._warm_owner_graph, strict=True)
        ):
            raise _WarmSourceRefusalError("graph_changed")
        store = pending.store
        store._require_invocation_context()
        with store._lock:
            record = store._records.get(pending.reference)
        if record is not pending.record or self._warm_handle_pin(record.handle) != pending.handle_pin:
            raise _WarmSourceRefusalError("record_pin_changed")
        if record.handle.extra["bindings"]["runtime_session_id"] != store.runtime_session_id:
            raise _WarmSourceRefusalError("correlation_changed")
        if store._campaign_binding is not None:
            store._require_original_campaign_context()
            store._campaign_binding.require_current_exposure(pending.reference)
        state = self._invocation_state()
        decision = request_approval(
            action="run_processing_job",
            cost_estimate=None,
            impact="continue original model acquisition under current caller policy",
            policy=state["_amlas_approval_policy"],
        )
        if decision["decision"] != "approved":
            raise _WarmSourceRefusalError("policy_refused")
        return state

    def _require_diagnostic_origin_current(self, pending):
        """Post-qualifier local identity/policy fence; invokes no producer SDK."""
        from ...control_plane.approvals import request_approval
        from .engine import _WARM_CHAT_ENGINE_TYPE

        selected = self.approval_policy
        if selected is not None and not isinstance(selected, ApprovalPolicy):
            raise ValueError("diagnostic policy differs")
        policy = (
            replace(selected, allowlist=list(selected.allowlist))
            if selected is not None
            else ApprovalPolicy()
        )
        if (
            request_approval(
                action="run_processing_job",
                cost_estimate=None,
                impact="observe original diagnostic under current caller policy",
                policy=policy,
            )["decision"]
            != "approved"
        ):
            raise ValueError("diagnostic policy refused")
        if (
            type(pending) is not _PendingOriginalModel
            or pending.pid != os.getpid()
            or self._warm_pid != os.getpid()
            or type(self._warm_owner) is not _WARM_CHAT_ENGINE_TYPE
            or self._warm_owner.tool_backend is not self
            or any(a is not b for a, b in zip(self._warm_graph(), pending.graph, strict=True))
            or any(a is not b for a, b in zip(self._warm_graph(), self._warm_owner_graph, strict=True))
        ):
            raise ValueError("diagnostic original owner differs")
        with self._warm_capture_lock:
            if self._warm_closed is not None or self._warm_pending is not pending:
                raise ValueError("diagnostic original closed")
        with pending.store._lock:
            if pending.store._records.get(pending.reference) is not pending.record:
                raise ValueError("diagnostic original record differs")
        if self._warm_handle_pin(pending.record.handle) != pending.handle_pin:
            raise ValueError("diagnostic original pin differs")

    def _stage_original_model_diagnostic(self, slot):
        from ...api import _PromptModelDiagnosticSlot
        from ...backends.adapters.model_weight_acquisition import _project_model_diagnostic

        if type(slot) is not _PromptModelDiagnosticSlot or not slot._producer_current():
            return
        with self._warm_capture_lock:
            pending = self._warm_pending
        if type(pending) is not _PendingOriginalModel:
            return
        try:
            self._qualify_warm_original(pending)
            self._require_diagnostic_origin_current(pending)
            value = _project_model_diagnostic(
                pending.record.handle.extra.get("submission_diagnostic"), "create_boundary"
            )
            if value is not None:
                slot._stage(
                    _OriginalModelDiagnosticPublication(
                        self,
                        pending,
                        copy_context(),
                        tuple(value[key] for key in _MODEL_DIAGNOSTIC_FIELDS),
                        _model_diagnostic_now() + 15.0,
                        os.getpid(),
                        threading.get_ident(),
                    )
                )
        except Exception as error:
            logger.debug(
                "MODEL_DIAGNOSTIC_PUBLICATION_DROP phase=producer error_class=%s", _warm_error_class(error)
            )
            # Failure of optional DATA publication leaves existing warm decisions.
            return

    @staticmethod
    def _producer_key(pending):
        return pending.kind, id(pending.store), pending.reference, pending.tool_use_id

    @staticmethod
    def _producer_handle(pending):
        return pending.record.handle if pending.kind in {"data", "preparation"} else pending.record

    @staticmethod
    def _producer_handle_pin(kind, handle):
        # Explicit immutable source projections; lifecycle/cache telemetry excluded.
        from ...arbiter.coding import CodingEvaluationBinding
        from ...backends.types import DatasetArtifactRef, DatasetInspectionSource, VerifiedPreparationSource
        from ...control_plane.coding_case_state import CaseFreezeCatalog, VerifiedCaseEvaluationSource
        from ...control_plane.generation_state import VerifiedCodingGenerationSource
        from ...control_plane.paired_state import PairedCodingCatalog, PairedCodingInputs
        from ...control_plane.preparation_state import PreparationCatalog
        from ...control_plane.sft_state import (
            SftCatalog,
            VerifiedSftAdoptedModelSource,
            VerifiedSftCheckpointSource,
            VerifiedSftModelSource,
            VerifiedSftPreparedSource,
        )
        from ...spec.generation import ManifestRef

        def value(item):
            if item is None or type(item) in (str, int, bool, bytes):
                return type(item).__name__, item
            if type(item) in (tuple, list):
                return type(item).__name__, tuple(value(part) for part in item)
            if type(item) is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("original producer mapping differs")
                return "mapping", tuple((key, value(item[key])) for key in sorted(item))
            fields = {
                CodingEvaluationBinding: (
                    "profile_id",
                    "evaluation_id",
                    "spec_sha256",
                    "case_manifest",
                    "source_sha256",
                    "runner_sha256",
                    "runtime_sha256",
                    "budgets_sha256",
                    "generation_sha256",
                    "base_model_sha256",
                    "tuned_model_sha256",
                ),
                ManifestRef: ("uri", "version_id", "sha256", "byte_count", "format"),
                DatasetArtifactRef: ("uri", "version_id", "sha256", "byte_count", "format"),
                DatasetInspectionSource: (
                    "artifact_ref",
                    "account_id",
                    "region",
                    "bucket",
                    "runtime_session_id",
                    "deployment_id",
                    "deployment_prefix",
                    "acquisition_job_arn",
                ),
                VerifiedPreparationSource: (
                    "inspection_source",
                    "manifest_refs",
                    "repo_id",
                    "revision",
                    "filename",
                    "acquisition_image_uri",
                    "acquisition_source_commit",
                ),
                CaseFreezeCatalog: (
                    "account_id",
                    "region",
                    "bucket",
                    "execution_role_arn",
                    "image_uri",
                    "source_commit",
                    "admission_digest",
                    "deployment_id",
                    "deployment_prefix",
                ),
                PreparationCatalog: (
                    "account_id",
                    "region",
                    "bucket",
                    "execution_role_arn",
                    "image_uri",
                    "source_commit",
                    "admission_digest",
                    "deployment_id",
                    "deployment_prefix",
                    "exclusions_ref",
                    "instance_type",
                    "volume_size_gb",
                    "max_runtime_seconds",
                ),
                SftCatalog: (
                    "account_id",
                    "region",
                    "bucket",
                    "execution_role_arn",
                    "image_uri",
                    "source_commit",
                    "admission_digest",
                    "deployment_id",
                    "deployment_prefix",
                ),
                VerifiedSftAdoptedModelSource: (
                    "runtime_session_id",
                    "catalog_id",
                    "catalog_version",
                    "original",
                    "file_refs",
                    "manifest_refs",
                    "adoption_ref",
                ),
                VerifiedSftModelSource: (
                    "runtime_session_id",
                    "catalog_id",
                    "catalog_version",
                    "producer_job_arn",
                    "producer_bindings",
                    "file_refs",
                    "manifest_refs",
                ),
                VerifiedSftPreparedSource: (
                    "runtime_session_id",
                    "producer_job_arn",
                    "header",
                    "preparation_ref",
                    "split_refs",
                    "sidecar_refs",
                    "exclusions_ref",
                    "source_row_counts",
                ),
                VerifiedSftCheckpointSource: (
                    "runtime_session_id",
                    "producer_job_arn",
                    "recipe_id",
                    "source_commit",
                    "checkpoint_ref",
                    "file_refs",
                ),
                VerifiedCaseEvaluationSource: (
                    "runtime_session_id",
                    "producer_job_arn",
                    "producer_header",
                    "original_source",
                    "file_refs",
                ),
                VerifiedCodingGenerationSource: (
                    "runtime_session_id",
                    "producer_job_arn",
                    "evaluation_id",
                    "arm",
                    "file_refs",
                ),
                PairedCodingCatalog: ("execution", "launcher_sha256"),
                PairedCodingInputs: ("cases", "base", "tuned", "spec_bytes", "spec_sha256"),
            }
            selected = fields.get(type(item))
            if selected is None:
                raise ValueError("untyped original producer field")
            return type(item).__name__, tuple((name, value(getattr(item, name))) for name in selected)

        extra = handle.extra
        selected = {
            "data": (
                "repo_id",
                "revision",
                "filename",
                "bucket",
                "output_prefix",
                "output_uri",
                "region",
                "role_arn",
                "account_id",
                "image_uri",
                "image_digest",
                "source_commit",
                "runtime_session_id",
                "deployment_id",
                "deployment_prefix",
                "egress_environment",
                "privilege_profile",
                "receipt_schema_version",
            ),
            "case": (
                "runtime_session_id",
                "source",
                "catalog",
                "header",
                "output_prefix",
                "binding_ref",
                "request",
            ),
            "preparation": (
                "runtime_session_id",
                "catalog",
                "sources",
                "header",
                "output_prefix",
                "selection_sha256",
                "exclusions_sha256",
                "source_records",
                "source_row_counts",
                "request",
            ),
            "sft": (
                "runtime_session_id",
                "catalog",
                "model_source",
                "prepared_source",
                "operation",
                "recipe_id",
                "request",
                "run_ref",
                "training_record_count",
                "staged_refs",
            ),
            "reload": (
                "runtime_session_id",
                "catalog",
                "model_source",
                "prepared_source",
                "operation",
                "recipe_id",
                "request",
                "run_ref",
                "training_record_count",
                "staged_refs",
            ),
            "generation": (
                "runtime_session_id",
                "catalog",
                "request",
                "run_ref",
                "run_binding",
                "staged_refs",
                "original_base_refs",
                "arm",
                "evaluation_id",
            ),
            "pair": (
                "runtime_session_id",
                "catalog",
                "pair_catalog",
                "request",
                "run_ref",
                "run_binding",
                "evaluation_binding",
                "inputs",
                "staged_refs",
                "spec_ref",
                "output_prefix",
            ),
        }[kind]
        return (
            handle.backend,
            handle.id,
            handle.arn,
            tuple((name, value(extra.get(name))) for name in selected),
        )

    def _qualify_warm_producer(self, pending):
        from ...control_plane.approvals import request_approval
        from .engine import _WARM_CHAT_ENGINE_TYPE

        if (
            type(pending) is not _PendingOriginalProducer
            or pending.pid != os.getpid()
            or self._warm_pid != os.getpid()
            or type(self._warm_owner) is not _WARM_CHAT_ENGINE_TYPE
            or self._warm_owner.tool_backend is not self
        ):
            raise ValueError("original producer owner differs")
        with self._warm_capture_lock:
            if (
                self._warm_closed is not None
                or self._warm_pending_producers.get(self._producer_key(pending)) is not pending
            ):
                raise ValueError("original producer closed")
        if any(a is not b for a, b in zip(self._warm_graph(), pending.graph, strict=True)) or any(
            a is not b for a, b in zip(self._warm_graph(), self._warm_owner_graph, strict=True)
        ):
            raise ValueError("original producer graph differs")
        with pending.store._lock:
            if pending.store._records.get(pending.reference) is not pending.record:
                raise ValueError("original producer record changed")
            adapter = (
                pending.record.adapter if pending.kind in {"data", "preparation"} else pending.store._adapter
            )
            if adapter is not pending.adapter:
                raise ValueError("original producer adapter changed")
        handle = self._producer_handle(pending)
        if (
            self._producer_handle_pin(pending.kind, handle) != pending.handle_pin
            or handle.extra.get("runtime_session_id") != pending.store.runtime_session_id
        ):
            raise ValueError("original producer pin changed")
        state = self._invocation_state()
        action = next(config[5] for config in _ORIGINAL_PRODUCERS.values() if config[0] == pending.kind)
        if (
            request_approval(
                action=action,
                cost_estimate=None,
                impact="read original managed producer",
                policy=state["_amlas_approval_policy"],
            )["decision"]
            != "approved"
        ):
            raise ValueError("original producer policy refused")
        return state

    def _await_original_case_failure(self, pending):
        """Bound only this source-qualified CASE's consecutive failed observations."""
        self._qualify_warm_producer(pending)
        if self._input_hold:
            raise _InputAdmissionHeldError()
        key = self._producer_key(pending)
        with self._warm_capture_lock:
            if (
                pending.kind != "case"
                or self._warm_closed is not None
                or self._warm_pending_producers.get(key) is not pending
            ):
                raise ValueError("original CASE failure owner changed")
            count = self._warm_case_failure_counts.get(key, 0) + 1
            self._warm_case_failure_counts[key] = count
            if count < _MODEL_UNKNOWN_OBSERVATION_LIMIT:
                return _WarmModelStep("awaiting_original")
            diagnostic = _WarmRefusalDiagnostic(
                "native_observation", "original_unknown_budget_exhausted", "UNKNOWN"
            )
            self._warm_closed = "submission_unresolved"
            self._warm_closed_diagnostic = diagnostic
        return _WarmModelStep(
            "submission_unresolved",
            AgentTurn(
                final_text="The original native CASE outcome remains UNKNOWN.",
                executions=[],
                input_admission=self._input_admission_data(),
            ),
            diagnostic,
        )

    def _observe_pending_producer_once(self):
        with self._warm_lock:
            with self._warm_capture_lock:
                if self._warm_closed is not None:
                    return _WarmModelStep(self._warm_closed, diagnostic=self._warm_closed_diagnostic)
                pending_values = tuple(self._warm_pending_producers.values())
            if not pending_values:
                return _WarmModelStep("refused")
            # Rotate over retained originals; per-tick work is finite, retention is not capped.
            pending = pending_values[self._warm_producer_cursor % len(pending_values)]
            resume_assessment = None
            self._warm_producer_cursor += 1
            try:
                self._qualify_warm_producer(pending)
                with pending.adapter.qualified_io(lambda: self._qualify_warm_producer(pending)):
                    observed = pending.store.observe(pending.reference)
                self._qualify_warm_producer(pending)
                status = observed.get("status")
                from ...backends.adapters.artifact_rejection import is_closed_rejection

                if (
                    status in {"verification_failed", "output_verification_failed"}
                    and observed.get("success") is False
                    and is_closed_rejection(observed.get("verification_rejection"))
                ):
                    ref_key = "case_ref" if pending.kind == "case" else "request_ref"
                    handle = self._producer_handle(pending)
                    if observed.get(ref_key) != pending.reference or (
                        observed.get("job_name"),
                        observed.get("job_arn"),
                    ) != (handle.id, handle.arn):
                        raise ValueError("closed original rejection identity differs")
                    self._qualify_warm_producer(pending)
                    with self._warm_capture_lock:
                        key = self._producer_key(pending)
                        self._warm_consumed.add(key)
                        self._warm_consumed.add(
                            ("rejected_original", pending.kind, id(pending.store), pending.reference)
                        )
                        self._warm_pending_producers.pop(key, None)
                    return _WarmModelStep("terminal_failed")
                if status in {"FAILED", "STOPPED", "failed", "stopped", "submission_rejected"}:
                    with self._warm_capture_lock:
                        self._warm_pending_producers.pop(self._producer_key(pending), None)
                    return _WarmModelStep("terminal_failed")
                if pending.kind == "case":
                    if (
                        type(status) is str
                        and status in {"verification_failed", "observation_failed"}
                        and observed.get("success") is False
                    ):
                        return self._await_original_case_failure(pending)
                    handle = self._producer_handle(pending)
                    if (
                        type(status) is str
                        and status in {"PENDING", "RUNNING"}
                        and observed.get("accepted") is True
                        and observed.get("case_ref") == pending.reference
                        and (observed.get("job_name"), observed.get("job_arn")) == (handle.id, handle.arn)
                    ):
                        self._warm_case_failure_counts.pop(self._producer_key(pending), None)
                complete = {
                    "data": observed.get("success") is True and observed.get("acquired") is True,
                    "case": observed.get("success") is True
                    and observed.get("frozen") is True
                    and observed.get("case_count") == 164,
                    "preparation": observed.get("success") is True and observed.get("prepared") is True,
                    "sft": observed.get("success") is True
                    and observed.get("trained") is True
                    and (
                        self._producer_handle(pending).extra.get("recipe_id") != "full_v1"
                        or observed.get("full_training_complete") is True
                    ),
                    "reload": observed.get("success") is True
                    and status == "reloaded"
                    and observed.get("fresh_process_greedy_equal") is True,
                    "generation": observed.get("success") is True and observed.get("generated") is True,
                    "pair": observed.get("success") is True and observed.get("evaluated") is True,
                }[pending.kind]
                if not complete:
                    if status in _PRECOMMIT_UNRESOLVED_STATUSES:
                        try:
                            self._note_precommit_unresolved(pending)
                        except _InputAdmissionHeldError:
                            return _WarmModelStep(
                                "submission_unresolved", diagnostic=self._warm_closed_diagnostic
                            )
                    else:
                        self._clear_precommit_unresolved(pending)
                    return _WarmModelStep("awaiting_original")
                self._clear_precommit_unresolved(pending)
                ref_key = "case_ref" if pending.kind == "case" else "request_ref"
                handle = self._producer_handle(pending)
                if observed.get(ref_key) != pending.reference or (
                    observed.get("job_name"),
                    observed.get("job_arn"),
                ) != (handle.id, handle.arn):
                    raise ValueError("original completed producer identity differs")
                matches = [
                    block["toolResult"]
                    for message in pending.agent.messages
                    for block in message["content"]
                    if type(block) is dict
                    and type(block.get("toolResult")) is dict
                    and block["toolResult"].get("toolUseId") == pending.tool_use_id
                ]
                if len(matches) != 1:
                    raise ValueError("original producer causal result differs")
                original_payload = self._planning_result_payload(matches[0])
                if original_payload.get(ref_key) != pending.reference or (
                    original_payload.get("job_name"),
                    original_payload.get("job_arn"),
                ) != (handle.id, handle.arn):
                    raise ValueError("original producer historical identity differs")
                self._qualify_warm_producer(pending)
                event = {
                    "schema_version": 1,
                    "event_type": "original_" + pending.kind + "_completed",
                    "data_only": True,
                    "request_ref": pending.reference,
                    "tool_use_id": pending.tool_use_id,
                    "status": "verified_completed",
                }
                if pending.kind == "case":
                    event["case_ref"] = pending.reference
                encoded = json.dumps(
                    event, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
                )
                if len(encoded.encode()) > _COMPLETION_EVENT_MAX_BYTES:
                    raise ValueError("original producer completion DATA exceeds bound")
                self._input_cycle = secrets.token_hex(16)
                encoded = self._admit_input(encoded, "COMPLETION_DATA", immutable=True)
                with self._warm_capture_lock:
                    key = self._producer_key(pending)
                    if key in self._warm_consumed:
                        raise ValueError("original producer already consumed")
                    self._warm_consumed.add(key)
                if pending.kind == "sft":
                    _note_normal_sft_result_delivered()
                capture = GuardrailAssessmentCapture()
                previous = pending.agent.callback_handler
                composed = capture.wrap(previous) if callable(previous) else None
                if composed is not None:
                    pending.agent.callback_handler = composed
                try:
                    result = pending.agent(
                        [{"text": encoded}], invocation_state=self._qualify_warm_producer(pending)
                    )
                    self._qualify_warm_producer(pending)
                    turn = AgentTurn(
                        (
                            "Waiting for original native operations to complete."
                            if self._pending_pause_result(pending.agent, result)
                            else self._extract_final_text(result)
                        ),
                        self._extract_qualified_normal_executions(result, pending.agent),
                        self._extract_guardrail_outcome(result),
                        (resume_assessment := capture.close()),
                        input_admission=self._input_admission_data(),
                    )
                finally:
                    self._pending_batch_pause = None
                    resume_assessment = capture.close() or resume_assessment
                    if composed is not None and pending.agent.callback_handler is composed:
                        pending.agent.callback_handler = previous
                with self._warm_capture_lock:
                    self._warm_pending_producers.pop(key, None)
                    if turn.guardrail_outcome is not None:
                        self._warm_closed = "guardrail_intervened"
                return _WarmModelStep(
                    "guardrail_intervened" if turn.guardrail_outcome is not None else "continued_original",
                    turn,
                )
            except Exception as error:
                with self._warm_capture_lock:
                    self._warm_closed = "refused"
                if self._input_hold:
                    return _WarmModelStep(
                        "refused", self._input_hold_turn(guardrail_assessment=resume_assessment)
                    )
                return _WarmModelStep(
                    "refused",
                    diagnostic=_WarmRefusalDiagnostic(
                        "native_observation", "UNKNOWN", _warm_error_class(error)
                    ),
                )

    def _await_original_model_unknown(self, pending):
        """Bound only unresolved original observations, never known running work."""
        self._qualify_warm_original(pending)
        with self._warm_capture_lock:
            if self._warm_closed is not None or self._warm_pending is not pending:
                raise _WarmSourceRefusalError("pending_closed_or_replaced")
            if self._warm_unknown_pending is not pending:
                self._warm_unknown_pending = pending
                self._warm_unknown_polls = 0
            self._warm_unknown_polls += 1
            if self._warm_unknown_polls < _MODEL_UNKNOWN_OBSERVATION_LIMIT:
                return _WarmModelStep("awaiting_original")
            diagnostic = _WarmRefusalDiagnostic(
                "native_observation", "original_unknown_budget_exhausted", "UNKNOWN"
            )
            # Retain the exact original, private record and SDK history. This
            # closes only its conversation wait, without consuming/cancelling or
            # proving rejection, failure, acceptance or native job termination.
            self._warm_closed = "submission_unresolved"
            self._warm_closed_diagnostic = diagnostic
            return _WarmModelStep("submission_unresolved", diagnostic=diagnostic)

    def _observe_pending_model_once(self):
        with self._warm_capture_lock:
            only_producers = self._warm_pending is None and bool(self._warm_pending_producers)
        if only_producers:
            return self._observe_pending_producer_once()
        if self._warm_owner is None or self._warm_pid != os.getpid():
            return _WarmModelStep(
                "refused",
                diagnostic=_WarmRefusalDiagnostic("initial_qualification", "owner_unavailable", "UNKNOWN"),
            )
        with self._warm_lock:
            if self._warm_closed is not None:
                return _WarmModelStep(self._warm_closed, diagnostic=self._warm_closed_diagnostic)
            pending = self._warm_pending
            if pending is None:
                return _WarmModelStep("refused")
            phase = "initial_qualification"
            resume_assessment = None
            try:
                self._qualify_warm_original(pending)
                phase = "native_observation"
                with pending.record.adapter.qualified_io(lambda: self._qualify_warm_original(pending)):
                    observed = pending.store.observe(pending.reference)
                phase = "post_observation_qualification"
                self._qualify_warm_original(pending)
                phase = "observation_identity"
                if observed.get("request_ref") != pending.reference or (
                    observed.get("job_name"),
                    observed.get("job_arn"),
                ) != (pending.record.handle.id, pending.record.handle.arn):
                    if observed.get("status") in {"observation_failed", "UNKNOWN"}:
                        return self._await_original_model_unknown(pending)
                    raise _WarmSourceRefusalError("observation_identity_changed")
                status = observed.get("status")
                if status == "UNKNOWN":
                    return self._await_original_model_unknown(pending)
                if status in {"PENDING", "RUNNING"}:
                    with self._warm_capture_lock:
                        if self._warm_closed is not None or self._warm_pending is not pending:
                            raise _WarmSourceRefusalError("pending_closed_or_replaced")
                        self._warm_unknown_pending = pending
                        self._warm_unknown_polls = 0
                    return _WarmModelStep("awaiting_original")
                if status in {"FAILED", "STOPPED"}:
                    with self._warm_capture_lock:
                        self._warm_pending = None
                        if not self._warm_pending_producers:
                            self._warm_closed = "terminal_failed"
                    return _WarmModelStep("terminal_failed")
                phase = "artifact_result"
                if (
                    status != "SUCCEEDED"
                    or observed.get("success") is not True
                    or observed.get("acquired") is not True
                ):
                    if status == "verification_failed":
                        error_class = observed.get("error_type")
                        if type(error_class) is not str or error_class not in _WARM_REFUSAL_CLASSES:
                            error_class = "UNKNOWN"
                        raise _WarmSourceRefusalError("artifact_verification_failed", error_class)
                    raise _WarmSourceRefusalError("artifact_result_unverified", "UNKNOWN")
                phase = "artifact_inventory"
                files, manifests = observed.get("file_refs"), observed.get("manifest_refs")
                if (
                    type(files) is not list
                    or len(files) != 9
                    or type(manifests) is not dict
                    or set(manifests) != {"receipt", "commit", "lineage"}
                ):
                    raise _WarmSourceRefusalError("artifact_inventory_invalid")
                for artifact in [*files, *manifests.values()]:
                    if (
                        type(artifact) is not dict
                        or type(artifact.get("version_id")) is not str
                        or not artifact["version_id"]
                        or type(artifact.get("sha256")) is not str
                        or not re.fullmatch("[a-f0-9]{64}", artifact["sha256"])
                    ):
                        raise _WarmSourceRefusalError("artifact_pin_invalid")
                phase = "history_causality"
                matches = [
                    block["toolResult"]
                    for message in pending.agent.messages
                    for block in message["content"]
                    if type(block) is dict
                    and type(block.get("toolResult")) is dict
                    and block["toolResult"].get("toolUseId") == pending.tool_use_id
                ]
                if len(matches) != 1:
                    raise _WarmSourceRefusalError("history_causality_changed")
                original_payload = self._planning_result_payload(matches[0])
                if original_payload.get("request_ref") != pending.reference or (
                    original_payload.get("job_name"),
                    original_payload.get("job_arn"),
                ) != (pending.record.handle.id, pending.record.handle.arn):
                    raise _WarmSourceRefusalError("history_causality_changed")
                phase = "consume_transition"
                self._qualify_warm_original(pending)
                completion = _OriginalModelCompletionEvent(pending.reference, pending.tool_use_id)
                completion_content = completion.as_content()
                self._input_cycle = secrets.token_hex(16)
                self._admit_input(completion_content[0]["text"], "COMPLETION_DATA", immutable=True)
                with self._warm_capture_lock:
                    if self._warm_closed is not None or self._warm_pending is not pending:
                        raise _WarmSourceRefusalError("transition_closed_or_replaced")
                    self._warm_consumed.add((id(pending.store), pending.reference, pending.handle_pin))
                # Completed provider turns must retain their entire historical
                # prefix, including pending toolResult and signed thinking blocks.
                phase = "resume_qualification"
                capture = GuardrailAssessmentCapture()
                previous = pending.agent.callback_handler
                composed = capture.wrap(previous) if callable(previous) else None
                if composed is not None:
                    pending.agent.callback_handler = composed
                resume_completed = False
                try:
                    state = self._qualify_warm_original(pending)
                    phase = "resume"
                    result = pending.agent(completion_content, invocation_state=state)
                    phase = "post_resume_qualification"
                    self._qualify_warm_original(pending)
                    turn = AgentTurn(
                        final_text=(
                            "Waiting for original native operations to complete."
                            if self._pending_pause_result(pending.agent, result)
                            else self._extract_final_text(result)
                        ),
                        executions=self._extract_qualified_normal_executions(result, pending.agent),
                        guardrail_outcome=self._extract_guardrail_outcome(result),
                        guardrail_assessment=(resume_assessment := capture.close()),
                        input_admission=self._input_admission_data(),
                    )
                    resume_completed = True
                finally:
                    self._pending_batch_pause = None
                    if not resume_completed:
                        with self._warm_capture_lock:
                            self._warm_closed = self._warm_closed or "refused"
                    resume_assessment = capture.close() or resume_assessment
                    if composed is not None and pending.agent.callback_handler is composed:
                        pending.agent.callback_handler = previous
                phase = "exposure"
                self._qualify_warm_original(pending)
                with self._warm_capture_lock:
                    if self._warm_closed is not None or self._warm_pending is not pending:
                        raise _WarmSourceRefusalError("exposure_closed_or_replaced")
                    self._warm_pending = None
                    if turn.guardrail_outcome is not None:
                        self._warm_closed = "guardrail_intervened"
                if turn.guardrail_outcome is not None:
                    return _WarmModelStep("guardrail_intervened", turn)
                return _WarmModelStep("continued_original", turn)
            except Exception as error:
                reason = "continuation_failed" if phase == "resume" else "UNKNOWN"
                error_class = _warm_error_class(error)
                if type(error) is _WarmSourceRefusalError:
                    reason, error_class = error.reason, error.error_class
                diagnostic = _WarmRefusalDiagnostic(phase, reason, error_class)
                with self._warm_capture_lock:
                    self._warm_closed = "refused"
                    if self._warm_closed_diagnostic is None:
                        self._warm_closed_diagnostic = diagnostic
                    diagnostic = self._warm_closed_diagnostic
                if self._input_hold:
                    return _WarmModelStep(
                        "refused", self._input_hold_turn(guardrail_assessment=resume_assessment), diagnostic
                    )
                return _WarmModelStep("refused", diagnostic=diagnostic)

    @staticmethod
    def _extract_guardrail_outcome(result: Any) -> GuardrailOutcome | None:
        """Read only the verified SDK result primitive, never text or trace."""
        try:
            from strands.agent.agent_result import AgentResult
        except ImportError:
            return None
        if type(result) is not AgentResult:
            return None
        reason = result.stop_reason
        if type(reason) is str and reason == "guardrail_intervened":
            return GuardrailOutcome()
        return None

    @staticmethod
    def _extract_final_text(result: Any) -> str:
        """Pull the assistant text out of a Strands AgentResult (or stringify)."""
        message = getattr(result, "message", None)
        if isinstance(message, dict):
            content = message.get("content", [])
            texts = [block["text"] for block in content if isinstance(block, dict) and "text" in block]
            if texts:
                return "\n".join(texts)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, list):
                texts = [
                    getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
                    for b in content
                ]
                texts = [t for t in texts if t]
                if texts:
                    return "\n".join(texts)
        return str(result)

    def _extract_qualified_normal_executions(self, result: Any, agent: Any) -> list[ToolExecution]:
        """Use one admitted-history trace only for the current normal run."""
        from ...control_plane.normal_run_checkpoint import current_normal_run

        if current_normal_run() is None:
            return self._extract_executions(result)
        if (
            self._input_hold
            or self._warm_pid != os.getpid()
            or self._warm_owner is None
            or self._warm_owner.tool_backend is not self
        ):
            raise _InputAdmissionHeldError()
        # Owning callers already qualify their Agent/history before extraction
        # and exposure. This projects display DATA, never permission or metrics.
        return self._extract_normal_completion_executions(result, admitted_messages=agent.messages)

    def _extract_normal_completion_executions(
        self, result: Any, *, admitted_messages: list[dict[str, Any]]
    ) -> list[ToolExecution]:
        """Join admitted cold history with current trace; never restore SDK metrics.

        The normal claim/load path admits and qualifies these retained messages
        before this DATA-only projection. A historical result describes a past
        tool call; it grants no permission, native completion or retry authority.
        """
        import copy

        if type(admitted_messages) is not list:
            raise ValueError("admitted normal history DATA required")
        uses = {}
        results = {}
        ambiguous = set()
        for message_index, message in enumerate(admitted_messages):
            if type(message) is not dict or type(message.get("content")) is not list:
                raise ValueError("admitted normal message DATA required")
            for block_index, block in enumerate(message["content"]):
                if type(block) is not dict:
                    raise ValueError("admitted normal content DATA required")
                position = (message_index, block_index)
                if "toolUse" in block:
                    use = block["toolUse"]
                    if type(use) is not dict:
                        raise ValueError("admitted normal tool use DATA required")
                    identifier = use.get("toolUseId")
                    if type(identifier) is not str or not identifier:
                        raise ValueError("admitted normal tool use identity required")
                    if identifier in uses:
                        ambiguous.add(identifier)
                    name, args = self._tool_use_name_args(use)
                    if (
                        message.get("role") != "assistant"
                        or set(use) != {"toolUseId", "name", "input"}
                        or type(name) is not str
                        or not name
                        or type(use.get("input")) is not dict
                    ):
                        ambiguous.add(identifier)
                    uses.setdefault(identifier, (position, name, args))
                if "toolResult" in block:
                    returned = block["toolResult"]
                    if type(returned) is not dict:
                        raise ValueError("admitted normal tool result DATA required")
                    identifier = returned.get("toolUseId")
                    if type(identifier) is not str or not identifier:
                        raise ValueError("admitted normal tool result identity required")
                    if identifier in results:
                        ambiguous.add(identifier)
                    success = (
                        message.get("role") == "user"
                        and set(returned) == {"toolUseId", "status", "content"}
                        and returned["status"] == "success"
                        and type(returned["content"]) is list
                    )
                    results.setdefault(identifier, (position, success))
        executions: list[ToolExecution] = []
        seen: set[str] = set()
        for identifier, (position, name, args) in uses.items():
            returned = results.get(identifier)
            if (
                identifier in ambiguous
                or returned is None
                or returned[0] <= position
                or not returned[1]
                or name in seen
            ):
                continue
            seen.add(name)
            executions.append(
                ToolExecution(
                    tool_name=name,
                    args=copy.deepcopy(args),
                    status="executed",
                    result=None,
                    result_summary=name,
                )
            )
        for execution in self._extract_executions(result):
            if execution.tool_name not in seen:
                seen.add(execution.tool_name)
                executions.append(execution)
        return executions

    def _extract_executions(self, result: Any) -> list[ToolExecution]:
        """Best-effort extraction of executed tool calls from an AgentResult.

        Two complementary sources are merged (deduped by tool name, first-seen
        order preserved):

        1. ``EventLoopMetrics.tool_metrics`` — the AUTHORITATIVE cross-cycle record
           of every tool the agent invoked during the turn. The final assistant
           ``message`` only carries ``toolUse`` blocks for the LAST cycle, so a
           multi-cycle tool loop (e.g. ``list_directory`` in cycle 1, then the text
           answer in cycle 2) would otherwise leave the trace EMPTY even though a
           real tool fired — exactly the filesystem user-story turn (design doc 06).
           ``metrics.tool_metrics`` records ALL of them, so it is the reliable signal
           the engine surfaces as ``data["tools_used"]`` and the "🔧 used: …" line.
        2. The final ``message`` ``toolUse`` blocks — kept for the (common) single-
           cycle case and to capture the concrete call ``input`` args.

        Kept defensive — never raises; an unparseable result yields an empty trace
        (the final text still renders).
        """
        executions: list[ToolExecution] = []
        seen: set[str] = set()

        # 1. Authoritative cross-cycle record: EventLoopMetrics.tool_metrics.
        metrics = getattr(result, "metrics", None)
        tool_metrics = getattr(metrics, "tool_metrics", None)
        if isinstance(tool_metrics, dict):
            for tm in tool_metrics.values():
                name, args = self._tool_use_name_args(getattr(tm, "tool", None))
                if not name or name in seen:
                    continue
                seen.add(name)
                executions.append(
                    ToolExecution(
                        tool_name=name,
                        args=args,
                        status="executed",
                        result=None,
                        result_summary=name,
                    )
                )

        # 2. Final-message toolUse blocks (single-cycle case / concrete args).
        message = getattr(result, "message", None)
        content = message.get("content", []) if isinstance(message, dict) else []
        for block in content:
            if isinstance(block, dict) and "toolUse" in block:
                tu = block["toolUse"]
                name = tu.get("name", "tool")
                if not name or name in seen:
                    continue
                seen.add(name)
                executions.append(
                    ToolExecution(
                        tool_name=name,
                        args=tu.get("input", {}) or {},
                        status="executed",
                        result=None,
                        result_summary=name,
                    )
                )
        return executions

    @staticmethod
    def _tool_use_name_args(tool_use: Any) -> tuple[str, dict[str, Any]]:
        """Extract ``(name, input-args)`` from a Strands ``ToolUse`` (dict or object).

        ``ToolUse`` is a TypedDict (``{"toolUseId","name","input"}``) in practice,
        but attribute-style access is accepted too so this stays robust across
        Strands versions. Returns ``("", {})`` when nothing usable is present.
        """
        if tool_use is None:
            return "", {}
        if isinstance(tool_use, dict):
            name = tool_use.get("name") or ""
            args = tool_use.get("input") or {}
        else:
            name = getattr(tool_use, "name", "") or ""
            args = getattr(tool_use, "input", {}) or {}
        if not isinstance(args, dict):
            args = {}
        return name, args
