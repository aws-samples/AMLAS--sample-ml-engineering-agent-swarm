"""AgentCore Runtime entry point.

Hosts the normal prompt API and legacy Orchestrator behind `@app.entrypoint`.

A prompt-only payload is exactly ``{"prompt": "natural-language request"}``. It
uses ``api.run_prompt`` with safe default or explicit deployment approval, one new chat engine per
invocation, and no inferred durable conversation or lifecycle completion. The
legacy specification payload contract remains:

    { "spec_path":  "<path to task_spec.md inside the container>",     # one-of
      "spec_text":  "<inline task_spec.md content>",                   # one-of
      "stage_data": true,                                              # Phase 14b N1/N2
      "max_rows":   <int, optional cap for a cheap verification slice>,
      "y_true_path":"<local path OR s3:// uri to a .npy of ground truth>",
      "prompt":     "<optional free-form override — rarely used>"
    }

Phase 14b (N1/N2/B3/B5):
  * N1 — when `stage_data` is truthy, the runtime resolves the spec's
    `golden_dataset` (S3 or local) to concrete train/test files on the container
    filesystem via `cloud.staging.stage_dataset(...)` and threads those paths
    into the Orchestrator's problem card (so D1 `data_integrity` runs on REAL
    files instead of guessing `data/ai4i/train.csv`).
  * N2 — the SAME stage produces an aligned `y_true.npy`; if the operator passes
    an explicit `y_true_path` (incl. an `s3://` uri) it is resolved via
    `cloud.staging.load_y_true_npy(...)` instead.
  * B3 — `cloud.secrets.resolve_secret_into_env()` injects `TAVILY_API_KEY` from
    Secrets Manager (graceful no-op until Phase 14c populates the secret).
  * B5 — this `app.py` path builds the real Bedrock Lead hierarchy + agentic
    Orchestrator (research_search/Tavily + leads/subagents), NOT the offline
    deterministic `MLPipeline` KB.

The Runtime session exits when the entrypoint returns. For the async long-training
pathway (Phase K), the FineTune sub-agent submits a SageMaker job and the Runtime
exits; EventBridge → Lambda re-invokes with runtime_session_id preserved.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from .config.aws_region import resolve_aws_region

app = BedrockAgentCoreApp()

# Only active originals are retained; completed reports remain in the existing Session store.
_ACTIVE_PROMPT_REPORTS: set[str] = set()
_PROMPT_REPORT_LOCK = threading.Lock()

# Resume-owned workers, guarded by _PROMPT_REPORT_LOCK: run_id -> {"state", "diagnostic", "since"}.
# A resume request is an idempotent state query and starter. The closed answer is
# restoring (a resume-owned worker exists and has not started the continuation),
# continuing (it claimed the completion and is starting the continuation), completed
# (durable report), original_active (the run is active here but was not started by a
# resume), failed_transient and failed (the previous restore ended with a classified
# error). The stream Lambda acknowledges a wake only on continuing or completed.
# An original worker whose model received its SFT call result, and which ended with a
# retained report, records completed here too: a late resume has nothing to continue.
_RESUME_WORKERS: dict[str, dict[str, Any]] = {}
# A permanent failure answers "failed" for this long, then the next request restores again,
# so a misclassified error cannot strand the run in a live process.
_RESUME_FAILED_COOLDOWN_SECONDS = 30.0
# At most three application-level attempts around the replay-safe restore section only,
# separated by full-jitter pauses drawn from [0, cap]; boto3 clients stay single-attempt.
_RESTORE_ATTEMPTS = 3
_RESTORE_JITTER_CAPS = (0.5, 1.0)
_RESUME_STAGES = frozenset(
    {
        "run_prepare",
        "checkpoint_promote",
        "checkpoint_load",
        "snapshot_load",
        "continuation",
        "report_retention",
    }
)
_TRANSIENT_ERROR_CODES = frozenset(
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
        "InternalServerError",
        "InternalFailure",
        "InternalServerException",
        "ServiceUnavailable",
        "ServiceUnavailableException",
    }
)
_RESTORE_ERROR_CATEGORIES = {
    ValueError: "ValueError",
    TypeError: "TypeError",
    RuntimeError: "RuntimeError",
    AssertionError: "AssertionError",
    AttributeError: "AttributeError",
    KeyError: "KeyError",
    LookupError: "LookupError",
    OSError: "OSError",
    FileNotFoundError: "FileNotFoundError",
    PermissionError: "PermissionError",
    TimeoutError: "TimeoutError",
    NotImplementedError: "NotImplementedError",
}


def _resume_clock() -> float:
    return time.monotonic()


def _restore_pause(seconds: float) -> None:
    time.sleep(seconds)  # nosemgrep: arbitrary-sleep


def _restore_backoff(attempt: int) -> float:
    """Full jitter before the next restore attempt: uniform in [0, cap] for the attempt that failed."""
    cap = _RESTORE_JITTER_CAPS[min(attempt, len(_RESTORE_JITTER_CAPS) - 1)]
    return secrets.SystemRandom().uniform(0.0, cap)


def _restore_error_is_transient(error: BaseException) -> bool:
    """True only for throttling, HTTP 5xx and connection/timeout classes.

    Schema, digest, binding, byte-limit and immutable-prefix refusals are ValueError/TypeError,
    access and qualification refusals are PermissionError or an access-denied ClientError, and
    none of them is ever retried. A ValueError raised `from` an explicit transient cause (for
    example a conditional write whose outcome could not be read back) takes that cause's class.
    """
    try:
        from botocore.exceptions import (
            ClientError,
            ConnectionClosedError,
            ConnectTimeoutError,
            EndpointConnectionError,
            ReadTimeoutError,
        )

        client_error = ClientError
        network = (EndpointConnectionError, ConnectionClosedError, ConnectTimeoutError, ReadTimeoutError)
    except ImportError:  # pragma: no cover - boto3 is a runtime dependency
        client_error, network = None, ()
    current: BaseException | None = error
    for _ in range(4):
        if current is None or isinstance(current, PermissionError):
            return False
        if client_error is not None and isinstance(current, client_error):
            response = current.response if type(current.response) is dict else {}
            fields = response.get("Error") if type(response.get("Error")) is dict else {}
            meta = response.get("ResponseMetadata") if type(response.get("ResponseMetadata")) is dict else {}
            status = meta.get("HTTPStatusCode")
            return fields.get("Code") in _TRANSIENT_ERROR_CODES or (type(status) is int and status >= 500)
        if network and isinstance(current, network):
            return True
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if type(current) is ValueError:
            current = current.__cause__
            continue
        return False
    return False


def _restore_error_category(error: BaseException) -> str:
    return _RESTORE_ERROR_CATEGORIES.get(type(error), "Exception")


def _resume_state_event(run_id: str, state: str, diagnostic: dict[str, str] | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {"event": "resume_state", "state": state, "run_id": run_id}
    if diagnostic is not None:
        event["diagnostic"] = diagnostic
    return event


def _resume_failure_event(
    run_id: str, stage: str, error: BaseException, *, transient: bool
) -> dict[str, Any]:
    """Closed failure projection: fixed stage and reviewed exception category only."""
    return _resume_state_event(
        run_id,
        "failed_transient" if transient else "failed",
        {
            "restore_stage": stage if stage in _RESUME_STAGES else "run_prepare",
            "exception_class": _restore_error_category(error),
        },
    )


# FIX-26 (operator directive: NO arbitrary caps on the work the system does): there is NO
# absolute wall-clock ceiling. A run executes to its NATURAL completion. A hard deadline is
# applied ONLY when the operator explicitly asks for one — either a positive
# ``wall_clock_minutes`` in the spec, or the ``AMLAS_MAX_RUN_SECONDS`` ops escape hatch (also
# used by offline callers to drive a sub-minute deadline deterministically). Absent both, the run
# is UNBOUNDED. The heartbeat + terminal-event guarantee below are INFRA scaffolding (keep-alive
# so the AgentCore SSE stream never idle-closes without a terminal) — NOT a cap on the work.


def _record_finished_original(run_id: str, context: Any, box: dict[str, Any]) -> None:
    """Runs under _PROMPT_REPORT_LOCK when an original worker ends, with the release of its busy fence.

    An original whose model received the result of its SFT call has made the one batch transition a replacement
    would make. When it also returned a response and retained its report, a later resume of the run has nothing to
    continue: it answers completed (a state the stream Lambda acknowledges) and starts no restore, claim or model
    call. An original that ended before its SFT result reached the model (a pause, a held input, a stopped worker)
    records nothing, so a later resume restores the run as before.
    """
    if (
        context is not None
        and getattr(context, "_original_result_delivered", False) is True
        and "response" in box
        and "error_type" not in box
    ):
        _RESUME_WORKERS[run_id] = {"state": "completed", "diagnostic": None, "since": _resume_clock()}


# Closed labels for why an original SFT job ended; the native reason text itself is never copied.
_SFT_FAILURE_CLASSES = frozenset({"AlgorithmError", "CapacityError", "ClientError", "InternalServerError"})
_SFT_STOP_CLASSES = frozenset({"Stopped", "MaxRuntimeExceeded", "MaxWaitTimeExceeded"})


def _original_sft_failure_class(native_status: str, description: Any) -> str:
    """The class of an ended SFT job's reason: the FailureReason prefix or the SecondaryStatus, from closed sets."""
    if type(description) is not dict:
        return "unclassified"
    if native_status == "Failed":
        reason = description.get("FailureReason")
        label = reason.split(":", 1)[0].strip() if type(reason) is str else ""
        return label if label in _SFT_FAILURE_CLASSES else "unclassified"
    if native_status == "Stopped":
        secondary = description.get("SecondaryStatus")
        return secondary if type(secondary) is str and secondary in _SFT_STOP_CLASSES else "unclassified"
    return "unclassified"


class _OriginalSftEnd:
    """Closed facts of an original SFT job that ended Failed or Stopped; no authority."""

    __slots__ = ("failure_class", "job_name", "native_status", "recipe_id", "selected")

    def __init__(
        self, job_name: str, native_status: str, failure_class: str, recipe_id: Any, selected: Any = None
    ) -> None:
        self.job_name, self.native_status, self.failure_class = job_name, native_status, failure_class
        self.recipe_id = recipe_id if recipe_id in ("fit_v1", "full_v1") else None
        # SelectedInstanceType and SelectedInstanceCount of a preference-list job, both or neither.
        kind = selected.get("SelectedInstanceType") if type(selected) is dict else None
        count = selected.get("SelectedInstanceCount") if type(selected) is dict else None
        self.selected = (
            {"SelectedInstanceType": kind, "SelectedInstanceCount": count}
            if type(kind) is str and type(count) is int
            else {}
        )


def _original_sft_end(backend: Any, result: dict[str, Any]) -> _OriginalSftEnd:
    """Read one fresh native observation of the restored original and keep only closed facts."""
    store = backend._sft_store
    handle = store._records[result["request_ref"]]
    private = store._reload_observation_kwargs(handle.extra.get("reload_compatibility"))
    observed = store._adapter.status(handle, **private)
    description = observed.raw if type(observed.raw) is dict else {}
    native = description.get("TrainingJobStatus")
    expected = {"FAILED": "Failed", "STOPPED": "Stopped"}[result["status"]]
    if native != expected or handle.id != result.get("job_name"):
        raise ValueError("original SFT terminal observation differs")
    from .backends.adapters.qwen_sft import selected_instance

    return _OriginalSftEnd(
        handle.id,
        native,
        _original_sft_failure_class(native, description),
        result.get("recipe_id"),
        # Empty for a fixed-type job and for a preference-list job that ended before SageMaker selected a type.
        selected_instance(description, handle.extra.get("request")),
    )


def _resume_envelope(run_id: str) -> dict[str, Any]:
    """The closed envelope of a resumed run's stream and report."""
    try:
        # The replacement restores and continues under the deployment policy; the report names it as
        # the original's does. An invalid configuration ends the restore before any report exists.
        policy_source = (
            "deployment_allowlist" if _runtime_prompt_policy() is not None else "default_noninteractive"
        )
    except ValueError:
        policy_source = "default_noninteractive"
    return {
        "mode": "prompt",
        "run_id": run_id,
        "response_kind": "conversation",
        "state_scope": "single_invocation",
        "durable_resume": False,
        "approval_policy_source": policy_source,
        "lifecycle_verified": False,
        "quality_success": False,
    }


def _report_original_sft_end(run_id: str, session_hash: str, end: _OriginalSftEnd) -> dict[str, Any]:
    """Retain one source-written final report for a run whose original SFT job ended without a checkpoint.

    The report goes to the original's single-assignment pointer (no report is retained yet; the caller
    checked). It names the job, its native terminal state and the closed class of the reason, and says
    that nothing ran after it and nothing was retried. The resume then answers completed.
    """
    message = (
        "Fine-tuning did not finish, so this run ends here. The SFT training job "
        + end.job_name
        + " ended "
        + end.native_status
        + " (reason class: "
        + end.failure_class
        + ") after the worker that started the run had ended, and it produced no checkpoint. "
        "No generation or paired evaluation ran, no job was submitted again and nothing was retried. "
        "To train again, start a new run."
    )
    response = {
        "message": message,
        "action": "original_sft_ended",
        "data": {
            "original_sft": {
                "job_name": end.job_name,
                "native_status": end.native_status,
                "failure_class": end.failure_class,
                "recipe_id": end.recipe_id,
                "checkpoint": False,
                **end.selected,
            },
            "data_only": True,
            "quality_success": False,
        },
        "suggestions": [],
    }
    try:
        _retain_prompt_report(run_id, _resume_envelope(run_id), response, session_hash)
    except Exception as error:
        _warn_normal_restore_failure("report_retention", error)
        return _resume_failure_event(run_id, "report_retention", error, transient=False)
    return _resume_state_event(run_id, "completed")


def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to ``default`` on unset/invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _prompt_response(value: Any) -> dict[str, Any]:
    """Snapshot a typed conversation response without interpreting its claims."""
    import json

    from .tui.chat.models import ChatResponse

    if (
        not isinstance(value, ChatResponse)
        or type(value.message) is not str
        or type(value.action) is not str
        or type(value.data) is not dict
        or type(value.suggestions) is not list
        or any(type(item) is not str for item in value.suggestions)
    ):
        raise TypeError("prompt API must return a typed ChatResponse")
    return json.loads(
        json.dumps(
            {
                "message": value.message,
                "action": value.action,
                "data": value.data,
                "suggestions": value.suggestions,
            },
            allow_nan=False,
        )
    )


def _prompt_session_identity():
    """Capture SDK transport routing DATA and derive its existing hash once."""
    try:
        import hashlib

        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        session_id = BedrockAgentCoreContext.get_session_id()
        if type(session_id) is not str or not session_id.strip():
            return None, None
        return session_id, hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    except Exception:
        return None, None


def _prompt_report_session_hash():
    """Preserve the existing passive report session-hash contract."""
    return _prompt_session_identity()[1]


def _read_prompt_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Read one original completed report without starting execution."""
    run_id = payload.get("run_id")
    if (
        set(payload) != {"action", "run_id"}
        or type(run_id) is not str
        or len(run_id) != 32
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        return {"event": "error", "error_code": "invalid_report_request"}
    session_hash = _prompt_report_session_hash()
    if session_hash is None:
        return {"event": "error", "error_code": "report_session_unavailable"}
    unavailable = {
        "event": "prompt_report",
        "status": "unavailable",
        "run_id": run_id,
        "found": False,
    }
    with _PROMPT_REPORT_LOCK:
        if run_id in _ACTIVE_PROMPT_REPORTS:
            return unavailable
    try:
        from .control_plane.normal_run_checkpoint import read_normal_prompt_report

        report = read_normal_prompt_report(run_id, session_hash)
        if report is None:
            # Only absent normal configuration retains the legacy local path.
            from .api import resume

            restored = resume(run_id)
            if restored.get("found") is not True:
                return unavailable
            metadata = restored["context"]["metadata"]
            owner = metadata.get("normal_prompt_report_owner")
            report = metadata.get("normal_prompt_report")
        else:
            owner = {"schema_version": 1, "session_sha256": session_hash}
        if (
            type(owner) is not dict
            or set(owner) != {"schema_version", "session_sha256"}
            or type(owner["schema_version"]) is not int
            or owner["schema_version"] != 1
            or owner["session_sha256"] != session_hash
            or type(report) is not dict
            or report.get("run_id") != run_id
            or report.get("event") != "run_completed"
            or report.get("status") != "response_received"
            or report.get("worker_finished") is not True
            or report.get("mode") != "prompt"
            or report.get("quality_success") is not False
            or report.get("lifecycle_verified") is not False
            or report.get("durable_resume") is not False
            or report.get("state_scope") != "single_invocation"
        ):
            return unavailable
        return {
            "event": "prompt_report",
            "status": "available",
            "run_id": run_id,
            "found": True,
            "report": report,
        }
    except Exception:
        return unavailable


def _read_prompt_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Read or reconcile only an existing normal run; payload supplies no authority."""
    run_id = payload.get("run_id")
    if (
        set(payload) != {"action", "run_id"}
        or type(run_id) is not str
        or len(run_id) != 32
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        return {"event": "error", "error_code": "invalid_run_status_request"}
    session_hash = _prompt_report_session_hash()
    if session_hash is None:
        return {"event": "error", "error_code": "run_session_unavailable"}
    unavailable = {
        "event": "prompt_run_status",
        "status": "unavailable",
        "run_id": run_id,
        "found": False,
    }
    try:
        from .control_plane.normal_run_checkpoint import read_normal_run_status

        resume = payload["action"] == "resume_prompt_run"
        policy = _runtime_prompt_policy() if resume else None
        return read_normal_run_status(run_id, session_hash, policy=policy, resume=resume)
    except Exception:
        # Missing, foreign, malformed and unqualified records share one projection.
        return unavailable


# Status and report reads that run for a poll: one run per (action, run, session) at a time, on a daemon thread.
_READ_FLIGHTS: dict[tuple, tuple] = {}
_READ_FLIGHT_LOCK = threading.Lock()
# A poll waits this long (seconds) for its read. The client's read timeout is 120 s; the read continues on its thread.
_READ_RESPONSE_SECONDS = 60.0


async def _read_off_loop(reader, payload: dict[str, Any], event: str) -> dict[str, Any]:
    """Answer a status or report poll without occupying the app's event loop.

    The AgentCore app serves heartbeats, status polls and the prompt streams on one event loop. A status read of a run in
    the SFT stage verifies stored model bytes (a full re-hash of the pack when none was verified in this process), which on
    the loop thread stopped every heartbeat and every other poll for tens of minutes. The read runs on a daemon thread with
    the caller's contextvars (the SDK session identity, the active run). Polls with the same payload and session share one
    run of it and each receives its own copy of the answer. A poll waits at most AMLAS_STATUS_RESPONSE_SECONDS (default 60,
    below the client's read timeout) and then receives the closed unavailable answer, which the client already treats as "no
    verified status now"; the read keeps running and the next poll joins it. A cancelled poll does not cancel the read.
    """
    import asyncio
    import contextvars
    import copy
    import math

    run_id = payload.get("run_id")
    if set(payload) != {"action", "run_id"} or type(run_id) is not str or len(run_id) != 32:
        return reader(payload)  # a closed refusal of the request shape; it does no I/O
    key = (payload["action"], run_id, _prompt_report_session_hash())
    loop = asyncio.get_running_loop()
    frozen = dict(payload)
    with _READ_FLIGHT_LOCK:
        flight = _READ_FLIGHTS.get(key)
        starting = flight is None or flight[0] is not loop
        if starting:
            flight = (loop, loop.create_future())
            _READ_FLIGHTS[key] = flight
    future = flight[1]
    if starting:
        context = contextvars.copy_context()

        def work() -> None:
            try:
                outcome = (context.run(reader, frozen), None)
            except BaseException as error:
                outcome = (None, error)
            with _READ_FLIGHT_LOCK:
                if _READ_FLIGHTS.get(key) is flight:
                    del _READ_FLIGHTS[key]

            def finish() -> None:
                if future.done():
                    return
                if outcome[1] is None:
                    future.set_result(outcome[0])
                else:
                    future.set_exception(outcome[1])

            try:
                loop.call_soon_threadsafe(finish)
            except RuntimeError:  # the loop closed while the read ran
                pass

        future.add_done_callback(lambda done: done.cancelled() or done.exception())
        try:
            threading.Thread(target=work, name="prompt-read-" + run_id, daemon=True).start()
        except BaseException:
            with _READ_FLIGHT_LOCK:
                if _READ_FLIGHTS.get(key) is flight:
                    del _READ_FLIGHTS[key]
            raise
    seconds = _env_float("AMLAS_STATUS_RESPONSE_SECONDS", _READ_RESPONSE_SECONDS)
    if not math.isfinite(seconds) or seconds <= 0:
        seconds = _READ_RESPONSE_SECONDS
    waiting = asyncio.shield(future)
    done, _pending = await asyncio.wait([waiting], timeout=seconds)
    if not done:
        waiting.cancel()
        return {"event": event, "status": "unavailable", "run_id": run_id, "found": False}
    return copy.deepcopy(waiting.result())


def _warn_normal_restore_failure(stage: str, error: Exception) -> None:
    """Log the final refusal with its cause in the text: reason=final, fixed stage, reviewed exception category."""
    try:
        safe_stage = (
            stage
            if stage
            in {"run_prepare", "checkpoint_promote", "checkpoint_load", "snapshot_load", "continuation"}
            else "run_prepare"
        )
        category = _restore_error_category(error)
        logging.getLogger(__name__).warning(
            "normal_restore_unavailable reason=final restore_stage=%s exception_class=%s",
            safe_stage,
            category,
            extra={"restore_stage": safe_stage, "exception_class": category},
        )
    except Exception:
        pass


def _warn_normal_restore_retry(stage: str, error: Exception, attempt: int) -> None:
    """A transient restore attempt that the loop repeats: fixed reason, attempt number, stage and reviewed category.

    Only the attempt that ends the restore writes the `reason=final` refusal, so a reader can tell an attempt that healed
    from a final refusal by its reason. Like that refusal it never logs a message, an argument or frame details.
    """
    try:
        safe_stage = (
            stage
            if stage
            in {"run_prepare", "checkpoint_promote", "checkpoint_load", "snapshot_load", "continuation"}
            else "run_prepare"
        )
        logging.getLogger(__name__).warning(
            "normal_restore_unavailable reason=transient_retry attempt=%d attempts=%d restore_stage=%s"
            " exception_class=%s",
            attempt,
            _RESTORE_ATTEMPTS,
            safe_stage,
            _restore_error_category(error),
        )
    except Exception:
        return


def _resume_normal_prompt_body(payload, *, continue_completed=True, on_ready=None):
    """Restore current originals and optionally continue their admitted completion.

    The restore section is replayed at most three times, only for transient classes, with a
    fresh context and engine each time (every write inside it is a conditional write that
    reconciles an unknown outcome by reading the row back). The continuation is never inside
    that retry. A restore that ends in an error returns the closed `resume_state` failure
    event; it is never reported as an unavailable-but-successful response.
    """
    from contextlib import ExitStack

    from .control_plane.normal_run_checkpoint import (
        _original_training_complete,
        activate_normal_run,
        prepare_normal_run,
    )
    from .tui.chat.engine import ChatEngine

    run_id = payload.get("run_id")
    transport_session_id, session_hash = _prompt_session_identity()
    if (
        set(payload) != {"action", "run_id"}
        or type(run_id) is not str
        or len(run_id) != 32
        or any(char not in "0123456789abcdef" for char in run_id)
        or (session_hash is None)
    ):
        return _read_prompt_run(payload)
    unavailable = {"event": "prompt_run_status", "status": "unavailable", "run_id": run_id, "found": False}
    restore_stage = "run_prepare"

    def attempt(stack):
        """One restore attempt; returns (outcome, None) or (None, engine with the completion claimed)."""
        nonlocal restore_stage
        restore_stage = "run_prepare"
        policy = _runtime_prompt_policy()
        if transport_session_id is not None:
            context = prepare_normal_run(run_id, session_hash, transport_session_id=transport_session_id)
        else:
            context = prepare_normal_run(run_id, session_hash)
        if context is None:
            return unavailable, None
        stack.enter_context(activate_normal_run(context))
        has_task = context.has_task_checkpoint(policy=policy)
        if not has_task and context.has_prepared_checkpoint(policy=policy):
            restore_stage = "checkpoint_promote"
            context.promote_prepared_task(policy=policy)
            has_task = context.has_task_checkpoint(policy=policy)
            if not has_task:
                raise ValueError("promoted normal TASK checkpoint unavailable")
        if not has_task:
            return _read_prompt_run(payload), None
        restore_stage = "checkpoint_load"
        loaded = context.load_qualified_task(policy=policy)
        task, _parents, _original, _binding = loaded
        restore_stage = "snapshot_load"
        engine = ChatEngine(approval_policy=policy, normal_restore=(context, *loaded))
        result = engine.tool_backend._normal_original_result
        if result.get("status") in ("PENDING", "RUNNING"):
            engine.load_normal_snapshot(task)
            return _read_prompt_run(payload), None
        if _original_training_complete(result):
            engine.claim_normal_completion(context, task)
            return None, engine
        if result.get("status") in ("FAILED", "STOPPED"):
            # The original SFT job ended without a checkpoint, so nothing can continue. The run gets one
            # source-written report naming that end, unless a report is already retained.
            if context.has_retained_report(policy=policy):
                return _resume_state_event(run_id, "completed"), None
            return _original_sft_end(engine.tool_backend, result), None
        return unavailable, None

    stack = ExitStack()
    outcome, engine = unavailable, None
    for number in range(_RESTORE_ATTEMPTS):
        stack = ExitStack()
        try:
            outcome, engine = attempt(stack)
        except BaseException as error:
            stack.close()
            if not isinstance(error, Exception):
                raise
            transient = _restore_error_is_transient(error)
            if transient and number + 1 < _RESTORE_ATTEMPTS:
                _warn_normal_restore_retry(restore_stage, error, number + 1)
                _restore_pause(_restore_backoff(number))
                continue
            _warn_normal_restore_failure(restore_stage, error)
            return _resume_failure_event(run_id, restore_stage, error, transient=transient)
        break
    try:
        if type(outcome) is _OriginalSftEnd:
            # Retained once, outside the replay-safe restore attempts.
            if not continue_completed:
                return unavailable
            return _report_original_sft_end(run_id, session_hash, outcome)
        if engine is None or not continue_completed:
            return outcome if engine is None else unavailable
        continuation = engine.continue_normal_completion
        if not callable(continuation):
            raise ValueError("normal completion continuation unavailable")
        if on_ready is not None:
            on_ready()
        return continuation()
    except Exception as error:
        _warn_normal_restore_failure("continuation", error)
        return _resume_failure_event(run_id, "continuation", error, transient=False)
    finally:
        stack.close()


async def _resume_normal_prompt(payload):
    """Stream the established resume route under existing worker ownership."""
    async for event in _resume_normal_prompt_events(payload):
        yield event


async def _resume_normal_prompt_events(payload):
    """Answer a resume request with its closed state; start the restore in the owned worker.

    The first event is always a closed `resume_state`, emitted before any restore work, so the
    caller's read timeout and drain deadline never decide the outcome. The restore and the
    continuation run in the worker thread under the existing ownership; the stream may end or
    be abandoned at any time without stopping them.
    """
    run_id = payload.get("run_id")
    session_hash = _prompt_report_session_hash()
    if (
        set(payload) != {"action", "run_id"}
        or type(run_id) is not str
        or len(run_id) != 32
        or any(char not in "0123456789abcdef" for char in run_id)
        or (session_hash is None)
    ):
        yield _read_prompt_run(payload)
        return
    envelope = _resume_envelope(run_id)

    import threading

    ready = threading.Event()

    def mark_continuing():
        with _PROMPT_REPORT_LOCK:
            record = _RESUME_WORKERS.get(run_id)
            if record is not None and record["state"] == "restoring":
                record["state"] = "continuing"
        ready.set()

    def work_body():
        return _resume_normal_prompt_body(
            {"action": "resume_prompt_run", "run_id": run_id}, on_ready=mark_continuing
        )

    def admission(busy):
        # Runs under _PROMPT_REPORT_LOCK, atomically with the busy check and the ownership claim.
        record = _RESUME_WORKERS.get(run_id)
        if busy:
            state = (
                record["state"]
                if record is not None and record["state"] in ("restoring", "continuing")
                else None
            )
            return _resume_state_event(run_id, state or "original_active")
        if record is not None:
            if record["state"] == "completed":
                return _resume_state_event(run_id, "completed")
            if (
                record["state"] == "failed"
                and _resume_clock() - record["since"] < _RESUME_FAILED_COOLDOWN_SECONDS
            ):
                return _resume_state_event(run_id, "failed", record["diagnostic"])
        _RESUME_WORKERS[run_id] = {"state": "restoring", "diagnostic": None, "since": _resume_clock()}
        return None

    def finish(box):
        # Runs under _PROMPT_REPORT_LOCK in the worker's final step, atomically with the release
        # of the busy fence: a concurrent request sees either the active state or the final one.
        record = _RESUME_WORKERS.get(run_id)
        status = box.get("status")
        now = _resume_clock()
        if "error_type" in box:
            stage = (
                "continuation"
                if record is not None and record["state"] == "continuing" and "response" not in box
                else "report_retention"
                if "response" in box
                else "run_prepare"
            )
            category = (
                box["error_type"] if box["error_type"] in _RESTORE_ERROR_CATEGORIES.values() else "Exception"
            )
            _RESUME_WORKERS[run_id] = {
                "state": "failed",
                "diagnostic": {"restore_stage": stage, "exception_class": category},
                "since": now,
            }
        elif type(status) is dict and status.get("event") == "resume_state":
            if status.get("state") == "completed":
                _RESUME_WORKERS[run_id] = {"state": "completed", "diagnostic": None, "since": now}
            else:
                _RESUME_WORKERS[run_id] = {
                    "state": status.get("state"),
                    "diagnostic": status.get("diagnostic"),
                    "since": now,
                }
        elif status is not None or "response" not in box:
            # A pending or unqualified projection ended the worker without a continuation:
            # nothing is recorded, so the next request restores again.
            _RESUME_WORKERS.pop(run_id, None)
        else:
            _RESUME_WORKERS[run_id] = {"state": "completed", "diagnostic": None, "since": now}

    async for event in _prompt_worker_events(
        run_id,
        session_hash,
        envelope,
        work_body,
        start_signal=ready,
        admission=admission,
        started_event=_resume_state_event(run_id, "restoring"),
        on_finish=finish,
        error_status=_resume_state_event(
            run_id, "failed", {"restore_stage": "run_prepare", "exception_class": "Exception"}
        ),
        report_resumed=True,
    ):
        yield event


def _retain_prompt_report(
    run_id: str,
    envelope: dict[str, Any],
    response: dict[str, Any],
    session_hash: str | None,
    *,
    _native_report_required: bool = True,
    _resumed: bool = False,
) -> None:
    """Retain the typed report of one worker without execution or new authority.

    `_resumed` is set only by the worker of a resumed (replacement) run: its final report is
    published as the separate single-assignment resumed pointer and supersedes the original
    worker's report; the original pointer and its conflict refusal stay unchanged.
    """
    report = {
        "event": "run_completed",
        "status": "response_received",
        "response": response,
        "worker_finished": True,
        **envelope,
    }
    if type(_native_report_required) is not bool or type(_resumed) is not bool:
        raise TypeError("private report applicability must be source-owned bool DATA")
    try:
        if _native_report_required:
            from .control_plane.normal_run_checkpoint import retain_normal_prompt_report

            # Replay-safe: an existing pointer is verified byte for byte, the object is immutable and an
            # ambiguous pointer write is read back, so only transient classes are repeated (at most three).
            for number in range(_RESTORE_ATTEMPTS):
                try:
                    if _resumed:
                        retain_normal_prompt_report(run_id, session_hash, report, resumed=True)
                    else:
                        retain_normal_prompt_report(run_id, session_hash, report)
                    break
                except Exception as error:
                    if number + 1 >= _RESTORE_ATTEMPTS or not _restore_error_is_transient(error):
                        raise
                    _restore_pause(_restore_backoff(number))
    except Exception:
        # Required configured-native publication cannot masquerade as local durability.
        logging.getLogger(__name__).warning(
            "PROMPT_REPORT_PERSISTENCE_FAILED error_class=native_checkpoint_error"
        )
        raise ValueError("normal report native publication failed") from None
    try:
        from .control_plane.runtime_session import Session, save_session

        metadata = {"normal_prompt_report": report}
        if session_hash is not None:
            metadata["normal_prompt_report_owner"] = {
                "schema_version": 1,
                "session_sha256": session_hash,
            }
        save_session(Session(session_ref=run_id, metadata=metadata))
    except Exception as error:
        if isinstance(error, OSError):
            failure_class = "filesystem_error"
        elif isinstance(error, (TypeError, ValueError)):
            failure_class = "serialization_error"
        else:
            failure_class = "persistence_error"
        try:
            logging.getLogger(__name__).warning(
                "PROMPT_REPORT_PERSISTENCE_FAILED error_class=%s", failure_class
            )
        except Exception:
            # Optional diagnostic failure cannot replace the original response or cleanup.
            pass


_CAMPAIGN_APPROVAL_ACTIONS = frozenset(
    {
        "data_collection_flow",
        "coding_case_freeze_flow",
        "data_preparation_flow",
        "run_processing_job",
        "qwen_sft_flow",
        "coding_generation_flow",
        "coding_pair_flow",
        "acquire_huggingface_dataset",
        "inspect_acquired_dataset",
        "freeze_coding_cases",
        "prepare_coding_dataset",
        "acquire_model_weights",
        "submit_qwen_sft",
        "reload_qwen_sft",
        "submit_coding_generation",
        "resolve_coding_evaluation_spec",
        "submit_coding_pair",
    }
)


def _runtime_prompt_policy():
    """Read explicit deployment approval; public prompt fields grant no authority."""
    import json

    from .control_plane.approvals import ApprovalMode, ApprovalPolicy

    raw = os.environ.get("AMLAS_RUNTIME_APPROVED_ACTIONS", "").strip()
    if not raw:
        return None
    if len(raw.encode()) > 4096:
        raise ValueError("deployment approval configuration exceeds bound")
    try:
        names = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise ValueError("invalid deployment approval configuration") from error
    if (
        type(names) is not list
        or any(type(name) is not str or name not in _CAMPAIGN_APPROVAL_ACTIONS for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("closed campaign deployment action list required")
    if not names:
        return None
    return ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST,
        allowlist=list(names),
        max_auto_cost=None,
        interactive=False,
    )


def _warn_prompt_worker_failure(run_id: str, phase: str, error: BaseException) -> None:
    """Publish only closed worker failure labels independently of the consumer."""
    import asyncio

    categories = {
        ValueError: "ValueError",
        TypeError: "TypeError",
        RuntimeError: "RuntimeError",
        AssertionError: "AssertionError",
        AttributeError: "AttributeError",
        KeyError: "KeyError",
        LookupError: "LookupError",
        OSError: "OSError",
        FileNotFoundError: "FileNotFoundError",
        PermissionError: "PermissionError",
        TimeoutError: "TimeoutError",
        NotImplementedError: "NotImplementedError",
        asyncio.CancelledError: "CancelledError",
        KeyboardInterrupt: "KeyboardInterrupt",
        SystemExit: "SystemExit",
    }
    try:
        if (
            type(run_id) is not str
            or len(run_id) != 32
            or any(character not in "0123456789abcdef" for character in run_id)
            or phase not in {"work_body", "cleanup_report"}
        ):
            return
        category = categories.get(type(error), "Exception")
        source_labels = (
            ("/amlas/app.py", "app"),
            ("/amlas/api.py", "api"),
            ("/amlas/control_plane/campaign_runtime.py", "campaign_runtime"),
            ("/amlas/control_plane/normal_run_checkpoint.py", "normal_run_checkpoint"),
            ("/amlas/tui/chat/engine.py", "chat_engine"),
            ("/amlas/tui/chat/tool_binding_backend.py", "tool_binding_backend"),
            ("/amlas/tui/chat/research_runtime_profile.py", "research_runtime_profile"),
            ("/amlas/config/settings.py", "settings"),
            ("/amlas/config/chat_model.py", "chat_model"),
            ("/amlas/config/aws_region.py", "aws_region"),
            ("/native_factory_entry.py", "native_factory"),
            ("/managed_final_deployment_binding.py", "managed_final_binding"),
            ("/managed_actual_assembly.py", "managed_actual_assembly"),
            ("/automatic_controlled_completion_sender.py", "automatic_sender"),
            ("/automatic_sender.py", "automatic_sender"),
            ("/managed_fixed_stage_publication.py", "managed_fixed_stage"),
            ("/managed_model_sft_publication.py", "managed_model_publisher"),
            ("/managed_runtime_integration.py", "managed_runtime_integration"),
        )
        source_label, source_line = "unknown", 0
        traceback = error.__traceback__
        while traceback is not None:
            filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
            for suffix, label in source_labels:
                if filename.endswith(suffix) and 1 <= traceback.tb_lineno <= 200_000:
                    source_label, source_line = label, traceback.tb_lineno
                    break
            traceback = traceback.tb_next
        logging.getLogger(__name__).warning(
            "normal_worker_failed run_id=%s phase=%s error_class=%s source_label=%s source_line=%s",
            run_id,
            phase,
            category,
            source_label,
            source_line,
            extra={
                "run_id": run_id,
                "worker_phase": phase,
                "exception_class": category,
                "source_label": source_label,
                "source_line": source_line,
            },
        )
    except Exception:
        # Ordinary logger errors cannot replace the original failure or its cleanup.
        pass


async def _prompt_worker_events(
    run_id,
    report_session_hash,
    envelope,
    work_body,
    *,
    entry_context=None,
    diagnostic_slot=None,
    entry_context_getter=None,
    native_report_required_getter=None,
    start_signal=None,
    admission=None,
    started_event=None,
    on_finish=None,
    error_status=None,
    report_resumed=False,
) -> AsyncIterator[dict[str, Any]]:
    """Share existing worker ownership, bounded stream and report retention."""
    import asyncio
    import math
    import threading
    import time
    from contextvars import copy_context

    box: dict[str, Any] = {"entry_context": entry_context}
    done = threading.Event()
    task_app = app
    task_id = None

    def work() -> None:
        try:
            value = work_body()
            if entry_context_getter is not None:
                box["entry_context"] = entry_context_getter()
            if (
                type(value) is dict
                and value.get("event") in ("prompt_run_status", "resume_state")
                and (value.get("run_id") == run_id)
            ):
                box["status"] = value
            else:
                box["response"] = _prompt_response(value)
        except BaseException as error:
            box["error_type"] = type(error).__name__
            _warn_prompt_worker_failure(run_id, "work_body", error)
        finally:
            try:
                task_app.complete_async_task(task_id)
                if "error_type" not in box and "response" in box:
                    current_entry = box.get("entry_context")
                    if current_entry is not None:
                        current_entry.require_current()
                    if "status" not in box:
                        if native_report_required_getter is None:
                            _retain_prompt_report(
                                run_id,
                                envelope,
                                box["response"],
                                report_session_hash,
                                _resumed=report_resumed,
                            )
                        else:
                            _retain_prompt_report(
                                run_id,
                                envelope,
                                box["response"],
                                report_session_hash,
                                _native_report_required=native_report_required_getter(),
                            )
            except BaseException as error:
                box["error_type"] = type(error).__name__
                _warn_prompt_worker_failure(run_id, "cleanup_report", error)
            finally:
                with _PROMPT_REPORT_LOCK:
                    if on_finish is not None:
                        try:
                            on_finish(box)
                        except Exception as error:
                            _warn_prompt_worker_failure(run_id, "cleanup_report", error)
                    done.set()
                    _ACTIVE_PROMPT_REPORTS.discard(run_id)

    worker = threading.Thread(
        target=copy_context().run,
        args=(work,),
        name=f"prompt-run-{run_id}",
        daemon=True,
    )
    started = time.monotonic()
    interval = _env_float("AMLAS_HEARTBEAT_SECONDS", 15.0)
    if not math.isfinite(interval) or interval <= 0:
        interval = 15.0
    limit = _env_float("AMLAS_MAX_RUN_SECONDS", 0.0)
    deadline = started + limit if math.isfinite(limit) and limit > 0 else None
    with _PROMPT_REPORT_LOCK:
        busy = run_id in _ACTIVE_PROMPT_REPORTS
        answer = admission(busy) if admission is not None else None
        if answer is None and not busy:
            _ACTIVE_PROMPT_REPORTS.add(run_id)
    if answer is not None:
        yield answer
        return
    if busy:
        yield {
            "event": "prompt_run_status",
            "status": "unavailable",
            "run_id": run_id,
            "found": False,
        }
        return
    try:
        task_id = task_app.add_async_task("normal_prompt")
        worker.start()
    except Exception as error:
        try:
            if task_id is not None:
                task_app.complete_async_task(task_id)
        finally:
            with _PROMPT_REPORT_LOCK:
                if on_finish is not None:
                    try:
                        on_finish({"error_type": type(error).__name__})
                    except Exception as finish_error:
                        _warn_prompt_worker_failure(run_id, "cleanup_report", finish_error)
                _ACTIVE_PROMPT_REPORTS.discard(run_id)
        yield {
            "event": "run_completed",
            "status": "failed",
            "error_type": type(error).__name__,
            "worker_finished": True,
            **envelope,
        }
        return
    announced = start_signal is None
    if started_event is not None:
        yield dict(started_event)
    if announced:
        yield {"event": "run_started", **envelope}
    try:
        last_heartbeat = started
        heartbeats = 0
        while True:
            if not announced and start_signal.is_set() and not (done.is_set() and "status" in box):
                announced = True
                yield {"event": "run_started", **envelope}
            if done.is_set():
                if "error_type" not in box and box.get("entry_context") is not None:
                    try:
                        box["entry_context"].require_current()
                    except Exception as error:
                        box["error_type"] = type(error).__name__
                if "status" in box:
                    if "error_type" in box:
                        yield error_status or {
                            "event": "prompt_run_status",
                            "status": "unavailable",
                            "run_id": run_id,
                            "found": False,
                        }
                    else:
                        yield box["status"]
                    return
                if "error_type" in box:
                    yield {
                        "event": "run_completed",
                        "status": "failed",
                        "error_type": box["error_type"],
                        "worker_finished": True,
                        **envelope,
                    }
                else:
                    yield {
                        "event": "run_completed",
                        "status": "response_received",
                        "response": box["response"],
                        "worker_finished": True,
                        **envelope,
                    }
                return
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                yield {
                    "event": "run_completed",
                    "status": "outcome_unknown",
                    "reason": "wall_clock_exceeded",
                    "worker_finished": False,
                    **envelope,
                }
                return
            if not announced:
                # Refused/pending restores retain their exact closed projection.
                await asyncio.sleep(min(interval, 0.01))
                continue
            diagnostic = diagnostic_slot.take_current() if diagnostic_slot is not None else None
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                yield {
                    "event": "run_completed",
                    "status": "outcome_unknown",
                    "reason": "wall_clock_exceeded",
                    "worker_finished": False,
                    **envelope,
                }
                return
            if diagnostic is not None:
                if done.is_set():
                    continue
                current_entry = entry_context
                if current_entry is not None:
                    try:
                        current_entry.require_current()
                    except Exception as error:
                        from .tui.chat.tool_binding_backend import _warm_error_class

                        logging.getLogger(__name__).debug(
                            "MODEL_DIAGNOSTIC_PUBLICATION_DROP phase=stream_entry error_class=%s",
                            _warm_error_class(error),
                        )
                        diagnostic = None
                if done.is_set():
                    continue
                if diagnostic is None or not diagnostic_slot._is_active():
                    continue
                yield {
                    "event": "model_acquisition_diagnostic",
                    "diagnostic": diagnostic,
                    "worker_finished": False,
                    **envelope,
                }
            if now - last_heartbeat >= interval:
                heartbeats += 1
                last_heartbeat = now
                yield {
                    "event": "heartbeat",
                    "elapsed_seconds": round(now - started, 1),
                    "n": heartbeats,
                    **envelope,
                }
            pause = min(interval, 1.0)
            if deadline is not None:
                pause = min(pause, max(0.0, deadline - time.monotonic()))
            await asyncio.sleep(pause)
    finally:
        if diagnostic_slot is not None:
            diagnostic_slot.close_consumer()


async def _prompt_events(prompt: str, *, entry_context=None) -> AsyncIterator[dict[str, Any]]:
    """Keep the stream alive while the existing API processes one exact prompt."""
    import asyncio
    import uuid
    from contextlib import nullcontext

    from .api import _prompt_model_diagnostic_scope, _PromptModelDiagnosticSlot, run_prompt
    from .control_plane.campaign_runtime import activate_campaign_entry, campaign_prompt_scope
    from .control_plane.normal_run_checkpoint import activate_normal_run, prepare_normal_run

    run_id = uuid.uuid4().hex
    transport_session_id, report_session_hash = _prompt_session_identity()
    envelope = {
        "mode": "prompt",
        "run_id": run_id,
        "response_kind": "conversation",
        "state_scope": "single_invocation",
        "durable_resume": False,
        "approval_policy_source": "default_noninteractive",
        "lifecycle_verified": False,
        "quality_success": False,
    }
    try:
        policy = _runtime_prompt_policy()
    except ValueError:
        envelope["approval_policy_source"] = "invalid_deployment_policy"
        yield {"event": "run_started", **envelope}
        yield {
            "event": "run_completed",
            "status": "failed",
            "error_type": "ValueError",
            "worker_finished": True,
            **envelope,
        }
        return
    if policy is not None:
        envelope["approval_policy_source"] = "deployment_allowlist"
    try:
        if transport_session_id is not None:
            normal_context = prepare_normal_run(
                run_id, report_session_hash, transport_session_id=transport_session_id
            )
        else:
            normal_context = prepare_normal_run(run_id, report_session_hash)
    except ValueError:
        yield {"event": "run_started", **envelope}
        yield {
            "event": "run_completed",
            "status": "failed",
            "error_type": "ValueError",
            "worker_finished": True,
            **envelope,
        }
        return
    diagnostic_slot = _PromptModelDiagnosticSlot()
    worker_entry = {}

    def work_body():  # nosemgrep: useless-inner-function
        activation = activate_campaign_entry(entry_context) if entry_context is not None else nullcontext()
        with (
            activation,
            activate_normal_run(normal_context),
            campaign_prompt_scope(prompt, policy=policy) as current,
            _prompt_model_diagnostic_scope(diagnostic_slot),
        ):
            response = run_prompt(prompt, policy=policy) if policy is not None else run_prompt(prompt)
            value = asyncio.run(response)
            if current is not None:
                current.require_current()
                worker_entry["current"] = current
            return value

    def native_report_required():  # nosemgrep: useless-inner-function
        # The original server context, not public/model response DATA, selects this.
        return normal_context is not None and normal_context._report_retention_applicable()

    async for event in _prompt_worker_events(
        run_id,
        report_session_hash,
        envelope,
        work_body,
        entry_context=entry_context,
        diagnostic_slot=diagnostic_slot,
        entry_context_getter=lambda: worker_entry.get("current", entry_context),
        native_report_required_getter=native_report_required,
        on_finish=lambda box: _record_finished_original(run_id, normal_context, box),
    ):
        yield event


@app.entrypoint
async def agent_invocation(payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """Entry point invoked by AgentCore Runtime.

    Streams orchestrator status events. On completion yields the final ArbiterReport
    (or budget_exhausted summary).
    """
    # Layered-config: overlay the AgentCore/Strands runtime ``.env.agentcore``
    # (runtime ARN, OTEL/STRANDS telemetry toggles, per-role model tiers, boto
    # retry/timeout) on top of the shared-base root ``.env`` BEFORE the Lead
    # hierarchy / Bedrock models are built below. Idempotent + shell-wins (see
    # ``config/env_loader``); best-effort so it never blocks a Runtime invoke.
    try:
        from .config.env_loader import load_layered_env

        load_layered_env("agentcore")
    except Exception:  # nosec B110
        pass

    if payload.get("action") == "get_prompt_report":
        yield await _read_off_loop(_read_prompt_report, payload, "prompt_report")
        return

    if payload.get("action") == "resume_prompt_run":
        async for event in _resume_normal_prompt(payload):
            yield event
        return
    if payload.get("action") == "get_prompt_run_status":
        yield await _read_off_loop(_read_prompt_run, payload, "prompt_run_status")
        return

    from .control_plane.campaign_coordinator import _WAKE, parse_wake_hint
    from .control_plane.campaign_runtime import current_campaign_runtime

    wake_mode = (
        "spec_path" not in payload
        and "spec_text" not in payload
        and "campaign_ref" in payload
        and "prompt" not in payload
    )
    prompt_mode = "spec_path" not in payload and "spec_text" not in payload and "prompt" in payload
    if prompt_mode:
        if set(payload) != {"prompt"}:
            yield {"event": "error", "error_code": "unsupported_prompt_fields"}
            return
        if type(payload["prompt"]) is not str or not payload["prompt"].strip():
            yield {"event": "error", "error_code": "invalid_prompt"}
            return

    # Deferred imports — Runtime cold-start latency matters; imports only fire on first invoke.

    # : point the process-wide error surface at a real sink at entrypoint start so
    # the 7-agent backend's errors are FUNNELED (local JSONL always + CloudWatch mirror in
    # this managed mode) instead of only printed. Best-effort — never blocks a Runtime invoke.
    try:
        from .telemetry.error_log import init_error_log

        init_error_log()
    except Exception:  # nosec B110
        pass

    from .cloud.secrets import resolve_secret_into_env

    # The three ``build_*_lead`` imports are gone: ``harness.leads.lead_for`` resolves the
    # local builder through the lead registry, so importing them here would be a second path
    # to the same factory.
    region = resolve_aws_region()

    # B3 mechanism: pull TAVILY_API_KEY from Secrets Manager into the env BEFORE
    # the research agents are built. No-op (and no failure) until Phase 14c
    # populates the secret → research degrades to arxiv + Semantic Scholar.
    try:
        applied = resolve_secret_into_env(region=region)
        # Phase 17 instrumentation: the streamed `tavily_configured` event is
        # buffered by AgentCore until the (multi-minute) generator finishes, so
        # it is NOT visible in CloudWatch mid-run. Emit a flush-print marker
        # (same pattern as N5's ARBITER_GRADED) so the key pickup is provable
        # in-cloud. NEVER print the key — only configured/not + a masked prefix.
        _tav = os.environ.get("TAVILY_API_KEY", "")
        print(
            f"TAVILY_SECRET: configured={bool(_tav)} newly_applied={applied}",
            flush=True,
        )
        if applied:
            yield {"event": "tavily_configured", "source": "secrets_manager"}
    except Exception as e:  # never let secret resolution break a run
        print(f"TAVILY_SECRET: error {type(e).__name__}", flush=True)
        yield {"event": "tavily_unconfigured", "detail": f"{type(e).__name__}: {e}"}

    if wake_mode:
        factory = current_campaign_runtime()
        if set(payload) != set(_WAKE) or factory is None:
            yield {"event": "error", "error_code": "unsupported_private_wake"}
            return
        import json

        try:
            hint = parse_wake_hint(json.dumps(payload, allow_nan=False).encode())
            outcome = factory.begin_wake(hint, policy=_runtime_prompt_policy())
        except (ValueError, TypeError):
            yield {"event": "error", "error_code": "private_wake_refused"}
            return
        if outcome.disposition != "continue_original":
            yield {
                "event": "run_completed",
                "status": "awaiting_original",
                "campaign_ref": outcome.campaign_ref,
                "operation_digest": outcome.operation_digest,
                "job_arn": outcome.job_arn,
                "native_status": outcome.native_status,
                "lifecycle_verified": False,
                "quality_success": False,
            }
            return
        async for event in _prompt_events(outcome.context.literal_prompt, entry_context=outcome.context):
            yield event
        return

    if prompt_mode:
        async for event in _prompt_events(payload["prompt"]):
            yield event
        return

    from .orchestrator import Orchestrator
    from .spec import load_spec_from_path, load_spec_from_text

    if "spec_path" in payload:
        spec = load_spec_from_path(payload["spec_path"])
    elif "spec_text" in payload:
        spec = load_spec_from_text(payload["spec_text"])
    else:
        yield {
            "event": "error",
            "error": "payload must include 'spec_path' or 'spec_text'",
        }
        return

    yield {"event": "spec_loaded", "project_id": spec.project_id}

    # N1/N2: stage the dataset + materialize an aligned y_true on the container.
    data_paths: dict[str, Any] | None = None
    staged_y_true_path: str | None = None
    if payload.get("stage_data"):
        from .cloud.staging import StagingError, stage_dataset

        try:
            staged = stage_dataset(
                spec,
                region=region,
                max_rows=payload.get("max_rows"),
            )
        except StagingError as e:
            yield {"event": "error", "error": f"data staging failed: {e}"}
            return
        data_paths = {
            "train_path": staged.train_path,
            "test_path": staged.test_path,
            "target": staged.target,
        }
        staged_y_true_path = staged.y_true_path
        # Publish the staged pair so ``dispatch_preflight``'s staged_test_alignment check can find
        # it with no argument from the agent. y_true is materialised once here and never
        # regenerated, so the check exists to catch a later in-place rewrite of test.csv before a
        # job is dispatched against it.
        os.environ["AMLAS_STAGED_TEST_PATH"] = staged.test_path
        os.environ["AMLAS_STAGED_Y_TRUE_PATH"] = staged.y_true_path
        yield {
            "event": "data_staged",
            "train_path": staged.train_path,
            "test_path": staged.test_path,
            "y_true_path": staged.y_true_path,
            "target": staged.target,
            "n_train": staged.n_train,
            "n_test": staged.n_test,
            "source": staged.source,
        }

        # N1.b: the runtime PROCESS fs (where data_integrity's pandas tools read)
        # and the Code Interpreter SANDBOX fs (where tabular_fe/trainer_tuner run
        # via execute_python_tool) are SEPARATE. stage_dataset wrote only the
        # process fs, so the sandbox sees "no data file" (CW 2026-05-29 13:43).
        # Mirror the staged CSVs into the hoisted sandbox session at the SAME
        # paths so the agents' execute_python_tool calls find them. Best-effort:
        # if the sandbox can't be reached we still yield an event and let D1
        # decide — never crash the run.
        try:
            from .cloud.sandbox_staging import stage_files_into_sandbox

            sandbox_result = stage_files_into_sandbox(
                {
                    staged.train_path: staged.train_path,
                    staged.test_path: staged.test_path,
                },
                region=region,
            )
            yield {
                "event": "data_staged_sandbox",
                "ok": all(sandbox_result.values()),
                "results": sandbox_result,
            }
        except Exception as e:
            yield {
                "event": "data_staged_sandbox",
                "ok": False,
                "detail": f"{type(e).__name__}: {e}",
            }

    # Build the Lead hierarchy. ``lead_for`` routes each Lead to its managed AgentCore harness
    # when that Lead is opted in via ``AMLAS_HARNESS_LEADS`` AND its harness ARN resolves
    # (ADR-0007 phase 1); otherwise it builds exactly the same in-process Agent as before, with
    # real Bedrock models. With the flag unset this is byte-for-byte the previous behaviour —
    # the cutover is a decision, never a side effect of HarnessStack being deployed.
    #
    # ``actor_id`` is the managed-memory scope: facts namespace under
    # ``/actors/{actorId}/facts/``, so passing the project id is what makes long-term memory
    # carry over per project instead of being unscoped. ``run_id`` is deliberately NOT passed:
    # the Orchestrator owns run-id generation and is constructed below, and each adapter is
    # created per invocation anyway, so its session ledger is already per-run.
    from .harness.leads import HarnessLead, lead_for

    research_lead = lead_for("research_lead", actor_id=spec.project_id)
    fe_lead = lead_for("fe_lead", actor_id=spec.project_id)
    ml_lead = lead_for("ml_execution_lead", actor_id=spec.project_id)
    # Emit which Leads are harness-hosted, so a run's transcript records where each one ran
    # rather than leaving it to be inferred from the environment afterwards.
    for _lead in (research_lead, fe_lead, ml_lead):
        if isinstance(_lead, HarnessLead):
            yield {"event": "lead_harness_hosted", "lead": _lead.lead_name}

    # N2: resolve y_true — prefer the staged (aligned) one; else the operator's
    # explicit path (local OR s3://). The Arbiter needs this for grading.
    y_true_arg = payload.get("y_true_path") or staged_y_true_path
    if not y_true_arg:
        yield {
            "event": "error",
            "error": "payload must include 'y_true_path' (or set stage_data=true)",
        }
        return

    from .cloud.sandbox_staging import load_predictions
    from .cloud.staging import load_y_true_npy

    def _y_true_loader() -> Any:
        # Local and S3 labels use the shared bounded data-only loader.
        # String labels are Unicode arrays; object-pickle compatibility is forbidden.
        return load_y_true_npy(str(y_true_arg), region=region)

    def _y_pred_loader(path: str) -> Any:
        # N5: the ML Lead's trainer_tuner writes y_pred INSIDE the Code
        # Interpreter SANDBOX fs (e.g. /tmp/amlas/y_pred_test.csv), but this
        # loader runs in the runtime PROCESS fs — two separate filesystems
        # (CW 2026-05-29 14:04:44 FileNotFoundError). load_predictions egresses
        # the file OUT of the sandbox (reverse of N1.b) when it's absent on the
        # process fs, and is format-agnostic (.npy via np.load, .csv via a robust
        # parser) so the Arbiter can grade whatever the agent emitted.
        # Instrumentation (N5 verification): these prints land in CloudWatch so
        # the sandbox→process bridge + Arbiter grading are observable server-side.
        print(f"N5_YPRED_LOADER: loading y_pred from {path!r}", flush=True)
        preds = load_predictions(path, egress=True, region=region)
        try:
            n = len(preds)
        except TypeError:
            n = -1
        print(
            f"N5_YPRED_LOADED: ok n={n} dtype={getattr(preds, 'dtype', '?')}",
            flush=True,
        )
        return preds

    orch = Orchestrator(
        spec=spec,
        research_lead=research_lead,
        fe_lead=fe_lead,
        ml_lead=ml_lead,
        y_true_loader=_y_true_loader,
        y_pred_loader=_y_pred_loader,
        data_paths=data_paths,  # N1: thread concrete train/test/target into the problem card
    )
    yield {"event": "run_started", "run_id": orch.run_id}

    # FIX-15 (Subtask-5 durable server-side termination guard) — the decisive fix.
    #
    # Root cause of the Subtask-4 stall AND the Subtask-5 v2 idle-close: this
    # entrypoint used to call the SYNCHRONOUS ``orch.run()`` inline, so the async
    # generator yielded NOTHING between ``run_started`` and the final
    # ``run_completed``. On a long agent run that gap spans many minutes; AgentCore
    # then (a) never sees a terminal event if the loop hangs (Subtask-4: client
    # blocks to read_timeout with 0 usable events) or (b) closes the idle SSE
    # stream (Subtask-5 v2: client got a clean EOF at ~128s, no terminal event,
    # while the server was still working).
    #
    # The fix runs ``orch.run()`` on a background worker thread (via the event
    # loop's default executor — a SINGLE consistent thread for the whole run, so
    # the nested Strands + Code-Interpreter sandbox execution keeps its thread
    # context, exactly as when run() ran inline) while the generator:
    #   1. emits periodic ``heartbeat`` events so the SSE stream never goes idle;
    #   2. ALWAYS yields a terminal ``run_completed`` — when run() finishes with
    #      its real result, OR when the HARD wall-clock deadline elapses first
    #      (status=budget_exhausted, reason=wall_clock_exceeded), then returns.
    # The worker is a daemon thread; the Runtime session ends when this generator
    # returns, so an overrun run() cannot keep the container alive.
    #
    # IMPLEMENTATION NOTE: we deliberately use a DAEMON ``threading.Thread`` (not
    # ``loop.run_in_executor`` — whose default ThreadPoolExecutor threads are
    # NON-daemon and would make ``asyncio.run()`` / the process BLOCK on join
    # against a hung ``orch.run()`` at exit, keeping the container alive forever).
    # A daemon thread lets the generator return and the Runtime session end even
    # when the run overran. We poll the result box with ``asyncio.sleep`` so the
    # event loop stays free to emit heartbeats.
    import asyncio
    import threading
    import time as _time

    # FIX-26: the deadline is OPT-IN, not a default cap. Precedence:
    #   1. AMLAS_MAX_RUN_SECONDS (ops escape hatch + deterministic sub-minute test deadline);
    #   2. else a POSITIVE spec ``wall_clock_minutes`` the operator explicitly set;
    #   3. else UNBOUNDED — the run goes to natural completion (``hard_deadline = None``).
    # NO absolute ceiling clamps the operator's number. The heartbeat + terminal-event
    # guarantee are infra scaffolding (SSE keep-alive), independent of any work cap.
    run_start = _time.monotonic()
    _explicit_secs = _env_float("AMLAS_MAX_RUN_SECONDS", 0.0)
    spec_minutes = float(getattr(spec.compute_budget, "wall_clock_minutes", 0) or 0)
    if _explicit_secs > 0:
        max_run_seconds: float | None = _explicit_secs
    elif spec_minutes > 0:
        max_run_seconds = spec_minutes * 60.0
    else:
        max_run_seconds = None  # unbounded — run to natural completion
    hard_deadline = (run_start + max_run_seconds) if max_run_seconds is not None else None
    heartbeat_interval = _env_float("AMLAS_HEARTBEAT_SECONDS", 15.0)
    poll_interval = min(heartbeat_interval, 1.0)
    print(
        f"FIX26_TERMINATION_GUARD: wall_clock_minutes={spec.compute_budget.wall_clock_minutes} "
        f"max_run_seconds={'unbounded' if max_run_seconds is None else f'{max_run_seconds:.0f}'} "
        f"heartbeat_seconds={heartbeat_interval}",
        flush=True,
    )

    run_box: dict[str, Any] = {}

    def _run_worker() -> None:
        try:
            run_box["result"] = orch.run()
        except BaseException as e:  # capture so the generator can report it as a terminal error
            run_box["error"] = e
        finally:
            run_box["done"] = True

    worker = threading.Thread(target=_run_worker, name=f"orch-run-{orch.run_id}", daemon=True)
    worker.start()

    def _terminal_from_result(res: Any) -> dict[str, Any]:
        return {
            "event": "run_completed",
            "status": res.status,
            "iterations": res.iterations,
            "arbiter_report": (res.arbiter_report.model_dump(mode="json") if res.arbiter_report else None),
            "kanban_path": str(orch.run_dir / "kanban.json"),
        }

    def _last_chance_arbiter_report() -> dict[str, Any] | None:
        """FIX-20c (Subtask-13): grade the DURABLE y_pred mirror when the run is abandoned.

        On the HARD wall-clock deadline (or a run-level error) the daemon ``orch.run()`` is
        abandoned mid-iteration, so the orchestrator's own durable-mirror grade (FIX-20b) never
        runs and the terminal event would carry ``arbiter_report: null`` even though the agent
        already produced a real y_pred that FIX-20 mirrored to the process fs (string-label classification run #6
        on v8: CV f1_macro=0.7942, y_pred mirrored x2, yet arbiter_report=null because the 50-min
        deadline fired while run() was still executing). Here — from the entrypoint, on the SAME
        durable mirror — we grade that y_pred so the terminal event carries a REAL arbiter_report
        (G-METRIC/G-A3). Best-effort: NEVER raises, returns ``None`` when there is no durable
        y_pred or grading is not possible (the guard's terminal event is emitted regardless)."""
        try:
            from .orchestrator import _durable_ypred_path

            path = _durable_ypred_path()
            if not path:
                return None
            y_true = _y_true_loader()
            y_pred = _y_pred_loader(path)
            # orch._grade is the deterministic module ``grade`` (ADR-003) — same decision the
            # in-loop arbiter would make; identical, non-delegated.
            report = orch._grade(spec=orch.spec, y_true=y_true, y_pred=y_pred, phase_log=[])
            print(
                f"FIX20C_LASTCHANCE_GRADED: overall_score={report.overall_score} "
                f"direction={report.direction} passes_acceptance={report.passes_acceptance} "
                f"(graded the durable y_pred mirror on the abandoned-run terminal path)",
                flush=True,
            )
            return report.model_dump(mode="json")  # type: ignore[no-any-return]
        except Exception as e:  # pragma: no cover — last-chance grade must never break the terminal
            print(f"FIX20C_LASTCHANCE_SKIPPED: {type(e).__name__}: {e}", flush=True)
            return None

    heartbeats = 0
    last_heartbeat = run_start
    terminal_emitted = False
    while True:
        if run_box.get("done"):
            # run() finished (or raised) within budget — emit its REAL outcome.
            if "error" in run_box:
                err = run_box["error"]
                print(f"FIX15_RUN_ERROR: {type(err).__name__}: {err}", flush=True)
                # : funnel the terminal run error into the centralized surface
                # (local JSONL + CloudWatch mirror), classified by the taxonomy so the
                # terminal ``reason`` carries a stable error code. Best-effort.
                try:
                    from .errors import classify
                    from .telemetry.error_log import capture_error

                    classified = classify(err, source="agents.orchestrator")
                    capture_error(
                        err if isinstance(err, BaseException) else Exception(str(err)),
                        source="agents.orchestrator",
                        error_type=classified.code,
                        run_id=str(getattr(orch, "run_id", "") or ""),
                        context={
                            "category": str(classified.category),
                            "retryable": classified.retryable,
                        },
                    )
                except Exception:  # nosec B110
                    pass
                yield {
                    "event": "run_completed",
                    "status": "failed",
                    "reason": f"{type(err).__name__}: {err}",
                    "iterations": 0,
                    # FIX-20c: even on a run-level error, if the agent already produced a
                    # y_pred that FIX-20 mirrored durably, grade it so the terminal carries
                    # a REAL arbiter_report instead of null.
                    "arbiter_report": _last_chance_arbiter_report(),
                    "kanban_path": str(orch.run_dir / "kanban.json"),
                }
            else:
                yield _terminal_from_result(run_box["result"])
            terminal_emitted = True
            break

        now = _time.monotonic()
        if hard_deadline is not None and now >= hard_deadline:
            # HARD wall-clock deadline hit while run() is still executing. Emit a
            # graded terminal event so the client ALWAYS receives a terminal event
            # (never a hung/idle-closed stream). The daemon worker is abandoned;
            # the Runtime session ends when this generator returns.
            print(
                f"FIX15_WALL_CLOCK_EXCEEDED: emitting terminal budget_exhausted after "
                f"{heartbeats} heartbeats (run still executing)",
                flush=True,
            )
            # FIX-20c: the daemon run() is abandoned mid-iteration, so the orchestrator's
            # own durable-mirror grade (FIX-20b) never runs. Grade the durable y_pred mirror
            # HERE so the terminal event carries a REAL arbiter_report (G-METRIC/G-A3) when the
            # agent already produced a y_pred that FIX-20 mirrored to the process fs.
            yield {
                "event": "run_completed",
                "status": "budget_exhausted",
                "reason": "wall_clock_exceeded",
                "iterations": orch.spec.compute_budget.max_retreats + 1,
                "arbiter_report": _last_chance_arbiter_report(),
                "kanban_path": str(orch.run_dir / "kanban.json"),
            }
            terminal_emitted = True
            break

        # Emit a heartbeat when the interval elapses (keeps the SSE stream alive).
        if now - last_heartbeat >= heartbeat_interval:
            heartbeats += 1
            last_heartbeat = now
            yield {
                "event": "heartbeat",
                "elapsed_seconds": round(now - run_start, 1),
                "n": heartbeats,
            }

        # Yield control back to the event loop briefly, then re-check. Unbounded runs
        # (no deadline) just sleep the poll interval between heartbeat/result checks.
        if hard_deadline is None:
            await asyncio.sleep(poll_interval)
        else:
            await asyncio.sleep(min(poll_interval, max(0.0, hard_deadline - _time.monotonic())))

    # Defensive belt: guarantee a terminal event was emitted (should always be).
    if not terminal_emitted:  # pragma: no cover — unreachable by construction
        yield {
            "event": "run_completed",
            "status": "budget_exhausted",
            "reason": "guard_fallthrough",
            "iterations": 0,
            "arbiter_report": None,
            "kanban_path": str(orch.run_dir / "kanban.json"),
        }


if __name__ == "__main__":  # pragma: no cover — entered only in AgentCore Runtime container
    app.run()
