"""Bounded original CodeInterpreter invocation recording; no grading authority."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import signal
import stat
import sys
import threading
from contextlib import contextmanager

MAX_CODE_BYTES = 128 * 1024
MAX_EVENT_BYTES = 65536
MAX_EVENTS = 64


class CandidateExecutionError(Exception):
    """Closed failed native execution, never an oracle verdict or transport retry."""

    def __init__(self, exit_code):
        if type(exit_code) is not int or exit_code == 0 or not -255 <= exit_code <= 255:
            raise ValueError("closed nonzero native execution status required")
        self.exit_code = exit_code
        super().__init__("candidate execution returned a nonzero native status")


def _validate_service_error(value):
    if type(value) is not dict or not value or not set(value) <= {"code", "http_status", "request_id"}:
        raise ValueError("closed native service error metadata required")
    for key in ("code", "request_id"):
        if key in value and (
            type(value[key]) is not str or not re.fullmatch("[A-Za-z0-9_-]{1,128}", value[key])
        ):
            raise ValueError("closed native service error label required")
    if "http_status" in value and (
        type(value["http_status"]) is not int or not 100 <= value["http_status"] <= 599
    ):
        raise ValueError("closed native service error HTTP status required")
    return value


def _service_error_metadata(error):
    # Only typed SDK ClientError supplies service metadata. Messages/bodies are discarded.
    from botocore.exceptions import ClientError

    if not isinstance(error, ClientError):
        return {}
    response = error.response
    if type(response) is not dict:
        return {}
    details, transport = response.get("Error", {}), response.get("ResponseMetadata", {})
    value = {}
    if type(details) is dict:
        code = details.get("Code")
        if type(code) is str and re.fullmatch("[A-Za-z0-9_-]{1,128}", code):
            value["code"] = code
    if type(transport) is dict:
        status, identity = transport.get("HTTPStatusCode"), transport.get("RequestId")
        if type(status) is int and 100 <= status <= 599:
            value["http_status"] = status
        if type(identity) is str and re.fullmatch("[A-Za-z0-9_-]{1,128}", identity):
            value["request_id"] = identity
    return {"service_error": value} if value else {}


def _canonical(value):
    pending = [(value, 0)]
    count = 0
    characters = 0
    while pending:
        part, depth = pending.pop()
        count += 1
        if count > 10000 or depth > 16:
            raise ValueError("coding frame item/depth bound exceeded")
        kind = type(part)
        if kind is str:
            characters += len(part)
            if characters > 4 * MAX_CODE_BYTES or len(part) > MAX_CODE_BYTES:
                raise ValueError("coding frame string bound exceeded")
        elif kind is dict:
            if len(part) > 10000 or any(type(key) is not str for key in part):
                raise ValueError("coding frame mapping differs")
            pending.extend((item, depth + 1) for pair in part.items() for item in pair)
        elif kind is list:
            if len(part) > 10000:
                raise ValueError("coding frame item bound exceeded")
            pending.extend((item, depth + 1) for item in part)
        elif kind is int:
            if part.bit_length() > 4096:
                raise ValueError("coding frame integer bound exceeded")
        elif kind is float:
            if not math.isfinite(part):
                raise ValueError("coding frame nonfinite value")
        elif kind not in (bool, type(None)):
            raise ValueError("coding frame non-data value")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


_ACTIVE_DEADLINE = None


class _CodingDeadlineExpired(BaseException):
    """Private interruption bypasses SDK Exception/retry handlers."""


def require_deadline_context():
    """Fail closed before native effects when the fixed interrupt path is absent."""
    if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("coding deadlines require the fixed Linux main thread")
    if signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK, set()):
        raise RuntimeError("coding deadline signal is blocked")
    if _ACTIVE_DEADLINE is None and signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("unrelated active alarm prevents coding deadline ownership")


@contextmanager
def absolute_deadline(deadline, *, clock):
    """Interrupt synchronous SDK/socket/iterator waits on the fixed Linux worker.

    An owned nested guard reuses the original alarm and never extends its deadline.
    Unrelated timers are refused, and the previous handler/timer are restored on
    success, timeout and cancellation. This does not grant native containment.
    """
    global _ACTIVE_DEADLINE
    require_deadline_context()
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or not callable(clock):
        raise ValueError("finite original deadline and clock required")
    now = clock()
    if type(now) not in (int, float) or not math.isfinite(now) or now >= deadline:
        raise TimeoutError("original absolute coding deadline elapsed")
    if _ACTIVE_DEADLINE is not None:
        owner_deadline, owner_clock = _ACTIVE_DEADLINE
        if clock is not owner_clock or deadline < owner_deadline:
            raise RuntimeError("nested coding deadline differs from original owner")
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def expire(_number, _frame):
        raise _CodingDeadlineExpired()

    failure = restoration_failure = None
    try:
        signal.signal(signal.SIGALRM, expire)
        _ACTIVE_DEADLINE = (deadline, clock)
        signal.setitimer(signal.ITIMER_REAL, deadline - now)
        yield
    except BaseException as error:
        # Preserve unrelated cancellation while translating only our sentinel.
        failure = error
    finally:
        try:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            finally:
                try:
                    _ACTIVE_DEADLINE = None
                    signal.signal(signal.SIGALRM, previous_handler)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        except _CodingDeadlineExpired as error:
            # A one-shot owned alarm can arrive at the disarm/restore boundary.
            # Finish restoration once; no native operation is repeated.
            restoration_failure = error
            _ACTIVE_DEADLINE = None
            signal.signal(signal.SIGALRM, previous_handler)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    if failure is not None and not isinstance(failure, _CodingDeadlineExpired):
        raise failure
    interruption = failure if failure is not None else restoration_failure
    if interruption is not None:
        raise TimeoutError("original absolute coding deadline interrupted blocking call") from interruption


def close_original_streams(streams):
    """Close each original response handle once inside its owner's cleanup guard."""
    closed, first_error = set(), None
    for stream in streams:
        identity = id(stream)
        if identity in closed:
            continue
        closed.add(identity)
        try:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def invoke_recorded(client, request, *, binding, persist, deadline, clock, retain_stream=None):
    """Record one interruptible original invocation without retry or grading.

    A trusted parent may retain original streams for its one bounded cleanup.
    Otherwise this seam closes them under a separate thirty-second cleanup cap.
    """
    if retain_stream is not None and not callable(retain_stream):
        raise ValueError("trusted original stream retention callback required")
    streams = []
    try:
        with absolute_deadline(deadline, clock=clock):
            return _invoke_recorded(
                client,
                request,
                binding=binding,
                persist=persist,
                deadline=deadline,
                clock=clock,
                retain_stream=retain_stream if retain_stream is not None else streams.append,
            )
    finally:
        if retain_stream is None and streams:
            try:
                with absolute_deadline(clock() + 30, clock=clock):
                    close_original_streams(streams)
            except Exception:
                # Stream-close failure cannot turn transport data into cleanup proof.
                pass


def _invoke_recorded(client, request, *, binding, persist, deadline, clock, retain_stream):
    """Record one native attempt and bounded response; never retry or grade it.

    ``persist`` is a trusted host sink that synchronously commits each record.
    Transport completion is data only, not proof of completed oracle assertions.
    """
    if (
        type(request) is not dict
        or set(request) != {"codeInterpreterIdentifier", "sessionId", "name", "arguments"}
        or request["name"] != "executeCode"
        or type(request["codeInterpreterIdentifier"]) is not str
        or (
            request["codeInterpreterIdentifier"] != "aws.codeinterpreter.v1"
            and not re.fullmatch("[A-Za-z0-9_-]{1,128}", request["codeInterpreterIdentifier"])
        )
        or type(request["sessionId"]) is not str
        or not re.fullmatch("[A-Za-z0-9_-]{1,128}", request["sessionId"])
        or type(request["arguments"]) is not dict
        or set(request["arguments"]) != {"language", "clearContext", "code"}
        or request["arguments"]["language"] != "python"
        or request["arguments"]["clearContext"] is not False
        or type(request["arguments"]["code"]) is not str
        or not 0 < len(request["arguments"]["code"].encode()) <= MAX_CODE_BYTES
        or type(binding) is not dict
        or set(binding) != {"evaluation_id", "case_id", "arm", "ordinal"}
        or type(binding["evaluation_id"]) is not str
        or not re.fullmatch("[A-Za-z0-9_-]{1,128}", binding["evaluation_id"])
        or type(binding["case_id"]) is not str
        or not re.fullmatch("HumanEval/(?:[0-9]|[1-9][0-9]|1[0-5][0-9]|16[0-3])", binding["case_id"])
        or binding["arm"] not in ("base", "tuned")
        or type(binding["ordinal"]) is not int
        or not 0 <= binding["ordinal"] <= 512
        or type(deadline) not in (int, float)
        or not math.isfinite(deadline)
        or not callable(persist)
        or not callable(clock)
    ):
        raise ValueError("closed original coding invocation required")
    original = copy.deepcopy(request)
    context = {
        **copy.deepcopy(binding),
        "binding": copy.deepcopy(binding),
        "session_id": original["sessionId"],
        "interpreter_id": original["codeInterpreterIdentifier"],
    }
    now = clock()
    if type(now) not in (int, float) or not math.isfinite(now) or now >= deadline:
        raise TimeoutError("coding case deadline elapsed before invocation")
    request_sha = hashlib.sha256(_canonical(original)).hexdigest()
    code_sha = hashlib.sha256(original["arguments"]["code"].encode()).hexdigest()
    persist(
        {
            **context,
            "kind": "attempt",
            "request_sha256": request_sha,
            "code_sha256": code_sha,
            "time": now,
            "deadline": deadline,
        }
    )
    invoked = False
    events = []
    try:
        if clock() >= deadline:
            raise TimeoutError("coding case deadline elapsed after attempt")
        invoked = True
        response = client.invoke_code_interpreter(**original)
        if type(response) is dict and "stream" in response:
            retain_stream(response["stream"])
        if (
            type(response) is not dict
            or response.get("sessionId") != original["sessionId"]
            or "stream" not in response
        ):
            raise ValueError("original coding response session/framing differs")
        metadata = response.get("ResponseMetadata", {})
        request_id = metadata.get("RequestId") if type(metadata) is dict else None
        if request_id is not None and (
            type(request_id) is not str or len(request_id) > 128 or any(ord(char) < 32 for char in request_id)
        ):
            raise ValueError("native request identity differs")
        persist(
            {
                **context,
                "kind": "response",
                "request_sha256": request_sha,
                "request_id": request_id,
                "time": clock(),
                "deadline": deadline,
            }
        )
        total = 0
        if clock() >= deadline:
            raise TimeoutError("coding case response deadline elapsed")
        stream = iter(response["stream"])
        while True:
            if clock() >= deadline:
                raise TimeoutError("coding case response deadline elapsed before read")
            if len(events) >= MAX_EVENTS:
                raise ValueError("coding event count allowance exhausted before read")
            try:
                event = next(stream)
            except StopIteration:
                break
            if clock() >= deadline:
                raise TimeoutError("coding case response deadline elapsed")
            index = len(events)
            if type(event) is not dict or len(event) != 1:
                raise ValueError("coding event union differs")
            raw = _canonical(event)
            total += len(raw)
            if len(raw) > MAX_EVENT_BYTES or total > MAX_EVENT_BYTES * MAX_EVENTS:
                raise ValueError("coding event byte bound exceeded")
            event_kind = next(iter(event))
            if type(event_kind) is not str or not re.fullmatch("[A-Za-z][A-Za-z0-9]{0,63}", event_kind):
                raise ValueError("coding event kind differs")
            persist(
                {
                    **context,
                    "kind": "event",
                    "index": index,
                    "event_kind": event_kind,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_count": len(raw),
                    "time": clock(),
                    "deadline": deadline,
                }
            )
            events.append(copy.deepcopy(event))
        if clock() >= deadline or len(events) != 1 or set(events[0]) != {"result"}:
            raise ValueError("coding result incomplete or deadline elapsed")
        result = events[0]["result"]
        structured = result.get("structuredContent") if type(result) is dict else None
        if (
            type(result) is not dict
            or type(result.get("isError")) is not bool
            or type(structured) is not dict
            or type(structured.get("exitCode")) is not int
        ):
            raise ValueError("coding native execution incomplete")
        if structured["exitCode"] != 0:
            if not -255 <= structured["exitCode"] <= 255 or any(
                type(structured.get(name)) is not str or len(structured[name].encode()) > MAX_EVENT_BYTES
                for name in ("stdout", "stderr")
            ):
                raise ValueError("coding native execution incomplete")
            raise CandidateExecutionError(structured["exitCode"])
        if result["isError"] is not False:
            raise ValueError("coding native execution incomplete")
        persist(
            {
                **context,
                "kind": "completed",
                "request_sha256": request_sha,
                "event_count": len(events),
                "event_byte_count": total,
                "time": clock(),
                "deadline": deadline,
                "oracle_completed": False,
                "quality_success": False,
            }
        )
        return tuple(events)
    except BaseException as error:
        if invoked and not isinstance(error, _CodingDeadlineExpired):
            try:
                unknown_time = clock()
                if (
                    type(unknown_time) in (int, float)
                    and math.isfinite(unknown_time)
                    and unknown_time < deadline
                ):
                    persist(
                        {
                            **context,
                            "kind": "unknown",
                            "request_sha256": request_sha,
                            "event_count": len(events),
                            "error_type": type(error).__name__,
                            **_service_error_metadata(error),
                            "time": unknown_time,
                            "deadline": deadline,
                            "oracle_completed": False,
                            "quality_success": False,
                        }
                    )
            except Exception:
                # Keep the original failure on ordinary sink error; deadline and
                # unrelated cancellation must propagate rather than be swallowed.
                pass
        # Expiry retains its durable prefix unresolved, without another write.
        raise


_RECORD_COMMON = {
    "evaluation_id",
    "case_id",
    "arm",
    "ordinal",
    "binding",
    "session_id",
    "interpreter_id",
    "kind",
    "time",
    "deadline",
}
_RECORD_FIELDS = {
    "attempt": {"request_sha256", "code_sha256"},
    "response": {"request_sha256", "request_id"},
    "event": {"index", "event_kind", "sha256", "byte_count"},
    "completed": {
        "request_sha256",
        "event_count",
        "event_byte_count",
        "oracle_completed",
        "quality_success",
    },
    "unknown": {
        "request_sha256",
        "event_count",
        "error_type",
        "oracle_completed",
        "quality_success",
    },
}
MAX_RECORD_BYTES = 16384


def _record_bytes(record):
    raw = _canonical(record)
    if len(raw) > MAX_RECORD_BYTES or type(record) is not dict:
        raise ValueError("bounded hash-only invocation record required")
    record = json.loads(raw)
    kind = record.get("kind")
    if type(kind) is not str or kind not in _RECORD_FIELDS:
        raise ValueError("invocation record kind differs")
    optional = {"service_error"} if kind == "unknown" else set()
    if (
        not _RECORD_COMMON | _RECORD_FIELDS[kind] <= set(record)
        or not set(record) <= _RECORD_COMMON | _RECORD_FIELDS[kind] | optional
    ):
        raise ValueError("closed hash-only invocation record required")
    if "service_error" in record:
        _validate_service_error(record["service_error"])
    binding = record["binding"]
    if type(binding) is not dict or set(binding) != {"evaluation_id", "case_id", "arm", "ordinal"}:
        raise ValueError("invocation record binding differs")
    if _canonical(binding) != _canonical({key: record[key] for key in binding}):
        raise ValueError("invocation record original binding differs")
    for key in ("evaluation_id", "session_id", "interpreter_id"):
        if type(record[key]) is not str or (
            not (key == "interpreter_id" and record[key] == "aws.codeinterpreter.v1")
            and not re.fullmatch("[A-Za-z0-9_-]{1,128}", record[key])
        ):
            raise ValueError("invocation record identity differs")
    if (
        type(record["case_id"]) is not str
        or not re.fullmatch("HumanEval/(?:[0-9]|[1-9][0-9]|1[0-5][0-9]|16[0-3])", record["case_id"])
        or record["arm"] not in ("base", "tuned")
        or type(record["ordinal"]) is not int
        or not 0 <= record["ordinal"] <= 512
    ):
        raise ValueError("invocation record case binding differs")
    for key in ("time", "deadline"):
        if type(record[key]) not in (int, float) or not math.isfinite(record[key]):
            raise ValueError("invocation record time differs")
    for key in ("request_sha256", "code_sha256", "sha256"):
        if key in record and (type(record[key]) is not str or not re.fullmatch("[0-9a-f]{64}", record[key])):
            raise ValueError("invocation record hash differs")
    for key in ("request_id", "event_kind", "error_type"):
        if (
            key in record
            and record[key] is not None
            and (type(record[key]) is not str or not re.fullmatch("[A-Za-z0-9_-]{1,128}", record[key]))
        ):
            raise ValueError("invocation record transport label differs")
    for key, bound in (
        ("index", MAX_EVENTS - 1),
        ("event_count", MAX_EVENTS),
        ("byte_count", MAX_EVENT_BYTES),
        ("event_byte_count", MAX_EVENTS * MAX_EVENT_BYTES),
    ):
        if key in record and (type(record[key]) is not int or not 0 <= record[key] <= bound):
            raise ValueError("invocation record count differs")
    for key in ("oracle_completed", "quality_success"):
        if key in record and record[key] is not False:
            raise ValueError("invocation records grant no oracle or quality authority")
    return raw, kind


class FileInvocationRecorder:
    """Synchronously persist hash-only receipts in one trusted empty run directory.

    The caller owns directory creation, access policy and lifecycle. Existing or
    partial files are never reused, removed or rewritten. A failed sync raises;
    a later unknown record may still describe that original interrupted call.
    """

    def __init__(self, directory):
        self._lock = threading.RLock()
        self._next = 0
        self._fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if os.listdir(self._fd):
                raise ValueError("invocation run directory must be empty")
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()

    def close(self):
        with self._lock:
            if self._fd is not None:
                fd, self._fd = self._fd, None
                os.close(fd)

    def __call__(self, record):
        raw, kind = _record_bytes(record)
        with self._lock:
            if self._fd is None:
                raise ValueError("invocation recorder is closed")
            if self._next >= 1000000:
                raise ValueError("invocation run record bound exceeded")
            name = f"{self._next:08d}-{kind}.json"
            self._next += 1
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=self._fd,
            )
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("invocation receipt must be an exclusive regular file")
                os.fchmod(fd, 0o600)
                remaining = memoryview(raw)
                while remaining:
                    count = os.write(fd, remaining)
                    if count <= 0:
                        raise OSError("invocation receipt write made no progress")
                    remaining = remaining[count:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(self._fd)
