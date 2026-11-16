"""Neutral recorded transport fakes; no SDK, execution or grading authority."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from amlas.evaluation.coding_executor import invoke_recorded

REQUEST = {
    "codeInterpreterIdentifier": "fixture-interpreter",
    "sessionId": "fixture-session",
    "name": "executeCode",
    "arguments": {"language": "python", "clearContext": False, "code": "print(1)"},
}
BINDING = {"evaluation_id": "fixture-evaluation", "case_id": "HumanEval/0", "arm": "base", "ordinal": 0}
RESULT = {
    "result": {
        "content": [{"type": "text", "text": "1\n"}],
        "structuredContent": {"stdout": "1\n", "stderr": "", "exitCode": 0},
        "isError": False,
    }
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class Fake:
    def __init__(self, frames=None, call_error=None, stream_error=None, session=None):
        self.records, self.timeline, self.calls = [], [], []
        self.frames = [copy.deepcopy(RESULT)] if frames is None else frames
        self.call_error, self.stream_error = call_error, stream_error
        self.session = session or REQUEST["sessionId"]
        self.iterations = 0

    def persist(self, record):
        self.timeline.append("persist:" + record["kind"])
        self.records.append(copy.deepcopy(record))

    def invoke_code_interpreter(self, **request):
        assert self.records[0]["kind"] == "attempt", "attempt missing before native invoke"
        self.timeline.append("invoke")
        self.calls.append(copy.deepcopy(request))
        if self.call_error is not None:
            raise self.call_error
        return {
            "sessionId": self.session,
            "ResponseMetadata": {"RequestId": "fixture-request"},
            "stream": self.stream(),
        }

    def stream(self):
        assert any(record["kind"] == "response" for record in self.records), (
            "response missing before stream iteration"
        )
        self.timeline.append("iterate")
        self.iterations += 1
        yield from self.frames
        if self.stream_error is not None:
            raise self.stream_error


def run(fake, *, persist=None, deadline=10.0, now=1.0):
    return invoke_recorded(
        fake,
        copy.deepcopy(REQUEST),
        binding=copy.deepcopy(BINDING),
        persist=persist or fake.persist,
        deadline=deadline,
        clock=lambda: now,
    )


def happy(fake):
    try:
        return run(fake)
    except (NotImplementedError, ValueError, TimeoutError) as error:
        pytest.fail("recorded invocation happy path missing: " + type(error).__name__)


def assert_unknown(fake):
    assert len(fake.calls) == 1
    unknown = [record for record in fake.records if record["kind"] == "unknown"]
    assert unknown and unknown[-1]["session_id"] == REQUEST["sessionId"]
    assert unknown[-1]["binding"] == BINDING
    assert not any(record["kind"] == "completed" for record in fake.records)


def assert_deadline_prefix(fake, kinds):
    assert len(fake.calls) == 1
    assert [row["kind"] for row in fake.records] == kinds
    assert all(
        row["session_id"] == REQUEST["sessionId"] and row["binding"] == BINDING for row in fake.records
    )
    assert not any(row["kind"] in ("unknown", "completed") for row in fake.records)


def test_attempt_response_event_completion_order_and_hash_only_records():
    fake = Fake()
    result = happy(fake)
    assert (
        fake.timeline.index("persist:attempt")
        < fake.timeline.index("invoke")
        < fake.timeline.index("persist:response")
        < fake.timeline.index("iterate")
    )
    assert isinstance(result, (tuple, list)) and list(result) == [RESULT]
    assert [record["kind"] for record in fake.records] == ["attempt", "response", "event", "completed"]
    attempt = fake.records[0]
    assert attempt["code_sha256"] == hashlib.sha256(b"print(1)").hexdigest()
    assert attempt["request_sha256"] == digest(REQUEST)
    assert attempt["session_id"] == REQUEST["sessionId"] and attempt["binding"] == BINDING
    assert attempt["deadline"] == 10.0 and isinstance(attempt["time"], (float, int))
    assert fake.records[1]["request_id"] == "fixture-request"
    assert fake.records[2]["index"] == 0 and fake.records[2]["sha256"] == digest(RESULT)
    assert isinstance(fake.records[2]["byte_count"], int)
    assert "print(1)" not in json.dumps(fake.records) and "code" not in attempt
    assert fake.records[-1].get("quality_success") is False
    assert fake.calls == [REQUEST]


def test_attempt_persistence_failure_means_zero_native_invokes():
    fake = Fake()

    def persist(_record):
        raise OSError("synthetic sync failure before call")

    with pytest.raises(OSError):
        run(fake, persist=persist)
    assert fake.calls == [] and fake.iterations == 0


def test_elapsed_deadline_means_zero_native_invokes():
    fake = Fake()
    with pytest.raises(TimeoutError):
        run(fake, deadline=1.0, now=1.0)
    assert fake.calls == [] and fake.iterations == 0


@pytest.mark.parametrize("phase", ["invoke", "stream"])
def test_interruption_retains_original_unknown_identity_without_retry(phase):
    failure = ConnectionError("synthetic interrupted observation")
    fake = (
        Fake(call_error=failure)
        if phase == "invoke"
        else Fake(frames=[{"progress": {"text": "step"}}], stream_error=failure)
    )
    with pytest.raises(ConnectionError):
        run(fake)
    assert_unknown(fake)
    if phase == "stream":
        assert any(record["kind"] == "event" for record in fake.records)


def test_post_response_event_persistence_failure_is_unknown_without_retry():
    fake = Fake()

    def persist(record):
        if record["kind"] == "event":
            raise OSError("synthetic event sync failure")
        fake.persist(record)

    with pytest.raises(OSError):
        run(fake, persist=persist)
    assert_unknown(fake)
    assert any(record["kind"] == "response" for record in fake.records)


def test_wrong_response_session_cannot_complete_or_iterate():
    fake = Fake(session="different-session")
    with pytest.raises(ValueError):
        run(fake)
    assert_unknown(fake)
    assert fake.iterations == 0


@pytest.mark.parametrize("limit", ["frame_bytes", "event_count"])
def test_stream_bounds_fail_with_partial_records_and_no_completion(limit):
    frames = [{"progress": {"text": "step"}}]
    frames += (
        [{"progress": {"text": "a" * (1024**2 + 1)}}]
        if limit == "frame_bytes"
        else [{"progress": {"text": "step"}}] * 1024
    )
    fake = Fake(frames=frames)
    with pytest.raises(ValueError):
        run(fake)
    assert_unknown(fake)
    assert any(record["kind"] == "event" for record in fake.records)


def test_cancellation_records_unknown_and_reraises_original_without_retry():
    fake = Fake(stream_error=KeyboardInterrupt("synthetic cancelled observation"))
    with pytest.raises(KeyboardInterrupt):
        run(fake)
    assert_unknown(fake)


class CountedStream:
    """Ordinary finite frames; count each attempted read without blocking."""

    def __init__(self, frames):
        self.frames = iter(frames)
        self.reads = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.reads += 1
        return next(self.frames)


@pytest.mark.parametrize("phase", ["invoke", "response_persist", "event_persist"])
def test_deadline_is_checked_before_starting_each_stream_read(phase):
    now = [1.0]
    stream = CountedStream([copy.deepcopy(RESULT), copy.deepcopy(RESULT)])

    class TimedFake(Fake):
        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            response["stream"] = stream
            if phase == "invoke":
                now[0] = 10.0
            return response

    fake = TimedFake()

    def persist(record):
        fake.persist(record)
        if (phase, record["kind"]) in (("response_persist", "response"), ("event_persist", "event")):
            now[0] = 10.0

    with pytest.raises(TimeoutError):
        invoke_recorded(
            fake,
            copy.deepcopy(REQUEST),
            binding=copy.deepcopy(BINDING),
            persist=persist,
            deadline=10.0,
            clock=lambda: now[0],
        )
    assert stream.reads == (1 if phase == "event_persist" else 0)
    assert_deadline_prefix(fake, ["attempt", "response"] + (["event"] if phase == "event_persist" else []))
    assert len(fake.calls) == 1


@pytest.mark.parametrize("available", [64, 65])
def test_event_allowance_is_checked_before_any_sixty_fifth_read(available):
    stream = CountedStream([{"progress": {"text": "step"}} for _ in range(available)])

    class CountedFake(Fake):
        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            response["stream"] = stream
            return response

    fake = CountedFake()
    with pytest.raises(ValueError, match="count"):
        run(fake)
    assert stream.reads == 64
    events = [record for record in fake.records if record["kind"] == "event"]
    assert len(events) == 64 and [record["index"] for record in events] == list(range(64))
    assert_unknown(fake)
    assert fake.records[-1]["event_count"] == 64


@pytest.mark.parametrize("phase", ["invoke", "stream"])
def test_absolute_deadline_interrupts_actual_blocking_native_boundary(phase):
    import time

    class DelayedStream:
        closed = 0

        def __iter__(self):
            return self

        def __next__(self):
            time.sleep(1.0)
            return copy.deepcopy(RESULT)

        def close(self):
            self.closed += 1

    stream = DelayedStream()

    class DelayedFake(Fake):
        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            if phase == "invoke":
                time.sleep(1.0)
            response["stream"] = stream
            return response

    fake = DelayedFake()
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        invoke_recorded(
            fake,
            copy.deepcopy(REQUEST),
            binding=copy.deepcopy(BINDING),
            persist=fake.persist,
            deadline=started + 0.2,
            clock=time.monotonic,
        )
    assert time.monotonic() - started < 0.7
    assert_deadline_prefix(fake, ["attempt"] + (["response"] if phase == "stream" else []))
    assert stream.closed == (1 if phase == "stream" else 0)


def test_deadline_restores_existing_handler_after_success_and_cancellation():
    import signal

    original = signal.getsignal(signal.SIGALRM)

    def handler(_number, _frame):
        raise AssertionError("unrelated alarm must not run")

    signal.signal(signal.SIGALRM, handler)
    try:
        happy(Fake())
        assert signal.getsignal(signal.SIGALRM) is handler
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        fake = Fake(stream_error=KeyboardInterrupt("neutral cancellation"))
        with pytest.raises(KeyboardInterrupt):
            run(fake)
        assert_unknown(fake)
        assert signal.getsignal(signal.SIGALRM) is handler
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)


def test_unrelated_active_alarm_refuses_before_attempt_and_native_call():
    import signal

    fake = Fake()
    signal.setitimer(signal.ITIMER_REAL, 5.0, 7.0)
    try:
        with pytest.raises(RuntimeError, match="alarm"):
            run(fake)
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 4.0 < remaining <= 5.0 and interval == 7.0
        assert fake.calls == fake.records == []
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def test_non_main_thread_refuses_before_attempt_and_native_call():
    import threading

    fake = Fake()
    failures = []

    def call():
        try:
            run(fake)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=call)
    thread.start()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert len(failures) == 1 and isinstance(failures[0], RuntimeError)
    assert fake.calls == fake.records == []


def test_unsupported_platform_refuses_before_attempt_and_native_call(monkeypatch):
    import sys

    fake = Fake()
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux"):
        run(fake)
    assert fake.calls == fake.records == []


def test_blocked_deadline_signal_refuses_before_attempt_and_native_call():
    import signal

    fake = Fake()
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        with pytest.raises(RuntimeError, match="blocked"):
            run(fake)
        assert fake.calls == fake.records == []
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def test_original_stream_is_closed_once_on_native_cancellation():
    class OriginalStream:
        closes = 0

        def __iter__(self):
            return self

        def __next__(self):
            raise KeyboardInterrupt("neutral original stream cancellation")

        def close(self):
            self.closes += 1

    stream = OriginalStream()

    class StreamFake(Fake):
        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            response["stream"] = stream
            return response

    fake = StreamFake()
    with pytest.raises(KeyboardInterrupt):
        run(fake)
    assert_unknown(fake)
    assert stream.closes == 1


def test_owned_nested_deadline_never_extends_original_alarm_and_restores_on_timeout():
    import signal
    import time

    from amlas.evaluation.coding_executor import absolute_deadline

    original = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        with absolute_deadline(started + 0.2, clock=time.monotonic):
            with absolute_deadline(started + 10.0, clock=time.monotonic):
                time.sleep(1.0)
    assert time.monotonic() - started < 0.7
    assert signal.getsignal(signal.SIGALRM) is original
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_stream_retained_by_parent_is_not_closed_outside_its_cleanup_owner():
    class Stream:
        closes = 0

        def __iter__(self):
            return iter([copy.deepcopy(RESULT)])

        def close(self):
            self.closes += 1

    stream = Stream()

    class StreamFake(Fake):
        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            response["stream"] = stream
            return response

    fake, retained = StreamFake(), []
    assert invoke_recorded(
        fake,
        copy.deepcopy(REQUEST),
        binding=copy.deepcopy(BINDING),
        persist=fake.persist,
        deadline=10.0,
        clock=lambda: 1.0,
        retain_stream=retained.append,
    ) == (RESULT,)
    assert retained == [stream] and stream.closes == 0


@pytest.mark.parametrize("phase", ["arm", "disarm", "restore_handler"])
def test_owned_signal_at_guard_setup_or_teardown_is_translated_and_restored(monkeypatch, phase):
    import signal
    import time

    from amlas.evaluation import coding_executor as executor

    original = signal.getsignal(signal.SIGALRM)
    real_timer, real_handler = signal.setitimer, signal.signal
    fired, entered = [], []

    def timer(which, seconds, interval=0.0):
        if not fired and (
            (phase == "arm" and seconds > 0)
            or (phase == "disarm" and seconds == 0 and executor._ACTIVE_DEADLINE is not None)
        ):
            fired.append(phase)
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
        return real_timer(which, seconds, interval)

    def handler(number, callback):
        if (
            phase == "restore_handler"
            and not fired
            and callback is original
            and signal.getsignal(number) is not original
        ):
            fired.append(phase)
            signal.getsignal(number)(number, None)
        return real_handler(number, callback)

    monkeypatch.setattr(signal, "setitimer", timer)
    monkeypatch.setattr(signal, "signal", handler)
    try:
        with pytest.raises(TimeoutError):
            with executor.absolute_deadline(time.monotonic() + 1.0, clock=time.monotonic):
                entered.append(True)
        assert entered == ([] if phase == "arm" else [True])
        assert signal.getsignal(signal.SIGALRM) is original
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        assert executor._ACTIVE_DEADLINE is None
    finally:
        real_timer(signal.ITIMER_REAL, 0.0)
        real_handler(signal.SIGALRM, original)
        executor._ACTIVE_DEADLINE = None


@pytest.mark.parametrize("failure", [KeyboardInterrupt("neutral cancel"), SystemExit(7)])
def test_owned_late_signal_does_not_swallow_unrelated_cancellation(monkeypatch, failure):
    import signal
    import time

    from amlas.evaluation import coding_executor as executor

    original = signal.getsignal(signal.SIGALRM)
    real_timer = signal.setitimer
    fired = []

    def timer(which, seconds, interval=0.0):
        if not fired and seconds == 0 and executor._ACTIVE_DEADLINE is not None:
            fired.append(True)
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
        return real_timer(which, seconds, interval)

    monkeypatch.setattr(signal, "setitimer", timer)
    try:
        with pytest.raises(type(failure)) as caught:
            with executor.absolute_deadline(time.monotonic() + 1.0, clock=time.monotonic):
                raise failure
        assert caught.value is failure
        assert signal.getsignal(signal.SIGALRM) is original
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        assert executor._ACTIVE_DEADLINE is None
    finally:
        real_timer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, original)
        executor._ACTIVE_DEADLINE = None


def test_unknown_persistence_uses_only_remaining_original_deadline_and_keeps_prefix():
    import time

    fake = Fake(
        frames=[{"progress": {"text": "neutral observation"}}],
        stream_error=ConnectionError("neutral interrupted stream"),
    )
    started = time.monotonic()

    def persist(record):
        if record["kind"] == "unknown":
            time.sleep(1.0)
        fake.persist(record)

    with pytest.raises(TimeoutError):
        invoke_recorded(
            fake,
            copy.deepcopy(REQUEST),
            binding=copy.deepcopy(BINDING),
            persist=persist,
            deadline=started + 0.2,
            clock=time.monotonic,
        )
    assert time.monotonic() - started < 0.7
    assert_deadline_prefix(fake, ["attempt", "response", "event"])


@pytest.mark.parametrize("failure", [KeyboardInterrupt("neutral cancellation"), SystemExit(9)])
def test_unknown_persistence_does_not_swallow_unrelated_cancellation(failure):
    fake = Fake(call_error=ConnectionError("neutral original failure"))

    def persist(record):
        if record["kind"] == "unknown":
            raise failure
        fake.persist(record)

    with pytest.raises(type(failure)) as caught:
        run(fake, persist=persist)
    assert caught.value is failure
    assert_deadline_prefix(fake, ["attempt"])


def test_ordinary_native_error_after_deadline_does_not_start_unknown_persistence():
    now = [1.0]
    error = ConnectionError("neutral late observation")

    class LateFake(Fake):
        def invoke_code_interpreter(self, **request):
            self.calls.append(copy.deepcopy(request))
            now[0] = 10.0
            raise error

    fake = LateFake()
    with pytest.raises(ConnectionError) as caught:
        invoke_recorded(
            fake,
            copy.deepcopy(REQUEST),
            binding=copy.deepcopy(BINDING),
            persist=fake.persist,
            deadline=10.0,
            clock=lambda: now[0],
        )
    assert caught.value is error
    assert_deadline_prefix(fake, ["attempt"])


def test_ordinary_unknown_sink_failure_retains_original_error_and_unresolved_prefix():
    original = ConnectionError("neutral original observation failure")
    fake = Fake(call_error=original)

    def persist(record):
        if record["kind"] == "unknown":
            raise OSError("neutral sink interruption")
        fake.persist(record)

    with pytest.raises(ConnectionError) as caught:
        run(fake, persist=persist)
    assert caught.value is original
    assert_deadline_prefix(fake, ["attempt"])


@pytest.mark.parametrize("is_error", [False, True])
def test_completed_nonzero_native_execution_is_distinct_from_transport_unknown(is_error):
    from amlas.evaluation import coding_executor as executor

    frame = copy.deepcopy(RESULT)
    frame["result"]["isError"] = is_error
    frame["result"]["structuredContent"] = {
        "stdout": "",
        "stderr": "neutral candidate failure",
        "exitCode": 1,
    }
    fake = Fake(frames=[frame])
    with pytest.raises(executor.CandidateExecutionError) as caught:
        run(fake)
    assert caught.value.exit_code == 1
    assert str(caught.value) == "candidate execution returned a nonzero native status"
    assert len(fake.calls) == 1
    assert [x["kind"] for x in fake.records] == ["attempt", "response", "event", "unknown"]
    assert "neutral candidate failure" not in json.dumps(fake.records)


@pytest.mark.parametrize("exit_code", [None, True, 0, 256, -256, "1"])
def test_service_error_without_closed_nonzero_execution_status_stays_unknown(exit_code):
    frame = copy.deepcopy(RESULT)
    frame["result"]["isError"] = True
    frame["result"]["structuredContent"]["exitCode"] = exit_code
    fake = Fake(frames=[frame])
    with pytest.raises(ValueError):
        run(fake)
    assert_unknown(fake)


@pytest.mark.parametrize("field,value", [("stdout", None), ("stderr", None), ("exitCode", True)])
def test_malformed_candidate_status_never_becomes_runtime_error(field, value):
    frame = copy.deepcopy(RESULT)
    frame["result"]["isError"] = True
    frame["result"]["structuredContent"]["exitCode"] = 1
    frame["result"]["structuredContent"][field] = value
    fake = Fake(frames=[frame])
    with pytest.raises(ValueError):
        run(fake)
    assert_unknown(fake)


def test_transport_client_error_keeps_only_closed_service_metadata():
    from botocore.exceptions import ClientError

    fake = Fake(
        call_error=ClientError(
            {
                "Error": {"Code": "ThrottlingException", "Message": "withheld service body"},
                "ResponseMetadata": {"HTTPStatusCode": 429, "RequestId": "fixture-error-request"},
            },
            "InvokeCodeInterpreter",
        )
    )
    with pytest.raises(ClientError):
        run(fake)
    assert_unknown(fake)
    assert fake.records[-1]["service_error"] == {
        "code": "ThrottlingException",
        "http_status": 429,
        "request_id": "fixture-error-request",
    }
    assert "withheld service body" not in json.dumps(fake.records)
