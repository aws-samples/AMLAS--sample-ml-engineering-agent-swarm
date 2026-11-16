"""Qualifier time is admission work: it is credited back to the returned body's wall-clock allowance.

The 120-second allowance of a returned preparation body bounds stream, decode and consumer work. The caller's
qualifier runs about 62,000 times during one lineage pass and its cost grows with the paused conversation and the
pending originals, so it must not spend that allowance. Every qualification call site stays. Neutral stand-ins only.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import threading
import time
from types import SimpleNamespace

import pytest

from amlas.backends.adapters import coding_preparation as prep
from amlas.backends.types import JobState

BACKEND = prep.ProcessingCodingPreparationBackend
BUCKET, ACCOUNT, VERSION = "ordinary-bucket", "ordinary-account", "ordinary-version"
LINEAGE_KEY = "ordinary/prefix/LINEAGE.jsonl"
RAW_KEY = "ordinary/prefix/train.jsonl"
LOGGER = prep.__name__


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    # The module reads only time.monotonic(); the virtual clock drives every deadline check.
    monkeypatch.setattr(prep, "time", SimpleNamespace(monotonic=value.monotonic))
    return value


class Body(io.BytesIO):
    """Returned stream whose reads cost virtual I/O seconds."""

    def __init__(self, raw, clock, read_cost=0.0):
        super().__init__(raw)
        self.clock, self.read_cost, self.reads = clock, read_cost, 0

    def read(self, size=-1):
        self.reads += 1
        self.clock.advance(self.read_cost)
        return super().read(size)


class Qualifier:
    """Admission stand-in: it passes, and every call costs virtual seconds."""

    def __init__(self, clock, cost):
        self.clock, self.cost, self.calls = clock, cost, 0

    def __call__(self):
        self.calls += 1
        self.clock.advance(self.cost)


def s3_for(raw, clock, key, read_cost=0.0):
    body = Body(raw, clock, read_cost)

    def get_object(**kwargs):
        assert kwargs["Bucket"] == BUCKET and kwargs["Key"] == key and kwargs["VersionId"] == VERSION
        return {"VersionId": VERSION, "ContentLength": len(raw), "Body": body}

    return SimpleNamespace(get_object=get_object), body


def rows(count, pad=0):
    return b"".join(
        json.dumps({"ordinary": index, "pad": "x" * pad}).encode() + b"\n" for index in range(count)
    )


def lineage(clock, count, *, pad=0, read_cost=0.0):
    raw = rows(count, pad)
    s3, body = s3_for(raw, clock, LINEAGE_KEY, read_cost)
    ref = {
        "uri": "s3://" + BUCKET + "/" + LINEAGE_KEY,
        "version_id": VERSION,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "format": "jsonl",
    }
    catalog = SimpleNamespace(bucket=BUCKET, account_id=ACCOUNT)
    return BACKEND(s3_client=s3, sagemaker_client=object()), s3, catalog, ref, body


def test_admission_time_does_not_spend_the_lineage_body_allowance(clock):
    backend, s3, catalog, ref, body = lineage(clock, 300)
    qualify = Qualifier(clock, 0.5)
    records = list(backend._stream_lineage_objects(s3, catalog, ref, qualify))
    assert [record["ordinary"] for record in records] == list(range(300))
    assert qualify.calls * qualify.cost > 120 and body.closed


def test_scope_callback_time_inside_lineage_and_reads_is_credited(clock):
    backend, s3, catalog, ref, body = lineage(clock, 300)
    qualify = Qualifier(clock, 0.5)
    with backend.qualified_io(qualify):
        records = list(backend._lineage_objects(s3, catalog, ref))
    assert len(records) == 300 and qualify.calls * qualify.cost > 120 and body.closed


def test_raw_read_credits_the_scope_callback(clock):
    raw = b"ordinary\n" * (3 * 1024 * 1024 // 9 + 7)
    s3, body = s3_for(raw, clock, RAW_KEY)
    backend = BACKEND(s3_client=s3, sagemaker_client=object())
    qualify = Qualifier(clock, 40.0)
    digest = hashlib.sha256(raw).hexdigest()
    with backend.qualified_io(qualify):
        _data, actual, lines = backend._raw(s3, BUCKET, RAW_KEY, VERSION, ACCOUNT, len(raw), digest)
    assert actual == digest and lines == raw.count(b"\n")
    assert qualify.calls * qualify.cost > 120 and body.closed


def test_slow_body_reads_still_exhaust_the_allowance(clock):
    backend, s3, catalog, ref, body = lineage(clock, 3000, pad=100, read_cost=70.0)
    with pytest.raises(ValueError, match="lineage metadata deadline exceeded"):
        list(backend._stream_lineage_objects(s3, catalog, ref, Qualifier(clock, 0.0)))
    assert body.closed


def test_slow_raw_reads_still_exhaust_the_allowance(clock):
    raw = b"ordinary\n" * (3 * 1024 * 1024 // 9 + 7)
    s3, body = s3_for(raw, clock, RAW_KEY, read_cost=70.0)
    backend = BACKEND(s3_client=s3, sagemaker_client=object())
    with pytest.raises(ValueError, match="preparation raw I/O deadline exceeded"):
        backend._raw(s3, BUCKET, RAW_KEY, VERSION, ACCOUNT, len(raw), hashlib.sha256(raw).hexdigest())
    assert body.closed


def test_credit_is_bounded_so_the_allowance_stays_finite(clock):
    backend, s3, catalog, ref, body = lineage(clock, 50)
    with pytest.raises(ValueError, match="lineage metadata deadline exceeded"):
        list(backend._stream_lineage_objects(s3, catalog, ref, Qualifier(clock, 700.0)))
    assert body.closed


def test_refusing_qualifier_still_closes_the_body_and_propagates(clock):
    backend, s3, catalog, ref, body = lineage(clock, 50)
    refusal = RuntimeError("ordinary synthetic refusal")
    seen = []

    def qualify():
        seen.append(True)
        if len(seen) > 5:
            raise refusal

    with pytest.raises(RuntimeError) as error:
        list(backend._stream_lineage_objects(s3, catalog, ref, qualify))
    assert error.value is refusal and body.closed


# -- the allowance object: a running admission call is credited while it runs (virtual clock) --


def test_running_admission_call_extends_the_deadline_while_it_runs(clock):
    budget = prep._BodyBudget(clock.now + 120)
    seen = []

    def call():
        clock.advance(500)
        seen.append((budget.expired(), budget.remaining()))

    budget.timed(call)
    assert seen == [(False, 120.0)]
    assert budget.credited == 500 and budget.calls == 1 and not budget.expired()


def test_running_admission_credit_is_capped_by_the_ceiling(clock):
    budget = prep._BodyBudget(clock.now + 120)
    seen = []

    def call():
        clock.advance(1000)
        seen.append(budget.expired())
        clock.advance(4000)
        seen.append(budget.expired())

    budget.timed(call)
    assert seen == [False, True]
    assert budget.credited == prep.QUALIFIER_CREDIT_CEILING_SECONDS and budget.expired()


def test_a_raising_admission_call_is_credited_and_propagates(clock):
    budget = prep._BodyBudget(clock.now + 120)
    refusal = RuntimeError("ordinary synthetic refusal")

    def call():
        clock.advance(30)
        raise refusal

    with pytest.raises(RuntimeError) as error:
        budget.timed(call)
    assert error.value is refusal and budget.credited == 30 and budget.calls == 1


# -- the close timer follows the credited allowance (real time, small allowance) --


class BlockingBody(io.BytesIO):
    """A read that waits until the close timer closes the stream."""

    def __init__(self, closed_at):
        super().__init__(b"")
        self.closed_at, self.gate = closed_at, threading.Event()

    def read(self, size=-1):
        self.gate.wait(15)
        return b""

    def close(self):
        if not self.closed_at:
            self.closed_at.append(time.monotonic())
        self.gate.set()
        super().close()


def held_open(steps):
    """Open a body with a 0.4 s allowance, run the admission steps, block in a read; return the closing delay."""
    backend = BACKEND(s3_client=object(), sagemaker_client=object())
    closed_at = []
    body = BlockingBody(closed_at)
    opened = time.monotonic()
    with backend._owned_observation_body(body, seconds=0.4, join=True) as budget:
        for step in steps:
            budget.timed(step)
        body.read()
    return closed_at[0] - opened


def test_close_timer_without_admission_time_fires_at_the_allowance():
    assert 0.38 <= held_open([]) < 5


def test_close_timer_follows_completed_admission_calls():
    assert 0.85 <= held_open([lambda: time.sleep(0.25), lambda: time.sleep(0.25)]) < 5


def test_close_timer_does_not_fire_inside_a_running_admission_call():
    # One call outlasts the whole 0.4 s allowance; it must be credited while it runs.
    assert 0.85 <= held_open([lambda: time.sleep(0.5)]) < 5


def test_lineage_close_timer_follows_the_credited_allowance(monkeypatch):
    monkeypatch.setattr(BACKEND, "_BODY_SECONDS", 0.4)
    closed_at, entered, started = [], threading.Event(), []

    class Blocking(BlockingBody):
        def read(self, size=-1):
            entered.set()
            return super().read(size)

    body = Blocking(closed_at)

    def get_object(**kwargs):
        started.append(time.monotonic())
        return {"VersionId": VERSION, "ContentLength": 10, "Body": body}

    def qualify():
        if not entered.is_set():
            time.sleep(0.25)

    backend = BACKEND(s3_client=SimpleNamespace(get_object=get_object), sagemaker_client=object())
    catalog = SimpleNamespace(bucket=BUCKET, account_id=ACCOUNT)
    ref = {
        "uri": "s3://" + BUCKET + "/" + LINEAGE_KEY,
        "version_id": VERSION,
        "sha256": "0" * 64,
        "byte_count": 10,
    }
    with pytest.raises(ValueError):
        list(backend._stream_lineage_objects(backend._s3, catalog, ref, qualify))
    # Two qualifier calls run inside the body's scope before the blocked read, 0.25 s each.
    assert 0.85 <= closed_at[0] - started[0] < 5


# -- diagnostics --


def messages_of(caplog):
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


def test_each_body_reports_one_budget_summary(clock, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    backend, s3, catalog, ref, _body = lineage(clock, 40)
    qualify = Qualifier(clock, 0.25)
    list(backend._stream_lineage_objects(s3, catalog, ref, qualify))
    messages = messages_of(caplog)
    assert len(messages) == 1
    found = re.fullmatch(
        r"preparation_body_budget label=lineage calls=(\d+) credited_seconds=(\d+\.\d) wall_seconds=\d+\.\d outcome=ok",
        messages[0],
    )
    # `calls` counts every credited admission invocation (the scope-less read checks cost no virtual time).
    # The first qualification (before the stream is requested) and the last (after the body closed) run outside
    # the body's scope and are not credited.
    assert found and int(found.group(1)) >= qualify.calls - 2 > 40
    assert abs(float(found.group(2)) - (qualify.calls - 2) * 0.25) < 0.1


def test_failed_body_reports_its_exception_class_only(clock, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    backend, s3, catalog, ref, _body = lineage(clock, 3000, pad=100, read_cost=70.0)
    with pytest.raises(ValueError):
        list(backend._stream_lineage_objects(s3, catalog, ref, Qualifier(clock, 0.0)))
    messages = messages_of(caplog)
    assert len(messages) == 1 and messages[0].endswith("outcome=ValueError")
    assert "lineage metadata deadline exceeded" not in messages[0]


def test_cleanup_failure_of_a_successful_body_is_not_reported_as_ok(clock, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    backend, s3, catalog, ref, body = lineage(clock, 5)

    def close():
        io.BytesIO.close(body)
        raise OSError("ordinary synthetic cleanup failure")

    body.close = close
    with pytest.raises(OSError, match="ordinary synthetic cleanup failure"):
        list(backend._stream_lineage_objects(s3, catalog, ref, Qualifier(clock, 0.0)))
    messages = messages_of(caplog)
    assert len(messages) == 1 and messages[0].endswith("outcome=OSError")


def test_deadline_wording_is_on_the_allowlist():
    for wording in ("lineage metadata deadline exceeded", "preparation raw I/O deadline exceeded"):
        assert wording in prep._REASONS


@pytest.mark.parametrize(
    ("error", "expected", "logged"),
    [
        (
            ValueError("lineage metadata deadline exceeded"),
            {"error_type": "ValueError", "reason": "lineage metadata deadline exceeded"},
            "preparation_observation_failed error_type=ValueError reason=lineage metadata deadline exceeded",
        ),
        (
            ValueError("preparation raw I/O deadline exceeded"),
            {"error_type": "ValueError", "reason": "preparation raw I/O deadline exceeded"},
            "preparation_observation_failed error_type=ValueError reason=preparation raw I/O deadline exceeded",
        ),
        (
            ValueError("I/O operation on closed file."),
            {"error_type": "ValueError", "reason": "I/O operation on closed file."},
            "preparation_observation_failed error_type=ValueError reason=I/O operation on closed file.",
        ),
        (
            ValueError("bucket ordinary-bucket differs"),
            {"error_type": "ValueError"},
            "preparation_observation_failed error_type=ValueError",
        ),
        (
            ValueError("invalid path ordinary/prefix/train-jsonl"),
            {"error_type": "ValueError"},
            "preparation_observation_failed error_type=ValueError",
        ),
        (
            ValueError("payload confidential customer text"),
            {"error_type": "ValueError"},
            "preparation_observation_failed error_type=ValueError",
        ),
        (
            ValueError("bucket ordinary-bucket-0123 differs"),
            {"error_type": "ValueError"},
            "preparation_observation_failed error_type=ValueError",
        ),
        (
            OSError("lineage metadata deadline exceeded"),
            {"error_type": "OSError"},
            "preparation_observation_failed error_type=OSError",
        ),
    ],
)
def test_unknown_status_names_only_allowlisted_reasons(monkeypatch, caplog, error, expected, logged):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    backend = BACKEND(s3_client=object(), sagemaker_client=object())

    def fail(handle):
        raise error

    monkeypatch.setattr(backend, "_description", fail)
    status = backend.status(object())
    assert status.state is JobState.UNKNOWN and status.raw == expected
    assert messages_of(caplog) == [logged]
