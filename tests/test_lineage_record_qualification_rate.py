"""A callback may offer a cheaper per-record check; the full check still brackets every read.

One lineage pass yields tens of thousands of records. The SFT parent-capture callback makes several native round
trips per call, so running it in full for every record turns a pass into hours. A `RecordRecheck` callback runs its
local checks for every record and its full check once per interval counted from the start of the last full one that
returned normally. Plain callbacks, scope checks and every read bracket are unchanged. Neutral stand-ins only.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from types import SimpleNamespace

import pytest

from amlas.backends.adapters import coding_preparation as prep

BACKEND = prep.ProcessingCodingPreparationBackend
BUCKET, ACCOUNT, VERSION = "ordinary-bucket", "ordinary-account", "ordinary-version"
LINEAGE_KEY = "ordinary/prefix/LINEAGE.jsonl"
INTERVAL = 1.0
SCOPE_REFUSAL = "expired or foreign-thread processing observation refused"


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
    # The module reads only time.monotonic(); the virtual clock drives every deadline and interval check.
    monkeypatch.setattr(prep, "time", SimpleNamespace(monotonic=value.monotonic))
    return value


class Body(io.BytesIO):
    def __init__(self, raw):
        super().__init__(raw)
        self.reads, self.sizes = 0, []

    def read(self, size=-1):
        self.reads += 1
        self.sizes.append(size)
        return super().read(size)


class Check:
    """A check stand-in: it costs virtual seconds, remembers when its last success began, and can refuse."""

    def __init__(self, clock, cost=0.0):
        self.clock, self.cost = clock, cost
        self.calls, self.last_start, self.refuse_from = 0, None, None

    def __call__(self):
        self.calls += 1
        if self.refuse_from is not None and self.clock.now >= self.refuse_from:
            raise PermissionError("ordinary synthetic refusal")
        started = self.clock.now
        self.clock.advance(self.cost)
        self.last_start = started


def recheck(clock, full_cost=0.0):
    full, local = Check(clock, full_cost), Check(clock)
    return prep.RecordRecheck(full, local, INTERVAL), full, local


def lineage(count, pad=180):
    raw = b"".join(
        json.dumps({"ordinary": index, "pad": "x" * pad}).encode() + b"\n" for index in range(count)
    )
    body = Body(raw)

    def get_object(**kwargs):
        assert kwargs["Bucket"] == BUCKET and kwargs["Key"] == LINEAGE_KEY and kwargs["VersionId"] == VERSION
        return {"VersionId": VERSION, "ContentLength": len(raw), "Body": body}

    ref = {
        "uri": "s3://" + BUCKET + "/" + LINEAGE_KEY,
        "version_id": VERSION,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "format": "jsonl",
    }
    catalog = SimpleNamespace(bucket=BUCKET, account_id=ACCOUNT)
    s3 = SimpleNamespace(get_object=get_object)
    return BACKEND(s3_client=s3, sagemaker_client=object()), s3, catalog, ref, body, raw


# -- the callback object --


def test_calling_the_callback_runs_the_full_check_and_restarts_the_interval(clock):
    callback, full, local = recheck(clock)
    callback()
    assert (full.calls, local.calls) == (1, 0)
    clock.advance(0.5)
    callback.record_check()
    assert (full.calls, local.calls) == (1, 1)


def test_first_record_check_runs_the_full_check(clock):
    callback, full, local = recheck(clock)
    callback.record_check()
    assert (full.calls, local.calls) == (1, 0)


def test_interval_counts_from_the_start_of_the_last_full_check(clock):
    callback, full, local = recheck(clock, full_cost=0.6)
    callback.record_check()  # full at t=0, ends at t=0.6
    assert full.calls == 1
    callback.record_check()  # 0.6 s after the start: still inside the interval
    assert (full.calls, local.calls) == (1, 1)
    clock.advance(0.5)  # 1.1 s after the start
    callback.record_check()
    assert full.calls == 2


def test_a_failed_full_check_does_not_restart_the_interval(clock):
    callback, full, local = recheck(clock)
    full.refuse_from = clock.now
    with pytest.raises(PermissionError):
        callback.record_check()
    full.refuse_from = None
    callback.record_check()
    assert full.calls == 2 and local.calls == 0


def test_a_local_refusal_is_raised_for_the_very_next_record(clock):
    callback, _full, local = recheck(clock)
    callback.record_check()
    local.refuse_from = clock.now
    with pytest.raises(PermissionError):
        callback.record_check()


def test_callback_needs_both_checks_and_a_non_negative_interval():
    def ok():
        return None

    for args in ((None, ok, 1.0), (ok, None, 1.0), (ok, ok, -1.0)):
        with pytest.raises(ValueError):
            prep.RecordRecheck(*args)


def test_interval_zero_runs_the_full_check_for_every_record(clock):
    full, local = Check(clock), Check(clock)
    callback = prep.RecordRecheck(full, local, 0.0)
    for _ in range(5):
        callback.record_check()
    assert (full.calls, local.calls) == (5, 0)


def test_restore_helper_builds_a_callback_that_skips_only_the_remote_steps(clock):
    from amlas.control_plane import normal_run_checkpoint as checkpoint

    seen = []
    callback = checkpoint._record_recheck(lambda remote: seen.append(remote))
    assert type(callback) is prep.RecordRecheck
    callback()
    callback.record_check()
    clock.advance(checkpoint._PARENT_RECORD_RECHECK_SECONDS + 0.1)
    callback.record_check()
    assert seen == [True, False, True]
    assert checkpoint._PARENT_RECORD_RECHECK_SECONDS == 1.0


# -- the lineage stream --


def test_plain_callbacks_still_run_in_full_for_every_record(clock):
    backend, s3, catalog, ref, _body, _raw = lineage(2000)
    plain = Check(clock)
    with backend.qualified_io(plain):
        records = list(backend._lineage_objects(s3, catalog, ref))
    assert len(records) == 2000
    assert plain.calls >= 2000


def test_records_between_reads_cost_local_checks_only(clock):
    backend, s3, catalog, ref, body, raw = lineage(20000)
    assert len(raw) > 3 * 1024 * 1024
    callback, full, local = recheck(clock)
    with backend.qualified_io(callback):
        records = list(backend._lineage_objects(s3, catalog, ref))
    assert [record["ordinary"] for record in records] == list(range(20000))
    assert body.reads >= 4
    assert local.calls == 20000
    assert full.calls <= 4 * body.reads + 10


def test_full_checks_do_not_grow_with_the_record_count(clock):
    counts = []
    for count in (4000, 40000):
        backend, s3, catalog, ref, body, _raw = lineage(count)
        callback, full, local = recheck(clock)
        with backend.qualified_io(callback):
            list(backend._lineage_objects(s3, catalog, ref))
        counts.append((full.calls, body.reads, local.calls))
    for (full_calls, reads, local_calls), count in zip(counts, (4000, 40000), strict=True):
        assert full_calls <= 4 * reads + 10
        assert local_calls == count


def test_slow_consumer_gets_a_full_check_before_the_interval_passes(clock):
    backend, s3, catalog, ref, body, _raw = lineage(200)
    callback, full, _local = recheck(clock, full_cost=0.2)
    ages = []
    with backend.qualified_io(callback):
        for _record in backend._lineage_objects(s3, catalog, ref):
            ages.append(clock.now - full.last_start)
            clock.advance(0.6)
    assert len(ages) == 200
    assert max(ages) < INTERVAL
    assert full.calls > 4 * body.reads + 5


def test_consumer_suspended_beyond_the_interval_gets_a_full_check_first(clock):
    backend, s3, catalog, ref, _body, _raw = lineage(50)
    callback, full, _local = recheck(clock)
    with backend.qualified_io(callback):
        stream = backend._lineage_objects(s3, catalog, ref)
        next(stream)
        before = full.calls
        clock.advance(5.0)
        next(stream)
        assert full.calls == before + 1
        stream.close()


def test_local_refusal_stops_the_stream_before_the_next_buffered_record(clock):
    backend, s3, catalog, ref, body, _raw = lineage(20000)
    callback, _full, local = recheck(clock)
    with backend.qualified_io(callback):
        stream = backend._lineage_objects(s3, catalog, ref)
        assert next(stream)["ordinary"] == 0
        local.refuse_from = clock.now
        with pytest.raises(PermissionError):
            next(stream)
    assert body.closed


def test_full_refusal_at_the_interval_stops_the_stream(clock):
    backend, s3, catalog, ref, body, _raw = lineage(200)
    callback, full, _local = recheck(clock)
    with backend.qualified_io(callback):
        stream = backend._lineage_objects(s3, catalog, ref)
        next(stream)
        full.refuse_from = clock.now + INTERVAL
        clock.advance(INTERVAL + 0.1)
        with pytest.raises(PermissionError):
            next(stream)
    assert body.closed


def test_scope_closed_between_records_is_refused_for_the_very_next_record(clock):
    backend, s3, catalog, ref, body, _raw = lineage(20000)
    callback, _full, _local = recheck(clock)
    with backend.qualified_io(callback):
        stream = backend._lineage_objects(s3, catalog, ref)
        next(stream)
        next(stream)
    with pytest.raises(ValueError, match=SCOPE_REFUSAL):
        next(stream)
    assert body.closed


def test_stream_handed_to_another_thread_is_refused_for_its_next_record(clock):
    backend, s3, catalog, ref, body, _raw = lineage(20000)
    callback, _full, _local = recheck(clock)
    refusals = []
    with backend.qualified_io(callback):
        stream = backend._lineage_objects(s3, catalog, ref)
        next(stream)

        def other_thread():
            try:
                next(stream)
            except ValueError as error:
                refusals.append(str(error))

        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join()
    assert refusals == [SCOPE_REFUSAL]
    assert body.closed


def test_real_clock_pass_costs_full_checks_per_read_not_per_record():
    backend, s3, catalog, ref, body, _raw = lineage(3000)
    full, local = [], []
    callback = prep.RecordRecheck(lambda: full.append(1), lambda: local.append(1), INTERVAL)
    with backend.qualified_io(callback):
        records = list(backend._lineage_objects(s3, catalog, ref))
    assert len(records) == 3000
    assert len(full) <= 4 * body.reads + 12
    assert len(local) >= 2900


def test_lineage_is_read_in_mebibyte_requests(clock):
    backend, s3, catalog, ref, body, raw = lineage(20000)
    callback, _full, _local = recheck(clock)
    with backend.qualified_io(callback):
        list(backend._lineage_objects(s3, catalog, ref))
    assert prep.LINEAGE_READ_BYTES == 1024 * 1024
    assert max(body.sizes) == prep.LINEAGE_READ_BYTES
    assert body.reads <= len(raw) // prep.LINEAGE_READ_BYTES + 2
