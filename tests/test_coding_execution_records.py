"""Ordinary file durability checks; no native SDK or candidate execution."""

from __future__ import annotations

import copy
import json
import os
import stat

import pytest

from amlas.evaluation import coding_executor as executor

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


def record(kind="attempt"):
    value = {
        **BINDING,
        "binding": copy.deepcopy(BINDING),
        "session_id": REQUEST["sessionId"],
        "interpreter_id": REQUEST["codeInterpreterIdentifier"],
        "kind": kind,
        "request_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "time": 1.0,
        "deadline": 10.0,
    }
    return value


def sink(path):
    return executor.FileInvocationRecorder(path)


def records(path):
    return [json.loads(item.read_bytes()) for item in sorted(path.iterdir())]


class Fake:
    def __init__(self, path, timeline, failure=None):
        self.path, self.timeline, self.failure = path, timeline, failure
        self.calls = self.iterations = 0

    def invoke_code_interpreter(self, **request):
        self.calls += 1
        assert request == REQUEST
        assert records(self.path)[0]["kind"] == "attempt"
        assert self.timeline[-2:] == ["file", "directory"]
        if self.failure is not None:
            raise self.failure
        return {
            "sessionId": REQUEST["sessionId"],
            "ResponseMetadata": {"RequestId": "fixture-request"},
            "stream": self.stream(),
        }

    def stream(self):
        self.iterations += 1
        assert records(self.path)[-1]["kind"] == "response"
        assert self.timeline[-2:] == ["file", "directory"]
        yield copy.deepcopy(RESULT)


def run(fake, persist):
    return executor.invoke_recorded(
        fake,
        copy.deepcopy(REQUEST),
        binding=copy.deepcopy(BINDING),
        persist=persist,
        deadline=10.0,
        clock=lambda: 1.0,
    )


def observe_fsync(monkeypatch):
    real = os.fsync
    timeline = []

    def sync(fd):
        timeline.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    return timeline


def test_real_files_are_synced_before_native_call_and_stream_advance(tmp_path, monkeypatch):
    timeline = observe_fsync(monkeypatch)
    path = tmp_path / "run"
    path.mkdir()
    fake = Fake(path, timeline)
    with sink(path) as persist:
        assert run(fake, persist) == (RESULT,)
    saved = records(path)
    assert [value["kind"] for value in saved] == ["attempt", "response", "event", "completed"]
    assert timeline == ["file", "directory"] * 4
    assert fake.calls == fake.iterations == 1
    assert all(stat.S_IMODE(item.stat().st_mode) == 0o600 for item in path.iterdir())
    assert all(item.stat().st_size <= 16384 for item in path.iterdir())
    assert "print(1)" not in json.dumps(saved)
    assert saved[-1]["oracle_completed"] is False and saved[-1]["quality_success"] is False


def test_record_names_are_unique_and_run_directory_cannot_be_reused(tmp_path):
    with sink(tmp_path) as persist:
        persist(record())
        first = next(tmp_path.iterdir()).read_bytes()
        persist(record())
    assert len(list(tmp_path.iterdir())) == 2
    assert sorted(tmp_path.iterdir())[0].read_bytes() == first
    with pytest.raises(ValueError, match="empty"):
        sink(tmp_path)


def test_held_directory_receives_records_after_ordinary_rename(tmp_path):
    original = tmp_path / "run"
    original.mkdir()
    moved = tmp_path / "retained"
    with sink(original) as persist:
        original.rename(moved)
        original.mkdir()
        persist(record())
    assert list(original.iterdir()) == []
    assert records(moved) == [record()]


@pytest.mark.parametrize("failure", [ConnectionError("fixture interruption"), KeyboardInterrupt()])
def test_interruption_keeps_original_unknown_without_retry(tmp_path, monkeypatch, failure):
    timeline = observe_fsync(monkeypatch)
    fake = Fake(tmp_path, timeline, failure)
    with sink(tmp_path) as persist:
        with pytest.raises(type(failure)):
            run(fake, persist)
    saved = records(tmp_path)
    assert [value["kind"] for value in saved] == ["attempt", "unknown"]
    assert saved[-1]["session_id"] == REQUEST["sessionId"]
    assert saved[-1]["binding"] == BINDING
    assert fake.calls == 1 and fake.iterations == 0


@pytest.mark.parametrize("stage", ["file", "directory"])
def test_attempt_sync_failure_prevents_native_call_and_preserves_file(tmp_path, monkeypatch, stage):
    real = os.fsync
    timeline = []

    def sync(fd):
        kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        timeline.append(kind)
        if kind == stage:
            raise OSError("fixture sync interruption")
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    fake = Fake(tmp_path, timeline)
    with sink(tmp_path) as persist:
        with pytest.raises(OSError, match="fixture sync"):
            run(fake, persist)
    assert fake.calls == fake.iterations == 0
    assert len(list(tmp_path.iterdir())) == 1


def test_response_sync_failure_records_unknown_before_any_iteration(tmp_path, monkeypatch):
    real = os.fsync
    timeline = []
    files = 0

    def sync(fd):
        nonlocal files
        kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        timeline.append(kind)
        if kind == "file":
            files += 1
            if files == 2:
                raise OSError("fixture response sync interruption")
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    fake = Fake(tmp_path, timeline)
    with sink(tmp_path) as persist:
        with pytest.raises(OSError, match="fixture response"):
            run(fake, persist)
    saved = records(tmp_path)
    assert [value["kind"] for value in saved] == ["attempt", "response", "unknown"]
    assert saved[-1]["session_id"] == REQUEST["sessionId"]
    assert fake.calls == 1 and fake.iterations == 0


@pytest.mark.parametrize("change", ["body", "authority", "hash", "identity"])
def test_closed_hash_only_record_schema_refuses_before_file_creation(tmp_path, change):
    value = record()
    if change == "body":
        value["code"] = "print(1)"
    elif change == "authority":
        value["quality_success"] = True
    elif change == "hash":
        value["code_sha256"] = "missing"
    else:
        value["session_id"] = "x" * 129
    with sink(tmp_path) as persist:
        with pytest.raises(ValueError):
            persist(value)
    assert list(tmp_path.iterdir()) == []


def test_short_writes_finish_and_closed_sink_refuses_new_records(tmp_path, monkeypatch):
    real = os.write
    calls = []

    def short(fd, data):
        calls.append(len(data))
        return real(fd, data[:17])

    monkeypatch.setattr(os, "write", short)
    persist = sink(tmp_path)
    persist(record())
    persist.close()
    persist.close()
    assert len(calls) > 1 and records(tmp_path) == [record()]
    with pytest.raises(ValueError, match="closed"):
        persist(record())


def test_file_name_and_body_use_same_record_snapshot(tmp_path, monkeypatch):
    real = executor._record_bytes

    def encoded(value):
        result = real(value)
        value["kind"] = "completed"
        return result

    monkeypatch.setattr(executor, "_record_bytes", encoded)
    with sink(tmp_path) as persist:
        persist(record())
    item = next(tmp_path.iterdir())
    assert item.name.endswith("-attempt.json")
    assert json.loads(item.read_bytes())["kind"] == "attempt"


def test_nested_binding_keeps_exact_plain_types(tmp_path):
    value = record()
    value["binding"]["ordinal"] = False
    with sink(tmp_path) as persist:
        with pytest.raises(ValueError, match="binding"):
            persist(value)
    assert list(tmp_path.iterdir()) == []


def test_real_ten_second_case_alarm_does_not_start_unknown_fsync_that_delays_cleanup(tmp_path, monkeypatch):
    import signal
    import time

    from test_coding_arbiter import binding
    from test_paired_coding_runner import Native, Oracle, population

    from amlas.evaluation.coding_pair_runner import run_paired

    root = tmp_path / "run"
    root.mkdir()
    cases, candidates = population()
    candidates["tuned"] = tuple({**row, "outcome": "generation_error"} for row in candidates["tuned"])
    observed = {"unknown_fsyncs": 0, "case_start": None, "cleanup_start": None, "stream_closes": 0}
    active_kind = [None]
    original_record, original_fsync = executor.FileInvocationRecorder.__call__, os.fsync

    def write(self, value):
        active_kind[0] = value["kind"]
        if value["kind"] == "attempt":
            observed["case_start"] = value["deadline"] - 10.0
        try:
            return original_record(self, value)
        finally:
            active_kind[0] = None

    def sync(fd):
        if active_kind[0] == "unknown" and not observed["unknown_fsyncs"]:
            observed["unknown_fsyncs"] += 1
            observed["owned_guard_active"] = executor._ACTIVE_DEADLINE is not None
            observed["alarm_consumed"] = signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
            observed["unknown_fsync_start"] = time.monotonic()
            time.sleep(1.0)
            observed["unknown_fsync_end"] = time.monotonic()
        original_fsync(fd)

    monkeypatch.setattr(executor.FileInvocationRecorder, "__call__", write)
    monkeypatch.setattr(os, "fsync", sync)

    class OriginalStream:
        def __init__(self):
            self.reads = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.reads += 1
            if self.reads == 1:
                return {"progress": {"text": "neutral original observation"}}
            time.sleep(11.0)
            raise AssertionError("original case alarm must interrupt this delayed read")

        def close(self):
            observed["stream_closes"] += 1

    class DelayedNative(Native):
        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            response["stream"] = OriginalStream()
            return response

        def stop_code_interpreter_session(self, **request):
            observed["cleanup_start"] = time.monotonic()
            return super().stop_code_interpreter_session(**request)

    native = DelayedNative(root)
    started = time.monotonic()
    result = run_paired(
        binding=binding(),
        cases=cases,
        candidates=candidates,
        client=native,
        cleanup_client=native,
        evidence_root=root,
        oracle_runner=Oracle(),
    )
    observed["wall_seconds"] = time.monotonic() - started
    observed["cleanup_case_seconds"] = observed["cleanup_start"] - observed["case_start"]
    observed["fixed_case_seconds"] = result["case_wall_seconds"]
    print("CASE-PERSISTENCE-PROBE " + json.dumps(observed, sort_keys=True))
    assert result["case_wall_seconds"] == 10 and result["allocation_excluded"] is True
    assert len(result["receipts"]) == 328 and result["receipts"][0].outcome == "timeout"
    assert result["native_admission"] is result["quality_success"] is False
    assert len(native.starts) == len(native.invokes) == len(native.stops) == len(native.gets) == 1
    assert observed["stream_closes"] == 1 and result["cleanup_complete"] is True
    # 10.7 detects the injected one-second delay; it does not change the fixed10s budget.
    assert observed["cleanup_case_seconds"] < 10.7, "post-alarm unknown fsync postponed original cleanup"
    assert observed["unknown_fsyncs"] == 0
    saved = records(root / "base" / "000" / "calls")
    assert [row["kind"] for row in saved] == ["attempt", "response", "event"]
    assert all(row["session_id"] == "session-1" for row in saved)
    assert not any(row["kind"] == "completed" for row in saved)
