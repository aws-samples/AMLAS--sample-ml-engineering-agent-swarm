"""Neutral transport/oracle fakes; no candidate, test-code or subprocess execution."""

from __future__ import annotations

import hashlib
import json

import pytest
from test_coding_arbiter import binding

from amlas.evaluation.coding_oracle import encode
from amlas.evaluation.coding_pair import run_paired


def population():
    cases = tuple(
        {
            "prompt": {
                "case_id": "HumanEval/" + str(i),
                "prompt": "neutral prompt",
                "entry_point": "fixture_" + str(i),
            },
            "oracle": {
                "case_id": "HumanEval/" + str(i),
                "test": "FIXED_PARENT_ONLY",
                "canonical_solution": "reference metadata",
            },
            "oracle_sha256": f"{i + 1:064x}",
        }
        for i in range(164)
    )
    candidates = {}
    for arm in ("base", "tuned"):
        rows = []
        for i in range(164):
            source = "def fixture_0():\n    return None\n" if i == 0 else None
            rows.append(
                {
                    "case_id": "HumanEval/" + str(i),
                    "outcome": "generated" if i == 0 else "generation_error",
                    "source": source,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
                }
            )
        candidates[arm] = tuple(rows)
    return cases, candidates


class Native:
    def __init__(self, root, *, start_error=False, cleanup_error=False):
        self.root = root
        self.starts, self.invokes, self.stops, self.gets = [], [], [], []
        self.start_error, self.cleanup_error = start_error, cleanup_error

    def start_code_interpreter_session(self, **request):
        assert list(self.root.rglob("*start-attempt.json")), "allocation attempt must be durable first"
        self.starts.append(request)
        if self.start_error:
            raise TimeoutError("neutral allocation ambiguity")
        return {"sessionId": "session-" + str(len(self.starts))}

    def invoke_code_interpreter(self, **request):
        assert list(self.root.rglob("*attempt.json")), "call attempt must be durable first"
        self.invokes.append(request)
        assert request["codeInterpreterIdentifier"] == "aws.codeinterpreter.v1"
        assert "FIXED_PARENT_ONLY" not in request["arguments"]["code"]
        ordinal = sum(item["sessionId"] == request["sessionId"] for item in self.invokes) - 1
        stdout = (
            json.dumps({"ready": True})
            if ordinal == 0
            else json.dumps({"ordinal": ordinal, "value": encode(("value", {(1, 2): 3}))})
        )
        return {
            "sessionId": request["sessionId"],
            "ResponseMetadata": {"RequestId": "request-" + str(len(self.invokes))},
            "stream": iter(
                [
                    {
                        "result": {
                            "isError": False,
                            "content": [{"type": "text", "text": stdout}],
                            "structuredContent": {"exitCode": 0, "stdout": stdout, "stderr": ""},
                        }
                    }
                ]
            ),
        }

    def stop_code_interpreter_session(self, **request):
        assert list(self.root.rglob("*stop-attempt.json"))
        self.stops.append(request)
        if self.cleanup_error:
            raise TimeoutError("neutral original cleanup ambiguity")
        return {"sessionId": request["sessionId"]}

    def get_code_interpreter_session(self, **request):
        assert list(self.root.rglob("*get-attempt.json"))
        self.gets.append(request)
        return {"sessionId": request["sessionId"], "status": "TERMINATED"}


class Oracle:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def run(self, case, proxy, *, deadline, clock):
        self.calls.append(case)
        assert case["oracle"]["test"] == "FIXED_PARENT_ONLY"
        value = proxy((1, 2), {3: "v"}, flag=True)
        assert value == ("value", {(1, 2): 3})
        if self.failure:
            raise self.failure
        return {"outcome": "passed", "oracle_completed": True, "exit_code": 0}


def run(root, *, oracle=None, native=None):
    root.mkdir()
    native = native or Native(root)
    cases, candidates = population()
    result = run_paired(
        binding=binding(),
        cases=cases,
        candidates=candidates,
        client=native,
        cleanup_client=native,
        evidence_root=root,
        oracle_runner=oracle or Oracle(),
        clock=lambda: 1.0,
        sleep=lambda _seconds: None,
    )
    return result, native


def test_actual_parent_flow_retains328_and_uses_recorded_native_rpc_with_original_cleanup(tmp_path):
    result, native = run(tmp_path / "run")
    receipts = result["receipts"]
    assert len(receipts) == 328 and sum(row.outcome == "passed" for row in receipts) == 2
    assert len(native.starts) == len(native.stops) == len(native.gets) == 2
    assert all(request["sessionTimeoutSeconds"] == 60 for request in native.starts)
    assert result["case_wall_seconds"] == 10 and result["allocation_excluded"] is True
    assert result["cleanup_complete"] is True
    assert result["native_admission"] is result["quality_success"] is False
    assert len({row.execution_id for row in receipts}) == 328
    assert all(row.oracle_completed is False for row in receipts if row.outcome != "passed")
    call_records = [
        json.loads(path.read_bytes())
        for path in (tmp_path / "run").rglob("*-attempt.json")
        if "calls" in path.parts
    ]
    assert len(call_records) == 4
    assert all(row["interpreter_id"] == "aws.codeinterpreter.v1" for row in call_records)


@pytest.mark.parametrize(
    "failure,outcome",
    [
        (TimeoutError("neutral original deadline"), "timeout"),
        (ValueError("neutral protocol error"), "infrastructure_error"),
    ],
)
def test_parent_unknown_never_reinvokes_candidate_and_preserves_every_denominator_case(
    tmp_path, failure, outcome
):
    result, native = run(tmp_path / "run", oracle=Oracle(failure))
    assert len(result["receipts"]) == 328
    assert sum(row.outcome == outcome for row in result["receipts"]) == 2
    assert len(native.starts) == len(native.stops) == 2 and len(native.invokes) == 4
    assert result["cleanup_complete"] is True


def test_unknown_allocation_and_cleanup_remain_unqualified_without_resubmit(tmp_path):
    root = tmp_path / "run"
    native = Native(root, start_error=True)
    result, native = run(root, native=native)
    assert len(result["receipts"]) == 328 and len(native.starts) == 2
    assert not native.invokes and not native.stops
    assert result["cleanup_complete"] is False
    root2 = tmp_path / "run2"
    native2 = Native(root2, cleanup_error=True)
    result2, native2 = run(root2, native=native2)
    assert len(native2.stops) == 2 and not native2.gets
    assert len(result2["receipts"]) == 328 and result2["cleanup_complete"] is False


def test_trusted_oracle_child_arguments_hold_no_credentials_or_test_text(tmp_path):
    from amlas.evaluation.coding_pair_runner import oracle_child_command

    argv, environment = oracle_child_command()
    assert argv == ["/usr/local/bin/amlas-oracle-init"]
    assert not any(name.startswith(("AWS_", "AMLAS_", "TAVILY", "BEDROCK")) for name in environment)
    assert "HOME" not in environment and "PYTHONPATH" not in environment
    assert "FIXED_PARENT_ONLY" not in " ".join(argv)


def test_whole_case_watchdog_stops_original_before_bootstrap_and_never_retries(tmp_path):
    from amlas.evaluation.coding_pair_runner import run_paired

    root = tmp_path / "run"
    root.mkdir()
    native = Native(root)
    cases, candidates = population()

    class ImmediateTimer:
        def __init__(self, seconds, callback):
            assert seconds == 10
            self.callback = callback

        def start(self):
            self.callback()

        def cancel(self):
            pass

    result = run_paired(
        binding=binding(),
        cases=cases,
        candidates=candidates,
        client=native,
        cleanup_client=native,
        evidence_root=root,
        oracle_runner=Oracle(),
        clock=lambda: 1.0,
        sleep=lambda _: None,
        timer_factory=ImmediateTimer,
    )
    assert len(result["receipts"]) == 328
    assert len(native.starts) == len(native.stops) == len(native.gets) == 2
    assert not native.invokes
    assert sum(row.outcome == "timeout" for row in result["receipts"]) == 2
    assert result["cleanup_complete"] is True


def test_actual_ten_second_case_deadline_interrupts_blocking_original_invoke(tmp_path):
    import time

    from amlas.evaluation.coding_pair_runner import run_paired as fixed_run

    root = tmp_path / "run"
    root.mkdir()
    cases, candidates = population()
    candidates["tuned"] = tuple({**row, "outcome": "generation_error"} for row in candidates["tuned"])

    class DelayedNative(Native):
        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            time.sleep(11.0)
            return response

    native = DelayedNative(root)
    started = time.monotonic()
    result = fixed_run(
        binding=binding(),
        cases=cases,
        candidates=candidates,
        client=native,
        cleanup_client=native,
        evidence_root=root,
        oracle_runner=Oracle(),
    )
    elapsed = time.monotonic() - started
    assert elapsed < 10.7, f"original ten-second deadline returned after {elapsed:.3f}s"
    assert len(result["receipts"]) == 328
    assert result["receipts"][0].outcome == "timeout"
    assert len(native.starts) == len(native.invokes) == len(native.stops) == len(native.gets) == 1
    saved = [json.loads(p.read_bytes()) for p in (root / "base" / "000" / "calls").iterdir()]
    assert [row["kind"] for row in saved] == ["attempt"]
    assert saved[0]["session_id"] == "session-1"
    assert result["cleanup_complete"] is True
    assert result["native_admission"] is result["quality_success"] is False


@pytest.mark.parametrize("phase", ["stop", "get"])
def test_cleanup_remaining_absolute_budget_interrupts_actual_blocking_native_call(tmp_path, phase):
    import time

    from amlas.evaluation.coding_pair_runner import _cleanup, _SessionRecords

    root = tmp_path / "session"
    root.mkdir()
    records = _SessionRecords(root, "fixture-evaluation", "HumanEval/0", "base")
    calls = []
    started = time.monotonic()
    ticks = iter([started])

    def clock():
        return next(ticks, started + 29.8)

    class DelayedCleanup:
        def stop_code_interpreter_session(self, **request):
            calls.append(("stop", request))
            if phase == "stop":
                time.sleep(1.0)
            return {"sessionId": request["sessionId"]}

        def get_code_interpreter_session(self, **request):
            calls.append(("get", request))
            if phase == "get":
                time.sleep(1.0)
            return {"sessionId": request["sessionId"], "status": "TERMINATED"}

    assert _cleanup(DelayedCleanup(), "original-session", records, clock=clock, sleep=time.sleep) is False
    assert time.monotonic() - started < 0.7
    assert [name for name, _request in calls] == (["stop"] if phase == "stop" else ["stop", "get"])
    saved = [
        {"kind": p.name.split("-", 1)[1][:-5], **json.loads(p.read_bytes())} for p in sorted(root.iterdir())
    ]
    assert not any(row["kind"] == "cleanup-unknown" for row in saved)
    assert saved[-1]["kind"] == ("stop-attempt" if phase == "stop" else "get-attempt")
    assert saved[-1]["session_id"] == "original-session"
    assert all(request["sessionId"] == "original-session" for _name, request in calls)


def test_original_stream_close_stop_and_get_have_one_synchronous_cleanup_owner(tmp_path):
    import threading
    import time

    from amlas.evaluation.coding_pair_runner import _cleanup, _SessionRecords

    root = tmp_path / "session"
    root.mkdir()
    records = _SessionRecords(root, "fixture-evaluation", "HumanEval/0", "base")
    calls = []

    class Stream:
        def close(self):
            calls.append(("close", threading.get_ident()))

    class Cleanup:
        def stop_code_interpreter_session(self, **request):
            calls.append(("stop", threading.get_ident()))
            assert request["sessionId"] == "original-session"
            return {"sessionId": request["sessionId"]}

        def get_code_interpreter_session(self, **request):
            calls.append(("get", threading.get_ident()))
            assert request["sessionId"] == "original-session"
            return {"sessionId": request["sessionId"], "status": "TERMINATED"}

    stream = Stream()
    assert (
        _cleanup(
            Cleanup(),
            "original-session",
            records,
            clock=time.monotonic,
            sleep=time.sleep,
            streams=[stream, stream],
        )
        is True
    )
    assert calls == [(kind, threading.get_ident()) for kind in ("close", "stop", "get")]


def test_original_stream_close_shares_same_thirty_second_cleanup_cap(tmp_path):
    import time

    from amlas.evaluation.coding_pair_runner import _cleanup, _SessionRecords

    root = tmp_path / "session"
    root.mkdir()
    records = _SessionRecords(root, "fixture-evaluation", "HumanEval/0", "base")
    calls = []
    started = time.monotonic()
    ticks = iter([started])

    class Stream:
        def close(self):
            calls.append("close")
            time.sleep(1.0)

    class Cleanup:
        def stop_code_interpreter_session(self, **request):
            calls.append("stop")
            raise AssertionError("cleanup deadline exhausted before stop")

        def get_code_interpreter_session(self, **request):
            calls.append("get")
            raise AssertionError("cleanup deadline exhausted before get")

    assert (
        _cleanup(
            Cleanup(),
            "original-session",
            records,
            clock=lambda: next(ticks, started + 29.8),
            sleep=time.sleep,
            streams=[Stream()],
        )
        is False
    )
    assert time.monotonic() - started < 0.7 and calls == ["close"]
    saved = [json.loads(p.read_bytes()) for p in root.iterdir()]
    assert saved == [], "expired stream-close budget must not start a new persistence call"


def test_cleanup_closes_every_unique_stream_and_defers_one_unknown_until_after_native_cleanup(tmp_path):
    import time

    from amlas.evaluation.coding_pair_runner import _cleanup, _SessionRecords

    directory = tmp_path / "session"
    directory.mkdir()
    records = _SessionRecords(directory, "fixture-evaluation", "HumanEval/0", "base")
    events = []
    real_write = records.write

    def write(kind, value):
        events.append(kind)
        real_write(kind, value)

    records.write = write

    class Stream:
        def __init__(self, name, fails=False):
            self.name, self.fails = name, fails

        def close(self):
            events.append(self.name)
            if self.fails:
                raise OSError("neutral original stream close failure")

    class NativeCleanup:
        def stop_code_interpreter_session(self, **request):
            events.append("stop")
            assert request["sessionId"] == "original-session"
            return {"sessionId": request["sessionId"]}

        def get_code_interpreter_session(self, **request):
            events.append("get")
            assert request["sessionId"] == "original-session"
            return {"sessionId": request["sessionId"], "status": "TERMINATED"}

    first, second, third = Stream("first", True), Stream("second"), Stream("third", True)
    assert (
        _cleanup(
            NativeCleanup(),
            "original-session",
            records,
            clock=time.monotonic,
            sleep=time.sleep,
            streams=[first, second, first, third],
        )
        is False
    )
    assert events == [
        "first",
        "second",
        "third",
        "stop-attempt",
        "stop",
        "stop-accepted",
        "get-attempt",
        "get",
        "get-observed",
        "cleanup-unknown",
    ]
    saved = [
        {"kind": p.name.split("-", 1)[1][:-5], **json.loads(p.read_bytes())}
        for p in sorted(directory.iterdir())
    ]
    assert [row["kind"] for row in saved].count("cleanup-unknown") == 1
    assert saved[-1]["kind"] == "cleanup-unknown" and saved[-1]["error_type"] == "OSError"


@pytest.mark.parametrize("phase", ["expired", "remaining_error"])
def test_cleanup_never_starts_unbounded_unknown_persistence_outside_original_cap(tmp_path, phase):
    import time

    from amlas.evaluation.coding_pair_runner import _cleanup, _SessionRecords

    directory = tmp_path / "session"
    directory.mkdir()
    records = _SessionRecords(directory, "fixture-evaluation", "HumanEval/0", "base")
    writes, calls = [], []
    real_write = records.write

    def write(kind, value):
        writes.append(kind)
        if kind == "cleanup-unknown":
            time.sleep(1.0)
        real_write(kind, value)

    records.write = write

    class NativeCleanup:
        def stop_code_interpreter_session(self, **request):
            calls.append("stop")
            if phase == "expired":
                time.sleep(1.0)
            raise ConnectionError("neutral cleanup observation interruption")

        def get_code_interpreter_session(self, **request):
            calls.append("get")
            raise AssertionError("no cleanup retry after original interruption")

    started = time.monotonic()
    ticks = iter([started])
    assert (
        _cleanup(
            NativeCleanup(),
            "original-session",
            records,
            clock=lambda: next(ticks, started + 29.8),
            sleep=time.sleep,
        )
        is False
    )
    assert time.monotonic() - started < 0.7
    assert calls == ["stop"]
    assert writes == (["stop-attempt"] if phase == "expired" else ["stop-attempt", "cleanup-unknown"])
    saved = [
        {"kind": p.name.split("-", 1)[1][:-5], **json.loads(p.read_bytes())}
        for p in sorted(directory.iterdir())
    ]
    assert [row["kind"] for row in saved] == ["stop-attempt"]


class NeutralOracleProcess:
    def __init__(self, verdict, *, exit_code=0, trailing=b"", wait_error=None, blocking=False):
        import io

        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(json.dumps({"kind": "verdict", **verdict}).encode() + b"\n" + trailing)
        self.returncode = None
        self.exit_code = exit_code
        self.wait_error = wait_error
        self.blocking = blocking
        self.kills = 0
        self.waits = []

    def poll(self):
        return self.returncode

    def kill(self):
        self.kills += 1
        self.returncode = -9

    def wait(self, timeout):
        import time

        self.waits.append(timeout)
        if self.returncode is None:
            if self.wait_error is not None:
                raise self.wait_error
            if self.blocking:
                time.sleep(12)
            self.returncode = self.exit_code
        return self.returncode


def neutral_oracle_process(monkeypatch, **kwargs):
    from amlas.evaluation import coding_pair_runner as runner

    verdict = kwargs.pop("verdict", {"outcome": "passed", "oracle_completed": True, "exit_code": 0})
    process = NeutralOracleProcess(verdict, **kwargs)
    timers = []
    spawns = []

    class Timer:
        def __init__(self, seconds, callback):
            self.seconds, self.callback, self.cancelled = seconds, callback, False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    class Selector:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def register(self, stream, _events):
            assert stream is process.stdout

        def select(self, remaining):
            assert remaining > 0
            return [True]

    def popen(argv, **options):
        spawns.append((argv, options))
        assert argv == ["/usr/local/bin/amlas-oracle-init"] and options["close_fds"] is True
        return process

    monkeypatch.setattr(runner, "verify_pinned_case", lambda *_args: None)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.threading, "Timer", Timer)
    monkeypatch.setattr(runner.selectors, "DefaultSelector", Selector)
    case = population()[0][0]
    return runner, case, process, timers, spawns


@pytest.mark.parametrize("outcome,exit_code", [("passed", 0), ("failed", 1)])
def test_oracle_verdict_before_normal_exit_is_not_killed_or_accepted_early(monkeypatch, outcome, exit_code):
    verdict = {"outcome": outcome, "oracle_completed": True, "exit_code": exit_code}
    runner, case, process, timers, spawns = neutral_oracle_process(monkeypatch, verdict=verdict)
    result = runner.OracleProcessRunner().run(case, lambda *_a, **_k: None, deadline=11.0, clock=lambda: 1.0)
    assert result == verdict and process.returncode == 0 and process.kills == 0
    assert process.waits[0] == 10.0 and len(spawns) == 1
    assert timers[0].cancelled and process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize("exit_code", [7, -9])
def test_oracle_verdict_cannot_override_nonzero_or_signal_exit(monkeypatch, exit_code):
    runner, case, process, timers, spawns = neutral_oracle_process(monkeypatch, exit_code=exit_code)
    with pytest.raises(ValueError, match="normal exit"):
        runner.OracleProcessRunner().run(case, lambda *_a, **_k: None, deadline=11.0, clock=lambda: 1.0)
    assert process.returncode == exit_code and process.kills == 0 and len(spawns) == 1
    assert timers[0].cancelled and process.stdin.closed and process.stdout.closed


def test_oracle_postverdict_hang_uses_remaining_deadline_and_original_cleanup(monkeypatch):
    import subprocess

    error = subprocess.TimeoutExpired("neutral original oracle", 10)
    runner, case, process, timers, spawns = neutral_oracle_process(monkeypatch, wait_error=error)
    with pytest.raises(TimeoutError):
        runner.OracleProcessRunner().run(case, lambda *_a, **_k: None, deadline=11.0, clock=lambda: 1.0)
    assert process.waits[0] == 10.0 and process.kills == 1 and len(spawns) == 1
    assert process.waits[-1] == 2 and timers[0].cancelled
    assert process.stdin.closed and process.stdout.closed


def test_oracle_normal_exit_with_trailing_protocol_bytes_refuses(monkeypatch):
    runner, case, process, timers, spawns = neutral_oracle_process(
        monkeypatch, trailing=b"ordinary trailing bytes\n"
    )
    with pytest.raises(ValueError, match="trailing"):
        runner.OracleProcessRunner().run(case, lambda *_a, **_k: None, deadline=11.0, clock=lambda: 1.0)
    assert process.returncode == 0 and process.kills == 0 and len(spawns) == 1
    assert timers[0].cancelled and process.stdin.closed and process.stdout.closed


def test_oracle_postverdict_unrelated_cancellation_keeps_original_identity(monkeypatch):
    error = KeyboardInterrupt("neutral terminal wait cancellation")
    runner, case, process, timers, spawns = neutral_oracle_process(monkeypatch, wait_error=error)
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.OracleProcessRunner().run(case, lambda *_a, **_k: None, deadline=11.0, clock=lambda: 1.0)
    assert caught.value is error and process.kills == 1 and len(spawns) == 1
    assert timers[0].cancelled and process.stdin.closed and process.stdout.closed


def test_actual_ten_second_postverdict_wait_interruption_reaps_original(monkeypatch):
    import time

    from amlas.evaluation.coding_executor import absolute_deadline

    runner, case, process, timers, spawns = neutral_oracle_process(monkeypatch, blocking=True)
    start = time.monotonic()
    deadline = start + 10.0
    with pytest.raises(TimeoutError):
        with absolute_deadline(deadline, clock=time.monotonic):
            runner.OracleProcessRunner().run(
                case, lambda *_a, **_k: None, deadline=deadline, clock=time.monotonic
            )
    elapsed = time.monotonic() - start
    assert 0 < timers[0].seconds <= 10.0 and 0 < process.waits[0] <= 10.0
    assert process.kills == 1 and len(spawns) == 1 and process.waits[-1] == 2
    assert process.stdin.closed and process.stdout.closed and timers[0].cancelled
    assert elapsed >= 10.0
    print("neutral postverdict interruption seconds", elapsed)


def test_terminal_timeout_keeps_existing_control_process_facade(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    error = subprocess.TimeoutExpired("neutral original oracle", 10)
    runner, case, process, timers, spawns = neutral_oracle_process(monkeypatch, wait_error=error)
    module = runner.subprocess
    monkeypatch.setattr(
        runner, "subprocess", SimpleNamespace(Popen=module.Popen, PIPE=module.PIPE, DEVNULL=module.DEVNULL)
    )
    with pytest.raises(TimeoutError):
        runner.OracleProcessRunner().run(case, lambda *_a, **_k: None, deadline=11.0, clock=lambda: 1.0)
    assert process.kills == 1 and len(spawns) == 1 and timers[0].cancelled


@pytest.mark.parametrize("is_error", [False, True])
def test_source_bound_completed_candidate_error_is_runtime_error_not_infrastructure(tmp_path, is_error):
    class CandidateErrorNative(Native):
        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            ordinal = sum(x["sessionId"] == request["sessionId"] for x in self.invokes) - 1
            if ordinal == 1:
                response["stream"] = iter(
                    [
                        {
                            "result": {
                                "isError": is_error,
                                "content": [],
                                "structuredContent": {
                                    "exitCode": 1,
                                    "stdout": "",
                                    "stderr": "neutral candidate failure",
                                },
                            }
                        }
                    ]
                )
            return response

    root = tmp_path / "run"
    native = CandidateErrorNative(root)
    result, native = run(root, native=native)
    assert len(result["receipts"]) == 328
    assert sum(row.outcome == "runtime_error" for row in result["receipts"]) == 2
    assert all(
        row.oracle_completed is False and row.exit_code == 1
        for row in result["receipts"]
        if row.outcome == "runtime_error"
    )
    assert len(native.starts) == len(native.stops) == 2 and len(native.invokes) == 4
    assert result["cleanup_complete"] is True


def test_cleanup_reconciles_one_transient_get_error_on_same_original_under_existing_budget(tmp_path):
    from botocore.exceptions import ClientError

    from amlas.evaluation.coding_pair_runner import _cleanup, _SessionRecords

    directory = tmp_path / "session"
    directory.mkdir()
    records = _SessionRecords(directory, "fixture-evaluation", "HumanEval/0", "base")
    calls = []

    class NativeCleanup:
        def stop_code_interpreter_session(self, **request):
            calls.append(("stop", request.copy()))
            return {"sessionId": request["sessionId"]}

        def get_code_interpreter_session(self, **request):
            calls.append(("get", request.copy()))
            if len(calls) == 2:
                raise ClientError(
                    {
                        "Error": {"Code": "ThrottlingException", "Message": "neutral withheld"},
                        "ResponseMetadata": {"HTTPStatusCode": 429, "RequestId": "fixture-request"},
                    },
                    "GetCodeInterpreterSession",
                )
            return {"sessionId": request["sessionId"], "status": "TERMINATED"}

    assert (
        _cleanup(NativeCleanup(), "original-session", records, clock=lambda: 1.0, sleep=lambda _seconds: None)
        is True
    )
    assert [name for name, _request in calls] == ["stop", "get", "get"]
    assert all(request["sessionId"] == "original-session" for _name, request in calls)
    saved = [json.loads(p.read_bytes()) for p in sorted(directory.iterdir())]
    kinds = [p.name.split("-", 1)[1][:-5] for p in sorted(directory.iterdir())]
    assert kinds == [
        "stop-attempt",
        "stop-accepted",
        "get-attempt",
        "get-failed",
        "get-attempt",
        "get-observed",
    ]
    failure = saved[3]
    assert failure["error_type"] == "ClientError"
    assert failure["service_error"] == {
        "code": "ThrottlingException",
        "http_status": 429,
        "request_id": "fixture-request",
    }
    assert "neutral withheld" not in json.dumps(saved)
    assert saved[-1]["status"] == "TERMINATED"


def test_cleanup_read_reconciliation_keeps_original_twelve_poll_bound(tmp_path):
    from botocore.exceptions import ClientError

    from amlas.evaluation.coding_pair_runner import _cleanup, _SessionRecords

    directory = tmp_path / "session"
    directory.mkdir()
    records = _SessionRecords(directory, "fixture-evaluation", "HumanEval/0", "base")
    calls = []

    class NativeCleanup:
        def stop_code_interpreter_session(self, **request):
            calls.append("stop")
            return {"sessionId": request["sessionId"]}

        def get_code_interpreter_session(self, **request):
            calls.append("get")
            raise ClientError(
                {
                    "Error": {"Code": "ThrottlingException", "Message": "withheld"},
                    "ResponseMetadata": {"HTTPStatusCode": 429},
                },
                "GetCodeInterpreterSession",
            )

    assert (
        _cleanup(NativeCleanup(), "original-session", records, clock=lambda: 1.0, sleep=lambda _: None)
        is False
    )
    assert calls == ["stop"] + ["get"] * 12
    kinds = [p.name.split("-", 1)[1][:-5] for p in sorted(directory.iterdir())]
    assert kinds.count("get-attempt") == kinds.count("get-failed") == 12
    assert kinds[-1] == "cleanup-unknown"


def test_cleanup_never_reconciles_foreign_terminal_response(tmp_path):
    from amlas.evaluation.coding_pair_runner import _cleanup, _SessionRecords

    directory = tmp_path / "session"
    directory.mkdir()
    records = _SessionRecords(directory, "fixture-evaluation", "HumanEval/0", "base")
    calls = []

    class NativeCleanup:
        def stop_code_interpreter_session(self, **request):
            calls.append("stop")
            return {"sessionId": request["sessionId"]}

        def get_code_interpreter_session(self, **request):
            calls.append("get")
            return {"sessionId": "foreign-session", "status": "TERMINATED"}

    assert (
        _cleanup(NativeCleanup(), "original-session", records, clock=lambda: 1.0, sleep=lambda _: None)
        is False
    )
    assert calls == ["stop", "get"]
