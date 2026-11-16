"""A status poll never blocks the app's event loop and never starts a second copy of the same work.

`agent_invocation` is an async generator on the AgentCore app's single event loop. `get_prompt_run_status` called
`_read_prompt_run` directly, and for a run in the SFT stage that re-hashed the 17.99 GiB model pack on the loop thread:
heartbeats stopped, other status polls timed out and the response stream broke after 15 minutes of silence. The read now runs
on a daemon worker thread with the caller's contextvars; polls with the same payload and session share one run of it; a
poll waits a bounded time (below the client's read timeout) and then gets the closed `unavailable` answer while the work
continues for the next poll.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import pytest

from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint

RUN = "a" * 32
OTHER = "b" * 32
MARK = contextvars.ContextVar("poll_test_mark", default="unset")


def status_event(run_id, **extra):
    return {"event": "prompt_run_status", "status": "available", "run_id": run_id, "found": True, **extra}


def unavailable(run_id):
    return {"event": "prompt_run_status", "status": "unavailable", "run_id": run_id, "found": False}


class Handler:
    def __init__(self, seconds=0.0, gate=None):
        self.calls, self.seen, self.seconds, self.gate = 0, [], seconds, gate
        self.lock = threading.Lock()

    def __call__(self, run_id, session_hash, policy=None, resume=False):
        with self.lock:
            self.calls += 1
            self.seen.append((threading.current_thread().daemon, MARK.get(), session_hash, resume))
        if self.gate is not None:
            self.gate.wait(10)
        time.sleep(self.seconds)
        return status_event(run_id, progress={"n": [self.calls]})


@pytest.fixture
def handler(monkeypatch):
    monkeypatch.delenv("AMLAS_STATUS_RESPONSE_SECONDS", raising=False)
    monkeypatch.setattr(runtime, "_prompt_report_session_hash", lambda: "c" * 64)
    value = Handler()
    monkeypatch.setattr(checkpoint, "read_normal_run_status", value)
    return value


def poll(run_id=RUN, action="get_prompt_run_status"):
    async def collect():
        return [event async for event in runtime.agent_invocation({"action": action, "run_id": run_id})]

    return collect()


def test_a_blocking_status_read_leaves_the_event_loop_serving(handler):
    handler.seconds = 1.5

    async def main():
        loop = asyncio.get_running_loop()
        gaps, stop = [], asyncio.Event()

        async def ticker():
            last = loop.time()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                now = loop.time()
                gaps.append(now - last)
                last = now

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)
        events = await poll()
        stop.set()
        await task
        return events, max(gaps)

    events, worst = asyncio.run(main())
    assert events[0]["status"] == "available" and handler.calls == 1
    assert worst < 0.5, f"the event loop was blocked for {worst:.2f} s by the status read"


def test_the_read_runs_on_a_daemon_thread_with_the_callers_context(handler):
    async def main():
        MARK.set("caller")
        return await poll()

    asyncio.run(main())
    assert handler.seen == [(True, "caller", "c" * 64, False)]


def test_polls_with_the_same_payload_share_one_run_and_get_independent_answers(handler):
    handler.gate = threading.Event()

    async def main():
        first = asyncio.ensure_future(poll())
        second = asyncio.ensure_future(poll())
        await asyncio.sleep(0.3)
        handler.gate.set()
        return await first, await second

    one, two = asyncio.run(main())
    assert handler.calls == 1 and one == two
    assert one[0] is not two[0] and one[0]["progress"] is not two[0]["progress"]
    asyncio.run(poll())
    assert handler.calls == 2, "a finished run must not be served to a later poll"


def test_different_runs_do_not_wait_for_each_other(handler, monkeypatch):
    barrier = threading.Barrier(2, timeout=5)
    original = handler.__call__

    def meet(run_id, session_hash, policy=None, resume=False):
        barrier.wait()
        return original(run_id, session_hash, policy, resume)

    monkeypatch.setattr(checkpoint, "read_normal_run_status", meet)

    async def main():
        return await asyncio.gather(poll(RUN), poll(OTHER))

    results = asyncio.run(main())
    assert [r[0]["run_id"] for r in results] == [RUN, OTHER]


def test_a_cancelled_poll_does_not_cancel_the_shared_work(handler):
    handler.gate = threading.Event()

    async def main():
        first = asyncio.ensure_future(poll())
        second = asyncio.ensure_future(poll())
        await asyncio.sleep(0.2)
        first.cancel()
        await asyncio.sleep(0.05)
        handler.gate.set()
        return await second

    assert asyncio.run(main())[0]["status"] == "available" and handler.calls == 1


def test_a_poll_waits_a_bounded_time_then_answers_unavailable_while_the_work_continues(handler, monkeypatch):
    monkeypatch.setenv("AMLAS_STATUS_RESPONSE_SECONDS", "0.3")
    handler.gate = threading.Event()

    async def main():
        started = time.monotonic()
        first = await poll()
        waited = time.monotonic() - started
        later = asyncio.ensure_future(poll())
        await asyncio.sleep(0.1)
        monkeypatch.setenv("AMLAS_STATUS_RESPONSE_SECONDS", "5")
        handler.gate.set()
        return first, waited, await later

    first, waited, later = asyncio.run(main())
    assert first == [unavailable(RUN)] and 0.25 < waited < 2.0
    assert later[0]["status"] == "available" or later == [unavailable(RUN)]
    assert handler.calls == 1, "the poll after the timeout must join the running work"


def test_report_reads_also_run_off_the_loop(monkeypatch):
    monkeypatch.setattr(runtime, "_prompt_report_session_hash", lambda: "c" * 64)
    seen = []

    def slow(payload):
        seen.append(threading.current_thread().daemon)
        time.sleep(0.5)
        return {
            "event": "prompt_report",
            "status": "unavailable",
            "run_id": payload["run_id"],
            "found": False,
        }

    monkeypatch.setattr(runtime, "_read_prompt_report", slow)

    async def main():
        loop = asyncio.get_running_loop()
        stop, gaps = asyncio.Event(), []

        async def ticker():
            last = loop.time()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                gaps.append(loop.time() - last)
                last = loop.time()

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)
        events = [e async for e in runtime.agent_invocation({"action": "get_prompt_report", "run_id": RUN})]
        stop.set()
        await task
        return events, max(gaps)

    events, worst = asyncio.run(main())
    assert events[0]["event"] == "prompt_report" and seen == [True] and worst < 0.5


def test_invalid_requests_still_answer_their_closed_errors(monkeypatch):
    monkeypatch.setattr(runtime, "_prompt_report_session_hash", lambda: "c" * 64)
    events = asyncio.run(poll("not-a-run-id"))
    assert events == [{"event": "error", "error_code": "invalid_run_status_request"}]
