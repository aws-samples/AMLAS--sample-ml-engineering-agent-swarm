"""Recheck actual worker completion after legitimate diagnostic consumption."""

from __future__ import annotations

import asyncio
import time

import pytest
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_original_model_diagnostic_events import (
    diagnostic_events,
    read_events,
    source_http400_attempt,
)

from amlas import api

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def test_actual_worker_finish_after_consumer_before_yield_suppresses_stale_diagnostic(monkeypatch):
    work = connected_work(monkeypatch)
    source_http400_attempt(monkeypatch, work)
    actual_take = api._PromptModelDiagnosticSlot.take_current
    consumed = []

    def worker_finishes_after_current_read(slot):
        value = actual_take(slot)
        if value is not None:
            consumed.append(value)
            # A normal current-policy refusal naturally closes the real original
            # API worker. Do not cancel a worker, invent a handle or replay Create.
            work.engines[0].tool_backend.approval_policy = None
            deadline = time.monotonic() + 5
            while work.owned_app.get_async_task_info()["active_count"]:
                assert time.monotonic() < deadline, "ordinary original worker did not finish"
                time.sleep(0.001)
            assert work.api_returned.is_set()
        return value

    monkeypatch.setattr(api._PromptModelDiagnosticSlot, "take_current", worker_finishes_after_current_read)

    async def scenario():
        events = []
        reader = asyncio.create_task(read_events(work, events))
        try:
            await asyncio.wait_for(reader, 5)
            assert len(consumed) == 1
            assert diagnostic_events(events) == [], "finished worker published a stale worker_finishedFalse diagnostic"
            assert events[-1]["event"] == "run_completed" and events[-1]["worker_finished"] is True
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            assert len(work.seen_prompts) == 1 and work.seen_prompts[0][0] == work.literal
            assert work.model.calls == 2
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
