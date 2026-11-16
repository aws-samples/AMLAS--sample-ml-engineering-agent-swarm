"""A late resume of a run whose original worker already finished starts nothing.

With no worker stop, the original worker waits for its SFT job, sees it complete, runs the tail and retains its final
report; it never writes a continuation record. The readiness record for that job can still be delivered later. Such a
late resume must not restore the run from the SFT checkpoint, claim it, call the model, submit a generation, log a
restore line or retain a second report.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from normal_sft_cold_fixture import (
    close,
    cloud_data,
    finish_generation_pair,
    original_precommit,
    setup,
    wait_for,
)
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas import app as runtime
from amlas.async_callback import lambda_handler as callback_source

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")

# Upper bound for the wait on the stages a restored run must still reach. A slow
# machine needs the room; a stage that is never reached still fails when it ends.
LIVENESS_WAIT_SECONDS = 180


class Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


async def collect(payload):
    return [event async for event in runtime.agent_invocation(payload)]


def test_late_resume_after_the_original_finished_the_run_starts_nothing(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        lines = Lines()
        logging.getLogger().addHandler(lines)
        try:
            await original_precommit(state, monkeypatch)
            cloud_data(state, completed=True)  # installs the completed SFT output the original reads
            state.native.states[state.original.id] = "Completed"
            terminal = await finish_generation_pair(state, state.reader)
            assert terminal[-1]["event"] == "run_completed"
            before = await collect({"action": "get_prompt_report", "run_id": state.run_id})
            calls, training, processing = (
                state.model.calls,
                len(state.native.calls),
                len(state.native.processing),
            )
            assert not any(m.startswith("normal_restore_completed") for m in lines.messages)
            events = await collect({"action": "resume_prompt_run", "run_id": state.run_id})
            after = await collect({"action": "get_prompt_report", "run_id": state.run_id})
            seen = [
                (e.get("event"), e.get("state") or e.get("status"))
                for e in events
                if e.get("event") != "heartbeat"
            ]
            assert not any(e.get("event") == "run_started" for e in events), (
                f"a resume of a run the original worker already finished started a continuation: {seen}"
            )
            assert state.model.calls == calls, (
                f"the late resume called the model {state.model.calls - calls} more times"
            )
            assert (
                state.native.duplicates == []
                and len(state.native.calls) == training
                and len(state.native.processing) == processing
            )
            assert not any(m.startswith("normal_restore_completed") for m in lines.messages)
            assert after == before, "the retained report changed after the late resume"
        finally:
            logging.getLogger().removeHandler(lines)
            await close(state)

    asyncio.run(scenario())


def _acknowledged(event, run_id):
    """The stream Lambda's own parser and acknowledgement set decide whether this first event ends the retries."""
    state = callback_source._resume_event_state(json.dumps(event, separators=(",", ":")), run_id)
    return state in callback_source._RESUME_ACKNOWLEDGED


def test_late_resume_keeps_original_active_while_busy_and_is_acknowledged_once_the_original_finished(
    monkeypatch, tmp_path
):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        lines = Lines()
        logging.getLogger().addHandler(lines)
        try:
            await original_precommit(state, monkeypatch)
            busy = await collect({"action": "resume_prompt_run", "run_id": state.run_id})
            assert busy == [{"event": "resume_state", "state": "original_active", "run_id": state.run_id}]
            assert not _acknowledged(busy[0], state.run_id), "a busy original must keep the record retrying"
            cloud_data(state, completed=True)
            state.native.states[state.original.id] = "Completed"
            terminal = await finish_generation_pair(state, state.reader)
            assert terminal[-1]["event"] == "run_completed"
            calls, training, processing = (
                state.model.calls,
                len(state.native.calls),
                len(state.native.processing),
            )
            for _ in range(2):
                events = await collect({"action": "resume_prompt_run", "run_id": state.run_id})
                assert events == [{"event": "resume_state", "state": "completed", "run_id": state.run_id}]
                assert _acknowledged(events[0], state.run_id), "the stream record would retry a finished run"
            assert state.model.calls == calls
            assert len(state.native.calls) == training and len(state.native.processing) == processing
            assert not any(m.startswith("normal_restore_completed") for m in lines.messages)
        finally:
            logging.getLogger().removeHandler(lines)
            await close(state)

    asyncio.run(scenario())


def test_resume_after_an_original_that_ended_before_its_sft_result_still_restores_and_continues(
    monkeypatch, tmp_path
):
    """The original ends on its own while its SFT job runs (no SFT result reached its model): a later resume in the
    same process restores the run and starts the continuation as before."""

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        lines = Lines()
        logging.getLogger().addHandler(lines)
        try:
            await original_precommit(state, monkeypatch)
            await close_original_worker(state.work, state.reader)
            calls, engines = state.model.calls, len(state.work.engines)
            cloud_data(state, completed=True)
            state.native.states[state.original.id] = "Completed"
            reader = asyncio.ensure_future(collect({"action": "resume_prompt_run", "run_id": state.run_id}))
            await wait_for(state, reader, lambda: len(state.work.engines) > engines, "replacement engine")
            deadline = asyncio.get_running_loop().time() + LIVENESS_WAIT_SECONDS
            events = await finish_generation_pair(state, reader, deadline=deadline)
            seen = [
                (e.get("event"), e.get("state") or e.get("status"))
                for e in events
                if e.get("event") != "heartbeat"
            ]
            assert seen[0] == ("resume_state", "restoring"), seen
            assert ("run_started", None) in seen, (
                f"a resume of an interrupted original did not continue: {seen}"
            )
            assert state.model.calls > calls
            assert any(m.startswith("normal_restore_completed") for m in lines.messages)
        finally:
            logging.getLogger().removeHandler(lines)
            await close(state)

    asyncio.run(scenario())
