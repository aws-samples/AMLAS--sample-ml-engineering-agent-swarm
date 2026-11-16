"""Direct API cancellation must close/join its original observation, never AWS jobs."""

from __future__ import annotations

import asyncio
import copy
import threading
from types import SimpleNamespace

import pytest
from test_runtime_connected_warm_model import connected_work
from test_runtime_warm_model_continuation import complete_fixture

from amlas import api
from amlas import app as runtime


@pytest.mark.parametrize("boundary", ["describe", "sleep", "sdk_resume"])
def test_direct_api_cancellation_closes_and_joins_exact_original_before_scope_exit(monkeypatch, tmp_path, boundary):
    work = connected_work(monkeypatch)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []
    if boundary == "describe":
        actual_describe = work.native.describe_processing_job

        def blocked_describe(**kwargs):
            calls.append("describe")
            entered.set()
            try:
                assert release.wait(10), "owned native barrier was not released"
                return actual_describe(**kwargs)
            finally:
                finished.set()

        monkeypatch.setattr(work.native, "describe_processing_job", blocked_describe)
    elif boundary == "sleep":
        actual_sleep = asyncio.sleep

        async def observed_sleep(delay, *args, **kwargs):
            if delay == 15.0:
                entered.set()
                try:
                    await actual_sleep(delay, *args, **kwargs)
                finally:
                    finished.set()
            else:
                await actual_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(asyncio, "sleep", observed_sleep)
    else:
        actual_stream = work.model.stream

        async def blocked_resume(messages, tool_specs=None, system_prompt=None, **kwargs):
            if work.model.calls < 2:
                async for event in actual_stream(messages, tool_specs, system_prompt, **kwargs):
                    yield event
                return
            calls.append("sdk_resume")
            entered.set()
            try:
                assert release.wait(10), "owned SDK barrier was not released"
                async for event in actual_stream(messages, tool_specs, system_prompt, **kwargs):
                    yield event
            finally:
                finished.set()

        monkeypatch.setattr(work.model, "stream", blocked_resume)
        # Control only the15s clock seam before the real API task can sleep.
        actual_sleep = asyncio.sleep
        async def fast_pending_sleep(delay, *args, **kwargs):
            await actual_sleep(0.01 if delay == 15.0 else delay, *args, **kwargs)
        monkeypatch.setattr(asyncio, "sleep", fast_pending_sleep)

    async def scenario():
        task = asyncio.create_task(api.run_prompt(work.literal, policy=runtime._runtime_prompt_policy()))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not work.accepted.is_set() or work.model.calls < 2:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            backend, agent = work.engines[0].tool_backend, work.agents[0]
            original = work.originals[0][1]
            reference = original["request_ref"]
            handle = backend._model_acquisition_store._records[reference].handle
            if boundary == "sdk_resume":
                complete_fixture(monkeypatch, tmp_path, SimpleNamespace(handle=handle, adapter=work.adapter, native=work.native))
            deadline = asyncio.get_running_loop().time() + 5
            while not entered.is_set():
                assert asyncio.get_running_loop().time() < deadline, "original boundary did not block"
                await asyncio.sleep(0.005)
            before = copy.deepcopy(agent.messages)
            task.cancel()
            await asyncio.sleep(0.02)
            if boundary != "sleep":
                assert not task.done(), "API propagated cancellation while its copied-context observer was still live"
                assert not finished.is_set()
                # Repeated cancellation must not abandon the exact join.
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert finished.is_set()
            assert backend._warm_closed == "refused"
            assert backend._model_acquisition_store._records[reference].handle is handle
            assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            assert agent.callback_handler is work.callbacks[0]
            if boundary != "sdk_resume":
                assert work.model.calls == 2 and agent.messages == before
            reads, model_calls = list(work.observations), work.model.calls
            assert backend._observe_pending_model_once().disposition == "refused"
            assert work.observations == reads and work.model.calls == model_calls
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # A failing old implementation still leaves a real thread; drain
            # only this neutral owned barrier before fixture teardown.
            deadline = asyncio.get_running_loop().time() + 5
            while entered.is_set() and not finished.is_set():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)

    asyncio.run(scenario())
