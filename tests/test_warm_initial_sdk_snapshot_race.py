"""Controlled real SDK hook boundary reproduces the pre-settled history snapshot."""

from __future__ import annotations

import asyncio
import copy
import threading

import pytest
from multi_producer_fixtures import (
    SequentialProducerStream,
    configured_data,
    install_bulk_initial_create_clock,
)
from strands.hooks import HookRegistry
from strands.hooks.events import AfterToolCallEvent
from test_runtime_connected_warm_model import close_original_worker, connected_work

from amlas import app as runtime
from amlas.tui.chat import engine as engine_source

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def test_actual_65_capture_snapshot_precedes_settled_initial_sdk_history(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")

    class ManyData(SequentialProducerStream):
        @staticmethod
        def data_calls():
            return [
                (
                    "original-data-" + str(index),
                    "acquire_huggingface_dataset",
                    {
                        "repo_id": "fixture/open-code",
                        "revision": "b" * 40,
                        "filename": "data/shard-" + str(index) + ".jsonl",
                    },
                )
                for index in range(65)
            ]

    pacing = install_bulk_initial_create_clock(monkeypatch)
    work = connected_work(monkeypatch, model=ManyData(concurrent=True))
    data = configured_data(monkeypatch)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","data_collection_flow"]')
    settled, entered, release = threading.Event(), threading.Event(), threading.Event()
    lock = threading.Lock()
    actual_process = engine_source._WARM_CHAT_ENGINE_TYPE.process_message

    async def completed_initial_turn(engine, message, **kwargs):
        response = await actual_process(engine, message, **kwargs)
        settled.set()
        return response

    monkeypatch.setattr(engine_source._WARM_CHAT_ENGINE_TYPE, "process_message", completed_initial_turn)
    actual_add = HookRegistry.add_callback

    def add_callback(registry, event_type, callback, **options):
        # The SDK registers some callbacks with keyword options (Strands 1.65: order=); pass them on.
        if event_type is AfterToolCallEvent and callback.__name__ == "capture_original":
            actual_capture = callback

            def capture_then_hold(event):
                result = actual_capture(event)
                if event.tool_use["toolUseId"].startswith("original-data-"):
                    with lock:
                        chosen = (
                            len([value for use, value in work.originals if use.startswith("original-data-")])
                            == 65
                            and not entered.is_set()
                        )
                        if chosen:
                            entered.set()
                    if chosen:
                        assert release.wait(30), "controlled initial SDK hook must be released"
                return result

            return actual_add(registry, event_type, capture_then_hold, **options)
        return actual_add(registry, event_type, callback, **options)

    monkeypatch.setattr(HookRegistry, "add_callback", add_callback)

    async def scenario():
        async def collect():
            return [event async for event in runtime.agent_invocation({"prompt": work.literal})]

        reader = asyncio.create_task(collect())
        try:
            deadline = asyncio.get_running_loop().time() + 30
            while not entered.is_set():
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert not settled.is_set()
            original_results = [value for use, value in work.originals if use.startswith("original-data-")]
            assert len(original_results) == 65 and all(
                value["accepted"] is True for value in original_results
            )
            backend = work.engines[0].tool_backend
            records = tuple(backend._acquisition_store._records.values())
            assert len(records) == len(data.native.calls) == 65
            assert {record.handle.id for record in records} == set(data.native.requests)
            before = copy.deepcopy(work.agents[0].messages)
            release.set()
            while not settled.is_set():
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert work.model.calls == 2 and len(work.native.requests) == 1
            pacing.assert_effects(len(data.native.calls) + len(work.native.requests))
            assert len(backend._warm_pending_producers) == 65 and not reader.done()
            # A source capture is earlier than the SDK's ordinary history append.
            assert work.agents[0].messages != before
            assert len(work.agents[0].messages) == len(before) + 2
            assert work.agents[0].messages[: len(before)] == before
            # The same real-return gate precedes this immutable snapshot.
            assert settled.is_set()
            settled_history = copy.deepcopy(work.agents[0].messages)
            pending = dict(backend._warm_pending_producers)
            keep = records[-1].handle.id
            for name in data.native.states:
                if name != keep:
                    data.native.states[name] = "Failed"
            work.native.state = "Failed"
            deadline = asyncio.get_running_loop().time() + 30
            while len(backend._warm_pending_producers) != 1 or keep not in data.native.observations:
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert backend._warm_pending is None and backend._warm_closed is None
            assert all(value in pending.values() for value in backend._warm_pending_producers.values())
            assert all(
                data.native.observation_states.count((name, "Failed")) == 1
                for name in data.native.states
                if name != keep
            )
            assert work.model.calls == 2 and work.agents[0].messages == settled_history
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            backend.approval_policy = None
            await asyncio.wait_for(reader, 5)
            assert work.agents[0].messages == settled_history
            assert len(data.native.calls) == 65 and len(work.native.requests) == 1
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            release.set()
            await close_original_worker(work, reader)

    asyncio.run(scenario())
