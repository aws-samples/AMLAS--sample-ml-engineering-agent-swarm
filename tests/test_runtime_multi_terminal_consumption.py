"""A failed source original is consumed once without closing genuine siblings."""

from __future__ import annotations

import asyncio
import copy

import pytest
from multi_producer_fixtures import SequentialProducerStream, configured_data
from test_runtime_connected_warm_model import close_original_worker, connected_work

from amlas import app as runtime
from amlas.tui.chat import engine as engine_source

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


@pytest.mark.parametrize("siblings", [False, True], ids=["terminal-model-only", "terminal-model-live-data-siblings"])
def test_actual_sdk_api_consumes_failed_model_once_and_preserves_other_originals(monkeypatch, siblings):
    model = SequentialProducerStream(concurrent=siblings)
    work = connected_work(monkeypatch, model=model)
    data = configured_data(monkeypatch)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","data_collection_flow"]')
    terminal_checks = []
    actual_observe = engine_source._WARM_CHAT_ENGINE_TYPE._observe_pending_model_once

    async def checked_observe(engine):
        step = await actual_observe(engine)
        if step.disposition == "terminal_failed":
            backend = engine.tool_backend
            # Check the real source transition before API can spin on a stale
            # alias. A failing check ends only this neutral fixture worker.
            valid = backend._warm_pending is None
            terminal_checks.append(valid)
            assert valid, "failed original MODEL remained pending after its verified terminal observation"
            if siblings:
                assert len(backend._warm_pending_producers) == 2
                assert backend._warm_closed is None, "one failed MODEL closed unrelated genuine DATA originals"
        return step

    monkeypatch.setattr(engine_source._WARM_CHAT_ENGINE_TYPE, "_observe_pending_model_once", checked_observe)

    async def scenario():
        async def read():
            return [event async for event in runtime.agent_invocation({"prompt": work.literal})]
        reader = asyncio.create_task(read())
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not work.observing.is_set() and not reader.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            backend = work.engines[0].tool_backend
            original = next(value for identifier, value in work.originals if identifier == "original-model-use")
            assert original["accepted"] is True
            assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            store = backend._model_acquisition_store
            record = store._records[original["request_ref"]]
            assert backend._warm_pending.record is record
            assert record.handle.id == work.native.requests[0]["ProcessingJobName"]
            graph = backend._warm_graph()
            if siblings:
                deadline = asyncio.get_running_loop().time() + 5
                while len([value for identifier, value in work.originals if identifier.startswith("original-data-")]) < 2:
                    assert not reader.done() and asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.005)
                results = [value for identifier, value in work.originals if identifier.startswith("original-data-")]
                assert all(value["accepted"] is True for value in results)
                records = tuple(backend._acquisition_store._records.values())
                assert len(records) == len(data.native.calls) == len(backend._warm_pending_producers) == 2
                assert all(item.adapter is data.adapter for item in records)
                assert {item.handle.id for item in records} == set(data.native.requests)
                pending = dict(backend._warm_pending_producers)
            before = copy.deepcopy(work.agents[0].messages)
            work.native.state = "Failed"
            deadline = asyncio.get_running_loop().time() + 5
            while not terminal_checks and not reader.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert terminal_checks == [True], "actual verified terminal MODEL was not consumed from pending ownership"
            assert backend._warm_pending is None
            assert work.model.calls == 2 and len(work.native.requests) == 1
            assert backend._warm_graph() == graph and store._records[original["request_ref"]] is record
            assert work.agents[0].messages == before
            assert len(work.seen_prompts) == 1
            if siblings:
                deadline = asyncio.get_running_loop().time() + 5
                while not set(data.native.requests) <= set(data.native.observations):
                    assert not reader.done() and asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.005)
                assert not work.api_returned.is_set() and not reader.done()
                assert work.owned_app.get_async_task_info()["active_count"] == 1
                assert backend._warm_closed is None
                assert backend._warm_pending_producers == pending
                assert all(backend._warm_pending_producers[key] is value for key, value in pending.items())
                assert len(data.native.calls) == 2
            else:
                events = await asyncio.wait_for(reader, 5)
                assert events[-1]["response"]["data"]["warm_model_continuation"] == {"disposition": "terminal_failed"}
                assert work.owned_app.get_async_task_info()["active_count"] == 0
                assert backend._warm_pending_producers == {}
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
