"""Warm continuation must retain genuine next and concurrent SDK producers."""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest
from multi_producer_fixtures import SequentialProducerStream, configured_data
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_warm_model_continuation import complete_fixture

from amlas import app as runtime

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


@pytest.mark.parametrize("concurrent", [False, True], ids=["next-data-during-model-resume", "concurrent-model-and-data"])
def test_actual_normal_sdk_api_keeps_genuine_data_originals_after_model_completion(monkeypatch, tmp_path, concurrent):
    model = SequentialProducerStream(concurrent=concurrent)
    work = connected_work(monkeypatch, model=model)
    data = configured_data(monkeypatch)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","data_collection_flow"]')

    async def scenario():
        events = []
        async def read():
            async for event in runtime.agent_invocation({"prompt": work.literal}):
                events.append(event)
            return events
        reader = asyncio.create_task(read())
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not (work.api_returned.is_set() or work.observing.is_set()):
                assert asyncio.get_running_loop().time() < deadline, "genuine original MODEL did not reach observation"
                await asyncio.sleep(0.005)
            backend = work.engines[0].tool_backend
            model_results = [value for identifier, value in work.originals if identifier == "original-model-use"]
            assert len(model_results) == 1 and model_results[0]["accepted"] is True
            before = copy.deepcopy(work.agents[0].messages)
            graph = backend._warm_graph()
            original_model = model_results[0]
            handle = backend._model_acquisition_store._records[original_model["request_ref"]].handle
            complete_fixture(monkeypatch, tmp_path, SimpleNamespace(handle=handle, adapter=work.adapter, native=work.native))
            deadline = asyncio.get_running_loop().time() + 5
            while len([value for identifier, value in work.originals if identifier in {"original-data-one", "original-data-two"}]) < 2 and not reader.done():
                assert asyncio.get_running_loop().time() < deadline, "registered DATA tools did not return their original source results"
                await asyncio.sleep(0.005)
            data_results = [value for identifier, value in work.originals if identifier in {"original-data-one", "original-data-two"}]
            assert len(data_results) == len(data.native.calls) == 2, "genuine DATA tools must return two actual source originals"
            assert all(value.get("accepted") is True for value in data_results), data_results
            acquired = backend._acquisition_store
            assert len(acquired._records) == 2
            records = tuple(acquired._records.values())
            assert all(record.adapter is data.adapter for record in records)
            assert {record.handle.id for record in records} == {request["ProcessingJobName"] for request in data.native.calls}
            assert all(record.handle.extra["runtime_session_id"] == acquired.runtime_session_id for record in records)
            assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            assert all(a is b for a, b in zip(graph, backend._warm_graph(), strict=True))
            assert work.agents[0].messages[:len(before)] == before
            # Wait through the actual completed MODEL resume, not merely entry
            # into its SDK cycle. A baseline return is the behavior under test.
            deadline = asyncio.get_running_loop().time() + 5
            while model.calls < (4 if not concurrent else 3) and not work.api_returned.is_set():
                assert asyncio.get_running_loop().time() < deadline, "same Agent did not finish the completed MODEL turn"
                await asyncio.sleep(0.005)
            for _ in range(20):
                if work.api_returned.is_set():
                    break
                await asyncio.sleep(0.005)
            assert not work.api_returned.is_set(), "normal API stopped at first MODEL despite genuine pending DATA originals"
            assert not reader.done()
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            assert work.owned_app.get_current_ping_status().value == "HealthyBusy"
            for _ in range(4):
                await asyncio.sleep(0.01)
            assert {record.handle.id for record in records} <= set(data.native.observations)
            assert len(data.native.calls) == 2 and len(work.seen_prompts) == 1
            assert work.seen_prompts[0][0] == work.literal
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
