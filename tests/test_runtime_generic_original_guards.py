"""Actual SDK DATA history, uncertainty and active-observation cancellation controls."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from types import SimpleNamespace

import pytest
from multi_producer_fixtures import SequentialProducerStream, configured_data
from test_managed_dataset_acquisition import populate
from test_runtime_connected_warm_model import close_original_worker, connected_work

from amlas import api
from amlas import app as runtime

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


class DataStream(SequentialProducerStream):
    @staticmethod
    def data_calls():
        return SequentialProducerStream.data_calls()[:1]


def connected_data(monkeypatch):
    work = connected_work(monkeypatch, model=DataStream(concurrent=True))
    data = configured_data(monkeypatch)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","data_collection_flow"]')
    return work, data


async def data_original(work, data):
    deadline = asyncio.get_running_loop().time() + 5
    while not [value for use, value in work.originals if use == "original-data-one"]:
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.005)
    result = next(value for use, value in work.originals if use == "original-data-one")
    assert result["accepted"] is True
    backend = work.engines[0].tool_backend
    record = backend._acquisition_store._records[result["request_ref"]]
    assert record.handle.id == data.native.calls[0]["ProcessingJobName"] and len(data.native.calls) == 1
    while not backend._warm_pending_producers:
        assert not work.api_returned.is_set() and asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.005)
    assert backend._warm_pending_producers
    work.native.state = "Failed"
    return SimpleNamespace(backend=backend, record=record, result=result, handle=record.handle)


async def original_result(work):
    """Return the original tool result from the agent's history, once it is there.

    The capture that data_original waits for happens in the after-tool hook; the framework appends
    the tool-result message to the history after that hook, on the worker's side.
    """
    deadline = asyncio.get_running_loop().time() + 5
    while True:
        for message in work.agents[0].messages:
            for block in message["content"]:
                if block.get("toolResult", {}).get("toolUseId") == "original-data-one":
                    return block["toolResult"]
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.005)


def publish_data(data, owned):
    handle = owned.handle
    populate(handle, data.s3)
    key = handle.extra["output_prefix"] + "ACQUIRED.json"
    receipt = json.loads(data.s3.objects[key])
    for name in (
        "runtime_session_id",
        "deployment_id",
        "deployment_prefix",
        "image_uri",
        "source_commit",
        "repo_id",
        "revision",
        "filename",
    ):
        receipt[name] = handle.extra[name]
    data.s3.objects[key] = json.dumps(receipt).encode()
    data.native.states[handle.id] = "Completed"


@pytest.mark.parametrize("history", ["genuine", "foreign-reference", "foreign-native-identity"])
def test_actual_data_completion_requires_original_historical_payload_and_consumes_once(monkeypatch, history):
    work, data = connected_data(monkeypatch)

    async def scenario():
        reader = asyncio.create_task(collect(work))
        try:
            owned = await data_original(work, data)
            result = await original_result(work)
            before = copy.deepcopy(work.agents[0].messages)
            if history != "genuine":
                payload = copy.deepcopy(owned.result)
                if history == "foreign-reference":
                    payload["request_ref"] = "acq_" + "e" * 32
                else:
                    payload["job_name"] = "ordinary-foreign-job"
                result["content"] = [{"json": payload}]
            publish_data(data, owned)
            events = await asyncio.wait_for(reader, 5)
            disposition = events[-1]["response"]["data"]["warm_model_continuation"]["disposition"]
            assert disposition == ("continued_original" if history == "genuine" else "refused")
            assert len(data.native.calls) == len(work.native.requests) == 1
            assert work.agents[0].callback_handler is work.callbacks[0]
            if history == "genuine":
                assert work.agents[0].messages[: len(before)] == before
                assert work.model.calls == 3
                assert len(owned.backend._warm_consumed) == 1
                assert owned.backend._warm_pending_producers == {}
                before_reads = len(data.native.observations)
                step = await work.engines[0]._observe_pending_model_once()
                assert step.disposition == "refused" and len(data.native.observations) == before_reads
            else:
                assert work.model.calls == 2
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


async def collect(work):
    return [event async for event in runtime.agent_invocation({"prompt": work.literal})]


def test_actual_generic_observation_cancel_closes_and_drains_before_sdk_busy_finishes(monkeypatch):
    work, data = connected_data(monkeypatch)
    entered, release, finished, armed = (threading.Event() for _ in range(4))
    original, actual_api = data.native.describe_processing_job, api.run_prompt
    owner, cancellations = {}, []

    def blocked(**kwargs):
        value = original(**kwargs)
        if armed.is_set():
            entered.set()
            assert release.wait(10)
        return value

    monkeypatch.setattr(data.native, "describe_processing_job", blocked)

    async def owned_api(value, **kwargs):
        owner["loop"], owner["task"] = asyncio.get_running_loop(), asyncio.current_task()
        try:
            return await actual_api(value, **kwargs)
        except asyncio.CancelledError as error:
            cancellations.append(error)
            raise
        finally:
            finished.set()

    monkeypatch.setattr(api, "run_prompt", owned_api)

    async def scenario():
        reader = asyncio.create_task(collect(work))
        try:
            owned = await data_original(work, data)
            armed.set()
            deadline = asyncio.get_running_loop().time() + 5
            while not entered.is_set():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            owner["loop"].call_soon_threadsafe(owner["task"].cancel)
            while owned.backend._warm_closed is None:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert owned.backend._warm_closed == "refused" and not finished.is_set()
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            assert work.owned_app.get_current_ping_status().value == "HealthyBusy"
            assert len(data.native.calls) == len(work.native.requests) == 1
            release.set()
            events = await asyncio.wait_for(reader, 5)
            assert len(cancellations) == 1 and finished.is_set()
            assert events[-1]["error_type"] == "CancelledError" and events[-1]["worker_finished"] is True
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert owned.backend._acquisition_store._records[owned.result["request_ref"]] is owned.record
            assert len(data.native.calls) == len(work.native.requests) == 1 and work.model.calls == 2
            assert work.agents[0].callback_handler is work.callbacks[0]
        finally:
            release.set()
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_actual_generic_resume_process_control_restores_original_callback_and_no_replay(monkeypatch):
    work, data = connected_data(monkeypatch)
    original_stream = work.model.stream

    async def cancelled_resume(messages, tool_specs=None, system_prompt=None, **kwargs):
        if work.model.calls == 2:
            work.model.calls += 1
            raise asyncio.CancelledError("ordinary neutral same-Agent resume cancellation")
        async for event in original_stream(messages, tool_specs, system_prompt, **kwargs):
            yield event

    monkeypatch.setattr(work.model, "stream", cancelled_resume)

    async def scenario():
        reader = asyncio.create_task(collect(work))
        try:
            owned = await data_original(work, data)
            before = copy.deepcopy(work.agents[0].messages)
            publish_data(data, owned)
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["error_type"] == "CancelledError" and events[-1]["worker_finished"] is True
            assert owned.backend._warm_closed == "refused"
            assert work.agents[0].callback_handler is work.callbacks[0]
            assert work.agents[0].messages[: len(before)] == before
            assert len(data.native.calls) == len(work.native.requests) == 1 and work.model.calls == 3
            assert owned.backend._acquisition_store._records[owned.result["request_ref"]] is owned.record
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
