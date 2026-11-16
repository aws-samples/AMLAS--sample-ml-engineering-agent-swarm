"""Actual SDK/API capture after one original Create loses its reply; neutral only."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
from types import SimpleNamespace

import model_acquisition_fixtures as f
import pytest
from botocore.exceptions import ClientError
from strands.hooks.events import AfterToolCallEvent
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_warm_model_continuation import complete_fixture, tool_payload

from amlas import api
from amlas import app as runtime
from amlas.backends.types import JobState
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def lose_original_create_reply(monkeypatch, work):
    actual_create = work.native.create_processing_job
    calls = []

    def create(**kwargs):
        # The existing strict native fake validates and retains this exact request.
        result = actual_create(**kwargs)
        calls.append((kwargs["ProcessingJobName"], result["ProcessingJobArn"]))
        raise TimeoutError("ordinary synthetic lost original Processing reply")

    monkeypatch.setattr(work.native, "create_processing_job", create)
    return calls


async def attempted_original(work):
    deadline = asyncio.get_running_loop().time() + 5
    while not (work.api_returned.is_set() or work.observing.is_set()):
        assert asyncio.get_running_loop().time() < deadline, "actual SDK attempt did not reach its boundary"
        await asyncio.sleep(0.005)
    assert work.accepted.is_set() and len(work.originals) == 1
    assert work.originals[0][0] == "original-model-use"
    original = work.originals[0][1]
    assert original["executed"] is True
    assert original["accepted"] is False and original["status"] == "submission_unknown"
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    backend = work.engines[0].tool_backend
    store = backend._model_acquisition_store
    reference = original["request_ref"]
    record = store._records[reference]
    handle = record.handle
    assert handle.extra["accepted"] is False and handle.extra["submission_ambiguous"] is True
    assert (original["job_name"], original["job_arn"]) == (handle.id, handle.arn)
    assert handle.id == work.native.requests[0]["ProcessingJobName"]
    pending = backend._warm_pending
    assert pending is not None, "actual ambiguous SDK original was dropped by the accepted-only capture gate"
    assert pending.store is store and pending.record is record and pending.agent is work.agents[0]
    assert pending.reference == reference and pending.tool_use_id == "original-model-use"
    assert pending.pid == os.getpid() and pending.handle_pin == backend._warm_handle_pin(handle)
    assert all(a is b for a, b in zip(pending.graph, backend._warm_graph(), strict=True))
    return SimpleNamespace(backend=backend, store=store, record=record, handle=handle, reference=reference)


def test_actual_lost_create_reply_waits_for_same_original_verified_completion(monkeypatch, tmp_path):
    work = connected_work(monkeypatch)
    creates = lose_original_create_reply(monkeypatch, work)

    async def scenario():
        reader = asyncio.create_task(collect_events(work))
        try:
            owned = await attempted_original(work)
            assert not work.api_returned.is_set() and not reader.done()
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            assert work.owned_app.get_current_ping_status().value == "HealthyBusy"
            assert work.model.calls == 2 and len(creates) == 1
            graph = owned.backend._warm_graph()
            before = copy.deepcopy(work.agents[0].messages)
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=owned.handle, adapter=work.adapter, native=work.native),
            )
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["status"] == "response_received"
            assert "Original exact-version artifacts are verified." in events[-1]["response"]["message"]
            assert work.api_returned.is_set() and work.owned_app.get_async_task_info()["active_count"] == 0
            assert len(creates) == len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            assert work.model.calls == 3 and work.agents[0].messages[0] == before[0]
            assert len(work.seen_prompts) == 1 and work.seen_prompts[0][0] == work.literal
            assert owned.store._records[owned.reference] is owned.record
            assert owned.record.handle is owned.handle
            assert owned.handle.extra["accepted"] is False
            assert owned.handle.extra["submission_ambiguous"] is True
            assert all(a is b for a, b in zip(graph, owned.backend._warm_graph(), strict=True))
            results = [
                block["toolResult"]
                for message in work.agents[0].messages
                for block in message["content"]
                if block.get("toolResult", {}).get("toolUseId") == "original-model-use"
            ]
            assert len(results) == 1
            # The original SDK pending ToolResult remains byte-for-byte data.
            assert work.agents[0].messages[: len(before)] == before
            verified = tool_payload(results[0])
            assert verified == tool_payload(
                next(
                    block["toolResult"]
                    for message in before
                    for block in message["content"]
                    if block.get("toolResult", {}).get("toolUseId") == "original-model-use"
                )
            )
            assert verified["request_ref"] == owned.reference and verified["acquired"] is False
            assert (verified["job_name"], verified["job_arn"]) == (owned.handle.id, owned.handle.arn)
            completions = [
                json.loads(block["text"])
                for message in work.agents[0].messages[len(before) :]
                if message["role"] == "user"
                for block in message["content"]
                if "text" in block
            ]
            assert len(completions) == 1
            completion = completions[0]
            assert completion["event_type"] == "original_model_acquisition_completed"
            assert (
                completion["request_ref"] == owned.reference
                and completion["tool_use_id"] == "original-model-use"
            )
            assert completion["acquired"] is True and completion["data_only"] is True
            assert completion["file_count"] == 9
            assert completion["manifest_roles"] == ["receipt", "commit", "lineage"]
            observed = owned.store.observe(owned.reference)
            assert observed["acquired"] is True and len(observed["file_refs"]) == 9
            assert set(observed["manifest_refs"]) == {"receipt", "commit", "lineage"}
            assert owned.backend._warm_pending is None
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


async def collect_events(work):
    return [event async for event in runtime.agent_invocation({"prompt": work.literal})]


def test_actual_ambiguous_missing_native_job_holds_without_create_replay(monkeypatch):
    work = connected_work(monkeypatch)
    creates = lose_original_create_reply(monkeypatch, work)
    actual_describe = work.native.describe_processing_job

    def missing(**kwargs):
        # Validate the same original request and retain the neutral read marker.
        actual_describe(**kwargs)
        raise ClientError({"Error": {"Code": "ResourceNotFound"}}, "DescribeProcessingJob")

    monkeypatch.setattr(work.native, "describe_processing_job", missing)

    async def scenario():
        reader = asyncio.create_task(collect_events(work))
        try:
            owned = await attempted_original(work)
            deadline = asyncio.get_running_loop().time() + 5
            while len(work.observations) < 2:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert not work.api_returned.is_set() and not reader.done()
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            assert work.model.calls == 2 and len(creates) == len(work.native.requests) == 1
            assert owned.store._records[owned.reference] is owned.record
            assert owned.backend._warm_pending.handle_pin == owned.backend._warm_handle_pin(owned.handle)
            while len(work.observations) < 8:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.03)
            assert work.api_returned.is_set(), "original UNKNOWN observation wait has no finite outcome"
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["response"]["data"]["warm_model_continuation"] == {
                "disposition": "submission_unresolved",
                "diagnostic": {
                    "phase": "native_observation",
                    "reason": "original_unknown_budget_exhausted",
                    "error_class": "UNKNOWN",
                },
            }
            assert len(work.observations) == 8
            assert events[-1]["worker_finished"] is True
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert (
                owned.handle.extra["accepted"] is False and owned.handle.extra["submission_ambiguous"] is True
            )
            assert owned.backend._warm_pending.reference == owned.reference
            assert owned.backend._warm_consumed == set()
            assert work.model.calls == 2 and len(creates) == len(work.native.requests) == 1
            assert owned.store._records[owned.reference] is owned.record
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode,expected", (("policy", "refused"), ("guardrail", "guardrail_intervened")))
def test_actual_ambiguous_attempt_respects_original_policy_and_guardrail_closure(monkeypatch, mode, expected):
    work = connected_work(monkeypatch, mode=mode)
    creates = lose_original_create_reply(monkeypatch, work)
    events = asyncio.run(collect_events(work))
    assert len(creates) == len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    original = work.originals[0][1]
    assert original["accepted"] is False and original["status"] == "submission_unknown"
    backend = work.engines[0].tool_backend
    handle = backend._model_acquisition_store._records[original["request_ref"]].handle
    assert handle.extra["accepted"] is False and handle.extra["submission_ambiguous"] is True
    continuation = {"disposition": expected}
    if mode == "policy":
        continuation["diagnostic"] = {
            "phase": "initial_qualification",
            "reason": "policy_refused",
            "error_class": "ValueError",
        }
    assert events[-1]["response"]["data"]["warm_model_continuation"] == continuation
    assert work.model.calls == 2 and work.owned_app.get_async_task_info()["active_count"] == 0
    if mode == "policy":
        assert work.observations == []


def test_actual_sdk_prepared_no_attempt_handle_does_not_become_pending_original(monkeypatch):
    work = connected_work(monkeypatch)

    def prepared_only(**kwargs):
        # Negative DATA control: actual pure source builder, no synthetic flag edit.
        return work.adapter.prepare_submission(**kwargs)

    monkeypatch.setattr(work.adapter, "submit", prepared_only)
    asyncio.run(collect_events(work))
    assert len(work.engines) == len(work.agents) == len(work.originals) == 1
    backend = work.engines[0].tool_backend
    original = work.originals[0][1]
    handle = backend._model_acquisition_store._records[original["request_ref"]].handle
    assert handle.extra["accepted"] is False and handle.extra["submission_ambiguous"] is False
    assert backend._warm_pending is None and work.native.requests == []
    assert work.observations == [] and work.owned_app.get_async_task_info()["active_count"] == 0


def test_public_unknown_reference_cannot_adopt_an_original_without_actual_sdk_attempt(monkeypatch):
    # Existing negative fixture returns a public claim while its actual Agent emits no tool.
    work = connected_work(monkeypatch, mode="no_original")
    asyncio.run(collect_events(work))
    backend = work.engines[0].tool_backend
    assert work.originals == [] and backend._model_acquisition_store._records == {}
    assert backend._warm_pending is None and work.native.requests == []
    assert work.observations == [] and work.owned_app.get_async_task_info()["active_count"] == 0


@pytest.mark.parametrize("lost_reply,expected", ((False, (True, False)), (True, (False, True))))
def test_native_describe_preserves_original_create_lifecycle_and_pin(monkeypatch, lost_reply, expected):
    work = connected_work(monkeypatch)
    if lost_reply:
        lose_original_create_reply(monkeypatch, work)
    handle = work.adapter.submit(
        catalog_id=f.CATALOG,
        catalog_version=f.VERSION,
        runtime_session_id=f.SESSION,
    )
    assert (handle.extra["accepted"], handle.extra["submission_ambiguous"]) == expected
    pin = ToolBindingBackend._warm_handle_pin(handle)
    with work.adapter.qualified_io(lambda: None):
        assert work.adapter.status(handle).state == JobState.RUNNING
    assert (handle.extra["accepted"], handle.extra["submission_ambiguous"]) == expected
    assert ToolBindingBackend._warm_handle_pin(handle) == pin
    assert len(work.native.requests) == 1


@pytest.mark.parametrize(
    "field,value,missing",
    (
        ("accepted", None, True),
        ("submission_ambiguous", None, False),
        ("accepted", 1, False),
        ("submission_ambiguous", 0, False),
        ("submission_ambiguous", 1, False),
        ("submission_ambiguous", True, False),
    ),
)
def test_invalid_private_lifecycle_cannot_select_warm_capture(monkeypatch, field, value, missing):
    work = connected_work(monkeypatch)
    actual_submit = work.adapter.submit

    def corrupted(**kwargs):
        handle = actual_submit(**kwargs)
        # Deliberate negative corruption after a genuine normal source submission.
        if missing:
            handle.extra.pop(field)
        else:
            handle.extra[field] = value
        return handle

    monkeypatch.setattr(work.adapter, "submit", corrupted)
    asyncio.run(collect_events(work))
    backend = work.engines[0].tool_backend
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    assert len(backend._model_acquisition_store._records) == 1
    assert backend._warm_pending is None and work.observations == []
    assert work.owned_app.get_async_task_info()["active_count"] == 0


def test_actual_public_accepted_true_does_not_admit_private_prepared_false_false(monkeypatch):
    work = connected_work(monkeypatch)
    monkeypatch.setattr(work.adapter, "submit", lambda **kwargs: work.adapter.prepare_submission(**kwargs))
    actual_build = ToolBindingBackend.build_agent

    def captured_build(backend, **kwargs):
        agent = actual_build(backend, **kwargs)

        def corrupted(event):
            if event.tool_use.get("name") == "acquire_model_weights":
                payload = tool_payload(event.result)
                payload["accepted"] = True
                event.result = {**event.result, "content": [{"json": payload}]}

        agent.hooks.add_callback(AfterToolCallEvent, corrupted)
        return agent

    monkeypatch.setattr(ToolBindingBackend, "build_agent", captured_build)
    asyncio.run(collect_events(work))
    backend = work.engines[0].tool_backend
    original = work.originals[0][1]
    assert original["accepted"] is True  # Public-negative fixture, never a private grant.
    handle = backend._model_acquisition_store._records[original["request_ref"]].handle
    assert handle.extra["accepted"] is False and handle.extra["submission_ambiguous"] is False
    assert backend._warm_pending is None and work.native.requests == [] and work.observations == []


def test_actual_ambiguous_worker_cancellation_closes_and_drains_before_sdk_task_completion(monkeypatch):
    work = connected_work(monkeypatch)
    creates = lose_original_create_reply(monkeypatch, work)
    entered, release, api_finished = threading.Event(), threading.Event(), threading.Event()
    actual_describe, actual_api = work.native.describe_processing_job, api.run_prompt
    owner, cancellations = {}, []

    def blocked_describe(**kwargs):
        result = actual_describe(**kwargs)
        entered.set()
        assert release.wait(10), "neutral original native read barrier was not released"
        return result

    async def owned_api(value, **kwargs):
        owner["loop"] = asyncio.get_running_loop()
        owner["task"] = asyncio.current_task()
        try:
            return await actual_api(value, **kwargs)
        except asyncio.CancelledError as error:
            cancellations.append(error)
            raise
        finally:
            api_finished.set()

    monkeypatch.setattr(work.native, "describe_processing_job", blocked_describe)
    monkeypatch.setattr(api, "run_prompt", owned_api)

    async def scenario():
        reader = asyncio.create_task(collect_events(work))
        try:
            owned = await attempted_original(work)
            deadline = asyncio.get_running_loop().time() + 5
            while not entered.is_set():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            owner["loop"].call_soon_threadsafe(owner["task"].cancel)
            while owned.backend._warm_closed is None:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert owned.backend._warm_closed == "refused"
            assert not api_finished.is_set() and not work.api_returned.is_set()
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            assert work.owned_app.get_current_ping_status().value == "HealthyBusy"
            assert len(creates) == len(work.native.requests) == 1 and work.model.calls == 2
            release.set()
            events = await asyncio.wait_for(reader, 5)
            assert len(cancellations) == 1 and isinstance(cancellations[0], asyncio.CancelledError)
            assert api_finished.is_set() and not work.api_returned.is_set()
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert events[-1]["error_type"] == "CancelledError" and events[-1]["worker_finished"] is True
            assert owned.store._records[owned.reference] is owned.record
            assert len(creates) == len(work.native.requests) == 1 and work.model.calls == 2
            assert work.agents[0].callback_handler is work.callbacks[0]
        finally:
            release.set()
            await close_original_worker(work, reader)

    asyncio.run(scenario())
