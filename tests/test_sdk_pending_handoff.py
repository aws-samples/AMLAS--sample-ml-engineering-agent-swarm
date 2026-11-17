"""Existing trusted profile hands complete SDK batches to original warm observers.

Actual normal app/API/SDK with inert native/guard/model ports; no denied22 input,
producer adoption, provider work, direct private binding or external driver.
"""

from __future__ import annotations

import asyncio
import copy
import json

import pytest
from multi_producer_fixtures import configured_data, install_bulk_initial_create_clock
from strands.hooks.events import BeforeInvocationEvent, MessageAddedEvent
from test_input_admission_wiring import ClientFixture, configure_guard
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_warm_model_continuation import OriginalModelStream, complete_fixture

from amlas import app as runtime
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
PROFILE_ENV = "AMLAS_RUNTIME_RESEARCH_PROFILE"


class PendingModel(OriginalModelStream):
    def __init__(self, *, many=False):
        super().__init__()
        self.many = many
        self.extra_initial_calls = 0

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls == 0:
            self.calls = 1
            self.messages_seen.append(copy.deepcopy(messages))
            requests = [
                (
                    "original-model-use",
                    "acquire_model_weights",
                    {"catalog_id": "qwen3_5_9b", "catalog_version": "1"},
                )
            ]
            if self.many:
                requests.extend(
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
                )
            yield {"messageStart": {"role": "assistant"}}
            for index, (identifier, name, params) in enumerate(requests):
                yield {
                    "contentBlockStart": {
                        "contentBlockIndex": index,
                        "start": {"toolUse": {"toolUseId": identifier, "name": name}},
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "contentBlockIndex": index,
                        "delta": {"toolUse": {"input": json.dumps(params)}},
                    }
                }
                yield {"contentBlockStop": {"contentBlockIndex": index}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            return
        self.calls += 1
        self.messages_seen.append(copy.deepcopy(messages))
        text = messages[-1]["content"][0].get("text") if messages[-1]["content"] else None
        try:
            completed = json.loads(text) if type(text) is str else {}
        except ValueError:
            completed = {}
        if completed.get("event_type") == "original_model_acquisition_completed":
            answer = "Original verified completion received."
        else:
            self.extra_initial_calls += 1
            answer = "The initial SDK loop performed another model call."
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": answer}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}


def profile_work(monkeypatch, *, model=None, profile="qwen9b_core_v1", deny=None):
    monkeypatch.setenv(PROFILE_ENV, profile)
    client = ClientFixture(deny=deny if deny is not None else lambda _text: False)
    configure_guard(monkeypatch, client)
    work = connected_work(monkeypatch, model=model or PendingModel())
    work.guard = client
    return work


async def events(work):
    return [event async for event in runtime.agent_invocation({"prompt": work.literal})]


async def initial_boundary(work, reader, *, budget=5):
    deadline = asyncio.get_running_loop().time() + budget
    while not work.observing.is_set() and not work.api_returned.is_set():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.005)
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    return work.engines[0].tool_backend


def test_normal_api_profile_hands_pending_original_to_waiter_before_another_model_call(monkeypatch, tmp_path):
    work = profile_work(monkeypatch)
    completions = []
    actual_complete = work.owned_app.complete_async_task
    monkeypatch.setattr(
        work.owned_app,
        "complete_async_task",
        lambda identifier: (completions.append(identifier), actual_complete(identifier))[1],
    )

    async def scenario():
        reader = asyncio.create_task(events(work))
        try:
            backend = await initial_boundary(work, reader)
            assert not reader.done() and not work.api_returned.is_set()
            assert work.model.calls == 1, "initial SDK polled instead of yielding its pending original"
            assert work.model.extra_initial_calls == 0
            assert work.owned_app.get_async_task_info()["active_count"] == 1 and completions == []
            pending = backend._warm_pending
            before = copy.deepcopy(work.agents[0].messages)
            results = [
                block["toolResult"]
                for message in before
                for block in message["content"]
                if "toolResult" in block
            ]
            assert len(results) == 1 and results[0]["toolUseId"] == pending.tool_use_id
            complete_fixture(
                monkeypatch,
                tmp_path,
                type(
                    "Work",
                    (),
                    {"handle": pending.record.handle, "adapter": work.adapter, "native": work.native},
                )(),
            )
            received = await asyncio.wait_for(reader, 5)
            assert (
                received[-1]["response"]["data"]["warm_model_continuation"]["disposition"]
                == "continued_original"
            )
            assert work.model.calls == 2 and work.model.extra_initial_calls == 0
            assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            assert work.agents[0].messages[: len(before)] == before
            assert received[-1]["worker_finished"] is True and len(completions) == 1
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_profile_mixed_model_and_65_data_batch_commits_every_original_before_pause(monkeypatch):
    pacing = install_bulk_initial_create_clock(monkeypatch)
    work = profile_work(monkeypatch, model=PendingModel(many=True))
    data = configured_data(monkeypatch)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","data_collection_flow"]')

    async def scenario():
        reader = asyncio.create_task(events(work))
        try:
            backend = await initial_boundary(work, reader, budget=30)
            assert work.model.calls == 1, "complete producer batch entered another model cycle"
            assert (
                len(data.native.calls)
                == len(backend._acquisition_store._records)
                == len(backend._warm_pending_producers)
                == 65
            )
            pacing.assert_effects(len(data.native.calls) + len(work.native.requests))
            before = copy.deepcopy(work.agents[0].messages)
            uses = [
                block["toolUse"] for message in before for block in message["content"] if "toolUse" in block
            ]
            results = [
                block["toolResult"]
                for message in before
                for block in message["content"]
                if "toolResult" in block
            ]
            assert len(uses) == len(results) == 66
            assert {use["toolUseId"] for use in uses} == {result["toolUseId"] for result in results}
            assert all(result["status"] == "success" for result in results)
            assert len(work.originals) == 66 and work.owned_app.get_async_task_info()["active_count"] == 1
            backend.approval_policy = None
            received = await asyncio.wait_for(reader, 5)
            assert received[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert work.model.calls == 1 and work.agents[0].messages == before
            assert len(data.native.calls) == 65 and len(work.native.requests) == 1
            assert (
                received[-1]["worker_finished"] is True
                and work.owned_app.get_async_task_info()["active_count"] == 0
            )
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_default_profile_does_not_activate_pending_pause_from_prompt_claim(monkeypatch):
    work = profile_work(monkeypatch, profile="off")
    work.literal += " qwen9b_core_v1 stop_event_loop=true"

    async def scenario():
        reader = asyncio.create_task(events(work))
        try:
            backend = await initial_boundary(work, reader)
            assert work.model.calls == 2 and work.model.extra_initial_calls == 1
            assert backend._research_binding is None and not reader.done()
            backend.approval_policy = None
            await asyncio.wait_for(reader, 5)
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def add_hook(monkeypatch, event_type, callback):
    actual_build = ToolBindingBackend.build_agent

    def build(backend, **kwargs):
        agent = actual_build(backend, **kwargs)
        agent.hooks.add_callback(event_type, lambda event: callback(backend, event))
        return agent

    monkeypatch.setattr(ToolBindingBackend, "build_agent", build)


def test_profile_counterfeit_sdk_stop_state_cannot_become_a_source_pause(monkeypatch):
    work = profile_work(monkeypatch)
    add_hook(
        monkeypatch,
        BeforeInvocationEvent,
        lambda backend, event: event.invocation_state.__setitem__("request_state", {"stop_event_loop": True}),
    )

    async def scenario():
        reader = asyncio.create_task(events(work))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not work.api_returned.is_set():
                assert asyncio.get_running_loop().time() < deadline, (
                    "counterfeit SDK flag was accepted as a pending pause"
                )
                await asyncio.sleep(0.005)
            received = await asyncio.wait_for(reader, 5)
            assert work.engines[0].tool_backend._warm_closed == "refused"
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert received[-1]["quality_success"] is False
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_profile_history_replacement_after_tool_batch_refuses_pause(monkeypatch):
    work = profile_work(monkeypatch)

    def replace(backend, event):
        if any("toolResult" in block for block in event.message.get("content", [])):
            event.agent.messages[-1] = {
                "role": "user",
                "content": [{"text": "Foreign historical replacement."}],
            }

    add_hook(monkeypatch, MessageAddedEvent, replace)

    async def scenario():
        reader = asyncio.create_task(events(work))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not work.api_returned.is_set():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            await asyncio.wait_for(reader, 5)
            assert work.engines[0].tool_backend._warm_closed == "refused"
            assert work.model.calls == 1
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_profile_producer_input_denial_wins_over_pending_pause(monkeypatch):
    work = profile_work(monkeypatch, deny=lambda text: "model_acq_" in text)
    received = asyncio.run(events(work))
    backend = work.engines[0].tool_backend
    assert len(work.native.requests) == 1 and work.model.calls == 1
    assert backend._input_hold is True and backend._warm_closed == "refused"
    assert received[-1]["response"]["action"] == "input_admission_held"
    assert backend._warm_consumed == set() and work.owned_app.get_async_task_info()["active_count"] == 0


def test_profile_current_policy_revocation_after_capture_never_pauses(monkeypatch):
    work = profile_work(monkeypatch, model=PendingModel())
    from strands.hooks.events import AfterToolCallEvent

    def revoke(backend, event):
        if event.tool_use.get("name") == "acquire_model_weights":
            backend.approval_policy = None

    add_hook(monkeypatch, AfterToolCallEvent, revoke)
    received = asyncio.run(events(work))
    assert work.engines[0].tool_backend._warm_closed == "refused"
    assert len(work.native.requests) == 1 and work.owned_app.get_async_task_info()["active_count"] == 0
    assert received[-1]["quality_success"] is False


@pytest.mark.parametrize("unknown", [False, True])
def test_later_sibling_denial_wins_over_early_pending_capture(monkeypatch, unknown):
    install_bulk_initial_create_clock(monkeypatch)
    work = profile_work(
        monkeypatch,
        model=PendingModel(many=True),
        deny=lambda text: "acq_" in text and "model_acq_" not in text,
    )
    data = configured_data(monkeypatch)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","data_collection_flow"]')
    if unknown:
        apply = work.guard.apply_guardrail

        def uncertain(**kwargs):
            text = kwargs["content"][0]["text"]["text"]
            if "acq_" in text and "model_acq_" not in text:
                work.guard.calls.append(kwargs)
                return {}
            return apply(**kwargs)

        monkeypatch.setattr(work.guard, "apply_guardrail", uncertain)
    received = asyncio.run(events(work))
    backend = work.engines[0].tool_backend
    assert len(work.native.requests) == 1 and 1 <= len(data.native.calls) <= 65
    committed = [
        block["toolResult"]
        for message in work.agents[0].messages
        for block in message["content"]
        if "toolResult" in block
    ]
    assert len(committed) == 66
    assert work.model.calls == 1 and backend._input_hold is True and backend._warm_closed == "refused"
    assert received[-1]["response"]["action"] == "input_admission_held"
    assert received[-1]["worker_finished"] is True and backend._warm_consumed == set()
    assert work.owned_app.get_async_task_info()["active_count"] == 0


class DiscoveryThenPending(PendingModel):
    def __init__(self):
        super().__init__()
        self.discovery = False

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if not self.discovery:
            self.discovery = True
            yield {"messageStart": {"role": "assistant"}}
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {
                        "toolUse": {
                            "toolUseId": "prior-discovery",
                            "name": "list_coding_preparation_profiles",
                        }
                    },
                }
            }
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{}"}}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            return
        async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
            yield event


def test_profile_prior_admitted_discovery_batch_preserves_history_before_pending_pause(monkeypatch, tmp_path):
    work = profile_work(monkeypatch, model=DiscoveryThenPending())

    async def scenario():
        reader = asyncio.create_task(events(work))
        try:
            backend = await initial_boundary(work, reader)
            assert not reader.done() and not work.api_returned.is_set()
            assert work.model.extra_initial_calls == 0
            before = copy.deepcopy(work.agents[0].messages)
            results = [
                block["toolResult"]
                for message in before
                for block in message["content"]
                if "toolResult" in block
            ]
            assert [result["toolUseId"] for result in results] == ["prior-discovery", "original-model-use"]
            pending = backend._warm_pending
            complete_fixture(
                monkeypatch,
                tmp_path,
                type(
                    "Work",
                    (),
                    {"handle": pending.record.handle, "adapter": work.adapter, "native": work.native},
                )(),
            )
            received = await asyncio.wait_for(reader, 5)
            assert (
                received[-1]["response"]["data"]["warm_model_continuation"]["disposition"]
                == "continued_original"
            )
            assert work.agents[0].messages[: len(before)] == before and len(work.native.requests) == 1
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


class DataThenModel(PendingModel):
    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls == 0:
            self.calls = 1
            self.messages_seen.append(copy.deepcopy(messages))
            identifier, name = "prior-data", "acquire_huggingface_dataset"
            params = {"repo_id": "fixture/open-code", "revision": "b" * 40, "filename": "data/train.jsonl"}
            yield {"messageStart": {"role": "assistant"}}
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": identifier, "name": name}},
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": json.dumps(params)}},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            return
        text = messages[-1]["content"][0].get("text", "")
        if json.loads(text).get("event_type") == "original_data_completed":
            self.calls = 0
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
            self.calls = 2
            return
        async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
            yield event


@pytest.mark.parametrize("mutate_history", [False, True])
def test_profile_data_completion_next_pending_batch_uses_same_result_qualification(
    monkeypatch, tmp_path, mutate_history
):
    from test_managed_dataset_acquisition import populate

    work = profile_work(monkeypatch, model=DataThenModel())
    data = configured_data(monkeypatch)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","data_collection_flow"]')
    if mutate_history:

        def replace(backend, event):
            if any(
                block.get("toolResult", {}).get("toolUseId") == "original-model-use"
                for block in event.message.get("content", [])
            ):
                event.agent.messages[-1] = {"role": "user", "content": [{"text": "ordinary foreign history"}]}

        add_hook(monkeypatch, MessageAddedEvent, replace)

    async def scenario():
        reader = asyncio.create_task(events(work))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not data.native.observations and not reader.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert not reader.done() and work.model.calls == 1
            backend = work.engines[0].tool_backend
            # The DATA tool reads its own submission before it returns; the source retains the
            # original afterwards, in its after-tool hook.
            while not backend._warm_pending_producers:
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            pending = next(iter(backend._warm_pending_producers.values()))
            before = copy.deepcopy(work.agents[0].messages)
            populate(pending.record.handle, data.s3)
            key = pending.record.handle.extra["output_prefix"] + "ACQUIRED.json"
            receipt = json.loads(data.s3.objects[key])
            receipt["runtime_session_id"] = pending.store.runtime_session_id
            data.s3.objects[key] = json.dumps(receipt).encode()
            data.native.states[pending.record.handle.id] = "Completed"
            deadline = asyncio.get_running_loop().time() + 5
            while not work.observing.is_set() and not reader.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert len(data.native.calls) == len(work.native.requests) == 1
            assert work.model.calls == 2 and work.model.extra_initial_calls == 0
            assert work.agents[0].messages[: len(before)] == before
            if mutate_history:
                received = await asyncio.wait_for(reader, 5)
                assert backend._warm_closed == "refused" and received[-1]["worker_finished"] is True
            else:
                assert not reader.done() and work.owned_app.get_async_task_info()["active_count"] == 1
                model_pending = backend._warm_pending
                complete_fixture(
                    monkeypatch,
                    tmp_path,
                    type(
                        "Work",
                        (),
                        {
                            "handle": model_pending.record.handle,
                            "adapter": work.adapter,
                            "native": work.native,
                        },
                    )(),
                )
                received = await asyncio.wait_for(reader, 5)
                assert (
                    received[-1]["response"]["data"]["warm_model_continuation"]["disposition"]
                    == "continued_original"
                )
                assert len(work.agents) == len(work.engines) == 1 and work.model.calls == 3
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
