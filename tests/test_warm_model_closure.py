"""Bounded original-owner closure controls using actual SDK and neutral ports."""

from __future__ import annotations

import asyncio
import copy
import json
import threading

import model_acquisition_fixtures as f
import pytest
from strands.hooks.events import AfterToolCallEvent
from test_runtime_warm_model_continuation import (
    complete_fixture,
    original_work,
    step,
    tool_payload,
)


def original_results(work):
    return [
        block["toolResult"]
        for message in work.agent.messages
        for block in message["content"]
        if "toolResult" in block
    ]


def retained_original(work):
    names = (
        "_acquisition_store",
        "_model_acquisition_store",
        "_case_freeze_store",
        "_preparation_store",
        "_sft_store",
        "_generation_store",
        "_pair_store",
    )
    assert all(getattr(work.backend, name) is store for name, store in zip(names, work.stores, strict=True))
    assert work.backend._model_acquisition_store._records[work.reference].handle is work.handle
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    assert work.agent.messages[0] == work.before[0]
    results = [value for value in original_results(work) if value["toolUseId"] == "original-model-use"]
    assert len(results) == 1
    payload = tool_payload(results[0])
    assert payload["request_ref"] == work.reference
    assert payload["job_name"] == work.handle.id
    assert payload["job_arn"] == work.handle.arn


def refuse_new_agent(monkeypatch, work):
    monkeypatch.setattr(
        work.backend,
        "build_agent",
        lambda **kwargs: pytest.fail("closed original owner constructed a new SDK Agent"),
    )


@pytest.mark.parametrize(
    "closure,disposition",
    [("guardrail", "guardrail_intervened"), ("failed", "terminal_failed"), ("policy", "refused")],
)
def test_closed_original_cannot_revive_on_new_normal_user_turn(monkeypatch, closure, disposition):
    work = original_work(monkeypatch, intervened=closure == "guardrail")
    if closure == "failed":
        work.native.state = "Failed"
    elif closure == "policy":
        work.backend.approval_policy = None
    assert step(work).disposition == disposition
    before = copy.deepcopy(work.agent.messages)
    refuse_new_agent(monkeypatch, work)
    monkeypatch.setattr(
        work.native,
        "describe_processing_job",
        lambda **kwargs: pytest.fail("closed original owner read native state again"),
    )
    work.backend.chat_with_tools("A separate ordinary user turn.")
    assert step(work).disposition == disposition
    assert work.model.calls == 2
    assert work.agent.messages == before
    retained_original(work)


def test_more_than_64_actual_data_originals_retained_consumed_once_and_policy_closed(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    from multi_producer_fixtures import (
        SequentialProducerStream,
        configured_data,
        install_bulk_initial_create_clock,
    )
    from test_runtime_connected_warm_model import close_original_worker, connected_work

    from amlas import app as runtime
    from amlas.tui.chat import engine as engine_source

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
    model = ManyData(concurrent=True)
    work = connected_work(monkeypatch, model=model)
    data = configured_data(monkeypatch)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","data_collection_flow"]')

    initial_turn_settled = threading.Event()
    actual_process_message = engine_source._WARM_CHAT_ENGINE_TYPE.process_message

    async def record_initial_turn_return(engine, message, **kwargs):
        response = await actual_process_message(engine, message, **kwargs)
        initial_turn_settled.set()
        return response

    monkeypatch.setattr(engine_source._WARM_CHAT_ENGINE_TYPE, "process_message", record_initial_turn_return)

    async def scenario():
        async def read():
            return [event async for event in runtime.agent_invocation({"prompt": work.literal})]

        reader = asyncio.create_task(read())
        try:
            deadline = asyncio.get_running_loop().time() + 30
            while (
                len([value for use, value in work.originals if use.startswith("original-data-")]) < 65
                or not initial_turn_settled.is_set()
            ):
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            backend = work.engines[0].tool_backend
            results = [value for use, value in work.originals if use.startswith("original-data-")]
            assert all(value["accepted"] is True for value in results)
            records = tuple(backend._acquisition_store._records.values())
            assert len(records) == len(data.native.calls) == len(backend._warm_pending_producers) == 65
            pacing.assert_effects(len(data.native.calls) + len(work.native.requests))
            assert len({value["request_ref"] for value in results}) == 65
            assert {record.handle.id for record in records} == set(data.native.requests)
            pending = dict(backend._warm_pending_producers)
            before = copy.deepcopy(work.agents[0].messages)
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
            assert len(backend._acquisition_store._records) == 65
            assert all(value in pending.values() for value in backend._warm_pending_producers.values())
            assert all(
                data.native.observation_states.count((name, "Failed")) == 1
                for name in data.native.states
                if name != keep
            )
            assert len(data.native.calls) == 65 and len(work.native.requests) == 1
            assert work.model.calls == 2 and work.agents[0].messages == before
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            backend.approval_policy = None
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert len(backend._acquisition_store._records) == 65
            assert len(data.native.calls) == 65 and len(work.native.requests) == 1
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


@pytest.mark.parametrize("resume", ["exception", "guardrail"])
def test_actual_none_resume_restores_callback_and_closes_once(monkeypatch, tmp_path, resume):
    work = original_work(monkeypatch)
    complete_fixture(monkeypatch, tmp_path, work)
    bound_stream = work.model.stream
    callbacks = []

    def previous_callback(**event):
        callbacks.append(tuple(event))

    work.agent.callback_handler = previous_callback

    async def resumed_stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        if work.model.calls < 2:
            async for event in bound_stream(
                messages, tool_specs=tool_specs, system_prompt=system_prompt, **kwargs
            ):
                yield event
            return
        work.model.calls += 1
        work.model.messages_seen.append(copy.deepcopy(messages))
        if resume == "exception":
            raise RuntimeError("ordinary neutral resume failure")
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "Original observation remains retained."},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "guardrail_intervened"}}

    monkeypatch.setattr(work.model, "stream", resumed_stream)
    result = step(work)
    expected = "refused" if resume == "exception" else "guardrail_intervened"
    assert result.disposition == expected
    assert work.agent.callback_handler is previous_callback
    assert work.model.calls == 3
    if resume == "guardrail":
        assert result.turn.guardrail_outcome.stop_reason == "guardrail_intervened"
        assert result.turn.guardrail_outcome.action == "UNKNOWN"
    retained_original(work)
    assert work.agent.messages[: len(work.before)] == work.before
    completion_messages = [
        message for message in work.agent.messages[len(work.before) :] if message["role"] == "user"
    ]
    assert len(completion_messages) == 1
    assert len(completion_messages[0]["content"]) == 1
    completion = json.loads(completion_messages[0]["content"][0]["text"])
    assert completion == {
        "schema_version": 1,
        "event_type": "original_model_acquisition_completed",
        "data_only": True,
        "request_ref": work.reference,
        "tool_use_id": "original-model-use",
        "status": "SUCCEEDED",
        "acquired": True,
        "file_count": 9,
        "manifest_roles": ["receipt", "commit", "lineage"],
    }
    history = copy.deepcopy(work.agent.messages)
    callback_count = len(callbacks)
    assert step(work).disposition == expected
    assert work.model.calls == 3 and len(callbacks) == callback_count
    assert work.agent.messages == history
    assert work.agent.callback_handler is previous_callback
    retained_original(work)


@pytest.mark.parametrize("revoke_at", ["describe_processing_job", "list_objects_v2"])
def test_mid_original_native_io_revocation_blocks_next_sdk_boundary(monkeypatch, tmp_path, revoke_at):
    work = original_work(monkeypatch)
    complete_fixture(monkeypatch, tmp_path, work)
    reads = []
    for port, name in [
        (work.native, "describe_processing_job"),
        *[
            (work.s3, name)
            for name in (
                "get_bucket_location",
                "get_bucket_versioning",
                "list_objects_v2",
                "head_object",
                "get_object",
            )
        ],
    ]:
        bound_read = getattr(port, name)

        def current_read(*, _name=name, _bound=bound_read, **kwargs):
            reads.append(_name)
            result = _bound(**kwargs)
            if _name == revoke_at:
                work.backend.approval_policy = None
            return result

        monkeypatch.setattr(port, name, current_read)
    assert step(work).disposition == "refused"
    assert reads and reads[-1] == revoke_at and reads.count(revoke_at) == 1
    assert work.model.calls == 2 and work.agent.messages == work.before
    assert work.adapter._qualified_io.get() is None
    retained_original(work)
    before_reads = list(reads)
    assert step(work).disposition == "refused" and reads == before_reads


def test_actual_sdk_two_same_catalog_results_close_without_last_capture_winning(monkeypatch):
    work = original_work(monkeypatch)
    pending = work.backend._warm_pending
    assert pending.reference == work.reference
    bound_stream = work.model.stream
    captures = []
    work.agent.hooks.add_callback(
        AfterToolCallEvent,
        lambda event: captures.append((event.tool_use["toolUseId"], tool_payload(event.result))),
    )

    async def two_tools_stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        if work.model.calls != 2:
            async for event in bound_stream(
                messages, tool_specs=tool_specs, system_prompt=system_prompt, **kwargs
            ):
                yield event
            return
        work.model.calls += 1
        work.model.messages_seen.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        # The installed SDK owns dispatch of both tools from one actual message;
        # the existing store lock/dedup path and tool implementations are unchanged.
        for index, identifier in enumerate(("same-original-left", "same-original-right")):
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": index,
                    "start": {"toolUse": {"toolUseId": identifier, "name": "acquire_model_weights"}},
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": index,
                    "delta": {
                        "toolUse": {
                            "input": json.dumps({"catalog_id": f.CATALOG, "catalog_version": f.VERSION})
                        }
                    },
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": index}}
        yield {"messageStop": {"stopReason": "tool_use"}}

    monkeypatch.setattr(work.model, "stream", two_tools_stream)
    work.agent(None, invocation_state=work.backend._invocation_state())
    assert {identifier for identifier, _ in captures} == {"same-original-left", "same-original-right"}
    assert len(captures) == 2
    assert all(
        payload["accepted"] is True and payload["request_ref"] == work.reference for _, payload in captures
    )
    assert all(
        payload["job_name"] == work.handle.id and payload["job_arn"] == work.handle.arn
        for _, payload in captures
    )
    assert work.backend._warm_pending is pending
    assert work.backend._warm_closed == "refused"
    assert work.model.calls == 4
    assert step(work).disposition == "refused"
    assert work.model.calls == 4
    retained_original(work)


def test_actual_first_turn_model_error_after_accepted_original_closes_without_observation(monkeypatch):
    work = original_work(monkeypatch, error_after_accept=True)
    retained_original(work)
    assert work.model.calls == 2
    assert work.backend._warm_pending.reference == work.reference
    before = copy.deepcopy(work.agent.messages)
    monkeypatch.setattr(
        work.native,
        "describe_processing_job",
        lambda **kwargs: pytest.fail("failed original turn must not observe native state"),
    )
    assert work.backend._warm_closed == "refused"
    assert step(work).disposition == "refused"
    assert step(work).disposition == "refused"
    assert work.model.calls == 2 and work.agent.messages == before
    retained_original(work)


def test_real_duplicate_sdk_capture_during_original_read_permanently_stops_observer(monkeypatch, tmp_path):
    work = original_work(monkeypatch)
    complete_fixture(monkeypatch, tmp_path, work)
    entered, release = threading.Event(), threading.Event()
    observer_ids, reads, outcomes, failures = [], [], [], []
    captures = []
    work.agent.hooks.add_callback(
        AfterToolCallEvent,
        lambda event: captures.append((event.tool_use["toolUseId"], tool_payload(event.result))),
    )
    ports = [
        (work.native, "describe_processing_job"),
        *[
            (work.s3, name)
            for name in (
                "get_bucket_location",
                "get_bucket_versioning",
                "list_objects_v2",
                "head_object",
                "get_object",
            )
        ],
    ]
    for port, name in ports:
        bound_read = getattr(port, name)

        def current_read(*, _name=name, _bound=bound_read, **kwargs):
            if observer_ids and threading.get_ident() == observer_ids[0]:
                reads.append(_name)
                if len(reads) == 1:
                    entered.set()
                    assert release.wait(10), "coordinated duplicate capture did not release original read"
            return _bound(**kwargs)

        monkeypatch.setattr(port, name, current_read)

    def observe_original():
        observer_ids.append(threading.get_ident())
        try:
            outcomes.append(work.backend._observe_pending_model_once())
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=observe_original)
    worker.start()
    try:
        assert entered.wait(10), "original observer did not enter its first real SDK read"
        original_event = work.tool_events[0]
        assert type(original_event) is AfterToolCallEvent and original_event.agent is work.agent
        assert original_event.invocation_state["_amlas_model_acquisition_store"] is work.stores[1]
        original_payload = tool_payload(original_event.result)
        assert original_payload["accepted"] is True and original_payload["request_ref"] == work.reference
        assert (original_payload["job_name"], original_payload["job_arn"]) == (
            work.handle.id,
            work.handle.arn,
        )
        returned, interrupts = work.agent.hooks.invoke_callbacks(original_event)
        assert returned is original_event and not interrupts
        assert len(captures) == 1 and captures[0][0] == "original-model-use"
        assert captures[0][1] == original_payload
        assert work.backend._warm_closed == "refused"
        assert worker.is_alive() and len(reads) == 1
        history, calls = copy.deepcopy(work.agent.messages), work.model.calls
    finally:
        work.native.state = "Completed"
        release.set()
        worker.join(10)
    assert not worker.is_alive() and not failures
    assert len(outcomes) == 1 and outcomes[0].disposition == "refused"
    assert len(reads) == 1
    assert work.model.calls == calls and work.agent.messages == history
    assert step(work).disposition == "refused"
    assert len(reads) == 1 and work.model.calls == calls
    retained_original(work)


def test_actual_cancelled_resume_restores_callback_and_permanently_closes_original(monkeypatch, tmp_path):
    work = original_work(monkeypatch)
    complete_fixture(monkeypatch, tmp_path, work)
    previous = work.agent.callback_handler

    async def cancelled_stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        work.model.calls += 1
        work.model.messages_seen.append(copy.deepcopy(messages))
        raise asyncio.CancelledError("ordinary neutral cancelled SDK stream")
        yield {}

    monkeypatch.setattr(work.model, "stream", cancelled_stream)
    with pytest.raises(asyncio.CancelledError):
        step(work)
    assert work.agent.callback_handler is previous
    assert work.model.calls == 3
    assert work.backend._warm_closed == "refused"
    history = copy.deepcopy(work.agent.messages)
    assert step(work).disposition == "refused"
    assert work.model.calls == 3 and work.agent.messages == history
    retained_original(work)
