"""Private diagnostic handoff lifetime over a genuine connected SDK original."""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from strands.hooks.events import AfterToolCallEvent
from test_runtime_ambiguous_model_capture import attempted_original
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_original_model_diagnostic_events import read_events, source_http400_attempt

from amlas import api
from amlas.tui.chat import tool_binding_backend as binding_source
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


@asynccontextmanager
async def held_original_slot(monkeypatch):
    work = connected_work(monkeypatch)
    captured, calls = {}, []
    entered, release = threading.Event(), threading.Event()
    slot_type = getattr(api, "_PromptModelDiagnosticSlot", None)
    actual_take = None if slot_type is None else slot_type.take_current
    if slot_type is not None:
        # Hold only optional stream publication; keep the real API/SDK worker.
        monkeypatch.setattr(slot_type, "take_current", lambda self: None)
    actual_build = ToolBindingBackend.build_agent

    def build(backend, **kwargs):
        agent = actual_build(backend, **kwargs)

        def capture(event):
            if event.tool_use.get("name") == "acquire_model_weights":
                getter = getattr(api, "_current_prompt_model_diagnostic_slot", None)
                if getter is not None:
                    captured["slot"] = getter()

        agent.hooks.add_callback(AfterToolCallEvent, capture)
        return agent

    monkeypatch.setattr(ToolBindingBackend, "build_agent", build)
    source_http400_attempt(monkeypatch, work)
    actual_describe = work.native.describe_processing_job

    def held_describe(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 2:
            entered.set()
            assert release.wait(10), "neutral original observation barrier was not released"
        return actual_describe(**kwargs)

    monkeypatch.setattr(work.native, "describe_processing_job", held_describe)
    events = []
    reader = asyncio.create_task(read_events(work, events))
    try:
        owned = await attempted_original(work)
        slot = captured.get("slot")
        assert slot is not None, "private original diagnostic slot is absent after genuine SDK capture"
        deadline = asyncio.get_running_loop().time() + 5
        while slot._candidate is None or not entered.is_set():
            assert asyncio.get_running_loop().time() < deadline, "original worker did not stage its bounded diagnostic"
            await asyncio.sleep(0.005)
        assert work.owned_app.get_async_task_info()["active_count"] == 1
        assert not work.api_returned.is_set() and work.model.calls == 2
        yield SimpleNamespace(work=work, owned=owned, slot=slot, candidate=slot._candidate,
                              take=lambda: actual_take(slot), release=release, reader=reader,
                              events=events, calls=calls)
    finally:
        release.set()
        await close_original_worker(work, reader)


def assert_original_busy(case):
    assert not case.work.api_returned.is_set()
    assert case.work.owned_app.get_async_task_info()["active_count"] == 1
    assert case.work.model.calls == 2 and len(case.work.native.requests) == 1
    assert case.owned.backend._warm_pending.record is case.owned.record
    assert case.owned.record.handle is case.owned.handle


def test_saved_source_context_cannot_publish_after_consumer_scope_closes(monkeypatch):
    async def scenario():
        async with held_original_slot(monkeypatch) as case:
            saved = case.candidate.context
            case.slot.close_consumer()
            assert saved.run(api._current_prompt_model_diagnostic_slot) is None
            assert case.take() is None
            assert_original_busy(case)
    asyncio.run(scenario())


def test_expired_diagnostic_publication_cannot_publish_or_renew_from_context_copy(monkeypatch):
    async def scenario():
        async with held_original_slot(monkeypatch) as case:
            monkeypatch.setattr(binding_source, "_model_diagnostic_now", lambda: case.candidate.deadline)
            assert case.take() is None
            assert case.take() is None
            assert_original_busy(case)
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["policy", "pin"])
def test_postqualification_original_change_suppresses_final_projection(monkeypatch, change):
    async def scenario():
        async with held_original_slot(monkeypatch) as case:
            actual = case.owned.backend._qualify_warm_original
            consumer_thread = threading.get_ident()

            def qualify(pending):
                result = actual(pending)
                if threading.get_ident() == consumer_thread:
                    if change == "policy":
                        case.owned.backend.approval_policy = None
                    else:
                        pending.record.handle.extra["neutral_pin_changed"] = True
                return result

            monkeypatch.setattr(case.owned.backend, "_qualify_warm_original", qualify)
            assert case.take() is None
            assert_original_busy(case)
    asyncio.run(scenario())


def test_two_consumers_publish_once_and_invoke_no_producer_sdk_operations(monkeypatch):
    async def scenario():
        async with held_original_slot(monkeypatch) as case:
            sdk_calls, values, errors = [], [], []

            def forbidden(**kwargs):
                sdk_calls.append(dict(kwargs))
                raise AssertionError("diagnostic consumer invoked a producer SDK operation")

            monkeypatch.setattr(case.work.native, "create_processing_job", forbidden)
            monkeypatch.setattr(case.work.native, "describe_processing_job", forbidden)
            monkeypatch.setattr(case.work.s3, "get_object", forbidden)
            barrier = threading.Barrier(3)

            def consume():
                barrier.wait(5)
                try:
                    values.append(case.take())
                except BaseException as error:
                    errors.append(error)

            workers = [threading.Thread(target=consume) for _ in range(2)]
            for worker in workers:
                worker.start()
            barrier.wait(5)
            for worker in workers:
                worker.join(5)
                assert not worker.is_alive()
            assert errors == [] and sdk_calls == []
            assert sum(value is not None for value in values) == 1
            assert case.take() is None
            assert_original_busy(case)
    asyncio.run(scenario())


def test_actual_worker_finish_invalidates_saved_publication_context(monkeypatch):
    async def scenario():
        async with held_original_slot(monkeypatch) as case:
            saved = case.candidate.context
            case.owned.backend.approval_policy = None
            case.release.set()
            await asyncio.wait_for(case.reader, 5)
            assert case.work.api_returned.is_set()
            assert case.work.owned_app.get_async_task_info()["active_count"] == 0
            assert saved.run(api._current_prompt_model_diagnostic_slot) is None
            assert case.take() is None
            assert len(case.work.native.requests) == 1 and case.work.model.calls == 2
    asyncio.run(scenario())


def test_optional_publication_qualification_failure_preserves_original_worker(monkeypatch):
    async def scenario():
        async with held_original_slot(monkeypatch) as case:
            def refuse(pending):
                raise RuntimeError("ordinary optional publication refusal")
            monkeypatch.setattr(case.owned.backend, "_qualify_warm_original", refuse)
            assert case.take() is None
            assert_original_busy(case)
    asyncio.run(scenario())


def test_publication_process_control_propagates_without_completing_original_worker(monkeypatch):
    async def scenario():
        async with held_original_slot(monkeypatch) as case:
            original = asyncio.CancelledError("ordinary publication control")
            actual = case.owned.backend._qualify_warm_original
            consumer_thread = threading.get_ident()

            def interrupted(pending):
                if threading.get_ident() == consumer_thread:
                    raise original
                return actual(pending)

            monkeypatch.setattr(case.owned.backend, "_qualify_warm_original", interrupted)
            with pytest.raises(asyncio.CancelledError) as raised:
                case.take()
            assert raised.value is original
            assert_original_busy(case)
    asyncio.run(scenario())
