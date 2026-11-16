"""Actual app/API + SDK CASE/PREP originals; source-only fixtures, no native ML proof."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest
from case_prep_continuation_fixtures import (
    TARGET_TOOL_USE,
    publish_original_outputs,
    stage_work,
    start_original_target,
    wait_until,
)
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_warm_model_continuation import tool_payload

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def capture_original(backend, work, reference, record):
    matches = [
        pending
        for pending in backend._warm_pending_producers.values()
        if pending.kind == work.target.kind and pending.reference == reference
    ]
    assert len(matches) == 1, (
        "actual registered target producer was not retained by its private original owner"
    )
    pending = matches[0]
    assert pending.record is record and pending.store is work.target.store
    assert pending.agent is work.agents[0] and pending.tool_use_id == TARGET_TOOL_USE
    assert len(pending.graph) == 7 and backend._warm_graph() == pending.graph
    return pending


@pytest.mark.parametrize("kind", ["case", "preparation"])
def test_normal_registered_target_capture_keeps_same_worker_graph_and_original_no_replay(
    request,
    monkeypatch,
    tmp_path,
    kind,
):
    work = stage_work(request, monkeypatch, kind)

    async def scenario():
        reader = None
        try:
            reader, backend, result, reference, record, handle = await start_original_target(
                work, monkeypatch, tmp_path
            )
            pending = capture_original(backend, work, reference, record)
            await wait_until(
                lambda: bool(work.target.actual_observed), reader, "original target observer boundary missing"
            )
            assert not reader.done() and not work.api_returned.is_set()
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            assert work.owned_app.get_current_ping_status().value == "HealthyBusy"
            assert work.model.calls == 4 and len(work.engines) == len(work.agents) == 1
            assert work.target.store._records[reference] is record
            assert backend._producer_handle(pending) is handle
            assert result["frozen" if kind == "case" else "prepared"] is False
            graph = backend._warm_graph()
            backend.approval_policy = None
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert work.model.calls == 4 and backend._warm_graph() == graph
            assert len(work.target.native.creates) == len(work.native.requests) == 1
        finally:
            owned_reader = reader if reader is not None else getattr(work, "reader", None)
            if owned_reader is not None:
                await close_original_worker(work, owned_reader)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["case", "preparation"])
def test_actual_target_verification_finishes_original_with_append_only_same_agent_data(
    request,
    monkeypatch,
    tmp_path,
    kind,
):
    work = stage_work(request, monkeypatch, kind)

    async def scenario():
        reader = None
        try:
            reader, backend, result, reference, record, handle = await start_original_target(
                work, monkeypatch, tmp_path
            )
            pending = capture_original(backend, work, reference, record)
            await wait_until(
                lambda: bool(work.target.actual_observed), reader, "original target observer boundary missing"
            )
            assert not reader.done() and work.owned_app.get_async_task_info()["active_count"] == 1
            before = copy.deepcopy(work.agents[0].messages)
            graph = backend._warm_graph()
            publish_original_outputs(work, handle)
            events = await asyncio.wait_for(reader, 5)
            completed = [
                value
                for ref, value in work.target.actual_observed
                if ref == reference
                and value.get("success") is True
                and value.get("frozen" if kind == "case" else "prepared") is True
            ]
            assert completed, (
                "actual owning adapter/store artifact verifier must complete the original target"
            )
            assert (
                events[-1]["response"]["data"]["warm_model_continuation"]["disposition"]
                == "continued_original"
            )
            assert work.agents[0].messages[: len(before)] == before
            assert backend._warm_graph() == graph and work.target.store._records[reference] is record
            assert pending.agent is work.agents[0] and len(work.engines) == len(work.agents) == 1
            assert work.model.calls == 5 and len(work.target.native.creates) == len(work.native.requests) == 1
            assert len(work.seen_prompts) == 1 and work.seen_prompts[0][0] == work.literal
            retained = [
                block["toolResult"]
                for message in work.agents[0].messages
                for block in message["content"]
                if block.get("toolResult", {}).get("toolUseId") == TARGET_TOOL_USE
            ]
            assert len(retained) == 1 and tool_payload(retained[0]) == result
            event = json.loads(work.model.messages_seen[-1][-1]["content"][0]["text"])
            expected = {
                "schema_version": 1,
                "event_type": "original_" + kind + "_completed",
                "data_only": True,
                "request_ref": reference,
                "tool_use_id": TARGET_TOOL_USE,
                "status": "verified_completed",
            }
            if kind == "case":
                expected["case_ref"] = reference
            assert event == expected
            assert backend._warm_pending_producers == {}
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert work.agents[0].callback_handler is work.callbacks[0]
            gets = [params for name, params in work.target.s3.calls if name == "get"]
            assert gets and all(params.get("VersionId") for params in gets)
        finally:
            owned_reader = reader if reader is not None else getattr(work, "reader", None)
            if owned_reader is not None:
                await close_original_worker(work, owned_reader)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["case", "preparation"])
def test_original_unknown_observation_holds_worker_and_current_policy_refuses_without_resume(
    request,
    monkeypatch,
    tmp_path,
    kind,
):
    work = stage_work(request, monkeypatch, kind)

    async def scenario():
        reader = None
        try:
            reader, backend, _result, reference, record, _handle = await start_original_target(
                work, monkeypatch, tmp_path
            )
            capture_original(backend, work, reference, record)
            work.target.read_boundary.clear()
            work.target.unknown = True
            await wait_until(
                lambda: work.target.read_boundary.is_set(), reader, "unknown target read boundary missing"
            )
            assert not reader.done() and work.owned_app.get_async_task_info()["active_count"] == 1
            assert work.model.calls == 4
            before = copy.deepcopy(work.agents[0].messages)
            backend.approval_policy = None
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert work.agents[0].messages == before and work.model.calls == 4
            assert len(work.target.native.creates) == len(work.native.requests) == 1
            assert work.target.store._records[reference] is record
        finally:
            owned_reader = reader if reader is not None else getattr(work, "reader", None)
            if owned_reader is not None:
                await close_original_worker(work, owned_reader)

    asyncio.run(scenario())
