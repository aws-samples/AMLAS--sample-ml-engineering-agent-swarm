"""Finite original MODEL UNKNOWN polls; actual API/SDK, inert native fixtures.

No live runtime21 state, real model/provider/Create, retries or quota changes.
PENDING/RUNNING work stays uncapped; unresolved outcomes retain original identity.
"""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from test_runtime_ambiguous_model_capture import (
    attempted_original,
    collect_events,
    lose_original_create_reply,
)
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_warm_model_continuation import complete_fixture

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def missing_original(monkeypatch, work, *, throttled=False):
    original = work.native.create_processing_job
    creates = []

    def create(**kwargs):
        response = original(**kwargs)
        creates.append((kwargs["ProcessingJobName"], response["ProcessingJobArn"]))
        if throttled:
            raise ClientError(
                {
                    "Error": {"Code": "ThrottlingException"},
                    "ResponseMetadata": {
                        "HTTPStatusCode": 400,
                        "RequestId": "11111111-1111-4111-8111-aa11bb22cc33",
                        "RetryAttempts": 0,
                    },
                },
                "CreateProcessingJob",
            )
        raise TimeoutError("ordinary lost original Create response")

    monkeypatch.setattr(work.native, "create_processing_job", create)
    describe = work.native.describe_processing_job

    def missing(**kwargs):
        describe(**kwargs)
        raise ClientError({"Error": {"Code": "ResourceNotFound"}}, "DescribeProcessingJob")

    monkeypatch.setattr(work.native, "describe_processing_job", missing)
    return creates


@pytest.mark.parametrize("throttled", [False, True])
def test_normal_api_missing_original_returns_unresolved_at_exact_unknown_budget(monkeypatch, throttled):
    work = connected_work(monkeypatch)
    creates = missing_original(monkeypatch, work, throttled=throttled)
    completions = []
    complete_task = work.owned_app.complete_async_task

    def complete_once(task_id):
        completions.append(task_id)
        return complete_task(task_id)

    monkeypatch.setattr(work.owned_app, "complete_async_task", complete_once)

    async def scenario():
        reader = asyncio.create_task(collect_events(work))
        try:
            owned = await attempted_original(work)
            before = copy.deepcopy(work.agents[0].messages)
            pending = owned.backend._warm_pending
            deadline = asyncio.get_running_loop().time() + 5
            while len(work.observations) < 8:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.03)
            assert work.api_returned.is_set(), (
                "UNKNOWN original did not terminate after eight qualified polls"
            )
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["worker_finished"] is True
            assert events[-1]["response"]["data"]["warm_model_continuation"] == {
                "disposition": "submission_unresolved",
                "diagnostic": {
                    "phase": "native_observation",
                    "reason": "original_unknown_budget_exhausted",
                    "error_class": "UNKNOWN",
                },
            }
            assert events[-1]["quality_success"] is False and events[-1]["lifecycle_verified"] is False
            assert len(work.observations) == 8 and len(creates) == len(work.native.requests) == 1
            assert work.model.calls == 2 and len(work.engines) == len(work.agents) == 1
            assert (
                owned.store._records[owned.reference] is owned.record and owned.record.handle is owned.handle
            )
            assert owned.backend._warm_pending is pending and pending.reference == owned.reference
            assert (
                owned.backend._warm_closed == "submission_unresolved"
                and owned.backend._warm_consumed == set()
            )
            assert (
                owned.handle.extra["accepted"] is False and owned.handle.extra["submission_ambiguous"] is True
            )
            assert work.agents[0].messages == before and len(completions) == 1
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            reads = len(work.observations)
            await work.engines[0]._observe_pending_model_once()
            await work.engines[0].process_message(work.literal)
            assert len(work.observations) == reads and len(creates) == 1 and work.model.calls == 2
            assert work.agents[0].messages == before
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_running_original_exceeds_unknown_budget_without_overall_wait_cap(monkeypatch):
    work = connected_work(monkeypatch)
    creates = lose_original_create_reply(monkeypatch, work)

    async def scenario():
        reader = asyncio.create_task(collect_events(work))
        try:
            owned = await attempted_original(work)
            deadline = asyncio.get_running_loop().time() + 5
            while len(work.observations) < 12:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert not work.api_returned.is_set() and not reader.done()
            assert work.owned_app.get_async_task_info()["active_count"] == 1 and work.model.calls == 2
            assert len(creates) == 1 and owned.backend._warm_pending.record is owned.record
            owned.backend.approval_policy = None
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_running_observation_resets_unknown_budget_before_same_agent_completion(monkeypatch, tmp_path):
    work = connected_work(monkeypatch)
    creates = lose_original_create_reply(monkeypatch, work)
    describe = work.native.describe_processing_job
    reads = []

    def sequence(**kwargs):
        reads.append(kwargs["ProcessingJobName"])
        count = len(reads)
        if count == 18:
            backend = work.engines[0].tool_backend
            pending = backend._warm_pending
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=pending.record.handle, adapter=work.adapter, native=work.native),
            )
        response = describe(**kwargs)
        if count <= 7 or 11 <= count <= 17:
            raise ClientError({"Error": {"Code": "ResourceNotFound"}}, "DescribeProcessingJob")
        return response

    monkeypatch.setattr(work.native, "describe_processing_job", sequence)

    async def scenario():
        reader = asyncio.create_task(collect_events(work))
        try:
            owned = await attempted_original(work)
            before = copy.deepcopy(work.agents[0].messages)
            events = await asyncio.wait_for(reader, 5)
            assert (
                events[-1]["response"]["data"]["warm_model_continuation"]["disposition"]
                == "continued_original"
            )
            assert len(reads) == 18 and len(creates) == 1 and work.model.calls == 3
            assert work.agents[0].messages[: len(before)] == before
            assert owned.record.handle is owned.handle and owned.handle.extra["accepted"] is False
            assert owned.handle.extra["submission_ambiguous"] is True
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_policy_revocation_at_eighth_unknown_poll_wins_over_unresolved_budget(monkeypatch):
    work = connected_work(monkeypatch)
    creates = lose_original_create_reply(monkeypatch, work)
    describe = work.native.describe_processing_job
    reads = []

    def revoked(**kwargs):
        reads.append(kwargs["ProcessingJobName"])
        describe(**kwargs)
        if len(reads) == 8:
            work.engines[0].tool_backend.approval_policy = None
        raise ClientError({"Error": {"Code": "ResourceNotFound"}}, "DescribeProcessingJob")

    monkeypatch.setattr(work.native, "describe_processing_job", revoked)

    async def scenario():
        reader = asyncio.create_task(collect_events(work))
        try:
            owned = await attempted_original(work)
            before = copy.deepcopy(work.agents[0].messages)
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert len(reads) == 8 and len(creates) == 1 and work.model.calls == 2
            assert work.agents[0].messages == before and owned.backend._warm_consumed == set()
            assert owned.store._records[owned.reference] is owned.record
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
