"""The real readiness-stream record is acknowledged only by the positive closed event.

Runs the actual TASK-publication stream record through the real stream handler over the
in-memory ports of the cold-restore fixture. Anything but `continuing` or `completed` makes
the handler raise, which is what makes the event source mapping retry the same record.
"""

from __future__ import annotations

import asyncio

import pytest
from test_campaign_callback import Body
from test_normal_wake_stream_transition import _fresh_actual_task_stream
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.async_callback import lambda_handler as callback_source


def _stream_deliver(case):
    with callback_source.campaign_callback_scope(
        case.qualifier, dynamodb_client=case.ddb, runtime_client=case.runtime, clock=lambda: 1000.0
    ):
        return callback_source.handler({"Records": [case.record]})


@pytest.mark.parametrize("state", ["continuing", "completed"])
def test_positive_state_acknowledges_the_stream_record(monkeypatch, tmp_path, state):
    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            calls = len(case.runtime.calls)
            case.runtime.resume_state = state
            result = _stream_deliver(case)
            assert result["statusCode"] == 200 and result["delivery_status"] == "delivered"
            assert len(case.runtime.calls) == calls + 1 and case.runtime.body.closed_called

    asyncio.run(scenario())


@pytest.mark.parametrize("state", ["restoring", "original_active", "failed_transient", "failed"])
def test_every_other_state_makes_the_stream_handler_raise_so_the_record_is_retried(
    monkeypatch, tmp_path, state
):
    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            case.runtime.resume_state = state
            with pytest.raises(RuntimeError, match="normal readiness delivery unresolved"):
                _stream_deliver(case)
            assert case.runtime.body.closed_called

    asyncio.run(scenario())


def test_old_neutral_body_no_longer_acknowledges_the_stream_record(monkeypatch, tmp_path):
    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            case.runtime.resume_state = None
            case.runtime.body = Body()
            with pytest.raises(RuntimeError, match="normal readiness delivery unresolved"):
                _stream_deliver(case)

    asyncio.run(scenario())


def test_retry_after_restoring_reaches_continuing_and_only_then_acknowledges(monkeypatch, tmp_path):
    """The cold replacement: first delivery sees restoring, a later one sees continuing."""

    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            outcomes = []
            for state in ("restoring", "restoring", "continuing"):
                case.runtime.resume_state = state
                try:
                    outcomes.append(_stream_deliver(case)["delivery_status"])
                except RuntimeError:
                    outcomes.append("retry")
            assert outcomes == ["retry", "retry", "delivered"]

    asyncio.run(scenario())


def test_task_publication_delivery_remote_call_budget(monkeypatch, tmp_path):
    """Pins the sequential single-attempt remote calls of one delivery (Lambda timeout 120 s).

    Each qualification is 1 STS + 3 control-plane reads; a native() bracket is 3 qualifications
    + 1 DescribeTable; read() is 2 brackets + 1 GetItem; current() is 3 pinned rows + 1 bracket.
    It runs before the invoke, again right before it, after the invoke (drain) and after the drain.
    """

    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            counts = {"get_item": 0, "describe_table": 0}
            actual_get, actual_describe = case.ddb.get_item, case.ddb.describe_table

            def get_item(**kwargs):
                counts["get_item"] += 1
                return actual_get(**kwargs)

            def describe_table(**kwargs):
                counts["describe_table"] += 1
                return actual_describe(**kwargs)

            case.ddb.get_item, case.ddb.describe_table = get_item, describe_table
            qualifications = len(case.qualifier.calls)
            invokes = len(case.runtime.calls)
            _stream_deliver(case)
            qualifications = len(case.qualifier.calls) - qualifications
            assert len(case.runtime.calls) == invokes + 1
            total = qualifications * 4 + counts["describe_table"] + counts["get_item"] + 1
            assert (qualifications, counts["describe_table"], counts["get_item"]) == (106, 35, 15), (
                qualifications,
                counts,
            )
            assert total == 475

    asyncio.run(scenario())
