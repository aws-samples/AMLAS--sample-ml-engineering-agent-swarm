"""The continuation fixture hands over only after the turn that submitted the target has settled.

The stream stand-in counts a call when its stream starts. The SDK appends that call's answer
later, so a history copied at the count can miss it. Source fixtures only; no native ML proof.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from case_prep_continuation_fixtures import stage_work, start_original_target, wait_until
from test_runtime_connected_warm_model import close_original_worker

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def test_original_target_fixture_returns_after_the_submitting_turn_is_in_history(
    request, monkeypatch, tmp_path
):
    work = stage_work(request, monkeypatch, "case")
    counted, release = threading.Event(), threading.Event()
    actual_stream = work.model.stream

    async def held_fourth_answer(messages, tool_specs=None, system_prompt=None, **kwargs):
        first = True
        async for event in actual_stream(messages, tool_specs, system_prompt, **kwargs):
            if first and work.model.calls == 4:
                # The fourth call is counted; its answer has not reached the SDK history.
                counted.set()
                assert release.wait(5), "held fourth answer was not released"
            first = False
            yield event

    monkeypatch.setattr(work.model, "stream", held_fourth_answer)

    async def scenario():
        starting = asyncio.create_task(start_original_target(work, monkeypatch, tmp_path))
        try:
            await wait_until(counted.is_set, starting, "fourth model call was not counted")
            assert counted.is_set() and work.model.calls == 4
            held_history = len(work.agents[0].messages)
            # Several fixture poll intervals: a fixture that trusts the count returns here.
            for _ in range(10):
                await asyncio.sleep(0.005)
            returned_while_held = starting.done()
            release.set()
            reader, _backend, _result, reference, _record, _handle = await asyncio.wait_for(starting, 5)
            assert not returned_while_held, "fixture returned before the fourth answer reached the history"
            messages = work.agents[0].messages
            assert len(messages) == held_history + 1 and messages[-1]["role"] == "assistant"
            assert messages[-1]["content"] == [{"text": "Original producers remain pending."}]
            assert any(observed == reference for observed, _ in work.target.actual_observed)
            assert work.model.calls == 4 and not reader.done()
        finally:
            release.set()
            if not starting.done():
                starting.cancel()
            await asyncio.gather(starting, return_exceptions=True)
            owned = getattr(work, "reader", None)
            if owned is not None:
                await close_original_worker(work, owned)

    asyncio.run(scenario())
