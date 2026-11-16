"""SFT batch admission and repeat submissions through the real normal path and a cold replacement.

Every source handle comes from a registered tool and the controlled native fixture, as in the cold continuation
tests. Component evidence only: no AgentCore worker loss, native IAM or real model is established.
"""

from __future__ import annotations

import asyncio
import copy

import pytest
from normal_sft_cold_fixture import (
    FINAL_TEXT,
    SFT_USE,
    close,
    cloud_data,
    finish_generation_pair,
    prepare_original_submission,
    setup,
    wait_for,
)
from strands.tools.decorator import DecoratedFunctionTool
from test_normal_sft_cold_continuation import child, marked_results
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401  (fixture used by pytestmark)

from amlas.control_plane.generation_state import SessionCodingGenerationStore
from amlas.control_plane.normal_run_checkpoint import NormalRunContext
from amlas.control_plane.sft_state import SessionSftStore
from amlas.tui.chat.tool_binding_backend import _record_messages

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
MIXED_IDS = {"mixed-sft", "mixed-other", "mixed-gen"}


def spies(monkeypatch):
    """Count what actually reaches a tool, the SFT parent capture and the two stores."""
    seen = {"dispatched": [], "captures": [], "sft_submits": 0, "gen_submits": 0}
    stream = DecoratedFunctionTool.stream
    capture = NormalRunContext.capture_sft_call
    sft_submit, gen_submit = SessionSftStore.submit, SessionCodingGenerationStore.submit

    async def dispatching(self, tool_use, invocation_state, **kwargs):
        seen["dispatched"].append((tool_use["toolUseId"], tool_use["name"]))
        async for event in stream(self, tool_use, invocation_state, **kwargs):
            yield event

    def capturing(self, pending_tool_use_id, *args, **kwargs):
        seen["captures"].append(pending_tool_use_id)
        return capture(self, pending_tool_use_id, *args, **kwargs)

    def sft_submitting(self, *args, **kwargs):
        seen["sft_submits"] += 1
        return sft_submit(self, *args, **kwargs)

    def gen_submitting(self, *args, **kwargs):
        seen["gen_submits"] += 1
        return gen_submit(self, *args, **kwargs)

    monkeypatch.setattr(DecoratedFunctionTool, "stream", dispatching)
    monkeypatch.setattr(NormalRunContext, "capture_sft_call", capturing)
    monkeypatch.setattr(SessionSftStore, "submit", sft_submitting)
    monkeypatch.setattr(SessionCodingGenerationStore, "submit", gen_submitting)
    return seen


def uses_of(message):
    return [block["toolUse"] for block in message["content"] if "toolUse" in block]


def results_of(message):
    return [block["toolResult"] for block in message["content"] if "toolResult" in block]


async def reach_accepted_original(state, monkeypatch):
    """Drive the real prompt to the accepted original SFT, with its result still uncommitted."""
    reader, received, backend = await prepare_original_submission(state, monkeypatch)
    await wait_for(
        state,
        reader,
        lambda: (
            backend._sft_store._records
            and any(value.kind == "sft" for value in backend._warm_pending_producers.values())
        ),
        "accepted PRECOMMIT original SFT",
    )
    await wait_for(state, reader, state.task_published.is_set, "successful admitted TASK publication")
    agent = state.work.agents[0]
    state.backend, state.agent = backend, agent
    state.original = next(iter(backend._sft_store._records.values()))
    state.run_id = next(event["run_id"] for event in received if event["event"] == "run_started")
    state.prefix = copy.deepcopy(agent.messages)
    return state


@pytest.mark.parametrize("order", [("sft", "other", "gen"), ("other", "gen", "sft")])
def test_mixed_sft_batch_is_refused_whole_and_the_alone_sft_survives_a_cold_replacement(
    monkeypatch, tmp_path, order
):
    async def scenario():
        seen = spies(monkeypatch)
        state = setup(monkeypatch, tmp_path, variant="mixed-" + "-".join(order))
        try:
            await reach_accepted_original(state, monkeypatch)
            history = state.agent.messages
            dispatched = [item for item in seen["dispatched"] if item[0] in MIXED_IDS]
            assert dispatched == [], "a tool of the mixed assistant message was dispatched: " + repr(
                dispatched
            )
            start = next(
                index
                for index, message in enumerate(history)
                if message["role"] == "assistant"
                and {use["toolUseId"] for use in uses_of(message)} == MIXED_IDS
            )
            assert [use["toolUseId"] for use in uses_of(history[start])] == [
                "mixed-" + name for name in order
            ]
            answers = results_of(history[start + 1])
            assert sorted(value["toolUseId"] for value in answers) == sorted(MIXED_IDS)
            assert all(value["status"] == "error" for value in answers)
            assert seen["captures"] == [SFT_USE], "only the SFT called alone may be captured"
            assert seen["sft_submits"] == 1 and seen["gen_submits"] == 0
            assert len(state.native.calls) == 1 and len(state.backend._generation_store._records) == 0
            assert [use["toolUseId"] for use in uses_of(history[-1])] == [SFT_USE]
            data = cloud_data(state, completed=True)
            answer = child(data, tmp_path / "replacement")
            assert answer["loaded"] and answer["invoked"], (
                "the alone SFT after a refused mixed message did not continue in the replacement"
            )
            restored = answer["invoked"][0]["messages"]
            assert _record_messages(restored)[: len(state.prefix)] == _record_messages(state.prefix), (
                "the refused message left the admitted history"
            )
            assert len(answer["new_training"]) == 2 and len(answer["new_processing"]) == 1
            assert all(value["TrainingJobName"] != state.original.id for value in answer["new_training"])
            terminal = next(
                value for value in reversed(answer["events"]) if value.get("event") == "run_completed"
            )
            assert terminal["worker_finished"] is True and FINAL_TEXT in terminal["response"]["message"]
        finally:
            await close(state)

    asyncio.run(scenario())


async def run_repeat(monkeypatch, tmp_path, variant, where):
    """Complete the run with one extra submit_qwen_sft call after the admitted SFT result, warm or cold."""
    state = setup(monkeypatch, tmp_path, variant=variant if where == "warm" else None)
    try:
        await reach_accepted_original(state, monkeypatch)
        alias = next(iter(state.backend._sft_store._records))
        data = cloud_data(state, completed=True)
        if where == "warm":
            state.native.states[state.original.id] = "Completed"
            events = await finish_generation_pair(state, state.reader)
            return {
                "alias": alias,
                "original": state.original.id,
                "marked": marked_results(state.model),
                "events": events,
                "new_training": state.native.calls[1:],
                "pair_records": len(state.backend._pair_store._records),
            }
        monkeypatch.setenv("AMLAS_COLD_MODEL_VARIANT", variant)
        answer = child(data, tmp_path / "replacement")
        return {
            "alias": alias,
            "original": state.original.id,
            "marked": answer["marked_results"],
            "events": answer["events"],
            "new_training": answer["new_training"],
            "pair_records": len(answer["new_processing"]),
        }
    finally:
        await close(state)


def assert_completed_once(seen):
    """The run finished: the two generation jobs and the pair exist once, and no second SFT was created."""
    terminal = next(value for value in reversed(seen["events"]) if value.get("event") == "run_completed")
    assert terminal["response"]["action"] != "input_admission_held", "the repeat call held the run"
    assert terminal["worker_finished"] is True and FINAL_TEXT in terminal["response"]["message"]
    assert len(seen["new_training"]) == 2 and seen["pair_records"] == 1
    assert all(value["TrainingJobName"] != seen["original"] for value in seen["new_training"])


@pytest.mark.parametrize("where", ["warm", "cold"])
def test_repeat_sft_with_the_same_inputs_returns_the_recorded_submission_and_the_run_completes(
    monkeypatch, tmp_path, where
):
    seen = asyncio.run(run_repeat(monkeypatch, tmp_path, "repeat-same", where))
    repeat = seen["marked"]["repeat-sft"]
    assert repeat["status"] == "success", "the repeat call did not come back as a normal tool result"
    payload = repeat["payload"]
    assert payload["request_ref"] == seen["alias"] and payload["reused"] is True
    assert payload["job_name"] == seen["original"] and payload.get("accepted") is not False
    assert payload["status"] != "refused"
    assert_completed_once(seen)


@pytest.mark.parametrize("where", ["warm", "cold"])
def test_repeat_sft_with_another_recipe_is_refused_by_name_and_the_run_completes(
    monkeypatch, tmp_path, where
):
    seen = asyncio.run(run_repeat(monkeypatch, tmp_path, "repeat-different", where))
    payload = seen["marked"]["repeat-sft"]["payload"]
    assert payload["status"] == "sft_already_admitted", "the different recipe was not refused by name"
    assert payload["request_ref"] == seen["alias"] and payload["accepted"] is False
    assert seen["alias"] in payload["message"]
    assert_completed_once(seen)
