"""A repeated generation request after a worker loss meets the job that was accepted before the loss.

Component evidence only (a child process ends with os._exit after a generation backend accepted a job; the next child
receives the native transport DATA the first one left behind). Nothing here shows AgentCore stop behaviour.
"""

from __future__ import annotations

import asyncio

import pytest
from normal_sft_cold_fixture import FINAL_TEXT, close, cloud_data, original_precommit, payloads, setup
from test_normal_continuation_loss_windows import completed
from test_normal_sft_cold_continuation import child
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401  (fixture used by pytestmark)

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def test_replacement_repeating_generation_after_a_loss_meets_the_accepted_job(monkeypatch, tmp_path):
    """The repeated request derives the same reference and meets the job that was accepted before the loss.

    Taking over a continuation that is already 'invoking' is the open defect above and is not built. This test
    puts the run row back to the claimed state, so that it isolates what a repeated generation request does once a
    replacement runs the continuation again: no second native job and the reference handed out before the loss.
    """

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            data = cloud_data(state, completed=True)
            claimed = child(data, tmp_path / "claimed", loss="after-claim")
            lost = child(data, tmp_path / "lost-after-gen", loss="after-gen")
            before = [value["TrainingJobName"] for value in lost["transport"]["training_requests"]]
            assert len(before) >= 2 and len(before) == len(set(before)), "no generation job before the cut"
            transport = {**lost["transport"], "ddb": claimed["transport"]["ddb"]}
            resumed = child(transport, tmp_path / "replay")
            names = [value["TrainingJobName"] for value in resumed["transport"]["training_requests"]]
            assert len(names) == len(set(names)) == 3, (
                "a repeated request created a second native job: " + repr(names)
            )
            assert set(before) <= set(names)
            refs = {value["payload"]["request_ref"] for value in resumed["marked_results"].values()}
            assert lost["accepted_ref"] in refs, "the accepted job did not keep its reference"
            assert len(resumed["new_processing"]) == 1
            terminal = completed(resumed)
            assert terminal is not None and FINAL_TEXT in terminal["response"]["message"]
        finally:
            await close(state)

    asyncio.run(scenario())


def test_replacement_accepts_the_specification_references_it_was_handed_before_the_stop(
    monkeypatch, tmp_path
):
    """The specification is bound and resolved before the SFT; the replacement meets those references in its history."""

    async def scenario():
        state = setup(monkeypatch, tmp_path, variant="spec-first")
        try:
            await original_precommit(state, monkeypatch)
            data = cloud_data(state, completed=True)
            old = payloads(state.prefix)["cold-original-resolved-spec"]["spec_ref"]
            answer = child(data, tmp_path / "replacement")
            marked = answer["marked_results"]
            for arm in ("base", "tuned"):
                result = marked["cold-original-generation-" + arm]
                assert result["status"] == "success"
                assert result["payload"].get("event_type") == "original_generation_completed", (
                    "the old specification reference was refused after the replacement: "
                    + repr(result["payload"])
                )
            assert len(answer["new_training"]) == 2 and len(answer["new_processing"]) == 1
            assert old.startswith("spec_")
            terminal = next(
                value for value in reversed(answer["events"]) if value.get("event") == "run_completed"
            )
            assert terminal["response"]["action"] != "input_admission_held"
            assert FINAL_TEXT in terminal["response"]["message"]
        finally:
            await close(state)

    asyncio.run(scenario())
