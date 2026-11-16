"""Process loss between the completion claim and the first effect of the continuation, and inside it.

Component evidence only (a child process ends with os._exit at a named point; the next child receives the native
transport DATA the first one left behind). Nothing here shows AgentCore stop behaviour or a deployed wake.
"""

from __future__ import annotations

import asyncio

import pytest
from normal_sft_cold_fixture import FINAL_TEXT, close, cloud_data, original_precommit, setup
from test_normal_sft_cold_continuation import child
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401  (fixture used by pytestmark)

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def require(condition, message):
    """A precondition of an expected-failure test; it must not read as the expected failure."""
    if not condition:
        raise RuntimeError(message)


def completed(answer):
    return next(
        (value for value in reversed(answer["events"]) if value.get("event") == "run_completed"), None
    )


def test_replacement_recovers_after_loss_between_the_claim_and_the_continuation(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            data = cloud_data(state, completed=True)
            lost = child(data, tmp_path / "lost-after-claim", loss="after-claim")
            assert lost["cut"] == "after-claim" and lost["loaded"] == 1 and lost["invoked"] == 0
            assert lost["training_at_loss"] == len(data["training_requests"]), "an effect ran before the cut"
            row = lost["transport"]["ddb"]["PROMPT_RUN#" + state.run_id]["record"]["S"]
            assert '"status":"consumed-ready"' in row and '"invocation_state":"ready"' in row
            resumed = child(lost["transport"], tmp_path / "after-claim")
            assert resumed["loaded"] and resumed["invoked"], (
                "the replacement did not continue the claimed run"
            )
            assert len(resumed["new_training"]) == 2 and len(resumed["new_processing"]) == 1
            assert all(value["TrainingJobName"] != state.original.id for value in resumed["new_training"])
            terminal = completed(resumed)
            assert terminal is not None and FINAL_TEXT in terminal["response"]["message"]
        finally:
            await close(state)

    asyncio.run(scenario())


def test_loss_inside_the_continuation_is_not_recovered_and_repeats_no_effect(monkeypatch, tmp_path):
    """Characterises an OPEN limitation (a worker lost after begin_completion_invocation persisted 'invoking').

    No replacement may claim the run (the original continuation already has an owner), so nothing continues and the wake
    is answered as not delivered; but no backend submission is repeated. When the takeover is built this test becomes the
    recovery test (the replacement continues, exactly one backend submission per generation arm, the accepted job keeps its
    reference, the run completes).
    """

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            data = cloud_data(state, completed=True)
            lost = child(data, tmp_path / "lost-after-gen", loss="after-gen")
            accepted = len(lost["transport"]["training_requests"]) - len(data["training_requests"])
            require(accepted >= 1, "no generation job was accepted before the cut")
            row = lost["transport"]["ddb"]["PROMPT_RUN#" + state.run_id]["record"]["S"]
            require('"invocation_state":"invoking"' in row, "the continuation was not invoking at the cut")
            resumed = child(lost["transport"], tmp_path / "after-gen")
            names = [value["TrainingJobName"] for value in resumed["transport"]["training_requests"]]
            assert not (resumed["loaded"] and resumed["invoked"]), (
                "the limitation is closed: update this test to the recovery expectations"
            )
            assert len(names) == len(set(names)), "a backend submission was repeated"
            assert completed(resumed) is None
        finally:
            await close(state)

    asyncio.run(scenario())
