"""An assistant message that mixes an effectful submit_qwen_sft with any other tool use is refused whole.

The refusal runs before any tool of the message is dispatched and before the SFT parent capture: nothing
executes, every tool use gets one error result that says so, the assistant message and the results stay in the
history, and the model's next message may call submit_qwen_sft alone. Neutral stand-in tools; no AWS.
"""

from __future__ import annotations

import itertools
import json

import pytest
from strands import Agent, tool
from strands.hooks.events import BeforeToolCallEvent
from strands.models import Model

from amlas.tui.chat import tool_binding_backend as binding

SFT_ARGS = {"model_request_ref": "model_acq_" + "1" * 32, "preparation_request_ref": "prep_" + "2" * 32}
USES = {
    "sft": ("batch-sft", "submit_qwen_sft", {**SFT_ARGS, "recipe_id": "full_v1"}),
    "other": ("batch-other", "get_coding_case_freeze", {"case_ref": "case_" + "3" * 32}),
    "resolve": (
        "batch-resolve",
        "resolve_coding_evaluation_spec",
        {"case_request_ref": "case_" + "3" * 32, "spec_ref": "spec_" + "4" * 32},
    ),
    "gen": (
        "batch-gen",
        "submit_coding_generation",
        {
            "model_request_ref": "model_acq_" + "1" * 32,
            "case_request_ref": "case_" + "3" * 32,
            "spec_ref": "spec_" + "4" * 32,
        },
    ),
}


class ScriptedModel(Model):
    """Each step is a list of (id, name, arguments) tool uses or a final text."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = 0

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "neutral-scripted-batch"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("ordinary SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        if type(step) is str:
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": step}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            for index, (identifier, name, arguments) in enumerate(step):
                yield {
                    "contentBlockStart": {
                        "contentBlockIndex": index,
                        "start": {"toolUse": {"toolUseId": identifier, "name": name}},
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "contentBlockIndex": index,
                        "delta": {"toolUse": {"input": json.dumps(arguments)}},
                    }
                }
                yield {"contentBlockStop": {"contentBlockIndex": index}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }


def run(steps, *, attach=True):
    executed = []

    @tool
    def submit_qwen_sft(
        model_request_ref: str = "",
        preparation_request_ref: str = "",
        recipe_id: str = "fit_v1",
        plan_only: bool = False,
    ) -> dict:
        """Stand-in SFT submission."""
        executed.append(("submit_qwen_sft", plan_only))
        return {"success": True, "request_ref": "sft_" + "5" * 32}

    @tool
    def get_coding_case_freeze(case_ref: str) -> dict:
        """Stand-in read."""
        executed.append(("get_coding_case_freeze", False))
        return {"success": True}

    @tool
    def resolve_coding_evaluation_spec(case_request_ref: str, spec_ref: str) -> dict:
        """Stand-in spec resolution."""
        executed.append(("resolve_coding_evaluation_spec", False))
        return {"success": True, "spec_ref": "spec_" + "7" * 32}

    @tool
    def submit_coding_generation(
        model_request_ref: str, case_request_ref: str, spec_ref: str, sft_request_ref: str = ""
    ) -> dict:
        """Stand-in generation submission."""
        executed.append(("submit_coding_generation", False))
        return {"success": True, "request_ref": "gen_" + "6" * 32}

    backend = object.__new__(binding.ToolBindingBackend)
    backend._refused_tool_results = {}
    agent = Agent(
        model=ScriptedModel(steps),
        tools=[
            submit_qwen_sft,
            get_coding_case_freeze,
            resolve_coding_evaluation_spec,
            submit_coding_generation,
        ],
        callback_handler=None,
    )
    if attach:
        agent.hooks.add_callback(BeforeToolCallEvent, backend._refuse_mixed_sft_batch)
    agent("go")
    return agent, executed, backend


def results_of(message):
    return [block["toolResult"] for block in message["content"] if "toolResult" in block]


def uses_of(message):
    return [block["toolUse"] for block in message["content"] if "toolUse" in block]


REAL_SHAPES = [
    shape
    for parts in (["sft", "resolve"], ["sft", "gen"], ["sft", "resolve", "gen"], ["sft", "other", "gen"])
    for shape in itertools.permutations(parts)
]


@pytest.mark.parametrize("order", REAL_SHAPES, ids=lambda order: "-".join(order))
def test_mixed_batch_is_refused_whole_for_every_position_of_the_sft(order):
    batch = [USES[name] for name in order]
    agent, executed, backend = run([batch, "done"])
    assert executed == [], "a tool of the mixed batch was dispatched"
    assert [use["toolUseId"] for use in uses_of(agent.messages[1])] == [item[0] for item in batch]
    answers = results_of(agent.messages[2])
    assert sorted(value["toolUseId"] for value in answers) == sorted(item[0] for item in batch)
    assert all(value["status"] == "error" for value in answers)
    by_id = {value["toolUseId"]: json.loads(value["content"][0]["text"]) for value in answers}
    assert all(value["status"] == "batch_refused" and value["executed"] is False for value in by_id.values())
    assert "No tool in this message was executed" in by_id["batch-sft"]["message"]
    assert "alone" in by_id["batch-sft"]["message"]
    for sibling in (item[0] for item in batch if item[0] != "batch-sft"):
        assert "Not executed" in by_id[sibling]["message"] and "submit_qwen_sft" in by_id[sibling]["message"]
    assert agent.messages[3]["content"][0]["text"] == "done"
    assert set(backend._refused_tool_results) == {item[0] for item in batch}


def test_two_effectful_sft_calls_in_one_message_are_both_refused():
    second = ("batch-sft-2", "submit_qwen_sft", {**SFT_ARGS, "recipe_id": "fit_v1"})
    agent, executed, _backend = run([[USES["sft"], second], "done"])
    assert executed == []
    answers = results_of(agent.messages[2])
    assert {value["toolUseId"] for value in answers} == {"batch-sft", "batch-sft-2"}
    assert all(value["status"] == "error" for value in answers)


def test_the_refusal_keeps_the_next_message_free_to_call_the_sft_alone():
    agent, executed, _backend = run([[USES["sft"], USES["other"], USES["gen"]], [USES["sft"]], "done"])
    assert executed == [("submit_qwen_sft", False)]
    assert all(value["status"] == "error" for value in results_of(agent.messages[2]))
    final = results_of(agent.messages[4])
    assert [value["toolUseId"] for value in final] == ["batch-sft"] and final[0]["status"] == "success"


def test_an_sft_alone_and_a_batch_without_an_sft_are_unchanged():
    _agent, executed, backend = run([[USES["sft"]], "done"])
    assert executed == [("submit_qwen_sft", False)] and backend._refused_tool_results == {}
    _agent, executed, backend = run([[USES["other"], USES["gen"]], "done"])
    assert sorted(executed) == [("get_coding_case_freeze", False), ("submit_coding_generation", False)]
    assert backend._refused_tool_results == {}


@pytest.mark.parametrize("sibling", ["other", "resolve", "gen"])
def test_a_plan_only_preview_has_no_effect_and_may_share_a_message(sibling):
    preview = ("batch-preview", "submit_qwen_sft", {**SFT_ARGS, "plan_only": True})
    name = USES[sibling][1]
    _agent, executed, backend = run([[preview, USES[sibling]], "done"])
    assert sorted(executed) == sorted([(name, False), ("submit_qwen_sft", True)])
    assert backend._refused_tool_results == {}


def test_an_effectful_sft_with_a_preview_in_one_message_is_refused():
    preview = ("batch-preview", "submit_qwen_sft", {**SFT_ARGS, "plan_only": True})
    _agent, executed, backend = run([[USES["sft"], preview], "done"])
    assert executed == [] and set(backend._refused_tool_results) == {"batch-sft", "batch-preview"}


def test_without_the_admission_hook_the_same_message_dispatches_every_tool():
    _agent, executed, _backend = run([[USES["sft"], USES["other"], USES["gen"]], "done"], attach=False)
    assert len(executed) == 3, "the control run must show the unguarded dispatch"


MODEL = "model_acq_" + "1" * 32
PREPARATION = "prep_" + "2" * 32
ADMITTED = "sft_" + "5" * 32


def durable_store(*, durable=True, recipe="full_v1"):
    """A store that already holds its one original training record, built without any AWS client."""
    import threading
    from types import SimpleNamespace

    from amlas.backends.types import JobHandle
    from amlas.control_plane.sft_state import SessionSftStore

    store = object.__new__(SessionSftStore)
    store._campaign_state = object() if durable else None
    store._catalog = SimpleNamespace(name="current")
    store._lock = threading.RLock()
    handle = JobHandle(
        "job-a",
        "qwen_sft",
        "arn:aws:sagemaker:us-west-2:111122223333:training-job/job-a",
        {
            "operation": "train",
            "catalog": store._catalog,
            "recipe_id": recipe,
            "submission_state": "accepted",
        },
    )
    store._records = {ADMITTED: handle, "sft_" + "6" * 32: handle}
    store._keys = {
        (MODEL, PREPARATION, recipe, "train"): ADMITTED,
        (ADMITTED, "reload"): "sft_" + "6" * 32,
        ("historical_train", "arn", MODEL, PREPARATION, recipe, "train"): ADMITTED,
    }
    return store


def test_repeat_kind_compares_the_logical_inputs_with_the_one_admitted_training_record():
    store = durable_store()
    assert store.repeat_kind(MODEL, PREPARATION, "full_v1") == "same"
    assert store.repeat_kind(MODEL, PREPARATION, "fit_v1") == "different"
    assert store.repeat_kind("model_acq_" + "9" * 32, PREPARATION, "full_v1") == "different"
    assert store.repeat_kind(MODEL, "prep_" + "9" * 32, "full_v1") == "different"


def test_a_store_without_the_durable_journal_never_classifies_a_call_as_a_repeat():
    store = durable_store(durable=False)
    assert store.repeat_kind(MODEL, PREPARATION, "full_v1") is None
    assert store.repeat_kind(MODEL, PREPARATION, "fit_v1") is None


def test_reload_and_historical_records_do_not_count_as_the_admitted_training_submission():
    store = durable_store()
    store._keys = {key: ref for key, ref in store._keys.items() if len(key) != 4}
    assert store.repeat_kind(MODEL, PREPARATION, "full_v1") is None


def test_other_inputs_are_refused_by_name_with_the_admitted_reference():
    result = durable_store()._already_admitted(ADMITTED)
    assert result["status"] == "sft_already_admitted" and result["request_ref"] == ADMITTED
    assert result["accepted"] is False and result["success"] is False and result["trained"] is False
    assert (
        result["recipe_id"] == "full_v1" and result["job_name"] == "job-a" and ADMITTED in result["message"]
    )


def test_the_same_inputs_return_the_recorded_state_and_the_new_call_never_submits(monkeypatch):
    store = durable_store()
    monkeypatch.setattr(
        store,
        "observe",
        lambda ref: {"success": True, "trained": True, "status": "SUCCEEDED", "request_ref": ref},
    )
    result = store._recorded_submission(ADMITTED)
    assert result["reused"] is True and result["request_ref"] == ADMITTED
    assert result["trained"] is True and result["status"] == "SUCCEEDED" and result["job_name"] == "job-a"
