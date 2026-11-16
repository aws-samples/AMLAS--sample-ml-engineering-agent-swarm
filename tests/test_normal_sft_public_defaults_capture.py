"""Public SFT defaults remain original causal DATA, never producer authority.

These pure capture-contract controls contain no native port, SDK invocation,
training, snapshot replay or historical rewrite.
"""

from __future__ import annotations

import copy

import pytest

from amlas.control_plane.normal_run_checkpoint import _causal_call
from amlas.tools import qwen_sft

MODEL_REF = "model_acq_" + "a" * 32
PREPARATION_REF = "prep_" + "b" * 32
USE_ID = "ordinary-original-sft-use"


def original_prefix(optional):
    inputs = {
        "model_request_ref": MODEL_REF,
        "preparation_request_ref": PREPARATION_REF,
        **optional,
    }
    return [
        {
            "role": "user",
            "content": [{"text": "Continue the original prepared request."}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": USE_ID,
                        "name": "submit_qwen_sft",
                        "input": inputs,
                    }
                }
            ],
        },
    ]


def test_actual_public_tool_schema_and_effective_python_defaults(monkeypatch):
    schema = qwen_sft.submit_qwen_sft.tool_spec["inputSchema"]["json"]
    assert set(schema["required"]) == {"model_request_ref", "preparation_request_ref"}
    assert schema["properties"]["recipe_id"]["type"] == "string"
    assert schema["properties"]["plan_only"]["type"] == "boolean"
    calls = []

    def record_flow(**kwargs):
        calls.append(kwargs)
        return {"recorded_public_defaults": True}

    monkeypatch.setattr(qwen_sft, "run_qwen_sft_flow", record_flow)
    monkeypatch.setattr(qwen_sft, "_context", lambda _: {})
    assert qwen_sft.submit_qwen_sft(MODEL_REF, PREPARATION_REF) == {"recorded_public_defaults": True}
    assert calls == [
        {
            "model_request_ref": MODEL_REF,
            "preparation_request_ref": PREPARATION_REF,
            "recipe_id": "fit_v1",
            "plan_only": False,
        }
    ]


@pytest.mark.parametrize(
    "optional",
    [
        {},
        {"plan_only": False},
        {"recipe_id": "fit_v1", "plan_only": False},
        {"recipe_id": "full_v1", "plan_only": False},
    ],
    ids=[
        "omitted-recipe",
        "omitted-recipe-explicit-false",
        "fit-explicit-false",
        "full-explicit-false",
    ],
)
def test_public_effectful_defaults_capture_without_rewriting_raw_sdk_history(optional):
    prefix = original_prefix(optional)
    before = copy.deepcopy(prefix)
    try:
        captured = _causal_call(USE_ID, prefix)
    except ValueError as error:
        pytest.fail("public_optional_selection_refused_by_causal_capture: " + str(error))
    assert prefix == before
    assert captured["causal_prefix"] == before
    assert captured["tool_use"] == before[-1]["content"][0]["toolUse"]
    assert captured["tool_use"]["input"] == before[-1]["content"][0]["toolUse"]["input"]
    if "recipe_id" not in optional:
        assert "recipe_id" not in captured["causal_prefix"][-1]["content"][0]["toolUse"]["input"]
    assert prefix[0]["content"][0]["text"] == before[0]["content"][0]["text"]


@pytest.mark.parametrize("recipe", ["fit_v1", "full_v1"])
def test_existing_explicit_recipe_capture_preserves_original_history(recipe):
    prefix = original_prefix({"recipe_id": recipe})
    before = copy.deepcopy(prefix)
    captured = _causal_call(USE_ID, prefix)
    assert prefix == before and captured["causal_prefix"] == before
    assert captured["tool_use"]["input"]["recipe_id"] == recipe


@pytest.mark.parametrize(
    "optional",
    [
        {"recipe_id": "full_v1", "plan_only": True},
        {"recipe_id": "full_v1", "plan_only": None},
        {"recipe_id": "full_v1", "plan_only": 0},
        {"recipe_id": "full_v1", "plan_only": "false"},
        {"recipe_id": "full_v1", "ordinary_extra": "unselected"},
        {"recipe_id": "unknown_recipe"},
    ],
    ids=[
        "preview-true",
        "flag-none",
        "flag-int",
        "flag-string",
        "unknown-field",
        "unknown-recipe",
    ],
)
def test_preview_unknown_and_nonboolean_selection_cannot_gain_capture_authority(
    optional,
):
    prefix = original_prefix(optional)
    before = copy.deepcopy(prefix)
    with pytest.raises(ValueError):
        _causal_call(USE_ID, prefix)
    assert prefix == before


@pytest.mark.parametrize(
    "optional, expected_recipe",
    [
        ({}, "fit_v1"),
        ({"plan_only": False}, "fit_v1"),
        ({"recipe_id": "fit_v1", "plan_only": False}, "fit_v1"),
        ({"recipe_id": "full_v1", "plan_only": False}, "full_v1"),
    ],
    ids=["omitted-default-fit", "false-default-fit", "explicit-fit", "explicit-full"],
)
def test_effective_recipe_resolves_public_defaults_without_raw_input_mutation(optional, expected_recipe):
    from amlas.control_plane import normal_run_checkpoint

    inputs = {
        "model_request_ref": MODEL_REF,
        "preparation_request_ref": PREPARATION_REF,
        **optional,
    }
    before = copy.deepcopy(inputs)
    resolve = getattr(normal_run_checkpoint, "_effective_sft_inputs", None)
    assert callable(resolve), "effective_sft_inputs_missing"
    selected = resolve(inputs)
    assert selected == {
        "model_request_ref": MODEL_REF,
        "preparation_request_ref": PREPARATION_REF,
        "recipe_id": expected_recipe,
    }
    assert selected is not inputs
    assert inputs == before
    assert set(selected) == {
        "model_request_ref",
        "preparation_request_ref",
        "recipe_id",
    }


def test_sft_predicate_diagnostic_hook_is_private_and_optional():
    import inspect

    parameter = inspect.signature(_causal_call).parameters.get("_diagnostic_predicate")
    assert parameter is not None
    assert parameter.default is None


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("input_type", "INPUT_TYPE"),
        ("required_keys", "REQUIRED_KEYS"),
        ("extra_keys", "EXTRA_KEYS"),
        ("model_type", "MODEL_REF"),
        ("model_format", "MODEL_REF"),
        ("preparation_type", "PREPARATION_REF"),
        ("preparation_format", "PREPARATION_REF"),
        ("recipe", "RECIPE"),
        ("recipe_type", "RECIPE"),
        ("plan_type", "PLAN_TYPE"),
        ("plan_preview", "PLAN_PREVIEW"),
    ],
)
def test_sft_predicate_diagnostic_tracks_first_original_guard(case, expected):
    prefix = original_prefix({})
    use = next(block["toolUse"] for block in prefix[-1]["content"] if "toolUse" in block)
    inputs = use["input"]
    if case == "input_type":
        use["input"] = "ordinary input"
    elif case == "required_keys":
        del inputs["preparation_request_ref"]
        inputs["ordinary_extra"] = "ordinary"
    elif case == "extra_keys":
        inputs["ordinary_extra"] = "ordinary"
    elif case == "model_type":
        inputs["model_request_ref"] = 1
    elif case == "model_format":
        inputs["model_request_ref"] = "model_acq_" + "z" * 32
    elif case == "preparation_type":
        inputs["preparation_request_ref"] = 1
    elif case == "preparation_format":
        inputs["preparation_request_ref"] = "prep_" + "z" * 32
    elif case == "recipe":
        inputs["recipe_id"] = "ordinary_recipe"
    elif case == "recipe_type":
        inputs["recipe_id"] = 1
    elif case == "plan_type":
        inputs["plan_only"] = "false"
    else:
        inputs["plan_only"] = True
    original = copy.deepcopy(prefix)
    phases, predicates = [], []
    with pytest.raises(ValueError, match=r"^original SFT tool selection differs$"):
        _causal_call(
            USE_ID,
            prefix,
            _diagnostic_phase=phases.append,
            _diagnostic_predicate=predicates.append,
        )
    assert phases[-1] == "causal_sft_inputs"
    assert predicates == [expected]
    assert prefix == original


@pytest.mark.parametrize("optional", [{}, {"recipe_id": "full_v1", "plan_only": False}])
def test_sft_predicate_diagnostic_does_not_touch_valid_public_defaults(optional):
    prefix = original_prefix(optional)
    before = copy.deepcopy(prefix)
    predicates = []
    result = _causal_call(USE_ID, prefix, _diagnostic_predicate=predicates.append)
    assert predicates == []
    assert prefix == before
    assert result["causal_prefix"] == before


def test_sft_predicate_diagnostic_preserves_refusal_if_callback_fails():
    prefix = original_prefix({"plan_only": True})

    def fail(_code):
        raise RuntimeError("ordinary diagnostic failure")

    with pytest.raises(ValueError, match=r"^original SFT tool selection differs$"):
        _causal_call(USE_ID, prefix, _diagnostic_predicate=fail)


def test_sft_predicate_guard_short_circuits_without_rechecking(monkeypatch):
    from amlas.control_plane import normal_run_checkpoint as checkpoint

    calls = []
    original = checkpoint.re.fullmatch

    def observe(pattern, value, *args, **kwargs):
        if pattern in ("model_acq_[a-f0-9]{32}", "prep_[a-f0-9]{32}"):
            calls.append(pattern)
        return original(pattern, value, *args, **kwargs)

    monkeypatch.setattr(checkpoint.re, "fullmatch", observe)
    values = {"model_request_ref": "bad", "preparation_request_ref": PREPARATION_REF}
    with pytest.raises(ValueError, match=r"^original SFT tool selection differs$"):
        checkpoint._effective_sft_inputs(values, _diagnostic_predicate=lambda _code: None)
    assert calls == ["model_acq_[a-f0-9]{32}"]


def test_normal_sdk_preview_then_effectful_sft_keeps_commit_boundary_closed(monkeypatch, tmp_path):
    import asyncio
    import json

    import normal_sft_cold_fixture as cold

    from amlas.control_plane import normal_run_checkpoint as checkpoint

    async def scenario():
        state = None
        captures = []
        snapshots = []
        actual_capture = checkpoint.NormalRunContext.capture_sft_call
        actual_snapshot = cold.Agent.take_snapshot

        def capture(context, identifier, prefix, **kwargs):
            captures.append((copy.deepcopy(prefix), copy.deepcopy(kwargs["agent_snapshot_dict"])))
            return actual_capture(context, identifier, prefix, **kwargs)

        def snapshot(agent, *args, **kwargs):
            if agent.messages:
                for block in agent.messages[-1].get("content", []):
                    tool = block.get("toolUse", {})
                    if tool.get("name") == "submit_qwen_sft":
                        snapshots.append(tool["input"].get("plan_only", False))
            return actual_snapshot(agent, *args, **kwargs)

        class PreviewThenSubmit(cold.FullGraphModel):
            def __init__(self):
                super().__init__()
                self.preview_id = None
                self.preview_response = None
                self.preview_counts = None

            async def stream(self, messages, *args, **kwargs):
                observed = cold.payloads(messages)
                if self.preview_id in observed and self.preview_response is None:
                    self.preview_response = copy.deepcopy(observed[self.preview_id])
                    self.preview_counts = (
                        len(snapshots),
                        len(captures),
                        len(state.work.engines[0].tool_backend._sft_store._records),
                        len(state.native.calls),
                        state.task_published.is_set(),
                    )
                events = [copy.deepcopy(event) async for event in super().stream(messages, *args, **kwargs)]
                selected = None
                preview = False
                for event in events:
                    start = event.get("contentBlockStart", {})
                    tool = start.get("start", {}).get("toolUse", {})
                    if tool.get("name") == "submit_qwen_sft":
                        selected = start["contentBlockIndex"]
                        preview = self.preview_id is None
                        if preview:
                            self.preview_id = tool["toolUseId"] + "-preview"
                            tool["toolUseId"] = self.preview_id
                    delta = event.get("contentBlockDelta", {})
                    if selected is not None and delta.get("contentBlockIndex") == selected:
                        tool_delta = delta.get("delta", {}).get("toolUse")
                        if tool_delta is not None:
                            inputs = json.loads(tool_delta["input"])
                            inputs["plan_only"] = preview
                            tool_delta["input"] = json.dumps(inputs)
                for event in events:
                    yield event

        monkeypatch.setattr(cold, "FullGraphModel", PreviewThenSubmit)
        state = cold.setup(monkeypatch, tmp_path)
        monkeypatch.setattr(checkpoint.NormalRunContext, "capture_sft_call", capture)
        monkeypatch.setattr(cold.Agent, "take_snapshot", snapshot)
        try:
            await cold.original_precommit(state, monkeypatch)
            model = state.model
            assert model.preview_response is not None
            response = model.preview_response.get("data", model.preview_response)
            assert response["status"] == "planned"
            assert model.preview_counts == (0, 0, 0, 0, False)
            assert snapshots and all(mode is False for mode in snapshots)
            assert len(captures) == 1
            prefix, sdk_snapshot = captures[0]
            assert sdk_snapshot["data"]["messages"] == prefix
            selected = [
                block["toolUse"]
                for message in prefix
                for block in message.get("content", [])
                if block.get("toolUse", {}).get("name") == "submit_qwen_sft"
            ]
            assert len(selected) == 2
            assert selected[0]["toolUseId"] == model.preview_id
            assert selected[0]["input"]["plan_only"] is True
            assert selected[1]["input"]["plan_only"] is False
            assert len(state.work.engines[0].tool_backend._sft_store._records) == 1
            assert len(state.native.calls) == 1
        finally:
            await cold.close(state)

    asyncio.run(scenario())
