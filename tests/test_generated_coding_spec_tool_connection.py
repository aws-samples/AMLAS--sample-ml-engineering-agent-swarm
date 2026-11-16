"""Real registered SDK planning tools must connect without fixture-written YAML."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from strands import Agent
from strands.hooks.events import AfterToolCallEvent
from strands.models import Model
from test_managed_generation_product import state as _generation_fixture

from amlas.control_plane.approvals import ApprovalPolicy
from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest
from amlas.spec.loader import load_spec_from_text
from amlas.tui.chat.agent_tools import get_safe_llm_tools, tool_display_name

generation_state = _generation_fixture

DESCRIPTION = (
    "Fine-tune Qwen/Qwen3.5-9B as a coding model for Python code generation. "
    "The target is 'python_module'. Evaluate held-out pass@1 with the untouched "
    "base and tuned model on the same cases. Acceptance requires tuned pass@1 "
    "strictly greater than the untouched-base pass@1, not an invented absolute score. "
    "The owner requests unbounded trials, wall-clock minutes and retreats, "
    "with no declared GPU-hour ceiling."
)


def payload(event: AfterToolCallEvent) -> dict:
    for block in event.result["content"]:
        if type(block.get("json")) is dict:
            return block["json"]
        if type(block.get("text")) is str:
            value = json.loads(block["text"])
            if type(value) is dict:
                return value
    raise AssertionError("actual SDK planning ToolResult object required")


class PlanningConnectionModel(Model):
    """Script tool choices only; every specification byte comes from the real tool."""

    def __init__(
        self,
        results: dict,
        save_path: str,
        planning_data=None,
        data_path="./cases.json",
        description=DESCRIPTION,
    ):
        self.results = results
        self.save_path = save_path
        self.planning_data = planning_data
        self.data_path = data_path
        self.description = description
        self.cycles = 0

    def update_config(self, **model_config):
        pass

    def get_config(self):
        return {"model_id": "offline-generated-planning-connection"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("ordinary registered SDK tool loop required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        cycle = self.cycles
        self.cycles += 1
        request = None
        if cycle == 0:
            inputs = {"description": self.description, "data_path": self.data_path}
            identifier = "generated"
            if self.planning_data is not None:
                inputs = {"description": "Fine-tune Qwen as a coding assistant; evaluate held-out pass@1."}
                identifier = "research"
            request = (identifier, "generate_spec_from_description", inputs)
        elif cycle == 1 and self.planning_data is not None:
            request = (
                "generated",
                "generate_spec_from_description",
                {
                    "description": "Fine-tune Qwen as a coding assistant; evaluate held-out pass@1.",
                    "planning_data": self.planning_data,
                },
            )
        else:
            generated = self.results.get("generated", {})
            text = generated.get("spec_text")
            consumer_cycle = cycle - (1 if self.planning_data is not None else 0)
            if type(text) is str:
                if consumer_cycle == 1:
                    request = ("validated", "validate_spec", {"yaml_text": text})
                elif consumer_cycle == 2 and self.results.get("validated", {}).get("valid") is True:
                    request = ("saved", "save_spec", {"yaml_text": text, "path": self.save_path})
                elif consumer_cycle == 3 and self.results.get("saved", {}).get("saved") is True:
                    request = ("bound", "bind_coding_generation_spec", {"yaml_text": text})
        yield {"messageStart": {"role": "assistant"}}
        if request is None:
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "planning stopped"}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return
        identifier, name, inputs = request
        yield {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {
                    "toolUse": {"toolUseId": identifier, "name": name},
                },
            }
        }
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "toolUse": {"input": json.dumps(inputs)},
                },
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "tool_use"}}


@pytest.mark.parametrize(
    "description",
    [
        DESCRIPTION,
        DESCRIPTION.replace(
            "The owner requests unbounded trials",
            "The owner requests no restrictions: unbounded trials",
        ),
    ],
    ids=["original_unbounded", "no_restrictions_unbounded"],
)
def test_registered_generator_text_connects_validate_save_and_private_binder(
    generation_state,
    tmp_path: Path,
    description,
):
    store, adapter, reads, _, _, _ = generation_state
    results = {}
    model = PlanningConnectionModel(results, str(tmp_path / "generated.md"), description=description)
    names = {"generate_spec_from_description", "validate_spec", "save_spec", "bind_coding_generation_spec"}
    registered = {tool_display_name(tool): tool for tool in get_safe_llm_tools()}
    assert names <= registered.keys()
    tools = [registered[name] for name in sorted(names)]
    for tool in tools:
        properties = tool.tool_spec["inputSchema"]["json"]["properties"]
        assert not set(properties) & {"tool_context", "policy", "native_handle", "spec_ref"}
        assert not any(name.startswith("_amlas_") for name in properties)
    agent = Agent(model=model, tools=tools, callback_handler=lambda *args, **kwargs: None)

    def collect(event: AfterToolCallEvent):
        results[event.tool_use["toolUseId"]] = payload(event)

    agent.hooks.add_callback(AfterToolCallEvent, collect)
    policy = ApprovalPolicy()
    original_models, original_cases, original_sft = store._models, store._cases, store._sft
    agent(description, invocation_state={"_amlas_generation_store": store, "_amlas_approval_policy": policy})
    assert "generated" in results
    generated = results["generated"]
    assert generated["task_type"] == "text_generation"
    assert generated["planning_status"] == "schema_valid"
    assert generated["native_admission"] is False
    assert generated["quality_success"] is False
    assert generated["executed"] is False
    assert generated["primary_metric"] == "pass_at_1"
    assert generated["target"] == "python_module"
    assert generated["success_threshold"] is None
    assert adapter.calls == [] and reads == []
    assert store._models is original_models and store._cases is original_cases and store._sft is original_sft
    assert type(generated.get("spec_text")) is str, (
        "real generator must produce coding planning text for its normal registered consumers"
    )
    assert list(results) == ["generated", "validated", "saved", "bound"]
    assert results["validated"]["valid"] is True
    assert results["saved"]["saved"] is True
    assert (tmp_path / "generated.md").read_text() == generated["spec_text"]
    assert results["bound"]["success"] is True
    assert results["bound"]["native_admission"] is results["bound"]["quality_success"] is False
    parsed = load_spec_from_text(generated["spec_text"])
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    assert parsed.generation.model.repo_id == expected["repo_id"]
    assert parsed.generation.model.revision == expected["revision"]
    assert parsed.generation.tokenizer == parsed.generation.model
    assert parsed.generation.recipe.id == "full_v1" and parsed.generation.recipe.version == "1"
    assert parsed.generation.quality.metric == "pass_at_1"
    assert parsed.baseline.model_dump()["pass_at_1"] is None
    assert parsed.generation.baseline_measurement.state == "unmeasured"
    assert parsed.generation.preparation.manifest_ref is None
    assert parsed.generation.evaluation.case_manifest_ref is None
    assert parsed.acceptance.primary_metric_delta_vs_baseline == "> 0.0"
    assert (
        parsed.compute_budget.gpu_hours
        == parsed.compute_budget.trials
        == parsed.compute_budget.wall_clock_minutes
        == parsed.compute_budget.max_retreats
        == 0
    )
    assert len(store._specs) == 1 and adapter.calls == [] and reads == []


@pytest.mark.parametrize(
    "budget,unbounded",
    [
        ({"trials": 4, "gpu_hours": 40, "wall_clock_minutes": 180}, False),
        ({"trials": 0, "gpu_hours": 0, "wall_clock_minutes": 0}, False),
        ({"trials": 0, "gpu_hours": 0, "wall_clock_minutes": 0, "max_retreats": 0}, True),
    ],
)
def test_registered_generator_preserves_explicit_planning_data_without_magic_prose(
    generation_state,
    tmp_path: Path,
    budget,
    unbounded,
):
    store, adapter, reads, _, _, _ = generation_state
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    planning_data = {
        "model_catalog": {"id": "qwen3_5_9b", "version": "1"},
        "target": "python_module",
        "golden_dataset": "./heldout_cases.json",
        "acceptance": {"primary_metric_delta_vs_baseline": ">= 0.02", "no_subgroup_regression": False},
        **({"unbounded_budget": True} if unbounded else {"compute_budget": budget}),
    }
    results = {}
    model = PlanningConnectionModel(results, str(tmp_path / "finite.md"), planning_data)
    names = {"generate_spec_from_description", "validate_spec", "save_spec", "bind_coding_generation_spec"}
    registered = {tool_display_name(tool): tool for tool in get_safe_llm_tools()}
    agent = Agent(
        model=model,
        tools=[registered[name] for name in sorted(names)],
        callback_handler=lambda *args, **kwargs: None,
    )
    agent.hooks.add_callback(
        AfterToolCallEvent, lambda event: results.__setitem__(event.tool_use["toolUseId"], payload(event))
    )
    agent(
        "Use the explicit owner/researched planning DATA.",
        invocation_state={"_amlas_generation_store": store, "_amlas_approval_policy": ApprovalPolicy()},
    )
    assert "generated" in results
    generated = results["generated"]
    assert type(generated.get("spec_text")) is str, (
        "actual generator must accept explicit finite planning DATA instead of requiring magic prose"
    )
    assert results["research"]["planning_status"] == "research_needed"
    assert not results["research"].get("spec_text")
    assert {"model_selection", "target", "golden_dataset", "acceptance"} <= set(
        results["research"]["needs_research"]
    )
    assert list(results) == ["research", "generated", "validated", "saved", "bound"]
    assert results["validated"]["valid"] is True and results["saved"]["saved"] is True
    assert results["bound"]["success"] is True
    parsed = load_spec_from_text(generated["spec_text"])
    assert parsed.task_type == "text_generation" and parsed.primary_metric == "pass_at_1"
    assert parsed.target == "python_module" and parsed.golden_dataset == "./heldout_cases.json"
    assert parsed.generation.model.repo_id == expected["repo_id"]
    assert parsed.generation.model.revision == expected["revision"]
    assert parsed.generation.tokenizer == parsed.generation.model
    assert parsed.acceptance.primary_metric_delta_vs_baseline == ">= 0.02"
    assert parsed.acceptance.no_subgroup_regression is False
    for name, value in budget.items():
        assert getattr(parsed.compute_budget, name) == value
    if unbounded:
        assert generated["source_schema_defaults"] == []
    else:
        assert "max_retreats:" not in generated["spec_text"], (
            "generator must not invent an omitted internal retreat budget"
        )
        assert generated["source_schema_defaults"] == ["compute_budget.max_retreats"]
    assert parsed.baseline.model_dump()["pass_at_1"] is None
    assert parsed.generation.baseline_measurement.state == "unmeasured"
    assert (
        parsed.generation.preparation.manifest_ref is parsed.generation.evaluation.case_manifest_ref is None
    )
    assert adapter.calls == [] and reads == [] and len(store._specs) == 1
    assert results["bound"]["native_admission"] is results["bound"]["quality_success"] is False


def test_registered_coding_generator_treats_s3_path_as_planning_data(
    generation_state,
    tmp_path: Path,
    monkeypatch,
):
    import pandas as pd

    from amlas.tools import spec_generation_tools

    store, adapter, reads, _, _, _ = generation_state
    parser_calls = []

    def column_parser(path, target):
        parser_calls.append(("column", path, target))
        return False

    def csv_parser(path, **kwargs):
        parser_calls.append(("csv", path, kwargs))
        raise ValueError("ordinary planning parser sentinel")

    monkeypatch.setattr(spec_generation_tools, "_validate_column_in_data", column_parser)
    monkeypatch.setattr(pd, "read_csv", csv_parser)
    dataset = "s3://planning-fixture/coding-cases.json"
    results = {}
    model = PlanningConnectionModel(results, str(tmp_path / "coding-path.md"), data_path=dataset)
    names = {"generate_spec_from_description", "validate_spec", "save_spec", "bind_coding_generation_spec"}
    registered = {tool_display_name(tool): tool for tool in get_safe_llm_tools()}
    agent = Agent(
        model=model,
        tools=[registered[name] for name in sorted(names)],
        callback_handler=lambda *args, **kwargs: None,
    )
    agent.hooks.add_callback(
        AfterToolCallEvent, lambda event: results.__setitem__(event.tool_use["toolUseId"], payload(event))
    )
    agent(
        DESCRIPTION,
        invocation_state={"_amlas_generation_store": store, "_amlas_approval_policy": ApprovalPolicy()},
    )
    assert list(results) == ["generated", "validated", "saved", "bound"]
    assert results["generated"]["task_type"] == "text_generation"
    assert results["validated"]["valid"] is True and results["saved"]["saved"] is True
    assert results["bound"]["success"] is True
    parsed = load_spec_from_text(results["generated"]["spec_text"])
    assert parsed.golden_dataset == dataset
    assert parsed.baseline.model_dump()["pass_at_1"] is None
    assert adapter.calls == [] and reads == [] and len(store._specs) == 1
    assert results["bound"]["native_admission"] is results["bound"]["quality_success"] is False
    assert parser_calls == [], "coding planning DATA must never enter caller-process dataset parsers"


def test_registered_tabular_generator_retains_column_validation(
    generation_state,
    tmp_path: Path,
    monkeypatch,
):
    from amlas.tools import spec_generation_tools

    store, adapter, reads, _, _, _ = generation_state
    parser_calls = []

    def column_parser(path, target):
        parser_calls.append((path, target))
        return True

    monkeypatch.setattr(spec_generation_tools, "_validate_column_in_data", column_parser)
    results = {}
    description = "Classification of customer churn. The target is 'outcome'. Require F1 score."
    model = PlanningConnectionModel(
        results,
        str(tmp_path / "unused.md"),
        data_path="./ordinary.csv",
        description=description,
    )
    registered = {tool_display_name(tool): tool for tool in get_safe_llm_tools()}
    agent = Agent(
        model=model,
        tools=[registered["generate_spec_from_description"]],
        callback_handler=lambda *args, **kwargs: None,
    )
    agent.hooks.add_callback(
        AfterToolCallEvent, lambda event: results.__setitem__(event.tool_use["toolUseId"], payload(event))
    )
    agent(
        description,
        invocation_state={"_amlas_generation_store": store, "_amlas_approval_policy": ApprovalPolicy()},
    )
    assert list(results) == ["generated"]
    assert results["generated"]["task_type"] == "classification"
    assert results["generated"]["target"] == "outcome"
    assert parser_calls == [("./ordinary.csv", "outcome")]
    assert "spec_text" not in results["generated"]
    assert adapter.calls == [] and reads == [] and not store._specs


@pytest.mark.parametrize(
    "declaration",
    [
        "The owner explicitly rejects unbounded trials",
        "The owner does not request unbounded trials",
        "The owner must not use unbounded trials",
        "The owner does not want unbounded trials",
        "The owner requests no unbounded trials",
        "The owner permits no use of unbounded trials",
    ],
)
def test_registered_generator_negated_unbounded_budget_needs_research(tmp_path, declaration):
    description = DESCRIPTION.replace("The owner requests unbounded trials", declaration)
    results = {}
    class SinglePlanningModel(PlanningConnectionModel):
        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            if self.cycles:
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
                yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "planning observed"}}}
                yield {"contentBlockStop": {"contentBlockIndex": 0}}
                yield {"messageStop": {"stopReason": "end_turn"}}
                return
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event

    model = SinglePlanningModel(results, str(tmp_path / "budget.md"), description=description)
    registered = {tool_display_name(tool): tool for tool in get_safe_llm_tools()}
    # A single real registered generator runs; no fixture-authored YAML or consumer.
    agent = Agent(
        model=model,
        tools=[registered["generate_spec_from_description"]],
        callback_handler=lambda *args, **kwargs: None,
    )

    def collect(event: AfterToolCallEvent):
        results[event.tool_use["toolUseId"]] = payload(event)

    agent.hooks.add_callback(AfterToolCallEvent, collect)
    agent(description, invocation_state={"_amlas_approval_policy": ApprovalPolicy()})
    generated = results["generated"]
    assert generated["planning_status"] == "research_needed"
    assert "compute_budget" in generated["needs_research"]
    assert "spec_text" not in generated
    assert generated["native_admission"] is False
    assert generated["quality_success"] is False
    assert generated["executed"] is False
