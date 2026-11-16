"""The SFT preview and tool text state the single-submission rule and name the lifecycle recipe.

A normal run keeps exactly one original training call; a second effectful submit_qwen_sft ends the run. The preview
(plan_only) is the agent's last point to choose the recipe, so it says so. Neutral stand-ins only; no AWS.
"""

from __future__ import annotations

import pytest

from amlas.flows.qwen_sft_flow import run_qwen_sft_flow
from amlas.tools.qwen_sft import get_qwen_sft, submit_qwen_sft


def plan(recipe_id):
    return run_qwen_sft_flow(plan_only=True, recipe_id=recipe_id)["plan"]


@pytest.mark.parametrize("recipe_id", ["fit_v1", "full_v1"])
def test_preview_states_the_single_submission_rule_and_the_lifecycle_recipe(recipe_id):
    result = plan(recipe_id)
    assert result["lifecycle_recipe"] == "full_v1"
    rule = result["submission_rule"]
    assert "one training submission" in rule and "plan_only" in rule and "get_qwen_sft" in rule


def test_the_pilot_preview_warns_that_only_full_v1_continues_the_lifecycle():
    note = plan("fit_v1")["recipe_note"]
    assert "pilot" in note and "full_v1" in note and "no second submission" in note


def test_the_full_preview_has_no_pilot_warning():
    assert "recipe_note" not in plan("full_v1")


@pytest.mark.parametrize("recipe_id", ["fit_v1", "full_v1"])
def test_a_refused_call_still_carries_the_same_plan_text(recipe_id):
    result = run_qwen_sft_flow(recipe_id=recipe_id)
    assert (
        result["status"] == "denied"
        and result["plan"]["submission_rule"] == plan(recipe_id)["submission_rule"]
    )


def test_an_unsupported_recipe_keeps_its_refusal_without_new_text_claims():
    result = run_qwen_sft_flow(plan_only=True, recipe_id="other")
    assert result["status"] == "unsupported_recipe"


def description(tool):
    return tool.tool_spec["description"]


def test_tool_texts_say_to_submit_once_and_poll():
    submit = description(submit_qwen_sft)
    assert "one training submission" in submit and "get_qwen_sft" in submit and "full_v1" in submit
    assert "never submit a replacement" in description(get_qwen_sft)
