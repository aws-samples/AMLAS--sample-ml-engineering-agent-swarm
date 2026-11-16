"""The process reports which provider answers a flow's SageMaker calls; the report never authorizes anything.

Neutral stand-ins only. A fake installation of the same shape as the declared controlled provider's stands in for it.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest import mock

import boto3
import pytest

from amlas.backends.adapters._util import raw_tool
from amlas.control_plane import provider_disclosure as disclosure
from amlas.control_plane.generation_state import CodingSpecValidationError, SessionCodingGenerationStore
from amlas.flows.coding_case_freeze_flow import run_coding_case_freeze_flow
from amlas.flows.coding_generation_flow import run_coding_generation_flow
from amlas.flows.coding_pair_flow import run_coding_pair_flow
from amlas.flows.coding_preparation_flow import run_coding_preparation_flow
from amlas.flows.dataset_acquisition_flow import run_dataset_acquisition_flow
from amlas.flows.model_weight_acquisition_flow import run_model_weight_acquisition_flow
from amlas.flows.qwen_sft_flow import run_qwen_sft_flow
from amlas.tools import coding_generation
from amlas.tools.coding_generation import bind_coding_generation_spec
from amlas.tools.qwen_sft import submit_qwen_sft

ALL_OPERATIONS = {
    "CreateTrainingJob",
    "DescribeTrainingJob",
    "CreateProcessingJob",
    "DescribeProcessingJob",
    "ListTags",
}
REF = "acq_" + "c" * 32


def fake_installation(
    allowed=ALL_OPERATIONS,
    *,
    pid=None,
    closed=False,
    check=None,
    module="managed_sagemaker_client_overlay",
    name="Installation",
):
    kind = type(name, (), {"__module__": module, "_check": check or (lambda self: None)})
    owner = kind()
    owner._allowed = frozenset(allowed) if type(allowed) in (set, frozenset, list, tuple) else allowed
    owner._pid = os.getpid() if pid is None else pid
    owner._closed = closed
    return owner


def install(monkeypatch, owner):
    def factory(session, service_name, *args, **kwargs):  # never called by these tests
        raise AssertionError("the SageMaker client factory must not be called")

    factory._managed_test_overlay_owner = owner
    monkeypatch.setattr(boto3.session.Session, "client", factory)


def test_without_a_marker_the_provider_is_unverified_and_never_claimed_real():
    result = disclosure.execution_provider(disclosure.TRAINING)
    assert result["mode"] == "unverified" and result["reason"] == "marker_absent"
    assert result["real_sagemaker_execution_possible"] is True
    assert "controlled_operations" not in result and "SIMULATED" not in result["result_label"]
    assert "real" not in {result["sagemaker_job_backing"], result["mode"]}


@pytest.mark.parametrize("operations", [disclosure.PROCESSING, disclosure.TRAINING])
def test_a_live_installation_covering_the_operations_is_reported_controlled(monkeypatch, operations):
    install(monkeypatch, fake_installation())
    result = disclosure.execution_provider(operations)
    assert (
        result["mode"] == "declared_controlled" and result["sagemaker_job_backing"] == "controlled_provider"
    )
    assert result["controlled_operations"] == sorted(operations)
    assert result["sagemaker_instances_started"] is False
    assert result["result_label"].startswith("SIMULATED (controlled provider)")
    assert "not model-quality evidence" in result["result_label"] and "plan_note" in result


def test_an_installation_that_misses_an_operation_is_not_reported_controlled(monkeypatch):
    install(monkeypatch, fake_installation({"CreateProcessingJob", "DescribeProcessingJob"}))
    assert disclosure.execution_provider(disclosure.PROCESSING)["mode"] == "declared_controlled"
    result = disclosure.execution_provider(disclosure.TRAINING)
    assert result["mode"] == "unverified" and result["reason"] == "operations_not_covered"


def refusing_check(self):
    raise RuntimeError("ordinary synthetic ownership failure detail")


@pytest.mark.parametrize(
    "owner",
    [
        fake_installation(name="Other"),
        fake_installation(module="some.other.module"),
        fake_installation(set()),
        fake_installation("CreateTrainingJob"),
        fake_installation([1, 2]),
        fake_installation(pid=os.getpid() + 1),
        fake_installation(closed=True),
        fake_installation(check=refusing_check),
        mock.MagicMock(),
        SimpleNamespace(),
        "installed",
    ],
    ids=[
        "wrong-class",
        "wrong-module",
        "empty-scope",
        "string-scope",
        "non-string-scope",
        "other-process",
        "closed",
        "ownership-check-raises",
        "mock",
        "bare-namespace",
        "string-marker",
    ],
)
def test_a_marker_that_is_not_a_live_well_formed_installation_fails_closed(monkeypatch, owner):
    install(monkeypatch, owner)
    result = disclosure.execution_provider(disclosure.TRAINING)
    assert result["mode"] == "unverified" and result["reason"] == "check_failed"
    assert "ordinary synthetic ownership failure detail" not in json.dumps(result)


@pytest.mark.parametrize("operations", [[], [["CreateTrainingJob"]], [1, "CreateTrainingJob"], None, 7])
def test_malformed_operation_names_fail_closed_instead_of_raising(monkeypatch, operations):
    install(monkeypatch, fake_installation())
    result = disclosure.execution_provider(operations)
    assert result["mode"] == "unverified" and result["reason"] == "check_failed"


def test_a_replaced_client_factory_that_is_not_a_plain_function_has_no_marker(monkeypatch):
    stand_in = mock.MagicMock()
    stand_in._managed_test_overlay_owner = fake_installation()
    monkeypatch.setattr(boto3.session.Session, "client", stand_in)
    assert disclosure.execution_provider(disclosure.TRAINING)["reason"] == "marker_absent"


def test_the_report_is_computed_on_every_call_and_never_cached(monkeypatch):
    install(monkeypatch, fake_installation())
    assert disclosure.execution_provider(disclosure.TRAINING)["mode"] == "declared_controlled"
    install(monkeypatch, fake_installation(closed=True))
    assert disclosure.execution_provider(disclosure.TRAINING)["mode"] == "unverified"
    install(monkeypatch, fake_installation())
    assert disclosure.execution_provider(disclosure.TRAINING)["mode"] == "declared_controlled"


# -- every managed flow carries the report in plan_only previews and refusals --


def previews():
    from amlas.data_eng import commitpack_profile as pins

    return {
        "dataset-acquisition": (
            lambda: run_dataset_acquisition_flow(
                repo_id=pins.DATASET_REPO, revision=pins.REVISION, filename=pins.FILENAME, plan_only=True
            ),
            disclosure.PROCESSING,
        ),
        "model-acquisition": (
            lambda: run_model_weight_acquisition_flow(
                catalog_id="qwen3_5_9b", catalog_version="1", plan_only=True
            ),
            disclosure.PROCESSING,
        ),
        "case-freeze": (lambda: run_coding_case_freeze_flow(plan_only=True), disclosure.PROCESSING),
        "preparation": (
            lambda: run_coding_preparation_flow(acquisition_request_refs=[REF], plan_only=True),
            disclosure.PROCESSING,
        ),
        "sft": (lambda: run_qwen_sft_flow(plan_only=True), disclosure.TRAINING),
        "generation": (lambda: run_coding_generation_flow(plan_only=True), disclosure.TRAINING),
        "pair": (lambda: run_coding_pair_flow(plan_only=True), disclosure.PROCESSING),
    }


@pytest.mark.parametrize("flow", sorted(previews()))
def test_each_managed_flow_preview_carries_the_report(monkeypatch, flow):
    call, operations = previews()[flow]
    unverified = call()
    assert unverified["execution_provider"]["mode"] == "unverified" and unverified["status"] == "planned"
    install(monkeypatch, fake_installation())
    controlled = call()
    assert controlled["execution_provider"] == disclosure.execution_provider(operations)
    assert controlled["execution_provider"]["mode"] == "declared_controlled"


@pytest.mark.parametrize(
    "call",
    [
        lambda: run_qwen_sft_flow(),
        lambda: run_coding_generation_flow(),
        lambda: run_coding_pair_flow(),
        lambda: run_coding_case_freeze_flow(),
    ],
    ids=["sft", "generation", "pair", "case-freeze"],
)
def test_a_controlled_provider_never_authorizes_an_effect(monkeypatch, call):
    install(monkeypatch, fake_installation())
    result = call()
    assert result["status"] == "denied" and result["executed"] is False
    assert result["execution_provider"]["mode"] == "declared_controlled"


# -- tool guidance for the next stalls --


def description(tool):
    return tool.tool_spec["description"]


def test_sft_tool_text_points_to_the_provider_report_and_the_lifecycle_recipe():
    text = description(submit_qwen_sft)
    assert "execution_provider" in text and "declared_controlled" in text and "result_label" in text
    assert "full_v1" in text
    recipe = submit_qwen_sft.tool_spec["inputSchema"]["json"]["properties"]["recipe_id"]["description"]
    assert "32" in recipe and "full_v1" in recipe


def test_spec_binder_text_names_the_generator_and_the_fixed_fields():
    text = description(bind_coding_generation_spec)
    for word in (
        "generate_spec_from_description",
        "planning_data",
        "text_generation",
        "pass_at_1",
        "full_v1",
    ):
        assert word in text


def test_every_closed_refusal_reason_has_a_static_next_step():
    from amlas.control_plane.generation_state import _CODING_SPEC_REASON_PATHS

    steps = coding_generation._NEXT_STEP
    assert set(steps) == set(_CODING_SPEC_REASON_PATHS)
    assert all(type(step) is str and step and "{" not in step for step in steps.values())


def test_a_spec_refusal_returns_the_static_next_step_and_never_the_input():
    store = SessionCodingGenerationStore.__new__(SessionCodingGenerationStore)

    def refuse(text):
        raise CodingSpecValidationError("generation_required", "ordinary synthetic refusal")

    store.bind_spec = refuse
    context = SimpleNamespace(invocation_state={"_amlas_generation_store": store})
    result = raw_tool(bind_coding_generation_spec)(
        "ordinary private specification detail", tool_context=context
    )
    assert result["status"] == "refused" and result["validation"]["reason_codes"] == ["generation_required"]
    assert result["next_step"] == coding_generation._NEXT_STEP["generation_required"]
    assert "ordinary private specification detail" not in json.dumps(result)
