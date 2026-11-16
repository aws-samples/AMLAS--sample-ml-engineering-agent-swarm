"""A replacement continues a completed fit_v1 original as the original worker's own wait does.

The adapter reports `full_training_complete` only for `full_v1` (a fit pilot saves a checkpoint and is
never a complete full training). The original worker's wait accepts a completed fit_v1 original; the four
restore sites required the full flag and so never continued one. Each site is exercised with freshly
declared observations: a completed fit pilot is accepted, and a trained `full_v1` original without its
full-training proof is refused exactly as before. Declared observations are unit DATA; they establish no
native completion, ownership or lifecycle proof.
"""

from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace

import pytest
import test_adopted_parent_restore_sites as claim_fixture
from strands import Agent
from test_normal_completion_route import _routing_case
from test_normal_completion_sdk_snapshot import InertSnapshotModel
from test_runtime_busy_ownership import managed_report_context as managed_report_context

from amlas.control_plane import normal_run_checkpoint, sft_state
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat import research_runtime_profile
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend, _record_session_snapshot

FIT = {"recipe_id": "fit_v1", "trained": True, "full_training_complete": False}
FULL = {"recipe_id": "full_v1", "trained": True, "full_training_complete": True}
FULL_WITHOUT_PROOF = {"recipe_id": "full_v1", "trained": True, "full_training_complete": False}
UNKNOWN_RECIPE = {"recipe_id": "other_v1", "trained": True, "full_training_complete": False}
NO_RECIPE = {"trained": True, "full_training_complete": False}
UNTRAINED_FIT = {"recipe_id": "fit_v1", "trained": False, "full_training_complete": False}
REFUSED = {
    "full-without-proof": FULL_WITHOUT_PROOF,
    "unknown-recipe": UNKNOWN_RECIPE,
    "no-recipe": NO_RECIPE,
    "untrained-fit": UNTRAINED_FIT,
}


def test_restore_completion_rule_accepts_what_the_original_wait_accepts():
    rule = normal_run_checkpoint._original_training_complete
    assert rule(FIT) is True and rule(FULL) is True
    # An observation without a recipe or with another recipe is continued only on the full-training
    # proof, as before.
    assert rule({"trained": True, "full_training_complete": True}) is True
    assert rule({**UNKNOWN_RECIPE, "full_training_complete": True}) is True
    for name, observed in REFUSED.items():
        assert rule(observed) is False, name
    # Only the exact values count; a truthy stand-in proves nothing.
    assert rule({**FIT, "trained": 1}) is False
    assert rule({**FULL_WITHOUT_PROOF, "full_training_complete": 1}) is False
    assert rule({**FIT, "trained": "true"}) is False


def test_resume_route_claims_a_completed_fit_original(monkeypatch, managed_report_context):
    _routing_case(
        monkeypatch,
        "completed",
        "checkpoint_saved",
        managed_report_context,
        observation={"status": "checkpoint_saved", **FIT},
    )


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_resume_route_claims_nothing_without_a_provable_completion(monkeypatch, managed_report_context, name):
    _routing_case(
        monkeypatch,
        "refused",
        "checkpoint_saved",
        managed_report_context,
        observation={"status": "checkpoint_saved", **REFUSED[name]},
    )


def _assembly_world(monkeypatch, observed, recipe_id):
    digest = "1" * 64
    use_id = "declared-fit-sft-use"
    job_name = "declared-fit-original"
    job_arn = "arn:aws:sagemaker:us-west-2:111122223333:training-job/" + job_name
    prefix = [
        {"role": "user", "content": [{"text": "Declared fit pilot snapshot."}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": use_id,
                        "name": "submit_qwen_sft",
                        "input": {
                            "model_ref": "model_acq_" + "2" * 32,
                            "preparation_ref": "prep_" + "3" * 32,
                            "recipe_id": recipe_id,
                        },
                    }
                }
            ],
        },
    ]
    original = Agent(
        model=InertSnapshotModel(), tools=[], messages=copy.deepcopy(prefix), callback_handler=None
    )
    task = {
        "schema_version": 2,
        "logical_run_id": "0" * 32,
        "private_store_correlation": "declared-component-store",
        "source_refs": [],
        "causal_parent_ref": None,
        "operation_digest": digest,
        "native_job_name": job_name,
        "native_job_arn": job_arn,
        "pending_tool_use_id": use_id,
        "sdk_snapshot": _record_session_snapshot(original),
    }
    assessments = []

    class DeclaredChecker:
        def check(self, text, *, boundary, opaque_cycle, tool_name):
            assessments.append((boundary, tool_name))
            return SimpleNamespace(
                outcome="ALLOW", forward_text=text, receipt={"boundary": boundary, "outcome": "ALLOW"}
            )

    backend = object.__new__(ToolBindingBackend)
    backend.approval_policy = ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"])
    backend._input_lock = threading.RLock()
    backend._input_hold = False
    backend._input_cycle = "declared-sdk-component-cycle"
    backend._input_receipts = []
    backend._normal_original_result = {
        "request_ref": "sft_" + digest[:32],
        "job_name": job_name,
        "job_arn": job_arn,
        **observed,
    }
    checker = DeclaredChecker()
    monkeypatch.setattr(backend, "_selected_input_checker", lambda: checker)
    engine = object.__new__(ChatEngine)
    engine.tool_backend = backend
    return engine, task, prefix, use_id, digest, assessments


def test_completion_assembly_accepts_a_completed_fit_original(monkeypatch):
    engine, task, prefix, use_id, digest, assessments = _assembly_world(monkeypatch, FIT, "fit_v1")
    ready = engine.assemble_normal_completion(task)
    assert ready["data"]["messages"][:-1] == prefix
    result = ready["data"]["messages"][-1]["content"][0]["toolResult"]
    assert result["toolUseId"] == use_id and result["status"] == "success"
    assert json.loads(result["content"][0]["text"]) == {
        "schema_version": 1,
        "event_type": "original_sft_completed",
        "data_only": True,
        "request_ref": "sft_" + digest[:32],
        "tool_use_id": use_id,
        "status": "verified_completed",
    }
    assert ("TOOL_RESULT", "submit_qwen_sft") in assessments


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_completion_assembly_refuses_before_any_admission(monkeypatch, name):
    observed = REFUSED[name]
    engine, task, _prefix, _use, _digest, assessments = _assembly_world(
        monkeypatch, observed, observed.get("recipe_id", "full_v1")
    )
    with pytest.raises(ValueError, match="fresh full original SFT completion required"):
        engine.assemble_normal_completion(task)
    assert assessments == [] and engine.tool_backend._input_hold is False


@pytest.mark.parametrize("field", ["request_ref", "job_name", "job_arn"])
def test_completion_assembly_refuses_a_fit_original_of_another_job(monkeypatch, field):
    engine, task, _prefix, _use, _digest, assessments = _assembly_world(monkeypatch, FIT, "fit_v1")
    engine.tool_backend._normal_original_result[field] += "-other"
    with pytest.raises(ValueError, match="fresh full original SFT completion required"):
        engine.assemble_normal_completion(task)
    assert assessments == [] and engine.tool_backend._input_hold is False


RESEARCH_GATE = "claimed original research reconstruction refused"
RESEARCH_LINK = "original research result linkage differs"


def _research_restore(monkeypatch, observed):
    """A declared claimed TASK whose last research result is an error.

    Its envelope and snapshot are valid, so the claim gate is the first check that can refuse; past the
    gate the invalid research link refuses, before any admission.
    """
    plan = research_runtime_profile._plan()
    prefix = [{"role": "user", "content": [{"text": "fresh declared request"}]}]
    for index, unit in enumerate(plan.units):
        identifier = "declared-use-" + str(index)
        prefix.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": identifier,
                            "name": "research_fetch_urls",
                            "input": {"urls": list(unit.urls), "max_pages": 3},
                        }
                    }
                ],
            }
        )
        prefix.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": identifier,
                            "status": "success",
                            "content": [{"json": {"pages": [{"text": "fresh declared source"}]}}],
                        }
                    }
                ],
            }
        )
    prefix[-1]["content"][0]["toolResult"]["status"] = "error"
    saved = _record_session_snapshot(
        Agent(model=InertSnapshotModel(), tools=[], messages=copy.deepcopy(prefix), callback_handler=None)
    )
    task = {
        "schema_version": 2,
        "logical_run_id": "0" * 32,
        "private_store_correlation": "declared-component-store",
        "source_refs": [],
        "causal_parent_ref": None,
        "operation_digest": "1" * 64,
        "native_job_name": "declared-original",
        "native_job_arn": "declared-arn",
        "pending_tool_use_id": "declared-sft-use",
        "sdk_snapshot": saved,
    }
    snapshot = copy.deepcopy(saved)
    snapshot["data"]["messages"].append({"role": "user", "content": []})
    context = SimpleNamespace(
        _completion_receipt=("declared-token", json.dumps({"sdk_snapshot": snapshot}), "declared-task")
    )
    monkeypatch.setattr(normal_run_checkpoint, "current_normal_run", lambda: context)
    admissions = []
    state = {"phase": "UNUSED", "pending": False, "units": {unit.unit_id: unit for unit in plan.units}}
    backend = SimpleNamespace(
        _research_started=False,
        _normal_original_result={"job_name": "declared-original", "job_arn": "declared-arn", **observed},
        _qualify_research_binding=lambda **kwargs: state,
        _planning_result_payload=ToolBindingBackend._planning_result_payload,
        _admit_input=lambda *args, **kwargs: admissions.append(True),
    )
    owner = SimpleNamespace(tool_backend=backend)
    with pytest.raises(ValueError) as refused:
        research_runtime_profile._restore_runtime_research_profile(
            owner, context, task, "declared-token", snapshot
        )
    assert admissions == [] and state["phase"] == "UNUSED"
    return str(refused.value)


@pytest.mark.parametrize("name", ["fit", "full"])
def test_research_restore_claim_gate_passes_a_completed_original(monkeypatch, name):
    # Past the gate the declared late research link is what refuses, for both recipes alike.
    assert _research_restore(monkeypatch, {"fit": FIT, "full": FULL}[name]) == RESEARCH_LINK


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_research_restore_claim_gate_refuses_without_a_provable_completion(monkeypatch, name):
    assert _research_restore(monkeypatch, REFUSED[name]) == RESEARCH_GATE


def _claim_world(monkeypatch, tmp_path, observed):
    hydrated, model_ref, binding, _policy = claim_fixture._adopted_cold_parents(monkeypatch, tmp_path)
    _acquired, models, _cases, prepared = hydrated
    original, built, sft_adapter = claim_fixture._sft_world(model_ref, binding)
    declared = {
        "request_ref": claim_fixture.SFT_ALIAS,
        "job_name": claim_fixture.JOB_NAME,
        "job_arn": claim_fixture.JOB_ARN,
        "status": "checkpoint_saved",
        **observed,
    }
    built.observe = lambda alias: dict(declared)
    fenced = claim_fixture._install_seams(monkeypatch, original, binding, built)
    return models, prepared, built, original, binding, sft_adapter, fenced


def test_claim_qualifier_accepts_a_completed_fit_original(monkeypatch, tmp_path):
    models, prepared, built, original, binding, sft_adapter, fenced = _claim_world(monkeypatch, tmp_path, FIT)
    claim_fixture._claim_qualifier(models, prepared, built, original, binding)
    assert fenced == [(prepared._adapter, sft_adapter)]


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_claim_qualifier_refuses_without_a_provable_completion(monkeypatch, tmp_path, name):
    models, prepared, built, original, binding, _adapter, _fenced = _claim_world(
        monkeypatch, tmp_path, REFUSED[name]
    )
    with pytest.raises(ValueError, match="fresh full original completion proof unavailable"):
        claim_fixture._claim_qualifier(models, prepared, built, original, binding)


@pytest.mark.parametrize("field", ["request_ref", "job_name", "job_arn"])
def test_claim_qualifier_refuses_a_fit_original_of_another_job(monkeypatch, tmp_path, field):
    other = {
        field: {"request_ref": "sft_" + "f" * 32, "job_name": "other-job", "job_arn": "other-arn"}[field]
    }
    models, prepared, built, original, binding, _adapter, _fenced = _claim_world(
        monkeypatch, tmp_path, {**FIT, **other}
    )
    with pytest.raises(ValueError, match="fresh full original completion proof unavailable"):
        claim_fixture._claim_qualifier(models, prepared, built, original, binding)


def test_a_fit_pilot_checkpoint_still_cannot_feed_tuned_generation():
    """Continuing a completed fit pilot changes nothing for generation: only a full_v1 original qualifies."""
    reference, model = "sft_" + "a" * 32, object()

    class NoObservation:
        def artifacts(self, handle):
            raise AssertionError("a fit original is refused before any observation")

    store = object.__new__(sft_state.SessionSftStore)
    store._lock = threading.RLock()
    store.runtime_session_id = "declared-session"
    store._adapter = NoObservation()
    store._records = {
        reference: SimpleNamespace(
            extra={
                "operation": "train",
                "recipe_id": "fit_v1",
                "runtime_session_id": "declared-session",
                "model_source": model,
            }
        )
    }
    with pytest.raises(ValueError, match="original same-model full train reference required"):
        store._verified_generation_checkpoint(reference, model)
