"""Synthetic completion admission at the held normal assembly/claim boundary.

Declared inputs and inert effect probes establish no native/context qualification,
real completion CAS, checkpoint authority, SDK continuation or cold recovery proof.
Expected initial failure is a missing current ChatEngine claim/assembly helper.
"""

from __future__ import annotations

import copy
import threading
from types import SimpleNamespace

import pytest

from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend, _InputAdmissionHeldError


@pytest.mark.parametrize("completion_allowed", [False, True], ids=["denied", "allowed"])
def test_new_normal_completion_is_tool_result_admitted_before_claim_and_load(monkeypatch, completion_allowed):
    digest = "1" * 64
    use_id = "declared-sft-use"
    job_name = "declared-original-sft"
    job_arn = "arn:aws:sagemaker:us-west-2:111122223333:training-job/" + job_name
    prefix = [
        {"role": "user", "content": [{"text": "Declared completion admission component."}]},
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
                            "recipe_id": "full_v1",
                        },
                    }
                }
            ],
        },
    ]
    # This is the assembly method's declared DATA input, not an enrolled TASK.
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
        "sdk_snapshot": {
            "scope": "agent",
            "schema_version": "1.0",
            "created_at": "2026-01-01T00:00:00+00:00",
            "data": {
                "messages": copy.deepcopy(prefix),
                "state": {},
                "conversation_manager_state": {},
                "interrupt_state": {},
            },
            "app_data": {},
        },
    }
    assessments = []
    effects = []

    class DeclaredChecker:
        def check(self, text, *, boundary, opaque_cycle, tool_name):
            assert opaque_cycle == "declared-component-cycle"
            assessments.append((boundary, tool_name))
            allowed = boundary == "HISTORY_TEXT" or (
                boundary == "TOOL_RESULT" and tool_name == "submit_qwen_sft" and completion_allowed
            )
            return SimpleNamespace(
                outcome="ALLOW" if allowed else "BLOCKED",
                forward_text=text if allowed else None,
                receipt={"boundary": boundary, "outcome": "ALLOW" if allowed else "BLOCKED"},
            )

    checker = DeclaredChecker()
    backend = object.__new__(ToolBindingBackend)
    backend.approval_policy = ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"])
    backend._input_lock = threading.RLock()
    backend._input_hold = False
    backend._input_cycle = "declared-component-cycle"
    backend._input_receipts = []
    backend._sft_store = object()
    backend._normal_original_result = {
        "request_ref": "sft_" + digest[:32],
        "job_name": job_name,
        "job_arn": job_arn,
        "trained": True,
        "full_training_complete": True,
    }
    monkeypatch.setattr(backend, "_selected_input_checker", lambda: checker)
    monkeypatch.setattr(backend, "_activate_research_binding", lambda: None)
    monkeypatch.setattr(backend, "_qualify_research_binding", lambda *, active=True: None)

    class InertAgent:
        def load_snapshot(self, snapshot):
            effects.append("snapshot_load")

        async def invoke_async(self, *args, **kwargs):
            effects.append("inference")

    def build_agent():
        effects.append("agent_build")
        return InertAgent()

    monkeypatch.setattr(backend, "build_agent", build_agent)

    class InertContext:
        def qualify_original_completion(self, *args, **kwargs):
            effects.append("original_qualification")

        def claim_completion(self, *args, **kwargs):
            effects.append("completion_cas")
            return object()

        def qualify_completion(self, *args, **kwargs):
            effects.append("completion_qualification")

    engine = object.__new__(ChatEngine)
    engine.tool_backend = backend
    # Demonstrate that historical text is accepted by the real admission method;
    # denial must arise from the newly assembled completion TOOL_RESULT boundary.
    assert backend._admit_input("Declared historical input.", "HISTORY_TEXT", immutable=True)
    assert assessments == [("HISTORY_TEXT", "UNKNOWN")]
    assessments.clear()

    if not completion_allowed:
        with pytest.raises(_InputAdmissionHeldError):
            engine.claim_normal_completion(InertContext(), task)
        assert backend._input_hold is True
        assert effects == [], "completion denial must precede CAS, snapshot load and inference"
    else:
        token = engine.claim_normal_completion(InertContext(), task)
        assert token is engine._normal_completion[1]
        assert backend._input_hold is False
        assert effects == [
            "completion_cas",
            "completion_qualification",
            "agent_build",
            "completion_qualification",
            "snapshot_load",
            "completion_qualification",
        ]
        assert "inference" not in effects
    assert ("TOOL_RESULT", "submit_qwen_sft") in assessments
