"""Mechanical fresh public SDK snapshot roundtrip; no runtime authority proof.

Agents, pending history, task and observation are freshly declared. No invocation,
native/context qualification, real completion CAS or cold recovery is exercised.
"""

from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace

import pytest
from strands import Agent, Snapshot
from strands.models import Model

from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.tool_binding_backend import (
    ToolBindingBackend,
    _InputAdmissionHeldError,
    _record_session_snapshot,
)


class InertSnapshotModel(Model):
    def __init__(self):
        self.stream_calls = 0

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "declared-snapshot-component"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("snapshot mechanics cannot invoke structured output")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.stream_calls += 1
        yield {"messageStart": {"role": "assistant"}}
        raise AssertionError("snapshot mechanics cannot invoke a model stream")


@pytest.mark.parametrize("completion_allowed", [True, False], ids=["allowed", "denied"])
def test_fresh_sdk_completion_snapshot_preserves_history_and_all_other_state(monkeypatch, completion_allowed):
    digest = "1" * 64
    use_id = "declared-sdk-sft-use"
    job_name = "declared-sdk-original"
    job_arn = "arn:aws:sagemaker:us-west-2:111122223333:training-job/" + job_name
    prefix = [
        {"role": "user", "content": [{"text": "Declared SDK mechanical snapshot."}]},
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
    saved_model, fresh_model = InertSnapshotModel(), InertSnapshotModel()
    original = Agent(model=saved_model, tools=[], messages=copy.deepcopy(prefix), callback_handler=None)
    saved = _record_session_snapshot(original)
    assert set(saved) == {"scope", "schema_version", "created_at", "data", "app_data"}
    assert set(saved["data"]) == {"messages", "state", "conversation_manager_state", "interrupt_state"}
    assert saved["data"]["messages"] == prefix
    immutable_saved = copy.deepcopy(saved)
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
        "sdk_snapshot": saved,
    }
    immutable_task = copy.deepcopy(task)
    assessments = []

    class DeclaredChecker:
        def check(self, text, *, boundary, opaque_cycle, tool_name):
            assert opaque_cycle == "declared-sdk-component-cycle"
            assessments.append((boundary, tool_name))
            allowed = boundary == "HISTORY_TEXT" or (
                boundary == "TOOL_RESULT" and tool_name == "submit_qwen_sft" and completion_allowed
            )
            return SimpleNamespace(
                outcome="ALLOW" if allowed else "BLOCKED",
                forward_text=text if allowed else None,
                receipt={"boundary": boundary, "outcome": "ALLOW" if allowed else "BLOCKED"},
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
        "trained": True,
        "full_training_complete": True,
    }
    checker = DeclaredChecker()
    monkeypatch.setattr(backend, "_selected_input_checker", lambda: checker)
    engine = object.__new__(ChatEngine)
    engine.tool_backend = backend
    fresh = Agent(model=fresh_model, tools=[], callback_handler=None)
    untouched_fresh = copy.deepcopy(fresh.messages)

    if completion_allowed:
        ready = engine.assemble_normal_completion(task)
        assert ready["data"]["messages"][:-1] == prefix
        completion = ready["data"]["messages"][-1]
        result = completion["content"][0]["toolResult"]
        assert completion["role"] == "user"
        assert result["toolUseId"] == use_id and result["status"] == "success"
        assert json.loads(result["content"][0]["text"]) == {
            "schema_version": 1,
            "event_type": "original_sft_completed",
            "data_only": True,
            "request_ref": "sft_" + digest[:32],
            "tool_use_id": use_id,
            "status": "verified_completed",
        }
        for field in ("scope", "schema_version", "created_at", "app_data"):
            assert ready[field] == immutable_saved[field]
        for field in ("state", "conversation_manager_state", "interrupt_state"):
            assert ready["data"][field] == immutable_saved["data"][field]
        fresh.load_snapshot(Snapshot.from_dict(ready))
        loaded = _record_session_snapshot(fresh)
        assert loaded["data"] == ready["data"]
        assert fresh.messages == ready["data"]["messages"]
    else:
        with pytest.raises(_InputAdmissionHeldError):
            engine.assemble_normal_completion(task)
        assert backend._input_hold is True
        assert fresh.messages == untouched_fresh

    assert ("TOOL_RESULT", "submit_qwen_sft") in assessments
    assert saved == immutable_saved and task == immutable_task
    assert original.messages == prefix
    assert saved_model.stream_calls == fresh_model.stream_calls == 0
