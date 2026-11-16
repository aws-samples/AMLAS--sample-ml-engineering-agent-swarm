"""A PENDING/RUNNING normal restore proceeds when guardrail checks are disabled.

The deployed configuration sets AMLAS_GUARDRAIL_ENABLED=false, so the selected INPUT
checker is None by design and admission is a pass-through. The completed-claim path
already accepted that; the pending branch refused with "current normal snapshot
admission unavailable". Enabled checks with no checker still refuse.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from test_normal_checkpoint_cas_contract import _policy

from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend


def _task(messages):
    snapshot = {
        "scope": "agent",
        "schema_version": "1.0",
        "created_at": "2026-11-07T00:00:00+00:00",
        "data": {
            "messages": messages,
            "state": {},
            "conversation_manager_state": {},
            "interrupt_state": {},
        },
        "app_data": {},
    }
    return {
        "schema_version": 2,
        "logical_run_id": "1" * 32,
        "private_store_correlation": "12345678-1234-4234-9234-00000000dd41",
        "native_job_name": "declared-sft-0123456789ab",
        "native_job_arn": "arn:aws:sagemaker:us-west-2:111122223333:training-job/declared-sft-0123456789ab",
        "operation_digest": "d" * 64,
        "source_refs": ["model_acq_" + "a" * 32, "prep_" + "b" * 32],
        "causal_parent_ref": {"uri": "s3://declared/parent", "version_id": "1", "sha256": "e" * 64},
        "sdk_snapshot": snapshot,
        "pending_tool_use_id": "declared-tool-use",
    }


def _engine(monkeypatch):
    messages = [
        {"role": "user", "content": [{"text": "Declared admitted goal."}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "declared-tool-use", "name": "submit_qwen_sft", "input": {}}}
            ],
        },
    ]
    loaded = []

    class DeclaredAgent:
        def __init__(self):
            self.messages = []

        def load_snapshot(self, snapshot):
            loaded.append(snapshot)
            self.messages = list(snapshot["data"]["messages"])

    agent = DeclaredAgent()
    monkeypatch.setattr("strands.Snapshot", SimpleNamespace(from_dict=lambda value: value))
    backend = object.__new__(ToolBindingBackend)
    backend.approval_policy = _policy()
    backend._input_lock = threading.RLock()
    backend._input_hold = False
    backend._input_receipts = []
    backend._input_checker = None
    backend._normal_original_result = {"status": "RUNNING"}
    monkeypatch.setattr(backend, "_activate_research_binding", lambda: None)
    monkeypatch.setattr(backend, "_qualify_research_binding", lambda active=True: None)
    monkeypatch.setattr(backend, "build_agent", lambda: agent)
    engine = object.__new__(ChatEngine)
    engine.tool_backend = backend
    return engine, backend, agent, loaded, _task(messages)


def test_pending_restore_proceeds_when_guardrail_checks_are_disabled(monkeypatch):
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", "declared-guardrail-id")
    engine, backend, agent, loaded, task = _engine(monkeypatch)
    assert backend._selected_input_checker() is None
    result = engine.load_normal_snapshot(task)
    assert result == {"status": "RUNNING"}
    assert engine._normal_agent is agent and len(loaded) == 1
    assert agent.messages == task["sdk_snapshot"]["data"]["messages"]


def test_pending_restore_still_refuses_when_checks_are_enabled_but_no_checker_is_selected(monkeypatch):
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "true")
    monkeypatch.delenv("AMLAS_GUARDRAIL_ID", raising=False)
    engine, backend, _agent, loaded, task = _engine(monkeypatch)
    assert backend._selected_input_checker() is None
    with pytest.raises(ValueError, match="current normal snapshot admission unavailable"):
        engine.load_normal_snapshot(task)
    assert loaded == [] and not hasattr(engine, "_normal_agent")


def test_pending_restore_still_refuses_a_snapshot_that_differs_from_the_prefix(monkeypatch):
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    engine, _backend, _agent, loaded, task = _engine(monkeypatch)
    monkeypatch.setattr(checkpoint, "_task_causal_prefix", lambda value: [{"role": "user", "content": []}])
    with pytest.raises(ValueError, match="current normal snapshot admission unavailable"):
        engine.load_normal_snapshot(task)
    assert loaded == []
