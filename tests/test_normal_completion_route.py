"""Stateless normal resume routing; synthetic seams supply no native authority.

These observations are declared unit DATA, not qualified SDK/session ownership,
original completion, real CAS, checkpoint hydration or cold lifecycle evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_runtime_busy_ownership import managed_report_context as managed_report_context

from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import engine as chat_engine


def _routing_case(monkeypatch, kind, status, sdk_context, observation=None):
    """kind: pending, completed, or refused (a declared observation that may not be continued)."""
    with monkeypatch.context() as patch:
        calls = []
        active = [False]
        run_id = "1" * 32
        request = {"action": "resume_prompt_run", "run_id": run_id}
        sdk_session = "declared-routing-sdk-transport-session-0123456789"
        sdk_context.set_request_context("declared-routing-request", sdk_session)
        policy = object()  # Declared wiring marker, never an admission grant.
        task = {"component": "declared routing input"}
        if observation is None:
            observation = {
                "status": status,
                "trained": kind == "completed",
                "full_training_complete": kind == "completed",
            }
        pending = {
            "event": "prompt_run_status",
            "status": "available",
            "run_id": run_id,
            "found": True,
        }

        class DeclaredContext:
            def has_task_checkpoint(self, *, policy):
                calls.append("has_task")
                return True

            def load_qualified_task(self, *, policy):
                calls.append("checkpoint_load")
                return task, (), None, None

        context = DeclaredContext()

        @contextmanager
        def declared_scope(value):
            assert value is context
            assert not active[0]
            active[0] = True
            try:
                yield
            finally:
                active[0] = False

        class DeclaredEngine:
            def __init__(self, *, approval_policy, normal_restore):
                assert approval_policy is policy
                assert normal_restore == (context, task, (), None, None)
                self.tool_backend = SimpleNamespace(_normal_original_result=observation)

            def load_normal_snapshot(self, value):
                assert value is task
                calls.append("pending_snapshot_load")
                return observation

            def claim_normal_completion(self, value, checkpoint_task):
                assert value is context and checkpoint_task is task
                calls.append("completion_claim")
                return object()

            async def invoke_async(self, *args, **kwargs):
                calls.append("inference")
                raise AssertionError("routing alone cannot admit invocation")

        def prepare(value, session_hash, *, transport_session_id):
            assert value == run_id
            assert transport_session_id == sdk_session == sdk_context.get_session_id()
            assert session_hash == hashlib.sha256(sdk_session.encode("utf-8")).hexdigest()
            return context

        def read_status(payload):
            assert payload == request
            calls.append("status_read")
            return pending

        patch.setattr(runtime, "_runtime_prompt_policy", lambda: policy)
        patch.setattr(runtime, "_read_prompt_run", read_status)
        patch.setattr(checkpoint, "prepare_normal_run", prepare)
        patch.setattr(checkpoint, "activate_normal_run", declared_scope)
        patch.setattr(chat_engine, "ChatEngine", DeclaredEngine)

        async def read():
            events = []
            async for event in runtime._resume_normal_prompt(request):
                events.append(event)
            # The restore scope belongs to the worker thread; it is closed before the stream ends.
            assert active[0] is False
            return events

        events = asyncio.run(read())
        restoring = {"event": "resume_state", "state": "restoring", "run_id": run_id}
        if kind == "pending":
            assert events == [restoring, pending]
            assert calls == ["has_task", "checkpoint_load", "pending_snapshot_load", "status_read"]
        elif kind == "refused":
            # Neither pending nor a provable completion: nothing is claimed and nothing is loaded.
            assert events == [
                restoring,
                {"event": "prompt_run_status", "status": "unavailable", "run_id": run_id, "found": False},
            ]
            assert calls == ["has_task", "checkpoint_load"]
        else:
            # The declared engine has no continuation: a closed failure, never an unavailable success.
            assert events == [
                restoring,
                {
                    "event": "resume_state",
                    "state": "failed",
                    "run_id": run_id,
                    "diagnostic": {"restore_stage": "continuation", "exception_class": "AttributeError"},
                },
            ]
            assert calls == ["has_task", "checkpoint_load", "completion_claim"]
        assert "inference" not in calls and active[0] is False


@pytest.mark.parametrize("kind", ["pending", "completed"])
def test_normal_resume_routes_completion_before_any_pending_snapshot_load(
    monkeypatch, kind, managed_report_context
):
    statuses = ["PENDING", "RUNNING"] if kind == "pending" else ["trained"]
    for status in statuses:
        _routing_case(monkeypatch, kind, status, managed_report_context)
