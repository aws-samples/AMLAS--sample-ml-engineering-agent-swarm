"""Pending recovery refusal controls; mocked upstream DATA, actual normal SDK SFT.

An original must be submitted through the real SDK/tool/SessionSftStore/Dynamo
path before any negative is prepared. Only copies of its actual persisted DATA
are changed. Current policy comes from trusted code, never a row or request.
SDK session is correlation, not authenticated human/tenant identity. These
component controls do not prove native IAM, genuine upstreams, actual controlled
worker loss, automatic wake, continuation or full R1-R4/C1-C4 acceptance.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from test_normal_run_checkpoint import assert_safe_data, decoded, unavailable
from test_normal_run_sft_checkpoint import (
    cloud_snapshot,
    configured_work,
    pending_progress,
    start_original,
    wire_decode,
)
from test_normal_run_sft_checkpoint import (
    original_sft as original_sft,
)
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup

from amlas import app as runtime
from amlas.backends.adapters.campaign_state import _STATUS
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.campaign_state import _encode_private_record


@pytest.fixture
def captured_original(original_sft):
    """No row seeding: export after the original worker/report cleanup settles."""
    value = original_sft

    async def capture():
        reader = None
        try:
            reader, original, run_id = await start_original(value)
            name = value.native.calls[0]["TrainingJobName"]
            store = value.work.engines[0].tool_backend._sft_store
            assert store._campaign_state is not None, "original normal SFT has no Dynamo journal"
            record = store._campaign_state.load_original(store._execution_binding, name)
            assert record.disposition == "ACCEPTED" and record.native_job_name == name
            await close_original_worker(value.work, reader)
            reader = None
            data = wire_decode(cloud_snapshot(value, run_id))
            assert data["ddb"][name]["private_record"]["B"] == _encode_private_record(record)
            item = data["ddb"]["PROMPT_RUN#" + run_id]
            companion = json.loads(item["record"]["S"])
            assert companion["stage"] == "sft" and companion["sft"]["native_job_name"] == name
            assert original["job_name"] == name and len(data["training_requests"]) == 1
            return SimpleNamespace(data=data, record=record, name=name, run_id=run_id)
        finally:
            if reader is not None:
                await close_original_worker(value.work, reader)

    return asyncio.run(capture())


def fresh_read(snapshot, monkeypatch, directory, *, action="resume_prompt_run", policy=None, prepare=None):
    """Fresh configured normal route; captured native records are DATA only."""
    directory.mkdir()
    context = None
    data = copy.deepcopy(snapshot.data)
    try:
        inert_runtime_setup.__wrapped__(monkeypatch)
        context, value = configured_work(monkeypatch, directory, captured=data)
        assert len(value.native.calls) == 1
        assert value.s3.mutations == value.s3.staging_attempts == []
        assert_safe_data(decoded(value.ports.items))
        if policy is not None:
            monkeypatch.setattr(runtime, "_runtime_prompt_policy", lambda: policy)
        markers = {} if prepare is None else prepare(value, monkeypatch)

        async def forbidden_model(*args, **kwargs):
            value.model.calls += 1
            raise AssertionError("closed recovery attempted model dispatch")
            yield None

        monkeypatch.setattr(value.model, "stream", forbidden_model)

        async def read():
            return [
                event
                async for event in runtime.agent_invocation({"action": action, "run_id": snapshot.run_id})
            ]

        events = asyncio.run(read())
        if action == "resume_prompt_run":
            # The resume action answers its state before any restore work; the projection under test follows.
            assert events[:1] == [{"event": "resume_state", "state": "restoring", "run_id": snapshot.run_id}]
            events = events[1:]
        assert len(value.native.calls) == 1 and value.native.stops == []
        assert value.model.calls == 0
        assert value.s3.mutations == value.s3.staging_attempts == []
        assert "PutObject" not in [name for name, _args in value.ports.calls]
        assert value.work.owned_app.get_async_task_info()["active_count"] == 0
        return events, markers
    finally:
        if context is not None:
            context.close()


def assert_available(snapshot, monkeypatch, tmp_path):
    """The untouched actual original must work; unconditional refusal is no oracle."""
    with monkeypatch.context() as control:
        events, _markers = fresh_read(snapshot, control, tmp_path / "available-control")
    assert events == [pending_progress(snapshot.run_id, snapshot.record.native_job_arn)]


def failed_prepare(run_id):
    """A refused restore ends as a closed failure at the first stage; it is never an unavailable success."""
    return {
        "event": "resume_state",
        "state": "failed",
        "run_id": run_id,
        "diagnostic": {"restore_stage": "run_prepare", "exception_class": "ValueError"},
    }


@pytest.mark.parametrize("kind", ["none", "ask", "missing_sft_action"])
def test_current_denied_policy_closes_pending_resume(captured_original, monkeypatch, tmp_path, kind):
    snapshot = captured_original
    assert_available(snapshot, monkeypatch, tmp_path)

    def prepare(_value, patch):
        if kind == "none":
            patch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", "[]")
        elif kind == "ask":
            denied = ApprovalPolicy(mode=ApprovalMode.ASK, interactive=False)
            patch.setattr(runtime, "_runtime_prompt_policy", lambda: denied)
        else:
            patch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job"]')
        return {}

    events, _markers = fresh_read(snapshot, monkeypatch, tmp_path / "denied-policy", prepare=prepare)
    assert events == [failed_prepare(snapshot.run_id)]


def test_foreign_actual_sdk_session_closes_without_existence_details(
    captured_original, monkeypatch, tmp_path
):
    snapshot = captured_original
    assert_available(snapshot, monkeypatch, tmp_path)

    def prepare(_value, _patch):
        BedrockAgentCoreContext.set_request_context("ordinary-foreign-request", "ordinary-foreign-session")
        return {}

    events, _markers = fresh_read(snapshot, monkeypatch, tmp_path / "foreign-session", prepare=prepare)
    assert events == [failed_prepare(snapshot.run_id)]


def replace_companion(snapshot, change):
    """Recompute its self-SHA deliberately; integrity cannot trust that hash alone."""
    snapshot = copy.copy(snapshot)
    snapshot.data = copy.deepcopy(snapshot.data)
    item = snapshot.data["ddb"]["PROMPT_RUN#" + snapshot.run_id]
    record = json.loads(item["record"]["S"])
    change(item, record)
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    item["record"], item["record_sha256"] = {"S": raw}, {"S": hashlib.sha256(raw.encode()).hexdigest()}
    assert_safe_data(record)
    return snapshot


@pytest.mark.parametrize(
    "kind", ["association", "model_alias", "preparation_alias", "intent", "job_name", "table_id", "role_id"]
)
def test_rehashed_association_or_scope_tamper_is_unavailable(captured_original, monkeypatch, tmp_path, kind):
    original = captured_original
    assert_available(original, monkeypatch, tmp_path)

    def change(item, record):
        if kind == "association":
            record["sft"]["native_job_arn"] = record["sft"]["native_job_arn"] + "-ordinary"
        elif kind == "model_alias":
            record["sft"]["model_ref"] = "model_acq_" + "e" * 32
        elif kind == "preparation_alias":
            record["sft"]["preparation_ref"] = "prep_" + "e" * 32
        elif kind == "intent":
            record["sft"]["intent"]["recipe_id"] = "fit_v1"
        elif kind == "job_name":
            item["job_name"] = {"S": "PROMPT_RUN#" + "e" * 32}
        elif kind == "table_id":
            record["table_id"] = "d2ad9dd0-d021-4cdc-bb58-53b0345f6f6c"
        else:
            record["scope"]["execution_role_id"] = "AROA" + "Z" * 17

    changed = replace_companion(original, change)
    events, _markers = fresh_read(
        changed, monkeypatch, tmp_path / "changed-companion", action="get_prompt_run_status"
    )
    assert events == [unavailable(original.run_id)]


@pytest.mark.parametrize("state", ["RESERVED", "CREATE_ATTEMPTED", "AMBIGUOUS", "REJECTED"])
def test_nonaccepted_journal_never_adopts_the_verified_native_original(
    captured_original, monkeypatch, tmp_path, state
):
    original = captured_original
    assert_available(original, monkeypatch, tmp_path)
    changed = copy.copy(original)
    changed.data = copy.deepcopy(original.data)
    row = changed.data["ddb"][changed.name]
    replacement = dataclasses.replace(original.record, disposition=state)
    row["private_record"] = {"B": _encode_private_record(replacement)}
    row["record_state"], row["status"] = {"S": state}, {"S": _STATUS[state]}
    assert replacement.native_job_name == original.record.native_job_name
    assert changed.data["training_states"][changed.name] == "InProgress"
    events, _markers = fresh_read(changed, monkeypatch, tmp_path / "unknown-journal")
    assert events == [unavailable(original.run_id)]


@pytest.mark.parametrize("kind", ["policy", "role_id", "table_id"])
def test_qualification_change_after_journal_read_never_exposes_pending_body(
    captured_original, monkeypatch, tmp_path, kind
):
    snapshot = captured_original
    assert_available(snapshot, monkeypatch, tmp_path)
    policy = ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST,
        allowlist=["run_processing_job", "qwen_sft_flow"],
        max_auto_cost=None,
        interactive=False,
    )

    def prepare(value, patch):
        state = {"journal_returned": False}
        original_get = value.ports.get_item
        original_identity, original_table = value.ports.get_caller_identity, value.ports.describe_table

        def get_item(**kwargs):
            result = original_get(**kwargs)
            if kwargs["Key"]["job_name"]["S"] == snapshot.name:
                state["journal_returned"] = True
                if kind == "policy":
                    policy.allowlist.clear()
            return result

        def identity(**kwargs):
            result = original_identity(**kwargs)
            if kind == "role_id" and state["journal_returned"]:
                result["UserId"] = "AROA" + "Z" * 17 + ":ordinary"
            return result

        def table(**kwargs):
            result = original_table(**kwargs)
            if kind == "table_id" and state["journal_returned"]:
                result["Table"]["TableId"] = "d2ad9dd0-d021-4cdc-bb58-53b0345f6f6c"
            return result

        patch.setattr(value.ports, "get_item", get_item)
        patch.setattr(value.ports, "get_caller_identity", identity)
        patch.setattr(value.ports, "describe_table", table)
        return state

    events, markers = fresh_read(
        snapshot, monkeypatch, tmp_path / "qualification-change", policy=policy, prepare=prepare
    )
    assert markers["journal_returned"], "qualification change did not reach the real read boundary"
    assert events == [unavailable(snapshot.run_id)]
