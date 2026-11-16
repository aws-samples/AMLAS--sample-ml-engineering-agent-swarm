"""Accepted TASK publication from one fresh PREPARED record, not cold recovery.

Actual normal SDK capture, prepared CAS, accepted SFT journal and current native
provider checks remain. The only fault is an inert external TASK-reference CAS
refusal. Fresh server policy is not read from stored DATA. Worker cleanup is not
physical loss or native cancellation; no hydration/inference/wake/O4 is proven.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import threading

import pytest
from bedrock_agentcore.runtime import BedrockAgentCoreContext
from botocore.exceptions import ClientError
from normal_sft_cold_fixture import SFT_USE, close, prepare_original_submission, setup
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.normal_run_checkpoint import (
    _task_causal_prefix,
    activate_normal_run,
    prepare_normal_run,
)

# Upper bound for waits on a boundary that the run must reach. A slow machine
# needs the room; a boundary that is never reached still fails when it ends.
LIVENESS_WAIT_SECONDS = 180

TASK_KEYS = {
    "schema_version",
    "logical_run_id",
    "private_store_correlation",
    "native_job_name",
    "native_job_arn",
    "operation_digest",
    "source_refs",
    "causal_parent_ref",
    "sdk_snapshot",
    "pending_tool_use_id",
}


def _exact_object(state, descriptor):
    assert set(descriptor) == {"uri", "version_id", "sha256", "byte_count", "format"}
    assert descriptor["format"] == "json" and descriptor["version_id"]
    assert type(descriptor["byte_count"]) is int and 0 < descriptor["byte_count"] <= 160 * 1024
    key = descriptor["uri"].split("/", 3)[3]
    raw = state.objects.objects.get((key, descriptor["version_id"]))
    if raw is None:
        raw = state.ports.objects.get((key, descriptor["version_id"]))
    assert type(raw) is bytes and len(raw) == descriptor["byte_count"]
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    return raw


def _promotion_s3_fidelity(monkeypatch, state, row):
    """Native collision/latest semantics for already Source-created TASK keys."""
    objects = state.objects
    actual_put, actual_get = objects.put_object, objects.get_object
    prepared_uri = row["prepared_checkpoint_ref"]["uri"]
    bucket = prepared_uri.split("/", 3)[2]
    prefix = (
        "sessions/" + row["private_store_correlation"] + "/prompt-runs/" + row["logical_run_id"] + "/tasks/"
    )
    existing = {key: version for key, version in objects.versions.items() if key.startswith(prefix)}

    def put(**values):
        key = values.get("Key")
        if key not in existing or values.get("IfNoneMatch") != "*":
            return actual_put(**values)
        objects.check("PutObject", values)
        assert values["Bucket"] == bucket
        assert values.get("ExpectedBucketOwner") == row["scope"]["account_id"]
        raise ClientError(
            {"Error": {"Code": "PreconditionFailed"}, "ResponseMetadata": {"HTTPStatusCode": 412}},
            "PutObject",
        )

    def get(**values):
        key = values.get("Key")
        if key not in existing or values.get("VersionId") is not None:
            return actual_get(**values)
        objects.check("GetObject", values)
        assert values["Bucket"] == bucket
        assert values.get("ExpectedBucketOwner") == row["scope"]["account_id"]
        version = objects.versions[key]
        assert version == existing[key]
        raw = objects.objects[(key, version)]
        assert type(raw) is bytes
        return {"VersionId": version, "ContentLength": len(raw), "Body": io.BytesIO(raw)}

    monkeypatch.setattr(objects, "put_object", put)
    monkeypatch.setattr(objects, "get_object", get)


def _settled_prepared_record(state, key, before):
    """Pin promotion's exact prestate after the original report worker settles."""
    with state.ports.lock:
        settled = copy.deepcopy(state.ports.items[key])
    prior_row, settled_row = json.loads(before["record"]["S"]), json.loads(settled["record"]["S"])
    assert settled == {
        **before,
        "record": settled["record"],
        "record_sha256": settled["record_sha256"],
    }
    assert settled_row == prior_row or (
        "report_checkpoint_ref" not in prior_row
        and "report_checkpoint_ref" in settled_row
        and settled_row == {**prior_row, "report_checkpoint_ref": settled_row["report_checkpoint_ref"]}
    ), "original cleanup changed more than its optional report pointer"
    return settled, settled_row


def test_accepted_original_promotes_exact_prepared_task_idempotently_without_dispatch(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        refused = threading.Event()
        task_cas_attempts = []
        actual_update = state.ports.update_item

        def refuse_task_cas(**values):
            raw = values.get("ExpressionAttributeValues", {}).get(":body", {}).get("S")
            if (
                values["Key"]["job_name"]["S"].startswith("PROMPT_RUN#")
                and type(raw) is str
                and "task_checkpoint_ref" in json.loads(raw)
            ):
                state.ports._ddb("UpdateItem", values)
                task_cas_attempts.append(copy.deepcopy(values))
                refused.set()
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
            return actual_update(**values)

        monkeypatch.setattr(state.ports, "update_item", refuse_task_cas)
        try:
            reader, events, backend = await prepare_original_submission(state, monkeypatch)
            deadline = asyncio.get_running_loop().time() + LIVENESS_WAIT_SECONDS
            while not refused.is_set() and not reader.done() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.005)
            assert refused.is_set(), "actual accepted-TASK CAS boundary was never reached"
            assert len(task_cas_attempts) == 1 and len(state.native.calls) == 1
            assert state.native.stops == [] and not state.task_published.is_set()
            original = next(iter(backend._sft_store._records.values()))
            assert original.extra["submission_state"] == "accepted"
            # Settle original report publication before reading the exact journal.
            # The accepted native original is never stopped.
            await close_original_worker(state.work, reader)
            accepted = backend._sft_store._campaign_state.load_original(
                backend._sft_store._execution_binding, original.id
            )
            assert accepted.disposition == "ACCEPTED" and accepted.native_job_name == original.id
            assert any(
                use == SFT_USE and answer.get("accepted") is True for use, answer in state.work.originals
            )
            run_id = next(event["run_id"] for event in events if event["event"] == "run_started")
            key = "PROMPT_RUN#" + run_id
            with state.ports.lock:
                retained = copy.deepcopy(state.ports.items[key])
            row = json.loads(retained["record"]["S"])
            assert "prepared_checkpoint_ref" in row and "task_checkpoint_ref" not in row
            prepared_ref = copy.deepcopy(row["prepared_checkpoint_ref"])
            prepared_raw = _exact_object(state, prepared_ref)
            prepared = json.loads(prepared_raw)
            assert prepared["schema_version"] == 2
            assert set(prepared) == TASK_KEYS
            assert prepared["logical_run_id"] == run_id
            assert prepared["native_job_name"] == original.id
            assert prepared["native_job_arn"] == accepted.native_job_arn
            assert prepared["operation_digest"] == accepted.operation_digest
            assert prepared["source_refs"] == list(accepted.intent.source_refs)
            causal_parent = json.loads(_exact_object(state, prepared["causal_parent_ref"]))
            assert causal_parent["pending_tool_use_id"] == prepared["pending_tool_use_id"] == SFT_USE
            assert (
                prepared["sdk_snapshot"]["data"]["messages"]
                == _task_causal_prefix(prepared)
                == causal_parent["causal_prefix"]
            )

            assert state.work.owned_app.get_async_task_info()["active_count"] == 0
            retained, row = _settled_prepared_record(state, key, retained)
            monkeypatch.setattr(state.ports, "update_item", actual_update)
            _promotion_s3_fidelity(monkeypatch, state, row)
            transport = BedrockAgentCoreContext.get_session_id()
            context = prepare_normal_run(
                run_id, hashlib.sha256(transport.encode("utf-8")).hexdigest(), transport_session_id=transport
            )
            policy = runtime._runtime_prompt_policy()
            assert type(policy) is ApprovalPolicy and policy.mode is ApprovalMode.ALLOWLIST
            assert "qwen_sft_flow" in policy.allowlist and policy.interactive is False
            model_calls = state.model.calls
            native_reads = len(state.native.reads)
            source_calls = len(state.work.seen_prompts)
            with activate_normal_run(context):
                context.promote_prepared_task(policy=policy)
            with state.ports.lock:
                promoted = copy.deepcopy(state.ports.items[key])
            promoted_row = json.loads(promoted["record"]["S"])
            assert promoted_row == {**row, "task_checkpoint_ref": promoted_row["task_checkpoint_ref"]}
            task_ref = promoted_row["task_checkpoint_ref"]
            assert "/tasks/" in task_ref["uri"] and "/prepared-tasks/" in prepared_ref["uri"]
            assert task_ref["version_id"] and _exact_object(state, task_ref) == prepared_raw
            assert _exact_object(state, prepared_ref) == prepared_raw
            assert original.id in state.native.reads[native_reads:], (
                "promotion must freshly observe its native original"
            )
            assert context.run_id == run_id and context.transport_session_id == transport
            assert context.session_hash == hashlib.sha256(transport.encode("utf-8")).hexdigest()

            writes = sum(
                operation in ("PutItem", "UpdateItem", "PutObject") for operation, _ in state.ports.calls
            )
            object_count = len(state.objects.objects)
            with activate_normal_run(context):
                context.promote_prepared_task(policy=policy)
            with state.ports.lock:
                assert state.ports.items[key] == promoted
            assert (
                sum(operation in ("PutItem", "UpdateItem", "PutObject") for operation, _ in state.ports.calls)
                == writes
            )
            assert len(state.objects.objects) == object_count
            assert state.model.calls == model_calls and len(state.work.seen_prompts) == source_calls == 1
            assert len(state.native.calls) == 1 and state.native.stops == []
            assert state.loaded == [] and state.invoked == []
        finally:
            await close(state)

    asyncio.run(scenario())


async def _fresh_unpromoted_original(state, monkeypatch):
    """Same real fresh prestate; external TASK CAS denial supplies no authority."""
    refused = threading.Event()
    actual_update = state.ports.update_item

    def refuse_task_cas(**values):
        raw = values.get("ExpressionAttributeValues", {}).get(":body", {}).get("S")
        if (
            values["Key"]["job_name"]["S"].startswith("PROMPT_RUN#")
            and type(raw) is str
            and "task_checkpoint_ref" in json.loads(raw)
        ):
            state.ports._ddb("UpdateItem", values)
            refused.set()
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
        return actual_update(**values)

    monkeypatch.setattr(state.ports, "update_item", refuse_task_cas)
    reader, events, backend = await prepare_original_submission(state, monkeypatch)
    deadline = asyncio.get_running_loop().time() + LIVENESS_WAIT_SECONDS
    while not refused.is_set() and not reader.done() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.005)
    assert refused.is_set(), "fresh accepted-TASK CAS prerequisite was never reached"
    assert len(state.native.calls) == 1 and state.native.stops == []
    original = next(iter(backend._sft_store._records.values()))
    # Settle original report publication before reading the exact journal.
    await close_original_worker(state.work, reader)
    accepted = backend._sft_store._campaign_state.load_original(
        backend._sft_store._execution_binding, original.id
    )
    assert original.extra["submission_state"] == "accepted" and accepted.disposition == "ACCEPTED"
    run_id = next(event["run_id"] for event in events if event["event"] == "run_started")
    key = "PROMPT_RUN#" + run_id
    with state.ports.lock:
        before = copy.deepcopy(state.ports.items[key])
    row = json.loads(before["record"]["S"])
    assert "prepared_checkpoint_ref" in row and "task_checkpoint_ref" not in row
    prepared_raw = _exact_object(state, row["prepared_checkpoint_ref"])
    assert json.loads(prepared_raw)["operation_digest"] == accepted.operation_digest
    assert state.work.owned_app.get_async_task_info()["active_count"] == 0
    before, row = _settled_prepared_record(state, key, before)
    monkeypatch.setattr(state.ports, "update_item", actual_update)
    _promotion_s3_fidelity(monkeypatch, state, row)
    transport = BedrockAgentCoreContext.get_session_id()
    context = prepare_normal_run(
        run_id, hashlib.sha256(transport.encode()).hexdigest(), transport_session_id=transport
    )
    policy = runtime._runtime_prompt_policy()
    assert type(policy) is ApprovalPolicy and policy.mode is ApprovalMode.ALLOWLIST
    assert "qwen_sft_flow" in policy.allowlist and policy.interactive is False
    return context, policy, original, key, before, prepared_raw


@pytest.mark.parametrize("native_observation", ["Stopped", "UNKNOWN"])
def test_prepared_promotion_refuses_stopped_or_unknown_current_native_original(
    monkeypatch, tmp_path, native_observation
):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            context, policy, original, key, before, prepared_raw = await _fresh_unpromoted_original(
                state, monkeypatch
            )
            actual_describe = state.native.describe_training_job
            observed = []

            def describe(**values):
                state.native.check("DescribeTrainingJob", values)
                if values["TrainingJobName"] == original.id:
                    observed.append(copy.deepcopy(values))
                    if native_observation == "UNKNOWN":
                        raise TimeoutError("Declared current native observation unavailable.")
                return actual_describe(**values)

            if native_observation == "Stopped":
                state.native.states[original.id] = "Stopped"
            monkeypatch.setattr(state.native, "describe_training_job", describe)
            calls, object_count = state.model.calls, len(state.objects.objects)
            with activate_normal_run(context), pytest.raises(ValueError):
                context.promote_prepared_task(policy=policy)
            assert observed, "promotion did not reach current native observation"
            with state.ports.lock:
                assert state.ports.items[key] == before
            assert "task_checkpoint_ref" not in json.loads(before["record"]["S"])
            assert (
                _exact_object(state, json.loads(before["record"]["S"])["prepared_checkpoint_ref"])
                == prepared_raw
            )
            assert len(state.objects.objects) == object_count
            assert state.model.calls == calls and len(state.work.seen_prompts) == 1
            assert len(state.native.calls) == 1 and state.native.stops == []
            assert not state.task_published.is_set() and state.loaded == [] and state.invoked == []
        finally:
            await close(state)

    asyncio.run(scenario())


def test_prepared_promotion_cannot_overwrite_current_row_changed_at_task_cas(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            context, policy, original, key, before, prepared_raw = await _fresh_unpromoted_original(
                state, monkeypatch
            )
            actual_update = state.ports.update_item
            changed_items, attempts = [], []

            def change_before_cas(**values):
                raw = values.get("ExpressionAttributeValues", {}).get(":body", {}).get("S")
                if (
                    values["Key"] == {"job_name": before["job_name"]}
                    and type(raw) is str
                    and "task_checkpoint_ref" in json.loads(raw)
                ):
                    attempts.append(copy.deepcopy(values))
                    changed = copy.deepcopy(before)
                    body = json.loads(changed["record"]["S"])
                    body["admitted_prompt"] += " Declared concurrent row revision."
                    changed["record"]["S"] = json.dumps(
                        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
                    )
                    changed["record_sha256"] = {
                        "S": hashlib.sha256(changed["record"]["S"].encode()).hexdigest()
                    }
                    with state.ports.lock:
                        state.ports.items[key] = changed
                    changed_items.append(copy.deepcopy(changed))
                return actual_update(**values)

            monkeypatch.setattr(state.ports, "update_item", change_before_cas)
            calls = state.model.calls
            native_read_count = len(state.native.reads)
            with activate_normal_run(context), pytest.raises(ValueError):
                context.promote_prepared_task(policy=policy)
            assert len(attempts) == len(changed_items) == 1, (
                "current-row TASK CAS race was not reached exactly once"
            )
            assert original.id in state.native.reads[native_read_count:]
            with state.ports.lock:
                assert state.ports.items[key] == changed_items[0]
            changed = json.loads(changed_items[0]["record"]["S"])
            assert "task_checkpoint_ref" not in changed
            assert _exact_object(state, changed["prepared_checkpoint_ref"]) == prepared_raw
            assert state.model.calls == calls and len(state.work.seen_prompts) == 1
            assert len(state.native.calls) == 1 and state.native.stops == []
            assert not state.task_published.is_set() and state.loaded == [] and state.invoked == []
        finally:
            await close(state)

    asyncio.run(scenario())


def test_save_admitted_task_cannot_adopt_row_replacement_against_expected_prior(monkeypatch, tmp_path):
    """Expected-record DATA pins one fresh caller view; it is not a grant."""

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            context, policy, original, key, before, prepared_raw = await _fresh_unpromoted_original(
                state, monkeypatch
            )
            prior = json.loads(before["record"]["S"])
            prepared = json.loads(prepared_raw)
            replacement = copy.deepcopy(before)
            changed = copy.deepcopy(prior)
            changed["admitted_prompt"] += " Declared concurrent replacement."
            raw = json.dumps(
                changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            replacement["record"] = {"S": raw}
            replacement["record_sha256"] = {"S": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
            with state.ports.lock:
                state.ports.items[key] = copy.deepcopy(replacement)
            writes = sum(
                operation in ("PutItem", "UpdateItem", "PutObject") for operation, _ in state.ports.calls
            )
            object_writes = sum(
                operation in ("PutObject", "CopyObject", "CreateMultipartUpload", "CompleteMultipartUpload")
                for operation, _ in state.objects.calls
            )
            calls = state.model.calls
            with activate_normal_run(context), pytest.raises(ValueError):
                context.save_admitted_task(
                    copy.deepcopy(prepared["sdk_snapshot"]),
                    prepared["pending_tool_use_id"],
                    _task_causal_prefix(prepared),
                    original_job_name=original.id,
                    policy=policy,
                    _expected_record=copy.deepcopy(prior),
                )
            with state.ports.lock:
                assert state.ports.items[key] == replacement
            assert (
                sum(operation in ("PutItem", "UpdateItem", "PutObject") for operation, _ in state.ports.calls)
                == writes
            )
            assert (
                sum(
                    operation
                    in ("PutObject", "CopyObject", "CreateMultipartUpload", "CompleteMultipartUpload")
                    for operation, _ in state.objects.calls
                )
                == object_writes
            )
            assert "task_checkpoint_ref" not in changed
            assert _exact_object(state, changed["prepared_checkpoint_ref"]) == prepared_raw
            assert state.model.calls == calls and len(state.work.seen_prompts) == 1
            assert len(state.native.calls) == 1 and state.native.stops == []
            assert not state.task_published.is_set() and state.loaded == [] and state.invoked == []
        finally:
            await close(state)

    asyncio.run(scenario())
