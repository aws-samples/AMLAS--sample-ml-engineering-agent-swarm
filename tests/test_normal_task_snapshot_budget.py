"""Normal SFT capture must budget exact SDK history without duplicating it."""

from __future__ import annotations

import copy
import json

import pytest
from strands import Agent, Snapshot
from test_normal_run_checkpoint import normal_work as normal_work
from test_normal_sft_capture_diagnostics import (
    _context_and_snapshot,
)
from test_normal_sft_capture_diagnostics import (
    qualified_sft_catalog as qualified_sft_catalog,
)

from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat.tool_binding_backend import _record_session_snapshot


def _bytes(value):
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _original_sdk_case(work, note_bytes=0):
    context, prefix, _, model = _context_and_snapshot(work)
    prefix = copy.deepcopy(prefix)
    if note_bytes:
        prefix[0]["content"][0]["text"] += "n" * note_bytes
    agent = Agent(model=model, messages=copy.deepcopy(prefix))
    snapshot = _record_session_snapshot(agent)
    uses = [item["toolUse"] for item in prefix[-1]["content"] if "toolUse" in item]
    assert len(uses) == 1 and uses[0]["name"] == "submit_qwen_sft"
    return context, prefix, snapshot, model, uses[0]["toolUseId"]


def _capture_and_restore(case):
    context, prefix, snapshot, model, identifier = case
    before = copy.deepcopy(snapshot)
    phases = []
    with checkpoint.activate_normal_run(context):
        context.capture_sft_call(
            identifier,
            prefix,
            agent_snapshot_dict=snapshot,
            _diagnostic_phase=phases.append,
        )
        retained = context._captured_sft_snapshot(identifier, prefix)
    assert retained == before
    assert snapshot == before
    restored = Agent(model=model)
    restored.load_snapshot(Snapshot.from_dict(retained))
    assert _record_session_snapshot(restored)["data"] == before["data"]
    assert checkpoint._captured_snapshot_messages(retained) == prefix
    assert "snapshot_budget" in phases


def test_small_original_sdk_capture_still_restores_exact_state(
    qualified_sft_catalog,
):
    _capture_and_restore(_original_sdk_case(qualified_sft_catalog))


def test_individually_bounded_original_history_captures_and_restores_once(
    qualified_sft_catalog,
):
    case = _original_sdk_case(qualified_sft_catalog, note_bytes=96 * 1024)
    _, prefix, snapshot, _, identifier = case
    call = checkpoint._causal_call(identifier, prefix)
    assert _bytes(call) < checkpoint._MAX_BYTES
    assert _bytes(snapshot) < checkpoint._MAX_BYTES
    assert checkpoint._captured_snapshot_messages(snapshot) == prefix
    # SDK messages and the causal prefix are the same exact history. Their
    # duplicate alone exceeds the existing TASK cap before reference overhead.
    assert _bytes(snapshot) + _bytes(prefix) > checkpoint._MAX_BYTES
    assert checkpoint._canonical(call)
    assert checkpoint._canonical(snapshot)
    _capture_and_restore(case)


def test_oversized_individual_sdk_history_remains_closed(
    qualified_sft_catalog,
):
    _, _, snapshot, _, _ = _original_sdk_case(qualified_sft_catalog, note_bytes=checkpoint._MAX_BYTES + 1)
    assert _bytes(snapshot) > checkpoint._MAX_BYTES
    with pytest.raises(ValueError, match="normal run DATA size exceeded"):
        checkpoint._canonical(snapshot)


def _wire(case, version=2):
    context, prefix, snapshot, _, identifier = case
    value = {
        "schema_version": version,
        "logical_run_id": context.run_id,
        "private_store_correlation": "owned-original",
        "native_job_name": "owned-original",
        "native_job_arn": "owned-original",
        "operation_digest": "0" * 64,
        "source_refs": {},
        "causal_parent_ref": {},
        "sdk_snapshot": copy.deepcopy(snapshot),
        "pending_tool_use_id": identifier,
    }
    if version == 1:
        value["causal_prefix"] = copy.deepcopy(prefix)
    return value


@pytest.mark.parametrize("version", [1, 2])
def test_task_history_preserves_wire_identity_and_returns_an_independent_copy(
    qualified_sft_catalog,
    version,
):
    case = _original_sdk_case(qualified_sft_catalog)
    wire = _wire(case, version)
    before = copy.deepcopy(wire)
    before_bytes = checkpoint._canonical(wire)
    assert checkpoint._task_envelope(wire) is wire
    history = checkpoint._task_causal_prefix(wire)
    assert history == case[1]
    history[0]["content"][0]["text"] += " independent local copy"
    assert wire == before
    assert checkpoint._canonical(wire) == before_bytes
    assert ("causal_prefix" in wire) is (version == 1)


@pytest.mark.parametrize("version", [True, False, "1", "2", None, 0, 3])
def test_task_wire_rejects_nonexact_or_unknown_schema(
    qualified_sft_catalog,
    version,
):
    wire = _wire(_original_sdk_case(qualified_sft_catalog))
    wire["schema_version"] = version
    with pytest.raises(ValueError):
        checkpoint._task_envelope(wire)


@pytest.mark.parametrize("mode", ["compact_prefix", "extra", "legacy_missing"])
def test_task_wire_has_one_exact_versioned_inventory(
    qualified_sft_catalog,
    mode,
):
    wire = _wire(_original_sdk_case(qualified_sft_catalog), 1 if mode == "legacy_missing" else 2)
    if mode == "compact_prefix":
        wire["causal_prefix"] = copy.deepcopy(wire["sdk_snapshot"]["data"]["messages"])
    elif mode == "extra":
        wire["unowned_field"] = "ordinary fixture data"
    else:
        wire.pop("causal_prefix")
    with pytest.raises(ValueError):
        checkpoint._task_envelope(wire)


@pytest.mark.parametrize("mode", ["order", "typed_selection"])
def test_legacy_prefix_cannot_differ_from_exact_sdk_history(
    qualified_sft_catalog,
    mode,
):
    wire = _wire(_original_sdk_case(qualified_sft_catalog), 1)
    if mode == "order":
        wire["causal_prefix"].reverse()
    else:
        sdk_input = wire["sdk_snapshot"]["data"]["messages"][-1]["content"][0]["toolUse"]["input"]
        old_input = wire["causal_prefix"][-1]["content"][0]["toolUse"]["input"]
        sdk_input["plan_only"] = False
        old_input["plan_only"] = 0
        assert wire["causal_prefix"] == wire["sdk_snapshot"]["data"]["messages"]
    with pytest.raises(ValueError, match="legacy TASK snapshot prefix differs"):
        checkpoint._task_causal_prefix(wire)


@pytest.mark.parametrize("mode", ["state_missing", "extra_state", "scope"])
def test_compact_history_requires_the_strict_public_sdk_inventory(
    qualified_sft_catalog,
    mode,
):
    wire = _wire(_original_sdk_case(qualified_sft_catalog))
    if mode == "state_missing":
        wire["sdk_snapshot"]["data"].pop("state")
    elif mode == "extra_state":
        wire["sdk_snapshot"]["data"]["unowned_field"] = "ordinary fixture data"
    else:
        wire["sdk_snapshot"]["scope"] = "unowned"
    with pytest.raises(ValueError):
        checkpoint._task_causal_prefix(wire)


def test_compact_task_still_refuses_oversized_full_wire_before_projection(
    qualified_sft_catalog,
):
    wire = _wire(_original_sdk_case(qualified_sft_catalog))
    wire["private_store_correlation"] = "n" * checkpoint._MAX_BYTES
    assert _bytes(wire) > checkpoint._MAX_BYTES
    with pytest.raises(ValueError, match="normal run DATA size exceeded"):
        checkpoint._task_causal_prefix(wire)


def test_legacy_prepared_writer_and_admission_keep_identical_native_wire(
    monkeypatch,
    tmp_path,
):
    import asyncio

    from normal_sft_cold_fixture import close, original_precommit, setup

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        actual_version = checkpoint._prepared_task_wire_version
        actual_put = checkpoint._put_object
        published = []
        verified_legacy_versions = []

        def prepare_legacy(context, binding, record, qualify):
            if record.get("prepared_checkpoint_ref") is None:
                return 1
            version = actual_version(context, binding, record, qualify)
            verified_legacy_versions.append(version)
            return version

        def retain_original_publication(context, binding, raw, kind, qualify):
            ref = actual_put(context, binding, raw, kind, qualify)
            if kind in {"prepared-tasks", "tasks"}:
                published.append((kind, raw, ref))
            return ref

        monkeypatch.setattr(checkpoint, "_prepared_task_wire_version", prepare_legacy)
        monkeypatch.setattr(checkpoint, "_put_object", retain_original_publication)
        try:
            await original_precommit(state, monkeypatch)
            prepared = [item for item in published if item[0] == "prepared-tasks"]
            admitted = [item for item in published if item[0] == "tasks"]
            assert len(prepared) == len(admitted) == 1
            prepared_raw, prepared_ref = prepared[0][1:]
            admitted_raw, _ = admitted[0][1:]
            assert prepared_raw == admitted_raw
            before = copy.deepcopy(prepared_ref)
            prepared_wire = json.loads(prepared_raw)
            admitted_wire = json.loads(admitted_raw)
            assert prepared_wire["schema_version"] == 1
            assert admitted_wire["schema_version"] == 1
            assert checkpoint._task_causal_prefix(prepared_wire) == checkpoint._task_causal_prefix(
                admitted_wire
            )
            assert verified_legacy_versions and set(verified_legacy_versions) == {1}
            assert prepared_ref == before
            assert len(state.native.calls) == 1
        finally:
            await close(state)

    asyncio.run(scenario())


def test_legacy_task_only_republish_keeps_original_wire_and_effect_count(monkeypatch, tmp_path):
    import asyncio
    import hashlib

    from normal_sft_cold_fixture import close, original_precommit, setup

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        actual_version = checkpoint._prepared_task_wire_version
        actual_put = checkpoint._put_object
        actual_replace = checkpoint._replace_record
        captured = {}
        publications = []

        def prepare_legacy(context, binding, record, qualify):
            # Emulate only the legacy producer's selected wire version.
            if record.get("prepared_checkpoint_ref") is None:
                return 1
            return actual_version(context, binding, record, qualify)

        def retain_publication(context, binding, raw, kind, qualify):
            ref = actual_put(context, binding, raw, kind, qualify)
            if kind == "tasks":
                publications.append((raw, ref))
                captured.update(context=context, binding=binding)
            return ref

        def retain_policy(context, client, table_id, qualify, item, record, *, policy):
            result = actual_replace(context, client, table_id, qualify, item, record, policy=policy)
            if record.get("task_checkpoint_ref") is not None:
                captured["policy"] = policy
            return result

        monkeypatch.setattr(checkpoint, "_prepared_task_wire_version", prepare_legacy)
        monkeypatch.setattr(checkpoint, "_put_object", retain_publication)
        monkeypatch.setattr(checkpoint, "_replace_record", retain_policy)
        try:
            await original_precommit(state, monkeypatch)
            monkeypatch.setattr(checkpoint, "_prepared_task_wire_version", actual_version)
            assert len(publications) == 1
            raw, ref = publications[0]
            task = checkpoint._task_envelope(json.loads(raw))
            assert type(task["schema_version"]) is int and task["schema_version"] == 1
            assert len(task) == 11
            prefix = checkpoint._task_causal_prefix(task)
            context = captured["context"]
            policy = captured["policy"]
            with checkpoint.activate_normal_run(context):
                assert checkpoint.current_normal_run() is context
                with checkpoint._native(context) as (client, table_id, qualify, _sts):
                    item, record = checkpoint._load(context, client, table_id, qualify)
                    assert "prepared_checkpoint_ref" in record and "task_checkpoint_ref" in record
                    task_only = copy.deepcopy(record)
                    del task_only["prepared_checkpoint_ref"]
                    actual_replace(context, client, table_id, qualify, item, task_only, policy=policy)
                    _item, current = checkpoint._load(context, client, table_id, qualify)
                    assert current == task_only
                    assert current["task_checkpoint_ref"] == record["task_checkpoint_ref"]
                    assert current["task_checkpoint_ref"]["sha256"] == hashlib.sha256(raw).hexdigest()
                    assert current["task_checkpoint_ref"]["byte_count"] == len(raw)
                    assert len(state.native.calls) == 1
                    context.save_admitted_task(
                        copy.deepcopy(task["sdk_snapshot"]),
                        task["pending_tool_use_id"],
                        prefix,
                        original_job_name=task["native_job_name"],
                        policy=policy,
                        _expected_record=current,
                    )
                    _item, after = checkpoint._load(context, client, table_id, qualify)
                    assert after == current
                    assert checkpoint._read_object(context, captured["binding"], ref, "tasks", qualify) == raw
                    assert publications == [(raw, ref)]
                    assert len(state.native.calls) == 1
        finally:
            await close(state)

    asyncio.run(scenario())
