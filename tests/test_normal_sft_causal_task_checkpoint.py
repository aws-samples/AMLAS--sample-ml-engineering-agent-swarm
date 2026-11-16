"""Native-bound causal TASK ABI; real normal SDK originals, inert providers only.

No private authority/store/journal is seeded. Missing publication is an assertion
RED after genuine warm prerequisites, not an import/missing-method oracle.
Native IAM, real worker loss/wake and full ML acceptance remain separate.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json

from normal_sft_cold_fixture import (
    SFT_USE,
    close,
    cloud_data,
    original_precommit,
    payloads,
    setup,
    wait_for,
)
from test_normal_run_checkpoint import assert_safe_data
from test_normal_sft_cold_continuation import child
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.control_plane.campaign_state import (
    _intent_fields,
    build_sft_intent,
    operation_digest,
)
from amlas.control_plane.normal_run_checkpoint import _task_causal_prefix
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend, _record_messages

REF_KEYS = {"uri", "version_id", "sha256", "byte_count", "format"}
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


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def reference(value):
    if type(value) is dict and set(value) == {"type", "fields"}:
        assert value["type"] == "DatasetArtifactRef"
        value = value["fields"]
    assert type(value) is dict and set(value) == REF_KEYS
    assert value["format"] == "json" and value["version_id"] not in (None, "", "null")
    return value


def artifact(objects, checkpoints, descriptor):
    descriptor = reference(descriptor)
    key = descriptor["uri"].split("/", 3)[3]
    raw = checkpoints.get((key, descriptor["version_id"]))
    if raw is None:
        raw = objects.get((key, descriptor["version_id"]))
    assert type(raw) is bytes, "checkpoint descriptor has no exact published version"
    assert len(raw) == descriptor["byte_count"] and hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    result = json.loads(raw)
    assert canonical(result) == raw and type(result) is dict
    assert_safe_data(result)
    return result


def normal_row(state, rows=None):
    rows = state.ports.items if rows is None else rows
    return json.loads(rows["PROMPT_RUN#" + state.run_id]["record"]["S"])


def task_publication(state, rows=None, objects=None, checkpoints=None):
    record = normal_row(state, rows)
    assert "task_checkpoint_ref" in record, "admitted PRECOMMIT has no source-owned TASK checkpoint"
    descriptor = reference(record["task_checkpoint_ref"])
    task = artifact(
        state.objects.objects if objects is None else objects,
        state.ports.objects if checkpoints is None else checkpoints,
        descriptor,
    )
    assert set(task) == TASK_KEYS and task["schema_version"] == 2
    assert task["logical_run_id"] == state.run_id and task["pending_tool_use_id"] == SFT_USE
    assert task["private_store_correlation"] == state.backend._acquisition_store.runtime_session_id
    assert task["native_job_name"] == state.original.id and task["native_job_arn"] == state.original.arn
    assert task["operation_digest"] == state.original.extra["operation_digest"]
    store = state.backend._sft_store
    accepted = store._campaign_state.load_original(store._execution_binding, state.original.id)
    assert task["source_refs"] == list(accepted.intent.source_refs)
    assert reference(task["causal_parent_ref"]) == reference(
        _intent_fields(accepted.intent)["causal_parent_ref"]
    )
    prefix = _task_causal_prefix(task)
    assert type(task["sdk_snapshot"]) is dict and type(prefix) is list
    assert prefix == _record_messages(state.prefix)
    assert SFT_USE not in payloads(prefix)
    return descriptor, task


def rejection_assessor(state, guard, original_assess, mode, assessed):
    def assess(**kwargs):
        text = kwargs["content"][0]["text"]["text"]
        projection = {}
        if 0 < len(text) <= 256 * 1024:
            try:
                blocks = json.loads(text)
                if (
                    type(blocks) is list
                    and len(blocks) == 1
                    and type(blocks[0]) is dict
                    and type(blocks[0].get("text")) is str
                ):
                    projection = json.loads(blocks[0]["text"])
            except (TypeError, ValueError):
                pass
        if (
            type(projection) is dict
            and len(state.native.calls) == 1
            and projection.get("job_name") == state.native.calls[0]["TrainingJobName"]
            and projection.get("operation") == "train"
            and projection.get("recipe_id") == "full_v1"
            and projection.get("accepted") is True
            and projection.get("status") == "submitted"
        ):
            assessed.append(mode)
            if mode == "UNKNOWN":
                return {}
            original_deny = guard.deny
            try:
                guard.deny = lambda _text: True
                return original_assess(**kwargs)
            finally:
                guard.deny = original_deny
        return original_assess(**kwargs)

    return assess


def test_native_original_commits_prospective_parent_before_create_without_legacy_relabel(
    monkeypatch, tmp_path
):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        observed = []
        original_create = state.native.create_training_job

        def create(**kwargs):
            observed.append(
                (
                    copy.deepcopy(state.ports.items),
                    copy.deepcopy(state.objects.objects),
                    copy.deepcopy(state.ports.objects),
                )
            )
            return original_create(**kwargs)

        monkeypatch.setattr(state.native, "create_training_job", create)
        try:
            await original_precommit(state, monkeypatch)
            store = state.backend._sft_store
            accepted = store._campaign_state.load_original(store._execution_binding, state.original.id)
            legacy = build_sft_intent(
                accepted.binding,
                accepted.intent.catalog,
                accepted.intent.inputs,
                recipe_id=accepted.intent.recipe_id,
                source_refs=accepted.intent.source_refs,
            )
            assert "causal_parent_ref" not in _intent_fields(legacy)
            assert operation_digest(legacy) == hashlib.sha256(canonical(_intent_fields(legacy))).hexdigest()
            assert len(state.native.calls) == len(observed) == 1
            rows, objects, checkpoints = observed[0]
            raw = json.loads(rows[state.original.id]["private_record"]["B"])
            assert "causal_parent_ref" in raw["intent"], (
                "original native Create has no prospective independent parent commitment"
            )
            descriptor = reference(raw["intent"]["causal_parent_ref"])
            parent = artifact(objects, checkpoints, descriptor)
            assert parent["logical_run_id"] == state.run_id
            fields = {"binding": raw["binding"], **raw["intent"]}
            digest = hashlib.sha256(canonical(fields)).hexdigest()
            assert digest == raw["operation_digest"] == accepted.operation_digest
            prefix = accepted.binding.scope.deployment_prefix
            assert state.original.id == prefix + "-sft-" + digest[: 63 - len(prefix) - 5]
            changed = copy.deepcopy(fields)
            changed["causal_parent_ref"]["fields"]["version_id"] = "ordinary-different-version"
            assert hashlib.sha256(canonical(changed)).hexdigest() != digest
            assert operation_digest(legacy) != digest
        finally:
            await close(state)

    asyncio.run(scenario())


def test_admitted_precommit_saves_public_sdk_task_before_wait_and_denials_publish_none(monkeypatch, tmp_path):
    async def scenario():
        for mode in ("denied", "UNKNOWN"):
            with monkeypatch.context() as rejected:
                directory = tmp_path / mode
                directory.mkdir()
                state = setup(rejected, directory)
                guard = state.work.guard
                original_assess = guard.apply_guardrail
                assessed = []

                assess = rejection_assessor(state, guard, original_assess, mode, assessed)

                rejected.setattr(guard, "apply_guardrail", assess)
                try:
                    try:
                        await original_precommit(state, rejected)
                    except (AssertionError, json.JSONDecodeError):
                        assert assessed and len(state.native.calls) == 1
                    events = await asyncio.wait_for(state.reader, 15)
                    terminal = events[-1]
                    assert assessed and len(state.native.calls) == 1
                    assert terminal["response"]["action"] == "input_admission_held"
                    assert terminal["response"]["data"]["input_admission"]["status"] == "held"
                    run_id = next(event["run_id"] for event in events if event["event"] == "run_started")
                    row = json.loads(state.ports.items["PROMPT_RUN#" + run_id]["record"]["S"])
                    assert "task_checkpoint_ref" not in row
                finally:
                    await close(state)

        with monkeypatch.context() as positive:
            directory = tmp_path / "admitted"
            directory.mkdir()
            state = setup(positive, directory)
            observed = []
            original_wait = ToolBindingBackend._wait_precommit_original

            async def wait(backend):
                if any(value.kind == "sft" for value in backend._warm_pending_producers.values()):
                    observed.append(
                        (
                            copy.deepcopy(state.ports.items),
                            copy.deepcopy(state.objects.objects),
                            copy.deepcopy(state.ports.objects),
                        )
                    )
                await original_wait(backend)

            positive.setattr(ToolBindingBackend, "_wait_precommit_original", wait)
            try:
                await original_precommit(state, positive)
                await wait_for(state, state.reader, lambda: observed, "first original SFT PRECOMMIT wait")
                descriptor, task = task_publication(state, *observed[0])
                answer = child(cloud_data(state), tmp_path / "admitted-replacement")
                assert answer["loaded"] == [task["sdk_snapshot"]], (
                    "published TASK is not the actual public SDK snapshot consumed by load_snapshot"
                )
                assert answer["invoked"] == [] and answer["model_calls"] == 0
                assert answer["new_training"] == answer["new_processing"] == answer["stops"] == []
                assert descriptor == normal_row(state)["task_checkpoint_ref"]
            finally:
                await close(state)

    asyncio.run(scenario())


def test_coherent_task_selfhash_rewrite_cannot_manufacture_cold_continuation(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            descriptor, task = task_publication(state)
            control = child(cloud_data(state), tmp_path / "untouched-replacement")
            assert control["loaded"] == [task["sdk_snapshot"]] and control["invoked"] == []
            data = cloud_data(state)
            changed = copy.deepcopy(task)
            changed["pending_tool_use_id"] = "ordinary-different-original-use"
            raw = canonical(changed)
            forged = {
                **descriptor,
                "version_id": "ordinary-self-hashed-version",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
            key = descriptor["uri"].split("/", 3)[3]
            data["checkpoint_objects"].append({"key": key, "version": forged["version_id"], "raw": raw})
            item = data["ddb"]["PROMPT_RUN#" + state.run_id]
            record = json.loads(item["record"]["S"])
            record["task_checkpoint_ref"] = forged
            body = canonical(record)
            item["record"] = {"S": body.decode()}
            item["record_sha256"] = {"S": hashlib.sha256(body).hexdigest()}
            answer = child(data, tmp_path / "coherent-tamper-replacement")
            # The resume action answers its state first; the forged TASK then ends the restore as a
            # closed failure at the checkpoint loader (never as an unavailable success).
            assert answer["events"] == [
                {"event": "resume_state", "state": "restoring", "run_id": state.run_id},
                {
                    "event": "resume_state",
                    "state": "failed",
                    "run_id": state.run_id,
                    "diagnostic": {"restore_stage": "checkpoint_load", "exception_class": "ValueError"},
                },
            ]
            assert answer["loaded"] == answer["invoked"] == [] and answer["model_calls"] == 0
            assert answer["new_training"] == answer["new_processing"] == answer["stops"] == []
        finally:
            await close(state)

    asyncio.run(scenario())
