"""PREPARED persistence before effects over one fresh qualified warm SDK path.

The existing fresh full-graph fixture controls inert external providers only.
Actual Source input/parent/native qualifiers and SDK snapshot capture remain;
no capture, journal, approval or TASK-ready state is seeded. This proves bounded
wire/order preservation, not native permission, cold recovery, promotion, wake
liveness or full O4. The existing close helper is thread cleanup only.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading

import pytest
from botocore.exceptions import ClientError
from normal_sft_cold_fixture import close, original_precommit, prepare_original_submission, setup
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.control_plane.normal_run_checkpoint import (
    _task_causal_prefix,
    current_normal_run,
)

# Upper bound for waits on a boundary that the run must reach. A slow machine
# needs the room; a boundary that is never reached still fails when it ends.
LIVENESS_WAIT_SECONDS = 180

REF_KEYS = {"uri", "version_id", "sha256", "byte_count", "format"}
PREPARED_KEYS = {
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


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def test_fresh_captured_sft_persists_exact_prepared_snapshot_before_wake_and_effects(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        observed = []
        actual_route_put = state.ports.put_item
        actual_create = state.native.create_training_job

        def record(operation):
            context = current_normal_run()
            if context is None:
                return
            with state.ports.lock:
                item = state.ports.items.get("PROMPT_RUN#" + context.run_id)
                row = json.loads(item["record"]["S"]) if item is not None else {}
                descriptor = copy.deepcopy(row.get("prepared_checkpoint_ref"))
            raw = None
            if type(descriptor) is dict:
                uri, version = descriptor.get("uri"), descriptor.get("version_id")
                if type(uri) is str and uri.startswith("s3://") and type(version) is str:
                    key = uri.split("/", 3)[3]
                    raw = state.objects.objects.get((key, version))
                    if raw is None:
                        raw = state.ports.objects.get((key, version))
            observed.append(
                {
                    "operation": operation,
                    "run_id": context.run_id,
                    "private_correlation": row.get("private_store_correlation"),
                    "association": copy.deepcopy(row.get("sft")),
                    "descriptor": descriptor,
                    "raw": copy.deepcopy(raw),
                    "captured_snapshot": context._sft_snapshot,
                    "captured_call": context._causal_call,
                }
            )

        def route_put(**kwargs):
            if kwargs["Item"]["job_name"]["S"].startswith("PROMPT_WAKE_JOB#Training#"):
                record("wake_route_put")
            return actual_route_put(**kwargs)

        def create(**kwargs):
            record("CreateTrainingJob")
            return actual_create(**kwargs)

        def stage_spy(operation, actual):
            def call(**kwargs):
                context = current_normal_run()
                # Upstream fixture publication and parents/TASK objects are not
                # SFT staging; after actual capture only native-input writes count.
                if (
                    context is not None
                    and context._sft_snapshot is not None
                    and "/prompt-runs/" not in kwargs.get("Key", "")
                ):
                    record(operation)
                return actual(**kwargs)

            return call

        monkeypatch.setattr(state.ports, "put_item", route_put)
        monkeypatch.setattr(state.native, "create_training_job", create)
        for operation in (
            "put_object",
            "copy_object",
            "create_multipart_upload",
            "upload_part_copy",
            "complete_multipart_upload",
        ):
            actual = getattr(state.objects, operation, None)
            if callable(actual):
                monkeypatch.setattr(state.objects, operation, stage_spy(operation, actual))
        try:
            await original_precommit(state, monkeypatch)
            assert len(state.native.calls) == 1 and state.native.stops == []
            assert observed and observed[0]["captured_snapshot"] is not None
            assert observed[0]["captured_call"] is not None
            assert any(value["operation"] == "wake_route_put" for value in observed)
            assert any(value["operation"] == "CreateTrainingJob" for value in observed)
            assert any(
                value["operation"] not in ("wake_route_put", "CreateTrainingJob") for value in observed
            )
            expected_snapshot = json.loads(observed[0]["captured_snapshot"])
            expected_call = json.loads(observed[0]["captured_call"])
            assert expected_snapshot["data"]["messages"] == expected_call["causal_prefix"]
            for value in observed:
                descriptor = value["descriptor"]
                assert descriptor is not None, (
                    "captured SDK PREPARED checkpoint is missing before wake/staging/Create"
                )
                assert set(descriptor) == REF_KEYS
                assert type(descriptor["version_id"]) is str and descriptor["version_id"]
                assert type(descriptor["byte_count"]) is int and 0 < descriptor["byte_count"] <= 160 * 1024
                assert descriptor["format"] == "json"
                raw = value["raw"]
                assert type(raw) is bytes and len(raw) == descriptor["byte_count"]
                digest = hashlib.sha256(raw).hexdigest()
                assert digest == descriptor["sha256"]
                suffix = (
                    "/sessions/"
                    + value["private_correlation"]
                    + "/prompt-runs/"
                    + state.run_id
                    + "/prepared-tasks/"
                    + digest
                    + ".json"
                )
                assert descriptor["uri"].endswith(suffix)
                body = json.loads(raw)
                assert set(body) == PREPARED_KEYS
                assert type(body["schema_version"]) is int and body["schema_version"] == 2
                association = value["association"]
                parent_ref = association["intent"]["causal_parent_ref"]
                if set(parent_ref) == {"type", "fields"}:
                    assert parent_ref["type"] == "DatasetArtifactRef"
                    parent_ref = parent_ref["fields"]
                assert set(parent_ref) == REF_KEYS
                assert body == {
                    "schema_version": 2,
                    "logical_run_id": state.run_id,
                    "private_store_correlation": value["private_correlation"],
                    "native_job_name": association["native_job_name"],
                    "native_job_arn": association["native_job_arn"],
                    "operation_digest": association["operation_digest"],
                    "source_refs": [association["model_ref"], association["preparation_ref"]],
                    "causal_parent_ref": parent_ref,
                    "sdk_snapshot": expected_snapshot,
                    "pending_tool_use_id": expected_call["pending_tool_use_id"],
                }
                assert raw == _canonical(body)
                assert body["sdk_snapshot"]["data"]["messages"] == _task_causal_prefix(body)
                assert _task_causal_prefix(body) == expected_call["causal_prefix"]
                for field in ("state", "conversation_manager_state", "interrupt_state"):
                    assert body["sdk_snapshot"]["data"][field] == expected_snapshot["data"][field]
                assert "status" not in body and "accepted" not in body and "ready" not in body
                assert value["captured_snapshot"] == observed[0]["captured_snapshot"]
                assert value["captured_call"] == observed[0]["captured_call"]
        finally:
            await close(state)

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["put_denied", "lost_ack_exact_version_unavailable"])
def test_prepared_publication_denial_or_unresolved_version_prevents_sft_effects(monkeypatch, tmp_path, fault):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        prepared_boundary, checked, reached = threading.Event(), threading.Event(), threading.Event()
        puts, denied_versions, effects, committed = [], [], [], []
        actual_put, actual_get = state.objects.put_object, state.objects.get_object
        actual_create = state.native.create_training_job

        def prepared_key(values):
            key = values.get("Key", "")
            return type(key) is str and "/prompt-runs/" in key and "/prepared-tasks/" in key

        def put(**values):
            if not prepared_key(values):
                return actual_put(**values)
            prepared_boundary.set()
            state.objects.check("PutObject", values)
            checked.set()
            puts.append(copy.deepcopy(values))
            reached.set()
            if fault == "put_denied":
                raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
            response = actual_put(**values)
            committed.append((values["Key"], response["VersionId"]))
            raise TimeoutError("Declared prepared-object ACK is unavailable.")

        def get(**values):
            if prepared_key(values) and reached.is_set():
                state.objects.check("GetObject", values)
                if fault == "put_denied":
                    raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
                if values.get("VersionId") is not None:
                    denied_versions.append(copy.deepcopy(values))
                    raise TimeoutError("Declared exact prepared version is unavailable.")
                # Model native latest-GET resolution from the one committed Put;
                # the later Source exact-version confirmation remains unavailable.
                return actual_get(**{**values, "VersionId": committed[0][1]})
            return actual_get(**values)

        def create(**values):
            effects.append("CreateTrainingJob")
            return actual_create(**values)

        def stage_spy(operation, actual):
            def call(**values):
                context = current_normal_run()
                if (
                    context is not None
                    and context._sft_snapshot is not None
                    and "/prompt-runs/" not in values.get("Key", "")
                ):
                    effects.append(operation)
                return actual(**values)

            return call

        monkeypatch.setattr(state.objects, "put_object", put)
        monkeypatch.setattr(state.objects, "get_object", get)
        monkeypatch.setattr(state.native, "create_training_job", create)
        for operation in (
            "copy_object",
            "create_multipart_upload",
            "upload_part_copy",
            "complete_multipart_upload",
        ):
            actual = getattr(state.objects, operation, None)
            if callable(actual):
                monkeypatch.setattr(state.objects, operation, stage_spy(operation, actual))
        # Native input JSON Put uses the same client; record it while leaving
        # prepared/parent/TASK publication under the strict fault selector.
        fault_put = state.objects.put_object
        monkeypatch.setattr(state.objects, "put_object", stage_spy("put_object", fault_put))
        try:
            reader, _received, _backend = await prepare_original_submission(state, monkeypatch)
            deadline = asyncio.get_running_loop().time() + LIVENESS_WAIT_SECONDS
            while (
                not reached.is_set()
                and not state.native.calls
                and not reader.done()
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.005)
            assert prepared_boundary.is_set(), "matching prepared-object port boundary was never reached"
            assert checked.is_set(), "prepared-object wire validation did not pass"
            assert reached.is_set(), "required prepared-object publication fault was never reached"
            await asyncio.wait_for(asyncio.shield(reader), LIVENESS_WAIT_SECONDS)
            assert len(puts) == 1 and puts[0]["IfNoneMatch"] == "*"
            assert effects == [] and state.native.calls == [] and state.native.stops == []
            assert len(state.work.engines) == len(state.work.agents) == 1
            assert len(state.work.seen_prompts) == 1 and state.work.seen_prompts[0][0] == state.work.literal
            terminal = state.events[-1]
            assert terminal["event"] == "run_completed" and terminal["worker_finished"] is True
            assert terminal["lifecycle_verified"] is False and terminal["quality_success"] is False
            assert not state.task_published.is_set()
            if fault == "put_denied":
                assert committed == []
            else:
                assert len(committed) == 1 and denied_versions
                raw = state.objects.objects[committed[0]]
                assert type(raw) is bytes and set(json.loads(raw)) == PREPARED_KEYS
                assert all(request["VersionId"] == committed[0][1] for request in denied_versions)
        finally:
            await close(state)

    asyncio.run(scenario())


def test_fresh_no_fault_prepared_wrappers_reach_same_source_boundary(monkeypatch, tmp_path):
    """Fresh wrapper wiring control only; no denial or recovery evidence."""
    fault = "no_fault"

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        prepared_boundary, checked, reached = threading.Event(), threading.Event(), threading.Event()
        puts, denied_versions, effects, committed = [], [], [], []
        actual_put, actual_get = state.objects.put_object, state.objects.get_object
        actual_create = state.native.create_training_job

        def prepared_key(values):
            key = values.get("Key", "")
            return type(key) is str and "/prompt-runs/" in key and "/prepared-tasks/" in key

        def put(**values):
            if not prepared_key(values):
                return actual_put(**values)
            prepared_boundary.set()
            state.objects.check("PutObject", values)
            checked.set()
            puts.append(copy.deepcopy(values))
            reached.set()
            if fault == "put_denied":
                raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
            response = actual_put(**values)
            committed.append((values["Key"], response["VersionId"]))
            return response

        def get(**values):
            if prepared_key(values) and reached.is_set():
                state.objects.check("GetObject", values)
            return actual_get(**values)

        def create(**values):
            effects.append("CreateTrainingJob")
            return actual_create(**values)

        def stage_spy(operation, actual):
            def call(**values):
                context = current_normal_run()
                if (
                    context is not None
                    and context._sft_snapshot is not None
                    and "/prompt-runs/" not in values.get("Key", "")
                ):
                    effects.append(operation)
                return actual(**values)

            return call

        monkeypatch.setattr(state.objects, "put_object", put)
        monkeypatch.setattr(state.objects, "get_object", get)
        monkeypatch.setattr(state.native, "create_training_job", create)
        for operation in (
            "copy_object",
            "create_multipart_upload",
            "upload_part_copy",
            "complete_multipart_upload",
        ):
            actual = getattr(state.objects, operation, None)
            if callable(actual):
                monkeypatch.setattr(state.objects, operation, stage_spy(operation, actual))
        # Native input JSON Put uses the same client; record it while leaving
        # prepared/parent/TASK publication under the strict fault selector.
        fault_put = state.objects.put_object
        monkeypatch.setattr(state.objects, "put_object", stage_spy("put_object", fault_put))
        try:
            await original_precommit(state, monkeypatch)
            assert prepared_boundary.is_set(), "no-fault wrappers did not reach prepared-object boundary"
            assert checked.is_set() and reached.is_set()
            assert len(puts) == len(committed) == 1
            assert denied_versions == []
            assert len(state.native.calls) == 1 and state.native.stops == []
            assert "CreateTrainingJob" in effects
            assert len(state.work.engines) == len(state.work.agents) == 1
        finally:
            await close(state)

    asyncio.run(scenario())


def test_confirmed_prepared_cas_cannot_revert_to_prior_row_before_effects(monkeypatch, tmp_path):
    """Fresh external row race only; stored bytes/data grant no continuation."""

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        cas_applied, reverted = threading.Event(), threading.Event()
        prior_items, after_items, confirmed_reads = [], [], []
        normal_reads, wake_puts, effects = [], [], []
        actual_update, actual_get, actual_put = (
            state.ports.update_item,
            state.ports.get_item,
            state.ports.put_item,
        )
        actual_create = state.native.create_training_job

        def update(**values):
            attributes = values.get("ExpressionAttributeValues", {})
            body = attributes.get(":body", {}).get("S")
            if (
                not values["Key"]["job_name"]["S"].startswith("PROMPT_RUN#")
                or type(body) is not str
                or "prepared_checkpoint_ref" not in json.loads(body)
            ):
                return actual_update(**values)
            prior = {
                "job_name": copy.deepcopy(values["Key"]["job_name"]),
                "record": copy.deepcopy(attributes[":prior"]),
                "record_sha256": copy.deepcopy(attributes[":prior_sha"]),
            }
            after = {
                "job_name": copy.deepcopy(values["Key"]["job_name"]),
                "record": copy.deepcopy(attributes[":body"]),
                "record_sha256": copy.deepcopy(attributes[":sha"]),
            }
            response = actual_update(**values)
            prior_items.append(prior)
            after_items.append(after)
            cas_applied.set()
            return response

        def get(**values):
            response = actual_get(**values)
            if cas_applied.is_set() and values["Key"] == {"job_name": after_items[0]["job_name"]}:
                normal_reads.append(copy.deepcopy(values))
                # Source sequence after CAS: replacement finally-current,
                # publisher current, explicit confirmed-load, final current.
                if len(normal_reads) <= 3:
                    confirmed_reads.append(copy.deepcopy(response.get("Item")))
                elif len(normal_reads) == 4 and all(item == after_items[0] for item in confirmed_reads):
                    key = prior_items[0]["job_name"]["S"]
                    with state.ports.lock:
                        state.ports.items[key] = copy.deepcopy(prior_items[0])
                    reverted.set()
                    return {"Item": copy.deepcopy(prior_items[0])}
            return response

        def put(**values):
            if values["Item"]["job_name"]["S"].startswith("PROMPT_WAKE_JOB#Training#"):
                wake_puts.append(copy.deepcopy(values))
            return actual_put(**values)

        def create(**values):
            effects.append("CreateTrainingJob")
            return actual_create(**values)

        def stage_spy(operation, actual):
            def call(**values):
                context = current_normal_run()
                if (
                    context is not None
                    and context._sft_snapshot is not None
                    and "/prompt-runs/" not in values.get("Key", "")
                ):
                    effects.append(operation)
                return actual(**values)

            return call

        monkeypatch.setattr(state.ports, "update_item", update)
        monkeypatch.setattr(state.ports, "get_item", get)
        monkeypatch.setattr(state.ports, "put_item", put)
        monkeypatch.setattr(state.native, "create_training_job", create)
        for operation in (
            "put_object",
            "copy_object",
            "create_multipart_upload",
            "upload_part_copy",
            "complete_multipart_upload",
        ):
            actual = getattr(state.objects, operation, None)
            if callable(actual):
                monkeypatch.setattr(state.objects, operation, stage_spy(operation, actual))
        try:
            reader, _received, _backend = await prepare_original_submission(state, monkeypatch)
            deadline = asyncio.get_running_loop().time() + LIVENESS_WAIT_SECONDS
            while (
                not reverted.is_set()
                and not state.native.calls
                and not reader.done()
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.005)
            assert cas_applied.is_set(), "actual prepared conditional update was never reached"
            assert confirmed_reads == [after_items[0]] * 3, (
                "three Source-defined after-row reads must precede the race"
            )
            assert reverted.is_set(), "final-current prior-row race was never reached"
            prior, after = (
                json.loads(prior_items[0]["record"]["S"]),
                json.loads(after_items[0]["record"]["S"]),
            )
            assert prior["stage"] == "sft" and "prepared_checkpoint_ref" not in prior
            assert after == {**prior, "prepared_checkpoint_ref": after["prepared_checkpoint_ref"]}
            for item in (prior_items[0], after_items[0]):
                raw = item["record"]["S"]
                assert raw.encode() == _canonical(json.loads(raw))
                assert item["record_sha256"] == {"S": hashlib.sha256(raw.encode()).hexdigest()}
            publication_deadline = asyncio.get_running_loop().time() + LIVENESS_WAIT_SECONDS
            while not reader.done() and not wake_puts and not effects:
                assert asyncio.get_running_loop().time() < publication_deadline, (
                    "row-race outcome remained unresolved"
                )
                await asyncio.sleep(0.005)
            assert wake_puts == [] and effects == [] and state.native.calls == [] and state.native.stops == []
            await asyncio.wait_for(asyncio.shield(reader), LIVENESS_WAIT_SECONDS)
            assert not state.task_published.is_set()
            assert len(state.work.engines) == len(state.work.agents) == 1
            assert len(state.work.seen_prompts) == 1 and state.work.seen_prompts[0][0] == state.work.literal
            assert (
                state.events[-1]["event"] == "run_completed" and state.events[-1]["worker_finished"] is True
            )
            assert (
                state.events[-1]["lifecycle_verified"] is False
                and state.events[-1]["quality_success"] is False
            )
        finally:
            await close(state)

    asyncio.run(scenario())
