"""Finite original CASE verification uncertainty; actual API/SDK, inert ports only.

An observation failure budget does not impose a native training-age deadline.
Original records/history remain retained; no replay, adoption or cancellation.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from case_prep_continuation_fixtures import (
    collect_original,
    publish_original_outputs,
    stage_work,
    start_original_target,
    wait_until,
)
from multi_producer_fixtures import SequentialProducerStream
from test_qwen_sft_backend import backend as sft_fixture
from test_qwen_sft_backend import catalog
from test_runtime_case_prep_continuation import capture_original
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_sft_reload_originals import current_inputs
from test_runtime_warm_model_continuation import complete_fixture

from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
from amlas.control_plane.preparation_state import PreparationCatalog, SessionPreparationStore
from amlas.control_plane.sft_state import SessionSftStore
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


@pytest.mark.parametrize("failure", [ValueError, RuntimeError])
def test_persistent_original_case_failure_returns_finite_unresolved_without_replay(
    request, monkeypatch, tmp_path, failure
):
    work = stage_work(request, monkeypatch, "case")
    completions = []
    complete_task = work.owned_app.complete_async_task

    def complete_once(task_id):
        completions.append(task_id)
        return complete_task(task_id)

    monkeypatch.setattr(work.owned_app, "complete_async_task", complete_once)

    async def scenario():
        reader = None
        try:
            reader, backend, _result, reference, record, handle = await start_original_target(
                work, monkeypatch, tmp_path
            )
            pending = capture_original(backend, work, reference, record)
            before = copy.deepcopy(work.agents[0].messages)
            consumed = set(backend._warm_consumed)
            native_reads = len(work.target.observations)
            s3_reads = len(work.target.s3.calls)
            failures = []
            work.target.native.state = "Completed"

            def unavailable(original):
                assert original is handle
                failures.append(original.id)
                raise failure("ordinary unavailable original case verification")

            monkeypatch.setattr(work.target.adapter, "artifacts", unavailable)
            await wait_until(lambda: len(failures) >= 8, reader, "eight qualified failures missing")
            await asyncio.sleep(0.03)
            assert work.api_returned.is_set(), (
                "CASE failure wait did not close after eight qualified observations"
            )
            events = await asyncio.wait_for(reader, 5)
            terminal = events[-1]
            assert terminal["worker_finished"] is True
            assert "UNKNOWN" in terminal["response"]["message"]
            assert terminal["response"]["data"]["warm_model_continuation"] == {
                "disposition": "submission_unresolved",
                "diagnostic": {
                    "phase": "native_observation",
                    "reason": "original_unknown_budget_exhausted",
                    "error_class": "UNKNOWN",
                },
            }
            assert terminal["quality_success"] is False and terminal["lifecycle_verified"] is False
            assert len(failures) == 8 and len(completions) == 1
            assert len(work.target.observations) == native_reads and len(work.target.s3.calls) == s3_reads
            assert len(work.target.native.creates) == len(work.native.requests) == 1
            assert work.target.store._records[reference] is record
            assert backend._warm_pending_producers[backend._producer_key(pending)] is pending
            assert backend._warm_consumed == consumed and backend._warm_closed == "submission_unresolved"
            assert work.model.calls == 4 and work.agents[0].messages == before
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            await work.engines[0]._observe_pending_model_once()
            assert len(failures) == 8 and len(work.target.native.creates) == 1
        finally:
            owned = reader if reader is not None else getattr(work, "reader", None)
            if owned is not None:
                await close_original_worker(work, owned)

    asyncio.run(scenario())


def test_healthy_running_case_resets_failure_streak_before_verified_same_agent_completion(
    request, monkeypatch, tmp_path
):
    work = stage_work(request, monkeypatch, "case")

    async def scenario():
        reader = None
        try:
            reader, backend, _result, reference, record, handle = await start_original_target(
                work, monkeypatch, tmp_path
            )
            capture_original(backend, work, reference, record)
            before = copy.deepcopy(work.agents[0].messages)
            actual_artifacts = work.target.adapter.artifacts
            attempts = []

            def sequence(original):
                assert original is handle
                attempts.append(original.id)
                count = len(attempts)
                if count <= 7 or 9 <= count <= 15:
                    raise ValueError("ordinary unavailable original case verification")
                if count == 16:
                    publish_original_outputs(work, handle)
                return actual_artifacts(original)

            monkeypatch.setattr(work.target.adapter, "artifacts", sequence)
            events = await asyncio.wait_for(reader, 5)
            assert (
                events[-1]["response"]["data"]["warm_model_continuation"]["disposition"]
                == "continued_original"
            )
            assert len(attempts) == 16 and work.model.calls == 5
            assert work.agents[0].messages[: len(before)] == before
            assert work.target.store._records[reference] is record
            assert len(work.target.native.creates) == len(work.native.requests) == 1
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            owned = reader if reader is not None else getattr(work, "reader", None)
            if owned is not None:
                await close_original_worker(work, owned)

    asyncio.run(scenario())


class FullTrainingStream(SequentialProducerStream):
    def data_calls(self):
        return [
            (
                "original-train-use",
                "submit_qwen_sft",
                {
                    "model_request_ref": "model_acq_" + "a" * 32,
                    "preparation_request_ref": "prep_" + "b" * 32,
                    "recipe_id": "full_v1",
                },
            )
        ]


def test_healthy_three_hour_full_training_is_not_an_observation_failure(monkeypatch, tmp_path):
    adapter, native, s3, inputs = sft_fixture()
    requests, observations = {}, []
    create_training = native.create_training_job

    def create(**kwargs):
        result = create_training(**kwargs)
        requests[kwargs["TrainingJobName"]] = kwargs
        return result

    def describe(**kwargs):
        name = kwargs["TrainingJobName"]
        request = requests[name]
        observations.append(name)
        return {
            **{key: value for key, value in request.items() if key != "Tags"},
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":training-job/" + name,
            "TrainingJobStatus": "InProgress",
            "TrainingStartTime": datetime.now(UTC) - timedelta(hours=3),
            "TrainingTimeInSeconds": 10800,
            "ModelArtifacts": {"S3ModelArtifacts": request["OutputDataConfig"]["S3OutputPath"] + "/model/"},
        }

    monkeypatch.setattr(native, "create_training_job", create)
    monkeypatch.setattr(native, "describe_training_job", describe)
    monkeypatch.setattr(
        native, "list_tags", lambda **kwargs: {"Tags": requests[kwargs["ResourceArn"].split("/")[-1]]["Tags"]}
    )
    work = connected_work(monkeypatch, model=FullTrainingStream())
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","qwen_sft_flow"]')

    def prepare(acquired, **kwargs):
        selected = PreparationCatalog(
            **dataclasses.asdict(catalog()), exclusions_ref=inputs.prepared.exclusions_ref
        )
        return SessionPreparationStore(acquired, ProcessingCodingPreparationBackend(), selected)

    def make_sft(models, prepared):
        selected = current_inputs(inputs, s3, models.runtime_session_id)
        monkeypatch.setattr(models, "_verified_sft_model_snapshot", lambda reference: selected.model)
        monkeypatch.setattr(prepared, "_verified_sft_snapshot", lambda reference: selected.prepared)
        return SessionSftStore(models, prepared, adapter, catalog())

    monkeypatch.setattr(binding, "build_preparation_store", prepare)
    monkeypatch.setattr(binding, "build_sft_store", make_sft)

    async def scenario():
        reader = asyncio.create_task(collect_original(work))
        try:
            await wait_until(work.observing.is_set, reader, "original model observer missing")
            backend = work.engines[0].tool_backend
            result = next(value for use, value in work.originals if use == "original-model-use")
            handle = backend._model_acquisition_store._records[result["request_ref"]].handle
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=handle, adapter=work.adapter, native=work.native),
            )
            await wait_until(
                lambda: len(observations) >= 12, reader, "healthy full training observations missing"
            )
            assert not reader.done() and not work.api_returned.is_set()
            pending = next(value for value in backend._warm_pending_producers.values() if value.kind == "sft")
            assert backend._producer_handle(pending).extra["recipe_id"] == "full_v1"
            assert len(requests) == len(work.native.requests) == 1 and work.model.calls == 4
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            backend.approval_policy = None
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert len(requests) == 1 and work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["policy", "record"])
def test_eighth_case_failure_current_refusal_precedes_budget(request, monkeypatch, tmp_path, change):
    work = stage_work(request, monkeypatch, "case")

    async def scenario():
        reader = None
        try:
            reader, backend, _result, reference, record, handle = await start_original_target(
                work, monkeypatch, tmp_path
            )
            pending = capture_original(backend, work, reference, record)
            before = copy.deepcopy(work.agents[0].messages)
            consumed = set(backend._warm_consumed)
            failures = []

            def unavailable(original):
                assert original is handle
                failures.append(original.id)
                if len(failures) == 8:
                    if change == "policy":
                        backend.approval_policy = None
                    else:
                        monkeypatch.setitem(work.target.store._records, reference, copy.copy(record))
                raise ValueError("ordinary unavailable original case verification")

            monkeypatch.setattr(work.target.adapter, "artifacts", unavailable)
            events = await asyncio.wait_for(reader, 5)
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert len(failures) == 8 and backend._warm_closed == "refused"
            assert pending.record is record and backend._warm_consumed == consumed
            assert work.agents[0].messages == before and work.model.calls == 4
            assert len(work.target.native.creates) == len(work.native.requests) == 1
            assert events[-1]["worker_finished"] is True
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            owned = reader if reader is not None else getattr(work, "reader", None)
            if owned is not None:
                await close_original_worker(work, owned)

    asyncio.run(scenario())


def test_eighth_case_closed_immutable_rejection_precedes_generic_failure_budget(
    request, monkeypatch, tmp_path
):
    from amlas.backends.adapters.artifact_rejection import is_closed_rejection

    work = stage_work(request, monkeypatch, "case")

    async def scenario():
        reader = None
        try:
            reader, backend, _result, reference, record, handle = await start_original_target(
                work, monkeypatch, tmp_path
            )
            pending = capture_original(backend, work, reference, record)
            before = copy.deepcopy(work.agents[0].messages)
            actual_artifacts = work.target.adapter.artifacts
            attempts = []

            def sequence(original):
                assert original is handle
                attempts.append(original.id)
                if len(attempts) <= 7:
                    raise ValueError("ordinary unavailable original case verification")
                publish_original_outputs(work, handle)
                key = handle.extra["output_prefix"] + "ORACLES.jsonl"
                raw, version = work.target.s3.values[key]
                changed = raw.replace(b"return 0", b"return 9", 1)
                assert len(changed) == len(raw) and changed != raw
                work.target.s3.values[key] = (changed, version)
                return actual_artifacts(original)

            monkeypatch.setattr(work.target.adapter, "artifacts", sequence)
            events = await asyncio.wait_for(reader, 5)
            assert (
                events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "terminal_failed"
            )
            assert len(attempts) == 8 and backend._warm_closed is None
            assert is_closed_rejection(work.target.actual_observed[-1][1]["verification_rejection"])
            assert backend._producer_key(pending) not in backend._warm_pending_producers
            assert backend._producer_key(pending) in backend._warm_consumed
            assert work.target.store._records[reference] is record
            assert work.agents[0].messages == before and work.model.calls == 4
            assert len(work.target.native.creates) == len(work.native.requests) == 1
            assert events[-1]["worker_finished"] is True
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            owned = reader if reader is not None else getattr(work, "reader", None)
            if owned is not None:
                await close_original_worker(work, owned)

    asyncio.run(scenario())
