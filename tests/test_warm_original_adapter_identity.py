"""Actual SDK originals must retain the adapter receiving their private IO scope."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from types import SimpleNamespace

import pytest
from botocore.session import Session
from botocore.validate import validate_parameters
from case_prep_continuation_fixtures import TARGET_TOOL_USE, stage_work
from test_runtime_generic_original_guards import connected_data
from test_runtime_warm_model_continuation import complete_fixture

from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat import engine as engine_source

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


class NativeObservationRecorder:
    """Delegate only validated reads to the exact existing strict native fixture."""

    def __init__(self, original):
        self.original = original
        self.calls = []
        self.model = Session().get_service_model("sagemaker")

    def describe_processing_job(self, **kwargs):
        validate_parameters(kwargs, self.model.operation_model("DescribeProcessingJob").input_shape)
        self.calls.append("describe")
        return self.original.describe_processing_job(**kwargs)

    def list_tags(self, **kwargs):
        validate_parameters(kwargs, self.model.operation_model("ListTags").input_shape)
        self.calls.append("tags")
        return self.original.list_tags(**kwargs)


def fresh_policy():
    return ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=None, interactive=False)


async def prep_original(request, monkeypatch, tmp_path):
    work = stage_work(request, monkeypatch, "preparation")
    engine = engine_source.ChatEngine(settings=Settings(), approval_policy=fresh_policy())
    await engine.process_message(work.literal)
    backend = engine.tool_backend
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    original_model = next(value for use, value in work.originals if use == "original-model-use")
    model_handle = backend._model_acquisition_store._records[original_model["request_ref"]].handle
    complete_fixture(
        monkeypatch, tmp_path,
        SimpleNamespace(handle=model_handle, adapter=work.adapter, native=work.native),
    )
    assert (await engine._observe_pending_model_once()).disposition == "continued_original"
    results = [value for use, value in work.originals if use == TARGET_TOOL_USE]
    assert len(results) == 1 and results[0]["accepted"] is True
    reference = results[0]["request_ref"]
    record = work.target.store._records[reference]
    pending = next(value for value in backend._warm_pending_producers.values()
                   if value.kind == "preparation" and value.reference == reference)
    assert pending.store is work.target.store and pending.record is record
    assert pending.adapter is record.adapter is work.target.adapter
    assert pending.agent is work.agents[0] and pending.tool_use_id == TARGET_TOOL_USE
    assert pending.handle_pin == backend._producer_handle_pin("preparation", record.handle)
    assert backend._producer_handle(pending) is record.handle
    assert (record.handle.id, record.handle.arn) == (results[0]["job_name"], results[0]["job_arn"])
    assert len(work.target.native.creates) == 1 and work.model.calls == 4
    assert len(pending.graph) == 7 and backend._warm_graph() == pending.graph
    return work, backend, pending, record


def test_registered_prep_unchanged_adapter_observes_same_original(request, monkeypatch, tmp_path):
    async def scenario():
        work, backend, pending, record = await prep_original(request, monkeypatch, tmp_path)
        before = copy.deepcopy(work.agents[0].messages)
        reads = len(work.target.observations)
        step = backend._observe_pending_producer_once()
        assert step.disposition == "awaiting_original"
        assert len(work.target.observations) == reads + 1
        assert pending.store._records[pending.reference] is record
        assert record.adapter is pending.adapter and backend._warm_graph() == pending.graph
        assert work.agents[0].messages == before and work.model.calls == 4
        assert len(work.target.native.creates) == len(work.native.requests) == 1

    asyncio.run(scenario())


def test_registered_prep_adapter_replacement_refuses_before_replacement_io(request, monkeypatch, tmp_path):
    async def scenario():
        work, backend, pending, record = await prep_original(request, monkeypatch, tmp_path)
        before = copy.deepcopy(work.agents[0].messages)
        recorder = NativeObservationRecorder(work.target.native)
        replacement = type(pending.adapter)()
        replacement._sm, replacement._s3 = recorder, work.target.s3
        assert type(replacement) is type(pending.adapter) and replacement is not pending.adapter
        record.adapter = replacement
        assert pending.store._records[pending.reference] is record and record.adapter is replacement
        assert pending.adapter is work.target.adapter
        assert backend._producer_handle(pending) is record.handle
        assert pending.handle_pin == backend._producer_handle_pin("preparation", record.handle)
        step = backend._observe_pending_producer_once()
        assert pending.store._records[pending.reference] is record
        assert backend._warm_graph() == pending.graph and work.agents[0].messages == before
        assert work.model.calls == 4 and len(work.target.native.creates) == len(work.native.requests) == 1
        assert recorder.calls == [], "replacement adapter received unfenced original native reads"
        assert step.disposition == "refused"

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["frozen-field", "whole-record"] )
def test_registered_data_frozen_record_cannot_replace_captured_adapter(monkeypatch, change):
    async def scenario():
        work, data = connected_data(monkeypatch)
        engine = engine_source.ChatEngine(settings=Settings(), approval_policy=fresh_policy())
        await engine.process_message(work.literal)
        backend = engine.tool_backend
        originals = [value for use, value in work.originals if use == "original-data-one"]
        assert len(originals) == 1 and originals[0]["accepted"] is True
        reference = originals[0]["request_ref"]
        record = backend._acquisition_store._records[reference]
        pending = next(value for value in backend._warm_pending_producers.values()
                       if value.kind == "data" and value.reference == reference)
        assert pending.record is record and pending.adapter is record.adapter is data.adapter
        assert (record.handle.id, record.handle.arn) == (originals[0]["job_name"], originals[0]["job_arn"])
        recorder = NativeObservationRecorder(data.native)
        replacement = type(data.adapter)()
        replacement._sm, replacement._s3 = recorder, data.s3
        reads = len(data.native.observations)
        if change == "frozen-field":
            with pytest.raises(dataclasses.FrozenInstanceError):
                record.adapter = replacement
            assert backend._qualify_warm_producer(pending)["_amlas_acquisition_store"] is pending.store
            assert record.adapter is pending.adapter
        else:
            with pending.store._lock:
                pending.store._records[reference] = dataclasses.replace(record, adapter=replacement)
            assert backend._observe_pending_producer_once().disposition == "refused"
        assert recorder.calls == [] and len(data.native.observations) == reads
        assert len(data.native.calls) == len(work.native.requests) == 1
        assert backend._warm_graph() == pending.graph

    asyncio.run(scenario())
