"""Actual original artifact corruption closes only its entry; uncertain reads retain it."""

from __future__ import annotations

import asyncio
import copy
import threading

import pytest
from case_prep_continuation_fixtures import (
    publish_original_outputs,
    stage_work,
    start_original_target,
    wait_until,
)
from multi_producer_fixtures import SequentialProducerStream, configured_data
from test_runtime_connected_warm_model import close_original_worker

from amlas.tui.chat import engine as engine_source

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


@pytest.mark.parametrize("kind", ["case", "preparation"])
@pytest.mark.parametrize("failure", ["immutable-corruption", "transport-timeout", "missing-publication", "contradictory-observation"])
def test_actual_completed_integrity_rejection_consumes_only_target_and_keeps_live_original(request, monkeypatch, tmp_path, kind, failure):
    work = stage_work(request, monkeypatch, kind)
    data = configured_data(monkeypatch)
    original_calls = work.model.data_calls
    monkeypatch.setattr(work.model,"data_calls",lambda:[*original_calls(),*SequentialProducerStream.data_calls()[:1]])
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","coding_case_freeze_flow","data_preparation_flow","data_collection_flow"]')
    checks, armed, publish_lock = [], threading.Event(), threading.Lock()
    actual_step = engine_source._WARM_CHAT_ENGINE_TYPE._observe_pending_model_once
    async def checked_step(engine):
        step = await actual_step(engine)
        target = work.target
        if armed.is_set() and target.actual_observed:
            result = target.actual_observed[-1][1]
            if result.get("status") in {"verification_failed","output_verification_failed"}:
                pending = engine.tool_backend._warm_pending_producers
                retained = [value for value in pending.values() if value.kind == kind]
                if failure == "immutable-corruption":
                    checks.append(not retained)
                    assert not retained, "completed immutable artifact rejection remained awaiting_original"
                    assert step.disposition in {"terminal_failed","refused"}
                    assert engine.tool_backend._warm_closed is None, "one closed target rejected its unrelated live original"
                else:
                    checks.append(bool(retained))
                    assert retained and step.disposition == "awaiting_original"
        return step
    monkeypatch.setattr(engine_source._WARM_CHAT_ENGINE_TYPE,"_observe_pending_model_once",checked_step)
    async def scenario():
        reader=None
        try:
            reader,backend,_result,reference,record,handle=await start_original_target(work,monkeypatch,tmp_path)
            siblings=[value for use,value in work.originals if use=="original-data-one"]
            assert len(siblings)==1 and siblings[0]["accepted"] is True and len(data.native.calls)==1
            sibling_record=backend._acquisition_store._records[siblings[0]["request_ref"]]
            assert sibling_record.handle.id==data.native.calls[0]["ProcessingJobName"]
            await wait_until(lambda:len(backend._warm_pending_producers)==2,reader,"both genuine originals must be captured")
            sibling_pending=next(value for value in backend._warm_pending_producers.values() if value.kind=="data")
            assert sibling_pending.record is sibling_record
            before=copy.deepcopy(work.agents[0].messages)
            original_observe=work.target.store.observe
            def observed(ref):
                with publish_lock:
                    value = original_observe(ref)
                    if failure == "contradictory-observation" and value.get("verification_rejection"):
                        return {**value, "success": True}
                    return value
            monkeypatch.setattr(work.target.store,"observe",observed)
            with publish_lock:
                publish_original_outputs(work,handle)
                prefix=handle.extra["output_prefix"]
                if failure in {"immutable-corruption", "contradictory-observation"}:
                    name="ORACLES.jsonl" if kind=="case" else "train.jsonl"
                    raw,version=work.target.s3.values[prefix+name]
                    changed=raw.replace(b"return 0",b"return 9",1) if kind=="case" else raw.replace(b"{}",b"[]",1)
                    assert len(changed)==len(raw) and changed!=raw
                    work.target.s3.values[prefix+name]=(changed,version)
                elif failure=="transport-timeout":
                    native_get=work.target.s3.get_object
                    def unavailable(**kwargs):
                        if kwargs["Key"].startswith(prefix):
                            raise TimeoutError("ordinary synthetic artifact read uncertainty")
                        return native_get(**kwargs)
                    monkeypatch.setattr(work.target.s3,"get_object",unavailable)
                else:
                    work.target.s3.values.pop(prefix+"COMMITTED.json")
                armed.set()
            await wait_until(lambda:bool(checks),reader,"actual owning failed-artifact observation did not run")
            assert checks and all(checks), "owning immutable rejection did not close its original pending entry"
            assert work.target.native.state=="Completed" and len(work.target.native.creates)==1
            assert work.target.store._records[reference] is record
            assert backend._warm_pending_producers[backend._producer_key(sibling_pending)] is sibling_pending
            assert backend._acquisition_store._records[siblings[0]["request_ref"]] is sibling_record
            assert not reader.done() and work.owned_app.get_async_task_info()["active_count"]==1
            assert work.agents[0].messages==before and work.model.calls==4
            assert len(work.native.requests)==len(data.native.calls)==1
            armed.clear()
            backend.approval_policy=None
            events=await asyncio.wait_for(reader,5)
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"]=="refused"
            assert len(work.target.native.creates)==len(data.native.calls)==1
        finally:
            if reader is not None:
                await close_original_worker(work,reader)
    asyncio.run(scenario())
