"""Normal-client prepared promotion routing only, not cold-loader acceptance.

Fresh actual accepted/prepared Source state is produced by the normal SDK path;
only the existing external TASK-CAS refusal fixture is used for the prestate.
Observe the external publication and public response/approved closed warning,
never graph/load/guard methods or successful hydration. Source/native/cold5 and
full O4 authority/liveness remain separate requirements.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging

from bedrock_agentcore.runtime import BedrockAgentCoreContext
from normal_sft_cold_fixture import close, setup
from test_normal_prepared_promotion import _exact_object, _fresh_unpromoted_original
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime


def _final_restore_line(record):
    """The final restore refusal of `amlas.app`: its text states the same stage and category as its extras."""
    stage = getattr(record, "restore_stage", None)
    category = getattr(record, "exception_class", None)
    return record.name == "amlas.app" and record.getMessage() == (
        f"normal_restore_unavailable reason=final restore_stage={stage} exception_class={category}"
    )


def test_normal_resume_publishes_accepted_task_from_prepared_before_loader_outcome(
    monkeypatch, tmp_path, caplog
):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        logger = logging.getLogger("amlas.app")
        observer = None
        try:
            context, _policy, original, key, before, prepared_raw = await _fresh_unpromoted_original(
                state, monkeypatch
            )
            run_id = context.run_id
            transport = BedrockAgentCoreContext.get_session_id()
            assert transport == context.transport_session_id
            assert "task_checkpoint_ref" not in json.loads(before["record"]["S"])
            actual_update = state.ports.update_item
            publications, loader_outcomes = [], []

            class ClosedOutcome(logging.Handler):
                def emit(self, record):
                    if _final_restore_line(record):
                        stage = getattr(record, "restore_stage", None)
                        category = getattr(record, "exception_class", None)
                        if stage in ("checkpoint_load", "snapshot_load"):
                            loader_outcomes.append((stage, category, bool(publications)))

            observer = ClosedOutcome()
            logger.addHandler(observer)
            calls = state.model.calls
            source_calls = len(state.work.seen_prompts)
            prompt_threads = state.work.owned_app.get_async_task_info()["active_count"]
            assert prompt_threads == 0

            def update(**values):
                raw = values.get("ExpressionAttributeValues", {}).get(":body", {}).get("S")
                task_update = (
                    values["Key"] == {"job_name": before["job_name"]}
                    and type(raw) is str
                    and "task_checkpoint_ref" in json.loads(raw)
                )
                response = actual_update(**values)
                if task_update:
                    publications.append(copy.deepcopy(values))
                return response

            monkeypatch.setattr(state.ports, "update_item", update)
            caplog.set_level(logging.WARNING, logger="amlas.app")
            caplog.clear()
            request = {"action": "resume_prompt_run", "run_id": run_id}
            events = [event async for event in runtime.agent_invocation(request)]
            assert len(publications) == 1, (
                "normal resume must promote its accepted PREPARED task before loading"
            )
            with state.ports.lock:
                retained = copy.deepcopy(state.ports.items[key])
            row = json.loads(retained["record"]["S"])
            prior = json.loads(before["record"]["S"])
            assert row == {**prior, "task_checkpoint_ref": row["task_checkpoint_ref"]}
            assert _exact_object(state, row["task_checkpoint_ref"]) == prepared_raw
            assert BedrockAgentCoreContext.get_session_id() == transport
            assert row["logical_run_id"] == run_id
            assert row["sft"]["native_job_name"] == original.id
            assert state.model.calls == calls and len(state.work.seen_prompts) == source_calls == 1
            assert len(state.native.calls) == 1 and state.native.stops == []
            assert len(state.loaded) <= 1 and state.invoked == []
            assert state.work.owned_app.get_async_task_info()["active_count"] == 0
            assert events[:1] == [{"event": "resume_state", "state": "restoring", "run_id": run_id}]
            events = events[1:]
            assert len(events) == 1 and events[0]["event"] == "prompt_run_status"
            assert events[0]["run_id"] == run_id
            assert events[0]["status"] in ("available", "unavailable")
            assert events[0]["found"] is (events[0]["status"] == "available")
            if events[0]["status"] == "unavailable":
                assert events == [
                    {"event": "prompt_run_status", "status": "unavailable", "run_id": run_id, "found": False}
                ]
                records = [record for record in caplog.records if _final_restore_line(record)]
                assert [
                    record.getMessage()
                    for record in caplog.records
                    if record.name == "amlas.app"
                    and record.getMessage().startswith("normal_restore_unavailable")
                ] == [record.getMessage() for record in records]
                assert len(records) == 1
                assert records[0].restore_stage in ("checkpoint_load", "snapshot_load")
                assert records[0].exception_class in ("ValueError", "RuntimeError", "Exception")
                assert loader_outcomes == [(records[0].restore_stage, records[0].exception_class, True)]
                assert records[0].args == (records[0].restore_stage, records[0].exception_class)
                assert records[0].exc_info is None and records[0].stack_info is None
            else:
                progress = events[0]["progress"]
                assert progress["pending"] is True
                assert progress["disposition"] == "awaiting_original"
                assert progress["status"] in ("InProgress", "Stopping")
                assert progress["lifecycle_verified"] is False and progress["quality_success"] is False
        finally:
            if observer is not None:
                logger.removeHandler(observer)
            await close(state)

    asyncio.run(scenario())
