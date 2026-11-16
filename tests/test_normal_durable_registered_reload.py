"""Fresh controlled NORMAL SDK durable-reload contract; not an actual model/job replay."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ARMED = True


def test_normal_durable_registered_reload_has_its_own_accepted_original(monkeypatch, tmp_path):
    if not ARMED:
        raise RuntimeError("UNARMED normal-durable reload contract has no fixture effects")
    import dataclasses

    import normal_reload_original_completion as publisher
    import normal_sft_cold_fixture as f
    import pytest
    from test_runtime_prompt_entry import inert_runtime_setup

    from amlas import app as runtime
    from amlas.backends.adapters.campaign_state import DynamoCampaignStateBackend
    from amlas.control_plane import normal_run_checkpoint as normal
    from amlas.control_plane.campaign_state import CampaignStateError

    assert (
        hashlib.sha256(Path(f.__file__).read_bytes()).hexdigest()
        == "e92132b6db9fd4fcd7f5c700f2b52b8f52122171fadb2e224f426a2b93555bec"  # pragma: allowlist secret
    )
    assert (
        hashlib.sha256(Path(publisher.__file__).read_bytes()).hexdigest()
        == "e98fbad502bc8f91a6145fcea9046c73b5f0f342c0a709a6ab528b3caea472dd"  # pragma: allowlist secret
    )
    assert (
        hashlib.sha256(Path(normal.__file__).read_bytes()).hexdigest()
        == "42f9e8a87b9d80568b4c68f3d0c828cef97f44fe7434227c847d9281db30f830"  # pragma: allowlist secret
    )
    holder = SimpleNamespace(state=None, primary=None, primary_row=None, journal_frames=[], cleanup=False)
    original_model_class = f.FullGraphModel

    class NormalReloadModel(original_model_class):
        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            known = f.payloads(messages)
            trained = known.get(f.SFT_USE, {})
            if trained.get("event_type") == "original_sft_completed":
                if not getattr(self, "reload_selected", False):
                    state = holder.state
                    backend = state.work.engines[0].tool_backend
                    store = backend._sft_store
                    primary = store._records[trained["request_ref"]]
                    assert type(store._campaign_state) is normal._Journal
                    assert (
                        primary.extra["operation"] == "train"
                        and primary.extra["submission_state"] == "accepted"
                    )
                    assert (
                        type(primary.extra.get("request")) is dict
                        and type(primary.extra.get("staged_refs")) is dict
                    )
                    holder.primary = primary
                    holder.primary_row = copy.deepcopy(state.ports.items[primary.id])
                    self.reload_selected = True
                    self.calls += 1
                    self.messages_seen.append(copy.deepcopy(messages))
                    yield {"messageStart": {"role": "assistant"}}
                    yield {
                        "contentBlockStart": {
                            "contentBlockIndex": 0,
                            "start": {
                                "toolUse": {
                                    "toolUseId": "normal-durable-registered-reload",
                                    "name": "reload_qwen_sft",
                                }
                            },
                        }
                    }
                    yield {
                        "contentBlockDelta": {
                            "contentBlockIndex": 0,
                            "delta": {
                                "toolUse": {
                                    "input": json.dumps(
                                        {"request_ref": trained["request_ref"], "plan_only": False}
                                    )
                                }
                            },
                        }
                    }
                    yield {"contentBlockStop": {"contentBlockIndex": 0}}
                    yield {"messageStop": {"stopReason": "tool_use"}}
                    return
                self.calls += 1
                self.messages_seen.append(copy.deepcopy(messages))
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
                yield {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": "Controlled reload boundary complete."},
                    }
                }
                yield {"contentBlockStop": {"contentBlockIndex": 0}}
                yield {"messageStop": {"stopReason": "end_turn"}}
                return
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event

    original_journal_call = normal._Journal._call

    def journal_call(journal, name, *args, **kwargs):
        try:
            return original_journal_call(journal, name, *args, **kwargs)
        except Exception as error:
            intent = args[1] if name == "reserve_original_job" and len(args) == 2 else kwargs.get("intent")
            if name == "reserve_original_job" and getattr(intent, "operation", None) == "reload":
                frame = error.__traceback__
                selected = None
                while frame is not None:
                    if Path(frame.tb_frame.f_code.co_filename).name == "normal_run_checkpoint.py":
                        selected = frame
                    frame = frame.tb_next
                holder.journal_frames.append(
                    {
                        "phase": "reserve_original_job",
                        "operation": "RELOAD",
                        "error_class": type(error).__name__
                        if type(error).__module__ == "builtins"
                        and type(error).__name__
                        in {"ValueError", "RuntimeError", "TypeError", "KeyError", "AssertionError"}
                        else "OTHER",
                        "public_source_filename": "normal_run_checkpoint.py"
                        if selected is not None
                        else "OTHER",
                        "public_source_function": selected.tb_frame.f_code.co_name
                        if selected is not None
                        and selected.tb_frame.f_code.co_name in {"_call", "_native", "_load"}
                        else "OTHER",
                        "public_source_line": selected.tb_lineno if selected is not None else None,
                        "before_fixture_cleanup": not holder.cleanup,
                    }
                )
            raise

    inert_runtime_setup.__wrapped__(monkeypatch)
    monkeypatch.setattr(f, "FullGraphModel", NormalReloadModel)
    monkeypatch.setattr(normal._Journal, "_call", journal_call)

    async def scenario():
        state = f.setup(monkeypatch, tmp_path)
        holder.state = state
        completion = publisher.OriginalCompletion(state, monkeypatch).attach()
        events = []
        state.events = events

        async def read():
            async for event in runtime.agent_invocation({"prompt": state.work.literal}):
                events.append(event)
            return events

        state.reader = asyncio.create_task(read())
        try:
            await f.wait_for(
                state,
                state.reader,
                lambda: (
                    state.work.engines
                    and any(
                        value.extra.get("operation") == "reload"
                        for value in state.work.engines[0].tool_backend._sft_store._records.values()
                    )
                ),
                "registered NORMAL durable reload retained",
            )
            store = state.work.engines[0].tool_backend._sft_store
            reload_handle = next(
                value for value in store._records.values() if value.extra.get("operation") == "reload"
            )
            assert type(store._campaign_state) is normal._Journal
            assert holder.primary is not None and holder.primary.extra["submission_state"] == "accepted"
            assert state.ports.items[holder.primary.id] == holder.primary_row
            fixed = {
                "scope": "FRESH_CONTROLLED_NORMAL_DURABLE_RELOAD_COMPONENT",
                "normal_journal_verified": True,
                "primary_train_row_unchanged": True,
                "reload_submission_state": reload_handle.extra.get("submission_state")
                if reload_handle.extra.get("submission_state") in {"accepted", "ambiguous", "rejected"}
                else "OTHER",
                "reload_has_request": type(reload_handle.extra.get("request")) is dict,
                "reload_has_staged_refs": type(reload_handle.extra.get("staged_refs")) is dict,
                "reload_error_class": reload_handle.extra.get("error_type")
                if reload_handle.extra.get("error_type")
                in {"ValueError", "RuntimeError", "TypeError", "KeyError", "AssertionError"}
                else "OTHER",
                "reload_durability_status": reload_handle.extra.get("durability_status")
                if reload_handle.extra.get("durability_status")
                in {
                    "recorded",
                    "reservation_uncertain",
                    "marker_uncertain",
                    "outcome_uncertain",
                    "snapshot_uncertain",
                }
                else "OTHER",
                "fake_training_creates": len(state.native.calls),
                "accepted_fake_jobs": len(completion.accepted),
                "journal_frames": holder.journal_frames,
                "native_ml_jobs": 0,
                "live_model_calls": 0,
                "manual_SDK_history_or_witness": False,
                "actual_r4_replayed": False,
            }
            (tmp_path / "normal-durable-reload-journal.closed.json").write_text(
                json.dumps(fixed, sort_keys=True) + "\n"
            )
            # Desired contract; preserve the strict primary TRAIN/journal guard.
            assert reload_handle.extra["submission_state"] == "accepted", fixed
            assert type(reload_handle.extra.get("request")) is dict
            assert type(reload_handle.extra.get("staged_refs")) is dict
            assert len(state.native.calls) == 2

            journal = store._campaign_state
            with normal.activate_normal_run(journal.context):
                binding = store._execution_binding
                consumer = journal.load_original(binding, reload_handle.id)
                assert consumer.intent.operation == "reload"
                assert consumer.native_job_name == reload_handle.id
                assert consumer.native_job_arn == reload_handle.arn
                assert consumer.native_job_name != holder.primary.id
                rows_before = copy.deepcopy(state.ports.items)
                creates_before = len(state.native.calls)
                reserve_events = []
                actual_reserve = DynamoCampaignStateBackend.reserve_original_job

                def counted_reserve(backend, *args, **kwargs):
                    reserve_events.append(True)
                    return actual_reserve(backend, *args, **kwargs)

                monkeypatch.setattr(DynamoCampaignStateBackend, "reserve_original_job", counted_reserve)
                reused = journal.reserve_original_job(binding, consumer.intent)
                assert reused.reused is True
                assert reused.operation_digest == consumer.operation_digest
                assert reused.native_job_name == consumer.native_job_name
                assert reused.native_job_arn == consumer.native_job_arn
                assert len(reserve_events) == 1
                assert state.ports.items == rows_before
                assert len(state.native.calls) == creates_before
                restored = store._adapter.restore_original(
                    consumer,
                    fresh_inputs=consumer.intent.inputs,
                    _reload_compatibility_reader=getattr(store, "_reload_compatibility_reader", None),
                )
                assert restored.id == consumer.native_job_name
                assert restored.arn == consumer.native_job_arn
                assert journal.load_original(binding, restored.id).operation_digest == (
                    consumer.operation_digest
                )
                assert len(state.native.calls) == creates_before
                with pytest.raises((CampaignStateError, ValueError)):
                    store._adapter.restore_original(
                        dataclasses.replace(consumer, native_job_name="declared-foreign-consumer"),
                        fresh_inputs=consumer.intent.inputs,
                        _reload_compatibility_reader=getattr(store, "_reload_compatibility_reader", None),
                    )
                assert len(state.native.calls) == creates_before
                assert state.ports.items == rows_before
                checkpoint = consumer.intent.checkpoint
                foreign_arn = checkpoint.producer_job_arn.rsplit("/", 1)[0]
                foreign_arn += "/declared-foreign-producer"
                changed = (
                    dataclasses.replace(
                        consumer.intent,
                        checkpoint=dataclasses.replace(checkpoint, producer_job_arn=foreign_arn),
                    ),
                    dataclasses.replace(
                        consumer.intent,
                        checkpoint=dataclasses.replace(checkpoint, runtime_session_id="f" * 32),
                    ),
                    dataclasses.replace(
                        consumer.intent,
                        checkpoint=dataclasses.replace(checkpoint, recipe_id="declared-foreign-recipe"),
                    ),
                    dataclasses.replace(
                        consumer.intent,
                        catalog=dataclasses.replace(consumer.intent.catalog, source_commit="f" * 40),
                    ),
                )
                for intent in changed:
                    with pytest.raises((CampaignStateError, ValueError)):
                        journal.reserve_original_job(binding, intent)
                    assert len(reserve_events) == 1
                    assert state.ports.items == rows_before
                    assert len(state.native.calls) == creates_before
                stale = journal.bind_execution(campaign_id=binding.campaign_id, runtime_session_id="f" * 32)
                stale_intent = dataclasses.replace(
                    consumer.intent,
                    binding=stale,
                    checkpoint=dataclasses.replace(checkpoint, runtime_session_id=stale.runtime_session_id),
                )
                with pytest.raises((CampaignStateError, ValueError)):
                    journal.reserve_original_job(stale, stale_intent)
                assert len(reserve_events) == 1
                assert state.ports.items == rows_before
                assert state.ports.items[holder.primary.id] == holder.primary_row
                assert len(state.native.calls) == creates_before
        finally:
            holder.cleanup = True
            await f.close(state)

    asyncio.run(scenario())
