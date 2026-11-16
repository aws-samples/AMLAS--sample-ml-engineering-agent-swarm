"""Fresh replay setup contract with fixed assertions and opaque provider DATA.

No old checkpoint, row, graph, returned TASK, error contents or frames are read.
Only test-owned fixed stage labels may be projected into regression evidence.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time

import pytest
from normal_sft_cold_fixture import close, cloud_data, original_precommit, setup
from test_normal_run_sft_checkpoint import wire_decode, wire_encode
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint, normal_run_parents
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.coding_case_state import SessionCodingCaseFreezeStore
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.control_plane.normal_run_checkpoint import (
    activate_normal_run,
    current_normal_run,
    prepare_normal_run,
)
from amlas.control_plane.preparation_state import SessionPreparationStore
from amlas.tui.chat.engine import ChatEngine


def test_actual_cold_completion_restores_consumed_research_without_refetch(monkeypatch, tmp_path):
    async def scenario():
        initial_patch, replay_patch = pytest.MonkeyPatch(), pytest.MonkeyPatch()
        state = replacement = None
        original_closed = False
        stage = "initial_setup"
        try:
            inert_runtime_setup.__wrapped__(initial_patch)
            state = setup(initial_patch, tmp_path)
            actual_publish = normal_run_parents.publish_parent_graph
            contexts = []

            def publish(store, model_ref, preparation_ref, *, run_id, qualify):
                raw = actual_publish(store, model_ref, preparation_ref, run_id=run_id, qualify=qualify)
                contexts.append(current_normal_run())
                return raw

            initial_patch.setattr(normal_run_parents, "publish_parent_graph", publish)
            stage = "original_submission"
            await original_precommit(state, initial_patch)
            original_context = contexts[0]
            stage = "transport_roundtrip"
            data = wire_decode(json.loads(json.dumps(wire_encode(cloud_data(state, completed=True)))))
            stage = "initial_cleanup"
            await close(state)
            original_closed = True
            initial_patch.undo()
            stage = "replacement_setup"
            inert_runtime_setup.__wrapped__(replay_patch)
            replacement_dir = tmp_path / "fresh-provider"
            replacement_dir.mkdir()
            replacement = setup(replay_patch, replacement_dir, captured=data)
            stage = "replacement_context"
            context = prepare_normal_run(
                original_context.run_id,
                original_context.session_hash,
                transport_session_id=original_context.transport_session_id,
            )
            stage = "replacement_policy"
            policy = runtime._runtime_prompt_policy()
            loader_contract = ["entry"]

            def observed(function, label):
                def call(*args, **kwargs):
                    loader_contract[0] = label
                    result = function(*args, **kwargs)
                    loader_contract[0] = label + "_returned"
                    return result

                return call

            replay_patch.setattr(
                normal_run_checkpoint,
                "_original",
                observed(normal_run_checkpoint._original, "original_association"),
            )
            replay_patch.setattr(
                normal_run_checkpoint,
                "_task_envelope",
                observed(normal_run_checkpoint._task_envelope, "task_envelope"),
            )
            replay_patch.setattr(
                normal_run_checkpoint,
                "_captured_snapshot_messages",
                observed(normal_run_checkpoint._captured_snapshot_messages, "snapshot_contract"),
            )
            replay_patch.setattr(
                normal_run_parents,
                "hydrate_parent_graph",
                observed(normal_run_parents.hydrate_parent_graph, "parent_hydration"),
            )
            for store_type, name, label in (
                (SessionAcquisitionStore, "_restore_original_producers", "data_restore"),
                (SessionAcquisitionStore, "_verified_preparation_source", "data_verify"),
                (SessionModelAcquisitionStore, "_restore_original_producer", "model_restore"),
                (SessionModelAcquisitionStore, "_verified_sft_model_snapshot", "model_verify"),
                (SessionCodingCaseFreezeStore, "_restore_original_producer", "case_restore"),
                (SessionCodingCaseFreezeStore, "_verified", "case_verify"),
                (SessionPreparationStore, "_restore_original_producer", "preparation_restore"),
                (SessionPreparationStore, "_restore_original_selection", "preparation_selection"),
                (SessionPreparationStore, "_verified_sft_snapshot", "preparation_verify"),
            ):
                replay_patch.setattr(store_type, name, observed(getattr(store_type, name), label))
            stage = "original_loader"
            try:
                loaded = context.load_qualified_task(policy=policy)
            except (Exception, pytest.fail.Exception):
                stage = "original_loader_" + loader_contract[0]
                raise

            assert type(loaded) is tuple and len(loaded) == 4
            timings = {}
            started = time.monotonic()
            stage = "restored_engine"
            with activate_normal_run(context):
                engine = ChatEngine(approval_policy=policy, normal_restore=(context, *loaded))
                timings["engine_seconds"] = time.monotonic() - started
                stage = "completed_result"
                result = engine.tool_backend._normal_original_result
                assert result.get("trained") is True and result.get("full_training_complete") is True
                stage = "completion_claim"
                started = time.monotonic()
                engine.claim_normal_completion(context, loaded[0])
                timings["claim_seconds"] = time.monotonic() - started
                stage = "consumed_research_phase"
                research = engine.tool_backend._qualify_research_binding()
                assert research is not None and research["phase"] == "CONSUMED"
                units = set(research["units"])
                assert len(units) == 3 and research["attempted"] == units
                assert set(research["tool_uses"].values()) == units
                assert len(research["tool_uses"]) == 3
                assert research["results"] == set(research["tool_uses"])
                assert research["pending"] is False
                assert research["original_completion_ready"] is True
                stage = "consumed_research_controls"
                from amlas.tui.chat.research_runtime_profile import (
                    _research_profile_prompt,
                    _restore_runtime_research_profile,
                )

                assert "do not fetch or repeat research" in _research_profile_prompt(engine.tool_backend)
                original_policy = engine.tool_backend.approval_policy
                engine.tool_backend.approval_policy = None
                try:
                    with pytest.raises(ValueError):
                        engine.tool_backend._qualify_research_binding()
                finally:
                    engine.tool_backend.approval_policy = original_policy
                selected = os.environ["AMLAS_RUNTIME_RESEARCH_PROFILE"]
                os.environ["AMLAS_RUNTIME_RESEARCH_PROFILE"] = "off"
                try:
                    with pytest.raises(ValueError):
                        engine.tool_backend._qualify_research_binding()
                finally:
                    os.environ["AMLAS_RUNTIME_RESEARCH_PROFILE"] = selected
                claimed = json.loads(engine._normal_completion_snapshot)
                receipt_data = json.loads(context._completion_receipt[1])
                assert type(receipt_data["schema_version"]) is int
                assert receipt_data["schema_version"] == 3
                assert "sdk_snapshot" not in receipt_data
                assert receipt_data["sdk_snapshot_sha256"] == (
                    normal_run_checkpoint._sdk_snapshot_sha256(claimed)
                )
                token = engine._normal_completion[1]
                with pytest.raises(ValueError):
                    _restore_runtime_research_profile(engine, context, loaded[0], token, claimed)
                # Test-owned corruption cannot obtain a phase from altered task DATA.
                for mutation in ("prefix", "native", "snapshot"):
                    original_task = copy.deepcopy(loaded[0])
                    altered_snapshot = copy.deepcopy(claimed)
                    research["phase"] = "UNUSED"
                    engine.tool_backend._research_started = False
                    if mutation == "prefix":
                        original_task["sdk_snapshot"]["data"]["messages"] = original_task["sdk_snapshot"][
                            "data"
                        ]["messages"][:1]
                    elif mutation == "native":
                        original_task["native_job_name"] = "declared-foreign-original"
                    else:
                        altered_snapshot["data"]["messages"][0]["content"] = [
                            {"text": "declared changed history"}
                        ]
                    try:
                        with pytest.raises(ValueError):
                            _restore_runtime_research_profile(
                                engine, context, original_task, token, altered_snapshot
                            )
                    finally:
                        original_receipt = context._completion_receipt
                        corrupted_receipt = {
                            **receipt_data,
                            "sdk_snapshot_sha256": "0" * 64,
                        }
                        object.__setattr__(
                            context,
                            "_completion_receipt",
                            (
                                original_receipt[0],
                                normal_run_checkpoint._canonical(corrupted_receipt),
                                original_receipt[2],
                            ),
                        )
                        try:
                            with pytest.raises(ValueError):
                                _restore_runtime_research_profile(engine, context, loaded[0], token, claimed)
                        finally:
                            object.__setattr__(context, "_completion_receipt", original_receipt)

                        research["phase"] = "CONSUMED"
                        engine.tool_backend._research_started = True
                engine.tool_backend._qualify_research_binding()
            (tmp_path / "fresh-contract-timings.json").write_text(json.dumps(timings, sort_keys=True))
            assert replacement.invoked == []
            assert replacement.native.stops == []
        except (Exception, pytest.fail.Exception):
            raise AssertionError("fresh_replay_contract_refused_" + stage) from None
        finally:
            if replacement is not None:
                await close(replacement)
            replay_patch.undo()
            if state is not None and not original_closed:
                await close(state)
            initial_patch.undo()

    asyncio.run(scenario())
