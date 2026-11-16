"""Fresh replay setup contract with fixed assertions and opaque provider DATA.

No old checkpoint, row, graph, returned TASK, error contents or frames are read.
Only test-owned fixed stage labels may be projected into regression evidence.
"""

from __future__ import annotations

import asyncio
import json
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


@pytest.mark.parametrize("strict", [True, False], ids=["every-record", "record-cadence"])
def test_genuine_original_completion_current_checks_avoid_full_row_redecode(monkeypatch, tmp_path, strict):
    # Interval 0 keeps the exact every-record profile; the default interval is the production record cadence.
    monkeypatch.setattr(normal_run_checkpoint, "_PARENT_RECORD_RECHECK_SECONDS", 0.0 if strict else 1.0)

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
                from contextlib import contextmanager

                from amlas.control_plane import producer_snapshot
                from amlas.control_plane.sft_state import SessionSftStore

                methods = {}
                active_depth = {}

                def measured(name, function):
                    methods[name] = {
                        "entries": 0,
                        "outer_calls": 0,
                        "returned": 0,
                        "total_seconds": 0.0,
                        "max_seconds": 0.0,
                        "errors": {
                            category: 0
                            for category in (
                                "ValueError",
                                "TypeError",
                                "RuntimeError",
                                "AssertionError",
                                "AttributeError",
                                "KeyError",
                                "PermissionError",
                                "TimeoutError",
                                "Exception",
                            )
                        },
                    }
                    active_depth[name] = 0

                    def call(*args, **kwargs):
                        row = methods[name]
                        row["entries"] += 1
                        outer = active_depth[name] == 0
                        if outer:
                            row["outer_calls"] += 1
                        active_depth[name] += 1
                        began = time.monotonic()
                        try:
                            value = function(*args, **kwargs)
                        except Exception as error:
                            categories = {
                                ValueError: "ValueError",
                                TypeError: "TypeError",
                                RuntimeError: "RuntimeError",
                                AssertionError: "AssertionError",
                                AttributeError: "AttributeError",
                                KeyError: "KeyError",
                                PermissionError: "PermissionError",
                                TimeoutError: "TimeoutError",
                            }
                            row["errors"][categories.get(type(error), "Exception")] += 1
                            raise
                        else:
                            row["returned"] += 1
                            return value
                        finally:
                            elapsed = time.monotonic() - began
                            active_depth[name] -= 1
                            if outer:
                                row["total_seconds"] += elapsed
                                row["max_seconds"] = max(row["max_seconds"], elapsed)

                    return call

                for owner, name, label in (
                    (normal_run_checkpoint, "_load", "checkpoint_load"),
                    (normal_run_checkpoint, "_record_data", "record_data"),
                    (normal_run_checkpoint, "_prepared_ref", "prepared_ref"),
                    (normal_run_checkpoint, "_checkpoint_data", "checkpoint_data"),
                    (normal_run_checkpoint, "_original", "checkpoint_original"),
                    (producer_snapshot, "_approved_read", "policy_approved_read"),
                    (replacement.ports, "get_item", "fresh_get_item"),
                    (replacement.ports, "get_caller_identity", "native_role_identity"),
                    (replacement.ports, "describe_table", "native_table_identity"),
                    (SessionAcquisitionStore, "_verified_preparation_source", "data_verified_preparation"),
                    (SessionModelAcquisitionStore, "_verified_model_snapshot", "model_verified_snapshot"),
                    (
                        SessionModelAcquisitionStore,
                        "_verified_sft_model_snapshot",
                        "model_verified_sft_snapshot",
                    ),
                    (SessionCodingCaseFreezeStore, "_verified", "case_verified_snapshot"),
                    (SessionPreparationStore, "_verified_sft_snapshot", "preparation_verified_sft_snapshot"),
                    (SessionSftStore, "observe", "sft_observe"),
                    (type(context), "qualify_original_completion", "original_completion_proof"),
                ):
                    replay_patch.setattr(owner, name, measured(label, getattr(owner, name)))
                actual_native = normal_run_checkpoint._native
                measured("native_scope", lambda action: action())

                @contextmanager
                def native(*args, **kwargs):
                    row = methods["native_scope"]
                    row["entries"] += 1
                    row["outer_calls"] += 1
                    began = time.monotonic()
                    try:
                        with actual_native(*args, **kwargs) as unchanged:
                            yield unchanged
                    except Exception as error:
                        categories = {
                            ValueError: "ValueError",
                            TypeError: "TypeError",
                            RuntimeError: "RuntimeError",
                            AssertionError: "AssertionError",
                            AttributeError: "AttributeError",
                            KeyError: "KeyError",
                            PermissionError: "PermissionError",
                            TimeoutError: "TimeoutError",
                        }
                        row["errors"][categories.get(type(error), "Exception")] += 1
                        raise
                    else:
                        row["returned"] += 1
                    finally:
                        elapsed = time.monotonic() - began
                        row["total_seconds"] += elapsed
                        row["max_seconds"] = max(row["max_seconds"], elapsed)

                replay_patch.setattr(normal_run_checkpoint, "_native", native)
                original_callback = measured("fresh_original_callback", engine._normal_completion[2])
                stage = "explicit_current_qualifier"
                started = time.monotonic()
                context.qualify_completion(
                    engine._normal_completion[1],
                    policy=engine.tool_backend.approval_policy,
                    qualify_original=original_callback,
                    sdk_snapshot=json.loads(engine._normal_completion_snapshot),
                )
                qualifier_seconds = time.monotonic() - started
                assert replacement.invoked == [] and replacement.native.stops == []
                (tmp_path / "fresh-original-current-decode-budget.json").write_text(
                    json.dumps(
                        {
                            "explicit_calls": 1,
                            "qualifier_seconds": qualifier_seconds,
                            "methods": methods,
                            "task_dimensions_recorded": False,
                            "authority_preserved": True,
                            "sdk_invocations": len(replacement.invoked),
                        },
                        sort_keys=True,
                    )
                )
                stage = "full_row_decode_budget"
                # Schema 3 adds four authenticated TASK current checks: before/after
                # get_object, after body.read, and after snapshot digest validation.
                # Each retains one policy/checkpoint and three role/table checks. The lineage reader asks for 1 MiB
                # reads, so these counts are 744 current checks (186 read chunks x 4 brackets) below the 64 KiB profile.
                if strict:
                    assert methods["checkpoint_load"]["entries"] == 13839
                    assert methods["native_role_identity"]["entries"] == 41508
                    assert methods["native_table_identity"]["entries"] == 41500
                    assert methods["policy_approved_read"]["entries"] == 13814
                else:
                    # Record cadence: full checks at every read bracket plus about one per second, so the
                    # native round trips no longer grow with the number of lineage records.
                    # Measured 1,839 loads against 13,839 in the every-record profile; the remainder is the
                    # read brackets, about one full check per second and callbacks outside the lineage reader.
                    assert 0 < methods["checkpoint_load"]["entries"] < 2500
                    assert 0 < methods["native_role_identity"]["entries"] < 7500
                    assert 0 < methods["native_table_identity"]["entries"] < 7500
                    # Policy and owner checks still run for every record.
                    assert methods["policy_approved_read"]["entries"] > 10000
                assert methods["fresh_get_item"]["entries"] >= methods["checkpoint_load"]["entries"]
                assert methods["record_data"]["entries"] * 2 < methods["fresh_get_item"]["entries"], (
                    "original current checks repeated full snapshot and history decode"
                )
        except (Exception, pytest.fail.Exception) as error:
            categories = {
                ValueError: "ValueError",
                TypeError: "TypeError",
                RuntimeError: "RuntimeError",
                AssertionError: "AssertionError",
                AttributeError: "AttributeError",
                KeyError: "KeyError",
                PermissionError: "PermissionError",
                TimeoutError: "TimeoutError",
            }
            raise AssertionError(
                "fresh_qualifier_profile_refused_" + stage + "_" + categories.get(type(error), "Exception")
            ) from None
        finally:
            if replacement is not None:
                await close(replacement)
            replay_patch.undo()
            if state is not None and not original_closed:
                await close(state)
            initial_patch.undo()

    asyncio.run(scenario())
