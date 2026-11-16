"""Deterministic readonly TASK-addition RED; actual Source writer and originals.

The Source SDK creates every original and prepared snapshot. Its SFT precommit
callback is held while the genuine prepared TASK is promoted during the journal read.
No TASK/ref/SDK graph is fabricated. Native ports are inert Source evidence only.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import threading

import pytest
from normal_sft_cold_fixture import SFT_USE, close, prepare_original_submission, setup, wait_for

# Initialize the owning fixture before product imports.
# isort: split
from amlas.control_plane import normal_run_checkpoint as normal
from amlas.tui.chat import tool_binding_backend as binding_backend
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


async def run_case(monkeypatch, tmp_path, mode):
    state = setup(monkeypatch, tmp_path)
    actual_capture = normal.NormalRunContext.capture_sft_call
    actual_complete = ToolBindingBackend._complete_original_before_commit
    release = threading.Event()
    capture = {}

    def capture_source(context, *args, **kwargs):
        result = actual_capture(context, *args, **kwargs)
        capture["context"] = context
        return result

    async def held_sft_precommit(backend, event, witness):
        with backend._warm_capture_lock:
            selected = [
                pending
                for pending in backend._warm_pending_producers.values()
                if type(pending) is binding_backend._PendingOriginalProducer
                and pending.kind == "sft"
                and pending.agent is event.agent
                and pending.tool_use_id == event.tool_use.get("toolUseId") == SFT_USE
            ]
        if len(selected) == 1:
            deadline = asyncio.get_running_loop().time() + 60
            while not release.is_set():
                assert asyncio.get_running_loop().time() < deadline, "fixture SFT handoff not released"
                await asyncio.sleep(0.005)
        return await actual_complete(backend, event, witness)

    monkeypatch.setattr(normal.NormalRunContext, "capture_sft_call", capture_source)
    monkeypatch.setattr(ToolBindingBackend, "_complete_original_before_commit", held_sft_precommit)
    try:
        _reader, _events, backend = await prepare_original_submission(state, monkeypatch)
        await wait_for(
            state,
            state.reader,
            lambda: (
                capture
                and backend._sft_store._records
                and any(value.kind == "sft" for value in backend._warm_pending_producers.values())
            ),
            "accepted Source SFT and captured prepared TASK",
            timeout=60,
        )
        context = capture["context"]
        with normal.activate_normal_run(context), normal._native(context) as (ddb, table_id, qualify, _sts):
            _item, before = normal._load(context, ddb, table_id, qualify)
            original, binding = normal._original(context, before, table_id)
            assert "prepared_checkpoint_ref" in before and "task_checkpoint_ref" not in before
            assert len(state.native.calls) == 1
            job_name = original.native_job_name
            journal = normal._Journal(context, table_id)
            # Mint the existing binding before selecting the injected read. It
            # is verified Source state, not constructed by this fixture.
            current_binding = journal.bind_execution(
                campaign_id=binding.campaign_id, runtime_session_id=binding.runtime_session_id
            )
            if mode in {"task-replaced", "task-removed"}:
                # Existing TASK is created by the actual Source writer before
                # the next journal baseline; no pointer is seeded by this test.
                context.promote_prepared_task(policy=backend.approval_policy)
            original_get = state.ports.get_item
            injected = []
            writing = False

            def is_guarded_write():
                frame = inspect.currentframe().f_back
                while frame is not None:
                    if frame.f_code.co_name == "call" and frame.f_code.co_filename.endswith(
                        "normal_run_checkpoint.py"
                    ):
                        return frame.f_locals.get("name") in {"put_item", "update_item"}
                    frame = frame.f_back
                return False

            def get_item(**kwargs):
                nonlocal writing
                key = kwargs["Key"]["job_name"]["S"]
                chosen = (
                    key == job_name
                    if mode != "write-task"
                    else key.startswith("PROMPT_RUN#") and is_guarded_write()
                )
                if not injected and not writing and chosen:
                    writing = True
                    try:
                        context.promote_prepared_task(policy=backend.approval_policy)
                        with state.ports.lock:
                            row = copy.deepcopy(state.ports.items["PROMPT_RUN#" + context.run_id])
                        after = normal._record_data(row["record"]["S"])
                        assert "task_checkpoint_ref" in after
                        assert after == {**before, "task_checkpoint_ref": after["task_checkpoint_ref"]}
                        injected.append("source_TASK_CAS")
                        if mode in {"changed-original", "task-replaced", "task-removed", "history-changed"}:
                            # Negative provider DATA only, after genuine Source
                            # TASK publication. No grant/original/SDK row seeded.
                            if mode == "changed-original":
                                after["private_store_correlation"] = (
                                    "3" * 8 + "-" + "3" * 4 + "-4" + "3" * 3 + "-8" + "3" * 3 + "-" + "3" * 12
                                )
                            elif mode == "task-replaced":
                                ref = dict(after["task_checkpoint_ref"])
                                first = "0" if ref["sha256"][0] != "0" else "1"
                                ref["sha256"] = first + ref["sha256"][1:]
                                after["task_checkpoint_ref"] = ref
                            elif mode == "task-removed":
                                after.pop("task_checkpoint_ref")
                            else:
                                after["history"] = [
                                    *after["history"],
                                    {
                                        "role": "user",
                                        "content": [{"text": "Unrelated declared history changed."}],
                                    },
                                ]
                            raw = normal._canonical(after)
                            row["record"] = {"S": raw}
                            row["record_sha256"] = {
                                "S": __import__("hashlib").sha256(raw.encode()).hexdigest()
                            }
                            state.ports.items["PROMPT_RUN#" + context.run_id] = row
                    finally:
                        writing = False
                return original_get(**kwargs)

            monkeypatch.setattr(state.ports, "get_item", get_item)
            if mode == "read-task":
                recovered = journal.load_original(current_binding, job_name)
                assert (
                    recovered.native_job_name == job_name
                    and recovered.native_job_arn == original.native_job_arn
                )
                assert recovered.operation_digest == original.operation_digest
            elif mode == "write-task":
                with pytest.raises(ValueError):
                    journal.reserve_original_job(current_binding, original.intent)
            else:
                with pytest.raises(ValueError):
                    journal.load_original(current_binding, job_name)
            assert injected == ["source_TASK_CAS"]
            assert len(state.native.calls) == 1 and state.native.stops == []
    finally:
        release.set()
        await close(state)


def test_readonly_original_load_tolerates_genuine_same_original_task_addition(monkeypatch, tmp_path):
    asyncio.run(run_case(monkeypatch, tmp_path, "read-task"))


def test_writing_reservation_keeps_strict_whole_record_fence_on_task_addition(monkeypatch, tmp_path):
    asyncio.run(run_case(monkeypatch, tmp_path, "write-task"))


def test_readonly_load_still_refuses_changed_original_private_correlation(monkeypatch, tmp_path):
    asyncio.run(run_case(monkeypatch, tmp_path, "changed-original"))


def test_readonly_load_refuses_replacement_of_previously_published_task_ref(monkeypatch, tmp_path):
    asyncio.run(run_case(monkeypatch, tmp_path, "task-replaced"))


def test_readonly_load_refuses_removal_of_previously_published_task_ref(monkeypatch, tmp_path):
    asyncio.run(run_case(monkeypatch, tmp_path, "task-removed"))


def test_readonly_load_refuses_unrelated_history_change_with_genuine_task_addition(monkeypatch, tmp_path):
    asyncio.run(run_case(monkeypatch, tmp_path, "history-changed"))
