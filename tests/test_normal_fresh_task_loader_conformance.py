"""Fresh original public TASK loader Source conformance, no private data output.

Context comes from the actual initial publisher; all external ports remain inert.
Returned TASK/parent/original data are neither inspected nor exported.
"""

from __future__ import annotations

import asyncio

from normal_sft_cold_fixture import close, original_precommit, setup
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.control_plane import normal_run_parents
from amlas.control_plane.normal_run_checkpoint import current_normal_run


def test_fresh_original_task_loader_composes(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        actual_publish = normal_run_parents.publish_parent_graph
        contexts = []

        def publish(store, model_ref, preparation_ref, *, run_id, qualify):
            raw = actual_publish(store, model_ref, preparation_ref, run_id=run_id, qualify=qualify)
            contexts.append(current_normal_run())
            return raw

        monkeypatch.setattr(normal_run_parents, "publish_parent_graph", publish)
        try:
            await original_precommit(state, monkeypatch)
            assert len(contexts) == 1
            try:
                loaded = contexts[0].load_qualified_task(policy=state.backend.approval_policy)
                assert type(loaded) is tuple and len(loaded) == 4
            except Exception:
                raise AssertionError("fresh_original_task_loader_refused") from None
            assert len(state.native.calls) == 1 and state.native.stops == []
            assert state.loaded == [] and state.invoked == []
        finally:
            await close(state)

    asyncio.run(scenario())
