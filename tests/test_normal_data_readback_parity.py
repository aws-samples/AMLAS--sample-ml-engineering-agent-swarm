"""Declared warm/cold DATA providers project the same original native request.

Fresh synthetic inputs only. Exact Source parent-field comparison is asserted;
no persisted row, checkpoint or private graph is inspected/exported.
"""

from __future__ import annotations

import asyncio
import copy

from normal_sft_cold_fixture import NativeJobs, close, original_precommit, setup
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.control_plane.normal_run_parents import _DATA_NATIVE_KEYS


def test_fresh_data_original_describe_parity(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            for record in state.backend._acquisition_store._records.values():
                original = state.data.native.requests[record.handle.id]
                cold = NativeJobs()
                cold.processing[record.handle.id] = copy.deepcopy(original)
                cold.states[record.handle.id] = state.data.native.states[record.handle.id]
                warm = state.data.native.describe_processing_job(ProcessingJobName=record.handle.id)
                fresh = cold.describe_processing_job(ProcessingJobName=record.handle.id)
                assert {key: warm.get(key) for key in _DATA_NATIVE_KEYS} == {
                    key: fresh.get(key) for key in _DATA_NATIVE_KEYS
                }, "fresh_data_original_readback_parity_refused"
            assert len(state.native.calls) == 1 and state.native.stops == []
        finally:
            await close(state)

    asyncio.run(scenario())
