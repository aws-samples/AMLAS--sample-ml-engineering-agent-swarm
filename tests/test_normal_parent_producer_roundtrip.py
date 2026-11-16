"""Fresh registered producer handles compose through their Source wire codecs."""

from __future__ import annotations

import asyncio

import pytest
from normal_sft_cold_fixture import close, original_precommit, setup
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.control_plane.campaign_state import _binding_fields
from amlas.control_plane.producer_snapshot import _capture, _producer


@pytest.mark.parametrize("kind", ("model", "preparation"))
def test_fresh_registered_parent_producer_roundtrip(monkeypatch, tmp_path, kind):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            stores = {
                "model": state.backend._model_acquisition_store,
                "preparation": state.backend._preparation_store,
            }
            store = stores[kind]
            assert len(store._records) == 1
            alias, original = next(iter(store._records.items()))
            handle = original.handle
            scope = _binding_fields(state.backend._sft_store._execution_binding)
            before_model = state.model.calls
            before_native = len(state.native.calls)
            before_loaded = len(state.loaded)
            before_invoked = len(state.invoked)
            try:
                wire = _capture(handle, alias, model=kind == "model")
                restored = _producer(wire, scope, model=kind == "model")
                recaptured = _capture(restored, alias, model=kind == "model")
                assert recaptured == wire
                assert type(restored) is type(handle)
                assert restored.id == handle.id
                assert restored.arn == handle.arn
                assert restored.backend == handle.backend
            except Exception:
                if kind == "model":
                    raise AssertionError("fresh_model_producer_roundtrip_refused") from None
                raise AssertionError("fresh_preparation_producer_roundtrip_refused") from None
            assert state.model.calls == before_model
            assert len(state.native.calls) == before_native
            assert len(state.loaded) == before_loaded
            assert len(state.invoked) == before_invoked
            assert state.native.stops == []
        finally:
            await close(state)

    asyncio.run(scenario())
