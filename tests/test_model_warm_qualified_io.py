"""Real model adapter nested qualification; neutral reads and no native providers."""

from __future__ import annotations

import pytest

from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend


def test_nested_model_callbacks_both_guard_each_original_read():
    adapter = ProcessingModelWeightAcquisitionBackend(region="us-west-2")
    seen = []

    def read():
        seen.append("read")
        return {"ordinary": True}

    with adapter.qualified_io(lambda: seen.append("outer")):
        with adapter.qualified_io(lambda: seen.append("inner")):
            assert adapter._sdk_io(read) == {"ordinary": True}
    assert seen == ["outer", "inner", "read", "outer", "inner"]


def test_inner_model_none_cannot_erase_original_outer_guard():
    adapter = ProcessingModelWeightAcquisitionBackend(region="us-west-2")
    seen = []
    with adapter.qualified_io(lambda: seen.append("outer")):
        with adapter.qualified_io(None):
            adapter._sdk_io(lambda: seen.append("read") or {})
    assert seen == ["outer", "read", "outer"]


def test_outer_revocation_during_inner_scope_blocks_next_original_sdk_read():
    adapter = ProcessingModelWeightAcquisitionBackend(region="us-west-2")
    state, reads = {"current": True}, []

    def outer():
        if not state["current"]:
            raise ValueError("ordinary current policy refused")

    with adapter.qualified_io(outer):
        with adapter.qualified_io(lambda: None):
            adapter._sdk_io(lambda: reads.append("first") or {})
            state["current"] = False
            with pytest.raises(ValueError):
                adapter._sdk_io(lambda: reads.append("second") or {})
    assert reads == ["first"]
    # Original lexical scopes reset; no qualifier/grant survives either scope.
    assert adapter._qualified_io.get() is None
