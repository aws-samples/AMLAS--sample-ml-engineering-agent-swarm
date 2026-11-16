"""Real registered preparation selectors reach the unchanged normal SFT guard.

Strict neutral SageMaker/S3 ports replace only expensive external I/O. No AWS,
training, manually inserted preparation record or fixed preparation selector is
used. The actual registered submission tool creates every positive opaque ref.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from test_coding_preparation_binding import REF, Acquisition, auto, sdk_invoke
from test_managed_coding_preparation import fixture as base_fixture
from test_managed_commitpack_preparation import complete_outputs, staged_fixture
from test_normal_sft_public_defaults_capture import MODEL_REF

from amlas.backends.adapters._util import raw_tool
from amlas.control_plane.normal_run_checkpoint import _effective_sft_inputs
from amlas.control_plane.preparation_state import SessionPreparationStore
from amlas.tools.coding_preparation import get_coding_preparation


@pytest.fixture
def original_preparation(monkeypatch):
    value = staged_fixture(base_fixture.__wrapped__(monkeypatch), monkeypatch)
    adapter, _native, _objects, catalog, source = value
    acquired = Acquisition()
    acquired.runtime_session_id = source.inspection_source.runtime_session_id
    acquired.sources = {REF: source}
    store = SessionPreparationStore(acquired, adapter, catalog)
    submitted = sdk_invoke(store, auto())
    assert submitted["accepted"] is True and submitted["status"] == "submitted"
    reference = submitted["request_ref"]
    assert len(value[1].creates) == 1
    handle = store._records[reference].handle
    context = SimpleNamespace(invocation_state={"_amlas_preparation_store": store})
    return store, value, handle, submitted, context


@pytest.mark.parametrize("stage", ["submitted", "reused", "running", "completed"])
def test_registered_preparation_exposes_same_original_sft_selector(original_preparation, stage):
    store, value, handle, submitted, context = original_preparation
    artifact_before = None
    if stage == "submitted":
        result = submitted
    elif stage == "reused":
        result = sdk_invoke(store, auto())
        assert result["reused"] is True
    else:
        if stage == "completed":
            complete_outputs(value, handle)
            artifact_before = copy.deepcopy(value[0].artifacts(handle))
        result = raw_tool(get_coding_preparation)(submitted["request_ref"], tool_context=context)

    assert result["request_ref"] == submitted["request_ref"]
    assert result["prepared"] is (stage == "completed")
    assert len(value[1].creates) == 1
    if artifact_before is not None:
        assert type(result["preparation_ref"]) is dict
        assert result["preparation_ref"] == artifact_before["preparation_ref"]
        assert result["split_refs"] == artifact_before["split_refs"]
        assert result["sidecar_refs"] == artifact_before["sidecar_refs"]

    assert "preparation_request_ref" in result, "owned producer must expose its opaque SFT selector"
    assert result["preparation_request_ref"] == result["request_ref"]
    inputs = {
        "model_request_ref": MODEL_REF,
        "preparation_request_ref": result["preparation_request_ref"],
        "recipe_id": "full_v1",
    }
    before = copy.deepcopy(inputs)
    effective = _effective_sft_inputs(inputs)
    assert effective["preparation_request_ref"] == submitted["request_ref"]
    assert inputs == before
    assert store._records[submitted["request_ref"]].handle is handle
    assert len(value[1].creates) == 1


@pytest.mark.parametrize("selection", ["artifact", "artifact_uri", "recipe"])
def test_artifact_and_recipe_are_not_opaque_sft_authority(original_preparation, selection):
    store, value, handle, submitted, context = original_preparation
    complete_outputs(value, handle)
    observed = raw_tool(get_coding_preparation)(submitted["request_ref"], tool_context=context)
    artifact = observed["preparation_ref"]
    invalid = {"artifact": artifact, "artifact_uri": artifact["uri"], "recipe": "full_v1"}[selection]
    inputs = {"model_request_ref": MODEL_REF, "preparation_request_ref": invalid}
    before = copy.deepcopy(inputs)
    predicates = []
    with pytest.raises(ValueError, match="original SFT tool selection differs"):
        _effective_sft_inputs(inputs, _diagnostic_predicate=predicates.append)
    assert predicates == ["PREPARATION_REF"]
    assert inputs == before
    assert store._records[submitted["request_ref"]].handle is handle
    assert len(value[1].creates) == 1


def test_foreign_store_cannot_resolve_original_selector_or_gain_alias(original_preparation):
    store, value, handle, submitted, _context = original_preparation
    acquired = Acquisition()
    foreign = SessionPreparationStore(acquired, value[0], value[3])
    result = raw_tool(get_coding_preparation)(
        submitted["request_ref"],
        tool_context=SimpleNamespace(invocation_state={"_amlas_preparation_store": foreign}),
    )
    assert result["status"] == "unknown_reference"
    assert result["accepted"] is False and result["prepared"] is False
    assert "preparation_request_ref" not in result
    assert "request_ref" not in result
    with pytest.raises(ValueError, match="unknown or foreign original SFT preparation ref"):
        foreign._verified_sft_snapshot(submitted["request_ref"])
    assert store._records[submitted["request_ref"]].handle is handle
    assert len(value[1].creates) == 1


def test_missing_fresh_policy_cannot_expose_new_selector(original_preparation):
    store, value, handle, submitted, _context = original_preparation
    denied = sdk_invoke(store, None)
    assert denied["status"] == "denied" and denied["accepted"] is False
    assert "preparation_request_ref" not in denied and "request_ref" not in denied
    assert store._records[submitted["request_ref"]].handle is handle
    assert len(value[1].creates) == 1
