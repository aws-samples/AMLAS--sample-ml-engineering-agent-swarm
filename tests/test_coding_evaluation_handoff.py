"""Private frozen-case handoff for the source-owned evaluator; no execution grant."""

from __future__ import annotations

import dataclasses

import pytest
from test_coding_case_tools import ACQ
from test_coding_case_tools import state as state

from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane import coding_case_state
from amlas.data_eng.coding_case_codec import FILES


def test_original_case_evaluation_handoff_is_fresh_frozen_and_complete(state):
    store, backend, _reads = state
    ref = store.submit(ACQ)["case_ref"]
    accessor = getattr(store, "_verified_evaluation_source", None)
    assert callable(accessor), "private evaluator case accessor missing"
    before = backend.reads
    result = accessor(ref)
    expected_type = getattr(coding_case_state, "VerifiedCaseEvaluationSource", None)
    assert expected_type is not None and type(result) is expected_type
    assert backend.reads == before + 1
    handle = store._records[ref]
    assert result.runtime_session_id == store.runtime_session_id
    assert result.producer_job_arn == handle.arn
    assert result.producer_header == tuple(sorted(handle.extra["header"].items()))
    assert result.original_source is handle.extra["source"]
    assert set(dict(result.file_refs)) == set(FILES)
    assert all(type(value) is DatasetArtifactRef for _name, value in result.file_refs)
    assert "ORACLES.jsonl" in dict(result.file_refs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.runtime_session_id = "different"
    public = store.observe(ref)
    assert "ORACLES" not in str(public) and "file_refs" not in public
    assert public["execution_status"] == "not_executed"


@pytest.mark.parametrize("bad", [None, {}, "case_" + "f" * 32])
def test_unknown_or_public_case_input_refuses_before_backend_read(state, bad):
    store, backend, _reads = state
    accessor = getattr(store, "_verified_evaluation_source", None)
    assert callable(accessor), "private evaluator case accessor missing"
    before = backend.reads
    with pytest.raises(ValueError):
        accessor(bad)
    assert backend.reads == before


def test_unverified_case_cannot_supply_evaluator_files(state, monkeypatch):
    store, backend, _reads = state
    ref = store.submit(ACQ)["case_ref"]
    accessor = getattr(store, "_verified_evaluation_source", None)
    assert callable(accessor), "private evaluator case accessor missing"
    monkeypatch.setattr(backend, "artifacts", lambda _handle: {"success": False, "frozen": False})
    with pytest.raises(ValueError):
        accessor(ref)


def test_private_case_roles_revalidate_current_versions_before_handoff(state, monkeypatch):
    store, backend, _reads = state
    ref = store.submit(ACQ)["case_ref"]
    accessor = getattr(store, "_verified_evaluation_source", None)
    assert callable(accessor), "private evaluator case accessor missing"
    original = backend.artifacts

    def changed(handle):
        result = original(handle)
        result["files"]["ORACLES.jsonl"] = dataclasses.replace(
            result["files"]["ORACLES.jsonl"], version_id="null"
        )
        return result

    monkeypatch.setattr(backend, "artifacts", changed)
    with pytest.raises(ValueError):
        accessor(ref)
