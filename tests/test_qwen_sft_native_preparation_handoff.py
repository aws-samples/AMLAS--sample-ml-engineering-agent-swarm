"""Real preparation success protocol reaches private SFT source without public merge."""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import pytest
from test_managed_coding_preparation import fixture as base_fixture
from test_managed_commitpack_preparation import complete_outputs, staged_fixture

from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.preparation_state import SessionPreparationStore, _PreparationRecord


@pytest.fixture
def native_prepared(monkeypatch):
    value = staged_fixture(base_fixture.__wrapped__(monkeypatch), monkeypatch)
    adapter, _sm, _s3, catalog, source = value
    handle = adapter.submit(sources=(source,), catalog=catalog, runtime_session_id="test_session")
    complete_outputs(value, handle)
    acquired = SessionAcquisitionStore()
    acquired.runtime_session_id = "test_session"
    store = SessionPreparationStore(acquired, adapter, catalog)
    ref = "prep_" + "a" * 32
    store._records[ref] = _PreparationRecord(adapter, handle)
    return store, ref, value, handle


def frozen_result(native_prepared, monkeypatch, *, typed=False):
    store, ref, value, handle = native_prepared
    adapter = value[0]
    result = adapter.artifacts(handle)
    assert result["success"] is True and result["prepared"] is True
    # Isolate the separate exact-typed-ref compatibility boundary; these fields
    # are the required producer protocol, not authority reconstructed from public JSON.
    result.update(job_name=handle.id, job_arn=handle.arn)
    if typed:
        result["preparation_ref"] = DatasetArtifactRef(**result["preparation_ref"])
        for key in ("split_refs", "sidecar_refs"):
            result[key] = {name: DatasetArtifactRef(**item) for name, item in result[key].items()}
    reads = []

    def artifacts(observed):
        assert observed is handle
        reads.append(observed)
        return copy.deepcopy(result)

    monkeypatch.setattr(adapter, "artifacts", artifacts)
    return store, ref, result, reads, handle


def test_real_native_commitpack_success_supplies_original_job_identity(native_prepared):
    _store, _ref, value, handle = native_prepared
    result = value[0].artifacts(handle)
    assert result["success"] is True
    assert result.get("job_name") == handle.id
    assert result.get("job_arn") == handle.arn
    assert type(result["preparation_ref"]) is dict  # measured current native protocol


def test_private_handoff_consumes_real_adapter_without_public_observe(native_prepared):
    store, ref, value, handle = native_prepared
    snapshot = store._verified_sft_snapshot(ref)
    assert snapshot.producer_job_arn == handle.arn
    assert dict(snapshot.header) == handle.extra["header"]
    assert type(snapshot.preparation_ref) is DatasetArtifactRef
    assert len(value[1].creates) == 1


@pytest.mark.parametrize("typed", [False, True])
def test_closed_dict_and_exact_typed_internal_refs_retain_one_fresh_observation(
    native_prepared, monkeypatch, typed
):
    store, ref, result, reads, handle = frozen_result(native_prepared, monkeypatch, typed=typed)
    before = copy.deepcopy(result)
    snapshot = store._verified_sft_snapshot(ref)
    assert type(snapshot.preparation_ref) is DatasetArtifactRef
    assert snapshot.producer_job_arn == handle.arn
    assert reads == [handle] and result == before
    assert snapshot.exclusions_ref == handle.extra["catalog"].exclusions_ref


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        {"version_id": None},
        {"version_id": "null"},
        {"sha256": "short"},
        {"byte_count": True},
        {"byte_count": 0},
        {"byte_count": 256 * 1024**2 + 1},
        {"format": "json"},
        {"uri": "s3://foreign-bucket/sessions/test_session/train.jsonl"},
        {"uri": "s3://owned-bucket/sessions/foreign/train.jsonl"},
    ],
)
def test_malformed_internal_ref_refuses_after_original_native_observation(
    native_prepared, monkeypatch, typed, change
):
    store, ref, result, reads, handle = frozen_result(native_prepared, monkeypatch, typed=typed)
    value = result["split_refs"]["train"]
    result["split_refs"]["train"] = dataclasses.replace(value, **change) if typed else {**value, **change}
    with pytest.raises(ValueError):
        store._verified_sft_snapshot(ref)
    assert reads == [handle]


@pytest.mark.parametrize("change", ["same_session_other_job", "wrong_role_name", "extra_grant", "ducktype"])
def test_wrong_role_or_producer_path_and_nonclosed_types_refuse(native_prepared, monkeypatch, change):
    store, ref, result, _reads, _handle = frozen_result(native_prepared, monkeypatch)
    value = result["split_refs"]["train"]
    if change == "same_session_other_job":
        value["uri"] = value["uri"].replace("/preparation/", "/other-preparation/")
    elif change == "wrong_role_name":
        value["uri"] = value["uri"].removesuffix("train.jsonl") + "REPORT.json"
    elif change == "extra_grant":
        value["ready"] = True
    else:
        result["split_refs"]["train"] = SimpleNamespace(**value)
    with pytest.raises(ValueError):
        store._verified_sft_snapshot(ref)


@pytest.mark.parametrize("field", ["job_name", "job_arn"])
def test_foreign_observed_identity_is_not_replaced_by_private_handle(native_prepared, monkeypatch, field):
    store, ref, result, reads, handle = frozen_result(native_prepared, monkeypatch)
    result[field] += "-foreign"
    with pytest.raises(ValueError, match="native identity"):
        store._verified_sft_snapshot(ref)
    assert reads == [handle]


@pytest.mark.parametrize("field", ["deployment_id", "runtime_session_id", "image_uri", "source_commit"])
def test_original_header_drift_refuses_even_if_result_claims_success(native_prepared, monkeypatch, field):
    store, ref, _result, _reads, handle = frozen_result(native_prepared, monkeypatch)
    handle.extra["header"][field] += "-changed"
    with pytest.raises(ValueError):
        store._verified_sft_snapshot(ref)


def test_foreign_session_denies_before_any_new_adapter_read(native_prepared, monkeypatch):
    store, ref, _result, reads, _handle = frozen_result(native_prepared, monkeypatch)
    store.runtime_session_id = "foreign_session"
    with pytest.raises(ValueError):
        store._verified_sft_snapshot(ref)
    assert reads == []


@pytest.mark.parametrize("kind", ["typed_subclass", "dict_subclass"])
def test_ref_subclasses_do_not_enter_the_private_exact_type_contract(native_prepared, monkeypatch, kind):
    store, ref, result, _reads, _handle = frozen_result(native_prepared, monkeypatch)
    value = result["split_refs"]["train"]
    if kind == "typed_subclass":

        class RefSubclass(DatasetArtifactRef):
            pass

        result["split_refs"]["train"] = RefSubclass(**value)
    else:

        class DictSubclass(dict):
            pass

        result["split_refs"]["train"] = DictSubclass(value)
    with pytest.raises(ValueError, match="ref shape"):
        store._verified_sft_snapshot(ref)
