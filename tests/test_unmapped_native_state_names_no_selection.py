"""An UNKNOWN observation names no selected instance, also when the native state is not in the adapter's map.

`SageMakerQwenSftBackend.status` maps InProgress, Stopping, Completed, Failed and Stopped. Any other
TrainingJobStatus (SageMaker also defines Deleting; a later API may add more; a description may lack the field)
reads UNKNOWN. Such a description can still carry SelectedInstanceType and SelectedInstanceCount of a preference-list
job. An UNKNOWN observation is not a resolved state of the job, so its status detail and its observation record must
not name a selected instance, in the SFT adapter, the generation adapter and the two stores. Mapped states keep
naming it. Strict installed-SDK fakes, no AWS call.
"""

from __future__ import annotations

import pytest
from test_capacity_preference_lists import QuotaNative
from test_generation_capacity_rejection_retry import _real_adapter
from test_managed_generation_product import (  # noqa: F401  (pytest fixture)
    CASE_REF,
    MODEL_REF,
    state,
)
from test_qwen_sft_binding import real_store
from test_qwen_sft_durable_submission import durable, injected_store, submit

from amlas.backends.types import JobState

SELECTED = {"SelectedInstanceType", "SelectedInstanceCount"}
# TrainingJobStatus values outside the adapter's map: a real one, a possible later one, a lower-case one, and no value.
UNMAPPED = ["Deleting", "Paused", "completed", "", None]
MAPPED = {
    "InProgress": JobState.RUNNING,
    "Stopping": JobState.RUNNING,
    "Completed": JobState.SUCCEEDED,
    "Failed": JobState.FAILED,
    "Stopped": JobState.STOPPED,
}


def _described_state(native, value):
    """Make the strict SFT fake describe `value` as TrainingJobStatus (the field is absent for None)."""
    describe = native.describe_training_job

    def described(**kw):
        result = describe(**kw)
        assert SELECTED <= set(result["ResourceConfig"])
        if value is None:
            del result["TrainingJobStatus"]
        else:
            result["TrainingJobStatus"] = value
        return result

    native.describe_training_job = described


@pytest.mark.parametrize("native_state", UNMAPPED)
def test_sft_status_and_artifacts_of_an_unmapped_native_state_name_no_selection(native_state):
    p = durable()
    h = submit(p)
    _described_state(p[1], native_state)
    observed = p[0].status(h)
    assert observed.state is JobState.UNKNOWN
    assert "Selected" not in observed.detail and observed.detail == ""
    # The description itself was read and validated: this is the unmapped-state path, not a failed observation.
    assert observed.raw.get("TrainingJobStatus") == native_state and "error_type" not in observed.raw
    result = p[0].artifacts(h)
    assert result["status"] == "UNKNOWN" and result["success"] is False and result["trained"] is False
    assert not SELECTED & set(result)


@pytest.mark.parametrize("native_state,expected", sorted(MAPPED.items()))
def test_sft_status_of_a_mapped_native_state_still_names_the_selection(native_state, expected):
    p = durable()
    h = submit(p)
    _described_state(p[1], native_state)
    observed = p[0].status(h)
    assert observed.state is expected
    assert observed.detail == "SelectedInstanceType=ml.g6e.16xlarge SelectedInstanceCount=1"


@pytest.mark.parametrize("native_state", ["InProgress", "Stopping", "Failed", "Stopped"])
def test_sft_artifacts_of_a_mapped_unfinished_state_still_name_the_selection(native_state):
    p = durable()
    h = submit(p)
    _described_state(p[1], native_state)
    result = p[0].artifacts(h)
    assert result["status"] == MAPPED[native_state].value
    assert (result["SelectedInstanceType"], result["SelectedInstanceCount"]) == ("ml.g6e.16xlarge", 1)


@pytest.mark.parametrize("native_state", UNMAPPED)
def test_generation_observation_of_an_unmapped_native_state_names_no_selection(state, native_state):  # noqa: F811
    native = QuotaNative()
    store, spec = _real_adapter(state, native)
    base = store.submit(MODEL_REF, CASE_REF, spec)
    running = store.observe(base["request_ref"])
    assert running["status"] == "RUNNING" and running["SelectedInstanceType"] == "ml.g6e.16xlarge"
    if native_state is None:
        describe = native.describe_training_job
        native.describe_training_job = lambda **kw: {
            key: value for key, value in describe(**kw).items() if key != "TrainingJobStatus"
        }
    else:
        native.states[base["job_name"]] = native_state
    handle = store._records[base["request_ref"]]
    observed = store._adapter.status(handle)
    assert observed.state is JobState.UNKNOWN and observed.detail == ""
    assert observed.raw["ResourceConfig"]["SelectedInstanceType"] == "ml.g6e.16xlarge"
    read = store.observe(base["request_ref"])
    assert read["status"] == "UNKNOWN" and read["accepted"] is True and read["generated"] is False
    assert not SELECTED & set(read)
    # An UNKNOWN job of this run is not counted as a holder either (unchanged).
    from amlas.control_plane.sft_state import GENERATION_INSTANCE_PREFERENCES

    assert store._active_holders(GENERATION_INSTANCE_PREFERENCES, "gen_" + "f" * 32) == (
        {name: None for name in GENERATION_INSTANCE_PREFERENCES},
        [],
    )


@pytest.mark.parametrize("native_state", UNMAPPED)
def test_durable_sft_store_observation_of_an_unmapped_native_state_names_no_selection(
    monkeypatch, native_state
):
    store, native, _s3, model_ref, prep_ref, *_rest = injected_store(monkeypatch)
    first = store.submit(model_ref, prep_ref, "fit_v1")
    assert SELECTED <= set(store.observe(first["request_ref"]))
    _described_state(native, native_state)
    read = store.observe(first["request_ref"])
    assert read["request_ref"] == first["request_ref"] and read["trained"] is False
    assert read["status"] == "UNKNOWN" and not SELECTED & set(read)


@pytest.mark.parametrize("native_state", UNMAPPED)
def test_session_sft_store_observation_of_an_unmapped_native_state_names_no_selection(
    monkeypatch, native_state
):
    store, native, _s3, model_ref, prep_ref, *_rest = real_store(monkeypatch)
    first = store.submit(model_ref, prep_ref, "fit_v1")
    assert SELECTED <= set(store.observe(first["request_ref"]))
    _described_state(native, native_state)
    read = store.observe(first["request_ref"])
    assert read["request_ref"] == first["request_ref"] and read["trained"] is False
    assert read["status"] == "UNKNOWN" and not SELECTED & set(read)
