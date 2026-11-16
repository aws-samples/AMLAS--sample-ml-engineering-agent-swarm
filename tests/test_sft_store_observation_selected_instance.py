"""The SFT store observation carries the instance SageMaker selected from the training preference list.

`SessionSftStore.observe` returns the adapter's observation through two branches: the durable branch (campaign
state injected: the original is restored from its reservation and then observed) and the session branch (no campaign
state). Both must pass `SelectedInstanceType` and `SelectedInstanceCount` through as the description names them,
for a running and for a completed job, and must carry neither key while the job waits for a machine. A store that
dropped the two keys passed every earlier test (found by a mutation check). Strict
installed-SDK fakes, no AWS call.
"""

from __future__ import annotations

import pytest
from test_qwen_sft_backend import install_output
from test_qwen_sft_binding import real_store
from test_qwen_sft_durable_submission import injected_store

from amlas.control_plane.sft_state import TRAIN_INSTANCE_PREFERENCES

SELECTED = {"SelectedInstanceType", "SelectedInstanceCount"}
BRANCHES = {"durable": injected_store, "session": real_store}


def _submitted(branch, monkeypatch):
    """(store, native fake, S3 fake, submit result) of one accepted fit_v1 training original."""
    store, native, s3, model_ref, prep_ref, *_rest = BRANCHES[branch](monkeypatch)
    assert (store._campaign_state is not None) is (branch == "durable")
    first = store.submit(model_ref, prep_ref, "fit_v1")
    assert first["accepted"] is True and not SELECTED & set(first)
    listed = [item["InstanceType"] for item in native.calls[0]["ResourceConfig"]["InstancePreferences"]]
    assert tuple(listed) == TRAIN_INSTANCE_PREFERENCES
    return store, native, s3, first


def _describe_selection(native, selected):
    """Make the strict fake describe `selected` as the chosen type, or no selection yet for None."""
    describe = getattr(native, "strict_describe", None) or native.describe_training_job
    native.strict_describe = describe

    def described(**kw):
        result = describe(**kw)
        resource = result["ResourceConfig"]
        if selected is None:
            del resource["SelectedInstanceType"], resource["SelectedInstanceCount"]
            result["SecondaryStatus"] = "Pending"
        else:
            resource["SelectedInstanceType"] = selected
        return result

    native.describe_training_job = described


@pytest.mark.parametrize("selected", TRAIN_INSTANCE_PREFERENCES)
@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_running_observation_names_the_selected_type_and_count(branch, selected, monkeypatch):
    store, native, _s3, first = _submitted(branch, monkeypatch)
    _describe_selection(native, selected)
    observed = store.observe(first["request_ref"])
    assert observed["status"] == "RUNNING" and observed["accepted"] is True and observed["trained"] is False
    assert observed["request_ref"] == first["request_ref"] and observed["job_name"] == first["job_name"]
    assert observed["SelectedInstanceType"] == selected
    assert observed["SelectedInstanceCount"] == 1 and type(observed["SelectedInstanceCount"]) is int
    assert len(native.calls) == 1


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_completed_observation_names_the_selected_type_and_count(branch, monkeypatch):
    store, native, s3, first = _submitted(branch, monkeypatch)
    install_output(store._adapter, native, s3, store._records[first["request_ref"]])
    observed = store.observe(first["request_ref"])
    assert observed["trained"] is True and observed["status"] == "checkpoint_saved"
    assert observed["request_ref"] == first["request_ref"]
    assert (observed["SelectedInstanceType"], observed["SelectedInstanceCount"]) == (
        TRAIN_INSTANCE_PREFERENCES[0],
        1,
    )
    assert len(native.calls) == 1


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_observation_of_a_job_that_waits_for_a_machine_names_no_selection(branch, monkeypatch):
    store, native, _s3, first = _submitted(branch, monkeypatch)
    _describe_selection(native, None)
    observed = store.observe(first["request_ref"])
    assert observed["status"] == "RUNNING" and observed["accepted"] is True
    assert observed["request_ref"] == first["request_ref"] and not SELECTED & set(observed)
    # The selection appears in the same store's next observation once SageMaker has chosen a type.
    _describe_selection(native, "ml.g6e.8xlarge")
    later = store.observe(first["request_ref"])
    assert (later["SelectedInstanceType"], later["SelectedInstanceCount"]) == ("ml.g6e.8xlarge", 1)
