"""Saved native I/O default shapes; public-only source fixtures, no service calls."""

from __future__ import annotations

import copy

import pytest
from test_managed_coding_preparation import complete_outputs, encoded, submit
from test_managed_coding_preparation import fixture as base_fixture
from test_managed_open_swe_preparation import parquet_fixture, v2_outputs

from amlas.backends.types import JobState


@pytest.fixture
def fixture(monkeypatch):
    return base_fixture.__wrapped__(monkeypatch)


def captured_defaults(value, handle, *, scope=None, app_managed=False, extra=False):
    sm = value[1]
    native = sm.describe_processing_job
    descriptions = []

    def describe(**kwargs):
        descriptions.append(kwargs)
        observed = copy.deepcopy(native(**kwargs))
        for category, entries in (
            ("input", observed["ProcessingInputs"]),
            ("output", observed["ProcessingOutputConfig"]["Outputs"]),
        ):
            for item in entries:
                item["AppManaged"] = app_managed if scope == category else False
                if extra and scope == category:
                    item["UnexpectedDefault"] = False
        return observed

    sm.describe_processing_job = describe
    sm.create_processing_job = lambda **kwargs: pytest.fail("observer resubmitted original handle")
    value[2].put_object = lambda **kwargs: pytest.fail("observer staged a new object")
    value[2].copy_object = lambda **kwargs: pytest.fail("observer copied a new input")
    return descriptions


@pytest.mark.parametrize("profile", ["v1", "v2"])
@pytest.mark.parametrize(
    "native,expected",
    [("InProgress", JobState.RUNNING), ("Completed", JobState.SUCCEEDED), ("Failed", JobState.FAILED)],
)
def test_original_handle_accepts_only_false_native_defaults_without_mutation(
    fixture, profile, native, expected
):
    value = parquet_fixture(fixture) if profile == "v2" else fixture
    handle = submit(value)
    request_before = copy.deepcopy(handle.extra["request"])
    descriptions = captured_defaults(value, handle)
    value[1].state = native
    calls_before = list(value[2].calls)
    assert value[0].status(handle).state == expected
    assert descriptions == [{"ProcessingJobName": handle.id}]
    assert value[2].calls == calls_before
    assert handle.extra["request"] == request_before and len(value[1].creates) == 1


@pytest.mark.parametrize("scope", ["input", "output"])
@pytest.mark.parametrize("invalid", [True, None, 0, 1, "false", [], {}])
def test_true_null_nonbool_or_unknown_defaults_refuse_before_output_reads(fixture, scope, invalid):
    handle = submit(fixture)
    captured_defaults(fixture, handle, scope=scope, app_managed=invalid)
    fixture[1].state = "Completed"
    before = list(fixture[2].calls)
    assert fixture[0].status(handle).state == JobState.UNKNOWN
    assert fixture[0].artifacts(handle)["prepared"] is False
    assert fixture[2].calls == before and len(fixture[1].creates) == 1


@pytest.mark.parametrize("scope", ["input", "output"])
def test_unrelated_extra_fields_stay_rejected(fixture, scope):
    handle = submit(fixture)
    captured_defaults(fixture, handle, scope=scope, extra=True)
    assert fixture[0].status(handle).state == JobState.UNKNOWN


@pytest.mark.parametrize("profile", ["v1", "v2"])
def test_completed_default_readback_keeps_versioned_output_verification(fixture, profile):
    value = parquet_fixture(fixture) if profile == "v2" else fixture
    handle = submit(value)
    (v2_outputs if profile == "v2" else complete_outputs)(value, handle)
    captured_defaults(value, handle)
    report = value[0].artifacts(handle)
    assert report["success"] is True and report["prepared"] is True
    assert report["trained"] is False and len(value[1].creates) == 1
    for ref in [report["preparation_ref"], *report["split_refs"].values(), *report["sidecar_refs"].values()]:
        assert ref["version_id"] not in ("", "null", None)


@pytest.mark.parametrize("mutation", ["source", "profile", "proof", "count"])
def test_rehashed_commit_substitution_remains_refused_after_false_defaults(fixture, mutation):
    handle = submit(fixture)
    prefix, commit = complete_outputs(fixture, handle)
    if mutation == "source":
        commit["source_commit"] = "c" * 40
    elif mutation == "profile":
        commit["privilege_profile"] = "caller_profile"
    elif mutation == "proof":
        commit["guest_evidence"]["privilege_proof"] = {"cap_eff": 0}
    else:
        commit["counts"]["source_count"] = 3
    fixture[2].add(prefix + "COMMITTED.json", encoded(commit))
    captured_defaults(fixture, handle)
    assert fixture[0].status(handle).state == JobState.SUCCEEDED
    result = fixture[0].artifacts(handle)
    assert result["prepared"] is False and result["success"] is False
    assert len(fixture[1].creates) == 1


@pytest.mark.parametrize("name", ["ProcessingInputs", "ProcessingOutputConfig"])
def test_defaults_cover_each_entry_and_preserve_explicit_false_and_other_fields(name):
    from amlas.backends.adapters.coding_preparation import _native_request_field

    entries = [{"Name": "first", "AppManaged": False}, {"Name": "second", "AppManaged": False}]
    raw = entries if name == "ProcessingInputs" else {"Outputs": entries, "OtherField": "retained"}
    before = copy.deepcopy(raw)
    expected_entries = [{"Name": "first"}, {"Name": "second"}]
    expected = (
        expected_entries
        if name == "ProcessingInputs"
        else {"Outputs": expected_entries, "OtherField": "retained"}
    )
    assert _native_request_field(name, raw) == expected
    assert raw == before


@pytest.mark.parametrize("name", ["ProcessingInputs", "ProcessingOutputConfig"])
def test_second_entry_integer_zero_is_not_false(name):
    from amlas.backends.adapters.coding_preparation import _native_request_field

    entries = [{"Name": "first", "AppManaged": False}, {"Name": "second", "AppManaged": 0}]
    raw = entries if name == "ProcessingInputs" else {"Outputs": entries}
    before = copy.deepcopy(raw)
    with pytest.raises(ValueError):
        _native_request_field(name, raw)
    assert raw == before
