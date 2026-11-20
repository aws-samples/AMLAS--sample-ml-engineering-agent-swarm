"""Fixed native request shapes and private source/output integrity, no AWS."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest
from botocore.session import Session
from botocore.validate import validate_parameters
from test_managed_coding_case_envelope import ACCOUNT, BUCKET, HEADER, IMAGE, SESSION, rows, source
from test_managed_coding_preparation import S3, sha

from amlas.backends.adapters.coding_case_freeze import ProcessingCodingCaseFreezeBackend
from amlas.backends.types import JobState
from amlas.control_plane.coding_case_state import CaseFreezeCatalog
from amlas.data_eng.coding_case_managed import freeze_managed_case_bundle


class Native:
    meta = SimpleNamespace(region_name="us-west-2")

    def __init__(self):
        self.creates = []
        self.state = "InProgress"
        self.fail = None
        self.drift = False

    def create_processing_job(self, **request):
        assert 1 <= len(request["ProcessingJobName"]) <= 63, "native Processing name exceeds63"
        validate_parameters(
            request,
            Session().get_service_model("sagemaker").operation_model("CreateProcessingJob").input_shape,
        )
        self.creates.append(request)
        if self.fail:
            raise self.fail
        return {
            "ProcessingJobArn": "arn:aws:sagemaker:us-west-2:"
            + ACCOUNT
            + ":processing-job/"
            + request["ProcessingJobName"]
        }

    def describe_processing_job(self, **request):
        original = self.creates[0]
        assert request["ProcessingJobName"] == original["ProcessingJobName"]
        value = {
            **original,
            "ProcessingJobArn": "arn:aws:sagemaker:us-west-2:"
            + ACCOUNT
            + ":processing-job/"
            + original["ProcessingJobName"],
            "ProcessingJobStatus": self.state,
        }
        value.pop("Tags")
        if self.drift:
            value["NetworkConfig"] = {"EnableNetworkIsolation": False}
        return value

    def list_tags(self, **request):
        return {"Tags": self.creates[0]["Tags"]}


@pytest.fixture
def native(monkeypatch):
    import test_managed_coding_preparation as io_fixture

    # These modules copy SOURCE_SHA256 when they are first imported. Load them before the change
    # below: a first import during this test would keep the changed value for later tests.
    import amlas.arbiter.coding
    import amlas.data_eng.coding_case_profile as profile
    import amlas.evaluation.coding_pair  # noqa: F401

    # Preserve strict ownership assertions while selecting this fixture account.
    monkeypatch.setattr(io_fixture, "ACCOUNT", ACCOUNT)

    raw = b"opaque synthetic parquet bytes; parent must never parse"
    monkeypatch.setattr(profile, "SOURCE_SHA256", sha(raw))
    monkeypatch.setattr(profile, "SOURCE_BYTES", len(raw))
    original = source()
    original = dataclasses.replace(
        original,
        inspection_source=dataclasses.replace(
            original.inspection_source,
            artifact_ref=dataclasses.replace(
                original.inspection_source.artifact_ref, sha256=sha(raw), byte_count=len(raw)
            ),
        ),
    )
    s3 = S3()
    s3.add(
        original.inspection_source.artifact_ref.uri.split("/", 3)[3],
        raw,
        original.inspection_source.artifact_ref.version_id,
    )
    entries = []
    for name, uri, version, _hash in original.manifest_refs:
        body = (json.dumps({"synthetic_original_manifest": name}) + "\n").encode()
        s3.add(uri.split("/", 3)[3], body, version)
        entries.append((name, uri, version, sha(body)))
    original = dataclasses.replace(original, manifest_refs=tuple(entries))
    catalog = CaseFreezeCatalog(
        ACCOUNT,
        "us-west-2",
        BUCKET,
        "arn:aws:iam::" + ACCOUNT + ":role/owned",
        IMAGE,
        "b" * 40,
        "a" * 64,
        HEADER["deployment_id"],
        "owned",
    )
    sm = Native()
    backend = ProcessingCodingCaseFreezeBackend(sagemaker_client=sm, s3_client=s3)
    return backend, sm, s3, original, catalog


def test_native_fixed_request_isolated_distinct_names_no_environment(native):
    backend, sm, _s3, original, catalog = native
    handle = backend.submit(source=original, catalog=catalog, runtime_session_id=SESSION)
    assert handle.extra["submission_state"] == "accepted" and len(sm.creates) == 1
    request = sm.creates[0]
    assert request["NetworkConfig"] == {
        "EnableNetworkIsolation": True,
        "EnableInterContainerTrafficEncryption": False,
    }
    assert "Environment" not in request
    assert request["AppSpecification"]["ContainerEntrypoint"] == [
        "/usr/local/bin/amlas-processing-init",
        "case-freeze",
    ]
    assert (
        request["ProcessingInputs"][0]["InputName"]
        != request["ProcessingOutputConfig"]["Outputs"][0]["OutputName"]
    )
    assert request["ProcessingResources"]["ClusterConfig"] == {
        "InstanceCount": 1,
        "InstanceType": "ml.m5.xlarge",
        "VolumeSizeInGB": 30,
    }


@pytest.mark.parametrize(
    "state,expected",
    [("InProgress", JobState.RUNNING), ("Failed", JobState.FAILED), ("Stopped", JobState.STOPPED)],
)
def test_original_nonterminal_or_failed_job_never_freezes(native, state, expected):
    backend, sm, _s3, original, catalog = native
    handle = backend.submit(source=original, catalog=catalog, runtime_session_id=SESSION)
    sm.state = state
    assert backend.status(handle).state == expected
    assert backend.artifacts(handle)["frozen"] is False and len(sm.creates) == 1


def test_full_native_outputs_replay_and_private_roles_are_version_bound(native):
    backend, sm, s3, original, catalog = native
    handle = backend.submit(source=original, catalog=catalog, runtime_session_id=SESSION)
    bundle = freeze_managed_case_bundle(rows(), source=original, header=handle.extra["header"])
    for name, raw in bundle.items():
        s3.add(handle.extra["output_prefix"] + name, raw)
    sm.state = "Completed"
    result = backend.artifacts(handle)
    assert result["success"] is True and result["frozen"] is True and result["case_count"] == 164
    assert set(result["files"]) == set(bundle)
    assert all(ref.version_id not in ("", "null", None) for ref in result["files"].values())
    assert len(sm.creates) == 1


def test_original_native_policy_drift_refuses_before_output_reads(native):
    backend, sm, s3, original, catalog = native
    handle = backend.submit(source=original, catalog=catalog, runtime_session_id=SESSION)
    sm.state = "Completed"
    sm.drift = True
    s3.calls.clear()
    assert backend.artifacts(handle)["frozen"] is False
    assert s3.calls == []


def test_ambiguous_native_create_response_retains_preconstructed_handle(native):
    backend, sm, _s3, original, catalog = native
    sm.fail = TimeoutError("synthetic transport loss")
    handle = backend.submit(source=original, catalog=catalog, runtime_session_id=SESSION)
    assert handle.extra["submission_state"] == "ambiguous" and len(sm.creates) == 1


def test_same_length_original_payload_change_refuses_before_any_write_or_create(native):
    backend, sm, s3, original, catalog = native
    key = original.inspection_source.artifact_ref.uri.split("/", 3)[3]
    raw, version = s3.values[key]
    s3.values[key] = (bytes([raw[0] ^ 1]) + raw[1:], version)
    with pytest.raises(ValueError):
        backend.submit(source=original, catalog=catalog, runtime_session_id=SESSION)
    assert sm.creates == [] and not any(name == "put" for name, _request in s3.calls)


def test_completed_same_length_output_change_with_rehashed_outer_refs_refuses(native):
    backend, sm, s3, original, catalog = native
    handle = backend.submit(source=original, catalog=catalog, runtime_session_id=SESSION)
    bundle = freeze_managed_case_bundle(rows(), source=original, header=handle.extra["header"])
    bundle["ORACLES.jsonl"] = bundle["ORACLES.jsonl"].replace(b"return 0", b"return 9", 1)
    for name, raw in bundle.items():
        s3.add(handle.extra["output_prefix"] + name, raw)
    sm.state = "Completed"
    assert backend.artifacts(handle)["frozen"] is False and len(sm.creates) == 1


@pytest.mark.parametrize("length", [38, 39, 40])
def test_full_selected_prefix_grammar_produces_valid_native_job_before_staging(native, length):
    backend, _sm, s3, original, catalog = native
    prefix = "a" * length
    old_job = original.inspection_source.acquisition_job_arn.rsplit("/", 1)[-1]
    new_job = prefix + "-data-acq-a"
    origin = original.inspection_source
    original = dataclasses.replace(
        original,
        inspection_source=dataclasses.replace(
            origin,
            deployment_prefix=prefix,
            acquisition_job_arn=origin.acquisition_job_arn.replace(old_job, new_job),
            artifact_ref=dataclasses.replace(
                origin.artifact_ref, uri=origin.artifact_ref.uri.replace(old_job, new_job)
            ),
        ),
        manifest_refs=tuple(
            (name, uri.replace(old_job, new_job), version, sha)
            for name, uri, version, sha in original.manifest_refs
        ),
    )
    catalog = dataclasses.replace(catalog, deployment_prefix=prefix)
    for key, value in list(s3.values.items()):
        if old_job in key:
            s3.values[key.replace(old_job, new_job)] = value
    handle = backend.submit(source=original, catalog=catalog, runtime_session_id=SESSION)
    assert len(handle.id) <= 63 and handle.extra["submission_state"] == "accepted"


@pytest.mark.parametrize("error_metadata", [None, "malformed", []], ids=["null", "string", "list"])
def test_malformed_sdk_error_metadata_retains_original_case_identity_without_recreate(
    native, monkeypatch, error_metadata
):
    from amlas.control_plane.acquisition_state import SessionAcquisitionStore
    from amlas.control_plane.coding_case_state import SessionCodingCaseFreezeStore

    backend, sm, _s3, original, catalog = native
    fault = RuntimeError("synthetic SDK error metadata")
    fault.response = {"Error": error_metadata}
    sm.fail = fault
    acquired = SessionAcquisitionStore()
    acquired.runtime_session_id = SESSION
    acquisition_ref = "acq_" + "a" * 32

    def verified_original(reference):
        assert reference == acquisition_ref
        return original

    monkeypatch.setattr(acquired, "_verified_preparation_source", verified_original)
    store = SessionCodingCaseFreezeStore(acquired, backend, catalog)
    try:
        first = store.submit(acquisition_ref)
        second = store.submit(acquisition_ref)
    except Exception as error:
        pytest.fail(
            "Case submission discarded the original ambiguous identity for malformed SDK metadata: "
            + type(error).__name__
        )

    assert first["status"] == second["status"] == "ambiguous"
    assert first["case_ref"] == second["case_ref"]
    assert first["job_name"] == second["job_name"] == sm.creates[0]["ProcessingJobName"]
    assert first["job_arn"] == second["job_arn"]
    assert first["accepted"] is second["accepted"] is False
    assert first["reused"] is False and second["reused"] is True
    assert len(sm.creates) == len(store._records) == len(store._keys) == 1
    handle = store._records[first["case_ref"]]
    assert handle.id == first["job_name"] and handle.arn == first["job_arn"]
    assert handle.extra["submission_state"] == "ambiguous"
