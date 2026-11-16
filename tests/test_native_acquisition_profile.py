"""Native receipts are provenance telemetry, never a privilege or admission grant."""

from __future__ import annotations

import json

import pytest
from test_managed_dataset_acquisition import (
    IMAGE,
    REGION,
    ROLE,
    NativeProcessing,
    NativeS3,
    populate,
    submit,
)
from test_managed_dataset_acquisition import ownership as ownership

from amlas.backends.adapters import dataset_acquisition as acquisition
from amlas.backends.ports import DataAcquisitionBackend
from amlas.backends.registry import REGISTRY
from amlas.backends.types import JobState

PROFILE = "sagemaker_native_ceiling_v1"


def native_backend(sm, s3):
    return acquisition.SageMakerNativeDatasetAcquisitionBackend(
        region=REGION,
        role_arn=ROLE,
        image_uri=IMAGE,
        session_bucket="fixture-private-bucket",
        source_commit="c" * 40,
        sagemaker_client=sm,
        s3_client=s3,
    )


def proof():
    return {
        "res_uid": [10001] * 3,
        "res_gid": [10001] * 3,
        "groups": [],
        "cap_eff": 0,
        "cap_prm": 0,
        "cap_inh": 0,
        "cap_amb": 0,
        "cap_bnd": 0x200404FB,
        "no_new_privs": 1,
        "keep_caps": 0,
        "seccomp": 2,
        "fs_uid": 10001,
        "fs_gid": 10001,
        "securebits": 0,
    }


def native_receipt(handle, s3):
    populate(handle, s3)
    key = handle.extra["output_prefix"] + "ACQUIRED.json"
    receipt = json.loads(s3.objects[key])
    receipt.update(schema_version=2, privilege_profile=PROFILE, privilege_proof=proof())
    s3.objects[key] = json.dumps(receipt).encode()
    return key, receipt


def test_native_default_uses_source_owned_profile(ownership):
    backend = REGISTRY.resolve(DataAcquisitionBackend).adapter
    assert isinstance(backend, acquisition.SageMakerNativeDatasetAcquisitionBackend)
    assert backend.privilege_profile == PROFILE


def test_native_receipt_verifies_bytes_without_claiming_sandbox_attestation(ownership):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = native_backend(sm, s3)
    handle = submit(backend)
    native_receipt(handle, s3)
    result = backend.artifacts(handle)
    assert result["acquired"] is True
    assert result["privilege_profile"] == PROFILE
    assert result["privilege_proof"] == proof()
    assert result["integrity"]["guest_privilege_claim_attested"] is False
    assert result["prepared"] is False and result["trained"] is False
    assert handle.extra["privilege_profile"] == PROFILE
    assert len(sm.calls) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("schema_version", True),
        ("privilege_profile", "zero_bounding_v1"),
        ("privilege_profile", None),
        ("privilege_proof", None),
    ],
)
def test_native_receipt_profile_drift_refuses_before_payload_hash(ownership, field, value):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = native_backend(sm, s3)
    handle = submit(backend)
    key, receipt = native_receipt(handle, s3)
    receipt[field] = value
    s3.objects[key] = json.dumps(receipt).encode()
    s3.head_object = lambda **kwargs: pytest.fail("profile drift reached payload I/O")
    assert backend.artifacts(handle)["acquired"] is False
    assert len(sm.calls) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cap_eff", 1),
        ("cap_bnd", 1 << 8),
        ("cap_bnd", 1 << 63),
        ("no_new_privs", 0),
        ("no_new_privs", True),
        ("keep_caps", 1),
        ("seccomp", 0),
        ("fs_uid", 0),
        ("fs_gid", 0),
        ("securebits", 1 << 4),
        ("securebits", 1 << 2),
        ("res_uid", [10001, 10001, 0]),
        ("groups", [0]),
    ],
)
def test_native_receipt_privilege_drift_refuses_before_payload(ownership, field, value):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = native_backend(sm, s3)
    handle = submit(backend)
    key, receipt = native_receipt(handle, s3)
    receipt["privilege_proof"][field] = value
    s3.objects[key] = json.dumps(receipt).encode()
    s3.head_object = lambda **kwargs: pytest.fail("privilege drift reached payload I/O")
    assert backend.artifacts(handle)["acquired"] is False


def test_native_handle_profile_substitution_refuses_native_read(ownership):
    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = native_backend(sm, s3)
    handle = submit(backend)
    handle.extra["privilege_profile"] = "zero_bounding_v1"
    sm.describe_processing_job = lambda **kwargs: pytest.fail("profile substitution reached native read")
    assert backend.status(handle).state == JobState.UNKNOWN


@pytest.mark.parametrize("field", ["privilege_profile", "receipt_schema_version", "initializer_feature"])
def test_resources_cannot_select_privilege_profile(ownership, field):
    sm, s3 = NativeProcessing(), NativeS3()
    backend = native_backend(sm, s3)
    with pytest.raises(ValueError):
        submit(backend, **{field: PROFILE})
    assert not sm.calls


def test_strict_receipt_does_not_accept_native_schema(ownership):
    from test_managed_dataset_acquisition import adapter

    sm, s3 = NativeProcessing(state="Completed"), NativeS3()
    backend = adapter(sm=sm, s3=s3)
    handle = submit(backend)
    native_receipt(handle, s3)
    assert backend.artifacts(handle)["acquired"] is False
