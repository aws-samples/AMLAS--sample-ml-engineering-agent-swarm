"""A schema-4 reload record written with the earlier `reload_single_gpu_v1` contract keeps validating and restoring.

Before the instance preference lists, an ordinary reload pinned `reload_single_gpu_v1` in its schema-4 record: exactly
one ml.g6e.4xlarge. Such records exist in deployed tables. This file writes one the way that code did (the same
ordinary reload path, with the contract name it pinned), then proves on the current source that the record decodes,
validates, restores as verified in a fresh adapter and observes its job, and that it accepts no other request. The
written bytes are pinned to those the earlier source itself wrote over the same fixtures.
Strict installed-SDK fakes, no AWS call.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
from test_qwen_sft_reload_resources import ordinary_reload, parts

from amlas.backends.adapters import qwen_sft
from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.backends.types import JobState
from amlas.control_plane.campaign_state import (
    CampaignStateError,
    _decode_private_record,
    _encode_private_record,
    _json,
    validate_submission_snapshot,
)
from amlas.control_plane.sft_state import (
    RELOAD_INSTANCE_PREFERENCES,
    RELOAD_PREFERENCE_CONTRACT,
    RELOAD_RESOURCE_CONTRACT,
    TRAIN_INSTANCE_PREFERENCES,
)

SINGLE_GPU = {"InstanceType": "ml.g6e.4xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100}
# What the source before the preference lists wrote for an ordinary reload over these fixtures,
# without and with a reload compatibility selection: (sha256 of the encoded private record, operation digest).
# Measured by running that source in the CI-equivalent container on 2026-11-10.
EARLIER_RECORD = {
    False: (
        "af02dea8f9cb4b830c760aabd61a2c54e5496789aa321ed710a748682e0d3360",  # pragma: allowlist secret
        "b4fc92dcb82f8b37c2ce7eb9ab0e33e32bed72675f865981c0b2c95242893e47",  # pragma: allowlist secret
    ),
    True: (
        "231bc58ccc0c70669858aa31b51751ba63aaf6e820460bc78f0d53a9cddb8766",  # pragma: allowlist secret
        "f000146268c07ea574e9350e4a5f25869e1a81574887ff8502770d21d7e4eceb",  # pragma: allowlist secret
    ),
}


def _earlier_writer_reload(monkeypatch, compatible=False):
    """Write a schema-4 reload record as the earlier source did: the ordinary reload pinned reload_single_gpu_v1."""
    assert RELOAD_RESOURCE_CONTRACT == "reload_single_gpu_v1"
    assert (
        qwen_sft.RELOAD_PREFERENCE_CONTRACT == RELOAD_PREFERENCE_CONTRACT == "reload_instance_preferences_v1"
    )
    p, checkpoint, selected, reader = parts(compatible)
    with monkeypatch.context() as patched:
        patched.setattr(qwen_sft, "RELOAD_PREFERENCE_CONTRACT", RELOAD_RESOURCE_CONTRACT)
        handle = ordinary_reload(p, checkpoint, selected, reader)
    assert qwen_sft.RELOAD_PREFERENCE_CONTRACT == "reload_instance_preferences_v1"
    return p, handle, reader


@pytest.mark.parametrize("compatible", [False, True])
def test_single_gpu_record_bytes_equal_those_the_earlier_source_wrote(monkeypatch, compatible):
    p, handle, _reader = _earlier_writer_reload(monkeypatch, compatible)
    record = p[4].load_original(p[5], handle.id)
    raw = _encode_private_record(record)
    assert (hashlib.sha256(raw).hexdigest(), record.operation_digest) == EARLIER_RECORD[compatible]


def test_single_gpu_record_is_schema_four_with_the_exact_fixed_request(monkeypatch):
    p, handle, _reader = _earlier_writer_reload(monkeypatch)
    assert handle.extra["submission_state"] == "accepted"
    assert handle.extra["request"]["ResourceConfig"] == SINGLE_GPU
    assert p[1].calls[-1]["ResourceConfig"] == SINGLE_GPU
    record = p[4].load_original(p[5], handle.id)
    raw = json.loads(_encode_private_record(record))
    assert raw["schema_version"] == 4 and raw["intent"]["reload_resource_contract"] == "reload_single_gpu_v1"
    assert (
        record.intent.reload_resource_contract == "reload_single_gpu_v1" and record.disposition == "ACCEPTED"
    )
    assert json.loads(record.submission_snapshot.request_json)["ResourceConfig"] == SINGLE_GPU


@pytest.mark.parametrize("compatible", [False, True])
def test_single_gpu_record_validates_and_restores_in_a_fresh_adapter(monkeypatch, compatible):
    p, handle, reader = _earlier_writer_reload(monkeypatch, compatible)
    raw = _encode_private_record(p[4].load_original(p[5], handle.id))
    decoded = _decode_private_record(raw, p[5])
    validate_submission_snapshot(decoded)
    cold = SageMakerQwenSftBackend(sagemaker_client=p[1], s3_client=p[2])
    writes, creates = list(p[2].mutations), len(p[1].calls)
    restored = cold.restore_original(decoded, fresh_inputs=p[3], _reload_compatibility_reader=reader)
    assert restored.id == handle.id and restored.arn == handle.arn
    assert restored.extra["restoration_status"] == "verified"
    assert restored.extra["request"] == json.loads(decoded.submission_snapshot.request_json)
    assert restored.extra["request"]["ResourceConfig"] == SINGLE_GPU
    observed = cold.status(restored, _reload_compatibility_reader=reader)
    assert observed.state == JobState.SUCCEEDED and observed.detail == ""
    assert p[2].mutations == writes and len(p[1].calls) == creates


@pytest.mark.parametrize(
    "resources",
    [
        {
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
            "InstancePreferences": [{"InstanceType": name} for name in RELOAD_INSTANCE_PREFERENCES],
        },
        {
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
            "InstancePreferences": [{"InstanceType": name} for name in TRAIN_INSTANCE_PREFERENCES],
        },
        {
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
            "InstancePreferences": [{"InstanceType": "ml.g6e.4xlarge"}],
        },
        {"InstanceType": "ml.g6e.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100},
        {"InstanceType": "ml.g6e.16xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100},
        {"InstanceType": "ml.g6e.4xlarge", "InstanceCount": 2, "VolumeSizeInGB": 100},
        {"InstanceType": "ml.g6e.4xlarge", "InstanceCount": 1, "VolumeSizeInGB": 200},
    ],
)
def test_single_gpu_record_accepts_no_other_request(monkeypatch, resources):
    p, handle, reader = _earlier_writer_reload(monkeypatch)
    record = p[4].load_original(p[5], handle.id)
    request = json.loads(record.submission_snapshot.request_json)
    request["ResourceConfig"] = resources
    changed = dataclasses.replace(
        record,
        submission_snapshot=dataclasses.replace(
            record.submission_snapshot, request_json=_json(request).decode()
        ),
    )
    reads, creates = p[1].reads, len(p[1].calls)
    with pytest.raises(CampaignStateError):
        validate_submission_snapshot(changed)
    with pytest.raises(CampaignStateError):
        p[0].restore_original(changed, fresh_inputs=p[3], _reload_compatibility_reader=reader)
    assert (p[1].reads, len(p[1].calls)) == (reads, creates)


def test_single_gpu_record_cannot_be_relabelled_as_the_preference_contract(monkeypatch):
    p, handle, _reader = _earlier_writer_reload(monkeypatch)
    raw = json.loads(_encode_private_record(p[4].load_original(p[5], handle.id)))
    raw["intent"]["reload_resource_contract"] = "reload_instance_preferences_v1"
    with pytest.raises(CampaignStateError):
        _decode_private_record(_json(raw), p[5])
