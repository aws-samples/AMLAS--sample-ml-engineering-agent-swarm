"""Source-only reload sizing; strict installed SDK fakes, no native admission."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
from test_qwen_sft_backend import SESSION, catalog
from test_qwen_sft_durable_submission import durable, submit
from test_qwen_sft_reload_compatibility_path import (
    checkpoint_fixture,
    reader_for,
    selection,
)
from test_qwen_sft_reload_compatibility_review import per_job_observer

from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.backends.types import JobState
from amlas.control_plane.campaign_state import (
    CampaignOperationIntent,
    CampaignStateError,
    _decode_private_record,
    _encode_private_record,
    _json,
    _record_fields,
    build_sft_intent,
    operation_digest,
)
from amlas.control_plane.sft_state import (
    RELOAD_INSTANCE_PREFERENCES,
    TRAIN_INSTANCE_PREFERENCES,
    fixed_reload_binding,
    fixed_sft_request,
)

# The contract new ordinary reloads pin: one instance from the RELOAD_INSTANCE_PREFERENCES list (one type since
# 2026-11-11, still sent as a list).
CONTRACT = "reload_instance_preferences_v1"
LEGACY = {
    2: "d54e825e281098ae7d27c2f96c4c53a8e62ad5091046e2489b80ce447523797e",  # pragma: allowlist secret
    3: "3d3bc0dbea69cfb07c56b5b586987ecfc9f29018d1eef837b88e72282081321b",  # pragma: allowlist secret
}
LEGACY_DIGEST = {
    2: "df0350367beeedb551f9c1e6fc46259a69469aa301955456ddb4fbce0a503771",  # pragma: allowlist secret
    3: "825f587992ec66e57853a709328549e1194ad78008b5adcd8d3a51dbd7319191",  # pragma: allowlist secret
}
LEGACY_ONE = "bf8ad3a53d40809e6f5d303cd5410c163a7c0c067052631d82d0b97c0239aed5"  # pragma: allowlist secret
TRAIN_REQUEST = "80aed81a4098edb131525060103bbff66b214a8ca82ed87d4d6b55b66916ea6b"  # pragma: allowlist secret


def parts(compatible=False):
    p = durable()
    f = checkpoint_fixture(parts=p)
    per_job_observer(p[1])
    selected = selection(f[-1]) if compatible else None
    reader = reader_for(selected)[0] if selected is not None else None
    return p, f[-1], selected, reader


def original_reload(p, checkpoint, selected, reader):
    """Seed exact old writer behavior independently of the new ordinary default."""
    adapter, _native, _s3, inputs, campaign, binding, _ddb = p
    target = selected.consumer_catalog if selected else catalog()
    intent = build_sft_intent(
        binding,
        target,
        inputs,
        recipe_id=checkpoint.recipe_id,
        operation="reload",
        checkpoint=checkpoint,
        reload_compatibility=selected,
    )
    return adapter._submit(
        model=inputs.model,
        catalog=target,
        session=SESSION,
        recipe_id=checkpoint.recipe_id,
        operation="reload",
        other_files=dict(checkpoint.file_refs),
        other_channel="adapter",
        binding=fixed_reload_binding(target, checkpoint, selected),
        prepared=None,
        campaign_state=campaign,
        execution_binding=binding,
        intent=intent,
        reload_compatibility=selected,
        reload_compatibility_reader=reader,
    )


def ordinary_reload(p, checkpoint, selected, reader):
    return p[0].reload(
        model=p[3].model,
        checkpoint=checkpoint,
        catalog=selected.consumer_catalog if selected else catalog(),
        runtime_session_id=SESSION,
        _campaign_state=p[4],
        _execution_binding=p[5],
        _durable_inputs=p[3],
        _reload_compatibility_reader=reader,
    )


def contract_intent(p, checkpoint, *, contract=CONTRACT, operation="reload"):
    assert "reload_resource_contract" in CampaignOperationIntent.__dataclass_fields__, (
        "missing-symbol: CampaignOperationIntent.reload_resource_contract"
    )
    return build_sft_intent(
        p[5],
        catalog(),
        p[3],
        operation=operation,
        checkpoint=checkpoint if operation == "reload" else None,
        reload_resource_contract=contract,
    )


@pytest.mark.parametrize("compatible", [False, True])
def test_normal_reload_pins_single_gpu_contract_before_fake_create(compatible):
    """New ordinary reloads pin reload_instance_preferences_v1 before the create (test id kept)."""
    p, checkpoint, selected, reader = parts(compatible)
    observed = []
    create = p[1].create_training_job

    def capture(**request):
        observed.append(json.loads(p[6].items[request["TrainingJobName"]]["private_record"]["B"]))
        return create(**request)

    p[1].create_training_job = capture
    h = ordinary_reload(p, checkpoint, selected, reader)
    assert h.extra["submission_state"] == "accepted"
    request = p[1].calls[-1]
    assert request["ResourceConfig"] == {
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
        "InstancePreferences": [{"InstanceType": name} for name in RELOAD_INSTANCE_PREFERENCES],
    }
    assert request["ResourceConfig"]["InstancePreferences"] == [{"InstanceType": "ml.g6e.12xlarge"}]
    assert "InstanceType" not in request["ResourceConfig"]
    before = observed[0]
    assert before["schema_version"] == 4
    assert before["intent"]["reload_resource_contract"] == CONTRACT
    assert (
        before["intent"]["reload_compatibility"] is None
        if not compatible
        else (before["intent"]["reload_compatibility"] is not None)
    )
    assert before["disposition"] == "CREATE_ATTEMPTED"
    assert json.loads(before["submission_snapshot"]["fields"]["request_json"]) == request
    reservation = p[4].load_original(p[5], h.id)
    assert reservation.operation_digest == operation_digest(reservation.intent)
    assert len(p[1].calls) == 2


@pytest.mark.parametrize("compatible", [False, True])
def test_baseline_pinned_legacy_reload_bytes_and_digest_restore_exact_twelve_x(compatible):
    p, checkpoint, selected, reader = parts(compatible)
    h = original_reload(p, checkpoint, selected, reader)
    r = p[4].load_original(p[5], h.id)
    raw = _encode_private_record(r)
    version = 3 if compatible else 2
    assert hashlib.sha256(raw).hexdigest() == LEGACY[version]
    assert r.operation_digest == LEGACY_DIGEST[version]
    assert json.loads(raw)["schema_version"] == version
    assert "reload_resource_contract" not in json.loads(raw)["intent"]
    assert h.extra["request"]["ResourceConfig"]["InstanceType"] == "ml.g6e.12xlarge"
    cold = SageMakerQwenSftBackend(sagemaker_client=p[1], s3_client=p[2])
    before = list(p[2].mutations)
    restored = cold.restore_original(
        _decode_private_record(raw, p[5]),
        fresh_inputs=p[3],
        _reload_compatibility_reader=reader,
    )
    assert restored.id == h.id and restored.arn == h.arn
    assert restored.extra["request"] == json.loads(r.submission_snapshot.request_json)
    assert cold.status(restored, _reload_compatibility_reader=reader).state == JobState.SUCCEEDED
    assert p[2].mutations == before and len(p[1].calls) == 2


def test_baseline_pinned_schema_one_remains_descriptive_without_create():
    p, checkpoint, selected, reader = parts()
    h = original_reload(p, checkpoint, selected, reader)
    r = dataclasses.replace(p[4].load_original(p[5], h.id), submission_snapshot=None)
    raw = _json(_record_fields(r, schema_version=1))
    assert hashlib.sha256(raw).hexdigest() == LEGACY_ONE
    restored = p[0].restore_original(_decode_private_record(raw, p[5]), fresh_inputs=p[3])
    assert restored.id == h.id and "request" not in restored.extra
    assert restored.extra["submission_state"] == "ambiguous" and len(p[1].calls) == 2


def test_training_request_is_byte_identical_and_remains_twelve_x():
    """TRAIN_REQUEST is the frozen digest of the earlier fixed ml.g6e.12xlarge training request.

    That historical request stays byte-identical and reproducible; a new training snapshot lists
    TRAIN_INSTANCE_PREFERENCES instead (ml.g6e.16xlarge first).
    """
    p = durable()
    h = submit(p)
    r = p[4].load_original(p[5], h.id)
    run_ref = r.submission_snapshot.run_ref

    def encoded(request):
        return json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

    current = fixed_sft_request(catalog(), SESSION, h.id, "train", "fit_v1", run_ref)
    historical = fixed_sft_request(catalog(), SESSION, h.id, "train", "fit_v1", run_ref, historical=True)
    assert r.submission_snapshot.request_json == encoded(current)
    assert hashlib.sha256(encoded(historical).encode()).hexdigest() == TRAIN_REQUEST
    assert hashlib.sha256(r.submission_snapshot.request_json.encode()).hexdigest() != TRAIN_REQUEST
    assert h.extra["request"]["ResourceConfig"]["InstancePreferences"] == [
        {"InstanceType": name} for name in TRAIN_INSTANCE_PREFERENCES
    ]
    assert TRAIN_INSTANCE_PREFERENCES[0] == "ml.g6e.16xlarge"
    assert historical["ResourceConfig"] == {
        "InstanceType": "ml.g6e.12xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
    }
    assert json.loads(_encode_private_record(r))["schema_version"] == 2
    assert "reload_resource_contract" not in json.loads(_encode_private_record(r))["intent"]


@pytest.mark.parametrize("compatible", [False, True])
def test_new_schema_four_cold_restoration_and_normal_repeat_keep_one_reload_create(compatible):
    p, checkpoint, selected, reader = parts(compatible)
    h = ordinary_reload(p, checkpoint, selected, reader)
    r = p[4].load_original(p[5], h.id)
    raw = _encode_private_record(r)
    assert json.loads(raw)["schema_version"] == 4
    before = list(p[2].mutations)
    cold = SageMakerQwenSftBackend(sagemaker_client=p[1], s3_client=p[2])
    restored = cold.restore_original(
        _decode_private_record(raw, p[5]), fresh_inputs=p[3], _reload_compatibility_reader=reader
    )
    assert restored.id == h.id and restored.arn == h.arn
    assert restored.extra["request"] == h.extra["request"]
    p = (cold, *p[1:])
    repeated = ordinary_reload(p, checkpoint, selected, reader)
    assert repeated.id == h.id and repeated.extra["request"] == h.extra["request"]
    assert len(p[1].calls) == 2 and p[2].mutations == before


@pytest.mark.parametrize("contract", ["unknown", "", 1])
def test_unknown_resource_contract_refuses_before_reservation_or_create(contract):
    p, checkpoint, _selected, _reader = parts()
    before = len(p[6].calls), len(p[1].calls), list(p[2].mutations)
    assert "reload_resource_contract" in CampaignOperationIntent.__dataclass_fields__, (
        "missing-symbol: CampaignOperationIntent.reload_resource_contract"
    )
    with pytest.raises(CampaignStateError):
        contract_intent(p, checkpoint, contract=contract)
    assert (len(p[6].calls), len(p[1].calls), p[2].mutations) == before


def test_reload_resource_contract_on_train_refuses():
    p, checkpoint, _selected, _reader = parts()
    assert "reload_resource_contract" in CampaignOperationIntent.__dataclass_fields__, (
        "missing-symbol: CampaignOperationIntent.reload_resource_contract"
    )
    with pytest.raises(CampaignStateError):
        contract_intent(p, checkpoint, operation="train")
    with pytest.raises(ValueError):
        fixed_sft_request(
            catalog(),
            SESSION,
            p[1].calls[0]["TrainingJobName"],
            "train",
            "fit_v1",
            p[4].load_original(p[5], p[1].calls[0]["TrainingJobName"]).submission_snapshot.run_ref,
            resource_contract=CONTRACT,
        )


def test_original_resource_selector_cannot_be_replaced_by_snapshot_request():
    p, checkpoint, selected, reader = parts()
    h = ordinary_reload(p, checkpoint, selected, reader)
    r = p[4].load_original(p[5], h.id)
    assert getattr(r.intent, "reload_resource_contract", None) == CONTRACT
    request = json.loads(r.submission_snapshot.request_json)
    request["ResourceConfig"] = {"InstanceType": "ml.g6e.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100}
    changed = dataclasses.replace(
        r,
        submission_snapshot=dataclasses.replace(r.submission_snapshot, request_json=_json(request).decode()),
    )
    before = p[1].reads, len(p[1].calls)
    with pytest.raises(CampaignStateError):
        p[0].restore_original(changed, fresh_inputs=p[3])
    assert (p[1].reads, len(p[1].calls)) == before


def test_schema_four_missing_selector_has_no_legacy_downgrade():
    p, checkpoint, selected, reader = parts()
    h = ordinary_reload(p, checkpoint, selected, reader)
    raw = json.loads(_encode_private_record(p[4].load_original(p[5], h.id)))
    assert raw["schema_version"] == 4
    raw["intent"].pop("reload_resource_contract")
    with pytest.raises(CampaignStateError):
        _decode_private_record(_json(raw), p[5])


def test_reordered_preference_snapshot_cannot_restore():
    """A reload snapshot whose list differs from the pinned one cannot restore (test id kept).

    The reload lists one type since 2026-11-11, so its own list has no second order. The changed lists here are
    the earlier two-type list in both orders (the reordering this test was written for), the pinned type twice,
    the pinned type with a second type before and after it, another single type and an empty list. A reordered
    training list is refused in test_capacity_preference_lists.py.
    """
    p, checkpoint, selected, reader = parts()
    h = ordinary_reload(p, checkpoint, selected, reader)
    r = p[4].load_original(p[5], h.id)
    stored = json.loads(r.submission_snapshot.request_json)
    assert stored["ResourceConfig"]["InstancePreferences"] == [{"InstanceType": "ml.g6e.12xlarge"}]
    for listed in (
        ("ml.g6e.16xlarge", "ml.g6e.12xlarge"),
        ("ml.g6e.12xlarge", "ml.g6e.16xlarge"),
        ("ml.g6e.12xlarge", "ml.g6e.12xlarge"),
        ("ml.g6e.12xlarge", "ml.g6e.4xlarge"),
        ("ml.g6e.4xlarge", "ml.g6e.12xlarge"),
        ("ml.g6e.16xlarge",),
        (),
    ):
        request = json.loads(r.submission_snapshot.request_json)
        request["ResourceConfig"]["InstancePreferences"] = [{"InstanceType": name} for name in listed]
        assert request != stored
        changed = dataclasses.replace(
            r,
            submission_snapshot=dataclasses.replace(
                r.submission_snapshot, request_json=_json(request).decode()
            ),
        )
        before = p[1].reads, len(p[1].calls)
        with pytest.raises(CampaignStateError):
            p[0].restore_original(changed, fresh_inputs=p[3])
        assert (p[1].reads, len(p[1].calls)) == before


@pytest.mark.parametrize(
    "resources,accepted",
    [
        ({"InstanceType": "ml.g6e.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100}, True),
        ({"InstanceType": "ml.g6e.16xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100}, False),
        ({"InstanceType": "ml.g6e.12xlarge", "InstanceCount": 2, "VolumeSizeInGB": 100}, False),
    ],
)
def test_training_snapshot_from_before_the_capacity_option_still_validates(resources, accepted):
    """A training record written with the earlier fixed ml.g6e.12xlarge request keeps restoring; nothing else does."""
    from amlas.control_plane.campaign_state import validate_submission_snapshot

    p = durable()
    h = submit(p)
    r = p[4].load_original(p[5], h.id)
    validate_submission_snapshot(r)
    request = json.loads(r.submission_snapshot.request_json)
    request["ResourceConfig"] = resources
    historical = dataclasses.replace(
        r,
        submission_snapshot=dataclasses.replace(r.submission_snapshot, request_json=_json(request).decode()),
    )
    if accepted:
        validate_submission_snapshot(historical)
    else:
        with pytest.raises(CampaignStateError):
            validate_submission_snapshot(historical)
