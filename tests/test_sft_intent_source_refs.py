"""Pure private intent compatibility; synthetic fixture scope, no native grants.

These tests do not replace actual normal-SDK RED72. Legacy optional absence
retains original bytes/native identity; refs are DATA, never action authority
or independently qualified upstream publication.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from test_campaign_state import fixture

from amlas.control_plane.campaign_state import (
    CampaignStateError,
    OriginalJobReservation,
    _binding_fields,
    _decode_private_record,
    _encode_private_record,
    _intent_fields,
    _native_identity,
    _pack,
    build_sft_intent,
    operation_digest,
)

SOURCE_REFS = ("model_acq_" + "a" * 32, "prep_" + "b" * 32)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def test_absent_source_refs_preserve_legacy_intent_record_and_native_identity():
    _store, binding, intent, ddb, _sts, sm = fixture()
    legacy = {
        "binding": _binding_fields(binding),
        "catalog": _pack(intent.catalog),
        "inputs": _pack(intent.inputs),
        "recipe_id": intent.recipe_id,
        "recipe_version": "1",
        "operation": "train",
        "checkpoint": None,
    }
    assert _intent_fields(intent) == legacy
    digest = hashlib.sha256(canonical(legacy)).hexdigest()
    prefix = binding.scope.deployment_prefix
    name = prefix + "-sft-" + digest[: 63 - len(prefix) - 5]
    arn = "arn:aws:sagemaker:us-west-2:" + binding.scope.account_id + ":training-job/" + name
    assert operation_digest(intent) == digest and _native_identity(intent) == (name, arn)
    record = OriginalJobReservation(binding, intent, digest, name, arn)
    expected = {
        "schema_version": 2,
        "binding": legacy["binding"],
        "intent": {key: value for key, value in legacy.items() if key != "binding"},
        "operation_digest": digest,
        "native_job_name": name,
        "native_job_arn": arn,
        "disposition": "RESERVED",
        "revision": 0,
        "error_code": None,
        "submission_snapshot": None,
    }
    raw = _encode_private_record(record)
    assert raw == canonical(expected)
    restored = _decode_private_record(raw, binding)
    assert _encode_private_record(restored) == raw
    assert getattr(restored.intent, "source_refs", None) is None
    assert ddb.calls == [] and sm.reads == 0


def test_source_refs_commit_to_original_native_identity_and_closed_codec():
    _store, binding, legacy, ddb, _sts, sm = fixture()
    intent = build_sft_intent(binding, legacy.catalog, legacy.inputs, source_refs=SOURCE_REFS)
    assert intent.source_refs == SOURCE_REFS
    assert _intent_fields(intent)["source_refs"] == _pack(SOURCE_REFS)
    assert operation_digest(intent) != operation_digest(legacy)
    name, arn = _native_identity(intent)
    assert (name, arn) != _native_identity(legacy)
    record = OriginalJobReservation(binding, intent, operation_digest(intent), name, arn)
    raw = _encode_private_record(record)
    assert json.loads(raw)["schema_version"] == 5
    restored = _decode_private_record(raw, binding)
    assert restored.intent.source_refs == SOURCE_REFS and _encode_private_record(restored) == raw
    changed = build_sft_intent(
        binding, legacy.catalog, legacy.inputs, source_refs=("model_acq_" + "e" * 32, SOURCE_REFS[1])
    )
    assert _native_identity(changed) != (name, arn)
    assert ddb.calls == [] and sm.reads == 0


def test_partial_or_malformed_source_refs_refuse_before_journal_or_native_io():
    _store, binding, legacy, ddb, _sts, sm = fixture()
    for invalid in (
        (SOURCE_REFS[0],),
        (SOURCE_REFS[0], None),
        list(SOURCE_REFS),
        (SOURCE_REFS[1], SOURCE_REFS[0]),
    ):
        with pytest.raises(CampaignStateError):
            build_sft_intent(binding, legacy.catalog, legacy.inputs, source_refs=invalid)
    assert ddb.calls == [] and sm.reads == 0
