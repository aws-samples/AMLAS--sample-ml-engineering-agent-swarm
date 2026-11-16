"""Pure causal-parent descriptor controls; synthetic DATA, no native grants."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid

import pytest
from test_campaign_state import fixture

from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.campaign_state import (
    CampaignStateError,
    OriginalJobReservation,
    _decode_private_record,
    _encode_private_record,
    _intent_fields,
    _native_identity,
    _pack,
    build_sft_intent,
    operation_digest,
)
from amlas.control_plane.sft_state import VerifiedSftCheckpointSource

SOURCE_REFS = ("model_acq_" + "a" * 32, "prep_" + "b" * 32)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parent_ref(binding, catalog):
    digest = "a" * 64
    return DatasetArtifactRef(
        uri="s3://"
        + catalog.bucket
        + "/sessions/"
        + binding.runtime_session_id
        + "/prompt-runs/"
        + uuid.UUID(binding.campaign_id).hex
        + "/parents/"
        + digest
        + ".json",
        version_id="ordinary-immutable-parent-version",
        sha256=digest,
        byte_count=1024 * 1024,
        format="json",
    )


def reservation(intent):
    name, arn = _native_identity(intent)
    return OriginalJobReservation(intent.binding, intent, operation_digest(intent), name, arn)


def test_absent_causal_parent_preserves_exact_legacy_schema5_bytes_and_identity():
    _store, binding, original, ddb, _sts, sm = fixture()
    legacy = build_sft_intent(binding, original.catalog, original.inputs, source_refs=SOURCE_REFS)
    expected = _intent_fields(legacy)
    assert set(expected) == {
        "binding",
        "catalog",
        "inputs",
        "recipe_id",
        "recipe_version",
        "operation",
        "checkpoint",
        "source_refs",
    }
    assert expected["source_refs"] == _pack(SOURCE_REFS)
    digest = hashlib.sha256(canonical(expected)).hexdigest()
    assert operation_digest(legacy) == digest
    raw = _encode_private_record(reservation(legacy))
    assert json.loads(raw)["schema_version"] == 5
    restored = _decode_private_record(raw, binding)
    assert getattr(restored.intent, "causal_parent_ref", None) is None
    assert _encode_private_record(restored) == raw
    absent = build_sft_intent(
        binding, original.catalog, original.inputs, source_refs=SOURCE_REFS, causal_parent_ref=None
    )
    assert _intent_fields(absent) == expected and _native_identity(absent) == _native_identity(legacy)
    assert _encode_private_record(reservation(absent)) == raw
    assert ddb.calls == [] and sm.reads == 0


def test_causal_parent_schema6_roundtrip_commits_exact_descriptor_and_rejects_tamper():
    _store, binding, original, ddb, _sts, sm = fixture()
    parent = parent_ref(binding, original.catalog)
    intent = build_sft_intent(
        binding, original.catalog, original.inputs, source_refs=SOURCE_REFS, causal_parent_ref=parent
    )
    fields = _intent_fields(intent)
    assert fields["causal_parent_ref"] == _pack(parent)
    assert operation_digest(intent) == hashlib.sha256(canonical(fields)).hexdigest()
    raw = _encode_private_record(reservation(intent))
    assert json.loads(raw)["schema_version"] == 6
    restored = _decode_private_record(raw, binding)
    assert restored.intent.causal_parent_ref == parent and _encode_private_record(restored) == raw
    changed = build_sft_intent(
        binding,
        original.catalog,
        original.inputs,
        source_refs=SOURCE_REFS,
        causal_parent_ref=dataclasses.replace(parent, version_id="different-immutable-version"),
    )
    assert _native_identity(changed) != _native_identity(intent)
    for schema, body in ((5, json.loads(raw)), (6, json.loads(raw))):
        body["schema_version"] = schema
        if schema == 6:
            body["intent"]["causal_parent_ref"]["fields"]["format"] = "jsonl"
        with pytest.raises(CampaignStateError):
            _decode_private_record(canonical(body), binding)
    assert ddb.calls == [] and sm.reads == 0


def test_malformed_causal_parent_refuses_before_journal_or_native_io():
    _store, binding, original, ddb, _sts, sm = fixture()
    parent = parent_ref(binding, original.catalog)
    prefix, filename = parent.uri.rsplit("/", 1)
    invalids = (
        dataclasses.asdict(parent),
        dataclasses.replace(parent, format="jsonl"),
        dataclasses.replace(parent, version_id="null"),
        dataclasses.replace(parent, version_id=""),
        dataclasses.replace(parent, byte_count=True),
        dataclasses.replace(parent, byte_count=0),
        dataclasses.replace(parent, byte_count=1024 * 1024 + 1),
        dataclasses.replace(
            parent, uri=parent.uri.replace(original.catalog.bucket, "other-source-bucket", 1)
        ),
        dataclasses.replace(
            parent, uri=parent.uri.replace(binding.runtime_session_id, "other-correlation", 1)
        ),
        dataclasses.replace(parent, uri=parent.uri.replace(uuid.UUID(binding.campaign_id).hex, "b" * 32, 1)),
        dataclasses.replace(parent, uri=parent.uri.replace("/parents/", "/tasks/", 1)),
        dataclasses.replace(parent, uri=prefix + "/" + "b" * 64 + ".json"),
        dataclasses.replace(parent, uri=prefix + "/" + filename + "/extra"),
    )
    for invalid in invalids:
        with pytest.raises(CampaignStateError):
            build_sft_intent(
                binding,
                original.catalog,
                original.inputs,
                source_refs=SOURCE_REFS,
                causal_parent_ref=invalid,
            )
    with pytest.raises(CampaignStateError):
        build_sft_intent(
            binding, original.catalog, original.inputs, source_refs=None, causal_parent_ref=parent
        )
    assert ddb.calls == [] and sm.reads == 0


def test_reload_refuses_causal_parent_with_valid_legacy_reload_prerequisites():
    _store, binding, original, ddb, _sts, sm = fixture()
    parent = parent_ref(binding, original.catalog)
    prefix = "s3://" + original.catalog.bucket + "/sessions/" + binding.runtime_session_id + "/checkpoint/"
    refs = tuple(
        (name, dataclasses.replace(parent, uri=prefix + name, byte_count=64, format=fmt))
        for name, fmt in (
            ("CHECKPOINT.json", "json"),
            ("adapter_config.json", "json"),
            ("adapter_model.safetensors", "safetensors"),
        )
    )
    checkpoint = VerifiedSftCheckpointSource(
        binding.runtime_session_id,
        "arn:aws:sagemaker:"
        + original.catalog.region
        + ":"
        + original.catalog.account_id
        + ":training-job/"
        + original.catalog.deployment_prefix
        + "-sft-original",
        original.recipe_id,
        original.catalog.source_commit,
        refs[0][1],
        refs,
    )
    legacy = build_sft_intent(
        binding, original.catalog, original.inputs, operation="reload", checkpoint=checkpoint
    )
    assert legacy.operation == "reload" and "causal_parent_ref" not in _intent_fields(legacy)
    with pytest.raises(CampaignStateError):
        build_sft_intent(
            binding,
            original.catalog,
            original.inputs,
            operation="reload",
            checkpoint=checkpoint,
            causal_parent_ref=parent,
        )
    assert ddb.calls == [] and sm.reads == 0
