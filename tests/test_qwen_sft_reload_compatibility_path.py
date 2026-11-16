"""Closed selected cross-source reload through normal backend/store/durable paths."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
from test_qwen_sft_backend import SESSION, backend, catalog, encoded, install_output
from test_qwen_sft_durable_submission import durable, submit

from amlas.control_plane import sft_binding, sft_state
from amlas.control_plane.campaign_state import _decode_private_record, _encode_private_record
from amlas.control_plane.sft_state import VerifiedSftCheckpointSource


def consumer_catalog():
    return catalog(
        source_commit="c" * 40,
        image_uri=catalog().image_uri.replace("b" * 64, "d" * 64),
        admission_digest="d" * 64,
    )


def checkpoint_fixture(*, parts=None):
    if parts is None:
        adapter, native, s3, inputs = backend()
        original = adapter.submit(
            inputs=inputs, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION
        )
    else:
        original = submit(parts)
        adapter, native, s3, inputs = parts[:4]
    install_output(adapter, native, s3, original)
    result = adapter.artifacts(original)
    assert result["trained"] is True
    checkpoint = VerifiedSftCheckpointSource(
        SESSION,
        original.arn,
        "fit_v1",
        catalog().source_commit,
        result["checkpoint_ref"],
        tuple(sorted(result["checkpoint_refs"].items())),
    )
    return adapter, native, s3, inputs, original, checkpoint


def selection(checkpoint):
    cls = getattr(sft_state, "ReloadCompatibilitySelection", None)
    assert cls is not None, "missing-symbol: ReloadCompatibilitySelection"
    files = sft_state.expected_model_weight_manifest("qwen3_5_9b", "1")["files"]
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return cls(
        producer_catalog=catalog(),
        consumer_catalog=consumer_catalog(),
        checkpoint=checkpoint,
        contract_version="qwen3_5_9b.safe_adapter_reload.v1",
        base_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
    )


def reader_for(selected):
    resolver = getattr(sft_binding, "resolve_reload_compatibility_catalog", None)
    assert callable(resolver), "missing-symbol: resolve_reload_compatibility_catalog"
    env = {
        "AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG": json.dumps(
            {"schema_version": 1, "selections": [dataclasses.asdict(selected)]}
        )
    }
    return lambda: resolver(env=env, catalog=selected.consumer_catalog), env


def submit_qualified(fixture, reader, *, parts=None):
    adapter, _native, _s3, inputs, _original, checkpoint = fixture
    private = (
        {}
        if parts is None
        else {"_campaign_state": parts[4], "_execution_binding": parts[5], "_durable_inputs": inputs}
    )
    return adapter.reload(
        model=inputs.model,
        checkpoint=checkpoint,
        catalog=consumer_catalog(),
        runtime_session_id=SESSION,
        _reload_compatibility_reader=reader,
        **private,
    )


def test_backend_selected_v2_preserves_producer_and_has_closed_exact_run():
    fixture = checkpoint_fixture()
    selected = selection(fixture[-1])
    reader, _env = reader_for(selected)
    h = submit_qualified(fixture, reader)
    adapter, native, s3, _inputs, original, checkpoint = fixture
    assert len(native.calls) == 2 and h.id != original.id
    raw = s3.objects[h.extra["run_ref"].uri.split("/", 3)[3]][1]
    run = json.loads(raw)
    assert len(run) == 12 and run["schema_version"] == 2
    assert run["source_commit"] == consumer_catalog().source_commit
    assert run["checkpoint_source_commit"] == checkpoint.source_commit
    assert run["compatibility_digest"] == sft_state.reload_compatibility_digest(selected)
    assert h.extra["reload_compatibility"] == selected
    assert original.extra["catalog"] == catalog()
    assert adapter.artifacts(h)["success"] is False


@pytest.mark.parametrize(
    "field", ["producer", "consumer", "image", "checkpoint", "version", "recipe", "base", "contract"]
)
def test_selected_tuple_drift_refuses_before_staging_or_create(field):
    fixture = checkpoint_fixture()
    selected = selection(fixture[-1])
    if field == "producer":
        selected = dataclasses.replace(
            selected, checkpoint=dataclasses.replace(selected.checkpoint, source_commit="f" * 40)
        )
    elif field == "consumer":
        selected = dataclasses.replace(
            selected, consumer_catalog=dataclasses.replace(selected.consumer_catalog, source_commit="f" * 40)
        )
    elif field == "image":
        selected = dataclasses.replace(
            selected,
            consumer_catalog=dataclasses.replace(
                selected.consumer_catalog, image_uri=catalog().image_uri, admission_digest="b" * 64
            ),
        )
    elif field in ("checkpoint", "version"):
        ref = dataclasses.replace(
            selected.checkpoint.checkpoint_ref,
            **({"sha256": "f" * 64} if field == "checkpoint" else {"version_id": "different-version"}),
        )
        selected = dataclasses.replace(
            selected, checkpoint=dataclasses.replace(selected.checkpoint, checkpoint_ref=ref)
        )
    elif field == "recipe":
        selected = dataclasses.replace(
            selected, checkpoint=dataclasses.replace(selected.checkpoint, recipe_id="full_v1")
        )
    elif field == "base":
        selected = dataclasses.replace(selected, base_manifest_sha256="f" * 64)
    else:
        selected = dataclasses.replace(selected, contract_version="unknown")
    reader, _ = reader_for(selected)
    before = list(fixture[2].mutations)
    with pytest.raises(ValueError):
        submit_qualified(fixture, reader)
    assert len(fixture[1].calls) == 1 and fixture[2].mutations == before


def test_absent_or_plain_public_catalog_cannot_supply_compatibility():
    fixture = checkpoint_fixture()
    selection(fixture[-1])
    with pytest.raises(ValueError):
        submit_qualified(fixture, lambda: {"approved": True})
    assert len(fixture[1].calls) == 1


def test_durable_schema4_cold_restore_reuses_qualified_original_and_fresh_selection():
    parts = durable()
    fixture = checkpoint_fixture(parts=parts)
    selected = selection(fixture[-1])
    reader, env = reader_for(selected)
    first = submit_qualified(fixture, reader, parts=parts)
    reservation = parts[4].load_original(parts[5], first.id)
    raw = _encode_private_record(reservation)
    assert json.loads(raw)["schema_version"] == 4
    assert reservation.intent.reload_resource_contract == "reload_instance_preferences_v1"
    assert first.extra["request"]["ResourceConfig"]["InstancePreferences"] == [
        {"InstanceType": "ml.g6e.12xlarge"}
    ]
    decoded = _decode_private_record(raw, parts[5])
    assert decoded.intent.reload_compatibility == selected
    before = list(parts[2].mutations)
    restored = parts[0].restore_original(decoded, fresh_inputs=parts[3], _reload_compatibility_reader=reader)
    assert restored.id == first.id and len(parts[1].calls) == 2 and parts[2].mutations == before
    env.clear()
    with pytest.raises(ValueError):
        parts[0].restore_original(decoded, fresh_inputs=parts[3], _reload_compatibility_reader=reader)
    assert len(parts[1].calls) == 2 and parts[2].mutations == before


def test_store_consumer_selection_does_not_reuse_failed_legacy_reload(monkeypatch):
    from test_qwen_sft_binding import real_store

    store, native, _s3, model_ref, prep_ref, _reads, _mh, _ph = real_store(monkeypatch)
    original_ref = store.submit(model_ref, prep_ref, "fit_v1")["request_ref"]
    original = store._records[original_ref]
    adapter = store._adapter
    install_output(adapter, native, adapter._s3, original)
    old = store.reload(original_ref)
    old_handle = store._records[old["request_ref"]]
    result = adapter.artifacts(original)
    checkpoint = VerifiedSftCheckpointSource(
        SESSION,
        original.arn,
        "fit_v1",
        catalog().source_commit,
        result["checkpoint_ref"],
        tuple(sorted(result["checkpoint_refs"].items())),
    )
    selected = selection(checkpoint)
    reader, env = reader_for(selected)
    current = sft_state.SessionSftStore(
        store._models, store._prepared, adapter, consumer_catalog(), reload_compatibility_reader=reader
    )
    current._records = dict(store._records)
    current._keys = dict(store._keys)
    current._source_refs = dict(store._source_refs)
    new = current.reload(original_ref)
    assert new["job_name"] != old["job_name"] and len(native.calls) == 3
    assert current.reload(original_ref)["job_name"] == new["job_name"] and len(native.calls) == 3
    assert old_handle.extra["catalog"] == catalog()
    env.clear()
    with pytest.raises(ValueError):
        current.reload(original_ref)
    assert len(native.calls) == 3


def publish_reload_output(fixture, handle, selected, *, change=None):
    from amlas.backends.types import DatasetArtifactRef

    adapter, native, s3, _inputs, _original, checkpoint = fixture
    request = handle.extra["request"]
    native.calls[:] = [request]
    native.state = "Completed"
    prefix = request["OutputDataConfig"]["S3OutputPath"].split("/", 3)[3] + "/model/"
    receipt = {
        "schema_version": 2,
        "operation": "reload",
        "source_commit": selected.consumer_catalog.source_commit,
        "checkpoint_source_commit": selected.checkpoint.source_commit,
        "reload_contract_version": selected.contract_version,
        "compatibility_digest": sft_state.reload_compatibility_digest(selected),
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "recipe_id": "fit_v1",
        "checkpoint_sha256": checkpoint.checkpoint_ref.sha256,
        "fresh_process_greedy_equal": True,
        "quality_success": False,
        "privilege_proof": {
            "res_uid": [10001] * 3,
            "res_gid": [10001] * 3,
            "groups": [],
            "cap_eff": 0,
            "cap_prm": 0,
            "cap_inh": 0,
            "cap_amb": 0,
            "cap_bnd": 0,
            "no_new_privs": 1,
            "keep_caps": 0,
            "seccomp": 2,
            "fs_uid": 10001,
            "fs_gid": 10001,
            "securebits": 0,
        },
    }
    final = {
        "schema_version": 2,
        "operation": "reload",
        "source_commit": receipt["source_commit"],
        "checkpoint_source_commit": receipt["checkpoint_source_commit"],
        "reload_contract_version": receipt["reload_contract_version"],
        "compatibility_digest": receipt["compatibility_digest"],
        "quality_success": False,
    }
    if change == "legacy_fallback":
        for value in (receipt, final):
            value["schema_version"] = 1
            for key in ("checkpoint_source_commit", "reload_contract_version", "compatibility_digest"):
                del value[key]
    elif change is not None:
        key, value = change
        receipt[key] = value
        if key in final:
            final[key] = value

    def put(name, value):
        raw = encoded(value)
        ref = DatasetArtifactRef(
            "s3://" + catalog().bucket + "/" + prefix + name,
            "selected-" + name,
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            "json",
        )
        s3.install(ref, raw)
        return ref

    ref = put("RELOAD.json", receipt)
    final["files"] = {"RELOAD.json": {"sha256": ref.sha256, "byte_count": ref.byte_count}}
    put("COMMITTED.json", final)
    return adapter


def test_selected_schema2_readback_requires_fresh_selector_and_original_producer():
    fixture = checkpoint_fixture()
    selected = selection(fixture[-1])
    reader, env = reader_for(selected)
    handle = submit_qualified(fixture, reader)
    adapter = publish_reload_output(fixture, handle, selected)
    result = adapter.artifacts(handle, _reload_compatibility_reader=reader)
    assert result["success"] is True and result["fresh_process_greedy_equal"] is True
    assert result["quality_success"] is False and result["production_admission"] is False
    env.clear()
    assert adapter.artifacts(handle, _reload_compatibility_reader=reader)["success"] is False


@pytest.mark.parametrize(
    "change",
    [
        "legacy_fallback",
        ("schema_version", True),
        ("source_commit", "f" * 40),
        ("checkpoint_source_commit", "f" * 40),
        ("reload_contract_version", "unknown"),
        ("compatibility_digest", "f" * 64),
        ("checkpoint_sha256", "f" * 64),
        ("fresh_process_greedy_equal", False),
        ("ready", True),
    ],
)
def test_rehashed_selected_schema2_drift_cannot_replace_submission_selection(change):
    fixture = checkpoint_fixture()
    selected = selection(fixture[-1])
    reader, _ = reader_for(selected)
    handle = submit_qualified(fixture, reader)
    adapter = publish_reload_output(fixture, handle, selected, change=change)
    assert adapter.artifacts(handle, _reload_compatibility_reader=reader)["success"] is False
