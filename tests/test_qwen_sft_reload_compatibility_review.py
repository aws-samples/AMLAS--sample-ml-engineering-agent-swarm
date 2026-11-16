"""Semantic review regressions; strict fake SDKs and real bounded JSON writer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import pytest
from test_campaign_state import CAMPAIGN, DDB, STS, state
from test_qwen_sft_backend import SESSION, catalog, encoded, install_output
from test_qwen_sft_binding import real_store
from test_qwen_sft_reload_compatibility_path import (
    checkpoint_fixture,
    consumer_catalog,
    publish_reload_output,
    reader_for,
    selection,
    submit_qualified,
)

from amlas.backends.adapters.campaign_state import DynamoCampaignStateBackend
from amlas.backends.types import DatasetArtifactRef, JobState
from amlas.control_plane.sft_state import SessionSftStore, VerifiedSftCheckpointSource
from amlas.training import qwen_sft as core


def per_job_observer(native):
    states = {}
    original_describe = native.describe_training_job

    def describe(**kwargs):
        requests = list(native.calls)
        request = next(r for r in requests if r["TrainingJobName"] == kwargs["TrainingJobName"])
        old_state = native.state
        native.calls = [request]
        native.state = states.get(kwargs["TrainingJobName"], old_state)
        try:
            return original_describe(**kwargs)
        finally:
            native.calls = requests
            native.state = old_state

    def tags(**kwargs):
        name = kwargs["ResourceArn"].rsplit("/", 1)[-1]
        request = next(r for r in native.calls if r["TrainingJobName"] == name)
        return {"Tags": request["Tags"]}

    native.describe_training_job = describe
    native.list_tags = tags
    return states


def historical_store(monkeypatch):
    store, native, s3, model_ref, prep_ref, _reads, _mh, _ph = real_store(monkeypatch)
    campaign = DynamoCampaignStateBackend(
        state(), dynamodb_client=DDB(), sts_client=STS(), sagemaker_client=native
    )
    binding = campaign.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=SESSION)
    historical = SessionSftStore(
        store._models,
        store._prepared,
        store._adapter,
        catalog(),
        campaign_state=campaign,
        execution_binding=binding,
    )
    original_ref = historical.submit(model_ref, prep_ref, "fit_v1")["request_ref"]
    original = historical._records[original_ref]
    install_output(historical._adapter, native, s3, original)
    result = historical._adapter.artifacts(original)
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
    current = SessionSftStore(
        store._models,
        store._prepared,
        store._adapter,
        consumer_catalog(),
        campaign_state=campaign,
        execution_binding=binding,
        reload_compatibility_reader=reader,
    )
    return {
        "old": historical,
        "current": current,
        "native": native,
        "s3": s3,
        "model_ref": model_ref,
        "prep_ref": prep_ref,
        "original": original,
        "original_ref": original_ref,
        "selected": selected,
        "reader": reader,
        "env": env,
    }


def resume(parts):
    return parts["current"].resume_original(parts["original"].id, parts["model_ref"], parts["prep_ref"])


def test_historical_train_resume_cannot_populate_current_submit_dedup(monkeypatch):
    p = historical_store(monkeypatch)
    assert resume(p)["trained"] is True
    p["env"].clear()
    before = len(p["native"].calls)
    with pytest.raises(ValueError):
        p["current"].submit(p["model_ref"], p["prep_ref"], "fit_v1")
    assert len(p["native"].calls) == before


def test_coherent_new_checkpoint_cannot_replace_exact_selected_history(monkeypatch):
    p = historical_store(monkeypatch)
    restored = resume(p)
    assert restored["trained"] is True
    _cp, _receipt, commit, put = install_output(p["old"]._adapter, p["native"], p["s3"], p["original"])
    ref = p["selected"].checkpoint.checkpoint_ref
    raw = p["s3"].objects[ref.uri.split("/", 3)[3]][1]
    value = json.loads(raw)
    value["greedy_output_ids"][0][-1] = 5
    raw = encoded(value)
    changed = dataclasses.replace(put("CHECKPOINT.json", raw), version_id="independent-checkpoint-B")
    p["s3"].install(changed, raw)
    commit["files"]["CHECKPOINT.json"] = {"sha256": changed.sha256, "byte_count": changed.byte_count}
    put("COMMITTED.json", encoded(commit))
    result = p["current"].observe(restored["request_ref"])
    assert result["success"] is False and result.get("trained") is not True


@pytest.mark.parametrize("operation", ["observe", "resume"])
def test_historical_selector_revoked_during_artifact_read_cannot_expose_success(monkeypatch, operation):
    p = historical_store(monkeypatch)
    restored = resume(p)
    adapter = p["old"]._adapter
    original = adapter.artifacts

    def revoked(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.get("trained") is True:
            p["env"].clear()
        return result

    monkeypatch.setattr(adapter, "artifacts", revoked)
    result = p["current"].observe(restored["request_ref"]) if operation == "observe" else resume(p)
    assert result["success"] is False and result.get("trained") is not True


@pytest.mark.parametrize("boundary", ["native_status", "output_read"])
def test_selected_reload_revoked_inside_readback_cannot_return_success(monkeypatch, boundary):
    fixture = checkpoint_fixture()
    selected = selection(fixture[-1])
    reader, env = reader_for(selected)
    handle = submit_qualified(fixture, reader)
    adapter = publish_reload_output(fixture, handle, selected)
    if boundary == "native_status":
        original = fixture[1].list_tags

        def revoked(**kwargs):
            result = original(**kwargs)
            env.clear()
            return result

        monkeypatch.setattr(fixture[1], "list_tags", revoked)
        assert adapter.status(handle, _reload_compatibility_reader=reader).state is JobState.UNKNOWN
    else:
        original = adapter._output

        def revoked(*args, **kwargs):
            result = original(*args, **kwargs)
            if args[1].endswith("/RELOAD.json"):
                env.clear()
            return result

        monkeypatch.setattr(adapter, "_output", revoked)
        assert adapter.artifacts(handle, _reload_compatibility_reader=reader)["success"] is False


@pytest.mark.parametrize("qualified", [False, True])
def test_actual_fixed_writer_committed_descriptor_roundtrips_parent(tmp_path, monkeypatch, qualified):
    fixture = checkpoint_fixture()
    selected = selection(fixture[-1])
    reader, _ = reader_for(selected)
    adapter, _native, s3, inputs, _old, checkpoint = fixture
    handle = (
        submit_qualified(fixture, reader)
        if qualified
        else adapter.reload(
            model=inputs.model, checkpoint=checkpoint, catalog=catalog(), runtime_session_id=SESSION
        )
    )
    publish_reload_output(fixture, handle, selected)
    prefix = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"].split("/", 3)[3] + "/model/"
    receipt = json.loads(s3.objects[prefix + "RELOAD.json"][1])
    final = json.loads(s3.objects[prefix + "COMMITTED.json"][1])
    if not qualified:
        receipt["schema_version"] = final["schema_version"] = 1
        receipt["source_commit"] = final["source_commit"] = catalog().source_commit
        for key in ("checkpoint_source_commit", "reload_contract_version", "compatibility_digest"):
            receipt.pop(key)
            final.pop(key)
    monkeypatch.setattr(core, "FIXED_OUTPUT", str(tmp_path))
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        actual_ref = core._publish_fixed_json(directory, "RELOAD.json", receipt)
        final["files"] = {"RELOAD.json": actual_ref}
        core._publish_fixed_json(directory, "COMMITTED.json", final)
    finally:
        os.close(directory)
    for name in ("RELOAD.json", "COMMITTED.json"):
        raw = (tmp_path / name).read_bytes()
        s3.install(
            DatasetArtifactRef(
                "s3://" + catalog().bucket + "/" + prefix + name,
                "actual-writer-" + name,
                hashlib.sha256(raw).hexdigest(),
                len(raw),
                "json",
            ),
            raw,
        )
    kwargs = {"_reload_compatibility_reader": reader} if qualified else {}
    assert adapter.artifacts(handle, **kwargs)["success"] is True
    assert set(actual_ref) == {"sha256", "byte_count"}


def test_real_terminal_failed_historical_reload_stays_failed_and_distinct(monkeypatch):
    p = historical_store(monkeypatch)
    states = per_job_observer(p["native"])
    old_reload = p["old"].reload(p["original_ref"])
    old_handle = p["old"]._records[old_reload["request_ref"]]
    states[p["original"].id] = "Completed"
    states[old_handle.id] = "Failed"
    assert p["old"]._adapter.status(old_handle).state is JobState.FAILED
    assert p["old"]._adapter.artifacts(old_handle)["status"] == JobState.FAILED.value
    restored = resume(p)
    fresh = p["current"].reload(restored["request_ref"])
    assert fresh["job_name"] != old_handle.id and len(p["native"].calls) == 3
    assert p["current"].reload(restored["request_ref"])["job_name"] == fresh["job_name"]
    assert p["old"]._adapter.status(old_handle).state is JobState.FAILED
    assert old_handle.extra["catalog"] == catalog()


@pytest.mark.parametrize("kind", ["canonical", "legacy_json", "unknown_format", "extra_key", "bool_count"])
def test_historical_fixed_receipt_descriptor_has_narrow_legacy_compatibility(kind):
    fixture = checkpoint_fixture()
    adapter, native, s3, _inputs, original, _checkpoint = fixture
    _cp, _receipt, commit, put = install_output(adapter, native, s3, original)
    descriptor = commit["files"]["TRAINING.json"]
    if kind == "legacy_json":
        descriptor["format"] = "json"
    elif kind == "unknown_format":
        descriptor["format"] = "safetensors"
    elif kind == "extra_key":
        descriptor["ready"] = True
    elif kind == "bool_count":
        descriptor["byte_count"] = True
    put("COMMITTED.json", encoded(commit))
    assert adapter.artifacts(original)["trained"] is (kind in ("canonical", "legacy_json"))


def test_actual_training_writer_commit_roundtrips_parent(tmp_path, monkeypatch):
    fixture = checkpoint_fixture()
    adapter, native, s3, _inputs, original, _checkpoint = fixture
    _cp, receipt, commit, _put = install_output(adapter, native, s3, original)
    prefix = original.extra["request"]["OutputDataConfig"]["S3OutputPath"].split("/", 3)[3] + "/model/"
    for name in ("adapter_model.safetensors", "adapter_config.json", "CHECKPOINT.json"):
        (tmp_path / name).write_bytes(s3.objects[prefix + name][1])
    monkeypatch.setattr(core, "FIXED_OUTPUT", str(tmp_path))
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        receipt_ref = core._publish_fixed_json(directory, "TRAINING.json", receipt)
        commit["files"]["TRAINING.json"] = receipt_ref
        core._publish_fixed_json(directory, "COMMITTED.json", commit)
    finally:
        os.close(directory)
    for name in ("TRAINING.json", "COMMITTED.json"):
        raw = (tmp_path / name).read_bytes()
        s3.install(
            DatasetArtifactRef(
                "s3://" + catalog().bucket + "/" + prefix + name,
                "actual-training-writer-" + name,
                hashlib.sha256(raw).hexdigest(),
                len(raw),
                "json",
            ),
            raw,
        )
    assert adapter.artifacts(original)["trained"] is True
    assert set(receipt_ref) == {"sha256", "byte_count"}


def test_existing_exact_current_train_reuse_precedes_historical_observation_block(monkeypatch):
    p = historical_store(monkeypatch)
    states = per_job_observer(p["native"])
    states[p["original"].id] = "Completed"
    current = p["current"].submit(p["model_ref"], p["prep_ref"], "fit_v1")
    current_handle = p["current"]._records[current["request_ref"]]
    states[current_handle.id] = "InProgress"
    assert current_handle.extra["catalog"] == consumer_catalog()
    assert resume(p)["trained"] is True
    before = len(p["native"].calls)
    reused = p["current"].submit(p["model_ref"], p["prep_ref"], "fit_v1")
    assert reused["request_ref"] == current["request_ref"]
    assert reused["job_name"] == current["job_name"] and reused["reused"] is True
    assert len(p["native"].calls) == before == 2
    assert p["original"].extra["catalog"] == catalog()
    assert p["original"].id != current_handle.id
