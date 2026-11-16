"""Normal managed schema3 recovery consumes committed producer schema2, never grants."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from test_sft_producer_review_fixes import install_fresh
from test_sft_restore_aliases import normal_seed, policy

from amlas.backends.types import JobState
from amlas.control_plane import sft_binding, sft_state
from amlas.control_plane.producer_snapshot import ProducerSnapshotRepository, hydrate_sft_campaign
from amlas.control_plane.sft_state import (
    ReloadCompatibilitySelection,
    SessionSftStore,
    VerifiedSftCheckpointSource,
)


def recovery_seed(monkeypatch, tmp_path, *, current_train=False):
    value = normal_seed(monkeypatch, tmp_path)
    from amlas.training import qwen_sft as training_core

    monkeypatch.setattr(
        training_core, "expected_model_weight_manifest", lambda *_: copy.deepcopy(value.manifest)
    )
    old = value.sft
    train_ref = value.original["request_ref"]
    train = old._records[train_ref]
    from test_sft_producer_snapshot import fixture

    fixture.sf.install_output(old._adapter, value.native, value.sft_s3, train)
    value.native.states[train.id] = "Completed"
    result = old._adapter.artifacts(train)
    assert result["trained"] is True
    checkpoint = VerifiedSftCheckpointSource(
        old.runtime_session_id,
        train.arn,
        "fit_v1",
        old._catalog.source_commit,
        result["checkpoint_ref"],
        tuple(sorted(result["checkpoint_refs"].items())),
    )
    failed = old.reload(train_ref)
    failed_handle = old._records[failed["request_ref"]]
    value.native.states[failed_handle.id] = "Failed"
    assert old._adapter.status(failed_handle).state is JobState.FAILED
    consumer = dataclasses.replace(
        old._catalog,
        source_commit="c" * 40,
        image_uri=old._catalog.image_uri.rsplit("@sha256:", 1)[0] + "@sha256:" + "d" * 64,
        admission_digest="d" * 64,
    )
    selected = ReloadCompatibilitySelection(
        old._catalog,
        consumer,
        checkpoint,
        "qwen3_5_9b.safe_adapter_reload.v1",
        sft_state._reload_base_manifest_sha256(),
    )
    env = dict(value.env)
    env.update(
        AMLAS_QWEN_SFT_SOURCE_COMMIT=consumer.source_commit,
        AMLAS_QWEN_SFT_IMAGE_URI=consumer.image_uri,
        AMLAS_QWEN_SFT_ADMISSION_DIGEST=consumer.admission_digest,
        AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG=json.dumps(
            {"schema_version": 1, "selections": [dataclasses.asdict(selected)]}
        ),
    )

    def reader():
        return sft_binding.resolve_reload_compatibility_catalog(env=env, catalog=consumer)

    current = SessionSftStore(
        old._models,
        old._prepared,
        old._adapter,
        consumer,
        campaign_state=value.campaign,
        execution_binding=value.binding,
        reload_compatibility_reader=reader,
    )
    accepted_current = current.submit(value.model_ref, value.prep_ref, "fit_v1") if current_train else None
    restored = current.resume_original(train.id, value.model_ref, value.prep_ref)
    assert restored["trained"] is True and restored["request_ref"] == train_ref
    recovery = current.reload(train_ref)
    assert recovery["accepted"] and recovery["job_name"] != failed_handle.id
    record = value.campaign.load_original(value.binding, recovery["job_name"])
    assert record.intent.reload_compatibility == selected
    if accepted_current is not None:
        before = len(value.native.calls)
        reused = current.submit(value.model_ref, value.prep_ref, "fit_v1")
        assert reused["request_ref"] == accepted_current["request_ref"] and len(value.native.calls) == before
    return value, current, train_ref, failed_handle, recovery, selected, env


def publication(parts):
    value, current, _train_ref, _failed, recovery, _selected, _env = parts
    repo = ProducerSnapshotRepository(
        value.campaign,
        value.binding,
        dynamodb_client=value.table,
        reload_compatibility_reader=current._reload_compatibility_reader,
    )
    acknowledgement = repo.publish_restore_aliases_from_store(current, recovery["request_ref"])
    return repo, acknowledgement


def test_normal_committed_recovery_hydration_preserves_exact_history_and_successor(monkeypatch, tmp_path):
    parts = recovery_seed(monkeypatch, tmp_path, current_train=True)
    value, _current, train_ref, failed, recovery, selected, env = parts
    repo, acknowledgement = publication(parts)
    raw = value.table.items[acknowledgement["record_key"]]["snapshot"]["B"]
    projection = json.loads(raw)
    assert projection["schema_version"] == 2
    assert projection["training_parent"]["request_ref"] == train_ref
    context = repo.load_original(
        recovery["job_name"], expected_snapshot_sha256=acknowledgement["snapshot_sha256"]
    )
    install_fresh(value, monkeypatch)
    creates = len(value.native.calls)
    _acq, _model, prepared, restored = hydrate_sft_campaign(context, policy=policy(), env=env)
    key = (train_ref, "reload", sft_state.reload_compatibility_digest(selected))
    assert restored._keys[key] == recovery["request_ref"]
    assert restored._catalog == selected.consumer_catalog
    assert restored._records[train_ref].extra["catalog"] == selected.producer_catalog
    assert restored._records[recovery["request_ref"]].id == recovery["job_name"]
    assert restored.reload(train_ref)["job_name"] == recovery["job_name"]
    assert len(value.native.calls) == creates
    assert value.native.states[failed.id] == "Failed" and failed.extra["catalog"] == selected.producer_catalog
    assert prepared._submissions
    with pytest.raises(ValueError):
        restored.submit(value.model_ref, value.prep_ref, "fit_v1")
    assert len(value.native.calls) == creates


@pytest.mark.parametrize("changed", ["absent", "changed", "revoked_at_return"])
def test_recovery_hydration_needs_fresh_independent_current_selection(monkeypatch, tmp_path, changed):
    parts = recovery_seed(monkeypatch, tmp_path)
    value, _current, _train, _failed, recovery, _selected, env = parts
    repo, acknowledgement = publication(parts)
    context = repo.load_original(
        recovery["job_name"], expected_snapshot_sha256=acknowledgement["snapshot_sha256"]
    )
    install_fresh(value, monkeypatch)
    if changed == "absent":
        env.pop("AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG")
    elif changed == "changed":
        value_config = json.loads(env["AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG"])
        value_config["selections"][0]["checkpoint"]["source_commit"] = "e" * 40
        env["AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG"] = json.dumps(value_config)
    else:
        original = sft_state.SessionSftStore.resume_original

        def revoked(self, job_name, *args):
            answer = original(self, job_name, *args)
            if job_name == recovery["job_name"]:
                env.pop("AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG")
            return answer

        monkeypatch.setattr(sft_state.SessionSftStore, "resume_original", revoked)
    before = len(value.native.calls)
    with pytest.raises(ValueError):
        hydrate_sft_campaign(context, policy=policy(), env=env)
    assert len(value.native.calls) == before


def test_recovery_publication_refuses_revoked_selection_and_load_requires_original_sha(monkeypatch, tmp_path):
    parts = recovery_seed(monkeypatch, tmp_path)
    _value, current, _train, _failed, recovery, _selected, env = parts
    repo, _acknowledgement = publication(parts)
    with pytest.raises(ValueError):
        repo.load_original(recovery["job_name"])
    with pytest.raises(ValueError):
        repo.load_original(recovery["job_name"], expected_snapshot_sha256="e" * 64)
    env.pop("AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG")
    with pytest.raises(ValueError):
        repo.publish_restore_aliases_from_store(current, recovery["request_ref"])


@pytest.mark.parametrize(
    "changed",
    [
        "missing_reader",
        "changed_during_row_read",
        "wrong_pid",
        "inherited_repo",
        "exception",
        "malformed",
        "current_catalog_changed",
    ],
)
def test_correct_snapshot_commitment_cannot_mint_context_without_fresh_current_selection(
    monkeypatch, tmp_path, changed
):
    parts = recovery_seed(monkeypatch, tmp_path)
    value, current, _train, _failed, recovery, _selection, env = parts
    repo, acknowledgement = publication(parts)
    if changed == "missing_reader":
        repo = ProducerSnapshotRepository(value.campaign, value.binding, dynamodb_client=value.table)
    elif changed == "changed_during_row_read":
        original = repo._read

        def revoked(*args, **kwargs):
            answer = original(*args, **kwargs)
            if kwargs.get("schema_version") == 2:
                env.pop("AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG")
            return answer

        monkeypatch.setattr(repo, "_read", revoked)
    elif changed == "wrong_pid":
        reader = current._reload_compatibility_reader
        monkeypatch.setattr(
            repo, "_reload_compatibility_reader", lambda: dataclasses.replace(reader(), issuer_pid=-1)
        )
    elif changed == "inherited_repo":
        monkeypatch.setattr(repo, "_pid", -1)
    elif changed == "exception":

        def failed():
            raise RuntimeError("neutral current configuration unavailable")

        monkeypatch.setattr(repo, "_reload_compatibility_reader", failed)
    elif changed == "malformed":
        monkeypatch.setattr(repo, "_reload_compatibility_reader", lambda: {"selections": []})
    else:
        different_catalog = dataclasses.replace(_selection.consumer_catalog, source_commit="e" * 40)
        different = dataclasses.replace(_selection, consumer_catalog=different_catalog)
        different_env = {
            "AMLAS_QWEN_RELOAD_COMPATIBILITY_CATALOG": json.dumps(
                {"schema_version": 1, "selections": [dataclasses.asdict(different)]}
            )
        }
        monkeypatch.setattr(
            repo,
            "_reload_compatibility_reader",
            lambda: sft_binding.resolve_reload_compatibility_catalog(
                env=different_env, catalog=different_catalog
            ),
        )
    before = len(value.native.calls)
    with pytest.raises(ValueError):
        repo.load_original(recovery["job_name"], expected_snapshot_sha256=acknowledgement["snapshot_sha256"])
    assert len(value.native.calls) == before


def test_inherited_repository_cannot_publish_selected_recovery(monkeypatch, tmp_path):
    parts = recovery_seed(monkeypatch, tmp_path)
    _value, current, _train, _failed, recovery, _selected, _env = parts
    repo, _ack = publication(parts)
    monkeypatch.setattr(repo, "_pid", -1)
    with pytest.raises(ValueError):
        repo.publish_restore_aliases_from_store(current, recovery["request_ref"])


def test_selected_recovery_schema1_diagnostic_cannot_mint_context_without_original_sha(monkeypatch, tmp_path):
    parts = recovery_seed(monkeypatch, tmp_path)
    value, current, _train, _failed, recovery, _selected, _env = parts
    repo = ProducerSnapshotRepository(
        value.campaign,
        value.binding,
        dynamodb_client=value.table,
        reload_compatibility_reader=current._reload_compatibility_reader,
    )
    diagnostic = repo.publish_from_store(current, recovery["request_ref"])
    assert diagnostic["published"] is True
    raw = value.table.items[diagnostic["record_key"]]["snapshot"]["B"]
    assert json.loads(raw)["schema_version"] == 1
    with pytest.raises(ValueError):
        repo.load_original(recovery["job_name"])


def test_selected_recovery_deleted_schema2_cannot_downgrade_to_diagnostic_schema1(monkeypatch, tmp_path):
    parts = recovery_seed(monkeypatch, tmp_path)
    value, current, _train, _failed, recovery, _selected, _env = parts
    repo, acknowledgement = publication(parts)
    repo.publish_from_store(current, recovery["request_ref"])
    context = repo.load_original(
        recovery["job_name"], expected_snapshot_sha256=acknowledgement["snapshot_sha256"]
    )
    assert context.original_job_name == recovery["job_name"]
    value.table.items.pop(acknowledgement["record_key"])
    with pytest.raises(ValueError):
        repo.load_original(recovery["job_name"])
    with pytest.raises(ValueError):
        repo.load_original(recovery["job_name"], expected_snapshot_sha256=acknowledgement["snapshot_sha256"])
