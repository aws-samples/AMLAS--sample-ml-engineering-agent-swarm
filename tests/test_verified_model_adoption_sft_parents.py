"""Typed adoption and normal parent model-branch controls, over neutral descriptors.

No model bytes, native job, approval witness or private artifact is read. Other
parent ports are inert branch fixtures; existing DATA/CASE/PREP guards stay owned
by their separate native-shaped suites. These tests grant no AWS admission.
"""

from __future__ import annotations

import copy
import inspect
import threading
from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_campaign_state import CAMPAIGN, state
from test_qwen_sft_backend import catalog

from amlas.backends.ports import DataAcquisitionBackend
from amlas.backends.registry import REGISTRY
from amlas.backends.types import DatasetArtifactRef, JobHandle
from amlas.control_plane import normal_run_parents as parents
from amlas.control_plane import sft_state as sft
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.campaign_state import _binding_fields, _mint_binding
from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest

OLD = "pack-producer"
CURRENT = "pack-consumer"
MODEL_REF = "model_" + "a" * 32
PREP_REF = "prep_" + "b" * 32
CASE_REF = "case_" + "c" * 32
DATA_REF = "acq_" + "d" * 32


def _reference(session, path, *, sha="a" * 64, count=7, fmt="json"):
    return DatasetArtifactRef(
        "s3://" + catalog().bucket + "/sessions/" + session + "/verified-pack/" + path,
        "version-" + session + "-" + path.replace("/", "-"),
        sha,
        count,
        fmt,
    )


def _original():
    files = []
    for item in expected_model_weight_manifest("qwen3_5_9b", "1")["files"]:
        fmt = (
            "safetensors"
            if item["role"] == "weight_shard"
            else "jinja"
            if item["role"] == "chat_template"
            else "json"
        )
        files.append(
            (
                item["path"],
                _reference(OLD, item["path"], sha=item["sha256"], count=item["byte_count"], fmt=fmt),
            )
        )
    return sft.VerifiedSftModelSource(
        OLD,
        "qwen3_5_9b",
        "1",
        "arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":processing-job/owned-model-pack",
        tuple(
            dict(
                runtime_session_id=OLD,
                deployment_id=catalog().deployment_id,
                deployment_prefix=catalog().deployment_prefix,
            ).items()
        ),
        tuple(files),
        tuple((name, _reference(OLD, name + ".json")) for name in ("commit", "lineage", "receipt")),
    )


def _adopted():
    cls = getattr(sft, "VerifiedSftAdoptedModelSource", None)
    assert cls is not None, "exact typed consumer-owned adopted SFT model source is missing"
    original = _original()

    def copied(ref):
        return replace(
            ref,
            uri=ref.uri.replace("/sessions/" + OLD + "/", "/sessions/" + CURRENT + "/"),
            version_id="consumer-" + ref.version_id,
        )

    return cls(
        CURRENT,
        original.catalog_id,
        original.catalog_version,
        original,
        tuple((name, copied(ref)) for name, ref in original.file_refs),
        tuple((name, copied(ref)) for name, ref in original.manifest_refs),
        _reference(CURRENT, "ADOPTED.json"),
    )


def test_adopted_sft_source_preserves_original_and_current_exact_refs():
    adopted = _adopted()
    original = copy.deepcopy(adopted.original)
    sft.validate_model(adopted, catalog(), CURRENT)
    assert adopted.original == original and adopted.original.runtime_session_id == OLD
    assert adopted.original.producer_job_arn.endswith("owned-model-pack")
    assert adopted.file_refs != original.file_refs
    assert inspect.isclass(type(adopted)) and type(adopted).__dataclass_params__.frozen
    sft.validate_model(original, catalog(), OLD)
    with pytest.raises(ValueError):
        sft.validate_model(original, catalog(), CURRENT)


@pytest.mark.parametrize(
    "mutation",
    [
        "consumer_session",
        "catalog",
        "version",
        "original_session",
        "original_producer",
        "original_hash",
        "foreign_file",
        "file_hash",
        "file_count",
        "file_format",
        "manifest_bytes",
        "foreign_manifest",
        "manifest_format",
        "foreign_adoption",
        "adoption_format",
        "adoption_version",
        "adoption_bound",
        "duplicate_files",
    ],
)
def test_adopted_sft_source_refuses_foreign_or_changed_evidence(mutation):
    value = _adopted()
    files = list(value.file_refs)
    manifests = list(value.manifest_refs)
    if mutation == "consumer_session":
        value = replace(value, runtime_session_id=OLD)
    elif mutation == "catalog":
        value = replace(value, catalog_id="other")
    elif mutation == "version":
        value = replace(value, catalog_version="2")
    elif mutation == "original_session":
        value = replace(value, original=replace(value.original, runtime_session_id=CURRENT))
    elif mutation == "original_producer":
        value = replace(value, original=replace(value.original, producer_job_arn="foreign"))
    elif mutation == "original_hash":
        original_files = list(value.original.file_refs)
        original_files[0] = (original_files[0][0], replace(original_files[0][1], sha256="f" * 64))
        value = replace(value, original=replace(value.original, file_refs=tuple(original_files)))
    elif mutation.startswith("file_") or mutation == "foreign_file":
        changes = {
            "foreign_file": {"uri": value.original.file_refs[0][1].uri},
            "file_hash": {"sha256": "f" * 64},
            "file_count": {"byte_count": files[0][1].byte_count + 1},
            "file_format": {"format": "jsonl"},
        }
        files[0] = (files[0][0], replace(files[0][1], **changes[mutation]))
        value = replace(value, file_refs=tuple(files))
    elif mutation.startswith("manifest_") or mutation == "foreign_manifest":
        changes = {
            "manifest_bytes": {"sha256": "f" * 64},
            "foreign_manifest": {"uri": value.original.manifest_refs[0][1].uri},
            "manifest_format": {"format": "safetensors"},
        }
        manifests[0] = (manifests[0][0], replace(manifests[0][1], **changes[mutation]))
        value = replace(value, manifest_refs=tuple(manifests))
    elif mutation == "duplicate_files":
        value = replace(value, file_refs=(files[0],) * len(files))
    else:
        changes = {
            "foreign_adoption": {"uri": _reference(OLD, "ADOPTED.json").uri},
            "adoption_format": {"format": "jsonl"},
            "adoption_version": {"version_id": "null"},
            "adoption_bound": {"byte_count": 1024**2 + 1},
        }
        value = replace(value, adoption_ref=replace(value.adoption_ref, **changes[mutation]))
    with pytest.raises(ValueError):
        sft.validate_model(value, catalog(), CURRENT)


def test_adopted_sft_source_requires_exact_type():
    value = _adopted()

    class Derived(type(value)):
        pass

    for untyped in (SimpleNamespace(**asdict(value)), Derived(**value.__dict__)):
        with pytest.raises(ValueError):
            sft.validate_model(untyped, catalog(), CURRENT)


def test_legacy_model_still_refuses_foreign_session_without_adoption():
    original = _original()
    sft.validate_model(original, catalog(), OLD)
    with pytest.raises(ValueError):
        sft.validate_model(original, catalog(), CURRENT)


def _forbidden(message):
    raise AssertionError(message)


def _parent_fixture(monkeypatch):
    from amlas.config import externalized
    from amlas.control_plane import (
        acquisition_state,
        case_binding,
        model_acquisition_state,
        normal_run_checkpoint,
        preparation_binding,
    )

    binding = _mint_binding(state(), CAMPAIGN, CURRENT)
    scope = _binding_fields(binding)
    policy = ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"])
    qualify_calls = []

    def qualify():
        qualify_calls.append(True)

    adapter = SimpleNamespace(
        _region=scope["region"],
        _role="owned-role",
        _image="owned-image",
        _bucket=catalog().bucket,
        _source="a" * 40,
        _description=lambda handle: {},
    )
    data_extra = dict.fromkeys(parents._DATA_FIELDS, "neutral")
    data_extra.update(
        region=adapter._region,
        role_arn=adapter._role,
        image_uri=adapter._image,
        bucket=adapter._bucket,
        source_commit=adapter._source,
    )
    data_handle = JobHandle("owned-data", "processing", "neutral-data-arn", data_extra)
    acquired = SimpleNamespace(
        _lock=threading.RLock(),
        _records={DATA_REF: SimpleNamespace(adapter=adapter, handle=data_handle)},
        _verified_preparation_source=lambda ref: {"descriptor": "unchanged-data"},
        _restore_original_producers=lambda entries, owner: None,
    )
    case_handle = JobHandle(
        "owned-case",
        "case_freeze",
        "neutral-case-arn",
        {
            "header": {},
            "source": {},
            "request": {},
            "output_prefix": "neutral/",
            "binding_ref": _reference(CURRENT, "CASE.json"),
        },
    )
    case_refs = {"cases": _reference(CURRENT, "CASES.json")}
    cases = SimpleNamespace(
        _lock=threading.RLock(),
        _keys={(DATA_REF, "humaneval_164", "1"): CASE_REF},
        _catalog=catalog(),
        _adapter=adapter,
        _verified=lambda ref: (case_handle, {"frozen": True}, case_refs),
        _restore_original_producer=lambda *args: None,
    )
    prep_handle = JobHandle("owned-prep", "preparation", "neutral-prep-arn", {})
    prepared = SimpleNamespace(
        _lock=threading.RLock(),
        _submissions={((DATA_REF,), CASE_REF): PREP_REF},
        _records={PREP_REF: SimpleNamespace(adapter=adapter)},
        _acquired=acquired,
        _case_store=cases,
        _catalog=catalog(),
        _adapter=adapter,
        _original_producer_handle=lambda ref, owner: prep_handle,
        _verified_sft_snapshot=lambda ref: catalog(),
        _restore_original_producer=lambda *args: None,
        _restore_original_selection=lambda *args: None,
    )
    trace = []

    @contextmanager
    def reads(adapters, region, current):
        assert all(item is adapter for item in adapters), "adopted MODEL adapter must not be reconstructed"
        current()
        yield
        current()

    def resolve(port):
        assert port is DataAcquisitionBackend, "adopted cold restore must not resolve a legacy MODEL adapter"
        return SimpleNamespace(adapter=adapter)

    def producer(value, current_scope, *, model):
        assert model is False, "adopted origin must not enter original MODEL decoder"
        return prep_handle

    monkeypatch.setattr(REGISTRY, "resolve", resolve)
    monkeypatch.setattr(parents, "_qualified_reads", reads)
    monkeypatch.setattr(parents, "_source_wire", lambda value: value)
    monkeypatch.setattr(parents, "_source", lambda value: value)
    monkeypatch.setattr(parents, "_data_handle", lambda row, current_scope: data_handle)
    monkeypatch.setattr(parents, "_case_handle", lambda row, current_cases, current_scope: case_handle)
    monkeypatch.setattr(parents, "_producer", producer)
    monkeypatch.setattr(parents, "_capture", lambda handle, ref, *, model: {"request_ref": ref})
    monkeypatch.setattr(
        acquisition_state.SessionAcquisitionStore, "_for_original_correlation", lambda owner: acquired
    )
    monkeypatch.setattr(case_binding, "build_case_freeze_store", lambda *args, **kwargs: cases)
    monkeypatch.setattr(preparation_binding, "build_preparation_store", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        externalized,
        "model_weight_acquisition_selectors",
        lambda **kwargs: _forbidden("legacy MODEL selectors used for adoption"),
    )
    monkeypatch.setattr(
        normal_run_checkpoint,
        "current_normal_run",
        lambda: SimpleNamespace(
            run_id=CAMPAIGN.replace("-", ""),
            _captured_sft_call=lambda *args: {"pending_tool_use_id": "neutral-sft", "causal_prefix": []},
        ),
    )

    def snapshot(ref, owner, policy=None, qualify=None):
        trace.append(("snapshot", ref, owner, policy, qualify))
        assert ref == MODEL_REF and owner is binding and qualify is not None
        qualify()
        return copy.deepcopy(provider_snapshot)

    def restore(value, owner, current_policy, qualify):
        trace.append(("restore", value, owner, current_policy, qualify))
        assert value == provider_snapshot and owner is binding and current_policy is policy
        qualify()
        return MODEL_REF

    models = SimpleNamespace(
        _lock=threading.RLock(),
        _records={},
        _catalog=catalog(),
        _adopted_parent_snapshot=snapshot,
        _restore_adopted_parent=restore,
        _original_producer_handle=lambda *args: _forbidden(
            "adopted model must not manufacture an original handle"
        ),
        _restore_original_producer=lambda *args: _forbidden(
            "adopted restore must not recreate an original producer"
        ),
        _verified_sft_model_snapshot=lambda ref, *, _qualify=None: _adopted(),
    )
    provider_snapshot = {
        "schema_version": 1,
        "origin": "verified_pack_adoption",
        "pack_ref": "pack_" + "e" * 64,
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "runtime_session_id": CURRENT,
        "producer": {"original": "unchanged-producer-codec"},
        "original_artifacts": {"file_refs": [], "manifest_refs": []},
        "file_refs": [],
        "manifest_refs": [],
        "adoption_ref": asdict(_reference(CURRENT, "ADOPTED.json")),
    }
    monkeypatch.setattr(
        model_acquisition_state,
        "build_model_acquisition_store",
        lambda runtime_session_id, **kwargs: models,
        raising=False,
    )
    store = object.__new__(sft.SessionSftStore)
    store._execution_binding, store._models, store._prepared, store._catalog = (
        binding,
        models,
        prepared,
        catalog(),
    )
    return SimpleNamespace(
        binding=binding,
        policy=policy,
        qualify=qualify,
        models=models,
        store=store,
        trace=trace,
        snapshot=provider_snapshot,
        qualify_calls=qualify_calls,
    )


def test_adopted_parent_capture_propagates_selected_failure_without_legacy_fallback(monkeypatch):
    fixture = _parent_fixture(monkeypatch)

    def refused(*args, **kwargs):
        raise ValueError("selected adopted parent is unavailable")

    fixture.models._adopted_parent_snapshot = refused
    with pytest.raises(ValueError, match="selected adopted parent is unavailable"):
        parents.publish_parent_graph(
            fixture.store, MODEL_REF, PREP_REF, run_id=CAMPAIGN.replace("-", ""), qualify=fixture.qualify
        )


def test_adopted_parent_roundtrip_preserves_snapshot_and_uses_read_only_store_hook(monkeypatch):
    fixture = _parent_fixture(monkeypatch)
    raw = parents.publish_parent_graph(
        fixture.store, MODEL_REF, PREP_REF, run_id=CAMPAIGN.replace("-", ""), qualify=fixture.qualify
    )
    value = parents._decode(raw)
    assert value["schema_version"] == 1
    assert value["model"] == {
        "origin": "verified_pack_adoption",
        "request_ref": MODEL_REF,
        "snapshot": fixture.snapshot,
    }
    frozen_snapshot = copy.deepcopy(fixture.snapshot)
    restored = parents.hydrate_parent_graph(raw, fixture.binding, fixture.policy, qualify=fixture.qualify)
    assert restored[1] is fixture.models
    assert fixture.snapshot == frozen_snapshot
    assert [item[0] for item in fixture.trace] == ["snapshot", "restore"]
    assert fixture.trace[-1][3:] == (fixture.policy, fixture.qualify)
    assert fixture.qualify_calls


@pytest.mark.parametrize("failure", ["missing", "changed", "denied"])
def test_adopted_parent_restore_refuses_without_legacy_or_effect_fallback(monkeypatch, failure):
    fixture = _parent_fixture(monkeypatch)
    raw = parents.publish_parent_graph(
        fixture.store, MODEL_REF, PREP_REF, run_id=CAMPAIGN.replace("-", ""), qualify=fixture.qualify
    )
    if failure == "changed":
        actual = _adopted()
        fixture.models._verified_sft_model_snapshot = lambda ref, *, _qualify=None: replace(
            actual, adoption_ref=replace(actual.adoption_ref, version_id="changed")
        )
    elif failure == "missing":

        def missing(*args, **kwargs):
            raise ValueError("selected adopted parent is missing")

        fixture.models._restore_adopted_parent = missing
    else:
        fixture.policy.allowlist.clear()
    with pytest.raises(ValueError):
        parents.hydrate_parent_graph(raw, fixture.binding, fixture.policy, qualify=fixture.qualify)
