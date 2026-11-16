"""Cold restore of an ADOPTED verified model parent must not index the non-adopted record map.

In production the model parent of a normal run is an adopted verified pack: the cold model store holds it
in `_adoptions` and leaves `_records` empty (`_restore_adopted_parent`). `restore_sft_store` and
`qualify_original_completion` looked the model adapter up in `_records` and raised KeyError on the real
replacement restore. Every earlier restore test used a non-adopted fixture.

The parents are built the production way: a warm store adopts the pack, the parent graph is published,
a cold store restores it through `hydrate_parent_graph`. Only the context-bound row reads and the SFT
store construction are replaced by the seams below; no native provider is called and no AWS access is
granted. The reads that decide which adapters are fenced are recorded, so a test states exactly which
adapters the restore reads through.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_campaign_state as journal_fixture
import test_qwen_sft_backend as qwen_fixture
import test_verified_model_adoption_sft_parents as parent_fixture
from test_model_pack_adoption_integration import CONSUMER, PACK, f, policy, real_case

from amlas.control_plane import model_acquisition_state as model_state
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.control_plane import normal_run_parents as parents
from amlas.control_plane import producer_snapshot, sft_binding, sft_state

OPERATION_DIGEST = "e" * 64
SFT_ALIAS = "sft_" + OPERATION_DIGEST[:32]
JOB_NAME = "amlas-qwen-test-sft-" + "1" * 40
JOB_ARN = "arn:aws:sagemaker:us-west-2:" + f.ACCOUNT + ":training-job/" + JOB_NAME


def _adopted_cold_parents(monkeypatch, tmp_path):
    """Return (hydrated parents, model alias, binding, live policy) of a cold adopted restore."""
    case = real_case(monkeypatch, tmp_path)
    live_policy = policy()
    warm = model_state.SessionModelAcquisitionStore(
        runtime_session_id=CONSUMER,
        adoption_provider=case.provider,
        policy_getter=lambda: live_policy,
    )
    result = warm.reuse_verified(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, pack_ref=PACK, policy=live_policy
    )
    model_ref = result["request_ref"]
    f.patch_expected(monkeypatch, [sft_state])
    monkeypatch.setattr(qwen_fixture, "ACCOUNT", f.ACCOUNT)
    monkeypatch.setattr(qwen_fixture, "BUCKET", f.BUCKET)
    monkeypatch.setattr(qwen_fixture, "IMAGE", f.IMAGE)
    monkeypatch.setattr(
        qwen_fixture,
        "HEADER",
        {
            **qwen_fixture.HEADER,
            "deployment_id": f.DEPLOYMENT,
            "deployment_prefix": "fixture-lab",
            "image_uri": f.IMAGE,
        },
    )
    monkeypatch.setattr(parent_fixture, "CURRENT", CONSUMER)
    monkeypatch.setattr(parent_fixture, "MODEL_REF", model_ref)
    monkeypatch.setattr(
        parent_fixture, "catalog", lambda: qwen_fixture.catalog(deployment_prefix="fixture-lab")
    )
    original_state = journal_fixture.state
    monkeypatch.setattr(
        parent_fixture,
        "state",
        lambda: replace(
            original_state(),
            account_id=f.ACCOUNT,
            deployment_id=f.DEPLOYMENT,
            deployment_prefix="fixture-lab",
            jobs_table="fixture-lab-training-jobs",
            execution_role_arn="arn:aws:iam::" + f.ACCOUNT + ":role/owned-control-plane",
        ),
    )
    selected = parent_fixture._parent_fixture(monkeypatch)
    selected.store._models = warm
    raw = parents.publish_parent_graph(
        selected.store,
        model_ref,
        parent_fixture.PREP_REF,
        run_id=journal_fixture.CAMPAIGN.replace("-", ""),
        qualify=selected.qualify,
    )
    cold = model_state.SessionModelAcquisitionStore(
        runtime_session_id=CONSUMER,
        adoption_provider=case.provider,
        policy_getter=lambda: live_policy,
    )
    monkeypatch.setattr(model_state, "build_model_acquisition_store", lambda *args, **kwargs: cold)
    hydrated = parents.hydrate_parent_graph(raw, selected.binding, live_policy, qualify=selected.qualify)
    acquired, _models, cases, prepared = hydrated
    for inert in (acquired, cases, prepared):
        # The unrelated parent ports are inert branch fixtures; give them the correlation the restore reads.
        if not hasattr(inert, "runtime_session_id"):
            inert.runtime_session_id = selected.binding.runtime_session_id
    if not hasattr(prepared, "_adapter"):
        prepared._adapter = SimpleNamespace(name="prepared-adapter")
    return hydrated, model_ref, selected.binding, live_policy


def _sft_world(model_ref, binding):
    """A neutral original SFT reservation and store; the model store itself stays the real one."""
    original = SimpleNamespace(
        native_job_name=JOB_NAME,
        native_job_arn=JOB_ARN,
        operation_digest=OPERATION_DIGEST,
        intent=SimpleNamespace(source_refs=(model_ref, parent_fixture.PREP_REF)),
    )
    sft_adapter = SimpleNamespace(name="sft-adapter")
    result = {
        "request_ref": SFT_ALIAS,
        "job_name": JOB_NAME,
        "job_arn": JOB_ARN,
        "status": "COMPLETED",
        "trained": True,
        "full_training_complete": True,
    }
    handle = SimpleNamespace(extra={"restoration_status": "verified"})
    built = SimpleNamespace(
        _adapter=sft_adapter,
        _records={SFT_ALIAS: handle},
        resume_original=lambda name, *refs: result,
        observe=lambda alias: dict(result),
    )
    return original, built, sft_adapter


def _install_seams(monkeypatch, original, binding, built):
    """Replace only the context-bound row reads and the SFT store builder; record the fenced adapters."""
    fenced = []

    @contextmanager
    def native(context):
        yield object(), "table-id", (lambda: None), object()

    @contextmanager
    def qualified_reads(adapters, region, current):
        fenced.append(tuple(adapters))
        yield

    record = {"sft": {"native_job_name": JOB_NAME}, "private_store_correlation": binding.runtime_session_id}
    monkeypatch.setattr(checkpoint, "_native", native)
    monkeypatch.setattr(checkpoint, "_load", lambda *args, **kwargs: ({}, record))
    monkeypatch.setattr(checkpoint, "_original", lambda *args, **kwargs: (original, binding))
    monkeypatch.setattr(checkpoint, "_record_recheck", lambda check: lambda: None)
    monkeypatch.setattr(checkpoint, "_Journal", lambda context, table_id: object())
    monkeypatch.setattr(producer_snapshot, "_approved_read", lambda selected: None)
    monkeypatch.setattr(parents, "_qualified_reads", qualified_reads)
    monkeypatch.setattr(sft_binding, "build_sft_store", lambda *args, **kwargs: built)
    return fenced


CONTEXT = SimpleNamespace(catalog=SimpleNamespace(region="us-west-2"))


def _restore(parents_tuple, original, binding):
    return checkpoint.NormalRunContext.restore_sft_store(
        CONTEXT, parents_tuple, original, binding, policy=None
    )


def _claim_qualifier(models, prepared, built, original, binding):
    store = SimpleNamespace(
        runtime_session_id=binding.runtime_session_id,
        _source_refs={SFT_ALIAS: original.intent.source_refs},
        _models=models,
        _prepared=prepared,
        _adapter=built._adapter,
        observe=built.observe,
    )
    task = {"native_job_name": JOB_NAME}
    return checkpoint.NormalRunContext.qualify_original_completion(CONTEXT, task, store, policy=None)


def test_adopted_parent_restore_reads_through_prepared_and_sft_only(monkeypatch, tmp_path):
    hydrated, model_ref, binding, _policy = _adopted_cold_parents(monkeypatch, tmp_path)
    _acquired, models, _cases, prepared = hydrated
    assert models._records == {} and model_ref in models._adoptions
    original, built, sft_adapter = _sft_world(model_ref, binding)
    fenced = _install_seams(monkeypatch, original, binding, built)
    sft, result = _restore(hydrated, original, binding)
    assert sft is built and result["request_ref"] == SFT_ALIAS
    assert fenced == [(prepared._adapter, sft_adapter)]


def test_adopted_parent_claim_qualifier_reads_through_prepared_and_sft_only(monkeypatch, tmp_path):
    hydrated, model_ref, binding, _policy = _adopted_cold_parents(monkeypatch, tmp_path)
    _acquired, models, _cases, prepared = hydrated
    original, built, sft_adapter = _sft_world(model_ref, binding)
    fenced = _install_seams(monkeypatch, original, binding, built)
    _claim_qualifier(models, prepared, built, original, binding)
    assert fenced == [(prepared._adapter, sft_adapter)]


def test_adopted_cold_resume_chain_restore_then_claim(monkeypatch, tmp_path):
    """hydrate_parent_graph -> restore_sft_store -> the claim qualifier, one adopted graph."""
    hydrated, model_ref, binding, _policy = _adopted_cold_parents(monkeypatch, tmp_path)
    _acquired, models, _cases, prepared = hydrated
    original, built, sft_adapter = _sft_world(model_ref, binding)
    fenced = _install_seams(monkeypatch, original, binding, built)
    sft, result = _restore(hydrated, original, binding)
    assert result["status"] == "COMPLETED"
    _claim_qualifier(models, prepared, sft, original, binding)
    assert fenced == [(prepared._adapter, sft_adapter)] * 2


def _record_store(alias, adapter):
    store = model_state.SessionModelAcquisitionStore(runtime_session_id=CONSUMER)
    store._records[alias] = SimpleNamespace(adapter=adapter)
    return store


def test_non_adopted_parent_still_reads_through_the_model_adapter(monkeypatch, tmp_path):
    hydrated, _adopted_ref, binding, _policy = _adopted_cold_parents(monkeypatch, tmp_path)
    acquired, _models, cases, prepared = hydrated
    alias = "model_" + "a" * 32
    model_adapter = SimpleNamespace(name="model-adapter")
    models = _record_store(alias, model_adapter)
    original, built, sft_adapter = _sft_world(alias, binding)
    fenced = _install_seams(monkeypatch, original, binding, built)
    _restore((acquired, models, cases, prepared), original, binding)
    _claim_qualifier(models, prepared, built, original, binding)
    assert fenced == [(model_adapter, prepared._adapter, sft_adapter)] * 2


@pytest.mark.parametrize("use_adopted", [True, False])
def test_mixed_model_store_selects_by_the_original_alias(monkeypatch, tmp_path, use_adopted):
    hydrated, adopted_ref, binding, _policy = _adopted_cold_parents(monkeypatch, tmp_path)
    acquired, adopted_models, cases, prepared = hydrated
    record_alias = "model_" + "b" * 32
    model_adapter = SimpleNamespace(name="model-adapter")
    adopted_models._records[record_alias] = SimpleNamespace(adapter=model_adapter)
    alias = adopted_ref if use_adopted else record_alias
    original, built, sft_adapter = _sft_world(alias, binding)
    fenced = _install_seams(monkeypatch, original, binding, built)
    _restore((acquired, adopted_models, cases, prepared), original, binding)
    _claim_qualifier(adopted_models, prepared, built, original, binding)
    expected = (
        (prepared._adapter, sft_adapter) if use_adopted else (model_adapter, prepared._adapter, sft_adapter)
    )
    assert fenced == [expected] * 2


def test_unknown_model_alias_is_still_refused(monkeypatch, tmp_path):
    hydrated, _adopted_ref, binding, _policy = _adopted_cold_parents(monkeypatch, tmp_path)
    original, built, _sft_adapter = _sft_world("model_" + "c" * 32, binding)
    fenced = _install_seams(monkeypatch, original, binding, built)
    with pytest.raises((KeyError, ValueError)):
        _restore(hydrated, original, binding)
    with pytest.raises((KeyError, ValueError)):
        _claim_qualifier(hydrated[1], hydrated[3], built, original, binding)
    assert fenced == []
