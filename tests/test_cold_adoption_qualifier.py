"""Cold adopted-parent reverification must carry its current private IO qualifier.

Reuse the existing neutral hydrate fixture for other parents; adopted-model
capture, store, original codec and nine tiny model artifacts are genuine Source.
No SDK graph or caller authority is serialized, and no native provider is called.
"""

from dataclasses import replace

import test_campaign_state as journal_fixture
import test_qwen_sft_backend as qwen_fixture
import test_verified_model_adoption_sft_parents as parent_fixture
from test_model_pack_adoption_integration import CONSUMER, PACK, f, policy, real_case

from amlas.control_plane import model_acquisition_state as model_state
from amlas.control_plane import normal_run_parents as parents
from amlas.control_plane import sft_state


def test_cold_adopted_reverification_fences_next_provider_io(monkeypatch, tmp_path):
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
    # Real adopted store/provider publication; only the unrelated parent ports
    # retain their established source fixture substitutes.
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
    valid = {"value": True}
    restored_calls = []
    qualifiers = []
    io_after_revocation = []
    provider_restore = case.provider.restore

    def restore_and_revoke(*args, **kwargs):
        restored_calls.append(True)
        if len(restored_calls) == 2:
            valid["value"] = False
        return provider_restore(*args, **kwargs)

    def qualify():
        qualifiers.append(valid["value"])
        if not valid["value"]:
            raise ValueError("Neutral parent witness has been revoked.")

    monkeypatch.setattr(case.provider, "restore", restore_and_revoke)
    for client, method, code in (
        (case.identity, "get_caller_identity", "STS"),
        (case.rows, "get_item", "DDB_GET"),
        (case.objects, "get_object", "S3_GET"),
    ):
        original = getattr(client, method)

        def counted(*args, _original=original, _code=code, **kwargs):
            if not valid["value"]:
                io_after_revocation.append(_code)
            return _original(*args, **kwargs)

        monkeypatch.setattr(client, method, counted)
    counts = (case.copies.calls, case.rows.updates, len(case.native.requests))
    hydrated = None
    refused = False
    try:
        hydrated = parents.hydrate_parent_graph(raw, selected.binding, live_policy, qualify=qualify)
    except ValueError:
        refused = True
    assert len(restored_calls) == 2 and valid["value"] is False
    assert io_after_revocation == [], "cold_parent_qualifier_did_not_fence_reverification_io"
    assert refused and hydrated is None
    assert False in qualifiers
    assert (case.copies.calls, case.rows.updates, len(case.native.requests)) == counts
