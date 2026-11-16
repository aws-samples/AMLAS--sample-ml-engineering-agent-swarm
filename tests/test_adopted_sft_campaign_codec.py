"""Closed adopted SFT codec and existing fake-native journal roundtrip controls.

No live AWS model, artifact or state is read; no training Create is permitted.
"""

import copy
from dataclasses import replace

import pytest
from test_campaign_state import CAMPAIGN, fixture, state
from test_qwen_sft_backend import SESSION, catalog, sources
from test_verified_model_adoption_sft_parents import CURRENT, OLD, _adopted

from amlas.backends.adapters.campaign_state import DynamoCampaignStateBackend
from amlas.control_plane import campaign_state as campaign
from amlas.control_plane.sft_state import VerifiedSftAdoptedModelSource, VerifiedSftInputs, validate_inputs


def adopted_inputs():
    model = _adopted()

    def current(ref):
        return replace(ref, uri=ref.uri.replace("/sessions/" + CURRENT + "/", "/sessions/" + SESSION + "/"))

    model = replace(
        model,
        runtime_session_id=SESSION,
        file_refs=tuple((path, current(ref)) for path, ref in model.file_refs),
        manifest_refs=tuple((path, current(ref)) for path, ref in model.manifest_refs),
        adoption_ref=current(model.adoption_ref),
    )
    inputs = VerifiedSftInputs(model, sources()[1].prepared)
    validate_inputs(inputs, catalog(), SESSION)
    return inputs


def test_exact_adopted_sft_inputs_roundtrip_closed_campaign_codec():
    inputs = adopted_inputs()
    try:
        packed = campaign._pack(inputs)
    except campaign.CampaignStateError:
        raise AssertionError("adopted SFT type is absent from the closed campaign codec") from None
    restored = campaign._unpack(packed)
    assert type(restored) is VerifiedSftInputs
    assert type(restored.model) is VerifiedSftAdoptedModelSource
    assert restored == inputs and restored.model.original.runtime_session_id == OLD
    assert restored.model.runtime_session_id == SESSION
    malformed = copy.deepcopy(packed)
    malformed["fields"]["model"]["fields"]["unexpected"] = "neutral"
    with pytest.raises(campaign.CampaignStateError):
        campaign._unpack(malformed)


def test_adopted_sft_intent_survives_existing_native_journal_reservation_reload():
    store, binding, _legacy, ddb, sts, native = fixture()
    inputs = adopted_inputs()
    try:
        intent = campaign.build_sft_intent(binding, catalog(), inputs)
        reservation = store.reserve_original_job(binding, intent)
    except campaign.CampaignStateError:
        raise AssertionError("genuine adopted SFT intent cannot persist in the existing journal") from None
    restarted = DynamoCampaignStateBackend(
        state(),
        dynamodb_client=ddb,
        sts_client=sts,
        sagemaker_client=native,
    )
    current = restarted.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=SESSION)
    restored = restarted.load_original(current, reservation.native_job_name)
    assert restored.intent.inputs == inputs
    assert type(restored.intent.inputs.model) is VerifiedSftAdoptedModelSource
    assert restored.intent.inputs.model.original == inputs.model.original
    assert restored.intent.inputs.model.original.runtime_session_id == OLD
    assert restored.intent.inputs.model.runtime_session_id == SESSION
    assert restored.native_job_name == reservation.native_job_name
    assert restored.operation_digest == campaign.operation_digest(intent)
    assert len(ddb.items) == 1 and restored.disposition == "RESERVED"


def test_genuine_adopted_store_source_survives_sft_journal_and_cold_store_restore(monkeypatch, tmp_path):
    import model_acquisition_fixtures as f
    import test_campaign_state as journal_fixture
    import test_qwen_sft_backend as preparation_fixture
    from botocore.validate import validate_parameters
    from test_model_pack_adoption_integration import CONSUMER, PACK, policy, real_case

    from amlas.backends.adapters import qwen_sft as sft_adapter
    from amlas.control_plane import sft_state
    from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore

    case = real_case(monkeypatch, tmp_path)
    # The original codec imports the already patched model_weight_manifest alias
    # locally. SFT consumes exactly the same nine tiny expected byte pins.
    f.patch_expected(monkeypatch, [sft_state, sft_adapter])
    live_policy = policy()
    models = SessionModelAcquisitionStore(
        runtime_session_id=CONSUMER,
        adoption_provider=case.provider,
        policy_getter=lambda: live_policy,
    )
    original_producer = copy.deepcopy(case.producer)
    result = models.reuse_verified(
        catalog_id=f.CATALOG,
        catalog_version=f.VERSION,
        pack_ref=PACK,
        policy=live_policy,
    )
    reference = result["request_ref"]
    assert result["status"] == "reused_verified" and result["native_processing_created"] is False
    model = models._verified_sft_model_snapshot(reference)
    assert type(model) is VerifiedSftAdoptedModelSource
    assert model.original.producer_job_arn == case.handle.arn
    assert model.original.runtime_session_id == f.SESSION
    assert model.runtime_session_id == CONSUMER

    # Reuse the existing preparation byte fixture with source-selected namespace
    # inputs; neither model/provider hooks nor source/native codecs are replaced.
    monkeypatch.setattr(preparation_fixture, "ACCOUNT", f.ACCOUNT)
    monkeypatch.setattr(preparation_fixture, "BUCKET", f.BUCKET)
    monkeypatch.setattr(preparation_fixture, "SESSION", CONSUMER)
    image = f.ACCOUNT + ".dkr.ecr." + f.REGION + ".amazonaws.com/source@sha256:" + "b" * 64
    monkeypatch.setattr(preparation_fixture, "IMAGE", image)
    monkeypatch.setattr(
        preparation_fixture,
        "HEADER",
        {
            **preparation_fixture.HEADER,
            "job_name": "fixture-lab-data-prep-a",
            "runtime_session_id": CONSUMER,
            "deployment_id": f.DEPLOYMENT,
            "deployment_prefix": "fixture-lab",
            "image_uri": image,
        },
    )
    catalog = preparation_fixture.catalog(deployment_prefix="fixture-lab")
    prepared = replace(
        preparation_fixture.sources()[1].prepared,
        producer_job_arn="arn:aws:sagemaker:"
        + f.REGION
        + ":"
        + f.ACCOUNT
        + ":processing-job/fixture-lab-data-prep-a",
    )
    inputs = VerifiedSftInputs(model, prepared)
    validate_inputs(inputs, catalog, CONSUMER)
    scope = replace(
        journal_fixture.state(),
        account_id=f.ACCOUNT,
        execution_role_arn="arn:aws:iam::" + f.ACCOUNT + ":role/owned-control-plane",
        deployment_prefix="fixture-lab",
        jobs_table="fixture-lab-training-jobs",
    )

    class JournalRows(journal_fixture.DDB):
        def _shape(self, operation, values):
            validate_parameters(
                values, journal_fixture.MODELS["dynamodb"].operation_model(operation).input_shape
            )
            assert values["TableName"] == scope.jobs_table
            self.calls.append((operation, copy.deepcopy(values)))

    rows, identity, native = JournalRows(), journal_fixture.STS(), journal_fixture.SM()
    identity.account = f.ACCOUNT
    journal = DynamoCampaignStateBackend(
        scope,
        dynamodb_client=rows,
        sts_client=identity,
        sagemaker_client=native,
    )
    binding = journal.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=CONSUMER)
    intent = campaign.build_sft_intent(binding, catalog, inputs)
    reservation = journal.reserve_original_job(binding, intent)
    restarted = DynamoCampaignStateBackend(
        scope,
        dynamodb_client=rows,
        sts_client=identity,
        sagemaker_client=native,
    )
    fresh_binding = restarted.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=CONSUMER)
    loaded = restarted.load_original(fresh_binding, reservation.native_job_name)
    assert loaded.intent.inputs == inputs
    assert loaded.intent.inputs.model.original.producer_job_arn == case.handle.arn
    assert loaded.intent.inputs.model.manifest_refs == model.manifest_refs

    checks = []

    def qualify():
        checks.append(True)

    snapshot = models._adopted_parent_snapshot(reference, binding, live_policy, qualify=qualify)
    counts = (case.copies.calls, case.rows.updates, case.objects.writes, len(case.native.requests))
    cold = SessionModelAcquisitionStore(
        runtime_session_id=CONSUMER,
        adoption_provider=case.provider,
        policy_getter=lambda: live_policy,
    )
    restored_ref = cold._restore_adopted_parent(snapshot, fresh_binding, live_policy, qualify=qualify)
    restored = cold._verified_sft_model_snapshot(restored_ref)
    assert restored_ref == reference and restored == model
    assert restored.original.runtime_session_id == f.SESSION and restored.runtime_session_id == CONSUMER
    assert restored.original.producer_job_arn == case.handle.arn
    assert all(
        ref.uri.startswith("s3://" + f.BUCKET + "/sessions/" + CONSUMER + "/")
        for _, ref in restored.manifest_refs
    )
    assert case.producer == original_producer
    assert (case.copies.calls, case.rows.updates, case.objects.writes, len(case.native.requests)) == counts
    assert checks
    cold_intent = campaign.build_sft_intent(fresh_binding, catalog, VerifiedSftInputs(restored, prepared))
    assert campaign.operation_digest(cold_intent) == loaded.operation_digest
