"""Private durable SFT lifecycle, strict offline SDK fakes; no native admission."""

from __future__ import annotations

import copy
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_campaign_state import CAMPAIGN, DDB, STS, state
from test_qwen_sft_backend import SESSION, backend, catalog
from test_qwen_sft_binding import real_store

from amlas.backends.adapters.campaign_state import DynamoCampaignStateBackend
from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.control_plane.campaign_state import CampaignStateError, build_sft_intent, operation_digest
from amlas.control_plane.sft_state import SessionSftStore


class FaultDDB(DDB):
    def __init__(self, phase=None, after=False):
        super().__init__()
        self.phase, self.after = phase, after
        self.timeline = []

    def _call(self, phase, effect):
        self.timeline.append(phase)
        if self.phase == phase and not self.after:
            raise TimeoutError("bounded synthetic acknowledgement fault")
        answer = effect()
        if self.phase == phase:
            raise TimeoutError("bounded synthetic acknowledgement fault")
        return answer

    def put_item(self, **kwargs):
        return self._call("reserve", lambda: super(FaultDDB, self).put_item(**kwargs))

    def update_item(self, **kwargs):
        values = kwargs["ExpressionAttributeValues"]
        old, new = values[":old"]["S"], values[":new"]["S"]
        phase = "marker" if old == "RESERVED" else "snapshot" if new == "CREATE_ATTEMPTED" else "outcome"
        return self._call(phase, lambda: super(FaultDDB, self).update_item(**kwargs))


def durable(*, phase=None, after=False, ddb=None):
    adapter, native, s3, inputs = backend()
    table, sts = ddb or FaultDDB(phase, after), STS()
    campaign = DynamoCampaignStateBackend(
        state(), dynamodb_client=table, sts_client=sts, sagemaker_client=native
    )
    binding = campaign.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=SESSION)
    return adapter, native, s3, inputs, campaign, binding, table


def submit(parts):
    adapter, _native, _s3, inputs, campaign, binding, _table = parts
    return adapter.submit(
        inputs=inputs,
        catalog=catalog(),
        recipe_id="fit_v1",
        runtime_session_id=SESSION,
        _campaign_state=campaign,
        _execution_binding=binding,
    )


def test_original_fixed_name_is_reserved_and_snapshot_acknowledged_before_one_create():
    parts = durable()
    h = submit(parts)
    _a, n, _s, i, c, b, d = parts
    r = c.load_original(b, h.id)
    assert n.calls[0]["TrainingJobName"] == r.native_job_name == h.id
    assert r.disposition == "ACCEPTED" and r.revision == 3
    assert r.operation_digest == operation_digest(build_sft_intent(b, catalog(), i))
    assert r.submission_snapshot.run_ref == h.extra["run_ref"]
    assert dict(r.submission_snapshot.staged_refs) == h.extra["staged_refs"]
    assert json.loads(r.submission_snapshot.request_json) == n.calls[0]
    assert r.submission_snapshot.training_record_count == 80
    assert d.timeline == ["reserve", "marker", "snapshot", "outcome"] and len(n.calls) == 1


def test_concurrent_adapters_choose_one_staging_and_create_owner():
    parts = durable()
    a, n, s, i, c, b, _d = parts
    other = SageMakerQwenSftBackend(sagemaker_client=n, s3_client=s)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                x.submit,
                inputs=i,
                catalog=catalog(),
                recipe_id="fit_v1",
                runtime_session_id=SESSION,
                _campaign_state=c,
                _execution_binding=b,
            )
            for x in (a, other)
        ]
        handles = [f.result() for f in futures]
    assert handles[0].id == handles[1].id and len(n.calls) == 1
    assert len([x for x in s.mutations if x == "put"]) == 1


@pytest.mark.parametrize("phase", ["reserve", "marker", "snapshot"])
@pytest.mark.parametrize("after", [False, True])
def test_uncertain_reservation_marker_or_snapshot_never_allows_create(phase, after):
    parts = durable(phase=phase, after=after)
    first = submit(parts)
    _a, n, s, _i, c, b, d = parts
    assert first.extra["submission_state"] != "accepted" and not n.calls
    if phase in ("reserve", "marker"):
        assert not s.mutations
    d.phase = None
    if after or phase != "reserve":
        before = list(s.mutations)
        again = submit(parts)
        assert again.id == first.id and not n.calls and s.mutations == before
        assert c.load_original(b, first.id).native_job_name == first.id


@pytest.mark.parametrize("after", [False, True])
def test_post_acceptance_persistence_failure_preserves_accepted_native_handle(after):
    parts = durable(phase="outcome", after=after)
    h = submit(parts)
    _a, n, _s, _i, c, b, d = parts
    assert len(n.calls) == 1 and h.extra["submission_state"] == "accepted"
    assert h.extra["durability_status"] == "outcome_uncertain"
    d.phase = None
    assert c.load_original(b, h.id).native_job_arn == h.arn
    assert submit(parts).id == h.id and len(n.calls) == 1


def test_snapshot_attaches_once_without_mutating_complete_intent_digest():
    parts = durable()
    h = submit(parts)
    _a, _n, _s, _i, c, b, _d = parts
    r = c.load_original(b, h.id)
    with pytest.raises(CampaignStateError):
        c.attach_submission_snapshot(b, r, r.submission_snapshot)
    assert c.load_original(b, h.id).operation_digest == r.operation_digest


def test_new_adapter_restore_revalidates_original_request_sources_and_exact_versions_without_writes():
    parts = durable()
    h = submit(parts)
    _a, n, s, i, c, b, d = parts
    r = c.load_original(b, h.id)
    before_s3, before_ddb = list(s.mutations), len(d.calls)
    new = SageMakerQwenSftBackend(sagemaker_client=n, s3_client=s)
    restored = new.restore_original(r, fresh_inputs=i)
    assert restored.id == h.id and restored.arn == h.arn
    assert restored.extra["submission_state"] == "accepted"
    assert new.artifacts(restored)["trained"] is False
    assert len(n.calls) == 1 and s.mutations == before_s3 and len(d.calls) == before_ddb


@pytest.mark.parametrize("field", ["request", "run", "staged", "count", "digest"])
def test_closed_snapshot_tamper_refuses_before_native_create_or_observation(field):
    parts = durable()
    h = submit(parts)
    a, n, _s, i, c, b, _d = parts
    r = c.load_original(b, h.id)
    snap = r.submission_snapshot
    if field == "request":
        value = json.loads(snap.request_json)
        value["RoleArn"] += "-foreign"
        snap = dataclasses.replace(
            snap, request_json=json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
    elif field == "run":
        snap = dataclasses.replace(snap, run_ref=dataclasses.replace(snap.run_ref, sha256="f" * 64))
    elif field == "staged":
        snap = dataclasses.replace(snap, staged_refs=(*snap.staged_refs, ("policy", {"mode": "auto"})))
    elif field == "count":
        snap = dataclasses.replace(snap, training_record_count=81)
    else:
        snap = dataclasses.replace(snap, operation_digest="f" * 64)
    bad = dataclasses.replace(r, submission_snapshot=snap)
    before = n.reads
    with pytest.raises((CampaignStateError, ValueError)):
        a.restore_original(bad, fresh_inputs=i)
    assert n.reads == before and len(n.calls) == 1


def test_native_complete_request_drift_remains_unresolved_without_resubmission():
    parts = durable()
    h = submit(parts)
    a, n, _s, i, c, b, _d = parts
    r = c.load_original(b, h.id)
    n.calls[0] = copy.deepcopy(n.calls[0])
    n.calls[0]["ResourceConfig"]["VolumeSizeInGB"] += 1
    restored = a.restore_original(r, fresh_inputs=i)
    assert restored.extra["submission_state"] == "ambiguous"
    assert a.artifacts(restored)["trained"] is False and len(n.calls) == 1


def test_changed_original_or_staged_version_is_refused_before_native_observation():
    parts = durable()
    h = submit(parts)
    a, n, s, i, c, b, _d = parts
    r = c.load_original(b, h.id)
    ref = r.submission_snapshot.run_ref
    key = ref.uri.split("/", 3)[3]
    s.objects[key] = (dataclasses.replace(ref, version_id="foreign-version"), s.objects[key][1])
    before = n.reads
    with pytest.raises((ValueError, AssertionError)):
        a.restore_original(r, fresh_inputs=i)
    assert n.reads == before and len(n.calls) == 1


def injected_store(monkeypatch):
    original, n, s3, model_ref, prep_ref, reads, _m, _p = real_store(monkeypatch)
    d, sts = FaultDDB(), STS()
    c = DynamoCampaignStateBackend(state(), dynamodb_client=d, sts_client=sts, sagemaker_client=n)
    b = c.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=SESSION)
    store = SessionSftStore(
        original._models,
        original._prepared,
        original._adapter,
        catalog(),
        campaign_state=c,
        execution_binding=b,
    )
    return store, n, s3, model_ref, prep_ref, reads, c, b, d


def test_fresh_store_resume_uses_original_private_accessors_and_retains_stable_opaque_ref(monkeypatch):
    store, n, s3, mr, pr, reads, c, b, d = injected_store(monkeypatch)
    first = store.submit(mr, pr, "fit_v1")
    reset = SessionSftStore(
        store._models, store._prepared, store._adapter, catalog(), campaign_state=c, execution_binding=b
    )
    before_s3, before_ddb = list(s3.mutations), len(d.calls)
    resumed = reset.resume_original(first["job_name"], mr, pr)
    assert resumed["request_ref"] == first["request_ref"] and resumed["job_name"] == first["job_name"]
    assert resumed["accepted"] is True and resumed["trained"] is False and len(n.calls) == 1
    assert reads == {"model": 2, "prep": 2}
    assert s3.mutations == before_s3
    assert all(name == "GetItem" for name, _params in d.calls[before_ddb:])


def test_missing_upstream_private_record_stays_honestly_unresolved(monkeypatch):
    store, n, _s3, mr, pr, _reads, _c, _b, _d = injected_store(monkeypatch)
    first = store.submit(mr, pr, "fit_v1")
    store._models._records.clear()
    resumed = store.resume_original(first["job_name"], mr, pr)
    assert resumed["status"] == "resume_unresolved" and resumed["accepted"] is False
    assert resumed["trained"] is False and len(n.calls) == 1


def test_private_injection_is_all_or_none_and_does_not_infer_worker_identity(monkeypatch):
    original, native, s3, _mr, _pr, reads, _m, _p = real_store(monkeypatch)
    _a, _n, _s, _i, c, b, _d = durable()
    for pair in ({"campaign_state": c}, {"execution_binding": b}):
        with pytest.raises(ValueError):
            SessionSftStore(original._models, original._prepared, original._adapter, catalog(), **pair)
    assert native.calls == [] and s3.mutations == [] and reads == {"model": 0, "prep": 0}
    assert b.scope.execution_role_arn != catalog().execution_role_arn


def test_legacy_uninjected_path_remains_conversation_local():
    a, n, _s, i = backend()
    first = a.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    second = a.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    assert first.id != second.id and len(n.calls) == 2
    assert "campaign_reservation" not in first.extra


def test_outcome_update_cannot_replace_an_attached_valid_snapshot():
    parts = durable()
    parts[1].error = TimeoutError("lost native acknowledgement")
    h = submit(parts)
    _a, _n, _s, _i, c, b, _d = parts
    r = c.load_original(b, h.id)
    run = dataclasses.replace(r.submission_snapshot.run_ref, version_id="other-immutable-version")
    snapshot = dataclasses.replace(
        r.submission_snapshot,
        run_ref=run,
        staged_refs=tuple(
            (name, run if name == "bindings/RUN.json" else ref)
            for name, ref in r.submission_snapshot.staged_refs
        ),
    )
    changed = dataclasses.replace(r, submission_snapshot=snapshot)
    with pytest.raises(CampaignStateError):
        c.record_submission_outcome(b, changed, disposition="AMBIGUOUS", error_code="TimeoutError")
    assert c.load_original(b, h.id).submission_snapshot == r.submission_snapshot


def test_resume_refuses_a_different_current_trainer_catalog(monkeypatch):
    store, n, _s3, mr, pr, reads, c, b, _d = injected_store(monkeypatch)
    first = store.submit(mr, pr, "fit_v1")
    other = SessionSftStore(
        store._models,
        store._prepared,
        store._adapter,
        catalog(source_commit="c" * 40),
        campaign_state=c,
        execution_binding=b,
    )
    before_reads, before_native = dict(reads), n.reads
    answer = other.resume_original(first["job_name"], mr, pr)
    assert answer["status"] == "resume_unresolved" and answer["accepted"] is False
    assert reads == before_reads and n.reads == before_native and len(n.calls) == 1


@pytest.mark.parametrize(
    "code,disposition",
    [
        ("ValidationException", "REJECTED"),
        ("AccessDeniedException", "REJECTED"),
        ("ResourceLimitExceeded", "REJECTED"),
        ("InternalFailure", "AMBIGUOUS"),
    ],
)
def test_native_submission_outcomes_are_retained_and_retries_never_create(code, disposition):
    from botocore.exceptions import ClientError

    parts = durable()
    parts[1].error = ClientError(
        {"Error": {"Code": code, "Message": "synthetic private error text"}}, "CreateTrainingJob"
    )
    h = submit(parts)
    _a, n, s3, _i, c, b, _d = parts
    r = c.load_original(b, h.id)
    assert r.disposition == disposition and r.submission_snapshot is not None
    before = list(s3.mutations)
    assert submit(parts).id == h.id and len(n.calls) == 1 and s3.mutations == before
    assert b"synthetic private error text" not in r._persisted_body


def test_version_one_descriptive_record_reads_and_upgrades_via_exact_body_cas():
    parts = durable()
    _a, n, s3, i, c, b, d = parts
    r = c.reserve_original_job(b, build_sft_intent(b, catalog(), i))
    row = d.items[r.native_job_name]
    payload = json.loads(row["private_record"]["B"])
    payload["schema_version"] = 1
    payload.pop("submission_snapshot")
    row["private_record"] = {"B": json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()}
    loaded = c.load_original(b, r.native_job_name)
    assert loaded.submission_snapshot is None
    next_record = c.mark_create_attempted(b, loaded)
    assert next_record.revision == 1 and json.loads(next_record._persisted_body)["schema_version"] == 2
    assert not n.calls and not s3.mutations


def test_stale_snapshot_marker_cannot_attach_twice():
    parts = durable(phase="outcome")
    h = submit(parts)
    _a, _n, _s, _i, c, b, d = parts
    r = c.load_original(b, h.id)
    assert r.disposition == "CREATE_ATTEMPTED" and r.submission_snapshot is not None
    with pytest.raises(CampaignStateError):
        c.attach_submission_snapshot(b, r, r.submission_snapshot)
    d.phase = None
    assert c.load_original(b, h.id).submission_snapshot == r.submission_snapshot


def test_durable_reload_retains_original_checkpoint_and_separate_fixed_identity():
    from test_qwen_sft_backend import install_output

    from amlas.control_plane.sft_state import VerifiedSftCheckpointSource

    parts = durable()
    original = submit(parts)
    a, n, s3, i, c, b, _d = parts
    install_output(a, n, s3, original)
    result = a.artifacts(original)
    assert result["trained"] is True and result["quality_success"] is False
    checkpoint = VerifiedSftCheckpointSource(
        SESSION,
        original.arn,
        "fit_v1",
        catalog().source_commit,
        result["checkpoint_ref"],
        tuple(sorted(result["checkpoint_refs"].items())),
    )
    h = a.reload(
        model=i.model,
        checkpoint=checkpoint,
        catalog=catalog(),
        runtime_session_id=SESSION,
        _campaign_state=c,
        _execution_binding=b,
        _durable_inputs=i,
    )
    r = c.load_original(b, h.id)
    assert h.id != original.id and len(n.calls) == 2 and r.disposition == "ACCEPTED"
    assert r.intent.operation == "reload" and r.submission_snapshot.training_record_count is None
    before = list(s3.mutations)
    assert (
        a.reload(
            model=i.model,
            checkpoint=checkpoint,
            catalog=catalog(),
            runtime_session_id=SESSION,
            _campaign_state=c,
            _execution_binding=b,
            _durable_inputs=i,
        ).id
        == h.id
    )
    assert len(n.calls) == 2 and s3.mutations == before


def test_completed_restored_fit_uses_existing_output_verifier_without_quality_promotion():
    from test_qwen_sft_backend import install_output

    parts = durable()
    h = submit(parts)
    a, n, s3, i, c, b, _d = parts
    install_output(a, n, s3, h)
    r = c.load_original(b, h.id)
    before = list(s3.mutations)
    restored = a.restore_original(r, fresh_inputs=i)
    answer = a.artifacts(restored)
    assert answer["trained"] is True and answer["status"] == "checkpoint_saved"
    assert answer["quality_success"] is False and answer["full_training_complete"] is False
    assert len(n.calls) == 1 and s3.mutations == before
