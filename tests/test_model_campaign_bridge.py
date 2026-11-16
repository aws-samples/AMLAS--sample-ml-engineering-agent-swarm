"""Private normal first-model bridge; neutral source ports, no native authority."""

from __future__ import annotations

import importlib

import pytest


def module():
    try:
        return importlib.import_module("amlas.control_plane.model_campaign_binding")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.model_campaign_binding":
            pytest.fail("missing-symbol: model_campaign_binding", pytrace=False)
        raise


def test_private_source_model_campaign_bridge_exists():
    m = module()
    for name in (
        "ModelPlanningBody",
        "ModelCampaignServices",
        "ModelCampaignModelBinding",
        "ModelSubmissionResult",
        "activate_model_campaign_services",
        "current_model_campaign_services",
    ):
        assert hasattr(m, name), "missing-symbol: " + name


def test_journal_inspects_qualified_protected_first_without_publishing_or_reserving():
    from test_model_campaign_journal import fixture

    parts = fixture()
    _m, j, intent, p, t, _a, current, *_ = parts
    assert callable(getattr(j, "inspect_original", None)), "missing-symbol: inspect_original"
    with j.qualified_io(current):
        inspected = j.inspect_original(intent)
        assert inspected.disposition == "ABSENT" and inspected.reservation is None
        assert p.writes == 0 and not t.calls
        j.publish_original(intent)
        partial = j.inspect_original(intent)
        assert partial.disposition == "PRESENT" and partial.reservation is None
        assert p.writes == 1 and not [v for v in t.calls if v[0] != "GetItem"]


@pytest.mark.parametrize("lost", [False, True])
def test_journal_reserve_outcome_distinguishes_acknowledged_first_existing_and_lost(lost):
    from test_model_campaign_journal import fixture

    parts = fixture()
    _m, j, intent, _p, t, _a, current, *_ = parts
    assert callable(getattr(j, "reserve_for_submission", None)), "missing-symbol: reserve_for_submission"
    with j.qualified_io(current):
        publication = j.publish_original(intent)
        t.lose_put = lost
        result = j.reserve_for_submission(intent, publication=publication)
        assert result.disposition == ("UNKNOWN" if lost else "NEW_CONFIRMED")
        second = j.reserve_for_submission(intent, publication=publication)
        assert second.disposition == "EXISTING" and second.reservation == result.reservation
        assert len([v for v in t.calls if v[0] == "PutItem"]) == 1


def bridge_fixture(monkeypatch):
    import copy
    import hashlib
    import json
    from types import SimpleNamespace

    from botocore.session import Session
    from botocore.validate import validate_parameters
    from test_campaign_coordinator import PROMPT, TABLE
    from test_campaign_runtime_entry import fixture_factory
    from test_model_campaign_journal import Protected, Table

    from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
    from amlas.control_plane import model_campaign as journal_source
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
    from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest

    m = module()
    values = fixture_factory()
    cr, _, _, runtime, guard, _, clock, *_ = values
    scope = runtime.coordinator._scope(guard.verify())
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    spec = {
        "project_id": "neutral_coding",
        "task_type": "text_generation",
        "target": "response",
        "primary_metric": "resolved_rate",
        "golden_dataset": "s3://fixture-data/cases.json",
        "baseline": {"type": "unmeasured"},
        "acceptance": {"primary_metric_delta_vs_baseline": ">= 0"},
        "generation": {
            "kind": "coding_sft",
            "model": {"repo_id": expected["repo_id"], "revision": expected["revision"]},
        },
    }
    raw = ("---\n" + json.dumps(spec, sort_keys=True) + "\n---\nNeutral original coding plan.").encode()
    body = m.ModelPlanningBody("spec_bridge", hashlib.sha256(raw).hexdigest(), raw)

    class Planning:
        qualified = True
        current = body

        def load_original_refs(self, **kwargs):
            return (cr.CampaignPlanningRef("spec", body.reference, body.sha256),)

        def load_original(self, entry, *, qualify_io):
            qualify_io()
            answer = self.current
            qualify_io()
            return answer

        def require_current(self, body, entry, *, qualify_io):
            qualify_io()
            if not self.qualified:
                raise ValueError("neutral planning origin revoked")
            qualify_io()
            return {"neutral_original": True}

    planning = Planning()
    runtime._planning = planning
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=None, interactive=False)
    entry = runtime.begin_prompt(PROMPT, policy=policy)
    cfg = journal_source.ModelAcquisitionConfig(
        scope["account_id"],
        scope["region"],
        "fixture-model-bucket",
        "arn:aws:iam::" + scope["account_id"] + ":role/fixture-model",
        scope["account_id"] + ".dkr.ecr." + scope["region"] + ".amazonaws.com/model@sha256:" + "b" * 64,
        "c" * 40,
        "b" * 64,
        scope["deployment_id"],
        scope["deployment_prefix"],
    )
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", cfg.deployment_id)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", cfg.deployment_prefix)
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    monkeypatch.delenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", raising=False)
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *a, **kw: "")
    table = Table()
    protected = Protected(journal_source)
    active = {"current": True}

    class Native:
        def __init__(self):
            self.requests = []
            self.ambiguous = False
            self.state = "InProgress"
            self.read_failure = False

        meta = SimpleNamespace(region_name=cfg.region)

        def create_processing_job(self, **kwargs):
            validate_parameters(
                kwargs,
                Session().get_service_model("sagemaker").operation_model("CreateProcessingJob").input_shape,
            )
            assert hint.calls and any(
                json.loads(row["private_record"]["B"])["state"] == "CREATE_ATTEMPTED"
                for row in table.items.values()
            )
            self.requests.append(copy.deepcopy(kwargs))
            if self.ambiguous:
                raise TimeoutError("neutral lost native response")
            return {
                "ProcessingJobArn": "arn:aws:sagemaker:"
                + cfg.region
                + ":"
                + cfg.account_id
                + ":processing-job/"
                + kwargs["ProcessingJobName"]
            }

        def describe_processing_job(self, **kwargs):
            validate_parameters(
                kwargs,
                Session().get_service_model("sagemaker").operation_model("DescribeProcessingJob").input_shape,
            )
            if self.read_failure or not self.requests:
                raise TimeoutError("neutral original native read unresolved")
            request = self.requests[-1]
            assert kwargs["ProcessingJobName"] == request["ProcessingJobName"]
            return {
                **copy.deepcopy(request),
                "ProcessingJobArn": "arn:aws:sagemaker:"
                + cfg.region
                + ":"
                + cfg.account_id
                + ":processing-job/"
                + request["ProcessingJobName"],
                "ProcessingJobStatus": self.state,
            }

    class Destination:
        meta = SimpleNamespace(region_name=cfg.region)
        after_read = None

        def get_bucket_location(self, **kwargs):
            validate_parameters(
                kwargs, Session().get_service_model("s3").operation_model("GetBucketLocation").input_shape
            )
            assert kwargs == {"Bucket": cfg.bucket, "ExpectedBucketOwner": cfg.account_id}
            if self.after_read:
                self.after_read()
            return {"LocationConstraint": cfg.region}

        def get_bucket_versioning(self, **kwargs):
            validate_parameters(
                kwargs, Session().get_service_model("s3").operation_model("GetBucketVersioning").input_shape
            )
            return {"Status": "Enabled"}

    native = Native()
    destination = Destination()
    adapter = ProcessingModelWeightAcquisitionBackend(
        region=cfg.region,
        role_arn=cfg.execution_role_arn,
        image_uri=cfg.image_uri,
        session_bucket=cfg.bucket,
        source_commit=cfg.source_commit,
        admission_digest=cfg.admission_digest,
        sagemaker_client=native,
        s3_client=destination,
    )

    class Config:
        current = cfg

        def read_current(self, entry, adapter, *, qualify_io):
            qualify_io()
            answer = self.current
            qualify_io()
            return answer

    config = Config()

    class Hint:
        def __init__(self):
            self.calls = []
            self.fail = False
            self.after_publish = None

        def publish_original_wake_binding(self, intent, reservation, entry, *, qualify_io):
            from amlas.async_callback.lambda_handler import encode_wake_job_binding

            qualify_io()
            data = json.loads(intent.canonical_bytes)
            config = data["config"]
            raw = encode_wake_job_binding(
                {
                    "schema_version": 1,
                    "record_kind": "campaign_wake_job_v1",
                    "account_id": config["account_id"],
                    "region": config["region"],
                    "deployment_id": config["deployment_id"],
                    "kind": "Processing",
                    "job_name": reservation.job_name,
                    "job_arn": reservation.job_arn,
                    "campaign_ref": entry.campaign_ref,
                    "operation_digest": data["operation_digest"],
                    "transport_session_id": entry.transport_session_id,
                }
            )
            self.calls.append(raw)
            if self.after_publish:
                self.after_publish()
            qualify_io()
            if self.fail:
                raise TimeoutError("neutral hint ack unresolved")
            return raw

        def require_current(self, *args, qualify_io):
            qualify_io()
            qualify_io()
            return {"neutral_immutable_original": True}

    hint = Hint()

    def boundary(*args):
        if not active["current"] or not planning.qualified or not protected.qualified:
            raise ValueError("neutral current boundary revoked")

    def planning_origin(port, body, entry, evidence):
        assert port is planning and planning.qualified and evidence == {"neutral_original": True}

    def config_origin(port, selected, entry):
        assert port is config and selected == config.current

    def protected_origin(port, pub, original, binding, evidence):
        assert port is protected and protected.qualified and evidence["native_capability"] is False

    def hint_origin(port, *args):
        assert port is hint and args[-1] == {"neutral_immutable_original": True}

    def journal_factory(entry, selected_adapter, config):
        return journal_source.DynamoModelCampaignJournal(
            entry,
            entry_qualifier=lambda e: e.require_current(),
            protected_original_port=protected,
            protected_origin_qualifier=protected_origin,
            coordinator=runtime.coordinator,
            table_name=TABLE,
            dynamodb_client=table,
            clock_ms=lambda: int(clock() * 1000),
            validate_prepared=selected_adapter.validate_prepared,
        )

    services = m.ModelCampaignServices(
        runtime_factory=runtime,
        planning_port=planning,
        planning_origin_qualifier=planning_origin,
        config_port=config,
        config_origin_qualifier=config_origin,
        wake_hint_port=hint,
        wake_hint_origin_qualifier=hint_origin,
        journal_factory=journal_factory,
        current_boundary_qualifier=boundary,
    )
    return SimpleNamespace(
        m=m,
        services=services,
        entry=entry,
        adapter=adapter,
        planning=planning,
        config=config,
        protected=protected,
        table=table,
        policy=policy,
        hint=hint,
        native=native,
        destination=destination,
        active=active,
        runtime=runtime,
    )


def test_actual_adapter_bridge_protects_original_hint_attempt_before_one_create_and_cold_reuse(monkeypatch):
    import json

    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    with x.m.activate_model_campaign_services(x.services):
        assert x.m.current_model_campaign_services() is x.services
    assert x.m.current_model_campaign_services() is None
    result = binding.submit_original("qwen3_5_9b", "1", x.policy)
    assert result.disposition == "ACCEPTED" and result.executed and result.native_admission is False
    assert result.handle.extra["bindings"]["runtime_session_id"] == x.entry.private_store_correlation
    assert (
        result.handle.extra["acquired"] is False and len(x.native.requests) == 1 and x.protected.writes == 1
    )
    hint = json.loads(x.hint.calls[0])
    assert hint["transport_session_id"] == x.entry.transport_session_id
    cold = x.services.build_for_entry(x.entry, x.adapter)
    same = cold.submit_original("qwen3_5_9b", "1", x.policy)
    assert (same.reference, same.handle.id, same.handle.arn) == (
        result.reference,
        result.handle.id,
        result.handle.arn,
    )
    assert not same.executed and len(x.native.requests) == 1


@pytest.mark.parametrize("failure", ["hint", "marker", "native"])
def test_unknown_original_acknowledgements_never_create_replacement(monkeypatch, failure):
    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    if failure == "hint":
        x.hint.fail = True
    elif failure == "marker":
        x.hint.after_publish = lambda: setattr(x.table, "lose_update", True)
    else:
        x.native.ambiguous = True
    if failure == "hint":
        with pytest.raises(ValueError):
            binding.submit_original("qwen3_5_9b", "1", x.policy)
    else:
        first = binding.submit_original("qwen3_5_9b", "1", x.policy)
        assert first.disposition == "UNKNOWN"
        if failure == "marker":
            assert not first.executed
    # Independent original readbacks never authorize recreation.
    x.hint.fail = False
    x.native.read_failure = True
    second = x.services.build_for_entry(x.entry, x.adapter).submit_original("qwen3_5_9b", "1", x.policy)
    assert second.disposition == "UNKNOWN" and not second.executed
    assert len(x.native.requests) == (1 if failure == "native" else 0)


def test_actual_effect_rechecks_policy_after_hint_without_persisting_policy(monkeypatch):
    import json

    from amlas.control_plane.approvals import ApprovalMode

    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    x.hint.after_publish = lambda: setattr(x.policy, "mode", ApprovalMode.ASK)
    with pytest.raises(ValueError):
        binding.submit_original("qwen3_5_9b", "1", x.policy)
    assert not x.native.requests
    for row in x.table.items.values():
        assert "policy" not in json.loads(row["private_record"]["B"])


def test_actual_destination_callback_revocation_refuses_before_create(monkeypatch):
    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    x.destination.after_read = lambda: x.active.update(current=False)
    with pytest.raises(ValueError):
        binding.submit_original("qwen3_5_9b", "1", x.policy)
    assert not x.native.requests


def test_changed_original_spec_or_config_and_foreign_factory_refuse_before_effect(monkeypatch):
    import dataclasses

    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    x.config.current = dataclasses.replace(
        x.config.current, execution_role_arn=x.config.current.execution_role_arn + "-other"
    )
    with pytest.raises(ValueError):
        binding.submit_original("qwen3_5_9b", "1", x.policy)
    assert not x.native.requests and x.protected.writes == 0
    x.services._runtime = object()
    with pytest.raises(ValueError):
        x.services.build_for_entry(x.entry, x.adapter)


def test_lost_reservation_acknowledgement_stays_read_only_no_attempt_or_create(monkeypatch):
    import json

    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    x.table.lose_put = True
    result = binding.submit_original("qwen3_5_9b", "1", x.policy)
    assert result.disposition == "UNKNOWN" and not result.executed and not x.native.requests
    assert all(
        json.loads(row["private_record"]["B"])["state"] == "RESERVED" for row in x.table.items.values()
    )
    repeated = x.services.build_for_entry(x.entry, x.adapter).submit_original("qwen3_5_9b", "1", x.policy)
    assert repeated.reference == result.reference and not repeated.executed and not x.native.requests


def test_concurrent_first_bridge_keeps_one_original_ref_and_at_most_one_create(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    x = bridge_fixture(monkeypatch)
    bindings = [
        x.services.build_for_entry(x.entry, x.adapter),
        x.services.build_for_entry(x.entry, x.adapter),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(b.submit_original, "qwen3_5_9b", "1", x.policy) for b in bindings]
        results = [v.result() for v in futures]
    assert len({v.reference for v in results}) == 1 and len({v.handle.id for v in results}) == 1
    assert len(x.native.requests) == 1 and sum(v.executed for v in results) == 1


def test_original_body_changed_or_origin_revoked_before_build_has_zero_effects(monkeypatch):
    import dataclasses
    import hashlib

    x = bridge_fixture(monkeypatch)
    changed = x.planning.current.body + b" Ordinary changed plan."
    x.planning.current = dataclasses.replace(
        x.planning.current, body=changed, sha256=hashlib.sha256(changed).hexdigest()
    )
    with pytest.raises(ValueError):
        x.services.build_for_entry(x.entry, x.adapter)
    assert x.protected.writes == 0 and not x.native.requests


def test_bridge_result_cold_installs_same_ref_into_real_private_store_only(monkeypatch):
    from amlas.control_plane.campaign_state import CampaignStateCatalog, _mint_binding
    from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore

    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    first = binding.submit_original("qwen3_5_9b", "1", x.policy)
    restored = x.services.build_for_entry(x.entry, x.adapter).submit_original("qwen3_5_9b", "1", x.policy)
    cfg = x.config.current
    scope = CampaignStateCatalog(
        cfg.account_id,
        cfg.region,
        cfg.deployment_id,
        cfg.deployment_prefix,
        cfg.deployment_prefix + "-training-jobs",
        cfg.execution_role_arn,
        "AROA" + "A" * 16,
    )
    trusted = _mint_binding(scope, x.entry.campaign_id, x.entry.private_store_correlation)
    fresh = SessionModelAcquisitionStore(runtime_session_id=x.entry.private_store_correlation)
    fresh._restore_original_producer(restored.reference, x.adapter, restored.handle, trusted)
    assert fresh.existing("qwen3_5_9b", "1") == first.reference
    assert fresh._original_producer_handle(first.reference, trusted).id == first.handle.id
    transport = SessionModelAcquisitionStore(runtime_session_id=x.entry.transport_session_id)
    with pytest.raises(ValueError):
        transport._restore_original_producer(restored.reference, x.adapter, restored.handle, trusted)
    assert len(x.native.requests) == 1 and restored.reference == first.reference


def test_read_only_binding_observation_uses_original_retained_handle_and_current_sdk_scope(monkeypatch):
    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    assert callable(getattr(binding, "observe_original", None)), "missing-symbol: observe_original"
    first = binding.submit_original("qwen3_5_9b", "1", x.policy)
    before = len(x.native.requests)
    result = binding.observe_original(first.reference)
    assert result["request_ref"] == first.reference and result["acquired"] is False
    assert len(x.native.requests) == before
    with pytest.raises(ValueError):
        binding.observe_original("model_acq_" + "f" * 32)
    x.active["current"] = False
    with pytest.raises(ValueError):
        binding.observe_original(first.reference)
    assert len(x.native.requests) == before
