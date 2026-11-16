"""Normal private campaign entry; neutral exact stores, no native authority proof."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import importlib
import os
import threading

import pytest
from test_campaign_coordinator import (
    DDB,
    PROMPT,
    TABLE,
    TRANSPORT,
    NativeJobs,
    coordinator_module,
    original,
    wake,
)
from test_campaign_initialization import fixture_initializer
from test_runtime_prompt_entry import collect, inert_runtime_setup  # noqa: F401 - imported autouse fixture

from amlas import api
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat.models import ChatResponse


def module():
    try:
        return importlib.import_module("amlas.control_plane.campaign_runtime")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.campaign_runtime":
            pytest.fail("missing-symbol: campaign_runtime.CampaignRuntimeFactory", pytrace=False)
        raise


def policy():
    return ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST,
        allowlist=["acquire_model_weights", "submit_qwen_sft"],
        max_auto_cost=None,
        interactive=False,
    )


class Ingress:
    """Untrusted neutral private observation, never native IAM attestation."""

    def __init__(self, cr, requirement, clock):
        self.cr, self.requirement, self.clock = cr, requirement, clock
        self.mode, self.calls, self.bad = "new_prompt", [], None

    def read_current(self, *, request_sha256, purpose):
        self.calls.append((request_sha256, purpose))
        if self.bad == "missing":
            return None
        now = self.clock()
        result = self.cr.CampaignIngressObservation(
            admission_manifest_sha256=self.requirement.admission_manifest_sha256,
            ingress_boundary_sha256=self.requirement.ingress_boundary_sha256,
            mode=self.mode,
            transport_session_id=TRANSPORT,
            request_sha256=request_sha256,
            observed_at=now,
            expires_at=now + 10,
            process_id=os.getpid(),
            reader_binding=self,
        )
        if self.bad == "expired":
            return dataclasses.replace(result, expires_at=now)
        if self.bad == "digest":
            return dataclasses.replace(
                result, ingress_boundary_sha256=hashlib.sha256(b"different fixture boundary").hexdigest()
            )
        if self.bad == "issuer":
            return dataclasses.replace(result, reader_binding=object())
        return result


class Terminal:
    def __init__(self, cr):
        self.cr, self.calls, self.verified = cr, [], True

    def verify_terminal_original(self, operation, *, policy=None):
        self.calls.append((operation, policy))
        return self.cr.CampaignTerminalVerification(
            operation_digest=operation.operation_digest,
            job_arn=operation.job_arn,
            request_sha256=operation.request_sha256,
            source_binding_sha256=operation.source_binding_sha256,
            status="Completed",
            artifact_sha256=hashlib.sha256(b"verified neutral original artifacts").hexdigest(),
            verified=self.verified,
        )


def fixture_factory():
    cr = module()
    _, ca, initializer, guard, native, clock, tx, _, _ = fixture_initializer()
    cc = coordinator_module()
    ddb = DDB()
    ddb.items, ddb.lock = tx.items, tx.lock
    coordinator = cc.DynamoCampaignCoordinator(guard, ddb, table_name=TABLE, clock=clock)
    requirement = cr.CampaignIngressRequirement(
        admission_manifest_sha256=guard.approved.manifest_sha256,
        ingress_boundary_sha256=hashlib.sha256(b"fixture independently approved ingress").hexdigest(),
        approval_reference="fixture-ingress-review",
    )
    ingress = Ingress(cr, requirement, clock)
    jobs, terminal = NativeJobs(cc), Terminal(cr)
    factory = cr.CampaignRuntimeFactory(
        guard=guard,
        initializer=initializer,
        coordinator=coordinator,
        approved_ingress=cr.approve_campaign_ingress(requirement),
        ingress_reader=ingress,
        private_observer=jobs,
        terminal_verifier=terminal,
        clock=clock,
    )
    return cr, ca, cc, factory, guard, native, clock, tx, ddb, ingress, jobs, terminal


def registered_wake(values):
    _cr, _, cc, factory, _guard, _, _, _, _, ingress, _, _ = values
    context = factory.begin_prompt(PROMPT, policy=policy())
    record = context.load_original_campaign()
    lease = context.claim_original_lease(expected_revision=record.revision)
    operation = original(cc, "Processing")
    factory.coordinator.register_original(context.require_current(), lease, operation)
    record = context.load_original_campaign()
    ingress.mode = "private_wake"
    return context, wake(cc, record, operation)


def engine_spy(monkeypatch, cr, seen, *, failure=False):
    from amlas.tui.chat import engine

    class NormalEngine:
        def __init__(self, *, settings=None, approval_policy=None):
            self.policy = approval_policy

        async def process_message(self, prompt):
            seen.append((prompt, cr.current_campaign_entry(), self.policy, threading.get_ident()))
            if failure:
                raise LookupError("neutral engine failure")
            return ChatResponse(message="ordinary engine response")

    monkeypatch.setattr(engine, "ChatEngine", NormalEngine)


def test_normal_api_uses_original_context_and_clears_it(monkeypatch):
    values = fixture_factory()
    cr, _, _, factory, _, _, _, _, _, _, _, _ = values
    seen, current_policy = [], policy()
    engine_spy(monkeypatch, cr, seen)
    with cr.activate_campaign_runtime(factory):
        result = asyncio.run(api.run_prompt(PROMPT, policy=current_policy))
    assert result.message == "ordinary engine response"
    assert seen[0][0] == PROMPT
    assert seen[0][1].literal_prompt == PROMPT
    assert seen[0][1].private_store_correlation
    assert seen[0][2] is current_policy
    assert not hasattr(seen[0][1], "policy") and not hasattr(seen[0][1], "witness")
    assert cr.current_campaign_entry() is None


def test_app_copies_private_factory_into_normal_api_worker(monkeypatch):
    cr, _, _, factory, _, _, _, _, _, _, _, _ = fixture_factory()
    seen = []
    engine_spy(monkeypatch, cr, seen)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["acquire_model_weights"]')
    with cr.activate_campaign_runtime(factory):
        events = collect({"prompt": PROMPT})
    assert events[-1]["status"] == "response_received"
    assert seen[0][1] is not None and seen[0][0] == PROMPT
    assert seen[0][3] != threading.get_ident()
    assert cr.current_campaign_entry() is None
    assert events[-1]["lifecycle_verified"] is False


def test_engine_failure_resets_original_context(monkeypatch):
    cr, _, _, factory, _, _, _, _, _, _, _, _ = fixture_factory()
    seen = []
    engine_spy(monkeypatch, cr, seen, failure=True)
    with cr.activate_campaign_runtime(factory), pytest.raises(LookupError):
        asyncio.run(api.run_prompt(PROMPT, policy=policy()))
    assert cr.current_campaign_entry() is None


@pytest.mark.parametrize("bad", ["missing", "expired", "digest", "issuer"])
def test_current_ingress_refusal_happens_before_durable_or_engine_io(monkeypatch, bad):
    cr, _, _, factory, _, _, _, tx, ddb, ingress, _, _ = fixture_factory()
    ingress.bad = bad
    seen = []
    engine_spy(monkeypatch, cr, seen)
    with cr.activate_campaign_runtime(factory), pytest.raises(cr.CampaignRuntimeError):
        asyncio.run(api.run_prompt(PROMPT, policy=policy()))
    assert tx.calls == [] and ddb.calls == [] and seen == []
    assert cr.current_campaign_entry() is None


def test_redelivery_resolves_original_without_new_publication():
    cr, _, _, factory, _, _, _, tx, _, ingress, _, _ = fixture_factory()
    first = factory.begin_prompt(PROMPT, policy=policy())
    writes = sum(name == "TransactWriteItems" for name, _ in tx.calls)
    ingress.mode = "redelivery"
    again = factory.begin_prompt(PROMPT, policy=policy())
    assert again.campaign_ref == first.campaign_ref
    assert again.private_store_correlation == first.private_store_correlation
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == writes
    assert cr.current_campaign_entry() is None


@pytest.mark.parametrize("state", ["UNKNOWN", "InProgress", "Stopping", "Failed", "Stopped"])
def test_private_nonready_wake_never_calls_api_or_agent(monkeypatch, state):
    values = fixture_factory()
    cr, _, _, factory, _, _, _, tx, _, _, jobs, terminal = values
    _, hint = registered_wake(values)
    jobs.state = state
    seen = []
    engine_spy(monkeypatch, cr, seen)
    writes = sum(name == "TransactWriteItems" for name, _ in tx.calls)
    with cr.activate_campaign_runtime(factory):
        events = collect({name: getattr(hint, name) for name in coordinator_module()._WAKE})
    assert events[-1]["status"] == "awaiting_original"
    assert events[-1]["job_arn"] == hint.job_arn
    assert seen == [] and terminal.calls == []
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == writes
    assert cr.current_campaign_entry() is None


def test_private_completed_verified_wake_continues_exact_original_prompt(monkeypatch):
    values = fixture_factory()
    cr, _, _, factory, _, _, _, _, _, _, _, terminal = values
    original_context, hint = registered_wake(values)
    seen = []
    engine_spy(monkeypatch, cr, seen)
    with cr.activate_campaign_runtime(factory):
        events = collect({name: getattr(hint, name) for name in coordinator_module()._WAKE})
    assert events[-1]["status"] == "response_received"
    assert seen[0][0] == PROMPT
    assert seen[0][1].campaign_ref == original_context.campaign_ref
    assert len(terminal.calls) == 1
    assert cr.current_campaign_entry() is None


def test_completed_unverified_artifacts_are_awaiting_not_agent_input(monkeypatch):
    values = fixture_factory()
    cr, _, _, factory, _, _, _, _, _, _, _, terminal = values
    _, hint = registered_wake(values)
    terminal.verified = False
    seen = []
    engine_spy(monkeypatch, cr, seen)
    with cr.activate_campaign_runtime(factory):
        events = collect({name: getattr(hint, name) for name in coordinator_module()._WAKE})
    assert events[-1]["status"] == "awaiting_original" and seen == []


def test_public_fields_cannot_attach_owner_or_policy(monkeypatch):
    cr, _, _, factory, _, _, _, tx, _, _, _, _ = fixture_factory()
    seen = []
    engine_spy(monkeypatch, cr, seen)
    with cr.activate_campaign_runtime(factory):
        events = collect({"prompt": PROMPT, "owner_namespace": "caller-authored"})
    assert events[-1]["error_code"] == "unsupported_prompt_fields"
    assert seen == [] and tx.calls == []


def test_unsealed_context_and_inherited_activation_refuse(monkeypatch):
    cr, _, _, factory, _, _, _, _, _, _, _, _ = fixture_factory()
    with pytest.raises(cr.CampaignRuntimeError), cr.activate_campaign_entry({"campaign_ref": "public"}):
        pytest.fail("public dictionary was activated")
    context = factory.begin_prompt(PROMPT, policy=policy())
    monkeypatch.setattr(cr.os, "getpid", lambda: context._pid + 1)
    with pytest.raises(cr.CampaignRuntimeError), cr.activate_campaign_entry(context):
        pytest.fail("inherited context was activated")


def test_untyped_ingress_digest_cannot_compare_equal_to_approved_evidence(monkeypatch):
    cr, _, _, factory, _, _, _, tx, ddb, ingress, _, _ = fixture_factory()
    read = ingress.read_current

    class EqualClaim:
        def __eq__(self, other):
            return True

    ingress.read_current = lambda **kwargs: dataclasses.replace(
        read(**kwargs), ingress_boundary_sha256=EqualClaim()
    )
    with pytest.raises(cr.CampaignRuntimeError):
        factory.begin_prompt(PROMPT, policy=policy())
    assert tx.calls == [] and ddb.calls == []


def test_untyped_terminal_pin_cannot_authorize_agent_continuation(monkeypatch):
    values = fixture_factory()
    cr, _, _, factory, _, _, _, _, _, _, _, terminal = values
    _, hint = registered_wake(values)
    verify = terminal.verify_terminal_original

    class EqualClaim:
        def __eq__(self, other):
            return True

    terminal.verify_terminal_original = lambda operation, **kwargs: dataclasses.replace(
        verify(operation, **kwargs), operation_digest=EqualClaim()
    )
    seen = []
    engine_spy(monkeypatch, cr, seen)
    with cr.activate_campaign_runtime(factory):
        events = collect({name: getattr(hint, name) for name in coordinator_module()._WAKE})
    assert seen == []
    assert events[-1].get("status") != "response_received"


def test_no_factory_keeps_normal_legacy_api_without_private_context(monkeypatch):
    cr = module()
    seen = []
    engine_spy(monkeypatch, cr, seen)
    response = asyncio.run(api.run_prompt(PROMPT))
    assert response.message == "ordinary engine response"
    assert seen[0][1] is None and seen[0][2] is None


def test_unapproved_factory_configuration_and_public_policy_refuse_before_io():
    cr, _, _, factory, _, _, _, tx, ddb, _, _, _ = fixture_factory()
    with pytest.raises(cr.CampaignRuntimeError):
        cr.ApprovedCampaignIngress(factory._approved.requirement)
    with pytest.raises(cr.CampaignRuntimeError):
        factory.begin_prompt(PROMPT, policy={"mode": "auto"})
    assert tx.calls == [] and ddb.calls == []


def test_foreign_wake_reference_and_unregistered_digest_never_call_engine(monkeypatch):
    values = fixture_factory()
    cr, _, _, factory, _, _, _, _, _, _, jobs, _ = values
    _, hint = registered_wake(values)
    seen = []
    engine_spy(monkeypatch, cr, seen)
    hints = [
        dataclasses.replace(hint, operation_digest=hashlib.sha256(b"unregistered fixture").hexdigest()),
        dataclasses.replace(hint, campaign_ref="campaign_" + "0" * 64 + "_" + "1" * 64),
    ]
    for changed in hints:
        with cr.activate_campaign_runtime(factory):
            events = collect({name: getattr(changed, name) for name in coordinator_module()._WAKE})
        assert events[-1]["error_code"] == "private_wake_refused"
    assert seen == [] and jobs.calls == []


def test_terminal_verifier_failure_holds_original_without_agent(monkeypatch):
    values = fixture_factory()
    cr, _, _, factory, _, _, _, _, _, _, _, terminal = values
    _, hint = registered_wake(values)

    def unavailable(*args, **kwargs):
        raise TimeoutError("neutral original artifact check unavailable")

    terminal.verify_terminal_original = unavailable
    seen = []
    engine_spy(monkeypatch, cr, seen)
    with cr.activate_campaign_runtime(factory):
        events = collect({name: getattr(hint, name) for name in coordinator_module()._WAKE})
    assert events[-1]["status"] == "awaiting_original" and seen == []


def test_cancellation_and_nested_scopes_reset_context(monkeypatch):
    from amlas.tui.chat import engine

    cr, _, _, factory, _, _, _, _, _, _, _, _ = fixture_factory()
    outer = factory.begin_prompt(PROMPT, policy=policy())

    class CancelledEngine:
        def __init__(self, **kwargs):
            pass

        async def process_message(self, prompt):
            assert cr.current_campaign_entry() is not None
            raise asyncio.CancelledError

    monkeypatch.setattr(engine, "ChatEngine", CancelledEngine)
    with cr.activate_campaign_runtime(factory):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(api.run_prompt(PROMPT, policy=policy()))
        assert cr.current_campaign_entry() is None
        with cr.activate_campaign_entry(outer):
            with cr.activate_campaign_entry(outer):
                assert cr.current_campaign_entry() is outer
            assert cr.current_campaign_entry() is outer
        assert cr.current_campaign_entry() is None


def test_native_revocation_refuses_before_initialization_or_engine(monkeypatch):
    cr, ca, _, factory, _, native, _, tx, ddb, _, _, _ = fixture_factory()
    native.observation = dataclasses.replace(native.observation, endpoint_live_version="2")
    seen = []
    engine_spy(monkeypatch, cr, seen)
    with cr.activate_campaign_runtime(factory), pytest.raises(ca.AdmissionError):
        asyncio.run(api.run_prompt(PROMPT, policy=policy()))
    assert tx.calls == [] and ddb.calls == [] and seen == []


def test_cold_private_wake_uses_fresh_objects_and_protected_original(monkeypatch):
    from test_campaign_initialization import ProtectionReader

    values = fixture_factory()
    cr, _, cc, factory, guard, _, clock, tx, ddb, _, _, _ = values
    previous, hint = registered_wake(values)
    clock.value += 61  # Original owner lease honestly expires, not waived.
    initializer = type(factory.initializer)(
        guard,
        factory.initializer._boundary,
        private_transaction_client=tx,
        boundary_reader=ProtectionReader(
            module_initialization(), factory.initializer._boundary.configuration, clock
        ),
        clock=clock,
    )
    coordinator = cc.DynamoCampaignCoordinator(guard, ddb, table_name=TABLE, clock=clock)
    ingress = Ingress(cr, factory._approved.requirement, clock)
    ingress.mode = "private_wake"
    cold = cr.CampaignRuntimeFactory(
        guard=guard,
        initializer=initializer,
        coordinator=coordinator,
        approved_ingress=factory._approved,
        ingress_reader=ingress,
        private_observer=NativeJobs(cc),
        terminal_verifier=Terminal(cr),
        clock=clock,
    )
    seen = []
    engine_spy(monkeypatch, cr, seen)
    writes = sum(name == "TransactWriteItems" for name, _ in tx.calls)
    with cr.activate_campaign_runtime(cold):
        events = collect({name: getattr(hint, name) for name in cc._WAKE})
    assert events[-1]["status"] == "response_received"
    assert seen[0][1].campaign_ref == previous.campaign_ref
    assert seen[0][1].private_store_correlation == previous.private_store_correlation
    assert seen[0][1]._factory is cold
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == writes


def module_initialization():
    return importlib.import_module("amlas.control_plane.campaign_initialization")


def test_planning_refs_are_data_and_current_context_denies_later_ingress_revocation():
    cr, _, _, factory, _, _, _, _, _, ingress, _, _ = fixture_factory()

    class Plans:
        def load_original_refs(self, **kwargs):
            return (
                cr.CampaignPlanningRef(
                    "spec",
                    "spec_" + "a" * 32,
                    hashlib.sha256(b"original validated planning spec").hexdigest(),
                ),
            )

    factory._planning = Plans()
    context = factory.begin_prompt(PROMPT, policy=policy())
    assert context.planning_refs[0].kind == "spec"
    ingress.bad = "missing"
    with pytest.raises(cr.CampaignRuntimeError):
        context.require_current()


@pytest.mark.parametrize("value", [True, float("inf"), float("nan")])
def test_private_clock_invalid_values_refuse_before_ingress_or_durable_io(value):
    cr, _, _, factory, _, _, _, tx, ddb, ingress, _, _ = fixture_factory()
    factory._clock = lambda: value
    with pytest.raises(cr.CampaignRuntimeError):
        factory.begin_prompt(PROMPT, policy=policy())
    assert ingress.calls == [] and tx.calls == [] and ddb.calls == []


def test_terminal_verifier_receives_fresh_policy_snapshot_outside_context():
    values = fixture_factory()
    _cr, _, _, factory, _, _, _, _, _, _, _, terminal = values
    _, hint = registered_wake(values)
    caller_policy = policy()
    outcome = factory.begin_wake(hint, policy=caller_policy)
    observed_policy = terminal.calls[-1][1]
    assert observed_policy is not caller_policy
    assert observed_policy.allowlist == caller_policy.allowlist
    observed_policy.allowlist.append("fixture_private_snapshot_only")
    assert "fixture_private_snapshot_only" not in caller_policy.allowlist
    assert not hasattr(outcome.context, "policy")


def all_writes(tx, ddb):
    return [
        entry for entry in tx.calls + ddb.calls if entry[0] in ("TransactWriteItems", "PutItem", "UpdateItem")
    ]


def test_ingress_revoked_during_initial_read_prevents_actual_publication():
    cr, _, _, factory, _, _, _, tx, ddb, ingress, _, _ = fixture_factory()
    read = tx.transact_get_items

    def revoke_after_read(**kwargs):
        result = read(**kwargs)
        ingress.bad = "missing"
        return result

    tx.transact_get_items = revoke_after_read
    with pytest.raises(ValueError):
        factory.begin_prompt(PROMPT, policy=policy())
    assert all_writes(tx, ddb) == []
    assert cr.current_campaign_entry() is None


def test_context_read_revocation_refuses_returned_campaign():
    _cr, _, _, factory, _, _, _, _, ddb, ingress, _, _ = fixture_factory()
    context = factory.begin_prompt(PROMPT, policy=policy())
    read = ddb.get_item

    def revoke_after_read(**kwargs):
        result = read(**kwargs)
        ingress.bad = "missing"
        return result

    ddb.get_item = revoke_after_read
    with pytest.raises(ValueError):
        context.load_original_campaign()


def test_observer_revocation_prevents_observation_cas():
    values = fixture_factory()
    _cr, _, _, factory, _, _, _, tx, ddb, ingress, jobs, _ = values
    _, hint = registered_wake(values)
    observe = jobs.observe_original

    def revoke_after_observation(operation):
        result = observe(operation)
        ingress.bad = "missing"
        return result

    jobs.observe_original = revoke_after_observation
    before = len(all_writes(tx, ddb))
    with pytest.raises(ValueError):
        factory.begin_wake(hint, policy=policy())
    # One new claim may occur; observer's revocation must prevent observation CAS.
    assert len(all_writes(tx, ddb)) - before == 1


@pytest.mark.parametrize(
    "first,second,expected",
    [("Completed", "UNKNOWN", "awaiting_original"), ("UNKNOWN", "Completed", "continue_original")],
)
def test_duplicate_wake_readiness_is_fresh_not_stored_status(first, second, expected):
    values = fixture_factory()
    _cr, _, _, factory, _, _, _, _, _, _, jobs, _ = values
    _, hint = registered_wake(values)
    jobs.state = first
    factory.begin_wake(hint, policy=policy())
    calls = len(jobs.calls)
    jobs.state = second
    result = factory.begin_wake(hint, policy=policy())
    assert len(jobs.calls) == calls + 1
    assert result.disposition == expected


def test_factory_refuses_coordinator_on_another_unqualified_table():
    cr, _, cc, factory, guard, _, clock, tx, ddb, _, _, _ = fixture_factory()
    different = cc.DynamoCampaignCoordinator(guard, ddb, table_name="fixture-other-table", clock=clock)
    with pytest.raises(cr.CampaignRuntimeError):
        cr.CampaignRuntimeFactory(
            guard=guard,
            initializer=factory.initializer,
            coordinator=different,
            approved_ingress=factory._approved,
            ingress_reader=factory._ingress,
            private_observer=factory._observer,
            clock=clock,
        )
    assert all_writes(tx, ddb) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_id", "000000000000"),
        ("region", "us-east-1"),
        ("deployment_id", "12345678-1234-4234-9234-00000000ee91"),
        ("kind", "Training"),
        ("job_arn", "wrong"),
    ],
)
def test_full_wake_mismatch_refuses_before_any_lease_write(field, value):
    values = fixture_factory()
    _cr, _, _, factory, _, _, _, tx, ddb, _, jobs, _ = values
    _, hint = registered_wake(values)
    changed = dataclasses.replace(hint, **{field: hint.job_arn + "-wrong" if field == "job_arn" else value})
    before_items, before_writes = copy.deepcopy(ddb.items), len(all_writes(tx, ddb))
    with pytest.raises(ValueError):
        factory.begin_wake(changed, policy=policy())
    assert ddb.items == before_items
    assert len(all_writes(tx, ddb)) == before_writes
    assert jobs.calls == []


def test_wake_generator_never_activates_entry_in_consumer_task(monkeypatch):
    from amlas import app as runtime

    values = fixture_factory()
    cr, _, _, factory, _, _, _, _, _, _, _, _ = values
    _, hint = registered_wake(values)
    engine_spy(monkeypatch, cr, [])

    async def read_first():
        with cr.activate_campaign_runtime(factory):
            stream = runtime.agent_invocation(
                {name: getattr(hint, name) for name in coordinator_module()._WAKE}
            )
            try:
                await anext(stream)
                assert cr.current_campaign_entry() is None
            finally:
                await stream.aclose()
            assert cr.current_campaign_entry() is None

    asyncio.run(read_first())


def test_conflicting_factory_with_existing_entry_refuses_before_engine(monkeypatch):
    cr, _, _, factory_a, _, _, _, _, _, _, _, _ = fixture_factory()
    _, _, _, factory_b, _, _, _, _, _, _, _, _ = fixture_factory()
    context = factory_a.begin_prompt(PROMPT, policy=policy())
    seen = []
    engine_spy(monkeypatch, cr, seen)
    with (
        cr.activate_campaign_entry(context),
        cr.activate_campaign_runtime(factory_b),
        pytest.raises(cr.CampaignRuntimeError),
    ):
        asyncio.run(api.run_prompt(PROMPT, policy=policy()))
    assert seen == []


@pytest.mark.parametrize("point", ["construction", "response"])
def test_current_entry_revocation_prevents_model_dispatch_or_response(monkeypatch, point):
    from amlas.tui.chat import engine

    cr, _, _, factory, _, _, _, _, _, ingress, _, _ = fixture_factory()
    seen = []

    class Engine:
        def __init__(self, **kwargs):
            if point == "construction":
                ingress.bad = "missing"

        async def process_message(self, prompt):
            seen.append(prompt)
            ingress.bad = "missing"
            return ChatResponse(message="not currently qualified")

    monkeypatch.setattr(engine, "ChatEngine", Engine)
    with cr.activate_campaign_runtime(factory), pytest.raises(cr.CampaignRuntimeError):
        asyncio.run(api.run_prompt(PROMPT, policy=policy()))
    if point == "construction":
        assert seen == []


def test_legacy_spec_precedes_unrelated_campaign_ref_without_factory(monkeypatch):
    from amlas import spec

    seen = []

    def load(value):
        seen.append(value)
        raise LookupError("legacy loader reached")

    monkeypatch.setattr(spec, "load_spec_from_text", load)
    with pytest.raises(LookupError):
        collect({"spec_text": "existing spec", "campaign_ref": "ignored legacy extra"})
    assert seen == ["existing spec"]


def test_cold_wake_loads_same_runtime_policy_overlay_before_verification(monkeypatch):
    from amlas.config import env_loader

    values = fixture_factory()
    cr, _, _, factory, _, _, _, _, _, _, _, terminal = values
    _, hint = registered_wake(values)
    monkeypatch.delenv("AMLAS_RUNTIME_APPROVED_ACTIONS", raising=False)
    monkeypatch.setattr(
        env_loader,
        "load_layered_env",
        lambda *_: monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["acquire_model_weights"]'),
    )
    engine_spy(monkeypatch, cr, [])
    with cr.activate_campaign_runtime(factory):
        collect({name: getattr(hint, name) for name in coordinator_module()._WAKE})
    assert terminal.calls[-1][1] is not None
    assert terminal.calls[-1][1].allowlist == ["acquire_model_weights"]


def test_optional_io_callback_resets_and_legacy_none_preserves_behavior():
    _cr, _, _, factory, guard, _, _, tx, ddb, _, _, _ = fixture_factory()
    calls = []

    def deny():
        calls.append("denied")
        raise LookupError("neutral current boundary denied")

    with factory.initializer.qualified_io(deny), pytest.raises(LookupError):
        factory.initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert all_writes(tx, ddb) == [] and calls
    # Exact same object is usable only after the scoped denied callback resets.
    with factory.initializer.qualified_io(None):
        record = factory.initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    with factory.coordinator.qualified_io(deny), pytest.raises(LookupError):
        factory.coordinator.load_campaign(guard.verify(), record.campaign_ref)
    assert (
        factory.coordinator.load_campaign(guard.verify(), record.campaign_ref).campaign_ref
        == record.campaign_ref
    )


def test_context_primitive_qualifier_is_fresh_and_has_no_durable_io():
    _cr, _, _, factory, _, _, _, tx, ddb, ingress, _, _ = fixture_factory()
    context = factory.begin_prompt(PROMPT, policy=policy())
    before = (len(tx.calls), len(ddb.calls), len(ingress.calls))
    assert context.qualify_current_io() is None
    assert (len(tx.calls), len(ddb.calls)) == before[:2]
    assert len(ingress.calls) > before[2]


def test_stream_response_revocation_after_worker_serialization_refuses(monkeypatch):
    from amlas import app as runtime

    cr, _, _, factory, _, _, _, _, _, ingress, _, _ = fixture_factory()
    engine_spy(monkeypatch, cr, [])
    serialize = runtime._prompt_response

    def revoke_after_serialization(value):
        result = serialize(value)
        ingress.bad = "missing"
        return result

    monkeypatch.setattr(runtime, "_prompt_response", revoke_after_serialization)
    with cr.activate_campaign_runtime(factory):
        events = collect({"prompt": PROMPT})
    assert events[-1]["status"] == "failed"
    assert "response" not in events[-1]
    assert cr.current_campaign_entry() is None


def test_all_wake_stream_yields_leave_consumer_context_empty(monkeypatch):
    from amlas import app as runtime

    values = fixture_factory()
    cr, _, _, factory, _, _, _, _, _, _, _, _ = values
    _, hint = registered_wake(values)
    engine_spy(monkeypatch, cr, [])

    async def read():
        with cr.activate_campaign_runtime(factory):
            events = []
            async for event in runtime.agent_invocation(
                {name: getattr(hint, name) for name in coordinator_module()._WAKE}
            ):
                assert cr.current_campaign_entry() is None
                events.append(event)
            return events

    assert asyncio.run(read())[-1]["status"] == "response_received"


def test_current_primitive_qualification_refuses_changed_coordinator_table():
    cr, _, _, factory, _, _, _, tx, ddb, _, _, _ = fixture_factory()
    context = factory.begin_prompt(PROMPT, policy=policy())
    before = (len(tx.calls), len(ddb.calls))
    factory.coordinator._table = "fixture-other-table"
    with pytest.raises(cr.CampaignRuntimeError):
        context.qualify_current_io()
    assert (len(tx.calls), len(ddb.calls)) == before


def test_callback_delay_cannot_dispatch_expired_registration():
    from test_campaign_coordinator import open_and_claim

    cc, _, coordinator, _, _, clock, ddb, _, witness, _, lease = open_and_claim()
    before, calls = copy.deepcopy(ddb.items), [0]

    def delayed_current():
        calls[0] += 1
        if calls[0] == 3:  # Get pre/post, actual CAS pre.
            clock.value += 61

    with coordinator.qualified_io(delayed_current), pytest.raises(ValueError):
        coordinator.register_original(witness, lease, original(cc))
    assert ddb.items == before
    assert sum(name == "UpdateItem" for name, _ in ddb.calls) == 1  # only initial lease claim


def test_callback_delay_cannot_observe_under_expired_lease():
    from test_campaign_coordinator import open_and_claim

    cc, _, coordinator, guard, _, clock, _ddb, _, witness, record, lease = open_and_claim()
    operation = original(cc)
    coordinator.register_original(witness, lease, operation)
    observer, calls = NativeJobs(cc), [0]

    def delayed_current():
        calls[0] += 1
        if calls[0] == 3:  # Get pre/post, native observation pre.
            clock.value += 61

    with coordinator.qualified_io(delayed_current), pytest.raises(ValueError):
        coordinator.observe_wake(guard.verify(), lease, wake(cc, record, operation), observer)
    assert observer.calls == []


def test_preput_denial_retains_known_unattempted_creation_eligibility():
    from test_campaign_coordinator import fixture_coordinator

    _cc, _, coordinator, guard, _, _, ddb, _ = fixture_coordinator()
    calls = [0]

    def deny_preput():
        calls[0] += 1
        if calls[0] == 3:  # Get pre/post, Put pre.
            raise LookupError("neutral pre-Put qualification denied")

    with coordinator.qualified_io(deny_preput), pytest.raises(LookupError):
        coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    assert sum(name == "PutItem" for name, _ in ddb.calls) == 0
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    assert record.literal_prompt == PROMPT
    assert sum(name == "PutItem" for name, _ in ddb.calls) == 1


def test_table_changed_during_current_ingress_read_refuses_before_io():
    cr, _, _, factory, _, _, _, tx, ddb, ingress, _, _ = fixture_factory()
    context = factory.begin_prompt(PROMPT, policy=policy())
    read = ingress.read_current
    before = (len(tx.calls), len(ddb.calls))

    def drift_during_read(**kwargs):
        evidence = read(**kwargs)
        factory.coordinator._table = "fixture-other-table"
        return evidence

    ingress.read_current = drift_during_read
    with pytest.raises(cr.CampaignRuntimeError):
        context.qualify_current_io()
    assert (len(tx.calls), len(ddb.calls)) == before
