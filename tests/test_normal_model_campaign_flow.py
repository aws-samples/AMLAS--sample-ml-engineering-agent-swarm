"""Normal model tool routes through the original private campaign bridge."""

import threading

import pytest
from test_model_campaign_bridge import bridge_fixture

from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.flows import model_weight_acquisition_flow as flow


def bound_store(x):
    binding = x.services.build_for_entry(x.entry, x.adapter)
    return SessionModelAcquisitionStore(
        runtime_session_id=x.entry.private_store_correlation, campaign_binding=binding
    )


def test_normal_model_flow_uses_real_journal_bridge_and_original_reference(monkeypatch):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)
    monkeypatch.setattr(flow, "_resolve_backend", lambda: pytest.fail("campaign reached legacy resolver"))
    first = flow.run_model_weight_acquisition_flow(
        catalog_id="qwen3_5_9b", catalog_version="1", policy=x.policy, store=store
    )
    assert first["accepted"] and first["executed"] and len(x.native.requests) == 1
    expected = x.services.build_for_entry(x.entry, x.adapter).submit_original("qwen3_5_9b", "1", x.policy)
    assert first["request_ref"] == expected.reference
    assert store._records[first["request_ref"]].handle.id == expected.handle.id
    assert store.runtime_session_id != x.entry.transport_session_id


def test_fresh_normal_store_repeat_is_read_only_same_reference(monkeypatch):
    x = bridge_fixture(monkeypatch)
    monkeypatch.setattr(flow, "_resolve_backend", lambda: pytest.fail("campaign reached legacy resolver"))
    first = flow.run_model_weight_acquisition_flow(
        catalog_id="qwen3_5_9b", catalog_version="1", policy=x.policy, store=bound_store(x)
    )
    second = flow.run_model_weight_acquisition_flow(
        catalog_id="qwen3_5_9b", catalog_version="1", policy=x.policy, store=bound_store(x)
    )
    assert first["request_ref"] == second["request_ref"]
    assert first["job_name"] == second["job_name"]
    assert second["executed"] is False and second["reused"] is True
    assert len(x.native.requests) == 1


def test_overlapping_submission_is_answered_busy_without_a_second_create(monkeypatch):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)
    monkeypatch.setattr(flow, "_resolve_backend", lambda: pytest.fail("campaign reached legacy resolver"))
    planned = flow.run_model_weight_acquisition_flow(
        catalog_id="qwen3_5_9b", catalog_version="1", plan_only=True
    )
    create_taken, release = threading.Event(), threading.Event()
    creates = []
    bound_create = x.native.create_processing_job

    def held_create(**kwargs):
        # The stand-in has taken the request and holds back its answer, so the first
        # submission stays inside the store's submission lock until the test releases it.
        creates.append(kwargs["ProcessingJobName"])
        answer = bound_create(**kwargs)
        create_taken.set()
        assert release.wait(10), "held first submission was never released"
        return answer

    monkeypatch.setattr(x.native, "create_processing_job", held_create)
    results, failures = {}, []

    def submit(name):
        try:
            results[name] = flow.run_model_weight_acquisition_flow(
                catalog_id="qwen3_5_9b", catalog_version="1", policy=x.policy, store=store
            )
        except BaseException as error:
            failures.append((name, error))

    first = threading.Thread(target=submit, args=("first",))
    second = threading.Thread(target=submit, args=("second",))
    first.start()
    try:
        assert create_taken.wait(10), "first submission did not reach the native create"
        assert store._submission_lock.locked()
        second.start()
        second.join(10)
        assert not second.is_alive(), "second submission waited for the first instead of answering"
        assert not failures
        busy = results["second"]
        assert busy["status"] == "busy" and busy["accepted"] is False
        assert busy["decision"]["decision"] == "approved"
        # Apart from its status and the approval, the answer equals a plan preview:
        # no reference, no job and no effect.
        assert busy == {**planned, "status": "busy", "decision": busy["decision"]}
        assert len(creates) == 1 and len(x.native.requests) == 1
        assert first.is_alive() and "first" not in results
        assert store._submission_lock.locked()
    finally:
        release.set()
        first.join(10)
        if second.ident is not None:
            second.join(10)
    assert not first.is_alive() and not second.is_alive() and not failures
    done = results["first"]
    assert done["status"] == "submitted" and done["accepted"] is True and done["executed"] is True
    assert done["job_name"] == creates[0]
    assert store._records[done["request_ref"]].handle.id == done["job_name"]
    assert len(creates) == 1 and len(x.native.requests) == 1
    assert not store._submission_lock.locked()
    later = flow.run_model_weight_acquisition_flow(
        catalog_id="qwen3_5_9b", catalog_version="1", policy=x.policy, store=store
    )
    assert later["request_ref"] == done["request_ref"] and later["job_name"] == done["job_name"]
    assert later["executed"] is False and later["reused"] is True
    assert len(creates) == 1 and len(x.native.requests) == 1


def test_normal_model_observe_requalifies_original_binding(monkeypatch):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)
    first = flow.run_model_weight_acquisition_flow(
        catalog_id="qwen3_5_9b", catalog_version="1", policy=x.policy, store=store
    )
    x.active["current"] = False
    monkeypatch.setattr(
        x.adapter, "artifacts", lambda *args: pytest.fail("revoked binding reached raw artifacts")
    )
    result = store.observe(first["request_ref"])
    assert result["success"] is False and result["status"] == "observation_failed"
    assert len(x.native.requests) == 1


def test_foreign_internal_store_correlation_cannot_take_campaign_binding(monkeypatch):
    x = bridge_fixture(monkeypatch)
    binding = x.services.build_for_entry(x.entry, x.adapter)
    with pytest.raises(ValueError):
        SessionModelAcquisitionStore(
            runtime_session_id=x.entry.transport_session_id, campaign_binding=binding
        )


def test_normal_tool_backend_keeps_private_campaign_correlation_and_dispatches_real_tool(monkeypatch):
    from amlas.config.settings import Settings
    from amlas.control_plane.campaign_runtime import activate_campaign_entry, activate_campaign_runtime
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    x = bridge_fixture(monkeypatch)
    monkeypatch.setattr(flow, "_resolve_backend", lambda: x.adapter)
    with (
        activate_campaign_runtime(x.runtime),
        activate_campaign_entry(x.entry),
        x.m.activate_model_campaign_services(x.services),
    ):
        backend = ToolBindingBackend(Settings(), approval_policy=x.policy)
        assert backend._acquisition_store.runtime_session_id == x.entry.private_store_correlation
        assert backend._model_acquisition_store.runtime_session_id == x.entry.private_store_correlation
        assert not x.native.requests
        result = backend.execute_tool_use(
            "acquire_model_weights", {"catalog_id": "qwen3_5_9b", "catalog_version": "1"}, confirmed=True
        ).result
    import json

    print("NORMAL_TOOL_RESULT " + json.dumps(result, sort_keys=True))
    assert result["accepted"] and len(x.native.requests) == 1, result
    assert result["request_ref"] in backend._model_acquisition_store._records


def test_active_campaign_without_private_model_services_has_no_legacy_effect(monkeypatch):
    from amlas.config.settings import Settings
    from amlas.control_plane.campaign_runtime import activate_campaign_entry, activate_campaign_runtime
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    x = bridge_fixture(monkeypatch)
    monkeypatch.setattr(
        flow, "_resolve_backend", lambda: pytest.fail("missing campaign services reached legacy backend")
    )
    with activate_campaign_runtime(x.runtime), activate_campaign_entry(x.entry):
        backend = ToolBindingBackend(Settings(), approval_policy=x.policy)
        result = backend.execute_tool_use(
            "acquire_model_weights", {"catalog_id": "qwen3_5_9b", "catalog_version": "1"}, confirmed=True
        ).result
    assert result["accepted"] is False and len(x.native.requests) == 0


def test_actual_sdk_normal_campaign_model_tool_and_fresh_backend_reuse(monkeypatch):
    from test_sdk_acquisition_binding import invoke, request

    from amlas.config.settings import Settings
    from amlas.control_plane.campaign_runtime import activate_campaign_entry, activate_campaign_runtime
    from amlas.tools import model_weight_tools
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    x = bridge_fixture(monkeypatch)
    monkeypatch.setattr(flow, "_resolve_backend", lambda: x.adapter)
    with (
        activate_campaign_runtime(x.runtime),
        activate_campaign_entry(x.entry),
        x.m.activate_model_campaign_services(x.services),
    ):
        first = ToolBindingBackend(Settings(), approval_policy=x.policy)
        result = invoke(
            monkeypatch,
            first,
            [request("first", "acquire_model_weights", {"catalog_id": "qwen3_5_9b", "catalog_version": "1"})],
            tools=[model_weight_tools.acquire_model_weights],
            prompt=x.entry.literal_prompt,
        )["first"]
        cold = ToolBindingBackend(Settings(), approval_policy=x.policy)
        same = invoke(
            monkeypatch,
            cold,
            [request("cold", "acquire_model_weights", {"catalog_id": "qwen3_5_9b", "catalog_version": "1"})],
            tools=[model_weight_tools.acquire_model_weights],
            prompt=x.entry.literal_prompt,
        )["cold"]
    assert result["accepted"] and result["executed"]
    assert same["request_ref"] == result["request_ref"] and same["job_name"] == result["job_name"]
    assert same["executed"] is False and same["reused"] is True and len(x.native.requests) == 1


def test_literal_normal_api_uses_actual_engine_sdk_and_campaign_model_flow(monkeypatch):
    import asyncio

    from test_sdk_acquisition_binding import request
    from test_sdk_prompt_policy import ScriptedModel

    from amlas import api
    from amlas.control_plane.campaign_runtime import activate_campaign_entry, activate_campaign_runtime
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    x = bridge_fixture(monkeypatch)
    # This API regression owns submission/correlation, with a real terminal
    # failure observation; completed artifacts belong to connected owning tests.
    x.native.state = "Failed"
    monkeypatch.setattr(flow, "_resolve_backend", lambda: x.adapter)
    original = ToolBindingBackend.build_agent
    backends = []

    def source_agent(backend, **kwargs):
        backend._model = ScriptedModel(
            [
                [
                    request(
                        "api-model",
                        "acquire_model_weights",
                        {"catalog_id": "qwen3_5_9b", "catalog_version": "1"},
                    )
                ]
            ]
        )
        backends.append(backend)
        agent = original(backend, **kwargs)
        agent.callback_handler = lambda *args, **kwargs: None
        return agent

    monkeypatch.setattr(ToolBindingBackend, "build_agent", source_agent)
    with (
        activate_campaign_runtime(x.runtime),
        activate_campaign_entry(x.entry),
        x.m.activate_model_campaign_services(x.services),
    ):
        response = asyncio.run(api.run_prompt(x.entry.literal_prompt, policy=x.policy))
    assert response is not None and backends
    assert response.data["warm_model_continuation"] == {"disposition": "terminal_failed"}
    assert len(x.native.requests) == 1
    assert backends[0]._model_acquisition_store.runtime_session_id == x.entry.private_store_correlation


def test_model_store_cannot_reuse_bound_campaign_under_foreign_active_entry(monkeypatch):
    from amlas.control_plane.campaign_runtime import activate_campaign_entry, activate_campaign_runtime

    first = bridge_fixture(monkeypatch)
    store = bound_store(first)
    other = bridge_fixture(monkeypatch)
    with activate_campaign_runtime(other.runtime), activate_campaign_entry(other.entry):
        result = flow.run_model_weight_acquisition_flow(
            catalog_id="qwen3_5_9b", catalog_version="1", policy=first.policy, store=store
        )
    assert result["accepted"] is False
    assert first.native.requests == [] and other.native.requests == []


def _active_entry(x, *, services=True):
    from contextlib import ExitStack

    from amlas.control_plane.campaign_runtime import activate_campaign_entry, activate_campaign_runtime

    stack = ExitStack()
    stack.enter_context(activate_campaign_runtime(x.runtime))
    stack.enter_context(activate_campaign_entry(x.entry))
    if services:
        stack.enter_context(x.m.activate_model_campaign_services(x.services))
    return stack


def _acquire(x, store):
    return flow.run_model_weight_acquisition_flow(
        catalog_id="qwen3_5_9b", catalog_version="1", policy=x.policy, store=store
    )


def test_legacy_backend_constructed_before_campaign_denies_before_legacy_resolver(monkeypatch):
    from amlas.config.settings import Settings
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    x = bridge_fixture(monkeypatch)
    backend = ToolBindingBackend(Settings(), approval_policy=x.policy)
    reached = []

    def legacy():
        reached.append("legacy")
        raise ValueError("neutral legacy resolver")

    monkeypatch.setattr(flow, "_resolve_backend", legacy)
    with _active_entry(x, services=False):
        execution = backend.execute_tool_use(
            "acquire_model_weights", {"catalog_id": "qwen3_5_9b", "catalog_version": "1"}, confirmed=True
        )
    assert reached == [], "active campaign reached legacy resolver from old backend"
    assert execution.status == "error" and execution.result is None and x.native.requests == []


@pytest.mark.parametrize("existing", [False, True])
def test_legacy_store_cannot_dispatch_or_adopt_catalog_in_active_campaign(monkeypatch, existing):
    x = bridge_fixture(monkeypatch)
    store = SessionModelAcquisitionStore()
    reached = []
    if existing:
        store._catalogs[("qwen3_5_9b", "1")] = "model_acq_" + "d" * 32
        monkeypatch.setattr(
            store, "observe", lambda ref: reached.append("legacy-reference") or {"accepted": True}
        )

    def legacy():
        reached.append("legacy-resolver")
        raise ValueError("neutral legacy resolver")

    monkeypatch.setattr(flow, "_resolve_backend", legacy)
    with _active_entry(x):
        result = _acquire(x, store)
    assert reached == [], "active campaign adopted legacy ordinary state"
    assert result["accepted"] is False and x.native.requests == []


def test_unbound_campaign_store_preserves_construction_factory_even_matching_original_ids(monkeypatch):
    import copy

    x = bridge_fixture(monkeypatch)
    store = SessionModelAcquisitionStore(
        runtime_session_id=x.entry.private_store_correlation,
        campaign_entry=x.entry,
        campaign_services=x.services,
    )
    original_entry = x.entry
    other_runtime = copy.copy(x.runtime)
    other_entry = other_runtime.begin_prompt(original_entry.literal_prompt, policy=x.policy)
    assert other_entry._factory is not original_entry._factory
    assert (other_entry.campaign_ref, other_entry.campaign_id, other_entry.private_store_correlation) == (
        original_entry.campaign_ref,
        original_entry.campaign_id,
        original_entry.private_store_correlation,
    )
    other_services = copy.copy(x.services)
    other_services._runtime = other_runtime
    x.runtime, x.entry, x.services = other_runtime, other_entry, other_services
    monkeypatch.setattr(flow, "_resolve_backend", lambda: x.adapter)
    with _active_entry(x):
        result = _acquire(x, store)
    assert result["accepted"] is False, "lazy binding replaced original factory anchor"
    assert store._campaign_binding is None and x.native.requests == []


def test_same_entry_empty_planning_then_original_spec_publication_refreshes_before_model_binding(monkeypatch):
    from amlas.config.settings import Settings
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    x = bridge_fixture(monkeypatch)
    original_loader = x.planning.load_original_refs
    monkeypatch.setattr(x.planning, "load_original_refs", lambda **kwargs: ())
    x.entry = x.runtime.begin_prompt(x.entry.literal_prompt, policy=x.policy)
    assert x.entry.planning_refs == ()
    old_entry = x.entry
    monkeypatch.setattr(flow, "_resolve_backend", lambda: x.adapter)
    with _active_entry(x):
        backend = ToolBindingBackend(Settings(), approval_policy=x.policy)
        monkeypatch.setattr(x.planning, "load_original_refs", original_loader)
        result = backend.execute_tool_use(
            "acquire_model_weights", {"catalog_id": "qwen3_5_9b", "catalog_version": "1"}, confirmed=True
        ).result
    assert result["accepted"] is True, "normal lazy bind kept stale original planning snapshot"
    assert old_entry.planning_refs == (), "frozen original entry was mutated"
    assert backend._model_acquisition_store._campaign_entry is old_entry
    assert backend._model_acquisition_store._campaign_binding.entry is not old_entry
    assert len(x.native.requests) == 1


def test_post_create_model_revocation_is_unknown_outcome_not_false_no_effect(monkeypatch):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)
    original_create = x.native.create_processing_job

    def create(**kwargs):
        answer = original_create(**kwargs)
        x.active["current"] = False
        return answer

    monkeypatch.setattr(x.native, "create_processing_job", create)
    result = _acquire(x, store)
    assert len(x.native.requests) == 1
    assert result["executed"] is None, "post-dispatch revocation falsely reported no effect"
    assert result["status"] == "outcome_unknown" and result["accepted"] is False
    assert not {"request_ref", "job_name", "job_arn"}.intersection(result)
    assert store._records == {} and store._catalogs == {}


def test_post_create_outcome_record_failure_retains_unknown_without_another_create(monkeypatch):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)

    def fail(*args, **kwargs):
        raise TimeoutError("neutral outcome acknowledgement unavailable")

    monkeypatch.setattr(store._campaign_binding.journal, "record_outcome", fail)
    result = _acquire(x, store)
    assert len(x.native.requests) == 1
    assert result["executed"] is not False, "known attempted Create was called unexecuted"
    assert result["status"] == "outcome_unknown"
    second = _acquire(x, bound_store(x))
    assert len(x.native.requests) == 1 and second["reused"] is True


@pytest.mark.parametrize("operation", ["submit", "observe"])
def test_store_requalifies_full_model_boundary_after_its_runtime_exposure_checks(monkeypatch, operation):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)
    binding = store._campaign_binding
    phase = {"returned": False}
    if operation == "observe":
        first = _acquire(x, store)
        original_operation = binding.observe_original
    else:
        original_operation = binding.submit_original

    def returned(*args, **kwargs):
        answer = original_operation(*args, **kwargs)
        phase["returned"] = True
        return answer

    name = "observe_original" if operation == "observe" else "submit_original"
    monkeypatch.setattr(binding, name, returned)
    original_check = type(x.entry).require_current

    def runtime_only(entry):
        answer = original_check(entry)
        if entry is x.entry and phase["returned"]:
            x.active["current"] = False
        return answer

    monkeypatch.setattr(type(x.entry), "require_current", runtime_only)
    result = store.observe(first["request_ref"]) if operation == "observe" else _acquire(x, store)
    assert result["accepted"] is False, "store exposed after model-specific boundary revocation"
    assert result["status"] in {"observation_failed", "outcome_unknown"}
    assert len(x.native.requests) == 1


def test_failed_final_store_qualification_never_publishes_partial_ordinary_aliases(monkeypatch):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)
    original = store._require_original_campaign_context
    count = {"checks": 0}

    def final():
        count["checks"] += 1
        if count["checks"] == 2:
            raise ValueError("neutral final exposure qualification unavailable")
        return original()

    monkeypatch.setattr(store, "_require_original_campaign_context", final)
    result = _acquire(x, store)
    assert len(x.native.requests) == 1 and count["checks"] == 2
    assert store._records == {} and store._catalogs == {}, (
        "refused store operation published ordinary aliases"
    )
    assert store.existing("qwen3_5_9b", "1") is None
    assert result["executed"] is None and result["status"] == "outcome_unknown"


def test_exact_before_create_refusal_stays_false_even_when_current_qualification_is_lost(monkeypatch):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)
    binding = store._campaign_binding
    original = binding._approve
    calls = {"approval": 0}

    def approval(policy):
        calls["approval"] += 1
        if calls["approval"] == 2:
            x.active["current"] = False
            raise ValueError("neutral current model effect policy refused")
        return original(policy)

    monkeypatch.setattr(binding, "_approve", approval)
    result = _acquire(x, store)
    assert calls["approval"] == 2 and x.native.requests == []
    assert result["executed"] is False, "independent source before_create refusal lost its no-dispatch fact"
    assert result["status"] == "submission_failed" and result["accepted"] is False
    assert not {"request_ref", "job_name", "job_arn"}.intersection(result)


def test_opaque_adapter_exception_is_unknown_even_when_consumed_marker_exists(monkeypatch):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)

    def opaque(*args, **kwargs):
        raise RuntimeError("neutral opaque model invocation failure")

    monkeypatch.setattr(x.adapter, "submit_prepared", opaque)
    result = _acquire(x, store)
    assert x.native.requests == []
    assert result["executed"] is None and result["status"] == "outcome_unknown"
    assert not {"request_ref", "job_name", "job_arn"}.intersection(result)


def test_arbitrary_store_invocation_exception_cannot_assert_no_native_dispatch(monkeypatch):
    x = bridge_fixture(monkeypatch)
    store = bound_store(x)

    def opaque(**kwargs):
        raise RuntimeError("neutral opaque private model invocation failure")

    monkeypatch.setattr(store, "submit_original", opaque)
    result = _acquire(x, store)
    assert result["executed"] is None and result["status"] == "outcome_unknown"
    assert not {"request_ref", "job_name", "job_arn"}.intersection(result)
    assert x.native.requests == []


def test_reused_original_final_exposure_denial_is_unknown_not_before_create_refusal(monkeypatch):
    x = bridge_fixture(monkeypatch)
    first = _acquire(x, bound_store(x))
    assert first["executed"] is True and len(x.native.requests) == 1
    cold = bound_store(x)
    original = cold._require_original_campaign_context
    calls = {"checks": 0}

    def final():
        calls["checks"] += 1
        if calls["checks"] == 2:
            raise ValueError("neutral reused original exposure qualification unavailable")
        return original()

    monkeypatch.setattr(cold, "_require_original_campaign_context", final)
    result = _acquire(x, cold)
    assert calls["checks"] == 2 and len(x.native.requests) == 1
    assert result["executed"] is None and result["status"] == "outcome_unknown"
    assert cold._records == {} and cold._catalogs == {}
    assert not {"request_ref", "job_name", "job_arn"}.intersection(result)
