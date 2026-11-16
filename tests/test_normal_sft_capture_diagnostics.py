"""Closed SFT-capture diagnostics; neutral SDK state and no native effects."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from strands import Agent
from test_normal_completion_sdk_none_probe import FinalTextModel

from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("test_normal_run_checkpoint",)
MESSAGE = "normal_sft_capture_unavailable run_id=%s phase=%s error_class=%s"
PRIVATE_DETAIL = "ordinary withheld diagnostic fixture"


class NeutralFixtureError(RuntimeError):
    pass


@pytest.fixture
def qualified_sft_catalog(monkeypatch, normal_work):
    import model_acquisition_fixtures as model_fixture
    import test_qwen_sft_backend as sft_fixture

    for key, value in {
        "SAGEMAKER_EXECUTION_ROLE_ARN": model_fixture.ROLE,
        "AMLAS_QWEN_SFT_IMAGE_URI": model_fixture.ACCOUNT + sft_fixture.IMAGE[len(sft_fixture.ACCOUNT) :],
        "AMLAS_QWEN_SFT_SOURCE_COMMIT": sft_fixture.SOURCE,
        "AMLAS_QWEN_SFT_ADMISSION_DIGEST": "b" * 64,
        "AMLAS_DEPLOYMENT_ID": model_fixture.DEPLOYMENT,
        "AMLAS_DEPLOYMENT_PREFIX": "fixture-lab",
    }.items():
        monkeypatch.setenv(key, value)
    return normal_work


@pytest.mark.parametrize(
    ("error", "category"),
    [(ValueError(PRIVATE_DETAIL), "ValueError"), (NeutralFixtureError(PRIVATE_DETAIL), "Exception")],
)
def test_closed_warning_validates_labels_and_hides_exception(monkeypatch, caplog, error, category):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    binding._warn_normal_sft_capture("a" * 32, "snapshot_budget", error)
    binding._warn_normal_sft_capture("a" * 31, "snapshot_budget", error)
    binding._warn_normal_sft_capture("a" * 32, "unreviewed_phase", error)
    records = [record for record in caplog.records if record.msg == MESSAGE]
    assert len(records) == 1
    assert records[0].args == ("a" * 32, "snapshot_budget", category)
    assert records[0].exc_info is None and records[0].stack_info is None
    assert PRIVATE_DETAIL not in caplog.text and "NeutralFixtureError" not in caplog.text


def test_warning_failures_do_not_change_refusal_or_process_control(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError(PRIVATE_DETAIL)

    monkeypatch.setattr(binding.logger, "warning", unavailable)
    assert binding._warn_normal_sft_capture("a" * 32, "snapshot_budget", ValueError()) is None
    control = KeyboardInterrupt()

    def stop(*args, **kwargs):
        raise control

    monkeypatch.setattr(binding.logger, "warning", stop)
    with pytest.raises(KeyboardInterrupt) as observed:
        binding._warn_normal_sft_capture("a" * 32, "snapshot_budget", ValueError())
    assert observed.value is control


def test_unselected_tool_produces_no_capture_diagnostic(caplog):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    backend = object.__new__(binding.ToolBindingBackend)
    backend._capture_normal_sft_call(SimpleNamespace(tool_use={"name": "unrelated_tool"}))
    assert not [record for record in caplog.records if record.msg == MESSAGE]


def _prefix(text):
    return [
        {"role": "user", "content": [{"text": text}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "ordinary-use",
                        "name": "submit_qwen_sft",
                        "input": {
                            "model_request_ref": "model_acq_" + "1" * 32,
                            "preparation_request_ref": "prep_" + "2" * 32,
                        },
                    }
                }
            ],
        },
    ]


def test_simultaneous_capture_phases_are_per_call(qualified_sft_catalog):
    transport, session_hash = runtime._prompt_session_identity()
    contexts = [
        checkpoint.prepare_normal_run(c * 32, session_hash, transport_session_id=transport) for c in "ab"
    ]
    prefixes = [_prefix("ordinary small history"), _prefix("ordinary large history " + "x" * (158 * 1024))]
    agents = [
        Agent(model=FinalTextModel(), tools=[], messages=copy.deepcopy(prefix), callback_handler=None)
        for prefix in prefixes
    ]
    snapshots = [binding._record_session_snapshot(agent) for agent in agents]
    for prefix, snapshot in zip(prefixes, snapshots, strict=True):
        call = checkpoint._causal_call("ordinary-use", prefix)
        assert len(checkpoint._canonical(call)) < checkpoint._MAX_BYTES
        assert len(checkpoint._canonical(snapshot)) < checkpoint._MAX_BYTES
    snapshots[0].pop("app_data")
    barrier = threading.Barrier(2)

    def capture(index):
        phases = []

        def on_phase(phase):
            phases.append(phase)
            if phase == "snapshot_inventory":
                barrier.wait(timeout=5)

        with checkpoint.activate_normal_run(contexts[index]):
            with pytest.raises(ValueError) as observed:
                contexts[index].capture_sft_call(
                    "ordinary-use",
                    prefixes[index],
                    agent_snapshot_dict=snapshots[index],
                    _diagnostic_phase=on_phase,
                )
            assert type(observed.value) is ValueError
            expected = (
                "public SDK session snapshot envelope differs"
                if index == 0
                else "normal run DATA size exceeded"
            )
            assert str(observed.value) == expected
        return phases

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(capture, 0)
        second = pool.submit(capture, 1)
        a, b = first.result(timeout=15), second.result(timeout=15)
    assert a[-1] == "snapshot_inventory"
    assert b[-1] == "snapshot_budget"
    assert not qualified_sft_catalog[2].calls


def _context_and_snapshot(normal_work):
    transport, session_hash = runtime._prompt_session_identity()
    context = checkpoint.prepare_normal_run("c" * 32, session_hash, transport_session_id=transport)
    prefix = _prefix("ordinary admitted history")
    model = FinalTextModel()
    snapshot = binding._record_session_snapshot(
        Agent(model=model, tools=[], messages=copy.deepcopy(prefix), callback_handler=None)
    )
    return context, prefix, snapshot, model


def test_successful_sdk_capture_ignores_bad_phase_callback_and_retains_exact_snapshot(
    qualified_sft_catalog, caplog
):
    context, prefix, snapshot, model = _context_and_snapshot(qualified_sft_catalog)
    original = copy.deepcopy(snapshot)
    phases = []

    def refuse_callback(phase):
        phases.append(phase)
        raise RuntimeError(PRIVATE_DETAIL)

    caplog.set_level(logging.WARNING, logger=binding.__name__)
    with checkpoint.activate_normal_run(context):
        context.capture_sft_call(
            "ordinary-use", prefix, agent_snapshot_dict=snapshot, _diagnostic_phase=refuse_callback
        )
        assert context._captured_sft_snapshot("ordinary-use", prefix) == original
    assert phases and snapshot == original and not model.received_messages
    assert not [record for record in caplog.records if record.msg == MESSAGE]


def test_bad_phase_callback_cannot_replace_original_capture_error(qualified_sft_catalog, monkeypatch):
    context, prefix, snapshot, _model = _context_and_snapshot(qualified_sft_catalog)
    original = ValueError("ordinary fixed failure")

    def fail_budget(*args, **kwargs):
        raise original

    def fail_callback(*args, **kwargs):
        raise RuntimeError(PRIVATE_DETAIL)

    monkeypatch.setattr(checkpoint, "_budget_sft_snapshot", fail_budget)
    with checkpoint.activate_normal_run(context), pytest.raises(ValueError) as observed:
        context.capture_sft_call(
            "ordinary-use", prefix, agent_snapshot_dict=snapshot, _diagnostic_phase=fail_callback
        )
    assert observed.value is original


@pytest.mark.parametrize(
    "failure_phase",
    [
        "current_before",
        "message_binding",
        "snapshot_create",
        "causal_call",
        "snapshot_inventory",
        "snapshot_budget",
        "capture_repeat",
        "current_after",
    ],
)
def test_each_capture_boundary_reports_its_own_closed_phase(monkeypatch, caplog, failure_phase):
    run_id = "d" * 32
    use = {"toolUseId": "ordinary-use", "name": "submit_qwen_sft", "input": {}}
    messages = [{"role": "assistant", "content": [{"toolUse": use}]}]
    state = {}
    error = ValueError(PRIVATE_DETAIL)
    calls = []

    def snapshot(*args, **kwargs):
        if failure_phase == "snapshot_create":
            raise error
        return SimpleNamespace(to_dict=lambda: {})

    agent = SimpleNamespace(messages=messages, take_snapshot=snapshot)
    backend = object.__new__(binding.ToolBindingBackend)
    witness = binding._PendingBatchPause(
        agent=agent,
        invocation_state=state,
        request_state={},
        research=None,
        checker=None,
        pid=0,
        prefix=backend._pause_history([]),
        batch=backend._pause_history(messages[-1]),
        precommit=True,
    )
    backend._pending_batch_pause = witness
    if failure_phase == "message_binding":
        witness.batch = "different ordinary batch"

    def qualify(_witness):
        calls.append(True)
        if failure_phase == "current_before" or (failure_phase == "current_after" and len(calls) == 2):
            raise error

    def capture(*args, _diagnostic_phase, **kwargs):
        if failure_phase in {"causal_call", "snapshot_inventory", "snapshot_budget", "capture_repeat"}:
            _diagnostic_phase(failure_phase)
            raise error

    monkeypatch.setattr(backend, "_qualify_pending_pause", qualify)
    monkeypatch.setattr(
        checkpoint, "current_normal_run", lambda: SimpleNamespace(run_id=run_id, capture_sft_call=capture)
    )
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    phases = []
    expected_error = binding._InputAdmissionHeldError if failure_phase == "message_binding" else ValueError
    with pytest.raises(expected_error) as observed:
        backend._capture_normal_sft_call(
            SimpleNamespace(tool_use=use, agent=agent, invocation_state=state),
            _diagnostic_phase=phases.append,
        )
    if failure_phase != "message_binding":
        assert observed.value is error
    assert phases[-1] == failure_phase
    assert not [record for record in caplog.records if record.msg == MESSAGE]
    assert PRIVATE_DETAIL not in caplog.text


def test_actual_sdk_successful_capture_has_no_warning_or_snapshot_phase_data(
    qualified_sft_catalog, monkeypatch, caplog
):
    context, prefix, _snapshot, model = _context_and_snapshot(qualified_sft_catalog)
    agent = Agent(model=model, tools=[], messages=copy.deepcopy(prefix), callback_handler=None)
    observed_snapshots = []
    original_snapshot = agent.take_snapshot

    def observe_snapshot(*args, **kwargs):
        snapshot = original_snapshot(*args, **kwargs)

        def to_dict():
            value = snapshot.to_dict()
            observed_snapshots.append(copy.deepcopy(value))
            return value

        return SimpleNamespace(to_dict=to_dict)

    monkeypatch.setattr(agent, "take_snapshot", observe_snapshot)
    backend = object.__new__(binding.ToolBindingBackend)
    state = {}
    backend._pending_batch_pause = binding._PendingBatchPause(
        agent=agent,
        invocation_state=state,
        request_state={},
        research=None,
        checker=None,
        pid=0,
        prefix=backend._pause_history(prefix[:-1]),
        batch=backend._pause_history(prefix[-1]),
        precommit=True,
    )
    monkeypatch.setattr(backend, "_qualify_pending_pause", lambda _witness: None)
    event = SimpleNamespace(tool_use=prefix[-1]["content"][0]["toolUse"], agent=agent, invocation_state=state)
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    with checkpoint.activate_normal_run(context):
        backend._capture_normal_sft_call(event)
        captured = context._captured_sft_snapshot("ordinary-use", prefix)
    # The capture is the record form of the one snapshot the SDK gave.
    observed = SimpleNamespace(
        take_snapshot=lambda preset: SimpleNamespace(to_dict=lambda: copy.deepcopy(observed_snapshots[0]))
    )
    assert len(observed_snapshots) == 1 and captured == binding._record_session_snapshot(observed)
    assert captured["data"]["messages"] == prefix and not model.received_messages
    assert set(captured) == {"scope", "schema_version", "created_at", "data", "app_data"}
    assert set(captured["data"]) == {"messages", "state", "conversation_manager_state", "interrupt_state"}
    assert not [record for record in caplog.records if record.msg == MESSAGE]


CAUSAL_SUBPHASES = (
    "causal_bounds",
    "causal_checkpoint_data",
    "causal_canonical_data",
    "causal_assistant_batch",
    "causal_unique_sft_use",
    "causal_sft_inputs",
    "causal_call_budget",
)


def _measured_canonical_bytes(value):
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _causal_failure_case(case):
    """Neutral DATA hits a real guard; never inject an exception into a validator."""
    identifier = "ordinary-use"
    prefix = _prefix("ordinary admitted history")
    if case == "bounds":
        identifier = ""
        phase, message = "causal_bounds", "bounded original SFT call DATA required"
    elif case == "private_state":
        prefix[0]["ordinary_metadata"] = {"client": None}
        phase, message = "causal_checkpoint_data", "checkpoint contains private execution state"
    elif case == "nonfinite":
        prefix[0]["ordinary_metadata"] = {"ordinary_metric": float("inf")}
        phase, message = "causal_canonical_data", "Out of range float values are not JSON compliant"
    elif case == "prefix_budget":
        prefix[0]["content"][0]["text"] = "ordinary text " + "x" * (160 * 1024)
        phase, message = "causal_canonical_data", "normal run DATA size exceeded"
    elif case == "assistant_batch":
        prefix = prefix[:-1]
        phase, message = "causal_assistant_batch", "original assistant SFT batch missing"
    elif case == "duplicate_use":
        prefix[-1]["content"].append(copy.deepcopy(prefix[-1]["content"][0]))
        phase, message = "causal_unique_sft_use", "unique actual original SFT tool use required"
    elif case == "preview":
        prefix[-1]["content"][0]["toolUse"]["input"]["plan_only"] = True
        phase, message = "causal_sft_inputs", "original SFT tool selection differs"
    else:
        assert case == "final_budget"
        use = prefix[-1]["content"][0]["toolUse"]
        wrapper = {"pending_tool_use_id": identifier, "causal_prefix": prefix}
        final = {**wrapper, "tool_use": use}
        addition = _measured_canonical_bytes(final) - _measured_canonical_bytes(wrapper)
        padding = 160 * 1024 - _measured_canonical_bytes(wrapper) - addition // 2
        assert padding > 0
        prefix[0]["content"][0]["text"] += "x" * padding
        assert _measured_canonical_bytes(wrapper) <= 160 * 1024
        assert _measured_canonical_bytes(final) > 160 * 1024
        phase, message = "causal_call_budget", "normal run DATA size exceeded"
    return identifier, prefix, phase, message


@pytest.mark.parametrize(
    "case",
    [
        "bounds",
        "private_state",
        "nonfinite",
        "prefix_budget",
        "assistant_batch",
        "duplicate_use",
        "preview",
        "final_budget",
    ],
)
def test_actual_causal_validator_failure_reports_exact_subphase(qualified_sft_catalog, caplog, case):
    context, _valid_prefix, snapshot, model = _context_and_snapshot(qualified_sft_catalog)
    identifier, prefix, expected_phase, expected_message = _causal_failure_case(case)
    prefix_before, snapshot_before = copy.deepcopy(prefix), copy.deepcopy(snapshot)
    _work, ports, _normal_model = qualified_sft_catalog
    native_before = (copy.deepcopy(ports.items), copy.deepcopy(ports.objects), list(ports.calls))
    phases = []
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    with checkpoint.activate_normal_run(context), pytest.raises(ValueError) as observed:
        context.capture_sft_call(
            identifier, prefix, agent_snapshot_dict=snapshot, _diagnostic_phase=phases.append
        )
    assert str(observed.value) == expected_message, "the real first failing guard changed"
    assert phases[-1] == expected_phase, "the real causal refusal still has an undifferentiated phase"
    assert prefix == prefix_before and snapshot == snapshot_before
    assert context._causal_call is None and context._sft_snapshot is None
    assert not model.received_messages
    assert (ports.items, ports.objects, ports.calls) == native_before
    # The existing warning consumes only the trusted literal marked by the real
    # validator; actual exception text, input, snapshot and references stay private.
    binding._warn_normal_sft_capture(context.run_id, phases[-1], observed.value)
    found = [record for record in caplog.records if record.msg == MESSAGE]
    assert len(found) == 1 and found[0].args == (context.run_id, expected_phase, "ValueError")
    assert found[0].exc_info is None and found[0].stack_info is None
    assert expected_message not in found[0].getMessage()
    assert "model_acq_" not in found[0].getMessage() and "prep_" not in found[0].getMessage()


def test_real_causal_call_optional_diagnostic_preserves_default_result_and_prefix():
    prefix = _prefix("ordinary direct causal DATA")
    before = copy.deepcopy(prefix)
    expected = checkpoint._causal_call("ordinary-use", prefix)
    phases = []
    actual = checkpoint._causal_call("ordinary-use", prefix, _diagnostic_phase=phases.append)
    assert actual == expected and prefix == before
    assert phases == list(CAUSAL_SUBPHASES[:-1])


def test_actual_successful_capture_forwards_all_causal_subphases_without_warning(
    qualified_sft_catalog, caplog
):
    context, prefix, snapshot, model = _context_and_snapshot(qualified_sft_catalog)
    original = copy.deepcopy(snapshot)
    phases = []
    _work, ports, _normal_model = qualified_sft_catalog
    native_before = (copy.deepcopy(ports.items), copy.deepcopy(ports.objects), list(ports.calls))
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    with checkpoint.activate_normal_run(context):
        context.capture_sft_call(
            "ordinary-use", prefix, agent_snapshot_dict=snapshot, _diagnostic_phase=phases.append
        )
        assert context._captured_sft_snapshot("ordinary-use", prefix) == original
    assert [phase for phase in phases if phase in CAUSAL_SUBPHASES] == list(CAUSAL_SUBPHASES)
    assert snapshot == original and not model.received_messages
    assert (ports.items, ports.objects, ports.calls) == native_before
    assert not [record for record in caplog.records if record.msg == MESSAGE]


def test_ordinary_diagnostic_failure_cannot_replace_real_causal_refusal(qualified_sft_catalog):
    context, _valid_prefix, snapshot, model = _context_and_snapshot(qualified_sft_catalog)
    identifier, prefix, expected_phase, expected_message = _causal_failure_case("preview")
    before = copy.deepcopy(prefix), copy.deepcopy(snapshot)
    phases = []

    def refuse_diagnostic(phase):
        phases.append(phase)
        raise RuntimeError(PRIVATE_DETAIL)

    with checkpoint.activate_normal_run(context), pytest.raises(ValueError) as observed:
        context.capture_sft_call(
            identifier, prefix, agent_snapshot_dict=snapshot, _diagnostic_phase=refuse_diagnostic
        )
    assert str(observed.value) == expected_message and phases[-1] == expected_phase
    assert (prefix, snapshot) == before and not model.received_messages
    assert context._causal_call is None and context._sft_snapshot is None


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_causal_subphase_process_control_propagates_without_capture(qualified_sft_catalog, control_type):
    context, prefix, snapshot, model = _context_and_snapshot(qualified_sft_catalog)
    before = copy.deepcopy(prefix), copy.deepcopy(snapshot)
    control = control_type(PRIVATE_DETAIL)
    phases = []

    def stop(phase):
        phases.append(phase)
        if phase == "causal_sft_inputs":
            raise control

    with checkpoint.activate_normal_run(context), pytest.raises(control_type) as observed:
        context.capture_sft_call("ordinary-use", prefix, agent_snapshot_dict=snapshot, _diagnostic_phase=stop)
    assert observed.value is control and phases[-1] == "causal_sft_inputs"
    assert (prefix, snapshot) == before and not model.received_messages
    assert context._causal_call is None and context._sft_snapshot is None


def test_sdk_capture_keeps_coarse_phase_and_reports_closed_input_predicate(qualified_sft_catalog):
    context, _valid_prefix, snapshot, _model = _context_and_snapshot(qualified_sft_catalog)
    identifier, prefix, _expected_phase, _expected_message = _causal_failure_case("preview")
    before_prefix, before_snapshot = copy.deepcopy(prefix), copy.deepcopy(snapshot)
    phases, predicates = [], []
    with (
        checkpoint.activate_normal_run(context),
        pytest.raises(ValueError, match=r"^original SFT tool selection differs$"),
    ):
        context.capture_sft_call(
            identifier,
            prefix,
            agent_snapshot_dict=snapshot,
            _diagnostic_phase=phases.append,
            _diagnostic_predicate=predicates.append,
        )
    assert phases[-1] == "causal_sft_inputs"
    assert predicates == ["PLAN_PREVIEW"]
    assert prefix == before_prefix
    assert snapshot == before_snapshot


def test_closed_sft_predicate_warning_exposes_only_fixed_code(caplog):
    warn = getattr(binding, "_warn_normal_sft_input_predicate", None)
    assert callable(warn)
    caplog.set_level(logging.WARNING)
    warn("a" * 32, "causal_sft_inputs", "EXTRA_KEYS")
    warn("a" * 32, "causal_sft_inputs", "ordinary_extra")
    warn("a" * 32, "other_phase", "EXTRA_KEYS")
    messages = [r.getMessage() for r in caplog.records if "normal_sft_input_predicate" in r.getMessage()]
    assert messages == ["normal_sft_input_predicate run_id=" + "a" * 32 + " check=EXTRA_KEYS"]
    assert all("ordinary_extra" not in message for message in messages)


def test_sft_predicate_warning_rejects_non_plain_phase_without_comparison(caplog):
    class Phase:
        def __eq__(self, other):
            raise AssertionError("phase must not be compared")

    caplog.set_level(logging.WARNING)
    binding._warn_normal_sft_input_predicate("a" * 32, Phase(), "EXTRA_KEYS")
    assert not caplog.records
