"""Before-effect parent capture boundaries through the genuine normal SDK graph.

Inert native response ports prove source ordering only, never native authority,
worker-loss recovery, executable paired quality or full lifecycle acceptance.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from types import SimpleNamespace

import pytest
from normal_sft_cold_fixture import (
    SFT_USE,
    close,
    complete_fixture,
    freeze_managed_case_bundle,
    official_case_data,
    payloads,
    publish_data,
    publish_preparation,
    runtime,
    setup,
    wait_for,
)
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.control_plane import normal_run_parents
from amlas.control_plane.normal_run_checkpoint import _causal_call
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

MAX_TASK_BYTES = 160 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


async def reach_boundary(state, monkeypatch, observed):
    """Complete actual original parents; do not require an accepted SFT effect."""
    received = []

    async def read():
        async for event in runtime.agent_invocation({"prompt": state.work.literal}):
            received.append(event)
        return received

    state.reader = asyncio.create_task(read())
    state.events = received
    await wait_for(
        state,
        state.reader,
        lambda: (
            state.work.engines
            and len(state.work.engines[0].tool_backend._acquisition_store._records) == 2
            and state.work.engines[0].tool_backend._model_acquisition_store._records
        ),
        "original model and two acquired data records",
    )
    backend = state.work.engines[0].tool_backend
    state.backend = backend
    model = next(iter(backend._model_acquisition_store._records.values()))
    complete_fixture(
        monkeypatch,
        state.directory / "native-model",
        SimpleNamespace(handle=model.handle, adapter=state.work.adapter, native=state.work.native),
    )
    for key, (version, raw) in state.work.s3.objects.items():
        state.objects.install(key, raw, version)
    for record in backend._acquisition_store._records.values():
        publish_data(state, record)
    await wait_for(
        state, state.reader, lambda: backend._case_freeze_store._records, "original case native record"
    )
    cases = next(iter(backend._case_freeze_store._records.values()))
    for name, raw in freeze_managed_case_bundle(
        official_case_data()[1], source=cases.extra["source"], header=cases.extra["header"]
    ).items():
        state.objects.install(cases.extra["output_prefix"] + name, raw)
    state.native.states[cases.id] = "Completed"
    await wait_for(
        state, state.reader, lambda: backend._preparation_store._records, "original preparation native record"
    )
    preparation = next(iter(backend._preparation_store._records.values())).handle
    publish_preparation(state, preparation)
    await wait_for(state, state.reader, lambda: observed, "actual original SFT before-effect boundary")
    await wait_for(
        state,
        state.reader,
        lambda: state.reader.done() or state.native.calls,
        "closed before-effect refusal or original native Create",
    )
    assert len(backend._warm_graph()) == 7
    assert len(backend._acquisition_store._records) == 2
    assert len(backend._model_acquisition_store._records) == 1
    assert len(backend._case_freeze_store._records) == len(backend._preparation_store._records) == 1
    assert state.native.states[cases.id] == state.native.states[preparation.id] == "Completed"
    if state.reader.done():
        await state.reader


def test_parent_capture_table_incarnation_revocation_stops_next_native_read_and_effect(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        observed, later_reads, later_writes = [], [], []
        phase = {"capture": False, "revoked": False}
        original_capture = normal_run_parents.publish_parent_graph
        original_table = state.ports.describe_table

        def table(**kwargs):
            result = original_table(**kwargs)
            if phase["revoked"]:
                result["Table"]["TableId"] = "aabbccdd-c954-4a9e-99cc-e036a74353ba"
            return result

        def capture(*args, **kwargs):
            assert len(state.native.calls) == 0
            phase["capture"] = True
            observed.append(True)
            try:
                return original_capture(*args, **kwargs)
            finally:
                phase["capture"] = False

        def observe(port, name, *, read):
            original = getattr(port, name)

            def call(*args, **kwargs):
                if phase["revoked"]:
                    (later_reads if read else later_writes).append(name)
                result = original(*args, **kwargs)
                if phase["capture"] and not phase["revoked"] and name == "describe_processing_job":
                    phase["revoked"] = True
                return result

            monkeypatch.setattr(port, name, call)

        monkeypatch.setattr(state.ports, "describe_table", table)
        monkeypatch.setattr(normal_run_parents, "publish_parent_graph", capture)
        for port in (state.work.native, state.native, state.data.native):
            for name in ("describe_processing_job", "describe_training_job", "list_tags"):
                if hasattr(port, name):
                    observe(port, name, read=True)
            for name in ("create_processing_job", "create_training_job"):
                if hasattr(port, name):
                    observe(port, name, read=False)
        for port in (state.objects, state.work.s3, state.data.s3, state.ports):
            for name in ("get_object", "head_object", "get_bucket_location"):
                if hasattr(port, name):
                    observe(port, name, read=True)
            for name in ("put_object", "copy_object"):
                if hasattr(port, name):
                    observe(port, name, read=False)
        try:
            await reach_boundary(state, monkeypatch, observed)
            assert phase["revoked"], "the real first parent Describe did not reach the revocation seam"
            assert state.native.calls == [] and later_writes == []
            assert state.ports.objects == {}, "unqualified parent native bodies were published"
            assert later_reads == [], "parent capture performed another native read after TableId revocation"
        finally:
            await close(state)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "logger_failure",
    [False, True, "diagnostic", "interrupt"],
    ids=["closed-warning", "raising-logger", "diagnostic-logger-failure", "diagnostic-process-control"],
)
def test_full_sdk_task_budget_refuses_before_original_sft_create(
    monkeypatch, tmp_path, caplog, logger_failure
):
    from amlas.tui.chat import tool_binding_backend as binding
    from amlas.tui.chat.research_input_recovery import _CANCELLED

    expected = "AMLAS_INPUT_HOLD phase=sft_capture reason=callback_refused"
    warning_attempts = []
    diagnostic_attempts = []
    log_order = []
    fixture_states = []
    captured = []
    run_ids = []
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    if logger_failure:
        original_warning = binding.logger.warning

        def refuse_warning(message, *args, **kwargs):
            if message == expected:
                log_order.append("generic")
                if logger_failure is True:
                    warning_attempts.append((message, args, kwargs))
                    raise RuntimeError("ordinary diagnostic logger unavailable")
            if message == "normal_sft_capture_unavailable run_id=%s phase=%s error_class=%s":
                assert log_order == ["generic"]
                assert fixture_states[0].backend._input_hold is True
                assert captured[0][0].cancel_tool == _CANCELLED
                diagnostic_attempts.append(args)
                if logger_failure == "interrupt":
                    raise KeyboardInterrupt()
                if logger_failure == "diagnostic":
                    raise RuntimeError("ordinary diagnostic logger unavailable")
            return original_warning(message, *args, **kwargs)

        monkeypatch.setattr(binding.logger, "warning", refuse_warning)

    async def scenario():
        from strands import Agent

        from amlas.control_plane import normal_run_checkpoint as checkpoint

        state = setup(monkeypatch, tmp_path)
        fixture_states.append(state)
        actual_snapshot = Agent.take_snapshot

        def record_form(snapshot):
            # The product budgets the record form of the SDK snapshot; the search measures the same bytes.
            copied = SimpleNamespace(to_dict=lambda: copy.deepcopy(snapshot.to_dict()))
            return binding._record_session_snapshot(SimpleNamespace(take_snapshot=lambda preset: copied))

        def whole_task_overflow_snapshot(agent, *args, **kwargs):
            context = checkpoint.current_normal_run()
            uses = [
                block.get("toolUse")
                for message in agent.messages[-1:]
                for block in message.get("content", [])
                if type(block) is dict and type(block.get("toolUse")) is dict
            ]
            if context is None or not any(use.get("toolUseId") == SFT_USE for use in uses):
                return actual_snapshot(agent, *args, **kwargs)
            call = checkpoint._causal_call(SFT_USE, binding._record_messages(agent.messages))
            low, high = 0, MAX_TASK_BYTES
            # A fixed finite search keeps the genuine schema2 TASK boundary exact.
            for _ in range(MAX_TASK_BYTES.bit_length()):
                if low >= high:
                    break
                middle = (low + high + 1) // 2
                agent.state.set("ordinary_sft_budget_note", "n" * middle)
                snapshot = actual_snapshot(agent, *args, **kwargs)
                try:
                    checkpoint._budget_sft_snapshot(context, call, record_form(snapshot))
                except ValueError as error:
                    if str(error) != "normal run DATA size exceeded":
                        raise
                    high = middle - 1
                else:
                    low = middle
            assert low == high
            # Exceed the TASK budget with rendering headroom, while retaining an
            # individually valid public SDK snapshot and unchanged causal call.
            agent.state.set("ordinary_sft_budget_note", "n" * (low + 64))
            snapshot = actual_snapshot(agent, *args, **kwargs)
            assert len(canonical(call)) <= MAX_TASK_BYTES
            assert len(canonical(record_form(snapshot))) <= MAX_TASK_BYTES
            with pytest.raises(ValueError, match=r"^normal run DATA size exceeded$"):
                checkpoint._budget_sft_snapshot(context, call, record_form(snapshot))
            return snapshot

        monkeypatch.setattr(Agent, "take_snapshot", whole_task_overflow_snapshot)
        history = [
            {"role": "user", "content": [{"text": "Ordinary prior research note. " + "n" * (42 * 1024)}]},
            {
                "role": "assistant",
                "content": [{"text": "Ordinary prior research acknowledgement. " + "a" * (42 * 1024)}],
            },
        ]
        assert all(len(canonical(message)) < 64 * 1024 for message in history)
        admitted, observed = [], []
        original_admit = ToolBindingBackend._admit_input
        original_capture = ToolBindingBackend._capture_normal_sft_call

        def admit(backend, text, phase, **kwargs):
            value = original_admit(backend, text, phase, **kwargs)
            if phase == "HISTORY_TEXT":
                admitted.append(value)
            return value

        def capture(
            backend, event, *, _diagnostic_phase=None, _diagnostic_predicate=None, _diagnostic_size=None
        ):
            if event.tool_use.get("name") == "submit_qwen_sft":
                from amlas.control_plane.normal_run_checkpoint import current_normal_run

                context = current_normal_run()
                assert context is not None
                run_ids.append(context.run_id)
                snapshot = binding._record_session_snapshot(event.agent)
                prefix = copy.deepcopy(binding._record_messages(event.agent.messages))
                assert snapshot["data"]["messages"] == prefix
                assert payloads(prefix)["cold-original-preparation"]
                assert payloads(prefix)["cold-original-model"]
                assert len(canonical(_causal_call(SFT_USE, prefix))) <= MAX_TASK_BYTES
                assert len(canonical(snapshot)) <= MAX_TASK_BYTES
                # Compact schema2 retains history once. The actual whole TASK
                # budget must fail without invalidating either individual input.
                with pytest.raises(ValueError, match=r"^normal run DATA size exceeded$"):
                    checkpoint._budget_sft_snapshot(
                        context, checkpoint._causal_call(SFT_USE, prefix), snapshot
                    )
                observed.append(True)
                captured.append((event, copy.deepcopy(prefix), set(backend._warm_consumed)))
            options = {}
            if _diagnostic_phase is not None:
                options["_diagnostic_phase"] = _diagnostic_phase
            if _diagnostic_predicate is not None:
                options["_diagnostic_predicate"] = _diagnostic_predicate
            if _diagnostic_size is not None:
                options["_diagnostic_size"] = _diagnostic_size
            return original_capture(backend, event, **options)

        monkeypatch.setattr(ChatEngine, "_build_agent_history", lambda _engine: copy.deepcopy(history))
        monkeypatch.setattr(ToolBindingBackend, "_admit_input", admit)
        monkeypatch.setattr(ToolBindingBackend, "_capture_normal_sft_call", capture)
        try:
            await reach_boundary(state, monkeypatch, observed)
            assert len(admitted) == 2 and all(type(value) is str for value in admitted)
            assert state.native.calls == [], (
                "oversized full SDK TASK was discovered only after native SFT Create"
            )
            assert len(captured) == 1
            event, prefix, consumed = captured[0]
            assert event.cancel_tool == _CANCELLED
            assert state.backend._input_hold is True
            assert binding._record_messages(event.agent.messages)[: len(prefix)] == prefix
            assert state.backend._warm_consumed == consumed
            records = [
                record
                for record in caplog.records
                if record.msg == ("normal_sft_capture_unavailable run_id=%s phase=%s error_class=%s")
            ]
            assert len(run_ids) == 1
            if logger_failure in {"diagnostic", "interrupt"}:
                assert diagnostic_attempts == [(run_ids[0], "snapshot_budget", "ValueError")]
                assert records == []
            else:
                assert len(records) == 1, "missing attributed SFT capture diagnostic"
                assert records[0].args == (run_ids[0], "snapshot_budget", "ValueError")
                assert records[0].exc_info is None and records[0].stack_info is None
            if logger_failure is True:
                assert warning_attempts == [(expected, (), {})], "sft_capture_warning_attempt_missing"
            else:
                warnings = [
                    record
                    for record in caplog.records
                    if record.name == binding.__name__
                    and record.getMessage().startswith("AMLAS_INPUT_HOLD phase=sft_capture ")
                ]
                assert [record.getMessage() for record in warnings] == [expected], (
                    "sft_capture_first_hold_marker_missing"
                )
                assert all(
                    record.args == () and record.exc_info is None and record.stack_info is None
                    for record in warnings
                )
        finally:
            await close(state)

    asyncio.run(scenario())


def test_public_rejected_recipe_logs_closed_predicate_after_hold(monkeypatch, tmp_path):
    from normal_sft_cold_fixture import FullGraphModel

    from amlas.tui.chat import tool_binding_backend as binding
    from amlas.tui.chat.research_input_recovery import _CANCELLED

    original_stream = FullGraphModel.stream
    original_capture = ToolBindingBackend._capture_normal_sft_call
    original_warning = binding.logger.warning
    captured, order, observed, states = [], [], [], []

    async def rejected_recipe_stream(model, messages, tool_specs, system_prompt, **kwargs):
        index = None
        async for event in original_stream(model, messages, tool_specs, system_prompt, **kwargs):
            start = event.get("contentBlockStart", {})
            if start.get("start", {}).get("toolUse", {}).get("name") == "submit_qwen_sft":
                index = start["contentBlockIndex"]
            delta = event.get("contentBlockDelta", {})
            if (
                index is not None
                and delta.get("contentBlockIndex") == index
                and "toolUse" in delta.get("delta", {})
            ):
                event = copy.deepcopy(event)
                part = event["contentBlockDelta"]["delta"]["toolUse"]
                inputs = json.loads(part["input"])
                inputs["plan_only"] = False
                inputs["recipe_id"] = "declared_invalid_recipe"
                part["input"] = json.dumps(inputs)
            yield event

    def capture(backend, event, *, _diagnostic_phase=None, _diagnostic_predicate=None, _diagnostic_size=None):
        if event.tool_use.get("name") == "submit_qwen_sft":
            prefix = copy.deepcopy(event.agent.messages)
            snapshot = event.agent.take_snapshot(preset="session").to_dict()
            assert snapshot["data"]["messages"] == prefix
            captured.append((event, prefix, snapshot, len(states[0].native.calls)))
            observed.append(True)
        options = {}
        if _diagnostic_phase is not None:
            options["_diagnostic_phase"] = _diagnostic_phase
        if _diagnostic_predicate is not None:
            options["_diagnostic_predicate"] = _diagnostic_predicate
        if _diagnostic_size is not None:
            options["_diagnostic_size"] = _diagnostic_size
        return original_capture(backend, event, **options)

    def warning(message, *args, **kwargs):
        watched = {
            "AMLAS_INPUT_HOLD phase=sft_capture reason=callback_refused": "generic",
            "normal_sft_capture_unavailable run_id=%s phase=%s error_class=%s": "phase",
            "normal_sft_input_predicate run_id=%s check=%s": "predicate",
        }
        if message in watched:
            assert states[0].backend._input_hold is True
            assert captured[0][0].cancel_tool == _CANCELLED
            order.append((watched[message], args))
        return original_warning(message, *args, **kwargs)

    async def scenario():
        monkeypatch.setattr(FullGraphModel, "stream", rejected_recipe_stream)
        state = setup(monkeypatch, tmp_path)
        states.append(state)
        monkeypatch.setattr(ToolBindingBackend, "_capture_normal_sft_call", capture)
        monkeypatch.setattr(binding.logger, "warning", warning)
        try:
            await reach_boundary(state, monkeypatch, observed)
            assert len(captured) == 1
            event, prefix, snapshot, creates_before = captured[0]
            assert event.tool_use["input"].get("plan_only") is False
            assert event.tool_use["input"]["recipe_id"] == "declared_invalid_recipe"
            assert len(state.native.calls) == creates_before
            assert state.backend._input_hold is True and event.cancel_tool == _CANCELLED
            assert snapshot["data"]["messages"] == prefix
            assert event.agent.messages[: len(prefix)] == prefix
            assert [entry[0] for entry in order] == ["generic", "phase", "predicate"]
            assert order[1][1][1:] == ("causal_sft_inputs", "ValueError")
            assert order[2][1][1:] == ("RECIPE",)
            assert order[2][1][0] == order[1][1][0]
        finally:
            await close(state)

    asyncio.run(scenario())
