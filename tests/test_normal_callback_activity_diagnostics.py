"""Closed callback activity; actual SDK boundaries, inert ports, no native proof."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from types import SimpleNamespace

import pytest
from strands import tool
from strands.hooks.events import AfterModelCallEvent
from test_normal_model_progress_diagnostics import bind_normal_fixture
from test_normal_model_progress_diagnostics import current_context as _current_context
from test_normal_model_progress_diagnostics import normal_work as _normal_work
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_precommit_original_completion import (
    ThreeOriginals,
    captured_originals,
    protocol_work,
    publish_original_data,
    results,
)
from test_runtime_warm_model_continuation import complete_fixture
from test_sdk_prompt_policy import ScriptedModel

from amlas import app as runtime
from amlas.config.settings import Settings
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
normal_work = _normal_work
current_context = _current_context
# Keep the actual builder before the shared normal fixture attaches its observer.
SDK_BUILD = binding.ToolBindingBackend.build_agent
MESSAGE = "normal_callback_activity run_id=%s phase=%s tool_category=%s"
PRIVATE = "ordinary withheld callback fixture detail"
PHASES = (
    "before_model_enter",
    "before_model_exit",
    "after_model_enter",
    "before_tool_enter",
    "before_tool_exit",
    "after_tool_enter",
    "after_tool_exit",
)
CATEGORIES = ("NONE", "RESEARCH", "MODEL", "DATA", "CASE", "PREP", "SFT", "RELOAD", "GEN", "PAIR", "OTHER")


def records(caplog):
    return [record for record in caplog.records if record.msg == MESSAGE]


def warning():
    helper = getattr(binding, "_warn_normal_callback_activity", None)
    assert callable(helper), "normal callbacks have no optional closed activity diagnostic"
    return helper


@pytest.mark.parametrize("phase", PHASES)
def test_closed_warning_has_current_run_and_no_payload(current_context, caplog, phase):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    warning()(phase, tool_category="NONE")
    found = records(caplog)
    assert len(found) == 1
    assert found[0].args == (current_context.run_id, phase, "NONE")
    assert found[0].levelno == logging.WARNING
    assert found[0].exc_info is None and found[0].stack_info is None
    assert found[0].getMessage() == (
        f"normal_callback_activity run_id={current_context.run_id} phase={phase} tool_category=NONE"
    )


@pytest.mark.parametrize("category", CATEGORIES)
def test_only_closed_tool_categories_are_rendered(current_context, caplog, category):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    warning()("before_tool_enter", tool_category=category)
    assert [record.args for record in records(caplog)] == [
        (current_context.run_id, "before_tool_enter", category)
    ]


@pytest.mark.parametrize(
    ("phase", "category"),
    [(PRIVATE, "NONE"), (None, "NONE"), ("before_tool_enter", PRIVATE), ("before_tool_enter", {})],
)
def test_invalid_diagnostic_labels_are_silent(current_context, caplog, phase, category):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    warning()(phase, tool_category=category)
    assert not records(caplog) and PRIVATE not in caplog.text


@pytest.mark.parametrize("mode", ["absent", "changed-scope", "foreign-pid"])
def test_default_or_stale_context_cannot_publish(monkeypatch, current_context, caplog, mode):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    helper = warning()
    if mode == "absent":
        with checkpoint.activate_normal_run(None):
            helper("before_model_enter", tool_category="NONE")
    elif mode == "changed-scope":
        monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", "different-neutral-scope")
        helper("before_model_enter", tool_category="NONE")
    else:
        original = current_context.process_id
        object.__setattr__(current_context, "process_id", original + 1)
        try:
            helper("before_model_enter", tool_category="NONE")
        finally:
            object.__setattr__(current_context, "process_id", original)
    assert not records(caplog)


def test_ordinary_logger_fault_is_optional(monkeypatch, current_context):
    attempted = []

    def refuse(*args, **kwargs):
        attempted.append(True)
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(binding.logger, "warning", refuse)
    assert warning()("before_model_enter", tool_category="NONE") is None
    assert attempted == [True]


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_logger_process_control_propagates_exact_object(monkeypatch, current_context, control_type):
    control = control_type(PRIVATE)

    def stop(*args, **kwargs):
        raise control

    monkeypatch.setattr(binding.logger, "warning", stop)
    with pytest.raises(control_type) as observed:
        warning()("before_model_enter", tool_category="NONE")
    assert observed.value is control


def test_copied_concurrent_contexts_keep_their_own_correlation(current_context, caplog):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    helper = warning()
    contexts = [checkpoint.NormalRunContext(current_context.catalog, value * 32, "b" * 64) for value in "cd"]
    barrier = threading.Barrier(2)
    copies = []
    for context in contexts:
        with checkpoint.activate_normal_run(context):
            copies.append(contextvars.copy_context())

    def publish():
        barrier.wait(timeout=10)
        helper("before_model_enter", tool_category="NONE")

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(context.run, publish) for context in copies]
        for future in pending:
            future.result(timeout=15)
    assert sorted(record.args for record in records(caplog)) == [
        (value * 32, "before_model_enter", "NONE") for value in "cd"
    ]
    assert checkpoint.current_normal_run() is current_context


class Gate:
    """A finite test-owned pause at one actual callback/model/tool boundary."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def wait(self):
        self.entered.set()
        assert self.release.wait(15), "test-owned activity boundary was not released"


def sdk_work(monkeypatch, *, model_gate=None, tool_gate=None):
    effects = []

    @tool
    def ordinary_activity_tool(marker: str) -> str:
        """Return inert neutral DATA without a producer or native effect."""
        effects.append(marker)
        if tool_gate is not None:
            tool_gate.wait()
        return PRIVATE

    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [ordinary_activity_tool])
    model = ScriptedModel(
        [[{"id": "ordinary-use", "name": "ordinary_activity_tool", "input": {"marker": PRIVATE}}]],
        on_first_cycle=model_gate.wait if model_gate is not None else None,
    )
    backend = binding.ToolBindingBackend(Settings(), model=model)
    assert backend.is_tool_capable, "activity coverage requires the installed SDK, never a shim"
    agent = SDK_BUILD(backend)
    agent.callback_handler = lambda **event: None
    return SimpleNamespace(backend=backend, agent=agent, model=model, effects=effects)


def invoke(work):
    return work.agent("ordinary callback activity question", invocation_state={})


@pytest.mark.parametrize(
    ("boundary", "last_phase", "effects_before_release"),
    [
        ("qualification", "before_model_enter", 0),
        ("model", "before_model_exit", 0),
        ("tool_qualification", "before_tool_enter", 0),
        ("tool", "before_tool_exit", 1),
        ("post_tool", "after_tool_enter", 1),
    ],
)
def test_actual_sdk_distinguishes_callback_pause_from_model_and_tool(
    monkeypatch, current_context, caplog, record_property, boundary, last_phase, effects_before_release
):
    record_property("strands_agents_version", version("strands-agents"))
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    gate = Gate()
    work = sdk_work(
        monkeypatch,
        model_gate=gate if boundary == "model" else None,
        tool_gate=gate if boundary == "tool" else None,
    )
    if boundary in {"qualification", "tool_qualification", "post_tool"}:
        name = {
            "qualification": "_qualify_research_binding",
            "tool_qualification": "_before_research_tool",
            "post_tool": "_guard_new_tool_projection",
        }[boundary]
        original = getattr(work.backend, name)
        entered = []

        def pause(*args, **kwargs):
            if not entered:
                entered.append(True)
                gate.wait()
            return original(*args, **kwargs)

        monkeypatch.setattr(work.backend, name, pause)
    copied = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(copied.run, invoke, work)
        try:
            assert gate.entered.wait(10), "actual SDK did not reach the selected inert boundary"
            assert not future.done()
            found = records(caplog)
            assert found, "live SDK callback/model/tool wait has no activity warning"
            assert found[-1].args[1] == last_phase
            assert len(work.effects) == effects_before_release
        finally:
            gate.release.set()
            result = future.result(timeout=20)
    assert result.message["role"] == "assistant"
    assert result.message["content"] == [{"text": "done"}]
    assert work.effects == [PRIVATE] and not work.backend._input_hold
    phases = [record.args[1] for record in records(caplog)]
    assert phases == [
        "before_model_enter",
        "before_model_exit",
        "after_model_enter",
        "before_tool_enter",
        "before_tool_exit",
        "after_tool_enter",
        "after_tool_exit",
        "before_model_enter",
        "before_model_exit",
        "after_model_enter",
    ]
    assert all(record.args[0] == current_context.run_id for record in records(caplog))
    assert all(record.args[2] in {"NONE", "OTHER"} for record in records(caplog))
    assert PRIVATE not in "\n".join(record.getMessage() for record in records(caplog))
    assert "ordinary_activity_tool" not in "\n".join(record.getMessage() for record in records(caplog))
    snapshot = work.agent.take_snapshot(preset="session").to_dict()
    assert snapshot["data"]["messages"] == work.agent.messages
    assert set(snapshot) == {"scope", "schema_version", "created_at", "data", "app_data"}
    assert "normal_callback_activity" not in repr(snapshot)


@pytest.mark.parametrize("position", ["registered_before", "registered_after"])
def test_actual_aftermodel_reverse_order_cannot_invent_callback_exit(
    monkeypatch, current_context, caplog, position
):
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    from amlas import _build

    observed = []

    def independent(event):
        observed.append((type(event), len(records(caplog))))

    # Preserve actual SDK registration and reverse delivery, without replacing
    # its dispatcher or mutating its private callback registry.
    actual_factory = _build.build_agent

    def factory(**kwargs):
        agent = actual_factory(**kwargs)
        agent.hooks.add_callback(AfterModelCallEvent, independent)
        return agent

    if position == "registered_before":
        monkeypatch.setattr(_build, "build_agent", factory)
    work = sdk_work(monkeypatch)
    if position == "registered_after":
        work.agent.hooks.add_callback(AfterModelCallEvent, independent)
    invoke(work)
    assert len(observed) == 2
    assert sum(record.args[1] == "after_model_enter" for record in records(caplog)) == 2
    assert all(record.args[1] != "after_model_exit" for record in records(caplog))


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_original_qualification_control_is_not_masked_by_exit_logging(
    monkeypatch, current_context, control_type
):
    work = sdk_work(monkeypatch)
    control = control_type(PRIVATE)
    warnings = []

    def original_failure(*args, **kwargs):
        raise control

    def warn(message, *args, **kwargs):
        if message == MESSAGE:
            warnings.append(args)
            if args[1].endswith("exit"):
                raise RuntimeError("ordinary cleanup logger failure")

    monkeypatch.setattr(work.backend, "_qualify_research_binding", original_failure)
    monkeypatch.setattr(binding.logger, "warning", warn)
    with pytest.raises(control_type) as observed:
        invoke(work)
    assert observed.value is control and work.effects == []
    assert warnings == [(current_context.run_id, "before_model_enter", "NONE")]


@pytest.mark.parametrize("mode", ["silent-default", "logger-failure"])
def test_actual_sdk_optional_warning_preserves_guard_history_and_effects(
    monkeypatch, current_context, normal_work, caplog, mode
):
    work_fixture, ports, _model = normal_work
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    work = sdk_work(monkeypatch)
    before = (copy.deepcopy(ports.items), dict(ports.objects), list(ports.calls))
    attempted = []
    if mode == "logger-failure":
        original = binding.logger.warning

        def optional(message, *args, **kwargs):
            if message == MESSAGE:
                attempted.append(True)
                raise RuntimeError(PRIVATE)
            return original(message, *args, **kwargs)

        monkeypatch.setattr(binding.logger, "warning", optional)
        result = invoke(work)
    else:
        with checkpoint.activate_normal_run(None):
            result = invoke(work)
    assert result.message["role"] == "assistant"
    assert result.message["content"] == [{"text": "done"}]
    assert work.effects == [PRIVATE]
    assert not work.backend._input_hold and not records(caplog)
    assert (ports.items, ports.objects, ports.calls) == before
    assert work_fixture.guard.calls and all(call["source"] == "INPUT" for call in work_fixture.guard.calls)
    assert work.agent.messages[-1]["content"] == [{"text": "done"}]
    assert any("toolResult" in block for message in work.agent.messages for block in message["content"])
    if mode == "logger-failure":
        assert attempted, "actual callbacks never attempted their optional diagnostic"


def test_actual_precommit_aftertool_exit_waits_for_original_async_completion(monkeypatch, tmp_path, caplog):
    model = ThreeOriginals()
    work, data, _fetched = protocol_work(monkeypatch, model)
    ports, tokens = bind_normal_fixture(monkeypatch, work)
    caplog.set_level(logging.WARNING, logger=binding.__name__)
    events = []

    async def read():
        async for event in runtime.agent_invocation({"prompt": work.literal}):
            events.append(event)

    async def scenario():
        reader = asyncio.create_task(read())
        try:
            backend = await captured_originals(work, data, reader)
            run_id = next(event["run_id"] for event in events if event["event"] == "run_started")
            original = backend._warm_pending
            producers = tuple(backend._warm_pending_producers.values())
            assert original is not None and len(producers) == 2 and not reader.done()
            before = copy.deepcopy(work.agents[0].messages)
            current = records(caplog)
            producer = [record.args[1:3] for record in current if record.args[2] in {"MODEL", "DATA"}]
            assert producer.count(("after_tool_enter", "MODEL")) == 1
            assert producer.count(("after_tool_enter", "DATA")) == 2
            assert not any(phase == "after_tool_exit" for phase, _kind in producer)
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=original.record.handle, adapter=work.adapter, native=work.native),
            )
            for pending in producers:
                publish_original_data(pending.record, data)
            await asyncio.wait_for(reader, 10)
            producer = [record.args[1:3] for record in records(caplog) if record.args[2] in {"MODEL", "DATA"}]
            assert producer.count(("after_tool_exit", "MODEL")) == 1
            assert producer.count(("after_tool_exit", "DATA")) == 2
            assert all(record.args[0] == run_id for record in records(caplog))
            assert len(work.native.requests) == 1 and len(data.native.calls) == 2
            assert len(work.agents) == len(work.engines) == 1 and not backend._input_hold
            assert work.agents[0].messages[: len(before)] == before
            assert len(results(work.agents[0])) >= 3
            assert events[-1]["worker_finished"] is True
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    try:
        asyncio.run(scenario())
    finally:
        from bedrock_agentcore.runtime import BedrockAgentCoreContext

        BedrockAgentCoreContext._request_id.reset(tokens[0])
        BedrockAgentCoreContext._session_id.reset(tokens[1])
    assert ports.calls
