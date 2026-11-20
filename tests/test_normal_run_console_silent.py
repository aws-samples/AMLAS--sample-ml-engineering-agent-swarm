"""A prompt run in the runtime writes no agent text to the console.

The agent of a prompt run (the first turn and both restore builders) held the SDK's default console
handler. Every assistant sentence, the final answer and one `Tool #N: name` line per tool call went to
stdout and from there into the runtime log. The product stores none of that text except the final report,
so a reader of the log cannot match such a line against any record. On a prompt run the builder assigns
the SDK's own `null_callback_handler`; a build with a thought trace and a build without a run context keep
their handlers.

Real SDK Agent, scripted model, inert native ports. Tests named `red` fail on the earlier code. Tests named
`guard` pass before and after the change and pin what stays as it is: the product's own log lines, the
final text, the stored report, the guardrail capture and the interactive output.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from types import SimpleNamespace

import pytest
from strands import Agent, tool
from strands.event_loop import streaming
from strands.handlers.callback_handler import PrintingCallbackHandler, null_callback_handler
from strands.models import Model
from test_normal_checkpoint_cas_contract import _policy
from test_normal_model_progress_diagnostics import current_context as _current_context
from test_normal_model_progress_diagnostics import normal_work as _normal_work
from test_normal_pending_restore_guardrail_off import _task
from test_normal_run_checkpoint import PROMPT, collect
from test_recovery_notice import FakeContext

from amlas.config.settings import Settings
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import recovery_notice, research_runtime_profile
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.guardrail_assessment import GuardrailAssessmentCapture
from amlas.tui.chat.models import ChatResponse

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
normal_work = _normal_work
current_context = _current_context
# The actual builder, kept before the shared normal fixture replaces it with one that sets its own handler.
SDK_BUILD = binding.ToolBindingBackend.build_agent
SAFE_TOOLS = binding.get_safe_llm_tools
TOOL = "ordinary_activity_tool"
BEFORE = "The data is prepared. I will check the record before going further."
BETWEEN = "The first check is complete. Next is the second check."
FINAL = "Both checks are complete."
SENTENCES = (BEFORE, BETWEEN, FINAL)
# The run id and the identifiers of the input admission records are drawn fresh for every run.
PER_RUN_KEYS = {"run_id", "check_id", "keyed_digest", "opaque_cycle"}
RUNTIME_STARTUP_LINE = "TAVILY_SECRET: configured=False newly_applied=False\n"
USAGE = {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, "metrics": {"latencyMs": 0}}


@pytest.fixture(autouse=True)
def scripted_model_has_no_measured_first_byte_time(monkeypatch):
    """Keep the metadata of every assistant message as the scripted model wrote it.

    The SDK reads the wall clock at the first stream event and adds `timeToFirstByteMs` to the message's
    metadata once the time since the model call reaches 1 ms. The scripted model answers at once, so the field
    appeared only when the test process was held up. Two runs of the same script then differed in their
    messages and in the sizes the product logs. The SDK's clock reading in its stream module is zero here,
    and the SDK then adds no such field.
    """
    monkeypatch.setattr(streaming, "time", SimpleNamespace(time=lambda: 0.0))


class NarratingModel(Model):
    """Each step is one model cycle: a list of ``("text", value)`` and ``("tool", use_id, name, arguments)`` blocks."""

    def __init__(self, steps):
        self.steps = [list(step) for step in steps]
        self.calls = 0

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "neutral-narrating-model"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("ordinary installed SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        step = self.steps[self.calls]
        self.calls += 1
        stop = "end_turn"
        yield {"messageStart": {"role": "assistant"}}
        for index, block in enumerate(step):
            if block[0] == "tool":
                _, identifier, name, arguments = block
                stop = "tool_use"
                start = {"toolUse": {"toolUseId": identifier, "name": name}}
                delta = {"toolUse": {"input": json.dumps(arguments)}}
            else:
                start, delta = {}, {"text": block[1]}
            yield {"contentBlockStart": {"contentBlockIndex": index, "start": start}}
            yield {"contentBlockDelta": {"contentBlockIndex": index, "delta": delta}}
            yield {"contentBlockStop": {"contentBlockIndex": index}}
        yield {"messageStop": {"stopReason": stop}}
        yield {"metadata": copy.deepcopy(USAGE)}


def narrated():
    """A sentence before the first tool call, a sentence between the two tool calls, then the final answer."""
    return [
        [("text", BEFORE), ("tool", "ordinary-use-1", TOOL, {"marker": "one"})],
        [("text", BETWEEN), ("tool", "ordinary-use-2", TOOL, {"marker": "two"})],
        [("text", FINAL)],
    ]


def work(monkeypatch, steps=None):
    effects = []

    @tool
    def ordinary_activity_tool(marker: str) -> str:
        """Return inert neutral DATA without a producer or native effect."""
        effects.append(marker)
        return "ordinary inert result " + marker

    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [ordinary_activity_tool])
    model = NarratingModel(narrated() if steps is None else steps)
    backend = binding.ToolBindingBackend(Settings(), model=model)
    assert backend.is_tool_capable, "console coverage requires the installed SDK, never a shim"
    return SimpleNamespace(backend=backend, model=model, effects=effects)


def invoke(agent):
    return agent("ordinary console question", invocation_state={})


def product_lines(caplog):
    """The rendered callback-activity and conversation-size lines of the product's own logger, in order."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == binding.__name__
        and record.getMessage().startswith(("normal_callback_activity ", "normal_conversation_size "))
    ]


def drive_console_events(handler):
    """The four event shapes for which the SDK's console handler prints."""
    handler(data="A text delta of the assistant.")
    handler(data="The last text delta of the assistant.", complete=True)
    handler(reasoningText="A reasoning delta.")
    handler(event={"contentBlockStart": {"start": {"toolUse": {"toolUseId": "ordinary-use", "name": TOOL}}}})


def assert_no_agent_text(out, expected=""):
    assert not [sentence for sentence in SENTENCES if sentence in out], f"agent text reached stdout: {out!r}"
    assert "Tool #" not in out, f"a console tool line reached stdout: {out!r}"
    assert out == expected, f"the prompt-run agent wrote to stdout: {out!r}"


# ---------------------------------------------------------------------------------------------------------
# red: fail on the earlier code
# ---------------------------------------------------------------------------------------------------------


def test_red_prompt_run_builder_gives_a_callable_handler_that_prints_nothing(
    monkeypatch, current_context, capsys
):
    agent = SDK_BUILD(work(monkeypatch).backend)
    handler = agent.callback_handler
    assert callable(handler), "the guardrail capture wraps only a callable handler"
    drive_console_events(handler)
    assert_no_agent_text(capsys.readouterr().out)
    assert handler is null_callback_handler


def test_red_text_between_tool_calls_and_the_final_answer_stay_off_stdout(
    monkeypatch, current_context, caplog, capsys
):
    caplog.set_level(logging.INFO, logger=binding.__name__)
    # Baseline: the same run with the SDK's console handler. It proves the trigger and gives the lines to compare.
    base = work(monkeypatch)
    base_agent = SDK_BUILD(base.backend)
    base_agent.callback_handler = PrintingCallbackHandler()
    base_result = invoke(base_agent)
    base_out = capsys.readouterr().out
    base_lines = product_lines(caplog)
    assert all(sentence in base_out for sentence in SENTENCES), (
        "TRIGGER ABSENT: the SDK printer printed no text"
    )
    assert "Tool #1: " + TOOL in base_out and "Tool #2: " + TOOL in base_out
    caplog.clear()

    subject = work(monkeypatch)
    agent = SDK_BUILD(subject.backend)
    result = invoke(agent)
    assert_no_agent_text(capsys.readouterr().out)
    # What stays: the product's own lines, the tool effects, the conversation and the final text.
    lines = product_lines(caplog)
    assert lines == base_lines
    assert sum(" phase=before_tool_enter " in line for line in lines) == 2
    assert sum(line.startswith("normal_conversation_size ") for line in lines) == 3
    assert all("run_id=" + current_context.run_id + " " in line for line in lines)
    assert subject.effects == base.effects == ["one", "two"] and subject.model.calls == 3
    assert binding._record_messages(agent.messages) == binding._record_messages(base_agent.messages)
    extract = binding.ToolBindingBackend._extract_final_text
    assert extract(result) == extract(base_result) == FINAL


def restore_engine(monkeypatch, w):
    """An engine over the real backend and builder; only the qualification seams are declared."""
    agents = []

    def recording_build(backend, **kwargs):
        agent = SDK_BUILD(backend, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(binding.ToolBindingBackend, "build_agent", recording_build)
    w.backend.approval_policy = _policy()
    monkeypatch.setattr(w.backend, "_activate_research_binding", lambda: None)
    monkeypatch.setattr(w.backend, "_qualify_research_binding", lambda *args, **kwargs: None)
    engine = object.__new__(ChatEngine)
    engine.tool_backend = w.backend
    # The engine owns its backend in this process, as the constructor of a restored engine binds it.
    w.backend._warm_owner, w.backend._warm_pid = engine, os.getpid()
    return engine, agents


def session_snapshot(messages):
    origin = Agent(
        model=NarratingModel([]), tools=[], messages=copy.deepcopy(messages), callback_handler=None
    )
    return binding._record_session_snapshot(origin)


def test_red_pending_restore_builder_gives_a_silent_agent_that_stays_silent_after_load(
    monkeypatch, current_context, capsys
):
    w = work(monkeypatch)
    engine, agents = restore_engine(monkeypatch, w)
    prefix = [
        {"role": "user", "content": [{"text": "Declared admitted goal."}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "declared-tool-use", "name": "submit_qwen_sft", "input": {}}}
            ],
        },
    ]
    task = {**_task(prefix), "sdk_snapshot": session_snapshot(prefix)}
    engine.load_normal_snapshot(task)
    agent = engine._normal_agent
    assert agents == [agent] and agent.messages == prefix
    handler = agent.callback_handler
    assert callable(handler)
    drive_console_events(handler)
    assert_no_agent_text(capsys.readouterr().out)
    assert handler is null_callback_handler


class ClaimContext(FakeContext):
    """The continuation context of the recovery-notice tests plus the two calls of the claim."""

    def claim_completion(self, ready, *, policy, qualify_original):
        return "token"

    def qualify_original_completion(self, task, store, *, policy):
        return None


def test_red_completed_restore_builder_stays_silent_through_the_restored_invocation(
    monkeypatch, current_context, caplog, capsys
):
    caplog.set_level(logging.INFO)
    w = work(
        monkeypatch,
        steps=[[("text", BETWEEN), ("tool", "ordinary-use-2", TOOL, {"marker": "two"})], [("text", FINAL)]],
    )
    engine, agents = restore_engine(monkeypatch, w)
    prefix = [
        {"role": "user", "content": [{"text": "Run the whole workflow."}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "use-1", "name": "submit_qwen_sft", "input": {}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "use-1",
                        "status": "success",
                        "content": [
                            {"text": '{"event_type":"original_sft_completed","status":"verified_completed"}'}
                        ],
                    }
                }
            ],
        },
    ]
    snapshot = session_snapshot(prefix)
    context = ClaimContext({"lock": threading.Lock(), "state": "ready"})
    monkeypatch.setattr(engine, "assemble_normal_completion", lambda task: copy.deepcopy(snapshot))
    monkeypatch.setattr(research_runtime_profile, "_restore_runtime_research_profile", lambda *args: False)
    monkeypatch.setattr(
        engine,
        "_agent_turn_response",
        lambda turn: ChatResponse(message=turn.final_text, action="tool_agent_turn"),
    )
    assert engine.claim_normal_completion(context, {"declared": "task"}) == "token"
    agent = engine._normal_agent
    assert agents == [agent] and agent.messages == prefix
    assert callable(agent.callback_handler)
    response = engine.continue_normal_completion()
    assert_no_agent_text(capsys.readouterr().out)
    assert agent.callback_handler is null_callback_handler, (
        "the composed capture handler was not taken off again"
    )
    # What stays: one restore line, the restored invocation, its tool effect and its final text.
    restored = [
        record.getMessage()
        for record in caplog.records
        if record.name == recovery_notice.__name__
        and record.getMessage().startswith(recovery_notice.RESTORE_MARKER)
    ]
    assert len(restored) == 1 and "run_id=" + context.run_id + " " in restored[0]
    assert w.model.calls == 2 and w.effects == ["two"] and context.row["state"] == "invoking"
    assert response.message.startswith(FINAL)
    assert sum(" phase=before_tool_enter " in line for line in product_lines(caplog)) == 1


def runtime_run(monkeypatch, capsys, caplog, handler_factory):
    """One prompt through the runtime entry point with the real builder; returns what a caller and the log see."""
    model = NarratingModel(narrated())
    agents = []

    def build(backend, **kwargs):
        backend._model = model
        agent = SDK_BUILD(backend, **kwargs)
        if handler_factory is not None:
            agent.callback_handler = handler_factory()
        agents.append(agent)
        return agent

    monkeypatch.setattr(binding.ToolBindingBackend, "build_agent", build)
    caplog.clear()
    capsys.readouterr()
    events = collect({"prompt": PROMPT})
    out = capsys.readouterr().out
    run_id = next(event["run_id"] for event in events if event["event"] == "run_started")
    terminal = events[-1]
    assert terminal["event"] == "run_completed" and terminal["status"] == "response_received", terminal
    stored = collect({"action": "get_prompt_report", "run_id": run_id})
    lines = [line.replace("run_id=" + run_id, "run_id=RUN") for line in product_lines(caplog)]
    assert model.calls == 3 and len(agents) == 1
    return SimpleNamespace(
        out=out, run_id=run_id, terminal=terminal, stored=stored, lines=lines, agent=agents[0]
    )


def differences(left, right, path=""):
    """The paths at which two JSON values differ, apart from the identifiers drawn fresh for every run."""
    if type(left) is dict and type(right) is dict:
        keys = sorted((set(left) | set(right)) - PER_RUN_KEYS)
        return [d for key in keys for d in differences(left.get(key), right.get(key), f"{path}/{key}")]
    if type(left) is list and type(right) is list and len(left) == len(right):
        pairs = enumerate(zip(left, right, strict=True))
        return [d for index, (a, b) in pairs for d in differences(a, b, f"{path}/{index}")]
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


def assert_same_apart_from_run_identifiers(left, right):
    found = differences(left, right)
    assert not found, "\n".join(found)


def test_red_runtime_prompt_run_prints_no_agent_text_and_stores_the_same_report(
    monkeypatch, normal_work, caplog, capsys
):
    caplog.set_level(logging.INFO, logger=binding.__name__)

    @tool
    def ordinary_activity_tool(marker: str) -> str:
        """Return inert neutral DATA without a producer or native effect."""
        return "ordinary inert result " + marker

    # The runtime path qualifies the product's own tool registry, so the inert tool is added to it.
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [*SAFE_TOOLS(), ordinary_activity_tool])
    base = runtime_run(monkeypatch, capsys, caplog, PrintingCallbackHandler)
    assert all(sentence in base.out for sentence in SENTENCES), (
        "TRIGGER ABSENT: the SDK printer printed no text"
    )
    assert "Tool #1: " + TOOL in base.out and "Tool #2: " + TOOL in base.out

    subject = runtime_run(monkeypatch, capsys, caplog, None)
    # The entry point's own startup line is the product's and stays; nothing of the agent follows it.
    assert base.out.startswith(RUNTIME_STARTUP_LINE)
    assert_no_agent_text(subject.out, expected=RUNTIME_STARTUP_LINE)
    # What stays: the answer a caller receives, the stored report and the product's own lines.
    assert FINAL in subject.terminal["response"]["message"]
    assert_same_apart_from_run_identifiers(subject.terminal, base.terminal)
    assert subject.stored and subject.stored[-1]["event"] == "prompt_report", subject.stored
    assert subject.stored[-1]["status"] == "available" and subject.stored[-1]["found"] is True
    assert subject.stored[-1]["report"]["response"] == subject.terminal["response"]
    assert FINAL in subject.stored[-1]["report"]["response"]["message"]
    assert_same_apart_from_run_identifiers(subject.stored, base.stored)
    assert subject.lines == base.lines and len(subject.lines) > 0
    assert sum(" phase=before_tool_enter " in line for line in subject.lines) == 2
    assert binding._record_messages(subject.agent.messages) == binding._record_messages(base.agent.messages)


def helper():
    found = getattr(binding, "_normal_run_console_silent", None)
    assert callable(found), "the builder has no rule that silences the console on a prompt run"
    return found


def test_red_rule_is_false_without_a_run_and_true_with_one(current_context):
    assert helper()() is True
    with checkpoint.activate_normal_run(None):
        assert helper()() is False
    assert helper()() is True


@pytest.mark.parametrize("mode", ["changed-scope", "foreign-pid", "lookup-fault"])
def test_red_failing_run_lookup_still_gives_a_silent_agent(monkeypatch, current_context, capsys, mode):
    w = work(monkeypatch)
    original = current_context.process_id
    expected = ValueError
    with monkeypatch.context() as patch:
        if mode == "changed-scope":
            patch.setenv("AMLAS_DEPLOYMENT_ID", "different-neutral-scope")
        elif mode == "foreign-pid":
            object.__setattr__(current_context, "process_id", original + 1)
        else:
            # Any ordinary fault of the lookup keeps the console silent, whatever its class.
            expected = RuntimeError

            def fault():
                raise RuntimeError("ordinary lookup fault")

            patch.setattr(checkpoint, "current_normal_run", fault)
        try:
            with pytest.raises(expected):
                checkpoint.current_normal_run()
            assert helper()() is True
            agent = SDK_BUILD(w.backend)
        finally:
            object.__setattr__(current_context, "process_id", original)
    assert checkpoint.current_normal_run() is current_context
    assert callable(agent.callback_handler)
    drive_console_events(agent.callback_handler)
    assert_no_agent_text(capsys.readouterr().out)


# ---------------------------------------------------------------------------------------------------------
# guard: pass before and after the change
# ---------------------------------------------------------------------------------------------------------


def test_guard_interactive_build_without_a_run_keeps_the_sdk_console_output(
    monkeypatch, current_context, capsys
):
    w = work(monkeypatch)
    with checkpoint.activate_normal_run(None):
        agent = SDK_BUILD(w.backend)
        assert isinstance(agent.callback_handler, PrintingCallbackHandler)
        result = invoke(agent)
    out = capsys.readouterr().out
    assert all(sentence in out for sentence in SENTENCES)
    assert "Tool #1: " + TOOL in out and "Tool #2: " + TOOL in out
    assert binding.ToolBindingBackend._extract_final_text(result) == FINAL


@pytest.mark.parametrize("run_active", [False, True])
def test_guard_thought_trace_handler_wins_with_and_without_a_run(
    monkeypatch, current_context, capsys, run_active
):
    seen = []

    def trace_callback(**event):
        seen.append(sorted(event))

    traces = []
    monkeypatch.setattr(
        binding, "_make_trace_callback", lambda trace: (traces.append(trace), trace_callback)[1]
    )
    trace = object()
    w = work(monkeypatch)
    if run_active:
        agent = SDK_BUILD(w.backend, thought_trace=trace)
        invoke(agent)
    else:
        with checkpoint.activate_normal_run(None):
            agent = SDK_BUILD(w.backend, thought_trace=trace)
            invoke(agent)
    assert agent.callback_handler is trace_callback and traces == [trace]
    assert seen, "the thought-trace handler received no SDK event"
    assert capsys.readouterr().out == ""


def test_guard_guardrail_capture_wraps_the_prompt_run_handler_and_sees_every_cycle(
    monkeypatch, current_context
):
    agent = SDK_BUILD(work(monkeypatch).backend)
    previous = agent.callback_handler
    assert callable(previous)
    capture = GuardrailAssessmentCapture()
    agent.callback_handler = capture.wrap(previous)
    try:
        invoke(agent)
        # An event that carries a guardrail trace reaches the capture through the composed handler.
        agent.callback_handler(event={"metadata": {"trace": {"guardrail": {}}}})
    finally:
        agent.callback_handler = previous
    assert capture._cycles == 3 and capture._seen is True


def test_guard_handler_is_chosen_at_build_time_and_holds_in_a_thread_without_the_run(
    monkeypatch, current_context, capsys
):
    w = work(monkeypatch)
    agent = SDK_BUILD(w.backend)
    chosen = agent.callback_handler
    seen = []

    def in_plain_thread():
        seen.append(checkpoint.current_normal_run())
        return invoke(agent)

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(in_plain_thread).result(timeout=30)
    assert seen == [None], "the plain thread inherited the run context"
    assert agent.callback_handler is chosen
    assert binding.ToolBindingBackend._extract_final_text(result) == FINAL and w.effects == ["one", "two"]
    # On the earlier code the handler chosen at build time is the SDK printer and this thread prints.
    out = capsys.readouterr().out
    if isinstance(chosen, PrintingCallbackHandler):
        assert all(sentence in out for sentence in SENTENCES)
    else:
        assert out == ""


def test_guard_sdk_contract_the_rule_relies_on(record_property, capsys):
    record_property("strands_agents_version", version("strands-agents"))
    assert null_callback_handler(data="text", complete=True, reasoningText="reasoning") is None
    assert capsys.readouterr().out == ""
    assert (
        Agent(model=NarratingModel([]), tools=[], callback_handler=None).callback_handler
        is null_callback_handler
    )
    default = Agent(model=NarratingModel([]), tools=[])
    assert isinstance(default.callback_handler, PrintingCallbackHandler)
    # The SDK reads `agent.callback_handler` at each invocation, so a handler assigned after construction is used.
    events = []
    default.model = NarratingModel([[("text", FINAL)]])
    default.callback_handler = lambda **event: events.append(sorted(event))
    default("ordinary console question")
    assert events and capsys.readouterr().out == ""


def test_guard_slow_first_stream_event_leaves_the_scripted_metadata_unchanged(monkeypatch, current_context):
    scripted_stream = NarratingModel.stream

    async def slow_first_event(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        # Longer than the 1 ms from which the SDK records a measured time to the first stream event.
        time.sleep(0.005)
        async for event in scripted_stream(self, messages, tool_specs, system_prompt, **kwargs):
            yield event

    monkeypatch.setattr(NarratingModel, "stream", slow_first_event)
    w = work(monkeypatch)
    agent = SDK_BUILD(w.backend)
    invoke(agent)
    answers = [message for message in agent.messages if message["role"] == "assistant"]
    assert len(answers) == w.model.calls == 3
    assert [message["metadata"] for message in answers] == [USAGE] * 3, (
        "the SDK changed the scripted metadata; the run comparisons of this file need equal metadata in both runs"
    )
