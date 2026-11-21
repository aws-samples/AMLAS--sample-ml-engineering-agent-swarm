"""Live SDK data enters the product's records in record form.

Newer strands-agents releases write a random ``tracking_id`` on every message of the live history and add
``model_state`` to the session snapshot. Stored records keep the shape written before those two fields
existed: four snapshot DATA fields and messages without the id. These tests run a real SDK Agent against
a scripted model through the product's capture, witness, restore and size code. No provider, native job
or AWS call happens. They pass on the SDK release that wrote ``sdk_record_form_golden.json`` and on the
locked one; the version-dependent expectations name the first release that carries each field.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import itertools
import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
from sdk_scripted_model import USAGE, ScriptedModel
from strands import Agent, Snapshot, tool
from strands.event_loop import streaming
from strands.hooks.events import (
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)
from test_normal_checkpoint_cas_contract import _policy
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas import app as runtime
from amlas.config.settings import Settings
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytest_plugins = ("test_normal_run_checkpoint",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


@pytest.fixture(autouse=True)
def scripted_model_has_no_measured_first_byte_time(monkeypatch):
    """Keep the metadata of every assistant message as the scripted model wrote it.

    The SDK adds a measured `timeToFirstByteMs` to a message once its first stream event comes 1 ms or more
    after the model call, which happens only when the test process is held up. The SDK's clock reading in
    its stream module is zero here, and the SDK then adds no such field.
    """
    monkeypatch.setattr(streaming, "time", SimpleNamespace(time=lambda: 0.0))


SDK_VERSION = tuple(int(part) for part in version("strands-agents").split(".")[:2])
MODEL_STATE_SINCE = (1, 43)  # first release whose session preset carries ``model_state``
TRACKING_ID_SINCE = (1, 47)  # first release that writes ``tracking_id`` on live messages
ENVELOPE_FIELDS = {"scope", "schema_version", "created_at", "data", "app_data"}
RECORD_FIELDS = {"messages", "state", "conversation_manager_state", "interrupt_state"}
MESSAGE_KEYS = {"role", "content", "metadata", "tracking_id"}
DECISION = (
    "the SDK writes a field that no record form knows; a person decides whether records keep, drop or "
    "refuse it before this expectation changes"
)
INVENTORY_REFUSAL = "public SDK session snapshot DATA inventory differs"
SIZE_REFUSAL = "normal run DATA size exceeded"
LOGGER = "amlas.tui.chat.tool_binding_backend"
PREFIX_REFUSED = "AMLAS_INPUT_HOLD phase=history_qualification reason=history_prefix_refused"
BATCH_REFUSED = "AMLAS_INPUT_HOLD phase=history_qualification reason=history_batch_refused"
SIZE_LINE = re.compile(
    r"^normal_conversation_size run_id=([0-9a-f]{32}) model_call=([0-9]+) bytes=([0-9]+) "
    r"messages=([0-9]+) limit=([0-9]+)$"
)
RUN_ID = "ce" * 16
SFT_USE = "ordinary-use"
SFT_INPUT = {"model_request_ref": "model_acq_" + "1" * 32, "preparation_request_ref": "prep_" + "2" * 32}
PROMPT = "Declared scripted goal."
FINAL_TEXT = "Declared final answer."
FIRST_BYTE_MS = 125
MEASURED = {"usage": USAGE["usage"], "metrics": {**USAGE["metrics"], "timeToFirstByteMs": FIRST_BYTE_MS}}
GOLDEN = Path(__file__).with_name("sdk_record_form_golden.json")
# sha256 of the canonical stored snapshot inside the golden TASK, as the earlier SDK release wrote it.
GOLDEN_SNAPSHOT_SHA256 = (
    "8a434763cdda8c87feb3b590e439d1e6d0e140351f037f9b07fcae18275eb7e0"  # pragma: allowlist secret
)
UNSET = object()
RESTORED_HISTORY = [
    {"role": "user", "content": [{"text": "Declared earlier prompt."}]},
    {
        "role": "assistant",
        "content": [{"text": "Declared earlier answer."}],
        "metadata": {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 0},
        },
    },
]


@tool(name="submit_qwen_sft")
def scripted_sft(model_request_ref: str, preparation_request_ref: str) -> str:
    """Inert stand-in that carries the fine-tune tool's name; no native effect."""
    return "accepted"


@tool
def echo_step(text: str) -> str:
    """Inert tool for the witness tests."""
    return "echo " + text


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def without_ids(messages):
    """This module's own statement of record form; the product helper is never used to check itself."""
    return [{key: value for key, value in message.items() if key != "tracking_id"} for message in messages]


def blank_created_at(snapshot):
    return canonical({**snapshot, "created_at": ""})


def hold_lines(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER and record.getMessage().startswith("AMLAS_INPUT_HOLD ")
    ]


def invoke(agent, prompt=None):
    """One invocation; a refusal raised inside an SDK callback is returned, never raised."""
    try:
        return agent(prompt), None
    except Exception as error:
        return None, error


@pytest.fixture
def record_catalog(monkeypatch, normal_work):
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


def run_context(label):
    transport, session_hash = runtime._prompt_session_identity()
    return checkpoint.prepare_normal_run(label * 32, session_hash, transport_session_id=transport)


def sft_turn(monkeypatch, context, prompt=PROMPT, *, model_state=UNSET):
    """One real SDK turn that calls the fine-tune tool; the product capture runs in the tool hook.

    ``seen.raw`` is the session snapshot as the SDK returned it, ``seen.live`` the live history at the
    capture. ``model_state`` replaces that field of the raw snapshot before the product reads it.
    """
    backend = object.__new__(ToolBindingBackend)
    monkeypatch.setattr(backend, "_qualify_pending_pause", lambda witness: None)
    model = ScriptedModel([("tool", SFT_USE, "submit_qwen_sft", SFT_INPUT), ("text", FINAL_TEXT)])
    agent = Agent(model=model, tools=[scripted_sft], callback_handler=None)
    seen = SimpleNamespace(error=None, phases=[], live=None, raw=None, presets=[])
    take_snapshot = agent.take_snapshot

    def observed_snapshot(*args, **kwargs):
        seen.presets.append((args, kwargs))
        snapshot = take_snapshot(*args, **kwargs)

        def to_dict():
            value = snapshot.to_dict()
            if model_state is not UNSET:
                value["data"]["model_state"] = copy.deepcopy(model_state)
            seen.raw = copy.deepcopy(value)
            return value

        return SimpleNamespace(to_dict=to_dict)

    monkeypatch.setattr(agent, "take_snapshot", observed_snapshot)

    def capture(event):
        messages = event.agent.messages
        seen.live = copy.deepcopy(messages)
        backend._pending_batch_pause = binding._PendingBatchPause(
            agent=event.agent,
            invocation_state=event.invocation_state,
            request_state={},
            research=None,
            checker=None,
            pid=os.getpid(),
            prefix=backend._pause_history(messages[:-1]),
            batch=backend._pause_history(messages[-1]),
            precommit=True,
        )
        try:
            with checkpoint.activate_normal_run(context):
                backend._capture_normal_sft_call(event, _diagnostic_phase=seen.phases.append)
        except Exception as error:
            seen.error = error

    agent.hooks.add_callback(BeforeToolCallEvent, capture)
    seen.result = agent(prompt)
    assert seen.live is not None and model.calls == 2, "the scripted turn never reached the fine-tune tool"
    return agent, seen


def captured_record(context, seen):
    with checkpoint.activate_normal_run(context):
        return context._captured_sft_snapshot(SFT_USE, without_ids(seen.live))


def witness_backend(monkeypatch, registry=()):
    """The product witness on a bare backend: its owner checks run, native originals are absent."""
    backend = object.__new__(ToolBindingBackend)
    backend._research_binding = {"original_completion_ready": False}
    backend._pending_batch_pause = None
    backend._input_hold = False
    backend._input_lock = threading.RLock()
    backend._warm_capture_lock = threading.RLock()
    backend._warm_closed = None
    backend._warm_pending = None
    backend._warm_pending_producers = {}
    backend._precommit_unresolved = {}
    backend._tool_registry_cache = {name: object() for name in registry}
    monkeypatch.setattr(backend, "_pause_profile_current", lambda: backend._research_binding)
    monkeypatch.setattr(backend, "_selected_input_checker", lambda: None)
    return backend


def witnessed_agent(backend, script, *, tools=(), messages=None):
    """A real SDK Agent with the two witness callbacks the product registers at build."""
    model = ScriptedModel(script)
    agent = Agent(model=model, tools=list(tools), messages=messages, callback_handler=None)
    agent.hooks.add_callback(BeforeInvocationEvent, backend._start_pending_pause)
    agent.hooks.add_callback(MessageAddedEvent, backend._commit_pending_pause)
    return agent, model


# ── D1/D4: the two helpers ──


def test_record_messages_copies_each_message_without_its_top_level_id_only():
    live = [
        {"role": "user", "content": [{"text": "t", "tracking_id": "nested"}], "tracking_id": "id-1"},
        {
            "role": "assistant",
            "content": [{"text": "a"}],
            "metadata": {"usage": {"inputTokens": 1}, "tracking_id": "inner"},
            "tracking_id": "id-2",
        },
        {"role": "user", "content": [{"text": "written before the id existed"}]},
    ]
    kept = copy.deepcopy(live)
    record = binding._record_messages(live)
    assert record == without_ids(kept)
    assert live == kept, "the live history was edited"
    assert record is not live and all(a is not b for a, b in zip(record[:2], live[:2], strict=True))
    assert record[0]["content"][0]["tracking_id"] == "nested"
    assert record[1]["metadata"] == {"usage": {"inputTokens": 1}, "tracking_id": "inner"}


def test_witness_text_drops_the_id_of_a_message_and_keeps_a_tool_result_exact():
    message = {"role": "user", "content": [{"text": "t"}], "tracking_id": "id-1"}
    plain = {"role": "user", "content": [{"text": "t"}]}
    history = ToolBindingBackend._pause_history
    assert history(message) == history(plain) and history([message, plain]) == history([plain, plain])
    result = {"toolUseId": "use-1", "status": "success", "content": [{"text": "t"}]}
    exact = ToolBindingBackend._pause_result
    assert exact(result) == json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    assert exact({**result, "tracking_id": "id-1"}) != exact(result)


# ── R1: capture inventory ──


def test_r1_capture_after_a_real_sdk_turn_stores_the_four_field_inventory(record_catalog, monkeypatch):
    context = run_context("a")
    agent, seen = sft_turn(monkeypatch, context)
    if SDK_VERSION >= MODEL_STATE_SINCE:
        assert seen.raw["data"]["model_state"] == {}, "TRIGGER ABSENT: no model_state in the raw snapshot"
    if SDK_VERSION >= TRACKING_ID_SINCE:
        assert all("tracking_id" in message for message in seen.live), "TRIGGER ABSENT: no live ids"
    assert seen.error is None, "the product capture refused a real SDK turn: " + repr(seen.error)
    assert seen.presets == [((), {"preset": "session"})]
    captured = captured_record(context, seen)
    assert set(captured) == ENVELOPE_FIELDS
    assert set(captured["data"]) == RECORD_FIELDS
    assert all("tracking_id" not in message for message in captured["data"]["messages"])
    assert captured["data"]["messages"] == without_ids(seen.live)
    assert [message["role"] for message in captured["data"]["messages"]] == ["user", "assistant"]
    assert agent.messages[:2] == seen.live, "the capture edited the live history"


# ── R2: model_state ──


@pytest.mark.parametrize("state", [{"response_id": "x"}, None, [], "", 0])
def test_r2_any_provider_model_state_is_refused_at_capture(record_catalog, monkeypatch, state):
    context = run_context("b")
    _agent, seen = sft_turn(monkeypatch, context, model_state=state)
    assert type(seen.error) is ValueError and str(seen.error) == INVENTORY_REFUSAL
    assert seen.phases[-1] == "snapshot_create", "the refusal belongs to the snapshot boundary"
    assert context._causal_call is None and context._sft_snapshot is None


def test_r2_an_empty_model_state_is_dropped_and_never_stored(record_catalog, monkeypatch):
    context = run_context("c")
    _agent, seen = sft_turn(monkeypatch, context, model_state={})
    assert seen.raw["data"]["model_state"] == {}
    assert seen.error is None, "an empty model_state was refused: " + repr(seen.error)
    assert set(captured_record(context, seen)["data"]) == RECORD_FIELDS
    assert "model_state" not in context._sft_snapshot


# ── R3: the witness across real SDK appends ──


@pytest.mark.parametrize("restored", [False, True])
def test_r3_real_sdk_appends_do_not_hold(monkeypatch, caplog, restored):
    caplog.set_level(logging.INFO, logger=LOGGER)
    backend = witness_backend(monkeypatch)
    history = copy.deepcopy(RESTORED_HISTORY) if restored else None
    agent, model = witnessed_agent(
        backend, [("text", "first answer"), ("text", "second answer")], messages=history
    )
    for prompt in ("first prompt", "second prompt"):
        result, error = invoke(agent, prompt)
        assert hold_lines(caplog) == [] and error is None and backend._input_hold is False
        assert result.stop_reason == "end_turn"
    assert model.calls == 2 and len(agent.messages) == (6 if restored else 4)
    if SDK_VERSION >= TRACKING_ID_SINCE:
        assert all("tracking_id" in message for message in agent.messages), "TRIGGER ABSENT: no live ids"
    if restored:
        assert without_ids(agent.messages)[:2] == RESTORED_HISTORY


def _set_id(messages):
    messages[0]["tracking_id"] = "another-id"


def _drop_ids(messages):
    for message in messages:
        message.pop("tracking_id", None)


@pytest.mark.parametrize("change", [_set_id, _drop_ids])
def test_r3_a_top_level_id_may_appear_change_or_vanish(monkeypatch, caplog, change):
    caplog.set_level(logging.INFO, logger=LOGGER)
    backend = witness_backend(monkeypatch)
    agent, model = witnessed_agent(backend, [("text", "answer")], messages=copy.deepcopy(RESTORED_HISTORY))
    agent.hooks.add_callback(BeforeModelCallEvent, lambda event: change(event.agent.messages))
    result, error = invoke(agent, "next prompt")
    assert hold_lines(caplog) == [] and error is None and backend._input_hold is False
    assert result.stop_reason == "end_turn" and model.calls == 1


def _change_content(messages):
    messages[0]["content"][0]["text"] = "changed text"


def _change_role(messages):
    messages[0]["role"] = "assistant"


def _change_metadata(messages):
    messages[1]["metadata"]["usage"]["inputTokens"] = 7


def _remove_metadata(messages):
    del messages[1]["metadata"]


def _remove_message(messages):
    del messages[0]


def _add_message(messages):
    messages.insert(0, {"role": "user", "content": [{"text": "unadmitted"}]})


def _nest_an_id(messages):
    messages[0]["content"][0]["tracking_id"] = "nested"


def _add_another_key(messages):
    messages[0]["unreviewed_key"] = "x"


@pytest.mark.parametrize(
    "change",
    [
        _change_content,
        _change_role,
        _change_metadata,
        _remove_metadata,
        _remove_message,
        _add_message,
        _nest_an_id,
        _add_another_key,
    ],
)
def test_r3_any_other_change_of_the_admitted_history_still_holds(monkeypatch, caplog, change):
    caplog.set_level(logging.INFO, logger=LOGGER)
    backend = witness_backend(monkeypatch)
    agent, model = witnessed_agent(backend, [("text", "answer")], messages=copy.deepcopy(RESTORED_HISTORY))
    agent.hooks.add_callback(BeforeModelCallEvent, lambda event: change(event.agent.messages))
    result, error = invoke(agent, "next prompt")
    assert result is None and error is not None and backend._input_hold is True
    assert hold_lines(caplog) == [PREFIX_REFUSED]
    assert model.calls == 1, "the hold belongs to the message after the changed history"


# ── R4: a TASK written under the earlier SDK release ──


def golden_task():
    stored = GOLDEN.read_bytes()
    task = json.loads(stored)["task"]
    assert set(task["sdk_snapshot"]["data"]) == RECORD_FIELDS
    assert all(
        set(message) <= MESSAGE_KEYS - {"tracking_id"} for message in checkpoint._task_causal_prefix(task)
    )
    assert checkpoint._sdk_snapshot_sha256(task["sdk_snapshot"]) == GOLDEN_SNAPSHOT_SHA256
    return stored, task


def restore_engine(monkeypatch, build):
    backend = witness_backend(monkeypatch)
    backend.approval_policy = _policy()
    backend._input_receipts = []
    backend._input_checker = None
    backend._normal_original_result = {"status": "RUNNING"}
    monkeypatch.setattr(backend, "_activate_research_binding", lambda: None)
    monkeypatch.setattr(backend, "_qualify_research_binding", lambda active=True: None)
    monkeypatch.setattr(backend, "build_agent", lambda: build(backend))
    engine = object.__new__(ChatEngine)
    engine.tool_backend = backend
    return engine, backend


def test_r4_a_pending_task_of_the_earlier_sdk_loads_through_the_restore(monkeypatch, caplog):
    """The PENDING/RUNNING branch loads the stored snapshot into a real Agent and invokes nothing."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    stored, task = golden_task()
    kept = copy.deepcopy(task)
    models = []

    def build(backend):
        agent, model = witnessed_agent(backend, [], tools=[scripted_sft])
        models.append(model)
        return agent

    engine, backend = restore_engine(monkeypatch, build)
    assert engine.load_normal_snapshot(task) == {"status": "RUNNING"}
    agent, prefix = engine._normal_agent, task["sdk_snapshot"]["data"]["messages"]
    assert without_ids(agent.messages) == prefix and models[0].calls == 0
    assert hold_lines(caplog) == [] and backend._input_hold is False
    assert task == kept and GOLDEN.read_bytes() == stored
    assert checkpoint._sdk_snapshot_sha256(task["sdk_snapshot"]) == GOLDEN_SNAPSHOT_SHA256


def test_r4_a_completed_task_of_the_earlier_sdk_continues_from_its_stored_snapshot(monkeypatch, caplog):
    """The completed restore: the product's completion snapshot of the stored TASK, loaded from its canonical
    text, then the one invocation without a prompt, with the witness active."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    stored, task = golden_task()
    snapshot = checkpoint._completed_task_snapshot(task, task["operation_digest"])
    snapshot_raw = checkpoint._canonical(snapshot)
    digest = checkpoint._sdk_snapshot_sha256(snapshot)
    backend = witness_backend(monkeypatch)
    agent, model = witnessed_agent(backend, [("text", FINAL_TEXT)], tools=[scripted_sft])
    agent.load_snapshot(Snapshot.from_dict(json.loads(snapshot_raw)))
    assert without_ids(agent.messages) == snapshot["data"]["messages"]
    result, error = invoke(agent, None)
    assert hold_lines(caplog) == [] and error is None and backend._input_hold is False
    assert result.stop_reason == "end_turn" and model.calls == 1
    assert without_ids(agent.messages)[:-1] == snapshot["data"]["messages"]
    if SDK_VERSION >= TRACKING_ID_SINCE:
        assert all("tracking_id" in message for message in agent.messages), "TRIGGER ABSENT: no live ids"
    assert checkpoint._canonical(snapshot) == snapshot_raw and GOLDEN.read_bytes() == stored
    assert checkpoint._sdk_snapshot_sha256(snapshot) == digest


class _LoadedAgent:
    """An agent whose load writes an id on every restored message."""

    def __init__(self, edit=None):
        self.messages, self.edit = [], edit

    def load_snapshot(self, snapshot):
        self.messages = copy.deepcopy(snapshot["data"]["messages"])
        for index, message in enumerate(self.messages):
            message["tracking_id"] = f"loaded-{index}"
        if self.edit is not None:
            self.edit(self.messages)


@pytest.mark.parametrize("edit", [None, _change_content, _remove_message, _nest_an_id])
def test_d6_a_restored_history_is_compared_with_the_stored_prefix_in_record_form(monkeypatch, edit):
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    monkeypatch.setattr("strands.Snapshot", SimpleNamespace(from_dict=lambda value: value))
    _stored, task = golden_task()
    agent = _LoadedAgent(edit)
    engine, _backend = restore_engine(monkeypatch, lambda backend: agent)
    if edit is None:
        assert engine.load_normal_snapshot(task) == {"status": "RUNNING"}
        assert engine._normal_agent is agent
        assert all("tracking_id" in message for message in agent.messages)
    else:
        with pytest.raises(ValueError, match=r"^loaded SDK snapshot changed admitted history$"):
            engine.load_normal_snapshot(task)
        assert not hasattr(engine, "_normal_agent")


# ── R5: writer equivalence ──


def test_r5_two_captures_equal_each_other_and_the_golden_of_the_earlier_sdk(record_catalog, monkeypatch):
    _stored, task = golden_task()
    texts = []
    for label in "de":
        context = run_context(label)
        agent, seen = sft_turn(monkeypatch, context)
        assert seen.error is None, "the product capture refused a real SDK turn: " + repr(seen.error)
        texts.append(blank_created_at(captured_record(context, seen)))
        assert agent.messages[:2] == seen.live, "the capture edited the live history"
        if SDK_VERSION >= TRACKING_ID_SINCE:
            assert all("tracking_id" in message for message in agent.messages), "the live ids are gone"
    assert texts[0] == texts[1], "two captures of one conversation differ"
    assert texts[0] == blank_created_at(task["sdk_snapshot"]), "the capture differs from the golden"


# ── R6: size ──


def test_r6_the_size_line_counts_the_bytes_of_the_record_form(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    monkeypatch.setattr(checkpoint, "current_normal_run", lambda: SimpleNamespace(run_id=RUN_ID))
    backend = ToolBindingBackend(Settings())
    backend._model = ScriptedModel([("text", "first answer"), ("text", "second answer")])
    agent = backend.build_agent()
    agent.callback_handler = lambda *args, **kwargs: None
    sent = []
    agent.hooks.add_callback(
        BeforeModelCallEvent, lambda event: sent.append(copy.deepcopy(event.agent.messages))
    )
    agent("first prompt")
    agent("second prompt")
    parsed = [
        SIZE_LINE.fullmatch(record.getMessage())
        for record in caplog.records
        if record.name == LOGGER and record.getMessage().startswith("normal_conversation_size")
    ]
    assert all(parsed) and [int(match.group(4)) for match in parsed] == [1, 3]
    assert [int(match.group(3)) for match in parsed] == [
        len(checkpoint._canonical(without_ids(messages)).encode("utf-8")) for messages in sent
    ]
    assert {int(match.group(5)) for match in parsed} == {checkpoint._MAX_BYTES} == {160 * 1024}
    if SDK_VERSION >= TRACKING_ID_SINCE:
        assert all("tracking_id" in message for messages in sent for message in messages)


def _record_fits(context, seen, record, padding):
    """Whether the record form of the scripted turn with ``padding`` more prompt bytes is within the cap."""
    messages = without_ids(copy.deepcopy(seen.live))
    messages[0]["content"][0]["text"] = PROMPT + "x" * padding
    snapshot = copy.deepcopy(record)
    snapshot["data"]["messages"] = messages
    try:
        checkpoint._budget_sft_snapshot(context, checkpoint._causal_call(SFT_USE, messages), snapshot)
    except ValueError as error:
        assert str(error) == SIZE_REFUSAL
        return False
    return True


def test_r6_the_largest_record_that_fits_the_cap_is_accepted_from_a_real_turn(record_catalog, monkeypatch):
    assert checkpoint._MAX_BYTES == 160 * 1024
    context = run_context("a")
    _agent, seen = sft_turn(monkeypatch, context)
    assert seen.error is None, "the product capture refused a real SDK turn: " + repr(seen.error)
    record = captured_record(context, seen)
    fits, over = 0, checkpoint._MAX_BYTES
    assert _record_fits(context, seen, record, fits) and not _record_fits(context, seen, record, over)
    while over - fits > 1:
        middle = (fits + over) // 2
        if _record_fits(context, seen, record, middle):
            fits = middle
        else:
            over = middle
    assert fits > 150 * 1024, "the scripted turn is not near the cap"
    context = run_context("b")
    agent, seen = sft_turn(monkeypatch, context, PROMPT + "x" * fits)
    assert seen.error is None, "a record within the cap was refused: " + repr(seen.error)
    stored = captured_record(context, seen)
    assert stored["data"]["messages"] == without_ids(seen.live) == without_ids(agent.messages[:2])
    context = run_context("c")
    _agent, seen = sft_turn(monkeypatch, context, PROMPT + "x" * over)
    assert type(seen.error) is ValueError and str(seen.error) == SIZE_REFUSAL
    assert context._causal_call is None and context._sft_snapshot is None


# ── R7: tool results stay exact ──


@pytest.mark.parametrize("added_key", [None, "tracking_id", "unreviewed_key"])
def test_r7_a_tool_result_with_any_added_top_level_key_is_refused(monkeypatch, caplog, added_key):
    caplog.set_level(logging.INFO, logger=LOGGER)
    backend = witness_backend(monkeypatch, registry=("echo_step",))
    agent, model = witnessed_agent(
        backend, [("tool", "use-1", "echo_step", {"text": "x"}), ("text", "done")], tools=[echo_step]
    )
    marked = []

    def mark_then_change(event):
        backend._mark_pending_pause(event)
        marked.append(dict(backend._pending_batch_pause.admitted))
        if added_key is not None:
            event.result[added_key] = "added after admission"

    agent.hooks.add_callback(AfterToolCallEvent, mark_then_change)
    result, error = invoke(agent, "use the tool")
    assert len(marked) == 1 and set(marked[0]) == {"use-1"}, "the tool result was never admitted"
    if added_key is None:
        assert hold_lines(caplog) == [] and error is None and backend._input_hold is False
        assert result.stop_reason == "end_turn" and model.calls == 2
    else:
        assert result is None and error is not None and backend._input_hold is True
        assert hold_lines(caplog) == [BATCH_REFUSED]
        assert model.calls == 1


# ── R8: raw SDK tripwire ──


def test_r8_raw_sdk_output_carries_only_the_reviewed_fields():
    agent = Agent(
        model=ScriptedModel([("tool", "use-1", "echo_step", {"text": "x"}), ("text", "done")]),
        tools=[echo_step],
        callback_handler=None,
    )
    agent("use the tool")
    raw = agent.take_snapshot(preset="session").to_dict()
    assert set(raw) == ENVELOPE_FIELDS, DECISION
    assert (raw["scope"], raw["schema_version"], raw["app_data"]) == ("agent", "1.0", {}), DECISION
    expected = RECORD_FIELDS | ({"model_state"} if SDK_VERSION >= MODEL_STATE_SINCE else set())
    assert set(raw["data"]) == expected, DECISION
    assert raw["data"].get("model_state", {}) == {}, DECISION
    record = copy.deepcopy(raw)
    record["data"].pop("model_state", None)
    record["data"]["messages"] = without_ids(record["data"]["messages"])
    restored = Agent(model=ScriptedModel([("text", "again")]), tools=[echo_step], callback_handler=None)
    restored.load_snapshot(Snapshot.from_dict(record))
    restored("continue")
    again = restored.take_snapshot(preset="session").to_dict()
    assert set(again["data"]) == expected, DECISION
    appended, reloaded = agent.messages, restored.messages
    assert len(appended) == 4 and len(reloaded) == 6
    for message in [*appended, *raw["data"]["messages"], *reloaded, *again["data"]["messages"]]:
        assert {"role", "content"} <= set(message) <= MESSAGE_KEYS, DECISION
    expected_ids = SDK_VERSION >= TRACKING_ID_SINCE
    assert all(("tracking_id" in message) is expected_ids for message in [*appended, *reloaded]), DECISION


# ── R9: the pinned sources ──


def test_r9_the_three_pinned_sources_keep_their_checksums():
    tests = Path(__file__).resolve().parent
    files = {
        "f": tests / "normal_sft_cold_fixture.py",
        "publisher": tests / "normal_reload_original_completion.py",
        "normal": Path(checkpoint.__file__),
    }
    pins = dict(
        re.findall(
            r'hashlib\.sha256\(Path\((\w+)\.__file__\)\.read_bytes\(\)\)\.hexdigest\(\)\s*==\s*"([0-9a-f]{64})"',
            (tests / "test_normal_durable_registered_reload.py").read_text(encoding="utf-8"),
        )
    )
    assert set(pins) == set(files), "the checksum test no longer pins these three sources"
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins[name], path.name + " changed"


# ── The witness on the worker thread that observes the original job ──


def test_the_witness_on_the_observing_thread_reads_a_history_the_sdk_leaves_alone(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    monkeypatch.setattr(binding, "_warn_precommit_handoff", lambda *args, **kwargs: None)
    backend = witness_backend(monkeypatch, registry=("echo_step",))
    monkeypatch.setattr(backend, "_qualify_warm_producer", lambda pending: None)
    agent, model = witnessed_agent(
        backend, [("tool", "use-1", "echo_step", {"text": "x"}), ("text", "done")], tools=[echo_step]
    )
    seen = SimpleNamespace(error=None, content=UNSET, loop=None, histories=[], qualified=[], observed=[])
    qualify = backend._qualify_precommit_original

    def qualifying(*args):
        qualify(*args)
        seen.qualified.append(threading.get_ident())

    monkeypatch.setattr(backend, "_qualify_precommit_original", qualifying)

    class Adapter:
        @contextmanager
        def qualified_io(self, callback):
            callback()
            yield
            callback()

    class Store:
        def observe(self, reference):
            seen.observed.append(threading.get_ident())
            seen.histories.append(canonical(agent.messages))
            return {
                "request_ref": reference,
                "job_name": "job-1",
                "job_arn": "arn:job-1",
                "status": "RUNNING",
                "success": False,
            }

    handle = SimpleNamespace(id="job-1", arn="arn:job-1", extra={})
    pending = binding._PendingOriginalProducer(
        kind="data",
        tool_use_id="use-1",
        reference="ref-1",
        store=Store(),
        record=SimpleNamespace(handle=handle, adapter=Adapter()),
        adapter=Adapter(),
        agent=agent,
        graph=(),
        handle_pin=None,
        pid=os.getpid(),
    )

    async def observe_original(event):
        witness = backend._pending_batch_pause
        witness.originals[pending.tool_use_id] = pending
        seen.loop = threading.get_ident()
        seen.histories.append(canonical(event.agent.messages))
        try:
            seen.content = await asyncio.to_thread(
                backend._read_precommit_original, event, witness, pending, threading.Event()
            )
        except Exception as error:
            seen.error = error
        seen.histories.append(canonical(event.agent.messages))
        witness.originals.pop(pending.tool_use_id)
        backend._mark_pending_pause(event)

    agent.hooks.add_callback(AfterToolCallEvent, observe_original)
    result, error = invoke(agent, "use the tool")
    assert hold_lines(caplog) == [] and error is None and seen.error is None
    assert backend._input_hold is False and seen.content is None, "a running original has no result yet"
    assert len(seen.observed) == 1 and seen.observed[0] != seen.loop, "the read ran on the agent's thread"
    assert seen.qualified == seen.observed * 4, "the witness ran four times on the observing thread"
    assert len(seen.histories) == 3 and len(set(seen.histories)) == 1, "the history changed during the read"
    if SDK_VERSION >= TRACKING_ID_SINCE:
        assert all("tracking_id" in message for message in json.loads(seen.histories[0]))
    assert result.stop_reason == "end_turn" and model.calls == 2


# ── A measured time to first byte is DATA of the record ──


def every_model_call_measures_the_same_first_byte_time(monkeypatch):
    """Make the SDK measure ``FIRST_BYTE_MS`` on every model call, however long the call takes.

    The SDK reads its clock when a model call starts and again at the first content event of the stream,
    and writes the difference to the message metadata as ``timeToFirstByteMs`` once it is 1 ms or more.
    Here each reading of that clock is 0.125 s after the one before, a step that binary floats hold
    exactly, so every run of one script carries the same value.
    """
    readings = itertools.count()
    clock = SimpleNamespace(time=lambda: 1000.0 + next(readings) * FIRST_BYTE_MS / 1000)
    monkeypatch.setattr(streaming, "time", clock)
    try:  # newer strands-agents releases read the start of a model call in this module
        from strands.models import _model_proxy
    except ImportError:
        return
    monkeypatch.setattr(_model_proxy, "time", clock)


def without_first_byte_time(messages):
    """A copy of record-form messages without the measured field; this module's own statement."""
    messages = copy.deepcopy(messages)
    for message in messages:
        message.get("metadata", {}).get("metrics", {}).pop("timeToFirstByteMs", None)
    return messages


def sized_conversation(monkeypatch, caplog):
    """Two prompts through the product's agent build with a scripted model: the size lines and the histories."""
    caplog.clear()
    sent = []
    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "current_normal_run", lambda: SimpleNamespace(run_id=RUN_ID))
        backend = ToolBindingBackend(Settings())
        model = backend._model = ScriptedModel([("text", "first answer"), ("text", "second answer")])
        agent = backend.build_agent()
        assert agent.model is model, "the product build replaced the scripted model"
        agent.callback_handler = lambda *args, **kwargs: None
        agent.hooks.add_callback(
            BeforeModelCallEvent, lambda event: sent.append(without_ids(copy.deepcopy(event.agent.messages)))
        )
        agent("first prompt")
        agent("second prompt")
    lines = [
        entry.getMessage()
        for entry in caplog.records
        if entry.name == LOGGER and entry.getMessage().startswith("normal_conversation_size")
    ]
    parsed = [SIZE_LINE.fullmatch(line) for line in lines]
    assert all(parsed) and [int(match.group(4)) for match in parsed] == [1, 3]
    sizes = [int(match.group(3)) for match in parsed]
    assert sizes == [len(canonical(messages).encode("utf-8")) for messages in sent]
    return SimpleNamespace(
        lines=lines,
        sizes=sizes,
        sent=sent,
        history=without_ids(agent.messages),
        witness=ToolBindingBackend._pause_history(agent.messages),
    )


def captured_turn(monkeypatch, label):
    """One real SDK turn that calls the fine-tune tool: the live history at the capture and the stored record."""
    context = run_context(label)
    _agent, seen = sft_turn(monkeypatch, context)
    assert seen.error is None, "the product capture refused a real SDK turn: " + repr(seen.error)
    return SimpleNamespace(live=without_ids(seen.live), stored=captured_record(context, seen))


def test_a_measured_first_byte_time_stays_in_the_record_counts_in_the_size_and_is_equal_in_two_runs(
    request, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger=LOGGER)
    plain_talk = sized_conversation(monkeypatch, caplog)
    with monkeypatch.context() as patch:
        every_model_call_measures_the_same_first_byte_time(patch)
        talks = [sized_conversation(monkeypatch, caplog) for _ in range(2)]
    # The capture environment replaces the model of every agent the product builds, so it starts only here.
    request.getfixturevalue("record_catalog")
    plain_turn = captured_turn(monkeypatch, "a")
    with monkeypatch.context() as patch:
        every_model_call_measures_the_same_first_byte_time(patch)
        turns = [captured_turn(monkeypatch, label) for label in "bc"]

    field_bytes = len(canonical(MEASURED)) - len(canonical(USAGE))
    assert field_bytes == len(f',"timeToFirstByteMs":{FIRST_BYTE_MS}')
    plain_answers = [message["metadata"] for message in plain_talk.history if message["role"] == "assistant"]
    assert plain_answers == [USAGE] * 2 and plain_turn.stored["data"]["messages"][1]["metadata"] == USAGE, (
        "the clock fixture of this file is not in force"
    )

    # The stored record keeps the field with its value; nothing else differs from the record without it.
    for turn in turns:
        assert turn.live[1]["metadata"] == MEASURED, "TRIGGER ABSENT: the SDK added no measured time"
        messages = turn.stored["data"]["messages"]
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert messages[1]["metadata"] == MEASURED, "the stored record lost or changed the measured time"
        assert messages == turn.live
        assert without_first_byte_time(messages) == plain_turn.stored["data"]["messages"]

    # The logged size is the canonical size of the record form: each assistant message adds the field's bytes.
    for talk in talks:
        answers = [message["metadata"] for message in talk.history if message["role"] == "assistant"]
        assert answers == [MEASURED] * 2, "TRIGGER ABSENT: the SDK added no measured time"
        assistants = [sum(message["role"] == "assistant" for message in sent) for sent in talk.sent]
        assert assistants == [0, 1]
        assert talk.sizes == [
            size + field_bytes * count for size, count in zip(plain_talk.sizes, assistants, strict=True)
        ]
        assert without_first_byte_time(talk.history) == plain_talk.history
        assert f'"timeToFirstByteMs":{FIRST_BYTE_MS}' in talk.witness

    # Two runs with the same measured time give equal records, size lines, histories and witness text.
    assert blank_created_at(turns[0].stored) == blank_created_at(turns[1].stored), "two captures differ"
    assert talks[0].lines == talks[1].lines and talks[0].sizes == talks[1].sizes
    assert talks[0].history == talks[1].history and talks[0].witness == talks[1].witness

    # The comparison a restore makes: the loaded history in record form against the stored prefix.
    restored = Agent(model=ScriptedModel([]), tools=[scripted_sft], callback_handler=None)
    restored.load_snapshot(Snapshot.from_dict(copy.deepcopy(turns[0].stored)))
    assert binding._record_messages(restored.messages) == turns[0].stored["data"]["messages"]
