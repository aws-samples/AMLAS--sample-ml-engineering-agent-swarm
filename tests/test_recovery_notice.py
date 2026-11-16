"""Recovery notice for a conversation restored from its saved checkpoint.

The resumed Agent used to learn nothing about its restoration: the final text of a recovered run said "No worker interruption occurred". The runtime
now appends one short factual notice to the system prompt of the claimed Agent, once, after the start CAS (`begin_completion_invocation`) succeeded and
before the single invocation. These tests drive the real `ChatEngine.continue_normal_completion` with a scripted model that records the system prompt it
receives, a real Strands Agent and a fake continuation context. They pin: the notice is present exactly once after a restore; absent when the start CAS is
lost (second worker) or when nothing was restored; calling the helper again does not duplicate it; the messages, the claimed snapshot and every
qualification input are byte-identical to the unchanged code (the notice lives outside the SDK snapshot).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from strands import Agent
from strands.models import Model

from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.models import ChatResponse

MARKER = "Runtime notice (restoration):"
BASE_PROMPT = "BASE SYSTEM PROMPT"
RUN_ID = hashlib.sha256(b"recovery-notice-run").hexdigest()[:32]
OP = hashlib.sha256(b"recovery-notice-operation").hexdigest()
SDK = hashlib.sha256(b"recovery-notice-sdk-snapshot").hexdigest()
TASK = hashlib.sha256(b"recovery-notice-task-checkpoint").hexdigest()
SRC = Path(__file__).resolve().parents[1] / "src" / "amlas"


class RecordingModel(Model):
    def __init__(self):
        self.system_prompts = []
        self.received_messages = []
        self.config = {"model_id": "recovery-notice-component"}

    def update_config(self, **kwargs):
        self.config.update(kwargs)

    def get_config(self):
        return copy.deepcopy(self.config)

    def structured_output(self, *args, **kwargs):
        raise AssertionError("not used")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.system_prompts.append(system_prompt)
        self.received_messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Continuation complete."}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }


def _prefix():
    return [
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


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _continuation(epoch, state):
    return {
        "schema_version": 3,
        "epoch": epoch,
        "token": "t" * 32,
        "status": "consumed-ready",
        "operation_digest": OP,
        "sdk_snapshot_sha256": SDK,
        "invocation_state": state,
    }


def _legacy_continuation():
    """The schema 1 shape: epoch 1, no digests, no invocation state."""
    return {"schema_version": 1, "epoch": 1, "token": "t" * 32, "status": "consumed-ready"}


def _task_ref():
    return _canonical(
        {
            "byte_count": 4096,
            "format": "json",
            "sha256": TASK,
            "uri": "s3://declared-bucket/sessions/declared/tasks/" + TASK + ".json",
            "version_id": "declared-version",
        }
    )


class FakeContext:
    """The parts of NormalRunContext that continue_normal_completion touches."""

    def __init__(self, row, *, schema=3, prior=False):
        self.row = row
        self.schema = schema
        self.run_id = RUN_ID
        self.qualified = []
        self._completion_prior_report = prior
        first = _continuation(1, "ready") if schema == 3 else _legacy_continuation()
        self._completion_receipt = ("token", _canonical(first), _task_ref())

    def qualify_completion(self, token, *, policy, qualify_original, sdk_snapshot):
        self.qualified.append(hashlib.sha256(_canonical(sdk_snapshot).encode()).hexdigest())

    def begin_completion_invocation(self, token, *, policy, qualify_original, sdk_snapshot):
        self.qualify_completion(
            token, policy=policy, qualify_original=qualify_original, sdk_snapshot=sdk_snapshot
        )
        with self.row["lock"]:
            if self.row["state"] != "ready":
                raise ValueError("original completion invocation already attempted")
            self.row["state"] = "invoking"
        if self.schema == 3:
            self._completion_receipt = ("token", _canonical(_continuation(2, "invoking")), _task_ref())


class FakeBackend:
    def __init__(self):
        self._input_lock = threading.RLock()
        self._input_hold = False
        self.approval_policy = object()

    def _qualify_research_binding(self, *args, **kwargs):
        return None

    def _invocation_state(self):
        return {}

    def _pending_pause_result(self, agent, result):
        return False

    @staticmethod
    def _extract_final_text(result):
        return "done"

    def _extract_qualified_normal_executions(self, result, agent):
        return []

    def _extract_guardrail_outcome(self, result):
        return None

    def _input_admission_data(self):
        return None


def _engine(row, *, schema=3, prior=False):
    model = RecordingModel()
    prefix = _prefix()
    agent = Agent(
        model=model,
        tools=[],
        messages=copy.deepcopy(prefix),
        system_prompt=BASE_PROMPT,
        callback_handler=None,
    )
    context = FakeContext(row, schema=schema, prior=prior)
    snapshot_raw = _canonical({"data": {"messages": prefix}})
    engine = ChatEngine.__new__(ChatEngine)
    engine.tool_backend = FakeBackend()
    engine._normal_agent = agent
    engine._normal_completion = (context, "token", lambda: None)
    engine._normal_completion_snapshot = snapshot_raw
    engine._agent_turn_response = lambda turn: ChatResponse(message=turn.final_text, action="tool_agent_turn")
    return engine, agent, model, context, prefix, snapshot_raw


def _row():
    return {"lock": threading.Lock(), "state": "ready"}


def test_restored_agent_receives_the_notice_exactly_once():
    engine, agent, model, *_ = _engine(_row())
    assert engine.continue_normal_completion().message.split("\n\n", 1)[0] == "done"
    assert len(model.system_prompts) == 1, "one model call in the declared continuation"
    prompt = model.system_prompts[0]
    assert prompt.startswith(BASE_PROMPT), "the existing system prompt is kept as it is"
    assert prompt.count(MARKER) == 1
    assert "invoking epoch 2" in prompt and "submit_qwen_sft" in prompt
    assert agent.system_prompt.count(MARKER) == 1


def test_messages_snapshot_and_qualification_inputs_are_unchanged():
    engine, _agent, model, context, prefix, snapshot_raw = _engine(_row())
    engine.continue_normal_completion()
    assert model.received_messages == [prefix], (
        "the conversation sent to the model is exactly the claimed history"
    )
    assert engine._normal_completion_snapshot == snapshot_raw
    expected = hashlib.sha256(_canonical(json.loads(snapshot_raw)).encode()).hexdigest()
    assert set(context.qualified) == {expected} and len(context.qualified) >= 3, (
        "every qualification saw the same claimed snapshot"
    )


def test_a_lost_start_cas_leaves_no_notice_and_makes_no_model_call():
    row = _row()
    winner = _engine(row)
    loser = _engine(row)
    winner[0].continue_normal_completion()
    with pytest.raises(ValueError):
        loser[0].continue_normal_completion()
    assert loser[2].system_prompts == [], "the second worker never reaches the model"
    assert loser[1].system_prompt == BASE_PROMPT, "and carries no notice"
    seen = [prompt for built in (winner, loser) for prompt in built[2].system_prompts]
    assert sum(prompt.count(MARKER) for prompt in seen) == 1, "exactly one notice across both workers"


def test_no_notice_without_a_restore():
    model = RecordingModel()
    agent = Agent(model=model, tools=[], system_prompt=BASE_PROMPT, callback_handler=None)
    agent("hello")
    assert model.system_prompts == [BASE_PROMPT] and MARKER not in model.system_prompts[0]


def test_helper_is_idempotent_and_reads_the_epoch_of_the_invoking_row():
    from amlas.tui.chat.recovery_notice import apply_recovery_notice, recovery_notice

    agent = SimpleNamespace(system_prompt=BASE_PROMPT)
    context = SimpleNamespace(_completion_receipt=("token", _canonical(_continuation(2, "invoking")), "ref"))
    assert apply_recovery_notice(agent, context) is True
    once = agent.system_prompt
    assert apply_recovery_notice(agent, context) is False
    assert agent.system_prompt == once and once.count(MARKER) == 1
    assert once == BASE_PROMPT + recovery_notice(2)
    legacy = SimpleNamespace(_completion_receipt=("token", _canonical(_continuation(1, "ready")), "ref"))
    assert "invoking epoch 1" in recovery_notice(1)
    assert apply_recovery_notice(SimpleNamespace(system_prompt="x"), legacy) is True


def test_helper_refuses_a_missing_receipt_or_an_unknown_epoch():
    from amlas.tui.chat.recovery_notice import apply_recovery_notice, recovery_notice

    with pytest.raises(ValueError):
        apply_recovery_notice(SimpleNamespace(system_prompt=BASE_PROMPT), SimpleNamespace())
    for bad in (0, 3, True, "2", None):
        with pytest.raises(ValueError):
            recovery_notice(bad)


def test_notice_text_is_short_factual_and_free_of_identifiers():
    from amlas.tui.chat.recovery_notice import recovery_notice

    text = recovery_notice(2).strip()
    assert text.startswith(MARKER) and text.isascii() and len(text) <= 900
    sentences = [s for s in re.split(r"(?<=[.])\s+", text) if s]
    assert 2 <= len(sentences) <= 4, sentences
    assert not re.search(r"\d{12}", text), "no account-number-shaped digits (governance rule)"
    for forbidden in (
        "arn:",
        "s3://",
        "http",
        "token",
        "secret",
        "password",
        "caused by",
        "crash",
        "failure of",
    ):
        assert forbidden not in text.lower(), forbidden
    assert recovery_notice(2) == recovery_notice(2), "deterministic"


def test_only_the_continuation_applies_the_notice():
    users = [
        p
        for p in SRC.rglob("*.py")
        if "apply_recovery_notice" in p.read_text() and p.name != "recovery_notice.py"
    ]
    assert [p.name for p in users] == ["engine.py"]
    engine_text = (SRC / "tui" / "chat" / "engine.py").read_text()
    claim = engine_text.split("def claim_normal_completion", 1)[1].split("def continue_normal_completion", 1)[
        0
    ]
    assert "recovery_notice" not in claim, (
        "claim is replayed by restore retries and qualification: it must not add the notice"
    )
    cont = engine_text.split("def continue_normal_completion", 1)[1].split("def load_normal_snapshot", 1)[0]
    assert (
        cont.index("check_recovery_notice_ready(agent, context)")
        < cont.index("begin_completion_invocation(")
        < cont.index("apply_recovery_notice(agent, context)")
        < cont.index("agent(None")
    ), "checked before the start CAS, applied after it and before the invocation"


def test_a_refusal_before_the_start_cas_burns_nothing():
    row = _row()
    engine, agent, model, *_ = _engine(row)
    agent.system_prompt = [
        {"text": BASE_PROMPT},
        {"cachePoint": {"type": "default"}},
    ]  # blocks a text notice would drop
    with pytest.raises(ValueError):
        engine.continue_normal_completion()
    assert row["state"] == "ready", "the single-use start CAS is not spent by the notice check"
    assert model.system_prompts == []


def test_prompt_drift_after_the_notice_is_refused_by_the_fence():
    from strands.hooks import BeforeModelCallEvent

    engine, agent, model, *_ = _engine(_row())
    agent.hooks.add_callback(BeforeModelCallEvent, lambda event: setattr(agent, "system_prompt", "DRIFT"))
    with pytest.raises(ValueError):
        engine.continue_normal_completion()
    assert model.system_prompts == [], "the fence refuses before the model sees a changed prompt"
    assert engine.tool_backend._input_hold is True


def test_a_legacy_schema_one_receipt_states_epoch_one():
    engine, _agent, model, *_ = _engine(_row(), schema=1)
    engine.continue_normal_completion()
    assert "invoking epoch 1" in model.system_prompts[0]


def test_an_agent_without_a_system_prompt_gets_the_notice_alone_once():
    from amlas.tui.chat.recovery_notice import apply_recovery_notice, recovery_notice

    engine, agent, model, context, *_ = _engine(_row())
    agent.system_prompt = None
    engine.continue_normal_completion()
    assert model.system_prompts == [recovery_notice(2).lstrip("\n")], (
        "no leading blank lines in front of a lone notice"
    )
    assert apply_recovery_notice(agent, context) is False, "a second call changes nothing"
    assert agent.system_prompt.count(MARKER) == 1


def test_an_unreadable_receipt_is_refused_before_the_start_cas():
    row = _row()
    engine, agent, model, context, *_ = _engine(row)
    context._completion_receipt = None
    with pytest.raises(ValueError):
        engine.continue_normal_completion()
    assert row["state"] == "ready", "the single-use start CAS is not spent when the epoch cannot be read"
    assert model.system_prompts == [] and agent.system_prompt == BASE_PROMPT


def test_notice_asks_for_the_restoration_in_the_final_report_without_a_cause():
    from amlas.tui.chat.recovery_notice import recovery_notice

    text = recovery_notice(2)
    for phrase in (
        "Your final report must state this restoration as a fact",
        "in the section where you describe persistence and recovery",
        "must not say that no restoration or worker interruption occurred",
        "name a cause",
        "claim that you tested or built recovery",
        "no cause is recorded here",
    ):
        assert phrase in text, phrase
    assert "replaced" not in text, "the runtime does not know why the earlier invocation ended"
    assert len(text.strip()) <= 900


def test_the_model_request_carries_the_report_instruction():
    engine, _agent, model, *_ = _engine(_row())
    engine.continue_normal_completion()
    assert "persistence and recovery" in model.system_prompts[0]
    assert model.system_prompts[0].endswith("or claim that you tested or built recovery.")
