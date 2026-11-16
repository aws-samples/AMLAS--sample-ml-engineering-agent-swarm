"""Unresolved (UNKNOWN) polling of an original operation is bounded by wall clock.

Inert ports only: a fake store answers every observation and a fake clock replaces the
15-second wait, so no AWS, model or native call happens. Success and failure rules stay as they were.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
LOGGER = "amlas.tui.chat.tool_binding_backend"
RUN_ID = "ab" * 16
CADENCE = 15.0
CAP_POLLS = 61  # observations at 0, 15, ... 900 s of fake time; the 61st is the first at or past 900 s
LINE = re.compile(
    r"^normal_unknown_poll run_id=(none|[0-9a-f]{32}) operation=([A-Z]+) polls=([0-9]+) "
    r"elapsed_s=([0-9]+) limit_s=([0-9]+) outcome=(waiting|closed)$"
)
KIND_LABEL = {
    "data": "DATA",
    "case": "CASE",
    "preparation": "PREP",
    "sft": "SFT",
    "reload": "RELOAD",
    "generation": "GEN",
    "pair": "PAIR",
}


class UnboundedPollingError(AssertionError):
    """Raised by the fake store when the code polls without end."""


class Clock:
    def __init__(self, now=5000.0):
        self.now = now

    def __call__(self):
        return self.now


class Adapter:
    @contextmanager
    def qualified_io(self, callback):
        yield


class Store:
    def __init__(self, script, limit=400):
        self.script, self.limit, self.calls = script, limit, 0

    def observe(self, reference):
        self.calls += 1
        if self.calls > self.limit:
            raise UnboundedPollingError("polling did not end")
        step = self.script(self.calls) if callable(self.script) else self.script[self.calls - 1]
        return {"request_ref": reference, "job_name": "job-1", "job_arn": "arn:job-1", **step}


def unknown(status="UNKNOWN"):
    return {"status": status, "success": False}


RUNNING = {"status": "RUNNING", "success": False}
DATA_DONE = {"status": "SUCCEEDED", "success": True, "acquired": True}


class Rig:
    def __init__(self, monkeypatch, kind, script, *, limit=400):
        self.clock = Clock()
        monkeypatch.setattr(binding, "_precommit_unknown_clock", self.clock, raising=False)

        async def wait(_backend):
            self.clock.now += CADENCE

        monkeypatch.setattr(ToolBindingBackend, "_wait_precommit_original", wait)
        monkeypatch.setattr(
            ToolBindingBackend,
            "_qualify_precommit_original",
            lambda self, event, witness, pending, closed: None,
        )
        monkeypatch.setattr(ToolBindingBackend, "_qualify_pending_pause", lambda self, witness: None)
        monkeypatch.setattr(binding, "_warn_precommit_handoff", lambda *args, **kwargs: None)
        self.backend = ToolBindingBackend(Settings())
        self.agent = SimpleNamespace(messages=[])
        handle = SimpleNamespace(id="job-1", arn="arn:job-1", extra={})
        record = (
            SimpleNamespace(handle=handle, adapter=Adapter()) if kind in {"data", "preparation"} else handle
        )
        self.store = Store(script, limit)
        self.pending = binding._PendingOriginalProducer(
            kind=kind,
            tool_use_id="use-1",
            reference="ref-1",
            store=self.store,
            record=record,
            adapter=Adapter(),
            agent=self.agent,
            graph=(),
            handle_pin=None,
            pid=os.getpid(),
        )
        self.backend._warm_pending_producers[self.backend._producer_key(self.pending)] = self.pending
        self.event = SimpleNamespace(
            agent=self.agent, tool_use={"toolUseId": "use-1", "name": "tool"}, retry=False, result=None
        )
        self.witness = binding._PendingBatchPause(self.agent, {}, {}, None, None, os.getpid(), "[]")

    def loop(self):
        return asyncio.run(self.backend._complete_original_before_commit(self.event, self.witness))

    def read(self):
        return self.backend._read_precommit_original(
            self.event, self.witness, self.pending, threading.Event()
        )


def lines(caplog):
    return [
        r for r in caplog.records if r.name == LOGGER and r.getMessage().startswith("normal_unknown_poll ")
    ]


@pytest.fixture(autouse=True)
def active_run(monkeypatch):
    from amlas.control_plane import normal_run_checkpoint

    run = SimpleNamespace(
        run_id=RUN_ID,
        _captured_sft_snapshot=lambda *args: {"inert": True},
        save_admitted_task=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(normal_run_checkpoint, "current_normal_run", lambda: run)


CASES = [
    ("data", "UNKNOWN"),
    ("data", "observation_failed"),
    ("preparation", "UNKNOWN"),
    ("preparation", "observation_failed"),
    ("sft", "UNKNOWN"),
    ("sft", "observation_failed"),
    ("reload", "UNKNOWN"),
    ("generation", "observation_failed"),
    ("pair", "UNKNOWN"),
    ("case", "UNKNOWN"),
    ("data", "verification_failed"),
    ("sft", "output_verification_failed"),
]


@pytest.mark.parametrize(("kind", "status"), CASES, ids=[f"{k}-{s}" for k, s in CASES])
def test_unresolved_polling_ends_in_a_hold_after_the_wall_clock_cap(monkeypatch, caplog, kind, status):
    caplog.set_level(logging.INFO, logger=LOGGER)
    rig = Rig(monkeypatch, kind, lambda call: unknown(status))
    with pytest.raises(binding._InputAdmissionHeldError):
        rig.loop()
    assert rig.store.calls == CAP_POLLS
    assert rig.clock.now - 5000.0 == 900.0
    assert rig.backend._warm_closed == "submission_unresolved"
    assert rig.backend._warm_closed_diagnostic.as_data() == {
        "phase": "native_observation",
        "reason": "original_unknown_budget_exhausted",
        "error_class": "UNKNOWN",
    }
    observed = [LINE.fullmatch(r.getMessage()) for r in lines(caplog)]
    assert all(observed), [r.getMessage() for r in lines(caplog)]
    assert {m.group(2) for m in observed} == {KIND_LABEL[kind]}
    assert {m.group(1) for m in observed} == {RUN_ID}
    assert [int(m.group(3)) for m in observed] == [1, 8, 16, 24, 32, 40, 48, 56, CAP_POLLS]
    assert [m.group(6) for m in observed] == ["waiting"] * 8 + ["closed"]
    assert (int(observed[-1].group(4)), int(observed[-1].group(5))) == (900, 900)
    assert all(r.exc_info is None and r.stack_info is None for r in lines(caplog))


def test_cap_is_between_the_model_case_budget_and_the_driver_budget():
    assert binding._PRECOMMIT_UNKNOWN_WALL_CLOCK_SECONDS == 900.0
    assert 8 * CADENCE < binding._PRECOMMIT_UNKNOWN_WALL_CLOCK_SECONDS < 4 * 3600 / 8


def test_pending_or_running_observation_resets_the_unresolved_run(monkeypatch):
    sequence = [unknown()] * 50 + [RUNNING] + [unknown("observation_failed")] * 50
    sequence += [{"status": "PENDING", "success": False}] + [unknown()] * 50 + [DATA_DONE]
    rig = Rig(monkeypatch, "data", lambda call: sequence[call - 1])
    outcomes = []
    for _ in sequence:
        outcomes.append(rig.read())
        rig.clock.now += CADENCE
    assert outcomes[:-1] == [None] * (len(sequence) - 1)
    assert type(outcomes[-1]) is list and outcomes[-1][0]["text"].startswith("{")
    assert rig.backend._warm_closed is None
    assert rig.backend._precommit_unresolved == {}


def test_healthy_running_time_does_not_spend_the_unresolved_budget(monkeypatch):
    sequence = [RUNNING] * 700 + [unknown()] * 70
    rig = Rig(monkeypatch, "sft", lambda call: sequence[call - 1], limit=1000)
    with pytest.raises(binding._InputAdmissionHeldError):
        rig.loop()
    assert rig.store.calls == 700 + CAP_POLLS
    assert rig.clock.now - 5000.0 == 15.0 * (700 + 60)


def test_success_is_unchanged_and_clears_the_run(monkeypatch):
    rig = Rig(monkeypatch, "data", [unknown(), unknown(), DATA_DONE])
    assert rig.read() is None and rig.read() is None
    assert set(rig.backend._precommit_unresolved) == {"use-1"}
    content = rig.read()
    assert type(content) is list and "verified_completed" in content[0]["text"]
    assert rig.backend._precommit_unresolved == {}


def test_terminal_failure_still_holds_at_once(monkeypatch):
    rig = Rig(monkeypatch, "data", [{"status": "FAILED", "success": False}])
    with pytest.raises(binding._InputAdmissionHeldError):
        rig.read()
    assert rig.backend._precommit_unresolved == {}


def test_logging_failure_cannot_change_the_closing_hold(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("ordinary diagnostic logger unavailable")

    monkeypatch.setattr(binding.logger, "warning", refuse)
    rig = Rig(monkeypatch, "sft", lambda call: unknown())
    with pytest.raises(binding._InputAdmissionHeldError):
        rig.loop()
    assert rig.store.calls == CAP_POLLS


@pytest.mark.parametrize(
    "arguments",
    [
        ("OTHER", 1, 0, "waiting"),
        ("SFT", 0, 0, "waiting"),
        ("SFT", True, 0, "waiting"),
        ("SFT", 1, -1, "waiting"),
        ("SFT", 1, 0, "done"),
        ("SFT", "1", 0, "waiting"),
    ],
)
def test_unknown_poll_helper_rejects_values_outside_the_closed_set(caplog, arguments):
    caplog.set_level(logging.INFO, logger=LOGGER)
    binding._warn_normal_unknown_poll(*arguments)
    assert not lines(caplog)


WARM_CASES = [
    ("data", "UNKNOWN"),
    ("preparation", "observation_failed"),
    ("sft", "UNKNOWN"),
    ("generation", "verification_failed"),
    ("case", "UNKNOWN"),
]


@pytest.mark.parametrize(("kind", "status"), WARM_CASES, ids=[f"{k}-{s}" for k, s in WARM_CASES])
def test_warm_producer_observation_closes_at_the_same_cap(monkeypatch, kind, status):
    """The API warm loop polls producers through `_observe_pending_producer_once` with the same gap."""
    rig = Rig(monkeypatch, kind, lambda call: unknown(status))
    monkeypatch.setattr(ToolBindingBackend, "_qualify_warm_producer", lambda self, pending: None)
    steps = []
    while True:
        step = rig.backend._observe_pending_producer_once()
        steps.append(step)
        if step.disposition != "awaiting_original":
            break
        rig.clock.now += CADENCE
    assert len(steps) == CAP_POLLS and rig.store.calls == CAP_POLLS
    assert step.disposition == "submission_unresolved" and step.turn is None
    assert step.diagnostic.as_data() == {
        "phase": "native_observation",
        "reason": "original_unknown_budget_exhausted",
        "error_class": "UNKNOWN",
    }
    assert rig.backend._warm_closed == "submission_unresolved"


def test_warm_producer_running_answer_resets_the_unresolved_run(monkeypatch):
    sequence = ([unknown()] * 50 + [RUNNING]) * 4
    rig = Rig(monkeypatch, "data", lambda call: sequence[call - 1])
    monkeypatch.setattr(ToolBindingBackend, "_qualify_warm_producer", lambda self, pending: None)
    for _ in sequence:
        assert rig.backend._observe_pending_producer_once().disposition == "awaiting_original"
        rig.clock.now += CADENCE
    assert rig.backend._warm_closed is None
