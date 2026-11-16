"""Inert connected fixtures isolate pacing history; real pacing stays intact."""

from __future__ import annotations

import threading
import time

import model_acquisition_fixtures as f
from test_runtime_connected_warm_model import connected_work
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.backends.adapters import _util


def test_connected_fixture_does_not_inherit_prior_synthetic_account_pacing(monkeypatch):
    key = (f.ACCOUNT, f.REGION)
    foreign = ("9" * 12, f.REGION)
    prior_state = [threading.Lock(), time.monotonic()]
    foreign_state = [threading.Lock(), None]
    prior = {key: prior_state, foreign: foreign_state}
    monkeypatch.setattr(_util, "_PROCESSING_CREATE_STATES", prior)
    interval, clock, sleep, lock, pid = (
        _util._PROCESSING_CREATE_INTERVAL,
        _util._processing_create_now,
        _util._processing_create_sleep,
        _util._PROCESSING_CREATE_LOCK,
        _util._PROCESSING_CREATE_PID,
    )
    work = connected_work(monkeypatch)
    assert _util._PROCESSING_CREATE_STATES is not prior, "fixture reused other tests' pacing map"
    assert key not in _util._PROCESSING_CREATE_STATES, "fixture inherited prior synthetic pacing"
    assert _util._PROCESSING_CREATE_STATES[foreign] is foreign_state
    assert prior[key] is prior_state and prior[foreign] is foreign_state
    assert _util._PROCESSING_CREATE_INTERVAL == interval == 1.25
    assert _util._processing_create_now is clock and _util._processing_create_sleep is sleep
    assert _util._PROCESSING_CREATE_LOCK is lock and _util._PROCESSING_CREATE_PID == pid
    assert work.native.requests == [] and work.engines == [] and work.agents == []


def test_connected_fixture_keeps_real_intra_scope_spacing_and_restores_prior_map(monkeypatch):
    key = (f.ACCOUNT, f.REGION)
    foreign = ("9" * 12, f.REGION)
    prior_state, foreign_state = [threading.Lock(), None], [threading.Lock(), None]
    prior = {key: prior_state, foreign: foreign_state}
    monkeypatch.setattr(_util, "_PROCESSING_CREATE_STATES", prior)
    effects = []
    with monkeypatch.context() as local:
        work = connected_work(local)
        with _util._initial_processing_create(*key):
            effects.append(time.monotonic())
        with _util._initial_processing_create(*key):
            effects.append(time.monotonic())
        assert effects[1] - effects[0] >= 1.25, "fixture bypassed actual completion-based pacing"
        assert _util._PROCESSING_CREATE_STATES[foreign] is foreign_state
        assert work.native.requests == [] and work.engines == [] and work.agents == []
    assert _util._PROCESSING_CREATE_STATES is prior
    assert prior[key] is prior_state and prior_state[1] is None, "fixture changed prior test pacing"
    assert prior[foreign] is foreign_state and foreign_state[1] is None
