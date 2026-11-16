"""One-shot begin ABI over fresh declared claim DATA; no invocation authority.

Actual completion qualification and row parsing remain. Inert native/original
seams grant no real ownership, guard, native, inference or cold-recovery proof.
"""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from test_normal_checkpoint_cas_contract import TABLE_ID, _item, _policy
from test_normal_completion_record_codec import DeclaredGetPort, _declared_sft_record

from amlas.control_plane import normal_run_checkpoint as checkpoint


def _declared_capability(monkeypatch):
    context, record = _declared_sft_record()
    claim = record["continuation"]
    snapshot = copy.deepcopy(claim["sdk_snapshot"])
    object.__setattr__(
        context,
        "_completion_receipt",
        (
            claim["token"],
            checkpoint._canonical(claim),
            checkpoint._canonical(record["task_checkpoint_ref"]),
        ),
    )
    port = DeclaredGetPort(context, _item(context, record))
    original_reads = []

    @contextmanager
    def declared_native(value):
        assert value is context
        yield port, TABLE_ID, lambda: None, None

    def declared_original(value, _record, table_id):
        assert value is context and table_id == TABLE_ID
        original_reads.append("declared-original-boundary")
        return None

    def no_original_grant():
        return None

    monkeypatch.setattr(checkpoint, "_native", declared_native)
    monkeypatch.setattr(checkpoint, "_original", declared_original)
    return context, claim["token"], snapshot, no_original_grant, port, original_reads


def _begin(context, token, snapshot, original, policy=None):
    return context.begin_completion_invocation(
        token,
        policy=_policy() if policy is None else policy,
        qualify_original=original,
        sdk_snapshot=snapshot,
    )


def test_successful_begin_is_not_reusable_after_external_invocation_failure(monkeypatch):
    context, token, snapshot, original, port, original_reads = _declared_capability(monkeypatch)
    assert getattr(context, "_completion_attempted_token", None) is None
    assert _begin(context, token, snapshot, original) is None
    assert context._completion_attempted_token == token
    assert port.calls >= 1 and original_reads
    invocations = []

    def declared_invocation():
        invocations.append("attempt")
        raise RuntimeError("Declared external invocation failure.")

    with pytest.raises(RuntimeError):
        declared_invocation()
    with pytest.raises(ValueError):
        _begin(context, token, snapshot, original)
    assert invocations == ["attempt"] and context._completion_attempted_token == token


def test_simultaneous_begin_callers_allow_only_one_attempt(monkeypatch):
    context, token, snapshot, original, port, original_reads = _declared_capability(monkeypatch)
    barrier = threading.Barrier(2)

    def attempt():
        barrier.wait(timeout=5)
        try:
            _begin(context, token, snapshot, original)
        except ValueError:
            return "refused"
        return "began"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt) for _ in range(2)]
        outcomes = [future.result(timeout=10) for future in futures]
    assert sorted(outcomes) == ["began", "refused"]
    assert context._completion_attempted_token == token
    assert port.calls >= 1 and original_reads


def test_prequalification_refusal_cannot_mark_attempt_or_invoke(monkeypatch):
    context, token, snapshot, original, port, original_reads = _declared_capability(monkeypatch)
    invocations = []
    with pytest.raises(ValueError):
        _begin(context, token, snapshot, original, policy=_policy(False))
        invocations.append("attempt")
    assert getattr(context, "_completion_attempted_token", None) is None
    assert invocations == [] and port.calls == 0 and original_reads == []
