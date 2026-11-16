"""Durable pre-invoke ownership over strict inert service-shaped row ports.

These tests exercise real row codec/CAS/claim methods, with explicit native and
original-qualification seams. They grant no live SDK, native job or recovery
admission. Legacy ambiguous ownership must remain closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_normal_checkpoint_cas_contract import TABLE_ID, StrictDynamoPort, _item, _policy
from test_normal_completion_claim_receipt import _declared_claim

from amlas.control_plane import normal_run_checkpoint as checkpoint


class AttemptPort(StrictDynamoPort):
    def __init__(self, context, item):
        super().__init__(context, item)
        self.foreign_attempt_after_lost_ack = False
        self.attempts = []

    def update_item(self, **kwargs):
        candidate = json.loads(kwargs["ExpressionAttributeValues"][":body"]["S"])
        continuation = candidate.get("continuation", {})
        if continuation.get("invocation_state") == "invoking":
            with self.lock:
                self.attempts.append(copy.deepcopy(continuation))
        try:
            return super().update_item(**kwargs)
        except ConnectionError:
            if self.foreign_attempt_after_lost_ack:
                with self.lock:
                    changed = json.loads(self.item["record"]["S"])
                    owned = changed["continuation"]
                    owned["invocation_token"] = (
                        "e" * 32 if owned["invocation_token"] != "e" * 32 else "f" * 32
                    )
                    self.item = _item(self.context, changed)
            raise


@pytest.fixture
def declared(monkeypatch):
    context, record, stored, ready = _declared_claim()
    contexts = [context]
    port = AttemptPort(context, _item(context, record))
    qualifiers = []

    @contextmanager
    def native(value):
        assert any(value is candidate for candidate in contexts)
        yield port, TABLE_ID, lambda: None, None

    def original(value, row, table_id, job_name=None):
        assert any(value is candidate for candidate in contexts) and table_id == TABLE_ID
        if row["sft"]["operation_digest"] != stored["operation_digest"]:
            raise ValueError("Declared original digest changed.")
        if job_name is not None and job_name != ready["native_job_name"]:
            raise ValueError("Declared original job changed.")
        return SimpleNamespace(operation_digest=stored["operation_digest"]), None

    def immutable_task(value, binding, ref, kind, qualify):
        assert any(value is candidate for candidate in contexts) and kind == "tasks"
        if ref.version_id != record["task_checkpoint_ref"]["version_id"]:
            raise ValueError("Declared immutable TASK version changed.")
        qualify()
        return checkpoint._canonical(stored).encode()

    def qualify_original():
        qualifiers.append("fresh-declared-original")

    def fresh():
        replacement = checkpoint.NormalRunContext(
            context.catalog,
            context.run_id,
            context.session_hash,
            transport_session_id=context.transport_session_id,
        )
        assert replacement._completion_receipt is None
        assert replacement._completion_attempted_token is None
        contexts.append(replacement)
        return replacement

    monkeypatch.setattr(checkpoint, "_native", native)
    monkeypatch.setattr(checkpoint, "_original", original)
    monkeypatch.setattr(checkpoint, "_read_object", immutable_task)
    return SimpleNamespace(
        context=context,
        port=port,
        record=record,
        stored=stored,
        ready=ready,
        original=qualify_original,
        fresh=fresh,
        qualifiers=qualifiers,
    )


def claim(state, context=None):
    return (state.context if context is None else context).claim_completion(
        copy.deepcopy(state.ready),
        policy=_policy(),
        qualify_original=state.original,
    )


def begin(state, context, token, policy=None):
    return context.begin_completion_invocation(
        token,
        policy=_policy() if policy is None else policy,
        qualify_original=state.original,
        sdk_snapshot=copy.deepcopy(state.ready["sdk_snapshot"]),
    )


def qualify(state, context, token):
    context.qualify_completion(
        token,
        policy=_policy(),
        qualify_original=state.original,
        sdk_snapshot=copy.deepcopy(state.ready["sdk_snapshot"]),
    )


def row(state):
    return json.loads(state.port.item["record"]["S"])


def test_new_claim_distinguishes_ready_from_possible_invocation(declared):
    token = claim(declared)
    data = row(declared)["continuation"]
    assert data["schema_version"] == 3
    assert data["invocation_state"] == "ready" and data["invocation_token"] is None
    assert data["epoch"] == 1 and data["token"] == token
    expected_snapshot = checkpoint._completed_task_snapshot(
        declared.stored, declared.stored["operation_digest"]
    )
    assert expected_snapshot == declared.ready["sdk_snapshot"]
    assert expected_snapshot is not declared.ready["sdk_snapshot"]
    assert (
        data["sdk_snapshot_sha256"]
        == hashlib.sha256(checkpoint._canonical(expected_snapshot).encode()).hexdigest()
    )
    assert declared.port.commits == 1


def test_fresh_context_reconstructs_ready_claim_without_new_mutation(declared):
    token = claim(declared)
    initial = copy.deepcopy(declared.port.item)
    replacement = declared.fresh()
    assert claim(declared, replacement) == token
    qualify(declared, replacement, token)
    assert replacement._completion_receipt == declared.context._completion_receipt
    assert declared.port.item == initial and declared.port.commits == 1
    assert declared.qualifiers


def test_begin_persists_unique_invoking_before_admitting_attempt(declared):
    token = claim(declared)
    assert begin(declared, declared.context, token) is None
    owned = row(declared)["continuation"]
    assert owned["schema_version"] == 3 and owned["invocation_state"] == "invoking"
    assert (
        owned["sdk_snapshot_sha256"]
        == hashlib.sha256(checkpoint._canonical(declared.ready["sdk_snapshot"]).encode()).hexdigest()
    )
    assert owned["epoch"] == 2 and owned["token"] == token
    assert type(owned["invocation_token"]) is str and len(owned["invocation_token"]) == 32
    assert owned["invocation_token"] != token
    assert declared.context._completion_attempted_token == token
    assert declared.context._completion_receipt[1] == checkpoint._canonical(owned)
    qualify(declared, declared.context, token)
    assert declared.port.commits == 2
    after = copy.deepcopy(declared.port.item)
    with pytest.raises(ValueError):
        begin(declared, declared.context, token)
    assert declared.port.item == after and declared.port.commits == 2


def test_independent_contexts_cas_distinct_attempts_and_only_one_wins(declared):
    token = claim(declared)
    first, second = declared.fresh(), declared.fresh()
    assert claim(declared, first) == claim(declared, second) == token
    declared.port.barrier = threading.Barrier(2)

    def attempt(context):
        try:
            begin(declared, context, token)
        except ValueError:
            return "refused"
        return "began"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(attempt, context) for context in (first, second)]
        outcomes = [future.result(timeout=15) for future in results]
    assert sorted(outcomes) == ["began", "refused"]
    assert declared.port.commits == 2 and declared.port.update_calls == 3
    assert len(declared.port.attempts) == 2
    assert len({item["invocation_token"] for item in declared.port.attempts}) == 2
    winner = first if outcomes[0] == "began" else second
    loser = second if winner is first else first
    qualify(declared, winner, token)
    with pytest.raises(ValueError):
        qualify(declared, loser, token)


def test_stale_ready_receipt_cannot_start_or_qualify_after_other_owner(declared):
    token = claim(declared)
    replacement = declared.fresh()
    assert claim(declared, replacement) == token
    begin(declared, replacement, token)
    after = copy.deepcopy(declared.port.item)
    for operation in (qualify, begin):
        with pytest.raises(ValueError):
            operation(declared, declared.context, token)
    assert declared.port.item == after and declared.port.commits == 2


def test_invoking_loss_refuses_reconstruction_and_keeps_original_row(declared):
    token = claim(declared)
    begin(declared, declared.context, token)
    after = copy.deepcopy(declared.port.item)
    replacement = declared.fresh()
    with pytest.raises(ValueError):
        claim(declared, replacement)
    assert replacement._completion_receipt is None
    assert replacement._completion_attempted_token is None
    assert declared.port.item == after and declared.port.commits == 2


@pytest.mark.parametrize("foreign", [False, True], ids=["exact-attempt", "different-attempt"])
def test_lost_ack_resolves_only_this_unique_attempt(declared, foreign):
    token = claim(declared)
    declared.port.lost_ack = True
    declared.port.foreign_attempt_after_lost_ack = foreign
    if foreign:
        with pytest.raises(ValueError):
            begin(declared, declared.context, token)
        with pytest.raises(ValueError):
            qualify(declared, declared.context, token)
    else:
        begin(declared, declared.context, token)
        qualify(declared, declared.context, token)
    assert declared.port.commits == 2 and declared.port.update_calls == 2
    assert declared.context._completion_attempted_token == token
    before_retry = copy.deepcopy(declared.port.item)
    with pytest.raises(ValueError):
        begin(declared, declared.context, token)
    assert declared.port.item == before_retry and declared.port.update_calls == 2


def test_denied_start_leaves_ready_unspent(declared):
    token = claim(declared)
    before = copy.deepcopy(declared.port.item)
    with pytest.raises(ValueError):
        begin(declared, declared.context, token, policy=_policy(False))
    assert declared.context._completion_attempted_token is None
    assert declared.port.item == before and declared.port.commits == 1


def test_legacy_consumed_claim_does_not_gain_cold_recovery(declared):
    claim(declared)
    legacy = row(declared)
    owned = legacy["continuation"]
    owned["schema_version"] = 1
    owned.pop("sdk_snapshot_sha256")
    owned["sdk_snapshot"] = copy.deepcopy(declared.ready["sdk_snapshot"])
    owned.pop("invocation_state", None)
    owned.pop("invocation_token", None)
    declared.port.item = _item(declared.context, legacy)
    _decoded_item, decoded = checkpoint._load(declared.context, declared.port, TABLE_ID, lambda: None)
    assert decoded == legacy
    before = copy.deepcopy(declared.port.item)
    replacement = declared.fresh()
    with pytest.raises(ValueError):
        claim(declared, replacement)
    assert replacement._completion_receipt is None
    assert declared.port.item == before and declared.port.commits == 1


@pytest.mark.parametrize(
    "fault",
    [
        "missing-state",
        "missing-attempt",
        "ready-attempt",
        "ready-epoch",
        "invoking-without-attempt",
        "invoking-epoch",
        "unknown-state",
        "extra-state",
    ],
)
def test_malformed_new_schema_cannot_be_read_as_ready(declared, fault):
    claim(declared)
    altered = row(declared)
    owned = altered["continuation"]
    owned.update(schema_version=2, invocation_state="ready", invocation_token=None)
    if fault == "missing-state":
        owned.pop("invocation_state")
    elif fault == "missing-attempt":
        owned.pop("invocation_token")
    elif fault == "ready-attempt":
        owned["invocation_token"] = "c" * 32
    elif fault == "ready-epoch":
        owned["epoch"] = 2
    elif fault == "invoking-without-attempt":
        owned.update(invocation_state="invoking", epoch=2)
    elif fault == "invoking-epoch":
        owned.update(invocation_state="invoking", invocation_token="c" * 32)
    elif fault == "unknown-state":
        owned["invocation_state"] = "unknown"
    else:
        owned["extra"] = False
    declared.port.item = _item(declared.context, altered)
    before = copy.deepcopy(declared.port.item)
    with pytest.raises(ValueError):
        checkpoint._load(declared.context, declared.port, TABLE_ID, lambda: None)
    assert declared.port.item == before and declared.port.commits == 1
