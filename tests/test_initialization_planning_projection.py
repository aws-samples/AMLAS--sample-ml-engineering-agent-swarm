"""Original C/H commitment DATA uses qualified native-shaped neutral reads only."""

import dataclasses
import hashlib
import json

import pytest
from test_campaign_initialization import PROMPT, TRANSPORT, fixture_initializer, split_items


def accessor(initializer):
    method = getattr(initializer, "resolve_original_planning", None)
    assert callable(method), "missing-symbol: DynamoCampaignInitialization.resolve_original_planning"
    return method


def test_qualified_projection_carries_original_h_commitments_after_c_revision_advance():
    ci, _, initializer, guard, _, _, tx, *_ = fixture_initializer()
    first = initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    c, h = split_items(tx)
    original_h = bytes(h["private_record"]["B"])
    expected = json.loads(original_h)["initial_campaign_body_sha256"]
    changed = json.loads(c["private_record"]["B"])
    changed["revision"] = 1
    c["coordinator_revision"]["N"] = "1"
    c["private_record"]["B"] = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
    before = len(tx.calls)
    calls = []
    answer = accessor(initializer)(
        guard.verify(), TRANSPORT, literal_prompt=PROMPT, qualify_io=lambda: calls.append("current")
    )
    assert type(answer) is ci.InitializationPlanningObservation
    assert answer.original.campaign_ref == first.campaign_ref and answer.original.current_revision == 1
    assert answer.initial_campaign_body_sha256 == expected
    assert answer.initialization_publication_sha256 == hashlib.sha256(original_h).hexdigest()
    assert answer.initial_campaign_body_sha256 != hashlib.sha256(c["private_record"]["B"]).hexdigest()
    assert answer.native_admission is False and answer.action_approved is False
    assert calls and [name for name, _ in tx.calls[before:]] == ["TransactGetItems"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.initial_campaign_body_sha256 = "d" * 64


@pytest.mark.parametrize(
    "mode", ["missing_h", "h_digest", "changed_prompt", "revoked_before", "revoked_after", "inherited_pid"]
)
def test_projection_refuses_original_or_current_drift_without_any_write(monkeypatch, mode):
    _, _, initializer, guard, _, _, tx, *_ = fixture_initializer()
    initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    method = accessor(initializer)
    _, h = split_items(tx)
    if mode == "missing_h":
        tx.items.pop(h["job_name"]["S"])
    elif mode == "h_digest":
        body = json.loads(h["private_record"]["B"])
        body["initial_campaign_body_sha256"] = "d" * 64
        h["private_record"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    elif mode == "inherited_pid":
        monkeypatch.setattr(initializer, "_process_id", initializer._process_id + 1)
    reads = {"seen": 0}

    def current():
        if mode == "revoked_before" or (mode == "revoked_after" and reads["seen"]):
            raise ValueError("neutral private current qualification revoked")

    original = tx.transact_get_items

    def read(**kwargs):
        answer = original(**kwargs)
        reads["seen"] += 1
        return answer

    monkeypatch.setattr(tx, "transact_get_items", read)
    before = len(tx.calls)
    with pytest.raises(ValueError):
        method(
            guard.verify(),
            TRANSPORT,
            literal_prompt="Changed original" if mode == "changed_prompt" else PROMPT,
            qualify_io=current,
        )
    assert all(name == "TransactGetItems" for name, _ in tx.calls[before:])
    if mode in {"revoked_before", "inherited_pid"}:
        assert tx.calls[before:] == []
