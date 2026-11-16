"""Neutral exact-source advisory reproductions; no native or hostile execution."""

from __future__ import annotations

import copy
import hashlib
import json
import re

import pytest
from test_campaign_admission import OWNER, fixture_guard, native_fields
from test_campaign_coordinator import (
    PROMPT,
    TRANSPORT,
    NativeJobs,
    fixture_coordinator,
    open_and_claim,
    original,
    wake,
)


def _replace_immutable(ddb, field, replacement):
    row = next(iter(ddb.items.values()))
    body = json.loads(row["private_record"]["B"])
    body["binding"][field] = replacement
    if field == "literal_prompt":
        body["binding"]["prompt_sha256"] = hashlib.sha256(replacement.encode()).hexdigest()
    raw = json.dumps(body["binding"], sort_keys=True, separators=(",", ":")).encode()
    body["binding_sha256"] = hashlib.sha256(raw).hexdigest()
    row["private_record"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize(
    ("field", "replacement", "action"),
    [
        ("campaign_id", "12345678-1234-4234-9234-00000000cc01", "load"),
        ("private_store_correlation", "12345678-1234-4234-9234-00000000cc02", "load"),
        ("literal_prompt", "A coherently changed neutral prompt.", "load"),
        ("campaign_id", "12345678-1234-4234-9234-00000000cc01", "reopen"),
        ("private_store_correlation", "12345678-1234-4234-9234-00000000cc02", "reopen"),
    ],
)
def test_original_commitment_refuses_coherent_immutable_replacement(field, replacement, action):
    cc, _, coordinator, guard, _, _, ddb, _ = fixture_coordinator()
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    _replace_immutable(ddb, field, replacement)
    with pytest.raises(cc.CampaignCoordinatorError):
        if action == "load":
            coordinator.load_campaign(guard.verify(), record.campaign_ref)
        else:
            coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)


def test_reference_contains_original_binding_commitment_not_only_mapping_selector():
    _, _, coordinator, guard, _, _, _, _ = fixture_coordinator()
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    assert re.fullmatch(r"campaign_[a-f0-9]{64}_[a-f0-9]{64}", record.campaign_ref)


def test_cold_transport_only_open_does_not_authenticate_its_own_row_hash():
    cc, _, coordinator, guard, _, clock, ddb, _ = fixture_coordinator()
    coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    cold = cc.DynamoCampaignCoordinator(guard, ddb, table_name="fixture-campaign-jobs", clock=clock)
    with pytest.raises(cc.CampaignCoordinatorError):
        cold.open_campaign(guard.verify(), TRANSPORT, PROMPT)


def test_inherited_coordinator_refuses_before_io_even_with_fresh_child_witness(monkeypatch):
    cc, _, coordinator, guard, reader, _, ddb, _, _, record, _ = open_and_claim()
    current = coordinator.load_campaign(guard.verify(), record.campaign_ref)
    pid = cc.os.getpid()
    monkeypatch.setattr(cc.os, "getpid", lambda: pid + 1)
    _, child_guard, _, _, _ = fixture_guard()
    fresh = child_guard.verify()
    prior_calls, prior_reads = len(ddb.calls), reader.calls
    before = copy.deepcopy(ddb.items)
    with pytest.raises(cc.CampaignCoordinatorError):
        coordinator.claim_lease(fresh, record.campaign_ref, expected_revision=current.revision)
    assert len(ddb.calls) == prior_calls
    assert reader.calls == prior_reads
    assert ddb.items == before


def test_guard_clock_highwater_refuses_rollback_inside_still_live_interval():
    ca, guard, _, clock, _ = fixture_guard()
    witness = guard.verify()
    clock.value = 1020
    guard.require(witness)
    clock.value = 1010
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)


def test_expired_witness_cannot_become_live_again_after_clock_rollback():
    ca, guard, _, clock, _ = fixture_guard()
    witness = guard.verify()
    clock.value = 1030
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)
    clock.value = 1029
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)


def test_native_reader_clock_rollback_inside_interval_refuses():
    ca, guard, reader, clock, _ = fixture_guard()
    witness = guard.verify()
    clock.value = 1020
    read = reader.read_current

    def rollback():
        observation = read()
        clock.value = 1010
        return observation

    reader.read_current = rollback
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)


@pytest.mark.parametrize("action", ["register", "observe"])
def test_exact_unknown_ack_can_resolve_readonly_after_lease_expiry(action):
    cc, _, coordinator, guard, _, clock, ddb, _, witness, record, lease = open_and_claim()
    operation = original(cc)
    if action == "observe":
        registered = coordinator.register_original(witness, lease, operation)
        lease = coordinator.claim_lease(
            guard.verify(), record.campaign_ref, expected_revision=registered.revision
        )
    real_update = ddb.update_item

    def lost_update(**kwargs):
        ddb.lose_update = True
        try:
            return real_update(**kwargs)
        finally:
            ddb.fail_reads = True

    ddb.update_item = lost_update
    observer = NativeJobs(cc)
    with pytest.raises(cc.CampaignUnresolved):
        if action == "register":
            coordinator.register_original(guard.verify(), lease, operation)
        else:
            coordinator.observe_wake(guard.verify(), lease, wake(cc, record, operation), observer)
    writes, observations = sum(name == "UpdateItem" for name, _ in ddb.calls), len(observer.calls)
    ddb.fail_reads = False
    ddb.update_item = real_update
    clock.value += 61
    fresh = guard.verify()
    loaded = coordinator.load_campaign(fresh, record.campaign_ref)
    assert sum(name == "UpdateItem" for name, _ in ddb.calls) == writes
    assert len(observer.calls) == observations
    new_lease = coordinator.claim_lease(fresh, record.campaign_ref, expected_revision=loaded.revision)
    assert new_lease.epoch > lease.epoch


@pytest.mark.parametrize("field", ["runtime_policy", "trust_policy"])
def test_finite_policy_parser_refuses_overflow_number_with_coherent_sha(field):
    ca, _, _, _, _ = fixture_guard()
    raw = b'{"x":1e999}'
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ca.AdmissionError):
        if field == "trust_policy":
            ca.RoleIncarnation(
                role_arn="arn:aws:iam::111122223333:role/fixture-owner",
                role_id="AROA" + "X" * 17,
                trust_policy=raw,
                trust_sha256=digest,
            )
        else:
            fields = native_fields(ca)
            fields["runtime_policy"], fields["runtime_policy_sha256"] = raw, digest
            ca.AdmissionManifest(**fields, owner_namespace=OWNER, change_control_reference="fixture-freeze")


def test_independently_retained_ref_supports_cold_open_and_refuses_legacy():
    cc, _, coordinator, guard, _, clock, ddb, _ = fixture_coordinator()
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    cold = cc.DynamoCampaignCoordinator(guard, ddb, table_name="fixture-campaign-jobs", clock=clock)
    restored = cold.open_campaign(
        guard.verify(), TRANSPORT, PROMPT, expected_campaign_ref=record.campaign_ref
    )
    assert (restored.campaign_id, restored.private_store_correlation) == (
        record.campaign_id,
        record.private_store_correlation,
    )
    before = len(ddb.calls)
    with pytest.raises(cc.CampaignCoordinatorError):
        cold.load_campaign(guard.verify(), record.campaign_ref.rsplit("_", 1)[0])
    assert len(ddb.calls) == before
    assert sum(name == "PutItem" for name, _ in ddb.calls) == 1


def test_original_ref_fences_cold_load_and_claim_after_binding_replacement():
    cc, _, coordinator, guard, _, clock, ddb, _ = fixture_coordinator()
    record = coordinator.open_campaign(guard.verify(), TRANSPORT, PROMPT)
    cold = cc.DynamoCampaignCoordinator(guard, ddb, table_name="fixture-campaign-jobs", clock=clock)
    _replace_immutable(ddb, "private_store_correlation", "12345678-1234-4234-9234-00000000cc22")
    before = copy.deepcopy(ddb.items)
    with pytest.raises(cc.CampaignCoordinatorError):
        cold.load_campaign(guard.verify(), record.campaign_ref)
    with pytest.raises(cc.CampaignCoordinatorError):
        cold.claim_lease(guard.verify(), record.campaign_ref, expected_revision=0)
    assert ddb.items == before


@pytest.mark.parametrize("failure", ["missing", "changed", "unreadable"])
def test_expired_readonly_resolution_keeps_unknown_until_exact_original_returns(failure):
    cc, _, coordinator, guard, _, clock, ddb, _, witness, record, lease = open_and_claim()
    operation = original(cc)
    real_update = ddb.update_item

    def lost_update(**kwargs):
        ddb.lose_update = True
        try:
            return real_update(**kwargs)
        finally:
            ddb.fail_reads = True

    ddb.update_item = lost_update
    with pytest.raises(cc.CampaignUnresolved):
        coordinator.register_original(witness, lease, operation)
    exact_items = copy.deepcopy(ddb.items)
    writes = sum(name == "UpdateItem" for name, _ in ddb.calls)
    ddb.update_item = real_update
    ddb.fail_reads = failure == "unreadable"
    if failure == "missing":
        ddb.items.clear()
    elif failure == "changed":
        row = next(iter(ddb.items.values()))
        body = json.loads(row["private_record"]["B"])
        body["revision"] += 1
        row["coordinator_revision"]["N"] = str(body["revision"])
        row["private_record"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    clock.value += 61
    with pytest.raises(cc.CampaignUnresolved):
        coordinator.load_campaign(guard.verify(), record.campaign_ref)
    ddb.items, ddb.fail_reads = exact_items, False
    loaded = coordinator.load_campaign(guard.verify(), record.campaign_ref)
    assert loaded.operations[0].original == operation
    assert sum(name == "UpdateItem" for name, _ in ddb.calls) == writes


def test_new_child_coordinator_cannot_take_a_live_parent_holder(monkeypatch):
    cc, _, coordinator, guard, _, clock, ddb, _, _, record, _ = open_and_claim()
    current = coordinator.load_campaign(guard.verify(), record.campaign_ref)
    pid = cc.os.getpid()
    monkeypatch.setattr(cc.os, "getpid", lambda: pid + 1)
    _, child_guard, _, _, _ = fixture_guard()
    child = cc.DynamoCampaignCoordinator(child_guard, ddb, table_name="fixture-campaign-jobs", clock=clock)
    writes = sum(name == "UpdateItem" for name, _ in ddb.calls)
    with pytest.raises(cc.CampaignCoordinatorError):
        child.claim_lease(child_guard.verify(), record.campaign_ref, expected_revision=current.revision)
    assert sum(name == "UpdateItem" for name, _ in ddb.calls) == writes


def test_inherited_guard_refuses_before_clock_lock_or_reader(monkeypatch):
    ca, guard, reader, _, _ = fixture_guard()
    pid = ca.os.getpid()
    calls = reader.calls

    class UnreachableInheritedLock:
        def __enter__(self):
            pytest.fail("inherited clock lock was reached before process refusal")

        def __exit__(self, *_):
            return False

    guard._clock_lock = UnreachableInheritedLock()
    monkeypatch.setattr(ca.os, "getpid", lambda: pid + 1)
    with pytest.raises(ca.AdmissionError):
        guard.verify()
    assert reader.calls == calls


def test_actual_fork_refuses_inherited_thread_locked_guard_within_bound():
    import os
    import select
    import signal
    import threading

    ca, guard, _, _, _ = fixture_guard()
    acquired, release = threading.Event(), threading.Event()

    def hold_parent_lock():
        with guard._clock_lock:
            acquired.set()
            release.wait(5)

    holder = threading.Thread(target=hold_parent_lock, daemon=True)
    holder.start()
    assert acquired.wait(1)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            guard.verify()
        except ca.AdmissionError:
            os.write(write_fd, b"refused")
            os._exit(0)
        except BaseException:
            os.write(write_fd, b"unexpected")
            os._exit(2)
        os.write(write_fd, b"accepted")
        os._exit(3)
    os.close(write_fd)
    ready, data, status = False, b"", None
    try:
        ready = bool(select.select([read_fd], [], [], 1)[0])
        if ready:
            data = os.read(read_fd, 32)
        else:
            os.kill(child, signal.SIGTERM)
        _, status = os.waitpid(child, 0)
    finally:
        os.close(read_fd)
        release.set()
        holder.join(1)
    assert ready, "inherited guard deadlocked beyond the one-second neutral refusal bound"
    assert data == b"refused"
    assert os.waitstatus_to_exitcode(status) == 0


def test_independent_guard_construction_in_child_process_is_supported(monkeypatch):
    ca, original_guard, _, _, _ = fixture_guard()
    pid = ca.os.getpid()
    monkeypatch.setattr(ca.os, "getpid", lambda: pid + 1)
    _, fresh_guard, fresh_reader, _, _ = fixture_guard()
    witness = fresh_guard.verify()
    fresh_guard.require(witness)
    assert fresh_reader.calls == 2
    assert witness.process_id == pid + 1
    assert fresh_guard is not original_guard
