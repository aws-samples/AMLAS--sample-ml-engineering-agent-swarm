"""The process record store: bounded, single-flight, fork-reset, and a discard cannot be undone by an older pass."""

from __future__ import annotations

import threading

import pytest

from amlas.backends.adapters.process_pins import ProcessRecords, fingerprint, reusable_fingerprint

HEAD = {
    "VersionId": "v1",
    "ContentLength": 7,
    "ETag": '"abc"',
    "ChecksumType": "FULL_OBJECT",
    "ChecksumCRC64NVME": "Y3JjNjQ=",
}


def test_a_record_needs_matching_etags_and_a_checksum_field():
    assert reusable_fingerprint(HEAD, {"ETag": '"abc"'}) == fingerprint(HEAD)
    assert reusable_fingerprint(HEAD, {"ETag": '"other"'}) is None
    assert reusable_fingerprint(HEAD, {}) is None
    assert reusable_fingerprint({**HEAD, "ETag": ""}, {"ETag": ""}) is None
    assert (
        reusable_fingerprint({k: v for k, v in HEAD.items() if k != "ChecksumCRC64NVME"}, {"ETag": '"abc"'})
        is None
    )
    assert reusable_fingerprint(None, {}) is None


def test_the_fingerprint_marks_an_absent_field_as_different_from_a_present_one():
    shorter = {k: v for k, v in HEAD.items() if k != "ETag"}
    assert fingerprint(shorter) != fingerprint(HEAD)


def test_records_are_bounded_and_the_oldest_goes_first():
    records = ProcessRecords(limit=3)
    for number in range(5):
        records.put(number, ("pin", number))
    assert [records.get(n) for n in range(5)] == [None, None, ("pin", 2), ("pin", 3), ("pin", 4)]


def test_a_discard_cannot_be_undone_by_a_pass_that_began_before_it():
    records = ProcessRecords()
    epoch = records.epoch("key")
    records.forget("key")
    assert records.put("key", "stale", epoch) is False and records.get("key") is None
    assert records.put("key", "fresh", records.epoch("key")) is True and records.get("key") == "fresh"


def test_the_first_asker_leads_and_a_second_waits_for_it():
    records = ProcessRecords()
    flight, leader = records.claim("key")
    again, second = records.claim("key")
    assert leader is True and second is False and again is flight
    seen = []
    waiter = threading.Thread(
        target=lambda: (records.wait(flight, lambda: seen.append("check"), tick=0.05), seen.append("done"))
    )
    waiter.start()
    waiter.join(0.3)
    assert waiter.is_alive() and "check" in seen
    records.release("key", flight)
    waiter.join(5)
    assert seen[-1] == "done"
    assert records.claim("key")[1] is True, "a released key can be claimed again"


def test_a_failing_check_stops_the_wait():
    records = ProcessRecords()
    flight, _leader = records.claim("key")

    def refuse():
        raise TimeoutError("deadline")

    holder = {}

    def run():
        try:
            records.wait(flight, refuse, tick=0.05)
        except TimeoutError as error:
            holder["error"] = error

    other = threading.Thread(target=run)
    other.start()
    other.join(5)
    assert "error" in holder
    records.release("key", flight)


def test_a_thread_cannot_wait_for_its_own_read():
    records = ProcessRecords()
    flight, _leader = records.claim("key")
    with pytest.raises(ValueError, match="own read"):
        records.wait(flight, lambda: None)
    records.release("key", flight)


def test_reset_empties_records_flights_and_epochs_as_a_forked_child_does():
    records = ProcessRecords()
    records.put("key", "pin")
    records.forget("other")
    flight, _leader = records.claim("busy")
    records.reset()
    assert records.get("key") is None and records.epoch("other") == 0
    assert records.claim("busy")[1] is True
    assert flight.done.is_set() is False
