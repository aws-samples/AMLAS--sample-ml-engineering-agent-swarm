"""Process-scoped records of exact S3 object versions whose bytes this process has read and hashed in full.

S3 carries no SHA-256 for the adopted model pack (staged copies hold a CRC64NVME, originals a composite CRC32C), so each
unpinned verification is a real read of every byte: 17.99 GiB, 8 to 17 minutes. The first hash of an object in a process
stays a full read. It leaves a record: the exact version and the HEAD fingerprint (ETag, checksums, storage class) of the
read. A later pass in the same process asks S3 for the HEAD of that exact version and compares it with the record; a
missing record, a changed fingerprint or any other difference takes the full read again, and a failing HEAD fails the
pass. The record certifies bytes only. Authority, identity, scope, namespace and policy checks run on every pass.

One thread per key hashes at a time (`claim`); another thread asking for the same key waits on the leader, running its own
deadline and qualifier checks, and then makes its own HEAD. A record is a fingerprint tuple (immutable); nothing mutable or
credential-bearing is kept. A forked child starts empty. After the first full hash in a process a later silent S3
corruption is not detected until the next process.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

FINGERPRINT_FIELDS = (
    "VersionId",
    "ContentLength",
    "ETag",
    "ChecksumType",
    "ChecksumSHA256",
    "ChecksumSHA1",
    "ChecksumCRC32",
    "ChecksumCRC32C",
    "ChecksumCRC64NVME",
    "StorageClass",
    "ArchiveStatus",
)
_CHECKSUM_FIELDS = (
    "ChecksumSHA256",
    "ChecksumSHA1",
    "ChecksumCRC32",
    "ChecksumCRC32C",
    "ChecksumCRC64NVME",
)
_LIMIT = 4096


def fingerprint(head: dict) -> tuple:
    """The HEAD fields that cannot change for one immutable object version; an absent field differs from a present one."""
    return tuple((field, head[field]) for field in FINGERPRINT_FIELDS if field in head)


def reusable_fingerprint(head: Any, result: Any) -> tuple | None:
    """The fingerprint of a read that may become a record, else None.

    The HEAD and the GET of the hashed bytes must both name the same ETag, and the HEAD must carry a checksum, so the
    record is tied to the response whose bytes were hashed and to a content-bound field.
    """
    if type(head) is not dict or type(result) is not dict:
        return None
    etag = head.get("ETag")
    if type(etag) is not str or not etag or result.get("ETag") != etag:
        return None
    if not any(type(head.get(name)) is str and head[name] for name in _CHECKSUM_FIELDS):
        return None
    return fingerprint(head)


class Flight:
    """The claim of one key by the thread that is reading it."""

    __slots__ = ("done", "owner")

    def __init__(self) -> None:
        self.owner = threading.get_ident()
        self.done = threading.Event()


class ProcessRecords:
    """Bounded, thread-safe records with single-flight claims; a discard cannot be undone by a pass that began before it."""

    def __init__(self, limit: int = _LIMIT) -> None:
        self._limit = limit
        self.reset()
        os.register_at_fork(after_in_child=self.reset)

    def reset(self) -> None:
        # Also the fork hook: a lock held by a thread that does not exist in the child is replaced, not released.
        self._lock = threading.Lock()
        self._records: OrderedDict[Any, Any] = OrderedDict()
        self._flights: dict[Any, Flight] = {}
        self._epochs: dict[Any, int] = {}

    def get(self, key: Any) -> Any:
        with self._lock:
            value = self._records.get(key)
            if value is not None:
                self._records.move_to_end(key)
            return value

    def epoch(self, key: Any) -> int:
        with self._lock:
            return self._epochs.get(key, 0)

    def put(self, key: Any, value: Any, epoch: int | None = None) -> bool:
        """Store a record; with `epoch` only when the key was not discarded since that epoch was read."""
        with self._lock:
            if epoch is not None and self._epochs.get(key, 0) != epoch:
                return False
            self._records[key] = value
            self._records.move_to_end(key)
            while len(self._records) > self._limit:
                self._records.popitem(last=False)
            return True

    def forget(self, key: Any) -> None:
        with self._lock:
            self._records.pop(key, None)
            self._epochs[key] = self._epochs.get(key, 0) + 1
            if len(self._epochs) > 4 * self._limit:
                self._epochs = {
                    k: v for k, v in self._epochs.items() if k in self._records or k in self._flights
                }

    def claim(self, key: Any) -> tuple[Flight, bool]:
        """(flight, True) for the first asker, which must `release` it; (flight, False) for a later one."""
        with self._lock:
            flight = self._flights.get(key)
            if flight is not None:
                return flight, False
            flight = self._flights[key] = Flight()
            return flight, True

    def release(self, key: Any, flight: Flight) -> None:
        with self._lock:
            if self._flights.get(key) is flight:
                del self._flights[key]
        flight.done.set()

    @staticmethod
    def wait(flight: Flight, check: Any, tick: float = 1.0) -> None:
        """Wait for the leader of a key, running the caller's own deadline and qualifier checks every `tick` seconds."""
        if flight.owner == threading.get_ident():
            raise ValueError("a pass cannot wait for its own read")
        while not flight.done.wait(tick):
            check()
        check()
