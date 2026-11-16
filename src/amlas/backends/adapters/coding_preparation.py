"""Fixed admitted Processing preparation; raw exact-version I/O, no dataset parser."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import Counter, OrderedDict
from contextlib import closing, contextmanager
from contextvars import ContextVar
from typing import Any

from ...control_plane.preparation_state import (
    PreparationCatalog,
    VerifiedPreparationSource,
    validate_catalog,
    validate_source,
)
from ...sagemaker_entrypoints.coding_preparation import (
    HEADER,
    OUTPUT_FILES,
    POLICY_VERSION,
    PRIVILEGE_PROFILE,
    PROFILE_ID,
    decode_object,
)
from ..types import JobHandle, JobState, JobStatus
from .artifact_rejection import ImmutableArtifactRejectionError, rejection_data

MAX_BYTES = 512 * 1024 * 1024
METADATA_LIMIT = 2 * 1024 * 1024
DERIVED_PROFILE_ID = "nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2"
DERIVED_POLICY_VERSION = "coding-preparation.open-swe.v2"
DERIVED_LINKAGE_STATUS = "derived_not_original_source_verified"
MAX_SOURCE_ROWS = 100000
LINEAGE_RECORD_LIMIT = 32 * 1024 * 1024
LINEAGE_READ_BYTES = 1024 * 1024  # one read bracket per MiB; a 184 MB lineage object is about 175 brackets
BODY_SECONDS = 120.0
QUALIFIER_CREDIT_CEILING_SECONDS = 1800.0
REPLAY_MEMO_ENTRIES = (
    2  # verified CommitPack replay results held per process (about 7 MB each at 56,700 rows)
)
REPLAY_MEMO_WAIT_SECONDS = (
    300.0  # longest a second caller waits for the first one's replay before computing alone
)
_LOG = logging.getLogger(__name__)
# Fixed wording of this module's own refusals plus the standard closed-stream message; nothing else is named.
_REASONS = frozenset(
    {
        "CommitPack actual output line count differs",
        "CommitPack committed header differs",
        "CommitPack committed inventory differs",
        "CommitPack exclusions are not exact canonical staged bytes",
        "CommitPack file descriptor differs",
        "CommitPack file manifest differs",
        "CommitPack guest observations differ",
        "CommitPack independently replayed report differs",
        "CommitPack output inventory truncated",
        "CommitPack output row bound exceeded",
        "CommitPack output type/hash differs",
        "CommitPack report missing",
        "CommitPack report record count differs",
        "acquisition manifest version differs",
        "closed preparation profile/policy pair required",
        "derived core lineage decision differs",
        "derived edge envelope differs",
        "derived edge schema differs",
        "derived lineage lacks original private sources",
        "derived lineage normalization differs",
        "derived lineage original provenance differs",
        "derived lineage original row repeats/differs",
        "derived lineage original source differs",
        "derived lineage selection bound exceeded",
        "derived lineage source selection differs",
        "derived lineage source/ordinal attribution differs",
        "derived lineage status/schema differs",
        "derived original row ordinals are incomplete",
        "derived original-row lineage conservation differs",
        "derived qualified linkage differs",
        "empty lineage metadata row",
        "exact preparation object identity changed",
        "expired or foreign-thread processing observation refused",
        "fixed pinned exclusion schema required",
        "foreign preparation object",
        "inherited processing observation refused",
        "inherited processing observation scope refused",
        "invalid native preparation I/O entries",
        "invalid native preparation I/O entry",
        "invalid native preparation output config",
        "invalid preparation count",
        "lineage exact-version identity differs",
        "lineage metadata byte count differs",
        "lineage metadata deadline exceeded",
        "lineage metadata row bound exceeded",
        "lineage metadata termination/hash differs",
        "mixed preparation profiles are not supported",
        "native preparation AppManaged differs",
        "native preparation environment drifted",
        "native preparation identity drifted",
        "native preparation request drifted",
        "non-null native object version required",
        "normalization refusal counts differ",
        "normalization refusal label differs",
        "normalization refusal lineage differs",
        "normalization/core drop conservation differs",
        "normalized canonical digest/identity differs",
        "preparation actual lineage/membership count differs",
        "preparation actual record count differs",
        "preparation actual split count differs",
        "preparation committed output inventory differs",
        "preparation committed provenance differs",
        "preparation committed schema/profile differs",
        "preparation count conservation differs",
        "preparation count schema differs",
        "preparation coverage claim differs",
        "preparation descriptive guest evidence differs",
        "preparation diagnostic label differs",
        "preparation drop/detector schema differs",
        "preparation exclusive drop counts differ",
        "preparation file binding differs",
        "preparation file manifest differs",
        "preparation file type/hash differs",
        "preparation object exceeds committed count",
        "preparation output byte count differs",
        "preparation raw I/O bound exceeded",
        "preparation raw I/O deadline exceeded",
        "preparation report differs from commit",
        "preparation requires selected same-region versioned bucket",
        "preparation selection manifest bound exceeded",
        "preparation split membership differs",
        "preparation split schema differs",
        "private CommitPack source linecounts required",
        "private bounded preparation selection required",
        "private preparation handle required",
        "private preparation source required",
        "private preparation sources required",
        "private processing observation qualifier required",
        "recognized CommitPack source cannot fall back",
        "unconfined preparation object",
        "unsupported preparation source format",
        "I/O operation on closed file",
        "I/O operation on closed file.",
    }
)


class _BodyBudget:
    """Wall-clock allowance of one returned body.

    The allowance bounds stream, decode and consumer work. Time spent inside the caller's qualifier while the
    body is open is admission work, so it is credited back, up to a fixed ceiling per body, and that includes the
    call that is still running (the close timer reads the allowance while a call is in flight). Every
    qualification call site and check is unchanged; only the clock they are charged to differs.
    """

    def __init__(self, deadline: float) -> None:
        self._lock = threading.Lock()
        self._deadline = deadline
        self._active: float | None = None
        self.credited = 0.0
        self.calls = 0

    def deadline(self) -> float:
        with self._lock:
            if self._active is None:
                return self._deadline
            running = max(time.monotonic() - self._active, 0.0)
            return self._deadline + min(running, QUALIFIER_CREDIT_CEILING_SECONDS - self.credited)

    def remaining(self) -> float:
        return self.deadline() - time.monotonic()

    def expired(self) -> bool:
        return time.monotonic() >= self.deadline()

    def timed(self, call: Any) -> Any:
        started = time.monotonic()
        with self._lock:
            self._active = started
        try:
            return call()
        finally:
            elapsed = time.monotonic() - started
            with self._lock:
                self._active = None
                self.calls += 1
                grant = min(max(elapsed, 0.0), QUALIFIER_CREDIT_CEILING_SECONDS - self.credited)
                self.credited += grant
                self._deadline += grant


class RecordRecheck:
    """Qualification callback that offers a cheaper check between yielded lineage records or body chunks.

    Calling it runs the full check, like any callback, and every read bracket does. `record_check()` is asked for
    only between the records of a lineage stream or between the chunks of one object body (`record_step`): it runs
    `local` every time and the full check once `interval` seconds have passed since the start of the last full
    check that returned normally (the first call included); an interval of 0 runs the full check every time.
    A failing check never counts, so the next record runs it again. One thread: the scope pins the thread.
    """

    def __init__(self, full: Any, local: Any, interval: float) -> None:
        if not callable(full) or not callable(local) or not interval >= 0:
            raise ValueError("record recheck needs full and local checks and a non-negative interval")
        self._full, self._local, self._interval = full, local, interval
        self._began: float | None = None

    def __call__(self) -> Any:
        started = time.monotonic()
        result = self._full()
        self._began = started
        return result

    def record_check(self) -> Any:
        now = time.monotonic()
        if self._began is None or now - self._began >= self._interval:
            result = self._full()
            self._began = now
            return result
        return self._local()


def record_step(callback: Any) -> Any:
    """One check of a qualification callback between two records or chunks of a single byte stream.

    A `RecordRecheck` offers its cheaper variant; any other callback declares none, so it runs in full as before.
    """
    if type(callback) is RecordRecheck:
        return callback.record_check()
    return callback()


def _version(value: Any) -> str:
    if (
        type(value) is not str
        or value in ("", "null")
        or len(value) > 1024
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("non-null native object version required")
    return value


def _key(uri: str, bucket: str) -> str:
    prefix = "s3://" + bucket + "/"
    if type(uri) is not str or not uri.startswith(prefix):
        raise ValueError("foreign preparation object")
    key = uri[len(prefix) :]
    if not key or any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("unconfined preparation object")
    return key


def _encode(value: Any) -> bytes:
    import json

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def _freeze(value: Any) -> Any:
    """A private snapshot of a plain replay result, kept as one immutable bytes object.

    One contiguous object keeps no per-element objects alive: tens of thousands of small strings scattered through
    the allocator would pin its arenas (about 160 MB of resident memory after a pass at 56,700 rows). A result that
    does not survive a JSON round trip unchanged (a tuple, a non-string key, a float) is kept as a deep copy.
    """
    try:
        blob = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
        if json.loads(blob) == value:
            return blob
    except (TypeError, ValueError):
        pass
    return copy.deepcopy(value)


def _thaw(kept: Any) -> Any:
    return json.loads(kept) if type(kept) is bytes else copy.deepcopy(kept)


class _ReplayMemo:
    """Bounded, process-local, thread-safe memory of pure replay results, never persisted.

    A result is held only after its computation returned normally; an exception is raised to its caller and
    remembered by nobody. Callers get private copies, so no caller can change what the next one receives. One
    thread computes a key while others ask for the same key wait for it (never past `wait_seconds`, after which
    the waiter computes alone and shares nothing); if the computing thread fails, each waiter in turn computes
    for itself and sees its own outcome. The lock is never held while computing or copying. Other keys never wait.
    """

    def __init__(
        self, entries: int = REPLAY_MEMO_ENTRIES, wait_seconds: float = REPLAY_MEMO_WAIT_SECONDS
    ) -> None:
        if type(entries) is not int or entries < 1 or not wait_seconds > 0:
            raise ValueError("replay memory needs a positive integer size and a positive wait")
        self._entries, self._wait = entries, float(wait_seconds)
        self._lock = threading.Lock()
        self._held: OrderedDict[str, Any] = OrderedDict()
        self._flying: dict[str, threading.Event] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._held)

    def clear(self) -> None:
        with self._lock:
            self._held.clear()

    def fetch(self, key: str, compute: Any) -> Any:
        deadline = time.monotonic() + self._wait
        while True:
            with self._lock:
                stored = self._held.get(key)
                if stored is not None:
                    self._held.move_to_end(key)
                    flight, owner = None, False
                else:
                    flight = self._flying.get(key)
                    owner = flight is None
                    if owner:
                        flight = self._flying[key] = threading.Event()
            if stored is not None:
                return _thaw(stored)
            if owner:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not flight.wait(remaining):
                return compute()
        try:
            result = compute()
            kept = _freeze(result)
            with self._lock:
                self._held[key] = kept
                self._held.move_to_end(key)
                while len(self._held) > self._entries:
                    self._held.popitem(last=False)
            return result
        finally:
            with self._lock:
                self._flying.pop(key, None)
            flight.set()


_REPLAY_MEMO = _ReplayMemo()


def _replay_key(
    *, refs: dict[str, Any], records: dict[str, list[Any]], sources: Any, source_counts: Any
) -> str | None:
    """Digest of everything the CommitPack replay result is a function of, or None when an input is not plain data.

    The replay is pure: its result depends only on the parsed bodies, the expected sources and counts, and the
    module constants it reads. Each body enters by the exact-version identity and SHA-256 that this very pass
    measured over the bytes it read (the stream raises at its end when they differ), and the record counts tie the
    parsed lists to those bodies. None means "do not remember": the replay then runs and refuses as before.
    """
    from ...data_eng import commitpack_preparation as cp
    from ...data_eng import commitpack_profile as profile

    try:
        if type(sources) is not tuple or type(source_counts) is not dict:
            return None
        payload = {
            "memo": 1,
            "profile_id": cp.PROFILE_ID,
            "policy_version": cp.POLICY_VERSION,
            "constants": {
                "dataset_repo": profile.DATASET_REPO,
                "revision": profile.REVISION,
                "filename": profile.FILENAME,
                "artifact_sha256": profile.ARTIFACT_SHA256,
                "source_bytes": cp.SOURCE_BYTES,
                "max_rows": cp.MAX_ROWS,
                "supervision_kind": cp.SUPERVISION_KIND,
                "splits": list(cp.SPLITS),
                "coverage": cp.COVERAGE,
            },
            "bodies": {
                name: {
                    field: refs[name][field]
                    for field in ("uri", "version_id", "sha256", "byte_count", "format")
                }
                for name in OUTPUT_FILES
            },
            "record_counts": {name: len(records[name]) for name in OUTPUT_FILES if name != "REPORT.json"},
            "sources": list(sources),
            "source_counts": source_counts,
        }
        return hashlib.sha256(cp.encode_commitpack_metadata(payload)).hexdigest()
    except (KeyError, TypeError, ValueError):
        return None


def _validate_native_proof(value: Any) -> dict[str, Any]:
    # The owner supplies this exact pure validator; absence is a closed integration gap.
    from ...data_eng.inspection_schema import validate_native_privilege_proof

    return validate_native_privilege_proof(value)


def _integer(value: Any, *, positive: bool = False) -> None:
    if type(value) is not int or not (1 if positive else 0) <= value <= 2**63 - 1:
        raise ValueError("invalid preparation count")


def _native_request_field(name: str, value: Any) -> Any:
    """Admit only the observed false AppManaged default, without mutating handles."""
    if name not in ("ProcessingInputs", "ProcessingOutputConfig"):
        return value
    normalized = copy.deepcopy(value)
    if name == "ProcessingOutputConfig":
        if type(normalized) is not dict:
            raise ValueError("invalid native preparation output config")
        entries = normalized.get("Outputs")
    else:
        entries = normalized
    if type(entries) is not list:
        raise ValueError("invalid native preparation I/O entries")
    for entry in entries:
        if type(entry) is not dict:
            raise ValueError("invalid native preparation I/O entry")
        if "AppManaged" in entry:
            if entry["AppManaged"] is not False:
                raise ValueError("native preparation AppManaged differs")
            del entry["AppManaged"]
    return normalized


def _preparation_pair(sources: tuple[VerifiedPreparationSource, ...]) -> tuple[str, str]:
    from ...data_eng import commitpack_preparation, commitpack_profile

    if type(sources) is not tuple or not sources:
        raise ValueError("private preparation sources required")
    pairs = set()
    for source in sources:
        if type(source) is not VerifiedPreparationSource:
            raise ValueError("private preparation source required")
        ref = source.inspection_source.artifact_ref
        if source.repo_id == commitpack_profile.DATASET_REPO:
            if not commitpack_preparation.matches_commitpack_source(
                source.repo_id, source.revision, source.filename, ref
            ):
                raise ValueError("recognized CommitPack source cannot fall back")
            pairs.add((commitpack_preparation.PROFILE_ID, commitpack_preparation.POLICY_VERSION))
        elif ref.format == "jsonl":
            pairs.add((PROFILE_ID, POLICY_VERSION))
        elif ref.format == "parquet":
            pairs.add((DERIVED_PROFILE_ID, DERIVED_POLICY_VERSION))
        else:
            raise ValueError("unsupported preparation source format")
    if len(pairs) != 1:
        raise ValueError("mixed preparation profiles are not supported")
    return next(iter(pairs))


class ProcessingCodingPreparationBackend:
    backend_name = "processing"

    _BODY_SECONDS = BODY_SECONDS

    def __init__(self, *, sagemaker_client: Any = None, s3_client: Any = None) -> None:
        self._sm, self._s3 = sagemaker_client, s3_client
        self._qualified_io_pid = os.getpid()
        self._qualified_io = ContextVar("preparation_observation_" + uuid.uuid4().hex, default=None)

    @contextmanager
    def qualified_io(self, current_qualification=None):
        """Private observation lifetime; callback data never grants submission."""
        if os.getpid() != self._qualified_io_pid:
            raise ValueError("inherited processing observation scope refused")
        if current_qualification is not None and not callable(current_qualification):
            raise ValueError("private processing observation qualifier required")
        scope = {
            "callback": current_qualification,
            "active": True,
            "thread_id": threading.get_ident(),
            "parent": self._qualified_io.get(),
        }
        token = self._qualified_io.set(scope)
        try:
            yield self
        finally:
            scope["active"] = False
            self._qualified_io.reset(token)

    def _scope_chain(self, scope):
        """Cheap local checks: process, scope lifetime and thread. No callback runs."""
        if os.getpid() != self._qualified_io_pid:
            raise ValueError("inherited processing observation refused")
        scopes = []
        while scope is not None:
            if not scope["active"] or scope["thread_id"] != threading.get_ident():
                raise ValueError("expired or foreign-thread processing observation refused")
            scopes.append(scope)
            scope = scope["parent"]
        return scopes

    def _qualify_scope(self, scope):
        for current in reversed(self._scope_chain(scope)):
            callback = current["callback"]
            if callback is not None:
                callback()

    def _qualify_scope_record(self, scope):
        """Between lineage records: scope checks always, and a callback's own cheaper variant when it has one."""
        for current in reversed(self._scope_chain(scope)):
            callback = current["callback"]
            if callback is not None:
                record_step(callback)

    def _qualify_io(self):
        self._qualify_scope(self._qualified_io.get())

    def _observation_io(self, method, **kwargs):
        self._qualify_io()
        result = method(**kwargs)
        try:
            self._qualify_io()
        except BaseException:
            body = result.get("Body") if type(result) is dict else None
            close = getattr(body, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    pass
            raise
        return result

    @contextmanager
    def _owned_observation_body(self, body, *, seconds=None, deadline=None, join=False, label="body"):
        """Own returned streams before timer setup; retain any in-flight failure.

        With an allowance the context yields its `_BodyBudget`: the caller checks `expired()` and credits the time
        its qualifier spends inside the scope, and the close timer re-arms for whatever was credited.
        """
        closer, started = None, False
        pending = False
        budget, outcome, begun = None, "ok", time.monotonic()
        guard, stopped, current = threading.Lock(), threading.Event(), []
        try:
            if seconds is not None or deadline is not None:
                if deadline is None:
                    deadline = time.monotonic() + seconds
                budget = _BodyBudget(deadline)

                def expire():
                    # Timer thread: re-arm for credited admission time, otherwise close the stream (fail closed).
                    with guard:
                        if stopped.is_set():
                            return
                        left = budget.remaining()
                        if left > 0:
                            try:
                                again = threading.Timer(left, expire)
                                again.daemon = True
                                again.start()
                            except BaseException:
                                pass
                            else:
                                current[0] = again
                                return
                    body.close()

                delay = max(0, deadline - time.monotonic())
                closer = threading.Timer(delay, expire)
                closer.daemon = True
                closer.start()
                current.append(closer)
                started = True
            yield budget
        except BaseException as error:
            pending = True
            outcome = type(error).__name__
            raise
        finally:
            cleanup_error = None
            if started:
                with guard:
                    stopped.set()
                    closer = current[0]
                try:
                    closer.cancel()
                except BaseException as error:
                    cleanup_error = error
            try:
                body.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            if started and join:
                try:
                    closer.join(timeout=0.1)
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None and not pending:
                outcome = type(cleanup_error).__name__
            if budget is not None:
                try:
                    _LOG.warning(
                        "preparation_body_budget label=%s calls=%d credited_seconds=%.1f wall_seconds=%.1f outcome=%s",
                        label,
                        budget.calls,
                        budget.credited,
                        time.monotonic() - begun,
                        outcome,
                    )
                except Exception:
                    pass
            if cleanup_error is not None and not pending:
                raise cleanup_error

    def _qualify_chunk(self):
        """Between two chunks of one object body: scope checks always, a `RecordRecheck` offers its cheap variant."""
        self._qualify_scope_record(self._qualified_io.get())

    def _observation_read(self, body, count, budget=None, *, between_chunks=False):
        check = self._qualify_chunk if between_chunks else self._qualify_io
        if budget is None:
            check()
            result = body.read(count)
            check()
            return result
        budget.timed(check)
        result = body.read(count)
        budget.timed(check)
        return result

    def _clients(self, region: str) -> tuple[Any, Any]:
        if self._sm is not None and self._s3 is not None:
            return self._sm, self._s3
        import boto3
        from botocore.config import Config

        session = boto3.Session(region_name=region)
        config = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 0})
        return self._sm or session.client("sagemaker", config=config), self._s3 or session.client(
            "s3", config=config
        )

    def _raw(
        self,
        s3: Any,
        bucket: str,
        key: str,
        version: str,
        account: str,
        count: int,
        digest: str | None,
        *,
        metadata: bool = False,
        count_lines: bool = True,
        metadata_limit: int = METADATA_LIMIT,
    ) -> tuple[bytes | None, str, int | None]:
        limit = metadata_limit if metadata else MAX_BYTES
        if type(count) is not int or not 0 < count <= limit:
            raise ValueError("preparation raw I/O bound exceeded")
        response = self._observation_io(
            s3.get_object, Bucket=bucket, Key=key, VersionId=_version(version), ExpectedBucketOwner=account
        )
        body = response["Body"]
        with self._owned_observation_body(body, seconds=self._BODY_SECONDS, join=True, label="raw") as budget:
            if response.get("VersionId") != version or response.get("ContentLength") != count:
                raise ValueError("exact preparation object identity changed")
            measured = 0
            checksum = hashlib.sha256()
            lines = 0
            last = b""
            chunks = []
            while True:
                if budget.expired():
                    raise ValueError("preparation raw I/O deadline exceeded")
                chunk = self._observation_read(
                    body, min(1024 * 1024, count - measured + 1), budget, between_chunks=True
                )
                if budget.expired():
                    raise ValueError("preparation raw I/O deadline exceeded")
                if not chunk:
                    break
                measured += len(chunk)
                if measured > count:
                    raise ValueError("preparation object exceeds committed count")
                checksum.update(chunk)
                if count_lines:
                    lines += chunk.count(b"\n")
                    last = chunk[-1:]
                if metadata:
                    chunks.append(chunk)
            actual = checksum.hexdigest()
            if measured != count or (digest is not None and actual != digest):
                raise ImmutableArtifactRejectionError("artifact_hash_mismatch")
            result = (
                b"".join(chunks) if metadata else None,
                actual,
                lines + int(last != b"\n") if count_lines else None,
            )
        self._qualify_io()
        return result

    def _stage(
        self,
        s3: Any,
        catalog: PreparationCatalog,
        uri: str,
        version: str,
        count: int,
        digest: str,
        target: str,
    ) -> str:
        source_key = _key(uri, catalog.bucket)
        self._raw(
            s3, catalog.bucket, source_key, version, catalog.account_id, count, digest, count_lines=False
        )
        response = s3.copy_object(
            Bucket=catalog.bucket,
            Key=target,
            CopySource={"Bucket": catalog.bucket, "Key": source_key, "VersionId": version},
            ExpectedSourceBucketOwner=catalog.account_id,
            ExpectedBucketOwner=catalog.account_id,
        )
        staged = _version(response.get("VersionId"))
        self._raw(s3, catalog.bucket, target, staged, catalog.account_id, count, digest, count_lines=False)
        return staged

    def _put(
        self,
        s3: Any,
        catalog: PreparationCatalog,
        key: str,
        raw: bytes,
        *,
        metadata_limit: int = METADATA_LIMIT,
    ) -> str:
        response = s3.put_object(
            Bucket=catalog.bucket, Key=key, Body=raw, ExpectedBucketOwner=catalog.account_id
        )
        version = _version(response.get("VersionId"))
        self._raw(
            s3,
            catalog.bucket,
            key,
            version,
            catalog.account_id,
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            metadata=True,
            metadata_limit=metadata_limit,
        )
        return version

    def submit(
        self,
        *,
        sources: tuple[VerifiedPreparationSource, ...],
        catalog: PreparationCatalog,
        runtime_session_id: str,
    ) -> JobHandle:
        from ...data_eng import commitpack_preparation

        profile_id, policy_version = _preparation_pair(sources)
        commitpack = profile_id == commitpack_preparation.PROFILE_ID
        validate_catalog(
            catalog,
            exclusions_limit=commitpack_preparation.MAX_EXCLUSIONS_BYTES if commitpack else METADATA_LIMIT,
        )
        if (
            type(sources) is not tuple
            or not 0 < len(sources) <= 256
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", runtime_session_id)
        ):
            raise ValueError("private bounded preparation selection required")
        for source in sources:
            validate_source(source, catalog, runtime_session_id)
        derived = profile_id == DERIVED_PROFILE_ID
        sm, s3 = self._clients(catalog.region)
        location = (
            s3.get_bucket_location(Bucket=catalog.bucket, ExpectedBucketOwner=catalog.account_id).get(
                "LocationConstraint"
            )
            or "us-east-1"
        )
        if (
            location != catalog.region
            or s3.get_bucket_versioning(Bucket=catalog.bucket, ExpectedBucketOwner=catalog.account_id).get(
                "Status"
            )
            != "Enabled"
        ):
            raise ValueError("preparation requires selected same-region versioned bucket")
        job = catalog.deployment_prefix + "-data-prep-" + uuid.uuid4().hex[:12]
        prefix = "sessions/" + runtime_session_id + "/preparation/" + job + "/"
        input_prefix, output_prefix = prefix + "input/", prefix + "output/"
        measured_sources = []
        source_records = 0
        # Verify every original object before mutating any staging prefix.
        for source in sources:
            ref = source.inspection_source.artifact_ref
            _raw, _digest, records = self._raw(
                s3,
                catalog.bucket,
                _key(ref.uri, catalog.bucket),
                ref.version_id,
                catalog.account_id,
                ref.byte_count,
                ref.sha256,
                count_lines=not derived,
            )
            if not derived:
                source_records += records
            manifests = []
            for name, uri, version, digest in source.manifest_refs:
                head = s3.head_object(
                    Bucket=catalog.bucket,
                    Key=_key(uri, catalog.bucket),
                    VersionId=version,
                    ExpectedBucketOwner=catalog.account_id,
                )
                if head.get("VersionId") != version:
                    raise ValueError("acquisition manifest version differs")
                count = head.get("ContentLength")
                raw, _actual, _lines = self._raw(
                    s3,
                    catalog.bucket,
                    _key(uri, catalog.bucket),
                    version,
                    catalog.account_id,
                    count,
                    digest,
                    metadata=True,
                )
                decode_object(raw)
                manifests.append((name, uri, version, digest, count))
            measured_sources.append((source, manifests, records))
        excluded = catalog.exclusions_ref
        excluded_raw, _digest, _lines = self._raw(
            s3,
            catalog.bucket,
            _key(excluded.uri, catalog.bucket),
            excluded.version_id,
            catalog.account_id,
            excluded.byte_count,
            excluded.sha256,
            metadata=True,
            metadata_limit=commitpack_preparation.MAX_EXCLUSIONS_BYTES if commitpack else METADATA_LIMIT,
        )
        exclusions = decode_object(
            excluded_raw, limit=commitpack_preparation.MAX_EXCLUSIONS_BYTES if commitpack else METADATA_LIMIT
        )
        if commitpack:
            commitpack_preparation.validate_commitpack_exclusions(exclusions)
            if excluded_raw != commitpack_preparation.encode_commitpack_metadata(exclusions):
                raise ValueError("CommitPack exclusions are not exact canonical staged bytes")
        elif set(exclusions) != {"instance_ids", "patch_sha256"}:
            raise ValueError("fixed pinned exclusion schema required")
        header = {
            "job_name": job,
            "runtime_session_id": runtime_session_id,
            "deployment_id": catalog.deployment_id,
            "deployment_prefix": catalog.deployment_prefix,
            "image_uri": catalog.image_uri,
            "source_commit": catalog.source_commit,
            "profile_id": profile_id,
            "policy_version": policy_version,
        }
        entries = []
        for index, (source, manifests, _records) in enumerate(measured_sources):
            ref = source.inspection_source.artifact_ref
            relative = f"data/{index:06d}.{ref.format}"
            staged = self._stage(
                s3, catalog, ref.uri, ref.version_id, ref.byte_count, ref.sha256, input_prefix + relative
            )
            data = {
                "relative_name": relative,
                "byte_count": ref.byte_count,
                "sha256": ref.sha256,
                "format": ref.format,
                "original_uri": ref.uri,
                "original_version_id": ref.version_id,
                "staged_version_id": staged,
            }
            manifest_entries = []
            for name, uri, version, digest, count in manifests:
                relative_manifest = f"manifests/{index:06d}-{name}.json"
                staged_manifest = self._stage(
                    s3, catalog, uri, version, count, digest, input_prefix + relative_manifest
                )
                manifest_entries.append(
                    {
                        "name": name,
                        "relative_name": relative_manifest,
                        "byte_count": count,
                        "sha256": digest,
                        "original_uri": uri,
                        "original_version_id": version,
                        "staged_version_id": staged_manifest,
                    }
                )
            entries.append(
                {
                    "data": data,
                    "manifest_files": manifest_entries,
                    "repo_id": source.repo_id,
                    "revision": source.revision,
                    "filename": source.filename,
                    "acquisition_image_uri": source.acquisition_image_uri,
                    "acquisition_source_commit": source.acquisition_source_commit,
                }
            )
        excluded_version = self._put(
            s3,
            catalog,
            input_prefix + "exclusions.json",
            excluded_raw,
            metadata_limit=commitpack_preparation.MAX_EXCLUSIONS_BYTES if commitpack else METADATA_LIMIT,
        )
        selection = {
            "schema_version": 1,
            **header,
            "sources": entries,
            "exclusions": {
                "byte_count": excluded.byte_count,
                "sha256": excluded.sha256,
                "original_uri": excluded.uri,
                "original_version_id": excluded.version_id,
                "staged_version_id": excluded_version,
            },
        }
        selection_raw = _encode(selection)
        if len(selection_raw) > METADATA_LIMIT:
            raise ValueError("preparation selection manifest bound exceeded")
        selection_sha = hashlib.sha256(selection_raw).hexdigest()
        self._put(s3, catalog, input_prefix + "selection.json", selection_raw)
        cli = {
            "selection_sha256": selection_sha,
            "selection_bytes": len(selection_raw),
            "exclusions_sha256": excluded.sha256,
            "exclusions_bytes": excluded.byte_count,
            **header,
        }
        arguments = [
            part for key, value in cli.items() for part in ("--" + key.replace("_", "-"), str(value))
        ]
        request = {
            "ProcessingJobName": job,
            "RoleArn": catalog.execution_role_arn,
            "AppSpecification": {
                "ImageUri": catalog.image_uri,
                "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "preparation"],
                "ContainerArguments": arguments,
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": catalog.instance_type,
                    "VolumeSizeInGB": catalog.volume_size_gb,
                }
            },
            "ProcessingInputs": [
                {
                    "InputName": "preparation-input",
                    "S3Input": {
                        "S3Uri": "s3://" + catalog.bucket + "/" + input_prefix,
                        "LocalPath": "/opt/ml/processing/input/preparation",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                        "S3CompressionType": "None",
                    },
                }
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "preparation-output",
                        "S3Output": {
                            "S3Uri": "s3://" + catalog.bucket + "/" + output_prefix,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": catalog.max_runtime_seconds},
            "NetworkConfig": {"EnableNetworkIsolation": True, "EnableInterContainerTrafficEncryption": False},
            "Tags": [
                {"Key": "amlas:deployment-id", "Value": catalog.deployment_id},
                {"Key": "amlas:deployment-prefix", "Value": catalog.deployment_prefix},
                {"Key": "amlas:runtime-session-id", "Value": runtime_session_id},
            ],
        }
        accepted = sm.create_processing_job(**request)
        arn = f"arn:aws:sagemaker:{catalog.region}:{catalog.account_id}:processing-job/{job}"
        if accepted.get("ProcessingJobArn") != arn:
            raise RuntimeError("preparation accepted another native identity")
        return JobHandle(
            id=job,
            backend=self.backend_name,
            arn=arn,
            extra={
                "runtime_session_id": runtime_session_id,
                "catalog": catalog,
                "request": copy.deepcopy(request),
                "header": header,
                "output_prefix": output_prefix,
                "selection_sha256": selection_sha,
                "exclusions_sha256": excluded.sha256,
                "source_records": None if derived else source_records,
                "sources": sources,
                **(
                    {
                        "source_row_counts": {
                            commitpack_preparation.commitpack_source_key(
                                source.inspection_source.artifact_ref.uri,
                                source.inspection_source.artifact_ref.version_id,
                                source.inspection_source.artifact_ref.sha256,
                            ): records
                            for source, _manifests, records in measured_sources
                        }
                    }
                    if commitpack
                    else {}
                ),
            },
        )

    def _description(self, handle: JobHandle) -> dict[str, Any]:
        if (
            type(handle) is not JobHandle
            or handle.backend != self.backend_name
            or type(handle.extra.get("catalog")) is not PreparationCatalog
        ):
            raise ValueError("private preparation handle required")
        catalog = handle.extra["catalog"]
        sm, _s3 = self._clients(catalog.region)
        value = self._observation_io(sm.describe_processing_job, ProcessingJobName=handle.id)
        request = handle.extra["request"]
        if value.get("ProcessingJobName") != handle.id or value.get("ProcessingJobArn") != handle.arn:
            raise ValueError("native preparation identity drifted")
        for key in (
            "RoleArn",
            "AppSpecification",
            "ProcessingResources",
            "ProcessingInputs",
            "ProcessingOutputConfig",
            "StoppingCondition",
            "NetworkConfig",
        ):
            if _native_request_field(key, value.get(key)) != _native_request_field(key, request[key]):
                raise ValueError("native preparation request drifted")
        if value.get("Environment", {}) != {}:
            raise ValueError("native preparation environment drifted")
        self._qualify_io()
        return value

    def status(self, handle: JobHandle) -> JobStatus:
        try:
            value = self._description(handle)
        except Exception as error:
            raw = {"error_type": type(error).__name__}
            # Only this module's own fixed ValueError wording is named; anything else stays a class name.
            reason = str(error) if type(error) is ValueError else ""
            if reason in _REASONS:
                raw["reason"] = reason
            try:
                _LOG.warning(
                    "preparation_observation_failed error_type=%s%s",
                    type(error).__name__,
                    " reason=" + reason if "reason" in raw else "",
                )
            except Exception:
                pass
            return JobStatus(JobState.UNKNOWN, "preparation observation failed", raw)
        state = {
            "InProgress": JobState.RUNNING,
            "Stopping": JobState.RUNNING,
            "Completed": JobState.SUCCEEDED,
            "Failed": JobState.FAILED,
            "Stopped": JobState.STOPPED,
        }.get(value.get("ProcessingJobStatus"), JobState.UNKNOWN)
        self._qualify_io()
        return JobStatus(state, raw=value)

    def _output(
        self,
        s3: Any,
        catalog: PreparationCatalog,
        prefix: str,
        name: str,
        *,
        metadata: bool,
        count: int | None = None,
        digest: str | None = None,
        metadata_limit: int = METADATA_LIMIT,
    ) -> tuple[bytes | None, dict[str, Any], int]:
        key = prefix + name
        head = self._observation_io(
            s3.head_object, Bucket=catalog.bucket, Key=key, ExpectedBucketOwner=catalog.account_id
        )
        version = _version(head.get("VersionId"))
        size = head.get("ContentLength")
        if count is not None and size != count:
            raise ValueError("preparation output byte count differs")
        raw, actual, lines = self._raw(
            s3,
            catalog.bucket,
            key,
            version,
            catalog.account_id,
            size,
            digest,
            metadata=metadata,
            metadata_limit=metadata_limit,
        )
        self._qualify_io()
        return (
            raw,
            {
                "uri": "s3://" + catalog.bucket + "/" + key,
                "version_id": version,
                "sha256": actual,
                "byte_count": size,
                "format": "json" if name.endswith(".json") else "jsonl",
            },
            lines,
        )

    def _lineage_objects(self, s3: Any, catalog: PreparationCatalog, ref: dict[str, Any]) -> Any:
        """Capture the original lexical scope before returning lazy metadata I/O."""
        scope = self._qualified_io.get()

        def qualify():
            self._qualify_io()
            if scope is not self._qualified_io.get():
                self._qualify_scope(scope)

        def record_qualify():
            self._qualify_scope_record(self._qualified_io.get())
            if scope is not self._qualified_io.get():
                self._qualify_scope_record(scope)

        return self._stream_lineage_objects(s3, catalog, ref, qualify, record_qualify)

    def _stream_lineage_objects(self, s3, catalog, ref, qualify, record_qualify=None):
        from ...data_eng.inspection_schema import decode_object as decode_lineage_object

        count, version = ref["byte_count"], _version(ref["version_id"])
        qualify()
        response = self._observation_io(
            s3.get_object,
            Bucket=catalog.bucket,
            Key=_key(ref["uri"], catalog.bucket),
            VersionId=version,
            ExpectedBucketOwner=catalog.account_id,
        )
        body = response["Body"]
        with self._owned_observation_body(
            body, seconds=self._BODY_SECONDS, join=True, label="lineage"
        ) as budget:
            measured, checksum, pending = 0, hashlib.sha256(), bytearray()
            budget.timed(qualify)
            if response.get("VersionId") != version or response.get("ContentLength") != count:
                raise ValueError("lineage exact-version identity differs")
            while True:
                if budget.expired():
                    raise ValueError("lineage metadata deadline exceeded")
                budget.timed(qualify)
                chunk = self._observation_read(body, min(LINEAGE_READ_BYTES, count - measured + 1), budget)
                budget.timed(qualify)
                if budget.expired():
                    raise ValueError("lineage metadata deadline exceeded")
                if not chunk:
                    break
                measured += len(chunk)
                if measured > count:
                    raise ValueError("lineage metadata byte count differs")
                checksum.update(chunk)
                pending.extend(chunk)
                while True:
                    newline = pending.find(b"\n")
                    if newline < 0:
                        break
                    if newline > LINEAGE_RECORD_LIMIT:
                        raise ValueError("lineage metadata row bound exceeded")
                    line = bytes(pending[:newline])
                    del pending[: newline + 1]
                    if not line:
                        raise ValueError("empty lineage metadata row")
                    line.decode("utf-8")
                    record = decode_lineage_object(line, limit=LINEAGE_RECORD_LIMIT)
                    budget.timed(record_qualify if record_qualify is not None else qualify)
                    yield record
                if len(pending) > LINEAGE_RECORD_LIMIT:
                    raise ValueError("lineage metadata row bound exceeded")
            if pending or measured != count or checksum.hexdigest() != ref["sha256"]:
                raise ValueError("lineage metadata termination/hash differs")
        qualify()

    def _verify_derived_lineage(
        self,
        s3: Any,
        catalog: PreparationCatalog,
        handle: JobHandle,
        ref: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        from ...data_eng.open_swe_canonical import CARD_SHA256, DATA_LICENSE

        sources = handle.extra.get("sources")
        if type(sources) is not tuple or not sources:
            raise ValueError("derived lineage lacks original private sources")
        known = {}
        for index, source in enumerate(sources):
            validate_source(source, catalog, handle.extra["runtime_session_id"])
            artifact = source.inspection_source.artifact_ref
            key = (
                artifact.uri,
                artifact.version_id,
                artifact.sha256,
                source.repo_id,
                source.revision,
                source.filename,
            )
            if key in known or artifact.format != "parquet":
                raise ValueError("derived lineage source selection differs")
            known[key] = index
        rows = [set() for _source in sources]
        accepted, rejected, retained, edge_count = 0, 0, 0, 0
        core_reasons, refused_reasons = Counter(), Counter()
        provenance_keys = {
            "profile_id",
            "dataset_repo",
            "revision",
            "config_name",
            "filename",
            "artifact_uri",
            "artifact_version_id",
            "artifact_sha256",
            "row_index",
            "data_license",
            "card_sha256",
            "coverage",
        }
        edge_keys = {
            "source_row_ordinal",
            "result_message_index",
            "assistant_message_index",
            "call_index",
            "explicit_originating_call_id",
            "original_result_id_state",
            "normalization_policy_version",
            "linkage_status",
        }
        common = {
            "source_locator",
            "source_version_id",
            "source_sha256",
            "record_id",
            "decision",
            "cluster_id",
            "representative_id",
            "normalization",
        }
        with closing(self._lineage_objects(s3, catalog, ref)) as stream:
            for entry in stream:
                if accepted + rejected >= MAX_SOURCE_ROWS:
                    raise ValueError("derived lineage selection bound exceeded")
                norm = entry.get("normalization")
                if type(norm) is not dict or norm.get("profile_id") != DERIVED_PROFILE_ID:
                    raise ValueError("derived lineage normalization differs")
                status = norm.get("status")
                expected = {
                    "profile_id",
                    "status",
                    "normalized_record_sha256",
                    "normalization_lineage",
                    "source_provenance",
                }
                if status == "rejected":
                    expected.add("reason")
                if status not in ("accepted", "rejected") or set(norm) != expected:
                    raise ValueError("derived lineage status/schema differs")
                prov = norm["source_provenance"]
                if type(prov) is not dict or set(prov) != provenance_keys:
                    raise ValueError("derived lineage original provenance differs")
                key = tuple(
                    prov[name]
                    for name in (
                        "artifact_uri",
                        "artifact_version_id",
                        "artifact_sha256",
                        "dataset_repo",
                        "revision",
                        "filename",
                    )
                )
                if any(type(value) is not str for value in key) or key not in known:
                    raise ValueError("derived lineage original source differs")
                ordinal = prov["row_index"]
                _integer(ordinal)
                source_index = known[key]
                if ordinal >= MAX_SOURCE_ROWS or ordinal in rows[source_index]:
                    raise ValueError("derived lineage original row repeats/differs")
                rows[source_index].add(ordinal)
                if (
                    prov["profile_id"] != DERIVED_PROFILE_ID
                    or prov["config_name"] != "v1.2"
                    or prov["data_license"] != DATA_LICENSE
                    or prov["card_sha256"] != CARD_SHA256
                    or prov["coverage"]
                    != "qualified singleton derived linkage; no original-link verification, pinned corpus acceptance or near-dedup claim"
                    or entry.get("source_locator") != key[0] + "#version=" + key[1] + "#row=" + str(ordinal)
                    or entry.get("source_version_id") != key[1]
                    or entry.get("source_sha256") != key[2]
                ):
                    raise ValueError("derived lineage source/ordinal attribution differs")
                edges = norm["normalization_lineage"]
                if type(edges) is not list or len(edges) > 8192:
                    raise ValueError("derived edge envelope differs")
                digest = norm["normalized_record_sha256"]
                if status == "rejected":
                    reason = norm["reason"]
                    if (
                        type(reason) is not str
                        or reason not in report["normalization_rejection_reasons"]
                        or digest is not None
                        or edges
                        or set(entry) != common
                        or entry["decision"] != "normalization_" + reason
                        or any(
                            entry[name] is not None
                            for name in ("record_id", "cluster_id", "representative_id")
                        )
                    ):
                        raise ValueError("normalization refusal lineage differs")
                    refused_reasons[reason] += 1
                    rejected += 1
                    continue
                if (
                    type(digest) is not str
                    or not re.fullmatch(r"[a-f0-9]{64}", digest)
                    or set(entry) != common | {"repository", "instance_id", "trajectory_id", "license"}
                    or type(entry["record_id"]) is not str
                    or not re.fullmatch(r"[a-f0-9]{64}", entry["record_id"])
                ):
                    raise ValueError("normalized canonical digest/identity differs")
                seen_calls, seen_results = set(), set()
                for edge in edges:
                    if type(edge) is not dict or set(edge) != edge_keys:
                        raise ValueError("derived edge schema differs")
                    for name in (
                        "source_row_ordinal",
                        "result_message_index",
                        "assistant_message_index",
                        "call_index",
                    ):
                        _integer(edge[name])
                    call, result_index, assistant_index = (
                        edge["explicit_originating_call_id"],
                        edge["result_message_index"],
                        edge["assistant_message_index"],
                    )
                    if (
                        edge["source_row_ordinal"] != ordinal
                        or edge["call_index"] != 0
                        or not 0 <= assistant_index < result_index < 16384
                        or type(call) is not str
                        or not 0 < len(call) <= 512
                        or any(ord(c) < 32 for c in call)
                        or call in seen_calls
                        or result_index in seen_results
                        or edge["original_result_id_state"] != "absent"
                        or edge["normalization_policy_version"] != DERIVED_PROFILE_ID
                        or edge["linkage_status"] != DERIVED_LINKAGE_STATUS
                    ):
                        raise ValueError("derived qualified linkage differs")
                    seen_calls.add(call)
                    seen_results.add(result_index)
                decision = entry["decision"]
                if decision == "retained":
                    retained += 1
                elif type(decision) is str and decision in report["primary_drop_counts"]:
                    core_reasons[decision] += 1
                else:
                    raise ValueError("derived core lineage decision differs")
                accepted += 1
                edge_count += len(edges)
        if any(values and max(values) + 1 != len(values) for values in rows):
            raise ValueError("derived original row ordinals are incomplete")
        if (
            accepted != report["normalization_accepted_count"]
            or rejected != report["normalization_rejected_count"]
            or retained != report["counts"]["retained_representatives"]
            or accepted + rejected != report["normalization_source_count"]
            or dict(core_reasons) != report["primary_drop_counts"]
            or dict(refused_reasons) != report["normalization_rejection_reasons"]
            or edge_count != report["derived_edge_count"]
        ):
            raise ValueError("derived original-row lineage conservation differs")

    def _commitpack_artifacts(self, handle: JobHandle, base: dict[str, Any]) -> dict[str, Any]:
        from ...data_eng import commitpack_preparation as cp

        catalog = handle.extra["catalog"]
        _sm, s3 = self._clients(catalog.region)
        prefix = handle.extra["output_prefix"]
        listed = self._observation_io(
            s3.list_objects_v2,
            Bucket=catalog.bucket,
            Prefix=prefix,
            MaxKeys=8,
            ExpectedBucketOwner=catalog.account_id,
        )
        names = {item["Key"] for item in listed.get("Contents", [])}
        if listed.get("IsTruncated"):
            raise ValueError("CommitPack output inventory truncated")
        if len(listed.get("Contents", [])) == 1 and names == {prefix + "REPORT.json"}:
            raw, ref, _lines = self._output(
                s3, catalog, prefix, "REPORT.json", metadata=True, metadata_limit=262144
            )
            outcome = cp.validate_nonusable_outcome(decode_object(raw, limit=262144))
            self._qualify_io()
            return {
                **base,
                "status": "incomplete",
                "native_job": handle.id,
                "native_arn": handle.arn,
                "nonusable_outcome": outcome,
                "nonusable_report_ref": ref,
                "training_ready": False,
            }
        if len(listed.get("Contents", [])) != 7 or names != {
            prefix + name for name in (*OUTPUT_FILES, "COMMITTED.json")
        }:
            raise ValueError("CommitPack committed inventory differs")
        raw, committed_ref, _lines = self._output(s3, catalog, prefix, "COMMITTED.json", metadata=True)
        committed = decode_object(raw)
        keys = {
            "schema_version",
            *HEADER,
            "privilege_profile",
            "selection_sha256",
            "exclusions_sha256",
            "files",
            "counts",
            "coverage",
            "guest_evidence",
        }
        if (
            set(committed) != keys
            or type(committed["schema_version"]) is not int
            or committed["schema_version"] != 2
            or committed["privilege_profile"] != PRIVILEGE_PROFILE
            or any(committed[key] != handle.extra["header"][key] for key in HEADER)
            or any(committed[key] != handle.extra[key] for key in ("selection_sha256", "exclusions_sha256"))
        ):
            raise ValueError("CommitPack committed header differs")
        evidence = committed["guest_evidence"]
        if (
            type(evidence) is not dict
            or set(evidence) != {"privilege_proof", "credential_observation", "network_observation"}
            or evidence["credential_observation"] != "env_and_home_file_presence"
            or evidence["network_observation"] != "fixed_public_and_metadata_tcp_denial"
        ):
            raise ValueError("CommitPack guest observations differ")
        _validate_native_proof(evidence["privilege_proof"])
        files = committed["files"]
        if type(files) is not dict or set(files) != set(OUTPUT_FILES):
            raise ValueError("CommitPack file manifest differs")
        refs, records, report = {}, {}, None
        for name in OUTPUT_FILES:
            fields = files[name]
            if type(fields) is not dict or set(fields) != {"byte_count", "sha256", "format", "record_count"}:
                raise ValueError("CommitPack file descriptor differs")
            _integer(fields["byte_count"], positive=True)
            _integer(fields["record_count"], positive=True)
            if (
                type(fields["sha256"]) is not str
                or not re.fullmatch(r"[a-f0-9]{64}", fields["sha256"])
                or fields["format"] != ("json" if name == "REPORT.json" else "jsonl")
            ):
                raise ValueError("CommitPack output type/hash differs")
            raw, ref, lines = self._output(
                s3,
                catalog,
                prefix,
                name,
                metadata=name == "REPORT.json",
                count=fields["byte_count"],
                digest=fields["sha256"],
                metadata_limit=cp.COMMITPACK_REPORT_LIMIT if name == "REPORT.json" else METADATA_LIMIT,
            )
            if name.endswith(".jsonl") and lines != fields["record_count"]:
                raise ValueError("CommitPack actual output line count differs")
            refs[name] = ref
            if name == "REPORT.json":
                report = decode_object(raw, limit=cp.COMMITPACK_REPORT_LIMIT)
                if fields["record_count"] != 1:
                    raise ValueError("CommitPack report record count differs")
            else:
                values = []
                with closing(self._lineage_objects(s3, catalog, ref)) as stream:
                    for record in stream:
                        if len(values) >= cp.MAX_ROWS:
                            raise ValueError("CommitPack output row bound exceeded")
                        values.append(record)
                records[name] = values
        if report is None:
            raise ValueError("CommitPack report missing")
        sources = tuple(
            {
                "repo_id": source.repo_id,
                "revision": source.revision,
                "filename": source.filename,
                "artifact_ref": {
                    "uri": source.inspection_source.artifact_ref.uri,
                    "version_id": source.inspection_source.artifact_ref.version_id,
                    "sha256": source.inspection_source.artifact_ref.sha256,
                    "byte_count": source.inspection_source.artifact_ref.byte_count,
                    "format": source.inspection_source.artifact_ref.format,
                },
            }
            for source in handle.extra["sources"]
        )
        source_counts = handle.extra.get("source_row_counts")
        if type(source_counts) is not dict or not source_counts:
            raise ValueError("private CommitPack source linecounts required")
        source_counts = dict(source_counts)  # one snapshot for the memory key and for the replay

        def replay() -> dict[str, Any]:
            return cp.replay_commitpack_preparation(
                lineage_records=records["lineage.jsonl"],
                membership_records=records["membership.jsonl"],
                split_records={name: records[name + ".jsonl"] for name in cp.SPLITS},
                report=report,
                expected_sources=sources,
                expected_source_counts=source_counts,
            )

        # Every body above was just read at its exact version and hashed; only an identical re-derivation of the
        # same verified bytes under the same parameters is taken from the process-local memory.
        key = _replay_key(refs=refs, records=records, sources=sources, source_counts=source_counts)
        verified = replay() if key is None else _REPLAY_MEMO.fetch(key, replay)
        if (
            cp.encode_commitpack_metadata(committed["counts"])
            != cp.encode_commitpack_metadata(verified["counts"])
            or cp.encode_commitpack_metadata(committed["coverage"])
            != cp.encode_commitpack_metadata(verified["coverage"])
            or verified["exclusion_policy_sha256"] != handle.extra["exclusions_sha256"]
            or type(handle.extra["source_records"]) is not int
            or verified["counts"]["source"] != handle.extra["source_records"]
        ):
            raise ValueError("CommitPack independently replayed report differs")
        self._qualify_io()
        return {
            **base,
            "success": True,
            "prepared": True,
            "training_ready": False,
            "status": "prepared",
            "native_job": handle.id,
            "preparation_ref": committed_ref,
            "split_refs": {name: refs[name + ".jsonl"] for name in cp.SPLITS},
            "sidecar_refs": {name: refs[name + ".jsonl"] for name in ("membership", "lineage")}
            | {"report": refs["REPORT.json"]},
            "counts": verified["counts"],
            "coverage": verified["coverage"],
            "supervision_kind": verified["supervision_kind"],
        }

    def artifacts(self, handle: JobHandle) -> dict[str, Any]:
        state = self.status(handle)
        base = {
            "success": False,
            "accepted": True,
            "prepared": False,
            "trained": False,
            "status": state.state.value,
        }
        if state.state != JobState.SUCCEEDED:
            return base
        # Native status/readback already bound this original handle; include the
        # same identity in every successful profile's private artifact protocol.
        base.update(job_name=handle.id, job_arn=handle.arn)
        try:
            catalog = handle.extra["catalog"]
            _sm, s3 = self._clients(catalog.region)
            prefix = handle.extra["output_prefix"]
            pair = (handle.extra["header"].get("profile_id"), handle.extra["header"].get("policy_version"))
            from ...data_eng import commitpack_preparation

            if pair == (commitpack_preparation.PROFILE_ID, commitpack_preparation.POLICY_VERSION):
                return self._commitpack_artifacts(handle, base)
            if pair not in ((PROFILE_ID, POLICY_VERSION), (DERIVED_PROFILE_ID, DERIVED_POLICY_VERSION)):
                raise ValueError("closed preparation profile/policy pair required")
            derived = pair[0] == DERIVED_PROFILE_ID
            listed = self._observation_io(
                s3.list_objects_v2,
                Bucket=catalog.bucket,
                Prefix=prefix,
                MaxKeys=8,
                ExpectedBucketOwner=catalog.account_id,
            )
            expected = {prefix + name for name in (*OUTPUT_FILES, "COMMITTED.json")}
            if (
                listed.get("IsTruncated")
                or len(listed.get("Contents", [])) != 7
                or {item["Key"] for item in listed["Contents"]} != expected
            ):
                raise ValueError("preparation committed output inventory differs")
            raw, committed_ref, _lines = self._output(s3, catalog, prefix, "COMMITTED.json", metadata=True)
            committed = decode_object(raw)
            keys = {
                "schema_version",
                *HEADER,
                "privilege_profile",
                "selection_sha256",
                "exclusions_sha256",
                "files",
                "counts",
                "coverage",
                "guest_evidence",
            }
            if (
                set(committed) != keys
                or type(committed["schema_version"]) is not int
                or committed["schema_version"] != 2
                or committed["privilege_profile"] != PRIVILEGE_PROFILE
            ):
                raise ValueError("preparation committed schema/profile differs")
            if any(committed[key] != handle.extra["header"][key] for key in HEADER) or any(
                committed[key] != handle.extra[key] for key in ("selection_sha256", "exclusions_sha256")
            ):
                raise ValueError("preparation committed provenance differs")
            evidence = committed["guest_evidence"]
            if (
                type(evidence) is not dict
                or set(evidence) != {"privilege_proof", "credential_observation", "network_observation"}
                or evidence["credential_observation"] != "env_and_home_file_presence"
                or evidence["network_observation"] != "fixed_public_and_metadata_tcp_denial"
            ):
                raise ValueError("preparation descriptive guest evidence differs")
            _validate_native_proof(evidence["privilege_proof"])
            counts = committed["counts"]
            if type(counts) is not dict or set(counts) != {
                "source_count",
                "retained_representatives",
                "primary_drop_count",
                "unresolved_record_count",
                "split_counts",
            }:
                raise ValueError("preparation count schema differs")
            for key in (
                "source_count",
                "retained_representatives",
                "primary_drop_count",
                "unresolved_record_count",
            ):
                _integer(counts[key])
            if (
                (not derived and counts["source_count"] != handle.extra["source_records"])
                or (
                    derived
                    and (
                        handle.extra["source_records"] is not None
                        or not 0 < counts["source_count"] <= MAX_SOURCE_ROWS
                    )
                )
                or counts["unresolved_record_count"] != 0
                or counts["source_count"] != counts["retained_representatives"] + counts["primary_drop_count"]
            ):
                raise ValueError("preparation count conservation differs")
            splits = counts["split_counts"]
            if type(splits) is not dict or set(splits) != {"train", "validation", "test"}:
                raise ValueError("preparation split schema differs")
            for value in splits.values():
                _integer(value, positive=True)
            if sum(splits.values()) != counts["retained_representatives"]:
                raise ValueError("preparation split membership differs")
            coverage = {
                "near_dedup": "not_implemented",
                "exclusion_scope": "supplied_instance_ids_and_patch_digests_only",
            }
            if committed["coverage"] != coverage:
                raise ValueError("preparation coverage claim differs")
            files = committed["files"]
            if type(files) is not dict or set(files) != set(OUTPUT_FILES):
                raise ValueError("preparation file manifest differs")
            refs = {}
            report = None
            for name in OUTPUT_FILES:
                fields = files[name]
                if type(fields) is not dict or set(fields) != {
                    "byte_count",
                    "sha256",
                    "format",
                    "record_count",
                }:
                    raise ValueError("preparation file binding differs")
                _integer(fields["byte_count"], positive=True)
                _integer(fields["record_count"], positive=True)
                if (
                    type(fields["sha256"]) is not str
                    or not re.fullmatch(r"[a-f0-9]{64}", fields["sha256"])
                    or fields["format"] != ("json" if name.endswith(".json") else "jsonl")
                ):
                    raise ValueError("preparation file type/hash differs")
                raw, ref, lines = self._output(
                    s3,
                    catalog,
                    prefix,
                    name,
                    metadata=name == "REPORT.json",
                    count=fields["byte_count"],
                    digest=fields["sha256"],
                )
                if name.endswith(".jsonl") and lines != fields["record_count"]:
                    raise ValueError("preparation actual record count differs")
                refs[name] = ref
                if name == "REPORT.json":
                    report = decode_object(raw)
            if (
                report is None
                or set(report)
                != {
                    "schema_version",
                    "status",
                    "counts",
                    "primary_drop_counts",
                    "secret_pattern_counts",
                    "coverage",
                }
                | (
                    {
                        "normalization_source_count",
                        "normalization_accepted_count",
                        "normalization_rejected_count",
                        "normalization_rejection_reasons",
                        "derived_edge_count",
                        "linkage_status",
                    }
                    if derived
                    else set()
                )
                or type(report["schema_version"]) is not int
                or report["schema_version"] != (2 if derived else 1)
                or report["status"] != "complete"
                or report["counts"] != counts
                or report["coverage"] != coverage
            ):
                raise ValueError("preparation report differs from commit")
            for mapping in (report["primary_drop_counts"], report["secret_pattern_counts"]):
                if type(mapping) is not dict:
                    raise ValueError("preparation drop/detector schema differs")
                for key, value in mapping.items():
                    if type(key) is not str or not re.fullmatch(r"[a-z_]{1,64}", key):
                        raise ValueError("preparation diagnostic label differs")
                    _integer(value)
            if derived:
                for name in (
                    "normalization_source_count",
                    "normalization_accepted_count",
                    "normalization_rejected_count",
                    "derived_edge_count",
                ):
                    _integer(report[name])
                reasons = report["normalization_rejection_reasons"]
                if type(reasons) is not dict:
                    raise ValueError("normalization refusal counts differ")
                for reason, count in reasons.items():
                    if type(reason) is not str or not re.fullmatch(r"[a-z0-9_]{1,128}", reason):
                        raise ValueError("normalization refusal label differs")
                    _integer(count, positive=True)
                core_drops = sum(report["primary_drop_counts"].values())
                if (
                    report["normalization_source_count"] != counts["source_count"]
                    or report["normalization_source_count"]
                    != report["normalization_accepted_count"] + report["normalization_rejected_count"]
                    or report["normalization_accepted_count"]
                    != counts["retained_representatives"] + core_drops
                    or sum(reasons.values()) != report["normalization_rejected_count"]
                    or core_drops + report["normalization_rejected_count"] != counts["primary_drop_count"]
                    or report["linkage_status"] != DERIVED_LINKAGE_STATUS
                ):
                    raise ValueError("normalization/core drop conservation differs")
                self._verify_derived_lineage(s3, catalog, handle, refs["lineage.jsonl"], report)
            elif sum(report["primary_drop_counts"].values()) != counts["primary_drop_count"]:
                raise ValueError("preparation exclusive drop counts differ")
            for split in splits:
                if files[split + ".jsonl"]["record_count"] != splits[split]:
                    raise ValueError("preparation actual split count differs")
            if (
                files["membership.jsonl"]["record_count"] != counts["retained_representatives"]
                or files["lineage.jsonl"]["record_count"] != counts["source_count"]
            ):
                raise ValueError("preparation actual lineage/membership count differs")
            self._qualify_io()
            return {
                **base,
                "success": True,
                "prepared": True,
                "status": "completed",
                "preparation_ref": committed_ref,
                "split_refs": {name: refs[name + ".jsonl"] for name in splits},
                "sidecar_refs": {name: refs[name + ".jsonl"] for name in ("membership", "lineage")}
                | {"report": refs["REPORT.json"]},
                "counts": counts,
                "coverage": coverage,
                "profile_id": pair[0],
                "policy_version": pair[1],
                **(
                    {
                        "normalization": {
                            name: report[name]
                            for name in (
                                "normalization_source_count",
                                "normalization_accepted_count",
                                "normalization_rejected_count",
                                "normalization_rejection_reasons",
                                "derived_edge_count",
                                "linkage_status",
                            )
                        }
                    }
                    if derived
                    else {}
                ),
                "privilege_profile": PRIVILEGE_PROFILE,
                "image_uri": catalog.image_uri,
                "source_commit": catalog.source_commit,
            }
        except Exception as error:
            return {
                **base,
                "status": "verification_failed",
                "error_type": type(error).__name__,
                **rejection_data(error),
            }
