"""A model pack is read in 1 MiB chunks; its remote full checks run by time, not by chunk.

The adopted pack is 19.3 GB. Every re-verification of it (the adoption restore, the cold re-hash of the original,
the SFT input binding, the post-SFT output read) reads it in 1 MiB chunks. A bracket of full checks around each
chunk (an STS read, a table description and the caller's own qualifier, twice) turned one pass into 88 minutes.
A `RecordRecheck` qualifier now runs its cheap local variant between chunks and its full check once per interval
counted from the start of the last full one. Plain callbacks, the scope checks (process, lifetime, thread), every
object-level bracket and every refusal are unchanged. Synthetic bytes and neutral stand-ins only; the clock is virtual.
"""

from __future__ import annotations

import functools
import hashlib
import math
import threading
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_model_pack_adoption import ACCOUNT, BUCKET, DDB, PACK, S3, SESSION, STS, Copier, Model

from amlas.backends.adapters import coding_preparation as prep
from amlas.backends.adapters import dataset_acquisition as data_backend
from amlas.backends.adapters import model_weight_acquisition as model_backend
from amlas.backends.adapters import qwen_sft
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane import model_pack_adoption as adoption
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore

MIB = 1024 * 1024
INTERVAL = 1.0
READ_SECONDS = 0.032  # about 31 MiB/s, the rate measured for the adopted pack
CHECK_SECONDS = 0.03  # one STS or DescribeTable round trip
CATALOG = SimpleNamespace(region="us-west-2", bucket=BUCKET, account_id=ACCOUNT)
SFT_SCOPE_REFUSAL = "expired or foreign-thread SFT observation refused"
MODEL_SCOPE_REFUSAL = "expired or foreign-thread model IO scope refused"
# Brackets around objects (HEAD, GET, close) are few and independent of the size.
RAW_FIXED = 8


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    stub = SimpleNamespace(monotonic=value.monotonic)
    for module in (prep, qwen_sft, model_backend, data_backend, adoption):
        monkeypatch.setattr(module, "time", stub)
    return value


@functools.cache
def pack(mib):
    raw = (b"synthetic-bytes!" * (MIB // 16)) * mib
    return raw, hashlib.sha256(raw).hexdigest()


class Body:
    def __init__(self, raw, clock, seconds_per_mib=0.0, hook=None):
        self.view, self.pos, self.clock = memoryview(raw), 0, clock
        self.cost, self.hook = seconds_per_mib, hook
        self.reads, self.closed = 0, False

    def read(self, size=-1):
        if self.closed:
            raise ValueError("ordinary closed body")
        self.reads += 1
        if self.hook is not None:
            self.hook(self.reads)
        left = len(self.view) - self.pos
        count = left if size is None or size < 0 else min(size, left)
        chunk = bytes(self.view[self.pos : self.pos + count])
        self.pos += count
        self.clock.advance(self.cost * count / MIB)
        return chunk

    def close(self):
        self.closed = True


class Objects:
    """One synthetic pack object with no service checksum, so every pass must read the bytes."""

    meta = SimpleNamespace(region_name="us-west-2")

    def __init__(self, clock, raw, seconds_per_mib=0.0, hook=None):
        self.clock, self.raw, self.cost, self.hook = clock, raw, seconds_per_mib, hook
        self.bodies = []

    def head_object(self, **kwargs):
        return {"VersionId": "pack-v1", "ContentLength": len(self.raw)}

    def get_object(self, **kwargs):
        body = Body(self.raw, self.clock, self.cost, self.hook)
        self.bodies.append(body)
        return {"VersionId": "pack-v1", "ContentLength": len(self.raw), "Body": body}


class Check:
    """A qualifier step: costs virtual seconds, counts its calls and can refuse from a given time."""

    def __init__(self, clock, cost=0.0):
        self.clock, self.cost = clock, cost
        self.calls, self.refuse_from = 0, None

    def __call__(self):
        self.calls += 1
        if self.refuse_from is not None and self.clock.now >= self.refuse_from:
            raise PermissionError("ordinary synthetic refusal")
        self.clock.advance(self.cost)


def recheck(clock, full_cost=0.06):
    full, local = Check(clock, full_cost), Check(clock)
    return prep.RecordRecheck(full, local, INTERVAL), full, local


def ref_of(mib):
    raw, sha = pack(mib)
    return DatasetArtifactRef(
        "s3://" + BUCKET + "/pack/model-00001.safetensors", "pack-v1", sha, len(raw), "safetensors"
    )


# -- the helper that the chunk loops share --


def test_record_step_asks_a_record_recheck_for_its_cheap_variant_and_runs_any_other_callback_in_full(clock):
    plain = Check(clock)
    prep.record_step(plain)
    prep.record_step(plain)
    assert plain.calls == 2
    callback, full, local = recheck(clock)
    callback()
    prep.record_step(callback)
    assert (full.calls, local.calls) == (1, 1)


# -- SageMakerQwenSftBackend._raw: SFT input binding, post-copy re-hash, restore_original, staged copies --


def sft_pass(clock, mib, qualifier, seconds_per_mib=0.0, hook=None, backend_hook=None):
    raw, _sha = pack(mib)
    objects = Objects(clock, raw, seconds_per_mib, hook)
    backend = qwen_sft.SageMakerQwenSftBackend(sagemaker_client=object(), s3_client=objects)
    if backend_hook is not None:
        backend_hook(backend)
    start = clock.now
    try:
        with backend.qualified_io(qualifier):
            backend._raw(ref_of(mib), CATALOG, clock.now + 3600)
    finally:
        sft_pass.last = (objects.bodies[0], clock.now - start)
    return sft_pass.last


def test_plain_callback_still_runs_for_every_chunk_bracket(clock):
    plain = Check(clock)
    body, _elapsed = sft_pass(clock, 64, plain)
    assert body.reads == 65
    assert plain.calls >= 2 * body.reads


def test_one_pass_over_64_mib_costs_full_checks_by_time_not_by_chunk(clock):
    callback, full, local = recheck(clock)
    body, elapsed = sft_pass(clock, 64, callback, READ_SECONDS)
    assert body.reads == 65
    assert full.calls <= math.ceil(elapsed / INTERVAL) + RAW_FIXED, (full.calls, elapsed)
    assert full.calls + local.calls >= 2 * body.reads  # every chunk bracket still asks the callback


def test_full_checks_do_not_grow_with_the_size_of_the_object(clock):
    counts = {}
    for mib in (1, 64):
        callback, full, _local = recheck(clock, full_cost=0.0)
        sft_pass(clock, mib, callback)
        counts[mib] = full.calls
    assert counts[64] <= counts[1] + 1, counts


def test_nested_record_rechecks_each_run_by_time(clock):
    outer, outer_full, _ = recheck(clock)
    inner, inner_full, _ = recheck(clock)
    raw, _sha = pack(64)
    objects = Objects(clock, raw, READ_SECONDS)
    backend = qwen_sft.SageMakerQwenSftBackend(sagemaker_client=object(), s3_client=objects)
    start = clock.now
    with backend.qualified_io(outer), backend.qualified_io(inner):
        backend._raw(ref_of(64), CATALOG, clock.now + 3600)
    bound = math.ceil((clock.now - start) / INTERVAL) + 2 * RAW_FIXED
    assert outer_full.calls <= bound and inner_full.calls <= bound, (outer_full.calls, inner_full.calls)


def test_local_refusal_stops_the_pass_at_the_next_chunk(clock):
    callback, _full, local = recheck(clock)
    local.refuse_from = clock.now + 1.0
    with pytest.raises(PermissionError):
        sft_pass(clock, 64, callback, READ_SECONDS)
    body, _elapsed = sft_pass.last
    assert body.reads <= math.ceil(1.0 / READ_SECONDS) + 3 and body.closed


def test_full_refusal_stops_the_pass_within_one_interval(clock):
    callback, full, _local = recheck(clock)
    full.refuse_from = clock.now + 1.5
    with pytest.raises(PermissionError):
        sft_pass(clock, 64, callback, READ_SECONDS)
    body, _elapsed = sft_pass.last
    assert body.reads <= math.ceil((1.5 + INTERVAL) / READ_SECONDS) + 4 and body.closed


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("scope_closed", SFT_SCOPE_REFUSAL),
        ("foreign_thread", SFT_SCOPE_REFUSAL),
        ("inherited_process", "inherited SFT observation refused"),
    ],
)
def test_scope_checks_still_run_for_every_chunk(clock, mutation, message):
    box = {}

    def hook(index):
        if index != 5:
            return
        backend = box["backend"]
        scope = backend._qualified_io.get()
        if mutation == "scope_closed":
            scope["active"] = False
        elif mutation == "foreign_thread":
            scope["thread_id"] = threading.get_ident() + 1
        else:
            backend._qualified_io_pid = -1

    callback, _full, _local = recheck(clock)
    with pytest.raises(ValueError, match=message):
        sft_pass(clock, 64, callback, READ_SECONDS, hook, lambda backend: box.update(backend=backend))
    assert sft_pass.last[0].reads == 5  # refused by the bracket right after the fifth read


# -- ProcessingModelWeightAcquisitionBackend._payload_observed: the cold re-hash of the original --


def model_pass(clock, mib, qualifier, seconds_per_mib=0.0, hook=None, backend_hook=None):
    raw, sha = pack(mib)
    objects = Objects(clock, raw, seconds_per_mib, hook)
    backend = model_backend.ProcessingModelWeightAcquisitionBackend(
        sagemaker_client=SimpleNamespace(meta=Objects.meta), s3_client=objects
    )
    if backend_hook is not None:
        backend_hook(backend)
    handle = SimpleNamespace(
        extra={"region": "us-west-2", "bucket": BUCKET, "output_prefix": "out/", "account_id": ACCOUNT}
    )
    ref = {"path": "model-00001.safetensors", "byte_count": len(raw), "sha256": sha, "role": "weight_shard"}
    start = clock.now
    try:
        with backend.qualified_io(qualifier):
            result = backend._payload_observed(handle, ref, clock.now + 3600)
    finally:
        model_pass.last = (objects.bodies[0] if objects.bodies else None, clock.now - start)
    return result


def test_model_plain_callback_still_runs_for_every_chunk_bracket(clock):
    plain = Check(clock)
    _ref, raw_verified = model_pass(clock, 64, plain)
    assert raw_verified is True
    assert plain.calls >= 2 * model_pass.last[0].reads


def test_cold_model_hash_over_64_mib_costs_full_checks_by_time_not_by_chunk(clock):
    callback, full, local = recheck(clock)
    _ref, raw_verified = model_pass(clock, 64, callback, READ_SECONDS)
    body, elapsed = model_pass.last
    assert raw_verified is True and body.reads == 65
    assert full.calls <= math.ceil(elapsed / INTERVAL) + RAW_FIXED, (full.calls, elapsed)
    assert full.calls + local.calls >= 2 * body.reads


def test_cold_model_hash_full_checks_do_not_grow_with_the_size(clock):
    counts = {}
    for mib in (1, 64):
        callback, full, _local = recheck(clock, full_cost=0.0)
        model_pass(clock, mib, callback)
        counts[mib] = full.calls
    assert counts[64] <= counts[1] + 1, counts


def test_cold_model_hash_local_refusal_stops_at_the_next_chunk(clock):
    callback, _full, local = recheck(clock)
    local.refuse_from = clock.now + 1.0
    with pytest.raises(PermissionError):
        model_pass(clock, 64, callback, READ_SECONDS)
    body, _elapsed = model_pass.last
    assert body.reads <= math.ceil(1.0 / READ_SECONDS) + 3 and body.closed


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("scope_closed", MODEL_SCOPE_REFUSAL),
        ("foreign_thread", MODEL_SCOPE_REFUSAL),
        ("inherited_process", "inherited model adapter IO refused"),
    ],
)
def test_cold_model_hash_scope_checks_still_run_for_every_chunk(clock, mutation, message):
    box = {}

    def hook(index):
        if index != 5:
            return
        backend = box["backend"]
        scope = backend._qualified_io.get()
        if mutation == "scope_closed":
            scope["active"] = False
        elif mutation == "foreign_thread":
            scope["thread_id"] = threading.get_ident() + 1
        else:
            backend._qualified_io_pid = -1

    callback, _full, _local = recheck(clock)
    with pytest.raises(ValueError, match=message):
        model_pass(clock, 64, callback, READ_SECONDS, hook, lambda backend: box.update(backend=backend))
    assert model_pass.last[0].reads == 5


# -- the other byte readers under the same scopes: acquired data and preparation files of the SFT parent capture,
# -- SFT output files


def data_pass(clock, mib, qualifier, seconds_per_mib=0.0):
    raw, sha = pack(mib)
    objects = Objects(clock, raw, seconds_per_mib)
    backend = data_backend.ProcessingDatasetAcquisitionBackend(
        sagemaker_client=SimpleNamespace(meta=Objects.meta), s3_client=objects
    )
    handle = SimpleNamespace(
        extra={"region": "us-west-2", "bucket": BUCKET, "output_prefix": "out/", "account_id": ACCOUNT}
    )
    start = clock.now
    try:
        with backend.qualified_io(qualifier):
            backend._verify_version_payload(handle, "data/train.jsonl", "pack-v1", sha, len(raw))
    finally:
        data_pass.last = (objects.bodies[0], clock.now - start)


def output_pass(clock, mib, qualifier, seconds_per_mib=0.0):
    raw, sha = pack(mib)
    objects = Objects(clock, raw, seconds_per_mib)
    backend = qwen_sft.SageMakerQwenSftBackend(sagemaker_client=object(), s3_client=objects)
    start = clock.now
    try:
        with backend.qualified_io(qualifier):
            ref, _none = backend._output(
                SimpleNamespace(extra={"catalog": CATALOG}),
                "out/model.safetensors",
                clock.now + 3600,
                limit=len(raw) + 1,
            )
    finally:
        output_pass.last = (objects.bodies[0], clock.now - start)
    assert ref.sha256 == sha


def prep_pass(clock, mib, qualifier, seconds_per_mib=0.0):
    """The preparation adapter's exact-version raw reader: the SFT parent capture hashes preparation files with it."""
    raw, sha = pack(mib)
    objects = Objects(clock, raw, seconds_per_mib)
    backend = prep.ProcessingCodingPreparationBackend(sagemaker_client=object(), s3_client=objects)
    start = clock.now
    try:
        with backend.qualified_io(qualifier):
            backend._raw(
                objects, BUCKET, "prep/train.jsonl", "pack-v1", ACCOUNT, len(raw), sha, count_lines=False
            )
    finally:
        prep_pass.last = (objects.bodies[0], clock.now - start)


@pytest.mark.parametrize("reader", [data_pass, output_pass, prep_pass])
def test_other_chunked_readers_check_by_time_not_by_chunk(clock, reader):
    callback, full, local = recheck(clock)
    reader(clock, 64, callback, READ_SECONDS)
    body, elapsed = reader.last
    assert body.reads == 65
    assert full.calls <= math.ceil(elapsed / INTERVAL) + RAW_FIXED, (full.calls, elapsed)
    assert full.calls + local.calls >= 2 * body.reads


@pytest.mark.parametrize("reader", [data_pass, output_pass, prep_pass])
def test_other_chunked_readers_still_run_a_plain_callback_for_every_bracket(clock, reader):
    plain = Check(clock)
    reader(clock, 64, plain)
    assert plain.calls >= 2 * reader.last[0].reads


@pytest.mark.parametrize("reader", [data_pass, output_pass, prep_pass])
def test_other_chunked_readers_stop_at_a_local_refusal(clock, reader):
    callback, _full, local = recheck(clock)
    local.refuse_from = clock.now + 1.0
    with pytest.raises(PermissionError):
        reader(clock, 64, callback, READ_SECONDS)
    body, _elapsed = reader.last
    assert body.reads <= math.ceil(1.0 / READ_SECONDS) + 3 and body.closed


# -- ModelPackAdoptionProvider: adopt, restore, and the model-poll path through the store --


class PackS3(S3):
    """The shared fake plus large synthetic objects without a service checksum, and a multipart copy."""

    meta = SimpleNamespace(region_name="us-west-2")

    def __init__(self, clock, seconds_per_mib=0.0, hook=None):
        super().__init__()
        self.clock, self.cost, self.hook = clock, seconds_per_mib, hook
        self.big, self.bodies, self.uploads = {}, [], 0

    def put_big(self, key, mib):
        raw, sha = pack(mib)
        version = "big" + str(len(self.big) + 1)
        self.big[(key, version)] = raw
        return DatasetArtifactRef("s3://" + BUCKET + "/" + key, version, sha, len(raw), "safetensors")

    def head_object(self, **kwargs):
        raw = self.big.get((kwargs["Key"], kwargs.get("VersionId")))
        if raw is None:
            return super().head_object(**kwargs)
        return {"VersionId": kwargs["VersionId"], "ContentLength": len(raw)}

    def get_object(self, **kwargs):
        raw = self.big.get((kwargs["Key"], kwargs.get("VersionId")))
        if raw is None:
            return super().get_object(**kwargs)
        body = Body(raw, self.clock, self.cost, self.hook)
        self.bodies.append(body)
        return {"Body": body, "VersionId": kwargs["VersionId"], "ContentLength": len(raw)}

    def create_multipart_upload(self, **kwargs):
        self.uploads += 1
        return {"UploadId": "upload-" + str(self.uploads), "Key": kwargs["Key"]}

    def upload_part_copy(self, **kwargs):
        source = kwargs["CopySource"]
        version = source.split("?versionId=", 1)[1]
        self._pending = (source.split("/", 1)[1].split("?", 1)[0], version)
        return {"CopySourceVersionId": version, "CopyPartResult": {"ETag": "etag"}}

    def complete_multipart_upload(self, **kwargs):
        key, version = self._pending
        new = "big" + str(len(self.big) + len(self.objects) + 1)
        if (key, version) in self.big:
            self.big[(kwargs["Key"], new)] = self.big[(key, version)]
        else:
            self.objects[(kwargs["Key"], new)] = self.objects[(key, version)]
        return {"VersionId": new}

    def abort_multipart_upload(self, **kwargs):
        return {}


class CountingSTS(STS):
    def __init__(self, clock, cost=CHECK_SECONDS):
        self.clock, self.cost, self.calls, self.refuse_from = clock, cost, 0, None

    def get_caller_identity(self):
        self.calls += 1
        if self.refuse_from is not None and self.clock.now >= self.refuse_from:
            raise PermissionError("ordinary synthetic refusal")
        self.clock.advance(self.cost)
        return super().get_caller_identity()


class CountingDDB(DDB):
    def __init__(self, clock, cost=CHECK_SECONDS):
        super().__init__()
        self.clock, self.cost, self.describes, self.table_id = clock, cost, 0, "table-incarnation"

    def describe_table(self, **kwargs):
        self.describes += 1
        self.clock.advance(self.cost)
        reply = super().describe_table(**kwargs)
        reply["Table"]["TableId"] = self.table_id
        return reply


def pack_case(monkeypatch, clock, mib, seconds_per_mib=0.0, hook=None):
    """Adopt a synthetic pack once (phase one), then hand back what a later restore needs (phase two)."""
    s3, ddb = PackS3(clock, seconds_per_mib, hook), CountingDDB(clock)
    source = s3.put_big("sessions/producer/model/model-00001.safetensors", mib)
    manifests = {
        kind: s3.put("sessions/producer/model/" + kind + ".json", b"ordinary provenance")
        for kind in ("commit", "lineage", "receipt")
    }
    result = {
        "success": True,
        "acquired": True,
        "file_refs": [{**asdict(source), "path": "model-00001.safetensors"}],
        "manifest_refs": {kind: asdict(ref) for kind, ref in manifests.items()},
    }
    producer = {
        "request_ref": "model_acq_" + "b" * 32,
        "id": "fixture-lab-model-acq-" + "b" * 12,
        "arn": "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/fixture-lab-model-acq-" + "b" * 12,
        "backend": "model_processing",
        "bindings": {"runtime_session_id": "producer-session"},
        "bucket": BUCKET,
    }
    row = {
        "pack_ref": PACK,
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "producer": producer,
        "artifacts": result,
    }
    import json

    catalog_ref = s3.put(
        "model-packs/catalog.json",
        json.dumps({"schema_version": 1, "packs": [row]}, sort_keys=True, separators=(",", ":")).encode(),
    )
    scope = {
        "account_id": ACCOUNT,
        "region": "us-west-2",
        "bucket": BUCKET,
        "jobs_table": "jobs",
        "execution_role_arn": "arn:aws:iam::" + ACCOUNT + ":role/runtime",
        "execution_role_id": "AROATEST",
        "deployment_id": "fd0d82ab-4e03-44c0-9bf1-908af5b2b031",
        "deployment_prefix": "fixture-lab",
    }
    monkeypatch.setattr(
        adoption,
        "_decode_original",
        lambda value, _scope: SimpleNamespace(
            id=value["id"], arn=value["arn"], extra={"bindings": value["bindings"]}
        ),
    )
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO)
    return SimpleNamespace(
        s3=s3,
        ddb=ddb,
        result=result,
        scope=scope,
        catalog_ref=catalog_ref,
        policy=policy,
        mib=mib,
        board={"scope": (scope, catalog_ref)},
    )


class PackCopier(Copier):
    """Server-side copy stand-in for phase one: registers the staged copy without reading it."""

    def _copy(self, ref, key, catalog, deadline):
        old = ref.uri.split("/", 3)[3]
        if (old, ref.version_id) in self.s3.big:
            self.copies += 1
            version = "big" + str(len(self.s3.big) + 1)
            self.s3.big[(key, version)] = self.s3.big[(old, ref.version_id)]
            return replace(ref, uri="s3://" + BUCKET + "/" + key, version_id=version)
        return super()._copy(ref, key, catalog, deadline)

    def _raw(self, ref, catalog, deadline, collect=False):
        if (ref.uri.split("/", 3)[3], ref.version_id) in self.s3.big:
            return None
        return super()._raw(ref, catalog, deadline, collect)


def provider_for(case, clock, copy_adapter, sts=None, with_getter=True):
    getter = (lambda: case.board["scope"]) if with_getter else None
    return adoption.ModelPackAdoptionProvider(
        catalog_ref=case.catalog_ref,
        scope=case.scope,
        s3=case.s3,
        ddb=case.ddb,
        sts=sts or CountingSTS(clock),
        model_adapter=Model(case.result),
        copy_adapter=copy_adapter,
        scope_getter=getter,
    )


def adopted(monkeypatch, clock, mib, seconds_per_mib=0.0, hook=None):
    case = pack_case(monkeypatch, clock, mib, seconds_per_mib, hook)
    case.snapshot = provider_for(case, clock, PackCopier(case.s3)).adopt(PACK, SESSION, case.policy)
    return case


def real_provider(case, clock, sts=None):
    copy_adapter = qwen_sft.SageMakerQwenSftBackend(sagemaker_client=object(), s3_client=case.s3)
    return provider_for(case, clock, copy_adapter, sts)


def restore_pass(case, clock, qualify=None, sts=None):
    sts = sts or CountingSTS(clock)
    case.sts = sts
    case.provider = real_provider(case, clock, sts)
    case.ddb.describes = 0
    case.s3.bodies.clear()
    start = clock.now
    try:
        return case.provider.restore(case.snapshot, SESSION, case.policy, qualify)
    finally:
        case.elapsed = clock.now - start


def reads_of(case):
    return sum(body.reads for body in case.s3.bodies)


def fixed_provider_checks(monkeypatch, clock, qualify=None):
    """STS reads of one restore of the smallest pack: the object-level brackets that do not depend on the size."""
    case = adopted(monkeypatch, clock, 1)
    restore_pass(case, clock, qualify)
    return case.sts.calls


def test_restore_provider_checks_do_not_grow_with_the_size_of_the_pack(monkeypatch, clock):
    counts = {}
    for mib in (1, 64):
        case = adopted(monkeypatch, clock, mib)
        restore_pass(case, clock)
        counts[mib] = (case.sts.calls, case.ddb.describes)
    assert counts[64][0] <= counts[1][0] + 1 and counts[64][1] <= counts[1][1] + 1, counts


def test_restore_of_64_mib_pack_costs_provider_checks_by_time_not_by_chunk(monkeypatch, clock):
    fixed = fixed_provider_checks(monkeypatch, clock)
    case = adopted(monkeypatch, clock, 64, READ_SECONDS)
    restore_pass(case, clock)
    assert reads_of(case) == 65
    bound = fixed + math.ceil(case.elapsed / INTERVAL) + 1
    assert case.sts.calls <= bound and case.ddb.describes <= bound, (
        case.sts.calls,
        case.ddb.describes,
        bound,
    )


def test_restore_runs_the_callers_record_recheck_by_time_and_asks_it_on_every_chunk(monkeypatch, clock):
    small, small_full, _ = recheck(clock, full_cost=0.02)
    fixed = fixed_provider_checks(monkeypatch, clock, small)
    callback, full, local = recheck(clock, full_cost=0.02)
    case = adopted(monkeypatch, clock, 64, READ_SECONDS)
    restore_pass(case, clock, callback)
    assert reads_of(case) == 65
    # One provider full check asks the caller's full check twice.
    assert full.calls <= small_full.calls + 2 * (math.ceil(case.elapsed / INTERVAL) + 1), (
        full.calls,
        small_full.calls,
    )
    assert full.calls + local.calls >= 2 * reads_of(case)
    assert case.sts.calls <= fixed + math.ceil(case.elapsed / INTERVAL) + 1


def test_restore_still_runs_a_plain_caller_qualifier_for_every_chunk_bracket(monkeypatch, clock):
    plain = Check(clock)
    fixed = fixed_provider_checks(monkeypatch, clock, Check(clock))
    case = adopted(monkeypatch, clock, 64, READ_SECONDS)
    restore_pass(case, clock, plain)
    assert plain.calls >= 2 * reads_of(case)
    assert (
        case.sts.calls <= fixed + math.ceil(case.elapsed / INTERVAL) + 1
    )  # its own checks follow the cadence


def test_adopt_of_a_64_mib_pack_checks_by_time_through_copy_and_staged_verification(monkeypatch, clock):
    small = pack_case(monkeypatch, clock, 1)
    sts = CountingSTS(clock)
    provider_for(
        small, clock, qwen_sft.SageMakerQwenSftBackend(sagemaker_client=object(), s3_client=small.s3), sts
    ).adopt(PACK, SESSION, small.policy)
    fixed = sts.calls
    big = pack_case(monkeypatch, clock, 64)
    big.s3.cost = READ_SECONDS
    sts = CountingSTS(clock)
    copy_adapter = qwen_sft.SageMakerQwenSftBackend(sagemaker_client=object(), s3_client=big.s3)
    start = clock.now
    snapshot = provider_for(big, clock, copy_adapter, sts).adopt(PACK, SESSION, big.policy)
    elapsed = clock.now - start
    assert snapshot["origin"] == "verified_pack_adoption"
    assert reads_of(big) == 2 * 65  # the post-copy re-hash and the staged verification
    assert sts.calls <= fixed + math.ceil(elapsed / INTERVAL) + 2, (sts.calls, fixed, elapsed)


# -- refusals that must still stop a restore, and how soon --


def refused_restore(monkeypatch, clock, arm, qualify=None, sts=None):
    case = adopted(monkeypatch, clock, 64, READ_SECONDS)
    case.s3.hook = arm(case)
    with pytest.raises((PermissionError, ValueError)) as caught:
        restore_pass(case, clock, qualify, sts)
    return case, caught.value


def at_read(index, action):
    return lambda i: action() if i == index else None


def test_a_caller_local_refusal_stops_the_restore_at_the_next_chunk(monkeypatch, clock):
    callback, _full, local = recheck(clock, full_cost=0.02)
    case, error = refused_restore(
        monkeypatch,
        clock,
        lambda case: at_read(5, lambda: setattr(local, "refuse_from", clock.now)),
        callback,
    )
    assert type(error) is PermissionError and reads_of(case) <= 6


def test_a_caller_full_refusal_stops_the_restore_within_one_interval(monkeypatch, clock):
    callback, full, _local = recheck(clock, full_cost=0.02)
    case, error = refused_restore(
        monkeypatch, clock, lambda case: at_read(5, lambda: setattr(full, "refuse_from", clock.now)), callback
    )
    assert type(error) is PermissionError
    assert reads_of(case) <= 5 + math.ceil((INTERVAL + 0.3) / READ_SECONDS) + 2


def test_a_revoked_policy_stops_the_restore_at_the_next_chunk(monkeypatch, clock):
    case, error = refused_restore(
        monkeypatch,
        clock,
        lambda case: at_read(5, lambda: setattr(case.policy, "mode", ApprovalMode.ASK)),
    )
    assert "policy denied" in str(error) and reads_of(case) <= 6


def test_a_changed_adoption_scope_stops_the_restore_at_the_next_chunk(monkeypatch, clock):
    def arm(case):
        def change():
            case.board["scope"] = ({**case.scope, "bucket": "other-bucket"}, case.catalog_ref)

        return at_read(5, change)

    case, error = refused_restore(monkeypatch, clock, arm)
    assert "adoption scope changed" in str(error) and reads_of(case) <= 6


def test_a_closed_chunk_scope_stops_the_restore_at_the_next_chunk(monkeypatch, clock):
    def arm(case):
        def close():
            case.provider.copy_adapter._qualified_io.get()["active"] = False

        return at_read(5, close)

    case, error = refused_restore(monkeypatch, clock, arm)
    assert SFT_SCOPE_REFUSAL in str(error) and reads_of(case) <= 6


def test_a_failing_identity_read_stops_the_restore_within_one_interval(monkeypatch, clock):
    sts = CountingSTS(clock)
    case, error = refused_restore(
        monkeypatch, clock, lambda case: at_read(5, lambda: setattr(sts, "refuse_from", clock.now)), None, sts
    )
    assert type(error) is PermissionError
    assert reads_of(case) <= 5 + math.ceil((INTERVAL + 0.3) / READ_SECONDS) + 2


def test_a_replaced_table_stops_the_restore_within_one_interval(monkeypatch, clock):
    case, error = refused_restore(
        monkeypatch,
        clock,
        lambda case: at_read(5, lambda: setattr(case.ddb, "table_id", "replaced-incarnation")),
    )
    assert "table incarnation changed" in str(error)
    assert reads_of(case) <= 5 + math.ceil((INTERVAL + 0.3) / READ_SECONDS) + 2


# -- the store: every poll of an adopted ref restores it, so it hands the provider the caller's cadence --


def store_for(case, policy_getter=None):
    return SessionModelAcquisitionStore(
        runtime_session_id=SESSION,
        adoption_provider=case.provider,
        policy_getter=policy_getter or (lambda: case.policy),
    )


def test_the_store_hands_the_provider_a_record_recheck_that_follows_the_callers_cadence(monkeypatch, clock):
    case = adopted(monkeypatch, clock, 1)
    case.provider = real_provider(case, clock)
    seen = []
    monkeypatch.setattr(case.provider, "restore", lambda *args: seen.append(args[3]) or {"stub": True})
    outer, outer_full, outer_local = recheck(clock)
    store_for(case)._verified_adoption({"stub": True}, qualify=outer)
    current = seen[0]
    assert type(current) is prep.RecordRecheck
    current()
    assert (outer_full.calls, outer_local.calls) == (1, 0)
    for _ in range(1000):
        clock.advance(0.0005)
        current.record_check()
    assert (outer_full.calls, outer_local.calls) == (1, 1000)
    clock.advance(INTERVAL)
    current.record_check()
    assert outer_full.calls == 2


def test_the_store_runs_a_plain_caller_qualifier_in_full_on_every_check(monkeypatch, clock):
    case = adopted(monkeypatch, clock, 1)
    case.provider = real_provider(case, clock)
    seen = []
    monkeypatch.setattr(case.provider, "restore", lambda *args: seen.append(args[3]) or {"stub": True})
    plain = Check(clock)
    store_for(case)._verified_adoption({"stub": True}, qualify=plain)
    for _ in range(10):
        seen[0].record_check()
    assert plain.calls == 10


def observe_pass(case, clock):
    case.provider = real_provider(case, clock)
    store = store_for(case)
    ref = "model_acq_" + "c" * 32
    store._adoptions[ref] = case.snapshot
    case.sts = case.provider.sts
    case.s3.bodies.clear()
    start = clock.now
    result = store.observe(ref)
    case.elapsed = clock.now - start
    return result


def test_every_poll_of_an_adopted_ref_checks_by_time_not_by_chunk(monkeypatch, clock):
    small = adopted(monkeypatch, clock, 1)
    assert observe_pass(small, clock)["status"] == "reused_verified"
    fixed = small.sts.calls
    case = adopted(monkeypatch, clock, 64, READ_SECONDS)
    assert observe_pass(case, clock)["status"] == "reused_verified"
    assert reads_of(case) == 65
    assert case.sts.calls <= fixed + math.ceil(case.elapsed / INTERVAL) + 1, (
        case.sts.calls,
        fixed,
        case.elapsed,
    )


def capture_pass(case, clock, qualify):
    """The SFT parent capture's second restore: the typed adopted model source for one alias."""
    case.provider = real_provider(case, clock)
    store = store_for(case)
    ref = "model_acq_" + "d" * 32
    store._adoptions[ref] = case.snapshot
    case.sts = case.provider.sts
    case.s3.bodies.clear()
    start = clock.now
    source = store._verified_sft_model_snapshot(ref, _qualify=qualify)
    case.elapsed = clock.now - start
    return source


def test_the_sft_capture_restore_checks_by_time_and_asks_the_capture_callback_on_every_chunk(
    monkeypatch, clock
):
    small_callback, small_full, _ = recheck(clock, full_cost=0.02)
    small = adopted(monkeypatch, clock, 1)
    capture_pass(small, clock, small_callback)
    callback, full, local = recheck(clock, full_cost=0.02)
    case = adopted(monkeypatch, clock, 64, READ_SECONDS)
    source = capture_pass(case, clock, callback)
    assert type(source).__name__ == "VerifiedSftAdoptedModelSource" and reads_of(case) == 65
    assert full.calls <= small_full.calls + 2 * (math.ceil(case.elapsed / INTERVAL) + 1), (
        full.calls,
        small_full.calls,
    )
    assert full.calls + local.calls >= 2 * reads_of(case)  # the pause witness and policy checks run per chunk
    assert case.sts.calls <= small.sts.calls + math.ceil(case.elapsed / INTERVAL) + 1


def test_the_sft_capture_restore_stops_at_the_capture_callbacks_local_refusal(monkeypatch, clock):
    callback, _full, local = recheck(clock, full_cost=0.02)
    case = adopted(monkeypatch, clock, 64, READ_SECONDS)
    case.s3.hook = at_read(5, lambda: setattr(local, "refuse_from", clock.now))
    with pytest.raises(PermissionError):
        capture_pass(case, clock, callback)
    assert reads_of(case) <= 6
