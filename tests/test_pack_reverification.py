"""A model pack is read and hashed once per process; later passes re-check the exact version by HEAD.

At production size the adopted pack is four shards of 5.28, 5.34, 5.37 and 3.33 GB plus small files, and S3 holds no
SHA-256 for any of them, so every unpinned verification reads every byte. The diagnosis of the accepted-run stall found
each `restore`, `restore_original` and observation re-hashing 17.99 GiB (two or three passes per 15 s loop iteration,
9 to 12.5 minutes each). The fake store below has that shape: lazy bodies that count the bytes read and advance a virtual
clock at a measured rate, HEAD answers with ETag and CRC64NVME but no SHA-256, and a counting stand-in for `hashlib.sha256`
so no real gigabyte is hashed. Small objects carry real bytes and real hashes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_model_pack_adoption import ACCOUNT, BUCKET, DDB, PACK, SESSION, STS, Model, subject

from amlas.backends.adapters import coding_preparation as prep
from amlas.backends.adapters import dataset_acquisition as data_backend
from amlas.backends.adapters import model_weight_acquisition as model_backend
from amlas.backends.adapters import qwen_sft
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane import model_pack_adoption as adoption
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy

MIB = 1024 * 1024
RATE = 36 * MIB
SHARDS = (5_276_436_216, 5_335_161_512, 5_368_717_440, 3_325_995_712)
PACK_BYTES = sum(SHARDS)
POLICY = ApprovalPolicy(mode=ApprovalMode.AUTO)
SMALL = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.lock = threading.Lock()

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        with self.lock:
            self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    stub = SimpleNamespace(monotonic=value.monotonic)
    for module in (prep, qwen_sft, model_backend, data_backend, adoption):
        monkeypatch.setattr(module, "time", stub)
    return value


class Marked(bytes):
    """A chunk of synthetic zeros that carries the digest its whole object is declared to have."""

    sha = None


class CountingSha:
    total = 0

    def __init__(self, data=b""):
        self._real, self._pin, self._count = hashlib.sha256(), None, 0
        if data:
            self.update(data)

    def update(self, chunk):
        if isinstance(chunk, Marked):
            CountingSha.total += len(chunk)
            self._pin = chunk.sha
        else:
            self._real.update(chunk)

    def hexdigest(self):
        return self._pin if self._pin is not None else self._real.hexdigest()

    def digest(self):
        return bytes.fromhex(self.hexdigest())


class HashlibStandIn:
    sha256 = staticmethod(CountingSha)

    def __getattr__(self, name):
        return getattr(hashlib, name)


@pytest.fixture(autouse=True)
def counting_sha(monkeypatch):
    monkeypatch.setattr(qwen_sft, "hashlib", HashlibStandIn())
    CountingSha.total = 0
    yield


class Virtual:
    """A large object that exists only as a size and a declared digest."""

    def __init__(self, size, sha):
        self.size, self.sha = size, sha
        self.full = Marked(b"\0" * MIB)
        self.full.sha = sha


class LazyBody:
    def __init__(self, store, obj, hook=None):
        self.store, self.obj, self.pos, self.closed, self.hook = store, obj, 0, False, hook

    def read(self, size=-1):
        if self.closed:
            raise ValueError("ordinary closed body")
        if self.hook is not None:
            self.hook(self)
        left = self.obj.size - self.pos
        count = left if size is None or size < 0 else min(size, left)
        self.pos += count
        if count == 0:
            return b""
        with self.store.lock:
            self.store.big_bytes += count
        self.store.clock.advance(count / RATE)
        if count == MIB:
            return self.obj.full
        chunk = Marked(b"\0" * count)
        chunk.sha = self.obj.sha
        return chunk

    def close(self):
        self.closed = True


class PackS3:
    """Versioned store with HEAD answers shaped like S3: ETag and CRC64NVME, no SHA-256."""

    meta = SimpleNamespace(region_name="us-west-2")

    def __init__(self, clock):
        self.clock, self.lock = clock, threading.Lock()
        self.objects, self.versions = {}, {}
        self.heads, self.gets = [], []
        self.big_bytes = 0
        self.etag_override, self.size_override, self.missing = {}, {}, set()
        self.body_hook = None

    def _etag(self, key, version, obj):
        if (key, version) in self.etag_override:
            return self.etag_override[(key, version)]
        sha, size = (
            (obj.sha, obj.size) if isinstance(obj, Virtual) else (hashlib.sha256(obj).hexdigest(), len(obj))
        )
        return '"' + sha[:32] + "-" + str(size) + '"'

    def put(self, key, obj, fmt="json"):
        version = f"v{len(self.objects) + 1}"
        self.objects[(key, version)] = obj
        self.versions[key] = version
        if isinstance(obj, Virtual):
            sha, size = obj.sha, obj.size
        else:
            sha, size = hashlib.sha256(obj).hexdigest(), len(obj)
        return DatasetArtifactRef("s3://" + BUCKET + "/" + key, version, sha, size, fmt)

    def head_object(self, **kw):
        assert kw["ExpectedBucketOwner"] == ACCOUNT and kw["Bucket"] == BUCKET
        key, version = kw["Key"], kw.get("VersionId") or self.versions[kw["Key"]]
        with self.lock:
            self.heads.append((key, version))
        if (key, version) in self.missing:
            raise FileNotFoundError("closed exact-version deletion")
        obj = self.objects[(key, version)]
        size = obj.size if isinstance(obj, Virtual) else len(obj)
        return {
            "VersionId": version,
            "ContentLength": self.size_override.get((key, version), size),
            "ETag": self._etag(key, version, obj),
            "ChecksumType": "FULL_OBJECT",
            "ChecksumCRC64NVME": base64.b64encode(
                hashlib.sha256(self._etag(key, version, obj).encode()).digest()[:8]
            ).decode(),
        }

    def get_object(self, **kw):
        assert kw["ExpectedBucketOwner"] == ACCOUNT and kw["Bucket"] == BUCKET
        key, version = kw["Key"], kw["VersionId"]
        with self.lock:
            self.gets.append((key, version))
        obj = self.objects[(key, version)]
        if isinstance(obj, Virtual):
            return {
                "VersionId": version,
                "ContentLength": obj.size,
                "ETag": self._etag(key, version, obj),
                "Body": LazyBody(self, obj, self.body_hook),
            }
        return {
            "VersionId": version,
            "ContentLength": len(obj),
            "ETag": self._etag(key, version, obj),
            "Body": io.BytesIO(obj),
        }

    def put_object(self, **kw):
        assert kw["IfNoneMatch"] == "*" and kw["ExpectedBucketOwner"] == ACCOUNT
        return {"VersionId": self.put(kw["Key"], kw["Body"]).version_id}


class CopyForAdopt:
    """The adoption copy step; verification is the code under test and is not repeated here."""

    def __init__(self, s3):
        self.s3 = s3

    def _copy(self, ref, key, catalog, deadline):
        obj = self.s3.objects[(ref.uri.split("/", 3)[3], ref.version_id)]
        return replace(self.s3.put(key, obj), format=ref.format)

    def _raw(self, ref, catalog, deadline, collect=False):
        return None

    _verified = _raw

    def qualified_io(self, current):
        from contextlib import nullcontext

        return nullcontext()


class Pack:
    """Producer pack, committed adoption and the consumers that restore it."""

    def __init__(self, monkeypatch, clock):
        self.m, self.clock = subject(), clock
        self.s3, self.ddb = PackS3(clock), DDB()
        files = []
        for index, size in enumerate(SHARDS):
            name = f"model-0000{index + 1}-of-00004.safetensors"
            sha = hashlib.sha256(b"declared digest of " + name.encode()).hexdigest()
            ref = self.s3.put("sessions/producer/model/" + name, Virtual(size, sha), "safetensors")
            files.append({**asdict(ref), "path": name})
        for name in SMALL:
            ref = self.s3.put(
                "sessions/producer/model/" + name, b"ordinary small model file " + name.encode()
            )
            files.append({**asdict(ref), "path": name})
        manifests = {
            kind: asdict(
                self.s3.put("sessions/producer/model/" + kind + ".json", b"ordinary " + kind.encode())
            )
            for kind in ("commit", "lineage", "receipt")
        }
        self.result = {"success": True, "acquired": True, "file_refs": files, "manifest_refs": manifests}
        producer = {
            "request_ref": "model_acq_" + "b" * 32,
            "id": "fixture-lab-model-acq-" + "b" * 12,
            "arn": "arn:aws:sagemaker:us-west-2:"
            + ACCOUNT
            + ":processing-job/fixture-lab-model-acq-"
            + "b" * 12,
            "backend": "model_processing",
            "bindings": {"runtime_session_id": "producer-session"},
            "bucket": BUCKET,
        }
        row = {
            "pack_ref": PACK,
            "catalog_id": "qwen3_5_9b",
            "catalog_version": "1",
            "producer": producer,
            "artifacts": self.result,
        }
        raw = json.dumps(
            {"schema_version": 1, "packs": [row]}, sort_keys=True, separators=(",", ":")
        ).encode()
        self.catref = self.s3.put("model-packs/catalog.json", raw)
        self.scope = {
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
            self.m,
            "_decode_original",
            lambda row, scope: SimpleNamespace(
                id=row["id"], arn=row["arn"], extra={"bindings": row["bindings"]}
            ),
        )
        adopter = self.consumer(CopyForAdopt(self.s3))
        self.snapshot = adopter.adopt(PACK, SESSION, POLICY)
        self.s3.heads.clear()
        self.s3.gets.clear()
        self.s3.big_bytes = 0
        CountingSha.total = 0

    def backend(self):
        return qwen_sft.SageMakerQwenSftBackend(sagemaker_client=object(), s3_client=self.s3)

    def consumer(self, copier=None):
        return self.m.ModelPackAdoptionProvider(
            catalog_ref=self.catref,
            scope=self.scope,
            s3=self.s3,
            ddb=self.ddb,
            sts=STS(),
            model_adapter=Model(self.result),
            copy_adapter=copier or self.backend(),
        )

    def restore(self, provider):
        started, before = self.clock.now, self.s3.big_bytes
        assert provider.restore(deepcopy(self.snapshot), SESSION, POLICY) == self.snapshot
        return self.s3.big_bytes - before, self.clock.now - started

    def staged_shards(self):
        return [r for r in self.snapshot["file_refs"] if r["path"].endswith(".safetensors")]


@pytest.fixture
def pack(monkeypatch, clock):
    return Pack(monkeypatch, clock)


def test_the_pack_has_the_production_shape(pack):
    assert [r["byte_count"] for r in pack.staged_shards()] == list(SHARDS)
    assert PACK_BYTES == 19_306_310_880 and round(PACK_BYTES / 1024**3, 2) == 17.98
    assert len(pack.snapshot["file_refs"]) == 9


def test_five_restores_on_one_provider_read_the_staged_pack_once(pack):
    provider = pack.consumer()
    reads, seconds = zip(*[pack.restore(provider) for _ in range(5)], strict=True)
    assert reads[0] == PACK_BYTES, "the first restore in a process reads every byte of the pack once"
    assert reads[1:] == (0, 0, 0, 0), "later restores re-hash the pack: " + repr(reads)
    assert seconds[0] >= PACK_BYTES / RATE - 1 and max(seconds[1:]) < 60, seconds
    assert CountingSha.total == PACK_BYTES


def test_each_later_restore_checks_every_staged_object_by_exact_version_head(pack):
    provider = pack.consumer()
    pack.restore(provider)
    pack.s3.heads.clear()
    pack.restore(provider)
    staged = {(r["uri"].split("/", 3)[3], r["version_id"]) for r in pack.snapshot["file_refs"]}
    staged |= {(r["uri"].split("/", 3)[3], r["version_id"]) for r in pack.snapshot["manifest_refs"].values()}
    assert staged <= set(pack.s3.heads), "a staged object was accepted without a HEAD on its exact version"
    assert pack.s3.big_bytes == PACK_BYTES


def test_a_second_provider_and_adapter_in_the_process_reuse_the_first_hash(pack):
    first, second = pack.consumer(), pack.consumer()
    assert pack.restore(first)[0] == PACK_BYTES
    reads, seconds = pack.restore(second)
    assert reads == 0 and seconds < 60, "a fresh adapter re-hashed the pack"


def test_a_changed_etag_forces_a_full_read_of_that_object_only(pack):
    provider = pack.consumer()
    pack.restore(provider)
    shard = pack.staged_shards()[1]
    key = shard["uri"].split("/", 3)[3]
    pack.s3.etag_override[(key, shard["version_id"])] = '"replaced"'
    assert pack.restore(provider)[0] == SHARDS[1]
    assert pack.restore(provider)[0] == 0, "the re-read must record the new fingerprint"


def test_a_deleted_exact_version_refuses_without_reading(pack):
    provider = pack.consumer()
    pack.restore(provider)
    shard = pack.staged_shards()[0]
    pack.s3.missing.add((shard["uri"].split("/", 3)[3], shard["version_id"]))
    before = pack.s3.big_bytes
    with pytest.raises(FileNotFoundError):
        provider.restore(deepcopy(pack.snapshot), SESSION, POLICY)
    assert pack.s3.big_bytes == before
    pack.s3.missing.clear()
    assert pack.restore(provider)[0] == 0, "a failed HEAD must not cost the pin"


def test_a_truncated_size_refuses_even_for_a_pinned_object(pack):
    provider = pack.consumer()
    pack.restore(provider)
    shard = pack.staged_shards()[2]
    pack.s3.size_override[(shard["uri"].split("/", 3)[3], shard["version_id"])] = SHARDS[2] - 1
    with pytest.raises(ValueError, match="size differs"):
        provider.restore(deepcopy(pack.snapshot), SESSION, POLICY)


def test_bytes_that_do_not_hash_to_the_pin_are_refused_and_never_pinned(pack):
    shard = pack.staged_shards()[3]
    key = shard["uri"].split("/", 3)[3]
    wrong = Virtual(SHARDS[3], hashlib.sha256(b"different bytes").hexdigest())
    good = pack.s3.objects[(key, shard["version_id"])]
    pack.s3.objects[(key, shard["version_id"])] = wrong
    provider = pack.consumer()
    for _ in range(2):
        with pytest.raises(ValueError, match="raw SHA differs"):
            provider.restore(deepcopy(pack.snapshot), SESSION, POLICY)
    pack.s3.objects[(key, shard["version_id"])] = good
    assert pack.restore(provider)[0] > 0


def test_two_threads_asking_for_the_same_object_hash_it_once(pack):
    first, second = pack.consumer(), pack.consumer()
    started, release = threading.Event(), threading.Event()

    def hook(body):
        if not started.is_set():
            started.set()
            release.wait(10)

    pack.s3.body_hook = hook
    outcome = {}

    def run(name, provider):
        outcome[name] = provider.restore(deepcopy(pack.snapshot), SESSION, POLICY)

    one = threading.Thread(target=run, args=("one", first))
    one.start()
    assert started.wait(10)
    two = threading.Thread(target=run, args=("two", second))
    two.start()
    two.join(0.5)
    assert two.is_alive(), "the second asker should wait for the first hash of the same object"
    release.set()
    one.join(30)
    two.join(30)
    assert outcome["one"] == outcome["two"] == pack.snapshot
    assert pack.s3.big_bytes == PACK_BYTES


def test_a_pin_is_recorded_only_for_a_get_that_matches_the_head_etag(pack):
    shard = pack.staged_shards()[0]
    key = shard["uri"].split("/", 3)[3]
    original = pack.s3.get_object

    def without_etag(**kw):
        result = original(**kw)
        if kw["Key"] == key:
            result.pop("ETag")
        return result

    pack.s3.get_object = without_etag
    provider = pack.consumer()
    assert pack.restore(provider)[0] == PACK_BYTES
    assert pack.restore(provider)[0] == SHARDS[0], (
        "a read whose response carried no ETag must not become a pin"
    )
