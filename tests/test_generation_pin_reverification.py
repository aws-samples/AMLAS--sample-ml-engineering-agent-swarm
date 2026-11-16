"""Generation observation re-checks each pinned model version without re-streaming the model bytes.

Every pin (the nine original model files and the staged copies, the prompts, the adapter files and RUN.json) is
hash-verified by `submit` in the process that builds the handle, and that verification is recorded in the handle with the
HEAD fingerprint it used. `artifacts` then confirms the same object version is still present with the same fingerprint,
confirms the staged key still resolves to the pinned version, and never re-reads the 36 GiB. A pin that has no matching
record is still verified in full, under a deadline that scales with its bytes, and any mismatch fails closed.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import time

import pytest
from test_managed_generation_product import installed_generation
from test_managed_generation_product import state as state
from test_qwen_sft_backend import BUCKET, S3, SESSION, Native, catalog, ref

from amlas.backends.adapters import qwen_sft
from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend
from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.backends.types import DatasetArtifactRef

MIB = 1024**2


def symbol(module, name):
    value = getattr(module, name, None)
    assert value is not None, "missing-symbol: " + name
    return value


def pins(handle):
    return [*handle.extra["staged_refs"].values(), *handle.extra["original_base_refs"].values()]


def pin_ids(handle):
    return {(artifact.uri, artifact.version_id) for artifact in pins(handle)}


@pytest.fixture
def raw_calls(monkeypatch):
    calls = []
    original = SageMakerQwenSftBackend._raw

    def spy(self, artifact, catalog_, deadline, *args, **kwargs):
        calls.append((artifact, deadline, kwargs.get("collect", False)))
        return original(self, artifact, catalog_, deadline, *args, **kwargs)

    monkeypatch.setattr(SageMakerQwenSftBackend, "_raw", spy)
    return calls


def head_log(s3, monkeypatch):
    log = []
    original = s3.head_object

    def head(**kw):
        log.append((kw["Key"], kw.get("VersionId")))
        return original(**kw)

    monkeypatch.setattr(s3, "head_object", head)
    return log


def test_observation_runs_no_hash_pass_over_the_pinned_model_files(state, raw_calls):
    backend, _native, _s3, handle, *_ = installed_generation(state)
    submitted = len(raw_calls)
    result = backend.artifacts(handle)
    assert result["success"] is result["generated"] is True and result["case_count"] == 164
    seen = {(artifact.uri, artifact.version_id) for artifact, *_ in raw_calls[submitted:]}
    assert not seen & pin_ids(handle), "pinned versions were hash-verified again at observation"


def test_each_pin_is_hash_verified_once_by_submit_across_observe_and_the_pair_chain(state, raw_calls):
    from amlas.control_plane.generation_state import SessionCodingGenerationStore

    store = state[0]
    backend, _native, _s3, handle, *_ = installed_generation(state)
    assert type(store) is SessionCodingGenerationStore
    counts = collections.Counter((a.uri, a.version_id) for a, *_ in raw_calls)
    ids = pin_ids(handle)
    assert all(counts[i] >= 1 for i in ids)
    submitted = {i: counts[i] for i in ids}
    store._adapter = backend
    reference = "gen_" + "1" * 32
    store._records[reference] = handle
    first = store.observe(reference)
    again = store.observe(reference)
    base = store._verified_evaluation_source(reference)
    tuned = store._verified_evaluation_source(reference)
    assert (
        first["generated"] is again["generated"] is True and base.producer_job_arn == tuned.producer_job_arn
    )
    after = collections.Counter((a.uri, a.version_id) for a, *_ in raw_calls)
    assert {i: after[i] for i in ids} == submitted, "observe or the pair chain re-hashed a pin"


def test_observation_rechecks_every_pinned_version_and_the_current_staged_version(state, monkeypatch):
    backend, _native, s3, handle, *_ = installed_generation(state)
    log = head_log(s3, monkeypatch)
    assert backend.artifacts(handle)["success"] is True
    staged = list(handle.extra["staged_refs"].values())
    for artifact in pins(handle):
        assert (artifact.uri.split("/", 3)[3], artifact.version_id) in log
    for artifact in staged:
        assert (artifact.uri.split("/", 3)[3], None) in log, "current version of a staged key not checked"
    originals = {a.uri.split("/", 3)[3] for a in handle.extra["original_base_refs"].values()}
    assert not any(key in originals and version is None for key, version in log), (
        "the original keys may be re-versioned by their producer; only the pinned version is checked"
    )


def test_submit_records_a_fingerprint_for_every_pin_and_adopted_handles_carry_it_too(state):
    pin_key = symbol(qwen_sft, "_pin_key")
    _backend, _native, s3, handle, *_ = installed_generation(state)
    recorded = handle.extra["verified_pins"]
    assert {pin_key(a) for a in pins(handle)} <= set(recorded)
    for artifact in pins(handle):
        head = s3.head_object(Bucket=BUCKET, Key=artifact.uri.split("/", 3)[3], VersionId=artifact.version_id)
        assert dict(recorded[pin_key(artifact)])["VersionId"] == head["VersionId"] == artifact.version_id
        assert dict(recorded[pin_key(artifact)])["ContentLength"] == head["ContentLength"]


def test_a_replacement_process_adopting_the_native_job_records_its_own_verification(state):
    pin_key = symbol(qwen_sft, "_pin_key")
    _backend, _native, s3, handle, *_ = installed_generation(state)
    replacement_native = Native()

    class InUseError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.response = {"Error": {"Code": "ResourceInUse"}}

    replacement_native.error = InUseError("neutral duplicate create")
    replacement = SageMakerCodingGenerationBackend(sagemaker_client=replacement_native, s3_client=s3)
    _store, _adapter, _reads, _s3, inputs, case = state
    prompts = DatasetArtifactRef(**handle.extra["run_binding"]["prompts_ref"])
    case = dataclasses.replace(
        case,
        file_refs=tuple((n, prompts if n == "PROMPTS.jsonl" else a) for n, a in case.file_refs),
    )
    adopted = replacement.submit(
        model=inputs.model,
        cases=case,
        checkpoint=None,
        catalog=catalog(),
        spec_sha256="a" * 64,
        evaluation_id="fixture-paired",
        runtime_session_id=SESSION,
    )
    assert adopted.extra.get("adopted_existing") is True and adopted.id == handle.id
    assert {pin_key(a) for a in pins(adopted)} <= set(adopted.extra["verified_pins"])
    assert set(adopted.extra["verified_pins"]) == set(handle.extra["verified_pins"])


def test_the_record_travels_with_the_handle_not_with_the_process(state, raw_calls):
    backend, native, s3, handle, *_ = installed_generation(state)
    other = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    submitted = len(raw_calls)
    assert other.artifacts(handle)["success"] is True
    assert not {(a.uri, a.version_id) for a, *_ in raw_calls[submitted:]} & pin_ids(handle)
    assert backend is not other


def test_a_handle_without_a_record_is_verified_in_full_under_a_deadline_scaled_to_its_bytes(
    state, raw_calls, monkeypatch
):
    hash_seconds = symbol(qwen_sft, "_hash_seconds")
    backend, _native, _s3, handle, *_ = installed_generation(state)
    handle.extra.pop("verified_pins", None)
    now = 1000.0
    monkeypatch.setattr("amlas.backends.adapters.coding_generation.time.monotonic", lambda: now)
    submitted = len(raw_calls)
    assert backend.artifacts(handle)["success"] is True
    checked = {(a.uri, a.version_id): deadline for a, deadline, _c in raw_calls[submitted:]}
    assert pin_ids(handle) <= set(checked), "unrecorded pins must be verified in full"
    total = sum(a.byte_count for a in pins(handle))
    assert min(checked[i] for i in pin_ids(handle)) >= now + 600 + hash_seconds(total)


def test_the_hash_deadline_is_three_times_slower_than_the_slowest_measured_stream():
    hash_seconds = symbol(qwen_sft, "_hash_seconds")
    pack = 19_319_226_111
    # slowest measured exact-version stream: 39.9 MiB/s (original pack of 1,025 parts); staged copies: 72 to 95 MiB/s
    assert hash_seconds(pack) >= 3 * pack / (39.9 * MIB) - 1
    assert hash_seconds(0) == 0 and hash_seconds(1) == 1


@pytest.mark.parametrize("field", ["version", "size", "sha"])
def test_a_changed_pin_still_fails_closed(state, field):
    backend, _native, _s3, handle, *_ = installed_generation(state)
    name, artifact = next(iter(handle.extra["staged_refs"].items()))
    changed = {
        "version": dataclasses.replace(artifact, version_id="another-version"),
        "size": dataclasses.replace(artifact, byte_count=artifact.byte_count + 1),
        "sha": dataclasses.replace(artifact, sha256="e" * 64),
    }[field]
    handle.extra["staged_refs"][name] = changed
    result = backend.artifacts(handle)
    assert result["success"] is False and result["status"] == "output_verification_failed"


def test_a_deleted_pinned_version_fails_closed(state):
    backend, _native, s3, handle, *_ = installed_generation(state)
    artifact = next(iter(handle.extra["original_base_refs"].values()))
    s3.objects.pop(artifact.uri.split("/", 3)[3])
    result = backend.artifacts(handle)
    assert result["success"] is False and result["status"] == "output_verification_failed"


def test_a_staged_key_whose_current_version_moved_on_fails_closed(state):
    backend, _native, s3, handle, *_ = installed_generation(state)
    artifact = next(a for n, a in handle.extra["staged_refs"].items() if n.startswith("base_model/"))
    key = artifact.uri.split("/", 3)[3]
    request = dict(Bucket=BUCKET, Key=key, ExpectedBucketOwner=catalog().account_id)
    pinned_head = s3.head_object(**request, VersionId=artifact.version_id)
    head = s3.head_object
    s3.objects[key] = (dataclasses.replace(artifact, version_id="overwritten-later"), None)

    def versioned_head(**kw):
        # the pinned version is still retained; the key now resolves to a newer one
        return pinned_head if kw.get("VersionId") == artifact.version_id else head(**kw)

    s3.head_object = versioned_head
    result = backend.artifacts(handle)
    assert result["success"] is False and result["status"] == "output_verification_failed"


def test_changed_metadata_on_the_same_version_falls_back_to_a_full_read_that_can_still_pass(state, raw_calls):
    backend, _native, s3, handle, *_ = installed_generation(state)
    artifact = next(iter(handle.extra["staged_refs"].values()))
    original_head = s3.head_object

    def head(**kw):
        value = original_head(**kw)
        if kw["Key"] == artifact.uri.split("/", 3)[3]:
            value = {**value, "ETag": '"changed-etag"'}
        return value

    s3.head_object = head
    submitted = len(raw_calls)
    assert backend.artifacts(handle)["success"] is True
    assert (artifact.uri, artifact.version_id) in {(a.uri, a.version_id) for a, *_ in raw_calls[submitted:]}


def test_changed_metadata_with_different_bytes_fails_closed(state):
    backend, _native, s3, handle, *_ = installed_generation(state)
    artifact = next(iter(handle.extra["staged_refs"].values()))
    original_head = s3.head_object
    flipped = hashlib.sha256(b"different bytes").digest()

    def head(**kw):
        value = original_head(**kw)
        if kw["Key"] == artifact.uri.split("/", 3)[3]:
            import base64

            value = {**value, "ETag": '"changed-etag"', "ChecksumSHA256": base64.b64encode(flipped).decode()}
        return value

    s3.head_object = head
    result = backend.artifacts(handle)
    assert result["success"] is False and result["status"] == "output_verification_failed"


class StreamingS3(S3):
    """Head shape of real S3: no SHA-256 full-object checksum, so every verification streams the bytes."""

    def __init__(self):
        super().__init__()
        self.streamed = collections.Counter()

    def head_object(self, **kw):
        value = super().head_object(**kw)
        value.pop("ChecksumSHA256")
        value.update(ChecksumCRC64NVME="x", ETag='"e-' + value["VersionId"] + '"')
        return value

    def get_object(self, **kw):
        value = super().get_object(**kw)
        key = kw["Key"]
        body = value["Body"]
        read = body.read

        def counted(amount=-1):
            data = read(amount)
            self.streamed[key] += len(data)
            return data

        body.read = counted
        return value


def small(name, raw):
    return ref(name, raw=raw)


def test_raw_records_a_pin_only_after_a_complete_verification():
    pin_key = symbol(qwen_sft, "_pin_key")
    s3 = StreamingS3()
    good = b"model bytes" * 1000
    artifact, wrong = small("a.bin", good), dataclasses.replace(small("b.bin", good), sha256="0" * 64)
    s3.install(artifact, good)
    s3.install(wrong, good)
    backend = SageMakerQwenSftBackend(sagemaker_client=Native(), s3_client=s3)
    seen = {}
    backend._raw(artifact, catalog(), 10**12, seen=seen)
    assert set(seen) == {pin_key(artifact)} and s3.streamed[artifact.uri.split("/", 3)[3]] == len(good)
    with pytest.raises(ValueError):
        backend._raw(wrong, catalog(), 10**12, seen=seen)
    assert set(seen) == {pin_key(artifact)}


def test_reverify_does_not_stream_a_recorded_pin_and_streams_an_unrecorded_one():
    pin_key = symbol(qwen_sft, "_pin_key")
    s3 = StreamingS3()
    raw = b"model bytes" * 1000
    artifact = small("a.bin", raw)
    s3.install(artifact, raw)
    backend = SageMakerQwenSftBackend(sagemaker_client=Native(), s3_client=s3)
    seen = {}
    backend._raw(artifact, catalog(), 10**12, seen=seen)
    key = artifact.uri.split("/", 3)[3]
    s3.streamed.clear()
    backend._reverify(artifact, catalog(), 10**12, seen)
    assert s3.streamed[key] == 0
    backend._reverify(artifact, catalog(), 10**12, {})
    assert s3.streamed[key] == len(raw)
    assert pin_key(artifact) in seen


def test_raw_refuses_bytes_whose_etag_differs_from_the_head_that_was_recorded():
    s3 = StreamingS3()
    raw = b"model bytes" * 1000
    artifact = small("a.bin", raw)
    s3.install(artifact, raw)
    get = s3.get_object
    s3.get_object = lambda **kw: {**get(**kw), "ETag": '"another-object"'}
    backend = SageMakerQwenSftBackend(sagemaker_client=Native(), s3_client=s3)
    seen = {}
    with pytest.raises(ValueError, match="ETag differs"):
        backend._raw(artifact, catalog(), 10**12, seen=seen)
    assert seen == {}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def throttled_streams(monkeypatch, clock):
    """Hashing a pin costs the measured time: 42.5 MiB/s for the producer's 5 MiB-part objects, 94 MiB/s for staged copies."""
    original = SageMakerQwenSftBackend._raw
    spent = []

    def raw(self, artifact, catalog_, deadline, *args, **kwargs):
        if artifact.byte_count > MIB:
            rate = 94 * MIB if "/generation/" in artifact.uri else 42.5 * MIB
            clock.now += artifact.byte_count / rate
            spent.append(artifact.byte_count / rate)
            qwen_sft._deadline(deadline)
        return original(self, artifact, catalog_, deadline, *args, **kwargs)

    monkeypatch.setattr(time, "monotonic", clock)
    monkeypatch.setattr(SageMakerQwenSftBackend, "_raw", raw)
    return spent


def test_the_measured_shape_of_the_model_pack_fits_one_observation_and_the_pair_chain(state, monkeypatch):
    clock = Clock()
    spent = throttled_streams(monkeypatch, clock)
    backend, _native, _s3, handle, *_ = installed_generation(state)
    assert sum(spent) > 600, "the scaled-down stand-in must take longer than the old fixed deadline"
    before = clock.now
    assert backend.artifacts(handle)["success"] is True
    assert backend.artifacts(handle)["success"] is True
    assert clock.now - before < 60


def test_a_pack_without_a_record_takes_the_measured_time_and_still_finishes(state, monkeypatch):
    clock = Clock()
    throttled_streams(monkeypatch, clock)
    backend, _native, _s3, handle, *_ = installed_generation(state)
    handle.extra.pop("verified_pins")
    before = clock.now
    result = backend.artifacts(handle)
    assert result["success"] is True, result
    assert 600 < clock.now - before < 600 + qwen_sft._hash_seconds(sum(a.byte_count for a in pins(handle)))


def test_the_tuned_arm_records_its_adapter_pins_too(state):
    from amlas.control_plane.sft_state import VerifiedSftCheckpointSource

    pin_key = symbol(qwen_sft, "_pin_key")
    _store, _adapter, _reads, s3, inputs, case = state
    backend = SageMakerCodingGenerationBackend(sagemaker_client=Native(), s3_client=s3)
    s3.install(dict(case.file_refs)["PROMPTS.jsonl"], b"{}\n")
    cp, cfg = ref("CHECKPOINT.json", raw=b"{}"), ref("adapter_config.json", raw=b"{}")
    tensor = ref("adapter_model.safetensors", raw=b"abc", fmt="safetensors")
    for artifact, raw in ((cp, b"{}"), (cfg, b"{}"), (tensor, b"abc")):
        s3.install(artifact, raw)
    checkpoint = VerifiedSftCheckpointSource(
        SESSION,
        "arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":training-job/owned-original",
        "full_v1",
        "d" * 40,
        cp,
        (("CHECKPOINT.json", cp), ("adapter_config.json", cfg), ("adapter_model.safetensors", tensor)),
    )
    handle = backend.submit(
        model=inputs.model,
        cases=case,
        checkpoint=checkpoint,
        catalog=catalog(),
        spec_sha256="a" * 64,
        evaluation_id="paired",
        runtime_session_id=SESSION,
    )
    adapter_pins = {pin_key(a) for n, a in handle.extra["staged_refs"].items() if n.startswith("adapter/")}
    assert len(adapter_pins) == 3
    assert {pin_key(a) for a in pins(handle)} <= set(handle.extra["verified_pins"])
