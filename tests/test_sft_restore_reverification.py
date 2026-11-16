"""The original SFT job is re-verified against its pack once per process.

`restore_original` runs for every status poll of the pending SFT stage (`_pending` builds a new adapter per poll), for every
15 s iteration of the after-tool wait loop (`SftStore.observe`), and for every cold restore. At production size each call
re-hashed the adopted pack (17.99 GiB) and the job's staged `base_model` copy (another 17.99 GiB). These tests drive the
real `restore_original` on the durable-submission fixture, with the pack's real shard sizes served by lazy bodies, and
the model adapter's byte proof with a second adapter over a re-decoded handle.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io

import model_acquisition_fixtures as f
import pytest
import test_pack_reverification as shared
from test_model_acquisition_proof_reuse import VersionedS3, payload_reads
from test_pack_reverification import PACK_BYTES, Virtual
from test_pack_reverification import LazyBody as _Lazy
from test_qwen_sft_backend import S3 as BaseS3  # noqa: N811
from test_qwen_sft_durable_submission import durable, submit

from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest

clock = shared.clock  # fixtures of the shared module
counting_sha = shared.counting_sha

MODEL_BYTES = sum(item["byte_count"] for item in expected_model_weight_manifest("qwen3_5_9b", "1")["files"])


class ShapedS3(BaseS3):
    """The durable fixture's store with S3-shaped HEAD answers (ETag, CRC64NVME, no SHA-256) and lazy large bodies."""

    def __init__(self, clock, objects):
        super().__init__()
        self.clock, self.objects = clock, objects
        self.big_bytes, self.gets, self.heads = 0, [], []
        import threading

        self.lock = threading.Lock()
        self.etag_override = {}

    def _etag(self, key, r):
        return self.etag_override.get(key, '"' + r.sha256[:32] + "-" + str(r.byte_count) + '"')

    def head_object(self, **kw):
        r, _raw = self.objects[kw["Key"]]
        assert kw.get("VersionId", r.version_id) == r.version_id
        self.heads.append(kw["Key"])
        etag = self._etag(kw["Key"], r)
        return {
            "VersionId": r.version_id,
            "ContentLength": r.byte_count,
            "ETag": etag,
            "ChecksumType": "FULL_OBJECT",
            "ChecksumCRC64NVME": base64.b64encode(hashlib.sha256(etag.encode()).digest()[:8]).decode(),
        }

    def get_object(self, **kw):
        r, raw = self.objects[kw["Key"]]
        assert kw["VersionId"] == r.version_id
        self.gets.append(kw["Key"])
        common = {"VersionId": r.version_id, "ContentLength": r.byte_count, "ETag": self._etag(kw["Key"], r)}
        if raw is None:
            return {**common, "Body": _Lazy(self, Virtual(r.byte_count, r.sha256))}
        return {**common, "Body": io.BytesIO(raw)}


def original():
    parts = durable()
    handle = submit(parts)
    _adapter, native, s3, inputs, campaign, binding, _table = parts
    return native, s3, inputs, campaign.load_original(binding, handle.id), handle


def test_each_new_adapter_restoring_the_original_reuses_the_hash_of_the_pack(clock):
    native, s3, inputs, reservation, _handle = original()
    shaped = ShapedS3(clock, s3.objects)
    reads = []
    for _ in range(5):
        # `_pending` builds a new adapter for every status poll.
        adapter = SageMakerQwenSftBackend(sagemaker_client=native, s3_client=shaped)
        before = shaped.big_bytes
        restored = adapter.restore_original(reservation, fresh_inputs=inputs)
        assert restored.extra["restoration_status"] == "verified"
        reads.append(shaped.big_bytes - before)
    assert MODEL_BYTES == PACK_BYTES + sum(
        i["byte_count"]
        for i in expected_model_weight_manifest("qwen3_5_9b", "1")["files"]
        if i["byte_count"] < 10**8
    )
    assert reads[0] == 2 * MODEL_BYTES, (
        "first call: the original pack and the staged base_model copy, once each"
    )
    assert reads[1:] == [0, 0, 0, 0], "later restores re-hashed the pack: " + repr(reads)


def test_later_restores_check_the_pinned_objects_by_exact_version_head(clock):
    native, s3, inputs, reservation, _handle = original()
    shaped = ShapedS3(clock, s3.objects)
    adapter = SageMakerQwenSftBackend(sagemaker_client=native, s3_client=shaped)
    adapter.restore_original(reservation, fresh_inputs=inputs)
    shaped.heads.clear()
    adapter.restore_original(reservation, fresh_inputs=inputs)
    model_keys = {r.uri.split("/", 3)[3] for _name, r in inputs.model.file_refs}
    staged = {r.uri.split("/", 3)[3] for name, r in reservation.submission_snapshot.staged_refs}
    assert model_keys <= set(shaped.heads) and staged & set(shaped.heads)


def test_a_replaced_original_shard_is_read_in_full_and_refused_when_its_bytes_differ(clock):
    native, s3, inputs, reservation, _handle = original()
    shaped = ShapedS3(clock, s3.objects)
    adapter = SageMakerQwenSftBackend(sagemaker_client=native, s3_client=shaped)
    adapter.restore_original(reservation, fresh_inputs=inputs)
    path, ref = next((p, r) for p, r in inputs.model.file_refs if r.byte_count > 10**9)
    key = ref.uri.split("/", 3)[3]
    shaped.etag_override[key] = '"replaced"'
    before = shaped.big_bytes
    adapter.restore_original(reservation, fresh_inputs=inputs)
    assert shaped.big_bytes - before == ref.byte_count, path
    r, raw = shaped.objects[key]
    shaped.objects[key] = (r.__class__(r.uri, r.version_id, r.sha256, r.byte_count - 1, r.format), raw)
    with pytest.raises(ValueError, match="size differs"):
        adapter.restore_original(reservation, fresh_inputs=inputs)


class ShapedModelS3(VersionedS3):
    def head_object(self, **kwargs):
        result = super().head_object(**kwargs)
        _version, raw = self.objects[kwargs["Key"]]
        etag = '"' + hashlib.sha256(raw).hexdigest()[:32] + '"'
        etag = getattr(self, "etags", {}).get(kwargs["Key"], etag)
        return {**result, "ETag": etag, "ChecksumType": "FULL_OBJECT", "ChecksumCRC64NVME": "Y3JjNjQ="}

    def get_object(self, **kwargs):
        result = super().get_object(**kwargs)
        result["ETag"] = self.head_object(**kwargs)["ETag"]
        return result


class Forwarding:
    """A different client object over the same store, as a second boto3 session would be."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


def model_case(monkeypatch, tmp_path):
    module = f.require_module(
        "amlas.backends.adapters.model_weight_acquisition", "ProcessingModelWeightAcquisitionBackend"
    )
    f.patch_expected(
        monkeypatch, [module, __import__("amlas.data_eng.model_acquisition_schema", fromlist=["x"])]
    )
    monkeypatch.setattr(
        __import__("amlas.config.externalized", fromlist=["x"]), "_default_ssm_getter", lambda *a, **k: ""
    )
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    sm, s3 = f.Processing(), ShapedModelS3()
    adapter = f.configured_backend(module, sm, s3)
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    f.publish_fixture(monkeypatch, tmp_path, adapter, handle)
    sm.state = "Completed"
    return module, adapter, handle, s3, sm


def test_a_second_model_adapter_over_a_redecoded_handle_reuses_the_byte_proof(monkeypatch, tmp_path):
    module, adapter, handle, s3, sm = model_case(monkeypatch, tmp_path)
    first = adapter.artifacts(handle)
    assert first["success"] is True and len(payload_reads(s3)) == 9
    other = f.configured_backend(module, sm, Forwarding(s3))
    second = other.artifacts(copy.deepcopy(handle))
    assert second == first
    assert len(payload_reads(s3)) == 9, "a second adapter re-read the nine payloads"
    heads = [h for h in s3.heads if h.get("VersionId")]
    assert heads and all(h["ExpectedBucketOwner"] == f.ACCOUNT for h in s3.heads)


def test_a_changed_payload_fingerprint_refuses_then_recovers_by_a_full_read(monkeypatch, tmp_path):
    module, adapter, handle, s3, sm = model_case(monkeypatch, tmp_path)
    first = adapter.artifacts(handle)
    other = f.configured_backend(module, sm, Forwarding(s3))
    twin = copy.deepcopy(handle)
    key = first["file_refs"][0]["uri"].split("/", 3)[3]
    s3.etags = {key: '"replaced"'}
    refused = other.artifacts(twin)
    assert refused["success"] is False and refused["status"] == "verification_failed"
    recovered = other.artifacts(twin)
    assert recovered["success"] is True and len(payload_reads(s3)) == 18


def test_a_discarded_proof_is_not_resurrected_by_a_pass_that_started_before_it(monkeypatch, tmp_path):
    module, adapter, handle, s3, sm = model_case(monkeypatch, tmp_path)
    adapter.artifacts(handle)
    sm.state = "InProgress"
    assert adapter.artifacts(handle)["success"] is False
    sm.state = "Completed"
    other = f.configured_backend(module, sm, Forwarding(s3))
    assert other.artifacts(copy.deepcopy(handle))["success"] is True
    assert len(payload_reads(s3)) == 18, "a status that was not SUCCEEDED must drop the shared proof as well"
