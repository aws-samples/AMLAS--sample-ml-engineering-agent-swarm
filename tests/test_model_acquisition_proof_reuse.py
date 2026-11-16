"""Private whole-acquisition byte proofs; immutable-version service-shape controls."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import model_acquisition_fixtures as f
import pytest


@pytest.fixture(autouse=True)
def namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")


class VersionedS3(f.S3):
    """Enforce actual S3 input shapes, version availability and owner requests."""

    def __init__(self) -> None:
        super().__init__()
        self.heads: list[dict[str, Any]] = []
        self.deleted_versions: set[tuple[str, str]] = set()
        self.fail_payload: str | None = None
        self.transfer_bytes = 0
        self._calls_lock = threading.Lock()
        from botocore.session import Session

        self._service = Session().get_service_model("s3")

    def check(self, operation: str, kwargs: dict[str, Any]) -> None:
        from botocore.validate import validate_parameters

        validate_parameters(kwargs, self._service.operation_model(operation).input_shape)
        if kwargs.get("ExpectedBucketOwner") != f.ACCOUNT or kwargs.get("Bucket") != f.BUCKET:
            raise PermissionError("closed owner or bucket denial")
        if (kwargs.get("Key"), kwargs.get("VersionId")) in self.deleted_versions:
            raise FileNotFoundError("closed exact-version deletion")

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.check("ListObjectsV2", kwargs)
        return super().list_objects_v2(**kwargs)

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.check("HeadObject", kwargs)
        with self._calls_lock:
            self.heads.append(deepcopy(kwargs))
        return super().head_object(**kwargs)

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.check("GetObject", kwargs)
        if kwargs["Key"] == self.fail_payload:
            raise TimeoutError("closed incomplete transfer")
        result = super().get_object(**kwargs)
        if not kwargs["Key"].endswith(("COMMITTED.json", "LINEAGE.json", "MODEL_ACQUIRED.json")):
            with self._calls_lock:
                self.transfer_bytes += result["ContentLength"]
        return result


def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Any, Any, Any]:
    module = f.require_module(
        "amlas.backends.adapters.model_weight_acquisition", "ProcessingModelWeightAcquisitionBackend"
    )
    f.patch_expected(
        monkeypatch, [module, __import__("amlas.data_eng.model_acquisition_schema", fromlist=["x"])]
    )
    sm, s3 = f.Processing(), VersionedS3()
    adapter = f.configured_backend(module, sm, s3)
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    f.publish_fixture(monkeypatch, tmp_path, adapter, handle)
    sm.state = "Completed"
    return module, adapter, handle, s3


def payload_reads(s3: VersionedS3) -> list[dict[str, Any]]:
    return [
        call
        for call in s3.reads
        if not call["Key"].endswith(("COMMITTED.json", "LINEAGE.json", "MODEL_ACQUIRED.json"))
    ]


def test_repeated_success_reuses_only_completed_private_raw_byte_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    first = adapter.artifacts(handle)
    assert first["success"] is True
    assert len(payload_reads(s3)) == 9
    original_extra = deepcopy(handle.extra)
    payload_byte_count = sum(ref["byte_count"] for ref in first["file_refs"])
    assert s3.transfer_bytes == payload_byte_count
    before = s3.transfer_bytes
    heads_before = len(s3.heads)
    second = adapter.artifacts(handle)
    assert second == first
    assert len(payload_reads(s3)) == 9, "repeated observations retransferred immutable model payload bytes"
    assert s3.transfer_bytes == before
    new_heads = s3.heads[heads_before:]
    for ref in first["file_refs"]:
        key = ref["uri"].split(f"s3://{f.BUCKET}/", 1)[1]
        assert any(call["Key"] == key and call.get("VersionId") == ref["version_id"] for call in new_heads)
    assert all(call["ExpectedBucketOwner"] == f.ACCOUNT for call in new_heads)
    assert handle.extra == original_extra
    assert not set(first) & {"proof", "cache", "raw_verified"}
    assert first["trusted"] is False and first["trained"] is False


@pytest.mark.parametrize(
    "case",
    [
        "version",
        "null",
        "length",
        "deleted",
        "native_role",
        "session",
        "source",
        "image",
        "owner",
        "region",
        "catalog_digest",
        "manifest_version",
        "manifest_recomputed",
    ],
)
def test_success_proof_refuses_and_invalidates_any_changed_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    first = adapter.artifacts(handle)
    assert first["success"] is True
    original_objects = deepcopy(s3.objects)
    original_extra = deepcopy(handle.extra)
    ref = first["file_refs"][0]
    key = ref["uri"].split(f"s3://{f.BUCKET}/", 1)[1]
    version, raw = s3.objects[key]
    if case == "version":
        s3.objects[key] = ("replacement-version", raw)
    elif case == "null":
        s3.objects[key] = ("null", raw)
    elif case == "length":
        s3.objects[key] = (version, raw + b"!")
    elif case == "deleted":
        s3.deleted_versions.add((key, version))
    elif case == "native_role":
        adapter._sm.overrides["RoleArn"] = f.ROLE + "-foreign"
    elif case in {"session", "source", "image"}:
        field = {"session": "runtime_session_id", "source": "source_commit", "image": "image_uri"}[case]
        handle.extra["bindings"][field] += "-foreign"
    elif case in {"owner", "region", "catalog_digest"}:
        field = {"owner": "account_id", "region": "region", "catalog_digest": "expected_manifest_sha256"}[
            case
        ]
        handle.extra[field] = (
            "000000000000" if case == "owner" else "us-east-1" if case == "region" else "0" * 64
        )
    elif case == "manifest_version":
        metadata_key = handle.extra["output_prefix"] + "COMMITTED.json"
        _, metadata = s3.objects[metadata_key]
        s3.objects[metadata_key] = ("replacement-manifest-version", metadata)
    else:
        receipt_key = handle.extra["output_prefix"] + "MODEL_ACQUIRED.json"
        receipt_version, receipt_raw = s3.objects[receipt_key]
        # A different valid encoding plus a coherent outer hash cannot reuse original manifest proof.
        receipt = json.dumps(json.loads(receipt_raw), indent=2).encode()
        s3.objects[receipt_key] = (receipt_version, receipt)
        commit_key = handle.extra["output_prefix"] + "COMMITTED.json"
        commit_version, commit_raw = s3.objects[commit_key]
        commit = json.loads(commit_raw)
        commit["receipt"].update(sha256=f.digest(receipt), byte_count=len(receipt))
        s3.objects[commit_key] = (commit_version, json.dumps(commit).encode())
    assert adapter.artifacts(handle)["success"] is False
    s3.objects = original_objects
    s3.deleted_versions.clear()
    adapter._sm.overrides.clear()
    handle.extra.clear()
    handle.extra.update(original_extra)
    before = len(payload_reads(s3))
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == before + 9, "failed observation left a reusable byte proof"


def test_caller_result_json_and_equal_foreign_handle_do_not_grant_proof_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    first = adapter.artifacts(handle)
    assert first["success"] is True
    first["file_refs"][0]["sha256"] = "0" * 64
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 9
    copy = deepcopy(handle)
    assert copy is not handle
    assert adapter.artifacts(copy)["success"] is True
    assert len(payload_reads(s3)) == 18
    other = f.configured_backend(module, adapter._sm, s3)
    assert other.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 27


@pytest.mark.parametrize("failure", ["payload", "manifest", "native", "running", "ambiguous"])
def test_incomplete_or_failed_acquisition_does_not_populate_any_reusable_partial_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    original = deepcopy(s3.objects)
    keys = [key for key in s3.objects if key.endswith(".safetensors")]
    if failure == "payload":
        s3.fail_payload = keys[-1]
    elif failure == "manifest":
        s3.objects.pop(handle.extra["output_prefix"] + "LINEAGE.json")
    elif failure == "native":
        adapter._sm.read_failure = True
    elif failure == "running":
        adapter._sm.state = "InProgress"
    else:
        handle.extra["accepted"] = False
        handle.extra["submission_ambiguous"] = True
        adapter._sm.read_failure = True
    assert adapter.artifacts(handle)["success"] is False
    before = len(payload_reads(s3))
    s3.objects = original
    s3.fail_payload = None
    adapter._sm.read_failure = False
    adapter._sm.state = "Completed"
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == before + 9


def test_full_object_checksum_success_does_not_create_a_raw_byte_witness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    s3.full_checksums = True
    assert adapter.artifacts(handle)["success"] is True
    assert payload_reads(s3) == []
    s3.full_checksums = False
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 9
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 9


def test_concurrent_observations_publish_no_partial_proof_and_transfer_one_complete_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    barrier = threading.Barrier(4)

    def observe() -> dict[str, Any]:
        barrier.wait(timeout=10)
        return adapter.artifacts(handle)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: observe(), range(4)))
    assert all(result["success"] is True for result in results)
    assert all(result == results[0] for result in results)
    assert len(payload_reads(s3)) == 9, "concurrent observations repeated model transfers"


@pytest.mark.parametrize("state", ["read_error", "Failed", "InProgress"])
def test_separate_native_status_failure_invalidates_prior_byte_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str
) -> None:
    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    assert adapter.artifacts(handle)["success"] is True
    if state == "read_error":
        adapter._sm.read_failure = True
    else:
        adapter._sm.state = state
    assert adapter.status(handle).state.name != "SUCCEEDED"
    adapter._sm.read_failure = False
    adapter._sm.state = "Completed"
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 18


@pytest.mark.parametrize("case", ["response_version", "response_size", "deadline", "checksum"])
def test_cached_exact_version_head_refuses_bad_response_without_payload_get(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    module, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    assert adapter.artifacts(handle)["success"] is True
    original = s3.head_object
    clock = [0.0]
    if case == "deadline":
        monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def head(**kwargs: Any) -> dict[str, Any]:
        result = original(**kwargs)
        if "VersionId" in kwargs:
            if case == "response_version":
                result["VersionId"] = "foreign-version"
            elif case == "response_size":
                result["ContentLength"] += 1
            elif case == "checksum":
                import base64

                result.update(ChecksumType="FULL_OBJECT", ChecksumSHA256=base64.b64encode(bytes(32)).decode())
            else:
                clock[0] = 3601.0
        return result

    monkeypatch.setattr(s3, "head_object", head)
    assert adapter.artifacts(handle)["success"] is False
    assert len(payload_reads(s3)) == 9
    monkeypatch.setattr(s3, "head_object", original)
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 18


def test_unrelated_job_status_is_not_blocked_by_another_model_raw_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dataclasses import replace

    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    name = handle.id + "-other"
    other = replace(handle, id=name, arn=adapter._sm.arn(name), extra=deepcopy(handle.extra))
    other.extra["request"]["ProcessingJobName"] = name
    original_description = adapter._sm.describe_processing_job

    def description(**kwargs: Any) -> dict[str, Any]:
        if kwargs["ProcessingJobName"] == name:
            return {
                **deepcopy(other.extra["request"]),
                "ProcessingJobArn": other.arn,
                "ProcessingJobStatus": "Completed",
            }
        return original_description(**kwargs)

    monkeypatch.setattr(adapter._sm, "describe_processing_job", description)
    started, release = threading.Event(), threading.Event()
    original_get = s3.get_object
    first_payload = next(key for key in s3.objects if key.endswith(".safetensors"))

    class BlockedBody:
        def __init__(self, body: Any) -> None:
            self.body = body

        def read(self, size: int) -> bytes:
            started.set()
            assert release.wait(timeout=10), "test failed to release blocked payload"
            return self.body.read(size)

        def close(self) -> None:
            self.body.close()

    def get(**kwargs: Any) -> dict[str, Any]:
        result = original_get(**kwargs)
        if kwargs["Key"] == first_payload:
            result["Body"] = BlockedBody(result["Body"])
        return result

    monkeypatch.setattr(s3, "get_object", get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(adapter.artifacts, handle)
        try:
            assert started.wait(timeout=5)
            status = pool.submit(adapter.status, other)
            assert status.result(timeout=0.5).state.name == "SUCCEEDED"
        finally:
            release.set()
        assert first.result(timeout=10)["success"] is True


def test_mixed_service_checksum_and_raw_inventory_cannot_create_partial_witness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import base64
    import hashlib

    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    original = s3.head_object
    key = next(key for key in s3.objects if key.endswith(".safetensors"))

    def head(**kwargs: Any) -> dict[str, Any]:
        result = original(**kwargs)
        if kwargs["Key"] == key:
            result.update(
                ChecksumType="FULL_OBJECT",
                ChecksumSHA256=base64.b64encode(hashlib.sha256(s3.objects[key][1]).digest()).decode(),
            )
        return result

    monkeypatch.setattr(s3, "head_object", head)
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 8
    monkeypatch.setattr(s3, "head_object", original)
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 17
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 17


@pytest.mark.parametrize("change", ["new_latest", "delete_marker"])
def test_latest_drift_refuses_even_when_original_exact_version_remains_readable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str
) -> None:
    import io

    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    first = adapter.artifacts(handle)
    assert first["success"] is True
    ref = first["file_refs"][0]
    key = ref["uri"].split(f"s3://{f.BUCKET}/", 1)[1]
    original_version, original_raw = s3.objects[key]
    original_head, original_get = s3.head_object, s3.get_object

    def head(**kwargs: Any) -> dict[str, Any]:
        s3.check("HeadObject", kwargs)
        if kwargs["Key"] == key:
            if kwargs.get("VersionId") == original_version:
                return {"VersionId": original_version, "ContentLength": len(original_raw)}
            if change == "delete_marker":
                raise FileNotFoundError("closed current delete marker; original retained")
        return original_head(**kwargs)

    def get(**kwargs: Any) -> dict[str, Any]:
        s3.check("GetObject", kwargs)
        if kwargs["Key"] == key and kwargs.get("VersionId") == original_version:
            return {
                "VersionId": original_version,
                "ContentLength": len(original_raw),
                "Body": io.BytesIO(original_raw),
            }
        return original_get(**kwargs)

    s3.objects[key] = ("new-latest-version", original_raw)
    monkeypatch.setattr(s3, "head_object", head)
    monkeypatch.setattr(s3, "get_object", get)
    owned_exact = {
        "Bucket": f.BUCKET,
        "Key": key,
        "ExpectedBucketOwner": f.ACCOUNT,
        "VersionId": original_version,
    }
    assert s3.head_object(**owned_exact)["VersionId"] == original_version
    retained = s3.get_object(**owned_exact)
    assert retained["Body"].read() == original_raw
    retained["Body"].close()
    assert adapter.artifacts(handle)["success"] is False
    assert len(payload_reads(s3)) == 9
    monkeypatch.setattr(s3, "head_object", original_head)
    monkeypatch.setattr(s3, "get_object", original_get)
    s3.objects[key] = (original_version, original_raw)
    assert adapter.artifacts(handle)["success"] is True
    assert len(payload_reads(s3)) == 18


def test_concurrent_first_failure_cannot_publish_partial_inventory_to_waiting_observer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, adapter, handle, s3 = setup(monkeypatch, tmp_path)
    started, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    key = [key for key in s3.objects if key.endswith(".safetensors")][-1]
    original = s3.get_object
    failed = [False]

    def get(**kwargs: Any) -> dict[str, Any]:
        if kwargs["Key"] == key and not failed[0]:
            started.set()
            assert release.wait(timeout=10)
            failed[0] = True
            raise TimeoutError("closed failure after partial raw inventory")
        return original(**kwargs)

    def second() -> dict[str, Any]:
        second_entered.set()
        return adapter.artifacts(handle)

    monkeypatch.setattr(s3, "get_object", get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(adapter.artifacts, handle)
        try:
            assert started.wait(timeout=5)
            count = len(payload_reads(s3))
            assert count > 0
            waiting = pool.submit(second)
            assert second_entered.wait(timeout=5)
        finally:
            release.set()
        assert first.result(timeout=10)["success"] is False
        assert waiting.result(timeout=10)["success"] is True
    assert len(payload_reads(s3)) == count + 9


def test_bounded_lru_evicts_original_handle_and_requires_new_raw_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, adapter, first, s3 = setup(monkeypatch, tmp_path)
    assert adapter.artifacts(first)["success"] is True
    handles = [first]

    def description(**kwargs: Any) -> dict[str, Any]:
        request = next(
            request
            for request in adapter._sm.requests
            if request["ProcessingJobName"] == kwargs["ProcessingJobName"]
        )
        return {
            **deepcopy(request),
            "ProcessingJobArn": adapter._sm.arn(request["ProcessingJobName"]),
            "ProcessingJobStatus": "Completed",
        }

    monkeypatch.setattr(adapter._sm, "describe_processing_job", description)
    for index in range(1, 64):
        handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
        folder = tmp_path / str(index)
        folder.mkdir()
        f.publish_fixture(monkeypatch, folder, adapter, handle)
        assert adapter.artifacts(handle)["success"] is True
        handles.append(handle)
    count = len(payload_reads(s3))
    assert count == 64 * 9
    assert adapter.artifacts(first)["success"] is True
    assert len(payload_reads(s3)) == count
    final = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    folder = tmp_path / "64"
    folder.mkdir()
    f.publish_fixture(monkeypatch, folder, adapter, final)
    assert adapter.artifacts(final)["success"] is True
    assert adapter.artifacts(first)["success"] is True
    assert len(payload_reads(s3)) == count + 9
    assert adapter.artifacts(handles[1])["success"] is True
    assert len(payload_reads(s3)) == count + 18
    # Pure witness construction never changed the public adapter registration.
    assert isinstance(adapter, module.ProcessingModelWeightAcquisitionBackend)
