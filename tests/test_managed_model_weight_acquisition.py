"""Dedicated model Processing shapes and exact-version raw integrity, closed SDK."""

from __future__ import annotations

import json
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


def backend_module() -> Any:
    return f.require_module(
        "amlas.backends.adapters.model_weight_acquisition", "ProcessingModelWeightAcquisitionBackend"
    )


def test_fixed_model_request_requires_immutable_image_marker_and_service_upload() -> None:
    module = backend_module()
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(module, sm, s3)
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    request = sm.requests[0]
    assert request["AppSpecification"]["ContainerEntrypoint"] == [
        "/usr/local/bin/amlas-processing-init",
        "model-acquisition",
    ]
    assert request["ProcessingResources"]["ClusterConfig"] == {
        "InstanceType": "ml.m5.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 80,
    }
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 7200}
    assert request["NetworkConfig"] == {"EnableNetworkIsolation": False}
    assert not request.get("ProcessingInputs")
    assert request["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3UploadMode"] == "EndOfJob"
    assert set(request["Environment"]) <= {"AMLAS_AGENT_REMOTE_FETCH", "AMLAS_AGENT_HTTP_ALLOW_HOSTS"}
    arguments = request["AppSpecification"]["ContainerArguments"]
    assert arguments[:4] == ["--catalog-id", f.CATALOG, "--catalog-version", f.VERSION]
    assert not set(arguments) & {"--repo-id", "--filename", "--output-dir", "--role-arn", "--environment"}
    assert handle.backend == "model_processing" and handle.extra["accepted"] is True
    assert handle.extra["acquired"] is False


@pytest.mark.parametrize(
    "case",
    [
        "missing_marker",
        "stale_marker",
        "image_region",
        "role_account",
        "bad_source",
        "bucket_versioning",
        "catalog",
        "version",
        "session",
        "disabled",
    ],
)
def test_model_submission_binding_failure_denies_before_create(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    module = backend_module()
    sm, s3 = f.Processing(), f.S3()
    overrides = {}
    catalog = f.CATALOG
    version = f.VERSION
    session = f.SESSION
    if case == "missing_marker":
        overrides["admission_digest"] = None
    elif case == "stale_marker":
        overrides["admission_digest"] = "c" * 64
    elif case == "image_region":
        overrides["image_uri"] = f.IMAGE.replace(f.REGION, "us-east-1")
    elif case == "role_account":
        overrides["role_arn"] = f.ROLE.replace(f.ACCOUNT, "9876" * 3)
    elif case == "bad_source":
        overrides["source_commit"] = "main"
    elif case == "bucket_versioning":
        s3.versioning = "Suspended"
    elif case == "catalog":
        catalog = "qwen3_8_27b"
    elif case == "version":
        version = "latest"
    elif case == "session":
        session = "../foreign"
    else:
        monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "0")
    with pytest.raises(ValueError):
        f.configured_backend(module, sm, s3, **overrides).submit(
            catalog_id=catalog, catalog_version=version, runtime_session_id=session
        )
    assert sm.requests == []


@pytest.mark.parametrize(
    "field",
    [
        "RoleArn",
        "AppSpecification",
        "Environment",
        "NetworkConfig",
        "ProcessingResources",
        "StoppingCondition",
        "ProcessingOutputConfig",
    ],
)
def test_model_native_readback_refuses_any_execution_drift(field: str) -> None:
    module = backend_module()
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(module, sm, s3)
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    sm.overrides[field] = (
        "foreign"
        if field == "RoleArn"
        else {"EnableNetworkIsolation": True}
        if field == "NetworkConfig"
        else {}
    )
    assert adapter.status(handle).state.name == "UNKNOWN"
    assert len(sm.requests) == 1


def test_accepted_model_handle_survives_read_error_and_ambiguous_create() -> None:
    module = backend_module()
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(module, sm, s3)
    sm.read_failure = True
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    assert handle.arn and handle.extra["accepted"] is True
    assert adapter.status(handle).state.name == "UNKNOWN"
    assert len(sm.requests) == 1
    other = f.Processing()
    other.ambiguous_submit = True
    handle = f.configured_backend(module, other, s3).submit(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION
    )
    assert handle.extra["submission_ambiguous"] is True and handle.extra["accepted"] is False
    assert handle.id and len(other.requests) == 1


def test_completed_model_snapshot_verifies_all_exact_versions_without_parsing_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = backend_module()
    # Select the synthetic source catalog before the submitted private pin is frozen.
    f.patch_expected(
        monkeypatch, [module, __import__("amlas.data_eng.model_acquisition_schema", fromlist=["x"])]
    )
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(module, sm, s3)
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    f.publish_fixture(monkeypatch, tmp_path, adapter, handle)
    sm.state = "Completed"
    result = adapter.artifacts(handle)
    assert result["success"] is True and result["acquired"] is True
    assert result["trusted"] is False and result["prepared"] is False and result["trained"] is False
    assert result["guest_claim_attested"] is False
    assert len(result["file_refs"]) == 9 and set(result["manifest_refs"]) == {"receipt", "commit", "lineage"}
    assert all(ref["version_id"] for ref in result["file_refs"])
    assert all(call["VersionId"] for call in s3.reads)
    assert all(call["ExpectedBucketOwner"] == f.ACCOUNT for call in s3.reads)
    assert len(sm.requests) == 1


@pytest.mark.parametrize(
    "case",
    [
        "payload",
        "same_size",
        "extra",
        "null_version",
        "version_changed",
        "receipt_job",
        "receipt_proof",
        "lineage_hash",
        "commit_extra",
        "missing",
        "duplicate_json",
    ],
)
def test_native_completed_is_not_model_integrity_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    module = backend_module()
    # Select the synthetic source catalog before the submitted private pin is frozen.
    f.patch_expected(
        monkeypatch, [module, __import__("amlas.data_eng.model_acquisition_schema", fromlist=["x"])]
    )
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(module, sm, s3)
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    f.publish_fixture(monkeypatch, tmp_path, adapter, handle)
    sm.state = "Completed"
    prefix = handle.extra["output_prefix"]
    if case in {"payload", "same_size", "null_version"}:
        key = next(key for key in s3.objects if key.endswith(".safetensors"))
        version, raw = s3.objects[key]
        s3.objects[key] = (
            "null" if case == "null_version" else version,
            b"!" + raw[1:] if case == "same_size" else raw + b"!" if case == "payload" else raw,
        )
    elif case == "extra":
        s3.extra_key = True
    elif case == "version_changed":
        s3.version_mismatch = True
    elif case == "missing":
        s3.objects.pop(prefix + "LINEAGE.json")
    else:
        name = (
            "MODEL_ACQUIRED.json"
            if case.startswith("receipt")
            else "LINEAGE.json"
            if case == "lineage_hash"
            else "COMMITTED.json"
        )
        key = prefix + name
        version, raw = s3.objects[key]
        value = json.loads(raw)
        if case == "receipt_job":
            value["job_name"] = "foreign"
        elif case == "receipt_proof":
            value["evidence"]["privilege_proof"]["cap_eff"] = 1
        elif case == "lineage_hash":
            value["files"][0]["sha256"] = "c" * 64
        elif case == "commit_extra":
            value["trusted"] = True
        if case == "duplicate_json":
            # Duplicate only one existing key; every other valid field remains.
            first_key = next(iter(value))
            duplicate = (json.dumps(first_key) + ":" + json.dumps(value[first_key]) + ",").encode()
            raw = b"{" + duplicate + raw[1:]
            assert json.loads(raw) == value
        else:
            raw = json.dumps(value).encode()
        s3.objects[key] = (version, raw)
        if case == "receipt_proof":
            # Keep the outer commit valid so refusal isolates privilege proof.
            commit_key = prefix + "COMMITTED.json"
            commit_version, commit_raw = s3.objects[commit_key]
            commit = json.loads(commit_raw)
            commit["receipt"].update(sha256=f.digest(raw), byte_count=len(raw))
            s3.objects[commit_key] = (commit_version, json.dumps(commit).encode())
    result = adapter.artifacts(handle)
    assert result["success"] is False and result["acquired"] is False
    assert len(sm.requests) == 1


@pytest.mark.parametrize("mode", ["before_head", "after_head"])
@pytest.mark.parametrize("full", [False, True])
def test_payload_global_deadline_applies_to_checksum_and_stream_paths(
    monkeypatch: pytest.MonkeyPatch, mode: str, full: bool
) -> None:
    module = backend_module()
    expected, bodies = f.patch_expected(monkeypatch, [module])
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(module, sm, s3)
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    ref = expected["files"][0]
    s3.objects[handle.extra["output_prefix"] + ref["path"]] = ("pinned-version", bodies[ref["path"]])
    s3.full_checksums = full
    clock = [11.0 if mode == "before_head" else 0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    original = s3.head_object
    calls = []

    def head(**kwargs: Any) -> Any:
        calls.append(kwargs)
        value = original(**kwargs)
        clock[0] = 11.0
        return value

    monkeypatch.setattr(s3, "head_object", head)
    with pytest.raises(ValueError, match="deadline"):
        adapter._payload(handle, ref, 10.0)
    if mode == "before_head":
        assert calls == []
    assert s3.reads == []


@pytest.mark.parametrize(
    "kind,correct,success,raw_read",
    [
        ("FULL_OBJECT", True, True, False),
        ("FULL_OBJECT", False, False, False),
        ("COMPOSITE", True, True, True),
        ("COMPOSITE", False, True, True),
        (None, True, True, True),
    ],
)
def test_full_composite_missing_checksum_matrix_pins_version(
    monkeypatch: pytest.MonkeyPatch, kind: Any, correct: bool, success: bool, raw_read: bool
) -> None:
    import base64

    module = backend_module()
    expected, bodies = f.patch_expected(monkeypatch, [module])
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(module, sm, s3)
    handle = adapter.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    ref = expected["files"][0]
    s3.objects[handle.extra["output_prefix"] + ref["path"]] = ("pinned-version", bodies[ref["path"]])
    original = s3.head_object

    def head(**kwargs: Any) -> Any:
        value = original(**kwargs)
        if kind:
            value.update(
                ChecksumType=kind,
                ChecksumSHA256=base64.b64encode(
                    bytes.fromhex(ref["sha256"] if correct else "0" * 64)
                ).decode(),
            )
        return value

    monkeypatch.setattr(s3, "head_object", head)
    if success:
        assert adapter._payload(handle, ref, module.time.monotonic() + 10)["version_id"] == "pinned-version"
    else:
        with pytest.raises(ValueError, match="SHA"):
            adapter._payload(handle, ref, module.time.monotonic() + 10)
    assert bool(s3.reads) is raw_read
    assert all(value["VersionId"] == "pinned-version" for value in s3.reads)


def test_model_worker_recipe_pins_fixed_installer_and_removes_runtime_tools() -> None:
    """Fixed installer bytes/order and final absence remain image contracts."""
    recipe = (Path(__file__).parents[1] / "Dockerfile.model-acquisition").read_text()
    checksum = "931c303696af6fa3417112103b1cad26890e5a07eccb5b99783700e33f2b8aad"  # pragma: allowlist secret
    assert "ADD --checksum=sha256:" + checksum in recipe
    assert "pip-26.2-py3-none-any.whl" in recipe
    assert "--no-index --no-deps --only-binary=:all: --require-hashes" in recipe
    verified = recipe.index('assert m.version("pip")=="26.2"')
    removed_build_tools = recipe.index("pip uninstall --yes setuptools wheel")
    locked_install = recipe.index(
        "pip install --no-cache-dir --only-binary=:all: --require-hashes -r /app/requirements.model-acquisition.txt"
    )
    removed_pip = recipe.index("pip uninstall --yes pip")
    assert verified < removed_build_tools < locked_install < removed_pip
    assert "--mount=type=bind,from=installer-bootstrap" in recipe
    assert "all(u.find_spec(n) is None for n in (" in recipe
    assert "USER root" in recipe
    assert 'ENTRYPOINT ["/usr/local/bin/amlas-processing-init", "model-acquisition"]' in recipe
