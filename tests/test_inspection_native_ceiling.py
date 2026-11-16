"""Additive fixed native identity profile; closed offline observations only."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import test_inspection_worker as strict_worker
import test_managed_dataset_inspection as managed

PROFILE = "sagemaker_native_ceiling_v1"
CEILING = 0x200404FB


def schema() -> Any:
    return importlib.import_module("amlas.data_eng.inspection_schema")


def native_proof() -> dict[str, Any]:
    return {
        **strict_worker.admitted_privilege_proof(),
        "cap_bnd": CEILING,
        "keep_caps": 0,
        "seccomp": 2,
        "fs_uid": 10001,
        "fs_gid": 10001,
        "securebits": 0,
    }


def validator() -> Any:
    function = getattr(schema(), "validate_native_privilege_proof", None)
    assert callable(function), "missing-symbol: additive native privilege proof validator"
    return function


@pytest.fixture(autouse=True)
def fixture_context(monkeypatch: pytest.MonkeyPatch) -> Any:
    from amlas.config import externalized

    before = len(strict_worker._FIXTURE_FDS)
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", managed.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    for name in ("AMLAS_OFFLINE", "AMLAS_BACKENDS_OFFLINE", "DATA_INSPECTION_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    yield
    while len(strict_worker._FIXTURE_FDS) > before:
        os.close(strict_worker._FIXTURE_FDS.pop())


@pytest.mark.parametrize("bounding", [0, 1, CEILING])
@pytest.mark.parametrize("securebits", [0, 0x28, 0xC3])
def test_native_proof_admits_only_fixed_ceiling_with_permanent_drop(bounding: int, securebits: int) -> None:
    value = native_proof()
    value.update(cap_bnd=bounding, securebits=securebits)
    assert validator()(value) == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("res_uid", [10001, 10001, 0]),
        ("res_gid", [10001, 0, 10001]),
        ("res_uid", [True, 10001, 10001]),
        ("groups", [10001]),
        ("cap_eff", 1),
        ("cap_prm", 1),
        ("cap_inh", 1),
        ("cap_amb", 1),
        ("cap_bnd", CEILING | (1 << 2)),
        ("cap_bnd", CEILING | (1 << 8)),
        ("cap_bnd", -1),
        ("cap_bnd", True),
        ("no_new_privs", 0),
        ("no_new_privs", True),
        ("keep_caps", 1),
        ("keep_caps", False),
        ("seccomp", 0),
        ("seccomp", 1),
        ("seccomp", True),
        ("fs_uid", 0),
        ("fs_gid", 0),
        ("fs_uid", True),
        ("securebits", 1 << 2),
        ("securebits", 1 << 4),
        ("securebits", 0xFF),
        ("securebits", 0x100),
        ("securebits", -1),
        ("securebits", False),
    ],
)
def test_native_proof_drift_refuses(field: str, value: Any) -> None:
    proof = native_proof()
    proof[field] = value
    with pytest.raises(ValueError):
        validator()(proof)


def test_native_proof_requires_exact_fields_and_strict_zero_function_survives() -> None:
    function = validator()
    for proof in ({}, {**native_proof(), "ready": True}):
        with pytest.raises(ValueError):
            function(proof)
    changed = native_proof()
    changed.pop("fs_gid")
    with pytest.raises(ValueError):
        function(changed)
    old = strict_worker.admitted_privilege_proof()
    assert schema().validate_privilege_proof(old) == old
    old["cap_bnd"] = 1
    with pytest.raises(ValueError):
        schema().validate_privilege_proof(old)


def receipt(*, native: bool = True) -> dict[str, Any]:
    result = {
        "schema_version": 2 if native else 1,
        **strict_worker.binding(managed.PAYLOAD),
        "summary_sha256": "a" * 64,
        "guest_evidence": {
            "output_directory_proof": strict_worker.admitted_output_proof(),
            "credential_present": False,
            "provider_method": None,
            "credential_env_names": [],
            "metadata_reachable": False,
            "network_reachable": False,
            "privilege_proof": native_proof() if native else strict_worker.admitted_privilege_proof(),
        },
        "prepared": False,
        "trained": False,
    }
    result["sha256"] = result.pop("expected_sha256")
    if native:
        result["privilege_profile"] = PROFILE
    return result


def test_versioned_receipt_requires_parent_selected_expected_profile() -> None:
    decode = schema().decode_receipt
    assert "expected_profile" in inspect.signature(decode).parameters, (
        "missing-symbol: source-owned expected receipt profile"
    )
    raw = json.dumps(receipt()).encode()
    assert decode(raw, expected_profile=PROFILE)["guest_evidence"]["privilege_proof"]["cap_bnd"] == CEILING
    with pytest.raises(ValueError):
        decode(raw)
    old = json.dumps(receipt(native=False)).encode()
    assert decode(old)["schema_version"] == 1
    with pytest.raises(ValueError):
        decode(old, expected_profile=PROFILE)


@pytest.mark.parametrize("case", ["missing", "strict", "unknown", "version", "extra", "bnd", "sdk"])
def test_native_receipt_profile_or_proof_drift_refuses(case: str) -> None:
    decode = schema().decode_receipt
    assert "expected_profile" in inspect.signature(decode).parameters, (
        "missing-symbol: expected receipt profile"
    )
    value = receipt()
    if case == "missing":
        value.pop("privilege_profile")
    elif case == "strict":
        value["privilege_profile"] = "zero_bounding_v1"
    elif case == "unknown":
        value["privilege_profile"] = "arbitrary_profile"
    elif case == "version":
        value["schema_version"] = 1
    elif case == "extra":
        value["ready"] = True
    elif case == "bnd":
        value["guest_evidence"]["privilege_proof"]["cap_bnd"] |= 1 << 8
    else:
        value["guest_evidence"]["credential_present"] = True
    with pytest.raises(ValueError):
        decode(json.dumps(value).encode(), expected_profile=PROFILE)


def test_native_worker_consumes_fixed_shared_noarg_guard_before_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = strict_worker.worker()
    preflight = getattr(module, "_native_guest_preflight", None)
    assert callable(preflight), "missing-symbol: fixed native guest preflight"
    events: list[str] = []

    def identity() -> dict[str, Any]:
        events.append("identity")
        return native_proof()

    monkeypatch.setitem(sys.modules, "processing_identity", SimpleNamespace(verify_native_identity=identity))
    monkeypatch.setattr(module, "_output_directory_proof", lambda: strict_worker.admitted_output_proof())
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    for name in module._CREDENTIAL_ENV | module._PROXY_ENV | {"AWS_EC2_METADATA_DISABLED"}:
        monkeypatch.delenv(name, raising=False)

    def credentials() -> tuple[bool, None]:
        events.append("sdk")
        return False, None

    monkeypatch.setattr(module, "_credential_probe", credentials)
    monkeypatch.setattr(module, "_network_reachable", lambda *args: False)
    result = preflight()
    assert events == ["identity", "sdk"]
    assert result["privilege_proof"] == native_proof()

    def refusal() -> Any:
        raise ValueError("process privilege native fixture drift")

    monkeypatch.setitem(sys.modules, "processing_identity", SimpleNamespace(verify_native_identity=refusal))
    monkeypatch.setattr(module, "_credential_probe", lambda: pytest.fail("SDK reached after identity drift"))
    with pytest.raises(ValueError, match="privilege"):
        preflight()


def native_adapter(sm: Any, s3: Any) -> Any:
    module, _ = managed.modules()
    cls = getattr(module, "SageMakerNativeDatasetInspectionBackend", None)
    assert cls is not None, "missing-symbol: source-owned native inspection adapter"
    return cls(
        region=managed.REGION,
        role_arn=managed.ROLE,
        image_uri=managed.IMAGE,
        session_bucket=managed.BUCKET,
        source_commit=managed.SOURCE_COMMIT,
        admission_digest="a" * 64,
        sagemaker_client=sm,
        s3_client=s3,
    )


def test_native_parent_has_fixed_profile_without_constructor_selector() -> None:
    sm, s3 = managed.Processing(), managed.S3()
    adapter = native_adapter(sm, s3)
    assert "privilege_profile" not in inspect.signature(type(adapter)).parameters
    handle = adapter.submit(source=managed.source())
    assert handle.extra["privilege_profile"] == PROFILE
    assert sm.calls[0]["NetworkConfig"] == {"EnableNetworkIsolation": True}
    assert not sm.calls[0].get("Environment")
    assert not any("profile" in value for value in sm.calls[0]["AppSpecification"]["ContainerArguments"])
    handle.extra["privilege_profile"] = "zero_bounding_v1"
    assert adapter.status(handle).state.name == "UNKNOWN"


def publish_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, handle: Any, s3: Any) -> None:
    module, inputs, outputs = strict_worker.setup(monkeypatch, tmp_path, managed.PAYLOAD)
    arguments = strict_worker.binding(managed.PAYLOAD)
    arguments.update(job_name=handle.id, image_uri=managed.IMAGE)
    module.run_inspection(input_dir=inputs, output_dir=outputs, target_column="label", **arguments)
    for name in ("SUMMARY.json", "INSPECTED.json"):
        s3.versions[handle.extra["output_prefix"] + name] = (name + "-version", (outputs / name).read_bytes())


def test_native_parent_rejects_strict_receipt_and_verifies_native_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sm, s3 = managed.Processing(), managed.S3()
    adapter = native_adapter(sm, s3)
    handle = adapter.submit(source=managed.source(), target_column="label")
    publish_fixture(monkeypatch, tmp_path, handle, s3)
    sm.state = "Completed"
    assert adapter.report(handle)["success"] is False
    key = handle.extra["output_prefix"] + "INSPECTED.json"
    version, raw = s3.versions[key]
    value = json.loads(raw)
    value.update(schema_version=2, privilege_profile=PROFILE)
    value["guest_evidence"]["privilege_proof"] = native_proof()
    s3.versions[key] = (version, json.dumps(value).encode())
    assert adapter.report(handle)["success"] is True
    for name in ("image_uri", "source_commit", "job_name", "staged_version_id"):
        altered = deepcopy(value)
        altered[name] = "foreign"
        s3.versions[key] = (version, json.dumps(altered).encode())
        assert adapter.report(handle)["success"] is False
    s3.versions[key] = (version, json.dumps(value).encode())
    sm.extra_description = {"NetworkConfig": {"EnableNetworkIsolation": False}}
    assert adapter.report(handle)["success"] is False


def test_fixed_native_image_entrypoint_and_default_are_source_owned() -> None:
    module, _ = managed.modules()
    cls = getattr(module, "SageMakerNativeDatasetInspectionBackend", None)
    assert cls is not None, "missing-symbol: fixed native Processing default"
    from amlas.backends import registry
    from amlas.backends.ports import DataInspectionBackend

    assert isinstance(registry.REGISTRY.resolve(DataInspectionBackend).adapter, cls)
    text = (Path(__file__).resolve().parents[1] / "Dockerfile.inspection").read_text()
    assert "--features sagemaker-native-v1" in text
    assert "COPY processing_bootstrap/processing_identity.py /app/processing_identity.py" in text
    source = inspect.getsource(strict_worker.worker().main)
    assert "_run_native_inspection(" in source
    assert "privilege-profile" not in source


def test_fixed_native_worker_emits_versioned_bound_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, inputs, outputs = strict_worker.setup(monkeypatch, tmp_path, managed.PAYLOAD)
    run = getattr(module, "_run_native_inspection", None)
    assert callable(run), "missing-symbol: fixed native inspection path"
    assert "privilege_profile" not in inspect.signature(run).parameters
    assert "expected_profile" not in inspect.signature(run).parameters
    evidence = module._guest_preflight()
    evidence["privilege_proof"] = native_proof()
    monkeypatch.setattr(module, "_native_guest_preflight", lambda: evidence)
    run(input_dir=inputs, output_dir=outputs, target_column="label", **strict_worker.binding(managed.PAYLOAD))
    assert sorted(path.name for path in outputs.iterdir()) == ["INSPECTED.json", "SUMMARY.json"]
    raw = (outputs / "INSPECTED.json").read_bytes()
    decoded = schema().decode_receipt(raw, expected_profile=PROFILE)
    assert decoded["schema_version"] == 2 and decoded["privilege_profile"] == PROFILE
    assert decoded["guest_evidence"]["privilege_proof"] == native_proof()
    assert decoded["source_commit"] == managed.SOURCE_COMMIT
    assert decoded["source_version_id"] == "source-version"
    assert decoded["staged_version_id"] == "staged-version"
    assert decoded["prepared"] is False and decoded["trained"] is False
    with pytest.raises(ValueError):
        schema().decode_receipt(raw)
