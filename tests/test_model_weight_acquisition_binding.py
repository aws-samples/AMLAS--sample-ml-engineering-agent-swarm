"""Normal model tools preserve caller authority/opaque sessions, closed providers."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import model_acquisition_fixtures as f
import pytest

from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy


@pytest.fixture(autouse=True)
def namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *a, **k: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")


def modules() -> tuple[Any, Any, Any]:
    store = f.require_module("amlas.control_plane.model_acquisition_state", "SessionModelAcquisitionStore")
    flow = f.require_module("amlas.flows.model_weight_acquisition_flow", "run_model_weight_acquisition_flow")
    backend = f.require_module(
        "amlas.backends.adapters.model_weight_acquisition", "ProcessingModelWeightAcquisitionBackend"
    )
    return store, flow, backend


@pytest.mark.parametrize("case", ["missing", "dict_policy", "dict_store", "ask", "capped_unknown"])
def test_typed_caller_authority_denies_before_submission(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    stores, flow, backend = modules()
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(backend, sm, s3)
    monkeypatch.setattr(flow, "_resolve_backend", lambda: adapter)
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO)
    store = stores.SessionModelAcquisitionStore(runtime_session_id=f.SESSION)
    if case == "missing":
        policy = None
    elif case == "dict_policy":
        policy = {"mode": "auto"}
    elif case == "dict_store":
        store = {"runtime_session_id": f.SESSION}
    elif case == "ask":
        policy = ApprovalPolicy(mode=ApprovalMode.ASK)
    else:
        policy = ApprovalPolicy(
            mode=ApprovalMode.ALLOWLIST, allowlist=["run_processing_job"], max_auto_cost=10
        )
    result = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, policy=policy, store=store
    )
    assert result["accepted"] is False and result["status"] == "denied"
    assert sm.requests == []


def test_plan_only_and_session_opaque_refs_never_reconstruct_native_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores, flow, backend = modules()
    sm, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(backend, sm, s3)
    monkeypatch.setattr(flow, "_resolve_backend", lambda: adapter)
    plan = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, plan_only=True
    )
    assert plan["status"] == "planned" and sm.requests == []
    store = stores.SessionModelAcquisitionStore(runtime_session_id=f.SESSION)
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO)
    first = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, policy=policy, store=store
    )
    assert first["request_ref"].startswith("model_acq_") and first["accepted"] is True
    again = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, policy=policy, store=store
    )
    assert again["request_ref"] == first["request_ref"] and len(sm.requests) == 1
    other = stores.SessionModelAcquisitionStore(runtime_session_id="another-session")
    assert other.observe(first["request_ref"])["status"] == "unknown_reference"
    assert store.observe({"job_arn": first["job_arn"], "approved": True})["status"] == "unknown_reference"
    sm.read_failure = True
    observed = store.observe(first["request_ref"])
    assert observed["request_ref"] == first["request_ref"] and observed["acquired"] is False
    assert len(sm.requests) == 1


def test_model_tools_hide_policy_store_and_native_configuration_from_schema() -> None:
    modules()
    toolmod = f.require_module(
        "amlas.tools.model_weight_tools", "acquire_model_weights", "get_model_weight_acquisition"
    )
    for tool in (toolmod.acquire_model_weights, toolmod.get_model_weight_acquisition):
        properties = tool.tool_spec["inputSchema"]["json"]["properties"]
        assert not set(properties) & {
            "policy",
            "store",
            "tool_context",
            "role_arn",
            "image_uri",
            "source_commit",
            "environment",
            "output_dir",
            "native_handle",
            "url",
        }
    schema = toolmod.acquire_model_weights.tool_spec["inputSchema"]["json"]
    assert set(schema["properties"]) == {
        "catalog_id",
        "catalog_version",
        "plan_only",
        "mode",
        "pack_ref",
    }
    assert schema["properties"]["mode"]["enum"] == ["acquire", "reuse_verified"]
    assert schema["properties"]["mode"]["default"] == "acquire"
    assert set(schema["required"]) == {"catalog_id", "catalog_version"}


def test_fixed_worker_identity_and_source_gates_precede_core_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = f.require_module("amlas.sagemaker_entrypoints.model_weight_acquisition", "main", "_guard_worker")
    monkeypatch.setitem(
        sys.modules,
        "processing_identity",
        SimpleNamespace(
            verify_native_identity=lambda: (_ for _ in ()).throw(ValueError("processing identity refused"))
        ),
    )
    monkeypatch.setattr(worker, "_load_core", lambda: pytest.fail("core imported before guard"))
    with pytest.raises((ValueError, RuntimeError)):
        worker._guard_worker()


@pytest.mark.parametrize(
    "name",
    [
        "AWS_ACCESS_KEY_ID",
        "AWS_PROFILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "HTTP_PROXY",
        "https_proxy",
        "AWS_EC2_METADATA_DISABLED",
    ],
)
def test_fixed_worker_provider_hint_presence_never_becomes_fake_absence(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    worker = f.require_module("amlas.sagemaker_entrypoints.model_weight_acquisition", "_credential_absence")
    for key in worker._PROVIDER_HINTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(name, "private-fixture-value-never-log")
    with pytest.raises((ValueError, RuntimeError)) as error:
        worker._credential_absence()
    assert "private-fixture-value-never-log" not in str(error.value)


def test_inaccessible_root_provider_paths_refuse_without_claiming_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = f.require_module("amlas.sagemaker_entrypoints.model_weight_acquisition", "_credential_absence")
    for key in worker._PROVIDER_HINTS:
        monkeypatch.delenv(key, raising=False)
    paths = []

    def observed(path: Any) -> Any:
        paths.append(str(path))
        if str(path).startswith("/root/"):
            raise PermissionError("closed inaccessible boundary")
        raise FileNotFoundError

    monkeypatch.setattr(worker.os, "lstat", observed)
    monkeypatch.setattr(
        worker.socket,
        "create_connection",
        lambda *args, **kwargs: pytest.fail("metadata test bypassed unknown credential path"),
    )
    with pytest.raises(RuntimeError, match="unverified"):
        worker._credential_absence()
    assert "/root/.aws/credentials" in paths


def test_fixed_worker_loads_only_package_core_after_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = f.require_module("amlas.sagemaker_entrypoints.model_weight_acquisition", "_load_core")
    from amlas.data_eng import model_weight_acquisition

    foreign = SimpleNamespace(unbound_fixture=True)
    monkeypatch.setitem(sys.modules, "model_weight_acquisition", foreign)
    monkeypatch.setattr(worker, "__package__", "")
    monkeypatch.setattr(worker.sys, "path", list(sys.path))
    loaded = worker._load_core()
    assert loaded is model_weight_acquisition, "worker loaded an unbound top-level core"
    assert worker.sys.path[0] == "/app/src"


def test_worker_unverified_inherited_descriptor_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    import stat

    worker = f.require_module("amlas.sagemaker_entrypoints.model_weight_acquisition", "_guard_worker")
    monkeypatch.setitem(
        sys.modules,
        "processing_identity",
        SimpleNamespace(verify_native_identity=lambda: f.evidence()["privilege_proof"]),
    )
    metadata = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=10001, st_gid=10001, st_dev=1, st_ino=2)

    scanned = []

    def observed(fd: int) -> Any:
        if fd == 7:
            scanned.append(fd)
            raise PermissionError("closed unknown descriptor")
        return metadata

    monkeypatch.setattr(worker.os, "fstat", observed)
    monkeypatch.setattr(worker, "_open_directory", lambda path: 6)
    monkeypatch.setattr(worker, "_open_output_contender", lambda: 6)
    monkeypatch.setattr(worker, "_read_provider_inventory", lambda _fd: {"inventory": {}, "sha256": "a" * 64})
    monkeypatch.setattr(worker.os, "close", lambda fd: None)

    def lock(fd: int, flags: int) -> None:
        if fd == 6 and flags & worker.fcntl.LOCK_EX:
            raise BlockingIOError

    monkeypatch.setattr(worker.fcntl, "flock", lock)
    monkeypatch.setattr(worker.os, "listdir", lambda path: ["0", "1", "2", "3", "7"])
    monkeypatch.setattr(worker, "_credential_absence", lambda: {"provider_files_present": False})
    with pytest.raises((PermissionError, RuntimeError)):
        worker._guard_worker()
    assert scanned == [7], "fixture must reach the intended inherited-FD observation"


@pytest.mark.parametrize("code", [24, 12, 23, 105])
def test_metadata_resource_observation_failure_refuses(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    worker = f.require_module("amlas.sagemaker_entrypoints.model_weight_acquisition", "_credential_absence")
    for key in worker._PROVIDER_HINTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(worker.os, "lstat", lambda path: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(
        worker.socket,
        "create_connection",
        lambda *a, **k: (_ for _ in ()).throw(OSError(code, "resource observation unverified")),
    )
    with pytest.raises(RuntimeError, match="unverified"):
        worker._credential_absence()


@pytest.mark.parametrize("reply", [None, [], "malformed", 1])
def test_post_create_response_interpretation_preserves_ambiguous_record(
    monkeypatch: pytest.MonkeyPatch, reply: Any
) -> None:
    stores, flow, backend = modules()

    class AcceptedThenMalformed(f.Processing):
        def create_processing_job(self, **kwargs: Any) -> Any:
            super().create_processing_job(**kwargs)
            return reply

    sm, s3 = AcceptedThenMalformed(), f.S3()
    adapter = f.configured_backend(backend, sm, s3)
    monkeypatch.setattr(flow, "_resolve_backend", lambda: adapter)
    store = stores.SessionModelAcquisitionStore(runtime_session_id=f.SESSION)
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO)
    first = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, policy=policy, store=store
    )
    assert first["status"] == "submission_unknown" and first["request_ref"]
    again = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, policy=policy, store=store
    )
    assert again["request_ref"] == first["request_ref"] and len(sm.requests) == 1


@pytest.mark.parametrize("code", [111, 110, 101, 113, 13, 1, None])
def test_qualified_fixed_metadata_target_negative_observation_preserves_all_provider_probes(
    monkeypatch: pytest.MonkeyPatch, code: Any
) -> None:
    worker = f.require_module("amlas.sagemaker_entrypoints.model_weight_acquisition", "_credential_absence")
    for key in worker._PROVIDER_HINTS:
        monkeypatch.delenv(key, raising=False)
    paths = []
    endpoints = []

    def absent(path: Any) -> Any:
        paths.append(str(path))
        raise FileNotFoundError

    def denied(address: Any, timeout: float) -> Any:
        endpoints.append((address, timeout))
        if code is None:
            raise TimeoutError
        raise OSError(code, "bounded fixed target observation")

    monkeypatch.setattr(worker.os, "lstat", absent)
    monkeypatch.setattr(worker.socket, "create_connection", denied)
    observation = worker._credential_absence()
    assert observation["metadata_endpoint_reachable"] is False
    assert paths == [
        "/home/appuser/.aws/credentials",
        "/home/appuser/.aws/config",
        "/root/.aws/credentials",
        "/root/.aws/config",
    ]
    assert endpoints == [(("169.254.169.254", 80), 0.5), (("169.254.170.2", 80), 0.5)]


def test_post_create_mapping_get_fault_still_retains_original_model_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores, flow, backend = modules()

    class BrokenMapping(dict):
        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("closed response interpretation fault")

    class Processing(f.Processing):
        def create_processing_job(self, **kwargs: Any) -> Any:
            reply = super().create_processing_job(**kwargs)
            return BrokenMapping(reply)

    sm, s3 = Processing(), f.S3()
    adapter = f.configured_backend(backend, sm, s3)
    monkeypatch.setattr(flow, "_resolve_backend", lambda: adapter)
    store = stores.SessionModelAcquisitionStore(runtime_session_id=f.SESSION)
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO)
    first = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, policy=policy, store=store
    )
    again = flow.run_model_weight_acquisition_flow(
        catalog_id=f.CATALOG, catalog_version=f.VERSION, policy=policy, store=store
    )
    assert first["status"] == "submission_unknown" and again["request_ref"] == first["request_ref"]
    assert len(sm.requests) == 1
