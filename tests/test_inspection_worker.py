"""Fixed data-only guest controls; synthetic local inputs, closed probes."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_FIXTURE_FDS: list[int] = []


@pytest.fixture(autouse=True)
def _close_fixture_descriptors() -> Any:
    before = len(_FIXTURE_FDS)
    yield
    while len(_FIXTURE_FDS) > before:
        os.close(_FIXTURE_FDS.pop())


def worker() -> Any:
    try:
        return importlib.import_module("amlas.sagemaker_entrypoints.dataset_inspection")
    except ModuleNotFoundError:
        pytest.fail("missing-symbol: fixed dataset inspection entrypoint")


def binding(data: bytes, fmt: str = "csv") -> dict[str, Any]:
    return {
        "expected_sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
        "format": fmt,
        "job_name": "fixture-inspection",
        "runtime_session_id": "session-a",
        "deployment_id": "fd0d82ab-4e03-44c0-9bf1-908af5b2b031",
        "deployment_prefix": "fixture-lab",
        "image_uri": "1234" * 3 + ".dkr.ecr.us-west-2.amazonaws.com/inspection@sha256:" + "a" * 64,
        "source_commit": "b" * 40,
        "source_version_id": "source-version",
        "staged_version_id": "staged-version",
    }


def setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data: bytes, fmt: str = "csv"
) -> tuple[Any, Path, Path]:
    module = worker()
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    _FIXTURE_FDS.append(descriptor)
    monkeypatch.setattr(module, "_OUTPUT_FD", descriptor)
    (source / ("data." + fmt)).write_bytes(data)
    monkeypatch.setattr(
        module,
        "_guest_preflight",
        lambda: {
            "output_directory_proof": admitted_output_proof(),
            "credential_present": False,
            "provider_method": None,
            "credential_env_names": [],
            "metadata_reachable": False,
            "network_reachable": False,
            "privilege_proof": {
                "res_uid": [10001, 10001, 10001],
                "res_gid": [10001, 10001, 10001],
                "groups": [],
                "cap_eff": 0,
                "cap_prm": 0,
                "cap_inh": 0,
                "cap_amb": 0,
                "cap_bnd": 0,
                "no_new_privs": 1,
            },
        },
    )
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "b" * 40)
    return module, source, output


def test_fixed_worker_emits_only_bounded_summary_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = b"label,value\na,1\nb,2\na,3\n"
    module, source, output = setup(monkeypatch, tmp_path, data)
    module.run_inspection(input_dir=source, output_dir=output, target_column="label", **binding(data))
    assert sorted(item.name for item in output.iterdir()) == ["INSPECTED.json", "SUMMARY.json"]
    summary = json.loads((output / "SUMMARY.json").read_text())
    assert summary["rows_scanned"] == 3 and summary["rows_total"] == 3
    assert summary["target_distribution"] == {"a": 2, "b": 1}
    assert summary["untrusted_data"] is True
    assert summary["prepared"] is False and summary["trained"] is False
    assert (output / "SUMMARY.json").stat().st_size <= 65536
    receipt = json.loads((output / "INSPECTED.json").read_text())
    assert receipt["summary_sha256"] == hashlib.sha256((output / "SUMMARY.json").read_bytes()).hexdigest()
    assert receipt["source_version_id"] == "source-version"


@pytest.mark.parametrize("case", ["hash", "extra_input", "symlink", "source_commit"])
def test_input_binding_failures_never_reach_parser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    data = b"label\na\n"
    module, source, output = setup(monkeypatch, tmp_path, data)
    args = binding(data)
    if case == "hash":
        args["expected_sha256"] = "c" * 64
    elif case == "extra_input":
        (source / "extra.csv").write_bytes(b"extra")
    elif case == "symlink":
        original = source / "data.csv"
        original.rename(tmp_path / "moved.csv")
        original.symlink_to(tmp_path / "moved.csv")
    else:
        args["source_commit"] = "c" * 40
    monkeypatch.setattr(module, "_inspect_file", lambda *a, **k: pytest.fail("unverified input parsed"))
    with pytest.raises(ValueError):
        module.run_inspection(input_dir=source, output_dir=output, **args)
    assert list(output.iterdir()) == []


def test_nested_parquet_preview_preserves_structure_and_bounds_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "fixture.parquet"
    pd.DataFrame(
        [
            {
                "messages": [{"role": "assistant", "content": "x" * 500000}],
                "resolved": True,
            }
        ]
    ).to_parquet(source_file, index=False)
    data = source_file.read_bytes()
    module, source, output = setup(monkeypatch, tmp_path, data, "parquet")
    module.run_inspection(input_dir=source, output_dir=output, **binding(data, "parquet"))
    summary = json.loads((output / "SUMMARY.json").read_text())
    assert summary["sample_truncated"] is True
    assert len(summary["sample_rows"][0]["messages"][0]["content"]) <= 512
    assert (output / "SUMMARY.json").stat().st_size <= 65536


def test_credential_preflight_denies_before_any_untrusted_parsing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = b"label\na\n"
    module, source, output = setup(monkeypatch, tmp_path, data)

    def refusal() -> Any:
        raise ValueError("credential provider present")

    monkeypatch.setattr(module, "_guest_preflight", refusal)
    monkeypatch.setattr(
        module, "_inspect_file", lambda *a, **k: pytest.fail("credentialful guest parsed data")
    )
    with pytest.raises(ValueError, match="credential"):
        module.run_inspection(input_dir=source, output_dir=output, **binding(data))
    assert list(output.iterdir()) == []


def test_strict_summary_rejects_extra_fields_and_nonfinite_json() -> None:
    try:
        schema = importlib.import_module("amlas.data_eng.inspection_schema")
    except ModuleNotFoundError:
        pytest.fail("missing-symbol: strict fixed inspection schema")
    with pytest.raises(ValueError):
        schema.decode_summary(b'{"schema_version":1,"arbitrary_export":"secret"}')
    with pytest.raises(ValueError):
        schema.decode_summary(b'{"schema_version":NaN}')
    with pytest.raises(ValueError):
        schema.decode_summary(b'{"schema_version":1,"schema_version":1}')


def test_image_is_fixed_amd64_nonroot_and_uses_only_locked_data_probe_closure() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "Dockerfile.inspection"
    assert path.is_file(), "missing-symbol: locked inspection image"
    text = path.read_text()
    assert "linux/amd64" in text and "@sha256:" in text
    assert "--require-hashes" in text and "--only-binary=:all:" in text
    assert "USER root" in text
    assert "amlas-processing-init" in text
    assert "inspect_dataset.py" in text
    assert "inspection_schema.py" in text
    assert "AWS_EC2_METADATA_DISABLED" not in text
    assert "torch" not in text and "strands" not in text


def test_credential_probe_reads_presence_method_without_values_or_service_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import boto3

    module = worker()

    class Credentials:
        method = "env"

        @property
        def access_key(self) -> Any:
            pytest.fail("credential value read")

        @property
        def secret_key(self) -> Any:
            pytest.fail("credential value read")

    class Session:
        _session = SimpleNamespace(set_config_variable=lambda *args: None)

        def get_credentials(self) -> Credentials:
            return Credentials()

        def client(self, *args: Any, **kwargs: Any) -> Any:
            pytest.fail("guest SDK service client created")

    monkeypatch.setattr(boto3, "Session", Session)
    assert module._credential_probe() == (True, "env")


def test_forced_metadata_disable_does_not_fake_native_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = worker()
    monkeypatch.setattr(module, "_process_privilege_proof", admitted_privilege_proof)
    monkeypatch.setattr(module, "_output_directory_proof", admitted_output_proof)
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(module.os, "geteuid", lambda: 10001)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    with pytest.raises(ValueError, match="metadata"):
        module._guest_preflight()


@pytest.mark.parametrize("kind", ["arrays", "objects"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-32"])
def test_structural_depth_is_rejected_before_json_allocation(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    encoding: str,
) -> None:
    schema = importlib.import_module("amlas.data_eng.inspection_schema")
    nested = "[" * 16 + "0" + "]" * 16 if kind == "arrays" else '{"v":' * 16 + "0" + "}" * 16
    data = ('{"value":' + nested + "}").encode(encoding)
    calls = []

    def forbidden_loads(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        raise ValueError("json decoder reached")

    monkeypatch.setattr(schema.json, "loads", forbidden_loads)
    with pytest.raises(ValueError):
        schema.decode_object(data, limit=65536)
    assert calls == [], "depth must be admitted before json.loads materializes nested objects"


@pytest.mark.parametrize(
    "text",
    [
        "{[}]" * 100,
        'escaped quote " then [ { ] }',
        'backslash \\ and "quote',
        "unicode 雪 with { brackets [ inside ] text }",
    ],
)
def test_depth_scan_ignores_string_delimiters_and_escapes(text: str) -> None:
    schema = importlib.import_module("amlas.data_eng.inspection_schema")
    value = {"sample": [{"value": text}], "after": {"value": "["}}
    assert schema.decode_object(json.dumps(value, ensure_ascii=False).encode(), limit=65536) == value


@pytest.mark.parametrize(
    "data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":[}']
)
def test_depth_admission_keeps_strict_json_failures(data: bytes) -> None:
    schema = importlib.import_module("amlas.data_eng.inspection_schema")
    with pytest.raises(ValueError):
        schema.decode_object(data, limit=65536)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-32"])
def test_json_depth_boundary_and_encoded_escapes_remain_valid(encoding: str) -> None:
    schema = importlib.import_module("amlas.data_eng.inspection_schema")
    leaf = json.dumps("quoted text " + chr(34) + chr(92) + chr(34) + "[{]}")
    nested = "[" * 15 + leaf + "]" * 15
    raw = ('{"value":' + nested + "}").encode(encoding)
    assert schema.decode_object(raw, limit=65536) == json.loads(raw)


@pytest.mark.parametrize("external_reachable", [False, True])
def test_guest_preflight_probes_fixed_admitted_numeric_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    external_reachable: bool,
) -> None:
    from contextlib import nullcontext

    module = worker()
    monkeypatch.setattr(module, "_process_privilege_proof", admitted_privilege_proof)
    monkeypatch.setattr(module, "_output_directory_proof", admitted_output_proof)
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(module.os, "geteuid", lambda: 10001)
    for name in module._CREDENTIAL_ENV | {"AWS_EC2_METADATA_DISABLED"}:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "_credential_probe", lambda: (False, None))
    calls: list[tuple[tuple[str, int], float]] = []
    metadata_endpoints = [("169.254.169.254", 80), ("169.254.170.2", 80)]
    admitted_external_endpoint = ("1.1.1.1", 443)

    def connection(address: tuple[str, int], *, timeout: float) -> Any:
        calls.append((address, timeout))
        assert address in [*metadata_endpoints, admitted_external_endpoint], (
            "guest must probe the numeric endpoint verified by the native admission control"
        )
        if address in metadata_endpoints or not external_reachable:
            raise OSError("closed fixture endpoint")
        return nullcontext()

    monkeypatch.setattr(module.socket, "create_connection", connection)
    if external_reachable:
        with pytest.raises(ValueError, match="network-denial"):
            module._guest_preflight()
    else:
        report = module._guest_preflight()
        assert report["network_reachable"] is False
    assert calls == [(endpoint, 0.5) for endpoint in [*metadata_endpoints, admitted_external_endpoint]]


def admitted_privilege_proof() -> dict[str, Any]:
    return {
        "res_uid": [10001, 10001, 10001],
        "res_gid": [10001, 10001, 10001],
        "groups": [],
        "cap_eff": 0,
        "cap_prm": 0,
        "cap_inh": 0,
        "cap_amb": 0,
        "cap_bnd": 0,
        "no_new_privs": 1,
    }


@pytest.mark.parametrize("case", ["root", "saved_uid", "gid", "groups", "cap_eff", "cap_bnd", "nnp"])
def test_worker_requires_permanent_privilege_proof_before_sdk(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    module = worker()
    proof_function = getattr(module, "_process_privilege_proof", None)
    assert callable(proof_function), "missing-symbol: permanent worker privilege proof"
    uid = (10001, 10001, 10001)
    gid = (10001, 10001, 10001)
    groups: list[int] = []
    status = {"CapEff": "0", "CapPrm": "0", "CapInh": "0", "CapAmb": "0", "CapBnd": "0", "NoNewPrivs": "1"}
    if case == "root":
        uid = (0, 0, 0)
    elif case == "saved_uid":
        uid = (10001, 10001, 0)
    elif case == "gid":
        gid = (10001, 0, 10001)
    elif case == "groups":
        groups = [0]
    elif case == "cap_eff":
        status["CapEff"] = "1"
    elif case == "cap_bnd":
        status["CapBnd"] = "1"
    else:
        status["NoNewPrivs"] = "0"
    monkeypatch.setattr(module.os, "getresuid", lambda: uid)
    monkeypatch.setattr(module.os, "getresgid", lambda: gid)
    monkeypatch.setattr(module.os, "getgroups", lambda: groups)
    monkeypatch.setattr(module, "_read_process_status", lambda: status)
    monkeypatch.setattr(module, "_credential_probe", lambda: pytest.fail("SDK reached before drop"))
    with pytest.raises(ValueError, match="privilege"):
        module._guest_preflight()


def test_process_privilege_proof_records_actual_kernel_values(monkeypatch: pytest.MonkeyPatch) -> None:
    module = worker()
    function = getattr(module, "_process_privilege_proof", None)
    assert callable(function), "missing-symbol: permanent worker privilege proof"
    monkeypatch.setattr(module.os, "getresuid", lambda: (10001, 10001, 10001))
    monkeypatch.setattr(module.os, "getresgid", lambda: (10001, 10001, 10001))
    monkeypatch.setattr(module.os, "getgroups", lambda: [])
    monkeypatch.setattr(
        module,
        "_read_process_status",
        lambda: {
            "CapEff": "0",
            "CapPrm": "0",
            "CapInh": "0",
            "CapAmb": "0",
            "CapBnd": "0",
            "NoNewPrivs": "1",
        },
    )
    assert function() == admitted_privilege_proof()


@pytest.mark.parametrize("case", ["missing", "bool_uid", "root_uid", "caps", "nnp", "extra"])
def test_receipt_requires_strict_actual_privilege_proof(case: str) -> None:
    from copy import deepcopy

    schema = importlib.import_module("amlas.data_eng.inspection_schema")
    evidence = {
        "output_directory_proof": admitted_output_proof(),
        "credential_present": False,
        "provider_method": None,
        "credential_env_names": [],
        "metadata_reachable": False,
        "network_reachable": False,
    }
    receipt = {
        "schema_version": 1,
        **binding(b"label\na\n"),
        "summary_sha256": "a" * 64,
        "guest_evidence": evidence,
        "prepared": False,
        "trained": False,
    }
    receipt.pop("expected_sha256")
    receipt["sha256"] = hashlib.sha256(b"label\na\n").hexdigest()
    if case != "missing":
        proof = deepcopy(admitted_privilege_proof())
        if case == "bool_uid":
            proof["res_uid"][0] = True
        elif case == "root_uid":
            proof["res_uid"][1] = 0
        elif case == "caps":
            proof["cap_eff"] = 1
        elif case == "nnp":
            proof["no_new_privs"] = 0
        else:
            proof["arbitrary"] = True
        evidence["privilege_proof"] = proof
    with pytest.raises(ValueError, match="privilege"):
        schema.decode_receipt(json.dumps(receipt).encode())


def test_image_binds_root_only_trusted_initializer_before_fixed_worker() -> None:
    text = (Path(__file__).resolve().parents[1] / "Dockerfile.inspection").read_text()
    assert "rust:1.90.0-alpine3.22@sha256:" in text
    assert "processing_bootstrap" in text
    assert 'ENTRYPOINT ["/usr/local/bin/amlas-processing-init", "inspection"]' in text
    assert "USER root" in text
    assert "chmod 0555 /app/inspect_dataset.py /app/inspection_schema.py" in text


def test_isolated_worker_refuses_privilege_before_schema_or_sdk_import() -> None:
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "src/amlas/sagemaker_entrypoints/dataset_inspection.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            "--sha256",
            "a" * 64,
            "--byte-count",
            "1",
            "--format",
            "csv",
            "--job-name",
            "fixture-job",
            "--runtime-session-id",
            "session-a",
            "--deployment-id",
            "fixture-deployment",
            "--deployment-prefix",
            "fixture-lab",
            "--image-uri",
            "fixture-image",
            "--source-commit",
            "b" * 40,
            "--source-version-id",
            "source-version",
            "--staged-version-id",
            "staged-version",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "AMLAS_SOURCE_COMMIT": "b" * 40},
    )
    assert completed.returncode != 0
    assert "process privilege" in completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr


@pytest.mark.parametrize(
    "name",
    [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ],
)
def test_proxy_hint_presence_denies_before_sdk_without_logging_values(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    module = worker()
    monkeypatch.setattr(module, "_process_privilege_proof", admitted_privilege_proof)
    monkeypatch.setattr(module, "_output_directory_proof", admitted_output_proof)
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    for variable in module._CREDENTIAL_ENV | {
        "AWS_EC2_METADATA_DISABLED",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, "private-fixture-value-never-log")
    monkeypatch.setattr(module, "_credential_probe", lambda: pytest.fail("SDK reached with proxy hint"))
    with pytest.raises(ValueError, match="proxy") as caught:
        module._guest_preflight()
    assert "private-fixture-value-never-log" not in str(caught.value)


@pytest.mark.parametrize("value", ["", "false", "0", "true"])
def test_any_metadata_disable_hint_presence_refuses_before_sdk(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    module = worker()
    monkeypatch.setattr(module, "_process_privilege_proof", admitted_privilege_proof)
    monkeypatch.setattr(module, "_output_directory_proof", admitted_output_proof)
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", value)
    monkeypatch.setattr(
        module, "_credential_probe", lambda: pytest.fail("SDK reached with disabled-discovery hint")
    )
    with pytest.raises(ValueError, match="metadata"):
        module._guest_preflight()


@pytest.mark.parametrize(
    "name",
    [
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
    ],
)
def test_provider_hint_presence_refuses_before_sdk_without_values(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    module = worker()
    monkeypatch.setattr(module, "_process_privilege_proof", admitted_privilege_proof)
    monkeypatch.setattr(module, "_output_directory_proof", admitted_output_proof)
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    for variable in module._CREDENTIAL_ENV | module._PROXY_ENV | {"AWS_EC2_METADATA_DISABLED"}:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, "fixture-provider-hint-never-log")
    monkeypatch.setattr(module, "_credential_probe", lambda: pytest.fail("SDK reached with provider hint"))
    with pytest.raises(ValueError, match="credential") as caught:
        module._guest_preflight()
    assert "fixture-provider-hint-never-log" not in str(caught.value)


def admitted_output_proof() -> dict[str, Any]:
    return {
        "fd": 3,
        "uid": 10001,
        "gid": 10001,
        "mode": 0o700,
        "device": 1,
        "inode": 1,
        "lease_active": True,
    }


def test_v2_source_binding_refuses_before_sdk_or_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = b"label\na\n"
    module, source, output = setup(monkeypatch, tmp_path, data)
    args = binding(data)
    args["source_commit"] = "c" * 40
    monkeypatch.setattr(
        module, "_guest_preflight", lambda: pytest.fail("SDK guard reached before source binding")
    )
    monkeypatch.setattr(module, "_load_schema", lambda: pytest.fail("schema reached before source binding"))
    with pytest.raises(ValueError, match=r"source|binding"):
        module.run_inspection(input_dir=source, output_dir=output, **args)


def test_v2_input_original_parent_symlink_refuses_before_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = b"label\na\n"
    module, source, output = setup(monkeypatch, tmp_path, data)
    alias = tmp_path / "input-alias"
    alias.symlink_to(source, target_is_directory=True)
    monkeypatch.setattr(module, "_inspect_file", lambda *args: pytest.fail("symlink parent reached parser"))
    with pytest.raises((ValueError, OSError)):
        module.run_inspection(input_dir=alias, output_dir=output, **binding(data))


def test_v2_output_substitution_refuses_publication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os

    data = b"label\na\n"
    module, source, output = setup(monkeypatch, tmp_path, data)
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    monkeypatch.setattr(module, "_OUTPUT_FD", descriptor, raising=False)
    original_inspect = module._inspect_file
    moved = tmp_path / "original-output"

    def replace_after_inspection(*args: Any) -> dict[str, Any]:
        result = original_inspect(*args)
        output.rename(moved)
        output.mkdir()
        return result

    monkeypatch.setattr(module, "_inspect_file", replace_after_inspection)
    try:
        with pytest.raises(ValueError, match=r"output|identity"):
            module.run_inspection(input_dir=source, output_dir=output, **binding(data))
        assert list(moved.iterdir()) == []
        assert list(output.iterdir()) == []
    finally:
        os.close(descriptor)


def test_v2_json_publication_uses_only_authoritative_output_fd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    data = b"label\na\n"
    module, source, output = setup(monkeypatch, tmp_path, data)
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    monkeypatch.setattr(module, "_OUTPUT_FD", descriptor, raising=False)
    real_open = os.open
    published = []

    def confined_open(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        if str(path).endswith(("SUMMARY.json", "INSPECTED.json")):
            assert path in {"SUMMARY.json", "INSPECTED.json"}, "absolute output pathname reopened"
            assert kwargs.get("dir_fd") == descriptor, "authoritative output FD bypassed"
            assert flags & os.O_EXCL and flags & os.O_NOFOLLOW
            published.append(path)
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(module.os, "open", confined_open)
    try:
        module.run_inspection(input_dir=source, output_dir=output, **binding(data))
        assert published == ["SUMMARY.json", "INSPECTED.json"]
    finally:
        os.close(descriptor)


def test_v2_directory_lease_gate_precedes_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = worker()
    monkeypatch.setattr(module, "_process_privilege_proof", admitted_privilege_proof)
    monkeypatch.setattr(module, "_output_directory_proof", admitted_output_proof)

    def refusal() -> Any:
        raise ValueError("output directory lease refused")

    monkeypatch.setattr(module, "_output_directory_proof", refusal, raising=False)
    monkeypatch.setattr(module, "_credential_probe", lambda: pytest.fail("SDK reached before directory gate"))
    with pytest.raises(ValueError, match="output directory lease"):
        module._guest_preflight()


@pytest.mark.parametrize(
    "field,value",
    [
        ("fd", 4),
        ("uid", 0),
        ("gid", 0),
        ("mode", 0o755),
        ("device", True),
        ("device", -1),
        ("inode", 0),
        ("inode", True),
        ("lease_active", False),
        ("lease_active", 1),
    ],
)
def test_v2_output_receipt_rejects_unadmitted_descriptor_proof(field: str, value: Any) -> None:
    schema = importlib.import_module("amlas.data_eng.inspection_schema")
    validator = getattr(schema, "validate_output_directory_proof", None)
    assert callable(validator), "missing-symbol: strict output directory proof"
    proof = admitted_output_proof()
    proof[field] = value
    with pytest.raises(ValueError):
        validator(proof)


def test_v2_output_receipt_requires_exact_bounded_fields() -> None:
    schema = importlib.import_module("amlas.data_eng.inspection_schema")
    validator = getattr(schema, "validate_output_directory_proof", None)
    assert callable(validator), "missing-symbol: strict output directory proof"
    assert validator(admitted_output_proof()) == admitted_output_proof()
    for proof in (
        {},
        {**admitted_output_proof(), "ready": True},
        {**admitted_output_proof(), "inode": 2**65},
    ):
        with pytest.raises(ValueError):
            validator(proof)


def test_v2_same_inode_input_mutation_refuses_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = b"label\na\n"
    module, source, output = setup(monkeypatch, tmp_path, data)
    original_inspect = module._inspect_file
    original = (source / "data.csv").stat()
    mutated = []

    def mutate_same_inode(*args: Any) -> dict[str, Any]:
        result = original_inspect(*args)
        path = source / "data.csv"
        with path.open("r+b") as stream:
            stream.write(b"label\nb\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        changed = path.stat()
        assert (changed.st_dev, changed.st_ino, changed.st_size) == (
            original.st_dev,
            original.st_ino,
            original.st_size,
        )
        mutated.append(True)
        return result

    monkeypatch.setattr(module, "_inspect_file", mutate_same_inode)
    with pytest.raises(ValueError, match=r"input.*stability|input.*changed"):
        module.run_inspection(input_dir=source, output_dir=output, **binding(data))
    assert mutated == [True]
    assert list(output.iterdir()) == []


def test_v2_constant_stat_same_inode_mutation_requires_post_parse_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = b"label\na\n"
    module, source, output = setup(monkeypatch, tmp_path, data)
    path = source / "data.csv"
    original = path.stat()
    original_inspect = module._inspect_file
    real_fstat = os.fstat
    mutations = []

    def coarse_stat(descriptor: int) -> os.stat_result:
        actual = real_fstat(descriptor)
        if (actual.st_dev, actual.st_ino) == (original.st_dev, original.st_ino):
            return original
        return actual

    def mutate_same_inode(*args: Any) -> dict[str, Any]:
        result = original_inspect(*args)
        with path.open("r+b") as stream:
            stream.write(b"label\nb\n")
            stream.flush()
            os.fsync(stream.fileno())
        changed = path.stat()
        assert (changed.st_dev, changed.st_ino, changed.st_size) == (
            original.st_dev,
            original.st_ino,
            original.st_size,
        )
        mutations.append(True)
        return result

    monkeypatch.setattr(module.os, "fstat", coarse_stat)
    monkeypatch.setattr(module, "_inspect_file", mutate_same_inode)
    with pytest.raises(ValueError, match=r"input.*hash|input.*stability"):
        module.run_inspection(input_dir=source, output_dir=output, **binding(data))
    assert mutations == [True]
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("name", ["SUMMARY.json", "INSPECTED.json"])
def test_same_inode_output_fsync_mutation_refuses_before_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    data = b"label\na\n"
    module, source, output = setup(monkeypatch, tmp_path, data)
    real_fsync = os.fsync
    mutated = []

    def corrupt_written_file(descriptor: int) -> None:
        path = output / name
        if path.exists() and not mutated:
            actual, named = os.fstat(descriptor), path.stat()
            if (actual.st_dev, actual.st_ino) == (named.st_dev, named.st_ino):
                os.pwrite(descriptor, b"!", 0)
                changed = os.fstat(descriptor)
                assert (changed.st_dev, changed.st_ino, changed.st_size) == (
                    actual.st_dev,
                    actual.st_ino,
                    actual.st_size,
                )
                mutated.append(name)
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", corrupt_written_file)
    with pytest.raises(ValueError, match=r"output.*hash|output.*bytes|output.*integrity"):
        module.run_inspection(input_dir=source, output_dir=output, **binding(data))
    assert mutated == [name]
