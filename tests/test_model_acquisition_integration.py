"""Normal model acquisition route, pure startup and hidden SDK state, no AWS."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import model_acquisition_fixtures as f
import pytest
from test_acquisition_release_config import writes
from test_runtime_deployment_advisory import run_runtime, runtime_calls
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import ACCOUNT
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

from amlas.backends import ports
from amlas.backends.registry import AdapterUnavailableError, BackendRegistry
from amlas.config import externalized
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.tui.chat import tool_binding_backend


def selectors(env: Any) -> Any:
    function = getattr(externalized, "model_weight_acquisition_selectors", None)
    assert callable(function), "missing model startup selector contract"
    return function(env=env)


def configured() -> dict[str, str]:
    return {
        "AMLAS_AWS_ACCOUNT_ID": f.ACCOUNT,
        "AMLAS_AWS_REGION": f.REGION,
        "AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI": f.IMAGE,
        "AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT": f.COMMIT,
        "AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST": "a" * 64,
    }


def test_registry_has_fixed_managed_only_port_without_startup_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("startup SDK"))
    port = getattr(ports, "ModelWeightAcquisitionBackend", None)
    assert port in ports.ALL_PORTS
    assert ports.port_key(port) == "MODEL_WEIGHT_ACQUISITION_BACKEND"
    resolved = BackendRegistry().resolve(port)
    assert resolved.adapter_name == "processing" and isinstance(resolved.adapter, port)
    monkeypatch.setenv("MODEL_WEIGHT_ACQUISITION_BACKEND", "local")
    with pytest.raises(AdapterUnavailableError):
        BackendRegistry().resolve(port)


def test_selector_absence_is_pure_unconfigured_and_complete_is_unadmitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("startup SDK"))
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *a, **k: pytest.fail("startup SSM"))
    assert selectors({})["state"] == "unconfigured"
    value = selectors(configured())
    assert value["state"] == "unadmitted" and value["native_admitted"] is False
    assert value["source_commit"] == f.COMMIT


@pytest.mark.parametrize(
    "change",
    [
        {"AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT": ""},
        {"AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST": "c" * 64},
        {"AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI": "caller:latest"},
        {"AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT": "main"},
        {"AMLAS_AWS_ACCOUNT_ID": "9876" * 3},
        {"AMLAS_AWS_REGION": "us-east-1"},
    ],
)
def test_partial_mutable_malformed_or_foreign_tuple_refuses(change: dict[str, str]) -> None:
    env = configured()
    env.update(change)
    with pytest.raises(ValueError):
        selectors(env)


def test_model_configuration_keys_have_selected_namespace_ssm_and_no_defaults():
    for suffix in ("image_uri", "source_commit", "admission_digest"):
        key = externalized.CONFIG_KEYS.get("model_weight_acquisition_" + suffix)
        assert key is not None
        assert key.env == ("AMLAS_MODEL_WEIGHT_ACQUISITION_" + suffix.upper(),)
        assert key.ssm == "/amlas/model-weight-acquisition-" + suffix.replace("_", "-")
        assert key.default == ""


def test_catalog_is_pure_fresh_source_description_without_grants(monkeypatch: pytest.MonkeyPatch):
    from test_model_weight_acquisition_binding import modules

    from amlas.tools import model_weight_tools

    modules()
    function = getattr(model_weight_tools, "list_model_weight_catalog", None)
    assert function is not None, "missing pure model catalog tool"
    public = function.tool_spec["inputSchema"]["json"]["properties"]
    assert public == {}
    first = function()
    assert first[0]["catalog_id"] == f.CATALOG and first[0]["catalog_version"] == f.VERSION
    assert len(first[0]["revision"]) == 40 and len(first[0]["files"]) == 9
    assert first[0]["license"] == "apache-2.0" and first[0]["architectures"]
    assert first[0]["acquired"] is False and first[0]["ready"] is False
    first[0]["files"].clear()
    assert len(function()[0]["files"]) == 9


def test_normal_constructor_owns_unique_private_model_store_and_fresh_policy(monkeypatch):
    monkeypatch.setattr(tool_binding_backend.ToolBindingBackend, "is_tool_capable", property(lambda _: True))
    states = []

    def agent(_prompt, *, invocation_state):
        states.append(invocation_state)
        return SimpleNamespace(message={"content": [{"text": "done"}]})

    backend = tool_binding_backend.ToolBindingBackend(
        Settings(), approval_policy=ApprovalPolicy(mode=ApprovalMode.AUTO)
    )
    monkeypatch.setattr(backend, "build_agent", lambda **_: agent)
    backend.chat_with_tools("acquire selected model")
    backend.approval_policy = None
    backend.chat_with_tools("poll selected model")
    store = states[0].get("_amlas_model_acquisition_store")
    assert isinstance(store, SessionModelAcquisitionStore)
    assert store is states[1]["_amlas_model_acquisition_store"] and not hasattr(store, "policy")
    assert store.runtime_session_id == backend._acquisition_store.runtime_session_id
    assert states[1]["_amlas_approval_policy"].mode is ApprovalMode.ASK
    other = tool_binding_backend.ToolBindingBackend(Settings())
    assert other._model_acquisition_store.runtime_session_id != store.runtime_session_id


def test_model_tools_route_to_collection_and_ml_without_exposing_authority():
    from amlas.subagents.declarations import DATA_COLLECTION_AGENT, TRAINER_TUNER
    from amlas.tools import model_weight_tools
    from amlas.tui.chat.agent_tools import READ_ONLY_TOOLS, get_safe_llm_tools, tool_display_name

    expected = {"list_model_weight_catalog", "acquire_model_weights", "get_model_weight_acquisition"}
    assert expected <= {tool_display_name(t) for t in get_safe_llm_tools()}
    for declaration in (DATA_COLLECTION_AGENT, TRAINER_TUNER):
        assert expected <= {tool_display_name(t) for t in declaration.tools}
    assert {"list_model_weight_catalog", "get_model_weight_acquisition"} <= READ_ONLY_TOOLS
    assert "acquire_model_weights" not in READ_ONLY_TOOLS
    for tool in (model_weight_tools.acquire_model_weights, model_weight_tools.get_model_weight_acquisition):
        properties = tool.tool_spec["inputSchema"]["json"]["properties"]
        assert not set(properties) & {
            "policy",
            "store",
            "role_arn",
            "image_uri",
            "tool_context",
            "native_handle",
            "source_commit",
        }


_MODEL_RELEASE_KEYS = (
    "AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI",
    "AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT",
    "AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST",
)


def _model_release_tuple():
    return dict(
        zip(
            _MODEL_RELEASE_KEYS,
            (
                f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "b" * 64,
                "c" * 40,
                "b" * 64,
            ),
            strict=True,
        )
    )


def _model_parameter_tuple():
    values = _model_release_tuple()
    return {
        "/fixture-model/model-weight-acquisition-" + suffix: values[key]
        for suffix, key in zip(
            ("image-uri", "source-commit", "admission-digest"), _MODEL_RELEASE_KEYS, strict=True
        )
    }


def _model_parameter_reads(workspace):
    return [
        call
        for call in runtime_calls(workspace)
        if call["args"][:2] == ["ssm", "get-parameter"]
        and "/model-weight-acquisition-" in call["args"][call["args"].index("--name") + 1]
    ]


def test_model_ssm_only_tuple_reaches_actual_governed_runtime_without_env(runtime_workspace):
    env = runtime_workspace[1]
    for key in _MODEL_RELEASE_KEYS:
        env.pop(key, None)
    env.update(
        AMLAS_SSM_PREFIX="/fixture-model", TEST_MODEL_RELEASE_VALUES=json.dumps(_model_parameter_tuple())
    )
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    names = [
        call["args"][call["args"].index("--name") + 1] for call in _model_parameter_reads(runtime_workspace)
    ]
    assert names == list(_model_parameter_tuple()), (
        "registered model SSM tuple was not resolved by the governed helper"
    )
    native = json.loads(Path(env["TEST_STATE"]).read_text())["environmentVariables"]
    assert all(native.get(key) == value for key, value in _model_release_tuple().items())
    state = selectors(native)
    assert state["state"] == "unadmitted" and state["native_admitted"] is False


def test_model_complete_env_preserves_governed_transport_without_ssm(runtime_workspace):
    env = runtime_workspace[1]
    env.update(
        _model_release_tuple(),
        AMLAS_SSM_PREFIX="/fixture-model",
        TEST_MODEL_RELEASE_VALUES=json.dumps(_model_parameter_tuple()),
    )
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _model_parameter_reads(runtime_workspace) == []
    native = json.loads(Path(env["TEST_STATE"]).read_text())["environmentVariables"]
    assert all(native.get(key) == value for key, value in _model_release_tuple().items())
    assert selectors(native)["native_admitted"] is False


@pytest.mark.parametrize(
    "mode",
    [
        "partial_env",
        "partial_ssm",
        "foreign_name",
        "denied",
        "mutable_image",
        "marker_mismatch",
        "foreign_account",
        "foreign_region",
        "bad_source",
    ],
)
def test_model_release_partial_foreign_or_unverified_refuses_before_writes(runtime_workspace, mode):
    env = runtime_workspace[1]
    for key in _MODEL_RELEASE_KEYS:
        env.pop(key, None)
    parameters = _model_parameter_tuple()
    env["AMLAS_SSM_PREFIX"] = "/fixture-model"
    if mode == "partial_env":
        env[_MODEL_RELEASE_KEYS[0]] = _model_release_tuple()[_MODEL_RELEASE_KEYS[0]]
    elif mode == "partial_ssm":
        parameters.pop("/fixture-model/model-weight-acquisition-source-commit")
    elif mode in ("foreign_name", "denied"):
        env["TEST_MODEL_RELEASE_FAILURE"] = "foreign" if mode == "foreign_name" else mode
    else:
        selected = _model_release_tuple()
        if mode == "mutable_image":
            selected[_MODEL_RELEASE_KEYS[0]] = selected[_MODEL_RELEASE_KEYS[0]].split("@")[0] + ":latest"
        elif mode == "marker_mismatch":
            selected[_MODEL_RELEASE_KEYS[2]] = "f" * 64
        elif mode == "foreign_account":
            selected[_MODEL_RELEASE_KEYS[0]] = selected[_MODEL_RELEASE_KEYS[0]].replace(ACCOUNT, "9" * 12)
        elif mode == "foreign_region":
            selected[_MODEL_RELEASE_KEYS[0]] = selected[_MODEL_RELEASE_KEYS[0]].replace(
                "us-east-1", "us-west-2"
            )
        else:
            selected[_MODEL_RELEASE_KEYS[1]] = "main"
        env.update(selected)
    env["TEST_MODEL_RELEASE_VALUES"] = json.dumps(parameters)
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0, "invalid model selector snapshot reached deployment writes"
    assert writes(runtime_workspace) == []
    if mode == "partial_env":
        assert _model_parameter_reads(runtime_workspace) == [], (
            "partial ENV must never be filled from old SSM"
        )


def test_model_release_all_absent_stays_unconfigured_without_grant(runtime_workspace):
    env = runtime_workspace[1]
    for key in _MODEL_RELEASE_KEYS:
        env.pop(key, None)
    env.update(AMLAS_SSM_PREFIX="/fixture-model", TEST_MODEL_RELEASE_VALUES="{}")
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(Path(env["TEST_STATE"]).read_text())["environmentVariables"]
    assert not set(_MODEL_RELEASE_KEYS) & native.keys()
    assert selectors(native) == {"state": "unconfigured", "native_admitted": False}


@pytest.mark.parametrize(
    "mode",
    ["denied_mentions_missing", "duplicate_name", "duplicate_value", "nonfinite_json", "overflow_float_json"],
)
def test_model_release_denied_or_duplicate_provider_metadata_refuses_before_writes(runtime_workspace, mode):
    env = runtime_workspace[1]
    for key in _MODEL_RELEASE_KEYS:
        env.pop(key, None)
    env.update(
        AMLAS_SSM_PREFIX="/fixture-model",
        TEST_MODEL_RELEASE_VALUES=json.dumps(_model_parameter_tuple()),
        TEST_MODEL_RELEASE_FAILURE=mode,
    )
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0, "unverified provider metadata reached deployment writes"
    assert writes(runtime_workspace) == []


@pytest.mark.parametrize(
    "mode", ["all_empty", "all_whitespace", "one_empty_two_missing", "one_whitespace_two_missing"]
)
def test_model_release_successful_blank_ssm_never_becomes_missing(runtime_workspace, mode):
    env = runtime_workspace[1]
    for key in _MODEL_RELEASE_KEYS:
        env.pop(key, None)
    parameters = _model_parameter_tuple()
    if mode == "all_empty":
        parameters = {name: "" for name in parameters}
    elif mode == "all_whitespace":
        parameters = {name: "   " for name in parameters}
    elif mode == "one_empty_two_missing":
        parameters = {"/fixture-model/model-weight-acquisition-image-uri": ""}
    else:
        parameters = {"/fixture-model/model-weight-acquisition-source-commit": "   "}
    env.update(AMLAS_SSM_PREFIX="/fixture-model", TEST_MODEL_RELEASE_VALUES=json.dumps(parameters))
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0, (
        "successful blank SSM values were treated as genuinely missing configuration"
    )
    assert writes(runtime_workspace) == []


def test_model_release_three_genuine_notfound_preserves_env_empty_convention(runtime_workspace):
    env = runtime_workspace[1]
    env.update({key: "   " for key in _MODEL_RELEASE_KEYS})
    env.update(AMLAS_SSM_PREFIX="/fixture-model", TEST_MODEL_RELEASE_VALUES="{}")
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    names = [
        call["args"][call["args"].index("--name") + 1] for call in _model_parameter_reads(runtime_workspace)
    ]
    assert names == list(_model_parameter_tuple())
    native = json.loads(Path(env["TEST_STATE"]).read_text())["environmentVariables"]
    assert selectors(native) == {"state": "unconfigured", "native_admitted": False}


def test_model_release_captured_native_missing_cli_framing_stays_unconfigured(runtime_workspace):
    raw = b"\naws: [ERROR]: An error occurred (ParameterNotFound) when calling the GetParameter operation:\n"
    assert len(raw) == 94
    assert (
        hashlib.sha256(raw).hexdigest()
        == "c182af4dc38fc7b8d344a84864fd7db180d21b31fe41233941a42c247ad61250"  # pragma: allowlist secret — public94B CLI error checksum
    )
    env = runtime_workspace[1]
    for key in _MODEL_RELEASE_KEYS:
        env.pop(key, None)
    env.update(
        AMLAS_SSM_PREFIX="/fixture-model",
        TEST_MODEL_RELEASE_FAILURE="captured_missing_cli",
        TEST_MODEL_RELEASE_VALUES="{}",
    )
    probe = subprocess.run(
        [
            str(Path(env["PATH"].split(":")[0]) / "aws"),
            "ssm",
            "get-parameter",
            "--name",
            "/fixture-model/model-weight-acquisition-image-uri",
            "--region",
            "us-east-1",
            "--output",
            "json",
        ],
        env=env,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 254 and probe.stdout == b"" and probe.stderr == raw
    assert hashlib.sha256(probe.stderr).hexdigest() == hashlib.sha256(raw).hexdigest()
    Path(runtime_workspace[2]).write_text("")
    result = run_runtime(runtime_workspace)
    assert "SyntaxError" not in result.stderr
    assert result.returncode == 0, "captured genuine missing-parameter framing was refused"
    names = [
        call["args"][call["args"].index("--name") + 1] for call in _model_parameter_reads(runtime_workspace)
    ]
    assert names == list(_model_parameter_tuple())
    native = json.loads(Path(env["TEST_STATE"]).read_text())["environmentVariables"]
    assert selectors(native) == {"state": "unconfigured", "native_admitted": False}
