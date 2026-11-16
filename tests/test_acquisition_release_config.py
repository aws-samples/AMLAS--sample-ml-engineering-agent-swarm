"""Capture immutable acquisition configuration at real shell/runtime boundaries."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_runtime_deployment_advisory import run_runtime, runtime_calls
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import ACCOUNT
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

from amlas.config.externalized import resolve

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "d" * 40
IMAGE = f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/acquisition@sha256:" + "a" * 64


def _input_shape_probe(args):
    if "--generate-cli-skeleton" not in args:
        return False
    index = args.index("--generate-cli-skeleton")
    return index + 1 < len(args) and args[index + 1] == "input"


def writes(workspace):
    return [
        item
        for item in runtime_calls(workspace)
        if item["args"][:2]
        in (
            ["cloudformation", "create-stack"],
            ["iam", "put-role-policy"],
            ["bedrock-agentcore-control", "create-agent-runtime"],
            ["bedrock-agentcore-control", "update-agent-runtime"],
            ["ssm", "put-parameter"],
        )
        and not _input_shape_probe(item["args"])
    ]


def test_acquisition_ssm_discovery_uses_selected_namespace_and_preserves_env_precedence(monkeypatch):
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-green")
    monkeypatch.setenv("AMLAS_SSM_PREFIX", "/fixture-selected")
    monkeypatch.delenv("AMLAS_DATA_ACQUISITION_IMAGE_URI", raising=False)
    monkeypatch.delenv("AMLAS_DATA_ACQUISITION_SOURCE_COMMIT", raising=False)
    calls = []

    def getter(key, region):
        calls.append((key, region))
        return IMAGE if key.endswith("image-uri") else SOURCE

    assert resolve("data_acquisition_image_uri", region="us-east-1", ssm_getter=getter) == IMAGE
    assert resolve("data_acquisition_source_commit", region="us-east-1", ssm_getter=getter) == SOURCE
    assert calls == [
        ("/fixture-selected/data-acquisition-image-uri", "us-east-1"),
        ("/fixture-selected/data-acquisition-source-commit", "us-east-1"),
    ]
    monkeypatch.setenv("AMLAS_DATA_ACQUISITION_IMAGE_URI", IMAGE.replace("a" * 64, "b" * 64))
    assert resolve("data_acquisition_image_uri", region="us-east-1", ssm_getter=getter).endswith("b" * 64)
    assert len(calls) == 2


def test_runtime_json_carries_verified_pair_and_explicit_backend(runtime_workspace):
    runtime_workspace[1].update(
        AMLAS_DATA_ACQUISITION_IMAGE_URI=IMAGE,
        AMLAS_DATA_ACQUISITION_SOURCE_COMMIT=SOURCE,
        DATA_ACQUISITION_BACKEND="processing",
        AMLAS_AGENT_REMOTE_FETCH="on",
        AMLAS_AGENT_HTTP_ALLOW_HOSTS="huggingface.co,cdn.example",
    )
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    environment = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())["environmentVariables"]
    assert environment["AMLAS_DATA_ACQUISITION_IMAGE_URI"] == IMAGE
    assert environment["AMLAS_DATA_ACQUISITION_SOURCE_COMMIT"] == SOURCE
    assert environment["DATA_ACQUISITION_BACKEND"] == "processing"
    assert environment["AMLAS_AGENT_REMOTE_FETCH"] == "on"
    assert environment["AMLAS_AGENT_HTTP_ALLOW_HOSTS"] == "huggingface.co,cdn.example"
    policy = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())["policy"]["Statement"]
    statements = {item["Sid"]: item for item in policy}
    assert statements["SessionVersioningRead"]["Resource"] == "arn:aws:s3:::example-session-bucket"
    assert statements["SessionVersionArtifactRead"]["Resource"] == "arn:aws:s3:::example-session-bucket/*"
    assert f"RESOLVED_SAGEMAKER_ROLE_ARN=arn:aws:iam::{ACCOUNT}:role/amlas-sagemaker-exec" in result.stdout
    assert f"RESOLVED_DATA_ACQUISITION_IMAGE_URI={IMAGE}" in result.stdout
    assert f"RESOLVED_DATA_ACQUISITION_SOURCE_COMMIT={SOURCE}" in result.stdout


@pytest.mark.parametrize(
    "image,source",
    [
        (IMAGE, ""),
        ("", SOURCE),
        (IMAGE.split("@")[0] + ":latest", SOURCE),
        (IMAGE.replace("us-east-1", "us-west-2"), SOURCE),
        (IMAGE.replace(ACCOUNT, "0" * 12), SOURCE),
        (IMAGE, "main"),
    ],
)
def test_invalid_or_partial_pair_fails_before_runtime_writes(runtime_workspace, image, source):
    runtime_workspace[1].update(
        AMLAS_DATA_ACQUISITION_IMAGE_URI=image,
        AMLAS_DATA_ACQUISITION_SOURCE_COMMIT=source,
    )
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0
    assert writes(runtime_workspace) == []


def test_absent_pair_is_unconfigured_and_preserves_runtime_success(runtime_workspace):
    runtime_workspace[1].pop("AMLAS_DATA_ACQUISITION_IMAGE_URI", None)
    runtime_workspace[1].pop("AMLAS_DATA_ACQUISITION_SOURCE_COMMIT", None)
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    environment = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())["environmentVariables"]
    assert "AMLAS_DATA_ACQUISITION_IMAGE_URI" not in environment
    assert "AMLAS_DATA_ACQUISITION_SOURCE_COMMIT" not in environment


@pytest.mark.parametrize("role", ["not-an-arn", "arn:aws:iam::000000000000:role/foreign"])
def test_explicit_invalid_or_foreign_sagemaker_role_is_not_silently_replaced(runtime_workspace, role):
    runtime_workspace[1]["SAGEMAKER_EXECUTION_ROLE_ARN"] = role
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0
    assert writes(runtime_workspace) == []


@pytest.mark.parametrize("mode", ["denied", "foreign", "trust"])
def test_native_sagemaker_role_identity_or_trust_failure_prevents_writes(runtime_workspace, mode):
    runtime_workspace[1]["TEST_SM_ROLE_FAILURE"] = mode
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0
    assert writes(runtime_workspace) == []


def test_native_sagemaker_role_is_read_before_any_write(runtime_workspace):
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = runtime_calls(runtime_workspace)
    selected = next(
        index
        for index, call in enumerate(calls)
        if call["args"][:2] == ["iam", "get-role"] and "amlas-sagemaker-exec" in call["args"]
    )
    first_write = min(calls.index(value) for value in writes(runtime_workspace))
    assert selected < first_write


def publication_fragment():
    source = (ROOT / "scripts/setup.sh").read_text()
    begin = source.index("  # Extract the ARN + final status")
    end = source.index(
        "\nfi\n\n# =============================================================================\n# STEP 11",
        begin,
    )
    return source[begin:end]


@pytest.mark.parametrize("ready", [True, False])
def test_setup_publishes_selected_bindings_only_after_verified_ready(tmp_path, ready):
    role = f"arn:aws:iam::{ACCOUNT}:role/selected-sm"
    out = "\n".join(
        [
            "AGENTCORE_RUNTIME_ARN=fixture-runtime",
            "FINAL_STATUS=" + ("READY" if ready else "FAILED"),
            "AGENTCORE_RUNTIME_DEPLOY_DONE",
            "RESOLVED_SESSION_BUCKET=fixture-bucket",  # pragma: allowlist secret — public lifecycle sentinel and synthetic bucket
            "RESOLVED_SAGEMAKER_ROLE_ARN=" + role,
            "RESOLVED_DATA_ACQUISITION_IMAGE_URI=" + IMAGE,
            "RESOLVED_DATA_ACQUISITION_SOURCE_COMMIT=" + SOURCE,
        ]
    )
    script = """
set -euo pipefail
source scripts/lib/deploy_common.sh
log_info(){ :; }
log_warn(){ :; }
publish_owned_parameter(){ printf 'PARAM=%s=%s\\n' "$1" "$2"; }
AMLAS_AWS_ACCOUNT_ID=$TEST_ACCOUNT
AWS_REGION=us-east-1
AMLAS_SSM_PREFIX=/fixture-selected
AMLAS_AGENTCORE_SSM_PARAM=/fixture-selected/agentcore-runtime-arn
AMLAS_SESSION_BUCKET_SSM_PARAM=/fixture-selected/session-bucket
RUNTIME_DEPLOY_OUT=$TEST_RUNTIME_OUTPUT
""" + publication_fragment()
    import os

    environment = dict(os.environ, TEST_ACCOUNT=ACCOUNT, TEST_RUNTIME_OUTPUT=out)
    result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=environment, capture_output=True, text=True)
    if not ready:
        assert result.returncode != 0
        assert "PARAM=" not in result.stdout
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PARAM=/fixture-selected/sagemaker-execution-role-arn={role}" in result.stdout
    assert f"PARAM=/fixture-selected/data-acquisition-image-uri={IMAGE}" in result.stdout
    assert f"PARAM=/fixture-selected/data-acquisition-source-commit={SOURCE}" in result.stdout


def test_mutation_oracle_keeps_native_requests_and_excludes_local_input_shape(runtime_workspace):
    runtime_workspace[1]["TEST_CREATE"] = "1"
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = runtime_calls(runtime_workspace)
    probes = [
        call
        for call in calls
        if "--generate-cli-skeleton" in call["args"]
        and call["args"][call["args"].index("--generate-cli-skeleton") + 1] == "input"
    ]
    assert probes, "actual helper must exercise local CLI input shape discovery"
    effects = writes(runtime_workspace)
    assert not any(call in effects for call in probes), "local input shape discovery is not a native write"
    for operation in ("create-agent-runtime", "update-agent-runtime"):
        actual = [
            call
            for call in calls
            if call["args"][:2] == ["bedrock-agentcore-control", operation]
            and "--generate-cli-skeleton" not in call["args"]
        ]
        assert actual and all(call in effects for call in actual), (
            "actual native request attempts must remain writes"
        )
