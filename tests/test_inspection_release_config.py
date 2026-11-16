"""Inspection release selectors transport admission evidence, never invent it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_acquisition_release_config import publication_fragment, writes
from test_runtime_deployment_advisory import run_runtime
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import ACCOUNT, calls, invoke
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "c" * 64
IMAGE = f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/inspection@sha256:" + DIGEST
SOURCE = "b" * 40


@pytest.mark.parametrize("marker", ["", DIGEST])
def test_runtime_preserves_inspection_pair_and_only_supplied_admission(runtime_workspace, marker):
    env = runtime_workspace[1]
    env.update(
        AMLAS_DATA_INSPECTION_IMAGE_URI=IMAGE,
        AMLAS_DATA_INSPECTION_SOURCE_COMMIT=SOURCE,
    )
    if marker:
        env["AMLAS_DATA_INSPECTION_ADMISSION_DIGEST"] = marker
    else:
        env.pop("AMLAS_DATA_INSPECTION_ADMISSION_DIGEST", None)
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(Path(env["TEST_STATE"]).read_text())["environmentVariables"]
    assert native["AMLAS_DATA_INSPECTION_IMAGE_URI"] == IMAGE
    assert native["AMLAS_DATA_INSPECTION_SOURCE_COMMIT"] == SOURCE
    if marker:
        assert native["AMLAS_DATA_INSPECTION_ADMISSION_DIGEST"] == DIGEST
    else:
        assert "AMLAS_DATA_INSPECTION_ADMISSION_DIGEST" not in native
    assert f"RESOLVED_DATA_INSPECTION_IMAGE_URI={IMAGE}" in result.stdout
    assert f"RESOLVED_DATA_INSPECTION_SOURCE_COMMIT={SOURCE}" in result.stdout
    assert f"RESOLVED_DATA_INSPECTION_ADMISSION_DIGEST={marker}" in result.stdout


@pytest.mark.parametrize("image,source,marker", [
    (IMAGE, "", ""), ("", SOURCE, ""), ("", "", DIGEST),
    (IMAGE.replace("us-east-1", "us-west-2"), SOURCE, ""),
    (IMAGE.replace(ACCOUNT, "0" * 12), SOURCE, ""),
    (IMAGE.split("@")[0] + ":latest", SOURCE, ""),
    (IMAGE, "main", ""), (IMAGE, SOURCE, "d" * 64),
    (IMAGE, SOURCE, "sha256:" + DIGEST),
])
def test_invalid_inspection_release_fails_before_writes(runtime_workspace, image, source, marker):
    runtime_workspace[1].update(
        AMLAS_DATA_INSPECTION_IMAGE_URI=image,
        AMLAS_DATA_INSPECTION_SOURCE_COMMIT=source,
        AMLAS_DATA_INSPECTION_ADMISSION_DIGEST=marker,
    )
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0
    assert writes(runtime_workspace) == []


def test_absent_inspection_release_stays_unconfigured(runtime_workspace):
    env = runtime_workspace[1]
    for key in ("AMLAS_DATA_INSPECTION_IMAGE_URI", "AMLAS_DATA_INSPECTION_SOURCE_COMMIT",
                "AMLAS_DATA_INSPECTION_ADMISSION_DIGEST"):
        env.pop(key, None)
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(Path(env["TEST_STATE"]).read_text())["environmentVariables"]
    assert not any(key.startswith("AMLAS_DATA_INSPECTION_") for key in native)


@pytest.mark.parametrize("ready,marker", [(True, ""), (True, DIGEST), (False, DIGEST)])
def test_setup_owned_publication_requires_ready_and_does_not_synthesize_marker(ready, marker):
    role = f"arn:aws:iam::{ACCOUNT}:role/selected-sm"
    output = "\n".join([
        "AGENTCORE_RUNTIME_ARN=fixture-runtime",
        "FINAL_STATUS=" + ("READY" if ready else "FAILED"),
        "AGENTCORE_RUNTIME_DEPLOY_DONE",
        "RESOLVED_SESSION_BUCKET=fixture-bucket",  # pragma: allowlist secret — public field and synthetic bucket
        "RESOLVED_SAGEMAKER_ROLE_ARN=" + role,
        "RESOLVED_DATA_INSPECTION_IMAGE_URI=" + IMAGE,
        "RESOLVED_DATA_INSPECTION_SOURCE_COMMIT=" + SOURCE,
        "RESOLVED_DATA_INSPECTION_ADMISSION_DIGEST=" + marker,
    ])
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
    result = subprocess.run(
        ["bash", "-c", script], cwd=ROOT,
        env=dict(os.environ, TEST_ACCOUNT=ACCOUNT, TEST_RUNTIME_OUTPUT=output),
        capture_output=True, text=True,
    )
    if not ready or not marker:
        assert result.returncode != 0
        assert "PARAM=" not in result.stdout
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PARAM=/fixture-selected/data-inspection-image-uri={IMAGE}" in result.stdout
    assert f"PARAM=/fixture-selected/data-inspection-source-commit={SOURCE}" in result.stdout
    if marker:
        assert f"PARAM=/fixture-selected/data-inspection-admission-digest={marker}" in result.stdout
    else:
        assert "PARAM=/fixture-selected/data-inspection-admission-digest=" not in result.stdout


def test_setup_rejects_unadmitted_inspection_before_setup_writes(shell_workspace):
    env = shell_workspace[1]
    env.update(AMLAS_DATA_INSPECTION_IMAGE_URI=IMAGE, AMLAS_DATA_INSPECTION_SOURCE_COMMIT=SOURCE)
    env.pop("AMLAS_DATA_INSPECTION_ADMISSION_DIGEST", None)
    result = invoke(shell_workspace, "setup.sh", "--yes", "--skip-deps", "--agent-runtime-only")
    assert result.returncode != 0
    assert "Inspection discovery publication requires a supplied matching native admission marker" in (
        result.stdout + result.stderr
    )
    assert all(call["args"][:2] in (["sts", "get-caller-identity"], ["configure", "get"])
               or call["args"][:1] == ["--version"] for call in calls(shell_workspace)
               if call["tool"] == "aws")


@pytest.mark.parametrize("mode", ["--preflight-only", "--synth-only"])
def test_read_only_setup_keeps_unadmitted_inspection_bootstrap_configuration(shell_workspace, mode):
    shell_workspace[1].update(
        AMLAS_DATA_INSPECTION_IMAGE_URI=IMAGE, AMLAS_DATA_INSPECTION_SOURCE_COMMIT=SOURCE,
    )
    shell_workspace[1].pop("AMLAS_DATA_INSPECTION_ADMISSION_DIGEST", None)
    result = invoke(shell_workspace, "setup.sh", "--yes", "--skip-deps", mode)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("marker,owner", [("", "owned"), ("", "foreign"), (DIGEST, "owned"), (DIGEST, "foreign")])
def test_setup_actual_owned_publisher_never_keeps_stale_admission_for_new_pair(tmp_path, marker, owner):
    """Use native SSM CLI shapes with the real validator and owned publisher.

    Shape references: AWS CLI ssm describe-parameters, list-tags-for-resource,
    put-parameter. Existing updates must use --overwrite and preserve tags;
    no delete or tagging/adoption operation is part of this test contract.
    """
    prefix = "/fixture-selected"
    deployment_id = "333af729-c6bd-4283-95a4-fe5139e24276"
    role = f"arn:aws:iam::{ACCOUNT}:role/selected-sm"
    old_digest = "a" * 64
    old = {
        prefix + "/agentcore-runtime-arn": "previous-runtime",
        prefix + "/sagemaker-execution-role-arn": role,
        prefix + "/session-bucket": "previous-bucket",
        prefix + "/data-inspection-image-uri": IMAGE.replace(DIGEST, old_digest),
        prefix + "/data-inspection-source-commit": "a" * 40,
        prefix + "/data-inspection-admission-digest": old_digest,
    }
    state = tmp_path / "ssm-state.json"
    state.write_text(json.dumps(old))
    call_log = tmp_path / "ssm-calls.jsonl"
    fake = tmp_path / "aws"
    fake.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
state_path = Path(os.environ["TEST_SSM_STATE"])
state = json.loads(state_path.read_text())
with open(os.environ["TEST_SSM_CALLS"], "a") as stream:
    stream.write(json.dumps(args) + "\n")
def value(flag):
    assert args.count(flag) == 1, "native flag must occur once"
    return args[args.index(flag) + 1]
assert value("--region") == "us-east-1"
if args[:2] == ["ssm", "describe-parameters"]:
    target = value("--parameter-filters")
    assert target.startswith("Key=Name,Option=Equals,Values=/fixture-selected/")
    name = target.split("Values=", 1)[1]
    assert value("--query") == "length(Parameters)" and value("--output") == "text"
    print("1" if name in state else "0")
elif args[:2] == ["ssm", "list-tags-for-resource"]:
    assert value("--resource-type") == "Parameter" and value("--output") == "json"
    assert value("--resource-id") in state
    owner = os.environ["AMLAS_DEPLOYMENT_ID"] if os.environ["TEST_SSM_OWNER"] == "owned" else "foreign"
    print(json.dumps({"TagList": [
        {"Key": "amlas:deployment-id", "Value": owner},
        {"Key": "amlas:deployment-prefix", "Value": os.environ["AMLAS_DEPLOYMENT_PREFIX"]},
    ]}))
elif args[:2] == ["ssm", "put-parameter"]:
    name = value("--name")
    assert name in state and "--overwrite" in args and "--tags" not in args
    assert value("--type") == "String" and os.environ["TEST_SSM_OWNER"] == "owned"
    state[name] = value("--value")
    state_path.write_text(json.dumps(state))
    print(json.dumps({"Version": 2, "Tier": "Standard"}))
else:
    raise SystemExit("unexpected native operation; delete/adoption forbidden")
''')
    fake.chmod(0o700)
    output = "\n".join([
        "AGENTCORE_RUNTIME_ARN=next-runtime", "FINAL_STATUS=READY",
        "AGENTCORE_RUNTIME_DEPLOY_DONE",  # pragma: allowlist secret — public lifecycle sentinel
        "RESOLVED_SESSION_BUCKET=next-bucket", "RESOLVED_SAGEMAKER_ROLE_ARN=" + role,
        "RESOLVED_DATA_INSPECTION_IMAGE_URI=" + IMAGE,
        "RESOLVED_DATA_INSPECTION_SOURCE_COMMIT=" + SOURCE,
        "RESOLVED_DATA_INSPECTION_ADMISSION_DIGEST=" + marker,
    ])
    script = """
set -euo pipefail
source scripts/lib/deploy_common.sh
log_info(){ :; }
log_warn(){ :; }
AMLAS_AWS_ACCOUNT_ID=$TEST_ACCOUNT
AWS_REGION=us-east-1
AMLAS_SSM_PREFIX=/fixture-selected
AMLAS_AGENTCORE_SSM_PARAM=/fixture-selected/agentcore-runtime-arn
AMLAS_SESSION_BUCKET_SSM_PARAM=/fixture-selected/session-bucket
RUNTIME_DEPLOY_OUT=$TEST_RUNTIME_OUTPUT
""" + publication_fragment()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("AWS_", "AMLAS_", "AGENTICML_"))}
    env.update(
        PATH=str(tmp_path) + ":" + env.get("PATH", ""), TEST_ACCOUNT=ACCOUNT,
        TEST_RUNTIME_OUTPUT=output, TEST_SSM_STATE=str(state), TEST_SSM_CALLS=str(call_log),
        TEST_SSM_OWNER=owner, AMLAS_DEPLOYMENT_ID=deployment_id,
        AMLAS_DEPLOYMENT_PREFIX="fixture-selected",
    )
    result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    observed = json.loads(state.read_text())
    native_calls = [json.loads(line) for line in call_log.read_text().splitlines()] if call_log.exists() else []
    mutations = [call for call in native_calls if call[:2] == ["ssm", "put-parameter"]]
    if not marker or owner == "foreign":
        assert result.returncode != 0, {"published": observed, "native_calls": native_calls}
        assert mutations == []
        assert observed == old
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert observed[prefix + "/data-inspection-image-uri"] == IMAGE
        assert observed[prefix + "/data-inspection-source-commit"] == SOURCE
        assert observed[prefix + "/data-inspection-admission-digest"] == DIGEST
        assert len(mutations) == 6
