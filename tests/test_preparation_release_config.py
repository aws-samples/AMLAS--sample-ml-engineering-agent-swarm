"""Exact operator selector tuples and normal runtime overlay; no AWS calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_acquisition_release_config import writes
from test_preparation_integration import configuration_env
from test_runtime_deployment_advisory import run_runtime
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

ROOT = Path(__file__).parents[1]
KEYS = {key: value for key, value in configuration_env().items() if key.startswith("AMLAS_DATA_PREPARATION_")}
TRIPLE = {
    key: KEYS[key]
    for key in (
        "AMLAS_DATA_PREPARATION_IMAGE_URI",
        "AMLAS_DATA_PREPARATION_SOURCE_COMMIT",
        "AMLAS_DATA_PREPARATION_ADMISSION_DIGEST",
    )
}
EXCLUSION_PARAMETERS = [
    "/fixture-selected/data-preparation-exclusions-" + suffix
    for suffix in ("uri", "version-id", "sha256", "byte-count")
]


def operator(tmp_path, *, env_values=None, ssm_values=None, failure="", foreign=False):
    executable = tmp_path / "aws"
    executable.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json,os,sys
args=sys.argv[1:]
assert args[:2]==['ssm','get-parameter']
name=args[args.index('--name')+1]
assert name.startswith('/fixture-selected/data-preparation-')
assert args[args.index('--region')+1]=='us-west-2'
with open(os.environ['TEST_CALLS'],'a') as stream: stream.write(json.dumps(args)+'\n')
if os.environ['TEST_FAILURE']:
 print('AccessDeniedException',file=sys.stderr);raise SystemExit(254)
values=json.loads(os.environ['TEST_VALUES'])
if name not in values:
 print('ParameterNotFound',file=sys.stderr);raise SystemExit(254)
print(json.dumps({'Parameter':{'Name':'/foreign/unrelated' if os.environ['TEST_FOREIGN'] else name,'Value':values[name]}}))
"""
    )
    executable.chmod(0o700)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AWS_", "AMLAS_", "AGENTICML_"))
    }
    env.update(env_values or {})
    env.update(
        PATH=str(tmp_path) + ":" + env.get("PATH", ""),
        TEST_CALLS=str(tmp_path / "calls.jsonl"),
        TEST_VALUES=json.dumps(ssm_values or {}),
        TEST_FAILURE=failure,
        TEST_FOREIGN="yes" if foreign else "",
    )
    script = """set -euo pipefail
source scripts/lib/deploy_common.sh
resolve_preparation_release_config FIXTURE_ACCOUNT us-west-2 /fixture-selected
printf 'WRITE-REACHED\\n'
for name in AMLAS_DATA_PREPARATION_IMAGE_URI AMLAS_DATA_PREPARATION_ADMISSION_DIGEST AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID; do
 printf '%s=%s\\n' "$name" "${!name:-}"
done
""".replace("FIXTURE_ACCOUNT", "1" * 12)
    result = subprocess.run(
        ["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30
    )
    calls = (
        [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        if (tmp_path / "calls.jsonl").exists()
        else []
    )
    return result, calls


def ssm_tuple():
    return {
        "/fixture-selected/" + key.removeprefix("AMLAS_").lower().replace("_", "-"): value
        for key, value in KEYS.items()
    }


def test_operator_ssm_only_resolves_exact_complete_tuple(tmp_path):
    result, calls = operator(tmp_path, ssm_values=ssm_tuple())
    assert result.returncode == 0, result.stderr
    assert len(calls) == 7
    assert (
        "AMLAS_DATA_PREPARATION_IMAGE_URI=" + KEYS["AMLAS_DATA_PREPARATION_IMAGE_URI"]
        in result.stdout  # pragma: allowlist secret
    )
    expected_version = (
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID=exclusions-v1"  # pragma: allowlist secret
    )
    assert expected_version in result.stdout


def test_operator_complete_env_never_reads_ssm(tmp_path):
    result, calls = operator(tmp_path, env_values=KEYS, ssm_values={})
    assert result.returncode == 0, result.stderr
    assert calls == []


def test_operator_absent_tuple_stays_unconfigured(tmp_path):
    result, calls = operator(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 7 and "AMLAS_DATA_PREPARATION_IMAGE_URI=\n" in result.stdout


@pytest.mark.parametrize(
    "mode",
    [
        "partial_env",
        "partial_ssm",
        "stale_admission",
        "mutable_version",
        "foreign",
        "denied",
        "exclusions_without_triple",
        "partial_exclusions_env",
        "partial_exclusions_ssm",
        "partial_triple_with_exclusions",
        "triple_with_stale_ssm_exclusions",
        "triple_with_unreadable_ssm_exclusions",
    ],
)
def test_operator_refuses_partial_foreign_and_unverified_before_writes(tmp_path, mode):
    env = KEYS if mode in {"stale_admission", "mutable_version"} else {}
    env = dict(env)
    ssm = ssm_tuple()
    if mode == "partial_env":
        env = {"AMLAS_DATA_PREPARATION_IMAGE_URI": KEYS["AMLAS_DATA_PREPARATION_IMAGE_URI"]}
    if mode == "partial_ssm":
        ssm.pop("/fixture-selected/data-preparation-source-commit")
    if mode == "exclusions_without_triple":
        env = {key: value for key, value in KEYS.items() if key not in TRIPLE}
    if mode == "partial_exclusions_env":
        env = dict(KEYS)
        env.pop("AMLAS_DATA_PREPARATION_EXCLUSIONS_SHA256")
    if mode == "partial_exclusions_ssm":
        ssm.pop("/fixture-selected/data-preparation-exclusions-byte-count")
    if mode == "partial_triple_with_exclusions":
        env = dict(KEYS)
        env.pop("AMLAS_DATA_PREPARATION_SOURCE_COMMIT")
    if mode == "triple_with_stale_ssm_exclusions":
        # A triple-only publication would leave these beside the new triple, and a
        # later SSM-only resolution would combine selections never chosen together.
        env = dict(TRIPLE)
        ssm = {name: value for name, value in ssm.items() if "-exclusions-version-id" in name}
    if mode == "triple_with_unreadable_ssm_exclusions":
        env = dict(TRIPLE)
    if mode == "stale_admission":
        env["AMLAS_DATA_PREPARATION_ADMISSION_DIGEST"] = "f" * 64
    if mode == "mutable_version":
        env["AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID"] = "null"
    result, _ = operator(
        tmp_path,
        env_values=env,
        ssm_values=ssm,
        foreign=mode == "foreign",
        failure="read" if mode in {"denied", "triple_with_unreadable_ssm_exclusions"} else "",
    )
    assert result.returncode != 0
    assert "WRITE-REACHED" not in result.stdout


def test_normal_runtime_overlay_transports_complete_selected_tuple(runtime_workspace):
    selected = dict(KEYS)
    selected["AMLAS_DATA_PREPARATION_IMAGE_URI"] = (
        selected["AMLAS_DATA_PREPARATION_IMAGE_URI"]
        .replace("1" * 12, "111122223333")
        .replace("us-west-2", "us-east-1")
    )
    runtime_workspace[1].update(selected, DATA_PREPARATION_BACKEND="processing")
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())["environmentVariables"]
    assert all(native.get(key) == value for key, value in selected.items())
    assert native["DATA_PREPARATION_BACKEND"] == "processing"


def test_partial_runtime_config_refuses_before_any_write(runtime_workspace):
    runtime_workspace[1].update(AMLAS_DATA_PREPARATION_IMAGE_URI=KEYS["AMLAS_DATA_PREPARATION_IMAGE_URI"])
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0
    assert writes(runtime_workspace) == []


def test_operator_triple_only_env_selects_resources_without_global_exclusions(tmp_path):
    """The product prepares coding data with the triple alone when a frozen case supplies exclusions."""
    result, calls = operator(tmp_path, env_values=TRIPLE, ssm_values={})
    assert result.returncode == 0, result.stderr
    assert "WRITE-REACHED" in result.stdout
    assert "AMLAS_DATA_PREPARATION_IMAGE_URI=" + TRIPLE["AMLAS_DATA_PREPARATION_IMAGE_URI"] in result.stdout
    assert "AMLAS_DATA_PREPARATION_ADMISSION_DIGEST=" + TRIPLE["AMLAS_DATA_PREPARATION_ADMISSION_DIGEST"] in (
        result.stdout  # pragma: allowlist secret
    )
    assert "AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID=\n" in result.stdout
    # Only the stale-exclusions guard reads SSM; explicit selectors are never mixed with SSM values.
    assert [call[call.index("--name") + 1] for call in calls] == EXCLUSION_PARAMETERS


def test_operator_ssm_triple_only_resolves_without_exclusions(tmp_path):
    ssm = {name: value for name, value in ssm_tuple().items() if "-exclusions-" not in name}
    result, calls = operator(tmp_path, ssm_values=ssm)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 7
    assert "AMLAS_DATA_PREPARATION_IMAGE_URI=" + TRIPLE["AMLAS_DATA_PREPARATION_IMAGE_URI"] in result.stdout
    assert "AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID=\n" in result.stdout


def test_normal_runtime_overlay_transports_selected_triple_only(runtime_workspace):
    selected = {
        key: value.replace("1" * 12, "111122223333").replace("us-west-2", "us-east-1")
        for key, value in TRIPLE.items()
    }
    runtime_workspace[1].update(selected, DATA_PREPARATION_BACKEND="processing")
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())["environmentVariables"]
    assert all(native.get(key) == value for key, value in selected.items())
    assert not any("EXCLUSIONS" in key for key in native)
    assert (
        "RESOLVED_DATA_PREPARATION_IMAGE_URI=" + selected["AMLAS_DATA_PREPARATION_IMAGE_URI"] in result.stdout
    )
    assert "RESOLVED_DATA_PREPARATION_EXCLUSIONS_URI=\n" in result.stdout


def publisher(tmp_path, env_values):
    executable = tmp_path / "aws"
    executable.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json,os,sys
args=sys.argv[1:]
with open(os.environ['TEST_CALLS'],'a') as stream: stream.write(json.dumps(args)+'\n')
if args[:2]==['ssm','describe-parameters']:
 print('0');raise SystemExit(0)
if args[:2]==['ssm','put-parameter']:
 assert args[args.index('--value')+1], 'SSM refuses empty parameter values'
 raise SystemExit(0)
print('UNEXPECTED-AWS-CALL',file=sys.stderr);raise SystemExit(9)
"""
    )
    executable.chmod(0o700)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AWS_", "AMLAS_", "AGENTICML_"))
    }
    env.update(env_values)
    env.update(
        PATH=str(tmp_path) + ":" + env.get("PATH", ""),
        TEST_CALLS=str(tmp_path / "calls.jsonl"),
        AWS_REGION="us-west-2",
        AMLAS_AWS_ACCOUNT_ID="1" * 12,
        AMLAS_SSM_PREFIX="/fixture-selected",
        AMLAS_DEPLOYMENT_PREFIX="fixture",
        AMLAS_DEPLOYMENT_ID="abcdefab-cdef-4abc-8def-abcdefabcdef",
    )
    script = "set -euo pipefail\nsource scripts/lib/deploy_common.sh\npublish_preparation_selection\nprintf 'PUBLISHED\\n'\n"
    result = subprocess.run(  # nosec B603 B607 - fixed test script; the fake aws is first on PATH
        ["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30
    )
    calls = (
        [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        if (tmp_path / "calls.jsonl").exists()
        else []
    )
    return result, [call[call.index("--name") + 1] for call in calls if call[:2] == ["ssm", "put-parameter"]]


@pytest.mark.parametrize("selection", ["none", "triple", "all"])
def test_setup_publishes_exactly_the_selected_preparation_values(tmp_path, selection):
    values = {"none": {}, "triple": TRIPLE, "all": KEYS}[selection]
    result, published = publisher(tmp_path, values)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PUBLISHED" in result.stdout
    expected = ["/fixture-selected/" + key.removeprefix("AMLAS_").lower().replace("_", "-") for key in values]
    assert published == expected


@pytest.mark.parametrize("mode", ["partial_triple", "exclusions_without_image", "partial_exclusions"])
def test_setup_publishes_nothing_for_a_partial_selection(tmp_path, mode):
    values = dict(KEYS)
    if mode == "partial_triple":
        values.pop("AMLAS_DATA_PREPARATION_ADMISSION_DIGEST")
    if mode == "exclusions_without_image":
        values.pop("AMLAS_DATA_PREPARATION_IMAGE_URI")
    if mode == "partial_exclusions":
        values.pop("AMLAS_DATA_PREPARATION_EXCLUSIONS_URI")
    result, published = publisher(tmp_path, values)
    assert result.returncode != 0
    assert "PUBLISHED" not in result.stdout
    assert published == []
