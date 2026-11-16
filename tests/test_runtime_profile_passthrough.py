"""The deploy script carries the research profile and the guardrail switch into the runtime environment.

A real deployment runs the plain product image through the product's own deploy script. The managed completion path and the
pause witness switch on only with `AMLAS_RUNTIME_RESEARCH_PROFILE=qwen9b_core_v1`, and the deployed configuration
runs with `AMLAS_GUARDRAIL_ENABLED=false`; without the pass-through the runtime gets neither. No AWS call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_acquisition_release_config import writes
from test_runtime_deployment_advisory import run_runtime
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

KEYS = ("AMLAS_RUNTIME_RESEARCH_PROFILE", "AMLAS_GUARDRAIL_ENABLED")


def _environment(workspace):
    return json.loads(Path(workspace[1]["TEST_STATE"]).read_text())["environmentVariables"]


@pytest.mark.parametrize(
    "profile,guardrail",
    [("qwen9b_core_v1", "false"), ("qwen9b_core_v1", "true"), ("off", "false")],
)
def test_runtime_receives_the_research_profile_and_guardrail_switch(runtime_workspace, profile, guardrail):
    runtime_workspace[1].update(AMLAS_RUNTIME_RESEARCH_PROFILE=profile, AMLAS_GUARDRAIL_ENABLED=guardrail)
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    environment = _environment(runtime_workspace)
    assert environment["AMLAS_RUNTIME_RESEARCH_PROFILE"] == profile
    assert environment["AMLAS_GUARDRAIL_ENABLED"] == guardrail


def test_unset_values_stay_absent_from_the_runtime(runtime_workspace):
    for key in KEYS:
        runtime_workspace[1].pop(key, None)
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    environment = _environment(runtime_workspace)
    assert not set(KEYS) & set(environment)


@pytest.mark.parametrize(
    "key,value",
    [
        ("AMLAS_RUNTIME_RESEARCH_PROFILE", "qwen9b_core_v2"),
        ("AMLAS_RUNTIME_RESEARCH_PROFILE", ""),
        ("AMLAS_RUNTIME_RESEARCH_PROFILE", "QWEN9B_CORE_V1"),
        ("AMLAS_GUARDRAIL_ENABLED", "flase"),
        ("AMLAS_GUARDRAIL_ENABLED", ""),
        ("AMLAS_GUARDRAIL_ENABLED", "False "),
    ],
)
def test_unknown_values_fail_before_runtime_writes(runtime_workspace, key, value):
    for name in KEYS:
        runtime_workspace[1].pop(name, None)
    runtime_workspace[1][key] = value
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0
    assert writes(runtime_workspace) == []
