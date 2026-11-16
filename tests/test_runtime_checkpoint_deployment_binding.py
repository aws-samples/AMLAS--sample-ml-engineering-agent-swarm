"""Real deployment-script checkpoint bindings over the existing closed AWS CLI stub.

These regressions inspect the accepted runtime request and generated IAM policy.
They make no native calls and establish no runtime/checkpoint admission.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import pytest
from test_runtime_deployment_advisory import run_runtime, runtime_calls
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import ACCOUNT
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

# The existing closed GetRole fixture returns this native RoleId on both paths.
NATIVE_ROLE_ID = "AROA" + "X" * 16


def _runtime_state(workspace: tuple[Path, dict[str, str], Path]) -> dict:
    return json.loads(Path(workspace[1]["TEST_STATE"]).read_text())


@pytest.mark.parametrize("reuse", [False, True], ids=["new-role", "reused-role"])
def test_verified_native_runtime_role_reaches_checkpoint_environment(runtime_workspace, reuse):
    env = runtime_workspace[1]
    trainer_role = f"arn:aws:iam::{ACCOUNT}:role/separate-sagemaker-exec"
    env.update(
        SAGEMAKER_EXECUTION_ROLE_ARN=trainer_role,
        AMLAS_RUNTIME_EXEC_ROLE_ARN=f"arn:aws:iam::{ACCOUNT}:role/stale-runtime-selector",
        AMLAS_RUNTIME_EXEC_ROLE_ID="AROA" + "Z" * 16,
    )
    before = None
    if reuse:
        first = run_runtime(runtime_workspace)
        assert first.returncode == 0, first.stdout + first.stderr
        before = _runtime_state(runtime_workspace)
        env["TEST_EXISTING_ROLE"] = "captured"
    start = len(runtime_calls(runtime_workspace))

    result = run_runtime(runtime_workspace)

    assert result.returncode == 0, result.stdout + result.stderr
    accepted = _runtime_state(runtime_workspace)
    actual = accepted["environmentVariables"]
    calls = runtime_calls(runtime_workspace)[start:]
    runtime_role = f"arn:aws:iam::{ACCOUNT}:role/amlas-runtime-exec-us-east-1"
    assert accepted["roleArn"] == runtime_role
    assert actual.get("AMLAS_RUNTIME_EXEC_ROLE_ARN") == runtime_role, (
        "Checkpoint runtime scope must use the verified native execution role, "
        "not stale input or the SageMaker job role"
    )
    assert actual.get("AMLAS_RUNTIME_EXEC_ROLE_ID") == NATIVE_ROLE_ID, (
        "Checkpoint runtime scope must retain the native GetRole incarnation "
        "on both new and reused role paths"
    )
    assert actual["SAGEMAKER_EXECUTION_ROLE_ARN"] == trainer_role
    assert actual["AMLAS_RUNTIME_EXEC_ROLE_ARN"] != trainer_role
    assert any(
        call["args"][:2] == ["iam", "get-role"]
        and call["args"][call["args"].index("--role-name") + 1] == "amlas-runtime-exec-us-east-1"
        for call in calls
    ), "The accepted binding must follow actual script GetRole verification"
    if before is not None:
        assert accepted["trust"] == before["trust"]
        assert accepted["policy"] == before["policy"]
        assert not any(
            call["args"][:2]
            in (
                ["iam", "put-role-policy"],
                ["iam", "update-assume-role-policy"],
                ["cloudformation", "create-stack"],
                ["cloudformation", "update-stack"],
            )
            for call in calls
        ), "Reusing a compatible role must not alter its grants, trust or lifecycle"


@pytest.mark.parametrize("prefix", ["amlas", "checkpoint-binding"])
def test_checkpoint_describe_table_grant_is_exact_selected_table(runtime_workspace, prefix):
    runtime_workspace[1].update(
        AMLAS_DEPLOYMENT_PREFIX=prefix,
        AMLAS_SSM_PREFIX=f"/{prefix}",
    )

    result = run_runtime(runtime_workspace)

    assert result.returncode == 0, result.stdout + result.stderr
    accepted = _runtime_state(runtime_workspace)
    actual = accepted["environmentVariables"]
    table = actual["AMLAS_JOBS_TABLE"]
    assert table == f"{prefix}-training-jobs"
    expected = f"arn:aws:dynamodb:us-east-1:{ACCOUNT}:table/{table}"
    grants = []
    for statement in accepted["policy"]["Statement"]:
        if statement.get("Effect") != "Allow":
            continue
        actions = statement.get("Action", [])
        actions = [actions] if isinstance(actions, str) else actions
        if any(fnmatch.fnmatchcase("dynamodb:describetable", action.lower()) for action in actions):
            grants.append(statement)
    assert grants, "Normal checkpoint table identity read requires DescribeTable permission"
    for statement in grants:
        resources = statement.get("Resource", [])
        resources = [resources] if isinstance(resources, str) else resources
        assert set(resources) == {expected}, (
            "Checkpoint metadata permission must target only the actually selected table; "
            "no wildcard, other namespace, account, Region or index scope"
        )
