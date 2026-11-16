"""The runtime role reads the ownership tags of its deployment's own training jobs.

`SageMakerQwenSftBackend.status()` serves the fine-tune, reload and both generation jobs. After the request readback it
reads the job's ownership tags with ListTags; the restore path makes the same read. In the first real normal-path run
(`fixtures/sagemaker_training_tag_read_real.json`) IAM refused every one of those reads on the fine-tune job: each
observation of the healthy job was UNKNOWN and the wait closed at its 900 s cap. The same role read the tags of its
case-freeze processing job without a refusal.

These tests replay the recorded calls against the policy the deploy script generates. The matcher covers what these
statements use: Allow statements, action and resource patterns with `*`; no Deny and no Condition applies to them.
"""

from __future__ import annotations

import ast
import json
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
from botocore import xform_name
from botocore.session import Session
from test_runtime_binding_completeness import state
from test_runtime_deployment_advisory import run_runtime
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import ACCOUNT
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

FIXTURE = Path(__file__).parent / "fixtures" / "sagemaker_training_tag_read_real.json"
RECORDED = json.loads(FIXTURE.read_text())
REGION = "us-east-1"  # the Region of the deploy-script workspace
ADAPTERS = Path(__file__).parents[1] / "src" / "amlas" / "backends" / "adapters"
# The SageMaker job operations of the normal-path adapters: module -> (operations, resource type, job name families).
# `coding_generation` inherits `status()` of `qwen_sft`, so the tag read of `qwen_sft` also serves -gen- jobs.
NATIVE_JOB_CALLS = {
    "qwen_sft.py": ({"CreateTrainingJob", "DescribeTrainingJob", "ListTags"}, "training-job", ("sft", "gen")),
    "coding_generation.py": ({"CreateTrainingJob", "DescribeTrainingJob"}, "training-job", ("gen",)),
    "campaign_state.py": ({"DescribeTrainingJob", "ListTags"}, "training-job", ("sft",)),
    "coding_pair.py": (
        {"CreateProcessingJob", "DescribeProcessingJob", "ListTags"},
        "processing-job",
        ("pair",),
    ),
    "coding_case_freeze.py": (
        {"CreateProcessingJob", "DescribeProcessingJob", "ListTags"},
        "processing-job",
        ("case-freeze",),
    ),
    "coding_preparation.py": (
        {"CreateProcessingJob", "DescribeProcessingJob"},
        "processing-job",
        ("data-prep",),
    ),
    "dataset_acquisition.py": (
        {"CreateProcessingJob", "DescribeProcessingJob"},
        "processing-job",
        ("data-acq",),
    ),
    "dataset_inspection.py": (
        {"CreateProcessingJob", "DescribeProcessingJob"},
        "processing-job",
        ("data-inspect",),
    ),
    "model_weight_acquisition.py": (
        {"CreateProcessingJob", "DescribeProcessingJob"},
        "processing-job",
        ("model-acq",),
    ),
}


def _values(value):
    return value if isinstance(value, list) else [value]


def allowed(statements, action, resource):
    return any(
        item["Effect"] == "Allow"
        and "Condition" not in item
        and any(fnmatchcase(action, pattern) for pattern in _values(item["Action"]))
        and any(fnmatchcase(resource, pattern) for pattern in _values(item["Resource"]))
        for item in statements
    )


def job_arn(kind, name):
    return f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:{kind}/{name}"


def generated_statements(workspace, prefix):
    workspace[1]["AMLAS_DEPLOYMENT_PREFIX"] = prefix
    result = run_runtime(workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    return state(workspace)["policy"]["Statement"]


def test_recorded_run_refused_the_training_tag_read_and_nothing_else():
    calls = {(c["action"], c["resource_type"], c["outcome"]): c for c in RECORDED["runtime_role_calls"]}
    assert set(calls) == {
        ("sagemaker:CreateTrainingJob", "training-job", "ok"),
        ("sagemaker:DescribeTrainingJob", "training-job", "ok"),
        ("sagemaker:ListTags", "processing-job", "ok"),
        ("sagemaker:ListTags", "training-job", "AccessDenied"),
    }
    refused = calls["sagemaker:ListTags", "training-job", "AccessDenied"]
    described = calls["sagemaker:DescribeTrainingJob", "training-job", "ok"]
    # Every observation got past the request readback (one Describe) and was then refused its tag read.
    assert refused["count"] == described["count"] == 71
    assert (
        refused["job_name"]
        == described["job_name"]
        == RECORDED["sft_job"]["stored_request"]["TrainingJobName"]
    )
    detail = RECORDED["refused_tag_read"]
    assert detail["cloudtrail_error_code"] == "AccessDenied"
    assert "is not authorized to perform: sagemaker:ListTags on resource: " in detail["message"]
    assert (
        ":training-job/" + refused["job_name"] + " because no identity-based policy allows"
        in detail["message"]
    )
    assert refused["job_name"].startswith(detail["deployment_prefix"] + "-sft-")
    assert RECORDED["held_report"] == {
        "disposition": "submission_unresolved",
        "hold_kind": "input_admission_held",
        "diagnostic": {
            "phase": "native_observation",
            "reason": "original_unknown_budget_exhausted",
            "error_class": "UNKNOWN",
        },
        "wait_log": {"polls": 27, "elapsed_s": 903, "limit_s": 900, "outcome": "closed"},
    }


def test_generated_runtime_policy_allows_every_recorded_call_of_the_run(runtime_workspace):
    prefix = RECORDED["refused_tag_read"]["deployment_prefix"]
    statements = generated_statements(runtime_workspace, prefix)
    refused = [
        (call["action"], call["resource_type"], call["job_name"])
        for call in RECORDED["runtime_role_calls"]
        if not allowed(statements, call["action"], job_arn(call["resource_type"], call["job_name"]))
    ]
    assert refused == []


@pytest.mark.parametrize("prefix", ["amlas", "binding-blue", "amlas-demo"])
def test_generated_runtime_reads_tags_of_owned_training_and_processing_jobs_only(runtime_workspace, prefix):
    statements = generated_statements(runtime_workspace, prefix)
    grants = [item for item in statements if "sagemaker:ListTags" in item["Action"]]
    assert grants == [
        {
            "Sid": "SageMakerProcessingTagRead",
            "Effect": "Allow",
            "Action": ["sagemaker:ListTags"],
            "Resource": [job_arn("processing-job", prefix + "-*")],
        },
        {
            "Sid": "SageMakerTrainingTagRead",
            "Effect": "Allow",
            "Action": ["sagemaker:ListTags"],
            "Resource": [job_arn("training-job", prefix + "-*")],
        },
    ]
    # Training jobs: the fine-tune and the reload are -sft-, base and tuned generation -gen-.
    for family in ("sft", "gen"):
        assert allowed(
            statements, "sagemaker:ListTags", job_arn("training-job", f"{prefix}-{family}-0123abcd")
        )
    # Processing jobs whose readback lists tags: the paired evaluation and the case freeze.
    for family in ("pair", "case-freeze"):
        assert allowed(
            statements, "sagemaker:ListTags", job_arn("processing-job", f"{prefix}-{family}-0123abcd")
        )
    assert not allowed(
        statements, "sagemaker:ListTags", job_arn("training-job", "other-" + prefix + "-sft-0123abcd")
    )
    assert not allowed(statements, "sagemaker:ListTags", job_arn("training-job", prefix + "x-sft-0123abcd"))
    assert not allowed(statements, "sagemaker:ListTags", job_arn("endpoint", prefix + "-sft-0123abcd"))
    # The tag read is the only new grant: every other action on a training job keeps the statements it had.
    training = job_arn("training-job", prefix + "-sft-0123abcd")
    for action in ("sagemaker:DeleteTags", "sagemaker:UpdateTrainingJob", "sagemaker:CreateModel"):
        assert not allowed(statements, action, training)
    tag_write = next(item for item in statements if item["Sid"] == "SageMakerJobTags")
    assert tag_write["Action"] == ["sagemaker:AddTags"]
    assert tag_write["Resource"] == [
        job_arn("training-job", prefix + "-*"),
        job_arn("processing-job", prefix + "-*"),
    ]


def test_native_job_call_inventory_matches_the_adapter_sources():
    """Every SageMaker client operation named in a normal-path adapter is in the inventory, and nothing else."""
    operations = {xform_name(name): name for name in Session().get_service_model("sagemaker").operation_names}
    for module, (expected, _kind, _families) in NATIVE_JOB_CALLS.items():
        tree = ast.parse((ADAPTERS / module).read_text())
        found = {
            operations[node.attr]
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in operations
        }
        assert found == expected, module


@pytest.mark.parametrize("prefix", ["amlas", "amlas-demo"])
def test_generated_runtime_policy_allows_every_native_job_call_of_the_normal_path(runtime_workspace, prefix):
    """No authorization model stood between these calls and the test doubles: the recorded run found the gap."""
    statements = generated_statements(runtime_workspace, prefix)
    refused = []
    for module, (operations, kind, families) in sorted(NATIVE_JOB_CALLS.items()):
        for family in families:
            resource = job_arn(kind, f"{prefix}-{family}-0123abcd")
            # A Create call carries the ownership tags, which IAM authorizes as AddTags on the new job.
            actions = {"sagemaker:" + name for name in operations}
            actions |= (
                {"sagemaker:AddTags"} if any(name.startswith("Create") for name in operations) else set()
            )
            refused += [
                (module, action, resource)
                for action in sorted(actions)
                if not allowed(statements, action, resource)
            ]
    assert refused == []
