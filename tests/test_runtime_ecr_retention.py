"""Actual CDK synthesis protects tagged runtime/worker images from count cleanup."""

from __future__ import annotations

import json

import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Template

from infra.cdk_stacks.agentcore_runtime_stack import AgentCoreRuntimeStack


@pytest.fixture(params=("amlas", "retention-a"))
def repository(request, tmp_path):
    app = App(outdir=str(tmp_path / "assembly"))
    stack = AgentCoreRuntimeStack(
        app, "Retention", hyphen_project=request.param,
        env=Environment(account="111122223333", region="us-west-2"),
    )
    template = Template.from_stack(stack).to_json()
    resources = [resource for resource in template["Resources"].values()
                 if resource["Type"] == "AWS::ECR::Repository"]
    assert len(resources) == 1
    return request.param, resources[0]


def test_count_cleanup_never_selects_tagged_digest_pinned_runtime_or_worker_images(repository):
    _prefix, resource = repository
    policy = json.loads(resource["Properties"]["LifecyclePolicy"]["LifecyclePolicyText"])
    assert len(policy["rules"]) == 1
    rule = policy["rules"][0]
    assert rule["selection"]["tagStatus"] == "untagged", (
        "ANY count cleanup expires tagged images even when consumers pin their digest"
    )
    assert rule["selection"] == {
        "tagStatus": "untagged", "countType": "imageCountMoreThan", "countNumber": 10,
    }
    assert rule["rulePriority"] == 1 and rule["action"] == {"type": "expire"}


def test_runtime_repository_settings_and_ten_image_cleanup_are_preserved(repository):
    prefix, resource = repository
    properties = resource["Properties"]
    assert properties["RepositoryName"] == prefix + "-runtime"
    assert properties["ImageScanningConfiguration"] == {"ScanOnPush": True}
    assert properties["EmptyOnDelete"] is True
    assert resource["DeletionPolicy"] == resource["UpdateReplacePolicy"] == "Delete"
    rule = json.loads(properties["LifecyclePolicy"]["LifecyclePolicyText"])["rules"][0]
    assert rule["rulePriority"] == 1
    assert rule["selection"]["countType"] == "imageCountMoreThan"
    assert rule["selection"]["countNumber"] == 10
    assert rule["action"] == {"type": "expire"}
