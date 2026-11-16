"""Actual CDK callback routes; source-only, no deployment or event authority."""

from __future__ import annotations

import pytest
from test_infra_deployment_namespace import assemblies  # noqa: F401 - owning shared synth fixture


@pytest.mark.parametrize("stack_name", ["AsyncTrainingStack", "RuntimeStack"])
@pytest.mark.parametrize("prefix", ["amlas", "trial-a", "trial-b", "a" * 24])
def test_processing_terminal_events_use_existing_callback_with_deployment_prefix(
    assemblies,  # noqa: F811 - shared owning pytest fixture
    stack_name,
    prefix,
):
    resources = assemblies[prefix][1][stack_name]["Resources"]
    rules = {
        name: resource["Properties"]
        for name, resource in resources.items()
        if resource["Type"] == "AWS::Events::Rule"
    }
    training = [
        rule
        for rule in rules.values()
        if rule["EventPattern"].get("detail-type") == ["SageMaker Training Job State Change"]
    ]
    assert len(training) == 1
    # The training rule carries the same job-name prefix filter as the processing rule, so one deployment's
    # callback never receives another deployment's training events in the same account and Region.
    assert training[0]["EventPattern"] == {
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Training Job State Change"],
        "detail": {
            "TrainingJobStatus": ["Completed", "Failed", "Stopped"],
            "TrainingJobName": [{"prefix": prefix + "-"}],
        },
    }
    processing = [
        (name, rule)
        for name, rule in rules.items()
        if rule["EventPattern"].get("detail-type") == ["SageMaker Processing Job State Change"]
    ]
    assert len(processing) == 1, "missing terminal Processing callback rule"
    logical, rule = processing[0]
    assert rule["EventPattern"] == {
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Processing Job State Change"],
        "detail": {
            "ProcessingJobStatus": ["Completed", "Failed", "Stopped"],
            "ProcessingJobName": [{"prefix": prefix + "-"}],
        },
    }
    assert rule["Name"] == prefix + "-sagemaker-processing-state-change"
    assert rule["Targets"] == training[0]["Targets"]
    callback = next(
        resource["Properties"]
        for resource in resources.values()
        if resource["Type"] == "AWS::Lambda::Function"
        and resource["Properties"].get("Handler") == "amlas.async_callback.lambda_handler.handler"
    )
    assert set(callback["Environment"]["Variables"]) == {"AGENTCORE_RUNTIME_ARN", "JOBS_TABLE"}
    permissions = [
        resource["Properties"]
        for resource in resources.values()
        if resource["Type"] == "AWS::Lambda::Permission"
        and resource["Properties"].get("SourceArn") == {"Fn::GetAtt": [logical, "Arn"]}
    ]
    assert len(permissions) == 1
    assert permissions[0]["Principal"] == "events.amazonaws.com"
    assert permissions[0]["Action"] == "lambda:InvokeFunction"
    assert permissions[0]["FunctionName"] == rule["Targets"][0]["Arn"]
