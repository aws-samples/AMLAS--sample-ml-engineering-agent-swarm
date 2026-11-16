"""Actual broker metadata gate; strict SDK shapes, no session or code executes.

Primary response identity contract:
https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_GetCodeInterpreter.html
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-resource-management.html
The verified packet/installed-launcher ports are upstream fixtures; this suite
qualifies only the actual broker's returned resource identity boundary.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from botocore.session import get_session
from botocore.validate import validate_parameters
from test_coding_arbiter import binding
from test_paired_coding_runner import population

from amlas.sagemaker_entrypoints import coding_pair as worker

SOURCE = "a" * 40
REGION = "us-west-2"
ACCOUNT = "000000000000"
BUILTIN = "aws.codeinterpreter.v1"
ARN = "arn:aws:bedrock-agentcore:" + REGION + ":aws:code-interpreter/" + BUILTIN


class DispatchReachedError(Exception):
    """Harmless sentinel after resource validation; no session allocation."""


def configured_broker(monkeypatch, tmp_path, resource, *, omitted_identity=None, omit_network_mode=False):
    events = []
    run = {"region": REGION, "account_id": ACCOUNT}
    cases, candidates = population()
    bound = binding()
    packet = json.dumps({"evaluation_binding": {"runtime_sha256": bound.runtime_sha256}}).encode()
    monkeypatch.setenv("AMLAS_CODING_PAIR_SOURCE_COMMIT", SOURCE)
    monkeypatch.setenv("AMLAS_CODING_PAIR_LAUNCHER_SHA256", "b" * 64)
    monkeypatch.setattr(worker.os, "geteuid", lambda: 10001)
    monkeypatch.setattr(worker.os, "getegid", lambda: 10001)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--binding-sha256",
            "c" * 64,
            "--binding-byte-count",
            str(len(packet)),
            "--source-commit",
            SOURCE,
        ],
    )
    monkeypatch.setattr(worker, "verify_oracle_launcher", lambda _expected: "b" * 64)
    monkeypatch.setattr(worker, "_read", lambda *_args, **_kwargs: packet)

    def verified_packet(*_args, **kwargs):
        assert kwargs["source_commit"] == SOURCE
        assert kwargs["runtime_sha256"] == bound.runtime_sha256
        assert kwargs["oracle_launcher_sha256"] == "b" * 64
        events.append("upstream_packet")
        return run, bound, cases, candidates

    monkeypatch.setattr(worker, "load_packet", verified_packet)
    monkeypatch.setattr(worker, "OUTPUT", tmp_path)
    models = get_session()

    class Metadata:
        def get_caller_identity(self):
            events.append("sts")
            return {"Account": ACCOUNT}

        def get_code_interpreter(self, **kwargs):
            model = models.get_service_model("bedrock-agentcore-control")
            operation = model.operation_model("GetCodeInterpreter")
            validate_parameters(kwargs, operation.input_shape)
            validate_parameters(
                {
                    key: value
                    for key, value in resource.items()
                    if key != "ResponseMetadata" and value is not None
                },
                operation.output_shape,
            )
            assert kwargs == {"codeInterpreterId": BUILTIN}
            events.append("get_resource")
            response = copy.deepcopy(resource)
            if omitted_identity is not None:
                response.pop(omitted_identity)
            if omit_network_mode:
                response["networkConfiguration"].pop("networkMode")
            return response

    metadata = Metadata()

    class Session:
        def __init__(self, *, region_name):
            assert region_name == REGION

        def get_partition_for_region(self, region_name):
            assert region_name == REGION
            return models.get_partition_for_region(region_name)

        def client(self, service, **kwargs):
            if service in {"sts", "bedrock-agentcore-control"}:
                return metadata
            assert service == "bedrock-agentcore"
            events.append("data_client")
            return SimpleNamespace()

    import boto3

    monkeypatch.setattr(boto3, "Session", Session)

    def dispatch(**kwargs):
        assert kwargs["binding"] is bound
        assert kwargs["cases"] is cases and kwargs["candidates"] is candidates
        events.append("paired_dispatch")
        raise DispatchReachedError()

    monkeypatch.setattr(worker, "run_paired", dispatch)
    monkeypatch.setattr(worker, "publish_result", lambda *_a, **_k: pytest.fail("control published result"))
    return events


def resource_metadata():
    return {
        "codeInterpreterId": BUILTIN,
        "codeInterpreterArn": ARN,
        "name": "Amazon Bedrock AgentCore Code Interpreter",
        "status": "READY",
        "networkConfiguration": {"networkMode": "SANDBOX"},
        "createdAt": datetime(2026, 10, 19, tzinfo=UTC),
        "lastUpdatedAt": datetime(2026, 10, 19, tzinfo=UTC),
        "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
    }


@pytest.mark.parametrize(
    "change",
    [
        "wrong_id",
        "wrong_arn_id",
        "wrong_region",
        "wrong_partition",
        "wrong_publisher",
        "missing_id",
        "missing_arn",
    ],
)
def test_actual_broker_refuses_unqualified_builtin_identity_before_session_dispatch(
    monkeypatch, tmp_path, change
):
    response = resource_metadata()
    omitted_identity = None
    if change == "wrong_id":
        response["codeInterpreterId"] = "ordinary-custom-resource"
    elif change == "wrong_arn_id":
        response["codeInterpreterArn"] = ARN.replace(BUILTIN, "ordinary-custom-resource")
    elif change == "wrong_region":
        response["codeInterpreterArn"] = ARN.replace(REGION, "us-east-1")
    elif change == "wrong_partition":
        response["codeInterpreterArn"] = ARN.replace("arn:aws:", "arn:aws-us-gov:")
    elif change == "wrong_publisher":
        response["codeInterpreterArn"] = ARN.replace(
            ":aws:code-interpreter/", ":" + ACCOUNT + ":code-interpreter/"
        )
    else:
        # The SDK-shape-valid baseline is validated first. This intentional
        # malformed reply is then delivered to the actual broker identity gate.
        omitted_identity = "codeInterpreterId" if change == "missing_id" else "codeInterpreterArn"
    events = configured_broker(monkeypatch, tmp_path, response, omitted_identity=omitted_identity)
    with pytest.raises(ValueError, match="interpreter"):
        worker.main()
    assert events == ["upstream_packet", "sts", "get_resource"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("configuration", ["omitted", "explicit_sandbox"])
def test_actual_broker_accepts_exact_builtin_identity_without_inventing_optional_configuration(
    monkeypatch, tmp_path, configuration
):
    response = resource_metadata()
    if configuration == "explicit_sandbox":
        response.update(
            executionRoleArn=None,
            filesystemConfigurations=[],
            networkConfiguration={"networkMode": "SANDBOX"},
        )
    events = configured_broker(monkeypatch, tmp_path, response, omit_network_mode=configuration == "omitted")
    with pytest.raises(DispatchReachedError):
        worker.main()
    assert events[:3] == ["upstream_packet", "sts", "get_resource"]
    assert events.count("paired_dispatch") == 1 and events.count("data_client") == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["INVOCATIONS"]
