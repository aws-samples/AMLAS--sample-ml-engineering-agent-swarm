"""Native qualifier Source RED: inert SDK-shaped metadata, never IAM/effect proof."""

import copy
from types import SimpleNamespace

import pytest
from botocore.session import Session
from botocore.validate import validate_parameters
from test_campaign_callback import ACCOUNT, RUNTIME
from test_normal_wake_routing import selected_callback_env  # noqa: F401

from amlas.async_callback import lambda_handler as module

ROLE = "arn:aws:iam::" + ACCOUNT + ":role/fixture-callback"
ROLE_ID = "AROA" + "FIXTURE0000000001"
METADATA = {"HTTPStatusCode": 200, "RetryAttempts": 0}


class MetadataPort:
    def __init__(self, service):
        self.model = Session().get_service_model(service)
        self.calls = []
        self.meta = SimpleNamespace(
            region_name="us-west-2",
            config=SimpleNamespace(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10),
        )
        self.identity = {
            "Account": ACCOUNT,
            "Arn": "arn:aws:sts::" + ACCOUNT + ":assumed-role/fixture-callback/fixtureCallbackSession",
            "UserId": ROLE_ID + ":fixtureCallbackSession",
            "ResponseMetadata": copy.deepcopy(METADATA),
        }
        self.endpoint = {
            "agentRuntimeArn": RUNTIME,
            "name": "NORMAL",
            "status": "READY",
            "liveVersion": "3",
            "targetVersion": "3",
            "ResponseMetadata": copy.deepcopy(METADATA),
        }
        self.runtime = {
            "agentRuntimeArn": RUNTIME,
            "agentRuntimeVersion": "3",
            "status": "READY",
            "ResponseMetadata": copy.deepcopy(METADATA),
        }

    def checked(self, operation, kwargs, response):
        validate_parameters(kwargs, self.model.operation_model(operation).input_shape)
        self.calls.append((operation, copy.deepcopy(kwargs)))
        return copy.deepcopy(response)

    def get_caller_identity(self, **kwargs):
        return self.checked("GetCallerIdentity", kwargs, self.identity)

    def get_agent_runtime_endpoint(self, **kwargs):
        assert kwargs == {"agentRuntimeId": RUNTIME.rsplit("/", 1)[1], "endpointName": "NORMAL"}
        return self.checked("GetAgentRuntimeEndpoint", kwargs, self.endpoint)

    def get_agent_runtime(self, **kwargs):
        assert kwargs == {"agentRuntimeId": RUNTIME.rsplit("/", 1)[1], "agentRuntimeVersion": "3"}
        return self.checked("GetAgentRuntime", kwargs, self.runtime)


def fixture_qualifier(monkeypatch):
    cls = getattr(module, "_NativeCallbackRouteQualifier", None)
    assert callable(cls), "missing Source native callback role/current-route qualifier"
    sts, control = MetadataPort("sts"), MetadataPort("bedrock-agentcore-control")
    monkeypatch.setattr(
        module.boto3,
        "client",
        lambda *args, **kwargs: pytest.fail("injected qualifier reached another SDK client"),
    )
    config = module._campaign_config()
    qualifier = cls(
        config,
        expected_role_arn=ROLE,
        expected_role_id=ROLE_ID,
        sts_client=sts,
        control_client=control,
        clock=lambda: 1000.0,
    )
    return qualifier, config, sts, control


def test_current_native_role_and_ready_exact_route_issue_source_observation(monkeypatch):
    qualifier, config, sts, control = fixture_qualifier(monkeypatch)
    observed = qualifier.qualify_current(config, {"kind": "Training"})
    assert type(observed) is module.CallbackRouteObservation
    assert observed.issuer is qualifier and observed.runtime_arn == config.runtime_arn
    assert observed.runtime_version == config.runtime_version == "3"
    assert observed.route_sha256 == config.route_sha256
    assert observed.observed_at == 1000.0 and 1000.0 < observed.expires_at <= 1060.0
    assert sts.calls and {name for name, _args in control.calls} == {
        "GetAgentRuntimeEndpoint",
        "GetAgentRuntime",
    }


def test_role_incarnation_is_rechecked_without_reusing_prior_observation(monkeypatch):
    qualifier, config, sts, control = fixture_qualifier(monkeypatch)
    qualifier.qualify_current(config, {"kind": "Training"})
    before = len(control.calls)
    sts.identity["UserId"] = "AROA" + "OTHER000000000000:fixtureCallbackSession"
    with pytest.raises(module.CallbackQualificationError):
        qualifier.qualify_current(config, {"kind": "Training"})
    assert len(sts.calls) == 2 and len(control.calls) == before


def test_endpoint_live_version_drift_refuses_selected_route(monkeypatch):
    qualifier, config, _sts, control = fixture_qualifier(monkeypatch)
    control.endpoint["liveVersion"] = "4"
    with pytest.raises(module.CallbackQualificationError):
        qualifier.qualify_current(config, {"kind": "Training"})


def test_unqualified_runtime_metadata_refuses_observation(monkeypatch):
    qualifier, config, _sts, control = fixture_qualifier(monkeypatch)
    control.runtime["ResponseMetadata"]["RetryAttempts"] = 1
    with pytest.raises(module.CallbackQualificationError):
        qualifier.qualify_current(config, {"kind": "Training"})
