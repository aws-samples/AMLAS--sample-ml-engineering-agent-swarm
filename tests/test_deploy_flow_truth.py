"""Deployment truth through the real flow, tools and managed adapters.

Strict installed-SDK request/response fixtures perform no network, model load or
native mutation. A successful monitor identity control is separate from the
flow's currently missing baseline/role/output configuration.
"""

from __future__ import annotations

import copy
import io
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import botocore.session
import pytest
from botocore.exceptions import ClientError, ParamValidationError
from botocore.handlers import generate_idempotent_uuid
from botocore.validate import validate_parameters

from amlas.backends.adapters.deployment import SageMakerDeploymentBackend
from amlas.backends.adapters.monitoring import SageMakerModelMonitorBackend
from amlas.backends.ports import DeploymentBackend, MonitoringBackend
from amlas.backends.registry import REGISTRY, ResolvedBackend
from amlas.backends.types import ModelRef, ScheduleRef
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.flows._util import raw
from amlas.flows.deploy_flow import run_deploy_flow
from amlas.tools.mlops.monitor import setup_model_monitor

REGION = "eu-west-1"
ROLE = "arn:aws:iam::000000000000:role/test-deployment"
ENDPOINT = "truth-endpoint"
MODEL = ModelRef(uri="s3://model-bucket/model.tar.gz", backend="sagemaker", algo="xgboost")


class StrictSageMaker:
    """SDK-model-validated recorder retaining the actual created identities."""

    def __init__(self, *, endpoint_status: str | None = "InService", monitor_failure: bool = False) -> None:
        self.meta = SimpleNamespace(region_name=REGION)
        self.model = botocore.session.get_session().get_service_model("sagemaker")
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.endpoint_status = endpoint_status
        self.monitor_failure = monitor_failure
        self.models: set[str] = set()
        self.configs: dict[str, dict[str, Any]] = {}
        self.endpoints: dict[str, str] = {}

    def _record(self, operation: str, kwargs: dict[str, Any]) -> None:
        model = self.model.operation_model(operation)
        generate_idempotent_uuid(kwargs, model)
        validate_parameters(kwargs, model.input_shape)
        self.requests.append((operation, copy.deepcopy(kwargs)))

    def _response(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        validate_parameters(payload, self.model.operation_model(operation).output_shape)
        return payload

    def calls(self, operation: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.requests if name == operation]

    def create_model(self, **kwargs: Any) -> dict[str, Any]:
        self._record("CreateModel", kwargs)
        assert kwargs["ModelName"] not in self.models, "fixture must reject duplicate native model names"
        self.models.add(kwargs["ModelName"])
        return self._response("CreateModel", {"ModelArn": self.model_arn(kwargs["ModelName"])})

    @staticmethod
    def model_arn(name: str) -> str:
        return f"arn:aws:sagemaker:{REGION}:000000000000:model/{name}"

    def create_endpoint_config(self, **kwargs: Any) -> dict[str, Any]:
        self._record("CreateEndpointConfig", kwargs)
        assert kwargs["EndpointConfigName"] not in self.configs
        assert all(variant["ModelName"] in self.models for variant in kwargs["ProductionVariants"])
        self.configs[kwargs["EndpointConfigName"]] = copy.deepcopy(kwargs)
        return self._response(
            "CreateEndpointConfig",
            {
                "EndpointConfigArn": f"arn:aws:sagemaker:{REGION}:000000000000:endpoint-config/{kwargs['EndpointConfigName']}",
            },
        )

    def create_endpoint(self, **kwargs: Any) -> dict[str, Any]:
        self._record("CreateEndpoint", kwargs)
        assert kwargs["EndpointName"] not in self.endpoints
        assert kwargs["EndpointConfigName"] in self.configs
        self.endpoints[kwargs["EndpointName"]] = kwargs["EndpointConfigName"]
        return self._response("CreateEndpoint", {"EndpointArn": self.endpoint_arn(kwargs["EndpointName"])})

    @staticmethod
    def endpoint_arn(name: str) -> str:
        return f"arn:aws:sagemaker:{REGION}:000000000000:endpoint/{name}"

    def describe_endpoint(self, **kwargs: Any) -> dict[str, Any]:
        self._record("DescribeEndpoint", kwargs)
        name = kwargs["EndpointName"]
        assert name in self.endpoints
        if self.endpoint_status is None:
            raise TimeoutError("neutral fixture describe outcome unavailable")
        payload = {
            "EndpointName": name,
            "EndpointArn": self.endpoint_arn(name),
            "EndpointConfigName": self.endpoints[name],
            "EndpointStatus": self.endpoint_status,
            "CreationTime": datetime(2026, 1, 1, tzinfo=UTC),
            "LastModifiedTime": datetime(2026, 1, 1, tzinfo=UTC),
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "CurrentWeight": 1.0,
                    "DesiredWeight": 1.0,
                    "CurrentInstanceCount": 1,
                    "DesiredInstanceCount": 1,
                }
            ],
        }
        if self.endpoint_status == "Failed":
            payload["FailureReason"] = "neutral fixture endpoint startup failed"
        return self._response("DescribeEndpoint", payload)

    def delete_endpoint(self, **kwargs: Any) -> dict[str, Any]:
        self._record("DeleteEndpoint", kwargs)
        assert kwargs["EndpointName"] in self.endpoints
        return self._response("DeleteEndpoint", {})

    def create_monitoring_schedule(self, **kwargs: Any) -> dict[str, Any]:
        self._record("CreateMonitoringSchedule", kwargs)
        definition = kwargs["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        inputs = definition["BaselineConfig"]
        output = definition["MonitoringOutputConfig"]["MonitoringOutputs"][0]["S3Output"]["S3Uri"]
        configured = (
            inputs["ConstraintsResource"]["S3Uri"].startswith("s3://baseline-bucket/")
            and inputs["StatisticsResource"]["S3Uri"].startswith("s3://baseline-bucket/")
            and output.startswith("s3://monitor-output-bucket/")
            and definition["RoleArn"] == ROLE
        )
        if self.monitor_failure or not configured:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "neutral fixture monitor configuration refused",
                    },
                    "ResponseMetadata": {"HTTPStatusCode": 400},
                },
                "CreateMonitoringSchedule",
            )
        return self._response(
            "CreateMonitoringSchedule",
            {
                "MonitoringScheduleArn": (
                    f"arn:aws:sagemaker:{REGION}:000000000000:monitoring-schedule/{kwargs['MonitoringScheduleName']}"
                ),
            },
        )


class StrictRuntime:
    """No native invocation: model-shaped input and an ordinary streaming reply."""

    def __init__(self, sm: StrictSageMaker) -> None:
        self.sm = sm
        self.model = botocore.session.get_session().get_service_model("sagemaker-runtime")
        self.requests: list[dict[str, Any]] = []

    def invoke_endpoint(self, **kwargs: Any) -> dict[str, Any]:
        operation = self.model.operation_model("InvokeEndpoint")
        validate_parameters(kwargs, operation.input_shape)
        assert kwargs["EndpointName"] in self.sm.endpoints
        self.requests.append(copy.deepcopy(kwargs))
        payload = {"Body": b"0.75", "ContentType": "text/csv"}
        validate_parameters(payload, operation.output_shape)
        return {**payload, "Body": io.BytesIO(payload["Body"])}


@pytest.fixture(autouse=True)
def no_native_clients(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Every adapter must use its explicit neutral transport, never credentials."""
    import boto3

    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-credentials"))

    def forbidden_client(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("native clients are forbidden in deployment truth fixtures")

    monkeypatch.setattr(boto3, "client", forbidden_client)
    monkeypatch.setattr(boto3.session.Session, "client", forbidden_client)


def bind_managed_adapters(
    monkeypatch: pytest.MonkeyPatch,
    sm: StrictSageMaker,
    monitor: Any = None,
    *,
    role_arn: str | None = ROLE,
):
    runtime = StrictRuntime(sm)
    constructor = {"region": REGION, "sagemaker_client": sm, "runtime_client": runtime}
    if role_arn is not None:
        constructor["role_arn"] = role_arn
    deployment = SageMakerDeploymentBackend(**constructor)
    managed_monitor = monitor or SageMakerModelMonitorBackend(region=REGION, sm_client=sm)

    def resolve(port: Any, **_kwargs: Any) -> ResolvedBackend:
        if port is DeploymentBackend:
            return ResolvedBackend(deployment, "deployment", "sagemaker", "explicit neutral transport")
        if port is MonitoringBackend:
            return ResolvedBackend(managed_monitor, "monitoring", "sagemaker", "explicit neutral transport")
        raise AssertionError(f"unexpected backend resolution: {port}")

    monkeypatch.setattr(REGISTRY, "resolve", resolve)
    return deployment, runtime


def run_flow(*, wire_monitoring: bool = False, smoke: bool = False):
    return run_deploy_flow(
        model=MODEL,
        resources={"endpoint_name": ENDPOINT},
        smoke_payload="1.0" if smoke else None,
        wire_monitoring=wire_monitoring,
        policy=ApprovalPolicy(mode=ApprovalMode.AUTO),
    )


def test_native_recorder_rejects_invalid_model_request_before_recording() -> None:
    sm = StrictSageMaker()
    with pytest.raises(ParamValidationError):
        sm.create_model(ModelName=123)
    assert sm.requests == []


def test_actual_adapter_packages_one_original_native_model(monkeypatch: pytest.MonkeyPatch) -> None:
    sm = StrictSageMaker()
    deployment, _runtime = bind_managed_adapters(monkeypatch, sm)
    package = deployment.package(model_ref=MODEL)
    assert len(sm.calls("CreateModel")) == 1
    original = sm.calls("CreateModel")[0]
    assert package.group == original["ModelName"]
    assert package.arn == sm.model_arn(original["ModelName"])
    assert original["PrimaryContainer"]["ModelDataUrl"] == MODEL.uri


def test_flow_consumes_its_one_original_package_without_model_name_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sm = StrictSageMaker()
    bind_managed_adapters(monkeypatch, sm)
    result = run_flow()
    assert result.status == "deployed", result.detail
    models = sm.calls("CreateModel")
    assert len(models) == 1, "flow packaged twice instead of consuming its original native model"
    original = models[0]["ModelName"]
    variant = sm.calls("CreateEndpointConfig")[0]["ProductionVariants"][0]
    assert result.package["group"] == original == variant["ModelName"]
    assert result.package["package_arn"] == sm.model_arn(original)
    assert len(sm.calls("CreateEndpoint")) == 1


@pytest.mark.parametrize(
    "canonical,alias,explicit,expected",
    [
        pytest.param(ROLE, None, None, ROLE, id="canonical-only"),
        pytest.param(ROLE, ROLE + "-alias", None, ROLE, id="canonical-over-alias"),
        pytest.param(ROLE + "-canonical", ROLE + "-alias", ROLE, ROLE, id="explicit-over-env"),
        pytest.param(None, ROLE, None, ROLE, id="alias-only"),
        pytest.param("  " + ROLE + "  ", None, None, ROLE, id="canonical-trim"),
        pytest.param("<SAGEMAKER_EXECUTION_ROLE_ARN>", ROLE, None, ROLE, id="placeholder-to-alias"),
    ],
)
def test_approved_deploy_flow_binds_selected_role_to_original_model_and_hosting(
    monkeypatch: pytest.MonkeyPatch,
    canonical: str | None,
    alias: str | None,
    explicit: str | None,
    expected: str,
) -> None:
    """Canonical deployment selection reaches one native-shaped model/host flow."""
    from amlas.config import externalized

    for name in ("SAGEMAKER_EXECUTION_ROLE_ARN", "AMLAS_SAGEMAKER_ROLE_ARN"):
        monkeypatch.delenv(name, raising=False)
    if canonical is not None:
        monkeypatch.setenv("SAGEMAKER_EXECUTION_ROLE_ARN", canonical)
    if alias is not None:
        monkeypatch.setenv("AMLAS_SAGEMAKER_ROLE_ARN", alias)

    def no_ssm(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("serving role ENV selection must not fetch SSM configuration")

    monkeypatch.setattr(externalized, "_default_ssm_getter", no_ssm)
    sm = StrictSageMaker()
    deployment, runtime = bind_managed_adapters(monkeypatch, sm, role_arn=explicit)
    assert sm.requests == runtime.requests == [], "adapter construction made an SDK call"
    result = run_flow(smoke=True)
    assert result.status == "deployed", result.detail
    models = sm.calls("CreateModel")
    assert len(models) == 1, "approved flow must consume its one original model"
    original = models[0]
    assert original["ExecutionRoleArn"] == expected
    assert deployment._package_configuration()[1] == expected
    assert original["PrimaryContainer"]["ModelDataUrl"] == MODEL.uri
    assert result.package["group"] == original["ModelName"]
    assert result.package["package_arn"] == sm.model_arn(original["ModelName"])
    configurations = sm.calls("CreateEndpointConfig")
    endpoints = sm.calls("CreateEndpoint")
    assert len(configurations) == len(endpoints) == 1
    assert configurations[0]["ProductionVariants"][0]["ModelName"] == original["ModelName"]
    assert endpoints[0]["EndpointConfigName"] == configurations[0]["EndpointConfigName"]
    assert result.endpoint_name == endpoints[0]["EndpointName"] == ENDPOINT
    assert sm.endpoints[ENDPOINT] == configurations[0]["EndpointConfigName"]
    assert len(runtime.requests) == 1 and runtime.requests[0]["EndpointName"] == ENDPOINT
    assert not sm.calls("DeleteEndpoint")


@pytest.mark.parametrize("state", ["Creating", "Failed"])
def test_nonready_endpoint_never_claims_service_deployment_smoke_or_release(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    sm = StrictSageMaker(endpoint_status=state)
    _deployment, runtime = bind_managed_adapters(monkeypatch, sm)
    result = run_flow(smoke=True)
    assert result.detail.get("deploy", {}).get("status") != "InService", (
        "native nonready status was fabricated"
    )
    assert result.status != "deployed", "nonready native endpoint was released as deployed"
    assert not result.release, "release metadata must wait for confirmed InService"
    assert not runtime.requests, "nonready endpoint must not receive a smoke invocation"
    assert len(sm.calls("CreateEndpoint")) == len(sm.calls("CreateEndpointConfig")) == 1


def test_unknown_describe_preserves_original_endpoint_without_create_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sm = StrictSageMaker(endpoint_status=None)
    _deployment, runtime = bind_managed_adapters(monkeypatch, sm)
    result = run_flow(smoke=True)
    assert result.status != "deployed"
    assert result.detail.get("deploy", {}).get("status") != "InService"
    assert not result.release and not runtime.requests
    assert len(sm.calls("CreateEndpoint")) == len(sm.calls("CreateEndpointConfig")) == 1
    assert sm.calls("DescribeEndpoint")[0]["EndpointName"] == sm.calls("CreateEndpoint")[0]["EndpointName"]


def test_confirmed_inservice_identity_reaches_smoke_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    sm = StrictSageMaker()
    _deployment, runtime = bind_managed_adapters(monkeypatch, sm)
    result = run_flow(smoke=True)
    assert result.status == "deployed"
    assert result.detail["deploy"]["status"] == "InService"
    assert result.endpoint_name == ENDPOINT
    assert result.detail["deploy"]["arn"] == sm.endpoint_arn(ENDPOINT)
    assert result.prediction == "0.75"
    assert len(runtime.requests) == 1
    assert runtime.requests[0]["EndpointName"] == result.release["endpoint"] == ENDPOINT
    assert len(sm.calls("CreateEndpoint")) == len(sm.calls("CreateEndpointConfig")) == 1


def test_failed_native_monitor_is_degraded_in_actual_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    sm = StrictSageMaker(monitor_failure=True)
    monitor = SageMakerModelMonitorBackend(region=REGION, sm_client=sm)
    ref = monitor.setup(endpoint=ENDPOINT, baseline_uri="", monitor_types=["data_quality"], schedule="hourly")
    assert ref.arn == ""
    assert ref.name == f"sm-monitor-{ENDPOINT}"
    assert len(sm.calls("CreateMonitoringSchedule")) == 1


def test_flow_never_claims_no_arn_degraded_monitor_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    sm = StrictSageMaker(monitor_failure=True)
    bind_managed_adapters(monkeypatch, sm)
    result = run_flow(wire_monitoring=True)
    assert result.status == "deployed", result.detail
    assert len(sm.calls("CreateMonitoringSchedule")) == 1
    assert result.monitoring.get("wired") is not True, (
        "degraded no-ARN schedule cannot establish monitoring wiring"
    )


def test_explicit_native_monitor_identity_is_preserved_by_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    sm = StrictSageMaker()
    created = raw(setup_model_monitor)(
        ENDPOINT,
        "s3://baseline-bucket/baseline/",
        "hourly",
        role_arn=ROLE,
        output_path="s3://monitor-output-bucket/reports/",
        region=REGION,
        sm_client=sm,
    )
    assert created.get("success") is not False, created
    ref = ScheduleRef(
        name=created["schedule_name"],
        arn=created["monitor_schedule_arn"],
        monitor_type=created["monitor_type"],
        schedule=created["schedule"],
    )
    assert ref.arn.endswith("/" + ref.name)
    observed: list[dict[str, Any]] = []

    class OriginalSchedule:
        def setup(self, **kwargs: Any) -> ScheduleRef:
            assert kwargs["endpoint"] == ENDPOINT
            observed.append(copy.deepcopy(kwargs))
            return ref

    bind_managed_adapters(monkeypatch, sm, monitor=OriginalSchedule())
    result = run_flow(wire_monitoring=True)
    assert result.status == "deployed"
    assert result.monitoring["wired"] is True
    assert result.monitoring["schedule_name"] == ref.name
    assert result.monitoring["monitor_type"] == ref.monitor_type
    assert len(observed) == len(sm.calls("CreateMonitoringSchedule")) == 1


def test_unknown_describe_exposes_the_original_accepted_endpoint_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sm = StrictSageMaker(endpoint_status=None)
    bind_managed_adapters(monkeypatch, sm)
    result = run_flow(smoke=True)
    assert result.endpoint_name == ENDPOINT, "unknown observation lost its original accepted endpoint"
    assert result.detail["deploy"]["arn"] == sm.endpoint_arn(ENDPOINT)
    assert result.detail["deploy"]["status"] == "Unknown"
    assert result.status != "deployed" and not result.release
    assert len(sm.calls("CreateEndpoint")) == 1


@pytest.mark.parametrize("replacement", ["model_ref", "package_ref", "pid"])
def test_foreign_private_package_cannot_redirect_deployment(
    monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    from dataclasses import replace

    from amlas.backends.types import PackageRef
    from amlas.tools import deploy_neutral

    sm = StrictSageMaker()
    bind_managed_adapters(monkeypatch, sm)
    assert hasattr(deploy_neutral, "_packaged_deployment"), "missing source-owned original package scope"
    with deploy_neutral._packaged_deployment(MODEL) as original:
        changed = {
            "model_ref": ModelRef(uri="s3://other-model-bucket/model.tar.gz"),
            "package_ref": PackageRef(arn=sm.model_arn("foreign-model"), group="foreign-model"),
            "pid": original.pid + 1,
        }[replacement]
        foreign = replace(original, **{replacement: changed})
        result = raw(deploy_neutral.deploy_endpoint)(foreign, resources={"endpoint_name": ENDPOINT})
        assert result.get("success") is False
        assert not sm.calls("CreateEndpointConfig") and not sm.calls("CreateEndpoint")
        assert len(sm.calls("CreateModel")) == 1


def test_private_package_scope_expires_and_copied_context_cannot_revive_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextvars

    from amlas.tools import deploy_neutral

    sm = StrictSageMaker()
    bind_managed_adapters(monkeypatch, sm)
    assert hasattr(deploy_neutral, "_packaged_deployment"), "missing source-owned original package scope"
    with deploy_neutral._packaged_deployment(MODEL) as original:
        copied = contextvars.copy_context()
    result = copied.run(raw(deploy_neutral.deploy_endpoint), original, resources={"endpoint_name": ENDPOINT})
    assert result.get("success") is False
    assert len(sm.calls("CreateModel")) == 1
    assert not sm.calls("CreateEndpointConfig") and not sm.calls("CreateEndpoint")


def test_changed_backend_selection_refuses_original_package_before_endpoint_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.tools import deploy_neutral

    sm = StrictSageMaker()
    adapter, _runtime = bind_managed_adapters(monkeypatch, sm)
    assert hasattr(deploy_neutral, "_packaged_deployment"), "missing source-owned original package scope"
    with deploy_neutral._packaged_deployment(MODEL) as original:
        monkeypatch.setattr(
            REGISTRY,
            "resolve",
            lambda *_args, **_kwargs: ResolvedBackend(
                adapter,
                "deployment",
                "bedrock",
                "changed selection",
            ),
        )
        result = raw(deploy_neutral.deploy_endpoint)(original, resources={"endpoint_name": ENDPOINT})
        assert result.get("success") is False
        assert len(sm.calls("CreateModel")) == 1
        assert not sm.calls("CreateEndpointConfig") and not sm.calls("CreateEndpoint")


def test_public_package_dto_is_not_an_original_private_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.tools import deploy_neutral

    sm = StrictSageMaker()
    bind_managed_adapters(monkeypatch, sm)
    dto = raw(deploy_neutral.package_model)(MODEL)
    assert dto["package_arn"] and dto["group"]
    result = raw(deploy_neutral.deploy_endpoint)(dto, resources={"endpoint_name": ENDPOINT})
    assert result.get("success") is False
    assert len(sm.calls("CreateModel")) == 1, (
        "public package fields cannot select or remint the original native model"
    )
    assert not sm.calls("CreateEndpointConfig") and not sm.calls("CreateEndpoint")


def test_default_approval_denial_has_no_package_or_endpoint_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    sm = StrictSageMaker()
    bind_managed_adapters(monkeypatch, sm)
    result = run_deploy_flow(model=MODEL, policy=ApprovalPolicy())
    assert result.status == "denied" and result.approved is False
    assert not sm.requests


def test_confirmed_serverless_serving_remains_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    sm = StrictSageMaker()
    bind_managed_adapters(monkeypatch, sm)
    result = run_deploy_flow(
        model=MODEL,
        mode="serverless",
        resources={"endpoint_name": ENDPOINT, "memory_mb": 2048},
        wire_monitoring=False,
        policy=ApprovalPolicy(mode=ApprovalMode.AUTO),
    )
    assert result.status == "deployed" and result.mode == "serverless"
    variant = sm.calls("CreateEndpointConfig")[0]["ProductionVariants"][0]
    assert variant["ServerlessConfig"]["MemorySizeInMB"] == 2048
    assert "InstanceType" not in variant


def test_real_local_serving_keeps_prediction_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.backends.adapters.deployment import LocalServeBackend

    adapter = LocalServeBackend()
    monkeypatch.setattr(
        REGISTRY,
        "resolve",
        lambda *_args, **_kwargs: ResolvedBackend(
            adapter,
            "deployment",
            "local",
            "neutral local predictor",
        ),
    )
    result = run_deploy_flow(
        model=ModelRef(uri="local://neutral-model", algo="linear"),
        resources={"predict_fn": lambda rows: [sum(row) for row in rows]},
        smoke_payload=[1.0, 2.0],
        wire_monitoring=False,
        policy=ApprovalPolicy(mode=ApprovalMode.AUTO),
    )
    try:
        assert result.status == "deployed" and result.prediction == 3.0
        assert result.release["endpoint"] == result.endpoint_name
    finally:
        if result.endpoint_name:
            from amlas.backends.types import EndpointRef

            adapter.delete(endpoint=EndpointRef(result.endpoint_name, backend="local"))


def test_unknown_describe_logs_only_the_closed_failure_class(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "neutral-observation-detail-must-stay-private"

    class UnavailableDescription(StrictSageMaker):
        def describe_endpoint(self, **kwargs: Any) -> dict[str, Any]:
            self._record("DescribeEndpoint", kwargs)
            raise TimeoutError(marker)

    sm = UnavailableDescription()
    bind_managed_adapters(monkeypatch, sm)
    result = run_flow(smoke=True)
    assert result.endpoint_name == ENDPOINT and result.detail["deploy"]["status"] == "Unknown"
    messages = [record.getMessage() for record in caplog.records]
    assert "deployment endpoint observation unavailable: error_class=TimeoutError" in messages
    assert all(marker not in message for message in messages)
    assert all(sm.endpoint_arn(ENDPOINT) not in message for message in messages)


def test_package_scope_expiry_during_config_create_stops_before_endpoint_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextvars
    import threading

    from amlas.tools import deploy_neutral

    created = threading.Event()
    proceed = threading.Event()

    class PausedConfigCreation(StrictSageMaker):
        def create_endpoint_config(self, **kwargs: Any) -> dict[str, Any]:
            response = super().create_endpoint_config(**kwargs)
            created.set()
            assert proceed.wait(5), "neutral fixture config barrier timed out"
            return response

    sm = PausedConfigCreation()
    bind_managed_adapters(monkeypatch, sm)
    results: list[dict[str, Any]] = []
    with deploy_neutral._packaged_deployment(MODEL) as original:
        copied = contextvars.copy_context()
        worker = threading.Thread(
            target=lambda: results.append(
                copied.run(
                    raw(deploy_neutral.deploy_endpoint),
                    original,
                    resources={"endpoint_name": ENDPOINT},
                )
            )
        )
        worker.start()
        try:
            assert created.wait(5), "neutral fixture did not reach original config creation"
            original.scope.active = False
        finally:
            proceed.set()
            worker.join(5)
        assert not worker.is_alive()
    assert len(results) == 1 and results[0].get("success") is False
    assert len(sm.calls("CreateModel")) == len(sm.calls("CreateEndpointConfig")) == 1
    assert not sm.calls("CreateEndpoint") and not sm.calls("DescribeEndpoint")


def test_scope_expiry_after_endpoint_create_retains_private_original_unknown_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.tools import deploy_neutral

    class ExpiringEndpointCreation(StrictSageMaker):
        on_create: Any = None

        def create_endpoint(self, **kwargs: Any) -> dict[str, Any]:
            response = super().create_endpoint(**kwargs)
            self.on_create()
            return response

    sm = ExpiringEndpointCreation()
    bind_managed_adapters(monkeypatch, sm)
    with deploy_neutral._packaged_deployment(MODEL) as original:
        sm.on_create = lambda: setattr(original.scope, "active", False)
        result = raw(deploy_neutral.deploy_endpoint)(original, resources={"endpoint_name": ENDPOINT})
        assert result.get("success") is False and result.get("status") == "Unknown"
        assert result.get("outcome_unknown") is True
        retained = getattr(original.scope, "accepted_endpoint", None)
        assert retained is not None and retained.name == ENDPOINT
        assert retained.arn == sm.endpoint_arn(ENDPOINT)
        assert "endpoint_name" not in result and "arn" not in result
        assert len(sm.calls("CreateEndpoint")) == len(sm.calls("CreateEndpointConfig")) == 1
        assert not sm.calls("DescribeEndpoint")


def test_configuration_drift_inside_model_create_is_not_blessed_by_late_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    created = threading.Event()
    proceed = threading.Event()

    class PausedModelCreation(StrictSageMaker):
        def create_model(self, **kwargs: Any) -> dict[str, Any]:
            response = super().create_model(**kwargs)
            created.set()
            assert proceed.wait(5), "neutral model-create barrier timed out"
            return response

    sm = PausedModelCreation()
    adapter, _runtime = bind_managed_adapters(monkeypatch, sm)
    results = []
    worker = threading.Thread(target=lambda: results.append(run_flow()))
    worker.start()
    try:
        assert created.wait(5), "neutral model-create barrier was not reached"
        adapter._role = "arn:aws:iam::000000000000:role/changed-deployment"
    finally:
        proceed.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(results) == 1 and results[0].status != "deployed"
    assert sm.calls("CreateModel")[0]["ExecutionRoleArn"] == ROLE
    assert len(sm.calls("CreateModel")) == 1
    assert not sm.calls("CreateEndpointConfig") and not sm.calls("CreateEndpoint")


def test_resolved_backend_adapter_field_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    original = object()
    resolved = ResolvedBackend(original, "deployment", "sagemaker", "neutral original")
    with pytest.raises(FrozenInstanceError):
        resolved.adapter = object()
    assert resolved.adapter is original


@pytest.mark.parametrize("invalid", ["whitespace", "wrong_resource", "wrong_schedule"])
def test_native_monitor_arn_must_bind_its_actual_schedule_name(
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    name = "amlas-monitor-truth-endpoint"
    arn = {
        "whitespace": " ",
        "wrong_resource": f"arn:aws:sagemaker:{REGION}:000000000000:endpoint/{name}",
        "wrong_schedule": f"arn:aws:sagemaker:{REGION}:000000000000:monitoring-schedule/other-schedule",
    }[invalid]

    class InvalidNativeMonitor:
        def setup(self, **_kwargs: Any) -> ScheduleRef:
            return ScheduleRef(name=name, arn=arn, monitor_type="data_quality", schedule="hourly")

    sm = StrictSageMaker()
    bind_managed_adapters(monkeypatch, sm, monitor=InvalidNativeMonitor())
    result = run_flow(wire_monitoring=True)
    assert result.status == "deployed"
    assert result.monitoring.get("wired") is not True, (
        "malformed or mismatched native schedule reference was accepted"
    )


def test_same_foreign_create_and_describe_arn_never_binds_requested_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongEndpointArn(StrictSageMaker):
        @staticmethod
        def endpoint_arn(_name: str) -> str:
            return StrictSageMaker.endpoint_arn("other-endpoint")

    sm = WrongEndpointArn()
    _adapter, runtime = bind_managed_adapters(monkeypatch, sm)
    result = run_flow(smoke=True)
    assert result.status != "deployed", "matching foreign ARN cannot bind the requested endpoint name"
    assert result.detail["deploy"]["status"] == "Unknown"
    assert not result.release and not runtime.requests
    assert len(sm.calls("CreateEndpoint")) == 1


def test_legacy_transport_omitted_create_arn_uses_correct_native_describe_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingCreateArn(StrictSageMaker):
        def create_endpoint(self, **kwargs: Any) -> dict[str, Any]:
            # The actual native response is first validated. Only this explicit
            # legacy transport projection drops its required ARN; it is not a
            # native-shaped Create success or a new admission grant.
            super().create_endpoint(**kwargs)
            return {}

    sm = MissingCreateArn()
    bind_managed_adapters(monkeypatch, sm)
    result = run_flow(smoke=True)
    assert result.status == "deployed", result.detail
    assert result.endpoint_name == ENDPOINT and result.detail["deploy"]["arn"] == sm.endpoint_arn(ENDPOINT)
    assert len(sm.calls("CreateEndpoint")) == len(sm.calls("DescribeEndpoint")) == 1


@pytest.mark.parametrize("stage", ["describe", "register_target", "put_policy"])
def test_scope_expiry_at_observation_or_autoscale_boundary_refuses_exposure(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    import uuid

    from amlas.tools import deploy_neutral

    class ExpiringDescription(StrictSageMaker):
        on_describe: Any = None

        def describe_endpoint(self, **kwargs: Any) -> dict[str, Any]:
            response = super().describe_endpoint(**kwargs)
            if stage == "describe":
                self.on_describe()
            return response

    class StrictAutoscaling:
        def __init__(self) -> None:
            self.model = botocore.session.get_session().get_service_model("application-autoscaling")
            self.requests: list[str] = []
            self.on_native: Any = None

        def _record(self, operation: str, kwargs: dict[str, Any]) -> None:
            validate_parameters(kwargs, self.model.operation_model(operation).input_shape)
            self.requests.append(operation)

        def register_scalable_target(self, **kwargs: Any) -> dict[str, Any]:
            self._record("RegisterScalableTarget", kwargs)
            if stage == "register_target":
                self.on_native()
            response = {
                "ScalableTargetARN": f"arn:aws:application-autoscaling:{REGION}:000000000000:scalable-target/neutral"
            }
            validate_parameters(response, self.model.operation_model("RegisterScalableTarget").output_shape)
            return response

        def put_scaling_policy(self, **kwargs: Any) -> dict[str, Any]:
            self._record("PutScalingPolicy", kwargs)
            if stage == "put_policy":
                self.on_native()
            response = {
                "PolicyARN": f"arn:aws:autoscaling:{REGION}:000000000000:scalingPolicy:{uuid.uuid4()}:resource/neutral"
            }
            validate_parameters(response, self.model.operation_model("PutScalingPolicy").output_shape)
            return response

    sm = ExpiringDescription()
    adapter, _runtime = bind_managed_adapters(monkeypatch, sm)
    scaling = StrictAutoscaling()
    adapter._aas = scaling
    with deploy_neutral._packaged_deployment(MODEL) as original:

        def expire() -> None:
            original.scope.active = False

        sm.on_describe = expire
        scaling.on_native = expire
        result = raw(deploy_neutral.deploy_endpoint)(
            original,
            resources={"endpoint_name": ENDPOINT},
            autoscaling={"min_capacity": 1, "max_capacity": 2},
        )
        assert result.get("success") is False and result.get("status") == "Unknown"
        assert result.get("outcome_unknown") is True and "endpoint_name" not in result and "arn" not in result
        assert original.scope.accepted_endpoint is not None
        assert original.scope.accepted_endpoint.name == ENDPOINT
    assert (
        len(sm.calls("CreateModel"))
        == len(sm.calls("CreateEndpointConfig"))
        == len(sm.calls("CreateEndpoint"))
        == 1
    )
    assert len(sm.calls("DescribeEndpoint")) == 1
    expected = {
        "describe": [],
        "register_target": ["RegisterScalableTarget"],
        "put_policy": ["RegisterScalableTarget", "PutScalingPolicy"],
    }[stage]
    assert scaling.requests == expected


def test_legacy_plain_endpoint_reference_remains_conservatively_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.backends.types import EndpointRef
    from amlas.tools import deploy_neutral

    class PlainEndpointAdapter:
        def deploy(self, **_kwargs: Any) -> EndpointRef:
            return EndpointRef(name=ENDPOINT, arn=StrictSageMaker.endpoint_arn(ENDPOINT), backend="sagemaker")

    monkeypatch.setattr(
        REGISTRY,
        "resolve",
        lambda *_args, **_kwargs: ResolvedBackend(
            PlainEndpointAdapter(),
            "deployment",
            "sagemaker",
            "plain legacy reference",
        ),
    )
    result = raw(deploy_neutral.deploy_endpoint)(MODEL)
    assert result["status"] == "Unknown" and result["endpoint_name"] == ENDPOINT
