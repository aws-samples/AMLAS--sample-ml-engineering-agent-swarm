"""Production installer RED over actual handler; inert private qualifier/SDK ports only."""

import json
from contextlib import contextmanager

import pytest
from test_callback_native_qualifier import ROLE, ROLE_ID, MetadataPort
from test_campaign_callback import RUNTIME
from test_campaign_callback import (
    test_legacy_disabled_handler_retains_original_training_route as check_legacy,
)
from test_normal_wake_routing import (
    RUN_ID,
    TRANSPORT_SESSION,
    _declared_route,
    selected_callback_env,  # noqa: F401
)

from amlas.async_callback import lambda_handler as module


def install_test_owned_factory(monkeypatch, *, unavailable=False):
    value, ddb, runtime, route = _declared_route()
    original_scope = module.campaign_callback_scope
    calls = []

    @contextmanager
    def private_factory():
        # This test-owned factory stands in for private native bootstrap only.
        # Event/env/types are DATA; they do not manufacture native authentication.
        calls.append("private_factory")
        if unavailable:
            raise module.CallbackQualificationError("neutral current role or route metadata unavailable")
        config = module._campaign_config()
        assert config.runtime_arn == RUNTIME and config.runtime_version == "3"
        with original_scope(route, dynamodb_client=ddb, runtime_client=runtime, clock=lambda: 1000.0):
            yield

    monkeypatch.setattr(module, "_normal_callback_bootstrap", private_factory, raising=False)
    monkeypatch.setattr(
        module.boto3,
        "client",
        lambda *args, **kwargs: pytest.fail("unqualified SDK factory reached"),
    )
    assert module._CAMPAIGN_SERVICES.get() is None
    return value, ddb, runtime, route, calls


def test_mode1_handler_installs_private_factory_and_delivers_normal_current_target(
    monkeypatch,
):
    value, ddb, runtime, route, calls = install_test_owned_factory(monkeypatch)
    result = module.handler(value)
    assert calls == ["private_factory"], "missing production installer before mode1 routing"
    assert result["delivery_status"] == "delivered"
    assert len(runtime.calls) == 1 and len(ddb.writes) == 1 and ddb.descriptions
    sent = runtime.calls[0]
    assert sent["agentRuntimeArn"] == RUNTIME and sent["qualifier"] == "NORMAL"
    assert sent["runtimeSessionId"] == TRANSPORT_SESSION
    assert json.loads(sent["payload"]) == {
        "action": "resume_prompt_run",
        "run_id": RUN_ID,
    }
    assert route.calls and all(config.runtime_version == "3" for config, _identity in route.calls)
    assert runtime.body.closed_called and module._CAMPAIGN_SERVICES.get() is None
    assert result["lifecycle_verified"] is False and result["quality_success"] is False


def test_unknown_private_bootstrap_dependency_is_fail_closed_before_io(monkeypatch):
    value, ddb, runtime, _route, calls = install_test_owned_factory(monkeypatch, unavailable=True)
    result = module.handler(value)
    assert calls == ["private_factory"], "missing production metadata dependency qualification"
    assert result["statusCode"] in (403, 503)
    assert result["delivery_status"] in (
        "qualification_missing",
        "qualification_refused",
        "qualification_unavailable",
        "setup_unavailable",
    )
    assert ddb.calls == [] and ddb.descriptions == [] and ddb.writes == []
    assert runtime.calls == [] and module._CAMPAIGN_SERVICES.get() is None


def test_legacy_disabled_route_never_installs_normal_private_factory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("legacy path attempted normal bootstrap")

    monkeypatch.setattr(module, "_normal_callback_bootstrap", forbidden, raising=False)
    check_legacy(monkeypatch)


def test_unscoped_oversized_readiness_stream_is_bounded_before_private_factory(monkeypatch):
    calls = []

    @contextmanager
    def unavailable_factory():
        calls.append("factory")
        raise module.CallbackQualificationError("neutral unavailable provider")
        yield

    monkeypatch.setattr(module, "_normal_callback_bootstrap", unavailable_factory)
    assert module._CAMPAIGN_SERVICES.get() is None
    value = {"Records": [{"OrdinaryData": "a" * (4 * 1024 * 1024 + 1)}]}
    with pytest.raises(RuntimeError, match="normal readiness delivery unresolved"):
        module.handler(value)
    assert calls == [], "existing stream bytes bound must precede production provider/client setup"


class BootstrapOwnedPort(MetadataPort):
    def __init__(self, service):
        super().__init__(service)
        self.closes = 0
        self.close_error = False

    def close(self):
        self.closes += 1
        if self.close_error:
            raise OSError("neutral owned client close unavailable")


def real_bootstrap_ports(monkeypatch, *, second_setup_failure=False, close_errors=False):
    sts, control = BootstrapOwnedPort("sts"), BootstrapOwnedPort("bedrock-agentcore-control")
    sts.close_error = control.close_error = close_errors
    created = []
    monkeypatch.setenv("AMLAS_CALLBACK_ROLE_ARN", ROLE)
    monkeypatch.setenv("AMLAS_CALLBACK_ROLE_ID", ROLE_ID)

    def client(name, **kwargs):
        assert kwargs["region_name"] == "us-west-2"
        assert kwargs["config"].retries["total_max_attempts"] == 1
        assert kwargs["config"].connect_timeout == 5 and kwargs["config"].read_timeout == 10
        if name == "bedrock-agentcore-control" and second_setup_failure:
            raise OSError("neutral second client setup unavailable")
        value = sts if name == "sts" else control
        assert name in ("sts", "bedrock-agentcore-control")
        created.append(value)
        return value

    monkeypatch.setattr(module.boto3, "client", client)
    assert module._CAMPAIGN_SERVICES.get() is None
    return sts, control, created


def test_real_bootstrap_closes_owned_clients_after_success(monkeypatch):
    sts, control, created = real_bootstrap_ports(monkeypatch)
    with module._normal_callback_bootstrap():
        services = module._CAMPAIGN_SERVICES.get()
        assert type(services) is module._CallbackServices and services.active is True
        module._qualify(services, module._campaign_config(), {"kind": "Training"})
    assert created == [sts, control]
    assert [port.closes for port in created] == [1, 1], "bootstrap owned SDK clients not closed"
    assert module._CAMPAIGN_SERVICES.get() is None and services.active is False
    assert services.route.active is False


def test_real_bootstrap_cleanup_preserves_original_dispatch_failure(monkeypatch):
    sts, control, created = real_bootstrap_ports(monkeypatch, close_errors=True)
    original = RuntimeError("neutral original dispatch failure")
    with pytest.raises(RuntimeError) as caught:
        with module._normal_callback_bootstrap():
            services = module._CAMPAIGN_SERVICES.get()
            raise original
    assert caught.value is original
    assert created == [sts, control]
    assert [port.closes for port in created] == [1, 1], "dispatch failure abandoned owned clients"
    assert module._CAMPAIGN_SERVICES.get() is None and services.active is False
    assert services.route.active is False


def test_real_bootstrap_partial_client_setup_closes_original_without_masking(monkeypatch):
    sts, control, created = real_bootstrap_ports(monkeypatch, second_setup_failure=True, close_errors=True)
    with pytest.raises(module.CallbackQualificationError):
        with module._normal_callback_bootstrap():
            pytest.fail("partial setup entered callback scope")
    assert created == [sts]
    assert sts.closes == 1, "partial setup abandoned returned STS client"
    assert control.closes == 0 and module._CAMPAIGN_SERVICES.get() is None


def test_real_bootstrap_qualifier_construction_failure_closes_all_created_clients(monkeypatch):
    sts, control, created = real_bootstrap_ports(monkeypatch, close_errors=True)
    parent = module._NativeCallbackRouteQualifier

    class ConstructorFailure(parent):
        def __init__(self, *args, **kwargs):
            raise module.CallbackQualificationError("neutral qualifier construction unavailable")

    monkeypatch.setattr(module, "_NativeCallbackRouteQualifier", ConstructorFailure)
    with pytest.raises(module.CallbackQualificationError):
        with module._normal_callback_bootstrap():
            pytest.fail("failed qualifier entered callback scope")
    assert created == [sts, control]
    assert [port.closes for port in created] == [1, 1], "constructor failure abandoned owned clients"
    assert module._CAMPAIGN_SERVICES.get() is None
