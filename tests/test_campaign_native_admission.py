"""SDK-shaped neutral native-read contracts; no live policy or cohort approval."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters
from test_campaign_admission import ACCOUNT, DEPLOYMENT, REGION, admission_module, native_fields

MODELS = {name: Session().get_service_model(name) for name in ("bedrock-agentcore-control", "iam")}


def native_module():
    try:
        return importlib.import_module("amlas.control_plane.campaign_native_admission")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.campaign_native_admission":
            pytest.fail("missing-symbol: campaign_native_admission.NativeAdmissionReader", pytrace=False)
        raise


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def manifest():
    ca = admission_module()
    fields = native_fields(ca)
    trust = canonical({"Version": "2012-10-17", "Statement": []})
    for field in ["runtime_execution_role"]:
        fields[field] = dataclasses.replace(
            fields[field], trust_policy=trust, trust_sha256=hashlib.sha256(trust).hexdigest()
        )
    fields["co_owner_roles"] = tuple(
        dataclasses.replace(role, trust_policy=trust, trust_sha256=hashlib.sha256(trust).hexdigest())
        for role in fields["co_owner_roles"]
    )
    return ca.AdmissionManifest(
        **fields,
        owner_namespace="12345678-1234-4234-9234-00000000dd01",
        change_control_reference="fixture-freeze",
    )


def meta(service, region):
    endpoint = (
        f"https://bedrock-agentcore-control.{region}.amazonaws.com"
        if service != "iam"
        else "https://iam.amazonaws.com"
    )
    return SimpleNamespace(
        service_model=MODELS[service],
        region_name=region,
        endpoint_url=endpoint,
        config=SimpleNamespace(retries={"total_max_attempts": 1}),
    )


def response_metadata():
    return {"HTTPStatusCode": 200, "RequestId": "fixture-read", "RetryAttempts": 0, "HTTPHeaders": {}}


class Control:
    def __init__(self, selected):
        self.meta = meta("bedrock-agentcore-control", REGION)
        self.calls = []
        self.failure = None
        now = datetime(2026, 10, 17, tzinfo=UTC)
        runtime_id = selected.runtime_arn.rsplit("/", 1)[1]
        self.runtime = {
            "agentRuntimeArn": selected.runtime_arn,
            "agentRuntimeName": "amlasRuntime",
            "agentRuntimeId": runtime_id,
            "agentRuntimeVersion": selected.runtime_version,
            "createdAt": now,
            "lastUpdatedAt": now,
            "roleArn": selected.runtime_execution_role.role_arn,
            "networkConfiguration": {"networkMode": "PUBLIC"},
            "protocolConfiguration": {"serverProtocol": "HTTP"},
            "status": "READY",
            "lifecycleConfiguration": {"idleRuntimeSessionTimeout": 900, "maxLifetime": 28800},
            "environmentVariables": {"PRIVATE_FIXTURE_VALUE": "not-retained-fixture"},
            "ResponseMetadata": response_metadata(),
        }
        self.endpoint = {
            "agentRuntimeEndpointArn": selected.endpoint_arn,
            "agentRuntimeArn": selected.runtime_arn,
            "status": "READY",
            "createdAt": now,
            "lastUpdatedAt": now,
            "name": selected.endpoint_name,
            "id": "fixtureEndpoint-ABCDEFGHIJ",
            "liveVersion": selected.endpoint_live_version,
            "ResponseMetadata": response_metadata(),
        }
        self.tags = {
            "tags": {
                "amlas:deployment-id": DEPLOYMENT,
                "amlas:deployment-prefix": selected.deployment_prefix,
            },
            "ResponseMetadata": response_metadata(),
        }
        self.policies = {
            selected.runtime_arn: {
                "policy": selected.runtime_policy.decode(),
                "ResponseMetadata": response_metadata(),
            },
            selected.endpoint_arn: {
                "policy": selected.endpoint_policy.decode(),
                "ResponseMetadata": response_metadata(),
            },
        }

    def _read(self, name, kwargs, value):
        validate_parameters(kwargs, MODELS["bedrock-agentcore-control"].operation_model(name).input_shape)
        self.calls.append((name, copy.deepcopy(kwargs)))
        if self.failure == name:
            raise ClientError({"Error": {"Code": "AccessDeniedException"}}, name)
        return copy.deepcopy(value)

    def get_agent_runtime(self, **kwargs):
        return self._read("GetAgentRuntime", kwargs, self.runtime)

    def get_agent_runtime_endpoint(self, **kwargs):
        return self._read("GetAgentRuntimeEndpoint", kwargs, self.endpoint)

    def list_tags_for_resource(self, **kwargs):
        return self._read("ListTagsForResource", kwargs, self.tags)

    def get_resource_policy(self, **kwargs):
        return self._read("GetResourcePolicy", kwargs, self.policies[kwargs["resourceArn"]])

    def put_resource_policy(self, **kwargs):
        pytest.fail("reader cannot mutate policy")


class IAM:
    def __init__(self, selected):
        self.meta = meta("iam", "aws-global")
        self.calls = []
        self.roles = {}
        for role in (selected.runtime_execution_role, *selected.co_owner_roles):
            name = role.role_arn.rsplit("/", 1)[1]
            self.roles[name] = {
                "Role": {
                    "Path": "/",
                    "RoleName": name,
                    "RoleId": role.role_id,
                    "Arn": role.role_arn,
                    "CreateDate": datetime(2026, 10, 17, tzinfo=UTC),
                    "AssumeRolePolicyDocument": json.loads(role.trust_policy),
                },
                "ResponseMetadata": response_metadata(),
            }

    def get_role(self, **kwargs):
        validate_parameters(kwargs, MODELS["iam"].operation_model("GetRole").input_shape)
        self.calls.append(copy.deepcopy(kwargs))
        return copy.deepcopy(self.roles[kwargs["RoleName"]])

    def update_assume_role_policy(self, **kwargs):
        pytest.fail("reader cannot mutate trust")


def fixture_reader():
    cn = native_module()
    selected = manifest()
    selectors = cn.NativeAdmissionSelectors.from_manifest(selected)
    control, iam = Control(selected), IAM(selected)
    reader = cn.NativeAdmissionReader(selectors, control_client=control, iam_client=iam)
    return cn, reader, selectors, control, iam, selected


def test_exact_fresh_reads_return_untrusted_observation_only():
    cn, reader, _, control, iam, selected = fixture_reader()
    one, two = reader.read_current(), reader.read_current()
    ca = admission_module()
    assert type(one) is type(two) is ca.AdmissionObservation
    assert one is not two
    assert one.runtime_arn == selected.runtime_arn
    assert one.endpoint_live_version == selected.runtime_version
    assert one.co_owner_roles == selected.co_owner_roles
    assert len(control.calls) == 10 and len(iam.calls) == 4
    assert control.calls[0] == (
        "GetAgentRuntime",
        {"agentRuntimeId": selected.runtime_arn.rsplit("/", 1)[1], "agentRuntimeVersion": "1"},
    )
    assert not hasattr(one, "owner_namespace") and not hasattr(one, "approval_reference")
    assert not hasattr(reader, "approved") and not hasattr(reader, "witness")
    assert cn is not None


def test_raw_policy_string_and_defined_sdk_trust_encoding_are_distinct():
    _, reader, _, control, iam, selected = fixture_reader()
    raw = " \n" + selected.runtime_policy.decode() + "\n "
    control.policies[selected.runtime_arn]["policy"] = raw
    observation = reader.read_current()
    assert observation.runtime_policy == raw.encode("utf-8")
    assert observation.runtime_policy_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    native_trust = iam.roles["fixture-owner"]["Role"]["AssumeRolePolicyDocument"]
    assert observation.co_owner_roles[0].trust_policy == canonical(native_trust)


@pytest.mark.parametrize("value", [None, "", "   ", True, b"{}"])
def test_blank_missing_or_wrong_policy_remains_unknown(value):
    cn, reader, _, control, _, selected = fixture_reader()
    control.policies[selected.runtime_arn]["policy"] = value
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()


def test_wrong_policy_output_key_cannot_be_guessed_as_an_alias():
    cn, reader, _, control, _, selected = fixture_reader()
    control.policies[selected.runtime_arn] = {
        "resourcePolicy": selected.runtime_policy.decode(),
        "ResponseMetadata": response_metadata(),
    }
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()


@pytest.mark.parametrize(
    ("which", "field", "value"),
    [
        ("runtime", "status", "UPDATING"),
        ("runtime", "agentRuntimeVersion", "2"),
        (
            "runtime",
            "agentRuntimeArn",
            "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/otherRuntime-ABCDEFGHIJ",
        ),
        ("runtime", "roleArn", "arn:aws:iam::111122223333:role/other-role"),
        ("endpoint", "liveVersion", None),
        ("endpoint", "targetVersion", "2"),
        ("endpoint", "status", "UPDATING"),
        ("endpoint", "name", "DEFAULT"),
    ],
)
def test_nonqualified_runtime_or_endpoint_refuses(which, field, value):
    cn, reader, _, control, _, _ = fixture_reader()
    getattr(control, which)[field] = value
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()


@pytest.mark.parametrize("field", ["RoleId", "Arn", "AssumeRolePolicyDocument"])
def test_role_identity_and_canonical_trust_drift_refuses(field):
    cn, reader, _, _, iam, _ = fixture_reader()
    role = iam.roles["fixture-owner"]["Role"]
    role[field] = {
        "RoleId": "AROA" + "Z" * 17,
        "Arn": f"arn:aws:iam::{ACCOUNT}:role/another-owner",
        "AssumeRolePolicyDocument": {"Statement": [], "Version": "changed"},
    }[field]
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()


def test_iam_encoded_or_missing_trust_is_not_raw_http_evidence():
    cn, reader, _, _, iam, _ = fixture_reader()
    iam.roles["fixture-owner"]["Role"]["AssumeRolePolicyDocument"] = "%7B%7D"
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()
    iam.roles["fixture-owner"]["Role"].pop("AssumeRolePolicyDocument")
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()


def test_deployment_native_tags_are_required_and_checked():
    cn, reader, _, control, _, _ = fixture_reader()
    control.tags["tags"]["amlas:deployment-id"] = "12345678-1234-4234-9234-00000000dd02"
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()


def test_sdk_read_failure_does_not_retry_or_return_partial_observation():
    cn, reader, _, control, iam, _ = fixture_reader()
    control.failure = "GetResourcePolicy"
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()
    assert sum(name == "GetResourcePolicy" for name, _ in control.calls) == 1
    assert iam.calls == []


@pytest.mark.parametrize("attempts", [None, 2, True])
def test_one_attempt_private_transport_required_before_any_read(attempts):
    cn, _, selectors, control, iam, _ = fixture_reader()
    control.meta.config.retries = {"total_max_attempts": attempts}
    with pytest.raises(cn.NativeAdmissionReadError):
        cn.NativeAdmissionReader(selectors, control_client=control, iam_client=iam)
    assert control.calls == [] and iam.calls == []


def test_inherited_reader_refuses_before_any_sdk_call(monkeypatch):
    cn, reader, _, control, iam, _ = fixture_reader()
    pid = cn.os.getpid()
    monkeypatch.setattr(cn.os, "getpid", lambda: pid + 1)
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()
    assert control.calls == [] and iam.calls == []


def test_public_dict_cannot_supply_native_selectors_or_owner_authority():
    cn = native_module()
    selected = manifest()
    control, iam = Control(selected), IAM(selected)
    with pytest.raises(cn.NativeAdmissionReadError):
        cn.NativeAdmissionReader(
            {"runtime_arn": selected.runtime_arn, "approved": True}, control_client=control, iam_client=iam
        )
    assert control.calls == [] and iam.calls == []


@pytest.mark.parametrize("target", ["absent", None, "1"])
def test_target_projection_preserves_actual_reported_or_unreported_value(target):
    _, reader, _, control, _, _ = fixture_reader()
    if target != "absent":
        control.endpoint["targetVersion"] = target
    before = copy.deepcopy(control.endpoint)
    observation = reader.read_current()
    assert observation.endpoint_target_version == (None if target == "absent" else target)
    assert control.endpoint == before


@pytest.mark.parametrize("target", ["absent", None])
def test_actual_guard_does_not_match_unreported_target_to_approved_string(target):
    cn = native_module()
    ca = admission_module()
    expected = dataclasses.replace(manifest(), endpoint_target_version="1")
    selectors = cn.NativeAdmissionSelectors.from_manifest(expected)
    control, iam = Control(expected), IAM(expected)
    if target is None:
        control.endpoint["targetVersion"] = None
    reader = cn.NativeAdmissionReader(selectors, control_client=control, iam_client=iam)
    assert reader.read_current().endpoint_target_version is None
    approved = ca.approve_manifest(expected, approval_reference="fixture-reported-target-review")
    with pytest.raises(ca.AdmissionError):
        ca.AdmissionGuard(approved, reader).verify()


def test_actual_guard_compares_untrusted_reader_without_minting_approval():
    _, reader, _, control, _, selected = fixture_reader()
    ca = admission_module()
    approved = ca.approve_manifest(selected, approval_reference="fixture-native-metadata-review")
    guard = ca.AdmissionGuard(approved, reader)
    witness = guard.verify()
    control.policies[selected.runtime_arn]["policy"] += " "
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)


@pytest.mark.parametrize(
    ("response", "field"), [("runtime", "createdAt"), ("endpoint", "liveVersion"), ("endpoint", "id")]
)
def test_missing_required_or_admission_required_sdk_fields_refuse(response, field):
    cn, reader, _, control, _, _ = fixture_reader()
    getattr(control, response).pop(field)
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()


def test_unknown_authority_fields_and_retry_marked_response_refuse():
    cn, reader, _, control, _, _ = fixture_reader()
    control.runtime["native_admission"] = True
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()
    control.runtime.pop("native_admission")
    control.runtime["ResponseMetadata"]["RetryAttempts"] = 1
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()


def test_client_configuration_is_rechecked_before_fresh_native_reads():
    cn, reader, _, control, iam, _ = fixture_reader()
    control.meta.config.retries["total_max_attempts"] = 2
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()
    assert control.calls == [] and iam.calls == []


def test_missing_protocol_is_not_reported_http_qualification():
    cn, reader, _, control, _, _ = fixture_reader()
    control.runtime.pop("protocolConfiguration")
    assert "protocolConfiguration" not in control.runtime
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()


def test_positive_fixture_endpoint_id_conforms_to_installed_model_pattern():
    import re

    _, reader, _, control, _, _ = fixture_reader()
    reader.read_current()
    shape = (
        MODELS["bedrock-agentcore-control"]
        .operation_model("GetAgentRuntimeEndpoint")
        .output_shape.members["id"]
    )
    assert re.fullmatch(shape.metadata["pattern"], control.endpoint["id"])


def test_native_default_id_is_non_authoritative_and_not_blanket_pattern_gated():
    import re

    cn = native_module()
    expected = dataclasses.replace(
        manifest(), endpoint_name="DEFAULT", endpoint_arn=manifest().runtime_arn + "/runtime-endpoint/DEFAULT"
    )
    selectors = cn.NativeAdmissionSelectors.from_manifest(expected)
    control, iam = Control(expected), IAM(expected)
    # Existing native GetEndpoint receipt reports this ID; it is not a selector.
    control.endpoint["id"] = "DEFAULT"
    shape = (
        MODELS["bedrock-agentcore-control"]
        .operation_model("GetAgentRuntimeEndpoint")
        .output_shape.members["id"]
    )
    assert re.fullmatch(shape.metadata["pattern"], control.endpoint["id"]) is None
    reader = cn.NativeAdmissionReader(selectors, control_client=control, iam_client=iam)
    observed = reader.read_current()
    assert observed.endpoint_name == "DEFAULT"
    assert observed.endpoint_arn == expected.endpoint_arn
    assert not hasattr(observed, "endpoint_id")


@pytest.mark.parametrize("protocol", [None, {}, {"serverProtocol": "MCP"}])
def test_only_explicit_http_protocol_is_qualified(protocol):
    cn, reader, _, control, _, _ = fixture_reader()
    control.runtime["protocolConfiguration"] = protocol
    with pytest.raises(cn.NativeAdmissionReadError):
        reader.read_current()
