"""Actual offline CDK normal-wake wiring; no native/event/lifecycle authority."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Template

from infra.cdk_stacks import AsyncTrainingStack, RuntimeStack
from infra.cdk_stacks import runtime_stack as runtime_module

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT, REGION = "111122223333", "us-west-2"
DEPLOYMENT = str(UUID("aaaabbbb-cccc-4ddd-8eee-ffff0000aaaa"))
QUALIFIER, VERSION, ROUTE_SHA = "candidate", "37", "a" * 64
RUNTIME_ARN = f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/declared_runtime"
KINDS = (AsyncTrainingStack, RuntimeStack)
CALLBACK_HANDLER = "amlas.async_callback.lambda_handler.handler"
RUNTIME_PARAMETER = "AgentRuntimeArn"
RUNTIME_PARAMETER_DESCRIPTION = (
    "ARN of the AgentCore Runtime to resume on training callback. "
    "Leave blank if Runtime isn't deployed yet \u2014 Lambda will still update DDB."
)
EARLIER_RUNTIME_ARN = f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/earlier_runtime"
RUNTIME_FALLBACK = f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/*"
RUNTIME_ACTIONS = (
    "bedrock-agentcore:InvokeAgentRuntime",
    "bedrock-agentcore:GetAgentRuntime",
    "bedrock-agentcore:GetAgentRuntimeEndpoint",
)


class _FixedDatetime:
    @staticmethod
    def now(tz):
        assert tz is UTC
        return datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def stable_existing_build_timestamp(monkeypatch):
    # Fix only the existing unrelated synth timestamp for whole-template equality.
    monkeypatch.setattr(runtime_module, "datetime", _FixedDatetime)


def _options(kind):
    values = {
        "normal_callback_enabled": True,
        "normal_callback_deployment_id": DEPLOYMENT,
        "normal_callback_qualifier": QUALIFIER,
        "normal_callback_runtime_version": VERSION,
        "normal_callback_route_sha256": ROUTE_SHA,
    }
    if kind is AsyncTrainingStack:
        values["normal_callback_runtime_arn"] = RUNTIME_ARN
    return values


def _template(tmp_path, kind, prefix="trial-a", **options):
    app = App(outdir=str(tmp_path / str(uuid4())))
    args = {"project_name": prefix.replace("-", "_")} if kind is RuntimeStack else {}
    stack = kind(
        app,
        kind.__name__,
        env=Environment(account=ACCOUNT, region=REGION),
        hyphen_project=prefix,
        **args,
        **options,
    )
    return Template.from_stack(stack).to_json()


def _of_type(template, resource_type):
    return {
        key: row["Properties"] for key, row in template["Resources"].items() if row["Type"] == resource_type
    }


def _bindings(template):
    callbacks = {
        key: value
        for key, value in _of_type(template, "AWS::Lambda::Function").items()
        if value.get("Handler") == CALLBACK_HANDLER
    }
    assert len(callbacks) == 1
    roles = {
        key: value
        for key, value in _of_type(template, "AWS::IAM::Role").items()
        if value.get("RoleName", "").endswith(("-training-callback-role", "-callback-lambda"))
    }
    assert len(roles) == 1
    tables = _of_type(template, "AWS::DynamoDB::Table")
    assert len(tables) == 1
    return next(iter(callbacks.items())), next(iter(roles.items())), next(iter(tables.items()))


def _statements(template, role):
    return [
        statement
        for policy in _of_type(template, "AWS::IAM::Policy").values()
        if {"Ref": role} in policy.get("Roles", [])
        for statement in policy["PolicyDocument"]["Statement"]
        if statement["Effect"] == "Allow"
    ]


def _render_runtime(value, runtime_ref):
    if isinstance(value, str):
        return value
    if value == runtime_ref:
        return RUNTIME_ARN
    if isinstance(value, dict) and set(value) == {"Fn::Join"}:
        separator, parts = value["Fn::Join"]
        return separator.join(_render_runtime(part, runtime_ref) for part in parts)
    raise AssertionError("normal route resource is not an exact runtime/endpoint binding")


def _action_resources(template, role, action):
    resources = []
    for statement in _statements(template, role):
        actions = statement["Action"]
        if action in ([actions] if isinstance(actions, str) else actions):
            value = statement["Resource"]
            resources.extend(value if isinstance(value, list) else [value])
    return resources


def _parameter_reads(value, name):
    """Every node outside the Parameters section that reads the named stack parameter."""
    found = []
    if isinstance(value, dict):
        if value == {"Ref": name}:
            found.append(value)
        for key, item in value.items():
            if key == "Fn::Sub" and "${" + name + "}" in (item if isinstance(item, str) else item[0]):
                found.append(value)
            found.extend(_parameter_reads(item, name))
    elif isinstance(value, list):
        for item in value:
            found.extend(_parameter_reads(item, name))
    return found


def _with_stored_parameters(value, stored, conditions=None):
    """Resolve Ref, Fn::Join and Fn::If as an update that reuses stored parameter values would."""
    if isinstance(value, dict):
        if set(value) == {"Ref"} and value["Ref"] in stored:
            return stored[value["Ref"]]
        if set(value) == {"Fn::Join"}:
            separator, parts = value["Fn::Join"]
            parts = [_with_stored_parameters(part, stored, conditions) for part in parts]
            if all(isinstance(part, str) for part in parts):
                return separator.join(parts)
            return {"Fn::Join": [separator, parts]}
        if conditions and set(value) == {"Fn::If"} and value["Fn::If"][0] in conditions:
            name, when_true, when_false = value["Fn::If"]
            chosen = when_true if _condition_holds(conditions[name], stored) else when_false
            return _with_stored_parameters(chosen, stored, conditions)
        return {key: _with_stored_parameters(item, stored, conditions) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_stored_parameters(item, stored, conditions) for item in value]
    return value


def _condition_holds(expression, stored):
    if set(expression) == {"Fn::Not"}:
        return not _condition_holds(expression["Fn::Not"][0], stored)
    if set(expression) == {"Fn::Equals"}:
        left, right = (_with_stored_parameters(item, stored) for item in expression["Fn::Equals"])
        return left == right
    raise AssertionError("condition form not modelled")


def _stored_runtime_parameter(template, stack=None, passed=None):
    """What CloudFormation stores for AgentRuntimeArn after a deploy of this template.

    A passed value wins, then the value the stack already stores, then the template default
    (stack=None creates the stack). A template that does not declare the parameter drops it.
    """
    declared = template.get("Parameters", {}).get(RUNTIME_PARAMETER)
    if declared is None:
        assert passed is None, "CloudFormation refuses a value for a parameter the template lacks"
        return {}
    if passed is not None:
        return {RUNTIME_PARAMETER: passed}
    if stack and RUNTIME_PARAMETER in stack:
        return {RUNTIME_PARAMETER: stack[RUNTIME_PARAMETER]}
    return {RUNTIME_PARAMETER: declared["Default"]}


def _deployed(template, stored):
    """The template with its resources resolved against the stored parameter values."""
    return dict(
        template,
        Resources=_with_stored_parameters(template["Resources"], stored, template.get("Conditions", {})),
    )


@pytest.mark.parametrize("kind", KINDS)
def test_legacy_baseline_has_only_existing_callback_configuration(tmp_path, kind):
    template = _template(tmp_path, kind)
    (_, callback), _, (_, table) = _bindings(template)
    assert set(callback["Environment"]["Variables"]) == {"AGENTCORE_RUNTIME_ARN", "JOBS_TABLE"}
    assert "StreamSpecification" not in table
    assert _of_type(template, "AWS::Lambda::EventSourceMapping") == {}


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_disabled_normal_callback_preserves_complete_legacy_template(tmp_path, kind):
    before = _template(tmp_path, kind)
    after = _template(tmp_path, kind, normal_callback_enabled=False)
    assert after == before


@pytest.mark.parametrize("kind", KINDS)
def test_opt_in_binds_exact_current_role_route_and_same_table_readiness(tmp_path, kind):
    template = _template(tmp_path, kind, **_options(kind))
    (callback_id, callback), (role_id, _), (table_id, table) = _bindings(template)
    # AsyncTrainingStack carries the validated literal; RuntimeStack reads its own runtime resource.
    runtime_ref = (
        RUNTIME_ARN
        if kind is AsyncTrainingStack
        else {
            "Fn::GetAtt": [
                next(iter(_of_type(template, "AWS::BedrockAgentCore::Runtime"))),
                "AgentRuntimeArn",
            ]
        }
    )
    assert callback["Role"] == {"Fn::GetAtt": [role_id, "Arn"]}
    assert callback["Environment"]["Variables"] == {
        "AGENTCORE_RUNTIME_ARN": runtime_ref,
        "JOBS_TABLE": {"Ref": table_id},
        "AMLAS_CAMPAIGN_CALLBACK_MODE": "1",
        "AMLAS_AWS_ACCOUNT_ID": ACCOUNT,
        "AMLAS_DEPLOYMENT_ID": DEPLOYMENT,
        "AGENTCORE_RUNTIME_QUALIFIER": QUALIFIER,
        "AGENTCORE_RUNTIME_VERSION": VERSION,
        "AMLAS_CALLBACK_ROUTE_SHA256": ROUTE_SHA,
        "AMLAS_CALLBACK_ROLE_ARN": {"Fn::GetAtt": [role_id, "Arn"]},
        "AMLAS_CALLBACK_ROLE_ID": {"Fn::GetAtt": [role_id, "RoleId"]},
    }
    if kind is AsyncTrainingStack:
        # Declared for stored and passed values only; no consumer reads it (pinned below).
        assert template["Parameters"][RUNTIME_PARAMETER]["Default"] == RUNTIME_ARN
    assert table["StreamSpecification"] == {"StreamViewType": "NEW_AND_OLD_IMAGES"}
    mappings = _of_type(template, "AWS::Lambda::EventSourceMapping")
    assert len(mappings) == 1
    mapping = next(iter(mappings.values()))
    assert mapping["EventSourceArn"] == {"Fn::GetAtt": [table_id, "StreamArn"]}
    assert mapping["FunctionName"] in ({"Ref": callback_id}, {"Fn::GetAtt": [callback_id, "Arn"]})
    assert mapping["StartingPosition"] == "LATEST"
    assert mapping["BatchSize"] == 1
    assert mapping.get("Enabled", True) is True
    assert not mapping.get("FunctionResponseTypes")
    assert not mapping.get("FilterCriteria")
    statements = _statements(template, role_id)
    endpoint = RUNTIME_ARN + "/runtime-endpoint/" + QUALIFIER
    for action, expected in (
        ("bedrock-agentcore:GetAgentRuntime", {RUNTIME_ARN}),
        ("bedrock-agentcore:GetAgentRuntimeEndpoint", {RUNTIME_ARN, endpoint}),
        ("bedrock-agentcore:InvokeAgentRuntime", {RUNTIME_ARN, endpoint}),
    ):
        actual = set()
        for statement in statements:
            actions = statement["Action"]
            if action in ([actions] if isinstance(actions, str) else actions):
                resources = statement["Resource"]
                for resource in resources if isinstance(resources, list) else [resources]:
                    actual.add(_render_runtime(resource, runtime_ref))
        assert actual == expected
    for action in ("dynamodb:DescribeStream", "dynamodb:GetRecords", "dynamodb:GetShardIterator"):
        matching = [
            statement
            for statement in statements
            if action
            in ([statement["Action"]] if isinstance(statement["Action"], str) else statement["Action"])
        ]
        assert matching
        for statement in matching:
            resources = statement["Resource"]
            assert (resources if isinstance(resources, list) else [resources]) == [
                {"Fn::GetAtt": [table_id, "StreamArn"]}
            ]
    rules = list(_of_type(template, "AWS::Events::Rule").values())
    assert len(rules) == 2
    assert {rule["EventPattern"]["source"][0] for rule in rules} == {"aws.sagemaker"}
    assert {rule["EventPattern"]["detail-type"][0] for rule in rules} == {
        "SageMaker Training Job State Change",
        "SageMaker Processing Job State Change",
    }
    assert rules[0]["Targets"] == rules[1]["Targets"]
    assert rules[0]["Targets"][0]["Arn"] == {"Fn::GetAtt": [callback_id, "Arn"]}


def test_async_opt_in_binds_validated_literal_runtime_and_declares_unread_parameter(tmp_path):
    template = _template(tmp_path, AsyncTrainingStack, **_options(AsyncTrainingStack))
    (_, callback), (role_id, _), _ = _bindings(template)
    endpoint = RUNTIME_ARN + "/runtime-endpoint/" + QUALIFIER
    assert callback["Environment"]["Variables"]["AGENTCORE_RUNTIME_ARN"] == RUNTIME_ARN
    invoke = [row for row in _statements(template, role_id) if row.get("Sid") == "InvokeAgentRuntime"]
    assert len(invoke) == 1
    assert invoke[0]["Action"] == "bedrock-agentcore:InvokeAgentRuntime"
    assert invoke[0]["Resource"] == [RUNTIME_ARN, endpoint]
    assert _action_resources(template, role_id, "bedrock-agentcore:GetAgentRuntime") == [RUNTIME_ARN]
    assert _action_resources(template, role_id, "bedrock-agentcore:GetAgentRuntimeEndpoint") == [
        RUNTIME_ARN,
        endpoint,
    ]
    body = {key: value for key, value in template.items() if key != "Parameters"}
    assert _parameter_reads(body, RUNTIME_PARAMETER) == []
    assert "RuntimeArnSupplied" not in template.get("Conditions", {})
    # The declaration from before the literal binding: String, the literal as default, same text.
    assert template["Parameters"][RUNTIME_PARAMETER] == {
        "Type": "String",
        "Default": RUNTIME_ARN,
        "Description": RUNTIME_PARAMETER_DESCRIPTION,
    }


def test_async_opt_in_on_existing_stack_does_not_read_the_stored_runtime_parameter(tmp_path):
    # An update that passes no value reuses the value stored on the stack. A stack created
    # before the opt-in stores AgentRuntimeArn="" whatever default the new template declares.
    template = _template(tmp_path, AsyncTrainingStack, **_options(AsyncTrainingStack))
    stored = _stored_runtime_parameter(template, stack={RUNTIME_PARAMETER: ""})
    assert stored == {RUNTIME_PARAMETER: ""}
    updated = dict(template, Resources=_with_stored_parameters(template["Resources"], stored))
    (_, callback), (role_id, _), _ = _bindings(updated)
    for action in (
        "bedrock-agentcore:InvokeAgentRuntime",
        "bedrock-agentcore:GetAgentRuntime",
        "bedrock-agentcore:GetAgentRuntimeEndpoint",
    ):
        resources = _action_resources(updated, role_id, action)
        assert resources
        for resource in resources:
            # IAM refuses an empty Resource: "Resource must be in ARN format or *".
            assert resource in (RUNTIME_ARN, RUNTIME_ARN + "/runtime-endpoint/" + QUALIFIER), (
                action,
                resource,
            )
    assert callback["Environment"]["Variables"]["AGENTCORE_RUNTIME_ARN"] == RUNTIME_ARN


def test_async_opt_in_resolves_the_same_for_any_stored_or_passed_runtime_parameter(tmp_path):
    # A deploy command that still passes AgentRuntimeArn names a declared parameter, so
    # CloudFormation accepts it; neither that value nor a stored one reaches any resource.
    template = _template(tmp_path, AsyncTrainingStack, **_options(AsyncTrainingStack))
    unresolved = _deployed(template, {})
    for stack, passed in (
        ({RUNTIME_PARAMETER: ""}, None),
        ({RUNTIME_PARAMETER: EARLIER_RUNTIME_ARN}, None),
        ({RUNTIME_PARAMETER: ""}, RUNTIME_ARN),
        (None, EARLIER_RUNTIME_ARN),
    ):
        stored = _stored_runtime_parameter(template, stack, passed)
        assert stored == {RUNTIME_PARAMETER: stack[RUNTIME_PARAMETER] if passed is None else passed}
        assert _deployed(template, stored) == unresolved, (stack, passed)


def test_async_opt_out_after_opt_in_reads_the_runtime_parameter_the_opt_in_kept(tmp_path):
    # An opt-in deploy, then a deploy with the opt-in off. The legacy template reads the stored
    # parameter, so the opt-in template has to keep it: a stack that drops it comes back with
    # the empty default and the grant on every runtime of the account and Region.
    enabled = _template(tmp_path, AsyncTrainingStack, **_options(AsyncTrainingStack))
    legacy = _template(tmp_path, AsyncTrainingStack)
    for stack, passed, expected in (
        (None, None, RUNTIME_ARN),  # created by the opt-in deploy: the default seeds the value
        ({RUNTIME_PARAMETER: EARLIER_RUNTIME_ARN}, None, EARLIER_RUNTIME_ARN),
        ({RUNTIME_PARAMETER: ""}, RUNTIME_ARN, RUNTIME_ARN),  # the deploy command still passes it
        ({RUNTIME_PARAMETER: ""}, None, RUNTIME_FALLBACK),  # never had a runtime: as before the opt-in
    ):
        case = (stack, passed)
        after_opt_in = _stored_runtime_parameter(enabled, stack, passed)
        after_opt_out = _stored_runtime_parameter(legacy, after_opt_in)
        deployed = _deployed(legacy, after_opt_out)
        (_, callback), (role_id, _), _ = _bindings(deployed)
        invoke, get_runtime, get_endpoint = (
            _action_resources(deployed, role_id, action) for action in RUNTIME_ACTIONS
        )
        assert invoke == [expected], case
        assert get_runtime == [], case
        assert get_endpoint == [], case
        assert callback["Environment"]["Variables"]["AGENTCORE_RUNTIME_ARN"] == (
            "" if expected == RUNTIME_FALLBACK else expected
        ), case
        if stack is not None and passed is None:
            # Where the same stack ends when it deploys the legacy template without any opt-in.
            assert deployed == _deployed(legacy, _stored_runtime_parameter(legacy, stack)), case


def test_async_legacy_template_keeps_the_runtime_parameter_route(tmp_path):
    template = _template(tmp_path, AsyncTrainingStack)
    (_, callback), (role_id, _), _ = _bindings(template)
    assert template["Parameters"][RUNTIME_PARAMETER] == {
        "Type": "String",
        "Default": "",
        "Description": RUNTIME_PARAMETER_DESCRIPTION,
    }
    assert callback["Environment"]["Variables"]["AGENTCORE_RUNTIME_ARN"] == {"Ref": "AgentRuntimeArn"}
    assert template["Conditions"]["RuntimeArnSupplied"] == {
        "Fn::Not": [{"Fn::Equals": [{"Ref": "AgentRuntimeArn"}, ""]}]
    }
    assert _action_resources(template, role_id, "bedrock-agentcore:InvokeAgentRuntime") == [
        {
            "Fn::If": [
                "RuntimeArnSupplied",
                {"Ref": "AgentRuntimeArn"},
                f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/*",
            ]
        }
    ]
    assert _action_resources(template, role_id, "bedrock-agentcore:GetAgentRuntime") == []
    assert _action_resources(template, role_id, "bedrock-agentcore:GetAgentRuntimeEndpoint") == []


@pytest.mark.parametrize("kind", KINDS)
def test_normal_callback_requires_complete_validated_route_before_synthesis(tmp_path, kind):
    for key, bad in (
        ("normal_callback_deployment_id", ""),
        ("normal_callback_deployment_id", "not-a-uuid"),
        ("normal_callback_qualifier", ""),
        ("normal_callback_qualifier", "foreign-endpoint"),
        ("normal_callback_runtime_version", ""),
        ("normal_callback_runtime_version", "0"),
        ("normal_callback_route_sha256", ""),
        ("normal_callback_route_sha256", "z" * 64),
    ):
        options = _options(kind)
        options[key] = bad
        with pytest.raises(ValueError):
            _template(tmp_path, kind, **options)


def test_async_opt_in_requires_literal_runtime_in_selected_account_and_region(tmp_path):
    for bad in (
        "",
        "unresolved-runtime",
        RUNTIME_ARN.replace(ACCOUNT, "4" * 4 + "5" * 4 + "6" * 4),
        RUNTIME_ARN.replace(REGION, "us-east-1"),
    ):
        options = _options(AsyncTrainingStack)
        options["normal_callback_runtime_arn"] = bad
        with pytest.raises(ValueError):
            _template(tmp_path, AsyncTrainingStack, **options)


@pytest.mark.parametrize("kind", KINDS)
def test_constructor_switch_rejects_non_boolean_values(tmp_path, kind):
    for bad in (1, "true", None):
        with pytest.raises(ValueError):
            _template(tmp_path, kind, normal_callback_enabled=bad)


@pytest.mark.parametrize("kind", KINDS)
def test_normal_callback_namespaces_keep_existing_table_role_and_function_isolated(tmp_path, kind):
    bindings = [
        _bindings(_template(tmp_path, kind, prefix, **_options(kind))) for prefix in ("trial-a", "trial-b")
    ]
    for index, field in ((0, "FunctionName"), (1, "RoleName"), (2, "TableName")):
        assert [binding[index][1][field] for binding in bindings] == [
            "trial-a"
            + (
                "-training-callback"
                if index == 0
                else "-training-jobs"
                if index == 2
                else "-training-callback-role"
                if kind is AsyncTrainingStack
                else "-callback-lambda"
            ),
            "trial-b"
            + (
                "-training-callback"
                if index == 0
                else "-training-jobs"
                if index == 2
                else "-training-callback-role"
                if kind is AsyncTrainingStack
                else "-callback-lambda"
            ),
        ]


def _app(tmp_path, switch):
    out = tmp_path / str(uuid4())
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AWS_", "AMLAS_", "AGENTICML_", "AGENTCORE_"))
    }
    env.update(
        AMLAS_NORMAL_CALLBACK_ENABLED=switch,
        AMLAS_DEPLOYMENT_PREFIX="trial-a",
        AMLAS_DEPLOYMENT_ID=DEPLOYMENT,
        AGENTCORE_RUNTIME_ARN=RUNTIME_ARN,
        AGENTCORE_RUNTIME_QUALIFIER=QUALIFIER,
        AGENTCORE_RUNTIME_VERSION=VERSION,
        AMLAS_CALLBACK_ROUTE_SHA256=ROUTE_SHA,
        CDK_DEFAULT_ACCOUNT=ACCOUNT,
        CDK_DEFAULT_REGION=REGION,
        CDK_OUTDIR=str(out),
        AWS_EC2_METADATA_DISABLED="true",
        CDK_CONTEXT_JSON=json.dumps(
            {f"availability-zones:account={ACCOUNT}:region={REGION}": [REGION + "a", REGION + "b"]}
        ),
    )
    result = subprocess.run(
        [sys.executable, "-m", "infra.app"], cwd=ROOT, env=env, capture_output=True, timeout=180, check=False
    )
    templates = {}
    if result.returncode == 0:
        manifest = json.loads((out / "manifest.json").read_text())
        templates = {
            key: json.loads((out / value["properties"]["templateFile"]).read_text())
            for key, value in manifest["artifacts"].items()
            if value["type"] == "aws:cloudformation:stack"
        }
    return result.returncode, templates


def test_real_cdk_app_routes_true_opt_in_to_both_existing_stacks(tmp_path):
    code, templates = _app(tmp_path, "true")
    assert code == 0
    for name in ("AsyncTrainingStack", "RuntimeStack"):
        (_, callback), _, (_, table) = _bindings(templates[name])
        assert callback["Environment"]["Variables"]["AMLAS_CAMPAIGN_CALLBACK_MODE"] == "1"
        assert callback["Environment"]["Variables"]["AMLAS_CALLBACK_ROUTE_SHA256"] == ROUTE_SHA
        assert table["StreamSpecification"]["StreamViewType"] == "NEW_AND_OLD_IMAGES"
        assert len(_of_type(templates[name], "AWS::Lambda::EventSourceMapping")) == 1
    asynchronous = templates["AsyncTrainingStack"]
    (_, callback), (role_id, _), _ = _bindings(asynchronous)
    assert callback["Environment"]["Variables"]["AGENTCORE_RUNTIME_ARN"] == RUNTIME_ARN
    assert _action_resources(asynchronous, role_id, "bedrock-agentcore:InvokeAgentRuntime") == [
        RUNTIME_ARN,
        RUNTIME_ARN + "/runtime-endpoint/" + QUALIFIER,
    ]
    body = {key: value for key, value in asynchronous.items() if key != "Parameters"}
    assert _parameter_reads(body, RUNTIME_PARAMETER) == []
    assert asynchronous["Parameters"][RUNTIME_PARAMETER]["Default"] == RUNTIME_ARN


def test_real_cdk_app_false_keeps_legacy_route_even_with_complete_selectors(tmp_path):
    code, templates = _app(tmp_path, "false")
    assert code == 0
    for name in ("AsyncTrainingStack", "RuntimeStack"):
        (_, callback), _, (_, table) = _bindings(templates[name])
        assert set(callback["Environment"]["Variables"]) == {"AGENTCORE_RUNTIME_ARN", "JOBS_TABLE"}
        assert "StreamSpecification" not in table
        assert _of_type(templates[name], "AWS::Lambda::EventSourceMapping") == {}


def test_real_cdk_app_unknown_switch_refuses_before_synthesis(tmp_path):
    code, templates = _app(tmp_path, "yes")
    assert code != 0
    assert templates == {}
