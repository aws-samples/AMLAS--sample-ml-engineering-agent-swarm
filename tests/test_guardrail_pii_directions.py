"""Synthesize actual guardrail directions; offline Source evidence, no policy selection."""

from aws_cdk import App, Environment
from aws_cdk.assertions import Template

from infra.cdk_stacks.guardrail_stack import GuardrailStack

ENTITIES = (
    "EMAIL",
    "PHONE",
    "NAME",
    "US_SOCIAL_SECURITY_NUMBER",
    "CREDIT_DEBIT_CARD_NUMBER",
    "US_BANK_ACCOUNT_NUMBER",
    "IP_ADDRESS",
)
CONTENT = ("SEXUAL", "VIOLENCE", "HATE", "INSULTS", "MISCONDUCT")


def test_guardrail_pii_both_directions_explicit_preserves_high_and_legacy_action(
    tmp_path,
):
    app = App(outdir=str(tmp_path / "guardrail-synth"))
    stack = GuardrailStack(
        app,
        "GuardrailStack",
        env=Environment(account="111122223333", region="us-west-2"),
    )
    template = Template.from_stack(stack).to_json()
    parameters = template["Parameters"]
    assert parameters["ContentFilterStrength"]["Default"] == "HIGH"
    assert parameters["ContentFilterStrength"]["AllowedValues"] == [
        "NONE",
        "LOW",
        "MEDIUM",
        "HIGH",
    ]
    assert parameters["PiiDefaultAction"]["Default"] == "ANONYMIZE"
    assert parameters["PiiDefaultAction"]["AllowedValues"] == ["ANONYMIZE", "BLOCK"]
    resources = [
        value["Properties"]
        for value in template["Resources"].values()
        if value["Type"] == "AWS::Bedrock::Guardrail"
    ]
    assert len(resources) == 1
    properties = resources[0]
    strength, pii_action = {"Ref": "ContentFilterStrength"}, {"Ref": "PiiDefaultAction"}
    assert properties["ContentPolicyConfig"]["FiltersConfig"] == [
        {"Type": kind, "InputStrength": strength, "OutputStrength": strength} for kind in CONTENT
    ] + [{"Type": "PROMPT_ATTACK", "InputStrength": strength, "OutputStrength": "NONE"}]
    assert properties["BlockedInputMessaging"] == {"Ref": "BlockedInputMessage"}
    assert properties["BlockedOutputsMessaging"] == {"Ref": "BlockedOutputMessage"}
    versions = [
        value["Properties"]["Value"]
        for value in template["Resources"].values()
        if value["Type"] == "AWS::SSM::Parameter"
        and value["Properties"]["Name"].endswith("/guardrail-version")
    ]
    assert versions == [{"Fn::GetAtt": ["Guardrail", "Version"]}]
    # One exact template oracle retains all seven legacy type/action entries and
    # requires protection explicitly on both directions. Defaults were omitted,
    # not asserted disabled; this test does not invoke/select a native policy.
    assert properties["SensitiveInformationPolicyConfig"]["PiiEntitiesConfig"] == [
        {
            "Type": entity,
            "Action": pii_action,
            "InputEnabled": True,
            "OutputEnabled": True,
            "InputAction": pii_action,
            "OutputAction": pii_action,
        }
        for entity in ENTITIES
    ]
