"""Managed runtime environment budget; no AWS calls or credentials.

AgentCore refuses a runtime environment whose key and value lengths sum to more
than 4,000 bytes (UpdateAgentRuntime answered "Environment variables exceed the
maximum size of 4000 bytes. Current size: 4097 bytes" for 48 keys whose lengths
sum to exactly 4,097). The full selection of a fresh deployment is larger than
that, so the control client sends the closed stage-pin family as one lossless
pack, and the image expands it at package import before any reader runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess  # nosec B404 - starts this test's own interpreter
import sys
from pathlib import Path

import pytest
from test_runtime_deployment_advisory import run_runtime, runtime_calls
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_runtime_ownership_controller import MODULE as CONTROL
from test_runtime_ownership_controller import ClosedRuntimeService
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

ROOT = Path(__file__).resolve().parents[1]
PACKER_PATH = ROOT / "src/amlas/_runtime_env_pack.py"
ACCOUNT = "111122223333"  # AWS documentation account; never a deployment target.
LIMIT = 4000
TARGET = 3600
PACK_KEY = "AMLAS_RUNTIME_ENV_PACK"
PACK_SHA256_KEY = "AMLAS_RUNTIME_ENV_PACK_SHA256"
SWITCH = "AMLAS_RUNTIME_IMAGE_EXPANDS_ENV_PACK"
# The approved-action value the previous managed runtime carried, byte for byte.
DEPLOYED_APPROVED_ACTIONS = (
    '["acquire_huggingface_dataset","acquire_model_weights","coding_case_freeze_flow",'
    '"coding_generation_flow","coding_pair_flow","data_collection_flow","data_preparation_flow",'
    '"freeze_coding_cases","inspect_acquired_dataset","prepare_coding_dataset","qwen_sft_flow",'
    '"reload_qwen_sft","resolve_coding_evaluation_spec","run_processing_job",'
    '"submit_coding_generation","submit_coding_pair","submit_qwen_sft"]'
)
RESEARCH_VALUES = {"AMLAS_RUNTIME_RESEARCH_PROFILE": "qwen9b_core_v1", "AMLAS_GUARDRAIL_ENABLED": "false"}


def agentcore_bytes(environment: dict[str, str]) -> int:
    """The service's measure: the sum of key and value lengths (UTF-8 bytes)."""
    return sum(len(key.encode("utf-8")) + len(value.encode("utf-8")) for key, value in environment.items())


def packer():
    spec = importlib.util.spec_from_file_location("budget_runtime_env_pack", PACKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hexed(pattern: str, length: int) -> str:
    """Letters-only hex fixtures keep digit runs out of the public tree."""
    return (pattern * length)[:length]


def stage_pins(account: str, region: str) -> dict[str, str]:
    repository = f"{account}.dkr.ecr.{region}.amazonaws.com/amlas-pilot-001-runtime"
    stages = {
        "DATA_ACQUISITION": ("ab", "abcdef", None),
        "DATA_PREPARATION": ("cd", "bcdefa", True),
        "CODING_PAIR": ("ef", "cdefab", True),
        "CODING_GENERATION": ("ba", "defabc", True),
        "CODING_CASE_FREEZE": ("dc", "efabcd", True),
        "QWEN_SFT": ("ba", "defabc", True),
        "MODEL_WEIGHT_ACQUISITION": ("fe", "fabcde", True),
    }
    pins = {}
    for stage, (digest_pattern, commit_pattern, admitted) in stages.items():
        digest = hexed(digest_pattern, 64)
        pins[f"AMLAS_{stage}_IMAGE_URI"] = f"{repository}@sha256:{digest}"
        pins[f"AMLAS_{stage}_SOURCE_COMMIT"] = hexed(commit_pattern, 40)
        if admitted:
            pins[f"AMLAS_{stage}_ADMISSION_DIGEST"] = digest
    pins["AMLAS_CODING_PAIR_LAUNCHER_SHA256"] = hexed("fd", 64)
    return pins


def exclusions(bucket: str) -> dict[str, str]:
    return {
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_URI": (
            f"s3://{bucket}/deployment/preparation/governed-exclusions/fixture-release/EXCLUSIONS.json"
        ),
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID": "ExampleVersionIdentifierAbCdEfGh",
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_SHA256": hexed("ae", 64),
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_BYTE_COUNT": "2097152",
    }


def fresh_deployment_environment() -> dict[str, str]:
    """The selection of a deployed test stage, with the value lengths of the measured deployment.

    It is the deploy-script environment (the 46 keys read back from the
    rehearsal runtime plus the two lake prefixes), the two research values
    and all seven preparation values.
    """
    region = "us-west-2"
    bucket = "amlas-demo-asynctrainingstac-sessionbucketfixture-examplebucket"
    lake = f"amlas-demo-lake-{ACCOUNT}-{region}"
    environment = {
        "ETL_BACKEND": "processing",
        "DQ_BACKEND": "processing",
        "SAGEMAKER_EXECUTION_ROLE_ARN": f"arn:aws:iam::{ACCOUNT}:role/amlas-demo-sagemaker-exec",
        "AMLAS_RUNTIME_EXEC_ROLE_ARN": f"arn:aws:iam::{ACCOUNT}:role/amlas-demo-runtime-exec-{region}",
        "AMLAS_RUNTIME_EXEC_ROLE_ID": "role-id-example-abcde",  # 21 characters, as a native role ID
        "AMLAS_AWS_REGION": region,
        "AMLAS_AWS_ACCOUNT_ID": ACCOUNT,
        "AMLAS_SESSION_BUCKET": bucket,
        "AMLAS_DEPLOYMENT_PREFIX": "amlas-demo",
        "AMLAS_SSM_PREFIX": "/amlas-demo",
        "AMLAS_JOBS_TABLE": "amlas-demo-training-jobs",
        "AMLAS_LAKE_BUCKET": lake,
        "AMLAS_MODEL_PACKAGE_GROUP": "amlas-demo-models",
        "AMLAS_FEATURE_STORE_DB": "amlas_demo_features",
        "AMLAS_LAKE_SILVER_PREFIX": f"s3://{lake}/silver/",
        "AMLAS_LAKE_GOLD_PREFIX": f"s3://{lake}/gold/",
        "AMLAS_GUARDRAIL_ID": "exampleguard",
        "AMLAS_GUARDRAIL_VERSION": "DRAFT",
        "AMLAS_TAVILY_SECRET_ID": "amlas-pilot-001/tavily-api-key",
        "AMLAS_DEPLOYMENT_ID": "abcdefab-cdef-4abc-8def-abcdefabcdef",
        "AMLAS_CHAT_MODEL": "global.anthropic.claude-sonnet-5-5",
        "AMLAS_MODEL_PLANNER": "global.anthropic.claude-sonnet-5-5",
        "AMLAS_MODEL_REVIEWER": "us.openai.gpt-6.1-sol",
        "AMLAS_MODEL_FINETUNE": "global.anthropic.claude-sonnet-5-5",
        "AMLAS_RUNTIME_APPROVED_ACTIONS": DEPLOYED_APPROVED_ACTIONS,
        **stage_pins(ACCOUNT, region),
        **RESEARCH_VALUES,
        **exclusions(bucket),
    }
    return environment


def control_options(environment: dict[str, str]) -> argparse.Namespace:
    return argparse.Namespace(
        region="us-east-1",
        name="owned_runtime",
        image="example-image",
        role="arn:aws:iam::" + ACCOUNT + ":role/example",
        network_mode="PUBLIC",
        protocol="HTTP",
        environment_json=json.dumps(environment),
        platform_version="1.0.0",
        tags_json="{}",
    )


def sent_environments(service: ClosedRuntimeService) -> list[dict[str, str]]:
    return [
        payload["environmentVariables"]
        for operation, payload in service.requests
        if operation in ("create-agent-runtime", "update-agent-runtime")
    ]


def deploy(environment: dict[str, str]) -> ClosedRuntimeService:
    service = ClosedRuntimeService()
    control = CONTROL.RuntimeControl("us-east-1")
    control.call = service.call
    control.deploy(control_options(environment))
    return service


def largest_keys(environment: dict[str, str]) -> list[str]:
    return sorted(environment, key=lambda key: (-(len(key) + len(environment[key])), key))[:3]


def with_digest(text: str) -> dict[str, str]:
    return {PACK_KEY: text, PACK_SHA256_KEY: hashlib.sha256(text.encode()).hexdigest()}


def test_fresh_deployment_selection_is_over_the_service_limit_today():
    """The measured problem: the logical selection cannot be sent as plain keys."""
    environment = fresh_deployment_environment()
    assert len(environment) == 52
    assert agentcore_bytes(environment) > LIMIT


def test_control_client_sends_the_full_selection_within_the_budget(monkeypatch):
    monkeypatch.setenv("AMLAS_RUNTIME_POLL_SECONDS", "0")
    monkeypatch.setenv(SWITCH, "1")
    environment = fresh_deployment_environment()
    service = deploy(environment)
    sent = sent_environments(service)
    sizes = [agentcore_bytes(item) for item in sent]
    assert sent and all(size <= TARGET for size in sizes), (
        f"sent {sizes} bytes; logical selection {agentcore_bytes(environment)} bytes; target {TARGET}"
    )
    assert all(packer().unpack(item) == environment for item in sent)
    assert all(set(item) & {PACK_KEY, PACK_SHA256_KEY} == {PACK_KEY, PACK_SHA256_KEY} for item in sent)


def test_environment_over_the_limit_after_packing_is_refused_before_any_call(monkeypatch):
    monkeypatch.setenv("AMLAS_RUNTIME_POLL_SECONDS", "0")
    monkeypatch.setenv(SWITCH, "1")
    environment = dict(fresh_deployment_environment(), AMLAS_AGENT_HTTP_ALLOW_HOSTS="h" * 3000)
    service = ClosedRuntimeService()
    control = CONTROL.RuntimeControl("us-east-1")
    control.call = service.call
    with pytest.raises(CONTROL.RuntimeDeployError) as refused:
        control.deploy(control_options(environment))
    message = str(refused.value)
    packed = packer().pack(environment)
    assert f"{agentcore_bytes(packed)} bytes" in message and f"at most {LIMIT} bytes" in message
    assert all(key in message for key in largest_keys(packed))
    assert "h" * 50 not in message, "values never appear in the refusal"
    assert service.requests == []


def test_over_the_limit_needs_the_image_switch_before_any_call(monkeypatch):
    """An image built before the expander ignores a pack; packing needs the operator's statement."""
    monkeypatch.setenv("AMLAS_RUNTIME_POLL_SECONDS", "0")
    monkeypatch.delenv(SWITCH, raising=False)
    environment = fresh_deployment_environment()
    service = ClosedRuntimeService()
    control = CONTROL.RuntimeControl("us-east-1")
    control.call = service.call
    with pytest.raises(CONTROL.RuntimeDeployError) as refused:
        control.deploy(control_options(environment))
    message = str(refused.value)
    assert f"{agentcore_bytes(environment)} bytes" in message and f"at most {LIMIT} bytes" in message
    assert all(key in message for key in largest_keys(environment)) and SWITCH in message
    assert f"{agentcore_bytes(packer().pack(environment))}-byte pack" in message
    assert service.requests == []


def test_environment_within_the_limit_is_sent_unchanged(monkeypatch):
    """A selection that fits keeps today's plain keys, so any image and any rollback still run it."""
    monkeypatch.setenv("AMLAS_RUNTIME_POLL_SECONDS", "0")
    monkeypatch.setenv(SWITCH, "1")
    environment = {
        key: value for key, value in fresh_deployment_environment().items() if "EXCLUSIONS" not in key
    }
    environment.pop("AMLAS_RUNTIME_APPROVED_ACTIONS")
    assert agentcore_bytes(environment) <= LIMIT
    sent = sent_environments(deploy(environment))
    assert sent and all(item == environment for item in sent)


def product_selection(account: str, region: str) -> dict[str, str]:
    return {
        **stage_pins(account, region),
        **exclusions("amlas-fixture-session-bucket"),
        "AMLAS_RUNTIME_APPROVED_ACTIONS": DEPLOYED_APPROVED_ACTIONS,
        "AMLAS_CHAT_MODEL": "global.anthropic.claude-sonnet-5-5",
        "AMLAS_MODEL_REVIEWER": "us.openai.gpt-6.1-sol",
        "AMLAS_MODEL_FINETUNE": "global.anthropic.claude-sonnet-5-5",
        **RESEARCH_VALUES,
    }


def copy_packer(repo: Path) -> None:
    """The fixture workspace copies scripts/ only; the deploy also needs the product's packer."""
    if PACKER_PATH.exists():
        (repo / "src/amlas").mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKER_PATH, repo / "src/amlas" / PACKER_PATH.name)


def test_product_deploy_script_sends_the_full_selection_within_the_budget(runtime_workspace):
    repo, env, _ = runtime_workspace
    copy_packer(repo)
    selected = product_selection(ACCOUNT, "us-east-1")
    env.update(selected, AGENTICML_MODEL_PLANNER="global.anthropic.claude-sonnet-5-5")
    env[SWITCH] = "1"
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(Path(env["TEST_STATE"]).read_text())["environmentVariables"]
    pending = {key: value for key, value in RESEARCH_VALUES.items() if key not in packer_view(native)}
    assert pending == {}, "the deploy builder passes both research values through"
    assert agentcore_bytes(native) <= TARGET, f"deploy sent {agentcore_bytes(native)} bytes"
    expanded = packer().unpack(native)
    assert all(expanded.get(key) == value for key, value in selected.items())
    assert f"RUNTIME_ENVIRONMENT_BYTES={agentcore_bytes(native)}" in result.stdout
    assert "RUNTIME_ENVIRONMENT_PACKED=1" in result.stdout


@pytest.mark.parametrize("switch", ["1", ""])
def test_product_deploy_script_refuses_an_oversized_environment_before_any_runtime_call(
    runtime_workspace, switch
):
    repo, env, _ = runtime_workspace
    copy_packer(repo)
    env.update(
        product_selection(ACCOUNT, "us-east-1"), AMLAS_AGENT_HTTP_ALLOW_HOSTS="h" * (3000 if switch else 1)
    )
    env[SWITCH] = switch
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0
    assert f"at most {LIMIT} bytes" in result.stderr and "Largest keys:" in result.stderr
    calls = runtime_calls(runtime_workspace)
    assert not any(
        call["args"][:2]
        in (
            ["bedrock-agentcore-control", "create-agent-runtime"],
            ["bedrock-agentcore-control", "update-agent-runtime"],
        )
        and "--generate-cli-skeleton" not in call["args"]
        for call in calls
    )
    assert "AGENTCORE_RUNTIME_DEPLOY_DONE" not in result.stdout


def packer_view(environment: dict[str, str]) -> dict[str, str]:
    return packer().unpack(environment) if PACK_KEY in environment else environment


def other_repository_image() -> str:
    return f"{ACCOUNT}.dkr.ecr.us-west-2.amazonaws.com/another-runtime@sha256:" + hexed("ca", 64)


@pytest.mark.parametrize(
    "case",
    [
        "full",
        "triple_only",
        "empty_values",
        "other_repository",
        "inspection",
        "noncanonical_actions",
        "no_family",
    ],
)
def test_pack_round_trip_is_lossless(case):
    module = packer()
    environment = fresh_deployment_environment()
    if case == "triple_only":
        environment = {key: value for key, value in environment.items() if "EXCLUSIONS" not in key}
    if case == "empty_values":
        environment["AMLAS_DATA_PREPARATION_EXCLUSIONS_URI"] = ""
        environment["AMLAS_QWEN_SFT_SOURCE_COMMIT"] = ""
    if case == "other_repository":
        environment["AMLAS_CODING_PAIR_IMAGE_URI"] = other_repository_image()
    if case == "inspection":
        environment["AMLAS_DATA_INSPECTION_IMAGE_URI"] = other_repository_image()
        environment["AMLAS_DATA_INSPECTION_SOURCE_COMMIT"] = hexed("cafe", 40)
        environment["AMLAS_DATA_INSPECTION_ADMISSION_DIGEST"] = hexed("ca", 64)
    if case == "noncanonical_actions":
        environment["AMLAS_RUNTIME_APPROVED_ACTIONS"] = json.dumps(json.loads(DEPLOYED_APPROVED_ACTIONS))
    if case == "no_family":
        environment = {
            key: value
            for key, value in environment.items()
            if not module.is_packed_key(key) and key != "AMLAS_RUNTIME_APPROVED_ACTIONS"
        }
    packed = module.pack(environment)
    assert module.unpack(packed) == environment
    assert module.pack(environment) == packed, "packing is deterministic"
    if case == "no_family":
        assert packed == environment
    else:
        assert packed[PACK_SHA256_KEY] == hashlib.sha256(packed[PACK_KEY].encode()).hexdigest()
        assert not any(module.is_packed_key(key) for key in packed)
    if case == "noncanonical_actions":
        assert packed["AMLAS_RUNTIME_APPROVED_ACTIONS"] == environment["AMLAS_RUNTIME_APPROVED_ACTIONS"]


def selections_from(environment: dict[str, str], seed: int, count: int = 300):
    """Deterministic random sub-selections, including empty values and another repository."""
    chooser = random.Random(seed)  # nosec B311 - deterministic test sampling
    keys = sorted(environment)
    for _ in range(count):
        selected = {key: environment[key] for key in keys if chooser.random() < 0.6}
        for key in [key for key in selected if chooser.random() < 0.05]:
            selected[key] = ""
        if "AMLAS_CODING_PAIR_IMAGE_URI" in selected and chooser.random() < 0.2:
            selected["AMLAS_CODING_PAIR_IMAGE_URI"] = other_repository_image()
        yield selected


def default_deploy_selection(runtime_workspace) -> dict[str, str]:
    """The deploy builder's own output for the default fixture plus the stage pins it passes through."""
    repo, env, _ = runtime_workspace
    copy_packer(repo)
    env.update(stage_pins(ACCOUNT, "us-east-1"))
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(Path(env["TEST_STATE"]).read_text())["environmentVariables"]
    assert agentcore_bytes(native) <= LIMIT and PACK_KEY not in native, "a selection that fits is sent plain"
    return native


def test_expansion_matches_the_plain_selection_key_for_key(runtime_workspace):
    module = packer()
    sources = {
        "amlas-demo": fresh_deployment_environment(),
        "default-deploy": default_deploy_selection(runtime_workspace),
    }
    checked = 0
    for name, environment in sources.items():
        for index, selected in enumerate([environment, *selections_from(environment, seed=len(name))]):
            packed = module.pack(selected)
            target = dict(packed)
            result = module.expand_environ(target)
            assert target == selected, (name, index)
            assert module.expand_environ(target) is None and target == selected, (
                "a second expansion changes nothing"
            )
            if result is not None:
                assert result == (len(selected) - len(packed) + 2, packed[PACK_SHA256_KEY])
            checked += 1
    assert checked == 602


def test_approved_action_profile_is_frozen_to_the_deployed_list(monkeypatch):
    from amlas import app

    module = packer()
    assert module.APPROVED_ACTION_PROFILES == {"campaign_v1": tuple(json.loads(DEPLOYED_APPROVED_ACTIONS))}
    assert set(module.APPROVED_ACTION_PROFILES["campaign_v1"]) == set(app._CAMPAIGN_APPROVAL_ACTIONS)
    packed = module.pack({"AMLAS_RUNTIME_APPROVED_ACTIONS": DEPLOYED_APPROVED_ACTIONS})
    assert set(packed) == {PACK_KEY, PACK_SHA256_KEY} and json.loads(packed[PACK_KEY])["a"] == "campaign_v1"
    expanded = module.unpack(packed)
    assert expanded == {"AMLAS_RUNTIME_APPROVED_ACTIONS": DEPLOYED_APPROVED_ACTIONS}
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", DEPLOYED_APPROVED_ACTIONS)
    explicit = app._runtime_prompt_policy()
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", expanded["AMLAS_RUNTIME_APPROVED_ACTIONS"])
    restored = app._runtime_prompt_policy()
    assert (restored.mode, restored.allowlist, restored.max_auto_cost, restored.interactive) == (
        explicit.mode,
        explicit.allowlist,
        explicit.max_auto_cost,
        explicit.interactive,
    )


@pytest.mark.parametrize(
    "value",
    [
        '["acquire_model_weights"]',
        '["acquire_model_weights","not_an_action"]',
        '["acquire_model_weights","acquire_model_weights"]',
        "[]",
        "not json",
        '{"acquire_model_weights": true}',
        json.dumps(sorted(json.loads(DEPLOYED_APPROVED_ACTIONS), reverse=True)),
        DEPLOYED_APPROVED_ACTIONS,
    ],
)
def test_approval_decision_is_the_same_after_expansion(monkeypatch, value):
    """The enabled refusal sees exactly the value it saw before, so it decides the same."""
    from amlas import app

    def decide():
        try:
            policy = app._runtime_prompt_policy()
        except ValueError as error:
            return ("refused", str(error))
        return None if policy is None else (policy.mode, tuple(policy.allowlist), policy.interactive)

    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", value)
    plain = decide()
    target = packer().pack({"AMLAS_RUNTIME_APPROVED_ACTIONS": value})
    packer().expand_environ(target)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", target["AMLAS_RUNTIME_APPROVED_ACTIONS"])
    assert decide() == plain


def import_environment(tmp_path: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Start a fresh interpreter the way the container does: the package import comes first."""
    base = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src"), "HOME": str(tmp_path)}
    base.update(TMPDIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    probe = (
        "import json, os, amlas\n"
        "from amlas.config import env_loader\n"
        "print(json.dumps({'env': dict(os.environ), 'shell': sorted(env_loader._SHELL_KEYS)}, sort_keys=True))"
    )
    return subprocess.run(  # nosec B603 - this interpreter with a fixed probe
        [sys.executable, "-c", probe],
        env={**base, **environment},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_package_import_expands_the_pack_before_any_reader(tmp_path):
    environment = fresh_deployment_environment()
    packed = packer().pack(environment)
    result = import_environment(tmp_path, packed)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    seen = observed["env"]
    assert PACK_KEY not in seen and PACK_SHA256_KEY not in seen, "child processes see today's plain keys only"
    assert {key: seen.get(key) for key in environment} == environment
    # Expanded deployment values count as process values, so a later .env layer cannot override them.
    assert set(environment) <= set(observed["shell"])
    restored = len(environment) - len(packed) + 2
    line = f"RUNTIME_ENV_PACK_EXPANDED keys={restored} sha256={packed[PACK_SHA256_KEY]}"
    assert [text for text in result.stderr.splitlines() if text.startswith("RUNTIME_ENV_PACK")] == [line]
    assert not any(value in result.stderr for value in environment.values() if len(value) > 8)


MALFORMED = (
    "control_character",
    "digest_differs",
    "digest_missing",
    "digest_without_pack",
    "duplicate_key",
    "field_not_allowed",
    "not_a_string",
    "not_json",
    "plain_and_packed_differ",
    "plain_and_packed_equal",
    "short_image_without_repository",
    "unknown_field",
    "unknown_profile",
    "unknown_stage",
    "unknown_top_level",
    "wrong_version",
)


def malformed_packs() -> dict[str, dict[str, str]]:
    environment = fresh_deployment_environment()
    packed = packer().pack(environment)
    good = json.loads(packed[PACK_KEY])
    stage = next(iter(good["s"]))
    texts = {
        "not_json": "{",
        "duplicate_key": '{"v":1,"v":1}',
        "wrong_version": json.dumps(dict(good, v=2)),
        "unknown_stage": json.dumps(dict(good, s=dict(good["s"], NOT_A_STAGE={"c": hexed("ab", 40)}))),
        "unknown_field": json.dumps(dict(good, s=dict(good["s"], **{stage: dict(good["s"][stage], z="x")}))),
        "field_not_allowed": json.dumps(
            dict(good, s=dict(good["s"], QWEN_SFT=dict(good["s"]["QWEN_SFT"], l=hexed("fd", 64))))
        ),
        "unknown_profile": json.dumps(dict(good, a="campaign_v9")),
        "short_image_without_repository": json.dumps({k: v for k, v in good.items() if k != "r"}),
        "control_character": json.dumps(dict(good, r=good["r"] + "\n")),
        "not_a_string": json.dumps(dict(good, s=dict(good["s"], **{stage: dict(good["s"][stage], c=7)}))),
        "unknown_top_level": json.dumps(dict(good, extra=1)),
    }
    cases = {name: with_digest(text) for name, text in texts.items()}
    cases["digest_differs"] = dict(packed, **{PACK_SHA256_KEY: hexed("ab", 64)})
    cases["digest_missing"] = {PACK_KEY: packed[PACK_KEY]}
    cases["digest_without_pack"] = {PACK_SHA256_KEY: packed[PACK_SHA256_KEY]}
    commit = environment["AMLAS_QWEN_SFT_SOURCE_COMMIT"]
    cases["plain_and_packed_equal"] = dict(packed, AMLAS_QWEN_SFT_SOURCE_COMMIT=commit)
    cases["plain_and_packed_differ"] = dict(packed, AMLAS_QWEN_SFT_SOURCE_COMMIT=hexed("ff", 40))
    assert set(cases) == set(MALFORMED)
    return cases


@pytest.mark.parametrize("case", MALFORMED)
def test_malformed_pack_is_refused_and_nothing_changes(tmp_path, case):
    module = packer()
    selected = malformed_packs()[case]
    with pytest.raises(ValueError):
        module.unpack(selected)
    target = dict(selected)
    with pytest.raises(ValueError):
        module.expand_environ(target)
    assert target == selected, "a refusal leaves the environment unchanged"
    result = import_environment(tmp_path, selected)
    assert result.returncode != 0
    assert "AMLAS_RUNTIME_ENV_PACK refused" in result.stderr
    assert hexed("ab", 40) not in result.stderr and ACCOUNT not in result.stderr


def test_pack_refuses_an_environment_that_already_carries_a_pack():
    module = packer()
    packed = module.pack(fresh_deployment_environment())
    for key in (PACK_KEY, PACK_SHA256_KEY):
        with pytest.raises(ValueError):
            module.pack(dict(fresh_deployment_environment(), **{key: packed[key]}))


def test_pack_key_from_a_dotenv_file_is_refused(monkeypatch):
    """Only the process environment may carry a pack; a later .env layer cannot add one."""
    import amlas

    monkeypatch.delenv(PACK_KEY, raising=False)
    monkeypatch.setattr(amlas, "_load_dotenv", lambda: os.environ.__setitem__(PACK_KEY, "{}"))
    try:
        with pytest.raises(RuntimeError, match="AMLAS_RUNTIME_ENV_PACK refused"):
            amlas._initialize_environment()
    finally:
        os.environ.pop(PACK_KEY, None)


CONTROL_TESTS = (
    "tests/test_qwen_sft_public_binding.py",
    "tests/test_paired_launcher_binding.py",
    "tests/test_case_runtime_wiring.py",
    "tests/test_preparation_integration.py",
    "tests/test_campaign_runtime_entry.py",
    "tests/test_managed_generation_product.py",
    "tests/test_paired_coding_deployability.py",
    "tests/test_model_acquisition_integration.py",
)


def test_existing_refusal_and_pin_tests_pass_on_expanded_environments(tmp_path):
    """Re-run the existing control suites with every startup reader fed the expanded environment."""
    counts = tmp_path / "counts.json"
    env = dict(os.environ, RUNTIME_ENV_PACK_CONTROLS_COUNTS=str(counts))
    # The nested pytest loads its plugin with -p before any test directory is on sys.path, so name the directory.
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(  # nosec B603 - nested pytest over fixed test files
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "runtime_env_pack_controls_plugin",
            f"--basetemp={tmp_path / 'nested'}",
            *CONTROL_TESTS,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    seen = json.loads(counts.read_text())
    # No existing suite builds the generation store from selected pins; the next test covers it.
    for reader in (
        "build_sft_store",
        "build_pair_store",
        "build_case_freeze_store",
        "build_preparation_store",
        "model_weight_acquisition_selectors",
    ):
        assert seen.get(reader + ":packed", 0) > 0, (reader, seen)
    assert (
        sum(count for name, count in seen.items() if name.startswith("_runtime_prompt_policy:environ")) > 0
    ), seen


@pytest.mark.parametrize(
    "change", ["none", "admission_differs", "foreign_account", "mutable_tag", "commit_missing"]
)
def test_generation_pin_decision_is_the_same_after_expansion(monkeypatch, change):
    from amlas.control_plane import generation_binding
    from amlas.control_plane.sft_state import validate_catalog

    captured = []
    monkeypatch.setattr(
        generation_binding,
        "SessionCodingGenerationStore",
        lambda *args: captured.append(args[-1]) or args[-1],
    )
    environment = fresh_deployment_environment()
    image = environment["AMLAS_CODING_GENERATION_IMAGE_URI"]
    if change == "admission_differs":
        environment["AMLAS_CODING_GENERATION_ADMISSION_DIGEST"] = hexed("fe", 64)
    if change == "foreign_account":
        environment["AMLAS_CODING_GENERATION_IMAGE_URI"] = image.replace(ACCOUNT, ACCOUNT[::-1])
    if change == "mutable_tag":
        environment["AMLAS_CODING_GENERATION_IMAGE_URI"] = image.split("@", 1)[0] + ":latest"
    if change == "commit_missing":
        environment["AMLAS_CODING_GENERATION_SOURCE_COMMIT"] = ""

    def decide(selected):
        catalog = generation_binding.build_generation_store(None, None, None, env=selected)
        try:
            validate_catalog(catalog)
        except ValueError as error:
            return catalog, ("refused", type(error).__name__, str(error))
        return catalog, "accepted"

    expanded = packer().pack(environment)
    packer().expand_environ(expanded)
    plain = decide(environment)
    assert decide(expanded) == plain
    assert (plain[1] == "accepted") is (change == "none")
