"""Trusted deployment approval configuration stays outside the public prompt."""

import asyncio
import json

import pytest

from amlas import api
from amlas import app as runtime
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat.models import ChatResponse

ENV = "AMLAS_RUNTIME_APPROVED_ACTIONS"


@pytest.fixture(autouse=True)
def inert_setup(monkeypatch):
    from amlas.cloud import secrets
    from amlas.config import env_loader
    from amlas.telemetry import error_log

    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("AMLAS_MAX_RUN_SECONDS", raising=False)
    monkeypatch.setenv("AMLAS_HEARTBEAT_SECONDS", "0.01")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setattr(env_loader, "load_layered_env", lambda *_: None)
    monkeypatch.setattr(error_log, "init_error_log", lambda: None)
    monkeypatch.setattr(secrets, "resolve_secret_into_env", lambda **_: False)


def collect(payload):
    async def read():
        return [event async for event in runtime.agent_invocation(payload)]

    return asyncio.run(read())


def test_missing_deployment_policy_keeps_safe_default():
    assert runtime._runtime_prompt_policy() is None


def test_empty_deployment_allowlist_grants_nothing(monkeypatch):
    monkeypatch.setenv(ENV, "[]")
    assert runtime._runtime_prompt_policy() is None


def test_explicit_deployment_actions_build_typed_allowlist(monkeypatch):
    names = ["run_processing_job", "submit_qwen_sft", "coding_generation_flow"]
    monkeypatch.setenv(ENV, json.dumps(names))
    policy = runtime._runtime_prompt_policy()
    assert isinstance(policy, ApprovalPolicy)
    assert policy.mode is ApprovalMode.ALLOWLIST
    assert policy.allowlist == names
    assert policy.max_auto_cost is None
    assert policy.interactive is False
    assert policy.responder is None


@pytest.mark.parametrize(
    "raw",
    [
        "invalid",
        "null",
        "{}",
        "true",
        "[1]",
        '["submit_qwen_sft","submit_qwen_sft"]',
        '["*"]',
        '["grant_role_policy"]',
        '["request_quota_increase"]',
        '["open_pr"]',
    ],
)
def test_invalid_or_unrelated_deployment_actions_refuse(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError):
        runtime._runtime_prompt_policy()


def test_runtime_transports_only_trusted_typed_policy(monkeypatch):
    called = []
    prompt = "  Research and train one coding model.\n"
    monkeypatch.setenv(ENV, '["run_processing_job","submit_qwen_sft"]')

    async def run_prompt(value, **kwargs):
        called.append((value, kwargs))
        return ChatResponse(message="An original job was submitted.", action="job_submitted")

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    events = collect({"prompt": prompt})
    assert len(called) == 1
    assert called[0][0] == prompt
    assert set(called[0][1]) == {"policy"}
    policy = called[0][1]["policy"]
    assert isinstance(policy, ApprovalPolicy)
    assert policy.mode is ApprovalMode.ALLOWLIST
    assert policy.allowlist == ["run_processing_job", "submit_qwen_sft"]
    terminal = events[-1]
    assert terminal["approval_policy_source"] == "deployment_allowlist"
    assert terminal["lifecycle_verified"] is False
    assert terminal["quality_success"] is False


def test_bad_deployment_policy_refuses_before_api(monkeypatch):
    called = []
    monkeypatch.setenv(ENV, '["open_pr"]')

    async def run_prompt(*args, **kwargs):
        called.append((args, kwargs))
        return ChatResponse(message="unexpected")

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    terminal = collect({"prompt": "Run the campaign."})[-1]
    assert called == []
    assert terminal["event"] == "run_completed"
    assert terminal["status"] == "failed"
    assert terminal["error_type"] == "ValueError"
    assert terminal["approval_policy_source"] == "invalid_deployment_policy"
    assert terminal["worker_finished"] is True
    assert terminal["lifecycle_verified"] is False


def test_public_payload_cannot_override_deployment_policy(monkeypatch):
    monkeypatch.setenv(ENV, '["submit_qwen_sft"]')
    called = []

    async def run_prompt(*args, **kwargs):
        called.append((args, kwargs))
        return ChatResponse(message="unexpected")

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    terminal = collect({"prompt": "Run the campaign.", "policy": {"mode": "auto"}})[-1]
    assert called == []
    assert terminal["error_code"] == "unsupported_prompt_fields"


def test_typed_policy_allows_only_explicit_campaign_actions(monkeypatch):
    from amlas.control_plane.approvals import request_approval

    monkeypatch.setenv(ENV, '["qwen_sft_flow","run_processing_job"]')
    policy = runtime._runtime_prompt_policy()
    for action in ("qwen_sft_flow", "run_processing_job"):
        decision = request_approval(action=action, cost_estimate=None, policy=policy)
        assert decision["decision"] == "approved"
    for action in ("open_pr", "request_quota_increase", "grant_role_policy"):
        decision = request_approval(action=action, cost_estimate=None, policy=policy)
        assert decision["decision"] == "denied"
