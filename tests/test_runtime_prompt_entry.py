"""Normal AgentCore prompts use the existing API without granting caller authority."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from amlas import api
from amlas import app as runtime
from amlas.tui.chat.models import ChatResponse


@pytest.fixture(autouse=True)
def inert_runtime_setup(monkeypatch):
    from amlas.cloud import secrets
    from amlas.config import env_loader
    from amlas.telemetry import error_log

    runtime._RESUME_WORKERS.clear()  # resume state is per process; tests reuse run ids
    monkeypatch.setattr(env_loader, "load_layered_env", lambda *_: None)
    monkeypatch.setattr(error_log, "init_error_log", lambda: None)
    monkeypatch.setattr(secrets, "resolve_secret_into_env", lambda **_: False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("AMLAS_MAX_RUN_SECONDS", raising=False)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AMLAS_HEARTBEAT_SECONDS", "0.01")


def collect(payload):
    async def read():
        return [event async for event in runtime.agent_invocation(payload)]

    return asyncio.run(read())


def test_runtime_prompt_delegates_exact_input_without_public_authority(monkeypatch):
    called = []
    prompt = "  Research and fine-tune one licensed coding model.\n"

    async def run_prompt(value, **kwargs):
        called.append((value, kwargs))
        return ChatResponse(message="A campaign handle was submitted.", action="job_submitted")

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    events = collect({"prompt": prompt})
    assert called == [(prompt, {})], "normal prompt must reach existing API exactly"
    start = next(event for event in events if event["event"] == "run_started")
    assert start["mode"] == "prompt"
    assert start["state_scope"] == "single_invocation"
    assert start["durable_resume"] is False
    terminal = events[-1]
    assert terminal["event"] == "run_completed"
    assert terminal["status"] == "response_received"
    assert terminal["response"]["action"] == "job_submitted"
    assert terminal["lifecycle_verified"] is False
    assert terminal["quality_success"] is False
    assert terminal["approval_policy_source"] == "default_noninteractive"


def test_runtime_prompt_text_and_data_do_not_become_quality_evidence(monkeypatch):
    async def run_prompt(_):
        return ChatResponse(
            message="Training completed with ideal coding quality.",
            action="trained",
            data={"quality_success": True, "trained": True},
        )

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    terminal = collect({"prompt": "Run the campaign."})[-1]
    assert terminal["event"] == "run_completed"
    assert terminal["status"] == "response_received"
    assert terminal["response"]["data"]["quality_success"] is True
    assert terminal["quality_success"] is False
    assert terminal["lifecycle_verified"] is False
    assert "arbiter_report" not in terminal


@pytest.mark.parametrize("prompt", ["", "   ", None, 7])
def test_runtime_invalid_prompt_refuses_before_api(monkeypatch, prompt):
    called = []

    async def run_prompt(*args, **kwargs):
        called.append((args, kwargs))

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    terminal = collect({"prompt": prompt})[-1]
    assert called == []
    assert terminal["event"] == "error"
    assert terminal["error_code"] == "invalid_prompt"


@pytest.mark.parametrize(
    "field",
    [
        "approval_policy",
        "settings",
        "caller_identity",
        "backend_catalog",
        "native_admission",
        "environment",
        "runtime_session_id",
    ],
)
def test_runtime_prompt_public_fields_cannot_configure_trusted_context(monkeypatch, field):
    called = []

    async def run_prompt(*args, **kwargs):
        called.append((args, kwargs))

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    terminal = collect({"prompt": "Run the campaign.", field: {"mode": "allow"}})[-1]
    assert called == []
    assert terminal["event"] == "error"
    assert terminal["error_code"] == "unsupported_prompt_fields"


def test_runtime_prompt_failure_is_typed_without_raw_error_detail(monkeypatch):
    async def run_prompt(_):
        raise RuntimeError("private provider diagnostic")

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    terminal = collect({"prompt": "Run the campaign."})[-1]
    assert terminal["event"] == "run_completed"
    assert terminal["status"] == "failed"
    assert terminal["error_type"] == "RuntimeError"
    assert "private provider diagnostic" not in str(terminal)
    assert terminal["lifecycle_verified"] is False


def test_runtime_prompt_rejects_non_chat_response(monkeypatch):
    async def run_prompt(_):
        return SimpleNamespace(message="done", action="trained", data={}, suggestions=[])

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    terminal = collect({"prompt": "Run the campaign."})[-1]
    assert terminal["event"] == "run_completed"
    assert terminal["status"] == "failed"
    assert terminal["error_type"] == "TypeError"


def test_runtime_prompt_emits_heartbeat_while_api_is_running(monkeypatch):
    release = threading.Event()

    async def run_prompt(_):
        assert release.wait(2), "test consumer must observe heartbeat and release worker"
        return ChatResponse(message="done")

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def read():
        events = []
        async for event in runtime.agent_invocation({"prompt": "Run the campaign."}):
            events.append(event)
            if event["event"] == "heartbeat":
                release.set()
        return events

    try:
        events = asyncio.run(read())
    finally:
        release.set()
    assert any(event["event"] == "heartbeat" and event["mode"] == "prompt" for event in events)
    assert events[-1]["status"] == "response_received"


def test_existing_spec_route_retains_optional_prompt_override(monkeypatch):
    from amlas import spec

    observed = []

    def load_spec(value):
        observed.append(value)
        raise LookupError("existing spec route reached")

    async def forbidden_prompt(*args, **kwargs):
        raise AssertionError("spec invocations must retain legacy route")

    monkeypatch.setattr(spec, "load_spec_from_text", load_spec)
    monkeypatch.setattr(api, "run_prompt", forbidden_prompt)
    with pytest.raises(LookupError, match="existing spec route reached"):
        collect({"spec_text": "existing spec", "prompt": "legacy override"})
    assert observed == ["existing spec"]


def test_runtime_prompt_keeps_trusted_context_outside_payload(monkeypatch):
    from contextvars import ContextVar

    caller = ContextVar("test_runtime_caller", default=None)
    observed = []

    async def run_prompt(_):
        observed.append(caller.get())
        return ChatResponse(message="done")

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    token = caller.set("trusted_code_context")
    try:
        terminal = collect({"prompt": "Run the campaign."})[-1]
    finally:
        caller.reset(token)
    assert observed == ["trusted_code_context"]
    assert terminal["status"] == "response_received"


def test_runtime_prompt_explicit_deadline_does_not_claim_worker_stopped(monkeypatch):
    release = threading.Event()
    monkeypatch.setenv("AMLAS_MAX_RUN_SECONDS", "0.02")

    async def run_prompt(_):
        assert release.wait(2)
        return ChatResponse(message="done")

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    try:
        terminal = collect({"prompt": "Run the campaign."})[-1]
    finally:
        release.set()
    assert terminal["event"] == "run_completed"
    assert terminal["status"] == "outcome_unknown"
    assert terminal["reason"] == "wall_clock_exceeded"
    assert terminal["worker_finished"] is False
    assert terminal["lifecycle_verified"] is False


def test_runtime_prompt_refuses_invalid_typed_response_fields(monkeypatch):
    async def run_prompt(_):
        return ChatResponse(message="done", data={"metric": float("nan")})

    monkeypatch.setattr(api, "run_prompt", run_prompt)
    terminal = collect({"prompt": "Run the campaign."})[-1]
    assert terminal["event"] == "run_completed"
    assert terminal["status"] == "failed"
    assert terminal["error_type"] == "ValueError"
