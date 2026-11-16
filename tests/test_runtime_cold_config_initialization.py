"""AgentCore overlay ordering on ordinary cold read/resume entrypoint routes.

Declared handlers and an overlay spy use no runtime state, model, native client,
SDK session, checkpoint, provider observation or execution authority.
"""

from __future__ import annotations

import asyncio

import pytest

from amlas import app as runtime
from amlas.config import env_loader


@pytest.mark.parametrize(
    "action", ["resume_prompt_run", "get_prompt_run_status", "get_prompt_report"]
)
def test_cold_read_and_resume_initialize_agentcore_overlay_before_handler(monkeypatch, action):
    order = []
    request = {"action": action, "run_id": "1" * 32}
    result = {
        "event": "prompt_report" if action == "get_prompt_report" else "prompt_run_status",
        "status": "unavailable",
        "run_id": request["run_id"],
        "found": False,
    }

    def overlay(layer=None, *, force=False):
        assert layer == "agentcore" and force is False
        order.append("overlay")

    def require_overlay(payload):
        assert payload == request
        assert order == ["overlay"], "one AgentCore overlay must precede the cold route handler"
        order.append("handler")
        return result

    async def resume(payload):
        yield require_overlay(payload)

    monkeypatch.setattr(env_loader, "load_layered_env", overlay)
    if action == "resume_prompt_run":
        monkeypatch.setattr(runtime, "_resume_normal_prompt", resume)
    elif action == "get_prompt_run_status":
        monkeypatch.setattr(runtime, "_read_prompt_run", require_overlay)
    else:
        monkeypatch.setattr(runtime, "_read_prompt_report", require_overlay)

    async def read():
        return [event async for event in runtime.agent_invocation(request)]

    assert asyncio.run(read()) == [result]
    assert order == ["overlay", "handler"]
