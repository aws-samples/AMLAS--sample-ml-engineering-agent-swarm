"""A transient model-call failure is retried inside the run, with the conversation unchanged.

Closed set: throttling, 5xx and connection errors. At most three retries per failing call, full-jitter
backoff with caps of 1, 2 and 4 seconds. The retry is the SDK's AfterModelCallEvent.retry, so the
failed call is repeated alone: the prompt is not appended again and no tool runs twice. A real SDK
Agent runs against a scripted model; no provider is contacted.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from sdk_scripted_model import ScriptedModel
from strands import tool
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.control_plane import normal_run_checkpoint
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
LOGGER = "amlas.tui.chat.tool_binding_backend"
RUN_ID = "12" * 16
LINE = re.compile(
    r"^model_call_retry run_id=(none|[0-9a-f]{32}) attempt=([0-9]+) limit=([0-9]+) "
    r"reason=(throttled|server_error|connection) delay_ms=([0-9]+)$"
)


def client_error(code, status):
    return ClientError(
        {"Error": {"Code": code, "Message": "neutral"}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "ConverseStream",
    )


def throttled():
    return ModelThrottledException("neutral throttle")


def server_error():
    return client_error("InternalServerException", 500)


def unavailable():
    return client_error("ServiceUnavailableException", 503)


def connection():
    return EndpointConnectionError(endpoint_url="https://bedrock.invalid")


class Rig:
    def __init__(self, monkeypatch, script, tools=()):
        self.slept = []

        async def sleep(delay):
            self.slept.append(delay)

        monkeypatch.setattr(binding, "_model_retry_sleep", sleep, raising=False)
        monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: list(tools))
        self.model = ScriptedModel(script)
        self.backend = ToolBindingBackend(Settings())
        self.backend._model = self.model
        original = self.backend.build_agent

        def build(*args, **kwargs):
            agent = original(*args, **kwargs)
            agent.callback_handler = lambda *a, **k: None
            self.agent = agent
            return agent

        monkeypatch.setattr(self.backend, "build_agent", build)

    def run(self):
        return self.backend._chat_with_tools_once("neutral prompt")


def retry_lines(caplog):
    return [r for r in caplog.records if r.name == LOGGER and r.getMessage().startswith("model_call_retry ")]


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (throttled, "throttled"),
        (server_error, "server_error"),
        (unavailable, "server_error"),
        (lambda: client_error("ModelStreamErrorException", 400), "server_error"),
        (lambda: client_error("ThrottlingException", 429), "throttled"),
        (lambda: client_error("SomeOtherCode", 502), "server_error"),
        (connection, "connection"),
        (lambda: ReadTimeoutError(endpoint_url="https://bedrock.invalid"), "connection"),
    ],
)
def test_one_transient_failure_is_retried_and_the_conversation_is_unchanged(
    monkeypatch, caplog, failure, reason
):
    caplog.set_level(logging.INFO, logger=LOGGER)
    rig = Rig(monkeypatch, [failure(), ("text", "all done")])
    turn = rig.run()
    assert turn.final_text == "all done" and rig.model.calls == 2
    assert rig.model.messages_seen[0] == rig.model.messages_seen[1]
    assert len(rig.model.messages_seen[1]) == 1
    assert [m["role"] for m in rig.agent.messages] == ["user", "assistant"]
    observed = [LINE.fullmatch(r.getMessage()) for r in retry_lines(caplog)]
    assert len(observed) == 1 and observed[0], [r.getMessage() for r in retry_lines(caplog)]
    assert observed[0].group(2, 3, 4) == ("1", "3", reason)
    assert len(rig.slept) == 1 and 0 <= rig.slept[0] <= 1.0
    assert int(observed[0].group(5)) == round(rig.slept[0] * 1000)


def test_three_retries_then_the_original_error_ends_the_run(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    rig = Rig(monkeypatch, [unavailable() for _ in range(4)])
    turn = rig.run()
    assert (
        turn.final_text.startswith("⚠️ Tool-agent error:") and "ServiceUnavailableException" in turn.final_text
    )
    assert rig.model.calls == 4
    observed = [LINE.fullmatch(r.getMessage()) for r in retry_lines(caplog)]
    assert [m.group(2) for m in observed] == ["1", "2", "3"] and {m.group(3) for m in observed} == {"3"}
    assert [d <= cap for d, cap in zip(rig.slept, (1.0, 2.0, 4.0), strict=True)] == [True, True, True]
    assert all(0 <= d for d in rig.slept)


def test_failure_after_a_tool_ran_repeats_only_the_model_call(monkeypatch):
    runs = []

    @tool
    def inert_echo(text: str) -> str:
        """Return the text; counts how often it ran."""
        runs.append(text)
        return text

    script = [("tool", "use-1", "inert_echo", {"text": "once"}), server_error(), ("text", "finished")]
    rig = Rig(monkeypatch, script, tools=[inert_echo])
    turn = rig.run()
    assert turn.final_text == "finished" and runs == ["once"] and rig.model.calls == 3
    assert rig.model.messages_seen[1] == rig.model.messages_seen[2]
    assert [m["role"] for m in rig.agent.messages] == ["user", "assistant", "user", "assistant"]


def test_the_budget_restarts_after_a_successful_model_call(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)

    @tool
    def inert_echo(text: str) -> str:
        """Return the text."""
        return text

    script = [
        server_error(),
        ("tool", "use-1", "inert_echo", {"text": "x"}),
        unavailable(),
        ("text", "finished"),
    ]
    rig = Rig(monkeypatch, script, tools=[inert_echo])
    assert rig.run().final_text == "finished"
    assert [LINE.fullmatch(r.getMessage()).group(2) for r in retry_lines(caplog)] == ["1", "1"]


@pytest.mark.parametrize(
    "failure",
    [
        lambda: RuntimeError("neutral failure"),
        lambda: client_error("AccessDeniedException", 403),
        lambda: client_error("ValidationException", 400),
        lambda: client_error("ExpiredTokenException", 400),
        lambda: ContextWindowOverflowException("neutral overflow"),
        lambda: TimeoutError("neutral timeout"),
        lambda: KeyError("neutral"),
    ],
)
def test_other_failures_are_not_retried(monkeypatch, caplog, failure):
    caplog.set_level(logging.INFO, logger=LOGGER)
    rig = Rig(monkeypatch, [failure(), ("text", "must not be reached")])
    turn = rig.run()
    assert turn.final_text and "must not be reached" not in turn.final_text
    assert rig.model.calls == 1 and rig.slept == [] and retry_lines(caplog) == []


def test_a_held_run_is_not_retried(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    rig = Rig(monkeypatch, [server_error(), ("text", "must not be reached")])
    original = rig.backend.build_agent

    def build(*args, **kwargs):
        agent = original(*args, **kwargs)
        rig.backend._input_hold = True
        return agent

    monkeypatch.setattr(rig.backend, "build_agent", build)
    turn = rig.run()
    assert rig.model.calls == 1 and retry_lines(caplog) == []
    assert turn.input_admission is not None


def test_run_id_is_logged_when_a_normal_run_is_active(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    monkeypatch.setattr(normal_run_checkpoint, "current_normal_run", lambda: SimpleNamespace(run_id=RUN_ID))
    rig = Rig(monkeypatch, [throttled(), ("text", "done")])
    agent = rig.backend.build_agent()
    agent("neutral prompt")
    assert rig.model.calls == 2
    assert LINE.fullmatch(retry_lines(caplog)[0].getMessage()).group(1) == RUN_ID


def test_delay_is_full_jitter_inside_the_caps():
    assert [binding._model_retry_delay(n, uniform=lambda low, high: high) for n in (1, 2, 3)] == [
        1.0,
        2.0,
        4.0,
    ]
    assert [binding._model_retry_delay(n, uniform=lambda low, high: low) for n in (1, 2, 3)] == [
        0.0,
        0.0,
        0.0,
    ]
    assert all(0 <= binding._model_retry_delay(3) <= 4.0 for _ in range(200))
    assert binding._MODEL_CALL_RETRIES == 3


def test_logger_failure_cannot_change_the_retry(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("ordinary diagnostic logger unavailable")

    monkeypatch.setattr(binding.logger, "warning", refuse)
    rig = Rig(monkeypatch, [server_error(), ("text", "all done")])
    assert rig.run().final_text == "all done" and rig.model.calls == 2
