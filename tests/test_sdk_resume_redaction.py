"""Neutral actual-SDK history/redaction discriminator; no live guard/ML proof."""

from __future__ import annotations

import asyncio
import copy
import threading
from types import SimpleNamespace

import model_acquisition_fixtures as f
import pytest
from strands.hooks.events import BeforeModelCallEvent
from strands.models.bedrock import BedrockModel
from test_normal_research_profile import PROMPT, LifecycleModel, setup
from test_runtime_warm_model_continuation import complete_fixture

from amlas import api
from amlas.backends.adapters import model_weight_acquisition as adapter_source
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode
from amlas.data_eng import model_acquisition_schema
from amlas.flows import model_weight_acquisition_flow
from amlas.tools.coding_preparation import list_coding_preparation_profiles
from amlas.tools.model_weight_tools import acquire_model_weights
from amlas.tui.chat.guardrail_assessment import project_guardrail_trace


class ResumeDiscriminator(LifecycleModel):
    def __init__(self, mode, trace_action=None):
        super().__init__(model_acquisition=True)
        self.mode = mode
        self.trace_action = trace_action
        self.trace_value = None

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls < len(self.requests):
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
            return
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "Neutral original completion observed."},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {
            "messageStop": {
                "stopReason": "guardrail_intervened"
                if self.mode in {"provider_redaction", "primitive_guardrail"}
                else "end_turn"
            }
        }
        if self.mode == "provider_redaction":
            # Pinned Bedrock streaming order: messageStop, generated redaction,
            # original metadata trace. SDK consumes all before MessageAdded.
            yield {"redactContent": {"redactUserContentMessage": "Neutral SDK redaction marker."}}
        if self.trace_action is not None:
            assessment = {
                "contentPolicy": {
                    "filters": [
                        {
                            "type": "VIOLENCE",
                            "confidence": "HIGH",
                            "detected": self.trace_action != "NONE",
                            "action": "BLOCKED" if self.trace_action == "UNKNOWN" else self.trace_action,
                        }
                    ]
                }
            }
            if self.trace_action == "UNKNOWN":
                assessment["topicPolicy"] = {}
            self.trace_value = {"guardrail": {"inputAssessment": {"neutral-fixture": assessment}}}
            # Pure pinned SDK predicate only: no client/provider construction. The per-assessment
            # predicate is the one the SDK builds its blocked-trace decision on in every version
            # (Strands 1.42 _has_blocked_guardrail, Strands 1.65 _should_redact_guardrail_content).
            predicate = object.__new__(BedrockModel)
            assert any(
                predicate._find_detected_and_blocked_policy(value)
                for value in self.trace_value["guardrail"]["inputAssessment"].values()
            ) is (self.trace_action != "NONE")
        metadata = {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 0},
        }
        if self.trace_value is not None:
            metadata["trace"] = self.trace_value
        yield {"metadata": metadata}


@pytest.mark.parametrize(
    "mode", ["normal_final", "primitive_guardrail", "provider_redaction", "history_change"]
)
def test_normal_selected_profile_quarantine_model_resume_distinguishes_redaction_from_plain_stop(
    monkeypatch, tmp_path, mode, trace_action=None
):
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    f.patch_expected(monkeypatch, [adapter_source, model_acquisition_schema])
    native, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(adapter_source, native, s3)
    observing = threading.Event()
    describe = native.describe_processing_job

    def observe_original(**kwargs):
        # Passive actual warm-observer boundary proves the initial SDK result
        # and original capture have settled; stream-start/Create do not.
        observing.set()
        return describe(**kwargs)

    monkeypatch.setattr(native, "describe_processing_job", observe_original)
    monkeypatch.setattr(model_weight_acquisition_flow, "_resolve_backend", lambda: adapter)
    model = ResumeDiscriminator(mode, trace_action=trace_action)
    selected = setup(
        monkeypatch,
        model_override=model,
        extra_tools=(list_coding_preparation_profiles, acquire_model_weights),
    )
    selected.policy.mode = ApprovalMode.AUTO
    original_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        await original_sleep(0.005 if delay == 15 else delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    async def scenario():
        task = asyncio.create_task(api.run_prompt(PROMPT, settings=Settings(), policy=selected.policy))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not task.done() and not observing.is_set():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert not task.done() and model.calls == 4
            backend, agent = selected.backends[0], selected.agents[0]
            assert backend._research_binding["phase"] == "CONSUMED"
            assert backend._input_checker is not None and backend._input_guard_client is not None
            before = copy.deepcopy(agent.messages)
            pending = backend._warm_pending
            assert pending is not None and pending.agent is agent
            if mode == "history_change":

                def replace_latest(event):
                    # Source-owned neutral mutation control, without content from
                    # any rejected session and without touching guard settings.
                    event.agent.messages[-1]["content"] = [{"text": "Neutral foreign history marker."}]

                agent.hooks.add_callback(BeforeModelCallEvent, replace_latest)
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=pending.record.handle, adapter=adapter, native=native),
            )
            response = await asyncio.wait_for(task, 5)
            assert len(native.requests) == len(selected.agents) == len(selected.backends) == 1
            assert model.calls == 5 and agent.messages[: len(before)] == before
            records = response.data["input_admission"]["records"]
            assert records[-1]["boundary"] == "COMPLETION_DATA" and records[-1]["outcome"] == "ALLOW"
            assert sum(record["outcome"] == "BLOCKED" for record in records) == 1
            assert backend._research_binding["checker"] is backend._input_checker
            if mode in {"provider_redaction", "history_change"}:
                assert response.action == "input_admission_held"
                assert response.data["input_admission"]["status"] == "held"
                assert response.data["warm_model_continuation"] == {
                    "disposition": "refused",
                    "diagnostic": {
                        "phase": "resume",
                        "reason": "continuation_failed",
                        "error_class": "UNKNOWN",
                    },
                }
                assert backend._input_hold is True and backend._warm_closed == "refused"
                if trace_action is not None:
                    expected = project_guardrail_trace(model.trace_value).to_dict()
                    assert expected["action"] == trace_action
                    assert response.data.get("guardrail_assessment") == expected, (
                        "already captured primitive-only SDK assessment was lost on held resume"
                    )
                else:
                    assert response.data.get("guardrail_assessment") is None
                assert agent.messages[-2]["content"] == [
                    {
                        "text": "Neutral SDK redaction marker."
                        if mode == "provider_redaction"
                        else "Neutral foreign history marker."
                    }
                ]
            elif mode == "primitive_guardrail":
                assert response.data["warm_model_continuation"]["disposition"] == "guardrail_intervened"
                assert response.action == "guardrail_intervened" and not backend._input_hold
            else:
                assert response.data["warm_model_continuation"]["disposition"] == "continued_original"
                assert not backend._input_hold
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("trace_action", ["BLOCKED", "NONE", "UNKNOWN"])
def test_refused_resume_retains_callback_populated_closed_assessment_without_changing_hold(
    monkeypatch, tmp_path, trace_action
):
    test_normal_selected_profile_quarantine_model_resume_distinguishes_redaction_from_plain_stop(
        monkeypatch,
        tmp_path,
        "history_change" if trace_action == "NONE" else "provider_redaction",
        trace_action=trace_action,
    )
