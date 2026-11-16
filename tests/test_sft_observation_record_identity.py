"""Keep the genuine normal durable SFT record through fresh observation.

The ordinary app/API/SDK submits one registered full_v1 original using the
existing normal fixture, strict native transports and conditional journal.
Upstream suppliers and provider artifacts are fixture DATA, not native training
or full parent/loss/wake/quality proof. No journal, private pending witness,
authority or acceptance flag is seeded. The existing full warm GEN/PAIR control
remains in test_normal_sft_cold_continuation.py unchanged.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses

import pytest
import test_qwen_sft_backend as sft_fixture
from strands.hooks.events import AfterToolCallEvent
from test_normal_run_sft_checkpoint import TOOL_USE, configured_work, start_original
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup
from test_runtime_warm_model_continuation import tool_payload

from amlas.tui.chat.tool_binding_backend import ToolBindingBackend


def observe_registered_original(monkeypatch, tmp_path, *, completed, substitute=False):
    """Capture DATA at the real SDK tool boundary; retain the ordinary worker."""

    async def scenario():
        context, value = configured_work(monkeypatch, tmp_path)
        actual_build = ToolBindingBackend.build_agent
        captured = {}
        reader = None

        def build(backend, **kwargs):
            agent = actual_build(backend, **kwargs)

            def observe(event):
                if event.tool_use["toolUseId"] != TOOL_USE or captured:
                    return
                answer = tool_payload(event.result)
                if answer.get("accepted") is not True:
                    return
                store = backend._sft_store
                reference = answer["request_ref"]
                with store._lock:
                    original = store._records[reference]
                adapter = store._adapter
                captured.update(
                    original=original,
                    reference=reference,
                    before_pin=backend._producer_handle_pin("sft", original),
                    campaign_state=store._campaign_state,
                    execution_binding=store._execution_binding,
                )
                if completed:
                    sft_fixture.install_output(adapter, value.native, value.s3, original, recipe="full_v1")
                    value.native.states[original.id] = "Completed"
                before_mutations = copy.deepcopy(value.s3.mutations)
                before_sources = len(value.sources)
                reads, restored = [], []
                actual_describe = value.native.describe_training_job
                actual_restore, actual_artifacts = adapter.restore_original, adapter.artifacts

                def describe(**kw):
                    reads.append(kw["TrainingJobName"])
                    return actual_describe(**kw)

                def restore(*args, **kw):
                    checked = actual_restore(*args, **kw)
                    restored.append(checked)
                    return checked

                def artifacts(checked, **kw):
                    result = actual_artifacts(checked, **kw)
                    if substitute:
                        replacement = dataclasses.replace(original, extra=copy.deepcopy(original.extra))
                        with store._lock:
                            store._records[reference] = replacement
                        captured["replacement"] = replacement
                    return result

                with monkeypatch.context() as observation_patch:
                    observation_patch.setattr(value.native, "describe_training_job", describe)
                    observation_patch.setattr(adapter, "restore_original", restore)
                    observation_patch.setattr(adapter, "artifacts", artifacts)
                    captured["result"] = store.observe(reference)
                with store._lock:
                    current = store._records[reference]
                captured.update(
                    current=current,
                    after_pin=backend._producer_handle_pin("sft", current),
                    native_reads=reads,
                    restored=restored,
                    fresh_source_reads=len(value.sources) - before_sources,
                    staged_during_observation=value.s3.mutations != before_mutations,
                    bodies_closed=all(body.closed for body in value.s3.bodies),
                    creates=len(value.native.calls),
                )

            # The SDK invokes newest AfterToolCall callbacks first. This oracle
            # sees the actual accepted record before the normal capture/wait;
            # it does not create or replace the production pending witness.
            agent.hooks.add_callback(AfterToolCallEvent, observe)
            return agent

        monkeypatch.setattr(ToolBindingBackend, "build_agent", build)
        try:
            reader, answer, _run_id = await start_original(value)
            assert captured.get("result") is not None, "normal SFT observation prerequisite did not finish"
            assert captured["campaign_state"] is not None and captured["execution_binding"] is not None
            assert answer["request_ref"] == captured["reference"]
            assert captured["original"].id == value.native.calls[0]["TrainingJobName"]
            assert captured["creates"] == len(value.native.calls) == 1
            assert captured["native_reads"] and set(captured["native_reads"]) == {captured["original"].id}
            assert len(captured["restored"]) == 1
            assert captured["fresh_source_reads"] >= 2
            assert captured["bodies_closed"] is True
            assert captured["staged_during_observation"] is False
            return captured
        finally:
            if reader is not None:
                await close_original_worker(value.work, reader)
            context.close()

    return asyncio.run(scenario())


@pytest.mark.parametrize("completed", [False, True], ids=["running", "completed"])
def test_normal_durable_sft_observation_preserves_original_record_and_pin(monkeypatch, tmp_path, completed):
    captured = observe_registered_original(monkeypatch, tmp_path, completed=completed)
    result = captured["result"]
    assert result["accepted"] is True and result["request_ref"] == captured["reference"]
    assert result["success"] is completed and result["trained"] is completed
    assert result["full_training_complete"] is completed
    if completed:
        assert result["training_stage"] == "full"
        assert result["checkpoint_ref"].version_id
    assert captured["current"] is captured["original"], (
        "fresh durable SFT observation replaced the genuine captured original record"
    )
    assert captured["after_pin"] == captured["before_pin"], (
        "fresh durable SFT observation changed the captured producer pin"
    )


def test_normal_durable_sft_record_substitution_refuses_before_completed_result(monkeypatch, tmp_path):
    captured = observe_registered_original(monkeypatch, tmp_path, completed=True, substitute=True)
    result = captured["result"]
    assert result.get("success") is False and result.get("status") == "observation_failed", (
        "durable SFT observation exposed completion after its original record was substituted"
    )
    assert result.get("trained") is not True and result.get("full_training_complete") is not True
    assert captured["current"] is captured["replacement"], (
        "durable SFT observation overwrote a concurrently substituted record"
    )
