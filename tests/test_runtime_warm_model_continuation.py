"""One ordinary SDK/model job must retain warm causality; no native AWS/model calls."""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import model_acquisition_fixtures as f
import pytest
from strands.hooks.events import AfterToolCallEvent
from strands.models import Model

from amlas.backends.adapters import model_weight_acquisition as adapter_source
from amlas.config import externalized
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.data_eng import model_acquisition_schema
from amlas.flows import model_weight_acquisition_flow
from amlas.tui.chat import engine as engine_source
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend, _record_messages


class OriginalModelStream(Model):
    """Emit one registered tool and observe only harmless synthetic SDK history."""

    def __init__(self, *, intervened=False, error_after_accept=False):
        self.intervened = intervened
        self.error_after_accept = error_after_accept
        self.reacquire = False
        self.calls = 0
        self.messages_seen = []

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "neutral-warm-original-model"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("ordinary installed SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.messages_seen.append(copy.deepcopy(messages))
        self.calls += 1
        if self.error_after_accept and self.calls == 2:
            raise RuntimeError("ordinary neutral first-turn model failure")
        yield {"messageStart": {"role": "assistant"}}
        if self.calls == 1 or (self.reacquire and self.calls == 3):
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {
                        "toolUse": {
                            "toolUseId": "original-model-use" if self.calls == 1 else "dedup-original-use",
                            "name": "acquire_model_weights",
                        }
                    },
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {
                        "toolUse": {
                            "input": json.dumps({"catalog_id": f.CATALOG, "catalog_version": f.VERSION})
                        }
                    },
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            assert self.calls in (2, 3, 4), "no prompt/model replay"
            text = (
                "The original acquisition is pending."
                if self.calls == 2
                else "Original exact-version artifacts are verified."
            )
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {
                "messageStop": {
                    "stopReason": "guardrail_intervened"
                    if self.intervened and self.calls == 2
                    else "end_turn"
                }
            }


def tool_payload(result):
    for block in result["content"]:
        if "json" in block:
            return block["json"]
        if "text" in block:
            return json.loads(block["text"])
    raise AssertionError("normal SDK lost original structured tool result")


_ACTUAL_ENGINE = engine_source.ChatEngine
_ACTUAL_BUILD = ToolBindingBackend.build_agent


def original_work(monkeypatch, *, intervened=False, error_after_accept=False):
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    f.patch_expected(monkeypatch, [adapter_source, model_acquisition_schema])
    native, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(adapter_source, native, s3)
    monkeypatch.setattr(model_weight_acquisition_flow, "_resolve_backend", lambda: adapter)
    policy = ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=None, interactive=False)
    model = OriginalModelStream(intervened=intervened, error_after_accept=error_after_accept)
    engines, agents, results, tool_events = [], [], [], []
    actual_engine = _ACTUAL_ENGINE
    actual_build = _ACTUAL_BUILD

    def capture_engine(*args, **kwargs):
        value = actual_engine(*args, **kwargs)
        engines.append(value)
        return value

    def capture_agent(backend, **kwargs):
        backend._model = model
        value = actual_build(backend, **kwargs)
        value.callback_handler = lambda *args, **kwargs: None

        def capture_original_event(event):
            tool_events.append(event)
            results.append((event.tool_use["toolUseId"], tool_payload(event.result)))

        value.hooks.add_callback(AfterToolCallEvent, capture_original_event)
        agents.append(value)
        return value

    monkeypatch.setattr(engine_source, "ChatEngine", capture_engine)
    monkeypatch.setattr(ToolBindingBackend, "build_agent", capture_agent)
    literal = "Research one licensed coding model and continue after its original acquisition."
    # This fixture owns one manual engine turn; connected app/API waiting has its
    # separate actual normal-path acceptance suite.
    engine = capture_engine(settings=Settings(), approval_policy=policy)
    response = asyncio.run(engine.process_message(literal))
    if error_after_accept:
        assert "Tool-agent error" in response.message
    else:
        assert "original acquisition is pending" in response.message
    assert len(native.requests) == len(engines) == len(agents) == 1
    assert len(results) == 1 and results[0][0] == "original-model-use"
    original = results[0][1]
    assert original["status"] == "submitted" and original["accepted"] is True
    reference = original["request_ref"]
    engine, agent = engines[0], agents[0]
    backend = engine.tool_backend
    stores = tuple(
        getattr(backend, name)
        for name in (
            "_acquisition_store",
            "_model_acquisition_store",
            "_case_freeze_store",
            "_preparation_store",
            "_sft_store",
            "_generation_store",
            "_pair_store",
        )
    )
    store = backend._model_acquisition_store
    handle = store._records[reference].handle
    assert handle.id == native.requests[0]["ProcessingJobName"]
    assert handle.extra["bindings"]["runtime_session_id"] == backend._acquisition_store.runtime_session_id
    assert model.calls == 2
    if not error_after_accept:
        assert store.observe(reference)["status"] == "RUNNING"
    before = copy.deepcopy(agent.messages)
    assert any(
        block.get("toolResult", {}).get("toolUseId") == "original-model-use"
        for message in before
        for block in message["content"]
    )

    return SimpleNamespace(
        engine=engine,
        backend=backend,
        agent=agent,
        model=model,
        native=native,
        s3=s3,
        adapter=adapter,
        reference=reference,
        handle=handle,
        stores=stores,
        response=response,
        before=before,
        engines=engines,
        agents=agents,
        tool_events=tool_events,
    )


def step(work):
    observe = getattr(work.engine, "_observe_pending_model_once", None)
    assert callable(observe), "warm-observation-missing after one original SDK ProcessingCreate"
    return asyncio.run(observe())


def complete_fixture(monkeypatch, tmp_path, work):
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setattr(f, "bindings", lambda: dict(work.handle.extra["bindings"]))
    f.publish_fixture(monkeypatch, tmp_path, work.adapter, work.handle)
    work.native.state = "Completed"


def test_manual_normal_engine_retains_original_model_for_readonly_warm_completion(monkeypatch, tmp_path):
    work = original_work(monkeypatch)
    engine, backend, model, native, s3 = work.engine, work.backend, work.model, work.native, work.s3
    stores, store, reference, handle = (
        work.stores,
        work.backend._model_acquisition_store,
        work.reference,
        work.handle,
    )
    engines, agents = work.engines, work.agents
    # A private selector-free step must consume only this original owning record.
    observe = getattr(engine, "_observe_pending_model_once", None)
    assert callable(observe), "warm-observation-missing after one original SDK ProcessingCreate"
    running = asyncio.run(observe())
    assert running.disposition == "awaiting_original" and model.calls == 2
    assert (
        tuple(
            getattr(backend, name)
            for name in (
                "_acquisition_store",
                "_model_acquisition_store",
                "_case_freeze_store",
                "_preparation_store",
                "_sft_store",
                "_generation_store",
                "_pair_store",
            )
        )
        == stores
    )
    complete_fixture(monkeypatch, tmp_path, work)
    completed = asyncio.run(observe())
    assert completed.disposition == "continued_original" and model.calls == 3
    assert len(engines) == len(agents) == len(native.requests) == 1
    assert store._records[reference].handle is handle
    snapshot = store.observe(reference)
    assert snapshot["acquired"] is True and len(snapshot["file_refs"]) == 9
    assert all(read["VersionId"] for read in s3.reads)
    history = model.messages_seen[-1]
    assert work.agent.messages[:len(work.before)] == work.before
    user_texts = [message for message in history if message["role"] == "user"
                  and any("text" in block for block in message["content"])]
    assert len(user_texts) == 2  # One external literal plus one internal completion DATA event.
    assert user_texts[0] == _record_messages(work.before)[0]
    completion = json.loads(user_texts[-1]["content"][0]["text"])
    assert completion["event_type"] == "original_model_acquisition_completed" and completion["data_only"] is True
    assert completion["request_ref"] == reference and completion["tool_use_id"] == "original-model-use"
    assert completion["acquired"] is True and completion["status"] == "SUCCEEDED"
    assert completion["file_count"] == 9 and completion["manifest_roles"] == ["receipt", "commit", "lineage"]
    terminal = next(
        block["toolResult"]
        for message in history
        for block in message["content"]
        if block.get("toolResult", {}).get("toolUseId") == "original-model-use"
    )
    pending_payload = tool_payload(terminal)
    assert pending_payload["request_ref"] == reference and pending_payload["job_name"] == handle.id
    assert pending_payload["status"] == "submitted" and pending_payload["accepted"] is True


@pytest.mark.parametrize(
    "native_state,read_failure,disposition",
    [
        ("InProgress", True, "awaiting_original"),
        ("Failed", False, "terminal_failed"),
        ("Stopped", False, "terminal_failed"),
    ],
)
def test_original_unknown_or_failed_holds_history_without_agent_or_create(
    monkeypatch, native_state, read_failure, disposition
):
    work = original_work(monkeypatch)
    work.native.state, work.native.read_failure = native_state, read_failure
    result = step(work)
    assert result.disposition == disposition
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert work.agent.messages == work.before
    assert work.backend._model_acquisition_store._records[work.reference].handle is work.handle


@pytest.mark.parametrize("policy", [None, ApprovalPolicy()])
def test_warm_policy_revocation_refuses_before_original_read(monkeypatch, policy):
    work = original_work(monkeypatch)
    work.backend.approval_policy = policy
    monkeypatch.setattr(
        work.adapter, "artifacts", lambda *_: pytest.fail("revoked warm policy read original SDK")
    )
    assert step(work).disposition == "refused"
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert work.agent.messages == work.before


def test_foreign_original_record_refuses_before_any_observation(monkeypatch):
    left, right = original_work(monkeypatch), original_work(monkeypatch)
    left.backend._model_acquisition_store._records[left.reference] = (
        right.backend._model_acquisition_store._records[right.reference]
    )
    monkeypatch.setattr(left.adapter, "artifacts", lambda *_: pytest.fail("foreign record reached observer"))
    monkeypatch.setattr(
        right.adapter, "artifacts", lambda *_: pytest.fail("foreign record selected another observer")
    )
    assert step(left).disposition == "refused"
    assert left.model.calls == right.model.calls == 2
    assert len(left.native.requests) == len(right.native.requests) == 1


def test_public_status_and_ref_do_not_select_or_replace_original_pending(monkeypatch):
    work = original_work(monkeypatch)
    work.engine.context["running_jobs"] = [
        {"status": "Completed", "request_ref": "model_acq_" + "f" * 32, "acquired": True}
    ]
    work.response.data.update(status="Completed", request_ref="model_acq_" + "e" * 32, acquired=True)
    assert step(work).disposition == "awaiting_original"
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert work.agent.messages == work.before
    assert work.backend._model_acquisition_store._records[work.reference].handle is work.handle


def test_guardrail_intervention_closes_warm_pending_without_read_or_grant(monkeypatch):
    work = original_work(monkeypatch, intervened=True)
    assert work.response.action == "guardrail_intervened"
    monkeypatch.setattr(work.adapter, "artifacts", lambda *_: pytest.fail("intervention resumed observation"))
    assert step(work).disposition == "guardrail_intervened"
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert work.agent.messages == work.before


def test_two_warm_owners_keep_original_graph_agent_history_and_completion_isolated(monkeypatch, tmp_path):
    left, right = original_work(monkeypatch), original_work(monkeypatch)
    assert all(a is not b for a, b in zip(left.stores, right.stores, strict=True))
    assert left.reference != right.reference and left.agent is not right.agent
    complete_fixture(monkeypatch, tmp_path / "left", left)
    assert step(left).disposition == "continued_original"
    assert left.model.calls == 3 and right.model.calls == 2
    assert right.agent.messages == right.before
    assert step(right).disposition == "awaiting_original"
    assert len(left.native.requests) == len(right.native.requests) == 1


def test_new_user_turn_cannot_discard_original_active_warm_ownership(monkeypatch):
    work = original_work(monkeypatch)
    before = copy.deepcopy(work.agent.messages)
    work.backend.chat_with_tools("A separate ordinary user turn.")
    assert work.model.calls == 2, "active original warm owner was silently discarded"
    assert work.agent.messages == before
    assert len(work.native.requests) == 1
    assert step(work).disposition == "awaiting_original"


def test_same_agent_deduped_original_is_consumed_once_without_new_warm_owner(monkeypatch, tmp_path):
    work = original_work(monkeypatch)
    work.model.reacquire = True
    complete_fixture(monkeypatch, tmp_path, work)
    result = step(work)
    assert result.disposition == "continued_original"
    assert work.model.calls == 4 and len(work.native.requests) == 1
    assert len(work.agents) == 1 and len(work.engines) == 1
    assert work.backend._model_acquisition_store._records[work.reference].handle is work.handle
    uses = [
        block["toolResult"]
        for message in work.agent.messages
        for block in message["content"]
        if "toolResult" in block
    ]
    original = next(value for value in uses if value["toolUseId"] == "original-model-use")
    dedup = next(value for value in uses if value["toolUseId"] == "dedup-original-use")
    assert tool_payload(original)["request_ref"] == tool_payload(dedup)["request_ref"] == work.reference
    assert tool_payload(dedup)["acquired"] is True
    before = copy.deepcopy(work.agent.messages)
    assert step(work).disposition == "refused"
    assert work.agent.messages == before and work.model.calls == 4
    assert len(work.native.requests) == 1


def test_verified_full_object_source_completion_needs_no_private_cache_grant(monkeypatch, tmp_path):
    work = original_work(monkeypatch)
    complete_fixture(monkeypatch, tmp_path, work)
    work.s3.full_checksums = True
    assert id(work.handle) not in work.adapter._byte_proofs
    assert step(work).disposition == "continued_original"
    assert id(work.handle) not in work.adapter._byte_proofs
    assert work.model.calls == 3 and len(work.native.requests) == 1
    assert work.backend._model_acquisition_store._records[work.reference].handle is work.handle
