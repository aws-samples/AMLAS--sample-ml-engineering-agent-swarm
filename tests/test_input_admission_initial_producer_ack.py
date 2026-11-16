"""Initial genuine SFT acknowledgement retires an owned idle guard pool.

Strict inert native/guard ports exercise the ordinary consumed-profile SDK path.
The idle flag models transport age; it does not establish the native incident cause.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json

import pytest
from test_input_admission_connection_lifecycle import GUARD, GuardTransport
from test_qwen_sft_backend import backend as sft_fixture
from test_qwen_sft_backend import catalog, described_resource_config, install_output
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_precommit_original_completion import ThreeOriginals, _events, protocol_work
from test_runtime_sft_reload_originals import current_inputs
from test_runtime_warm_model_continuation import tool_payload

from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
from amlas.control_plane.preparation_state import PreparationCatalog, SessionPreparationStore
from amlas.control_plane.sft_state import SessionSftStore
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
USE = "original-initial-train-use"


class InitialTrainStream(ThreeOriginals):
    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls < self.research_count:
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
            return
        self.calls += 1
        self.messages_seen.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        if self.calls == 4:
            arguments = {
                "model_request_ref": "model_acq_" + "a" * 32,
                "preparation_request_ref": "prep_" + "b" * 32,
                "recipe_id": "full_v1",
            }
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": USE, "name": "submit_qwen_sft"}},
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": json.dumps(arguments)}},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            assert self.calls == 5, "original producer entered an extra model cycle"
            assert messages[-1]["role"] == "user" and len(messages[-1]["content"]) == 1
            result = messages[-1]["content"][0]["toolResult"]
            assert result["toolUseId"] == USE
            payload = tool_payload(result)
            assert payload["data_only"] is True and payload["event_type"] == "original_sft_completed"
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Original training artifacts verified."},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }


def test_initial_native_sft_ack_retires_owned_pool_before_first_assessment(monkeypatch):
    import boto3

    adapter, native, s3, inputs = sft_fixture()
    model = InitialTrainStream()
    work, _data, _fetched = protocol_work(monkeypatch, model)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","qwen_sft_flow"]')
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", GUARD)
    events, sessions, clients, requests, states, before = [], [], [], {}, {}, {}

    class Session:
        def __init__(self, *, profile_name, region_name):
            assert region_name == "us-west-2"
            sessions.append(self)

        def client(self, service, *, region_name, config):
            assert service == "bedrock-runtime" and region_name == "us-west-2"
            assert config.connect_timeout == 5 and config.read_timeout == 15
            assert config.retries == {"total_max_attempts": 1}
            assert len(clients) < 3, "original acknowledgement/completion attempted extra assessments"
            if clients:
                clients[-1].retire()
                clients[-1].retired = True
                assert clients[-1].closed and self is sessions[0]
            label = str(len(clients))
            value = GuardTransport(label, events)

            def retire_owned_pool():
                assert self is sessions[0], "retirement changed the owning SDK Session"
                assert value.closed, "retirement preceded clearing the owned pool"
                value.close()

            value.retire = retire_owned_pool
            if not clients:
                actual_close = value.close

                def close_after_capture():
                    backend = work.engines[0].tool_backend
                    captured = tuple(backend._warm_pending_producers.values())
                    assert len(captured) == 1 and captured[0].kind == "sft"
                    pending = captured[0]
                    assert pending.store is backend._sft_store and pending.agent is work.agents[0]
                    assert pending.record is pending.store._records[pending.reference]
                    assert pending.tool_use_id == USE and pending.record.id in requests
                    events.append(("captured", label))
                    actual_close()

                def clear_owned_pool():
                    actual_close()
                    events[-1] = ("pool_clear", value.label)

                value.retire = close_after_capture
                value.close = clear_owned_pool
            clients.append(value)
            events.append(("create", label))
            return value

    monkeypatch.setattr(boto3, "Session", Session)
    actual_create = native.create_training_job

    def create(**kwargs):
        result = actual_create(**kwargs)
        requests[kwargs["TrainingJobName"]] = kwargs
        states[kwargs["TrainingJobName"]] = "InProgress"
        backend = work.engines[0].tool_backend
        assert backend._research_binding["phase"] == "CONSUMED"
        assert len(clients) == len(sessions) == 1
        checker = backend._input_checker
        before.update(
            checker=checker,
            config=checker.config,
            key=checker._key,
            cycle=backend._input_cycle,
            policy=backend.approval_policy,
            graph=backend._warm_graph(),
            calls=len(clients[0].calls),
        )
        clients[0].idle = True
        events.append(("native_accepted", "0"))
        return result

    def describe(**kwargs):
        name = kwargs["TrainingJobName"]
        request = requests[name]
        return {
            **{key: value for key, value in request.items() if key != "Tags"},
            "ResourceConfig": described_resource_config(request["ResourceConfig"]),
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":training-job/" + name,
            "TrainingJobStatus": states[name],
            "ModelArtifacts": {"S3ModelArtifacts": request["OutputDataConfig"]["S3OutputPath"] + "/model/"},
        }

    monkeypatch.setattr(native, "create_training_job", create)
    monkeypatch.setattr(native, "describe_training_job", describe)
    monkeypatch.setattr(
        native, "list_tags", lambda **kwargs: {"Tags": requests[kwargs["ResourceArn"].split("/")[-1]]["Tags"]}
    )

    def prepare(acquired, **kwargs):
        selected = PreparationCatalog(
            **dataclasses.asdict(catalog()), exclusions_ref=inputs.prepared.exclusions_ref
        )
        return SessionPreparationStore(acquired, ProcessingCodingPreparationBackend(), selected)

    def make_sft(models, prepared):
        selected = current_inputs(inputs, s3, models.runtime_session_id)
        monkeypatch.setattr(models, "_verified_sft_model_snapshot", lambda reference: selected.model)
        monkeypatch.setattr(prepared, "_verified_sft_snapshot", lambda reference: selected.prepared)
        return SessionSftStore(models, prepared, adapter, catalog())

    monkeypatch.setattr(binding, "build_preparation_store", prepare)
    monkeypatch.setattr(binding, "build_sft_store", make_sft)

    async def scenario():
        reader = asyncio.create_task(_events(work))
        try:
            deadline = asyncio.get_running_loop().time() + 10
            while not reader.done() and not (len(clients) > 1 and clients[1].calls):
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            backend = work.engines[0].tool_backend
            captured = tuple(backend._warm_pending_producers.values())
            assert len(native.calls) == 1 and len(captured) == 1 and captured[0].kind == "sft"
            pending = captured[0]
            original = pending.record
            assert original is backend._sft_store._records[pending.reference]
            assert original.extra["operation"] == "train" and original.extra["recipe_id"] == "full_v1"
            # The genuine retained original exists before the required freshness assertion.
            assert len(clients) == 2 and clients[0].closed, (
                "initial native SFT acknowledgement reused its cached guard client before first assessment"
            )
            assert len(clients[0].calls) == before["calls"] and len(clients[1].calls) == 1
            assert events.index(("native_accepted", "0")) < events.index(("captured", "0"))
            assert events.index(("captured", "0")) < events.index(("close", "0"))
            assert events.index(("close", "0")) < events.index(("create", "1")) < events.index(("apply", "1"))
            assert (
                backend._input_checker.config == before["config"]
                and backend._input_checker._key == before["key"]
            )
            assert backend._input_cycle == before["cycle"] and backend.approval_policy is before["policy"]
            assert backend._warm_graph() == before["graph"] and len(sessions) == 1
            witness = backend._pending_batch_pause
            assert witness.precommit and witness.checker is backend._input_checker
            history = copy.deepcopy(work.agents[0].messages)
            install_output(adapter, native, s3, original, recipe="full_v1")
            states[original.id] = "Completed"
            terminal = await asyncio.wait_for(reader, 5)
            assert terminal[-1]["worker_finished"] is True
            assert terminal[-1]["response"]["action"] == "tool_agent_turn"
            assert backend._input_admission_data()["status"] == "admitted"
            assert backend._producer_key(pending) in backend._warm_consumed
            finals = [
                tool_payload(block["toolResult"])
                for message in work.agents[0].messages
                for block in message["content"]
                if "toolResult" in block and block["toolResult"].get("toolUseId") == USE
            ]
            assert len(finals) == 1
            assert finals[0]["event_type"] == "original_sft_completed"
            assert finals[0]["data_only"] is True
            assert finals[0]["tool_use_id"] == USE and finals[0]["request_ref"] == pending.reference
            assert model.calls == 5 and len(native.calls) == len(work.agents) == 1
            assert work.agents[0].messages[: len(history)] == history
            assert backend._sft_store._records[pending.reference] is original
            assert backend._warm_pending_producers == {} and backend._input_hold is False
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert len(clients) == 3 and len(clients[2].calls) == 1
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
