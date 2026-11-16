"""Actual SDK/Bedrock format must use one original append-only completion event.

Transport is strict installed-SDK-shaped, neutral and credential-free. No provider,
model tensors, ML effect, fabricated store/ref or history authorization is used.
"""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from types import SimpleNamespace
from typing import Any

import botocore.session
import pytest
from botocore.exceptions import ClientError
from botocore.hooks import HierarchicalEmitter
from botocore.validate import validate_parameters
from strands.models import BedrockModel
from test_runtime_connected_warm_model import connected_work, started_original
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401
from test_runtime_warm_model_continuation import complete_fixture

from amlas import app as runtime


class StrictHistoryBedrock:
    """Real BedrockModel transport with model-shaped messages and history policy."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.service = botocore.session.get_session().get_service_model("bedrock-runtime")
        self.meta = SimpleNamespace(region_name="us-west-2", service_model=self.service, events=HierarchicalEmitter())
        self.requests: list[dict[str, Any]] = []
        self.prefix: list[dict[str, Any]] | None = None
        self.rejections: list[dict[str, bool]] = []

    def _events(self, *, tool: bool, text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = [{"messageStart": {"role": "assistant"}}]
        if tool:
            events += [
                {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {
                    "toolUseId": "original-model-use", "name": "acquire_model_weights",
                }}}},
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {
                    "input": json.dumps({"catalog_id": "qwen3_5_9b", "catalog_version": "1"}),
                }}}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        else:
            events += [
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        events.append({"metadata": {"usage": {"inputTokens": 4, "outputTokens": 3, "totalTokens": 7},
                                    "metrics": {"latencyMs": 1}}})
        shape = self.service.operation_model("ConverseStream").output_shape.members["stream"]
        for event in events:
            validate_parameters(event, shape)
        return events

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        validate_parameters(kwargs, self.service.operation_model("ConverseStream").input_shape)
        request = copy.deepcopy(kwargs)
        self.requests.append(request)
        if self.prefix is not None:
            messages = request["messages"]
            tail = messages[-1]["role"] == "assistant"
            mutated = messages[:len(self.prefix)] != self.prefix
            if (self.mode == "tail" and tail) or (self.mode == "prefix" and mutated):
                self.rejections.append({"trailing_assistant": tail, "prior_prefix_mutated": mutated})
                raise ClientError({
                    "Error": {"Code": "ValidationException", "Message": "neutral completed-history contract refused"},
                    "ResponseMetadata": {"HTTPStatusCode": 400, "RequestId": str(uuid.uuid4())},
                }, "ConverseStream")
        count = len(self.requests)
        assert count in (1, 2, 3), "no provider request replay"
        events = self._events(tool=count == 1, text=(
            "The original acquisition is pending." if count == 2 else "Original exact-version artifacts are verified."
        ))
        return {"stream": iter(events), "ResponseMetadata": {"HTTPStatusCode": 200}}


class NeutralSession:
    region_name = "us-west-2"

    def __init__(self, client: StrictHistoryBedrock) -> None:
        self.transport = client

    def client(self, service_name: str, **_kwargs: Any) -> StrictHistoryBedrock:
        assert service_name == "bedrock-runtime"
        return self.transport


@pytest.mark.parametrize("mode", ["tail", "prefix"])
def test_connected_original_sdk_uses_append_only_source_completion_event(monkeypatch, tmp_path, mode: str) -> None:
    transport = StrictHistoryBedrock(mode)
    model = BedrockModel(boto_session=NeutralSession(transport), model_id="neutral-sonnet55-history", max_tokens=512)
    work = connected_work(monkeypatch, model=model)

    async def scenario() -> None:
        async def read():
            return [event async for event in runtime.agent_invocation({"prompt": work.literal})]

        reader = asyncio.create_task(read())
        try:
            backend = await started_original(work)
            assert len(transport.requests) == 2
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            agent = work.agents[0]
            before = copy.deepcopy(agent.messages)
            assert before[-1]["role"] == "assistant"
            transport.prefix = copy.deepcopy(model._format_bedrock_messages(before))
            graph = backend._warm_graph()
            reference = work.originals[0][1]["request_ref"]
            handle = backend._model_acquisition_store._records[reference].handle
            complete_fixture(monkeypatch, tmp_path, SimpleNamespace(handle=handle, adapter=work.adapter, native=work.native))
            events = await asyncio.wait_for(reader, 5)
            terminal = events[-1]
            assert terminal["response"]["data"]["warm_model_continuation"]["disposition"] == "continued_original", (
                "original warm continuation violated the completed provider history contract", transport.rejections,
            )
            assert transport.rejections == []
            sent = transport.requests[-1]["messages"]
            assert sent[:len(transport.prefix)] == transport.prefix
            assert agent.messages[:len(before)] == before, "prior toolResult/thinking/history must stay immutable"
            assert len(sent) == len(transport.prefix) + 1 and sent[-1]["role"] == "user"
            blocks = sent[-1]["content"]
            assert len(blocks) == 1 and set(blocks[0]) == {"text"}
            event = json.loads(blocks[0]["text"])
            assert event == {
                "schema_version": 1, "event_type": "original_model_acquisition_completed", "data_only": True,
                "request_ref": reference, "tool_use_id": "original-model-use", "status": "SUCCEEDED",
                "acquired": True, "file_count": 9, "manifest_roles": ["receipt", "commit", "lineage"],
            }
            assert len(blocks[0]["text"].encode("utf-8")) <= 512
            assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
            assert len(transport.requests) == 3 and len(work.seen_prompts) == 1
            assert work.seen_prompts[0][0] == work.literal
            assert backend._warm_graph() == graph
            assert backend._model_acquisition_store._records[reference].handle is handle
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert all(read["VersionId"] for read in work.s3.reads)
        finally:
            if not reader.done():
                reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    asyncio.run(scenario())


def test_completion_event_preserves_existing_maximum_unicode_tool_identifier() -> None:
    from amlas.tui.chat.tool_binding_backend import _OriginalModelCompletionEvent

    identifier = chr(0x1F642) * 256
    reference = "model_acq_" + "a" * 32
    blocks = _OriginalModelCompletionEvent(reference, identifier).as_content()
    assert len(blocks) == 1 and set(blocks[0]) == {"text"}
    encoded = blocks[0]["text"].encode("utf-8")
    assert len(encoded) <= 4096
    event = json.loads(blocks[0]["text"])
    assert event["tool_use_id"] == identifier and event["request_ref"] == reference
    assert event["data_only"] is True and event["file_count"] == 9
