"""Actual MODEL SDK history DATA must bind the same retained original before resume."""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_warm_model_continuation import complete_fixture, tool_payload

from amlas import app as runtime

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def replace_only_history_data(tool_result, field, value):
    """Change one ordinary serialized DATA value, preserving toolUseId/content shape."""
    original = tool_payload(tool_result)
    changed = {**original, field: value}
    assert set(changed) == set(original)
    for content in tool_result["content"]:
        if type(content.get("json")) is dict:
            content["json"] = changed
            return
        if type(content.get("text")) is str:
            decoded = json.loads(content["text"])
            if type(decoded) is dict:
                content["text"] = json.dumps(changed, sort_keys=True)
                return
    raise AssertionError("actual original SDK ToolResult DATA required")


@pytest.mark.parametrize("field", ["request_ref", "job_name", "job_arn"])
def test_normal_model_resume_refuses_substituted_historical_data_before_provider(
    monkeypatch,
    tmp_path,
    field,
):
    work = connected_work(monkeypatch)

    async def scenario():
        async def collect():
            return [event async for event in runtime.agent_invocation({"prompt": work.literal})]

        reader = asyncio.create_task(collect())
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not work.observing.is_set() and not reader.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert len(work.engines) == len(work.agents) == len(work.native.requests) == 1
            assert work.model.calls == 2 and work.originals[0][0] == "original-model-use"
            original = work.originals[0][1]
            assert original["accepted"] is True and original["status"] == "submitted"
            backend, agent = work.engines[0].tool_backend, work.agents[0]
            reference = original["request_ref"]
            record = backend._model_acquisition_store._records[reference]
            handle = record.handle
            assert backend._warm_pending.record is record and backend._warm_pending.agent is agent
            graph = backend._warm_graph()
            matching = [
                block["toolResult"]
                for message in agent.messages
                for block in message["content"]
                if block.get("toolResult", {}).get("toolUseId") == "original-model-use"
            ]
            assert len(matching) == 1 and tool_payload(matching[0]) == original
            replacement = {
                "request_ref": "model_acq_" + "b" * 32,
                "job_name": "ordinary-other-model-job",
                "job_arn": handle.arn.rsplit("/", 1)[0] + "/ordinary-other-model-job",
            }[field]
            replace_only_history_data(matching[0], field, replacement)
            assert matching[0]["toolUseId"] == "original-model-use"
            assert tool_payload(matching[0])[field] == replacement
            before = copy.deepcopy(agent.messages)
            reads_before = len(work.s3.reads)
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=handle, adapter=work.adapter, native=work.native),
            )
            events = await asyncio.wait_for(reader, 5)
            continuation = events[-1]["response"]["data"]["warm_model_continuation"]
            assert continuation["disposition"] == "refused", (
                "MODEL unique toolUseId alone must not allow a substituted historical original identity"
            )
            assert continuation["diagnostic"]["phase"] == "history_causality"
            assert continuation["diagnostic"]["reason"] == "history_causality_changed"
            assert work.model.calls == 2, "history refusal must precede another model/provider call"
            assert agent.messages == before
            assert backend._model_acquisition_store._records[reference] is record
            assert backend._warm_graph() == graph and len(graph) == 7
            assert len(work.native.requests) == 1 and len(work.s3.reads) > reads_before
            assert all(request.get("VersionId") for request in work.s3.reads)
            assert agent.callback_handler is work.callbacks[0]
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
