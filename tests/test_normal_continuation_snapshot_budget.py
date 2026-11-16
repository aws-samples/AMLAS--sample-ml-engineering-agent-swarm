"""One O2/O3 continuation budget discriminator over actual normal strict ports.

Hypothesis: a genuine SDK snapshot admitted by unchanged compact TASK preflight
can exceed the same160KiB cap when the completed snapshot enters the owned row.
Pass condition: either reject before original effect, or an admitted original can
claim its qualified completion. A cap refusal only at the later row write is RED.
No cap/parser/auth/CAS is mocked. Scripted SDK/native fixtures prove Source only,
not real ML, AWS authorization, model quality or deployed recovery.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json

import pytest
from bedrock_agentcore.runtime import BedrockAgentCoreContext
from normal_sft_cold_fixture import SFT_USE, close, original_precommit, setup
from strands import Agent
from test_runtime_connected_warm_model import close_original_worker

from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint


@pytest.mark.parametrize("near_cap", [False, True], ids=["small-control", "task-boundary"])
def test_task_admitted_sdk_state_fits_the_qualified_completion_row(monkeypatch, tmp_path, near_cap):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        actual_snapshot = Agent.take_snapshot
        selected_sizes = []

        def genuine_snapshot(agent, *args, **kwargs):
            context = checkpoint.current_normal_run()
            uses = [
                block.get("toolUse")
                for message in agent.messages[-1:]
                for block in message.get("content", [])
                if type(block) is dict and type(block.get("toolUse")) is dict
            ]
            if context is None or not any(use.get("toolUseId") == SFT_USE for use in uses):
                return actual_snapshot(agent, *args, **kwargs)
            call = checkpoint._causal_call(SFT_USE, agent.messages)
            if near_cap:
                low, high = 0, checkpoint._MAX_BYTES
                while low < high:
                    middle = (low + high + 1) // 2
                    agent.state.set("ordinary_continuation_budget_note", "n" * middle)
                    snapshot = actual_snapshot(agent, *args, **kwargs)
                    try:
                        checkpoint._budget_sft_snapshot(context, call, snapshot.to_dict())
                    except ValueError as error:
                        if str(error) != "normal run DATA size exceeded":
                            raise
                        high = middle - 1
                    else:
                        low = middle
                # Small timestamp/rendering headroom; the Source bound remains exact.
                note_bytes = max(0, low - 32)
            else:
                note_bytes = 64
            agent.state.set("ordinary_continuation_budget_note", "n" * note_bytes)
            snapshot = actual_snapshot(agent, *args, **kwargs)
            checkpoint._budget_sft_snapshot(context, call, snapshot.to_dict())
            selected_sizes.append(note_bytes)
            return snapshot

        monkeypatch.setattr(Agent, "take_snapshot", genuine_snapshot)
        try:
            await original_precommit(state, monkeypatch)
            assert selected_sizes and len(state.native.calls) == 1
            await close_original_worker(state.work, state.reader)
            from test_qwen_sft_backend import install_output

            install_output(
                state.backend._sft_store._adapter,
                state.native,
                state.objects,
                state.original,
                recipe="full_v1",
            )
            state.native.states[state.original.id] = "Completed"
            transport = BedrockAgentCoreContext.get_session_id()
            context = checkpoint.prepare_normal_run(
                state.run_id,
                hashlib.sha256(transport.encode()).hexdigest(),
                transport_session_id=transport,
            )
            policy = runtime._runtime_prompt_policy()
            with checkpoint.activate_normal_run(context):
                task, _parents, _original, _binding = context.load_qualified_task(policy=policy)
                assert task["schema_version"] == 2
                ready = copy.deepcopy(task)
                event = {
                    "schema_version": 1,
                    "event_type": "original_sft_completed",
                    "data_only": True,
                    "request_ref": "sft_" + task["operation_digest"][:32],
                    "tool_use_id": task["pending_tool_use_id"],
                    "status": "verified_completed",
                }
                result = {
                    "toolUseId": task["pending_tool_use_id"],
                    "status": "success",
                    "content": [{"text": json.dumps(event, sort_keys=True, separators=(",", ":"))}],
                }
                ready["sdk_snapshot"]["data"]["messages"].append(
                    {"role": "user", "content": [{"toolResult": result}]}
                )

                def qualify_original():
                    context.qualify_original_completion(task, state.backend._sft_store, policy=policy)

                qualify_original()
                before_creates = len(state.native.calls)
                token = context.claim_completion(ready, policy=policy, qualify_original=qualify_original)
                context.qualify_completion(
                    token,
                    policy=policy,
                    qualify_original=qualify_original,
                    sdk_snapshot=ready["sdk_snapshot"],
                )
                row = json.loads(state.ports.items["PROMPT_RUN#" + state.run_id]["record"]["S"])
                assert row["continuation"]["schema_version"] == 3
                assert (
                    row["continuation"]["sdk_snapshot_sha256"]
                    == hashlib.sha256(checkpoint._canonical(ready["sdk_snapshot"]).encode()).hexdigest()
                )
                assert len(checkpoint._canonical(row).encode()) <= checkpoint._MAX_BYTES
                assert len(state.native.calls) == before_creates == 1
                assert state.native.stops == []
        finally:
            await close(state)

    asyncio.run(scenario())
