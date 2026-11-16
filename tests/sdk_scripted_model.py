"""Scripted Strands model used by the telemetry and retry tests (no provider is contacted)."""

from __future__ import annotations

import copy
import json

from strands.models import Model

USAGE = {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, "metrics": {"latencyMs": 0}}


class ScriptedModel(Model):
    """Each step is an exception to raise, ``("text", value)`` or ``("tool", use_id, name, arguments)``."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.messages_seen = []

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "neutral-scripted-model"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("ordinary installed SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.messages_seen.append(copy.deepcopy(messages))
        step = self.script[self.calls]
        self.calls += 1
        if isinstance(step, BaseException):
            raise step
        yield {"messageStart": {"role": "assistant"}}
        if step[0] == "tool":
            _, identifier, name, arguments = step
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": identifier, "name": name}},
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
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": step[1]}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": copy.deepcopy(USAGE)}
