"""The cold test must wait for a normal response after partial native progress."""

import asyncio
from types import SimpleNamespace

import test_normal_sft_cold_continuation as cold
from test_normal_run_sft_checkpoint import wire_decode


def test_cold_lookup_waits_for_pending_normal_response(monkeypatch, tmp_path):
    async def scenario():
        state = SimpleNamespace(
            work=SimpleNamespace(engines=[]),
            ports=SimpleNamespace(items={}, objects={}),
            native=SimpleNamespace(
                calls=[], processing={}, reads=[], stops=[], states={"ordinary": "InProgress"}
            ),
            objects=SimpleNamespace(objects={}),
            loaded=[],
            invoked=[],
            rejected_cas=[],
            model=SimpleNamespace(calls=0, messages_seen=[]),
        )
        public = {"event": "prompt_run_status", "status": "available", "found": True}
        cleaned_after_response = []

        async def invocation(payload):
            if payload["action"] == "resume_prompt_run":
                state.work.engines.append(
                    SimpleNamespace(tool_backend=SimpleNamespace(_warm_graph=lambda: []))
                )
                state.native.reads.append("ordinary synthetic observation")
                await asyncio.sleep(0.05)
            yield public

        async def close(value):
            cleaned_after_response.append(value.reader.done())
            if not value.reader.done():
                value.reader.cancel()
                try:
                    await value.reader
                except asyncio.CancelledError:
                    pass

        monkeypatch.setattr(cold, "inert_runtime_setup", SimpleNamespace(__wrapped__=lambda _patch: None))
        monkeypatch.setattr(cold, "setup", lambda *args, **kwargs: state)
        monkeypatch.setattr(cold.runtime, "agent_invocation", invocation)
        monkeypatch.setattr(cold, "close", close)
        data = {"run_id": "a" * 32, "original_job": "ordinary", "states": {"ordinary": "InProgress"}}
        answer = wire_decode(await cold.cold_lookup(data, tmp_path, False))
        assert answer["events"] == [public], "cold harness returned before its normal pending response"
        assert cleaned_after_response == [True], (
            "cold harness cancelled a normal reader still producing its response"
        )

    asyncio.run(scenario())
