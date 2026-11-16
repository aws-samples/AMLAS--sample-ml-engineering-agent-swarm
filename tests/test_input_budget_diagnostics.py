"""Closed budget DATA from genuine SDK tool output; harmless neutral fixtures."""

from __future__ import annotations

import copy
import json

import pytest
from strands import tool
from test_input_admission_wiring import ClientFixture, ModelFixture, configure_guard

from amlas.config.settings import Settings
from amlas.tui.chat import input_admission
from amlas.tui.chat import tool_binding_backend as binding

LIMIT = 65536


@pytest.mark.parametrize(
    "name,payload,measurement",
    [
        ("research_fetch_urls", "ordinary metadata " * 5000, "LOWER_BOUND"),
        ("research_fetch_urls", "é" * 35000, "EXACT"),
        ("ordinary_unlisted_metadata", "ordinary metadata " * 5000, "LOWER_BOUND"),
    ],
    ids=["known-character-limit", "known-utf8-limit", "unlisted-name-not-echoed"],
)
def test_actual_registered_sdk_budget_hold_has_closed_name_and_measurement(
    monkeypatch, name, payload, measurement
):
    client = ClientFixture(deny=lambda _text: False)
    configure_guard(monkeypatch, client)
    calls = []

    def metadata_result() -> str:
        """Return ordinary synthetic metadata without any provider or lifecycle call."""
        calls.append("returned")
        return payload

    metadata_result.__name__ = name
    registered = tool(metadata_result)
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [registered])

    class SelectedToolModel(ModelFixture):
        async def stream(self, *args, **kwargs):
            async for event in super().stream(*args, **kwargs):
                event = copy.deepcopy(event)
                start = event.get("contentBlockStart", {}).get("start", {})
                if "toolUse" in start:
                    start["toolUse"]["name"] = name
                yield event

    model = SelectedToolModel()
    backend = binding.ToolBindingBackend(Settings(), model=model)
    turn = backend.chat_with_tools("ordinary metadata request")
    assert calls == ["returned"] and model.calls == 1
    assert input_admission.MAX_INPUT_BYTES == LIMIT
    assert turn.input_admission["status"] == "held"
    records = turn.input_admission["records"]
    assert len(records) <= 128
    record = records[-1]
    assert record["boundary"] == "TOOL_RESULT" and record["reason"] == "INPUT_LIMIT"
    assert record["outcome"] == "UNKNOWN"
    assert record["data_only"] is True
    assert record["native_admission"] is record["quality_success"] is False
    # Real SDK emits the original string as a text ContentBlock, assessed once.
    assessed = json.dumps([{"text": payload}], ensure_ascii=False, separators=(",", ":"))
    assert record["tool_name"] == (name if name == "research_fetch_urls" else "UNKNOWN")
    assert record["limit_bytes"] == LIMIT
    assert record["character_count"] == len(assessed)
    assert record["byte_measurement"] == measurement
    if measurement == "EXACT":
        assert record["byte_count"] == len(assessed.encode("utf-8"))
        assert record["byte_count_lower_bound"] is None
    else:
        assert record["byte_count"] is None
        assert record["byte_count_lower_bound"] == len(assessed[: LIMIT + 1].encode("utf-8"))
        assert LIMIT < record["byte_count_lower_bound"] <= 4 * (LIMIT + 1)
    assert record["keyed_digest"] is None
    assert all(len(call["content"][0]["text"]["text"].encode()) <= LIMIT for call in client.calls)
    assert len(client.calls) == 1, "oversized projection reached native assessment"
    encoded = json.dumps(turn.input_admission, ensure_ascii=False)
    assert payload not in encoded and "ordinary metadata " not in encoded
    assert not {"args", "rawbody", "url", "request_ref", "job_name", "job_arn"} & set(record)
    if name != "research_fetch_urls":
        assert name not in encoded, "an uncurated registered name was reflected"
    assert not backend._model_acquisition_store._records
    assert not backend._acquisition_store._records
    assert not backend._warm_pending_producers and backend._warm_pending is None
