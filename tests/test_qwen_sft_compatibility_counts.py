"""Closed counted compatibility refusals contain no conversation payload."""

from __future__ import annotations

import json

import pytest
from test_qwen_sft_native_checkpoint import module
from test_qwen_sft_worker import module as worker_module


def test_compatibility_rows_are_counted_without_partial_training(monkeypatch):
    m = module()
    rows = [
        {
            "messages": [
                {"role": "developer", "content": "private fixture text"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            "tools": [],
        }
    ] * 2
    rows.append(
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": "private reasoning"},
                {"role": "user", "content": "later"},
                {"role": "assistant", "content": "answer"},
            ],
            "tools": [],
        }
    )
    raw = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"
    monkeypatch.setattr(m, "_read_exact_relative", lambda *_: raw)
    with pytest.raises(ValueError) as caught:
        m._load_split_encoded(0, "train.jsonl", {"record_count": 3}, None)
    assert caught.value.reason_counts == {"developer_role": 2, "historical_reasoning_unsupported": 1}
    assert "private" not in str(caught.value)


def test_worker_formats_only_closed_compatibility_counts():
    m = module()
    w = worker_module()
    assert callable(getattr(w, "_format_refusal", None)), "missing-symbol: safe refusal counts"
    value = w._format_refusal(m.ConversationCompatibilityError({"developer_role": 2}))
    assert json.loads(value)["reason_counts"] == {"developer_role": 2}
    assert "private" not in value
