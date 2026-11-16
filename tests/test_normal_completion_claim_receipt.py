"""Actual claim receipt capture with fresh DATA and explicitly inert boundaries.

Native transport/original/object qualification is not proven. Real decoder/CAS
code runs against declared strict ports; no runtime snapshot or private capture,
SDK loading, inference or cold recovery is involved.
"""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_normal_checkpoint_cas_contract import TABLE_ID, StrictDynamoPort, _item, _policy
from test_normal_completion_record_codec import _declared_sft_record

from amlas.control_plane import normal_run_checkpoint as checkpoint


def _declared_claim():
    context, record = _declared_sft_record()
    record.pop("continuation")
    digest = record["sft"]["operation_digest"]
    use_id = "declared-receipt-use"
    prefix = [
        {"role": "user", "content": [{"text": "Declared receipt capture DATA."}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": use_id, "name": "submit_qwen_sft", "input": {}}}],
        },
    ]
    snapshot = {
        "scope": "agent",
        "schema_version": "1.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "app_data": {},
        "data": {
            "messages": copy.deepcopy(prefix),
            "state": {},
            "conversation_manager_state": {},
            "interrupt_state": {},
        },
    }
    stored = {
        "schema_version": 1,
        "logical_run_id": context.run_id,
        "private_store_correlation": record["private_store_correlation"],
        "native_job_name": "declared-receipt-original",
        "native_job_arn": "arn:aws:sagemaker:us-west-2:111122223333:training-job/declared-receipt-original",
        "operation_digest": digest,
        "source_refs": ["model_acq_" + "4" * 32, "prep_" + "5" * 32],
        "causal_parent_ref": copy.deepcopy(record["task_checkpoint_ref"]),
        "sdk_snapshot": snapshot,
        "pending_tool_use_id": use_id,
        "causal_prefix": copy.deepcopy(prefix),
    }
    ready = copy.deepcopy(stored)
    event = {
        "schema_version": 1,
        "event_type": "original_sft_completed",
        "data_only": True,
        "request_ref": "sft_" + digest[:32],
        "tool_use_id": use_id,
        "status": "verified_completed",
    }
    result = {
        "toolUseId": use_id,
        "status": "success",
        "content": [{"text": json.dumps(event, sort_keys=True, separators=(",", ":"))}],
    }
    ready["sdk_snapshot"]["data"]["messages"].append({"role": "user", "content": [{"toolResult": result}]})
    return context, record, stored, ready


class DeclaredClaimPort(StrictDynamoPort):
    def __init__(self, context, item, mode):
        super().__init__(context, item)
        self.mode = mode

    def update_item(self, **kwargs):
        if self.mode == "cas_failure":
            changed = json.loads(self.item["record"]["S"])
            changed["admitted_prompt"] = "Declared concurrent before-CAS change."
            self.item = _item(self.context, changed)
        result = super().update_item(**kwargs)
        if self.mode == "readback_mismatch":
            changed = json.loads(self.item["record"]["S"])
            prior_digest = changed["continuation"]["sdk_snapshot_sha256"]
            changed["continuation"]["sdk_snapshot_sha256"] = (
                "0" * 64 if prior_digest != "0" * 64 else "1" * 64
            )
            self.item = _item(self.context, changed)
        return result


@pytest.mark.parametrize("mode", ["confirmed", "cas_failure", "readback_mismatch"])
def test_actual_claim_retains_receipt_only_after_confirmed_content(monkeypatch, mode):
    context, record, stored, ready = _declared_claim()
    expected_snapshot = copy.deepcopy(ready["sdk_snapshot"])
    expected_snapshot_sha256 = hashlib.sha256(checkpoint._canonical(expected_snapshot).encode()).hexdigest()
    port = DeclaredClaimPort(context, _item(context, record), mode)
    assert context._completion_receipt is None
    stored_bytes = checkpoint._canonical(stored).encode("utf-8")

    @contextmanager
    def declared_native(value):
        assert value is context
        yield port, TABLE_ID, lambda: None, None

    def declared_original(value, _record, table_id, job_name):
        assert value is context and table_id == TABLE_ID
        assert job_name == ready["native_job_name"]
        return SimpleNamespace(operation_digest=stored["operation_digest"]), None

    def declared_object(value, _binding, _ref, kind, qualify):
        assert value is context and kind == "tasks"
        qualify()
        return stored_bytes

    def no_original_grant():
        return None

    monkeypatch.setattr(checkpoint, "_native", declared_native)
    monkeypatch.setattr(checkpoint, "_original", declared_original)
    monkeypatch.setattr(checkpoint, "_read_object", declared_object)
    if mode == "confirmed":
        token = context.claim_completion(ready, policy=_policy(), qualify_original=no_original_grant)
        _committed_item, committed = checkpoint._load(context, port, TABLE_ID, lambda: None)
        expected = (
            token,
            checkpoint._canonical(committed["continuation"]),
            checkpoint._canonical(record["task_checkpoint_ref"]),
        )
        assert context._completion_receipt == expected
        assert committed["continuation"]["schema_version"] == 3
        assert committed["continuation"]["sdk_snapshot_sha256"] == expected_snapshot_sha256
        assert checkpoint._completed_task_snapshot(stored, stored["operation_digest"]) == expected_snapshot
        assert all(type(value) is str for value in expected)
        ready["sdk_snapshot"]["data"]["state"]["caller_change"] = True
        assert context._completion_receipt == expected
        _still_item, still = checkpoint._load(context, port, TABLE_ID, lambda: None)
        assert still["continuation"] == committed["continuation"]
        assert still["continuation"]["sdk_snapshot_sha256"] == expected_snapshot_sha256
        detached_snapshot = checkpoint._completed_task_snapshot(stored, stored["operation_digest"])
        assert detached_snapshot == expected_snapshot
        assert "caller_change" not in detached_snapshot["data"]["state"]
        assert detached_snapshot is not ready["sdk_snapshot"]
        assert port.commits == 1 and port.get_calls >= 2
    else:
        with pytest.raises(ValueError):
            context.claim_completion(ready, policy=_policy(), qualify_original=no_original_grant)
        assert context._completion_receipt is None
        assert port.commits == (0 if mode == "cas_failure" else 1)
