"""Fresh promotion must retain canonical row identity after its successful CAS."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json

import pytest
from normal_sft_cold_fixture import close, setup
from test_normal_prepared_promotion import (
    _exact_object,
    _fresh_unpromoted_original,
)
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.control_plane.normal_run_checkpoint import activate_normal_run


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def test_prepared_promotion_refuses_canonical_distinct_equal_row_after_task_cas(
    monkeypatch, tmp_path,
):
    """One declared DDB race changes only a fresh history counter after TASK CAS."""
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            context, policy, original, key, before, prepared_raw = (
                await _fresh_unpromoted_original(state, monkeypatch)
            )
            # This is declared DATA on the same fresh Source-created row. The
            # real promotion loads it normally before obtaining its expected row.
            with state.ports.lock:
                current = copy.deepcopy(state.ports.items[key])
                assert current == before
                row = json.loads(current["record"]["S"])
                row["history"].append({"counter": 1})
                raw = _canonical(row)
                current["record"] = {"S": raw}
                current["record_sha256"] = {
                    "S": hashlib.sha256(raw.encode()).hexdigest(),
                }
                state.ports.items[key] = current

            actual_update = state.ports.update_item
            successful_rows, changed_rows = [], []

            def change_after_successful_task_cas(**values):
                response = actual_update(**values)
                raw = values.get("ExpressionAttributeValues", {}).get(
                    ":body", {},
                ).get("S")
                if (
                    values["Key"] == {"job_name": before["job_name"]}
                    and type(raw) is str
                    and "task_checkpoint_ref" in json.loads(raw)
                ):
                    with state.ports.lock:
                        successful = copy.deepcopy(state.ports.items[key])
                        assert successful["record"]["S"] == raw
                        successful_body = json.loads(raw)
                        assert type(successful_body["history"][-1]["counter"]) is int
                        assert successful_body["history"][-1]["counter"] == 1
                        assert _exact_object(
                            state, successful_body["task_checkpoint_ref"],
                        ) == prepared_raw
                        changed = copy.deepcopy(successful)
                        changed_body = copy.deepcopy(successful_body)
                        changed_body["history"][-1]["counter"] = True
                        changed_raw = _canonical(changed_body)
                        # Python equality aliases bool/int; canonical JSON does not.
                        assert changed_body == successful_body
                        assert changed_raw != raw
                        restored = copy.deepcopy(changed_body)
                        restored["history"][-1]["counter"] = 1
                        assert _canonical(restored) == raw
                        changed["record"] = {"S": changed_raw}
                        changed["record_sha256"] = {
                            "S": hashlib.sha256(changed_raw.encode()).hexdigest(),
                        }
                        state.ports.items[key] = changed
                        successful_rows.append(successful)
                        changed_rows.append(copy.deepcopy(changed))
                return response

            monkeypatch.setattr(
                state.ports, "update_item", change_after_successful_task_cas,
            )
            model_calls = state.model.calls
            native_read_count = len(state.native.reads)
            source_calls = len(state.work.seen_prompts)
            try:
                with activate_normal_run(context), pytest.raises(ValueError):
                    context.promote_prepared_task(policy=policy)
            finally:
                # Validate the genuine native/status/successful-CAS/race boundary
                # even when the regression is RED with DID_NOT_RAISE.
                assert len(successful_rows) == len(changed_rows) == 1
                assert original.id in state.native.reads[native_read_count:]
                successful_body = json.loads(successful_rows[0]["record"]["S"])
                changed_body = json.loads(changed_rows[0]["record"]["S"])
                assert successful_body == changed_body
                assert _canonical(successful_body) != _canonical(changed_body)
                assert type(changed_body["history"][-1]["counter"]) is bool
                with state.ports.lock:
                    assert state.ports.items[key] == changed_rows[0]
                assert _exact_object(
                    state, changed_body["prepared_checkpoint_ref"],
                ) == prepared_raw
                assert _exact_object(
                    state, changed_body["task_checkpoint_ref"],
                ) == prepared_raw
                assert state.model.calls == model_calls
                assert len(state.work.seen_prompts) == source_calls == 1
                assert len(state.native.calls) == 1 and state.native.stops == []
                assert state.loaded == [] and state.invoked == []
        finally:
            await close(state)

    asyncio.run(scenario())
