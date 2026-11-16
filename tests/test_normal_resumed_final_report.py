"""The final report of a resumed run supersedes the original worker's pause-turn report.

The original worker may end on its own (a source refusal, a closed wait) and retain its
pause-turn response under the run row's single-assignment `report_checkpoint_ref`. A
replacement that later claims and finishes the completion publishes its own report. The
conflict refusal for a different second report stays; the replacement uses the separate
single-assignment `resumed_report_checkpoint_ref`, allowed only on a row whose completion
invocation a replacement has started. `get_prompt_report` returns the original's report
until the resumed one exists and never returns the original's once it does.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json

import pytest
from normal_sft_cold_fixture import FINAL_TEXT, close, cloud_data, original_precommit, setup
from test_normal_prompt_report_durability import (
    RUN,
    enrolled,
    mutation_calls,
    passive_read,
    report_parts,
    retain,
    rewrite_record,
    unavailable,
)
from test_normal_run_checkpoint import normal_work as normal_work
from test_normal_sft_cold_continuation import child
from test_normal_wake_stream_transition import _deliver_stream, _fresh_actual_task_stream, _run_image
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime
from amlas.async_callback import lambda_handler as callback_source
from amlas.control_plane import normal_run_checkpoint as checkpoint

DESCRIPTOR = {
    "uri": "s3://declared-bucket/sessions/declared/prompt-runs/declared/reports/" + "a" * 64 + ".json",
    "version_id": "declared-version",
    "sha256": "a" * 64,
    "byte_count": 512,
    "format": "json",
}


async def read_report(run_id):
    return [
        event async for event in runtime.agent_invocation({"action": "get_prompt_report", "run_id": run_id})
    ]


def terminal_of(events):
    return next(event for event in reversed(events) if event.get("event") == "run_completed")


def test_replacement_final_report_supersedes_the_original_pause_report(monkeypatch, tmp_path):
    """Cold fixture: the original ends with its pause-turn report; a replacement finishes the run."""

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            run_key = "PROMPT_RUN#" + state.run_id
            # The original worker ends by itself (the fixture's source-refusal closure) while its SFT
            # is still running natively; its cleanup retains the early-end response (a held or
            # pause-turn conversation) under the run row's single-assignment report pointer.
            await close_original_worker(state.work, state.reader)
            row = json.loads(state.ports.items[run_key]["record"]["S"])
            assert type(row.get("report_checkpoint_ref")) is dict, "the original did not retain its report"
            original_pointer = copy.deepcopy(row["report_checkpoint_ref"])
            before = await read_report(state.run_id)
            assert [item["status"] for item in before] == ["available"], before
            early = before[0]["report"]
            assert early["status"] == "response_received" and early["worker_finished"] is True
            assert FINAL_TEXT not in early["response"]["message"]
            print(
                "ORIGINAL_EARLY_REPORT="
                + json.dumps(
                    {"action": early["response"]["action"], "message": early["response"]["message"][:100]},
                    sort_keys=True,
                )
            )
            data = cloud_data(state, completed=True)
            answer = child(data, tmp_path / "pause-report-then-replacement")
            final = terminal_of(answer["events"])
            print(
                "REPLACEMENT_TERMINAL="
                + json.dumps(
                    {k: final.get(k) for k in ("status", "error_type", "worker_finished")}, sort_keys=True
                )
            )
            print(
                "REPORT_AFTER_REPLACEMENT="
                + json.dumps(
                    [
                        {
                            "status": item.get("status"),
                            "message": (item.get("report") or {}).get("response", {}).get("message", "")[:80],
                        }
                        for item in answer["report"]
                    ],
                    sort_keys=True,
                )
            )
            assert final["status"] == "response_received", (
                "the replacement finished the run but its final report was refused: " + repr(final)
            )
            assert FINAL_TEXT in final["response"]["message"]
            assert final["response"]["message"] != early["response"]["message"]
            assert answer["report"] == [
                {
                    "event": "prompt_report",
                    "status": "available",
                    "run_id": state.run_id,
                    "found": True,
                    "report": final,
                }
            ], "get_prompt_report must return the replacement's final report, not the original's early report"
            # No duplicate effects: the same two generation jobs and one pair job as the control.
            assert len(answer["new_training"]) == 2 and len(answer["new_processing"]) == 1
            after = json.loads(answer["transport"]["ddb"][run_key]["record"]["S"])
            assert after["report_checkpoint_ref"] == original_pointer, "the original pointer was changed"
            assert type(after.get("resumed_report_checkpoint_ref")) is dict
            assert after["resumed_report_checkpoint_ref"] != original_pointer
            # A second fresh replacement reads the same final report without any effect.
            again = child(answer["transport"], tmp_path / "final-report-reader", read_report=True)
            assert again["report"] == answer["report"]
            assert again["new_training"] == again["new_processing"] == again["invoked"] == []
        finally:
            await close(state)

    asyncio.run(scenario())


def test_resumed_report_needs_a_started_replacement_and_changes_nothing_otherwise(normal_work):
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    first = passive_read()
    items, objects, mutations = (
        copy.deepcopy(ports.items),
        copy.deepcopy(ports.objects),
        mutation_calls(ports),
    )
    envelope, response = report_parts()
    response["message"] = "Declared final report of a resumed run."
    with pytest.raises(ValueError):
        runtime._retain_prompt_report(RUN, envelope, response, session_hash, _resumed=True)
    assert passive_read() == first
    assert ports.items == items and ports.objects == objects and mutation_calls(ports) == mutations
    assert model.calls == 0 and not work.engines


def test_resumed_report_is_single_assignment_and_never_overwrites_the_original(normal_work, monkeypatch):
    # The gate (a started replacement invocation) is proved by the cold fixture test; here it is
    # opened to exercise the pointer rules on the light fixture.
    monkeypatch.setattr(checkpoint, "_replacement_started", lambda record: True)
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    first = passive_read()
    row_key = "PROMPT_RUN#" + RUN
    original_pointer = json.loads(ports.items[row_key]["record"]["S"])["report_checkpoint_ref"]
    envelope, final = report_parts()
    final["message"] = "Declared final report of a resumed run."
    runtime._retain_prompt_report(RUN, envelope, final, session_hash, _resumed=True)
    read = passive_read()
    assert read["status"] == "available" and read["report"]["response"] == final
    assert read != first
    row = json.loads(ports.items[row_key]["record"]["S"])
    assert row["report_checkpoint_ref"] == original_pointer
    resumed_pointer = copy.deepcopy(row["resumed_report_checkpoint_ref"])
    assert resumed_pointer != original_pointer
    # An identical replay (a retry) keeps the winning version and mutates nothing.
    items, objects, mutations = (
        copy.deepcopy(ports.items),
        copy.deepcopy(ports.objects),
        mutation_calls(ports),
    )
    runtime._retain_prompt_report(RUN, envelope, final, session_hash, _resumed=True)
    assert ports.items == items and ports.objects == objects and mutation_calls(ports) == mutations
    # A different second final report is refused and nothing changes.
    _envelope, other = report_parts()
    other["message"] = "Declared different final report."
    with pytest.raises(ValueError):
        runtime._retain_prompt_report(RUN, envelope, other, session_hash, _resumed=True)
    # The original path keeps its conflict refusal for any different report as well.
    with pytest.raises(ValueError):
        runtime._retain_prompt_report(RUN, envelope, other, session_hash)
    assert ports.items == items and ports.objects == objects and mutation_calls(ports) == mutations
    assert passive_read() == read
    assert model.calls == 0 and not work.engines


def test_a_damaged_resumed_pointer_fails_closed_and_never_revives_the_pause_report(normal_work, monkeypatch):
    monkeypatch.setattr(checkpoint, "_replacement_started", lambda record: True)
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    envelope, final = report_parts()
    final["message"] = "Declared final report of a resumed run."
    runtime._retain_prompt_report(RUN, envelope, final, session_hash, _resumed=True)
    assert passive_read()["report"]["response"] == final
    rewrite_record(ports, lambda record: record["resumed_report_checkpoint_ref"].update(sha256="f" * 64))
    assert passive_read() == unavailable(), "a damaged final pointer fell back to the pause-turn report"
    assert model.calls == 0 and not work.engines


def test_a_resumed_pointer_on_a_row_without_a_started_replacement_is_an_invalid_row(normal_work):
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    rewrite_record(
        ports, lambda record: record.update(resumed_report_checkpoint_ref=copy.deepcopy(DESCRIPTOR))
    )
    assert passive_read() == unavailable()
    assert model.calls == 0 and not work.engines


def test_replacement_started_gate_accepts_only_a_started_invocation_of_an_sft_row():
    started = {
        "stage": "sft",
        "continuation": {
            "schema_version": 3,
            "status": "consumed-ready",
            "invocation_state": "invoking",
        },
    }
    assert checkpoint._replacement_started(started) is True
    assert checkpoint._replacement_started({**started, "stage": "enrolled"}) is False
    assert checkpoint._replacement_started({"stage": "sft"}) is False
    for change in (
        {"invocation_state": "ready"},
        {"status": "released"},
        {"schema_version": 1},
        {"invocation_state": None},
    ):
        row = {**started, "continuation": {**started["continuation"], **change}}
        assert checkpoint._replacement_started(row) is False, change


def _linkage_body(**extra):
    return {
        "schema_version": 1,
        "logical_run_id": "6" * 32,
        "session_sha256": "7" * 64,
        "scope": {"account_id": "declared"},
        "table_id": "12345678-1234-4234-9234-00000000000a",
        "private_store_correlation": "12345678-1234-4234-9234-00000000000b",
        "admitted_prompt": "Declared prompt.",
        "history": [],
        "stage": "sft",
        "sft": {"native_job_name": "declared"},
        "task_checkpoint_ref": {"uri": "s3://declared"},
        **extra,
    }


@pytest.mark.parametrize("field", ["report_checkpoint_ref", "resumed_report_checkpoint_ref"])
def test_stream_linkage_ignores_the_report_pointers(field):
    plain = callback_source._normal_stream_linkage(_linkage_body())
    assert (
        callback_source._normal_stream_linkage(_linkage_body(**{field: copy.deepcopy(DESCRIPTOR)})) == plain
    ), "a report pointer appended after the TASK transition changed the immutable stream linkage"


def test_stream_linkage_still_binds_the_task_and_association():
    plain = callback_source._normal_stream_linkage(_linkage_body())
    assert (
        callback_source._normal_stream_linkage(_linkage_body(task_checkpoint_ref={"uri": "s3://other"}))
        != plain
    )
    assert callback_source._normal_stream_linkage(_linkage_body(sft={"native_job_name": "other"})) != plain


@pytest.mark.parametrize("field", ["report_checkpoint_ref", "resumed_report_checkpoint_ref"])
def test_a_report_pointer_on_the_current_row_does_not_block_the_wake_of_a_task_transition(
    monkeypatch, tmp_path, field
):
    """The original ends (and retains its report) before the retried stream delivery reaches the Lambda."""

    async def scenario():
        async with _fresh_actual_task_stream(monkeypatch, tmp_path) as case:
            baseline = len(case.runtime.calls)
            changed = _run_image(case.after, **{field: copy.deepcopy(DESCRIPTOR)})
            digest = hashlib.sha256(changed["record"]["S"].encode()).hexdigest()
            assert changed["record_sha256"] == {"S": digest}
            with case.state.ports.lock:
                case.state.ports.items[case.run_key] = copy.deepcopy(changed)
            _deliver_stream(case, [copy.deepcopy(case.record)])
            assert len(case.runtime.calls) == baseline + 1, (
                "a report pointer on the current row refused the wake of a real TASK transition"
            )
            assert json.loads(case.runtime.calls[-1]["payload"]) == {
                "action": "resume_prompt_run",
                "run_id": case.context.run_id,
            }

    asyncio.run(scenario())
