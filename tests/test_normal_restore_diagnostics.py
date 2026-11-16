"""Closed restoration logs through the normal SDK-session app entrypoint.

The existing normal enrollment fixture supplies strict inert native ports. Loader
and snapshot faults are read-boundary injections only; no restored graph, native
admission, producer permission or full recovery acceptance is manufactured.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from types import SimpleNamespace

import pytest
from test_normal_run_checkpoint import PROMPT, collect
from test_normal_run_checkpoint import normal_work as normal_work
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import engine as chat_engine

LOGGER = "amlas.app"
BARE = "normal_restore_unavailable"
# The final line names its cause in the text: logging extras do not reach the runtime log stream.
FINAL_FORMAT = "normal_restore_unavailable reason=final restore_stage=%s exception_class=%s"
RETRY_FORMAT = "normal_restore_unavailable reason=transient_retry attempt=%d attempts=%d restore_stage=%s exception_class=%s"
LOG_STAGES = ("run_prepare", "checkpoint_promote", "checkpoint_load", "snapshot_load", "continuation")
REVIEWED = (
    ValueError,
    TypeError,
    RuntimeError,
    AssertionError,
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    FileNotFoundError,
    PermissionError,
    TimeoutError,
    NotImplementedError,
)
TOKEN = re.compile(r"[A-Za-z_]+")
DETAIL = "ordinary restoration diagnostic detail"
FIELDS = {"restore_stage", "exception_class"}


class UnreviewedRestorationError(ValueError):
    """An unreviewed custom subclass is not a builtin error category."""


def _failure(error_type, calls):
    def refuse(*_args, **_kwargs):
        calls.append("fault")
        raise error_type(DETAIL)

    return refuse


def _inject_boundary(monkeypatch, stage, failure):
    if stage == "run_prepare":
        monkeypatch.setattr(checkpoint, "prepare_normal_run", failure)
        return
    if stage == "checkpoint_promote":
        # Read selectors reach only the closed catch; they grant no promotion.
        monkeypatch.setattr(checkpoint.NormalRunContext, "has_task_checkpoint", lambda self, **_: False)
        monkeypatch.setattr(checkpoint.NormalRunContext, "has_prepared_checkpoint", lambda self, **_: True)
        monkeypatch.setattr(checkpoint.NormalRunContext, "promote_prepared_task", failure)
        return

    # Fault injection reaches only the app catch. True is a read-selector seam,
    # never evidence that a TASK/graph is qualified or permission to continue.
    monkeypatch.setattr(checkpoint.NormalRunContext, "has_task_checkpoint", lambda self, **_: True)
    if stage == "checkpoint_load":
        monkeypatch.setattr(checkpoint.NormalRunContext, "load_qualified_task", failure)
        return

    marker = object()
    monkeypatch.setattr(
        checkpoint.NormalRunContext,
        "load_qualified_task",
        lambda self, **_: (marker, (), None, None),
    )

    def inert_engine(*, approval_policy, normal_restore):
        assert normal_restore[1] is marker
        assert approval_policy is not None
        return SimpleNamespace(
            tool_backend=SimpleNamespace(_normal_original_result={"status": "PENDING"}),
            load_normal_snapshot=failure,
        )

    monkeypatch.setattr(chat_engine, "ChatEngine", inert_engine)


def _selected_run(normal_work):
    _work, ports, model = normal_work
    events = collect({"prompt": PROMPT})
    assert events[-1]["status"] == "response_received"
    assert events[-1]["response"]["message"] == "Admitted plan remains descriptive."
    assert model.calls == 1
    started = next(event for event in events if event["event"] == "run_started")
    return started["run_id"], copy.deepcopy(ports.items), copy.deepcopy(ports.objects)


@pytest.mark.parametrize(
    ("stage", "error_type", "category"),
    [
        ("run_prepare", ValueError, "ValueError"),
        ("checkpoint_promote", ValueError, "ValueError"),
        ("checkpoint_load", RuntimeError, "RuntimeError"),
        ("checkpoint_promote", RuntimeError, "RuntimeError"),
        ("snapshot_load", RuntimeError, "RuntimeError"),
        ("checkpoint_load", UnreviewedRestorationError, "Exception"),
    ],
)
def test_normal_restore_failure_logs_only_closed_source_metadata(
    normal_work, monkeypatch, caplog, stage, error_type, category
):
    work, ports, model = normal_work
    run_id, items, objects = _selected_run(normal_work)
    calls = []
    _inject_boundary(monkeypatch, stage, _failure(error_type, calls))
    caplog.set_level(logging.WARNING, logger=LOGGER)
    caplog.clear()

    events = collect({"action": "resume_prompt_run", "run_id": run_id})

    assert events == [
        {"event": "resume_state", "state": "restoring", "run_id": run_id},
        {
            "event": "resume_state",
            "state": "failed",
            "run_id": run_id,
            "diagnostic": {"restore_stage": stage, "exception_class": category},
        },
    ]
    assert calls == ["fault"]
    assert model.calls == 1 and work.native.requests == []
    assert ports.items == items and ports.objects == objects
    records = [record for record in caplog.records if record.name == LOGGER]
    assert len(records) == 1, "restoration catch must retain one closed private diagnostic"
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == f"{BARE} reason=final restore_stage={stage} exception_class={category}"
    assert record.msg == FINAL_FORMAT and record.args == (stage, category)
    assert record.exc_info is None and record.stack_info is None
    standard = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
    assert set(record.__dict__) - standard == FIELDS
    metadata = {name: record.__dict__[name] for name in FIELDS}
    assert metadata == {
        "restore_stage": stage,
        "exception_class": category,
    }
    visible = json.dumps({"events": events, "metadata": metadata}) + caplog.text
    assert DETAIL not in visible and "UnreviewedRestorationError" not in visible


def test_normal_restore_warning_failure_preserves_the_same_closed_response(normal_work, monkeypatch):
    work, ports, model = normal_work
    run_id, items, objects = _selected_run(normal_work)
    calls = []
    _inject_boundary(monkeypatch, "run_prepare", _failure(ValueError, calls))

    def broken_warning(*_args, **_kwargs):
        raise RuntimeError(DETAIL)

    monkeypatch.setattr(logging.getLogger(LOGGER), "warning", broken_warning)
    events = collect({"action": "resume_prompt_run", "run_id": run_id})
    assert events == [
        {"event": "resume_state", "state": "restoring", "run_id": run_id},
        {
            "event": "resume_state",
            "state": "failed",
            "run_id": run_id,
            "diagnostic": {"restore_stage": "run_prepare", "exception_class": "ValueError"},
        },
    ]
    assert calls == ["fault"]
    assert model.calls == 1 and work.native.requests == []
    assert ports.items == items and ports.objects == objects
    assert DETAIL not in json.dumps(events)


def _direct_records(caplog, call):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    caplog.clear()
    call()
    return [record for record in caplog.records if record.name == LOGGER]


def _extras(record):
    standard = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
    return {name: value for name, value in record.__dict__.items() if name not in standard}


@pytest.mark.parametrize("stage", LOG_STAGES)
def test_final_restore_line_carries_reason_stage_and_category_in_its_text(caplog, stage):
    """A reader of the log stream sees only the text, so the final line states its fixed stage and category."""
    records = _direct_records(
        caplog, lambda: runtime._warn_normal_restore_failure(stage, RuntimeError(DETAIL))
    )
    assert [record.getMessage() for record in records] == [
        f"normal_restore_unavailable reason=final restore_stage={stage} exception_class=RuntimeError"
    ], "exactly one final line, with its cause in the text"
    record = records[0]
    assert record.levelno == logging.WARNING and record.exc_info is None and record.stack_info is None
    assert record.msg == FINAL_FORMAT and record.args == (stage, "RuntimeError")
    assert _extras(record) == {"restore_stage": stage, "exception_class": "RuntimeError"}
    assert DETAIL not in caplog.text


@pytest.mark.parametrize(
    "stage", ["report_retention", "declared unknown stage", "", "reason=transient_retry", "a\nb"]
)
def test_final_restore_line_of_a_stage_outside_the_fixed_set_falls_back_to_run_prepare(caplog, stage):
    records = _direct_records(caplog, lambda: runtime._warn_normal_restore_failure(stage, ValueError(DETAIL)))
    assert [record.getMessage() for record in records] == [
        "normal_restore_unavailable reason=final restore_stage=run_prepare exception_class=ValueError"
    ]
    assert records[0].args == ("run_prepare", "ValueError")
    assert _extras(records[0]) == {"restore_stage": "run_prepare", "exception_class": "ValueError"}


def test_every_final_restore_line_is_four_whole_key_value_tokens(caplog):
    """Every stage and every reviewed category gives token-safe text: no message, id, path or free text."""
    assert set(runtime._RESTORE_ERROR_CATEGORIES.values()) == {error_type.__name__ for error_type in REVIEWED}
    seen = set()
    for stage in (*LOG_STAGES, "declared unknown stage"):
        for error_type in (*REVIEWED, UnreviewedRestorationError, Exception):
            records = _direct_records(
                caplog, lambda s=stage, e=error_type: runtime._warn_normal_restore_failure(s, e(DETAIL))
            )
            assert len(records) == 1
            tokens = records[0].getMessage().split(" ")
            assert len(tokens) == 4 and tokens[:2] == [BARE, "reason=final"], tokens
            stage_key, _, stage_value = tokens[2].partition("=")
            class_key, _, class_value = tokens[3].partition("=")
            assert (stage_key, class_key) == ("restore_stage", "exception_class")
            assert stage_value == (stage if stage in LOG_STAGES else "run_prepare")
            expected = error_type.__name__ if error_type in REVIEWED else "Exception"
            assert class_value == expected and TOKEN.fullmatch(class_value) and TOKEN.fullmatch(stage_value)
            assert _extras(records[0]) == {"restore_stage": stage_value, "exception_class": class_value}
            assert DETAIL not in caplog.text and "UnreviewedRestorationError" not in caplog.text
            seen.add((stage_value, class_value))
    assert len(seen) == len(LOG_STAGES) * (len(REVIEWED) + 1)


def test_transient_restore_retry_line_keeps_its_exact_text_format_and_arguments(caplog):
    """Control: the retry form is unchanged, and its reason token differs from the final line's."""
    records = _direct_records(
        caplog, lambda: runtime._warn_normal_restore_retry("checkpoint_load", TimeoutError(DETAIL), 2)
    )
    assert len(records) == 1
    record = records[0]
    assert record.msg == RETRY_FORMAT and record.args == (2, 3, "checkpoint_load", "TimeoutError")
    assert record.getMessage() == (
        "normal_restore_unavailable reason=transient_retry attempt=2 attempts=3"
        " restore_stage=checkpoint_load exception_class=TimeoutError"
    )
    assert _extras(record) == {}
    assert record.levelno == logging.WARNING and record.exc_info is None and record.stack_info is None
    assert "reason=final" not in record.getMessage().split(" ") and DETAIL not in caplog.text
