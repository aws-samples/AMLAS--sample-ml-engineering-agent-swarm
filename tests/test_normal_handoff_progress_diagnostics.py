"""Closed precommit warning scalars; source context and inert native ports only."""

from __future__ import annotations

import asyncio
import logging

import pytest
from test_normal_model_progress_diagnostics import current_context as _current_context
from test_normal_model_progress_diagnostics import normal_work as _normal_work

from amlas.control_plane import normal_run_checkpoint as checkpoint

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
current_context = _current_context
normal_work = _normal_work
MESSAGE = "normal_handoff run_id=%s phase=%s kind=%s state=%s ready=%s"


def records(caplog):
    return [record for record in caplog.records if record.msg == MESSAGE]


def warn(*values):
    helper = getattr(checkpoint, "_warn_normal_handoff", None)
    assert callable(helper), "missing source-owned normal handoff diagnostic helper"
    return helper(*values)


@pytest.mark.parametrize(
    ("phase", "kind", "state", "ready"),
    [
        ("OBSERVING", "MODEL", "UNKNOWN", False),
        ("OBSERVED", "DATA", "PENDING", False),
        ("OBSERVED", "CASE", "RUNNING", False),
        ("READY", "PREP", "SUCCEEDED", True),
        ("ADMITTING", "MODEL", "FAILED", False),
        ("ADMITTED", "DATA", "STOPPED", False),
        ("ADMISSION_REFUSED", "CASE", "OBSERVATION_FAILED", False),
        ("OBSERVED", "PREP", "VERIFICATION_FAILED", False),
        ("BATCH_COMMITTED", "BATCH", "SUBMISSION_REJECTED", False),
    ],
)
def test_handoff_warning_has_only_closed_guarded_scalars(current_context, caplog, phase, kind, state, ready):
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    assert warn(phase, kind, state, ready) is None
    found = records(caplog)
    assert len(found) == 1, "qualified handoff has no independent closed warning"
    record = found[0]
    assert record.args == (current_context.run_id, phase, kind, state, ready)
    assert record.getMessage() == (
        f"normal_handoff run_id={current_context.run_id} phase={phase} "
        f"kind={kind} state={state} ready={ready}"
    )
    assert record.levelno == logging.WARNING and record.exc_info is None and record.stack_info is None


@pytest.mark.parametrize(
    ("phase", "kind", "state", "ready"),
    [
        ("other_phase", "MODEL", "RUNNING", False),
        ("OBSERVED", "other_kind", "RUNNING", False),
        ("OBSERVED", "MODEL", "other_state", False),
        (1, "MODEL", "RUNNING", False),
        ("OBSERVED", 1, "RUNNING", False),
        ("OBSERVED", "MODEL", 1, False),
        ("OBSERVED", "MODEL", "RUNNING", 1),
        ("OBSERVED", "MODEL", "RUNNING", None),
        ("OBSERVED", "MODEL", "RUNNING", True),
        ("READY", "MODEL", "SUCCEEDED", False),
    ],
)
def test_handoff_warning_refuses_wrong_scalar_types_or_ready_phase(
    current_context, caplog, phase, kind, state, ready
):
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    assert warn(phase, kind, state, ready) is None
    assert not records(caplog), "invalid handoff DATA escaped the closed warning schema"


@pytest.mark.parametrize("mode", ["absent", "changed-scope", "foreign-pid"])
def test_handoff_warning_requires_current_local_context(monkeypatch, current_context, caplog, mode):
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    if mode == "absent":
        with checkpoint.activate_normal_run(None):
            warn("OBSERVED", "MODEL", "RUNNING", False)
    else:
        original = current_context.process_id
        if mode == "changed-scope":
            monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", "ordinary-other-scope")
        else:
            object.__setattr__(current_context, "process_id", original + 1)
        try:
            warn("OBSERVED", "MODEL", "RUNNING", False)
        finally:
            if mode == "foreign-pid":
                object.__setattr__(current_context, "process_id", original)
    assert not records(caplog), "absent or stale normal context published a handoff warning"


def test_handoff_logger_failure_is_optional(monkeypatch, current_context):
    attempted = []

    def refuse(*args, **kwargs):
        attempted.append(True)
        raise RuntimeError("ordinary neutral logger unavailable")

    monkeypatch.setattr(logging.getLogger(checkpoint.__name__), "warning", refuse)
    assert warn("OBSERVED", "MODEL", "RUNNING", False) is None
    assert attempted == [True]


@pytest.mark.parametrize("control_type", [SystemExit, KeyboardInterrupt, asyncio.CancelledError])
def test_handoff_logger_process_control_propagates(monkeypatch, current_context, control_type):
    control = control_type("ordinary neutral process control")

    def stop(*args, **kwargs):
        raise control

    monkeypatch.setattr(logging.getLogger(checkpoint.__name__), "warning", stop)
    with pytest.raises(control_type) as observed:
        warn("OBSERVED", "MODEL", "RUNNING", False)
    assert observed.value is control
