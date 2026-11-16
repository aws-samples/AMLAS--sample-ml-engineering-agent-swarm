"""Disconnected owned worker failure diagnostics; neutral AWS-only fixtures."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest
from test_runtime_busy_ownership import entered
from test_runtime_busy_ownership import owned_app as _owned_app
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas import api
from amlas import app as runtime
from amlas.tui.chat.models import ChatResponse

owned_app = _owned_app

PRIVATE_DETAIL = "ordinary withheld worker fixture detail"


class NeutralFixtureError(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("phase", "error_type", "category"),
    [
        ("work_body", RuntimeError, "RuntimeError"),
        ("work_body", NeutralFixtureError, "Exception"),
        ("work_body", asyncio.CancelledError, "CancelledError"),
        ("cleanup_report", LookupError, "LookupError"),
        ("cleanup_report", NeutralFixtureError, "Exception"),
    ],
)
def test_disconnected_worker_failure_has_closed_diagnostic_and_finishes_owned_task(
    monkeypatch, caplog, owned_app, phase, error_type, category
):
    app, completed = owned_app
    started, release, cleanup = threading.Event(), threading.Event(), threading.Event()
    retained = []
    active_at_log = []
    actual_complete = app.complete_async_task

    def complete(task_id):
        try:
            return actual_complete(task_id)
        finally:
            cleanup.set()

    def retain(*args, **kwargs):
        retained.append(True)
        if phase == "cleanup_report":
            raise error_type(PRIVATE_DETAIL)

    class Capture(logging.Handler):
        def emit(self, record):
            if (
                record.msg
                == "normal_worker_failed run_id=%s phase=%s error_class=%s source_label=%s source_line=%s"
            ):
                active_at_log.append(app.get_async_task_info()["active_count"])

    handler = Capture()
    logger = logging.getLogger(runtime.__name__)
    logger.addHandler(handler)
    monkeypatch.setattr(app, "complete_async_task", complete)
    monkeypatch.setattr(runtime, "_retain_prompt_report", retain)
    caplog.set_level(logging.WARNING, logger=runtime.__name__)

    async def run_prompt(_):
        started.set()
        assert release.wait(4), "test consumer must close before failure"
        if phase == "work_body":
            raise error_type(PRIVATE_DETAIL)
        return ChatResponse(message="ordinary fixture result")

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        stream = runtime.agent_invocation({"prompt": "ordinary diagnostic request"})
        first = await anext(stream)
        identifier = first["run_id"]
        try:
            await entered(started)
            assert app.get_async_task_info()["active_count"] == 1
            await stream.aclose()
            assert app.get_async_task_info()["active_count"] == 1
        finally:
            release.set()
            await stream.aclose()
        assert await asyncio.to_thread(cleanup.wait, 3), "owned task cleanup did not run"
        deadline = asyncio.get_running_loop().time() + 3
        while identifier in runtime._ACTIVE_PROMPT_REPORTS:
            assert asyncio.get_running_loop().time() < deadline, "worker finally did not finish"
            await asyncio.sleep(0.005)
        return identifier

    try:
        identifier = asyncio.run(scenario())
    finally:
        logger.removeHandler(handler)
    records = [
        r
        for r in caplog.records
        if r.msg == "normal_worker_failed run_id=%s phase=%s error_class=%s source_label=%s source_line=%s"
    ]
    assert len(records) == 1, "disconnected worker failure has no independent closed diagnostic"
    record = records[0]
    assert record.run_id == identifier
    assert record.worker_phase == phase and record.exception_class == category
    assert record.getMessage() == (
        f"normal_worker_failed run_id={identifier} phase={phase} error_class={category} "
        f"source_label={record.source_label} source_line={record.source_line}"
    )
    assert record.exc_info is None and record.stack_info is None
    assert PRIVATE_DETAIL not in caplog.text and "NeutralFixtureError" not in caplog.text
    assert len(completed) == 1 and app.get_async_task_info()["active_count"] == 0
    assert active_at_log == ([1] if phase == "work_body" else [0])
    assert retained == ([] if phase == "work_body" else [True])


def test_worker_diagnostic_logger_failure_cannot_skip_cleanup(monkeypatch, owned_app):
    app, completed = owned_app
    starts, release = threading.Event(), threading.Event()
    diagnostics = []
    retained = []
    logger = logging.getLogger(runtime.__name__)

    def refuse(*args, **kwargs):
        diagnostics.append((args, kwargs))
        raise RuntimeError("ordinary logger fixture unavailable")

    monkeypatch.setattr(logger, "warning", refuse)
    monkeypatch.setattr(runtime, "_retain_prompt_report", lambda *args, **kwargs: retained.append(True))

    async def run_prompt(_):
        starts.set()
        assert release.wait(4)
        raise RuntimeError(PRIVATE_DETAIL)

    monkeypatch.setattr(api, "run_prompt", run_prompt)

    async def scenario():
        stream = runtime.agent_invocation({"prompt": "ordinary logger request"})
        first = await anext(stream)
        try:
            await entered(starts)
            await stream.aclose()
        finally:
            release.set()
            await stream.aclose()
        deadline = asyncio.get_running_loop().time() + 3
        while first["run_id"] in runtime._ACTIVE_PROMPT_REPORTS:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.005)

    asyncio.run(scenario())
    assert len(diagnostics) == 1, "worker never attempted independent failure diagnostic"
    assert PRIVATE_DETAIL not in str(diagnostics)
    assert len(completed) == 1 and app.get_async_task_info()["active_count"] == 0
    assert retained == []


@pytest.mark.parametrize("control_type", [SystemExit, KeyboardInterrupt, asyncio.CancelledError])
def test_diagnostic_preserves_logger_process_control(monkeypatch, control_type):
    control = control_type("ordinary process control fixture")

    def stop(*args, **kwargs):
        raise control

    monkeypatch.setattr(logging.getLogger(runtime.__name__), "warning", stop)
    with pytest.raises(control_type) as observed:
        runtime._warn_prompt_worker_failure("a" * 32, "work_body", RuntimeError(PRIVATE_DETAIL))
    assert observed.value is control


class TestWorkerSourceLocation:
    """Exercise the actual warning body without loading runtime dependencies."""

    @staticmethod
    def _warning():
        import ast
        from pathlib import Path

        source = Path(__file__).parents[1] / "src/amlas/app.py"
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_warn_prompt_worker_failure"
        )
        namespace = {"logging": logging, "__name__": "amlas.app"}
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, "source-owned-warning-body", "exec"), namespace)
        return namespace["_warn_prompt_worker_failure"]

    @staticmethod
    def _error(filename, deeper=None):
        namespace = {}
        if deeper is None:
            text = "def fail():\n    raise ValueError('withheld ordinary report detail')\n"
        else:
            namespace["deeper"] = deeper
            text = "def fail():\n    deeper()\n"
        exec(compile(text, filename, "exec"), namespace)
        try:
            namespace["fail"]()
        except ValueError as error:
            return error
        raise AssertionError("fixture must raise")

    def _record(self, error):
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("amlas.app")
        handler = Capture()
        previous = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            self._warning()("a" * 32, "work_body", error)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)
        assert len(records) == 1
        record = records[0]
        text = record.getMessage()
        assert "withheld ordinary report detail" not in text
        assert "/private/" not in text
        assert "Traceback" not in text
        assert record.exc_info is None
        return record

    def test_deepest_mapped_frame_survives_unknown_private_frame(self):
        inner = self._error("/private/withheld/customer_frame.py")

        def deeper():
            raise inner

        error = self._error("/work/src/amlas/api.py", deeper)
        record = self._record(error)
        assert record.source_label == "api"
        assert record.source_line == 2
        assert "source_label=api source_line=2" in record.getMessage()

    def test_native_frame_has_fixed_label(self):
        record = self._record(self._error("/guarded/native_factory_entry.py"))
        assert record.source_label == "native_factory"
        assert record.source_line == 2
        assert "native_factory_entry.py" not in record.getMessage()

    def test_unknown_frame_is_closed(self):
        record = self._record(self._error("/private/withheld/customer_frame.py"))
        assert record.source_label == "unknown"
        assert record.source_line == 0

    def test_absent_traceback_is_closed(self):
        record = self._record(ValueError("withheld ordinary report detail"))
        assert record.source_label == "unknown"
        assert record.source_line == 0

    def test_logger_failure_does_not_replace_original(self):
        from unittest.mock import patch

        error = self._error("/work/src/amlas/api.py")
        before = error.args
        with patch.object(
            logging.getLogger("amlas.app"), "warning", side_effect=RuntimeError("logger unavailable")
        ):
            self._warning()("a" * 32, "work_body", error)
        assert error.args == before
