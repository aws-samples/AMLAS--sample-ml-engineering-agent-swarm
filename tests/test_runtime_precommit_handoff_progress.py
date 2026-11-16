"""Actual SDK precommit custody plus optional closed handoff warnings; inert ports."""

from __future__ import annotations

import asyncio
import builtins
import copy
import logging
import sys
from types import SimpleNamespace

import pytest
from bedrock_agentcore.runtime import BedrockAgentCoreContext
from test_normal_handoff_progress_diagnostics import MESSAGE, records
from test_normal_model_progress_diagnostics import bind_normal_fixture
from test_normal_model_progress_diagnostics import current_context as _current_context
from test_normal_model_progress_diagnostics import normal_work as _normal_work
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_precommit_original_completion import (
    USES,
    ThreeOriginals,
    captured_originals,
    protocol_work,
    publish_original_data,
    results,
)
from test_runtime_warm_model_continuation import complete_fixture

from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
current_context = _current_context
normal_work = _normal_work


@pytest.mark.parametrize(
    ("kind", "status", "expected_kind", "expected_state"),
    [
        ("model", None, "MODEL", "UNKNOWN"),
        ("data", "RUNNING", "DATA", "RUNNING"),
        ("data", "running", "DATA", "RUNNING"),
        ("data", "submitted", "DATA", "PENDING"),
        ("data", "completed", "DATA", "SUCCEEDED"),
        ("case", "running", "CASE", "UNKNOWN"),
        ("preparation", "completed", "PREP", "UNKNOWN"),
        ("case", "observation_failed", "CASE", "OBSERVATION_FAILED"),
        ("case", "output_verification_failed", "CASE", "VERIFICATION_FAILED"),
        ("preparation", "submission_rejected", "PREP", "SUBMISSION_REJECTED"),
        ("preparation", object(), "PREP", "UNKNOWN"),
    ],
)
def test_precommit_warning_reads_only_exact_pending_kind(
    current_context, caplog, kind, status, expected_kind, expected_state
):
    # Harmless missing fields would raise if diagnostics inspect graph/handles.
    pending = object.__new__(
        binding._PendingOriginalModel if kind == "model" else binding._PendingOriginalProducer
    )
    if kind != "model":
        object.__setattr__(pending, "kind", kind)
    helper = getattr(binding, "_warn_precommit_handoff", None)
    assert callable(helper), "missing optional precommit handoff warning adapter"
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    assert helper("OBSERVED", pending, status=status) is None
    assert [record.args for record in records(caplog)] == [
        (current_context.run_id, "OBSERVED", expected_kind, expected_state, False)
    ]


@pytest.mark.parametrize("mode", ["complete", "denied-sibling", "logger-failure", "import-failure"])
def test_actual_sdk_handoff_warnings_preserve_original_commit_or_hold(monkeypatch, tmp_path, caplog, mode):
    def deny(text):
        return mode == "denied-sibling" and "original_data_completed" in text and USES[2] in text

    model = ThreeOriginals()
    work, data, _fetched = protocol_work(monkeypatch, model, deny=deny)
    _ports, tokens = bind_normal_fixture(monkeypatch, work)
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    attempted, commits = [], []
    actual_warning = logging.getLogger(checkpoint.__name__).warning
    actual_import = builtins.__import__
    actual_commit = binding.ToolBindingBackend._commit_pending_pause

    def warning(message, *args, **kwargs):
        if message == MESSAGE:
            attempted.append(True)
            if mode == "logger-failure":
                raise RuntimeError("ordinary optional logger unavailable")
            if args[1] == "BATCH_COMMITTED":
                backend = work.engines[0].tool_backend
                witness = backend._pending_batch_pause
                assert not backend._warm_pending and not backend._warm_pending_producers
                assert not witness.completed and not witness.originals
                assert witness.history is not None if witness.marked else witness.batch is None
                assert len([value for value in results(work.agents[0]) if value["toolUseId"] in USES]) == 3
        return actual_warning(message, *args, **kwargs)

    def import_optional(name, globals=None, locals=None, fromlist=(), level=0):
        selected = name.endswith("normal_run_checkpoint") or "normal_run_checkpoint" in (fromlist or ())
        if (
            mode == "import-failure"
            and selected
            and sys._getframe(1).f_code.co_name == "_warn_precommit_handoff"
        ):
            attempted.append(True)
            raise RuntimeError("ordinary optional handoff import unavailable")
        return actual_import(name, globals, locals, fromlist, level)

    def commit(backend, event):
        before = len(records(caplog))
        result = actual_commit(backend, event)
        if any(record.args[1] == "BATCH_COMMITTED" for record in records(caplog)[before:]):
            commits.append(True)
        return result

    monkeypatch.setattr(logging.getLogger(checkpoint.__name__), "warning", warning)
    monkeypatch.setattr(binding.ToolBindingBackend, "_commit_pending_pause", commit)
    events = []

    async def read():
        async for event in runtime.agent_invocation({"prompt": work.literal}):
            events.append(event)

    async def scenario():
        reader = asyncio.create_task(read())
        try:
            backend = await captured_originals(work, data, reader)
            identifier = next(event["run_id"] for event in events if event["event"] == "run_started")
            original = backend._warm_pending
            producers = tuple(backend._warm_pending_producers.values())
            assert original is not None and len(producers) == 2 and not reader.done()
            before = copy.deepcopy(work.agents[0].messages)
            assert not [value for value in results(work.agents[0]) if value["toolUseId"] in USES]
            waiting = list(records(caplog))
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=original.record.handle, adapter=work.adapter, native=work.native),
            )
            for pending in producers:
                publish_original_data(pending.record, data)
            await asyncio.wait_for(reader, 5)
            assert events[-1]["worker_finished"] is True
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert len(work.native.requests) == 1 and len(data.native.calls) == 2
            assert len(work.agents) == len(work.engines) == 1
            assert work.agents[0].messages[: len(before)] == before
            found = records(caplog)
            if mode == "denied-sibling":
                assert backend._input_hold is True and not backend._warm_consumed and model.calls == 4
                assert backend._warm_pending is original
                assert tuple(backend._warm_pending_producers.values()) == producers
                assert events[-1]["response"]["data"]["input_admission"]["status"] == "held"
                assert any(
                    record["outcome"] == "BLOCKED"
                    for record in events[-1]["response"]["data"]["input_admission"]["records"]
                )
                assert any(record.args[1] == "ADMISSION_REFUSED" for record in found), (
                    "original admission hold lacks its closed phase warning"
                )
                assert not commits and not any(record.args[1] == "BATCH_COMMITTED" for record in found)
            else:
                assert not backend._input_hold and backend._warm_consumed and model.calls == 5
                final = [value for value in results(work.agents[0]) if value["toolUseId"] in USES]
                assert len(final) == 3 and all(value["status"] == "success" for value in final)
                if mode in {"logger-failure", "import-failure"}:
                    assert attempted, "precommit never attempted its optional handoff diagnostic"
                    assert not found
                else:
                    for kind in ("MODEL", "DATA"):
                        assert any(
                            record.args[1:4] == ("OBSERVED", kind, "RUNNING") and record.args[4] is False
                            for record in waiting
                        ), "genuine pending original lacks its observed wait marker"
                        phases = [record.args[1] for record in found if record.args[2] == kind]
                        for phase in ("OBSERVING", "OBSERVED", "READY", "ADMITTING", "ADMITTED"):
                            assert phase in phases, (
                                "qualified original completion lacks a closed handoff phase"
                            )
                        assert phases.index("READY") < phases.index("ADMITTING") < phases.index("ADMITTED")
                    assert commits == [True], (
                        "atomic original batch settlement has no single postcommit warning"
                    )
                    assert found[-1].args == (identifier, "BATCH_COMMITTED", "BATCH", "UNKNOWN", False)
            assert all(
                record.args[0] == identifier and record.exc_info is None and record.stack_info is None
                for record in found
            )
            assert all((record.args[1] == "READY") is record.args[4] for record in found)
        finally:
            await close_original_worker(work, reader)

    try:
        with monkeypatch.context() as importing:
            importing.setattr(builtins, "__import__", import_optional)
            asyncio.run(scenario())
    finally:
        BedrockAgentCoreContext._request_id.reset(tokens[0])
        BedrockAgentCoreContext._session_id.reset(tokens[1])
