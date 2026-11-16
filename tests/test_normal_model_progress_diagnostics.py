"""Closed MODEL loading warnings; real SDK over existing neutral inert ports."""

from __future__ import annotations

import asyncio
import builtins
import copy
import logging
import re
import sys
from types import SimpleNamespace

import boto3
import model_acquisition_fixtures as model_fixture
import pytest
from bedrock_agentcore.runtime import BedrockAgentCoreContext
from test_normal_run_checkpoint import ACCOUNT, REGION, ROLE_ID, RUNTIME_ROLE, SESSION, TABLE, NativePorts
from test_normal_run_checkpoint import normal_work as _normal_work
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_precommit_original_completion import (
    ThreeOriginals,
    captured_originals,
    protocol_work,
    publish_original_data,
)
from test_runtime_warm_model_continuation import complete_fixture

from amlas import app as runtime
from amlas.backends.adapters import model_weight_acquisition as model_adapter
from amlas.control_plane import normal_run_checkpoint as checkpoint

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
normal_work = _normal_work
MESSAGE = "normal_model_progress run_id=%s phase=%s state=%s verified_payloads=%s"


def progress_records(caplog):
    return [record for record in caplog.records if record.msg == MESSAGE]


@pytest.fixture
def current_context(monkeypatch, normal_work):
    _work, ports, _model = normal_work
    context = checkpoint.NormalRunContext(checkpoint._catalog(), "a" * 32, "b" * 64)
    before = (dict(context.__dict__), copy.deepcopy(ports.items), list(ports.calls), dict(ports.objects))
    with checkpoint.activate_normal_run(context):
        yield context
    assert context.__dict__ == before[0], "progress warning mutated normal checkpoint DATA"
    assert ports.items == before[1] and ports.calls == before[2] and ports.objects == before[3], (
        "progress warning added native or history effects"
    )


@pytest.mark.parametrize(
    ("phase", "state", "count"),
    [
        ("model_native_observation", "UNKNOWN", 0),
        ("model_native_wait", "PENDING", 0),
        ("model_native_wait", "RUNNING", 0),
        ("model_artifact_verification", "SUCCEEDED", 3),
        ("model_artifact_ready", "SUCCEEDED", 9),
        ("model_artifact_refused", "FAILED", 2),
        ("model_artifact_refused", "STOPPED", 0),
    ],
)
def test_progress_warning_has_only_closed_rendered_fields(current_context, caplog, phase, state, count):
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    assert checkpoint._warn_normal_model_progress(phase, state, count) is None
    # A refused pass given no cause renders the closed defaults; every other phase renders no cause.
    cause = ("Exception", "none", "unknown:0", "none") if phase == "model_artifact_refused" else ()
    template = MESSAGE + (" exception_class=%s native_code=%s site=%s cause_class=%s" if cause else "")
    suffix = (
        f" exception_class={cause[0]} native_code={cause[1]} site={cause[2]} cause_class={cause[3]}"
        if cause
        else ""
    )
    records = [record for record in caplog.records if record.msg == template]
    assert len(records) == 1, "qualified normal context has no independent loading warning"
    record = records[0]
    assert record.args == (current_context.run_id, phase, state, count, *cause)
    assert record.getMessage() == (
        f"normal_model_progress run_id={current_context.run_id} phase={phase} "
        f"state={state} verified_payloads={count}{suffix}"
    )
    assert record.levelno == logging.WARNING and record.exc_info is None and record.stack_info is None


def _rendered(caplog):
    return [
        record
        for record in caplog.records
        if record.name == checkpoint.__name__ and record.getMessage().startswith("normal_model_progress ")
    ]


@pytest.mark.parametrize(
    ("state", "count", "cause", "rendered"),
    [
        (
            "SUCCEEDED",
            2,
            ("ClientError", "SlowDown", "model_weight_acquisition.py:279", "none"),
            "exception_class=ClientError native_code=SlowDown site=model_weight_acquisition.py:279 cause_class=none",
        ),
        (
            "FAILED",
            2,
            ("ValueError", "none", "normal_run_checkpoint.py:1460", "ConnectTimeoutError"),
            "exception_class=ValueError native_code=none site=normal_run_checkpoint.py:1460"
            " cause_class=ConnectTimeoutError",
        ),
        ("STOPPED", 0, (), "exception_class=Exception native_code=none site=unknown:0 cause_class=none"),
        (
            "SUCCEEDED",
            5,
            ("ClientError", "a message naming an identifier", "model_weight_acquisition.py:279", "none"),
            "exception_class=Exception native_code=none site=unknown:0 cause_class=none",
        ),
        (
            "SUCCEEDED",
            5,
            ("UnreviewedError", "none", "/abs/path/model_weight_acquisition.py:279", "none"),
            "exception_class=Exception native_code=none site=unknown:0 cause_class=none",
        ),
    ],
)
def test_refused_progress_names_a_closed_cause_never_a_message(
    current_context, caplog, state, count, cause, rendered
):
    """A refused pass always renders, with its closed cause or the closed defaults; never free text."""
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    assert checkpoint._warn_normal_model_progress("model_artifact_refused", state, count, cause) is None
    records = _rendered(caplog)
    assert len(records) == 1
    assert records[0].getMessage() == (
        f"normal_model_progress run_id={current_context.run_id} phase=model_artifact_refused "
        f"state={state} verified_payloads={count} {rendered}"
    )
    assert records[0].exc_info is None and records[0].stack_info is None
    assert "identifier" not in caplog.text and "/abs/path" not in caplog.text


@pytest.mark.parametrize(
    ("cause", "rendered"),
    [
        (
            ("run_record_changed", "normal_run_checkpoint.py:471"),
            "reason=run_record_changed site=normal_run_checkpoint.py:471",
        ),
        (
            ("sft_catalog_changed", "normal_run_checkpoint.py:473"),
            "reason=sft_catalog_changed site=normal_run_checkpoint.py:473",
        ),
        (("run_record_changed", "not a site"), "reason=run_record_changed site=unknown:0"),
    ],
)
def test_interrupted_progress_names_its_closed_reason_and_site(current_context, caplog, cause, rendered):
    """The caller's own record recheck ended the pass: a typed line that is not a refusal."""
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    assert (
        checkpoint._warn_normal_model_progress("model_verification_interrupted", "SUCCEEDED", 3, cause)
        is None
    )
    records = _rendered(caplog)
    assert len(records) == 1
    assert records[0].getMessage() == (
        f"normal_model_progress run_id={current_context.run_id} phase=model_verification_interrupted "
        f"state=SUCCEEDED verified_payloads=3 {rendered}"
    )
    assert "refused" not in records[0].getMessage()


@pytest.mark.parametrize(
    ("phase", "cause"),
    [
        ("model_verification_interrupted", ()),
        ("model_verification_interrupted", ("refused_bytes", "normal_run_checkpoint.py:471")),
        ("model_verification_interrupted", ("run_record_changed",)),
        ("model_native_wait", ("run_record_changed", "normal_run_checkpoint.py:471")),
        ("model_artifact_ready", ("ClientError", "SlowDown", "model_weight_acquisition.py:279", "none")),
    ],
)
def test_progress_warning_refuses_an_unclosed_cause(current_context, caplog, phase, cause):
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    assert checkpoint._warn_normal_model_progress(phase, "RUNNING", 0, cause) is None
    assert not _rendered(caplog), "a cause outside its closed phase escaped its finite schema"


@pytest.mark.parametrize(
    ("phase", "state", "count"),
    [
        ("other_phase", "RUNNING", 0),
        ("model_native_wait", "other_state", 0),
        ("model_native_wait", "NONE", 0),
        ("model_native_wait", "RUNNING", True),
        ("model_native_wait", "RUNNING", -1),
        ("model_native_wait", "RUNNING", 10),
        ("model_native_wait", "RUNNING", 1.0),
    ],
)
def test_progress_warning_refuses_malformed_closed_fields(current_context, caplog, phase, state, count):
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    assert checkpoint._warn_normal_model_progress(phase, state, count) is None
    assert not progress_records(caplog), "malformed diagnostic labels escaped their finite schema"


@pytest.mark.parametrize("mode", ["absent", "changed-scope", "foreign-pid"])
def test_progress_warning_requires_current_local_context(monkeypatch, current_context, caplog, mode):
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    if mode == "absent":
        with checkpoint.activate_normal_run(None):
            checkpoint._warn_normal_model_progress("model_native_wait", "RUNNING", 0)
    else:
        if mode == "changed-scope":
            monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", "changed-neutral-scope")
        else:
            original = current_context.process_id
            object.__setattr__(current_context, "process_id", original + 1)
        try:
            checkpoint._warn_normal_model_progress("model_native_wait", "RUNNING", 0)
        finally:
            if mode == "foreign-pid":
                object.__setattr__(current_context, "process_id", original)
    assert not progress_records(caplog), "absent or stale context published loading progress"


def test_progress_logger_failure_is_optional(monkeypatch, current_context):
    attempted = []

    def refuse(*args, **kwargs):
        attempted.append(True)
        raise RuntimeError("ordinary neutral logger unavailable")

    monkeypatch.setattr(logging.getLogger(checkpoint.__name__), "warning", refuse)
    assert checkpoint._warn_normal_model_progress("model_native_wait", "RUNNING", 0) is None
    assert attempted == [True]


@pytest.mark.parametrize("control_type", [SystemExit, KeyboardInterrupt, asyncio.CancelledError])
def test_progress_logger_process_control_propagates(monkeypatch, current_context, control_type):
    control = control_type("ordinary neutral process control")

    def stop(*args, **kwargs):
        raise control

    monkeypatch.setattr(logging.getLogger(checkpoint.__name__), "warning", stop)
    with pytest.raises(control_type) as observed:
        checkpoint._warn_normal_model_progress("model_native_wait", "RUNNING", 0)
    assert observed.value is control


def bind_normal_fixture(monkeypatch, work):
    # Same strict native enrollment setup as normal_work, bound to THIS SDK work.
    # Calling profile_work again would install a different model/capture/guard.
    ports = NativePorts()
    for key, value in {
        "AMLAS_AWS_ACCOUNT_ID": ACCOUNT,
        "AMLAS_AWS_REGION": REGION,
        "AMLAS_JOBS_TABLE": TABLE,
        "AMLAS_SESSION_BUCKET": model_fixture.BUCKET,
        "AMLAS_RUNTIME_EXEC_ROLE_ARN": RUNTIME_ROLE,
        "AMLAS_RUNTIME_EXEC_ROLE_ID": ROLE_ID,
    }.items():
        monkeypatch.setenv(key, value)

    def client(service, **kwargs):
        assert kwargs.get("region_name", REGION) == REGION
        if service == "bedrock-runtime":
            return work.guard
        assert service in {"sts", "iam", "dynamodb", "s3"}, "unexpected live SDK boundary"
        return ports

    class NativeSession:
        def __init__(self, *args, **kwargs):
            pass

        def client(self, service, **kwargs):
            return client(service, **kwargs)

    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setattr(boto3, "Session", NativeSession)
    monkeypatch.setattr(boto3.session, "Session", NativeSession)
    request_token = BedrockAgentCoreContext._request_id.set(None)
    session_token = BedrockAgentCoreContext._session_id.set(None)
    BedrockAgentCoreContext.set_request_context("ordinary-progress-request", SESSION)
    return ports, (request_token, session_token)


@pytest.mark.parametrize("fault_type", [RuntimeError, SystemExit, KeyboardInterrupt, asyncio.CancelledError])
def test_optional_progress_import_preserves_original_status_and_effects(
    monkeypatch, current_context, fault_type
):
    native, objects = model_fixture.Processing(), model_fixture.S3()
    adapter = model_fixture.configured_backend(model_adapter, native, objects)
    handle = adapter.submit(
        catalog_id=model_fixture.CATALOG,
        catalog_version=model_fixture.VERSION,
        runtime_session_id=model_fixture.SESSION,
    )
    before = adapter.artifacts(handle)
    assert before["status"] == "RUNNING" and len(native.requests) == 1
    calls, attempted = [], []
    actual_describe, actual_import = native.describe_processing_job, builtins.__import__

    def describe(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return actual_describe(**kwargs)

    fault = fault_type("ordinary optional progress import unavailable")

    def import_optional(name, globals=None, locals=None, fromlist=(), level=0):
        selected = name.endswith("normal_run_checkpoint") or "normal_run_checkpoint" in (fromlist or ())
        if selected and sys._getframe(1).f_code.co_name == "_model_progress":
            attempted.append(True)
            raise fault
        return actual_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(native, "describe_processing_job", describe)
    outcome, caught = None, None
    with monkeypatch.context() as importing:
        importing.setattr(builtins, "__import__", import_optional)
        try:
            outcome = adapter.artifacts(handle)
        except BaseException as error:
            caught = error
    # Restore the scoped import fault before pytest assertions or reporting.
    if fault_type is RuntimeError:
        assert caught is None and outcome == before
        assert len(calls) == 1, "optional diagnostic changed the original native observation count"
    else:
        assert caught is fault, "optional diagnostic swallowed or skipped process control"
        assert calls == []
    assert attempted, "original observation never attempted the optional loading diagnostic import"
    assert len(native.requests) == 1


def test_actual_sdk_original_model_wait_then_nine_payloads_logs_closed_progress(
    monkeypatch, tmp_path, caplog
):
    model = ThreeOriginals()
    work, data, _fetched = protocol_work(monkeypatch, model)
    _ports, tokens = bind_normal_fixture(monkeypatch, work)
    caplog.set_level(logging.WARNING, logger=checkpoint.__name__)
    events = []

    async def read():
        async for event in runtime.agent_invocation({"prompt": work.literal}):
            events.append(event)

    async def scenario():
        reader = asyncio.create_task(read())
        try:
            backend = await captured_originals(work, data, reader)
            identifier = next(event["run_id"] for event in events if event["event"] == "run_started")
            assert re.fullmatch(r"[0-9a-f]{32}", identifier)
            original = backend._warm_pending
            assert original is not None and not reader.done() and len(work.native.requests) == 1
            wait = [record for record in progress_records(caplog) if record.args[1] == "model_native_wait"]
            assert wait, "genuine pending MODEL has no closed loading warning"
            assert all(record.args == (identifier, "model_native_wait", "RUNNING", 0) for record in wait)
            before = copy.deepcopy(work.agents[0].messages)
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=original.record.handle, adapter=work.adapter, native=work.native),
            )
            for pending in backend._warm_pending_producers.values():
                publish_original_data(pending.record, data)
            await asyncio.wait_for(reader, 5)
            records = progress_records(caplog)
            verification = [
                record.args[3] for record in records if record.args[1] == "model_artifact_verification"
            ]
            assert any(
                verification[index : index + 10] == list(range(10)) for index in range(len(verification))
            ), "MODEL exact nine-payload verification lacks source-owned 0..9 progress"
            ready = [record for record in records if record.args[1] == "model_artifact_ready"]
            assert ready and all(
                record.args == (identifier, "model_artifact_ready", "SUCCEEDED", 9) for record in ready
            )
            assert all(
                record.args[0] == identifier and record.exc_info is None and record.stack_info is None
                for record in records
            )
            assert len(work.native.requests) == 1 and len(data.native.calls) == 2
            assert work.agents[0].messages[: len(before)] == before
            assert events[-1]["worker_finished"] is True and backend._warm_consumed
        finally:
            await close_original_worker(work, reader)

    try:
        asyncio.run(scenario())
    finally:
        BedrockAgentCoreContext._request_id.reset(tokens[0])
        BedrockAgentCoreContext._session_id.reset(tokens[1])
