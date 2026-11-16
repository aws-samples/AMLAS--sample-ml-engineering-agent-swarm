"""Tiny normal report retain/read across fresh CWD; no producer or model replay."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid

from amlas import app as runtime
from amlas.control_plane.normal_run_checkpoint import prepare_normal_run

pytest_plugins = ("test_normal_run_checkpoint",)
RUN = "6" * 32


def report_parts(run_id=RUN):
    envelope = {
        "mode": "prompt",
        "run_id": run_id,
        "response_kind": "conversation",
        "state_scope": "single_invocation",
        "durable_resume": False,
        "approval_policy_source": "default_noninteractive",
        "lifecycle_verified": False,
        "quality_success": False,
    }
    response = {
        "message": "Declared retained conversation report.",
        "action": "tool_agent_turn",
        "data": {"tools_used": [], "pending_confirmation": [], "tool_summary": ""},
        "suggestions": [],
    }
    return envelope, response


def expected_report():
    envelope, response = report_parts()
    return {
        "event": "run_completed",
        "status": "response_received",
        "response": response,
        "worker_finished": True,
        **envelope,
    }


def expected_read():
    return {
        "event": "prompt_report",
        "status": "available",
        "run_id": RUN,
        "found": True,
        "report": expected_report(),
    }


def enrolled(normal_work):
    work, ports, model = normal_work
    transport, session_hash = runtime._prompt_session_identity()
    context = prepare_normal_run(RUN, session_hash, transport_session_id=transport)
    assert context is not None
    context.enroll(str(uuid.UUID("4" * 32)), "Declared normal report persistence.", [])
    assert model.calls == 0 and not work.engines
    return work, ports, model, session_hash


def retain(session_hash):
    envelope, response = report_parts()
    runtime._retain_prompt_report(RUN, envelope, response, session_hash)


def passive_read():
    return runtime._read_prompt_report({"action": "get_prompt_report", "run_id": RUN})


def unavailable():
    return {"event": "prompt_report", "status": "unavailable", "run_id": RUN, "found": False}


def rewrite_record(ports, transform):
    item = ports.items["PROMPT_RUN#" + RUN]
    record = json.loads(item["record"]["S"])
    transform(record)
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    item["record"] = {"S": raw}
    item["record_sha256"] = {"S": hashlib.sha256(raw.encode()).hexdigest()}


def mutation_calls(ports):
    return [copy.deepcopy(call) for call in ports.calls if call[0] in {"PutItem", "UpdateItem", "PutObject"}]


def test_normal_report_survives_fresh_directory_using_only_native_transport(
    normal_work, monkeypatch, tmp_path
):
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    assert passive_read() == expected_read()
    items, objects, mutations = (
        copy.deepcopy(ports.items),
        copy.deepcopy(ports.objects),
        mutation_calls(ports),
    )
    replacement = tmp_path / "declared-report-replacement"
    replacement.mkdir()
    monkeypatch.chdir(replacement)
    assert passive_read() == expected_read(), "native final report was lost with the original CWD"
    record = json.loads(ports.items["PROMPT_RUN#" + RUN]["record"]["S"])
    assert type(record.get("report_checkpoint_ref")) is dict and ports.objects
    assert ports.items == items and ports.objects == objects and mutation_calls(ports) == mutations
    assert model.calls == 0 and not work.engines


def test_missing_native_report_descriptor_never_revives_local_copy(normal_work):
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    assert passive_read() == expected_read()
    rewrite_record(ports, lambda record: record.pop("report_checkpoint_ref", None))
    items, mutations = copy.deepcopy(ports.items), mutation_calls(ports)
    assert passive_read() == unavailable(), "selected native report absence fell back to a stale local copy"
    assert ports.items == items and mutation_calls(ports) == mutations
    assert model.calls == 0 and not work.engines


def test_changed_native_report_descriptor_holds_without_effect(normal_work, monkeypatch, tmp_path):
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    record = json.loads(ports.items["PROMPT_RUN#" + RUN]["record"]["S"])
    assert type(record.get("report_checkpoint_ref")) is dict, "missing native final report descriptor"
    rewrite_record(ports, lambda row: row["report_checkpoint_ref"].update(sha256="f" * 64))
    replacement = tmp_path / "declared-descriptor-replacement"
    replacement.mkdir()
    monkeypatch.chdir(replacement)
    items, mutations = copy.deepcopy(ports.items), mutation_calls(ports)
    assert passive_read() == unavailable()
    assert ports.items == items and mutation_calls(ports) == mutations
    assert model.calls == 0 and not work.engines


def test_changed_native_report_bytes_hold_without_effect(normal_work, monkeypatch, tmp_path):
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    record = json.loads(ports.items["PROMPT_RUN#" + RUN]["record"]["S"])
    descriptor = record.get("report_checkpoint_ref")
    assert type(descriptor) is dict, "missing native final report descriptor"
    key = descriptor["uri"].split("/", 3)[3]
    original = ports.objects[(key, descriptor["version_id"])]
    ports.objects[(key, descriptor["version_id"])] = original[:-1] + (
        b" " if original[-1:] != b" " else b"\n"
    )
    replacement = tmp_path / "declared-bytes-replacement"
    replacement.mkdir()
    monkeypatch.chdir(replacement)
    items, objects, mutations = (
        copy.deepcopy(ports.items),
        copy.deepcopy(ports.objects),
        mutation_calls(ports),
    )
    assert passive_read() == unavailable()
    assert ports.items == items and ports.objects == objects and mutation_calls(ports) == mutations
    assert model.calls == 0 and not work.engines


def test_unconfigured_normal_runtime_preserves_local_report(monkeypatch, tmp_path):
    import boto3

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AMLAS_RUNTIME_EXEC_ROLE_ARN", raising=False)
    monkeypatch.delenv("AMLAS_RUNTIME_EXEC_ROLE_ID", raising=False)
    session_hash = "7" * 64
    monkeypatch.setattr(runtime, "_prompt_report_session_hash", lambda: session_hash)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy local report cannot call native storage")

    monkeypatch.setattr(boto3, "Session", forbidden)
    monkeypatch.setattr(boto3, "client", forbidden)
    retain(session_hash)
    assert passive_read() == expected_read()


def test_identical_native_report_retention_keeps_winning_version_without_mutation(normal_work):
    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    first = passive_read()
    assert first == expected_read()
    items, objects, mutations = (
        copy.deepcopy(ports.items),
        copy.deepcopy(ports.objects),
        mutation_calls(ports),
    )
    winning = copy.deepcopy(
        json.loads(ports.items["PROMPT_RUN#" + RUN]["record"]["S"])["report_checkpoint_ref"]
    )
    retain(session_hash)
    assert passive_read() == first
    assert json.loads(ports.items["PROMPT_RUN#" + RUN]["record"]["S"])["report_checkpoint_ref"] == winning
    assert ports.items == items and ports.objects == objects and mutation_calls(ports) == mutations
    assert model.calls == 0 and not work.engines


def test_conflicting_native_report_retention_refuses_and_preserves_first(normal_work):
    import pytest

    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    first = passive_read()
    items, objects, mutations = (
        copy.deepcopy(ports.items),
        copy.deepcopy(ports.objects),
        mutation_calls(ports),
    )
    envelope, response = report_parts()
    response["message"] = "Declared conflicting terminal conversation."
    with pytest.raises(ValueError):
        runtime._retain_prompt_report(RUN, envelope, response, session_hash)
    assert passive_read() == first == expected_read()
    assert ports.items == items and ports.objects == objects and mutation_calls(ports) == mutations
    assert model.calls == 0 and not work.engines


def test_selected_passive_native_report_never_calls_resume_or_claims(normal_work, monkeypatch, tmp_path):
    from amlas import api

    work, ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    replacement = tmp_path / "declared-passive-only-reader"
    replacement.mkdir()
    monkeypatch.chdir(replacement)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("resume")
        raise AssertionError("selected native report cannot use API resume or local fallback")

    monkeypatch.setattr(api, "resume", forbidden)
    items, objects, mutations = (
        copy.deepcopy(ports.items),
        copy.deepcopy(ports.objects),
        mutation_calls(ports),
    )
    assert passive_read() == expected_read()
    assert calls == [] and ports.items == items and ports.objects == objects
    assert mutation_calls(ports) == mutations and model.calls == 0 and not work.engines
    rewrite_record(ports, lambda record: record.pop("report_checkpoint_ref"))
    items, mutations = copy.deepcopy(ports.items), mutation_calls(ports)
    assert passive_read() == unavailable()
    assert calls == [] and ports.items == items and mutation_calls(ports) == mutations
    assert model.calls == 0 and not work.engines


def test_unconfigured_native_report_helpers_never_call_normal_resume_factory(monkeypatch, tmp_path):
    import boto3

    from amlas.control_plane import normal_run_checkpoint as checkpoint

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AMLAS_RUNTIME_EXEC_ROLE_ARN", raising=False)
    monkeypatch.delenv("AMLAS_RUNTIME_EXEC_ROLE_ID", raising=False)
    session_hash = "7" * 64
    monkeypatch.setattr(runtime, "_prompt_report_session_hash", lambda: session_hash)
    calls = []

    def selected_factory(*args, **kwargs):
        calls.append("selected_factory")
        raise AssertionError("unconfigured local report cannot enter normal resume factory")

    def forbidden_provider(*args, **kwargs):
        calls.append("provider")
        raise AssertionError("unconfigured local report cannot call a provider")

    monkeypatch.setattr(checkpoint, "prepare_normal_run", selected_factory)
    monkeypatch.setattr(boto3, "Session", forbidden_provider)
    monkeypatch.setattr(boto3, "client", forbidden_provider)
    retain(session_hash)
    assert passive_read() == expected_read()
    assert calls == []


def test_a_retained_report_pointer_is_never_a_completion_proof_for_resume(normal_work):
    """The original worker also retains its pause-turn response, so a pointer must not read as completed."""
    from test_normal_run_checkpoint import collect

    work, _ports, model, session_hash = enrolled(normal_work)
    retain(session_hash)
    assert passive_read() == expected_read()
    runtime._RESUME_WORKERS.clear()
    events = collect({"action": "resume_prompt_run", "run_id": RUN})
    assert events[0] == {"event": "resume_state", "state": "restoring", "run_id": RUN}
    assert all(event.get("state") != "completed" for event in events)
    assert model.calls == 0 and not work.engines
