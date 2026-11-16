"""Proposed cold original PRECOMMIT continuation through the ordinary SDK.

Every source handle comes from a registered tool/native request. Distinct-PID
transport replay is component evidence only: no AgentCore loss, native IAM,
competing lease, real model/oracle execution or full C1-C4/R1-R4 is established.
The third test proves completion-CAS refusal, not a complete lease algorithm.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from normal_sft_cold_fixture import (
    FINAL_TEXT,
    FIT_FINAL_TEXT,
    FIT_STATUS_USE,
    SFT_USE,
    close,
    cloud_data,
    finish_generation_pair,
    original_precommit,
    payloads,
    setup,
)
from test_normal_run_checkpoint import assert_safe_data, decoded
from test_normal_run_sft_checkpoint import wire_decode, wire_encode
from test_runtime_prompt_entry import inert_runtime_setup
from test_runtime_warm_model_continuation import tool_payload

from amlas import app as runtime
from amlas.tui.chat.tool_binding_backend import _record_messages

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
PATH = Path(__file__).resolve()
# Upper bound for one distinct-process continuation: the same budget for the
# subprocess and for its own wait. A slow machine needs the room; a child that
# never finishes still fails when it ends.
COLD_CHILD_WAIT_SECONDS = 900


def marked_results(model):
    """Status and decoded payload of the variant-marked tool results the model saw last."""
    if not model.messages_seen:
        return {}
    marked = {}
    for message in model.messages_seen[-1]:
        for block in message["content"]:
            result = block.get("toolResult")
            if result is not None and result["toolUseId"].startswith(
                ("repeat-", "mixed-", "cold-original-generation-")
            ):
                marked[result["toolUseId"]] = {"status": result["status"], "payload": tool_payload(result)}
    return marked


def stable_response(response):
    """All conversation fields remain stable except independently fresh receipts."""
    assert set(response) == {"message", "action", "data", "suggestions"}
    result = copy.deepcopy(response)
    evidence = result["data"]["input_admission"]
    assert type(evidence) is dict and "records" in evidence
    evidence.pop("records")
    return result


def fresh_admission_evidence(response, *, required_boundaries):
    """Validate actual Source receipt ABI, not cross-instance receipt identity."""
    from dataclasses import fields

    from amlas.tui.chat.input_admission import MAX_RECORDS, InputAdmissionReceipt

    evidence = response["data"]["input_admission"]
    assert set(evidence) == {
        "schema_version",
        "data_only",
        "status",
        "records",
        "native_admission",
        "quality_success",
        "blocked_component_attribution",
    }
    assert evidence["schema_version"] == 1 and evidence["data_only"] is True
    assert evidence["status"] == "admitted" and evidence["native_admission"] is False
    assert evidence["quality_success"] is False
    assert evidence["blocked_component_attribution"] == "UNESTABLISHED"
    records = evidence["records"]
    assert type(records) is list and 1 <= len(records) <= MAX_RECORDS
    checked = []
    for record in records:
        arguments = {
            field.name: record[field.name]
            for field in fields(InputAdmissionReceipt)
            if field.init and field.name not in {"transport_diagnostic", "blocking_categories"}
        }
        arguments["categories"] = tuple(arguments["categories"])
        arguments["blocking_categories"] = tuple(record.get("blocking_categories", ()))
        receipt = InputAdmissionReceipt(**arguments)
        assert receipt.to_dict() == record, "fresh receipt differs from closed Source ABI"
        assert receipt.outcome == "ALLOW" and receipt.reason == "NATIVE_NONE"
        assert receipt.categories == tuple(sorted(set(receipt.categories)))
        assert receipt.byte_measurement == "EXACT"
        assert receipt.keyed_digest is not None
        checked.append(receipt)
    assert required_boundaries.issubset({item.boundary for item in checked})
    checks = {item.check_id for item in checked}
    cycles = {item.opaque_cycle for item in checked}
    digests = {item.keyed_digest for item in checked}
    guards = {(item.guardrail_id, item.guardrail_version, item.region) for item in checked}
    assert len(checks) == len(checked) and len(cycles) == len(guards) == 1
    return checks, cycles, digests, guards


def _closed_baseline_receipt(directory, answer):
    """Retain fixed cold baseline labels; raw transport stays in the AWS test workspace."""
    if directory.name != "baseline-completed":
        return
    receipt = {
        "schema_version": 1,
        "phase": "baseline-completed",
        "loaded_count": len(answer["loaded"]),
        "invoked_count": len(answer["invoked"]),
        "baseline_predicate": bool(answer["loaded"] and answer["invoked"]),
        "new_training_count": len(answer["new_training"]),
        "new_processing_count": len(answer["new_processing"]),
        "loader_diagnostic": answer.get("loader_diagnostic", []),
        "restore_diagnostic": answer.get("restore_diagnostic", []),
    }
    (directory / "baseline-result.private.json").write_bytes(
        json.dumps(wire_encode(answer), sort_keys=True, separators=(",", ":")).encode()
    )
    (directory / "baseline.closed.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _capture_cold_restore_diagnostic(monkeypatch):
    import logging

    records = []
    stages = {
        "run_prepare",
        "checkpoint_promote",
        "checkpoint_load",
        "snapshot_load",
    }
    categories = {
        "ValueError",
        "TypeError",
        "RuntimeError",
        "AssertionError",
        "AttributeError",
        "KeyError",
        "LookupError",
        "OSError",
        "FileNotFoundError",
        "PermissionError",
        "TimeoutError",
        "NotImplementedError",
        "Exception",
    }

    frames = []
    source_root = Path(__file__).resolve().parents[1] / "src" / "amlas"
    original_warning = runtime._warn_normal_restore_failure

    def warning(stage, error):
        frames.clear()
        if stage == "snapshot_load" and type(error) is KeyError:
            trace = error.__traceback__
            while trace is not None:
                filename = Path(trace.tb_frame.f_code.co_filename)
                try:
                    source = filename.resolve().relative_to(source_root)
                except (ValueError, OSError):
                    pass
                else:
                    if source.suffix == ".py" and filename.is_file():
                        frames.append(
                            {
                                "source": "src/amlas/" + source.as_posix(),
                                "function": trace.tb_frame.f_code.co_name,
                                "line": trace.tb_lineno,
                            }
                        )
                trace = trace.tb_next
        return original_warning(stage, error)

    monkeypatch.setattr(runtime, "_warn_normal_restore_failure", warning)

    class RestoreDiagnosticHandler(logging.Handler):
        def emit(self, record) -> None:
            stage = getattr(record, "restore_stage", None)
            category = getattr(record, "exception_class", None)
            # The final restore refusal states the same fixed stage and reviewed category in its text.
            if record.name != "amlas.app" or record.getMessage() != (
                f"normal_restore_unavailable reason=final restore_stage={stage} exception_class={category}"
            ):
                return
            if type(stage) is str and stage in stages:
                if type(category) is str and category in categories:
                    records.append(
                        {
                            "restore_stage": stage,
                            "exception_class": category,
                            "frames": list(frames),
                        }
                    )

    logger = logging.getLogger("amlas.app")
    handler = RestoreDiagnosticHandler()
    logger.addHandler(handler)
    return records, logger, handler


def _capture_cold_loader_diagnostic(monkeypatch):
    from amlas.control_plane.normal_run_checkpoint import NormalRunContext

    records = []
    original = NormalRunContext.load_qualified_task

    def load(context, *args, **kwargs):
        try:
            result = original(context, *args, **kwargs)
        except BaseException as error:
            frames = []
            current = error.__traceback__
            while current is not None:
                name = Path(current.tb_frame.f_code.co_filename).name
                if name in {
                    "normal_run_checkpoint.py",
                    "normal_run_parents.py",
                    "engine.py",
                    "research_runtime_profile.py",
                }:
                    frames.append({"source": name, "line": current.tb_lineno})
                current = current.tb_next
            records.append(
                {
                    "stage": "load_qualified_task",
                    "outcome": "exception",
                    "exception_type": type(error).__name__,
                    "frames": frames,
                }
            )
            raise
        records.append({"stage": "load_qualified_task", "outcome": "returned", "none": result is None})
        return result

    monkeypatch.setattr(NormalRunContext, "load_qualified_task", load)
    return records


def child(data, directory, *, stale_cas=False, read_report=False, claim_loss=False, loss=None):
    """New PID receives only actual native transport DATA, never a graph/snapshot.

    claim_loss cuts inside the completion claim; loss names another cut: after-claim (the claim returned, the
    continuation never started) or after-gen (a generation backend accepted a job, its tool result is not committed).
    """
    lossy = claim_loss or loss is not None
    raw = json.dumps(wire_encode(data), sort_keys=True, separators=(",", ":")).encode()
    directory.mkdir()
    if directory.name == "baseline-completed":
        (directory / "baseline-input.private.json").write_bytes(raw)
    source_root = PATH.parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "src"), str(source_root / "tests"), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PATH),
            "--cold-component",
            str(directory),
            "loss"
            if claim_loss
            else ("loss-" + loss)
            if loss is not None
            else "report"
            if read_report
            else "stale"
            if stale_cas
            else "current",
            hashlib.sha256(raw).hexdigest(),
        ],
        input=raw,
        capture_output=True,
        env=env,
        timeout=COLD_CHILD_WAIT_SECONDS,
    )
    if lossy:
        assert result.returncode == 73, result.stderr.decode()[-5000:]
        answer = wire_decode(json.loads((directory / "claim-loss-transport.json").read_bytes()))
    else:
        assert result.returncode == 0, result.stderr.decode()[-5000:]
        answer = wire_decode(json.loads(result.stdout.splitlines()[-1]))
    assert answer["pid"] != os.getpid(), "replacement must not retain original process objects"
    _closed_baseline_receipt(directory, answer)
    return answer


def current_transport(state, data):
    """The native transport DATA of this process, in the shape a replacement child receives."""
    return {
        "run_id": data["run_id"],
        "ddb": copy.deepcopy(state.ports.items),
        "checkpoint_objects": [
            {"key": key, "version": version, "raw": raw}
            for (key, version), raw in state.ports.objects.items()
        ],
        "stage_objects": [
            {"key": key[0], "version": key[1], "raw": raw}
            for key, raw in state.objects.objects.items()
            if type(key) is tuple
        ],
        "training_requests": copy.deepcopy(state.native.calls),
        "processing_requests": copy.deepcopy(state.native.processing),
        "states": copy.deepcopy(state.native.states),
        "original_job": data["original_job"],
        **({"unselected": sorted(state.native.unselected)} if state.native.unselected else {}),
    }


def exit_with_transport(state, data, directory, label, **extra):
    """Persist the transport and end the process at once, as a worker loss does."""
    for _attempt in range(20):
        try:
            evidence = {
                "pid": os.getpid(),
                "transport": current_transport(state, data),
                "cut": label,
                "training_at_loss": len(state.native.calls),
                "processing_at_loss": len(state.native.processing),
                "loaded": len(state.loaded),
                "invoked": len(state.invoked),
                **extra,
            }
            raw = json.dumps(wire_encode(evidence), sort_keys=True, separators=(",", ":")).encode()
            break
        except RuntimeError:
            continue
    else:
        raise AssertionError("transport changed during every loss snapshot")
    with (directory / "claim-loss-transport.json").open("xb") as saved:
        saved.write(raw)
        saved.flush()
        os.fsync(saved.fileno())
    os._exit(73)


def install_loss(monkeypatch, state, data, directory, label):
    if label == "after-claim":
        # The fixture replaces the module attribute with a capturing factory; cut on the real class.
        from test_runtime_warm_model_continuation import _ACTUAL_ENGINE

        def lost_before_continuation(engine):
            exit_with_transport(state, data, directory, label)

        monkeypatch.setattr(_ACTUAL_ENGINE, "continue_normal_completion", lost_before_continuation)
    elif label == "after-gen":
        from amlas.control_plane.generation_state import SessionCodingGenerationStore

        submit = SessionCodingGenerationStore.submit

        def accepted_then_lost(store, *args, **kwargs):
            result = submit(store, *args, **kwargs)
            if result.get("accepted") is True:
                exit_with_transport(state, data, directory, label, accepted_ref=result["request_ref"])
            return result

        monkeypatch.setattr(SessionCodingGenerationStore, "submit", accepted_then_lost)
    else:
        raise AssertionError("unknown loss point")


async def cold_lookup(data, directory, stale_cas, read_report=False, claim_loss=False, loss=None):
    monkeypatch = pytest.MonkeyPatch()
    inert_runtime_setup.__wrapped__(monkeypatch)
    state = setup(monkeypatch, directory, captured=data, stale_cas=stale_cas)
    loader_diagnostic = _capture_cold_loader_diagnostic(monkeypatch)
    restore_diagnostic, restore_logger, restore_handler = _capture_cold_restore_diagnostic(monkeypatch)
    events = []
    initial = copy.deepcopy(state.ports.items)
    original_training = len(state.native.calls)
    original_processing = len(state.native.processing)
    if loss is not None:
        install_loss(monkeypatch, state, data, directory, loss)
    if claim_loss:
        from amlas.control_plane.normal_run_checkpoint import NormalRunContext

        source, start = inspect.getsourcelines(NormalRunContext.claim_completion)
        receipt_lines = [number for number, line in enumerate(source, start) if "object.__setattr__(" in line]
        assert len(receipt_lines) == 1
        receipt_line = receipt_lines[0]

        def cut_after_confirmed_readback(frame, event, arg):
            if frame.f_code is not NormalRunContext.claim_completion.__code__:
                return None
            if (
                event == "line"
                and frame.f_code is NormalRunContext.claim_completion.__code__
                and frame.f_lineno == receipt_line
            ):
                current = frame.f_locals
                assert current["self"]._completion_receipt is None
                assert current["confirmed"]["continuation"] == current["continuation"]
                assert current["confirmed"]["task_checkpoint_ref"] == current["record"]["task_checkpoint_ref"]
                assert current["continuation"]["status"] == "consumed-ready"
                row = state.ports.items["PROMPT_RUN#" + data["run_id"]]
                raw_row = row["record"]["S"]
                assert hashlib.sha256(raw_row.encode()).hexdigest() == row["record_sha256"]["S"]
                assert json.loads(raw_row) == current["confirmed"]
                matching_cas = [
                    kwargs
                    for operation, kwargs in state.ports.calls
                    if operation == "UpdateItem"
                    and kwargs["Key"]["job_name"]["S"] == "PROMPT_RUN#" + data["run_id"]
                    and kwargs["ExpressionAttributeValues"][":body"] == row["record"]
                ]
                assert len(matching_cas) == 1 and matching_cas[0].get("ConditionExpression")
                assert len(state.native.calls) == original_training
                assert len(state.native.processing) == original_processing
                assert state.loaded == [] and state.invoked == []
                transport = copy.deepcopy(data)
                transport["ddb"] = copy.deepcopy(state.ports.items)
                evidence = {
                    "pid": os.getpid(),
                    "transport": transport,
                    "confirmed_consumed_ready": True,
                    "conditional_update_and_readback": True,
                    "local_receipt_absent": True,
                    "sdk_load_or_invocation": False,
                    "training_before": original_training,
                    "training_at_loss": len(state.native.calls),
                    "processing_before": original_processing,
                    "processing_at_loss": len(state.native.processing),
                }
                raw = json.dumps(wire_encode(evidence), sort_keys=True, separators=(",", ":")).encode()
                with (directory / "claim-loss-transport.json").open("xb") as saved:
                    saved.write(raw)
                    saved.flush()
                    os.fsync(saved.fileno())
                os._exit(73)
            return cut_after_confirmed_readback

        threading.settrace(cut_after_confirmed_readback)

    async def read():
        async for event in runtime.agent_invocation(
            {"action": "get_prompt_report" if read_report else "resume_prompt_run", "run_id": data["run_id"]}
        ):
            events.append(event)
        return events

    reader = asyncio.create_task(read())
    state.reader, state.events = reader, events
    report = []
    try:
        deadline = asyncio.get_running_loop().time() + COLD_CHILD_WAIT_SECONDS
        # The normal resume stream starts only after successful completion claim.
        # Engine construction alone does not establish readiness to generate.
        while not reader.done() and not any(event.get("event") == "run_started" for event in events):
            assert asyncio.get_running_loop().time() < deadline, (
                "normal cold route neither continued nor returned a closed result"
            )
            await asyncio.sleep(0.005)
        if (
            state.work.engines
            and data["states"][data["original_job"]] == "Completed"
            and not stale_cas
            and not read_report
            and any(event.get("event") == "run_started" for event in events)
            # A fit pilot script ends after the SFT result: it has no generation or paired evaluation.
            and state.model.original_recipe_id == "full_v1"
        ):
            await asyncio.wait_for(
                finish_generation_pair(state, reader, deadline=deadline),
                timeout=max(0, deadline - asyncio.get_running_loop().time()),
            )
        elif not reader.done():
            # A native observation can precede snapshot admission and the public
            # response. Let the normal route finish before collecting evidence
            # or cancelling its reader during fixture cleanup.
            await asyncio.wait_for(
                asyncio.shield(reader),
                timeout=max(0, deadline - asyncio.get_running_loop().time()),
            )
        if reader.done():
            await reader
            report = (
                events
                if read_report
                else [
                    event
                    async for event in runtime.agent_invocation(
                        {"action": "get_prompt_report", "run_id": data["run_id"]}
                    )
                ]
            )
        graphs = []
        for engine in state.work.engines:
            graph = engine.tool_backend._warm_graph()
            graphs.append(
                {
                    "count": len(graph),
                    "correlations": [store.runtime_session_id for store in graph],
                    "records": [len(store._records) for store in graph],
                }
            )
        transport = {
            "run_id": data["run_id"],
            "ddb": copy.deepcopy(state.ports.items),
            "checkpoint_objects": [
                {"key": key, "version": version, "raw": raw}
                for (key, version), raw in state.ports.objects.items()
            ],
            "stage_objects": [
                {"key": key[0], "version": key[1], "raw": raw}
                for key, raw in state.objects.objects.items()
                if type(key) is tuple
            ],
            "training_requests": copy.deepcopy(state.native.calls),
            "processing_requests": copy.deepcopy(state.native.processing),
            "states": copy.deepcopy(state.native.states),
            "original_job": data["original_job"],
        }
        return wire_encode(
            {
                "transport": transport,
                "pid": os.getpid(),
                "events": events,
                "report": report,
                "loader_diagnostic": loader_diagnostic,
                "restore_diagnostic": restore_diagnostic,
                "loaded": state.loaded,
                "invoked": state.invoked,
                "graphs": graphs,
                "new_training": state.native.calls[original_training:],
                "new_processing": list(state.native.processing.values())[original_processing:],
                "stops": state.native.stops,
                "rejected_cas": state.rejected_cas,
                "ddb_before": initial,
                "ddb_after": copy.deepcopy(state.ports.items),
                "model_calls": state.model.calls,
                "marked_results": marked_results(state.model),
            }
        )
    finally:
        restore_logger.removeHandler(restore_handler)
        await close(state)
        if claim_loss:
            threading.settrace(None)
        monkeypatch.undo()


def test_normal_cold_pending_restores_original_graph_without_create(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            data = cloud_data(state)
            assert_safe_data(decoded(data["ddb"]))
            answer = child(data, tmp_path / "pending-replacement")
            assert answer["loaded"], (
                "normal cold pending route projected the job but never loaded its admitted SDK checkpoint"
            )
            assert answer["graphs"] and answer["graphs"][0]["count"] == 7
            assert answer["graphs"][0]["records"][:5] == [2, 1, 1, 1, 1], (
                "cold pending original lost genuine acquisition/model/case/preparation/SFT parents"
            )
            assert len(set(answer["graphs"][0]["correlations"])) == 1
            assert answer["new_training"] == answer["new_processing"] == answer["stops"] == []
            assert answer["model_calls"] == 0 and answer["invoked"] == [], (
                "pending native original must not replay model or tools"
            )
            assert state.agent.messages == state.prefix and SFT_USE not in payloads(state.agent.messages)
        finally:
            await close(state)

    asyncio.run(scenario())


def test_normal_cold_completed_sft_consumes_original_batch_once_then_actual_gen_pair(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            data = cloud_data(state, completed=True)
            # One full warm control on the SAME genuinely submitted original.
            # Export happened at PRECOMMIT; no successor records go to the child.
            state.native.states[state.original.id] = "Completed"
            warm = await finish_generation_pair(state, state.reader)
            terminal = warm[-1]
            assert terminal["event"] == "run_completed" and terminal["worker_finished"] is True
            assert FINAL_TEXT in terminal["response"]["message"]
            assert len(state.native.calls) == 3 and len(state.backend._pair_store._records) == 1
            completed = payloads(state.agent.messages)[SFT_USE]
            assert completed["event_type"] == "original_sft_completed"
            assert completed["status"] == "verified_completed" and completed["data_only"] is True
            assert completed["request_ref"] == next(iter(state.backend._sft_store._records))
            assert state.original.extra["recipe_id"] == "full_v1"
            verified = state.backend._sft_store.observe(completed["request_ref"])
            assert verified["full_training_complete"] is True and verified["trained"] is True
            answer = child(data, tmp_path / "completed-replacement")
            assert answer["loaded"] and answer["invoked"], (
                "normal Completed original refused instead of restoring its admitted batch and continuing the SDK"
            )
            assert all(value["prompt"] is None for value in answer["invoked"])
            restored = answer["invoked"][0]["messages"]
            assert _record_messages(restored)[: len(state.prefix)] == _record_messages(state.prefix), (
                "cold continuation rewrote prior admitted history"
            )
            sft_results = [
                block["toolResult"]
                for message in restored
                for block in message["content"]
                if block.get("toolResult", {}).get("toolUseId") == SFT_USE
            ]
            assert len(sft_results) == 1, (
                "original completed result must be committed exactly once before load_snapshot"
            )
            assert (
                payloads(restored)[SFT_USE]["request_ref"]
                == payloads(state.agent.messages)[SFT_USE]["request_ref"]
            )
            assert len(answer["new_training"]) == 2 and len(answer["new_processing"]) == 1
            assert all(value["TrainingJobName"] != state.original.id for value in answer["new_training"])
            assert answer["stops"] == []
            cold_terminal = next(
                value for value in reversed(answer["events"]) if value.get("event") == "run_completed"
            )
            assert cold_terminal["run_id"] == state.run_id and cold_terminal["worker_finished"] is True
            # The restored run's final message ends with the runtime recovery record; the warm control has none.
            cold_head, cold_separator, cold_record = cold_terminal["response"]["message"].partition(
                "\n\n## Runtime recovery record\n"
            )
            assert (
                cold_separator
                and cold_record.count("\n") == 2
                and cold_record.splitlines()[-1].startswith("Records: ")
            )
            cold_conversation = {**cold_terminal["response"], "message": cold_head}
            assert stable_response(cold_conversation) == stable_response(terminal["response"])
            warm_evidence = fresh_admission_evidence(
                terminal["response"], required_boundaries={"OWNER_PROMPT", "TOOL_RESULT"}
            )
            cold_evidence = fresh_admission_evidence(
                cold_terminal["response"], required_boundaries={"HISTORY_TEXT", "TOOL_RESULT"}
            )
            assert warm_evidence[3] == cold_evidence[3], (
                "replacement selected a different guard configuration"
            )
            for warm_ids, cold_ids in zip(warm_evidence[:3], cold_evidence[:3], strict=True):
                assert warm_ids.isdisjoint(cold_ids), "replacement reused prior admission identities or HMAC"
            expected = [
                {
                    "event": "prompt_report",
                    "status": "available",
                    "run_id": state.run_id,
                    "found": True,
                    "report": cold_terminal,
                }
            ]
            assert answer["report"] == expected, (
                "the unchanged passive report route lost the same original result"
            )
            again = child(answer["transport"], tmp_path / "report-replacement", read_report=True)
            assert again["report"] == expected, "a second replacement lost the durable final report"
            assert again["new_training"] == again["new_processing"] == again["invoked"] == []
            assert again["model_calls"] == 0
        finally:
            await close(state)

    asyncio.run(scenario())


def test_normal_cold_completed_fit_sft_continues_to_the_final_report(monkeypatch, tmp_path):
    """A replacement continues a completed fit_v1 original (the pilot recipe) to its final report.

    The warm control is the original worker on the same submitted job. No reload, generation or paired
    evaluation belongs to this script, so a replacement that continues creates no native job at all.
    """
    monkeypatch.setenv("AMLAS_COLD_MODEL_VARIANT", "fit-only")

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            assert state.original.extra["recipe_id"] == "fit_v1"
            data = cloud_data(state, completed=True)
            state.native.states[state.original.id] = "Completed"
            warm = await asyncio.wait_for(state.reader, 120)
            terminal = warm[-1]
            assert terminal["event"] == "run_completed" and terminal["worker_finished"] is True
            assert FIT_FINAL_TEXT in terminal["response"]["message"]
            assert len(state.native.calls) == 1 and not state.backend._generation_store._records
            warm_results = payloads(state.agent.messages)
            completed = warm_results[SFT_USE]
            assert completed["event_type"] == "original_sft_completed"
            assert completed["status"] == "verified_completed" and completed["data_only"] is True
            assert completed["request_ref"] == next(iter(state.backend._sft_store._records))
            verified = state.backend._sft_store.observe(completed["request_ref"])
            assert verified["trained"] is True and verified["recipe_id"] == "fit_v1"
            # The adapter never reports a fit pilot as a complete full training.
            assert verified["full_training_complete"] is False
            assert warm_results[FIT_STATUS_USE]["trained"] is True

            answer = child(data, tmp_path / "fit-completed-replacement")
            closed = [
                {key: event.get(key) for key in ("event", "state", "status") if key in event}
                for event in answer["events"]
                if event.get("event") != "heartbeat"
            ]
            assert answer["loaded"] and answer["invoked"], (
                "a completed fit_v1 original was not continued by the replacement: "
                + json.dumps({"events": closed, "restore": answer["restore_diagnostic"]}, sort_keys=True)
            )
            assert closed[0] == {"event": "resume_state", "state": "restoring"}
            assert [value["event"] for value in closed[1:]] == ["run_started", "run_completed"]
            assert all(value["prompt"] is None for value in answer["invoked"])
            restored = answer["invoked"][0]["messages"]
            assert _record_messages(restored)[: len(state.prefix)] == _record_messages(state.prefix), (
                "cold continuation rewrote prior admitted history"
            )
            sft_results = [
                block["toolResult"]
                for message in restored
                for block in message["content"]
                if block.get("toolResult", {}).get("toolUseId") == SFT_USE
            ]
            assert len(sft_results) == 1, (
                "original completed result must be committed exactly once before load_snapshot"
            )
            assert payloads(restored)[SFT_USE]["request_ref"] == completed["request_ref"]
            # The replacement reads the same original job and submits nothing.
            assert answer["new_training"] == answer["new_processing"] == answer["stops"] == []
            assert answer["model_calls"] == 2
            cold_terminal = answer["events"][-1]
            assert cold_terminal["event"] == "run_completed"
            assert cold_terminal["run_id"] == state.run_id and cold_terminal["worker_finished"] is True
            cold_head, cold_separator, cold_record = cold_terminal["response"]["message"].partition(
                "\n\n## Runtime recovery record\n"
            )
            assert FIT_FINAL_TEXT in cold_head
            assert (
                cold_separator
                and cold_record.count("\n") == 2
                and cold_record.splitlines()[-1].startswith("Records: ")
            )
            cold_conversation = {**cold_terminal["response"], "message": cold_head}
            assert stable_response(cold_conversation) == stable_response(terminal["response"])
            expected = [
                {
                    "event": "prompt_report",
                    "status": "available",
                    "run_id": state.run_id,
                    "found": True,
                    "report": cold_terminal,
                }
            ]
            assert answer["report"] == expected, "the passive report route lost the fit pilot's result"
            again = child(answer["transport"], tmp_path / "fit-report-replacement", read_report=True)
            assert again["report"] == expected, "a second replacement lost the durable final report"
            assert again["new_training"] == again["new_processing"] == again["invoked"] == []
            assert again["model_calls"] == 0
        finally:
            await close(state)

    asyncio.run(scenario())


def test_normal_stale_completion_cas_refuses_result_and_successor_intent(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            data = cloud_data(state, completed=True)
            answer = child(data, tmp_path / "stale-replacement", stale_cas=True)
            assert answer["rejected_cas"], (
                "Completed original reached no conditional completion checkpoint fence"
            )
            key = "PROMPT_RUN#" + state.run_id
            assert answer["ddb_after"][key] == answer["ddb_before"][key], (
                "failed completion CAS altered the last confirmed original run row"
            )
            assert answer["invoked"] == [] and answer["model_calls"] == 0
            assert answer["new_training"] == answer["new_processing"] == answer["stops"] == []
            assert not any(
                value.get("event") == "run_completed" and value.get("worker_finished") is True
                for value in answer["events"]
            )
            assert not any(value.get("status") == "available" for value in answer["report"])
        finally:
            await close(state)

    asyncio.run(scenario())


def test_normal_cold_recovers_same_operation_after_confirmed_claim_process_loss(monkeypatch, tmp_path):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            data = cloud_data(state, completed=True)
            baseline = child(data, tmp_path / "baseline-completed")
            assert baseline["loaded"] and baseline["invoked"]
            assert len(baseline["new_training"]) == 2 and len(baseline["new_processing"]) == 1
            assert any(event.get("event") == "run_completed" for event in baseline["events"])
            loss = child(data, tmp_path / "confirmed-claim-loss", claim_loss=True)
            assert loss["confirmed_consumed_ready"] and loss["conditional_update_and_readback"]
            assert loss["local_receipt_absent"] and loss["sdk_load_or_invocation"] is False
            assert loss["training_before"] == loss["training_at_loss"]
            assert loss["processing_before"] == loss["processing_at_loss"]
            resumed = child(loss["transport"], tmp_path / "after-claim-loss")
            if not resumed["invoked"]:
                assert resumed["new_training"] == resumed["new_processing"] == resumed["stops"] == []
                assert resumed["ddb_after"] == loss["transport"]["ddb"]
            statuses = [
                item.get("status") for item in resumed["events"] if item.get("event") == "prompt_run_status"
            ]
            fixed = {
                "baseline_completed": True,
                "baseline_new_training": len(baseline["new_training"]),
                "baseline_new_processing": len(baseline["new_processing"]),
                "kill_after_confirmed_readback_before_receipt": True,
                "persisted_status": "consumed-ready",
                "training_before_loss": loss["training_before"],
                "training_at_loss": loss["training_at_loss"],
                "processing_before_loss": loss["processing_before"],
                "processing_at_loss": loss["processing_at_loss"],
                "replacement_statuses": statuses,
                "replacement_loaded": bool(resumed["loaded"]),
                "replacement_invoked": bool(resumed["invoked"]),
                "replacement_new_training": len(resumed["new_training"]),
                "replacement_new_processing": len(resumed["new_processing"]),
            }
            print("CLAIM_LOSS_FIXED=" + json.dumps(fixed, sort_keys=True), flush=True)
            assert resumed["loaded"] and resumed["invoked"], (
                "same original operation did not progress after confirmed claim process loss"
            )
            assert len(resumed["new_training"]) == 2 and len(resumed["new_processing"]) == 1
            assert all(value["TrainingJobName"] != state.original.id for value in resumed["new_training"])
        finally:
            await close(state)

    asyncio.run(scenario())


if __name__ == "__main__":
    assert sys.argv[1] == "--cold-component"
    raw = sys.stdin.buffer.read()
    assert hashlib.sha256(raw).hexdigest() == sys.argv[4]
    directory = Path(sys.argv[2])
    data = wire_decode(json.loads(raw))
    print(
        json.dumps(
            asyncio.run(
                cold_lookup(
                    data,
                    directory,
                    sys.argv[3] == "stale",
                    sys.argv[3] == "report",
                    sys.argv[3] == "loss",
                    sys.argv[3][5:] if sys.argv[3].startswith("loss-") else None,
                )
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
