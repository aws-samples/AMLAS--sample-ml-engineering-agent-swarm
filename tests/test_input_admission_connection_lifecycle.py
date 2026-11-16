"""Actual original SDK completion with owned transport rotation; no native calls.

Botocore1.43.57 maps urllib3.ProtocolError to ConnectionClosedError. This strict
idle flag is a transport hypothesis, not proof of the original native cause.
"""

from __future__ import annotations

import asyncio
import copy
import json
from itertools import pairwise
from types import SimpleNamespace

import pytest
from botocore.exceptions import ConnectionClosedError
from botocore.session import get_session
from botocore.validate import validate_parameters
from test_input_admission import none_response
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_precommit_original_completion import (
    USES,
    ThreeOriginals,
    _events,
    captured_originals,
    protocol_work,
    publish_original_data,
    results,
)
from test_runtime_warm_model_continuation import complete_fixture, original_work, step
from test_warm_model_closure import original_results, retained_original

pytest_plugins = ("test_runtime_prompt_entry",)

ENDPOINT = "https://bedrock-runtime.us-west-2.amazonaws.com"
GUARD = "fixtureguard"


class GuardTransport:
    def __init__(self, label, events, *, fail=False):
        self.label, self.events, self.fail = label, events, fail
        self.calls = []
        self.closed = False
        self.retired = False
        self.idle = False
        self.input_shape = (
            get_session().get_service_model("bedrock-runtime").operation_model("ApplyGuardrail").input_shape
        )

    def apply_guardrail(self, **kwargs):
        validate_parameters(kwargs, self.input_shape)
        assert kwargs["source"] == "INPUT" and kwargs["outputScope"] == "FULL"
        assert kwargs["guardrailIdentifier"] == GUARD and kwargs["guardrailVersion"] == "DRAFT"
        assert not self.retired, "retired guard transport was used"
        # SDK close clears pools; apply creates a fresh pool on the same client.
        self.closed = False
        self.calls.append(copy.deepcopy(kwargs))
        self.events.append(("apply", self.label))
        if self.idle or self.fail:
            raise ConnectionClosedError(endpoint_url=ENDPOINT)
        response = none_response()
        count = len(kwargs["content"][0]["text"]["text"])
        response["guardrailCoverage"]["textCharacters"] = {"guarded": count, "total": count}
        return response

    def close(self):
        self.events.append(("close", self.label))
        self.closed = True
        self.idle = False


def original_with_transport(monkeypatch, tmp_path, *, fail_fresh=False):
    import boto3

    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", GUARD)
    monkeypatch.setenv("AMLAS_GUARDRAIL_VERSION", "DRAFT")
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "true")
    events, sessions, clients = [], [], []

    class Session:
        def __init__(self, *, profile_name, region_name):
            assert region_name == "us-west-2"
            self.profile = profile_name
            sessions.append(self)

        def client(self, service, *, region_name, config):
            assert service == "bedrock-runtime" and region_name == "us-west-2"
            assert config.connect_timeout == 5 and config.read_timeout == 15
            assert config.retries == {"total_max_attempts": 1}
            assert len(clients) < 2, "failed assessment created a third client"
            if clients:
                assert clients[0].closed, "fresh transport was created before retiring the original"
                assert self is sessions[0], "transport rotation changed the original SDK Session"
                clients[-1].retired = True
            label = "A" if not clients else "B"
            value = GuardTransport(label, events, fail=fail_fresh and label == "B")
            clients.append(value)
            events.append(("create", label))
            return value

    monkeypatch.setattr(boto3, "Session", Session)
    work = original_work(monkeypatch)
    retained_original(work)
    assert len(work.native.requests) == 1 and work.model.calls == 2
    assert len(clients) == len(sessions) == 1 and clients[0].calls
    assert work.backend._warm_pending.record.handle is work.handle
    checker = work.backend._input_checker
    config, key = checker.config, checker._key
    before = copy.deepcopy(work.agent.messages)
    pending_results = copy.deepcopy(original_results(work))
    callback = work.agent.callback_handler
    policy = work.backend.approval_policy
    consumed = set(work.backend._warm_consumed)
    complete_fixture(monkeypatch, tmp_path, work)
    # Actual owning9file/3manifest verification precedes the intended RED.
    verified = work.backend._model_acquisition_store.observe(work.reference)
    assert verified["success"] is True and verified["acquired"] is True
    assert len(verified["file_refs"]) == 9 and set(verified["manifest_refs"]) == {
        "receipt",
        "commit",
        "lineage",
    }
    clients[0].idle = True
    early_count = len(clients[0].calls)
    outcome = step(work)
    assert len(work.native.requests) == len(work.engines) == len(work.agents) == 1
    assert work.backend._model_acquisition_store._records[work.reference].handle is work.handle
    # Baseline821 fails here: it reusedA and held rather than retiring beforeB.
    assert len(clients) == 2 and clients[0].closed, (
        "warm completion did not retire its original guard transport"
    )
    assert len(clients[0].calls) == early_count, "the new logical assessment reached the idle pool"
    assert len(clients[1].calls) == 1 and len(sessions) == 1
    assert events.index(("close", "A")) < events.index(("create", "B")) < events.index(("apply", "B"))
    assert work.backend._input_checker.config == config
    assert work.backend._input_checker._key == key
    assert work.agent.callback_handler is callback
    assert work.backend.approval_policy is policy
    assert original_results(work) == pending_results
    retained_original(work)
    return SimpleNamespace(
        work=work,
        outcome=outcome,
        clients=clients,
        events=events,
        before=before,
        consumed=consumed,
        key=key,
        config=config,
    )


def test_original_warm_completion_retires_idle_transport_before_one_fresh_assessment(monkeypatch, tmp_path):
    value = original_with_transport(monkeypatch, tmp_path)
    work = value.work
    assert value.outcome.disposition == "continued_original"
    assert work.model.calls == 3 and work.backend._warm_pending is None
    assert work.agent.messages[: len(value.before)] == value.before
    completed = [
        json.loads(block["text"])
        for message in work.agent.messages[len(value.before) :]
        if message["role"] == "user"
        for block in message["content"]
        if "text" in block
    ]
    assert len(completed) == 1 and completed[0]["event_type"] == "original_model_acquisition_completed"
    assert (
        completed[0]["request_ref"] == work.reference and completed[0]["tool_use_id"] == "original-model-use"
    )
    assert len(work.backend._warm_consumed - value.consumed) == 1


def test_fresh_transport_connection_failure_holds_without_client_or_assessment_replay(monkeypatch, tmp_path):
    value = original_with_transport(monkeypatch, tmp_path, fail_fresh=True)
    work = value.work
    assert value.outcome.disposition == "refused" and work.backend._input_hold is True
    assert work.model.calls == 2 and work.agent.messages == value.before
    assert work.backend._warm_consumed == value.consumed
    assert work.backend._warm_pending.record.handle is work.handle
    record = work.backend._input_admission_data()["records"][-1]
    assert record["boundary"] == "COMPLETION_DATA" and record["reason"] == "TRANSPORT_UNKNOWN"
    assert record["outcome"] == "UNKNOWN"
    assert record["transport_diagnostic"]["error_class"] == "ConnectionClosedError"
    assert record["native_admission"] is record["quality_success"] is False
    events = list(value.events)
    assert step(work).disposition == "refused"
    assert value.events == events and len(value.clients) == 2
    assert work.model.calls == 2 and len(work.native.requests) == 1
    assert work.agent.messages == value.before and work.backend._warm_consumed == value.consumed


@pytest.mark.parametrize("fail_fresh", [False, True], ids=["fresh-final-admission", "fresh-unknown-holds"])
@pytest.mark.usefixtures("inert_runtime_setup")
def test_normal_precommit_originals_retire_owned_pool_before_first_final_assessment(
    monkeypatch, tmp_path, fail_fresh
):
    import boto3

    model = ThreeOriginals()
    work, data, _fetched = protocol_work(monkeypatch, model)
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", GUARD)
    events, sessions, clients = [], [], []

    class Session:
        def __init__(self, *, profile_name, region_name):
            assert region_name == "us-west-2"
            sessions.append(self)

        def client(self, service, *, region_name, config):
            assert service == "bedrock-runtime" and region_name == "us-west-2"
            assert config.connect_timeout == 5 and config.read_timeout == 15
            assert config.retries == {"total_max_attempts": 1}
            assert len(clients) < 4, "completion attempted an extra client/assessment"
            if clients:
                assert clients[-1].closed, "fresh client preceded retirement of current owned pool"
                assert self is sessions[0], "refresh changed the owning SDK Session"
                clients[-1].retired = True
            label = str(len(clients))
            value = GuardTransport(label, events, fail=fail_fresh and bool(clients))
            clients.append(value)
            events.append(("create", label))
            return value

    monkeypatch.setattr(boto3, "Session", Session)

    async def scenario():
        reader = asyncio.create_task(_events(work))
        try:
            backend = await captured_originals(work, data, reader)
            deadline = asyncio.get_running_loop().time() + 5
            while len(backend._pending_batch_pause.originals) != 3:
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert len(clients) == len(sessions) == 1 and model.calls == 4
            original = backend._warm_pending
            producers = tuple(backend._warm_pending_producers.values())
            before = copy.deepcopy(work.agents[0].messages)
            checker, cycle = backend._input_checker, backend._input_cycle
            config, key, policy = checker.config, checker._key, backend.approval_policy
            graph = backend._warm_graph()
            early_count = len(clients[0].calls)
            clients[0].idle = True
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=original.record.handle, adapter=work.adapter, native=work.native),
            )
            for pending in producers:
                publish_original_data(pending.record, data)
            terminal = await asyncio.wait_for(reader, 5)
            assert len(clients) > 1 and clients[0].closed, (
                "precommit original completion did not retire its owned pool"
            )
            assert len(clients[0].calls) == early_count, "first final assessment reused the prior owned pool"
            assert (
                len(sessions) == 1
                and backend._input_checker.config == config
                and backend._input_checker._key == key
            )
            assert (
                backend.approval_policy is policy
                and backend._input_cycle == cycle
                and backend._warm_graph() == graph
            )
            assert backend._research_binding["checker"] is backend._input_checker
            assert (
                len(work.native.requests) == 1
                and len(data.native.calls) == 2
                and len(work.agents) == len(work.engines) == 1
            )
            assert work.agents[0].messages[: len(before)] == before
            assert (
                terminal[-1]["worker_finished"] is True
                and work.owned_app.get_async_task_info()["active_count"] == 0
            )
            for previous, current in pairwise(clients):
                assert (
                    events.index(("close", previous.label))
                    < events.index(("create", current.label))
                    < events.index(("apply", current.label))
                )
                assert len(current.calls) == 1
            if fail_fresh:
                assert (
                    len(clients) == 2
                    and backend._input_hold
                    and not backend._warm_consumed
                    and model.calls == 4
                )
                assert (
                    backend._warm_pending is original
                    and tuple(backend._warm_pending_producers.values()) == producers
                )
                record = backend._input_admission_data()["records"][-1]
                assert (
                    record["boundary"] == "TOOL_RESULT"
                    and record["outcome"] == "UNKNOWN"
                    and record["reason"] == "TRANSPORT_UNKNOWN"
                )
                assert record["transport_diagnostic"]["error_class"] == "ConnectionClosedError"
                assert all(
                    value["status"] == "error"
                    for value in results(work.agents[0])
                    if value["toolUseId"] in USES
                )
            else:
                assert len(clients) == 4 and model.calls == 5 and not backend._input_hold
                assert backend._warm_pending is None and not backend._warm_pending_producers
                assert len(backend._warm_consumed) == 3
                final = [value for value in results(work.agents[0]) if value["toolUseId"] in USES]
                assert len(final) == 3 and all(value["status"] == "success" for value in final)
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
