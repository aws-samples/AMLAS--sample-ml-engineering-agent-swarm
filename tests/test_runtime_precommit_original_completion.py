"""Product RED: selected normal profile owns final original tool-result completion.

Actual app/API/ChatEngine/SDK over strict inert MODEL/DATA/guard ports. Synthetic
bytes prove source custody/protocol only, not native guard/ML/lifecycle acceptance.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
from case_prep_continuation_fixtures import move_source_port, wait_until
from multi_producer_fixtures import configured_data
from strands.hooks.events import BeforeInvocationEvent, BeforeModelCallEvent
from strands.tools.executors import ConcurrentToolExecutor
from test_managed_coding_case_envelope import rows
from test_managed_dataset_acquisition import populate
from test_normal_research_profile import CASE_URLS, MODEL_URLS, TRAIN_URL
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_warm_model_continuation import OriginalModelStream, complete_fixture, tool_payload
from test_sdk_pending_handoff import profile_work

from amlas import app as runtime
from amlas.backends.adapters.artifact_rejection import is_closed_rejection
from amlas.control_plane.coding_case_state import SessionCodingCaseFreezeStore
from amlas.data_eng.coding_case_managed import freeze_managed_case_bundle
from amlas.tools.research import public_fetch
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
USES = ("original-model-use", "original-data-one", "original-data-two")


class ThreeOriginals(OriginalModelStream):
    def __init__(self, *, legacy=False, research_count=None):
        super().__init__()
        self.legacy = legacy
        self.research_count = (0 if legacy else 3) if research_count is None else research_count
        self.original_cycles = 0

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.messages_seen.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        if self.calls <= self.research_count:
            batch = (MODEL_URLS, (TRAIN_URL,), CASE_URLS)[self.calls - 1]
            selected = [
                (
                    "research-unit-" + str(self.calls),
                    "research_fetch_urls",
                    {"urls": list(batch), "max_pages": 3},
                )
            ]
        elif self.calls == self.research_count + 1:
            self.original_cycles += 1
            selected = [
                (USES[0], "acquire_model_weights", {"catalog_id": "qwen3_5_9b", "catalog_version": "1"}),
                (
                    USES[1],
                    "acquire_huggingface_dataset",
                    {"repo_id": "fixture/open-code", "revision": "b" * 40, "filename": "data/train.jsonl"},
                ),
                (
                    USES[2],
                    "acquire_huggingface_dataset",
                    {"repo_id": "fixture/open-code", "revision": "b" * 40, "filename": "data/extra.jsonl"},
                ),
            ]
        else:
            self.original_cycles += 1
            assert self.original_cycles == 2
            assert messages[-1]["role"] == "user"
            assert {block["toolResult"]["toolUseId"] for block in messages[-1]["content"]} == set(USES)
            if self.research_count == 3:
                payloads = [tool_payload(block["toolResult"]) for block in messages[-1]["content"]]
                assert all(value.get("data_only") is True for value in payloads)
                assert {value["tool_use_id"] for value in payloads} == set(USES)
                assert {value["event_type"] for value in payloads} == {
                    "original_model_acquisition_completed",
                    "original_data_completed",
                }
                assert all("file_refs" not in value and "manifest_refs" not in value for value in payloads)
            selected = []
        if selected:
            for index, (identifier, name, arguments) in enumerate(selected):
                yield {
                    "contentBlockStart": {
                        "contentBlockIndex": index,
                        "start": {"toolUse": {"toolUseId": identifier, "name": name}},
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "contentBlockIndex": index,
                        "delta": {"toolUse": {"input": json.dumps(arguments)}},
                    }
                }
                yield {"contentBlockStop": {"contentBlockIndex": index}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {
                        "text": "Original operations remain pending."
                        if self.legacy
                        else "All original artifacts verified."
                    },
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }


def results(agent):
    return [
        block["toolResult"]
        for message in agent.messages
        for block in message["content"]
        if "toolResult" in block
    ]


def publish_original_data(record, data):
    handle = record.handle
    populate(handle, data.s3)
    prefix = handle.extra["output_prefix"]
    key = prefix + "ACQUIRED.json"
    receipt = json.loads(data.s3.objects[key])
    receipt["runtime_session_id"] = handle.extra["runtime_session_id"]
    receipt["filename"] = handle.extra["filename"]
    lineage_key = prefix + receipt["lineage"]
    lineage = json.loads(data.s3.objects[lineage_key])
    lineage["filename"] = handle.extra["filename"]
    lineage["source_uri"] = (
        "https://huggingface.co/datasets/fixture/open-code/resolve/"
        + handle.extra["revision"]
        + "/"
        + handle.extra["filename"]
    )
    lineage_raw = json.dumps(lineage).encode()
    data.s3.objects[lineage_key] = lineage_raw
    commit_key = prefix + receipt["commit_manifest"]
    committed = json.loads(data.s3.objects[commit_key])
    committed["lineage_sha256"] = hashlib.sha256(lineage_raw).hexdigest()
    committed_raw = json.dumps(committed).encode()
    data.s3.objects[commit_key] = committed_raw
    receipt["commit_manifest_sha256"] = hashlib.sha256(committed_raw).hexdigest()
    data.s3.objects[key] = json.dumps(receipt).encode()
    data.native.states[handle.id] = "Completed"


def protocol_work(monkeypatch, model, *, profile="qwen9b_core_v1", deny=None):
    async def short_original_wait(_backend):
        await asyncio.sleep(0.01)

    monkeypatch.setattr(ToolBindingBackend, "_wait_precommit_original", short_original_wait)
    work = profile_work(monkeypatch, model=model, profile=profile, deny=deny)
    data = configured_data(monkeypatch)
    fetched = []
    allowed_urls = (*MODEL_URLS, TRAIN_URL, *CASE_URLS)

    def checked(url, timeout):
        assert url in allowed_urls and 0 < timeout <= 5
        return ("93.184.216.34",)

    def request_source(url, ips, timeout, max_bytes):
        assert ips == ("93.184.216.34",) and 0 < timeout <= 5 and max_bytes == 256 * 1024
        fetched.append(url)
        return 200, {"content-type": "text/plain"}, b"Neutral independently licensed coding research fixture."

    monkeypatch.setattr(public_fetch, "_checked_addresses", checked)
    monkeypatch.setattr(public_fetch, "_request_public", request_source)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","data_collection_flow"]')
    return work, data, fetched


@pytest.mark.parametrize(
    ("legacy", "record_order"),
    [(False, "natural"), (True, "natural"), (False, "train-first"), (False, "extra-first")],
    ids=[
        "qualified-profile-final-original-results",
        "default-legacy-pending-results",
        "qualified-profile-train-first-records",
        "qualified-profile-extra-first-records",
    ],
)
def test_normal_app_original_model_and_data_complete_before_selected_sdk_result_commit(
    monkeypatch, tmp_path, legacy, record_order
):
    model = ThreeOriginals(legacy=legacy)
    work, data, fetched = protocol_work(monkeypatch, model, profile="off" if legacy else "qwen9b_core_v1")
    allowed_urls = (*MODEL_URLS, TRAIN_URL, *CASE_URLS)
    completions = []
    actual_complete = work.owned_app.complete_async_task

    def complete_once(identifier):
        completions.append(identifier)
        return actual_complete(identifier)

    monkeypatch.setattr(work.owned_app, "complete_async_task", complete_once)

    async def scenario():
        reader = asyncio.create_task(_events(work))
        try:
            deadline = asyncio.get_running_loop().time() + 10
            # Actual original Describe boundary proves source observation; stream
            # start or the last native Create does not prove SDK hook settlement.
            while not (
                work.observing.is_set()
                and len(data.native.calls) == 2
                and len(work.engines[0].tool_backend._acquisition_store._records) == 2
                and len(work.engines[0].tool_backend._warm_pending_producers) == 2
            ):
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            backend, agent = work.engines[0].tool_backend, work.agents[0]
            assert type(agent.tool_executor) is ConcurrentToolExecutor
            assert len(work.native.requests) == 1 and len(data.native.calls) == 2
            assert len(work.engines) == len(work.agents) == 1
            assert not reader.done() and work.owned_app.get_async_task_info()["active_count"] == 1
            assert completions == []
            if legacy:
                assert backend._research_binding is None and model.calls == 2
                assert len(results(agent)) == 3
                assert all(tool_payload(result).get("acquired") is False for result in results(agent))
                backend.approval_policy = None
                terminal = await asyncio.wait_for(reader, 5)
                assert terminal[-1]["worker_finished"] is True and len(completions) == 1
                return
            assert backend._research_binding is not None and backend._research_binding["phase"] == "CONSUMED"
            assert fetched == list(allowed_urls)
            assert model.calls == 4 and model.original_cycles == 1
            acquisitions = [result for result in results(agent) if result["toolUseId"] in USES]
            assert acquisitions == [], (
                "pending native acknowledgements committed before qualified original completion"
            )
            before = copy.deepcopy(agent.messages)
            assert before[0]["content"] == [{"text": work.literal}]
            assert (
                len([result for result in results(agent) if result["toolUseId"].startswith("research-unit-")])
                == 3
            )
            assert {block["toolUse"]["toolUseId"] for block in before[-1]["content"]} == set(USES)
            original_model = next(iter(backend._model_acquisition_store._records.values()))
            if record_order != "natural":
                # Permute only registration order of the same genuine original
                # records; producer custody, handles, keys and state stay intact.
                first_filename = "data/train.jsonl" if record_order == "train-first" else "data/extra.jsonl"
                with backend._acquisition_store._lock:
                    ordered_records = sorted(
                        backend._acquisition_store._records.items(),
                        key=lambda item: item[1].handle.extra["filename"] != first_filename,
                    )
                    backend._acquisition_store._records.clear()
                    backend._acquisition_store._records.update(ordered_records)
            data_records = tuple(
                next(
                    pending.record
                    for pending in backend._warm_pending_producers.values()
                    if pending.tool_use_id == identifier
                )
                for identifier in USES[1:]
            )
            assert len(data_records) == 2 and all(record.adapter is data.adapter for record in data_records)
            assert {id(pending.record) for pending in backend._warm_pending_producers.values()} == {
                id(record) for record in data_records
            }
            assert all(
                pending.agent is agent and pending.store is backend._acquisition_store
                for pending in backend._warm_pending_producers.values()
            )
            publish_original_data(data_records[0], data)
            deadline = asyncio.get_running_loop().time() + 5
            while USES[1] not in backend._pending_batch_pause.completed:
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert model.calls == 4 and not [
                result for result in results(agent) if result["toolUseId"] in USES
            ]
            assert backend._pending_batch_pause.request_state.get("stop_event_loop") is not True
            assert len(backend._warm_pending_producers) == 2 and backend._warm_pending is not None
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=original_model.handle, adapter=work.adapter, native=work.native),
            )
            publish_original_data(data_records[1], data)
            terminal = await asyncio.wait_for(reader, 5)
            assert terminal[-1]["worker_finished"] is True and len(completions) == 1
            assert work.owned_app.get_async_task_info()["active_count"] == 0
            assert (
                model.calls == 5 and model.original_cycles == 2 and len(work.agents) == len(work.engines) == 1
            )
            assert len(work.native.requests) == 1 and len(data.native.calls) == 2
            assert agent.messages[: len(before)] == before
            final = [result for result in results(agent) if result["toolUseId"] in USES]
            assert len(final) == 3 and {result["toolUseId"] for result in final} == set(USES)
            assert all(result["status"] == "success" for result in final)
            payloads = [tool_payload(result) for result in final]
            assert all(value["data_only"] is True for value in payloads)
            assert {value["tool_use_id"] for value in payloads} == set(USES)
            assert {value["event_type"] for value in payloads} == {
                "original_model_acquisition_completed",
                "original_data_completed",
            }
            model_data = next(value for value in payloads if value["tool_use_id"] == USES[0])
            assert model_data["acquired"] is True and model_data["file_count"] == 9
            assert model_data["manifest_roles"] == ["receipt", "commit", "lineage"]
            assert all(
                value["status"] == "verified_completed"
                for value in payloads
                if value["tool_use_id"] != USES[0]
            )
            assert all("file_refs" not in value and "manifest_refs" not in value for value in payloads)
            assert work.adapter.artifacts(original_model.handle)["acquired"] is True
            assert all(data.adapter.artifacts(record.handle)["success"] is True for record in data_records)
            later_users = [message for message in agent.messages[len(before) :] if message["role"] == "user"]
            assert len(later_users) == 1 and all(
                set(block) == {"toolResult"} for block in later_users[0]["content"]
            )
            assert all(
                record["outcome"] == "ALLOW"
                for record in terminal[-1]["response"]["data"]["input_admission"]["records"]
            )
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


async def _events(work):
    return [event async for event in runtime.agent_invocation({"prompt": work.literal})]


async def captured_originals(work, data, reader):
    deadline = asyncio.get_running_loop().time() + 10
    while not (
        work.observing.is_set()
        and len(data.native.calls) == 2
        and len(work.engines[0].tool_backend._warm_pending_producers) == 2
    ):
        assert not reader.done() and asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.005)
    backend = work.engines[0].tool_backend
    assert len(backend._acquisition_store._records) == 2
    assert backend._warm_pending is not None and len(work.agents) == len(work.engines) == 1
    return backend


@pytest.mark.parametrize("mode", ["missing-unit", "mutated-unit-set", "revoked-binding"])
def test_precommit_requires_exact_completed_current_source_research(monkeypatch, mode):
    model = ThreeOriginals(research_count=2 if mode == "missing-unit" else 3)
    work, data, _fetched = protocol_work(monkeypatch, model)
    original_build = ToolBindingBackend.build_agent

    def build(backend, *args, **kwargs):
        agent = original_build(backend, *args, **kwargs)

        def before_model(_event):
            if model.calls == 3:
                if mode == "mutated-unit-set":
                    backend._research_binding["units"].pop("training-card")
                elif mode == "revoked-binding":
                    backend.approval_policy.allowlist.clear()

        agent.hooks.add_callback(BeforeModelCallEvent, before_model)
        return agent

    monkeypatch.setattr(ToolBindingBackend, "build_agent", build)

    async def scenario():
        reader = asyncio.create_task(_events(work))
        try:
            if mode == "missing-unit":
                backend = await captured_originals(work, data, reader)
                assert backend._research_binding["phase"] == "CONSUMED"
                assert backend._research_binding["original_completion_ready"] is False
                assert (
                    model.calls == 3
                    and len([result for result in results(work.agents[0]) if result["toolUseId"] in USES])
                    == 3
                )
                backend.approval_policy = None
            terminal = await asyncio.wait_for(reader, 5)
            if mode != "missing-unit":
                assert not work.native.requests and not data.native.calls
                assert terminal[-1]["response"]["data"]["input_admission"]["status"] == "held"
            assert terminal[-1]["worker_finished"] is True
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["denied-sibling", "policy-during-wait"])
def test_precommit_denial_or_revocation_retains_all_originals_without_next_model(monkeypatch, tmp_path, mode):
    def deny(text):
        return mode == "denied-sibling" and "original_data_completed" in text and USES[2] in text

    model = ThreeOriginals()
    work, data, _fetched = protocol_work(monkeypatch, model, deny=deny)

    async def scenario():
        reader = asyncio.create_task(_events(work))
        try:
            backend = await captured_originals(work, data, reader)
            original = backend._warm_pending
            producers = tuple(backend._warm_pending_producers.values())
            before = copy.deepcopy(work.agents[0].messages)
            if mode == "policy-during-wait":
                backend.approval_policy = None
            else:
                complete_fixture(
                    monkeypatch,
                    tmp_path,
                    SimpleNamespace(handle=original.record.handle, adapter=work.adapter, native=work.native),
                )
                for pending in producers:
                    publish_original_data(pending.record, data)
            terminal = await asyncio.wait_for(reader, 5)
            assert model.calls == 4 and len(work.native.requests) == 1 and len(data.native.calls) == 2
            assert backend._input_hold is True and not backend._warm_consumed
            assert backend._warm_pending is original
            assert tuple(backend._warm_pending_producers.values()) == producers
            assert work.agents[0].messages[: len(before)] == before
            assert terminal[-1]["response"]["data"]["input_admission"]["status"] == "held"
            if mode == "denied-sibling":
                assert any(
                    record["outcome"] == "BLOCKED"
                    for record in terminal[-1]["response"]["data"]["input_admission"]["records"]
                )
                assert {
                    value["toolUseId"] for value in results(work.agents[0]) if value["toolUseId"] in USES
                } == set(USES)
            assert terminal[-1]["worker_finished"] is True
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


def test_precommit_cancellation_drains_exact_native_read_before_worker_finishes(monkeypatch):
    model = ThreeOriginals()
    work, data, _fetched = protocol_work(monkeypatch, model)
    entered, release = threading.Event(), threading.Event()
    actual_describe = work.native.describe_processing_job

    def describe(**kwargs):
        entered.set()
        assert release.wait(5)
        return actual_describe(**kwargs)

    monkeypatch.setattr(work.native, "describe_processing_job", describe)
    invocation = {}
    original_build = ToolBindingBackend.build_agent

    def build(backend, *args, **kwargs):
        agent = original_build(backend, *args, **kwargs)

        def capture(_event):
            invocation["loop"], invocation["task"] = asyncio.get_running_loop(), asyncio.current_task()

        agent.hooks.add_callback(BeforeInvocationEvent, capture)
        return agent

    monkeypatch.setattr(ToolBindingBackend, "build_agent", build)

    async def scenario():
        reader = asyncio.create_task(_events(work))
        try:
            deadline = asyncio.get_running_loop().time() + 10
            while not (
                entered.is_set()
                and len(data.native.calls) == 2
                and len(work.engines[0].tool_backend._warm_pending_producers) == 2
            ):
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            backend, agent = work.engines[0].tool_backend, work.agents[0]
            before = copy.deepcopy(agent.messages)
            invocation["loop"].call_soon_threadsafe(invocation["task"].cancel)
            await asyncio.sleep(0.02)
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            assert not reader.done() and model.calls == 4 and not backend._warm_consumed
            release.set()
            deadline = asyncio.get_running_loop().time() + 5
            while work.owned_app.get_async_task_info()["active_count"]:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            await asyncio.wait_for(asyncio.gather(reader, return_exceptions=True), 5)
            assert backend._warm_closed == "refused" and not backend._warm_consumed
            assert agent.messages == before and model.calls == 4
            assert len(work.native.requests) == 1 and len(data.native.calls) == 2
            assert backend._warm_pending is not None and len(backend._warm_pending_producers) == 2
        finally:
            release.set()
            await close_original_worker(work, reader)

    asyncio.run(scenario())


class OneCaseOriginal(ThreeOriginals):
    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls < 3:
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
            return
        self.calls += 1
        self.messages_seen.append(copy.deepcopy(messages))
        assert self.calls == 4, "closed original CASE entered another model cycle"
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "original-case-use", "name": "freeze_coding_cases"}},
            }
        }
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": json.dumps({"acquisition_request_ref": "acq_" + "a" * 32})}},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }


def test_actual_case_store_closed_rejection_never_enters_uncertainty_poll_budget(request, monkeypatch):
    adapter, native, s3, original_source, catalog = request.getfixturevalue("_case_native")
    model = OneCaseOriginal()
    work, _data, _fetched = protocol_work(monkeypatch, model)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","coding_case_freeze_flow"]')
    ports, closed_reads = SimpleNamespace(), []

    def make_case(acquired, **kwargs):
        ports.source = move_source_port(original_source, s3, acquired.runtime_session_id)

        def upstream(reference):
            assert reference == "acq_" + "a" * 32
            return ports.source

        monkeypatch.setattr(acquired, "_verified_preparation_source", upstream)
        store = SessionCodingCaseFreezeStore(acquired, adapter, catalog)
        actual_observe = store.observe

        def observe(reference):
            value = actual_observe(reference)
            if is_closed_rejection(value.get("verification_rejection")):
                closed_reads.append(value)
            return value

        monkeypatch.setattr(store, "observe", observe)
        ports.store = store
        return store

    monkeypatch.setattr(binding, "build_case_freeze_store", make_case)

    async def scenario():
        reader = asyncio.create_task(_events(work))
        try:
            await wait_until(
                lambda: bool(work.engines) and bool(work.engines[0].tool_backend._warm_pending_producers),
                reader,
                "actual original CASE custody missing",
            )
            backend = work.engines[0].tool_backend
            pending = next(iter(backend._warm_pending_producers.values()))
            assert pending.kind == "case" and pending.store is ports.store and pending.agent is work.agents[0]
            assert backend._research_binding["original_completion_ready"] is True and model.calls == 4
            assert (
                len(native.creates) == 1
                and not results(work.agents[0])[-1]["toolUseId"] == pending.tool_use_id
            )
            bundle = freeze_managed_case_bundle(
                rows(), source=ports.source, header=pending.record.extra["header"]
            )
            prefix = pending.record.extra["output_prefix"]
            for name, raw in bundle.items():
                s3.add(prefix + name, raw)
            key = prefix + "ORACLES.jsonl"
            raw, version = s3.values[key]
            changed = raw.replace(b"return 0", b"return 9", 1)
            assert changed != raw and len(changed) == len(raw)
            s3.values[key] = changed, version
            native.state = "Completed"
            terminal = await asyncio.wait_for(reader, 5)
            assert closed_reads and all(
                value["status"] == "verification_failed" and value["success"] is False
                for value in closed_reads
            )
            assert len(closed_reads) == 1, (
                "definitive original CASE artifact rejection was polled as uncertainty"
            )
            assert backend._producer_key(pending) not in backend._warm_case_failure_counts
            assert backend._input_hold is True and not backend._warm_consumed and model.calls == 4
            assert backend._warm_pending_producers[backend._producer_key(pending)] is pending
            assert len(native.creates) == 1 and terminal[-1]["worker_finished"] is True
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
