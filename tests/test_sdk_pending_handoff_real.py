"""Actual SDK concurrent PREP/CASE siblings; inert I/O cannot prove live ML."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
from case_prep_continuation_fixtures import move_source_port, wait_until
from strands.tools.executors import ConcurrentToolExecutor
from test_input_admission_wiring import ClientFixture, configure_guard
from test_managed_coding_case_envelope import rows
from test_managed_commitpack_preparation import staged_fixture
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_warm_model_continuation import OriginalModelStream, complete_fixture

from amlas import app as runtime
from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
from amlas.control_plane.coding_case_state import SessionCodingCaseFreezeStore
from amlas.control_plane.preparation_state import SessionPreparationStore
from amlas.data_eng.coding_case_managed import freeze_managed_case_bundle
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("case_prep_continuation_fixtures", "test_runtime_prompt_entry")
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
CASE_SOURCE = "acq_" + "a" * 32
TRAIN_SOURCE = "acq_" + "b" * 32


class CaseThenConcurrentPreparation(OriginalModelStream):
    def __init__(self, *, canonical_case_ref=False):
        super().__init__()
        self.canonical_case_ref = canonical_case_ref
        self.case_completion = None
        self.case_ref = None
        self.batch_selected = threading.Event()

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.messages_seen.append(copy.deepcopy(messages))
        requests = []
        if self.calls == 1:
            requests = [
                (
                    "model-original",
                    "acquire_model_weights",
                    {"catalog_id": "qwen3_5_9b", "catalog_version": "1"},
                )
            ]
        else:
            completion = json.loads(messages[-1]["content"][0]["text"])
            if self.calls == 2:
                assert completion["event_type"] == "original_model_acquisition_completed"
                assert "case_ref" not in completion
                requests = [
                    ("case-original", "freeze_coding_cases", {"acquisition_request_ref": CASE_SOURCE})
                ]
            elif self.calls == 3:
                assert completion["event_type"] == "original_case_completed"
                self.case_completion = copy.deepcopy(completion)
                self.case_ref = (
                    completion["case_ref"] if self.canonical_case_ref else completion["request_ref"]
                )
                requests = [
                    (
                        "preparation-original",
                        "prepare_coding_dataset",
                        {"acquisition_request_refs": [TRAIN_SOURCE], "case_request_ref": self.case_ref},
                    ),
                    ("case-getter-sibling", "get_coding_case_freeze", {"case_ref": self.case_ref}),
                ]
                self.batch_selected.set()
            else:
                assert self.calls == 4 and completion["event_type"] == "original_preparation_completed"
        yield {"messageStart": {"role": "assistant"}}
        if requests:
            for index, (identifier, name, arguments) in enumerate(requests):
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
                    "delta": {"text": "Original preparation completion received."},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}


@pytest.mark.parametrize(
    "case_specific_only", [False, True], ids=["startup-exclusions", "case-exclusions-only"]
)
def test_actual_concurrent_sdk_case_resume_commits_preparation_and_getter_then_pauses(
    request, monkeypatch, tmp_path, case_specific_only
):
    # Reuse both inert owning ports, with one explicit synthetic cohort. No private
    # original record is injected: registered SDK tools create all target records.
    _old_adapter, prep_native, prep_s3, prep_catalog, training = staged_fixture(
        request.getfixturevalue("_preparation_native"), monkeypatch
    )
    case_adapter, case_native, case_s3, original_case, case_catalog = request.getfixturevalue("_case_native")
    for key, (raw, version) in prep_s3.values.items():
        case_s3.add(key, raw, version)
    origin = training.inspection_source
    training = dataclasses.replace(
        training,
        acquisition_image_uri=case_catalog.image_uri,
        inspection_source=dataclasses.replace(
            origin,
            account_id=case_catalog.account_id,
            bucket=case_catalog.bucket,
            deployment_id=case_catalog.deployment_id,
            deployment_prefix=case_catalog.deployment_prefix,
            acquisition_job_arn="arn:aws:sagemaker:us-west-2:"
            + case_catalog.account_id
            + ":processing-job/owned-training",
            artifact_ref=dataclasses.replace(
                origin.artifact_ref,
                uri=origin.artifact_ref.uri.replace(prep_catalog.bucket, case_catalog.bucket),
            ),
        ),
        manifest_refs=tuple(
            (name, uri.replace(prep_catalog.bucket, case_catalog.bucket), version, digest)
            for name, uri, version, digest in training.manifest_refs
        ),
    )
    mapped_manifests = []
    for name, uri, version, digest in training.manifest_refs:
        if name == "receipt":
            key = uri.split("/", 3)[3]
            receipt = json.loads(case_s3.values[key][0])
            receipt["image_uri"] = training.acquisition_image_uri
            raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
            case_s3.add(key, raw, version)
            digest = hashlib.sha256(raw).hexdigest()
        mapped_manifests.append((name, uri, version, digest))
    training = dataclasses.replace(training, manifest_refs=tuple(mapped_manifests))
    selected_catalog = dataclasses.replace(
        prep_catalog,
        account_id=case_catalog.account_id,
        bucket=case_catalog.bucket,
        execution_role_arn=case_catalog.execution_role_arn,
        image_uri=case_catalog.image_uri,
        deployment_id=case_catalog.deployment_id,
        deployment_prefix=case_catalog.deployment_prefix,
        exclusions_ref=dataclasses.replace(
            prep_catalog.exclusions_ref,
            uri=prep_catalog.exclusions_ref.uri.replace(prep_catalog.bucket, case_catalog.bucket),
        ),
    )
    if case_specific_only:
        selected_catalog = dataclasses.replace(
            selected_catalog,
            exclusions_ref=dataclasses.replace(
                selected_catalog.exclusions_ref, uri="", version_id="", sha256="", byte_count=0
            ),
        )
    prep_adapter = ProcessingCodingPreparationBackend(sagemaker_client=prep_native, s3_client=case_s3)
    submitted_catalogs = []
    original_submit = prep_adapter.submit

    def capture_submit(*, sources, catalog, runtime_session_id):
        submitted_catalogs.append(catalog)
        return original_submit(sources=sources, catalog=catalog, runtime_session_id=runtime_session_id)

    monkeypatch.setattr(prep_adapter, "submit", capture_submit)
    model = CaseThenConcurrentPreparation(canonical_case_ref=case_specific_only)
    configure_guard(monkeypatch, ClientFixture(deny=lambda _text: False))
    monkeypatch.setenv("AMLAS_RUNTIME_RESEARCH_PROFILE", "qwen9b_core_v1")
    work = connected_work(monkeypatch, model=model)
    monkeypatch.setenv(
        "AMLAS_RUNTIME_APPROVED_ACTIONS",
        '["run_processing_job","coding_case_freeze_flow","data_preparation_flow"]',
    )
    ports = SimpleNamespace(case_source=None, training=None)

    def make_case(acquired, **kwargs):
        ports.case_source = move_source_port(original_case, case_s3, acquired.runtime_session_id)
        ports.training = move_source_port(training, case_s3, acquired.runtime_session_id)

        def upstream(reference):
            assert reference in (CASE_SOURCE, TRAIN_SOURCE)
            return ports.case_source if reference == CASE_SOURCE else ports.training

        monkeypatch.setattr(acquired, "_verified_preparation_source", upstream)
        return SessionCodingCaseFreezeStore(acquired, case_adapter, case_catalog)

    def make_preparation(acquired, *, case_store=None, **kwargs):
        return SessionPreparationStore(acquired, prep_adapter, selected_catalog, case_store=case_store)

    monkeypatch.setattr(binding, "build_case_freeze_store", make_case)
    monkeypatch.setattr(binding, "build_preparation_store", make_preparation)
    siblings = threading.Barrier(2, timeout=3)
    original_artifacts = case_adapter.artifacts
    arrivals = []
    lock = threading.Lock()

    def concurrent_artifacts(handle):
        if model.batch_selected.is_set():
            with lock:
                arrivals.append(threading.get_ident())
                participate = len(arrivals) <= 2
            if participate:
                siblings.wait()
        return original_artifacts(handle)

    monkeypatch.setattr(case_adapter, "artifacts", concurrent_artifacts)

    async def scenario():
        reader = asyncio.create_task(_events(work))
        try:
            await wait_until(work.observing.is_set, reader, "original MODEL observation missing")
            backend = work.engines[0].tool_backend
            assert type(work.agents[0].tool_executor) is ConcurrentToolExecutor
            assert model.calls == 1 and backend._research_binding["phase"] == "CONSUMED"
            pending = backend._warm_pending
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=pending.record.handle, adapter=work.adapter, native=work.native),
            )
            await wait_until(
                lambda: bool(backend._warm_pending_producers), reader, "original CASE capture missing"
            )
            case_pending = next(iter(backend._warm_pending_producers.values()))
            assert case_pending.kind == "case" and model.calls == 2
            case_receipts = [value for use, value in work.originals if use == "case-original"]
            assert len(case_receipts) == 1 and case_receipts[0]["case_ref"] == case_pending.reference
            before = copy.deepcopy(work.agents[0].messages)
            bundle = freeze_managed_case_bundle(
                rows(), source=ports.case_source, header=case_pending.record.extra["header"]
            )
            for name, raw in bundle.items():
                case_s3.add(case_pending.record.extra["output_prefix"] + name, raw)
            case_native.state = "Completed"
            if case_specific_only:
                await wait_until(
                    lambda: model.case_completion is not None, reader, "original CASE completion missing"
                )
                assert model.case_completion is not None, "original CASE completion missing"
                assert model.case_completion["request_ref"] == case_pending.reference
                assert model.case_completion.get("case_ref") == case_pending.reference, (
                    "verified CASE completion missing canonical case_ref"
                )
            await wait_until(
                lambda: (
                    bool(prep_native.creates)
                    and not any(p.kind == "case" for p in backend._warm_pending_producers.values())
                ),
                reader,
                "concurrent PREP/CASE batch did not settle",
            )
            assert not reader.done() and model.calls == 3, {
                "calls": model.calls,
                "native_creates": len(prep_native.creates),
                "results": [
                    (use, value.get("status"), value.get("error_type"), value.get("detail"))
                    for use, value in work.originals
                ],
            }
            assert work.agents[0].messages[: len(before)] == before
            results = [
                block["toolResult"]
                for message in work.agents[0].messages
                for block in message["content"]
                if "toolResult" in block
            ]
            selected = [
                result
                for result in results
                if result["toolUseId"] in {"preparation-original", "case-getter-sibling"}
            ]
            assert len(selected) == 2 and all(result["status"] == "success" for result in selected)
            assert len(set(arrivals[:2])) == 2, "actual SDK sibling I/O did not overlap"
            assert len(case_native.creates) == len(prep_native.creates) == len(work.native.requests) == 1
            if case_specific_only:
                frozen = case_pending.store._verified_preparation_exclusions(case_pending.reference)
                assert len(submitted_catalogs) == 1
                assert selected_catalog.exclusions_ref.byte_count == 0
                assert submitted_catalogs[0].exclusions_ref == frozen.exclusions_ref
            assert work.owned_app.get_async_task_info()["active_count"] == 1
            prep_pending = next(iter(backend._warm_pending_producers.values()))
            assert prep_pending.kind == "preparation" and prep_pending.agent is work.agents[0]
            # The batch has yielded to the ordinary original-handle waiter. End
            # this inert I/O test through its existing current-policy closure,
            # retaining the still-running preparation and its unchanged history.
            settled = copy.deepcopy(work.agents[0].messages)
            backend.approval_policy = None
            received = await asyncio.wait_for(reader, 5)
            assert received[-1]["worker_finished"] is True
            assert received[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "refused"
            assert model.calls == 3 and len(work.agents) == len(work.engines) == 1
            assert work.agents[0].messages == settled and len(prep_native.creates) == 1
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


async def _events(work):
    return [event async for event in runtime.agent_invocation({"prompt": work.literal})]
