"""Source-only upstream ports; CASE/PREP records and results come from real producers."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import threading
from types import SimpleNamespace

from multi_producer_fixtures import SequentialProducerStream
from test_coding_case_native import native as _case_native  # noqa: F401
from test_managed_coding_case_envelope import rows
from test_managed_coding_preparation import complete_outputs
from test_managed_coding_preparation import fixture as _preparation_native  # noqa: F401
from test_runtime_connected_warm_model import connected_work
from test_runtime_warm_model_continuation import complete_fixture

from amlas import app as runtime
from amlas.control_plane.coding_case_state import SessionCodingCaseFreezeStore
from amlas.control_plane.preparation_state import SessionPreparationStore
from amlas.data_eng.coding_case_managed import freeze_managed_case_bundle
from amlas.tui.chat import tool_binding_backend as binding

UPSTREAM_SOURCE_REF = "acq_" + "a" * 32
TARGET_TOOL_USE = "original-case-prep-use"


class CasePrepStream(SequentialProducerStream):
    """Ordinary SDK tools produce originals; fixture never injects target records."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def data_calls(self):
        if self.kind == "case":
            return [
                (TARGET_TOOL_USE, "freeze_coding_cases", {"acquisition_request_ref": UPSTREAM_SOURCE_REF})
            ]
        return [
            (TARGET_TOOL_USE, "prepare_coding_dataset", {"acquisition_request_refs": [UPSTREAM_SOURCE_REF]})
        ]


def move_source_port(original, s3, runtime_session_id):
    """Relink only the synthetic upstream supplier bytes to the actual private graph."""
    origin = original.inspection_source

    def move(uri, version):
        key = uri.split("/", 3)[3]
        raw, recorded_version = s3.values[key]
        assert recorded_version == version
        target = key.replace("/" + origin.runtime_session_id + "/", "/" + runtime_session_id + "/")
        s3.add(target, raw, version)
        return "s3://" + origin.bucket + "/" + target

    artifact = origin.artifact_ref
    artifact = dataclasses.replace(artifact, uri=move(artifact.uri, artifact.version_id))
    manifests = tuple(
        (name, move(uri, version), version, digest) for name, uri, version, digest in original.manifest_refs
    )
    return dataclasses.replace(
        original,
        inspection_source=dataclasses.replace(
            origin, runtime_session_id=runtime_session_id, artifact_ref=artifact
        ),
        manifest_refs=manifests,
    )


def stage_work(request, monkeypatch, kind):
    # Load only the selected owning fixture: their profile monkeypatches must not mix.
    selected = request.getfixturevalue("_case_native" if kind == "case" else "_preparation_native")
    if kind == "case":
        adapter, native, s3, original, catalog = selected
    else:
        adapter, native, s3, catalog, original = selected
    model = CasePrepStream(kind)
    work = connected_work(monkeypatch, model=model)
    monkeypatch.setenv(
        "AMLAS_RUNTIME_APPROVED_ACTIONS",
        '["run_processing_job","coding_case_freeze_flow","data_preparation_flow"]',
    )
    target = SimpleNamespace(
        kind=kind,
        selected=selected,
        adapter=adapter,
        native=native,
        s3=s3,
        unknown=False,
        observations=[],
        actual_observed=[],
        read_boundary=threading.Event(),
        store=None,
        source=None,
    )
    original_describe = native.describe_processing_job

    def describe_original(**kwargs):
        target.observations.append(native.state)
        target.read_boundary.set()
        if target.unknown:
            raise RuntimeError("ordinary unavailable original target observation")
        return original_describe(**kwargs)

    monkeypatch.setattr(native, "describe_processing_job", describe_original)

    def make_store(acquired, **kwargs):
        source = move_source_port(original, s3, acquired.runtime_session_id)

        def verified_upstream_source(reference):
            assert reference == UPSTREAM_SOURCE_REF
            return source

        # This is only the existing upstream verified-artifact supplier port.
        # Target store submit/observe/adapter/artifact verification are never fabricated.
        monkeypatch.setattr(acquired, "_verified_preparation_source", verified_upstream_source)
        store = (
            SessionCodingCaseFreezeStore(acquired, adapter, catalog)
            if kind == "case"
            else SessionPreparationStore(acquired, adapter, catalog)
        )
        target.store, target.source = store, source
        actual_observe = store.observe

        def observe_original(reference):
            observed = actual_observe(reference)
            target.actual_observed.append((reference, copy.deepcopy(observed)))
            return observed

        monkeypatch.setattr(store, "observe", observe_original)
        return store

    monkeypatch.setattr(
        binding, "build_case_freeze_store" if kind == "case" else "build_preparation_store", make_store
    )
    work.target = target
    return work


async def collect_original(work):
    return [event async for event in runtime.agent_invocation({"prompt": work.literal})]


async def wait_until(predicate, reader, description):
    deadline = asyncio.get_running_loop().time() + 5
    while not predicate() and not reader.done():
        assert asyncio.get_running_loop().time() < deadline, description
        await asyncio.sleep(0.005)


async def start_original_target(work, monkeypatch, tmp_path):
    reader = asyncio.create_task(collect_original(work))
    work.reader = reader
    await wait_until(
        lambda: work.observing.is_set(), reader, "normal original model did not reach observation"
    )
    assert work.originals and work.originals[0][0] == "original-model-use"
    backend = work.engines[0].tool_backend
    model_result = work.originals[0][1]
    model_handle = backend._model_acquisition_store._records[model_result["request_ref"]].handle
    complete_fixture(
        monkeypatch, tmp_path, SimpleNamespace(handle=model_handle, adapter=work.adapter, native=work.native)
    )
    await wait_until(
        lambda: any(use == TARGET_TOOL_USE for use, _ in work.originals),
        reader,
        "normal registered target tool did not return its original submission",
    )
    matches = [value for use, value in work.originals if use == TARGET_TOOL_USE]
    assert len(matches) == 1, "actual registered target submission must be present before continuation checks"
    # The external AfterToolCall recorder may run before later SDK capture hooks.
    # The next model stream proves the initial tool-result turn/hook chain completed.
    await wait_until(lambda: work.model.calls >= 4, reader, "target initial SDK turn did not finish")
    assert work.model.calls == 4, {"calls": work.model.calls, "reader_done": reader.done()}
    # The stream stand-in counts a call when its stream starts; the SDK appends that call's
    # answer afterwards. The source reads the target original only after that turn has returned,
    # so its first read is the passive boundary where history and capture are settled.
    await wait_until(
        lambda: bool(work.target.actual_observed), reader, "original target observer boundary missing"
    )
    result = matches[0]
    assert result.get("accepted") is True, {
        "status": result.get("status"),
        "error_type": result.get("error_type"),
    }
    reference = result["case_ref" if work.target.kind == "case" else "request_ref"]
    record = work.target.store._records[reference]
    handle = record if work.target.kind == "case" else record.handle
    assert len(work.target.native.creates) == 1 and len(work.native.requests) == 1
    assert (handle.id, handle.arn) == (result["job_name"], result["job_arn"])
    return reader, backend, result, reference, record, handle


def publish_original_outputs(work, handle):
    """Write ordinary synthetic metadata/data bytes; never execute model or oracle code."""
    target = work.target
    if target.kind == "case":
        bundle = freeze_managed_case_bundle(rows(), source=target.source, header=handle.extra["header"])
        for name, raw in bundle.items():
            target.s3.add(handle.extra["output_prefix"] + name, raw)
        target.native.state = "Completed"
    else:
        complete_outputs(target.selected, handle)
