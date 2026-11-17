"""Real registered Pair submit retains its actual typed original; upstream ports are fixture DATA."""

from __future__ import annotations

import asyncio
import copy
import dataclasses

import pytest
from multi_producer_fixtures import SequentialProducerStream
from test_coding_arbiter import SOURCE_SHA256 as DECLARED_CASE_SOURCE_SHA256
from test_coding_case_native import native as case_fixture  # noqa: F401
from test_paired_coding_backend import packet as pair_fixture  # noqa: F401
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_warm_model_continuation import complete_fixture

from amlas import app as runtime
from amlas.arbiter.coding import CodingCasePin, CodingEvaluatorFactory
from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend
from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
from amlas.control_plane.coding_case_state import SessionCodingCaseFreezeStore
from amlas.control_plane.generation_state import SessionCodingGenerationStore
from amlas.control_plane.paired_state import PairedCodingCatalog, SessionPairedCodingStore
from amlas.control_plane.preparation_state import PreparationCatalog, SessionPreparationStore
from amlas.control_plane.sft_state import SessionSftStore
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


class PairStream(SequentialProducerStream):
    @staticmethod
    def data_calls():
        return [("original-pair-use", "submit_coding_pair", {
            "case_request_ref": "case_" + "a" * 32,
            "base_generation_ref": "gen_" + "b" * 32,
            "tuned_generation_ref": "gen_" + "c" * 32,
            "resolved_spec_ref": "spec_" + "d" * 32})]


@pytest.mark.parametrize("complete", [False, True], ids=["capture-only", "full-328-result-verifier"])
def test_normal_sdk_pair_retains_actual_typed_native_original_after_model_resume(monkeypatch, tmp_path, pair_fixture, case_fixture, complete):  # noqa: F811 - registered pytest fixture alias
    adapter, inputs, catalog, native, s3, validated = pair_fixture
    case_adapter, _, _, _, case_catalog = case_fixture
    model = PairStream()
    work = connected_work(monkeypatch, model=model)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","coding_pair_flow"]')
    monkeypatch.setattr(binding, "build_case_freeze_store",
        lambda acquired: SessionCodingCaseFreezeStore(acquired, case_adapter, case_catalog))

    def preparation_store(acquired, *, case_store=None):
        selected = PreparationCatalog(**dataclasses.asdict(catalog), exclusions_ref=dict(inputs.cases.file_refs)["EXCLUSIONS.json"])
        return SessionPreparationStore(acquired, ProcessingCodingPreparationBackend(), selected, case_store=case_store)
    monkeypatch.setattr(binding, "build_preparation_store", preparation_store)
    monkeypatch.setattr(binding, "build_generation_store", lambda models, cases, sft:
        SessionCodingGenerationStore(models, cases, sft, SageMakerCodingGenerationBackend(), catalog))

    def sft_store(models, prepared):
        assert prepared is not None
        return SessionSftStore(models, prepared, adapter, catalog)

    # Supply only source-selected configuration at normal graph construction.
    # No target Pair record, handle, accepted flag or native identity is injected.
    monkeypatch.setattr(binding, "build_sft_store", sft_store)

    def pair_store(generation, cases):
        session = cases.runtime_session_id
        def source(source):
            refs = []
            for name, original in source.file_refs:
                key = original.uri.split("/", 3)[3]
                raw = s3.objects[(key, original.version_id)]
                target = key.replace("/" + source.runtime_session_id + "/", "/" + session + "/")
                refs.append((name, s3.install(target, raw, original.version_id)))
            return dataclasses.replace(source, runtime_session_id=session, file_refs=tuple(refs))
        case, base, tuned = (source(value) for value in (inputs.cases, inputs.base, inputs.tuned))
        case = dataclasses.replace(case, producer_header=tuple(sorted({
            **dict(case.producer_header), "runtime_session_id":session}.items())))
        # These strict upstream supplier ports are neutral fixture DATA. This
        # control proves target Pair original capture, not native upstream jobs.
        monkeypatch.setattr(generation, "_verified_resolved_spec",
            lambda case_ref, spec_ref: (inputs.spec_bytes, inputs.spec_sha256, case))
        monkeypatch.setattr(generation, "_verified_evaluation_source",
            lambda reference: base if reference == "gen_" + "b" * 32 else tuned)
        return SessionPairedCodingStore(generation, cases, adapter, PairedCodingCatalog(catalog, "d" * 64))

    monkeypatch.setattr(binding, "build_pair_store", pair_store)

    async def scenario():
        async def read():
            return [event async for event in runtime.agent_invocation({"prompt": work.literal})]
        reader = asyncio.create_task(read())
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not work.observing.is_set() and not reader.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            backend = work.engines[0].tool_backend
            original = next(value for identifier, value in work.originals if identifier == "original-model-use")
            assert original["accepted"] is True
            handle = backend._model_acquisition_store._records[original["request_ref"]].handle
            complete_fixture(monkeypatch, tmp_path, type("Original", (), {"handle":handle, "adapter":work.adapter, "native":work.native})())
            deadline = asyncio.get_running_loop().time() + 5
            while not [value for use, value in work.originals if use == "original-pair-use"] and not reader.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            results = [value for use, value in work.originals if use == "original-pair-use"]
            store = backend._pair_store
            bounded = {"native_creates": native.calls.count("create"), "retained":len(store._records),
                "status":results[0].get("status") if len(results)==1 else "missing",
                "error_type":results[0].get("error_type") if len(results)==1 else None}
            assert len(results) == 1 and results[0]["accepted"] is True, bounded
            assert len(store._records) == 1 and native.calls.count("create") == 1
            record = store._records[results[0]["request_ref"]]
            assert (record.id, record.arn) == (results[0]["job_name"], results[0]["job_arn"])
            assert type(record.extra["evaluation_binding"]).__name__ == "CodingEvaluationBinding"
            assert record.extra["evaluation_binding"].source_sha256 == DECLARED_CASE_SOURCE_SHA256
            assert record.extra["runtime_session_id"] == store.runtime_session_id
            assert len(validated) == 1
            # The fixture's result recorder runs before the source's after-tool hook retains the original.
            while not [value for value in backend._warm_pending_producers.values() if value.kind == "pair"]:
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            pending = [value for value in backend._warm_pending_producers.values() if value.kind == "pair"]
            assert len(pending) == 1, "genuine source-built Pair original was lost at its typed pin projection"
            assert pending[0].record is record and pending[0].store is store and pending[0].adapter is adapter
            assert pending[0].agent is work.agents[0] and pending[0].tool_use_id == "original-pair-use"
            assert not reader.done() and work.owned_app.get_async_task_info()["active_count"] == 1
            deadline = asyncio.get_running_loop().time() + 5
            while model.calls < 4:
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert len(work.native.requests) == 1 and model.calls == 4
            if complete:
                from test_paired_coding_runner import population

                from amlas.evaluation.coding_oracle_catalog import EXPECTED_CASES
                from amlas.evaluation.coding_pair import run_paired
                cases, candidates = population()
                cases = tuple({**case, "oracle_sha256":EXPECTED_CASES[index]["oracle_record_sha256"]}
                    for index,case in enumerate(cases))
                candidates = {arm:tuple({**row,"source":None,"source_sha256":None,"outcome":"generation_error"}
                    for row in rows) for arm,rows in candidates.items()}
                class NoExecution:
                    def __getattr__(self, name):
                        raise AssertionError("no unresolved candidate is executed")
                root = tmp_path / "pure-complete-pair"
                root.mkdir()
                result = run_paired(binding=record.extra["evaluation_binding"],cases=cases,candidates=candidates,
                    client=NoExecution(),cleanup_client=NoExecution(),evidence_root=root,
                    clock=lambda:1.0,sleep=lambda seconds:None)
                import hashlib

                from amlas.training.coding_generation import canonical_bytes
                assert len(result["receipts"]) == 328 and result["cleanup_complete"] is True
                pins = tuple(CodingCasePin(case_id=row["case_id"],oracle_sha256=row["oracle_record_sha256"]) for row in EXPECTED_CASES)
                report = CodingEvaluatorFactory().build_report(binding=record.extra["evaluation_binding"],case_pins=pins,receipts=result["receipts"])
                receipts = b"".join(canonical_bytes(row.model_dump(mode="json")) + b"\n" for row in result["receipts"])
                trace_raw = b"\n"
                trace = {"sha256":hashlib.sha256(trace_raw).hexdigest(),"byte_count":1,"format":"jsonl","record_count":0}
                evaluation = {"schema_version":1,"operation":"paired_eval","source_commit":catalog.source_commit,
                    "binding":record.extra["evaluation_binding"].model_dump(mode="json"),"case_count":164,"receipt_count":328,
                    "quality_success":False,"native_admission":False,"cleanup_complete":True,"summary":report.describe(),"invocations_manifest":trace}
                evaluation_raw = canonical_bytes(evaluation) + b"\n"
                payloads = {"RECEIPTS.jsonl":receipts,"EVALUATION.json":evaluation_raw,"INVOCATIONS_MANIFEST.jsonl":trace_raw}
                files = {name:{"sha256":hashlib.sha256(raw).hexdigest(),"byte_count":len(raw),"format":"jsonl" if name.endswith(".jsonl") else "json"}
                    for name,raw in payloads.items() if name != "INVOCATIONS_MANIFEST.jsonl"}
                commit = {"schema_version":1,"operation":"paired_eval","source_commit":catalog.source_commit,"case_count":164,
                    "quality_success":False,"native_admission":False,"files":files,"invocations_manifest":trace}
                payloads["COMMITTED.json"] = canonical_bytes(commit) + b"\n"
                before = copy.deepcopy(work.agents[0].messages)
                for name,raw in payloads.items():
                    s3.install(record.extra["output_prefix"] + name,raw)
                native.state = "Completed"
                events = await asyncio.wait_for(reader,5)
                assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"] == "continued_original"
                assert work.agents[0].messages[:len(before)] == before
                assert native.calls.count("create") == 1 and model.calls == 5
                assert backend._warm_pending_producers == {} and work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())
