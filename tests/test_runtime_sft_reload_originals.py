"""Actual registered full training and distinct reload originals; upstream suppliers are fixture DATA."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from multi_producer_fixtures import SequentialProducerStream
from test_qwen_sft_backend import backend as sft_fixture
from test_qwen_sft_backend import catalog, described_resource_config, encoded, install_output
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_warm_model_continuation import complete_fixture

from amlas import app as runtime
from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.preparation_state import PreparationCatalog, SessionPreparationStore
from amlas.control_plane.sft_state import SessionSftStore
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


class TrainReloadStream(SequentialProducerStream):
    def __init__(self):
        super().__init__()
        self.train_ref = None
    def data_calls(self):
        return [("original-train-use", "submit_qwen_sft", {"model_request_ref":"model_acq_" + "a" * 32,
            "preparation_request_ref":"prep_" + "b" * 32, "recipe_id":"full_v1"})]
    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls == 4:
            self.calls += 1
            self.messages_seen.append(copy.deepcopy(messages))
            event = json.loads(messages[-1]["content"][0]["text"])
            assert event["event_type"] == "original_sft_completed"
            self.train_ref = event["request_ref"]
            yield {"messageStart":{"role":"assistant"}}
            yield {"contentBlockStart":{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"original-reload-use","name":"reload_qwen_sft"}}}}
            yield {"contentBlockDelta":{"contentBlockIndex":0,"delta":{"toolUse":{"input":json.dumps({"request_ref":self.train_ref})}}}}
            yield {"contentBlockStop":{"contentBlockIndex":0}}
            yield {"messageStop":{"stopReason":"tool_use"}}
        else:
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event


def current_inputs(inputs, s3, session):
    def artifact(ref):
        key = ref.uri.split("/",3)[3]
        _old, raw = s3.objects[key]
        target = key.replace("/" + inputs.model.runtime_session_id + "/", "/" + session + "/")
        value = dataclasses.replace(ref, uri="s3://" + catalog().bucket + "/" + target)
        s3.install(value, raw)
        return value
    model = dataclasses.replace(inputs.model, runtime_session_id=session,
        producer_bindings=tuple(sorted({**dict(inputs.model.producer_bindings), "runtime_session_id":session}.items())),
        file_refs=tuple((name, artifact(ref)) for name, ref in inputs.model.file_refs),
        manifest_refs=tuple((name, artifact(ref)) for name, ref in inputs.model.manifest_refs))
    original_commit = inputs.prepared.preparation_ref
    committed = json.loads(s3.objects[original_commit.uri.split("/",3)[3]][1])
    committed["runtime_session_id"] = session
    committed["counts"].update(source=1250, normalized=1250, included_origins=1250,
        retained_representatives=1250, split_counts={"train":1000,"validation":125,"test":125})
    for name in ("membership.jsonl","lineage.jsonl"):
        committed["files"][name]["record_count"] = 1250
    for split, count in committed["counts"]["split_counts"].items():
        committed["files"][split + ".jsonl"]["record_count"] = count
    commit_raw = encoded(committed)
    current_commit = artifact(original_commit)
    current_commit = dataclasses.replace(current_commit, sha256=hashlib.sha256(commit_raw).hexdigest(), byte_count=len(commit_raw))
    s3.install(current_commit, commit_raw)
    prepared = dataclasses.replace(inputs.prepared, runtime_session_id=session,
        source_row_counts=tuple((sha,1250) for sha,_count in inputs.prepared.source_row_counts),
        header=tuple(sorted({**dict(inputs.prepared.header), "runtime_session_id":session}.items())),
        preparation_ref=current_commit, exclusions_ref=artifact(inputs.prepared.exclusions_ref),
        split_refs=tuple((name, artifact(ref)) for name,ref in inputs.prepared.split_refs),
        sidecar_refs=tuple((name, artifact(ref)) for name,ref in inputs.prepared.sidecar_refs))
    return dataclasses.replace(inputs, model=model, prepared=prepared)


def publish_reload(handle, s3):
    prefix = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"].split("/",3)[3] + "/model/"
    proof = {"res_uid":[10001]*3,"res_gid":[10001]*3,"groups":[],"cap_eff":0,"cap_prm":0,"cap_inh":0,"cap_amb":0,"cap_bnd":0,"no_new_privs":1,"keep_caps":0,"seccomp":2,"fs_uid":10001,"fs_gid":10001,"securebits":0}
    raw = encoded({"schema_version":1,"operation":"reload","source_commit":catalog().source_commit,
        "catalog_id":"qwen3_5_9b","catalog_version":"1","recipe_id":"full_v1",
        "checkpoint_sha256":handle.extra["staged_refs"]["adapter/CHECKPOINT.json"].sha256,
        "fresh_process_greedy_equal":True,"quality_success":False,"privilege_proof":proof})
    def put(name, raw):
        ref = DatasetArtifactRef("s3://" + catalog().bucket + "/" + prefix + name,"reload-" + name,hashlib.sha256(raw).hexdigest(),len(raw),"json")
        s3.install(ref,raw)
        return ref
    ref = put("RELOAD.json",raw)
    put("COMMITTED.json",encoded({"schema_version":1,"operation":"reload","source_commit":catalog().source_commit,
        "files":{"RELOAD.json":{"sha256":ref.sha256,"byte_count":ref.byte_count}},"quality_success":False}))


def test_normal_sdk_full_train_then_distinct_reload_both_continue_same_original_graph(monkeypatch, tmp_path):
    adapter, native, s3, inputs = sft_fixture()
    requests, states, source_failures = {}, {}, []
    actual_submit = adapter.submit
    def submit_original(**kwargs):
        try:
            return actual_submit(**kwargs)
        except Exception as error:
            import traceback
            frames = traceback.extract_tb(error.__traceback__)
            frame = frames[-1]
            source_failures.append((type(error).__name__,frame.name,frame.lineno))
            raise
    monkeypatch.setattr(adapter,"submit",submit_original)
    actual_create = native.create_training_job
    def create(**kwargs):
        result = actual_create(**kwargs)
        requests[kwargs["TrainingJobName"]] = kwargs
        states[kwargs["TrainingJobName"]] = "InProgress"
        return result
    def describe(**kwargs):
        name = kwargs["TrainingJobName"]
        request = requests[name]
        return {**{k:v for k,v in request.items() if k != "Tags"},"ResourceConfig":described_resource_config(request["ResourceConfig"]),"TrainingJobArn":"arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":training-job/" + name,
            "TrainingJobStatus":states[name],"ModelArtifacts":{"S3ModelArtifacts":request["OutputDataConfig"]["S3OutputPath"] + "/model/"}}
    monkeypatch.setattr(native,"create_training_job",create)
    monkeypatch.setattr(native,"describe_training_job",describe)
    monkeypatch.setattr(native,"list_tags",lambda **kwargs:{"Tags":requests[kwargs["ResourceArn"].split("/")[-1]]["Tags"]})
    model = TrainReloadStream()
    work = connected_work(monkeypatch,model=model)
    monkeypatch.setenv("AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","qwen_sft_flow"]')
    def prepare(acquired, **kwargs):
        selected = PreparationCatalog(**dataclasses.asdict(catalog()),exclusions_ref=inputs.prepared.exclusions_ref)
        return SessionPreparationStore(acquired,ProcessingCodingPreparationBackend(),selected)
    monkeypatch.setattr(binding,"build_preparation_store",prepare)
    def make_sft(models, prepared):
        selected = current_inputs(inputs,s3,models.runtime_session_id)
        monkeypatch.setattr(models,"_verified_sft_model_snapshot",lambda reference:selected.model)
        monkeypatch.setattr(prepared,"_verified_sft_snapshot",lambda reference:selected.prepared)
        return SessionSftStore(models,prepared,adapter,catalog())
    monkeypatch.setattr(binding,"build_sft_store",make_sft)
    async def scenario():
        reader = asyncio.create_task(collect(work))
        try:
            deadline = asyncio.get_running_loop().time()+5
            while not work.observing.is_set() and not reader.done():
                assert asyncio.get_running_loop().time()<deadline
                await asyncio.sleep(0.005)
            backend = work.engines[0].tool_backend
            result = next(value for use,value in work.originals if use=="original-model-use")
            handle = backend._model_acquisition_store._records[result["request_ref"]].handle
            complete_fixture(monkeypatch,tmp_path,SimpleNamespace(handle=handle,adapter=work.adapter,native=work.native))
            deadline = asyncio.get_running_loop().time()+5
            while not [value for use,value in work.originals if use=="original-train-use"] and not reader.done():
                assert asyncio.get_running_loop().time()<deadline
                await asyncio.sleep(0.005)
            train = next(value for use,value in work.originals if use=="original-train-use")
            assert train["accepted"] is True, {"status":train.get("status"),"error_type":train.get("error_type"),"native_creates":len(native.calls),"source_failure":source_failures}
            store = backend._sft_store
            original = store._records[train["request_ref"]]
            assert len(native.calls)==1 and original.extra["operation"]=="train" and original.extra["recipe_id"]=="full_v1"
            deadline=asyncio.get_running_loop().time()+5
            while not any(value.kind=="sft" for value in backend._warm_pending_producers.values()):
                assert not reader.done() and asyncio.get_running_loop().time()<deadline
                await asyncio.sleep(0.005)
            before=copy.deepcopy(work.agents[0].messages)
            install_output(adapter,native,s3,original,recipe="full_v1")
            states[original.id]="Completed"
            failures = []
            previous_trace = sys.gettrace()
            def trace(frame, event, args):
                if event == "exception" and frame.f_code.co_filename.endswith("/adapters/qwen_sft.py"):
                    failures.append((frame.f_code.co_name,frame.f_lineno,args[0].__name__))
                return trace
            sys.settrace(trace)
            try:
                verified = store.observe(train["request_ref"])
            finally:
                sys.settrace(previous_trace)
            assert verified.get("trained") is True and verified.get("full_training_complete") is True, {
                "status":verified.get("status"),"error_type":verified.get("error_type"),"native_creates":len(native.calls),"source_frames":failures[-4:]}
            deadline=asyncio.get_running_loop().time()+5
            while not [value for use,value in work.originals if use=="original-reload-use"] and not reader.done():
                assert asyncio.get_running_loop().time()<deadline
                await asyncio.sleep(0.005)
            reloaded=next(value for use,value in work.originals if use=="original-reload-use")
            assert reloaded["accepted"] is True, {"status":reloaded.get("status"),"error_type":reloaded.get("error_type")}
            reload_handle=store._records[reloaded["request_ref"]]
            assert reload_handle is not original and reload_handle.id!=original.id and len(native.calls)==2
            assert reload_handle.extra["operation"]=="reload" and reloaded["request_ref"]!=train["request_ref"]
            publish_reload(reload_handle,s3)
            states[reload_handle.id]="Completed"
            events=await asyncio.wait_for(reader,5)
            assert events[-1]["response"]["data"]["warm_model_continuation"]["disposition"]=="continued_original"
            assert store.observe(train["request_ref"])["full_training_complete"] is True
            assert store.observe(reloaded["request_ref"])["fresh_process_greedy_equal"] is True
            assert work.agents[0].messages[:len(before)]==before
            assert len(native.calls)==2 and len(work.native.requests)==1 and len(work.agents)==1
            assert backend._warm_pending_producers=={} and work.owned_app.get_async_task_info()["active_count"]==0
            assert store._records[train["request_ref"]] is original
            assert work.agents[0].callback_handler is work.callbacks[0]
        finally:
            await close_original_worker(work,reader)
    asyncio.run(scenario())


async def collect(work):
    return [event async for event in runtime.agent_invocation({"prompt":work.literal})]
