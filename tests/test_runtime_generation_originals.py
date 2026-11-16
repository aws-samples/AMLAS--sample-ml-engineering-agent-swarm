"""Actual registered base/tuned generation targets and 164-row owning verification."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
from multi_producer_fixtures import SequentialProducerStream
from test_coding_case_native import native as case_fixture  # noqa: F401
from test_managed_generation_product import spec_text
from test_qwen_coding_generation_managed import _native_proof
from test_qwen_sft_backend import Native, catalog, described_resource_config, encoded, sources
from test_runtime_connected_warm_model import close_original_worker, connected_work
from test_runtime_sft_reload_originals import current_inputs
from test_runtime_warm_model_continuation import complete_fixture, tool_payload

from amlas import app as runtime
from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend
from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.coding_case_state import SessionCodingCaseFreezeStore, VerifiedCaseEvaluationSource
from amlas.control_plane.generation_state import SessionCodingGenerationStore
from amlas.control_plane.preparation_state import PreparationCatalog, SessionPreparationStore
from amlas.control_plane.sft_state import SessionSftStore, VerifiedSftCheckpointSource
from amlas.training.coding_generation import CONTRACT_SHA256, canonical_bytes
from amlas.tui.chat import tool_binding_backend as binding

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


class GenerationStream(SequentialProducerStream):
    def data_calls(self):
        return []

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls < 2:
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
            return
        self.calls += 1
        self.messages_seen.append(copy.deepcopy(messages))
        calls = []
        if self.calls == 3:
            calls = [("original-spec-use", "bind_coding_generation_spec", {"yaml_text": spec_text()})]
        elif self.calls == 4:
            result = next(
                block["toolResult"]
                for message in messages
                for block in message["content"]
                if block.get("toolResult", {}).get("toolUseId") == "original-spec-use"
            )
            calls = [
                (
                    "original-resolve-use",
                    "resolve_coding_evaluation_spec",
                    {"case_request_ref": "case_" + "b" * 32, "spec_ref": tool_payload(result)["spec_ref"]},
                )
            ]
        elif self.calls == 5:
            result = next(
                block["toolResult"]
                for message in messages
                for block in message["content"]
                if block.get("toolResult", {}).get("toolUseId") == "original-resolve-use"
            )
            spec = tool_payload(result)["spec_ref"]
            calls = [
                (
                    "original-gen-" + arm,
                    "submit_coding_generation",
                    {
                        "model_request_ref": "model_acq_" + "a" * 32,
                        "case_request_ref": "case_" + "b" * 32,
                        "spec_ref": spec,
                        "sft_request_ref": "sft_" + "c" * 32 if arm == "tuned" else "",
                    },
                )
                for arm in ("base", "tuned")
            ]
        yield {"messageStart": {"role": "assistant"}}
        if calls:
            for index, (use, name, args) in enumerate(calls):
                yield {
                    "contentBlockStart": {
                        "contentBlockIndex": index,
                        "start": {"toolUse": {"toolUseId": use, "name": name}},
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "contentBlockIndex": index,
                        "delta": {"toolUse": {"input": json.dumps(args)}},
                    }
                }
                yield {"contentBlockStop": {"contentBlockIndex": index}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Original generation remains source-owned."},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def publish_generation(s3, handle, prompts):
    binding = handle.extra["run_binding"]
    rows = [
        {
            "case_id": "HumanEval/" + str(index),
            "prompt_sha256": hashlib.sha256(canonical_bytes(prompts[index])).hexdigest(),
            "rendered_sha256": "c" * 64,
            "input_tokens": 32,
            "generated_tokens": 1,
            "outcome": "invalid",
            "source": "pass\n",
            "source_sha256": hashlib.sha256(b"pass\n").hexdigest(),
            "source_byte_count": 5,
            "failure_stage": "decode",
        }
        for index in range(164)
    ]
    model_files = binding["model_files"]
    receipt = {
        "schema_version": 1,
        "operation": "generate",
        "source_commit": catalog().source_commit,
        "case_count": 164,
        "quality_success": False,
        "native_admission": False,
        "contract_sha256": CONTRACT_SHA256,
        "binding": {
            "evaluation_id": binding["evaluation_id"],
            "arm": binding["arm"],
            "prompts_sha256": binding["prompts_ref"]["sha256"],
            "model_sha256": hashlib.sha256(encoded(model_files)).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(
                encoded(
                    [
                        item
                        for item in model_files
                        if item["role"] in ("tokenizer", "tokenizer_config", "chat_template")
                    ]
                )
            ).hexdigest(),
            "spec_sha256": binding["spec_sha256"],
        },
        "cases": [{key: value for key, value in row.items() if key != "source"} for row in rows],
        "stage_manifest_sha256": handle.extra["run_ref"].sha256,
        "privilege_proof": _native_proof(),
    }
    if binding["arm"] == "tuned":
        receipt["binding"]["model_sha256"] = hashlib.sha256(
            (receipt["binding"]["model_sha256"] + binding["checkpoint_ref"]["sha256"]).encode()
        ).hexdigest()
    prefix = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"] + "/model/"
    files = {}
    for name, raw in (
        ("GENERATION.json", encoded(receipt)),
        ("CANDIDATES.jsonl", b"".join(encoded(row) for row in rows)),
    ):
        artifact = DatasetArtifactRef(
            prefix + name,
            "out-" + name,
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            "jsonl" if name.endswith(".jsonl") else "json",
        )
        s3.install(artifact, raw)
        files[name] = {
            "sha256": artifact.sha256,
            "byte_count": artifact.byte_count,
            "format": artifact.format,
        }
    raw = encoded(
        {
            "schema_version": 1,
            "operation": "generate",
            "source_commit": catalog().source_commit,
            "case_count": 164,
            "quality_success": False,
            "native_admission": False,
            "files": files,
        }
    )
    s3.install(
        DatasetArtifactRef(
            prefix + "COMMITTED.json", "committed-version", hashlib.sha256(raw).hexdigest(), len(raw), "json"
        ),
        raw,
    )


def test_actual_registered_base_and_tuned_generation_each_complete_and_resume_once(
    monkeypatch, tmp_path, case_fixture  # noqa: F811 - registered pytest fixture alias
):
    s3, inputs = sources()
    # Keep the nine model payloads in one closed prefix; metadata is separate.
    manifests = []
    for name, artifact in inputs.model.manifest_refs:
        raw = s3.objects[artifact.uri.split("/", 3)[3]][1]
        moved = dataclasses.replace(
            artifact, uri=artifact.uri.replace("/original/model/", "/original/model-metadata/")
        )
        s3.install(moved, raw)
        manifests.append((name, moved))
    inputs = dataclasses.replace(
        inputs, model=dataclasses.replace(inputs.model, manifest_refs=tuple(manifests))
    )
    native = Native()
    requests, states = {}, {}
    actual_create = native.create_training_job

    def create(**kwargs):
        result = actual_create(**kwargs)
        requests[kwargs["TrainingJobName"]] = kwargs
        states[kwargs["TrainingJobName"]] = "InProgress"
        return result

    def describe(**kwargs):
        name = kwargs["TrainingJobName"]
        request = requests[name]
        return {
            **{key: value for key, value in request.items() if key != "Tags"},
            "ResourceConfig": described_resource_config(request["ResourceConfig"]),
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":training-job/" + name,
            "TrainingJobStatus": states[name],
            "ModelArtifacts": {"S3ModelArtifacts": request["OutputDataConfig"]["S3OutputPath"] + "/model/"},
        }

    monkeypatch.setattr(native, "create_training_job", create)
    monkeypatch.setattr(native, "describe_training_job", describe)
    monkeypatch.setattr(
        native, "list_tags", lambda **kwargs: {"Tags": requests[kwargs["ResourceArn"].split("/")[-1]]["Tags"]}
    )
    adapter = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    model = GenerationStream()
    work = connected_work(monkeypatch, model=model)
    monkeypatch.setenv(
        "AMLAS_RUNTIME_APPROVED_ACTIONS", '["run_processing_job","coding_pair_flow","coding_generation_flow"]'
    )
    case_adapter, _, _, _, case_catalog = case_fixture
    monkeypatch.setattr(
        binding,
        "build_case_freeze_store",
        lambda acquired: SessionCodingCaseFreezeStore(acquired, case_adapter, case_catalog),
    )
    monkeypatch.setattr(
        binding,
        "build_preparation_store",
        lambda acquired, **kwargs: SessionPreparationStore(
            acquired,
            ProcessingCodingPreparationBackend(),
            PreparationCatalog(
                **dataclasses.asdict(catalog()), exclusions_ref=inputs.prepared.exclusions_ref
            ),
        ),
    )
    monkeypatch.setattr(
        binding,
        "build_sft_store",
        lambda models, prepared: SessionSftStore(models, prepared, adapter, catalog()),
    )
    holder = {}

    def generation(models, cases, sft):
        selected = current_inputs(inputs, s3, models.runtime_session_id)
        prompts = tuple(
            {
                "case_id": "HumanEval/" + str(index),
                "prompt": "def function_" + str(index) + "():\n    pass",
                "entry_point": "function_" + str(index),
            }
            for index in range(164)
        )
        refs = []
        for name in (
            "PROMPTS.jsonl",
            "ORACLES.jsonl",
            "EXCLUSIONS.json",
            "CASE_MANIFEST.json",
            "COMMITTED.json",
        ):
            raw = b"".join(encoded(row) for row in prompts) if name == "PROMPTS.jsonl" else b"{}\n"
            artifact = DatasetArtifactRef(
                "s3://"
                + catalog().bucket
                + "/sessions/"
                + models.runtime_session_id
                + "/cases/original/"
                + name,
                "case-" + name,
                hashlib.sha256(raw).hexdigest(),
                len(raw),
                "jsonl" if name.endswith(".jsonl") else "json",
            )
            s3.install(artifact, raw)
            refs.append((name, artifact))
        case = VerifiedCaseEvaluationSource(
            models.runtime_session_id,
            "arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":processing-job/owned-cases",
            tuple(
                sorted(
                    {
                        "runtime_session_id": models.runtime_session_id,
                        "deployment_id": catalog().deployment_id,
                        "deployment_prefix": catalog().deployment_prefix,
                    }.items()
                )
            ),
            None,
            tuple(refs),
        )
        checkpoint = []
        for name in ("CHECKPOINT.json", "adapter_config.json", "adapter_model.safetensors"):
            raw = b"bounded neutral bytes"
            artifact = DatasetArtifactRef(
                "s3://"
                + catalog().bucket
                + "/sessions/"
                + models.runtime_session_id
                + "/sft/original/"
                + name,
                "checkpoint-" + name,
                hashlib.sha256(raw).hexdigest(),
                len(raw),
                "safetensors" if name.endswith(".safetensors") else "json",
            )
            s3.install(artifact, raw)
            checkpoint.append((name, artifact))
        source = VerifiedSftCheckpointSource(
            models.runtime_session_id,
            "arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":training-job/owned-train",
            "full_v1",
            catalog().source_commit,
            dict(checkpoint)["CHECKPOINT.json"],
            tuple(checkpoint),
        )
        monkeypatch.setattr(models, "_verified_sft_model_snapshot", lambda reference: selected.model)
        monkeypatch.setattr(cases, "_verified_evaluation_source", lambda reference: case)
        monkeypatch.setattr(sft, "_verified_generation_checkpoint", lambda reference, model: source)
        holder.update(prompts=prompts)
        return SessionCodingGenerationStore(models, cases, sft, adapter, catalog())

    monkeypatch.setattr(binding, "build_generation_store", generation)

    async def scenario():
        reader = asyncio.create_task(collect(work))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not work.observing.is_set() and not reader.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            backend = work.engines[0].tool_backend
            result = next(value for use, value in work.originals if use == "original-model-use")
            handle = backend._model_acquisition_store._records[result["request_ref"]].handle
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=handle, adapter=work.adapter, native=work.native),
            )
            deadline = asyncio.get_running_loop().time() + 5
            while (
                len([value for use, value in work.originals if use.startswith("original-gen-")]) < 2
                and not reader.done()
            ):
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            results = [value for use, value in work.originals if use.startswith("original-gen-")]
            assert len(results) == 2 and all(value["accepted"] is True for value in results), [
                {"status": value.get("status"), "error_type": value.get("error_type")} for value in results
            ]
            store = backend._generation_store
            records = tuple(store._records.values())
            assert len(native.calls) == len(records) == 2 and {record.extra["arm"] for record in records} == {
                "base",
                "tuned",
            }
            assert len({record.id for record in records}) == 2
            deadline = asyncio.get_running_loop().time() + 5
            while (
                len(
                    [
                        value
                        for value in backend._warm_pending_producers.values()
                        if value.kind == "generation"
                    ]
                )
                < 2
            ):
                assert not reader.done() and asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            before = copy.deepcopy(work.agents[0].messages)
            for record in records:
                publish_generation(s3, record, holder["prompts"])
                states[record.id] = "Completed"
            for reference in store._records:
                verified = store.observe(reference)
                assert verified.get("generated") is True, {
                    "arm": verified.get("arm"),
                    "status": verified.get("status"),
                    "error_type": verified.get("error_type"),
                }
            events = await asyncio.wait_for(reader, 5)
            assert (
                events[-1]["response"]["data"]["warm_model_continuation"]["disposition"]
                == "continued_original"
            )
            assert work.agents[0].messages[: len(before)] == before
            assert backend._warm_pending_producers == {} and model.calls == 8 and len(native.calls) == 2
            assert work.owned_app.get_async_task_info()["active_count"] == 0
        finally:
            await close_original_worker(work, reader)

    asyncio.run(scenario())


async def collect(work):
    return [event async for event in runtime.agent_invocation({"prompt": work.literal})]
