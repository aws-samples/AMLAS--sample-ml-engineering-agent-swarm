"""Managed prompt-to-generation plumbing; fixtures grant no native execution admission."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_qwen_sft_backend import ACCOUNT, SESSION, Native, catalog, ref, sources

from amlas.backends.types import DatasetArtifactRef, JobHandle, JobState
from amlas.control_plane.coding_case_state import SessionCodingCaseFreezeStore, VerifiedCaseEvaluationSource
from amlas.control_plane.generation_state import SessionCodingGenerationStore, fixed_generation_request
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.control_plane.sft_state import SessionSftStore
from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest
from amlas.spec.loader import load_spec_from_text

MODEL_REF = "model_acq_" + "a" * 32
CASE_REF = "case_" + "b" * 32
SFT_REF = "sft_" + "c" * 32


def spec_text(**change):
    pin = {"repo_id": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
    value = {
        "project_id": "coding_generation",
        "task_type": "text_generation",
        "target": "python_module",
        "primary_metric": "pass_at_1",
        "golden_dataset": "./cases.json",
        "baseline": {"type": "pinned_base", "pass_at_1": None},
        "acceptance": {"primary_metric_delta_vs_baseline": ">= 0.01"},
        "generation": {
            "model": pin,
            "tokenizer": pin,
            "recipe": {"id": "full_v1", "version": "1"},
            "quality": {"metric": "pass_at_1"},
        },
    }
    value.update(change)
    return "---\n" + json.dumps(value) + "\n---\nResearch-owned frozen spec."


class Adapter:
    backend_name = "coding_generation"

    def __init__(self):
        self.calls = []
        self.reads = 0

    def submit(self, **kw):
        self.calls.append(kw)
        return JobHandle(
            "owned-generation-a",
            self.backend_name,
            "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":training-job/owned-generation-a",
            {
                "submission_state": "ambiguous",
                "arm": "base" if kw["checkpoint"] is None else "tuned",
                "runtime_session_id": SESSION,
                "evaluation_id": kw["evaluation_id"],
            },
        )

    def status(self, handle):
        return SimpleNamespace(state=JobState.UNKNOWN)

    def artifacts(self, handle):
        self.reads += 1
        raise TimeoutError("neutral unavailable observation")


@pytest.fixture
def state(monkeypatch):
    s3, inputs = sources()
    manifests = []
    for name, artifact in inputs.model.manifest_refs:
        original = s3.objects.pop(artifact.uri.split("/", 3)[3])
        moved = dataclasses.replace(
            artifact, uri=artifact.uri.replace("/original/model/", "/original/model-metadata/")
        )
        s3.install(moved, original[1])
        manifests.append((name, moved))
    inputs = dataclasses.replace(
        inputs, model=dataclasses.replace(inputs.model, manifest_refs=tuple(manifests))
    )
    models = object.__new__(SessionModelAcquisitionStore)
    models.runtime_session_id = SESSION
    cases = object.__new__(SessionCodingCaseFreezeStore)
    cases.runtime_session_id = SESSION
    sft = object.__new__(SessionSftStore)
    sft.runtime_session_id = SESSION
    sft._models = models
    reads = []
    files = tuple(
        (name, ref(name, raw=b"{}\n", fmt="jsonl" if name.endswith(".jsonl") else "json"))
        for name in (
            "PROMPTS.jsonl",
            "ORACLES.jsonl",
            "EXCLUSIONS.json",
            "CASE_MANIFEST.json",
            "COMMITTED.json",
        )
    )
    case = VerifiedCaseEvaluationSource(
        SESSION,
        "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/owned-cases",
        tuple(
            sorted(
                {
                    "runtime_session_id": SESSION,
                    "deployment_id": catalog().deployment_id,
                    "deployment_prefix": "owned",
                }.items()
            )
        ),
        None,
        files,
    )

    def model_snapshot(value):
        assert value == MODEL_REF
        reads.append("model")
        return inputs.model

    def case_snapshot(value):
        assert value == CASE_REF
        reads.append("case")
        return case

    monkeypatch.setattr(models, "_verified_sft_model_snapshot", model_snapshot)
    monkeypatch.setattr(cases, "_verified_evaluation_source", case_snapshot)
    adapter = Adapter()
    store = SessionCodingGenerationStore(models, cases, sft, adapter, catalog())
    return store, adapter, reads, s3, inputs, case


def resolved_spec(store, reads=None):
    original = store.bind_spec(spec_text())["spec_ref"]
    result = store.resolve_case_spec(CASE_REF, original)["spec_ref"]
    if reads is not None:
        reads.clear()
    return result


def test_spec_binding_uses_actual_validated_canonical_spec_and_no_source_reads(state):
    store, adapter, reads, *_ = state
    result = store.bind_spec(spec_text())
    from amlas.arbiter.coding import coding_spec_sha256

    assert result["spec_sha256"] == coding_spec_sha256(load_spec_from_text(spec_text()))
    assert result["spec_ref"].startswith("spec_") and not adapter.calls and not reads
    assert result["native_admission"] is result["quality_success"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"task_type": "regression"},
        {"primary_metric": "resolved_rate"},
        {"generation": {"model": {"repo_id": "Qwen/other", "revision": "a" * 40}}},
    ],
)
def test_wrong_semantic_spec_cannot_be_bound(state, change):
    with pytest.raises(ValueError):
        state[0].bind_spec(spec_text(**change))
    assert state[1].calls == [] and state[2] == []


def test_base_and_tuned_share_private_evaluation_identity_and_unknown_retains_original(state, monkeypatch):
    store, adapter, reads, _, _inputs, _ = state
    spec = resolved_spec(store, reads)
    base = store.submit(MODEL_REF, CASE_REF, spec)
    again = store.submit(MODEL_REF, CASE_REF, spec)
    assert base["request_ref"] == again["request_ref"] and len(adapter.calls) == 1
    assert base["status"] == "acceptance_unknown" and base["accepted"] is False
    observed = store.observe(base["request_ref"])
    assert observed["request_ref"] == base["request_ref"] and observed["status"] == "observation_failed"
    assert len(adapter.calls) == 1 and reads == ["model", "case"]
    from amlas.control_plane.sft_state import VerifiedSftCheckpointSource

    cp = ref("CHECKPOINT.json", raw=b"{}")
    checkpoint = VerifiedSftCheckpointSource(
        SESSION,
        "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":training-job/owned-original",
        "full_v1",
        "d" * 40,
        cp,
        (
            ("CHECKPOINT.json", cp),
            ("adapter_config.json", ref("adapter_config.json", raw=b"{}")),
            ("adapter_model.safetensors", ref("adapter_model.safetensors", raw=b"abc", fmt="safetensors")),
        ),
    )
    monkeypatch.setattr(store._sft, "_verified_generation_checkpoint", lambda value, model: checkpoint)
    tuned = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
    assert (
        tuned["evaluation_id"] == base["evaluation_id"]
        and adapter.calls[-1]["checkpoint"].source_commit == "d" * 40
    )
    assert "ORACLES" not in json.dumps(base) and "execution_role" not in json.dumps(base)


def test_foreign_spec_ref_and_wrong_session_refuse_before_original_reads(state):
    store, adapter, reads, *_ = state
    with pytest.raises(ValueError):
        store.submit(MODEL_REF, CASE_REF, "spec_" + "f" * 32)
    assert reads == [] and adapter.calls == []
    store._models.runtime_session_id = "different"
    spec = resolved_spec(store, reads)
    with pytest.raises(ValueError):
        store.submit(MODEL_REF, CASE_REF, spec)
    assert reads == [] and adapter.calls == []


def test_fixed_request_has_only_public_prompts_and_closed_gpu_contract():
    run = ref("RUN.json", raw=b"{}")
    run = dataclasses.replace(
        run,
        uri="s3://owned-test-bucket/sessions/" + SESSION + "/generation/owned-gen/input/bindings/RUN.json",
    )
    request = fixed_generation_request(catalog(), SESSION, "owned-gen", run, "base")
    from botocore.session import Session
    from botocore.validate import validate_parameters

    validate_parameters(
        request, Session().get_service_model("sagemaker").operation_model("CreateTrainingJob").input_shape
    )
    assert request["ResourceConfig"]["InstancePreferences"] == [
        {"InstanceType": name} for name in ("ml.g6e.16xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge")
    ]
    assert "InstanceType" not in request["ResourceConfig"]
    assert request["EnableNetworkIsolation"] is True
    assert [x["ChannelName"] for x in request["InputDataConfig"]] == ["base_model", "prompts", "bindings"]
    assert "Environment" not in request and "HyperParameters" not in request
    assert request["AlgorithmSpecification"]["ContainerArguments"][:4] == [
        "--operation",
        "generate",
        "--recipe-id",
        "humaneval_greedy_v1",
    ]
    assert "ORACLES" not in json.dumps(request)


def test_tools_hidden_schema_registration_and_policy_denial_before_reads(state):
    from amlas.backends.adapters._util import raw_tool
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
    from amlas.tools.coding_generation import (
        bind_coding_generation_spec,
        get_coding_generation,
        submit_coding_generation,
    )
    from amlas.tui.chat.agent_tools import (
        READ_ONLY_TOOLS,
        STATE_CHANGING_TOOLS,
        get_safe_llm_tools,
        tool_display_name,
    )

    store, adapter, reads, *_ = state
    context = SimpleNamespace(
        invocation_state={"_amlas_generation_store": store, "_amlas_approval_policy": ApprovalPolicy()}
    )
    original_spec = raw_tool(bind_coding_generation_spec)(spec_text(), tool_context=context)["spec_ref"]
    from amlas.tools.coding_pair import resolve_coding_evaluation_spec

    context.invocation_state["_amlas_approval_policy"] = ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST, allowlist=["coding_pair_flow"], max_auto_cost=None
    )
    spec = raw_tool(resolve_coding_evaluation_spec)(CASE_REF, original_spec, tool_context=context)["spec_ref"]
    reads.clear()
    for policy in (
        ApprovalPolicy(),
        {},
        ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["coding_generation_flow"], max_auto_cost=100),
    ):
        context.invocation_state["_amlas_approval_policy"] = policy
        result = raw_tool(submit_coding_generation)(MODEL_REF, CASE_REF, spec, tool_context=context)
        assert result["status"] == "denied" and reads == [] and adapter.calls == []
    context.invocation_state["_amlas_approval_policy"] = ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST, allowlist=["coding_generation_flow"], max_auto_cost=None
    )
    result = raw_tool(submit_coding_generation)(MODEL_REF, CASE_REF, spec, tool_context=context)
    assert result["status"] == "acceptance_unknown" and len(adapter.calls) == 1
    assert (
        raw_tool(get_coding_generation)(result["request_ref"], tool_context=context)["status"]
        == "observation_failed"
    )
    names = [tool_display_name(tool) for tool in get_safe_llm_tools()]
    assert all(
        names.count(n) == 1
        for n in ("bind_coding_generation_spec", "submit_coding_generation", "get_coding_generation")
    )
    assert "get_coding_generation" in READ_ONLY_TOOLS
    assert {"submit_coding_generation", "bind_coding_generation_spec"} <= STATE_CHANGING_TOOLS
    for tool in (submit_coding_generation, get_coding_generation):
        assert not set(tool.tool_spec["inputSchema"]["json"]["properties"]) & {
            "tool_context",
            "policy",
            "source_commit",
            "image_uri",
            "recipe_id",
            "native_handle",
        }


def test_real_adapter_derives_run_and_preserves_one_unknown_create(state, monkeypatch):
    from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend

    _store, _, _, s3, inputs, case = state
    native = Native()
    native.error = TimeoutError("neutral create timeout")
    backend = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    prompts = dict(case.file_refs)["PROMPTS.jsonl"]
    s3.install(prompts, b"{}\n")
    h = backend.submit(
        model=inputs.model,
        cases=case,
        checkpoint=None,
        catalog=catalog(),
        spec_sha256=hashlib.sha256(b"spec").hexdigest(),
        evaluation_id="same-paired",
        runtime_session_id=SESSION,
    )
    assert len(native.calls) == 1 and h.extra["submission_state"] == "ambiguous"
    run = h.extra["run_binding"]
    assert run["arm"] == "base" and set(h.extra["staged_refs"]) == {
        "prompts/PROMPTS.jsonl",
        "bindings/RUN.json",
    } | {"base_model/" + name for name, _artifact in inputs.model.file_refs}
    assert (
        "ORACLES" not in json.dumps(run)
        and run["model_files"] == expected_model_weight_manifest("qwen3_5_9b", "1")["files"]
    )


def test_generation_gpu_operations_are_expensive_not_bounded_compute():
    from amlas.control_plane.approvals import ActionClass, classify_action

    assert classify_action("submit_coding_generation") is ActionClass.EXPENSIVE_JOB
    assert classify_action("coding_generation_flow") is ActionClass.EXPENSIVE_JOB


def test_contaminated_base_prefix_refuses_before_staging_or_create(state):
    from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend

    _, _, _, s3, inputs, case = state
    native = Native()
    backend = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    parent = inputs.model.file_refs[0][1].uri.rsplit("/", 1)[0]
    s3.install(dataclasses.replace(ref("extra.json", raw=b"{}"), uri=parent + "/extra.json"), b"{}")
    with pytest.raises(ValueError, match="extra or missing"):
        backend.submit(
            model=inputs.model,
            cases=case,
            checkpoint=None,
            catalog=catalog(),
            spec_sha256="a" * 64,
            evaluation_id="paired",
            runtime_session_id=SESSION,
        )
    assert not native.calls and not s3.mutations


def installed_generation(state):
    from test_qwen_coding_generation_managed import _native_proof

    from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend
    from amlas.backends.adapters.qwen_sft import _json
    from amlas.training.coding_generation import CONTRACT_SHA256

    _, _, _, s3, inputs, case = state
    native = Native()
    backend = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    public_rows = tuple(
        {
            "case_id": "HumanEval/" + str(index),
            "prompt": "def function_" + str(index) + "():\n    pass",
            "entry_point": "function_" + str(index),
        }
        for index in range(164)
    )
    public_raw = b"".join(_json(row) for row in public_rows)
    original_prompt = dict(case.file_refs)["PROMPTS.jsonl"]
    prompt_ref = dataclasses.replace(
        original_prompt, sha256=hashlib.sha256(public_raw).hexdigest(), byte_count=len(public_raw)
    )
    case = dataclasses.replace(
        case,
        file_refs=tuple(
            (name, prompt_ref if name == "PROMPTS.jsonl" else artifact) for name, artifact in case.file_refs
        ),
    )
    s3.install(prompt_ref, public_raw)
    handle = backend.submit(
        model=inputs.model,
        cases=case,
        checkpoint=None,
        catalog=catalog(),
        spec_sha256="a" * 64,
        evaluation_id="fixture-paired",
        runtime_session_id=SESSION,
    )
    binding = handle.extra["run_binding"]
    files = binding["model_files"]
    model_sha = hashlib.sha256(_json(files)).hexdigest()
    tokenizer_sha = hashlib.sha256(
        _json([item for item in files if item["role"] in ("tokenizer", "tokenizer_config", "chat_template")])
    ).hexdigest()
    rows = [
        {
            "case_id": "HumanEval/" + str(index),
            "prompt_sha256": hashlib.sha256(_json(public_rows[index]).rstrip(b"\n")).hexdigest(),
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
    metadata = {
        "schema_version": 1,
        "operation": "generate",
        "source_commit": catalog().source_commit,
        "case_count": 164,
        "quality_success": False,
        "native_admission": False,
        "contract_sha256": CONTRACT_SHA256,
        "binding": {
            "evaluation_id": binding["evaluation_id"],
            "arm": "base",
            "prompts_sha256": binding["prompts_ref"]["sha256"],
            "model_sha256": model_sha,
            "tokenizer_sha256": tokenizer_sha,
            "spec_sha256": "a" * 64,
        },
        "cases": [{key: value for key, value in row.items() if key != "source"} for row in rows],
        "stage_manifest_sha256": handle.extra["run_ref"].sha256,
        "privilege_proof": _native_proof(),
    }

    def publish():
        prefix = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"] + "/model/"
        raws = {"GENERATION.json": _json(metadata), "CANDIDATES.jsonl": b"".join(_json(row) for row in rows)}
        descriptors = {}
        for name, raw in raws.items():
            r = DatasetArtifactRef(
                prefix + name,
                "output-" + name,
                hashlib.sha256(raw).hexdigest(),
                len(raw),
                "jsonl" if name.endswith(".jsonl") else "json",
            )
            s3.install(r, raw)
            descriptors[name] = {"sha256": r.sha256, "byte_count": r.byte_count, "format": r.format}
        commit = {
            "schema_version": 1,
            "operation": "generate",
            "source_commit": catalog().source_commit,
            "case_count": 164,
            "quality_success": False,
            "native_admission": False,
            "files": descriptors,
        }
        raw = _json(commit)
        s3.install(
            DatasetArtifactRef(
                prefix + "COMMITTED.json",
                "committed-version",
                hashlib.sha256(raw).hexdigest(),
                len(raw),
                "json",
            ),
            raw,
        )

    publish()
    native.state = "Completed"
    return backend, native, s3, handle, metadata, rows, publish


def test_completed_native_job_requires_all_committed_bytes_and_keeps_quality_false(state):
    backend, native, s3, handle, _, _, _ = installed_generation(state)
    before = list(s3.mutations)
    result = backend.artifacts(handle)
    assert result["success"] is result["generated"] is True
    assert result["quality_success"] is result["native_admission"] is False
    assert result["case_count"] == 164 and result["candidate_ref"].format == "jsonl"
    assert before == s3.mutations and len(native.calls) == 1


@pytest.mark.parametrize("change", ["spec", "model", "source", "case_missing", "candidate_bytes", "quality"])
def test_rehashed_output_claims_cannot_replace_original_binding_or_population(state, change):
    backend, native, _, handle, metadata, rows, publish = installed_generation(state)
    if change == "spec":
        metadata["binding"]["spec_sha256"] = "f" * 64
    elif change == "model":
        metadata["binding"]["model_sha256"] = "f" * 64
    elif change == "source":
        metadata["source_commit"] = "f" * 40
    elif change == "case_missing":
        rows.pop()
        metadata["cases"].pop()
    elif change == "candidate_bytes":
        rows[0]["source"] = "pass  \n"
    else:
        metadata["quality_success"] = True
    publish()
    result = backend.artifacts(handle)
    assert result["generated"] is result["success"] is False
    assert result["status"] == "output_verification_failed" and len(native.calls) == 1


def test_tuned_run_preserves_original_checkpoint_source_distinct_from_generation_image(state):
    from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend
    from amlas.control_plane.sft_state import VerifiedSftCheckpointSource

    _, _, _, s3, inputs, case = state
    native = Native()
    backend = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    s3.install(dict(case.file_refs)["PROMPTS.jsonl"], b"{}\n")
    cp = ref("CHECKPOINT.json", raw=b"{}")
    cfg = ref("adapter_config.json", raw=b"{}")
    tensor = ref("adapter_model.safetensors", raw=b"abc", fmt="safetensors")
    for artifact, raw in ((cp, b"{}"), (cfg, b"{}"), (tensor, b"abc")):
        s3.install(artifact, raw)
    checkpoint = VerifiedSftCheckpointSource(
        SESSION,
        "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":training-job/owned-original",
        "full_v1",
        "d" * 40,
        cp,
        (("CHECKPOINT.json", cp), ("adapter_config.json", cfg), ("adapter_model.safetensors", tensor)),
    )
    handle = backend.submit(
        model=inputs.model,
        cases=case,
        checkpoint=checkpoint,
        catalog=catalog(),
        spec_sha256="a" * 64,
        evaluation_id="paired",
        runtime_session_id=SESSION,
    )
    assert handle.extra["run_binding"]["checkpoint_source_commit"] == "d" * 40
    assert handle.extra["run_binding"]["source_commit"] == catalog().source_commit
    assert "adapter" in [item["ChannelName"] for item in native.calls[0]["InputDataConfig"]]
    assert set(handle.extra["staged_refs"]) == {
        "prompts/PROMPTS.jsonl",
        "bindings/RUN.json",
        "adapter/CHECKPOINT.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
    } | {"base_model/" + name for name, _artifact in inputs.model.file_refs}


def test_pure_startup_missing_generation_catalog_remains_unadmitted(state, monkeypatch):
    from amlas.backends.registry import REGISTRY
    from amlas.control_plane.generation_binding import build_generation_store
    from amlas.control_plane.sft_state import SftConfigurationMissingError

    store, adapter, reads, *_ = state
    monkeypatch.setattr(REGISTRY, "resolve", lambda _port: SimpleNamespace(adapter=adapter))
    unadmitted = build_generation_store(store._models, store._cases, store._sft, env={})
    spec = unadmitted.bind_spec(spec_text())["spec_ref"]
    with pytest.raises(SftConfigurationMissingError):
        unadmitted.submit(MODEL_REF, CASE_REF, spec)
    assert not adapter.calls and not reads


def test_private_generation_handoff_reobserves_original_and_cannot_use_public_json(state, monkeypatch):
    store, adapter, reads, *_ = state
    spec = resolved_spec(store, reads)
    result = store.submit(MODEL_REF, CASE_REF, spec)
    accessor = getattr(store, "_verified_evaluation_source", None)
    assert callable(accessor), "fresh private generation evaluator handoff missing"
    calls = []
    files = {
        name: ref(name, raw=b"{}\n", fmt="jsonl" if name.endswith(".jsonl") else "json")
        for name in ("CANDIDATES.jsonl", "GENERATION.json", "COMMITTED.json")
    }

    def fresh(handle):
        calls.append(handle)
        return {
            "success": True,
            "generated": True,
            "candidate_ref": files["CANDIDATES.jsonl"],
            "generation_ref": files["GENERATION.json"],
            "committed_ref": files["COMMITTED.json"],
            "case_count": 164,
        }

    monkeypatch.setattr(adapter, "artifacts", fresh)
    private = accessor(result["request_ref"])
    assert len(calls) == 1 and private.runtime_session_id == SESSION
    assert private.producer_job_arn == result["job_arn"] and private.evaluation_id == result["evaluation_id"]
    assert dict(private.file_refs) == files
    with pytest.raises(dataclasses.FrozenInstanceError):
        private.evaluation_id = "changed"
    with pytest.raises(ValueError):
        accessor(result)
    assert len(calls) == 1
    monkeypatch.setattr(adapter, "artifacts", lambda _h: {"success": False, "generated": False})
    with pytest.raises(ValueError):
        accessor(result["request_ref"])


def test_coding_spec_fingerprint_survives_validated_json_roundtrip(state):
    from amlas.arbiter.coding import coding_spec_sha256
    from amlas.spec.schema import SpecModel

    original = load_spec_from_text(spec_text())
    restored = SpecModel.model_validate(original.model_dump(mode="json"))
    assert coding_spec_sha256(original) == coding_spec_sha256(restored)
    result = state[0].bind_spec(spec_text())
    retained = state[0]._specs[result["spec_ref"]][1]
    assert hashlib.sha256(retained).hexdigest() == result["spec_sha256"]
    assert coding_spec_sha256(SpecModel.model_validate_json(retained)) == result["spec_sha256"]


def test_context_overflow_remains_in_complete_population_without_generation_or_quality(state):
    backend, native, _, handle, metadata, rows, publish = installed_generation(state)
    rows[0].update(
        input_tokens=5000,
        generated_tokens=0,
        outcome="context_overflow",
        source=None,
        source_sha256=None,
        source_byte_count=0,
        failure_stage="tokenize",
    )
    metadata["cases"][0] = {key: value for key, value in rows[0].items() if key != "source"}
    publish()
    result = backend.artifacts(handle)
    assert result["success"] is result["generated"] is True
    assert result["case_count"] == 164 and result["quality_success"] is False
    assert len(native.calls) == 1


@pytest.mark.parametrize("error_value", [None, [], "neutral-error", 0, {"Code": None}, {"Code": []}])
def test_review_malformed_native_error_keeps_original_opaque_handle_without_retry(state, error_value):
    from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend

    store, _, reads, s3, _inputs, case = state

    class NativeFailureError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.response = {"Error": error_value}

    native = Native()
    native.error = NativeFailureError("neutral Create observation failure")
    s3.install(dict(case.file_refs)["PROMPTS.jsonl"], b"{}\n")
    store._adapter = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    spec = resolved_spec(store, reads)
    result = store.submit(MODEL_REF, CASE_REF, spec)
    again = store.submit(MODEL_REF, CASE_REF, spec)
    assert result["status"] == "acceptance_unknown" and result["request_ref"] == again["request_ref"]
    assert len(native.calls) == 1


def test_review_equivalent_specs_keep_one_original_unknown_submission(state):
    store, adapter, reads, *_ = state
    first = store.bind_spec(spec_text())
    equivalent = store.bind_spec(
        spec_text().replace('"project_id": "coding_generation",', '"project_id" : "coding_generation",')
    )
    assert first["spec_ref"] == equivalent["spec_ref"]
    first = store.resolve_case_spec(CASE_REF, first["spec_ref"])
    equivalent = store.resolve_case_spec(CASE_REF, equivalent["spec_ref"])
    assert first["spec_ref"] == equivalent["spec_ref"]
    reads.clear()
    original = store.submit(MODEL_REF, CASE_REF, first["spec_ref"])
    repeated = store.submit(MODEL_REF, CASE_REF, equivalent["spec_ref"])
    assert original["request_ref"] == repeated["request_ref"] and len(adapter.calls) == 1
    assert reads == ["model", "case"]


@pytest.mark.parametrize(
    "change",
    [
        "generated_null",
        "invalid_null",
        "prompt_hash",
        "render_hash",
        "render_type",
        "stage_type",
        "stage_value",
        "source_count_type",
    ],
)
def test_review_rehashed_candidate_claims_require_closed_original_per_case_contract(state, change):
    backend, native, _, handle, metadata, rows, publish = installed_generation(state)
    row = rows[0]
    if change in ("generated_null", "invalid_null"):
        row.update(
            outcome="generated" if change == "generated_null" else "invalid",
            source=None,
            source_sha256=None,
            source_byte_count=0,
            failure_stage="none" if change == "generated_null" else "decode",
        )
    elif change == "prompt_hash":
        row["prompt_sha256"] = "f" * 64
    elif change == "render_hash":
        row["rendered_sha256"] = "short"
    elif change == "render_type":
        row["rendered_sha256"] = None
    elif change == "stage_type":
        row["failure_stage"] = 12
    elif change == "stage_value":
        row["failure_stage"] = "unknown_stage"
    else:
        row.update(
            source=None,
            source_sha256=None,
            source_byte_count=False,
            outcome="setup_error",
            rendered_sha256="",
            input_tokens=0,
            generated_tokens=0,
            failure_stage="setup",
        )
    metadata["cases"][0] = {key: value for key, value in row.items() if key != "source"}
    publish()
    result = backend.artifacts(handle)
    assert result["status"] == "output_verification_failed" and result["generated"] is False
    assert len(native.calls) == 1


@pytest.mark.parametrize(
    "outcome",
    [
        "generated",
        "invalid",
        "setup_error",
        "generation_error",
        "context_overflow",
        "output_overflow",
        "invalid_termination",
        "truncated",
        "incomplete",
    ],
)
def test_review_retains_every_worker_outcome_and_bounded_partial_failure(state, outcome):
    backend, native, _, handle, metadata, rows, publish = installed_generation(state)
    row = rows[0]
    row["outcome"] = outcome
    if outcome == "generated":
        raw = b"def function_0():\n    return True\n"
        row.update(
            source=raw.decode(),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            source_byte_count=len(raw),
            generated_tokens=2,
            failure_stage="none",
        )
    elif outcome == "setup_error":
        row.update(
            source=None,
            source_sha256=None,
            source_byte_count=0,
            input_tokens=0,
            generated_tokens=0,
            rendered_sha256="",
            failure_stage="setup",
        )
    elif outcome == "context_overflow":
        row.update(
            source=None,
            source_sha256=None,
            source_byte_count=0,
            input_tokens=5000,
            generated_tokens=0,
            failure_stage="tokenize",
        )
    elif outcome == "output_overflow":
        row.update(source=None, source_sha256="e" * 64, source_byte_count=65537, failure_stage="decode")
    elif outcome == "truncated":
        row.update(generated_tokens=1024, failure_stage="decode")
    else:
        row["failure_stage"] = "decode"
    metadata["cases"][0] = {key: value for key, value in row.items() if key != "source"}
    publish()
    result = backend.artifacts(handle)
    assert result["success"] is result["generated"] is True
    assert result["case_count"] == 164 and result["quality_success"] is result["native_admission"] is False
    assert len(native.calls) == 1


@pytest.mark.parametrize("field,value", [("generated_tokens", True), ("source_byte_count", 5.0)])
def test_review_manifest_case_types_cannot_alias_candidate_integer_values(state, field, value):
    backend, native, _, handle, metadata, _, publish = installed_generation(state)
    metadata["cases"][0][field] = value
    publish()
    result = backend.artifacts(handle)
    assert result["status"] == "output_verification_failed" and result["generated"] is False
    assert len(native.calls) == 1


@pytest.mark.parametrize("trailing_separator", [False, True], ids=["native-no-slash", "legacy-slash"])
def test_native_generation_artifact_root_accepts_one_optional_separator(
    state, monkeypatch, trailing_separator
):
    backend, native, s3, handle, _, _, _ = installed_generation(state)
    observed = native.describe_training_job(TrainingJobName=handle.id)
    original = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"] + "/model"
    uri = original + ("/" if trailing_separator else "")
    observed["ModelArtifacts"]["S3ModelArtifacts"] = uri
    monkeypatch.setattr(native, "describe_training_job", lambda **_kw: observed)
    before = list(s3.mutations)
    result = backend.artifacts(handle)
    assert result["success"] is result["generated"] is True, result
    assert result["case_count"] == 164
    assert result["candidate_ref"].uri == original + "/CANDIDATES.jsonl"
    assert result["quality_success"] is result["native_admission"] is False
    assert observed["ModelArtifacts"]["S3ModelArtifacts"] == uri
    assert before == s3.mutations and len(native.calls) == 1


@pytest.mark.parametrize("trailing_separator", [False, True], ids=["native-no-slash", "one-slash"])
def test_native_generation_artifact_root_accepts_the_recorded_real_job_bound_layout(
    state, monkeypatch, trailing_separator
):
    # Real DescribeTrainingJob answers carry <S3OutputPath>/<TrainingJobName>/output/model with no trailing separator.
    backend, native, s3, handle, _, _, _ = installed_generation(state)
    root = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"]
    old = root.split("/", 3)[3] + "/model/"
    new = root.split("/", 3)[3] + "/" + handle.id + "/output/model/"
    for key in [k for k in s3.objects if k.startswith(old)]:
        s3.objects[new + key[len(old) :]] = s3.objects.pop(key)
    observed = native.describe_training_job(TrainingJobName=handle.id)
    uri = root + "/" + handle.id + "/output/model" + ("/" if trailing_separator else "")
    observed["ModelArtifacts"]["S3ModelArtifacts"] = uri
    monkeypatch.setattr(native, "describe_training_job", lambda **_kw: observed)
    result = backend.artifacts(handle)
    assert result["success"] is result["generated"] is True, result
    assert result["candidate_ref"].uri == root + "/" + handle.id + "/output/model/CANDIDATES.jsonl"
    assert result["quality_success"] is result["native_admission"] is False


@pytest.mark.parametrize(
    "variant",
    [
        "wrong-bucket",
        "wrong-root",
        "prefix-sibling",
        "root-only",
        "empty",
        "dot",
        "dotdot",
        "double-slash",
        "double-trailing",
        "query",
        "fragment",
        "backslash",
    ],
)
def test_native_generation_artifact_root_refuses_other_normalizations(state, monkeypatch, variant):
    backend, native, s3, handle, _, _, _ = installed_generation(state)
    root = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"]
    good = root + "/model"
    uris = {
        "wrong-bucket": good.replace("s3://owned-test-bucket/", "s3://other-test-bucket/", 1),
        "wrong-root": "s3://owned-test-bucket/unrelated/model",
        "prefix-sibling": root + "-other/model",
        "root-only": root + "/",
        "empty": "",
        "dot": root + "/./model",
        "dotdot": root + "/../model",
        "double-slash": root + "//model",
        "double-trailing": good + "//",
        "query": good + "?revision=1",
        "fragment": good + "#summary",
        "backslash": good + chr(92) + "summary",
    }
    observed = native.describe_training_job(TrainingJobName=handle.id)
    observed["ModelArtifacts"]["S3ModelArtifacts"] = uris[variant]
    monkeypatch.setattr(native, "describe_training_job", lambda **_kw: observed)
    reads = []
    original_list = s3.list_objects_v2
    monkeypatch.setattr(s3, "list_objects_v2", lambda **kw: reads.append(kw) or original_list(**kw))
    before = list(s3.mutations)
    result = backend.artifacts(handle)
    assert result["success"] is result["generated"] is False
    assert result["status"] == "output_verification_failed"
    assert not reads, "malformed native root reached S3 inventory"
    assert observed["ModelArtifacts"]["S3ModelArtifacts"] == uris[variant]
    assert before == s3.mutations and len(native.calls) == 1
