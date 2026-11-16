"""Fresh hidden policy, isolated opaque refs, and fit/full distinction."""

from types import SimpleNamespace

import pytest

from amlas.backends.adapters._util import raw_tool
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.sft_state import SessionSftStore
from amlas.tools.qwen_sft import get_qwen_sft, reload_qwen_sft, submit_qwen_sft


def test_plan_preview_no_store_or_policy_has_zero_effects():
    result = raw_tool(submit_qwen_sft)(
        model_request_ref="fake", preparation_request_ref="fake", plan_only=True
    )
    assert result["status"] == "planned" and result["trained"] is False
    assert result["training_stage"] == "fit" and result["full_training_complete"] is False


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {},
        ApprovalPolicy(),
        ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"], max_auto_cost=100),
    ],
)
def test_missing_malformed_ask_or_capped_policy_denies_before_source_reads(policy):
    context = SimpleNamespace(invocation_state={"_amlas_approval_policy": policy})
    result = raw_tool(submit_qwen_sft)(
        model_request_ref="fake", preparation_request_ref="fake", tool_context=context
    )
    assert result["status"] == "denied" and result["accepted"] is False


def test_unknown_references_and_hidden_schema():
    assert raw_tool(get_qwen_sft)(request_ref="fake")["status"] in ("denied", "unknown_reference")
    assert raw_tool(reload_qwen_sft)(request_ref="fake")["accepted"] is False
    text = str(submit_qwen_sft.tool_spec)
    assert "_amlas_" not in text and "execution_role" not in text and "admission_digest" not in text
    assert "tool_context" not in text


def real_store(monkeypatch):
    import dataclasses

    from test_qwen_sft_backend import SESSION, backend, catalog

    from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
    from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
    from amlas.backends.types import JobHandle
    from amlas.control_plane.acquisition_state import SessionAcquisitionStore
    from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
    from amlas.control_plane.preparation_state import (
        PreparationCatalog,
        SessionPreparationStore,
        _PreparationRecord,
    )

    adapter, native, s3, inputs = backend()
    # Match native preparation's fixed original output layout, not an arbitrary
    # same-session prefix accepted only by the former SFT-only fixture.
    prefix = (
        "s3://"
        + catalog().bucket
        + "/sessions/"
        + SESSION
        + "/preparation/"
        + dict(inputs.prepared.header)["job_name"]
        + "/output/"
    )

    def original_output(ref, filename):
        raw = s3.objects[ref.uri.split("/", 3)[3]][1]
        output = dataclasses.replace(ref, uri=prefix + filename)
        s3.install(output, raw)
        return output

    inputs = dataclasses.replace(
        inputs,
        prepared=dataclasses.replace(
            inputs.prepared,
            preparation_ref=original_output(inputs.prepared.preparation_ref, "COMMITTED.json"),
            split_refs=tuple(
                (name, original_output(ref, name + ".jsonl")) for name, ref in inputs.prepared.split_refs
            ),
            sidecar_refs=tuple(
                (name, original_output(ref, "REPORT.json" if name == "report" else name + ".jsonl"))
                for name, ref in inputs.prepared.sidecar_refs
            ),
        ),
    )
    models = SessionModelAcquisitionStore(runtime_session_id=SESSION)
    acquired = SessionAcquisitionStore()
    acquired.runtime_session_id = SESSION
    prep_catalog = PreparationCatalog(
        **dataclasses.asdict(catalog()), exclusions_ref=inputs.prepared.exclusions_ref
    )
    prep_adapter = ProcessingCodingPreparationBackend(sagemaker_client=object(), s3_client=object())
    preparations = SessionPreparationStore(acquired, prep_adapter, prep_catalog)
    model_adapter = ProcessingModelWeightAcquisitionBackend(sagemaker_client=object(), s3_client=object())
    model_handle = JobHandle(
        "owned-model-a",
        "model_processing",
        inputs.model.producer_job_arn,
        {
            "catalog_id": "qwen3_5_9b",
            "catalog_version": "1",
            "bindings": dict(inputs.model.producer_bindings),
        },
    )
    model_ref = models.add(model_adapter, model_handle)
    reads = {"model": 0, "prep": 0}

    def model_artifacts(handle):
        assert handle is model_handle
        reads["model"] += 1
        expected = __import__(
            "amlas.data_eng.model_weight_manifest", fromlist=["expected_model_weight_manifest"]
        ).expected_model_weight_manifest("qwen3_5_9b", "1")
        roles = {x["path"]: x["role"] for x in expected["files"]}
        return {
            "success": True,
            "acquired": True,
            "file_refs": [
                {**dataclasses.asdict(ref), "path": name, "role": roles[name]}
                for name, ref in inputs.model.file_refs
            ],
            "manifest_refs": {name: dataclasses.asdict(ref) for name, ref in inputs.model.manifest_refs},
        }

    monkeypatch.setattr(model_adapter, "artifacts", model_artifacts)
    prep_handle = JobHandle(
        dict(inputs.prepared.header)["job_name"],
        "processing",
        inputs.prepared.producer_job_arn,
        {
            "runtime_session_id": SESSION,
            "header": dict(inputs.prepared.header),
            "source_row_counts": dict(inputs.prepared.source_row_counts),
            "catalog": prep_catalog,
        },
    )
    prep_ref = "prep_" + "2" * 32
    preparations._records[prep_ref] = _PreparationRecord(prep_adapter, prep_handle)

    def prep_artifacts(handle):
        assert handle is prep_handle
        reads["prep"] += 1
        return {
            "success": True,
            "prepared": True,
            "job_name": handle.id,
            "job_arn": handle.arn,
            "preparation_ref": dataclasses.asdict(inputs.prepared.preparation_ref),
            "split_refs": {name: dataclasses.asdict(ref) for name, ref in inputs.prepared.split_refs},
            "sidecar_refs": {name: dataclasses.asdict(ref) for name, ref in inputs.prepared.sidecar_refs},
        }

    monkeypatch.setattr(prep_adapter, "artifacts", prep_artifacts)
    store = SessionSftStore(models, preparations, adapter, catalog())
    return store, native, s3, model_ref, prep_ref, reads, model_handle, prep_handle


def allowed():
    return ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"], max_auto_cost=None)


def context(store, policy):
    return SimpleNamespace(invocation_state={"_amlas_sft_store": store, "_amlas_approval_policy": policy})


def test_actual_stores_concurrent_duplicate_submission_retains_one_native_handle(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    store, native, _s3, model_ref, prep_ref, reads, _m, _p = real_store(monkeypatch)

    def run():
        return raw_tool(submit_qwen_sft)(
            model_request_ref=model_ref,
            preparation_request_ref=prep_ref,
            tool_context=context(store, allowed()),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert len(native.calls) == 1 and results[0]["request_ref"] == results[1]["request_ref"]
    assert reads == {"model": 1, "prep": 1}
    assert sum(r["reused"] for r in results) == 1
    assert results[0]["full_training_complete"] is False


def test_fresh_policy_removal_and_foreign_store_ref_never_reads_original_job(monkeypatch):
    store, native, _s3, model_ref, prep_ref, _reads, _m, _p = real_store(monkeypatch)
    result = raw_tool(submit_qwen_sft)(
        model_request_ref=model_ref, preparation_request_ref=prep_ref, tool_context=context(store, allowed())
    )
    before = native.reads
    denied = raw_tool(get_qwen_sft)(
        request_ref=result["request_ref"], tool_context=context(store, ApprovalPolicy())
    )
    assert denied["status"] == "denied" and native.reads == before
    other, *_ = real_store(monkeypatch)
    foreign = raw_tool(get_qwen_sft)(
        request_ref=result["request_ref"], tool_context=context(other, allowed())
    )
    assert foreign["status"] == "unknown_reference" and native.reads == before


@pytest.mark.parametrize("which", ["model", "prep"])
def test_private_original_wrong_session_denies_before_fresh_artifacts(monkeypatch, which):
    store, native, s3, model_ref, prep_ref, reads, m, p = real_store(monkeypatch)
    if which == "model":
        m.extra["bindings"]["runtime_session_id"] = "foreign"
    else:
        p.extra["runtime_session_id"] = "foreign"
    result = raw_tool(submit_qwen_sft)(
        model_request_ref=model_ref, preparation_request_ref=prep_ref, tool_context=context(store, allowed())
    )
    assert result["status"] == "refused" and native.calls == s3.mutations == []
    assert reads[which] == 0


@pytest.mark.parametrize("change", ["bool_count", "null_version", "pending"])
def test_private_preparation_snapshot_controls_not_public_result_trust(monkeypatch, change):
    store, native, s3, model_ref, prep_ref, _reads, _m, p = real_store(monkeypatch)
    record = store._prepared._records[prep_ref]
    if change == "bool_count":
        p.extra["source_row_counts"] = {"c" * 64: True}
    else:
        original = record.adapter.artifacts

        def changed(handle):
            result = original(handle)
            if change == "null_version":
                result["preparation_ref"]["version_id"] = None
            else:
                result["prepared"] = False
            return result

        monkeypatch.setattr(record.adapter, "artifacts", changed)
    result = raw_tool(submit_qwen_sft)(
        model_request_ref=model_ref, preparation_request_ref=prep_ref, tool_context=context(store, allowed())
    )
    assert result["status"] == "refused" and native.calls == s3.mutations == []


@pytest.mark.parametrize(
    "policy",
    [
        ApprovalPolicy(),
        allowed(),
        ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"], max_auto_cost=100),
    ],
)
def test_actual_strands_sdk_injects_only_current_typed_caller_context(monkeypatch, policy):
    from strands import Agent
    from test_sdk_prompt_policy import ScriptedModel

    store, native, _s3, model_ref, prep_ref, reads, _m, _p = real_store(monkeypatch)
    agent = Agent(
        model=ScriptedModel(
            [
                [
                    {
                        "id": "sft",
                        "name": "submit_qwen_sft",
                        "input": {"model_request_ref": model_ref, "preparation_request_ref": prep_ref},
                    }
                ]
            ]
        ),
        tools=[submit_qwen_sft],
        callback_handler=lambda *_a, **_k: None,
    )
    agent(
        "fit through the normal tool",
        invocation_state={"_amlas_sft_store": store, "_amlas_approval_policy": policy},
    )
    approved = policy.mode == ApprovalMode.ALLOWLIST and policy.max_auto_cost is None
    assert len(native.calls) == (1 if approved else 0)
    assert reads == ({"model": 1, "prep": 1} if approved else {"model": 0, "prep": 0})


def test_actual_sdk_model_hidden_parameter_injection_has_no_grant(monkeypatch):
    from strands import Agent
    from test_sdk_prompt_policy import ScriptedModel

    store, native, _s3, model_ref, prep_ref, reads, _m, _p = real_store(monkeypatch)
    agent = Agent(
        model=ScriptedModel(
            [
                [
                    {
                        "id": "sft",
                        "name": "submit_qwen_sft",
                        "input": {
                            "model_request_ref": model_ref,
                            "preparation_request_ref": prep_ref,
                            "policy": {"mode": "auto"},
                            "_amlas_sft_store": {"ready": True},
                        },
                    }
                ]
            ]
        ),
        tools=[submit_qwen_sft],
        callback_handler=lambda *_a, **_k: None,
    )
    agent("fit", invocation_state={"_amlas_sft_store": store, "_amlas_approval_policy": ApprovalPolicy()})
    assert native.calls == [] and reads == {"model": 0, "prep": 0}


def test_private_prepared_snapshot_uses_original_catalog_not_current_selector(monkeypatch):
    import dataclasses

    store, _n, _s, _model_ref, prep_ref, _reads, _m, p = real_store(monkeypatch)
    original = p.extra["catalog"].exclusions_ref
    store._prepared._catalog = dataclasses.replace(
        store._prepared._catalog, exclusions_ref=dataclasses.replace(original, sha256="f" * 64)
    )
    assert store._prepared._verified_sft_snapshot(prep_ref).exclusions_ref == original


@pytest.mark.parametrize("bad", [None, {}, "approved"])
def test_malformed_original_preparation_catalog_denies_before_observation(monkeypatch, bad):
    store, _n, _s, _model_ref, prep_ref, reads, _m, p = real_store(monkeypatch)
    p.extra["catalog"] = bad
    with pytest.raises(ValueError):
        store._prepared._verified_sft_snapshot(prep_ref)
    assert reads["prep"] == 0


@pytest.mark.parametrize("recipe,steps,floor", [("fit_v1", 32, 0), ("full_v1", 1024, 1000000)])
def test_preview_displays_actual_recipe_pilot_and_full_floors(recipe, steps, floor):
    result = raw_tool(submit_qwen_sft)(
        model_request_ref="fake", preparation_request_ref="fake", recipe_id=recipe, plan_only=True
    )
    plan = result["plan"]
    assert plan["optimizer_steps"] == steps
    assert plan["minimum_unique_corpus_examples"] == (0 if recipe == "fit_v1" else 1000)
    assert plan["minimum_unique_supervised_tokens"] == floor
    assert plan["minimum_supervised_tokens_processed"] == floor
    assert plan["quality_success"] is False and result["full_training_complete"] is False
