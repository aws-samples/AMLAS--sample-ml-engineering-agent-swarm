"""Neutral cold-process graph fixture; no model/candidate code or native service."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Bootstrap ordinary product tools first, before optional producer module imports.
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

# isort: split
import model_acquisition_fixtures as mf
import pytest
import test_coding_preparation_binding as prep_binding
import test_qwen_sft_backend as sf
from test_campaign_state import DDB
from test_managed_coding_preparation import fixture as preparation_fixture
from test_managed_commitpack_preparation import complete_outputs, staged_fixture

from amlas.backends.adapters.campaign_state import DynamoCampaignStateBackend
from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.campaign_state import CampaignStateCatalog
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.control_plane.preparation_state import SessionPreparationStore, _PreparationRecord
from amlas.control_plane.sft_state import SessionSftStore, SftCatalog

SESSION = "cold-producer-session"
CAMPAIGN = "00000000-0000-4000-8000-00000000aa01"
ROLE_ID = "AROA" + "X" * 17
MODEL_REF = "model_acq_" + "1" * 32
PREP_REF = "prep_" + "2" * 32


def environment(catalog):
    return {
        "AMLAS_AWS_ACCOUNT_ID": catalog.account_id,
        "AWS_REGION": catalog.region,
        "AMLAS_SESSION_BUCKET": catalog.bucket,
        "SAGEMAKER_EXECUTION_ROLE_ARN": catalog.execution_role_arn,
        "AMLAS_DEPLOYMENT_ID": catalog.deployment_id,
        "AMLAS_DEPLOYMENT_PREFIX": catalog.deployment_prefix,
        "AMLAS_AGENT_REMOTE_FETCH": "1",
        "AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI": catalog.image_uri,
        "AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT": catalog.source_commit,
        "AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST": catalog.admission_digest,
        "AMLAS_DATA_PREPARATION_IMAGE_URI": catalog.image_uri,
        "AMLAS_DATA_PREPARATION_SOURCE_COMMIT": catalog.source_commit,
        "AMLAS_DATA_PREPARATION_ADMISSION_DIGEST": catalog.admission_digest,
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_URI": catalog.exclusions_ref.uri,
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_VERSION_ID": catalog.exclusions_ref.version_id,
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_SHA256": catalog.exclusions_ref.sha256,
        "AMLAS_DATA_PREPARATION_EXCLUSIONS_BYTE_COUNT": str(catalog.exclusions_ref.byte_count),
        "AMLAS_QWEN_SFT_IMAGE_URI": catalog.image_uri,
        "AMLAS_QWEN_SFT_SOURCE_COMMIT": catalog.source_commit,
        "AMLAS_QWEN_SFT_ADMISSION_DIGEST": catalog.admission_digest,
    }


class Table(DDB):
    def _shape(self, operation, kwargs):
        from botocore.validate import validate_parameters
        from test_campaign_state import MODELS

        validate_parameters(kwargs, MODELS["dynamodb"].operation_model(operation).input_shape)
        assert kwargs["TableName"] == prep_binding.PREFIX + "-training-jobs"
        self.calls.append((operation, copy.deepcopy(kwargs)))


class Identity:
    def get_caller_identity(self):
        return {
            "Account": prep_binding.ACCOUNT,
            "UserId": ROLE_ID + ":fresh",
            "Arn": "arn:aws:sts::" + prep_binding.ACCOUNT + ":assumed-role/cold-control/fresh",
        }


def control(table, native):
    catalog = CampaignStateCatalog(
        prep_binding.ACCOUNT,
        prep_binding.REGION,
        prep_binding.DEPLOYMENT,
        prep_binding.PREFIX,
        prep_binding.PREFIX + "-training-jobs",
        "arn:aws:iam::" + prep_binding.ACCOUNT + ":role/cold-control",
        ROLE_ID,
    )
    backend = DynamoCampaignStateBackend(
        catalog, dynamodb_client=table, sts_client=Identity(), sagemaker_client=native
    )
    return backend, backend.bind_execution(campaign_id=CAMPAIGN, runtime_session_id=SESSION)


def patch_catalog(monkeypatch, manifest=None):
    import amlas.backends.adapters.model_weight_acquisition as adapter
    import amlas.control_plane.sft_state as sft_state
    import amlas.data_eng.model_acquisition_schema as schema
    import amlas.data_eng.model_weight_acquisition as core
    import amlas.data_eng.model_weight_manifest as weights

    manifest = manifest or mf.fixture_manifest()[0]
    for module in (weights, schema, adapter, core, sft_state):
        monkeypatch.setattr(module, "expected_model_weight_manifest", lambda *args: copy.deepcopy(manifest))
    return manifest


def seed(monkeypatch, tmp_path):
    import dataclasses

    base = preparation_fixture.__wrapped__(monkeypatch)
    original = base[4]
    bound = original.inspection_source
    prefix = (
        "s3://"
        + bound.bucket
        + "/sessions/"
        + bound.runtime_session_id
        + "/acquisition/"
        + bound.acquisition_job_arn.rsplit("/", 1)[-1]
        + "/"
    )
    artifact = dataclasses.replace(bound.artifact_ref, uri=prefix + "bundle/data.jsonl")
    names = {"receipt": "ACQUIRED.json", "commit": "bundle/COMMITTED.json", "lineage": "bundle/LINEAGE.json"}
    refs = tuple(
        (name, prefix + names[name], version, sha) for name, _uri, version, sha in original.manifest_refs
    )
    source = dataclasses.replace(
        original, inspection_source=dataclasses.replace(bound, artifact_ref=artifact), manifest_refs=refs
    )
    renamed = {
        bound.artifact_ref.uri: artifact.uri,
        **{old[1]: new[1] for old, new in zip(original.manifest_refs, refs, strict=True)},
    }
    for old, new in renamed.items():
        key = old.split("/", 3)[3]
        if key in base[2].values:
            base[2].values[new.split("/", 3)[3]] = base[2].values.pop(key)
    value = staged_fixture((*base[:4], source), monkeypatch)
    prep_adapter, prep_native, prep_s3, catalog, source = value
    # This producer fixture normally uses test_session; preserve its original bytes.
    global SESSION
    SESSION = source.inspection_source.runtime_session_id
    for key, val in environment(catalog).items():
        monkeypatch.setenv(key, val)
    for key, val in {
        "ACCOUNT": catalog.account_id,
        "REGION": catalog.region,
        "BUCKET": catalog.bucket,
        "ROLE": catalog.execution_role_arn,
        "IMAGE": catalog.image_uri,
        "COMMIT": catalog.source_commit,
        "DEPLOYMENT": catalog.deployment_id,
        "SESSION": SESSION,
    }.items():
        monkeypatch.setattr(mf, key, val)
    monkeypatch.setattr(sf, "ACCOUNT", catalog.account_id)
    monkeypatch.setattr(sf, "BUCKET", catalog.bucket)
    monkeypatch.setattr(sf, "SESSION", SESSION)
    manifest = patch_catalog(monkeypatch)
    import amlas.backends.adapters.model_weight_acquisition as model_module

    model_native, model_s3 = mf.Processing(), mf.S3()
    model_adapter = mf.configured_backend(
        model_module, model_native, model_s3, admission_digest=catalog.admission_digest
    )
    model_handle = model_adapter.submit(
        catalog_id=mf.CATALOG, catalog_version=mf.VERSION, runtime_session_id=SESSION
    )
    monkeypatch.setattr(mf, "bindings", lambda: copy.deepcopy(model_handle.extra["bindings"]))
    mf.publish_fixture(monkeypatch, tmp_path, model_adapter, model_handle)
    model_native.state = "Completed"
    prepared_handle = prep_adapter.submit(sources=(source,), catalog=catalog, runtime_session_id=SESSION)
    complete_outputs(value, prepared_handle)
    acquired = SessionAcquisitionStore()
    acquired.runtime_session_id = SESSION
    models = SessionModelAcquisitionStore(runtime_session_id=SESSION)
    actual_model_ref = models.add(model_adapter, model_handle)
    models._records[MODEL_REF] = models._records.pop(actual_model_ref)
    models._catalogs[(mf.CATALOG, mf.VERSION)] = MODEL_REF
    prepared = SessionPreparationStore(acquired, prep_adapter, catalog)
    prepared._records[PREP_REF] = _PreparationRecord(prep_adapter, prepared_handle)
    model_source = models._verified_sft_model_snapshot(MODEL_REF)
    prep_source = prepared._verified_sft_snapshot(PREP_REF)
    s3 = sf.S3()
    refs = [r for _, r in model_source.file_refs] + [r for _, r in model_source.manifest_refs]
    refs += [prep_source.preparation_ref, prep_source.exclusions_ref]
    refs += [r for _, r in prep_source.split_refs] + [r for _, r in prep_source.sidecar_refs]
    for ref in refs:
        key = ref.uri.split("/", 3)[3]
        raw = model_s3.objects[key][1] if key in model_s3.objects else prep_s3.values[key][0]
        s3.install(ref, raw)
    native = sf.Native()
    table = Table()
    campaign, binding = control(table, native)
    sft_catalog = SftCatalog(
        catalog.account_id,
        catalog.region,
        catalog.bucket,
        catalog.execution_role_arn,
        catalog.image_uri,
        catalog.source_commit,
        catalog.admission_digest,
        catalog.deployment_id,
        catalog.deployment_prefix,
    )
    adapter = SageMakerQwenSftBackend(sagemaker_client=native, s3_client=s3)
    sft = SessionSftStore(
        models, prepared, adapter, sft_catalog, campaign_state=campaign, execution_binding=binding
    )
    original = sft.submit(MODEL_REF, PREP_REF, "fit_v1")
    assert original["accepted"], original
    return SimpleNamespace(
        sft=sft,
        campaign=campaign,
        binding=binding,
        table=table,
        native=native,
        sft_s3=s3,
        model_native=model_native,
        model_s3=model_s3,
        prep_native=prep_native,
        prep_s3=prep_s3,
        env=environment(catalog),
        manifest=manifest,
        original=original,
    )


def export_state(value):
    def b64(raw):
        return base64.b64encode(raw).decode()

    items = {
        name: {
            key: {kind: b64(content) if kind == "B" else content for kind, content in item.items()}
            for key, item in row.items()
        }
        for name, row in value.table.items.items()
    }
    sf_objects = []
    for key, (ref, raw) in value.sft_s3.objects.items():
        sf_objects.append(
            {
                "key": key,
                "ref": {
                    "uri": ref.uri,
                    "version_id": ref.version_id,
                    "sha256": ref.sha256,
                    "byte_count": ref.byte_count,
                    "format": ref.format,
                },
                "body": b64(raw) if raw is not None else None,
            }
        )
    return {
        "env": value.env,
        "manifest": value.manifest,
        "original": value.original,
        "session": SESSION,
        "ddb": items,
        "model_request": value.model_native.requests[0],
        "model_objects": {key: [version, b64(raw)] for key, (version, raw) in value.model_s3.objects.items()},
        "prep_request": value.prep_native.creates[0],
        "prep_values": {key: [b64(raw), version] for key, (raw, version) in value.prep_s3.values.items()},
        "sft_request": value.native.calls[0],
        "sft_objects": sf_objects,
    }


def cold_observe(data):
    # Everything here is reconstructed after the parent process and all its stores died.
    from test_managed_coding_preparation import S3, SM

    from amlas import api
    from amlas.backends.adapters._util import raw_tool
    from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
    from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
    from amlas.backends.ports import CodingPreparationBackend, ModelWeightAcquisitionBackend, QwenSftBackend
    from amlas.backends.registry import REGISTRY
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
    from amlas.control_plane.producer_snapshot import ProducerSnapshotRepository, activate_sft_campaign
    from amlas.tools.qwen_sft import get_qwen_sft

    global SESSION
    SESSION = data["session"]
    os.environ.update(data["env"])
    mp = pytest.MonkeyPatch()
    for name, val in {
        "ACCOUNT": prep_binding.ACCOUNT,
        "REGION": prep_binding.REGION,
        "BUCKET": prep_binding.BUCKET,
        "SESSION": SESSION,
    }.items():
        mp.setattr(mf, name, val)
    mp.setattr(sf, "ACCOUNT", prep_binding.ACCOUNT)
    mp.setattr(sf, "BUCKET", prep_binding.BUCKET)
    mp.setattr(sf, "SESSION", SESSION)
    patch_catalog(mp, data["manifest"])
    import amlas.data_eng.commitpack_preparation as commitpack
    import amlas.data_eng.commitpack_profile as profile

    # Restore the neutral fixture source-pin override, not a model/dataset authority.
    for key, (raw_b64, _version) in data["prep_values"].items():
        raw = base64.b64decode(raw_b64)
        if key.endswith("data.jsonl") and b"old_contents" in raw:
            mp.setattr(profile, "ARTIFACT_SHA256", hashlib.sha256(raw).hexdigest())
            mp.setattr(commitpack, "SOURCE_BYTES", len(raw), raising=False)
    model_native, model_s3 = mf.Processing(), mf.S3()
    model_native.requests = [data["model_request"]]
    model_native.state = "Completed"
    model_s3.objects = {
        key: (version, base64.b64decode(raw)) for key, (version, raw) in data["model_objects"].items()
    }
    prep_native, prep_s3 = SM(), S3()
    prep_native.creates = [data["prep_request"]]
    prep_native.state = "Completed"
    prep_s3.values = {
        key: (base64.b64decode(raw), version) for key, (raw, version) in data["prep_values"].items()
    }
    native, s3 = sf.Native(), sf.S3()
    native.calls = [data["sft_request"]]
    for row in data["sft_objects"]:
        s3.install(
            DatasetArtifactRef(**row["ref"]),
            base64.b64decode(row["body"]) if row["body"] is not None else None,
        )
    table = Table()
    table.items = {
        name: {
            key: {
                kind: base64.b64decode(content) if kind == "B" else content for kind, content in val.items()
            }
            for key, val in row.items()
        }
        for name, row in data["ddb"].items()
    }
    campaign, binding = control(table, native)
    repository = ProducerSnapshotRepository(campaign, binding, dynamodb_client=table)
    context = repository.load_original(data["original"]["job_name"])
    mutations = []

    def forbidden(**kwargs):
        mutations.append(kwargs)
        raise AssertionError("cold read attempted mutation")

    for provider, names in (
        (model_native, ("create_processing_job",)),
        (prep_native, ("create_processing_job",)),
        (native, ("create_training_job",)),
        (s3, ("put_object", "create_multipart_upload", "upload_part_copy", "complete_multipart_upload")),
        (prep_s3, ("put_object", "copy_object")),
    ):
        for name in names:
            mp.setattr(provider, name, forbidden)
    original_resolve = REGISTRY.resolve
    adapters = []

    def resolve(port, *args, **kwargs):
        if port is ModelWeightAcquisitionBackend:
            adapter = ProcessingModelWeightAcquisitionBackend(
                sagemaker_client=model_native, s3_client=model_s3
            )
        elif port is CodingPreparationBackend:
            if data.get("uninjected_preparation"):
                import boto3

                class Session:
                    def __init__(self, **kwargs):
                        pass

                    def client(self, name, **kwargs):
                        return prep_native if name == "sagemaker" else prep_s3

                mp.setattr(boto3, "Session", Session)
                adapter = ProcessingCodingPreparationBackend()
            else:
                adapter = ProcessingCodingPreparationBackend(sagemaker_client=prep_native, s3_client=prep_s3)
        elif port is QwenSftBackend:
            adapter = SageMakerQwenSftBackend(sagemaker_client=native, s3_client=s3)
        else:
            return original_resolve(port, *args, **kwargs)
        adapters.append(adapter)
        return SimpleNamespace(adapter=adapter)

    mp.setattr(REGISTRY, "resolve", resolve)
    observed = {}

    class Agent:
        def __call__(self, prompt, *, invocation_state):
            observed["prompt"] = prompt
            observed["stores"] = [
                id(invocation_state[key])
                for key in ("_amlas_model_acquisition_store", "_amlas_preparation_store", "_amlas_sft_store")
            ]
            result = raw_tool(get_qwen_sft)(
                data["original"]["request_ref"],
                tool_context=SimpleNamespace(invocation_state=invocation_state),
            )
            observed["result"] = result
            return SimpleNamespace(message={"role": "assistant", "content": [{"text": json.dumps(result)}]})

    mp.setattr(ToolBindingBackend, "build_agent", lambda self, **kwargs: Agent())
    with activate_sft_campaign(context):
        response = asyncio.run(
            api.run_prompt(
                "Observe the original Qwen training job.",
                policy=ApprovalPolicy(
                    mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"], interactive=False
                ),
            )
        )
    result = observed["result"]
    assert result.get("job_name") == data["original"]["job_name"], result
    assert result["request_ref"] == data["original"]["request_ref"]
    assert not mutations and all(name == "GetItem" for name, _ in table.calls)
    assert model_s3.reads and prep_s3.calls and len(native.calls) == 1
    assert len(adapters) == 3 and all(adapter is not None for adapter in adapters)
    if data.get("uninjected_preparation"):
        assert adapters[0]._sm is None and adapters[0]._s3 is None
    return {
        "result": result,
        "response": response.message,
        "mutations": len(mutations),
        "model_reads": len(model_s3.reads),
        "prep_reads": len(prep_s3.calls),
        "sft_reads": native.reads,
        "new_adapter_count": len(adapters),
        "ddb_operations": [name for name, _ in table.calls],
    }


if __name__ == "__main__":
    print(json.dumps(cold_observe(json.loads(Path(sys.argv[1]).read_text())), sort_keys=True))
