"""Cold normal-tool alias regressions with strict neutral SDK fixtures only."""

from __future__ import annotations

import asyncio
import base64
import copy
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_sft_producer_review_fixes import install_fresh
from test_sft_producer_snapshot import fixture

from amlas.backends.adapters._util import raw_tool
from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.control_plane.preparation_state import SessionPreparationStore
from amlas.control_plane.producer_snapshot import ProducerSnapshotRepository
from amlas.control_plane.sft_state import SessionSftStore, SftCatalog
from amlas.tools.coding_preparation import prepare_coding_dataset
from amlas.tools.qwen_sft import reload_qwen_sft, submit_qwen_sft

ACQ_REF = "acq_" + "3" * 32
SECOND_ACQ_REF = "acq_" + "1" * 32
PATH = Path(__file__).resolve()


class OriginalNative(fixture.sf.Native):
    """Strict per-original-job responses; observations never create jobs."""

    def __init__(self):
        super().__init__()
        self.states = {}

    def create_training_job(self, **kwargs):
        answer = super().create_training_job(**kwargs)
        self.states[kwargs["TrainingJobName"]] = "InProgress"
        return answer

    def describe_training_job(self, **kwargs):
        selected = next(row for row in self.calls if row["TrainingJobName"] == kwargs["TrainingJobName"])
        calls, state = self.calls, self.state
        try:
            self.calls = [selected]
            self.state = self.states[selected["TrainingJobName"]]
            return super().describe_training_job(**kwargs)
        finally:
            self.calls, self.state = calls, state

    def list_tags(self, **kwargs):
        name = kwargs["ResourceArn"].rsplit("/", 1)[-1]
        selected = next(row for row in self.calls if row["TrainingJobName"] == name)
        return {"Tags": copy.deepcopy(selected["Tags"])}


def policy():
    return ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow", "data_preparation_flow"], interactive=False
    )


def patch_alias_catalog(monkeypatch, manifest=None):
    # Load before patching weights so cleanup restores the actual source function.
    import amlas.training.qwen_sft as training_core

    selected = fixture.patch_catalog(monkeypatch, manifest)
    monkeypatch.setattr(
        training_core, "expected_model_weight_manifest", lambda *args: copy.deepcopy(selected)
    )
    return selected


def complete_multisource_outputs(value, handle):
    """Build actual CommitPack outputs for two independently addressed sources."""
    from test_coding_preparation_worker import PROOF
    from test_commitpack_profile import row
    from test_managed_coding_preparation import encoded, sha

    from amlas.data_eng import commitpack_preparation as prep
    from amlas.data_eng import commitpack_profile as profile

    parsed = []
    for source in handle.extra["sources"]:
        ref = source.inspection_source.artifact_ref
        for index in range(100):
            pin = profile.CommitPackSource(
                source.repo_id, source.revision, source.filename, ref.uri, ref.version_id, ref.sha256, index
            )
            parsed.append(prep.CommitPackParsedRow(row(index + 1), pin, sha(encoded(row(index + 1)))))
    result = prep.build_commitpack_preparation(
        parsed,
        exclusion_policy={
            "schema_version": 1,
            "repositories": [],
            "commits": [],
            "canonical_sha256": [],
            "input_sha256": [],
            "code_pair_sha256": [],
        },
    )
    payloads = {
        name + ".jsonl": b"".join(encoded(record) for record in result["splits"][name])
        for name in prep.SPLITS
    }
    for name in ("membership", "lineage"):
        payloads[name + ".jsonl"] = b"".join(encoded(record) for record in result[name])
    payloads["REPORT.json"] = encoded(result["report"])
    files = {}
    for name, raw in payloads.items():
        value[2].add(handle.extra["output_prefix"] + name, raw)
        files[name] = {
            "byte_count": len(raw),
            "sha256": sha(raw),
            "format": "json" if name.endswith(".json") else "jsonl",
            "record_count": 1 if name.endswith(".json") else len(raw.splitlines()),
        }
    value[2].add(
        handle.extra["output_prefix"] + "COMMITTED.json",
        encoded(
            {
                "schema_version": 2,
                **handle.extra["header"],
                "privilege_profile": "sagemaker_native_ceiling_v1",
                "selection_sha256": handle.extra["selection_sha256"],
                "exclusions_sha256": handle.extra["exclusions_sha256"],
                "files": files,
                "counts": result["report"]["counts"],
                "coverage": prep.COVERAGE,
                "guest_evidence": {
                    "privilege_proof": PROOF,
                    "credential_observation": "env_and_home_file_presence",
                    "network_observation": "fixed_public_and_metadata_tcp_denial",
                },
            }
        ),
    )
    value[1].state = "Completed"


def normal_seed(monkeypatch, tmp_path, *, multiple_sources=False):
    """Use existing source/SDK fixtures, obtaining aliases through real tools."""
    base = fixture.preparation_fixture.__wrapped__(monkeypatch)
    source, bound = base[4], base[4].inspection_source
    prefix = (
        f"s3://{bound.bucket}/sessions/{bound.runtime_session_id}/acquisition/"
        + bound.acquisition_job_arn.rsplit("/", 1)[-1]
        + "/"
    )
    artifact = dataclasses.replace(bound.artifact_ref, uri=prefix + "bundle/data.jsonl")
    names = {"receipt": "ACQUIRED.json", "commit": "bundle/COMMITTED.json", "lineage": "bundle/LINEAGE.json"}
    refs = tuple(
        (name, prefix + names[name], version, sha) for name, _uri, version, sha in source.manifest_refs
    )
    source = dataclasses.replace(
        source, inspection_source=dataclasses.replace(bound, artifact_ref=artifact), manifest_refs=refs
    )
    renamed = {
        bound.artifact_ref.uri: artifact.uri,
        **{old[1]: new[1] for old, new in zip(base[4].manifest_refs, refs, strict=True)},
    }
    for old, new in renamed.items():
        key = old.split("/", 3)[3]
        if key in base[2].values:
            base[2].values[new.split("/", 3)[3]] = base[2].values.pop(key)
    value = fixture.staged_fixture((*base[:4], source), monkeypatch)
    prep_adapter, prep_native, prep_s3, catalog, source = value
    fixture.SESSION = source.inspection_source.runtime_session_id
    for key, val in fixture.environment(catalog).items():
        monkeypatch.setenv(key, val)
    for key, val in {
        "ACCOUNT": catalog.account_id,
        "REGION": catalog.region,
        "BUCKET": catalog.bucket,
        "ROLE": catalog.execution_role_arn,
        "IMAGE": catalog.image_uri,
        "COMMIT": catalog.source_commit,
        "DEPLOYMENT": catalog.deployment_id,
        "SESSION": fixture.SESSION,
    }.items():
        monkeypatch.setattr(fixture.mf, key, val)
    for key, val in {
        "ACCOUNT": catalog.account_id,
        "BUCKET": catalog.bucket,
        "SESSION": fixture.SESSION,
        "SOURCE": catalog.source_commit,
    }.items():
        monkeypatch.setattr(fixture.sf, key, val)
    manifest = patch_alias_catalog(monkeypatch)
    monkeypatch.setattr(fixture.sf, "expected_model_weight_manifest", lambda *args: copy.deepcopy(manifest))
    import amlas.backends.adapters.model_weight_acquisition as model_module

    model_native, model_s3 = fixture.mf.Processing(), fixture.mf.S3()
    model_adapter = fixture.mf.configured_backend(
        model_module, model_native, model_s3, admission_digest=catalog.admission_digest
    )
    model_handle = model_adapter.submit(
        catalog_id=fixture.mf.CATALOG, catalog_version=fixture.mf.VERSION, runtime_session_id=fixture.SESSION
    )
    monkeypatch.setattr(fixture.mf, "bindings", lambda: copy.deepcopy(model_handle.extra["bindings"]))
    fixture.mf.publish_fixture(monkeypatch, tmp_path, model_adapter, model_handle)
    model_native.state = "Completed"

    acquired = SessionAcquisitionStore()
    acquired.runtime_session_id = fixture.SESSION

    sources = {ACQ_REF: source}
    if multiple_sources:
        original_bound = source.inspection_source
        original_prefix = original_bound.artifact_ref.uri.rsplit("bundle/", 1)[0]
        second_job = original_bound.deployment_prefix + "-second-acquisition"
        second_prefix = original_prefix.rsplit("/", 2)[0] + "/" + second_job + "/"
        second_artifact = dataclasses.replace(
            original_bound.artifact_ref,
            uri=original_bound.artifact_ref.uri.replace(original_prefix, second_prefix, 1),
        )
        second_refs = tuple(
            (name, uri.replace(original_prefix, second_prefix, 1), version, sha)
            for name, uri, version, sha in source.manifest_refs
        )
        second_source = dataclasses.replace(
            source,
            inspection_source=dataclasses.replace(
                original_bound,
                artifact_ref=second_artifact,
                acquisition_job_arn=original_bound.acquisition_job_arn.rsplit("/", 1)[0] + "/" + second_job,
            ),
            manifest_refs=second_refs,
        )
        for old_uri, new_uri in [
            (original_bound.artifact_ref.uri, second_artifact.uri),
            *((old[1], new[1]) for old, new in zip(source.manifest_refs, second_refs, strict=True)),
        ]:
            prep_s3.values[new_uri.split("/", 3)[3]] = copy.deepcopy(prep_s3.values[old_uri.split("/", 3)[3]])
        sources[SECOND_ACQ_REF] = second_source

    # Typed upstream source fixtures; the normal preparation tool owns submission.
    def original_source(reference):
        return sources[reference]

    monkeypatch.setattr(acquired, "_verified_preparation_source", original_source)
    prepared = SessionPreparationStore(acquired, prep_adapter, catalog)
    prep_result = raw_tool(prepare_coding_dataset)(
        acquisition_request_refs=list(reversed(sorted(sources))),
        tool_context=SimpleNamespace(
            invocation_state={"_amlas_approval_policy": policy(), "_amlas_preparation_store": prepared}
        ),
    )
    assert prep_result["accepted"], prep_result
    prep_ref = prep_result["request_ref"]
    prepared_handle = prepared._records[prep_ref].handle
    if multiple_sources:
        complete_multisource_outputs(value, prepared_handle)
    else:
        fixture.complete_outputs(value, prepared_handle)
    assert prepared._submissions == {tuple(sorted(sources)): prep_ref} and len(prep_native.creates) == 1

    models = SessionModelAcquisitionStore(runtime_session_id=fixture.SESSION)
    model_ref = models.add(model_adapter, model_handle)
    model_source = models._verified_sft_model_snapshot(model_ref)
    prep_source = prepared._verified_sft_snapshot(prep_ref)
    s3 = fixture.sf.S3()
    all_refs = [r for _, r in model_source.file_refs + model_source.manifest_refs]
    all_refs += [prep_source.preparation_ref, prep_source.exclusions_ref]
    all_refs += [r for _, r in prep_source.split_refs + prep_source.sidecar_refs]
    for ref in all_refs:
        key = ref.uri.split("/", 3)[3]
        raw = model_s3.objects[key][1] if key in model_s3.objects else prep_s3.values[key][0]
        s3.install(ref, raw)
    native, table = OriginalNative(), fixture.Table()
    campaign, binding = fixture.control(table, native)
    sft_catalog = SftCatalog(
        *(
            getattr(catalog, name)
            for name in (
                "account_id",
                "region",
                "bucket",
                "execution_role_arn",
                "image_uri",
                "source_commit",
                "admission_digest",
                "deployment_id",
                "deployment_prefix",
            )
        )
    )
    adapter = SageMakerQwenSftBackend(sagemaker_client=native, s3_client=s3)
    sft = SessionSftStore(
        models, prepared, adapter, sft_catalog, campaign_state=campaign, execution_binding=binding
    )
    original = raw_tool(submit_qwen_sft)(
        model_request_ref=model_ref,
        preparation_request_ref=prep_ref,
        tool_context=SimpleNamespace(
            invocation_state={"_amlas_approval_policy": policy(), "_amlas_sft_store": sft}
        ),
    )
    assert original["accepted"] and len(native.calls) == 1, original
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
        env=fixture.environment(catalog),
        manifest=manifest,
        original=original,
        model_ref=model_ref,
        prep_ref=prep_ref,
        acquisition_refs=list(reversed(sorted(sources))),
    )


def export(value, mode):
    data = fixture.export_state(value)
    data.update(
        mode=mode,
        acquisition_ref=ACQ_REF,
        acquisition_refs=value.acquisition_refs,
        model_ref=value.model_ref,
        prep_ref=value.prep_ref,
        sft_requests=value.native.calls,
        sft_states=value.native.states,
        train_ref=value.original["request_ref"],
    )
    if mode == "reload":
        train_handle = value.sft._records[value.original["request_ref"]]
        fixture.sf.install_output(value.sft._adapter, value.native, value.sft_s3, train_handle)
        value.native.states[train_handle.id] = "Completed"
        reloaded = raw_tool(reload_qwen_sft)(
            request_ref=data["train_ref"],
            tool_context=SimpleNamespace(
                invocation_state={"_amlas_approval_policy": policy(), "_amlas_sft_store": value.sft}
            ),
        )
        if reloaded.get("accepted") is not True:
            print("NEUTRAL_RELOAD_SEED_RESPONSE " + json.dumps(reloaded, sort_keys=True))
        assert reloaded["accepted"] and len(value.native.calls) == 2, reloaded
        value.original = reloaded
        data = {
            **fixture.export_state(value),
            **{
                key: data[key]
                for key in (
                    "mode",
                    "acquisition_ref",
                    "acquisition_refs",
                    "model_ref",
                    "prep_ref",
                    "train_ref",
                )
            },
            "sft_requests": value.native.calls,
            "sft_states": value.native.states,
        }
    repo = ProducerSnapshotRepository(value.campaign, value.binding, dynamodb_client=value.table)
    publication = repo.publish_restore_aliases_from_store(value.sft, value.original["request_ref"])
    # Retain the publisher acknowledgement separately from the selected row.
    # This fixture transport does not qualify a production server handoff.
    data["producer_snapshot_sha256"] = publication["snapshot_sha256"]
    final = fixture.export_state(value)
    data["ddb"] = final["ddb"]
    return data


def cold_repeat(data):
    """Fresh adapters/stores and normal API/tool execution; SDKs remain strict fakes."""
    from test_managed_coding_preparation import S3, SM

    from amlas import api
    from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
    from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
    from amlas.backends.ports import CodingPreparationBackend, ModelWeightAcquisitionBackend, QwenSftBackend
    from amlas.backends.registry import REGISTRY
    from amlas.control_plane.producer_snapshot import activate_sft_campaign

    fixture.SESSION = data["session"]
    os.environ.update(data["env"])
    mp = pytest.MonkeyPatch()
    for key, val in {
        "ACCOUNT": fixture.prep_binding.ACCOUNT,
        "REGION": fixture.prep_binding.REGION,
        "BUCKET": fixture.prep_binding.BUCKET,
        "SESSION": fixture.SESSION,
    }.items():
        mp.setattr(fixture.mf, key, val)
    for key, val in {
        "ACCOUNT": fixture.prep_binding.ACCOUNT,
        "BUCKET": fixture.prep_binding.BUCKET,
        "SESSION": fixture.SESSION,
        "SOURCE": data["env"]["AMLAS_QWEN_SFT_SOURCE_COMMIT"],
    }.items():
        mp.setattr(fixture.sf, key, val)
    patch_alias_catalog(mp, data["manifest"])
    import amlas.data_eng.commitpack_preparation as commitpack
    import amlas.data_eng.commitpack_profile as profile

    for key, (raw_b64, _version) in data["prep_values"].items():
        raw = base64.b64decode(raw_b64)
        if key.endswith("data.jsonl") and b"old_contents" in raw:
            mp.setattr(profile, "ARTIFACT_SHA256", hashlib.sha256(raw).hexdigest())
            mp.setattr(commitpack, "SOURCE_BYTES", len(raw), raising=False)
    model_native, model_s3 = fixture.mf.Processing(), fixture.mf.S3()
    model_native.requests, model_native.state = [data["model_request"]], "Completed"
    model_s3.objects = {
        key: (version, base64.b64decode(raw)) for key, (version, raw) in data["model_objects"].items()
    }
    prep_native, prep_s3 = SM(), S3()
    prep_native.creates, prep_native.state = [data["prep_request"]], "Completed"
    prep_s3.values = {
        key: (base64.b64decode(raw), version) for key, (raw, version) in data["prep_values"].items()
    }
    native, s3 = OriginalNative(), fixture.sf.S3()
    native.calls, native.states = data["sft_requests"], data["sft_states"]
    for row in data["sft_objects"]:
        s3.install(
            DatasetArtifactRef(**row["ref"]),
            base64.b64decode(row["body"]) if row["body"] is not None else None,
        )
    table = fixture.Table()
    table.items = {
        name: {
            key: {
                kind: base64.b64decode(content) if kind == "B" else content for kind, content in val.items()
            }
            for key, val in row.items()
        }
        for name, row in data["ddb"].items()
    }
    campaign, binding = fixture.control(table, native)
    context = ProducerSnapshotRepository(campaign, binding, dynamodb_client=table).load_original(
        data["original"]["job_name"], expected_snapshot_sha256=data["producer_snapshot_sha256"]
    )
    original_resolve = REGISTRY.resolve
    fresh = []

    def resolve(port, *args, **kwargs):
        if port is ModelWeightAcquisitionBackend:
            adapter = ProcessingModelWeightAcquisitionBackend(
                sagemaker_client=model_native, s3_client=model_s3
            )
        elif port is CodingPreparationBackend:
            adapter = ProcessingCodingPreparationBackend(sagemaker_client=prep_native, s3_client=prep_s3)
        elif port is QwenSftBackend:
            adapter = SageMakerQwenSftBackend(sagemaker_client=native, s3_client=s3)
        else:
            return original_resolve(port, *args, **kwargs)
        fresh.append(adapter)
        return SimpleNamespace(adapter=adapter)

    mp.setattr(REGISTRY, "resolve", resolve)
    observed = []

    class Agent:
        def __call__(self, prompt, *, invocation_state):
            context = SimpleNamespace(invocation_state=invocation_state)
            for _ in range(2):
                if data["mode"] == "prep":
                    answer = raw_tool(prepare_coding_dataset)(
                        acquisition_request_refs=data["acquisition_refs"], tool_context=context
                    )
                else:
                    answer = raw_tool(reload_qwen_sft)(request_ref=data["train_ref"], tool_context=context)
                observed.append(answer)
            return SimpleNamespace(message={"role": "assistant", "content": [{"text": json.dumps(observed)}]})

    mp.setattr(fixture.ToolBindingBackend, "build_agent", lambda self, **kwargs: Agent())
    with activate_sft_campaign(context):
        asyncio.run(api.run_prompt("Repeat the original preparation or reload request.", policy=policy()))
    return {
        "results": observed,
        "preparation_creates": len(prep_native.creates),
        "training_creates": len(native.calls),
        "fresh_adapter_count": len(fresh),
    }


@pytest.mark.parametrize("mode", ["prep", "reload"])
def test_fresh_process_normal_repeat_preserves_original_alias_and_create_count(monkeypatch, tmp_path, mode):
    original = normal_seed(monkeypatch, tmp_path)
    data = export(original, mode)
    path = tmp_path / "normal-alias-state.json"
    path.write_text(json.dumps(data))
    child = subprocess.run(
        [sys.executable, str(PATH), str(path)],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "AWS_EC2_METADATA_DISABLED": "true", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout.splitlines()[-1])
    expected = data["prep_ref"] if mode == "prep" else data["original"]["request_ref"]
    assert [row.get("request_ref") for row in result["results"]] == [expected, expected], result
    assert all(row.get("reused") is True for row in result["results"]), result
    assert result["preparation_creates"] == 1
    assert result["training_creates"] == (1 if mode == "prep" else 2)
    assert result["fresh_adapter_count"] == 3


def test_original_reload_seed_is_independent_of_training_core_import_order(monkeypatch, tmp_path):
    import amlas.training.qwen_sft as training_core  # Import before the source fixture.

    original = normal_seed(monkeypatch, tmp_path)
    verify = original.sft._adapter._verified_run_binding

    def observed_validation(**kwargs):
        try:
            return verify(**kwargs)
        except ValueError as error:
            print("NEUTRAL_RELOAD_SEED_VALIDATION " + str(error))
            raise

    monkeypatch.setattr(original.sft._adapter, "_verified_run_binding", observed_validation)
    data = export(original, "reload")
    assert training_core.expected_model_weight_manifest("qwen3_5_9b", "1") == original.manifest
    assert data["original"]["request_ref"] != data["train_ref"]
    assert len(original.prep_native.creates) == 1 and len(original.native.calls) == 2


def test_two_source_normal_ordering_retains_exact_alias_to_source_indices(monkeypatch, tmp_path):
    original = normal_seed(monkeypatch, tmp_path, multiple_sources=True)
    data = export(original, "prep")
    row = next(row for key, row in original.table.items.items() if "-sft-producers-v2-" in key)
    body = json.loads(row["snapshot"]["B"])
    assert body["preparation_selection"]["acquisitions"] == [
        {"request_ref": SECOND_ACQ_REF, "source_index": 0},
        {"request_ref": ACQ_REF, "source_index": 1},
    ]
    actual_sources = original.sft._prepared._records[original.prep_ref].handle.extra["sources"]
    assert [
        source["inspection_source"]["artifact_ref"]["uri"] for source in body["preparation"]["sources"]
    ] == [source.inspection_source.artifact_ref.uri for source in actual_sources]
    assert (
        actual_sources[0].inspection_source.artifact_ref.uri
        != actual_sources[1].inspection_source.artifact_ref.uri
    )
    path = tmp_path / "two-source-alias-state.json"
    path.write_text(json.dumps(data))
    child = subprocess.run(
        [sys.executable, str(PATH), str(path)],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "AWS_EC2_METADATA_DISABLED": "true", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout.splitlines()[-1])
    assert [answer.get("request_ref") for answer in result["results"]] == [data["prep_ref"]] * 2, result
    assert all(answer.get("reused") is True for answer in result["results"]), result
    assert result["preparation_creates"] == result["training_creates"] == 1
    assert result["fresh_adapter_count"] == 3


def test_valid_alias_coherent_rehash_refuses_alias_complete_graph(monkeypatch, tmp_path):
    from amlas.control_plane.producer_snapshot import hydrate_sft_campaign

    original = normal_seed(monkeypatch, tmp_path)
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    publication = repo.publish_restore_aliases_from_store(original.sft, original.original["request_ref"])
    independently_retained_sha = publication["snapshot_sha256"]
    row = original.table.items[publication["record_key"]]
    value = json.loads(row["snapshot"]["B"])
    foreign_ref = "acq_" + "4" * 32
    value["preparation_selection"]["acquisitions"][0]["request_ref"] = foreign_ref
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    row["snapshot"]["B"], row["snapshot_sha256"]["S"] = raw, hashlib.sha256(raw).hexdigest()
    assert row["snapshot_sha256"]["S"] != independently_retained_sha
    install_fresh(original, monkeypatch)
    try:
        context = repo.load_original(
            original.original["job_name"], expected_snapshot_sha256=independently_retained_sha
        )
        _acquired, _models, prepared, _sft = hydrate_sft_campaign(context, policy=policy())
    except ValueError:
        return
    assert len(original.native.calls) == len(original.prep_native.creates) == 1
    pytest.fail(
        "coherently replaced valid acquisition alias exposed an alias-complete graph: "
        f"{prepared._submissions!r}; original_commitment={independently_retained_sha}; "
        f"changed_row_sha256={row['snapshot_sha256']['S']}"
    )


@pytest.mark.parametrize("replace_alias", [False, True])
def test_present_schema2_without_independent_commitment_never_uses_row_selfhash(
    monkeypatch, tmp_path, replace_alias
):
    original = normal_seed(monkeypatch, tmp_path)
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    repo.publish_from_store(original.sft, original.original["request_ref"])
    publication = repo.publish_restore_aliases_from_store(original.sft, original.original["request_ref"])
    if replace_alias:
        row = original.table.items[publication["record_key"]]
        body = json.loads(row["snapshot"]["B"])
        body["preparation_selection"]["acquisitions"][0]["request_ref"] = "acq_" + "4" * 32
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        row["snapshot"]["B"], row["snapshot_sha256"]["S"] = raw, hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        repo.load_original(original.original["job_name"])
    assert len(original.native.calls) == len(original.prep_native.creates) == 1


@pytest.mark.parametrize(
    "expected",
    [None, "", "A" * 64, "0" * 63, True, b"0" * 64, "0" * 64],
    ids=["missing", "empty", "uppercase", "short", "boolean", "bytes", "unequal"],
)
def test_schema2_requires_exact_independent_original_commitment(monkeypatch, tmp_path, expected):
    original = normal_seed(monkeypatch, tmp_path)
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    repo.publish_from_store(original.sft, original.original["request_ref"])
    repo.publish_restore_aliases_from_store(original.sft, original.original["request_ref"])
    with pytest.raises(ValueError):
        repo.load_original(original.original["job_name"], expected_snapshot_sha256=expected)
    assert len(original.native.calls) == len(original.prep_native.creates) == 1


def test_schema2_commitment_cannot_select_absent_row_or_downgrade_to_schema1(monkeypatch, tmp_path):
    original = normal_seed(monkeypatch, tmp_path)
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    repo.publish_from_store(original.sft, original.original["request_ref"])
    publication = repo.publish_restore_aliases_from_store(original.sft, original.original["request_ref"])
    independently_retained_sha = publication["snapshot_sha256"]
    original.table.items.pop(publication["record_key"])
    with pytest.raises(ValueError):
        repo.load_original(original.original["job_name"], expected_snapshot_sha256=independently_retained_sha)
    legacy = repo.load_original(original.original["job_name"])
    assert legacy.original_sft_ref == original.original["request_ref"]
    assert len(original.native.calls) == len(original.prep_native.creates) == 1


def test_other_original_snapshot_commitment_cannot_select_current_operation(monkeypatch, tmp_path):
    original = normal_seed(monkeypatch, tmp_path)
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    train = repo.publish_restore_aliases_from_store(original.sft, original.original["request_ref"])
    export(original, "reload")
    with pytest.raises(ValueError):
        repo.load_original(original.original["job_name"], expected_snapshot_sha256=train["snapshot_sha256"])
    assert len(original.native.calls) == 2 and len(original.prep_native.creates) == 1


@pytest.mark.parametrize("case", ["missing", "duplicate", "case_bound"])
def test_alias_complete_publication_refuses_missing_or_noncanonical_original_selection(
    monkeypatch, tmp_path, case
):
    original = normal_seed(monkeypatch, tmp_path)
    prepared = original.sft._prepared
    if case == "missing":
        prepared._submissions.clear()
    elif case == "duplicate":
        prepared._submissions[(ACQ_REF, ACQ_REF)] = prepared._submissions.pop((ACQ_REF,))
    else:
        prepared._submissions[((ACQ_REF,), ("unrestored-case",))] = prepared._submissions.pop((ACQ_REF,))
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    with pytest.raises(ValueError):
        repo.publish_restore_aliases_from_store(original.sft, original.original["request_ref"])
    assert not any("-sft-producers-v2-" in key for key in original.table.items)
    assert len(original.native.calls) == 1 and len(original.prep_native.creates) == 1


@pytest.mark.parametrize("field", ["index", "duplicate", "missing", "foreign_alias"])
def test_coherently_rehashed_schema2_alias_drift_never_downgrades_to_valid_schema1(
    monkeypatch, tmp_path, field
):
    original = normal_seed(monkeypatch, tmp_path)
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    repo.publish_from_store(original.sft, original.original["request_ref"])
    selected = repo.publish_restore_aliases_from_store(original.sft, original.original["request_ref"])
    row = original.table.items[selected["record_key"]]
    value = json.loads(row["snapshot"]["B"])
    if field == "index":
        value["preparation_selection"]["acquisitions"][0]["source_index"] = 1
    elif field == "duplicate":
        value["preparation_selection"]["acquisitions"] *= 2
    elif field == "missing":
        value.pop("preparation_selection")
    else:
        value["preparation_selection"]["acquisitions"][0]["request_ref"] = "foreign"
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    row["snapshot"]["B"], row["snapshot_sha256"]["S"] = raw, hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        repo.load_original(
            original.original["job_name"], expected_snapshot_sha256=selected["snapshot_sha256"]
        )
    assert len(original.native.calls) == 1


@pytest.mark.parametrize("field", ["missing", "alias", "digest", "name"])
def test_rehashed_reload_parent_drift_refuses_before_graph_exposure(monkeypatch, tmp_path, field):
    original = normal_seed(monkeypatch, tmp_path)
    export(original, "reload")
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    selected = repo.publish_restore_aliases_from_store(original.sft, original.original["request_ref"])
    row = original.table.items[selected["record_key"]]
    value = json.loads(row["snapshot"]["B"])
    if field == "missing":
        value["training_parent"] = None
    else:
        key = {"alias": "request_ref", "digest": "operation_digest", "name": "native_job_name"}[field]
        value["training_parent"][key] = "sft_" + "0" * 32 if field == "alias" else "0" * 64
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    row["snapshot"]["B"], row["snapshot_sha256"]["S"] = raw, hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        repo.load_original(
            original.original["job_name"], expected_snapshot_sha256=selected["snapshot_sha256"]
        )
    assert len(original.native.calls) == 2


@pytest.mark.parametrize("field", ["key", "record", "source_link"])
def test_alias_publication_never_invents_missing_original_train_parent(monkeypatch, tmp_path, field):
    original = normal_seed(monkeypatch, tmp_path)
    data = export(original, "reload")
    train_ref = data["train_ref"]
    if field == "key":
        original.sft._keys.pop((train_ref, "reload"))
    elif field == "record":
        original.sft._records.pop(train_ref)
    else:
        original.sft._source_refs[train_ref] = ("model_acq_" + "0" * 32, original.prep_ref)
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    with pytest.raises(ValueError):
        repo.publish_restore_aliases_from_store(original.sft, original.original["request_ref"])
    assert len(original.native.calls) == 2


def test_revocation_between_parent_and_selected_reload_exposes_no_graph(monkeypatch, tmp_path):
    from amlas.control_plane.producer_snapshot import hydrate_sft_campaign

    original = normal_seed(monkeypatch, tmp_path)
    data = export(original, "reload")
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    context = repo.load_original(
        original.original["job_name"], expected_snapshot_sha256=data["producer_snapshot_sha256"]
    )
    current = policy()
    describe = original.native.describe_training_job
    train_name = original.sft._records[data["train_ref"]].id

    def revoked(**kwargs):
        result = describe(**kwargs)
        if kwargs["TrainingJobName"] == train_name:
            current.allowlist.clear()
        return result

    monkeypatch.setattr(original.native, "describe_training_job", revoked)
    install_fresh(original, monkeypatch)
    with pytest.raises(ValueError):
        hydrate_sft_campaign(context, policy=current)
    assert len(original.native.calls) == 2 and len(original.prep_native.creates) == 1


def test_legacy_projection_still_loads_descriptively_without_claiming_aliases(monkeypatch, tmp_path):
    from amlas.control_plane.producer_snapshot import hydrate_sft_campaign

    original = normal_seed(monkeypatch, tmp_path)
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    selected = repo.publish_from_store(original.sft, original.original["request_ref"])
    legacy = original.table.items[selected["record_key"]]["snapshot"]["B"]
    assert json.loads(legacy)["schema_version"] == 1
    context = repo.load_original(original.original["job_name"])
    install_fresh(original, monkeypatch)
    _acquired, _models, prepared, sft = hydrate_sft_campaign(context, policy=policy())
    assert prepared._submissions == {} and context.original_sft_ref in sft._records
    assert original.table.items[selected["record_key"]]["snapshot"]["B"] == legacy
    assert not any("-sft-producers-v2-" in key for key in original.table.items)
    assert len(original.native.calls) == 1 and len(original.prep_native.creates) == 1


if __name__ == "__main__":
    print(json.dumps(cold_repeat(json.loads(Path(sys.argv[1]).read_text())), sort_keys=True))
