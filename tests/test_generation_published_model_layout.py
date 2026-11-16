"""Actual acquisition layout retains nine model files and three provenance files."""

import dataclasses
import hashlib

from test_managed_generation_product import state as state
from test_qwen_sft_backend import SESSION, Native, catalog

from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend


def test_published_model_receipts_are_verified_but_not_in_native_model_channel(state):
    _store, _adapter, _reads, s3, inputs, cases = state
    parent = inputs.model.file_refs[0][1].uri.rsplit("/", 1)[0] + "/"
    names = {"commit": "COMMITTED.json", "lineage": "LINEAGE.json", "receipt": "MODEL_ACQUIRED.json"}
    manifests = []
    for role, artifact in inputs.model.manifest_refs:
        raw = s3.objects[artifact.uri.split("/", 3)[3]][1]
        moved = dataclasses.replace(artifact, uri=parent + names[role])
        s3.install(moved, raw)
        manifests.append((role, moved))
    model = dataclasses.replace(inputs.model, manifest_refs=tuple(manifests))
    prompts = dict(cases.file_refs)["PROMPTS.jsonl"]
    s3.install(prompts, b"{}\n")
    native = Native()
    backend = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    handle = backend.submit(
        model=model,
        cases=cases,
        checkpoint=None,
        catalog=catalog(),
        spec_sha256=hashlib.sha256(b"declared spec").hexdigest(),
        evaluation_id="declared-paired",
        runtime_session_id=SESSION,
    )
    assert len(native.calls) == 1
    staged = {
        name.removeprefix("base_model/"): artifact
        for name, artifact in handle.extra["staged_refs"].items()
        if name.startswith("base_model/")
    }
    original = dict(model.file_refs)
    assert set(staged) == set(original) and len(staged) == 9
    prefix = (
        "s3://"
        + catalog().bucket
        + "/sessions/"
        + SESSION
        + "/generation/"
        + handle.id
        + "/input/base_model/"
    )
    assert handle.extra["request"]["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == prefix
    for name, artifact in staged.items():
        assert artifact.uri == prefix + name and artifact.uri != original[name].uri
        assert (artifact.sha256, artifact.byte_count, artifact.format) == (
            original[name].sha256,
            original[name].byte_count,
            original[name].format,
        )
        assert artifact.version_id and artifact.version_id != original[name].version_id
    assert not set(names.values()) & set(staged)
    assert handle.extra["original_base_refs"] == original
