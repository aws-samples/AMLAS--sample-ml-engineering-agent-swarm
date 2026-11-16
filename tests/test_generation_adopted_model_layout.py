"""An adopted verified model pack keeps its adoption record beside the nine model files.

`ModelPackAdoptionProvider.adopt` writes `adoption.json` into the same session prefix as the nine copied model
files and the three copied provenance files. The generation submit lists that prefix and compares it with the
files it verifies, so an adopted base model must count its own adoption record as part of the exact inventory.
"""

import dataclasses
import hashlib
import json

import pytest
from test_managed_generation_product import state as state
from test_qwen_sft_backend import SESSION, Native, catalog

from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.sft_state import VerifiedSftAdoptedModelSource

NAMES = {"commit": "COMMITTED.json", "lineage": "LINEAGE.json", "receipt": "MODEL_ACQUIRED.json"}


def _adopted(inputs, s3, *, record_in_prefix=True):
    parent = inputs.model.file_refs[0][1].uri.rsplit("/", 1)[0] + "/"
    manifests = []
    for role, artifact in inputs.model.manifest_refs:
        raw = s3.objects[artifact.uri.split("/", 3)[3]][1]
        moved = dataclasses.replace(artifact, uri=parent + NAMES[role])
        s3.install(moved, raw)
        manifests.append((role, moved))
    raw = json.dumps({"origin": "verified_pack_adoption"}, sort_keys=True).encode()
    uri = (parent if record_in_prefix else parent + "elsewhere/") + "adoption.json"
    record = DatasetArtifactRef(uri, "adoption-version-1", hashlib.sha256(raw).hexdigest(), len(raw), "json")
    s3.install(record, raw)
    return VerifiedSftAdoptedModelSource(
        SESSION,
        "qwen3_5_9b",
        "1",
        inputs.model,
        inputs.model.file_refs,
        tuple(manifests),
        record,
    )


def _submit(backend, model, cases):
    return backend.submit(
        model=model,
        cases=cases,
        checkpoint=None,
        catalog=catalog(),
        spec_sha256=hashlib.sha256(b"declared spec").hexdigest(),
        evaluation_id="declared-paired",
        runtime_session_id=SESSION,
    )


def test_adopted_base_model_counts_its_adoption_record_in_the_exact_inventory(state):
    _store, _adapter, _reads, s3, inputs, cases = state
    model = _adopted(inputs, s3)
    s3.install(dict(cases.file_refs)["PROMPTS.jsonl"], b"{}\n")
    native = Native()
    handle = _submit(SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3), model, cases)
    assert len(native.calls) == 1
    staged = {
        name.removeprefix("base_model/")
        for name in handle.extra["staged_refs"]
        if name.startswith("base_model/")
    }
    assert staged == set(dict(model.file_refs)) and len(staged) == 9
    assert "adoption.json" not in staged


def test_adopted_base_model_with_an_adoption_record_outside_its_prefix_is_still_refused(state):
    _store, _adapter, _reads, s3, inputs, cases = state
    model = _adopted(inputs, s3, record_in_prefix=False)
    s3.install(dict(cases.file_refs)["PROMPTS.jsonl"], b"{}\n")
    native = Native()
    backend = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    with pytest.raises(ValueError):
        _submit(backend, model, cases)
    assert native.calls == []
