"""Fresh publisher-view continuity only; no native or cold recovery authority.

The strict external port contains only 164 declared harmless prompts. The actual
completion driver must read the pinned prompt bytes and publish both outputs
before the isolated pair boundary.
No model, SDK agent, training job, case oracle or captured run is executed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import model_acquisition_fixtures as model_fixture
import normal_sft_cold_fixture as cold_fixture
import pytest
import test_paired_coding_backend as pair_fixture
from normal_sft_cold_fixture import NativeObjects
from test_normal_run_checkpoint import NativePorts
from test_qwen_coding_generation_managed import binding
from test_qwen_sft_backend import catalog, encoded

from amlas.backends.types import DatasetArtifactRef, JobHandle
from amlas.control_plane.generation_state import fixed_generation_request


def test_generation_publisher_uses_the_same_fresh_versioned_prompt_inventory(monkeypatch):
    monkeypatch.setattr(pair_fixture, "ACCOUNT", model_fixture.ACCOUNT)
    ports = NativePorts()
    objects = NativeObjects(ports)
    prompts = [
        {"case_id": "HumanEval/" + str(index), "prompt": "Return the declared integer."}
        for index in range(164)
    ]
    raw = b"".join(encoded(prompt) for prompt in prompts)
    key = "sessions/declared-publisher/case-freeze/declared-case/output/PROMPTS.jsonl"
    ref = DatasetArtifactRef(
        "s3://" + model_fixture.BUCKET + "/" + key,
        "declared-prompts-version",
        hashlib.sha256(raw).hexdigest(),
        len(raw),
        "jsonl",
    )
    # Fresh external response DATA, not a captured checkpoint or authority row.
    ports.objects[(key, ref.version_id)] = raw
    request = {
        "Bucket": model_fixture.BUCKET,
        "Key": key,
        "VersionId": ref.version_id,
        "ExpectedBucketOwner": model_fixture.ACCOUNT,
    }
    reply = objects.get_object(**request)
    with reply["Body"] as body:
        observed = body.read()
    assert reply["VersionId"] == ref.version_id and reply["ContentLength"] == ref.byte_count
    assert observed == raw and hashlib.sha256(observed).hexdigest() == ref.sha256
    assert objects.head_object(**request) == {"VersionId": ref.version_id, "ContentLength": ref.byte_count}
    inventory = objects.list_objects_v2(
        Bucket=model_fixture.BUCKET,
        Prefix=key.rsplit("/", 1)[0] + "/",
        ExpectedBucketOwner=model_fixture.ACCOUNT,
        MaxKeys=2,
    )
    assert inventory == {"IsTruncated": False, "Contents": [{"Key": key, "Size": len(raw)}]}

    generation_binding = binding()
    generation_binding["source_commit"] = catalog().source_commit
    generation_binding["prompts_ref"] = dataclasses.asdict(ref)
    run_raw = encoded(generation_binding)
    run_ref = objects.install(
        "sessions/declared-publisher/generation/owned-gen-declared/input/bindings/RUN.json", run_raw
    )
    native_request = fixed_generation_request(
        catalog(), "declared-publisher", "owned-gen-declared", run_ref, "base"
    )
    handle = JobHandle(
        "owned-gen-declared",
        "coding_generation",
        "arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":training-job/owned-gen-declared",
        {
            "run_binding": generation_binding,
            "run_ref": run_ref,
            "request": native_request,
            "catalog": catalog(),
        },
    )

    class PairBoundaryReachedError(Exception):
        pass

    second_run_ref = objects.install(
        "sessions/declared-publisher/generation/owned-gen-second/input/bindings/RUN.json", run_raw
    )
    second_request = fixed_generation_request(
        catalog(), "declared-publisher", "owned-gen-second", second_run_ref, "base"
    )
    second = JobHandle(
        "owned-gen-second",
        "coding_generation",
        handle.arn.replace(handle.id, "owned-gen-second"),
        {
            "run_binding": generation_binding,
            "run_ref": second_run_ref,
            "request": second_request,
            "catalog": catalog(),
        },
    )
    backend = SimpleNamespace(_generation_store=SimpleNamespace(_records={"first": handle, "second": second}))
    state = SimpleNamespace(
        work=SimpleNamespace(engines=[SimpleNamespace(tool_backend=backend)]),
        objects=objects,
        native=SimpleNamespace(states={}),
    )
    waits = []

    async def wait(owner, _reader, predicate, label):
        assert owner is state
        waits.append(label)
        if label == "two actual registered generation originals":
            assert predicate()
            return
        assert label == "actual registered paired original"
        assert state.native.states == {handle.id: "Completed", second.id: "Completed"}
        raise PairBoundaryReachedError

    monkeypatch.setattr(cold_fixture, "wait_for", wait)

    async def run_driver():
        try:
            await cold_fixture.finish_generation_pair(state, None)
        except PairBoundaryReachedError:
            return
        except (Exception, pytest.fail.Exception):
            raise AssertionError("generation_original_prompt_driver_refused") from None
        raise AssertionError("generation_pair_boundary_not_reached")

    asyncio.run(run_driver())
    assert waits == ["two actual registered generation originals", "actual registered paired original"]
    for published_request in (native_request, second_request):
        prefix = published_request["OutputDataConfig"]["S3OutputPath"].split("/", 3)[3] + "/model/"
        for name in ("GENERATION.json", "CANDIDATES.jsonl", "COMMITTED.json"):
            output_ref, output_raw = objects.objects[prefix + name]
            assert output_ref.version_id and output_ref.byte_count == len(output_raw)
            assert output_ref.sha256 == hashlib.sha256(output_raw).hexdigest()
            assert objects.objects[(prefix + name, output_ref.version_id)] == output_raw
        assert len(objects.objects[prefix + "CANDIDATES.jsonl"][1].splitlines()) == 164
        committed = json.loads(objects.objects[prefix + "COMMITTED.json"][1])
        assert set(committed["files"]) == {"GENERATION.json", "CANDIDATES.jsonl"}
        assert committed["native_admission"] is committed["quality_success"] is False
