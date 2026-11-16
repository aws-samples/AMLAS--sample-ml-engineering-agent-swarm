"""Actual owning submits separate closed immutable rejection from provider uncertainty."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_managed_dataset_acquisition import adapter as data_adapter
from test_managed_dataset_acquisition import ownership as _ownership  # noqa: F401
from test_managed_dataset_acquisition import populate
from test_managed_dataset_acquisition import submit as submit_data
from test_managed_generation_product import installed_generation
from test_paired_coding_backend import outputs
from test_paired_coding_backend import submit as submit_pair
from test_qwen_sft_backend import SESSION, catalog, described_resource_config, encoded, install_output
from test_qwen_sft_backend import backend as sft_fixture
from test_runtime_sft_reload_originals import publish_reload

from amlas.backends.adapters.artifact_rejection import is_closed_rejection
from amlas.control_plane.sft_state import VerifiedSftCheckpointSource

pytest_plugins = ("test_managed_generation_product", "test_paired_coding_backend")


def replace_raw(s3, key, raw, pair=False):
    if pair:
        version = s3.versions[key]
        s3.objects[(key, version)] = raw
    else:
        ref, _old = s3.objects[key]
        s3.install(dataclasses.replace(ref, sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw)), raw)


@pytest.fixture(params=("data", "sft", "reload", "generation", "pair"))
def original(request, monkeypatch):
    kind = request.param
    if kind == "data":
        request.getfixturevalue("_ownership")
        adapter = data_adapter()
        native, s3 = adapter._sm, adapter._s3
        handle = submit_data(adapter)
        populate(handle, s3)
        native.state = "Completed"
        prefix = handle.extra["output_prefix"]
        def count():
            return len(native.calls)
    elif kind in {"sft", "reload"}:
        adapter, native, s3, inputs = sft_fixture()
        train = adapter.submit(
            inputs=inputs, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION
        )
        install_output(adapter, native, s3, train)
        handle = train
        if kind == "reload":
            result = adapter.artifacts(train)
            assert result["trained"] is True
            checkpoint = VerifiedSftCheckpointSource(
                SESSION,
                train.arn,
                "fit_v1",
                catalog().source_commit,
                result["checkpoint_ref"],
                tuple(result["checkpoint_refs"].items()),
            )

            def describe(**kwargs):
                selected = next(
                    value for value in native.calls if value["TrainingJobName"] == kwargs["TrainingJobName"]
                )
                return {
                    **{key: value for key, value in selected.items() if key != "Tags"},
                    "ResourceConfig": described_resource_config(selected["ResourceConfig"]),
                    "TrainingJobArn": "arn:aws:sagemaker:us-west-2:"
                    + catalog().account_id
                    + ":training-job/"
                    + selected["TrainingJobName"],
                    "TrainingJobStatus": native.state,
                    "ModelArtifacts": {
                        "S3ModelArtifacts": selected["OutputDataConfig"]["S3OutputPath"] + "/model/"
                    },
                }

            monkeypatch.setattr(native, "describe_training_job", describe)
            monkeypatch.setattr(
                native,
                "list_tags",
                lambda **kwargs: {
                    "Tags": next(
                        value
                        for value in native.calls
                        if kwargs["ResourceArn"].endswith("/" + value["TrainingJobName"])
                    )["Tags"]
                },
            )
            handle = adapter.reload(
                model=inputs.model, checkpoint=checkpoint, catalog=catalog(), runtime_session_id=SESSION
            )
            publish_reload(handle, s3)
            prefix = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"].split("/", 3)[3] + "/model/"
            ref, raw = s3.objects[prefix + "RELOAD.json"]
            receipt = json.loads(raw)
            receipt["recipe_id"] = "fit_v1"
            replace_raw(s3, prefix + "RELOAD.json", encoded(receipt))
            ref, _raw = s3.objects[prefix + "RELOAD.json"]
            commit = json.loads(s3.objects[prefix + "COMMITTED.json"][1])
            commit["files"]["RELOAD.json"] = {"sha256": ref.sha256, "byte_count": ref.byte_count}
            replace_raw(s3, prefix + "COMMITTED.json", encoded(commit))
            native.state = "Completed"
        prefix = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"].split("/", 3)[3] + "/model/"
        def count():
            return len(native.calls)
    elif kind == "generation":
        adapter, native, s3, handle, *_rest = installed_generation(request.getfixturevalue("state"))
        native.state = "Completed"
        prefix = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"].split("/", 3)[3] + "/model/"
        def count():
            return len(native.calls)
    else:
        packet = request.getfixturevalue("packet")
        adapter, _inputs, _catalog, native, s3, _validated = packet
        handle = submit_pair(packet)
        outputs(packet, handle)
        prefix = handle.extra["output_prefix"]
        def count():
            return native.calls.count("create")
    expected = 2 if kind == "reload" else 1
    assert count() == expected and handle.arn
    if kind == "data":
        assert handle.extra["accepted"] is True
    else:
        assert handle.extra["submission_state"] == "accepted"
    return SimpleNamespace(
        kind=kind,
        adapter=adapter,
        native=native,
        s3=s3,
        handle=handle,
        prefix=prefix,
        count=count,
        expected=expected,
    )


@pytest.mark.parametrize("failure", ["immutable-corruption", "transport-timeout", "missing-publication"])
def test_completed_actual_original_typed_rejection_is_not_ordinary_uncertainty(
    original, monkeypatch, failure
):
    value = original
    if failure == "immutable-corruption":
        if value.kind == "data":
            key = value.prefix + "acquisition-test/data.jsonl"
            raw = value.s3.objects[key]
            altered = raw.replace(b"[]", b"{}", 1)
            assert len(altered) == len(raw) and altered != raw
            value.s3.objects[key] = altered
        else:
            key = value.prefix + "COMMITTED.json"
            raw = (
                value.s3.objects[(key, value.s3.versions[key])]
                if value.kind == "pair"
                else value.s3.objects[key][1]
            )
            commit = json.loads(raw)
            commit["source_commit"] = "f" * 40
            replace_raw(value.s3, key, encoded(commit), pair=value.kind == "pair")
    elif failure == "transport-timeout":
        method = value.s3.get_object

        def unavailable(**kwargs):
            if kwargs["Key"].startswith(value.prefix):
                raise TimeoutError("ordinary original artifact read uncertainty")
            return method(**kwargs)

        monkeypatch.setattr(value.s3, "get_object", unavailable)
    else:
        key = value.prefix + ("acquisition-test/COMMITTED.json" if value.kind == "data" else "COMMITTED.json")
        if value.kind == "pair":
            value.s3.objects.pop((key, value.s3.versions[key]))
        else:
            value.s3.objects.pop(key)
    observed = value.adapter.artifacts(value.handle)
    assert observed.get("success") is False and value.count() == value.expected
    assert (observed.get("job_name"), observed.get("job_arn")) == (value.handle.id, value.handle.arn)
    marker = observed.get("verification_rejection")
    if failure == "immutable-corruption":
        assert is_closed_rejection(marker), {
            "kind": value.kind,
            "status": observed.get("status"),
            "error_type": observed.get("error_type"),
        }
        assert observed["error_type"] == "ValueError"
    else:
        assert marker is None
    assert value.count() == value.expected
