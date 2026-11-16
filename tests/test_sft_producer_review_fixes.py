"""Introduced restore-boundary regressions with neutral SDK/metadata fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_sft_producer_snapshot import PATH, fixture


@pytest.fixture
def original(monkeypatch, tmp_path):
    return fixture.seed(monkeypatch, tmp_path)


def repository(original):
    from amlas.control_plane.producer_snapshot import ProducerSnapshotRepository

    result = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=original.table)
    selected = result.publish_from_store(original.sft, original.original["request_ref"])
    return result, selected


def install_fresh(original, monkeypatch):
    from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
    from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
    from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
    from amlas.backends.ports import CodingPreparationBackend, ModelWeightAcquisitionBackend, QwenSftBackend
    from amlas.backends.registry import REGISTRY

    actual = REGISTRY.resolve
    adapters = []

    def resolve(port, *args, **kwargs):
        if port is ModelWeightAcquisitionBackend:
            adapter = ProcessingModelWeightAcquisitionBackend(
                sagemaker_client=original.model_native, s3_client=original.model_s3
            )
        elif port is CodingPreparationBackend:
            adapter = ProcessingCodingPreparationBackend(
                sagemaker_client=original.prep_native, s3_client=original.prep_s3
            )
        elif port is QwenSftBackend:
            adapter = SageMakerQwenSftBackend(sagemaker_client=original.native, s3_client=original.sft_s3)
        else:
            return actual(port, *args, **kwargs)
        adapters.append(adapter)
        return SimpleNamespace(adapter=adapter)

    monkeypatch.setattr(REGISTRY, "resolve", resolve)
    return adapters


def approved():
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy

    return ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"], interactive=False)


def test_uninjected_preparation_factory_returned_clients_are_guarded_and_lifecycle_restored(
    original, tmp_path
):
    repository(original)
    state = fixture.export_state(original)
    state["uninjected_preparation"] = True
    path = tmp_path / "neutral-uninjected-state.json"
    path.write_text(json.dumps(state))
    response = subprocess.run(
        [sys.executable, str(PATH), str(path)],
        text=True,
        capture_output=True,
        timeout=45,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "AWS_EC2_METADATA_DISABLED": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert response.returncode == 0, response.stdout + response.stderr
    assert json.loads(response.stdout.splitlines()[-1])["mutations"] == 0


@pytest.mark.parametrize(
    "case", ["unavailable", "native_identity", "completed_missing_outputs", "revoked_before_status"]
)
def test_unknown_invalid_complete_or_revoked_sft_never_exposes_graph(original, monkeypatch, case):
    from amlas.control_plane.producer_snapshot import hydrate_sft_campaign

    repo, _ = repository(original)
    context = repo.load_original(original.original["job_name"])
    policy = approved()
    actual = original.native.describe_training_job

    def described(**kwargs):
        if case == "unavailable":
            raise RuntimeError("neutral unavailable observation")
        if case == "revoked_before_status":
            policy.allowlist.clear()
        value = actual(**kwargs)
        if case == "native_identity":
            value["TrainingJobArn"] += "-foreign"
        return value

    monkeypatch.setattr(original.native, "describe_training_job", described)
    if case == "completed_missing_outputs":
        original.native.state = "Completed"
    adapters = install_fresh(original, monkeypatch)
    with pytest.raises(ValueError):
        hydrate_sft_campaign(context, policy=policy)
    assert len(original.native.calls) == 1
    assert all(
        type(adapter._sm).__name__ != "_ApprovedReads" and type(adapter._s3).__name__ != "_ApprovedReads"
        for adapter in adapters
    )


@pytest.mark.parametrize(
    "case",
    [
        "prep_header_image",
        "prep_header_source",
        "acquisition_image",
        "acquisition_arn",
        "model_producer_identity",
    ],
)
def test_coherent_rehash_semantic_contradiction_refuses_before_producer_reads(original, case):
    repo, selected = repository(original)
    row = original.table.items[selected["record_key"]]
    value = json.loads(row["snapshot"]["B"])
    if case.startswith("prep_header"):
        field = "image_uri" if case == "prep_header_image" else "source_commit"
        altered = (
            value["preparation"]["header"][field].replace("a", "b") if field == "image_uri" else "c" * 40
        )
        value["preparation"]["header"][field] = altered
        request = json.loads(value["preparation"]["request_json"])
        args = request["AppSpecification"]["ContainerArguments"]
        args[args.index("--" + field.replace("_", "-")) + 1] = altered
        value["preparation"]["request_json"] = json.dumps(request, sort_keys=True, separators=(",", ":"))
    elif case == "acquisition_image":
        value["preparation"]["sources"][0]["acquisition_image_uri"] = "ordinary-invalid-image"
    elif case == "acquisition_arn":
        value["preparation"]["sources"][0]["inspection_source"]["acquisition_job_arn"] = (
            "ordinary-invalid-arn"
        )
    else:
        model = value["model"]
        old = model["id"]
        new = old + "-other"
        model["id"] = new
        model["arn"] = model["arn"].replace(old, new)
        model["bindings"]["job_name"] = new
        model["output_prefix"] = model["output_prefix"].replace(old, new)
        request = json.loads(model["request_json"])
        request["ProcessingJobName"] = new
        args = request["AppSpecification"]["ContainerArguments"]
        args[args.index("--job-name") + 1] = new
        request["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"] = request[
            "ProcessingOutputConfig"
        ]["Outputs"][0]["S3Output"]["S3Uri"].replace(old, new)
        model["request_json"] = json.dumps(request, sort_keys=True, separators=(",", ":"))
    row["snapshot"]["B"] = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    row["snapshot_sha256"]["S"] = hashlib.sha256(row["snapshot"]["B"]).hexdigest()
    before = (len(original.model_s3.reads), len(original.prep_s3.calls), original.native.reads)
    with pytest.raises(ValueError):
        repo.load_original(original.original["job_name"])
    assert (len(original.model_s3.reads), len(original.prep_s3.calls), original.native.reads) == before


@pytest.mark.parametrize("attempts", [1, 3])
def test_real_botocore_transport_cannot_retry_publication_put(original, monkeypatch, attempts):
    import boto3
    from botocore import UNSIGNED
    from botocore.awsrequest import AWSResponse
    from botocore.config import Config

    from amlas.control_plane.producer_snapshot import ProducerSnapshotRepository

    client = boto3.client(
        "dynamodb",
        region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED, retries={"mode": "standard", "total_max_attempts": attempts}
        ),
    )
    wire = []

    class Raw:
        def __init__(self, raw):
            self.raw = raw

        def stream(self, *args, **kwargs):
            yield self.raw

    def sent(request):
        target = request.headers["X-Amz-Target"]
        if isinstance(target, bytes):
            target = target.decode()
        operation = target.rsplit(".", 1)[-1]
        wire.append(operation)
        raw = (
            b"{}"
            if operation == "GetItem"
            else b'{"__type":"ProvisionedThroughputExceededException","message":"neutral retryable fixture response"}'
        )
        return AWSResponse(
            request.url,
            200 if operation == "GetItem" else 500,
            {"content-type": "application/x-amz-json-1.0", "x-amzn-requestid": "neutral-request"},
            Raw(raw),
        )

    monkeypatch.setattr(client._endpoint.http_session, "send", sent)
    monkeypatch.setattr("botocore.endpoint.time.sleep", lambda _: None)
    repo = ProducerSnapshotRepository(original.campaign, original.binding, dynamodb_client=client)
    with pytest.raises(ValueError):
        repo.publish_from_store(original.sft, original.original["request_ref"])
    assert wire.count("PutItem") == (1 if attempts == 1 else 0), (
        f"actual mutation wire attempts={wire.count('PutItem')}"
    )


def test_forked_activation_fails_closed_instead_of_inheriting_live_context(original):
    from amlas.control_plane.producer_snapshot import activate_sft_campaign, current_sft_campaign

    repo, _ = repository(original)
    context = repo.load_original(original.original["job_name"])
    read, write = os.pipe()
    with activate_sft_campaign(context):
        pid = os.fork()
        if pid == 0:
            os.close(read)
            try:
                try:
                    value = "rejected" if current_sft_campaign() is None else "inherited"
                except ValueError:
                    value = "rejected"
                os.write(write, value.encode())
            finally:
                os._exit(0)
        os.close(write)
        observed = os.read(read, 64)
        os.close(read)
        os.waitpid(pid, 0)
    assert observed == b"rejected"
    assert current_sft_campaign() is None


@pytest.mark.parametrize("native_status", ["InProgress", "Failed", "Stopped"])
def test_explicit_verified_original_non_success_states_remain_observable(
    original, monkeypatch, native_status
):
    from amlas.control_plane.producer_snapshot import hydrate_sft_campaign

    repo, _ = repository(original)
    context = repo.load_original(original.original["job_name"])
    original.native.state = native_status
    adapters = install_fresh(original, monkeypatch)
    _acq, _models, _prepared, sft = hydrate_sft_campaign(context, policy=approved())
    handle = sft._records[original.original["request_ref"]]
    assert handle.extra["restoration_status"] == "verified" and handle.id == original.original["job_name"]
    assert all(
        type(adapter._sm).__name__ != "_ApprovedReads" and type(adapter._s3).__name__ != "_ApprovedReads"
        for adapter in adapters
    )
    assert len(original.native.calls) == 1


def test_temporary_denial_latch_survives_policy_reapproval():
    from amlas.control_plane.producer_snapshot import _ApprovedReads

    policy = approved()
    denial = [False]
    calls = []

    class Client:
        def head_object(self, **kwargs):
            calls.append(kwargs)
            return {}

    guard = _ApprovedReads(Client(), policy, denial)
    policy.allowlist.clear()
    with pytest.raises(ValueError, match="policy"):
        guard.head_object(Bucket="neutral", Key="neutral")
    policy.allowlist.append("qwen_sft_flow")
    guard.head_object(Bucket="neutral", Key="neutral")
    assert denial == [True] and len(calls) == 1
