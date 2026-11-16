"""Normal SDK SFT checkpoint component tests with mocked upstream DATA only.

This does not establish genuine upstream preparation, native admission, actual
worker loss, automatic wake or full R1-R4/C1-C4. Exact-ref upstream suppliers are
existing synthetic fixture DATA; actual SDK, policy, SFT catalog/native request,
S3 staging and journal/normal lookup paths remain the tested boundary. No SFT
journal, campaign factory or granted ContextVar is seeded by the fixture.
"""

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

import boto3
import model_acquisition_fixtures as model_fixture
import pytest
import test_normal_run_checkpoint as normal_fixture
import test_qwen_sft_backend as sft_fixture
from botocore.session import Session
from botocore.validate import validate_parameters
from test_normal_run_checkpoint import (
    PROMPT,
    assert_safe_data,
    decoded,
)
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup
from test_runtime_sft_reload_originals import current_inputs
from test_sft_restore_aliases import OriginalNative

from amlas import app as runtime
from amlas.backends.ports import QwenSftBackend
from amlas.backends.registry import REGISTRY
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

MODEL_REF = "model_acq_" + "a" * 32
PREP_REF = "prep_" + "b" * 32
TOOL_USE = "normal-original-sft-use"
PATH = Path(__file__).resolve()
SM_MODEL = Session().get_service_model("sagemaker")


class JournalObservedNative(OriginalNative):
    """Existing strict per-job SFT transport; snapshot before accepting Create."""

    def __init__(self, ports):
        super().__init__()
        self.ports = ports
        self.before_create = []
        self.stops = []

    def create_training_job(self, **kwargs):
        validate_parameters(kwargs, SM_MODEL.operation_model("CreateTrainingJob").input_shape)
        self.before_create.append(copy.deepcopy(self.ports.items))
        return super().create_training_job(**kwargs)

    def describe_training_job(self, **kwargs):
        validate_parameters(kwargs, SM_MODEL.operation_model("DescribeTrainingJob").input_shape)
        return super().describe_training_job(**kwargs)

    def stop_training_job(self, **kwargs):
        self.stops.append(copy.deepcopy(kwargs))
        raise AssertionError("pending original lookup must never Stop its job")


class CapturedS3(sft_fixture.S3):
    """Read exact captured artifact/checkpoint versions; no fresh artifact minting."""

    def __init__(self, ports):
        super().__init__()
        self.ports = ports
        self.staging_attempts = []

    def get_bucket_location(self, **kwargs):
        assert kwargs["Bucket"] == sft_fixture.BUCKET
        assert kwargs["ExpectedBucketOwner"] == sft_fixture.ACCOUNT
        return {"LocationConstraint": "us-west-2"}

    def get_object(self, **kwargs):
        assert kwargs["Bucket"] == sft_fixture.BUCKET and kwargs.get("VersionId")
        if (kwargs["Key"], kwargs["VersionId"]) in self.ports.objects:
            return self.ports.get_object(**kwargs)
        return super().get_object(**kwargs)

    def head_object(self, **kwargs):
        assert kwargs["Bucket"] == sft_fixture.BUCKET and kwargs.get("VersionId")
        body = self.ports.objects.get((kwargs["Key"], kwargs["VersionId"]))
        if body is not None:
            return {
                "VersionId": kwargs["VersionId"],
                "ContentLength": len(body),
                "ChecksumType": "FULL_OBJECT",
                "ChecksumSHA256": base64.b64encode(hashlib.sha256(body).digest()).decode(),
            }
        return super().head_object(**kwargs)

    def forbidden_stage(self, operation):
        self.staging_attempts.append(operation)
        raise AssertionError("pending cold lookup must not stage or regenerate artifacts")

    def put_object(self, **kwargs):
        return self.forbidden_stage("PutObject")

    def copy_object(self, **kwargs):
        return self.forbidden_stage("CopyObject")

    def create_multipart_upload(self, **kwargs):
        return self.forbidden_stage("CreateMultipartUpload")

    def upload_part_copy(self, **kwargs):
        return self.forbidden_stage("UploadPartCopy")

    def complete_multipart_upload(self, **kwargs):
        return self.forbidden_stage("CompleteMultipartUpload")


def install_mock_sources(monkeypatch, work, ports, model, *, captured=None):
    """Attach exact-ref test DATA to normal graph; keep SFT boundary validators."""
    native = JournalObservedNative(ports)
    if captured is None:
        _unused_adapter, _unused_native, s3, inputs = sft_fixture.backend()
    else:
        from amlas.backends.types import DatasetArtifactRef

        inputs = None
        ports.items = captured["ddb"]
        ports.objects = {
            (row["key"], row["version_id"]): row["body"] for row in captured["checkpoint_objects"]
        }
        native.calls, native.states = captured["training_requests"], captured["training_states"]
        s3 = CapturedS3(ports)
        for row in captured["artifacts"]:
            s3.install(DatasetArtifactRef(**row["ref"]), row["body"])
    from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend

    adapter = SageMakerQwenSftBackend(sagemaker_client=native, s3_client=s3)
    for key, value in {
        "SAGEMAKER_EXECUTION_ROLE_ARN": sft_fixture.catalog().execution_role_arn,
        "AMLAS_QWEN_SFT_IMAGE_URI": sft_fixture.IMAGE,
        "AMLAS_QWEN_SFT_SOURCE_COMMIT": sft_fixture.SOURCE,
        "AMLAS_QWEN_SFT_ADMISSION_DIGEST": "b" * 64,
    }.items():
        monkeypatch.setenv(key, value)
    actual_resolve, actual_init, actual_client = REGISTRY.resolve, ToolBindingBackend.__init__, boto3.client
    sources = []

    def resolve(port, *args, **kwargs):
        if port is QwenSftBackend:
            return SimpleNamespace(adapter=adapter, adapter_name=adapter.backend_name)
        return actual_resolve(port, *args, **kwargs)

    def initialize(backend, *args, **kwargs):
        actual_init(backend, *args, **kwargs)
        if inputs is None:
            # Cold graphs may be restored only by the production journal/codec path.
            return
        models, prepared = backend._model_acquisition_store, backend._preparation_store
        selected = current_inputs(inputs, s3, models.runtime_session_id)

        def model_source(reference):
            if reference != MODEL_REF:
                raise ValueError("unknown mocked upstream model reference")
            sources.append(("model", reference))
            return selected.model

        def prepared_source(reference):
            if reference != PREP_REF:
                raise ValueError("unknown mocked upstream preparation reference")
            sources.append(("preparation", reference))
            return selected.prepared

        monkeypatch.setattr(models, "_verified_sft_model_snapshot", model_source)
        monkeypatch.setattr(prepared, "_verified_sft_snapshot", prepared_source)
        assert backend._sft_store._models is models and backend._sft_store._prepared is prepared

    def client(service, **kwargs):
        if service == "sagemaker":
            assert kwargs.get("region_name", "us-west-2") == "us-west-2"
            return native
        if service == "s3" and captured is not None:
            return s3
        return actual_client(service, **kwargs)

    async def stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        model.calls += 1
        ports.before_model.append(copy.deepcopy(ports.items))
        yield {"messageStart": {"role": "assistant"}}
        if model.calls == 1:
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": TOOL_USE, "name": "submit_qwen_sft"}},
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {
                        "toolUse": {
                            "input": json.dumps(
                                {
                                    "model_request_ref": MODEL_REF,
                                    "preparation_request_ref": PREP_REF,
                                    "recipe_id": "full_v1",
                                }
                            )
                        }
                    },
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Original is pending."}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}

    monkeypatch.setattr(REGISTRY, "resolve", resolve)
    monkeypatch.setattr(ToolBindingBackend, "__init__", initialize)

    class NativeSession:
        def __init__(self, *args, **kwargs):
            pass

        def client(self, service, **kwargs):
            return client(service, **kwargs)

    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setattr(boto3, "Session", NativeSession)
    monkeypatch.setattr(boto3.session, "Session", NativeSession)
    monkeypatch.setattr(model, "stream", stream)
    return SimpleNamespace(work=work, ports=ports, model=model, native=native, s3=s3, sources=sources)


def configured_work(monkeypatch, tmp_path, *, captured=None):
    """Use one synthetic owner scope; this is fixture configuration, not a grant."""
    catalog = sft_fixture.catalog()
    for name, value in {
        "ACCOUNT": catalog.account_id,
        "BUCKET": catalog.bucket,
        "ROLE": catalog.execution_role_arn,
        "IMAGE": catalog.image_uri,
        "COMMIT": catalog.source_commit,
        "DEPLOYMENT": catalog.deployment_id,
    }.items():
        monkeypatch.setattr(model_fixture, name, value)
    for name, value in {
        "ACCOUNT": catalog.account_id,
        "PREFIX": catalog.deployment_prefix,
        "TABLE": catalog.deployment_prefix + "-training-jobs",
        "RUNTIME_ROLE": "arn:aws:iam::" + catalog.account_id + ":role/owned-normal-runtime",
    }.items():
        monkeypatch.setattr(normal_fixture, name, value)
    context = normal_fixture.normal_work.__wrapped__(monkeypatch, tmp_path)
    work, ports, model = next(context)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", catalog.deployment_prefix)
    value = install_mock_sources(monkeypatch, work, ports, model, captured=captured)
    return context, value


@pytest.fixture
def original_sft(monkeypatch, tmp_path):
    context, value = configured_work(monkeypatch, tmp_path)
    try:
        yield value
    finally:
        context.close()


async def start_original(value):
    events = []

    async def read():
        async for event in runtime.agent_invocation({"prompt": PROMPT}):
            events.append(event)
        return events

    reader = asyncio.create_task(read())
    try:
        deadline = asyncio.get_running_loop().time() + 5
        while not value.work.originals and not reader.done():
            assert asyncio.get_running_loop().time() < deadline, (
                "normal registered SFT did not return its result"
            )
            await asyncio.sleep(0.005)
        answer = next((answer for use, answer in value.work.originals if use == TOOL_USE), None)
        assert answer and answer.get("accepted") is True, {
            "result": answer,
            "creates": len(value.native.calls),
        }
        assert len(value.native.calls) == len(value.work.engines) == len(value.work.agents) == 1
        assert value.native.stops == []
        policy = value.work.engines[0].tool_backend.approval_policy
        assert type(policy) is ApprovalPolicy and policy.mode is ApprovalMode.ALLOWLIST
        assert "qwen_sft_flow" in policy.allowlist and policy.interactive is False
        assert value.sources[:2] == [("model", MODEL_REF), ("preparation", PREP_REF)]
        assert set(value.sources) == {("model", MODEL_REF), ("preparation", PREP_REF)}
        assert value.work.guard.calls and all(call["source"] == "INPUT" for call in value.work.guard.calls)
        run_id = next(event["run_id"] for event in events if event["event"] == "run_started")
        return reader, answer, run_id
    except BaseException:
        await close_original_worker(value.work, reader)
        raise


def pending_progress(run_id, arn):
    return {
        "event": "prompt_run_status",
        "status": "available",
        "run_id": run_id,
        "found": True,
        "progress": {
            "schema_version": 1,
            "logical_run_id": run_id,
            "stage": "sft",
            "disposition": "awaiting_original",
            "native_job_arn": arn,
            "status": "InProgress",
            "pending": True,
            "lifecycle_verified": False,
            "quality_success": False,
        },
    }


def test_normal_sdk_sft_journals_original_and_parents_before_create(original_sft):
    value = original_sft

    async def scenario():
        reader = None
        try:
            reader, original, run_id = await start_original(value)
            request = value.native.calls[0]
            name = request["TrainingJobName"]
            prior = value.native.before_create[0]
            assert name in prior, "normal SFT Create had no prospective conditional Dynamo journal"
            data = decoded(prior)
            assert_safe_data(data)
            body = json.dumps(data, sort_keys=True)
            assert run_id in body and MODEL_REF in body and PREP_REF in body
            assert "PROMPT_RUN#" + run_id in prior
            store = value.work.engines[0].tool_backend._sft_store
            assert store._campaign_state is not None, "normal tool SFT did not use the Dynamo backend"
            assert original["job_name"] == name and original["full_training_complete"] is False
        finally:
            if reader is not None:
                await close_original_worker(value.work, reader)

    asyncio.run(scenario())


def wire_encode(value):
    """Data-only fixture cloud export; no pickle, runtime objects or grants."""
    if isinstance(value, bytes):
        return {"fixture_bytes_base64": base64.b64encode(value).decode()}
    if type(value) is dict:
        return {key: wire_encode(item) for key, item in value.items()}
    if type(value) is list:
        return [wire_encode(item) for item in value]
    return value


def wire_decode(value):
    if type(value) is dict and set(value) == {"fixture_bytes_base64"}:
        return base64.b64decode(value["fixture_bytes_base64"], validate=True)
    if type(value) is dict:
        return {key: wire_decode(item) for key, item in value.items()}
    if type(value) is list:
        return [wire_decode(item) for item in value]
    return value


def cloud_snapshot(value, run_id):
    """Capture actual transport records BEFORE cleanup revokes the warm policy."""
    data = {
        "schema_version": 1,
        "run_id": run_id,
        "ddb": copy.deepcopy(value.ports.items),
        "checkpoint_objects": [
            {"key": key, "version_id": version, "body": body}
            for (key, version), body in value.ports.objects.items()
        ],
        "training_requests": copy.deepcopy(value.native.calls),
        "training_states": copy.deepcopy(value.native.states),
        "artifacts": [
            {"ref": dataclasses.asdict(ref), "body": body} for ref, body in value.s3.objects.values()
        ],
    }
    assert len(data["training_requests"]) == 1
    assert_safe_data(decoded(data["ddb"]))
    return wire_encode(data)


def cold_lookup(data, action, directory):
    """New process normal route; fixture transport records supply DATA, no activation."""
    assert set(data) == {
        "schema_version",
        "run_id",
        "ddb",
        "checkpoint_objects",
        "training_requests",
        "training_states",
        "artifacts",
    }
    assert data["schema_version"] == 1
    data = wire_decode(data)
    monkeypatch = pytest.MonkeyPatch()
    context = None
    try:
        inert_runtime_setup.__wrapped__(monkeypatch)
        context, value = configured_work(monkeypatch, directory, captured=data)
        assert len(value.native.calls) == 1
        assert value.s3.mutations == [] and value.s3.staging_attempts == []
        assert [
            {"ref": dataclasses.asdict(ref), "body": body} for ref, body in value.s3.objects.values()
        ] == data["artifacts"]
        assert_safe_data(decoded(value.ports.items))

        async def forbidden_model(*args, **kwargs):
            value.model.calls += 1
            raise AssertionError("pending original restoration dispatched a model")
            yield None

        monkeypatch.setattr(value.model, "stream", forbidden_model)

        async def read():
            return [
                event
                async for event in runtime.agent_invocation({"action": action, "run_id": data["run_id"]})
            ]

        try:
            events = asyncio.run(read())
            route_error = None
        except ValueError as error:
            events, route_error = [], type(error).__name__
        return {
            "events": events,
            "route_error": route_error,
            "training_creates": len(value.native.calls),
            "stops": value.native.stops,
            "model_calls": value.model.calls,
            "s3_staging_attempts": value.s3.staging_attempts,
            "s3_stage_mutations": value.s3.mutations,
            "journal_writes": [
                name for name, _args in value.ports.calls if name in {"PutItem", "UpdateItem", "PutObject"}
            ],
            "sdk_tasks": value.work.owned_app.get_async_task_info()["active_count"],
        }
    finally:
        if context is not None:
            context.close()
        monkeypatch.undo()


@pytest.mark.parametrize("action", ["get_prompt_run_status", "resume_prompt_run"])
def test_fresh_process_pending_original_is_observed_without_replay(original_sft, tmp_path, action):
    value = original_sft

    async def capture():
        reader = None
        try:
            reader, _original, run_id = await start_original(value)
            data = cloud_snapshot(value, run_id)
            request = value.native.calls[0]
            arn = (
                "arn:aws:sagemaker:us-west-2:"
                + sft_fixture.ACCOUNT
                + ":training-job/"
                + request["TrainingJobName"]
            )
            return data, pending_progress(run_id, arn)
        finally:
            # Cleanup happens after export. Revocation is not simulated worker loss.
            if reader is not None:
                await close_original_worker(value.work, reader)

    state, expected = asyncio.run(capture())
    server_state = tmp_path / "fixture-cloud-transport.json"
    raw = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    server_state.write_bytes(raw)
    assert len(value.native.calls) == 1 and value.native.stops == []
    directory = tmp_path / "fresh-sdk-entry"
    directory.mkdir()
    child = subprocess.run(
        [sys.executable, str(PATH), str(server_state), hashlib.sha256(raw).hexdigest(), action],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "AWS_EC2_METADATA_DISABLED": "true", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert child.returncode == 0, child.stdout + child.stderr
    observed = json.loads(child.stdout.splitlines()[-1])
    assert observed["route_error"] is None, "normal checkpoint action reached legacy fresh-start dispatch"
    leading = (
        [{"event": "resume_state", "state": "restoring", "run_id": expected["run_id"]}]
        if action == "resume_prompt_run"
        else []
    )
    assert observed["events"] == [*leading, expected]
    assert observed["training_creates"] == 1 and observed["model_calls"] == 0
    assert observed["stops"] == [] and observed["sdk_tasks"] == 0
    assert observed["s3_staging_attempts"] == observed["s3_stage_mutations"] == []
    assert "PutObject" not in observed["journal_writes"]
    if action == "get_prompt_run_status":
        assert observed["journal_writes"] == []


if __name__ == "__main__":
    raw = Path(sys.argv[1]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
    assert sys.argv[3] in {"get_prompt_run_status", "resume_prompt_run"}
    print(json.dumps(cold_lookup(json.loads(raw), sys.argv[3], Path.cwd()), sort_keys=True))
