"""Trusted opaque preparation state and fresh domain-policy controls; no AWS."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from amlas.backends.adapters._util import raw_tool
from amlas.backends.types import DatasetArtifactRef, DatasetInspectionSource, JobHandle, JobState, JobStatus
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.preparation_state import (
    PreparationCatalog,
    SessionPreparationStore,
    VerifiedPreparationSource,
)
from amlas.flows.coding_preparation_flow import run_coding_preparation_flow
from amlas.tools.coding_preparation import get_coding_preparation, prepare_coding_dataset

ACCOUNT = "1" * 12
REGION = "us-west-2"
BUCKET = "amlas-test-session-bucket"
DEPLOYMENT = "00000000-0000-4000-8000-" + "000000" + "000001"
PREFIX = "amlas-test"
IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/preparation@sha256:" + "a" * 64
SOURCE = "b" * 40
REF = "acq_" + "c" * 32


def catalog(**changes):
    value = PreparationCatalog(
        account_id=ACCOUNT,
        region=REGION,
        bucket=BUCKET,
        execution_role_arn=f"arn:aws:iam::{ACCOUNT}:role/amlas-test-exec",
        image_uri=IMAGE,
        source_commit=SOURCE,
        admission_digest="a" * 64,
        deployment_id=DEPLOYMENT,
        deployment_prefix=PREFIX,
        exclusions_ref=DatasetArtifactRef(
            uri=f"s3://{BUCKET}/exclusions/cases.json",
            version_id="exclusions-v1",
            sha256="d" * 64,
            byte_count=2,
            format="json",
        ),
    )
    return dataclasses.replace(value, **changes)


def source(session, **changes):
    prefix = f"s3://{BUCKET}/sessions/{session}/acquisition/job/"
    inspection = DatasetInspectionSource(
        artifact_ref=DatasetArtifactRef(
            uri=prefix + "data.jsonl", version_id="data-v1", sha256="e" * 64, byte_count=20, format="jsonl"
        ),
        account_id=ACCOUNT,
        region=REGION,
        bucket=BUCKET,
        runtime_session_id=session,
        deployment_id=DEPLOYMENT,
        deployment_prefix=PREFIX,
        acquisition_job_arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/amlas-test-data-acq",
    )
    value = VerifiedPreparationSource(
        inspection_source=inspection,
        manifest_refs=tuple(
            (name, prefix + name + ".json", name + "-v1", "f" * 64)
            for name in ("receipt", "commit", "lineage")
        ),
        repo_id="public/canonical",
        revision="c" * 40,
        filename="records.jsonl",
        acquisition_image_uri=IMAGE,
        acquisition_source_commit=SOURCE,
    )
    return dataclasses.replace(value, **changes)


class Acquisition(SessionAcquisitionStore):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.sources = {REF: source(self.runtime_session_id)}

    def _verified_preparation_source(self, reference):
        self.calls += 1
        if reference not in self.sources:
            raise ValueError("unknown source")
        return self.sources[reference]


class Backend:
    backend_name = "processing"

    def __init__(self):
        self.calls = 0
        self.observations = 0
        self.state = JobState.RUNNING

    def submit(self, *, sources, catalog, runtime_session_id):
        self.calls += 1
        return JobHandle(
            id=f"{PREFIX}-data-prep-job",
            backend="processing",
            arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/job",
            extra={"runtime_session_id": runtime_session_id},
        )

    def status(self, handle):
        return JobStatus(self.state)

    def artifacts(self, handle):
        self.observations += 1
        return {"success": False, "prepared": False, "accepted": True, "status": "running"}


def store_pair(selected=None):
    acquired, backend = Acquisition(), Backend()
    return acquired, backend, SessionPreparationStore(acquired, backend, selected or catalog())


def auto(cap=None):
    return ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=cap)


@pytest.mark.parametrize("policy", [None, {"mode": "auto"}, ApprovalPolicy(), auto(10)])
def test_denied_or_unknown_capped_policy_precedes_source_or_submit(policy):
    acquired, backend, store = store_pair()
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=policy, store=store)
    assert result["status"] == "denied"
    assert acquired.calls == 0 and backend.calls == 0


def test_plan_only_never_resolves_sources_or_creates():
    acquired, backend, store = store_pair()
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], plan_only=True, store=store)
    assert result["status"] == "planned" and result["prepared"] is False
    assert acquired.calls == 0 and backend.calls == 0


def test_trusted_uncapped_domain_policy_submits_once_and_returns_opaque_ref():
    acquired, backend, store = store_pair()
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    assert result["status"] == "submitted" and result["prepared"] is False
    assert result["request_ref"].startswith("prep_")
    assert acquired.calls == 1 and backend.calls == 1
    again = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    assert again["request_ref"] == result["request_ref"] and backend.calls == 1


def test_removed_policy_is_not_cached_in_store():
    _acquired, backend, store = store_pair()
    run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    denied = run_coding_preparation_flow(acquisition_request_refs=[REF], store=store)
    assert denied["status"] == "denied" and backend.calls == 1
    assert not hasattr(store, "policy")


@pytest.mark.parametrize("refs", [[], ["fake"], [REF, REF], ["acq_" + "0" * 32]])
def test_invalid_opaque_source_reference_denies_without_submit(refs):
    _acquired, backend, store = store_pair()
    result = run_coding_preparation_flow(acquisition_request_refs=refs, policy=auto(), store=store)
    assert result["success"] is False and backend.calls == 0


def test_foreign_source_session_is_refused():
    acquired, backend, store = store_pair()
    acquired.sources[REF] = source("another-session")
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    assert result["success"] is False and backend.calls == 0


def test_unadmitted_image_denies_before_even_acquisition_read():
    acquired, backend, store = store_pair(catalog(admission_digest=""))
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    assert result["success"] is False
    assert acquired.calls == 0 and backend.calls == 0


def test_parquet_source_and_unknown_profile_are_explicit_refusal():
    acquired, backend, store = store_pair()
    current = acquired.sources[REF]
    artifact = dataclasses.replace(current.inspection_source.artifact_ref, format="parquet")
    acquired.sources[REF] = dataclasses.replace(
        current, inspection_source=dataclasses.replace(current.inspection_source, artifact_ref=artifact)
    )
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    assert result["success"] is False and backend.calls == 0
    result = run_coding_preparation_flow(
        acquisition_request_refs=[REF], profile_id="miniswe-guessed", policy=auto(), store=store
    )
    assert result["success"] is False and backend.calls == 0


def test_polling_is_read_only_and_foreign_conversation_ref_is_unknown():
    _, backend, store = store_pair()
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    for _ in range(2):
        assert store.observe(result["request_ref"])["prepared"] is False
    assert backend.calls == 1 and backend.observations == 2
    _, other_backend, other = store_pair()
    assert other.observe(result["request_ref"])["status"] == "unknown_reference"
    assert other_backend.observations == 0


def test_concurrent_duplicate_submission_is_not_resubmitted():
    _, backend, store = store_pair()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: run_coding_preparation_flow(
                    acquisition_request_refs=[REF], policy=auto(), store=store
                ),
                range(2),
            )
        )
    assert backend.calls == 1 and results[0]["request_ref"] == results[1]["request_ref"]


def test_tool_schema_has_no_policy_catalog_resources_handles_or_context():
    schema = prepare_coding_dataset.tool_spec["inputSchema"]["json"]["properties"]
    assert set(schema) == {
        "acquisition_request_refs",
        "case_request_ref",
        "profile_id",
        "profile_version",
        "plan_only",
    }
    assert set(get_coding_preparation.tool_spec["inputSchema"]["json"]["properties"]) == {"request_ref"}


def test_hidden_sdk_state_supplies_typed_policy_and_store_only():
    _, backend, store = store_pair()
    context = SimpleNamespace(
        invocation_state={"_amlas_approval_policy": auto(), "_amlas_preparation_store": store}
    )
    result = raw_tool(prepare_coding_dataset)([REF], tool_context=context)
    assert result["status"] == "submitted" and backend.calls == 1
    context.invocation_state["_amlas_approval_policy"] = {"mode": "auto"}
    assert raw_tool(prepare_coding_dataset)([REF], tool_context=context)["status"] == "denied"
    assert backend.calls == 1


def test_native_read_failure_keeps_handle_and_never_retries_submission():
    _, backend, store = store_pair()
    result = run_coding_preparation_flow(acquisition_request_refs=[REF], policy=auto(), store=store)
    backend.artifacts = lambda _handle: (_ for _ in ()).throw(RuntimeError("closed failure"))
    observed = store.observe(result["request_ref"])
    assert observed["prepared"] is False and observed["status"] == "observation_failed"
    assert observed["request_ref"] == result["request_ref"] and backend.calls == 1


def test_private_source_dto_is_frozen_and_has_no_grants():
    value = source("session")
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.repo_id = "changed"
    assert set(dataclasses.asdict(value)) == {
        "inspection_source",
        "manifest_refs",
        "repo_id",
        "revision",
        "filename",
        "acquisition_image_uri",
        "acquisition_source_commit",
    }


def sdk_invoke(store, policy, params=None):
    import json

    from strands import Agent
    from strands.hooks.events import AfterToolCallEvent
    from test_sdk_prompt_policy import ScriptedModel

    results = []
    request = {
        "id": "prep",
        "name": "prepare_coding_dataset",
        "input": params or {"acquisition_request_refs": [REF]},
    }
    agent = Agent(
        model=ScriptedModel([[request]]),
        tools=[prepare_coding_dataset],
        callback_handler=lambda *_a, **_kw: None,
    )

    def collect(event):
        for block in event.result["content"]:
            if "json" in block:
                results.append(block["json"])
                break
            if "text" in block:
                try:
                    value = json.loads(block["text"])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    results.append(value)
                    break
        if not results:
            results.append({"sdk_error": event.result["status"] == "error"})

    agent.hooks.add_callback(AfterToolCallEvent, collect)
    state = {"_amlas_preparation_store": store}
    if policy is not None:
        state["_amlas_approval_policy"] = policy
    agent("prepare canonical conversations", invocation_state=state)
    return results[0]


def test_actual_sdk_transports_private_state_without_caching_policy():
    _, backend, store = store_pair()
    assert sdk_invoke(store, auto())["status"] == "submitted"
    assert sdk_invoke(store, None)["status"] == "denied"
    assert backend.calls == 1


def test_actual_sdk_model_authored_context_cannot_authorize():
    _, backend, store = store_pair()
    result = sdk_invoke(
        store,
        None,
        params={
            "acquisition_request_refs": [REF],
            "tool_context": {"invocation_state": {"_amlas_approval_policy": {"mode": "auto"}}},
        },
    )
    assert result.get("status") == "denied" or result.get("sdk_error")
    assert backend.calls == 0


def test_actual_sdk_concurrent_conversations_keep_refs_and_handles_separate():
    pairs = [store_pair(), store_pair()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda pair: sdk_invoke(pair[2], auto()), pairs))
    assert all(result["status"] == "submitted" for result in results)
    assert results[0]["request_ref"] != results[1]["request_ref"]
    assert pairs[0][2].runtime_session_id != pairs[1][2].runtime_session_id
    assert pairs[0][2].observe(results[1]["request_ref"])["status"] == "unknown_reference"
    assert pairs[1][2].observe(results[0]["request_ref"])["status"] == "unknown_reference"
    assert all(pair[1].calls == 1 for pair in pairs)
