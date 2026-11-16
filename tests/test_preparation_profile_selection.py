"""A profile the listing advertises is accepted by the dispatcher; a profile argument carries no authority.

The private acquisition sources decide which preparation runs. The CommitPackFT selector is refused unless
every selected source is the recognized CommitPack repository. Neutral stand-ins only; no AWS.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from amlas.backends.adapters._util import raw_tool
from amlas.backends.types import DatasetArtifactRef, DatasetInspectionSource, JobHandle, JobState, JobStatus
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.preparation_capabilities import describe_preparation_capabilities
from amlas.control_plane.preparation_state import (
    PreparationCatalog,
    SessionPreparationStore,
    VerifiedPreparationSource,
)
from amlas.data_eng import commitpack_preparation as cp
from amlas.data_eng import commitpack_profile as pins
from amlas.flows.coding_preparation_flow import run_coding_preparation_flow
from amlas.tools.coding_preparation import prepare_coding_dataset

ACCOUNT = "1" * 12
REGION = "us-west-2"
BUCKET = "ordinary-session-bucket"
DEPLOYMENT = "00000000-0000-4000-8000-" + "000000" + "000001"
PREFIX = "ordinary-prefix"
IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/preparation@sha256:" + "a" * 64
SOURCE = "b" * 40
REF = "acq_" + "c" * 32
EXPLICIT = ("openai.explicit_conversations.v1", "1")
COMMITPACK = (pins.PROFILE_ID, cp.POLICY_VERSION)


def catalog(**changes):
    value = PreparationCatalog(
        account_id=ACCOUNT,
        region=REGION,
        bucket=BUCKET,
        execution_role_arn=f"arn:aws:iam::{ACCOUNT}:role/ordinary-exec",
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


def source(session, *, commitpack):
    prefix = f"s3://{BUCKET}/sessions/{session}/acquisition/job/"
    artifact = DatasetArtifactRef(
        uri=prefix + "data.jsonl",
        version_id="data-v1",
        sha256=pins.ARTIFACT_SHA256 if commitpack else "e" * 64,
        byte_count=cp.SOURCE_BYTES if commitpack else 20,
        format="jsonl",
    )
    inspection = DatasetInspectionSource(
        artifact_ref=artifact,
        account_id=ACCOUNT,
        region=REGION,
        bucket=BUCKET,
        runtime_session_id=session,
        deployment_id=DEPLOYMENT,
        deployment_prefix=PREFIX,
        acquisition_job_arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/ordinary-data-acq",
    )
    return VerifiedPreparationSource(
        inspection_source=inspection,
        manifest_refs=tuple(
            (name, prefix + name + ".json", name + "-v1", "f" * 64)
            for name in ("receipt", "commit", "lineage")
        ),
        repo_id=pins.DATASET_REPO if commitpack else "public/canonical",
        revision=pins.REVISION if commitpack else "c" * 40,
        filename=pins.FILENAME if commitpack else "records.jsonl",
        acquisition_image_uri=IMAGE,
        acquisition_source_commit=SOURCE,
    )


class Acquisition(SessionAcquisitionStore):
    def __init__(self, commitpack):
        super().__init__()
        self.calls = 0
        self.sources = {REF: source(self.runtime_session_id, commitpack=commitpack)}

    def _verified_preparation_source(self, reference):
        self.calls += 1
        if reference not in self.sources:
            raise ValueError("unknown source")
        return self.sources[reference]


class Backend:
    backend_name = "processing"

    def __init__(self):
        self.calls = 0

    def submit(self, *, sources, catalog, runtime_session_id):
        self.calls += 1
        return JobHandle(
            id=f"{PREFIX}-data-prep-job",
            backend="processing",
            arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/job",
            extra={"runtime_session_id": runtime_session_id},
        )

    def status(self, handle):
        return JobStatus(JobState.RUNNING)

    def artifacts(self, handle):
        return {"success": False, "prepared": False, "accepted": True, "status": "running"}


def pair_store(*, commitpack, selected=None):
    acquired, backend = Acquisition(commitpack), Backend()
    return acquired, backend, SessionPreparationStore(acquired, backend, selected or catalog())


def auto():
    return ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=None)


def run(store, pair, **extra):
    return run_coding_preparation_flow(
        acquisition_request_refs=[REF],
        profile_id=pair[0],
        profile_version=pair[1],
        policy=auto(),
        store=store,
        **extra,
    )


def listed_pair(profile_id):
    (entry,) = [p for p in describe_preparation_capabilities()["profiles"] if p["profile_id"] == profile_id]
    return entry["profile_id"], entry["policy_version"]


def test_listed_commitpack_profile_is_accepted_for_commitpack_sources():
    pair = listed_pair(pins.PROFILE_ID)
    assert pair == COMMITPACK
    _acquired, backend, store = pair_store(commitpack=True)
    result = run(store, pair)
    assert result["status"] == "submitted" and result["executed"] is True and result["prepared"] is False
    assert result["request_ref"].startswith("prep_") and backend.calls == 1


def test_commitpack_profile_accepts_the_tool_default_version():
    _acquired, backend, store = pair_store(commitpack=True)
    assert run(store, (pins.PROFILE_ID, "1"))["status"] == "submitted" and backend.calls == 1


@pytest.mark.parametrize("commitpack", [True, False])
def test_listed_canonical_profile_is_accepted_through_its_advertised_policy_version(commitpack):
    pair = listed_pair(EXPLICIT[0])
    assert pair != EXPLICIT
    _acquired, backend, store = pair_store(commitpack=commitpack)
    assert run(store, pair)["status"] == "submitted" and backend.calls == 1


@pytest.mark.parametrize("commitpack", [True, False])
def test_public_selector_is_accepted_for_both_source_kinds(commitpack):
    selector = describe_preparation_capabilities()["public_preparation_selector"]
    pair = (selector["profile_id"], selector["profile_version"])
    assert pair == EXPLICIT
    _acquired, backend, store = pair_store(commitpack=commitpack)
    assert run(store, pair)["status"] == "submitted" and backend.calls == 1


def test_commitpack_profile_is_refused_for_other_sources_without_dispatch():
    _acquired, backend, store = pair_store(commitpack=False)
    result = run(store, COMMITPACK)
    assert result["status"] == "refused" and result["executed"] is False and result["accepted"] is False
    assert "request_ref" not in result and backend.calls == 0


def test_family_check_also_guards_the_reused_submission():
    _acquired, backend, store = pair_store(commitpack=False)
    first = run(store, EXPLICIT)
    assert first["status"] == "submitted" and backend.calls == 1
    again = run(store, COMMITPACK)
    assert again["status"] == "refused" and "request_ref" not in again and backend.calls == 1


@pytest.mark.parametrize(
    "pair",
    [
        ("miniswe-guessed", "1"),
        (pins.PROFILE_ID, "2"),
        (pins.PROFILE_ID, "coding-preparation.commitpack.v2"),
        (EXPLICIT[0], "2"),
        (EXPLICIT[0], cp.POLICY_VERSION),
        ("nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2", "coding-preparation.open-swe.v2"),
        (None, None),
    ],
)
def test_every_other_pair_stays_unsupported_before_any_source_read(pair):
    acquired, backend, store = pair_store(commitpack=True)
    result = run(store, pair)
    assert result["status"] == "unsupported_profile" and result["executed"] is False
    assert acquired.calls == 0 and backend.calls == 0


@pytest.mark.parametrize(
    "pair,status", [(EXPLICIT, "planned"), (COMMITPACK, "planned"), (("guess", "1"), "unsupported_profile")]
)
def test_plan_only_reports_what_the_real_call_would_do_with_no_effects(pair, status):
    acquired, backend, store = pair_store(commitpack=True)
    result = run(store, pair, plan_only=True)
    assert result["status"] == status and result["prepared"] is False
    assert acquired.calls == 0 and backend.calls == 0


def test_unadmitted_catalog_still_denies_the_commitpack_pair_before_source_reads():
    acquired, backend, store = pair_store(commitpack=True, selected=catalog(admission_digest=""))
    result = run(store, COMMITPACK)
    assert result["success"] is False and result["status"] != "submitted"
    assert acquired.calls == 0 and backend.calls == 0


def test_tool_passes_the_commitpack_pair_through_hidden_state_only():
    _acquired, backend, store = pair_store(commitpack=True)
    context = SimpleNamespace(
        invocation_state={"_amlas_approval_policy": auto(), "_amlas_preparation_store": store}
    )
    result = raw_tool(prepare_coding_dataset)(
        [REF], profile_id=COMMITPACK[0], profile_version=COMMITPACK[1], tool_context=context
    )
    assert result["status"] == "submitted" and backend.calls == 1
    properties = prepare_coding_dataset.tool_spec["inputSchema"]["json"]["properties"]
    assert set(properties) == {
        "acquisition_request_refs",
        "case_request_ref",
        "profile_id",
        "profile_version",
        "plan_only",
    }
