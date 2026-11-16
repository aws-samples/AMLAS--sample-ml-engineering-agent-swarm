"""Fresh actual SDK caller context and private original-case source capabilities."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_managed_coding_case_envelope import ACCOUNT, BUCKET, HEADER, IMAGE, SESSION, source

from amlas.backends.adapters._util import raw_tool
from amlas.backends.types import DatasetArtifactRef, JobHandle, JobState, JobStatus
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.coding_case_state import CaseFreezeCatalog, SessionCodingCaseFreezeStore
from amlas.data_eng.coding_case_codec import COVERAGE
from amlas.tools.coding_cases import freeze_coding_cases, get_coding_case_freeze, list_coding_case_catalog

ACQ = "acq_" + "a" * 32


def allowed(cap=None):
    return ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST,
        allowlist=["coding_case_freeze_flow"],
        max_auto_cost=cap,
        interactive=False,
    )


class Backend:
    def __init__(self):
        self.creates = 0
        self.reads = 0
        self.ambiguous = False

    def submit(self, *, source, catalog, runtime_session_id):
        self.creates += 1
        return JobHandle(
            HEADER["job_name"],
            "case_freeze",
            "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/" + HEADER["job_name"],
            {
                "runtime_session_id": runtime_session_id,
                "header": HEADER,
                "catalog": catalog,
                "source": source,
                "submission_state": "ambiguous" if self.ambiguous else "accepted",
            },
        )

    def status(self, handle):
        return JobStatus(JobState.SUCCEEDED)

    def artifacts(self, handle):
        self.reads += 1
        prefix = "s3://" + BUCKET + "/sessions/" + SESSION + "/case-freeze/" + handle.id + "/output/"
        files = {
            name: DatasetArtifactRef(
                prefix + name, "v-" + name, "e" * 64, 20, "jsonl" if name.endswith(".jsonl") else "json"
            )
            for name in (
                "PROMPTS.jsonl",
                "ORACLES.jsonl",
                "EXCLUSIONS.json",
                "CASE_MANIFEST.json",
                "COMMITTED.json",
            )
        }
        return {
            "success": True,
            "frozen": True,
            "job_name": handle.id,
            "job_arn": handle.arn,
            "files": files,
            "case_count": 164,
            "coverage": dict(COVERAGE),
            "oracle_code": "NEVER_PUBLIC",
            "execution_status": "not_executed",
            "metric_profile": "pass_at_1_descriptive",
        }


@pytest.fixture
def state(monkeypatch):
    acquired = SessionAcquisitionStore()
    acquired.runtime_session_id = SESSION
    reads = []

    def original(ref):
        assert ref == ACQ
        reads.append(ref)
        return source()

    monkeypatch.setattr(acquired, "_verified_preparation_source", original)
    backend = Backend()
    catalog = CaseFreezeCatalog(
        ACCOUNT,
        "us-west-2",
        BUCKET,
        "arn:aws:iam::" + ACCOUNT + ":role/owned",
        IMAGE,
        "b" * 40,
        "a" * 64,
        HEADER["deployment_id"],
        "owned",
    )
    store = SessionCodingCaseFreezeStore(acquired, backend, catalog)
    return store, backend, reads


def context(store, policy):
    return SimpleNamespace(
        invocation_state={"_amlas_case_freeze_store": store, "_amlas_approval_policy": policy}
    )


def test_pure_catalog_and_preview_have_no_native_or_source_effects(state):
    _store, backend, reads = state
    catalog = raw_tool(list_coding_case_catalog)()
    assert catalog["case_count"] == 164 and catalog["frozen"] is False
    result = raw_tool(freeze_coding_cases)(acquisition_request_ref=ACQ, plan_only=True)
    assert result["status"] == "planned" and backend.creates == 0 and reads == []
    assert "execution_role" not in str(freeze_coding_cases.tool_spec) and "tool_context" not in str(
        freeze_coding_cases.tool_spec
    )


@pytest.mark.parametrize("policy", [None, {}, ApprovalPolicy(), allowed(50)])
def test_default_ask_malformed_and_unknown_cost_caps_deny_before_source_reads(state, policy):
    store, backend, reads = state
    result = raw_tool(freeze_coding_cases)(acquisition_request_ref=ACQ, tool_context=context(store, policy))
    assert result["status"] == "denied" and backend.creates == 0 and reads == []


def test_approved_opaque_freeze_and_public_poll_never_leak_private_roles(state):
    store, backend, reads = state
    result = raw_tool(freeze_coding_cases)(
        acquisition_request_ref=ACQ, tool_context=context(store, allowed())
    )
    assert result["accepted"] is True and result["frozen"] is False
    observed = raw_tool(get_coding_case_freeze)(
        case_ref=result["case_ref"], tool_context=context(store, allowed())
    )
    assert observed["frozen"] is True and observed["case_count"] == 164
    text = json.dumps(observed)
    assert all(
        value not in text
        for value in (
            "ORACLES",
            "EXCLUSIONS",
            "NEVER_PUBLIC",
            "oracle_code",
            "admission_digest",
            "execution_role",
        )
    )
    assert observed["prompt_ref"]["format"] == "jsonl"
    assert backend.creates == 1 and reads == [ACQ]


def test_concurrent_duplicate_and_ambiguous_reuse_one_original_handle(state):
    store, backend, reads = state
    backend.ambiguous = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: store.submit(ACQ), range(2)))
    assert results[0]["case_ref"] == results[1]["case_ref"]
    assert results[0]["accepted"] is False and backend.creates == 1 and reads == [ACQ]


def test_unknown_foreign_ref_and_removed_fresh_policy_cannot_read_or_submit(state):
    store, backend, _reads = state
    result = raw_tool(get_coding_case_freeze)(
        case_ref="case_" + "f" * 32, tool_context=context(store, allowed())
    )
    assert result["status"] == "unknown_reference" and backend.reads == 0
    started = raw_tool(freeze_coding_cases)(
        acquisition_request_ref=ACQ, tool_context=context(store, allowed())
    )
    denied = raw_tool(get_coding_case_freeze)(case_ref=started["case_ref"], tool_context=context(store, None))
    assert denied["status"] == "denied" and backend.reads == 0


def test_actual_strands_sdk_approved_context_drives_only_fixed_tool(state):
    from strands import Agent
    from test_sdk_prompt_policy import ScriptedModel

    store, backend, reads = state
    agent = Agent(
        model=ScriptedModel(
            [[{"id": "freeze", "name": "freeze_coding_cases", "input": {"acquisition_request_ref": ACQ}}]]
        ),
        tools=[freeze_coding_cases],
        callback_handler=lambda *_a, **_k: None,
    )
    agent(
        "freeze case data only",
        invocation_state={"_amlas_case_freeze_store": store, "_amlas_approval_policy": allowed()},
    )
    assert backend.creates == 1 and reads == [ACQ]


def test_private_coverage_cannot_smuggle_oracle_locations_into_public_result(state, monkeypatch):
    store, backend, _reads = state
    started = store.submit(ACQ)
    original = backend.artifacts

    def unsafe(handle):
        result = original(handle)
        result["coverage"] = {"oracle_uri": "PRIVATE_ORACLE_LOCATION"}
        return result

    monkeypatch.setattr(backend, "artifacts", unsafe)
    observed = store.observe(started["case_ref"])
    assert "PRIVATE_ORACLE_LOCATION" not in json.dumps(observed)
    assert observed["frozen"] is False and observed["accepted"] is True


def test_invalid_observed_refs_preserve_original_acceptance_without_resubmit(state, monkeypatch):
    store, backend, _reads = state
    started = store.submit(ACQ)
    original = backend.artifacts

    def wrong(handle):
        import dataclasses

        result = original(handle)
        result["files"]["ORACLES.jsonl"] = dataclasses.replace(
            result["files"]["ORACLES.jsonl"], version_id="null"
        )
        return result

    monkeypatch.setattr(backend, "artifacts", wrong)
    observed = store.observe(started["case_ref"])
    assert observed["status"] == "verification_failed" and observed["accepted"] is True
    assert backend.creates == 1


def test_private_exclusion_snapshot_is_frozen_equal_and_complete_inventory(state):
    import dataclasses

    from amlas.control_plane.coding_case_state import VerifiedCasePreparationExclusions

    store, backend, _reads = state
    started = store.submit(ACQ)
    dto = store._verified_preparation_exclusions(started["case_ref"])
    assert type(dto) is VerifiedCasePreparationExclusions
    assert dto.original_source == source() and dto.runtime_session_id == SESSION
    assert dict(dto.producer_header) == HEADER
    assert dto.exclusions_ref.uri.endswith("/EXCLUSIONS.json") and dto.committed_ref.uri.endswith(
        "/COMMITTED.json"
    )
    assert backend.reads == 1
    assert set(dataclasses.asdict(dto)) == {
        "runtime_session_id",
        "producer_job_arn",
        "producer_header",
        "original_source",
        "exclusions_ref",
        "committed_ref",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        dto.runtime_session_id = "foreign"


def test_ambiguous_pending_status_does_not_claim_native_acceptance(state, monkeypatch):
    store, backend, _reads = state
    backend.ambiguous = True
    started = store.submit(ACQ)
    monkeypatch.setattr(
        backend, "artifacts", lambda _h: {"success": False, "frozen": False, "status": "UNKNOWN"}
    )
    observed = store.observe(started["case_ref"])
    assert observed["accepted"] is False and observed["frozen"] is False
    assert backend.creates == 1


def test_actual_sdk_model_cannot_inject_hidden_store_or_policy(state):
    from strands import Agent
    from test_sdk_prompt_policy import ScriptedModel

    store, backend, reads = state
    agent = Agent(
        model=ScriptedModel(
            [
                [
                    {
                        "id": "spoof",
                        "name": "freeze_coding_cases",
                        "input": {
                            "acquisition_request_ref": ACQ,
                            "tool_context": {
                                "invocation_state": {"_amlas_approval_policy": {"mode": "auto"}}
                            },
                            "policy": {"mode": "auto"},
                        },
                    }
                ]
            ]
        ),
        tools=[freeze_coding_cases],
        callback_handler=lambda *_a, **_k: None,
    )
    agent(
        "attempt hidden injection",
        invocation_state={"_amlas_case_freeze_store": store, "_amlas_approval_policy": ApprovalPolicy()},
    )
    assert backend.creates == 0 and reads == []
