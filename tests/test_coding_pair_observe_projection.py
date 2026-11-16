"""What get_coding_pair hands to the agent stays bounded; the adapter still verifies and returns every record.

A real run (2026-11-12) observed a completed pair job with 13,146 invocation records. The tool result carried one
reference per record (about 6.1 MB in the conversation), and the next model call was refused as too long. These
tests pin the bounded agent view and guard the unchanged adapter and store answers.
"""

from __future__ import annotations

import dataclasses
import json
import threading

import pytest
from test_paired_coding_backend import packet as packet
from test_paired_coding_backend import submit
from test_paired_coding_real_path import _install_records, _paginate

from amlas.backends.adapters import coding_pair as module
from amlas.backends.types import DatasetArtifactRef, JobHandle
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.paired_state import SessionPairedCodingStore
from amlas.flows import coding_pair_flow
from amlas.flows.coding_pair_flow import run_coding_pair_flow
from amlas.tui.chat.tool_binding_backend import _conversation_bytes

REAL_RUN_RECORD_COUNT = 13146
CORE = ("COMMITTED.json", "RECEIPTS.jsonl", "EVALUATION.json", "INVOCATIONS_MANIFEST.jsonl")
REQUEST_REF = "pair_" + "c" * 32
BUDGET = 24576


def _policy():
    return ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["coding_pair_flow"], max_auto_cost=None)


def _store(adapter, handle):
    """The real store class and its real observe, holding one original record (the constructor needs a full
    generation conversation, which these tests do not exercise)."""
    store = SessionPairedCodingStore.__new__(SessionPairedCodingStore)
    store._adapter, store._records, store._keys = adapter, {REQUEST_REF: handle}, {}
    store._lock = threading.RLock()
    store.runtime_session_id = "isolated-session"
    return store


def _summary():
    """A summary of the real run's shape: 22 keys, 164 case outcome rows, that run's counts and pass rates."""
    outcomes = [
        {
            "case_id": "HumanEval/" + str(i),
            "base": "passed" if i < 138 else "failed",
            "tuned": "passed" if i < 106 else "failed",
            "base_attempts": 1,
            "tuned_attempts": 1,
        }
        for i in range(164)
    ]
    summary = {
        "base_pass_at_1": 138 / 164,
        "tuned_pass_at_1": 106 / 164,
        "delta_pass_at_1": 106 / 164 - 138 / 164,
        "base_pass_count": 138,
        "tuned_pass_count": 106,
        "both_pass": 101,
        "both_fail": 21,
        "base_only": 37,
        "tuned_only": 5,
        "execution_count": 328,
        "case_count": 164,
        "base_failures": {"failed": 23, "truncated": 2, "infrastructure_error": 1},
        "tuned_failures": {"failed": 43, "runtime_error": 9, "truncated": 3, "timeout": 2, "incomplete": 1},
        "cleanup_complete": True,
        "evaluation_id": "paired-id",
        "spec_sha256": "e" * 64,
        "binding": {
            "evaluation_id": "paired-id",
            "runner_sha256": "d" * 64,
            "case_manifest_sha256": "f" * 64,
        },
        "case_outcomes": outcomes,
        "schema_version": 1,
        "metric": "pass@1",
        "arms": ["base", "tuned"],
        "receipt_count": 328,
    }
    assert len(summary) == 22
    return summary


def _evaluated(*_args, **_kwargs):
    return {
        "case_count": 164,
        "success": True,
        "evaluated": True,
        "status": "evaluated",
        "summary": _summary(),
    }


def _real_shaped(packet, monkeypatch):
    handle = submit(packet)
    _install_records(packet, handle, REAL_RUN_RECORD_COUNT)
    _paginate(packet[4], monkeypatch)
    monkeypatch.setattr(module, "verify_pair_result", _evaluated, raising=False)
    return _store(packet[0], handle)


def _message_bytes(result):
    """The tool-result message in record form: one JSON text block in a user toolResult.

    ``json.dumps(result)`` is the text Strands 1.42.0 builds. The locked Strands 1.65.0 leaves non-ASCII
    characters unescaped, which is never longer in canonical bytes, so this size is the upper bound on both.
    """
    message = {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "tooluse_" + "a" * 22,
                    "status": "success",
                    "content": [{"text": json.dumps(result)}],
                }
            }
        ],
    }
    return _conversation_bytes([message])


def _observe(store):
    return run_coding_pair_flow(operation="observe", request_ref=REQUEST_REF, policy=_policy(), store=store)


def test_real_shaped_pair_observation_stays_bounded(packet, monkeypatch):
    store = _real_shaped(packet, monkeypatch)
    result = _observe(store)
    assert result["evaluated"] is True, {key: result.get(key) for key in ("status", "error_type")}
    assert _message_bytes(result) <= BUDGET, _message_bytes(result)


class _Adapter:
    def __init__(self, count):
        self.count = count

    def artifacts(self, handle):
        ref = DatasetArtifactRef(
            "s3://owned-test-bucket/sessions/s/paired/output/X", "v" * 32, "a" * 64, 465, "json"
        )
        files = {name: ref for name in CORE}
        files.update(
            {f"INVOCATIONS/base/{i % 164:03d}/calls/{i:08d}-attempt.json": ref for i in range(self.count)}
        )
        return {
            "success": True,
            "evaluated": True,
            "status": "evaluated",
            "case_count": 164,
            "receipt_count": 328,
            "summary": _summary(),
            "files": files,
            "quality_success": False,
            "native_admission": False,
        }


def _handle():
    return JobHandle(
        "owned-pair",
        "paired_coding",
        "arn:aws:sagemaker:us-west-2:111122223333:processing-job/owned-pair",
        {"submission_state": "accepted"},
    )


def test_bounded_pair_view_size_does_not_grow_with_record_count():
    sizes = {}
    for count in (0, 12786, REAL_RUN_RECORD_COUNT, 1_000_000):
        result = _observe(_store(_Adapter(count), _handle()))
        assert result["invocation_records_view"]["record_count"] == count
        sizes[count] = _message_bytes(result)
        assert sizes[count] - sizes[0] == len(str(count)) - len("0"), sizes


def test_bounded_pair_view_keeps_report_and_binding_fields(packet, monkeypatch):
    store = _real_shaped(packet, monkeypatch)
    adapter_view = store.observe(REQUEST_REF)
    result = _observe(store)
    assert set(result["files"]) == set(CORE)
    for name in CORE:
        assert result["files"][name] == dataclasses.asdict(adapter_view["files"][name])
        assert set(result["files"][name]) >= {"sha256", "version_id", "byte_count"}
    assert result["invocation_records_view"] == {
        "view": "bounded_v1",
        "evidence_field": "files.INVOCATIONS_MANIFEST.jsonl",
        "record_count": REAL_RUN_RECORD_COUNT,
    }
    summary = result["summary"]
    assert summary == _summary()
    assert (summary["base_pass_at_1"], summary["tuned_pass_at_1"]) == (138 / 164, 106 / 164)
    assert (summary["both_pass"], summary["both_fail"], summary["base_only"], summary["tuned_only"]) == (
        101,
        21,
        37,
        5,
    )
    for key in ("status", "evaluated", "request_ref", "job_name", "job_arn", "success", "accepted"):
        assert result[key] == adapter_view[key], key
    assert result["execution_provider"] == coding_pair_flow.execution_provider(coding_pair_flow.PROCESSING)
    assert (result["case_count"], result["receipt_count"]) == (164, 328)


def test_store_observe_still_returns_every_verified_ref(packet, monkeypatch):
    store = _real_shaped(packet, monkeypatch)
    s3 = packet[4]
    reads = []
    original = s3.get_object

    def counted(**kwargs):
        if "/INVOCATIONS/" in kwargs["Key"]:
            reads.append(kwargs["Key"])
        return original(**kwargs)

    monkeypatch.setattr(s3, "get_object", counted)
    observed = store.observe(REQUEST_REF)
    assert observed["evaluated"] is True
    assert len(observed["files"]) == REAL_RUN_RECORD_COUNT + 4
    assert len(set(reads)) == REAL_RUN_RECORD_COUNT


class _Shape:
    def __init__(self, value):
        self.value = value

    def artifacts(self, handle):
        if isinstance(self.value, Exception):
            raise self.value
        return dict(self.value)


_REJECTION = {"schema_version": 1, "reason": "paired_result_mismatch", "data_only": True}
SHAPES = {
    "running": {"success": False, "evaluated": False, "status": "running"},
    "output_verification_failed": {
        "success": False,
        "evaluated": False,
        "status": "output_verification_failed",
        "error_type": "ValueError",
        "verification_rejection": _REJECTION,
    },
    "observation_failed": TimeoutError("neutral observation failure"),
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_non_success_pair_shapes_pass_through_unchanged(name):
    store = _store(_Shape(SHAPES[name]), _handle())
    expected_inner = store.observe(REQUEST_REF)
    result = _observe(store)
    assert "invocation_records_view" not in result and "files" not in result
    assert result == coding_pair_flow._public(
        {
            **run_coding_pair_flow(plan_only=False, operation="denied-probe"),
            **expected_inner,
            "executed": False,
        }
        | {"status": expected_inner["status"]}
    )


def test_capacity_unknown_and_refused_shapes_pass_through_unchanged(monkeypatch):
    from amlas.backends.types import CAPACITY_REJECTION_CODE

    rejected = JobHandle(
        "owned-pair",
        "paired_coding",
        "arn:aws:sagemaker:us-west-2:111122223333:processing-job/x",
        {"submission_state": "rejected", "rejection_code": CAPACITY_REJECTION_CODE},
    )
    capacity = _observe(_store(_Shape({}), rejected))
    assert capacity["status"] == "submission_rejected" and "rejection_reason" in capacity
    assert "invocation_records_view" not in capacity and "files" not in capacity
    unknown = run_coding_pair_flow(
        operation="observe",
        request_ref="pair_" + "d" * 32,
        policy=_policy(),
        store=_store(_Shape({}), _handle()),
    )
    assert unknown["status"] == "unknown_reference" and "invocation_records_view" not in unknown

    store = _store(_Shape({}), _handle())
    monkeypatch.setattr(store, "observe", lambda ref: (_ for _ in ()).throw(RuntimeError("x")))
    refused = _observe(store)
    assert refused["status"] == "refused" and refused["error_type"] == "RuntimeError"
    assert "invocation_records_view" not in refused
