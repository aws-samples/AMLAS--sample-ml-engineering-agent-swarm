"""Pair adapter readback on the shapes of the real 2026-11-10 pair job; no AWS call, no evaluation runs.

The real job carried the deployment's own tags beside the three ownership tags and wrote 12,786 invocation
records. These tests rebuild those shapes over the strict native fakes of
test_paired_coding_backend.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from test_paired_coding_backend import canonical, outputs, submit
from test_paired_coding_backend import packet as packet

from amlas.backends.adapters import coding_pair as module
from amlas.backends.adapters import qwen_sft
from amlas.backends.types import JobState

# The tags the real pair job carries beyond the product's three ownership tags (ListTags of the real job).
DEPLOYMENT_TAGS = [
    {"Key": "project", "Value": "amlas-pilot-001"},
    {"Key": "purpose", "Value": "evaluation"},
    {"Key": "auto-delete", "Value": "no"},
]
# Fields a real DescribeProcessingJob adds to the echoed request (saved description of the real pair job).
REAL_DESCRIPTION_FIELDS = {
    "CreationTime": "2026-11-10T11:56:17.000Z",
    "LastModifiedTime": "2026-11-10T12:09:18.000Z",
    "ProcessingStartTime": "2026-11-10T12:00:31.000Z",
    "ProcessingEndTime": "2026-11-10T12:09:18.000Z",
    "ResponseMetadata": {"HTTPStatusCode": 200},
}
REAL_RECORD_COUNT = 12786


def _evaluated(*_args, **_kwargs):
    return {"case_count": 164, "success": True, "evaluated": True, "status": "evaluated"}


def _real_description(value):
    value.update(copy.deepcopy(REAL_DESCRIPTION_FIELDS))
    for entry in value["ProcessingInputs"] + value["ProcessingOutputConfig"]["Outputs"]:
        entry["AppManaged"] = False


def _submit_with_tags(packet, make):
    """A deployment client that adds its own tags at create and records the request it sent.

    `make(ownership)` returns the stored tag list and the live tag list (None: the stored list).
    """
    handle = submit(packet)
    sm = packet[3]
    stored, live = make(copy.deepcopy(handle.extra["request"]["Tags"]))
    handle.extra["request"] = dict(handle.extra["request"], Tags=copy.deepcopy(stored))
    sm.request = dict(sm.request, Tags=copy.deepcopy(stored if live is None else live))
    sm.drift = _real_description
    return handle


def _ownership(handle):
    return [tag for tag in handle.extra["request"]["Tags"] if tag["Key"].startswith("amlas:")]


def test_real_pair_job_with_deployment_tags_is_observed_and_verified(packet, monkeypatch):
    handle = _submit_with_tags(packet, lambda ownership: (ownership + DEPLOYMENT_TAGS, None))
    packet[3].state = "Completed"
    assert packet[0].status(handle).state == JobState.SUCCEEDED
    outputs(packet, handle)
    monkeypatch.setattr(module, "verify_pair_result", _evaluated, raising=False)
    result = packet[0].artifacts(handle)
    assert result["evaluated"] is True and result["status"] == "evaluated"


def test_live_tag_order_of_the_service_does_not_matter(packet):
    handle = _submit_with_tags(
        packet,
        lambda ownership: (ownership + DEPLOYMENT_TAGS, list(reversed(ownership + DEPLOYMENT_TAGS))),
    )
    assert packet[0].status(handle).state == JobState.RUNNING


@pytest.mark.parametrize(
    "case",
    [
        "live_extra",
        "live_missing",
        "live_value",
        "reserved_extra",
        "aws_reserved",
        "foreign_owner",
        "ownership_missing",
        "duplicate",
        "value_type",
        "shape",
    ],
)
def test_foreign_or_drifted_tag_sets_are_still_refused(packet, case):
    def make(ownership):
        stored, live = ownership + DEPLOYMENT_TAGS, None
        if case == "live_extra":
            live = [*stored, {"Key": "owner", "Value": "other"}]
        elif case == "live_missing":
            live = stored[:-1]
        elif case == "live_value":
            live = [*stored[:-1], {"Key": "auto-delete", "Value": "yes"}]
        elif case == "reserved_extra":
            stored = [*stored, {"Key": "amlas:owner", "Value": "other"}]
        elif case == "aws_reserved":
            stored = [*stored, {"Key": "aws:owner", "Value": "other"}]
        elif case == "foreign_owner":
            stored = [
                dict(tag, Value="other-deployment") if tag["Key"] == "amlas:deployment-id" else tag
                for tag in stored
            ]
        elif case == "ownership_missing":
            stored = stored[1:]
        elif case == "duplicate":
            stored = [*stored, {"Key": "project", "Value": "other"}]
        elif case == "value_type":
            stored = [*stored, {"Key": "cost-center", "Value": 7}]
        else:
            stored = [*stored, {"Key": "cost-center", "Value": "7", "Extra": "x"}]
        return stored, live

    handle = _submit_with_tags(packet, make)
    packet[3].state = "Completed"
    outputs(packet, handle)
    before = len(packet[4].calls)
    assert packet[0].status(handle).state == JobState.UNKNOWN
    assert packet[0].artifacts(handle)["evaluated"] is False
    assert len(packet[4].calls) == before and packet[3].calls.count("create") == 1


def test_handle_without_deployment_tags_is_unchanged(packet):
    handle = submit(packet)
    packet[3].drift = _real_description
    assert packet[0].status(handle).state == JobState.RUNNING
    assert [tag["Key"] for tag in _ownership(handle)] == [
        "amlas:deployment-id",
        "amlas:deployment-prefix",
        "amlas:runtime-session-id",
    ]


# ---- the invocation manifest and the record pass ---------------------------------------------------------


def _attempt(arm, case, ordinal):
    return canonical(
        {
            "evaluation_id": "paired-id",
            "case_id": "HumanEval/" + str(case),
            "arm": arm,
            "ordinal": ordinal,
            "binding": {
                "evaluation_id": "paired-id",
                "case_id": "HumanEval/" + str(case),
                "arm": arm,
                "ordinal": ordinal,
            },
            "session_id": "original-session",
            "interpreter_id": "aws.codeinterpreter.v1",
            "kind": "attempt",
            "time": 1.0,
            "deadline": 11.0,
            "request_sha256": "a" * 64,
            "code_sha256": "b" * 64,
        }
    )


def _install_records(packet, handle, count):
    """`count` hash-only call records under the real path grammar, with their manifest and commit."""
    payloads = outputs(packet, handle)
    s3, prefix = packet[4], handle.extra["output_prefix"]
    for key in list(s3.versions):
        if key.startswith(prefix + "INVOCATIONS/"):
            del s3.versions[key]
    rows = []
    for index in range(count):
        arm, rest = ("base", "tuned")[index % 2], index // 2
        case, ordinal = rest % 164, rest // 164
        name = f"INVOCATIONS/{arm}/{case:03d}/calls/{ordinal:08d}-attempt.json"
        raw = _attempt(arm, case, ordinal)
        s3.install(prefix + name, raw)
        rows.append(
            canonical({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)})
            + b"\n"
        )
    manifest = b"".join(rows)
    commit = json.loads(payloads["COMMITTED.json"])
    commit["invocations_manifest"] = {
        "sha256": hashlib.sha256(manifest).hexdigest(),
        "byte_count": len(manifest),
        "format": "jsonl",
        "record_count": count,
    }
    s3.install(prefix + "INVOCATIONS_MANIFEST.jsonl", manifest)
    s3.install(prefix + "COMMITTED.json", canonical(commit))
    return prefix + "INVOCATIONS_MANIFEST.jsonl"


def _paginate(s3, monkeypatch):
    """ListObjectsV2 pages of at most MaxKeys keys with opaque continuation tokens, as the service returns them."""

    def list_objects_v2(**kwargs):
        s3.check("ListObjectsV2", kwargs)
        names = sorted(key for key in s3.versions if key.startswith(kwargs["Prefix"]))
        start = int(kwargs["ContinuationToken"].split("-")[1]) if "ContinuationToken" in kwargs else 0
        page = names[start : start + kwargs["MaxKeys"]]
        response = {
            "IsTruncated": start + kwargs["MaxKeys"] < len(names),
            "Contents": [{"Key": key, "Size": len(s3.objects[(key, s3.versions[key])])} for key in page],
        }
        if response["IsTruncated"]:
            response["NextContinuationToken"] = "page-" + str(start + kwargs["MaxKeys"])
        return response

    monkeypatch.setattr(s3, "list_objects_v2", list_objects_v2)


class _TrackedBody(io.BytesIO):
    pass


def test_manifest_stream_is_closed_before_any_record_is_read(packet, monkeypatch):
    handle = submit(packet)
    manifest_key = _install_records(packet, handle, 6)
    s3 = packet[4]
    opened, violations = [], []
    original_get, original_head = s3.get_object, s3.head_object

    def get_object(**kwargs):
        response = original_get(**kwargs)
        if kwargs["Key"] == manifest_key:
            response["Body"] = _TrackedBody(response["Body"].getvalue())
            opened.append(response["Body"])
        elif "/INVOCATIONS/" in kwargs["Key"] and any(not body.closed for body in opened):
            violations.append(kwargs["Key"])
        return response

    def head_object(**kwargs):
        if "/INVOCATIONS/" in kwargs["Key"] and any(not body.closed for body in opened):
            violations.append(kwargs["Key"])
        return original_head(**kwargs)

    monkeypatch.setattr(s3, "get_object", get_object)
    monkeypatch.setattr(s3, "head_object", head_object)
    monkeypatch.setattr(module, "verify_pair_result", _evaluated, raising=False)
    result = packet[0].artifacts(handle)
    assert result["evaluated"] is True
    assert len(opened) == 1 and violations == []


def test_record_pass_of_the_real_volume_fits_its_budget(packet, monkeypatch):
    """12,786 records at the measured cost of one serial S3 call: 2,160 s for the real unshimmed pass (0.0338 s a call)."""
    handle = submit(packet)
    _install_records(packet, handle, REAL_RECORD_COUNT)
    s3 = packet[4]
    _paginate(s3, monkeypatch)
    clock = SimpleNamespace(now=1000.0)
    per_call = 2160 / (REAL_RECORD_COUNT * 5)

    def advance(method):
        def call(**kwargs):
            clock.now += per_call
            return method(**kwargs)

        return call

    for name in ("head_object", "get_object", "list_objects_v2"):
        monkeypatch.setattr(s3, name, advance(getattr(s3, name)))
    fake_time = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr(module, "time", fake_time)
    monkeypatch.setattr(qwen_sft, "time", fake_time)
    monkeypatch.setattr(module, "verify_pair_result", _evaluated, raising=False)
    result = packet[0].artifacts(handle)
    assert result["evaluated"] is True, {key: result.get(key) for key in ("status", "error_type", "reason")}
    assert len(result["files"]) == REAL_RECORD_COUNT + 4
    assert clock.now - 1000.0 > 600


def test_record_pass_still_has_a_finite_deadline(packet, monkeypatch):
    """A record pass far slower than the measured one still stops at its deadline."""
    handle = submit(packet)
    _install_records(packet, handle, 40)
    s3 = packet[4]
    clock = SimpleNamespace(now=1000.0)
    original = s3.get_object

    def slow(**kwargs):
        if "/INVOCATIONS/" in kwargs["Key"]:
            clock.now += 120
        return original(**kwargs)

    monkeypatch.setattr(s3, "get_object", slow)
    fake_time = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr(module, "time", fake_time)
    monkeypatch.setattr(qwen_sft, "time", fake_time)
    monkeypatch.setattr(module, "verify_pair_result", _evaluated, raising=False)
    result = packet[0].artifacts(handle)
    assert result["evaluated"] is False and result["status"] == "output_verification_failed"


def test_manifest_replaced_during_the_record_pass_is_refused(packet, monkeypatch):
    handle = submit(packet)
    manifest_key = _install_records(packet, handle, 4)
    s3 = packet[4]
    original = s3.get_object
    reads = []

    def replace_after_first_record(**kwargs):
        response = original(**kwargs)
        if "/INVOCATIONS/" in kwargs["Key"]:
            reads.append(kwargs["Key"])
            if len(reads) == 1:
                s3.install(manifest_key, s3.objects[(manifest_key, s3.versions[manifest_key])])
        return response

    monkeypatch.setattr(s3, "get_object", replace_after_first_record)
    monkeypatch.setattr(module, "verify_pair_result", _evaluated, raising=False)
    assert packet[0].artifacts(handle)["evaluated"] is False
