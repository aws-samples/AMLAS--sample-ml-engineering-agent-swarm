"""Characterize normal enrollment CAS using wholly declared, inert DATA.

These tests grant no runtime authority and prove neither claim_completion, native
DynamoDB permissions, SFT custody nor cold restoration. Expected baseline is GREEN.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters

from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.campaign_state import CampaignStateCatalog
from amlas.control_plane.producer_snapshot import _approved_read
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend, _InputAdmissionHeldError

DDB_MODEL = Session().get_service_model("dynamodb")
TABLE_ID = "11111111-2222-3333-4444-a55555555555"


def _policy(allowed=True):
    return ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"] if allowed else [])


def _item(context, record):
    raw = checkpoint._canonical(record)
    return {
        "job_name": {"S": "PROMPT_RUN#" + context.run_id},
        "record": {"S": raw},
        "record_sha256": {"S": hashlib.sha256(raw.encode()).hexdigest()},
    }


def _declared_enrollment():
    catalog = CampaignStateCatalog(
        account_id="111122223333",
        region="us-west-2",
        deployment_id="11111111-1111-1111-1111-a11111111111",
        deployment_prefix="declared-cas",
        jobs_table="declared-cas-training-jobs",
        execution_role_arn="arn:aws:iam::111122223333:role/declared-cas-runtime",
        execution_role_id="AROA" + "A" * 16,
    )
    context = checkpoint.NormalRunContext(catalog, "1" * 32, "2" * 64)
    record = {
        "schema_version": 1,
        "logical_run_id": context.run_id,
        "session_sha256": context.session_hash,
        "scope": asdict(catalog),
        "table_id": TABLE_ID,
        "private_store_correlation": "22222222-2222-2222-2222-a22222222222",
        "admitted_prompt": "Declared admitted enrollment input.",
        "history": [],
        "stage": "enrolled",
    }
    return context, record, _item(context, record)


class StrictDynamoPort:
    """Only current GetItem/UpdateItem wire shapes and atomic two-field CAS."""

    def __init__(self, context, item, *, barrier=None, lost_ack=False, altered_ack=False):
        self.context = context
        self.item = copy.deepcopy(item)
        self.lock = threading.Lock()
        self.barrier = barrier
        self.lost_ack = lost_ack
        self.altered_ack = altered_ack
        self.update_calls = self.get_calls = self.commits = 0

    def get_item(self, **kwargs):
        validate_parameters(kwargs, DDB_MODEL.operation_model("GetItem").input_shape)
        assert kwargs == {
            "TableName": self.context.catalog.jobs_table,
            "Key": {"job_name": self.item["job_name"]},
            "ConsistentRead": True,
        }
        with self.lock:
            self.get_calls += 1
            return {"Item": copy.deepcopy(self.item)}

    def update_item(self, **kwargs):
        validate_parameters(kwargs, DDB_MODEL.operation_model("UpdateItem").input_shape)
        assert set(kwargs) == {
            "TableName",
            "Key",
            "ConditionExpression",
            "UpdateExpression",
            "ExpressionAttributeNames",
            "ExpressionAttributeValues",
        }
        assert kwargs["TableName"] == self.context.catalog.jobs_table
        assert kwargs["Key"] == {"job_name": self.item["job_name"]}
        assert kwargs["ConditionExpression"] == "#record = :prior AND record_sha256 = :prior_sha"
        assert kwargs["UpdateExpression"] == "SET #record = :body, record_sha256 = :sha"
        assert kwargs["ExpressionAttributeNames"] == {"#record": "record"}
        values = kwargs["ExpressionAttributeValues"]
        assert set(values) == {":prior", ":prior_sha", ":body", ":sha"}
        with self.lock:
            self.update_calls += 1
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        with self.lock:
            if self.item["record"] != values[":prior"] or self.item["record_sha256"] != values[":prior_sha"]:
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
            self.item["record"] = copy.deepcopy(values[":body"])
            self.item["record_sha256"] = copy.deepcopy(values[":sha"])
            self.commits += 1
            if self.altered_ack:
                altered = json.loads(self.item["record"]["S"])
                altered["admitted_prompt"] = "Declared concurrent canonical enrollment change."
                self.item = _item(self.context, altered)
        if self.lost_ack:
            raise ConnectionError("Declared update acknowledgement lost.")
        return {}


def _replace(context, port, item, record, policy=None, qualify=None):
    current = _policy() if policy is None else policy
    current_qualify = (lambda: _approved_read(current)) if qualify is None else qualify
    checkpoint._replace_record(context, port, TABLE_ID, current_qualify, item, record, policy=current)


@pytest.mark.parametrize("stale", ["record", "record_sha256"], ids=["stale-body", "stale-sha"])
def test_stale_body_or_sha_refuses_without_mutation(stale):
    context, record, item = _declared_enrollment()
    prior = copy.deepcopy(item)
    if stale == "record":
        earlier = {**record, "admitted_prompt": "Declared earlier enrollment input."}
        prior["record"] = _item(context, earlier)["record"]
    else:
        prior["record_sha256"] = {"S": "f" * 64}
    after = {**record, "admitted_prompt": "Declared requested replacement."}
    port = StrictDynamoPort(context, item)

    with pytest.raises(ValueError, match="normal TASK CAS outcome unresolved"):
        _replace(context, port, prior, after)

    assert port.item == item and port.commits == 0
    assert port.update_calls == port.get_calls == 1


def test_two_contenders_make_only_one_atomic_commit():
    context, record, item = _declared_enrollment()
    port = StrictDynamoPort(context, item, barrier=threading.Barrier(2))
    choices = [
        {**record, "admitted_prompt": "Declared contender A."},
        {**record, "admitted_prompt": "Declared contender B."},
    ]

    def attempt(after):
        try:
            _replace(context, port, copy.deepcopy(item), after)
        except ValueError:
            return "refused"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt, after) for after in choices]
        outcomes = [future.result(timeout=10) for future in futures]

    assert sorted(outcomes) == ["committed", "refused"]
    assert port.update_calls == 2 and port.commits == 1 and port.get_calls == 1
    assert port.item in [_item(context, after) for after in choices]


@pytest.mark.parametrize("altered", [False, True], ids=["exact-after", "valid-altered-after"])
def test_lost_ack_recognizes_only_the_entire_exact_after_item(altered):
    context, record, item = _declared_enrollment()
    after = {**record, "admitted_prompt": "Declared committed replacement."}
    port = StrictDynamoPort(context, item, lost_ack=True, altered_ack=altered)
    if altered:
        with pytest.raises(ValueError, match="normal TASK CAS outcome unresolved"):
            _replace(context, port, item, after)
        # Prove refusal was exact-after mismatch, not failed canonical/schema load.
        _restored, accepted = checkpoint._load(context, port, TABLE_ID, lambda: None)
        assert accepted["admitted_prompt"] == "Declared concurrent canonical enrollment change."
        assert port.item != _item(context, after)
    else:
        _replace(context, port, item, after)
        assert port.item == _item(context, after)
    assert port.update_calls == port.commits == 1
    assert port.get_calls == (2 if altered else 1)


def test_fresh_denied_policy_prevents_update():
    context, record, item = _declared_enrollment()
    port = StrictDynamoPort(context, item)
    with pytest.raises(ValueError):
        _replace(context, port, item, record, policy=_policy(False))
    assert port.item == item and port.update_calls == port.get_calls == port.commits == 0


def test_fresh_denied_input_prevents_update(monkeypatch):
    context, record, item = _declared_enrollment()
    port = StrictDynamoPort(context, item)
    backend = object.__new__(ToolBindingBackend)
    backend._input_lock = threading.RLock()
    backend._input_hold = False
    backend._input_cycle = "declared-cas-admission-cycle"
    backend._input_receipts = []

    class DeniedChecker:
        def check(self, text, *, boundary, opaque_cycle, tool_name):
            assert boundary == "HISTORY_TEXT"
            assert opaque_cycle == "declared-cas-admission-cycle"
            return SimpleNamespace(
                outcome="BLOCKED",
                forward_text=None,
                receipt={"boundary": boundary, "outcome": "BLOCKED"},
            )

    monkeypatch.setattr(backend, "_selected_input_checker", lambda: DeniedChecker())

    def qualify():
        backend._admit_input("Declared current input.", "HISTORY_TEXT", immutable=True)

    with pytest.raises(_InputAdmissionHeldError):
        _replace(context, port, item, record, qualify=qualify)
    assert backend._input_hold is True
    assert port.item == item and port.update_calls == port.get_calls == port.commits == 0
