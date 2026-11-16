"""Declared claim-content fence; no native, guard, origin or cold proof.

Actual row parsing and completion qualification are retained. Only transport and
original-resolution boundaries are inert declared seams returning no authority.
The private canonical-string tuple is unit DATA, not a confirmed completion claim.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import pytest
from test_normal_checkpoint_cas_contract import TABLE_ID, _item, _policy
from test_normal_completion_record_codec import DeclaredGetPort, _declared_sft_record

from amlas.control_plane import normal_run_checkpoint as checkpoint


def _qualify(context, token, policy, sdk_snapshot):
    def no_original_grant():
        return None

    context.qualify_completion(
        token, policy=policy, qualify_original=no_original_grant, sdk_snapshot=sdk_snapshot
    )


@pytest.mark.parametrize("replacement", ["none", "snapshot", "task_ref", "missing_receipt", "local_snapshot"])
def test_same_owner_token_cannot_expose_replaced_claim_content(monkeypatch, replacement):
    context, original = _declared_sft_record()
    expected_claim = copy.deepcopy(original["continuation"])
    expected_task_ref = copy.deepcopy(original["task_checkpoint_ref"])
    # Explicit declared process-local DATA, never a real CAS/native/origin receipt.
    receipt = (
        expected_claim["token"],
        checkpoint._canonical(expected_claim),
        checkpoint._canonical(expected_task_ref),
    )
    object.__setattr__(
        context, "_completion_receipt", None if replacement == "missing_receipt" else receipt
    )
    current = copy.deepcopy(original)
    load_snapshot = copy.deepcopy(expected_claim["sdk_snapshot"])
    if replacement == "local_snapshot":
        load_snapshot["data"]["state"]["declared_local_change"] = True
    if replacement == "snapshot":
        current["continuation"]["sdk_snapshot"]["data"]["state"]["declared_change"] = True
    elif replacement == "task_ref":
        current["task_checkpoint_ref"]["version_id"] = "declared-replacement-version"
    assert all(
        current["continuation"][field] == expected_claim[field]
        for field in ("schema_version", "epoch", "token", "status", "operation_digest")
    )
    port = DeclaredGetPort(context, _item(context, current))

    @contextmanager
    def declared_transport(value):
        assert value is context
        yield port, TABLE_ID, lambda: None, None

    def declared_original(value, record, table_id):
        assert value is context and table_id == TABLE_ID
        # These are declared DATA, not a private original or native qualification.
        assert record["sft"]["operation_digest"] == original["sft"]["operation_digest"]
        return None

    monkeypatch.setattr(checkpoint, "_native", declared_transport)
    monkeypatch.setattr(checkpoint, "_original", declared_original)
    # Substitutions must remain valid canonical codec DATA; parser refusal alone
    # must not accidentally satisfy the content-continuity negative.
    _parsed_item, parsed = checkpoint._load(context, port, TABLE_ID, lambda: None)
    assert parsed == current
    if replacement == "none":
        _qualify(context, expected_claim["token"], _policy(), load_snapshot)
    else:
        with pytest.raises(ValueError):
            _qualify(context, expected_claim["token"], _policy(), load_snapshot)
    assert expected_claim == original["continuation"]
    assert expected_task_ref == original["task_checkpoint_ref"]
    assert receipt == (
        original["continuation"]["token"],
        checkpoint._canonical(original["continuation"]),
        checkpoint._canonical(original["task_checkpoint_ref"]),
    )
    assert port.item == _item(context, current)
