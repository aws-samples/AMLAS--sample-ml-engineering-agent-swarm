"""Genuine provider controls using only ordinary tiny fixture metadata/bytes.

The row, its SHA, current objects and adoption JSON agree with each other. These
controls isolate immutable original/catalog binding from mere self-consistency;
no hostile payload, native private data or actual service effect is involved.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import pytest
from test_model_pack_adoption_integration import CONSUMER, PACK, policy, real_case

from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.control_plane.model_pack_adoption import _bytes


def _committed_self_consistent_change(case, models, reference, mutation):
    snapshot = deepcopy(models._adoptions[reference])
    prefix = "sessions/" + CONSUMER + "/model-adoption/" + PACK + "/"
    if mutation == "replacement":
        first = snapshot["file_refs"][0]
        changed = case.objects.install(
            prefix + "ordinary-replacement",
            b"Ordinary replacement fixture bytes.",
            first["format"],
        )
        snapshot["file_refs"][0] = {**asdict(changed), "path": first["path"]}
    elif mutation == "omission":
        snapshot["file_refs"].pop()
    else:
        snapshot["catalog_version"] = "2"
    metadata = {key: value for key, value in snapshot.items() if key != "adoption_ref"}
    ref = case.objects.install(prefix + "ordinary-reconciled-adoption.json", _bytes(metadata))
    snapshot["adoption_ref"] = asdict(ref)
    key = case.provider._key(PACK, CONSUMER)
    committed = json.loads(case.rows.rows[key]["record"]["S"])
    assert committed["status"] == "COMMITTED"
    committed["snapshot"] = snapshot
    raw = _bytes(committed)
    case.rows.rows[key]["record"] = {"S": raw.decode()}
    case.rows.rows[key]["record_sha256"] = {"S": hashlib.sha256(raw).hexdigest()}
    models._adoptions[reference] = deepcopy(snapshot)
    # These assertions are the corruption fixture's independent self-consistency
    # precondition, not new provider authority or a substitute for original pins.
    assert json.loads(case.rows.rows[key]["record"]["S"])["snapshot"] == snapshot
    assert hashlib.sha256(raw).hexdigest() == case.rows.rows[key]["record_sha256"]["S"]
    actual_version, actual_raw = case.objects.objects[ref.uri.split("/", 3)[3]]
    assert actual_version == ref.version_id and actual_raw == _bytes(metadata)
    return snapshot


@pytest.mark.parametrize("mutation", ["replacement", "omission", "catalog_relabel"])
def test_genuine_restore_and_observe_reject_self_consistent_consumer_change(monkeypatch, tmp_path, mutation):
    case = real_case(monkeypatch, tmp_path)
    live_policy = policy()
    models = SessionModelAcquisitionStore(
        runtime_session_id=CONSUMER,
        adoption_provider=case.provider,
        policy_getter=lambda: live_policy,
    )
    original = deepcopy(case.producer)
    result = models.reuse_verified(
        catalog_id="qwen3_5_9b",
        catalog_version="1",
        pack_ref=PACK,
        policy=live_policy,
    )
    reference = result["request_ref"]
    nominal = deepcopy(models._adoptions[reference])
    assert case.provider.restore(nominal, CONSUMER, live_policy) == nominal
    snapshot = _committed_self_consistent_change(case, models, reference, mutation)
    assert snapshot["producer"] == nominal["producer"]
    assert snapshot["original_artifacts"] == nominal["original_artifacts"]
    before = (case.copies.calls, case.rows.updates, case.objects.writes, len(case.native.requests))
    refused = False
    try:
        case.provider.restore(snapshot, CONSUMER, live_policy)
    except ValueError:
        refused = True
    assert refused, "self-consistent consumer DATA must remain bound to the selected original catalog"
    observed = models.observe(reference)
    assert observed.get("success") is not True and observed.get("acquired") is not True
    assert case.producer == original
    assert (case.copies.calls, case.rows.updates, case.objects.writes, len(case.native.requests)) == before
