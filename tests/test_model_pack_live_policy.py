"""Current policy replacement must fence actual store adoption after first copy.

Actual original codec/model adapter use only the existing nine tiny synthetic
files. No policy/SDK graph is persisted; no native model or provider is called.
"""

import json
from dataclasses import replace

from test_model_pack_adoption_integration import CONSUMER, PACK, f, policy, real_case

from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore


def test_replaced_live_policy_stops_adoption_before_second_copy_or_alias(monkeypatch, tmp_path):
    case = real_case(monkeypatch, tmp_path)
    original_policy = policy()
    live = {"policy": original_policy}
    store = SessionModelAcquisitionStore(
        runtime_session_id=CONSUMER,
        adoption_provider=case.provider,
        policy_getter=lambda: live["policy"],
    )
    copied = case.copies._copy

    def copy_then_revoke(*args, **kwargs):
        result = copied(*args, **kwargs)
        if case.copies.calls == 1:
            live["policy"] = ApprovalPolicy(mode=ApprovalMode.ASK, max_auto_cost=None)
        return result

    monkeypatch.setattr(case.copies, "_copy", copy_then_revoke)
    # The real model flow passes this snapshot; current authority remains live.
    snapshot = replace(original_policy, allowlist=list(original_policy.allowlist))
    result = None
    refused = False
    try:
        result = store.reuse_verified(
            catalog_id=f.CATALOG,
            catalog_version=f.VERSION,
            pack_ref=PACK,
            policy=snapshot,
        )
    except ValueError:
        refused = True
    assert live["policy"].mode is ApprovalMode.ASK and snapshot.mode is ApprovalMode.AUTO
    assert case.copies.calls == 1, "live_policy_revocation_did_not_stop_second_copy"
    assert refused and result is None
    assert not store._adoptions and not store._adoption_keys and not store._records
    assert case.rows.puts == 1 and case.rows.updates == 0
    assert all(json.loads(row["record"]["S"])["status"] != "COMMITTED" for row in case.rows.rows.values())
    assert len(case.native.requests) == 1
