"""Fresh same-factory planning snapshots are DATA; original entry stays sealed."""

import copy

import pytest
from test_model_campaign_bridge import bridge_fixture


def test_factory_refresh_constructs_new_original_planning_view(monkeypatch):
    x = bridge_fixture(monkeypatch)
    loader = x.planning.load_original_refs
    monkeypatch.setattr(x.planning, "load_original_refs", lambda **kwargs: ())
    old = x.runtime.begin_prompt(x.entry.literal_prompt, policy=x.policy)
    monkeypatch.setattr(x.planning, "load_original_refs", loader)
    method = getattr(x.runtime, "refresh_original_planning", None)
    assert callable(method), "missing-symbol: CampaignRuntimeFactory.refresh_original_planning"
    fresh = method(old)
    assert fresh is not old and old.planning_refs == () and fresh.planning_refs
    assert (
        fresh.campaign_ref,
        fresh.campaign_id,
        fresh.private_store_correlation,
        fresh.literal_prompt,
        fresh.transport_session_id,
        fresh._factory,
    ) == (
        old.campaign_ref,
        old.campaign_id,
        old.private_store_correlation,
        old.literal_prompt,
        old.transport_session_id,
        old._factory,
    )
    assert x.native.requests == []


@pytest.mark.parametrize("mode", ["foreign_factory", "inherited_pid", "planning_shape", "revoked_ingress"])
def test_factory_refresh_refuses_changed_original_or_unqualified_view(monkeypatch, mode):
    x = bridge_fixture(monkeypatch)
    method = getattr(x.runtime, "refresh_original_planning", None)
    assert callable(method), "missing-symbol: CampaignRuntimeFactory.refresh_original_planning"
    if mode == "foreign_factory":
        foreign = copy.copy(x.runtime).begin_prompt(x.entry.literal_prompt, policy=x.policy)
        target = foreign
    else:
        target = x.entry
    if mode == "inherited_pid":
        monkeypatch.setattr(x.runtime, "_pid", x.runtime._pid + 1)
    elif mode == "planning_shape":
        monkeypatch.setattr(x.planning, "load_original_refs", lambda **kwargs: [])
    elif mode == "revoked_ingress":
        monkeypatch.setattr(x.runtime._ingress, "read_current", lambda **kwargs: None)
    with pytest.raises(ValueError):
        method(target)
    assert x.native.requests == []
