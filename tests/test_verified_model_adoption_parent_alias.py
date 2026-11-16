"""Returned adoption alias must match the immutable normal parent alias."""

from test_verified_model_adoption_sft_parents import CAMPAIGN, MODEL_REF, PREP_REF, _parent_fixture

from amlas.control_plane import normal_run_parents as parents


def test_cold_adopted_parent_refuses_different_restored_alias(monkeypatch):
    fixture = _parent_fixture(monkeypatch)
    raw = parents.publish_parent_graph(
        fixture.store,
        MODEL_REF,
        PREP_REF,
        run_id=CAMPAIGN.replace("-", ""),
        qualify=fixture.qualify,
    )
    fixture.models._restore_adopted_parent = lambda *args, **kwargs: "model_acq_" + "f" * 32
    try:
        parents.hydrate_parent_graph(raw, fixture.binding, fixture.policy, qualify=fixture.qualify)
    except ValueError as error:
        assert str(error) == "restored adopted model alias differs"
    else:
        raise AssertionError("cold restore exposed parents despite a different restored adoption alias")
