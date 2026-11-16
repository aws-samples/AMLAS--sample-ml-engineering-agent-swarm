"""Finite protected planning formats; strict neutral bridge, no native effects."""

import hashlib
import json

import pytest
from test_model_campaign_bridge import bridge_fixture

from amlas.spec.loader import SpecLoadError


def select_body(x, raw, body_format):
    from amlas.control_plane.campaign_runtime import CampaignPlanningRef

    body = x.m.ModelPlanningBody(
        "spec_canonical", hashlib.sha256(raw).hexdigest(), raw, body_format=body_format
    )
    x.planning.current = body
    x.planning.load_original_refs = lambda **kwargs: (
        CampaignPlanningRef("spec", body.reference, body.sha256),
    )
    x.services._planning_origin = lambda port, selected, entry, evidence: None
    entry = x.runtime.refresh_original_planning(x.entry)
    return entry


def canonical_body(x):
    from amlas.arbiter.coding import coding_spec_bytes
    from amlas.spec import load_spec_from_text

    return coding_spec_bytes(load_spec_from_text(x.planning.current.body.decode()))


def test_canonical_protected_spec_crosses_actual_bridge_and_reuses_original(monkeypatch):
    x = bridge_fixture(monkeypatch)
    raw = canonical_body(x)
    entry = select_body(x, raw, "canonical_spec_json_v1")
    first = x.services.build_for_entry(entry, x.adapter).submit_original("qwen3_5_9b", "1", x.policy)
    second = x.services.build_for_entry(entry, x.adapter).submit_original("qwen3_5_9b", "1", x.policy)
    assert first.executed is True and second.executed is False
    assert (first.reference, first.handle.arn) == (second.reference, second.handle.arn)
    assert len(x.native.requests) == 1


@pytest.mark.parametrize(
    "mode",
    [
        "unknown",
        "noncanonical",
        "duplicate",
        "nonfinite",
        "wrong_schema",
        "frontmatter_as_json",
        "json_as_frontmatter",
    ],
)
def test_explicit_closed_body_formats_refuse_before_any_create(monkeypatch, mode):
    x = bridge_fixture(monkeypatch)
    legacy = x.planning.current.body
    raw = canonical_body(x)
    fmt = "canonical_spec_json_v1"
    if mode == "unknown":
        fmt = "arbitrary"
    elif mode == "noncanonical":
        raw = json.dumps(json.loads(raw), indent=2).encode()
    elif mode == "duplicate":
        raw = raw[:-1] + b',"project_id":"neutral_coding"}'
    elif mode == "nonfinite":
        raw = raw[:-1] + b',"untrusted":NaN}'
    elif mode == "wrong_schema":
        raw = b'{"project_id":"neutral_coding"}'
    elif mode == "frontmatter_as_json":
        raw = legacy
    elif mode == "json_as_frontmatter":
        fmt = "frontmatter_v1"
    entry = select_body(x, raw, fmt)
    with pytest.raises((ValueError, SpecLoadError, x.m.ModelCampaignBindingError)):
        x.services.build_for_entry(entry, x.adapter)
    assert x.native.requests == [] and x.protected.writes == 0


def test_legacy_three_field_body_keeps_frontmatter_default(monkeypatch):
    x = bridge_fixture(monkeypatch)
    assert x.planning.current.body_format == "frontmatter_v1"
    x.services.build_for_entry(x.entry, x.adapter)
    assert x.native.requests == []
