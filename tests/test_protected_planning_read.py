"""Read the same protected original planning through pre-entry and model seams."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json

import pytest
from test_campaign_coordinator import PROMPT
from test_campaign_planning_publication import ProtectedPort, generation_for
from test_model_campaign_bridge import bridge_fixture

from amlas.arbiter.coding import coding_spec_bytes
from amlas.control_plane import campaign_planning as planning
from amlas.control_plane.campaign_runtime import CampaignRuntimeFactory
from amlas.spec import load_spec_from_text


def product():
    try:
        return importlib.import_module("amlas.control_plane.protected_planning_read")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.protected_planning_read":
            pytest.fail(
                "missing-symbol: protected_planning_read.ProtectedCampaignPlanningReader", pytrace=False
            )
        raise


class InitializationPort:
    """The exact source accessor is used; no native supplier implementation is claimed."""

    def __init__(self, runtime, entry):
        self.runtime, self.entry, self.current = runtime, entry, True

    def current(self):
        if not self.current:
            raise ValueError("neutral current supplier revoked")

    def read_current_scope(self, *, qualify_io):
        qualify_io()
        value = self.runtime.coordinator._scope(self.runtime.guard.verify())
        qualify_io()
        return {name: value[name] for name in ("account_id", "region", "deployment_id", "deployment_prefix")}

    def read_original_planning(self, *, campaign_ref, private_store_correlation, qualify_io):
        qualify_io()
        observation = self.runtime.initializer.resolve_original_planning(
            self.runtime.guard.verify(),
            self.entry.transport_session_id,
            literal_prompt=self.entry.literal_prompt,
            qualify_io=qualify_io,
        )
        assert observation.original.campaign_ref == campaign_ref
        assert observation.original.private_store_correlation == private_store_correlation
        qualify_io()
        return observation


def setup(monkeypatch, *, publish=True):
    module = product()
    x = bridge_fixture(monkeypatch)
    initialization = InitializationPort(x.runtime, x.entry)

    def current():
        assert initialization.current
        x.entry.qualify_current_io()

    scope = initialization.read_current_scope(qualify_io=current)
    observed = initialization.read_original_planning(
        campaign_ref=x.entry.campaign_ref,
        private_store_correlation=x.entry.private_store_correlation,
        qualify_io=current,
    )
    binding = planning.PlanningBinding(
        **scope,
        campaign_ref=observed.original.campaign_ref,
        campaign_id=observed.original.campaign_id,
        private_store_correlation=observed.original.private_store_correlation,
        prompt_sha256=observed.original.prompt_sha256,
        initial_campaign_body_sha256=observed.initial_campaign_body_sha256,
        initialization_publication_sha256=observed.initialization_publication_sha256,
    )
    protected = ProtectedPort(planning)

    def origin(
        initialization_port, protected_port, selected_binding, observation, selection, original, evidence
    ):
        assert initialization_port is initialization and initialization.current
        assert selected_binding == binding
        assert observation.initial_campaign_body_sha256 == observed.initial_campaign_body_sha256
        assert observation.initialization_publication_sha256 == observed.initialization_publication_sha256
        assert observation.original.campaign_ref == observed.original.campaign_ref
        assert observation.original.private_store_correlation == observed.original.private_store_correlation
        protected.origin(protected_port, selected_binding, selection, original, evidence)

    from test_managed_generation_product import spec_text

    text = spec_text() + "\nOriginal researched notes and human text.\n"
    publisher = planning.CampaignPlanningStore(
        x.entry,
        binding,
        generation_for(monkeypatch, x.entry),
        protected_port=protected,
        origin_qualifier=protected.origin,
    )
    ref = publisher.publish_validated_spec(text) if publish else None
    reader = module.ProtectedCampaignPlanningReader(
        initialization_port=initialization,
        protected_port=protected,
        origin_qualifier=origin,
        qualify_current=current,
    )
    return module, x, initialization, protected, reader, ref, text


def test_pre_entry_reader_needs_no_entry_or_generation_constructor(monkeypatch):
    _module, x, _initialization, protected, reader, ref, _text = setup(monkeypatch)
    from amlas.control_plane.campaign_runtime import CampaignEntryContext
    from amlas.control_plane.generation_state import SessionCodingGenerationStore

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only adapter must not construct ENTRY or generation state")

    monkeypatch.setattr(CampaignEntryContext, "__init__", forbidden)
    monkeypatch.setattr(SessionCodingGenerationStore, "__init__", forbidden)
    before = x.runtime._planning
    refs = reader.load_original_refs(
        campaign_ref=x.entry.campaign_ref, private_store_correlation=x.entry.private_store_correlation
    )
    assert refs == (ref,) and x.runtime._planning is before and protected.writes == 1


def test_model_adapter_returns_exact_full_canonical_json_and_separate_raw_text(monkeypatch):
    module, x, _initialization, protected, reader, ref, text = setup(monkeypatch)
    adapter = module.ModelPlanningPortAdapter(reader)
    view = reader.load_original(
        campaign_ref=x.entry.campaign_ref,
        private_store_correlation=x.entry.private_store_correlation,
        reference=ref.reference,
        qualify_io=x.entry.qualify_current_io,
    )
    runtime = new_runtime(x, reader)
    entry = runtime.begin_prompt(PROMPT, policy=x.policy)
    body = adapter.load_original(entry, qualify_io=entry.qualify_current_io)
    assert body.reference == ref.reference and body.sha256 == ref.sha256
    assert body.body == coding_spec_bytes(load_spec_from_text(text))
    assert body.body_format == "canonical_spec_json_v1"
    assert hashlib.sha256(body.body).hexdigest() == body.sha256
    assert view.raw_text == text and view.canonical_spec == body.body
    assert json.loads(body.body)["domain_notes"] == load_spec_from_text(text).domain_notes
    assert protected.writes == 1


def new_runtime(x, reader):
    original = x.runtime
    return CampaignRuntimeFactory(
        guard=original.guard,
        initializer=original.initializer,
        coordinator=original.coordinator,
        approved_ingress=original._approved,
        ingress_reader=original._ingress,
        private_observer=original._observer,
        planning_reader=reader,
        terminal_verifier=original._terminal,
        clock=original._clock,
    )


def test_actual_model_bridge_consumes_same_original_body_and_cold_ref(monkeypatch):
    module, x, _initialization, protected, reader, ref, _text = setup(monkeypatch)
    adapter = module.ModelPlanningPortAdapter(reader)
    runtime = new_runtime(x, reader)
    entry = runtime.begin_prompt(PROMPT, policy=x.policy)
    services = x.m.ModelCampaignServices(
        runtime_factory=runtime,
        planning_port=adapter,
        planning_origin_qualifier=adapter.qualify_model_origin,
        config_port=x.config,
        config_origin_qualifier=x.services._config_origin,
        wake_hint_port=x.hint,
        wake_hint_origin_qualifier=x.services._hint_origin,
        journal_factory=x.services._journal_factory,
        current_boundary_qualifier=x.services._boundary,
    )
    binding = services.build_for_entry(entry, x.adapter)
    result = binding.submit_original("qwen3_5_9b", "1", x.policy)
    assert result.executed and len(x.native.requests) == 1
    cold = services.build_for_entry(runtime.refresh_original_planning(entry), x.adapter)
    restored = cold.submit_original("qwen3_5_9b", "1", x.policy)
    assert restored.reference == result.reference and len(x.native.requests) == 1
    assert entry.planning_refs[0] == ref and protected.writes == 1


def test_qualified_absence_is_empty_before_entry_but_not_a_model_body(monkeypatch):
    module, x, _initialization, protected, reader, _ref, _text = setup(monkeypatch, publish=False)
    assert (
        reader.load_original_refs(
            campaign_ref=x.entry.campaign_ref, private_store_correlation=x.entry.private_store_correlation
        )
        == ()
    )
    with pytest.raises(module.ProtectedPlanningReadError):
        reader.load_original(
            campaign_ref=x.entry.campaign_ref,
            private_store_correlation=x.entry.private_store_correlation,
            qualify_io=x.entry.qualify_current_io,
        )
    assert protected.writes == 0


@pytest.mark.parametrize("kind", ["scope", "supplier", "unknown", "origin"])
def test_current_scope_supplier_unknown_and_origin_fail_closed(monkeypatch, kind):
    module, x, initialization, protected, reader, _ref, _text = setup(monkeypatch)
    if kind == "scope":
        old = initialization.read_current_scope
        initialization.read_current_scope = lambda **kwargs: {**old(**kwargs), "native_admission": True}
    elif kind == "supplier":
        initialization.current = False
    elif kind == "unknown":
        protected.active = planning.ProtectedPlanningRead("UNKNOWN")
    else:
        protected.current = False
    with pytest.raises((module.ProtectedPlanningReadError, AssertionError)):
        reader.load_original_refs(
            campaign_ref=x.entry.campaign_ref, private_store_correlation=x.entry.private_store_correlation
        )
    assert protected.writes == 1


def test_coherent_replaced_read_never_becomes_qualified_original(monkeypatch):
    _module, x, _initialization, protected, reader, _ref, _text = setup(monkeypatch)
    raw = protected.active.original.canonical_body
    ack = dataclasses.replace(protected.active.original.ack, version_id="foreign-v1")
    protected.active = planning.ProtectedPlanningRead(
        "PRESENT",
        dataclasses.replace(protected.active.selection, ack=ack),
        planning.ProtectedPlanningOriginal(ack, raw),
    )
    with pytest.raises(AssertionError):
        reader.load_original_refs(
            campaign_ref=x.entry.campaign_ref, private_store_correlation=x.entry.private_store_correlation
        )


def test_fresh_reader_keeps_cold_original_ref_and_body(monkeypatch):
    module, x, initialization, protected, reader, ref, text = setup(monkeypatch)
    cold = module.ProtectedCampaignPlanningReader(
        initialization_port=initialization,
        protected_port=protected,
        origin_qualifier=reader._origin_qualifier,
        qualify_current=reader._qualify_current,
    )
    view = cold.load_original(
        campaign_ref=x.entry.campaign_ref,
        private_store_correlation=x.entry.private_store_correlation,
        reference=ref.reference,
        qualify_io=x.entry.qualify_current_io,
    )
    assert view.refs[0] == ref and view.raw_text == text and protected.writes == 1


def test_missing_native_supplier_or_duck_entry_cannot_create_model_port(monkeypatch):
    module, x, _initialization, _protected, reader, _ref, _text = setup(monkeypatch)
    with pytest.raises(module.ProtectedPlanningReadError):
        module.ProtectedCampaignPlanningReader()
    with pytest.raises(module.ProtectedPlanningReadError):
        module.ModelPlanningPortAdapter(reader).load_original(
            copy.copy(x.entry).__dict__, qualify_io=x.entry.qualify_current_io
        )
