"""Protected planning DATA and ordinary spec tools; neutral source fixtures only."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest
from strands import Agent
from strands.hooks.events import AfterToolCallEvent
from test_campaign_runtime_entry import PROMPT, fixture_factory, policy
from test_managed_generation_product import spec_text, state
from test_sdk_prompt_policy import ScriptedModel

from amlas.backends.adapters._util import raw_tool
from amlas.spec import load_spec_from_text
from amlas.tools.spec_generation_tools import spec_dict_to_frontmatter
from amlas.tools.spec_tools import save_spec, validate_spec


def product():
    try:
        return importlib.import_module("amlas.control_plane.campaign_planning")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.campaign_planning":
            pytest.fail("missing-symbol: campaign_planning.CampaignPlanningStore", pytrace=False)
        raise


def test_private_planning_module_exposes_frozen_data_and_explicit_port():
    module = product()
    for name in (
        "CampaignPlanningStore",
        "PlanningBinding",
        "ProtectedPlanningAck",
        "ProtectedPlanningOriginal",
        "ActivePlanningSelection",
        "ProtectedPlanningRead",
        "BoundPlanningRead",
    ):
        assert hasattr(module, name), name


def test_renderer_carries_provided_generation_without_fabricating_pins():
    original = load_spec_from_text(spec_text()).model_dump(mode="json")
    text = spec_dict_to_frontmatter({**original, "description": "Original researched body"})
    restored = load_spec_from_text(text)
    assert restored.generation == load_spec_from_text(spec_text()).generation
    assert restored.baseline.model_dump()["pass_at_1"] is None


def test_restore_original_generation_spec_preserves_exact_ref_and_body(monkeypatch):
    original = state.__wrapped__(monkeypatch)[0]
    text = spec_text() + "\nHuman detail and ordering.\n"
    reference = original.bind_spec(text)
    fresh = state.__wrapped__(monkeypatch)[0]
    answer = fresh.restore_original_spec(
        reference["spec_ref"], text, expected_spec_sha256=reference["spec_sha256"]
    )
    assert answer == reference
    assert fresh._specs[reference["spec_ref"]][0] == text.encode()
    assert fresh.bind_spec(text) == reference


def test_public_save_schema_excludes_private_planning_authority():
    properties = save_spec.tool_spec["inputSchema"]["json"]["properties"]
    assert set(properties) == {"yaml_text", "path"}
    assert "tool_context" not in properties


def fake_disk(monkeypatch):
    calls = []

    class Target:
        parent = None

        def mkdir(self, **kwargs):
            calls.append(("mkdir", kwargs))

        def write_text(self, text, *, encoding):
            calls.append(("write", text, encoding))

        def __str__(self):
            return "./human-spec.md"

    target = Target()
    target.parent = target
    monkeypatch.setattr("amlas.tools.pathref.resolve_dataset_path", lambda _path: target)
    return calls


def test_legacy_save_preserves_exact_human_text_without_planning(monkeypatch):
    calls = fake_disk(monkeypatch)
    text = spec_text() + "\nHuman edits stay exactly here.\n"
    answer = raw_tool(save_spec)(yaml_text=text, path="./human-spec.md")
    assert answer["saved"] is True
    assert calls[-1] == ("write", text, "utf-8")


def test_validation_remains_pure_and_cannot_publish(monkeypatch):
    calls = fake_disk(monkeypatch)
    answer = raw_tool(validate_spec)(spec_text())
    assert answer["valid"] is True and calls == []
    assert "planning_ref" not in answer and "published" not in answer


def test_active_campaign_missing_private_hook_never_writes_or_falls_back(monkeypatch):
    cr, _, _, factory, *_ = fixture_factory()
    entry = factory.begin_prompt(PROMPT, policy=policy())
    calls = fake_disk(monkeypatch)
    with cr.activate_campaign_entry(entry):
        answer = raw_tool(save_spec)(
            yaml_text=spec_text(),
            path="./human-spec.md",
            tool_context=SimpleNamespace(invocation_state={}),
        )
    assert answer.get("saved") is False and answer.get("published") is False, answer
    assert answer["status"] == "denied" and calls == []


def test_caller_dictionary_cannot_supply_private_planning_hook(monkeypatch):
    calls = fake_disk(monkeypatch)
    answer = raw_tool(save_spec)(
        yaml_text=spec_text(),
        path="./human-spec.md",
        tool_context=SimpleNamespace(
            invocation_state={"_amlas_campaign_planning_store": {"published": True}}
        ),
    )
    assert answer.get("saved") is False and answer.get("published") is False, answer
    assert answer["status"] == "denied" and calls == []


class ProtectedPort:
    """Explicit atomic/conditional neutral port; no native protection is claimed."""

    def __init__(self, module):
        self.module = module
        self.active = module.ProtectedPlanningRead("ABSENT")
        self.originals, self.confirmed = {}, {}
        self.current, self.writes, self.mode = True, 0, "success"
        self.token = object()

    def require_current(self, binding, *, selection, original, qualify_io):
        qualify_io()
        assert self.current, "neutral current origin revoked"
        qualify_io()
        return self.token

    def origin(self, port, binding, selection, original, evidence):
        assert port is self and evidence is self.token and self.current
        if original is not None:
            assert self.confirmed[original.ack.reference] == original
            assert selection.ack == original.ack

    def read_active(self, binding, *, qualify_io):
        qualify_io()
        answer = copy.deepcopy(self.active)
        qualify_io()
        return answer

    def read_original(self, binding, *, reference, qualify_io):
        qualify_io()
        answer = self.originals.get(reference, self.module.ProtectedPlanningRead("ABSENT"))
        qualify_io()
        return copy.deepcopy(answer)

    def publish_first(self, binding, *, original_ref, canonical_body, expected_active, qualify_io):
        qualify_io()
        assert expected_active == self.active.selection
        self.writes += 1
        if self.mode == "unknown":
            self.active = self.module.ProtectedPlanningRead("UNKNOWN")
            raise TimeoutError("neutral unknown first acknowledgement")
        value = json.loads(canonical_body)
        previous = None if expected_active is None else expected_active.ack.publication_sha256
        revision = 1 if expected_active is None else expected_active.revision + 1
        body = hashlib.sha256(canonical_body).hexdigest()
        ack = self.module.ProtectedPlanningAck(
            original_ref,
            "protected-v" + str(revision),
            value["raw_sha256"],
            value["spec_sha256"],
            body,
            hashlib.sha256(("protected/" + body).encode()).hexdigest(),
        )
        selection = self.module.ActivePlanningSelection(ack, revision, previous)
        original = self.module.ProtectedPlanningOriginal(ack, canonical_body)
        self.active = self.module.ProtectedPlanningRead("PRESENT", selection, original)
        self.originals[original_ref] = copy.deepcopy(self.active)
        self.confirmed[original_ref] = copy.deepcopy(original)
        qualify_io()
        if self.mode == "lost":
            raise TimeoutError("neutral lost first acknowledgement")
        if self.mode == "revoke":
            self.current = False
        return selection


def generation_for(monkeypatch, entry):
    store = state.__wrapped__(monkeypatch)[0]
    store.runtime_session_id = entry.private_store_correlation
    for component in (store._models, store._cases, store._sft):
        component.runtime_session_id = entry.private_store_correlation
    return store


def planning_fixture(monkeypatch, *, port=None, bound=None):
    module = product()
    cr, _, _, factory, guard, *_ = fixture_factory()
    entry = factory.begin_prompt(PROMPT, policy=policy())
    scope = guard.approved.manifest
    binding = module.PlanningBinding(
        scope.account_id,
        scope.region,
        scope.deployment_id,
        scope.deployment_prefix,
        entry.campaign_ref,
        entry.campaign_id,
        entry.private_store_correlation,
        entry.prompt_sha256,
        hashlib.sha256(b"independently retained fixture C").hexdigest(),
        hashlib.sha256(b"independently retained fixture H").hexdigest(),
    )
    protected = ProtectedPort(module) if port is None else port
    bound = module.BoundPlanningRead("UNBOUND") if bound is None else bound

    def reader(_binding, *, qualify_io):
        qualify_io()
        assert protected.current
        qualify_io()
        return bound

    store = module.CampaignPlanningStore(
        entry,
        binding,
        generation_for(monkeypatch, entry),
        protected_port=protected,
        origin_qualifier=protected.origin,
        bound_spec_reader=reader,
    )
    return module, cr, entry, store, protected


def test_normal_sdk_save_publishes_original_data_and_never_uses_path(monkeypatch):
    _module, cr, entry, store, protected = planning_fixture(monkeypatch)
    store.observe_tool_result(
        "research_search",
        {
            "results": [
                {
                    "url": "https://example.org/researched-source",
                    "sources": ["TAVILY"],
                    "published": None,
                }
            ],
            "coverage": "queried_sources_only; failures remain explicit",
            "errors": {"semantic_scholar": "neutral unavailable source"},
        },
        observed_at=42.0,
    )
    calls = fake_disk(monkeypatch)
    text = spec_text() + "\nExact human-owned text.\n"
    model = ScriptedModel(
        [[{"id": "spec", "name": "save_spec", "input": {"yaml_text": text, "path": "./unused.md"}}]]
    )
    agent = Agent(model=model, tools=[save_spec], callback_handler=lambda **kwargs: None)
    outputs = []

    def observe(event):
        for block in event.result["content"]:
            if "json" in block:
                outputs.append(block["json"])
            elif "text" in block:
                try:
                    value = json.loads(block["text"])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    outputs.append(value)

    agent.hooks.add_callback(AfterToolCallEvent, observe)
    with cr.activate_campaign_entry(entry):
        agent(
            "Save the original researched planning data.",
            invocation_state={
                "_amlas_campaign_planning_store": store,
            },
        )
    assert outputs[0]["published"] is True and outputs[0]["saved"] is False
    assert calls == [] and protected.writes == 1
    assert entry.planning_refs == ()  # Root owns a new sealed view; this entry is immutable.
    refs = store.load_original_refs(
        campaign_ref=entry.campaign_ref, private_store_correlation=entry.private_store_correlation
    )
    assert refs[0].reference == outputs[0]["planning_ref"]
    assert len(refs) == 2 and refs[1].kind == "research"
    assert protected.active.original.canonical_body
    body = json.loads(protected.active.original.canonical_body)
    assert body["raw_text"] == text
    assert body["research"][0]["untrusted_data"] is True
    assert body["research"][0]["license_coverage"] == "unknown"
    assert body["research"][0]["error_sources"] == ["semantic_scholar"]


@pytest.mark.parametrize("mode", ["success", "lost"])
def test_duplicate_or_lost_ack_preserves_original_ref_without_write_replay(monkeypatch, mode):
    _module, _cr, _entry, store, protected = planning_fixture(monkeypatch)
    protected.mode = mode
    text = spec_text()
    first = store.publish_validated_spec(text)
    second = store.publish_validated_spec(text)
    assert first == second and protected.writes == 1


def test_unknown_first_ack_never_replays_even_with_later_absence(monkeypatch):
    module, _cr, _entry, store, protected = planning_fixture(monkeypatch)
    protected.mode = "unknown"
    with pytest.raises(module.CampaignPlanningError):
        store.publish_validated_spec(spec_text())
    protected.mode = "success"
    protected.active = module.ProtectedPlanningRead("ABSENT")
    with pytest.raises(module.CampaignPlanningError):
        store.publish_validated_spec(spec_text())
    assert protected.writes == 1


def test_cold_fresh_store_restores_exact_original_ref_without_remint(monkeypatch):
    module, _cr, entry, first, protected = planning_fixture(monkeypatch)
    text = spec_text() + "\nOriginal text and ordering.\n"
    original = first.publish_validated_spec(text)
    fresh = module.CampaignPlanningStore(
        entry,
        first.binding,
        generation_for(monkeypatch, entry),
        protected_port=protected,
        origin_qualifier=protected.origin,
    )
    restored = fresh.restore_original_spec()
    assert restored == original and protected.writes == 1
    assert fresh._generation._specs[original.reference][0] == text.encode()
    assert fresh._generation.bind_spec(text)["spec_ref"] == original.reference


def test_coherent_mutable_read_replacement_is_not_original_origin(monkeypatch):
    module, _cr, entry, store, protected = planning_fixture(monkeypatch)
    original = store.publish_validated_spec(spec_text())
    value = json.loads(protected.active.original.canonical_body)
    value["raw_text"] += "\nForeign changed body.\n"
    from amlas.arbiter.coding import coding_spec_bytes, coding_spec_sha256

    changed = load_spec_from_text(value["raw_text"])
    value["raw_sha256"] = hashlib.sha256(value["raw_text"].encode()).hexdigest()
    value["spec_projection"] = json.loads(coding_spec_bytes(changed))
    value["spec_sha256"] = coding_spec_sha256(changed)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ack = dataclasses.replace(
        protected.active.selection.ack,
        raw_sha256=value["raw_sha256"],
        spec_sha256=value["spec_sha256"],
        body_sha256=hashlib.sha256(raw).hexdigest(),
    )
    protected.active = module.ProtectedPlanningRead(
        "PRESENT",
        dataclasses.replace(protected.active.selection, ack=ack),
        module.ProtectedPlanningOriginal(ack, raw),
    )
    with pytest.raises(AssertionError):
        store.load_original_refs(
            campaign_ref=entry.campaign_ref, private_store_correlation=entry.private_store_correlation
        )
    assert original.reference in protected.confirmed and protected.writes == 1


@pytest.mark.parametrize("bound", ["BOUND", "UNKNOWN", "missing"])
def test_active_update_cannot_redirect_bound_or_unqualified_model_spec(monkeypatch, bound):
    module, _cr, _entry, store, protected = planning_fixture(monkeypatch)
    original = store.publish_validated_spec(spec_text())
    if bound == "missing":
        store._bound_reader = None
    else:
        result = (
            module.BoundPlanningRead(bound, original.reference, original.sha256)
            if bound == "BOUND"
            else module.BoundPlanningRead("UNKNOWN")
        )
        store._bound_reader = lambda _binding, **kwargs: result
    with pytest.raises(module.CampaignPlanningError):
        store.publish_validated_spec(spec_text(project_id="changed_plan"))
    assert protected.writes == 1


def test_current_origin_revocation_refuses_success_after_publication(monkeypatch):
    _module, _cr, _entry, store, protected = planning_fixture(monkeypatch)
    protected.mode = "revoke"
    with pytest.raises(AssertionError):
        store.publish_validated_spec(spec_text())
    assert protected.writes == 1


def test_default_missing_capability_and_duck_entry_cannot_publish(monkeypatch):
    module, _cr, entry, store, protected = planning_fixture(monkeypatch)
    missing = module.CampaignPlanningStore(entry, store.binding, generation_for(monkeypatch, entry))
    with pytest.raises(module.CampaignPlanningError):
        missing.publish_validated_spec(spec_text())
    with pytest.raises(module.CampaignPlanningError):
        module.CampaignPlanningStore(
            SimpleNamespace(campaign_ref=entry.campaign_ref), store.binding, store._generation
        )
    assert protected.writes == 0
