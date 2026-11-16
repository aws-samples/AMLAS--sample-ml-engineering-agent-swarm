"""Normal SDK planning wiring; synthetic protected ports, no native admission."""

import contextvars
import copy
import importlib
import json

import pytest
from test_campaign_planning_publication import fake_disk, planning_fixture, spec_text
from test_normal_chat_live_research import install_sources
from test_sdk_acquisition_binding import invoke, request

from amlas.config.settings import Settings
from amlas.control_plane.campaign_runtime import activate_campaign_entry, activate_campaign_runtime
from amlas.tools.research import strands_tools
from amlas.tools.spec_tools import save_spec
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend


def services_module():
    try:
        return importlib.import_module("amlas.control_plane.campaign_planning_services")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.campaign_planning_services":
            pytest.fail("missing-symbol: campaign_planning_services", pytrace=False)
        raise


def neutral_selectors(monkeypatch):
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *a, **kw: "")


def test_private_selected_planning_services_exist():
    m = services_module()
    for name in (
        "CampaignPlanningServices",
        "activate_campaign_planning_services",
        "current_campaign_planning_services",
    ):
        assert hasattr(m, name), "missing-symbol: " + name


@pytest.mark.parametrize("route", ["sdk", "direct"])
def test_existing_private_store_reaches_normal_save_context_without_disk(monkeypatch, route):
    neutral_selectors(monkeypatch)
    _, _cr, entry, store, protected = planning_fixture(monkeypatch)
    disk = fake_disk(monkeypatch)
    with activate_campaign_runtime(entry._factory), activate_campaign_entry(entry):
        backend = ToolBindingBackend(Settings())
        backend._campaign_planning_store = store  # private fixture injection, no public authority
        if route == "sdk":
            answer = invoke(
                monkeypatch,
                backend,
                [request("spec", "save_spec", {"yaml_text": spec_text(), "path": "./unused.md"})],
                tools=[save_spec],
            )["spec"]
        else:
            answer = backend.execute_tool_use(
                "save_spec", {"yaml_text": spec_text(), "path": "./unused.md"}, confirmed=True
            ).result
    assert answer["published"] is True and answer["saved"] is False
    assert disk == [] and protected.writes == 1
    assert entry.planning_refs == ()


def test_real_sdk_research_result_is_captured_before_spec_publication(monkeypatch):
    neutral_selectors(monkeypatch)
    source_calls = install_sources(monkeypatch)
    import test_sdk_acquisition_binding as sdk_helper
    from test_sdk_prompt_policy import ScriptedModel

    class PlanningCycles(ScriptedModel):
        def __init__(self, batches):
            super().__init__([[batches[0][0]], [batches[0][1]]])
            self.tool_cycles = 2

        async def stream(self, *args, **kwargs):
            if self.tool_cycles:
                self.pending_final = False
                self.tool_cycles -= 1
            async for event in super().stream(*args, **kwargs):
                yield event

    monkeypatch.setattr(sdk_helper, "ScriptedModel", PlanningCycles)
    _, _, entry, store, protected = planning_fixture(monkeypatch)
    disk = fake_disk(monkeypatch)
    with activate_campaign_runtime(entry._factory), activate_campaign_entry(entry):
        backend = ToolBindingBackend(Settings())
        backend._campaign_planning_store = store
        output = invoke(
            monkeypatch,
            backend,
            [
                request(
                    "research",
                    "research_search",
                    {"query": "Neutral public coding research", "max_results": 1},
                ),
                request("spec", "save_spec", {"yaml_text": spec_text(), "path": "./unused.md"}),
            ],
            tools=[strands_tools.research_search, save_spec],
        )
    assert source_calls and output["spec"]["published"] is True
    body = json.loads(protected.active.original.canonical_body)
    assert body["research"] and body["research"][0]["tool"] == "research_search"
    assert body["research"][0]["untrusted_data"] is True and disk == []


def service_fixture(monkeypatch):
    neutral_selectors(monkeypatch)
    module, _, entry, _store, protected = planning_fixture(monkeypatch)
    m = services_module()
    services = m.CampaignPlanningServices(
        runtime_factory=entry._factory,
        protected_port=protected,
        origin_qualifier=protected.origin,
        bound_spec_reader=lambda binding, *, qualify_io: module.BoundPlanningRead("UNBOUND"),
    )
    return m, entry, services, protected


def test_selected_services_construct_original_binding_and_normal_hidden_store(monkeypatch):
    m, entry, services, protected = service_fixture(monkeypatch)
    with (
        activate_campaign_runtime(entry._factory),
        activate_campaign_entry(entry),
        m.activate_campaign_planning_services(services),
    ):
        backend = ToolBindingBackend(Settings())
        store = backend._campaign_planning_store
        assert store.entry is entry and store._generation is backend._generation_store
        assert backend._invocation_state()["_amlas_campaign_planning_store"] is store
        assert store.binding.campaign_ref == entry.campaign_ref
        assert protected.writes == 0


@pytest.mark.parametrize("mode", ["foreign_factory", "inherited_pid", "expired_copy"])
def test_selected_services_reject_wrong_original_or_expired_selection(monkeypatch, mode):
    m, entry, services, _protected = service_fixture(monkeypatch)
    if mode == "expired_copy":
        with m.activate_campaign_planning_services(services):
            saved = contextvars.copy_context()
        with pytest.raises(ValueError):
            saved.run(m.current_campaign_planning_services)
    elif mode == "inherited_pid":
        monkeypatch.setattr(services, "_pid", services._pid + 1)
        with pytest.raises(ValueError):
            with m.activate_campaign_planning_services(services):
                pass
    else:
        foreign = copy.copy(entry._factory).begin_prompt(entry.literal_prompt)
        from test_campaign_planning_publication import generation_for

        with pytest.raises(ValueError):
            services.build_for_entry(foreign, generation_for(monkeypatch, foreign))


@pytest.mark.parametrize(
    "payload",
    [
        {"content": [{"text": '{"coverage":NaN}'}]},
        {"content": [{"text": '{"coverage":1e999}'}]},
        {"content": [{"text": "x" * 65537}]},
        {"content": [{"json": {"value": object()}}]},
    ],
)
def test_actual_sdk_result_decoder_refuses_nonfinite_or_unbounded_data(payload):
    with pytest.raises(ValueError):
        ToolBindingBackend._planning_result_payload(payload)


def test_normal_literal_api_publishes_spec_through_selected_services_without_disk(monkeypatch):
    import asyncio

    from test_sdk_prompt_policy import ScriptedModel

    from amlas import api
    from amlas.tui.chat import tool_binding_backend as backend_module

    m, entry, services, protected = service_fixture(monkeypatch)
    disk = fake_disk(monkeypatch)
    original = ToolBindingBackend.build_agent
    backends = []

    def build(backend, **kwargs):
        backend._model = ScriptedModel(
            [[request("spec", "save_spec", {"yaml_text": spec_text(), "path": "./unused.md"})]]
        )
        backends.append(backend)
        agent = original(backend, **kwargs)
        agent.callback_handler = lambda *args, **kwargs: None
        return agent

    monkeypatch.setattr(backend_module, "get_safe_llm_tools", lambda: [save_spec])
    monkeypatch.setattr(ToolBindingBackend, "build_agent", build)
    with (
        activate_campaign_runtime(entry._factory),
        activate_campaign_entry(entry),
        m.activate_campaign_planning_services(services),
    ):
        answer = asyncio.run(api.run_prompt(entry.literal_prompt))
    assert answer is not None and backends and protected.writes == 1 and disk == []
    assert protected.active.original.canonical_body and entry.planning_refs == ()


def test_cold_backend_restores_same_qualified_original_spec_reference(monkeypatch):
    m, entry, services, protected = service_fixture(monkeypatch)
    disk = fake_disk(monkeypatch)
    with (
        activate_campaign_runtime(entry._factory),
        activate_campaign_entry(entry),
        m.activate_campaign_planning_services(services),
    ):
        first = ToolBindingBackend(Settings())
        original = first._campaign_planning_store.publish_validated_spec(spec_text())
        cold = ToolBindingBackend(Settings())
        restored = cold._campaign_planning_store.restore_original_spec()
        refs = cold._campaign_planning_store.load_original_refs(
            campaign_ref=entry.campaign_ref, private_store_correlation=entry.private_store_correlation
        )
    assert restored == original and refs[0] == original and protected.writes == 1 and disk == []
    assert cold._generation_store._specs[original.reference][0] == spec_text().encode()


def test_default_legacy_state_omits_hook_and_active_missing_services_denies_before_disk(monkeypatch):
    neutral_selectors(monkeypatch)
    legacy = ToolBindingBackend(Settings())
    assert "_amlas_campaign_planning_store" not in legacy._invocation_state()
    _, _, entry, _store, protected = planning_fixture(monkeypatch)
    disk = fake_disk(monkeypatch)
    with activate_campaign_runtime(entry._factory), activate_campaign_entry(entry):
        active = ToolBindingBackend(Settings())
        result = active.execute_tool_use(
            "save_spec", {"yaml_text": spec_text(), "path": "./unused.md"}, confirmed=True
        ).result
    assert result["status"] == "denied" and result["published"] is False
    assert disk == [] and protected.writes == 0


def test_selected_services_missing_protected_capability_does_not_construct_usable_hook(monkeypatch):
    neutral_selectors(monkeypatch)
    _, _, entry, _store, protected = planning_fixture(monkeypatch)
    m = services_module()
    services = m.CampaignPlanningServices(runtime_factory=entry._factory)
    with (
        activate_campaign_runtime(entry._factory),
        activate_campaign_entry(entry),
        m.activate_campaign_planning_services(services),
    ):
        with pytest.raises(ValueError):
            ToolBindingBackend(Settings())
    assert protected.writes == 0
