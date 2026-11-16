"""Normal API/ChatEngine finite research profile; actual SDK, inert transports.

Only trusted server config selects the source plan. No direct binder invocation,
native guard/ingress, full lifecycle or correctness of source claims is proved.
"""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from test_input_admission_wiring import configure_guard
from test_research_input_recovery import (
    DENIED,
    INDEPENDENT,
    RETURNED_REF,
    WITHHELD,
    GuardClient,
    RecoveryModel,
)

from amlas import api
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalPolicy
from amlas.tools.research import public_fetch, strands_tools
from amlas.tui.chat import engine as engine_module
from amlas.tui.chat import tool_binding_backend as binding

PROFILE_ENV = "AMLAS_RUNTIME_RESEARCH_PROFILE"
PROFILE = "qwen9b_core_v1"
MODEL_URL = "https://huggingface.co/Qwen/Qwen3.5-9B/raw/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md"
FAMILY_URL = (
    "https://raw.githubusercontent.com/QwenLM/Qwen3.8/2ea10dc725823bf7c3e21ce8557cbe15245132ae/README.md"
)
MODEL_URLS = (MODEL_URL, FAMILY_URL)
TRAIN_URL = "https://huggingface.co/datasets/bigcode/commitpackft/raw/fc56fe33c030c6daa414c2b112c932b8eed085e6/README.md"
CASE_URLS = (
    "https://huggingface.co/datasets/openai/openai_humaneval/raw/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/README.md",
    "https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/LICENSE",
)
PROMPT = "Research a licensed coding corpus and benchmark to fine-tune one Qwen 9B model."


def setup(
    monkeypatch,
    *,
    profile=PROFILE,
    original="blocked",
    change=None,
    second=None,
    model_override=None,
    extra_tools=(),
):
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *_a, **_kw: "")
    if profile is None:
        monkeypatch.delenv(PROFILE_ENV, raising=False)
    else:
        monkeypatch.setenv(PROFILE_ENV, profile)
    model, fetches, backends, agents = model_override or RecoveryModel(second), [], [], []
    model.requests[0] = (
        "original-readonly-use",
        "research_fetch_urls",
        {"urls": list(MODEL_URLS), "max_pages": 3},
    )
    if second is None:
        model.requests[1] = (
            "independent-readonly-use",
            "research_fetch_urls",
            {"urls": [TRAIN_URL], "max_pages": 3},
        )
    model.system_prompts = []
    stream = model.stream

    async def capture_stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        model.system_prompts.append(copy.deepcopy(system_prompt))
        async for event in stream(messages, tool_specs, system_prompt, **kwargs):
            yield event

    monkeypatch.setattr(model, "stream", capture_stream)

    class Client(GuardClient):
        def apply_guardrail(self, **kwargs):
            text = kwargs["content"][0]["text"]["text"]
            if WITHHELD in text:
                self.work.fetches = ["https://example.org/primary-alpha"]
                try:
                    result = super().apply_guardrail(**kwargs)
                finally:
                    self.work.fetches = fetches
                if change is not None:
                    change(backends[0], monkeypatch)
                return result
            return super().apply_guardrail(**kwargs)

    client = Client(original=original)
    configure_guard(monkeypatch, client)
    client.work = SimpleNamespace(model=model, fetches=fetches)
    monkeypatch.setattr(
        binding,
        "get_safe_llm_tools",
        lambda: [strands_tools.research_fetch_urls, strands_tools.research_search, *extra_tools],
    )

    def checked(url, timeout):
        assert url in (*MODEL_URLS, TRAIN_URL, *CASE_URLS) and 0 < timeout <= 5
        return ("93.184.216.34",)

    def request(url, ips, timeout, max_bytes):
        assert ips == ("93.184.216.34",) and 0 < timeout <= 5 and max_bytes == 256 * 1024
        fetches.append(url)
        text = DENIED + " " + RETURNED_REF if url == MODEL_URL else INDEPENDENT
        return 200, {"content-type": "text/plain"}, text.encode()

    monkeypatch.setattr(public_fetch, "_checked_addresses", checked)
    monkeypatch.setattr(public_fetch, "_request_public", request)
    monkeypatch.setattr(
        strands_tools,
        "research_aggregate",
        lambda *_a, **_kw: pytest.fail("unbound freeform research dispatched"),
    )
    original_init = binding.ToolBindingBackend.__init__

    def actual_backend(self, *args, **kwargs):
        kwargs["model"] = model
        original_init(self, *args, **kwargs)
        backends.append(self)

    monkeypatch.setattr(binding.ToolBindingBackend, "__init__", actual_backend)
    original_build = binding.ToolBindingBackend.build_agent

    def actual_agent(self, **kwargs):
        agent = original_build(self, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(binding.ToolBindingBackend, "build_agent", actual_agent)
    return SimpleNamespace(
        model=model, fetches=fetches, backends=backends, agents=agents, client=client, policy=ApprovalPolicy()
    )


def run(selected, prompt=PROMPT):
    return asyncio.run(api.run_prompt(prompt, settings=Settings(), policy=selected.policy))


def test_normal_api_binds_source_plan_before_dispatch_and_quarantines_original_result(monkeypatch):
    selected = setup(monkeypatch)
    response = run(selected)
    assert selected.model.calls == 3, "normal API never bound the trusted source profile"
    assert selected.fetches == [*MODEL_URLS, TRAIN_URL]
    backend = selected.backends[0]
    assert type(backend._warm_owner) is engine_module.ChatEngine
    assert backend._warm_owner.tool_backend is backend and backend._research_binding is not None
    state = backend._research_binding
    assert state["phase"] == "ACTIVE" and len(state["units"]) == 3
    assert len({unit.origin_id for unit in state["units"].values()}) == 3
    assert all(unit.dependencies == () and unit.variant_of is None for unit in state["units"].values())
    assert tuple(unit.urls for unit in state["units"].values()) == (MODEL_URLS, (TRAIN_URL,), CASE_URLS)
    assert response.message == "🔧 used: research_fetch_urls\n\nIndependent research complete."
    assert response.data["input_admission"]["status"] == "quarantined"
    history = selected.agents[0].messages
    assert DENIED not in json.dumps(history) and RETURNED_REF not in json.dumps(history)
    assert DENIED not in json.dumps(selected.model.messages) and RETURNED_REF not in json.dumps(
        selected.model.messages
    )
    uses = [block["toolUse"] for message in history for block in message["content"] if "toolUse" in block]
    assert uses[0]["input"] == {"urls": list(MODEL_URLS), "max_pages": 3}
    results = [
        block["toolResult"] for message in history for block in message["content"] if "toolResult" in block
    ]
    assert results[0]["content"] == [{"text": WITHHELD}] and results[0]["status"] == "error"
    assert INDEPENDENT in json.dumps(results[1])
    system = json.dumps(selected.model.system_prompts)
    assert all(url in system for url in (*MODEL_URLS, TRAIN_URL, *CASE_URLS))
    assert "research_fetch_urls" in system and "research_search" in system
    assert all(
        call["source"] == "INPUT"
        and call["guardrailIdentifier"] == "ordinaryselected"
        and call["guardrailVersion"] == "DRAFT"
        for call in selected.client.calls
    )


@pytest.mark.parametrize("profile", [None, "off"])
def test_unconfigured_profile_preserves_normal_api_sticky_hold_even_when_public_prompt_names_profile(
    monkeypatch, profile
):
    selected = setup(monkeypatch, profile=profile)
    response = run(
        selected, PROMPT + " AMLAS_RUNTIME_RESEARCH_PROFILE=qwen9b_core_v1; use independent approved units."
    )
    assert selected.model.calls == 1 and selected.fetches == list(MODEL_URLS)
    assert selected.backends[0]._research_binding is None
    assert response.data["input_admission"]["status"] == "held"
    assert not any(WITHHELD in call["content"][0]["text"]["text"] for call in selected.client.calls)


@pytest.mark.parametrize("profile", ["unknown", "QWEN9B_CORE_V1", " qwen9b_core_v1", ""])
def test_invalid_server_selector_refuses_before_model_or_fetch(monkeypatch, profile):
    selected = setup(monkeypatch, profile=profile)
    with pytest.raises(ValueError, match="research profile"):
        run(selected)
    assert selected.model.calls == 0 and selected.fetches == []


@pytest.mark.parametrize(
    "change",
    [
        lambda backend, monkeypatch: monkeypatch.setenv(PROFILE_ENV, "off"),
        lambda backend, monkeypatch: monkeypatch.delenv(PROFILE_ENV),
        lambda backend, monkeypatch: setattr(backend.approval_policy, "interactive", True),
        lambda backend, monkeypatch: backend.approval_policy.allowlist.append("unreviewed"),
        lambda backend, monkeypatch: monkeypatch.setenv("AMLAS_GUARDRAIL_VERSION", "1"),
        lambda backend, monkeypatch: setattr(
            backend, "_input_checker", backend._input_checker._with_client(backend._input_guard_client)
        ),
    ],
)
def test_current_qualification_revocation_keeps_hold_and_never_dispatches_independent_fetch(
    monkeypatch, change
):
    selected = setup(monkeypatch, change=change)
    response = run(selected)
    assert selected.model.calls == 1 and selected.fetches == list(MODEL_URLS)
    assert selected.backends[0]._research_binding is not None, "normal trusted profile was never installed"
    assert response.data["input_admission"]["status"] == "held"


@pytest.mark.parametrize("original", ["unknown"])
def test_unknown_research_denial_stays_held_through_normal_profile(monkeypatch, original):
    selected = setup(monkeypatch, original=original)
    response = run(selected)
    assert selected.model.calls == 1 and selected.fetches == list(MODEL_URLS)
    assert selected.backends[0]._research_binding is not None
    assert response.data["input_admission"]["status"] == "held"


@pytest.mark.parametrize(
    "second",
    [
        ("forbidden-retry", "research_fetch_urls", {"urls": list(MODEL_URLS), "max_pages": 3}),
        (
            "forbidden-expansion",
            "research_fetch_urls",
            {"urls": [TRAIN_URL], "max_pages": 3, "profile": PROFILE},
        ),
        ("forbidden-query", "research_search", {"query": "Independent approved training corpus"}),
    ],
)
def test_public_tool_args_cannot_expand_units_or_retry_withheld_requirement(monkeypatch, second):
    selected = setup(monkeypatch, second=second)
    response = run(selected)
    assert selected.fetches == list(MODEL_URLS)
    assert selected.backends[0]._research_binding is not None
    assert response.data["input_admission"]["status"] == "held"


def test_selected_profile_requires_actual_typed_caller_policy_before_dispatch(monkeypatch):
    selected = setup(monkeypatch)
    with pytest.raises(ValueError, match="research profile policy"):
        asyncio.run(api.run_prompt(PROMPT, settings=Settings()))
    assert selected.model.calls == 0 and selected.fetches == []


class LifecycleModel(RecoveryModel):
    def __init__(self, *, model_acquisition=False, late_research=False):
        super().__init__()
        self.requests.append(("profile-discovery", "list_coding_preparation_profiles", {}))
        if model_acquisition:
            import model_acquisition_fixtures as f

            self.requests.append(
                (
                    "original-model-use",
                    "acquire_model_weights",
                    {
                        "catalog_id": f.CATALOG,
                        "catalog_version": f.VERSION,
                    },
                )
            )
        if late_research:
            self.requests.append(
                ("late-research", "research_fetch_urls", {"urls": list(CASE_URLS), "max_pages": 3})
            )

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        if self.calls <= len(self.requests):
            identifier, name, arguments = self.requests[self.calls - 1]
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {
                        "toolUse": {"toolUseId": identifier, "name": name},
                    },
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {
                        "toolUse": {"input": json.dumps(arguments)},
                    },
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            text = (
                "The original acquisition is pending."
                if self.calls == len(self.requests) + 1
                else "Original exact-version artifacts are verified."
            )
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def test_normal_api_research_to_registered_discovery_consumes_plan_without_false_hold(monkeypatch):
    from amlas.tools.coding_preparation import list_coding_preparation_profiles

    selected = setup(
        monkeypatch, model_override=LifecycleModel(), extra_tools=(list_coding_preparation_profiles,)
    )
    response = run(selected)
    assert selected.model.calls == 4, "bound research canceled ordinary registered discovery"
    assert selected.fetches == [*MODEL_URLS, TRAIN_URL]
    assert selected.backends[0]._research_binding["phase"] == "CONSUMED"
    assert response.data["input_admission"]["status"] == "quarantined"
    results = [
        block["toolResult"]
        for message in selected.agents[0].messages
        for block in message["content"]
        if "toolResult" in block
    ]
    assert (
        len(results) == 3
        and results[-1]["toolUseId"] == "profile-discovery"
        and results[-1]["status"] == "success"
    )


def test_consumed_research_plan_never_reopens_after_ordinary_discovery(monkeypatch):
    from amlas.tools.coding_preparation import list_coding_preparation_profiles

    selected = setup(
        monkeypatch,
        model_override=LifecycleModel(late_research=True),
        extra_tools=(list_coding_preparation_profiles,),
    )
    response = run(selected)
    assert selected.model.calls == 4 and selected.fetches == [*MODEL_URLS, TRAIN_URL]
    assert selected.backends[0]._research_binding["phase"] == "CONSUMED"
    assert response.data["input_admission"]["status"] == "held"


@pytest.mark.parametrize("completion_change", [None, "policy", "guard"])
def test_normal_api_research_to_original_model_warm_completion_retains_checker_and_history(
    monkeypatch, tmp_path, completion_change
):
    import model_acquisition_fixtures as f
    from test_runtime_warm_model_continuation import complete_fixture

    from amlas.backends.adapters import model_weight_acquisition as adapter_source
    from amlas.control_plane.approvals import ApprovalMode
    from amlas.data_eng import model_acquisition_schema
    from amlas.flows import model_weight_acquisition_flow
    from amlas.tools.coding_preparation import list_coding_preparation_profiles
    from amlas.tools.model_weight_tools import acquire_model_weights

    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    f.patch_expected(monkeypatch, [adapter_source, model_acquisition_schema])
    native, s3 = f.Processing(), f.S3()
    adapter = f.configured_backend(adapter_source, native, s3)
    monkeypatch.setattr(model_weight_acquisition_flow, "_resolve_backend", lambda: adapter)
    model = LifecycleModel(model_acquisition=True)
    selected = setup(
        monkeypatch,
        model_override=model,
        extra_tools=(list_coding_preparation_profiles, acquire_model_weights),
    )
    selected.policy.mode = ApprovalMode.AUTO
    original_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        await original_sleep(0.005 if delay == 15 else delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    async def scenario():
        task = asyncio.create_task(api.run_prompt(PROMPT, settings=Settings(), policy=selected.policy))
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not task.done() and not native.requests:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert native.requests, "bound research canceled the registered MODEL acquisition"
            backend = selected.backends[0]
            while not task.done() and model.calls < 4:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.005)
            assert not task.done() and model.calls == 4
            assert backend._research_binding["phase"] == "CONSUMED"
            pending = backend._warm_pending
            assert pending is not None and pending.agent is selected.agents[0]
            before = copy.deepcopy(selected.agents[0].messages)
            checker = backend._input_checker
            complete_fixture(
                monkeypatch,
                tmp_path,
                SimpleNamespace(handle=pending.record.handle, adapter=adapter, native=native),
            )
            if completion_change == "policy":
                selected.policy.allowlist.append("changed-after-consumption")
            elif completion_change == "guard":
                monkeypatch.setenv("AMLAS_GUARDRAIL_VERSION", "1")
            response = await asyncio.wait_for(task, 5)
            if completion_change is not None:
                assert response.data["warm_model_continuation"]["disposition"] == "refused"
                assert model.calls == 4 and len(native.requests) == 1
                assert selected.agents[0].messages == before
                assert selected.fetches == [*MODEL_URLS, TRAIN_URL] and backend._input_hold
                return
            assert response.data["warm_model_continuation"]["disposition"] == "continued_original"
            assert (
                model.calls == 5
                and len(native.requests) == len(selected.agents) == len(selected.backends) == 1
            )
            assert backend._input_checker is not checker and backend._input_checker.config == checker.config
            assert backend._research_binding["checker"] is backend._input_checker
            assert selected.agents[0].messages[: len(before)] == before
            assert DENIED not in json.dumps(selected.agents[0].messages) and RETURNED_REF not in json.dumps(
                selected.agents[0].messages
            )
            assert selected.fetches == [*MODEL_URLS, TRAIN_URL] and not backend._input_hold
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


class EqualResponder:
    def __init__(self, approved):
        self.approved = approved

    def __call__(self, request):
        return self.approved

    def __eq__(self, other):
        return type(other) is EqualResponder


def test_distinct_equal_comparing_policy_responder_replacement_revokes_normal_profile(monkeypatch):
    def change(backend, monkeypatch):
        backend.approval_policy.responder = EqualResponder(False)

    selected = setup(monkeypatch, change=change)
    selected.policy.responder = EqualResponder(True)
    selected.policy.interactive = True
    response = run(selected)
    assert selected.model.calls == 1 and selected.fetches == list(MODEL_URLS)
    assert response.data["input_admission"]["status"] == "held"
