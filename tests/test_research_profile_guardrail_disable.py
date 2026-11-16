"""Explicit guard disable preserves bounded normal research without guard claims.

Normal API/ChatEngine and actual installed SDK; neutral scripted model, source
bodies and inert provider ports. These regressions establish source behavior,
never native guard admission, source correctness, licensing or ML completion.
"""

from __future__ import annotations

import json

import pytest
from test_normal_research_profile import (
    CASE_URLS,
    MODEL_URLS,
    PROFILE,
    PROFILE_ENV,
    TRAIN_URL,
    run,
    setup,
)
from test_research_exhaustion import GAP, ExhaustionModel
from test_research_input_recovery import DENIED, INDEPENDENT, RETURNED_REF

COMPLETE = "Fixed research source bytes were retrieved."
REFUSAL = "explicitly disabled guardrails refused the normal finite research profile"


class ThreeUnitModel(ExhaustionModel):
    """Reuse the existing actual-SDK stream with an ordinary terminal message."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if delta.get("text") == GAP:
                delta["text"] = COMPLETE
            yield event


def disabled_setup(monkeypatch, *, repeat=False, extra_tools=()):
    selected = setup(monkeypatch, model_override=ThreeUnitModel(repeat=repeat), extra_tools=extra_tools)
    # setup selects a provisioned guard and creates the engine only inside run.
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    return selected


def run_disabled(selected):
    try:
        return run(selected)
    except ValueError:
        pytest.fail(REFUSAL, pytrace=False)


def no_guard_claims(selected, response):
    assert selected.client.calls == [], "disabled research invoked ApplyGuardrail"
    assert "guardrail_outcome" not in response.data
    evidence = response.data.get("input_admission")
    if evidence is not None:
        assert evidence["status"] == "held"
        assert evidence["records"] == [], "disabled research fabricated native INPUT receipts"
        assert evidence["native_admission"] is False and evidence["quality_success"] is False
    assert selected.backends[0]._input_receipts == []


def settled_disabled_research(backend):
    state = backend._research_binding
    units = {"model-card", "training-card", "benchmark-card"}
    assert set(state["units"]) == units
    assert state["attempted"] == state["forwarded"] == units
    assert len(state["tool_uses"]) == 3 and set(state["tool_uses"].values()) == units
    assert state["results"] == set(state["tool_uses"])
    assert state["pending"] is False
    assert state["admitted"] == set(), "disabled source bytes became strict native ALLOW evidence"
    assert state["blocked"] == set() and state["withheld"] == set()
    assert backend._input_receipts == []
    return state


def test_explicit_disable_normal_api_completes_all_fixed_research_units_without_guard_claims(monkeypatch):
    selected = disabled_setup(monkeypatch)
    response = run_disabled(selected)
    assert selected.model.calls == 4
    assert selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.message == "🔧 used: research_fetch_urls\n\n" + COMPLETE
    assert response.data.get("input_admission") is None
    no_guard_claims(selected, response)
    assert settled_disabled_research(selected.backends[0])["phase"] == "ACTIVE"
    history = selected.agents[0].messages
    uses = [block["toolUse"] for message in history for block in message["content"] if "toolUse" in block]
    results = [
        block["toolResult"] for message in history for block in message["content"] if "toolResult" in block
    ]
    assert [use["input"] for use in uses] == [
        {"urls": list(MODEL_URLS), "max_pages": 3},
        {"urls": [TRAIN_URL], "max_pages": 3},
        {"urls": list(CASE_URLS), "max_pages": 3},
    ]
    assert len(results) == 3 and all(result["status"] == "success" for result in results)
    assert [result["toolUseId"] for result in results] == [use["toolUseId"] for use in uses]
    assert all(value in json.dumps(history) for value in (DENIED, RETURNED_REF, INDEPENDENT))
    system = json.dumps(selected.model.system_prompts)
    assert all(url in system for url in (*MODEL_URLS, TRAIN_URL, *CASE_URLS))


@pytest.mark.parametrize(
    "tool_request",
    [
        ("expanded-batch", "research_fetch_urls", {"urls": [*MODEL_URLS, TRAIN_URL], "max_pages": 3}),
        ("partial-batch", "research_fetch_urls", {"urls": [MODEL_URLS[0]], "max_pages": 3}),
        ("reordered-batch", "research_fetch_urls", {"urls": list(reversed(MODEL_URLS)), "max_pages": 3}),
        ("changed-page-limit", "research_fetch_urls", {"urls": list(MODEL_URLS), "max_pages": 2}),
        (
            "public-profile-field",
            "research_fetch_urls",
            {"urls": list(MODEL_URLS), "max_pages": 3, "profile": PROFILE},
        ),
        ("unbounded-search", "research_search", {"query": "Ordinary coding model information"}),
    ],
    ids=["expanded", "partial", "reordered", "page-limit", "public-profile", "search"],
)
def test_disabled_guard_does_not_expand_exact_research_scope(monkeypatch, tool_request):
    selected = disabled_setup(monkeypatch)
    selected.model.requests[0] = tool_request
    response = run_disabled(selected)
    assert selected.model.calls == 1 and selected.fetches == []
    assert response.data["input_admission"]["status"] == "held"
    no_guard_claims(selected, response)


@pytest.mark.parametrize("repeat_at", ["first-unit", "last-unit"])
def test_disabled_guard_never_repeats_an_attempted_research_unit(monkeypatch, repeat_at):
    selected = disabled_setup(monkeypatch, repeat=repeat_at == "last-unit")
    if repeat_at == "first-unit":
        selected.model.requests[1] = (
            "repeated-model-card",
            "research_fetch_urls",
            {"urls": list(MODEL_URLS), "max_pages": 3},
        )
    response = run_disabled(selected)
    expected = list(MODEL_URLS) if repeat_at == "first-unit" else [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert selected.fetches == expected
    assert selected.model.calls == (2 if repeat_at == "first-unit" else 4)
    assert response.data["input_admission"]["status"] == "held"
    no_guard_claims(selected, response)


@pytest.mark.parametrize("boundary", [0, 1], ids=["before-first-fetch", "before-next-fetch"])
@pytest.mark.parametrize(
    "change",
    [
        lambda selected, monkeypatch: monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "true"),
        lambda selected, monkeypatch: monkeypatch.setenv(PROFILE_ENV, "off"),
        lambda selected, monkeypatch: selected.policy.allowlist.append("changed-current-policy"),
    ],
    ids=["guard-enabled", "profile-off", "policy-changed"],
)
def test_disabled_research_current_mode_profile_and_policy_drift_hold_before_fetch(
    monkeypatch, boundary, change
):
    selected = disabled_setup(monkeypatch)
    stream = selected.model.stream
    changes = []

    async def changed_stream(messages, tool_specs=None, system_prompt=None, **kwargs):
        if selected.model.calls == boundary:
            assert selected.fetches == ([] if boundary == 0 else list(MODEL_URLS))
            change(selected, monkeypatch)
            changes.append(boundary)
        async for event in stream(messages, tool_specs, system_prompt, **kwargs):
            yield event

    monkeypatch.setattr(selected.model, "stream", changed_stream)
    response = run_disabled(selected)
    assert changes == [boundary]
    assert selected.model.calls == boundary + 1
    assert selected.fetches == ([] if boundary == 0 else list(MODEL_URLS))
    assert response.data["input_admission"]["status"] == "held"
    no_guard_claims(selected, response)


def test_enabled_unprovisioned_research_profile_still_refuses_before_dispatch(monkeypatch):
    selected = setup(monkeypatch, model_override=ThreeUnitModel())
    monkeypatch.delenv("AMLAS_GUARDRAIL_ID")
    with pytest.raises(ValueError, match="research profile INPUT checker required"):
        run(selected)
    assert selected.model.calls == 0 and selected.fetches == []
    assert selected.client.calls == []


def test_disabled_all_fixed_research_units_consume_into_actual_registered_discovery(monkeypatch):
    from amlas.tools.coding_preparation import list_coding_preparation_profiles

    selected = disabled_setup(monkeypatch, extra_tools=(list_coding_preparation_profiles,))
    selected.model.requests.append(("disabled-profile-discovery", "list_coding_preparation_profiles", {}))
    response = run_disabled(selected)
    assert selected.model.calls == 5
    assert selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.data.get("input_admission") is None
    no_guard_claims(selected, response)
    state = settled_disabled_research(selected.backends[0])
    assert state["phase"] == "CONSUMED" and state["original_completion_ready"] is True
    results = [
        block["toolResult"]
        for message in selected.agents[0].messages
        for block in message["content"]
        if "toolResult" in block
    ]
    assert len(results) == 4 and all(result["status"] == "success" for result in results)
    assert results[-1]["toolUseId"] == "disabled-profile-discovery"
    assert len(selected.backends[0]._model_acquisition_store._records) == 0


def test_actual_cold_completion_with_disabled_guard_restores_settled_research_without_refetch(
    monkeypatch, tmp_path
):
    import test_normal_cold_research_phase as cold_research
    import test_sdk_pending_handoff as profile_fixture

    from amlas.tools.research import public_fetch

    original_guard = profile_fixture.configure_guard
    original_setup = cold_research.setup
    original_claim = cold_research.ChatEngine.claim_normal_completion
    clients, fetch_batches, restored = [], [], []

    def configure_disabled_guard(patch, client):
        original_guard(patch, client)
        patch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
        clients.append(client)

    def observed_setup(patch, directory, **kwargs):
        selected = original_setup(patch, directory, **kwargs)
        fetched = []
        fetch_batches.append(fetched)
        original_fetch = public_fetch._request_public

        def fetch(url, ips, timeout, max_bytes):
            fetched.append(url)
            return original_fetch(url, ips, timeout, max_bytes)

        patch.setattr(public_fetch, "_request_public", fetch)
        return selected

    def observed_claim(engine, *args, **kwargs):
        result = original_claim(engine, *args, **kwargs)
        restored.append(engine.tool_backend)
        return result

    # Both fresh setup MonkeyPatch objects call this imported helper. The outer
    # patch survives initial cleanup; no TASK, snapshot or qualification is seeded.
    monkeypatch.setattr(profile_fixture, "configure_guard", configure_disabled_guard)
    monkeypatch.setattr(cold_research, "setup", observed_setup)
    monkeypatch.setattr(cold_research.ChatEngine, "claim_normal_completion", observed_claim)
    cold_research.test_actual_cold_completion_restores_consumed_research_without_refetch(
        monkeypatch, tmp_path
    )
    assert len(clients) == 2 and all(client.calls == [] for client in clients)
    assert fetch_batches == [[*MODEL_URLS, TRAIN_URL, *CASE_URLS], []]
    assert len(restored) == 1
    state = settled_disabled_research(restored[0])
    assert state["phase"] == "CONSUMED" and state["original_completion_ready"] is True


def test_disabled_consumed_research_completion_keeps_absent_guard_transport(monkeypatch):
    from amlas.tools.coding_preparation import list_coding_preparation_profiles

    selected = disabled_setup(monkeypatch, extra_tools=(list_coding_preparation_profiles,))
    selected.model.requests.append(("disabled-completion-discovery", "list_coding_preparation_profiles", {}))
    response = run_disabled(selected)
    backend = selected.backends[0]
    state = settled_disabled_research(backend)
    assert state["phase"] == "CONSUMED" and state["original_completion_ready"] is True
    assert (
        backend._input_checker is backend._input_guard_client is backend._input_guard_client_factory is None
    )
    graph = backend._warm_graph()
    try:
        refreshed = backend._refresh_input_checker(None, state)
    except AttributeError:
        pytest.fail("disabled completion retired an absent guard transport", pytrace=False)
    assert refreshed is None and state["checker"] is None
    assert backend._warm_graph() == graph and backend._input_hold is False
    assert (
        backend._input_checker is backend._input_guard_client is backend._input_guard_client_factory is None
    )
    no_guard_claims(selected, response)
    settled_disabled_research(backend)
    assert selected.model.calls == 5 and selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]


def test_enabled_missing_checker_transport_refresh_still_refuses(monkeypatch):
    from amlas.tools.coding_preparation import list_coding_preparation_profiles
    from amlas.tui.chat.tool_binding_backend import _InputAdmissionHeldError

    selected = disabled_setup(monkeypatch, extra_tools=(list_coding_preparation_profiles,))
    selected.model.requests.append(("enabled-refusal-discovery", "list_coding_preparation_profiles", {}))
    response = run_disabled(selected)
    backend = selected.backends[0]
    state = settled_disabled_research(backend)
    # Enabled checks with no selected checker never retire, create or fake a guard transport.
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "true")
    for research in (state, None):
        with pytest.raises(_InputAdmissionHeldError):
            backend._refresh_input_checker(None, research)
    assert state["checker"] is None
    assert (
        backend._input_checker is backend._input_guard_client is backend._input_guard_client_factory is None
    )
    no_guard_claims(selected, response)
