"""Private preauthorized research recovery over actual SDK/owner/tool paths.

All provider/native transports are inert. Unit DATA comes from trusted test code
before execution, never from model JSON. This private binder route does not prove
normal prompt parsing, semantic independence, native guard service or ML autonomy.
Original history/request arguments remain unchanged; only denied RESULT data is
quarantined. No original native runtime17 request is replayed.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest
from botocore.session import get_session
from botocore.validate import validate_parameters
from strands.models import Model
from test_input_admission_wiring import configure_guard

from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalPolicy
from amlas.tools.research import public_fetch, strands_tools
from amlas.tui.chat import engine as engine_module
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.research_input_recovery import _ResearchPlan, _ResearchUnit

URL_A = "https://example.org/primary-alpha"
URL_B = "https://example.org/primary-beta"
URL_C = "https://example.org/primary-child"
URL_D = "https://example.org/primary-variant"
RETURNED_REF = "https://example.org/withheld-result-reference"
DENIED = "Ordinary denied source information."
INDEPENDENT = "Ordinary independently approved source information."
WITHHELD = (
    "Research source content was withheld by input admission. Do not retry this request or source. "
    "Continue only with independent approved research."
)


def plan():
    return _ResearchPlan(
        (
            _ResearchUnit("unit-alpha", "origin-alpha", (URL_A,)),
            _ResearchUnit("unit-beta", "origin-beta", (URL_B,)),
            _ResearchUnit("unit-child", "origin-child", (URL_C,), dependencies=("unit-alpha",)),
            _ResearchUnit("unit-variant", "origin-variant", (URL_D,), variant_of="unit-alpha"),
        )
    )


class RecoveryModel(Model):
    def __init__(self, second=None):
        self.requests = [
            ("original-readonly-use", "research_fetch_urls", {"urls": [URL_A], "max_pages": 3}),
            second or ("independent-readonly-use", "research_fetch_urls", {"urls": [URL_B], "max_pages": 3}),
        ]
        self.calls, self.messages = 0, []

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "offline-independent-research-fixture"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("actual normal SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        if self.calls <= 2:
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
            assert self.calls == 3
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Independent research complete."},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}


class GuardClient:
    def __init__(self, fixed="allow", original="blocked"):
        self.fixed, self.original = fixed, original
        self.calls = []
        self.work = None
        self.operation = get_session().get_service_model("bedrock-runtime").operation_model("ApplyGuardrail")

    def close(self):
        pass

    def apply_guardrail(self, **kwargs):
        validate_parameters(kwargs, self.operation.input_shape)
        self.calls.append(copy.deepcopy(kwargs))
        text = kwargs["content"][0]["text"]["text"]
        if WITHHELD in text:
            assert self.work.model.calls == 1 and self.work.fetches == [URL_A], (
                "pending recovery dispatched work"
            )
            outcome = self.fixed
        elif DENIED in text:
            outcome = self.original
        else:
            outcome = "allow"
        if outcome == "unknown":
            return {}
        if outcome == "masked":
            response = {
                "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
                "action": "GUARDRAIL_INTERVENED",
                "outputs": [{"text": text.replace("content", "{NAME}")}],
                "assessments": [
                    {
                        "sensitiveInformationPolicy": {
                            "piiEntities": [
                                {
                                    "type": "NAME",
                                    "action": "ANONYMIZED",
                                    "detected": True,
                                    "match": "content",
                                }
                            ],
                            "regexes": [],
                        }
                    }
                ],
            }
        else:
            blocked = outcome == "blocked"
            response = {
                "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
                "action": "GUARDRAIL_INTERVENED" if blocked else "NONE",
                "outputs": [],
                "assessments": [
                    {
                        "contentPolicy": {
                            "filters": [
                                {
                                    "type": "PROMPT_ATTACK",
                                    "confidence": "LOW" if blocked else "NONE",
                                    "filterStrength": "HIGH",
                                    "action": "BLOCKED" if blocked else "NONE",
                                    "detected": blocked,
                                }
                            ]
                        }
                    }
                ],
            }
        # ResponseMetadata is the SDK transport envelope, not a modeled body member.
        usage = self.operation.output_shape.members.get("usage")
        if usage is not None:
            response["usage"] = {
                name: 0 for name, member in usage.members.items() if member.type_name in {"integer", "long"}
            }
        validate_parameters(
            {key: value for key, value in response.items() if key != "ResponseMetadata"},
            self.operation.output_shape,
        )
        return response


def work(monkeypatch, *, fixed="allow", original="blocked", second=None):
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *_a, **_kw: "")
    client, model, fetches = GuardClient(fixed, original), RecoveryModel(second), []
    configure_guard(monkeypatch, client)
    monkeypatch.setattr(
        binding,
        "get_safe_llm_tools",
        lambda: [strands_tools.research_fetch_urls, strands_tools.research_search],
    )

    def checked(url, timeout):
        assert url.startswith("https://example.org/") and 0 < timeout <= 5
        return ("93.184.216.34",)

    def request(url, ips, timeout, max_bytes):
        assert ips == ("93.184.216.34",) and 0 < timeout <= 5 and max_bytes == 256 * 1024
        fetches.append(url)
        text = DENIED + " " + RETURNED_REF if url == URL_A else INDEPENDENT
        return 200, {"content-type": "text/plain"}, text.encode()

    monkeypatch.setattr(public_fetch, "_checked_addresses", checked)
    monkeypatch.setattr(public_fetch, "_request_public", request)
    monkeypatch.setattr(
        strands_tools,
        "research_aggregate",
        lambda *_a, **_kw: pytest.fail("freeform research dispatched after block"),
    )
    original_init = binding.ToolBindingBackend.__init__

    def actual_backend(self, *args, **kwargs):
        kwargs["model"] = model
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(binding.ToolBindingBackend, "__init__", actual_backend)
    policy = ApprovalPolicy()
    owner = engine_module.ChatEngine(Settings(), approval_policy=policy)
    backend, agents, callbacks, qualification = owner.tool_backend, [], [], []
    assert type(backend) is binding.ToolBindingBackend and backend._warm_owner is owner
    assert owner.tool_backend is backend and model.calls == 0 and not fetches
    build = backend.build_agent

    def capture(**kwargs):
        agent = build(**kwargs)

        def callback(**_event):
            return None

        agent.callback_handler = callback
        agents.append(agent)
        callbacks.append(callback)
        return agent

    monkeypatch.setattr(backend, "build_agent", capture)

    def qualify():
        assert backend._warm_owner is owner and owner.tool_backend is backend
        assert backend.approval_policy is policy
        qualification.append("current")

    selected = SimpleNamespace(
        client=client,
        model=model,
        fetches=fetches,
        owner=owner,
        backend=backend,
        agents=agents,
        callbacks=callbacks,
        qualify=qualify,
        qualification=qualification,
    )
    client.work = selected
    return selected


def bind(selected, manifest=None, *, owner=None):
    method = getattr(selected.backend, "_bind_research_units", None)
    assert callable(method), "missing-symbol: private source-owned preexecution research unit binder"
    return method(
        plan() if manifest is None else manifest,
        owner=selected.owner if owner is None else owner,
        current_qualification=selected.qualify,
    )


def run(selected):
    return selected.backend.chat_with_tools("Check the two independently authorized primary source units.")


def test_actual_sdk_preserves_blocked_receipt_and_runs_one_preapproved_independent_unit(monkeypatch):
    selected = work(monkeypatch)
    bind(selected)
    turn = run(selected)
    assert selected.model.calls == 3 and selected.fetches == [URL_A, URL_B]
    assert len(selected.agents) == 1 and selected.agents[0].callback_handler is selected.callbacks[0]
    assert selected.qualification and not selected.backend._input_hold
    messages = selected.agents[0].messages
    assert DENIED not in json.dumps(messages) and RETURNED_REF not in json.dumps(messages)
    assert DENIED not in json.dumps(selected.model.messages) and RETURNED_REF not in json.dumps(
        selected.model.messages
    )
    uses = [b["toolUse"] for m in messages for b in m["content"] if "toolUse" in b]
    assert uses[0]["input"] == {"urls": [URL_A], "max_pages": 3}, "original request history was rewritten"
    results = [b["toolResult"] for m in messages for b in m["content"] if "toolResult" in b]
    assert len(results) == 2 and results[0]["status"] == "error"
    assert results[0]["content"] == [{"text": WITHHELD}]
    assert INDEPENDENT in json.dumps(results[1])
    records = turn.input_admission["records"]
    source = [r for r in records if r["boundary"] == "TOOL_RESULT"]
    assert [r["outcome"] for r in source] == ["BLOCKED", "ALLOW", "ALLOW"]
    assert source[0]["reason"] == "NATIVE_BLOCKED" and source[0]["check_id"] != source[1]["check_id"]
    assert all(r["data_only"] is True and r["native_admission"] is False for r in source)
    assert all(
        c["source"] == "INPUT"
        and c["guardrailIdentifier"] == "ordinaryselected"
        and c["guardrailVersion"] == "DRAFT"
        for c in selected.client.calls
    )
    blocked_call = next(
        i for i, c in enumerate(selected.client.calls) if DENIED in c["content"][0]["text"]["text"]
    )
    assert WITHHELD in selected.client.calls[blocked_call + 1]["content"][0]["text"]["text"]


def test_without_preexecution_plan_original_research_hold_stays_sticky(monkeypatch):
    selected = work(monkeypatch)
    turn = run(selected)
    assert selected.model.calls == 1 and selected.fetches == [URL_A]
    assert turn.input_admission["status"] == "held"
    assert not any(WITHHELD in c["content"][0]["text"]["text"] for c in selected.client.calls)


@pytest.mark.parametrize("fixed", ["blocked", "unknown", "masked"])
def test_fixed_withheld_error_requires_unchanged_strict_allow_or_remains_held(monkeypatch, fixed):
    selected = work(monkeypatch, fixed=fixed)
    bind(selected)
    turn = run(selected)
    assert selected.model.calls == 1 and selected.fetches == [URL_A]
    assert turn.input_admission["status"] == "held"
    if fixed == "masked":
        assert turn.input_admission["records"][-1]["outcome"] == "ANONYMIZED"
    assert len([c for c in selected.client.calls if WITHHELD in c["content"][0]["text"]["text"]]) == 1


@pytest.mark.parametrize(
    "second",
    [
        ("repeat", "research_fetch_urls", {"urls": [URL_A], "max_pages": 3}),
        ("dependent", "research_fetch_urls", {"urls": [URL_C], "max_pages": 3}),
        ("variant", "research_fetch_urls", {"urls": [URL_D], "max_pages": 3}),
        ("dynamic", "research_fetch_urls", {"urls": ["https://example.org/dynamic-source"], "max_pages": 3}),
        (
            "freeform",
            "research_search",
            {"query": "An independently phrased dynamic query", "max_results": 1},
        ),
    ],
)
def test_recovery_never_dispatches_replay_dependent_variant_dynamic_or_freeform(monkeypatch, second):
    selected = work(monkeypatch, second=second)
    bind(selected)
    turn = run(selected)
    assert selected.model.calls == 2 and selected.fetches == [URL_A]
    assert turn.input_admission["status"] == "held"


def test_unknown_original_assessment_has_no_recovery_or_failed_assessment_retry(monkeypatch):
    selected = work(monkeypatch, original="unknown")
    bind(selected)
    turn = run(selected)
    assert selected.model.calls == 1 and selected.fetches == [URL_A]
    assert turn.input_admission["status"] == "held"
    assert len([c for c in selected.client.calls if DENIED in c["content"][0]["text"]["text"]]) == 1
    assert not any(WITHHELD in c["content"][0]["text"]["text"] for c in selected.client.calls)


@pytest.mark.parametrize("bad", ["public", "duplicate", "unknown-dependency", "cycle", "too-many"])
def test_private_binder_refuses_untyped_or_unclosed_graph_before_any_execution(monkeypatch, bad):
    selected = work(monkeypatch)
    manifest = plan()
    if bad == "public":
        manifest = dataclasses.asdict(manifest)
    elif bad == "duplicate":
        manifest = _ResearchPlan((manifest.units[0], manifest.units[0]))
    elif bad == "unknown-dependency":
        manifest = _ResearchPlan((dataclasses.replace(manifest.units[0], dependencies=("missing-unit",)),))
    elif bad == "cycle":
        manifest = _ResearchPlan(
            (
                dataclasses.replace(manifest.units[0], dependencies=("unit-beta",)),
                dataclasses.replace(manifest.units[1], dependencies=("unit-alpha",)),
            )
        )
    else:
        manifest = _ResearchPlan(
            tuple(
                _ResearchUnit(
                    "unit-" + str(i), "origin-" + str(i), ("https://example.org/primary-" + str(i),)
                )
                for i in range(9)
            )
        )
    with pytest.raises(ValueError):
        bind(selected, manifest)
    assert selected.model.calls == 0 and selected.fetches == []


def test_private_plan_foreign_owner_refuses_before_model_or_research(monkeypatch):
    selected = work(monkeypatch)
    with pytest.raises(ValueError):
        bind(selected, owner=object())
    assert selected.model.calls == 0 and selected.fetches == []


def test_bound_plan_guard_configuration_drift_stays_held_before_first_model(monkeypatch):
    selected = work(monkeypatch)
    bind(selected)
    monkeypatch.setenv("AMLAS_GUARDRAIL_VERSION", "1")
    turn = run(selected)
    assert selected.model.calls == 0 and selected.fetches == []
    assert turn.input_admission["status"] == "held"


def test_consumed_original_plan_cannot_retarget_itself_to_a_later_turn(monkeypatch):
    selected = work(monkeypatch)
    bind(selected)
    run(selected)
    assert selected.model.calls == 3 and selected.fetches == [URL_A, URL_B]
    turn = run(selected)
    assert selected.model.calls == 3 and selected.fetches == [URL_A, URL_B]
    assert turn.input_admission["status"] == "held"


def test_actual_planning_store_receives_only_independently_admitted_source_facts(monkeypatch):
    from test_campaign_planning_publication import planning_fixture

    from amlas.control_plane.campaign_runtime import activate_campaign_entry, activate_campaign_runtime

    selected = work(monkeypatch)
    _module, _correlation, original_entry, store, protected = planning_fixture(monkeypatch)
    selected.backend._campaign_planning_store = store
    with activate_campaign_runtime(original_entry._factory), activate_campaign_entry(original_entry):
        bind(selected)
        turn = run(selected)
    assert selected.model.calls == 3 and selected.fetches == [URL_A, URL_B]
    assert turn.input_admission["status"] != "held"
    assert len(store._research) == 1 and store._research[0]["url"] == URL_B
    assert store._research[0]["tool"] == "research_fetch_urls"
    assert store._research[0]["untrusted_data"] is True
    assert not any(row["url"] in {None, URL_A, RETURNED_REF} for row in store._research)
    assert protected.writes == 0
