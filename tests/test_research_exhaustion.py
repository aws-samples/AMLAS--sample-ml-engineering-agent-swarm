"""Last-unit quarantine uses only admitted independent evidence, never a grant.

Normal API and actual SDK with neutral source bodies and inert guard/transports.
The rejected runtime20 body is never used. No direct private binder invocation.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from botocore.session import get_session
from botocore.validate import validate_parameters
from test_input_admission_wiring import ClientFixture, configure_guard
from test_normal_research_profile import CASE_URLS, MODEL_URLS, TRAIN_URL, run, setup
from test_research_input_recovery import DENIED, RETURNED_REF, WITHHELD, RecoveryModel

from amlas.tools.research import public_fetch
from amlas.tui.chat import research_runtime_profile
from amlas.tui.chat.research_input_recovery import _ResearchPlan

FIRST = "Ordinary model information."
SECOND = "Ordinary corpus information."
GAP = (
    "Benchmark research was withheld. Its licence and research are not established; no training was started."
)


class ExhaustionModel(RecoveryModel):
    def __init__(self, *, repeat=False):
        super().__init__()
        self.requests.append(
            ("benchmark-readonly-use", "research_fetch_urls", {"urls": list(CASE_URLS), "max_pages": 3})
        )
        if repeat:
            self.requests.append(
                (
                    "benchmark-forbidden-retry",
                    "research_fetch_urls",
                    {"urls": list(CASE_URLS), "max_pages": 3},
                )
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
                    "start": {"toolUse": {"toolUseId": identifier, "name": name}},
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": json.dumps(arguments)}},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": GAP}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def scenario(monkeypatch, *, fixed="allow", earlier="allow", drift=None, repeat=False, first=None):
    selected = setup(monkeypatch, model_override=ExhaustionModel(repeat=repeat))
    operation = get_session().get_service_model("bedrock-runtime").operation_model("ApplyGuardrail")

    class Client(ClientFixture):
        def apply_guardrail(self, **kwargs):
            validate_parameters(kwargs, operation.input_shape)
            text = kwargs["content"][0]["text"]["text"]
            notice = WITHHELD in text
            outcome = (
                fixed
                if notice
                else (
                    "blocked"
                    if DENIED in text
                    else (
                        (earlier if first is None else first)
                        if FIRST in text
                        else (earlier if SECOND in text else "allow")
                    )
                )
            )
            self.deny = lambda _text: outcome == "blocked"
            response = super().apply_guardrail(**kwargs)
            if outcome == "unknown":
                return {}
            if outcome == "masked":
                response = {
                    "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
                    "action": "GUARDRAIL_INTERVENED",
                    "outputs": [{"text": text.replace("content" if notice else "information", "{NAME}")}],
                    "assessments": [
                        {
                            "sensitiveInformationPolicy": {
                                "piiEntities": [
                                    {
                                        "type": "NAME",
                                        "action": "ANONYMIZED",
                                        "detected": True,
                                        "match": "content" if notice else "information",
                                    }
                                ],
                                "regexes": [],
                            }
                        }
                    ],
                }
            usage = operation.output_shape.members.get("usage")
            if usage is not None:
                response["usage"] = {
                    name: 0
                    for name, member in usage.members.items()
                    if member.type_name in {"integer", "long"}
                }
            validate_parameters(
                {key: value for key, value in response.items() if key != "ResponseMetadata"},
                operation.output_shape,
            )
            if notice and drift is not None:
                drift(selected, monkeypatch)
            return response

    client = Client()
    configure_guard(monkeypatch, client)
    selected.client = client

    def request(url, ips, timeout, max_bytes):
        assert url in (*MODEL_URLS, TRAIN_URL, *CASE_URLS)
        assert ips == ("93.184.216.34",) and 0 < timeout <= 5 and max_bytes == 256 * 1024
        selected.fetches.append(url)
        text = FIRST if url in MODEL_URLS else (SECOND if url == TRAIN_URL else DENIED + " " + RETURNED_REF)
        return 200, {"content-type": "text/plain"}, text.encode()

    monkeypatch.setattr(public_fetch, "_request_public", request)
    return selected


def test_normal_api_last_unit_block_checks_notice_and_concludes_only_from_prior_strict_allow(monkeypatch):
    selected = scenario(monkeypatch)
    response = run(selected)
    assert selected.model.calls == 4, "exhausted plan skipped independently checked withheld notice"
    assert selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.message == "🔧 used: research_fetch_urls\n\n" + GAP
    assert response.data["input_admission"]["status"] == "quarantined"
    state = selected.backends[0]._research_binding
    assert state["admitted"] == {"model-card", "training-card"}
    assert state["attempted"] == {"model-card", "training-card", "benchmark-card"}
    assert state["blocked"] == {"benchmark-card"}
    assert not any(
        selected.backends[0]._research_unit_eligible(state, unit) for unit in state["units"].values()
    )
    records = [
        record
        for record in response.data["input_admission"]["records"]
        if record["boundary"] == "TOOL_RESULT"
    ]
    assert [record["outcome"] for record in records] == ["ALLOW", "ALLOW", "BLOCKED", "ALLOW"]
    assert records[-2]["check_id"] != records[-1]["check_id"]
    history = selected.agents[0].messages
    assert DENIED not in json.dumps(history) and RETURNED_REF not in json.dumps(history)
    assert DENIED not in json.dumps(selected.model.messages) and RETURNED_REF not in json.dumps(
        selected.model.messages
    )
    uses = [block["toolUse"] for message in history for block in message["content"] if "toolUse" in block]
    assert len(uses) == 3 and uses[-1]["input"] == {"urls": list(CASE_URLS), "max_pages": 3}
    results = [
        block["toolResult"] for message in history for block in message["content"] if "toolResult" in block
    ]
    assert (
        len(results) == 3
        and results[-1]["status"] == "error"
        and results[-1]["content"] == [{"text": WITHHELD}]
    )
    assert all(
        record["native_admission"] is False and record["quality_success"] is False for record in records
    )
    assert len(selected.backends[0]._model_acquisition_store._records) == 0
    assert (
        len([call for call in selected.client.calls if WITHHELD in call["content"][0]["text"]["text"]]) == 1
    )


@pytest.mark.parametrize("fixed", ["blocked", "unknown", "masked"])
def test_exhausted_plan_notice_requires_unchanged_strict_allow(monkeypatch, fixed):
    selected = scenario(monkeypatch, fixed=fixed)
    response = run(selected)
    assert selected.model.calls == 3 and selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.data["input_admission"]["status"] == "held"
    assert (
        len([call for call in selected.client.calls if WITHHELD in call["content"][0]["text"]["text"]]) == 1
    )


@pytest.mark.parametrize("earlier", ["blocked", "masked"])
def test_without_prior_strict_allow_exhaustion_stays_held_after_notice_check(monkeypatch, earlier):
    selected = scenario(monkeypatch, earlier=earlier)
    response = run(selected)
    assert selected.model.calls == 3 and selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.data["input_admission"]["status"] == "held"
    assert selected.backends[0]._research_binding["admitted"] == set()
    assert WITHHELD in selected.client.calls[-1]["content"][0]["text"]["text"]


def test_prior_unknown_assessment_still_stops_without_notice_or_additional_source(monkeypatch):
    selected = scenario(monkeypatch, earlier="unknown")
    response = run(selected)
    assert selected.model.calls == 1 and selected.fetches == list(MODEL_URLS)
    assert response.data["input_admission"]["status"] == "held"
    assert not any(WITHHELD in call["content"][0]["text"]["text"] for call in selected.client.calls)


@pytest.mark.parametrize(
    "drift",
    [
        lambda selected, monkeypatch: monkeypatch.setenv("AMLAS_GUARDRAIL_VERSION", "1"),
        lambda selected, monkeypatch: selected.policy.allowlist.append("changed-current-policy"),
    ],
)
def test_exhaustion_notice_never_bypasses_current_guard_or_policy(monkeypatch, drift):
    selected = scenario(monkeypatch, drift=drift)
    response = run(selected)
    assert selected.model.calls == 3 and selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.data["input_admission"]["status"] == "held"


def test_exhausted_notice_cannot_retry_last_source(monkeypatch):
    selected = scenario(monkeypatch, repeat=True)
    response = run(selected)
    assert selected.model.calls == 4 and selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.data["input_admission"]["status"] == "held"
    assert len(selected.backends[0]._model_acquisition_store._records) == 0


def test_equal_source_origins_do_not_count_as_independent_admitted_evidence(monkeypatch):
    original = research_runtime_profile._plan()
    shared = _ResearchPlan(
        tuple(dataclasses.replace(unit, origin_id="shared-research-origin") for unit in original.units)
    )
    monkeypatch.setattr(research_runtime_profile, "_plan", lambda: shared)
    selected = scenario(monkeypatch)
    response = run(selected)
    assert selected.model.calls == 3 and selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.data["input_admission"]["status"] == "held"
    assert WITHHELD in selected.client.calls[-1]["content"][0]["text"]["text"]


def test_exhausted_mixed_history_related_allow_plus_independent_allow_stays_held(monkeypatch):
    original = research_runtime_profile._plan()
    model, corpus, benchmark = original.units
    mixed = _ResearchPlan((dataclasses.replace(model, origin_id=benchmark.origin_id), corpus, benchmark))
    monkeypatch.setattr(research_runtime_profile, "_plan", lambda: mixed)
    selected = scenario(monkeypatch)
    response = run(selected)
    assert selected.model.calls == 3, "unrelated witness authorized related historical source content"
    assert selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.data["input_admission"]["status"] == "held"
    state = selected.backends[0]._research_binding
    assert state["admitted"] == {"model-card", "training-card"}
    assert state["forwarded"] == {"model-card", "training-card"}
    assert WITHHELD in selected.client.calls[-1]["content"][0]["text"]["text"]
    history = selected.agents[0].messages
    assert FIRST in json.dumps(history) and SECOND in json.dumps(history)
    assert DENIED not in json.dumps(history) and RETURNED_REF not in json.dumps(history)
    assert len(selected.backends[0]._model_acquisition_store._records) == 0


def test_exhausted_mixed_history_masked_plus_independent_allow_stays_held(monkeypatch):
    selected = scenario(monkeypatch, first="masked")
    response = run(selected)
    assert selected.model.calls == 3, "unrelated witness authorized masked historical source content"
    assert selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert response.data["input_admission"]["status"] == "held"
    state = selected.backends[0]._research_binding
    assert state["admitted"] == {"training-card"}
    assert state["forwarded"] == {"model-card", "training-card"}
    records = [
        record
        for record in response.data["input_admission"]["records"]
        if record["boundary"] == "TOOL_RESULT"
    ]
    assert [record["outcome"] for record in records] == ["ANONYMIZED", "ALLOW", "BLOCKED", "ALLOW"]
    history = selected.agents[0].messages
    assert "{NAME}" in json.dumps(history) and SECOND in json.dumps(history)
    assert DENIED not in json.dumps(history) and RETURNED_REF not in json.dumps(history)
    assert len(selected.backends[0]._model_acquisition_store._records) == 0
