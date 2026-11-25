"""Research results and a prompt that hold a character outside the Basic Multilingual Plane, as a full turn.

The scripted guard answers with the coverage count the native service reports: ``guardrailCoverage`` in
UTF-16 code units. The scripted guards of the other research turns of this folder return no coverage, so
those turns never reach the coverage check. The research tool, the backend callbacks and the SDK are real;
the pages, the name and the address are the test's own.

No provider or network call. These tests say nothing about a deployed guardrail or a replacement worker.
"""

from __future__ import annotations

import json
import logging

import pytest
from test_input_admission_wiring import ClientFixture, configure_guard
from test_normal_research_profile import CASE_URLS, MODEL_URLS, PROMPT, TRAIN_URL, LifecycleModel, run, setup

from amlas.tools.research import public_fetch

SUPPLEMENTARY = "\U0001f419"
PLAIN_PAGE = "Ordinary neutral source information."
SIGN_PAGE = "Ordinary neutral source information with one sign " + SUPPLEMENTARY + " in it."
NAME, ADDRESS = "Jordan", "jordan@example.com"
CONTACT_PAGE = "The corpus card " + SUPPLEMENTARY + " names " + NAME + " as contact, at " + ADDRESS + "."
UNITS = ["benchmark-card", "model-card", "training-card"]


def utf16_units(text):
    return len(text.encode("utf-16-le")) // 2


class CoverageClient(ClientFixture):
    """The folder's NONE answer plus full coverage in UTF-16 code units; a service mask for the contact."""

    def __init__(self):
        super().__init__(deny=lambda text: False)

    def apply_guardrail(self, **kwargs):
        text = kwargs["content"][0]["text"]["text"]
        response = super().apply_guardrail(**kwargs)
        counts = {"guarded": utf16_units(text), "total": utf16_units(text)}
        response["guardrailCoverage"] = {"textCharacters": counts}
        if NAME in text and ADDRESS in text:
            response["action"] = "GUARDRAIL_INTERVENED"
            response["outputs"] = [{"text": text.replace(ADDRESS, "{EMAIL}").replace(NAME, "{NAME}")}]
            response["assessments"][0]["sensitiveInformationPolicy"] = {
                "piiEntities": [
                    {"type": "EMAIL", "action": "ANONYMIZED", "detected": True, "match": ADDRESS},
                    {"type": "NAME", "action": "ANONYMIZED", "detected": True, "match": NAME},
                ],
                "regexes": [],
            }
        return response


class ThreeUnitModel(LifecycleModel):
    """Three research units, then one ordinary registered tool, then a final text."""

    def __init__(self):
        super().__init__()
        self.requests.insert(
            2, ("benchmark-readonly-use", "research_fetch_urls", {"urls": list(CASE_URLS), "max_pages": 3})
        )


def turn(monkeypatch, caplog, *, pages, prompt=PROMPT, client=None):
    from amlas.tools.coding_preparation import list_coding_preparation_profiles

    selected = setup(
        monkeypatch, model_override=ThreeUnitModel(), extra_tools=(list_coding_preparation_profiles,)
    )
    selected.client = client or CoverageClient()
    configure_guard(monkeypatch, selected.client)

    def request(url, ips, timeout, max_bytes):
        assert url in (*MODEL_URLS, TRAIN_URL, *CASE_URLS)
        selected.fetches.append(url)
        return 200, {"content-type": "text/plain"}, pages(url).encode()

    monkeypatch.setattr(public_fetch, "_request_public", request)
    with caplog.at_level(logging.WARNING):
        response = run(selected, prompt)
    admission = response.data["input_admission"]
    return {
        "selected": selected,
        "response": response,
        "state": selected.backends[0]._research_binding,
        "status": admission["status"],
        "records": [(r["boundary"], r["outcome"], r["reason"]) for r in admission["records"]],
        "results": [
            (block["toolResult"]["toolUseId"], block["toolResult"]["status"])
            for message in selected.agents[0].messages
            for block in message["content"]
            if "toolResult" in block
        ],
        "hold_lines": [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("AMLAS_INPUT_HOLD")
        ],
    }


ALLOWED = ("ALLOW", "NATIVE_NONE")
COMPLETED = [
    ("original-readonly-use", "success"),
    ("independent-readonly-use", "success"),
    ("benchmark-readonly-use", "success"),
    ("profile-discovery", "success"),
]


def test_research_pages_with_a_supplementary_character_are_admitted_and_the_turn_continues(
    monkeypatch, caplog
):
    seen = turn(monkeypatch, caplog, pages=lambda url: SIGN_PAGE)
    selected, state = seen["selected"], seen["state"]
    assert selected.model.calls == 5 and len(selected.fetches) == 5
    assert seen["status"] == "admitted" and selected.backends[0]._input_hold is False
    assert seen["records"] == [("OWNER_PROMPT", *ALLOWED)] + [("TOOL_RESULT", *ALLOWED)] * 4
    assert seen["results"] == COMPLETED and seen["hold_lines"] == []
    assert state["phase"] == "CONSUMED" and state["pending"] is False
    assert sorted(state["attempted"]) == sorted(state["forwarded"]) == sorted(state["admitted"]) == UNITS
    assert state["blocked"] == set()


def test_masked_research_page_with_a_supplementary_character_is_forwarded_and_the_turn_continues(
    monkeypatch, caplog
):
    seen = turn(monkeypatch, caplog, pages=lambda url: CONTACT_PAGE if url == TRAIN_URL else SIGN_PAGE)
    selected, state = seen["selected"], seen["state"]
    assert selected.model.calls == 5 and len(selected.fetches) == 5
    # The turn is not held: the masked page is forwarded and the ordinary tool after research runs.
    assert seen["status"] == "admitted" and selected.backends[0]._input_hold is False
    assert seen["records"] == [
        ("OWNER_PROMPT", *ALLOWED),
        ("TOOL_RESULT", *ALLOWED),
        ("TOOL_RESULT", "ANONYMIZED", "NATIVE_MASKED"),
        ("TOOL_RESULT", *ALLOWED),
        ("TOOL_RESULT", *ALLOWED),
    ]
    assert seen["results"] == COMPLETED and seen["hold_lines"] == []
    assert state["phase"] == "CONSUMED" and state["pending"] is False
    # A masked unit is forwarded; only an unchanged strict NONE counts as admitted source evidence.
    assert sorted(state["attempted"]) == sorted(state["forwarded"]) == UNITS
    assert sorted(state["admitted"]) == ["benchmark-card", "model-card"] and state["blocked"] == set()
    # The agent history and the model hold the service's substitution, never the name or the address.
    for stored in (selected.agents[0].messages, selected.model.messages):
        text = json.dumps(stored, ensure_ascii=False)
        assert "{NAME}" in text and "{EMAIL}" in text
        assert NAME not in text and ADDRESS not in text


def test_prompt_with_a_supplementary_character_is_admitted_and_the_turn_continues(monkeypatch, caplog):
    prompt = PROMPT + " " + SUPPLEMENTARY
    seen = turn(monkeypatch, caplog, pages=lambda url: PLAIN_PAGE, prompt=prompt)
    selected = seen["selected"]
    assert selected.client.calls[0]["content"] == [{"text": {"text": prompt}}]
    assert selected.model.calls == 5 and seen["status"] == "admitted"
    assert seen["records"] == [("OWNER_PROMPT", *ALLOWED)] + [("TOOL_RESULT", *ALLOWED)] * 4
    assert seen["results"] == COMPLETED and seen["hold_lines"] == []


@pytest.mark.parametrize("short", ["guarded", "both"])
def test_research_page_with_short_coverage_is_still_held(monkeypatch, caplog, short):
    class ShortCoverageClient(CoverageClient):
        def apply_guardrail(self, **kwargs):
            response = super().apply_guardrail(**kwargs)
            if SIGN_PAGE.split(SUPPLEMENTARY)[0] in kwargs["content"][0]["text"]["text"]:
                counts = response["guardrailCoverage"]["textCharacters"]
                counts["guarded"] -= 1
                if short == "both":
                    counts["total"] -= 1
            return response

    seen = turn(monkeypatch, caplog, pages=lambda url: SIGN_PAGE, client=ShortCoverageClient())
    selected = seen["selected"]
    assert seen["response"].message.endswith("Model input was held at the admission boundary.")
    assert seen["status"] == "held" and selected.backends[0]._input_hold is True
    assert seen["records"] == [("OWNER_PROMPT", *ALLOWED), ("TOOL_RESULT", "UNKNOWN", "NATIVE_INVALID")]
    assert selected.model.calls == 1 and seen["results"] == [("original-readonly-use", "error")]
    assert seen["hold_lines"] == ["AMLAS_INPUT_HOLD phase=research_after_result reason=callback_refused"]
