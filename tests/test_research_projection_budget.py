"""Registered SDK research projections: neutral ports, complete serialized budgets.

Execution belongs inside the AWS source-QA container. Synthetic public excerpts
and provider records are upstream inputs; target tools/aggregate/SDK remain real.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from strands import Agent
from strands.models import Model

from amlas.tools import pathref
from amlas.tools.research import public_fetch, strands_tools
from amlas.tools.research.types import ResearchResult, ResearchSource

TOOL_CONTENT_BYTES = 8000
URLS = [
    "https://example.com/research/one?topic=unicode&edition=1",
    "https://example.com/research/two?topic=unicode&edition=2",
    "https://example.com/research/three?topic=unicode&edition=3",
]
UNIT = 'Ordinary public research 研究🙂 "quoted passage" \\ excerpt.\n'
PAGE_TEXT = {url: UNIT * (710 + index) for index, url in enumerate(URLS)}
RESULT_URLS = [
    "https://example.org/paper/one",
    "https://example.org/paper/two",
    "https://example.org/paper/three",
]
SNIPPETS = {url: UNIT * (450 + index) for index, url in enumerate(RESULT_URLS)}


class RegisteredResearchModel(Model):
    """Installed SDK streams one genuine registered tool and receives its result."""

    def __init__(self, tool_name: str, tool_input: dict[str, Any]):
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.calls = 0
        self.results: list[dict[str, Any]] = []

    def update_config(self, **kwargs: Any) -> None:
        pass

    def get_config(self) -> dict[str, str]:
        return {"model_id": "offline-registered-research-projection"}

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ordinary registered SDK tool stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        if self.calls == 1:
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "research-projection", "name": self.tool_name}},
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": json.dumps(self.tool_input)}},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            return
        assert self.calls == 2
        self.results = [
            copy.deepcopy(block["toolResult"])
            for message in messages
            for block in message["content"]
            if "toolResult" in block
        ]
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "ordinary projection inspected"},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}


@pytest.fixture
def public_pages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Only native transport/DNS ports are synthetic; source fetch stays real."""
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "true")
    monkeypatch.delenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    seen: list[str] = []
    monkeypatch.setattr(pathref, "_resolve_host_ips", lambda _host: ["93.184.216.34"])

    def request(url: str, ips: list[str], timeout: float, limit: int):
        assert ips == ["93.184.216.34"]
        assert 0 < timeout <= 5 and limit == 262144
        seen.append(url)
        body = PAGE_TEXT[url].encode("utf-8")
        assert len(body) < limit
        return 200, {"content-type": "text/plain; charset=utf-8"}, body

    monkeypatch.setattr(public_fetch, "_request_public", request)
    return seen


def registered_result(tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = RegisteredResearchModel(tool_name, arguments)
    agent = Agent(
        model=model,
        tools=[strands_tools.research_fetch_urls, strands_tools.research_search],
        callback_handler=None,
    )
    agent("Inspect the ordinary public research projection.")
    assert model.calls == 2 and len(model.results) == 1
    result = model.results[0]
    assert result["toolUseId"] == "research-projection" and result["status"] == "success"
    content = result["content"]
    assert type(content) is list and len(content) == 1
    block = content[0]
    data = block["json"] if "json" in block else json.loads(block["text"])
    assert type(data) is dict
    return result, data


def assert_serialized_budget(result: dict[str, Any], data: dict[str, Any]) -> None:
    encoded = json.dumps(
        result["content"], ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    observed_bytes = len(encoded)
    assert observed_bytes <= TOOL_CONTENT_BYTES, (
        f"registered SDK ToolResult.content returned {observed_bytes} bytes"
    )
    assert data["projection_budget_bytes"] == TOOL_CONTENT_BYTES
    assert data["projection_coverage"] == "partial"


def assert_page_coverage(data: dict[str, Any]) -> None:
    assert data["coverage"] == "provided_urls_only" and data["untrusted_data"] is True
    assert not data["errors"] and data["skipped_count"] == 0
    assert [page["url"] for page in data["pages"]] == URLS
    for page in data["pages"]:
        original = PAGE_TEXT[page["url"]]
        assert 0 < len(page["text"]) <= 20000 and original.startswith(page["text"])
        assert page["text_characters_available"] == len(original)
        assert page["text_characters_returned"] == len(page["text"])
        assert page["bytes"] == len(original.encode("utf-8"))
        assert page["text_truncated"] is True and page["content_coverage"] == "partial"
        assert page["untrusted_data"] is True and page["http_status"] == 200


def test_registered_fetch_three_unicode_escaped_pages_fit_complete_tool_content_budget(
    public_pages,
):
    result, data = registered_result("research_fetch_urls", {"urls": URLS})
    assert public_pages == URLS
    assert_page_coverage(data)
    assert_serialized_budget(result, data)


def test_registered_search_pages_and_provider_metadata_share_complete_tool_content_budget(
    monkeypatch, public_pages
):
    records = [
        ResearchResult(
            source=ResearchSource.ARXIV if index < 2 else ResearchSource.TAVILY,
            title=f"Ordinary research item {index + 1}",
            url=url,
            snippet=SNIPPETS[url],
            authors=("Public Author",),
            identifiers={"public_label": f"ordinary-{index + 1}"},
        )
        for index, url in enumerate(RESULT_URLS)
    ]

    def arxiv(query, **kwargs):
        assert query == "ordinary coding research"
        return records[:2]

    def scholarly_unavailable(query, **kwargs):
        raise ValueError("ordinary source unavailable")

    def tavily(query, **kwargs):
        return records[2:]

    monkeypatch.setattr(
        strands_tools,
        "_build_default_sources",
        lambda: {"arxiv": arxiv, "semantic_scholar": scholarly_unavailable, "tavily": tavily},
    )
    result, data = registered_result(
        "research_search", {"query": "ordinary coding research", "max_results": 5, "urls": URLS}
    )
    assert public_pages == URLS
    assert data["queried_sources"] == ["arxiv", "semantic_scholar", "tavily"]
    assert set(data["errors"]) == {"semantic_scholar"}
    assert "unavailable" in data["errors"]["semantic_scholar"]
    assert data["coverage"] == "queried_sources_only; failures remain explicit"
    assert {item["url"] for item in data["results"]} == set(RESULT_URLS)
    assert_page_coverage(data["provided_url_retrieval"])
    assert_serialized_budget(result, data)
    for item in data["results"]:
        original = SNIPPETS[item["url"]]
        assert item["canonical_id"] == item["url"]
        assert item["sources"] == ["ARXIV" if item["url"] in RESULT_URLS[:2] else "TAVILY"]
        assert 0 < len(item["snippet"]) < len(original) and original.startswith(item["snippet"])
        assert item["snippet_characters_available"] == len(original)
        assert item["snippet_characters_returned"] == len(item["snippet"])
        assert item["snippet_truncated"] is True and item["content_coverage"] == "partial"
