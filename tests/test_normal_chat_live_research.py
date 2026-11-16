"""Normal chat research binding through real SDK tools; all providers are inert."""

from __future__ import annotations

import json

import boto3
import pytest
from strands.hooks.events import AfterToolCallEvent, BeforeInvocationEvent
from test_sdk_prompt_policy import ScriptedModel

from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tools.research import public_fetch, strands_tools
from amlas.tools.research.types import ResearchResult, ResearchSource
from amlas.tui.chat.agent_tools import (
    READ_ONLY_TOOLS,
    STATE_CHANGING_TOOLS,
    TUI_SYSTEM_PROMPT,
    get_safe_llm_tools,
    tool_display_name,
)
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

URL = "https://example.org/supplied-source"


@pytest.fixture(autouse=True)
def inert_aws(monkeypatch):
    """No credential setup or AWS request is needed for either research tool."""
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("AMLAS_TAVILY_KEYLESS", "true")
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("Research binding must not construct an AWS client")

    monkeypatch.setattr(boto3, "client", blocked)
    monkeypatch.setattr(boto3, "Session", blocked)
    yield
    assert attempts == []


def install_sources(monkeypatch, *, failure="none"):
    calls = []

    def provider(name, source):
        def search(query, *, max_results):
            calls.append((name, query, max_results))
            if failure == "all" or failure == name:
                raise RuntimeError("Neutral fixture provider unavailable")
            return [
                ResearchResult(
                    source=source,
                    title="Neutral source " + name,
                    url="https://example.org/research/" + name,
                    snippet="Unverified fixture source material.",
                )
            ]

        return search

    monkeypatch.setattr(strands_tools, "arxiv_search", provider("arxiv", ResearchSource.ARXIV))
    monkeypatch.setattr(
        strands_tools,
        "semantic_scholar_search",
        provider("semantic_scholar", ResearchSource.SEMANTIC_SCHOLAR),
    )
    monkeypatch.setattr(strands_tools, "tavily_search", provider("tavily", ResearchSource.TAVILY))
    return calls


def install_public_transport(monkeypatch):
    """Retain the real finite fetch loop/envelope; DNS and HTTP are inert fixtures."""
    calls = []

    def checked(url, timeout):
        assert url.startswith("https://example.org/")
        assert 0 < timeout <= 5
        return ("93.184.216.34",)

    def request(url, ips, timeout, max_bytes):
        assert ips == ("93.184.216.34",)
        assert 0 < timeout <= 5 and max_bytes == 256 * 1024
        calls.append(url)
        if url.endswith("/unavailable"):
            return 503, {"content-type": "text/plain"}, b"Unavailable fixture source"
        body = (
            "<html><title>Fixture source</title><body>"
            + "Ordinary unverified source material. " * 800
            + '<a href="https://example.org/unrequested-link">Other source</a></body></html>'
        ).encode()
        return 200, {"content-type": "text/html"}, body

    monkeypatch.setattr(public_fetch, "_checked_addresses", checked)
    monkeypatch.setattr(public_fetch, "_request_public", request)
    return calls


@pytest.mark.parametrize(
    ("name", "properties", "required"),
    [
        ("research_search", {"query", "max_results", "urls"}, {"query"}),
        ("research_fetch_urls", {"urls", "max_pages"}, {"urls"}),
    ],
)
def test_existing_research_tools_have_real_sdk_schema_and_readonly_binding(name, properties, required):
    tools = {tool_display_name(tool): tool for tool in get_safe_llm_tools()}
    assert name in tools
    assert tools[name] is getattr(strands_tools, name)
    assert name in READ_ONLY_TOOLS and name not in STATE_CHANGING_TOOLS
    schema = tools[name].tool_spec["inputSchema"]["json"]
    assert set(schema["properties"]) == properties
    assert set(schema["required"]) == required
    assert not any(key.startswith("_amlas_") for key in schema["properties"])


@pytest.mark.parametrize("failure", ["none", "tavily", "all"])
def test_normal_direct_search_keeps_provider_failures_and_coverage(monkeypatch, failure):
    calls = install_sources(monkeypatch, failure=failure)
    policy = ApprovalPolicy()
    backend = ToolBindingBackend(Settings(), approval_policy=policy)
    result = backend.execute_tool_use("research_search", {"query": "Public model research", "max_results": 2})
    assert result.status == "executed", result
    output = result.result
    assert set(output["queried_sources"]) == {"arxiv", "semantic_scholar", "tavily"}
    assert output["coverage"] == "queried_sources_only; failures remain explicit"
    assert len(output["results"]) <= 2
    assert {name for name, _query, _cap in calls} == set(output["queried_sources"])
    if failure == "none":
        assert output["errors"] == {} and len(output["results"]) == 2
    else:
        expected = {"arxiv", "semantic_scholar", "tavily"} if failure == "all" else {"tavily"}
        assert set(output["errors"]) == expected
        assert "research_fetch_urls" in output["next_action"]
        if failure == "all":
            assert output["results"] == []
    assert policy.mode == ApprovalMode.ASK and policy.allowlist == []


def test_normal_direct_fetch_retains_finite_coverage_and_untrusted_source_data(monkeypatch):
    calls = install_public_transport(monkeypatch)
    policy = ApprovalPolicy()
    backend = ToolBindingBackend(Settings(), approval_policy=policy)
    supplied = [URL, "https://example.org/unavailable", "https://example.org/skipped"]
    result = backend.execute_tool_use("research_fetch_urls", {"urls": supplied, "max_pages": 2})
    assert result.status == "executed", result
    output = result.result
    assert calls == supplied[:2]
    assert output["coverage"] == "provided_urls_only" and output["skipped_count"] == 1
    assert output["untrusted_data"] is True
    assert output["errors"] == [{"input_index": "1", "error": "public source returned HTTP 503"}]
    page = output["pages"][0]
    assert page["url"] == URL and page["untrusted_data"] is True
    assert page["text_truncated"] is True and page["content_coverage"] == "partial"
    assert 0 < page["text_characters_returned"] <= 8000
    assert output["budgets"] == {
        "max_pages": 2,
        "max_bytes_per_page": 256 * 1024,
        "timeout_seconds": 15.0,
        "max_redirects": 3,
        "link_hops": 0,
    }
    assert policy.mode == ApprovalMode.ASK and policy.allowlist == []


def result_payload(event):
    for block in event.result["content"]:
        if "json" in block:
            return block["json"]
        if "text" in block:
            try:
                parsed = json.loads(block["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def test_normal_sdk_research_turn_uses_real_tools_and_fresh_policy(monkeypatch):
    source_calls = install_sources(monkeypatch)
    fetch_calls = install_public_transport(monkeypatch)
    requests = [
        {"id": "search", "name": "research_search", "input": {"query": "Public model research"}},
        {"id": "fetch", "name": "research_fetch_urls", "input": {"urls": [URL]}},
    ]
    policy = ApprovalPolicy()
    backend = ToolBindingBackend(
        Settings(), model=ScriptedModel([requests, requests]), approval_policy=policy
    )
    assert backend.is_tool_capable, "This proof requires the installed SDK, never its shim"
    build = backend.build_agent
    policies, observations = [], []

    def observed_build(**kwargs):
        agent = build(**kwargs)
        agent.callback_handler = lambda *args, **kwargs: None
        agent.hooks.add_callback(
            BeforeInvocationEvent,
            lambda event: policies.append(event.invocation_state["_amlas_approval_policy"]),
        )
        agent.hooks.add_callback(
            AfterToolCallEvent,
            lambda event: observations.append(
                (event.tool_use["name"], event.result["status"], result_payload(event))
            ),
        )
        return agent

    monkeypatch.setattr(backend, "build_agent", observed_build)
    for _ in range(2):
        turn = backend.chat_with_tools("Research the public sources and the supplied source URL.")
        assert turn is not None and turn.final_text == "done", turn
    assert len(observations) == 4 and all(status == "success" for _, status, _ in observations)
    assert {name for name, _status, _payload in observations} == {"research_search", "research_fetch_urls"}
    assert len(source_calls) == 6 and fetch_calls == [URL, URL]
    assert len(policies) == 2 and policies[0] is not policies[1]
    assert all(value is not policy and value.mode == ApprovalMode.ASK for value in policies)
    assert all(value.allowlist == [] for value in policies) and policy.allowlist == []
    assert all(
        payload["untrusted_data"] is True
        for name, _status, payload in observations
        if name == "research_fetch_urls"
    )


def test_normal_research_guidance_distinguishes_source_data_from_authority():
    assert "research_search" in TUI_SYSTEM_PROMPT and "research_fetch_urls" in TUI_SYSTEM_PROMPT
    assert "untrusted source data" in TUI_SYSTEM_PROMPT
