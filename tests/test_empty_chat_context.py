"""Empty state adds no instructions; meaningful composite admission stays required.

Normal API/ChatEngine/SDK with neutral literal inputs and inert guard/model ports.
No denied native prompt is reconstructed, retried or used in these tests.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from test_input_admission_wiring import ClientFixture, ModelFixture, configure_guard, research_fetch_urls

from amlas import api
from amlas.config import externalized
from amlas.config.settings import LocalConfig, Settings
from amlas.tui.chat import engine as engine_source
from amlas.tui.chat import tool_binding_backend as binding

LITERAL = "Explain this neutral example."
EMPTY_HINT = "No system state available yet. Run /status or /leaderboard to load state."


def setup(monkeypatch, tmp_path, *, composite_block=False):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AMLAS_RUNTIME_RESEARCH_PROFILE", "off")
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *_a, **_kw: "")
    client = ClientFixture(deny=lambda text: composite_block and "\n\n## Context\n" in text)
    configure_guard(monkeypatch, client)
    model, backends, agents = ModelFixture(), [], []
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [research_fetch_urls])
    actual_init = binding.ToolBindingBackend.__init__
    actual_build = binding.ToolBindingBackend.build_agent

    def actual_backend(self, *args, **kwargs):
        kwargs["model"] = model
        actual_init(self, *args, **kwargs)
        backends.append(self)

    def actual_agent(self, **kwargs):
        agent = actual_build(self, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(binding.ToolBindingBackend, "__init__", actual_backend)
    monkeypatch.setattr(binding.ToolBindingBackend, "build_agent", actual_agent)
    settings = Settings(
        local=LocalConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            specs_dir=str(tmp_path / "specs"),
            data_dir=str(tmp_path / "data"),
        )
    )
    return SimpleNamespace(client=client, model=model, settings=settings, backends=backends, agents=agents)


def test_fresh_chat_engine_empty_state_is_absent_context(monkeypatch, tmp_path):
    selected = setup(monkeypatch, tmp_path)
    engine = engine_source.ChatEngine(settings=selected.settings)
    assert engine.context_bus is None and engine.session_ledger is None
    assert engine._build_context() == "", "empty state became synthetic instructions"
    assert selected.model.calls == 0 and selected.client.calls == []


def test_normal_api_empty_workspace_preserves_owner_literal_once_without_generated_context(
    monkeypatch, tmp_path
):
    selected = setup(monkeypatch, tmp_path)
    response = asyncio.run(api.run_prompt(LITERAL, settings=selected.settings))
    assert response.message == "🔧 used: research_fetch_urls\n\nordinary complete"
    assert len(selected.backends) == len(selected.agents) == 1 and selected.model.calls == 2
    assert selected.model.messages[0][0] == {"role": "user", "content": [{"text": LITERAL}]}
    assert EMPTY_HINT not in json.dumps(selected.model.messages)
    records = response.data["input_admission"]["records"]
    assert [record["boundary"] for record in records] == ["OWNER_PROMPT", "TOOL_RESULT"]
    assert records[0]["outcome"] == "ALLOW"
    assert sum(call["content"][0]["text"]["text"] == LITERAL for call in selected.client.calls) == 1
    assert all(
        call["source"] == "INPUT"
        and call["guardrailIdentifier"] == "ordinaryselected"
        and call["guardrailVersion"] == "DRAFT"
        for call in selected.client.calls
    )


def test_meaningful_context_components_allow_but_combined_block_still_holds_before_model(
    monkeypatch, tmp_path
):
    selected = setup(monkeypatch, tmp_path, composite_block=True)
    engine = engine_source.ChatEngine(settings=selected.settings)
    engine.context["data_summary"] = {"path": "neutral.csv", "rows": 3, "cols": 2, "columns": ["value"]}
    context = engine._build_context()
    assert "## Data Summary: neutral.csv (3 rows \u00d7 2 cols)" in context
    response = asyncio.run(engine.process_message(LITERAL))
    assert response.action == "input_admission_held"
    assert selected.model.calls == 0 and selected.agents == []
    records = response.data["input_admission"]["records"]
    assert [(record["boundary"], record["outcome"]) for record in records] == [
        ("OWNER_PROMPT", "ALLOW"),
        ("GENERATED_CONTEXT", "ALLOW"),
        ("GENERATED_CONTEXT", "BLOCKED"),
    ]
    assert [call["content"][0]["text"]["text"] for call in selected.client.calls] == [
        LITERAL,
        context,
        LITERAL + "\n\n## Context\n" + context,
    ]
    assert response.data["input_admission"]["status"] == "held"
    assert LITERAL not in json.dumps(response.data["input_admission"])
