"""Cold report DATA trace from admitted SDK history; no metrics restoration/effects."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from strands import Agent, Snapshot
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from test_normal_completion_sdk_snapshot import InertSnapshotModel

from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.tool_binding_backend import (
    ToolBindingBackend,
    _InputAdmissionHeldError,
    _record_session_snapshot,
    render_agent_turn,
)


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    import boto3

    def refused(*args, **kwargs):
        raise AssertionError("declared report DATA cannot call a provider")

    monkeypatch.setattr(boto3, "Session", refused)
    monkeypatch.setattr(boto3, "client", refused)


def use(identifier, name, args=None):
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": identifier, "name": name, "input": args or {}}}],
    }


def returned(identifier, *, status="success"):
    return {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": identifier,
                    "status": status,
                    "content": [{"json": {"data_only": True, "status": "declared_completed"}}],
                }
            }
        ],
    }


def admitted_prefix():
    return [
        {"role": "user", "content": [{"text": "Declared cold report history."}]},
        use("declared-research-use", "research_fetch_urls", {"urls": ["https://example.com/source"]}),
        returned("declared-research-use"),
        use(
            "declared-sft-use",
            "submit_qwen_sft",
            {
                "model_request_ref": "model_acq_" + "2" * 32,
                "preparation_request_ref": "prep_" + "3" * 32,
                "recipe_id": "full_v1",
            },
        ),
        returned("declared-sft-use"),
    ]


def sdk_result(names):
    metrics = EventLoopMetrics()
    for index, name in enumerate(names):
        metrics.tool_metrics[str(index)] = SimpleNamespace(
            tool={"toolUseId": "declared-current-" + str(index), "name": name, "input": {"declared": index}}
        )
    return AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "Declared final report."}]},
        metrics=metrics,
        state={},
    )


def backend():
    # These extraction methods are DATA-only; no stores, provider or guard is constructed.
    return ToolBindingBackend.__new__(ToolBindingBackend)


def extract_cold(owner, result, messages):
    function = getattr(owner, "_extract_normal_completion_executions", None)
    assert callable(function), "missing-symbol: cold admitted-history report trace"
    return function(result, admitted_messages=messages)


def test_fresh_public_snapshot_report_retains_pre_and_post_sft_tool_trace_without_metrics_restore():
    messages = admitted_prefix()
    saved_model, fresh_model = InertSnapshotModel(), InertSnapshotModel()
    original = Agent(model=saved_model, tools=[], messages=copy.deepcopy(messages), callback_handler=None)
    snapshot = _record_session_snapshot(original)
    assert set(snapshot["data"]) == {"messages", "state", "conversation_manager_state", "interrupt_state"}
    before = copy.deepcopy(snapshot)
    fresh = Agent(model=fresh_model, tools=[], callback_handler=None)
    fresh.load_snapshot(Snapshot.from_dict(snapshot))
    assert _record_session_snapshot(fresh)["data"] == before["data"]
    current_names = ["generate_coding_candidates", "evaluate_coding_pair"]
    warm_names = ["research_fetch_urls", "submit_qwen_sft", *current_names]
    result = sdk_result(current_names)
    metric_before = copy.deepcopy(result.metrics.tool_metrics)
    owner = backend()
    executions = extract_cold(owner, result, fresh.messages)
    assert [item.tool_name for item in executions] == warm_names
    assert all(item.status == "executed" and item.result is None for item in executions)
    from amlas.tui.chat.tool_binding_backend import AgentTurn

    cold = AgentTurn("Declared final report.", executions)
    warm = AgentTurn("Declared final report.", owner._extract_executions(sdk_result(warm_names)))
    assert render_agent_turn(cold) == render_agent_turn(warm)
    assert [
        item.tool.name if hasattr(item.tool, "name") else item.tool["name"]
        for item in result.metrics.tool_metrics.values()
    ] == current_names
    assert result.metrics.tool_metrics == metric_before
    assert snapshot == before and fresh.messages == messages
    assert saved_model.stream_calls == fresh_model.stream_calls == 0


@pytest.mark.parametrize(
    "kind",
    [
        "unmatched",
        "pending",
        "duplicate-use",
        "duplicate-result",
        "malformed-input",
        "failed-result",
        "wrong-order",
    ],
)
def test_ambiguous_or_unfinished_history_never_counts_as_executed(kind):
    historical_name = "declared_history_tool"
    history = [use("declared-history-use", historical_name), returned("declared-history-use")]
    if kind == "unmatched":
        history[1]["content"][0]["toolResult"]["toolUseId"] = "declared-foreign-use"
    elif kind == "pending":
        history.pop()
    elif kind == "duplicate-use":
        history.insert(1, copy.deepcopy(history[0]))
    elif kind == "duplicate-result":
        history.append(copy.deepcopy(history[1]))
    elif kind == "malformed-input":
        history[0]["content"][0]["toolUse"]["input"] = "declared malformed input"
    elif kind == "failed-result":
        history[1]["content"][0]["toolResult"]["status"] = "error"
    else:
        history.reverse()
    before = copy.deepcopy(history)
    result = sdk_result(["declared_current_tool"])
    owner = backend()
    function = getattr(owner, "_extract_normal_completion_executions", None)
    assert callable(function), "missing-symbol: cold admitted-history report trace"
    try:
        executions = function(result, admitted_messages=history)
    except ValueError:
        # Closed rejection is permitted; no historical execution is emitted.
        executions = []
    assert historical_name not in [item.tool_name for item in executions]
    assert history == before
    assert len(result.metrics.tool_metrics) == 1


def test_cold_history_and_current_metrics_keep_existing_first_seen_name_deduplication():
    history = [
        use("declared-first", "declared_history_tool"),
        returned("declared-first"),
        use("declared-second", "declared_history_tool"),
        returned("declared-second"),
    ]
    result = sdk_result(["declared_history_tool", "declared_current_tool"])
    executions = extract_cold(backend(), result, history)
    assert [item.tool_name for item in executions] == ["declared_history_tool", "declared_current_tool"]


def test_ordinary_extractor_keeps_current_metrics_order_and_argument_identity():
    result = sdk_result(["declared_current_tool", "declared_other_tool"])
    executions = backend()._extract_executions(result)
    assert [item.tool_name for item in executions] == ["declared_current_tool", "declared_other_tool"]
    assert executions[0].args is result.metrics.tool_metrics["0"].tool["input"]
    assert all(item.status == "executed" and item.result is None for item in executions)


def test_rejected_snapshot_input_cannot_load_agent_or_extract_history(monkeypatch):
    messages = admitted_prefix()
    model = InertSnapshotModel()
    original = Agent(model=model, tools=[], messages=copy.deepcopy(messages), callback_handler=None)
    snapshot = _record_session_snapshot(original)
    owner = backend()
    owner.approval_policy = ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"], interactive=False
    )
    owner._normal_original_result = None
    calls = []
    monkeypatch.setattr(owner, "_selected_input_checker", lambda: object())

    def rejected(value, boundary, *, immutable):
        assert boundary == "HISTORY_TEXT" and immutable is True
        # The first text of the first stored message: its text block.
        assert value == messages[0]["content"][0]["text"]
        calls.append("admit")
        raise _InputAdmissionHeldError()

    def forbidden(*args, **kwargs):
        calls.append("forbidden")
        raise AssertionError("rejected history cannot load, invoke or render")

    monkeypatch.setattr(owner, "_admit_input", rejected)
    monkeypatch.setattr(owner, "build_agent", forbidden)
    monkeypatch.setattr(owner, "_extract_normal_completion_executions", forbidden, raising=False)
    engine = ChatEngine.__new__(ChatEngine)
    engine.tool_backend = owner
    with pytest.raises(_InputAdmissionHeldError):
        engine.load_normal_snapshot(
            {
                "schema_version": 2,
                "logical_run_id": "0" * 32,
                "private_store_correlation": "declared-component-store",
                "native_job_name": "declared-component-sft",
                "native_job_arn": "declared-component-sft-arn",
                "operation_digest": "a" * 64,
                "source_refs": [],
                "causal_parent_ref": None,
                "sdk_snapshot": snapshot,
                "pending_tool_use_id": "declared-component-sft-use",
            }
        )
    assert calls == ["admit"] and not hasattr(engine, "_normal_agent")
    assert model.stream_calls == 0


def test_qualified_normal_warm_return_matches_cold_causal_tool_trace(monkeypatch):
    """Actual chat return path must use the same admitted history as cold return."""
    from amlas.config.settings import Settings
    from amlas.control_plane import normal_run_checkpoint
    from amlas.tui.chat.tool_binding_backend import AgentTurn

    history = admitted_prefix()
    current = sdk_result(["generate_coding_candidates", "evaluate_coding_pair"])
    owner = ToolBindingBackend(
        Settings(),
        model=InertSnapshotModel(),
        approval_policy=ApprovalPolicy(
            mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"], interactive=False
        ),
    )
    engine = ChatEngine.__new__(ChatEngine)
    engine.tool_backend = owner
    owner._warm_owner = engine
    calls = []
    normal = SimpleNamespace(
        enroll=lambda *args: calls.append("enroll"),
        bind_sft_store=lambda *args, **kwargs: calls.append("bind"),
        _check=lambda: calls.append("current"),
    )
    monkeypatch.setattr(normal_run_checkpoint, "current_normal_run", lambda: normal)
    monkeypatch.setattr(owner, "_selected_input_checker", lambda: object())

    def admitted(value, boundary, **kwargs):
        calls.append(boundary)
        return value

    class DeclaredAgent:
        callback_handler = None

        def __init__(self):
            self.messages = copy.deepcopy(history)

        def __call__(self, prompt, *, invocation_state):
            assert "OWNER_PROMPT" in calls and "HISTORY_TEXT" in calls
            calls.append("invoke")
            return current

    agent = DeclaredAgent()
    monkeypatch.setattr(owner, "_admit_input", admitted)
    monkeypatch.setattr(owner, "build_agent", lambda **kwargs: agent)
    monkeypatch.setattr(owner, "_qualify_research_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(owner, "_pending_pause_result", lambda *args: False)
    turn = owner.chat_with_tools("Declared normal report.", history=copy.deepcopy(history))
    cold = AgentTurn(
        "Declared final report.",
        owner._extract_normal_completion_executions(current, admitted_messages=copy.deepcopy(history)),
    )
    assert render_agent_turn(turn) == render_agent_turn(cold), (
        "qualified normal warm return lost admitted historical tools retained by cold return"
    )
    assert [item.tool_name for item in turn.executions] == [item.tool_name for item in cold.executions]
    assert calls.count("invoke") == 1 and "enroll" in calls and "bind" in calls
    assert agent.messages == history and len(current.metrics.tool_metrics) == 2
