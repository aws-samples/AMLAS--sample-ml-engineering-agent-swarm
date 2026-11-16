"""Fixed hold-site metadata through the real SDK; inert source/guard ports only."""

import logging

import pytest
from test_normal_research_profile import CASE_URLS, MODEL_URLS, TRAIN_URL, run
from test_research_exhaustion import scenario

LOGGER = "amlas.tui.chat.tool_binding_backend"
PREFIX = "AMLAS_INPUT_HOLD "
MARKER = "EXCLUDED_EXCEPTION_AND_POLICY_SENTINEL"


def source_records(caplog):
    return [
        record
        for record in caplog.records
        if record.name == LOGGER and record.getMessage().startswith(PREFIX)
    ]


def test_normal_sdk_independent_last_unit_notice_preserves_behavior_without_hold_log(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    selected = scenario(monkeypatch)
    response = run(selected)
    assert response.data["input_admission"]["status"] == "quarantined"
    assert selected.model.calls == 4
    assert selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    assert not source_records(caplog)
    assert not selected.backends[0]._model_acquisition_store._records


@pytest.mark.parametrize("failure", ["policy", "exception"])
def test_normal_sdk_post_notice_refusal_logs_only_fixed_source_phase(monkeypatch, caplog, failure):
    caplog.set_level(logging.INFO, logger=LOGGER)

    def revoke(selected, monkeypatch):
        if failure == "policy":
            selected.policy.allowlist.append(MARKER)
        else:

            def refuse(*args, **kwargs):
                raise RuntimeError(MARKER)

            monkeypatch.setattr(selected.backends[0], "_qualify_research_binding", refuse)

    selected = scenario(monkeypatch, drift=revoke)
    response = run(selected)
    assert response.data["input_admission"]["status"] == "held"
    assert selected.model.calls == 3
    assert selected.fetches == [*MODEL_URLS, TRAIN_URL, *CASE_URLS]
    receipts = [
        record
        for record in response.data["input_admission"]["records"]
        if record["boundary"] == "TOOL_RESULT"
    ]
    assert [record["outcome"] for record in receipts] == ["ALLOW", "ALLOW", "BLOCKED", "ALLOW"]
    records = source_records(caplog)
    assert len(records) == 1
    assert records[0].getMessage() == "AMLAS_INPUT_HOLD phase=research_after_result reason=callback_refused"
    assert records[0].args == () and records[0].exc_info is None
    assert MARKER not in "\n".join(record.getMessage() for record in records)
    assert not selected.backends[0]._model_acquisition_store._records


def test_actual_sdk_warning_failure_cannot_prevent_original_before_tool_refusal(monkeypatch):
    from amlas.tui.chat import tool_binding_backend

    selected = scenario(monkeypatch)
    selected.model.requests[0][2]["max_pages"] = 2

    def broken_warning(*args, **kwargs):
        raise RuntimeError(MARKER)

    monkeypatch.setattr(tool_binding_backend.logger, "warning", broken_warning)
    response = run(selected)
    assert response.data["input_admission"]["status"] == "held"
    assert selected.backends[0]._input_hold is True
    assert selected.model.calls == 1
    assert selected.fetches == []
    assert not selected.backends[0]._model_acquisition_store._records


def test_warning_helper_rejects_nonliteral_metadata_without_logging(monkeypatch):
    from amlas.tui.chat import tool_binding_backend

    calls = []
    monkeypatch.setattr(tool_binding_backend.logger, "warning", calls.append)
    tool_binding_backend._warn_input_hold(MARKER)
    tool_binding_backend._warn_input_hold({"phase": MARKER})
    assert calls == []


def test_warning_helper_preserves_process_control_propagation(monkeypatch):
    from amlas.tui.chat import tool_binding_backend

    def interrupted_warning(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(tool_binding_backend.logger, "warning", interrupted_warning)
    with pytest.raises(KeyboardInterrupt):
        tool_binding_backend._warn_input_hold(
            "AMLAS_INPUT_HOLD phase=research_before_tool reason=callback_refused"
        )
