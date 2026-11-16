"""First projection hold metadata only; neutral Source ports, no runtime effects.

This reproduces the silent nonresearch callback seam, not the native cause.
The producer parameter preserves accepted-original custody semantics; this test
never creates or seeds an accepted handle, model, task or protected row.
"""

import logging
from types import SimpleNamespace

import pytest
from test_input_admission_wiring import ClientFixture, configure_guard
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.tools.model_weight_tools import acquire_model_weights
from amlas.tools.qwen_sft import get_qwen_sft
from amlas.tui.chat import tool_binding_backend as binding

pytestmark = pytest.mark.usefixtures("inert_runtime_setup")
LOGGER = "amlas.tui.chat.tool_binding_backend"
PREFIX = "AMLAS_INPUT_HOLD "
EXCLUDED = "EXCLUDED_NEUTRAL_TOOL_CONTENT_AND_REFERENCE"
EXPECTED = "AMLAS_INPUT_HOLD phase=tool_projection reason=callback_refused"


def records(caplog):
    return [
        record
        for record in caplog.records
        if record.name == LOGGER and record.getMessage().startswith(PREFIX)
    ]


@pytest.mark.parametrize(
    ("name", "producer", "tool"),
    [
        ("get_qwen_sft", False, get_qwen_sft),
        ("acquire_model_weights", True, acquire_model_weights),
    ],
)
def test_first_nonresearch_projection_refusal_has_only_closed_marker(
    monkeypatch, caplog, name, producer, tool
):
    client = ClientFixture(deny=lambda _text: False)
    configure_guard(monkeypatch, client)
    backend = binding.ToolBindingBackend(Settings())
    backend._tool_registry_cache = {name: tool}
    backend._input_cycle = "a" * 32
    backend._admit_input("Ordinary public summary.", "OWNER_PROMPT")
    before = list(backend._input_receipts)
    assert len(before) == 1 and before[0].outcome == "ALLOW"
    reads = len(client.calls)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    event = SimpleNamespace(
        tool_use={"toolUseId": EXCLUDED, "name": name},
        result={
            "toolUseId": EXCLUDED,
            "status": "success",
            "content": [{"json": {EXCLUDED: float("nan")}}],
        },
    )
    backend._guard_new_tool_projection(event, producer=producer)
    assert backend._input_hold is True and backend._input_quarantined is True
    assert backend._input_receipts == before and len(client.calls) == reads
    assert event.result == {
        "toolUseId": EXCLUDED,
        "status": "error",
        "content": [{"text": "Tool content is unavailable at the input admission boundary."}],
    }
    # A later valid callback cannot clear the first hold or trigger assessment.
    event.result = {
        "toolUseId": EXCLUDED,
        "status": "success",
        "content": [{"text": "Ordinary later summary."}],
    }
    backend._guard_new_tool_projection(event, producer=producer)
    assert backend._input_hold is True
    assert backend._input_receipts == before and len(client.calls) == reads
    observed = records(caplog)
    assert len(observed) == 1, "first_nonresearch_callback_hold_marker_missing"
    assert observed[0].getMessage() == EXPECTED
    assert observed[0].args == () and observed[0].exc_info is None
    assert EXCLUDED not in observed[0].getMessage()
    assert "nan" not in observed[0].getMessage().lower()
    assert "json" not in observed[0].getMessage().lower()
