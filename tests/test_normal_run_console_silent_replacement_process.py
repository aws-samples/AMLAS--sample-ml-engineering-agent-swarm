"""A replacement worker in a new process writes no agent text to its stdout.

The tests of `test_normal_run_console_silent.py` run in one process. The place where a real run printed agent
text after a worker restore is the stdout of a replacement worker: a new process that restores the run through
`resume_prompt_run` and continues it. The cold fixture starts such a process (`child`), and its connected
fixture replaces the handler after every build, so the cold tests cannot see what the builder chose. This test
sets that fixture's handler switch, which a child process inherits through its environment:

- `sdk-printer`: the replacement's agent holds the SDK's console printer. Its stdout shows what the printer
  writes on this scripted run and proves the trigger.
- `built`: the replacement's agent keeps the handler the real builder chose. Its stdout must hold the fixture's
  answer line and nothing else.

The test fails on the earlier code, where the builder leaves the SDK printer in place under a prompt run. Component
evidence only: a child process of the cold fixture over neutral native ports, never the log of a deployed runtime.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from types import SimpleNamespace

import pytest
import test_normal_sft_cold_continuation as cold
from normal_sft_cold_fixture import FINAL_TEXT, close, cloud_data, original_precommit, setup
from test_normal_run_console_silent import RUNTIME_STARTUP_LINE
from test_normal_run_sft_checkpoint import wire_decode
from test_runtime_connected_warm_model import AGENT_HANDLER_SWITCH, close_original_worker

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def replacement(monkeypatch, data, directory, handler):
    """Run the cold fixture's replacement worker with the handler switch; return its console text and its answer."""
    finished = []
    actual_run = cold.subprocess.run

    def recording_run(*args, **kwargs):
        result = actual_run(*args, **kwargs)
        finished.append(result)
        return result

    with monkeypatch.context() as patch:
        patch.setenv(AGENT_HANDLER_SWITCH, handler)
        # Only the cold module's own process start is recorded: `child` looks `subprocess.run` up in that module.
        patch.setattr(cold, "subprocess", SimpleNamespace(run=recording_run))
        try:
            cold.child(data, directory)
        except ValueError:
            # `child` reads its answer from the last stdout line; console text in front of the answer breaks
            # that read. The answer is read again below, after the console text.
            pass
    processes = [result for result in finished if "--cold-component" in result.args]
    assert len(processes) == 1 and processes[0].returncode == 0, finished
    out = processes[0].stdout.decode()
    assert out.endswith("\n"), "the replacement process did not print the fixture's answer line"
    body, separator, last = out[:-1].rpartition("\n")
    start = last.find('{"')
    assert start >= 0, "the last stdout line of the replacement process holds no answer"
    answer = wire_decode(json.loads(last[start:]))
    assert answer["pid"] != os.getpid(), "the replacement did not run in a new process"
    terminal = next(event for event in reversed(answer["events"]) if event.get("event") == "run_completed")
    return SimpleNamespace(
        console=body + separator + last[:start],
        err=processes[0].stderr.decode(),
        answer=answer,
        terminal=terminal,
    )


def test_red_replacement_worker_in_a_new_process_writes_no_agent_text_to_stdout(monkeypatch, tmp_path, capfd):
    # The original worker in this process keeps the handler of the real builder as well.
    monkeypatch.setenv(AGENT_HANDLER_SWITCH, "built")

    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            await close_original_worker(state.work, state.reader)
            data = cloud_data(state, completed=True)
            # Baseline: the same transport DATA with the SDK's console printer in the replacement process.
            # It proves the trigger and gives the run to compare.
            base = replacement(monkeypatch, data, tmp_path / "printer-replacement", "sdk-printer")
            subject = replacement(monkeypatch, data, tmp_path / "built-replacement", "built")
            return state.run_id, base, subject
        finally:
            await close(state)

    run_id, base, subject = asyncio.run(scenario())
    original_out = capfd.readouterr().out

    tool_lines = re.findall(r"^Tool #(\d+): (\w+)$", base.console, re.MULTILINE)
    assert tool_lines and FINAL_TEXT in base.console, (
        f"TRIGGER ABSENT: the SDK printer printed no tool line or no final text: {base.console!r}"
    )
    assert [int(number) for number, _ in tool_lines] == list(range(1, len(tool_lines) + 1))

    # Both replacements continued the same scripted run to its final report.
    for run in (base, subject):
        assert run.terminal["run_id"] == run_id and run.terminal["status"] == "response_received"
        assert run.terminal["worker_finished"] is True
        assert FINAL_TEXT in run.terminal["response"]["message"]
        assert run.answer["loaded"] and run.answer["invoked"]
    assert subject.answer["model_calls"] == base.answer["model_calls"] > 1
    assert subject.answer["marked_results"] == base.answer["marked_results"]
    assert len(subject.answer["new_training"]) == len(base.answer["new_training"]) == 2
    assert len(subject.answer["new_processing"]) == len(base.answer["new_processing"]) == 1
    assert subject.terminal["response"]["message"] == base.terminal["response"]["message"]

    # The replacement with the handler of the real builder prints its answer line and nothing in front of it.
    assert "Tool #" not in subject.console, (
        f"a console tool line reached the stdout of the replacement process: {subject.console!r}"
    )
    assert FINAL_TEXT not in subject.console, (
        f"agent text reached the stdout of the replacement process: {subject.console!r}"
    )
    assert subject.console == "", f"the replacement process wrote to stdout: {subject.console!r}"
    # The printer writes to stdout only: the product's log lines on stderr hold no agent text in either run.
    for run in (base, subject):
        assert "Tool #" not in run.err and FINAL_TEXT not in run.err
    # The original worker in this process: only the entry point's own startup line.
    assert "Tool #" not in original_out, (
        f"a console tool line reached the original worker's stdout: {original_out!r}"
    )
    assert original_out == RUNTIME_STARTUP_LINE, f"the original worker wrote to stdout: {original_out!r}"
