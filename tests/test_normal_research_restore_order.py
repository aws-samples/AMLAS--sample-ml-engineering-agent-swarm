"""Structural restoration refusal performs no admission; declared Source-only DATA."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from amlas.control_plane import normal_run_checkpoint
from amlas.tui.chat import research_runtime_profile as source
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend


def test_invalid_late_research_link_refuses_before_any_input_admission(monkeypatch):
    plan = source._plan()
    prefix = [{"role": "user", "content": [{"text": "fresh declared request"}]}]
    for index, unit in enumerate(plan.units):
        identifier = "declared-use-" + str(index)
        prefix.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": identifier,
                            "name": "research_fetch_urls",
                            "input": {"urls": list(unit.urls), "max_pages": 3},
                        }
                    }
                ],
            }
        )
        prefix.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": identifier,
                            "status": "success",
                            "content": [{"json": {"pages": [{"text": "fresh declared source"}]}}],
                        }
                    }
                ],
            }
        )
    prefix[-1]["content"][0]["toolResult"]["status"] = "error"
    snapshot = {"data": {"messages": [*copy.deepcopy(prefix), {"role": "user", "content": []}]}}
    context = SimpleNamespace(
        _completion_receipt=("declared-token", json.dumps({"sdk_snapshot": snapshot}), "declared-task")
    )
    monkeypatch.setattr(normal_run_checkpoint, "current_normal_run", lambda: context)
    admissions = []
    state = {"phase": "UNUSED", "pending": False, "units": {unit.unit_id: unit for unit in plan.units}}
    backend = SimpleNamespace(
        _research_started=False,
        _normal_original_result={
            "job_name": "declared-original",
            "job_arn": "declared-arn",
            "trained": True,
            "full_training_complete": True,
        },
        _qualify_research_binding=lambda **kwargs: state,
        _planning_result_payload=ToolBindingBackend._planning_result_payload,
        _admit_input=lambda *args, **kwargs: admissions.append(True),
    )
    owner = SimpleNamespace(tool_backend=backend)
    task = {"causal_prefix": prefix, "native_job_name": "declared-original", "native_job_arn": "declared-arn"}
    with pytest.raises(ValueError):
        source._restore_runtime_research_profile(owner, context, task, "declared-token", snapshot)
    if admissions:
        raise AssertionError("source_order_refused_after_admission")
    assert state["phase"] == "UNUSED"
