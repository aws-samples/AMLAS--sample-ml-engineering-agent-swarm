"""Agent-bound dataset tools cannot open raw artifacts in the caller process."""

from pathlib import Path

import pandas as pd
import pytest

from amlas.tools import data_tools


@pytest.mark.parametrize("name", ["load_dataset", "describe_dataset", "split_dataset"])
def test_public_dataset_tool_denies_before_caller_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    path = tmp_path / "untrusted.parquet"
    path.write_bytes(b"untrusted-format fixture; parser replaced")
    opened = []

    def parse(ref):
        opened.append(ref)
        return pd.DataFrame({"label": [0, 1, 0, 1]})

    monkeypatch.setattr(data_tools, "read_dataframe", parse)
    args = {"path": str(path)}
    if name == "split_dataset":
        args.update(target_column="label", test_size=0.5)
    result = getattr(data_tools, name)(**args)
    assert opened == [], "raw dataset parser executed in the agent/client process"
    assert result["success"] is False
    assert result["status"] == "requires_isolated_inspection"
    assert result["next_tool"] == "inspect_acquired_dataset"


def test_local_parser_helpers_are_not_bound_to_the_agent() -> None:
    from amlas.tui.chat.agent_tools import get_safe_llm_tools

    names = {getattr(item, "tool_name", "") for item in get_safe_llm_tools()}
    assert not {"load_dataset_local", "describe_dataset_local", "split_dataset_local"} & names
