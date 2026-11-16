"""Named neutral fresh-process fixture for normal tool bootstrap and lazy defaults."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cold_normal_orchestration_import_then_tool_binding_resolves_both_adapters():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/coding_cold_bootstrap_fixture.py"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AWS_", "AMLAS_", "AGENTICML_"))
    }
    environment.update(AWS_EC2_METADATA_DISABLED="true", PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = str(root / "src") + ":" + os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, str(fixture)], cwd=root, env=environment, capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COLD_NORMAL_CODING_BOOTSTRAP_PASS" in result.stdout
    assert "cannot import name 'fixed_generation_request'" not in result.stderr
