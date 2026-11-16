"""Neutral command/source contracts; no native launcher or confinement executes."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from amlas.evaluation.coding_pair_runner import BUDGETS, oracle_child_command

ROOT = Path(__file__).resolve().parents[1]


def test_oracle_command_is_fixed_no_argument_unprivileged_launcher(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/neutral/parent-python")
    argv, environment = oracle_child_command()
    assert argv == ["/usr/local/bin/amlas-oracle-init"]
    assert environment == {"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    assert BUDGETS["case_wall_seconds"] == 10 and BUDGETS["cleanup_seconds"] == 30


def test_oracle_image_builds_static_fixed_binary_and_only_three_isolated_sources():
    recipe = (ROOT / "Dockerfile.coding-pair").read_text()
    assert "rust:1.90.0-alpine3.22@sha256:" in recipe
    assert "--features oracle-child-v1 --bin amlas-oracle-init" in recipe
    assert "x86_64-unknown-linux-musl" in recipe and "--locked --offline" in recipe
    assert "/usr/local/bin/amlas-oracle-init" in recipe
    assert "/app/oracle/coding_oracle_child.py" in recipe
    assert "/app/oracle/coding_oracle.py" in recipe
    assert "/app/oracle/coding_oracle_catalog.py" in recipe
    manifest = tomllib.loads((ROOT / "processing_bootstrap/Cargo.toml").read_text())
    binary = next(value for value in manifest["bin"] if value["name"] == "amlas-oracle-init")
    assert binary["required-features"] == ["oracle-child-v1"]


def test_fixed_rust_prctl_call_shapes_use_machine_width_complete_arguments():
    import re

    source = (ROOT / "processing_bootstrap/src/oracle_boundary.rs").read_text()
    assert re.search(r"prctl\(\s*39\s*,\s*0usize\s*,\s*0usize\s*,\s*0usize\s*,\s*0usize\s*,?\s*\)", source)
    assert re.search(r"prctl\(\s*21\s*,\s*0usize\s*,\s*0usize\s*,\s*0usize\s*,\s*0usize\s*,?\s*\)", source)
    assert re.search(
        r"prctl\(\s*22\s*,\s*2usize\s*,\s*&program as \*const Program\s*,\s*0usize\s*,\s*0usize\s*,?\s*\)",
        source,
    )


def test_fixed_exec_has_explicit_no_bytecache_and_unbuffered_flags():
    source = (ROOT / "processing_bootstrap/src/oracle_boundary.rs").read_text()
    assert '.args(["-I", "-S", "-B", "-u", CHILD])' in source


def test_original_official_case_import_leaves_extend_only_fixed_read_catalog():
    import hashlib
    import json
    import re

    source = (ROOT / "processing_bootstrap/src/oracle_boundary.rs").read_text()
    match = re.search(r"pub const RUNTIME_FILES[^=]*=\s*&\[(.*?)\];", source, re.S)
    assert match is not None
    paths = re.findall(r"\"([^\"]+)\"", match.group(1))
    official_import_leaves = {
        "/usr/local/lib/python3.11/typing.py",
        "/usr/local/lib/python3.11/copy.py",
        "/usr/local/lib/python3.11/weakref.py",
        "/usr/local/lib/python3.11/string.py",
    }
    assert official_import_leaves <= set(paths)
    retained = [path for path in paths if path not in official_import_leaves]
    expected_original = (
        "20e476a418cb07792c06fbae26d98d9961d5cdeea05e5427abd663bb3373f9be"  # pragma: allowlist secret
    )
    assert (
        hashlib.sha256(json.dumps(retained, separators=(",", ":")).encode()).hexdigest() == expected_original
    )
    assert len(paths) == len(set(paths)) == 78
    assert "/usr/local/lib/python3.11/lib-dynload/_typing.cpython-311-x86_64-linux-gnu.so" not in paths
