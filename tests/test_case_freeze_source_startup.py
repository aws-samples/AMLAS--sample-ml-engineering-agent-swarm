"""The fixed case worker denies source drift before any input observation."""

from __future__ import annotations

import sys

import pytest

from amlas.sagemaker_entrypoints import coding_case_freeze as worker


def test_case_source_mismatch_refuses_before_input_open(monkeypatch):
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "b" * 40)
    monkeypatch.setattr(
        sys,
        "argv",
        ["worker", "--binding-sha256", "a" * 64, "--binding-byte-count", "128", "--source-commit", "c" * 40],
    )
    monkeypatch.setattr(worker, "_input_directory", lambda: pytest.fail("input observed"))
    assert worker.main() == 78
