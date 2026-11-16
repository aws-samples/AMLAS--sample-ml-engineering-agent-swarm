"""Process-wide records of verified S3 object versions must not outlive a test.

`qwen_sft._PROCESS_PINS` and `model_weight_acquisition._PROCESS_PROOFS` remember, for the life of a process, which exact
object versions were hashed in full. A test that verifies a version must not make the next test's corrupted copy of the
same version look verified, so both are emptied around every test. Modules that were never imported are left alone.
"""

from __future__ import annotations

import sys

import pytest


def _reset_process_records() -> None:
    for module, name in (
        ("amlas.backends.adapters.qwen_sft", "_PROCESS_PINS"),
        ("amlas.backends.adapters.model_weight_acquisition", "_PROCESS_PROOFS"),
    ):
        loaded = sys.modules.get(module)
        records = getattr(loaded, name, None)
        if records is not None:
            records.reset()


@pytest.fixture(autouse=True)
def _fresh_process_records():
    _reset_process_records()
    yield
    _reset_process_records()
