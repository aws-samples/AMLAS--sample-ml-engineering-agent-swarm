"""Staged-label verification retains safe loading and logical CSV row counts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amlas.cloud.staging import verify_staged_test_split


class UnsafeLabel:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "unsafe pickle executed")


def test_staged_verification_never_executes_pickle_payload(tmp_path: Path) -> None:
    marker = tmp_path / "unexpected-execution"
    labels = tmp_path / "labels.npy"
    table = tmp_path / "test.csv"
    np.save(labels, np.array([UnsafeLabel(marker)], dtype=object))
    pd.DataFrame({"prompt": ["repair code"]}).to_csv(table, index=False)

    outcome = verify_staged_test_split(str(table), str(labels))

    assert not marker.exists(), "Verification must not execute labels from an artifact"
    assert outcome["ok"] is None


def test_multiline_code_counts_csv_records_instead_of_physical_lines(tmp_path: Path) -> None:
    labels = tmp_path / "labels.npy"
    table = tmp_path / "test.csv"
    pd.DataFrame({"code": ["def solve():\n    return 1", "print('ok')"]}).to_csv(table, index=False)
    np.save(labels, np.array([1, 0]))

    outcome = verify_staged_test_split(str(table), str(labels))

    assert outcome["ok"] is True
    assert outcome["n_test_rows"] == outcome["n_y_true"] == 2


def test_safe_string_labels_remain_supported(tmp_path: Path) -> None:
    from amlas.security.model_artifacts import dump_numpy_data

    labels = tmp_path / "labels.npy"
    table = tmp_path / "test.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(table, index=False)
    dump_numpy_data(np.array(["negative", "positive"], dtype=object), labels)
    assert verify_staged_test_split(str(table), str(labels))["ok"] is True
