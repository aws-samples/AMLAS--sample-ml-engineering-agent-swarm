"""Decoded batch accounting supplements the Parquet footer estimate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_parquet_decoded_batch_budget_is_checked_before_combining(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as parquet

    from amlas.tools import pathref

    class Metadata:
        num_row_groups = 1

        def row_group(self, index: int) -> Any:
            return type("Group", (), {"total_byte_size": 100})()

    class File:
        metadata = Metadata()

        def read(self) -> Any:
            pytest.fail("unbounded full Parquet materialization before decoded batch guard")

        def iter_batches(self, *, batch_size: int) -> Any:
            assert batch_size <= 1024
            yield type("Batch", (), {"nbytes": 512 * 1024 * 1024 + 1})()

    monkeypatch.setattr(parquet, "ParquetFile", lambda *args, **kwargs: File())
    path = tmp_path / "encoded.parquet"
    path.write_bytes(b"synthetic")
    with pytest.raises(pathref.PathRefError, match="decoded"):
        pathref.read_dataframe(pathref.resolve_ref(str(path), sandbox_local=False))
