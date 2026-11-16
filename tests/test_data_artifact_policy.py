"""Shared artifact policy rejects executable arrays and unreviewed object types."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from amlas.security import model_artifacts


@pytest.mark.parametrize("values", [[1, 2], ["healthy", "fault"], [True, False]])
def test_data_only_label_arrays_load_without_pickle(tmp_path: Path, values: list) -> None:
    path = tmp_path / "labels.npy"
    np.save(path, np.asarray(values))
    result = model_artifacts.load_numpy_data(path)
    assert result.tolist() == values


@pytest.mark.parametrize("values", [["healthy", "fault"], [1, 2]])
def test_object_arrays_are_rejected_even_when_their_values_look_harmless(
    tmp_path: Path, values: list,
) -> None:
    path = tmp_path / "legacy.npy"
    np.save(path, np.asarray(values, dtype=object))
    with pytest.raises(model_artifacts.UntrustedArtifactError, match=r"object|pickle"):
        model_artifacts.load_numpy_data(path)


def test_declared_allocation_is_checked_before_numpy_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "<f8", "fortran_order": False, "shape": (1000000000000,)},
        )
    invoked = []
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: invoked.append(True))
    with pytest.raises(model_artifacts.UntrustedArtifactError, match=r"limit|size|elements"):
        model_artifacts.load_numpy_data(path)
    assert not invoked


def test_model_namespace_is_not_automatic_type_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.skops"
    path.write_bytes(b"test fixture; introspection replaced")
    monkeypatch.setattr(
        model_artifacts.skops_io, "get_untrusted_types",
        lambda **kwargs: ["numpy.future.UnreviewedLoader"],
    )
    called = []
    monkeypatch.setattr(model_artifacts.skops_io, "load", lambda *args, **kwargs: called.append(True))
    with pytest.raises(model_artifacts.UntrustedArtifactError, match="unreviewed"):
        model_artifacts.load_model_artifact(path, trusted_root=tmp_path)
    assert not called


def test_shared_prediction_loader_refuses_legacy_object_pickle(tmp_path: Path) -> None:
    from amlas.cloud.sandbox_staging import _load_npy_secure

    path = tmp_path / "legacy-labels.npy"
    np.save(path, np.array(["negative", "positive"], dtype=object))
    with pytest.raises(model_artifacts.UntrustedArtifactError, match=r"object|pickle"):
        _load_npy_secure(path)


def test_writer_normalizes_string_labels_into_data_only_unicode(tmp_path: Path) -> None:
    path = tmp_path / "labels.npy"
    model_artifacts.dump_numpy_data(np.array(["negative", "positive"], dtype=object), path)
    loaded = np.load(path, allow_pickle=False)
    assert loaded.dtype.kind == "U"
    assert loaded.tolist() == ["negative", "positive"]


@pytest.mark.parametrize("input_kind", ["object_array", "list"])
def test_writer_refuses_unicode_expansion_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_kind: str,
) -> None:
    # Only about one MB of source strings would expand to 800 MB in fixed-width
    # Unicode storage. The spy prevents the test from making that allocation.
    items = ["x" * 1000000] + [""] * 199
    values = np.array(items, dtype=object) if input_kind == "object_array" else items
    original = np.asarray

    def observe(value, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if isinstance(value, list) and dtype is not object:
            raise AssertionError("oversized Unicode normalization attempted")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", observe)
    with pytest.raises(model_artifacts.UntrustedArtifactError, match=r"limit|size|exceeds"):
        model_artifacts.dump_numpy_data(values, tmp_path / "labels.npy")
    assert not list(tmp_path.iterdir())
