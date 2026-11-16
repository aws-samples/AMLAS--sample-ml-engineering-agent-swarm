"""Fail-closed persistence for estimator artifacts.

Skops stores estimators without Python's executable object deserialization.  AMLAS
still treats every artifact as untrusted: the path must remain under an explicit
trusted root and every type Skops marks as unknown must belong to a reviewed ML
library before loading is attempted.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Final

from amlas.config._env_compat import env_with_compat


class _SkopsIO:
    """Import estimator persistence only when a model operation needs it."""

    def __getattr__(self, name: str) -> Any:
        import skops.io

        return getattr(skops.io, name)


skops_io = _SkopsIO()

_ARTIFACT_SUFFIX: Final = ".skops"
# Namespaces are not authority: an unexpected future library type must fail
# closed rather than be approved merely because its module starts with numpy.
_REVIEWED_EXTRA_TYPES: Final = frozenset({
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "xgboost.sklearn.XGBRegressor",
})


class UntrustedArtifactError(RuntimeError):
    """Raised when an estimator artifact fails a trust-boundary check."""


def load_numpy_data(
    artifact_path: str | Path,
    *,
    max_bytes: int = 512 * 1024 * 1024,
    max_elements: int = 10000000,
) -> Any:
    """Load a bounded data-only NumPy vector, never a Python object pickle.

    The header, dtype, declared allocation and exact payload length are checked
    before NumPy allocates/deserializes. Object/structured arrays must be
    converted to a data-only format by a separately isolated conversion path.
    """
    import numpy as np

    if max_bytes <= 0 or max_elements <= 0:
        raise ValueError("array limits must be positive")
    path = Path(artifact_path)
    if path.is_symlink() or not path.is_file():
        raise UntrustedArtifactError("array artifact must be a regular file")
    if path.stat().st_size > max_bytes:
        raise UntrustedArtifactError("array file exceeds size limit")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10000)
        else:
            raise UntrustedArtifactError("unsupported array header version")
        if dtype.hasobject or dtype.fields is not None or dtype.kind not in "buifSU":
            raise UntrustedArtifactError("object/pickle or unsupported array dtype is forbidden")
        if len(shape) != 1 or any(not isinstance(n, int) or n < 0 for n in shape):
            raise UntrustedArtifactError("array must be a one-dimensional data vector")
        elements = math.prod(shape)
        data_bytes = elements * dtype.itemsize
        if elements > max_elements or data_bytes > max_bytes:
            raise UntrustedArtifactError("declared array allocation exceeds size/elements limit")
        if path.stat().st_size != stream.tell() + data_bytes:
            raise UntrustedArtifactError("array payload size does not match its header")
        stream.seek(0)
        return np.load(stream, allow_pickle=False, max_header_size=10000)


def dump_numpy_data(values: Any, artifact_path: str | Path) -> Path:
    """Write data-only label/prediction vectors without object serialization."""
    import numpy as np

    array = values if isinstance(values, np.ndarray) else None
    items = None
    if array is not None:
        if array.ndim != 1 or array.size > 10000000:
            raise UntrustedArtifactError("array must be a bounded one-dimensional vector")
        if array.nbytes > 512 * 1024 * 1024:
            raise UntrustedArtifactError("array data exceeds size limit")
        if array.dtype.hasobject:
            items = array.tolist()
    elif isinstance(values, (list, tuple)):
        if len(values) > 10000000:
            raise UntrustedArtifactError("array exceeds element limit")
        items = values
    else:
        raise UntrustedArtifactError("array values must be a data vector or primitive sequence")
    if items is not None:
        if all(isinstance(item, (str, np.str_)) for item in items):
            width = max((len(item) for item in items), default=0)
            if len(items) * max(width, 1) * np.dtype("U1").itemsize > 512 * 1024 * 1024:
                raise UntrustedArtifactError("normalized Unicode array exceeds size limit")
            array = np.asarray(items, dtype=np.str_)
        elif all(isinstance(item, (bytes, np.bytes_)) for item in items):
            width = max((len(item) for item in items), default=0)
            if len(items) * max(width, 1) > 512 * 1024 * 1024:
                raise UntrustedArtifactError("normalized byte array exceeds size limit")
            array = np.asarray(items, dtype=np.bytes_)
        elif all(isinstance(item, (bool, int, float, np.bool_, np.integer, np.floating))
                 for item in items):
            array = np.asarray(items)
        else:
            raise UntrustedArtifactError("object values cannot be normalized into data-only arrays")
    if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.kind not in "buifSU":
        raise UntrustedArtifactError("unsupported array dtype")
    if array.nbytes > 512 * 1024 * 1024:
        raise UntrustedArtifactError("array data exceeds size limit")
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".array-", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _default_trusted_roots() -> list[Path]:
    """Resolve operator-configured artifact roots, with cwd as the dev default."""
    configured = env_with_compat("AMLAS_TRUSTED_MODEL_ROOT", "").strip()
    if configured:
        return [Path(item).resolve() for item in configured.split(os.pathsep) if item.strip()]
    return [Path.cwd().resolve()]


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves beneath ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_artifact_path(
    artifact_path: str | Path,
    *,
    trusted_root: str | Path | None,
) -> Path:
    """Validate extension, existence, and containment before parsing an artifact."""
    path = Path(artifact_path).resolve()
    roots = (
        [Path(trusted_root).resolve()]
        if trusted_root is not None
        else _default_trusted_roots()
    )

    if path.suffix != _ARTIFACT_SUFFIX:
        raise UntrustedArtifactError(f"refusing to load non-.skops artifact: {path}")
    if not path.is_file():
        raise UntrustedArtifactError(f"artifact does not exist or is not a file: {path}")
    if not any(_is_within(path, root) for root in roots):
        raise UntrustedArtifactError(
            f"artifact path {path} is outside the trusted root(s) "
            f"{[str(root) for root in roots]}"
        )
    return path


def dump_model_artifact(model: Any, artifact_path: str | Path) -> Path:
    """Persist ``model`` in Skops format and return the resolved artifact path."""
    path = Path(artifact_path)
    if path.suffix != _ARTIFACT_SUFFIX:
        raise UntrustedArtifactError(f"refusing to write non-.skops artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    skops_io.dump(model, path)
    return path.resolve()


def load_model_artifact(
    artifact_path: str | Path,
    *,
    trusted_root: str | Path | None = None,
) -> Any:
    """Load a reviewed Skops artifact after path and serialized-type validation.

    Skops has a built-in trusted set for common scikit-learn primitives.  Any
    additional type is surfaced by :func:`skops.io.get_untrusted_types`; AMLAS
    permits only reviewed numerical/ML library namespaces and rejects custom
    application types before calling :func:`skops.io.load`.
    """
    path = _validate_artifact_path(artifact_path, trusted_root=trusted_root)
    unknown_types = sorted(skops_io.get_untrusted_types(file=path))
    unreviewed = [
        type_name
        for type_name in unknown_types
        if type_name not in _REVIEWED_EXTRA_TYPES
    ]
    if unreviewed:
        raise UntrustedArtifactError(
            f"artifact contains unreviewed serialized types: {unreviewed}"
        )
    return skops_io.load(path, trusted=unknown_types)
