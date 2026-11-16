"""Agent parser boundaries and explicit local/offline dataset helpers.

Agent-bound path tools cannot parse raw artifacts in the caller process.
Inspection uses an opaque acquired-artifact reference and managed compute.
Separately named local helpers preserve operator-controlled offline operations;
they are never registered as agent tools or used as an automatic fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from strands import tool

from ..errors import tool_error_boundary
from .pathref import PathRefError, read_dataframe, resolve_ref


def _json_preview(value: Any, *, depth: int = 0, state: dict[str, Any]) -> Any:
    """Produce a bounded JSON-safe preview, retaining nested message structure."""
    import math

    if state["items"] >= 500 or depth >= 8:
        state["truncated"] = True
        return "<preview truncated>"
    state["items"] += 1
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        remaining = max(0, 12000 - state.setdefault("characters", 0))
        limit = min(2000, remaining)
        if len(value) > limit:
            state["truncated"] = True
        state["characters"] += min(limit, len(value))
        return value[:limit]
    if isinstance(value, dict):
        if len(value) > 30:
            state["truncated"] = True
        result = {}
        for key, item in list(value.items())[:30]:
            preview_key = str(key)[:200]
            state["characters"] = state.setdefault("characters", 0) + len(preview_key)
            result[preview_key] = _json_preview(item, depth=depth + 1, state=state)
        return result
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        values = (
            value[:20].tolist()
            if hasattr(value, "tolist") and hasattr(value, "__len__")
            else value.tolist()
            if hasattr(value, "tolist")
            else value
        )
        if hasattr(value, "__len__") and len(value) > 20:
            state["truncated"] = True
        if not isinstance(values, (list, tuple)):
            return _json_preview(values, depth=depth + 1, state=state)
        if len(values) > 20:
            state["truncated"] = True
        return [_json_preview(item, depth=depth + 1, state=state) for item in values[:20]]
    if hasattr(value, "isoformat"):
        return str(value.isoformat())[:200]
    state["truncated"] = True
    return str(value)[:200]


@tool_error_boundary(source="tools.data_tools")
def load_dataset_local(path: str) -> dict[str, Any]:
    """Load a supported dataset and return a bounded JSON-safe preview.

    Accepts local paths, ``s3://`` URIs, and ``http(s)://`` URLs (remote reads ride
    the resolved AWS profile for S3 and an anonymous, SSRF-guarded GET for HTTP).

    Args:
        path: Path or URI to CSV, Parquet, JSON or JSONL data.
    """
    try:
        # Explicit user-provided dataset path (tier-1 ``/data`` action): opt out of
        # the agent fs sandbox so a legit local dataset OUTSIDE the workspace roots
        # (byte-identical to the pre-Feature-1 ``pd.read_csv(Path(path))`` contract)
        # still loads. Remote (S3/HTTP) gating + SSRF guard are UNAFFECTED.
        ref = resolve_ref(path, sandbox_local=False)
        df = read_dataframe(ref)
    except PathRefError as e:
        return {"success": False, "error": str(e)}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to read dataset: {e}"}

    preview_state: dict[str, Any] = {"items": 0, "truncated": False}
    sample = _json_preview(df.head(3).to_dict(orient="records"), state=preview_state)
    return {
        "success": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        # intentional schema preview (3 rows) — do not pass PII-bearing files
        # without redaction first (see pii_tools.redact_pii).
        "sample_rows": sample,
        "sample_truncated": preview_state["truncated"],
    }


def split_dataset_local(
    path: str,
    target_column: str,
    test_size: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Split a dataset into train/test sets with optional stratification.

    The source ``path`` accepts local paths, ``s3://`` URIs, and ``http(s)://`` URLs
    (the read rides ``pathref``); ``output_dir`` WRITES remain local-only.

    Args:
        path: Path or URI to CSV, Parquet, JSON or JSONL data.
        target_column: Name of the target column for stratification.
        test_size: Fraction of data to use as test set (0.0-1.0).
        stratify: Whether to stratify by target column.
        seed: Random seed for reproducibility.
        output_dir: Optional LOCAL directory to save train/test CSVs.
    """
    from sklearn.model_selection import train_test_split

    try:
        # Explicit user-provided dataset path: opt out of the agent fs sandbox
        # (see ``load_dataset``). ``output_dir`` writes below stay local-only.
        ref = resolve_ref(path, sandbox_local=False)
        df = read_dataframe(ref)
    except PathRefError as e:
        return {"success": False, "error": str(e)}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to read dataset: {e}"}

    if target_column not in df.columns:
        return {
            "success": False,
            "error": f"Target column '{target_column}' not in dataset. Available: {list(df.columns)}",
        }

    stratify_col = df[target_column] if stratify else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify_col)

    # Compute target ratios
    train_ratio = (
        train_df[target_column].mean()
        if train_df[target_column].dtype != object
        else (train_df[target_column] == train_df[target_column].mode().iloc[0]).mean()
    )
    test_ratio = (
        test_df[target_column].mean()
        if test_df[target_column].dtype != object
        else (test_df[target_column] == test_df[target_column].mode().iloc[0]).mean()
    )

    # Handle boolean target
    if train_df[target_column].dtype == bool or set(train_df[target_column].dropna().unique()) <= {
        True,
        False,
        "True",
        "False",
    }:
        train_ratio = (train_df[target_column].astype(str) == "True").mean()
        test_ratio = (test_df[target_column].astype(str) == "True").mean()

    result: dict[str, Any] = {
        "success": True,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_target_ratio": float(train_ratio),
        "test_target_ratio": float(test_ratio),
    }

    # Save to disk if output_dir specified
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_path = out / "train_split.csv"
        test_path = out / "test_split.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        result["train_path"] = str(train_path)
        result["test_path"] = str(test_path)

    return result


def describe_dataset_local(path: str, target_column: str | None = None) -> dict[str, Any]:
    """Return summary statistics for a dataset: shape, dtypes, missing values, distributions.

    Accepts local paths, ``s3://`` URIs, and ``http(s)://`` URLs (remote reads ride
    the resolved AWS profile for S3 and an anonymous, SSRF-guarded GET for HTTP).

    Args:
        path: Path or URI to CSV, Parquet, JSON or JSONL data.
        target_column: Optional target column for class distribution reporting.
    """
    try:
        # Explicit user-provided dataset path: opt out of the agent fs sandbox
        # (see ``load_dataset``). Remote (S3/HTTP) gating + SSRF guard unaffected.
        ref = resolve_ref(path, sandbox_local=False)
        df = read_dataframe(ref)
    except PathRefError as e:
        return {"success": False, "error": str(e)}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to read dataset: {e}"}

    # Missing value counts
    missing_counts = df.isnull().sum().to_dict()

    # Numeric statistics
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_stats = {}
    for col in numeric_cols:
        numeric_stats[col] = {
            "mean": float(df[col].mean()) if not df[col].isna().all() else None,
            "std": float(df[col].std()) if not df[col].isna().all() else None,
            "min": float(df[col].min()) if not df[col].isna().all() else None,
            "max": float(df[col].max()) if not df[col].isna().all() else None,
            "missing": int(df[col].isna().sum()),
        }

    # Categorical statistics
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    categorical_stats = {}
    for col in cat_cols:
        # Coding records contain lists/dicts/ndarrays. They are structure, not
        # hashable categories; do not flatten messages or stringify whole traces.
        values = df[col].dropna().head(100)
        nested = any(isinstance(value, (list, dict, tuple)) or hasattr(value, "tolist") for value in values)
        if nested:
            categorical_stats[col] = {
                "nested": True,
                "n_unique": None,
                "top_values": {},
                "missing": int(df[col].isna().sum()),
            }
            continue
        value_counts = df[col].value_counts().head(10).to_dict()
        categorical_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": {str(k)[:2000]: int(v) for k, v in value_counts.items()},
            "missing": int(df[col].isna().sum()),
        }

    result: dict[str, Any] = {
        "success": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {k: int(v) for k, v in missing_counts.items()},
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
    }

    # Target distribution
    if (
        target_column
        and target_column in df.columns
        and not categorical_stats.get(target_column, {}).get("nested")
    ):
        target_dist = df[target_column].value_counts().head(100).to_dict()
        result["target_distribution"] = {str(k)[:2000]: int(v) for k, v in target_dist.items()}

    return _json_preview(result, state={"items": 0, "truncated": False})


def _isolated_inspection_required() -> dict[str, Any]:
    return {
        "success": False,
        "status": "requires_isolated_inspection",
        "next_tool": "inspect_acquired_dataset",
        "error": (
            "Use the acquisition reference with inspect_acquired_dataset, then "
            "get_dataset_inspection to obtain a verified managed report."
        ),
        "prepared": False,
        "trained": False,
    }


@tool
def load_dataset(path: str) -> dict[str, Any]:
    """Require managed inspection of a dataset instead of opening raw bytes here.

    Args:
        path: Display path or URI of CSV, Parquet, JSON or JSONL data.
    """
    return _isolated_inspection_required()


@tool
def describe_dataset(path: str, target_column: str | None = None) -> dict[str, Any]:
    """Require a verified managed dataset inspection report.

    Args:
        path: Display path or URI of CSV, Parquet, JSON or JSONL data.
        target_column: Optional target column to request from the inspection tool.
    """
    return _isolated_inspection_required()


@tool
def split_dataset(
    path: str,
    target_column: str,
    test_size: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Require managed preparation; this tool never opens or splits raw bytes.

    Args:
        path: Display path or URI of the source artifact.
        target_column: Desired target column.
        test_size: Desired test fraction.
        stratify: Desired stratification.
        seed: Desired split seed.
        output_dir: Legacy display-only output option.
    """
    return _isolated_inspection_required()
