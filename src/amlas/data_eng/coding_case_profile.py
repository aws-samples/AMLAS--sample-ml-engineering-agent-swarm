"""Pinned case-source identity, never native readiness or authority."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodingCaseSource:
    repo_id: str
    revision: str
    filename: str
    sha256: str
    byte_count: int
    format: str
    license: str
    card_sha256: str


PROFILE_ID = "openai.humaneval.164.hub.parquet.v1"
PROFILE_VERSION = "1"
SOURCE_REPO = "openai/openai_humaneval"
SOURCE_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"  # pragma: allowlist secret
SOURCE_FILENAME = "openai_humaneval/test-00000-of-00001.parquet"
SOURCE_SHA256 = "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6"  # pragma: allowlist secret
SOURCE_BYTES = 83920
CARD_SHA256 = "f81cb59f7dae73d1ee1069d7de3ba1c0f9d2bd80f4df6d6990d12bf488e5b5ef"  # pragma: allowlist secret
ORIGINAL_REPO = "openai/human-eval"
ORIGINAL_REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"  # pragma: allowlist secret
CASE_IDS = tuple(f"HumanEval/{index}" for index in range(164))
SOURCE_FIELDS = frozenset({"task_id", "prompt", "test", "entry_point", "canonical_solution"})
MAX_FIELD_BYTES = 8192
MAX_DECODED_BYTES = 1024 * 1024
MAX_BUNDLE_BYTES = 2 * 1024 * 1024


def source_metadata(source: CodingCaseSource) -> dict:
    """Validate descriptive pins only; caller authenticates original bytes separately."""
    from dataclasses import asdict

    if type(source) is not CodingCaseSource:
        raise ValueError("exact coding-case source descriptor required")
    expected = {
        "repo_id": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "filename": SOURCE_FILENAME,
        "sha256": SOURCE_SHA256,
        "byte_count": SOURCE_BYTES,
        "format": "parquet",
        "license": "MIT",
        "card_sha256": CARD_SHA256,
    }
    values = asdict(source)
    if (
        any(type(value) is not (int if key == "byte_count" else str) for key, value in values.items())
        or values != expected
    ):
        raise ValueError("coding-case source pins/license/type differ")
    return values
