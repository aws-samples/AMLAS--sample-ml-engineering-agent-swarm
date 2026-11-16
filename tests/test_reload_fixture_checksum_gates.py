"""Reviewed synthetic integrity literals; no native or credential authority."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest
from test_repository_governance import governance

FIXTURE_PATH = "tests/test_qwen_sft_reload_resources.py"
FIXTURE_TEXT = (Path(__file__).parent / "test_qwen_sft_reload_resources.py").read_text()
MARKER = "  # pragma: allowlist secret"
FROZEN_NAMES = {"LEGACY", "LEGACY_DIGEST", "LEGACY_ONE", "TRAIN_REQUEST"}
FROZEN_VALUES = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in ast.parse(FIXTURE_TEXT).body
    if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id in FROZEN_NAMES
}
TRAIN_CHECKSUM = FROZEN_VALUES["TRAIN_REQUEST"]
EXPECTED_ANNOTATIONS = {
    line.removesuffix(MARKER).rstrip() + MARKER
    for line in FIXTURE_TEXT.splitlines()
    if re.search(r'"[a-f0-9]{64}"', line)
}


def annotation_errors(text):
    observed = {line for line in text.splitlines() if line.endswith(MARKER)}
    return [] if observed == EXPECTED_ANNOTATIONS else ["unreviewed checksum annotation"]


def test_verified_frozen_training_request_checksum_is_not_an_account():
    assert len(TRAIN_CHECKSUM) == 64
    assert re.findall(r"(?<!\d)\d{12}(?!\d)", TRAIN_CHECKSUM) == ["131525" + "060103"]
    assert governance.scan_text(FIXTURE_PATH, FIXTURE_TEXT) == []


def test_exact_verified_fixture_checksum_lines_require_annotations():
    assert len(EXPECTED_ANNOTATIONS) == 6
    assert annotation_errors(FIXTURE_TEXT) == []


@pytest.mark.parametrize("change", ["modified-hash", "extra-hash", "account-line"])
def test_changed_or_added_fixture_annotation_is_not_reviewed(change):
    reviewed = "\n".join(sorted(EXPECTED_ANNOTATIONS)) + "\n"
    if change == "modified-hash":
        changed = reviewed.replace(TRAIN_CHECKSUM, TRAIN_CHECKSUM[:-1] + "c", 1)
    elif change == "extra-hash":
        digest = hashlib.sha256(b"neutral unreviewed fixture metadata").hexdigest()
        changed = reviewed + 'EXTRA = "' + digest + '"' + MARKER + "\n"
    else:
        fragment = "131525" + "060103"
        changed = reviewed + 'ACCOUNT = "' + fragment + '"' + MARKER + "\n"
    assert annotation_errors(changed)


@pytest.mark.parametrize("context", ["bare", "arn", "ecr", "bucket", "domain", "adjacent"])
def test_fixture_checksum_exception_never_allows_account_coordinates(context):
    fragment = "131525" + "060103"
    texts = {
        "bare": 'ACCOUNT = "' + fragment + '"',
        "arn": 'ARN = "arn:aws:iam::' + fragment + ':role/fixture"',
        "ecr": 'IMAGE = "' + fragment + '.dkr.ecr.us-west-2.amazonaws.com/fixture"',
        "bucket": 'BUCKET = "fixture-' + fragment + '"',
        "domain": 'DOMAIN = "fixture-' + fragment + '.example.com"',
        "adjacent": 'CHECKSUM = "' + TRAIN_CHECKSUM + '" ACCOUNT = "' + fragment + '"',
    }
    findings = governance.scan_text(FIXTURE_PATH, texts[context])
    assert "unknown-account-id" in {item.code for item in findings}


@pytest.mark.parametrize("change", ["modified", "prefix", "suffix", "short", "upper", "unreviewed"])
def test_fixture_exception_requires_exact_complete_reviewed_checksum(change):
    fragment = "131525" + "060103"
    values = {
        "modified": TRAIN_CHECKSUM[:-1] + "c",
        "prefix": "a" + TRAIN_CHECKSUM,
        "suffix": TRAIN_CHECKSUM + "f",
        "short": TRAIN_CHECKSUM[:-1],
        "upper": TRAIN_CHECKSUM.upper(),
        "unreviewed": "a" * 26 + fragment + "b" * 26,
    }
    assert "unknown-account-id" in {
        item.code for item in governance.scan_text(FIXTURE_PATH, 'CHECKSUM = "' + values[change] + '"')
    }


def test_reviewed_checksum_cannot_hide_private_literal_and_strict_entropy():
    from detect_secrets.plugins.high_entropy_strings import HexHighEntropyString

    assert "private-literal" in {
        item.code for item in governance.scan_text(FIXTURE_PATH, TRAIN_CHECKSUM, (TRAIN_CHECKSUM,))
    }
    digest = hashlib.sha256(b"neutral unreviewed fixture metadata").hexdigest()
    assert HexHighEntropyString().analyze_line(FIXTURE_PATH, 'EXTRA = "' + digest + '"' + MARKER)
