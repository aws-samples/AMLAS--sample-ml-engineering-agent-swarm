"""Integrity hash tokens do not hide adjacent private release coordinates."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hash_governance", ROOT / "scripts/verify_repository_governance.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def account():
    return "123456" + "789012"


def test_complete_requirement_sha256_tokens_are_integrity_context():
    value = account()
    token = "a" * 26 + value + "b" * 26
    text = "package==1.2.3 \\\n    --hash=sha256:" + token + "\n"
    assert not MODULE.scan_text("requirements.lock", text)


@pytest.mark.parametrize(
    "text",
    [
        "--hash=sha256:{token}\n# account: {account}",
        "--hash=sha256:{token}\nACCOUNT={account}",
        "--hash=sha256:{token}9",
        "--hash=sha256:{token}f",
        "url=https://example.test/{account}",
    ],
)
def test_hash_context_never_exempts_other_numbers_or_malformed_tokens(text):
    value = account()
    token = "a" * 26 + value + "b" * 26
    findings = MODULE.scan_text("requirements.lock", text.format(token=token, account=value))
    assert "unknown-account-id" in {item.code for item in findings}
