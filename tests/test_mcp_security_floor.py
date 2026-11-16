"""MCP1.x security floor and exact published fixed release integrity."""

import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
FIXED_VERSION = "1.30.0"
# PyPI-published artifacts for the first patched1.x release of CVE-2026-59951.
# pragma: allowlist nextline secret
FIXED_WHEEL_SHA256 = "666edb5009503e1047c9d60346a756f94b261f05cc2625f23d41c728ffc484d0"
# pragma: allowlist nextline secret
FIXED_SDIST_SHA256 = "445414625fce5c295faa505bb11bacece661ab6f4028d57c935db57820b7a3e4"


def test_manifest_excludes_vulnerable_mcp_and_retains_compatible_major() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirement = next(
        Requirement(item) for item in project["project"]["dependencies"] if Requirement(item).name == "mcp"
    )
    assert not requirement.specifier.contains("1.29.1"), "MCP security floor allows CVE-2026-59951"
    assert requirement.specifier.contains(FIXED_VERSION)
    assert not requirement.specifier.contains("2.0.0"), "security repair must preserve SDK1.x"


def test_lock_selects_exact_fixed_mcp_with_official_artifact_hashes() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    package = next(item for item in lock["package"] if item["name"] == "mcp")
    assert package["version"] == FIXED_VERSION, "lock retains vulnerable MCP"
    assert package["sdist"]["hash"] == "sha256:" + FIXED_SDIST_SHA256
    assert {wheel["hash"] for wheel in package["wheels"]} == {"sha256:" + FIXED_WHEEL_SHA256}


def test_running_sdk_environment_is_the_fixed_mcp_major() -> None:
    installed = Version(metadata.version("mcp"))
    assert installed >= Version(FIXED_VERSION), "tests ran in old vulnerable MCP environment"
    assert installed.major == 1, "security repair unexpectedly crossed SDK major"
