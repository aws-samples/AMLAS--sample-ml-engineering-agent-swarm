from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    assert match is not None, f"unparseable version: {value}"
    return tuple(int(part) for part in match.group(1).split("."))


def test_repository_authority_policy_is_explicit() -> None:
    policy = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    required_statements = (
        "GitHub `main` is the authoritative public release tree.",
        "GitLab is the development system of record for branches and review.",
        "GitLab `main` must match the exact human-approved GitHub `main` tree.",
        "Preserve every other GitLab branch, tag, and historical commit.",
        "No change reaches GitHub until a human approves the sanitized public snapshot.",
    )
    for statement in required_statements:
        assert statement in policy


def _policy_sentences() -> str:
    """Return the root rule text with line wrapping removed so whole sentences can be pinned."""
    return " ".join((ROOT / "AGENTS.md").read_text(encoding="utf-8").split())


def test_release_form_adds_one_commit_by_a_plain_push() -> None:
    policy = _policy_sentences()
    required_statements = (
        "The tips of the two `main` branches have different commit IDs because GitLab retains "
        "development history while each GitHub release is one commit.",
        "After explicit approval, publish the approved tree to GitHub `main` as one new commit on top "
        "of the current GitHub `main`, using a plain push that is never forced.",
        "The new release commit must have the approved release tree.",
        "The first GitHub release was a root commit, and every later GitHub release adds one commit.",
        "Undo a wrong release with a revert commit.",
        "GitHub `main` history is never rewritten or force-pushed, and the first GitHub release is "
        "the root commit of GitHub `main`.",
        "Each later GitHub release adds one commit above all commits already on GitHub `main`.",
    )
    kept_statements = (
        "Present the exact candidate tree hash and material scan results for human approval.",
        "Never publish to GitHub first and sanitize afterward.",
        "Human approval applies to the content the approver saw. A later change that only merges a "
        "teammate's commit without a conflict, or only does what the maintainers' standing release "
        "rules require (for example a version or a date line), needs no new approval when the "
        "sanitization run and the required checks pass on the new tree. Any other content change "
        "needs a new approval.",
        "Treat every GitHub write as public disclosure, even when repository visibility is "
        "temporarily private.",
    )
    retired_phrases = (
        "fresh-root",
        "force-with-lease",
        "Human approval applies to the exact tree hash",
        "publish that exact tree",
        "exact-tree human approval",
    )
    problems = [f"missing: {statement}" for statement in required_statements if statement not in policy]
    problems += [f"dropped: {statement}" for statement in kept_statements if statement not in policy]
    problems += [f"retired phrase present: {phrase}" for phrase in retired_phrases if phrase in policy]
    assert problems == []


def test_github_first_change_has_a_recovery_step() -> None:
    policy = _policy_sentences()
    required_statements = (
        "### Changes that land on GitHub first",
        "Keep GitLab branch `mirror/github-main` at exactly the commits of GitHub `main`, and advance "
        "`mirror/github-main` only forward.",
        "Bring a change that has landed on GitHub first into GitLab `main` through a merge request.",
        "The merge commit for a GitHub-first recovery merge request must have GitLab `main` as first "
        "parent and GitHub `main` as second parent, and the merge commit's tree must equal the tree "
        "of GitHub `main`.",
        "Preserve all original commits with their authors and commit IDs without rewriting any commits.",
        "After a GitHub-first recovery merge, GitLab `main` contains GitHub `main`.",
    )
    kept_statements = (
        "Development work occurs on GitLab feature branches.",
        "Do not develop directly on either platform's `main` branch.",
        "No change reaches GitHub until a human approves the sanitized public snapshot.",
    )
    problems = [f"missing: {statement}" for statement in required_statements if statement not in policy]
    problems += [f"dropped: {statement}" for statement in kept_statements if statement not in policy]
    assert problems == []


def test_release_gate_states_the_small_update_exception() -> None:
    policy = _policy_sentences()
    heading = "### GitHub release regression and core-functionality gate"
    assert policy.count(heading) == 1
    gate = policy.split(heading, 1)[1].split("### ", 1)[0]
    small_update = (
        "Small documentation, artwork, metadata, and narrowly scoped dependency updates do not "
        "require a full deployed ML rerun. Run relevant code tests and dependency/security checks; "
        "for MCP or other SDK changes, include the affected integration and compatibility tests. "
        "Preserve the required native CI gates and record why the scope qualifies for this exception."
    )
    kept_statements = (
        "Focused tests alone do not qualify a release.",
        "Any failed, missing, incomplete, or unverifiable required check blocks publication.",
        "Passing checks does not itself authorize a GitHub write.",
    )
    problems = [] if small_update in gate else [f"missing: {small_update}"]
    problems += [f"dropped: {statement}" for statement in kept_statements if statement not in gate]
    assert problems == []


def test_anyio_security_floor_is_declared() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    floor = next(
        (dependency for dependency in dependencies if dependency.startswith("anyio>=")),
        None,
    )
    assert floor is not None
    assert _version_tuple(floor.removeprefix("anyio>=")) >= (4, 14, 2)


def test_anyio_lock_is_patched() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    anyio = next(package for package in lock["package"] if package["name"] == "anyio")
    assert _version_tuple(anyio["version"]) >= (4, 14, 2)


def test_gitpython_security_floor_is_declared() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    floor = next(
        (dependency for dependency in dependencies if dependency.startswith("gitpython>=")),
        None,
    )
    assert floor is not None
    assert _version_tuple(floor.removeprefix("gitpython>=")) >= (3, 1, 60)


def test_gitpython_lock_is_patched() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    gitpython = next(package for package in lock["package"] if package["name"] == "gitpython")
    assert _version_tuple(gitpython["version"]) >= (3, 1, 60)


def test_native_verification_extras_and_configuration_are_preserved() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert {"pytest>=9.0.3", "ruff>=0.6"}.issubset(set(extras.get("dev", [])))
    assert {
        "bandit>=1.7",
        "cyclonedx-bom>=4.0",
        "detect-secrets>=1.5",
        "pip-audit>=2.7",
    }.issubset(set(extras.get("security", [])))
    assert project["tool"]["bandit"]["exclude_dirs"] == [".venv", "build", "cdk.out", "dist"]
