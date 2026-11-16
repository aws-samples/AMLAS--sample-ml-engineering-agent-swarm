"""The root CHANGELOG.md keeps one shape, and its upkeep rule stays written down."""

from __future__ import annotations

import datetime
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"

TITLE = "# Changelog"
FORMAT_URL = "https://keepachangelog.com/en/1.1.0/"
UNRELEASED = "## [Unreleased]"
GROUP_ORDER = ("Added", "Changed", "Deprecated", "Removed", "Fixed")
RELEASE_HEADING = re.compile(r"^## \[(\d{4}-\d{2}-\d{2})\]$")
BULLET = re.compile(r"^- [A-Z][^\n]*\.$")

# Text a bullet must not carry. Path and host fragments are assembled so that this
# file holds no literal of the kind it rejects.
_SEP = "/"
BULLET_REJECTIONS = (
    ("work-item identifier", re.compile(r"(?<![\w`])[#!]\d+\b")),
    ("work-item identifier", re.compile(r"\b[A-Z]{1,6}-\d{2,}\b")),
    ("work-item identifier", re.compile(r"\b[A-Z]{1,6}-[A-Z]{2,}-\d+\b")),
    ("work-item identifier", re.compile(r"(?i)\b(?:issue|ticket|pull request|merge request)\s+#?\d+\b")),
    ("address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("address", re.compile(r"(?i)\bhttps?:")),
    ("address", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("host name", re.compile(r"(?i)\b[a-z0-9-]+\.(?:corp|internal|local|lan)\b")),
    ("host name", re.compile(r"(?i)\b[a-z0-9.-]+\.(?:com|net|org|dev|io)\b")),
    ("account number", re.compile(r"(?<!\d)\d{12}(?!\d)")),
    (
        "machine path",
        re.compile(
            "(?:"
            + "|".join(
                re.escape(_SEP + name + _SEP)
                for name in ("Users", "home", "Volumes", "local", "tmp", "var", "opt", "mnt", "root")
            )
            + r"|~"
            + re.escape(_SEP)
            + r"|\b[A-Za-z]:\\)"
        ),
    ),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("date", re.compile(r"\b(?:19|20)\d{2}\b")),
    (
        "date",
        re.compile(
            r"\b(?:January|February|March|April|June|July|August|September|October|November|December)\s+\d"
        ),
    ),
)


def _lines() -> list[str]:
    assert CHANGELOG.is_file(), "the repository root has no CHANGELOG.md"
    text = CHANGELOG.read_text(encoding="utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n"), "CHANGELOG.md must end with one newline"
    assert "\t" not in text and "\r" not in text
    return text.splitlines()


def _sections() -> list[tuple[str, list[str]]]:
    """Return (heading, body lines) for every level-two section, in file order."""
    sections: list[tuple[str, list[str]]] = []
    for line in _lines():
        if line.startswith("## "):
            sections.append((line, []))
        elif sections:
            sections[-1][1].append(line)
    return sections


def test_title_and_introduction() -> None:
    lines = _lines()
    assert lines[0] == TITLE
    first_section = next(index for index, line in enumerate(lines) if line.startswith("## "))
    introduction = [line for line in lines[1:first_section] if line]
    assert introduction, "the introduction is missing"
    assert FORMAT_URL in " ".join(introduction)
    for line in introduction:
        assert not line.startswith(("#", "-", " ")), f"introduction is plain sentences only: {line!r}"
    assert len(introduction) <= 3


def test_unreleased_is_first_and_releases_are_newest_first() -> None:
    headings = [heading for heading, _ in _sections()]
    assert headings, "CHANGELOG.md has no sections"
    assert headings[0] == UNRELEASED
    assert headings.count(UNRELEASED) == 1
    dates = []
    for heading in headings[1:]:
        match = RELEASE_HEADING.match(heading)
        assert match is not None, f"release heading is not '## [YYYY-MM-DD]': {heading!r}"
        dates.append(datetime.date.fromisoformat(match.group(1)))
    assert dates, "the file lists no release"
    assert dates == sorted(set(dates), reverse=True), "release headings must be unique and newest first"


def test_sections_hold_only_the_allowed_groups_with_one_line_bullets() -> None:
    for heading, body in _sections():
        groups: list[str] = []
        bullets_in_group = 0
        for line in body:
            if not line:
                continue
            if line.startswith("### "):
                if groups:
                    assert bullets_in_group, f"{heading}: group {groups[-1]} holds no line"
                group = line.removeprefix("### ")
                assert group in GROUP_ORDER, f"{heading}: group {group!r} is not allowed"
                groups.append(group)
                bullets_in_group = 0
                continue
            assert groups, f"{heading}: text outside a group: {line!r}"
            assert BULLET.match(line), f"{heading}: not a one-line bullet: {line!r}"
            bullets_in_group += 1
        if groups:
            assert bullets_in_group, f"{heading}: group {groups[-1]} holds no line"
        elif heading != UNRELEASED:
            raise AssertionError(f"{heading}: a release section needs at least one group")
        assert groups == [group for group in GROUP_ORDER if group in groups], (
            f"{heading}: groups repeat or are out of order: {groups}"
        )


def test_bullets_carry_no_identifiers_addresses_paths_or_dates() -> None:
    bullets = [line for _, body in _sections() for line in body if line.startswith("- ")]
    assert bullets, "CHANGELOG.md has no bullets"
    assert len(bullets) == len(set(bullets)), "a bullet is repeated"
    for line in bullets:
        for kind, pattern in BULLET_REJECTIONS:
            assert pattern.search(line) is None, f"bullet carries a {kind}: {line!r}"


def test_upkeep_rule_is_written_in_the_contract_and_the_contributor_guide() -> None:
    contract = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "### Changelog upkeep" in contract
    for statement in (
        "Add one line under `## [Unreleased]` in the root `CHANGELOG.md` for every user- or "
        "maintainer-visible change, in the same change.",
        "all changelogs except the root `CHANGELOG.md`",
        "`tests/test_changelog_shape.py`",
    ):
        assert statement in " ".join(contract.split()), f"contract lacks: {statement}"
    guide = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "## Changelog" in guide
    assert "add one line under `## [Unreleased]` in `CHANGELOG.md`" in " ".join(guide.split())
