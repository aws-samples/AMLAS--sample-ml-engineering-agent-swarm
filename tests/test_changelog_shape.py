"""The root CHANGELOG.md keeps one shape, and its upkeep rule stays written down."""

from __future__ import annotations

import datetime
import itertools
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"

TITLE = "# Changelog"
FORMAT_URL = "https://keepachangelog.com/en/1.1.0/"
UNRELEASED = "## [Unreleased]"
GROUP_ORDER = ("Added", "Changed", "Deprecated", "Removed", "Fixed")
RELEASE_HEADING = re.compile(r"^## \[(\d+)\.(\d{2,})\] - (\d{4}-\d{2}-\d{2})$")
RELEASE_HEADING_FORM = "## [N.NN] - YYYY-MM-DD"
BULLET = re.compile(r"^- [A-Z][^\n]*\.$")

# The introduction's note on the release numbers that were assigned afterwards. It records what
# those releases carried when they were published, so it stays word for word, as one paragraph.
ASSIGNED_NUMBERS_NOTE = (
    "The numbers `0.01` to `0.05` were assigned afterwards. Releases `0.01` to `0.05` and the first "
    "publication of `0.06` carried the package version `0.1.0`."
)

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


def follows(newer: tuple[int, int], older: tuple[int, int]) -> bool:
    """Return whether a version is the next release: the counter one higher, or a new release line at 00."""
    return newer == (older[0], older[1] + 1) or (newer[0] > older[0] and newer[1] == 0)


def release_heading_problems(headings: list[str]) -> list[str]:
    """Return what is wrong with the level-two headings of a changelog, given in file order."""
    problems = []
    if not headings or headings[0] != UNRELEASED:
        problems.append(f"the first section must be {UNRELEASED!r}")
    if headings.count(UNRELEASED) > 1:
        problems.append(f"{UNRELEASED!r} appears more than once")
    releases = []
    for heading in headings:
        if heading == UNRELEASED:
            continue
        match = RELEASE_HEADING.match(heading)
        if match is None:
            problems.append(f"release heading is not {RELEASE_HEADING_FORM!r}: {heading!r}")
            continue
        line, counter, day = match.groups()
        if line != str(int(line)) or counter != f"{int(counter):02d}":
            problems.append(f"release heading version has a spare leading zero: {heading!r}")
        try:
            date = datetime.date.fromisoformat(day)
        except ValueError:
            problems.append(f"release heading carries no real date: {heading!r}")
            continue
        releases.append(((int(line), int(counter)), date, heading))
    if not releases:
        problems.append("the file lists no release")
    for newer, older in itertools.pairwise(releases):
        (version, date, heading), (older_version, older_date, older_heading) = newer, older
        if version <= older_version:
            problems.append(f"versions must strictly decrease: {heading!r} is followed by {older_heading!r}")
        elif not follows(version, older_version):
            problems.append(
                "a release raises the counter by one or starts a new release line at 00: "
                f"{heading!r} is followed by {older_heading!r}"
            )
        if date < older_date:
            problems.append(f"dates must not increase: {heading!r} is followed by {older_heading!r}")
    return problems


def _introduction() -> list[str]:
    """Return the paragraphs between the title and the first level-two section."""
    lines = _lines()
    first_section = next(index for index, line in enumerate(lines) if line.startswith("## "))
    return [line for line in lines[1:first_section] if line]


def assigned_numbers_note_problems(introduction: list[str]) -> list[str]:
    """Return what is wrong with the note on the numbers assigned afterwards, given the introduction."""
    if introduction.count(ASSIGNED_NUMBERS_NOTE) == 1:
        return []
    if ASSIGNED_NUMBERS_NOTE in introduction:
        return ["the note on the release numbers assigned afterwards appears more than once"]
    return [
        "the introduction must keep this paragraph word for word, as one paragraph: " + ASSIGNED_NUMBERS_NOTE
    ]


def test_title_and_introduction() -> None:
    lines = _lines()
    assert lines[0] == TITLE
    first_section = next(index for index, line in enumerate(lines) if line.startswith("## "))
    introduction = [line for line in lines[1:first_section] if line]
    assert introduction, "the introduction is missing"
    assert FORMAT_URL in " ".join(introduction)
    assert "`VERSION`" in " ".join(introduction), "the introduction must say where the version lives"
    for line in introduction:
        assert not line.startswith(("#", "-", " ")), f"introduction is plain sentences only: {line!r}"
    assert len(introduction) <= 3


def test_introduction_keeps_the_note_on_the_numbers_assigned_afterwards() -> None:
    assert assigned_numbers_note_problems(_introduction()) == []


@pytest.mark.parametrize(
    "paragraphs",
    [
        [],
        ["Each release has a version `N.NN` held in the root `VERSION` file."],
        [
            "The numbers `0.01` to `0.05` were assigned afterwards, and those releases carried the package "
            "version `0.1.0`."
        ],
        [ASSIGNED_NUMBERS_NOTE.replace("and the first publication of `0.06` ", "")],
        ["The numbers `0.01` to `0.05` were assigned afterwards.", ASSIGNED_NUMBERS_NOTE.split(". ", 1)[1]],
        [ASSIGNED_NUMBERS_NOTE + " Later releases carry their own number."],
        [ASSIGNED_NUMBERS_NOTE, ASSIGNED_NUMBERS_NOTE],
    ],
)
def test_changed_missing_or_repeated_note_is_reported(paragraphs: list[str]) -> None:
    assert assigned_numbers_note_problems(paragraphs) != []


def test_unreleased_is_first_and_releases_are_versioned_newest_first() -> None:
    headings = [heading for heading, _ in _sections()]
    assert headings, "CHANGELOG.md has no sections"
    assert release_heading_problems(headings) == []


def test_aligned_headings_are_accepted() -> None:
    assert release_heading_problems([UNRELEASED, "## [0.06] - 2026-01-02"]) == []
    assert release_heading_problems([UNRELEASED, "## [1.00] - 2026-03-01", "## [0.123] - 2026-03-01"]) == []
    assert release_heading_problems([UNRELEASED, "## [0.06] - 2026-01-02", "## [0.05] - 2026-01-02"]) == []
    assert release_heading_problems([UNRELEASED, "## [0.100] - 2026-03-01", "## [0.99] - 2026-03-01"]) == []
    assert (
        release_heading_problems(
            [UNRELEASED, "## [2.00] - 2026-05-01", "## [1.01] - 2026-04-01", "## [1.00] - 2026-03-01"]
        )
        == []
    )


@pytest.mark.parametrize(
    "headings",
    [
        [],
        [UNRELEASED],
        ["## [0.06] - 2026-01-02"],
        ["## [0.06] - 2026-01-02", UNRELEASED],
        [UNRELEASED, UNRELEASED, "## [0.06] - 2026-01-02"],
        [UNRELEASED, "## [2026-01-02]"],
        [UNRELEASED, "## [0.06]"],
        [UNRELEASED, "## [0.6.0] - 2026-01-02"],
        [UNRELEASED, "## [0.6] - 2026-01-02"],
        [UNRELEASED, "## [0.006] - 2026-01-02"],
        [UNRELEASED, "## 0.06 - 2026-01-02"],
        [UNRELEASED, "## [0.06] - 2026-13-02"],
        [UNRELEASED, "## [0.06] 2026-01-02"],
        [UNRELEASED, "## [0.05] - 2026-01-01", "## [0.06] - 2026-01-02"],
        [UNRELEASED, "## [0.06] - 2026-01-02", "## [0.06] - 2026-01-01"],
        [UNRELEASED, "## [0.06] - 2026-01-01", "## [0.05] - 2026-01-02"],
        [UNRELEASED, "## [0.08] - 2026-01-03", "## [0.06] - 2026-01-02"],
        [UNRELEASED, "## [1.01] - 2026-03-01", "## [0.06] - 2026-01-02"],
        [UNRELEASED, "## [1.07] - 2026-03-01", "## [0.06] - 2026-01-02"],
        [UNRELEASED, "## [1.00] - 2026-03-01", "## [0.07] - 2026-02-01", "## [0.05] - 2026-01-01"],
    ],
)
def test_drifted_headings_are_reported(headings: list[str]) -> None:
    assert release_heading_problems(headings) != []


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
        "`" + RELEASE_HEADING_FORM + "`",
        "### Versioning",
        "Root `VERSION` is the one source of the project version.",
        "raises the counter in root `VERSION` by one",
        "raise the counter in root `VERSION` by one",
        "`N` rises only when the maintainers decide to start a new release line; `NN` then restarts at `00`.",
        "Versions compare as the pair of `N` and `NN` read as integers.",
        "A publication that changes nothing a user or maintainer sees keeps its `VERSION` and gets no new "
        "changelog heading.",
        "`tests/test_version_single_source.py`",
    ):
        assert statement in " ".join(contract.split()), f"contract lacks: {statement}"
    guide = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "## Changelog" in guide
    for statement in (
        "add one line under `## [Unreleased]` in `CHANGELOG.md`",
        "The root `VERSION` file is the one source of that number.",
        "together with the `pyproject.toml` version and the project's own record in `uv.lock`",
        "`" + RELEASE_HEADING_FORM + "`",
        "`N` rises only when the maintainers decide to start a new release line; `NN` then restarts at `00`.",
        "A publication that changes nothing a user or maintainer sees keeps its `VERSION` and gets no new "
        "changelog heading.",
        "Leave `VERSION` unchanged in any other change.",
    ):
        assert statement in " ".join(guide.split()), f"contributor guide lacks: {statement}"
