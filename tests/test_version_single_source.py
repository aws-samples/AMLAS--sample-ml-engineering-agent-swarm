"""Root ``VERSION`` is the one source of the project version; every other statement follows it.

``VERSION`` holds ``N.NN`` and one newline: ``N`` is the release line and ``NN`` a counter of at least two
digits. ``pyproject.toml`` carries the mapped package version ``N.<NN as an integer>.0``, ``uv.lock`` records
the same version for the project's own package, the newest release heading of ``CHANGELOG.md`` names
``VERSION``, and the application reports what these files say without a literal of its own.

Each check is a helper that returns the drift it finds. The cases at the end feed the helpers drifted
inputs, so a helper that stops reporting fails here. Nothing in this file builds or installs the package.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "uv.lock"
CHANGELOG = ROOT / "CHANGELOG.md"
SOURCE = ROOT / "src" / "amlas"

VERSION_TEXT = re.compile(r"[0-9]+\.[0-9]{2,}\n")
UNRELEASED = "## [Unreleased]"
RELEASE_HEADING = re.compile(r"^## \[([^\]]*)\] - \d{4}-\d{2}-\d{2}$")

# Places where a build tool would take the project version from somewhere other than ``project.version``.
SECOND_SOURCES = (
    ("tool", "hatch", "version"),
    ("tool", "hatch", "build", "hooks", "vcs"),
    ("tool", "setuptools", "dynamic", "version"),
    ("tool", "setuptools_scm"),
    ("tool", "poetry", "version"),
    ("tool", "pdm", "version"),
    ("tool", "bumpversion"),
    ("tool", "commitizen"),
    ("tool", "versioningit"),
)

# Text that states a version in application source instead of reading it.
VERSION_LITERALS = (
    ("__version__ assignment", re.compile(r"""__version__\s*(?::[^=\n]*)?=\s*["']""")),
    ("version literal", re.compile(r"""\bver(?:sion)?\s*=\s*["']\d+\.\d+(?:\.\d+)?["']""")),
)


def version_file_problems(text: str) -> list[str]:
    """Return what is wrong with the content of ``VERSION``."""
    if VERSION_TEXT.fullmatch(text) is None:
        return [f"VERSION must hold N.NN and one newline, found {text!r}"]
    line, counter = text.rstrip("\n").split(".")
    problems = []
    if line != str(int(line)):
        problems.append(f"VERSION release line has a leading zero: {line!r}")
    if counter != f"{int(counter):02d}":
        problems.append(f"VERSION counter has a leading zero beyond two digits: {counter!r}")
    return problems


def mapped_package_version(version: str) -> str:
    """Map ``N.NN`` to the package version ``N.<NN as an integer>.0``."""
    line, counter = version.strip().split(".")
    return f"{int(line)}.{int(counter)}.0"


def pyproject_problems(document: dict, expected: str) -> list[str]:
    """Return where a parsed ``pyproject.toml`` departs from the mapped version."""
    project = document.get("project", {})
    problems = []
    if project.get("version") != expected:
        problems.append(
            f"pyproject.toml project version is {project.get('version')!r}, VERSION maps to {expected!r}"
        )
    if "version" in project.get("dynamic", ()):
        problems.append("pyproject.toml declares the project version dynamic")
    for path in SECOND_SOURCES:
        node: object = document
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        else:
            problems.append("pyproject.toml holds a second version source at " + ".".join(path))
    return problems


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_problems(document: dict, name: str, expected: str) -> list[str]:
    """Return where a parsed ``uv.lock`` departs from the mapped version for the project's own package."""
    records = [
        record for record in document.get("package", []) if _normal(record.get("name", "")) == _normal(name)
    ]
    if len(records) != 1:
        return [f"uv.lock holds {len(records)} records for {name!r}, expected one"]
    record = records[0]
    problems = []
    if record.get("source") != {"editable": "."}:
        problems.append(f"uv.lock record for {name!r} is not the project itself: {record.get('source')!r}")
    if record.get("version") != expected:
        problems.append(
            f"uv.lock records {name!r} at {record.get('version')!r}, VERSION maps to {expected!r}"
        )
    return problems


def changelog_problems(text: str, version: str) -> list[str]:
    """Return what is wrong with the newest release heading of a changelog text."""
    for line in text.splitlines():
        if not line.startswith("## ") or line == UNRELEASED:
            continue
        match = RELEASE_HEADING.match(line)
        if match is None:
            return [f"newest release heading is not '## [N.NN] - YYYY-MM-DD': {line!r}"]
        if match.group(1) != version:
            return [f"newest release heading names {match.group(1)!r}, VERSION holds {version!r}"]
        return []
    return ["CHANGELOG.md has no release heading"]


def source_problems(sources: dict[str, str]) -> list[str]:
    """Return every line of application source that states a version instead of reading it."""
    problems = []
    for path, text in sorted(sources.items()):
        for number, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in VERSION_LITERALS:
                if pattern.search(line):
                    problems.append(f"{path}:{number}: {kind}: {line.strip()}")
    return problems


def _version() -> str:
    assert VERSION_FILE.is_file(), "the repository root has no VERSION file"
    text = VERSION_FILE.read_text(encoding="utf-8")
    assert version_file_problems(text) == []
    return text.rstrip("\n")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_version_file_holds_release_line_and_counter() -> None:
    assert VERSION_FILE.is_file(), "the repository root has no VERSION file"
    text = VERSION_FILE.read_text(encoding="utf-8")
    assert re.match(r"^[0-9]+\.[0-9]{2,}\n$", text), f"VERSION must hold N.NN and one newline, found {text!r}"
    assert version_file_problems(text) == []


def test_pyproject_carries_the_mapped_version_and_no_second_source() -> None:
    assert pyproject_problems(_pyproject(), mapped_package_version(_version())) == []


def test_lock_records_the_mapped_version_for_the_project() -> None:
    name = _pyproject()["project"]["name"]
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    assert lock_problems(lock, name, mapped_package_version(_version())) == []


def test_newest_changelog_heading_names_the_version() -> None:
    assert changelog_problems(CHANGELOG.read_text(encoding="utf-8"), _version()) == []


def test_application_source_states_no_version_literal() -> None:
    sources = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(SOURCE.rglob("*.py"))
    }
    assert len(sources) > 100, "the application source was not found"
    assert source_problems(sources) == []


def test_application_reports_the_mapped_version(tmp_path: Path) -> None:
    expected = mapped_package_version(_version())

    import amlas
    from amlas import _version as reader
    from amlas.cli.commands.info import version as version_command

    assert reader.DISTRIBUTION == _pyproject()["project"]["name"]
    assert reader.package_version() == expected
    assert amlas.__version__ == expected

    shown = CliRunner().invoke(version_command, ["--json"])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output) == {"version": expected, "name": "amlas"}

    search_path = os.pathsep.join(part for part in (str(ROOT / "src"), os.environ.get("PYTHONPATH")) if part)
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "amlas.cli", "--version"],
        cwd=tmp_path,
        env=dict(os.environ, HOME=str(tmp_path), PYTHONPATH=search_path),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"amlas, version {expected}\n"


# Drifted inputs: every helper must report each of these, and accept the aligned control.

ALIGNED_PYPROJECT = {
    "project": {"name": "example-project", "version": "0.6.0"},
    "tool": {"hatch": {"build": {}}},
}
ALIGNED_LOCK = {
    "package": [
        {"name": "example-project", "version": "0.6.0", "source": {"editable": "."}},
        {"name": "other", "version": "0.1.0", "source": {"registry": "https://pypi.org/simple"}},
    ]
}
ALIGNED_CHANGELOG = "# Changelog\n\n## [Unreleased]\n\n## [0.06] - 2026-01-02\n\n## [0.05] - 2026-01-01\n"


def test_helpers_accept_aligned_inputs() -> None:
    assert version_file_problems("0.06\n") == []
    assert version_file_problems("12.345\n") == []
    assert mapped_package_version("0.06") == "0.6.0"
    assert mapped_package_version("1.10\n") == "1.10.0"
    assert mapped_package_version("2.123") == "2.123.0"
    assert pyproject_problems(ALIGNED_PYPROJECT, "0.6.0") == []
    assert lock_problems(ALIGNED_LOCK, "Example_Project", "0.6.0") == []
    assert changelog_problems(ALIGNED_CHANGELOG, "0.06") == []
    assert (
        source_problems({"a.py": 'api_version = "2017-09-30"\nitem(version="1.2-1")\nversion = read()\n'})
        == []
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "0.06",
        "0.06\n\n",
        "0.6\n",
        "0.6.0\n",
        "v0.06\n",
        " 0.06\n",
        "0.06 \n",
        "0.06\r\n",
        "00.06\n",
        "0.006\n",
        "0.0a\n",
    ],
)
def test_drifted_version_file_is_reported(text: str) -> None:
    assert version_file_problems(text) != []


@pytest.mark.parametrize(
    "change",
    [
        {"project": {"name": "example-project", "version": "0.1.0"}},
        {"project": {"name": "example-project", "version": "0.06"}},
        {"project": {"name": "example-project", "version": "0.6"}},
        {"project": {"name": "example-project"}},
        {"project": {"name": "example-project", "version": "0.6.0", "dynamic": ["version"]}},
        {
            "project": {"name": "example-project", "version": "0.6.0"},
            "tool": {"hatch": {"version": {"path": "x"}}},
        },
        {
            "project": {"name": "example-project", "version": "0.6.0"},
            "tool": {"bumpversion": {"current_version": "0.6.0"}},
        },
        {"project": {"name": "example-project", "version": "0.6.0"}, "tool": {"setuptools_scm": {}}},
    ],
)
def test_drifted_pyproject_is_reported(change: dict) -> None:
    assert pyproject_problems(change, "0.6.0") != []


@pytest.mark.parametrize(
    "packages",
    [
        [],
        [{"name": "example-project", "version": "0.1.0", "source": {"editable": "."}}],
        [{"name": "example-project", "version": "0.6.0", "source": {"registry": "https://pypi.org/simple"}}],
        [{"name": "example-project", "source": {"editable": "."}}],
        [
            {"name": "example-project", "version": "0.6.0", "source": {"editable": "."}},
            {"name": "example_project", "version": "0.6.0", "source": {"editable": "."}},
        ],
        [{"name": "other", "version": "0.6.0", "source": {"editable": "."}}],
    ],
)
def test_drifted_lock_is_reported(packages: list) -> None:
    assert lock_problems({"package": packages}, "example-project", "0.6.0") != []


@pytest.mark.parametrize(
    "text",
    [
        "# Changelog\n\n## [Unreleased]\n",
        "# Changelog\n\n## [Unreleased]\n\n## [0.05] - 2026-01-01\n",
        "# Changelog\n\n## [Unreleased]\n\n## [2026-01-02]\n",
        "# Changelog\n\n## [Unreleased]\n\n## [0.06]\n",
        "# Changelog\n\n## [Unreleased]\n\n## [0.6.0] - 2026-01-02\n",
        "# Changelog\n\n## [Unreleased]\n\n## 0.06 - 2026-01-02\n",
        "# Changelog\n\n## [Unreleased]\n\n## [0.07] - 2026-01-03\n\n## [0.06] - 2026-01-02\n",
    ],
)
def test_drifted_changelog_is_reported(text: str) -> None:
    assert changelog_problems(text, "0.06") != []


@pytest.mark.parametrize(
    "line",
    [
        '__version__ = "0.6.0"',
        "__version__: str = '0.6.0'",
        '@click.version_option(version="0.6.0", prog_name="amlas")',
        '    ver = "0.6.0"',
        "version = '0.06'",
    ],
)
def test_version_literal_in_source_is_reported(line: str) -> None:
    assert len(source_problems({"module.py": "first = 1\n" + line + "\n"})) == 1
