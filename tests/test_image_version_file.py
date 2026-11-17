"""Container images that hold the application also hold root ``VERSION`` where the application reads it.

``src/amlas/_version.py`` maps root ``VERSION`` when the package runs from ``<root>/src/amlas``. An image
that copies the package source without installing the project has no distribution metadata of its own, so
it reports its version only when its recipe also copies ``VERSION`` beside the copied ``src`` directory.

The helpers read the final stage of a recipe and return what is missing. The cases at the end feed them
changed recipes, including each real recipe without its ``VERSION`` line, so a helper that stops reporting
fails here. Nothing in this file builds an image.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import posixpath
import shutil
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
READER = ROOT / "src" / "amlas" / "_version.py"

# Recipes whose final image holds the whole ``amlas`` package. Update this when a recipe starts or stops
# copying the package source.
HOLDERS = ("Dockerfile", "Dockerfile.coding-pair", "Dockerfile.model-acquisition", "Dockerfile.processing")

Copy = tuple[list[str], PurePosixPath, bool]


def _instructions(text: str) -> list[tuple[str, str]]:
    """Return (keyword, arguments) for every instruction of a recipe, continuation lines joined."""
    instructions = []
    pending = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        keyword, _, arguments = (pending + line).partition(" ")
        instructions.append((keyword.upper(), arguments.strip()))
        pending = ""
    return instructions


def _context_copies(text: str) -> tuple[list[Copy], list[str]]:
    """Return the build-context copies of the final stage and the copy lines this check cannot read.

    Each copy is (sources, destination, destination is a directory), with the destination made absolute
    against the working directory in force.
    """
    instructions = _instructions(text)
    starts = [index for index, (keyword, _) in enumerate(instructions) if keyword == "FROM"]
    final = instructions[starts[-1] + 1 :] if starts else []
    workdir = PurePosixPath("/")
    copies: list[Copy] = []
    unread = []
    for keyword, arguments in final:
        if keyword == "WORKDIR":
            workdir = workdir / arguments
            continue
        if keyword not in ("COPY", "ADD"):
            continue
        words = arguments.split()
        paths = [word for word in words if not word.startswith("--")]
        if any(word.startswith("--from=") for word in words):
            continue
        if len(paths) < 2 or paths[0].startswith("["):
            unread.append(f"{keyword} {arguments}")
            continue
        *sources, target = paths
        destination = PurePosixPath(posixpath.normpath(str(workdir / target)))
        copies.append((sources, destination, target.endswith("/") or len(sources) > 1))
    return copies, unread


def _package_directories(copies: list[Copy]) -> list[PurePosixPath]:
    """Return every image directory that receives the whole ``amlas`` package."""
    found = []
    for sources, destination, _ in copies:
        for source in sources:
            name = posixpath.normpath(source)
            if name == ".":
                found.append(destination / "src" / "amlas")
            elif name == "src":
                found.append(destination / "amlas")
            elif name == "src/amlas":
                found.append(destination)
    return found


def _version_files(copies: list[Copy]) -> list[PurePosixPath]:
    """Return every image path that receives root ``VERSION``."""
    found = []
    for sources, destination, directory in copies:
        for source in sources:
            name = posixpath.normpath(source)
            if name == ".":
                found.append(destination / "VERSION")
            elif name == "VERSION":
                found.append(destination / "VERSION" if directory else destination)
    return found


def recipe_problems(text: str) -> list[str]:
    """Return why an image built from a recipe would not report the version; empty without the package."""
    copies, unread = _context_copies(text)
    problems = [f"copy line is not readable by this check: {line}" for line in unread]
    files = _version_files(copies)
    for package in _package_directories(copies):
        if package.parent.name != "src":
            problems.append(f"the package lands in {package}, outside a directory named src")
            continue
        expected = package.parent.parent / "VERSION"
        if expected not in files:
            problems.append(f"the package lands in {package} but no COPY brings root VERSION to {expected}")
    return problems


def excluded_from_context(text: str, name: str = "VERSION") -> bool:
    """Return whether an ignore file leaves a root-level file out of the build context; the last match wins."""
    excluded = False
    for raw in text.splitlines():
        rule = raw.strip()
        if not rule or rule.startswith("#"):
            continue
        keep = rule.startswith("!")
        pattern = rule.removeprefix("!").strip().strip("/").removeprefix("**/")
        if fnmatch.fnmatchcase(name, pattern):
            excluded = not keep
    return excluded


def _recipes() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(ROOT.glob("Dockerfile*"))
        if path.is_file() and not path.name.endswith(".dockerignore")
    }


def _without_version_lines(text: str) -> str:
    """Return a recipe text without the lines that copy ``VERSION`` from the build context."""
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if not (line.lstrip().startswith("COPY ") and "VERSION" in line.split())
    )


def test_recipes_that_hold_the_package_are_the_listed_ones() -> None:
    recipes = _recipes()
    assert len(recipes) >= len(HOLDERS), "the container recipes were not found"
    holders = tuple(name for name, text in recipes.items() if _package_directories(_context_copies(text)[0]))
    assert holders == HOLDERS, "a recipe started or stopped copying the package source; update HOLDERS"


def test_every_recipe_that_holds_the_package_also_copies_version() -> None:
    assert VERSION_FILE.is_file(), "the repository root has no VERSION file"
    problems = {name: recipe_problems(text) for name, text in _recipes().items()}
    assert {name: found for name, found in problems.items() if found} == {}


def test_build_context_keeps_version() -> None:
    for name in ("", *_recipes()):
        ignore = ROOT / f"{name}.dockerignore"
        if ignore.is_file():
            assert not excluded_from_context(ignore.read_text(encoding="utf-8")), (
                f"{ignore.name} leaves VERSION out of the build context"
            )


def test_reader_finds_the_version_in_each_image_layout(tmp_path: Path) -> None:
    """Lay the package and ``VERSION`` out as each recipe copies them and ask the real reader."""
    line, counter = VERSION_FILE.read_text(encoding="utf-8").strip().split(".")
    expected = f"{int(line)}.{int(counter)}.0"
    for number, name in enumerate(HOLDERS):
        copies, _ = _context_copies(_recipes()[name])
        image = tmp_path / f"image-{number}"
        packages = _package_directories(copies)
        assert packages, f"{name} copies no package"
        readers = []
        for package in packages:
            directory = image / package.relative_to("/")
            directory.mkdir(parents=True)
            copied = directory / "_version.py"
            shutil.copyfile(READER, copied)
            spec = importlib.util.spec_from_file_location(f"image_version_reader_{number}", copied)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert module._checkout_version() is None, "the layout reports a version before VERSION is copied"
            readers.append(module)
        for file in _version_files(copies):
            path = image / file.relative_to("/")
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(VERSION_FILE, path)
        for module in readers:
            assert module._checkout_version() == expected, f"{name}: the copied package does not find VERSION"


# Changed recipes: the helper must report each of these, and accept the aligned controls.

BASE = "FROM base AS build\nWORKDIR /build\nCOPY VERSION ./VERSION\nFROM base\n"


@pytest.mark.parametrize(
    "final",
    [
        "WORKDIR /app\nCOPY src/ ./src/\nCOPY VERSION ./VERSION\n",
        "WORKDIR /app\nCOPY --chown=1:1 src/ ./src/\nCOPY --chown=1:1 VERSION ./\n",
        "COPY --chmod=0555 src/ /app/src/\nCOPY --chmod=0444 VERSION /app/VERSION\n",
        "COPY src /srv/tool/src\nCOPY VERSION pyproject.toml /srv/tool/\n",
        "WORKDIR /app\nCOPY src/amlas ./src/amlas\nCOPY \\\n  VERSION \\\n  /app/VERSION\n",
        "WORKDIR /app\nCOPY . .\n",
        "WORKDIR /app\nCOPY src/amlas/data_eng/inspection_schema.py /app/schema.py\n",
        "COPY --from=build /build/src /app/src\n",
    ],
)
def test_aligned_recipe_is_accepted(final: str) -> None:
    assert recipe_problems(BASE + final) == []


@pytest.mark.parametrize(
    "final",
    [
        "WORKDIR /app\nCOPY src/ ./src/\n",
        "WORKDIR /app\nCOPY src/ ./src/\nCOPY VERSION ./src/VERSION\n",
        "WORKDIR /app\nCOPY src/ ./src/\nCOPY VERSION /VERSION\n",
        "WORKDIR /app\nCOPY src/ ./src/\nCOPY VERSION /app\n",
        "WORKDIR /app\nCOPY src/ ./src/\nCOPY VERSION.txt ./VERSION\n",
        "WORKDIR /app\nCOPY src/ ./src/\nCOPY version ./\n",
        "WORKDIR /app\nCOPY src/ ./src/\nCOPY --from=build /build/VERSION ./VERSION\n",
        "WORKDIR /app\nCOPY src/ ./src/\n# COPY VERSION ./VERSION\n",
        "WORKDIR /app\nCOPY src/ ./src/\nWORKDIR /srv\nCOPY VERSION ./VERSION\n",
        "WORKDIR /app\nCOPY src/ ./code/\nCOPY VERSION ./VERSION\n",
        "WORKDIR /app\nCOPY src/amlas ./amlas\nCOPY VERSION ./VERSION\n",
        "COPY --chmod=0555 src/ /app/src/\nCOPY --chmod=0444 VERSION /opt/VERSION\n",
        'WORKDIR /app\nCOPY ["src/", "./src/"]\nCOPY VERSION ./VERSION\n',
    ],
)
def test_changed_recipe_is_reported(final: str) -> None:
    assert recipe_problems(BASE + final) != []


@pytest.mark.parametrize("name", HOLDERS)
def test_real_recipe_without_its_version_line_is_reported(name: str) -> None:
    text = _recipes()[name]
    changed = _without_version_lines(text)
    assert changed != text, f"{name} has no VERSION copy line to remove"
    assert recipe_problems(changed) != []


@pytest.mark.parametrize(
    "text",
    ["", "# VERSION\n", "*\n!VERSION\n", "docs/\n*.md\n", "*/VERSION\n", "version\n", "VERSION\n!VERSION\n"],
)
def test_ignore_file_that_keeps_version_is_accepted(text: str) -> None:
    assert not excluded_from_context(text)


@pytest.mark.parametrize(
    "text",
    ["VERSION\n", "/VERSION\n", "*\n", "**\n", "V*\n", "**/VERSION\n", "*\n!src\n", "!VERSION\nVERSION\n"],
)
def test_ignore_file_that_drops_version_is_reported(text: str) -> None:
    assert excluded_from_context(text)
