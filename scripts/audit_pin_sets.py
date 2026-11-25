#!/usr/bin/env python3
"""Dependency check for the pin sets that the lock checks of the pipelines do not reach.

Pin sets:
  tool-pins    the uv version named in Dockerfile, .gitlab-ci.yml and
               .github/workflows/public-release.yml (the three files must name one version)
               and the pip version named in Dockerfile.model-acquisition,
               Dockerfile.preparation and Dockerfile.qwen_sft (one version)
  uv-lock-all  every package version that uv.lock takes from a package index, for all extras
  floors       the lowest version that each requirement in pyproject.toml allows

Each set goes to pip-audit as pinned requirements (OSV service, no resolution). The check tool
always runs as the pip_audit module of the interpreter that runs this script, started with -P.
With -P, Python does not put the working folder on the tool's import path. Python still
searches folders named in PYTHONPATH before the installed packages. An entry of the report
that is accepted for a time is one [[accepted]] table in scripts/pin_audit_accepted.toml.

Exit codes:
  1  No accepted entry covers a reported entry, or pin_audit_accepted.toml or the file
     given with --accepted contains invalid TOML. An accepted entry lacks any of id,
     package, pin_set, reason, or expires, names an unknown pin set, has no valid expiry
     date, is past its expiry date, or belongs to a checked pin set and matches no
     reported entry. A tool file names no version or is absent, the files name different
     versions of one tool, or a requirement in pyproject.toml has no lowest version.
     The check tool cannot start, exits with a code other than 0 or 1, exits with code 1
     and its report holds no reported entry, writes no readable report, or reports a
     dependency as skipped, or a report leaves out a pin.
     An unhandled error ends the script with a traceback, for example when uv-lock-all is
     selected and uv.lock is absent, when floors is selected and pyproject.toml contains
     invalid TOML, or when --today gives no valid date.
  2  A usage error: an unknown option, an unknown or empty selection for --sets, or an
     option without its value. For example, --root --help leaves --root without a value.
  0  With --help, the script prints the help text and exits with code 0 without checking
     a pin set. Otherwise, it exits with code 0 only if it checks every selected pin set
     and none of these cases apply.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess  # nosec B404 - runs this interpreter's pip_audit module on fixed options, no shell
import sys
import tempfile
import tomllib
from pathlib import Path

PIN_SETS = ("tool-pins", "uv-lock-all", "floors")
TOOL_FILES = {
    "uv": ("Dockerfile", ".gitlab-ci.yml", ".github/workflows/public-release.yml"),
    "pip": ("Dockerfile.model-acquisition", "Dockerfile.preparation", "Dockerfile.qwen_sft"),
}
ENTRY_FIELDS = ("id", "package", "pin_set", "reason", "expires")
REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?\s*(?:>=|==|~=)\s*([0-9][^,;\s]*)")


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def tool_pins(root: Path, problems: list[str]) -> list[tuple[str, str]]:
    pins = []
    for tool, files in TOOL_FILES.items():
        found: dict[str, list[str]] = {}
        for relative in files:
            path = root / relative
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            versions = set(re.findall(rf"\b{tool}==([0-9][0-9A-Za-z.]*)", text))
            if not versions:
                problems.append(f"tool version {tool}: {relative} names no version")
            for version in versions:
                found.setdefault(version, []).append(relative)
        ordered = sorted(found, key=version_key)
        if len(ordered) > 1:
            named = "; ".join(f"{version} in {', '.join(found[version])}" for version in ordered)
            problems.append(f"tool version {tool}: expected one version, found {named}")
        pins += [(tool, version) for version in ordered]
    return pins


def lock_pins(root: Path) -> list[tuple[str, str]]:
    data = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    return [
        (canonical(package["name"]), package["version"])
        for package in data.get("package", [])
        if "registry" in package.get("source", {})
    ]


def floor_pins(root: Path, problems: list[str]) -> list[tuple[str, str]]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements += extra
    lowest: dict[str, str] = {}
    for requirement in requirements:
        match = REQUIREMENT.match(requirement)
        if not match:
            problems.append(f"requirement without a lowest version: {requirement}")
            continue
        name = canonical(match.group(1))
        if name not in lowest or version_key(match.group(2)) < version_key(lowest[name]):
            lowest[name] = match.group(2)
    return sorted(lowest.items())


def rounds(pins: list[tuple[str, str]]) -> list[list[str]]:
    """Split pins into requirement files that never name one package twice."""
    files: list[list[str]] = []
    seen: dict[str, int] = {}
    for name, version in pins:
        index = seen.get(name, 0)
        seen[name] = index + 1
        if index == len(files):
            files.append([])
        files[index].append(f"{name}=={version}")
    return files


def load_accepted(path: Path, today: datetime.date, problems: list[str]) -> list[dict]:
    """Return the entries that may excuse a reported row; report every entry that may not."""
    if not path.is_file():
        return []
    try:
        tables = tomllib.loads(path.read_text(encoding="utf-8")).get("accepted", [])
    except tomllib.TOMLDecodeError as error:
        problems.append(f"accepted file {path.name} is not valid TOML: {error}")
        return []
    entries = []
    for number, table in enumerate(tables, 1):
        label = f"accepted entry {table.get('id') or '#' + str(number)}"
        missing = [field for field in ENTRY_FIELDS if not table.get(field)]
        if missing:
            problems.append(f"{label} lacks {', '.join(missing)}")
            continue
        label += f" ({table['package']}, {table['pin_set']})"
        if table["pin_set"] not in PIN_SETS:
            problems.append(f"{label} names an unknown pin set; known sets: {', '.join(PIN_SETS)}")
            continue
        try:
            expires = datetime.date.fromisoformat(str(table["expires"]))
        except ValueError:
            problems.append(f"{label} has no valid expiry date (YYYY-MM-DD): {table['expires']}")
            continue
        if expires < today:
            problems.append(f"{label} expired on {expires.isoformat()}")
            continue
        entries.append(
            {
                "label": label,
                "id": str(table["id"]),
                "package": canonical(str(table["package"])),
                "pin_set": table["pin_set"],
                "expires": expires.isoformat(),
            }
        )
    return entries


def check(lines: list[str], directory: Path, label: str, problems: list[str]) -> list[dict]:
    """Run the check tool on one requirement file; return its reported rows."""
    requirements = directory / f"{label}.txt"
    report = directory / f"{label}.json"
    requirements.write_text("\n".join(lines) + "\n", encoding="utf-8")
    arguments = [sys.executable, "-P", "-m", "pip_audit", "--disable-pip", "--no-deps", "-r", str(requirements)]
    arguments += ["--strict", "--vulnerability-service", "osv", "-f", "json", "-o", str(report)]
    try:
        done = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit  # nosec B603 - this interpreter's pip_audit module, fixed options, own temporary files, no shell
            arguments, capture_output=True, text=True, check=False
        )
    except OSError as error:
        problems.append(f"check tool could not start for {label}: {error}")
        return []
    if done.returncode not in (0, 1):
        tail = " ".join(done.stderr.split())[-300:]
        problems.append(f"check tool failed for {label} (exit {done.returncode}): {tail}")
        return []
    try:
        dependencies = json.loads(report.read_text(encoding="utf-8"))["dependencies"]
        reported = {canonical(dependency["name"]) for dependency in dependencies}
    except (OSError, ValueError, KeyError, TypeError):
        problems.append(f"check tool wrote no readable report for {label} (exit {done.returncode})")
        return []
    missing = [line for line in lines if canonical(line.split("==")[0]) not in reported]
    if missing:
        problems.append(f"check tool report for {label} lacks {len(missing)} of {len(lines)} pins: {missing[:5]}")
    rows = []
    for dependency in dependencies:
        if "skip_reason" in dependency:
            problems.append(f"check tool skipped {dependency['name']} in {label}: {dependency['skip_reason']}")
        for item in dependency.get("vulns", []):
            rows.append(
                {
                    "package": canonical(dependency["name"]),
                    "version": dependency["version"],
                    "id": item["id"],
                    "ids": {item["id"], *item.get("aliases", [])},
                    "fix": ",".join(item.get("fix_versions", [])) or "none",
                }
            )
    if done.returncode == 1 and not rows:
        tail = " ".join(done.stderr.split())[-300:]
        problems.append(f"check tool failed for {label} (exit 1, report without a reported entry): {tail}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="repository root")
    parser.add_argument("--accepted", help="accepted entries file (default: scripts/pin_audit_accepted.toml)")
    parser.add_argument("--today", default=datetime.date.today().isoformat(), help="date for expiry, YYYY-MM-DD")
    parser.add_argument("--sets", default=",".join(PIN_SETS), help="comma-separated pin sets to check")
    args = parser.parse_args()
    root = Path(args.root)
    selected = [name for name in args.sets.split(",") if name]
    unknown = [name for name in selected if name not in PIN_SETS]
    if unknown or not selected:
        parser.error(f"unknown or empty pin set selection: {args.sets}")
    today = datetime.date.fromisoformat(args.today)
    accepted_path = Path(args.accepted) if args.accepted else root / "scripts" / "pin_audit_accepted.toml"
    problems: list[str] = []
    entries = load_accepted(accepted_path, today, problems)
    builders = {
        "tool-pins": lambda: tool_pins(root, problems),
        "uv-lock-all": lambda: lock_pins(root),
        "floors": lambda: floor_pins(root, problems),
    }
    new, in_force, used = [], [], set()
    with tempfile.TemporaryDirectory() as scratch:
        for name in selected:
            pins = builders[name]()
            for number, lines in enumerate(rounds(pins), 1):
                for row in check(lines, Path(scratch), f"{name}-{number}", problems):
                    hits = [
                        index
                        for index, entry in enumerate(entries)
                        if entry["pin_set"] == name and entry["package"] == row["package"] and entry["id"] in row["ids"]
                    ]
                    if hits:
                        used.update(hits)
                        in_force.append((name, row, entries[hits[0]]["expires"]))
                    else:
                        new.append((name, row))
            print(f"{name:<12} pins={len(pins)}")
    for index, entry in enumerate(entries):
        if index not in used and entry["pin_set"] in selected:
            problems.append(f"{entry['label']} matches nothing: remove it")
    for name, row in new:
        print(f"NEW      {name} {row['package']} {row['version']} {row['id']} fixed in {row['fix']}")
    for name, row, expires in in_force:
        print(f"ACCEPTED {name} {row['package']} {row['version']} {row['id']} until {expires}")
    for problem in problems:
        print(f"PROBLEM  {problem}")
    print(f"RESULT new={len(new)} accepted_rows={len(in_force)} problems={len(problems)}")
    return 1 if new or problems else 0


if __name__ == "__main__":
    sys.exit(main())
