"""Keep the dependency check of the pin sets that no lock check reaches honest about what it accepts."""

import json
import subprocess  # nosec B404 - runs the test's own interpreter on the script under test, no shell
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_pin_sets.py"
ACCEPTED = ROOT / "scripts" / "pin_audit_accepted.toml"
TODAY = "2026-06-01"
FINDING = {"id": "ID-1", "aliases": ["ALIAS-1"], "fix_versions": ["1.2"]}

# Stand-in for the check tool: the test writes it as a package named pip_audit and puts it on the import
# path of the script. It answers from a table, records each call and how it was started, and can fail in
# three ways. The table can also leave a pin out of the report, report a pin as skipped, and name the exit
# code that follows a complete report.
STAND_IN = r"""
import json, os, pathlib, sys

args = sys.argv[1:]
table = json.loads(pathlib.Path(os.environ["STAND_IN_TABLE"]).read_text())
lines = pathlib.Path(args[args.index("-r") + 1]).read_text().split()
with pathlib.Path(os.environ["STAND_IN_CALLS"]).open("a") as calls:
    calls.write(json.dumps({"args": args, "lines": lines, "started": sys.orig_argv}) + "\n")
if table.get("mode") == "crash":
    sys.stderr.write("stand-in: the service did not answer\n")
    sys.exit(2)
if table.get("mode") == "no-report":
    sys.exit(0)
report = pathlib.Path(args[args.index("-o") + 1])
if table.get("mode") == "broken-report":
    report.write_text("{not a report")
    sys.exit(0)
dependencies = []
for line in lines:
    name, version = line.split("==")
    if line in table.get("left_out", []):
        continue
    if line in table.get("skipped", {}):
        dependencies.append({"name": name, "skip_reason": table["skipped"][line]})
        continue
    dependencies.append({"name": name, "version": version, "vulns": table.get("findings", {}).get(line, [])})
report.write_text(json.dumps({"dependencies": dependencies, "fixes": []}))
if "exit_code" in table:
    sys.stderr.write("stand-in: ended with the exit code of its table\n")
    sys.exit(table["exit_code"])
sys.exit(1 if any(item.get("vulns") for item in dependencies) else 0)
"""

LOCK = """version = 1
requires-python = ">=3.11"

[[package]]
name = "sample"
version = "0.1.0"
source = { editable = "." }

[[package]]
name = "alpha"
version = "1.5"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "alpha"
version = "1.6"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "beta"
version = "2.4"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "gamma"
version = "0.9"
source = { registry = "https://pypi.org/simple" }
"""

PYPROJECT = """[project]
name = "sample"
version = "0.1.0"
dependencies = ["alpha>=1.2", "beta>=2.0,<3"]

[project.optional-dependencies]
one = ["alpha>=1.0", "Gamma_Pkg[x]>=0.5"]
"""


def _tree(tmp_path, tool_versions=("0.13.4", "0.13.4", "0.13.4")):
    root = tmp_path / "tree"
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "Dockerfile").write_text(f"RUN pip install --no-cache-dir uv=={tool_versions[0]}\n")
    (root / ".gitlab-ci.yml").write_text(f"    - python -m pip install --no-cache-dir uv=={tool_versions[1]}\n")
    (root / ".github" / "workflows" / "public-release.yml").write_text(
        f"      - run: python -m pip install --no-cache-dir uv=={tool_versions[2]}\n"
    )
    for name in ("Dockerfile.model-acquisition", "Dockerfile.preparation", "Dockerfile.qwen_sft"):
        (root / name).write_text("RUN printf '%s\\n' 'pip==26.2 --hash=sha256:00' > /app/requirements.txt\n")
    (root / "uv.lock").write_text(LOCK)
    (root / "pyproject.toml").write_text(PYPROJECT)
    return root


def _entry(identifier="ID-1", package="alpha", pin_set="floors", expires="2026-12-31", reason="waits for a release"):
    fields = {"id": identifier, "package": package, "pin_set": pin_set, "reason": reason, "expires": expires}
    return "[[accepted]]\n" + "".join(f'{key} = "{value}"\n' for key, value in fields.items() if value is not None)


def _run(tmp_path, root, accepted="", findings=None, mode="ok", sets=None, extra=(), answers=None, cwd=None):
    package = tmp_path / "stand_in" / "pip_audit"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(STAND_IN)
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"mode": mode, "findings": findings or {}, **(answers or {})}))
    calls = tmp_path / "calls.jsonl"
    accepted_file = tmp_path / "accepted.toml"
    accepted_file.write_text(accepted)
    command = [sys.executable, "-B", str(SCRIPT), "--root", str(root), "--accepted", str(accepted_file)]
    command += ["--today", TODAY, *extra]
    if sets:
        command += ["--sets", sets]
    environment = {"STAND_IN_TABLE": str(table), "STAND_IN_CALLS": str(calls), "PATH": "/usr/bin:/bin"}
    environment["PYTHONPATH"] = str(package.parent)
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    environment["TMPDIR"] = str(scratch)
    result = subprocess.run(  # nosec B603 - the test's own interpreter on the script under test, arguments the test builds, no shell
        command, capture_output=True, text=True, check=False, env=environment, cwd=cwd
    )
    recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return result, recorded


def test_a_tree_without_findings_passes_and_every_pin_set_is_checked(tmp_path):
    result, calls = _run(tmp_path, _tree(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT new=0 accepted_rows=0 problems=0" in result.stdout
    assert [call["lines"] for call in calls] == [
        ["uv==0.13.4", "pip==26.2"],
        ["alpha==1.5", "beta==2.4", "gamma==0.9"],
        ["alpha==1.6"],
        ["alpha==1.0", "beta==2.0", "gamma-pkg==0.5"],
    ], "tool versions, every index version in the lock, and the lowest version each requirement allows"
    for call in calls:
        for flag in ("--disable-pip", "--no-deps", "--strict", "--vulnerability-service", "osv", "-f", "json"):
            assert flag in call["args"]


def test_a_new_finding_fails(tmp_path):
    result, _ = _run(tmp_path, _tree(tmp_path), findings={"alpha==1.0": [FINDING]})
    assert result.returncode == 1
    assert "NEW      floors alpha 1.0 ID-1 fixed in 1.2" in result.stdout
    assert "RESULT new=1 accepted_rows=0 problems=0" in result.stdout


@pytest.mark.parametrize("identifier", ["ID-1", "ALIAS-1"])
def test_an_accepted_finding_passes_by_id_or_alias(identifier, tmp_path):
    result, _ = _run(tmp_path, _tree(tmp_path), accepted=_entry(identifier), findings={"alpha==1.0": [FINDING]})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACCEPTED floors alpha 1.0 ID-1 until 2026-12-31" in result.stdout
    assert "RESULT new=0 accepted_rows=1 problems=0" in result.stdout


@pytest.mark.parametrize(("expires", "code"), [("2026-05-31", 1), (TODAY, 0)])
def test_an_expired_entry_fails_and_the_expiry_day_itself_still_counts(expires, code, tmp_path):
    result, _ = _run(
        tmp_path, _tree(tmp_path), accepted=_entry(expires=expires), findings={"alpha==1.0": [FINDING]}
    )
    assert result.returncode == code, result.stdout + result.stderr
    assert ("expired on 2026-05-31" in result.stdout) == (code == 1)


def test_an_entry_that_matches_nothing_fails(tmp_path):
    result, _ = _run(tmp_path, _tree(tmp_path), accepted=_entry())
    assert result.returncode == 1
    assert "accepted entry ID-1 (alpha, floors) matches nothing" in result.stdout
    assert "RESULT new=0 accepted_rows=0 problems=1" in result.stdout


@pytest.mark.parametrize(
    ("identifier", "package", "pin_set"),
    [
        pytest.param("ALIAS-1", "beta", "floors", id="alias-for-another-package"),
        pytest.param("ALIAS-1", "alpha", "uv-lock-all", id="alias-for-another-pin-set"),
        pytest.param("ID-1", "beta", "floors", id="id-for-another-package"),
        pytest.param("ID-1", "alpha", "tool-pins", id="id-for-another-pin-set"),
    ],
)
def test_a_match_for_another_package_or_pin_set_does_not_excuse(identifier, package, pin_set, tmp_path):
    accepted = _entry(identifier, package, pin_set)
    result, _ = _run(tmp_path, _tree(tmp_path), accepted=accepted, findings={"alpha==1.0": [FINDING]})
    assert result.returncode == 1
    assert "NEW      floors alpha 1.0 ID-1 fixed in 1.2" in result.stdout
    assert f"accepted entry {identifier} ({package}, {pin_set}) matches nothing" in result.stdout
    assert "RESULT new=1 accepted_rows=0 problems=1" in result.stdout


@pytest.mark.parametrize("position", [0, 1, 2])
def test_files_naming_different_tool_versions_fail(position, tmp_path):
    versions = ["0.13.4", "0.13.4", "0.13.4"]
    versions[position] = "0.13.5"
    result, calls = _run(tmp_path, _tree(tmp_path, tuple(versions)))
    assert result.returncode == 1
    assert "PROBLEM  tool version uv: expected one version" in result.stdout
    assert "0.13.4" in result.stdout and "0.13.5" in result.stdout
    assert "RESULT new=0 accepted_rows=0 problems=1" in result.stdout
    assert [call["lines"] for call in calls[:2]] == [["uv==0.13.4", "pip==26.2"], ["uv==0.13.5"]], "both versions are checked"


@pytest.mark.parametrize("mode", ["crash", "no-report", "broken-report"])
def test_a_tool_error_fails(mode, tmp_path):
    result, calls = _run(tmp_path, _tree(tmp_path), mode=mode)
    assert result.returncode == 1
    assert result.stdout.count("PROBLEM  check tool") == 4, "one problem per pin set round that could not be checked"
    assert "RESULT new=0 accepted_rows=0 problems=4" in result.stdout
    assert len(calls) == 4


@pytest.mark.parametrize("code", [2, 3, 70])
def test_a_tool_exit_code_other_than_0_or_1_fails_after_a_complete_report_without_findings(code, tmp_path):
    result, calls = _run(tmp_path, _tree(tmp_path), answers={"exit_code": code})
    assert result.returncode == 1
    ended = f"(exit {code}): stand-in: ended with the exit code of its table"
    for name in ("tool-pins-1", "uv-lock-all-1", "uv-lock-all-2", "floors-1"):
        assert f"PROBLEM  check tool failed for {name} {ended}" in result.stdout
    assert "RESULT new=0 accepted_rows=0 problems=4" in result.stdout
    assert len(calls) == 4


def test_a_tool_exit_code_1_fails_after_a_complete_report_without_findings(tmp_path):
    result, calls = _run(tmp_path, _tree(tmp_path), answers={"exit_code": 1})
    assert result.returncode == 1, result.stdout + result.stderr
    ended = "(exit 1, report without a reported entry): stand-in: ended with the exit code of its table"
    for name in ("tool-pins-1", "uv-lock-all-1", "uv-lock-all-2", "floors-1"):
        assert f"PROBLEM  check tool failed for {name} {ended}" in result.stdout
    assert "RESULT new=0 accepted_rows=0 problems=4" in result.stdout
    assert len(calls) == 4, "every round was still sent to the tool"


def test_a_report_that_leaves_a_pin_out_fails(tmp_path):
    result, calls = _run(tmp_path, _tree(tmp_path), answers={"left_out": ["beta==2.0"]})
    assert result.returncode == 1
    assert "PROBLEM  check tool report for floors-1 lacks 1 of 3 pins: ['beta==2.0']" in result.stdout
    assert "RESULT new=0 accepted_rows=0 problems=1" in result.stdout
    assert len(calls) == 4, "every round was still sent to the tool"


def test_a_pin_that_the_tool_reports_as_skipped_fails(tmp_path):
    skipped = {"gamma==0.9": "the service knows no such version"}
    result, calls = _run(tmp_path, _tree(tmp_path), answers={"skipped": skipped})
    assert result.returncode == 1
    assert "PROBLEM  check tool skipped gamma in uv-lock-all-1: the service knows no such version" in result.stdout
    assert "RESULT new=0 accepted_rows=0 problems=1" in result.stdout
    assert len(calls) == 4, "every round was still sent to the tool"


@pytest.mark.parametrize(
    ("relative", "tool", "present"),
    [
        pytest.param("Dockerfile", "uv", True, id="recipe"),
        pytest.param(".gitlab-ci.yml", "uv", True, id="first-pipeline"),
        pytest.param(".github/workflows/public-release.yml", "uv", True, id="second-pipeline"),
        pytest.param("Dockerfile.model-acquisition", "pip", True, id="model-acquisition-recipe"),
        pytest.param("Dockerfile.preparation", "pip", True, id="preparation-recipe"),
        pytest.param("Dockerfile.qwen_sft", "pip", True, id="fine-tune-recipe"),
        pytest.param(".gitlab-ci.yml", "uv", False, id="absent-pipeline-file"),
        pytest.param("Dockerfile.preparation", "pip", False, id="absent-recipe"),
    ],
)
def test_a_tool_file_that_names_no_version_fails(relative, tool, present, tmp_path):
    root = _tree(tmp_path)
    if present:
        (root / relative).write_text(f"RUN python -m pip install --no-cache-dir {tool}\n")
    else:
        (root / relative).unlink()
    result, calls = _run(tmp_path, root)
    assert result.returncode == 1
    assert f"PROBLEM  tool version {tool}: {relative} names no version" in result.stdout
    assert "RESULT new=0 accepted_rows=0 problems=1" in result.stdout
    assert calls[0]["lines"] == ["uv==0.13.4", "pip==26.2"], "the versions that the other files name are still checked"


@pytest.mark.parametrize(
    "requirement",
    [
        pytest.param("delta", id="name-only"),
        pytest.param("delta<3", id="upper-bound-only"),
        pytest.param("delta!=1.4", id="exclusion-only"),
        pytest.param("Delta_Pkg[x]>1.0", id="open-lower-bound"),
    ],
)
def test_a_requirement_without_a_lowest_version_fails(requirement, tmp_path):
    root = _tree(tmp_path)
    project = PYPROJECT.replace('"beta>=2.0,<3"]', f'"beta>=2.0,<3", "{requirement}"]')
    assert project != PYPROJECT
    (root / "pyproject.toml").write_text(project)
    result, calls = _run(tmp_path, root)
    assert result.returncode == 1
    assert f"PROBLEM  requirement without a lowest version: {requirement}" in result.stdout
    assert "RESULT new=0 accepted_rows=0 problems=1" in result.stdout
    floors = ["alpha==1.0", "beta==2.0", "gamma-pkg==0.5"]
    assert calls[-1]["lines"] == floors, "the lowest versions of the other requirements are still checked"


@pytest.mark.parametrize(
    ("entry", "words"),
    [
        pytest.param(_entry(reason=None), "lacks reason", id="no-reason"),
        pytest.param(_entry(expires=None), "lacks expires", id="no-expiry"),
        pytest.param(_entry(expires="next spring"), "expiry date", id="expiry-is-no-date"),
        pytest.param(_entry(pin_set="everything"), "unknown pin set", id="unknown-pin-set"),
    ],
)
def test_an_incomplete_entry_fails_and_excuses_nothing(entry, words, tmp_path):
    result, _ = _run(tmp_path, _tree(tmp_path), accepted=entry, findings={"alpha==1.0": [FINDING]})
    assert result.returncode == 1
    assert words in result.stdout
    assert "NEW      floors alpha 1.0 ID-1 fixed in 1.2" in result.stdout


def test_a_run_limited_to_one_pin_set_leaves_entries_of_other_sets_alone(tmp_path):
    result, calls = _run(tmp_path, _tree(tmp_path), accepted=_entry(), sets="tool-pins")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call["lines"] for call in calls] == [["uv==0.13.4", "pip==26.2"]]


def test_the_shipped_accepted_file_is_empty_and_states_its_format():
    text = ACCEPTED.read_text(encoding="utf-8")
    assert tomllib.loads(text).get("accepted", []) == []
    comments = "\n".join(line for line in text.splitlines() if line.startswith("#"))
    for field in ("[[accepted]]", "id =", "package =", "pin_set =", "reason =", "expires ="):
        assert field in comments
    for pin_set in ("tool-pins", "uv-lock-all", "floors"):
        assert pin_set in comments


def test_the_repository_names_one_tool_version_in_its_three_files(tmp_path):
    result, calls = _run(tmp_path, ROOT, sets="tool-pins")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 1 and len(calls[0]["lines"]) == 2
    assert calls[0]["lines"][0].startswith("uv==") and calls[0]["lines"][1].startswith("pip==")


def test_the_script_has_no_option_that_names_another_check_tool(tmp_path):
    other = tmp_path / "other_tool.py"
    started = tmp_path / "other_tool_started"
    other.write_text(f"import pathlib\n\npathlib.Path({str(started)!r}).write_text('started')\n")
    result, calls = _run(tmp_path, _tree(tmp_path), extra=["--pip-audit", f"{sys.executable} -B {other}"])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "usage:" in result.stderr and "unrecognized arguments: --pip-audit" in result.stderr
    assert result.stdout == ""
    assert calls == [] and not started.exists(), "a refused command line runs nothing"


def test_the_check_tool_is_started_only_as_the_pip_audit_module_of_this_interpreter(tmp_path):
    result, calls = _run(tmp_path, _tree(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    scratch = Path(calls[0]["args"][3]).parent
    rounds = ["tool-pins-1", "uv-lock-all-1", "uv-lock-all-2", "floors-1"]
    first = [sys.executable, "-P", "-m", "pip_audit", "--disable-pip", "--no-deps", "-r"]
    second = ["--strict", "--vulnerability-service", "osv", "-f", "json", "-o"]
    assert [call["started"] for call in calls] == [
        [*first, str(scratch / f"{name}.txt"), *second, str(scratch / f"{name}.json")] for name in rounds
    ], "this interpreter, the module, the fixed options and the two files of each round, nothing else"
    assert scratch.is_absolute() and not scratch.exists(), "the files sat in the script's own temporary directory"


# A package named pip_audit in the working folder of the script. It leaves a marker when it runs and
# writes a report without rows, so a run that used it would end with exit code 0.
WORKING_FOLDER_PACKAGE = r"""
import json, pathlib, sys

pathlib.Path(MARKER).write_text("started")
args = sys.argv[1:]
lines = pathlib.Path(args[args.index("-r") + 1]).read_text().split()
clean = [{"name": line.split("==")[0], "version": line.split("==")[1], "vulns": []} for line in lines]
pathlib.Path(args[args.index("-o") + 1]).write_text(json.dumps({"dependencies": clean, "fixes": []}))
"""


def test_the_check_tool_is_not_taken_from_the_working_folder(tmp_path):
    """A module start puts the working folder first on the import path unless the start says otherwise."""
    folder = tmp_path / "working_folder"
    package = folder / "pip_audit"
    package.mkdir(parents=True)
    started = tmp_path / "working_folder_package_started"
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(WORKING_FOLDER_PACKAGE.replace("MARKER", repr(str(started))))
    result, calls = _run(tmp_path, _tree(tmp_path), findings={"alpha==1.0": [FINDING]}, cwd=folder)
    assert not started.exists(), "a pip_audit package in the working folder stood in for the check tool"
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NEW      floors alpha 1.0 ID-1 fixed in 1.2" in result.stdout
    assert len(calls) == 4, "every round went to the module that PYTHONPATH names, which the start leaves in place"
    assert all(call["started"][1] == "-P" for call in calls)


def test_an_option_without_its_value_is_a_usage_error(tmp_path):
    """--help in the place of a value is the missing value of that option, not a request for the help text."""
    result, calls = _run(tmp_path, _tree(tmp_path), extra=["--root", "--help"])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "usage:" in result.stderr and "argument --root: expected one argument" in result.stderr
    assert result.stdout == ""
    assert calls == [], "a refused command line runs nothing"
