"""Read-only copy of the terminal observer's assertions about ``out/``, bound to explicit pins.

The official observer additionally checks the run's launcher and runner pids, the QA clone state, the AWS actor
and the absence of the phase containers; those are run-specific and stay in the official observer. Everything the
observer asserts about the files in ``out/`` is repeated here, with the same regular expression for the log counts.
Standard library only, Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess  # nosec B404 - fixed docker inspect argv, no shell
import sys
import xml.etree.ElementTree as ET  # nosemgrep: use-defused-xml  # nosec B405 - parses the JUnit file pytest wrote in this run's own container; a malformed file is a refusal
from dataclasses import dataclass
from pathlib import Path

GATES = (
    "bandit ci-yaml collection compile cpu-audit cpu-closure dox format governance package-build package-inspect "
    "pytest requirements-export ruff runtime-audit rust-default rust-default-build rust-format rust-native "
    "rust-native-build sbom secrets shell-syntax"
).split()
PHASES = ("audit", "dox", "package", "python", "rust")
_COUNT = re.compile(r"([0-9]+) (passed|skipped|subtests passed)")


@dataclass(frozen=True)
class Pins:
    candidate: str
    tree: str
    collection_sha256: str
    nodeids_sha256: str
    collected: int
    skip_sha256: str
    skip_function: str
    skip_module: str
    gates: tuple = tuple(GATES)
    phases: tuple = PHASES


def _require(condition, message="observer assertion failed"):
    """The observer's `assert`, kept active under `python -O`."""
    if not condition:
        raise AssertionError(message)


def _h(raw):
    return hashlib.sha256(raw).hexdigest()


def check_out_dir(out, pins, qa_dir=None):
    out = Path(out)
    summary_raw = (out / "full-ci-result.metadata.json").read_bytes()
    summary = json.loads(summary_raw)
    _require(summary["candidate"] == pins.candidate and summary["tree"] == pins.tree)
    _require(summary["all_required_gates_passed"] is True)
    codes = summary["gate_exit_codes"]
    _require(set(codes) == set(pins.gates) and all(type(v) is int and v == 0 for v in codes.values()))
    phases = summary["phase_exit_codes"]
    _require(set(phases) == set(pins.phases) and all(type(v) is int and v == 0 for v in phases.values()))
    gates = {name: int((out / (name + ".exit")).read_text()) for name in pins.gates}
    _require(gates == codes)
    inventories = []
    for name in ("collection.metadata.json", "execution-collection.metadata.json"):
        raw = (out / name).read_bytes()
        value = json.loads(raw)
        _require(_h(raw) == pins.collection_sha256)
        _require(type(value["collected"]) is int and value["collected"] == pins.collected)
        _require(value["nodeids_sha256"] == pins.nodeids_sha256)
        _require(summary[name] == value)
        inventories.append(
            {"collected": value["collected"], "nodeids_sha256": value["nodeids_sha256"], "sha256": _h(raw)}
        )
    _require(inventories[0] == inventories[1])
    junit_raw = (out / "full-pytest.xml").read_bytes()
    root = ET.fromstring(junit_raw)  # nosec B314 - parses the JUnit file pytest wrote in this run's own container; a malformed file is a refusal
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    _require(counts == summary["junit"] and _h(junit_raw) == summary["junit_sha256"])
    _require(counts["failures"] == 0 and counts["errors"] == 0 and counts["skipped"] == 1)
    cases = list(root.iter("testcase"))
    unique = len({(c.get("classname"), c.get("name")) for c in cases})
    physical = len(cases)
    _require(unique == inventories[0]["collected"] and physical == inventories[0]["collected"])
    skips = [c for c in cases if c.find("skipped") is not None]
    _require(len(skips) == 1)
    skip = skips[0]
    _require(
        skip.get("classname", "").split(".")[-1] == pins.skip_module
        and skip.get("name") == pins.skip_function
    )
    _require(_h((skip.find("skipped").get("message") or "").encode()) == pins.skip_sha256)
    if qa_dir is not None:
        source = Path(qa_dir) / "tests" / (pins.skip_module + ".py")
        module = ast.parse(source.read_bytes())
        fn = next(
            n
            for n in ast.walk(module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == pins.skip_function
        )
        reasons = {
            _h(n.args[0].value.encode())
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "skip"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and type(n.args[0].value) is str
        }
        _require(pins.skip_sha256 in reasons)
    text = (out / "pytest.log").read_text(errors="replace")
    reported = {m.group(2): int(m.group(1)) for m in _COUNT.finditer(text)}
    subtests = reported.get("subtests passed", 0)
    _require(type(subtests) is int and subtests >= 0)
    _require(reported["passed"] + reported["skipped"] == physical and reported["skipped"] == 1)
    _require(counts["tests"] == physical + subtests)
    return {
        "gate_exit_codes": gates,
        "phase_exit_codes": phases,
        "summary_sha256": _h(summary_raw),
        "junit_sha256": _h(junit_raw),
        "junit": counts,
        "junit_unique_case_records": unique,
        "junit_physical_case_records": physical,
        "junit_runtime_subtest_records": subtests,
        "pytest_reported_counts": reported,
        "collection_metadata": inventories[0],
        "execution_collection_metadata": inventories[1],
    }


def check_shards(out, expected_n):
    """Additive checks of the sharded run (shards.metadata.json); the official observer does not read it."""
    out = Path(out)
    meta = json.loads((out / "shards.metadata.json").read_bytes())
    _require(meta["n"] == expected_n)
    _require(
        len(meta["shard_exits"]) == expected_n and all(type(v) is int and v == 0 for v in meta["shard_exits"])
    )
    inventory = json.loads((out / "collection.metadata.json").read_bytes())
    _require(sum(meta["shard_selected_counts"]) == inventory["collected"])
    plan_raw = (out / "shard-plan.json").read_bytes()
    _require(meta["plan_sha256"] == _h(plan_raw))
    return meta


def check_shard_containers_absent(out, expected_n, docker="docker"):
    """Each recorded shard container id must be gone from Docker (explicit 'No such object' answers only)."""
    out = Path(out)
    for index in range(expected_n):
        cid = (out / f"phase-python-s{index:02d}.cid").read_text().strip()
        _require(re.fullmatch("[a-f0-9]{12,64}", cid))
        proc = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit  # nosec B603 - fixed docker inspect argv, no shell
            [docker, "inspect", "--format", "{{json .State}}", cid],
            capture_output=True,
            text=True,
            timeout=15,
        )
        _require(
            proc.returncode != 0 and ("No such object" in proc.stderr or "No such container" in proc.stderr)
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--collection-sha256", required=True)
    parser.add_argument("--nodeids-sha256", required=True)
    parser.add_argument("--collected", required=True, type=int)
    parser.add_argument("--skip-sha256", required=True)
    parser.add_argument("--skip-function", required=True)
    parser.add_argument("--skip-module", required=True)
    parser.add_argument("--qa-dir", type=Path)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--check-containers", action="store_true")
    args = parser.parse_args(argv)
    pins = Pins(
        candidate=args.candidate,
        tree=args.tree,
        collection_sha256=args.collection_sha256,
        nodeids_sha256=args.nodeids_sha256,
        collected=args.collected,
        skip_sha256=args.skip_sha256,
        skip_function=args.skip_function,
        skip_module=args.skip_module,
    )
    record = check_out_dir(args.out, pins, args.qa_dir)
    if args.shards:
        record["shards"] = check_shards(args.out, args.shards)
        if args.check_containers:
            check_shard_containers_absent(args.out, args.shards)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
