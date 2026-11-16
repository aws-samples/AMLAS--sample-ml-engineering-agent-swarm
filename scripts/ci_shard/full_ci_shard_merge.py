"""Merge shard outputs into the files the single full-CI pipeline writes and the terminal observer reads.

Host-side tool: standard library only, Python 3.9 compatible. Reads ``out/collection.metadata.json``,
``out/shard-plan.json`` and ``out/shards/sNN/*``; writes ``full-pytest.xml``, ``pytest.log``, ``pytest.exit``,
``execution-collection.metadata.json`` and ``shards.metadata.json`` into ``out/`` only after every check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET  # nosemgrep: use-defused-xml  # nosec B405 - parses the JUnit file pytest wrote in this run's own container; a malformed file is a refusal
from pathlib import Path

import full_ci_shard_plan as shard_plan

REPORT_SCHEMA = "amlas-ci-shard-report-v1"
SHARDS_SCHEMA = "amlas-ci-shards-v1"
KNOWN_TYPES = (
    "failed",
    "passed",
    "skipped",
    "deselected",
    "xfailed",
    "xpassed",
    "warnings",
    "error",
    "subtests passed",
    "subtests failed",
    "subtests skipped",
)
OBSERVER_KEYS = ("passed", "skipped", "subtests passed")
_OBSERVER_COUNT = re.compile(r"([0-9]+) (passed|skipped|subtests passed)")
_SUMMARY_ITEM = re.compile(
    r"(\d+) (failed|passed|skipped|deselected|xfailed|xpassed|warnings?|errors?|subtests passed|subtests failed|subtests skipped)"
)
_FINAL_LINE = re.compile(r" in [0-9.]+s( \([0-9:]+\))?$")
_NOT_COMPARED = ("warnings", "deselected")
_REQUIRED_FILES = (
    "junit.xml",
    "pytest.log",
    "pytest.exit",
    "shard-report.json",
    "execution-collection.metadata.json",
)


class MergeError(Exception):
    """The shard outputs do not prove the full collection ran exactly once."""


def parse_summary_counts(line):
    counts = {}
    for match in _SUMMARY_ITEM.finditer(line):
        key = {"warning": "warnings", "errors": "error"}.get(match.group(2), match.group(2))
        counts[key] = int(match.group(1))
    return counts


def final_summary_line(text):
    """Last line shaped like pytest's summary; output printed after it (interpreter shutdown) is ignored."""
    for line in reversed(text.split("\n")):
        if _FINAL_LINE.search(line) and _SUMMARY_ITEM.search(line):
            return line
    return None


def format_summary_line(counts, seconds, zero_keys=()):
    parts = []
    for key in KNOWN_TYPES:
        value = counts.get(key, 0)
        if value:
            word = {"warnings": "warning", "error": "error"}.get(key)
            if word:
                parts.append(f"{value} {word}{'s' if value != 1 else ''}")
            else:
                parts.append(f"{value} {key}")
        elif key in zero_keys:
            parts.append(f"0 {key}")
    if not parts:
        parts = ["no tests ran"]
    text = f"{seconds:.2f}s"
    if seconds >= 60:
        whole = int(seconds)
        text += f" ({whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d})"
    return ", ".join(parts) + " in " + text


def _read(path, label):
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise MergeError(f"{label} is unreadable: {exc.__class__.__name__}") from exc


def _load_shard(out, index, plan, plan_sha, reference):
    directory = out / "shards" / f"s{index:02d}"
    if not directory.is_dir():
        raise MergeError(f"shard {index}: directory is missing")
    for name in _REQUIRED_FILES:
        if not (directory / name).is_file():
            raise MergeError(f"shard {index}: {name} is missing")
    try:
        exit_code = int((directory / "pytest.exit").read_text().strip())
    except ValueError as exc:
        raise MergeError(f"shard {index}: pytest.exit is not an integer") from exc
    try:
        report = json.loads((directory / "shard-report.json").read_bytes())
    except ValueError as exc:
        raise MergeError(f"shard {index}: shard-report.json is not JSON") from exc
    if report.get("schema") != REPORT_SCHEMA or report.get("index") != index:
        raise MergeError(f"shard {index}: shard-report.json has the wrong schema or index")
    if report.get("plan_sha256") != plan_sha:
        raise MergeError(f"shard {index}: built on a different plan")
    if report.get("collection") != json.loads(reference):
        raise MergeError(f"shard {index}: its full collection differs from the reference inventory")
    if report.get("exitstatus") != exit_code:
        raise MergeError(f"shard {index}: pytest.exit differs from the exit status in its report")
    assigned = shard_plan.assigned_ids(plan, index)
    selected = {"collected": len(assigned), "nodeids_sha256": shard_plan.nodeids_digest(assigned)}
    if report.get("selected") != selected:
        raise MergeError(f"shard {index}: its selected set differs from the plan")
    if (
        directory / "execution-collection.metadata.json"
    ).read_bytes() != shard_plan.collection_metadata_bytes(assigned):
        raise MergeError(f"shard {index}: its execution inventory differs from the plan")
    ledger = report.get("ledger")
    if not isinstance(ledger, dict) or not isinstance(report.get("counts"), dict):
        raise MergeError(f"shard {index}: shard-report.json lacks its ledger or counts")
    return {
        "index": index,
        "dir": directory,
        "exit": exit_code,
        "report": report,
        "assigned": assigned,
        "junit_bytes": (directory / "junit.xml").read_bytes(),
        "log": (directory / "pytest.log").read_text(encoding="utf-8", errors="replace"),
    }


def _parse_junit(shard, key_index, plan_position):
    try:
        root = ET.fromstring(shard["junit_bytes"])  # nosec B314 - parses the JUnit file pytest wrote in this run's own container; a malformed file is a refusal
    except ET.ParseError as exc:
        raise MergeError(f"shard {shard['index']}: junit.xml is not well-formed") from exc
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    if not suites:
        raise MergeError(f"shard {shard['index']}: junit.xml has no testsuite")
    attrs = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    entries = []
    anchor = -1
    for seq, case in enumerate(root.iter("testcase")):
        nodeid = key_index.get((case.get("classname"), case.get("name")))
        case.tail = None
        if nodeid is not None:
            anchor = plan_position[nodeid]
            entries.append((anchor, 0, shard["index"], seq, nodeid, case))
        elif case.find("failure") is not None or case.find("error") is not None:
            entries.append((anchor, 1, shard["index"], seq, None, case))
        else:
            raise MergeError(
                f"shard {shard['index']}: testcase {case.get('classname')}::{case.get('name')} is not in the plan"
            )
    times = [float(s.get("time", 0)) for s in suites]
    stamps = [s.get("timestamp") for s in suites if s.get("timestamp")]
    return attrs, entries, max(times), min(stamps) if stamps else ""


def merge(out_dir, *, wall_seconds=None):
    out = Path(out_dir)
    reference = _read(out / "collection.metadata.json", "collection.metadata.json")
    plan_raw = _read(out / "shard-plan.json", "shard-plan.json")
    try:
        plan = shard_plan.load_plan(out / "shard-plan.json")
    except (shard_plan.PlanError, ValueError) as exc:
        raise MergeError(f"plan: {exc}") from exc
    plan_sha = shard_plan.sha256_bytes(plan_raw)
    if shard_plan.collection_metadata_bytes(plan["ids"]) != reference:
        raise MergeError("inventory: the plan's ids do not reproduce collection.metadata.json")
    try:
        key_index = shard_plan.junit_key_index(plan["ids"])
    except shard_plan.PlanError as exc:
        raise MergeError(f"plan: {exc}") from exc
    plan_position = {nodeid: i for i, nodeid in enumerate(plan["ids"])}
    n = plan["n"]
    shards = [_load_shard(out, i, plan, plan_sha, reference) for i in range(n)]
    for shard in shards:
        shard["attrs"], shard["entries"], shard["suite_time"], shard["stamp"] = _parse_junit(
            shard, key_index, plan_position
        )
        if not shard["entries"] and not shard["report"]["ledger"]:
            raise MergeError(f"shard {shard['index']}: empty shard, nothing executed (exit {shard['exit']})")
    executed = {}
    for shard in shards:
        for _, _, _, _, nodeid, _ in shard["entries"]:
            if nodeid is not None:
                executed.setdefault(nodeid, []).append(shard["index"])
    duplicated = [nid for nid, where in executed.items() if len(where) > 1]
    if duplicated:
        raise MergeError(f"duplicate: {len(duplicated)} ids executed twice, first {duplicated[0]!r}")
    for nodeid, where in executed.items():
        planned = plan["shard_of"][plan_position[nodeid]]
        if where[0] != planned:
            raise MergeError(
                f"wrong shard: {nodeid!r} is planned for shard {planned} but ran in shard {where[0]}"
            )
    missing = [nid for nid in plan["ids"] if nid not in executed]
    if missing:
        raise MergeError(f"missing: {len(missing)} planned ids were not executed, first {missing[0]!r}")
    totals = {}
    for shard in shards:
        index = shard["index"]
        ledger = shard["report"]["ledger"]
        junit_ids = {e[4] for e in shard["entries"] if e[4] is not None}
        if set(ledger) != junit_ids:
            raise MergeError(f"shard {index}: the outcome ledger and junit.xml list different ids")
        ledger_skipped = sum(1 for v in ledger.values() if v == "skipped")
        ledger_xfailed = sum(1 for v in ledger.values() if v == "xfailed")
        counts = shard["report"]["counts"]
        if counts.get("skipped", 0) != ledger_skipped:
            raise MergeError(
                f"shard {index}: skipped count differs (ledger {ledger_skipped}, report {counts.get('skipped', 0)})"
            )
        if counts.get("xfailed", 0) != ledger_xfailed:
            raise MergeError(
                f"shard {index}: xfailed count differs (ledger {ledger_xfailed}, report {counts.get('xfailed', 0)})"
            )
        # pytest's JUnit writer reports an expected failure as a skipped testcase
        if shard["attrs"]["skipped"] != ledger_skipped + ledger_xfailed:
            raise MergeError(
                f"shard {index}: skipped count differs (junit {shard['attrs']['skipped']}, ledger skipped"
                f" {ledger_skipped} plus xfailed {ledger_xfailed})"
            )
        if counts.get("passed", 0) != sum(1 for v in ledger.values() if v == "passed"):
            raise MergeError(f"shard {index}: passed count differs from the ledger")
        line = final_summary_line(shard["log"])
        if line is None:
            raise MergeError(f"shard {index}: pytest.log lacks the final summary line")
        logged = parse_summary_counts(line)
        for key in KNOWN_TYPES:
            if key not in _NOT_COMPARED and logged.get(key, 0) != counts.get(key, 0):
                raise MergeError(f"shard {index}: pytest.log summary differs from the report for {key!r}")
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + int(value)
    failing = (
        any(s["exit"] != 0 for s in shards)
        or any(s["attrs"]["failures"] or s["attrs"]["errors"] for s in shards)
        or any(v in ("failed", "error", "incomplete") for s in shards for v in s["report"]["ledger"].values())
    )
    sums = {k: sum(s["attrs"][k] for s in shards) for k in ("tests", "failures", "errors", "skipped")}
    entries = sorted((e for s in shards for e in s["entries"]), key=lambda e: e[:4])
    physical = len(entries)
    if not failing:
        accounted = sum(totals.get(k, 0) for k in ("passed", "skipped", "xfailed", "xpassed"))
        if accounted != physical:
            raise MergeError("invariant: passed, skipped, xfailed and xpassed differ from the testcase count")
        if sums["tests"] != physical + totals.get("subtests passed", 0):
            raise MergeError("invariant: JUnit tests differ from testcases plus subtests passed")
    wall = (
        float(wall_seconds)
        if wall_seconds is not None
        else max(s["report"].get("wall_seconds", 0.0) for s in shards)
    )
    log_parts = []
    for shard in shards:
        header = (
            f"=== shard {shard['index']:02d} of {n} (exit {shard['exit']}, {len(shard['assigned'])} ids) ==="
        )
        log_parts.append(header + "\n" + shard["log"].rstrip("\n") + "\n")
    earlier = "".join(log_parts)
    leaked = {m.group(2) for m in _OBSERVER_COUNT.finditer(earlier)}
    zero_keys = tuple(k for k in OBSERVER_KEYS if k in leaked and totals.get(k, 0) == 0)
    log_text = earlier + format_summary_line(totals, wall, zero_keys) + "\n"
    reported = {}
    for match in _OBSERVER_COUNT.finditer(log_text):
        reported[match.group(2)] = int(match.group(1))
    for key in OBSERVER_KEYS:
        if reported.get(key, 0) != totals.get(key, 0):
            raise MergeError(
                f"log: the observer would read {key!r} as {reported.get(key, 0)}, expected {totals.get(key, 0)}"
            )
    suite = ET.Element(
        "testsuite",
        {
            "name": "pytest",
            "errors": str(sums["errors"]),
            "failures": str(sums["failures"]),
            "skipped": str(sums["skipped"]),
            "tests": str(sums["tests"]),
            "time": f"{wall:.3f}",
            "timestamp": min((s["stamp"] for s in shards if s["stamp"]), default=""),
            "hostname": "sharded",
        },
    )
    for entry in entries:
        suite.append(entry[5])
    root = ET.Element("testsuites", {"name": "pytest tests"})
    root.append(suite)
    junit = ('<?xml version="1.0" encoding="utf-8"?>' + ET.tostring(root, encoding="unicode")).encode("utf-8")
    meta = {
        "schema": SHARDS_SCHEMA,
        "n": n,
        "algorithm": plan["algorithm"],
        "plan_sha256": plan_sha,
        "shard_exits": [s["exit"] for s in shards],
        "shard_selected_counts": [len(s["assigned"]) for s in shards],
        "shard_planned_seconds": [s["planned_seconds"] for s in plan["shards"]],
        "shard_wall_seconds": [s["report"].get("wall_seconds") for s in shards],
        "shard_memory_peak_bytes": [s["report"].get("memory_peak_bytes") for s in shards],
        "merged_counts": totals,
        "wall_seconds": round(wall, 3),
    }
    exit_code = 1 if failing else 0
    files = {
        "full-pytest.xml": junit,
        "pytest.log": log_text.encode("utf-8"),
        "execution-collection.metadata.json": shard_plan.collection_metadata_bytes(plan["ids"]),
        "shards.metadata.json": (json.dumps(meta, sort_keys=True) + "\n").encode(),
        "pytest.exit": f"{exit_code}\n".encode(),
    }
    staged = []
    try:
        for name, raw in files.items():
            temp = out / f".merge-{os.getpid()}-{name}"
            with open(temp, "xb") as handle:
                handle.write(raw)
            staged.append((temp, out / name))
        for temp, final in staged:
            os.replace(temp, final)
        staged = []
    finally:
        for temp, _ in staged:
            if temp.exists():
                temp.unlink()
    return {
        "exit_code": exit_code,
        "counts": totals,
        "shard_exits": meta["shard_exits"],
        "wall_seconds": wall,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wall-seconds", type=float)
    args = parser.parse_args(argv)
    try:
        result = merge(args.out, wall_seconds=args.wall_seconds)
    except MergeError as exc:
        print(f"full_ci_shard_merge: refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"exit_code": result["exit_code"], "counts": result["counts"]}, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
