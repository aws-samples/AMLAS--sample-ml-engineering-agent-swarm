"""Shard plan for the sharded full CI: deterministic, duration-weighted, exact cover of the collection.

Host-side tool: standard library only, Python 3.9 compatible. It reads the collection inventory that the
unchanged ``collection`` gate produced and a pinned durations asset, and writes ``shard-plan.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET  # nosemgrep: use-defused-xml  # nosec B405 - parses the JUnit file pytest wrote in this run's own container; a malformed file is a refusal
from pathlib import Path

PLAN_SCHEMA = "amlas-ci-shard-plan-v1"
DURATIONS_SCHEMA = "amlas-ci-shard-durations-v1"
ALGORITHM = "module-lpt-v1"
DETAIL_SECONDS = 1.0
UNKNOWN_SECONDS = 0.5
DEFAULT_SPLIT_FRACTION = 0.5
DEFAULT_ISOLATE_SECONDS = 300.0

_ILLEGAL_XML = re.compile(r"[^\t\n\r\x20-\x7e\x80-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


class PlanError(Exception):
    """The plan cannot be built or does not match the inputs it is bound to."""


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_collection_log(text):
    """Node ids of ``pytest --collect-only -q``: every line before the first blank line."""
    ids = []
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if line == "":
            break
        ids.append(line)
    return ids


def nodeids_digest(ids):
    raw = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256_bytes(raw)


def collection_metadata_bytes(ids):
    """Byte-exact serialization written by the CI collection plugin."""
    body = {"collected": len(ids), "nodeids_sha256": nodeids_digest(ids)}
    return (json.dumps(body, sort_keys=True) + "\n").encode()


def verify_collection(ids, metadata_bytes):
    if collection_metadata_bytes(ids) != metadata_bytes:
        raise PlanError("the collected ids do not reproduce the reference collection inventory")


def module_of(nodeid):
    return nodeid.split("::", 1)[0]


def _escape_name(text):
    def repl(match):
        value = ord(match.group())
        return f"#x{value:02X}" if value <= 0xFF else f"#x{value:04X}"

    return _ILLEGAL_XML.sub(repl, text)


def junit_key(nodeid):
    """(classname, name) that pytest's junitxml writer derives from a node id."""
    path, bracket, params = nodeid.partition("[")
    names = path.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    names[-1] += bracket + params
    return ".".join(names[:-1]), _escape_name(names[-1])


def junit_key_index(ids):
    index = {}
    for nodeid in ids:
        key = junit_key(nodeid)
        if key in index and index[key] != nodeid:
            raise PlanError(f"two node ids share the JUnit key {key!r}: {index[key]!r} and {nodeid!r}")
        index[key] = nodeid
    return index


def read_junit_durations(junit_bytes, ids):
    index = junit_key_index(ids)
    seen = {}
    for case in ET.fromstring(junit_bytes).iter("testcase"):  # nosec B314 - parses the JUnit file pytest wrote in this run's own container; a malformed file is a refusal
        nodeid = index.get((case.get("classname"), case.get("name")))
        if nodeid is None:
            continue
        if nodeid in seen:
            raise PlanError(f"JUnit lists {nodeid!r} twice")
        seen[nodeid] = float(case.get("time", "0"))
    missing = [nid for nid in ids if nid not in seen]
    if missing:
        raise PlanError(f"JUnit lacks {len(missing)} collected ids, first {missing[0]!r}")
    return seen


def build_durations_asset(junit_bytes, ids):
    """Per module: test count, total seconds and the seconds of every test of at least DETAIL_SECONDS."""
    seen = read_junit_durations(junit_bytes, ids)
    modules = {}
    for nodeid in ids:
        module = module_of(nodeid)
        entry = modules.setdefault(module, {"count": 0, "total": 0.0, "tests": {}})
        entry["count"] += 1
        entry["total"] += seen[nodeid]
        if seen[nodeid] >= DETAIL_SECONDS:
            entry["tests"][nodeid[len(module) + 2 :]] = round(seen[nodeid], 3)
    for entry in modules.values():
        entry["total"] = round(entry["total"], 3)
    return {
        "schema": DURATIONS_SCHEMA,
        "detail_seconds": DETAIL_SECONDS,
        "source_junit_sha256": sha256_bytes(junit_bytes),
        "collected": len(ids),
        "modules": modules,
    }


def asset_bytes(asset):
    return (json.dumps(asset, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def duration_of(asset, nodeid):
    module = module_of(nodeid)
    entry = asset["modules"].get(module)
    if entry is None:
        return UNKNOWN_SECONDS
    known = entry["tests"].get(nodeid[len(module) + 2 :])
    if known is not None:
        return float(known)
    others = entry["count"] - len(entry["tests"])
    if others <= 0:
        return UNKNOWN_SECONDS
    remainder = entry["total"] - sum(entry["tests"].values())
    return max(remainder / others, 0.001)


def build_plan(
    ids,
    asset,
    n,
    *,
    durations_sha256,
    split_fraction=DEFAULT_SPLIT_FRACTION,
    isolate_seconds=DEFAULT_ISOLATE_SECONDS,
    run_nonce=None,
):
    """Module units, heavy modules split per test, very long tests alone, LPT over the remaining shards."""
    if not isinstance(n, int) or n < 1:
        raise PlanError("the shard count must be a positive integer")
    if len(set(ids)) != len(ids):
        raise PlanError("the collection lists a node id twice")
    seconds = [duration_of(asset, nodeid) for nodeid in ids]
    threshold = split_fraction * sum(seconds) / n
    by_module = {}
    for position, nodeid in enumerate(ids):
        by_module.setdefault(module_of(nodeid), []).append(position)
    units = []
    for module, positions in by_module.items():
        weight = sum(seconds[p] for p in positions)
        if n > 1 and len(positions) > 1 and weight > threshold:
            units.extend(("test", ids[p], [p], seconds[p], p) for p in positions)
        else:
            units.append(("module", module, positions, weight, positions[0]))
    long_tests = sorted(
        (u for u in units if u[0] == "test" and u[3] >= isolate_seconds), key=lambda u: (-u[3], u[4])
    )
    isolated = long_tests[: max(n - 1, 0)]
    isolated_ids = {id(u) for u in isolated}
    rest = sorted((u for u in units if id(u) not in isolated_ids), key=lambda u: (-u[3], u[4]))
    rest_shards = n - len(isolated)
    if rest_shards < 1 or len(rest) < rest_shards:
        raise PlanError(f"{n} shards requested but only {len(units)} units exist")
    shard_of = [-1] * len(ids)
    loads = [0.0] * n
    counts = [0] * n
    unit_counts = [0] * n
    for index, unit in enumerate(isolated):
        for position in unit[2]:
            shard_of[position] = index
        loads[index] += unit[3]
        counts[index] += len(unit[2])
        unit_counts[index] += 1
    first = len(isolated)
    for unit in rest:
        target = min(range(first, n), key=lambda j: (loads[j], j))
        for position in unit[2]:
            shard_of[position] = target
        loads[target] += unit[3]
        counts[target] += len(unit[2])
        unit_counts[target] += 1
    return {
        "schema": PLAN_SCHEMA,
        "algorithm": ALGORITHM,
        "n": n,
        "parameters": {"split_fraction": split_fraction, "isolate_seconds": isolate_seconds},
        "durations_sha256": durations_sha256,
        "run_nonce": run_nonce,
        "collection": {"collected": len(ids), "nodeids_sha256": nodeids_digest(ids)},
        "ids": list(ids),
        "shard_of": shard_of,
        "shards": [
            {
                "index": i,
                "ids_count": counts[i],
                "unit_count": unit_counts[i],
                "planned_seconds": round(loads[i], 3),
                "isolated": i < len(isolated),
            }
            for i in range(n)
        ],
    }


def plan_bytes(plan):
    return (json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def plan_sha256(plan):
    return sha256_bytes(plan_bytes(plan))


def load_plan(path, expected_sha256=None):
    """Read a plan file and check it against its pin and its own invariants."""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and sha256_bytes(raw) != expected_sha256:
        raise PlanError("the plan file does not match its pinned sha256")
    plan = json.loads(raw)
    if plan.get("schema") != PLAN_SCHEMA:
        raise PlanError("unknown plan schema")
    n, ids, shard_of = plan.get("n"), plan.get("ids"), plan.get("shard_of")
    if type(n) is not int or n < 1 or not isinstance(ids, list) or not isinstance(shard_of, list):
        raise PlanError("plan fields are malformed")
    if not isinstance(plan.get("run_nonce"), (str, type(None))):
        raise PlanError("plan run nonce is malformed")
    if len(ids) != len(shard_of) or len(set(ids)) != len(ids):
        raise PlanError("plan ids and assignments differ in length or repeat an id")
    if any(type(s) is not int or not 0 <= s < n for s in shard_of) or set(shard_of) != set(range(n)):
        raise PlanError("plan assignments are out of range or leave a shard empty")
    if plan.get("collection") != {"collected": len(ids), "nodeids_sha256": nodeids_digest(ids)}:
        raise PlanError("plan ids do not match the plan's own collection digest")
    if plan_bytes(plan) != raw:
        raise PlanError("plan file is not in canonical form")
    return plan


def assigned_ids(plan, index):
    shard_of = plan["shard_of"]
    return [nodeid for position, nodeid in enumerate(plan["ids"]) if shard_of[position] == index]


def _write_new(path, raw):
    with open(path, "xb") as handle:
        handle.write(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dur = sub.add_parser("durations", help="build the durations asset from a baseline JUnit file")
    dur.add_argument("--junit", required=True, type=Path)
    dur.add_argument("--collection-log", required=True, type=Path)
    dur.add_argument("--out", required=True, type=Path)
    pl = sub.add_parser("plan", help="build the shard plan")
    pl.add_argument("--collection-log", required=True, type=Path)
    pl.add_argument("--collection-metadata", required=True, type=Path)
    pl.add_argument("--durations", required=True, type=Path)
    pl.add_argument("--shards", required=True, type=int)
    pl.add_argument("--split-fraction", type=float, default=DEFAULT_SPLIT_FRACTION)
    pl.add_argument("--isolate-seconds", type=float, default=DEFAULT_ISOLATE_SECONDS)
    pl.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        ids = parse_collection_log(args.collection_log.read_text(encoding="utf-8"))
        if args.command == "durations":
            asset = build_durations_asset(args.junit.read_bytes(), ids)
            _write_new(args.out, asset_bytes(asset))
            return 0
        verify_collection(ids, args.collection_metadata.read_bytes())
        raw = args.durations.read_bytes()
        asset = json.loads(raw)
        if asset.get("schema") != DURATIONS_SCHEMA:
            raise PlanError("unknown durations schema")
        plan = build_plan(
            ids,
            asset,
            args.shards,
            durations_sha256=sha256_bytes(raw),
            split_fraction=args.split_fraction,
            isolate_seconds=args.isolate_seconds,
        )
        _write_new(args.out, plan_bytes(plan))
        return 0
    except PlanError as exc:
        print(f"full_ci_shard_plan: refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
