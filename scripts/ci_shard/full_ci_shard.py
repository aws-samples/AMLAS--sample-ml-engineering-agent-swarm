"""pytest plugin for one shard of the sharded full CI (load with ``-p full_ci_shard``).

It keeps exactly the items the plan assigns to this shard, refuses every option that would select, reorder or
stop the run differently, and writes ``shard-report.json`` and ``execution-collection.metadata.json`` into
``$AMLAS_CI_SHARD_OUT``. It changes no test behavior. Environment: AMLAS_CI_SHARD_PLAN, AMLAS_CI_SHARD_PLAN_SHA256,
AMLAS_CI_SHARD_INDEX, AMLAS_CI_SHARD_OUT.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

REPORT_SCHEMA = "amlas-ci-shard-report-v1"
PLAN_SCHEMA = "amlas-ci-shard-plan-v1"
_REQUIRED_ENV = (
    "AMLAS_CI_SHARD_PLAN",
    "AMLAS_CI_SHARD_PLAN_SHA256",
    "AMLAS_CI_SHARD_INDEX",
    "AMLAS_CI_SHARD_OUT",
)
_STATE = {}


def _refuse(message):
    raise pytest.UsageError("full_ci_shard: " + message)


def _digest(ids):
    raw = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _metadata_bytes(ids):
    return (
        json.dumps({"collected": len(ids), "nodeids_sha256": _digest(ids)}, sort_keys=True) + "\n"
    ).encode()


def _write_atomic(path, raw):
    temp = path.with_name("." + path.name + ".tmp-" + str(os.getpid()))
    temp.write_bytes(raw)
    os.replace(temp, path)


def _cgroup_peak():
    for name in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return int(Path(name).read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def _classify(reports):
    failed = [r for r in reports if r[1] == "failed"]
    if failed:
        return "failed" if any(r[0] == "call" for r in failed) else "error"
    if not any(r[0] == "teardown" for r in reports):
        return "incomplete"
    if any(r[1] == "skipped" for r in reports):
        return "xfailed" if any(r[2] for r in reports) else "skipped"
    return "xpassed" if any(r[2] for r in reports) else "passed"


def pytest_configure(config):
    missing = [key for key in _REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        _refuse("missing environment " + ", ".join(missing))
    opt = config.option
    refused = []
    if getattr(opt, "keyword", ""):
        refused.append("-k")
    if getattr(opt, "markexpr", ""):
        refused.append("-m")
    if getattr(opt, "deselect", None):
        refused.append("--deselect")
    if getattr(opt, "ignore", None) or getattr(opt, "ignore_glob", None):
        refused.append("--ignore")
    for attr, flag in (
        ("lf", "--lf"),
        ("failedfirst", "--ff"),
        ("newfirst", "--nf"),
        ("stepwise", "--sw"),
        ("stepwise_skip", "--stepwise-skip"),
        ("collectonly", "--collect-only"),
    ):
        if getattr(opt, attr, False):
            refused.append(flag)
    if getattr(opt, "maxfail", 0):
        refused.append("--maxfail/-x")
    if refused:
        _refuse("refuses selection options " + ", ".join(refused))
    plan_path = Path(os.environ["AMLAS_CI_SHARD_PLAN"])
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ["AMLAS_CI_SHARD_PLAN_SHA256"]:
        _refuse("the plan file does not match its pinned sha256")
    plan = json.loads(raw)
    ids, shard_of = plan.get("ids"), plan.get("shard_of")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or not isinstance(ids, list)
        or not isinstance(shard_of, list)
        or len(ids) != len(shard_of)
    ):
        _refuse("the plan is malformed")
    if plan["collection"] != {"collected": len(ids), "nodeids_sha256": _digest(ids)}:
        _refuse("the plan ids do not match the plan's collection digest")
    try:
        index = int(os.environ["AMLAS_CI_SHARD_INDEX"])
    except ValueError:
        _refuse("AMLAS_CI_SHARD_INDEX is not an integer")
    if not 0 <= index < plan["n"]:
        _refuse(f"shard index {index} is outside 0..{plan['n'] - 1}")
    out = Path(os.environ["AMLAS_CI_SHARD_OUT"])
    if not out.is_dir():
        _refuse("AMLAS_CI_SHARD_OUT is not a directory")
    _STATE.clear()
    _STATE.update(
        plan=plan,
        plan_sha256=os.environ["AMLAS_CI_SHARD_PLAN_SHA256"],
        index=index,
        out=out,
        shard_of={nodeid: shard_of[position] for position, nodeid in enumerate(ids)},
        reports={},
        started=time.monotonic(),
        selected=[],
        full=None,
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    full = [item.nodeid for item in items]
    summary = {"collected": len(full), "nodeids_sha256": _digest(full)}
    if summary != _STATE["plan"]["collection"]:
        _refuse("the collection differs from the plan's collection (the suite changed after planning)")
    _STATE["full"] = summary
    mine = [item for item in items if _STATE["shard_of"][item.nodeid] == _STATE["index"]]
    items[:] = mine
    _STATE["selected"] = [item.nodeid for item in mine]
    _write_atomic(_STATE["out"] / "execution-collection.metadata.json", _metadata_bytes(_STATE["selected"]))


def pytest_runtest_logreport(report):
    if getattr(report, "context", None) is not None:
        return
    _STATE["reports"].setdefault(report.nodeid, []).append(
        (report.when, report.outcome, hasattr(report, "wasxfail"))
    )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if not _STATE or _STATE.get("full") is None:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = getattr(reporter, "stats", {}) or {}
    counts = {
        key: len([r for r in value if getattr(r, "count_towards_summary", True)])
        for key, value in stats.items()
        if key
    }
    counts = {key: value for key, value in counts.items() if value}
    ledger = {nodeid: _classify(reports) for nodeid, reports in _STATE["reports"].items()}
    selected = _STATE["selected"]
    report = {
        "schema": REPORT_SCHEMA,
        "index": _STATE["index"],
        "plan_sha256": _STATE["plan_sha256"],
        "collection": _STATE["full"],
        "selected": {"collected": len(selected), "nodeids_sha256": _digest(selected)},
        "ledger": ledger,
        "counts": counts,
        "exitstatus": int(exitstatus),
        "wall_seconds": round(time.monotonic() - _STATE["started"], 3),
        "memory_peak_bytes": _cgroup_peak(),
    }
    _write_atomic(_STATE["out"] / "shard-report.json", (json.dumps(report, sort_keys=True) + "\n").encode())
