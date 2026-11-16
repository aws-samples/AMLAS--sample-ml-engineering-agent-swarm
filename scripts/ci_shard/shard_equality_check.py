"""Compare a sharded full-CI ``out/`` with a single-process baseline ``out/``: any difference exits nonzero.

Checked: both inventory files byte for byte, the ordered (classname, name, status) list of the JUnit file, its suite
totals, the skip identity, the pytest log counts the observer reads, every gate exit code, the phase exit codes and
the all-gates flag. Standard library only, Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET  # nosemgrep: use-defused-xml  # nosec B405 - parses the JUnit file pytest wrote in this run's own container; a malformed file is a refusal
from pathlib import Path

_COUNT = re.compile(r"([0-9]+) (passed|skipped|subtests passed)")


def _junit(path):
    root = ET.fromstring(Path(path).read_bytes())  # nosec B314 - parses the JUnit file pytest wrote in this run's own container; a malformed file is a refusal
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    totals = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    rows = []
    skips = []
    for case in root.iter("testcase"):
        status = "passed"
        for tag in ("failure", "error", "skipped"):
            if case.find(tag) is not None:
                status = tag
                break
        rows.append((case.get("classname"), case.get("name"), status))
        skipped = case.find("skipped")
        if skipped is not None:
            digest = hashlib.sha256((skipped.get("message") or "").encode()).hexdigest()
            skips.append((case.get("classname"), case.get("name"), digest))
    return totals, rows, skips


def _log_counts(path):
    text = Path(path).read_text(errors="replace")
    return {m.group(2): int(m.group(1)) for m in _COUNT.finditer(text)}


def compare(baseline, sharded):
    baseline, sharded = Path(baseline), Path(sharded)
    differences = []
    checks = {}

    def note(name, ok, detail):
        checks[name] = {"equal": ok, "detail": detail}
        if not ok:
            differences.append(f"{name}: {detail}")

    base_inventory = (baseline / "collection.metadata.json").read_bytes()
    for name in ("collection.metadata.json", "execution-collection.metadata.json"):
        raw = (sharded / name).read_bytes()
        note(
            name,
            raw == base_inventory,
            f"sha256 {hashlib.sha256(raw).hexdigest()} vs baseline {hashlib.sha256(base_inventory).hexdigest()}",
        )
    b_totals, b_rows, b_skips = _junit(baseline / "full-pytest.xml")
    s_totals, s_rows, s_skips = _junit(sharded / "full-pytest.xml")
    first = next((i for i in range(min(len(b_rows), len(s_rows))) if b_rows[i] != s_rows[i]), None)
    note(
        "junit testcase list",
        b_rows == s_rows,
        f"{len(s_rows)} vs {len(b_rows)} rows, first difference at {first}"
        if b_rows != s_rows
        else f"{len(s_rows)} rows equal in order",
    )
    note("junit suite totals", b_totals == s_totals, f"{s_totals} vs baseline {b_totals}")
    note("junit skip identity", b_skips == s_skips, f"{s_skips} vs baseline {b_skips}")
    b_log, s_log = _log_counts(baseline / "pytest.log"), _log_counts(sharded / "pytest.log")
    note("pytest log counts", b_log == s_log, f"{s_log} vs baseline {b_log}")
    b_sum = json.loads((baseline / "full-ci-result.metadata.json").read_bytes())
    s_sum = json.loads((sharded / "full-ci-result.metadata.json").read_bytes())
    for name in sorted(b_sum["gate_exit_codes"]):
        b_raw = (baseline / f"{name}.exit").read_text().strip()
        s_file = sharded / f"{name}.exit"
        s_raw = s_file.read_text().strip() if s_file.is_file() else "missing"
        note(
            f"gate exit {name}",
            b_raw == s_raw and b_sum["gate_exit_codes"].get(name) == s_sum["gate_exit_codes"].get(name),
            f"{s_raw} vs baseline {b_raw}",
        )
    note(
        "gate exit codes (summary)",
        b_sum["gate_exit_codes"] == s_sum["gate_exit_codes"],
        f"{s_sum['gate_exit_codes']}",
    )
    note(
        "phase exit codes",
        b_sum["phase_exit_codes"] == s_sum["phase_exit_codes"],
        f"{s_sum['phase_exit_codes']} vs {b_sum['phase_exit_codes']}",
    )
    note(
        "all_required_gates_passed",
        b_sum["all_required_gates_passed"] == s_sum["all_required_gates_passed"],
        str(s_sum["all_required_gates_passed"]),
    )
    note("summary junit totals", b_sum["junit"] == s_sum["junit"], f"{s_sum['junit']} vs {b_sum['junit']}")
    return {"equal": not differences, "differences": differences, "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("sharded", type=Path)
    args = parser.parse_args(argv)
    report = compare(args.baseline, args.sharded)
    print(json.dumps(report, sort_keys=True, indent=1))
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
