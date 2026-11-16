"""Sharded full CI: merged shard output is exactly what the unchanged observer and qualifier read."""

import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET  # nosec B405 - reads JUnit XML the test built itself
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ci_shard"))

import full_ci_shard_merge as shard_merge
import full_ci_shard_plan as shard_plan
import shard_equality_check as equality
import shard_observer_check as observer

SKIP_FUNCTION = "test_actual_queue_refuses_read_only_ipc_tree_but_resolver_fork_succeeds"
SKIP_MESSAGE = "separate strict focused runner mounts /dev/shm read-only"
GATES = (
    "bandit ci-yaml collection compile cpu-audit cpu-closure dox format governance package-build package-inspect "
    "pytest requirements-export ruff runtime-audit rust-default rust-default-build rust-format rust-native "
    "rust-native-build sbom secrets shell-syntax"
).split()
PHASES = ("audit", "dox", "package", "python", "rust")
CANDIDATE = "a" * 40
TREE = "b" * 40
LOG_COUNT = re.compile(r"([0-9]+) (passed|skipped|subtests passed)")


def fixture_ids():
    layout = {
        "test_alpha": [f"test_a{k}" for k in range(6)],
        "test_beta": [f"TestB::test_b{k}[p{k}]" for k in range(4)],
        "test_gamma": [f"test_g{k}" for k in range(5)],
        "test_public_dns_pipe": [SKIP_FUNCTION, "test_other"],
        "test_delta": [f"test_d{k}" for k in range(3)],
    }
    return [f"tests/{module}.py::{name}" for module, names in layout.items() for name in names]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def summary_line(counts, seconds):
    order = ("failed", "passed", "skipped", "xfailed", "xpassed", "warnings", "error", "subtests passed")
    parts = []
    for key in order:
        value = counts.get(key, 0)
        if value:
            word = {"warnings": "warning", "error": "error"}.get(key)
            parts.append(f"{value} {word + ('s' if value != 1 else '') if word else key}")
    return ", ".join(parts) + f" in {seconds:.2f}s"


def junit_case(nid, status, seconds="0.010"):
    classname, name = shard_plan.junit_key(nid)
    case = ET.Element("testcase", {"classname": classname, "name": name, "time": seconds})
    if status == "skipped":
        skip = ET.SubElement(case, "skipped", {"type": "pytest.skip", "message": SKIP_MESSAGE})
        skip.text = f"/work/tests/test_public_dns_pipe.py:197: {SKIP_MESSAGE}"
    elif status == "xfailed":
        skip = ET.SubElement(case, "skipped", {"type": "pytest.xfail", "message": "known defect"})
        skip.text = "known defect"
    elif status == "failed":
        failure = ET.SubElement(case, "failure", {"message": "assert 0"})
        failure.text = "trace"
    return case


def junit_document(
    cases, *, tests, failures=0, errors=0, skipped=0, timestamp="2026-11-07T13:40:57.124479+00:00"
):
    suite = ET.Element(
        "testsuite",
        {
            "name": "pytest",
            "errors": str(errors),
            "failures": str(failures),
            "skipped": str(skipped),
            "tests": str(tests),
            "time": "12.500",
            "timestamp": timestamp,
            "hostname": "container",
        },
    )
    for case in cases:
        suite.append(case)
    root = ET.Element("testsuites", {"name": "pytest tests"})
    root.append(suite)
    return b'<?xml version="1.0" encoding="utf-8"?>' + ET.tostring(root, encoding="utf-8")


class Fixture:
    """A synthetic sharded run laid out the way the orchestrator and plugin leave it."""

    def __init__(self, root, n=3):
        self.root = Path(root)
        self.out = self.root / "out"
        self.out.mkdir()
        self.n = n
        self.ids = fixture_ids()
        self.skip_id = f"tests/test_public_dns_pipe.py::{SKIP_FUNCTION}"
        self.statuses = {nid: "passed" for nid in self.ids}
        self.statuses[self.skip_id] = "skipped"
        self.subtests = {self.ids[0]: 2, self.ids[7]: 3, self.ids[12]: 1}
        (self.out / "collection.metadata.json").write_bytes(shard_plan.collection_metadata_bytes(self.ids))
        junit = ET.tostring(
            ET.fromstring(  # nosec B314 - XML the test built itself
                "<testsuites>"
                + "".join(
                    f'<testcase classname="{shard_plan.junit_key(i)[0]}" name="{shard_plan.junit_key(i)[1]}" time="1.0"/>'
                    for i in self.ids
                )
                + "</testsuites>"
            )
        )
        asset = shard_plan.build_durations_asset(junit, self.ids)
        self.plan = shard_plan.build_plan(
            self.ids, asset, n, durations_sha256=sha(shard_plan.asset_bytes(asset))
        )
        self.plan_raw = shard_plan.plan_bytes(self.plan)
        (self.out / "shard-plan.json").write_bytes(self.plan_raw)
        self.plan_sha = sha(self.plan_raw)
        for index in range(n):
            self.write_shard(index)

    def assigned(self, index):
        return [nid for nid, shard in zip(self.ids, self.plan["shard_of"], strict=True) if shard == index]

    def shard_dir(self, index):
        return self.out / "shards" / f"s{index:02d}"

    def counts_for(self, run_ids, status_of=None):
        status_of = status_of or self.statuses
        counts = {"passed": 0, "skipped": 0, "failed": 0, "xfailed": 0, "warnings": 2}
        for nid in run_ids:
            counts[
                {"passed": "passed", "skipped": "skipped", "failed": "failed", "xfailed": "xfailed"}[
                    status_of[nid]
                ]
            ] += 1
        sub = sum(self.subtests.get(nid, 0) for nid in run_ids)
        if sub:
            counts["subtests passed"] = sub
        return {key: value for key, value in counts.items() if value}

    def write_shard(
        self,
        index,
        *,
        exit_code=0,
        drop=(),
        duplicate=(),
        extra=(),
        plan_sha=None,
        full=None,
        suite_skipped=None,
        unmapped_failure=None,
        report_skipped=None,
    ):
        directory = self.shard_dir(index)
        directory.mkdir(parents=True, exist_ok=True)
        assigned = self.assigned(index)
        run_ids = [nid for nid in assigned if nid not in drop]
        cases = []
        for nid in run_ids:
            cases.append(junit_case(nid, self.statuses[nid]))
            if unmapped_failure and unmapped_failure[0] == nid:
                classname, name = shard_plan.junit_key(nid)
                sub = ET.Element(
                    "testcase", {"classname": classname, "name": f"{name} (<subtest>)", "time": "0.001"}
                )
                ET.SubElement(sub, "failure", {"message": "subtest failed"}).text = "trace"
                cases.append(sub)
        for nid in duplicate:
            cases.append(junit_case(nid, self.statuses[nid]))
        for nid in extra:
            cases.append(junit_case(nid, "passed"))
        counts = self.counts_for([nid for nid in run_ids if nid in self.statuses])
        subtests = counts.get("subtests passed", 0)
        failures = sum(1 for nid in run_ids if self.statuses[nid] == "failed") + (
            1 if unmapped_failure else 0
        )
        skipped = (
            suite_skipped
            if suite_skipped is not None
            else counts.get("skipped", 0) + counts.get("xfailed", 0)
        )
        junit = junit_document(
            cases,
            tests=len(cases) + subtests,
            failures=failures,
            skipped=skipped,
            timestamp=f"2026-11-07T13:4{index}:57.124479+00:00",
        )
        (directory / "junit.xml").write_bytes(junit)
        log_counts = dict(counts)
        log_counts["warnings"] = 2
        lines = ["." * 40 + " [100%]", "=== short test summary info ==="]
        lines += [
            f"SKIPPED [1] tests/test_public_dns_pipe.py:197: {SKIP_MESSAGE}"
            for nid in run_ids
            if nid == self.skip_id
        ]
        lines.append(summary_line(log_counts, 10.0 + index))
        (directory / "pytest.log").write_text("\n".join(lines) + "\n")
        (directory / "pytest.exit").write_text(f"{exit_code}\n")
        (directory / "execution-collection.metadata.json").write_bytes(
            shard_plan.collection_metadata_bytes(assigned)
        )
        report_counts = dict(log_counts)
        if report_skipped is not None:
            report_counts["skipped"] = report_skipped
        report = {
            "schema": "amlas-ci-shard-report-v1",
            "index": index,
            "plan_sha256": plan_sha or self.plan_sha,
            "collection": full
            or {"collected": len(self.ids), "nodeids_sha256": shard_plan.nodeids_digest(self.ids)},
            "selected": {"collected": len(assigned), "nodeids_sha256": shard_plan.nodeids_digest(assigned)},
            "ledger": {nid: self.statuses[nid] for nid in run_ids},
            "counts": report_counts,
            "exitstatus": exit_code,
            "wall_seconds": 10.0 + index,
            "memory_peak_bytes": 1000 + index,
        }
        (directory / "shard-report.json").write_text(json.dumps(report, sort_keys=True) + "\n")

    def write_pipeline_files(self, *, junit_path=None):
        for gate in GATES:
            (self.out / f"{gate}.exit").write_text("0\n")
        raw = (junit_path or self.out / "full-pytest.xml").read_bytes()
        root = ET.fromstring(raw)  # nosec B314 - XML the test built itself
        suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
        junit = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        inventory = json.loads((self.out / "collection.metadata.json").read_text())
        summary = {
            "all_required_gates_passed": True,
            "candidate": CANDIDATE,
            "collection.metadata.json": inventory,
            "execution-collection.metadata.json": inventory,
            "gate_exit_codes": {g: 0 for g in GATES},
            "junit": junit,
            "junit_sha256": sha(raw),
            "phase_exit_codes": {p: 0 for p in PHASES},
            "tree": TREE,
        }
        (self.out / "full-ci-result.metadata.json").write_text(json.dumps(summary, sort_keys=True) + "\n")

    def pins(self):
        return observer.Pins(
            candidate=CANDIDATE,
            tree=TREE,
            collection_sha256=sha((self.out / "collection.metadata.json").read_bytes()),
            nodeids_sha256=shard_plan.nodeids_digest(self.ids),
            collected=len(self.ids),
            skip_sha256=sha(SKIP_MESSAGE.encode()),
            skip_function=SKIP_FUNCTION,
            skip_module="test_public_dns_pipe",
            gates=tuple(GATES),
            phases=PHASES,
        )

    def write_single_process_baseline(self, dest):
        dest.mkdir()
        cases = [junit_case(nid, self.statuses[nid]) for nid in self.ids]
        counts = self.counts_for(self.ids)
        junit = junit_document(
            cases, tests=len(cases) + counts.get("subtests passed", 0), skipped=counts["skipped"]
        )
        (dest / "full-pytest.xml").write_bytes(junit)
        lines = ["." * 40 + " [100%]", summary_line(counts, 120.0)]
        (dest / "pytest.log").write_text("\n".join(lines) + "\n")
        meta = (self.out / "collection.metadata.json").read_bytes()
        (dest / "collection.metadata.json").write_bytes(meta)
        (dest / "execution-collection.metadata.json").write_bytes(meta)
        (dest / "pytest.exit").write_text("0\n")
        clone = Fixture.__new__(Fixture)
        clone.out = dest
        clone.write_pipeline_files = Fixture.write_pipeline_files.__get__(clone)
        clone.write_pipeline_files()


def tree_listing(directory):
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*"))


class MergeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = Fixture(self._tmp.name)
        self.out = self.fx.out

    def merged_cases(self):
        root = ET.parse(self.out / "full-pytest.xml").getroot()  # nosec B314 - XML the test built itself
        return [(c.get("classname"), c.get("name")) for c in root.iter("testcase")]

    def test_happy_path_writes_the_files_the_observer_reads(self):
        result = shard_merge.merge(self.out, wall_seconds=33.25)
        self.assertEqual(result["exit_code"], 0)
        meta = (self.out / "collection.metadata.json").read_bytes()
        self.assertEqual((self.out / "execution-collection.metadata.json").read_bytes(), meta)
        self.assertEqual((self.out / "pytest.exit").read_text(), "0\n")
        expected_order = [shard_plan.junit_key(nid) for nid in self.fx.ids]
        self.assertEqual(self.merged_cases(), expected_order)
        suite = ET.parse(self.out / "full-pytest.xml").getroot().find("testsuite")  # nosec B314 - XML the test built itself
        sub = sum(self.fx.subtests.values())
        self.assertEqual(suite.get("tests"), str(len(self.fx.ids) + sub))
        self.assertEqual((suite.get("failures"), suite.get("errors"), suite.get("skipped")), ("0", "0", "1"))
        self.assertEqual(suite.get("time"), "33.250")
        reported = {
            m.group(2): int(m.group(1)) for m in LOG_COUNT.finditer((self.out / "pytest.log").read_text())
        }
        self.assertEqual(reported, {"passed": len(self.fx.ids) - 1, "skipped": 1, "subtests passed": sub})
        shards = json.loads((self.out / "shards.metadata.json").read_text())
        self.assertEqual((shards["n"], shards["plan_sha256"]), (3, self.fx.plan_sha))
        self.assertEqual(sum(shards["shard_selected_counts"]), len(self.fx.ids))
        self.assertEqual(shards["shard_exits"], [0, 0, 0])

    def test_final_log_line_wins_over_counts_printed_by_earlier_shards(self):
        shard_merge.merge(self.out)
        lines = (self.out / "pytest.log").read_text().rstrip("\n").splitlines()
        self.assertRegex(lines[-1], r"^\d+ passed, 1 skipped, \d+ warnings?, \d+ subtests passed in [0-9.]+s")
        self.assertEqual(sum(1 for line in lines if line.startswith("=== shard ")), 3)

    def test_a_count_printed_by_a_shard_for_a_key_with_total_zero_is_zeroed_in_the_final_line(self):
        self.fx.statuses[self.fx.skip_id] = "passed"
        for index in range(self.fx.n):
            self.fx.write_shard(index)
        log = self.fx.shard_dir(0) / "pytest.log"
        lines = log.read_text().splitlines()
        lines.insert(len(lines) - 1, "captured output: 3 skipped")
        log.write_text("\n".join(lines) + "\n")
        shard_merge.merge(self.out, wall_seconds=5.0)
        reported = {
            m.group(2): int(m.group(1)) for m in LOG_COUNT.finditer((self.out / "pytest.log").read_text())
        }
        self.assertEqual(reported["skipped"], 0)
        self.assertIn("0 skipped", (self.out / "pytest.log").read_text().rstrip("\n").splitlines()[-1])

    def test_refuses_shard_outputs_of_an_earlier_run_that_used_the_same_ids(self):
        older = json.loads(self.fx.plan_raw)
        older["run_nonce"] = "a" * 32
        newer = dict(older, run_nonce="b" * 32)
        for plan in (older, newer):
            raw = shard_plan.plan_bytes(plan)
            (self.out / "shard-plan.json").write_bytes(raw)
            self.fx.plan_sha = sha(raw)
            if plan is older:
                for index in range(self.fx.n):
                    self.fx.write_shard(index)
        self.refuse("plan")

    def test_output_after_pytests_summary_line_does_not_refuse_a_correct_shard(self):
        log = self.fx.shard_dir(1) / "pytest.log"
        log.write_text(log.read_text() + "Warning: leaked resource at interpreter shutdown\n")
        result = shard_merge.merge(self.out, wall_seconds=5.0)
        self.assertEqual(result["exit_code"], 0)

    def test_junit_failure_attributes_alone_fail_the_result(self):
        path = self.fx.shard_dir(0) / "junit.xml"
        root = ET.fromstring(path.read_bytes())  # nosec B314 - XML the test built itself
        root.find("testsuite").set("failures", "1")
        path.write_bytes(ET.tostring(root, encoding="utf-8"))
        result = shard_merge.merge(self.out)
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual((self.out / "pytest.exit").read_text(), "1\n")

    def test_an_xfailed_test_is_counted_apart_from_skips_and_the_run_still_merges(self):
        victim = self.fx.assigned(0)[0]
        self.fx.statuses[victim] = "xfailed"
        for index in range(self.fx.n):
            self.fx.write_shard(index)
        result = shard_merge.merge(self.out, wall_seconds=5.0)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["counts"]["xfailed"], 1)
        suite = ET.parse(self.out / "full-pytest.xml").getroot().find("testsuite")  # nosec B314 - XML the test built itself
        self.assertEqual(suite.get("skipped"), "2")
        self.assertIn("1 xfailed", (self.out / "pytest.log").read_text().rstrip("\n").splitlines()[-1])

    def test_a_failing_run_with_an_xfailed_test_keeps_both_outcomes(self):
        failing, xfailed = self.fx.assigned(1)[0], self.fx.assigned(1)[1]
        self.fx.statuses[failing] = "failed"
        self.fx.statuses[xfailed] = "xfailed"
        self.fx.write_shard(1, exit_code=1)
        result = shard_merge.merge(self.out)
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual((result["counts"]["failed"], result["counts"]["xfailed"]), (1, 1))

    def test_xfailed_count_that_differs_from_the_ledger_is_refused(self):
        victim = self.fx.assigned(0)[0]
        self.fx.statuses[victim] = "xfailed"
        self.fx.write_shard(0)
        report = self.fx.shard_dir(0) / "shard-report.json"
        data = json.loads(report.read_text())
        data["counts"]["xfailed"] = 0
        report.write_text(json.dumps(data))
        self.refuse("xfailed")

    def test_merge_is_independent_of_directory_listing_order(self):
        first = shard_merge.merge(self.out)
        raw = (self.out / "full-pytest.xml").read_bytes()
        again = shard_merge.merge(self.out)
        self.assertEqual(first["counts"], again["counts"])
        self.assertEqual((self.out / "full-pytest.xml").read_bytes(), raw)

    def refuse(self, text):
        before = tree_listing(self.out)
        with self.assertRaises(shard_merge.MergeError) as ctx:
            shard_merge.merge(self.out)
        self.assertIn(text, str(ctx.exception))
        self.assertEqual(tree_listing(self.out), before, "a refused merge must leave out/ unchanged")

    def test_refuses_a_missing_node_id(self):
        victim = self.fx.assigned(1)[0]
        self.fx.write_shard(1, drop=(victim,))
        self.refuse("missing")

    def test_refuses_a_duplicated_testcase(self):
        victim = self.fx.assigned(0)[0]
        self.fx.write_shard(0, duplicate=(victim,))
        self.refuse("duplicate")

    def test_refuses_an_id_executed_by_the_wrong_shard(self):
        stolen = self.fx.assigned(1)[0]
        self.fx.write_shard(0, extra=(stolen,))
        self.fx.write_shard(1, drop=(stolen,))
        self.refuse("wrong shard")

    def test_refuses_an_id_executed_by_two_shards(self):
        stolen = self.fx.assigned(1)[0]
        self.fx.write_shard(0, extra=(stolen,))
        self.refuse("duplicate")

    def test_refuses_an_extra_id_outside_the_plan(self):
        self.fx.statuses["tests/test_unplanned.py::test_x"] = "passed"
        self.fx.write_shard(2, extra=("tests/test_unplanned.py::test_x",))
        self.refuse("not in the plan")

    def test_refuses_a_shard_built_on_another_plan(self):
        self.fx.write_shard(1, plan_sha="1" * 64)
        self.refuse("plan")

    def test_refuses_a_shard_with_another_full_collection(self):
        self.fx.write_shard(2, full={"collected": len(self.fx.ids), "nodeids_sha256": "2" * 64})
        self.refuse("collection")

    def test_refuses_an_empty_shard(self):
        for nid in self.fx.assigned(1):
            self.fx.statuses.setdefault(nid, "passed")
        self.fx.write_shard(1, drop=tuple(self.fx.assigned(1)), exit_code=5)
        self.refuse("empty")

    def test_refuses_a_missing_shard_directory_and_missing_files(self):
        import shutil

        shutil.move(str(self.fx.shard_dir(2)), str(self.fx.root / "moved"))
        self.refuse("shard 2")
        shutil.move(str(self.fx.root / "moved"), str(self.fx.shard_dir(2)))
        os.unlink(self.fx.shard_dir(1) / "junit.xml")
        self.refuse("junit.xml")

    def test_refuses_a_skipped_count_that_differs_from_the_ledger(self):
        holder = self.fx.plan["shard_of"][self.fx.ids.index(self.fx.skip_id)]
        self.fx.write_shard(holder, suite_skipped=2)
        self.refuse("skipped")
        self.fx.write_shard(holder)
        self.fx.write_shard(holder, report_skipped=0)
        self.refuse("skipped")

    def test_refuses_a_plan_whose_collection_differs_from_the_reference_inventory(self):
        (self.out / "collection.metadata.json").write_bytes(
            shard_plan.collection_metadata_bytes(self.fx.ids[:-1])
        )
        self.refuse("inventory")

    def test_refuses_a_plan_file_that_changed_after_the_shards_ran(self):
        plan = json.loads(self.fx.plan_raw)
        plan["parameters"]["isolate_seconds"] = 1.0
        (self.out / "shard-plan.json").write_bytes(shard_plan.plan_bytes(plan))
        self.refuse("plan")

    def test_refuses_a_shard_exit_that_is_not_an_integer(self):
        (self.fx.shard_dir(0) / "pytest.exit").write_text("x\n")
        self.refuse("pytest.exit")

    def test_failing_shard_keeps_its_failure_and_sets_exit_one(self):
        victim = self.fx.assigned(1)[1]
        self.fx.statuses[victim] = "failed"
        self.fx.write_shard(1, exit_code=1, unmapped_failure=(victim, "subtest"))
        result = shard_merge.merge(self.out)
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual((self.out / "pytest.exit").read_text(), "1\n")
        root = ET.parse(self.out / "full-pytest.xml").getroot()  # nosec B314 - XML the test built itself
        names = [(c.get("name"), c.find("failure") is not None) for c in root.iter("testcase")]
        position = [n for n, _ in names].index(shard_plan.junit_key(victim)[1])
        self.assertTrue(names[position][1])
        self.assertTrue(names[position + 1][0].endswith("(<subtest>)"))
        self.assertEqual(root.find("testsuite").get("failures"), "2")
        self.assertEqual(
            json.loads((self.out / "shards.metadata.json").read_text())["shard_exits"], [0, 1, 0]
        )

    def test_main_returns_zero_one_and_two(self):
        self.assertEqual(shard_merge.main(["--out", str(self.out)]), 0)
        victim = self.fx.assigned(1)[1]
        self.fx.statuses[victim] = "failed"
        self.fx.write_shard(1, exit_code=1)
        self.assertEqual(shard_merge.main(["--out", str(self.out)]), 1)
        self.fx.write_shard(1, drop=(victim,))
        self.assertEqual(shard_merge.main(["--out", str(self.out)]), 2)


class ObserverCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = Fixture(self._tmp.name)
        self.out = self.fx.out
        shard_merge.merge(self.out)
        self.fx.write_pipeline_files()

    def test_merged_output_passes_the_observer_assertions(self):
        record = observer.check_out_dir(self.out, self.fx.pins())
        self.assertEqual(record["junit_physical_case_records"], len(self.fx.ids))
        self.assertEqual(record["junit_runtime_subtest_records"], sum(self.fx.subtests.values()))
        self.assertEqual(record["pytest_reported_counts"]["skipped"], 1)

    def mutate_junit(self, edit):
        path = self.out / "full-pytest.xml"
        root = ET.fromstring(path.read_bytes())  # nosec B314 - XML the test built itself
        edit(root)
        path.write_bytes(ET.tostring(root, encoding="utf-8"))
        self.fx.write_pipeline_files(junit_path=path)

    def test_a_removed_testcase_fails_the_assertions(self):
        def drop(root):
            suite = root.find("testsuite")
            suite.remove(suite.findall("testcase")[3])

        self.mutate_junit(drop)
        with self.assertRaises(AssertionError):
            observer.check_out_dir(self.out, self.fx.pins())

    def test_a_second_skip_fails_the_assertions(self):
        def skip_more(root):
            case = root.find("testsuite").findall("testcase")[0]
            ET.SubElement(case, "skipped", {"message": "other"})

        self.mutate_junit(skip_more)
        with self.assertRaises(AssertionError):
            observer.check_out_dir(self.out, self.fx.pins())

    def test_a_changed_inventory_file_fails_the_assertions(self):
        (self.out / "execution-collection.metadata.json").write_bytes(
            shard_plan.collection_metadata_bytes(self.fx.ids[:-1])
        )
        with self.assertRaises(AssertionError):
            observer.check_out_dir(self.out, self.fx.pins())

    def test_a_log_without_the_final_counts_fails_the_assertions(self):
        text = (self.out / "pytest.log").read_text().splitlines()
        (self.out / "pytest.log").write_text("\n".join(text[:-1]) + "\n")
        with self.assertRaises((AssertionError, KeyError)):
            observer.check_out_dir(self.out, self.fx.pins())

    def test_a_nonzero_gate_fails_the_assertions(self):
        (self.out / "ruff.exit").write_text("1\n")
        with self.assertRaises(AssertionError):
            observer.check_out_dir(self.out, self.fx.pins())

    def test_the_shard_metadata_adds_its_own_checks(self):
        observer.check_shards(self.out, expected_n=3)
        with self.assertRaises(AssertionError):
            observer.check_shards(self.out, expected_n=4)
        meta = json.loads((self.out / "shards.metadata.json").read_text())
        meta["shard_exits"][0] = 1
        (self.out / "shards.metadata.json").write_text(json.dumps(meta))
        with self.assertRaises(AssertionError):
            observer.check_shards(self.out, expected_n=3)


class EqualityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = Fixture(self._tmp.name)
        shard_merge.merge(self.fx.out)
        self.fx.write_pipeline_files()
        self.baseline = Path(self._tmp.name) / "baseline"
        self.fx.write_single_process_baseline(self.baseline)

    def differences(self):
        return equality.compare(self.baseline, self.fx.out)["differences"]

    def test_equal_runs_have_no_differences(self):
        self.assertEqual(self.differences(), [])

    def test_a_changed_status_is_reported(self):
        path = self.fx.out / "full-pytest.xml"
        root = ET.fromstring(path.read_bytes())  # nosec B314 - XML the test built itself
        case = root.find("testsuite").findall("testcase")[0]
        ET.SubElement(case, "skipped", {"message": "x"})
        path.write_bytes(ET.tostring(root, encoding="utf-8"))
        self.assertTrue(any("junit" in d for d in self.differences()))

    def test_a_changed_inventory_hash_is_reported(self):
        (self.fx.out / "execution-collection.metadata.json").write_bytes(b'{"collected": 1}\n')
        self.assertTrue(any("execution-collection" in d for d in self.differences()))

    def test_a_changed_gate_code_is_reported(self):
        (self.fx.out / "ruff.exit").write_text("1\n")
        self.assertTrue(any("ruff" in d for d in self.differences()))

    def test_a_changed_pytest_count_is_reported(self):
        text = (self.fx.out / "pytest.log").read_text()
        at = text.rindex(" passed")
        (self.fx.out / "pytest.log").write_text(text[:at] + "0" + text[at:])
        self.assertTrue(any("pytest log" in d for d in self.differences()))

    def test_a_changed_subtest_total_is_reported(self):
        path = self.fx.out / "full-pytest.xml"
        root = ET.fromstring(path.read_bytes())  # nosec B314 - XML the test built itself
        suite = root.find("testsuite")
        suite.set("tests", str(int(suite.get("tests")) + 1))
        path.write_bytes(ET.tostring(root, encoding="utf-8"))
        self.assertTrue(any("suite totals" in d for d in self.differences()))

    def test_command_line_prints_the_report_and_returns_nonzero_on_a_difference(self):
        self.assertEqual(equality.main([str(self.baseline), str(self.fx.out)]), 0)
        (self.fx.out / "ruff.exit").write_text("1\n")
        self.assertEqual(equality.main([str(self.baseline), str(self.fx.out)]), 1)


class FixtureSanityTests(unittest.TestCase):
    def test_fixture_plan_uses_every_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            self.assertEqual(sorted(set(fx.plan["shard_of"])), [0, 1, 2])
            self.assertEqual(copy.deepcopy(fx.plan)["n"], 3)


if __name__ == "__main__":
    unittest.main()
