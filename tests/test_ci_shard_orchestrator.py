"""Sharded full CI: shard containers run side by side with the single pipeline's flags."""

import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ci_shard"))

import derive_sharded_assets as derive
import full_ci_shard_plan as shard_plan
import shard_orchestrator as orch

FAKE_DOCKER = """#!{python}
import json, os, sys, time
args = sys.argv[1:]
log = os.environ["FAKE_DOCKER_LOG"]
if args[:1] == ["run"]:
    name = args[args.index("--name") + 1]
    cidfile = args[args.index("--cidfile") + 1]
    open(cidfile, "w").write((name.encode().hex() * 4)[:64] + "\\n")
    start = time.time()
    time.sleep(float(os.environ.get("FAKE_DOCKER_SLEEP", "0.4")))
    with open(log, "a") as f:
        f.write(json.dumps({{"name": name, "start": start, "end": time.time(), "args": args}}) + "\\n")
    code = int(json.loads(os.environ.get("FAKE_DOCKER_EXIT", "{{}}")).get(name.rsplit("-", 1)[-1], 0))
    sys.exit(code)
if args[:1] == ["ps"]:
    sys.exit(0)
sys.exit(1)
"""


def make_config(root, docker="docker", n=3, **kw):
    root = Path(root)
    w = Path("/w")
    values = dict(
        docker=docker,
        run_id="ci-test",
        image="sha256:img",
        out=root / "out",
        cache=root / "cache",
        qa=root / "qa",
        base=Path("/base"),
        workplace=w,
        cpu=Path("/cpu"),
        certifi=Path("/certifi"),
        case_path=Path("/case.parquet"),
        assets=Path("/assets"),
        candidate="a" * 40,
        tree="b" * 40,
        n=n,
        cpus="4",
        memory="12g",
        cpusets=None,
        extra_ro_mounts=("/old/qa-one", "/old/qa-two"),
        script="full-ci-python-shard.sh",
    )
    values.update(kw)
    return orch.ShardRunConfig(**values)


def baseline_python_command(c):
    """The single pipeline's python-phase command (full-ci.native.py docker_phase), image and flags verbatim."""
    w = c.workplace
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        c.run_id + "-python",
        "--cidfile",
        str(c.out / "phase-python.cid"),
        "--network",
        "none",
        "--read-only",
        "--memory",
        "12g",
        "--cpus",
        "4",
        "-v",
        str(c.qa) + ":/work:ro",
        "-v",
        str(w / "repos") + ":" + str(w / "repos") + ":ro",
        "-v",
        str(c.base) + ":" + str(c.base) + ":ro",
        "-v",
        "/old/qa-one:/old/qa-one:ro",
        "-v",
        "/old/qa-two:/old/qa-two:ro",
        "-v",
        str(w / "tmp/ci-seed") + ":/ci-seed:ro",
        "-v",
        str(c.cpu) + ":/ci-cpu:ro",
        "-v",
        str(c.certifi) + ":/ci-certifi:ro",
        "-v",
        str(c.case_path) + ":/ci-source/humaneval-original.parquet:ro",
        "-v",
        str(c.assets) + ":/ci-assets:ro",
        "-v",
        str(c.out) + ":/ci-out",
        "-v",
        str(c.cache) + ":/ci-run",
        "-w",
        "/work",
        "-e",
        "AMLAS_CI_HEAD=" + c.candidate,
        "-e",
        "AMLAS_CI_TREE=" + c.tree,
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "AWS_EC2_METADATA_DISABLED=true",
        c.image,
        "bash",
        "/ci-assets/full-ci-python.sh",
    ]


class CommandTests(unittest.TestCase):
    def test_shard_command_is_the_single_pipeline_command_with_only_the_shard_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = make_config(tmp)
            expected = baseline_python_command(c)
            expected[expected.index("ci-test-python")] = "ci-test-python-s01"
            expected[expected.index(str(c.out / "phase-python.cid"))] = str(c.out / "phase-python-s01.cid")
            expected[expected.index(str(c.out) + ":/ci-out")] = str(c.out) + ":/ci-out:ro"
            at = expected.index(str(c.cache) + ":/ci-run")
            expected[at] = str(c.cache / "s01") + ":/ci-run"
            expected[at:at] = [str(c.out / "shards" / "s01") + ":/ci-shard", "-v"]
            tail = expected.index(c.image)
            expected[tail:tail] = [
                "-e",
                "AMLAS_CI_SHARD_PLAN=/ci-out/shard-plan.json",
                "-e",
                "AMLAS_CI_SHARD_PLAN_SHA256=" + "f" * 64,
                "-e",
                "AMLAS_CI_SHARD_INDEX=1",
                "-e",
                "AMLAS_CI_SHARD_OUT=/ci-shard",
            ]
            expected[-1] = "/ci-assets/full-ci-python-shard.sh"
            self.assertEqual(orch.docker_command(c, 1, plan_sha256="f" * 64), expected)

    def test_shard_names_cidfiles_and_mounts_are_unique_per_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = make_config(tmp, n=4)
            commands = [orch.docker_command(c, i, plan_sha256="1" * 64) for i in range(4)]
            for flag in ("--name", "--cidfile"):
                values = [cmd[cmd.index(flag) + 1] for cmd in commands]
                self.assertEqual(len(set(values)), 4)
            mounts = [
                [a for a in cmd if a.endswith(":/ci-run") or a.endswith(":/ci-shard")] for cmd in commands
            ]
            self.assertEqual(len({tuple(m) for m in mounts}), 4)
            for cmd in commands:
                self.assertIn(str(c.out) + ":/ci-out:ro", cmd)

    def test_cpusets_are_added_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = make_config(tmp, n=2, cpusets=("0-1,16-17", "2-3,18-19"))
            cmd = orch.docker_command(c, 1, plan_sha256="1" * 64)
            self.assertEqual(cmd[cmd.index("--cpuset-cpus") + 1], "2-3,18-19")
            self.assertEqual(cmd[cmd.index("--cpus") + 1], "4")

    def test_cpusets_follow_core_pairs_without_overlap(self):
        lscpu = "\n".join(["# comment"] + [f"{cpu},{cpu % 16},0" for cpu in range(32)])
        sets = orch.assign_cpusets(8, lscpu)
        self.assertEqual(len(sets), 8)
        seen = set()
        for text in sets:
            cpus = orch.parse_cpuset(text)
            self.assertEqual(len(cpus), 4)
            self.assertFalse(seen & cpus)
            seen |= cpus
            self.assertEqual({c % 16 for c in cpus}.__len__(), 2)
        self.assertEqual(seen, set(range(32)))
        self.assertEqual(len(orch.assign_cpusets(5, lscpu)), 5)
        with self.assertRaises(ValueError):
            orch.assign_cpusets(17, lscpu)


class RunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.docker = self.root / "docker"
        self.docker.write_text(FAKE_DOCKER.format(python=sys.executable))
        self.docker.chmod(self.docker.stat().st_mode | stat.S_IXUSR)
        self.log = self.root / "docker.log"
        patcher = mock.patch.dict(os.environ, {"FAKE_DOCKER_LOG": str(self.log), "FAKE_DOCKER_SLEEP": "0.5"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = make_config(self.root, docker=str(self.docker), n=3)
        self.config.out.mkdir()
        self.config.cache.mkdir()

    def test_all_shards_start_before_any_finishes_and_codes_are_returned_per_shard(self):
        with mock.patch.dict(os.environ, {"FAKE_DOCKER_EXIT": json.dumps({"s01": 1})}):
            started = time.time()
            codes = orch.run_shards(self.config, plan_sha256="a" * 64, poll_seconds=0.05)
        self.assertEqual(codes, [0, 1, 0])
        self.assertLess(time.time() - started, 1.4, "shards must run concurrently")
        runs = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(runs), 3)
        self.assertLess(max(r["start"] for r in runs), min(r["end"] for r in runs))
        for index in range(3):
            self.assertTrue((self.config.out / f"phase-python-s{index:02d}.cid").is_file())
            self.assertTrue((self.config.out / "shards" / f"s{index:02d}").is_dir())
            self.assertTrue((self.config.cache / f"s{index:02d}").is_dir())

    def test_a_refused_merge_still_writes_the_gate_files_with_exit_one(self):
        ids = [f"tests/test_m{k}.py::t" for k in range(3)]
        (self.config.out / "collection.log").write_text("\n".join(ids) + "\n\n3 tests collected in 1s\n")
        (self.config.out / "collection.metadata.json").write_bytes(shard_plan.collection_metadata_bytes(ids))
        asset = {"schema": shard_plan.DURATIONS_SCHEMA, "detail_seconds": 1.0, "modules": {}}
        durations = self.root / "shard-durations.json"
        durations.write_bytes(shard_plan.asset_bytes(asset))
        code = orch.run_python_shards(self.config, durations_path=durations, poll_seconds=0.05)
        self.assertEqual(code, 1)
        self.assertEqual((self.config.out / "pytest.exit").read_text(), "1\n")
        self.assertIn("merge refused", (self.config.out / "pytest.log").read_text())
        self.assertTrue((self.config.out / "shard-plan.json").is_file())
        self.assertEqual(len(list(self.config.out.glob("phase-python-s*.cid"))), 3)

    def test_run_from_native_applies_the_pinned_shard_parameters(self):
        assets = self.root / "assets"
        assets.mkdir()
        params = {
            "shards": 2,
            "cpus": "3",
            "memory": "5g",
            "cpusets": "none",
            "durations": "d.json",
            "extra_ro_mounts": ["/x/y"],
            "script": "s.sh",
        }
        (assets / "shard-params.json").write_text(json.dumps(params))
        ids = [f"tests/test_m{k}.py::t" for k in range(2)]
        (self.config.out / "collection.log").write_text("\n".join(ids) + "\n\n2 tests collected in 1s\n")
        (self.config.out / "collection.metadata.json").write_bytes(shard_plan.collection_metadata_bytes(ids))
        asset = {"schema": shard_plan.DURATIONS_SCHEMA, "detail_seconds": 1.0, "modules": {}}
        (assets / "d.json").write_bytes(shard_plan.asset_bytes(asset))
        code = orch.run_from_native(
            run_id="ci-native",
            image="sha256:img",
            out=self.config.out,
            cache=self.config.cache,
            qa=self.root / "qa",
            base="/base",
            workplace="/w",
            cpu="/cpu",
            certifi="/certifi",
            case_path="/case.parquet",
            assets=assets,
            candidate="a" * 40,
            tree="b" * 40,
            docker=str(self.docker),
        )
        self.assertEqual(code, 1, "no shard wrote a report, so the merge must refuse")
        runs = [json.loads(line)["args"] for line in self.log.read_text().splitlines()]
        self.assertEqual(len(runs), 2)
        for args in runs:
            self.assertEqual(args[args.index("--cpus") + 1], "3")
            self.assertEqual(args[args.index("--memory") + 1], "5g")
            self.assertNotIn("--cpuset-cpus", args)
            self.assertIn("/x/y:/x/y:ro", args)
            self.assertEqual(args[-1], "/ci-assets/s.sh")

    def test_run_from_native_turns_an_orchestrator_exception_into_gate_exit_one(self):
        assets = self.root / "assets"
        assets.mkdir()
        params = {"shards": 2, "cpus": "3", "memory": "5g", "cpusets": "none", "durations": "d.json"}
        (assets / "shard-params.json").write_text(json.dumps(params))
        ids = [f"tests/test_m{k}.py::t" for k in range(2)]
        (self.config.out / "collection.log").write_text("\n".join(ids) + "\n\n2 tests collected in 1s\n")
        (self.config.out / "collection.metadata.json").write_bytes(shard_plan.collection_metadata_bytes(ids))
        asset = {"schema": shard_plan.DURATIONS_SCHEMA, "detail_seconds": 1.0, "modules": {}}
        (assets / "d.json").write_bytes(shard_plan.asset_bytes(asset))
        with mock.patch.object(orch, "run_shards", side_effect=RuntimeError("daemon unreachable")):
            code = orch.run_from_native(
                run_id="ci-native",
                image="sha256:img",
                out=self.config.out,
                cache=self.config.cache,
                qa=self.root / "qa",
                base="/base",
                workplace="/w",
                cpu="/cpu",
                certifi="/certifi",
                case_path="/case.parquet",
                assets=assets,
                candidate="a" * 40,
                tree="b" * 40,
                docker=str(self.docker),
            )
        self.assertEqual(code, 1)
        self.assertEqual((self.config.out / "pytest.exit").read_text(), "1\n")
        self.assertIn("RuntimeError", (self.config.out / "pytest.log").read_text())

    def test_every_plan_carries_a_fresh_run_nonce(self):
        ids = [f"tests/test_m{k}.py::t" for k in range(3)]
        (self.config.out / "collection.log").write_text("\n".join(ids) + "\n\n3 tests collected in 1s\n")
        (self.config.out / "collection.metadata.json").write_bytes(shard_plan.collection_metadata_bytes(ids))
        asset = {"schema": shard_plan.DURATIONS_SCHEMA, "detail_seconds": 1.0, "modules": {}}
        durations = self.root / "d.json"
        durations.write_bytes(shard_plan.asset_bytes(asset))
        orch.run_python_shards(self.config, durations_path=durations, poll_seconds=0.05)
        nonces = {shard_plan.load_plan(self.config.out / "shard-plan.json")["run_nonce"]}
        other = self.root / "second"
        other.mkdir()
        (other / "out").mkdir()
        (other / "cache").mkdir()
        for name in ("collection.log", "collection.metadata.json"):
            (other / "out" / name).write_bytes((self.config.out / name).read_bytes())
        again = make_config(other, docker=str(self.docker), n=3)
        orch.run_python_shards(again, durations_path=durations, poll_seconds=0.05)
        nonces.add(shard_plan.load_plan(other / "out" / "shard-plan.json")["run_nonce"])
        self.assertEqual(len(nonces), 2)
        self.assertTrue(all(len(n) == 32 for n in nonces))

    def test_existing_shard_directories_are_refused(self):
        (self.config.out / "shards" / "s00").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            orch.run_shards(self.config, plan_sha256="a" * 64, poll_seconds=0.05)

    def test_a_shard_that_outlives_the_timeout_is_killed_and_reported_as_124(self):
        with mock.patch.dict(os.environ, {"FAKE_DOCKER_SLEEP": "6"}):
            started = time.time()
            codes = orch.run_shards(self.config, plan_sha256="a" * 64, poll_seconds=0.05, timeout_seconds=1.0)
        self.assertLess(time.time() - started, 5.0)
        self.assertEqual(codes, [124, 124, 124])

    def test_a_container_that_never_disappears_is_an_error_not_a_hang(self):
        always_present = self.root / "docker-present"
        always_present.write_text(
            FAKE_DOCKER.format(python=sys.executable)
            .replace(
                'if args[:1] == ["ps"]:\n    sys.exit(0)',
                'if args[:1] == ["ps"]:\n    print("deadbeef")\n    sys.exit(0)',
            )
            .replace(
                'if args[:1] == ["run"]:',
                'if args[:1] == ["inspect"]:\n    print("true")\n    sys.exit(0)\nif args[:1] == ["run"]:',
            )
        )
        always_present.chmod(always_present.stat().st_mode | stat.S_IXUSR)
        config = make_config(self.root, docker=str(always_present), n=1)
        with mock.patch.dict(os.environ, {"FAKE_DOCKER_SLEEP": "0.1"}):
            with self.assertRaises(RuntimeError):
                orch.run_shards(config, plan_sha256="a" * 64, poll_seconds=0.05, gone_timeout_seconds=0.5)

    def test_derived_native_marks_the_python_phase_failed_before_the_orchestrator_runs(self):
        text = derive.derive_native(ORIGINAL_NATIVE)
        self.assertLess(text.index('phase_codes["python"]=1'), text.index("run_from_native"))

    def test_derivation_refuses_anything_after_the_pytest_gate_but_the_closing_checks(self):
        extra = ORIGINAL_PYTHON.replace(
            'test -z "$(git status --porcelain)" || failed=1\nexit',
            'run_gate reads python read_pytest_exit.py\ntest -z "$(git status --porcelain)" || failed=1\nexit',
        )
        with self.assertRaises(derive.DeriveError):
            derive.derive_python_scripts(extra)


ORIGINAL_PYTHON = """#!/bin/bash
set -euo pipefail
export PATH=/ci-seed/node/bin:$PATH
export PYTHONPATH=/ci-certifi:$AMLAS_CI_BASE_CPU_PYTHONPATH:$base_pythonpath:/ci-assets
run_gate() { name=$1; shift; set +e; "$@" > "/ci-out/$name.log" 2>&1; code=$?; set -e; printf '%s\\n' "$code" > "/ci-out/$name.exit"; return "$code"; }
test "$(git rev-parse HEAD)" = "$AMLAS_CI_HEAD"
test -z "$(git status --porcelain)"
failed=0
run_gate ruff python -m ruff check src scripts infra tests || failed=1
if [ "$failed" != 0 ]; then exit 1; fi
export AMLAS_SDIST_INSPECTION_ARTIFACT=$(cat /ci-out/sdist-path)
export AMLAS_CI_COLLECTION_METADATA=/ci-out/collection.metadata.json
run_gate collection python -m pytest -p full_ci_collection --collect-only -q -o cache_dir=/ci-run/pytest tests
export AMLAS_CI_COLLECTION_METADATA=/ci-out/execution-collection.metadata.json
run_gate pytest python -m pytest -p full_ci_collection -q -ra --basetemp=/ci-run/tmp/full-owned -o cache_dir=/ci-run/pytest --junitxml=/ci-out/full-pytest.xml tests || failed=1
test -z "$(git status --porcelain)" || failed=1
exit "$failed"
"""

ORIGINAL_NATIVE = """def main():
    try:
        docker_phase("dox","full-ci-additional.sh",phase="dox")
        if phase_codes["dox"]==0:
            docker_phase("package","full-ci-additional.sh",network=c["package_network"],phase="package")
            docker_phase("audit","full-ci-additional.sh",network=c["audit_network"],phase="audit")
            if phase_codes["package"]==0:docker_phase("python","full-ci-python.sh")
            docker_phase("rust","full-ci-rust.sh",rust=True)
    finally:
        if (out/"full-pytest.xml").is_file():
            doc=ET.parse(out/"full-pytest.xml").getroot()
            summary["junit"]={k:1 for k in ("tests",)};summary["junit_sha256"]=sha(out/"full-pytest.xml")
        passed=set(phase_codes)=={"dox","package","audit","python","rust"} and all(v==0 for v in phase_codes.values())
        summary["all_required_gates_passed"]=passed
"""


class DeriveTests(unittest.TestCase):
    def test_python_script_splits_into_gates_without_pytest_and_a_shard_script(self):
        gates, shard = derive.derive_python_scripts(ORIGINAL_PYTHON)
        self.assertIn("run_gate collection python -m pytest -p full_ci_collection --collect-only", gates)
        self.assertNotIn("run_gate pytest", gates)
        self.assertNotIn("execution-collection.metadata.json", gates)
        self.assertIn('exit "$failed"', gates)
        self.assertNotIn("run_gate ruff", shard)
        self.assertNotIn("--collect-only", shard)
        self.assertIn("-p full_ci_shard", shard)
        self.assertIn("--junitxml=/ci-shard/junit.xml", shard)
        self.assertIn("/ci-shard/pytest.exit", shard)
        self.assertIn('test "$(git rev-parse HEAD)" = "$AMLAS_CI_HEAD"', shard)
        self.assertIn("export AMLAS_SDIST_INSPECTION_ARTIFACT=$(cat /ci-out/sdist-path)", shard)
        self.assertTrue(shard.rstrip().endswith('exit "$failed"'))
        for forbidden in ("-k ", "--deselect", "--lf"):
            self.assertNotIn(forbidden, shard)

    def test_native_runner_runs_gates_then_shards_and_reports_sharding(self):
        text = derive.derive_native(ORIGINAL_NATIVE)
        compile(text, "derived-native", "exec")
        self.assertIn('docker_phase("python","full-ci-python-gates.sh")', text)
        self.assertNotIn('docker_phase("python","full-ci-python.sh")', text)
        self.assertIn("shard_orchestrator", text)
        self.assertIn('summary["sharding"]', text)
        self.assertLess(text.index("full-ci-python-gates.sh"), text.index("shard_orchestrator"))

    def test_derivation_refuses_an_original_that_changed(self):
        with self.assertRaises(derive.DeriveError):
            derive.derive_python_scripts(ORIGINAL_PYTHON.replace("run_gate pytest", "run_gate other"))
        with self.assertRaises(derive.DeriveError):
            derive.derive_python_scripts(ORIGINAL_PYTHON + ORIGINAL_PYTHON.split("\n", 2)[2])
        with self.assertRaises(derive.DeriveError):
            derive.derive_native(ORIGINAL_NATIVE.replace('"full-ci-python.sh"', '"other.sh"'))
        with self.assertRaises(derive.DeriveError):
            derive.derive_native(ORIGINAL_NATIVE + ORIGINAL_NATIVE)


if __name__ == "__main__":
    unittest.main()
