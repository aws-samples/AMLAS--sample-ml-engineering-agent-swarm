"""Sharded full CI: the pytest plugin runs exactly one planned shard and records what ran."""

import hashlib
import json
import os
import subprocess  # nosec B404 - runs the test's own pytest child with a fixed argv
import sys
import tempfile
import unittest
from pathlib import Path

SHARD_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ci_shard"
sys.path.insert(0, str(SHARD_DIR))

import full_ci_shard_merge as shard_merge  # noqa: E402
import full_ci_shard_plan as shard_plan  # noqa: E402

SUITE = {
    "tests/test_alpha.py": (
        "import pytest\n\n\n"
        "def test_one():\n    assert True\n\n\n"
        "@pytest.mark.parametrize('value', ['a', 'b'])\n"
        "def test_param(value):\n    assert value\n\n\n"
        "def test_subtests(subtests):\n"
        "    for k in range(3):\n"
        "        with subtests.test(k=k):\n"
        "            assert k >= 0\n"
    ),
    "tests/test_beta.py": (
        "import pytest\n\n\n"
        "class TestBeta:\n"
        "    def test_x(self):\n        assert True\n\n"
        "    @pytest.mark.skip(reason='fixture skip')\n"
        "    def test_skipped(self):\n        raise AssertionError\n\n\n"
        "def test_tail():\n    assert True\n"
    ),
    "tests/test_gamma.py": "def test_g():\n    assert True\n",
}


def clean_environment():
    keep = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST", "AMLAS_CI"))}
    keep["PYTHONDONTWRITEBYTECODE"] = "1"
    return keep


class PluginFixture:
    def __init__(self, root, suite=None, shards=2):
        self.root = Path(root)
        self.out = self.root / "out"
        self.out.mkdir()
        self.write_suite(suite or SUITE)
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell
            [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
            cwd=self.root,
            env=clean_environment(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        self.ids = shard_plan.parse_collection_log(proc.stdout)
        (self.out / "collection.metadata.json").write_bytes(shard_plan.collection_metadata_bytes(self.ids))
        asset = {"schema": shard_plan.DURATIONS_SCHEMA, "detail_seconds": 1.0, "modules": {}}
        self.plan = shard_plan.build_plan(self.ids, asset, shards, durations_sha256="e" * 64)
        raw = shard_plan.plan_bytes(self.plan)
        (self.out / "shard-plan.json").write_bytes(raw)
        self.plan_sha = hashlib.sha256(raw).hexdigest()

    def write_suite(self, suite):
        for name, text in suite.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def run_shard(self, index, extra=(), plan_sha=None, env_drop=()):
        shard = self.out / "shards" / f"s{index:02d}"
        shard.mkdir(parents=True, exist_ok=True)
        env = clean_environment()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SHARD_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        env.update(
            AMLAS_CI_SHARD_PLAN=str(self.out / "shard-plan.json"),
            AMLAS_CI_SHARD_PLAN_SHA256=plan_sha or self.plan_sha,
            AMLAS_CI_SHARD_INDEX=str(index),
            AMLAS_CI_SHARD_OUT=str(shard),
        )
        for key in env_drop:
            env.pop(key, None)
        cmd = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "full_ci_shard",
            "-q",
            "-ra",
            "-o",
            f"cache_dir={self.root / 'cache' / str(index)}",
            f"--basetemp={self.root / 'basetemp' / str(index)}",
            f"--junitxml={shard / 'junit.xml'}",
            *extra,
            "tests",
        ]
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell
            cmd, cwd=self.root, env=env, capture_output=True, text=True, timeout=120
        )
        (shard / "pytest.log").write_text(proc.stdout + proc.stderr)
        (shard / "pytest.exit").write_text(f"{proc.returncode}\n")
        return proc, shard


class PluginTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_two_shards_run_the_plan_and_merge_to_the_single_run_result(self):
        fx = PluginFixture(self._tmp.name)
        self.assertEqual(len(fx.ids), 8)
        for index in range(2):
            proc, shard = fx.run_shard(index)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads((shard / "shard-report.json").read_text())
            assigned = [nid for nid, s in zip(fx.ids, fx.plan["shard_of"], strict=True) if s == index]
            self.assertEqual(report["index"], index)
            self.assertEqual(report["plan_sha256"], fx.plan_sha)
            self.assertEqual(
                report["collection"], json.loads((fx.out / "collection.metadata.json").read_text())
            )
            self.assertEqual(report["selected"]["collected"], len(assigned))
            self.assertEqual(report["selected"]["nodeids_sha256"], shard_plan.nodeids_digest(assigned))
            self.assertEqual(sorted(report["ledger"]), sorted(assigned))
            self.assertEqual(
                (shard / "execution-collection.metadata.json").read_bytes(),
                shard_plan.collection_metadata_bytes(assigned),
            )
            self.assertEqual(report["exitstatus"], 0)
            self.assertTrue(report["memory_peak_bytes"] is None or report["memory_peak_bytes"] > 0)
        result = shard_merge.merge(fx.out)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(
            (fx.out / "execution-collection.metadata.json").read_bytes(),
            shard_plan.collection_metadata_bytes(fx.ids),
        )
        self.assertEqual(result["counts"]["passed"], 7)
        self.assertEqual(result["counts"]["skipped"], 1)
        self.assertEqual(result["counts"]["subtests passed"], 3)
        self.assertIn("3 subtests passed", (fx.out / "pytest.log").read_text().splitlines()[-1])

    def test_selection_options_and_stale_pins_are_refused(self):
        fx = PluginFixture(self._tmp.name)
        refused = [
            ["-k", "test_one"],
            ["-m", "slow"],
            ["--deselect", "tests/test_gamma.py::test_g"],
            ["--lf"],
            ["--ff"],
            ["--nf"],
            ["--sw"],
            ["--stepwise-skip"],
            ["-x"],
            ["--maxfail=2"],
            ["--collect-only"],
        ]
        for extra in refused:
            proc, shard = fx.run_shard(0, extra=extra)
            self.assertNotEqual(proc.returncode, 0, extra)
            self.assertIn("full_ci_shard", proc.stdout + proc.stderr, extra)
            self.assertFalse((shard / "shard-report.json").exists(), extra)
        proc, _ = fx.run_shard(0, plan_sha="0" * 64)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("plan", proc.stdout + proc.stderr)
        for key in (
            "AMLAS_CI_SHARD_PLAN",
            "AMLAS_CI_SHARD_INDEX",
            "AMLAS_CI_SHARD_OUT",
            "AMLAS_CI_SHARD_PLAN_SHA256",
        ):
            proc, _ = fx.run_shard(0, env_drop=(key,))
            self.assertNotEqual(proc.returncode, 0, key)
            self.assertIn(key, proc.stdout + proc.stderr)
        proc, _ = fx.run_shard(5)
        self.assertNotEqual(proc.returncode, 0)

    def test_a_suite_that_changed_after_planning_is_refused(self):
        fx = PluginFixture(self._tmp.name)
        fx.write_suite({"tests/test_delta.py": "def test_new():\n    assert True\n"})
        proc, shard = fx.run_shard(0)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("collection", proc.stdout + proc.stderr)
        self.assertFalse((shard / "shard-report.json").exists())

    def test_failures_and_subtest_failures_are_recorded_and_the_merge_exits_one(self):
        suite = dict(SUITE)
        suite["tests/test_gamma.py"] = (
            "def test_g():\n    assert False\n\n\n"
            "def test_sub_fail(subtests):\n"
            "    with subtests.test(k=1):\n        assert False\n"
        )
        fx = PluginFixture(self._tmp.name, suite=suite)
        codes = []
        for index in range(2):
            proc, shard = fx.run_shard(index)
            codes.append(proc.returncode)
            report = json.loads((shard / "shard-report.json").read_text())
            assigned = [nid for nid, s in zip(fx.ids, fx.plan["shard_of"], strict=True) if s == index]
            self.assertEqual(sorted(report["ledger"]), sorted(assigned))
            self.assertEqual(report["exitstatus"], proc.returncode)
        self.assertEqual(sorted(codes), [0, 1])
        result = shard_merge.merge(fx.out)
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual((fx.out / "pytest.exit").read_text(), "1\n")
        self.assertGreaterEqual(result["counts"].get("failed", 0), 1)


if __name__ == "__main__":
    unittest.main()
