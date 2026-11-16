"""Sharded full CI: the plan splits the collection into disjoint, deterministic shards."""

import hashlib
import json
import random
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET  # nosec B405 - builds JUnit XML for the test
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ci_shard"))


import full_ci_shard_plan as shard_plan


def make_ids(layout):
    ids = []
    for module, names in layout.items():
        ids.extend(f"tests/{module}.py::{name}" for name in names)
    return ids


def junit_for(seconds):
    suite = ET.Element("testsuite", {"name": "pytest", "tests": str(len(seconds))})
    for nid, value in seconds.items():
        classname, name = shard_plan.junit_key(nid)
        ET.SubElement(suite, "testcase", {"classname": classname, "name": name, "time": f"{value:.3f}"})
    root = ET.Element("testsuites", {"name": "pytest tests"})
    root.append(suite)
    return ET.tostring(root, encoding="utf-8")


def synthetic(seed, modules=30, heavy=True):
    rng = random.Random(seed)  # nosec B311 - seeded test data, not a secret
    layout, seconds = {}, {}
    for index in range(modules):
        module = f"test_m{index:02d}"
        layout[module] = [f"t{k}" for k in range(rng.randint(1, 12))]
    if heavy:
        layout["test_heavy"] = ["slow_a", "slow_b", "slow_c[x]", "small_1", "small_2"]
    ids = make_ids(layout)
    for nid in ids:
        seconds[nid] = round(rng.expovariate(1 / 3.0), 3)
    if heavy:
        seconds["tests/test_heavy.py::slow_a"] = 338.0
        seconds["tests/test_heavy.py::slow_b"] = 140.0
        seconds["tests/test_heavy.py::slow_c[x]"] = 60.0
    asset = shard_plan.build_durations_asset(junit_for(seconds), ids)
    return ids, seconds, asset


def module_of(nid):
    return nid.split("::", 1)[0]


class CollectionTests(unittest.TestCase):
    def test_log_ids_stop_at_the_first_blank_line(self):
        text = "tests/a.py::t1\ntests/a.py::t2[x]\n\nwarnings summary\n2 tests collected in 0.1s\n"
        self.assertEqual(shard_plan.parse_collection_log(text), ["tests/a.py::t1", "tests/a.py::t2[x]"])

    def test_inventory_bytes_match_the_collection_plugin_serialization(self):
        ids = ["tests/a.py::t1", "tests/b.py::t2[é]"]
        raw = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        expected = (json.dumps({"collected": 2, "nodeids_sha256": digest}, sort_keys=True) + "\n").encode()
        self.assertEqual(shard_plan.nodeids_digest(ids), digest)
        self.assertEqual(shard_plan.collection_metadata_bytes(ids), expected)

    def test_verify_collection_refuses_an_altered_dropped_or_reordered_id(self):
        ids = make_ids({"m1": ["a", "b"], "m2": ["c"]})
        meta = shard_plan.collection_metadata_bytes(ids)
        shard_plan.verify_collection(ids, meta)
        for bad in ([*ids[:-1], ids[-1] + "x"], ids[:-1], list(reversed(ids)), ids + ids[:1]):
            with self.assertRaises(shard_plan.PlanError):
                shard_plan.verify_collection(bad, meta)

    def test_junit_key_follows_pytest_mangling(self):
        key = shard_plan.junit_key
        self.assertEqual(key("tests/test_a.py::test_x"), ("tests.test_a", "test_x"))
        self.assertEqual(key("tests/test_a.py::Cls::test_x[p-1]"), ("tests.test_a.Cls", "test_x[p-1]"))
        self.assertEqual(key("tests/test_a.py::Out::In::test_x"), ("tests.test_a.Out.In", "test_x"))
        self.assertEqual(
            key("tests/test_a.py::test_x[arn:aws::iam::root]"), ("tests.test_a", "test_x[arn:aws::iam::root]")
        )
        self.assertEqual(key("tests/test_a.py::test_x[\x01]"), ("tests.test_a", "test_x[#x01]"))

    def test_junit_key_index_refuses_two_ids_with_one_key(self):
        ids = ["tests/a.b.py::x", "tests/a/b.py::x"]
        with self.assertRaises(shard_plan.PlanError):
            shard_plan.junit_key_index(ids)


class DurationTests(unittest.TestCase):
    def test_durations_asset_keeps_slow_tests_and_module_remainders(self):
        ids = make_ids({"m1": ["a", "b", "c"], "m2": ["d"]})
        seconds = {ids[0]: 5.0, ids[1]: 0.25, ids[2]: 0.5, ids[3]: 2.0}
        asset = shard_plan.build_durations_asset(junit_for(seconds), ids)
        self.assertEqual(asset["modules"]["tests/m1.py"]["count"], 3)
        self.assertEqual(sorted(asset["modules"]["tests/m1.py"]["tests"]), ["a"])
        self.assertAlmostEqual(shard_plan.duration_of(asset, ids[0]), 5.0)
        self.assertAlmostEqual(shard_plan.duration_of(asset, ids[1]), 0.375)
        self.assertAlmostEqual(shard_plan.duration_of(asset, ids[3]), 2.0)
        self.assertEqual(shard_plan.duration_of(asset, "tests/unknown.py::z"), shard_plan.UNKNOWN_SECONDS)
        self.assertAlmostEqual(shard_plan.duration_of(asset, "tests/m1.py::new_test"), 0.375)

    def test_durations_refuse_a_junit_that_misses_a_collected_id(self):
        ids = make_ids({"m1": ["a", "b"]})
        with self.assertRaises(shard_plan.PlanError):
            shard_plan.build_durations_asset(junit_for({ids[0]: 1.0}), ids)

    def test_asset_bytes_are_canonical(self):
        _ids, _, asset = synthetic(3, modules=4)
        first = shard_plan.asset_bytes(asset)
        second = shard_plan.asset_bytes(json.loads(first))
        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))


class PlanTests(unittest.TestCase):
    def build(self, ids, asset, n, **kw):
        return shard_plan.build_plan(ids, asset, n, durations_sha256="d" * 64, **kw)

    def test_every_id_lands_in_exactly_one_nonempty_shard_for_every_shard_count(self):
        for seed in (1, 2, 3):
            ids, _, asset = synthetic(seed)
            for n in (1, 2, 3, 5, 8, 13, 21, 30):
                plan = self.build(ids, asset, n)
                self.assertEqual(plan["n"], n)
                self.assertEqual(plan["ids"], ids)
                self.assertEqual(len(plan["shard_of"]), len(ids))
                buckets = [[] for _ in range(n)]
                for nid, shard in zip(ids, plan["shard_of"], strict=True):
                    buckets[shard].append(nid)
                self.assertTrue(all(buckets), f"empty shard at n={n} seed={seed}")
                self.assertEqual(sorted(nid for bucket in buckets for nid in bucket), sorted(ids))
                self.assertEqual([s["ids_count"] for s in plan["shards"]], [len(b) for b in buckets])

    def test_plan_bytes_are_deterministic(self):
        ids, _, asset = synthetic(7)
        shuffled = json.loads(shard_plan.asset_bytes(asset))
        mods = list(shuffled["modules"].items())
        random.Random(5).shuffle(mods)  # nosec B311 - seeded test data, not a secret
        shuffled["modules"] = dict(mods)
        first = shard_plan.plan_bytes(self.build(ids, asset, 8))
        second = shard_plan.plan_bytes(self.build(ids, shuffled, 8))
        self.assertEqual(first, second)
        self.assertEqual(shard_plan.plan_sha256(self.build(ids, asset, 8)), hashlib.sha256(first).hexdigest())

    def test_longest_test_is_alone_in_its_shard_and_heavy_modules_are_split(self):
        ids, _, asset = synthetic(11)
        plan = self.build(ids, asset, 8)
        slow = ids.index("tests/test_heavy.py::slow_a")
        alone = plan["shard_of"][slow]
        self.assertEqual(plan["shard_of"].count(alone), 1)
        self.assertTrue(plan["shards"][alone]["isolated"])
        heavy_shards = {
            s for nid, s in zip(ids, plan["shard_of"], strict=True) if module_of(nid) == "tests/test_heavy.py"
        }
        self.assertGreater(len(heavy_shards), 1)

    def test_run_nonce_changes_the_plan_bytes_and_keeps_the_assignment(self):
        ids, _, asset = synthetic(16, modules=6)
        plain = self.build(ids, asset, 3)
        first = self.build(ids, asset, 3, run_nonce="a" * 32)
        second = self.build(ids, asset, 3, run_nonce="b" * 32)
        self.assertIsNone(plain["run_nonce"])
        self.assertEqual(first["run_nonce"], "a" * 32)
        self.assertEqual(first["shard_of"], second["shard_of"])
        self.assertEqual(len({shard_plan.plan_sha256(p) for p in (plain, first, second)}), 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard-plan.json"
            path.write_bytes(shard_plan.plan_bytes(first))
            self.assertEqual(shard_plan.load_plan(path)["run_nonce"], "a" * 32)

    def test_light_modules_are_never_split(self):
        ids, _, asset = synthetic(12)
        plan = self.build(ids, asset, 8)
        by_module = {}
        for nid, shard in zip(ids, plan["shard_of"], strict=True):
            by_module.setdefault(module_of(nid), set()).add(shard)
        for module, shards in by_module.items():
            if module != "tests/test_heavy.py":
                self.assertEqual(len(shards), 1, module)

    def test_balance_stays_within_the_list_scheduling_bound(self):
        ids, _, asset = synthetic(13)
        for n in (4, 8):
            plan = self.build(ids, asset, n)
            loads = [s["planned_seconds"] for s in plan["shards"] if not s["isolated"]]
            rest = sum(loads)
            self.assertLessEqual(max(loads), 1.5 * rest / len(loads) + 1e-6)

    def test_unknown_ids_are_assigned_with_the_default_weight(self):
        ids, _, asset = synthetic(14, modules=10)
        extra = [*ids, "tests/test_brand_new.py::t1", "tests/test_m00.py::added"]
        plan = self.build(extra, asset, 4)
        self.assertEqual(plan["ids"], extra)
        self.assertEqual(len(plan["shard_of"]), len(extra))
        self.assertTrue(all(0 <= s < 4 for s in plan["shard_of"]))

    def test_refuses_zero_shards_and_more_shards_than_units(self):
        ids = make_ids({"m1": ["a"], "m2": ["b"]})
        asset = shard_plan.build_durations_asset(junit_for({ids[0]: 1.0, ids[1]: 1.0}), ids)
        with self.assertRaises(shard_plan.PlanError):
            self.build(ids, asset, 0)
        with self.assertRaises(shard_plan.PlanError):
            self.build(ids, asset, 3)

    def test_load_plan_checks_pin_and_internal_consistency(self):
        ids, _, asset = synthetic(15, modules=6)
        plan = self.build(ids, asset, 3)
        raw = shard_plan.plan_bytes(plan)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard-plan.json"
            path.write_bytes(raw)
            loaded = shard_plan.load_plan(path, expected_sha256=digest)
            self.assertEqual(loaded["ids"], ids)
            with self.assertRaises(shard_plan.PlanError):
                shard_plan.load_plan(path, expected_sha256="0" * 64)
            bad = json.loads(raw)
            bad["shard_of"][0] = 99
            path.write_bytes(shard_plan.plan_bytes(bad))
            with self.assertRaises(shard_plan.PlanError):
                shard_plan.load_plan(path)
            bad = json.loads(raw)
            bad["ids"][0] = bad["ids"][0] + "x"
            path.write_bytes(shard_plan.plan_bytes(bad))
            with self.assertRaises(shard_plan.PlanError):
                shard_plan.load_plan(path)


class CommandLineTests(unittest.TestCase):
    def test_plan_and_durations_commands_write_the_expected_files(self):
        ids, seconds, _ = synthetic(21, modules=8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "collection.log").write_text(
                "\n".join(ids) + f"\n\nwarnings\n{len(ids)} tests collected in 1s\n"
            )
            (root / "collection.metadata.json").write_bytes(shard_plan.collection_metadata_bytes(ids))
            (root / "junit.xml").write_bytes(junit_for(seconds))
            code = shard_plan.main(
                [
                    "durations",
                    "--junit",
                    str(root / "junit.xml"),
                    "--collection-log",
                    str(root / "collection.log"),
                    "--out",
                    str(root / "durations.json"),
                ]
            )
            self.assertEqual(code, 0)
            code = shard_plan.main(
                [
                    "plan",
                    "--collection-log",
                    str(root / "collection.log"),
                    "--collection-metadata",
                    str(root / "collection.metadata.json"),
                    "--durations",
                    str(root / "durations.json"),
                    "--shards",
                    "3",
                    "--out",
                    str(root / "shard-plan.json"),
                ]
            )
            self.assertEqual(code, 0)
            plan = shard_plan.load_plan(root / "shard-plan.json")
            self.assertEqual(plan["n"], 3)
            self.assertEqual(plan["ids"], ids)
            self.assertEqual(
                plan["durations_sha256"], hashlib.sha256((root / "durations.json").read_bytes()).hexdigest()
            )
            (root / "collection.metadata.json").write_bytes(shard_plan.collection_metadata_bytes(ids[:-1]))
            code = shard_plan.main(
                [
                    "plan",
                    "--collection-log",
                    str(root / "collection.log"),
                    "--collection-metadata",
                    str(root / "collection.metadata.json"),
                    "--durations",
                    str(root / "durations.json"),
                    "--shards",
                    "3",
                    "--out",
                    str(root / "shard-plan-2.json"),
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse((root / "shard-plan-2.json").exists())


if __name__ == "__main__":
    unittest.main()
