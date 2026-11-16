"""Account-only preparation regressions; these execute no native helper."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("oracle_fixture_prepare", HERE / "prepare.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class FrozenPreparation(unittest.TestCase):
    def prepare(self, output):
        return PREPARE.prepare(ROOT, output)

    def test_composite_retains_exact_policy_prefix_and_three_include_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            receipt = self.prepare(output)
            policy = (ROOT / PREPARE.POLICY_PATH).read_bytes()
            composite = (output / PREPARE.POLICY_PATH).read_bytes()
            self.assertEqual(composite[: len(policy)], policy)
            self.assertEqual(hashlib.sha256(policy).hexdigest(), PREPARE.PINS[PREPARE.POLICY_PATH])
            self.assertEqual(receipt["policy_bytes"], len(policy))
            for path in PREPARE.PYTHON_PATHS:
                self.assertEqual((output / path).read_bytes(), (ROOT / path).read_bytes())
            self.assertFalse(receipt["native_admission"])
            self.assertEqual(receipt["policy_modes"], ["baseline", "complete", "landlock"])
            self.assertEqual(json.loads((output / "manifest.json").read_text()), receipt)

    def test_changed_production_source_refuses_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "fixture"
            for path in PREPARE.PINS:
                target = source / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / path).read_bytes())
            with (source / PREPARE.POLICY_PATH).open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "source_pin"):
                PREPARE.prepare(source, output)
            self.assertFalse(output.exists())

    def test_existing_output_refuses_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "output_exists"):
                self.prepare(output)
            self.assertEqual(list(output.iterdir()), [])

    def test_suffix_is_test_only_and_main_is_standalone(self):
        suffix = (HERE / "fixture_suffix.rs").read_text()
        self.assertIn("#[cfg(test)]", suffix)
        self.assertIn("fn main()", suffix)
        self.assertNotIn("#[test]", suffix)
        self.assertNotIn("Command::", suffix)


if __name__ == "__main__":
    unittest.main()
