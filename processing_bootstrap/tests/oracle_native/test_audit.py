"""Pure receipt controls; no native process or socket is created."""
import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("oracle_fixture_run", HERE / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class ReceiptAudit(unittest.TestCase):
    def receipt(self, mode):
        names = set(RUN.BASE_CONTROLS)
        if mode != "complete":
            names |= {"udp_send_unfiltered_without_seccomp", "original_child_reaped"}
        if mode == "baseline":
            names.add("peer_memory_value")
        return "\n".join(f"CONTROL {mode} {name} PASS rc=0 errno=0" for name in sorted(names)) + (
            f"\nFIXTURE {mode} PASS original_pid=42 wait_status=0 reaped=true\n"
        )

    def test_complete_inventory_and_original_pid_reap_are_required(self):
        for mode in ("baseline", "complete", "landlock"):
            text = self.receipt(mode)
            self.assertEqual(RUN.audit(mode, 0, text)["mode"], mode)
            mutations = (
                text.replace("stdout PASS", "stdout FAIL"),
                text.replace("source_read_0", "source_read_1"),
                text.replace("reaped=true", "reaped=false"),
                text.replace("wait_status=0", "wait_status=256"),
                text + f"CONTROL {mode} stdout PASS rc=0 errno=0\n",
            )
            for mutation in mutations:
                with self.assertRaises(ValueError):
                    RUN.audit(mode, 0, mutation)
            with self.assertRaises(ValueError):
                RUN.audit(mode, 125, text)

    def test_baseline_and_layer_controls_cannot_be_relabelled_as_complete(self):
        for mode in ("baseline", "landlock"):
            with self.assertRaises(ValueError):
                RUN.audit("complete", 0, self.receipt(mode).replace(mode, "complete"))


if __name__ == "__main__":
    unittest.main()
