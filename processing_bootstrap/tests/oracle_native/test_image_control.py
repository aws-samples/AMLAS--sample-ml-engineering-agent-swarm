"""Neutral mocked Docker metadata contracts; no Docker or native execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("oracle_fixture_run_control", HERE / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
PREPARE_SPEC = importlib.util.spec_from_file_location("oracle_fixture_prepare_control", HERE / "prepare.py")
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(PREPARE)
IMAGE_ID = "sha256:4cd30ffe6fb8e4f71623a5fa779cf3d65f5761972ffd11cf0aec5f95468beb61"
# Historical cold image/source pair is retained; it cannot be mixed with the new source.
SOURCE = "f00886683c6b405b6e7225b90ef2ccc9f09f7378"  # pragma: allowlist secret - reviewed immutable integrity pin
QUALIFIED_IMAGE_ID = "sha256:c33493df0f88c70bb7c359cfac9ded885d1be32938e5c3b10e0f234e215a883f"
QUALIFIED_SOURCE = "0d8bf7bee8c100a002c16ec7754ce3f1448c435a"  # pragma: allowlist secret - reviewed immutable integrity pin
CONTAINER_ID = "a" * 64


def image():
    return {
        "Id": IMAGE_ID, "Os": "linux", "Architecture": "amd64", "RepoDigests": [],
        "Config": {"User": "10001:10001", "Labels": {"org.opencontainers.image.revision": SOURCE},
                   "Env": ["AMLAS_CODING_PAIR_SOURCE_COMMIT=" + SOURCE]},
    }


def container(running=False):
    return {
        "Id": CONTAINER_ID, "Image": IMAGE_ID, "Config": {"User": "10001:10001"},
        "HostConfig": {
            "NetworkMode": "none", "Privileged": False, "CapDrop": ["ALL"], "CapAdd": [],
            "SecurityOpt": ["no-new-privileges:true", "seccomp=unconfined"],
            "PidMode": "", "IpcMode": "private", "Binds": [], "Devices": [], "GroupAdd": [],
        },
        "Mounts": [],
        "State": {"Running": running, "Pid": 42 if running else 0,
                  "Status": "running" if running else "exited", "ExitCode": 0},
    }


def result(value, code=0):
    return SimpleNamespace(returncode=code, stdout=(json.dumps([value]).encode() if isinstance(value, dict) else value))


class LocalImagePin(unittest.TestCase):
    def test_generated_manifest_matches_approved_cold_image_source_pair(self):
        cold = image()
        cold["Id"] = QUALIFIED_IMAGE_ID
        cold["Config"]["Labels"]["org.opencontainers.image.revision"] = QUALIFIED_SOURCE
        cold["Config"]["Env"] = ["AMLAS_CODING_PAIR_SOURCE_COMMIT=" + QUALIFIED_SOURCE]
        with tempfile.TemporaryDirectory() as directory:
            receipt = PREPARE.prepare(HERE.parents[2], Path(directory) / "prepared")
        observed = RUN.validate_local_image(receipt["qualified_image_config_id"],
                                           QUALIFIED_IMAGE_ID,
                                           receipt["qualified_source_commit"], [cold])
        self.assertEqual(observed["Id"], QUALIFIED_IMAGE_ID)
        self.assertEqual(receipt["qualified_source_commit"], QUALIFIED_SOURCE)
        self.assertEqual(receipt["qualified_launcher_sha256"], PREPARE.LAUNCHER_SHA)
        self.assertEqual(receipt["source_sha256"], PREPARE.PINS)

    def test_historical_image_source_pair_cannot_accept_new_source_stamp(self):
        with self.assertRaisesRegex(ValueError, "image_inspected_metadata_pin"):
            RUN.validate_local_image(IMAGE_ID, IMAGE_ID, QUALIFIED_SOURCE, [image()])

    def test_unpublished_exact_local_config_id_with_no_repo_digest_is_accepted(self):
        self.assertEqual(RUN.validate_local_image(IMAGE_ID, IMAGE_ID, SOURCE, [image()])["Id"], IMAGE_ID)
        for selected in ("fixture:latest", "fixture@" + IMAGE_ID, "sha256:" + "0" * 64):
            with self.assertRaises(ValueError):
                RUN.validate_local_image(selected, IMAGE_ID, SOURCE, [image()])

    def test_inspected_id_source_stamp_user_and_platform_must_match(self):
        for path, replacement in (
            (("Id",), "sha256:" + "0" * 64),
            (("Os",), "other"), (("Architecture",), "arm64"),
            (("Config", "User"), "root"),
            (("Config", "Labels", "org.opencontainers.image.revision"), "c" * 40),
            (("Config", "Env"), ["AMLAS_CODING_PAIR_SOURCE_COMMIT=" + "c" * 40]),
            (("Config", "Env"), ["AMLAS_CODING_PAIR_SOURCE_COMMIT=" + SOURCE] * 2),
        ):
            changed = image()
            target = changed
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = replacement
            with self.assertRaises(ValueError):
                RUN.validate_local_image(IMAGE_ID, IMAGE_ID, SOURCE, [changed])


class OriginalContainerCleanup(unittest.TestCase):
    def test_inspection_must_bind_original_id_image_and_fixture_limits(self):
        RUN.validate_container(CONTAINER_ID, IMAGE_ID, container())
        changes = (
            ("Id", "b" * 64), ("Image", "sha256:" + "0" * 64),
            ("Config.User", "root"), ("HostConfig.NetworkMode", "host"),
            ("HostConfig.Privileged", True), ("HostConfig.CapAdd", ["SYS_ADMIN"]),
            ("HostConfig.GroupAdd", ["10001"]), ("HostConfig.PidMode", "host"),
            ("HostConfig.IpcMode", "host"), ("Mounts", [{"Type": "bind"}]),
            ("HostConfig.SecurityOpt", ["seccomp=unconfined"]),
        )
        for path, replacement in changes:
            changed = container()
            target = changed
            parts = path.split(".")
            for component in parts[:-1]:
                target = target[component]
            target[parts[-1]] = replacement
            with self.assertRaises(ValueError):
                RUN.validate_container(CONTAINER_ID, IMAGE_ID, changed)
        missing = container()
        del missing["HostConfig"]["PidMode"]
        with self.assertRaises(ValueError):
            RUN.validate_container(CONTAINER_ID, IMAGE_ID, missing)

    def test_running_original_is_inspected_before_kill_and_stopped_before_remove(self):
        responses = [
            (result(container(True)), 0), (result(b"stopped\n"), 0),
            (result(container(False)), 0), (result(CONTAINER_ID.encode()), 0),
            (result(("Error: No such object: " + CONTAINER_ID).encode(), 1), 0),
        ]
        with patch.object(RUN, "command", side_effect=responses) as call:
            proof = RUN.cleanup_container(CONTAINER_ID, IMAGE_ID, Path("evidence"), "timeout")
        self.assertTrue(proof["stopped_confirmed"])
        self.assertTrue(proof["removed_confirmed"])
        verbs = [args[0][0][1:3] for args in call.call_args_list]
        self.assertEqual(verbs, [["container", "inspect"], ["container", "kill"],
                                 ["container", "inspect"], ["container", "rm"],
                                 ["container", "inspect"]])

    def test_missing_or_substituted_original_identity_never_reaches_kill(self):
        with self.assertRaisesRegex(ValueError, "cleanup_original_identity"):
            RUN.cleanup_container("", IMAGE_ID, Path("evidence"), "timeout")
        changed = container(True)
        changed["Id"] = "b" * 64
        with patch.object(RUN, "command", return_value=(result(changed), 0)) as call:
            with self.assertRaises(ValueError):
                RUN.cleanup_container(CONTAINER_ID, IMAGE_ID, Path("evidence"), "timeout")
            self.assertEqual(call.call_count, 1)
        with patch.object(RUN, "command", return_value=(result(b"unavailable", 1), 0)) as call:
            with self.assertRaises(ValueError):
                RUN.cleanup_container(CONTAINER_ID, IMAGE_ID, Path("evidence"), "timeout")
            self.assertEqual(call.call_count, 1)

    def test_kill_success_does_not_prove_stopped_or_removal(self):
        with patch.object(RUN, "command", side_effect=[
            (result(container(True)), 0), (result(b"stopped\n"), 0),
            (result(container(True)), 0),
        ]) as call:
            with self.assertRaisesRegex(ValueError, "cleanup_stop_unconfirmed"):
                RUN.cleanup_container(CONTAINER_ID, IMAGE_ID, Path("evidence"), "timeout")
            self.assertEqual(call.call_count, 3)


if __name__ == "__main__":
    unittest.main()
