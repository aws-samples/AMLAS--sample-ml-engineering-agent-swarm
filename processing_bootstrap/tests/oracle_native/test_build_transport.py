"""Neutral mocked local-builder contracts; no Docker or fixture execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("oracle_fixture_build_control", HERE / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
IMAGE_ID = "sha256:c33493df0f88c70bb7c359cfac9ded885d1be32938e5c3b10e0f234e215a883f"
SOURCE = "0d8bf7bee8c100a002c16ec7754ce3f1448c435a"  # pragma: allowlist secret - reviewed immutable integrity pin
HELPER_SHA = "f8403f6b346cd3ed71c0b0a4c92454fa70675b2853ca3498d8028021b3288b9e"  # pragma: allowlist secret - reviewed immutable integrity pin
CONTEXT = [{"Name": "default", "Endpoints": {"docker": {"Host": "unix:///var/run/docker.sock"}}}]
BUILDER = "Name: default\nDriver: docker\nNodes:\nName: default\nEndpoint: default\nStatus: running\n"
PROOF = {"execution_environment": "owned-aws-ec2", "same_account_targets": True}


def result(value, code=0):
    return SimpleNamespace(returncode=code, stdout=json.dumps(value).encode() if isinstance(value, list) else value)


def image(tag):
    return [{"Id": IMAGE_ID, "Os": "linux", "Architecture": "amd64", "RepoTags": [tag],
             "Config": {"User": "10001:10001", "Labels": {"org.opencontainers.image.revision": SOURCE},
                        "Env": ["AMLAS_CODING_PAIR_SOURCE_COMMIT=" + SOURCE]}}]


class LocalBuilderTransport(unittest.TestCase):
    def test_base_tag_names_immutable_config_and_owned_helper_scope(self):
        tag = RUN.local_base_tag(IMAGE_ID, HELPER_SHA)
        self.assertEqual(tag, "amlas-oracle-native-base-f8403f6b346c:" + IMAGE_ID[7:])
        for invalid in ("fixture:latest", "fixture@" + IMAGE_ID):
            with self.assertRaises(ValueError):
                RUN.local_base_tag(invalid, HELPER_SHA)

    def test_builder_requires_one_local_docker_node_and_same_account_scope(self):
        RUN.validate_local_builder(PROOF, "default", CONTEXT, BUILDER, "daemon-id")
        for selected, context, builder, daemon in (
            ("remote", CONTEXT, BUILDER, "daemon-id"),
            ("default", [{"Name": "default", "Endpoints": {"docker": {"Host": "tcp://remote"}}}], BUILDER, "daemon-id"),
            ("default", CONTEXT, BUILDER.replace("Driver: docker", "Driver: docker-container"), "daemon-id"),
            ("default", CONTEXT, BUILDER + "Name: another\nEndpoint: default\nStatus: running\n", "daemon-id"),
            ("default", CONTEXT, BUILDER, ""),
        ):
            with self.assertRaises(ValueError):
                RUN.validate_local_builder(PROOF, selected, context, builder, daemon)
        with self.assertRaises(ValueError):
            RUN.validate_local_builder({}, "default", CONTEXT, BUILDER, "daemon-id")
        with patch.object(RUN, "command") as call:
            with self.assertRaisesRegex(ValueError, "account_execution_proof"):
                RUN.build_with_local_base({}, IMAGE_ID, SOURCE, HELPER_SHA, "", HERE, HERE, "owned-output")
            call.assert_not_called()

    def responses(self, tag, after=None, build_code=0, existing=None):
        return [
            (result(b"default\n"), 0), (result(CONTEXT), 0),
            (result(BUILDER.encode()), 0), (result(b"daemon-id\n"), 0),
            (result(existing or ("Error: No such image: " + tag).encode(), 0 if existing else 1), 0),
            *(([(result(b""), 0)] if existing is None else [])),
            (result(image(tag)), 0), (result(b"build-log\n", build_code), 0),
            (result(after or image(tag)), 0), (result(b"daemon-id\n"), 0),
        ]

    def build(self, output):
        return RUN.build_with_local_base(PROOF, IMAGE_ID, SOURCE, HELPER_SHA,
                                         "FROM " + IMAGE_ID + "\nUSER 10001:10001\n",
                                         output, output, "owned-fixture-output")

    def test_build_binds_and_rechecks_tag_and_uses_one_explicit_local_builder(self):
        tag = RUN.local_base_tag(IMAGE_ID, HELPER_SHA)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.dict(RUN.os.environ, {}, clear=True), patch.object(RUN, "command", side_effect=self.responses(tag)) as call:
                record = self.build(output)
            self.assertEqual((output / "Dockerfile.fixture").read_text(), "FROM " + tag + "\nUSER 10001:10001\n")
        commands = [item.args[0] for item in call.call_args_list]
        self.assertIn(["docker", "image", "tag", IMAGE_ID, tag], commands)
        build = next(item for item in commands if item[:3] == ["docker", "buildx", "build"])
        for item in ("--builder", "default", "--load", "--pull=false", "--network", "none"):
            self.assertIn(item, build)
        self.assertNotIn("--push", build)
        self.assertFalse(record["atomic_tag_binding"])
        self.assertTrue(record["trusted_local_writers_assumed"])
        self.assertEqual(record["base_config_id"], IMAGE_ID)

    def test_existing_conflicting_tag_refuses_before_tag_or_build(self):
        tag = RUN.local_base_tag(IMAGE_ID, HELPER_SHA)
        wrong = image(tag)
        wrong[0]["Id"] = "sha256:" + "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(RUN.os.environ, {}, clear=True), patch.object(RUN, "command", side_effect=self.responses(tag, existing=wrong)) as call:
                with self.assertRaises(ValueError):
                    self.build(Path(directory))
        self.assertFalse(any(item.args[0][:3] in (["docker", "image", "tag"], ["docker", "buildx", "build"]) for item in call.call_args_list))

    def test_postbuild_tag_drift_and_build_failure_cannot_produce_success(self):
        tag = RUN.local_base_tag(IMAGE_ID, HELPER_SHA)
        wrong = image(tag)
        wrong[0]["Id"] = "sha256:" + "0" * 64
        for responses in (self.responses(tag, after=wrong), self.responses(tag, build_code=1)):
            with tempfile.TemporaryDirectory() as directory:
                with patch.dict(RUN.os.environ, {}, clear=True), patch.object(RUN, "command", side_effect=responses):
                    with self.assertRaises(ValueError):
                        self.build(Path(directory))


if __name__ == "__main__":
    unittest.main()
