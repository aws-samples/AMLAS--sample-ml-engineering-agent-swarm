"""Neutral source fixtures; no real Docker build, metadata or AWS call."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

NATIVE_BUILDX = "Name:   default\nDriver: docker\n\nNodes:\nName:      default\nEndpoint:  default\nStatus:    running\nBuildkit:  v0.12.6-m.1\nPlatforms: linux/arm64, linux/arm/v7, linux/arm/v6\nLabels:\n org.mobyproject.buildkit.worker.moby.host-gateway-ip: 172.17.0.1\n"

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "account_image_build", ROOT / "scripts/lib/account_image_build.py"
)


class AccountImageBuildTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(self.module)
        self.identity = {
            "accountId": "111122223333",
            "architecture": "arm64",
            "availabilityZone": "us-west-2b",
            "billingProducts": None,
            "devpayProductCodes": None,
            "imageId": "ami-0123456789abcdef0",
            "instanceId": "i-0123456789abcdef0",
            "instanceType": "m7g.large",
            "kernelId": None,
            "pendingTime": "2026-10-17T00:00:00Z",
            "privateIp": "10.0.0.1",
            "ramdiskId": None,
            "region": "us-west-2",
            "version": "2017-09-30",
            "marketplaceProductCodes": None,
        }
        self.calls = []
        self.builder_text = NATIVE_BUILDX

    def command(self, args):
        self.calls.append(args)
        if args[:3] == ["aws", "sts", "get-caller-identity"]:
            return {
                "Account": "111122223333",
                "UserId": "AROATEST:session",
                "Arn": "arn:aws:sts::111122223333:assumed-role/builder/session",
            }
        if "context" in args:
            return {"Host": "unix:///var/run/docker.sock", "SkipTLSVerify": False}
        if "info" in args:
            return {"Name": "ip-10-0-0-1", "OSType": "linux", "ID": "daemon-fixed"}
        if "buildx" in args:
            return {
                "Name": "default",
                "Driver": "docker",
                "Nodes": [{"Name": "default", "Endpoint": "default", "Status": "running"}],
            }
        self.fail("Unspecified fixture command")

    def run_guard(self, **kwargs):
        with (
            patch.object(self.module.platform, "system", return_value="Linux"),
            patch.object(self.module.socket, "gethostname", return_value="ip-10-0-0-1"),
            patch.object(self.module, "_instance_identity", return_value=self.identity),
            patch.object(self.module, "_command_json", side_effect=self.command),
            patch.object(self.module, "_local_socket", return_value=True),
            patch.object(self.module, "_command_text", return_value=self.builder_text, create=True),
            patch.dict(self.module.os.environ, {}, clear=True),
        ):
            return self.module.verify_build_host(
                kwargs.get("account", "111122223333"), kwargs.get("region", "us-west-2")
            )

    def test_verified_account_host_returns_closed_binding(self):
        value = self.run_guard()
        self.assertEqual(
            value,
            {
                "account": "111122223333",
                "region": "us-west-2",
                "instance_id": "i-0123456789abcdef0",
                "docker_host": "unix:///var/run/docker.sock",
                "builder": "default",
            },
        )

    def test_darwin_refuses_before_metadata_or_commands(self):
        with (
            patch.object(self.module.platform, "system", return_value="Darwin"),
            patch.object(self.module, "_instance_identity") as metadata,
            patch.object(self.module, "_command_json") as command,
        ):
            with self.assertRaises(self.module.BuildHostError):
                self.module.verify_build_host("111122223333", "us-west-2")
        metadata.assert_not_called()
        command.assert_not_called()

    def test_cloud_desktop_foreign_metadata_refuses_before_docker(self):
        self.identity["accountId"] = "000000000000"
        with self.assertRaises(self.module.BuildHostError):
            self.run_guard()
        self.assertFalse(any(args[0] == "docker" for args in self.calls))

    def test_wrong_region_refuses_before_docker(self):
        with self.assertRaises(self.module.BuildHostError):
            self.run_guard(region="us-east-1")
        self.assertFalse(any(args[0] == "docker" for args in self.calls))

    def test_unknown_identity_field_refuses(self):
        self.identity["unexpected"] = True
        with self.assertRaises(self.module.BuildHostError):
            self.run_guard()

    def test_selected_environment_account_cannot_replace_live_sts(self):
        original = self.command

        def changed(args):
            result = original(args)
            if args[0] == "aws":
                result["Account"] = "000000000000"
            return result

        self.command = changed
        with self.assertRaises(self.module.BuildHostError):
            self.run_guard()
        self.assertFalse(any(args[0] == "docker" for args in self.calls))

    def test_remote_context_refuses(self):
        original = self.command

        def changed(args):
            result = original(args)
            if "context" in args:
                result["Host"] = "ssh://builder.example"
            return result

        self.command = changed
        with self.assertRaises(self.module.BuildHostError):
            self.run_guard()

    def test_daemon_on_other_host_refuses(self):
        original = self.command

        def changed(args):
            result = original(args)
            if "info" in args:
                result["Name"] = "other-host"
            return result

        self.command = changed
        with self.assertRaises(self.module.BuildHostError):
            self.run_guard()

    def test_remote_buildx_driver_refuses(self):
        self.builder_text = NATIVE_BUILDX.replace("Driver: docker", "Driver: remote")
        with self.assertRaises(self.module.BuildHostError):
            self.run_guard()

    def test_non_socket_endpoint_refuses(self):
        with (
            patch.object(self.module, "_local_socket", return_value=False),
            patch.object(self.module.platform, "system", return_value="Linux"),
            patch.object(self.module, "_instance_identity", return_value=self.identity),
            patch.object(self.module, "_command_json", side_effect=self.command),
            patch.dict(self.module.os.environ, {}, clear=True),
        ):
            with self.assertRaises(self.module.BuildHostError):
                self.module.verify_build_host("111122223333", "us-west-2")

    def test_duplicate_native_json_fields_refuse(self):
        with self.assertRaises(self.module.BuildHostError):
            self.module._json(b'{"Account":"111122223333","Account":"000000000000"}')

    def test_duplicate_buildx_nodes_refuse(self):
        self.builder_text = NATIVE_BUILDX + "Name: other\nEndpoint: default\nStatus: running\n"
        with self.assertRaises(self.module.BuildHostError):
            self.run_guard()

    def test_actual_supported_native_buildx_capture(self):
        self.module._verify_builder_text(NATIVE_BUILDX)

    def test_plain_buildx_unknown_field_refuses(self):
        with self.assertRaises(self.module.BuildHostError):
            self.module._verify_builder_text(NATIVE_BUILDX + "Unknown: value\n")

    def test_plain_buildx_foreign_endpoint_refuses(self):
        with self.assertRaises(self.module.BuildHostError):
            self.module._verify_builder_text(
                NATIVE_BUILDX.replace("Endpoint:  default", "Endpoint:  ssh://builder.example")
            )

    def test_remote_environment_override_refuses_before_docker(self):
        with (
            patch.object(self.module.platform, "system", return_value="Linux"),
            patch.object(self.module, "_instance_identity", return_value=self.identity),
            patch.object(self.module, "_command_json", side_effect=self.command),
            patch.dict(self.module.os.environ, {"DOCKER_HOST": "tcp://builder.example:2376"}, clear=True),
        ):
            with self.assertRaises(self.module.BuildHostError):
                self.module.verify_build_host("111122223333", "us-west-2")
        self.assertFalse(any(args[0] == "docker" for args in self.calls))

    def test_location_check_has_no_docker_binding_or_calls(self):
        with (
            patch.object(self.module.platform, "system", return_value="Linux"),
            patch.object(self.module, "_instance_identity", return_value=self.identity),
            patch.object(self.module, "_command_json", side_effect=self.command),
            patch.dict(self.module.os.environ, {}, clear=True),
        ):
            binding = self.module.verify_build_host("111122223333", "us-west-2", location_only=True)
        self.assertEqual(binding["docker_host"], "")
        self.assertEqual(binding["builder"], "")
        self.assertFalse(any(args[0] == "docker" for args in self.calls))

    def test_custom_sts_endpoint_refuses_before_native_identity(self):
        for selector in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_STS"):
            with (
                self.subTest(selector=selector),
                patch.object(self.module.platform, "system", return_value="Linux"),
                patch.dict(self.module.os.environ, {selector: "https://example.invalid"}, clear=True),
                patch.object(self.module, "_instance_identity") as metadata,
            ):
                with self.assertRaises(self.module.BuildHostError):
                    self.module.verify_build_host("111122223333", "us-west-2")
                metadata.assert_not_called()

    def test_sts_request_pins_native_regional_endpoint(self):
        self.run_guard()
        caller = next(args for args in self.calls if args[0] == "aws")
        self.assertEqual(caller[caller.index("--endpoint-url") + 1], "https://sts.us-west-2.amazonaws.com")

    def test_runtime_guard_refusal_executes_no_auth_or_docker_operation(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            base = Path(directory)
            log = base / "calls"
            for name in ("python3", "aws", "docker"):
                executable = base / name
                executable.write_text(
                    '#!/bin/bash\nprintf "%s\\n" "' + name + '" >> "$FIXTURE_CALL_LOG"\nexit 1\n'
                )
                executable.chmod(0o700)
            env = dict(os.environ)
            env.update(
                PATH=str(base) + ":" + env["PATH"],
                AWS_REGION="us-west-2",
                AMLAS_AWS_ACCOUNT_ID="111122223333",
                FIXTURE_CALL_LOG=str(log),
                AMLAS_DEPLOYMENT_PREFIX="fixture-build",
                NON_INTERACTIVE="1",
            )
            result = subprocess.run(
                ["/bin/bash", str(ROOT / "scripts/lib/build_runtime_image.sh")],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(log.read_text().splitlines(), ["python3"])

    def test_runtime_guard_precedes_authentication_and_build(self):
        text = (ROOT / "scripts/lib/build_runtime_image.sh").read_text()
        self.assertLess(text.index("require_account_build_host"), text.index("get-login-password"))
        self.assertIn("--builder default", text)
        self.assertIn('--host "$AMLAS_VERIFIED_DOCKER_HOST"', text)
        self.assertNotIn("ACCT=${AWS_ACCOUNT_ID:-", text)

    def test_ci_runs_pinned_image_without_constructing_an_image(self):
        text = (ROOT / ".github/workflows/public-release.yml").read_text().split("  verify:", 1)[0]
        self.assertIn("runs-on: ubuntu-latest", text)
        self.assertNotIn(" build --platform", text)
        self.assertNotIn(" buildx build", text)
        self.assertNotIn("docker commit", text)
        self.assertIn("docker run --rm --platform linux/amd64", text)
        self.assertIn("RUSTUP_HOME=/worktmp/rustup", text)
        self.assertIn("--network none", text)
        self.assertEqual(
            text.count("@sha256:ee9b9f2ebd8741e4ecf1ade08918e492d642c4e625be481bb7b7954b89e9d504"), 2
        )

    def test_setup_guard_runs_before_host_prerequisite_changes(self):
        text = (ROOT / "scripts/setup.sh").read_text()
        self.assertLess(
            text.index('require_account_build_host "$AMLAS_AWS_ACCOUNT_ID"'),
            text.index("prepare_host_prerequisites\n"),
        )

    def test_workspace_only_buildx_survives_isolated_auth_config(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            base = Path(directory)
            original_config = base / "workspace-docker-config"
            plugins = original_config / "cli-plugins"
            plugins.mkdir(parents=True)
            plugin = plugins / "docker-buildx"
            plugin.write_text("#!/bin/sh\nexit 0\n")
            plugin.chmod(0o700)
            original_json = original_config / "config.json"
            original_bytes = b'{"fixture_original_configuration":"unchanged"}\n'
            original_json.write_bytes(original_bytes)
            log = base / "calls.jsonl"
            executable = base / "fixture-tool"
            executable.write_text(
                "#!"
                + sys.executable
                + "\n"
                + r"""
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
config = pathlib.Path(os.environ['DOCKER_CONFIG'])
if name == 'docker':
    assert args[:1] == ['--config'], args
    config = pathlib.Path(args[1]); args = args[2:]
    assert args[:2] == ['--host', 'unix:///var/run/docker.sock'], args
    args = args[2:]
with pathlib.Path(os.environ['FIXTURE_CALL_LOG']).open('a') as stream:
    stream.write(json.dumps({'tool':name,'args':args,'config':str(config)})+'\n')
if name == 'python3':
    assert (config/'cli-plugins/docker-buildx').is_file(), 'old guarded plugin missing'
    print('111122223333\tus-west-2\ti-0123456789abcdef0\tunix:///var/run/docker.sock\tdefault')
    sys.exit(0)
if name == 'aws':
    assert args[:2] == ['ecr','get-login-password'], args
    print('fixture-only-auth')
    sys.exit(0)
if args[0] == 'login':
    assert sys.stdin.read().strip() == 'fixture-only-auth'
    (config/'config.json').write_text('{"fixture_auth":true}\n')
    sys.exit(0)
if args[:2] == ['buildx','build']:
    plugin = config/'cli-plugins/docker-buildx'
    if not plugin.is_file(): sys.exit(86)
    sys.exit(subprocess.run([str(plugin)]).returncode)
if args[0] == 'logout':
    (config/'config.json').write_text('{}\n')
    sys.exit(0)
raise AssertionError((name,args))
"""
            )
            executable.chmod(0o700)
            for name in ("python3", "aws", "docker"):
                (base / name).symlink_to(executable)
            env = dict(os.environ)
            env.update(
                PATH=str(base) + ":" + env["PATH"],
                AWS_REGION="us-west-2",
                AMLAS_AWS_ACCOUNT_ID="111122223333",
                FIXTURE_CALL_LOG=str(log),
                AMLAS_DEPLOYMENT_PREFIX="fixture-build",
                NON_INTERACTIVE="1",
                DOCKER_CONFIG=str(original_config),
            )
            result = subprocess.run(
                ["/bin/bash", str(ROOT / "scripts/lib/build_runtime_image.sh")],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            guard = next(item for item in calls if item["tool"] == "python3")
            built = next(
                item for item in calls if item["tool"] == "docker" and item["args"][:2] == ["buildx", "build"]
            )
            self.assertEqual(guard["config"], str(original_config))
            self.assertNotEqual(built["config"], str(original_config))
            copied = Path(built["config"]) / "cli-plugins/docker-buildx"
            self.assertEqual(copied.read_bytes(), plugin.read_bytes())
            self.assertEqual(original_json.read_bytes(), original_bytes)
            self.assertEqual(json.loads((Path(built["config"]) / "config.json").read_text()), {})

    def test_auth_directory_failure_refuses_before_auth_or_docker(self):
        for mktemp_exit in (42, 0):
            with (
                self.subTest(mktemp_exit=mktemp_exit),
                tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory,
            ):
                base = Path(directory)
                original = base / "original-config"
                original.mkdir()
                config_json = original / "config.json"
                original_bytes = b'{"fixture_original":"untouched"}\n'
                config_json.write_bytes(original_bytes)
                log = base / "calls.jsonl"
                executable = base / "fixture-tool"
                executable.write_text(
                    "#!"
                    + sys.executable
                    + "\n"
                    + r"""
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with pathlib.Path(os.environ['FIXTURE_CALL_LOG']).open('a') as stream:
    stream.write(json.dumps({'tool':name,'args':args})+'\n')
if name == 'python3':
    print('111122223333\tus-west-2\ti-0123456789abcdef0\tunix:///var/run/docker.sock\tdefault')
    sys.exit(0)
if name == 'mktemp':
    sys.exit(int(os.environ['FIXTURE_MKTEMP_EXIT']))
if name == 'aws':
    print('fixture-only-auth')
    sys.exit(0)
if name == 'docker':
    operation = args[4]
    if operation == 'login':
        sys.stdin.read()
        original = pathlib.Path(os.environ['FIXTURE_ORIGINAL_CONFIG'])
        (original/'config.json').write_text('{"fixture_shared_write":true}\n')
    sys.exit(0)
raise AssertionError((name,args))
"""
                )
                executable.chmod(0o700)
                for name in ("python3", "mktemp", "aws", "docker"):
                    (base / name).symlink_to(executable)
                env = dict(os.environ)
                env.update(
                    PATH=str(base) + ":" + env["PATH"],
                    AWS_REGION="us-west-2",
                    AMLAS_AWS_ACCOUNT_ID="111122223333",
                    AMLAS_DEPLOYMENT_PREFIX="fixture-build",
                    DOCKER_CONFIG=str(original),
                    FIXTURE_CALL_LOG=str(log),
                    FIXTURE_ORIGINAL_CONFIG=str(original),
                    FIXTURE_MKTEMP_EXIT=str(mktemp_exit),
                )
                result = subprocess.run(
                    ["/bin/bash", str(ROOT / "scripts/lib/build_runtime_image.sh")],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual([item["tool"] for item in calls], ["python3", "mktemp"])
                self.assertEqual(config_json.read_bytes(), original_bytes)

    def test_build_entry_auth_is_scoped_outside_home(self):
        text = (ROOT / "scripts/lib/build_runtime_image.sh").read_text()
        self.assertIn("export DOCKER_CONFIG=", text)
        self.assertIn("--config", text)


if __name__ == "__main__":
    unittest.main()
