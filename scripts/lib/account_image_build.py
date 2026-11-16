"""Verify an account-local EC2 image build host before Docker mutations."""

import argparse
import json
import os
import platform
import re
import socket
import stat
import subprocess
import urllib.request


class BuildHostError(RuntimeError):
    """The selected account-local build location cannot be established."""


def _json(raw):
    if not isinstance(raw, bytes) or len(raw) > 262144:
        raise BuildHostError("Build-host metadata exceeded its bound.")
    try:

        def closed_pairs(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate field")
                value[key] = item
            return value

        value = json.loads(raw, object_pairs_hook=closed_pairs)
    except (UnicodeError, ValueError) as exc:
        raise BuildHostError("Build-host metadata was not valid JSON.") from exc
    if not isinstance(value, dict):
        raise BuildHostError("Build-host metadata must be an object.")
    return value


def _instance_identity():
    # Fixed link-local endpoint, IMDSv2 only, no proxy or caller-selected URL.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with opener.open(request, timeout=3) as response:
        raw_token = response.read(4097)
    if not raw_token or len(raw_token) > 4096:
        raise BuildHostError("IMDSv2 token was missing or oversized.")
    token = raw_token.decode("ascii")
    if any(ord(char) < 33 or ord(char) > 126 for char in token):
        raise BuildHostError("IMDSv2 token was malformed.")
    request = urllib.request.Request(
        "http://169.254.169.254/latest/dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with opener.open(request, timeout=3) as response:
        return _json(response.read(262145))


def _command_json(args):
    result = subprocess.run(args, capture_output=True, check=True, timeout=20)
    return _json(result.stdout)


def _command_text(args):
    result = subprocess.run(args, capture_output=True, check=True, timeout=20)
    if len(result.stdout) > 32768:
        raise BuildHostError("Buildx inspection exceeded its bound.")
    return result.stdout.decode("utf-8")


def _verify_builder_text(text):
    if not isinstance(text, str) or len(text) > 32768:
        raise BuildHostError("Buildx inspection was missing or oversized.")
    header, node, labels = {}, {}, {}
    section = header
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "Nodes:" and section is header:
            section = node
            continue
        if line == "Labels:" and section is node:
            section = labels
            continue
        if ":" not in line:
            raise BuildHostError("Buildx inspection fields were not recognized.")
        key, value = (item.strip() for item in line.split(":", 1))
        allowed = (
            {"Name", "Driver", "Last Activity"}
            if section is header
            else {"Name", "Endpoint", "Status", "Buildkit", "BuildKit version", "Platforms"}
            if section is node
            else {"org.mobyproject.buildkit.worker.moby.host-gateway-ip"}
        )
        if key not in allowed or key in section or not value or len(value) > 4096:
            raise BuildHostError("Buildx inspection fields were not recognized.")
        section[key] = value
    if (
        header.get("Name") != "default"
        or header.get("Driver") != "docker"
        or node.get("Name") != "default"
        or node.get("Endpoint") != "default"
        or node.get("Status") != "running"
    ):
        raise BuildHostError("Buildx must use the running default local Docker driver.")


def _local_socket(path):
    return stat.S_ISSOCK(os.stat(path, follow_symlinks=False).st_mode)


def _identity_fields(value, account, region):
    required = {
        "accountId",
        "architecture",
        "availabilityZone",
        "billingProducts",
        "devpayProductCodes",
        "imageId",
        "instanceId",
        "instanceType",
        "kernelId",
        "pendingTime",
        "privateIp",
        "ramdiskId",
        "region",
        "version",
    }
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - {"marketplaceProductCodes"}
    ):
        raise BuildHostError("EC2 identity document fields were not recognized.")
    for key in required - {"billingProducts", "devpayProductCodes", "kernelId", "ramdiskId"}:
        if not isinstance(value[key], str) or not value[key] or len(value[key]) > 256:
            raise BuildHostError("EC2 identity document types were not recognized.")
    for key in ("billingProducts", "devpayProductCodes", "marketplaceProductCodes"):
        field = value.get(key)
        if field is not None and (
            not isinstance(field, list)
            or len(field) > 32
            or any(not isinstance(item, str) or len(item) > 256 for item in field)
        ):
            raise BuildHostError("EC2 product identity types were not recognized.")
    for key in ("kernelId", "ramdiskId"):
        if value[key] is not None and (not isinstance(value[key], str) or len(value[key]) > 256):
            raise BuildHostError("EC2 identity document types were not recognized.")
    if value["accountId"] != account or value["region"] != region or value["version"] != "2017-09-30":
        raise BuildHostError("Build EC2 account or Region differs from the selected target.")
    if value["architecture"] not in {"arm64", "x86_64"} or not re.fullmatch(
        r"i-[0-9a-f]{8,17}", value["instanceId"]
    ):
        raise BuildHostError("EC2 build identity was not recognized.")
    if not value["availabilityZone"].startswith(region):
        raise BuildHostError("EC2 availability zone differs from the target Region.")


def verify_build_host(account, region, *, location_only=False):
    if platform.system() != "Linux":
        raise BuildHostError("Image builds and packaging require Linux EC2 in the selected AWS account.")
    if not re.fullmatch(r"[0-9]{12}", account) or not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-[0-9]+", region):
        raise BuildHostError("A 12-digit target account and explicit AWS Region are required.")
    if any(os.environ.get(key) for key in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_STS")):
        raise BuildHostError("Custom AWS endpoint overrides are unsupported for build identity verification.")
    identity = _instance_identity()
    _identity_fields(identity, account, region)
    caller = _command_json(
        [
            "aws",
            "sts",
            "get-caller-identity",
            "--region",
            region,
            "--endpoint-url",
            "https://sts." + region + (".amazonaws.com.cn" if region.startswith("cn-") else ".amazonaws.com"),
            "--output",
            "json",
        ]
    )
    if (
        set(caller) != {"Account", "Arn", "UserId"}
        or caller.get("Account") != account
        or any(not isinstance(caller.get(key), str) or not caller[key] for key in ("Arn", "UserId"))
    ):
        raise BuildHostError("Fresh STS caller identity differs from the selected build account.")
    if not re.fullmatch(r"arn:aws(?:-us-gov|-cn)?:[a-z0-9-]+::" + account + r":[^\s]+", caller["Arn"]):
        raise BuildHostError("STS ARN does not identify the selected account.")
    binding = {
        "account": account,
        "region": region,
        "instance_id": identity["instanceId"],
        "docker_host": "",
        "builder": "",
    }
    if location_only:
        return binding
    if any(
        os.environ.get(key) for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "BUILDX_BUILDER", "BUILDKIT_HOST")
    ):
        raise BuildHostError(
            "Unset Docker host/context/builder overrides; only the verified local daemon is supported."
        )
    endpoint = _command_json(
        ["docker", "context", "inspect", "default", "--format", "{{json .Endpoints.docker}}"]
    )
    if (
        set(endpoint) != {"Host", "SkipTLSVerify"}
        or endpoint.get("Host") != "unix:///var/run/docker.sock"
        or endpoint.get("SkipTLSVerify") is not False
        or not _local_socket("/var/run/docker.sock")
    ):
        raise BuildHostError("Docker must use the local Unix socket on the verified EC2 host.")
    docker = ["docker", "--host", endpoint["Host"]]
    daemon = _command_json(
        [*docker, "info", "--format", '{"Name":{{json .Name}},"OSType":{{json .OSType}},"ID":{{json .ID}}}']
    )
    if (
        set(daemon) != {"Name", "OSType", "ID"}
        or daemon.get("Name") != socket.gethostname()
        or daemon.get("OSType") != "linux"
        or not isinstance(daemon.get("ID"), str)
        or not daemon["ID"]
    ):
        raise BuildHostError("Docker daemon does not identify this verified Linux EC2 host.")
    _verify_builder_text(_command_text([*docker, "buildx", "inspect", "default"]))
    binding.update(docker_host=endpoint["Host"], builder="default")
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--location-only",
        action="store_true",
        help="Verify EC2/STS before host preparation; does not authorize Docker.",
    )
    parser.add_argument("--shell-binding", action="store_true")
    args = parser.parse_args()
    try:
        binding = verify_build_host(args.account, args.region, location_only=args.location_only)
    except (BuildHostError, OSError, ValueError, subprocess.SubprocessError) as exc:
        message = str(exc) if isinstance(exc, BuildHostError) else "Native build-host verification failed."
        parser.exit(1, "Image build host error: " + message + "\n")
    if args.shell_binding:
        print(
            "\t".join(binding[key] for key in ("account", "region", "instance_id", "docker_host", "builder"))
        )
    else:
        print(json.dumps(binding, sort_keys=True))


if __name__ == "__main__":
    main()
