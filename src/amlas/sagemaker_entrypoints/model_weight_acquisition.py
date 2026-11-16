"""Fixed native model downloader; only stdlib/shared OS guard before workload import."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import socket
import stat
import sys
from pathlib import Path
from typing import Any

_PROVIDER_HINTS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT",
        "AWS_EC2_METADATA_DISABLED",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)
_OUTPUT = Path("/opt/ml/processing/output")
_DIRECTORY_FLAGS = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_PROVIDER_PATHS = (
    "/home/appuser/.aws/credentials",
    "/home/appuser/.aws/config",
    "/root/.aws/credentials",
    "/root/.aws/config",
)


def _read_provider_inventory(descriptor: int) -> dict[str, Any]:
    """Private fixed FD4 consumer; no path, pickle or unsealed proof fallback."""
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or not 0 < before.st_size <= 16384
        or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
        or fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != 15
    ):
        raise RuntimeError("model sealed provider inventory refused")
    raw = os.pread(descriptor, 16385, 0)
    if len(raw) != before.st_size:
        raise RuntimeError("model sealed inventory size differs")
    text = raw.decode("utf-8")
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 8:
                raise RuntimeError("model inventory depth refused")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise RuntimeError("model inventory structure refused")
    if depth or quoted:
        raise RuntimeError("model inventory structure refused")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("model inventory duplicate field")
            result[key] = value
        return result

    def constant(_value):
        raise RuntimeError("model inventory nonfinite value")

    inventory = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    _validate_earlier_inventory(inventory)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
        raise RuntimeError("model sealed inventory identity differs")
    return {"inventory": inventory, "sha256": hashlib.sha256(raw).hexdigest()}


def _validate_earlier_inventory(inventory: Any) -> None:
    # Early checks stay OS/stdlib-only; the source-owned schema rechecks after
    # guarded package startup.
    if (
        type(inventory) is not dict
        or set(inventory) != {"schema_version", "inventory_kind", "observation", "providers"}
        or type(inventory["schema_version"]) is not int
        or inventory["schema_version"] != 1
        or inventory["inventory_kind"] != "raw_model_provider_absence_before_drop_v1"
        or inventory["observation"] != "earlier_original_component_absence"
        or type(inventory["providers"]) is not list
        or len(inventory["providers"]) != 4
    ):
        raise RuntimeError("model inventory closed fields refused")
    for index, record in enumerate(inventory["providers"]):
        if (
            type(record) is not dict
            or set(record) != {"provider_index", "missing_component_index", "ancestors"}
            or type(record["provider_index"]) is not int
            or record["provider_index"] != index
            or type(record["missing_component_index"]) is not int
            or not 1 <= record["missing_component_index"] <= (4 if index < 2 else 3)
            or type(record["ancestors"]) is not list
            or len(record["ancestors"]) != record["missing_component_index"]
        ):
            raise RuntimeError("model inventory original ordering refused")
        for component, node in enumerate(record["ancestors"]):
            if (
                type(node) is not dict
                or set(node) != {"component_index", "device", "inode", "uid", "gid", "mode"}
                or any(type(item) is not int or not 0 <= item < 2**64 for item in node.values())
                or node["component_index"] != component
                or node["inode"] == 0
                or node["mode"] > 0o7777
            ):
                raise RuntimeError("model inventory original ancestor refused")


def _credential_observation_v2(sealed: dict[str, Any]) -> dict[str, Any]:
    if (
        type(sealed) is not dict
        or set(sealed) != {"inventory", "sha256"}
        or type(sealed["sha256"]) is not str
        or not re.fullmatch(r"[a-f0-9]{64}", sealed["sha256"])
    ):
        raise RuntimeError("model earlier inventory binding refused")
    _validate_earlier_inventory(sealed["inventory"])
    if any(name in os.environ for name in _PROVIDER_HINTS):
        raise RuntimeError("model credential/provider hint presence refused")
    states = []
    for index, path in enumerate(_PROVIDER_PATHS):
        try:
            os.lstat(path)
        except FileNotFoundError:
            state = "current_absent"
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EPERM):
                raise RuntimeError("model current provider observation unverified") from None
            state = "denied_after_drop"
        else:
            raise RuntimeError("model credential provider file presence refused")
        states.append({"provider_index": index, "current_state": state})
    for address in ("169.254.169.254", "169.254.170.2"):
        try:
            with socket.create_connection((address, 80), timeout=0.5):
                raise RuntimeError("model metadata endpoint isolation refused")
        except OSError as error:
            if not isinstance(error, TimeoutError) and error.errno not in {
                errno.ECONNREFUSED,
                errno.ETIMEDOUT,
                errno.ENETUNREACH,
                errno.EHOSTUNREACH,
                errno.EACCES,
                errno.EPERM,
            }:
                raise RuntimeError("model metadata endpoint observation unverified") from None
    return {
        "observation_profile": "model_provider_inventory_v2",
        "environment_names": [],
        "earlier_inventory_sha256": sealed["sha256"],
        "earlier_inventory": sealed["inventory"],
        "provider_path_observations": states,
        "metadata_endpoint_reachable": False,
    }


def _open_directory(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError("model output original path refused")
    fd = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_output_contender() -> int:
    """Only the fixed final output is read-opened; original ancestors stay O_PATH."""
    parent = _open_directory(_OUTPUT.parent)
    try:
        return os.open(
            _OUTPUT.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent,
        )
    finally:
        os.close(parent)


def _credential_absence() -> dict[str, Any]:
    if any(name in os.environ for name in _PROVIDER_HINTS):
        raise RuntimeError("model credential/provider hint presence refused")
    for path in (
        Path("/home/appuser/.aws/credentials"),
        Path("/home/appuser/.aws/config"),
        Path("/root/.aws/credentials"),
        Path("/root/.aws/config"),
    ):
        try:
            os.lstat(path)
        except FileNotFoundError:
            continue
        except OSError:
            raise RuntimeError("model credential file absence unverified") from None
        raise RuntimeError("model credential provider file presence refused")
    for address in ("169.254.169.254", "169.254.170.2"):
        try:
            with socket.create_connection((address, 80), timeout=0.5):
                raise RuntimeError("model metadata endpoint isolation refused")
        except OSError as error:
            if not isinstance(error, TimeoutError) and error.errno not in {
                errno.ECONNREFUSED,
                errno.ETIMEDOUT,
                errno.ENETUNREACH,
                errno.EHOSTUNREACH,
                errno.EACCES,
                errno.EPERM,
            }:
                raise RuntimeError("model metadata endpoint observation unverified") from None
            # Only this fixed target/timeout was observed; not universal absence.
            continue
    return {"environment_names": [], "provider_files_present": False, "metadata_endpoint_reachable": False}


def _guard_worker() -> dict[str, Any]:
    if not __package__:
        if Path(__file__).resolve() != Path("/app/acquire_model.py"):
            raise RuntimeError("model fixed worker path refused")
        sys.path.insert(0, "/app")
    from processing_identity import verify_native_identity

    privilege = verify_native_identity()
    inventory = _read_provider_inventory(4)
    output = os.fstat(3)
    if not stat.S_ISDIR(output.st_mode) or (output.st_uid, output.st_gid, stat.S_IMODE(output.st_mode)) != (
        10001,
        10001,
        0o700,
    ):
        raise RuntimeError("model output FD3 authority refused")
    named = _open_directory(_OUTPUT)
    try:
        if (output.st_dev, output.st_ino) != (os.fstat(named).st_dev, os.fstat(named).st_ino):
            raise RuntimeError("model output original named identity refused")
    finally:
        os.close(named)
    fcntl.flock(3, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contender = _open_output_contender()
    try:
        current = os.fstat(contender)
        if (current.st_dev, current.st_ino) != (output.st_dev, output.st_ino):
            raise RuntimeError("model output contender original identity refused")
        try:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(contender, fcntl.LOCK_UN)
            raise RuntimeError("model output lease is not exclusive")
    finally:
        os.close(contender)
    for name in os.listdir("/proc/self/fd"):
        if int(name) <= 4:
            continue
        try:
            os.fstat(int(name))
        except OSError as error:
            if error.errno == errno.EBADF:
                continue
            raise RuntimeError("model inherited descriptor observation unverified") from None
        raise RuntimeError("model unexpected inherited descriptor refused")
    return {
        "privilege_profile": "sagemaker_native_ceiling_v1",
        "privilege_proof": privilege,
        "output_directory_proof": {
            "fd": 3,
            "uid": output.st_uid,
            "gid": output.st_gid,
            "mode": stat.S_IMODE(output.st_mode),
            "device": output.st_dev,
            "inode": output.st_ino,
            "lease_active": True,
        },
        "credential_observation": _credential_observation_v2(inventory),
        "guest_claim_attested": False,
    }


def _load_core() -> Any:
    if not __package__:
        # This root-owned image path is fixed, and is added only after actual guards.
        sys.path.insert(0, "/app/src")
    from amlas.data_eng import model_weight_acquisition

    return model_weight_acquisition


def main() -> None:
    try:
        keys = (
            "catalog_id",
            "catalog_version",
            "job_name",
            "runtime_session_id",
            "deployment_id",
            "deployment_prefix",
            "image_uri",
            "source_commit",
        )
        parser = argparse.ArgumentParser()
        for key in keys:
            flag = "--" + key.replace("_", "-")
            if sys.argv.count(flag) != 1:
                raise RuntimeError("model fixed CLI duplicate/missing binding refused")
            parser.add_argument(flag, required=True)
        arguments = vars(parser.parse_args())
        if (
            not re.fullmatch(r"[a-f0-9]{40}", os.environ.get("AMLAS_SOURCE_COMMIT", ""))
            or arguments["source_commit"] != os.environ["AMLAS_SOURCE_COMMIT"]
        ):
            raise RuntimeError("model compiled source binding refused")
        evidence = _guard_worker()
        core = _load_core()
        # Package imports must not introduce an unobserved developer/provider overlay.
        _credential_observation_v2(_read_provider_inventory(4))
        catalog_id, catalog_version = arguments.pop("catalog_id"), arguments.pop("catalog_version")
        core.acquire_model_into_fd(
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            output_fd=3,
            output_directory=_OUTPUT,
            bindings=arguments,
            evidence=evidence,
        )
    except Exception as error:
        print("fixed model acquisition refused (" + type(error).__name__ + ")", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
