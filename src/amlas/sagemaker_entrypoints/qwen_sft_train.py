"""Fixed train/reload child; OS/provider/source gates precede ML imports."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import socket
import stat
import sys
from pathlib import Path

BASE = Path("/opt/ml/input/data/base_model")
PREPARED = Path("/opt/ml/input/data/prepared")
BINDING = Path("/opt/ml/input/data/bindings/RUN.json")
ADAPTER = Path("/opt/ml/input/data/adapter")
OUTPUT = Path("/opt/ml/model")
FORBIDDEN = frozenset(
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


def _provider_module():
    if not __package__:
        if Path(__file__) != Path("/app/qwen_sft_train.py"):
            raise ValueError("fixed trainer code path refused")
        sys.path.insert(0, "/app")
        import model_worker

        if Path(model_worker.__file__) != Path("/app/model_worker.py"):
            raise ValueError("fixed trainer provider code path refused")
        return model_worker
    from amlas.sagemaker_entrypoints import model_weight_acquisition

    return model_weight_acquisition


_SOCKET_ERRNO_NAMES = {
    errno.EINVAL: "EINVAL",
    errno.EACCES: "EACCES",
    errno.EPERM: "EPERM",
    errno.ECONNREFUSED: "ECONNREFUSED",
    errno.ETIMEDOUT: "ETIMEDOUT",
    errno.ENETUNREACH: "ENETUNREACH",
    errno.EHOSTUNREACH: "EHOSTUNREACH",
    errno.EMFILE: "EMFILE",
    errno.ENOMEM: "ENOMEM",
}
_SOCKET_STAGES = frozenset({"socket_create", "timeout_setup", "connect", "close"})


def _socket_failure(index, stage, error):
    name = "unknown"
    try:
        if isinstance(error, TimeoutError):
            name = "ETIMEDOUT"
        elif isinstance(error, OSError) and type(error.errno) is int:
            name = _SOCKET_ERRNO_NAMES.get(error.errno, "unknown")
    except BaseException:
        # A diagnostic must preserve the original terminal refusal.
        pass
    return {
        "address_index": index,
        "stage": stage,
        "outcome": "timed_out" if stage == "connect" and isinstance(error, TimeoutError) else "failed",
        "errno_name": name,
    }


def _diagnose_metadata_sockets():
    """Observe only the fixed targets after refusal; never read or accept anything."""
    records = []
    for index, address in enumerate(("169.254.169.254", "169.254.170.2")):
        connection = None
        stage = "socket_create"
        try:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            records.append(
                {"address_index": index, "stage": stage, "outcome": "completed", "errno_name": "none"}
            )
            stage = "timeout_setup"
            connection.settimeout(0.5)
            records.append(
                {"address_index": index, "stage": stage, "outcome": "completed", "errno_name": "none"}
            )
            stage = "connect"
            connection.connect((address, 80))
            records.append(
                {"address_index": index, "stage": stage, "outcome": "reachable", "errno_name": "none"}
            )
        except BaseException as error:
            records.append(_socket_failure(index, stage, error))
        finally:
            if connection is not None:
                try:
                    connection.close()
                    records.append(
                        {
                            "address_index": index,
                            "stage": "close",
                            "outcome": "completed",
                            "errno_name": "none",
                        }
                    )
                except BaseException as error:
                    records.append(_socket_failure(index, "close", error))
    return records


def _closed_socket_diagnostics(records):
    if type(records) is not list or not 1 <= len(records) <= 8:
        return None
    names = frozenset(_SOCKET_ERRNO_NAMES.values()) | {"none", "unknown"}
    for record in records:
        if (
            type(record) is not dict
            or set(record) != {"address_index", "stage", "outcome", "errno_name"}
            or type(record["address_index"]) is not int
            or record["address_index"] not in (0, 1)
            or type(record["stage"]) is not str
            or record["stage"] not in _SOCKET_STAGES
            or type(record["outcome"]) is not str
            or record["outcome"] not in {"completed", "failed", "timed_out", "reachable"}
            or type(record["errno_name"]) is not str
            or record["errno_name"] not in names
            or (record["outcome"] == "reachable" and record["stage"] != "connect")
            or (record["outcome"] in {"completed", "reachable"} and record["errno_name"] != "none")
            or (record["outcome"] == "failed" and record["errno_name"] == "none")
            or (
                record["outcome"] == "timed_out"
                and (record["stage"] != "connect" or record["errno_name"] != "ETIMEDOUT")
            )
        ):
            return None
    return records


class FixedPrecoreError(RuntimeError):
    def __init__(self, stage, cause_type, refusal_code="unknown"):
        if stage not in {"provider_inventory", "provider_credentials", "provider_network", "core_import"}:
            raise ValueError("fixed pre-core stage required")
        self.stage = stage
        self.cause_type = cause_type
        if refusal_code not in {
            "unknown",
            "metadata_reachable",
            "provider_file_present",
            "provider_hint",
            "inventory_fd",
            "inventory_shape",
            "provider_observation_unknown",
        }:
            raise ValueError("fixed pre-core refusal code required")
        self.refusal_code = refusal_code
        super().__init__("fixed pre-core operation refused")


def _precore(stage, function):
    try:
        return function()
    except Exception as error:
        codes = {
            "model metadata endpoint isolation refused": "metadata_reachable",
            "model credential provider file presence refused": "provider_file_present",
            "model credential/provider hint presence refused": "provider_hint",
            "model sealed provider inventory refused": "inventory_fd",
            "model inventory closed fields refused": "inventory_shape",
            "model inventory original ordering refused": "inventory_shape",
            "model current provider observation unverified": "provider_observation_unknown",
            "model metadata endpoint observation unverified": "provider_observation_unknown",
        }
        message = error.args[0] if error.args and type(error.args[0]) is str else None
        code = codes.get(message, "unknown")
        wrapped = FixedPrecoreError(stage, type(error).__name__, code)
        kinds = {
            "model current provider observation unverified": "provider_path_stat",
            "model metadata endpoint observation unverified": "metadata_connect",
        }
        kind = kinds.get(message)
        context = error.__context__
        number = context.errno if isinstance(context, OSError) else None
        if kind is not None and type(number) is int and -4095 <= number <= 4095:
            wrapped.observation_kind = kind
            wrapped.native_errno = number
        if (
            stage == "provider_credentials"
            and message == "model metadata endpoint observation unverified"
            and type(number) is int
            and number == errno.EINVAL
        ):
            try:
                wrapped.socket_diagnostics = _diagnose_metadata_sockets()
            except BaseException:
                # Diagnostic implementation failures never mask the guard's refusal.
                pass
        raise wrapped from None


def _provider_observation(stage):
    import errno

    if stage not in ("guard", "before_core_import", "after_core_import"):
        raise ValueError("fixed trainer provider stage refused")
    if any(name in os.environ for name in FORBIDDEN):
        raise ValueError("trainer provider hint presence refused")
    module = _provider_module()
    before = os.fstat(4)
    sealed = _precore("provider_inventory", lambda: module._read_provider_inventory(4))
    observed = _precore("provider_credentials", lambda: module._credential_observation_v2(sealed))
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            raise ValueError("trainer network isolation observation refused")
    except OSError as error:
        if not isinstance(error, TimeoutError) and error.errno not in {
            errno.ECONNREFUSED,
            errno.ETIMEDOUT,
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
            errno.EACCES,
            errno.EPERM,
        }:
            raise ValueError("trainer network observation unverified") from None
    revalidated = module._read_provider_inventory(4)
    after = os.fstat(4)
    identity = [before.st_dev, before.st_ino, before.st_size]
    if identity != [after.st_dev, after.st_ino, after.st_size] or sealed["sha256"] != revalidated["sha256"]:
        raise ValueError("trainer provider descriptor changed during observation")
    result = {
        "inventory_sha256": sealed["sha256"],
        "fd_identity": identity,
        "provider_path_observations": observed["provider_path_observations"],
    }
    log = {
        "schema_version": 1,
        "observation_profile": "qwen_trainer_provider_inventory_v1",
        "stage": stage,
        "inventory_sha256": result["inventory_sha256"],
        "provider_path_observations": result["provider_path_observations"],
        "guest_claim_attested": False,
    }
    print(json.dumps(log, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return result


def _recheck_provider_observation(before, after):
    if before != after:
        raise ValueError("trainer provider observation changed after import")


def _guard():
    if os.getresuid() != (10001,) * 3 or os.getresgid() != (10001,) * 3 or os.getgroups():
        raise ValueError("fixed trainer identity refused")
    if any(name in os.environ for name in FORBIDDEN):
        raise ValueError("trainer provider hint presence refused")
    if not __package__:
        if Path(__file__) != Path("/app/qwen_sft_train.py"):
            raise ValueError("fixed trainer code path refused")
        sys.path.insert(0, "/app")
    from processing_identity import verify_native_identity

    proof = verify_native_identity()
    _provider_observation("guard")
    return proof


def _original_directory(path):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("trainer original directory refused")
    descriptor = os.open("/", flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError:
        os.close(descriptor)
        raise ValueError("trainer original directory refused") from None


def _guard_output():
    import errno
    import fcntl

    _provider_module()._read_provider_inventory(4)
    value = os.fstat(3)
    if not stat.S_ISDIR(value.st_mode) or (value.st_uid, value.st_gid, stat.S_IMODE(value.st_mode)) != (
        10001,
        10001,
        0o700,
    ):
        raise ValueError("trainer fixed output FD refused")
    for name in os.listdir("/proc/self/fd"):
        if name.isdecimal() and int(name) > 4:
            try:
                os.fstat(int(name))
            except OSError as error:
                if error.errno == errno.EBADF:
                    continue
                raise ValueError("trainer descriptor inventory unverified") from None
            raise ValueError("trainer unexpected inherited descriptor refused")
    fcntl.flock(3, fcntl.LOCK_EX | fcntl.LOCK_NB)
    named = _original_directory(OUTPUT)
    try:
        if (value.st_dev, value.st_ino) != (os.fstat(named).st_dev, os.fstat(named).st_ino):
            raise ValueError("trainer output original name changed")
        try:
            fcntl.flock(named, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("trainer output independent lease not denied")
    finally:
        os.close(named)
    if os.listdir(3):
        raise ValueError("trainer output must start empty")


def _read_binding(path, expected_count, expected_sha):
    directory = _original_directory(path.parent)
    descriptor = None
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != expected_count:
            raise ValueError("fixed stage manifest count differs")
        raw = os.pread(descriptor, 262145, 0)
        if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("fixed stage manifest bytes differ")
        # Bound depth before decoding, respecting JSON quoted strings and escapes.
        depth = 0
        quoted = escaped = False
        for char in raw:
            if quoted:
                if escaped:
                    escaped = False
                elif char == 92:
                    escaped = True
                elif char == 34:
                    quoted = False
            elif char == 34:
                quoted = True
            elif char in (91, 123):
                depth += 1
                if depth > 16:
                    raise ValueError("fixed stage manifest depth exceeded")
            elif char in (93, 125):
                depth -= 1

        def pairs(items):
            result = {}
            for key, part in items:
                if key in result:
                    raise ValueError("duplicate stage manifest field")
                result[key] = part
            return result

        def constant(_):
            raise ValueError("stage manifest finite JSON required")

        binding = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        after = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)

        def signature(item):
            return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)

        if (
            signature(before) != signature(after)
            or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
            or os.pread(descriptor, expected_count + 1, 0) != raw
        ):
            raise ValueError("stage manifest retained bytes changed")
        check = _original_directory(path.parent)
        try:
            if (os.fstat(directory).st_dev, os.fstat(directory).st_ino) != (
                os.fstat(check).st_dev,
                os.fstat(check).st_ino,
            ):
                raise ValueError("stage manifest original parent changed")
        finally:
            os.close(check)
        return binding
    except OSError:
        raise ValueError("fixed stage manifest original path refused") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _load_core():
    if not __package__:
        sys.path.insert(0, "/app/src")
        sys.path.insert(0, "/app")
        import qwen_sft_core

        return qwen_sft_core
    from amlas.training import qwen_sft

    return qwen_sft


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "operation",
        "recipe-id",
        "recipe-version",
        "source-commit",
        "manifest-sha256",
        "manifest-byte-count",
    ):
        flag = "--" + name
        if sys.argv.count(flag) != 1:
            raise ValueError("fixed trainer CLI duplicate/missing binding")
        parser.add_argument(flag, required=True)
    args = parser.parse_args()
    if (
        args.recipe_version != "1"
        or not re.fullmatch("[a-f0-9]{64}", args.manifest_sha256)
        or not re.fullmatch("[1-9][0-9]{0,5}", args.manifest_byte_count)
        or int(args.manifest_byte_count) > 262144
        or args.operation not in ("train", "reload", "generate")
        or args.recipe_id
        not in (("humaneval_greedy_v1",) if args.operation == "generate" else ("fit_v1", "full_v1"))
        or not re.fullmatch("[a-f0-9]{40}", args.source_commit)
        or os.environ.get("AMLAS_SOURCE_COMMIT") != args.source_commit
    ):
        raise ValueError("fixed trainer source/operation binding differs")
    proof = _guard()
    provider_before = _provider_observation("before_core_import")
    _guard_output()
    binding = _read_binding(BINDING, int(args.manifest_byte_count), args.manifest_sha256)
    if (
        type(binding) is not dict
        or type(binding.get("schema_version")) is not int
        or binding.get("schema_version") not in (1, 2)
        or (binding["schema_version"] == 2 and args.operation != "reload")
        or binding.get("catalog_id") != "qwen3_5_9b"
        or binding.get("catalog_version") != "1"
        or binding.get("recipe_id") != args.recipe_id
    ):
        raise ValueError("fixed stage manifest identity differs")
    core = _precore("core_import", _load_core)
    # Image/package startup cannot introduce provider overlays after OS measurement.
    if any(name in os.environ for name in FORBIDDEN):
        raise ValueError("trainer provider hints changed after import")
    provider_after = _provider_observation("after_core_import")
    _recheck_provider_observation(provider_before, provider_after)
    if binding["schema_version"] == 2:
        core.validate_reload_binding(binding, source_commit=args.source_commit, recipe_id=args.recipe_id)
    return core.managed_operation(
        args.operation,
        args.recipe_id,
        source_commit=args.source_commit,
        privilege_proof=proof,
        stage_binding=binding,
        stage_manifest_sha256=args.manifest_sha256,
    )


def _format_refusal(error):
    value = {
        "schema_version": 1,
        "status": "fixed_qwen_operation_refused",
        "error_type": type(error).__name__,
    }
    if type(error).__name__ in {"FixedStageError", "FixedPrecoreError"}:
        stage = getattr(error, "stage", None)
        cause = getattr(error, "cause_type", None)
        allowed = {
            "provider_inventory",
            "provider_credentials",
            "provider_network",
            "core_import",
            "input_verify",
            "base_input_verify",
            "ml_import",
            "gpu_probe",
            "tokenizer_load",
            "train_tokenize",
            "validation_tokenize",
            "model_load",
            "adapter_attach",
            "optimizer_train",
            "greedy_probe",
            "checkpoint_save",
            "reload_load",
            "reload_generate",
            "training_receipt",
            "final_readback",
            "final_publication",
        }
        if (
            type(stage) is str
            and stage in allowed
            and type(cause) is str
            and cause.isidentifier()
            and len(cause) <= 64
        ):
            value.update(stage=stage, cause_type=cause)
            if stage == "checkpoint_save" and type(error).__name__ == "FixedStageError":
                substage = getattr(error, "checkpoint_substage", None)
                reason = getattr(error, "checkpoint_reason", None)
                allowed_substages = {
                    "evidence",
                    "output_authority",
                    "output_empty",
                    "framework_save",
                    "model_card",
                    "file_inventory",
                    "file_readback",
                    "tensor_header",
                    "checkpoint_schema",
                    "receipt_commit",
                }
                if (
                    type(substage) is str
                    and substage in allowed_substages
                    and type(reason) is str
                    and reason == substage + "_refused"
                ):
                    value.update(checkpoint_substage=substage, checkpoint_reason=reason)
            if type(error).__name__ == "FixedPrecoreError":
                code = getattr(error, "refusal_code", None)
                if type(code) is str and code in {
                    "unknown",
                    "metadata_reachable",
                    "provider_file_present",
                    "provider_hint",
                    "inventory_fd",
                    "inventory_shape",
                    "provider_observation_unknown",
                }:
                    value["refusal_code"] = code
                kind = getattr(error, "observation_kind", None)
                number = getattr(error, "native_errno", None)
                if (
                    type(kind) is str
                    and kind in {"provider_path_stat", "metadata_connect"}
                    and type(number) is int
                    and -4095 <= number <= 4095
                ):
                    value.update(observation_kind=kind, native_errno=number)
                if (
                    stage == "provider_credentials"
                    and code == "provider_observation_unknown"
                    and kind == "metadata_connect"
                    and type(number) is int
                    and number == errno.EINVAL
                ):
                    diagnostics = _closed_socket_diagnostics(getattr(error, "socket_diagnostics", None))
                    if diagnostics is not None:
                        value["socket_diagnostics"] = diagnostics
    counts = getattr(error, "reason_counts", None)
    if (
        type(error).__name__ == "ConversationCompatibilityError"
        and type(counts) is dict
        and counts
        and not set(counts) - {"developer_role", "historical_reasoning_unsupported"}
        and all(type(count) is int and 0 < count <= 1000000 for count in counts.values())
    ):
        value["status"] = "model_template_compatibility_refused"
        value["reason_counts"] = counts
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(_format_refusal(error), file=sys.stderr)
        raise SystemExit(2) from None
