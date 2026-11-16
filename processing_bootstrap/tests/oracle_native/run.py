"""Account-only native execution. Root verifies account/EC2 identity before calling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
BASE_CONTROLS = {
    "source_read_0", "source_read_1", "source_read_2", "runtime_read", "getdents64",
    "unknown_sibling_metadata_exposed", "prlimit_self_read", "stdout",
    "ungranted_canary_read", "home_read", "input_read", "unknown_sibling_content",
    "proc_self_read", "proc_peer_read", "canary_write_open", "home_create",
    "low_fd_closed", "low_fd_closed_read", "high_fd_closed", "high_fd_closed_read",
    "tcp_socket", "udp_socket", "unix_socket", "tcp_connect",
    "cross_domain_abstract_connect", "cross_domain_signal0", "prlimit_peer_read",
    "prlimit_self_same_value_set", "peer_memory_read", "peer_memory_same_value_write",
    "fork_process", "unknown_syscall",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_local_image(selected, expected, source, inspection):
    """A local config ID is not a registry manifest digest."""
    if selected != expected or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
        raise ValueError("image_local_config_pin")
    if not isinstance(inspection, list) or len(inspection) != 1:
        raise ValueError("image_inspection")
    item = inspection[0]
    config = item.get("Config", {})
    stamps = [value for value in config.get("Env", [])
              if isinstance(value, str) and value.startswith("AMLAS_CODING_PAIR_SOURCE_COMMIT=")]
    if (item.get("Id") != expected or item.get("Os") != "linux"
            or item.get("Architecture") != "amd64" or config.get("User") != "10001:10001"
            or config.get("Labels", {}).get("org.opencontainers.image.revision") != source
            or stamps != ["AMLAS_CODING_PAIR_SOURCE_COMMIT=" + source]):
        raise ValueError("image_inspected_metadata_pin")
    return item


def local_base_tag(config_id, helper_sha):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", config_id) or not re.fullmatch(r"[0-9a-f]{64}", helper_sha):
        raise ValueError("local_base_tag_pin")
    return "amlas-oracle-native-base-" + helper_sha[:12] + ":" + config_id[7:]


def validate_local_builder(proof, selected, context, builder_text, daemon_id):
    if proof.get("execution_environment") != "owned-aws-ec2" or proof.get("same_account_targets") is not True:
        raise ValueError("account_execution_proof")
    fields = {}
    for line in builder_text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"Name", "Driver", "Endpoint", "Status"}:
            fields.setdefault(key.strip(), []).append(value.strip())
    if (selected != "default" or not isinstance(context, list) or len(context) != 1
            or context[0].get("Name") != "default"
            or context[0].get("Endpoints", {}).get("docker", {}).get("Host") not in
               {"unix:///var/run/docker.sock", "unix:///run/docker.sock"}
            or fields.get("Driver") != ["docker"]
            or fields.get("Endpoint") != ["default"] or fields.get("Status") != ["running"]
            or len(fields.get("Name", [])) != 2 or fields["Name"][0] != "default"
            or not daemon_id or len(daemon_id) > 256 or any(value.isspace() for value in daemon_id)):
        raise ValueError("local_single_builder")


def build_with_local_base(proof, config_id, source, helper_sha, dockerfile, prepared, output, fixture_tag):
    """Local tag observations are non-atomic; trusted local writers are required."""
    if proof.get("execution_environment") != "owned-aws-ec2" or proof.get("same_account_targets") is not True:
        raise ValueError("account_execution_proof")
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "BUILDX_BUILDER"):
        if os.environ.get(name):
            raise ValueError("docker_transport_override")
    selected, _ = command(["docker", "context", "show"], output, "builder-context", 10)
    context, _ = command(["docker", "context", "inspect", "default"], output, "builder-context-inspect", 10)
    builder, _ = command(["docker", "buildx", "inspect", "default"], output, "builder-inspect", 10)
    daemon, _ = command(["docker", "info", "--format", "{{.ID}}"], output, "builder-daemon-before", 10)
    if any(item.returncode != 0 for item in (selected, context, builder, daemon)):
        raise ValueError("builder_inspection_failure")
    daemon_id = daemon.stdout.decode().strip()
    validate_local_builder(proof, selected.stdout.decode().strip(), json.loads(context.stdout),
                           builder.stdout.decode(), daemon_id)
    base_tag = local_base_tag(config_id, helper_sha)

    def inspect_tag(stem):
        item, _ = command(["docker", "image", "inspect", base_tag], output, stem, 15)
        if item.returncode != 0:
            raise ValueError("base_tag_inspection")
        metadata = validate_local_image(config_id, config_id, source, json.loads(item.stdout))
        if base_tag not in metadata.get("RepoTags", []):
            raise ValueError("base_tag_name")
        return metadata

    existing, _ = command(["docker", "image", "inspect", base_tag], output, "base-tag-existing", 15)
    if existing.returncode == 0:
        metadata = validate_local_image(config_id, config_id, source, json.loads(existing.stdout))
        if base_tag not in metadata.get("RepoTags", []):
            raise ValueError("base_tag_name")
    else:
        message = existing.stdout.decode().strip()
        if existing.returncode != 1 or not re.fullmatch(
            rf"(?:\[\]\s*)?Error(?: response from daemon)?: No such (?:object|image): {re.escape(base_tag)}", message
        ):
            raise ValueError("base_tag_existing_unknown")
        bound, _ = command(["docker", "image", "tag", config_id, base_tag], output, "base-tag-bind", 15)
        if bound.returncode != 0:
            raise ValueError("base_tag_bind")
    before = inspect_tag("base-tag-before-build")
    prefix = "FROM " + config_id + "\n"
    if not dockerfile.startswith(prefix):
        raise ValueError("fixture_from_pin")
    (prepared / "Dockerfile.fixture").write_text("FROM " + base_tag + "\n" + dockerfile[len(prefix):])
    result, elapsed = command([
        "docker", "buildx", "build", "--builder", "default", "--load",
        "--network", "none", "--pull=false", "-f", str(prepared / "Dockerfile.fixture"),
        "-t", fixture_tag, str(prepared),
    ], output, "fixture-image-build", 180)
    after = inspect_tag("base-tag-after-build")
    daemon_after, _ = command(["docker", "info", "--format", "{{.ID}}"], output, "builder-daemon-after", 10)
    if daemon_after.returncode != 0 or daemon_after.stdout.decode().strip() != daemon_id:
        raise ValueError("builder_daemon_drift")
    if result.returncode != 0:
        raise ValueError("fixture_image_build")
    return {
        "base_tag": base_tag, "base_config_id": config_id, "base_source_commit": source,
        "before_metadata_sha256": sha(json.dumps(before, sort_keys=True).encode()),
        "after_metadata_sha256": sha(json.dumps(after, sort_keys=True).encode()),
        "builder": "default", "driver": "docker", "daemon_id": daemon_id, "seconds": elapsed,
        "atomic_tag_binding": False, "trusted_local_writers_assumed": True,
        "tag_race_limit": "pre/post observations cannot exclude temporary substitution",
    }


def validate_container(original_id, image_id, item):
    host = item.get("HostConfig", {})
    required = {"NetworkMode", "Privileged", "CapDrop", "CapAdd", "SecurityOpt",
                "PidMode", "IpcMode", "Binds", "Devices", "GroupAdd"}
    if (not required.issubset(host) or item.get("Id") != original_id or item.get("Image") != image_id
            or item.get("Config", {}).get("User") != "10001:10001"
            or host.get("NetworkMode") != "none" or host.get("Privileged") is not False
            or host.get("CapDrop") != ["ALL"] or host.get("CapAdd") not in (None, [])
            or set(host.get("SecurityOpt") or []) != {"no-new-privileges:true", "seccomp=unconfined"}
            or host.get("PidMode") != ""
            or host.get("IpcMode") not in ("", "private")
            or host.get("Binds") not in (None, []) or host.get("Devices") not in (None, [])
            or host.get("GroupAdd") not in (None, []) or item.get("Mounts") != []):
        raise ValueError("container_identity_or_scope")
    return item


def inspect_original(original_id, image_id, output, stem):
    result, _ = command(["docker", "container", "inspect", original_id], output, stem, 10)
    if result.returncode != 0:
        raise ValueError("cleanup_inspect_unconfirmed")
    inspection = json.loads(result.stdout)
    if not isinstance(inspection, list) or len(inspection) != 1:
        raise ValueError("cleanup_inspection_shape")
    return validate_container(original_id, image_id, inspection[0])


def cleanup_container(original_id, image_id, output, stem):
    """Inspect the original immutable CID; confirm stopped before removal."""
    if not re.fullmatch(r"[0-9a-f]{64}", original_id or ""):
        raise ValueError("cleanup_original_identity")
    before = inspect_original(original_id, image_id, output, stem + "-before")
    after = before
    if before.get("State", {}).get("Running") is True:
        killed, _ = command(["docker", "container", "kill", original_id], output, stem + "-kill", 10)
        if killed.returncode != 0:
            raise ValueError("cleanup_kill_unconfirmed")
        after = inspect_original(original_id, image_id, output, stem + "-after-kill")
    state = after.get("State", {})
    if state.get("Running") is not False or state.get("Pid") != 0 or state.get("Status") not in {"created", "exited", "dead"}:
        raise ValueError("cleanup_stop_unconfirmed")
    removed, _ = command(["docker", "container", "rm", original_id], output, stem + "-remove", 10)
    if removed.returncode != 0 or removed.stdout.decode().strip() != original_id:
        raise ValueError("cleanup_remove_unconfirmed")
    absent, _ = command(["docker", "container", "inspect", original_id], output, stem + "-absence", 10)
    message = absent.stdout.decode().strip()
    if absent.returncode != 1 or not re.fullmatch(
        rf"(?:\[\]\s*)?Error(?: response from daemon)?: No such (?:object|container): {original_id}", message
    ):
        raise ValueError("cleanup_absence_unconfirmed")
    return {"original_container_id": original_id, "image_id": image_id,
            "before_state": before["State"], "after_state": state,
            "stopped_confirmed": True, "removed_confirmed": True}


def audit(mode, returncode, text):
    expected = set(BASE_CONTROLS)
    if mode != "complete":
        expected |= {"udp_send_unfiltered_without_seccomp", "original_child_reaped"}
    if mode == "baseline":
        expected.add("peer_memory_value")
    rows = []
    for line in text.splitlines():
        if line.startswith("CONTROL "):
            match = re.fullmatch(r"CONTROL (\w+) (\w+) PASS rc=(-?\d+) errno=(-?\d+)", line)
            if not match or match.group(1) != mode:
                raise ValueError("control_failure")
            rows.append(match.group(2))
    if len(rows) != len(set(rows)) or set(rows) != expected:
        raise ValueError("control_inventory")
    finals = [line for line in text.splitlines() if line.startswith("FIXTURE ")]
    if returncode != 0 or len(finals) != 1 or not re.fullmatch(
        rf"FIXTURE {mode} PASS original_pid=\d+ wait_status=0 reaped=true", finals[0]
    ):
        raise ValueError("fixture_completion")
    return {"mode": mode, "controls": sorted(rows), "control_count": len(rows)}


def command(args, output, stem, timeout, **kwargs):
    started = time.monotonic()
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False, **kwargs)
    except subprocess.TimeoutExpired as error:
        captured = error.stdout or b""
        (output / (stem + ".log")).write_bytes(captured[:524288])
        (output / (stem + "-timeout.json")).write_text(json.dumps({
            "timeout_seconds": timeout, "captured_bytes": len(captured),
            "truncated": len(captured) > 524288,
        }) + "\n")
        raise
    if len(result.stdout) > 524288:
        raise ValueError("output_bound")
    (output / (stem + ".log")).write_bytes(result.stdout)
    return result, time.monotonic() - started


def run(prepared, production_image, output, account_proof):
    """One attempt per step. Uses no AWS API or credential value."""
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("output_exists")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest.get("schema") != "oracle_native_fixture_inputs_v2":
        raise ValueError("input_schema")
    if sha(Path(__file__).read_bytes()) != manifest["runner_sha256"] or sha((HERE / "prepare.py").read_bytes()) != manifest["preparer_sha256"]:
        raise ValueError("harness_pin")
    proof = json.loads(Path(account_proof).read_text())
    if proof.get("execution_environment") != "owned-aws-ec2" or proof.get("same_account_targets") is not True:
        raise ValueError("account_execution_proof")
    if production_image != manifest["qualified_image_config_id"]:
        raise ValueError("image_pin")
    composite = (prepared / "processing_bootstrap/src/oracle_boundary.rs").read_bytes()
    prefix = composite[: manifest["policy_bytes"]]
    if sha(composite) != manifest["composite_sha256"] or sha(prefix) != manifest["source_sha256"]["processing_bootstrap/src/oracle_boundary.rs"]:
        raise ValueError("composite_pin")
    for path, expected in manifest["source_sha256"].items():
        if path.startswith("src/") and sha((prepared / path).read_bytes()) != expected:
            raise ValueError("include_source_pin")
    output.mkdir(parents=True)
    inspected, _ = command(["docker", "image", "inspect", production_image],
                           output, "production-image-inspect", 15)
    if inspected.returncode != 0:
        raise ValueError("image_inspect_failure")
    production_metadata = validate_local_image(
        production_image, manifest["qualified_image_config_id"],
        manifest["qualified_source_commit"], json.loads(inspected.stdout),
    )
    # Use the immutable inspected local ID for FROM; no registry-digest lookup.
    production_image = production_metadata["Id"]
    (prepared / "compiler-tmp").mkdir()
    records = []
    for mode in manifest["policy_modes"]:
        binary = prepared / ("oracle-" + mode)
        if binary.exists():
            raise ValueError("binary_exists")
        result, elapsed = command([
            "docker", "run", "--rm", "--network", "none", "--pull", "never",
            "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{prepared}:/ctx",
            "--env", "TMPDIR=/ctx/compiler-tmp",
            manifest["compiler_image"], "rustc", "--edition=2021", "--cfg", "test",
            "--cfg", "oracle_fixture_" + mode, "-C", "panic=abort", "-O",
            "--target", "x86_64-unknown-linux-musl",
            "/ctx/processing_bootstrap/src/oracle_boundary.rs", "-o", "/ctx/oracle-" + mode,
        ], output, "compile-" + mode, 180)
        if result.returncode != 0:
            raise ValueError("compile_failure")
        records.append({"stage": "compile", "mode": mode, "seconds": elapsed,
                        "binary_sha256": sha(binary.read_bytes())})
    (prepared / "canary.txt").write_bytes(b"oracle-native-canary-v1\n")
    dockerfile = f"""FROM {production_image}
USER root
RUN mkdir -p /oracle-fixture /home/appuser /opt/ml/input && chown 10001:10001 /oracle-fixture /home/appuser && chmod 0755 /oracle-fixture /home/appuser /opt/ml/input
COPY --chmod=0555 oracle-baseline oracle-complete oracle-landlock /oracle-fixture/
COPY --chown=10001:10001 --chmod=0644 canary.txt /oracle-fixture/canary.txt
COPY --chown=10001:10001 --chmod=0644 canary.txt /home/appuser/oracle-native-canary.txt
COPY --chmod=0444 canary.txt /opt/ml/input/oracle-native-canary.txt
COPY --chmod=0444 canary.txt /app/oracle/oracle_native_unknown.txt
USER 10001:10001
"""
    tag = "amlas-oracle-native-" + manifest["helper_sha256"][:12]
    build_transport = build_with_local_base(proof, production_image, manifest["qualified_source_commit"],
                                           manifest["helper_sha256"], dockerfile, prepared, output, tag)
    result, _ = command(["docker", "image", "inspect", "--format", "{{.Id}}", tag],
                        output, "fixture-image-id", 15)
    fixture_id = result.stdout.decode().strip()
    if result.returncode != 0 or not re.fullmatch(r"sha256:[0-9a-f]{64}", fixture_id):
        raise ValueError("fixture_image_identity")
    # Verify production launcher and installed three sources in the derived image.
    paths = ["/usr/local/bin/amlas-oracle-init", "/app/oracle/coding_oracle_child.py",
             "/app/oracle/coding_oracle.py", "/app/oracle/coding_oracle_catalog.py"]
    result, _ = command(["docker", "run", "--rm", "--network", "none", "--pull", "never",
                        "--read-only", "--cap-drop", "ALL", "--entrypoint", "/usr/bin/sha256sum",
                        fixture_id, *paths], output, "installed-source-pins", 15)
    expected = [manifest["qualified_launcher_sha256"]] + [
        manifest["source_sha256"][path] for path in (
            "src/amlas/evaluation/coding_oracle_child.py", "src/amlas/evaluation/coding_oracle.py",
            "src/amlas/evaluation/coding_oracle_catalog.py")
    ]
    if result.returncode != 0 or result.stdout.decode().splitlines() != [
        digest + "  " + path for digest, path in zip(expected, paths)
    ]:
        raise ValueError("installed_source_pin")
    # Native negatives start only after the entire identical baseline passes.
    for mode in manifest["policy_modes"]:
        container_id_file = output / ("native-" + mode + ".cid")
        created, _ = command([
            "docker", "create", "--cidfile", str(container_id_file),
            "--network", "none", "--pull", "never", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true", "--security-opt", "seccomp=unconfined",
            "-i", "--entrypoint", "/oracle-fixture/oracle-" + mode, fixture_id,
        ], output, "native-" + mode + "-create", 15)
        if created.returncode != 0 or not container_id_file.exists():
            raise ValueError("container_create_or_identity")
        original_id = container_id_file.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{64}", original_id) or created.stdout.decode().strip() != original_id:
            raise ValueError("container_original_identity")
        initial = inspect_original(original_id, fixture_id, output, "native-" + mode + "-created")
        initial_state = initial.get("State", {})
        if initial_state.get("Running") is not False or initial_state.get("Pid") != 0 or initial_state.get("Status") != "created":
            raise ValueError("container_not_unstarted")
        try:
            result, elapsed = command([
                "docker", "start", "--attach", "--interactive", original_id,
            ], output, "native-" + mode, 25, input=b"")
        except subprocess.TimeoutExpired:
            cleanup = cleanup_container(original_id, fixture_id, output, "native-" + mode + "-timeout-cleanup")
            (output / ("native-" + mode + "-cleanup.json")).write_text(json.dumps(cleanup, indent=2) + "\n")
            raise
        cleanup = cleanup_container(original_id, fixture_id, output, "native-" + mode + "-cleanup")
        (output / ("native-" + mode + "-cleanup.json")).write_text(json.dumps(cleanup, indent=2) + "\n")
        stopped = cleanup["before_state"]
        if stopped.get("Running") is not False or stopped.get("Pid") != 0 or stopped.get("Status") != "exited" or stopped.get("ExitCode") != 0:
            raise ValueError("container_completion_unconfirmed")
        record = audit(mode, result.returncode, result.stdout.decode("utf-8", errors="strict"))
        record.update({"seconds": elapsed, "fixture_image_id": fixture_id,
                       "binary_sha256": sha((prepared / ("oracle-" + mode)).read_bytes()),
                       "cleanup": cleanup})
        records.append(record)
        (output / "partial.json").write_text(json.dumps(records, indent=2) + "\n")
    receipt = {"schema": "oracle_native_fixture_run_v1", "inputs": manifest,
               "account_proof_sha256": sha(Path(account_proof).read_bytes()),
               "production_image_config_id": production_image,
               "production_inspection_sha256": sha(inspected.stdout),
               "build_transport": build_transport,
               "fixture_image_id": fixture_id, "records": records,
               "native_admission": False, "official_case_or_model_quality_acceptance": False,
               "provider_availability": "unmeasured",
               "limits": ["test-only preexec fixture; no Python exec",
                          "no live provider request; owned loopback and abstract UNIX only",
                          "Landlock-only scope attribution is separate from complete policy",
                          "ptrace, pidfd, alternate-architecture/x32 and socketpair not measured"]}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "CONTROLLED_FIXTURE_PASS", "fixture_image_id": fixture_id,
                      "native_admission": False, "receipt_sha256": sha((output / "receipt.json").read_bytes())}))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--production-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--account-proof", type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared, args.production_image, args.output, args.account_proof)
