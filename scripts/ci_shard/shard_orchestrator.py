"""Run the pytest phase of the full CI as N concurrent containers on one host, then merge their outputs.

Host-side tool: standard library only, Python 3.9 compatible. Every container uses the single pipeline's image and
flags (network none, read-only root, same seeds and mounts); only the per-shard names, cache, output and the
read-only ``/ci-out`` differ. The merge result is the python phase's pytest gate.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess  # nosec B404 - fixed docker argv built from the pinned run configuration, no shell
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import full_ci_shard_merge as shard_merge
import full_ci_shard_plan as shard_plan


@dataclass(frozen=True)
class ShardRunConfig:
    docker: str
    run_id: str
    image: str
    out: Path
    cache: Path
    qa: Path
    base: Path
    workplace: Path
    cpu: Path
    certifi: Path
    case_path: Path
    assets: Path
    candidate: str
    tree: str
    n: int
    cpus: str
    memory: str
    cpusets: tuple[str, ...] | None
    extra_ro_mounts: tuple[str, ...]
    script: str
    split_fraction: float = shard_plan.DEFAULT_SPLIT_FRACTION
    isolate_seconds: float = shard_plan.DEFAULT_ISOLATE_SECONDS


def shard_name(index):
    return f"s{index:02d}"


def docker_command(c, index, plan_sha256):
    w = c.workplace
    name = shard_name(index)
    cmd = [
        c.docker,
        "run",
        "--rm",
        "--name",
        f"{c.run_id}-python-{name}",
        "--cidfile",
        str(c.out / f"phase-python-{name}.cid"),
        "--network",
        "none",
        "--read-only",
        "--memory",
        c.memory,
        "--cpus",
        c.cpus,
    ]
    if c.cpusets:
        cmd += ["--cpuset-cpus", c.cpusets[index]]
    mounts = [
        f"{c.qa}:/work:ro",
        f"{w / 'repos'}:{w / 'repos'}:ro",
        f"{c.base}:{c.base}:ro",
    ]
    mounts += [f"{path}:{path}:ro" for path in c.extra_ro_mounts]
    mounts += [
        f"{w / 'tmp/ci-seed'}:/ci-seed:ro",
        f"{c.cpu}:/ci-cpu:ro",
        f"{c.certifi}:/ci-certifi:ro",
        f"{c.case_path}:/ci-source/humaneval-original.parquet:ro",
        f"{c.assets}:/ci-assets:ro",
        f"{c.out}:/ci-out:ro",
        f"{c.out / 'shards' / name}:/ci-shard",
        f"{c.cache / name}:/ci-run",
    ]
    for mount in mounts:
        cmd += ["-v", mount]
    cmd += ["-w", "/work"]
    env = [
        "AMLAS_CI_HEAD=" + c.candidate,
        "AMLAS_CI_TREE=" + c.tree,
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        "AWS_EC2_METADATA_DISABLED=true",
        "AMLAS_CI_SHARD_PLAN=/ci-out/shard-plan.json",
        "AMLAS_CI_SHARD_PLAN_SHA256=" + plan_sha256,
        f"AMLAS_CI_SHARD_INDEX={index}",
        "AMLAS_CI_SHARD_OUT=/ci-shard",
    ]
    for item in env:
        cmd += ["-e", item]
    cmd += [c.image, "bash", f"/ci-assets/{c.script}"]
    return cmd


def parse_cpuset(text):
    cpus = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def format_cpuset(cpus):
    ordered = sorted(cpus)
    ranges = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append((start, prev))
        start = prev = cpu
    ranges.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def assign_cpusets(n, lscpu_parsable):
    """One disjoint set of whole cores (all hyperthreads) per shard, from ``lscpu -p=CPU,CORE,SOCKET`` text."""
    cores = {}
    for line in lscpu_parsable.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cpu, core, socket = (int(x) for x in line.split(",")[:3])
        cores.setdefault((socket, core), []).append(cpu)
    ordered = [cores[key] for key in sorted(cores)]
    per_shard = len(ordered) // n
    if per_shard < 1:
        raise ValueError(f"{n} shards need at least {n} cores, the host has {len(ordered)}")
    sets = []
    for index in range(n):
        chosen = [cpu for core in ordered[index * per_shard : (index + 1) * per_shard] for cpu in core]
        sets.append(format_cpuset(chosen))
    return tuple(sets)


def _container_gone(c, cid):
    state = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit  # nosec B603 - fixed docker argv, no shell
        [c.docker, "ps", "-aq", "--no-trunc", "--filter", "id=" + cid],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if state.returncode == 0 and not state.stdout.strip():
        return True
    status = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit  # nosec B603 - fixed docker argv, no shell
        [c.docker, "inspect", "--format", "{{.State.Running}}", cid],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return status.returncode == 0 and status.stdout.strip() == "false"


def _abort(c, procs):
    """Best-effort stop of every started shard container and its docker client."""
    for index, proc in enumerate(procs):
        name = f"{c.run_id}-python-{shard_name(index)}"
        try:
            subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit  # nosec B603 - fixed docker argv, no shell
                [c.docker, "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if proc.poll() is None:
            proc.kill()
    for proc in procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def run_shards(c, plan_sha256, poll_seconds=5.0, timeout_seconds=14400.0, gone_timeout_seconds=900.0):
    """Start every shard container at once and wait for all of them.

    A shard still running at the deadline is killed and reported as 124. Output and cache directories must be new:
    a leftover directory of an earlier run is refused. Each container must be stopped or removed before return.
    """
    log_files = contextlib.ExitStack()
    procs = []
    for index in range(c.n):
        (c.out / "shards" / shard_name(index)).mkdir(parents=True)
        (c.cache / shard_name(index)).mkdir(parents=True)
    deadline = time.monotonic() + timeout_seconds
    codes = []
    try:
        for index in range(c.n):
            log = log_files.enter_context(
                open(c.out / "shards" / shard_name(index) / "docker.native.log", "xb")
            )
            procs.append(
                subprocess.Popen(  # nosemgrep: dangerous-subprocess-use-audit  # nosec B603 - fixed docker argv, no shell
                    docker_command(c, index, plan_sha256), stdout=log, stderr=subprocess.STDOUT
                )
            )
        for proc in procs:
            try:
                codes.append(proc.wait(timeout=max(deadline - time.monotonic(), 0.0)))
            except subprocess.TimeoutExpired:
                codes.append(124)
        if 124 in codes:
            _abort(c, procs)
    except BaseException:
        _abort(c, procs)
        raise
    finally:
        log_files.close()
    for index in range(c.n):
        cidfile = c.out / f"phase-python-{shard_name(index)}.cid"
        if not cidfile.is_file():
            continue
        cid = cidfile.read_text().strip()
        if not re.fullmatch("[a-f0-9]{64}", cid):
            raise RuntimeError("owned shard container handle refused")
        give_up = time.monotonic() + gone_timeout_seconds
        while not _container_gone(c, cid):
            if time.monotonic() > give_up:
                raise RuntimeError(
                    f"shard {index} container did not stop within {gone_timeout_seconds:.0f} s"
                )
            time.sleep(poll_seconds)
    return codes


def _refuse_phase(c, message):
    (c.out / "pytest.log").write_text(message + "\n")
    (c.out / "pytest.exit").write_text("1\n")
    return 1


def run_python_shards(c, durations_path, poll_seconds=5.0, timeout_seconds=14400.0):
    """Plan, run the shard containers, merge. Returns the pytest gate's exit code (0 only if everything held)."""
    out = c.out
    try:
        ids = shard_plan.parse_collection_log((out / "collection.log").read_text(encoding="utf-8"))
        shard_plan.verify_collection(ids, (out / "collection.metadata.json").read_bytes())
        shard_plan.junit_key_index(ids)
        raw = Path(durations_path).read_bytes()
        plan = shard_plan.build_plan(
            ids,
            json.loads(raw),
            c.n,
            durations_sha256=shard_plan.sha256_bytes(raw),
            split_fraction=c.split_fraction,
            isolate_seconds=c.isolate_seconds,
            run_nonce=uuid.uuid4().hex,
        )
    except (shard_plan.PlanError, OSError, ValueError) as exc:
        return _refuse_phase(c, f"plan refused: {exc}")
    plan_raw = shard_plan.plan_bytes(plan)
    with open(out / "shard-plan.json", "xb") as handle:
        handle.write(plan_raw)
    plan_sha = shard_plan.sha256_bytes(plan_raw)
    started = time.monotonic()
    codes = run_shards(c, plan_sha, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
    wall = time.monotonic() - started
    try:
        result = shard_merge.merge(out, wall_seconds=wall)
    except shard_merge.MergeError as exc:
        return _refuse_phase(c, f"merge refused: {exc}\nshard container exit codes: {codes}")
    return 0 if result["exit_code"] == 0 and all(code == 0 for code in codes) else 1


def _run_from_native(
    *,
    run_id,
    image,
    out,
    cache,
    qa,
    base,
    workplace,
    cpu,
    certifi,
    case_path,
    assets,
    candidate,
    tree,
    docker="docker",
):
    """Entry point used by the sharded full-ci.native.py; the shard parameters are a pinned asset."""
    assets = Path(assets)
    params = json.loads((assets / "shard-params.json").read_text())
    n = int(params["shards"])
    cpusets = None
    if params.get("cpusets") == "auto":
        lscpu = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True)  # nosec B603 B607 - fixed argv, host tool
        cpusets = assign_cpusets(n, lscpu)
    c = ShardRunConfig(
        docker=docker,
        run_id=run_id,
        image=image,
        out=Path(out),
        cache=Path(cache),
        qa=Path(qa),
        base=Path(base),
        workplace=Path(workplace),
        cpu=Path(cpu),
        certifi=Path(certifi),
        case_path=Path(case_path),
        assets=assets,
        candidate=candidate,
        tree=tree,
        n=n,
        cpus=str(params["cpus"]),
        memory=str(params["memory"]),
        cpusets=cpusets,
        extra_ro_mounts=tuple(params.get("extra_ro_mounts", ())),
        script=params.get("script", "full-ci-python-shard.sh"),
        split_fraction=float(params.get("split_fraction", shard_plan.DEFAULT_SPLIT_FRACTION)),
        isolate_seconds=float(params.get("isolate_seconds", shard_plan.DEFAULT_ISOLATE_SECONDS)),
    )
    return run_python_shards(
        c, assets / params["durations"], timeout_seconds=float(params.get("timeout_seconds", 14400))
    )


def run_from_native(**kwargs):
    """Entry point used by the sharded full-ci.native.py. Any failure is the pytest gate's exit 1, never an exception."""
    try:
        return _run_from_native(**kwargs)
    except Exception as exc:  # the phase must fail closed and leave its reason in the gate log
        out = Path(kwargs["out"])
        (out / "pytest.log").write_text(f"sharded pytest phase refused: {exc.__class__.__name__}: {exc}\n")
        (out / "pytest.exit").write_text("1\n")
        return 1


if __name__ == "__main__":
    print("shard_orchestrator is imported by the sharded full-ci.native.py", file=sys.stderr)
    sys.exit(2)
