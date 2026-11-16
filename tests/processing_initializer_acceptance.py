"""Linux Docker acceptance oracle for the static initializer; never live AWS."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

BASE = "public.ecr.aws/docker/library/python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93"
WORKER = """import json,os,sys,time
from pathlib import Path
u=10001
assert os.getresuid()==(u,u,u) and os.getresgid()==(u,u,u)
assert os.getgroups()==[]
s=dict(x.split(':',1) for x in Path('/proc/self/status').read_text().splitlines() if ':' in x)
assert s['NoNewPrivs'].strip()=='1'
assert all(int(s[k],16)==0 for k in ('CapEff','CapPrm','CapInh','CapAmb','CapBnd'))
try: os.setuid(0)
except PermissionError: pass
else: raise RuntimeError('worker regained root')
assert os.environ['AMLAS_WORKSPACE_DIR']=='/opt/ml/processing'
if '--filename' in sys.argv and sys.argv[sys.argv.index('--filename')+1]=='lease.csv':
 print('LEASE-READY',flush=True); time.sleep(5)
Path('/opt/ml/processing/output/CANARY.txt').write_bytes(b'initializer-fixture-v1\\n')
print(json.dumps({'worker_uid':u,'groups':[],'no_new_privs':1,'capabilities':0,'regain_root':False}),flush=True)
"""
ARGS = [
    "--repo-id",
    "fixture/open",
    "--revision",
    "b" * 40,
    "--filename",
    "data.csv",
    "--job-name",
    "fixture",
    "--runtime-session-id",
    "session-a",
    "--deployment-id",
    "fixture",
    "--deployment-prefix",
    "fixture",
    "--image-uri",
    "fixture",
    "--source-commit",
    "c" * 40,
]


def command(args, **kwargs):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)


def run(binary: Path, scratch: Path, image: str, only_case: str | None = None) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    context = scratch / "context"
    context.mkdir(exist_ok=True)
    (context / "amlas-processing-init").write_bytes(binary.read_bytes())
    (context / "worker.py").write_text(WORKER)
    needs_kernel_fixture = only_case is None or only_case in ("empty-bounding-set", "present-undroppable")
    if needs_kernel_fixture:
        fixture_source = Path(__file__).with_name("processing_capability_fixture.rs")
        (context / "capability-fixture.rs").write_bytes(fixture_source.read_bytes())
        compiled = command(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--pull",
                "never",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{context}:/fixture",
                "public.ecr.aws/docker/library/rust:1.90.0-alpine3.22@sha256:ee9b9f2ebd8741e4ecf1ade08918e492d642c4e625be481bb7b7954b89e9d504",
                "rustc",
                "--edition=2021",
                "-C",
                "opt-level=2",
                "--target",
                "x86_64-unknown-linux-musl",
                "/fixture/capability-fixture.rs",
                "-o",
                "/fixture/amlas-capability-fixture",
            ]
        )
        (scratch / "capability-fixture-build.log").write_text(compiled.stdout)
        assert compiled.returncode == 0, compiled.stdout
    fixture_copy = (
        "COPY --chmod=0555 amlas-capability-fixture /usr/local/bin/amlas-capability-fixture"
        if needs_kernel_fixture
        else ""
    )
    (context / "Dockerfile").write_text(f"""FROM {BASE}
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/src /opt/ml/processing/output && chown 10001:10001 /opt/ml/processing && chmod 0755 /opt/ml/processing
COPY --chmod=0555 amlas-processing-init /usr/local/bin/amlas-processing-init
{fixture_copy}
COPY --chmod=0444 worker.py /app/acquisition_worker.py
COPY --chmod=0444 worker.py /app/inspect_dataset.py
COPY --chmod=0444 worker.py /app/inspection_schema.py
ENV AMLAS_SOURCE_COMMIT={"c" * 40}
WORKDIR /app
USER root
ENTRYPOINT ["/usr/local/bin/amlas-processing-init", "acquisition"]
""")
    built = command(["docker", "build", "--network", "none", "--pull=false", "-t", image, str(context)])
    (scratch / "build.log").write_text(built.stdout)
    assert built.returncode == 0, built.stdout[-2000:]
    results = []
    cases = [
        "root-bind",
        "idempotent",
        "nonempty",
        "foreign-parent",
        "symlink",
        "readonly",
        "fchown-denied",
        "loader-env",
        "python-env",
        "profile-override",
        "credential-hint",
        "metadata-disable-hint",
        "nonroot-start",
        "metadata-endpoint-hint",
        "metadata-endpoint-mode-hint",
        "boto-config-hint",
        "python-metadata-override",
        "inspection-bind",
    ]
    cases += ["source-mismatch", "lease-contender", "empty-bounding-set", "present-undroppable"]
    cases += [
        "proxy-" + name
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        )
    ]
    if only_case:
        cases = [case for case in cases if case == only_case]
        assert cases, "unknown test case"
    for index, case in enumerate(cases):
        path = scratch / f"case-{index}-{case}"
        path.mkdir(exist_ok=True)
        setup = """import os,sys
from pathlib import Path
p=Path('/fixture'); parent=p/'parent'; target=p/'output'
parent.mkdir(); target.mkdir()
os.chown(parent,10002 if sys.argv[1]=='foreign-parent' else 10001,10001); parent.chmod(0o755)
os.chown(target,10001 if sys.argv[1]=='idempotent' else 0,10001 if sys.argv[1]=='idempotent' else 0)
target.chmod(0o700 if sys.argv[1]=='idempotent' else 0o755)
if sys.argv[1]=='nonempty': (target/'foreign.txt').write_text('fixture')
if sys.argv[1]=='symlink': (parent/'output').symlink_to('/fixture-unused')
"""
        made = command(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                "0:0",
                "--entrypoint",
                "python",
                "-v",
                f"{path}:/fixture",
                image,
                "-I",
                "-c",
                setup,
                case,
            ]
        )
        assert made.returncode == 0, made.stdout
        args = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{path / 'parent'}:/opt/ml/processing",
        ]
        if case != "symlink":
            args += [
                "-v",
                f"{path / 'output'}:/opt/ml/processing/output" + (":ro" if case == "readonly" else ""),
            ]
        if case == "fchown-denied":
            args += ["--cap-drop", "CHOWN"]
        if case == "nonroot-start":
            args += ["--user", "10001:10001"]
        if case in ("loader-env", "python-env"):
            args += ["-e", "LD_PRELOAD=fixture" if case == "loader-env" else "PYTHONPATH=/fixture"]
        if case.startswith("proxy-") or case in (
            "credential-hint",
            "metadata-disable-hint",
            "metadata-endpoint-hint",
            "metadata-endpoint-mode-hint",
            "boto-config-hint",
            "inspection-bind",
        ):
            hints = {
                "credential-hint": "AWS_ACCESS_KEY_ID=fixture-hint",
                "metadata-disable-hint": "AWS_EC2_METADATA_DISABLED=true",
                "metadata-endpoint-hint": "AWS_EC2_METADATA_SERVICE_ENDPOINT=http://127.0.0.1:1",
                "metadata-endpoint-mode-hint": "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE=IPv6",
                "boto-config-hint": "BOTO_CONFIG=/fixture-unused",
            }
            if case.startswith("proxy-"):
                hints[case] = case[len("proxy-") :] + "=http://127.0.0.1:1"
            if case in hints:
                args += ["-e", hints[case]]
            args += ["--entrypoint", "/usr/local/bin/amlas-processing-init"]
        if case == "python-metadata-override":
            args += ["-e", "PYTHON_VERSION=fixture-mutated"]
        if case in ("empty-bounding-set", "present-undroppable"):
            args += ["--entrypoint", "/usr/local/bin/amlas-capability-fixture"]
        args += [image]
        if case in ("empty-bounding-set", "present-undroppable"):
            args += ["empty" if case == "empty-bounding-set" else "present"]
        if case.startswith("proxy-") or case in (
            "credential-hint",
            "metadata-disable-hint",
            "metadata-endpoint-hint",
            "metadata-endpoint-mode-hint",
            "boto-config-hint",
            "inspection-bind",
        ):
            args += [
                "inspection",
                "--sha256",
                "a" * 64,
                "--byte-count",
                "1",
                "--format",
                "csv",
                "--job-name",
                "fixture",
                "--runtime-session-id",
                "session-a",
                "--deployment-id",
                "fixture",
                "--deployment-prefix",
                "fixture",
                "--image-uri",
                "fixture",
                "--source-commit",
                "c" * 40,
                "--source-version-id",
                "version-a",
                "--staged-version-id",
                "version-b",
            ]
        else:
            args += ARGS
            if case == "source-mismatch":
                args[-1] = "d" * 40
            if case == "profile-override":
                args += ["--output-dir", "/fixture"]
        first = None
        if case == "lease-contender":
            paused = list(args)
            paused[paused.index("--filename") + 1] = "lease.csv"
            first = subprocess.Popen(paused, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            marker = first.stdout.readline()
            assert marker.strip() == "LEASE-READY", marker
        try:
            got = command(args)
        finally:
            if first is not None:
                first_stdout, _ = first.communicate(timeout=20)
                (scratch / "lease-first.log").write_text(marker + first_stdout)
                assert first.returncode == 0, first_stdout

        (scratch / f"{case}.log").write_text(got.stdout)
        kernel_proof = None
        if case in ("empty-bounding-set", "present-undroppable"):
            lines = got.stdout.splitlines()
            assert lines and lines[0].startswith("CAPABILITY-FIXTURE "), got.stdout
            kernel_proof = json.loads(lines[0][len("CAPABILITY-FIXTURE ") :])
            assert kernel_proof["phase"] == "after_exec" and kernel_proof["drop_absent_errno"] == 1
            assert kernel_proof["cap_bnd"] == (
                "0000000000000000" if case == "empty-bounding-set" else "00000000000000cb"
            )
            if case == "present-undroppable":
                assert got.returncode == 78 and "stage=bounding_capability_drop" in got.stdout
        expect = case in ("root-bind", "idempotent", "inspection-bind", "empty-bounding-set")
        assert (got.returncode == 0) == expect, f"{case}: exit{got.returncode}: {got.stdout}"
        if case == "lease-contender":
            assert (path / "output").stat().st_uid == 10001
            results.append(
                {
                    "case": case,
                    "contender_exit": got.returncode,
                    "expected_success": False,
                    "first_worker_exit": 0,
                }
            )
            continue
        oracle_code = "import os,json,pathlib; p=pathlib.Path('/oracle/CANARY.txt'); print(json.dumps({'artifact':p.exists(),'literal_ok':p.read_bytes()==b'initializer-fixture-v1\\n' if p.exists() else False,'names':sorted(x.name for x in pathlib.Path('/oracle').iterdir())}))"
        oracle = command(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--user",
                "10001:10001",
                "-v",
                f"{path / 'output'}:/oracle:ro",
                "--entrypoint",
                "python",
                image,
                "-I",
                "-c",
                oracle_code,
            ]
        )
        assert oracle.returncode == 0, oracle.stdout
        observed = json.loads(oracle.stdout)
        assert observed["artifact"] == expect, case
        if expect:
            assert observed["literal_ok"] and observed["names"] == ["CANARY.txt"], case
            assert (path / "output").stat().st_uid == 10001
            assert (path / "output").stat().st_mode & 0o777 == 0o700
            assert json.loads(got.stdout.splitlines()[-1])["regain_root"] is False
        elif case == "present-undroppable":
            assert (path / "output").stat().st_uid == 10001, case
            assert (path / "output").stat().st_mode & 0o777 == 0o700, case
        else:
            assert (path / "output").stat().st_uid == 0, case
            assert (path / "output").stat().st_mode & 0o777 == 0o755, case
        results.append(
            {
                "case": case,
                "exit": got.returncode,
                "expected_success": expect,
                "artifact": observed["artifact"],
                "kernel_proof": kernel_proof,
            }
        )
    (scratch / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PROCESSING-INIT-ACCEPTANCE-PASS", len(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--only-case")
    opts = parser.parse_args()
    run(opts.binary, opts.scratch, opts.image, opts.only_case)
