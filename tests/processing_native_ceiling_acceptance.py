"""Matched native-ceiling kernel oracle; fixtures never enter product images."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from processing_initializer_acceptance import ARGS, BASE

RUST_IMAGE = "public.ecr.aws/docker/library/rust:1.90.0-alpine3.22@sha256:ee9b9f2ebd8741e4ecf1ade08918e492d642c4e625be481bb7b7954b89e9d504"
CAPS = [
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "FSETID",
    "KILL",
    "SETGID",
    "SETUID",
    "NET_BIND_SERVICE",
    "SYS_CHROOT",
    "AUDIT_WRITE",
]
CEILING = 0x200404FB
WORKER = r"""import ctypes,errno,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/app')
from processing_identity import verify_native_identity
proof=verify_native_identity()
assert proof['cap_bnd']==0x200404fb
lib=ctypes.CDLL(None,use_errno=True)
denied={}
for name,function in [('uid_root',lambda:os.setresuid(0,0,0)),('gid_root',lambda:os.setresgid(0,0,0)),('groups_root',lambda:os.setgroups([0]))]:
 try:function()
 except PermissionError as e:denied[name]=e.errno
 else:raise RuntimeError(name+' unexpectedly restored')
class Header(ctypes.Structure):_fields_=[('version',ctypes.c_uint32),('pid',ctypes.c_int)]
class Data(ctypes.Structure):_fields_=[('effective',ctypes.c_uint32),('permitted',ctypes.c_uint32),('inheritable',ctypes.c_uint32)]
header=Header(0x20080522,0);data=(Data*2)(Data(1,1,0),Data(0,0,0))
assert lib.capset(ctypes.byref(header),data)==-1;denied['capset_raise']=ctypes.get_errno()
assert lib.prctl(47,2,0,0,0)==-1;denied['ambient_raise']=ctypes.get_errno()
assert lib.prctl(38,0,0,0,0)==-1;denied['nnp_clear']=ctypes.get_errno()
lib.setfsuid(ctypes.c_uint32(0));lib.setfsgid(ctypes.c_uint32(0))
assert verify_native_identity()['fs_uid']==10001 and verify_native_identity()['fs_gid']==10001
assert lib.setfsuid(ctypes.c_uint32(0xffffffff))==10001 and lib.setfsgid(ctypes.c_uint32(0xffffffff))==10001
namespace=Path('/proc/self/ns/user').stat().st_ino
assert lib.unshare(0x10000000)==-1;denied['unshare_user']=ctypes.get_errno()
fd=os.open('/proc/self/ns/user',os.O_RDONLY)
assert lib.setns(fd,0x10000000)==-1;denied['setns_self_user_not_alternate_target']=ctypes.get_errno();os.close(fd)
assert Path('/proc/self/ns/user').stat().st_ino==namespace
children={}
for kind in ['plain','setuid','setgid','filecap']:
 got=subprocess.run(['/app/report-'+kind],capture_output=True,text=True,check=True)
 row=json.loads(got.stdout)
 assert row['uid']==[10001]*4 and row['gid']==[10001]*4 and row['groups']==[]
 assert row['caps']==[0,0,0,0,0x200404fb] and row['no_new_privs']==1
 assert row['userns_inode']==namespace
 children[kind]=row
# Explicit fork+exec, in addition to the validated fixture subprocess execs.
pid=os.fork()
if pid==0:os.execv('/app/report-plain',['/app/report-plain'])
_,status=os.waitpid(pid,0);assert os.waitstatus_to_exitcode(status)==0
Path('/opt/ml/processing/output/CANARY.txt').write_bytes(b'native-ceiling-fixture-v1\n')
print(json.dumps({'proof':proof,'denied':denied,'children':children,'fs_restore_denied':True,'fork_exec_exit':0,'usernamespace_scope':'original; self-setns is not alternate-target closure','original_userns_inode':namespace}),flush=True)
"""


def command(args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def run(binary, scratch, image, guard_source, only_case=None):
    scratch.mkdir(parents=True, exist_ok=True)
    ctx = scratch / "context"
    ctx.mkdir()
    (ctx / "amlas-processing-init").write_bytes(binary.read_bytes())
    (ctx / "worker.py").write_text(WORKER)
    (ctx / "processing_identity.py").write_bytes(guard_source.read_bytes())
    reporter = Path(__file__).with_name("processing_native_reporter.rs")
    (ctx / "reporter.rs").write_bytes(reporter.read_bytes())
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
            f"{ctx}:/ctx",
            RUST_IMAGE,
            "rustc",
            "--edition=2021",
            "-C",
            "opt-level=2",
            "--target",
            "x86_64-unknown-linux-musl",
            "/ctx/reporter.rs",
            "-o",
            "/ctx/report",
        ]
    )
    assert compiled.returncode == 0, compiled.stdout
    locked = Path(__file__).with_name("processing_native_locked_fixture.rs")
    (ctx / "locked.rs").write_bytes(locked.read_bytes())
    built_locked = command(
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
            f"{ctx}:/ctx",
            RUST_IMAGE,
            "rustc",
            "--edition=2021",
            "-C",
            "opt-level=2",
            "--target",
            "x86_64-unknown-linux-musl",
            "/ctx/locked.rs",
            "-o",
            "/ctx/locked-fixture",
        ]
    )
    assert built_locked.returncode == 0, built_locked.stdout
    (ctx / "Dockerfile").write_text(
        f"""FROM {BASE}
RUN useradd --uid10001 --create-home appuser && mkdir -p /app/src /opt/ml/processing/output && chown10001:10001 /opt/ml/processing && chmod0755 /opt/ml/processing
COPY --chmod=0555 amlas-processing-init /usr/local/bin/amlas-processing-init
COPY --chmod=0444 worker.py /app/acquisition_worker.py
COPY --chmod=0444 processing_identity.py /app/processing_identity.py
COPY --chmod=0555 locked-fixture /app/locked-fixture
COPY --chmod=0555 report /app/report-plain
COPY --chmod=0555 report /app/report-setuid
COPY --chmod=0555 report /app/report-setgid
COPY --chmod=0555 report /app/report-filecap
RUN python -c "import os,struct;os.chmod('/app/report-setuid',0o4755);os.chown('/app/report-setgid',0,20002);os.chmod('/app/report-setgid',0o2755);os.setxattr('/app/report-filecap','security.capability',struct.pack('<IIIII',0x02000001,1,0,0,0))"
ENV AMLAS_SOURCE_COMMIT={"c" * 40}
USER root
ENTRYPOINT ["/usr/local/bin/amlas-processing-init","acquisition"]
""".replace("--uid10001", "--uid 10001")
        .replace("chown10001", "chown 10001")
        .replace("chmod0755", "chmod 0755")
    )
    built = command(["docker", "build", "--network", "none", "--pull=false", "-t", image, str(ctx)])
    (scratch / "build.log").write_text(built.stdout)
    assert built.returncode == 0, built.stdout[-2000:]
    caps = ["--cap-drop", "ALL"]
    for capability in CAPS:
        caps.extend(["--cap-add", capability])
    # Validate actual fixture effects before using their negative observations.
    positives = {}
    for kind in ["plain", "setuid", "setgid", "filecap"]:
        got = command(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                "10001:10001",
                *caps,
                "--entrypoint",
                "/app/report-" + kind,
                image,
            ]
        )
        assert got.returncode == 0, got.stdout
        row = json.loads(got.stdout)
        assert row["no_new_privs"] == 0 and row["securebits"] & 3 == 0
        if kind == "setuid":
            assert row["uid"][1:3] == [0, 0] and row["caps"][0] != 0
        if kind == "setgid":
            assert row["gid"][1:3] == [20002, 20002]
        if kind == "filecap":
            assert row["caps"][0] == 1 and row["caps"][1] == 1
        positives[kind] = row
    (scratch / "validated-fixtures.json").write_text(json.dumps(positives, indent=2) + "\n")
    cases = [
        "matched",
        "outside-ceiling",
        "setpcap-present",
        "supplementary-group",
        "nonroot-start",
        "runtime-profile-override",
        "locked-keepcaps-setter",
        "missing-setgid",
    ]
    if only_case:
        cases = [only_case]
    results = []
    for case in cases:
        p = scratch / case
        p.mkdir()
        setup = "import os;from pathlib import Path;p=Path('/fixture');a=p/'parent';b=p/'output';a.mkdir();b.mkdir();os.chown(a,10001,10001);a.chmod(0o755);b.chmod(0o755)"
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
                f"{p}:/fixture",
                image,
                "-I",
                "-c",
                setup,
            ]
        )
        assert made.returncode == 0, made.stdout
        args = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            *caps,
            "-v",
            f"{p}/parent:/opt/ml/processing",
            "-v",
            f"{p}/output:/opt/ml/processing/output",
        ]
        if case == "outside-ceiling":
            args += ["--cap-add", "NET_RAW"]
        if case == "setpcap-present":
            args += ["--cap-add", "SETPCAP"]
        if case == "supplementary-group":
            args += ["--group-add", "10002"]
        if case == "nonroot-start":
            args += ["--user", "10001:10001"]
        if case == "runtime-profile-override":
            args += [
                "-e",
                "AMLAS_PRIVILEGE_PROFILE=zero_bounding_v1",  # pragma: allowlist secret
            ]
        if case == "locked-keepcaps-setter":
            args += ["--cap-add", "SETPCAP", "--entrypoint", "/app/locked-fixture"]
        if case == "missing-setgid":
            position = args.index("SETGID")
            del args[position - 1 : position + 1]
        got = command([*args, image, *ARGS])
        (scratch / (case + ".log")).write_text(got.stdout)
        expected = case in ["matched", "runtime-profile-override"]
        assert (got.returncode == 0) == expected, f"{case}: exit{got.returncode}: {got.stdout}"
        if expected:
            info = (p / "output").stat()
            assert (info.st_uid, info.st_gid, info.st_mode & 0o777) == (10001, 10001, 0o700)
            row = json.loads(got.stdout.splitlines()[-1])
            assert row["proof"]["cap_bnd"] == CEILING
            results.append({"case": case, "exit": 0, "worker": row})
        elif case in ("locked-keepcaps-setter", "missing-setgid"):
            info = (p / "output").stat()
            assert (info.st_uid, info.st_gid, info.st_mode & 0o777) == (10001, 10001, 0o700)
            expected_stage = (
                "native_keep_ambient_set" if case == "locked-keepcaps-setter" else "native_identity_drop"
            )
            assert "stage=" + expected_stage in got.stdout
            results.append({"case": case, "exit": got.returncode, "setter_failure_refused": True})
        else:
            info = (p / "output").stat()
            assert (info.st_uid, info.st_gid, info.st_mode & 0o777) == (0, 0, 0o755)
            results.append({"case": case, "exit": got.returncode, "output_unchanged": True})
    (scratch / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PROCESSING-NATIVE-CEILING-PASS", len(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--guard-source", type=Path, required=True)
    p.add_argument("--only-case")
    a = p.parse_args()
    run(a.binary, a.scratch, a.image, a.guard_source, a.only_case)
