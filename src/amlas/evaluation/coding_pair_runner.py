"""Fixed parent broker: no candidate or oracle source executes in SDK custody."""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import threading
import time
import uuid
from pathlib import Path
from subprocess import TimeoutExpired

from ..arbiter.coding import CodingCaseReceipt
from ..data_eng.coding_case_profile import CASE_IDS
from ..data_eng.inspection_schema import decode_object
from .coding_executor import (
    CandidateExecutionError,
    FileInvocationRecorder,
    _service_error_metadata,
    _validate_service_error,
    absolute_deadline,
    close_original_streams,
    invoke_recorded,
    require_deadline_context,
)
from .coding_oracle import decode
from .coding_oracle_catalog import verify_pinned_case

INTERPRETER_ID = "aws.codeinterpreter.v1"
BUDGETS = {
    "case_wall_seconds": 10,
    "session_timeout_seconds": 60,
    "max_value_calls": 512,
    "allocation_excluded": True,
    "cleanup_seconds": 30,
}


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def paired_runner_sha256(*, oracle_launcher_sha256: str):
    """Fingerprint the fixed image-owned broker/oracle/recording closure."""
    if type(oracle_launcher_sha256) is not str or not re.fullmatch("[a-f0-9]{64}", oracle_launcher_sha256):
        raise ValueError("trusted oracle launcher digest required")
    root = Path(__file__).parent
    names = (
        "coding_pair_runner.py",
        "coding_oracle.py",
        "coding_oracle_child.py",
        "coding_oracle_catalog.py",
        "coding_executor.py",
    )
    return hashlib.sha256(
        _canonical(
            {
                "domain": "amlas-paired-runner-v2",
                "python_closure": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
                },
                "oracle_launcher_sha256": oracle_launcher_sha256,
            }
        )
    ).hexdigest()


ORACLE_LAUNCHER = "/usr/local/bin/amlas-oracle-init"
MAX_LAUNCHER_BYTES = 16 * 1024 * 1024


def verify_oracle_launcher(expected_sha256):
    """Match the fixed installed binary against trusted startup/image-bound intent.

    Actual ownership and same-FD bytes are independently observed; a computed
    hash or returned digest alone grants no native admission or credential proof.
    """
    if type(expected_sha256) is not str or not re.fullmatch("[a-f0-9]{64}", expected_sha256):
        raise ValueError("trusted oracle launcher digest required")
    path = Path(ORACLE_LAUNCHER)
    if not path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("fixed oracle launcher path differs")
    owned = []
    fd = None
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def directory(fd):
        value = os.fstat(fd)
        if not stat.S_ISDIR(value.st_mode) or value.st_uid != 0 or value.st_gid != 0 or value.st_mode & 0o022:
            raise ValueError("oracle launcher ancestor ownership differs")
        return value

    def snapshot(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_uid,
            value.st_gid,
            value.st_mode,
            value.st_nlink,
        )

    try:
        current = os.open("/", directory_flags)
        directory(current)
        owned.append((current, None, None))
        for part in path.parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current)
            owned.append((next_fd, current, part))
            directory(next_fd)
            current = next_fd
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != 0
            or before.st_gid != 0
            or before.st_mode & 0o022
            or not before.st_mode & 0o111
            or not 0 < before.st_size <= MAX_LAUNCHER_BYTES
        ):
            raise ValueError("fixed oracle launcher ownership or size differs")
        digest, count = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, min(65536, MAX_LAUNCHER_BYTES - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > MAX_LAUNCHER_BYTES:
                raise ValueError("fixed oracle launcher size differs")
            digest.update(chunk)
        after = os.fstat(fd)
        named = os.stat(path.name, dir_fd=current, follow_symlinks=False)
        if (
            snapshot(before) != snapshot(after)
            or count != before.st_size
            or (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino)
            or digest.hexdigest() != expected_sha256
        ):
            raise ValueError("installed oracle launcher differs from trusted digest")
        for held, parent, name in owned:
            actual = directory(held)
            if parent is not None:
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if (named.st_dev, named.st_ino) != (actual.st_dev, actual.st_ino):
                    raise ValueError("oracle launcher ancestor changed")
        return expected_sha256
    finally:
        if fd is not None:
            os.close(fd)
        for held, _parent, _name in reversed(owned):
            os.close(held)


def oracle_child_command():
    # Fixed Rust launcher owns policy and Python argv; payload remains stdin-only.
    return ["/usr/local/bin/amlas-oracle-init"], {
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


class OracleProcessRunner:
    """Run only catalog-verified official assertions in a credential-free exec child."""

    def run(self, case, proxy, *, deadline, clock):
        verify_pinned_case(case["prompt"], case["oracle"])
        if clock() >= deadline:
            raise TimeoutError("original oracle deadline elapsed")
        argv, environment = oracle_child_command()
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            close_fds=True,
        )
        timer = threading.Timer(max(0.0, deadline - clock()), process.kill)
        timer.daemon = True
        timer.start()
        try:
            payload = _canonical({"prompt": case["prompt"], "oracle": case["oracle"]}) + b"\n"
            if len(payload) > 65536:
                raise ValueError("fixed oracle child input bound exceeded")
            process.stdin.write(payload)
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - clock()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("original external oracle deadline elapsed")
                    line = process.stdout.readline(65537)
                    if not line or len(line) > 65536 or not line.endswith(b"\n"):
                        raise ValueError("oracle child frame incomplete")
                    frame = decode_object(line, limit=65536)
                    if frame.get("kind") == "call" and set(frame) == {"kind", "value"}:
                        args, kwargs = decode(frame["value"])
                        if type(args) is not tuple or type(kwargs) is not dict:
                            raise ValueError("fixed oracle argument frame differs")
                        value = proxy(*args, **kwargs)
                        from .coding_oracle import encode

                        reply = _canonical({"value": encode(value)}) + b"\n"
                        if len(reply) > 65536:
                            raise ValueError("oracle child response bound exceeded")
                        process.stdin.write(reply)
                        process.stdin.flush()
                    elif frame.get("kind") == "verdict" and set(frame) == {
                        "kind",
                        "outcome",
                        "oracle_completed",
                        "exit_code",
                    }:
                        # A flushed verdict precedes interpreter shutdown. Keep the
                        # original deadline active until actual normal exit and EOF.
                        with absolute_deadline(deadline, clock=clock):
                            remaining = deadline - clock()
                            if remaining <= 0:
                                raise TimeoutError("original oracle exit deadline elapsed")
                            try:
                                returncode = process.wait(timeout=remaining)
                            except TimeoutExpired as error:
                                raise TimeoutError("original oracle exit deadline elapsed") from error
                            if clock() >= deadline:
                                raise TimeoutError("original oracle exit deadline elapsed")
                            if returncode != 0:
                                raise ValueError("oracle child requires normal exit")
                            if process.stdout.read(1):
                                raise ValueError("oracle child has trailing protocol bytes")
                            if clock() >= deadline:
                                raise TimeoutError("original oracle EOF deadline elapsed")
                        return {key: frame[key] for key in ("outcome", "oracle_completed", "exit_code")}
                    else:
                        raise ValueError("oracle child protocol differs")
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
            process.stdin.close()
            process.stdout.close()


_SESSION_FIELDS = {
    "start-attempt": {"request_sha256", "client_token", "allocation_excluded"},
    "start-accepted": {"session_id"},
    "start-unknown": {"client_token", "error_type"},
    "stop-attempt": {"session_id", "request_sha256"},
    "stop-accepted": {"session_id"},
    "get-attempt": {"session_id"},
    "get-observed": {"session_id", "status"},
    "get-failed": {"session_id", "error_type"},
    "cleanup-unknown": {"session_id", "error_type"},
}


def validate_session_record(record, *, kind, case_id, arm, evaluation_id=None):
    if (
        type(record) is not dict
        or kind not in _SESSION_FIELDS
        or not _SESSION_FIELDS[kind] | {"evaluation_id", "case_id", "arm"} <= set(record)
        or not set(record)
        <= _SESSION_FIELDS[kind]
        | {"evaluation_id", "case_id", "arm"}
        | ({"service_error"} if kind in ("start-unknown", "cleanup-unknown", "get-failed") else set())
    ):
        raise ValueError("closed original session record required")
    if "service_error" in record:
        _validate_service_error(record["service_error"])
    if (
        record["case_id"] != case_id
        or case_id not in CASE_IDS
        or record["arm"] != arm
        or arm not in ("base", "tuned")
        or type(record["evaluation_id"]) is not str
        or not re.fullmatch("[A-Za-z0-9_-]{1,128}", record["evaluation_id"])
        or (evaluation_id is not None and record["evaluation_id"] != evaluation_id)
    ):
        raise ValueError("original session record scope differs")
    for name in ("session_id", "client_token", "error_type", "status"):
        if name in record and (
            type(record[name]) is not str or not re.fullmatch("[A-Za-z0-9_-]{1,128}", record[name])
        ):
            raise ValueError("bounded session record value required")
    if "request_sha256" in record and (
        type(record["request_sha256"]) is not str
        or not re.fullmatch("[a-f0-9]{64}", record["request_sha256"])
    ):
        raise ValueError("session request hash differs")
    if "allocation_excluded" in record and record["allocation_excluded"] is not True:
        raise ValueError("original allocation budget contract differs")
    if len(_canonical(record)) > 65536:
        raise ValueError("session record bound exceeded")
    return record


class _SessionRecords:
    def __init__(self, directory, evaluation_id, case_id, arm):
        self.directory = directory
        self.scope = {"evaluation_id": evaluation_id, "case_id": case_id, "arm": arm}
        self.index = 0
        self.lock = threading.RLock()

    def write(self, kind, value):
        value = {**value, **self.scope}
        validate_session_record(
            value,
            kind=kind,
            case_id=self.scope["case_id"],
            arm=self.scope["arm"],
            evaluation_id=self.scope["evaluation_id"],
        )
        raw = _canonical(value) + b"\n"
        if len(raw) > 65536:
            raise ValueError("session record bound exceeded")
        with self.lock:
            name = f"{self.index:08d}-{kind}.json"
            self.index += 1
            root = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=root,
                )
                try:
                    view = memoryview(raw)
                    while view:
                        count = os.write(fd, view)
                        if count <= 0:
                            raise OSError("session record write incomplete")
                        view = view[count:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.fsync(root)
            finally:
                os.close(root)


_GUEST_CODEC = """\ndef _wire_encode(v,d=0):
 if d>16: raise ValueError('depth')
 t=type(v)
 if t in (str,int,bool,type(None)): return {'t':t.__name__,'v':v}
 if t is float:
  if not math.isfinite(v): raise ValueError('nonfinite')
  return {'t':'float','v':v}
 if t in (list,tuple): return {'t':t.__name__,'v':[_wire_encode(x,d+1) for x in v]}
 if t is dict: return {'t':'dict','v':[[_wire_encode(k,d+1),_wire_encode(x,d+1)] for k,x in v.items()]}
 raise ValueError('type')
def _wire_decode(v):
 t,x=v['t'],v['v']
 if t in ('str','int','bool','NoneType','float'): return x
 if t=='list': return [_wire_decode(i) for i in x]
 if t=='tuple': return tuple(_wire_decode(i) for i in x)
 if t=='dict': return {_wire_decode(k):_wire_decode(i) for k,i in x}
 raise ValueError('tag')
"""


def _stdout(events):
    if type(events) is not tuple or len(events) != 1:
        raise ValueError("original native event population differs")
    content = events[0]["result"]["structuredContent"].get("stdout")
    if type(content) is not str or len(content.encode()) > 65536:
        raise ValueError("native value stdout bound differs")
    return decode_object(content.encode(), limit=65536), hashlib.sha256(content.encode()).hexdigest()


def _cleanup(client, session_id, records, *, clock, sleep, streams=()):
    deadline = clock() + BUDGETS["cleanup_seconds"]
    request = {"codeInterpreterIdentifier": INTERPRETER_ID, "sessionId": session_id}
    try:
        with absolute_deadline(deadline, clock=clock):
            cleanup_error = None
            try:
                close_original_streams(streams)
            except Exception as error:
                cleanup_error = error
            try:
                if clock() >= deadline:
                    raise TimeoutError("original cleanup budget elapsed")
                records.write(
                    "stop-attempt",
                    {
                        "session_id": session_id,
                        "request_sha256": hashlib.sha256(_canonical(request)).hexdigest(),
                    },
                )
                response = client.stop_code_interpreter_session(**request)
                if type(response) is not dict or response.get("sessionId", session_id) != session_id:
                    raise ValueError("original stop identity differs")
                records.write("stop-accepted", {"session_id": session_id})
                terminal = False
                for _index in range(12):
                    if clock() >= deadline:
                        raise TimeoutError("original cleanup budget elapsed")
                    records.write("get-attempt", {"session_id": session_id})
                    try:
                        response = client.get_code_interpreter_session(**request)
                    except Exception as error:
                        metadata = _service_error_metadata(error)
                        if not metadata:
                            raise
                        records.write(
                            "get-failed",
                            {"session_id": session_id, "error_type": type(error).__name__, **metadata},
                        )
                        if _index == 11:
                            raise
                        sleep(0.5)
                        continue
                    if (
                        type(response) is not dict
                        or response.get("sessionId", session_id) != session_id
                        or type(response.get("status")) is not str
                    ):
                        raise ValueError("original terminal identity differs")
                    records.write("get-observed", {"session_id": session_id, "status": response["status"]})
                    if response["status"] in ("TERMINATED", "STOPPED"):
                        terminal = True
                        break
                    sleep(0.5)
                if not terminal:
                    raise ValueError("original session not observed terminal")
            except Exception as error:
                cleanup_error = error
            if cleanup_error is not None:
                # At most one final marker, still inside the original armed cap.
                if clock() < deadline:
                    records.write(
                        "cleanup-unknown",
                        {
                            "session_id": session_id,
                            "error_type": type(cleanup_error).__name__,
                            **_service_error_metadata(cleanup_error),
                        },
                    )
                return False
            if clock() >= deadline:
                raise TimeoutError("original cleanup budget elapsed")
            return True
    except Exception:
        # Expired/interrupted persistence retains its original prefix unresolved.
        # Starting another synchronous write here would silently extend the cap.
        return False


def run_paired(
    *,
    binding,
    cases,
    candidates,
    client,
    cleanup_client,
    evidence_root,
    oracle_runner=None,
    clock=time.monotonic,
    sleep=time.sleep,
    timer_factory=threading.Timer,
):
    """Execute fixed original sessions once; every arm/case keeps its denominator slot."""
    if (
        type(cases) is not tuple
        or len(cases) != 164
        or type(candidates) is not dict
        or set(candidates) != {"base", "tuned"}
        or any(type(candidates[arm]) is not tuple or len(candidates[arm]) != 164 for arm in candidates)
    ):
        raise ValueError("fixed2x164 private population required")
    require_deadline_context()
    root = Path(evidence_root)
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError("trusted empty paired invocation directory required")
    oracle_runner = oracle_runner or OracleProcessRunner()
    receipts, cleanup_complete = [], True
    for arm in ("base", "tuned"):
        arm_root = root / arm
        arm_root.mkdir(mode=0o700)
        for index, case in enumerate(cases):
            row = candidates[arm][index]
            if (
                case["prompt"]["case_id"] != CASE_IDS[index]
                or case["oracle"]["case_id"] != CASE_IDS[index]
                or row["case_id"] != CASE_IDS[index]
            ):
                raise ValueError("private arm/case order differs")
            identity = hashlib.sha256(_canonical([binding.evaluation_id, arm, CASE_IDS[index]])).hexdigest()[
                :48
            ]
            outcome = "truncated" if row["outcome"] == "truncated" else "incomplete"
            completed, exit_code, stdout_hash = False, None, hashlib.sha256(b"").hexdigest()
            source_hash = row.get("source_sha256") or hashlib.sha256(_canonical(row)).hexdigest()
            session_id, cleanup_ok = None, True
            watchdog = None
            deadline_hit = threading.Event()
            streams = []
            if row["outcome"] == "generated" and type(row.get("source")) is str:
                folder = arm_root / f"{index:03d}"
                folder.mkdir(mode=0o700)
                session_folder, calls_folder = folder / "session", folder / "calls"
                session_folder.mkdir(mode=0o700)
                calls_folder.mkdir(mode=0o700)
                records = _SessionRecords(session_folder, binding.evaluation_id, CASE_IDS[index], arm)
                request = {
                    "codeInterpreterIdentifier": INTERPRETER_ID,
                    "name": "amlas-pair-" + arm + "-" + str(index) + "-" + uuid.uuid4().hex[:6],
                    "sessionTimeoutSeconds": 60,
                    "clientToken": str(uuid.uuid4()),
                }
                records.write(
                    "start-attempt",
                    {
                        "request_sha256": hashlib.sha256(_canonical(request)).hexdigest(),
                        "client_token": request["clientToken"],
                        "case_id": CASE_IDS[index],
                        "arm": arm,
                        "allocation_excluded": True,
                    },
                )
                try:
                    response = client.start_code_interpreter_session(**request)
                    if (
                        type(response) is not dict
                        or type(response.get("sessionId")) is not str
                        or not re.fullmatch("[A-Za-z0-9_-]{1,128}", response["sessionId"])
                    ):
                        raise ValueError("original allocation identity unresolved")
                    session_id = response["sessionId"]
                    records.write("start-accepted", {"session_id": session_id})
                    deadline = clock() + 10

                    def expire(_event=deadline_hit):
                        # Observation only: no SDK or record effect runs on a timer thread.
                        _event.set()

                    with absolute_deadline(deadline, clock=clock):
                        watchdog = timer_factory(10, expire)
                        watchdog.daemon = True
                        watchdog.start()
                        if deadline_hit.is_set():
                            raise TimeoutError("original whole-case external deadline elapsed")
                        with FileInvocationRecorder(calls_folder) as sink:
                            call_state = {"ordinal": 0}

                            def invoke(
                                code,
                                _session=session_id,
                                _sink=sink,
                                _deadline=deadline,
                                _case_id=CASE_IDS[index],
                                _arm=arm,
                                _state=call_state,
                                _streams=streams,
                            ):
                                nonlocal stdout_hash
                                request = {
                                    "codeInterpreterIdentifier": INTERPRETER_ID,
                                    "sessionId": _session,
                                    "name": "executeCode",
                                    "arguments": {"language": "python", "clearContext": False, "code": code},
                                }
                                events = invoke_recorded(
                                    client,
                                    request,
                                    binding={
                                        "evaluation_id": binding.evaluation_id,
                                        "case_id": _case_id,
                                        "arm": _arm,
                                        "ordinal": _state["ordinal"],
                                    },
                                    persist=_sink,
                                    deadline=_deadline,
                                    clock=clock,
                                    retain_stream=_streams.append,
                                )
                                value, stdout_hash = _stdout(events)
                                return value

                            bootstrap = (
                                "import json,math\n"
                                + row["source"]
                                + "\n"
                                + _GUEST_CODEC
                                + "\n_fn="
                                + case["prompt"]["entry_point"]
                                + "\nprint(json.dumps({'ready':True}))"
                            )
                            if invoke(bootstrap) != {"ready": True}:
                                raise ValueError("original candidate bootstrap incomplete")

                            def proxy(*args, _state=call_state, _deadline=deadline, _invoke=invoke, **kwargs):
                                from .coding_oracle import encode

                                _state["ordinal"] += 1
                                ordinal = _state["ordinal"]
                                if ordinal > 512 or clock() >= _deadline:
                                    raise TimeoutError("original fixed case budget elapsed")
                                wire = _canonical(encode((args, kwargs))).decode()
                                if len(wire.encode()) > 65536:
                                    raise ValueError("fixed value request byte bound exceeded")
                                code = (
                                    "_a,_k=_wire_decode(json.loads("
                                    + repr(wire)
                                    + "))\nprint(json.dumps({'ordinal':"
                                    + str(ordinal)
                                    + ",'value':_wire_encode(_fn(*_a,**_k))},sort_keys=True,allow_nan=False))"
                                )
                                value = _invoke(code)
                                if (
                                    type(value) is not dict
                                    or set(value) != {"ordinal", "value"}
                                    or type(value["ordinal"]) is not int
                                    or value["ordinal"] != ordinal
                                ):
                                    raise ValueError("original value RPC ordinal differs")
                                return decode(value["value"])

                            verdict = oracle_runner.run(case, proxy, deadline=deadline, clock=clock)
                            if clock() >= deadline or deadline_hit.is_set():
                                raise TimeoutError("original external deadline elapsed")
                            if (
                                type(verdict) is not dict
                                or set(verdict) != {"outcome", "oracle_completed", "exit_code"}
                                or verdict["outcome"]
                                not in ("passed", "failed", "runtime_error", "incomplete")
                                or type(verdict["oracle_completed"]) is not bool
                            ):
                                raise ValueError("fixed oracle verdict differs")
                            outcome, completed, exit_code = (
                                verdict["outcome"],
                                verdict["oracle_completed"],
                                verdict["exit_code"],
                            )
                except TimeoutError:
                    outcome, completed, exit_code = "timeout", False, None
                    if session_id is None:
                        records.write(
                            "start-unknown",
                            {"client_token": request["clientToken"], "error_type": "TimeoutError"},
                        )
                        cleanup_ok = False
                except CandidateExecutionError as error:
                    outcome, completed, exit_code = "runtime_error", False, error.exit_code
                except Exception as error:
                    outcome, completed, exit_code = "infrastructure_error", False, None
                    if session_id is None:
                        records.write(
                            "start-unknown",
                            {
                                "client_token": request["clientToken"],
                                "error_type": type(error).__name__,
                                **_service_error_metadata(error),
                            },
                        )
                        cleanup_ok = False
                finally:
                    if watchdog is not None:
                        watchdog.cancel()
                    if session_id is not None:
                        cleanup_ok = _cleanup(
                            cleanup_client, session_id, records, clock=clock, sleep=sleep, streams=streams
                        )
                    if deadline_hit.is_set():
                        outcome, completed, exit_code = "timeout", False, None
                cleanup_complete = cleanup_complete and cleanup_ok
            receipts.append(
                CodingCaseReceipt(
                    case_id=CASE_IDS[index],
                    arm=arm,
                    binding=binding,
                    model_sha256=binding.base_model_sha256 if arm == "base" else binding.tuned_model_sha256,
                    candidate_sha256=source_hash,
                    oracle_sha256=case["oracle_sha256"],
                    outcome=outcome,
                    oracle_completed=completed,
                    exit_code=exit_code,
                    execution_id=identity,
                    stdout_sha256=stdout_hash,
                )
            )
    return {
        "receipts": tuple(receipts),
        "cleanup_complete": cleanup_complete,
        "case_wall_seconds": 10,
        "allocation_excluded": True,
        "native_admission": False,
        "quality_success": False,
    }
