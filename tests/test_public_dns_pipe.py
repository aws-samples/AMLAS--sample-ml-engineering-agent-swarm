"""Real fork plus bounded malformed pipe frames; fake DNS and no network."""

from __future__ import annotations

import concurrent.futures
import json
import math
import multiprocessing
import multiprocessing.connection
import os
import re
import select
import signal
import struct
import time

import pytest

from amlas.tools.research import public_fetch as module

URL = "https://example.com/public"
PUBLIC = "1.1.1.1"
IPV6 = "2606:4700:4700::1111"
LIMIT = 4092
# Wall-clock values. LIVENESS only ends a hung case: it lies far above any delay of a loaded
# machine, and a call returns as soon as the helper answers, so no case waits for it. Cases
# whose subject is a frame, guard, schema or cleanup rule hand it to the helper as its budget.
# A stalled helper sleeps STALL, far longer than any bound a case asserts.
LIVENESS = 60.0
STALL = 5 * LIVENESS
REFUSED = "public URL validation failed"
TIMED_OUT = "public DNS validation exceeded the retrieval timeout"
ANONYMOUS_ONLY = "only anonymous HTTP(S) public URLs are allowed"
WEB_PORTS_ONLY = "URL control characters or nonstandard web ports are not allowed"
PUBLIC_HOSTS_ONLY = "internal hostnames are not public research sources"
HELPERS = []


def payload(ips=None):
    return json.dumps(
        {"status": "ok", "ips": [PUBLIC] if ips is None else ips}, separators=(",", ":")
    ).encode("ascii")


def noted(waits, name, call, default):
    """Pass one bounded wait of the product through unchanged and note how long it may take."""

    def wait(*args, **kwargs):
        timeout = args[0] if args else kwargs.get("timeout", default)
        waits.append((name, math.inf if timeout is None else timeout))
        return call(*args, **kwargs)

    return wait


class NotedClock:
    """The time module as the product sees it: every call passes through, a sleep is also noted."""

    def __init__(self, waits):
        self.sleep = noted(waits, "sleep", time.sleep, None)

    def __getattr__(self, name):
        return getattr(time, name)


class ForkContext:
    """Keep real fork/Pipe; replace only the child's DNS or adversarial writer."""

    def __init__(self, operation=None, started=None, waits=None):
        self.native = multiprocessing.get_context("fork")
        self.operation = operation
        self.started = started
        self.waits = waits
        self.ends = []
        self.processes = []
        self.queue_calls = 0

    def Queue(self, *args, **kwargs):  # noqa: N802 - exact multiprocessing context API
        self.queue_calls += 1
        raise PermissionError("sem_open disallowed in fixed read-only IPC tree")

    def Pipe(self, duplex=True):  # noqa: N802 - exact multiprocessing context API
        assert duplex is False
        self.ends = list(self.native.Pipe(duplex=False))
        if self.waits is not None:
            self.ends[0].poll = noted(self.waits, "poll", self.ends[0].poll, 0.0)
        return tuple(self.ends)

    def Process(self, *, target, daemon):  # noqa: N802 - exact multiprocessing context API
        assert daemon is True
        if self.operation is not None:
            reader, writer = self.ends

            def run():
                reader.close()
                self.operation(writer)
                writer.close()

            target = run
        process = self.native.Process(target=target, daemon=daemon)
        if self.started is not None:
            native_start = process.start

            def start():
                native_start()
                self.started()

            process.start = start
        if self.waits is not None:
            process.join = noted(self.waits, "join", process.join, None)
        self.processes.append(process)
        HELPERS.append(process)
        return process


@pytest.fixture(autouse=True)
def no_helper_outlives_its_case():
    """Only a failing case leaves a helper behind; end it instead of letting it sleep on."""
    yield
    while HELPERS:
        process = HELPERS.pop()
        if not process._closed and process.pid is not None:
            process.kill()
            process.join()
            process.close()


def configured(monkeypatch, operation=None, started=None, waits=None):
    """With a waits list, note every wait the parent asks for, as (name, seconds) in call order:
    each poll of its reader, each join of its helper and each sleep of the product module."""
    context = ForkContext(operation, started, waits)
    if waits is not None:
        monkeypatch.setattr(module, "time", NotedClock(waits))
    monkeypatch.setattr(
        module.multiprocessing,
        "get_context",
        lambda method: context if method == "fork" else pytest.fail("non-fork context"),
    )
    monkeypatch.setattr(module.pathref, "_guard_http", lambda _url: None)
    monkeypatch.setattr(module.pathref, "_resolve_host_ips", lambda _host: {PUBLIC, IPV6})
    return context


def assert_closed(context):
    assert context.queue_calls == 0
    assert all(end.closed for end in context.ends)
    for process in context.processes:
        assert process._closed, "reaped Process sentinel must be released"


def refusal(monkeypatch, operation, timeout=LIVENESS, reason=REFUSED, started=None, waits=None):
    context = configured(monkeypatch, operation, started, waits)
    try:
        module._checked_addresses(URL, timeout)
    except Exception as error:
        assert isinstance(error, module.PublicFetchError), (
            "pipe/DNS failure must have a closed public refusal"
        )
        assert "private-detail" not in str(error) and URL not in str(error)
        assert str(error) == reason, "the rule under test must refuse, not another limit"
    else:
        pytest.fail("malformed/failed DNS helper must not yield pinned addresses")
    assert_closed(context)


def decode(data):
    function = getattr(module, "_decode_dns_wire", None)
    assert callable(function), "missing bounded data-only DNS wire decoder"
    return function(data)


def test_actual_fork_success_without_semaphore_or_pickle_connections(monkeypatch):
    context = configured(monkeypatch)
    monkeypatch.setattr(multiprocessing.connection.Connection, "send", lambda *_: pytest.fail("pickle send"))
    monkeypatch.setattr(multiprocessing.connection.Connection, "recv", lambda *_: pytest.fail("pickle recv"))
    try:
        ips = module._checked_addresses(URL, LIVENESS)
    except OSError:
        pytest.fail("DNS still allocates a semaphore-backed Queue under sem_open denial")
    assert ips == [PUBLIC, IPV6]
    assert_closed(context)


def test_actual_executor_dns_helper_returns_valid_public_addresses(monkeypatch):
    context = configured(monkeypatch)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        addresses = executor.submit(module._checked_addresses, URL, LIVENESS).result(timeout=2 * LIVENESS)
    assert addresses == [PUBLIC, IPV6]
    assert_closed(context)


def test_actual_executor_dns_guard_refusal_stays_closed(monkeypatch):
    context = configured(monkeypatch)

    def denied(_url):
        raise ValueError("ordinary guard refusal")

    monkeypatch.setattr(module.pathref, "_guard_http", denied)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(module._checked_addresses, URL, LIVENESS)
        with pytest.raises(module.PublicFetchError, match=r"^public URL validation failed$"):
            future.result(timeout=2 * LIVENESS)
    assert_closed(context)


def test_actual_executor_dns_send_failure_does_not_return_addresses(monkeypatch):
    context = configured(monkeypatch)

    def failed_send(_self, _payload, *_args, **_kwargs):
        raise BrokenPipeError("ordinary send refusal")

    monkeypatch.setattr(multiprocessing.connection.Connection, "send_bytes", failed_send)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(module._checked_addresses, URL, LIVENESS)
        with pytest.raises(module.PublicFetchError, match=r"^public URL validation failed$"):
            future.result(timeout=2 * LIVENESS)
    assert_closed(context)


def test_dns_writer_close_failure_after_valid_packet_remains_abnormal(monkeypatch):
    context = configured(monkeypatch)
    original_pipe = context.Pipe
    owner_pid = os.getpid()
    observed_exitcodes = []
    native_close = multiprocessing.process.BaseProcess.close

    class Writer:
        def __init__(self, inner):
            self.inner = inner

        def fileno(self):
            return self.inner.fileno()

        def send_bytes(self, raw):
            return self.inner.send_bytes(raw)

        @property
        def closed(self):
            return self.inner.closed

        def close(self):
            self.inner.close()
            if os.getpid() != owner_pid:
                raise RuntimeError("ordinary writer close failure")

    def pipe(duplex=False):
        reader, writer = original_pipe(duplex=duplex)
        context.ends = [reader, Writer(writer)]
        return tuple(context.ends)

    def close(process):
        observed_exitcodes.append(process.exitcode)
        return native_close(process)

    monkeypatch.setattr(context, "Pipe", pipe)
    monkeypatch.setattr(multiprocessing.process.BaseProcess, "close", close)
    with pytest.raises(module.PublicFetchError, match=r"^public URL validation failed$"):
        module._checked_addresses(URL, LIVENESS)
    assert observed_exitcodes == [1]
    assert_closed(context)


def test_actual_queue_refuses_read_only_ipc_tree_but_resolver_fork_succeeds(monkeypatch):
    if not os.statvfs("/dev/shm").f_flag & os.ST_RDONLY:
        pytest.skip("separate strict focused runner mounts /dev/shm read-only")
    native = multiprocessing.get_context("fork")
    with pytest.raises(OSError) as error:
        native.Queue(maxsize=1)
    assert error.value.errno in {13, 30}
    monkeypatch.setattr(module.pathref, "_guard_http", lambda _url: None)
    monkeypatch.setattr(module.pathref, "_resolve_host_ips", lambda _host: {PUBLIC})
    try:
        ips = module._checked_addresses(URL, LIVENESS)
    except OSError:
        pytest.fail("real Queue allocation prevents DNS under read-only /dev/shm")
    assert ips == [PUBLIC]


def test_companion_maximum_atomic_frame_is_accepted_with_normal_helper_exit(monkeypatch):
    wire = payload()
    # JSON whitespace remains data-only and pads the small valid frame exactly.
    wire += b" " * (LIMIT - len(wire))
    context = configured(monkeypatch, lambda writer: writer.send_bytes(wire))
    assert module._checked_addresses(URL, LIVENESS) == [PUBLIC]
    assert_closed(context)


@pytest.mark.parametrize(
    "wire",
    [
        b"",
        b"not JSON private-detail",
        b"\xff",
        b'{"status":"ok","ips":["1.1.1.1"],"status":"ok"}',
        b'{"status":"ok","ips":[NaN]}',
        b'{"status":"ok","ips":[{"nested":[]}]}',
        b'{"status":"error","detail":"private-detail"}',
        b'{"status":"ok","ips":["1.1.1.1"],"extra":true}',
        b'{"status":"error"}',
        b"[]",
    ],
)
def test_actual_fork_malformed_frames_refuse_and_cleanup(monkeypatch, wire):
    refusal(monkeypatch, lambda writer: writer.send_bytes(wire))


@pytest.mark.parametrize(
    "header",
    [
        b"\x00",
        struct.pack("!i", 100),
        struct.pack("!i", -1),
        struct.pack("!i", -1) + struct.pack("!Q", 2**63),
        struct.pack("!i", LIMIT + 1),
    ],
)
def test_partial_header_body_64bit_and_oversize_refuse_promptly(monkeypatch, header):
    def incomplete(writer):
        os.write(writer.fileno(), header)
        time.sleep(STALL)

    waits = []
    start = time.monotonic()
    # REFUSED is the frame rule: the parent waited neither for the rest of the frame nor for
    # its time limit, and it answered long before the helper would have ended.
    refusal(monkeypatch, incomplete, waits=waits)
    assert time.monotonic() - start < LIVENESS
    # Prompt on a machine of any speed, read from what the parent asked for and not from a
    # clock: its first wait is the poll that met the partial frame, and every wait after it
    # (reaping the helper) adds up to less than a second.
    assert waits[0][0] == "poll"
    assert sum(seconds for _name, seconds in waits[1:]) < 1.0


def test_eof_without_message_refuses(monkeypatch):
    refusal(monkeypatch, lambda _writer: None)


def test_abnormal_exit_after_otherwise_valid_packet_refuses(monkeypatch):
    def failed(writer):
        writer.send_bytes(payload())
        os._exit(7)

    refusal(monkeypatch, failed)


def test_timeout_kills_term_ignoring_helper_and_closes_own_descriptors(monkeypatch):
    budget = 0.05
    ignoring, report = os.pipe()
    reported = []
    kills = []
    waits = []
    native_kill = multiprocessing.process.BaseProcess.kill

    def stalled(_writer):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.write(report, b"1")
        time.sleep(STALL)

    def ignores_sigterm():
        # Process.start returns once the helper ignores SIGTERM, so the time limit always
        # meets a helper that only SIGKILL ends, however slowly the machine forks.
        reported.append(bool(select.select([ignoring], [], [], LIVENESS)[0]))

    def kill(process):
        kills.append(process.pid)
        return native_kill(process)

    monkeypatch.setattr(multiprocessing.process.BaseProcess, "kill", kill)
    start = time.monotonic()
    try:
        refusal(monkeypatch, stalled, timeout=budget, reason=TIMED_OUT, started=ignores_sigterm, waits=waits)
    finally:
        os.close(ignoring)
        os.close(report)
    # The refusal came long before the helper would have ended, and SIGKILL was needed once.
    assert time.monotonic() - start < LIVENESS
    assert reported == [True]
    assert len(kills) == 1
    # The limit is enforced on time and reaping is short on a machine of any speed, read from
    # what the parent asked for and not from a clock: no poll may wait longer than the budget
    # (a start that already used the budget up leaves no poll), and all waits together (the
    # poll, then reaping after SIGTERM and after SIGKILL) add up to less than a second.
    assert max((seconds for name, seconds in waits if name == "poll"), default=0.0) <= budget
    assert sum(seconds for _name, seconds in waits) < 1.0


def test_helper_guard_failure_never_serializes_error_contents(monkeypatch):
    context = configured(monkeypatch)

    def denied(_url):
        raise RuntimeError("private-detail " + URL)

    monkeypatch.setattr(module.pathref, "_guard_http", denied)
    with pytest.raises(module.PublicFetchError) as error:
        module._checked_addresses(URL, LIVENESS)
    assert str(error.value) == "public URL validation failed"
    assert_closed(context)


@pytest.mark.parametrize(
    "data",
    [
        None,
        "not bytes",
        b" " * (LIMIT + 1),
        b"",
        b"\xff",
        b"{}",
        b'{"status":true,"ips":["1.1.1.1"]}',
        b'{"status":"ok","ips":"1.1.1.1"}',
        b'{"status":"ok","ips":[]}',
        b'{"status":"ok","ips":[1]}',
        b'{"status":"ok","ips":[true]}',
        b'{"status":"ok","ips":["private-detail"]}',
        b'{"status":"ok","ips":["127.0.0.1"]}',
        b'{"status":"ok","ips":["10.0.0.1"]}',
        b'{"status":"ok","ips":["224.0.0.1"]}',
        b'{"status":"ok","ips":["::1"]}',
        b'{"status":"ok","ips":["2001:db8::1"]}',
        b'{"status":"ok","ips":["2606:4700:4700::1111%eth0"]}',
        payload([PUBLIC] * 65),
        b'{"status":"ok","ips":["1.1.1.1"],"unknown":0}',
        b'{"status":"ok","ips":["1.1.1.1"],"status":"ok"}',
        b'{"status":"ok","ips":[Infinity]}',
        b'{"status":"ok","ips":[[["1.1.1.1"]]]}',
        b'{"status":"error"}',
    ],
)
def test_parent_exact_bounded_data_schema_refuses(data):
    with pytest.raises(module.PublicFetchError):
        decode(data)


def test_parent_decoder_repeats_both_ip_globality_guards(monkeypatch):
    assert decode(payload([IPV6, PUBLIC])) == [PUBLIC, IPV6]
    monkeypatch.setattr(module.pathref, "_is_blocked_ip", lambda _ip: True)
    with pytest.raises(module.PublicFetchError):
        decode(payload())


def test_measured_atomicity_mismatch_refuses_before_process_creation(monkeypatch):
    context = configured(monkeypatch)
    monkeypatch.setattr(os, "fpathconf", lambda *_: 512)
    with pytest.raises(module.PublicFetchError, match=r"^public DNS pipe atomicity bound is unavailable$"):
        module._checked_addresses(URL, LIVENESS)
    assert context.processes == []
    assert all(end.closed for end in context.ends)


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        (
            "https://user:password@example.com",  # pragma: allowlist secret - inert URL rejection fixture
            ANONYMOUS_ONLY,
        ),
        (
            "https://" + ".".join(("corp", "example", "com")),  # Same reserved-domain rejection fixture.
            PUBLIC_HOSTS_ONLY,
        ),
        ("http://localhost", PUBLIC_HOSTS_ONLY),
        ("file:///source", ANONYMOUS_ONLY),
        ("https://example.com:444", WEB_PORTS_ONLY),
    ],
)
def test_url_guards_remain_before_any_dns_ipc(monkeypatch, url, reason):
    monkeypatch.setattr(
        module.multiprocessing, "get_context", lambda *_: pytest.fail("IPC before anonymous URL guard")
    )
    with pytest.raises(module.PublicFetchError, match=f"^{re.escape(reason)}$"):
        module._checked_addresses(url, LIVENESS)


@pytest.mark.parametrize("phase", ["poll", "recv", "join", "cleanup"])
def test_absolute_deadline_is_rechecked_after_ready_frame_and_cleanup(monkeypatch, phase):
    from types import SimpleNamespace

    clock = [0.0]
    native_reader, native_writer = multiprocessing.get_context("fork").Pipe(duplex=False)

    class Reader:
        def fileno(self):
            return native_reader.fileno()

        def poll(self, timeout):
            assert timeout > 0
            if phase == "poll":
                clock[0] = 2.0
            return True

        def recv_bytes(self, maximum):
            assert maximum == LIMIT
            if phase == "recv":
                clock[0] = 2.0
            return payload()

        def close(self):
            native_reader.close()

    class Process:
        pid = 123
        exitcode = 0

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout):
            if phase == "join":
                clock[0] = 2.0

        def close(self):
            if phase == "cleanup":
                clock[0] = 2.0

    class Context:
        def Pipe(self, duplex):  # noqa: N802 - exact multiprocessing context API
            assert duplex is False
            return Reader(), native_writer

        def Process(self, *, target, daemon):  # noqa: N802 - exact multiprocessing context API
            assert daemon is True
            return Process()

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda _method: Context())
    with pytest.raises(module.PublicFetchError, match="timeout"):
        module._checked_addresses(URL, 1.0)
    assert native_reader.closed and native_writer.closed


def test_process_start_failure_still_closes_pipe_and_process_handle(monkeypatch):
    context = configured(monkeypatch)
    original = context.Process

    def process(**kwargs):
        value = original(**kwargs)

        def unavailable():
            raise OSError("private-detail fork allocation failure")

        value.start = unavailable
        return value

    monkeypatch.setattr(context, "Process", process)
    with pytest.raises(module.PublicFetchError, match=r"^public URL validation failed$") as error:
        module._checked_addresses(URL, LIVENESS)
    assert "private-detail" not in str(error.value)
    assert_closed(context)
