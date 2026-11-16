"""The CPU test provisioning must supply the one pinned public evaluation file, verified, or stop."""

import contextlib
import hashlib
import http.server
import importlib.util
import io
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/qwen_cpu_test_case_file.py"
SCRIPT = "scripts/provision_qwen_cpu_tests.sh"
NAME = "AMLAS_TEST_HUMANEVAL_PARQUET"
OFFLINE_LINE = "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1"
STAND_IN = b"stand-in public case rows\n" * 37
PROXY_NAMES = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
NO_PROXY_NAMES = ("no_proxy", "NO_PROXY")
SOURCE_ADDRESS = "https://source.invalid/data/file.bin"
BASH = shutil.which("bash")


def helper():
    path = ROOT / HELPER
    assert path.is_file(), "missing-symbol: qwen_cpu_test_case_file"
    spec = importlib.util.spec_from_file_location("qwen_cpu_test_case_file", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stand_in_pins(m, raw=STAND_IN, *, sha256=None, byte_count=None):
    return m.CasePins(
        repo="example/stand-in",
        revision="a" * 40,
        filename="data/stand-in.bin",
        sha256=hashlib.sha256(raw).hexdigest() if sha256 is None else sha256,
        byte_count=len(raw) if byte_count is None else byte_count,
    )


def no_request(*_args, **_kwargs):
    raise AssertionError("a request was made where none is allowed")


def tree(directory):
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*"))


@contextlib.contextmanager
def local_server(responses):
    """Serve the listed (status, headers, body) answers in order; the last one repeats."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            status, headers, body = responses[min(len(seen), len(responses) - 1)]
            seen.append(self.path)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture
def direct(monkeypatch):
    for name in PROXY_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_helper_reads_the_product_source_constants():
    from amlas.data_eng import coding_case_profile as product

    m = helper()
    pins = m.product_pins(ROOT)
    assert tuple(pins) == (
        product.SOURCE_REPO,
        product.SOURCE_REVISION,
        product.SOURCE_FILENAME,
        product.SOURCE_SHA256,
        product.SOURCE_BYTES,
    )
    assert m.source_url(pins) == (
        f"https://huggingface.co/datasets/{product.SOURCE_REPO}/resolve/"
        f"{product.SOURCE_REVISION}/{product.SOURCE_FILENAME}"
    )
    assert m.ENVIRONMENT_NAME == NAME


@pytest.mark.parametrize(
    "text",
    [
        "SOURCE_REPO = 'example/stand-in'\n",
        "SOURCE_REPO = 'example/stand-in'\nSOURCE_REVISION = 'a' * 40\nSOURCE_FILENAME = 'f'\n"
        "SOURCE_SHA256 = 'b' * 64\nSOURCE_BYTES = 3\n",
        "SOURCE_REPO = 'example/stand-in'\nSOURCE_REPO = 'example/other'\n",
        "SOURCE_REPO = '../x'\nSOURCE_REVISION = '" + "a" * 40 + "'\nSOURCE_FILENAME = 'f'\n"
        "SOURCE_SHA256 = '" + "b" * 64 + "'\nSOURCE_BYTES = 3\n",
        "SOURCE_REPO = 'example/stand-in'\nSOURCE_REVISION = '" + "a" * 40 + "'\nSOURCE_FILENAME = '../f'\n"
        "SOURCE_SHA256 = '" + "b" * 64 + "'\nSOURCE_BYTES = 3\n",
        "SOURCE_REPO = 'example/stand-in'\nSOURCE_REVISION = '" + "a" * 40 + "'\nSOURCE_FILENAME = 'f'\n"
        "SOURCE_SHA256 = '" + "b" * 64 + "'\nSOURCE_BYTES = True\n",
    ],
    ids=["missing", "not-literal", "duplicate", "repo-shape", "filename-shape", "count-type"],
)
def test_product_constants_must_be_complete_single_plain_literals(tmp_path, text):
    m = helper()
    profile = tmp_path / m.PROFILE_RELATIVE
    profile.parent.mkdir(parents=True)
    profile.write_text(text)
    with pytest.raises(m.CaseFileError, match="source constants"):
        m.product_pins(tmp_path)


def test_right_bytes_from_a_local_file_url_are_written_read_only(tmp_path):
    m = helper()
    source = tmp_path / "source.bin"
    source.write_bytes(STAND_IN)
    destination = tmp_path / "out" / "case.bin"
    pins = stand_in_pins(m)
    result = m.provide(destination, pins, url=source.as_uri(), schemes=("file",), pause=no_request)
    assert result == {
        "path": str(destination),
        "byte_count": len(STAND_IN),
        "sha256": pins.sha256,
        "source": "fetched",
    }
    assert destination.read_bytes() == STAND_IN
    assert destination.stat().st_mode & 0o222 == 0
    assert tree(tmp_path) == ["out", "out/case.bin", "source.bin"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sha256": hashlib.sha256(STAND_IN + b"x").hexdigest()}, "sha256"),
        ({"byte_count": len(STAND_IN) - 1}, "size"),
        ({"byte_count": len(STAND_IN) + 1}, "size"),
    ],
    ids=["wrong-hash", "source-longer", "source-shorter"],
)
def test_wrong_hash_or_size_refuses_and_leaves_nothing(tmp_path, change, message):
    m = helper()
    source = tmp_path / "source.bin"
    source.write_bytes(STAND_IN)
    destination = tmp_path / "out" / "case.bin"
    with pytest.raises(m.CaseFileError, match=message):
        m.provide(
            destination, stand_in_pins(m, **change), url=source.as_uri(), schemes=("file",), pause=no_request
        )
    assert tree(tmp_path) == ["source.bin"]


def test_missing_source_refuses_after_the_stated_attempts_and_leaves_nothing(tmp_path):
    m = helper()
    pauses = []
    with pytest.raises(m.CaseFileError, match="fetch failed after 3 attempts"):
        m.provide(
            tmp_path / "out" / "case.bin",
            stand_in_pins(m),
            url=(tmp_path / "absent.bin").as_uri(),
            schemes=("file",),
            pause=pauses.append,
        )
    assert len(pauses) == 2 and all(value > 0 for value in pauses)
    assert tree(tmp_path) == []


@pytest.mark.parametrize("scheme", ["file", "http", "ftp"])
def test_only_https_is_requested_by_default(tmp_path, monkeypatch, scheme):
    m = helper()
    monkeypatch.setattr(m.urllib.request, "build_opener", no_request)
    with pytest.raises(m.CaseFileError, match="scheme"):
        m.provide(
            tmp_path / "out" / "case.bin", stand_in_pins(m), url=f"{scheme}://127.0.0.1/x", pause=no_request
        )
    assert tree(tmp_path) == []


def test_relative_destination_refuses(tmp_path):
    m = helper()
    with pytest.raises(m.CaseFileError, match="absolute"):
        m.provide(Path("case.bin"), stand_in_pins(m), existing=str(tmp_path / "x"))


def test_local_server_bytes_pass_and_a_same_scheme_redirect_is_followed(tmp_path, direct):
    m = helper()
    answers = [(302, {"Location": "/final"}, b""), (200, {}, STAND_IN)]
    with local_server(answers) as (base, seen):
        result = m.provide(
            tmp_path / "out" / "case.bin",
            stand_in_pins(m),
            url=f"{base}/first",
            schemes=("http",),
            pause=no_request,
        )
    assert seen == ["/first", "/final"]
    assert result["source"] == "fetched"
    assert (tmp_path / "out" / "case.bin").read_bytes() == STAND_IN


def test_redirect_to_a_scheme_outside_the_allowed_set_refuses(tmp_path, direct):
    m = helper()
    with local_server([(302, {"Location": "https://127.0.0.1:1/elsewhere"}, b"")]) as (base, seen):
        with pytest.raises(m.CaseFileError, match="redirect"):
            m.provide(
                tmp_path / "out" / "case.bin",
                stand_in_pins(m),
                url=f"{base}/first",
                schemes=("http",),
                pause=no_request,
            )
    assert seen == ["/first"]
    assert tree(tmp_path) == []


def test_temporary_server_fault_is_tried_again_once_it_recovers(tmp_path, direct):
    m = helper()
    pauses = []
    with local_server([(503, {}, b"busy"), (200, {}, STAND_IN)]) as (base, seen):
        result = m.provide(
            tmp_path / "out" / "case.bin",
            stand_in_pins(m),
            url=f"{base}/file",
            schemes=("http",),
            pause=pauses.append,
        )
    assert len(seen) == 2 and len(pauses) == 1
    assert result["source"] == "fetched"


def test_absent_remote_file_refuses_at_once(tmp_path, direct):
    m = helper()
    with local_server([(404, {}, b"absent")]) as (base, seen):
        with pytest.raises(m.CaseFileError, match="HTTP 404"):
            m.provide(
                tmp_path / "out" / "case.bin",
                stand_in_pins(m),
                url=f"{base}/file",
                schemes=("http",),
                pause=no_request,
            )
    assert seen == ["/file"]
    assert tree(tmp_path) == []


def test_oversized_remote_body_is_read_to_one_byte_past_the_pin_and_refused(tmp_path, direct):
    m = helper()
    reads = []
    original = m._read_bounded

    def recording(stream, limit):
        raw = original(stream, limit)
        reads.append((limit, len(raw)))
        return raw

    m._read_bounded = recording
    with local_server([(200, {}, STAND_IN + b"y" * 300000)]) as (base, _seen):
        with pytest.raises(m.CaseFileError, match="size"):
            m.provide(
                tmp_path / "out" / "case.bin",
                stand_in_pins(m),
                url=f"{base}/file",
                schemes=("http",),
                pause=no_request,
            )
    assert reads == [(len(STAND_IN), len(STAND_IN) + 1)]
    assert tree(tmp_path) == []


class StandInAnswer:
    """What an opener returns: a readable body and the address it finally came from."""

    def __init__(self, body, address):
        self._stream = io.BytesIO(body)
        self._address = address

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return self._address

    def read(self, size=-1):
        return self._stream.read(size)


@contextlib.contextmanager
def refusing_proxy():
    """A stand-in forward proxy: it records each first request line and answers 503 without opening a tunnel."""
    seen = []

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            seen.append(self.rfile.readline(65537).decode("latin-1").strip())
            while self.rfile.readline(65537) not in (b"\r\n", b"\n", b""):
                pass
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_each_attempt_sends_a_new_request_for_the_source_address(tmp_path, monkeypatch):
    """The standard library's proxy handler rewrites the request object it sends; a second send of it goes elsewhere."""
    m = helper()
    sent = []

    class Opener:
        def open(self, request, timeout=None):
            sent.append((request, request.type, request.host, request.selector, request.full_url))
            request.set_proxy("proxy.invalid:3128", "http")
            if len(sent) < 3:
                raise urllib.error.URLError("stand-in transport fault")
            return StandInAnswer(STAND_IN, "https://storage.invalid/object")

    monkeypatch.setattr(m.urllib.request, "build_opener", lambda *_handlers: Opener())
    result = m.provide(
        tmp_path / "out" / "case.bin", stand_in_pins(m), url=SOURCE_ADDRESS, pause=lambda _seconds: None
    )
    assert result["source"] == "fetched"
    assert (tmp_path / "out" / "case.bin").read_bytes() == STAND_IN
    assert len({id(entry[0]) for entry in sent}) == 3
    assert [entry[1:] for entry in sent] == [
        ("https", "source.invalid", "/data/file.bin", SOURCE_ADDRESS)
    ] * 3


def test_every_attempt_behind_a_forward_proxy_asks_for_the_tls_port_of_the_source_host(tmp_path, monkeypatch):
    """Sent a second and third time, one request object left for port 80 as plain HTTP."""
    m = helper()
    for name in PROXY_NAMES + NO_PROXY_NAMES:
        monkeypatch.delenv(name, raising=False)
    with refusing_proxy() as (address, seen):
        monkeypatch.setenv("https_proxy", address)
        monkeypatch.setenv("http_proxy", address)
        with pytest.raises(m.CaseFileError, match="fetch failed after 3 attempts"):
            m.provide(
                tmp_path / "out" / "case.bin",
                stand_in_pins(m),
                url=SOURCE_ADDRESS,
                timeout=10.0,
                pause=lambda _seconds: None,
            )
    assert [line.rsplit(" ", 1)[0] for line in seen] == ["CONNECT source.invalid:443"] * 3
    assert tree(tmp_path) == []


@pytest.mark.parametrize(
    "change",
    [
        lambda request: request.set_proxy("proxy.invalid:3128", "http"),
        lambda request: [request.set_proxy("proxy.invalid:3128", "http") for _ in range(2)],
        lambda request: setattr(request, "full_url", "http://source.invalid/data/file.bin"),
        lambda request: setattr(request, "full_url", "https://other.invalid/data/file.bin"),
    ],
    ids=["sent-once-through-a-proxy", "sent-twice-through-a-proxy", "plain-http", "another-host"],
)
def test_a_request_that_no_longer_names_the_source_address_is_refused(change):
    m = helper()
    request = urllib.request.Request(SOURCE_ADDRESS)
    m._require_source_target(request, SOURCE_ADDRESS, ("https",))
    change(request)
    with pytest.raises(m.CaseFileError, match="source host"):
        m._require_source_target(request, SOURCE_ADDRESS, ("https",))


@pytest.mark.parametrize(
    "final", ["http://source.invalid/data/file.bin", "ftp://source.invalid/data/file.bin"]
)
def test_an_answer_from_a_final_address_outside_the_allowed_schemes_refuses_and_leaves_nothing(
    tmp_path, monkeypatch, final
):
    m = helper()
    opened = []

    class Opener:
        def open(self, request, timeout=None):
            opened.append(request.full_url)
            return StandInAnswer(STAND_IN, final)

    monkeypatch.setattr(m.urllib.request, "build_opener", lambda *_handlers: Opener())
    with pytest.raises(m.CaseFileError, match="final address"):
        m.provide(tmp_path / "out" / "case.bin", stand_in_pins(m), url=SOURCE_ADDRESS, pause=no_request)
    assert opened == [SOURCE_ADDRESS]
    assert tree(tmp_path) == []


def test_named_verified_file_is_reused_without_any_request(tmp_path, monkeypatch):
    m = helper()
    named = tmp_path / "named.bin"
    named.write_bytes(STAND_IN)
    monkeypatch.setattr(m, "_fetch", no_request)
    destination = tmp_path / "out" / "case.bin"
    result = m.provide(destination, stand_in_pins(m), existing=str(named))
    assert result["source"] == "reused" and result["path"] == str(destination)
    assert destination.read_bytes() == STAND_IN
    assert named.read_bytes() == STAND_IN


@pytest.mark.parametrize(
    "content", [STAND_IN + b"x", STAND_IN[:-1], None], ids=["longer", "shorter", "absent"]
)
def test_named_file_that_differs_refuses_without_request_or_write(tmp_path, monkeypatch, content):
    m = helper()
    named = tmp_path / "named.bin"
    if content is not None:
        named.write_bytes(content)
    before = tree(tmp_path)
    monkeypatch.setattr(m, "_fetch", no_request)
    with pytest.raises(m.CaseFileError, match=NAME):
        m.provide(tmp_path / "out" / "case.bin", stand_in_pins(m), existing=str(named))
    assert tree(tmp_path) == before


def test_present_destination_is_kept_when_verified_and_never_overwritten(tmp_path, monkeypatch):
    m = helper()
    monkeypatch.setattr(m, "_fetch", no_request)
    destination = tmp_path / "out" / "case.bin"
    destination.parent.mkdir()
    destination.write_bytes(STAND_IN)
    assert m.provide(destination, stand_in_pins(m))["source"] == "present"
    destination.write_bytes(b"foreign")
    with pytest.raises(m.CaseFileError, match="destination"):
        m.provide(destination, stand_in_pins(m))
    assert destination.read_bytes() == b"foreign"


def test_file_named_for_this_test_run_passes_with_the_product_pins(tmp_path, monkeypatch):
    from amlas.data_eng import coding_case_profile as product

    m = helper()
    named = os.environ.get(NAME)
    assert named, f"{NAME} must name the provisioned public file for the complete suite"
    monkeypatch.setattr(m, "_fetch", no_request)
    destination = tmp_path / "copy" / "case.parquet"
    result = m.provide(destination, m.product_pins(ROOT), existing=named)
    raw = destination.read_bytes()
    assert result["source"] == "reused"
    assert len(raw) == product.SOURCE_BYTES and hashlib.sha256(raw).hexdigest() == product.SOURCE_SHA256


def stand_in_project(tmp_path):
    """The real script and helper beside stand-in constants, lock, metadata step and uv."""
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "src/amlas/data_eng").mkdir(parents=True)
    for relative in (SCRIPT, HELPER):
        assert (ROOT / relative).is_file(), f"missing-symbol: {relative}"
        shutil.copyfile(ROOT / relative, project / relative)
    (project / "scripts/qwen_cpu_test.requirements.lock").write_text("")
    (project / "scripts/qwen_cpu_test_metadata.py").write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[sys.argv.index('--destination') + 1]).mkdir()\n"
        "print('{}')\n"
    )
    (project / "src/amlas/data_eng/coding_case_profile.py").write_text(
        "SOURCE_REPO = 'example/stand-in'\n"
        f"SOURCE_REVISION = '{'a' * 40}'\n"
        "SOURCE_FILENAME = 'data/stand-in.bin'\n"
        f"SOURCE_SHA256 = '{hashlib.sha256(STAND_IN).hexdigest()}'\n"
        f"SOURCE_BYTES = {len(STAND_IN)}\n"
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        "# offline stand-in: 'pip install --target DIR' makes DIR; 'run [flags] python ...' runs this interpreter\n"
        'if [ "$1" = pip ]; then\n'
        '  while [ "$#" -gt 1 ]; do if [ "$1" = --target ]; then mkdir -p "$2"; fi; shift; done\n'
        "  exit 0\n"
        "fi\n"
        '[ "$1" = run ] || exit 64\n'
        "shift\n"
        'while [ "${1#--}" != "$1" ]; do shift; done\n'
        '[ "$1" = python ] || exit 64\n'
        "shift\n"
        f'exec "{sys.executable}" -B "$@"\n'
    )
    uv.chmod(0o755)
    return project.resolve(), tools


def run_script(tmp_path, project, tools, extra):
    assert BASH, "bash is required for the provisioning script"
    output = tmp_path.resolve() / "provisioned"
    environment = {
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
    }
    environment.update(extra)
    done = subprocess.run(
        [BASH, str(project / SCRIPT), str(output)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return done, output


def test_script_exports_the_verified_file_before_the_offline_line(tmp_path):
    project, tools = stand_in_project(tmp_path)
    named = tmp_path / "named.bin"
    named.write_bytes(STAND_IN)
    done, output = run_script(tmp_path, project, tools, {NAME: str(named)})
    assert done.returncode == 0, done.stderr
    lines = (output / "environment.sh").read_text().splitlines()
    exported = [line for line in lines if line.startswith(f"export {NAME}=")]
    assert len(exported) == 1
    case_file = Path(exported[0].split("=", 1)[1])
    assert case_file.is_absolute() and output in case_file.parents and case_file != named
    assert case_file.read_bytes() == STAND_IN
    assert lines == [
        f"export PYTHONPATH={output}/site:{project}/src",
        f"export AMLAS_QWEN_TEST_METADATA={output}/metadata",
        exported[0],
        OFFLINE_LINE,
    ]
    sourced = subprocess.run(
        [
            BASH,
            "-c",
            'source "$1" && printf "%s|%s|%s" "$AMLAS_TEST_HUMANEVAL_PARQUET" "$HF_HUB_OFFLINE" "$PYTHONPATH"',
            "sh",
            str(output / "environment.sh"),
        ],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert sourced.stdout == f"{case_file}|1|{output}/site:{project}/src"
    assert '"source": "reused"' in done.stdout


def test_script_stops_without_environment_file_when_the_named_file_differs(tmp_path):
    project, tools = stand_in_project(tmp_path)
    named = tmp_path / "named.bin"
    named.write_bytes(STAND_IN + b"x")
    done, output = run_script(tmp_path, project, tools, {NAME: str(named)})
    assert done.returncode != 0
    assert "public case file not provided" in done.stderr and NAME in done.stderr
    assert not (output / "environment.sh").exists()
    assert not [path for path in output.rglob("*") if path.is_file() and path.suffix in (".bin", ".parquet")]


def test_script_stops_without_environment_file_when_the_fetch_fails(tmp_path):
    project, tools = stand_in_project(tmp_path)
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        unreachable = f"http://127.0.0.1:{closed.getsockname()[1]}"
    extra = dict.fromkeys(PROXY_NAMES, unreachable)
    extra.update({"no_proxy": "", "NO_PROXY": ""})
    done, output = run_script(tmp_path, project, tools, extra)
    assert done.returncode != 0
    assert "public case file not provided" in done.stderr and "fetch failed after 3 attempts" in done.stderr
    assert not (output / "environment.sh").exists()
    assert not [path for path in output.rglob("*") if path.is_file() and path.suffix in (".bin", ".parquet")]
    assert not (output / "humaneval").exists()
