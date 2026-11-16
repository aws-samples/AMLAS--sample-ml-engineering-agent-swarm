"""Hermetic fixed model transport/commit, synthetic byte streams only."""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any

import model_acquisition_fixtures as f
import pytest


def transport() -> Any:
    return f.require_module("amlas.data_eng.model_public_fetch", "stream_model_catalog_file")


def core() -> Any:
    return f.require_module("amlas.data_eng.model_weight_acquisition", "acquire_model_into_fd")


def test_model_stream_has_fixed_catalog_budget_and_preserves_dataset_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = transport()
    value, _ = f.patch_expected(monkeypatch, [module])
    selected = value["files"][0]
    selected.update(byte_count=1024**3, sha256=f.digest(b""))
    calls = []
    monkeypatch.setattr(module, "_checked_addresses", lambda url, timeout: ["93.184.216.34"])

    def request(url: str, ips: Any, deadline: float, limit: int, sink: Any, sha: Any, blob: Any) -> Any:
        calls.append((url, limit))
        return 200, {}, selected["byte_count"]

    monkeypatch.setattr(module, "_stream_request_public", request)
    result = module.stream_model_catalog_file(
        selected["path"], io.BytesIO(), deadline=time.monotonic() + 7000
    )
    assert result["byte_count"] == 1024**3
    assert calls[0][1] == 1024**3 and "/resolve/c202236235762e1c871ad0ccb60c8ee5ba337b9a/" in calls[0][0]
    from amlas.tools.research.public_fetch import PublicFetchError, stream_public_binary

    with pytest.raises(PublicFetchError):
        stream_public_binary("https://example.org/", io.BytesIO(), max_bytes=512 * 1024**2 + 1)


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "oversize",
        "wrong_hash",
        "wrong_count",
        "expired",
        "downgrade",
        "private_redirect",
        "redirect_loop",
        "http_error",
    ],
)
def test_guarded_model_stream_refuses_before_false_integrity(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    module = transport()
    value, bodies = f.patch_expected(monkeypatch, [module])
    selected = value["files"][0]
    calls = []

    def addresses(url: str, timeout: float) -> list[str]:
        calls.append(url)
        if "169.254.169.254" in url:
            raise ValueError("nonpublic fixture")
        return ["93.184.216.34"]

    monkeypatch.setattr(module, "_checked_addresses", addresses)

    def request(url: str, ips: Any, deadline: float, limit: int, sink: Any, sha: Any, blob: Any) -> Any:
        if case in {"downgrade", "private_redirect", "redirect_loop"}:
            return (
                302,
                {
                    "location": {
                        "downgrade": "http://example.org/file",
                        "private_redirect": "https://169.254.169.254/latest",
                        "redirect_loop": url,
                    }[case]
                },
                0,
            )
        if case == "http_error":
            return 403, {}, 0
        original = bodies[selected["path"]]
        body = b"!" + original[1:] if case == "wrong_hash" else original
        if case == "wrong_hash":
            assert len(body) == len(original) and f.digest(body) != selected["sha256"]
        sink.write(body)
        sha.update(body)
        return 200, {}, len(body) + (1 if case == "wrong_count" else 0)

    monkeypatch.setattr(module, "_stream_request_public", request)
    name = "unknown.py" if case == "unknown" else selected["path"]
    if case == "oversize":
        selected["byte_count"] = 8 * 1024**3 + 1
    deadline = time.monotonic() - 1 if case == "expired" else time.monotonic() + 7200
    with pytest.raises(ValueError):
        module.stream_model_catalog_file(name, io.BytesIO(), deadline=deadline)


def run_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, Path, int, dict[str, Any], dict[str, bytes]]:
    module = core()
    schema = __import__("amlas.data_eng.model_acquisition_schema", fromlist=["x"])
    value, bodies = f.patch_expected(monkeypatch, [module, schema])
    monkeypatch.setattr(
        module,
        "stream_model_catalog_file",
        lambda name, sink, **kwargs: (
            sink.write(bodies[name]),
            {"byte_count": len(bodies[name]), "sha256": f.digest(bodies[name])},
        )[1],
    )
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    return module, output, fd, value, bodies


def test_all_fixed_model_bytes_commit_last_through_retained_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, output, fd, _value, bodies = run_fixture(monkeypatch, tmp_path)
    events = []
    real_open = os.open

    def opened(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        if flags & os.O_CREAT:
            assert kwargs.get("dir_fd") == fd and flags & os.O_EXCL and flags & os.O_NOFOLLOW
            assert Path(str(path)).name == str(path)
            events.append(path)
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(module.os, "open", opened)
    try:
        result = module.acquire_model_into_fd(
            catalog_id=f.CATALOG,
            catalog_version=f.VERSION,
            output_fd=fd,
            output_directory=output,
            bindings=f.bindings(),
            evidence=f.evidence(),
        )
        assert result["acquired"] is True and result["trusted"] is False
        assert events[-1] == "COMMITTED.json"
        assert set(os.listdir(fd)) == set(bodies) | {"LINEAGE.json", "MODEL_ACQUIRED.json", "COMMITTED.json"}
        commit = json.loads((output / "COMMITTED.json").read_bytes())
        assert commit["receipt"]["sha256"] == f.digest((output / "MODEL_ACQUIRED.json").read_bytes())
        for name, raw in bodies.items():
            assert (output / name).read_bytes() == raw
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "case",
    [
        "digest",
        "count",
        "disk",
        "collision",
        "symlink",
        "replace_root",
        "same_inode_payload",
        "same_inode_receipt",
        "post_commit_payload",
        "extra",
    ],
)
def test_model_core_refuses_partial_or_mutated_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    module, output, fd, value, bodies = run_fixture(monkeypatch, tmp_path)
    real_fsync = os.fsync
    mutated = []
    if case == "disk":
        monkeypatch.setattr(
            module.os, "fstatvfs", lambda *args: type("Disk", (), {"f_bavail": 1, "f_frsize": 1})()
        )
    if case == "collision":
        (output / value["files"][0]["path"]).write_bytes(b"owned-existing")
    if case == "symlink":
        (output / value["files"][0]["path"]).symlink_to(tmp_path / "elsewhere")
    if case in {"digest", "count"}:

        def broken(name: str, sink: Any, **kwargs: Any) -> dict[str, Any]:
            sink.write(bodies[name])
            return {
                "byte_count": len(bodies[name]) + (case == "count"),
                "sha256": "c" * 64 if case == "digest" else f.digest(bodies[name]),
            }

        monkeypatch.setattr(module, "stream_model_catalog_file", broken)

    def changed(descriptor: int) -> None:
        if case == "replace_root" and not mutated:
            output.rename(tmp_path / "held-output")
            output.mkdir()
            mutated.append(True)
        if case == "extra" and not mutated:
            (output / "extra.bin").write_bytes(b"unexpected")
            mutated.append(True)
        if case in {"same_inode_payload", "same_inode_receipt", "post_commit_payload"}:
            name = value["files"][0]["path"] if case != "same_inode_receipt" else "MODEL_ACQUIRED.json"
            path = output / name
            ready = case != "post_commit_payload" or (output / "COMMITTED.json").exists()
            if path.exists() and ready and not mutated:
                if case == "post_commit_payload":
                    with path.open("r+b") as stream:
                        stream.write(b"!")
                    mutated.append(True)
                elif os.fstat(descriptor).st_ino == path.stat().st_ino:
                    os.pwrite(descriptor, b"!", 0)
                    mutated.append(True)
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", changed)
    try:
        with pytest.raises((ValueError, OSError)):
            module.acquire_model_into_fd(
                catalog_id=f.CATALOG,
                catalog_version=f.VERSION,
                output_fd=fd,
                output_directory=output,
                bindings=f.bindings(),
                evidence=f.evidence(),
            )
        assert not (output / "COMMITTED.json").exists()
    finally:
        os.close(fd)


def test_commit_fsync_failure_removes_only_the_owned_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, output, fd, _, _ = run_fixture(monkeypatch, tmp_path)
    original = os.fsync

    def failed(descriptor: int) -> None:
        commit = output / "COMMITTED.json"
        if commit.exists() and os.fstat(descriptor).st_ino == commit.stat().st_ino:
            raise OSError("closed commit fsync fixture")
        original(descriptor)

    monkeypatch.setattr(module.os, "fsync", failed)
    try:
        with pytest.raises(OSError):
            module.acquire_model_into_fd(
                catalog_id=f.CATALOG,
                catalog_version=f.VERSION,
                output_fd=fd,
                output_directory=output,
                bindings=f.bindings(),
                evidence=f.evidence(),
            )
        assert not (output / "COMMITTED.json").exists(), "failed commit retained a success marker"
    finally:
        os.close(fd)


def test_model_metadata_depth_is_rejected_before_shared_json_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = __import__("amlas.data_eng.model_acquisition_schema", fromlist=["x"])
    raw = b'{"extra":' + b"[" * 12 + b"0" + b"]" * 12 + b"}"
    monkeypatch.setattr(
        schema,
        "decode_object",
        lambda *args, **kwargs: pytest.fail("model depth guard did not precede decoder"),
    )
    with pytest.raises(ValueError, match="depth"):
        schema.decode_metadata(
            raw, kind="commit", catalog_id=f.CATALOG, catalog_version=f.VERSION, bindings=f.bindings()
        )
