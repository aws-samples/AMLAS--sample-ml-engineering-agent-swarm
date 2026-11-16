"""Real sealed inventory descriptors; earlier absence never means current absence."""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path

import model_acquisition_fixtures as fixtures
import pytest

from amlas.data_eng import model_acquisition_schema as schema
from amlas.sagemaker_entrypoints import model_weight_acquisition as worker


def inventory():
    return {
        "schema_version": 1,
        "inventory_kind": "raw_model_provider_absence_before_drop_v1",
        "observation": "earlier_original_component_absence",
        "providers": [
            {
                "provider_index": index,
                "missing_component_index": 2 if index < 2 else 1,
                "ancestors": [
                    {
                        "component_index": item,
                        "device": 1,
                        "inode": 100 + item,
                        "uid": 0 if item == 0 or index >= 2 else 10001,
                        "gid": 0 if item == 0 or index >= 2 else 10001,
                        "mode": 0o755 if item == 0 or index < 2 else 0o700,
                    }
                    for item in range(2 if index < 2 else 1)
                ],
            }
            for index in range(4)
        ],
    }


@pytest.fixture
def sealed():
    opened = []

    def make(value=None, *, seals=15, readonly=True, raw=None):
        payload = raw if raw is not None else json.dumps(value or inventory(), sort_keys=True).encode()
        fd = os.memfd_create("synthetic-provider-inventory", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
        opened.append(fd)
        os.write(fd, payload)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
        if readonly:
            fd = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
            opened.append(fd)
        return fd, payload

    yield make
    for fd in opened:
        os.close(fd)


def decoder():
    value = getattr(worker, "_read_provider_inventory", None)
    assert callable(value), "missing-symbol: _read_provider_inventory"
    return value


def test_actual_readonly_four_sealed_inventory_bytes_and_digest(sealed):
    fd, raw = sealed()
    result = decoder()(fd)
    assert result == {"inventory": inventory(), "sha256": hashlib.sha256(raw).hexdigest()}
    assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
    assert fcntl.fcntl(fd, fcntl.F_GET_SEALS) == 15
    with pytest.raises(OSError):
        os.write(fd, b"untrusted")


@pytest.mark.parametrize("seals,readonly", [(0, True), (7, True), (15, False)])
def test_unsealed_or_writable_inventory_refuses(sealed, seals, readonly):
    fd, _ = sealed(seals=seals, readonly=readonly)
    with pytest.raises((ValueError, RuntimeError)):
        decoder()(fd)


@pytest.mark.parametrize("change", ["duplicate", "bool", "missing", "extra", "ancestor", "kind"])
def test_inventory_closed_schema_and_original_component_order(sealed, change):
    value = inventory()
    if change == "duplicate":
        value["providers"][1]["provider_index"] = 0
    elif change == "bool":
        value["providers"][0]["missing_component_index"] = True
    elif change == "missing":
        value["providers"].pop()
    elif change == "extra":
        value["ready"] = True
    elif change == "ancestor":
        value["providers"][0]["ancestors"][0]["component_index"] = 1
    else:
        value["inventory_kind"] = "caller-absence-claim"
    fd, _ = sealed(value)
    with pytest.raises((ValueError, RuntimeError)):
        decoder()(fd)


def test_closed_v2_evidence_preserves_denial_and_legacy_is_unchanged():
    value = fixtures.evidence()
    schema.validate_evidence(copy.deepcopy(value))
    raw = json.dumps(inventory(), sort_keys=True).encode()
    value["credential_observation"] = {
        "observation_profile": "model_provider_inventory_v2",
        "environment_names": [],
        "earlier_inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "earlier_inventory": inventory(),
        "provider_path_observations": [
            {"provider_index": index, "current_state": "current_absent" if index < 2 else "denied_after_drop"}
            for index in range(4)
        ],
        "metadata_endpoint_reachable": False,
    }
    assert schema.validate_evidence(value) == value
    forged = copy.deepcopy(value)
    forged["credential_observation"]["provider_path_observations"][2]["current_state"] = "universal_absence"
    with pytest.raises(ValueError):
        schema.validate_evidence(forged)


def test_four_original_probes_execute_and_denied_paths_remain_distinct(monkeypatch, sealed):
    fd, raw = sealed()
    function = getattr(worker, "_credential_observation_v2", None)
    assert callable(function), "missing-symbol: _credential_observation_v2"
    for key in worker._PROVIDER_HINTS:
        monkeypatch.delenv(key, raising=False)
    seen = []

    def observe(path):
        seen.append(str(path))
        if str(path).startswith("/root/"):
            raise PermissionError(errno.EACCES, "synthetic denied boundary")
        raise FileNotFoundError

    monkeypatch.setattr(worker.os, "lstat", observe)
    monkeypatch.setattr(
        worker.socket,
        "create_connection",
        lambda *_args, **_kw: (_ for _ in ()).throw(
            OSError(errno.ENETUNREACH, "synthetic fixed-target denial")
        ),
    )
    result = function(decoder()(fd))
    assert seen == [
        "/home/appuser/.aws/credentials",
        "/home/appuser/.aws/config",
        "/root/.aws/credentials",
        "/root/.aws/config",
    ]
    assert [item["current_state"] for item in result["provider_path_observations"]] == [
        "current_absent",
        "current_absent",
        "denied_after_drop",
        "denied_after_drop",
    ]
    assert result["earlier_inventory_sha256"] == hashlib.sha256(raw).hexdigest()


def test_model_named_walk_uses_path_only_for_original_ancestors(monkeypatch, tmp_path):
    original = os.open
    flags = []

    def observed(path, mode, *args, **kwargs):
        flags.append(mode)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(worker.os, "open", observed)
    fd = worker._open_directory(Path(tmp_path))
    os.close(fd)
    assert flags and all(mode & os.O_PATH for mode in flags)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
        b"[" * 12 + b"0" + b"]" * 12,
        b"x" * 16385,
    ],
)
def test_sealed_bytes_do_not_bypass_json_or_size_limits(sealed, raw):
    fd, _ = sealed(raw=raw)
    with pytest.raises((ValueError, RuntimeError)):
        decoder()(fd)


@pytest.mark.parametrize("failure", [errno.EMFILE, errno.ENOMEM, errno.EIO])
def test_v2_metadata_resource_errors_do_not_become_denial(monkeypatch, sealed, failure):
    fd, _ = sealed()
    for key in worker._PROVIDER_HINTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(worker.os, "lstat", lambda _path: (_ for _ in ()).throw(FileNotFoundError))
    monkeypatch.setattr(
        worker.socket,
        "create_connection",
        lambda *_args, **_kw: (_ for _ in ()).throw(OSError(failure, "synthetic resource failure")),
    )
    with pytest.raises(RuntimeError):
        worker._credential_observation_v2(decoder()(fd))


def test_v2_denial_requires_valid_earlier_inventory(monkeypatch):
    for key in worker._PROVIDER_HINTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        worker.os, "lstat", lambda _path: (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied"))
    )
    monkeypatch.setattr(
        worker.socket,
        "create_connection",
        lambda *_args, **_kw: (_ for _ in ()).throw(OSError(errno.ENETUNREACH, "synthetic denial")),
    )
    with pytest.raises((ValueError, RuntimeError)):
        worker._credential_observation_v2({"inventory": {}, "sha256": "a" * 64})


def test_actual_output_contender_is_readable_and_contends_original_lease(monkeypatch, tmp_path):
    function = getattr(worker, "_open_output_contender", None)
    assert callable(function), "missing-symbol: _open_output_contender"
    monkeypatch.setattr(worker, "_OUTPUT", tmp_path)
    original = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(original, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contender = function()
    try:
        assert os.fstat(original).st_ino == os.fstat(contender).st_ino
        assert not fcntl.fcntl(contender, fcntl.F_GETFL) & os.O_PATH
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(original, fcntl.LOCK_UN)
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(contender)
        os.close(original)
