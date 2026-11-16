"""Real memfd provider inventory; observations are bounded and never authority."""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os

import pytest
from test_model_provider_inventory import inventory

from amlas.sagemaker_entrypoints import qwen_sft_train as worker


@pytest.fixture
def fixed_inventory():
    try:
        prior = os.dup(4)
    except OSError as error:
        assert error.errno == errno.EBADF
        prior = None
    opened = []

    def make(value=None, *, seals=15, readonly=True, raw=None):
        payload = (
            raw
            if raw is not None
            else json.dumps(inventory() if value is None else value, sort_keys=True).encode()
        )
        fd = os.memfd_create("synthetic-qwen-provider-inventory", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
        opened.append(fd)
        os.write(fd, payload)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
        read = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC) if readonly else fd
        if read != fd:
            opened.append(read)
        os.dup2(read, 4)
        return payload

    yield make
    if prior is None:
        try:
            os.close(4)
        except OSError:
            pass
    else:
        os.dup2(prior, 4)
        os.close(prior)
    for fd in opened:
        if fd != 4:
            try:
                os.close(fd)
            except OSError:
                pass


def observation():
    value = getattr(worker, "_provider_observation", None)
    assert callable(value), "missing-symbol: _provider_observation"
    return value


def absent(monkeypatch):
    for key in worker.FORBIDDEN:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        worker.os,
        "lstat",
        lambda path: (_ for _ in ()).throw(FileNotFoundError(errno.ENOENT, "closed fixture absent")),
    )
    monkeypatch.setattr(
        worker.socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError(errno.ENETUNREACH, "closed fixed-target denial")
        ),
    )


def test_actual_readonly_sealed_fd4_digest_and_four_states(monkeypatch, fixed_inventory, capsys):
    raw = fixed_inventory()
    absent(monkeypatch)
    result = observation()("before_core_import")
    assert result["inventory_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["fd_identity"] == [os.fstat(4).st_dev, os.fstat(4).st_ino, os.fstat(4).st_size]
    assert result["provider_path_observations"] == [
        {"provider_index": i, "current_state": "current_absent"} for i in range(4)
    ]
    logged = json.loads(capsys.readouterr().out)
    assert (
        logged["guest_claim_attested"] is False and logged["inventory_sha256"] == result["inventory_sha256"]
    )
    assert len(json.dumps(logged).encode()) <= 1024
    with pytest.raises(OSError):
        os.write(4, b"forged")


@pytest.mark.parametrize("seals,readonly", [(0, True), (7, True), (15, False)])
def test_unsealed_or_writable_fd4_never_qualifies(monkeypatch, fixed_inventory, seals, readonly):
    fixed_inventory(seals=seals, readonly=readonly)
    absent(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        observation()("before_core_import")


@pytest.mark.parametrize(
    "case", ["extra", "bool", "kind", "order", "ancestor", "duplicate_json", "nonfinite", "depth", "oversize"]
)
def test_closed_inventory_schema_refuses(monkeypatch, fixed_inventory, case):
    value = inventory()
    raw = None
    if case == "extra":
        value["ready"] = True
    elif case == "bool":
        value["providers"][0]["missing_component_index"] = True
    elif case == "kind":
        value["inventory_kind"] = "caller-absence"
    elif case == "order":
        value["providers"][1]["provider_index"] = 0
    elif case == "ancestor":
        value["providers"][0]["ancestors"][0]["component_index"] = 1
    elif case == "duplicate_json":
        raw = b'{"schema_version":1,"schema_version":1}'
    elif case == "nonfinite":
        raw = b'{"schema_version":NaN}'
    elif case == "depth":
        raw = b"[" * 20 + b"0" + b"]" * 20
    else:
        raw = b"x" * 16385
    fixed_inventory(value, raw=raw)
    absent(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        observation()("before_core_import")


@pytest.mark.parametrize("failure", [errno.EIO, errno.ENOMEM, errno.EMFILE, errno.ENOTDIR])
def test_unknown_provider_errors_are_not_absence(monkeypatch, fixed_inventory, failure):
    fixed_inventory()
    absent(monkeypatch)
    monkeypatch.setattr(worker.os, "lstat", lambda path: (_ for _ in ()).throw(OSError(failure, "unknown")))
    with pytest.raises((ValueError, RuntimeError)):
        observation()("before_core_import")


def test_denied_after_drop_is_distinct_from_absent(monkeypatch, fixed_inventory):
    fixed_inventory()
    absent(monkeypatch)
    seen = []

    def stat(path):
        seen.append(str(path))
        if str(path).startswith("/root/"):
            raise PermissionError(errno.EACCES, "denied")
        raise FileNotFoundError(errno.ENOENT, "absent")

    monkeypatch.setattr(worker.os, "lstat", stat)
    result = observation()("before_core_import")
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


def test_present_provider_refuses_without_echo(monkeypatch, fixed_inventory):
    fixed_inventory()
    absent(monkeypatch)
    monkeypatch.setattr(worker.os, "lstat", lambda path: object())
    with pytest.raises((ValueError, RuntimeError)):
        observation()("before_core_import")


@pytest.mark.parametrize("hint", ["AWS_PROFILE", "AWS_EC2_METADATA_DISABLED", "HTTPS_PROXY"])
def test_provider_hint_presence_is_not_cleared(monkeypatch, fixed_inventory, hint):
    fixed_inventory()
    absent(monkeypatch)
    monkeypatch.setenv(hint, "synthetic-sensitive-value")
    with pytest.raises((ValueError, RuntimeError)):
        observation()("before_core_import")
    assert os.environ[hint] == "synthetic-sensitive-value"


@pytest.mark.parametrize("failure", [errno.EIO, errno.ENOMEM, errno.EMFILE])
def test_unknown_network_errors_do_not_become_denial(monkeypatch, fixed_inventory, failure):
    fixed_inventory()
    absent(monkeypatch)
    monkeypatch.setattr(
        worker.socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError(failure, "resource failure")),
    )
    with pytest.raises((ValueError, RuntimeError)):
        observation()("before_core_import")


def test_public_endpoint_unexpected_error_refuses(monkeypatch, fixed_inventory):
    fixed_inventory()
    absent(monkeypatch)

    def connect(address, **kwargs):
        if address[0] == "1.1.1.1":
            raise OSError(errno.EIO, "public resource error")
        raise OSError(errno.ENETUNREACH, "metadata denial")

    monkeypatch.setattr(worker.socket, "create_connection", connect)
    with pytest.raises((ValueError, RuntimeError)):
        observation()("before_core_import")


def test_post_import_changed_inventory_or_states_refuses():
    function = getattr(worker, "_recheck_provider_observation", None)
    assert callable(function), "missing-symbol: _recheck_provider_observation"
    original = {
        "inventory_sha256": "a" * 64,
        "fd_identity": [1, 2, 3],
        "provider_path_observations": [
            {"provider_index": i, "current_state": "current_absent"} for i in range(4)
        ],
    }
    for key in original:
        changed = copy.deepcopy(original)
        changed[key] = "changed"
        with pytest.raises(ValueError):
            function(original, changed)
    assert function(original, copy.deepcopy(original)) is None


@pytest.mark.parametrize("change", ["unchanged", "environment", "states", "same_bytes_new_fd", "present"])
def test_main_rechecks_actual_inventory_and_current_states_before_core_operation(
    monkeypatch, fixed_inventory, tmp_path, change
):
    import sys
    from types import SimpleNamespace

    fixed_inventory()
    absent(monkeypatch)
    binding = {"schema_version": 1, "catalog_id": "qwen3_5_9b", "catalog_version": "1", "recipe_id": "fit_v1"}
    raw = json.dumps(binding).encode()
    path = tmp_path / "RUN.json"
    path.write_bytes(raw)
    monkeypatch.setattr(worker, "BINDING", path)
    # Isolate provider recheck from independently tested output/native identity gates.
    monkeypatch.setattr(worker, "_guard", lambda: None)
    monkeypatch.setattr(worker, "_guard_output", lambda: None)
    calls = []

    def load():
        if change == "environment":
            monkeypatch.setenv("AWS_PROFILE", "sensitive-fixture-value")
        elif change == "states":
            monkeypatch.setattr(
                worker.os,
                "lstat",
                lambda path: (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied")),
            )
        elif change == "same_bytes_new_fd":
            fixed_inventory()
        elif change == "present":
            monkeypatch.setattr(worker.os, "lstat", lambda path: object())
        return SimpleNamespace(managed_operation=lambda *args, **kwargs: calls.append((args, kwargs)))

    monkeypatch.setattr(worker, "_load_core", load)
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen_sft_train.py",
            "--operation",
            "train",
            "--recipe-id",
            "fit_v1",
            "--recipe-version",
            "1",
            "--source-commit",
            "a" * 40,
            "--manifest-sha256",
            hashlib.sha256(raw).hexdigest(),
            "--manifest-byte-count",
            str(len(raw)),
        ],
    )
    if change == "unchanged":
        worker.main()
        assert len(calls) == 1
    else:
        with pytest.raises((ValueError, RuntimeError)):
            worker.main()
        assert calls == []


def test_sealed_descriptor_swap_during_probes_is_refused(monkeypatch, fixed_inventory):
    fixed_inventory()
    absent(monkeypatch)

    def connect(address, **kwargs):
        if address[0] == "1.1.1.1":
            fixed_inventory()
        raise OSError(errno.ENETUNREACH, "fixed-target denied")

    monkeypatch.setattr(worker.socket, "create_connection", connect)
    with pytest.raises(ValueError, match="descriptor changed"):
        observation()("before_core_import")


def test_same_memfd_writable_rebind_at_probe_completion_refuses(monkeypatch, fixed_inventory):
    fixed_inventory()
    absent(monkeypatch)
    original = os.fstat(4)

    def connect(address, **kwargs):
        if address[0] == "1.1.1.1":
            writer = os.open("/proc/self/fd/4", os.O_RDWR | os.O_CLOEXEC)
            try:
                os.dup2(writer, 4)
            finally:
                os.close(writer)
        raise OSError(errno.ENETUNREACH, "fixed target denial")

    monkeypatch.setattr(worker.socket, "create_connection", connect)
    with pytest.raises((ValueError, RuntimeError)):
        observation()("after_core_import")
    assert (original.st_dev, original.st_ino, original.st_size) == (
        os.fstat(4).st_dev,
        os.fstat(4).st_ino,
        os.fstat(4).st_size,
    )
    assert fcntl.fcntl(4, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDWR
    assert fcntl.fcntl(4, fcntl.F_GET_SEALS) == 15
