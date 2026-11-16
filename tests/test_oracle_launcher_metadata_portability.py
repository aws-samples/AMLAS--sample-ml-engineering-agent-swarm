"""Neutral metadata portability; no launcher/confinement/candidate execution."""

from __future__ import annotations

import hashlib
import os
import stat
from types import SimpleNamespace

import pytest
import test_oracle_launcher_binding as binding


@pytest.mark.parametrize("mode", [0o755, 0o775, 0o777])
def test_declared_closed_positive_fixture_is_independent_of_real_temp_ancestor_mode(
    tmp_path, monkeypatch, mode
):
    ancestor = tmp_path / "controlled-ancestor"
    ancestor.mkdir()
    ancestor.chmod(mode)
    original = os.fstat
    with monkeypatch.context() as scoped:
        binding.test_fixed_installed_binary_is_hashed_with_closed_owner_metadata_and_expected_digest(
            ancestor,
            scoped,
        )
    assert os.fstat is original
    assert ancestor.stat().st_mode & 0o777 == mode


@pytest.mark.parametrize(
    "case",
    [
        "group-write",
        "world-write",
        "all-write",
        "ancestor-owner",
        "ancestor-group",
        "file-owner",
        "file-group",
        "file-write",
        "not-executable",
        "file-symlink",
        "ancestor-symlink",
        "wrong-digest",
    ],
)
def test_independent_observed_ownership_modes_and_links_still_refuse(tmp_path, monkeypatch, case):
    ancestor = tmp_path / "observed-ancestor"
    ancestor.mkdir()
    ancestor.chmod({"group-write": 0o775, "world-write": 0o757, "all-write": 0o777}.get(case, 0o755))
    target = ancestor / "amlas-oracle-init"
    body = b"neutral metadata fixture"
    target.write_bytes(body)
    target.chmod(0o555)
    original = os.fstat
    parent_identity = (ancestor.stat().st_dev, ancestor.stat().st_ino)
    file_identity = (target.stat().st_dev, target.stat().st_ino)

    def observed(fd):
        actual = original(fd)
        identity = (actual.st_dev, actual.st_ino)
        values = {name: getattr(actual, name) for name in dir(actual) if name.startswith("st_")}
        values.update(st_uid=0, st_gid=0)
        if stat.S_ISDIR(actual.st_mode):
            # Close unrelated harness ancestors; selected actual bad mode stays real.
            if identity != parent_identity:
                values["st_mode"] &= ~0o022
            if identity == parent_identity and case == "ancestor-owner":
                values["st_uid"] = 10001
            if identity == parent_identity and case == "ancestor-group":
                values["st_gid"] = 10001
        if identity == file_identity:
            if case == "file-owner":
                values["st_uid"] = 10001
            if case == "file-group":
                values["st_gid"] = 10001
        return SimpleNamespace(**values)

    if case == "file-write":
        target.chmod(0o575)
    elif case == "not-executable":
        target.chmod(0o444)
    selected = target
    if case == "file-symlink":
        selected = ancestor / "linked-file"
        selected.symlink_to(target.name)
    elif case == "ancestor-symlink":
        linked = tmp_path / "linked-ancestor"
        linked.symlink_to(ancestor.name)
        selected = linked / target.name
    expected = "f" * 64 if case == "wrong-digest" else hashlib.sha256(body).hexdigest()
    with monkeypatch.context() as scoped:
        scoped.setattr(binding.runner, "ORACLE_LAUNCHER", str(selected))
        scoped.setattr(os, "fstat", observed)
        with pytest.raises((ValueError, OSError)):
            binding.runner.verify_oracle_launcher(expected)
    assert os.fstat is original
    assert ancestor.stat().st_mode & 0o777 == {
        "group-write": 0o775,
        "world-write": 0o757,
        "all-write": 0o777,
    }.get(case, 0o755)
