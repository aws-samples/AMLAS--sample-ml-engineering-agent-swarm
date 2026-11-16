"""Source-owned native profile metadata only; no cloud or data operations."""

import importlib.util
from pathlib import Path

import pytest


def accepted(function):
    try:
        result = function()
    except ValueError:
        result = None
    assert result is not None, "valid measured identity unexpectedly refused"
    return result


@pytest.fixture
def guard(monkeypatch):
    p = Path(__file__).parents[1] / "processing_bootstrap/processing_identity.py"
    spec = importlib.util.spec_from_file_location("closed_processing_identity", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.os, "getresuid", lambda: (10001, 10001, 10001))
    monkeypatch.setattr(module.os, "getresgid", lambda: (10001, 10001, 10001))
    monkeypatch.setattr(module.os, "getgroups", lambda: [])
    status = {
        "Uid": "10001 10001 10001 10001",
        "Gid": "10001 10001 10001 10001",
        "CapEff": "0",
        "CapPrm": "0",
        "CapInh": "0",
        "CapAmb": "0",
        "CapBnd": "200404fb",
        "NoNewPrivs": "1",
        "Seccomp": "2",
    }
    monkeypatch.setattr(module, "_read_status", lambda: dict(status))
    monkeypatch.setattr(module, "_prctl", lambda op: 0)
    return module, status


def test_native_exact14_observed_fields(guard):
    module, _ = guard
    proof = accepted(module.verify_native_identity)
    assert proof == {
        "res_uid": [10001] * 3,
        "res_gid": [10001] * 3,
        "groups": [],
        "cap_eff": 0,
        "cap_prm": 0,
        "cap_inh": 0,
        "cap_amb": 0,
        "cap_bnd": 0x200404FB,
        "no_new_privs": 1,
        "keep_caps": 0,
        "seccomp": 2,
        "fs_uid": 10001,
        "fs_gid": 10001,
        "securebits": 0,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("CapEff", "1"),
        ("CapPrm", "1"),
        ("CapInh", "1"),
        ("CapAmb", "1"),
        ("CapBnd", "200405fb"),
        ("CapBnd", "-1"),
        ("NoNewPrivs", "0"),
        ("Seccomp", "0"),
        ("Uid", "10001 10001 10001 0"),
        ("Gid", "10001 10001 10001 0"),
    ],
)
def test_native_drift_refuses(guard, key, value):
    module, status = guard
    status[key] = value
    with pytest.raises(ValueError, match="identity refused"):
        module.verify_native_identity()


@pytest.mark.parametrize("bits", [0x04, 0x10, 0x0C, 0x30, 0x100, -1])
def test_securebit_behavior_or_unknown_bits_refuse(guard, monkeypatch, bits):
    module, _ = guard
    monkeypatch.setattr(module, "_prctl", lambda op: bits if op == 27 else 0)
    with pytest.raises(ValueError, match="identity refused"):
        module.verify_native_identity()


@pytest.mark.parametrize("bits", [0x00, 0x03, 0x08, 0x20, 0x80])
def test_securebits_known_locked_off_allowed_recorded(guard, monkeypatch, bits):
    module, _ = guard
    monkeypatch.setattr(module, "_prctl", lambda op: bits if op == 27 else 0)
    assert accepted(module.verify_native_identity)["securebits"] == bits


def test_keepcaps_failure_or_enabled_refuses(guard, monkeypatch):
    module, _ = guard
    for value in [-1, 1]:
        monkeypatch.setattr(module, "_prctl", lambda op, value=value: value if op == 7 else 0)
        with pytest.raises(ValueError, match="identity refused"):
            module.verify_native_identity()


def test_strict_separate_zero_bounding_unchanged(guard):
    module, status = guard
    with pytest.raises(ValueError, match="identity refused"):
        module.verify_zero_identity()
    status["CapBnd"] = "0"
    assert accepted(module.verify_zero_identity) == {
        "res_uid": [10001] * 3,
        "res_gid": [10001] * 3,
        "groups": [],
        "cap_eff": 0,
        "cap_prm": 0,
        "cap_inh": 0,
        "cap_amb": 0,
        "cap_bnd": 0,
        "no_new_privs": 1,
    }


def test_root_and_groups_refuse_before_prctl(guard, monkeypatch):
    module, _ = guard
    monkeypatch.setattr(module.os, "getresuid", lambda: (0, 0, 0))
    monkeypatch.setattr(module, "_prctl", lambda op: pytest.fail("root reached prctl"))
    with pytest.raises(ValueError, match="identity refused"):
        module.verify_native_identity()


def test_no_env_or_caller_profile_selector(guard, monkeypatch):
    module, _ = guard
    monkeypatch.setenv("AMLAS_PRIVILEGE_PROFILE", "zero_bounding_v1")
    assert accepted(module.verify_native_identity)["cap_bnd"] == 0x200404FB
    with pytest.raises(TypeError):
        module.verify_native_identity(profile="zero_bounding_v1")
