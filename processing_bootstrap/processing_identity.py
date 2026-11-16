"""Fixed OS-only profile guards; never accept caller or environment proof."""

import os
import re

NATIVE_PROFILE = "sagemaker_native_ceiling_v1"
ZERO_PROFILE = "zero_bounding_v1"
NATIVE_CEILING = 0x200404FB
_REASON = "processing identity refused"


def _read_status():
    with open("/proc/self/status", "rb") as stream:
        raw = stream.read(32769)
    if len(raw) > 32768:
        raise ValueError(_REASON)
    values = {}
    for line in raw.decode("ascii").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key in values:
                raise ValueError(_REASON)
            values[key] = value.strip()
    return values


def _prctl(option):
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    return libc.prctl(option, ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0))


def _base_identity():
    uid, gid, groups = os.getresuid(), os.getresgid(), os.getgroups()
    if uid != (10001, 10001, 10001) or gid != (10001, 10001, 10001) or groups != []:
        raise ValueError(_REASON)
    values = _read_status()
    proof = {"res_uid": list(uid), "res_gid": list(gid), "groups": list(groups)}
    for key, output in (
        ("CapEff", "cap_eff"),
        ("CapPrm", "cap_prm"),
        ("CapInh", "cap_inh"),
        ("CapAmb", "cap_amb"),
        ("CapBnd", "cap_bnd"),
    ):
        value = values.get(key, "")
        if not re.fullmatch(r"[a-fA-F0-9]{1,16}", value):
            raise ValueError(_REASON)
        proof[output] = int(value, 16)
    if values.get("NoNewPrivs") != "1" or any(
        proof[key] != 0 for key in ("cap_eff", "cap_prm", "cap_inh", "cap_amb")
    ):
        raise ValueError(_REASON)
    proof["no_new_privs"] = 1
    return proof, values


def verify_zero_identity():
    """Explicit strict profile: retain the original nine zero-bounding fields."""
    try:
        proof, _ = _base_identity()
        if proof["cap_bnd"] != 0:
            raise ValueError(_REASON)
        return proof
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        raise ValueError(_REASON) from None


def verify_native_identity():
    """Source-owned native profile; no selector or caller-authored proof."""
    try:
        proof, values = _base_identity()
        if proof["cap_bnd"] & ~NATIVE_CEILING or values.get("Seccomp") != "2":
            raise ValueError(_REASON)
        for field, output in (("Uid", "fs_uid"), ("Gid", "fs_gid")):
            ids = values.get(field, "").split()
            if len(ids) != 4 or any(value != "10001" for value in ids):
                raise ValueError(_REASON)
            proof[output] = 10001
        keep_caps, securebits = _prctl(7), _prctl(27)
        if keep_caps != 0 or not 0 <= securebits <= 0xFF or securebits & 0x14:
            raise ValueError(_REASON)
        proof.update(keep_caps=0, seccomp=2, securebits=securebits)
        return proof
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        raise ValueError(_REASON) from None
