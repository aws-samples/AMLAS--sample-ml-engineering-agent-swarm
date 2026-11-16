"""Fixed OS-only fixture; trusted test startup, never a production root Python path."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import sys

PATHS = (
    "/home/appuser/.aws/credentials",
    "/home/appuser/.aws/config",
    "/root/.aws/credentials",
    "/root/.aws/config",
)
mode = sys.argv[1]
assert mode in ("positive", "extra_fd", "unsealed", "writable")
assert os.geteuid() == 0
providers = []
for index, name in enumerate(PATHS):
    ancestors = []
    parts = ["", *name.strip("/").split("/")]
    for component in range(len(parts)):
        current = "/" + "/".join(parts[1 : component + 1])
        try:
            node = os.lstat(current)
        except FileNotFoundError:
            providers.append(
                {"provider_index": index, "missing_component_index": component, "ancestors": ancestors}
            )
            break
        assert stat.S_ISDIR(node.st_mode) and not stat.S_ISLNK(node.st_mode)
        ancestors.append(
            {
                "component_index": component,
                "device": node.st_dev,
                "inode": node.st_ino,
                "uid": node.st_uid,
                "gid": node.st_gid,
                "mode": stat.S_IMODE(node.st_mode),
            }
        )
    else:
        raise AssertionError("fixture provider unexpectedly exists")
raw = json.dumps(
    {
        "schema_version": 1,
        "inventory_kind": "raw_model_provider_absence_before_drop_v1",
        "observation": "earlier_original_component_absence",
        "providers": providers,
    },
    sort_keys=True,
).encode()
output = os.open("/opt/ml/model", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
assert output == 3
fcntl.flock(3, fcntl.LOCK_EX | fcntl.LOCK_NB)
writer = os.memfd_create("fixed-qwen-provider-fixture", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
assert writer == 4
os.write(writer, raw)
if mode != "unsealed":
    fcntl.fcntl(writer, fcntl.F_ADD_SEALS, 15)
if mode != "writable":
    reader = os.open("/proc/self/fd/4", os.O_RDONLY | os.O_CLOEXEC)
    os.dup2(reader, 4)
    os.close(reader)
os.setgroups([])
os.setresgid(10001, 10001, 10001)
os.setresuid(10001, 10001, 10001)
if mode == "extra_fd":
    assert os.dup(4) == 5
# Application imports occur only after actual permanent drop. Shared native proof
# and model provider consumer are unchanged source-owned dependencies.
sys.path.insert(0, "/workspace/processing_bootstrap")
from amlas.sagemaker_entrypoints import qwen_sft_train as worker  # noqa: E402 - only after permanent drop

try:
    proof = worker._guard()
    worker._guard_output()
except (ValueError, RuntimeError, OSError) as error:
    if mode == "positive":
        raise
    print(
        json.dumps(
            {
                "fixture": mode,
                "status": "REFUSED",
                "error_type": type(error).__name__,
                "native_admission": False,
            },
            sort_keys=True,
        )
    )
else:
    assert mode == "positive", "negative fixture accepted"
    observation = worker._provider_observation("before_core_import")
    states = [record["current_state"] for record in observation["provider_path_observations"]]
    assert states == ["current_absent", "current_absent", "denied_after_drop", "denied_after_drop"], states
    assert observation["inventory_sha256"] == hashlib.sha256(raw).hexdigest()
    assert proof["res_uid"] == proof["res_gid"] == [10001, 10001, 10001]
    assert proof["groups"] == [] and all(
        proof[key] == 0 for key in ("cap_eff", "cap_prm", "cap_inh", "cap_amb")
    )
    assert proof["no_new_privs"] == 1 and proof["seccomp"] == 2
    print(
        json.dumps(
            {
                "fixture": mode,
                "status": "PASS",
                "proof": proof,
                "states": states,
                "readonly": fcntl.fcntl(4, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY,
                "seals": fcntl.fcntl(4, fcntl.F_GET_SEALS),
                "native_admission": False,
            },
            sort_keys=True,
        )
    )
