"""Fixed trusted launcher: verify the drop before any application import."""

import fcntl
import os
import re
import stat
import sys

# Refuse direct root execution before importing even the shared trusted OS guard.
if os.getresuid() != (10001, 10001, 10001) or os.getresgid() != (10001, 10001, 10001) or os.getgroups() != []:
    raise RuntimeError("acquisition worker identity refused")
sys.path.insert(0, "/app")
from processing_identity import verify_native_identity

verify_native_identity()

source = os.environ.get("AMLAS_SOURCE_COMMIT", "")
if not re.fullmatch(r"[a-f0-9]{40}", source) or sys.argv.count("--source-commit") != 1:
    raise RuntimeError("acquisition source binding refused")
index = sys.argv.index("--source-commit")
if index + 1 >= len(sys.argv) or sys.argv[index + 1] != source:
    raise RuntimeError("acquisition source binding refused")
info = os.fstat(3)
if not stat.S_ISDIR(info.st_mode) or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (
    10001,
    10001,
    0o700,
):
    raise RuntimeError("acquisition output authority refused")
fcntl.flock(3, fcntl.LOCK_EX | fcntl.LOCK_NB)
sys.path.insert(0, "/app/src")
from amlas.sagemaker_entrypoints.dataset_acquisition import main_native  # noqa: E402

main_native()
