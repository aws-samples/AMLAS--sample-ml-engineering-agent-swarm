"""Run inside the existing AWS case image; inspect actual filesystem, no worker.

This proves image ancestor ownership only. Native input mount ownership, permanent
identity/output-FD qualification and successful case publication remain separate.
"""

from __future__ import annotations

import json
import os
import stat


def check_image_ownership():
    rows = []
    for path in ("/opt", "/opt/ml", "/opt/ml/processing"):
        value = os.stat(path, follow_symlinks=False)
        row = {
            "path": path,
            "uid": value.st_uid,
            "gid": value.st_gid,
            "mode": stat.S_IMODE(value.st_mode),
            "directory": stat.S_ISDIR(value.st_mode),
        }
        rows.append(row)
    print(json.dumps({"image_input_ancestors": rows}), flush=True)
    for row in rows:
        assert row["directory"] and row["uid"] == row["gid"] == 0, (
            "case image input ancestor must be root-owned",
            row,
        )
        assert not row["mode"] & 0o022, ("case image input ancestor must not be writable", row)
    output = os.stat("/opt/ml/processing/output", follow_symlinks=False)
    assert stat.S_ISDIR(output.st_mode) and not stat.S_ISLNK(output.st_mode)
    output_state = (output.st_uid, output.st_gid, stat.S_IMODE(output.st_mode))
    print(json.dumps({"image_output_tuple": output_state}), flush=True)
    assert output_state in ((0, 0, 0o755), (10001, 10001, 0o700)), (
        "case image output tuple must satisfy unchanged initializer guard",
        output_state,
    )
    initializer = os.stat("/usr/local/bin/amlas-processing-init", follow_symlinks=False)
    assert stat.S_ISREG(initializer.st_mode) and initializer.st_uid == initializer.st_gid == 0
    assert not stat.S_IMODE(initializer.st_mode) & 0o022
    print(json.dumps({"ownership_control": "PASS", "native_case_admission": False}), flush=True)


if __name__ == "__main__":
    check_image_ownership()
