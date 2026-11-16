"""Extract a source-owned binary-only hashed CPU import closure from uv.lock."""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOTS = ("boto3", "pydantic", "python-frontmatter", "pyyaml", "numpy", "scipy", "scikit-learn", "python-dotenv", "pydantic-settings", "requests")


def requirements(lock):
    packages = {}
    for package in lock["package"]:
        name = package["name"]
        packages.setdefault(name, []).append(package)
    pending, selected = list(ROOTS), {}
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        choices = packages[name]
        if len(choices) != 1:
            raise ValueError("selected paired CPU dependency is ambiguous")
        package = choices[0]
        if "registry" not in package.get("source", {}) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]*", package["version"]):
            raise ValueError("paired CPU closure requires locked registry versions")
        hashes = sorted({wheel["hash"] for wheel in package.get("wheels", [])})
        if not hashes or any(not re.fullmatch("sha256:[a-f0-9]{64}", digest) for digest in hashes):
            raise ValueError("paired CPU closure requires hashed binary artifacts")
        selected[name] = name + "==" + package["version"] + " " + " ".join("--hash=" + digest for digest in hashes)
        pending.extend(dependency["name"] for dependency in package.get("dependencies", []))
    return "\n".join(selected[name] for name in sorted(selected)) + "\n"


if __name__ == "__main__":
    lock = tomllib.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(requirements(lock))
