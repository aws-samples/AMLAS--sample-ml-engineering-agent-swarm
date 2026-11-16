"""Derive the sharded CI assets from the pinned originals by exact, counted replacements.

``full-ci-python.sh`` becomes a gates script (the original without its pytest gate; the ``collection`` gate stays)
and a shard script (the original's preamble and checks, then one shard's pytest). ``full-ci.native.py`` runs the
gates container, then the shard containers and the merge, and adds ``sharding`` to its summary. Anything that does
not match the pinned original exactly is refused. Standard library only, Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PYTHON_SCRIPT = "full-ci-python.sh"
NATIVE_SCRIPT = "full-ci.native.py"
GATES_SCRIPT = "full-ci-python-gates.sh"
SHARD_SCRIPT = "full-ci-python-shard.sh"
TOOL_FILES = (
    "full_ci_shard.py",
    "full_ci_shard_plan.py",
    "full_ci_shard_merge.py",
    "shard_orchestrator.py",
)
DATA_FILES = ("shard-durations.json", "shard-params.json")

EXEC_COLLECTION_EXPORT = "export AMLAS_CI_COLLECTION_METADATA=/ci-out/execution-collection.metadata.json"
PYTEST_GATE = (
    "run_gate pytest python -m pytest -p full_ci_collection -q -ra --basetemp=/ci-run/tmp/full-owned"
    " -o cache_dir=/ci-run/pytest --junitxml=/ci-out/full-pytest.xml tests || failed=1"
)
SDIST_EXPORT = "export AMLAS_SDIST_INSPECTION_ARTIFACT=$(cat /ci-out/sdist-path)"
SHARD_BLOCK = """failed=0
%(sdist)s
set +e
python -m pytest -p full_ci_shard -q -ra --basetemp=/ci-run/tmp/full-owned -o cache_dir=/ci-run/pytest --junitxml=/ci-shard/junit.xml tests > /ci-shard/pytest.log 2>&1
code=$?
set -e
printf '%%s\\n' "$code" > /ci-shard/pytest.exit
if [ "$code" != 0 ]; then failed=1; fi
test -z "$(git status --porcelain)" || failed=1
exit "$failed"
"""
GIT_STATUS_GATE = 'test -z "$(git status --porcelain)" || failed=1'
NATIVE_PYTHON_CALL = 'if phase_codes["package"]==0:docker_phase("python","full-ci-python.sh")'
NATIVE_PYTHON_SHARDED = (
    'if phase_codes["package"]==0:\n'
    '                docker_phase("python","full-ci-python-gates.sh")\n'
    '                if phase_codes["python"]==0:\n'
    '                    phase_codes["python"]=1\n'
    "                    sys.path.insert(0,str(assets));import shard_orchestrator\n"
    '                    phase_codes["python"]=shard_orchestrator.run_from_native(run_id=c["run_id"],image=PY_IMAGE,'
    'out=out,cache=cache,qa=qa,base=base,workplace=w,cpu=cpu,certifi=certifi,case_path=c["case_path"],assets=assets,'
    'candidate=c["candidate"],tree=c["tree"]);source_current()'
)
NATIVE_PASSED_ANCHOR = '        passed=set(phase_codes)=={"dox","package","audit","python","rust"}'
NATIVE_SHARDING_LINE = (
    '        if (out/"shards.metadata.json").is_file():'
    'summary["sharding"]=json.loads((out/"shards.metadata.json").read_text())\n'
)


class DeriveError(Exception):
    """The original asset is not the pinned text the derivation expects."""


def _once(text, needle, label):
    if text.count(needle) != 1:
        raise DeriveError(f"{label}: expected exactly one match, found {text.count(needle)}")


def derive_python_scripts(original):
    lines = original.split("\n")
    for needle in (EXEC_COLLECTION_EXPORT, PYTEST_GATE, SDIST_EXPORT, "failed=0"):
        if lines.count(needle) != 1:
            raise DeriveError(f"{needle!r}: expected exactly one line, found {lines.count(needle)}")
    after_gate = [line for line in lines[lines.index(PYTEST_GATE) + 1 :] if line != ""]
    if after_gate != [GIT_STATUS_GATE, 'exit "$failed"']:
        raise DeriveError("the original has steps after its pytest gate: " + repr(after_gate))
    gates = "\n".join(line for line in lines if line not in (EXEC_COLLECTION_EXPORT, PYTEST_GATE))
    preamble = "\n".join(lines[: lines.index("failed=0")]) + "\n"
    if 'test -z "$(git status --porcelain)"' not in preamble or "$AMLAS_CI_HEAD" not in preamble:
        raise DeriveError("the preamble lacks the source checks")
    shard = preamble + SHARD_BLOCK % {"sdist": SDIST_EXPORT}
    return gates, shard


def derive_native(original):
    _once(original, NATIVE_PYTHON_CALL, "python phase call")
    _once(original, NATIVE_PASSED_ANCHOR, "summary anchor")
    text = original.replace(NATIVE_PYTHON_CALL, NATIVE_PYTHON_SHARDED)
    return text.replace(NATIVE_PASSED_ANCHOR, NATIVE_SHARDING_LINE + NATIVE_PASSED_ANCHOR)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_directory(assets, config, tooling, out_assets, run_id):
    """Build the sharded assets directory and its configuration; returns the new configuration."""
    assets, out_assets = Path(assets), Path(out_assets)
    for name in (PYTHON_SCRIPT, NATIVE_SCRIPT):
        if _sha(assets / name) != config["asset_pins"][name]:
            raise DeriveError(f"{name} does not match its pin")
    gates, shard = derive_python_scripts((assets / PYTHON_SCRIPT).read_text())
    native = derive_native((assets / NATIVE_SCRIPT).read_text())
    out_assets.mkdir(mode=0o700)
    for entry in sorted(assets.iterdir()):
        if entry.is_file() and entry.name != NATIVE_SCRIPT:
            shutil.copyfile(entry, out_assets / entry.name)
    (out_assets / GATES_SCRIPT).write_text(gates)
    (out_assets / SHARD_SCRIPT).write_text(shard)
    (out_assets / NATIVE_SCRIPT).write_text(native)
    for name in TOOL_FILES + DATA_FILES:
        shutil.copyfile(Path(tooling) / name, out_assets / name)
    new = dict(config)
    new["run_id"] = run_id
    new["asset_dir"] = str(out_assets)
    new["asset_pins"] = {
        p.name: _sha(p) for p in sorted(out_assets.iterdir()) if p.is_file() and p.name != "Source.bundle"
    }
    return new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tooling", required=True, type=Path)
    parser.add_argument("--out-assets", required=True, type=Path)
    parser.add_argument("--config-out", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        new = derive_directory(
            args.assets, json.loads(args.config.read_text()), args.tooling, args.out_assets, args.run_id
        )
    except DeriveError as exc:
        print(f"derive_sharded_assets: refused: {exc}", file=sys.stderr)
        return 2
    with open(args.config_out, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(new, sort_keys=True))
    print(json.dumps({"run_id": args.run_id, "asset_pins": new["asset_pins"]}, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
