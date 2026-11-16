"""Freeze source-only exact-policy fixture inputs. Compile and execute only in AWS."""
import argparse
import hashlib
import json
from pathlib import Path


POLICY_PATH = "processing_bootstrap/src/oracle_boundary.rs"
PYTHON_PATHS = [
    "src/amlas/evaluation/coding_oracle_child.py",
    "src/amlas/evaluation/coding_oracle.py",
    "src/amlas/evaluation/coding_oracle_catalog.py",
]
PINS = {
    POLICY_PATH: "8efc77465802a9cc69d60304c822766514ed349efc09564d8a542b96f2789adb",  # pragma: allowlist secret - reviewed immutable integrity pin
    PYTHON_PATHS[0]: "0d1490997a7cb223abd80a29571690fb5b693cf51187c45425ad373603e7d2c3",  # pragma: allowlist secret - reviewed immutable integrity pin
    PYTHON_PATHS[1]: "f7af2ed7542ac7fb199119c4c9bbb1821d9ad398bb75aacfeb84a7f54c0c1450",  # pragma: allowlist secret - reviewed immutable integrity pin
    PYTHON_PATHS[2]: "0b244aebd870d99789896965cd059fabbde1fd574860eba3ed3cc002f76b1867",  # pragma: allowlist secret - reviewed immutable integrity pin
}
SOURCE_COMMIT = "0d8bf7bee8c100a002c16ec7754ce3f1448c435a"  # pragma: allowlist secret - reviewed immutable integrity pin
IMAGE_CONFIG_ID = "sha256:c33493df0f88c70bb7c359cfac9ded885d1be32938e5c3b10e0f234e215a883f"
LAUNCHER_SHA = "86a0fdd539ac7a73ef012e1f390122557d7058bced16397bfb21529dd2deba81"  # pragma: allowlist secret - reviewed immutable integrity pin
COMPILER_IMAGE = "public.ecr.aws/docker/library/rust:1.90.0-alpine3.22@sha256:ee9b9f2ebd8741e4ecf1ade08918e492d642c4e625be481bb7b7954b89e9d504"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(root: Path, output: Path):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("output_exists")
    inputs = {}
    for path, expected in PINS.items():
        value = (root / path).read_bytes()
        if digest(value) != expected:
            raise ValueError("source_pin")
        inputs[path] = value
    helper = Path(__file__).with_name("fixture_suffix.rs").read_bytes()
    if b"#[cfg(test)]" not in helper or b"fn main()" not in helper:
        raise ValueError("fixture_suffix")
    composite = inputs[POLICY_PATH] + helper
    receipt = {
        "schema": "oracle_native_fixture_inputs_v2",
        "qualified_source_commit": SOURCE_COMMIT,
        "qualified_image_config_id": IMAGE_CONFIG_ID,
        "qualified_launcher_sha256": LAUNCHER_SHA,
        "compiler_image": COMPILER_IMAGE,
        "source_sha256": dict(PINS),
        "policy_bytes": len(inputs[POLICY_PATH]),
        "helper_sha256": digest(helper),
        "preparer_sha256": digest(Path(__file__).read_bytes()),
        "runner_sha256": digest(Path(__file__).with_name("run.py").read_bytes()),
        "composite_sha256": digest(composite),
        "policy_modes": ["baseline", "complete", "landlock"],
        "native_admission": False,
    }
    output.mkdir(parents=True)
    for path, value in inputs.items():
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(composite if path == POLICY_PATH else value)
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output), sort_keys=True))
