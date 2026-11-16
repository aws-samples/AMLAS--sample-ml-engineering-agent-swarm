"""Require the security initializer's own toolchain checks in each source CI."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _builder_image() -> str:
    dockerfile = (ROOT / "Dockerfile.processing").read_text()
    match = re.search(r"^FROM --platform=linux/amd64 (\S+) AS initializer$", dockerfile, re.M)
    assert match is not None
    image = match.group(1)
    assert re.fullmatch(r"public\.ecr\.aws/docker/library/rust:[^@\s]+@sha256:[a-f0-9]{64}", image)
    return image


def _command_lines(commands: str) -> list[str]:
    return [line.strip() for line in commands.splitlines()]


def _require_checks(commands: str) -> None:
    lines = _command_lines(commands)
    assert "cargo test --workspace --locked --offline" in lines, (
        "the strict default tests need their own command; the native feature line does not stand in for it"
    )
    assert not any(re.search(r"\bcargo\b", line) and "||" in line for line in lines), (
        "a cargo command must be able to fail the job"
    )
    assert "cargo fmt --all --check" in commands
    assert re.search(r"cargo test\b[^\n]*--workspace[^\n]*--locked[^\n]*--offline", commands)
    assert re.search(
        r"cargo build\b[^\n]*--release[^\n]*--locked[^\n]*--offline"
        r"[^\n]*--target x86_64-unknown-linux-musl",
        commands,
    )
    assert "CARGO_TARGET_DIR" in commands
    assert "processing_bootstrap" in commands


def test_gitlab_runs_locked_initializer_checks_with_the_image_compiler():
    pipeline = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())
    job = pipeline.get("processing-bootstrap")
    assert isinstance(job, dict), "GitLab must check the initializer crate independently"
    assert job["stage"] == "verify"
    assert job["image"] == _builder_image()
    assert job.get("before_script") == [], "the Rust job must not inherit Python installation"
    assert not job.get("allow_failure", False)
    commands = "\n".join(job["script"])
    _require_checks(commands)


def test_public_release_checks_initializer_in_a_confined_pinned_compiler():
    workflow = yaml.safe_load((ROOT / ".github/workflows/public-release.yml").read_text())
    job = workflow["jobs"].get("processing-bootstrap")
    assert isinstance(job, dict), "the public release must include the initializer checks"
    assert not job.get("continue-on-error", False)
    steps = job["steps"]
    assert all(not step.get("continue-on-error", False) for step in steps)
    checkout = next(step["uses"] for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    assert re.fullmatch(r"actions/checkout@[a-f0-9]{40}", checkout)
    commands = "\n".join(step["run"] for step in steps if "run" in step)
    assert _builder_image() in commands
    assert "--network none" in commands and "--platform linux/amd64" in commands
    assert '"/work:ro"' in commands
    assert "if" not in job and all("if" not in step for step in steps), "the public checks are unconditional"
    assert not re.search(r"\bexit\b", commands), (
        "the public job has no early exit: it runs the tests on any kernel"
    )
    _require_checks(commands)


def _require_native_checks(commands: str) -> None:
    for operation, flags in [
        ("test", "--workspace"),
        ("build", "--release"),
    ]:
        assert re.search(
            rf"cargo {operation}\b[^\n]*{flags}[^\n]*--locked[^\n]*--offline"
            r"[^\n]*--features sagemaker-native-v1",
            commands,
        ), f"native initializer feature lacks locked offline cargo {operation} gate"
    assert "cargo test --workspace --locked --offline --features sagemaker-native-v1" in _command_lines(
        commands
    )


def test_gitlab_checks_native_profile_in_addition_to_strict_default():
    job = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())["processing-bootstrap"]
    commands = "\n".join(job["script"])
    _require_checks(commands)
    _require_native_checks(commands)


def test_public_release_checks_native_profile_in_addition_to_strict_default():
    job = yaml.safe_load((ROOT / ".github/workflows/public-release.yml").read_text())["jobs"][
        "processing-bootstrap"
    ]
    commands = "\n".join(step["run"] for step in job["steps"] if "run" in step)
    _require_checks(commands)
    _require_native_checks(commands)
