"""Keep the release runner able to execute the same complete locked test cohort."""

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "public-release.yml"
GITLAB_EXPORT_AUDIT = 'uv run pip-audit --disable-pip -r "$TMPDIR/requirements.txt" --strict'
GITLAB_CPU_LOCK_AUDIT = (
    "uv run --frozen --no-sync pip-audit -r scripts/qwen_cpu_test.requirements.lock "
    "--strict --disable-pip --vulnerability-service osv"
)
GITLAB_TRAINING_LOCK_AUDIT = (
    "uv run --frozen --no-sync pip-audit -r src/amlas/training/qwen_sft.requirements.lock "
    "--strict --disable-pip --vulnerability-service osv"
)
PIN_SET_CHECK = "uv run --frozen --no-sync python scripts/audit_pin_sets.py"


def _verify_job():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["verify"]


def _run_commands():
    return [step["run"] for step in _verify_job()["steps"] if "run" in step]


def test_release_provisions_the_declared_full_test_extras_and_node():
    """Native collection imports CDK; the full cohort includes managed CPU APIs."""
    commands = _run_commands()
    sync = next(command for command in commands if command.startswith("uv sync "))
    tokens = shlex.split(sync)
    extras = {tokens[i + 1] for i, token in enumerate(tokens) if token == "--extra"}
    assert {"dev", "security", "cdk", "agents", "runtime-cpu"} <= extras
    assert "--frozen" in tokens
    steps = _verify_job()["steps"]
    node = next((step for step in steps if step.get("uses", "").startswith("actions/setup-node@")), None)
    assert node is not None, "CDK/jsii must have an explicit supported Node prerequisite"
    assert re.fullmatch(r"actions/setup-node@[a-f0-9]{40}", node["uses"])
    assert str(node["with"]["node-version"]) == "22"
    assert steps.index(node) < next(i for i, step in enumerate(steps) if step.get("run") == sync)


def test_release_uses_a_scoped_runner_temp_directory_before_tests():
    """Ownership tests need TMPDIR; source checkout must stay separate from output."""
    job = _verify_job()
    value = job.get("env", {}).get("TMPDIR", "")
    # GitHub's context-availability table excludes runner from jobs.<job_id>.env.
    # https://docs.github.com/en/actions/reference/workflows-and-actions/contexts
    contexts = re.findall(r"\$\{\{\s*([a-z_]+)\.", value)
    assert contexts and set(contexts) <= {
        "github",
        "needs",
        "strategy",
        "matrix",
        "vars",
        "secrets",
        "inputs",
    }
    assert value == "${{ github.workspace }}/../amlas-public-release"
    checkout = Path("/workspace/repository")
    resolved = Path(value.replace("${{ github.workspace }}", str(checkout))).resolve()
    assert resolved != checkout and checkout not in resolved.parents
    commands = _run_commands()
    prepare = next((i for i, command in enumerate(commands) if 'mkdir -p "$TMPDIR"' in command), None)
    tests = next(i for i, command in enumerate(commands) if "pytest" in command)
    assert prepare is not None and prepare < tests


def test_release_executes_module_pytest_without_dropping_provisioned_extras():
    """Console pytest omitted repo-root imports and uv resync can discard CDK."""
    commands = _run_commands()
    tests = next(command for command in commands if "pytest" in command)
    tokens = shlex.split(tests)
    assert tokens[:6] == ["uv", "run", "--frozen", "--no-sync", "python", "-m"]
    assert tokens[6:] == ["pytest", "-q", "tests"]
    for command in commands:
        if "uv run " in command:
            assert "uv run --frozen --no-sync " in command


def test_release_keeps_required_security_checks_and_read_only_permissions():
    job = _verify_job()
    run_commands = _run_commands()
    commands = "\n".join(run_commands)
    for check in (
        "verify_repository_governance.py",
        "ruff check",
        "bandit",
        "pip-audit",
        "detect-secrets-hook",
        "cyclonedx-py",
    ):
        assert check in commands
    export = next(command for command in run_commands if "uv export " in command)
    audit = next(command for command in run_commands if "pip-audit" in command)
    assert "--no-hashes" not in shlex.split(export)
    assert "--disable-pip" in shlex.split(audit)
    assert "--no-deps" not in shlex.split(audit)
    for lock_audit in (GITLAB_CPU_LOCK_AUDIT, GITLAB_TRAINING_LOCK_AUDIT):
        assert run_commands.count(lock_audit) == 1, "each hashed lock keeps its own audit step"
    assert run_commands.count(PIN_SET_CHECK) == 1, "the pin sets that no lock check reaches keep their own step"
    gitlab = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())
    gitlab_audits = [line for line in gitlab["pip-audit"]["script"] if "pip-audit" in shlex.split(line)]
    assert len(gitlab_audits) == 3
    gitlab_export_audit, gitlab_cpu_lock_audit, gitlab_training_lock_audit = gitlab_audits
    assert "$TMPDIR/requirements.txt" in shlex.split(gitlab_export_audit)
    assert "--disable-pip" in shlex.split(gitlab_export_audit)
    assert "--strict" in shlex.split(gitlab_export_audit)
    assert "--no-deps" not in shlex.split(gitlab_export_audit)
    assert gitlab_cpu_lock_audit == GITLAB_CPU_LOCK_AUDIT
    assert gitlab_training_lock_audit == GITLAB_TRAINING_LOCK_AUDIT
    assert gitlab["pip-audit"]["script"].count(PIN_SET_CHECK) == 1
    assert yaml.safe_load(WORKFLOW.read_text())["permissions"] == {"contents": "read"}
    assert not job.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in job["steps"])


def _script_line(command):
    return f"    - {command}\n"


@pytest.mark.parametrize(
    ("original", "mutated"),
    [
        pytest.param(
            _script_line(GITLAB_EXPORT_AUDIT),
            _script_line(GITLAB_EXPORT_AUDIT.replace(" --disable-pip", "")),
            id="export-audit-loses-disable-pip",
        ),
        pytest.param(
            _script_line(GITLAB_EXPORT_AUDIT),
            _script_line(GITLAB_EXPORT_AUDIT.replace(" --strict", "")),
            id="export-audit-loses-strict",
        ),
        pytest.param(
            _script_line(GITLAB_EXPORT_AUDIT),
            _script_line(GITLAB_EXPORT_AUDIT + " --no-deps"),
            id="export-audit-gains-no-deps",
        ),
        pytest.param(_script_line(GITLAB_EXPORT_AUDIT), "", id="export-audit-removed"),
        pytest.param(
            _script_line(GITLAB_CPU_LOCK_AUDIT),
            _script_line(GITLAB_CPU_LOCK_AUDIT.replace(" --disable-pip", "")),
            id="cpu-lock-audit-loses-disable-pip",
        ),
        pytest.param(
            _script_line(GITLAB_CPU_LOCK_AUDIT),
            _script_line(GITLAB_CPU_LOCK_AUDIT.replace(" --strict", "")),
            id="cpu-lock-audit-loses-strict",
        ),
        pytest.param(
            _script_line(GITLAB_TRAINING_LOCK_AUDIT),
            _script_line(GITLAB_TRAINING_LOCK_AUDIT.replace(" --disable-pip", "")),
            id="training-lock-audit-loses-disable-pip",
        ),
        pytest.param(
            _script_line(GITLAB_TRAINING_LOCK_AUDIT),
            _script_line(GITLAB_TRAINING_LOCK_AUDIT.replace(" --strict", "")),
            id="training-lock-audit-loses-strict",
        ),
        pytest.param(_script_line(GITLAB_TRAINING_LOCK_AUDIT), "", id="training-lock-audit-removed"),
        pytest.param(
            _script_line(PIN_SET_CHECK),
            _script_line(PIN_SET_CHECK.replace(" --frozen --no-sync", "")),
            id="pin-set-check-loses-frozen-no-sync",
        ),
        pytest.param(_script_line(PIN_SET_CHECK), "", id="pin-set-check-removed"),
    ],
)
def test_each_gitlab_audit_line_is_checked_by_itself(original, mutated, tmp_path, monkeypatch):
    """A flag kept on one GitLab audit line must not hide its loss on another line."""
    text = (ROOT / ".gitlab-ci.yml").read_text()
    assert text.count(original) == 1, "the real audit line must be present for the mutation to mean anything"
    assert mutated != original
    copy = tmp_path / ".gitlab-ci.yml"
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    copy.write_text(text)
    test_release_keeps_required_security_checks_and_read_only_permissions()
    copy.write_text(text.replace(original, mutated))
    with pytest.raises(AssertionError):
        test_release_keeps_required_security_checks_and_read_only_permissions()


def _step_line(command):
    return f"      - run: {command}\n"


@pytest.mark.parametrize(
    ("original", "mutated"),
    [
        pytest.param(
            _step_line(GITLAB_CPU_LOCK_AUDIT),
            _step_line(GITLAB_CPU_LOCK_AUDIT.replace(" --strict", "")),
            id="cpu-lock-audit-loses-strict",
        ),
        pytest.param(_step_line(GITLAB_CPU_LOCK_AUDIT), "", id="cpu-lock-audit-removed"),
        pytest.param(
            _step_line(GITLAB_TRAINING_LOCK_AUDIT),
            _step_line(GITLAB_TRAINING_LOCK_AUDIT.replace(" --disable-pip", "")),
            id="training-lock-audit-loses-disable-pip",
        ),
        pytest.param(
            _step_line(GITLAB_TRAINING_LOCK_AUDIT),
            _step_line(GITLAB_TRAINING_LOCK_AUDIT.replace(" --strict", "")),
            id="training-lock-audit-loses-strict",
        ),
        pytest.param(_step_line(GITLAB_TRAINING_LOCK_AUDIT), "", id="training-lock-audit-removed"),
        pytest.param(_step_line(PIN_SET_CHECK), "", id="pin-set-check-removed"),
    ],
)
def test_each_release_lock_audit_step_is_checked_by_itself(original, mutated, tmp_path, monkeypatch):
    """The public workflow audits each hashed lock in its own step with the same command."""
    text = WORKFLOW.read_text()
    assert text.count(original) == 1, "the real audit step must be present for the mutation to mean anything"
    assert mutated != original
    copy = tmp_path / "public-release.yml"
    monkeypatch.setitem(globals(), "WORKFLOW", copy)
    copy.write_text(text)
    test_release_keeps_required_security_checks_and_read_only_permissions()
    copy.write_text(text.replace(original, mutated))
    with pytest.raises(AssertionError):
        test_release_keeps_required_security_checks_and_read_only_permissions()


EVALUATION_FILE_VARIABLE = "AMLAS_TEST_HUMANEVAL_PARQUET"
PROCESSING_CAPABILITY_DROPS = ("SETPCAP", "NET_RAW", "MKNOD", "SETFCAP")
LARGE_AMD64_RUNNER = ["arch:amd64", "size:large"]
INITIALIZER_COMMANDS = [
    "cargo fmt --all --check",
    "cargo test --workspace --locked --offline",
    "cargo build --release --locked --offline --target x86_64-unknown-linux-musl",
    'export CARGO_TARGET_DIR="$TMPDIR/native-target"',
    "cargo test --workspace --locked --offline --features sagemaker-native-v1",
    "cargo build --release --locked --offline --features sagemaker-native-v1 --target x86_64-unknown-linux-musl",
]
# What the stand-in cargo records: target directory name | RUST_TEST_THREADS | arguments.
ALL_INITIALIZER_CALLS = [
    "target|1|fmt --all --check",
    "target|1|test --workspace --locked --offline",
    "target|1|build --release --locked --offline --target x86_64-unknown-linux-musl",
    "native-target|1|test --workspace --locked --offline --features sagemaker-native-v1",
    "native-target|1|build --release --locked --offline --features sagemaker-native-v1"
    " --target x86_64-unknown-linux-musl",
]
BUILD_ONLY_INITIALIZER_CALLS = [call for call in ALL_INITIALIZER_CALLS if "|test " not in call]
# The public workflow's container has no kernel branch: it runs every command on whatever kernel it gets.
GITHUB_INITIALIZER_SCRIPT = [
    "uname -r && cargo fmt --all --check",
    "cargo test --workspace --locked --offline",
    "cargo build --release --locked --offline --target x86_64-unknown-linux-musl",
    "export CARGO_TARGET_DIR=/worktmp/native-target",
    "cargo test --workspace --locked --offline --features sagemaker-native-v1",
    "cargo build --release --locked --offline --features sagemaker-native-v1 --target x86_64-unknown-linux-musl",
]
OLDER_AND_NEWER_KERNELS = [
    "3.10.0-1160.el7.x86_64",
    "4.19.0-27-amd64",
    "5.10.0-28-amd64",
    "5.12.19",
    "5.15.0-1084-aws",
    "6.8.0-1017-azure",
]
VERIFY_STAGE_HEADER = "\nverify:\n  stage: verify\n"
STILL_RUNS_WORDS = "only the format check and both builds run here"


def test_release_jobs_stop_at_their_time_limits():
    """A passing verify job measured 77 minutes; neither job may run to the platform default."""
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    assert jobs["verify"]["timeout-minutes"] == 180
    assert jobs["processing-bootstrap"]["timeout-minutes"] == 30


def _run_environment_transport(tmp_path, exports):
    """Run the provisioning step's lines after the provisioning script against a prepared environment.sh."""
    commands = _run_commands()
    step = next(command for command in commands if '>> "$GITHUB_ENV"' in command)
    assert commands.index(step) < next(i for i, command in enumerate(commands) if "pytest" in command)
    lines = step.splitlines()
    assert lines[0] == 'bash scripts/provision_qwen_cpu_tests.sh "$TMPDIR/qwen-cpu-tests"'
    provisioned = tmp_path / "qwen-cpu-tests"
    provisioned.mkdir()
    (provisioned / "environment.sh").write_text(
        "".join(f"export {name}={shlex.quote(value)}\n" for name, value in exports.items())
    )
    github_env = tmp_path / "github-env"
    github_env.write_text("")
    script = tmp_path / "transport.sh"
    script.write_text("\n".join(lines[1:]) + "\n")
    result = subprocess.run(
        ["bash", "-e", str(script)],
        env={"PATH": os.environ["PATH"], "TMPDIR": str(tmp_path), "GITHUB_ENV": str(github_env)},
        capture_output=True,
        text=True,
        check=False,
    )
    return result, github_env.read_text()


def test_release_transports_the_provisioned_evaluation_file_to_the_pytest_step(tmp_path):
    """Each step is a new shell: what environment.sh exports reaches pytest only through GITHUB_ENV."""
    exports = {
        "PYTHONPATH": "/provisioned/site:/checkout/src",
        "AMLAS_QWEN_TEST_METADATA": "/provisioned/metadata",
        EVALUATION_FILE_VARIABLE: "/provisioned/evaluation/test.parquet",
    }
    result, written = _run_environment_transport(tmp_path, exports)
    assert result.returncode == 0, result.stderr
    assert dict(line.split("=", 1) for line in written.splitlines()) == {
        **exports,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
    }


def test_release_refuses_a_provisioned_environment_without_the_evaluation_file(tmp_path):
    """A missing file must stop the provisioning step, not 101 tests one hour later."""
    exports = {"PYTHONPATH": "/provisioned/site", "AMLAS_QWEN_TEST_METADATA": "/provisioned/metadata"}
    result, written = _run_environment_transport(tmp_path, exports)
    assert result.returncode != 0
    assert EVALUATION_FILE_VARIABLE in result.stderr
    assert written == ""


def test_gitlab_verify_sources_the_provisioned_environment_on_a_large_runner():
    """The script lines share one shell; the complete cohort needs more than the smallest runner size."""
    job = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())["verify"]
    script = job["script"]
    provision = script.index('bash scripts/provision_qwen_cpu_tests.sh "$TMPDIR/qwen-cpu-tests"')
    source = script.index('source "$TMPDIR/qwen-cpu-tests/environment.sh"')
    tests = script.index("uv run --frozen --no-sync python -m pytest -q tests")
    assert provision < source < tests
    assert sorted(job["tags"]) == LARGE_AMD64_RUNNER
    assert job["timeout"] == "3 hours"
    assert not job.get("allow_failure", False), "a failing complete cohort must fail the pipeline"
    assert job.get("when", "on_success") == "on_success"


def test_a_gitlab_verify_job_that_may_fail_is_noticed(tmp_path, monkeypatch):
    text = (ROOT / ".gitlab-ci.yml").read_text()
    assert text.count(VERIFY_STAGE_HEADER) == 1, (
        "the real job header must be present for the mutation to mean anything"
    )
    copy = tmp_path / ".gitlab-ci.yml"
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    copy.write_text(text)
    test_gitlab_verify_sources_the_provisioned_environment_on_a_large_runner()
    copy.write_text(text.replace(VERIFY_STAGE_HEADER, VERIFY_STAGE_HEADER + "  allow_failure: true\n"))
    with pytest.raises(AssertionError):
        test_gitlab_verify_sources_the_provisioned_environment_on_a_large_runner()


def _initializer_container_options(workflow_text):
    job = yaml.safe_load(workflow_text)["jobs"]["processing-bootstrap"]
    commands = [step["run"] for step in job["steps"] if "cargo test" in step.get("run", "")]
    assert len(commands) == 1
    tokens = shlex.split(commands[0].replace("\\\n", " "))
    assert tokens[:2] == ["docker", "run"]
    image = next(i for i, token in enumerate(tokens) if token.startswith("public.ecr.aws/"))
    return tokens[2:image]


def _option_values(options, flag):
    return [options[i + 1] for i, token in enumerate(options) if token == flag]


def _require_initializer_container_options(workflow_text):
    options = _initializer_container_options(workflow_text)
    assert sorted(_option_values(options, "--cap-drop")) == sorted(PROCESSING_CAPABILITY_DROPS)
    assert not any(token.startswith(("--cap-add", "--privileged")) for token in options)
    assert _option_values(options, "--ulimit") == ["nofile=65536:65536"]
    assert [value for value in _option_values(options, "-e") if value.startswith("RUST_TEST_THREADS=")] == [
        "RUST_TEST_THREADS=1"
    ]
    assert _option_values(options, "--network") == ["none"]


@pytest.mark.parametrize(
    ("original", "mutated"),
    [
        pytest.param("--cap-drop SETPCAP ", "", id="keeps-setpcap"),
        pytest.param("--cap-drop NET_RAW ", "", id="keeps-net-raw"),
        pytest.param("--cap-drop MKNOD ", "", id="keeps-mknod"),
        pytest.param(" --cap-drop SETFCAP", "", id="keeps-setfcap"),
        pytest.param(
            " --cap-drop SETFCAP", " --cap-drop SETFCAP --cap-add SYS_ADMIN", id="adds-a-capability"
        ),
        pytest.param(" --cap-drop SETFCAP", " --cap-drop SETFCAP --privileged", id="becomes-privileged"),
        pytest.param("--ulimit nofile=65536:65536", "", id="loses-descriptor-limit"),
        pytest.param(
            "--ulimit nofile=65536:65536", "--ulimit nofile=1024:524288", id="lowers-descriptor-limit"
        ),
        pytest.param("-e RUST_TEST_THREADS=1", "", id="loses-one-test-thread"),
        pytest.param("-e RUST_TEST_THREADS=1", "-e RUST_TEST_THREADS=2", id="uses-two-test-threads"),
    ],
)
def test_each_initializer_container_option_is_checked_by_itself(original, mutated):
    """The tests share one fixed directory, refuse above the processing ceiling and need descriptors above 4096."""
    text = WORKFLOW.read_text()
    assert text.count(original) == 1, "the real option must be present for the mutation to mean anything"
    _require_initializer_container_options(text)
    with pytest.raises(AssertionError):
        _require_initializer_container_options(text.replace(original, mutated))


def test_gitlab_initializer_job_keeps_its_commands_with_one_test_thread_on_a_large_runner():
    job = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())["processing-bootstrap"]
    assert str(job["variables"]["RUST_TEST_THREADS"]) == "1"
    assert sorted(job["tags"]) == LARGE_AMD64_RUNNER
    script = job["script"]
    kernel_line = script.index("uname -r")
    assert script[kernel_line + 2 :] == INITIALIZER_COMMANDS, "one kernel check, then the commands unchanged"


def _stand_in_tools(tmp_path):
    """Stand-ins for uname, rustup and cargo; cargo records each call and fails on the named subcommand."""
    tools = tmp_path / "tools"
    tools.mkdir()
    stand_ins = {
        "uname": '[ "$1" = "-r" ] || exit 64\nprintf "%s\\n" "$STAND_IN_KERNEL"\n',
        "rustup": "exit 0\n",
        "cargo": (
            'printf "%s|%s|%s\\n" "${CARGO_TARGET_DIR##*/}" "${RUST_TEST_THREADS-unset}" "$*" >> "$STAND_IN_LOG"\n'
            '[ "$1" != "$STAND_IN_FAILS" ]\n'
        ),
    }
    for name, body in stand_ins.items():
        path = tools / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    return tools, tmp_path / "cargo.log"


def _run_gitlab_initializer_job(tmp_path, kernel, failing="-"):
    """Run the whole job script with stand-ins for uname, rustup and cargo; return the result and cargo calls."""
    job = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())["processing-bootstrap"]
    project = tmp_path / "builds" / "project"
    (project / "processing_bootstrap").mkdir(parents=True)
    tools, log = _stand_in_tools(tmp_path)
    script = tmp_path / "job.sh"
    script.write_text("\n".join(job["script"]) + "\n")
    result = subprocess.run(
        ["sh", "-e", str(script)],
        cwd=project,
        env={
            **{name: str(value) for name, value in job.get("variables", {}).items()},
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "CI_PROJECT_DIR": str(project),
            "CI_JOB_ID": "1",
            "STAND_IN_KERNEL": kernel,
            "STAND_IN_LOG": str(log),
            "STAND_IN_FAILS": failing,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize(
    "kernel",
    ["5.13.0", "5.15.0-1084-aws", "6.8.0-1017-azure", "6.12.0-1-amd64", "10.0.0"],
)
def test_gitlab_initializer_job_runs_every_command_on_a_kernel_with_the_boundary_calls(kernel, tmp_path):
    result, calls = _run_gitlab_initializer_job(tmp_path, kernel)
    assert result.returncode == 0, result.stderr
    assert calls == ALL_INITIALIZER_CALLS
    assert "close_range" not in result.stdout


@pytest.mark.parametrize(
    "kernel",
    [
        "4.19.0-27-amd64",
        "5.10.0-28-amd64",
        "5.9.0",
        "5.12.19",
        "3.10.0-1160.el7.x86_64",
    ],
)
def test_gitlab_initializer_job_says_what_an_older_kernel_cannot_run_and_still_builds(kernel, tmp_path):
    """close_range arrived in Linux 5.9 and Landlock in 5.13; the public workflow runs the tests."""
    result, calls = _run_gitlab_initializer_job(tmp_path, kernel)
    assert result.returncode == 0, result.stderr
    assert calls == BUILD_ONLY_INITIALIZER_CALLS
    notices = [line for line in result.stdout.splitlines() if kernel in line and line != kernel]
    assert len(notices) == 1
    for word in ("5.13", "close_range", "Landlock", STILL_RUNS_WORDS, "GitHub"):
        assert word in notices[0]


def test_an_older_kernel_notice_that_does_not_say_what_still_runs_is_noticed(tmp_path, monkeypatch):
    text = (ROOT / ".gitlab-ci.yml").read_text()
    assert text.count(f"so {STILL_RUNS_WORDS}; ") == 1, (
        "the real words must be present for the mutation to mean anything"
    )
    copy = tmp_path / ".gitlab-ci.yml"
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    copy.write_text(text)
    kernel = "4.19.0-27-amd64"
    test_gitlab_initializer_job_says_what_an_older_kernel_cannot_run_and_still_builds(
        kernel, tmp_path / "kept"
    )
    copy.write_text(text.replace(f"so {STILL_RUNS_WORDS}; ", ""))
    with pytest.raises(AssertionError):
        test_gitlab_initializer_job_says_what_an_older_kernel_cannot_run_and_still_builds(
            kernel, tmp_path / "changed"
        )


@pytest.mark.parametrize(
    ("kernel", "failing", "calls_made"),
    [
        pytest.param("4.19.0-27-amd64", "fmt", BUILD_ONLY_INITIALIZER_CALLS[:1], id="older-format"),
        pytest.param("4.19.0-27-amd64", "build", BUILD_ONLY_INITIALIZER_CALLS[:2], id="older-build"),
        pytest.param("6.8.0-1017-azure", "test", ALL_INITIALIZER_CALLS[:2], id="newer-test"),
    ],
)
def test_gitlab_initializer_job_fails_when_a_command_it_runs_fails(kernel, failing, calls_made, tmp_path):
    result, calls = _run_gitlab_initializer_job(tmp_path, kernel, failing)
    assert result.returncode != 0
    assert calls == calls_made


@pytest.mark.parametrize("kernel", ["", "unknown", "5", "5x.3", "five.thirteen"])
def test_gitlab_initializer_job_refuses_a_kernel_version_it_cannot_read(kernel, tmp_path):
    result, calls = _run_gitlab_initializer_job(tmp_path, kernel)
    assert result.returncode != 0
    assert calls == []


def _github_initializer_script(workflow_text):
    """The shell text the confined container runs and the variables its options set."""
    job = yaml.safe_load(workflow_text)["jobs"]["processing-bootstrap"]
    commands = [step["run"] for step in job["steps"] if "cargo " in step.get("run", "")]
    assert len(commands) == 1
    tokens = shlex.split(commands[0].replace("\\\n", " "))
    assert tokens[:2] == ["docker", "run"]
    image = next(i for i, token in enumerate(tokens) if token.startswith("public.ecr.aws/"))
    assert tokens[image + 1 : -1] == ["sh", "-ec"], "the shell must stop at the first failing command"
    variables = dict(value.split("=", 1) for value in _option_values(tokens[2:image], "-e"))
    return tokens[-1], variables


def _require_github_initializer_script_lines(workflow_text):
    script, _variables = _github_initializer_script(workflow_text)
    assert [line.strip() for line in script.strip().splitlines()] == GITHUB_INITIALIZER_SCRIPT


def _run_github_initializer_script(tmp_path, workflow_text, kernel, failing="-"):
    """Run the container's shell text with stand-ins for uname and cargo; return the result and cargo calls."""
    script, variables = _github_initializer_script(workflow_text)
    tmp_path.mkdir(parents=True)
    tools, log = _stand_in_tools(tmp_path)
    result = subprocess.run(
        ["sh", "-ec", script],
        cwd=tmp_path,
        env={
            **variables,
            "TMPDIR": str(tmp_path),
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "STAND_IN_KERNEL": kernel,
            "STAND_IN_LOG": str(log),
            "STAND_IN_FAILS": failing,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, log.read_text().splitlines() if log.exists() else []


def _require_github_initializer_script_behaviour(workflow_text, tmp_path):
    """Every command on every kernel, and each kind of command able to fail the job."""
    for number, kernel in enumerate(OLDER_AND_NEWER_KERNELS):
        result, calls = _run_github_initializer_script(tmp_path / f"kernel-{number}", workflow_text, kernel)
        assert result.returncode == 0, result.stderr
        assert calls == ALL_INITIALIZER_CALLS, f"kernel {kernel}"
        assert result.stdout.splitlines() == [kernel]
    for failing, calls_made in (("fmt", 1), ("test", 2), ("build", 3)):
        result, calls = _run_github_initializer_script(
            tmp_path / f"failing-{failing}", workflow_text, OLDER_AND_NEWER_KERNELS[-1], failing
        )
        assert result.returncode != 0, f"a failing cargo {failing} must fail the job"
        assert calls == ALL_INITIALIZER_CALLS[:calls_made]


def test_github_initializer_container_keeps_its_commands_unchanged():
    _require_github_initializer_script_lines(WORKFLOW.read_text())


def test_github_initializer_container_runs_the_tests_on_every_kernel_and_fails_with_them(tmp_path):
    """The GitLab job leaves the boundary tests to this job on an older kernel, so this job may never skip them."""
    _require_github_initializer_script_behaviour(WORKFLOW.read_text(), tmp_path)


DEFAULT_TEST_LINE = "              cargo test --workspace --locked --offline\n"
NATIVE_TEST_LINE = "              cargo test --workspace --locked --offline --features sagemaker-native-v1\n"
FORMAT_WORDS = "uname -r && cargo fmt --all --check"
KERNEL_SKIP = "uname -r | grep -q '^[0-4]\\.' && exit 0; cargo fmt --all --check"


@pytest.mark.parametrize(
    ("original", "mutated", "seen_by_a_run"),
    [
        pytest.param(DEFAULT_TEST_LINE, "", True, id="default-tests-removed"),
        pytest.param(
            DEFAULT_TEST_LINE,
            DEFAULT_TEST_LINE.replace("\n", " || true\n"),
            True,
            id="default-tests-may-fail",
        ),
        pytest.param(FORMAT_WORDS, KERNEL_SKIP, True, id="older-kernel-skips-everything"),
        pytest.param(NATIVE_TEST_LINE, "", True, id="native-tests-removed"),
        pytest.param(
            NATIVE_TEST_LINE, NATIVE_TEST_LINE.replace("\n", " || true\n"), False, id="native-tests-may-fail"
        ),
        pytest.param(
            f"sh -ec '\n              {FORMAT_WORDS}",
            f"sh -c '\n              {FORMAT_WORDS}",
            False,
            id="shell-does-not-stop-at-a-failure",
        ),
    ],
)
def test_each_github_initializer_command_is_checked_by_itself(original, mutated, seen_by_a_run, tmp_path):
    text = WORKFLOW.read_text()
    assert text.count(original) == 1, "the real command must be present for the mutation to mean anything"
    changed = text.replace(original, mutated)
    assert changed != text
    with pytest.raises(AssertionError):
        _require_github_initializer_script_lines(changed)
    if seen_by_a_run:
        with pytest.raises(AssertionError):
            _require_github_initializer_script_behaviour(changed, tmp_path)


def _require_unconditional_release_jobs(workflow_text):
    """A job or step with a condition, or one allowed to fail, can leave its check undone under a green result."""
    jobs = yaml.safe_load(workflow_text)["jobs"]
    assert sorted(jobs) == ["processing-bootstrap", "verify"]
    for name, job in jobs.items():
        assert "if" not in job, f"job {name} is conditional"
        assert not job.get("continue-on-error", False), f"job {name} may fail"
        for step in job["steps"]:
            assert "if" not in step, f"a step of {name} is conditional: {step}"
            assert not step.get("continue-on-error", False), f"a step of {name} may fail: {step}"


def test_no_release_job_or_step_is_conditional_or_allowed_to_fail():
    _require_unconditional_release_jobs(WORKFLOW.read_text())


PYTEST_STEP = "      - run: uv run --frozen --no-sync python -m pytest -q tests\n"


@pytest.mark.parametrize(
    ("original", "mutated"),
    [
        pytest.param(PYTEST_STEP, PYTEST_STEP + "        if: false\n", id="pytest-step-disabled"),
        pytest.param(
            PYTEST_STEP,
            PYTEST_STEP + "        if: ${{ github.event_name == 'push' }}\n",
            id="pytest-step-on-push-only",
        ),
        pytest.param(
            PYTEST_STEP, PYTEST_STEP + "        continue-on-error: true\n", id="pytest-step-may-fail"
        ),
        pytest.param(
            "  verify:\n    runs-on: ubuntu-latest\n",
            "  verify:\n    if: false\n    runs-on: ubuntu-latest\n",
            id="verify-job-disabled",
        ),
        pytest.param(
            "  processing-bootstrap:\n    runs-on: ubuntu-latest\n",
            "  processing-bootstrap:\n    if: false\n    runs-on: ubuntu-latest\n",
            id="initializer-job-disabled",
        ),
    ],
)
def test_each_release_condition_is_checked_by_itself(original, mutated):
    text = WORKFLOW.read_text()
    assert text.count(original) == 1, "the real step or job must be present for the mutation to mean anything"
    _require_unconditional_release_jobs(text)
    with pytest.raises(AssertionError):
        _require_unconditional_release_jobs(text.replace(original, mutated))
