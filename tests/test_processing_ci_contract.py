"""Require the security initializer's own toolchain checks in each source CI."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _builder_image() -> str:
    dockerfile = (ROOT / "Dockerfile.processing").read_text()
    match = re.search(r"^FROM --platform=linux/amd64 (\S+) AS initializer$", dockerfile, re.M)
    assert match is not None
    image = match.group(1)
    assert re.fullmatch(r"public\.ecr\.aws/docker/library/rust:[^@\s]+@sha256:[a-f0-9]{64}", image)
    return image


def _second_source_image() -> str:
    """The recipe's compiler image on the second registry that serves it: same repository, tag and digest."""
    first = _builder_image()
    mirror = "public.ecr.aws/docker/library/"
    assert first.startswith(mirror)
    return "docker.io/library/" + first[len(mirror) :]


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


IMAGE_VARIABLE = "PINNED_COMPILER_IMAGE"
FETCHED_IMAGE = '"${PINNED_COMPILER_IMAGE:?}"'


def _container_options(run: str) -> str:
    """The options of a container step: its text between `docker run` and the image."""
    return run.split("docker run", 1)[1].split(FETCHED_IMAGE, 1)[0]


def _require_no_image_by_another_route(workflow: dict) -> None:
    """Close the places where the workflow could name an image that the search for digests cannot see.

    An image named by tag alone carries no digest. A job container, a service, a step that is itself an
    image or a docker command in another job would run one and leave the digest-pinned references as they are.
    """
    for name, job in workflow["jobs"].items():
        assert "container" not in job, f"job {name} runs inside an image that the fetch step did not fetch"
        assert "services" not in job, f"job {name} starts a service image"
        for step in job["steps"]:
            assert not str(step.get("uses", "")).startswith("docker://"), (
                f"a step of job {name} is itself an image: {step}"
            )
            if name != "processing-bootstrap":
                assert not re.search(r"\bdocker\b", step.get("run", "")), (
                    f"only processing-bootstrap starts docker, and a step of job {name} names it: {step}"
                )


def _require_confined_pinned_compiler(workflow_text: str) -> None:
    workflow = yaml.safe_load(workflow_text)
    job = workflow["jobs"].get("processing-bootstrap")
    assert isinstance(job, dict), "the public release must include the initializer checks"
    assert not job.get("continue-on-error", False)
    steps = job["steps"]
    assert all(not step.get("continue-on-error", False) for step in steps)
    checkout = next(step["uses"] for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    assert re.fullmatch(r"actions/checkout@[a-f0-9]{40}", checkout)
    runs = [step["run"] for step in steps if "run" in step]
    commands = "\n".join(runs)
    assert _builder_image() in commands
    references = re.findall(r"[^\s\"']+@sha256:[a-f0-9]{64}", commands)
    assert references == [_builder_image(), _second_source_image()], (
        "the job names the recipe's compiler image on its two sources, the first source first, and no other image"
    )
    assert all(reference in runs[0] for reference in references), "the first step fetches the image"
    assert re.findall(r"[^\s\"']+@sha256:[a-f0-9]{64}", workflow_text) == references, (
        "no other part of the workflow names an image by digest"
    )
    _require_no_image_by_another_route(workflow)
    docker_commands = [re.findall(r"\bdocker\s+(\S+)", run) for run in runs]
    assert [found for found in docker_commands if found] == [["pull"], ["run"], ["run"]], (
        "the job starts docker three times: one pull in the fetch step and one run in each container step"
    )
    assert len(re.findall(r"\bdocker\b", workflow_text)) == 5, (
        "the workflow text holds the word docker five times: in the three docker starts and in the two image "
        "references of the fetch step; one more, in a quoted command name or a shell key too, could start an image"
    )
    containers = [run for run in runs if "docker run" in run]
    assert len(containers) == 2 and all(run.count(f"{FETCHED_IMAGE} sh -ec") == 1 for run in containers), (
        "each container step runs the reference that the fetch step handed on"
    )
    assert workflow_text.count(IMAGE_VARIABLE) == 3, (
        "only the fetch step sets PINNED_COMPILER_IMAGE: the workflow text names the variable three times, "
        "in the write of the fetch step and in the read of each container step, and in no env entry"
    )
    assert commands.count("GITHUB_ENV") == 1, (
        "only the fetch step writes to the job environment: the run text of the job names GITHUB_ENV once"
    )
    assert "--network none" in commands
    assert all("--platform linux/amd64" in _container_options(run) for run in containers), (
        "each container step names its platform among its options; the pull of the fetch step does not stand in"
    )
    assert '"/work:ro"' in commands
    assert "if" not in job and all("if" not in step for step in steps), "the public checks are unconditional"
    assert not re.search(r"\bexit\b", commands), (
        "the public job has no early exit: it runs the tests on any kernel"
    )
    _require_checks(commands)


def test_public_release_checks_initializer_in_a_confined_pinned_compiler():
    _require_confined_pinned_compiler((ROOT / ".github/workflows/public-release.yml").read_text())


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda text: text.replace(_second_source_image(), _second_source_image().split("@")[0]),
            id="second-source-by-tag-alone",
        ),
        pytest.param(
            lambda text: text.replace(_second_source_image(), _second_source_image()[:-4] + "0000"),
            id="second-source-with-another-digest",
        ),
        pytest.param(
            lambda text: text.replace(
                _second_source_image(), _second_source_image().replace("docker.io/", "registry.example/")
            ),
            id="second-source-on-another-registry",
        ),
        pytest.param(
            lambda text: text.replace(
                f"{FETCHED_IMAGE} sh -ec '\n              uname -r",
                f"{_builder_image()} sh -ec '\n              uname -r",
            ),
            id="check-step-names-its-own-image",
        ),
    ],
)
def test_a_public_release_compiler_image_outside_the_two_pinned_sources_is_noticed(mutate):
    text = (ROOT / ".github/workflows/public-release.yml").read_text()
    _require_confined_pinned_compiler(text)
    changed = mutate(text)
    assert changed != text, "the real text must be present for the mutation to mean anything"
    with pytest.raises(AssertionError):
        _require_confined_pinned_compiler(changed)


PREPARATION_CONTAINER = "          docker run --rm --platform linux/amd64 \\\n"
CHECK_CONTAINER = "          docker run --rm --network none --platform linux/amd64 \\\n"


def _without_platform(text: str, *container_lines: str) -> str:
    for line in container_lines:
        assert text.count(line) == 1, (
            "the real container line must be present for the mutation to mean anything"
        )
        text = text.replace(line, line.replace(" --platform linux/amd64", ""))
    return text


CHECK_SCRIPT_START = "              uname -r && "


def _with_platform_in_the_check_script(text: str) -> str:
    assert text.count(CHECK_SCRIPT_START) == 1, (
        "the real script line must be present for the mutation to mean anything"
    )
    return text.replace(CHECK_SCRIPT_START, "              echo --platform linux/amd64 && uname -r && ")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda text: _without_platform(text, PREPARATION_CONTAINER), id="preparation-step"),
        pytest.param(lambda text: _without_platform(text, CHECK_CONTAINER), id="check-step"),
        pytest.param(
            lambda text: _without_platform(text, PREPARATION_CONTAINER, CHECK_CONTAINER),
            id="both-container-steps",
        ),
        pytest.param(
            lambda text: _with_platform_in_the_check_script(_without_platform(text, CHECK_CONTAINER)),
            id="check-step-names-it-in-the-container-script-only",
        ),
    ],
)
def test_a_container_step_without_its_platform_is_noticed(mutate):
    """The fetch step's pull names the platform too; that does not stand in for a container step."""
    text = (ROOT / ".github/workflows/public-release.yml").read_text()
    _require_confined_pinned_compiler(text)
    changed = mutate(text)
    assert changed != text, "the real text must be present for the mutation to mean anything"
    with pytest.raises(AssertionError, match="names its platform"):
        _require_confined_pinned_compiler(changed)


SAMPLE_IMAGE = "registry.example/sample:1"
INITIALIZER_JOB_HEADER = "  processing-bootstrap:\n    runs-on: ubuntu-latest\n"
VERIFY_JOB_HEADER = "  verify:\n    runs-on: ubuntu-latest\n"
VERIFY_STEP = '      - run: mkdir -p "$TMPDIR"\n'
PREPARATION_STEP_HEADER = "      - name: Prepare the pinned compiler and formatter in workspace tmp\n"
PREPARATION_FIRST_LINE = '          mkdir -p "$TMPDIR"\n'
JOB_CONTAINER = f"    container: {SAMPLE_IMAGE}\n"
JOB_SERVICE = f"    services:\n      sample:\n        image: {SAMPLE_IMAGE}\n"


@pytest.mark.parametrize(
    ("original", "mutated", "message"),
    [
        pytest.param(
            VERIFY_JOB_HEADER,
            VERIFY_JOB_HEADER + JOB_CONTAINER,
            "runs inside an image",
            id="another-job-in-a-container",
        ),
        pytest.param(
            VERIFY_JOB_HEADER,
            VERIFY_JOB_HEADER + JOB_SERVICE,
            "starts a service image",
            id="another-job-with-a-service",
        ),
        pytest.param(
            INITIALIZER_JOB_HEADER,
            INITIALIZER_JOB_HEADER + JOB_CONTAINER,
            "runs inside an image",
            id="initializer-job-in-a-container",
        ),
        pytest.param(
            INITIALIZER_JOB_HEADER,
            INITIALIZER_JOB_HEADER + JOB_SERVICE,
            "starts a service image",
            id="initializer-job-with-a-service",
        ),
        pytest.param(
            VERIFY_STEP,
            VERIFY_STEP + f"      - uses: docker://{SAMPLE_IMAGE}\n",
            "is itself an image",
            id="a-step-that-is-an-image",
        ),
        pytest.param(
            VERIFY_STEP,
            VERIFY_STEP + f"      - run: docker run --rm {SAMPLE_IMAGE} true\n",
            "only processing-bootstrap starts docker",
            id="another-job-runs-an-image",
        ),
        pytest.param(
            PREPARATION_STEP_HEADER,
            f"      - run: docker image pull {SAMPLE_IMAGE}\n" + PREPARATION_STEP_HEADER,
            "starts docker three times",
            id="initializer-job-with-a-further-docker-step",
        ),
        pytest.param(
            PREPARATION_FIRST_LINE,
            PREPARATION_FIRST_LINE + f"          docker create {SAMPLE_IMAGE}\n",
            "starts docker three times",
            id="container-step-with-a-second-docker-command",
        ),
        pytest.param(
            PREPARATION_STEP_HEADER,
            f'      - run: |\n          "docker" run --rm {SAMPLE_IMAGE} true\n' + PREPARATION_STEP_HEADER,
            "holds the word docker five times",
            id="initializer-job-step-with-a-quoted-command-name",
        ),
        pytest.param(
            VERIFY_STEP,
            VERIFY_STEP
            + f'      - run: echo "checks start"\n        shell: docker run --rm -i {SAMPLE_IMAGE} sh {{0}}\n',
            "holds the word docker five times",
            id="another-job-step-with-a-shell-key-that-starts-docker",
        ),
        pytest.param(
            f"{FETCHED_IMAGE} sh -ec '\n              mkdir -p /worktmp/rustup\n",
            f"{SAMPLE_IMAGE} sh -ec '\n              mkdir -p /worktmp/rustup\n",
            "runs the reference that the fetch step handed on",
            id="container-step-runs-an-image-by-tag",
        ),
    ],
)
def test_an_image_named_outside_the_fetch_step_of_the_public_workflow_is_noticed(original, mutated, message):
    """An image named by tag alone carries no digest, so the search for digest-pinned references cannot see it."""
    text = (ROOT / ".github/workflows/public-release.yml").read_text()
    assert text.count(original) == 1, "the real text must be present for the mutation to mean anything"
    _require_confined_pinned_compiler(text)
    with pytest.raises(AssertionError, match=message):
        _require_confined_pinned_compiler(text.replace(original, mutated))


def test_an_image_by_digest_outside_the_initializer_job_is_noticed():
    """The first search reads the run text of one job; the second reads the whole workflow text."""
    text = (ROOT / ".github/workflows/public-release.yml").read_text()
    assert text.count(VERIFY_STEP) == 1, "the real text must be present for the mutation to mean anything"
    _require_confined_pinned_compiler(text)
    changed = text.replace(
        VERIFY_STEP, VERIFY_STEP + f'      - run: echo "{SAMPLE_IMAGE}@sha256:{"0" * 64}"\n'
    )
    with pytest.raises(AssertionError, match="no other part of the workflow names an image by digest"):
        _require_confined_pinned_compiler(changed)


CHECK_STEP_HEADER = "      - name: Check and build the static initializer without network access\n"
INITIALIZER_JOB_ENV = "    env:\n      TMPDIR: ${{ github.workspace }}/../amlas-processing-ci\n"
JOBS_HEADER = "\njobs:\n"
ONE_SETTER = "only the fetch step sets PINNED_COMPILER_IMAGE"


@pytest.mark.parametrize(
    ("original", "mutated", "message"),
    [
        pytest.param(
            CHECK_STEP_HEADER,
            CHECK_STEP_HEADER + f"        env:\n          PINNED_COMPILER_IMAGE: {SAMPLE_IMAGE}\n",
            ONE_SETTER,
            id="a-container-step-env-entry",
        ),
        pytest.param(
            INITIALIZER_JOB_ENV,
            INITIALIZER_JOB_ENV + f"      PINNED_COMPILER_IMAGE: {SAMPLE_IMAGE}\n",
            ONE_SETTER,
            id="a-job-env-entry",
        ),
        pytest.param(
            JOBS_HEADER,
            f"\nenv:\n  PINNED_COMPILER_IMAGE: {SAMPLE_IMAGE}\n" + JOBS_HEADER,
            ONE_SETTER,
            id="a-workflow-env-entry",
        ),
        pytest.param(
            PREPARATION_STEP_HEADER,
            f'      - run: echo "PINNED_COMPILER_IMAGE={SAMPLE_IMAGE}" >> "$GITHUB_ENV"\n'
            + PREPARATION_STEP_HEADER,
            ONE_SETTER,
            id="another-step-writes-the-variable",
        ),
        pytest.param(
            PREPARATION_STEP_HEADER,
            '      - run: cat "$TMPDIR/settings" >> "$GITHUB_ENV"\n' + PREPARATION_STEP_HEADER,
            "only the fetch step writes to the job environment",
            id="another-step-writes-a-file-to-the-job-environment",
        ),
    ],
)
def test_a_second_place_that_sets_the_handed_on_image_variable_is_noticed(original, mutated, message):
    """The container steps run what the variable holds, so only the fetch step may set it."""
    text = (ROOT / ".github/workflows/public-release.yml").read_text()
    assert text.count(original) == 1, "the real text must be present for the mutation to mean anything"
    _require_confined_pinned_compiler(text)
    with pytest.raises(AssertionError, match=message):
        _require_confined_pinned_compiler(text.replace(original, mutated))


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


PYTHON_JOB_IMAGE = r"public\.ecr\.aws/docker/library/python:3\.11-slim@sha256:[a-f0-9]{64}"
DOX_JOB_HEADER = "\ndox:\n  stage: dox\n"


def _require_gitlab_images_by_digest(pipeline_text: str) -> None:
    pipeline = yaml.safe_load(pipeline_text)
    default = pipeline["default"]["image"]
    assert re.fullmatch(PYTHON_JOB_IMAGE, default), (
        "the default job image needs its index digest: a tag alone can move to other content "
        "between two runs of one commit"
    )
    images = [entry["image"] for entry in pipeline.values() if isinstance(entry, dict) and "image" in entry]
    assert sorted(images) == sorted([default, _builder_image()]), (
        "the pipeline names two images, the default image and the compiler image, each with its digest"
    )
    assert "image" not in pipeline and "services" not in pipeline, (
        "the pipeline names no image or service outside the default and the jobs"
    )
    assert all("services" not in entry for entry in pipeline.values() if isinstance(entry, dict)), (
        "neither the default nor a job starts a service image"
    )


def test_gitlab_jobs_name_their_images_by_digest():
    _require_gitlab_images_by_digest((ROOT / ".gitlab-ci.yml").read_text())


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda text, image: text.replace(image, image.split("@")[0]), id="tag-alone"),
        pytest.param(lambda text, image: text.replace(image, image[:-1]), id="digest-one-character-short"),
        pytest.param(
            lambda text, image: text.replace(image, image.split("@")[0] + "@sha256:" + "g" * 64),
            id="digest-with-other-characters",
        ),
        pytest.param(
            lambda text, image: text.replace(
                DOX_JOB_HEADER, DOX_JOB_HEADER + "  image: " + image.split("@")[0] + "\n"
            ),
            id="a-job-gains-an-image-named-by-tag",
        ),
    ],
)
def test_a_gitlab_job_image_without_its_full_digest_is_noticed(mutate):
    text = (ROOT / ".gitlab-ci.yml").read_text()
    pinned = re.search(rf"^  image: ({PYTHON_JOB_IMAGE})$", text, re.M)
    assert pinned is not None, "the real digest must be present for the mutation to mean anything"
    assert text.count(DOX_JOB_HEADER) == 1, (
        "the real job header must be present for the mutation to mean anything"
    )
    _require_gitlab_images_by_digest(text)
    changed = mutate(text, pinned.group(1))
    assert changed != text
    with pytest.raises(AssertionError):
        _require_gitlab_images_by_digest(changed)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda text: text.replace(
                DOX_JOB_HEADER, DOX_JOB_HEADER + f"  services:\n    - {SAMPLE_IMAGE}\n"
            ),
            "starts a service image",
            id="a-job-with-a-service",
        ),
        pytest.param(
            lambda text: text.replace("default:\n", f"default:\n  services:\n    - {SAMPLE_IMAGE}\n", 1),
            "starts a service image",
            id="the-default-with-a-service",
        ),
        pytest.param(
            lambda text: text + f"\nservices:\n  - {SAMPLE_IMAGE}\n",
            "outside the default and the jobs",
            id="a-service-for-the-whole-pipeline",
        ),
        pytest.param(
            lambda text: text + f"\nimage: {SAMPLE_IMAGE}\n",
            "outside the default and the jobs",
            id="an-image-for-the-whole-pipeline",
        ),
    ],
)
def test_a_gitlab_image_outside_the_two_pinned_images_is_noticed(mutate, message):
    """A service, or an image for the whole pipeline, runs without changing the two image entries."""
    text = (ROOT / ".gitlab-ci.yml").read_text()
    assert text.startswith("default:\n") and text.count(DOX_JOB_HEADER) == 1, (
        "the real entries must be present for the mutation to mean anything"
    )
    _require_gitlab_images_by_digest(text)
    changed = mutate(text)
    assert changed != text
    with pytest.raises(AssertionError, match=message):
        _require_gitlab_images_by_digest(changed)


OWN_CHECK_ADVICE = (
    "no dependency check reads processing_bootstrap/Cargo.lock, "
    "so the first crate from a registry must come with a check of its own"
)


def _dependency_tables(table: dict, prefix: str = "") -> list[str]:
    """Name every table of a Cargo manifest that can bring in another crate, at any depth."""
    found = []
    for key, value in table.items():
        name = f"{prefix}{key}"
        if "dependencies" in key or name in ("patch", "replace"):
            found.append(name)
        elif isinstance(value, dict):
            found += _dependency_tables(value, f"{name}.")
    return found


def _require_initializer_without_dependencies(root: Path) -> None:
    manifest = tomllib.loads((root / "processing_bootstrap" / "Cargo.toml").read_text())
    lock = tomllib.loads((root / "processing_bootstrap" / "Cargo.lock").read_text())
    tables = _dependency_tables(manifest)
    assert not tables, f"Cargo.toml has a dependency table ({', '.join(tables)}); {OWN_CHECK_ADVICE}"
    own = {"name": manifest["package"]["name"], "version": manifest["package"]["version"]}
    assert lock.get("package") == [own], (
        f"Cargo.lock names a package other than the crate itself; {OWN_CHECK_ADVICE}"
    )


def test_initializer_manifest_has_no_dependency_table_and_its_lock_names_only_the_crate():
    """The pipelines check Python pins only; this holds the state in which the crate has nothing to check."""
    _require_initializer_without_dependencies(ROOT)


REGISTRY_CRATE = """
[[package]]
name = "sample-crate"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "0000000000000000000000000000000000000000000000000000000000000000"
"""
LOCAL_CRATE = """
[[package]]
name = "local-crate"
version = "0.1.0"
"""


@pytest.mark.parametrize(
    ("file", "addition"),
    [
        pytest.param("Cargo.lock", REGISTRY_CRATE, id="lock-gains-a-registry-crate"),
        pytest.param("Cargo.lock", LOCAL_CRATE, id="lock-gains-a-path-crate"),
        pytest.param(
            "Cargo.toml", '\n[dependencies]\nsample-crate = "1"\n', id="manifest-gains-a-dependency"
        ),
        pytest.param(
            "Cargo.toml",
            '\n[dependencies]\nlocal-crate = { path = "../local" }\n',
            id="manifest-gains-a-path-dependency",
        ),
        pytest.param(
            "Cargo.toml",
            '\n[build-dependencies]\nsample-crate = "1"\n',
            id="manifest-gains-a-build-dependency",
        ),
        pytest.param(
            "Cargo.toml", '\n[dev-dependencies]\nsample-crate = "1"\n', id="manifest-gains-a-test-dependency"
        ),
        pytest.param(
            "Cargo.toml",
            '\n[target.x86_64-unknown-linux-musl.dependencies]\nsample-crate = "1"\n',
            id="manifest-gains-a-target-dependency",
        ),
        pytest.param(
            "Cargo.toml",
            '\n[workspace.dependencies]\nsample-crate = "1"\n',
            id="workspace-gains-a-dependency",
        ),
        pytest.param(
            "Cargo.toml",
            '\n[patch.crates-io]\nsample-crate = { path = "../local" }\n',
            id="manifest-gains-a-patch-table",
        ),
    ],
)
def test_a_first_crate_in_the_initializer_is_noticed_and_asks_for_its_own_check(file, addition, tmp_path):
    crate = tmp_path / "processing_bootstrap"
    crate.mkdir()
    for name in ("Cargo.toml", "Cargo.lock"):
        (crate / name).write_text((ROOT / "processing_bootstrap" / name).read_text())
    _require_initializer_without_dependencies(tmp_path)
    with (crate / file).open("a") as handle:
        handle.write(addition)
    with pytest.raises(AssertionError, match="must come with a check of its own"):
        _require_initializer_without_dependencies(tmp_path)
