"""Source image/lock closure oracles; not GPU/native image admission."""

import re
from pathlib import Path


def test_fixed_trainer_image_has_only_os_root_startup_and_locked_local_child():
    path = Path("Dockerfile.qwen_sft")
    assert path.is_file(), "missing trainer Dockerfile"
    text = path.read_text()
    assert "cargo build --release --locked --offline --features sagemaker-native-v1" in text
    assert "USER root" in text
    assert 'ENTRYPOINT ["/usr/local/bin/amlas-processing-init", "qwen-sft"]' in text
    assert "COPY processing_bootstrap/processing_identity.py /app/processing_identity.py" in text
    assert "COPY src/amlas/sagemaker_entrypoints/qwen_sft_train.py /app/qwen_sft_train.py" in text
    assert "--require-hashes" in text and "--only-binary=:all:" in text
    assert "trust_remote_code=True" not in text and "chmod 0777" not in text
    assert "HF_HUB_OFFLINE=1" in text and "TRANSFORMERS_OFFLINE=1" in text


def test_dedicated_gpu_lock_has_fixed_official_native_packages_and_no_agent_sdk():
    path = Path("src/amlas/training/qwen_sft.requirements.lock")
    assert path.is_file(), "missing dedicated GPU lock"
    text = path.read_text()
    for name, version in [
        ("torch", "2.14.0"),
        ("transformers", "5.20.0"),
        ("peft", "0.21.2"),
        ("safetensors", "0.8.0"),
        ("tokenizers", "0.23.1"),
    ]:
        assert re.search(r"^" + re.escape(name) + "==" + re.escape(version) + r"\s+\\", text, re.M)
    assert not re.search(r"^(boto3|botocore|strands-agents|sagemaker|kernels|flash-attn)==", text, re.M)
    assert "--hash=sha256:" in text
    assert "/local/" + "home/" not in text and "/" + "Users/" not in text


def test_docker_declared_pure_namespace_imports_commitpack_replay(tmp_path):
    """Exercise the image's declared pure files, without agent/runtime packages."""
    import shutil
    import subprocess
    import sys

    namespace = tmp_path / "amlas" / "data_eng"
    namespace.mkdir(parents=True)
    (namespace.parent / "__init__.py").write_text('"""Fixed pure namespace."""\n')
    (namespace / "__init__.py").write_text('"""Fixed pure data contracts."""\n')
    declared = []
    for line in Path("Dockerfile.qwen_sft").read_text().splitlines():
        if line.startswith("COPY ") and line.endswith(" /app/src/amlas/data_eng/"):
            declared.extend(line.split()[1:-1])
    for source in declared:
        shutil.copyfile(source, namespace / Path(source).name)
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import sys; sys.path.insert(0,sys.argv[1]); "
         "from amlas.data_eng.commitpack_preparation import replay_commitpack_preparation; "
         "assert callable(replay_commitpack_preparation)", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, "Docker declared pure replay closure cannot import: " + result.stderr


def test_fixed_trainer_image_copies_the_unchanged_stdlib_provider_guard():
    text = Path("Dockerfile.qwen_sft").read_text()
    assert "COPY src/amlas/sagemaker_entrypoints/model_weight_acquisition.py /app/model_worker.py" in text
    worker = Path("src/amlas/sagemaker_entrypoints/qwen_sft_train.py").read_text()
    assert "/app/model_worker.py" in worker


def _recipe_instructions(text):
    """Recipe instructions with their continuation lines joined and comment lines dropped."""
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return [step for step in re.sub(r"\\\n[ \t]*", "", "\n".join(lines)).splitlines() if step.strip()]


def test_fixed_trainer_image_installs_with_the_pinned_installer_and_keeps_no_installer():
    """The base image's installer neither installs the training lock nor stays installed in the image.

    Both installer steps are pinned whole; the guard program's assertions themselves run in the image build.
    """
    text = Path("Dockerfile.qwen_sft").read_text()
    checksum = "931c303696af6fa3417112103b1cad26890e5a07eccb5b99783700e33f2b8aad"  # pragma: allowlist secret
    added = "ADD --checksum=sha256:" + checksum + " https://files.pythonhosted.org/"
    assert added in text and "/bootstrap/pip-26.2-py3-none-any.whl" in text
    for recipe in ("Dockerfile.preparation", "Dockerfile.model-acquisition"):
        assert added in Path(recipe).read_text(), "the three recipes use one installer wheel"
    assert "FROM scratch AS installer-bootstrap" in text
    assert "--mount=type=bind,from=installer-bootstrap" in text and "readonly" in text
    assert "len(raw)==1816475" in text
    bootstrap = text.index(
        "--no-index --no-deps --only-binary=:all: --require-hashes --find-links=/bootstrap"
    )
    verified = text.index('assert m.version("pip")=="26.2"')
    removed_tools = text.index("python -m pip uninstall --yes setuptools wheel")
    locked = text.index(
        "python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /app/qwen_sft.requirements.lock"
    )
    consistent = text.index("python -m pip check &&")
    removed = text.index("python -m pip uninstall --yes pip &&")
    equal = text.index("assert got==sorted((c(n),v) for n,v in rows)")
    absent = text.index('all(u.find_spec(n) is None for n in ("pip","wheel"))')
    assert bootstrap < verified < removed_tools < locked < consistent < removed < equal < absent
    assert 'all(shutil.which(n) is None for n in ("pip","pip3","pip3.11","wheel"))' in text[absent:]
    assert "from packaging.requirements import Requirement as Q" in text[absent:]
    assert (
        'c(q.name) in ("pip","wheel") and (q.marker is None or q.marker.evaluate({"extra":""}))'
        in text[absent:]
    ), (
        "an active requirement on a removed tool fails the build; one under an extra that is not selected does not"
    )
    lock = Path("src/amlas/training/qwen_sft.requirements.lock").read_text()
    assert re.search(r"^packaging==", lock, re.M), (
        "the in-build check reads requirements with the packaging of the lock"
    )
    assert not re.search(r"(?<!python -m )\bpip (install|uninstall|check)\b", text), (
        "every installer step runs the module of the image's interpreter, never the launcher of the base image"
    )
    assert "/app/requirements.pip-bootstrap.txt" not in text, (
        "the bootstrap requirement line is not kept under /app"
    )
    written = text.index("> /tmp/requirements.pip-bootstrap.txt")
    assert written < bootstrap < text.index("rm /tmp/requirements.pip-bootstrap.txt") < verified
    steps = _recipe_instructions(text)
    bootstrap_step = [step for step in steps if "--find-links=/bootstrap" in step]
    guard_step = [step for step in steps if "QWEN_SFT_INSTALLED_EQUALS_LOCK" in step]
    assert len(bootstrap_step) == 1 and len(guard_step) == 1, "one bootstrap step and one guard step"
    lock_steps = [step for step in steps if "qwen_sft.requirements.lock" in step]
    assert lock_steps == [
        "COPY src/amlas/training/qwen_sft.requirements.lock /app/qwen_sft.requirements.lock",
        guard_step[0],
    ], "the lock in the image is the repository's training lock: only its copy and the guard step name it"
    wheel_check = [
        "import hashlib,pathlib",
        'raw=pathlib.Path("/bootstrap/pip-26.2-py3-none-any.whl").read_bytes()',
        'assert len(raw)==1816475 and hashlib.sha256(raw).hexdigest()=="' + checksum + '"',
    ]
    version_check = [
        "import importlib.metadata as m",
        'assert m.version("pip")=="26.2"',
        'print("MODEL_BUILD_INSTALLER_VERSION=26.2")',
    ]
    assert bootstrap_step[0] == " && ".join(
        [
            "RUN --mount=type=bind,from=installer-bootstrap,source=/bootstrap,target=/bootstrap,readonly"
            + " python -c '"
            + "; ".join(wheel_check)
            + "'",
            "printf '%s\\n' 'pip==26.2 --hash=sha256:" + checksum + "' > /tmp/requirements.pip-bootstrap.txt",
            "python -m pip install --no-index --no-deps --only-binary=:all: --require-hashes"
            + " --find-links=/bootstrap -r /tmp/requirements.pip-bootstrap.txt",
            "rm /tmp/requirements.pip-bootstrap.txt",
            "python -c '" + "; ".join(version_check) + "'",
            "python -m pip uninstall --yes setuptools wheel",
        ]
    ), "the bootstrap step is this chain alone; a command that fails stops the step"
    guard_program = [
        "import importlib.util as u,importlib.metadata as m,re,shutil",
        'c=lambda n: re.sub("[-_.]+","-",n).lower()',
        'rows=[l.split()[0].split("==") for l in open("/app/qwen_sft.requirements.lock") if l[:1].isalnum()]',
        'got=sorted((c(d.metadata["Name"]),d.version) for d in m.distributions())',
        'assert got==sorted((c(n),v) for n,v in rows), "installed distributions differ from the lock"',
        'assert all(u.find_spec(n) is None for n in ("pip","wheel")), "pip or wheel is importable"',
        'assert all(shutil.which(n) is None for n in ("pip","pip3","pip3.11","wheel")), '
        + '"a pip or wheel launcher is on PATH"',
        "from packaging.requirements import Requirement as Q",
        "need=[q for q in (Q(r) for d in m.distributions() for r in (d.requires or [])) "
        + 'if c(q.name) in ("pip","wheel") and (q.marker is None or q.marker.evaluate({"extra":""}))]',
        'assert not need, "an installed distribution requires pip or wheel"',
        'print("QWEN_SFT_INSTALLED_EQUALS_LOCK=%d" % len(got))',
    ]
    program = re.search(r"python -c '([^']*QWEN_SFT_INSTALLED_EQUALS_LOCK[^']*)'", guard_step[0])
    assert program and program.group(1).split("; ") == guard_program, (
        "the guard program is these statements alone: the lock side is read from the copied lock, "
        "the installed side from the installed distributions, and every assertion can fail"
    )
    assert guard_step[0] == " && ".join(
        [
            "RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes"
            + " -r /app/qwen_sft.requirements.lock",
            "python -m pip check",
            "python -m pip uninstall --yes pip",
            "python -c '" + "; ".join(guard_program) + "'",
            "useradd --create-home --uid 10001 appuser",
            "mkdir -p /opt/ml/model",
            "chown 10001:10001 /opt/ml/model",
            "chmod 0700 /opt/ml/model",
        ]
    ), "the guard step is this chain alone; a failed assertion of the guard program stops the build"
    later = "\n".join(steps[steps.index(guard_step[0]) + 1 :])
    assert not re.search(r"(?i)(?<![a-z])pip(?![a-z])|installer-bootstrap|^FROM ", later, re.M), (
        "after the guard step no instruction names pip or the bootstrap stage, and no new stage begins"
    )
    assert not re.search(r"(?i)ensurepip|\bvenv\b|virtualenv|PYTHONOPTIMIZE", "\n".join(steps)), (
        "the standard library keeps its bundled installer wheel, and an optimized interpreter skips "
        "assertions: no instruction unpacks the one or switches to the other"
    )
