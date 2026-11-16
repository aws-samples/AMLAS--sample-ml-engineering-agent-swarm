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
