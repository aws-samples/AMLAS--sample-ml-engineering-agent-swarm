"""Source CI must provision real pinned CPU tests rather than omit them."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workflow", ["gitlab", "github"])
def test_verify_provisions_cpu_and_metadata_before_complete_pytest(workflow):
    if workflow == "gitlab":
        steps = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())["verify"]["script"]
    else:
        job = yaml.safe_load((ROOT / ".github/workflows/public-release.yml").read_text())["jobs"]["verify"]
        steps = [s.get("run", "") for s in job["steps"]]
    text = "\n".join(steps)
    assert "scripts/provision_qwen_cpu_tests.sh" in text, "real Qwen CPU provisioning missing"
    assert "environment.sh" in text, "pinned metadata/offline environment missing"
    assert text.index("scripts/provision_qwen_cpu_tests.sh") < text.index("python -m pytest -q tests")
    assert "python -m pytest -q tests" in text
    assert "--ignore" not in text and "continue-on-error" not in text
    if workflow == "github":
        assert '>> "$GITHUB_ENV"' in text
        assert "AMLAS_QWEN_TEST_METADATA=%s" in text


def test_cpu_lock_is_separate_hashed_complete_and_without_cuda():
    path = ROOT / "scripts/qwen_cpu_test.requirements.lock"
    assert path.is_file(), "CPU lock missing"
    text = path.read_text()
    for package, version in (
        ("transformers", "5.20.0"),
        ("peft", "0.21.2"),
        ("tokenizers", "0.23.1"),
        ("safetensors", "0.8.0"),
    ):
        assert f"{package}=={version}" in text
    assert (
        "--find-links https://download.pytorch.org/whl/cpu/torch/"
        in text
    )
    assert "--hash=sha256:673dbf5c9bbadfffab7a386b6dd7a0c219f1408a328b7b4e86d0ae551cdafa42" in text
    assert not any(n in text.lower() for n in ("nvidia-", "cuda-toolkit", "cuda-bindings", "triton=="))
    assert "/local/home" not in text


def metadata_module():
    path = ROOT / "scripts/qwen_cpu_test_metadata.py"
    assert path.is_file(), "missing-symbol: qwen_cpu_test_metadata"
    spec = importlib.util.spec_from_file_location("qwen_cpu_test_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_manifest_exact_five_source_pinned_files_no_weights():
    m = metadata_module()
    refs = m.metadata_manifest()
    assert {r["path"] for r in refs} == {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    }
    assert all(r["role"] != "weight_shard" and r["byte_count"] > 0 and len(r["sha256"]) == 64 for r in refs)


def test_provisioning_enforces_hash_and_count_before_metadata_success(tmp_path, monkeypatch):
    m = metadata_module()
    import hashlib

    raw = b'{"fixture":true}'
    refs = (
        {
            "path": "config.json",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "role": "config",
        },
    )
    monkeypatch.setattr(m, "metadata_manifest", lambda: refs)
    calls = []

    def fetch(url, stream, **kwargs):
        calls.append((url, kwargs))
        stream.write(raw)

    monkeypatch.setattr(m, "stream_public_binary", fetch)
    result = m.provision_metadata(tmp_path / "positive")
    assert result["weight_shards_downloaded"] == 0
    assert result["files"][0]["sha256"] == refs[0]["sha256"]
    assert "c202236235762e1c871ad0ccb60c8ee5ba337b9a" in calls[0][0]  # pragma: allowlist secret - public Qwen model revision
    assert calls[0][1]["max_bytes"] == len(raw)

    def corrupt(url, stream, **kwargs):
        stream.write(b"x" * len(raw))

    monkeypatch.setattr(m, "stream_public_binary", corrupt)
    with pytest.raises(ValueError, match="integrity"):
        m.provision_metadata(tmp_path / "corrupt")
    assert not (tmp_path / "corrupt" / "METADATA.json").exists()


def test_metadata_refuses_existing_destination_without_overwriting(tmp_path):
    m = metadata_module()
    destination = tmp_path / "foreign"
    destination.mkdir()
    file = destination / "foreign.txt"
    file.write_text("preserve")
    with pytest.raises(FileExistsError):
        m.provision_metadata(destination)
    assert file.read_text() == "preserve"


def test_added_cpu_closure_has_strict_audit_in_both_paths():
    for path in (ROOT / ".gitlab-ci.yml", ROOT / ".github/workflows/public-release.yml"):
        text = path.read_text()
        assert "pip-audit -r scripts/qwen_cpu_test.requirements.lock --strict --disable-pip" in text


def test_cpu_audit_has_explicit_local_version_and_official_hashed_wheel_route():
    text = (ROOT / "scripts/qwen_cpu_test.requirements.lock").read_text()
    assert "torch==2.14.0+cpu" in text, "URL-only CPU requirement cannot be strictly audited"
    assert "torch @ " not in text
    assert (
        "--find-links https://download.pytorch.org/whl/cpu/torch/"
        in text
    )
    for path in (ROOT / ".gitlab-ci.yml", ROOT / ".github/workflows/public-release.yml"):
        assert "--strict --disable-pip --vulnerability-service osv" in path.read_text()
