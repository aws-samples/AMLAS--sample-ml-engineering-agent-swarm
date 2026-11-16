"""Fixed case image carries only reviewed data modules and root-owned wrapper."""

from pathlib import Path


def test_case_image_is_fixed_minimal_data_only():
    root = Path(__file__).resolve().parents[1]
    text = (root / "Dockerfile.case-freeze").read_text()
    assert "--features sagemaker-native-v1" in text and "--locked --offline" in text
    assert "--require-hashes" in text and 'todo=["pyarrow"]' in text
    assert "COPY src/amlas/ /" not in text and "pip install boto3" not in text
    for name in (
        "backends/types.py",
        "coding_case_codec.py",
        "coding_case_manifest.py",
        "coding_case_managed.py",
        "coding_case_profile.py",
        "inspection_schema.py",
    ):
        assert name in text
    assert 'ENTRYPOINT ["/usr/local/bin/amlas-processing-init", "case-freeze"]' in text
    wrapper = (root / "scripts/coding_case_worker_native.py").read_text()
    assert "case_worker_pkg.sagemaker_entrypoints.coding_case_freeze" in wrapper
    assert "boto3" not in wrapper and "pickle" not in wrapper
