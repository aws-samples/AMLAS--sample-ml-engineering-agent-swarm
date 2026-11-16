"""Production image fixed request interfaces; no builds or AWS writes."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_preparation_image_is_fixed_native_data_guest():
    image = ROOT / "Dockerfile.preparation"
    assert image.is_file()
    source = image.read_text()
    assert 'ENTRYPOINT ["/usr/local/bin/amlas-processing-init", "preparation"]' in source
    assert "--features sagemaker-native-v1" in source and "USER root" in source
    assert "processing_bootstrap/processing_identity.py /app/processing_identity.py" in source
    assert "src/amlas/sagemaker_entrypoints/coding_preparation.py /app/coding_preparation_worker.py" in source
    assert "src/amlas/data_eng/coding_preparation.py /app/coding_preparation_core.py" in source
    assert "torch" not in source and "boto3" not in source and "pandas" not in source
    assert "ENV AMLAS_SOURCE_COMMIT=$AMLAS_SOURCE_COMMIT" in source


def test_preparation_image_has_only_locked_parquet_parser_and_fixed_codec():
    source = (ROOT / "Dockerfile.preparation").read_text()
    assert "COPY uv.lock /app/uv.lock" in source
    assert 'todo=["pyarrow"]' in source
    assert "--only-binary=:all: --require-hashes" in source
    assert "src/amlas/data_eng/__init__.py" not in source
    for name in ("coding_preparation", "open_swe_canonical", "open_swe_derived"):
        assert f"COPY src/amlas/data_eng/{name}.py /app/preparation_profiles/{name}.py" in source
    assert "/app/preparation_profiles/__init__.py" in source
    initializer = (ROOT / "processing_bootstrap/src/main.rs").read_text()
    for name in ("__init__", "coding_preparation", "open_swe_canonical", "open_swe_derived"):
        assert f'fixed_root_path("/app/preparation_profiles/{name}.py", false)?' in initializer


def test_preparation_recipe_pins_local_fixed_installer_and_removes_runtime_tools():
    """Image recipe fixes installer execution and excludes final installer paths."""
    source = (ROOT / "Dockerfile.preparation").read_text()
    checksum = "931c303696af6fa3417112103b1cad26890e5a07eccb5b99783700e33f2b8aad"  # pragma: allowlist secret
    assert "ADD --checksum=sha256:" + checksum in source
    assert "pip-26.2-py3-none-any.whl" in source
    assert "len(raw)==1816475" in source
    assert "--no-index --no-deps --only-binary=:all: --require-hashes" in source
    assert "--mount=type=bind,from=installer-bootstrap" in source and "readonly" in source
    verified = source.index('assert m.version("pip")=="26.2"')
    removed_tools = source.index("pip uninstall --yes setuptools wheel")
    locked_parser = source.index(
        "pip install --no-cache-dir --only-binary=:all: --require-hashes -r /app/requirements.preparation.txt"
    )
    removed_pip = source.index("pip uninstall --yes pip")
    assert verified < removed_tools < locked_parser < removed_pip
    assert "all(u.find_spec(n) is None for n in (" in source
    assert 'todo=["pyarrow"]' in source
    assert "USER root" in source
    assert 'ENTRYPOINT ["/usr/local/bin/amlas-processing-init", "preparation"]' in source
