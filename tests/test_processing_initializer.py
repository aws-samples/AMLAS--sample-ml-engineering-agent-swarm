"""Fixed privileged startup contracts; no cloud or network calls."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from test_managed_dataset_acquisition import adapter, ownership, submit  # noqa: F401

from amlas.sagemaker_entrypoints import dataset_acquisition as worker


@pytest.mark.usefixtures("ownership")
def test_native_request_uses_fixed_initializer():
    backend = adapter()
    submit(backend)
    assert backend._sm.request["AppSpecification"]["ContainerEntrypoint"] == [
        "/usr/local/bin/amlas-processing-init",
        "acquisition",
    ]
    assert "--output-dir" not in backend._sm.request["AppSpecification"]["ContainerArguments"]


def test_root_worker_refuses_before_argument_or_core_import(monkeypatch):
    monkeypatch.setattr(worker.os, "getresuid", lambda: (0, 0, 0))
    monkeypatch.setattr(worker.os, "getresgid", lambda: (0, 0, 0))
    parser = Mock(side_effect=AssertionError("argument parsing ran before identity guard"))
    core = Mock(side_effect=AssertionError("core ran as root"))
    monkeypatch.setattr(worker.argparse, "ArgumentParser", parser)
    monkeypatch.setattr(worker, "_acquire_local", core)
    with pytest.raises(RuntimeError, match="worker identity"):
        worker.main()
    parser.assert_not_called()
    core.assert_not_called()


def test_processing_image_uses_trusted_bootstrap_and_fixed_worker():
    source = (Path(__file__).parents[1] / "Dockerfile.processing").read_text()
    assert 'ENTRYPOINT ["/usr/local/bin/amlas-processing-init", "acquisition"]' in source
    assert "USER root" in source
    assert "COPY --chown=10001:10001 src/" not in source
    # Official public Rust compiler image digest, not a credential.
    assert (
        # pragma: allowlist nextline secret
        "ee9b9f2ebd8741e4ecf1ade08918e492d642c4e625be481bb7b7954b89e9d504" in source
    )
    assert "x86_64-unknown-linux-musl" in source


def test_no_runtime_code_or_path_override():
    source = (Path(__file__).parents[1] / "processing_bootstrap/src/main.rs").read_text()
    assert '"/app/acquisition_worker.py"' in source
    assert '"/app/inspect_dataset.py"' in source
    assert '"/opt/ml/processing"' in source
    assert '"output"' in source
    assert "assert!(" not in source


@pytest.mark.usefixtures("ownership")
def test_native_readback_refuses_bypassed_initializer():
    from amlas.backends.types import JobState

    backend = adapter()
    handle = submit(backend)
    backend._sm.request["AppSpecification"]["ContainerEntrypoint"] = ["python", "-c", "pass"]
    assert backend.status(handle).state == JobState.UNKNOWN
