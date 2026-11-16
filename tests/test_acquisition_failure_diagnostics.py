"""Acquisition failures stay observable without emitting capability or data text."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pytest

from amlas.data_eng import hub_dataset as core
from amlas.sagemaker_entrypoints import dataset_acquisition as entrypoint

REPO = "fixture/open-data"
REVISION = "a" * 40
PAYLOAD = b"value\n1\n"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
PRIVATE_DETAIL = (
    "https://cdn.example/private?X-Amz-Signature=" + "capability-fragment" + "\nprivate-row-content"
)


@pytest.fixture
def guarded_fixture(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(core.pathref, "remote_fetch_enabled", lambda: True)
    monkeypatch.setattr(core.pathref, "_allowed_roots", lambda: [tmp_path])
    monkeypatch.setattr(
        core,
        "_metadata",
        lambda *args: {
            "id": REPO,
            "sha": REVISION,
            "private": False,
            "gated": False,
            "disabled": False,
            "siblings": [
                {
                    "rfilename": "data/test.csv",
                    "size": len(PAYLOAD),
                    "lfs": {"sha256": DIGEST, "size": len(PAYLOAD)},
                }
            ],
        },
    )

    def stream(url: str, sink: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["expected_size"] == len(PAYLOAD)
        assert kwargs["max_bytes"] == core.MAX_FILE_BYTES
        sink.write(PAYLOAD)
        return {"byte_count": len(PAYLOAD), "sha256": DIGEST, "git_blob_sha1": "b" * 40}

    monkeypatch.setattr(core.public_fetch, "stream_public_binary", stream)
    return tmp_path


def timeout(*args: Any, **kwargs: Any) -> Any:
    try:
        raise TimeoutError(PRIVATE_DETAIL)
    except TimeoutError as error:
        raise core.public_fetch.PublicFetchError("anonymous HTTPS retrieval failed (TimeoutError)") from error


@pytest.mark.parametrize(
    "stage,error_type,cause_type",
    [
        ("validation", "HubDatasetError", None),
        ("output_directory", "PermissionError", None),
        ("metadata", "PublicFetchError", "TimeoutError"),
        ("file_selection", "HubDatasetError", None),
        ("bundle_create", "PermissionError", None),
        ("download", "PublicFetchError", "TimeoutError"),
        ("integrity", "HubDatasetError", None),
        ("lineage", "OSError", None),
        ("commit", "FileExistsError", None),
    ],
)
def test_core_failure_has_safe_stage_and_exception_context(
    guarded_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_type: str,
    cause_type: Any,
) -> None:
    out = guarded_fixture
    revision = REVISION
    if stage == "validation":
        revision = "not-pinned"
    elif stage == "output_directory":

        def denied(*args: Any, **kwargs: Any) -> Any:
            raise PermissionError(PRIVATE_DETAIL)

        monkeypatch.setattr(core, "_output_directory", denied)
    elif stage == "metadata":
        monkeypatch.setattr(core, "_metadata", timeout)
    elif stage == "file_selection":
        monkeypatch.setattr(core, "_files", lambda *args, **kwargs: ([], False))
    elif stage == "bundle_create":
        original = os.mkdir

        def mkdir(path: Any, *args: Any, **kwargs: Any) -> Any:
            if str(path).startswith("acquisition-"):
                raise PermissionError(PRIVATE_DETAIL)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(core.os, "mkdir", mkdir)
    elif stage == "download":
        monkeypatch.setattr(core.public_fetch, "stream_public_binary", timeout)
    elif stage == "integrity":
        original_stream = core.public_fetch.stream_public_binary

        def mismatched(*args: Any, **kwargs: Any) -> Any:
            return {**original_stream(*args, **kwargs), "sha256": "c" * 64}

        monkeypatch.setattr(core.public_fetch, "stream_public_binary", mismatched)
    else:
        original_write = core._write_json

        def write(descriptor: int, name: str, value: dict[str, Any]) -> Any:
            if name == ("lineage.json" if stage == "lineage" else ".commit-pending.json"):
                if stage == "lineage":
                    raise OSError(PRIVATE_DETAIL)
                raise FileExistsError(PRIVATE_DETAIL)
            return original_write(descriptor, name, value)

        monkeypatch.setattr(core, "_write_json", write)
    result = core.acquire_huggingface_dataset_local(REPO, revision, "data/test.csv", output_dir=str(out))
    assert result["success"] is False
    assert result.get("error_stage") == stage
    assert result.get("error_type") == error_type
    assert result.get("error_cause_type") == cause_type
    assert PRIVATE_DETAIL not in json.dumps(result)
    assert list(out.iterdir()) == [], "failed core bundles must remain uncommitted and cleaned"


@pytest.mark.parametrize(
    "context,expected",
    [
        (
            {"error_stage": "metadata", "error_type": "PublicFetchError", "error_cause_type": "TimeoutError"},
            "dataset acquisition failed stage=metadata error_type=PublicFetchError cause_type=TimeoutError",
        ),
        (
            {"error_stage": "bundle_create", "error_type": "PermissionError", "error_cause_type": None},
            "dataset acquisition failed stage=bundle_create error_type=PermissionError",
        ),
        ({}, "dataset acquisition failed stage=unknown error_type=UnexpectedError"),
        (
            {"error_stage": PRIVATE_DETAIL, "error_type": PRIVATE_DETAIL, "error_cause_type": PRIVATE_DETAIL},
            "dataset acquisition failed stage=unknown error_type=UnexpectedError",
        ),
    ],
)
def test_native_entrypoint_preserves_only_safe_failure_context(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    context: Any,
    expected: str,
) -> None:
    monkeypatch.setattr(
        entrypoint,
        "_acquire_local",
        lambda **kwargs: {
            "success": False,
            "provider": "local",
            "error": PRIVATE_DETAIL,
            **context,
        },
    )
    with pytest.raises(RuntimeError) as error:
        entrypoint.run_acquisition(
            repo_id=REPO,
            revision=REVISION,
            filename="data/test.csv",
            output_dir=str(tmp_path),
            job_name="fixture-job",
            runtime_session_id="fixture-session",
            deployment_id="fixture-id",
            deployment_prefix="fixture-prefix",
            image_uri="fixture-image",
            source_commit="b" * 40,
        )
    assert str(error.value) == expected
    assert PRIVATE_DETAIL not in str(error.value)
    assert list(tmp_path.iterdir()) == []
