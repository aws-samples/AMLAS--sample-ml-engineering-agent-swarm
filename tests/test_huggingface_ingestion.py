"""Public Hub wire shapes and guarded portable core; no network or AWS calls.

Hub dataset_info REST shape: https://huggingface.co/.well-known/openapi.json
and https://huggingface.co/docs/huggingface_hub/package_reference/hf_api .
The fake includes rfilename/blobId/size/lfs from the actual blobs=true response.
Public tool authority and managed submission are covered by the binding suites.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REVISION = "a" * 40
PAYLOAD = b'{"messages":[{"role":"user","content":"repair code"}],"resolved":true}\n'
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


def ingestion() -> Any:
    try:
        return importlib.import_module("amlas.tools.huggingface_data")
    except ModuleNotFoundError:
        pytest.fail("missing-symbol: amlas.tools.huggingface_data")


def wire(*, lfs: bool = True) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "rfilename": "v1/train.jsonl",
        "size": len(PAYLOAD),
        "blobId": hashlib.sha1(
            b"blob " + str(len(PAYLOAD)).encode() + b"\0" + PAYLOAD, usedforsecurity=False
        ).hexdigest(),
    }
    if lfs:
        entry["lfs"] = {"sha256": SHA256, "size": len(PAYLOAD), "pointerSize": 130}
    return {
        "id": "example/coding",
        "sha": REVISION,
        "private": False,
        "gated": False,
        "disabled": False,
        "cardData": {
            "license": "cc-by-4.0",
            "configs": [{"config_name": "v1", "data_files": [{"split": "train", "path": "v1/*.jsonl"}]}],
        },
        "siblings": [entry, {"rfilename": "README.md", "size": 20, "blobId": "b" * 40}],
    }


@pytest.fixture
def guarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, list[str]]:
    from amlas.tools import pathref
    from amlas.tools.research import public_fetch

    monkeypatch.delenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", raising=False)
    monkeypatch.delenv("AMLAS_AGENT_REMOTE_FETCH", raising=False)
    monkeypatch.setattr(pathref, "_allowed_roots", lambda: [tmp_path])
    monkeypatch.setattr(pathref, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    module = importlib.import_module("amlas.data_eng.hub_dataset")
    calls: list[str] = []

    def metadata(url: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(url)
        return wire()

    def streamed(url: str, sink: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(url)
        assert kwargs["expected_size"] == len(PAYLOAD)
        sink.write(PAYLOAD)
        return {
            "byte_count": len(PAYLOAD),
            "sha256": SHA256,
            "git_blob_sha1": wire(lfs=False)["siblings"][0]["blobId"],
        }

    monkeypatch.setattr(public_fetch, "fetch_public_json", metadata, raising=False)
    monkeypatch.setattr(public_fetch, "stream_public_binary", streamed, raising=False)
    return module, calls


def test_inspection_resolves_revision_license_files_and_declared_splits(guarded: Any) -> None:
    module, calls = guarded
    result = module.inspect_huggingface_dataset_metadata("example/coding")
    assert result["success"] is True
    assert result["revision"] == REVISION
    assert result["license"] == "cc-by-4.0"
    assert result["public"] is True and result["gated"] is False
    assert result["files"] == [
        {
            "filename": "v1/train.jsonl",
            "format": "jsonl",
            "size": len(PAYLOAD),
            "sha256": SHA256,
            "blob_id": wire()["siblings"][0]["blobId"],
        }
    ]
    assert result["configs"][0]["config_name"] == "v1"
    assert result["splits"] == ["train"]
    assert "blobs=true" in calls[0]
    assert result["coverage"] == "complete"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "repo", ["../coding", "example/coding/extra", "https://example.com", "example%2fcoding"]
)
def test_invalid_repository_never_reaches_transport(guarded: Any, repo: str) -> None:
    module, calls = guarded
    assert module.inspect_huggingface_dataset_metadata(repo)["success"] is False
    assert calls == []


def test_unpinned_acquisition_never_reaches_transport(guarded: Any, tmp_path: Path) -> None:
    module, calls = guarded
    result = module.acquire_huggingface_dataset_local("example/coding", "main", "v1/train.jsonl", str(tmp_path))
    assert result["success"] is False and calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "filename", ["../data.jsonl", "/data.jsonl", "v1%2ftrain.jsonl", "v1/./train.jsonl", "v1\\train.jsonl"]
)
def test_unsafe_file_paths_never_reach_transport(guarded: Any, filename: str, tmp_path: Path) -> None:
    module, calls = guarded
    result = module.acquire_huggingface_dataset_local("example/coding", REVISION, filename, str(tmp_path))
    assert result["success"] is False and calls == []


@pytest.mark.parametrize("field,value", [("private", True), ("gated", "manual"), ("disabled", True)])
def test_restricted_metadata_cannot_authorize_acquisition(
    guarded: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    module, calls = guarded
    from amlas.tools.research import public_fetch

    info = wire()
    info[field] = value
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    result = module.acquire_huggingface_dataset_local("example/coding", REVISION, "v1/train.jsonl", str(tmp_path))
    assert result["success"] is False and calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("lfs", [True, False])
def test_acquisition_verifies_lfs_or_git_framed_hash_and_commits_bundle(
    guarded: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lfs: bool,
) -> None:
    module, calls = guarded
    from amlas.tools.research import public_fetch

    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: wire(lfs=lfs))
    result = module.acquire_huggingface_dataset_local("example/coding", REVISION, "v1/train.jsonl", str(tmp_path))
    assert result["success"] is True
    artifact = Path(result["local_path"])
    lineage = Path(result["lineage_path"])
    assert artifact.read_bytes() == PAYLOAD
    assert artifact.parent == lineage.parent
    assert artifact.parent.is_relative_to(tmp_path)
    assert result["sha256"] == SHA256 and result["byte_count"] == len(PAYLOAD)
    assert (
        result["source_uri"]
        == f"https://huggingface.co/datasets/example/coding/resolve/{REVISION}/v1/train.jsonl"
    )
    record = json.loads(lineage.read_text())
    assert record["sha256"] == SHA256 and record["revision"] == REVISION
    assert record["prepared"] is False and record["trained"] is False
    assert "X-Amz-" not in lineage.read_text()
    assert len(calls) == 1


@pytest.mark.parametrize("mismatch", ["hash", "size", "revision", "missing_file", "oversize"])
def test_failed_integrity_or_listing_leaves_no_committed_artifacts(
    guarded: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str,
) -> None:
    module, _ = guarded
    from amlas.tools.research import public_fetch

    info = wire()
    if mismatch == "hash":
        info["siblings"][0]["lfs"]["sha256"] = "0" * 64
    elif mismatch == "size":
        info["siblings"][0]["size"] += 1
        info["siblings"][0]["lfs"]["size"] += 1
    elif mismatch == "revision":
        info["sha"] = "b" * 40
    elif mismatch == "missing_file":
        info["siblings"] = []
    else:
        info["siblings"][0]["size"] = 512 * 1024 * 1024 + 1
        info["siblings"][0]["lfs"]["size"] = 512 * 1024 * 1024 + 1
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    result = module.acquire_huggingface_dataset_local("example/coding", REVISION, "v1/train.jsonl", str(tmp_path))
    assert result["success"] is False
    assert list(tmp_path.iterdir()) == []


def test_output_escape_and_symlink_directory_fail_before_metadata(
    guarded: Any,
    tmp_path: Path,
) -> None:
    module, calls = guarded
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    for output in [str(outside), str(tmp_path / "link"), str(tmp_path / ".." / "escape")]:
        assert (
            module.acquire_huggingface_dataset_local(
                "example/coding",
                REVISION,
                "v1/train.jsonl",
                output,
            )["success"]
            is False
        )
    assert calls == []
    assert list(outside.iterdir()) == []


def test_failure_sanitizes_signed_redirect_details(
    guarded: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _ = guarded
    from amlas.tools.research import public_fetch

    signed = "https://cdn.example.com/data?X-Amz-Signature=" + "test-secret"

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError(signed)

    monkeypatch.setattr(public_fetch, "stream_public_binary", fail)
    result = module.acquire_huggingface_dataset_local("example/coding", REVISION, "v1/train.jsonl", str(tmp_path))
    assert result["success"] is False
    assert signed not in json.dumps(result) and "test-secret" not in json.dumps(result)
    assert list(tmp_path.iterdir()) == []


def test_hub_entrypoints_honor_remote_disabled_before_transport(
    guarded: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, calls = guarded
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "off")
    assert module.inspect_huggingface_dataset_metadata("example/coding")["success"] is False
    assert (
        module.acquire_huggingface_dataset_local("example/coding", REVISION, "v1/train.jsonl", str(tmp_path))[
            "success"
        ]
        is False
    )
    assert calls == []


def test_metadata_file_count_has_explicit_partial_coverage(
    guarded: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = guarded
    from amlas.tools.research import public_fetch

    info = wire()
    info["siblings"] = [{**wire()["siblings"][0], "rfilename": f"data/{i:04d}.jsonl"} for i in range(700)]
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    result = module.inspect_huggingface_dataset_metadata("example/coding")
    assert result["success"] is True
    assert len(result["files"]) <= 512
    assert result["coverage"] == "partial"
    assert result["files_truncated"] is True


def test_missing_integrity_fields_and_lfs_pointer_are_rejected(
    guarded: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _ = guarded
    from amlas.tools.research import public_fetch

    info = wire(lfs=False)
    info["siblings"][0].pop("blobId")
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    result = module.acquire_huggingface_dataset_local("example/coding", REVISION, "v1/train.jsonl", str(tmp_path))
    assert result["success"] is False
    assert list(tmp_path.iterdir()) == []


def test_formats_preserve_nested_parquet_and_jsonl_records(tmp_path: Path) -> None:
    from amlas.tools import pathref

    frame = pd.DataFrame([{"messages": [{"role": "assistant", "content": "fixed"}], "resolved": True}])
    for suffix in [".parquet", ".jsonl", ".json"]:
        path = tmp_path / ("nested" + suffix)
        if suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_json(path, orient="records", lines=suffix == ".jsonl")
        loaded = pathref.read_dataframe(pathref.resolve_ref(str(path), sandbox_local=False))
        assert loaded["resolved"].tolist() == [True]
        assert next(iter(loaded.iloc[0]["messages"]))["content"] == "fixed"


def test_nested_dataset_summaries_are_bounded_and_strict_json(tmp_path: Path) -> None:
    from amlas.tools.data_tools import describe_dataset_local as describe_dataset
    from amlas.tools.data_tools import load_dataset_local as load_dataset

    path = tmp_path / "nested.parquet"
    frame = pd.DataFrame(
        [
            {
                "messages": [{"role": "user", "content": "x" * 500000}],
                "optional": None,
                "metric": float("nan"),
            }
        ]
    )
    frame.to_parquet(path, index=False)
    loaded = load_dataset(str(path))
    described = describe_dataset(str(path))
    assert loaded["success"] is True and described["success"] is True
    assert len(json.dumps(loaded, allow_nan=False)) < 20000
    assert len(json.dumps(described, allow_nan=False)) < 20000
    assert loaded["sample_truncated"] is True
    assert described["categorical_stats"]["messages"]["nested"] is True


def test_decoded_parquet_and_jsonl_record_caps_reject_before_full_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as parquet

    from amlas.tools import pathref

    path = tmp_path / "large.parquet"
    path.write_bytes(b"synthetic")

    class Metadata:
        num_row_groups = 1

        def row_group(self, index: int) -> Any:
            return type("Group", (), {"total_byte_size": 512 * 1024 * 1024 + 1})()

    monkeypatch.setattr(
        parquet, "ParquetFile", lambda *args, **kwargs: type("Parquet", (), {"metadata": Metadata()})()
    )
    with pytest.raises(pathref.PathRefError, match="decoded"):
        pathref.read_dataframe(pathref.resolve_ref(str(path), sandbox_local=False))
    huge = tmp_path / "huge.jsonl"
    huge.write_text(json.dumps({"text": "x" * (8 * 1024 * 1024 + 1)}) + "\n")
    with pytest.raises(pathref.PathRefError, match="record"):
        pathref.read_dataframe(pathref.resolve_ref(str(huge), sandbox_local=False))


def test_ingestion_tools_are_available_to_chat_and_dataset_agents() -> None:
    module = ingestion()
    from amlas.pipeline.agentic_runner import PIPELINE_TOOLS
    from amlas.subagents.declarations import DATA_COLLECTION_AGENT
    from amlas.tui.chat.agent_tools import READ_ONLY_TOOLS, STATE_CHANGING_TOOLS, get_safe_llm_tools

    required = {"inspect_huggingface_dataset", "acquire_huggingface_dataset"}

    def names(tools: Any) -> set[str]:
        return {getattr(tool, "tool_name", getattr(tool, "__name__", "")) for tool in tools}

    assert required <= names(get_safe_llm_tools())
    assert required <= names(DATA_COLLECTION_AGENT.tools)
    assert required <= names(PIPELINE_TOOLS)
    assert "inspect_huggingface_dataset" in READ_ONLY_TOOLS
    assert "acquire_huggingface_dataset" in STATE_CHANGING_TOOLS
    assert module.acquire_huggingface_dataset.tool_spec["inputSchema"]["json"]["required"] == [
        "repo_id",
        "revision",
        "filename",
    ]


def test_binary_transport_revalidates_https_each_hop_and_never_echoes_signed_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.tools import pathref
    from amlas.tools.research import public_fetch

    transfer = getattr(public_fetch, "stream_public_binary", None)
    assert callable(transfer), "missing-symbol: stream_public_binary"
    monkeypatch.setattr(pathref, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    monkeypatch.delenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", raising=False)
    calls: list[str] = []

    def request(url: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(url)
        return 302, {"location": "http://169.254.169.254/private?X-Amz-Signature=test-secret"}, 0

    monkeypatch.setattr(public_fetch, "_stream_request_public", request, raising=False)
    with pytest.raises(public_fetch.PublicFetchError) as failure:
        transfer("https://huggingface.co/data", io.BytesIO(), max_bytes=100, timeout=10)
    assert calls == ["https://huggingface.co/data"]
    assert "test-secret" not in str(failure.value)


@pytest.mark.parametrize(
    "ips", [["93.184.216.34", "10.0.0.1"], ["224.0.0.1"], ["127.0.0.1"], ["169.254.169.254"]]
)
def test_binary_transport_blocks_every_nonpublic_pinned_address(
    monkeypatch: pytest.MonkeyPatch,
    ips: list[str],
) -> None:
    from amlas.tools import pathref
    from amlas.tools.research import public_fetch

    transfer = getattr(public_fetch, "stream_public_binary", None)
    assert callable(transfer), "missing-symbol: stream_public_binary"
    monkeypatch.setattr(pathref, "_resolve_host_ips", lambda host: ips)
    monkeypatch.setattr(
        public_fetch,
        "_stream_request_public",
        lambda *args, **kwargs: pytest.fail("unsafe connection"),
        raising=False,
    )
    with pytest.raises(public_fetch.PublicFetchError):
        transfer("https://huggingface.co/data", io.BytesIO(), max_bytes=100, timeout=10)


def test_binary_transport_enforces_allowlist_on_redirect_and_remote_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.tools import pathref
    from amlas.tools.research import public_fetch

    transfer = getattr(public_fetch, "stream_public_binary", None)
    assert callable(transfer), "missing-symbol: stream_public_binary"
    monkeypatch.setattr(pathref, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    monkeypatch.setenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", "huggingface.co")
    calls: list[str] = []

    def request(url: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(url)
        return 302, {"location": "https://cdn.example.com/file"}, 0

    monkeypatch.setattr(public_fetch, "_stream_request_public", request, raising=False)
    with pytest.raises(public_fetch.PublicFetchError):
        transfer("https://huggingface.co/data", io.BytesIO(), max_bytes=100, timeout=10)
    assert calls == ["https://huggingface.co/data"]
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "0")
    calls.clear()
    with pytest.raises(public_fetch.PublicFetchError):
        transfer("https://huggingface.co/data", io.BytesIO(), max_bytes=100, timeout=10)
    assert calls == []


@pytest.mark.parametrize("status", [302, 206, 403])
def test_binary_request_checks_status_before_reading_response_body(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    from amlas.tools.research import public_fetch

    request = getattr(public_fetch, "_stream_request_public", None)
    assert callable(request), "missing-symbol: _stream_request_public"

    class Response:
        def __init__(self) -> None:
            self.status = status

        def getheaders(self) -> list[tuple[str, str]]:
            return [("Location", "https://cdn.example.com/file")]

        def read(self, amount: int) -> bytes:
            pytest.fail("redirect/error/partial body was read")

    class Connection:
        sock = None

        def __init__(self, *args: Any):
            pass

        def request(self, method: str, target: str, headers: dict[str, str]) -> None:
            assert method == "GET"
            assert not {"Authorization", "Cookie"} & set(headers)
            assert headers["Accept-Encoding"] == "identity"

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            pass

    monkeypatch.setattr(public_fetch, "_PinnedHTTPS", Connection)
    import time

    result = request(
        "https://example.com/data",
        ["93.184.216.34"],
        time.monotonic() + 10,
        100,
        io.BytesIO(),
        hashlib.sha256(),
        None,
    )
    assert result[0] == status and result[2] == 0


@pytest.mark.parametrize("case", ["oversize", "truncated", "encoded", "loop", "deadline"])
def test_binary_stream_budgets_partial_bytes_and_integrity(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    from amlas.tools import pathref
    from amlas.tools.research import public_fetch

    transfer = getattr(public_fetch, "stream_public_binary", None)
    assert callable(transfer), "missing-symbol: stream_public_binary"
    monkeypatch.setattr(pathref, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    monkeypatch.delenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", raising=False)
    monkeypatch.delenv("AMLAS_AGENT_REMOTE_FETCH", raising=False)

    class Response:
        status = 302 if case == "loop" else 200

        def getheaders(self) -> list[tuple[str, str]]:
            if case == "loop":
                return [("Location", "/data")]
            if case == "encoded":
                return [("Content-Encoding", "gzip")]
            return [("Content-Length", "4" if case != "oversize" else "200")]

        def read(self, amount: int) -> bytes:
            if case == "deadline":
                import time

                time.sleep(0.02)
            if getattr(self, "done", False):
                return b""
            self.done = True
            return b"xx" if case == "truncated" else b"data"

    class Connection:
        sock = None

        def __init__(self, *args: Any):
            pass

        def request(self, *args: Any, **kwargs: Any) -> None:
            pass

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            pass

    monkeypatch.setattr(public_fetch, "_PinnedHTTPS", Connection)
    with pytest.raises(public_fetch.PublicFetchError):
        transfer(
            "https://example.com/data",
            io.BytesIO(),
            max_bytes=100,
            timeout=0.005 if case == "deadline" else 10,
            expected_size=4,
        )
