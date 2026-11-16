"""A read-only qualification scope is private to the context that enters it.

`_qualified_reads` used to replace `_sm` and `_s3` on the adapter object that the whole session shares. The
tools of one assistant message run on worker threads while the BeforeToolCall fence of a later tool runs on
the event-loop thread, so a tool that wrote through the adapter during a fence's qualification met the
read-only proxy (`AttributeError: create_multipart_upload`). These tests force that interleaving with
events; none of them sleeps.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import threading
import time

import pytest

from amlas.backends.adapters.coding_case_freeze import ProcessingCodingCaseFreezeBackend
from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend
from amlas.backends.adapters.coding_pair import SageMakerPairedCodingBackend
from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
from amlas.backends.adapters.dataset_acquisition import ProcessingDatasetAcquisitionBackend
from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.normal_run_checkpoint import _NativeIO
from amlas.control_plane.normal_run_parents import _qualified_reads
from amlas.control_plane.sft_state import SftCatalog

REGION = "us-west-2"
WAIT = 30  # seconds; only bounds a hang, never a synchronisation
PAYLOAD = b"0123456789"
SHA = hashlib.sha256(PAYLOAD).hexdigest()
CATALOG = SftCatalog(
    account_id="test-account",
    region=REGION,
    bucket="owned-test-bucket",
    execution_role_arn="arn:aws:iam::test-account:role/test",
    image_uri="image",
    source_commit="a" * 40,
    admission_digest="b" * 64,
    deployment_id="deployment",
    deployment_prefix="owned",
)
SOURCE = DatasetArtifactRef(
    uri="s3://owned-test-bucket/source/object",
    version_id="v1",
    sha256=SHA,
    byte_count=len(PAYLOAD),
    format="bin",
)


class _Meta:
    region_name = REGION


class _S3:
    """Write-capable client: records every call."""

    meta = _Meta()

    def __init__(self):
        self.calls = []

    def create_multipart_upload(self, **kw):
        self.calls.append("create_multipart_upload")
        return {"UploadId": "upload"}

    def upload_part_copy(self, **kw):
        self.calls.append("upload_part_copy")
        return {"CopySourceVersionId": "v1", "CopyPartResult": {"ETag": "etag"}}

    def complete_multipart_upload(self, **kw):
        self.calls.append("complete_multipart_upload")
        return {"VersionId": "v2"}

    def abort_multipart_upload(self, **kw):
        self.calls.append("abort_multipart_upload")

    def head_object(self, **kw):
        self.calls.append("head_object")
        return {
            "VersionId": "v2",
            "ContentLength": len(PAYLOAD),
            "ChecksumType": "FULL_OBJECT",
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(SHA)).decode(),
        }

    def get_object(self, **kw):
        self.calls.append("get_object")
        raise AssertionError("not used by the staged-copy path")


class _Sm:
    meta = _Meta()

    def describe_training_job(self, **kw):
        return {}

    def list_tags(self, **kw):
        return {"Tags": []}


def _adapter(cls=SageMakerQwenSftBackend):
    sm, s3 = _Sm(), _S3()
    if cls is ProcessingDatasetAcquisitionBackend:
        return cls(region=REGION, sagemaker_client=sm, s3_client=s3), sm, s3
    return cls(sagemaker_client=sm, s3_client=s3), sm, s3


def _noop():
    return None


def _copy(adapter):
    return adapter._copy(SOURCE, "sessions/s/sft/job/input/object", CATALOG, time.monotonic() + WAIT)


def test_tool_write_is_accepted_while_another_thread_holds_a_read_scope():
    """The reported interleaving at thread level: scope open, then the tool's `_copy` runs."""
    adapter, sm, s3 = _adapter()
    scope_open, copy_done = threading.Event(), threading.Event()
    outcome = {}

    def tool():
        try:
            assert scope_open.wait(WAIT)
            outcome["staged"] = _copy(adapter)
        except BaseException as error:  # the test reads it below
            outcome["error"] = error
        finally:
            copy_done.set()

    worker = threading.Thread(target=tool)
    worker.start()
    try:
        with _qualified_reads((adapter,), REGION, _noop):
            scope_open.set()
            assert copy_done.wait(WAIT)
            # The fence's own context keeps the read-only view.
            seen_sm, seen_s3 = adapter._clients(REGION)
            assert type(seen_s3) is _NativeIO and type(seen_sm) is _NativeIO
            with pytest.raises(AttributeError):
                _ = seen_s3.create_multipart_upload
    finally:
        scope_open.set()
        worker.join(WAIT)
    assert not worker.is_alive()
    assert "error" not in outcome, repr(outcome.get("error"))
    assert outcome["staged"].version_id == "v2"
    assert s3.calls.count("create_multipart_upload") == 1
    assert adapter._clients(REGION) == (sm, s3)


def test_tool_in_a_worker_thread_while_the_fence_blocks_the_event_loop():
    """The Strands arrangement: the fence is a sync callback run inline on the loop thread while the
    other tool of the message runs through `asyncio.to_thread` (tools/executors/concurrent.py,
    hooks/registry.py invoke_callbacks_async, tools/decorator.py)."""
    adapter, sm, s3 = _adapter()
    scope_open, copy_done = threading.Event(), threading.Event()
    outcome = {}

    def reload_tool():
        try:
            assert scope_open.wait(WAIT)
            outcome["staged"] = _copy(adapter)
        except BaseException as error:
            outcome["error"] = error
        finally:
            copy_done.set()

    async def scenario():
        started = asyncio.Event()

        async def first_task():
            started.set()  # the worker is submitted by the next statement, before any other task runs
            await asyncio.to_thread(reload_tool)

        async def second_task():
            await started.wait()
            # BeforeToolCallEvent fence of the second tool: synchronous, blocks the loop.
            with _qualified_reads((adapter,), REGION, _noop):
                scope_open.set()
                assert copy_done.wait(WAIT)

        await asyncio.gather(first_task(), second_task())

    try:
        asyncio.run(scenario())
    finally:
        scope_open.set()
    assert "error" not in outcome, repr(outcome.get("error"))
    assert outcome["staged"].version_id == "v2"
    assert adapter._clients(REGION) == (sm, s3)


def test_overlapping_scopes_restore_the_original_clients():
    """Two scopes entered and left in the order enter, enter, leave, leave must not leave a proxy behind."""
    adapter, sm, s3 = _adapter()
    first_in, second_in, first_out = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def first():
        try:
            with _qualified_reads((adapter,), REGION, _noop):
                first_in.set()
                assert second_in.wait(WAIT)
        except BaseException as error:
            errors.append(error)
        finally:
            first_out.set()

    def second():
        try:
            assert first_in.wait(WAIT)
            with _qualified_reads((adapter,), REGION, _noop):
                second_in.set()
                assert first_out.wait(WAIT)
        except BaseException as error:
            errors.append(error)
        finally:
            second_in.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert adapter._clients(REGION) == (sm, s3)
    assert adapter._sm is sm and adapter._s3 is s3


def test_nested_scopes_in_one_context_keep_both_qualifiers():
    adapter, sm, s3 = _adapter()
    calls = []
    with _qualified_reads((adapter,), REGION, lambda: calls.append("outer")):
        with _qualified_reads((adapter,), REGION, lambda: calls.append("inner")):
            adapter._clients(REGION)[1].head_object(Bucket="b", Key="k")
        assert type(adapter._clients(REGION)[1]) is _NativeIO
    assert {"outer", "inner"} <= set(calls)
    assert adapter._clients(REGION) == (sm, s3)


def test_scope_is_released_when_entering_the_next_adapter_fails():
    first, sm, s3 = _adapter()

    class Refusing(SageMakerQwenSftBackend):
        def qualified_io(self, current_qualification=None):
            raise ValueError("scope refused")

    second, _sm, _s3 = _adapter(Refusing)
    with pytest.raises(ValueError, match="scope refused"):
        with _qualified_reads((first, second), REGION, _noop):
            raise AssertionError("not reached")
    assert first._clients(REGION) == (sm, s3)
    assert second._clients(REGION) == (_sm, _s3)


@pytest.mark.parametrize(
    "cls",
    [
        SageMakerQwenSftBackend,
        SageMakerCodingGenerationBackend,
        SageMakerPairedCodingBackend,
        ProcessingCodingPreparationBackend,
        ProcessingCodingCaseFreezeBackend,
        ProcessingDatasetAcquisitionBackend,
    ],
)
def test_every_wrapped_adapter_keeps_raw_clients_for_other_threads(cls):
    adapter, sm, s3 = _adapter(cls)
    scope_open, seen_done = threading.Event(), threading.Event()
    seen = {}

    def other_thread():
        try:
            assert scope_open.wait(WAIT)
            seen["clients"] = adapter._clients(REGION)
            seen["attrs"] = (adapter._sm, adapter._s3)
        except BaseException as error:
            seen["error"] = error
        finally:
            seen_done.set()

    worker = threading.Thread(target=other_thread)
    worker.start()
    try:
        with _qualified_reads((adapter,), REGION, _noop):
            scope_open.set()
            assert seen_done.wait(WAIT)
            assert all(type(client) is _NativeIO for client in adapter._clients(REGION))
            assert all(type(client) is _NativeIO for client in (adapter._sm, adapter._s3))
    finally:
        scope_open.set()
        worker.join(WAIT)
    assert "error" not in seen, repr(seen.get("error"))
    assert seen["clients"] == (sm, s3)
    assert seen["attrs"] == (sm, s3)
    assert adapter._clients(REGION) == (sm, s3)


def test_a_replaced_raw_client_is_seen_after_the_scope():
    """Lazy construction and tests assign the raw clients; an assignment outside any scope stays raw."""
    adapter, _sm, _s3 = _adapter()
    replacement = _S3()
    with _qualified_reads((adapter,), REGION, _noop):
        pass
    adapter._s3 = replacement
    assert adapter._s3 is replacement
    assert adapter._clients(REGION)[1] is replacement
