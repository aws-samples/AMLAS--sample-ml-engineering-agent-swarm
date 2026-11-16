"""Only source-original managed observation receives lexical per-leg IO fences."""

from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest
from test_managed_generation_product import installed_generation
from test_paired_coding_backend import outputs as pair_outputs
from test_paired_coding_backend import submit as submit_pair
from test_qwen_sft_backend import SESSION, catalog, ref
from test_qwen_sft_backend import backend as sft_fixture

from amlas.backends.types import JobState

pytest_plugins = ("test_managed_generation_product", "test_paired_coding_backend")


@pytest.fixture(params=("sft", "generation", "pair"))
def original(request, monkeypatch):
    kind = request.param
    if kind == "sft":
        adapter, native, s3, inputs = sft_fixture()
        handle = adapter.submit(inputs=inputs, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    elif kind == "generation":
        adapter, native, s3, handle, *_ = installed_generation(request.getfixturevalue("state"))
        native.state = "InProgress"
    else:
        packet = request.getfixturevalue("packet")
        adapter, _, _, native, s3, _ = packet
        handle = submit_pair(packet)
    assert len(native.calls) == 1 and handle.extra["submission_state"] == "accepted"
    calls, events = [], []
    for target, names in ((native, ("describe_training_job", "describe_processing_job", "list_tags")),
                          (s3, ("head_object", "get_object", "list_objects_v2", "put_object", "copy_object"))):
        for name in names:
            method = getattr(target, name, None)
            if method is None:
                continue
            def observed(*args, _method=method, _name=name, **kwargs):
                calls.append(_name)
                events.append(_name)
                return _method(*args, **kwargs)
            monkeypatch.setattr(target, name, observed)
    raw = b'{"neutral":true}\n'
    artifact = ref("qualified-output.json", raw)
    key = artifact.uri.split("/", 3)[3]
    if kind == "pair":
        artifact = s3.install(key, raw)
    else:
        s3.install(artifact, raw)
    def read():
        if kind == "pair":
            return adapter._read(artifact, catalog(), SESSION, time.monotonic() + 5, limit=256)
        return adapter._output(handle, key, time.monotonic() + 5, limit=256, collect=True)[1]
    return SimpleNamespace(kind=kind, adapter=adapter, native=native, s3=s3, handle=handle,
                           calls=calls, events=events, read=read, raw=raw)


def deny():
    raise PermissionError("ordinary current original observation refusal")


def assert_no_create(original):
    creates = original.native.calls.count("create") if original.kind == "pair" else len(original.native.calls)
    assert creates == 1
    assert not set(original.calls) & {"put_object", "copy_object"}


def test_status_fences_every_actual_native_read(original):
    with original.adapter.qualified_io(lambda: original.events.append("qualify")):
        status = original.adapter.status(original.handle)
    assert status.state == JobState.RUNNING
    assert original.calls[-1] == "list_tags"
    for index, event in enumerate(original.events):
        if event != "qualify":
            assert original.events[index - 1] == original.events[index + 1] == "qualify"
    assert_no_create(original)


def test_refusal_before_native_read_preserves_original_without_io(original):
    with original.adapter.qualified_io(deny):
        assert original.adapter.status(original.handle).state == JobState.UNKNOWN
    assert original.calls == []
    assert_no_create(original)


def test_native_read_revocation_blocks_tags_and_status(original, monkeypatch):
    current = [True]
    name = "describe_processing_job" if original.kind == "pair" else "describe_training_job"
    method = getattr(original.native, name)
    def changed(**kwargs):
        value = method(**kwargs)
        current[0] = False
        return value
    monkeypatch.setattr(original.native, name, changed)
    def qualify():
        if not current[0]:
            deny()
    with original.adapter.qualified_io(qualify):
        assert original.adapter.status(original.handle).state == JobState.UNKNOWN
    assert original.calls == [name]
    assert_no_create(original)


@pytest.mark.parametrize("copied", ["expired", "foreign-thread"])
def test_copied_scope_cannot_retain_or_transfer_original_io(original, copied):
    with original.adapter.qualified_io(lambda: None):
        saved = copy_context()
        if copied == "foreign-thread":
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(saved.run, original.adapter.status, original.handle).result(timeout=5)
            assert result.state == JobState.UNKNOWN
    if copied == "expired":
        assert saved.run(original.adapter.status, original.handle).state == JobState.UNKNOWN
    assert original.calls == []
    assert_no_create(original)


def test_inner_none_cannot_remove_outer_refusal_and_scope_resets(original):
    with original.adapter.qualified_io(deny):
        with original.adapter.qualified_io(None):
            assert original.adapter.status(original.handle).state == JobState.UNKNOWN
    assert original.calls == []
    assert original.adapter.status(original.handle).state == JobState.RUNNING
    assert_no_create(original)


def test_raw_observation_refuses_before_head(original):
    with original.adapter.qualified_io(deny):
        with pytest.raises(PermissionError):
            original.read()
    assert original.calls == []
    assert_no_create(original)


class Body(io.BytesIO):
    def __init__(self, raw, *, after_read=None, after_close=None):
        super().__init__(raw)
        self.read_count = 0
        self.after_read = after_read
        self.after_close = after_close
    def read(self, size=-1):
        self.read_count += 1
        value = super().read(size)
        if self.after_read:
            self.after_read()
        return value
    def close(self):
        super().close()
        if self.after_close:
            self.after_close()


@pytest.mark.parametrize("boundary", ["get", "read", "close"])
def test_returned_body_is_closed_and_current_before_raw_exposure(original, monkeypatch, boundary):
    current, bodies = [True], []
    method = original.s3.get_object
    def revoke():
        current[0] = False
    def get(**kwargs):
        response = method(**kwargs)
        response["Body"].close()
        body = Body(original.raw, after_read=revoke if boundary == "read" else None,
                    after_close=revoke if boundary == "close" else None)
        response["Body"] = body
        bodies.append(body)
        if boundary == "get":
            revoke()
        return response
    monkeypatch.setattr(original.s3, "get_object", get)
    def qualify():
        if not current[0]:
            deny()
    with original.adapter.qualified_io(qualify):
        with pytest.raises(PermissionError):
            original.read()
    assert len(bodies) == 1 and bodies[0].closed
    if boundary == "get":
        assert bodies[0].read_count == 0
    assert_no_create(original)


@pytest.fixture
def committed_pair_output(packet):
    adapter, _inputs, original_catalog, native, s3, _validated = packet
    handle = submit_pair(packet)
    payloads = pair_outputs(packet, handle)
    commit = json.loads(payloads["COMMITTED.json"])
    descriptor = commit["files"]["EVALUATION.json"]
    prefix = handle.extra["output_prefix"]
    expected = s3.install(prefix + "EVALUATION.json", payloads["EVALUATION.json"])
    assert descriptor == {
        "sha256": expected.sha256,
        "byte_count": expected.byte_count,
        "format": expected.format,
    }
    assert handle.extra["submission_state"] == "accepted" and native.calls.count("create") == 1
    s3.calls.clear()
    return SimpleNamespace(
        adapter=adapter, catalog=original_catalog, native=native, s3=s3,
        prefix=prefix, descriptor=descriptor, expected=expected,
    )


def declared_pair_output(original):
    return original.adapter._declared_output(
        original.catalog, original.prefix, "EVALUATION.json", original.descriptor,
        "json", 262144, time.monotonic() + 5,
    )


def test_pair_declared_output_preserves_committed_descriptor(committed_pair_output):
    original = committed_pair_output
    with original.adapter.qualified_io(lambda: None):
        assert declared_pair_output(original) == original.expected
    assert [name for name, _kwargs in original.s3.calls] == ["HeadObject"]
    assert original.native.calls.count("create") == 1


def test_pair_declared_output_refuses_before_head(committed_pair_output):
    original = committed_pair_output
    with original.adapter.qualified_io(deny):
        with pytest.raises(PermissionError):
            declared_pair_output(original)
    assert original.s3.calls == []
    assert original.native.calls.count("create") == 1


def test_pair_declared_output_revocation_after_head_refuses_exposure(committed_pair_output, monkeypatch):
    original = committed_pair_output
    current = [True]
    method = original.s3.head_object

    def revoke_after_head(**kwargs):
        response = method(**kwargs)
        current[0] = False
        return response

    def qualify():
        if not current[0]:
            deny()

    monkeypatch.setattr(original.s3, "head_object", revoke_after_head)
    with original.adapter.qualified_io(qualify):
        with pytest.raises(PermissionError):
            declared_pair_output(original)
    assert [name for name, _kwargs in original.s3.calls] == ["HeadObject"]
    assert original.native.calls.count("create") == 1
