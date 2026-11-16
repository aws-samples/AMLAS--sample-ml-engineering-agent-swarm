"""Observation-only current fences over neutral SDK fixtures; no native effects."""

from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any

import pytest
from test_coding_case_native import SESSION as CASE_SESSION
from test_coding_case_native import native as case_fixture  # noqa: F401
from test_managed_coding_preparation import fixture as preparation_fixture  # noqa: F401
from test_managed_coding_preparation import submit as submit_preparation
from test_managed_dataset_acquisition import adapter as data_adapter
from test_managed_dataset_acquisition import ownership, populate  # noqa: F401
from test_managed_dataset_acquisition import submit as submit_data

from amlas.backends.types import JobState


@dataclass
class Observation:
    kind: str
    backend: Any
    sm: Any
    s3: Any
    handle: Any
    calls: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    creates: int = 1

    def qualify(self):
        self.events.append("qualify")

    def assert_no_create(self):
        actual = len(self.sm.calls if self.kind == "data" else self.sm.creates)
        assert actual == self.creates
        assert not any(name in self.calls for name in ("put_object", "copy_object"))


@pytest.fixture(params=("data", "case", "prep"))
def observation(request, monkeypatch):
    request.getfixturevalue("ownership")
    if request.param == "data":
        backend = data_adapter()
        sm, s3 = backend._sm, backend._s3
        handle = submit_data(backend)
    elif request.param == "case":
        backend, sm, s3, source, catalog = request.getfixturevalue("case_fixture")
        handle = backend.submit(source=source, catalog=catalog, runtime_session_id=CASE_SESSION)
    else:
        selected = request.getfixturevalue("preparation_fixture")
        backend, sm, s3, _catalog, _source = selected
        handle = submit_preparation(selected)
    result = Observation(request.param, backend, sm, s3, handle)
    for target, name in (
        (sm, "describe_processing_job"),
        (sm, "list_tags"),
        (s3, "list_objects_v2"),
        (s3, "head_object"),
        (s3, "get_object"),
        (s3, "put_object"),
        (s3, "copy_object"),
    ):
        original = getattr(target, name, None)
        if original is None:
            continue

        def observed(*args, _original=original, _name=name, **kwargs):
            result.calls.append(_name)
            result.events.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(target, name, observed)
    return result


def assert_refused_status(observation):
    try:
        result = observation.backend.status(observation.handle)
    except (PermissionError, ValueError):
        return
    assert result.state == JobState.UNKNOWN


def test_original_status_has_fresh_before_and_after_each_sdk_read(observation):
    with observation.backend.qualified_io(observation.qualify):
        result = observation.backend.status(observation.handle)
    assert result.state == JobState.RUNNING
    assert observation.calls[0] == "describe_processing_job"
    for index, event in enumerate(observation.events):
        if event != "qualify":
            assert observation.events[index - 1] == observation.events[index + 1] == "qualify"
    observation.assert_no_create()


def test_pre_describe_refusal_performs_zero_sdk_reads(observation):
    def denied():
        raise PermissionError("ordinary current observation refusal")

    with observation.backend.qualified_io(denied):
        assert_refused_status(observation)
    assert observation.calls == []
    observation.assert_no_create()


def test_post_describe_revocation_cannot_expose_qualified_status(observation, monkeypatch):
    current = [True]
    original = observation.sm.describe_processing_job

    def revoke(**kwargs):
        result = original(**kwargs)
        current[0] = False
        return result

    monkeypatch.setattr(observation.sm, "describe_processing_job", revoke)

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary post-Describe observation refusal")

    with observation.backend.qualified_io(qualify):
        assert_refused_status(observation)
    assert observation.calls == ["describe_processing_job"]
    observation.assert_no_create()


def test_scopes_are_instance_local_and_finally_restore_legacy_observation(observation):
    if observation.kind == "data":
        other = data_adapter(sm=observation.sm, s3=observation.s3)
    else:
        other = type(observation.backend)(sagemaker_client=observation.sm, s3_client=observation.s3)

    def denied():
        raise PermissionError("ordinary scope refusal")

    with other.qualified_io(denied):
        assert observation.backend.status(observation.handle).state == JobState.RUNNING
    observation.calls.clear()
    with observation.backend.qualified_io(denied):
        assert_refused_status(observation)
    assert observation.calls == []
    assert observation.backend.status(observation.handle).state == JobState.RUNNING
    observation.assert_no_create()


def test_inner_none_scope_cannot_remove_outer_current_refusal(observation):
    def denied():
        raise PermissionError("ordinary outer observation refusal")

    with observation.backend.qualified_io(denied):
        with observation.backend.qualified_io(None):
            assert_refused_status(observation)
    assert observation.calls == []


def test_copied_scope_expires_after_original_lexical_owner_exits(observation):
    with observation.backend.qualified_io(observation.qualify):
        saved = copy_context()
    saved.run(assert_refused_status, observation)
    assert observation.calls == []
    observation.assert_no_create()


def test_copied_scope_cannot_observe_from_another_thread(observation):
    with observation.backend.qualified_io(observation.qualify):
        saved = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(saved.run, assert_refused_status, observation).result(timeout=5)
    assert observation.calls == []
    observation.assert_no_create()


def test_pid_changed_adapter_refuses_scope_before_sdk_read(observation):
    observation.backend._qualified_io_pid = -1
    with pytest.raises(ValueError):
        with observation.backend.qualified_io(observation.qualify):
            pytest.fail("inherited adapter scope activated")
    assert observation.calls == []


class Body(io.BytesIO):
    def __init__(self, raw, events, *, after_read=None, after_close=None):
        super().__init__(raw)
        self.events = events
        self.reads = 0
        self.after_read = after_read
        self.after_close = after_close

    def read(self, size=-1):
        self.reads += 1
        self.events.append("body_read")
        result = super().read(size)
        if self.after_read is not None:
            self.after_read()
        return result

    def close(self):
        was_closed = self.closed
        super().close()
        if not was_closed and self.after_close is not None:
            self.after_close()


def fixed_raw_read(observation, body, monkeypatch):
    raw = b'{"ordinary":true}\n'
    handle = observation.handle
    if observation.kind == "data":
        bucket = handle.extra["bucket"]
        account = handle.extra["account_id"]
    else:
        catalog = handle.extra["catalog"]
        bucket, account = catalog.bucket, catalog.account_id
    key = handle.extra["output_prefix"] + "QUALIFICATION.json"

    def get(**kwargs):
        observation.calls.append("get_object")
        observation.events.append("get_object")
        assert kwargs["Bucket"] == bucket and kwargs["Key"] == key
        assert kwargs["ExpectedBucketOwner"] == account
        if "VersionId" in kwargs:
            assert kwargs["VersionId"] == "ordinary-version"
        return {"VersionId": "ordinary-version", "ContentLength": len(raw), "Body": body}

    monkeypatch.setattr(observation.s3, "get_object", get)
    if observation.kind == "data":
        return lambda: observation.backend._read_manifest(handle, "QUALIFICATION.json")
    return lambda: observation.backend._raw(
        observation.s3,
        bucket,
        key,
        "ordinary-version",
        account,
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        metadata=True,
    )


def test_manifest_stream_each_read_has_current_before_and_after_fences(observation, monkeypatch):
    body = Body(b'{"ordinary":true}\n', observation.events)
    read = fixed_raw_read(observation, body, monkeypatch)
    with observation.backend.qualified_io(observation.qualify):
        read()
    assert body.closed and body.reads > 0
    for index, event in enumerate(observation.events):
        if event != "qualify":
            assert observation.events[index - 1] == observation.events[index + 1] == "qualify"
    observation.assert_no_create()


def test_post_get_refusal_closes_returned_stream_without_read(observation, monkeypatch):
    current = [True]
    body = Body(b'{"ordinary":true}\n', observation.events)
    read = fixed_raw_read(observation, body, monkeypatch)
    original = observation.s3.get_object

    def get(**kwargs):
        result = original(**kwargs)
        current[0] = False
        return result

    monkeypatch.setattr(observation.s3, "get_object", get)

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary post-Get observation refusal")

    with observation.backend.qualified_io(qualify):
        with pytest.raises(PermissionError):
            read()
    assert body.closed and body.reads == 0
    assert observation.calls == ["get_object"]
    observation.assert_no_create()


def test_revocation_in_stream_read_closes_body_before_next_read_or_result(observation, monkeypatch):
    current = [True]
    body = Body(
        b'{"ordinary":true}\n',
        observation.events,
        after_read=lambda: current.__setitem__(0, False),
    )
    read = fixed_raw_read(observation, body, monkeypatch)

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary stream observation refusal")

    with observation.backend.qualified_io(qualify):
        with pytest.raises(PermissionError):
            read()
    assert body.closed and body.reads == 1
    observation.assert_no_create()


def test_final_stream_close_revocation_cannot_expose_verified_raw_result(observation, monkeypatch):
    current = [True]
    body = Body(
        b'{"ordinary":true}\n',
        observation.events,
        after_close=lambda: current.__setitem__(0, False),
    )
    read = fixed_raw_read(observation, body, monkeypatch)

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary final exposure refusal")

    with observation.backend.qualified_io(qualify):
        with pytest.raises(PermissionError):
            read()
    assert body.closed
    observation.assert_no_create()


def lineage_stream(observation, monkeypatch):
    assert observation.kind in ("case", "prep")
    catalog = observation.handle.extra["catalog"]
    raw = b'{"ordinary":1}\n{"ordinary":2}\n'
    body = Body(raw, observation.events)
    key = observation.handle.extra["output_prefix"] + "LINEAGE.jsonl"
    ref = {
        "uri": "s3://" + catalog.bucket + "/" + key,
        "version_id": "ordinary-version",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "format": "jsonl",
    }

    def get(**kwargs):
        observation.calls.append("get_object")
        assert kwargs == {
            "Bucket": catalog.bucket,
            "Key": key,
            "VersionId": "ordinary-version",
            "ExpectedBucketOwner": catalog.account_id,
        }
        return {"VersionId": "ordinary-version", "ContentLength": len(raw), "Body": body}

    monkeypatch.setattr(observation.s3, "get_object", get)
    return body, lambda: observation.backend._lineage_objects(observation.s3, catalog, ref)


@pytest.mark.parametrize("observation", ("case", "prep"), indirect=True)
def test_lineage_generator_created_in_scope_cannot_start_after_scope_exit(observation, monkeypatch):
    body, create = lineage_stream(observation, monkeypatch)
    with observation.backend.qualified_io(observation.qualify):
        stream = create()
    with pytest.raises(ValueError):
        next(stream)
    assert observation.calls == [] and body.reads == 0
    observation.assert_no_create()


@pytest.mark.parametrize("observation", ("case", "prep"), indirect=True)
def test_started_lineage_generator_cannot_yield_pending_row_after_scope_exit(observation, monkeypatch):
    body, create = lineage_stream(observation, monkeypatch)
    with observation.backend.qualified_io(observation.qualify):
        stream = create()
        assert next(stream) == {"ordinary": 1}
    with pytest.raises(ValueError):
        next(stream)
    assert body.closed and body.reads == 1
    observation.assert_no_create()


@pytest.mark.parametrize("observation", ("case", "prep"), indirect=True)
def test_lineage_rechecks_current_callback_before_each_buffered_yield(observation, monkeypatch):
    current = [True]
    body, create = lineage_stream(observation, monkeypatch)

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary lineage yield refusal")

    with observation.backend.qualified_io(qualify):
        stream = create()
        assert next(stream) == {"ordinary": 1}
        current[0] = False
        with pytest.raises(PermissionError):
            next(stream)
    assert body.closed and body.reads == 1
    observation.assert_no_create()


def test_case_post_list_tags_refusal_blocks_artifact_reads(request, monkeypatch):
    backend, sm, s3, source, catalog = request.getfixturevalue("case_fixture")
    handle = backend.submit(source=source, catalog=catalog, runtime_session_id=CASE_SESSION)
    sm.state = "Completed"
    current = [True]
    original = sm.list_tags

    def tags(**kwargs):
        result = original(**kwargs)
        current[0] = False
        return result

    monkeypatch.setattr(sm, "list_tags", tags)
    s3.calls.clear()

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary post-tags observation refusal")

    with backend.qualified_io(qualify):
        assert backend.artifacts(handle)["frozen"] is False
    assert s3.calls == [] and len(sm.creates) == 1


@pytest.mark.usefixtures("ownership")
def test_completed_data_post_head_refusal_never_hashes_or_exposes_artifacts(monkeypatch):
    backend = data_adapter()
    handle = submit_data(backend)
    populate(handle, backend._s3)
    backend._sm.state = "Completed"
    current, gets = [True], []
    original_head = backend._s3.head_object
    original_get = backend._s3.get_object

    def head(**kwargs):
        result = original_head(**kwargs)
        current[0] = False
        return result

    def get(**kwargs):
        assert current[0], "post-Head refusal reached another stream"
        gets.append(kwargs["Key"])
        return original_get(**kwargs)

    monkeypatch.setattr(backend._s3, "head_object", head)
    monkeypatch.setattr(backend._s3, "get_object", get)

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary post-Head observation refusal")

    with backend.qualified_io(qualify):
        result = backend.artifacts(handle)
    assert result["success"] is False and not result.get("acquired", False)
    assert len(gets) == 3 and len(backend._sm.calls) == 1


@pytest.mark.parametrize("observation", ("case", "prep"), indirect=True)
def test_post_output_head_refusal_blocks_get_and_uses_actual_instance_scope(observation, monkeypatch):
    catalog = observation.handle.extra["catalog"]
    raw = b'{"ordinary":true}\n'
    key = observation.handle.extra["output_prefix"] + "QUALIFICATION.json"
    observation.s3.add(key, raw, "ordinary-version")
    current = [True]
    original = observation.s3.head_object

    def head(**kwargs):
        result = original(**kwargs)
        current[0] = False
        return result

    monkeypatch.setattr(observation.s3, "head_object", head)

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary post-Head observation refusal")

    with observation.backend.qualified_io(qualify):
        with pytest.raises(PermissionError):
            observation.backend._output(
                observation.s3,
                catalog,
                observation.handle.extra["output_prefix"],
                "QUALIFICATION.json",
                metadata=True,
                count=len(raw),
                digest=hashlib.sha256(raw).hexdigest(),
            )
    assert observation.calls == ["head_object"]
    observation.assert_no_create()


def test_case_post_inventory_refusal_blocks_closed_bundle_reads(request, monkeypatch):
    from test_coding_case_native import freeze_managed_case_bundle, rows

    backend, sm, s3, source, catalog = request.getfixturevalue("case_fixture")
    handle = backend.submit(source=source, catalog=catalog, runtime_session_id=CASE_SESSION)
    bundle = freeze_managed_case_bundle(rows(), source=source, header=handle.extra["header"])
    for name, raw in bundle.items():
        s3.add(handle.extra["output_prefix"] + name, raw)
    sm.state = "Completed"
    current = [True]
    original = s3.list_objects_v2

    def inventory(**kwargs):
        result = original(**kwargs)
        current[0] = False
        return result

    monkeypatch.setattr(s3, "list_objects_v2", inventory)
    s3.calls.clear()

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary post-inventory observation refusal")

    with backend.qualified_io(qualify):
        assert backend.artifacts(handle)["frozen"] is False
    assert [name for name, _ in s3.calls] == ["list"]
    assert len(sm.creates) == 1


def test_case_revocation_in_final_bundle_validation_withholds_success(request, monkeypatch):
    from test_coding_case_native import freeze_managed_case_bundle, rows

    from amlas.backends.adapters import coding_case_freeze

    backend, sm, s3, source, catalog = request.getfixturevalue("case_fixture")
    handle = backend.submit(source=source, catalog=catalog, runtime_session_id=CASE_SESSION)
    bundle = freeze_managed_case_bundle(rows(), source=source, header=handle.extra["header"])
    for name, raw in bundle.items():
        s3.add(handle.extra["output_prefix"] + name, raw)
    sm.state = "Completed"
    current = [True]
    original = coding_case_freeze.validate_managed_case_bundle

    def validate(*args, **kwargs):
        result = original(*args, **kwargs)
        current[0] = False
        return result

    monkeypatch.setattr(coding_case_freeze, "validate_managed_case_bundle", validate)

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary final case exposure refusal")

    with backend.qualified_io(qualify):
        result = backend.artifacts(handle)
    assert not current[0], "actual final validator was never reached"
    assert result["success"] is False and result["frozen"] is False
    assert len(sm.creates) == 1


@pytest.mark.usefixtures("ownership")
@pytest.mark.parametrize("revoke_after_read", (False, True))
def test_data_exact_version_payload_stream_uses_current_scope_and_closes(monkeypatch, revoke_after_read):
    backend = data_adapter()
    handle = submit_data(backend)
    current, events = [True], []
    raw = b"ordinary bounded byte transport"
    body = Body(
        raw,
        events,
        after_read=(lambda: current.__setitem__(0, False)) if revoke_after_read else None,
    )

    def get(**kwargs):
        events.append("get_object")
        assert kwargs == {
            "Bucket": handle.extra["bucket"],
            "Key": handle.extra["output_prefix"] + "data.jsonl",
            "VersionId": "ordinary-version",
            "ExpectedBucketOwner": handle.extra["account_id"],
        }
        return {"VersionId": "ordinary-version", "ContentLength": len(raw), "Body": body}

    monkeypatch.setattr(backend._s3, "get_object", get)

    def qualify():
        if not current[0]:
            raise PermissionError("ordinary payload byte observation refusal")
        events.append("qualify")

    with backend.qualified_io(qualify):
        if revoke_after_read:
            with pytest.raises(PermissionError):
                backend._verify_version_payload(
                    handle, "data.jsonl", "ordinary-version", hashlib.sha256(raw).hexdigest(), len(raw)
                )
        else:
            backend._verify_version_payload(
                handle, "data.jsonl", "ordinary-version", hashlib.sha256(raw).hexdigest(), len(raw)
            )
    assert body.closed
    if revoke_after_read:
        assert body.reads == 1
    else:
        for index, event in enumerate(events):
            if event != "qualify":
                assert events[index - 1] == events[index + 1] == "qualify"
    assert len(backend._sm.calls) == 1
