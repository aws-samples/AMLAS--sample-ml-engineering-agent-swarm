"""Neutral current qualification at actual model adapter native read/write edges."""

import pytest
from test_model_processing_plan import adapter, namespace, plan  # noqa: F401


def test_current_scope_gates_every_destination_read_and_actual_create(monkeypatch):
    backend, sm, s3 = adapter()
    value = plan(backend)
    calls = []

    def qualify():
        calls.append("qualify")

    for obj, name in (
        (s3, "get_bucket_location"),
        (s3, "get_bucket_versioning"),
        (sm, "create_processing_job"),
    ):
        original = getattr(obj, name)

        def observed(*args, _original=original, _name=name, **kwargs):
            assert calls[-1] == "qualify"
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(obj, name, observed)
    with backend.qualified_io(qualify):
        submitted = backend.submit_prepared(value)
    assert submitted.extra["accepted"] is True
    for index, call in enumerate(calls):
        if call != "qualify":
            assert calls[index - 1] == calls[index + 1] == "qualify"


def test_revocation_after_bucket_read_prevents_versioning_and_create(monkeypatch):
    backend, sm, s3 = adapter()
    value = plan(backend)
    allowed = [True]
    original = s3.get_bucket_location

    def revoked(**kwargs):
        result = original(**kwargs)
        allowed[0] = False
        return result

    monkeypatch.setattr(s3, "get_bucket_location", revoked)
    monkeypatch.setattr(
        s3, "get_bucket_versioning", lambda **kwargs: pytest.fail("revocation reached another SDK leg")
    )

    def qualify():
        if not allowed[0]:
            raise PermissionError("ordinary current boundary refusal")

    with backend.qualified_io(qualify):
        with pytest.raises(PermissionError):
            backend.submit_prepared(value)
    assert sm.requests == []


def test_scope_reset_preserves_disabled_legacy_behavior():
    backend, sm, _ = adapter()

    def refused():
        raise PermissionError("ordinary scoped refusal")

    with backend.qualified_io(refused):
        with pytest.raises(PermissionError):
            backend.submit_prepared(plan(backend))
    submitted = backend.submit_prepared(plan(backend))
    assert submitted.extra["accepted"] is True and len(sm.requests) == 1


def test_inherited_adapter_refuses_before_scope_activation():
    backend, _, _ = adapter()
    backend._qualified_io_pid = -1
    with pytest.raises(ValueError):
        with backend.qualified_io(lambda: None):
            pytest.fail("inherited adapter activated IO context")


def test_copied_adapter_scope_expires_with_owning_scope():
    from contextvars import copy_context

    backend, sm, _ = adapter()
    value = plan(backend)
    with backend.qualified_io(lambda: None):
        saved = copy_context()
    with pytest.raises(ValueError):
        saved.run(backend.submit_prepared, value)
    assert sm.requests == []


def test_post_get_qualification_denial_closes_stream_without_read(monkeypatch):
    import io
    import time

    backend, _, s3 = adapter()
    value = plan(backend)
    allowed = [True]
    body = io.BytesIO(b"ordinary")
    s3.objects[value.extra["output_prefix"] + "COMMITTED.json"] = ("version-one", b"ordinary")

    def returned(**kwargs):
        allowed[0] = False
        return {"VersionId": "version-one", "ContentLength": 8, "Body": body}

    monkeypatch.setattr(s3, "get_object", returned)

    def qualify():
        if not allowed[0]:
            raise PermissionError("ordinary post-Get qualification refusal")

    with backend.qualified_io(qualify):
        with pytest.raises(PermissionError):
            backend._object(value, "COMMITTED.json", limit=64, deadline=time.monotonic() + 30)
    assert body.closed


def test_post_create_revocation_retains_exact_unknown_identity(monkeypatch):
    backend, sm, _ = adapter()
    value = plan(backend)
    allowed = [True]
    original = sm.create_processing_job

    def returned(**kwargs):
        result = original(**kwargs)
        allowed[0] = False
        return result

    monkeypatch.setattr(sm, "create_processing_job", returned)

    def qualify():
        if not allowed[0]:
            raise PermissionError("ordinary post-Create qualification refusal")

    with backend.qualified_io(qualify):
        submitted = backend.submit_prepared(value)
    assert submitted.id == value.id and submitted.arn == value.arn
    assert submitted.extra["submission_ambiguous"] is True and submitted.extra["accepted"] is False
    assert len(sm.requests) == 1
