"""Neutral returned-body ownership and cancellation cleanup regressions."""

from __future__ import annotations

import asyncio
import hashlib
import io

import pytest
from test_data_case_prep_observation_io import Body, fixed_raw_read, lineage_stream

pytest_plugins = ("test_data_case_prep_observation_io",)

TIMED_READERS = (
    ("data", "payload"),
    ("case", "raw"),
    ("prep", "raw"),
    ("case", "lineage"),
    ("prep", "lineage"),
)
ALL_READERS = (("data", "manifest"), *TIMED_READERS)


class CloseFailureBody(io.BytesIO):
    def __init__(self, raw):
        super().__init__(raw)
        self.reads = 0
        self.close_attempts = 0

    def read(self, size=-1):
        self.reads += 1
        return super().read(size)

    def close(self):
        self.close_attempts += 1
        super().close()
        raise OSError("ordinary synthetic stream cleanup failure")


def reader(observation, selected, monkeypatch, *, close_failure=False):
    if selected == "lineage":
        body, create = lineage_stream(observation, monkeypatch)

        def read():
            return next(create())

        # Timer-failure tests exercise the existing valid metadata generator.
        if not close_failure:
            return body, read
    elif selected in ("manifest", "raw"):
        raw = b'{"ordinary":true}\n'
        body = CloseFailureBody(raw) if close_failure else Body(raw, observation.events)
        return body, fixed_raw_read(observation, body, monkeypatch)
    else:
        raw = b"ordinary bounded byte transport"
        body = CloseFailureBody(raw) if close_failure else Body(raw, observation.events)

        def read():
            return observation.backend._verify_version_payload(
                observation.handle,
                "data.jsonl",
                "ordinary-version",
                hashlib.sha256(raw).hexdigest(),
                len(raw),
            )

        monkeypatch.setattr(
            observation.s3,
            "get_object",
            lambda **kwargs: {
                "VersionId": "ordinary-version",
                "ContentLength": len(raw),
                "Body": body,
            },
        )
        return body, read
    # Replace only the valid returned synthetic lineage body, keeping its pins.
    original = observation.s3.get_object
    failed_body = CloseFailureBody(b'{"ordinary":1}\n{"ordinary":2}\n')

    def returned(**kwargs):
        result = original(**kwargs)
        result["Body"].close()
        result["Body"] = failed_body
        return result

    monkeypatch.setattr(observation.s3, "get_object", returned)
    return failed_body, read


@pytest.mark.parametrize("observation,selected", TIMED_READERS, indirect=("observation",))
@pytest.mark.parametrize("failure_at", ("construct", "start"))
def test_returned_body_is_owned_before_timer_construction_or_start(
    observation, selected, failure_at, monkeypatch
):
    body, read = reader(observation, selected, monkeypatch)
    lifecycle = []
    failure = RuntimeError("ordinary synthetic timer startup failure")

    class Timer:
        def __init__(self, *args, **kwargs):
            lifecycle.append("construct")
            if failure_at == "construct":
                raise failure

        def start(self):
            lifecycle.append("start")
            raise failure

        def cancel(self):
            lifecycle.append("cancel")

        def join(self, **kwargs):
            lifecycle.append("join")

    module = (
        "amlas.backends.adapters.dataset_acquisition"
        if observation.kind == "data"
        else "amlas.backends.adapters.coding_preparation"
    )
    monkeypatch.setattr(module + ".threading.Timer", Timer)
    with observation.backend.qualified_io(lambda: None):
        with pytest.raises(RuntimeError) as error:
            read()
    assert error.value is failure
    assert body.closed and body.reads == 0
    assert "cancel" not in lifecycle and "join" not in lifecycle
    observation.assert_no_create()


@pytest.mark.parametrize("observation,selected", ALL_READERS, indirect=("observation",))
def test_post_get_cancellation_survives_failing_body_close(observation, selected, monkeypatch):
    body, read = reader(observation, selected, monkeypatch, close_failure=True)
    current = [True]
    original = observation.s3.get_object
    cancellation = asyncio.CancelledError("ordinary synthetic cancellation")

    def returned(**kwargs):
        result = original(**kwargs)
        current[0] = False
        return result

    monkeypatch.setattr(observation.s3, "get_object", returned)

    def qualify():
        if not current[0]:
            raise cancellation

    with observation.backend.qualified_io(qualify):
        with pytest.raises(asyncio.CancelledError) as error:
            read()
    assert error.value is cancellation
    assert body.closed and body.reads == 0 and body.close_attempts == 1
    observation.assert_no_create()


def test_actual_artifact_cancellation_is_not_converted_to_unknown_or_failed_data(observation, monkeypatch):
    if observation.kind == "data":
        from test_managed_dataset_acquisition import populate

        populate(observation.handle, observation.s3)
        observation.sm.state = "Completed"
    elif observation.kind == "case":
        from test_coding_case_native import freeze_managed_case_bundle, rows

        bundle = freeze_managed_case_bundle(
            rows(), source=observation.handle.extra["source"], header=observation.handle.extra["header"]
        )
        for name, raw in bundle.items():
            observation.s3.add(observation.handle.extra["output_prefix"] + name, raw)
        observation.sm.state = "Completed"
    else:
        from test_managed_coding_preparation import complete_outputs

        complete_outputs(
            (
                observation.backend,
                observation.sm,
                observation.s3,
                observation.handle.extra["catalog"],
                observation.handle.extra["sources"][0],
            ),
            observation.handle,
        )
    current, bodies = [True], []
    original = observation.s3.get_object
    cancellation = asyncio.CancelledError("ordinary artifact cancellation")

    def returned(**kwargs):
        result = original(**kwargs)
        body = result["Body"]
        raw = body.read()
        body.close()
        failed = CloseFailureBody(raw)
        bodies.append(failed)
        result["Body"] = failed
        current[0] = False
        return result

    monkeypatch.setattr(observation.s3, "get_object", returned)

    def qualify():
        if not current[0]:
            raise cancellation

    with observation.backend.qualified_io(qualify):
        with pytest.raises(asyncio.CancelledError) as error:
            observation.backend.artifacts(observation.handle)
    assert error.value is cancellation
    assert len(bodies) == 1 and bodies[0].closed and bodies[0].reads == 0
    assert bodies[0].close_attempts == 1
    observation.assert_no_create()


def test_caller_exception_context_cannot_hide_normal_body_close_failure(observation):
    body = CloseFailureBody(b"ordinary")
    try:
        raise RuntimeError("ordinary surrounding caller error")
    except RuntimeError:
        with pytest.raises(OSError, match="ordinary synthetic stream cleanup failure"):
            with observation.backend._owned_observation_body(body):
                pass
    assert body.closed and body.reads == 0 and body.close_attempts == 1
    observation.assert_no_create()
