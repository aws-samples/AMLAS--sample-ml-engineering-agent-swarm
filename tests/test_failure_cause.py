"""Closed cause fields for refusal lines and the marker of a designed run-record recheck.

A recheck that finds its run record changed stays the exact ValueError with the same single text that every existing
handler matches; only a reader of an interrupted byte pass asks `record_change` for its closed reason. Cause fields name
a closed class, a closed native code, the innermost amlas code site and the closed root class, never a message.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError

from amlas.backends.adapters import failure_cause
from amlas.control_plane import normal_run_checkpoint

ROW, CATALOG = "RUN_RECORD_CHANGED", "SFT_CATALOG_CHANGED"
# Every run-record recheck of the normal-run checkpoint: (designed text, reason constant), one entry per raise.
RECHECKS = sorted(
    [
        ("normal prepared original selection changed", ROW),
        ("normal prepared original selection changed", CATALOG),
        ("normal original TASK selection changed", ROW),
        ("normal original TASK selection changed", CATALOG),
        ("original completion association changed", ROW),
        ("original continuation selection changed", ROW),
        ("normal original restoration row changed", ROW),
        ("normal parent SFT selection changed", CATALOG),
        ("normal original row changed during parent capture", ROW),
        ("normal parent SFT selection changed", CATALOG),
        ("normal journal original enrollment changed", ROW),
        ("normal run current SFT catalog changed", CATALOG),
        ("normal run original association changed", ROW),
    ]
)


def _raised(build):
    try:
        raise build()
    except BaseException as error:
        return error


def test_a_record_change_is_the_exact_value_error_with_the_same_text():
    error = failure_cause.record_changed("normal original TASK selection changed", failure_cause.RUN_RECORD_CHANGED)
    assert type(error) is ValueError and error.args == ("normal original TASK selection changed",)
    assert str(error) == "normal original TASK selection changed"
    assert failure_cause.record_change(error) == "run_record_changed"
    catalog = failure_cause.record_changed("normal run current SFT catalog changed", failure_cause.SFT_CATALOG_CHANGED)
    assert failure_cause.record_change(catalog) == "sft_catalog_changed"


def test_only_a_marked_exact_value_error_reads_as_a_record_change():
    class RecheckSubclassError(ValueError):
        pass

    plain = ValueError("normal original TASK selection changed")
    sub = RecheckSubclassError("normal original TASK selection changed")
    sub._amlas_record_change = "run_record_changed"
    forged = ValueError("normal original TASK selection changed")
    forged._amlas_record_change = "bytes_refused"
    marked = failure_cause.record_changed("normal original TASK selection changed", "run_record_changed")
    try:
        try:
            raise marked
        except ValueError as inner:
            raise ValueError("wrapped by a caller") from inner
    except ValueError as outer:
        wrapped = outer
    for value in (plain, sub, forged, wrapped, None, "run_record_changed", RuntimeError("x")):
        assert failure_cause.record_change(value) is None
    for text, reason in (("", "run_record_changed"), ("text", "refused"), (None, "run_record_changed")):
        with pytest.raises(ValueError):
            failure_cause.record_changed(text, reason)


@pytest.mark.parametrize(
    ("error", "name"),
    [
        (ValueError("x"), "ValueError"),
        (json.JSONDecodeError("x", "doc", 0), "ValueError"),
        (KeyError("x"), "KeyError"),
        (ConnectionResetError("x"), "ConnectionResetError"),
        (ClientError({"Error": {"Code": "SlowDown"}}, "HeadObject"), "ClientError"),
        (EndpointConnectionError(endpoint_url="https://s3.neutral.invalid"), "EndpointConnectionError"),
        (ConnectTimeoutError(endpoint_url="https://s3.neutral.invalid"), "ConnectTimeoutError"),
        (type("ModelAcquisitionRefusal", (ValueError,), {})("x"), "ValueError"),
        (type("ClientError", (Exception,), {})("x"), "Exception"),
        (type("Unreviewed", (BaseException,), {})("x"), "Exception"),
    ],
)
def test_the_class_is_the_nearest_reviewed_class_of_its_mro(error, name):
    assert failure_cause.closed_class(error) == name


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ClientError({"Error": {"Code": "SlowDown"}}, "HeadObject"), "SlowDown"),
        (ClientError({"Error": {"Code": "Throttling"}}, "GetCallerIdentity"), "Throttling"),
        (ClientError({"Error": {"Code": "ProvisionedThroughputExceededException"}}, "GetItem"), "ProvisionedThroughputExceededException"),
        (ClientError({"Error": {"Code": "404"}}, "HeadObject"), "404"),
        (ClientError({"Error": {"Code": "arn:aws:s3:::neutral-bucket/key"}}, "HeadObject"), "other"),
        (ClientError({}, "HeadObject"), "other"),
        (ValueError("SlowDown"), "none"),
        (EndpointConnectionError(endpoint_url="https://s3.neutral.invalid"), "none"),
    ],
)
def test_the_native_code_is_closed(error, code):
    assert failure_cause.native_code(error) == code


def test_the_site_is_the_innermost_amlas_frame_and_never_a_test_or_library_frame():
    inner = _raised(lambda: ValueError("raised by this test only"))
    assert failure_cause.code_site(inner) == "unknown:0"
    assert failure_cause.code_site(ValueError("never raised")) == "unknown:0"
    try:
        failure_cause.record_changed("", "run_record_changed")
    except ValueError as error:
        source, start = inspect.getsourcelines(failure_cause.record_changed)
        line = start + next(i for i, text in enumerate(source) if text.strip().startswith("raise ValueError"))
        assert failure_cause.code_site(error) == "failure_cause.py:" + str(line)


def test_cause_fields_name_class_code_site_and_root_class_never_the_message():
    try:
        try:
            raise ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "neutral-secret-identifier"}}, "GetItem"
            )
        except ClientError as cause:
            raise ValueError("outcome of neutral-secret-identifier unresolved") from cause
    except ValueError as error:
        fields = failure_cause.cause_fields(error)
    assert fields == ("ValueError", "none", "unknown:0", "ClientError")
    assert "neutral-secret-identifier" not in json.dumps(fields)
    assert failure_cause.cause_fields(ClientError({"Error": {"Code": "SlowDown"}}, "HeadObject")) == (
        "ClientError",
        "SlowDown",
        "unknown:0",
        "none",
    )


@pytest.mark.parametrize(
    ("value", "closed"),
    [
        (("ClientError", "SlowDown", "model_weight_acquisition.py:279", "none"), True),
        (("Exception", "other", "unknown:0", "ValueError"), True),
        (("ClientError", "SlowDown", "model_weight_acquisition.py:279"), False),
        (("ClientError", "free text", "model_weight_acquisition.py:279", "none"), False),
        (("ClientError", "SlowDown", "/abs/model_weight_acquisition.py:279", "none"), False),
        (("Unreviewed", "none", "unknown:0", "none"), False),
        (["ClientError", "SlowDown", "model_weight_acquisition.py:279", "none"], False),
    ],
)
def test_a_cause_is_closed_or_replaced_by_the_closed_defaults(value, closed):
    expected = tuple(value) if closed else ("Exception", "none", "unknown:0", "none")
    assert failure_cause.closed_cause(value) == expected


def test_every_run_record_recheck_of_the_checkpoint_raises_the_marked_error():
    tree = ast.parse(Path(normal_run_checkpoint.__file__).read_text(encoding="utf-8"))
    texts = {text for text, _reason in RECHECKS}
    plain, marked = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call) or not node.exc.args:
            continue
        first = node.exc.args[0]
        if not isinstance(first, ast.Constant) or first.value not in texts:
            continue
        if isinstance(node.exc.func, ast.Name) and node.exc.func.id == "record_changed":
            reason = node.exc.args[1]
            marked.append((first.value, reason.id if isinstance(reason, ast.Name) else repr(reason)))
        else:
            plain.append((first.value, node.lineno))
    assert plain == [], "a run-record recheck still raises an unmarked error"
    assert sorted(marked) == RECHECKS
