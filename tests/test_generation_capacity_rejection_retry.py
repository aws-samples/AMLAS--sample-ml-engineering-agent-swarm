"""A capacity rejection of a generation create is recorded but is not a permanent answer.

SageMaker answers ResourceLimitExceeded when another training job of the same instance type holds the account's
quota. The rejected create made no job. A later submit with the same arguments, after capacity frees up, must reach
CreateTrainingJob again. An ambiguous outcome (lost response) keeps the existing one-unknown-create rule.
"""

from __future__ import annotations

import dataclasses
import threading
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from test_managed_generation_product import (  # noqa: F401  (pytest fixture)
    CASE_REF,
    MODEL_REF,
    resolved_spec,
    state,
)
from test_qwen_sft_backend import Native

QUOTA = ClientError(
    {
        "Error": {
            "Code": "ResourceLimitExceeded",
            "Message": "The account-level service limit 'ml.g6e.4xlarge for training job usage' is 1 Instances, with current utilization of 1 Instances and a request delta of 1 Instances.",
        }
    },
    "CreateTrainingJob",
)


def _rejection(code):
    return ClientError(
        {"Error": {"Code": code, "Message": "neutral definite rejection"}}, "CreateTrainingJob"
    )


def _real_adapter(fixture_state, native):
    from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend

    store, _, reads, s3, _inputs, case = fixture_state
    s3.install(dict(case.file_refs)["PROMPTS.jsonl"], b"{}\n")
    store._adapter = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    return store, resolved_spec(store, reads)


def test_capacity_rejected_generation_is_created_when_submitted_again(state):  # noqa: F811
    from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend

    store, _, reads, s3, _inputs, case = state
    native = Native()
    native.error = QUOTA
    s3.install(dict(case.file_refs)["PROMPTS.jsonl"], b"{}\n")
    store._adapter = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    spec = resolved_spec(store, reads)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == "submission_rejected" and first["accepted"] is False and len(native.calls) == 1
    native.error = None  # the other job of the type ended; the slot is free
    again = store.submit(MODEL_REF, CASE_REF, spec)
    assert len(native.calls) == 2, (
        "a definite ResourceLimitExceeded rejection created no job, yet the same request was answered from the cached "
        f"rejection (reused={again.get('reused')}, status={again.get('status')}) and never reached CreateTrainingJob again"
    )
    assert again["accepted"] is True and again["status"] == "submitted"


def test_lost_generation_create_response_keeps_one_unknown_create(state):  # noqa: F811
    """Control: an ambiguous outcome (read timeout after the create) still never repeats the create (existing rule, must stay green)."""
    from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend

    store, _, reads, s3, _inputs, case = state
    native = Native()
    native.error = TimeoutError("read timeout after the create was applied")
    s3.install(dict(case.file_refs)["PROMPTS.jsonl"], b"{}\n")
    store._adapter = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    spec = resolved_spec(store, reads)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    native.error = None
    again = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == again["status"] == "acceptance_unknown" and len(native.calls) == 1


def test_capacity_rejection_records_its_native_code_on_the_original_handle(state):  # noqa: F811
    native = Native()
    native.error = QUOTA
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    handle = store._records[first["request_ref"]]
    assert handle.extra["submission_state"] == "rejected"
    assert handle.extra["rejection_code"] == "ResourceLimitExceeded"
    assert first["reused"] is False and len(native.calls) == 1


def test_capacity_rejected_twice_then_created_once_and_then_reused(state):  # noqa: F811
    native = Native()
    native.error = QUOTA
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    second = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == second["status"] == "submission_rejected"
    assert second["reused"] is False and len(native.calls) == 2
    native.error = None
    third = store.submit(MODEL_REF, CASE_REF, spec)
    assert third["status"] == "submitted" and third["accepted"] is True and len(native.calls) == 3
    fourth = store.submit(MODEL_REF, CASE_REF, spec)
    assert fourth["status"] == "submitted" and fourth["reused"] is True and len(native.calls) == 3
    # One request keeps one reference and one deterministic native name across every attempt.
    assert len({value["request_ref"] for value in (first, second, third, fourth)}) == 1
    assert len({call["TrainingJobName"] for call in native.calls}) == 1
    assert store._records[third["request_ref"]].extra["submission_state"] == "accepted"


@pytest.mark.parametrize("code", ["ValidationException", "AccessDeniedException"])
def test_other_definite_rejections_stay_the_cached_answer(state, code):  # noqa: F811
    native = Native()
    native.error = _rejection(code)
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    native.error = None
    again = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == again["status"] == "submission_rejected"
    assert again["reused"] is True and again["request_ref"] == first["request_ref"] and len(native.calls) == 1
    assert store._records[first["request_ref"]].extra["rejection_code"] == code


def test_a_handle_without_the_capacity_code_is_never_retried(state):  # noqa: F811
    """Only the recorded native code reopens a request: a rejected handle that names none stays the answer."""
    native = Native()
    native.error = QUOTA
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    handle = store._records[first["request_ref"]]
    store._records[first["request_ref"]] = dataclasses.replace(
        handle, extra={key: value for key, value in handle.extra.items() if key != "rejection_code"}
    )
    native.error = None
    again = store.submit(MODEL_REF, CASE_REF, spec)
    assert again["status"] == "submission_rejected" and again["reused"] is True and len(native.calls) == 1


class _ReloadAdapter:
    """Neutral SFT port: reports the scripted state of the reload job that holds the generation type."""

    def __init__(self):
        self.state = "RUNNING"
        self.observed = []

    def status(self, handle, **private):
        from amlas.backends.types import JobState, JobStatus

        self.observed.append(handle.id)
        return JobStatus(JobState(self.state))


def _hold_with_reload(store):
    from amlas.backends.types import JobHandle

    reload_job = "owned-sft-reload-holder"
    handle = JobHandle(
        reload_job,
        "qwen_sft",
        "arn:aws:sagemaker:us-west-2:111122223333:training-job/" + reload_job,
        {
            "submission_state": "accepted",
            "operation": "reload",
            "request": {"ResourceConfig": {"InstanceType": "ml.g6e.4xlarge", "InstanceCount": 1}},
        },
    )
    sft = store._sft
    sft._lock, sft._records, sft._adapter = threading.RLock(), {"sft_" + "c" * 32: handle}, _ReloadAdapter()
    return reload_job, sft._adapter


def test_capacity_rejection_tells_the_model_which_job_holds_the_type_and_a_later_submit_creates(state):  # noqa: F811
    """The base generation meets the reload of this run on the only ml.g6e.4xlarge slot."""
    from amlas.backends.adapters._util import raw_tool
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
    from amlas.tools.coding_generation import submit_coding_generation

    native = Native()
    native.error = QUOTA
    store, spec = _real_adapter(state, native)
    reload_job, holder = _hold_with_reload(store)
    context = SimpleNamespace(
        invocation_state={
            "_amlas_generation_store": store,
            "_amlas_approval_policy": ApprovalPolicy(
                mode=ApprovalMode.ALLOWLIST, allowlist=["coding_generation_flow"], max_auto_cost=None
            ),
        }
    )
    first = raw_tool(submit_coding_generation)(MODEL_REF, CASE_REF, spec, tool_context=context)
    assert first["status"] == "submission_rejected" and first["accepted"] is False and len(native.calls) == 1
    assert first["instance_types"] == ["ml.g6e.16xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge"]
    assert first["blocking_instance_type"] == "ml.g6e.4xlarge"
    assert first["type_holders"] == {
        "ml.g6e.16xlarge": None,
        "ml.g6e.4xlarge": reload_job,
        "ml.g6e.8xlarge": None,
    }
    assert first["holding_job_name"] == reload_job and first["waiting_job_names"] == []
    # Only the type SageMaker named is in the note: ml.g6e.16xlarge and 8xlarge were free and are not called held.
    assert first["next_step"] == (
        "Job " + reload_job + " of this run holds the ml.g6e.4xlarge capacity. Nothing was created. "
        "Submit this arm again with the same arguments after that job ends."
    )
    # The reload ends and frees the slot; the same call is a real create.
    holder.state, native.error = "SUCCEEDED", None
    again = raw_tool(submit_coding_generation)(MODEL_REF, CASE_REF, spec, tool_context=context)
    assert again["status"] == "submitted" and again["accepted"] is True and again["executed"] is True
    assert len(native.calls) == 2 and again["request_ref"] == first["request_ref"]
    assert "next_step" not in again and "holding_job_name" not in again


def test_capacity_rejection_without_a_holder_in_this_run_names_none(state):  # noqa: F811
    native = Native()
    native.error = QUOTA
    store, spec = _real_adapter(state, native)
    _reload_job, holder = _hold_with_reload(store)
    holder.state = "SUCCEEDED"
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == "submission_rejected" and first["holding_job_name"] is None
    assert first["blocking_instance_type"] == "ml.g6e.4xlarge"
    assert first["next_step"] == (
        "Another job in the account holds the ml.g6e.4xlarge capacity. Nothing was created. "
        "Submit this arm again with the same arguments after that job ends."
    )


def test_holder_lookup_reads_a_running_generation_of_this_run_and_skips_ended_or_other_types(state):  # noqa: F811
    from amlas.control_plane.sft_state import GENERATION_INSTANCE_PREFERENCES

    native = Native()
    store, spec = _real_adapter(state, native)
    base = store.submit(MODEL_REF, CASE_REF, spec)
    assert base["status"] == "submitted"
    other = "gen_" + "f" * 32
    # The fake describes the first listed type as SageMaker's selection: the base job holds ml.g6e.16xlarge.
    free = {name: None for name in GENERATION_INSTANCE_PREFERENCES}
    assert store._active_holders(GENERATION_INSTANCE_PREFERENCES, other) == (
        {**free, "ml.g6e.16xlarge": base["job_name"]},
        [],
    )
    assert store._active_holders(("ml.g6e.12xlarge",), other) == ({"ml.g6e.12xlarge": None}, [])
    assert store._active_holders(GENERATION_INSTANCE_PREFERENCES, base["request_ref"]) == (free, [])
    native.state = "Completed"
    assert store._active_holders(GENERATION_INSTANCE_PREFERENCES, other) == (free, [])


def test_other_rejections_carry_no_capacity_note(state):  # noqa: F811
    native = Native()
    native.error = _rejection("ValidationException")
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == "submission_rejected"
    assert "rejection_type" not in store._records[first["request_ref"]].extra
    assert not {
        "instance_type",
        "instance_types",
        "blocking_instance_type",
        "type_holders",
        "waiting_job_names",
        "holding_job_name",
        "next_step",
    } & set(first)


# --- the one type the native message names, as a closed field ----------------------------------------------------

_LIMIT = "The account-level service limit '%s for training job usage' is 1 Instances, with current utilization of 1 Instances and a request delta of 1 Instances."


def _limit(message):
    return ClientError({"Error": {"Code": "ResourceLimitExceeded", "Message": message}}, "CreateTrainingJob")


@pytest.mark.parametrize(
    "message,expected",
    [
        (_LIMIT % "ml.g6e.4xlarge", "ml.g6e.4xlarge"),
        (_LIMIT % "ml.g6e.8xlarge", "ml.g6e.8xlarge"),
        (_LIMIT % "ml.g6e.8xlarge" + " " + _LIMIT % "ml.g6e.8xlarge", "ml.g6e.8xlarge"),
        # ml.g6e.16xlarge is a generation type since 2026-11-11 (first in the list).
        (_LIMIT % "ml.g6e.16xlarge", "ml.g6e.16xlarge"),
        # A type the generation request does not list is not kept.
        (_LIMIT % "ml.g6e.12xlarge", None),
        (_LIMIT % "ml.g6e.2xlarge", None),
        (_LIMIT % "ml.p4d.24xlarge", None),
        # Two different types, another wording, another case, no type, an over-long or non-string message.
        (_LIMIT % "ml.g6e.4xlarge" + " " + _LIMIT % "ml.g6e.8xlarge", None),
        ("The account-level service limit 'ml.g6e.4xlarge for processing job usage' is 1 Instances.", None),
        ("ml.g6e.4xlarge for training job usage", None),
        (_LIMIT % "ML.G6E.4XLARGE", None),
        (_LIMIT % "ml.g6e.4xlarge; neutral other text", None),
        ("neutral limit message", None),
        ("", None),
        (_LIMIT % "ml.g6e.4xlarge" + " " * 2048, None),
        (None, None),
        (7, None),
        (["ml.g6e.4xlarge for training job usage"], None),
    ],
)
def test_rejection_type_is_one_listed_type_or_none(state, message, expected):  # noqa: F811
    native = Native()
    native.error = _limit(message)
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    extra = store._records[first["request_ref"]].extra
    assert extra["rejection_code"] == "ResourceLimitExceeded" and extra["rejection_type"] == expected
    assert first["status"] == "submission_rejected" and first["blocking_instance_type"] == expected
    assert not [value for value in extra.values() if type(value) is str and "service limit" in value]
    if expected is None:
        assert first["holding_job_name"] is None
        assert first["next_step"] == (
            "SageMaker did not name a listed instance type as the one at its limit. Nothing was created. "
            "Submit this arm again with the same arguments after a job on a listed type ends."
        )
    else:
        assert first["next_step"] == (
            "Another job in the account holds the " + expected + " capacity. Nothing was created. "
            "Submit this arm again with the same arguments after that job ends."
        )


def test_unnamed_limit_still_names_a_holder_of_this_run_as_supporting_detail(state):  # noqa: F811
    native = Native()
    native.error = _limit("neutral limit message")
    store, spec = _real_adapter(state, native)
    reload_job, _holder = _hold_with_reload(store)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["blocking_instance_type"] is None and first["holding_job_name"] == reload_job
    assert first["next_step"] == (
        "SageMaker did not name a listed instance type as the one at its limit. "
        "Job " + reload_job + " of this run holds the ml.g6e.4xlarge capacity. Nothing was created. "
        "Submit this arm again with the same arguments after a job on a listed type ends."
    )


@pytest.mark.parametrize("code", ["ValidationException", "AccessDeniedException"])
def test_other_rejection_codes_record_no_rejection_type(state, code):  # noqa: F811
    native = Native()
    native.error = ClientError(
        {"Error": {"Code": code, "Message": _LIMIT % "ml.g6e.4xlarge"}}, "CreateTrainingJob"
    )
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    extra = store._records[first["request_ref"]].extra
    assert extra["rejection_code"] == code and "rejection_type" not in extra
    assert "blocking_instance_type" not in first and "next_step" not in first


def test_a_capacity_rejection_without_a_response_records_none(state):  # noqa: F811
    """A capacity code on an error object with a malformed response keeps the code and names no type."""

    class ShapedError(Exception):
        def __init__(self):
            super().__init__("neutral")
            self.response = {
                "Error": {"Code": "ResourceLimitExceeded", "Message": {"text": _LIMIT % "ml.g6e.4xlarge"}}
            }

    native = Native()
    native.error = ShapedError()
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    extra = store._records[first["request_ref"]].extra
    assert extra["rejection_code"] == "ResourceLimitExceeded" and extra["rejection_type"] is None
