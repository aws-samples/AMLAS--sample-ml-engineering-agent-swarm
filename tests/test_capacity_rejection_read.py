"""Reading a capacity-refused generation or paired evaluation answers the refusal itself.

A ResourceLimitExceeded refusal made no native job, so a later read of that reference has no job to describe. The
read must answer what the submit answered: the closed state `submission_rejected` with the reason `capacity_limit`,
the listed type SageMaker named when it named one, the job of this run that holds a listed type now when there is
one, and a plain next step. While a job of this run holds the type the next step is the submit's own sentence; once
no running job of this run is found on a listed type it says to submit now, and it names the jobs of this run that
wait without a type or could not be read instead of stating that nothing holds the type. A read creates nothing,
describes nothing under the refused name, retries nothing and leaves the stored record as the refusal wrote it.
Strict installed-SDK fakes, no AWS call.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from test_capacity_preference_lists import (
    AFTER_THAT_JOB,
    GENERATION,
    QuotaNative,
    _SftHolder,
    _tuned_checkpoint,
)
from test_capacity_rejection_retry_pair import SequencePairAdapter, _pair_store
from test_generation_capacity_rejection_retry import _limit, _real_adapter, _rejection
from test_managed_generation_product import (  # noqa: F401  (pytest fixture)
    CASE_REF,
    MODEL_REF,
    SFT_REF,
    state,
)

NOTE_FIELDS = {
    "instance_types",
    "blocking_instance_type",
    "type_holders",
    "waiting_job_names",
    "holding_job_name",
    "next_step",
}
SUBMIT_NOW = (
    "Nothing was created. Submit this arm again with the same arguments now. "
    "If another job holds a listed type, SageMaker refuses again and creates nothing."
)
PAIR_NEXT_STEP = (
    "SageMaker refused this paired evaluation for a capacity limit. Nothing was created. "
    "Submit the paired evaluation again with the same arguments. "
    "If a job in the account still holds that capacity, SageMaker refuses again and creates nothing."
)


class CountingNative(QuotaNative):
    """QuotaNative that records the name of every job a read describes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.described = []

    def describe_training_job(self, **kw):
        self.described.append(kw["TrainingJobName"])
        return super().describe_training_job(**kw)


def _policy():
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy

    return ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST,
        allowlist=["coding_generation_flow", "coding_pair_flow"],
        max_auto_cost=None,
    )


def _get_generation(store, reference):
    from amlas.backends.adapters._util import raw_tool
    from amlas.tools.coding_generation import get_coding_generation

    context = SimpleNamespace(
        invocation_state={"_amlas_generation_store": store, "_amlas_approval_policy": _policy()}
    )
    return raw_tool(get_coding_generation)(reference, tool_context=context)


def _refused_tuned(fixture_state, monkeypatch, native):
    """Base generation accepted on ml.g6e.16xlarge (first listed), tuned generation refused for that type."""
    store, spec = _real_adapter(fixture_state, native)
    _tuned_checkpoint(store, fixture_state[3], monkeypatch)
    base = store.submit(MODEL_REF, CASE_REF, spec)
    tuned = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
    assert base["status"] == "submitted" and tuned["status"] == "submission_rejected"
    return store, spec, base, tuned


def _refusal(read, submitted):
    """The closed refusal fields every read of a capacity-refused generation carries."""
    assert read["status"] == "submission_rejected" and read["rejection_reason"] == "capacity_limit"
    assert read["success"] is False and read["accepted"] is False and read["generated"] is False
    assert read["quality_success"] is False and read["native_admission"] is False
    assert read["reused"] is False
    for name in ("request_ref", "job_name", "job_arn", "arm", "evaluation_id"):
        assert read[name] == submitted[name]
    assert read["instance_types"] == list(GENERATION)
    assert NOTE_FIELDS <= set(read)
    assert not {"SelectedInstanceType", "SelectedInstanceCount", "error_type"} & set(read)


# --- generation: read while the holder runs ------------------------------------------------------------------------


def test_read_after_refusal_while_the_holder_runs_answers_the_refusal(state, monkeypatch):  # noqa: F811
    native = CountingNative()
    store, _spec, base, tuned = _refused_tuned(state, monkeypatch, native)
    creates, jobs = len(native.calls), dict(native.jobs)
    native.described.clear()
    read = store.observe(tuned["request_ref"])
    _refusal(read, tuned)
    assert (
        read["blocking_instance_type"] == "ml.g6e.16xlarge" and read["holding_job_name"] == base["job_name"]
    )
    assert read["type_holders"] == {
        "ml.g6e.16xlarge": base["job_name"],
        "ml.g6e.4xlarge": None,
        "ml.g6e.8xlarge": None,
    }
    assert read["waiting_job_names"] == []
    # The same sentence the submit answered.
    assert read["next_step"] == tuned["next_step"]
    assert read["next_step"] == (
        "Job " + base["job_name"] + " of this run holds the ml.g6e.16xlarge capacity. " + AFTER_THAT_JOB
    )
    # Read-only: no create, and only the running job of this run was described (the refused name has no job).
    assert len(native.calls) == creates and native.jobs == jobs
    assert native.described and set(native.described) == {base["job_name"]}


def test_the_registered_read_tool_answers_the_same_refusal(state, monkeypatch):  # noqa: F811
    native = CountingNative()
    store, _spec, base, tuned = _refused_tuned(state, monkeypatch, native)
    creates = len(native.calls)
    read = _get_generation(store, tuned["request_ref"])
    _refusal(read, tuned)
    assert read["executed"] is False and read["holding_job_name"] == base["job_name"]
    assert read["next_step"] == tuned["next_step"] and len(native.calls) == creates


# --- generation: read after the holder ended -----------------------------------------------------------------------


def test_read_after_the_holder_ended_says_submit_now_and_the_submit_creates_once(state, monkeypatch):  # noqa: F811
    native = CountingNative()
    store, spec, base, tuned = _refused_tuned(state, monkeypatch, native)
    native.complete(base["job_name"])
    creates = len(native.calls)
    for read in (store.observe(tuned["request_ref"]), _get_generation(store, tuned["request_ref"])):
        _refusal(read, tuned)
        assert read["blocking_instance_type"] == "ml.g6e.16xlarge" and read["holding_job_name"] is None
        assert read["type_holders"] == {name: None for name in GENERATION}
        assert (
            read["next_step"]
            == "No running job of this run was found on the ml.g6e.16xlarge capacity. " + SUBMIT_NOW
        )
    assert len(native.calls) == creates and list(native.jobs) == [base["job_name"]]
    # The submit-again path still creates exactly once and then reuses.
    again = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
    assert again["status"] == "submitted" and again["request_ref"] == tuned["request_ref"]
    assert len(native.calls) == creates + 1 and "next_step" not in again
    reused = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
    assert reused["reused"] is True and len(native.calls) == creates + 1
    # The accepted original reads as a job again: its native state and selected instance, no refusal fields.
    observed = store.observe(again["request_ref"])
    assert observed["status"] == "RUNNING" and observed["accepted"] is True
    assert (observed["SelectedInstanceType"], observed["SelectedInstanceCount"]) == ("ml.g6e.16xlarge", 1)
    assert not ({"rejection_reason"} | NOTE_FIELDS) & set(observed)


# --- a read never creates, retries or changes the record -----------------------------------------------------------


def test_a_read_never_creates_and_leaves_the_refusal_record_unchanged(state, monkeypatch):  # noqa: F811
    native = CountingNative()
    store, _spec, base, tuned = _refused_tuned(state, monkeypatch, native)
    handle = store._records[tuned["request_ref"]]
    extra_keys, stored = (
        sorted(handle.extra),
        copy.deepcopy(
            {key: handle.extra[key] for key in ("submission_state", "rejection_code", "rejection_type")}
        ),
    )
    assert stored == {
        "submission_state": "rejected",
        "rejection_code": "ResourceLimitExceeded",
        "rejection_type": "ml.g6e.16xlarge",
    }
    creates, records, keys = len(native.calls), dict(store._records), dict(store._keys)
    # Capacity frees up between the reads: a read still creates nothing and the next step changes to submit now.
    reads = [store.observe(tuned["request_ref"]), _get_generation(store, tuned["request_ref"])]
    native.complete(base["job_name"])
    reads += [store.observe(tuned["request_ref"]), _get_generation(store, tuned["request_ref"])]
    assert [read["status"] for read in reads] == ["submission_rejected"] * 4
    assert [read["holding_job_name"] for read in reads] == [base["job_name"]] * 2 + [None] * 2
    assert len(native.calls) == creates and list(native.jobs) == [base["job_name"]]
    assert tuned["job_name"] not in native.described
    assert store._records == records and store._keys == keys
    assert store._records[tuned["request_ref"]] is handle and sorted(handle.extra) == extra_keys
    assert {key: handle.extra[key] for key in stored} == stored


# --- holders outside the run, on another listed type, or unnamed ---------------------------------------------------


def test_read_of_a_refusal_for_a_job_outside_the_run_names_no_holder(state):  # noqa: F811
    native = CountingNative(outside=("ml.g6e.8xlarge",))
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == "submission_rejected" and first["blocking_instance_type"] == "ml.g6e.8xlarge"
    read = store.observe(first["request_ref"])
    _refusal(read, first)
    assert read["blocking_instance_type"] == "ml.g6e.8xlarge" and read["holding_job_name"] is None
    assert (
        read["next_step"]
        == "No running job of this run was found on the ml.g6e.8xlarge capacity. " + SUBMIT_NOW
    )
    assert len(native.calls) == 1 and native.jobs == {} and native.described == []


def test_read_names_a_job_of_this_run_on_another_listed_type(state):  # noqa: F811
    """The named type was held outside the run; a fine-tune of this run holds the other listed type now."""
    native = CountingNative(outside=("ml.g6e.4xlarge",))
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["blocking_instance_type"] == "ml.g6e.4xlarge" and first["holding_job_name"] is None
    holder = _SftHolder(
        store,
        "owned-sft-train",
        ("ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge"),
        "ml.g6e.8xlarge",
    )
    read = store.observe(first["request_ref"])
    _refusal(read, first)
    # `holding_job_name` stays the holder of the named type; the job on the other listed type is in `type_holders`.
    assert read["blocking_instance_type"] == "ml.g6e.4xlarge" and read["holding_job_name"] is None
    assert read["type_holders"] == {
        "ml.g6e.16xlarge": None,
        "ml.g6e.4xlarge": None,
        "ml.g6e.8xlarge": "owned-sft-train",
    }
    assert read["next_step"] == (
        "No running job of this run was found on the ml.g6e.4xlarge capacity. "
        "Job owned-sft-train of this run holds the ml.g6e.8xlarge capacity. " + AFTER_THAT_JOB
    )
    assert holder.observed == ["owned-sft-train"] and len(native.calls) == 1


def test_read_of_an_unnamed_limit_with_and_without_a_holder_of_this_run(state):  # noqa: F811
    from test_qwen_sft_backend import Native

    native = Native()
    native.error = _limit("neutral limit message")
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["blocking_instance_type"] is None
    read = store.observe(first["request_ref"])
    _refusal(read, first)
    assert read["blocking_instance_type"] is None and read["holding_job_name"] is None
    assert read["next_step"] == (
        "SageMaker did not name a listed instance type as the one at its limit. "
        "No running job of this run was found on a listed instance type. " + SUBMIT_NOW
    )
    holder = _SftHolder(
        store,
        "owned-sft-train",
        ("ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge"),
        "ml.g6e.4xlarge",
    )
    held = store.observe(first["request_ref"])
    _refusal(held, first)
    assert held["blocking_instance_type"] is None and held["holding_job_name"] == "owned-sft-train"
    assert held["next_step"] == (
        "SageMaker did not name a listed instance type as the one at its limit. "
        "Job owned-sft-train of this run holds the ml.g6e.4xlarge capacity. Nothing was created. "
        "Submit this arm again with the same arguments after a job on a listed type ends."
    )
    assert holder.observed == ["owned-sft-train"] and len(native.calls) == 1


def test_read_note_names_every_job_of_this_run_on_the_other_listed_types():
    """The note is built for any pinned list: two jobs of this run on two other listed types are both named."""
    from amlas.control_plane.generation_state import SessionCodingGenerationStore

    listed = ("ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.8xlarge")
    holders = {"ml.g6e.16xlarge": None, "ml.g6e.12xlarge": "owned-a", "ml.g6e.8xlarge": "owned-b"}
    note = SessionCodingGenerationStore._capacity_note(listed, holders, [], "ml.g6e.16xlarge", read=True)
    assert note["blocking_instance_type"] == "ml.g6e.16xlarge" and note["holding_job_name"] is None
    assert note["next_step"] == (
        "No running job of this run was found on the ml.g6e.16xlarge capacity. "
        "Job owned-a of this run holds the ml.g6e.12xlarge capacity. "
        "Job owned-b of this run holds the ml.g6e.8xlarge capacity. Nothing was created. "
        "Submit this arm again with the same arguments after those jobs end."
    )
    # The submit's note for the same holders is unchanged: it names only the type SageMaker named.
    fresh = SessionCodingGenerationStore._capacity_note(listed, holders, [], "ml.g6e.16xlarge")
    assert fresh["holding_job_name"] is None and fresh["next_step"] == (
        "Another job in the account holds the ml.g6e.16xlarge capacity. " + AFTER_THAT_JOB
    )


def test_a_submit_that_replaces_the_refusal_during_a_read_is_what_the_read_observes(state, monkeypatch):  # noqa: F811
    """The holders are read outside the store lock; a submit that creates the job meanwhile wins."""
    native = CountingNative()
    store, spec, base, tuned = _refused_tuned(state, monkeypatch, native)
    native.complete(base["job_name"])
    creates, lookup = len(native.calls), store._observed_holders

    def submit_during_lookup(instance_types, reference):
        result = lookup(instance_types, reference)
        again = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
        assert again["status"] == "submitted" and again["request_ref"] == tuned["request_ref"]
        return result

    monkeypatch.setattr(store, "_observed_holders", submit_during_lookup)
    read = store.observe(tuned["request_ref"])
    assert read["status"] == "RUNNING" and read["accepted"] is True and read["job_name"] == tuned["job_name"]
    assert (read["SelectedInstanceType"], read["SelectedInstanceCount"]) == ("ml.g6e.16xlarge", 1)
    assert not ({"rejection_reason"} | NOTE_FIELDS) & set(read)
    assert len(native.calls) == creates + 1


def test_a_submit_refused_again_during_a_read_is_answered_from_its_own_refusal(state, monkeypatch):  # noqa: F811
    """The read still answers the refusal, with the type the newer refusal named, and never UNKNOWN."""
    native = CountingNative()
    store, spec, base, tuned = _refused_tuned(state, monkeypatch, native)
    first = store._records[tuned["request_ref"]]
    # The base job ends, and a job outside the run takes another listed type before the tuned arm is submitted again.
    native.complete(base["job_name"])
    native.account.held["ml.g6e.8xlarge"] = "outside-account-job"
    creates, lookup, lookups = len(native.calls), store._observed_holders, []

    def submit_during_lookup(instance_types, reference):
        result = lookup(instance_types, reference)
        lookups.append(reference)
        if len(lookups) == 1:  # the read's lookup; the refused submit below makes its own
            again = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
            assert again["status"] == "submission_rejected"
            assert again["blocking_instance_type"] == "ml.g6e.8xlarge"
        return result

    monkeypatch.setattr(store, "_observed_holders", submit_during_lookup)
    read = store.observe(tuned["request_ref"])
    _refusal(read, tuned)
    assert store._records[tuned["request_ref"]] is not first
    assert read["blocking_instance_type"] == "ml.g6e.8xlarge" and read["holding_job_name"] is None
    assert (
        read["next_step"]
        == "No running job of this run was found on the ml.g6e.8xlarge capacity. " + SUBMIT_NOW
    )
    assert len(native.calls) == creates + 1 and tuned["job_name"] not in native.described
    assert lookups == [tuned["request_ref"]] * 2


# --- what the read could not establish is named, never stated as free ------------------------------------------------

TRAIN_LIST = ("ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge")


def _refused_by_outside_4xlarge(fixture_state):
    native = CountingNative(outside=("ml.g6e.4xlarge",))
    store, spec = _real_adapter(fixture_state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == "submission_rejected" and first["blocking_instance_type"] == "ml.g6e.4xlarge"
    return native, store, first


def test_read_names_a_job_of_this_run_that_waits_without_a_type(state):  # noqa: F811
    """A list job of this run without a selected type holds nothing yet and may take a listed type next."""
    native, store, first = _refused_by_outside_4xlarge(state)
    _SftHolder(store, "owned-sft-train", TRAIN_LIST, None)
    read = store.observe(first["request_ref"])
    _refusal(read, first)
    assert read["holding_job_name"] is None and read["waiting_job_names"] == ["owned-sft-train"]
    assert read["type_holders"] == {name: None for name in GENERATION}
    assert read["next_step"] == (
        "No running job of this run was found on the ml.g6e.4xlarge capacity. "
        "Job owned-sft-train of this run waits for a machine and has no instance type yet. " + SUBMIT_NOW
    )
    assert len(native.calls) == 1


def test_read_names_a_job_of_this_run_that_could_not_be_read(state):  # noqa: F811
    """An UNKNOWN observation of a job of this run is not counted as a holder and is not reported as free."""
    native, store, first = _refused_by_outside_4xlarge(state)
    holder = _SftHolder(store, "owned-sft-train", TRAIN_LIST, "ml.g6e.4xlarge", native_state="UNKNOWN")
    read = store.observe(first["request_ref"])
    _refusal(read, first)
    assert read["holding_job_name"] is None and read["waiting_job_names"] == []
    assert read["type_holders"] == {name: None for name in GENERATION}
    assert read["next_step"] == (
        "No running job of this run was found on the ml.g6e.4xlarge capacity. "
        "Job owned-sft-train of this run could not be read. " + SUBMIT_NOW
    )
    assert holder.observed == ["owned-sft-train"] and len(native.calls) == 1
    # The submit path keeps its two results: the unread job is not a holder and not a waiting job.
    assert store._active_holders(GENERATION, first["request_ref"]) == (
        {name: None for name in GENERATION},
        [],
    )


def test_a_failed_holder_lookup_still_answers_the_refusal(state, monkeypatch):  # noqa: F811
    native, store, first = _refused_by_outside_4xlarge(state)

    def failing(instance_types, reference):
        raise RuntimeError("neutral lookup failure")

    monkeypatch.setattr(store, "_observed_holders", failing)
    read = store.observe(first["request_ref"])
    _refusal(read, first)
    assert read["blocking_instance_type"] == "ml.g6e.4xlarge" and read["holding_job_name"] is None
    assert read["type_holders"] == {name: None for name in GENERATION} and read["waiting_job_names"] == []
    assert read["next_step"] == (
        "No running job of this run was found on the ml.g6e.4xlarge capacity. "
        "The jobs of this run could not be read. " + SUBMIT_NOW
    )
    assert len(native.calls) == 1


# --- other outcomes keep their read --------------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["ValidationException", "AccessDeniedException"])
def test_read_of_another_rejection_carries_no_capacity_answer(state, code):  # noqa: F811
    native = CountingNative()
    store, spec = _real_adapter(state, native)
    native.create_training_job = lambda **kw: (_ for _ in ()).throw(_rejection(code))
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == "submission_rejected" and "next_step" not in first
    read = store.observe(first["request_ref"])
    assert read["accepted"] is False and read["generated"] is False and read["success"] is False
    assert not ({"rejection_reason"} | NOTE_FIELDS) & set(read)


def test_read_of_an_accepted_generation_is_unchanged(state):  # noqa: F811
    native = CountingNative()
    store, spec = _real_adapter(state, native)
    base = store.submit(MODEL_REF, CASE_REF, spec)
    read = store.observe(base["request_ref"])
    assert read["status"] == "RUNNING" and read["accepted"] is True
    assert not ({"rejection_reason"} | NOTE_FIELDS) & set(read)


# --- paired evaluation ----------------------------------------------------------------------------------------------


class CountingPairAdapter(SequencePairAdapter):
    """SequencePairAdapter that counts the native reads a store read asks for."""

    def __init__(self, outcomes):
        super().__init__(outcomes)
        self.reads = 0

    def status(self, handle):
        self.reads += 1
        return super().status(handle)

    def artifacts(self, handle):
        self.reads += 1
        return super().artifacts(handle)


def _get_pair(store, reference):
    from amlas.backends.adapters._util import raw_tool
    from amlas.tools.coding_pair import get_coding_pair

    context = SimpleNamespace(
        invocation_state={
            "_amlas_generation_store": store._generation,
            "_amlas_pair_store": store,
            "_amlas_approval_policy": _policy(),
        }
    )
    return raw_tool(get_coding_pair)(reference, tool_context=context)


def test_read_of_a_capacity_refused_pair_answers_the_refusal_and_creates_nothing(state, monkeypatch):  # noqa: F811
    adapter = CountingPairAdapter([("rejected", "ResourceLimitExceeded"), ("accepted", None)])
    store, args = _pair_store(state[0], monkeypatch, adapter)
    first = store.submit(*args)
    assert first["status"] == "submission_rejected" and len(adapter.calls) == 1
    handle = store._records[first["request_ref"]]
    extra = dict(handle.extra)
    for read in (store.observe(first["request_ref"]), _get_pair(store, first["request_ref"])):
        assert read["status"] == "submission_rejected" and read["rejection_reason"] == "capacity_limit"
        assert read["success"] is False and read["accepted"] is False and read["evaluated"] is False
        assert read["quality_success"] is False and read["native_admission"] is False
        assert read["reused"] is False and read["next_step"] == PAIR_NEXT_STEP
        for name in ("request_ref", "job_name", "job_arn"):
            assert read[name] == first[name]
        assert "error_type" not in read
    # No create, no native read under the refused name, and the record is the one the refusal wrote.
    assert len(adapter.calls) == 1 and adapter.reads == 0
    assert store._records[first["request_ref"]] is handle and handle.extra == extra
    # The submit-again path still creates exactly once and then reuses.
    again = store.submit(*args)
    assert again["status"] == "submitted" and again["request_ref"] == first["request_ref"]
    assert len(adapter.calls) == 2
    assert store.submit(*args)["reused"] is True and len(adapter.calls) == 2
    # The accepted original is read natively again (this stand-in's read fails), with no refusal fields.
    observed = store.observe(again["request_ref"])
    assert observed["status"] == "observation_failed" and adapter.reads == 1
    assert not {"rejection_reason", "next_step"} & set(observed)


class _SubmitAfterFirstRelease:
    """Store lock that runs one submit right after its first release: a submit between two steps of a read."""

    def __init__(self, submit):
        import threading

        self._inner, self._submit, self.fired = threading.RLock(), submit, False

    def __enter__(self):
        self._inner.acquire()
        return self

    def __exit__(self, *exc):
        self._inner.release()
        if not self.fired:
            self.fired = True
            self._submit()
        return False


def test_a_pair_submit_right_after_the_read_took_the_record_never_mixes_two_records(state, monkeypatch):  # noqa: F811
    adapter = CountingPairAdapter([("rejected", "ResourceLimitExceeded"), ("accepted", None)])
    store, args = _pair_store(state[0], monkeypatch, adapter)
    first = store.submit(*args)
    assert first["status"] == "submission_rejected"
    accepted = []
    store._lock = _SubmitAfterFirstRelease(lambda: accepted.append(store.submit(*args)))
    read = store.observe(first["request_ref"])
    # The submit ran after the read's lock was released, and the read answers one record: the refusal it read.
    assert store._lock.fired and accepted[0]["status"] == "submitted" and len(adapter.calls) == 2
    assert read["status"] == "submission_rejected" and read["accepted"] is False
    assert read["rejection_reason"] == "capacity_limit" and read["next_step"] == PAIR_NEXT_STEP
    # The next read observes the accepted job and carries no refusal field.
    later = store.observe(first["request_ref"])
    assert later["accepted"] is True and later["status"] == "observation_failed" and adapter.reads == 1
    assert not {"rejection_reason", "next_step"} & set(later)


@pytest.mark.parametrize(
    "outcome",
    [
        ("rejected", "ValidationException"),
        ("rejected", "AccessDeniedException"),
        ("rejected", None),
        ("ambiguous", None),
    ],
)
def test_read_of_another_pair_outcome_carries_no_capacity_answer(state, monkeypatch, outcome):  # noqa: F811
    adapter = CountingPairAdapter([outcome])
    store, args = _pair_store(state[0], monkeypatch, adapter)
    first = store.submit(*args)
    read = store.observe(first["request_ref"])
    assert read["status"] == "observation_failed" and adapter.reads == 1 and len(adapter.calls) == 1
    assert not {"rejection_reason", "next_step"} & set(read)
