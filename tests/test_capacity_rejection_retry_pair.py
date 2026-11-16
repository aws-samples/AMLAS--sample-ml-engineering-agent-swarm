"""The paired evaluation submit treats a capacity rejection like the generation submit.

A definite ResourceLimitExceeded rejection made no native job. The paired adapter records the native code of a
definite rejection on the original handle, and the paired store lets a later identical submit reach the create again
only for that code. ValidationException and AccessDeniedException keep today's cached answer, and an ambiguous create
still never repeats.
"""

from __future__ import annotations

import dataclasses

import pytest
from test_managed_generation_product import CASE_REF, spec_text
from test_managed_generation_product import state as generation_state  # noqa: F401  (pytest fixture)
from test_paired_coding_backend import packet  # noqa: F401  (pytest fixture)
from test_paired_coding_backend import submit as pair_submit

from amlas.backends.types import DatasetArtifactRef, JobHandle
from amlas.control_plane.paired_state import PairedCodingCatalog, SessionPairedCodingStore


def _code_error(code):
    error = RuntimeError("neutral definite rejection")
    error.response = {"Error": {"Code": code}}
    return error


@pytest.mark.parametrize("code", ["ValidationException", "AccessDeniedException", "ResourceLimitExceeded"])
def test_paired_backend_records_the_native_code_of_a_definite_rejection(packet, code):  # noqa: F811
    from amlas.backends.types import is_capacity_rejection

    packet[3].error = _code_error(code)
    handle = pair_submit(packet)
    assert handle.extra["submission_state"] == "rejected" and handle.extra["rejection_code"] == code
    assert is_capacity_rejection(handle) is (code == "ResourceLimitExceeded")
    assert packet[3].calls.count("create") == 1


def test_paired_backend_creates_the_same_job_again_after_a_capacity_rejection(packet):  # noqa: F811
    sm = packet[3]
    sm.error = _code_error("ResourceLimitExceeded")
    first = pair_submit(packet)
    sm.error = None
    second = pair_submit(packet)
    assert first.extra["submission_state"] == "rejected"
    assert second.extra["submission_state"] == "accepted" and "rejection_code" not in second.extra
    assert (first.id, first.arn) == (second.id, second.arn) and sm.calls.count("create") == 2


class SequencePairAdapter:
    """Neutral paired port: answers each submit with the next scripted outcome."""

    backend_name = "paired_coding"

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def submit(self, **kwargs):
        self.calls.append(kwargs)
        state, code = self.outcomes.pop(0)
        extra = {"submission_state": state, "runtime_session_id": kwargs["runtime_session_id"]}
        if code is not None:
            extra["rejection_code"] = code
        return JobHandle(
            "owned-pair",
            self.backend_name,
            "arn:aws:sagemaker:us-west-2:111122223333:processing-job/owned-pair",
            extra,
        )

    def status(self, handle):
        raise TimeoutError("neutral observation failure")

    def artifacts(self, handle):
        raise TimeoutError("neutral observation failure")


def _pair_store(generation, monkeypatch, adapter):
    original = generation.bind_spec(spec_text())
    resolved = generation.resolve_case_spec(CASE_REF, original["spec_ref"])
    from amlas.control_plane.generation_state import VerifiedCodingGenerationSource

    artifacts = tuple(
        (
            name,
            DatasetArtifactRef(
                "s3://owned-test-bucket/sessions/isolated-sft/generation/job/" + name,
                "v-" + name,
                "a" * 64,
                20,
                "jsonl" if name.endswith(".jsonl") else "json",
            ),
        )
        for name in ("CANDIDATES.jsonl", "GENERATION.json", "COMMITTED.json")
    )
    base = VerifiedCodingGenerationSource(
        "isolated-sft",
        "arn:aws:sagemaker:us-west-2:111122223333:training-job/owned-base",
        "paired-id",
        "base",
        artifacts,
    )
    tuned = dataclasses.replace(
        base, producer_job_arn=base.producer_job_arn.replace("base", "tuned"), arm="tuned"
    )
    refs = {"gen_" + "a" * 32: base, "gen_" + "b" * 32: tuned}
    monkeypatch.setattr(generation, "_verified_evaluation_source", lambda ref: refs[ref])
    store = SessionPairedCodingStore(
        generation, generation._cases, adapter, PairedCodingCatalog(generation._catalog, "d" * 64)
    )
    return store, (CASE_REF, "gen_" + "a" * 32, "gen_" + "b" * 32, resolved["spec_ref"])


def test_pair_store_submits_again_after_a_capacity_rejection_and_then_reuses(generation_state, monkeypatch):  # noqa: F811
    generation = generation_state[0]
    adapter = SequencePairAdapter(
        [("rejected", "ResourceLimitExceeded"), ("rejected", "ResourceLimitExceeded"), ("accepted", None)]
    )
    store, args = _pair_store(generation, monkeypatch, adapter)
    first, second, third = store.submit(*args), store.submit(*args), store.submit(*args)
    assert first["status"] == second["status"] == "submission_rejected" and second["reused"] is False
    assert third["status"] == "submitted" and third["accepted"] is True and len(adapter.calls) == 3
    fourth = store.submit(*args)
    assert fourth["reused"] is True and fourth["status"] == "submitted" and len(adapter.calls) == 3
    assert len({value["request_ref"] for value in (first, second, third, fourth)}) == 1


@pytest.mark.parametrize(
    "outcome",
    [
        ("rejected", "ValidationException"),
        ("rejected", "AccessDeniedException"),
        ("rejected", None),
        ("ambiguous", None),
    ],
)
def test_pair_store_keeps_other_outcomes_as_the_cached_answer(generation_state, monkeypatch, outcome):  # noqa: F811
    generation = generation_state[0]
    adapter = SequencePairAdapter([outcome, ("accepted", None)])
    store, args = _pair_store(generation, monkeypatch, adapter)
    first, again = store.submit(*args), store.submit(*args)
    assert again["reused"] is True and again["status"] == first["status"] and len(adapter.calls) == 1
