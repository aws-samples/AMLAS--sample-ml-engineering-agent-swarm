"""Spec, generation and pair references are functions of the request, and a repeat meets the same native job.

A replacement worker that repeats a request after a worker stop must derive the reference it handed out before the
stop, and its native Create must meet the job it already made instead of making a second one. Neutral stand-ins; no AWS.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
from botocore.exceptions import ClientError
from test_managed_generation_product import CASE_REF, MODEL_REF, Adapter, resolved_spec, spec_text
from test_paired_coding_backend import SESSION, submit
from test_qwen_sft_backend import Native

from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend
from amlas.control_plane.generation_state import SessionCodingGenerationStore
from amlas.control_plane.paired_state import pair_operation_digest

pytest_plugins = ("test_managed_generation_product", "test_paired_coding_backend")
IN_USE = ClientError({"Error": {"Code": "ResourceInUse", "Message": "name in use"}}, "CreateJob")


def fresh_store(state, backend=None):
    store, *_ = state
    return SessionCodingGenerationStore(
        store._models, store._cases, store._sft, backend or Adapter(), store._catalog
    )


def real_store(state, native):
    _store, _adapter, _reads, s3, _inputs, case = state
    s3.install(dict(case.file_refs)["PROMPTS.jsonl"], b"{}\n")
    backend = SageMakerCodingGenerationBackend(sagemaker_client=native, s3_client=s3)
    return fresh_store(state, backend)


def test_spec_and_generation_references_are_the_same_in_a_store_with_empty_memory(state):
    store, *_ = state
    other = fresh_store(state)
    assert store.bind_spec(spec_text())["spec_ref"] == other.bind_spec(spec_text())["spec_ref"]
    first, second = resolved_spec(store), resolved_spec(other)
    assert first == second and re.fullmatch("spec_[a-f0-9]{32}", first)
    ref = store.submit(MODEL_REF, CASE_REF, first)["request_ref"]
    assert ref == other.submit(MODEL_REF, CASE_REF, second)["request_ref"]
    assert re.fullmatch("gen_[a-f0-9]{32}", ref)


def test_a_different_request_gets_a_different_reference(state):
    store, *_ = state
    base = store.bind_spec(spec_text())["spec_ref"]
    changed = store.bind_spec(spec_text(project_id="another_generation"))["spec_ref"]
    assert base != changed
    resolved, other = (store.resolve_case_spec(CASE_REF, ref)["spec_ref"] for ref in (base, changed))
    assert resolved != other
    assert (
        store.submit(MODEL_REF, CASE_REF, resolved)["request_ref"]
        != store.submit(MODEL_REF, CASE_REF, other)["request_ref"]
    )
    assert base != resolved, "the resolved specification keeps its own identity"


def test_resubmission_in_a_store_with_empty_memory_meets_the_same_native_job(state):
    native = Native()
    first_store = real_store(state, native)
    first = first_store.submit(MODEL_REF, CASE_REF, resolved_spec(first_store))
    assert first["accepted"] is True and len(native.calls) == 1
    assert first["job_name"].endswith(first["request_ref"][4:20]), "the native name follows the reference"
    native.error = IN_USE
    second_store = real_store(state, native)
    second = second_store.submit(MODEL_REF, CASE_REF, resolved_spec(second_store))
    assert second["request_ref"] == first["request_ref"] and second["job_name"] == first["job_name"]
    assert second["accepted"] is True and second["status"] == "submitted"
    assert [call["TrainingJobName"] for call in native.calls] == [first["job_name"]] * 2


def test_a_name_in_use_by_another_job_is_not_adopted(state):
    native = Native()
    first_store = real_store(state, native)
    first = first_store.submit(MODEL_REF, CASE_REF, resolved_spec(first_store))
    native.error = IN_USE
    original = native.describe_training_job
    native.describe_training_job = lambda **kw: {
        **original(**kw),
        "TrainingJobArn": "arn:aws:sagemaker:other",
    }
    second_store = real_store(state, native)
    second = second_store.submit(MODEL_REF, CASE_REF, resolved_spec(second_store))
    assert second["accepted"] is False and second["job_name"] == first["job_name"]


def test_pair_digest_follows_the_inputs_and_the_pair_catalog(packet):
    _backend, inputs, catalog, *_ = packet
    from amlas.control_plane.paired_state import PairedCodingCatalog

    pair_catalog = PairedCodingCatalog(catalog, "d" * 64)
    digest = pair_operation_digest(SESSION, pair_catalog, inputs)
    assert digest == pair_operation_digest(SESSION, pair_catalog, dataclasses.replace(inputs))
    assert re.fullmatch("[a-f0-9]{64}", digest)
    other_spec = dataclasses.replace(inputs, spec_sha256="e" * 64)
    assert pair_operation_digest(SESSION, pair_catalog, other_spec) != digest
    assert pair_operation_digest(SESSION, PairedCodingCatalog(catalog, "e" * 64), inputs) != digest
    assert pair_operation_digest("another-session", pair_catalog, inputs) != digest


def test_pair_resubmission_meets_the_same_native_job(packet):
    first = submit(packet)
    _backend, _inputs, _catalog, sm, _s3, _validated = packet
    assert first.extra["submission_state"] == "accepted"
    sm.error = IN_USE
    second = submit(packet)
    assert second.id == first.id and second.arn == first.arn
    assert second.extra["submission_state"] == "accepted" and second.extra["adopted_existing"] is True
    assert first.id.startswith(first.extra["catalog"].deployment_prefix + "-pair-")


def planning_history(bound, resolved, *, sha=None, resolved_ref=None):
    """The earlier bind and resolve calls and their results, as a restored conversation holds them."""
    import json

    return [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "bind-1",
                        "name": "bind_coding_generation_spec",
                        "input": {"yaml_text": spec_text()},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "bind-1",
                        "status": "success",
                        "content": [
                            {
                                "text": json.dumps(
                                    {"success": True, **bound, "spec_sha256": sha or bound["spec_sha256"]}
                                )
                            }
                        ],
                    }
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "resolve-1",
                        "name": "resolve_coding_evaluation_spec",
                        "input": {"case_request_ref": CASE_REF, "spec_ref": bound["spec_ref"]},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "resolve-1",
                        "status": "success",
                        "content": [
                            {
                                "text": json.dumps(
                                    {
                                        "success": False,
                                        **resolved,
                                        "spec_ref": resolved_ref or resolved["spec_ref"],
                                        "executed": False,
                                    }
                                )
                            }
                        ],
                    }
                }
            ],
        },
    ]


def replacement_store(state, messages):
    """A store with empty memory whose backend can read the restored conversation."""
    from types import SimpleNamespace

    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    backend = object.__new__(ToolBindingBackend)
    backend._warm_owner = SimpleNamespace(_normal_agent=SimpleNamespace(messages=messages))
    store = fresh_store(state)
    calls = []

    def restore(target):
        calls.append(1)
        backend._restore_planning_refs(target)

    store._planning_restorer = restore
    return store, calls


def handed_out(state):
    store, *_ = state
    bound = store.bind_spec(spec_text())
    resolved = store.resolve_case_spec(CASE_REF, bound["spec_ref"])
    return bound, resolved, store.submit(MODEL_REF, CASE_REF, resolved["spec_ref"])["request_ref"]


def test_a_fresh_store_refuses_a_specification_reference_it_never_derived(state):
    _bound, resolved, _ref = handed_out(state)
    with_nothing = fresh_store(state)
    with pytest.raises(ValueError, match="unknown same-session spec reference"):
        with_nothing.submit(MODEL_REF, CASE_REF, resolved["spec_ref"])
    with pytest.raises(ValueError, match="unknown original specification reference"):
        with_nothing.resolve_case_spec(CASE_REF, "spec_" + "9" * 32)


def test_a_replacement_store_restores_the_references_from_the_admitted_history(state):
    bound, resolved, gen_ref = handed_out(state)
    store, calls = replacement_store(state, planning_history(bound, resolved))
    result = store.submit(MODEL_REF, CASE_REF, resolved["spec_ref"])
    assert result["request_ref"] == gen_ref, "the generation reference follows the restored specification"
    assert len(calls) == 1, "the restorer runs once per miss, not per lookup"
    store.submit(MODEL_REF, CASE_REF, resolved["spec_ref"])
    assert len(calls) == 1, "a known reference never calls the restorer"


def test_resolving_and_pairing_lookups_restore_the_same_references(state):
    bound, resolved, _ref = handed_out(state)
    store, calls = replacement_store(state, planning_history(bound, resolved))
    again = store.resolve_case_spec(CASE_REF, bound["spec_ref"])
    assert again["spec_ref"] == resolved["spec_ref"] and len(calls) == 1
    other, _calls = replacement_store(state, planning_history(bound, resolved))
    spec_bytes, _sha, _cases = other._verified_resolved_spec(CASE_REF, resolved["spec_ref"])
    assert spec_bytes


def test_a_history_that_does_not_verify_restores_nothing(state):
    bound, resolved, _ref = handed_out(state)
    altered_digest, _calls = replacement_store(state, planning_history(bound, resolved, sha="e" * 64))
    with pytest.raises(ValueError, match="unknown same-session spec reference"):
        altered_digest.submit(MODEL_REF, CASE_REF, resolved["spec_ref"])
    other_ref = "spec_" + "d" * 32
    altered_ref, _calls = replacement_store(state, planning_history(bound, resolved, resolved_ref=other_ref))
    with pytest.raises(ValueError, match="unknown same-session spec reference"):
        altered_ref.submit(MODEL_REF, CASE_REF, other_ref)
    empty, _calls = replacement_store(state, [])
    with pytest.raises(ValueError, match="unknown same-session spec reference"):
        empty.submit(MODEL_REF, CASE_REF, resolved["spec_ref"])
