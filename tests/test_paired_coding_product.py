"""Fixed paired source plumbing; neutral fakes never execute candidate/oracle code."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_managed_generation_product import CASE_REF, MODEL_REF, spec_text
from test_managed_generation_product import state as generation_state  # noqa: F401

from amlas.backends.types import DatasetArtifactRef, JobHandle
from amlas.control_plane.paired_state import PairedCodingCatalog, SessionPairedCodingStore
from amlas.evaluation.coding_oracle import decode, encode


def test_resolve_private_case_manifest_mints_new_spec_before_arms_and_keeps_old_bytes(generation_state):  # noqa: F811
    store, adapter, _reads, *_ = generation_state
    original = store.bind_spec(spec_text())
    old_bytes = store._specs[original["spec_ref"]][1]
    resolve = getattr(store, "resolve_case_spec", None)
    assert callable(resolve), "private case/spec binding missing"
    result = resolve(CASE_REF, original["spec_ref"])
    assert result["spec_ref"] != original["spec_ref"]
    assert store._specs[original["spec_ref"]][1] == old_bytes
    spec = json.loads(store._specs[result["spec_ref"]][1])
    case = store._cases._verified_evaluation_source(CASE_REF)
    assert spec["generation"]["evaluation"]["case_manifest_ref"] == dataclasses.asdict(
        dict(case.file_refs)["CASE_MANIFEST.json"]
    )
    assert hashlib.sha256(store._specs[result["spec_ref"]][1]).hexdigest() == result["spec_sha256"]
    assert result["quality_success"] is False and adapter.calls == []
    submitted = store.submit(MODEL_REF, CASE_REF, result["spec_ref"])
    assert submitted["evaluation_id"] and len(adapter.calls) == 1


def test_unresolved_spec_cannot_dispatch_generation(generation_state):  # noqa: F811
    store, adapter, reads, *_ = generation_state
    original = store.bind_spec(spec_text())
    with pytest.raises(ValueError, match="resolved"):
        store.submit(MODEL_REF, CASE_REF, original["spec_ref"])
    assert adapter.calls == [] and reads == []


@pytest.mark.parametrize(
    "value", [None, True, 33, 1.25, "neutral", [1, False], (1, "x"), {(1, 2): [True, None], 3: {"key": 4}}]
)
def test_tagged_codec_preserves_tuple_and_nonstring_dictionary_keys(value):
    assert decode(encode(value)) == value
    assert type(decode(encode(value))) is type(value)


@pytest.mark.parametrize(
    "value",
    [
        {"t": "int", "v": True},
        {"t": "tuple", "v": {}},
        {"t": "str", "v": 3},
        {
            "t": "dict",
            "v": [[{"t": "int", "v": 1}, {"t": "int", "v": 2}], [{"t": "int", "v": 1}, {"t": "int", "v": 3}]],
        },
    ],
)
def test_tagged_codec_refuses_shape_type_and_duplicate_keys(value):
    with pytest.raises(ValueError):
        decode(value)


def test_resolved_spec_tool_is_opaque_policy_gated_and_appears_in_normal_registry(generation_state):  # noqa: F811
    from amlas.backends.adapters._util import raw_tool
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
    from amlas.tools.coding_pair import resolve_coding_evaluation_spec
    from amlas.tui.chat.agent_tools import STATE_CHANGING_TOOLS, get_safe_llm_tools, tool_display_name

    store, adapter, reads, *_ = generation_state
    original = store.bind_spec(spec_text())
    context = SimpleNamespace(
        invocation_state={"_amlas_generation_store": store, "_amlas_approval_policy": ApprovalPolicy()}
    )
    result = raw_tool(resolve_coding_evaluation_spec)(CASE_REF, original["spec_ref"], tool_context=context)
    assert result["status"] == "denied" and reads == [] and adapter.calls == []
    context.invocation_state["_amlas_approval_policy"] = ApprovalPolicy(
        mode=ApprovalMode.ALLOWLIST, allowlist=["coding_pair_flow"], max_auto_cost=None
    )
    result = raw_tool(resolve_coding_evaluation_spec)(CASE_REF, original["spec_ref"], tool_context=context)
    assert result["case_bound"] is True and result["spec_ref"] != original["spec_ref"]
    assert "resolve_coding_evaluation_spec" in STATE_CHANGING_TOOLS
    assert "resolve_coding_evaluation_spec" in [tool_display_name(tool) for tool in get_safe_llm_tools()]
    assert not set(resolve_coding_evaluation_spec.tool_spec["inputSchema"]["json"]["properties"]) & {
        "tool_context",
        "policy",
        "case_manifest",
        "image_uri",
        "role",
    }


class PairAdapter:
    backend_name = "paired_coding"

    def __init__(self):
        self.calls = []

    def submit(self, **kwargs):
        self.calls.append(kwargs)
        return JobHandle(
            "owned-pair",
            self.backend_name,
            "arn:aws:sagemaker:us-west-2:111122223333:processing-job/owned-pair",
            {"submission_state": "ambiguous", "runtime_session_id": kwargs["runtime_session_id"]},
        )

    def status(self, handle):
        raise TimeoutError("neutral observation failure")

    def artifacts(self, handle):
        raise TimeoutError("neutral observation failure")


def test_pair_store_keeps_one_unknown_original_and_refuses_wrong_arm(generation_state, monkeypatch):  # noqa: F811
    generation, _, _, *_ = generation_state
    original = generation.bind_spec(spec_text())
    resolve = getattr(generation, "resolve_case_spec", None)
    assert callable(resolve), "resolved case spec prerequisite missing"
    resolved = resolve(CASE_REF, original["spec_ref"])
    from amlas.control_plane.generation_state import VerifiedCodingGenerationSource

    case = generation._cases._verified_evaluation_source(CASE_REF)
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
    adapter = PairAdapter()
    store = SessionPairedCodingStore(
        generation, generation._cases, adapter, PairedCodingCatalog(generation._catalog, "d" * 64)
    )
    result = store.submit(CASE_REF, "gen_" + "a" * 32, "gen_" + "b" * 32, resolved["spec_ref"])
    repeated = store.submit(CASE_REF, "gen_" + "a" * 32, "gen_" + "b" * 32, resolved["spec_ref"])
    assert result["request_ref"] == repeated["request_ref"] and len(adapter.calls) == 1
    assert result["status"] == "acceptance_unknown" and result["quality_success"] is False
    assert adapter.calls[0]["inputs"].cases == case
    assert store.observe(result["request_ref"])["status"] == "observation_failed"
    with pytest.raises(ValueError):
        store.submit(CASE_REF, "gen_" + "b" * 32, "gen_" + "a" * 32, resolved["spec_ref"])
    assert len(adapter.calls) == 1


def test_recorder_accepts_only_exact_native_builtin_without_widening_session_or_names(tmp_path):
    from test_coding_execution_records import record
    from test_coding_executor import BINDING, REQUEST, Fake

    from amlas.evaluation.coding_executor import FileInvocationRecorder, invoke_recorded

    fake = Fake()
    request = dict(REQUEST, codeInterpreterIdentifier="aws.codeinterpreter.v1")
    events = invoke_recorded(
        fake, request, binding=BINDING, persist=fake.persist, deadline=10.0, clock=lambda: 1.0
    )
    assert len(events) == 1 and fake.calls[0]["codeInterpreterIdentifier"] == "aws.codeinterpreter.v1"
    folder = tmp_path / "sink"
    folder.mkdir()
    with FileInvocationRecorder(folder) as sink:
        sink(dict(record(), interpreter_id="aws.codeinterpreter.v1"))
    assert len(list(folder.iterdir())) == 1
    for field, value in (
        ("codeInterpreterIdentifier", "aws.other.v1"),
        ("sessionId", "aws.codeinterpreter.v1"),
    ):
        with pytest.raises(ValueError):
            invoke_recorded(
                fake,
                dict(request, **{field: value}),
                binding=BINDING,
                persist=fake.persist,
                deadline=10.0,
                clock=lambda: 1.0,
            )
