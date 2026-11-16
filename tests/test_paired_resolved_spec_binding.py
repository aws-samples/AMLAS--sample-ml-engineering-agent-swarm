"""The pair binds the resolved evaluation spec, and the normal path hands it the resolved one.

The first base generation of an earlier real run (a direct driver, not the normal prompt path) was bound to the
unresolved October spec, which the pair refuses. Two things are pinned here. First, the pair's own validator, run on
actual published case and generation bytes, refuses every unresolved binding at its three guards (canonical
resolved bytes, case-manifest reference, generation spec digest). Second, the normal stores mint the resolved spec,
bind generation to its digest and hand the pair those same bytes, and refuse the unresolved reference. No agent,
native adapter, candidate or oracle code runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
from normal_sft_cold_fixture import official_case_data
from test_managed_coding_case_envelope import HEADER, source
from test_managed_generation_product import CASE_REF, MODEL_REF, spec_text
from test_managed_generation_product import state as generation_state  # noqa: F401
from test_paired_coding_product import PairAdapter
from test_qwen_coding_generation_managed import binding
from test_qwen_sft_backend import S3, catalog, encoded
from test_runtime_generation_originals import publish_generation

from amlas.arbiter.coding import coding_spec_bytes
from amlas.backends.adapters.coding_pair import BUDGETS
from amlas.backends.types import DatasetArtifactRef, JobHandle
from amlas.control_plane.coding_case_state import VerifiedCaseEvaluationSource
from amlas.control_plane.generation_state import VerifiedCodingGenerationSource, fixed_generation_request
from amlas.control_plane.paired_state import PairedCodingCatalog, PairedCodingInputs, SessionPairedCodingStore
from amlas.data_eng.coding_case_managed import freeze_managed_case_bundle
from amlas.evaluation.coding_pair import validate_pair_artifacts
from amlas.evaluation.coding_pair_runner import paired_runner_sha256
from amlas.spec.schema import SpecModel
from amlas.training.coding_generation import canonical_bytes

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")


def _material(*, generation_spec, pair_spec):
    """Actual managed case bytes and two published generation arms; each side is bound to the named spec form."""
    original_source = source()
    _raw, records = official_case_data()
    bundle = freeze_managed_case_bundle(records, source=original_source, header=HEADER)
    session = HEADER["runtime_session_id"]
    selected = catalog(bucket=original_source.inspection_source.bucket)
    root = "s3://" + selected.bucket + "/sessions/" + session + "/"

    def ref(uri, raw, fmt="json"):
        return DatasetArtifactRef(
            uri, "declared-original-version", hashlib.sha256(raw).hexdigest(), len(raw), fmt
        )

    case_refs = tuple(
        (
            name,
            ref(
                root + "case-freeze/" + HEADER["job_name"] + "/output/" + name,
                raw,
                "jsonl" if name.endswith(".jsonl") else "json",
            ),
        )
        for name, raw in bundle.items()
    )
    cases = VerifiedCaseEvaluationSource(
        session,
        "arn:aws:sagemaker:"
        + selected.region
        + ":"
        + selected.account_id
        + ":processing-job/"
        + HEADER["job_name"],
        tuple(sorted(HEADER.items())),
        original_source,
        case_refs,
    )
    declaration = json.loads(spec_text().split("---")[1])
    unresolved = coding_spec_bytes(SpecModel.model_validate(declaration))
    declaration["generation"]["evaluation"] = {
        "case_manifest_ref": dataclasses.asdict(dict(case_refs)["CASE_MANIFEST.json"])
    }
    resolved = coding_spec_bytes(SpecModel.model_validate(declaration))
    forms = {"resolved": resolved, "unresolved": unresolved}
    if pair_spec == "noncanonical":
        forms["noncanonical"] = json.dumps(json.loads(resolved), indent=1).encode()
    prompts = tuple(json.loads(line) for line in bundle["PROMPTS.jsonl"].splitlines())
    objects = S3()
    generations = []
    payloads = {"cases/" + name: raw for name, raw in bundle.items()}
    for arm in ("base", "tuned"):
        value = binding(arm)
        value["source_commit"] = selected.source_commit
        value["spec_sha256"] = hashlib.sha256(forms[generation_spec]).hexdigest()
        value["prompts_ref"] = dataclasses.asdict(dict(case_refs)["PROMPTS.jsonl"])
        if arm == "tuned":
            checkpoint = b'{"schema_version":1}\n'
            value["checkpoint_ref"] = dataclasses.asdict(
                ref(root + "sft/owned-sft/output/CHECKPOINT.json", checkpoint)
            )
            value["checkpoint_source_commit"] = selected.source_commit
        job = selected.deployment_prefix + "-gen-spec-" + arm
        run_ref = ref(root + "generation/" + job + "/input/bindings/RUN.json", encoded(value))
        request = fixed_generation_request(selected, session, job, run_ref, arm)
        handle = JobHandle(
            job,
            "coding_generation",
            "arn:aws:sagemaker:" + selected.region + ":" + selected.account_id + ":training-job/" + job,
            {"catalog": selected, "run_binding": value, "run_ref": run_ref, "request": request},
        )
        publish_generation(objects, handle, prompts)
        output = request["OutputDataConfig"]["S3OutputPath"].split("/", 3)[3] + "/model/"
        refs = []
        for name in ("CANDIDATES.jsonl", "GENERATION.json", "COMMITTED.json"):
            published_ref, raw = objects.objects[output + name]
            refs.append((name, published_ref))
            payloads[arm + "/" + name] = raw
        generations.append(
            VerifiedCodingGenerationSource(session, handle.arn, value["evaluation_id"], arm, tuple(refs))
        )
    spec_raw = forms[pair_spec]
    inputs = PairedCodingInputs(cases, *generations, spec_raw, hashlib.sha256(spec_raw).hexdigest())
    return inputs, payloads, selected, forms


def _validate(inputs, payloads, selected):
    return validate_pair_artifacts(
        inputs,
        payloads,
        runner_sha256=paired_runner_sha256(oracle_launcher_sha256="d" * 64),
        runtime_sha256=selected.admission_digest,
        budgets_sha256=hashlib.sha256(canonical_bytes(BUDGETS)).hexdigest(),
    )


def test_resolved_spec_on_both_sides_is_accepted():
    inputs, payloads, selected, forms = _material(generation_spec="resolved", pair_spec="resolved")
    bound, cases, _candidates = _validate(inputs, payloads, selected)
    assert bound.spec_sha256 == hashlib.sha256(forms["resolved"]).hexdigest() and len(cases) == 164
    assert hashlib.sha256(forms["resolved"]).hexdigest() != hashlib.sha256(forms["unresolved"]).hexdigest()


@pytest.mark.parametrize(
    "generation_spec,pair_spec,message",
    [
        # October's runs and that first base generation: outputs bound to the unresolved spec.
        ("unresolved", "resolved", "original paired generation binding differs"),
        ("unresolved", "unresolved", "resolved original private case manifest differs"),
        ("resolved", "unresolved", "resolved original private case manifest differs"),
        ("resolved", "noncanonical", "canonical original resolved spec differs"),
    ],
)
def test_pair_refuses_every_unresolved_spec_binding(generation_spec, pair_spec, message):
    inputs, payloads, selected, _forms = _material(generation_spec=generation_spec, pair_spec=pair_spec)
    with pytest.raises(ValueError) as caught:
        _validate(inputs, payloads, selected)
    assert caught.value.args == (message,)


def test_normal_stores_bind_generation_and_pair_to_the_same_resolved_spec(generation_state, monkeypatch):  # noqa: F811
    store, adapter, _reads, *_ = generation_state
    original = store.bind_spec(spec_text())
    resolved = store.resolve_case_spec(CASE_REF, original["spec_ref"])
    case = store._cases._verified_evaluation_source(CASE_REF)
    resolved_bytes = store._specs[resolved["spec_ref"]][1]
    assert resolved["spec_sha256"] == hashlib.sha256(resolved_bytes).hexdigest() != original["spec_sha256"]
    assert json.loads(resolved_bytes)["generation"]["evaluation"]["case_manifest_ref"] == dataclasses.asdict(
        dict(case.file_refs)["CASE_MANIFEST.json"]
    )
    # Generation: the unresolved reference is refused; the resolved one binds the job to the resolved digest.
    with pytest.raises(ValueError, match="resolved"):
        store.submit(MODEL_REF, CASE_REF, original["spec_ref"])
    store.submit(MODEL_REF, CASE_REF, resolved["spec_ref"])
    assert [call["spec_sha256"] for call in adapter.calls] == [resolved["spec_sha256"]]
    # Pair: the store hands the adapter the resolved bytes and digest the generation used.
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
    monkeypatch.setattr(store, "_verified_evaluation_source", lambda ref: refs[ref])
    pair_adapter = PairAdapter()
    pairs = SessionPairedCodingStore(
        store, store._cases, pair_adapter, PairedCodingCatalog(store._catalog, "d" * 64)
    )
    with pytest.raises(ValueError, match="resolved"):
        pairs.submit(CASE_REF, "gen_" + "a" * 32, "gen_" + "b" * 32, original["spec_ref"])
    assert pair_adapter.calls == []
    from amlas.backends.adapters._util import raw_tool
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
    from amlas.tools.coding_pair import submit_coding_pair

    context = SimpleNamespace(
        invocation_state={
            "_amlas_generation_store": store,
            "_amlas_pair_store": pairs,
            "_amlas_approval_policy": ApprovalPolicy(
                mode=ApprovalMode.ALLOWLIST, allowlist=["coding_pair_flow"], max_auto_cost=None
            ),
        }
    )
    result = raw_tool(submit_coding_pair)(
        CASE_REF, "gen_" + "a" * 32, "gen_" + "b" * 32, resolved["spec_ref"], tool_context=context
    )
    assert result["status"] == "acceptance_unknown", result
    inputs = pair_adapter.calls[0]["inputs"]
    assert inputs.spec_bytes == resolved_bytes and inputs.spec_sha256 == resolved["spec_sha256"]
    assert inputs.spec_sha256 == adapter.calls[0]["spec_sha256"]
