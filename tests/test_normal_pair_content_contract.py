"""Fresh producer-byte compatibility only; no execution or authority evidence.

The same managed CASE and GEN publishers used by the normal fixture produce
all bytes. Actual managed/candidate/whole-pair validators remain active. This
unit invokes no Agent, native adapter, candidate, or oracle code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
from normal_sft_cold_fixture import official_case_data
from test_managed_coding_case_envelope import HEADER, source
from test_managed_generation_product import spec_text
from test_qwen_coding_generation_managed import binding
from test_qwen_sft_backend import S3, catalog, encoded
from test_runtime_generation_originals import publish_generation

from amlas.arbiter.coding import coding_spec_bytes
from amlas.backends.adapters.coding_generation import SageMakerCodingGenerationBackend
from amlas.backends.adapters.coding_pair import BUDGETS
from amlas.backends.types import DatasetArtifactRef, JobHandle
from amlas.control_plane.coding_case_state import VerifiedCaseEvaluationSource
from amlas.control_plane.generation_state import VerifiedCodingGenerationSource, fixed_generation_request
from amlas.control_plane.paired_state import PairedCodingInputs
from amlas.data_eng.coding_case_managed import freeze_managed_case_bundle, validate_managed_case_bundle
from amlas.evaluation.coding_pair import validate_pair_artifacts
from amlas.evaluation.coding_pair_runner import paired_runner_sha256
from amlas.spec.schema import SpecModel
from amlas.training.coding_generation import canonical_bytes

pytest_plugins = ("test_runtime_prompt_entry",)
pytestmark = pytest.mark.usefixtures("inert_runtime_setup")

# Exact Source literals are matched only inside the AWS test process. No raw
# message, payload, identifier, traceback, or runtime state enters the artifact.
_CONTRACT_LABELS = {
    "bounded fixed paired records required": "RECORD_BOUND",
    "fixed paired record population differs": "RECORD_POPULATION",
    "typed original paired artifact inputs required": "INPUT_TYPE",
    "closed original paired artifact inventory differs": "ARTIFACT_INVENTORY",
    "original private paired bytes/hash differs": "ARTIFACT_BYTES_HASH",
    "canonical original resolved spec differs": "CANONICAL_SPEC",
    "resolved original private case manifest differs": "CASE_MANIFEST_BINDING",
    "fixed original case order differs": "CASE_ORDER",
    "original generation arm/session/evaluation differs": "GENERATION_IDENTITY",
    "closed original generation commit differs": "GENERATION_COMMIT",
    "original committed generation bytes differ": "GENERATION_COMMITTED_BYTES",
    "original paired generation binding differs": "GENERATION_BINDING",
    "original complete generation outcomes required": "GENERATION_POPULATION",
    "original candidate/manifest case differs": "CANDIDATE_MANIFEST",
    "distinct original base/tuned model bindings required": "DISTINCT_MODELS",
    "fixed official prompt/oracle fields required": "OFFICIAL_FIELDS",
    "fixed official case strings required": "OFFICIAL_STRINGS",
    "fixed official case ID differs": "OFFICIAL_ID",
    "fixed official case bytes differ": "OFFICIAL_PROMPT_ORACLE_BYTES",
    "closed bounded managed case files required": "MANAGED_INVENTORY",
    "managed case producer/source envelope differs": "MANAGED_ENVELOPE",
    "managed case original projection differs": "MANAGED_PROJECTION",
}
_ERROR_CLASSES = {
    ValueError: "ValueError",
    TypeError: "TypeError",
    AssertionError: "AssertionError",
    KeyError: "KeyError",
    AttributeError: "AttributeError",
    RuntimeError: "RuntimeError",
}


def test_actual_normal_case_and_generation_bytes_satisfy_whole_pair_contract(tmp_path):
    phase = "CASE_PUBLICATION"
    candidate_checks = 0
    artifact = tmp_path / "fresh-pair-content-contract.json"

    def emit(status, contract, category="None"):
        artifact.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": phase,
                    "status": status,
                    "contract": contract,
                    "error_class": category,
                    "case_count": 164,
                    "arm_count": 2,
                    "candidate_checks": candidate_checks,
                    "sdk_invocations": 0,
                    "native_calls": 0,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    try:
        original_source = source()
        source_raw, records = official_case_data()
        assert len(source_raw) == original_source.inspection_source.artifact_ref.byte_count
        assert hashlib.sha256(source_raw).hexdigest() == original_source.inspection_source.artifact_ref.sha256
        bundle = freeze_managed_case_bundle(records, source=original_source, header=HEADER)
        phase = "MANAGED_CASE_VALIDATION"
        validate_managed_case_bundle(bundle, source=original_source, header=HEADER)
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
        phase = "CANONICAL_SPEC"
        declaration = json.loads(spec_text().split("---")[1])
        declaration["generation"]["evaluation"] = {
            "case_manifest_ref": dataclasses.asdict(dict(case_refs)["CASE_MANIFEST.json"])
        }
        spec_raw = coding_spec_bytes(SpecModel.model_validate(declaration))
        restored_spec = SpecModel.model_validate_json(spec_raw)
        assert coding_spec_bytes(restored_spec) == spec_raw
        assert restored_spec.generation is not None
        assert restored_spec.generation.evaluation.case_manifest_ref is not None
        assert restored_spec.generation.evaluation.case_manifest_ref.model_dump() == dataclasses.asdict(
            dict(case_refs)["CASE_MANIFEST.json"]
        )
        spec_hash = hashlib.sha256(spec_raw).hexdigest()
        prompts = tuple(json.loads(line) for line in bundle["PROMPTS.jsonl"].splitlines())
        assert len(prompts) == 164
        objects = S3()
        generations = []
        payloads = {"cases/" + name: raw for name, raw in bundle.items()}
        for arm in ("base", "tuned"):
            phase = "GENERATION_PUBLICATION"
            value = binding(arm)
            value["source_commit"] = selected.source_commit
            value["spec_sha256"] = spec_hash
            value["prompts_ref"] = dataclasses.asdict(dict(case_refs)["PROMPTS.jsonl"])
            if arm == "tuned":
                checkpoint = b'{"schema_version":1}\n'
                value["checkpoint_ref"] = dataclasses.asdict(
                    ref(root + "sft/owned-sft/output/CHECKPOINT.json", checkpoint)
                )
                value["checkpoint_source_commit"] = selected.source_commit
            job = selected.deployment_prefix + "-gen-content-" + arm
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
                assert len(raw) == published_ref.byte_count
                assert hashlib.sha256(raw).hexdigest() == published_ref.sha256
                refs.append((name, published_ref))
                payloads[arm + "/" + name] = raw
            generations.append(
                VerifiedCodingGenerationSource(session, handle.arn, value["evaluation_id"], arm, tuple(refs))
            )
            phase = "CANDIDATE_VALIDATION"
            candidates = payloads[arm + "/CANDIDATES.jsonl"].splitlines()
            assert len(candidates) == 164
            for raw, prompt in zip(candidates, prompts, strict=True):
                SageMakerCodingGenerationBackend._validate_candidate(json.loads(raw), prompt)
                candidate_checks += 1
        assert candidate_checks == 328 and len(payloads) == 11
        inputs = PairedCodingInputs(cases, *generations, spec_raw, spec_hash)
        runner_hash = paired_runner_sha256(oracle_launcher_sha256="d" * 64)
        phase = "WHOLE_PAIR_VALIDATION"
        try:
            result = validate_pair_artifacts(
                inputs,
                payloads,
                runner_sha256=runner_hash,
                runtime_sha256=selected.admission_digest,
                budgets_sha256=hashlib.sha256(canonical_bytes(BUDGETS)).hexdigest(),
            )
        except ValueError as error:
            label = next(
                (label for literal, label in _CONTRACT_LABELS.items() if error.args == (literal,)),
                "UNCLASSIFIED",
            )
            emit("refused", label, "ValueError")
            pytest.fail("normal_pair_content_contract_" + label, pytrace=False)
        assert len(result[1]) == 164
        assert set(result[2]) == {"base", "tuned"}
        emit("accepted", "COMPLETE")
    except Exception as error:
        category = _ERROR_CLASSES.get(type(error), "Exception")
        emit("prerequisite_failed", "PREREQUISITE", category)
        pytest.fail("normal_pair_content_prerequisite_" + phase + "_" + category, pytrace=False)
