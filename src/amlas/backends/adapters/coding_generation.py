"""Fixed generation byte I/O and native observation; never load models or execute code."""

from __future__ import annotations

import dataclasses
import hashlib
import re
import time

from ...control_plane.generation_state import (
    fixed_generation_request,
    generation_operation_digest,
    validate_cases,
)
from ...control_plane.sft_state import (
    VerifiedSftCheckpointSource,
    validate_catalog,
    validate_model,
    validate_ref,
)
from ...data_eng.inspection_schema import decode_object, validate_native_privilege_proof
from ...data_eng.model_weight_manifest import expected_model_weight_manifest
from ...training.coding_generation import (
    CASE_IDS,
    CONTRACT_SHA256,
    _valid_module,
    canonical_bytes,
    prompts_sha256,
)
from ...training.qwen_sft import validate_generation_binding
from ..types import DatasetArtifactRef, JobHandle, JobState
from .artifact_rejection import ImmutableArtifactRejectionError, rejection_data
from .qwen_sft import (
    SageMakerQwenSftBackend,
    _deadline,
    _hash_seconds,
    _json,
    _pin_key,
    _version,
    observed_selected_instance,
)

# The one instance type a SageMaker ResourceLimitExceeded message names: "The account-level service limit
# 'ml.g6e.4xlarge for training job usage' is 1 Instances, with current utilization of 1 Instances ...".
_LIMIT_TYPE = re.compile(r"'(ml\.[a-z0-9]+\.[a-z0-9]+) for training job usage'")


class SageMakerCodingGenerationBackend(SageMakerQwenSftBackend):
    """Reuse the existing exact-version byte transport and native request verifier."""

    backend_name = "coding_generation"

    def submit(self, *, model, cases, checkpoint, catalog, spec_sha256, evaluation_id, runtime_session_id):
        validate_catalog(catalog)
        validate_model(model, catalog, runtime_session_id)
        files = validate_cases(cases, catalog, runtime_session_id)
        arm = "base" if checkpoint is None else "tuned"
        binding = {
            "schema_version": 1,
            "operation": "generate",
            "catalog_id": "qwen3_5_9b",
            "catalog_version": "1",
            "recipe_id": "humaneval_greedy_v1",
            "recipe_version": "1",
            "source_commit": catalog.source_commit,
            "evaluation_id": evaluation_id,
            "arm": arm,
            "spec_sha256": spec_sha256,
            "contract_sha256": CONTRACT_SHA256,
            "model_files": expected_model_weight_manifest("qwen3_5_9b", "1")["files"],
            "prompts_ref": dataclasses.asdict(files["PROMPTS.jsonl"]),
        }
        adapter_files = {}
        if checkpoint is not None:
            if (
                type(checkpoint) is not VerifiedSftCheckpointSource
                or checkpoint.runtime_session_id != runtime_session_id
                or checkpoint.recipe_id != "full_v1"
                or not re.fullmatch("[a-f0-9]{40}", checkpoint.source_commit)
                or not checkpoint.producer_job_arn.startswith(
                    "arn:aws:sagemaker:"
                    + catalog.region
                    + ":"
                    + catalog.account_id
                    + ":training-job/"
                    + catalog.deployment_prefix
                    + "-"
                )
            ):
                raise ValueError("original full checkpoint source differs")
            adapter_files = dict(checkpoint.file_refs)
            if (
                len(adapter_files) != len(checkpoint.file_refs)
                or set(adapter_files)
                != {"CHECKPOINT.json", "adapter_config.json", "adapter_model.safetensors"}
                or adapter_files["CHECKPOINT.json"] != checkpoint.checkpoint_ref
            ):
                raise ValueError("original closed adapter inventory required")
            for name, artifact in adapter_files.items():
                validate_ref(
                    artifact,
                    catalog,
                    runtime_session_id,
                    limit=2 * 1024**3 if name.endswith(".safetensors") else 262144,
                )
                if artifact.format != ("safetensors" if name.endswith(".safetensors") else "json"):
                    raise ValueError("original adapter format differs")
            binding.update(
                checkpoint_ref=dataclasses.asdict(checkpoint.checkpoint_ref),
                checkpoint_source_commit=checkpoint.source_commit,
            )
        validate_generation_binding(binding, source_commit=catalog.source_commit)
        deadline = time.monotonic() + 3600
        sm, s3 = self._clients(catalog.region)
        if (
            s3.get_bucket_versioning(Bucket=catalog.bucket, ExpectedBucketOwner=catalog.account_id).get(
                "Status"
            )
            != "Enabled"
        ):
            raise ValueError("generation requires versioned selected bucket")
        # Verify the original acquisition inventory before staging only the nine model files.
        base_refs = dict(model.file_refs)
        parents = {artifact.uri.rsplit("/", 1)[0] + "/" for artifact in base_refs.values()}
        if len(parents) != 1:
            raise ValueError("original base inventory requires one confined prefix")
        base_uri = parents.pop()
        keys = {artifact.uri.split("/", 3)[3] for artifact in base_refs.values()}
        colocated_manifests = tuple(
            artifact
            for _role, artifact in model.manifest_refs
            if artifact.uri.rsplit("/", 1)[0] + "/" == base_uri
        )
        # An adopted pack keeps its adoption record beside the copied files (ModelPackAdoptionProvider.adopt);
        # it is part of the exact inventory and is verified like the provenance files, never staged.
        adoption = getattr(model, "adoption_ref", None)
        if adoption is not None and adoption.uri.rsplit("/", 1)[0] + "/" == base_uri:
            colocated_manifests = (*colocated_manifests, adoption)
        inventory = keys | {artifact.uri.split("/", 3)[3] for artifact in colocated_manifests}
        listing = s3.list_objects_v2(
            Bucket=catalog.bucket,
            Prefix=base_uri.split("/", 3)[3],
            ExpectedBucketOwner=catalog.account_id,
            MaxKeys=len(inventory) + 1,
        )
        if (
            listing.get("IsTruncated") is not False
            or {item.get("Key") for item in listing.get("Contents", [])} != inventory
            or len(listing.get("Contents", [])) != len(inventory)
        ):
            raise ValueError("original native base prefix has extra or missing files")
        # Every pin below is hash-verified here, in the process that builds the handle, and recorded with the HEAD
        # fingerprint of the verification; observation re-checks the pins against that record (`_reverify`).
        verified = {}
        for artifact in (*base_refs.values(), *colocated_manifests):
            self._raw(artifact, catalog, deadline, seen=verified)
            current = s3.head_object(
                Bucket=catalog.bucket,
                Key=artifact.uri.split("/", 3)[3],
                ExpectedBucketOwner=catalog.account_id,
            )
            if current.get("VersionId") != artifact.version_id:
                raise ValueError("unversioned native base selection drifted")
        self._raw(files["PROMPTS.jsonl"], catalog, deadline, seen=verified)
        # The native name is a function of the request: a repeat after a worker loss meets the same job.
        job = (
            catalog.deployment_prefix
            + "-gen-"
            + generation_operation_digest(
                runtime_session_id, catalog, model, cases, checkpoint, spec_sha256, evaluation_id
            )[:16]
        )
        prefix = "sessions/" + runtime_session_id + "/generation/" + job + "/"
        staged = {
            "base_model/" + name: self._adopt_or_copy(
                artifact, prefix + "input/base_model/" + name, catalog, deadline, seen=verified
            )
            for name, artifact in base_refs.items()
        }
        staged["prompts/PROMPTS.jsonl"] = self._adopt_or_copy(
            files["PROMPTS.jsonl"], prefix + "input/prompts/PROMPTS.jsonl", catalog, deadline, seen=verified
        )
        for name, artifact in adapter_files.items():
            self._raw(artifact, catalog, deadline, seen=verified)
            staged["adapter/" + name] = self._adopt_or_copy(
                artifact, prefix + "input/adapter/" + name, catalog, deadline, seen=verified
            )
        raw = _json(binding)
        key = prefix + "input/bindings/RUN.json"
        existing = self._existing_version(key, catalog)
        result = (
            {"VersionId": existing}
            if existing is not None
            else s3.put_object(
                Bucket=catalog.bucket,
                Key=key,
                Body=raw,
                IfNoneMatch="*",
                ExpectedBucketOwner=catalog.account_id,
                ContentType="application/json",
            )
        )
        run_ref = DatasetArtifactRef(
            "s3://" + catalog.bucket + "/" + key,
            _version(result.get("VersionId")),
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            "json",
        )
        self._raw(run_ref, catalog, deadline, seen=verified)
        staged["bindings/RUN.json"] = run_ref
        request = fixed_generation_request(catalog, runtime_session_id, job, run_ref, arm)
        arn = "arn:aws:sagemaker:" + catalog.region + ":" + catalog.account_id + ":training-job/" + job
        handle = JobHandle(
            job,
            self.backend_name,
            arn,
            {
                "runtime_session_id": runtime_session_id,
                "catalog": catalog,
                "request": request,
                "run_ref": run_ref,
                "run_binding": binding,
                "staged_refs": staged,
                "original_base_refs": base_refs,
                "verified_pins": verified,
                "submission_state": "ambiguous",
                "arm": arm,
                "evaluation_id": evaluation_id,
            },
        )
        # Retain a deterministic native identity before a single Create; ambiguity never retries.
        try:
            result = sm.create_training_job(**request)
            if type(result) is not dict or result.get("TrainingJobArn") != arn:
                raise ValueError("generation native accepted identity differs")
            handle.extra["submission_state"] = "accepted"
        except Exception as error:
            code = self._error_code(error)
            if code == "ResourceInUse" and self._is_this_native_job(
                sm.describe_training_job, "TrainingJobName", job, "TrainingJobArn", arn
            ):
                # The deterministic name already belongs to this operation: a repeat after a worker loss.
                handle.extra["submission_state"] = "accepted"
                handle.extra["adopted_existing"] = True
                return handle
            if code in ("ValidationException", "AccessDeniedException", "ResourceLimitExceeded"):
                handle.extra["submission_state"] = "rejected"
                handle.extra["rejection_code"] = code
                if code == "ResourceLimitExceeded":
                    handle.extra["rejection_type"] = self._limit_type(error, request)
            handle.extra["error_type"] = type(error).__name__
        return handle

    @staticmethod
    def _limit_type(error, request):
        """The one listed instance type a ResourceLimitExceeded message names as at its limit, or None.

        Closed: the message must be a string of at most 2,048 characters that names exactly one type in SageMaker's
        form ("'<type> for training job usage'"), and that type must be one the request lists. No other text of the
        message is kept.
        """
        response = getattr(error, "response", None)
        nested = response.get("Error") if type(response) is dict else None
        message = nested.get("Message") if type(nested) is dict else None
        if type(message) is not str or len(message) > 2048:
            return None
        named = set(_LIMIT_TYPE.findall(message))
        listed = [item["InstanceType"] for item in request["ResourceConfig"]["InstancePreferences"]]
        if len(named) != 1:
            return None
        name = named.pop()
        return name if name in listed else None

    def _original_prompt_rows(self, artifact, catalog, session, deadline):
        """Read only the exact original public prompt version and validate its bytes."""
        validate_ref(artifact, catalog, session, limit=2 * 1024**2)
        if artifact.format != "jsonl":
            raise ValueError("original public prompt format differs")
        _deadline(deadline)
        response = self._observation_io(
            self._clients(catalog.region)[1].get_object,
            Bucket=catalog.bucket,
            Key=artifact.uri.split("/", 3)[3],
            ExpectedBucketOwner=catalog.account_id,
            VersionId=artifact.version_id,
        )
        body = response["Body"]
        with self._owned_observation_body(body, deadline=deadline):
            if (
                response.get("VersionId") != artifact.version_id
                or type(response.get("ContentLength")) is not int
                or response["ContentLength"] != artifact.byte_count
            ):
                raise ValueError("original prompt readback version/count differs")
            chunks, count, hashed = [], 0, hashlib.sha256()
            while True:
                _deadline(deadline)
                chunk = self._observation_read(body, min(65536, artifact.byte_count - count + 1))
                _deadline(deadline)
                if type(chunk) is not bytes:
                    raise ValueError("original prompt body must contain bytes")
                if not chunk:
                    break
                count += len(chunk)
                if count > artifact.byte_count:
                    raise ValueError("original prompt read bound exceeded")
                chunks.append(chunk)
                hashed.update(chunk)
            if count != artifact.byte_count or hashed.hexdigest() != artifact.sha256:
                raise ValueError("original prompt bytes/hash differ")
            lines = b"".join(chunks).splitlines(keepends=True)
            if len(lines) != 164 or any(len(line) > 65536 or not line.endswith(b"\n") for line in lines):
                raise ValueError("original public prompt population/row bound differs")
            rows = tuple(decode_object(line, limit=65536) for line in lines)
            if prompts_sha256(rows) != artifact.sha256:
                raise ValueError("original canonical public prompt population differs")
            self._qualify_io()
        self._qualify_io()
        return rows

    @staticmethod
    def _validate_candidate(case, prompt):
        """Validate the worker's closed outcome retention contract, never grade it."""
        outcome, stage = case["outcome"], case["failure_stage"]
        allowed = {
            "generated",
            "invalid",
            "generation_error",
            "setup_error",
            "context_overflow",
            "output_overflow",
            "invalid_termination",
            "truncated",
            "incomplete",
        }
        if (
            type(outcome) is not str
            or outcome not in allowed
            or type(stage) is not str
            or stage not in {"none", "setup", "render", "tokenize", "generate", "decode"}
        ):
            raise ValueError("closed generation outcome/stage required")
        if (
            type(case["prompt_sha256"]) is not str
            or case["prompt_sha256"] != hashlib.sha256(canonical_bytes(prompt)).hexdigest()
        ):
            raise ValueError("original per-case public prompt hash differs")
        rendered = case["rendered_sha256"]
        if type(rendered) is not str or (rendered != "" and not re.fullmatch("[a-f0-9]{64}", rendered)):
            raise ValueError("rendered prompt hash type/shape differs")
        input_tokens, generated_tokens = case["input_tokens"], case["generated_tokens"]
        byte_count, source_hash, source = case["source_byte_count"], case["source_sha256"], case["source"]
        if (
            type(input_tokens) is not int
            or not 0 <= input_tokens <= 65536
            or type(generated_tokens) is not int
            or not 0 <= generated_tokens <= 1024
            or type(byte_count) is not int
            or not 0 <= byte_count <= 2**64 - 1
        ):
            raise ValueError("closed generation integer bounds differ")
        if source_hash is not None and (
            type(source_hash) is not str or not re.fullmatch("[a-f0-9]{64}", source_hash)
        ):
            raise ValueError("retained source hash differs")
        if source is not None:
            if (
                type(source) is not str
                or len(source.encode()) > 65536
                or len(source.encode()) != byte_count
                or hashlib.sha256(source.encode()).hexdigest() != source_hash
            ):
                raise ValueError("retained candidate bytes differ")
        elif outcome != "output_overflow" and (source_hash is not None or byte_count != 0):
            raise ValueError("absent retained candidate metadata differs")
        if outcome == "setup_error":
            valid = (
                stage == "setup"
                and rendered == ""
                and input_tokens == generated_tokens == 0
                and source is None
            )
        elif outcome == "context_overflow":
            valid = (
                stage == "tokenize"
                and bool(rendered)
                and input_tokens > 3072
                and generated_tokens == 0
                and source is None
            )
        elif outcome == "generation_error":
            if stage == "render":
                valid = rendered == "" and input_tokens == generated_tokens == 0 and source is None
            elif stage == "tokenize":
                valid = bool(rendered) and generated_tokens == 0 and source is None
            elif stage == "generate":
                valid = (
                    bool(rendered) and 0 < input_tokens <= 3072 and generated_tokens == 0 and source is None
                )
            else:
                valid = stage == "decode" and bool(rendered) and 0 < input_tokens <= 3072
        else:
            valid = (
                bool(rendered)
                and 0 < input_tokens <= 3072
                and stage == ("none" if outcome == "generated" else "decode")
            )
            if outcome == "output_overflow":
                valid = (
                    valid
                    and source is None
                    and source_hash is not None
                    and byte_count > 65536
                    and generated_tokens > 0
                )
            else:
                valid = valid and source is not None
                if outcome == "generated":
                    valid = (
                        valid
                        and bool(source)
                        and 0 < generated_tokens < 1024
                        and _valid_module(source, prompt["entry_point"])
                    )
                elif outcome == "invalid":
                    valid = (
                        valid
                        and 0 < generated_tokens < 1024
                        and (not source or not _valid_module(source, prompt["entry_point"]))
                    )
                elif outcome == "truncated":
                    valid = valid and generated_tokens == 1024
                elif outcome == "incomplete":
                    valid = valid and generated_tokens < 1024
                else:
                    valid = valid and generated_tokens > 0
        if not valid:
            raise ValueError("worker outcome retention contract differs")

    def _require_current_version(self, artifact, catalog):
        """The SageMaker channel reads the current version of a staged key: it must still be the pinned one."""
        head = self._observation_io(
            self._clients(catalog.region)[1].head_object,
            Bucket=catalog.bucket,
            Key=artifact.uri.split("/", 3)[3],
            ExpectedBucketOwner=catalog.account_id,
        )
        if type(head) is not dict or head.get("VersionId") != artifact.version_id:
            raise ValueError("staged key no longer resolves to the pinned version")

    def artifacts(self, handle):
        status = self.status(handle)
        base = {
            "success": False,
            "accepted": handle.extra["submission_state"] == "accepted",
            "generated": False,
            "quality_success": False,
            "native_admission": False,
            "status": status.state.value,
            "job_name": handle.id,
            "job_arn": handle.arn,
            "arm": handle.extra["arm"],
            "evaluation_id": handle.extra["evaluation_id"],
            **observed_selected_instance(status, handle.extra.get("request")),
        }
        if status.state != JobState.SUCCEEDED:
            return base
        try:
            catalog = handle.extra["catalog"]
            staged_refs = handle.extra["staged_refs"]
            pin_refs = (*staged_refs.values(), *handle.extra["original_base_refs"].values())
            verified = handle.extra.get("verified_pins")
            verified = verified if type(verified) is dict else {}
            # 600 s for the fixed work (status, output reads, row checks) plus the time a full read of every pin
            # that `submit` did not record may need, at a third of the slowest measured stream.
            deadline = (
                time.monotonic()
                + 600
                + _hash_seconds(sum(a.byte_count for a in pin_refs if _pin_key(a) not in verified))
            )
            uri = status.raw.get("ModelArtifacts", {}).get("S3ModelArtifacts")
            root = handle.extra["request"]["OutputDataConfig"]["S3OutputPath"] + "/"
            if (
                type(uri) is not str
                or not uri.startswith(root)
                or any(char in uri for char in ("?", "#", "\\"))
            ):
                raise ValueError("uncompressed native generation output differs")
            relative = uri[len(root) :]
            if relative.endswith("/"):
                relative = relative[:-1]
            if any(part in ("", ".", "..") for part in relative.split("/")):
                raise ValueError("uncompressed native generation output differs")
            prefix = uri.split("/", 3)[3] + ("" if uri.endswith("/") else "/")
            s3 = self._clients(catalog.region)[1]
            listing = self._observation_io(
                s3.list_objects_v2,
                Bucket=catalog.bucket,
                Prefix=prefix,
                ExpectedBucketOwner=catalog.account_id,
                MaxKeys=4,
            )
            names = {"CANDIDATES.jsonl", "GENERATION.json", "COMMITTED.json"}
            if (
                listing.get("IsTruncated") is not False
                or len(listing.get("Contents", [])) != 3
                or {item.get("Key") for item in listing.get("Contents", [])}
                != {prefix + name for name in names}
            ):
                raise ValueError("closed generation output inventory differs")
            committed_ref, raw = self._output(
                handle, prefix + "COMMITTED.json", deadline, limit=262144, collect=True
            )
            commit = decode_object(raw, limit=262144)
            if (
                set(commit)
                != {
                    "schema_version",
                    "operation",
                    "source_commit",
                    "case_count",
                    "quality_success",
                    "native_admission",
                    "files",
                }
                or type(commit["schema_version"]) is not int
                or commit["schema_version"] != 1
                or commit["operation"] != "generate"
                or commit["source_commit"] != catalog.source_commit
                or type(commit["case_count"]) is not int
                or commit["case_count"] != 164
                or commit["quality_success"] is not False
                or commit["native_admission"] is not False
                or type(commit["files"]) is not dict
                or set(commit["files"]) != names - {"COMMITTED.json"}
            ):
                raise ImmutableArtifactRejectionError("closed_manifest_invalid")
            refs, values = {}, {}
            for name in ("GENERATION.json", "CANDIDATES.jsonl"):
                artifact, raw = self._output(
                    handle,
                    prefix + name,
                    deadline,
                    limit=12 * 1024**2 if name.endswith(".jsonl") else 262144,
                    collect=True,
                )
                artifact = dataclasses.replace(
                    artifact, format="jsonl" if name.endswith(".jsonl") else "json"
                )
                if commit["files"][name] != {
                    "sha256": artifact.sha256,
                    "byte_count": artifact.byte_count,
                    "format": artifact.format,
                }:
                    raise ValueError("generation committed bytes differ")
                refs[name], values[name] = artifact, raw
            receipt = decode_object(values["GENERATION.json"], limit=262144)
            binding = handle.extra["run_binding"]
            validate_generation_binding(binding, source_commit=catalog.source_commit)
            if (
                binding["arm"] != handle.extra["arm"]
                or binding["evaluation_id"] != handle.extra["evaluation_id"]
            ):
                raise ValueError("original generation header differs")
            expected_files = binding["model_files"]
            model_sha = hashlib.sha256(_json(expected_files)).hexdigest()
            if binding["arm"] == "tuned":
                model_sha = hashlib.sha256(
                    (model_sha + binding["checkpoint_ref"]["sha256"]).encode()
                ).hexdigest()
            tokenizer_sha = hashlib.sha256(
                _json(
                    [
                        item
                        for item in expected_files
                        if item["role"] in ("tokenizer", "tokenizer_config", "chat_template")
                    ]
                )
            ).hexdigest()
            expected_binding = {
                "evaluation_id": binding["evaluation_id"],
                "arm": binding["arm"],
                "prompts_sha256": binding["prompts_ref"]["sha256"],
                "model_sha256": model_sha,
                "tokenizer_sha256": tokenizer_sha,
                "spec_sha256": binding["spec_sha256"],
            }
            if (
                set(receipt)
                != {
                    "schema_version",
                    "binding",
                    "contract_sha256",
                    "case_count",
                    "cases",
                    "operation",
                    "source_commit",
                    "stage_manifest_sha256",
                    "privilege_proof",
                    "quality_success",
                    "native_admission",
                }
                or receipt["binding"] != expected_binding
                or receipt["contract_sha256"] != CONTRACT_SHA256
                or receipt["source_commit"] != catalog.source_commit
                or receipt["stage_manifest_sha256"] != handle.extra["run_ref"].sha256
                or type(receipt["schema_version"]) is not int
                or receipt["schema_version"] != 1
                or type(receipt["case_count"]) is not int
                or receipt["case_count"] != 164
                or receipt["operation"] != "generate"
                or receipt["quality_success"] is not False
                or receipt["native_admission"] is not False
            ):
                raise ValueError("generation original receipt binding differs")
            validate_native_privilege_proof(receipt["privilege_proof"])
            rows = values["CANDIDATES.jsonl"].splitlines(keepends=True)
            if len(rows) != 164 or type(receipt["cases"]) is not list or len(receipt["cases"]) != 164:
                raise ValueError("all164 generation outcomes required")
            original_ref = DatasetArtifactRef(**binding["prompts_ref"])
            validate_ref(original_ref, catalog, handle.extra["runtime_session_id"], limit=2 * 1024**2)
            original_prompts = self._original_prompt_rows(
                original_ref, catalog, handle.extra["runtime_session_id"], deadline
            )
            for index, line in enumerate(rows):
                if len(line) > 512 * 1024 or not line.endswith(b"\n"):
                    raise ValueError("generation row bound differs")
                case = decode_object(line, limit=512 * 1024)
                if (
                    set(case)
                    != {
                        "case_id",
                        "prompt_sha256",
                        "rendered_sha256",
                        "input_tokens",
                        "generated_tokens",
                        "outcome",
                        "source",
                        "source_sha256",
                        "source_byte_count",
                        "failure_stage",
                    }
                    or case["case_id"] != CASE_IDS[index]
                    or type(receipt["cases"][index]) is not dict
                    or canonical_bytes({key: value for key, value in case.items() if key != "source"})
                    != canonical_bytes(receipt["cases"][index])
                ):
                    raise ValueError("generation case metadata differs")
                self._validate_candidate(case, original_prompts[index])
            for artifact in pin_refs:
                self._reverify(artifact, catalog, deadline, verified)
            for artifact in staged_refs.values():
                self._require_current_version(artifact, catalog)
            self._raw(committed_ref, catalog, deadline)
            self._qualify_io()
            return {
                **base,
                "success": True,
                "generated": True,
                "status": "generated",
                "candidate_ref": refs["CANDIDATES.jsonl"],
                "generation_ref": refs["GENERATION.json"],
                "committed_ref": committed_ref,
                "case_count": 164,
            }
        except Exception as error:
            return {
                **base,
                "status": "output_verification_failed",
                "error_type": type(error).__name__,
                **rejection_data(error),
            }
