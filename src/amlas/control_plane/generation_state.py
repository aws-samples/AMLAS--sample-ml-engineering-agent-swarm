"""Private same-session generation state; specification data never grants execution."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading

from ..backends.ports import CodingGenerationBackend
from ..backends.types import DatasetArtifactRef, JobHandle, JobState, is_capacity_rejection
from .coding_case_state import SessionCodingCaseFreezeStore, VerifiedCaseEvaluationSource
from .model_acquisition_state import SessionModelAcquisitionStore
from .sft_state import (
    GENERATION_INSTANCE_PREFERENCES,
    SessionSftStore,
    SftCatalog,
    fixed_resource_config,
    validate_catalog,
    validate_model,
    validate_ref,
)

# The closed reason a read of a capacity-refused generation or paired evaluation names beside `submission_rejected`.
CAPACITY_REJECTION_REASON = "capacity_limit"

_CODING_SPEC_REASON_PATHS = {
    "bounded_text_required": "$",
    "parse_or_schema_invalid": "$",
    "task_type_mismatch": "task_type",
    "primary_metric_mismatch": "primary_metric",
    "generation_required": "generation",
    "generation_quality_metric_mismatch": "generation.quality.metric",
    "model_pin_required": "generation.model",
    "tokenizer_pin_mismatch": "generation.tokenizer",
    "model_profile_mismatch": "generation.model",
    "recipe_required": "generation.recipe",
    "recipe_profile_mismatch": "generation.recipe",
}


def _operation_digest(domain, material):
    """Domain-separated SHA-256 of the canonical JSON of closed typed DATA (bytes become hex)."""

    def plain(value):
        if type(value) is bytes:
            return value.hex()
        raise TypeError("operation identity requires plain DATA")

    raw = json.dumps(
        material, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False, default=plain
    ).encode()
    return hashlib.sha256(domain.encode() + b"\0" + raw).hexdigest()


def generation_operation_digest(
    runtime_session_id, catalog, model, cases, checkpoint, spec_sha256, evaluation_id
):
    """Identity of one generation request: the session, the deployment catalog and the verified inputs.

    The conversation, the tool-use identifier and the process are not part of it, so a replacement worker that
    repeats the request derives the same digest, the same reference and the same native job name.
    """
    return _operation_digest(
        "amlas.generation.operation.v1",
        {
            "session": runtime_session_id,
            "catalog": dataclasses.asdict(catalog),
            "model": dataclasses.asdict(model),
            "cases": dataclasses.asdict(cases),
            "checkpoint": None if checkpoint is None else dataclasses.asdict(checkpoint),
            "spec": spec_sha256,
            "evaluation": evaluation_id,
            "arm": "base" if checkpoint is None else "tuned",
        },
    )


@dataclasses.dataclass(frozen=True)
class CodingSpecValidationDiagnostic:
    """Closed planning-validation DATA; never source or execution authority."""

    reason_code: str

    def __post_init__(self):
        if type(self.reason_code) is not str or self.reason_code not in _CODING_SPEC_REASON_PATHS:
            raise ValueError("closed coding specification diagnostic required")

    def as_data(self):
        self.__post_init__()
        return {
            "schema_id": "amlas.fixed_qwen_coding_spec.v1",
            "status": "invalid",
            "reason_codes": [self.reason_code],
            "field_paths": [_CODING_SPEC_REASON_PATHS[self.reason_code]],
            "data_only": True,
            "native_admission": False,
            "quality_success": False,
        }


class CodingSpecValidationError(ValueError):
    """Known predicate refusal with an immutable no-reflection diagnostic."""

    def __init__(self, reason_code, message):
        self._validation = CodingSpecValidationDiagnostic(reason_code)
        super().__init__(message)

    @property
    def validation(self):
        return self._validation


@dataclasses.dataclass(frozen=True)
class VerifiedCodingGenerationSource:
    runtime_session_id: str
    producer_job_arn: str
    evaluation_id: str
    arm: str
    file_refs: tuple[tuple[str, DatasetArtifactRef], ...]


def fixed_generation_request(catalog, session, job, run_ref, arm):
    """Fixed source-owned request; no caller resource/recipe overrides."""
    validate_catalog(catalog)
    validate_ref(run_ref, catalog, session, limit=262144)
    if arm not in ("base", "tuned") or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,62}", job):
        raise ValueError("fixed generation arm/job required")
    prefix = "sessions/" + session + "/generation/" + job + "/"
    if run_ref.uri != "s3://" + catalog.bucket + "/" + prefix + "input/bindings/RUN.json":
        raise ValueError("generation RUN namespace differs")
    return {
        "TrainingJobName": job,
        "RoleArn": catalog.execution_role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": catalog.image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["/usr/local/bin/amlas-processing-init", "qwen-sft"],
            "ContainerArguments": [
                "--operation",
                "generate",
                "--recipe-id",
                "humaneval_greedy_v1",
                "--recipe-version",
                "1",
                "--source-commit",
                catalog.source_commit,
                "--manifest-sha256",
                run_ref.sha256,
                "--manifest-byte-count",
                str(run_ref.byte_count),
            ],
        },
        "InputDataConfig": [
            {
                "ChannelName": channel,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://" + catalog.bucket + "/" + prefix + "input/" + channel + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
                "CompressionType": "None",
            }
            for channel in ("base_model", "prompts", *(("adapter",) if arm == "tuned" else ()), "bindings")
        ],
        "ResourceConfig": fixed_resource_config(GENERATION_INSTANCE_PREFERENCES),
        "StoppingCondition": {"MaxRuntimeInSeconds": 14400},
        "EnableNetworkIsolation": True,
        "EnableInterContainerTrafficEncryption": False,
        "OutputDataConfig": {
            "S3OutputPath": "s3://" + catalog.bucket + "/" + prefix + "output",
            "CompressionType": "NONE",
        },
        "Tags": [
            {"Key": "amlas:" + key, "Value": value}
            for key, value in (
                ("deployment-id", catalog.deployment_id),
                ("deployment-prefix", catalog.deployment_prefix),
                ("runtime-session-id", session),
            )
        ],
    }


def validate_cases(cases, catalog, session):
    if type(cases) is not VerifiedCaseEvaluationSource or cases.runtime_session_id != session:
        raise ValueError("original typed same-session cases required")
    expected = {"PROMPTS.jsonl", "ORACLES.jsonl", "EXCLUSIONS.json", "CASE_MANIFEST.json", "COMMITTED.json"}
    refs = dict(cases.file_refs)
    if len(refs) != len(cases.file_refs) or set(refs) != expected:
        raise ValueError("private case inventory differs")
    header = dict(cases.producer_header)
    if any(
        header.get(key) != value
        for key, value in (
            ("runtime_session_id", session),
            ("deployment_id", catalog.deployment_id),
            ("deployment_prefix", catalog.deployment_prefix),
        )
    ):
        raise ValueError("private case producer scope differs")
    if not cases.producer_job_arn.startswith(
        "arn:aws:sagemaker:"
        + catalog.region
        + ":"
        + catalog.account_id
        + ":processing-job/"
        + catalog.deployment_prefix
        + "-"
    ):
        raise ValueError("private case producer differs")
    for name, artifact in refs.items():
        validate_ref(artifact, catalog, session, limit=2 * 1024**2 + 65536)
        if not artifact.uri.endswith("/" + name) or artifact.format != (
            "jsonl" if name.endswith(".jsonl") else "json"
        ):
            raise ValueError("private case role differs")
    return refs


def _observation_kwargs(sft, handle):
    """The SFT store's private observation arguments for one original, or None when they cannot be formed."""
    try:
        return sft._reload_observation_kwargs(handle.extra.get("reload_compatibility"))
    except Exception:
        return None


def _observed(status, handle, private):
    """One read-only native observation as (state, description); a failed observation reads UNKNOWN."""
    try:
        observed = status(handle, **private)
    except Exception:
        return JobState.UNKNOWN, {}
    raw = getattr(observed, "raw", None)
    return observed.state, raw if type(raw) is dict else {}


def _requested_types(handle):
    """(fixed type, listed types) of an original's stored request; (None, ()) when it names neither."""
    request = (
        handle.extra.get("request") if type(handle) is JobHandle and type(handle.extra) is dict else None
    )
    resource = request.get("ResourceConfig") if type(request) is dict else None
    if type(resource) is not dict:
        return None, ()
    if type(resource.get("InstanceType")) is str:
        return resource["InstanceType"], ()
    listed = resource.get("InstancePreferences")
    if type(listed) is not list:
        return None, ()
    return None, tuple(
        item["InstanceType"]
        for item in listed
        if type(item) is dict and type(item.get("InstanceType")) is str
    )


def _held_type(handle, description):
    """The type a PENDING or RUNNING original holds: its fixed type, or the type SageMaker selected from its list.

    None while SageMaker has not yet selected a type from the list (the job then waits for one of its listed types).
    """
    fixed, listed = _requested_types(handle)
    if fixed is not None:
        return fixed
    resource = description.get("ResourceConfig")
    selected = resource.get("SelectedInstanceType") if type(resource) is dict else None
    return selected if type(selected) is str and selected in listed else None


class SessionCodingGenerationStore:
    """Retain original handles during one conversation; no restart or policy persistence."""

    def __init__(self, models, cases, sft, adapter, catalog):
        if (
            not isinstance(models, SessionModelAcquisitionStore)
            or type(cases) is not SessionCodingCaseFreezeStore
            or not isinstance(sft, SessionSftStore)
            or not isinstance(adapter, CodingGenerationBackend)
            or type(catalog) is not SftCatalog
            or not (models.runtime_session_id == cases.runtime_session_id == sft.runtime_session_id)
            or sft._models is not models
        ):
            raise ValueError("typed original same-session generation stores required")
        self._models, self._cases, self._sft, self._adapter, self._catalog = (
            models,
            cases,
            sft,
            adapter,
            catalog,
        )
        self.runtime_session_id = models.runtime_session_id
        self._specs, self._records, self._keys = {}, {}, {}
        self._spec_identity = {}
        self._resolved_specs = {}
        self._lock = threading.RLock()
        # A replacement worker sets this to re-register, from its admitted conversation, the specification
        # references an earlier worker handed out. It runs at most once per lookup miss and registers nothing the
        # same tools would not have registered.
        self._planning_restorer = None
        self._restoring = False

    def _restore_planning(self) -> None:
        restorer = getattr(self, "_planning_restorer", None)
        if restorer is None or getattr(self, "_restoring", False):
            return
        self._restoring = True
        try:
            restorer(self)
        finally:
            self._restoring = False

    @staticmethod
    def _original_spec_material(yaml_text):
        from ..arbiter.coding import coding_spec_bytes, coding_spec_sha256
        from ..spec import load_spec_from_text

        if type(yaml_text) is not str or not 0 < len(yaml_text.encode()) <= 65536:
            raise CodingSpecValidationError("bounded_text_required", "bounded generation spec text required")
        from ..spec import SpecLoadError

        try:
            spec = load_spec_from_text(yaml_text)
        except SpecLoadError as error:
            raise CodingSpecValidationError(
                "parse_or_schema_invalid", "invalid generation specification"
            ) from error
        generation = spec.generation
        from ..data_eng.model_weight_manifest import expected_model_weight_manifest

        expected = expected_model_weight_manifest("qwen3_5_9b", "1")
        reason = None
        if spec.task_type != "text_generation":
            reason = "task_type_mismatch"
        elif spec.primary_metric != "pass_at_1":
            reason = "primary_metric_mismatch"
        elif generation is None:
            reason = "generation_required"
        elif generation.quality.metric != "pass_at_1":
            reason = "generation_quality_metric_mismatch"
        elif generation.model is None:
            reason = "model_pin_required"
        elif generation.tokenizer != generation.model:
            reason = "tokenizer_pin_mismatch"
        elif (generation.model.repo_id, generation.model.revision) != (
            expected["repo_id"],
            expected["revision"],
        ):
            reason = "model_profile_mismatch"
        elif generation.recipe is None:
            reason = "recipe_required"
        elif (generation.recipe.id, generation.recipe.version) != ("full_v1", "1"):
            reason = "recipe_profile_mismatch"
        if reason is not None:
            raise CodingSpecValidationError(reason, "fixed Qwen coding semantic spec required")
        return yaml_text.encode(), coding_spec_bytes(spec), coding_spec_sha256(spec)

    def bind_spec(self, yaml_text):
        text, raw, digest = self._original_spec_material(yaml_text)
        # Store immutable bytes; a mutable SpecModel instance is never retained as authority.
        with self._lock:
            reference = self._spec_identity.get(raw)
            if reference is None:
                reference = "spec_" + _operation_digest("amlas.spec.original.v1", {"spec": raw.hex()})[:32]
                self._specs[reference] = (text, raw, digest)
                self._spec_identity[raw] = reference
        return {
            "spec_ref": reference,
            "spec_sha256": digest,
            "native_admission": False,
            "quality_success": False,
        }

    def restore_original_spec(self, reference, yaml_text, *, expected_spec_sha256):
        """Private DATA restore; the caller independently qualifies protected origin."""
        if (
            type(reference) is not str
            or not re.fullmatch("spec_[a-f0-9]{32}", reference)
            or type(expected_spec_sha256) is not str
            or not re.fullmatch("[a-f0-9]{64}", expected_spec_sha256)
        ):
            raise ValueError("original spec ref and independent canonical digest required")
        text, raw, digest = self._original_spec_material(yaml_text)
        if digest != expected_spec_sha256:
            raise ValueError("original specification canonical bytes differ")
        with self._lock:
            existing = self._specs.get(reference)
            identity = self._spec_identity.get(raw)
            if (existing is not None and existing != (text, raw, digest)) or (
                identity is not None and identity != reference
            ):
                raise ValueError("original specification alias or body conflicts")
            self._specs[reference] = (text, raw, digest)
            self._spec_identity[raw] = reference
        return {
            "spec_ref": reference,
            "spec_sha256": digest,
            "native_admission": False,
            "quality_success": False,
        }

    def resolve_case_spec(self, case_ref, spec_ref):
        """Mint a separate immutable researched spec bound to original private cases."""
        from ..arbiter.coding import coding_spec_bytes, coding_spec_sha256
        from ..spec.schema import SpecModel

        if (
            type(case_ref) is not str
            or not re.fullmatch("case_[a-f0-9]{32}", case_ref)
            or type(spec_ref) is not str
            or not re.fullmatch("spec_[a-f0-9]{32}", spec_ref)
        ):
            raise ValueError("opaque original case/spec references required")
        with self._lock:
            original = self._specs.get(spec_ref)
            if original is None:
                self._restore_planning()
                original = self._specs.get(spec_ref)
            if original is None:
                raise ValueError("unknown original specification reference")
            cases = self._cases._verified_evaluation_source(case_ref)
            files = validate_cases(cases, self._catalog, self.runtime_session_id)
            projection = json.loads(original[1])
            declared = projection["generation"]["evaluation"]["case_manifest_ref"]
            manifest = dataclasses.asdict(files["CASE_MANIFEST.json"])
            if declared is not None and declared != manifest:
                raise ValueError("declared case manifest differs from original private cases")
            projection["generation"]["evaluation"]["case_manifest_ref"] = manifest
            resolved = SpecModel.model_validate(projection)
            raw, digest = coding_spec_bytes(resolved), coding_spec_sha256(resolved)
            key = (case_ref, raw, files["CASE_MANIFEST.json"], files["COMMITTED.json"])
            existing = next((ref for ref, value in self._resolved_specs.items() if value == key), None)
            reference = (
                existing
                or "spec_"
                + _operation_digest(
                    "amlas.spec.resolved.v1",
                    {
                        "case": case_ref,
                        "spec": raw.hex(),
                        "manifest": dataclasses.asdict(files["CASE_MANIFEST.json"]),
                        "committed": dataclasses.asdict(files["COMMITTED.json"]),
                    },
                )[:32]
            )
            if existing is None:
                self._specs[reference] = (original[0], raw, digest)
                self._resolved_specs[reference] = key
            return {
                "spec_ref": reference,
                "original_spec_ref": spec_ref,
                "spec_sha256": digest,
                "case_bound": True,
                "quality_success": False,
                "native_admission": False,
            }

    def _verified_resolved_spec(self, case_ref, spec_ref):
        """Fresh original case observation plus immutable resolved planning bytes."""
        with self._lock:
            frozen = self._resolved_specs.get(spec_ref) if type(spec_ref) is str else None
            if frozen is None and type(spec_ref) is str:
                self._restore_planning()
                frozen = self._resolved_specs.get(spec_ref)
            spec = self._specs.get(spec_ref) if type(spec_ref) is str else None
        if frozen is None or spec is None or frozen[0] != case_ref:
            raise ValueError("resolved same-session case specification required")
        cases = self._cases._verified_evaluation_source(case_ref)
        files = validate_cases(cases, self._catalog, self.runtime_session_id)
        if frozen != (case_ref, spec[1], files["CASE_MANIFEST.json"], files["COMMITTED.json"]):
            raise ValueError("resolved private case version changed")
        return spec[1], spec[2], cases

    def submit(self, model_ref, case_ref, spec_ref, sft_ref=""):
        validate_catalog(self._catalog)
        if any(
            store.runtime_session_id != self.runtime_session_id
            for store in (self._models, self._cases, self._sft)
        ):
            raise ValueError("generation conversation changed")
        for value, pattern in ((model_ref, "model_acq_"), (case_ref, "case_"), (spec_ref, "spec_")):
            if type(value) is not str or not re.fullmatch(pattern + "[a-f0-9]{32}", value):
                raise ValueError("opaque generation sources required")
        if type(sft_ref) is not str or (sft_ref and not re.fullmatch("sft_[a-f0-9]{32}", sft_ref)):
            raise ValueError("opaque original training reference required")
        with self._lock:
            spec = self._specs.get(spec_ref)
            if spec is None:
                self._restore_planning()
                spec = self._specs.get(spec_ref)
            if spec is None:
                raise ValueError("unknown same-session spec reference")
            resolved = self._resolved_specs.get(spec_ref)
            if resolved is None or resolved[0] != case_ref:
                raise ValueError("resolved original case specification required before generation")
            key = (model_ref, case_ref, spec[1], sft_ref)
            # A capacity rejection made no job: the same request reaches the adapter again (same reference and
            # deterministic native name). Every other recorded outcome stays the answer.
            if key in self._keys and not is_capacity_rejection(self._records[self._keys[key]]):
                return self._submitted(self._keys[key], reused=True)
            model = self._models._verified_sft_model_snapshot(model_ref)
            validate_model(model, self._catalog, self.runtime_session_id)
            cases = self._cases._verified_evaluation_source(case_ref)
            case_files = validate_cases(cases, self._catalog, self.runtime_session_id)
            if resolved != (
                case_ref,
                spec[1],
                case_files["CASE_MANIFEST.json"],
                case_files["COMMITTED.json"],
            ):
                raise ValueError("resolved original case versions differ")
            from ..spec.schema import SpecModel

            planning = SpecModel.model_validate_json(spec[1])
            declared = planning.generation.evaluation.case_manifest_ref
            if declared is not None and declared.model_dump() != dataclasses.asdict(
                case_files["CASE_MANIFEST.json"]
            ):
                raise ValueError("spec case manifest differs from original frozen case")
            evaluation = (
                "coding_"
                + hashlib.sha256(
                    json.dumps(
                        [
                            self.runtime_session_id,
                            spec[2],
                            dataclasses.asdict(case_files["COMMITTED.json"]),
                            dataclasses.asdict(model),
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()[:48]
            )
            checkpoint = self._sft._verified_generation_checkpoint(sft_ref, model) if sft_ref else None
            handle = self._adapter.submit(
                model=model,
                cases=cases,
                checkpoint=checkpoint,
                catalog=self._catalog,
                spec_sha256=spec[2],
                evaluation_id=evaluation,
                runtime_session_id=self.runtime_session_id,
            )
            if (
                type(handle) is not JobHandle
                or handle.backend != "coding_generation"
                or handle.extra.get("runtime_session_id") != self.runtime_session_id
            ):
                raise ValueError("original generation handle differs")
            reference = (
                "gen_"
                + generation_operation_digest(
                    self.runtime_session_id, self._catalog, model, cases, checkpoint, spec[2], evaluation
                )[:32]
            )
            self._records[reference] = handle
            self._keys[key] = reference
            if is_capacity_rejection(handle):
                # The tool result names the one listed type SageMaker refused for, who holds it and when to
                # submit again.
                return {
                    **self._submitted(reference),
                    **self._capacity_note(
                        GENERATION_INSTANCE_PREFERENCES,
                        *self._active_holders(GENERATION_INSTANCE_PREFERENCES, reference),
                        handle.extra.get("rejection_type"),
                    ),
                }
            return self._submitted(reference)

    def _active_holders(self, instance_types, reference):
        """Which jobs of this conversation hold the listed types, from read-only observations.

        Called only after a capacity rejection, to name the holders in the tool result. Returns `(holders, waiting)`:
        `holders` maps every type in `instance_types` to the name of a PENDING or RUNNING original that holds it (its
        fixed type, or the type SageMaker selected from its list), or None; `waiting` names PENDING or RUNNING
        originals whose list shares a type with `instance_types` and that have no selected type yet.

        Candidates are the other generation originals and the SFT store's originals whose create was accepted or is
        unknown. An observation that fails or reads UNKNOWN is not a holder; the native create then decides.
        """
        holders, waiting, _unread = self._observed_holders(instance_types, reference)
        return holders, waiting

    def _observed_holders(self, instance_types, reference):
        """`_active_holders` plus `unread`: the candidates whose observation failed or read UNKNOWN.

        A read of a capacity-refused reference uses it, so that its note does not state as fact what it could not
        read. The records are copied under the store lock; the native reads run outside it.
        """
        with self._lock:
            records = tuple(self._records.items())
        candidates = [(handle, self._adapter.status, {}) for other, handle in records if other != reference]
        sft = self._sft
        lock = getattr(sft, "_lock", None)
        if lock is not None:
            with lock:
                originals = tuple(getattr(sft, "_records", {}).values())
        else:
            originals = tuple(getattr(sft, "_records", {}).values())
        for handle in originals:
            private = _observation_kwargs(sft, handle)
            if private is not None:
                candidates.append((handle, sft._adapter.status, private))
        holders = {name: None for name in instance_types}
        waiting, unread = [], []
        for handle, status, private in candidates:
            fixed, listed = _requested_types(handle)
            if (
                type(handle) is not JobHandle
                or handle.extra.get("submission_state") not in ("accepted", "ambiguous")
                or not ({fixed} | set(listed)) & set(instance_types)
            ):
                continue
            state, description = _observed(status, handle, private)
            if state is JobState.UNKNOWN and handle.id not in unread:
                unread.append(handle.id)
            if state not in (JobState.PENDING, JobState.RUNNING):
                continue
            held = _held_type(handle, description)
            if held is None:
                if listed and handle.id not in waiting:
                    waiting.append(handle.id)
            elif held in holders and holders[held] is None:
                holders[held] = handle.id
        return holders, waiting, unread

    @staticmethod
    def _capacity_note(instance_types, holders, waiting, blocking, *, read=False, unread=()):
        """Plain next step for a capacity limit: the one listed type at its limit, who holds it, when to submit again.

        SageMaker refuses a create when any listed type is at the account limit and names that type. `blocking` is
        the closed `rejection_type` of the rejected handle: that listed type, or None when the message named none.
        Only that type is named. `type_holders` and `waiting_job_names` stay as supporting detail: a list job of
        this run that waits without a selected type holds no type, so it is never named as the holder.

        `read=True` writes the note for a later read of the refused reference, from the holders observed at the
        read. While a job of this run holds the named type the sentences are the submit's own. Otherwise the read
        says that no running job of this run was found on that type (the refusal is no longer fresh, so it does not
        claim that another job holds it). It then names the jobs of this run on the other listed types, which also
        refuse a create, and says to wait for them. When no job of this run was found on a listed type it names the
        jobs of this run that wait without a type and the ones in `unread` (None: the lookup itself failed), says to
        submit now and that a job on a listed type refuses the create again. `holding_job_name` stays the holder of
        the named type.
        """
        if type(blocking) is not str or blocking not in instance_types:
            blocking = None
        held = [(name, job) for name, job in holders.items() if job is not None]
        run_holders = [job for _name, job in held]
        held_sentences = [
            "Job " + job + " of this run holds the " + name + " capacity." for name, job in held
        ]
        submit_now = False
        if blocking is None:
            holder = run_holders[0] if run_holders else None
            sentences = [
                "SageMaker did not name a listed instance type as the one at its limit.",
                *held_sentences,
            ]
            after = "a job on a listed type ends."
            if read and not held:
                sentences.append("No running job of this run was found on a listed instance type.")
                submit_now = True
        else:
            holder = holders.get(blocking)
            after = "that job ends."
            if holder is not None:
                sentences = ["Job " + holder + " of this run holds the " + blocking + " capacity."]
            elif not read:
                sentences = ["Another job in the account holds the " + blocking + " capacity."]
            else:
                sentences = [
                    "No running job of this run was found on the " + blocking + " capacity.",
                    *held_sentences,
                ]
                if held:
                    after = "that job ends." if len(held) == 1 else "those jobs end."
                else:
                    submit_now = True
        if submit_now:
            sentences += [
                "Job " + job + " of this run waits for a machine and has no instance type yet."
                for job in waiting
            ]
            if unread is None:
                sentences.append("The jobs of this run could not be read.")
            else:
                sentences += ["Job " + job + " of this run could not be read." for job in unread]
        sentences.append("Nothing was created.")
        if submit_now:
            sentences.append("Submit this arm again with the same arguments now.")
            sentences.append(
                "If another job holds a listed type, SageMaker refuses again and creates nothing."
            )
        else:
            sentences.append("Submit this arm again with the same arguments after " + after)
        return {
            "instance_types": list(instance_types),
            "blocking_instance_type": blocking,
            "type_holders": dict(holders),
            "waiting_job_names": list(waiting),
            "holding_job_name": holder,
            "next_step": " ".join(sentences),
        }

    def _submitted(self, reference, *, reused=False):
        h = self._records[reference]
        accepted = h.extra["submission_state"] == "accepted"
        return {
            "success": False,
            "accepted": accepted,
            "generated": False,
            "quality_success": False,
            "native_admission": False,
            "request_ref": reference,
            "job_name": h.id,
            "job_arn": h.arn,
            "evaluation_id": h.extra["evaluation_id"],
            "arm": h.extra["arm"],
            "status": "submitted"
            if accepted
            else "submission_rejected"
            if h.extra["submission_state"] == "rejected"
            else "acceptance_unknown",
            "reused": reused,
        }

    def observe(self, reference):
        with self._lock:
            handle = (
                self._records.get(reference)
                if type(reference) is str and re.fullmatch("gen_[a-f0-9]{32}", reference)
                else None
            )
        if handle is None:
            return {
                "success": False,
                "accepted": False,
                "generated": False,
                "quality_success": False,
                "native_admission": False,
                "status": "unknown_reference",
            }
        try:
            if is_capacity_rejection(handle):
                refusal = self._capacity_refusal(reference)
                if refusal is not None:
                    return refusal
                # A submit created the job while the holders were read: observe the record it stored.
                with self._lock:
                    handle = self._records[reference]
            return {**self._submitted(reference), **self._adapter.artifacts(handle), "request_ref": reference}
        except Exception as error:
            return {
                **self._submitted(reference),
                "status": "observation_failed",
                "error_type": type(error).__name__,
            }

    def _capacity_refusal(self, reference):
        """The read of a capacity-refused original: the refusal itself, from its stored record and live holders.

        The refused create made no native job, so nothing is described under its name, and a read creates and
        retries nothing and writes nothing to the record. The answer is the submit's: `submission_rejected` with
        the closed reason `capacity_limit`, the stored `rejection_type` and the note for the jobs of this run that
        hold a listed type now (read-only observations, as the submit path makes them). The holders are read
        outside the store lock; the answer is then built from the record the store holds at that moment, so a
        submit that was refused again meanwhile is answered from its own refusal. A failed holder lookup does not
        replace the refusal: the note then says that the jobs of this run could not be read. Returns None when that
        record is no longer a capacity refusal.
        """
        try:
            holders, waiting, unread = self._observed_holders(GENERATION_INSTANCE_PREFERENCES, reference)
        except Exception:
            # The refusal is still the answer when the jobs of this run cannot be read.
            holders, waiting, unread = {name: None for name in GENERATION_INSTANCE_PREFERENCES}, [], None
        with self._lock:
            handle = self._records.get(reference)
            if not is_capacity_rejection(handle):
                return None
            return {
                **self._submitted(reference),
                "rejection_reason": CAPACITY_REJECTION_REASON,
                **self._capacity_note(
                    GENERATION_INSTANCE_PREFERENCES,
                    holders,
                    waiting,
                    handle.extra.get("rejection_type"),
                    read=True,
                    unread=unread,
                ),
            }

    def _verified_evaluation_source(self, reference):
        """Fresh private original output handoff, never reconstructed from public JSON."""
        with self._lock:
            handle = (
                self._records.get(reference)
                if type(reference) is str and re.fullmatch("gen_[a-f0-9]{32}", reference)
                else None
            )
        if handle is None or handle.extra.get("runtime_session_id") != self.runtime_session_id:
            raise ValueError("unknown original generation reference")
        result = self._adapter.artifacts(handle)
        if (
            type(result) is not dict
            or result.get("success") is not True
            or result.get("generated") is not True
            or type(result.get("case_count")) is not int
            or result["case_count"] != 164
        ):
            raise ValueError("original generated outputs unverified")
        refs = (
            ("CANDIDATES.jsonl", result["candidate_ref"]),
            ("GENERATION.json", result["generation_ref"]),
            ("COMMITTED.json", result["committed_ref"]),
        )
        for name, artifact in refs:
            validate_ref(
                artifact,
                self._catalog,
                self.runtime_session_id,
                limit=12 * 1024**2 if name.endswith(".jsonl") else 262144,
            )
            if not artifact.uri.endswith("/" + name) or artifact.format != (
                "jsonl" if name.endswith(".jsonl") else "json"
            ):
                raise ValueError("generation output role differs")
        return VerifiedCodingGenerationSource(
            self.runtime_session_id, handle.arn, handle.extra["evaluation_id"], handle.extra["arm"], refs
        )
