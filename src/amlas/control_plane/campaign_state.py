"""Closed private SFT operation records, scoped to a verified execution role.

This is server-only state transport, not user authentication or action authority.
The table adapter mints bindings after native identity verification. Public JSON,
opaque artifact refs and persisted records never supply approval policies. Restored
source descriptors require the original fresh native/artifact validators before
consumption. This slice does not restore conversation engines or run a campaign.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..backends.types import DatasetArtifactRef
from .sft_state import (
    RELOAD_RESOURCE_CONTRACTS,
    ReloadCompatibilitySelection,
    SftCatalog,
    SftSubmissionSnapshot,
    VerifiedSftAdoptedModelSource,
    VerifiedSftCheckpointSource,
    VerifiedSftInputs,
    VerifiedSftModelSource,
    VerifiedSftPreparedSource,
    fixed_sft_request,
    validate_inputs,
    validate_ref,
    validate_reload_compatibility,
)
from .sft_state import (
    validate_catalog as validate_sft_catalog,
)

_MAX_RECORD = 256 * 1024
_MAX_CAUSAL_PARENT_BYTES = 1024 * 1024
_SEAL = object()
_STATES = frozenset({"RESERVED", "CREATE_ATTEMPTED", "ACCEPTED", "AMBIGUOUS", "REJECTED"})
_TYPES = {
    cls.__name__: cls
    for cls in (
        SftCatalog,
        ReloadCompatibilitySelection,
        SftSubmissionSnapshot,
        VerifiedSftInputs,
        VerifiedSftAdoptedModelSource,
        VerifiedSftModelSource,
        VerifiedSftPreparedSource,
        VerifiedSftCheckpointSource,
        DatasetArtifactRef,
    )
}


class CampaignStateError(ValueError):
    """A private scope, codec or conditional state transition was refused."""


@dataclass(frozen=True)
class CampaignStateCatalog:
    account_id: str
    region: str
    deployment_id: str
    deployment_prefix: str
    jobs_table: str
    execution_role_arn: str
    execution_role_id: str

    def __post_init__(self):
        if any(type(v) is not str for v in dataclasses.asdict(self).values()):
            raise CampaignStateError("campaign catalog selector types differ")
        if not re.fullmatch(r"\d{12}", self.account_id) or not re.fullmatch(
            r"[a-z]{2}(?:-[a-z]+)+-\d", self.region
        ):
            raise CampaignStateError("campaign account or region differs")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", self.deployment_prefix):
            raise CampaignStateError("campaign deployment prefix differs")
        try:
            if str(uuid.UUID(self.deployment_id)) != self.deployment_id:
                raise ValueError
        except ValueError as error:
            raise CampaignStateError("campaign deployment identifier differs") from error
        if self.jobs_table != self.deployment_prefix + "-training-jobs":
            raise CampaignStateError("selected deployment jobs table required")
        if not re.fullmatch(
            r"arn:aws:iam::" + self.account_id + r":role/[A-Za-z0-9_+=,.@/-]{1,512}", self.execution_role_arn
        ):
            raise CampaignStateError("campaign execution role differs")
        if not re.fullmatch(r"AROA[A-Z0-9]{16,64}", self.execution_role_id):
            raise CampaignStateError("campaign execution role incarnation differs")


@dataclass(frozen=True, init=False)
class TrustedExecutionBinding:
    scope: CampaignStateCatalog
    campaign_id: str
    runtime_session_id: str
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self, scope: CampaignStateCatalog, campaign_id: str, runtime_session_id: str, *, _token: Any = None
    ):
        if _token is not _SEAL or type(scope) is not CampaignStateCatalog:
            raise CampaignStateError("server-verified execution binding required")
        try:
            if type(campaign_id) is not str or str(uuid.UUID(campaign_id)) != campaign_id:
                raise ValueError
        except ValueError as error:
            raise CampaignStateError("server campaign identifier differs") from error
        if type(runtime_session_id) is not str or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", runtime_session_id
        ):
            raise CampaignStateError("server runtime session identifier differs")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "campaign_id", campaign_id)
        object.__setattr__(self, "runtime_session_id", runtime_session_id)
        object.__setattr__(self, "_seal", _SEAL)


def _mint_binding(
    scope: CampaignStateCatalog, campaign_id: str, runtime_session_id: str
) -> TrustedExecutionBinding:
    return TrustedExecutionBinding(scope, campaign_id, runtime_session_id, _token=_SEAL)


def _binding_fields(binding: TrustedExecutionBinding) -> dict:
    if type(binding) is not TrustedExecutionBinding or binding._seal is not _SEAL:
        raise CampaignStateError("typed server execution binding required")
    return {
        **dataclasses.asdict(binding.scope),
        "campaign_id": binding.campaign_id,
        "runtime_session_id": binding.runtime_session_id,
    }


@dataclass(frozen=True)
class CampaignOperationIntent:
    binding: TrustedExecutionBinding = field(repr=False)
    catalog: SftCatalog = field(repr=False)
    inputs: VerifiedSftInputs = field(repr=False)
    recipe_id: str
    operation: str
    checkpoint: VerifiedSftCheckpointSource | None = field(default=None, repr=False)
    _seal: object = field(default=None, repr=False, compare=False)
    reload_compatibility: ReloadCompatibilitySelection | None = field(default=None, repr=False)
    reload_resource_contract: str | None = field(default=None, repr=False)
    source_refs: tuple[str, str] | None = field(default=None, repr=False)
    causal_parent_ref: DatasetArtifactRef | None = field(default=None, repr=False)


def _validate_intent_components(
    binding: TrustedExecutionBinding,
    catalog: SftCatalog,
    inputs: VerifiedSftInputs,
    recipe_id: str,
    operation: str,
    checkpoint: VerifiedSftCheckpointSource | None,
    reload_compatibility: ReloadCompatibilitySelection | None = None,
    reload_resource_contract: str | None = None,
    source_refs: tuple[str, str] | None = None,
    causal_parent_ref: DatasetArtifactRef | None = None,
) -> None:
    _binding_fields(binding)
    if source_refs is not None and (
        operation != "train"
        or type(source_refs) is not tuple
        or len(source_refs) != 2
        or type(source_refs[0]) is not str
        or not re.fullmatch("model_acq_[a-f0-9]{32}", source_refs[0])
        or type(source_refs[1]) is not str
        or not re.fullmatch("prep_[a-f0-9]{32}", source_refs[1])
        or reload_compatibility is not None
        or reload_resource_contract is not None
    ):
        raise CampaignStateError("closed original training source refs required")
    if type(catalog) is not SftCatalog or type(inputs) is not VerifiedSftInputs:
        raise CampaignStateError("original closed SFT source types required")
    try:
        validate_sft_catalog(catalog)
        validate_inputs(inputs, catalog, binding.runtime_session_id)
    except (TypeError, ValueError, KeyError) as error:
        raise CampaignStateError("private SFT source binding differs") from error
    if (catalog.account_id, catalog.region, catalog.deployment_id, catalog.deployment_prefix) != (
        binding.scope.account_id,
        binding.scope.region,
        binding.scope.deployment_id,
        binding.scope.deployment_prefix,
    ):
        raise CampaignStateError("private SFT deployment scope differs")
    if causal_parent_ref is not None:
        if operation != "train" or source_refs is None:
            raise CampaignStateError("causal parent requires original training source refs")
        try:
            validate_ref(
                causal_parent_ref,
                catalog,
                binding.runtime_session_id,
                limit=_MAX_CAUSAL_PARENT_BYTES,
            )
        except (ValueError, TypeError, KeyError) as error:
            raise CampaignStateError("private causal parent descriptor differs") from error
        prefix = (
            "s3://"
            + catalog.bucket
            + "/sessions/"
            + binding.runtime_session_id
            + "/prompt-runs/"
            + uuid.UUID(binding.campaign_id).hex
            + "/parents/"
        )
        if causal_parent_ref.format != "json" or causal_parent_ref.uri != (
            prefix + causal_parent_ref.sha256 + ".json"
        ):
            raise CampaignStateError("private causal parent namespace or format differs")
    if recipe_id not in ("fit_v1", "full_v1") or operation not in ("train", "reload"):
        raise CampaignStateError("fixed source-owned SFT operation required")
    if reload_resource_contract is not None and (
        operation != "reload"
        or type(reload_resource_contract) is not str
        or reload_resource_contract not in RELOAD_RESOURCE_CONTRACTS
    ):
        raise CampaignStateError("fixed private reload resource contract differs")
    if operation == "train" and (checkpoint is not None or reload_compatibility is not None):
        raise CampaignStateError("training intent must not contain a checkpoint/compatibility")
    if operation == "reload":
        if (
            type(checkpoint) is not VerifiedSftCheckpointSource
            or checkpoint.runtime_session_id != binding.runtime_session_id
            or (reload_compatibility is None and checkpoint.source_commit != catalog.source_commit)
            or checkpoint.recipe_id != recipe_id
        ):
            raise CampaignStateError("original private checkpoint binding differs")
        try:
            if reload_compatibility is not None:
                validate_reload_compatibility(reload_compatibility, catalog, checkpoint)
            validate_ref(checkpoint.checkpoint_ref, catalog, binding.runtime_session_id, limit=256 * 1024**2)
            for _name, ref in checkpoint.file_refs:
                validate_ref(ref, catalog, binding.runtime_session_id, limit=256 * 1024**2)
        except (ValueError, TypeError) as error:
            raise CampaignStateError("private checkpoint descriptor differs") from error


def build_sft_intent(
    binding: TrustedExecutionBinding,
    catalog: SftCatalog,
    inputs: VerifiedSftInputs,
    *,
    recipe_id: str = "fit_v1",
    operation: str = "train",
    checkpoint: VerifiedSftCheckpointSource | None = None,
    reload_compatibility: ReloadCompatibilitySelection | None = None,
    reload_resource_contract: str | None = None,
    source_refs: tuple[str, str] | None = None,
    causal_parent_ref: DatasetArtifactRef | None = None,
) -> CampaignOperationIntent:
    _validate_intent_components(
        binding,
        catalog,
        inputs,
        recipe_id,
        operation,
        checkpoint,
        reload_compatibility,
        reload_resource_contract,
        source_refs,
        causal_parent_ref,
    )
    value = CampaignOperationIntent(
        binding,
        catalog,
        inputs,
        recipe_id,
        operation,
        checkpoint,
        _SEAL,
        reload_compatibility,
        reload_resource_contract,
        source_refs,
        causal_parent_ref,
    )
    _intent_fields(value)
    return value


def _pack(value: Any) -> Any:
    if value is None or type(value) in (str, int):
        return value
    if type(value) is tuple:
        return {"type": "tuple", "items": [_pack(v) for v in value]}
    cls = type(value)
    if cls.__name__ in _TYPES and _TYPES[cls.__name__] is cls:
        return {
            "type": cls.__name__,
            "fields": {f.name: _pack(getattr(value, f.name)) for f in dataclasses.fields(cls)},
        }
    raise CampaignStateError("private codec type is not a closed SFT type")


def _unpack(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise CampaignStateError("private record nesting bound exceeded")
    if value is None or type(value) in (str, int):
        return value
    if type(value) is not dict:
        raise CampaignStateError("private codec primitive type differs")
    if value.get("type") == "tuple":
        if set(value) != {"type", "items"} or type(value["items"]) is not list:
            raise CampaignStateError("private tuple shape differs")
        return tuple(_unpack(x, depth=depth + 1) for x in value["items"])
    if (
        set(value) != {"type", "fields"}
        or value.get("type") not in _TYPES
        or type(value["fields"]) is not dict
    ):
        raise CampaignStateError("private record type differs")
    cls = _TYPES[value["type"]]
    if set(value["fields"]) != {f.name for f in dataclasses.fields(cls)}:
        raise CampaignStateError("private record field inventory differs")
    return cls(**{k: _unpack(v, depth=depth + 1) for k, v in value["fields"].items()})


def _intent_fields(intent: CampaignOperationIntent) -> dict:
    if type(intent) is not CampaignOperationIntent or intent._seal is not _SEAL:
        raise CampaignStateError("typed private operation intent required")
    _validate_intent_components(
        intent.binding,
        intent.catalog,
        intent.inputs,
        intent.recipe_id,
        intent.operation,
        intent.checkpoint,
        intent.reload_compatibility,
        intent.reload_resource_contract,
        intent.source_refs,
        intent.causal_parent_ref,
    )
    value = {
        "binding": _binding_fields(intent.binding),
        "catalog": _pack(intent.catalog),
        "inputs": _pack(intent.inputs),
        "recipe_id": intent.recipe_id,
        "recipe_version": "1",
        "operation": intent.operation,
        "checkpoint": _pack(intent.checkpoint),
    }
    if intent.source_refs is not None:
        value["source_refs"] = _pack(intent.source_refs)
    if intent.causal_parent_ref is not None:
        value["causal_parent_ref"] = _pack(intent.causal_parent_ref)
    if intent.reload_compatibility is not None or intent.reload_resource_contract is not None:
        value["reload_compatibility"] = _pack(intent.reload_compatibility)
    if intent.reload_resource_contract is not None:
        value["reload_resource_contract"] = intent.reload_resource_contract
    return value


def _json(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise CampaignStateError("private record encoding differs") from error
    if not 0 < len(raw) <= _MAX_RECORD:
        raise CampaignStateError("private record byte bound exceeded")
    return raw


def operation_digest(intent: CampaignOperationIntent) -> str:
    return hashlib.sha256(_json(_intent_fields(intent))).hexdigest()


def _native_identity(intent: CampaignOperationIntent) -> tuple[str, str]:
    digest = operation_digest(intent)
    prefix = intent.binding.scope.deployment_prefix
    name = prefix + "-sft-" + digest[: 63 - len(prefix) - 5]
    return (
        name,
        "arn:aws:sagemaker:"
        + intent.binding.scope.region
        + ":"
        + intent.binding.scope.account_id
        + ":training-job/"
        + name,
    )


@dataclass(frozen=True)
class OriginalJobReservation:
    binding: TrustedExecutionBinding = field(repr=False)
    intent: CampaignOperationIntent = field(repr=False)
    operation_digest: str
    native_job_name: str
    native_job_arn: str
    disposition: str = "RESERVED"
    revision: int = 0
    error_code: str | None = None
    reused: bool = False
    submission_snapshot: SftSubmissionSnapshot | None = field(default=None, repr=False)
    _persisted_body: bytes | None = field(default=None, repr=False, compare=False)


def validate_submission_snapshot(record: OriginalJobReservation) -> None:
    snapshot = record.submission_snapshot
    if snapshot is None:
        return
    if (
        type(snapshot) is not SftSubmissionSnapshot
        or (snapshot.operation_digest, snapshot.native_job_name, snapshot.native_job_arn)
        != (record.operation_digest, record.native_job_name, record.native_job_arn)
        or type(snapshot.request_json) is not str
        or not 0 < len(snapshot.request_json.encode()) <= 65536
    ):
        raise CampaignStateError("closed original submission snapshot differs")
    intent = record.intent
    catalog, session = intent.catalog, record.binding.runtime_session_id
    expected = {"base_model/" + name: ref for name, ref in intent.inputs.model.file_refs}
    if intent.operation == "train":
        prepared = intent.inputs.prepared
        expected.update({"prepared/COMMITTED.json": prepared.preparation_ref})
        expected.update({"prepared/" + name + ".jsonl": ref for name, ref in prepared.split_refs})
        expected.update(
            {
                "prepared/"
                + {"report": "REPORT.json", "lineage": "lineage.jsonl", "membership": "membership.jsonl"}[
                    name
                ]: ref
                for name, ref in prepared.sidecar_refs
            }
        )
        if (
            type(snapshot.training_record_count) is not int
            or not 0 < snapshot.training_record_count <= 100000
        ):
            raise CampaignStateError("original training count differs")
    else:
        expected.update({"adapter/" + name: ref for name, ref in intent.checkpoint.file_refs})
        if snapshot.training_record_count is not None:
            raise CampaignStateError("reload must not supply training count")
    if type(snapshot.staged_refs) is not tuple or any(
        type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str for pair in snapshot.staged_refs
    ):
        raise CampaignStateError("closed original staged inventory required")
    staged = dict(snapshot.staged_refs)
    if len(staged) != len(snapshot.staged_refs) or set(staged) != set(expected) | {"bindings/RUN.json"}:
        raise CampaignStateError("original staged file roles differ")
    prefix = "s3://" + catalog.bucket + "/sessions/" + session + "/sft/" + record.native_job_name + "/input/"
    for name, ref in staged.items():
        validate_ref(ref, catalog, session)
        if ref.uri != prefix + name:
            raise CampaignStateError("original staged path differs")
        if name in expected:
            original = expected[name]
            if (ref.sha256, ref.byte_count, ref.format) != (
                original.sha256,
                original.byte_count,
                original.format,
            ):
                raise CampaignStateError("original staged source bytes differ")
    if staged["bindings/RUN.json"] != snapshot.run_ref or snapshot.run_ref.format != "json":
        raise CampaignStateError("original RUN descriptor differs")
    request = fixed_sft_request(
        catalog,
        session,
        record.native_job_name,
        intent.operation,
        intent.recipe_id,
        snapshot.run_ref,
        resource_contract=intent.reload_resource_contract,
    )
    accepted = {_json(request).decode()}
    if intent.operation == "train":
        # A training record written before the capacity option carries the exact fixed ml.g6e.12xlarge request.
        accepted.add(
            _json(
                fixed_sft_request(
                    catalog,
                    session,
                    record.native_job_name,
                    intent.operation,
                    intent.recipe_id,
                    snapshot.run_ref,
                    historical=True,
                )
            ).decode()
        )
    if snapshot.request_json not in accepted:
        raise CampaignStateError("original fixed submission request differs")


def _record_fields(record: OriginalJobReservation, *, schema_version: int | None = None) -> dict:
    if schema_version is None:
        schema_version = (
            6
            if record.intent.causal_parent_ref is not None
            else 5
            if record.intent.source_refs is not None
            else 4
            if record.intent.reload_resource_contract is not None
            else 3
            if record.intent.reload_compatibility is not None
            else 2
        )
    if (schema_version == 6) != (record.intent.causal_parent_ref is not None):
        raise CampaignStateError("private causal-parent schema differs")
    if (schema_version in (5, 6)) != (record.intent.source_refs is not None):
        raise CampaignStateError("private source-ref schema differs")
    if schema_version in (5, 6) and (
        record.intent.operation != "train"
        or record.intent.reload_compatibility is not None
        or record.intent.reload_resource_contract is not None
    ):
        raise CampaignStateError("training source refs cannot use reload schema")
    if (schema_version == 4) != (record.intent.reload_resource_contract is not None):
        raise CampaignStateError("private resource schema differs")
    if schema_version != 4 and (schema_version == 3) != (record.intent.reload_compatibility is not None):
        raise CampaignStateError("private compatibility schema differs")
    if (
        type(record) is not OriginalJobReservation
        or record.disposition not in _STATES
        or type(record.revision) is not int
        or record.revision < 0
    ):
        raise CampaignStateError("private reservation shape differs")
    if (
        _binding_fields(record.binding) != _binding_fields(record.intent.binding)
        or record.operation_digest != operation_digest(record.intent)
        or (record.native_job_name, record.native_job_arn) != _native_identity(record.intent)
    ):
        raise CampaignStateError("private reservation immutable intent differs")
    if record.error_code is not None and (
        type(record.error_code) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", record.error_code)
    ):
        raise CampaignStateError("bounded error class required")
    validate_submission_snapshot(record)
    value = {
        "schema_version": schema_version,
        "binding": _binding_fields(record.binding),
        "intent": {k: v for k, v in _intent_fields(record.intent).items() if k != "binding"},
        "operation_digest": record.operation_digest,
        "native_job_name": record.native_job_name,
        "native_job_arn": record.native_job_arn,
        "disposition": record.disposition,
        "revision": record.revision,
        "error_code": record.error_code,
    }
    if schema_version in (2, 3, 4, 5, 6):
        value["submission_snapshot"] = _pack(record.submission_snapshot)
    elif schema_version != 1 or record.submission_snapshot is not None:
        raise CampaignStateError("private record schema differs")
    return value


def _encode_private_record(record: OriginalJobReservation) -> bytes:
    return _json(_record_fields(record))


def _decode_private_record(raw: bytes, binding: TrustedExecutionBinding) -> OriginalJobReservation:
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_RECORD:
        raise CampaignStateError("private record bytes differ")

    def pairs(values):
        out = {}
        for key, value in values:
            if key in out:
                raise CampaignStateError("duplicate private record field")
            out[key] = value
        return out

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
        keys = {
            "schema_version",
            "binding",
            "intent",
            "operation_digest",
            "native_job_name",
            "native_job_arn",
            "disposition",
            "revision",
            "error_code",
        }
        if type(value) is not dict:
            raise CampaignStateError("private record object required")
        if value.get("schema_version") in (2, 3, 4, 5, 6):
            keys.add("submission_snapshot")
        if (
            set(value) != keys
            or type(value["schema_version"]) is not int
            or value["schema_version"] not in (1, 2, 3, 4, 5, 6)
            or value["binding"] != _binding_fields(binding)
        ):
            raise CampaignStateError("authenticated private record scope or schema differs")
        data = value["intent"]
        intent_keys = {"catalog", "inputs", "recipe_id", "recipe_version", "operation", "checkpoint"}
        if value["schema_version"] in (3, 4):
            intent_keys.add("reload_compatibility")
        if value["schema_version"] == 4:
            intent_keys.add("reload_resource_contract")
        if value["schema_version"] in (5, 6):
            intent_keys.add("source_refs")
        if value["schema_version"] == 6:
            intent_keys.add("causal_parent_ref")
        if type(data) is not dict or set(data) != intent_keys or data["recipe_version"] != "1":
            raise CampaignStateError("private operation field inventory differs")
        intent = build_sft_intent(
            binding,
            _unpack(data["catalog"]),
            _unpack(data["inputs"]),
            recipe_id=data["recipe_id"],
            operation=data["operation"],
            checkpoint=_unpack(data["checkpoint"]),
            reload_compatibility=_unpack(data["reload_compatibility"])
            if value["schema_version"] in (3, 4)
            else None,
            reload_resource_contract=data["reload_resource_contract"]
            if value["schema_version"] == 4
            else None,
            source_refs=_unpack(data["source_refs"]) if value["schema_version"] in (5, 6) else None,
            **(
                {"causal_parent_ref": _unpack(data["causal_parent_ref"])}
                if value["schema_version"] == 6
                else {}
            ),
        )
        record = OriginalJobReservation(
            binding,
            intent,
            value["operation_digest"],
            value["native_job_name"],
            value["native_job_arn"],
            value["disposition"],
            value["revision"],
            value["error_code"],
            submission_snapshot=_unpack(value["submission_snapshot"])
            if value["schema_version"] in (2, 3, 4, 5, 6)
            else None,
        )
        if _json(_record_fields(record, schema_version=value["schema_version"])) != raw:
            raise CampaignStateError("private record canonical bytes or digest differs")
        return dataclasses.replace(record, _persisted_body=raw)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as error:
        raise CampaignStateError("authenticated private record validation failed") from error
