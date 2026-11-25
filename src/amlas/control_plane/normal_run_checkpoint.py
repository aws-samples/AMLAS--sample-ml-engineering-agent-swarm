"""Normal-run enrollment and verified pending originals; no continuation or grants."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from ..backends.adapters.campaign_state import DynamoCampaignStateBackend
from ..backends.adapters.failure_cause import (
    RUN_RECORD_CHANGED,
    SFT_CATALOG_CHANGED,
    closed_cause,
    closed_interruption,
    record_changed,
)
from ..config.externalized import resolve_env_snapshot
from .campaign_state import (
    CampaignStateCatalog,
    _intent_fields,
    _native_identity,
    build_sft_intent,
    operation_digest,
)

__all__ = [
    "NormalRunContext",
    "activate_normal_run",
    "current_normal_run",
    "prepare_normal_run",
    "read_normal_run_status",
]
_MAX_BYTES = 160 * 1024
# The replacement's final report pointer (single assignment, beside the original's `report_checkpoint_ref`).
_RESUMED_REPORT_FIELD = "resumed_report_checkpoint_ref"
# Between the records of a lineage stream, the parent-capture callback runs its local owner/catalog/witness checks
# every time and the native role/table/row checks at most this often (counted from the start of the last full one).
# Every adapter read and every publication is still bracketed by the full check.
_PARENT_RECORD_RECHECK_SECONDS = 1.0


def _record_recheck(check):
    """Callback for qualified lineage reads: `check(True)` is the full check, `check(False)` skips only its remote steps."""
    from ..backends.adapters.coding_preparation import RecordRecheck

    return RecordRecheck(lambda: check(True), lambda: check(False), _PARENT_RECORD_RECHECK_SECONDS)


def _model_read_adapters(models, alias):
    """Adapters that read the original model parent: none for an adopted pack, the record's for a produced one.

    A verified pack restored through `_restore_adopted_parent` lives in `models._adoptions` and leaves
    `models._records` empty; `hydrate_parent_graph` already leaves its model adapter out of the fenced reads.
    An alias in neither map is still refused (KeyError), never skipped.
    """
    if alias in models._adoptions:
        return ()
    return (models._records[alias].adapter,)


_FORBIDDEN = frozenset({"approval_policy", "allowlist", "SecretAccessKey", "AccessKeyId", "SessionToken"})
_CURRENT: ContextVar[NormalRunContext | None] = ContextVar("amlas_normal_run_data", default=None)


def _catalog() -> CampaignStateCatalog | None:
    arn = os.environ.get("AMLAS_RUNTIME_EXEC_ROLE_ARN", "").strip()
    role_id = os.environ.get("AMLAS_RUNTIME_EXEC_ROLE_ID", "").strip()
    if not arn and not role_id:
        return None
    if not arn or not role_id:
        raise ValueError("normal run runtime-role pair incomplete")
    return CampaignStateCatalog(
        account_id=resolve_env_snapshot("aws_account_id"),
        region=resolve_env_snapshot("aws_region"),
        deployment_id=os.environ.get("AMLAS_DEPLOYMENT_ID", "").strip(),
        deployment_prefix=os.environ.get("AMLAS_DEPLOYMENT_PREFIX", "").strip(),
        jobs_table=os.environ.get("AMLAS_JOBS_TABLE", "").strip(),
        execution_role_arn=arn,
        execution_role_id=role_id,
    )


def _refs(run_id: str, session_hash: str | None) -> None:
    if type(run_id) is not str or not re.fullmatch("[a-f0-9]{32}", run_id):
        raise ValueError("normal run identifier invalid")
    if type(session_hash) is not str or not re.fullmatch("[a-f0-9]{64}", session_hash):
        raise ValueError("normal run SDK session unavailable")


def _plain(value: Any, depth: int = 0) -> None:
    if depth > 24:
        raise ValueError("normal run DATA depth exceeded")
    if type(value) is dict:
        if any(type(key) is not str for key in value) or _FORBIDDEN.intersection(value):
            raise ValueError("normal run DATA fields refused")
        for item in value.values():
            _plain(item, depth + 1)
    elif type(value) is list:
        for item in value:
            _plain(item, depth + 1)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("normal run requires plain DATA")


def _canonical(value: Any) -> str:
    _plain(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("normal run DATA size exceeded")
    return raw


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("normal run duplicate DATA key")
        value[key] = item
    return value


def _transport_binding(transport_session_id, session_hash):
    if transport_session_id is None:
        return
    if (
        type(transport_session_id) is not str
        or not 33 <= len(transport_session_id) <= 256
        or len(transport_session_id.encode("utf-8")) > 256
        or any(ord(char) < 32 for char in transport_session_id)
        or hashlib.sha256(transport_session_id.encode("utf-8")).hexdigest() != session_hash
    ):
        raise ValueError("invalid normal transport binding")


@dataclass(frozen=True)
class NormalRunContext:
    """Selected server scope and original correlation DATA; no cached permission."""

    catalog: CampaignStateCatalog
    run_id: str
    session_hash: str
    process_id: int = field(default_factory=os.getpid, repr=False)
    _enrollment_confirmed: bool = field(default=False, init=False, repr=False, compare=False)
    _causal_call: str | None = field(default=None, repr=False, compare=False)
    _sft_snapshot: str | None = field(default=None, repr=False, compare=False)
    _completion_receipt: tuple[str, str, str] | None = field(default=None, repr=False, compare=False)
    # Recorded by the confirmed claim: whether the claimed row already held the original worker's `report_checkpoint_ref`.
    # None until a claim is confirmed; read only by the runtime recovery record of the chat continuation, never by any check.
    _completion_prior_report: bool | None = field(default=None, init=False, repr=False, compare=False)
    # Set in this process when the original worker's model received the result of its `submit_qwen_sft` call (its
    # in-process wait saw the job finish). Process-local DATA, never durable, never an authority; read only by the app
    # when the original worker ends, to record that a later resume of this run has nothing to continue.
    _original_result_delivered: bool = field(default=False, init=False, repr=False, compare=False)
    _completion_attempted_token: str | None = field(default=None, repr=False, compare=False)
    _completion_invocation_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    transport_session_id: str | None = field(default=None, kw_only=True, repr=False, compare=False)

    def __post_init__(self):
        _transport_binding(self.transport_session_id, self.session_hash)

    def _check(self) -> None:
        _refs(self.run_id, self.session_hash)
        _transport_binding(self.transport_session_id, self.session_hash)
        if self.process_id != os.getpid() or type(self.catalog) is not CampaignStateCatalog:
            raise ValueError("normal run process/scope differs")
        if _catalog() != self.catalog:
            raise ValueError("normal run selected configuration changed")

    def note_original_result_delivered(self) -> None:
        """Record that the original's model received its SFT call result in this process; no native effect."""
        object.__setattr__(self, "_original_result_delivered", True)

    def capture_sft_call(
        self,
        pending_tool_use_id,
        causal_prefix,
        *,
        agent_snapshot_dict=None,
        _diagnostic_phase=None,
        _diagnostic_predicate=None,
        _diagnostic_size=None,
    ) -> None:
        """Keep source-admitted DATA and bound its full TASK before any SFT effect."""

        def mark_phase(phase):
            if _diagnostic_phase is not None:
                try:
                    _diagnostic_phase(phase)
                except Exception:
                    pass

        mark_phase("current_before")
        self._check()
        if current_normal_run() is not self:
            raise ValueError("original active normal context required")
        mark_phase("causal_call")
        call = _causal_call(
            pending_tool_use_id,
            causal_prefix,
            _diagnostic_phase=mark_phase,
            _diagnostic_predicate=_diagnostic_predicate,
            _diagnostic_size=_diagnostic_size,
        )
        mark_phase("causal_call_budget")
        raw = _canonical(call)
        snapshot_raw = None
        if agent_snapshot_dict is not None:
            mark_phase("snapshot_inventory")
            if _canonical(_captured_snapshot_messages(agent_snapshot_dict)) != _canonical(
                call["causal_prefix"]
            ):
                raise ValueError("SDK snapshot rewrote the original admitted prefix")
            mark_phase("snapshot_budget")
            _budget_sft_snapshot(self, call, agent_snapshot_dict)
            snapshot_raw = _canonical(agent_snapshot_dict)
            mark_phase("capture_repeat")
            if self._sft_snapshot is not None and self._sft_snapshot != snapshot_raw:
                raise ValueError("original admitted SFT snapshot was already captured")
        else:
            mark_phase("capture_repeat")
        if self._causal_call is not None and self._causal_call != raw:
            raise ValueError("original normal SFT call was already captured")
        if snapshot_raw is not None:
            object.__setattr__(self, "_sft_snapshot", snapshot_raw)
        object.__setattr__(self, "_causal_call", raw)

    def _captured_sft_snapshot(self, pending_tool_use_id, causal_prefix):
        """Return detached budgeted DATA; omitted capture retains legacy behavior."""
        self._check()
        if current_normal_run() is not self:
            raise ValueError("original active normal context required")
        if self._sft_snapshot is None:
            return None
        if _canonical(_causal_call(pending_tool_use_id, causal_prefix)) != self._causal_call:
            raise ValueError("original admitted SFT snapshot call changed")
        snapshot = json.loads(self._sft_snapshot, object_pairs_hook=_pairs)
        if _canonical(_captured_snapshot_messages(snapshot)) != _canonical(causal_prefix):
            raise ValueError("original admitted SFT snapshot prefix changed")
        return snapshot

    def _captured_sft_call(self, model_ref, prep_ref):
        self._check()
        if self._causal_call is None:
            raise ValueError("original admitted SFT call missing")
        call = json.loads(self._causal_call, object_pairs_hook=_pairs)
        use = _causal_call(call["pending_tool_use_id"], call["causal_prefix"])["tool_use"]
        if (
            use["input"]["model_request_ref"] != model_ref
            or use["input"]["preparation_request_ref"] != prep_ref
        ):
            raise ValueError("original SFT source aliases changed")
        return call

    def save_admitted_task(
        self,
        agent_snapshot_dict,
        pending_tool_use_id,
        causal_prefix,
        *,
        original_job_name,
        policy,
        _expected_record=None,
    ):
        """Publish the admitted SDK DATA only as a native-bound causal continuation."""
        from .normal_run_parents import _decode
        from .producer_snapshot import _approved_read, _ref_wire

        _approved_read(policy)
        expected_raw = _canonical(_expected_record) if _expected_record is not None else None
        self._check()
        call = _causal_call(pending_tool_use_id, causal_prefix)
        with _native(self) as (client, table_id, qualify, _sts):
            item, record = _load(self, client, table_id, qualify)
            if expected_raw is not None and _canonical(record) != expected_raw:
                raise ValueError("verified normal TASK publication row changed")
            original, binding = _original(self, record, table_id, original_job_name)
            parent_ref = original.intent.causal_parent_ref

            def current():
                _approved_read(policy)
                qualify()
                _item, now = _load(self, client, table_id, qualify)
                if now != record:
                    raise ValueError("normal original record changed before TASK publication")

            parent = _decode(_read_object(self, binding, parent_ref, "parents", current))
            if (
                parent["pending_tool_use_id"] != pending_tool_use_id
                or _canonical(parent["causal_prefix"]) != _canonical(call["causal_prefix"])
                or parent["source_refs"] != list(original.intent.source_refs)
            ):
                raise ValueError("TASK does not continue the native-bound original call")
            if _canonical(_captured_snapshot_messages(agent_snapshot_dict)) != _canonical(
                call["causal_prefix"]
            ):
                raise ValueError("SDK snapshot rewrote the original admitted prefix")
            task_schema_version = _prepared_task_wire_version(self, binding, record, current)
            task = {
                "schema_version": task_schema_version,
                "logical_run_id": self.run_id,
                "private_store_correlation": record["private_store_correlation"],
                "native_job_name": original.native_job_name,
                "native_job_arn": original.native_job_arn,
                "operation_digest": original.operation_digest,
                "source_refs": list(original.intent.source_refs),
                "causal_parent_ref": _ref_wire(parent_ref),
                "sdk_snapshot": agent_snapshot_dict,
                "pending_tool_use_id": pending_tool_use_id,
            }
            if task_schema_version == 1:
                task["causal_prefix"] = call["causal_prefix"]
            _task_envelope(task)
            raw = _canonical(task).encode()
            prior = record.get("task_checkpoint_ref")
            if prior is not None:
                from .producer_snapshot import _ref

                if _read_object(self, binding, _ref(prior), "tasks", current) != raw:
                    raise ValueError("original TASK checkpoint already differs")
                current()
                return
            task_ref = _put_object(self, binding, raw, "tasks", current)
            _replace_record(
                self,
                client,
                table_id,
                qualify,
                item,
                {**record, "task_checkpoint_ref": _ref_wire(task_ref)},
                policy=policy,
            )
            _approved_read(policy)

    def promote_prepared_task(self, *, policy):
        """Publish prepared SDK DATA only after verifying the original native job."""
        import boto3
        from botocore.config import Config

        from ..backends.adapters.qwen_sft import SageMakerQwenSftBackend
        from ..backends.types import JobState
        from .normal_run_parents import _decode, _snapshot
        from .producer_snapshot import _approved_read, _ref, _ref_wire

        _approved_read(policy)
        self._check()
        with _native(self) as (client, table_id, qualify, _sts):
            _item, record = _load(self, client, table_id, qualify)
            original, binding = _original(self, record, table_id)
            if "prepared_checkpoint_ref" not in record:
                raise ValueError("owned normal prepared checkpoint unavailable")
            prepared_ref = _prepared_ref(self, record)
            parent_ref = original.intent.causal_parent_ref

            def current_check(remote):
                _approved_read(policy)
                if remote:
                    qualify()
                    _item, now = _load(self, client, table_id, qualify)
                    if _canonical(now) != _canonical(record):
                        raise record_changed("normal prepared original selection changed", RUN_RECORD_CHANGED)
                if _selected_sft_catalog() != original.intent.catalog:
                    raise record_changed("normal prepared original selection changed", SFT_CATALOG_CHANGED)
                _approved_read(policy)
                self._check()

            current = _record_recheck(current_check)
            current()
            raw = _read_object(self, binding, prepared_ref, "prepared-tasks", current)
            task = json.loads(raw, object_pairs_hook=_pairs)
            _task_envelope(task)
            if _canonical(task).encode("utf-8") != raw or (
                task["logical_run_id"],
                task["private_store_correlation"],
                task["native_job_name"],
                task["native_job_arn"],
                task["operation_digest"],
                task["source_refs"],
                task["causal_parent_ref"],
            ) != (
                self.run_id,
                record["private_store_correlation"],
                original.native_job_name,
                original.native_job_arn,
                original.operation_digest,
                list(original.intent.source_refs),
                _ref_wire(parent_ref),
            ):
                raise ValueError("prepared checkpoint native original association differs")
            parent = _decode(_read_object(self, binding, parent_ref, "parents", current))
            call = _causal_call(task["pending_tool_use_id"], _task_causal_prefix(task))
            if (
                parent["logical_run_id"] != self.run_id
                or parent["source_refs"] != task["source_refs"]
                or parent["pending_tool_use_id"] != call["pending_tool_use_id"]
                or _canonical(parent["causal_prefix"]) != _canonical(call["causal_prefix"])
                or _canonical(_captured_snapshot_messages(task["sdk_snapshot"]))
                != _canonical(call["causal_prefix"])
                or _effective_sft_inputs(call["tool_use"]["input"])
                != {
                    "model_request_ref": record["sft"]["model_ref"],
                    "preparation_request_ref": record["sft"]["preparation_ref"],
                    "recipe_id": original.intent.recipe_id,
                }
                or parent["model_artifacts"] != _snapshot(original.intent.inputs.model)
                or parent["preparation_artifacts"] != _snapshot(original.intent.inputs.prepared)
            ):
                raise ValueError("prepared checkpoint original causal parent differs")
            options = Config(connect_timeout=5, read_timeout=20, retries={"total_max_attempts": 1})
            session = boto3.Session(region_name=self.catalog.region)
            sm = s3 = None
            try:
                sm = session.client("sagemaker", region_name=self.catalog.region, config=options)
                s3 = session.client("s3", region_name=self.catalog.region, config=options)
                adapter = SageMakerQwenSftBackend(
                    sagemaker_client=_NativeIO(sm, current, {"describe_training_job", "list_tags"}),
                    s3_client=_NativeIO(s3, current, {"get_bucket_versioning", "get_object", "head_object"}),
                )
                with adapter.qualified_io(current):
                    handle = adapter.restore_original(original, fresh_inputs=original.intent.inputs)
                    # For a preference-list request, status.detail also names the instance SageMaker selected.
                    status = adapter.status(handle)
                    current()
                if handle.extra.get("restoration_status") != "verified" or status.state not in (
                    JobState.RUNNING,
                    JobState.SUCCEEDED,
                ):
                    raise ValueError("prepared original SFT is not freshly verified eligible")
                current()
                self.save_admitted_task(
                    task["sdk_snapshot"],
                    task["pending_tool_use_id"],
                    _task_causal_prefix(task),
                    original_job_name=original.native_job_name,
                    policy=policy,
                    _expected_record=record,
                )
                _approved_read(policy)
                _item, confirmed = _load(self, client, table_id, qualify)
                if "task_checkpoint_ref" not in confirmed:
                    raise ValueError("promoted TASK checkpoint unavailable")
                task_ref = _ref(confirmed["task_checkpoint_ref"])
                expected = {**record, "task_checkpoint_ref": _ref_wire(task_ref)}
                if _canonical(confirmed) != _canonical(expected):
                    raise ValueError("promoted TASK original record differs")
                record = expected
                if _read_object(self, binding, task_ref, "tasks", current) != raw:
                    raise ValueError("promoted TASK canonical bytes differ")
                current()
            finally:
                for native in (sm, s3):
                    close = getattr(native, "close", None)
                    if callable(close):
                        close()

    def load_qualified_task(self, *, policy):
        """Return temporary verified parents and SDK DATA, without invoking any agent."""
        import boto3
        from botocore.config import Config

        from ..backends.adapters.qwen_sft import SageMakerQwenSftBackend
        from .normal_run_parents import _decode, _snapshot, hydrate_parent_graph
        from .producer_snapshot import _approved_read, _ref, _ref_wire

        _approved_read(policy)
        with _native(self) as (client, table_id, qualify, _sts):
            _item, record = _load(self, client, table_id, qualify)
            original, binding = _original(self, record, table_id)
            if "task_checkpoint_ref" not in record:
                raise ValueError("owned normal TASK checkpoint unavailable")
            parent_ref = original.intent.causal_parent_ref

            def current_check(remote):
                _approved_read(policy)
                if remote:
                    qualify()
                    _item, now = _load(self, client, table_id, qualify)
                    if now != record:
                        raise record_changed("normal original TASK selection changed", RUN_RECORD_CHANGED)
                if _selected_sft_catalog() != original.intent.catalog:
                    raise record_changed("normal original TASK selection changed", SFT_CATALOG_CHANGED)

            current = _record_recheck(current_check)

            task_raw = _read_object(self, binding, _ref(record["task_checkpoint_ref"]), "tasks", current)
            task = json.loads(task_raw, object_pairs_hook=_pairs)
            _task_envelope(task)
            if _canonical(task).encode() != task_raw or (
                task["logical_run_id"],
                task["private_store_correlation"],
                task["native_job_name"],
                task["native_job_arn"],
                task["operation_digest"],
                task["source_refs"],
                task["causal_parent_ref"],
            ) != (
                self.run_id,
                record["private_store_correlation"],
                original.native_job_name,
                original.native_job_arn,
                original.operation_digest,
                list(original.intent.source_refs),
                _ref_wire(parent_ref),
            ):
                raise ValueError("TASK native original association differs")
            parent_raw = _read_object(self, binding, parent_ref, "parents", current)
            parent = _decode(parent_raw)
            call = _causal_call(task["pending_tool_use_id"], _task_causal_prefix(task))
            if (
                parent["logical_run_id"] != self.run_id
                or parent["source_refs"] != task["source_refs"]
                or parent["pending_tool_use_id"] != call["pending_tool_use_id"]
                or (_canonical(parent["causal_prefix"]) != _canonical(call["causal_prefix"]))
                or (
                    _canonical(_captured_snapshot_messages(task["sdk_snapshot"]))
                    != _canonical(call["causal_prefix"])
                )
            ):
                raise ValueError("TASK causal parent continuation differs")
            parents = hydrate_parent_graph(parent_raw, binding, policy, qualify=current)
            if parent["model_artifacts"] != _snapshot(original.intent.inputs.model) or parent[
                "preparation_artifacts"
            ] != _snapshot(original.intent.inputs.prepared):
                raise ValueError("fresh graph differs from the original native SFT inputs")
            options = Config(connect_timeout=5, read_timeout=20, retries={"total_max_attempts": 1})
            session = boto3.Session(region_name=self.catalog.region)
            sm = session.client("sagemaker", region_name=self.catalog.region, config=options)
            s3 = session.client("s3", region_name=self.catalog.region, config=options)
            try:
                adapter = SageMakerQwenSftBackend(
                    sagemaker_client=_NativeIO(sm, current, {"describe_training_job", "list_tags"}),
                    s3_client=_NativeIO(s3, current, {"get_bucket_versioning", "get_object", "head_object"}),
                )
                with adapter.qualified_io(current):
                    handle = adapter.restore_original(original, fresh_inputs=original.intent.inputs)
                    if handle.extra.get("restoration_status") != "verified":
                        raise ValueError("original SFT native request remains unverified")
                current()
            finally:
                for native in (sm, s3):
                    close = getattr(native, "close", None)
                    if callable(close):
                        close()
            return (task, parents, original, binding)

    def qualify_original_completion(self, task, store, *, policy):
        """Fresh original native/artifact proof; never accept a saved status flag."""
        from .normal_run_parents import _qualified_reads
        from .producer_snapshot import _approved_read

        _approved_read(policy)
        with _native(self) as (client, table_id, qualify, _sts):
            _item, record = _load(self, client, table_id, qualify)
            original, binding = _original(self, record, table_id, task["native_job_name"])
            alias = "sft_" + original.operation_digest[:32]
            if (
                store.runtime_session_id != binding.runtime_session_id
                or store._source_refs.get(alias) != original.intent.source_refs
            ):
                raise ValueError("original completed store binding changed")

            association_raw = _canonical(record["sft"])

            def current_check(remote):
                _approved_read(policy)
                if remote:
                    qualify()
                    now = _load(self, client, table_id, qualify, _projection=True)[1]
                    if (
                        now.sft != association_raw
                        or now.private_store_correlation != record["private_store_correlation"]
                    ):
                        raise record_changed("original completion association changed", RUN_RECORD_CHANGED)

            current = _record_recheck(current_check)

            model = _model_read_adapters(store._models, original.intent.source_refs[0])
            with _qualified_reads(
                (*model, store._prepared._adapter, store._adapter), self.catalog.region, current
            ):
                observed = store.observe(alias)
                if (
                    observed.get("request_ref") != alias
                    or observed.get("job_name") != original.native_job_name
                    or observed.get("job_arn") != original.native_job_arn
                    # A fit pilot is complete when trained; full_v1 also needs its full-training proof.
                    or not _original_training_complete(observed)
                ):
                    raise ValueError("fresh full original completion proof unavailable")
                current()

    def claim_completion(self, task, *, policy, qualify_original):
        """Fence one admitted whole-batch continuation; attempted CAS never repeats."""
        from .producer_snapshot import _approved_read, _ref

        if not callable(qualify_original):
            raise ValueError("source original completion qualifier required")
        _task_envelope(task)
        _captured_snapshot_messages(task["sdk_snapshot"])
        _approved_read(policy)
        qualify_original()
        with _native(self) as (client, table_id, qualify, _sts):
            item, record = _load(self, client, table_id, qualify)
            original, binding = _original(self, record, table_id, task["native_job_name"])

            def current():
                _approved_read(policy)
                qualify()
                qualify_original()

            stored = json.loads(
                _read_object(self, binding, _ref(record["task_checkpoint_ref"]), "tasks", current),
                object_pairs_hook=_pairs,
            )
            if {**task, "sdk_snapshot": stored["sdk_snapshot"]} != stored:
                raise ValueError("completion TASK original metadata changed")
            expected_snapshot = _completed_task_snapshot(stored, original.operation_digest)
            messages = task["sdk_snapshot"]["data"]["messages"]
            if _canonical(messages) != _canonical(expected_snapshot["data"]["messages"]):
                raise ValueError("completion snapshot is not the exact original batch transition")
            if _canonical(task["sdk_snapshot"]) != _canonical(expected_snapshot):
                raise ValueError("completion modified non-message SDK state")
            snapshot_sha256 = _sdk_snapshot_sha256(expected_snapshot)
            prior = record.get("continuation")
            if prior is None:
                token = uuid.uuid4().hex
                continuation = {
                    "schema_version": 3,
                    "epoch": 1,
                    "token": token,
                    "status": "consumed-ready",
                    "operation_digest": original.operation_digest,
                    "sdk_snapshot_sha256": snapshot_sha256,
                    "invocation_state": "ready",
                    "invocation_token": None,
                }
            elif (
                prior.get("schema_version") in (2, 3)
                and prior.get("invocation_state") == "ready"
                and prior.get("operation_digest") == original.operation_digest
                and (
                    prior.get("sdk_snapshot_sha256") == snapshot_sha256
                    if prior["schema_version"] == 3
                    else _canonical(prior["sdk_snapshot"]) == _canonical(task["sdk_snapshot"])
                )
            ):
                # READY proves no invocation start has been committed. Reconstruct
                # DATA only; each contender must still win the start CAS below.
                token = prior["token"]
                continuation = prior
            else:
                # Schema1 ownership and any started invocation are ambiguous.
                raise ValueError("original continuation already has an owner")
            current()
            if prior is None:
                _replace_record(
                    self,
                    client,
                    table_id,
                    current,
                    item,
                    {**record, "continuation": continuation},
                    policy=policy,
                )
            current()
            confirmed = _load(self, client, table_id, qualify)[1]
            if (
                confirmed.get("continuation") != continuation
                or confirmed.get("task_checkpoint_ref") != record["task_checkpoint_ref"]
            ):
                raise ValueError("original continuation claim unresolved")
            current()
            # One retaining statement: the receipt and the prior-report fact of the confirmed claim are installed together.
            for name, value in (
                ("_completion_prior_report", "report_checkpoint_ref" in confirmed),
                (
                    "_completion_receipt",
                    (token, _canonical(continuation), _canonical(record["task_checkpoint_ref"])),
                ),
            ):
                object.__setattr__(self, name, value)
            return token

    def qualify_completion(self, token, *, policy, qualify_original, sdk_snapshot):
        """Fresh authority and exact process-local content continuity checks."""
        from .producer_snapshot import _approved_read, _ref

        receipt = self._completion_receipt
        if type(receipt) is not tuple or len(receipt) != 3 or receipt[0] != token:
            raise ValueError("original continuation claim receipt unavailable")
        _approved_read(policy)
        qualify_original()
        with _native(self) as (client, table_id, qualify, _sts):
            _item, record = _load(self, client, table_id, qualify)
            owned = record.get("continuation")
            valid_phase = type(owned) is dict and (
                (owned.get("schema_version") == 1 and owned.get("epoch") == 1)
                or (
                    owned.get("schema_version") in (2, 3)
                    and (
                        (
                            owned.get("invocation_state") == "ready"
                            and self._completion_attempted_token is None
                        )
                        or (
                            owned.get("invocation_state") == "invoking"
                            and self._completion_attempted_token == token
                        )
                    )
                )
            )
            if (
                not valid_phase
                or owned.get("token") != token
                or (owned.get("status") != "consumed-ready")
                or (_canonical(owned) != receipt[1])
                or (_canonical(record["task_checkpoint_ref"]) != receipt[2])
            ):
                raise ValueError("original continuation ownership or content changed")
            if owned["schema_version"] == 3:
                original, binding = _original(self, record, table_id)

                def current():
                    _approved_read(policy)
                    qualify()
                    _item, now = _load(self, client, table_id, qualify)
                    if _canonical(now) != _canonical(record):
                        raise record_changed("original continuation selection changed", RUN_RECORD_CHANGED)

                raw = _read_object(self, binding, _ref(record["task_checkpoint_ref"]), "tasks", current)
                stored = json.loads(raw, object_pairs_hook=_pairs)
                if _canonical(stored).encode() != raw:
                    raise ValueError("original completion TASK bytes changed")
                expected_snapshot = _completed_task_snapshot(stored, original.operation_digest)
                if (
                    _sdk_snapshot_sha256(expected_snapshot) != owned["sdk_snapshot_sha256"]
                    or _sdk_snapshot_sha256(sdk_snapshot) != owned["sdk_snapshot_sha256"]
                ):
                    raise ValueError("original continuation ownership or content changed")
                current()
            else:
                if _canonical(sdk_snapshot) != _canonical(owned["sdk_snapshot"]):
                    raise ValueError("original continuation ownership or content changed")
                _original(self, record, table_id)
            qualify()
            qualify_original()
            _approved_read(policy)

    def begin_completion_invocation(self, token, *, policy, qualify_original, sdk_snapshot):
        """Persist one unique start attempt; never recover an ambiguous invocation."""
        from .producer_snapshot import _approved_read

        self.qualify_completion(
            token, policy=policy, qualify_original=qualify_original, sdk_snapshot=sdk_snapshot
        )
        with self._completion_invocation_lock:
            if self._completion_attempted_token is not None:
                raise ValueError("original completion invocation already attempted")
            receipt = self._completion_receipt
            owned = json.loads(receipt[1])
            if owned["schema_version"] == 1:
                # Preserve locally owned legacy behavior, without cold takeover.
                object.__setattr__(self, "_completion_attempted_token", token)
                return
            with _native(self) as (client, table_id, qualify, _sts):
                item, record = _load(self, client, table_id, qualify)
                owned = record.get("continuation")
                if (
                    owned is None
                    or _canonical(owned) != receipt[1]
                    or _canonical(record["task_checkpoint_ref"]) != receipt[2]
                    or owned.get("invocation_state") != "ready"
                ):
                    raise ValueError("original continuation ownership or content changed")

                def current():
                    _approved_read(policy)
                    qualify()
                    qualify_original()

                attempt = uuid.uuid4().hex
                if attempt == token:
                    raise ValueError("original invocation attempt identity differs")
                invoking = {**owned, "epoch": 2, "invocation_state": "invoking", "invocation_token": attempt}
                after = {**record, "continuation": invoking}
                current()
                # Spend locally before an attempted write. Unknown CAS outcomes
                # cannot repeat; only this unique complete after-item resolves it.
                object.__setattr__(self, "_completion_attempted_token", token)
                _replace_record(self, client, table_id, current, item, after, policy=policy)
                current()
                confirmed = _load(self, client, table_id, qualify)[1]
                if confirmed != after:
                    raise ValueError("original invocation start unresolved")
                current()
                object.__setattr__(
                    self,
                    "_completion_receipt",
                    (token, _canonical(invoking), _canonical(record["task_checkpoint_ref"])),
                )

    def has_task_checkpoint(self, *, policy):
        from .producer_snapshot import _approved_read

        _approved_read(policy)
        with _native(self) as (client, table_id, qualify, _sts):
            _item, record = _load(self, client, table_id, qualify)
            _approved_read(policy)
            return "task_checkpoint_ref" in record

    def has_retained_report(self, *, policy):
        """Return qualified presence DATA of a retained report pointer (the original's or a replacement's)."""
        from .producer_snapshot import _approved_read

        _approved_read(policy)
        with _native(self) as (client, table_id, qualify, _sts):
            _item, record = _load(self, client, table_id, qualify)
            _approved_read(policy)
            return "report_checkpoint_ref" in record or _RESUMED_REPORT_FIELD in record

    def has_prepared_checkpoint(self, *, policy):
        """Return qualified presence DATA without granting restoration authority."""
        from .producer_snapshot import _approved_read

        _approved_read(policy)
        with _native(self) as (client, table_id, qualify, _sts):
            _item, record = _load(self, client, table_id, qualify)
            _approved_read(policy)
            return "prepared_checkpoint_ref" in record

    def restore_sft_store(self, parents, original, binding, *, policy):
        """Read-only original restoration into the exact freshly verified parents."""
        from .normal_run_parents import _qualified_reads
        from .producer_snapshot import _approved_read
        from .sft_binding import build_sft_store

        _approved_read(policy)
        with _native(self) as (client, table_id, qualify, _sts):
            _item, record = _load(self, client, table_id, qualify)
            fresh, original_binding = _original(self, record, table_id, original.native_job_name)
            if fresh != original or original_binding != binding:
                raise ValueError("normal original restoration binding changed")
            _acquired, models, _cases, prepared = parents
            if any(store.runtime_session_id != binding.runtime_session_id for store in parents):
                raise ValueError("normal original parent graph correlation differs")
            sft = build_sft_store(
                models, prepared, campaign_state=_Journal(self, table_id), execution_binding=binding
            )

            def current_check(remote):
                _approved_read(policy)
                if remote:
                    qualify()
                    if _load(self, client, table_id, qualify)[1] != record:
                        raise record_changed("normal original restoration row changed", RUN_RECORD_CHANGED)

            current = _record_recheck(current_check)

            with _qualified_reads(
                (
                    *_model_read_adapters(models, original.intent.source_refs[0]),
                    prepared._adapter,
                    sft._adapter,
                ),
                self.catalog.region,
                current,
            ):
                result = sft.resume_original(original.native_job_name, *original.intent.source_refs)
                alias = "sft_" + original.operation_digest[:32]
                handle = sft._records.get(alias)
                if (
                    handle is None
                    or handle.extra.get("restoration_status") != "verified"
                    or result.get("request_ref") != alias
                    or (result.get("job_name") != original.native_job_name)
                    or (result.get("status") in ("resume_unresolved", "UNKNOWN"))
                ):
                    # A verified FAILED or STOPPED original is restored as such: the caller reports it and continues
                    # nothing (no checkpoint exists).
                    raise ValueError("normal original SFT restoration remains unresolved")
                current()
            return (sft, result)

    def _report_retention_applicable(self) -> bool:
        self._check()
        if type(self._enrollment_confirmed) is not bool:
            raise ValueError("normal enrollment applicability DATA differs")
        return self._enrollment_confirmed

    def enroll(self, private_store_correlation: str, admitted_prompt: str, history: list | None) -> None:
        self._check()
        if (
            type(private_store_correlation) is not str
            or str(uuid.UUID(private_store_correlation)) != private_store_correlation
        ):
            raise ValueError("normal run original graph correlation invalid")
        if type(admitted_prompt) is not str or not admitted_prompt.strip():
            raise ValueError("normal run admitted input missing")
        if history is not None and type(history) is not list:
            raise ValueError("normal run history DATA invalid")
        with _native(self) as (client, table_id, qualify, _sts):
            record = {
                "schema_version": 1,
                "logical_run_id": self.run_id,
                "session_sha256": self.session_hash,
                "scope": asdict(self.catalog),
                "table_id": table_id,
                "private_store_correlation": private_store_correlation,
                "admitted_prompt": admitted_prompt,
                "history": [] if history is None else history,
                "stage": "enrolled",
            }
            raw = _canonical(record)
            item = {
                "job_name": {"S": "PROMPT_RUN#" + self.run_id},
                "record": {"S": raw},
                "record_sha256": {"S": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
            }
            qualify()
            try:
                client.put_item(
                    TableName=self.catalog.jobs_table,
                    Item=item,
                    ConditionExpression="attribute_not_exists(job_name)",
                )
            except Exception as error:
                qualify()
                restored = client.get_item(
                    TableName=self.catalog.jobs_table, Key={"job_name": item["job_name"]}, ConsistentRead=True
                )
                qualify()
                if restored.get("Item") != item:
                    raise ValueError("normal run enrollment outcome unresolved") from error
            else:
                qualify()
        # Applicability DATA is minted only after the existing native acknowledgement
        # or exact readback succeeds; it grants no permission to any later write.
        object.__setattr__(self, "_enrollment_confirmed", True)

    def bind_sft_store(
        self,
        store,
        *,
        current_qualification=None,
        original_qualification=None,
        approval_policy_getter=None,
    ) -> None:
        """Bind the existing empty graph; absent SFT selectors remain unadmitted."""
        from .sft_state import SessionSftStore, SftConfigurationMissingError, validate_catalog

        self._check()
        if type(store) is not SessionSftStore:
            raise ValueError("normal run original SFT store required")
        if current_qualification is not None and (not callable(current_qualification)):
            raise ValueError("normal parent current qualifier invalid")
        if original_qualification is not None and (not callable(original_qualification)):
            raise ValueError("normal parent original qualifier invalid")
        try:
            validate_catalog(store._catalog)
        except SftConfigurationMissingError:
            return
        if store._catalog != _selected_sft_catalog():
            raise ValueError("normal run current SFT selection differs")
        with _native(self) as (client, table_id, qualify, _sts):
            _item, record = _load(self, client, table_id, qualify)
        if record["private_store_correlation"] != store.runtime_session_id:
            raise ValueError("normal run original SFT graph differs")
        journal = _Journal(self, table_id)
        binding = journal.bind_execution(
            campaign_id=str(uuid.UUID(hex=self.run_id)), runtime_session_id=store.runtime_session_id
        )

        def before_submit(model_ref, preparation_ref, recipe_id, inputs):
            self._check()
            if store._catalog != _selected_sft_catalog():
                raise ValueError("normal run SFT selection changed")
            from ..backends.adapters.coding_preparation import RecordRecheck
            from .normal_run_parents import publish_parent_graph

            parent_ref = None
            if self._causal_call is not None:
                call = self._captured_sft_call(model_ref, preparation_ref)
                if _effective_sft_inputs(call["tool_use"]["input"])["recipe_id"] != recipe_id:
                    raise ValueError("captured original SFT recipe changed")
                if current_qualification is None or original_qualification is None:
                    raise ValueError("normal captured parent qualifiers missing")
                if original_qualification() is not None:
                    raise ValueError("normal original qualifier returned permission DATA")
                with _native(self) as (parent_client, parent_table_id, parent_qualify, _parent_sts):
                    if parent_table_id != table_id:
                        raise ValueError("normal parent table incarnation changed")
                    _item, parent_record = _load(self, parent_client, parent_table_id, parent_qualify)

                    def full_recheck():
                        parent_qualify()
                        if store._catalog != _selected_sft_catalog():
                            raise record_changed("normal parent SFT selection changed", SFT_CATALOG_CHANGED)
                        _item, now = _load(self, parent_client, parent_table_id, parent_qualify)
                        if now != parent_record:
                            raise record_changed(
                                "normal original row changed during parent capture", RUN_RECORD_CHANGED
                            )
                        if current_qualification() is not None:
                            raise ValueError("normal parent qualifier returned permission DATA")
                        parent_qualify()

                    def local_recheck():
                        self._check()
                        if store._catalog != _selected_sft_catalog():
                            raise record_changed("normal parent SFT selection changed", SFT_CATALOG_CHANGED)
                        if current_qualification() is not None:
                            raise ValueError("normal parent qualifier returned permission DATA")

                    current = RecordRecheck(full_recheck, local_recheck, _PARENT_RECORD_RECHECK_SECONDS)
                    current()
                    parent_raw = publish_parent_graph(
                        store, model_ref, preparation_ref, run_id=self.run_id, qualify=current
                    )
                    if original_qualification() is not None:
                        raise ValueError("normal original qualifier returned permission DATA")
                    current()
                    parent_ref = _put_object(self, binding, parent_raw, "parents", current)
                    current()
            intent = build_sft_intent(
                binding,
                store._catalog,
                inputs,
                recipe_id=recipe_id,
                source_refs=(model_ref, preparation_ref),
                causal_parent_ref=parent_ref,
            )
            name, arn = _native_identity(intent)
            association = {
                "model_ref": model_ref,
                "preparation_ref": preparation_ref,
                "intent": _intent_fields(intent),
                "operation_digest": operation_digest(intent),
                "native_job_name": name,
                "native_job_arn": arn,
            }
            _associate(self, table_id, association)
            if parent_ref is not None:
                snapshot = self._captured_sft_snapshot(call["pending_tool_use_id"], call["causal_prefix"])
                if snapshot is not None:
                    _publish_prepared_checkpoint(
                        self,
                        binding,
                        table_id,
                        association,
                        call,
                        snapshot,
                        parent_ref,
                        policy_getter=approval_policy_getter,
                        current_qualification=current_qualification,
                        original_qualification=original_qualification,
                    )
            _publish_normal_wake(self, table_id, association)
            return parent_ref

        store.bind_durable(journal, binding, before_submit=before_submit)


def prepare_normal_run(
    run_id: str, session_hash: str | None, *, transport_session_id: str | None = None
) -> NormalRunContext | None:
    """Pure configuration validation; both absent role fields retain legacy."""
    catalog = _catalog()
    if catalog is None:
        return None
    _refs(run_id, session_hash)
    return NormalRunContext(catalog, run_id, session_hash, transport_session_id=transport_session_id)


@contextmanager
def activate_normal_run(context: NormalRunContext | None):
    if context is not None:
        if type(context) is not NormalRunContext:
            raise ValueError("normal run context DATA invalid")
        context._check()
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def current_normal_run() -> NormalRunContext | None:
    context = _CURRENT.get()
    if context is not None:
        if type(context) is not NormalRunContext:
            raise ValueError("normal run context DATA invalid")
        context._check()
    return context


def _warn_normal_model_progress(phase: str, state: str, verified_payloads: int, cause: Any = ()) -> None:
    """Optional closed loading diagnostics; no native observation or authority.

    A refused pass always names its cause (closed exception class, native code, innermost amlas site, root class of an
    explicit `raise ... from`), the closed defaults when the cause given is not closed. A pass that its caller's own
    run-record recheck ended (`model_verification_interrupted`, issue 112) names the closed reason and site; it is not
    a refusal. Every other phase takes no cause. An exception message is never rendered.
    """
    try:
        if (
            type(phase) is not str
            or phase
            not in {
                "model_native_observation",
                "model_native_wait",
                "model_artifact_verification",
                "model_artifact_ready",
                "model_artifact_refused",
                "model_verification_interrupted",
            }
            or type(state) is not str
            or state not in {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED", "UNKNOWN"}
            or type(verified_payloads) is not int
            or not 0 <= verified_payloads <= 9
        ):
            return
        message = "normal_model_progress run_id=%s phase=%s state=%s verified_payloads=%s"
        fields: tuple[str, ...] = ()
        if phase == "model_artifact_refused":
            message += " exception_class=%s native_code=%s site=%s cause_class=%s"
            fields = closed_cause(cause)
        elif phase == "model_verification_interrupted":
            interruption = closed_interruption(cause)
            if interruption is None:
                return
            message += " reason=%s site=%s"
            fields = interruption
        elif cause != ():
            return
        context = current_normal_run()
        if context is not None:
            logging.getLogger(__name__).warning(
                message,
                context.run_id,
                phase,
                state,
                verified_payloads,
                *fields,
            )
    except Exception:
        return


def _warn_normal_handoff(phase: str, kind: str, state: str, ready: bool) -> None:
    """Optional closed handoff diagnostics; no observation or authority."""
    try:
        if (
            type(phase) is not str
            or phase
            not in {
                "OBSERVING",
                "OBSERVED",
                "READY",
                "ADMITTING",
                "ADMITTED",
                "ADMISSION_REFUSED",
                "BATCH_COMMITTED",
            }
            or type(kind) is not str
            or kind not in {"MODEL", "DATA", "CASE", "PREP", "BATCH"}
            or type(state) is not str
            or state
            not in {
                "PENDING",
                "RUNNING",
                "SUCCEEDED",
                "FAILED",
                "STOPPED",
                "UNKNOWN",
                "OBSERVATION_FAILED",
                "VERIFICATION_FAILED",
                "SUBMISSION_REJECTED",
            }
            or type(ready) is not bool
            or ready is not (phase == "READY")
        ):
            return
        context = current_normal_run()
        if context is not None:
            logging.getLogger(__name__).warning(
                "normal_handoff run_id=%s phase=%s kind=%s state=%s ready=%s",
                context.run_id,
                phase,
                kind,
                state,
                ready,
            )
    except Exception:
        return


def _report_reference(record, pointer="report_checkpoint_ref"):
    """Closed optional report DATA descriptor; no native permission is encoded."""
    from ..backends.types import DatasetArtifactRef

    value = record.get(pointer)
    if (
        type(value) is not dict
        or set(value) != {"uri", "version_id", "sha256", "byte_count", "format"}
        or type(value["uri"]) is not str
        or not value["uri"].startswith("s3://")
        or value["format"] != "json"
        or type(value["byte_count"]) is not int
        or not 0 < value["byte_count"] <= _MAX_BYTES
        or type(value["sha256"]) is not str
        or re.fullmatch("[a-f0-9]{64}", value["sha256"]) is None
        or type(value["version_id"]) is not str
        or value["version_id"].lower() in ("", "null")
        or len(value["version_id"]) > 1024
        or any(ord(char) < 33 for char in value["version_id"])
    ):
        raise ValueError("normal report descriptor differs")
    return DatasetArtifactRef(**value)


def _report_bytes(context, report):
    """Accept only the exact typed terminal conversation, never ML authority."""
    if (
        type(report) is not dict
        or set(report)
        != {
            "event",
            "status",
            "response",
            "worker_finished",
            "mode",
            "run_id",
            "response_kind",
            "state_scope",
            "durable_resume",
            "approval_policy_source",
            "lifecycle_verified",
            "quality_success",
        }
        or report["event"] != "run_completed"
        or report["status"] != "response_received"
        or report["worker_finished"] is not True
        or report["mode"] != "prompt"
        or report["run_id"] != context.run_id
        or report["response_kind"] != "conversation"
        or report["state_scope"] != "single_invocation"
        or report["durable_resume"] is not False
        or report["lifecycle_verified"] is not False
        or report["quality_success"] is not False
        or report["approval_policy_source"] not in {"default_noninteractive", "deployment_allowlist"}
    ):
        raise ValueError("normal terminal conversation report differs")
    response = report["response"]
    if (
        type(response) is not dict
        or set(response) != {"message", "action", "data", "suggestions"}
        or type(response["message"]) is not str
        or type(response["action"]) is not str
        or type(response["data"]) is not dict
        or type(response["suggestions"]) is not list
        or any(type(item) is not str for item in response["suggestions"])
    ):
        raise ValueError("normal typed conversation response differs")
    raw = _canonical(report).encode("utf-8")
    if not 0 < len(raw) <= _MAX_BYTES:
        raise ValueError("normal report byte bound exceeded")
    return raw


def _report_binding(context, record, table_id):
    # This existing binding only verifies native scope; it enrolls/claims nothing.
    return _Journal(context, table_id).bind_execution(
        campaign_id=str(uuid.UUID(hex=context.run_id)),
        runtime_session_id=record["private_store_correlation"],
    )


def _replacement_started(record):
    """DATA only: an SFT row whose completion invocation a replacement has started (conditional write won)."""
    continuation = record.get("continuation")
    return (
        record.get("stage") == "sft"
        and type(continuation) is dict
        and continuation.get("schema_version") in (2, 3)
        and continuation.get("status") == "consumed-ready"
        and continuation.get("invocation_state") == "invoking"
    )


def _report_fields(record, expected):
    """Admit the optional report pointers; the resumed one needs a started replacement invocation."""
    for pointer in ("report_checkpoint_ref", _RESUMED_REPORT_FIELD):
        if pointer in record:
            expected.add(pointer)
            _report_reference(record, pointer)
    if _RESUMED_REPORT_FIELD in record and not _replacement_started(record):
        raise ValueError("normal resumed report without a started replacement invocation")


def _verified_report(context, binding, record, qualify, pointer="report_checkpoint_ref"):
    ref = _report_reference(record, pointer)
    raw = _read_object(context, binding, ref, "reports", qualify)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    if _report_bytes(context, value) != raw:
        raise ValueError("normal report canonical bytes differ")
    return value


def retain_normal_prompt_report(run_id, session_hash, report, *, resumed=False):
    """Append one report pointer under existing native ownership; no new claim.

    The original worker's report is the single-assignment `report_checkpoint_ref`. A replacement
    that started the completion invocation passes `resumed=True` and publishes its final report as
    the separate single-assignment `resumed_report_checkpoint_ref`; the original pointer is neither
    changed nor weakened, and a different second report on either pointer is refused.
    """
    if type(resumed) is not bool:
        raise TypeError("normal report resumed flag must be source-owned bool DATA")
    pointer = _RESUMED_REPORT_FIELD if resumed else "report_checkpoint_ref"
    if _catalog() is None:
        return False
    context = prepare_normal_run(run_id, session_hash)
    if context is None:
        return False
    raw = _report_bytes(context, report)
    with _native(context) as (client, table_id, qualify, _sts):
        item, record = _load(context, client, table_id, qualify)
        if resumed and not _replacement_started(record):
            raise ValueError("normal resumed report requires a started replacement invocation")
        binding = _report_binding(context, record, table_id)
        if pointer in record:
            if _report_bytes(context, _verified_report(context, binding, record, qualify, pointer)) != raw:
                raise ValueError("normal report publication conflicts with original")
            return True
        # Object is immutable and independently read back before pointer publication.
        ref = _put_object(context, binding, raw, "reports", qualify)
        updated = {**record, pointer: asdict(ref)}
        after_raw = _canonical(updated)
        after = {
            **item,
            "record": {"S": after_raw},
            "record_sha256": {"S": hashlib.sha256(after_raw.encode()).hexdigest()},
        }
        if len(after_raw.encode()) > _MAX_BYTES:
            raise ValueError("normal report pointer record exceeds byte bound")
        qualify()
        try:
            client.update_item(
                TableName=context.catalog.jobs_table,
                Key={"job_name": item["job_name"]},
                ConditionExpression="#record = :prior AND record_sha256 = :prior_sha",
                UpdateExpression="SET #record = :body, record_sha256 = :sha",
                ExpressionAttributeNames={"#record": "record"},
                ExpressionAttributeValues={
                    ":prior": item["record"],
                    ":prior_sha": item["record_sha256"],
                    ":body": after["record"],
                    ":sha": after["record_sha256"],
                },
            )
        except Exception:
            # An ambiguous/conflicting CAS is observed once, never retried.
            winner_item, winner = _load(context, client, table_id, qualify)
            baseline = {name: value for name, value in winner.items() if name != pointer}
            if baseline != record or pointer not in winner:
                raise ValueError("normal report pointer outcome unresolved") from None
            if _report_bytes(context, _verified_report(context, binding, winner, qualify, pointer)) != raw:
                raise ValueError("normal report pointer publication conflicts") from None
            # Keep the original winning descriptor/version; never overwrite it.
            if winner_item["job_name"] != item["job_name"]:
                raise ValueError("normal report winner identity differs") from None
            return True
        qualify()
        final_item, final_record = _load(context, client, table_id, qualify)
        if final_item != after or final_record != updated:
            raise ValueError("normal report pointer readback differs")
        _verified_report(context, binding, final_record, qualify, pointer)
        return True


def read_normal_prompt_report(run_id, session_hash):
    """Passive exact native read; None alone means unselected legacy storage.

    A resumed (final) report supersedes the original's once its pointer exists; a damaged resumed
    pointer refuses and never falls back to the original's report.
    """
    if _catalog() is None:
        return None
    context = prepare_normal_run(run_id, session_hash)
    if context is None:
        return None
    with _native(context) as (client, table_id, qualify, _sts):
        item, record = _load(context, client, table_id, qualify)
        pointer = _RESUMED_REPORT_FIELD if _RESUMED_REPORT_FIELD in record else "report_checkpoint_ref"
        if pointer not in record:
            raise ValueError("normal retained report unavailable")
        binding = _report_binding(context, record, table_id)
        value = _verified_report(context, binding, record, qualify, pointer)
        if _load(context, client, table_id, qualify)[0] != item:
            raise ValueError("normal report original record changed")
        qualify()
        return value


@contextmanager
def _native(context: NormalRunContext):
    import boto3
    from botocore.config import Config

    context._check()
    scope = context.catalog
    session = boto3.Session(region_name=scope.region)
    options = Config(connect_timeout=5, read_timeout=20, retries={"total_max_attempts": 1})
    sts = session.client("sts", region_name=scope.region, config=options)
    ddb = session.client("dynamodb", region_name=scope.region, config=options)
    table_id = None

    def qualify():
        nonlocal table_id
        context._check()
        identity = sts.get_caller_identity()
        role_name = scope.execution_role_arn.rsplit("/", 1)[1]
        arn = identity.get("Arn", "")
        user_id = identity.get("UserId")
        prefix = "arn:aws:sts::" + scope.account_id + ":assumed-role/" + role_name + "/"
        if (
            identity.get("Account") != scope.account_id
            or type(arn) is not str
            or not arn.startswith(prefix)
            or not arn[len(prefix) :]
            or type(user_id) is not str
            or user_id.split(":", 1)[0] != scope.execution_role_id
        ):
            raise ValueError("normal run native runtime-role binding refused")
        table = ddb.describe_table(TableName=scope.jobs_table).get("Table", {})
        incarnation = table.get("TableId")
        expected_arn = (
            "arn:aws:dynamodb:" + scope.region + ":" + scope.account_id + ":table/" + scope.jobs_table
        )
        if (
            table.get("TableName") != scope.jobs_table
            or table.get("TableArn") != expected_arn
            or table.get("TableStatus") != "ACTIVE"
            or table.get("KeySchema") != [{"AttributeName": "job_name", "KeyType": "HASH"}]
            or {
                entry["AttributeName"]: entry["AttributeType"]
                for entry in table.get("AttributeDefinitions", [])
            }.get("job_name")
            != "S"
            or type(incarnation) is not str
            or str(uuid.UUID(incarnation)) != incarnation
            or (table_id is not None and table_id != incarnation)
        ):
            raise ValueError("normal run native table binding refused")
        table_id = incarnation
        context._check()

    try:
        qualify()
        yield ddb, table_id, qualify, sts
    finally:
        for client in (sts, ddb):
            close = getattr(client, "close", None)
            if callable(close):
                close()


@lru_cache(maxsize=4)
def _validate_record_bytes(raw: str) -> None:
    """Reuse only deterministic DATA validation, never native or contextual authority."""
    record = json.loads(raw, object_pairs_hook=_pairs)
    if type(record) is not dict or _canonical(record) != raw:
        raise ValueError("normal run canonical record differs")
    if "continuation" in record:
        _validate_continuation_data(record)


def _record_data(raw: str) -> dict:
    if type(raw) is not str or len(raw.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("normal run bounded record differs")
    _validate_record_bytes(raw)
    # Parse afresh so no mutable record is retained or shared by the cache.
    return json.loads(raw)


@dataclass(frozen=True, slots=True)
class _RecordProjection:
    """Immutable exact-byte DATA view; no native or contextual grant is retained."""

    raw: str
    sft: str | None
    private_store_correlation: str
    logical_run_id: str
    session_sha256: str
    scope: str
    table_id: str
    prepared_checkpoint_ref: str | None


@lru_cache(maxsize=4)
def _record_projection(raw: str) -> _RecordProjection:
    """Reuse only pure row DATA; every loader still qualifies and reads afresh."""
    record = _record_data(raw)
    expected = {
        "schema_version",
        "logical_run_id",
        "session_sha256",
        "scope",
        "table_id",
        "private_store_correlation",
        "admitted_prompt",
        "history",
        "stage",
    }
    _report_fields(record, expected)
    if record.get("stage") == "sft":
        expected.add("sft")
        if "prepared_checkpoint_ref" in record:
            expected.add("prepared_checkpoint_ref")
        if "task_checkpoint_ref" in record:
            expected.add("task_checkpoint_ref")
        if "continuation" in record:
            expected.add("continuation")
    if (
        set(record) != expected
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or type(record["logical_run_id"]) is not str
        or type(record["session_sha256"]) is not str
        or type(record["table_id"]) is not str
        or record["stage"] not in ("enrolled", "sft")
        or type(record["admitted_prompt"]) is not str
        or not record["admitted_prompt"].strip()
        or type(record["history"]) is not list
        or type(record["private_store_correlation"]) is not str
        or str(uuid.UUID(record["private_store_correlation"])) != record["private_store_correlation"]
    ):
        raise ValueError("normal run owned record differs")
    return _RecordProjection(
        raw=raw,
        sft=_canonical(record["sft"]) if "sft" in record else None,
        private_store_correlation=record["private_store_correlation"],
        logical_run_id=record["logical_run_id"],
        session_sha256=record["session_sha256"],
        scope=_canonical(record["scope"]),
        table_id=record["table_id"],
        prepared_checkpoint_ref=(
            _canonical(record["prepared_checkpoint_ref"]) if "prepared_checkpoint_ref" in record else None
        ),
    )


def _load(context, client, table_id, qualify, *, _projection=False):
    qualify()
    item = client.get_item(
        TableName=context.catalog.jobs_table,
        Key={"job_name": {"S": "PROMPT_RUN#" + context.run_id}},
        ConsistentRead=True,
    ).get("Item")
    qualify()
    if type(item) is not dict or set(item) != {"job_name", "record", "record_sha256"}:
        raise ValueError("normal run record unavailable")
    if item["job_name"] != {"S": "PROMPT_RUN#" + context.run_id} or any(
        type(item[key]) is not dict or set(item[key]) != {"S"} for key in ("record", "record_sha256")
    ):
        raise ValueError("normal run scalar envelope differs")
    raw, digest = item["record"]["S"], item["record_sha256"]["S"]
    if (
        type(raw) is not str
        or len(raw.encode("utf-8")) > _MAX_BYTES
        or type(digest) is not str
        or not re.fullmatch("[a-f0-9]{64}", digest)
        or hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest
    ):
        raise ValueError("normal run bounded record differs")
    if _projection:
        record = _record_projection(raw)
        if (
            record.logical_run_id != context.run_id
            or record.session_sha256 != context.session_hash
            or record.scope != _canonical(asdict(context.catalog))
            or record.table_id != table_id
        ):
            raise ValueError("normal run owned record differs")
        if record.prepared_checkpoint_ref is not None:
            _prepared_ref(
                context,
                {
                    "private_store_correlation": record.private_store_correlation,
                    "prepared_checkpoint_ref": json.loads(record.prepared_checkpoint_ref),
                },
            )
        return item, record
    record = _record_data(raw)
    expected = {
        "schema_version",
        "logical_run_id",
        "session_sha256",
        "scope",
        "table_id",
        "private_store_correlation",
        "admitted_prompt",
        "history",
        "stage",
    }
    _report_fields(record, expected)
    if record.get("stage") == "sft":
        expected.add("sft")
        if "prepared_checkpoint_ref" in record:
            expected.add("prepared_checkpoint_ref")
        if "task_checkpoint_ref" in record:
            expected.add("task_checkpoint_ref")
        if "continuation" in record:
            expected.add("continuation")
    if (
        set(record) != expected
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["logical_run_id"] != context.run_id
        or record["session_sha256"] != context.session_hash
        or record["scope"] != asdict(context.catalog)
        or record["table_id"] != table_id
        or record["stage"] not in ("enrolled", "sft")
        or type(record["admitted_prompt"]) is not str
        or not record["admitted_prompt"].strip()
        or type(record["history"]) is not list
        or type(record["private_store_correlation"]) is not str
        or str(uuid.UUID(record["private_store_correlation"])) != record["private_store_correlation"]
    ):
        raise ValueError("normal run owned record differs")
    if "prepared_checkpoint_ref" in record:
        _prepared_ref(context, record)
    return item, record


def _validate_continuation_data(record):
    """Validate bounded ready DATA; no original qualification or effect authority."""
    from .producer_snapshot import _ref

    continuation = record["continuation"]
    version = continuation.get("schema_version") if type(continuation) is dict else None
    fields = {"schema_version", "epoch", "token", "status", "operation_digest"}
    fields |= {"sdk_snapshot_sha256"} if version == 3 else {"sdk_snapshot"}
    if version in (2, 3):
        fields |= {"invocation_state", "invocation_token"}
    if (
        type(continuation) is not dict
        or set(continuation) != fields
        or type(continuation["schema_version"]) is not int
        or continuation["schema_version"] not in (1, 2, 3)
        or type(continuation["epoch"]) is not int
        or continuation["epoch"] < 1
        or type(continuation["token"]) is not str
        or not re.fullmatch("[a-f0-9]{32}", continuation["token"])
        or continuation["status"] != "consumed-ready"
        or type(continuation["operation_digest"]) is not str
        or not re.fullmatch("[a-f0-9]{64}", continuation["operation_digest"])
        or type(record.get("sft")) is not dict
        or continuation["operation_digest"] != record["sft"].get("operation_digest")
        or "task_checkpoint_ref" not in record
    ):
        raise ValueError("normal continuation DATA differs")
    if version in (2, 3):
        state, attempt = continuation["invocation_state"], continuation["invocation_token"]
        valid = (state == "ready" and continuation["epoch"] == 1 and attempt is None) or (
            state == "invoking"
            and continuation["epoch"] == 2
            and type(attempt) is str
            and re.fullmatch("[a-f0-9]{32}", attempt)
            and attempt != continuation["token"]
        )
        if not valid:
            raise ValueError("normal continuation invocation DATA differs")
    task_ref = _ref(record["task_checkpoint_ref"])
    if task_ref.format != "json" or not 0 < task_ref.byte_count <= _MAX_BYTES:
        raise ValueError("normal continuation TASK descriptor differs")
    if version == 3:
        digest = continuation["sdk_snapshot_sha256"]
        if type(digest) is not str or not re.fullmatch("[a-f0-9]{64}", digest):
            raise ValueError("normal continuation SDK digest differs")
    else:
        _captured_snapshot_messages(continuation["sdk_snapshot"])


def _publish_normal_wake(context, original_table_id, association):
    """Confirm original normal routing DATA before adapter staging or Create."""
    if context.transport_session_id is None:
        return
    context._check()
    expected_association = json.loads(_canonical(association), object_pairs_hook=_pairs)
    with _native(context) as (client, table_id, qualify, _sts):
        if table_id != original_table_id:
            raise ValueError("changed normal publication table")

        def current():
            qualify()
            _item, record = _load(context, client, table_id, qualify)
            if record["stage"] != "sft" or record["sft"] != expected_association:
                raise ValueError("changed normal publication original")

        current()
        name = expected_association["native_job_name"]
        body = {
            "schema_version": 1,
            "record_kind": "normal_prompt_wake_job_v1",
            "logical_run_id": context.run_id,
            "transport_session_id": context.transport_session_id,
            "account_id": context.catalog.account_id,
            "region": context.catalog.region,
            "deployment_id": context.catalog.deployment_id,
            "table_id": table_id,
            "kind": "Training",
            "job_name": name,
            "job_arn": expected_association["native_job_arn"],
            "operation_digest": expected_association["operation_digest"],
        }
        _plain(body)
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        if len(raw.encode("utf-8")) > 8192:
            raise ValueError("normal routing budget exceeded")
        expected = {
            "job_name": {"S": "PROMPT_WAKE_JOB#Training#" + name},
            "record": {"S": raw},
            "record_sha256": {"S": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
        }
        current()
        try:
            client.put_item(
                TableName=context.catalog.jobs_table,
                Item=expected,
                ConditionExpression="attribute_not_exists(job_name)",
            )
        except PermissionError:
            raise
        except Exception as error:
            response = getattr(error, "response", None)
            details = response.get("Error") if type(response) is dict else None
            code = details.get("Code") if type(details) is dict else None
            if code is not None and code not in (
                "ConditionalCheckFailedException",
                "InternalServerError",
                "ServiceUnavailable",
                "ServiceUnavailableException",
                "RequestTimeout",
                "RequestTimeoutException",
            ):
                raise
        current()
        confirmed = client.get_item(
            TableName=context.catalog.jobs_table,
            Key={"job_name": expected["job_name"]},
            ConsistentRead=True,
        )
        current()
        if type(confirmed) is not dict or confirmed.get("Item") != expected:
            raise ValueError("normal routing publication unresolved")
        current()


def _prepared_ref(context, record):
    from .producer_snapshot import _ref

    ref = _ref(record["prepared_checkpoint_ref"])
    prefix = (
        "s3://"
        + resolve_env_snapshot("session_bucket")
        + "/sessions/"
        + record["private_store_correlation"]
        + "/prompt-runs/"
        + context.run_id
        + "/prepared-tasks/"
    )
    if (
        ref.format != "json"
        or type(ref.byte_count) is not int
        or not 0 < ref.byte_count <= _MAX_BYTES
        or type(ref.sha256) is not str
        or not re.fullmatch("[a-f0-9]{64}", ref.sha256)
        or ref.uri != prefix + ref.sha256 + ".json"
        or type(ref.version_id) is not str
        or ref.version_id.lower() in ("", "null")
        or len(ref.version_id) > 1024
        or any(ord(char) < 33 for char in ref.version_id)
    ):
        raise ValueError("normal prepared checkpoint exact version/namespace differs")
    return ref


def _publish_prepared_checkpoint(
    context,
    binding,
    original_table_id,
    association,
    call,
    snapshot,
    parent_ref,
    *,
    policy_getter,
    current_qualification,
    original_qualification,
):
    from .producer_snapshot import _approved_read, _ref_wire

    if not callable(policy_getter):
        raise ValueError("prepared checkpoint requires current approval policy")
    if not callable(current_qualification) or not callable(original_qualification):
        raise ValueError("prepared checkpoint requires live original qualification")
    _approved_read(policy_getter())
    if current_qualification() is not None:
        raise ValueError("prepared checkpoint qualifier returned permission DATA")
    if original_qualification() is not None:
        raise ValueError("prepared checkpoint original parent qualification unavailable")
    with _native(context) as (client, table_id, qualify, _sts):
        if table_id != original_table_id:
            raise ValueError("prepared checkpoint original table incarnation differs")
        item, record = _load(context, client, table_id, qualify)
        if (
            record["stage"] != "sft"
            or record["sft"] != association
            or record["private_store_correlation"] != binding.runtime_session_id
        ):
            raise ValueError("prepared checkpoint original SFT association differs")
        after = None

        def current():
            _approved_read(policy_getter())
            qualify()
            _item, now = _load(context, client, table_id, qualify)
            if now != record and now != after:
                raise ValueError("prepared checkpoint original record changed")
            if current_qualification() is not None:
                raise ValueError("prepared checkpoint qualifier returned permission DATA")
            if original_qualification() is not None:
                raise ValueError("prepared checkpoint original parent qualification unavailable")
            _approved_read(policy_getter())
            context._check()

        current()
        if _canonical(_captured_snapshot_messages(snapshot)) != _canonical(call["causal_prefix"]):
            raise ValueError("prepared SDK snapshot rewrote the prospective prefix")
        task_schema_version = _prepared_task_wire_version(context, binding, record, current)
        prepared = {
            "schema_version": task_schema_version,
            "logical_run_id": context.run_id,
            "private_store_correlation": record["private_store_correlation"],
            "native_job_name": association["native_job_name"],
            "native_job_arn": association["native_job_arn"],
            "operation_digest": association["operation_digest"],
            "source_refs": [association["model_ref"], association["preparation_ref"]],
            "causal_parent_ref": _ref_wire(parent_ref),
            "sdk_snapshot": snapshot,
            "pending_tool_use_id": call["pending_tool_use_id"],
        }
        if task_schema_version == 1:
            prepared["causal_prefix"] = call["causal_prefix"]
        _task_envelope(prepared)
        raw = _canonical(prepared).encode("utf-8")
        if "prepared_checkpoint_ref" in record:
            if (
                _read_object(context, binding, _prepared_ref(context, record), "prepared-tasks", current)
                != raw
            ):
                raise ValueError("original prepared checkpoint already differs")
            current()
            return
        prepared_ref = _put_object(context, binding, raw, "prepared-tasks", current)
        after = {**record, "prepared_checkpoint_ref": _ref_wire(prepared_ref)}
        _replace_record(
            context,
            client,
            table_id,
            current,
            item,
            after,
            policy=policy_getter(),
        )
        current()
        _item, confirmed = _load(context, client, table_id, qualify)
        if confirmed != after:
            raise ValueError("prepared checkpoint confirmed record differs")
        record = after
        after = None
        current()


def _associate(context, original_table_id, association):
    with _native(context) as (client, table_id, qualify, _sts):
        if table_id != original_table_id:
            raise ValueError("normal run original table incarnation differs")
        item, record = _load(context, client, table_id, qualify)
        if record["stage"] == "sft":
            if record["sft"] != association:
                raise ValueError("normal run original SFT association differs")
            return
        raw = _canonical({**record, "stage": "sft", "sft": association})
        after = {
            **item,
            "record": {"S": raw},
            "record_sha256": {"S": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
        }
        qualify()
        try:
            client.update_item(
                TableName=context.catalog.jobs_table,
                Key={"job_name": item["job_name"]},
                ConditionExpression="#record = :prior AND record_sha256 = :prior_sha",
                UpdateExpression="SET #record = :body, record_sha256 = :sha",
                ExpressionAttributeNames={"#record": "record"},
                ExpressionAttributeValues={
                    ":prior": item["record"],
                    ":prior_sha": item["record_sha256"],
                    ":body": after["record"],
                    ":sha": after["record_sha256"],
                },
            )
        except Exception as error:
            restored, _record = _load(context, client, table_id, qualify)
            if restored != after:
                raise ValueError("normal run SFT association outcome unresolved") from error
        finally:
            qualify()


class _NativeIO:
    """Ephemeral exact native methods; every actual call obtains fresh qualification."""

    def __init__(self, client, qualify, methods):
        self.client, self.qualify, self.methods = client, qualify, methods

    @property
    def meta(self):
        self.qualify()
        return self.client.meta

    def __getattr__(self, name):
        if name not in self.methods:
            raise AttributeError(name)

        def call(**kwargs):
            self.qualify()
            try:
                result = getattr(self.client, name)(**kwargs)
            except BaseException:
                self.qualify()
                raise
            try:
                self.qualify()
            except BaseException:
                body = result.get("Body") if type(result) is dict else None
                if body is not None:
                    body.close()
                raise
            return result

        return call


_JOURNAL_METHODS = frozenset(
    {
        "bind_execution",
        "reserve_original_job",
        "mark_create_attempted",
        "attach_submission_snapshot",
        "record_submission_outcome",
        "load_original",
        "observe_original",
    }
)


class _Journal(DynamoCampaignStateBackend):
    """Existing backend with fresh clients and the original enrollment incarnation."""

    def __init__(self, context, table_id):
        self.context, self.table_id, self.catalog = context, table_id, context.catalog

    def __getattribute__(self, name):
        if name in _JOURNAL_METHODS:
            return lambda *args, **kwargs: self._call(name, *args, **kwargs)
        return super().__getattribute__(name)

    def _call(self, name, *args, **kwargs):
        if name == "observe_original":
            raise ValueError("normal pending observation requires full SFT restore validation")
        with _native(self.context) as (client, table_id, qualify, sts):
            if table_id != self.table_id:
                raise ValueError("normal run journal table incarnation differs")
            _item, record = _load(self.context, client, table_id, qualify)
            if name == "reserve_original_job":
                intent = args[1] if len(args) == 2 else kwargs["intent"]
                fields, association = _intent_fields(intent), record.get("sft", {})
                native_name, arn = _native_identity(intent)
                if (
                    record["stage"] != "sft"
                    or type(association) is not dict
                    or record["private_store_correlation"] != fields["binding"]["runtime_session_id"]
                    or fields["binding"]["campaign_id"] != str(uuid.UUID(hex=self.context.run_id))
                ):
                    raise ValueError("normal journal original association differs")
                if intent.operation == "train":
                    if (
                        association.get("intent") != fields
                        or association.get("operation_digest") != operation_digest(intent)
                        or (association.get("native_job_name"), association.get("native_job_arn"))
                        != (native_name, arn)
                    ):
                        raise ValueError("normal journal original association differs")
                elif intent.operation == "reload":
                    original, binding = _original(self.context, record, table_id)
                    checkpoint = intent.checkpoint
                    if (
                        intent.binding != binding
                        or intent.catalog != original.intent.catalog
                        or intent.inputs != original.intent.inputs
                        or intent.recipe_id != original.intent.recipe_id
                        or checkpoint is None
                        or checkpoint.producer_job_arn != original.native_job_arn
                        or checkpoint.runtime_session_id != binding.runtime_session_id
                        or checkpoint.recipe_id != original.intent.recipe_id
                        or checkpoint.source_commit != original.intent.catalog.source_commit
                    ):
                        raise ValueError("normal journal reload producer association differs")
                else:
                    raise ValueError("normal journal operation differs")

            record_raw = _canonical(record)
            readonly = name in {"bind_execution", "load_original"}
            observed_task = None

            def current():
                nonlocal observed_task
                now_raw = _load(self.context, client, table_id, qualify, _projection=True)[1].raw
                if now_raw == record_raw and observed_task is None:
                    return
                if readonly and "task_checkpoint_ref" not in record:
                    from .producer_snapshot import _ref

                    now = _record_data(now_raw)
                    task = now.get("task_checkpoint_ref")
                    if task is not None and _canonical(now) == _canonical(
                        {**record, "task_checkpoint_ref": task}
                    ):
                        ref = _ref(task)
                        prefix = (
                            "s3://"
                            + resolve_env_snapshot("session_bucket")
                            + "/sessions/"
                            + record["private_store_correlation"]
                            + "/prompt-runs/"
                            + self.context.run_id
                            + "/tasks/"
                        )
                        if (
                            record["stage"] == "sft"
                            and "prepared_checkpoint_ref" in record
                            and ref.format == "json"
                            and type(ref.byte_count) is int
                            and 0 < ref.byte_count <= _MAX_BYTES
                            and type(ref.sha256) is str
                            and re.fullmatch("[a-f0-9]{64}", ref.sha256)
                            and ref.uri == prefix + ref.sha256 + ".json"
                            and type(ref.version_id) is str
                            and ref.version_id.lower() not in ("", "null")
                            and len(ref.version_id) <= 1024
                            and all(ord(char) >= 33 for char in ref.version_id)
                        ):
                            task_raw = _canonical(task)
                            if observed_task is None or observed_task == task_raw:
                                observed_task = task_raw
                                return
                raise record_changed("normal journal original enrollment changed", RUN_RECORD_CHANGED)

            backend = DynamoCampaignStateBackend(
                self.catalog,
                dynamodb_client=_NativeIO(client, current, {"get_item", "put_item", "update_item"}),
                sts_client=sts,
            )
            try:
                return getattr(backend, name)(*args, **kwargs)
            finally:
                current()


def _selected_sft_catalog():
    from .sft_state import SftCatalog, validate_catalog

    catalog = SftCatalog(
        account_id=resolve_env_snapshot("aws_account_id"),
        region=resolve_env_snapshot("aws_region"),
        bucket=resolve_env_snapshot("session_bucket"),
        execution_role_arn=resolve_env_snapshot("sagemaker_execution_role_arn"),
        **{
            key: os.environ.get(name, "").strip()
            for key, name in (
                ("image_uri", "AMLAS_QWEN_SFT_IMAGE_URI"),
                ("source_commit", "AMLAS_QWEN_SFT_SOURCE_COMMIT"),
                ("admission_digest", "AMLAS_QWEN_SFT_ADMISSION_DIGEST"),
                ("deployment_id", "AMLAS_DEPLOYMENT_ID"),
                ("deployment_prefix", "AMLAS_DEPLOYMENT_PREFIX"),
            )
        },
    )
    validate_catalog(catalog)
    return catalog


def _pending(context, client, record, table_id, qualify, policy, resume):
    import boto3
    from botocore.config import Config

    from ..backends.adapters.qwen_sft import SageMakerQwenSftBackend
    from ..backends.types import JobState
    from .producer_snapshot import _approved_read

    catalog = _selected_sft_catalog()
    association = record["sft"]
    if (
        type(association) is not dict
        or set(association)
        != {"model_ref", "preparation_ref", "intent", "operation_digest", "native_job_name", "native_job_arn"}
        or type(association["model_ref"]) is not str
        or not re.fullmatch("model_acq_[a-f0-9]{32}", association["model_ref"])
        or type(association["preparation_ref"]) is not str
        or not re.fullmatch("prep_[a-f0-9]{32}", association["preparation_ref"])
    ):
        raise ValueError("normal run SFT association envelope differs")
    journal = _Journal(context, table_id)
    binding = journal.bind_execution(
        campaign_id=str(uuid.UUID(hex=context.run_id)), runtime_session_id=record["private_store_correlation"]
    )
    original = journal.load_original(binding, association["native_job_name"])
    if (
        original.intent.catalog != catalog
        or original.intent.operation != "train"
        or original.disposition != "ACCEPTED"
        or original.intent.source_refs != (association["model_ref"], association["preparation_ref"])
        or original.submission_snapshot is None
        or association["intent"] != _intent_fields(original.intent)
        or association["operation_digest"] != operation_digest(original.intent)
        or (association["native_job_name"], association["native_job_arn"])
        != _native_identity(original.intent)
    ):
        raise ValueError("normal run original SFT intent differs")

    def current_check(remote):
        if resume:
            _approved_read(policy)
        if _selected_sft_catalog() != catalog:
            raise record_changed("normal run current SFT catalog changed", SFT_CATALOG_CHANGED)
        if remote:
            _item, current_record = _load(context, client, table_id, qualify)
            if current_record != record:
                raise record_changed("normal run original association changed", RUN_RECORD_CHANGED)
        if resume:
            _approved_read(policy)

    current = _record_recheck(current_check)
    current()
    session = boto3.Session(region_name=context.catalog.region)
    options = Config(connect_timeout=5, read_timeout=20, retries={"total_max_attempts": 1})
    sm = session.client("sagemaker", region_name=context.catalog.region, config=options)
    s3 = session.client("s3", region_name=context.catalog.region, config=options)
    adapter = SageMakerQwenSftBackend(
        sagemaker_client=_NativeIO(sm, current, {"describe_training_job", "list_tags"}),
        s3_client=_NativeIO(s3, current, {"get_bucket_versioning", "head_object", "get_object"}),
    )
    try:
        with adapter.qualified_io(current):
            handle = adapter.restore_original(original, fresh_inputs=original.intent.inputs)
            status = adapter.status(handle)
            current()
        if (
            handle.extra.get("restoration_status") != "verified"
            or status.state is not JobState.RUNNING
            or status.raw.get("TrainingJobStatus") not in ("InProgress", "Stopping")
        ):
            raise ValueError("normal run original SFT is not freshly verified pending")
        return {
            "schema_version": 1,
            "logical_run_id": context.run_id,
            "stage": "sft",
            "disposition": "awaiting_original",
            "native_job_arn": original.native_job_arn,
            "status": status.raw["TrainingJobStatus"],
            "pending": True,
            "lifecycle_verified": False,
            "quality_success": False,
        }
    finally:
        for client in (sm, s3):
            close = getattr(client, "close", None)
            if callable(close):
                close()


# The three keys that the status of a run that ended before fine-tuning adds to the unchanged enrolled answer.
ENDED_STATUS_KEYS = ("run_state", "ended_action", "message")
# What that status says, by the closed class of the retained report's action.
_ENDED_TEXT = {
    "input_admission_held": "This run has ended. Model input was held at the admission boundary.",
    "guardrail_intervened": "This run has ended. A guardrail intervened.",
    "response_received": "This run has ended with a final response.",
}
_ENDED_NEXT = (
    " The final report is kept: read it with the runtime action get_prompt_report and this run_id."
    " The run cannot be resumed. To try again, send a prompt; a new run starts from the beginning."
)


def _ended_before_sft(context, client, item, record, table_id, qualify):
    """DATA only: closed facts of a run that ended with a verified retained report before any SFT association.

    A row in stage `enrolled` admits no checkpoint and no continuation (`_load`), and a wake is published only
    with the SFT association. `_associate`, the only step out of this stage, runs inside the run's own worker,
    before that worker returns and retains its report. So nothing can continue such a row. A row in stage `sft`
    is never read as ended here: its original pointer may hold the report of an early-ended original that a
    replacement continues. This is a read-time projection: no stage, pointer or other record field changes, and
    it is no completion proof for a resume.
    """
    if record["stage"] != "enrolled" or "report_checkpoint_ref" not in record:
        return None
    report = _verified_report(context, _report_binding(context, record, table_id), record, qualify)
    if _load(context, client, table_id, qualify)[0] != item:
        raise ValueError("normal run record changed during the ended read")
    # The same closing qualification as the passive report read.
    qualify()
    action = report["response"]["action"]
    reason = action if action in ("input_admission_held", "guardrail_intervened") else "response_received"
    return dict(zip(ENDED_STATUS_KEYS, ("ended", reason, _ENDED_TEXT[reason] + _ENDED_NEXT), strict=True))


def read_normal_run_status(run_id: str, session_hash: str, policy=None, resume: bool = False) -> dict:
    """Read owned enrollment, its verified end before SFT, or verified pending SFT.

    Never dispatch continuation.
    """
    unavailable = {"event": "prompt_run_status", "status": "unavailable", "run_id": run_id, "found": False}
    try:
        _refs(run_id, session_hash)
        if type(resume) is not bool:
            raise ValueError("normal run read mode invalid")
        if resume:
            from .producer_snapshot import _approved_read

            _approved_read(policy)
        context = prepare_normal_run(run_id, session_hash)
        if context is None:
            return unavailable
        with _native(context) as (client, table_id, qualify, _sts):
            item, record = _load(context, client, table_id, qualify)
            if record["stage"] == "sft":
                progress = _pending(context, client, record, table_id, qualify, policy, resume)
                qualify()
                return {
                    "event": "prompt_run_status",
                    "status": "available",
                    "run_id": run_id,
                    "found": True,
                    "progress": progress,
                }
            ended = _ended_before_sft(context, client, item, record, table_id, qualify)
        if resume and ended is None:
            return unavailable
        return {
            "event": "prompt_run_status",
            "status": "enrolled",
            "run_id": run_id,
            "found": True,
            "stage": "enrolled",
            "resume_ready": False,
            "data_only": True,
            "durable_resume": False,
            "lifecycle_verified": False,
            "quality_success": False,
            **({} if ended is None else ended),
        }
    except Exception:
        return unavailable


def _checkpoint_data(value):
    """Restrict new checkpoint DATA without tightening existing legacy enrollment."""
    _plain(value)
    if type(value) is dict:
        forbidden = {"client", "clients", "witness", "pickle", "callback_handler", "policy"}
        if forbidden.intersection(value) or any(key.startswith("_amlas_") for key in value):
            raise ValueError("checkpoint contains private execution state")
        for item in value.values():
            _checkpoint_data(item)
    elif type(value) is list:
        for item in value:
            _checkpoint_data(item)


def _effective_sft_inputs(inputs, *, _diagnostic_predicate=None):
    """Resolve public defaults and report only the fixed failing predicate."""

    def refuse(predicate):
        if _diagnostic_predicate is not None:
            try:
                _diagnostic_predicate(predicate)
            except Exception:
                pass
        raise ValueError("original SFT tool selection differs")

    required = {"model_request_ref", "preparation_request_ref"}
    allowed = required | {"recipe_id", "plan_only"}
    if type(inputs) is not dict:
        refuse("INPUT_TYPE")
    if not required.issubset(inputs):
        refuse("REQUIRED_KEYS")
    if set(inputs) - allowed:
        refuse("EXTRA_KEYS")
    model_ref = inputs["model_request_ref"]
    preparation_ref = inputs["preparation_request_ref"]
    recipe_id = inputs.get("recipe_id", "fit_v1")
    plan_only = inputs.get("plan_only", False)
    if type(model_ref) is not str or not re.fullmatch("model_acq_[a-f0-9]{32}", model_ref):
        refuse("MODEL_REF")
    if type(preparation_ref) is not str or not re.fullmatch("prep_[a-f0-9]{32}", preparation_ref):
        refuse("PREPARATION_REF")
    if type(recipe_id) is not str or recipe_id not in ("fit_v1", "full_v1"):
        refuse("RECIPE")
    if type(plan_only) is not bool:
        refuse("PLAN_TYPE")
    if plan_only:
        refuse("PLAN_PREVIEW")
    return {
        "model_request_ref": model_ref,
        "preparation_request_ref": preparation_ref,
        "recipe_id": recipe_id,
    }


def _causal_call(
    identifier, prefix, *, _diagnostic_phase=None, _diagnostic_predicate=None, _diagnostic_size=None
):
    def mark_phase(phase):
        if _diagnostic_phase is not None:
            try:
                _diagnostic_phase(phase)
            except Exception:
                pass

    def report_size(count, outcome):
        if _diagnostic_size is not None:
            with suppress(Exception):
                _diagnostic_size(count, outcome)

    mark_phase("causal_bounds")
    if (
        type(identifier) is not str
        or not identifier
        or len(identifier) > 256
        or type(prefix) is not list
        or not prefix
    ):
        raise ValueError("bounded original SFT call DATA required")
    mark_phase("causal_checkpoint_data")
    _checkpoint_data(prefix)
    mark_phase("causal_canonical_data")
    envelope = {"pending_tool_use_id": identifier, "causal_prefix": prefix}
    try:
        canonical = _canonical(envelope)
    except ValueError:
        # Report only the byte count of a prefix that is over the cap; any other refusal (non-plain or
        # non-finite DATA) stays unreported because it is not a size failure.
        try:
            measured = len(
                json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            )
        except Exception:
            measured = None
        if measured is not None and measured > _MAX_BYTES:
            report_size(measured, "over")
        raise
    report_size(len(canonical.encode("utf-8")), "ok")
    call = json.loads(canonical)
    last = call["causal_prefix"][-1]
    mark_phase("causal_assistant_batch")
    if type(last) is not dict or last.get("role") != "assistant" or type(last.get("content")) is not list:
        raise ValueError("original assistant SFT batch missing")
    mark_phase("causal_unique_sft_use")
    uses = [block["toolUse"] for block in last["content"] if type(block) is dict and "toolUse" in block]
    matches = [use for use in uses if type(use) is dict and use.get("toolUseId") == identifier]
    if len(matches) != 1 or matches[0].get("name") != "submit_qwen_sft":
        raise ValueError("unique actual original SFT tool use required")
    use = matches[0]
    mark_phase("causal_sft_inputs")
    _effective_sft_inputs(use.get("input"), _diagnostic_predicate=_diagnostic_predicate)
    call["tool_use"] = use
    return call


def _budget_sft_snapshot(context, call, snapshot):
    """Pure worst-case sizing of the existing TASK inventory, never an intent."""
    # Native identity caps names at 63. Parent validation caps ASCII versions at
    # 1024 and bytes at 1MiB; quotes reserve the maximum JSON escape width.
    name = "n" * 63
    correlation = str(uuid.UUID(int=0))
    sha = "0" * 64
    parent_uri = (
        "s3://"
        + _selected_sft_catalog().bucket
        + "/sessions/"
        + correlation
        + "/prompt-runs/"
        + context.run_id
        + "/parents/"
        + sha
        + ".json"
    )
    task = {
        "schema_version": 2,
        "logical_run_id": context.run_id,
        "private_store_correlation": correlation,
        "native_job_name": name,
        "native_job_arn": "arn:aws:sagemaker:"
        + context.catalog.region
        + ":"
        + context.catalog.account_id
        + ":training-job/"
        + name,
        "operation_digest": sha,
        "source_refs": [
            call["tool_use"]["input"]["model_request_ref"],
            call["tool_use"]["input"]["preparation_request_ref"],
        ],
        "causal_parent_ref": {
            "uri": parent_uri,
            "version_id": chr(34) * 1024,
            "sha256": sha,
            "byte_count": 1024 * 1024,
            "format": "json",
        },
        "sdk_snapshot": snapshot,
        "pending_tool_use_id": call["pending_tool_use_id"],
    }
    _task_envelope(task)
    _canonical(task)


def _captured_snapshot_messages(snapshot):
    from datetime import datetime, timedelta

    if (
        type(snapshot) is not dict
        or set(snapshot) != {"scope", "schema_version", "created_at", "data", "app_data"}
        or snapshot["scope"] != "agent"
        or snapshot["schema_version"] != "1.0"
        or snapshot["app_data"] != {}
        or type(snapshot["created_at"]) is not str
        or len(snapshot["created_at"]) > 64
    ):
        raise ValueError("public SDK session snapshot envelope differs")
    created = datetime.fromisoformat(snapshot["created_at"].replace("Z", "+00:00"))
    if created.tzinfo is None or created.utcoffset() != timedelta(0):
        raise ValueError("public SDK session snapshot timestamp differs")
    _checkpoint_data(snapshot)
    _canonical(snapshot)
    data = snapshot["data"]
    if (
        type(data) is not dict
        or set(data) != {"messages", "state", "conversation_manager_state", "interrupt_state"}
        or type(data["messages"]) is not list
    ):
        raise ValueError("public SDK session snapshot DATA inventory differs")
    return data["messages"]


def _task_envelope(task: object) -> dict[str, Any]:
    if type(task) is not dict or type(task.get("schema_version")) is not int:
        raise ValueError("normal TASK field inventory differs")
    version = task["schema_version"]
    expected = {
        "schema_version",
        "logical_run_id",
        "private_store_correlation",
        "native_job_name",
        "native_job_arn",
        "operation_digest",
        "source_refs",
        "causal_parent_ref",
        "sdk_snapshot",
        "pending_tool_use_id",
    }
    if version == 1:
        expected.add("causal_prefix")
    elif version != 2:
        raise ValueError("normal TASK field inventory differs")
    if set(task) != expected:
        raise ValueError("normal TASK field inventory differs")
    return task


def _task_causal_prefix(task: object) -> list[dict[str, Any]]:
    """Read exact history without expanding the authenticated TASK wire."""
    wire = _task_envelope(task)
    _canonical(wire)
    prefix = _captured_snapshot_messages(wire["sdk_snapshot"])
    if wire["schema_version"] == 1:
        if _canonical(wire["causal_prefix"]) != _canonical(prefix):
            raise ValueError("legacy TASK snapshot prefix differs")
    return json.loads(_canonical(prefix))


def _completed_task_snapshot(task, operation_digest):
    """Derive only the fixed completion DATA from the authenticated original TASK."""
    prefix = _task_causal_prefix(task)
    if (
        type(operation_digest) is not str
        or not re.fullmatch("[a-f0-9]{64}", operation_digest)
        or task["operation_digest"] != operation_digest
    ):
        raise ValueError("original completion TASK operation differs")
    event = {
        "schema_version": 1,
        "event_type": "original_sft_completed",
        "data_only": True,
        "request_ref": "sft_" + operation_digest[:32],
        "tool_use_id": task["pending_tool_use_id"],
        "status": "verified_completed",
    }
    result = {
        "toolUseId": task["pending_tool_use_id"],
        "status": "success",
        "content": [
            {
                "text": json.dumps(
                    event, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
                )
            }
        ],
    }
    snapshot = {
        **task["sdk_snapshot"],
        "data": {
            **task["sdk_snapshot"]["data"],
            "messages": [*prefix, {"role": "user", "content": [{"toolResult": result}]}],
        },
    }
    _captured_snapshot_messages(snapshot)
    return snapshot


def _sdk_snapshot_sha256(snapshot):
    _captured_snapshot_messages(snapshot)
    return hashlib.sha256(_canonical(snapshot).encode()).hexdigest()


def _prepared_task_wire_version(context, binding, record, qualify) -> int:
    """Keep an existing prepared or admitted artifact's authenticated wire identity."""
    if record.get("prepared_checkpoint_ref") is not None:
        ref = _prepared_ref(context, record)
        kind = "prepared-tasks"
    elif record.get("task_checkpoint_ref") is not None:
        from .producer_snapshot import _ref

        ref = _ref(record["task_checkpoint_ref"])
        kind = "tasks"
    else:
        return 2
    raw = _read_object(context, binding, ref, kind, qualify)
    from .normal_run_parents import _decode

    task = _task_envelope(_decode(raw))
    _task_causal_prefix(task)
    return task["schema_version"]


def _original(context, record, table_id, name=None):
    if record["stage"] != "sft" or type(record.get("sft")) is not dict:
        raise ValueError("normal original SFT association missing")
    association = record["sft"]
    if set(association) != {
        "model_ref",
        "preparation_ref",
        "intent",
        "operation_digest",
        "native_job_name",
        "native_job_arn",
    }:
        raise ValueError("normal original association field inventory differs")
    journal = _Journal(context, table_id)
    binding = journal.bind_execution(
        campaign_id=str(uuid.UUID(hex=context.run_id)), runtime_session_id=record["private_store_correlation"]
    )
    if name is not None and name != association.get("native_job_name"):
        raise ValueError("normal original native job changed")
    original = journal.load_original(binding, association["native_job_name"])
    if (
        original.disposition != "ACCEPTED"
        or original.intent.operation != "train"
        or original.submission_snapshot is None
        or original.intent.catalog != _selected_sft_catalog()
        or original.intent.source_refs != (association.get("model_ref"), association.get("preparation_ref"))
        or association.get("intent") != _intent_fields(original.intent)
        or association.get("operation_digest") != operation_digest(original.intent)
        or (association.get("native_job_name"), association.get("native_job_arn"))
        != _native_identity(original.intent)
        or original.intent.causal_parent_ref is None
    ):
        raise ValueError("normal native-bound original TASK is unavailable")
    return original, binding


def _object_key(context, binding, ref, kind):
    from ..backends.types import DatasetArtifactRef

    if kind not in ("parents", "tasks", "prepared-tasks", "reports"):
        raise ValueError("normal checkpoint kind differs")
    limit = 1024 * 1024 if kind == "parents" else _MAX_BYTES
    prefix = (
        "sessions/"
        + binding.runtime_session_id
        + "/prompt-runs/"
        + uuid.UUID(binding.campaign_id).hex
        + "/"
        + kind
        + "/"
    )
    bucket = resolve_env_snapshot("session_bucket")
    if (
        type(ref) is not DatasetArtifactRef
        or ref.format != "json"
        or type(ref.byte_count) is not int
        or not 0 < ref.byte_count <= limit
        or type(ref.sha256) is not str
        or not re.fullmatch("[a-f0-9]{64}", ref.sha256)
        or ref.uri != "s3://" + bucket + "/" + prefix + ref.sha256 + ".json"
        or type(ref.version_id) is not str
        or ref.version_id.lower() in ("", "null")
        or len(ref.version_id) > 1024
        or any(ord(char) < 33 for char in ref.version_id)
    ):
        raise ValueError("normal checkpoint exact version/namespace differs")
    if uuid.UUID(binding.campaign_id).hex != context.run_id:
        raise ValueError("normal checkpoint original run differs")
    return bucket, prefix + ref.sha256 + ".json"


def _s3(context):
    import boto3
    from botocore.config import Config

    return boto3.Session(region_name=context.catalog.region).client(
        "s3",
        region_name=context.catalog.region,
        config=Config(connect_timeout=5, read_timeout=20, retries={"total_max_attempts": 1}),
    )


def _read_object(context, binding, ref, kind, qualify):
    bucket, key = _object_key(context, binding, ref, kind)
    client = _s3(context)
    body = None
    try:
        qualify()
        response = client.get_object(
            Bucket=bucket, Key=key, VersionId=ref.version_id, ExpectedBucketOwner=context.catalog.account_id
        )
        body = response["Body"]
        qualify()
        if response.get("VersionId") != ref.version_id or response.get("ContentLength") != ref.byte_count:
            raise ValueError("normal checkpoint native exact version differs")
        raw = body.read(ref.byte_count + 1)
        qualify()
        if (
            type(raw) is not bytes
            or len(raw) != ref.byte_count
            or hashlib.sha256(raw).hexdigest() != ref.sha256
        ):
            raise ValueError("normal checkpoint native bytes differ")
        return raw
    finally:
        if body is not None:
            body.close()
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _put_object(context, binding, raw, kind, qualify):
    from ..backends.types import DatasetArtifactRef

    limit = 1024 * 1024 if kind == "parents" else _MAX_BYTES
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError("bounded canonical checkpoint bytes required")
    sha = hashlib.sha256(raw).hexdigest()
    bucket = resolve_env_snapshot("session_bucket")
    key = (
        "sessions/"
        + binding.runtime_session_id
        + "/prompt-runs/"
        + uuid.UUID(binding.campaign_id).hex
        + "/"
        + kind
        + "/"
        + sha
        + ".json"
    )
    client = _s3(context)
    body = None
    try:
        qualify()
        if (
            client.get_bucket_versioning(Bucket=bucket, ExpectedBucketOwner=context.catalog.account_id).get(
                "Status"
            )
            != "Enabled"
        ):
            raise ValueError("normal checkpoint versioned storage required")
        qualify()
        try:
            response = client.put_object(
                Bucket=bucket,
                Key=key,
                ExpectedBucketOwner=context.catalog.account_id,
                IfNoneMatch="*",
                Body=raw,
            )
        except Exception:
            # Never repeat an attempted Put; resolve only this exact content key.
            qualify()
            response = client.get_object(
                Bucket=bucket, Key=key, ExpectedBucketOwner=context.catalog.account_id
            )
            body = response["Body"]
            qualify()
            if body.read(len(raw) + 1) != raw:
                raise ValueError("normal checkpoint publication outcome unresolved") from None
            qualify()
        else:
            qualify()
        ref = DatasetArtifactRef(
            "s3://" + bucket + "/" + key, response.get("VersionId"), sha, len(raw), "json"
        )
        _object_key(context, binding, ref, kind)
    finally:
        if body is not None:
            body.close()
        close = getattr(client, "close", None)
        if callable(close):
            close()
    if _read_object(context, binding, ref, kind, qualify) != raw:
        raise ValueError("normal checkpoint published bytes differ")
    return ref


def _replace_record(context, client, table_id, qualify, item, record, *, policy):
    from .producer_snapshot import _approved_read

    _approved_read(policy)
    raw = _canonical(record)
    after = {**item, "record": {"S": raw}, "record_sha256": {"S": hashlib.sha256(raw.encode()).hexdigest()}}
    qualify()
    _approved_read(policy)
    try:
        client.update_item(
            TableName=context.catalog.jobs_table,
            Key={"job_name": item["job_name"]},
            ConditionExpression="#record = :prior AND record_sha256 = :prior_sha",
            UpdateExpression="SET #record = :body, record_sha256 = :sha",
            ExpressionAttributeNames={"#record": "record"},
            ExpressionAttributeValues={
                ":prior": item["record"],
                ":prior_sha": item["record_sha256"],
                ":body": after["record"],
                ":sha": after["record_sha256"],
            },
        )
    except Exception as error:
        _approved_read(policy)
        restored, _record = _load(context, client, table_id, qualify)
        _approved_read(policy)
        if restored != after:
            raise ValueError("normal TASK CAS outcome unresolved") from error
    finally:
        qualify()
        _approved_read(policy)


def _original_training_complete(observed) -> bool:
    """Whether a freshly observed original SFT result is a completion the run may continue from.

    For `fit_v1` and `full_v1` this accepts what the original worker's own wait accepts
    (`ToolBindingBackend`, kind `sft`). The adapter sets `trained` only after the native completion and
    the pinned outputs verified, and sets `full_training_complete` for a `full_v1` original alone, so a
    `fit_v1` pilot is complete when it is trained. Every other or absent recipe needs the full-training
    proof.
    """
    if observed.get("trained") is not True:
        return False
    return observed.get("recipe_id") == "fit_v1" or observed.get("full_training_complete") is True
