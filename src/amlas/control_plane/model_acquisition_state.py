"""Session-owned private model fetch handles; refs carry no execution grants."""

from __future__ import annotations

import hashlib
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from ..backends.adapters.failure_cause import record_change
from ..backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
from ..backends.types import JobHandle


@dataclass(frozen=True)
class _Record:
    adapter: ProcessingModelWeightAcquisitionBackend
    handle: JobHandle


class SessionModelAcquisitionStore:
    """Conversation lifetime only; not process-restart persistence or authority."""

    def __init__(
        self,
        *,
        runtime_session_id: str | None = None,
        campaign_binding=None,
        campaign_entry=None,
        campaign_services=None,
        adoption_provider=None,
        policy_getter=None,
    ) -> None:
        if campaign_binding is not None:
            from .model_campaign_binding import ModelCampaignModelBinding

            if type(campaign_binding) is not ModelCampaignModelBinding:
                raise ValueError("exact private model campaign binding required")
            campaign_binding.entry.require_current()
            if runtime_session_id != campaign_binding.entry.private_store_correlation:
                raise ValueError("original private model store correlation differs")
        if campaign_entry is not None:
            from .campaign_runtime import CampaignEntryContext

            if type(campaign_entry) is not CampaignEntryContext:
                raise ValueError("exact current campaign entry required")
            campaign_entry.require_current()
            if runtime_session_id != campaign_entry.private_store_correlation:
                raise ValueError("original private campaign correlation differs")
        original = (
            campaign_entry
            if campaign_entry is not None
            else (campaign_binding.entry if campaign_binding is not None else None)
        )
        if campaign_binding is not None and original is not campaign_binding.entry:
            if original._factory is not campaign_binding.entry._factory or (
                original.campaign_ref,
                original.campaign_id,
                original.private_store_correlation,
            ) != (
                campaign_binding.entry.campaign_ref,
                campaign_binding.entry.campaign_id,
                campaign_binding.entry.private_store_correlation,
            ):
                raise ValueError("original campaign factory/binding differs")
        self._campaign_entry = original
        self._campaign_services = campaign_services
        self._campaign_binding = campaign_binding
        self.runtime_session_id = runtime_session_id or str(uuid.uuid4())
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.runtime_session_id):
            raise ValueError("model store session differs")
        if adoption_provider is not None:
            from .model_pack_adoption import ModelPackAdoptionProvider

            if type(adoption_provider) is not ModelPackAdoptionProvider:
                raise ValueError("exact configured model adoption provider required")
        if policy_getter is not None and not callable(policy_getter):
            raise ValueError("live model policy getter required")
        self._adoption_provider = adoption_provider
        self._policy_getter = policy_getter
        self._adoptions: dict[str, dict] = {}
        self._adoption_keys: dict[tuple[str, str, str], str] = {}
        self._records: dict[str, _Record] = {}
        self._catalogs: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()
        self._submission_lock = threading.Lock()

    def _require_invocation_context(self):
        """Pure original-factory fence before legacy resolution or ordinary lookup."""
        from .campaign_runtime import current_campaign_entry, current_campaign_runtime

        current, runtime = current_campaign_entry(), current_campaign_runtime()
        original = self._campaign_entry
        if original is None:
            if current is not None or runtime is not None:
                raise ValueError("active campaign cannot adopt legacy model state")
            return
        original._check()
        original._factory._process()
        if runtime is not None and runtime is not original._factory:
            raise ValueError("foreign active model campaign factory refused")
        if current is not None and (
            current._factory is not original._factory
            or (current.campaign_ref, current.campaign_id, current.private_store_correlation)
            != (original.campaign_ref, original.campaign_id, self.runtime_session_id)
        ):
            raise ValueError("foreign active model campaign entry refused")
        if (
            self._campaign_binding is not None
            and self._campaign_binding.entry._factory is not original._factory
        ):
            raise ValueError("original model factory anchor changed")

    def existing(self, catalog_id: str, catalog_version: str) -> str | None:
        self._require_invocation_context()
        with self._lock:
            reference = self._catalogs.get((catalog_id, catalog_version))
        if self._campaign_binding is not None:
            self._require_original_campaign_context()
            self._campaign_binding.require_current_exposure(reference)
        return reference

    def _selected_campaign_services(self):
        from .model_campaign_binding import ModelCampaignServices, current_model_campaign_services

        self._require_invocation_context()
        current = current_model_campaign_services()
        if (
            current is not None
            and self._campaign_services is not None
            and current is not self._campaign_services
        ):
            raise ValueError("original private model services replacement refused")
        services = current if current is not None else self._campaign_services
        if type(services) is not ModelCampaignServices or self._campaign_entry is None:
            raise ValueError("private model campaign services missing")
        if services._runtime is not self._campaign_entry._factory:
            raise ValueError("original model services factory differs")
        return services

    def bind_campaign(self, adapter):
        self._require_invocation_context()
        if self._campaign_binding is not None:
            if self._campaign_binding.adapter is not adapter:
                raise ValueError("original campaign model adapter differs")
            self._require_original_campaign_context()
            self._campaign_binding.require_current_exposure()
            return
        services = self._selected_campaign_services()
        self._require_original_campaign_context()
        # The retained constructor entry/factory remains the anchor; only its
        # factory may create a fresh planning DATA snapshot after publication.
        entry = self._campaign_entry._factory.refresh_original_planning(self._campaign_entry)
        candidate = services.build_for_entry(entry, adapter)
        self._require_original_campaign_context()
        candidate.require_current_exposure()
        self._campaign_binding = candidate

    def require_campaign_services(self):
        if self._campaign_binding is not None:
            self._require_invocation_context()
            return
        self._selected_campaign_services()

    def _require_original_campaign_context(self):
        self._require_invocation_context()
        if self._campaign_entry is None:
            raise ValueError("original model campaign entry absent")
        self._campaign_entry.require_current()

    def submit_original(self, *, catalog_id, catalog_version, policy):
        from .model_campaign_binding import (
            ModelCampaignBindingError,
            ModelSubmissionAttemptError,
            ModelSubmissionResult,
        )

        binding = self._campaign_binding
        if binding is None:
            raise ValueError("private model campaign binding absent")
        self._require_original_campaign_context()
        try:
            result = binding.submit_original(catalog_id, catalog_version, policy)
        except (ModelSubmissionAttemptError, ModelCampaignBindingError):
            raise
        except Exception as error:
            # Calling the private bridge is not proof that Create was dispatched.
            raise ModelSubmissionAttemptError(
                dispatch_attempted=None,
                currently_qualified=False,
                result=None,
            ) from error
        try:
            if type(result) is not ModelSubmissionResult or type(result.executed) is not bool:
                raise ValueError("exact original model submission result required")
            reference, handle = result.reference, result.handle
            if (
                type(reference) is not str
                or not re.fullmatch("model_acq_[a-f0-9]{32}", reference)
                or type(handle) is not JobHandle
                or handle.backend != "model_processing"
                or handle.extra["bindings"]["runtime_session_id"] != self.runtime_session_id
            ):
                raise ValueError("original model submission differs")
            with self._lock:
                prior = self._records.get(reference)
                selected = self._catalogs.get((catalog_id, catalog_version))
                if (
                    prior is not None
                    and (prior.adapter is not binding.adapter or prior.handle.id != handle.id)
                ) or (selected is not None and selected != reference):
                    raise ValueError("original model alias replacement refused")
                records = {**self._records, reference: _Record(binding.adapter, handle)}
                catalogs = {**self._catalogs, (catalog_id, catalog_version): reference}
                self._require_original_campaign_context()
                binding.require_current_exposure(reference)
                # Failed qualification leaves only the durable original journal;
                # no ordinary alias/map is published before the final checks.
                self._records, self._catalogs = records, catalogs
            return result
        except Exception as error:
            attempted = True if type(result) is ModelSubmissionResult and result.executed is True else None
            raise ModelSubmissionAttemptError(
                dispatch_attempted=attempted,
                currently_qualified=False,
                result=None,
            ) from error

    def add(self, adapter: ProcessingModelWeightAcquisitionBackend, handle: JobHandle) -> str:
        self._require_invocation_context()
        if self._campaign_entry is not None:
            raise ValueError("campaign models require the original journal path")
        if (
            not isinstance(adapter, ProcessingModelWeightAcquisitionBackend)
            or type(handle) is not JobHandle
            or handle.backend != "model_processing"
            or handle.extra["bindings"]["runtime_session_id"] != self.runtime_session_id
        ):
            raise ValueError("model private record type/session differs")
        reference = "model_acq_" + uuid.uuid4().hex
        with self._lock:
            self._records[reference] = _Record(adapter, handle)
            self._catalogs[(handle.extra["catalog_id"], handle.extra["catalog_version"])] = reference
        return reference

    def _original_producer_handle(self, reference, binding):
        from .campaign_state import _binding_fields

        fields = _binding_fields(binding)
        with self._lock:
            record = self._records.get(reference) if type(reference) is str else None
        if record is None or self.runtime_session_id != fields["runtime_session_id"]:
            raise ValueError("original scoped model producer missing")
        if record.handle.extra["bindings"]["runtime_session_id"] != self.runtime_session_id:
            raise ValueError("original model producer correlation differs")
        return record.handle

    def _restore_original_producer(self, reference, adapter, handle, binding):
        from .campaign_state import _binding_fields

        fields = _binding_fields(binding)
        if (
            type(reference) is not str
            or not re.fullmatch("model_acq_[a-f0-9]{32}", reference)
            or type(adapter) is not ProcessingModelWeightAcquisitionBackend
            or type(handle) is not JobHandle
            or handle.backend != "model_processing"
            or fields["runtime_session_id"] != self.runtime_session_id
            or handle.extra["bindings"]["runtime_session_id"] != self.runtime_session_id
        ):
            raise ValueError("typed original model producer required")
        with self._lock:
            if self._records or self._catalogs:
                raise ValueError("original model restore requires empty temporary state")
            self._records[reference] = _Record(adapter, handle)
            self._catalogs[(handle.extra["catalog_id"], handle.extra["catalog_version"])] = reference

    def observe(self, request_ref: str) -> dict[str, Any]:
        unknown = {
            "success": False,
            "accepted": False,
            "acquired": False,
            "trusted": False,
            "ready": False,
            "prepared": False,
            "trained": False,
            "status": "unknown_reference",
        }
        if type(request_ref) is not str or not re.fullmatch(r"model_acq_[a-f0-9]{32}", request_ref):
            return unknown
        try:
            self._require_invocation_context()
            with self._lock:
                record = self._records.get(request_ref)
            if record is None:
                with self._lock:
                    adopted = self._adoptions.get(request_ref)
                if adopted is None:
                    return unknown
                current = self._verified_adoption(adopted)
                return {
                    "success": True,
                    "accepted": True,
                    "acquired": True,
                    "trained": False,
                    "trusted": False,
                    "ready": False,
                    "status": "reused_verified",
                    "origin": "verified_pack_adoption",
                    "request_ref": request_ref,
                    "file_refs": current["file_refs"],
                    "manifest_refs": current["manifest_refs"],
                    "producer_job_arn": current["producer"]["arn"],
                }
            if self._campaign_binding is not None:
                self._require_original_campaign_context()
                result = self._campaign_binding.observe_original(request_ref)
                self._require_original_campaign_context()
                self._campaign_binding.require_current_exposure(request_ref)
            else:
                result = record.adapter.artifacts(record.handle)
            return {**result, "request_ref": request_ref}
        except Exception as error:
            if record_change(error) is not None:
                raise  # the caller's own run-record recheck decides; it is no observation failure
            return {
                **unknown,
                "status": "observation_failed",
                "error_type": type(error).__name__,
            }

    def _verified_model_snapshot(self, request_ref: str) -> dict[str, Any]:
        """Private source transfer uses a fresh parent-verified exact-version result."""
        result = self.observe(request_ref)
        if (
            result.get("success") is not True
            or result.get("acquired") is not True
            or type(result.get("file_refs")) is not list
            or len(result["file_refs"]) != 9
        ):
            raise ValueError("model fetch has no verified source snapshot")
        with self._lock:
            record = self._records[request_ref]
        if record.handle.extra["bindings"]["runtime_session_id"] != self.runtime_session_id:
            raise ValueError("model source snapshot belongs to another session")
        return {
            "catalog_id": record.handle.extra["catalog_id"],
            "catalog_version": record.handle.extra["catalog_version"],
            "native_job_arn": record.handle.arn,
            "bindings": dict(record.handle.extra["bindings"]),
            "file_refs": tuple(dict(ref) for ref in result["file_refs"]),
            "manifest_refs": {name: dict(ref) for name, ref in result["manifest_refs"].items()},
            "trusted": False,
            "ready": False,
            "trained": False,
        }

    def _verified_sft_model_snapshot(self, request_ref: str, *, _qualify=None):
        """Typed private handoff; reject foreign original state before any SDK read."""
        from ..backends.types import DatasetArtifactRef
        from .sft_state import VerifiedSftModelSource

        with self._lock:
            record = (
                self._records.get(request_ref)
                if type(request_ref) is str and re.fullmatch("model_acq_[a-f0-9]{32}", request_ref)
                else None
            )
        if record is None:
            with self._lock:
                adopted = self._adoptions.get(request_ref)
            if adopted is not None:
                return self._verified_adopted_sft_source(adopted, _qualify=_qualify)
        if record is None or record.handle.extra["bindings"]["runtime_session_id"] != self.runtime_session_id:
            raise ValueError("unknown or foreign original SFT model ref")
        snapshot = self._verified_model_snapshot(request_ref)

        def artifact(value):
            return DatasetArtifactRef(
                **{key: value[key] for key in ("uri", "version_id", "sha256", "byte_count", "format")}
            )

        return VerifiedSftModelSource(
            self.runtime_session_id,
            snapshot["catalog_id"],
            snapshot["catalog_version"],
            record.handle.arn,
            tuple(sorted(record.handle.extra["bindings"].items())),
            tuple((ref["path"], artifact(ref)) for ref in snapshot["file_refs"]),
            tuple(sorted((name, artifact(ref)) for name, ref in snapshot["manifest_refs"].items())),
        )

    def _adoption_policy(self, policy=None):
        from .approvals import ApprovalPolicy

        selected = self._policy_getter() if policy is None and self._policy_getter is not None else policy
        if type(selected) is not ApprovalPolicy:
            raise ValueError("current typed adoption policy required")
        return selected

    def verified_pack_candidates(self):
        self._require_invocation_context()
        if self._adoption_provider is None:
            return []
        return self._adoption_provider.candidates(policy=self._adoption_policy())

    def reuse_verified(self, *, catalog_id, catalog_version, pack_ref, policy):
        self._require_invocation_context()
        if self._adoption_provider is None:
            return {
                "success": False,
                "accepted": False,
                "acquired": False,
                "executed": False,
                "status": "reuse_unconfigured",
            }
        candidates = self._adoption_provider.candidates(policy=self._adoption_policy(policy))
        if {
            "catalog_id": catalog_id,
            "catalog_version": catalog_version,
            "pack_ref": pack_ref,
        } not in candidates:
            return {
                "success": False,
                "accepted": False,
                "acquired": False,
                "executed": False,
                "status": "unknown_pack_reference",
            }
        key = (catalog_id, catalog_version, pack_ref)

        def current():
            from .producer_snapshot import _approved_read

            self._require_invocation_context()
            live = (
                self._adoption_policy() if self._policy_getter is not None else self._adoption_policy(policy)
            )
            _approved_read(live)

        current()
        snapshot = self._adoption_provider.adopt(
            pack_ref, self.runtime_session_id, self._adoption_policy(policy), qualify=current
        )
        current()
        with self._lock:
            ref = self._adoption_keys.get(key)
            if ref is None:
                ref = (
                    "model_acq_"
                    + hashlib.sha256((self.runtime_session_id + pack_ref).encode()).hexdigest()[:32]
                )
                self._adoption_keys[key] = ref
                self._adoptions[ref] = snapshot
        return {
            "success": True,
            "accepted": True,
            "acquired": True,
            "executed": True,
            "trained": False,
            "trusted": False,
            "ready": False,
            "status": "reused_verified",
            "origin": "verified_pack_adoption",
            "request_ref": ref,
            "reused": True,
            "native_processing_created": False,
        }

    def _verified_adoption(self, snapshot, policy=None, qualify=None):
        self._require_invocation_context()
        if self._adoption_provider is None:
            raise ValueError("selected adoption provider unavailable")

        from ..backends.adapters.coding_preparation import record_step
        from .normal_run_checkpoint import _record_recheck

        def check(remote):
            from .producer_snapshot import _approved_read

            self._require_invocation_context()
            live = (
                self._adoption_policy() if self._policy_getter is not None else self._adoption_policy(policy)
            )
            _approved_read(live)
            if qualify is not None:
                # Between two chunks of a pack body the caller's own qualifier offers its cheap variant.
                if remote:
                    qualify()
                else:
                    record_step(qualify)

        current = _record_recheck(check)
        return self._adoption_provider.restore(
            snapshot, self.runtime_session_id, self._adoption_policy(policy), current
        )

    def _verified_adopted_sft_source(self, snapshot, *, _qualify=None):
        from .sft_state import VerifiedSftAdoptedModelSource, VerifiedSftModelSource

        value = self._verified_adoption(snapshot, qualify=_qualify)

        def artifact(row):
            return DatasetArtifactRef(
                **{key: row[key] for key in ("uri", "version_id", "sha256", "byte_count", "format")}
            )

        from ..backends.types import DatasetArtifactRef

        original = value["original_artifacts"]
        producer = value["producer"]
        old = VerifiedSftModelSource(
            producer["bindings"]["runtime_session_id"],
            value["catalog_id"],
            value["catalog_version"],
            producer["arn"],
            tuple(sorted(producer["bindings"].items())),
            tuple((r["path"], artifact(r)) for r in original["file_refs"]),
            tuple(sorted((k, artifact(r)) for k, r in original["manifest_refs"].items())),
        )
        return VerifiedSftAdoptedModelSource(
            self.runtime_session_id,
            value["catalog_id"],
            value["catalog_version"],
            old,
            tuple((r["path"], artifact(r)) for r in value["file_refs"]),
            tuple(sorted((k, artifact(r)) for k, r in value["manifest_refs"].items())),
            artifact(value["adoption_ref"]),
        )

    def _adopted_parent_snapshot(self, reference, binding, policy=None, qualify=None):
        from .campaign_state import _binding_fields

        if _binding_fields(binding)["runtime_session_id"] != self.runtime_session_id:
            raise ValueError("foreign adoption parent binding")
        with self._lock:
            value = self._adoptions.get(reference)
        if value is None:
            return None
        return self._verified_adoption(value, policy, qualify)

    def _restore_adopted_parent(self, snapshot, binding, policy, qualify=None):
        from .campaign_state import _binding_fields

        if _binding_fields(binding)["runtime_session_id"] != self.runtime_session_id:
            raise ValueError("foreign adoption restore binding")
        value = self._verified_adoption(snapshot, policy, qualify)
        ref = (
            "model_acq_"
            + hashlib.sha256((self.runtime_session_id + value["pack_ref"]).encode()).hexdigest()[:32]
        )
        with self._lock:
            if self._records or self._adoptions:
                raise ValueError("adoption restore requires empty model store")
            self._adoptions[ref] = value
            self._adoption_keys[(value["catalog_id"], value["catalog_version"], value["pack_ref"])] = ref
        return ref


def build_model_acquisition_store(*, runtime_session_id, env=None, policy_getter=None, **kwargs):
    import json
    import os

    from ..backends.types import DatasetArtifactRef
    from ..config.externalized import resolve_env_snapshot

    selected = os.environ if env is None else env
    raw = selected.get("AMLAS_VERIFIED_MODEL_PACK_CATALOG_REF", "").strip()
    if not raw:
        return SessionModelAcquisitionStore(
            runtime_session_id=runtime_session_id, policy_getter=policy_getter, **kwargs
        )
    from .model_pack_adoption import ModelPackAdoptionProvider
    from .normal_run_checkpoint import _pairs

    ref = DatasetArtifactRef(**json.loads(raw, object_pairs_hook=_pairs))
    import boto3
    from botocore.config import Config

    session = boto3.Session(region_name=resolve_env_snapshot("aws_region", env=env))
    opts = Config(connect_timeout=5, read_timeout=30, retries={"total_max_attempts": 1})
    scope = {
        "account_id": resolve_env_snapshot("aws_account_id", env=env),
        "region": session.region_name,
        "bucket": resolve_env_snapshot("session_bucket", env=env),
        "jobs_table": selected.get("AMLAS_JOBS_TABLE", ""),
        "execution_role_arn": selected.get("AMLAS_RUNTIME_EXEC_ROLE_ARN", ""),
        "execution_role_id": selected.get("AMLAS_RUNTIME_EXEC_ROLE_ID", ""),
        "deployment_id": selected.get("AMLAS_DEPLOYMENT_ID", ""),
        "deployment_prefix": selected.get("AMLAS_DEPLOYMENT_PREFIX", ""),
    }
    s3 = session.client("s3", config=opts)
    sm = session.client("sagemaker", config=opts)
    from ..backends.adapters.qwen_sft import SageMakerQwenSftBackend

    def current_scope():
        current = os.environ if env is None else env
        current_ref = DatasetArtifactRef(
            **json.loads(current.get("AMLAS_VERIFIED_MODEL_PACK_CATALOG_REF", ""), object_pairs_hook=_pairs)
        )
        fresh = {
            "account_id": resolve_env_snapshot("aws_account_id", env=env),
            "region": resolve_env_snapshot("aws_region", env=env),
            "bucket": resolve_env_snapshot("session_bucket", env=env),
            "jobs_table": current.get("AMLAS_JOBS_TABLE", ""),
            "execution_role_arn": current.get("AMLAS_RUNTIME_EXEC_ROLE_ARN", ""),
            "execution_role_id": current.get("AMLAS_RUNTIME_EXEC_ROLE_ID", ""),
            "deployment_id": current.get("AMLAS_DEPLOYMENT_ID", ""),
            "deployment_prefix": current.get("AMLAS_DEPLOYMENT_PREFIX", ""),
        }
        return fresh, current_ref

    provider = ModelPackAdoptionProvider(
        catalog_ref=ref,
        scope=scope,
        scope_getter=current_scope,
        s3=s3,
        ddb=session.client("dynamodb", config=opts),
        sts=session.client("sts", config=opts),
        model_adapter=ProcessingModelWeightAcquisitionBackend(sagemaker_client=sm, s3_client=s3),
        copy_adapter=SageMakerQwenSftBackend(sagemaker_client=sm, s3_client=s3),
    )
    return SessionModelAcquisitionStore(
        runtime_session_id=runtime_session_id,
        adoption_provider=provider,
        policy_getter=policy_getter,
        **kwargs,
    )
