"""Conversation-owned acquisition references; no policy or permission storage."""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, replace
from typing import Any

from ..backends.ports import DataAcquisitionBackend, DataInspectionBackend
from ..backends.types import DatasetArtifactRef, DatasetInspectionSource, JobHandle


def acquisition_agent_tool(agent: Any, *, name: str) -> Any:
    """Preserve SDK history/streaming while forwarding only typed caller state."""
    from strands.types._events import AgentAsToolStreamEvent, ToolInterruptEvent, ToolResultEvent
    from strands.types.tools import AgentTool

    from .approvals import ApprovalPolicy

    ordinary = agent.as_tool(name=name, preserve_context=True)

    class TypedAcquisitionAgentTool(AgentTool):
        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()

        @property
        def agent(self) -> Any:
            return agent

        @property
        def tool_name(self) -> str:
            return ordinary.tool_name

        @property
        def tool_spec(self) -> Any:
            return ordinary.tool_spec

        @property
        def tool_type(self) -> str:
            return ordinary.tool_type

        def get_display_properties(self) -> dict[str, str]:
            return ordinary.get_display_properties()

        async def stream(self, tool_use: Any, invocation_state: dict[str, Any], **kwargs: Any) -> Any:
            # The SDK's ordinary as_tool keeps conversation history but omits
            # caller invocation state. Forward typed caller objects per call;
            # neither child state nor model-visible input can supply authority.
            parent_state = invocation_state if isinstance(invocation_state, dict) else {}
            parent_policy = parent_state.get("_amlas_approval_policy")
            policy = (
                replace(parent_policy, allowlist=list(parent_policy.allowlist))
                if isinstance(parent_policy, ApprovalPolicy)
                else ApprovalPolicy()
            )
            from .coding_case_state import SessionCodingCaseFreezeStore
            from .generation_state import SessionCodingGenerationStore
            from .model_acquisition_state import SessionModelAcquisitionStore
            from .paired_state import SessionPairedCodingStore
            from .preparation_state import SessionPreparationStore
            from .sft_state import SessionSftStore

            parent_store = parent_state.get("_amlas_acquisition_store")
            cases = parent_state.get("_amlas_case_freeze_store")
            cases = (
                cases
                if type(cases) is SessionCodingCaseFreezeStore
                and isinstance(parent_store, SessionAcquisitionStore)
                and cases._acquired is parent_store
                and cases.runtime_session_id == parent_store.runtime_session_id
                else None
            )
            preparation = parent_state.get("_amlas_preparation_store")
            preparation = (
                preparation
                if (
                    isinstance(preparation, SessionPreparationStore)
                    and preparation._acquired is parent_store
                    and isinstance(parent_store, SessionAcquisitionStore)
                    and preparation.runtime_session_id == parent_store.runtime_session_id
                    and (preparation._case_store is None or preparation._case_store is cases)
                )
                else None
            )
            model_store = parent_state.get("_amlas_model_acquisition_store")
            model_store = (
                model_store
                if (
                    isinstance(model_store, SessionModelAcquisitionStore)
                    and isinstance(parent_store, SessionAcquisitionStore)
                    and model_store.runtime_session_id == parent_store.runtime_session_id
                )
                else None
            )
            sft = parent_state.get("_amlas_sft_store")
            sft = (
                sft
                if (
                    isinstance(sft, SessionSftStore)
                    and sft._models is model_store
                    and sft._prepared is preparation
                    and isinstance(parent_store, SessionAcquisitionStore)
                    and sft.runtime_session_id == parent_store.runtime_session_id
                )
                else None
            )
            generation = parent_state.get("_amlas_generation_store")
            generation = (
                generation
                if isinstance(parent_store, SessionAcquisitionStore)
                and type(generation) is SessionCodingGenerationStore
                and generation._models is model_store
                and generation._cases is cases
                and generation._sft is sft
                and generation.runtime_session_id == parent_store.runtime_session_id
                else None
            )
            pair = parent_state.get("_amlas_pair_store")
            pair = (
                pair
                if type(pair) is SessionPairedCodingStore
                and pair._generation is generation
                and pair._cases is cases
                and isinstance(parent_store, SessionAcquisitionStore)
                and pair.runtime_session_id == parent_store.runtime_session_id
                else None
            )
            state = {
                "_amlas_sft_store": sft,
                "_amlas_generation_store": generation,
                "_amlas_pair_store": pair,
                "_amlas_case_freeze_store": cases,
                "_amlas_approval_policy": policy,
                "_amlas_model_acquisition_store": model_store,
                "_amlas_preparation_store": preparation,
                "_amlas_acquisition_store": (
                    parent_store if isinstance(parent_store, SessionAcquisitionStore) else None
                ),
            }
            if not self._lock.acquire(blocking=False):
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use["toolUseId"],
                        "status": "error",
                        "content": [{"text": "Collection specialist is already processing a request."}],
                    }
                )
                return
            try:
                prompt = tool_use["input"].get("input", "")
                result = None
                async for event in agent.stream_async(prompt, invocation_state=state):
                    if "result" in event:
                        result = event["result"]
                    else:
                        yield AgentAsToolStreamEvent(tool_use, event, self)
                if result is None:
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use["toolUseId"],
                            "status": "error",
                            "content": [{"text": "Collection specialist produced no result."}],
                        }
                    )
                elif result.stop_reason == "interrupt" and result.interrupts:
                    yield ToolInterruptEvent(tool_use, list(result.interrupts))
                else:
                    content = (
                        [{"json": result.structured_output.model_dump()}]
                        if result.structured_output
                        else [{"text": str(result)}]
                    )
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use["toolUseId"],
                            "status": "success",
                            "content": content,
                        }
                    )
            finally:
                self._lock.release()

    return TypedAcquisitionAgentTool()


@dataclass(frozen=True)
class _AcquisitionRecord:
    adapter: DataAcquisitionBackend
    handle: JobHandle


@dataclass(frozen=True)
class _InspectionRecord:
    adapter: DataInspectionBackend
    handle: JobHandle


class SessionAcquisitionStore:
    """Hold native handles for one conversation, separately from turn authority.

    This in-memory store does not prove process-restart durability. Model-visible
    refs select existing records; they never reconstruct adapters or native handles.
    """

    def __init__(self) -> None:
        self.runtime_session_id = str(uuid.uuid4())
        self._records: dict[str, _AcquisitionRecord] = {}
        self._inspections: dict[str, _InspectionRecord] = {}
        self._lock = threading.Lock()

    @classmethod
    def _for_campaign_entry(cls, entry):
        """Private empty shell bound to the sealed original campaign correlation."""
        from .campaign_runtime import CampaignEntryContext

        if type(entry) is not CampaignEntryContext:
            raise ValueError("exact current campaign entry required")
        entry.require_current()
        value = cls()
        value.runtime_session_id = entry.private_store_correlation
        entry.require_current()
        return value

    @classmethod
    def _for_original_correlation(cls, binding):
        """Server-only empty shell; correlation is data, never approval."""
        from .campaign_state import _binding_fields

        fields = _binding_fields(binding)
        value = cls()
        value.runtime_session_id = fields["runtime_session_id"]
        return value

    def _restore_original_producers(self, entries, binding):
        """Install exact aliases into one empty temporary store; DATA grants nothing."""
        from .campaign_state import _binding_fields

        fields = _binding_fields(binding)
        if type(entries) is not tuple or not 1 <= len(entries) <= 16:
            raise ValueError("bounded original acquisition population required")
        records = {}
        for reference, adapter, handle in entries:
            if (
                type(reference) is not str
                or not re.fullmatch("acq_[a-f0-9]{32}", reference)
                or reference in records
                or not isinstance(adapter, DataAcquisitionBackend)
                or type(handle) is not JobHandle
                or handle.backend != adapter.backend_name
                or handle.backend != "processing"
                or self.runtime_session_id != fields["runtime_session_id"]
                or handle.extra.get("runtime_session_id") != self.runtime_session_id
            ):
                raise ValueError("typed original acquisition binding differs")
            records[reference] = _AcquisitionRecord(adapter, handle)
        with self._lock:
            if self._records or self._inspections:
                raise ValueError("acquisition restoration requires empty temporary state")
            self._records.update(records)

    def add(self, adapter: DataAcquisitionBackend, handle: JobHandle) -> str:
        if not isinstance(handle, JobHandle) or not isinstance(adapter, DataAcquisitionBackend):
            raise TypeError("acquisition records require a typed backend and native handle")
        if handle.backend not in ("processing", "local") or handle.backend != adapter.backend_name:
            raise ValueError("acquisition handle belongs to another backend")
        reference = "acq_" + uuid.uuid4().hex
        with self._lock:
            self._records[reference] = _AcquisitionRecord(adapter, handle)
        return reference

    def _verified_inspection_source(self, request_ref: str) -> DatasetInspectionSource:
        """Combine fresh verified bytes with the original private native record."""
        if not isinstance(request_ref, str) or not re.fullmatch(r"acq_[a-f0-9]{32}", request_ref):
            raise ValueError("no acquisition record exists in this conversation")
        with self._lock:
            record = self._records.get(request_ref)
        if record is None or record.handle.backend != "processing":
            raise ValueError("managed acquired bytes are required for inspection")
        result = record.adapter.artifacts(record.handle)
        fields = result.get("artifact_ref")
        if (
            result.get("success") is not True
            or result.get("acquired") is not True
            or not isinstance(fields, dict)
            or set(fields) != {"uri", "version_id", "sha256", "byte_count", "format"}
        ):
            raise ValueError("acquisition has no verified versioned artifact")
        extra = record.handle.extra
        if extra.get("runtime_session_id") != self.runtime_session_id:
            raise ValueError("acquisition belongs to another conversation")
        return DatasetInspectionSource(
            artifact_ref=DatasetArtifactRef(**fields),
            account_id=str(extra["account_id"]),
            region=str(extra["region"]),
            bucket=str(extra["bucket"]),
            runtime_session_id=str(extra["runtime_session_id"]),
            deployment_id=str(extra["deployment_id"]),
            deployment_prefix=str(extra["deployment_prefix"]),
            acquisition_job_arn=record.handle.arn,
        )

    def _verified_preparation_source(self, request_ref: str) -> Any:
        """Snapshot original pins and exact-version manifests from one verified record."""
        from urllib.parse import urlsplit

        from ..backends.adapters.dataset_acquisition import validate_source
        from .preparation_state import VerifiedPreparationSource

        if type(request_ref) is not str or not re.fullmatch(r"acq_[a-f0-9]{32}", request_ref):
            raise ValueError("no acquisition record exists in this conversation")
        with self._lock:
            record = self._records.get(request_ref)
        if record is None or record.handle.backend != "processing":
            raise ValueError("managed acquired bytes are required for preparation")
        extra = dict(record.handle.extra)
        if extra.get("runtime_session_id") != self.runtime_session_id:
            raise ValueError("acquisition belongs to another conversation")
        bucket = extra.get("bucket")
        output_uri = extra.get("output_uri")
        expected_uri = f"s3://{bucket}/sessions/{self.runtime_session_id}/acquisition/{record.handle.id}/"
        if type(bucket) is not str or type(output_uri) is not str or output_uri != expected_uri:
            raise ValueError("preparation source lacks an owned acquisition namespace")

        def owned_uri(value: Any) -> str:
            if (
                type(value) is not str
                or len(value) > 4096
                or not value.startswith(output_uri)
                or any(ord(char) < 32 for char in value)
            ):
                raise ValueError("preparation artifact is outside its native job namespace")
            parsed = urlsplit(value)
            relative = value[len(output_uri) :]
            if (
                parsed.scheme != "s3"
                or parsed.netloc != bucket
                or parsed.query
                or parsed.fragment
                or "\\" in relative
                or any(part in ("", ".", "..") for part in relative.split("/"))
            ):
                raise ValueError("preparation artifact lacks a confined original object path")
            return value

        def version(value: Any) -> str:
            if (
                type(value) is not str
                or not value
                or value.lower() == "null"
                or len(value) > 1024
                or any(char.isspace() or ord(char) < 32 for char in value)
            ):
                raise ValueError("preparation requires exact non-null object versions")
            return value

        def digest(value: Any) -> str:
            if type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value):
                raise ValueError("preparation requires verified object hashes")
            return value

        pins = [extra.get(name) for name in ("repo_id", "revision", "filename")]
        if any(type(value) is not str for value in pins):
            raise ValueError("preparation source lacks original immutable dataset pins")
        validate_source(*pins)
        image, source = extra.get("image_uri"), extra.get("source_commit")
        if (
            type(image) is not str
            or not re.fullmatch(
                rf"{re.escape(str(extra.get('account_id')))}\.dkr\.ecr\."
                rf"{re.escape(str(extra.get('region')))}\.amazonaws\.com(?:\.cn)?/"
                r"[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}",
                image,
            )
            or type(source) is not str
            or not re.fullmatch(r"[a-f0-9]{40}", source)
        ):
            raise ValueError("preparation source lacks its immutable acquisition image and source")
        result = record.adapter.artifacts(record.handle)
        fields, manifests = result.get("artifact_ref"), result.get("manifest_refs")
        if (
            result.get("success") is not True
            or result.get("acquired") is not True
            or type(fields) is not dict
            or set(fields) != {"uri", "version_id", "sha256", "byte_count", "format"}
            or type(manifests) is not dict
            or set(manifests) != {"receipt", "commit", "lineage"}
        ):
            raise ValueError("acquisition has no complete verified preparation source")
        artifact = DatasetArtifactRef(
            uri=owned_uri(fields["uri"]),
            version_id=version(fields["version_id"]),
            sha256=digest(fields["sha256"]),
            byte_count=fields["byte_count"],
            format=fields["format"],
        )
        if (
            type(artifact.byte_count) is not int
            or not 1 <= artifact.byte_count <= 512 * 1024 * 1024
            or artifact.format not in ("csv", "parquet", "json", "jsonl")
        ):
            raise ValueError("preparation source artifact metadata differs")
        references = []
        for name, fields in sorted(manifests.items()):
            if type(fields) is not dict or set(fields) != {"uri", "version_id", "sha256"}:
                raise ValueError("preparation manifest reference differs")
            uri = owned_uri(fields["uri"])
            if name == "receipt" and uri != output_uri + "ACQUIRED.json":
                raise ValueError("preparation receipt does not match its fixed native output")
            references.append((name, uri, version(fields["version_id"]), digest(fields["sha256"])))
        inspection_source = DatasetInspectionSource(
            artifact_ref=artifact,
            account_id=str(extra["account_id"]),
            region=str(extra["region"]),
            bucket=bucket,
            runtime_session_id=self.runtime_session_id,
            deployment_id=str(extra["deployment_id"]),
            deployment_prefix=str(extra["deployment_prefix"]),
            acquisition_job_arn=record.handle.arn,
        )
        return VerifiedPreparationSource(
            inspection_source=inspection_source,
            manifest_refs=tuple(references),
            repo_id=pins[0],
            revision=pins[1],
            filename=pins[2],
            acquisition_image_uri=image,
            acquisition_source_commit=source,
        )

    def add_inspection(self, adapter: DataInspectionBackend, handle: JobHandle) -> str:
        if not isinstance(adapter, DataInspectionBackend) or not isinstance(handle, JobHandle):
            raise TypeError("inspection records require a typed backend and native handle")
        if handle.backend != "processing" or handle.backend != adapter.backend_name:
            raise ValueError("inspection handle belongs to another backend")
        reference = "inspect_" + uuid.uuid4().hex
        with self._lock:
            self._inspections[reference] = _InspectionRecord(adapter, handle)
        return reference

    def observe_inspection(self, inspection_ref: str) -> dict[str, Any]:
        unavailable = {
            "success": False,
            "accepted": False,
            "inspected": False,
            "prepared": False,
            "trained": False,
            "status": "unknown_reference",
            "detail": "No inspection record exists in this conversation.",
        }
        if not isinstance(inspection_ref, str) or not re.fullmatch(r"inspect_[a-f0-9]{32}", inspection_ref):
            return unavailable
        with self._lock:
            record = self._inspections.get(inspection_ref)
        if record is None:
            return unavailable
        try:
            result = dict(record.adapter.report(record.handle))
            return {**result, "inspection_ref": inspection_ref, "accepted": True}
        except Exception as error:
            return {
                **unavailable,
                "accepted": True,
                "inspection_ref": inspection_ref,
                "status": "observation_failed",
                "error_type": type(error).__name__,
            }

    def observe(self, request_ref: str) -> dict[str, Any]:
        unavailable = {
            "success": False,
            "accepted": False,
            "acquired": False,
            "status": "unknown_reference",
            "detail": "No acquisition record exists in this conversation.",
        }
        if not isinstance(request_ref, str) or not re.fullmatch(r"acq_[a-f0-9]{32}", request_ref):
            return unavailable
        with self._lock:
            record = self._records.get(request_ref)
        if record is None:
            return unavailable
        try:
            result = dict(record.adapter.artifacts(record.handle))
            state = result.get("status")
            if not state:
                state = record.adapter.status(record.handle).state.value
            acquired = result.get("success") is True and result.get("acquired", True) is True
            status = {
                "RUNNING": "running",
                "PENDING": "submitted",
                "FAILED": "failed",
                "STOPPED": "stopped",
                "UNKNOWN": "observation_failed",
                "SUCCEEDED": "verification_failed",
            }.get(state, "observation_failed")
            return {
                **result,
                "request_ref": request_ref,
                "accepted": True,
                "acquired": acquired,
                "success": acquired,
                "job_state": state,
                "status": "completed" if acquired else status,
                "job_name": record.handle.id,
                "job_arn": record.handle.arn,
            }
        except Exception as error:
            return {
                "success": False,
                "accepted": True,
                "acquired": False,
                "request_ref": request_ref,
                "job_name": record.handle.id,
                "job_arn": record.handle.arn,
                "status": "observation_failed",
                "error_type": type(error).__name__,
            }
