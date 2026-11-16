"""Narrow server-only campaign persistence port; intentionally not globally registered."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..control_plane.campaign_state import (
    CampaignOperationIntent,
    OriginalJobReservation,
    TrustedExecutionBinding,
)
from ..control_plane.sft_state import SftSubmissionSnapshot


@runtime_checkable
class CampaignStateBackend(Protocol):
    def bind_execution(self, *, campaign_id: str, runtime_session_id: str) -> TrustedExecutionBinding: ...
    def reserve_original_job(
        self, binding: TrustedExecutionBinding, intent: CampaignOperationIntent
    ) -> OriginalJobReservation: ...
    def mark_create_attempted(
        self, binding: TrustedExecutionBinding, reservation: OriginalJobReservation
    ) -> OriginalJobReservation: ...
    def attach_submission_snapshot(
        self,
        binding: TrustedExecutionBinding,
        reservation: OriginalJobReservation,
        snapshot: SftSubmissionSnapshot,
    ) -> OriginalJobReservation: ...
    def record_submission_outcome(
        self,
        binding: TrustedExecutionBinding,
        reservation: OriginalJobReservation,
        *,
        disposition: str,
        native_job_arn: str | None = None,
        error_code: str | None = None,
    ) -> OriginalJobReservation: ...
    def load_original(self, binding: TrustedExecutionBinding, job_name: str) -> OriginalJobReservation: ...
    def observe_original(self, binding: TrustedExecutionBinding, job_name: str) -> dict: ...
