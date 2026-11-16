"""Private planning configuration selection; types and contexts grant no authority."""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .campaign_initialization import InitializationPlanningObservation
from .campaign_planning import CampaignPlanningStore, PlanningBinding
from .campaign_runtime import CampaignEntryContext, CampaignRuntimeFactory, current_campaign_runtime
from .generation_state import SessionCodingGenerationStore

_SELECTION = ContextVar("private_campaign_planning_selection", default=None)


class CampaignPlanningServicesError(ValueError):
    """Missing or foreign current private planning configuration."""


@dataclass
class _Selection:
    services: object
    process_id: int
    active: bool = True


class CampaignPlanningServices:
    def __init__(
        self,
        *,
        runtime_factory,
        protected_port=None,
        origin_qualifier=None,
        bound_spec_reader=None,
        clock=time.time,
    ):
        if type(runtime_factory) is not CampaignRuntimeFactory or not callable(clock):
            raise CampaignPlanningServicesError("exact selected planning runtime/clock required")
        runtime_factory._process()
        self._runtime, self._port, self._origin, self._bound = (
            runtime_factory,
            protected_port,
            origin_qualifier,
            bound_spec_reader,
        )
        self._clock, self._pid = clock, os.getpid()

    def _process(self):
        if self._pid != os.getpid():
            raise CampaignPlanningServicesError("inherited planning configuration refused")
        self._runtime._process()

    def require_current_selection(self, entry):
        self._process()
        if current_campaign_planning_services() is not self:
            raise CampaignPlanningServicesError("current private planning selection missing")
        if type(entry) is not CampaignEntryContext or entry._factory is not self._runtime:
            raise CampaignPlanningServicesError("same original planning factory required")
        runtime = current_campaign_runtime()
        if runtime is not None and runtime is not self._runtime:
            raise CampaignPlanningServicesError("foreign active planning runtime refused")
        entry._check()
        entry.qualify_current_io()

    def observed_at(self, entry):
        self.require_current_selection(entry)
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise CampaignPlanningServicesError("finite planning observation time required")
        return value

    def build_for_entry(self, entry, generation_store):
        self.require_current_selection(entry)
        if (
            type(generation_store) is not SessionCodingGenerationStore
            or generation_store.runtime_session_id != entry.private_store_correlation
            or not callable(self._origin)
            or not callable(self._bound)
            or not all(
                callable(getattr(self._port, name, None))
                for name in ("require_current", "read_active", "read_original", "publish_first")
            )
        ):
            raise CampaignPlanningServicesError("complete private protected planning capability required")
        witness = entry.require_current()
        with self._runtime._context_io(entry):
            projected = self._runtime.initializer.resolve_original_planning(
                witness,
                entry.transport_session_id,
                literal_prompt=entry.literal_prompt,
                qualify_io=entry.qualify_current_io,
            )
            if type(projected) is not InitializationPlanningObservation or (
                projected.original.campaign_ref,
                projected.original.campaign_id,
                projected.original.private_store_correlation,
                projected.original.literal_prompt,
                projected.original.prompt_sha256,
            ) != (
                entry.campaign_ref,
                entry.campaign_id,
                entry.private_store_correlation,
                entry.literal_prompt,
                entry.prompt_sha256,
            ):
                raise CampaignPlanningServicesError("protected original planning projection differs")
            scope = self._runtime.coordinator._scope(entry.require_current())
            binding = PlanningBinding(
                scope["account_id"],
                scope["region"],
                scope["deployment_id"],
                scope["deployment_prefix"],
                entry.campaign_ref,
                entry.campaign_id,
                entry.private_store_correlation,
                entry.prompt_sha256,
                projected.initial_campaign_body_sha256,
                projected.initialization_publication_sha256,
            )
            store = CampaignPlanningStore(
                entry,
                binding,
                generation_store,
                protected_port=self._port,
                origin_qualifier=self._origin,
                bound_spec_reader=self._bound,
            )
            self.require_current_selection(entry)
            entry.require_current()
            return store


@contextmanager
def activate_campaign_planning_services(services):
    if type(services) is not CampaignPlanningServices:
        raise CampaignPlanningServicesError("exact selected planning services required")
    services._process()
    selected = _Selection(services, os.getpid())
    token = _SELECTION.set(selected)
    try:
        yield services
    finally:
        selected.active = False
        _SELECTION.reset(token)


def current_campaign_planning_services():
    selected = _SELECTION.get()
    if selected is None:
        return None
    if (
        type(selected) is not _Selection
        or selected.active is not True
        or selected.process_id != os.getpid()
        or type(selected.services) is not CampaignPlanningServices
    ):
        raise CampaignPlanningServicesError("expired or inherited private planning selection refused")
    selected.services._process()
    return selected.services
