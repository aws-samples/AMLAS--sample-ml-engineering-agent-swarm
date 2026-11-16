"""Private read-only composition over the original protected planning publication."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Protocol

from .campaign_initialization import InitializationPlanningObservation
from .campaign_planning import (
    ActivePlanningSelection,
    PlanningBinding,
    ProtectedPlanningOriginal,
    ProtectedPlanningRead,
    _json,
    validate_original_planning,
)
from .campaign_runtime import CampaignEntryContext, CampaignPlanningRef, CampaignRuntimeFactory
from .model_campaign_binding import ModelPlanningBody


class ProtectedPlanningReadError(ValueError):
    """Absent or unqualified original planning; no action permission."""


class InitializationPlanningPort(Protocol):
    """Root supplies actual current ingress/witness/transport and native accessors."""

    def read_current_scope(self, *, qualify_io) -> dict: ...
    def read_original_planning(
        self, *, campaign_ref, private_store_correlation, qualify_io
    ) -> InitializationPlanningObservation: ...


@dataclass(frozen=True)
class ProtectedPlanningView:
    binding: PlanningBinding
    selection: ActivePlanningSelection
    original: ProtectedPlanningOriginal
    raw_text: str = field(repr=False)
    canonical_spec: bytes = field(repr=False)
    refs: tuple[CampaignPlanningRef, ...]


class ProtectedCampaignPlanningReader:
    """No ENTRY/generation constructor, publication or native supplier default."""

    def __init__(
        self,
        *,
        initialization_port=None,
        protected_port=None,
        origin_qualifier=None,
        qualify_current=None,
    ):
        if (
            not callable(origin_qualifier)
            or not callable(qualify_current)
            or not all(
                callable(getattr(initialization_port, name, None))
                for name in ("read_current_scope", "read_original_planning")
            )
            or not all(
                callable(getattr(protected_port, name, None))
                for name in ("require_current", "read_active", "read_original")
            )
        ):
            raise ProtectedPlanningReadError(
                "explicit current original initialization/protected ports required"
            )
        self._initialization, self._protected = initialization_port, protected_port
        self._origin_qualifier, self._qualify_current = origin_qualifier, qualify_current
        self._pid = os.getpid()
        self._gate()

    def _gate(self, outer=None):
        if self._pid != os.getpid():
            raise ProtectedPlanningReadError("inherited protected planning reader refused")
        if self._qualify_current() is not None:
            raise ProtectedPlanningReadError("current planning qualifier grants nothing")
        if outer is not None:
            if not callable(outer) or outer() is not None:
                raise ProtectedPlanningReadError("outer planning IO qualifier grants nothing")

    def _call(self, port, method, *, outer=None, **kwargs):
        self._gate(outer)
        try:
            return getattr(port, method)(qualify_io=lambda: self._gate(outer), **kwargs)
        finally:
            self._gate(outer)

    def _initial(self, campaign_ref, private_store_correlation, outer=None):
        scope = self._call(self._initialization, "read_current_scope", outer=outer)
        if (
            type(scope) is not dict
            or set(scope) != {"account_id", "region", "deployment_id", "deployment_prefix"}
            or any(type(value) is not str for value in scope.values())
        ):
            raise ProtectedPlanningReadError("closed exact current planning scope DATA required")
        observed = self._call(
            self._initialization,
            "read_original_planning",
            outer=outer,
            campaign_ref=campaign_ref,
            private_store_correlation=private_store_correlation,
        )
        if type(observed) is not InitializationPlanningObservation:
            raise ProtectedPlanningReadError("exact qualified original initialization observation required")
        observed.__post_init__()
        if (
            observed.original.campaign_ref != campaign_ref
            or observed.original.private_store_correlation != private_store_correlation
            or observed.native_admission is not False
            or observed.action_approved is not False
        ):
            raise ProtectedPlanningReadError("original initialization planning scope differs")
        binding = PlanningBinding(
            **scope,
            campaign_ref=observed.original.campaign_ref,
            campaign_id=observed.original.campaign_id,
            private_store_correlation=observed.original.private_store_correlation,
            prompt_sha256=observed.original.prompt_sha256,
            initial_campaign_body_sha256=observed.initial_campaign_body_sha256,
            initialization_publication_sha256=observed.initialization_publication_sha256,
        )
        binding.validate()
        return binding, observed

    def _origin(self, binding, observed, selection, original, outer=None):
        self._gate(outer)
        try:
            evidence = self._protected.require_current(
                binding,
                selection=selection,
                original=original,
                qualify_io=lambda: self._gate(outer),
            )
        finally:
            self._gate(outer)
        if (
            self._origin_qualifier(
                self._initialization, self._protected, binding, observed, selection, original, evidence
            )
            is not None
        ):
            raise ProtectedPlanningReadError("original planning origin qualifier grants nothing")
        self._gate(outer)

    @staticmethod
    def _refs(value):
        refs = [CampaignPlanningRef("spec", value["original_ref"], value["spec_sha256"])]
        for row in value["research"]:
            digest = hashlib.sha256(_json(row)).hexdigest()
            refs.append(CampaignPlanningRef("research", "research_" + digest[:32], digest))
        return tuple(refs)

    def _read(self, campaign_ref, private_store_correlation, reference=None, outer=None):
        binding, observed = self._initial(campaign_ref, private_store_correlation, outer)
        self._origin(binding, observed, None, None, outer)  # Configuration is not ABSENT.
        self._gate(outer)
        try:
            if reference is None:
                read = self._protected.read_active(binding, qualify_io=lambda: self._gate(outer))
            else:
                read = self._protected.read_original(
                    binding, reference=reference, qualify_io=lambda: self._gate(outer)
                )
        finally:
            self._gate(outer)
        if type(read) is not ProtectedPlanningRead:
            raise ProtectedPlanningReadError("explicit protected original read required")
        read.validate()
        if read.status == "UNKNOWN":
            raise ProtectedPlanningReadError("original protected planning existence unresolved")
        self._origin(binding, observed, read.selection, read.original, outer)
        if read.status == "ABSENT":
            current_binding, current = self._initial(campaign_ref, private_store_correlation, outer)
            if current_binding != binding:
                raise ProtectedPlanningReadError("original planning scope changed during absence read")
            self._origin(current_binding, current, None, None, outer)
            return None
        value = validate_original_planning(binding, read.original)
        if reference is not None and value["original_ref"] != reference:
            raise ProtectedPlanningReadError("original selected planning reference differs")
        canonical = _json(value["spec_projection"], 65536)
        if hashlib.sha256(canonical).hexdigest() != value["spec_sha256"]:
            raise ProtectedPlanningReadError("original canonical model spec bytes differ")
        view = ProtectedPlanningView(
            binding, read.selection, read.original, value["raw_text"], canonical, self._refs(value)
        )
        current_binding, current = self._initial(campaign_ref, private_store_correlation, outer)
        if current_binding != binding:
            raise ProtectedPlanningReadError("original planning scope changed during body read")
        self._origin(current_binding, current, view.selection, view.original, outer)
        self._gate(outer)
        return view

    def load_original_refs(self, *, campaign_ref, private_store_correlation):
        view = self._read(campaign_ref, private_store_correlation)
        return () if view is None else view.refs

    def load_original(
        self,
        *,
        campaign_ref,
        private_store_correlation,
        reference=None,
        qualify_io=None,
    ):
        if not callable(qualify_io):
            raise ProtectedPlanningReadError("fresh original body IO qualifier required")
        view = self._read(campaign_ref, private_store_correlation, reference, qualify_io)
        if view is None:
            raise ProtectedPlanningReadError("original protected planning body absent")
        return view

    def require_current(self, view, *, qualify_io):
        if type(view) is not ProtectedPlanningView:
            raise ProtectedPlanningReadError("original qualified planning view DATA required")
        fresh = self.load_original(
            campaign_ref=view.binding.campaign_ref,
            private_store_correlation=view.binding.private_store_correlation,
            reference=view.original.ack.reference,
            qualify_io=qualify_io,
        )
        if fresh != view:
            raise ProtectedPlanningReadError("independently current original planning changed")
        return fresh


class ModelPlanningPortAdapter:
    """Only the sealed entry-selected original body, in the finite canonical format."""

    def __init__(self, reader):
        if type(reader) is not ProtectedCampaignPlanningReader:
            raise ProtectedPlanningReadError("exact protected planning reader required")
        self._reader, self._pid = reader, os.getpid()

    def _entry(self, entry, qualify_io):
        if (
            self._pid != os.getpid()
            or type(entry) is not CampaignEntryContext
            or type(entry._factory) is not CampaignRuntimeFactory
            or not callable(qualify_io)
        ):
            raise ProtectedPlanningReadError("current sealed original model entry required")
        entry._check()
        entry.qualify_current_io()
        if qualify_io() is not None:
            raise ProtectedPlanningReadError("model planning callback grants nothing")
        specs = [ref for ref in entry.planning_refs if ref.kind == "spec"]
        if len(specs) != 1:
            raise ProtectedPlanningReadError("one exact original spec ref required")
        return specs[0]

    def load_original(self, entry, *, qualify_io):
        ref = self._entry(entry, qualify_io)
        view = self._reader.load_original(
            campaign_ref=entry.campaign_ref,
            private_store_correlation=entry.private_store_correlation,
            reference=ref.reference,
            qualify_io=qualify_io,
        )
        if view.refs[0] != ref:
            raise ProtectedPlanningReadError("entry-selected canonical original spec differs")
        self._entry(entry, qualify_io)
        return ModelPlanningBody(ref.reference, ref.sha256, view.canonical_spec, "canonical_spec_json_v1")

    def require_current(self, body, entry, *, qualify_io):
        ref = self._entry(entry, qualify_io)
        if type(body) is not ModelPlanningBody or body.body_format != "canonical_spec_json_v1":
            raise ProtectedPlanningReadError("exact canonical model planning body DATA required")
        view = self._reader.load_original(
            campaign_ref=entry.campaign_ref,
            private_store_correlation=entry.private_store_correlation,
            reference=ref.reference,
            qualify_io=qualify_io,
        )
        if (
            body.reference != ref.reference
            or body.sha256 != ref.sha256
            or body.body != view.canonical_spec
            or view.refs[0] != ref
        ):
            raise ProtectedPlanningReadError("independently protected model planning body differs")
        self._entry(entry, qualify_io)
        return view

    def qualify_model_origin(self, port, body, entry, evidence):
        if port is not self or type(evidence) is not ProtectedPlanningView:
            raise ProtectedPlanningReadError("same private model planning origin required")
        fresh = self.require_current(body, entry, qualify_io=entry.qualify_current_io)
        if fresh != evidence:
            raise ProtectedPlanningReadError("current protected planning evidence changed")
