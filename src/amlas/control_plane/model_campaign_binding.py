"""Conditional private model bridge; original DATA never grants native effects."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ..backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
from ..backends.types import JobHandle, JobState
from ..data_eng.model_weight_manifest import expected_model_weight_manifest
from ..spec.loader import load_spec_from_text
from .approvals import ApprovalPolicy, request_approval
from .campaign_runtime import CampaignEntryContext, CampaignRuntimeFactory
from .model_campaign import (
    DynamoModelCampaignJournal,
    ModelAcquisitionConfig,
    ModelOperationReservation,
    _json,
    _parse,
    prepare_model_intent,
)

_SERVICES = ContextVar("private_model_campaign_services", default=None)
_MAX_SPEC = 256 * 1024


class ModelCampaignBindingError(ValueError):
    """Current private first-model source boundary is absent or changed."""


@dataclass(frozen=True)
class ModelPlanningBody:
    reference: str
    sha256: str
    body: bytes = field(repr=False)
    body_format: str = "frontmatter_v1"


@dataclass(frozen=True)
class ModelSubmissionResult:
    reference: str
    handle: JobHandle
    disposition: str
    executed: bool
    native_admission: bool = field(default=False, init=False)


class ModelSubmissionAttemptError(ModelCampaignBindingError):
    """Closed historical attempt DATA; absence of current qualification withholds IDs."""

    def __init__(self, *, dispatch_attempted, currently_qualified, result=None):
        if (
            (dispatch_attempted is not None and type(dispatch_attempted) is not bool)
            or type(currently_qualified) is not bool
            or (result is not None and type(result) is not ModelSubmissionResult)
            or (not currently_qualified and result is not None)
            or (result is not None and (dispatch_attempted is not True or result.executed is not True))
        ):
            raise ModelCampaignBindingError("closed original model attempt error data required")
        super().__init__("original model submission outcome unresolved")
        self._dispatch_attempted = dispatch_attempted
        self._currently_qualified = currently_qualified
        self._result = result

    @property
    def dispatch_attempted(self):
        return self._dispatch_attempted

    @property
    def currently_qualified(self):
        return self._currently_qualified

    @property
    def result(self):
        return self._result


class _ModelPreDispatchRefusalError(ModelCampaignBindingError):
    """Issued only by the exact adapter's callback before native Create."""


def _planning_spec(body):
    if type(body.body_format) is not str:
        raise ModelCampaignBindingError("exact protected planning format required")
    if body.body_format == "frontmatter_v1":
        spec = load_spec_from_text(body.body.decode("utf-8"))
    elif body.body_format == "canonical_spec_json_v1":
        from ..arbiter.coding import coding_spec_bytes
        from ..spec.schema import SpecModel

        spec = SpecModel.model_validate(_parse(body.body))
        if coding_spec_bytes(spec) != body.body:
            raise ModelCampaignBindingError("canonical validated planning body differs")
    else:
        raise ModelCampaignBindingError("protected planning format refused")
    return spec


class ModelCampaignServices:
    def __init__(
        self,
        *,
        runtime_factory,
        planning_port,
        planning_origin_qualifier,
        config_port,
        config_origin_qualifier,
        wake_hint_port,
        wake_hint_origin_qualifier,
        journal_factory,
        current_boundary_qualifier,
    ):
        if type(runtime_factory) is not CampaignRuntimeFactory or not all(
            callable(value)
            for value in (
                planning_origin_qualifier,
                config_origin_qualifier,
                wake_hint_origin_qualifier,
                journal_factory,
                current_boundary_qualifier,
            )
        ):
            raise ModelCampaignBindingError("configured private model services required")
        for port, names in (
            (planning_port, ("load_original", "require_current")),
            (config_port, ("read_current",)),
            (wake_hint_port, ("publish_original_wake_binding", "require_current")),
        ):
            if not all(callable(getattr(port, name, None)) for name in names):
                raise ModelCampaignBindingError("complete private model source ports required")
        self._runtime, self._planning, self._config, self._hint = (
            runtime_factory,
            planning_port,
            config_port,
            wake_hint_port,
        )
        self._planning_origin, self._config_origin, self._hint_origin = (
            planning_origin_qualifier,
            config_origin_qualifier,
            wake_hint_origin_qualifier,
        )
        self._journal_factory, self._boundary = journal_factory, current_boundary_qualifier
        self._pid = os.getpid()

    def _process(self):
        if os.getpid() != self._pid:
            raise ModelCampaignBindingError("inherited private model services refused")

    def _entry(self, entry):
        self._process()
        if type(entry) is not CampaignEntryContext or entry._factory is not self._runtime:
            raise ModelCampaignBindingError("exact same-factory original model entry required")
        entry.require_current()

    def build_for_entry(self, entry, adapter):
        self._entry(entry)
        if type(adapter) is not ProcessingModelWeightAcquisitionBackend or not callable(
            getattr(type(adapter), "qualified_io", None)
        ):
            raise ModelCampaignBindingError("actual qualified model adapter required")
        return ModelCampaignModelBinding(self, entry, adapter, _token=_BINDING_SEAL)


@contextmanager
def activate_model_campaign_services(services):
    if type(services) is not ModelCampaignServices:
        raise ModelCampaignBindingError("exact private model services required")
    services._process()
    token = _SERVICES.set(services)
    try:
        yield services
    finally:
        _SERVICES.reset(token)


def current_model_campaign_services():
    services = _SERVICES.get()
    if services is not None:
        if type(services) is not ModelCampaignServices:
            raise ModelCampaignBindingError("private model activation differs")
        services._process()
    return services


_BINDING_SEAL = object()


class ModelCampaignModelBinding:
    def __init__(self, services, entry, adapter, *, _token=None):
        if _token is not _BINDING_SEAL or type(services) is not ModelCampaignServices:
            raise ModelCampaignBindingError("source-created private model binding required")
        services._entry(entry)
        self._services, self.entry, self.adapter = services, entry, adapter
        self._pid, self._lock = os.getpid(), threading.RLock()
        self._handles = {}
        self._retained_intent = None
        self._planning = self._read_planning()
        self._config = self._read_config()
        self._callback()
        journal = services._journal_factory(entry, adapter, self._config)
        if (
            type(journal) is not DynamoModelCampaignJournal
            or journal.entry is not entry
            or journal.coordinator is not services._runtime.coordinator
            or journal._table != services._runtime.coordinator._table
            or getattr(journal._validator, "__self__", None) is not adapter
            or getattr(journal._validator, "__func__", None)
            is not ProcessingModelWeightAcquisitionBackend.validate_prepared
        ):
            raise ModelCampaignBindingError("exact current model journal/adapter wiring required")
        self.journal = journal
        self._full_current()

    def _process(self):
        if self._pid != os.getpid():
            raise ModelCampaignBindingError("inherited original model binding refused")
        self._services._process()

    def _basic(self):
        self._process()
        self.entry.qualify_current_io()

    def _read_planning(self):
        self._basic()
        body = self._services._planning.load_original(self.entry, qualify_io=self._basic)
        self._basic()
        if (
            type(body) is not ModelPlanningBody
            or type(body.body) is not bytes
            or not 0 < len(body.body) <= _MAX_SPEC
            or type(body.reference) is not str
            or type(body.sha256) is not str
            or hashlib.sha256(body.body).hexdigest() != body.sha256
        ):
            raise ModelCampaignBindingError("bounded original planning body required")
        matches = [
            ref
            for ref in self.entry.planning_refs
            if ref.kind == "spec" and ref.reference == body.reference and ref.sha256 == body.sha256
        ]
        if len(matches) != 1 or sum(ref.kind == "spec" for ref in self.entry.planning_refs) != 1:
            raise ModelCampaignBindingError("exact original spec pointer/body differs")
        # Reuse existing specification parser; no executable planning codec.
        spec = _planning_spec(body)
        if spec.task_type != "text_generation" or spec.generation is None or spec.generation.model is None:
            raise ModelCampaignBindingError("validated original coding specification required")
        evidence = self._services._planning.require_current(body, self.entry, qualify_io=self._basic)
        self._basic()
        if self._services._planning_origin(self._services._planning, body, self.entry, evidence) is not None:
            raise ModelCampaignBindingError("planning origin qualifier grants nothing")
        self._basic()
        return body

    def _read_config(self):
        self._basic()
        config = self._services._config.read_current(self.entry, self.adapter, qualify_io=self._basic)
        self._basic()
        if type(config) is not ModelAcquisitionConfig:
            raise ModelCampaignBindingError("current explicit model configuration required")
        config.validate()
        if self._services._config_origin(self._services._config, config, self.entry) is not None:
            raise ModelCampaignBindingError("configuration origin qualifier grants nothing")
        self._basic()
        return config

    def _callback(self):
        """Pure current boundary callback: no DDB/S3 reads or cached grants."""
        self._basic()
        if self._services._boundary(self.entry, self.adapter, self._config, self._planning) is not None:
            raise ModelCampaignBindingError("current model boundary qualifier grants nothing")
        self._basic()
        # The last runtime I/O may revoke model-specific configuration/origin.
        if self._services._boundary(self.entry, self.adapter, self._config, self._planning) is not None:
            raise ModelCampaignBindingError("current model boundary qualifier grants nothing")

    def _full_current(self):
        self._process()
        self._services._entry(self.entry)
        if self._read_planning() != self._planning or self._read_config() != self._config:
            raise ModelCampaignBindingError("original model planning/configuration changed")
        self._callback()

    def _intent(self, catalog_id, catalog_version):
        self._full_current()
        expected = expected_model_weight_manifest(catalog_id, catalog_version)
        spec = _planning_spec(self._planning)
        model = spec.generation.model
        if (model.repo_id, model.revision) != (expected["repo_id"], expected["revision"]):
            raise ModelCampaignBindingError("original coding model/catalog selection differs")
        return prepare_model_intent(
            self.entry,
            entry_qualifier=lambda entry: entry.require_current(),
            planning_ref=self._planning.reference,
            spec_sha256=self._planning.sha256,
            config=self._config,
            prepare_submission=self.adapter.prepare_submission,
            validate_prepared=self.adapter.validate_prepared,
            catalog_id=catalog_id,
            catalog_version=catalog_version,
        )

    @staticmethod
    def _handle(intent):
        wire = _parse(intent.canonical_bytes)["prepared"]
        return JobHandle(wire["id"], wire["backend"], wire["arn"], _parse(_json(wire["extra"])))

    @staticmethod
    def _approve(policy):
        if type(policy) is not ApprovalPolicy:
            raise ModelCampaignBindingError("typed fresh caller model policy required")
        snapshot = dataclasses.replace(policy, allowlist=list(policy.allowlist))
        decision = request_approval(
            action="run_processing_job",
            cost_estimate=None,
            impact="fixed public model-weight Processing acquisition; exact cost quote unavailable",
            policy=snapshot,
        )
        if decision["decision"] != "approved":
            raise ModelCampaignBindingError("current caller model action refused")

    def _hint(self, intent, reservation):
        from ..async_callback.lambda_handler import encode_wake_job_binding, wake_job_lookup_key

        data = _parse(intent.canonical_bytes)
        config = data["config"]
        body = {
            "schema_version": 1,
            "record_kind": "campaign_wake_job_v1",
            "account_id": config["account_id"],
            "region": config["region"],
            "deployment_id": config["deployment_id"],
            "kind": "Processing",
            "job_name": reservation.job_name,
            "job_arn": reservation.job_arn,
            "campaign_ref": self.entry.campaign_ref,
            "operation_digest": data["operation_digest"],
            "transport_session_id": self.entry.transport_session_id,
        }
        key, expected = wake_job_lookup_key("Processing", reservation.job_name), encode_wake_job_binding(body)
        self._full_current()
        try:
            acknowledged = self._services._hint.publish_original_wake_binding(
                intent, reservation, self.entry, qualify_io=self._callback
            )
        except Exception:
            self._callback()
            raise ModelCampaignBindingError("original model wake hint publication unresolved") from None
        self._callback()
        if type(acknowledged) is not bytes or acknowledged != expected:
            raise ModelCampaignBindingError("original model wake hint acknowledgement unresolved")
        evidence = self._services._hint.require_current(
            intent, reservation, self.entry, key, expected, acknowledged, qualify_io=self._callback
        )
        if (
            self._services._hint_origin(
                self._services._hint, intent, reservation, self.entry, key, expected, acknowledged, evidence
            )
            is not None
        ):
            raise ModelCampaignBindingError("wake hint origin qualifier grants nothing")
        self._full_current()

    def _remember_intent(self, intent):
        if (
            self._retained_intent is not None
            and self._retained_intent.canonical_bytes != intent.canonical_bytes
        ):
            raise ModelCampaignBindingError("retained original model selection differs")
        self._retained_intent = intent

    def require_current_exposure(self, reference=None):
        """Current private model qualification; no flags, IDs, policy or grants."""
        self._process()
        with self._lock:
            self._full_current()
            with self.journal.qualified_io(self._callback):
                if self._retained_intent is None:
                    if reference is not None:
                        raise ModelCampaignBindingError("original model exposure reference absent")
                    self.journal._origin(None, None)
                    self._callback()
                    return
                reservation = self.journal.load_original(self._planning.reference)
                if reference is not None and (
                    type(reference) is not str or reference != reservation.model_ref
                ):
                    raise ModelCampaignBindingError("same original exposure reference required")
                data = _parse(self._retained_intent.canonical_bytes)
                current = self._intent(data["catalog_id"], data["catalog_version"])
                if (
                    current.canonical_bytes != reservation.intent.canonical_bytes
                    or current != self._retained_intent
                ):
                    raise ModelCampaignBindingError("original model exposure intent/row differs")
                self.journal._protected_original(current)
                self._original_handle(current)
                self._full_current()

    def _attempt_error(self, dispatch_attempted, result):
        try:
            self.require_current_exposure(result.reference if result is not None else None)
        except Exception:
            return ModelSubmissionAttemptError(
                dispatch_attempted=dispatch_attempted, currently_qualified=False, result=None
            )
        return ModelSubmissionAttemptError(
            dispatch_attempted=dispatch_attempted, currently_qualified=True, result=result
        )

    def _original_handle(self, intent):
        data = _parse(intent.canonical_bytes)
        reference = data["model_ref"]
        expected = self._handle(intent)
        handle = self._handles.get(reference)
        if handle is None:
            self._handles[reference] = expected
            return expected
        if type(handle) is not JobHandle or (handle.id, handle.backend, handle.arn) != (
            expected.id,
            expected.backend,
            expected.arn,
        ):
            raise ModelCampaignBindingError("retained original model identity differs")
        observational = {"accepted", "submission_ambiguous", "acquired", "native_status"}
        retained = {key: handle.extra.get(key) for key in expected.extra if key not in observational}
        original = {key: expected.extra[key] for key in expected.extra if key not in observational}
        if _json(retained) != _json(original):
            raise ModelCampaignBindingError("retained original model request/source differs")
        return handle

    def observe_original(self, reference):
        """Read-only original adapter artifacts under fresh private qualification."""
        self._process()
        with self._lock:
            self._full_current()
            with self.journal.qualified_io(self._callback):
                reservation = self.journal.load_original(self._planning.reference)
                if type(reference) is not str or reference != reservation.model_ref:
                    raise ModelCampaignBindingError("same original model reference required")
                original = _parse(reservation.intent.canonical_bytes)
                current = self._intent(original["catalog_id"], original["catalog_version"])
                if current.canonical_bytes != reservation.intent.canonical_bytes:
                    raise ModelCampaignBindingError("original model consumer selection differs")
                self.journal._protected_original(current)
                self._remember_intent(current)
                handle = self._original_handle(current)
                with self.adapter.qualified_io(self._callback):
                    artifacts = self.adapter.artifacts(handle)
                self._full_current()
                self.journal._protected_original(current)
                bounded = _parse(_json(artifacts))
                if type(bounded) is not dict or (bounded.get("job_name"), bounded.get("job_arn")) != (
                    reservation.job_name,
                    reservation.job_arn,
                ):
                    raise ModelCampaignBindingError("original model artifact observation identity differs")
                bounded["request_ref"] = reference
                self._full_current()
                return bounded

    def _observe(self, intent, reservation):
        self._remember_intent(intent)
        self._full_current()
        self.adapter.validate_prepared(self._handle(intent))
        handle = self._original_handle(intent)

        def description(original):
            self._full_current()
            with self.adapter.qualified_io(self._callback):
                status = self.adapter.status(handle)
            self._full_current()
            native_status = {
                JobState.RUNNING: "InProgress",
                JobState.SUCCEEDED: "Completed",
                JobState.FAILED: "Failed",
                JobState.STOPPED: "Stopped",
            }.get(status.state, "Unknown")
            return {
                "status": native_status,
                "job_name": original.job_name,
                "job_arn": original.job_arn,
                "request_sha256": original.request_sha256,
                "source_binding_sha256": original.source_binding_sha256,
            }

        if reservation is not None:
            observed = self.journal.restore_original(intent, describe_original=description)
            disposition = (
                reservation.state if reservation.state in {"ACCEPTED", "UNKNOWN", "REJECTED"} else "UNKNOWN"
            )
            if observed.status == "UNKNOWN":
                disposition = "UNKNOWN"
        else:
            # Protected original exists without ordinary reservation: never reserve/Create.
            with self.adapter.qualified_io(self._callback):
                self.adapter.status(handle)
            disposition = "UNKNOWN"
        self._full_current()
        handle.extra["acquired"] = False
        if disposition == "UNKNOWN":
            handle.extra["submission_ambiguous"] = True
        data = _parse(intent.canonical_bytes)
        return ModelSubmissionResult(data["model_ref"], handle, disposition, False)

    def submit_original(self, catalog_id, catalog_version, policy):
        self._process()
        with self._lock:
            intent = self._intent(catalog_id, catalog_version)
            self._approve(policy)
            with self.journal.qualified_io(self._callback):
                original = self.journal.inspect_original(intent)
                if original.disposition == "PRESENT":
                    return self._observe(intent, original.reservation)
                publication = self.journal.publish_original(intent)
                self._remember_intent(intent)
                outcome = self.journal.reserve_for_submission(intent, publication=publication)
                reservation = outcome.reservation
                if outcome.disposition != "NEW_CONFIRMED":
                    return self._observe(intent, reservation)
                self._full_current()
                record = self.entry.load_original_campaign()
                lease = self.entry.claim_original_lease(expected_revision=record.revision)
                reservation = self.journal.register_original(reservation, lease=lease)
                self._hint(intent, reservation)
                permit = self.journal.consume_create_attempt(reservation, lease=lease)
                if type(permit) is ModelOperationReservation:
                    return self._observe(intent, permit)
                handle = self._handle(intent)

                def effect():
                    try:
                        self._full_current()
                        self._approve(policy)
                        self.journal.require_effect(permit, reservation)
                    except Exception:
                        # Actual source adapter invokes this callback before Create,
                        # outside its native acceptance try. Permit use is not proof.
                        raise _ModelPreDispatchRefusalError(
                            "current pre-dispatch model effect refused"
                        ) from None

                submitted = None
                dispatch_attempted = None
                snapshot = None
                try:
                    with self.adapter.qualified_io(self._callback):
                        submitted = self.adapter.submit_prepared(handle, before_create=effect)
                    if type(submitted) is not JobHandle or not (
                        submitted.extra.get("accepted") is True
                        or submitted.extra.get("submission_ambiguous") is True
                    ):
                        raise ModelCampaignBindingError("actual source model dispatch outcome unknown")
                    dispatch_attempted = True
                    data = _parse(intent.canonical_bytes)
                    self._handles[data["model_ref"]] = submitted
                    snapshot = ModelSubmissionResult(data["model_ref"], submitted, "UNKNOWN", True)
                    self._full_current()
                    disposition = "ACCEPTED" if submitted.extra.get("accepted") is True else "UNKNOWN"
                    self.journal.record_outcome(
                        reservation,
                        permit=permit,
                        disposition=disposition,
                        native_job_arn=submitted.arn if disposition == "ACCEPTED" else None,
                        error_type=None if disposition == "ACCEPTED" else "SubmissionUnknown",
                    )
                    self._full_current()
                    submitted.extra["acquired"] = False
                    return ModelSubmissionResult(data["model_ref"], submitted, disposition, True)
                except _ModelPreDispatchRefusalError:
                    raise self._attempt_error(False, None) from None
                except Exception:
                    # No returned actual source outcome means opaque uncertainty,
                    # never evidence from a consumed marker or arbitrary exception.
                    if (
                        submitted is not None
                        and type(submitted) is JobHandle
                        and (
                            submitted.extra.get("accepted") is True
                            or submitted.extra.get("submission_ambiguous") is True
                        )
                    ):
                        dispatch_attempted = True
                        data = _parse(intent.canonical_bytes)
                        self._handles[data["model_ref"]] = submitted
                        snapshot = ModelSubmissionResult(data["model_ref"], submitted, "UNKNOWN", True)
                    raise self._attempt_error(dispatch_attempted, snapshot) from None
