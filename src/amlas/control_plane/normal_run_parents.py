"""Closed normal-run parent DATA; native-bound publication is checked by its caller."""

from __future__ import annotations

import json
import re
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict

from ..backends.types import JobHandle
from .campaign_state import _binding_fields
from .producer_snapshot import (
    _approved_read,
    _capture,
    _keys,
    _producer,
    _ref,
    _ref_wire,
    _source,
    _source_wire,
)

_MAX = 1024 * 1024
_DATA_NATIVE_KEYS = (
    "ProcessingJobName",
    "ProcessingJobArn",
    "RoleArn",
    "AppSpecification",
    "ProcessingResources",
    "ProcessingOutputConfig",
    "StoppingCondition",
    "Environment",
)
_DATA_FIELDS = (
    "repo_id",
    "revision",
    "filename",
    "bucket",
    "output_prefix",
    "output_uri",
    "region",
    "role_arn",
    "account_id",
    "image_uri",
    "image_digest",
    "source_commit",
    "runtime_session_id",
    "deployment_id",
    "deployment_prefix",
    "egress_environment",
)


def _encode(value):
    from .normal_run_checkpoint import _checkpoint_data

    _checkpoint_data(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if not 0 < len(raw) <= _MAX:
        raise ValueError("normal parent graph byte bound exceeded")
    return raw


def _decode(raw):
    from .normal_run_checkpoint import _pairs

    if type(raw) is not bytes or not 0 < len(raw) <= _MAX:
        raise ValueError("bounded parent graph bytes required")
    value = json.loads(raw, object_pairs_hook=_pairs)
    if _encode(value) != raw:
        raise ValueError("normal parent graph canonical bytes differ")
    return value


def _snapshot(value):
    return json.loads(json.dumps(asdict(value), sort_keys=True, allow_nan=False))


def _data_handle(row, scope):
    from ..backends.adapters.dataset_acquisition import validate_source

    _keys(row, {"reference", "id", "arn", "extra", "source", "native_request"})
    extra = row["extra"]
    _keys(extra, set(_DATA_FIELDS) | {"privilege_profile", "receipt_schema_version"})
    if (
        not re.fullmatch("acq_[a-f0-9]{32}", row["reference"])
        or extra["runtime_session_id"] != scope["runtime_session_id"]
        or extra["account_id"] != scope["account_id"]
        or extra["region"] != scope["region"]
        or extra["deployment_id"] != scope["deployment_id"]
        or extra["deployment_prefix"] != scope["deployment_prefix"]
        or not row["id"].startswith(scope["deployment_prefix"] + "-")
        or row["arn"]
        != "arn:aws:sagemaker:" + scope["region"] + ":" + scope["account_id"] + ":processing-job/" + row["id"]
        or extra["output_prefix"]
        != "sessions/" + scope["runtime_session_id"] + "/acquisition/" + row["id"] + "/"
        or extra["output_uri"] != "s3://" + extra["bucket"] + "/" + extra["output_prefix"]
    ):
        raise ValueError("original acquisition namespace differs")
    validate_source(extra["repo_id"], extra["revision"], extra["filename"])
    _source(row["source"])
    decoded = dict(extra)
    if decoded["privilege_profile"] is None:
        decoded.pop("privilege_profile")
        decoded.pop("receipt_schema_version")
    decoded.update(accepted=False, acquired=False, native_status="RestoreUnverified", observation_error="")
    return JobHandle(row["id"], "processing", row["arn"], decoded)


def _case_handle(row, cases, scope):
    from .coding_case_state import producer_header

    _keys(
        row,
        {
            "reference",
            "acquisition_ref",
            "id",
            "arn",
            "catalog",
            "header",
            "source",
            "request",
            "output_prefix",
            "binding_ref",
            "files",
        },
    )
    catalog = cases._catalog
    source = _source(row["source"])
    if (
        row["catalog"] != asdict(catalog)
        or not re.fullmatch("case_[a-f0-9]{32}", row["reference"])
        or row["header"] != producer_header(catalog, scope["runtime_session_id"], row["id"])
        or row["arn"]
        != "arn:aws:sagemaker:" + scope["region"] + ":" + scope["account_id"] + ":processing-job/" + row["id"]
        or row["output_prefix"]
        != "sessions/" + scope["runtime_session_id"] + "/case-freeze/" + row["id"] + "/output/"
        or row["request"].get("ProcessingJobName") != row["id"]
        or row["request"].get("RoleArn") != catalog.execution_role_arn
        or row["request"].get("AppSpecification", {}).get("ImageUri") != catalog.image_uri
    ):
        raise ValueError("original case catalog/request differs")
    return JobHandle(
        row["id"],
        "case_freeze",
        row["arn"],
        {
            "catalog": catalog,
            "source": source,
            "header": row["header"],
            "request": row["request"],
            "runtime_session_id": scope["runtime_session_id"],
            "output_prefix": row["output_prefix"],
            "binding_ref": _ref(row["binding_ref"]),
            "submission_state": "ambiguous",
        },
    )


def _adopted_model_parent(value):
    if type(value) is dict and "origin" in value:
        _keys(value, {"origin", "request_ref", "snapshot"})
        if (
            value["origin"] != "verified_pack_adoption"
            or type(value["request_ref"]) is not str
            or type(value["snapshot"]) is not dict
        ):
            raise ValueError("closed adopted model parent required")
        return True
    return False


def _adopted_sft_snapshot(models, reference, catalog, session, *, qualify):
    from .sft_state import VerifiedSftAdoptedModelSource, validate_model

    source = models._verified_sft_model_snapshot(reference, _qualify=qualify)
    if type(source) is not VerifiedSftAdoptedModelSource:
        raise ValueError("typed adopted SFT model parent required")
    validate_model(source, catalog, session)
    return _snapshot(source)


def publish_parent_graph(store, model_ref, prep_ref, *, run_id, qualify=None):
    """Capture genuine original aliases/artifacts before the SFT intent exists."""
    from .normal_run_checkpoint import current_normal_run
    from .sft_state import SessionSftStore

    context = current_normal_run()
    if type(store) is not SessionSftStore or context is None or context.run_id != run_id:
        raise ValueError("current normal original store required")
    if not callable(qualify):
        raise ValueError("current normal parent IO qualifier required")
    call = context._captured_sft_call(model_ref, prep_ref)
    binding = store._execution_binding
    scope = _binding_fields(binding)
    qualify()
    adoption_hook = getattr(store._models, "_adopted_parent_snapshot", None)
    adoption = adoption_hook(model_ref, binding, qualify=qualify) if callable(adoption_hook) else None
    adopted = adoption is not None
    model = None if adopted else store._models._original_producer_handle(model_ref, binding)
    model_parent = (
        {"origin": "verified_pack_adoption", "request_ref": model_ref, "snapshot": adoption}
        if adopted
        else _capture(model, model_ref, model=True)
    )
    if adopted:
        _adopted_model_parent(model_parent)
    prep = store._prepared._original_producer_handle(prep_ref, binding)
    acquired, cases = store._prepared._acquired, store._prepared._case_store
    if cases is None:
        raise ValueError("normal coding parent requires original frozen cases")
    with store._prepared._lock:
        selections = [key for key, alias in store._prepared._submissions.items() if alias == prep_ref]
    with cases._lock:
        case_keys = list(cases._keys.items())
    if len(selections) != 1 or len(case_keys) != 1:
        raise ValueError("one original case-inclusive preparation selection required")
    model_record = None
    if not adopted:
        with store._models._lock:
            model_record = store._models._records.get(model_ref)
    with store._prepared._lock:
        prep_record = store._prepared._records.get(prep_ref)
    with acquired._lock:
        original_data = tuple(acquired._records.values())
    if (not adopted and model_record is None) or prep_record is None or not 1 <= len(original_data) <= 16:
        raise ValueError("original parent adapters missing")
    adapters = []
    for adapter in (
        *((model_record.adapter,) if not adopted else ()),
        prep_record.adapter,
        cases._adapter,
        *(record.adapter for record in original_data),
    ):
        if not any(adapter is retained for retained in adapters):
            adapters.append(adapter)
    qualify()
    with _qualified_reads(adapters, scope["region"], qualify):
        case_key, case_ref = case_keys[0]
        handle, result, refs = cases._verified(case_ref)
        if refs is None or result.get("frozen") is not True:
            raise ValueError("fresh original frozen cases required")
        with acquired._lock:
            records = sorted(acquired._records.items())
        if not 1 <= len(records) <= 16:
            raise ValueError("bounded original acquired parent population required")
        data = []
        for alias, original in records:
            source = acquired._verified_preparation_source(alias)
            extra = {key: original.handle.extra[key] for key in _DATA_FIELDS}
            extra.update(
                privilege_profile=original.handle.extra.get("privilege_profile"),
                receipt_schema_version=original.handle.extra.get("receipt_schema_version"),
            )
            description = original.adapter._description(original.handle)
            data.append(
                {
                    "reference": alias,
                    "id": original.handle.id,
                    "arn": original.handle.arn,
                    "extra": extra,
                    "source": _source_wire(source),
                    "native_request": {key: description.get(key) for key in _DATA_NATIVE_KEYS},
                }
            )
        key = selections[0]
        if type(key) is not tuple or len(key) != 2 or type(key[0]) is not tuple:
            raise ValueError("original case-inclusive selection shape differs")
        if case_key[1:] != ("humaneval_164", "1"):
            raise ValueError("original fixed case catalog differs")
        return _encode(
            {
                "schema_version": 1,
                "logical_run_id": run_id,
                "binding": scope,
                "pending_tool_use_id": call["pending_tool_use_id"],
                "causal_prefix": call["causal_prefix"],
                "source_refs": [model_ref, prep_ref],
                "model": model_parent,
                "model_artifacts": (
                    _adopted_sft_snapshot(
                        store._models,
                        model_ref,
                        store._catalog,
                        scope["runtime_session_id"],
                        qualify=qualify,
                    )
                    if adopted
                    else _snapshot(store._models._verified_sft_model_snapshot(model_ref))
                ),
                "data": data,
                "case": {
                    "reference": case_ref,
                    "acquisition_ref": case_key[0],
                    "id": handle.id,
                    "arn": handle.arn,
                    "catalog": asdict(cases._catalog),
                    "header": handle.extra["header"],
                    "source": _source_wire(handle.extra["source"]),
                    "request": handle.extra["request"],
                    "output_prefix": handle.extra["output_prefix"],
                    "binding_ref": _ref_wire(handle.extra["binding_ref"]),
                    "files": {name: _ref_wire(ref) for name, ref in refs.items()},
                },
                "preparation": _capture(prep, prep_ref, model=False),
                "preparation_artifacts": _snapshot(store._prepared._verified_sft_snapshot(prep_ref)),
                "preparation_startup_catalog": _snapshot(store._prepared._catalog),
                "preparation_selection": {"acquisitions": list(key[0]), "case_request_ref": case_ref},
            }
        )


@contextmanager
def _qualified_reads(adapters, region, current):
    """Read-only, per-call qualified clients for the adapters, visible to the calling context only.

    The adapters are shared by the session's tools, which run on other threads concurrently with a fence:
    nothing here changes the adapter object for anyone else (scoped_native_clients).
    """
    from ..backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
    from .normal_run_checkpoint import _NativeIO
    from .scoped_native_clients import scoped_clients

    methods = {
        "describe_processing_job",
        "describe_training_job",
        "list_tags",
        "get_object",
        "head_object",
        "list_objects_v2",
        "get_bucket_versioning",
        "get_bucket_location",
    }
    with ExitStack() as stack:
        for adapter in adapters:
            # MODEL already fences SDK/body/chunk reads and binds its raw-byte
            # witness to these exact clients. A proxy would change that binding.
            if type(adapter) is not ProcessingModelWeightAcquisitionBackend:
                sagemaker, s3 = adapter._clients(region)
                stack.enter_context(
                    scoped_clients(
                        adapter, _NativeIO(sagemaker, current, methods), _NativeIO(s3, current, methods)
                    )
                )
            stack.enter_context(adapter.qualified_io(current))
        yield


def hydrate_parent_graph(raw, binding, policy, env=None, *, qualify=None):
    """Verify all temporary parents before returning any graph to its caller."""
    from ..backends.ports import DataAcquisitionBackend, ModelWeightAcquisitionBackend
    from ..backends.registry import REGISTRY
    from ..config.externalized import model_weight_acquisition_selectors, resolve_env_snapshot
    from .acquisition_state import SessionAcquisitionStore
    from .case_binding import build_case_freeze_store
    from .model_acquisition_state import build_model_acquisition_store
    from .normal_run_checkpoint import _record_recheck
    from .preparation_binding import build_preparation_store

    value = _decode(raw)
    _keys(
        value,
        {
            "schema_version",
            "logical_run_id",
            "binding",
            "pending_tool_use_id",
            "causal_prefix",
            "source_refs",
            "model",
            "model_artifacts",
            "data",
            "case",
            "preparation",
            "preparation_artifacts",
            "preparation_startup_catalog",
            "preparation_selection",
        },
    )
    scope = _binding_fields(binding)
    if (
        value["schema_version"] != 1
        or value["binding"] != scope
        or value["logical_run_id"] != uuid.UUID(binding.campaign_id).hex
    ):
        raise ValueError("original parent graph scope differs")
    _approved_read(policy)
    acquired = SessionAcquisitionStore._for_original_correlation(binding)
    models = build_model_acquisition_store(
        runtime_session_id=scope["runtime_session_id"], env=env, policy_getter=lambda: policy
    )
    cases = build_case_freeze_store(acquired, env=env)
    prepared = build_preparation_store(acquired, case_store=cases, env=env)
    adopted = _adopted_model_parent(value["model"])
    model = None if adopted else _producer(value["model"], scope, model=True)
    prep = _producer(value["preparation"], scope, model=False)
    data_adapter = REGISTRY.resolve(DataAcquisitionBackend).adapter
    model_adapter = None if adopted else REGISTRY.resolve(ModelWeightAcquisitionBackend).adapter
    selectors = None if adopted else model_weight_acquisition_selectors(env=env)
    model_bindings = None if adopted else model.extra["bindings"]
    current_resources = (prepared._catalog, cases._catalog, selectors)
    if value["preparation_startup_catalog"] != _snapshot(prepared._catalog):
        raise ValueError("current preparation startup resources differ")

    def current_check(remote):
        _approved_read(policy)
        if remote and qualify is not None:
            qualify()
        fresh_case = build_case_freeze_store(acquired, env=env)
        fresh_prep = build_preparation_store(acquired, case_store=fresh_case, env=env)
        if (
            fresh_prep._catalog,
            fresh_case._catalog,
            None if adopted else model_weight_acquisition_selectors(env=env),
        ) != current_resources:
            raise ValueError("current parent resource selection changed")
        if not adopted:
            if (
                selectors.get("image_uri"),
                selectors.get("source_commit"),
                selectors.get("admission_digest"),
            ) != (
                model_bindings["image_uri"],
                model_bindings["source_commit"],
                model_bindings["image_uri"].rsplit(":", 1)[1],
            ):
                raise ValueError("current model parent catalog differs")
            if (
                resolve_env_snapshot("session_bucket", env=env) != model.extra["bucket"]
                or resolve_env_snapshot("sagemaker_execution_role_arn", env=env)
                != model.extra["request"]["RoleArn"]
            ):
                raise ValueError("current model parent resources differ")
        for row in value["data"]:
            extra = row["extra"]
            selected = (
                data_adapter._region
                if data_adapter._region is not None
                else resolve_env_snapshot("aws_region", env=env),
                data_adapter._role
                if data_adapter._role is not None
                else resolve_env_snapshot("sagemaker_execution_role_arn", env=env),
                data_adapter._image
                if data_adapter._image is not None
                else resolve_env_snapshot("data_acquisition_image_uri", env=env),
                data_adapter._bucket
                if data_adapter._bucket is not None
                else resolve_env_snapshot("session_bucket", env=env),
                data_adapter._source
                if data_adapter._source is not None
                else resolve_env_snapshot("data_acquisition_source_commit", env=env),
            )
            if selected != tuple(
                extra[key] for key in ("region", "role_arn", "image_uri", "bucket", "source_commit")
            ):
                raise ValueError("current acquired parent resources differ")
        _approved_read(policy)

    current = _record_recheck(current_check)
    current()
    rows = value["data"]
    if type(rows) is not list or not 1 <= len(rows) <= 16:
        raise ValueError("bounded original acquired graph required")
    entries = tuple((row["reference"], data_adapter, _data_handle(row, scope)) for row in rows)
    acquired._restore_original_producers(entries, binding)
    if adopted:
        restored_ref = models._restore_adopted_parent(
            value["model"]["snapshot"], binding, policy, qualify=qualify
        )
        if restored_ref != value["model"]["request_ref"]:
            raise ValueError("restored adopted model alias differs")
        current()
    else:
        models._restore_original_producer(value["model"]["request_ref"], model_adapter, model, binding)
    selection = value["preparation_selection"]
    _keys(selection, {"acquisitions", "case_request_ref"})
    if value["source_refs"] != [value["model"]["request_ref"], value["preparation"]["request_ref"]]:
        raise ValueError("original parent source aliases differ")
    with _qualified_reads(
        (data_adapter, *((model_adapter,) if not adopted else ()), cases._adapter, prepared._adapter),
        scope["region"],
        current,
    ):
        for row in rows:
            if acquired._verified_preparation_source(row["reference"]) != _source(row["source"]):
                raise ValueError("fresh acquired parent artifact versions differ")
            handle = acquired._records[row["reference"]].handle
            description = data_adapter._description(handle)
            if {key: description.get(key) for key in _DATA_NATIVE_KEYS} != row["native_request"]:
                raise ValueError("fresh acquired parent native request differs")
        case = _case_handle(value["case"], cases, scope)
        cases._restore_original_producer(
            value["case"]["reference"], case, value["case"]["acquisition_ref"], binding
        )
        _handle, _result, case_files = cases._verified(value["case"]["reference"])
        if (
            case_files is None
            or {name: _ref_wire(ref) for name, ref in case_files.items()} != value["case"]["files"]
        ):
            raise ValueError("fresh frozen parent artifact versions differ")
        if selection["case_request_ref"] != value["case"]["reference"]:
            raise ValueError("original preparation/case alias differs")
        prepared._restore_original_producer(value["preparation"]["request_ref"], prep, binding)
        prepared._restore_original_selection(
            value["preparation"]["request_ref"],
            tuple(selection["acquisitions"]),
            selection["case_request_ref"],
            binding,
        )
        model_artifacts = (
            _adopted_sft_snapshot(
                models,
                value["model"]["request_ref"],
                prepared._catalog,
                scope["runtime_session_id"],
                qualify=current,
            )
            if adopted
            else _snapshot(models._verified_sft_model_snapshot(value["model"]["request_ref"]))
        )
        if (
            model_artifacts != value["model_artifacts"]
            or _snapshot(prepared._verified_sft_snapshot(value["preparation"]["request_ref"]))
            != value["preparation_artifacts"]
        ):
            raise ValueError("fresh model/preparation parent artifact versions differ")
        current()
    return acquired, models, cases, prepared
