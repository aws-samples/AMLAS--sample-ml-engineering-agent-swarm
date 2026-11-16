"""Fresh reconstructed acquisition/model stores retain verified Source contracts.

Uses only newly registered synthetic producer handles and current Source getters.
No stored parent bytes, checkpoint rows, traces or exception contents are read.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

import pytest
from normal_sft_cold_fixture import close, original_precommit, setup
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.backends.ports import DataAcquisitionBackend, ModelWeightAcquisitionBackend
from amlas.backends.registry import REGISTRY
from amlas.control_plane.acquisition_state import SessionAcquisitionStore
from amlas.control_plane.campaign_state import _binding_fields
from amlas.control_plane.case_binding import build_case_freeze_store
from amlas.control_plane.model_acquisition_state import SessionModelAcquisitionStore
from amlas.control_plane.normal_run_parents import (
    _DATA_FIELDS,
    _DATA_NATIVE_KEYS,
    _case_handle,
    _data_handle,
    _qualified_reads,
)
from amlas.control_plane.preparation_binding import build_preparation_store
from amlas.control_plane.producer_snapshot import _capture, _producer, _ref_wire, _source_wire


@pytest.mark.parametrize("fresh", ["model", "data"])
def test_fresh_registry_parent_adapters_compose(monkeypatch, tmp_path, fresh):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            actual_describe = state.native.describe_processing_job
            actual_tags = state.native.list_tags

            def shared_model_describe(**values):
                if any(
                    request["ProcessingJobName"] == values["ProcessingJobName"]
                    for request in state.work.native.requests
                ):
                    return state.work.native.describe_processing_job(**values)
                return actual_describe(**values)

            def shared_model_tags(**values):
                name = values["ResourceArn"].rsplit("/", 1)[-1]
                if any(request["ProcessingJobName"] == name for request in state.work.native.requests):
                    return state.work.native.list_tags(**values)
                return actual_tags(**values)

            monkeypatch.setattr(state.native, "describe_processing_job", shared_model_describe)
            monkeypatch.setattr(state.native, "list_tags", shared_model_tags)

            backend = state.backend
            binding = backend._sft_store._execution_binding
            scope = _binding_fields(binding)
            acquired = SessionAcquisitionStore._for_original_correlation(binding)
            entries = []
            for alias, record in backend._acquisition_store._records.items():
                source = backend._acquisition_store._verified_preparation_source(alias)
                description = record.adapter._description(record.handle)
                extra = {name: record.handle.extra[name] for name in _DATA_FIELDS}
                extra.update(
                    privilege_profile=record.handle.extra.get("privilege_profile"),
                    receipt_schema_version=record.handle.extra.get("receipt_schema_version"),
                )
                row = {
                    "reference": alias,
                    "id": record.handle.id,
                    "arn": record.handle.arn,
                    "extra": extra,
                    "source": _source_wire(source),
                    "native_request": {name: description.get(name) for name in _DATA_NATIVE_KEYS},
                }
                try:
                    restored = _data_handle(row, scope)
                    entries.append(
                        (
                            alias,
                            REGISTRY.resolve(DataAcquisitionBackend).adapter
                            if fresh == "data"
                            else record.adapter,
                            restored,
                        )
                    )
                except Exception:
                    raise AssertionError("fresh_acquisition_handle_reconstruction_refused") from None
            try:
                acquired._restore_original_producers(tuple(entries), binding)
                for alias, _adapter, _handle in entries:
                    assert acquired._verified_preparation_source(
                        alias
                    ) == backend._acquisition_store._verified_preparation_source(alias)
            except Exception:
                raise AssertionError("fresh_acquisition_store_reconstruction_refused") from None
            alias, record = next(iter(backend._model_acquisition_store._records.items()))
            models = SessionModelAcquisitionStore(runtime_session_id=scope["runtime_session_id"])
            try:
                restored = _producer(_capture(record.handle, alias, model=True), scope, model=True)
                models._restore_original_producer(
                    alias,
                    REGISTRY.resolve(ModelWeightAcquisitionBackend).adapter
                    if fresh == "model"
                    else record.adapter,
                    restored,
                    binding,
                )
                assert models._verified_sft_model_snapshot(
                    alias
                ) == backend._model_acquisition_store._verified_sft_model_snapshot(alias)
            except Exception:
                raise AssertionError("fresh_model_store_reconstruction_refused") from None
            cases = build_case_freeze_store(acquired)
            prepared = build_preparation_store(acquired, case_store=cases)
            case_key, case_alias = next(iter(backend._case_freeze_store._keys.items()))
            case_handle, _result, files = backend._case_freeze_store._verified(case_alias)
            case_row = {
                "reference": case_alias,
                "acquisition_ref": case_key[0],
                "id": case_handle.id,
                "arn": case_handle.arn,
                "catalog": asdict(backend._case_freeze_store._catalog),
                "header": case_handle.extra["header"],
                "source": _source_wire(case_handle.extra["source"]),
                "request": case_handle.extra["request"],
                "output_prefix": case_handle.extra["output_prefix"],
                "binding_ref": _ref_wire(case_handle.extra["binding_ref"]),
                "files": {name: _ref_wire(ref) for name, ref in files.items()},
            }
            try:
                case_restored = _case_handle(case_row, cases, scope)
                cases._restore_original_producer(case_alias, case_restored, case_key[0], binding)
                assert cases._verified_preparation_exclusions(
                    case_alias
                ) == backend._case_freeze_store._verified_preparation_exclusions(case_alias)
            except Exception:
                raise AssertionError("fresh_case_store_reconstruction_refused") from None
            prep_alias, record = next(iter(backend._preparation_store._records.items()))
            try:
                prep_restored = _producer(
                    _capture(record.handle, prep_alias, model=False), scope, model=False
                )
                prepared._restore_original_producer(prep_alias, prep_restored, binding)
                selection = next(
                    key
                    for key, alias in backend._preparation_store._submissions.items()
                    if alias == prep_alias
                )
                prepared._restore_original_selection(prep_alias, selection[0], case_alias, binding)
                assert prepared._verified_sft_snapshot(
                    prep_alias
                ) == backend._preparation_store._verified_sft_snapshot(prep_alias)
            except Exception:
                raise AssertionError("fresh_preparation_store_reconstruction_refused") from None
            callbacks = []

            def current():
                callbacks.append(True)

            retained_model_adapter = models._records[next(iter(models._records))].adapter
            data_adapter = entries[0][1]
            try:
                with _qualified_reads(
                    (data_adapter, retained_model_adapter, cases._adapter, prepared._adapter),
                    scope["region"],
                    current,
                ):
                    for data_alias, _adapter, _handle in entries:
                        acquired._verified_preparation_source(data_alias)
                    cases._verified_preparation_exclusions(case_alias)
                    prepared._verified_sft_snapshot(prep_alias)
                    models._verified_sft_model_snapshot(next(iter(models._records)))
                assert callbacks
            except Exception:
                raise AssertionError("fresh_parent_qualified_reads_refused") from None
            assert len(state.native.calls) == 1 and state.native.stops == []
            assert state.loaded == [] and state.invoked == []
        finally:
            await close(state)

    asyncio.run(scenario())
