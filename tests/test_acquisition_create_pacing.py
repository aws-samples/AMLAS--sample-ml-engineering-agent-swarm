"""Source-only initial native Create spacing; no SDK requests or wall sleeps."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from types import SimpleNamespace

import model_acquisition_fixtures as model_f
import pytest
from test_managed_dataset_acquisition import NativeProcessing, NativeS3, adapter, submit

from amlas.backends.adapters import _util
from amlas.backends.adapters import model_weight_acquisition as model_module


class Clock:
    def __init__(self):
        self.now = 10.0
        self.waits = []
        self.lock = threading.Lock()
        self.on_wait = None

    def monotonic(self):
        with self.lock:
            return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 1.25
        with self.lock:
            self.waits.append(seconds)
            self.now += seconds
        if self.on_wait:
            self.on_wait()


@pytest.fixture
def paced(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(_util, "_PROCESSING_CREATE_STATES", {}, raising=False)
    monkeypatch.setattr(_util, "_processing_create_now", clock.monotonic, raising=False)
    monkeypatch.setattr(_util, "_processing_create_sleep", clock.sleep, raising=False)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", model_f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    return clock


def test_concurrent_dataset_dataset_model_initial_calls_share_exact_spacing(paced):
    starts = []
    lock = threading.Lock()
    barrier = threading.Barrier(3)

    class Dataset(NativeProcessing):
        def create_processing_job(self, **request):
            with lock:
                starts.append(("dataset", paced.monotonic(), request["ProcessingJobName"]))
            return super().create_processing_job(**request)

    class Model(model_f.Processing):
        def create_processing_job(self, **request):
            with lock:
                starts.append(("model", paced.monotonic(), request["ProcessingJobName"]))
            return super().create_processing_job(**request)

    datasets = [Dataset(), Dataset()]
    model = Model()
    model_adapter = model_f.configured_backend(model_module, model, model_f.S3())

    def dataset_call(index):
        barrier.wait()
        return submit(adapter(sm=datasets[index], s3=NativeS3()))

    def model_call():
        barrier.wait()
        return model_adapter.submit(
            catalog_id=model_f.CATALOG, catalog_version=model_f.VERSION, runtime_session_id=model_f.SESSION
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(dataset_call, 0), pool.submit(dataset_call, 1), pool.submit(model_call)]
        handles = [future.result(timeout=10) for future in futures]
    assert len(starts) == len(handles) == 3
    times = sorted(row[1] for row in starts)
    assert all(b - a >= 1.25 for a, b in pairwise(times))
    assert sum(len(client.calls) for client in datasets) == 2 and len(model.requests) == 1
    assert len({handle.id for handle in handles}) == 3


def test_distinct_verified_account_or_region_keys_have_independent_initial_slots(paced):
    gate = _util._initial_processing_create
    with gate(model_f.ACCOUNT, model_f.REGION):
        pass
    with gate("9876" * 3, model_f.REGION):
        pass
    with gate(model_f.ACCOUNT, "us-east-1"):
        pass
    assert paced.waits == []
    with gate(model_f.ACCOUNT, model_f.REGION):
        pass
    assert paced.waits == [1.25]


def test_model_wait_rechecks_original_before_create_qualification_and_never_submits_revoked(paced):
    gate = _util._initial_processing_create
    with gate(model_f.ACCOUNT, model_f.REGION):
        pass
    allowed = [True]
    paced.on_wait = lambda: allowed.__setitem__(0, False)
    sm = model_f.Processing()
    backend = model_f.configured_backend(model_module, sm, model_f.S3())
    handle = backend.prepare_submission(
        catalog_id=model_f.CATALOG, catalog_version=model_f.VERSION, runtime_session_id=model_f.SESSION
    )

    def qualify():
        if not allowed[0]:
            raise ValueError("current original effect revoked")

    with pytest.raises(ValueError, match="revoked"):
        backend.submit_prepared(handle, before_create=qualify)
    assert sm.requests == []
    assert handle.extra["accepted"] is False and handle.extra["submission_ambiguous"] is False


def test_dataset_wait_rechecks_existing_private_scope_before_initial_create(paced):
    with _util._initial_processing_create(model_f.ACCOUNT, model_f.REGION):
        pass
    allowed = [True]
    paced.on_wait = lambda: allowed.__setitem__(0, False)
    sm = NativeProcessing()
    backend = adapter(sm=sm)

    def qualify():
        if not allowed[0]:
            raise ValueError("current dataset scope revoked")

    with backend.qualified_io(qualify), pytest.raises(ValueError, match="revoked"):
        submit(backend)
    assert sm.calls == []


def test_lost_model_reply_reserves_slot_and_only_one_initial_attempt(paced):
    sm = model_f.Processing()
    sm.ambiguous_submit = True
    backend = model_f.configured_backend(model_module, sm, model_f.S3())
    handle = backend.submit(
        catalog_id=model_f.CATALOG, catalog_version=model_f.VERSION, runtime_session_id=model_f.SESSION
    )
    assert len(sm.requests) == 1 and handle.extra["submission_ambiguous"] is True
    original_name = handle.id
    with _util._initial_processing_create(model_f.ACCOUNT, model_f.REGION):
        pass
    assert paced.waits == [1.25] and handle.id == original_name
    assert len(sm.requests) == 1


def test_initial_create_gate_refuses_inherited_process_state_before_wait(paced, monkeypatch):
    with _util._initial_processing_create(model_f.ACCOUNT, model_f.REGION):
        pass
    monkeypatch.setattr(_util.os, "getpid", lambda: _util._PROCESSING_CREATE_PID + 1)
    with pytest.raises(ValueError, match="inherited"):
        with _util._initial_processing_create(model_f.ACCOUNT, model_f.REGION):
            pass
    assert paced.waits == []


def test_dataset_sagemaker_mutating_client_has_one_total_attempt(monkeypatch):
    import boto3

    from amlas.backends.adapters.dataset_acquisition import ProcessingDatasetAcquisitionBackend

    selected = []

    class Session:
        def client(self, name, **kwargs):
            selected.append((name, kwargs))
            return SimpleNamespace(meta=SimpleNamespace(region_name=model_f.REGION))

    monkeypatch.setattr(boto3, "Session", lambda **kwargs: Session())
    ProcessingDatasetAcquisitionBackend()._clients(model_f.REGION)
    assert dict(selected)["sagemaker"]["config"].retries == {"total_max_attempts": 1}


@pytest.mark.parametrize("case", ["backwards", "nonfinite", "short_wait"])
def test_initial_create_clock_or_wait_uncertainty_refuses_before_effect(paced, monkeypatch, case):
    with _util._initial_processing_create(model_f.ACCOUNT, model_f.REGION):
        pass
    if case == "backwards":
        paced.now -= 1.0
    elif case == "nonfinite":
        paced.now = float("nan")
    else:
        monkeypatch.setattr(_util, "_processing_create_sleep", lambda _: None)
    effects = []
    with pytest.raises(ValueError):
        with _util._initial_processing_create(model_f.ACCOUNT, model_f.REGION):
            effects.append(True)
    assert effects == []


@pytest.mark.parametrize(
    "account,region",
    [("not-an-account", model_f.REGION), (model_f.ACCOUNT, "foreign"), (True, model_f.REGION)],
)
def test_initial_create_requires_validated_account_region_key(paced, account, region):
    with pytest.raises(ValueError):
        with _util._initial_processing_create(account, region):
            pytest.fail("invalid key reached effect")
    assert paced.waits == []


def test_slow_final_qualification_cannot_compress_actual_create_start_spacing(paced):
    starts = []

    class Model(model_f.Processing):
        def create_processing_job(self, **request):
            starts.append(paced.monotonic())
            return super().create_processing_job(**request)

    sm = Model()
    backend = model_f.configured_backend(model_module, sm, model_f.S3())
    first = backend.prepare_submission(
        catalog_id=model_f.CATALOG, catalog_version=model_f.VERSION, runtime_session_id=model_f.SESSION
    )

    def slow_qualifier():
        paced.now += 2.0

    backend.submit_prepared(first, before_create=slow_qualifier)
    backend.submit(
        catalog_id=model_f.CATALOG, catalog_version=model_f.VERSION, runtime_session_id=model_f.SESSION
    )
    assert len(starts) == 2
    assert starts[1] - starts[0] >= 1.25


def test_postcreate_clock_uncertainty_does_not_mask_accepted_or_lost_reply(paced, monkeypatch):
    class Accepted(model_f.Processing):
        def create_processing_job(self, **request):
            result = super().create_processing_job(**request)
            paced.now = float("nan")
            return result

    sm = Accepted()
    backend = model_f.configured_backend(model_module, sm, model_f.S3())
    handle = backend.submit(
        catalog_id=model_f.CATALOG, catalog_version=model_f.VERSION, runtime_session_id=model_f.SESSION
    )
    assert handle.extra["accepted"] is True and len(sm.requests) == 1
    with pytest.raises(ValueError):
        with _util._initial_processing_create(model_f.ACCOUNT, model_f.REGION):
            pytest.fail("unknown completion clock permits another initial send")


def test_postcreate_clock_uncertainty_preserves_original_ambiguous_handle(paced):
    class Lost(model_f.Processing):
        def create_processing_job(self, **request):
            self.requests.append(request.copy())
            paced.now = float("nan")
            raise TimeoutError("original lost reply")

    sm = Lost()
    backend = model_f.configured_backend(model_module, sm, model_f.S3())
    handle = backend.submit(
        catalog_id=model_f.CATALOG, catalog_version=model_f.VERSION, runtime_session_id=model_f.SESSION
    )
    assert handle.id == sm.requests[0]["ProcessingJobName"]
    assert handle.extra["submission_ambiguous"] is True and handle.extra["accepted"] is False
    assert len(sm.requests) == 1
