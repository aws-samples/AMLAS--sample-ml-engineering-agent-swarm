"""Preparation consumes original private case exclusions, never public artifact JSON."""

from __future__ import annotations

import dataclasses

import pytest
from test_coding_case_tools import ACQ
from test_coding_case_tools import state as case_state_fixture
from test_managed_coding_case_envelope import ACCOUNT, BUCKET, HEADER, IMAGE, SESSION

from amlas.backends.types import DatasetArtifactRef, JobHandle
from amlas.control_plane.preparation_state import PreparationCatalog, SessionPreparationStore

TRAIN = "acq_" + "b" * 32


class Backend:
    backend_name = "processing"

    def __init__(self):
        self.catalogs = []

    def submit(self, *, sources, catalog, runtime_session_id):
        self.catalogs.append(catalog)
        job = "owned-data-prep-" + str(len(self.catalogs))
        return JobHandle(
            job,
            self.backend_name,
            "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/" + job,
            {"runtime_session_id": runtime_session_id, "catalog": catalog},
        )

    def status(self, handle):
        raise AssertionError("not observed")

    def artifacts(self, handle):
        raise AssertionError("not observed")


@pytest.fixture
def case_state(monkeypatch):
    return case_state_fixture.__wrapped__(monkeypatch)


@pytest.fixture
def pair(case_state, monkeypatch):
    cases, case_backend, _reads = case_state
    frozen = cases.submit(ACQ)["case_ref"]
    original = cases._acquired._verified_preparation_source(ACQ)
    inspection = dataclasses.replace(
        original.inspection_source,
        artifact_ref=DatasetArtifactRef(
            "s3://" + BUCKET + "/sessions/" + SESSION + "/acquisition/owned-training/data.jsonl",
            "train-v1",
            "e" * 64,
            100,
            "jsonl",
        ),
    )
    training = dataclasses.replace(
        original,
        inspection_source=inspection,
        manifest_refs=tuple(
            (
                n,
                "s3://" + BUCKET + "/sessions/" + SESSION + "/acquisition/owned-training/" + n + ".json",
                n + "-v1",
                "f" * 64,
            )
            for n in ("receipt", "commit", "lineage")
        ),
        repo_id="public/canonical",
        filename="records.jsonl",
    )
    monkeypatch.setattr(
        cases._acquired, "_verified_preparation_source", lambda ref: training if ref == TRAIN else original
    )
    selected = PreparationCatalog(
        ACCOUNT,
        "us-west-2",
        BUCKET,
        "arn:aws:iam::" + ACCOUNT + ":role/owned",
        IMAGE,
        "b" * 40,
        "a" * 64,
        HEADER["deployment_id"],
        "owned",
        DatasetArtifactRef("s3://" + BUCKET + "/exclusions/startup.json", "old-v1", "d" * 64, 2, "json"),
    )
    backend = Backend()
    store = SessionPreparationStore(cases._acquired, backend, selected, case_store=cases)
    return store, cases, case_backend, backend, frozen


def test_original_frozen_exclusion_replaces_only_per_job_catalog(pair):
    store, cases, _case_backend, backend, ref = pair
    expected = cases._verified_preparation_exclusions(ref)
    original = store._catalog
    result = store.submit([TRAIN], case_request_ref=ref)
    assert result["accepted"] is True
    assert backend.catalogs[0].exclusions_ref == expected.exclusions_ref
    assert store._catalog is original and original.exclusions_ref != expected.exclusions_ref
    assert store._records[result["request_ref"]].handle.extra["catalog"] is backend.catalogs[0]
    assert "exclusions_ref" not in str(result) and "ORACLES" not in str(result)


def test_same_case_reuses_once_and_changed_immutable_case_identity_does_not(pair, monkeypatch):
    store, cases, _case_backend, backend, ref = pair
    first = store.submit([TRAIN], case_request_ref=ref)
    assert store.submit([TRAIN], case_request_ref=ref)["request_ref"] == first["request_ref"]
    expected = cases._verified_preparation_exclusions(ref)
    changed = dataclasses.replace(
        expected, committed_ref=dataclasses.replace(expected.committed_ref, version_id="new-commit-v2")
    )
    monkeypatch.setattr(cases, "_verified_preparation_exclusions", lambda _: changed)
    second = store.submit([TRAIN], case_request_ref=ref)
    assert second["request_ref"] != first["request_ref"]
    assert len(backend.catalogs) == 2


@pytest.mark.parametrize("bad", [None, {}, "case_" + "f" * 32])
def test_public_or_unknown_case_ref_never_dispatches(pair, bad):
    store, _cases, _case_backend, backend, _ref = pair
    with pytest.raises((ValueError, TypeError)):
        store.submit([TRAIN], case_request_ref=bad if bad is not None else {"frozen": True})
    assert backend.catalogs == []


def test_foreign_case_session_refuses_before_training_source(pair, monkeypatch):
    store, cases, _case_backend, backend, ref = pair
    expected = cases._verified_preparation_exclusions(ref)
    monkeypatch.setattr(
        cases,
        "_verified_preparation_exclusions",
        lambda _: dataclasses.replace(expected, runtime_session_id="foreign"),
    )
    with pytest.raises(ValueError):
        store.submit([TRAIN], case_request_ref=ref)
    assert backend.catalogs == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("producer_job_arn", "arn:aws:sagemaker:us-west-2:" + "9" * 12 + ":processing-job/foreign"),
        ("producer_header", (("job_name", "public"),)),
        (
            "exclusions_ref",
            DatasetArtifactRef("s3://foreign-bucket/EXCLUSIONS.json", "v1", "a" * 64, 20, "json"),
        ),
        (
            "committed_ref",
            DatasetArtifactRef(
                "s3://" + BUCKET + "/sessions/" + SESSION + "/case-freeze/wrong/output/COMMITTED.json",
                "v1",
                "a" * 64,
                20,
                "json",
            ),
        ),
    ],
)
def test_private_case_producer_and_role_mutation_refuses_before_dispatch(pair, monkeypatch, field, value):
    store, cases, _case_backend, backend, ref = pair
    expected = cases._verified_preparation_exclusions(ref)
    monkeypatch.setattr(
        cases, "_verified_preparation_exclusions", lambda _: dataclasses.replace(expected, **{field: value})
    )
    with pytest.raises(ValueError):
        store.submit([TRAIN], case_request_ref=ref)
    assert backend.catalogs == []


def test_missing_resource_catalog_refuses_before_case_observation(pair, monkeypatch):
    store, cases, _case_backend, backend, ref = pair
    store._catalog = dataclasses.replace(store._catalog, image_uri="")
    monkeypatch.setattr(
        cases,
        "_verified_preparation_exclusions",
        lambda _: pytest.fail("case source observed before resource validation"),
    )
    with pytest.raises(ValueError):
        store.submit([TRAIN], case_request_ref=ref)
    assert backend.catalogs == []


def test_frozen_case_needs_no_operator_startup_exclusion_artifact(pair):
    store, cases, _case_backend, backend, ref = pair
    expected = cases._verified_preparation_exclusions(ref)
    store._catalog = dataclasses.replace(
        store._catalog, exclusions_ref=DatasetArtifactRef("", "", "", 0, "json")
    )
    result = store.submit([TRAIN], case_request_ref=ref)
    assert result["accepted"] is True
    assert backend.catalogs[0].exclusions_ref == expected.exclusions_ref
    assert store._catalog.exclusions_ref.byte_count == 0
