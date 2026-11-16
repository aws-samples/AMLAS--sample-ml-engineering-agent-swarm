"""Closed parent CommitPack source/profile selection; no real source or SDK calls."""

from __future__ import annotations

import dataclasses

import pytest
from test_commitpack_profile import row
from test_managed_coding_preparation import encoded, sha
from test_managed_coding_preparation import fixture as base_fixture

from amlas.control_plane.preparation_state import validate_source
from amlas.data_eng import commitpack_preparation as prep
from amlas.data_eng import commitpack_profile as profile


@pytest.fixture
def fixture(monkeypatch):
    return base_fixture.__wrapped__(monkeypatch)


def pinned_source(fixture):
    source = fixture[4]
    ref = dataclasses.replace(
        source.inspection_source.artifact_ref, byte_count=135858935, sha256=profile.ARTIFACT_SHA256
    )
    return dataclasses.replace(
        source,
        repo_id=profile.DATASET_REPO,
        revision=profile.REVISION,
        filename=profile.FILENAME,
        inspection_source=dataclasses.replace(source.inspection_source, artifact_ref=ref),
    )


def test_exact_private_commitpack_tuple_selects_pair_before_clients(fixture):
    from amlas.backends.adapters.coding_preparation import _preparation_pair

    source = pinned_source(fixture)
    validate_source(source, fixture[3], "test_session")
    assert _preparation_pair((source,)) == (prep.PROFILE_ID, prep.POLICY_VERSION)
    assert fixture[1].creates == fixture[2].calls == []
    with pytest.raises(ValueError):
        _preparation_pair((source, fixture[4]))


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "a" * 40},
        {"filename": "other.jsonl"},
        {"sha256": "a" * 64},
        {"byte_count": 135858934},
        {"format": "parquet"},
    ],
)
def test_recognized_commitpack_wrong_pin_never_falls_back_to_legacy(fixture, change):
    source = pinned_source(fixture)
    if set(change) & {"revision", "filename"}:
        source = dataclasses.replace(source, **change)
    else:
        source = dataclasses.replace(
            source,
            inspection_source=dataclasses.replace(
                source.inspection_source,
                artifact_ref=dataclasses.replace(source.inspection_source.artifact_ref, **change),
            ),
        )
    with pytest.raises(ValueError):
        validate_source(source, fixture[3], "test_session")
    assert fixture[1].creates == fixture[2].calls == []


def staged_fixture(fixture, monkeypatch):
    backend, sm, s3, catalog, original = fixture
    raw = b"".join(encoded(row(i + 1)) for i in range(100))
    # Test-owned source pin, preserving the actual guarded hashes/staging path.
    monkeypatch.setattr(profile, "ARTIFACT_SHA256", sha(raw))
    monkeypatch.setattr(prep, "SOURCE_BYTES", len(raw), raising=False)
    source = pinned_source(fixture)
    source = dataclasses.replace(
        source,
        inspection_source=dataclasses.replace(
            source.inspection_source,
            artifact_ref=dataclasses.replace(
                source.inspection_source.artifact_ref, byte_count=len(raw), sha256=sha(raw)
            ),
        ),
    )
    ref = source.inspection_source.artifact_ref
    s3.add(ref.uri.split("/", 3)[3], raw, ref.version_id)
    lineage = encoded(
        {
            "sha256": ref.sha256,
            "byte_count": ref.byte_count,
            "format": "jsonl",
            "repo_id": source.repo_id,
            "revision": source.revision,
            "filename": source.filename,
        }
    )
    commit = encoded({"sha256": ref.sha256, "lineage_sha256": sha(lineage)})
    receipt = encoded(
        {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "filename": source.filename,
            "image_uri": source.acquisition_image_uri,
            "source_commit": source.acquisition_source_commit,
            "commit_manifest_sha256": sha(commit),
        }
    )
    manifests = []
    for name, body in (("receipt", receipt), ("commit", commit), ("lineage", lineage)):
        entry = next(x for x in original.manifest_refs if x[0] == name)
        version = s3.add(entry[1].split("/", 3)[3], body)
        manifests.append((name, entry[1], version, sha(body)))
    source = dataclasses.replace(source, manifest_refs=tuple(manifests))
    policy = {
        "schema_version": 1,
        "repositories": [],
        "commits": [],
        "canonical_sha256": [],
        "input_sha256": [],
        "code_pair_sha256": [],
    }
    body = encoded(policy)
    ex = catalog.exclusions_ref
    s3.add(ex.uri.split("/", 3)[3], body, ex.version_id)
    catalog = dataclasses.replace(
        catalog, exclusions_ref=dataclasses.replace(ex, byte_count=len(body), sha256=sha(body))
    )
    return backend, sm, s3, catalog, source


def test_normal_parent_stages_fixed_pair_with_original_typed_refs(fixture, monkeypatch):
    value = staged_fixture(fixture, monkeypatch)
    handle = value[0].submit(sources=(value[4],), catalog=value[3], runtime_session_id="test_session")
    assert len(value[1].creates) == 1
    assert (
        handle.extra["header"]["profile_id"] == prep.PROFILE_ID
        and handle.extra["header"]["policy_version"] == prep.POLICY_VERSION
    )
    assert handle.extra["source_records"] == 100
    assert handle.extra["sources"] == (value[4],)
    request = value[1].creates[0]
    assert request["ProcessingInputs"][0]["InputName"] == "preparation-input"
    assert request["ProcessingOutputConfig"]["Outputs"][0]["OutputName"] == "preparation-output"
    assert request["NetworkConfig"]["EnableNetworkIsolation"] is True


def complete_outputs(value, handle):
    from test_coding_preparation_worker import PROOF

    source = value[4]
    ref = source.inspection_source.artifact_ref
    parsed = []
    for index in range(100):
        source_pin = profile.CommitPackSource(
            source.repo_id, source.revision, source.filename, ref.uri, ref.version_id, ref.sha256, index
        )
        parsed.append(prep.CommitPackParsedRow(row(index + 1), source_pin, sha(encoded(row(index + 1)))))
    result = prep.build_commitpack_preparation(
        parsed,
        exclusion_policy={
            "schema_version": 1,
            "repositories": [],
            "commits": [],
            "canonical_sha256": [],
            "input_sha256": [],
            "code_pair_sha256": [],
        },
    )
    payloads = {
        name + ".jsonl": b"".join(encoded(record) for record in result["splits"][name])
        for name in prep.SPLITS
    }
    for name in ("membership", "lineage"):
        payloads[name + ".jsonl"] = b"".join(encoded(record) for record in result[name])
    payloads["REPORT.json"] = encoded(result["report"])
    files = {}
    for name, raw in payloads.items():
        value[2].add(handle.extra["output_prefix"] + name, raw)
        files[name] = {
            "byte_count": len(raw),
            "sha256": sha(raw),
            "format": "json" if name.endswith(".json") else "jsonl",
            "record_count": 1 if name.endswith(".json") else len(raw.splitlines()),
        }
    committed = {
        "schema_version": 2,
        **handle.extra["header"],
        "privilege_profile": "sagemaker_native_ceiling_v1",
        "selection_sha256": handle.extra["selection_sha256"],
        "exclusions_sha256": handle.extra["exclusions_sha256"],
        "files": files,
        "counts": result["report"]["counts"],
        "coverage": prep.COVERAGE,
        "guest_evidence": {
            "privilege_proof": PROOF,
            "credential_observation": "env_and_home_file_presence",
            "network_observation": "fixed_public_and_metadata_tcp_denial",
        },
    }
    value[2].add(handle.extra["output_prefix"] + "COMMITTED.json", encoded(committed))
    value[1].state = "Completed"
    return result, committed


def test_completed_parent_replays_all_origins_and_returns_metadata_only_refs(fixture, monkeypatch):
    value = staged_fixture(fixture, monkeypatch)
    handle = value[0].submit(sources=(value[4],), catalog=value[3], runtime_session_id="test_session")
    complete_outputs(value, handle)
    result = value[0].artifacts(handle)
    assert result["success"] is True and result["prepared"] is True
    assert result["trained"] is False and result["training_ready"] is False
    assert result["counts"]["source"] == 100 and result["counts"]["retained_representatives"] == 100
    assert result["coverage"] == prep.COVERAGE and set(result["split_refs"]) == set(prep.SPLITS)
    assert len(value[1].creates) == 1
    for ref in [result["preparation_ref"], *result["split_refs"].values(), *result["sidecar_refs"].values()]:
        assert ref["version_id"] not in ("", "null", None)


@pytest.mark.parametrize(
    "tamper", ["source_count", "coverage", "origin", "membership", "payload", "report_bool", "foreign_source"]
)
def test_parent_refuses_complete_rehashed_commitpack_substitution(fixture, monkeypatch, tamper):
    import json

    value = staged_fixture(fixture, monkeypatch)
    handle = value[0].submit(sources=(value[4],), catalog=value[3], runtime_session_id="test_session")
    _result, committed = complete_outputs(value, handle)
    prefix = handle.extra["output_prefix"]
    s3 = value[2]
    if tamper == "source_count":
        committed["counts"]["source"] = 99
    elif tamper == "coverage":
        committed["coverage"] = {**prep.COVERAGE, "tokenization": "ready"}
    else:
        filename = (
            "lineage.jsonl"
            if tamper in ("origin", "foreign_source")
            else "membership.jsonl"
            if tamper == "membership"
            else "train.jsonl"
            if tamper == "payload"
            else "REPORT.json"
        )
        raw = s3.values[prefix + filename][0]
        if filename.endswith(".jsonl"):
            rows = [json.loads(line) for line in raw.splitlines()]
            if tamper == "origin":
                rows[0]["origin_record"]["source_fields"]["new_contents"] = "changed code"
            elif tamper == "foreign_source":
                rows[0]["source_provenance"]["artifact_version_id"] = "foreign"
            elif tamper == "membership":
                rows[0]["split"] = "test"
            else:
                rows[0]["messages"][1]["content"] = "changed code"
            body = b"".join(encoded(record) for record in rows)
        else:
            report = json.loads(raw)
            report["counts"]["source"] = True
            body = encoded(report)
        s3.add(prefix + filename, body)
        committed["files"][filename].update(byte_count=len(body), sha256=sha(body))
    s3.add(prefix + "COMMITTED.json", encoded(committed))
    result = value[0].artifacts(handle)
    assert result["success"] is False and result["prepared"] is False
    assert len(value[1].creates) == 1


def test_parent_observes_nonusable_report_without_splitrefs_or_commit(fixture, monkeypatch):
    value = staged_fixture(fixture, monkeypatch)
    handle = value[0].submit(sources=(value[4],), catalog=value[3], runtime_session_id="test_session")
    outcome = prep._nonusable("group_resource_bound", 4, 3, 1)["nonusable_outcome"]
    value[2].add(handle.extra["output_prefix"] + "REPORT.json", encoded(outcome))
    value[1].state = "Completed"
    result = value[0].artifacts(handle)
    assert result["status"] == "incomplete" and result["prepared"] is False
    assert result["nonusable_outcome"]["scan_complete"] is False
    assert "split_refs" not in result and len(value[1].creates) == 1


def test_catalog_default_legacy_cap_remains_two_mib_and_cp_limit_is_explicit(fixture):
    from amlas.control_plane.preparation_state import validate_catalog

    ref = dataclasses.replace(fixture[3].exclusions_ref, byte_count=2 * 1024 * 1024 + 1)
    selected = dataclasses.replace(fixture[3], exclusions_ref=ref)
    with pytest.raises(ValueError):
        validate_catalog(selected)
    validate_catalog(selected, exclusions_limit=prep.MAX_EXCLUSIONS_BYTES)
    for invalid in (True, prep.MAX_EXCLUSIONS_BYTES + 1, 1.5):
        with pytest.raises(ValueError):
            validate_catalog(selected, exclusions_limit=invalid)


def test_private_store_resolves_cp_cap_without_weakening_legacy(fixture, monkeypatch):
    from amlas.backends.types import JobHandle, JobState, JobStatus
    from amlas.control_plane.acquisition_state import SessionAcquisitionStore
    from amlas.control_plane.preparation_state import SessionPreparationStore

    acquired = SessionAcquisitionStore()
    acquired.runtime_session_id = "test_session"
    chosen = pinned_source(fixture)
    calls = []
    monkeypatch.setattr(acquired, "_verified_preparation_source", lambda ref: chosen)

    class Adapter:
        backend_name = "processing"

        def submit(self, **kwargs):
            calls.append(kwargs)
            return JobHandle("test", "processing", extra={"runtime_session_id": "test_session"})

        def status(self, handle):
            return JobStatus(JobState.RUNNING)

        def artifacts(self, handle):
            return {"prepared": False}

    catalog = dataclasses.replace(
        fixture[3],
        exclusions_ref=dataclasses.replace(fixture[3].exclusions_ref, byte_count=2 * 1024 * 1024 + 1),
    )
    store = SessionPreparationStore(acquired, Adapter(), catalog)
    assert store.submit(["acq_" + "a" * 32])["accepted"] is True and len(calls) == 1
    chosen = fixture[4]
    with pytest.raises(ValueError):
        store.submit(["acq_" + "b" * 32])
    assert len(calls) == 1


def test_cp_exclusions_larger_than_legacy_cap_stage_exactly(fixture, monkeypatch):
    value = staged_fixture(fixture, monkeypatch)
    policy = {
        "schema_version": 1,
        "repositories": [],
        "commits": [],
        "canonical_sha256": [f"{i:064x}" for i in range(32000)],
        "input_sha256": [],
        "code_pair_sha256": [],
    }
    raw = prep.encode_commitpack_metadata(policy)
    assert 2 * 1024 * 1024 < len(raw) < prep.MAX_EXCLUSIONS_BYTES
    ref = value[3].exclusions_ref
    value[2].add(ref.uri.split("/", 3)[3], raw, ref.version_id)
    catalog = dataclasses.replace(
        value[3], exclusions_ref=dataclasses.replace(ref, byte_count=len(raw), sha256=sha(raw))
    )
    handle = value[0].submit(sources=(value[4],), catalog=catalog, runtime_session_id="test_session")
    assert handle.extra["exclusions_sha256"] == sha(raw) and len(value[1].creates) == 1
    staged = next(
        body for key, (body, _version) in value[2].values.items() if key.endswith("/input/exclusions.json")
    )
    assert staged == raw


def test_cp_report_limit_is_explicit_and_does_not_change_legacy_metadata_limit():
    from amlas.backends.adapters import coding_preparation as adapter
    from amlas.sagemaker_entrypoints import coding_preparation as worker

    assert prep.COMMITPACK_REPORT_LIMIT == 8 * 1024 * 1024 + 262144
    assert worker.COMMITPACK_REPORT_LIMIT == prep.COMMITPACK_REPORT_LIMIT
    assert adapter.METADATA_LIMIT == worker.MAX_METADATA == 2 * 1024 * 1024
    invalid = {
        "schema_version": 1,
        "repositories": [],
        "commits": [],
        "canonical_sha256": ["z" * 64],
        "input_sha256": [],
        "code_pair_sha256": [],
    }
    with pytest.raises(ValueError):
        prep.validate_commitpack_exclusions(invalid)


def multi_source_output(fixture, monkeypatch):
    from test_coding_preparation_worker import PROOF

    value = staged_fixture(fixture, monkeypatch)
    a = value[4]
    ref = a.inspection_source.artifact_ref
    ref_b = dataclasses.replace(ref, uri=ref.uri + ".second")
    b = dataclasses.replace(a, inspection_source=dataclasses.replace(a.inspection_source, artifact_ref=ref_b))
    raw = value[2].values[ref.uri.split("/", 3)[3]][0]
    value[2].add(ref_b.uri.split("/", 3)[3], raw, ref_b.version_id)
    handle = value[0].submit(sources=(a, b), catalog=value[3], runtime_session_id="test_session")
    parsed = []
    for chosen in (a, b):
        pin = chosen.inspection_source.artifact_ref
        for index in range(100):
            prov = profile.CommitPackSource(
                chosen.repo_id, chosen.revision, chosen.filename, pin.uri, pin.version_id, pin.sha256, index
            )
            # Coherent origin reassignment: B99 becomes A100; totalstill200.
            if chosen is b and index == 99:
                prov = dataclasses.replace(prov, artifact_uri=ref.uri, row_index=100)
            parsed.append(prep.CommitPackParsedRow(row(index + 1), prov, sha(encoded(row(index + 1)))))
    policy = {
        "schema_version": 1,
        "repositories": [],
        "commits": [],
        "canonical_sha256": [],
        "input_sha256": [],
        "code_pair_sha256": [],
    }
    result = prep.build_commitpack_preparation(parsed, exclusion_policy=policy)
    files = {}
    for name in ("train", "validation", "test", "membership", "lineage"):
        records = result["splits"][name] if name in prep.SPLITS else result[name]
        body = b"".join(encoded(record) for record in records)
        value[2].add(handle.extra["output_prefix"] + name + ".jsonl", body)
        files[name + ".jsonl"] = {
            "byte_count": len(body),
            "sha256": sha(body),
            "format": "jsonl",
            "record_count": len(records),
        }
    body = encoded(result["report"])
    value[2].add(handle.extra["output_prefix"] + "REPORT.json", body)
    files["REPORT.json"] = {"byte_count": len(body), "sha256": sha(body), "format": "json", "record_count": 1}
    committed = {
        "schema_version": 2,
        **handle.extra["header"],
        "privilege_profile": "sagemaker_native_ceiling_v1",
        "selection_sha256": handle.extra["selection_sha256"],
        "exclusions_sha256": handle.extra["exclusions_sha256"],
        "files": files,
        "counts": result["report"]["counts"],
        "coverage": prep.COVERAGE,
        "guest_evidence": {
            "privilege_proof": PROOF,
            "credential_observation": "env_and_home_file_presence",
            "network_observation": "fixed_public_and_metadata_tcp_denial",
        },
    }
    value[2].add(handle.extra["output_prefix"] + "COMMITTED.json", encoded(committed))
    value[1].state = "Completed"
    return value, handle


def test_parent_rejects_coherent_101_99_attribution_against_private_100_100_counts(fixture, monkeypatch):
    value, handle = multi_source_output(fixture, monkeypatch)
    result = value[0].artifacts(handle)
    assert result["success"] is False and result["prepared"] is False
    assert sorted(handle.extra["source_row_counts"].values()) == [100, 100]


@pytest.mark.parametrize("counts", [None, {}, {"foreign": 100}, {"a" * 64: True}])
def test_missing_or_malformed_private_source_counts_refuse(fixture, monkeypatch, counts):
    value = staged_fixture(fixture, monkeypatch)
    handle = value[0].submit(sources=(value[4],), catalog=value[3], runtime_session_id="test_session")
    complete_outputs(value, handle)
    handle.extra["source_row_counts"] = counts
    assert value[0].artifacts(handle)["prepared"] is False
