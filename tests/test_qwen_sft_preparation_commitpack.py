"""Closed CommitPack prepared-input replay; synthetic source rows, no native grants."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_qwen_sft_inputs import fixture as canonical_fixture

from amlas.data_eng import commitpack_profile as profile

PROFILE = profile.PROFILE_ID
POLICY = "coding-preparation.commitpack.v3"
ORIGIN_KEYS = (
    "source_provenance",
    "source_fields",
    "commit",
    "license",
    "repositories",
    "canonical_sha256",
    "input_sha256",
    "code_pair_sha256",
    "transformations",
)
COVERAGE = {
    "repository_affiliations": "all_declared",
    "commit_edges": "exact_source_commit",
    "conversation_duplicates": "exact",
    "full_input_code_duplicates": "exact",
    "near_duplicates": "not_implemented",
    "complete_upstream_fork_graph": "not_verified",
    "heldout_exclusions": "exact_list_component_exclusions",
    "tokenization": "not_applied",
    "exclusion_scope": "exact_listed_identity_and_digest_matches_only",
}
EXCLUSION_KEYS = ("repositories", "commits", "canonical_sha256", "input_sha256", "code_pair_sha256")


def encoded(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def row(index):
    return {
        "commit": f"{index + 1:040x}",
        "old_file": "code.py",
        "new_file": "code.py",
        "old_contents": f"value = {index}\n",
        "new_contents": f"value = {index + 1}\n",
        "subject": f"Change value {index}",
        "message": f"Recorded change {index}\n",
        "lang": "Python",
        "license": "mit",
        "repos": f"fixture/repo-{index}",
    }


def bodies(exclusions=None, bridge=False):
    policy = {"schema_version": 1, **{key: [] for key in EXCLUSION_KEYS}}
    if exclusions:
        policy.update(exclusions)
    originals = [row(index) for index in range(80)]
    if bridge:
        originals[1]["repos"] = "fixture/repo-0,fixture/repo-1"
        originals[2]["repos"] = "fixture/repo-1"
    duplicate = row(0)
    duplicate.update(commit="f" * 40, license="apache-2.0", message="Second recorded origin\n")
    originals.append(duplicate)
    rejected = row(81)
    rejected["license"] = "unknown"
    originals.append(rejected)
    envelopes = []
    raw_hashes = []
    for index, raw in enumerate(originals):
        source = profile.CommitPackSource(
            profile.DATASET_REPO,
            profile.REVISION,
            profile.FILENAME,
            "s3://fixture-bucket/raw/python.jsonl",
            "fixture-version",
            profile.ARTIFACT_SHA256,
            index,
        )
        envelopes.append(profile.normalize_commitpack_row(raw, source=source))
        raw_hashes.append(hashlib.sha256(encoded(raw) + b"\n").hexdigest())
    grouped = profile.group_commitpack_records(envelopes)
    assert grouped["status"] == "grouped"
    origin_map = {}
    component_records = {}
    for split, records in grouped["splits"].items():
        for record in records:
            component_records.setdefault(record["group_id"], []).append(record)
            representative = digest(record["origin_records"][0]["source_provenance"])
            for origin in record["origin_records"]:
                origin_map[origin["source_provenance"]["row_index"]] = (record, split, representative, origin)
    excluded = {}
    for group, records in component_records.items():
        for key in EXCLUSION_KEYS:
            field = {"repositories": "repositories", "commits": "commit"}.get(key, key)
            match = False
            for record in records:
                for origin in record["origin_records"]:
                    values = origin[field] if key == "repositories" else [origin[field]]
                    match = match or bool(set(values) & set(policy[key]))
            if match:
                excluded[group] = key
                break
    lineage = []
    reason_counts = {key: 0 for key in EXCLUSION_KEYS}
    normalized = included = excluded_count = duplicate_count = 0
    for index, envelope in enumerate(envelopes):
        entry = {
            "source_provenance": envelope["source_provenance"],
            "raw_line_sha256": raw_hashes[index],
            "decision": "normalization_rejected",
            "origin_record": None,
            "canonical_sha256": None,
            "group_id": None,
            "split": None,
            "representative_origin_sha256": None,
            "rejection_reason": envelope["issues"][0]["reason"] if envelope["status"] == "rejected" else None,
        }
        if envelope["status"] == "normalized":
            normalized += 1
            record, split, representative, origin = origin_map[index]
            is_excluded = record["group_id"] in excluded
            is_representative = digest(origin["source_provenance"]) == representative
            entry.update(
                decision="component_excluded"
                if is_excluded
                else "retained_representative"
                if is_representative
                else "duplicate_origin",
                origin_record={key: origin[key] for key in ORIGIN_KEYS},
                canonical_sha256=record["canonical_sha256"],
                group_id=record["group_id"],
                split=split,
                representative_origin_sha256=representative,
            )
            if is_excluded:
                excluded_count += 1
                reason_counts[excluded[record["group_id"]]] += 1
            else:
                included += 1
                duplicate_count += not is_representative
        lineage.append(entry)
    splits = {}
    membership = []
    for split in ("train", "validation", "test"):
        records = sorted(
            (record for record in grouped["splits"][split] if record["group_id"] not in excluded),
            key=lambda value: value["canonical_sha256"],
        )
        splits[split] = [{key: record[key] for key in ("messages", "tools")} for record in records]
        membership.extend(
            {"canonical_sha256": record["canonical_sha256"], "group_id": record["group_id"], "split": split}
            for record in records
        )
    counts = {
        "source": len(envelopes),
        "normalization_rejected": 1,
        "normalized": normalized,
        "excluded_origins": excluded_count,
        "included_origins": included,
        "retained_representatives": len(membership),
        "duplicate_origins": duplicate_count,
        "split_counts": {split: len(records) for split, records in splits.items()},
    }
    report = {
        "schema_version": 3,
        "profile_id": PROFILE,
        "policy_version": POLICY,
        "status": "complete",
        "counts": counts,
        "normalization_rejection_reasons": {"license_unsupported": 1},
        "exclusion_policy": policy,
        "exclusion_reason_counts": reason_counts,
        "coverage": copy.deepcopy(COVERAGE),
        "supervision_kind": "human_commit_change_supervision",
    }
    return {
        **{split + ".jsonl": records for split, records in splits.items()},
        "membership.jsonl": membership,
        "lineage.jsonl": lineage,
        "REPORT.json": report,
    }


def fixture(monkeypatch, tmp_path, exclusions=None, bridge=False):
    m, base, prepared, binding = canonical_fixture(monkeypatch, tmp_path)
    content = bodies(exclusions, bridge=bridge)
    committed = json.loads((prepared / "COMMITTED.json").read_bytes())
    committed.update(
        profile_id=PROFILE,
        policy_version=POLICY,
        counts=copy.deepcopy(content["REPORT.json"]["counts"]),
        coverage=copy.deepcopy(COVERAGE),
        exclusions_sha256=hashlib.sha256(
            encoded(content["REPORT.json"]["exclusion_policy"]) + b"\n"
        ).hexdigest(),
    )
    binding["preparation_header"].update(profile_id=PROFILE, policy_version=POLICY)
    commit_files(content, committed, prepared, binding)
    return m, base, prepared, binding, content, committed


def commit_files(content, committed, prepared, binding):
    for name, value in content.items():
        raw = (
            b"".join(encoded(item) + b"\n" for item in value)
            if name.endswith(".jsonl")
            else encoded(value) + b"\n"
        )
        (prepared / name).write_bytes(raw)
        ref = {
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "format": "jsonl" if name.endswith(".jsonl") else "json",
            "record_count": len(value) if name.endswith(".jsonl") else 1,
        }
        committed["files"][name] = ref
        if name.removesuffix(".jsonl") in binding["split_refs"]:
            binding["split_refs"][name.removesuffix(".jsonl")] = ref
    raw = encoded(committed) + b"\n"
    (prepared / "COMMITTED.json").write_bytes(raw)
    binding["preparation_commit"] = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}


def consumer(m):
    value = getattr(m, "_validate_preparation_commitpack", None)
    assert callable(value), "missing-symbol: _validate_preparation_commitpack"
    return value


def test_complete_commitpack_staging_replays_all_origins(monkeypatch, tmp_path):
    m, base, prepared, binding, content, committed = fixture(monkeypatch, tmp_path)
    consumer(m)
    result = m.validate_staged_inputs(binding, base, prepared)
    assert result["preparation_profile_id"] == PROFILE and result["preparation_policy_version"] == POLICY
    assert result["commitpack"]["counts"] == committed["counts"]
    assert result["commitpack"]["supervision_kind"] == "human_commit_change_supervision"
    assert result["trusted"] is False and result["test_used_for_training"] is False
    assert len(content["lineage.jsonl"]) == 82 and committed["counts"]["duplicate_origins"] == 1


def test_whole_component_exclusion_keeps_all_duplicate_origin_provenance(monkeypatch, tmp_path):
    m, base, prepared, binding, content, _committed = fixture(monkeypatch, tmp_path, {"commits": ["f" * 40]})
    consumer(m)
    result = m.validate_staged_inputs(binding, base, prepared)
    assert result["commitpack"]["counts"]["excluded_origins"] == 2
    assert [
        entry["origin_record"]["license"]
        for entry in content["lineage.jsonl"]
        if entry["decision"] == "component_excluded"
    ] == ["MIT", "Apache-2.0"]


@pytest.mark.parametrize(
    "case",
    [
        "profile",
        "policy",
        "report_bool",
        "count",
        "origin_license",
        "origin_hash",
        "origin_extra",
        "omit_duplicate",
        "ordinal",
        "raw_line",
        "group",
        "split",
        "membership",
        "rejection",
        "exclusion_digest",
        "unsorted_list",
        "exclusion_extra",
        "coverage",
        "fake_resolved",
        "duplicate_json",
    ],
)
def test_rehashed_commitpack_substitution_refuses(monkeypatch, tmp_path, case):
    m, base, prepared, binding, content, committed = fixture(monkeypatch, tmp_path)
    consumer(m)
    line = content["lineage.jsonl"][0]
    if case == "profile":
        committed["profile_id"] = binding["preparation_header"]["profile_id"] = "other.profile"
    elif case == "policy":
        committed["policy_version"] = binding["preparation_header"]["policy_version"] = (
            "coding-preparation.v1"
        )
    elif case == "report_bool":
        content["REPORT.json"]["counts"]["normalization_rejected"] = True
    elif case == "count":
        committed["counts"]["duplicate_origins"] += 1
    elif case.startswith("origin_"):
        if case == "origin_license":
            line["origin_record"]["license"] = "Apache-2.0"
        elif case == "origin_hash":
            line["origin_record"]["canonical_sha256"] = "0" * 64
        else:
            line["origin_record"]["approved"] = True
    elif case == "omit_duplicate":
        content["lineage.jsonl"].pop(80)
    elif case == "ordinal":
        line["source_provenance"]["row_index"] = 1
    elif case == "raw_line":
        line["raw_line_sha256"] = True
    elif case == "group":
        line["group_id"] = "0" * 64
    elif case == "split":
        line["split"] = {"train": "test", "test": "validation", "validation": "train"}[line["split"]]
    elif case == "membership":
        content["membership.jsonl"][0]["canonical_sha256"] = "0" * 64
    elif case == "rejection":
        content["lineage.jsonl"][-1]["rejection_reason"] = "executed_successfully"
    elif case == "exclusion_digest":
        committed["exclusions_sha256"] = "0" * 64
    elif case == "unsorted_list":
        content["REPORT.json"]["exclusion_policy"]["commits"] = ["f" * 40, "a" * 40]
        committed["exclusions_sha256"] = hashlib.sha256(
            encoded(content["REPORT.json"]["exclusion_policy"]) + b"\n"
        ).hexdigest()
    elif case == "exclusion_extra":
        content["REPORT.json"]["exclusion_policy"]["approved"] = True
        committed["exclusions_sha256"] = hashlib.sha256(
            encoded(content["REPORT.json"]["exclusion_policy"]) + b"\n"
        ).hexdigest()
    elif case == "coverage":
        content["REPORT.json"]["coverage"]["complete_upstream_fork_graph"] = "verified"
    elif case == "fake_resolved":
        line["origin_record"]["source_fields"]["resolved"] = 1
    commit_files(content, committed, prepared, binding)
    if case == "duplicate_json":
        raw = b'{"schema_version":3,' + (prepared / "REPORT.json").read_bytes()[1:]
        (prepared / "REPORT.json").write_bytes(raw)
        committed["files"]["REPORT.json"].update(byte_count=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        raw = encoded(committed) + b"\n"
        (prepared / "COMMITTED.json").write_bytes(raw)
        binding["preparation_commit"].update(byte_count=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        m.validate_staged_inputs(binding, base, prepared)


def test_full_training_floors_are_unchanged():
    from amlas.training.qwen_sft import recipe

    settings = recipe("full_v1")
    assert settings["minimum_unique_corpus_examples"] == 1000
    assert settings["minimum_unique_supervised_tokens"] == 1000000
    assert settings["optimizer_steps"] == 1024 and settings["minimum_supervised_tokens_processed"] == 1000000
    assert settings["max_sequence_tokens"] == 4096


def test_repository_bridge_exclusion_replays_before_dedup(monkeypatch, tmp_path):
    m, base, prepared, binding, content, committed = fixture(
        monkeypatch, tmp_path, {"commits": [f"{1:040x}"]}, bridge=True
    )
    consumer(m)
    result = m.validate_staged_inputs(binding, base, prepared)
    assert result["commitpack"]["counts"]["excluded_origins"] == 4
    excluded = [entry for entry in content["lineage.jsonl"] if entry["decision"] == "component_excluded"]
    assert len({entry["group_id"] for entry in excluded}) == 1
    # Rehashed partial component laundering must fail, including bridge/duplicate origins.
    excluded[1]["decision"] = "retained_representative"
    commit_files(content, committed, prepared, binding)
    with pytest.raises(ValueError):
        m.validate_staged_inputs(binding, base, prepared)


def test_commitpack_report_limit_is_explicit_and_legacy_cap_unchanged(monkeypatch, tmp_path):
    m, base, prepared, binding, _content, _committed = fixture(
        monkeypatch, tmp_path, {"commits": ["b" + f"{i:039x}" for i in range(7000)]}
    )
    assert getattr(m, "PREPARED_COMMITPACK_REPORT_LIMIT", None) == 8 * 1024**2 + 262144
    assert (prepared / "REPORT.json").stat().st_size > 262144
    m.validate_staged_inputs(binding, base, prepared)
    with pytest.raises(ValueError):
        m._decode_plain_metadata((prepared / "REPORT.json").read_bytes())


def test_rehashed_commitpack_report_over_fixed_limit_refuses(monkeypatch, tmp_path):
    m, base, prepared, binding, _content, committed = fixture(monkeypatch, tmp_path)
    consumer(m)
    limit = getattr(m, "PREPARED_COMMITPACK_REPORT_LIMIT", None)
    assert type(limit) is int, "missing-symbol: PREPARED_COMMITPACK_REPORT_LIMIT"
    raw = b" " * limit + (prepared / "REPORT.json").read_bytes()
    (prepared / "REPORT.json").write_bytes(raw)
    committed["files"]["REPORT.json"].update(byte_count=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    raw = encoded(committed) + b"\n"
    (prepared / "COMMITTED.json").write_bytes(raw)
    binding["preparation_commit"].update(byte_count=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        m.validate_staged_inputs(binding, base, prepared)


def test_commitpack_deep_split_refuses_before_unbounded_decoder(monkeypatch, tmp_path):
    m, base, prepared, binding, _content, committed = fixture(monkeypatch, tmp_path)
    raw = b'{"messages":[],"tools":' + b"[" * 2000 + b"]" * 2000 + b"}\n"
    split = "validation"
    count = committed["counts"]["split_counts"][split]
    data = raw * count
    (prepared / (split + ".jsonl")).write_bytes(data)
    ref = committed["files"][split + ".jsonl"]
    ref.update(byte_count=len(data), sha256=hashlib.sha256(data).hexdigest())
    binding["split_refs"][split] = ref
    raw_commit = encoded(committed) + b"\n"
    (prepared / "COMMITTED.json").write_bytes(raw_commit)
    binding["preparation_commit"].update(
        byte_count=len(raw_commit), sha256=hashlib.sha256(raw_commit).hexdigest()
    )
    with pytest.raises(ValueError):
        m.validate_staged_inputs(binding, base, prepared)


def test_commitpack_excess_line_refuses_before_excess_decode(monkeypatch, tmp_path):
    m, base, prepared, binding, _content, committed = fixture(monkeypatch, tmp_path)
    raw = b"{}\n" * 82 + b'{"messages":[],"tools":' + b"[" * 2000 + b"]" * 2000 + b"}\n"
    (prepared / "lineage.jsonl").write_bytes(raw)
    committed["files"]["lineage.jsonl"].update(
        byte_count=len(raw), sha256=hashlib.sha256(raw).hexdigest(), record_count=83
    )
    raw_commit = encoded(committed) + b"\n"
    (prepared / "COMMITTED.json").write_bytes(raw_commit)
    binding["preparation_commit"].update(
        byte_count=len(raw_commit), sha256=hashlib.sha256(raw_commit).hexdigest()
    )
    reads = []
    original = m._read_exact_relative

    def observed(directory, name, ref, limit, read_bytes, **kwargs):
        reads.append(name)
        return original(directory, name, ref, limit, read_bytes, **kwargs)

    monkeypatch.setattr(m, "_read_exact_relative", observed)
    with pytest.raises(ValueError):
        m.validate_staged_inputs(binding, base, prepared)
    assert "lineage.jsonl" not in reads, "CP descriptor/source cardinality must refuse before payload read"


def test_commitpack_report_read_uses_fixed_cap(monkeypatch, tmp_path):
    m, base, prepared, binding, _content, _committed = fixture(monkeypatch, tmp_path)
    original = m._read_exact_relative
    observed = []

    def read(directory, name, ref, limit, read_bytes, **kwargs):
        if name == "REPORT.json":
            observed.append(limit)
        return original(directory, name, ref, limit, read_bytes, **kwargs)

    monkeypatch.setattr(m, "_read_exact_relative", read)
    m.validate_staged_inputs(binding, base, prepared)
    assert observed == [8 * 1024**2 + 262144], "REPORT cap must apply at the same-FD I/O boundary"


def test_commitpack_fifo_refuses_without_waiting_for_writer(monkeypatch, tmp_path):
    import os
    import subprocess
    import sys

    _m, _base, prepared, _binding, _content, committed = fixture(monkeypatch, tmp_path)
    source = prepared / "REPORT.json"
    source.rename(prepared / "REPORT-original.json")
    os.mkfifo(source)
    # Isolated watchdog prevents a faulty blocking open from wedging the test suite.
    script = """import inspect,json,os,sys
from amlas.training.qwen_sft import _read_exact_relative
root=os.open(sys.argv[1],os.O_RDONLY|os.O_DIRECTORY)
try:
    try:
        _read_exact_relative(root,"REPORT.json",json.loads(sys.argv[2]),8*1024**2+262144,True,**({"nonblocking":True} if "nonblocking" in inspect.signature(_read_exact_relative).parameters else {}))
    except ValueError:
        print("refused")
    else:
        raise AssertionError("FIFO accepted")
finally: os.close(root)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(prepared), json.dumps(committed["files"]["REPORT.json"])],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0 and result.stdout.strip() == "refused", result.stderr


def test_commitpack_fixed_committed_pin_refuses_before_replay(monkeypatch, tmp_path):
    m, base, prepared, binding, _content, _committed = fixture(monkeypatch, tmp_path)
    raw = (prepared / "COMMITTED.json").read_bytes()
    changed = raw.replace(b'"selection_sha256":"' + b"c" * 64, b'"selection_sha256":"' + b"b" * 64)
    assert changed != raw and len(changed) == len(raw)
    (prepared / "COMMITTED.json").write_bytes(changed)
    monkeypatch.setattr(
        m,
        "_validate_preparation_commitpack",
        lambda *args: pytest.fail("replay reached after committed pin mismatch"),
    )
    with pytest.raises(ValueError):
        m.validate_staged_inputs(binding, base, prepared)


def test_commitpack_line_iteration_refuses_excess_before_decoding(monkeypatch):
    import amlas.data_eng.inspection_schema as schema
    import amlas.training.qwen_sft as m

    calls = []
    old = schema.decode_object

    def observed(raw, **kwargs):
        calls.append(raw)
        return old(raw, **kwargs)

    monkeypatch.setattr(schema, "decode_object", observed)
    with pytest.raises(ValueError):
        m._commitpack_records(b"{}\n" * 82 + b"[" * 2000 + b"]" * 2000 + b"\n", 82)
    assert len(calls) == 82, "Exactly the admitted records decode; excess bytes never reach a decoder"


def test_commitpack_record_parser_never_splitlines_allocates(monkeypatch):
    import amlas.training.qwen_sft as m

    monkeypatch.setattr(
        m, "_prepared_records", lambda *args: pytest.fail("CP must use the direct strict object decoder")
    )
    assert m._commitpack_records(b"\r" * 20000 + b"{}\n", 1) == [{}]
