"""Pure canonical validation reuse keeps fresh parses and native authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from amlas.control_plane import normal_run_checkpoint as source


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_identical_record_bytes_validate_once_and_return_independent_data(monkeypatch):
    source._validate_record_bytes.cache_clear()
    actual = source._canonical
    calls = []

    def validated(value):
        calls.append(True)
        return actual(value)

    monkeypatch.setattr(source, "_canonical", validated)
    raw = canonical({"history": [{"content": ["fresh declared DATA"]}], "schema_version": 1})
    first = source._record_data(raw)
    first["history"][0]["content"].append("caller mutation")
    second = source._record_data(raw)
    assert second == {"history": [{"content": ["fresh declared DATA"]}], "schema_version": 1}
    assert second is not first and calls == [True]
    changed = canonical({"history": [], "schema_version": 1})
    assert source._record_data(changed) == {"history": [], "schema_version": 1}
    assert calls == [True, True]
    source._validate_record_bytes.cache_clear()


def test_reused_validation_preserves_all_canonical_and_data_refusals():
    nested = "declared"
    for _ in range(26):
        nested = [nested]
    for raw in (
        '{"same":1,"same":2}',
        '{"nonfinite":NaN}',
        '{ "schema_version": 1 }',
        '{"approval_policy":"declared refused"}',
        "[]",
        "{",
        canonical({"nested": nested}),
    ):
        source._validate_record_bytes.cache_clear()
        for _ in range(2):
            with pytest.raises(ValueError):
                source._record_data(raw)
        assert source._validate_record_bytes.cache_info().currsize == 0


def test_record_validation_reuse_is_bounded_and_evicts():
    source._validate_record_bytes.cache_clear()
    for index in range(5):
        assert source._record_data(canonical({"counter": index})) == {"counter": index}
    info = source._validate_record_bytes.cache_info()
    assert info.maxsize == 4 and info.currsize == 4 and info.misses == 5
    source._record_data(canonical({"counter": 0}))
    assert source._validate_record_bytes.cache_info().misses == 6
    source._validate_record_bytes.cache_clear()


def test_record_validation_size_boundary_and_oversize_refusal():
    source._validate_record_bytes.cache_clear()
    raw = canonical({"text": ""})
    value = {"text": "a" * (source._MAX_BYTES - len(raw.encode()))}
    exact = canonical(value)
    assert len(exact.encode()) == source._MAX_BYTES
    assert source._record_data(exact) == value
    with pytest.raises(ValueError):
        source._record_data(canonical({"text": value["text"] + "a"}))
    assert source._validate_record_bytes.cache_info().currsize == 1
    source._validate_record_bytes.cache_clear()


def test_warm_validation_keeps_native_reads_hash_and_context_checks():
    @dataclass(frozen=True)
    class Catalog:
        jobs_table: str = "declared-owned-table"

    context = SimpleNamespace(catalog=Catalog(), run_id="fresh-run", session_hash="declared-session")
    record = {
        "schema_version": 1,
        "logical_run_id": context.run_id,
        "session_sha256": context.session_hash,
        "scope": {"jobs_table": context.catalog.jobs_table},
        "table_id": "declared-table-incarnation",
        "private_store_correlation": "00000000-0000-4000-8000-a00000000001",
        "admitted_prompt": "fresh declared test",
        "history": [],
        "stage": "enrolled",
    }
    raw = canonical(record)
    item = {
        "job_name": {"S": "PROMPT_RUN#" + context.run_id},
        "record": {"S": raw},
        "record_sha256": {"S": hashlib.sha256(raw.encode()).hexdigest()},
    }
    reads, checks = [], []

    class Native:
        def get_item(self, **kwargs):
            assert kwargs == {
                "TableName": context.catalog.jobs_table,
                "Key": {"job_name": {"S": "PROMPT_RUN#" + context.run_id}},
                "ConsistentRead": True,
            }
            reads.append(True)
            return {"Item": item}

    denied_at = [None]

    def qualify():
        checks.append(True)
        if denied_at[0] == len(checks):
            raise ValueError("fresh qualification refused")

    source._validate_record_bytes.cache_clear()
    native = Native()
    for _ in range(2):
        assert source._load(context, native, record["table_id"], qualify)[1] == record
    assert len(reads) == 2 and len(checks) == 4
    denied_at[0] = len(checks) + 2
    with pytest.raises(ValueError):
        source._load(context, native, record["table_id"], qualify)
    assert len(reads) == 3
    denied_at[0] = None
    item["record_sha256"]["S"] = "0" * 64
    with pytest.raises(ValueError):
        source._load(context, native, record["table_id"], qualify)
    item["record_sha256"]["S"] = hashlib.sha256(raw.encode()).hexdigest()
    context.session_hash = "changed-current-session"
    with pytest.raises(ValueError):
        source._load(context, native, record["table_id"], qualify)
    assert len(reads) == 5 and source._validate_record_bytes.cache_info().hits >= 1
    source._validate_record_bytes.cache_clear()
