"""Exact-byte pure continuation validation reuse; no native authority is supplied.

Actual row decoding, strict fresh GetItem requests and real typed read policy
remain active. The unit boundary records policy calls, not runtime role permission.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from test_normal_checkpoint_cas_contract import TABLE_ID, _item, _policy
from test_normal_completion_record_codec import DeclaredGetPort, _declared_sft_record

from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.control_plane.producer_snapshot import _approved_read


def _fixture(monkeypatch):
    context, record = _declared_sft_record()
    port = DeclaredGetPort(context, _item(context, record))
    checks = []
    validation_calls = []
    current = [_policy()]
    original = checkpoint._validate_continuation_data

    def validate(value):
        validation_calls.append(True)
        return original(value)

    def boundary():
        _approved_read(current[0])
        checks.append("fresh")

    monkeypatch.setattr(checkpoint, "_validate_continuation_data", validate)
    checkpoint._validate_record_bytes.cache_clear()
    return context, record, port, checks, validation_calls, current, boundary


def _load(context, port, boundary):
    return checkpoint._load(context, port, TABLE_ID, boundary)[1]


def test_identical_fresh_rows_validate_continuation_once(monkeypatch):
    context, record, port, checks, validations, _current, boundary = _fixture(monkeypatch)
    first = _load(context, port, boundary)
    second = _load(context, port, boundary)
    assert first == second == record and first is not second
    assert port.calls == 2 and checks == ["fresh"] * 4
    assert len(validations) == 1, "identical canonical continuation DATA was validated more than once"


def test_cached_validation_keeps_every_returned_parse_detached(monkeypatch):
    context, record, port, _checks, _validations, _current, boundary = _fixture(monkeypatch)
    first = _load(context, port, boundary)
    first["continuation"]["sdk_snapshot"]["data"]["messages"][0]["content"][0]["text"] = (
        "Changed detached DATA."
    )
    first["continuation"]["epoch"] = True
    assert _load(context, port, boundary) == record
    assert port.item == _item(context, record) and port.calls == 2


def test_new_rehashed_invalid_continuation_bytes_refuse(monkeypatch):
    context, record, port, _checks, validations, _current, boundary = _fixture(monkeypatch)
    _load(context, port, boundary)
    changed = copy.deepcopy(record)
    changed["continuation"]["epoch"] = True
    port.item = _item(context, changed)
    with pytest.raises(ValueError):
        _load(context, port, boundary)
    assert port.calls == 2 and len(validations) == 2


def test_fresh_hash_drift_refuses_before_cached_decode(monkeypatch):
    context, _record, port, checks, _validations, _current, boundary = _fixture(monkeypatch)
    _load(context, port, boundary)
    hits = checkpoint._validate_record_bytes.cache_info().hits
    port.item["record_sha256"] = {"S": "0" * 64}
    with pytest.raises(ValueError):
        _load(context, port, boundary)
    assert port.calls == 2 and checks == ["fresh"] * 4
    assert checkpoint._validate_record_bytes.cache_info().hits == hits


def test_current_denied_policy_refuses_without_second_get(monkeypatch):
    context, _record, port, _checks, _validations, current, boundary = _fixture(monkeypatch)
    _load(context, port, boundary)
    current[0] = _policy(False)
    with pytest.raises(ValueError):
        _load(context, port, boundary)
    assert port.calls == 1


def test_current_replacement_row_is_read_and_decoded(monkeypatch):
    context, record, port, checks, validations, _current, boundary = _fixture(monkeypatch)
    _load(context, port, boundary)
    changed = copy.deepcopy(record)
    changed["history"].append({"role": "user", "content": [{"text": "Declared new row DATA."}]})
    port.item = _item(context, changed)
    assert _load(context, port, boundary) == changed
    assert port.calls == 2 and checks == ["fresh"] * 4 and len(validations) == 2


def test_current_context_drift_refuses_even_identical_cached_bytes(monkeypatch):
    context, _record, port, checks, _validations, _current, boundary = _fixture(monkeypatch)
    _load(context, port, boundary)
    changed_context = dataclasses.replace(context, session_hash="9" * 64)
    with pytest.raises(ValueError):
        _load(changed_context, port, boundary)
    assert port.calls == 2 and checks == ["fresh"] * 4


def test_existing_four_entry_lru_eviction_revalidates_exact_data(monkeypatch):
    context, record, port, _checks, validations, _current, boundary = _fixture(monkeypatch)
    original_item = copy.deepcopy(port.item)
    for index in range(5):
        changed = copy.deepcopy(record)
        changed["history"] = [{"role": "user", "content": [{"text": "Declared variant " + str(index)}]}]
        port.item = original_item if index == 0 else _item(context, changed)
        _load(context, port, boundary)
    port.item = original_item
    assert _load(context, port, boundary) == record
    info = checkpoint._validate_record_bytes.cache_info()
    assert info.maxsize == 4 and info.currsize == 4 and info.misses == 6
    assert port.calls == 6 and len(validations) == 6


def test_validated_warm_detached_decode_does_not_repeat_duplicate_key_checks(monkeypatch):
    _context, record = _declared_sft_record()
    record["continuation"]["sdk_snapshot"]["data"]["state"] = {
        "declared_values": {
            "integer": 7,
            "decimal": 1.25,
            "boolean": True,
            "unicode": "café",
            "nested": [None, {"label": "Ordinary declared DATA."}],
        }
    }
    raw = checkpoint._canonical(record)
    expected = json.loads(raw)
    calls = []
    original_pairs = checkpoint._pairs

    def pairs(items):
        calls.append(True)
        return original_pairs(items)

    monkeypatch.setattr(checkpoint, "_pairs", pairs)
    checkpoint._validate_record_bytes.cache_clear()
    first = checkpoint._record_data(raw)
    assert first == expected
    cold_checks = len(calls)
    assert cold_checks > 0 and checkpoint._validate_record_bytes.cache_info().misses == 1
    first_values = first["continuation"]["sdk_snapshot"]["data"]["state"]["declared_values"]
    first_values["nested"][1]["label"] = "Changed detached DATA."
    first_values["integer"] = 99
    second = checkpoint._record_data(raw)
    second_values = second["continuation"]["sdk_snapshot"]["data"]["state"]["declared_values"]
    assert second == expected and first is not second and first_values is not second_values
    assert type(second_values["integer"]) is int and type(second_values["decimal"]) is float
    assert type(second_values["boolean"]) is bool and second_values["unicode"] == "café"
    info = checkpoint._validate_record_bytes.cache_info()
    assert info.maxsize == 4 and info.misses == 1 and info.hits == 1
    assert checkpoint._validate_record_bytes(raw) is None
    assert len(calls) == cold_checks, "validated detached decode repeated duplicate-key checks"


@pytest.mark.parametrize("warm_valid", [False, True], ids=["cold", "after_valid"])
def test_duplicate_key_bytes_refuse_cold_and_after_valid_warm_data(warm_valid):
    checkpoint._validate_record_bytes.cache_clear()
    if warm_valid:
        _context, record = _declared_sft_record()
        checkpoint._record_data(checkpoint._canonical(record))
    raw = '{"declared":1,"declared":2}'
    with pytest.raises(ValueError):
        checkpoint._record_data(raw)
    info = checkpoint._validate_record_bytes.cache_info()
    assert info.maxsize == 4 and info.currsize == int(warm_valid)
